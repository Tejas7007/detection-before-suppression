"""
EMNLP Final Lap — Last Experiments Before Writing
===================================================

PHASE 1: Toy model parameter sweep
  - Try multiple configs to find one that produces a genuine
    below-chance dip (not just init noise). If none works,
    the MLP ablation sign flip is still the key result.
  - Configs vary sigma1, rho, learning rate.

PHASE 2: Probe generalization Level 2 — held-out combinations
  - All names and all templates in training.
  - Specific (name, template) pairs held out for testing.
  - An intermediate held-out test.

PHASE 3: Probe generalization — names only held out
  - Names split 50/50, ALL templates in both train and test.
  - Decomposes the generalization gap: names vs templates.

PHASE 4: SAE re-run with proper sparsity
  - Layer 4, step 2000 only.
  - L1 values: 0.1, 0.3, 1.0 (current max was 0.01).
  - Target L0 of 10-50 instead of current 850.
  - Re-run feature ablation with the sparse SAE.

PHASE 5: Cross-task PCA transfer
  - Take PCA directions from IOI difference vectors.
  - Apply to greater-than activations at the year position.
  - If ablating IOI PCA directions from GT activations also
    helps GT accuracy, the duplication signal generalizes.

Output: results/emnlp_final_lap.json
Runtime: ~1-1.5 hours on A100.
"""

import os
import gc
import json
import time
import sys
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer

try:
    from circuitscaling.datasets import (
        IOIDataset, ALL_TEMPLATES, CANDIDATE_NAMES,
    )
except ImportError:
    sys.path.insert(0, "/workspace/ioi-sign-flip/src")
    from circuitscaling.datasets import (
        IOIDataset, ALL_TEMPLATES, CANDIDATE_NAMES,
    )

DEVICE = "cuda"
RETRAINED_REPO = "anonymous-research-sub/pythia-160m-retrained-seed42"
BASE_MODEL = "EleutherAI/pythia-160m-deduped"
SEED = 42
N_BOOTSTRAP = 10_000
RESULTS_PATH = "results/emnlp_final_lap.json"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def bootstrap_ci(values, n_resamples=N_BOOTSTRAP, alpha=0.05, seed=SEED):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, arr.shape[0], size=(n_resamples, arr.shape[0]))
    means = arr[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))

def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


def load_retrained(step):
    hf = AutoModelForCausalLM.from_pretrained(
        RETRAINED_REPO, subfolder=f"step_{step}", torch_dtype=torch.float32,
    )
    model = HookedTransformer.from_pretrained(
        BASE_MODEL, hf_model=hf, device=DEVICE,
        center_writing_weights=True, center_unembed=True, fold_ln=True,
    )
    del hf; torch.cuda.empty_cache()
    return model

def load_pythia_original(step):
    hf = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=f"step{step}", torch_dtype=torch.float32,
    )
    model = HookedTransformer.from_pretrained(
        BASE_MODEL, hf_model=hf, device=DEVICE,
        center_writing_weights=True, center_unembed=True, fold_ln=True,
    )
    del hf; torch.cuda.empty_cache()
    return model

def find_s2_position(token_row, s_token_id):
    seen = 0
    for j in range(1, token_row.shape[0]):
        if int(token_row[j].item()) == int(s_token_id):
            seen += 1
            if seen == 2:
                return j
    return -1

def get_single_token_names(tokenizer):
    ids, names = [], []
    for name in CANDIDATE_NAMES:
        toks = tokenizer.encode(" " + name, add_special_tokens=False)
        if len(toks) == 1:
            ids.append(int(toks[0]))
            names.append(name)
    return ids, names


# ====================================================================
# PHASE 1: TOY MODEL PARAMETER SWEEP
# ====================================================================

def run_toy_config(sigma1, rho, lr, n_steps=3000, d=10, n_train=5000,
                   n_test=2000, eval_every=10, seed=42):
    """Run one toy model config (linear + MLP). Returns trajectories."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    def make_data(n, s=0):
        rng = np.random.default_rng(s)
        x = rng.standard_normal((n, d)).astype(np.float32)
        x[:, 0] *= sigma1
        y = (x[:, 1] > 0).astype(np.float32)
        x[:, 0] -= rho * (2 * y - 1)
        return torch.tensor(x, device=DEVICE), torch.tensor(y, device=DEVICE)

    x_train, y_train = make_data(n_train, seed)
    x_test, y_test = make_data(n_test, seed + 1)

    def evaluate(model, x, y, ablate=False):
        with torch.no_grad():
            xi = x.clone()
            if ablate:
                xi[:, 0] = 0.0
            logits = model(xi).squeeze(-1)
            return float(((logits > 0).float() == y).float().mean().item())

    results = {}
    for name, model_fn in [
        ("linear", lambda: nn.Linear(d, 1).to(DEVICE)),
        ("mlp", lambda: nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1)
        ).to(DEVICE)),
    ]:
        model = model_fn()
        for p in model.parameters():
            nn.init.normal_(p, std=0.01)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        traj = []

        for step in range(n_steps + 1):
            if step % eval_every == 0:
                acc = evaluate(model, x_test, y_test)
                acc_abl = evaluate(model, x_test, y_test, ablate=True)
                traj.append({
                    "step": step, "acc": acc,
                    "acc_ablated": acc_abl, "delta": acc_abl - acc,
                })
            if step < n_steps:
                logits = model(x_train).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, y_train)
                opt.zero_grad()
                loss.backward()
                opt.step()

        min_acc = min(traj, key=lambda r: r["acc"])
        # Find min acc DURING TRAINING (exclude step 0)
        train_traj = [r for r in traj if r["step"] > 0]
        min_acc_train = min(train_traj, key=lambda r: r["acc"]) if train_traj else min_acc

        results[name] = {
            "trajectory": traj,
            "min_acc_overall": min_acc["acc"],
            "min_acc_step": min_acc["step"],
            "min_acc_during_training": min_acc_train["acc"],
            "min_acc_during_training_step": min_acc_train["step"],
            "dip_during_training": min_acc_train["acc"] < 0.48,
            "max_positive_delta": max(r["delta"] for r in traj),
            "min_delta": min(r["delta"] for r in traj),
        }
        del model, opt

    return results


def phase1_toy_sweep():
    log("=" * 60)
    log("PHASE 1: Toy model parameter sweep")
    log("=" * 60)

    configs = [
        {"sigma1": 10, "rho": 3, "lr": 0.01, "label": "original"},
        {"sigma1": 50, "rho": 10, "lr": 0.001, "label": "strong_separation"},
        {"sigma1": 20, "rho": 8, "lr": 0.003, "label": "moderate"},
        {"sigma1": 100, "rho": 15, "lr": 0.0003, "label": "extreme_separation"},
        {"sigma1": 30, "rho": 5, "lr": 0.005, "label": "balanced"},
    ]

    out = {"configs": {}}
    for cfg in configs:
        label = cfg.pop("label")
        log(f"  config '{label}': sigma1={cfg['sigma1']}, rho={cfg['rho']}, lr={cfg['lr']}")
        r = run_toy_config(**cfg)
        cfg["label"] = label
        for model_name in ["linear", "mlp"]:
            m = r[model_name]
            dip = "YES" if m["dip_during_training"] else "no"
            log(f"    {model_name}: min_acc_train={m['min_acc_during_training']:.3f} "
                f"@ step {m['min_acc_during_training_step']}  "
                f"dip={dip}  max_Δ={m['max_positive_delta']:+.3f}  "
                f"min_Δ={m['min_delta']:+.3f}")
        out["configs"][label] = {"params": cfg, "results": r}

    return out


# ====================================================================
# PHASE 2: PROBE GENERALIZATION — HELD-OUT COMBINATIONS
# ====================================================================

def collect_s2_activations(model, names_subset, templates_subset, layer, seed=SEED):
    """Collect S2 activations for IOI and control prompts."""
    single_ids, _ = get_single_token_names(model.tokenizer)
    rng = np.random.default_rng(seed)
    ioi_acts, ctrl_acts = [], []

    for tmpl in templates_subset:
        ds = IOIDataset(
            model=model, n_prompts=20, templates=[tmpl],
            names=names_subset, symmetric=True, seed=seed,
        )
        ioi_tokens = model.to_tokens(ds.prompts).to(DEVICE)
        n = ioi_tokens.shape[0]

        s2_positions = [find_s2_position(ioi_tokens[i].cpu(), ds.s_token_ids[i]) for i in range(n)]
        ctrl_tokens = ioi_tokens.clone()
        for i in range(n):
            pool = [t for t in single_ids if t != int(ds.io_token_ids[i]) and t != int(ds.s_token_ids[i])]
            if pool and s2_positions[i] > 0:
                ctrl_tokens[i, s2_positions[i]] = int(rng.choice(pool))

        hook_name = f"blocks.{layer}.hook_resid_post"
        for tokens, acts_list in [(ioi_tokens, ioi_acts), (ctrl_tokens, ctrl_acts)]:
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            for i in range(n):
                if s2_positions[i] > 0:
                    acts_list.append(cache[hook_name][i, s2_positions[i]].cpu().float().numpy())
            del cache; torch.cuda.empty_cache()

    return np.array(ioi_acts), np.array(ctrl_acts)


def phase2_probe_combinations():
    log("=" * 60)
    log("PHASE 2: Probe generalization — held-out combinations")
    log("=" * 60)

    PROBE_STEPS = [0, 2000, 143000]
    LAYER = 5
    HOLDOUT_FRAC = 0.2

    out = {"by_step": {}}

    for step in PROBE_STEPS:
        log(f"  step {step}:")
        model = load_pythia_original(step)
        _, all_names = get_single_token_names(model.tokenizer)
        all_templates = ALL_TEMPLATES[:30]

        # Generate all (name, template) combinations.
        rng = np.random.default_rng(SEED)
        combos = [(n, t) for n in all_names for t in all_templates]
        rng.shuffle(combos)
        split = int((1 - HOLDOUT_FRAC) * len(combos))
        train_combos = combos[:split]
        test_combos = combos[split:]

        # Group by template for dataset generation.
        def group_by_template(combo_list):
            groups = {}
            for name, tmpl in combo_list:
                groups.setdefault(tmpl, []).append(name)
            return groups

        train_groups = group_by_template(train_combos)
        test_groups = group_by_template(test_combos)

        # Collect activations.
        hook_name = f"blocks.{LAYER}.hook_resid_post"
        single_ids, _ = get_single_token_names(model.tokenizer)
        rng2 = np.random.default_rng(SEED + 1)

        def collect_from_groups(groups):
            ioi_acts, ctrl_acts = [], []
            for tmpl, names_list in groups.items():
                if len(names_list) < 2:
                    continue
                ds = IOIDataset(
                    model=model, n_prompts=min(20, len(names_list)),
                    templates=[tmpl], names=names_list,
                    symmetric=True, seed=SEED,
                )
                ioi_tokens = model.to_tokens(ds.prompts).to(DEVICE)
                n = ioi_tokens.shape[0]
                s2_pos = [find_s2_position(ioi_tokens[i].cpu(), ds.s_token_ids[i]) for i in range(n)]

                ctrl_tokens = ioi_tokens.clone()
                for i in range(n):
                    pool = [t for t in single_ids if t != int(ds.io_token_ids[i]) and t != int(ds.s_token_ids[i])]
                    if pool and s2_pos[i] > 0:
                        ctrl_tokens[i, s2_pos[i]] = int(rng2.choice(pool))

                for tokens, acts_list in [(ioi_tokens, ioi_acts), (ctrl_tokens, ctrl_acts)]:
                    with torch.no_grad():
                        _, cache = model.run_with_cache(tokens, names_filter=hook_name)
                    for i in range(n):
                        if s2_pos[i] > 0:
                            acts_list.append(cache[hook_name][i, s2_pos[i]].cpu().float().numpy())
                    del cache; torch.cuda.empty_cache()
            return np.array(ioi_acts) if ioi_acts else np.zeros((0, 768)), \
                   np.array(ctrl_acts) if ctrl_acts else np.zeros((0, 768))

        ioi_train, ctrl_train = collect_from_groups(train_groups)
        ioi_test, ctrl_test = collect_from_groups(test_groups)

        if ioi_train.shape[0] < 10 or ioi_test.shape[0] < 10:
            log(f"    insufficient data: train={ioi_train.shape[0]}, test={ioi_test.shape[0]}")
            continue

        X_train = np.concatenate([ioi_train, ctrl_train])
        y_train = np.concatenate([np.ones(len(ioi_train)), np.zeros(len(ctrl_train))])
        X_test = np.concatenate([ioi_test, ctrl_test])
        y_test = np.concatenate([np.ones(len(ioi_test)), np.zeros(len(ctrl_test))])

        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
        clf.fit(X_train, y_train)
        in_acc = clf.score(X_train, y_train)
        held_out_acc = clf.score(X_test, y_test)

        out["by_step"][f"step_{step}"] = {
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "in_distribution_acc": float(in_acc),
            "held_out_acc": float(held_out_acc),
            "gap": float(in_acc - held_out_acc),
        }
        log(f"    in_dist={in_acc*100:.1f}%  held_out={held_out_acc*100:.1f}%  "
            f"gap={in_acc - held_out_acc:+.1%}  "
            f"n_train={X_train.shape[0]} n_test={X_test.shape[0]}")

        del model; torch.cuda.empty_cache(); gc.collect()

    return out


# ====================================================================
# PHASE 3: PROBE — NAMES ONLY HELD OUT
# ====================================================================

def phase3_probe_names_only():
    log("=" * 60)
    log("PHASE 3: Probe generalization — names only held out")
    log("=" * 60)

    PROBE_STEPS = [0, 2000, 143000]
    LAYER = 5

    out = {"by_step": {}}

    for step in PROBE_STEPS:
        log(f"  step {step}:")
        model = load_pythia_original(step)
        _, all_names = get_single_token_names(model.tokenizer)
        all_templates = ALL_TEMPLATES[:30]

        # Split names 50/50, keep all templates.
        rng = np.random.default_rng(SEED)
        shuffled = list(all_names)
        rng.shuffle(shuffled)
        half = len(shuffled) // 2
        names_A = shuffled[:half]
        names_B = shuffled[half:]

        ioi_A, ctrl_A = collect_s2_activations(model, names_A, all_templates, LAYER, SEED)
        ioi_B, ctrl_B = collect_s2_activations(model, names_B, all_templates, LAYER, SEED + 1)

        if ioi_A.shape[0] < 10 or ioi_B.shape[0] < 10:
            log(f"    insufficient data")
            continue

        # Train on A, test on B.
        X_A = np.concatenate([ioi_A, ctrl_A])
        y_A = np.concatenate([np.ones(len(ioi_A)), np.zeros(len(ctrl_A))])
        X_B = np.concatenate([ioi_B, ctrl_B])
        y_B = np.concatenate([np.ones(len(ioi_B)), np.zeros(len(ctrl_B))])

        clf_A = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
        clf_A.fit(X_A, y_A)
        in_A = clf_A.score(X_A, y_A)
        out_B = clf_A.score(X_B, y_B)

        clf_B = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
        clf_B.fit(X_B, y_B)
        in_B = clf_B.score(X_B, y_B)
        out_A = clf_B.score(X_A, y_A)

        out["by_step"][f"step_{step}"] = {
            "trained_on_A": {"in_dist": float(in_A), "held_out": float(out_B), "gap": float(in_A - out_B)},
            "trained_on_B": {"in_dist": float(in_B), "held_out": float(out_A), "gap": float(in_B - out_A)},
            "mean_held_out": float((out_A + out_B) / 2),
        }
        log(f"    A->B: in={in_A*100:.1f}% out={out_B*100:.1f}%  "
            f"B->A: in={in_B*100:.1f}% out={out_A*100:.1f}%  "
            f"mean_held_out={(out_A+out_B)/2*100:.1f}%")

        del model; torch.cuda.empty_cache(); gc.collect()

    return out


# ====================================================================
# PHASE 4: SAE WITH PROPER SPARSITY
# ====================================================================

class SAE(nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.d_in, self.d_hidden = d_in, d_hidden
        self.b_pre = nn.Parameter(torch.zeros(d_in))
        scale = 1.0 / (d_in ** 0.5)
        self.W_enc = nn.Parameter(torch.randn(d_in, d_hidden) * scale)
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = nn.Parameter(self.W_enc.data.T.clone().contiguous())
        with torch.no_grad():
            self.W_dec.data /= (self.W_dec.data.norm(dim=-1, keepdim=True) + 1e-8)

    def encode(self, x):
        return F.relu((x - self.b_pre) @ self.W_enc + self.b_enc)

    def decode(self, h):
        return h @ self.W_dec + self.b_pre

    def forward(self, x):
        h = self.encode(x)
        return self.decode(h), h

    @torch.no_grad()
    def renorm_decoder(self):
        self.W_dec.data /= (self.W_dec.data.norm(dim=-1, keepdim=True) + 1e-8)


def phase4_sae_proper():
    log("=" * 60)
    log("PHASE 4: SAE with proper sparsity (L4, step 2000)")
    log("=" * 60)

    acts_path = "results/tier_s_activations/step2000_L4_general.pt"
    ioi_path = "results/tier_s_activations/step2000_L4_s2_ioi.pt"
    ctrl_path = "results/tier_s_activations/step2000_L4_s2_ctrl.pt"

    if not os.path.exists(acts_path):
        log("  Activation files not found. Generating...")
        os.makedirs("results/tier_s_activations", exist_ok=True)
        model = load_retrained(2000)
        hook_name = "blocks.4.hook_resid_post"
        single_ids, _ = get_single_token_names(model.tokenizer)
        rng = np.random.default_rng(SEED + 1)

        # General activations from natural text.
        general_sents = [
            "The cat sat on the mat and looked out the window at the birds.",
            "Scientists discovered a new species of fish in the deep ocean.",
            "The stock market rallied after the central bank announced new policies.",
            "She walked through the garden admiring the flowers in full bloom.",
            "The committee voted to approve the budget for the upcoming fiscal year.",
            "He picked up the guitar and played a melody he remembered from childhood.",
            "The train arrived at the station exactly on time despite the weather.",
            "Researchers published their findings in the most prestigious journal.",
            "The children played in the park while their parents watched from benches.",
            "After the meeting they went to a restaurant for dinner and conversation.",
        ] * 20  # 200 sentences
        general_acts = []
        for sent in general_sents:
            toks = model.to_tokens(sent).to(DEVICE)
            with torch.no_grad():
                _, cache = model.run_with_cache(toks, names_filter=hook_name)
            # Take activations at all positions (except BOS).
            acts_t = cache[hook_name][0, 1:, :].cpu().float()
            general_acts.append(acts_t)
            del cache; torch.cuda.empty_cache()
        general_tensor = torch.cat(general_acts, dim=0)
        torch.save(general_tensor, acts_path)
        log(f"    saved {general_tensor.shape[0]} general activations")

        # IOI and ctrl activations at S2.
        ioi_s2, ctrl_s2 = [], []
        for tmpl in ALL_TEMPLATES[:15]:
            ds = IOIDataset(model=model, n_prompts=30, templates=[tmpl],
                            symmetric=True, seed=SEED)
            ioi_tokens = model.to_tokens(ds.prompts).to(DEVICE)
            n = ioi_tokens.shape[0]
            s2_pos = [find_s2_position(ioi_tokens[i].cpu(), ds.s_token_ids[i]) for i in range(n)]
            ctrl_tokens = ioi_tokens.clone()
            for i in range(n):
                pool = [t for t in single_ids if t != int(ds.io_token_ids[i]) and t != int(ds.s_token_ids[i])]
                if pool and s2_pos[i] > 0:
                    ctrl_tokens[i, s2_pos[i]] = int(rng.choice(pool))

            for tokens, s2_list in [(ioi_tokens, ioi_s2), (ctrl_tokens, ctrl_s2)]:
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, names_filter=hook_name)
                for i in range(n):
                    if s2_pos[i] > 0:
                        s2_list.append(cache[hook_name][i, s2_pos[i]].cpu().float())
                del cache; torch.cuda.empty_cache()

        ioi_tensor = torch.stack(ioi_s2)
        ctrl_tensor = torch.stack(ctrl_s2)
        torch.save(ioi_tensor, ioi_path)
        torch.save(ctrl_tensor, ctrl_path)
        log(f"    saved {ioi_tensor.shape[0]} IOI and {ctrl_tensor.shape[0]} ctrl S2 activations")
        del model; torch.cuda.empty_cache(); gc.collect()

    acts = torch.load(acts_path)
    x_ioi = torch.load(ioi_path)
    x_ctrl = torch.load(ctrl_path)
    log(f"  Loaded {acts.shape[0]} general, {x_ioi.shape[0]} IOI, {x_ctrl.shape[0]} ctrl activations")

    L1_VALUES = [0.1, 0.3, 1.0]
    D_HIDDEN = 768 * 4  # 4x expansion
    N_STEPS = 3000
    BATCH = 1024

    out = {"sweeps": [], "best": None}

    for l1 in L1_VALUES:
        torch.manual_seed(SEED)
        sae = SAE(768, D_HIDDEN).to(DEVICE)
        opt = torch.optim.Adam(sae.parameters(), lr=3e-4)
        acts_dev = acts.to(DEVICE)
        n = acts_dev.shape[0]

        for step in range(N_STEPS):
            idx = torch.randint(0, n, (BATCH,), device=DEVICE)
            x = acts_dev[idx]
            x_hat, h = sae(x)
            recon = ((x - x_hat) ** 2).sum(dim=-1).mean()
            sparsity = h.abs().sum(dim=-1).mean()
            l1_now = l1 * min(1.0, (step + 1) / 200)
            loss = recon + l1_now * sparsity
            opt.zero_grad()
            loss.backward()
            opt.step()
            sae.renorm_decoder()

        # Evaluate.
        with torch.no_grad():
            _, h_all = sae(acts_dev)
            recon_mse = ((acts_dev - sae.decode(sae.encode(acts_dev))) ** 2).sum(-1).mean()
            l0 = (h_all > 0).float().sum(-1).mean()
            var_x = acts_dev.var(0).mean()
            evar = 1 - recon_mse / (var_x * 768)

        log(f"  L1={l1:.1f}: recon_MSE={recon_mse:.2f}  evar={evar:.3f}  L0={l0:.1f}")

        # Feature identification + ablation.
        with torch.no_grad():
            h_ioi = sae.encode(x_ioi.to(DEVICE)).cpu().numpy()
            h_ctrl = sae.encode(x_ctrl.to(DEVICE)).cpu().numpy()
        X_probe = np.concatenate([h_ioi, h_ctrl])
        y_probe = np.concatenate([np.ones(len(h_ioi)), np.zeros(len(h_ctrl))])
        clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
        clf.fit(X_probe, y_probe)
        probe_acc = clf.score(X_probe, y_probe)
        top_features = np.argsort(-np.abs(clf.coef_[0]))[:30].tolist()

        sweep_result = {
            "l1": l1,
            "recon_mse": float(recon_mse),
            "explained_variance": float(evar),
            "mean_l0": float(l0),
            "probe_acc": float(probe_acc),
            "top_features": top_features[:10],
        }

        # Quick ablation test: ablate top-10 features, measure effect on
        # a small IOI eval set using the saved model.
        # (Full model-based ablation requires loading the transformer,
        #  so we approximate by measuring the SAE reconstruction with/without
        #  top features and reporting the change in feature-space distance.)
        with torch.no_grad():
            h_ioi_t = sae.encode(x_ioi.to(DEVICE))
            h_ablated = h_ioi_t.clone()
            h_ablated[:, top_features[:10]] = 0.0
            recon_full = sae.decode(h_ioi_t)
            recon_ablated = sae.decode(h_ablated)
            diff_norm = (recon_full - recon_ablated).norm(dim=-1).mean()

        sweep_result["ablation_recon_diff_norm"] = float(diff_norm)
        log(f"    probe_acc={probe_acc*100:.1f}%  top10_ablation_norm_diff={diff_norm:.4f}")

        out["sweeps"].append(sweep_result)
        del sae, opt; torch.cuda.empty_cache()

    # Best by L0 closest to 30.
    best = min(out["sweeps"], key=lambda s: abs(s["mean_l0"] - 30))
    out["best_l1"] = best["l1"]
    out["best_l0"] = best["mean_l0"]
    log(f"  Best config: L1={best['l1']}  L0={best['mean_l0']:.1f}")

    return out


# ====================================================================
# PHASE 5: CROSS-TASK PCA TRANSFER
# ====================================================================

def phase5_cross_task_pca():
    log("=" * 60)
    log("PHASE 5: Cross-task PCA transfer (IOI → GT)")
    log("=" * 60)

    LAYER = 4
    GT_EVENTS = ["war", "battle", "dispute", "conflict"]
    GT_VERBS = ["lasted", "ran", "continued"]
    GT_N = 200

    # Step 1: Compute IOI PCA directions at step 2000.
    log("  Computing IOI PCA directions at step 2000...")
    model = load_retrained(2000)
    single_ids, _ = get_single_token_names(model.tokenizer)
    rng = np.random.default_rng(SEED + 1)
    hook_name = f"blocks.{LAYER}.hook_resid_post"

    ioi_acts, ctrl_acts = [], []
    for tmpl in ALL_TEMPLATES[:10]:
        ds = IOIDataset(model=model, n_prompts=30, templates=[tmpl],
                        symmetric=True, seed=SEED)
        ioi_tokens = model.to_tokens(ds.prompts).to(DEVICE)
        n = ioi_tokens.shape[0]
        s2_pos = [find_s2_position(ioi_tokens[i].cpu(), ds.s_token_ids[i]) for i in range(n)]
        ctrl_tokens = ioi_tokens.clone()
        for i in range(n):
            pool = [t for t in single_ids if t != int(ds.io_token_ids[i]) and t != int(ds.s_token_ids[i])]
            if pool and s2_pos[i] > 0:
                ctrl_tokens[i, s2_pos[i]] = int(rng.choice(pool))

        for tokens, acts in [(ioi_tokens, ioi_acts), (ctrl_tokens, ctrl_acts)]:
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            for i in range(n):
                if s2_pos[i] > 0:
                    acts.append(cache[hook_name][i, s2_pos[i]].cpu().float().numpy())
            del cache; torch.cuda.empty_cache()

    ioi_arr = np.array(ioi_acts)
    ctrl_arr = np.array(ctrl_acts)
    diff_arr = ioi_arr - ctrl_arr
    pca = PCA(n_components=20)
    pca.fit(diff_arr)
    components = torch.tensor(pca.components_, dtype=torch.float32, device=DEVICE)
    diff_mean = torch.tensor(pca.mean_, dtype=torch.float32, device=DEVICE)
    log(f"  IOI PCA: {diff_arr.shape[0]} diffs, top-3 explained var: {pca.explained_variance_ratio_[:3]}")

    del model; torch.cuda.empty_cache(); gc.collect()

    # Step 2: Load retrained Pythia at GT dip floor (step 1100).
    log("  Loading retrained Pythia at step 1100 (GT dip floor)...")
    model = load_retrained(1100)

    # Generate GT prompts and find year positions.
    rng_gt = np.random.default_rng(SEED)
    gt_prompts = []
    for _ in range(GT_N):
        event = rng_gt.choice(GT_EVENTS)
        verb = rng_gt.choice(GT_VERBS)
        y = int(rng_gt.integers(3, 97))
        gt_prompts.append({
            "prompt": f"The {event} {verb} from the year {1700+y} to the year 17",
            "start_yy": y,
        })

    digit_tokens = {}
    for d in range(100):
        ids = model.tokenizer.encode(f"{d:02d}", add_special_tokens=False)
        if len(ids) == 1:
            digit_tokens[d] = ids[0]
    valid_digits = sorted(digit_tokens.keys())
    token_ids = torch.tensor([digit_tokens[d] for d in valid_digits], device=DEVICE)

    # Find year positions.
    year_prefix_ids = model.tokenizer.encode(" 17", add_special_tokens=False)
    year_prefix_token = year_prefix_ids[0] if len(year_prefix_ids) == 1 else None

    # Evaluate GT with and without PCA ablation at year position.
    base_diffs, ablated_diffs = [], []
    for p in gt_prompts:
        toks = model.to_tokens(p["prompt"]).to(DEVICE)
        seq = toks[0].cpu().tolist()
        year_pos = -1
        if year_prefix_token is not None:
            for j in range(1, len(seq)):
                if seq[j] == year_prefix_token:
                    year_pos = j
                    break
        if year_pos < 0:
            continue

        # Baseline.
        with torch.no_grad():
            base_logits = model(toks)[0, -1, :]
        base_probs = F.softmax(base_logits.float(), dim=-1)[token_ids].cpu().numpy()
        gmask = np.array([d > p["start_yy"] for d in valid_digits])
        lmask = np.array([d <= p["start_yy"] for d in valid_digits])
        base_diffs.append(float(base_probs[gmask].sum() - base_probs[lmask].sum()))

        # PCA ablation at year position.
        def pca_hook(value, hook):
            act = value[0, year_pos, :]
            centered = act - diff_mean
            proj = centered @ components[:10].T
            recon = proj @ components[:10]
            value[0, year_pos, :] = act - recon
            return value

        with torch.no_grad():
            abl_logits = model.run_with_hooks(toks, fwd_hooks=[(hook_name, pca_hook)])[0, -1, :]
        abl_probs = F.softmax(abl_logits.float(), dim=-1)[token_ids].cpu().numpy()
        ablated_diffs.append(float(abl_probs[gmask].sum() - abl_probs[lmask].sum()))

    base_arr = np.array(base_diffs)
    abl_arr = np.array(ablated_diffs)
    deltas = abl_arr - base_arr
    delta_lo, delta_hi = bootstrap_ci(deltas)

    out = {
        "n_prompts": int(len(deltas)),
        "base_gt_diff": float(base_arr.mean()),
        "ablated_gt_diff": float(abl_arr.mean()),
        "delta_mean": float(deltas.mean()),
        "delta_ci95": [delta_lo, delta_hi],
        "ioi_pca_explained_var": pca.explained_variance_ratio_[:10].tolist(),
    }
    log(f"  GT at step 1100: base P(>)-P(<=)={base_arr.mean():+.4f}  "
        f"after IOI-PCA ablation={abl_arr.mean():+.4f}  "
        f"Δ={deltas.mean():+.4f} [{delta_lo:+.3f}, {delta_hi:+.3f}]")

    del model; torch.cuda.empty_cache(); gc.collect()
    return out


# ====================================================================
# PHASE 6: CLASS-IMBALANCED TOY MODEL
# ====================================================================

def phase6_imbalanced_toy():
    """
    Key insight for a GENUINE below-chance dip:

    With BALANCED classes, using a feature with the correct weight sign
    always gives >= 50% accuracy.

    With IMBALANCED classes, the model CAN go below 50% even with correct
    weight signs. Here is why:

    - P(y=1) = 0.3 (minority), P(y=0) = 0.7 (majority)
    - Dimension 1 is correlated with y=1: E[x1|y=1] = +mu, E[x1|y=0] = 0
    - The model learns w1 > 0 (CORRECT sign)
    - But x1 has high variance, so the model aggressively predicts y=1
    - Since y=0 is 70% of data, over-predicting y=1 gives < 50% accuracy
    - Later, dimension 2 catches up and the model calibrates

    The dip is genuine: the model learns a correct feature association but
    over-relies on it due to high variance, causing accuracy to drop below
    chance when the feature is associated with the MINORITY class.

    Ablation sign flip:
    - During dip: removing x1 stops over-prediction of minority class -> helps
    - At maturity: removing x1 removes useful signal -> hurts (sign flip)
    """
    log("=" * 60)
    log("PHASE 6: Class-imbalanced toy model")
    log("=" * 60)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    N_TRAIN = 8000
    N_TEST = 2000
    D = 10
    SIGMA_1 = 15.0
    SIGMA_2 = 1.0
    P_MINORITY = 0.3
    MU_SHIFT = 4.0
    N_STEPS = 5000
    EVAL_EVERY = 20
    LR = 0.003

    def make_data(n, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, D)).astype(np.float32)
        # Label: y=1 (minority, 30%) if x2 > quantile corresponding to 70th percentile
        # Simpler: just assign y randomly with P(y=1)=0.3, then make x2 predict y
        y = (rng.random(n) < P_MINORITY).astype(np.float32)
        # Dimension 2 predicts y: shift x2 based on label
        x[:, 1] = rng.standard_normal(n).astype(np.float32) * SIGMA_2
        x[:, 1] += 2.0 * (2 * y - 1)  # +2 for y=1, -2 for y=0
        # Dimension 1: high variance, correlated with MINORITY class (y=1)
        x[:, 0] = rng.standard_normal(n).astype(np.float32) * SIGMA_1
        x[:, 0] += MU_SHIFT * y  # shift positive for minority class only
        return torch.tensor(x, device=DEVICE), torch.tensor(y, device=DEVICE)

    x_train, y_train = make_data(N_TRAIN, SEED)
    x_test, y_test = make_data(N_TEST, SEED + 1)

    log(f"  P(y=1) in test: {y_test.mean().item():.2f}")
    corr1 = float(np.corrcoef(x_train[:, 0].cpu().numpy(), y_train.cpu().numpy())[0, 1])
    corr2 = float(np.corrcoef(x_train[:, 1].cpu().numpy(), y_train.cpu().numpy())[0, 1])
    log(f"  Corr(x1, y) = {corr1:+.3f}  Corr(x2, y) = {corr2:+.3f}")

    def evaluate(model, x, y, ablate=False):
        with torch.no_grad():
            xi = x.clone()
            if ablate:
                xi[:, 0] = xi[:, 0].mean()
            logits = model(xi).squeeze(-1)
            preds = (logits > 0).float()
            return float((preds == y).float().mean().item())

    out = {}
    for name, model_fn in [
        ("linear", lambda: nn.Linear(D, 1).to(DEVICE)),
        ("mlp", lambda: nn.Sequential(nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 1)).to(DEVICE)),
    ]:
        model = model_fn()
        for p in model.parameters():
            nn.init.normal_(p, std=0.01)
        opt = torch.optim.SGD(model.parameters(), lr=LR)
        traj = []

        for step in range(N_STEPS + 1):
            if step % EVAL_EVERY == 0:
                acc = evaluate(model, x_test, y_test)
                acc_abl = evaluate(model, x_test, y_test, ablate=True)
                traj.append({"step": step, "acc": acc, "acc_ablated": acc_abl, "delta": acc_abl - acc})
                if step % 500 == 0:
                    log(f"    {name} step={step:>5}  acc={acc:.3f}  acc_abl={acc_abl:.3f}  Δ={acc_abl-acc:+.4f}")
            if step < N_STEPS:
                logits = model(x_train).squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, y_train)
                opt.zero_grad()
                loss.backward()
                opt.step()

        train_traj = [r for r in traj if r["step"] > 0]
        min_train = min(train_traj, key=lambda r: r["acc"])
        out[name] = {
            "trajectory": traj,
            "min_acc_during_training": min_train["acc"],
            "min_acc_step": min_train["step"],
            "dip_below_50": min_train["acc"] < 0.50,
            "dip_below_45": min_train["acc"] < 0.45,
            "max_positive_delta": max(r["delta"] for r in traj),
            "min_delta": min(r["delta"] for r in traj),
            "final_acc": traj[-1]["acc"],
            "final_delta": traj[-1]["delta"],
        }
        log(f"  {name}: min_train_acc={min_train['acc']:.3f} @ step {min_train['step']}  "
            f"dip_below_50={'YES' if min_train['acc']<0.50 else 'no'}  "
            f"max_Δ={max(r['delta'] for r in traj):+.3f}  "
            f"min_Δ={min(r['delta'] for r in traj):+.3f}")
        del model, opt

    out["setup"] = {
        "sigma1": SIGMA_1, "sigma2": SIGMA_2, "p_minority": P_MINORITY,
        "mu_shift": MU_SHIFT, "lr": LR, "n_steps": N_STEPS,
        "corr_x1_y": corr1, "corr_x2_y": corr2,
    }
    return out


# ====================================================================
# PHASE 7: LOGIT LENS ON GREATER-THAN
# ====================================================================

def phase7_gt_logit_lens():
    """
    Same analysis as IOI logit lens but for greater-than.
    At each layer, project residual at final token through unembed.
    Compute P(>start_year) - P(<=start_year) per layer.
    If GT shows the same layer-8 transition pattern, the mechanism generalizes.
    """
    log("=" * 60)
    log("PHASE 7: Logit lens on greater-than")
    log("=" * 60)

    STEPS = [1000, 2000, 3000, 5000, 143000]
    GT_N = 100
    rng = np.random.default_rng(SEED)
    GT_EVENTS = ["war", "battle", "dispute", "conflict"]
    GT_VERBS = ["lasted", "ran", "continued"]

    out = {"by_step": {}}

    for step in STEPS:
        log(f"  step {step}:")
        model = load_pythia_original(step)

        # Build digit token mapping.
        digit_tokens = {}
        for d in range(100):
            ids = model.tokenizer.encode(f"{d:02d}", add_special_tokens=False)
            if len(ids) == 1:
                digit_tokens[d] = ids[0]
        valid_digits = sorted(digit_tokens.keys())

        # Generate prompts.
        prompts, start_years = [], []
        for _ in range(GT_N):
            event = rng.choice(GT_EVENTS)
            verb = rng.choice(GT_VERBS)
            y = int(rng.integers(3, 97))
            prompts.append(f"The {event} {verb} from the year {1700+y} to the year 17")
            start_years.append(y)

        tokens = model.to_tokens(prompts).to(DEVICE)

        # Cache all residual streams.
        names = [f"blocks.{L}.hook_resid_post" for L in range(model.cfg.n_layers)]
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=names)

        per_layer = {}
        for L in range(model.cfg.n_layers):
            resid = cache[f"blocks.{L}.hook_resid_post"][:, -1, :]
            normed = model.ln_final(resid)
            logits = normed @ model.W_U + model.b_U

            diffs = []
            for i in range(GT_N):
                probs = F.softmax(logits[i].float(), dim=-1).detach().cpu().numpy()
                gmask = [digit_tokens[d] for d in valid_digits if d > start_years[i]]
                lmask = [digit_tokens[d] for d in valid_digits if d <= start_years[i]]
                p_greater = sum(probs[t] for t in gmask) if gmask else 0
                p_less = sum(probs[t] for t in lmask) if lmask else 0
                diffs.append(p_greater - p_less)

            per_layer[f"layer_{L}"] = {
                "mean_diff": float(np.mean(diffs)),
                "std_diff": float(np.std(diffs)),
            }

        # Find flip layer.
        flip_layer = None
        for L in range(model.cfg.n_layers):
            if per_layer[f"layer_{L}"]["mean_diff"] > 0 and flip_layer is None:
                flip_layer = L

        ld0 = per_layer["layer_0"]["mean_diff"]
        ld_last = per_layer[f"layer_{model.cfg.n_layers-1}"]["mean_diff"]
        log(f"    L0={ld0:+.4f}  L{model.cfg.n_layers-1}={ld_last:+.4f}  flip_layer={flip_layer}")

        out["by_step"][f"step_{step}"] = {"per_layer": per_layer, "flip_layer": flip_layer}
        del model, cache; torch.cuda.empty_cache(); gc.collect()

    return out


# ====================================================================
# PHASE 8: PROBE — TEMPLATES ONLY HELD OUT
# ====================================================================

def phase8_probe_templates_only():
    log("=" * 60)
    log("PHASE 8: Probe generalization — templates only held out")
    log("=" * 60)

    PROBE_STEPS = [0, 2000, 143000]
    LAYER = 5

    out = {"by_step": {}}
    for step in PROBE_STEPS:
        log(f"  step {step}:")
        model = load_pythia_original(step)
        _, all_names = get_single_token_names(model.tokenizer)
        all_templates = ALL_TEMPLATES[:30]

        rng = np.random.default_rng(SEED)
        shuffled_t = list(all_templates)
        rng.shuffle(shuffled_t)
        half = len(shuffled_t) // 2
        templates_A = shuffled_t[:half]
        templates_B = shuffled_t[half:]

        ioi_A, ctrl_A = collect_s2_activations(model, all_names, templates_A, LAYER, SEED)
        ioi_B, ctrl_B = collect_s2_activations(model, all_names, templates_B, LAYER, SEED + 1)

        if ioi_A.shape[0] < 10 or ioi_B.shape[0] < 10:
            log(f"    insufficient data")
            continue

        X_A = np.concatenate([ioi_A, ctrl_A])
        y_A = np.concatenate([np.ones(len(ioi_A)), np.zeros(len(ctrl_A))])
        X_B = np.concatenate([ioi_B, ctrl_B])
        y_B = np.concatenate([np.ones(len(ioi_B)), np.zeros(len(ctrl_B))])

        clf_A = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
        clf_A.fit(X_A, y_A)
        in_A = clf_A.score(X_A, y_A)
        out_B = clf_A.score(X_B, y_B)

        clf_B = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
        clf_B.fit(X_B, y_B)
        in_B = clf_B.score(X_B, y_B)
        out_A = clf_B.score(X_A, y_A)

        out["by_step"][f"step_{step}"] = {
            "trained_on_A": {"in_dist": float(in_A), "held_out": float(out_B)},
            "trained_on_B": {"in_dist": float(in_B), "held_out": float(out_A)},
            "mean_held_out": float((out_A + out_B) / 2),
        }
        log(f"    A->B: in={in_A*100:.1f}% out={out_B*100:.1f}%  "
            f"B->A: in={in_B*100:.1f}% out={out_A*100:.1f}%  "
            f"mean={(out_A+out_B)/2*100:.1f}%")

        del model; torch.cuda.empty_cache(); gc.collect()

    return out


# ====================================================================
# MAIN
# ====================================================================

def main():
    os.makedirs("results", exist_ok=True)
    results = {"config": {"seed": SEED}}

    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH) as f:
                results = json.load(f)
        except Exception:
            pass

    phases = [
        ("phase1_toy_sweep", phase1_toy_sweep),
        ("phase2_probe_combinations", phase2_probe_combinations),
        ("phase3_probe_names_only", phase3_probe_names_only),
        ("phase4_sae_proper", phase4_sae_proper),
        ("phase5_cross_task_pca", phase5_cross_task_pca),
        ("phase6_imbalanced_toy", phase6_imbalanced_toy),
        ("phase7_gt_logit_lens", phase7_gt_logit_lens),
        ("phase8_probe_templates_only", phase8_probe_templates_only),
    ]

    t0 = time.time()
    for key, fn in phases:
        cached = results.get(key)
        if cached is not None and "error" not in cached:
            log(f"SKIP {key}: already done")
            continue
        if cached is not None and "error" in cached:
            log(f"RETRY {key}: previous error")
        log(f"START {key}")
        try:
            results[key] = fn()
            save_results(results)
        except Exception as e:
            log(f"FAILED {key}: {e}")
            traceback.print_exc()
            results[key] = {"error": str(e)}
            save_results(results)

    elapsed = (time.time() - t0) / 60.0
    log(f"Done. Total elapsed: {elapsed:.1f} min. Output: {RESULTS_PATH}")

if __name__ == "__main__":
    main()
