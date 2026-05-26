"""
EMNLP Tier S: SAE Feature Identification & Ablation
====================================================

Closes the main methodological gap in the paper. An earlier
random-perturbation control gave 88% of the S2 residual-stream patching
effect, suggesting the duplication signal may not be cleanly localized
in any small structure. SAEs let us test this: we train a sparse
decomposition of the residual stream, identify features that correlate
with duplication, and check whether ablating a small set of features
captures the patching effect.

PRE-REGISTERED HYPOTHESIS
-------------------------
If the S-promoting effect during the dip is mediated by a *small set of
learned features* writing to the residual stream at S2, then ablating
the top-K such features should reproduce most of the whole-residual
patching ΔLD. If the effect is genuinely distributed, even K=20+
features should capture less than ~50% of the full effect.

PHASES
------
PHASE 1: Activation collection
  - Retrained Pythia-160M at step 2000 (dip floor) and step 5000 (mature)
  - Layers 3, 4, 5 (the patching layers)
  - Mix of IOI prompts, controls, and natural text for diversity
  - Save activation tensors to disk

PHASE 2: SAE training (hyperparameter sweep)
  - Hand-rolled ReLU SAE with L1 sparsity penalty
  - Sweep: expansion ∈ {4x, 8x} × L1 coef ∈ {1e-3, 3e-3, 1e-2}
  - For each (step, layer, config): train 3000 steps
  - Save state dicts + training curves

PHASE 3: Feature identification (per best SAE per (step, layer))
  - Encode IOI and control activations at S2 position
  - Train logistic regression to classify IOI vs control
  - Top features by classifier weight magnitude

PHASE 4: Feature ablation at S2 (step 2000)
  - For K in {1, 3, 5, 10}: ablate top-K features at S2
  - Inject modified residual into the forward pass
  - Measure ΔLD vs SAE-reconstruction baseline
  - Compare to whole-residual patching ΔLD

PHASE 5: Cross-task transfer (greater-than)
  - At step 1100 of retrained Pythia (greater-than dip floor)
  - Encode activations at first-year token positions
  - Check if IOI-trained duplication features fire on greater-than
  - Headline cross-task mechanistic claim

Output: results/emnlp_tier_s.json
SAE weights: results/tier_s_saes/*.pt
Log: results/emnlp_tier_s_log.txt
Runtime: ~2.5-4 hours on A100.
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
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES, CANDIDATE_NAMES
except ImportError:
    sys.path.insert(0, "/workspace/ioi-sign-flip/src")
    from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES, CANDIDATE_NAMES


# ----------------------------- Config -----------------------------

DEVICE = "cuda"
RETRAINED_REPO = "anonymous-research-sub/pythia-160m-retrained-seed42"
BASE_MODEL = "EleutherAI/pythia-160m-deduped"
D_MODEL = 768
SEED = 42

# Phase 1: where and when to collect activations
COLLECTION_STEPS = [2000, 5000]
COLLECTION_LAYERS = [3, 4, 5]
N_IOI_PROMPTS = 500           # per condition (IOI / control)
N_NATURAL_PROMPTS = 100        # natural-text prompts for diversity

# Phase 2: SAE training sweep
EXPANSION_FACTORS = [4, 8]     # dict_size = D_MODEL * expansion
L1_COEFS = [1e-3, 3e-3, 1e-2]
N_TRAIN_STEPS = 3000
BATCH_SIZE = 1024
LR = 3e-4
L1_WARMUP_STEPS = 200

# Phase 3: feature identification
TOP_K_VALUES = [1, 3, 5, 10]
PROBE_C = 1.0                   # LR regularization

# Phase 4: ablation at S2 (use IOI dense-sweep dip step on retrained = 1400)
# But Phase 1 collected at step 2000, which is also in the dip. Match.
ABLATION_STEP = 2000

# Phase 5: cross-task transfer
GREATER_THAN_DIP_STEP = 1100   # retrained Pythia greater-than dip floor
GT_EVENTS = ["war", "battle", "dispute", "conflict", "argument", "siege"]
GT_VERBS = ["lasted", "ran", "extended", "continued", "stretched"]
GT_N_PROMPTS = 200

# Natural-text seed corpus for diversity in SAE training
NATURAL_TEXTS = [
    "The capital of France is Paris.",
    "Water boils at one hundred degrees Celsius.",
    "Shakespeare wrote many famous plays during his lifetime.",
    "Mount Everest stands as the tallest mountain on Earth.",
    "The Pacific Ocean covers most of the southern hemisphere.",
    "Light travels at three hundred thousand kilometers per second.",
    "The human body contains over two hundred bones.",
    "The Great Wall of China stretches across northern provinces.",
    "Albert Einstein developed the theory of general relativity.",
    "The Amazon rainforest contains millions of species.",
    "Music has been part of human culture for thousands of years.",
    "Coffee is grown in tropical regions around the world.",
    "The Renaissance began in Italy during the fourteenth century.",
    "Neutron stars are among the densest objects in the universe.",
    "The Roman Empire lasted for over a thousand years.",
    "Photosynthesis converts sunlight into chemical energy.",
    "The piano is a popular instrument with eighty-eight keys.",
    "Ancient Egypt was famous for its pyramids and pharaohs.",
    "The internet was developed in the late twentieth century.",
    "Diamonds form deep within the earth under intense pressure.",
    "The Mediterranean Sea connects three continents.",
    "Chess originated in India over fifteen hundred years ago.",
    "The Industrial Revolution transformed Western economies.",
    "Black holes have such strong gravity that not even light can escape.",
    "The English language has more than one hundred thousand words.",
]

RESULTS_PATH = "results/emnlp_tier_s.json"
SAE_DIR = "results/tier_s_saes"
ACTS_DIR = "results/tier_s_activations"


# --------------------------- Utilities ----------------------------

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_retrained(step):
    hf = AutoModelForCausalLM.from_pretrained(
        RETRAINED_REPO, subfolder=f"step_{step}", torch_dtype=torch.float32,
    )
    model = HookedTransformer.from_pretrained(
        BASE_MODEL, hf_model=hf, device=DEVICE,
        center_writing_weights=True, center_unembed=True, fold_ln=True,
    )
    del hf
    torch.cuda.empty_cache()
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
    ids = []
    for name in CANDIDATE_NAMES:
        toks = tokenizer.encode(" " + name, add_special_tokens=False)
        if len(toks) == 1:
            ids.append(int(toks[0]))
    return ids


def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


# ====================================================================
# PHASE 1: ACTIVATION COLLECTION
# ====================================================================

def collect_activations(model, layers):
    """Collect residual-stream activations from a mix of IOI prompts,
    control prompts, and natural text. Returns dict:
      {
        layer: {
          "s2_ioi":   tensor [N_IOI_PROMPTS, d_model],
          "s2_ctrl":  tensor [N_IOI_PROMPTS, d_model],
          "general":  tensor [N_total_tokens, d_model],
        }
      }
    """
    rng = np.random.default_rng(SEED)
    single_name_ids = get_single_token_names(model.tokenizer)

    # Build IOI/ctrl prompts.
    log("    generating IOI/control prompt pairs...")
    ioi_acts = {L: [] for L in layers}
    ctrl_acts = {L: [] for L in layers}
    general_acts = {L: [] for L in layers}

    pts_per_template = max(1, N_IOI_PROMPTS // len(ALL_TEMPLATES[:10]))
    for tmpl_idx, tmpl in enumerate(ALL_TEMPLATES[:10]):
        ds = IOIDataset(
            model=model, n_prompts=pts_per_template, templates=[tmpl],
            symmetric=True, seed=SEED + tmpl_idx,
        )
        ioi_tokens = model.to_tokens(ds.prompts).to(DEVICE)
        n = ioi_tokens.shape[0]

        s2_positions = []
        for i in range(n):
            s2_positions.append(find_s2_position(
                ioi_tokens[i].cpu(), ds.s_token_ids[i],
            ))

        ctrl_tokens = ioi_tokens.clone()
        for i in range(n):
            io_id = int(ds.io_token_ids[i])
            s_id = int(ds.s_token_ids[i])
            pool = [t for t in single_name_ids if t != io_id and t != s_id]
            if pool and s2_positions[i] > 0:
                ctrl_tokens[i, s2_positions[i]] = int(rng.choice(pool))

        hook_names = [f"blocks.{L}.hook_resid_post" for L in layers]
        cache = {}

        def make_cap(name):
            def fn(value, hook):
                cache[name] = value.detach()
                return value
            return fn

        # IOI activations.
        with torch.no_grad():
            model.run_with_hooks(
                ioi_tokens,
                fwd_hooks=[(n_, make_cap(n_)) for n_ in hook_names],
            )
        for L in layers:
            for i in range(n):
                p = s2_positions[i]
                if p > 0:
                    ioi_acts[L].append(cache[f"blocks.{L}.hook_resid_post"][i, p].cpu().float())
                    # Also use all positions for "general" pool.
                    for j in range(1, ioi_tokens.shape[1]):
                        general_acts[L].append(cache[f"blocks.{L}.hook_resid_post"][i, j].cpu().float())

        cache.clear()

        # Control activations.
        with torch.no_grad():
            model.run_with_hooks(
                ctrl_tokens,
                fwd_hooks=[(n_, make_cap(n_)) for n_ in hook_names],
            )
        for L in layers:
            for i in range(n):
                p = s2_positions[i]
                if p > 0:
                    ctrl_acts[L].append(cache[f"blocks.{L}.hook_resid_post"][i, p].cpu().float())

        del cache
        torch.cuda.empty_cache()

    # Add natural-text activations to general pool for diversity.
    log("    collecting natural-text activations...")
    for txt in NATURAL_TEXTS:
        tokens = model.to_tokens(txt).to(DEVICE)
        cache = {}
        hook_names = [f"blocks.{L}.hook_resid_post" for L in layers]

        def make_cap(name):
            def fn(value, hook):
                cache[name] = value.detach()
                return value
            return fn

        with torch.no_grad():
            model.run_with_hooks(
                tokens,
                fwd_hooks=[(n_, make_cap(n_)) for n_ in hook_names],
            )
        for L in layers:
            for j in range(1, tokens.shape[1]):
                general_acts[L].append(cache[f"blocks.{L}.hook_resid_post"][0, j].cpu().float())
        del cache
        torch.cuda.empty_cache()

    # Stack into tensors.
    result = {}
    for L in layers:
        result[L] = {
            "s2_ioi":   torch.stack(ioi_acts[L]),
            "s2_ctrl":  torch.stack(ctrl_acts[L]),
            "general":  torch.stack(general_acts[L]),
        }
        log(
            f"    layer {L}: ioi={result[L]['s2_ioi'].shape[0]}  "
            f"ctrl={result[L]['s2_ctrl'].shape[0]}  "
            f"general={result[L]['general'].shape[0]}"
        )
    return result


def phase1_collect_activations():
    log("=" * 60)
    log("PHASE 1: Activation collection")
    log("=" * 60)
    os.makedirs(ACTS_DIR, exist_ok=True)

    out = {"steps": {}}
    for step in COLLECTION_STEPS:
        log(f"-- step {step} --")
        model = load_retrained(step)
        acts = collect_activations(model, COLLECTION_LAYERS)

        # Save to disk for SAE training.
        for L in COLLECTION_LAYERS:
            for key in ("s2_ioi", "s2_ctrl", "general"):
                path = os.path.join(ACTS_DIR, f"step{step}_L{L}_{key}.pt")
                torch.save(acts[L][key], path)

        out["steps"][f"step_{step}"] = {
            f"layer_{L}": {
                "s2_ioi_n": int(acts[L]["s2_ioi"].shape[0]),
                "s2_ctrl_n": int(acts[L]["s2_ctrl"].shape[0]),
                "general_n": int(acts[L]["general"].shape[0]),
            }
            for L in COLLECTION_LAYERS
        }
        del model, acts
        torch.cuda.empty_cache()
        gc.collect()

    return out


# ====================================================================
# SAE class
# ====================================================================

class SAE(nn.Module):
    """Standard ReLU SAE with pre-bias (Anthropic style):
        h = ReLU((x - b_pre) W_enc + b_enc)
        x_hat = h W_dec + b_pre
    Decoder rows are unit-normalized after each optimizer step.
    """
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.b_pre = nn.Parameter(torch.zeros(d_in))
        # Kaiming-ish init for encoder
        scale = 1.0 / (d_in ** 0.5)
        self.W_enc = nn.Parameter(torch.randn(d_in, d_hidden) * scale)
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        # Decoder initialized as encoder transpose, then normalized.
        self.W_dec = nn.Parameter(self.W_enc.data.T.clone().contiguous())
        with torch.no_grad():
            self.W_dec.data /= (self.W_dec.data.norm(dim=-1, keepdim=True) + 1e-8)

    def encode(self, x):
        return F.relu((x - self.b_pre) @ self.W_enc + self.b_enc)

    def decode(self, h):
        return h @ self.W_dec + self.b_pre

    def forward(self, x):
        h = self.encode(x)
        x_hat = self.decode(h)
        return x_hat, h

    @torch.no_grad()
    def renorm_decoder(self):
        norms = self.W_dec.data.norm(dim=-1, keepdim=True) + 1e-8
        self.W_dec.data /= norms


def train_sae(acts, d_hidden, l1_coef, n_steps=N_TRAIN_STEPS,
              batch_size=BATCH_SIZE, lr=LR, l1_warmup=L1_WARMUP_STEPS,
              seed=SEED):
    """Train one SAE on activations. Returns trained SAE and metrics."""
    torch.manual_seed(seed)
    d_in = acts.shape[1]
    sae = SAE(d_in, d_hidden).to(DEVICE)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)

    acts_dev = acts.to(DEVICE)
    n = acts_dev.shape[0]

    history = []
    for step in range(n_steps):
        idx = torch.randint(0, n, (batch_size,), device=DEVICE)
        x = acts_dev[idx]
        x_hat, h = sae(x)
        recon = ((x - x_hat) ** 2).sum(dim=-1).mean()
        l0 = (h > 0).float().sum(dim=-1).mean()
        sparsity = h.abs().sum(dim=-1).mean()
        l1_now = l1_coef * min(1.0, (step + 1) / l1_warmup)
        loss = recon + l1_now * sparsity
        opt.zero_grad()
        loss.backward()
        opt.step()
        sae.renorm_decoder()
        if step == n_steps - 1 or step % 500 == 0:
            history.append({
                "step": step,
                "recon": float(recon.item()),
                "l0": float(l0.item()),
                "sparsity": float(sparsity.item()),
            })

    # Final evaluation on full activation set.
    with torch.no_grad():
        x_hat_all, h_all = sae(acts_dev)
        recon_full = ((acts_dev - x_hat_all) ** 2).sum(dim=-1).mean()
        # Explained variance.
        var_x = acts_dev.var(dim=0).mean()
        evar = 1 - recon_full / (var_x * d_in)
        l0_full = (h_all > 0).float().sum(dim=-1).mean()

    metrics = {
        "final_recon_mse": float(recon_full.item()),
        "explained_variance": float(evar.item()),
        "mean_l0": float(l0_full.item()),
        "history": history,
    }
    return sae, metrics


# ====================================================================
# PHASE 2: SAE TRAINING
# ====================================================================

def phase2_train_saes():
    log("=" * 60)
    log("PHASE 2: SAE training sweep")
    log("=" * 60)
    os.makedirs(SAE_DIR, exist_ok=True)

    out = {"sweeps": {}, "best_per_layer_step": {}}

    for step in COLLECTION_STEPS:
        out["sweeps"][f"step_{step}"] = {}
        for L in COLLECTION_LAYERS:
            path = os.path.join(ACTS_DIR, f"step{step}_L{L}_general.pt")
            if not os.path.exists(path):
                log(f"    missing {path}, skipping")
                continue
            acts = torch.load(path)
            log(
                f"  step {step}, layer {L}: training {len(EXPANSION_FACTORS) * len(L1_COEFS)} SAEs "
                f"on {acts.shape[0]} activations"
            )
            sweep_results = []
            for expansion in EXPANSION_FACTORS:
                d_hidden = D_MODEL * expansion
                for l1 in L1_COEFS:
                    t0 = time.time()
                    sae, metrics = train_sae(acts, d_hidden, l1)
                    elapsed = time.time() - t0
                    sae_key = f"step{step}_L{L}_e{expansion}_l1{l1:.0e}"
                    sae_path = os.path.join(SAE_DIR, f"{sae_key}.pt")
                    torch.save({
                        "state_dict": sae.state_dict(),
                        "d_in": sae.d_in,
                        "d_hidden": sae.d_hidden,
                        "expansion": expansion,
                        "l1_coef": l1,
                        "step": step,
                        "layer": L,
                    }, sae_path)
                    sweep_results.append({
                        "key": sae_key,
                        "expansion": expansion,
                        "l1_coef": l1,
                        "metrics": metrics,
                        "elapsed_sec": elapsed,
                    })
                    log(
                        f"    {sae_key}: recon={metrics['final_recon_mse']:.4f}  "
                        f"evar={metrics['explained_variance']:.3f}  "
                        f"L0={metrics['mean_l0']:.1f}  ({elapsed:.0f}s)"
                    )
                    del sae
                    torch.cuda.empty_cache()
            out["sweeps"][f"step_{step}"][f"layer_{L}"] = sweep_results

            # Select best by combined criterion:
            # high explained_variance and reasonable L0 (10 <= L0 <= 200).
            def score(r):
                m = r["metrics"]
                l0 = m["mean_l0"]
                if l0 < 5 or l0 > 500:
                    return -1e9    # too degenerate
                return m["explained_variance"] - 0.001 * l0
            best = max(sweep_results, key=score)
            out["best_per_layer_step"][f"step{step}_L{L}"] = best["key"]
            log(f"  >> best: {best['key']}")

    return out


# ====================================================================
# Hook-based SAE injection
# ====================================================================

def inject_sae_hook(sae, position_per_row, ablate_features=None):
    """Returns a hook that, at each row's specified position, replaces
    the residual with sae.decode(sae.encode(x)) with optional feature
    ablation (zero out specific features in the encoded space).
    Other positions pass through unchanged.
    """
    def hook_fn(value, hook):
        for i in range(value.shape[0]):
            p = int(position_per_row[i].item()) if torch.is_tensor(position_per_row) else int(position_per_row[i])
            if p > 0:
                x = value[i, p, :]
                h = sae.encode(x.unsqueeze(0)).squeeze(0)
                if ablate_features is not None:
                    h = h.clone()
                    h[ablate_features] = 0.0
                x_hat = sae.decode(h.unsqueeze(0)).squeeze(0)
                value[i, p, :] = x_hat
        return value
    return hook_fn


# ====================================================================
# PHASE 3: FEATURE IDENTIFICATION
# ====================================================================

def load_sae_from_path(path):
    ckpt = torch.load(path, map_location=DEVICE)
    sae = SAE(ckpt["d_in"], ckpt["d_hidden"]).to(DEVICE)
    sae.load_state_dict(ckpt["state_dict"])
    sae.eval()
    return sae, ckpt


def phase3_feature_identification(phase2_out):
    log("=" * 60)
    log("PHASE 3: Feature identification")
    log("=" * 60)

    out = {}
    for step in COLLECTION_STEPS:
        out[f"step_{step}"] = {}
        for L in COLLECTION_LAYERS:
            best_key = phase2_out["best_per_layer_step"].get(f"step{step}_L{L}")
            if best_key is None:
                continue
            sae_path = os.path.join(SAE_DIR, f"{best_key}.pt")
            sae, _ = load_sae_from_path(sae_path)

            # Encode IOI and control S2 activations.
            ioi_path = os.path.join(ACTS_DIR, f"step{step}_L{L}_s2_ioi.pt")
            ctrl_path = os.path.join(ACTS_DIR, f"step{step}_L{L}_s2_ctrl.pt")
            x_ioi = torch.load(ioi_path).to(DEVICE)
            x_ctrl = torch.load(ctrl_path).to(DEVICE)

            with torch.no_grad():
                h_ioi = sae.encode(x_ioi).cpu().numpy()
                h_ctrl = sae.encode(x_ctrl).cpu().numpy()

            X = np.concatenate([h_ioi, h_ctrl], axis=0)
            y = np.concatenate([
                np.ones(h_ioi.shape[0]),
                np.zeros(h_ctrl.shape[0]),
            ])

            # Logistic regression on feature activations.
            clf = LogisticRegression(max_iter=2000, C=PROBE_C, random_state=SEED)
            clf.fit(X, y)
            train_acc = clf.score(X, y)

            # Top features by |weight|.
            weights = clf.coef_[0]
            order = np.argsort(-np.abs(weights))
            top_features = []
            for k in range(min(30, len(order))):
                idx = int(order[k])
                top_features.append({
                    "feature_idx": idx,
                    "weight": float(weights[idx]),
                    "mean_act_ioi": float(h_ioi[:, idx].mean()),
                    "mean_act_ctrl": float(h_ctrl[:, idx].mean()),
                    "frac_active_ioi": float((h_ioi[:, idx] > 0).mean()),
                    "frac_active_ctrl": float((h_ctrl[:, idx] > 0).mean()),
                })

            out[f"step_{step}"][f"layer_{L}"] = {
                "sae_key": best_key,
                "probe_train_acc": float(train_acc),
                "top_features": top_features,
            }
            log(
                f"  step{step} L{L}: probe_acc={train_acc*100:.1f}%  "
                f"top feature idx={top_features[0]['feature_idx']} "
                f"weight={top_features[0]['weight']:+.3f}  "
                f"ioi_act={top_features[0]['mean_act_ioi']:.3f} "
                f"ctrl_act={top_features[0]['mean_act_ctrl']:.3f}"
            )

            del sae, x_ioi, x_ctrl
            torch.cuda.empty_cache()

    return out


# ====================================================================
# PHASE 4: FEATURE ABLATION AT S2
# ====================================================================

def phase4_feature_ablation(phase2_out, phase3_out):
    """At ABLATION_STEP (=2000), ablate top-K features at S2 for each
    layer and measure the change in IOI logit difference. Compare to
    the whole-residual patching ΔLD baseline from existing data.
    """
    log("=" * 60)
    log("PHASE 4: SAE feature ablation at S2")
    log("=" * 60)

    step = ABLATION_STEP
    model = load_retrained(step)
    single_name_ids = get_single_token_names(model.tokenizer)
    rng = np.random.default_rng(SEED + 7)

    # Build a fresh IOI/ctrl set for ablation eval.
    all_ioi_tokens = []
    all_s2_positions = []
    all_io_ids = []
    all_s_ids = []
    for tmpl in ALL_TEMPLATES[:10]:
        ds = IOIDataset(
            model=model, n_prompts=30, templates=[tmpl],
            symmetric=True, seed=SEED,
        )
        ioi_tokens = model.to_tokens(ds.prompts).to(DEVICE)
        n = ioi_tokens.shape[0]
        s2_positions = []
        for i in range(n):
            s2_positions.append(find_s2_position(
                ioi_tokens[i].cpu(), ds.s_token_ids[i],
            ))
        s2_positions = torch.tensor(s2_positions, dtype=torch.long, device=DEVICE)
        io_ids = torch.tensor(ds.io_token_ids, dtype=torch.long, device=DEVICE)
        s_ids = torch.tensor(ds.s_token_ids, dtype=torch.long, device=DEVICE)
        all_ioi_tokens.append(ioi_tokens)
        all_s2_positions.append(s2_positions)
        all_io_ids.append(io_ids)
        all_s_ids.append(s_ids)

    def compute_ld(model_call_fn):
        """Run the IOI prompts through model_call_fn and return per-prompt LDs."""
        lds = []
        for ioi_tokens, s2_positions, io_ids, s_ids in zip(
            all_ioi_tokens, all_s2_positions, all_io_ids, all_s_ids,
        ):
            with torch.no_grad():
                logits = model_call_fn(ioi_tokens, s2_positions)
            last = logits[:, -1, :]
            idx = torch.arange(last.shape[0], device=DEVICE)
            lds.extend((last[idx, io_ids] - last[idx, s_ids]).cpu().tolist())
        return np.array(lds)

    out = {"step": step, "by_layer": {}}

    # Baseline (no SAE intervention).
    base_lds = compute_ld(lambda toks, s2: model(toks))
    base_mean = float(base_lds.mean())
    log(f"  baseline LD={base_mean:+.4f}")

    for L in COLLECTION_LAYERS:
        best_key = phase2_out["best_per_layer_step"].get(f"step{step}_L{L}")
        if best_key is None:
            continue
        sae_path = os.path.join(SAE_DIR, f"{best_key}.pt")
        sae, _ = load_sae_from_path(sae_path)

        top_features = phase3_out[f"step_{step}"][f"layer_{L}"]["top_features"]
        top_idxs = [tf["feature_idx"] for tf in top_features]
        hook_name = f"blocks.{L}.hook_resid_post"

        # SAE-reconstruction-only baseline (no ablation): tells us how
        # much LD changes just from passing through the SAE.
        def call_with_sae_recon(toks, s2):
            return model.run_with_hooks(
                toks,
                fwd_hooks=[(hook_name, inject_sae_hook(sae, s2, None))],
            )
        recon_lds = compute_ld(call_with_sae_recon)
        recon_mean = float(recon_lds.mean())
        recon_delta = float(recon_mean - base_mean)
        log(f"  L{L} SAE-recon baseline: LD={recon_mean:+.4f}  Δ={recon_delta:+.4f}")

        layer_results = {
            "sae_key": best_key,
            "sae_recon_baseline_ld": recon_mean,
            "sae_recon_delta": recon_delta,
            "ablations": {},
        }

        for K in TOP_K_VALUES:
            ablate_idxs = torch.tensor(top_idxs[:K], device=DEVICE)

            def call_with_ablation(toks, s2):
                return model.run_with_hooks(
                    toks,
                    fwd_hooks=[(hook_name, inject_sae_hook(sae, s2, ablate_idxs))],
                )

            abl_lds = compute_ld(call_with_ablation)
            abl_mean = float(abl_lds.mean())
            # Delta vs SAE-recon baseline isolates the ablation effect.
            delta_vs_recon = float(abl_mean - recon_mean)
            # Delta vs unmodified isolates the total effect of injecting
            # SAE + ablating K features.
            delta_vs_base = float(abl_mean - base_mean)

            layer_results["ablations"][f"K_{K}"] = {
                "ablated_ld": abl_mean,
                "delta_vs_recon": delta_vs_recon,
                "delta_vs_base": delta_vs_base,
            }
            log(
                f"    K={K:>2}: LD={abl_mean:+.4f}  "
                f"Δ_vs_recon={delta_vs_recon:+.4f}  "
                f"Δ_vs_base={delta_vs_base:+.4f}"
            )

        out["by_layer"][f"layer_{L}"] = layer_results
        del sae
        torch.cuda.empty_cache()

    # Comparison reference: whole-residual S2 patching ΔLD from dense
    # sweep at step 2000 on retrained Pythia = +0.37 (approximately).
    out["whole_residual_patching_reference"] = {
        "step": step,
        "delta_ld_full": 0.37,
        "source": "whole-residual S2 patching at step 2000 (dense sweep)",
    }

    del model
    torch.cuda.empty_cache()
    return out


# ====================================================================
# PHASE 5: CROSS-TASK TRANSFER (GREATER-THAN)
# ====================================================================

def make_greater_than_prompts(seed=SEED):
    rng = np.random.default_rng(seed)
    prompts = []
    for _ in range(GT_N_PROMPTS):
        event = rng.choice(GT_EVENTS)
        verb = rng.choice(GT_VERBS)
        y_low_2 = int(rng.integers(3, 97))
        year = 1700 + y_low_2
        prompts.append({
            "prompt": f"The {event} {verb} from the year {year} to the year 17",
            "start_yy": y_low_2,
        })
    return prompts


def phase5_cross_task_transfer(phase2_out, phase3_out):
    """At greater-than dip floor (step 1100), encode activations at the
    first-year token position using IOI-trained SAEs (from step 2000).
    Check if duplication features (top-K from phase 3) have elevated
    activation on greater-than prompts.

    If yes → mechanistic cross-task transfer: same feature drives both
    the IOI dip and the greater-than dip.
    """
    log("=" * 60)
    log("PHASE 5: Cross-task feature transfer (greater-than)")
    log("=" * 60)

    out = {"dip_step": GREATER_THAN_DIP_STEP, "by_layer": {}}

    model = load_retrained(GREATER_THAN_DIP_STEP)
    prompts = make_greater_than_prompts(SEED)

    # Tokenize and find first-year token position per prompt. Pythia BPE
    # typically tokenizes " 1732" as " 17" + "32" (two tokens). The
    # year-start position is the first " 17" token; the prompt also
    # ends with " 17" but we want the first occurrence.
    log("    locating first-year positions in greater-than prompts...")
    year_prefix_ids = model.tokenizer.encode(" 17", add_special_tokens=False)
    if len(year_prefix_ids) == 1:
        year_prefix_token = year_prefix_ids[0]
    else:
        year_prefix_token = None
        log(f"    WARN: ' 17' tokenizes to {len(year_prefix_ids)} tokens, using fallback detection")

    year_positions = []
    tokens_list = []
    for p in prompts:
        toks = model.to_tokens(p["prompt"]).to(DEVICE)
        seq = toks[0].cpu().tolist()
        year_pos = -1

        # Primary: find first " 17" token (the year prefix).
        if year_prefix_token is not None:
            for j in range(1, toks.shape[1]):
                if seq[j] == year_prefix_token:
                    year_pos = j
                    break

        # Fallback: any token whose decoded form looks like a year prefix.
        if year_pos < 0:
            for j in range(1, toks.shape[1]):
                piece = model.tokenizer.decode([seq[j]]).strip()
                if piece.startswith("17") and 2 <= len(piece) <= 4 and piece.replace(" ", "").isdigit():
                    year_pos = j
                    break

        year_positions.append(year_pos)
        tokens_list.append(toks)
    log(
        f"    found year position in {sum(1 for p in year_positions if p >= 0)}/{len(year_positions)} prompts"
    )

    # For each (step, layer) with SAEs, run greater-than prompts and
    # extract activations at first-year positions, then encode through
    # the SAE.
    sae_step = 2000   # use SAEs trained at IOI dip step
    for L in COLLECTION_LAYERS:
        best_key = phase2_out["best_per_layer_step"].get(f"step{sae_step}_L{L}")
        if best_key is None:
            continue
        sae_path = os.path.join(SAE_DIR, f"{best_key}.pt")
        sae, _ = load_sae_from_path(sae_path)

        feature_activations = []
        hook_name = f"blocks.{L}.hook_resid_post"

        for toks, year_pos in zip(tokens_list, year_positions):
            if year_pos < 0:
                continue
            cache = {}

            def cap(value, hook):
                cache["resid"] = value.detach()
                return value

            with torch.no_grad():
                model.run_with_hooks(toks, fwd_hooks=[(hook_name, cap)])
            resid_at_year = cache["resid"][0, year_pos]
            with torch.no_grad():
                h = sae.encode(resid_at_year.unsqueeze(0)).squeeze(0).cpu().numpy()
            feature_activations.append(h)

        feature_activations = np.stack(feature_activations)  # [N, d_hidden]

        # Compare top-K duplication features' mean activation on
        # greater-than vs the IOI/ctrl baselines (from phase 3 data).
        top_features = phase3_out[f"step_{sae_step}"][f"layer_{L}"]["top_features"]

        feature_summary = []
        for tf in top_features[:10]:
            idx = tf["feature_idx"]
            gt_act = float(feature_activations[:, idx].mean())
            gt_frac = float((feature_activations[:, idx] > 0).mean())
            feature_summary.append({
                "feature_idx": idx,
                "ioi_mean": tf["mean_act_ioi"],
                "ctrl_mean": tf["mean_act_ctrl"],
                "gt_mean": gt_act,
                "ioi_frac_active": tf["frac_active_ioi"],
                "ctrl_frac_active": tf["frac_active_ctrl"],
                "gt_frac_active": gt_frac,
            })

        # Top duplication-feature activation on greater-than:
        top_idx = top_features[0]["feature_idx"]
        log(
            f"  layer {L} (SAE {best_key}): top dup feature idx={top_idx}  "
            f"ioi={top_features[0]['mean_act_ioi']:.3f}  "
            f"ctrl={top_features[0]['mean_act_ctrl']:.3f}  "
            f"gt={feature_activations[:, top_idx].mean():.3f}  "
            f"gt_active%={(feature_activations[:, top_idx] > 0).mean()*100:.1f}%"
        )

        out["by_layer"][f"layer_{L}"] = {
            "sae_key": best_key,
            "n_prompts_used": int(feature_activations.shape[0]),
            "top_features_cross_task": feature_summary,
        }

        del sae
        torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()
    return out


# ====================================================================
# MAIN
# ====================================================================

def main():
    os.makedirs("results", exist_ok=True)
    os.makedirs(SAE_DIR, exist_ok=True)
    os.makedirs(ACTS_DIR, exist_ok=True)

    results = {"config": {
        "model": RETRAINED_REPO,
        "seed": SEED,
        "steps": COLLECTION_STEPS,
        "layers": COLLECTION_LAYERS,
        "expansion_factors": EXPANSION_FACTORS,
        "l1_coefs": L1_COEFS,
        "n_train_steps": N_TRAIN_STEPS,
    }}

    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH) as f:
                results = json.load(f)
            done = [k for k in results if k.startswith("phase")]
            log(f"Resuming. Already done: {done}")
        except Exception as e:
            log(f"Could not resume: {e}")

    phase_fns = [
        ("phase1_collect_activations", phase1_collect_activations),
        ("phase2_train_saes", phase2_train_saes),
        ("phase3_feature_identification",
         lambda: phase3_feature_identification(results["phase2_train_saes"])),
        ("phase4_feature_ablation",
         lambda: phase4_feature_ablation(
             results["phase2_train_saes"],
             results["phase3_feature_identification"],
         )),
        ("phase5_cross_task_transfer",
         lambda: phase5_cross_task_transfer(
             results["phase2_train_saes"],
             results["phase3_feature_identification"],
         )),
    ]

    t0 = time.time()
    for key, fn in phase_fns:
        cached = results.get(key)
        if cached is not None and "error" not in cached:
            log(f"SKIP {key}: already done")
            continue
        if cached is not None and "error" in cached:
            log(f"RETRY {key}: previous run errored ({cached['error']})")
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
