"""
EMNLP Additional Control Experiments
======================================
Additional control experiments: analytical toy dip, logit-lens,
PCA ablation, multi-seed grokking, and a cross-family (OLMo) check.

PHASE 1: Toy model with analytical training dip
  - Linear classifier on anti-correlated features
  - Feature 1: high variance, anti-correlated with label → shortcut
  - Feature 2: low variance, correctly correlated → correct feature
  - ANALYTICAL DERIVATION: gradient flow gives closed-form w_i(t), from
    which the dip and ablation sign are derivable.
  - Also trains a 2-layer MLP on the same data to check for sign flip.
  Addresses: "limited novelty" + "'distributed' is not a mechanism"

PHASE 2: Logit lens across training
  - Per-layer logit diff at END token for IOI prompts
  - At dip floor vs maturity: shows WHERE in the network the S→IO
    prediction transition happens
  Addresses: "'distributed' is not a mechanism"

PHASE 3: PCA ablation at S2
  - Compute (IOI − ctrl) difference vectors at S2
  - PCA decomposition → top principal directions of "duplication signal"
  - Ablate top-K PCs and measure ΔLD
  - If top-1 PC captures majority of effect → cleaner than SAE
  Addresses: "SAE adds nothing"

PHASE 4: Multi-seed grokking (3 seeds at frac=0.3)
  - Verify Δ_test never positive across seeds
  Addresses: "grokking undercooked"

PHASE 5: OLMo training checkpoint sweep (best-effort)
  - Try to find OLMo-1B training checkpoints
  - Run IOI accuracy across training
  Addresses: "architecture diversity"

Output: results/emnlp_additional_controls.json
"""

import os
import gc
import json
import time
import sys
import traceback
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer

try:
    from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES, CANDIDATE_NAMES
except ImportError:
    sys.path.insert(0, "/workspace/ioi-sign-flip/src")
    from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES, CANDIDATE_NAMES


DEVICE = "cuda"
RETRAINED_REPO = "anonymous-research-sub/pythia-160m-retrained-seed42"
BASE_MODEL = "EleutherAI/pythia-160m-deduped"
SEED = 42
N_BOOTSTRAP = 10_000
RESULTS_PATH = "results/emnlp_additional_controls.json"


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


# =====================================================================
# PHASE 1: TOY MODEL WITH ANALYTICAL DIP
# =====================================================================

def phase1_toy_model():
    """
    ANALYTICAL DERIVATION
    ---------------------
    Data: x = (x_1, x_2, ..., x_d), y in {0, 1}.
    x_1 ~ N(0, sigma_1^2), sigma_1 = 10 (high variance, easy feature).
    x_2 ~ N(0, sigma_2^2), sigma_2 = 1  (low variance, correct feature).
    x_{3..d} ~ N(0, 1) (noise dimensions).
    y = 1{x_2 > 0} (label depends only on x_2).
    x_1 is shifted: x_1 -= rho * (2y - 1), so E[x_1|y=1] = -rho, E[x_1|y=0] = +rho.
    This makes Cov(x_1, y) ≈ -rho (anti-correlated shortcut).

    Linear model: f(x) = w^T x + b, trained with gradient descent on
    cross-entropy loss. For the dominant dynamics (ignoring cross-terms
    and the bias term):
        dw_i/dt ≈ Cov(x_i, y) - w_i * Var(x_i) * (correction)
    
    The key insight: w_1 converges on timescale O(1/sigma_1^2) = O(1/100),
    while w_2 converges on timescale O(1/sigma_2^2) = O(1). There is an
    intermediate phase where w_1 has converged (to a NEGATIVE value, since
    Cov(x_1, y) < 0) but w_2 hasn't yet learned. During this phase, the
    model predicts based on x_1, which is anti-correlated with y, giving
    BELOW-CHANCE accuracy. As w_2 catches up, accuracy recovers.

    The ablation effect (replacing x_1 with 0):
    - During dip: removing x_1 removes the wrong signal → acc INCREASES → Δ > 0
    - At convergence: w_2 dominates, removing x_1 has small positive effect
      (since w_1 is still negative). In a linear model, Δ stays ≥ 0.
    - In a nonlinear model (MLP), Δ can become negative if the model
      learns to USE x_1 constructively (e.g., for calibration/interaction).
    """
    log("=" * 60)
    log("PHASE 1: Toy model with analytical training dip")
    log("=" * 60)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Data generation.
    N_TRAIN = 5000
    N_TEST = 2000
    D = 10         # total input dimensions
    SIGMA_1 = 10.0 # high-variance shortcut dimension
    SIGMA_2 = 1.0  # low-variance correct dimension
    RHO = 3.0      # anti-correlation strength

    def make_data(n, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((n, D)).astype(np.float32)
        x[:, 0] *= SIGMA_1
        x[:, 1] *= SIGMA_2
        y = (x[:, 1] > 0).astype(np.float32)
        # Anti-correlate x_1 with y.
        x[:, 0] -= RHO * (2 * y - 1)
        return torch.tensor(x, device=DEVICE), torch.tensor(y, device=DEVICE)

    x_train, y_train = make_data(N_TRAIN, seed=SEED)
    x_test, y_test = make_data(N_TEST, seed=SEED + 1)

    # Verify anti-correlation.
    corr = float(np.corrcoef(x_train[:, 0].cpu().numpy(), y_train.cpu().numpy())[0, 1])
    log(f"  Corr(x_1, y) = {corr:.3f} (should be negative)")
    log(f"  Corr(x_2, y) = {float(np.corrcoef(x_train[:, 1].cpu().numpy(), y_train.cpu().numpy())[0, 1]):.3f} (should be positive)")

    N_STEPS = 2000
    EVAL_EVERY = 20

    results = {"analytical": {}, "linear": {}, "mlp": {}}
    results["analytical"] = {
        "sigma_1": SIGMA_1, "sigma_2": SIGMA_2, "rho": RHO, "d": D,
        "cov_x1_y": corr,
        "predicted_w1_converge_time": f"O(1/{SIGMA_1**2:.0f})",
        "predicted_w2_converge_time": f"O(1/{SIGMA_2**2:.0f})",
        "predicted_dip": "Yes, when w1 converged but w2 hasn't",
    }

    def evaluate(model, x, y, ablate_x1=False):
        with torch.no_grad():
            x_input = x.clone()
            if ablate_x1:
                x_input[:, 0] = 0.0  # replace with mean
            logits = model(x_input).squeeze(-1)
            preds = (logits > 0).float()
            return float((preds == y).float().mean().item())

    # --- LINEAR MODEL ---
    log("  Training linear model...")
    linear = nn.Linear(D, 1).to(DEVICE)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    opt_lin = torch.optim.SGD(linear.parameters(), lr=0.01)

    lin_trajectory = []
    for step in range(N_STEPS + 1):
        if step % EVAL_EVERY == 0:
            acc = evaluate(linear, x_test, y_test)
            acc_abl = evaluate(linear, x_test, y_test, ablate_x1=True)
            w1 = float(linear.weight.data[0, 0].item())
            w2 = float(linear.weight.data[0, 1].item())
            lin_trajectory.append({
                "step": step, "acc": acc, "acc_ablated": acc_abl,
                "delta": acc_abl - acc, "w1": w1, "w2": w2,
            })
            if step % 200 == 0:
                log(f"    step={step:>5}  acc={acc:.3f}  acc_abl={acc_abl:.3f}  "
                    f"Δ={acc_abl-acc:+.4f}  w1={w1:+.4f}  w2={w2:+.4f}")

        if step < N_STEPS:
            logits = linear(x_train).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y_train)
            opt_lin.zero_grad()
            loss.backward()
            opt_lin.step()

    min_acc_lin = min(lin_trajectory, key=lambda r: r["acc"])
    results["linear"] = {
        "trajectory": lin_trajectory,
        "min_acc": min_acc_lin["acc"],
        "min_acc_step": min_acc_lin["step"],
        "dip_detected": min_acc_lin["acc"] < 0.5,
        "max_positive_delta": max(r["delta"] for r in lin_trajectory),
        "final_delta": lin_trajectory[-1]["delta"],
    }
    log(f"  Linear: min_acc={min_acc_lin['acc']:.3f} @ step {min_acc_lin['step']}  "
        f"dip={'YES' if min_acc_lin['acc'] < 0.5 else 'NO'}  "
        f"max_Δ={max(r['delta'] for r in lin_trajectory):+.4f}")

    # --- 2-LAYER MLP ---
    log("  Training 2-layer MLP...")
    mlp = nn.Sequential(
        nn.Linear(D, 64), nn.ReLU(), nn.Linear(64, 1),
    ).to(DEVICE)
    # Small init to start near zero.
    for p in mlp.parameters():
        nn.init.normal_(p, std=0.01)
    opt_mlp = torch.optim.SGD(mlp.parameters(), lr=0.01)

    mlp_trajectory = []
    for step in range(N_STEPS + 1):
        if step % EVAL_EVERY == 0:
            acc = evaluate(mlp, x_test, y_test)
            acc_abl = evaluate(mlp, x_test, y_test, ablate_x1=True)
            mlp_trajectory.append({
                "step": step, "acc": acc, "acc_ablated": acc_abl,
                "delta": acc_abl - acc,
            })
            if step % 200 == 0:
                log(f"    step={step:>5}  acc={acc:.3f}  acc_abl={acc_abl:.3f}  "
                    f"Δ={acc_abl-acc:+.4f}")

        if step < N_STEPS:
            logits = mlp(x_train).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y_train)
            opt_mlp.zero_grad()
            loss.backward()
            opt_mlp.step()

    min_acc_mlp = min(mlp_trajectory, key=lambda r: r["acc"])
    results["mlp"] = {
        "trajectory": mlp_trajectory,
        "min_acc": min_acc_mlp["acc"],
        "min_acc_step": min_acc_mlp["step"],
        "dip_detected": min_acc_mlp["acc"] < 0.5,
        "max_positive_delta": max(r["delta"] for r in mlp_trajectory),
        "min_delta": min(r["delta"] for r in mlp_trajectory),
        "final_delta": mlp_trajectory[-1]["delta"],
    }
    log(f"  MLP: min_acc={min_acc_mlp['acc']:.3f} @ step {min_acc_mlp['step']}  "
        f"dip={'YES' if min_acc_mlp['acc'] < 0.5 else 'NO'}  "
        f"max_Δ={max(r['delta'] for r in mlp_trajectory):+.4f}  "
        f"min_Δ={min(r['delta'] for r in mlp_trajectory):+.4f}")

    return results


# =====================================================================
# PHASE 2: LOGIT LENS ACROSS TRAINING
# =====================================================================

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


def compute_logit_lens(model, n_prompts=100):
    """Compute per-layer logit diff (IO - S) at the END token across
    IOI prompts. Uses the residual stream at each layer projected
    through ln_final + unembed."""
    ds = IOIDataset(model=model, n_prompts=n_prompts, symmetric=True, seed=SEED)
    tokens = model.to_tokens(ds.prompts).to(DEVICE)
    io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
    s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)

    # Cache all residual streams.
    names = [f"blocks.{L}.hook_resid_post" for L in range(model.cfg.n_layers)]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=names)

    per_layer_ld = {}
    for L in range(model.cfg.n_layers):
        resid = cache[f"blocks.{L}.hook_resid_post"][:, -1, :]  # [B, d_model]
        # Apply final layer norm + unembed.
        normed = model.ln_final(resid)
        logits = normed @ model.W_U + model.b_U
        idx = torch.arange(logits.shape[0], device=DEVICE)
        ld = (logits[idx, io_ids] - logits[idx, s_ids]).detach().cpu().numpy()
        per_layer_ld[f"layer_{L}"] = {
            "mean_ld": float(ld.mean()),
            "std_ld": float(ld.std()),
        }

    del cache; torch.cuda.empty_cache()
    return per_layer_ld


def phase2_logit_lens():
    log("=" * 60)
    log("PHASE 2: Logit lens across training")
    log("=" * 60)

    LOGIT_STEPS = [1000, 2000, 3000, 5000, 143000]
    out = {"by_step": {}}

    for step in LOGIT_STEPS:
        log(f"  step {step}:")
        try:
            model = load_pythia_original(step)
        except Exception as e:
            log(f"    load failed: {e}")
            continue

        per_layer = compute_logit_lens(model)
        out["by_step"][f"step_{step}"] = per_layer

        # Print summary: at which layer does LD first become positive?
        flip_layer = None
        for L in range(model.cfg.n_layers):
            ld = per_layer[f"layer_{L}"]["mean_ld"]
            if ld > 0 and flip_layer is None:
                flip_layer = L
        ld_first = per_layer["layer_0"]["mean_ld"]
        ld_last = per_layer[f"layer_{model.cfg.n_layers - 1}"]["mean_ld"]
        log(f"    L0 LD={ld_first:+.3f}  L11 LD={ld_last:+.3f}  "
            f"flip_layer={flip_layer}")

        del model; torch.cuda.empty_cache(); gc.collect()

    return out


# =====================================================================
# PHASE 3: PCA ABLATION AT S2
# =====================================================================

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


def phase3_pca_ablation():
    log("=" * 60)
    log("PHASE 3: PCA ablation at S2")
    log("=" * 60)

    model = load_retrained(2000)
    single_ids = get_single_token_names(model.tokenizer)
    rng = np.random.default_rng(SEED + 1)

    LAYERS = [3, 4, 5]
    K_VALUES = [1, 3, 5, 10, 20]

    # Collect IOI and ctrl activations at S2 for difference vectors.
    log("  collecting activations...")
    ioi_acts = {L: [] for L in LAYERS}
    ctrl_acts = {L: [] for L in LAYERS}

    for tmpl in ALL_TEMPLATES[:10]:
        ds = IOIDataset(model=model, n_prompts=30, templates=[tmpl],
                        symmetric=True, seed=SEED)
        ioi_tokens = model.to_tokens(ds.prompts).to(DEVICE)
        n = ioi_tokens.shape[0]
        s2_pos = []
        for i in range(n):
            s2_pos.append(find_s2_position(ioi_tokens[i].cpu(), ds.s_token_ids[i]))

        ctrl_tokens = ioi_tokens.clone()
        for i in range(n):
            pool = [t for t in single_ids if t != int(ds.io_token_ids[i]) and t != int(ds.s_token_ids[i])]
            if pool and s2_pos[i] > 0:
                ctrl_tokens[i, s2_pos[i]] = int(rng.choice(pool))

        hook_names = [f"blocks.{L}.hook_resid_post" for L in LAYERS]
        for tokens_batch, acts_dict in [(ioi_tokens, ioi_acts), (ctrl_tokens, ctrl_acts)]:
            cache = {}
            def make_cap(name):
                def fn(value, hook):
                    cache[name] = value.detach()
                    return value
                return fn
            with torch.no_grad():
                model.run_with_hooks(
                    tokens_batch,
                    fwd_hooks=[(nm, make_cap(nm)) for nm in hook_names],
                )
            for L in LAYERS:
                for i in range(n):
                    if s2_pos[i] > 0:
                        acts_dict[L].append(
                            cache[f"blocks.{L}.hook_resid_post"][i, s2_pos[i]].cpu().float().numpy()
                        )
            del cache; torch.cuda.empty_cache()

    out = {"step": 2000, "by_layer": {}}

    for L in LAYERS:
        ioi_arr = np.array(ioi_acts[L])
        ctrl_arr = np.array(ctrl_acts[L])
        diff_arr = ioi_arr - ctrl_arr  # difference vectors
        log(f"  layer {L}: {diff_arr.shape[0]} difference vectors, d={diff_arr.shape[1]}")

        # PCA on difference vectors.
        pca = PCA(n_components=min(50, diff_arr.shape[0], diff_arr.shape[1]))
        pca.fit(diff_arr)
        explained = pca.explained_variance_ratio_
        log(f"    PCA top-5 explained variance: {explained[:5]}")

        # Now do ablation: for each K, project IOI activations onto top-K
        # PCs of the DIFFERENCE, zero those components, inject back.
        # The ablated activation = ioi_act - projection onto top-K PCs of diff.
        # This removes the "duplication direction" from the residual.

        io_ids_all = []
        s_ids_all = []
        s2_positions_all = []
        ioi_tokens_all = []

        # Re-collect for ablation evaluation (need tokens + positions).
        for tmpl in ALL_TEMPLATES[:10]:
            ds = IOIDataset(model=model, n_prompts=30, templates=[tmpl],
                            symmetric=True, seed=SEED)
            ioi_tokens = model.to_tokens(ds.prompts).to(DEVICE)
            n = ioi_tokens.shape[0]
            for i in range(n):
                s2_p = find_s2_position(ioi_tokens[i].cpu(), ds.s_token_ids[i])
                if s2_p > 0:
                    io_ids_all.append(int(ds.io_token_ids[i]))
                    s_ids_all.append(int(ds.s_token_ids[i]))
                    s2_positions_all.append(s2_p)
                    ioi_tokens_all.append(ioi_tokens[i])

        io_ids_t = torch.tensor(io_ids_all, device=DEVICE)
        s_ids_t = torch.tensor(s_ids_all, device=DEVICE)

        # Baseline LD (no intervention).
        base_lds = []
        for i, tok in enumerate(ioi_tokens_all):
            with torch.no_grad():
                logits = model(tok.unsqueeze(0))
            ld = logits[0, -1, io_ids_all[i]] - logits[0, -1, s_ids_all[i]]
            base_lds.append(float(ld.item()))
        base_mean = float(np.mean(base_lds))

        layer_results = {"base_ld": base_mean, "pca_explained_var": explained[:10].tolist(), "ablations": {}}

        # PCA components as torch tensors for hooking.
        components = torch.tensor(pca.components_, dtype=torch.float32, device=DEVICE)
        diff_mean = torch.tensor(pca.mean_, dtype=torch.float32, device=DEVICE)

        for K in K_VALUES:
            if K > components.shape[0]:
                continue
            top_K_components = components[:K]  # [K, d_model]

            # Hook: at S2, project activation onto top-K PCs of diff, remove.
            hook_name = f"blocks.{L}.hook_resid_post"
            abl_lds = []
            for idx_i in range(len(ioi_tokens_all)):
                tok = ioi_tokens_all[idx_i].unsqueeze(0)
                s2_p = s2_positions_all[idx_i]

                def make_pca_hook(pos, comps):
                    def fn(value, hook):
                        act = value[0, pos, :]
                        # Project onto top-K PCs and subtract.
                        centered = act - diff_mean
                        projections = centered @ comps.T  # [K]
                        reconstruction = projections @ comps  # [d_model]
                        value[0, pos, :] = act - reconstruction
                        return value
                    return fn

                with torch.no_grad():
                    logits = model.run_with_hooks(
                        tok,
                        fwd_hooks=[(hook_name, make_pca_hook(s2_p, top_K_components))],
                    )
                ld = logits[0, -1, io_ids_all[idx_i]] - logits[0, -1, s_ids_all[idx_i]]
                abl_lds.append(float(ld.item()))

            abl_mean = float(np.mean(abl_lds))
            deltas = np.array(abl_lds) - np.array(base_lds)
            delta_lo, delta_hi = bootstrap_ci(deltas)

            layer_results["ablations"][f"K_{K}"] = {
                "ablated_ld": abl_mean,
                "delta_mean": float(deltas.mean()),
                "delta_ci95": [delta_lo, delta_hi],
            }
            log(f"    K={K:>2}: LD={abl_mean:+.4f}  Δ={deltas.mean():+.4f} [{delta_lo:+.3f}, {delta_hi:+.3f}]")

        out["by_layer"][f"layer_{L}"] = layer_results

    del model; torch.cuda.empty_cache(); gc.collect()
    return out


# =====================================================================
# PHASE 4: MULTI-SEED GROKKING
# =====================================================================

# Import grokking model from the final_robustness script's architecture.
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_mlp):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model))
    def forward(self, x, mask=None):
        h = self.ln1(x); h, _ = self.attn(h, h, h, attn_mask=mask); x = x + h
        h = self.ln2(x); x = x + self.mlp(h); return x

class GrokTransformer(nn.Module):
    def __init__(self, p, d_model=128, n_heads=4, n_layers=2, d_mlp=512):
        super().__init__()
        self.p = p; self.embed = nn.Embedding(p+1, d_model)
        self.pos_embed = nn.Embedding(3, d_model)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_mlp) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, p, bias=False)
    def forward(self, a, b, ablate_a=False, mean_a_embed=None):
        B = a.shape[0]; device = a.device
        eq = torch.full((B,), self.p, dtype=torch.long, device=device)
        tokens = torch.stack([a, b, eq], dim=1)
        pos = torch.arange(3, device=device).unsqueeze(0).expand(B, -1)
        h = self.embed(tokens) + self.pos_embed(pos)
        if ablate_a and mean_a_embed is not None:
            h[:, 0, :] = mean_a_embed + self.pos_embed.weight[0]
        mask = torch.triu(torch.full((3, 3), float("-inf"), device=device), diagonal=1)
        for block in self.blocks: h = block(h, mask=mask)
        return self.unembed(self.ln_final(h[:, -1, :]))


def phase4_multiseed_grokking():
    log("=" * 60)
    log("PHASE 4: Multi-seed grokking (3 seeds, frac=0.3)")
    log("=" * 60)

    P = 113; FRAC = 0.3; N_STEPS = 50000; BATCH = 512
    EVAL_EVERY = 500; SEEDS = [42, 123, 456]

    rng = np.random.default_rng(0)
    pairs = [(a, b, (a+b) % P) for a in range(P) for b in range(P)]
    rng.shuffle(pairs)
    split = int(FRAC * len(pairs))
    train_p, test_p = pairs[:split], pairs[split:]
    train_a = torch.tensor([p[0] for p in train_p], device=DEVICE)
    train_b = torch.tensor([p[1] for p in train_p], device=DEVICE)
    train_t = torch.tensor([p[2] for p in train_p], device=DEVICE)
    test_a = torch.tensor([p[0] for p in test_p], device=DEVICE)
    test_b = torch.tensor([p[1] for p in test_p], device=DEVICE)
    test_t = torch.tensor([p[2] for p in test_p], device=DEVICE)

    out = {"seeds": {}}
    for seed in SEEDS:
        log(f"  seed={seed}")
        torch.manual_seed(seed)
        model = GrokTransformer(P).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0)
        max_delta = -999
        grok_step = None

        for step in range(N_STEPS + 1):
            if step % EVAL_EVERY == 0:
                model.eval()
                with torch.no_grad():
                    mean_a = model.embed.weight[:P].mean(dim=0)
                    test_logits = model(test_a, test_b)
                    test_acc = (test_logits.argmax(-1) == test_t).float().mean().item()
                    test_logits_abl = model(test_a, test_b, ablate_a=True, mean_a_embed=mean_a)
                    test_acc_abl = (test_logits_abl.argmax(-1) == test_t).float().mean().item()
                delta = test_acc_abl - test_acc
                max_delta = max(max_delta, delta)
                if test_acc > 0.95 and grok_step is None:
                    grok_step = step
                model.train()

            if step < N_STEPS:
                idx = torch.randint(0, len(train_p), (BATCH,), device=DEVICE)
                loss = F.cross_entropy(model(train_a[idx], train_b[idx]), train_t[idx])
                opt.zero_grad(); loss.backward(); opt.step()

        out["seeds"][f"seed_{seed}"] = {
            "grok_step": grok_step,
            "max_delta_test": float(max_delta),
            "positive_delta": max_delta > 0.01,
            "final_test_acc": float(test_acc),
        }
        log(f"    grok={grok_step}  max_Δ={max_delta:+.4f}  positive={'YES' if max_delta > 0.01 else 'NO'}")
        del model, opt; torch.cuda.empty_cache()

    out["verdict_any_positive"] = any(
        v["positive_delta"] for v in out["seeds"].values()
    )
    log(f"  VERDICT: any positive Δ across seeds? {out['verdict_any_positive']}")
    return out


# =====================================================================
# PHASE 5: OLMo CHECKPOINT SWEEP
# =====================================================================

def phase5_olmo():
    log("=" * 60)
    log("PHASE 5: OLMo training checkpoint sweep")
    log("=" * 60)

    # OLMo-1B-hf may have revisions. Try a few standard step formats.
    MODEL_NAME = "allenai/OLMo-1B-hf"
    REVISIONS_TO_TRY = [
        "step0-tokens0B", "step1000-tokens4B", "step5000-tokens21B",
        "step10000-tokens42B", "step50000-tokens210B", "step100000-tokens419B",
        "step250000-tokens1048B",
        # Also try plain step format:
        "step0", "step1000", "step5000", "step10000", "step50000",
        "main",
    ]

    out = {"model": MODEL_NAME, "checkpoints_tested": [], "results": {}}

    for rev in REVISIONS_TO_TRY:
        try:
            log(f"  Trying revision '{rev}'...")
            model = HookedTransformer.from_pretrained(
                MODEL_NAME, revision=rev, device=DEVICE,
                center_writing_weights=False, center_unembed=False, fold_ln=False,
            )
            # Quick IOI accuracy.
            ds = IOIDataset(model=model, n_prompts=100, symmetric=True, seed=SEED)
            tokens = model.to_tokens(ds.prompts).to(DEVICE)
            with torch.no_grad():
                logits = model(tokens)
            last = logits[:, -1, :]
            io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
            s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)
            idx = torch.arange(last.shape[0], device=DEVICE)
            ld = (last[idx, io_ids] - last[idx, s_ids]).cpu().numpy()
            acc = float((ld > 0).mean())
            out["results"][rev] = {"acc": acc, "mean_ld": float(ld.mean())}
            out["checkpoints_tested"].append(rev)
            log(f"    SUCCESS: acc={acc*100:.1f}%  LD={ld.mean():+.3f}")
            del model; torch.cuda.empty_cache(); gc.collect()
        except Exception as e:
            err_msg = str(e)[:100]
            if "revision" in err_msg.lower() or "not found" in err_msg.lower():
                pass  # silently skip non-existent revisions
            else:
                log(f"    failed: {err_msg}")
            continue

    log(f"  Found {len(out['checkpoints_tested'])} working checkpoints")
    return out


# =====================================================================
# MAIN
# =====================================================================

def main():
    os.makedirs("results", exist_ok=True)
    results = {"config": {"seed": SEED}}

    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH) as f:
                results = json.load(f)
            done = [k for k in results if k.startswith("phase")]
            log(f"Resuming. Already done: {done}")
        except Exception:
            pass

    phases = [
        ("phase1_toy_model", phase1_toy_model),
        ("phase2_logit_lens", phase2_logit_lens),
        ("phase3_pca_ablation", phase3_pca_ablation),
        ("phase4_multiseed_grokking", phase4_multiseed_grokking),
        ("phase5_olmo", phase5_olmo),
    ]

    t0 = time.time()
    for key, fn in phases:
        cached = results.get(key)
        if cached is not None and "error" not in cached:
            log(f"SKIP {key}: already done")
            continue
        if cached is not None and "error" in cached:
            log(f"RETRY {key}: previous error ({cached['error'][:60]})")
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
