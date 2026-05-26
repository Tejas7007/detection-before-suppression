"""
EMNLP Final Robustness Battery
================================

PHASE 1: Grokking sign-flip test (modular addition)
  - Train 2-layer transformer from scratch on (a+b) mod 113
  - Standard grokking setup: weight decay 1.0, long training
  - Track train/test accuracy across 50K steps
  - At each checkpoint: ablate input a (mean embedding replacement)
  - Measure ablation effect on test set across training
  - PREDICTION: reliance on a transitions from ~0 (pre-grok) to
    strongly negative (post-grok). Connects to broader grokking
    literature.

PHASE 2: Template sensitivity (greater-than + SVA)
  - 3 template variants per task on original Pythia-160M
  - Checkpoints: step 1000 (GT dip), step 5000 (recovered), step 143000
  - Verify dip is NOT template-specific

PHASE 3: Layer sensitivity for causal interventions
  - IOI at step 2000 on retrained Pythia-160M
  - Patch at L0-2, L3-5, L6-8, L9-11 separately
  - Quantify where the S2 patching effect concentrates

PHASE 4: OLMo architecture check (best-effort)
  - Try loading OLMo-1B via TransformerLens
  - If supported: run IOI accuracy at 3-4 checkpoints
  - If not: log gracefully and skip

Output: results/emnlp_final_robustness.json
Runtime: ~2-3 hours on A100.
"""

import os
import gc
import json
import time
import sys
import math
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
RESULTS_PATH = "results/emnlp_final_robustness.json"


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


# ====================================================================
# PHASE 1: GROKKING
# ====================================================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_mlp):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp),
            nn.GELU(),
            nn.Linear(d_mlp, d_model),
        )

    def forward(self, x, mask=None):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + h
        h = self.ln2(x)
        x = x + self.mlp(h)
        return x


class GrokTransformer(nn.Module):
    """Tiny transformer for modular addition grokking experiments.
    Input: two integers a, b ∈ {0..p-1}.
    Output: logits over p classes at position 2 (the '=' position).
    """
    def __init__(self, p, d_model=128, n_heads=4, n_layers=2, d_mlp=512):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.embed = nn.Embedding(p + 1, d_model)   # p numbers + 1 '=' token
        self.pos_embed = nn.Embedding(3, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_mlp)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, p, bias=False)

    def forward(self, a, b, ablate_a=False, mean_a_embed=None):
        """a, b: [batch] long tensors in [0, p). Returns [batch, p] logits."""
        B = a.shape[0]
        device = a.device
        eq = torch.full((B,), self.p, dtype=torch.long, device=device)
        tokens = torch.stack([a, b, eq], dim=1)        # [B, 3]
        pos = torch.arange(3, device=device).unsqueeze(0).expand(B, -1)

        h = self.embed(tokens) + self.pos_embed(pos)

        # Optional: ablate input a (position 0) with mean embedding.
        if ablate_a and mean_a_embed is not None:
            h[:, 0, :] = mean_a_embed + self.pos_embed.weight[0]

        # Causal attention mask.
        mask = torch.triu(torch.full((3, 3), float("-inf"), device=device), diagonal=1)
        for block in self.blocks:
            h = block(h, mask=mask)
        h = self.ln_final(h)
        return self.unembed(h[:, -1, :])               # predict at last position


def grok_evaluate(model, data_a, data_b, data_t, ablate_a=False, mean_a_embed=None):
    """Evaluate accuracy. data_a, data_b, data_t are tensors."""
    with torch.no_grad():
        logits = model(data_a, data_b, ablate_a=ablate_a, mean_a_embed=mean_a_embed)
    return (logits.argmax(-1) == data_t).float().mean().item()


def phase1_grokking():
    log("=" * 60)
    log("PHASE 1: Grokking sign-flip test (modular addition mod 113)")
    log("  Sweeping training fractions: 0.3 (clean grokking),")
    log("  0.5 (moderate), 0.7 (immediate generalization)")
    log("=" * 60)

    P = 113
    TRAIN_FRACS = [0.3, 0.5, 0.7]
    D_MODEL = 128
    N_HEADS = 4
    N_LAYERS = 2
    D_MLP = 512
    LR = 1e-3
    WD = 1.0
    BATCH = 512
    N_STEPS = 50000
    EVAL_EVERY = 500
    CKPT_EVAL_STEPS = set(range(0, N_STEPS + 1, EVAL_EVERY))

    # All (a, b, (a+b)%p) pairs, shuffled once.
    rng = np.random.default_rng(SEED)
    pairs = [(a, b, (a + b) % P) for a in range(P) for b in range(P)]
    rng.shuffle(pairs)

    out = {"config": {
        "p": P, "d_model": D_MODEL, "n_layers": N_LAYERS,
        "lr": LR, "weight_decay": WD, "train_fracs": TRAIN_FRACS,
        "n_steps": N_STEPS,
    }, "by_fraction": {}}

    for frac in TRAIN_FRACS:
        log(f"\n  --- train_frac={frac} ---")
        split = int(frac * len(pairs))
        train_pairs = pairs[:split]
        test_pairs = pairs[split:]

        train_a = torch.tensor([p[0] for p in train_pairs], device=DEVICE)
        train_b = torch.tensor([p[1] for p in train_pairs], device=DEVICE)
        train_t = torch.tensor([p[2] for p in train_pairs], device=DEVICE)
        test_a = torch.tensor([p[0] for p in test_pairs], device=DEVICE)
        test_b = torch.tensor([p[1] for p in test_pairs], device=DEVICE)
        test_t = torch.tensor([p[2] for p in test_pairs], device=DEVICE)

        log(f"  Dataset: {len(train_pairs)} train, {len(test_pairs)} test (p={P})")

        torch.manual_seed(SEED)
        model = GrokTransformer(P, D_MODEL, N_HEADS, N_LAYERS, D_MLP).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

        trajectory = []
        t0 = time.time()

        for step in range(N_STEPS + 1):
            if step in CKPT_EVAL_STEPS or step == N_STEPS:
                model.eval()
                with torch.no_grad():
                    mean_a_embed = model.embed.weight[:P].mean(dim=0)

                train_acc = grok_evaluate(model, train_a, train_b, train_t)
                test_acc = grok_evaluate(model, test_a, test_b, test_t)
                train_acc_abl = grok_evaluate(model, train_a, train_b, train_t,
                                              ablate_a=True, mean_a_embed=mean_a_embed)
                test_acc_abl = grok_evaluate(model, test_a, test_b, test_t,
                                             ablate_a=True, mean_a_embed=mean_a_embed)

                delta_train = train_acc_abl - train_acc
                delta_test = test_acc_abl - test_acc

                trajectory.append({
                    "step": step,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "train_acc_ablated": train_acc_abl,
                    "test_acc_ablated": test_acc_abl,
                    "delta_train": delta_train,
                    "delta_test": delta_test,
                })

                if step % 5000 == 0 or (len(trajectory) > 1 and abs(test_acc - trajectory[-2]["test_acc"]) > 0.05):
                    log(
                        f"    step={step:>6}  train={train_acc:.3f}  test={test_acc:.3f}  "
                        f"Δ_test={delta_test:+.4f}  Δ_train={delta_train:+.4f}  "
                        f"({time.time()-t0:.0f}s)"
                    )
                model.train()

            if step == N_STEPS:
                break

            idx = torch.randint(0, len(train_pairs), (BATCH,), device=DEVICE)
            logits = model(train_a[idx], train_b[idx])
            loss = F.cross_entropy(logits, train_t[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Grokking point: first step where test_acc > 0.95.
        grok_step = None
        for r in trajectory:
            if r["test_acc"] > 0.95:
                grok_step = r["step"]
                break

        min_test = min(trajectory, key=lambda r: r["test_acc"])

        # Check if Δ_test ever goes positive (would indicate shortcut misuse).
        max_delta_test = max(r["delta_test"] for r in trajectory)
        any_positive = max_delta_test > 0.01  # threshold above noise

        frac_out = {
            "train_frac": frac,
            "n_train": len(train_pairs),
            "n_test": len(test_pairs),
            "grokking_step": grok_step,
            "min_test_acc": min_test["test_acc"],
            "min_test_step": min_test["step"],
            "max_delta_test": max_delta_test,
            "any_positive_delta": any_positive,
            "trajectory": trajectory,
        }
        out["by_fraction"][f"frac_{frac}"] = frac_out

        log(
            f"  frac={frac}: grok_step={grok_step}  min_test={min_test['test_acc']:.3f} "
            f"@ step {min_test['step']}  max_Δ_test={max_delta_test:+.4f}  "
            f"positive_Δ={'YES' if any_positive else 'NO'}"
        )

        del model, opt
        torch.cuda.empty_cache()

    return out


# ====================================================================
# PHASE 2: TEMPLATE SENSITIVITY
# ====================================================================

def get_two_digit_completion_tokens(tokenizer):
    mapping = {}
    for d in range(100):
        s = f"{d:02d}"
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            mapping[d] = ids[0]
    return mapping


def eval_gt_with_template(model, template_fn, n_prompts=300, seed=SEED):
    """Evaluate greater-than accuracy using a given template function."""
    rng = np.random.default_rng(seed)
    digit_tokens = get_two_digit_completion_tokens(model.tokenizer)
    valid_digits = sorted(digit_tokens.keys())
    token_ids = torch.tensor([digit_tokens[d] for d in valid_digits], device=DEVICE)

    diffs, correct = [], []
    for _ in range(n_prompts):
        y = int(rng.integers(3, 97))
        prompt = template_fn(1700 + y, rng)
        tokens = model.to_tokens(prompt).to(DEVICE)
        with torch.no_grad():
            logits = model(tokens)[0, -1, :]
        probs = F.softmax(logits.float(), dim=-1)
        dp = probs[token_ids].cpu().numpy()
        gmask = np.array([d > y for d in valid_digits])
        lmask = np.array([d <= y for d in valid_digits])
        pg = float(dp[gmask].sum())
        pl = float(dp[lmask].sum())
        diffs.append(pg - pl)
        correct.append(1.0 if pg > pl else 0.0)
    return float(np.mean(correct)), float(np.mean(diffs))


GT_TEMPLATE_FNS = {
    "original": lambda yr, rng: (
        f"The {rng.choice(['war','battle','dispute','conflict'])} "
        f"{rng.choice(['lasted','ran','continued'])} from the year {yr} to the year 17"
    ),
    "between": lambda yr, rng: (
        f"The {rng.choice(['war','battle','dispute','conflict'])} "
        f"took place between the year {yr} and the year 17"
    ),
    "started": lambda yr, rng: (
        f"The {rng.choice(['war','battle','dispute','conflict'])} "
        f"started in the year {yr} and ended in 17"
    ),
}


SVA_SINGULAR = ["boy", "girl", "dog", "cat", "doctor", "writer", "teacher", "actor"]
SVA_PLURAL = ["boys", "girls", "dogs", "cats", "doctors", "writers", "teachers", "actors"]
SVA_VERB_PAIRS_RAW = [("is", "are"), ("was", "were"), ("has", "have")]


def eval_sva_with_template(model, template_fn, n_prompts=200, seed=SEED):
    """Evaluate SVA accuracy using a given template function."""
    rng = np.random.default_rng(seed)
    pairs = list(zip(SVA_SINGULAR, SVA_PLURAL))

    # Filter to single-token verb pairs.
    verb_pairs = []
    for s, p in SVA_VERB_PAIRS_RAW:
        si = model.tokenizer.encode(" " + s, add_special_tokens=False)
        pi = model.tokenizer.encode(" " + p, add_special_tokens=False)
        if len(si) == 1 and len(pi) == 1:
            verb_pairs.append({"sing": (s, si[0]), "plur": (p, pi[0])})
    if not verb_pairs:
        return float("nan"), float("nan")

    correct_list = []
    log_ratios = []
    for _ in range(n_prompts):
        subj_sing = bool(rng.integers(0, 2))
        i = int(rng.integers(0, len(pairs)))
        j = int(rng.integers(0, len(pairs)))
        vp = verb_pairs[int(rng.integers(0, len(verb_pairs)))]
        if subj_sing:
            subj, attr = pairs[i][0], pairs[j][1]
            c_id, a_id = vp["sing"][1], vp["plur"][1]
        else:
            subj, attr = pairs[i][1], pairs[j][0]
            c_id, a_id = vp["plur"][1], vp["sing"][1]

        prompt = template_fn(subj, attr, rng)
        tokens = model.to_tokens(prompt).to(DEVICE)
        with torch.no_grad():
            logits = model(tokens)[0, -1, :]
        probs = F.softmax(logits.float(), dim=-1)
        pc = float(probs[c_id].item())
        pa = float(probs[a_id].item())
        correct_list.append(1.0 if pc > pa else 0.0)
        log_ratios.append(float(np.log(max(pc, 1e-12)) - np.log(max(pa, 1e-12))))

    return float(np.mean(correct_list)), float(np.mean(log_ratios))


SVA_TEMPLATE_FNS = {
    "pp_attachment": lambda subj, attr, rng: (
        f"The {subj} {rng.choice(['near','beside','with','behind'])} the {attr}"
    ),
    "relative_clause": lambda subj, attr, rng: (
        f"The {subj} that the {attr} {rng.choice(['saw','liked','watched','followed'])}"
    ),
    "double_pp": lambda subj, attr, rng: (
        f"The {subj} {rng.choice(['near','beside'])} the "
        f"{rng.choice(['old','small','new','big'])} {attr}"
    ),
}


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


def phase2_template_sensitivity():
    log("=" * 60)
    log("PHASE 2: Template sensitivity (GT + SVA)")
    log("=" * 60)

    TEMPLATE_STEPS = [1000, 5000, 143000]
    out = {"greater_than": {}, "sva": {}}

    for step in TEMPLATE_STEPS:
        model = load_pythia_original(step)
        log(f"  step {step}:")

        # Greater-than templates.
        gt_results = {}
        for tname, tfn in GT_TEMPLATE_FNS.items():
            acc, diff = eval_gt_with_template(model, tfn)
            gt_results[tname] = {"acc": acc, "mean_diff": diff}
            marker = " <-- DIP" if acc < 0.5 else ""
            log(f"    GT '{tname}': acc={acc*100:.1f}%  diff={diff:+.4f}{marker}")
        out["greater_than"][f"step_{step}"] = gt_results

        # SVA templates.
        sva_results = {}
        for tname, tfn in SVA_TEMPLATE_FNS.items():
            acc, lr = eval_sva_with_template(model, tfn)
            sva_results[tname] = {"acc": acc, "log_ratio": lr}
            marker = " <-- DIP" if acc < 0.5 else ""
            log(f"    SVA '{tname}': acc={acc*100:.1f}%  logR={lr:+.3f}{marker}")
        out["sva"][f"step_{step}"] = sva_results

        del model; torch.cuda.empty_cache(); gc.collect()

    return out


# ====================================================================
# PHASE 3: LAYER SENSITIVITY
# ====================================================================

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


def run_ioi_patching(model, layer_range, n_templates=10, ppt=30):
    """IOI S2 patching at the specified layer range. Returns mean ΔLD."""
    rng = np.random.default_rng(SEED + 1)
    single_ids = get_single_token_names(model.tokenizer)
    base_lds, patched_lds = [], []

    for tmpl in ALL_TEMPLATES[:n_templates]:
        ds = IOIDataset(model=model, n_prompts=ppt, templates=[tmpl],
                        symmetric=True, seed=SEED)
        ioi_tokens = model.to_tokens(ds.prompts).to(DEVICE)
        n = ioi_tokens.shape[0]
        s2_pos = []
        for i in range(n):
            s2_pos.append(find_s2_position(ioi_tokens[i].cpu(), ds.s_token_ids[i]))
        s2_pos = torch.tensor(s2_pos, dtype=torch.long, device=DEVICE)

        ctrl_tokens = ioi_tokens.clone()
        for i in range(n):
            pool = [t for t in single_ids if t != int(ds.io_token_ids[i]) and t != int(ds.s_token_ids[i])]
            if pool and s2_pos[i] > 0:
                ctrl_tokens[i, s2_pos[i]] = int(rng.choice(pool))

        io_ids = torch.tensor(ds.io_token_ids, dtype=torch.long, device=DEVICE)
        s_ids = torch.tensor(ds.s_token_ids, dtype=torch.long, device=DEVICE)
        names = [f"blocks.{L}.hook_resid_post" for L in layer_range]
        donor = {}

        def make_cap(name):
            def fn(value, hook):
                donor[name] = value.detach()
                return value
            return fn

        with torch.no_grad():
            model.run_with_hooks(ctrl_tokens, fwd_hooks=[(nm, make_cap(nm)) for nm in names])

        with torch.no_grad():
            bl = model(ioi_tokens)
        idx = torch.arange(n, device=DEVICE)
        base_lds.extend((bl[:, -1, :][idx, io_ids] - bl[:, -1, :][idx, s_ids]).cpu().tolist())

        def make_patch(name):
            d = donor[name]
            def fn(value, hook):
                for i in range(value.shape[0]):
                    p = int(s2_pos[i].item())
                    if p > 0:
                        value[i, p, :] = d[i, p, :]
                return value
            return fn

        hooks = [(nm, make_patch(nm)) for nm in names]
        with torch.no_grad():
            pl = model.run_with_hooks(ioi_tokens, fwd_hooks=hooks)
        patched_lds.extend((pl[:, -1, :][idx, io_ids] - pl[:, -1, :][idx, s_ids]).cpu().tolist())
        del donor; torch.cuda.empty_cache()

    base_arr = np.asarray(base_lds)
    patched_arr = np.asarray(patched_lds)
    deltas = patched_arr - base_arr
    lo, hi = bootstrap_ci(deltas)
    return {
        "n": int(len(deltas)),
        "base_ld_mean": float(base_arr.mean()),
        "delta_ld_mean": float(deltas.mean()),
        "delta_ld_ci95": [lo, hi],
    }


def phase3_layer_sensitivity():
    log("=" * 60)
    log("PHASE 3: Layer sensitivity for IOI S2 patching")
    log("=" * 60)

    model = load_retrained(2000)
    LAYER_RANGES = {
        "L0-2": [0, 1, 2],
        "L3-5": [3, 4, 5],
        "L6-8": [6, 7, 8],
        "L9-11": [9, 10, 11],
    }
    out = {"step": 2000, "by_range": {}}

    for name, layers in LAYER_RANGES.items():
        r = run_ioi_patching(model, layers)
        out["by_range"][name] = r
        log(
            f"  {name}: ΔLD={r['delta_ld_mean']:+.4f} "
            f"[{r['delta_ld_ci95'][0]:+.3f}, {r['delta_ld_ci95'][1]:+.3f}]"
        )

    del model; torch.cuda.empty_cache(); gc.collect()
    return out


# ====================================================================
# PHASE 4: OLMo ARCHITECTURE CHECK
# ====================================================================

def phase4_olmo():
    log("=" * 60)
    log("PHASE 4: OLMo architecture check (best-effort)")
    log("=" * 60)

    # Try to load OLMo-1B. TransformerLens may or may not support it.
    # We try multiple model name formats.
    OLMO_CANDIDATES = [
        "allenai/OLMo-1B-hf",
        "allenai/OLMo-1B",
    ]
    OLMO_STEPS = [0, 10000, 100000]

    out = {"attempted": True, "success": False, "results": {}}

    for model_name in OLMO_CANDIDATES:
        log(f"  Trying {model_name}...")
        try:
            model = HookedTransformer.from_pretrained(
                model_name, device=DEVICE,
                center_writing_weights=False,
                center_unembed=False,
                fold_ln=False,
            )
            log(f"  SUCCESS: loaded {model_name}")
            out["model_name"] = model_name
            out["success"] = True
            out["n_layers"] = model.cfg.n_layers
            out["d_model"] = model.cfg.d_model

            # Quick IOI accuracy check.
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
            out["results"]["default"] = {
                "acc": acc,
                "mean_ld": float(ld.mean()),
            }
            log(f"    IOI acc={acc*100:.1f}%  mean_LD={ld.mean():+.4f}")

            del model; torch.cuda.empty_cache()
            break

        except Exception as e:
            log(f"  FAILED: {e}")
            out["results"][model_name] = {"error": str(e)[:200]}
            continue

    if not out["success"]:
        log("  OLMo not available via TransformerLens; skipping.")
        log("  (This is expected — not all architectures are supported.)")

    return out


# ====================================================================
# MAIN
# ====================================================================

def main():
    os.makedirs("results", exist_ok=True)
    results = {"config": {"seed": SEED, "n_bootstrap": N_BOOTSTRAP}}

    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH) as f:
                results = json.load(f)
            done = [k for k in results if k.startswith("phase")]
            log(f"Resuming. Already done: {done}")
        except Exception as e:
            log(f"Could not resume: {e}")

    phases = [
        ("phase1_grokking", phase1_grokking),
        ("phase2_template_sensitivity", phase2_template_sensitivity),
        ("phase3_layer_sensitivity", phase3_layer_sensitivity),
        ("phase4_olmo", phase4_olmo),
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
