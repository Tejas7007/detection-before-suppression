"""
EMNLP Robustness Battery
========================
A single script with 5 independent phases. Each phase saves results
incrementally to its own section of one JSON file. If a phase crashes,
the others still run. On restart, completed phases are skipped.

PHASE 1: Held-out probe generalization (kills template/name leakage)
PHASE 2: Position specificity scan at step 2000 (kills "any perturbation")
PHASE 3: L4H0 trajectory across 51 dense checkpoints (mechanism overlay)
PHASE 4: L4H0 zero-ablation validation at step 5000 (methodology)
PHASE 5: Multi-head joint patching at step 2000 (distributed-vs-compositional)

Output: results/emnlp_robustness_battery.json
Log:    results/emnlp_robustness_battery_log.txt
Runtime: ~45-60 minutes on A100.
"""

import os
import gc
import json
import time
import sys
import traceback
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    from circuitscaling.datasets import (
        IOIDataset, ALL_TEMPLATES, BABA_TEMPLATES, ABBA_TEMPLATES,
        CANDIDATE_NAMES,
    )
except ImportError:
    sys.path.insert(0, "/workspace/ioi-sign-flip/src")
    from circuitscaling.datasets import (
        IOIDataset, ALL_TEMPLATES, BABA_TEMPLATES, ABBA_TEMPLATES,
        CANDIDATE_NAMES,
    )


# ----------------------------- Config -----------------------------

DEVICE = "cuda"
RETRAINED_REPO = "anonymous-research-sub/pythia-160m-retrained-seed42"
BASE_MODEL = "EleutherAI/pythia-160m-deduped"
SEED = 42
N_BOOTSTRAP = 10_000

# Phase 1
PROBE_STEPS = [0, 2000, 143000]
PROBE_LAYERS = [1, 5]  # best layers from prior analysis
PROBE_PPT = 20         # per template, per condition (IOI/ctrl)

# Phase 2
POS_SCAN_STEP = 2000
PATCH_LAYERS = [3, 4, 5]
POS_TEMPLATES = ALL_TEMPLATES[:10]
POS_PPT = 30

# Phase 3
L4H0_LAYER = 4
L4H0_HEAD = 0
L8H9_LAYER = 8
L8H9_HEAD = 9
DENSE_STEPS = list(range(1000, 3001, 50)) + list(range(3200, 5001, 200))  # 51
TRAJ_TEMPLATES = ALL_TEMPLATES[:10]
TRAJ_PPT = 30

# Phase 4
ZERO_ABL_STEP = 5000
ZERO_ABL_TARGETS = [(4, 0), (8, 9)]  # L4H0 (our claim) + L8H9 (prior claim)

# Phase 5
MULTIHEAD_STEP = 2000
MULTIHEAD_K_VALUES = [1, 3, 5, 10]
HEAD_PATCHING_JSON = "results/emnlp_head_patching.json"

RESULTS_PATH = "results/emnlp_robustness_battery.json"


# --------------------------- Utilities ----------------------------

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_retrained(step, enable_attn_result=False):
    hf = AutoModelForCausalLM.from_pretrained(
        RETRAINED_REPO, subfolder=f"step_{step}", torch_dtype=torch.float32,
    )
    model = HookedTransformer.from_pretrained(
        BASE_MODEL,
        hf_model=hf,
        device=DEVICE,
        center_writing_weights=True,
        center_unembed=True,
        fold_ln=True,
    )
    del hf
    if enable_attn_result:
        model.set_use_attn_result(True)
    torch.cuda.empty_cache()
    return model


def load_pythia_original(step, enable_attn_result=False):
    """Original Pythia-160M-deduped revisions, which exist at all
    standard Pythia checkpoints including step 0 and step 143000 with
    proper configs. Use this for phases that need pre-training-end
    checkpoints (e.g., the probe analysis)."""
    hf = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, revision=f"step{step}", torch_dtype=torch.float32,
    )
    model = HookedTransformer.from_pretrained(
        BASE_MODEL,
        hf_model=hf,
        device=DEVICE,
        center_writing_weights=True,
        center_unembed=True,
        fold_ln=True,
    )
    del hf
    if enable_attn_result:
        model.set_use_attn_result(True)
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


def find_token_positions(token_row, io_token_id, s_token_id):
    """Return dict of {BOS, S1, S2, IO, END} positions for one prompt."""
    s_count = 0
    s1, s2, io_pos = -1, -1, -1
    for j in range(1, token_row.shape[0]):
        tok = int(token_row[j].item())
        if tok == int(s_token_id):
            s_count += 1
            if s_count == 1:
                s1 = j
            elif s_count == 2:
                s2 = j
        if tok == int(io_token_id) and io_pos == -1:
            io_pos = j
    return {
        "BOS": 0,
        "S1": s1,
        "S2": s2,
        "IO": io_pos,
        "END": token_row.shape[0] - 1,
    }


def get_single_token_names(tokenizer):
    ids = []
    name_to_id = {}
    for name in CANDIDATE_NAMES:
        toks = tokenizer.encode(" " + name, add_special_tokens=False)
        if len(toks) == 1:
            ids.append(int(toks[0]))
            name_to_id[name] = int(toks[0])
    return ids, name_to_id


def build_control(model, ds, single_name_ids, rng, position="S2"):
    """Tokenize IOI prompts and build matched controls by replacing one
    token. position="S2" matches the main IOI protocol; other
    positions support the position-specificity scan."""
    ioi_tokens = model.to_tokens(ds.prompts).to(DEVICE)
    n = ioi_tokens.shape[0]

    positions = []
    for i in range(n):
        pos_dict = find_token_positions(
            ioi_tokens[i].cpu(), ds.io_token_ids[i], ds.s_token_ids[i],
        )
        if position == "MID":
            # Midpoint between S1 and END as a template-relative
            # "neutral" content-token position. Falls back to END-3 if
            # S1 wasn't found.
            s1, end = pos_dict["S1"], pos_dict["END"]
            p = (s1 + end) // 2 if s1 > 0 else end - 3
        elif position in pos_dict:
            p = pos_dict[position]
        else:
            p = -1
        positions.append(p)
    positions = torch.tensor(positions, dtype=torch.long, device=DEVICE)

    ctrl_tokens = ioi_tokens.clone()
    if position == "S2":
        # Replace S2 with a third single-token name.
        for i in range(n):
            io_id = int(ds.io_token_ids[i])
            s_id = int(ds.s_token_ids[i])
            pool = [t for t in single_name_ids if t != io_id and t != s_id]
            if pool:
                p = int(positions[i].item())
                if p > 0:
                    ctrl_tokens[i, p] = int(rng.choice(pool))
    elif position in ("S1", "IO"):
        # Replace S1 or IO with a third name (matches the structure of
        # the S2 control: a single-token name substitution).
        for i in range(n):
            io_id = int(ds.io_token_ids[i])
            s_id = int(ds.s_token_ids[i])
            pool = [t for t in single_name_ids if t != io_id and t != s_id]
            if pool:
                p = int(positions[i].item())
                if p > 0:
                    ctrl_tokens[i, p] = int(rng.choice(pool))
    elif position == "BOS":
        # Replacing BOS with a random name would be nonsensical. Instead,
        # we replace token-1 (the first content token) with a random
        # other content token from the prompt's vocabulary. This gives
        # a defensible "any-non-S2-position" control.
        for i in range(n):
            p = 1
            other_tokens = [int(t.item()) for t in ioi_tokens[i, 2:].cpu()]
            if other_tokens:
                ctrl_tokens[i, p] = int(rng.choice(other_tokens))
        positions = torch.full((n,), 1, dtype=torch.long, device=DEVICE)
    elif position == "END":
        # Replace END token (last token of prompt, usually " to") with a
        # random other content token. Note: this changes the model's
        # final position, so we won't measure logit diff at END after
        # patching here — END patching only makes sense as residual
        # patching, since changing the token itself changes the LD by
        # construction. Use the SAME ioi END but patch its residual.
        # We don't perturb the token; positions vector already points
        # to END.
        pass
    elif position == "MID":
        # Position is already set at the top of the function. Replace
        # the token at that position with a different content token
        # from the same prompt.
        for i in range(n):
            p = int(positions[i].item())
            if p > 0 and p < ioi_tokens.shape[1] - 1:
                other_tokens = [
                    int(ioi_tokens[i, j].item())
                    for j in range(2, ioi_tokens.shape[1] - 1)
                    if j != p
                ]
                if other_tokens:
                    ctrl_tokens[i, p] = int(rng.choice(other_tokens))

    return ioi_tokens, ctrl_tokens, positions


def logit_diff_per_prompt(logits, io_ids, s_ids):
    last = logits[:, -1, :]
    idx = torch.arange(last.shape[0], device=last.device)
    return last[idx, io_ids] - last[idx, s_ids]


def bootstrap_ci(values, n_resamples=N_BOOTSTRAP, alpha=0.05, seed=SEED):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, arr.shape[0], size=(n_resamples, arr.shape[0]))
    means = arr[idx].mean(axis=1)
    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def save_results(results):
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)


# ====================================================================
# PHASE 1: HELD-OUT PROBE GENERALIZATION
# ====================================================================

def collect_probe_activations(model, names_subset, templates_subset, layer,
                              single_name_ids, rng):
    """Generate prompts using only the given name and template subsets,
    collect S2 activations for IOI and control conditions at one layer.
    Returns (X, y) for sklearn."""
    acts = []
    labels = []
    for tmpl in templates_subset:
        ds = IOIDataset(
            model=model, n_prompts=PROBE_PPT, templates=[tmpl],
            names=names_subset, symmetric=True, seed=SEED,
        )
        ioi_tokens = model.to_tokens(ds.prompts).to(DEVICE)
        n = ioi_tokens.shape[0]
        s2_positions = []
        for i in range(n):
            s2_positions.append(find_s2_position(
                ioi_tokens[i].cpu(), ds.s_token_ids[i],
            ))

        # Build control by replacing S2 with a third name.
        ctrl_tokens = ioi_tokens.clone()
        for i in range(n):
            io_id = int(ds.io_token_ids[i])
            s_id = int(ds.s_token_ids[i])
            pool = [t for t in single_name_ids if t != io_id and t != s_id]
            if pool and s2_positions[i] > 0:
                ctrl_tokens[i, s2_positions[i]] = int(rng.choice(pool))

        hook_name = f"blocks.{layer}.hook_resid_post"
        with torch.no_grad():
            _, cache_ioi = model.run_with_cache(
                ioi_tokens, names_filter=hook_name,
            )
            _, cache_ctrl = model.run_with_cache(
                ctrl_tokens, names_filter=hook_name,
            )

        for i in range(n):
            if s2_positions[i] > 0:
                acts.append(cache_ioi[hook_name][i, s2_positions[i], :].cpu().float().numpy())
                labels.append(1)
                acts.append(cache_ctrl[hook_name][i, s2_positions[i], :].cpu().float().numpy())
                labels.append(0)

        del cache_ioi, cache_ctrl
        torch.cuda.empty_cache()

    return np.array(acts), np.array(labels)


def phase1_heldout_probes():
    """Train probes on a name/template subset; evaluate on held-out
    names AND held-out templates. Test both splits (symmetric) so we
    can't be accused of cherry-picking the easy direction."""
    log("=" * 60)
    log("PHASE 1: Held-out probe generalization")
    log("=" * 60)

    out = {"steps": {}}

    for step in PROBE_STEPS:
        log(f"-- step {step} --")
        # Use ORIGINAL Pythia revisions: the retrained repo lacks step
        # 143000 and has an incomplete config at step 0. Original Pythia
        # has proper configs at every standard step revision.
        model = load_pythia_original(step)
        single_name_ids, name_to_id = get_single_token_names(model.tokenizer)
        single_names = list(name_to_id.keys())
        log(f"   {len(single_names)} single-token names available")

        # 50/50 split.
        random.Random(SEED).shuffle(single_names)
        half = len(single_names) // 2
        names_A = single_names[:half]
        names_B = single_names[half:]

        templates_A = ALL_TEMPLATES[:15]
        templates_B = ALL_TEMPLATES[15:30]

        rng = np.random.default_rng(SEED + 1)
        step_out = {"by_layer": {}}

        for layer in PROBE_LAYERS:
            log(f"   layer {layer}: collecting activations")
            X_A, y_A = collect_probe_activations(
                model, names_A, templates_A, layer, single_name_ids, rng,
            )
            X_B, y_B = collect_probe_activations(
                model, names_B, templates_B, layer, single_name_ids, rng,
            )

            # Train on A, test on B and vice versa.
            scaler_A = StandardScaler().fit(X_A)
            clf_A = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
            clf_A.fit(scaler_A.transform(X_A), y_A)
            in_A = clf_A.score(scaler_A.transform(X_A), y_A)
            out_B = clf_A.score(scaler_A.transform(X_B), y_B)

            scaler_B = StandardScaler().fit(X_B)
            clf_B = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
            clf_B.fit(scaler_B.transform(X_B), y_B)
            in_B = clf_B.score(scaler_B.transform(X_B), y_B)
            out_A = clf_B.score(scaler_B.transform(X_A), y_A)

            step_out["by_layer"][f"layer_{layer}"] = {
                "n_train_A": int(X_A.shape[0]),
                "n_train_B": int(X_B.shape[0]),
                "trained_on_A": {
                    "in_distribution_acc": float(in_A),
                    "held_out_acc": float(out_B),
                    "gap": float(in_A - out_B),
                },
                "trained_on_B": {
                    "in_distribution_acc": float(in_B),
                    "held_out_acc": float(out_A),
                    "gap": float(in_B - out_A),
                },
                "mean_held_out_acc": float((out_A + out_B) / 2),
            }
            log(
                f"      A->B: in={in_A*100:.1f}%, held-out={out_B*100:.1f}% "
                f"(gap={(in_A-out_B)*100:+.1f}%) | "
                f"B->A: in={in_B*100:.1f}%, held-out={out_A*100:.1f}% "
                f"(gap={(in_B-out_A)*100:+.1f}%)"
            )

        out["steps"][f"step_{step}"] = step_out
        del model
        torch.cuda.empty_cache()
        gc.collect()

    return out


# ====================================================================
# PHASE 2: POSITION SPECIFICITY AT STEP 2000
# ====================================================================

def phase2_position_specificity():
    """Patch the residual stream at PATCH_LAYERS at each of several
    positions and measure ΔLD. S2 should dominate; others should be
    near zero. Closes the 'any perturbation works' attack."""
    log("=" * 60)
    log("PHASE 2: Position specificity scan at step 2000")
    log("=" * 60)

    model = load_retrained(POS_SCAN_STEP)
    single_name_ids, _ = get_single_token_names(model.tokenizer)

    POSITIONS = ["S1", "S2", "IO", "BOS", "MID"]
    out = {"by_position": {}}

    for pos_name in POSITIONS:
        log(f"-- patching at {pos_name} --")
        rng = np.random.default_rng(SEED + 1)
        all_base = []
        all_patched = []

        for tmpl in POS_TEMPLATES:
            ds = IOIDataset(
                model=model, n_prompts=POS_PPT, templates=[tmpl],
                symmetric=True, seed=SEED,
            )
            ioi_tokens, ctrl_tokens, positions = build_control(
                model, ds, single_name_ids, rng, position=pos_name,
            )
            io_ids = torch.tensor(ds.io_token_ids, dtype=torch.long, device=DEVICE)
            s_ids = torch.tensor(ds.s_token_ids, dtype=torch.long, device=DEVICE)

            # Cache control's residual stream at PATCH_LAYERS.
            names = [f"blocks.{L}.hook_resid_post" for L in PATCH_LAYERS]
            donor = {}

            def make_cap(name):
                def fn(value, hook):
                    donor[name] = value.detach()
                    return value
                return fn

            with torch.no_grad():
                model.run_with_hooks(
                    ctrl_tokens,
                    fwd_hooks=[(n, make_cap(n)) for n in names],
                )

            # Baseline.
            with torch.no_grad():
                base_logits = model(ioi_tokens)
            base = logit_diff_per_prompt(base_logits, io_ids, s_ids).cpu().numpy()

            # Patch at the target position.
            def make_patch(layer_name, positions_tensor):
                donor_act = donor[layer_name]
                def fn(value, hook):
                    for i in range(value.shape[0]):
                        p = int(positions_tensor[i].item())
                        if p > 0 and p < value.shape[1]:
                            value[i, p, :] = donor_act[i, p, :]
                    return value
                return fn

            hooks = [(n, make_patch(n, positions)) for n in names]
            with torch.no_grad():
                patched_logits = model.run_with_hooks(ioi_tokens, fwd_hooks=hooks)
            patched = logit_diff_per_prompt(patched_logits, io_ids, s_ids).cpu().numpy()

            all_base.extend(base.tolist())
            all_patched.extend(patched.tolist())

            del ioi_tokens, ctrl_tokens, base_logits, patched_logits, donor
            torch.cuda.empty_cache()

        base_arr = np.asarray(all_base)
        patched_arr = np.asarray(all_patched)
        deltas = patched_arr - base_arr
        delta_lo, delta_hi = bootstrap_ci(deltas)

        out["by_position"][pos_name] = {
            "n": int(base_arr.shape[0]),
            "base_ld_mean": float(base_arr.mean()),
            "patched_ld_mean": float(patched_arr.mean()),
            "delta_ld_mean": float(deltas.mean()),
            "delta_ld_ci95": [delta_lo, delta_hi],
        }
        log(
            f"   {pos_name}: ΔLD={deltas.mean():+.4f} "
            f"[{delta_lo:+.3f}, {delta_hi:+.3f}]"
        )

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return out


# ====================================================================
# PHASE 3: L4H0 TRAJECTORY ACROSS 51 CHECKPOINTS
# ====================================================================

def phase3_l4h0_trajectory():
    """For each of 51 retrained checkpoints, measure:
      (a) L4H0 attention from END to {BOS, S1, S2, IO}
      (b) L4H0 OV projection onto W_U[IO] and W_U[S]
    Also measure L8H9 attention to S2 for comparison.
    This produces the correct mechanism overlay for the dense-sweep
    figure."""
    log("=" * 60)
    log("PHASE 3: L4H0 trajectory across 51 dense checkpoints")
    log("=" * 60)

    out = {"by_step": {}}
    t0 = time.time()

    for i, step in enumerate(DENSE_STEPS):
        try:
            model = load_retrained(step)
        except Exception as e:
            log(f"  step {step}: load failed: {e}")
            continue

        single_name_ids, _ = get_single_token_names(model.tokenizer)
        W_U = model.W_U  # [d_model, vocab]
        W_O = model.W_O  # [n_layers, n_heads, d_head, d_model]

        attn_to = {"BOS": [], "S1": [], "S2": [], "IO": []}
        l8h9_to_s2 = []
        l4h0_io_proj = []
        l4h0_s_proj = []

        for tmpl in TRAJ_TEMPLATES:
            ds = IOIDataset(
                model=model, n_prompts=TRAJ_PPT, templates=[tmpl],
                symmetric=True, seed=SEED,
            )
            ioi_tokens = model.to_tokens(ds.prompts).to(DEVICE)
            io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
            s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)

            # Capture attention patterns and z (head outputs pre-W_O).
            pat_cache = {}
            z_cache = {}

            def cap_pattern_l4(value, hook):
                pat_cache["l4"] = value.detach()
                return value

            def cap_pattern_l8(value, hook):
                pat_cache["l8"] = value.detach()
                return value

            def cap_z_l4(value, hook):
                z_cache["l4"] = value.detach()
                return value

            with torch.no_grad():
                model.run_with_hooks(
                    ioi_tokens,
                    fwd_hooks=[
                        (f"blocks.{L4H0_LAYER}.attn.hook_pattern", cap_pattern_l4),
                        (f"blocks.{L8H9_LAYER}.attn.hook_pattern", cap_pattern_l8),
                        (f"blocks.{L4H0_LAYER}.attn.hook_z", cap_z_l4),
                    ],
                )

            l4_pat = pat_cache["l4"][:, L4H0_HEAD, :, :]  # [B, Q, K]
            l8_pat = pat_cache["l8"][:, L8H9_HEAD, :, :]
            l4_z = z_cache["l4"][:, :, L4H0_HEAD, :]  # [B, T, d_head]

            for b in range(ioi_tokens.shape[0]):
                pos = find_token_positions(
                    ioi_tokens[b].cpu(), ds.io_token_ids[b], ds.s_token_ids[b],
                )
                end_q = pos["END"]

                for name in ("BOS", "S1", "S2", "IO"):
                    p = pos[name]
                    if p >= 0:
                        attn_to[name].append(float(l4_pat[b, end_q, p].item()))

                if pos["S2"] >= 0:
                    l8h9_to_s2.append(float(l8_pat[b, end_q, pos["S2"]].item()))

                # OV projection of L4H0 at END position onto IO and S
                # unembedding directions.
                head_out = l4_z[b, end_q] @ W_O[L4H0_LAYER, L4H0_HEAD]
                io_dir = W_U[:, int(io_ids[b].item())]
                s_dir = W_U[:, int(s_ids[b].item())]
                l4h0_io_proj.append(float(torch.dot(head_out, io_dir).item()))
                l4h0_s_proj.append(float(torch.dot(head_out, s_dir).item()))

            del pat_cache, z_cache
            torch.cuda.empty_cache()

        out["by_step"][f"step_{step}"] = {
            "l4h0_attn": {k: float(np.mean(v)) for k, v in attn_to.items()},
            "l8h9_attn_to_s2": float(np.mean(l8h9_to_s2)),
            "l4h0_ov_proj": {
                "mean_io": float(np.mean(l4h0_io_proj)),
                "mean_s": float(np.mean(l4h0_s_proj)),
                "io_minus_s": float(np.mean(l4h0_io_proj) - np.mean(l4h0_s_proj)),
            },
        }

        if (i + 1) % 5 == 0 or i == len(DENSE_STEPS) - 1:
            elapsed = time.time() - t0
            log(
                f"  [{i+1}/{len(DENSE_STEPS)}] step={step}: "
                f"L4H0→S2={out['by_step'][f'step_{step}']['l4h0_attn']['S2']:.3f}  "
                f"L8H9→S2={out['by_step'][f'step_{step}']['l8h9_attn_to_s2']:.3f}  "
                f"L4H0 OV(IO-S)={out['by_step'][f'step_{step}']['l4h0_ov_proj']['io_minus_s']:+.3f}  "
                f"({elapsed/(i+1):.1f}s/step)"
            )

        del model
        torch.cuda.empty_cache()
        gc.collect()

    return out


# ====================================================================
# PHASE 4: L4H0 ZERO-ABLATION VALIDATION AT STEP 5000
# ====================================================================

def phase4_zero_ablation():
    """At step 5000, zero-ablate each of the candidate heads (L4H0 -- our
    claim -- and L8H9 -- prior claim) and measure ΔLD.
    Compares against the existing causal_intervention.json convention
    (zero-ablation across all positions)."""
    log("=" * 60)
    log("PHASE 4: L4H0 zero-ablation validation at step 5000")
    log("=" * 60)

    model = load_retrained(ZERO_ABL_STEP)
    out = {"targets": {}}

    # Baseline.
    base_lds = []
    for tmpl in ALL_TEMPLATES[:10]:
        ds = IOIDataset(
            model=model, n_prompts=30, templates=[tmpl],
            symmetric=True, seed=SEED,
        )
        tokens = model.to_tokens(ds.prompts).to(DEVICE)
        io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
        s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)
        with torch.no_grad():
            logits = model(tokens)
        ld = logit_diff_per_prompt(logits, io_ids, s_ids).cpu().numpy()
        base_lds.extend(ld.tolist())
    base_mean = float(np.mean(base_lds))
    base_lo, base_hi = bootstrap_ci(base_lds)
    log(f"  baseline LD={base_mean:+.4f}  CI=[{base_lo:+.3f}, {base_hi:+.3f}]")

    for (L, H) in ZERO_ABL_TARGETS:
        abl_lds = []
        for tmpl in ALL_TEMPLATES[:10]:
            ds = IOIDataset(
                model=model, n_prompts=30, templates=[tmpl],
                symmetric=True, seed=SEED,
            )
            tokens = model.to_tokens(ds.prompts).to(DEVICE)
            io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
            s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)

            def zero_hook(value, hook, h=H):
                value[:, :, h, :] = 0.0
                return value

            with torch.no_grad():
                logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(f"blocks.{L}.attn.hook_z", zero_hook)],
                )
            ld = logit_diff_per_prompt(logits, io_ids, s_ids).cpu().numpy()
            abl_lds.extend(ld.tolist())

        abl_mean = float(np.mean(abl_lds))
        deltas = np.asarray(abl_lds) - np.asarray(base_lds)
        delta_lo, delta_hi = bootstrap_ci(deltas)
        out["targets"][f"L{L}H{H}"] = {
            "ablated_ld_mean": abl_mean,
            "delta_ld_mean": float(deltas.mean()),
            "delta_ld_ci95": [delta_lo, delta_hi],
        }
        log(
            f"  zero-ablate L{L}H{H}: LD={abl_mean:+.4f}  "
            f"ΔLD={deltas.mean():+.4f} [{delta_lo:+.3f}, {delta_hi:+.3f}]"
        )

    out["baseline_ld_mean"] = base_mean
    out["baseline_ld_ci95"] = [base_lo, base_hi]
    out["step"] = ZERO_ABL_STEP

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return out


# ====================================================================
# PHASE 5: MULTI-HEAD JOINT PATCHING AT STEP 2000
# ====================================================================

def phase5_multihead():
    """At step 2000, jointly patch the top-K S-promoting heads at S2 and
    measure cumulative ΔLD. Compare to a random-K control. Tests
    whether the S-bias is concentrated in few heads (compositional)
    or distributed across many (additive)."""
    log("=" * 60)
    log("PHASE 5: Multi-head joint patching at step 2000")
    log("=" * 60)

    # Load top S-promoting heads from the head-patching results.
    if not os.path.exists(HEAD_PATCHING_JSON):
        log(f"  WARN: {HEAD_PATCHING_JSON} not found; skipping")
        return None

    with open(HEAD_PATCHING_JSON) as f:
        hp = json.load(f)
    if f"step_{MULTIHEAD_STEP}" not in hp["by_step"]:
        log(f"  WARN: step_{MULTIHEAD_STEP} not in head_patching; skipping")
        return None

    heads_sorted = sorted(
        hp["by_step"][f"step_{MULTIHEAD_STEP}"]["heads"],
        key=lambda h: h["delta_ld_mean"], reverse=True,
    )
    top_heads_all = [(h["layer"], h["head"]) for h in heads_sorted]
    top_heads_promoting = top_heads_all[:max(MULTIHEAD_K_VALUES)]
    log(
        f"  Top {max(MULTIHEAD_K_VALUES)} S-promoting heads from head_patching: "
        + ", ".join(f"L{L}H{H}" for L, H in top_heads_promoting)
    )

    model = load_retrained(MULTIHEAD_STEP, enable_attn_result=True)
    single_name_ids, _ = get_single_token_names(model.tokenizer)
    rng_global = np.random.default_rng(SEED + 1)

    # Sample a fixed random-K head set for the control (matches max K).
    rng_random = np.random.default_rng(SEED + 7)
    all_heads = [(L, H) for L in range(12) for H in range(12)]
    promoting_set = set(top_heads_promoting)
    nonpromoting = [h for h in all_heads if h not in promoting_set]
    random_K = list(rng_random.choice(
        len(nonpromoting), size=max(MULTIHEAD_K_VALUES), replace=False,
    ))
    random_K_heads = [nonpromoting[i] for i in random_K]
    log("  Random-K control heads: " + ", ".join(f"L{L}H{H}" for L, H in random_K_heads))

    def run_joint_patch(model, head_set):
        """Patch all heads in head_set at S2 jointly with control output."""
        all_base = []
        all_patched = []
        rng = np.random.default_rng(SEED + 1)

        for tmpl in ALL_TEMPLATES[:10]:
            ds = IOIDataset(
                model=model, n_prompts=30, templates=[tmpl],
                symmetric=True, seed=SEED,
            )
            ioi_tokens, ctrl_tokens, s2_positions = build_control(
                model, ds, single_name_ids, rng, position="S2",
            )
            io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
            s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)

            # Cache control's per-head outputs at all layers needed.
            needed_layers = sorted({L for L, _ in head_set})
            result_names = [f"blocks.{L}.attn.hook_result" for L in needed_layers]
            donor = {}

            def make_cap(name):
                def fn(value, hook):
                    donor[name] = value.detach()
                    return value
                return fn

            with torch.no_grad():
                model.run_with_hooks(
                    ctrl_tokens,
                    fwd_hooks=[(n, make_cap(n)) for n in result_names],
                )

            # Baseline.
            with torch.no_grad():
                base_logits = model(ioi_tokens)
            base = logit_diff_per_prompt(base_logits, io_ids, s_ids).cpu().numpy()

            # Patch all targeted heads at S2 (group by layer).
            heads_by_layer = {}
            for L, H in head_set:
                heads_by_layer.setdefault(L, []).append(H)

            def make_multi_patch(layer, heads_in_layer, positions):
                name = f"blocks.{layer}.attn.hook_result"
                donor_act = donor[name]
                def fn(value, hook):
                    for i in range(value.shape[0]):
                        p = int(positions[i].item())
                        if p >= 0:
                            for H in heads_in_layer:
                                value[i, p, H, :] = donor_act[i, p, H, :]
                    return value
                return fn

            hooks = [
                (f"blocks.{L}.attn.hook_result",
                 make_multi_patch(L, heads_by_layer[L], s2_positions))
                for L in heads_by_layer
            ]
            with torch.no_grad():
                patched_logits = model.run_with_hooks(ioi_tokens, fwd_hooks=hooks)
            patched = logit_diff_per_prompt(patched_logits, io_ids, s_ids).cpu().numpy()

            all_base.extend(base.tolist())
            all_patched.extend(patched.tolist())

            del ioi_tokens, ctrl_tokens, base_logits, patched_logits, donor
            torch.cuda.empty_cache()

        base_arr = np.asarray(all_base)
        patched_arr = np.asarray(all_patched)
        deltas = patched_arr - base_arr
        delta_lo, delta_hi = bootstrap_ci(deltas)
        return {
            "n": int(base_arr.shape[0]),
            "base_ld_mean": float(base_arr.mean()),
            "patched_ld_mean": float(patched_arr.mean()),
            "delta_ld_mean": float(deltas.mean()),
            "delta_ld_ci95": [delta_lo, delta_hi],
        }

    out = {"top_K": {}, "random_K": {}, "config": {
        "top_heads_used": [[L, H] for L, H in top_heads_promoting],
        "random_heads_used": [[L, H] for L, H in random_K_heads],
        "K_values": MULTIHEAD_K_VALUES,
    }}

    for K in MULTIHEAD_K_VALUES:
        head_set = top_heads_promoting[:K]
        log(f"  Top-{K} joint patch: {head_set}")
        result = run_joint_patch(model, head_set)
        out["top_K"][f"K_{K}"] = result
        log(
            f"    ΔLD={result['delta_ld_mean']:+.4f} "
            f"[{result['delta_ld_ci95'][0]:+.3f}, {result['delta_ld_ci95'][1]:+.3f}]"
        )

    log("  Random-K control runs:")
    for K in MULTIHEAD_K_VALUES:
        head_set = random_K_heads[:K]
        result = run_joint_patch(model, head_set)
        out["random_K"][f"K_{K}"] = result
        log(
            f"    K={K}: ΔLD={result['delta_ld_mean']:+.4f} "
            f"[{result['delta_ld_ci95'][0]:+.3f}, {result['delta_ld_ci95'][1]:+.3f}]"
        )

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return out


# ====================================================================
# MAIN
# ====================================================================


def main():
    os.makedirs("results", exist_ok=True)

    results = {"config": {
        "model": RETRAINED_REPO,
        "seed": SEED,
        "n_bootstrap": N_BOOTSTRAP,
    }}

    # Resume.
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH) as f:
                results = json.load(f)
            done = [k for k in results if k.startswith("phase")]
            log(f"Resuming. Already done: {done}")
        except Exception as e:
            log(f"Could not resume: {e}")

    phases = [
        ("phase1_heldout_probes", phase1_heldout_probes),
        ("phase2_position_specificity", phase2_position_specificity),
        ("phase3_l4h0_trajectory", phase3_l4h0_trajectory),
        ("phase4_zero_ablation", phase4_zero_ablation),
        ("phase5_multihead", phase5_multihead),
    ]

    t0 = time.time()
    for key, fn in phases:
        cached = results.get(key)
        # Skip only if the phase has a real result. Errored entries
        # ({"error": "..."}) should be retried, not skipped.
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
