"""
Control Battery: is the sign flip position-specific and stage-specific?
========================================================================
The negative_controls.json showed that at the dip, a random-direction
perturbation at S2 (+0.83) nearly matches the real S2 patch (+0.94).
A natural question: is the effect about duplicate-token CONTENT, or
just about disrupting the S2 POSITION?

This script settles it by running a full battery at BOTH the dip
(step 2000) and maturity (step 143000), curated 10-template protocol:

  Arms (all at patch layers 3-5):
    1. real_S2     : S2 residual -> matched control-name residual
    2. random_S2   : S2 residual -> norm-matched random Gaussian
    3. random_S1   : S1 residual -> norm-matched random Gaussian
    4. random_IO   : IO residual -> norm-matched random Gaussian
    5. random_struct: a non-name structural token -> random Gaussian

Defensible claim if it holds:
  - real_S2 and random_S2 BOTH flip sign (positive at dip, negative
    at maturity) => the sign flip is about the S2 POSITION, robust to
    perturbation content.
  - random_struct ~ 0 at both stages => position-specific, not generic.
  - This lets us claim position+stage specificity, which we CAN defend,
    rather than content specificity, which we cannot.

Output: results/emnlp_control_battery.json
"""

import os, gc, json, time, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer

try:
    from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES, CANDIDATE_NAMES
except ImportError:
    sys.path.insert(0, "/workspace/ioi-sign-flip/src")
    from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES, CANDIDATE_NAMES

DEVICE = "cuda"
SEED = 42
N_BOOTSTRAP = 10_000
TEMPLATES = ALL_TEMPLATES[:10]
PROMPTS_PER_TEMPLATE = 30
PATCH_LAYERS = [3, 4, 5]
MODEL = "EleutherAI/pythia-160m-deduped"
STAGES = {"dip": 2000, "mature": 143000}
RESULTS_PATH = "results/emnlp_control_battery.json"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def bootstrap_ci(values, seed=SEED):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2: return float("nan"), float("nan")
    idx = rng.integers(0, len(arr), size=(N_BOOTSTRAP, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))

def find_nth_position(token_row, target_id, n):
    seen = 0
    for j in range(1, token_row.shape[0]):
        if int(token_row[j].item()) == int(target_id):
            seen += 1
            if seen == n:
                return j
    return -1

def get_single_token_names(tokenizer):
    ids = []
    for name in CANDIDATE_NAMES:
        toks = tokenizer.encode(" " + name, add_special_tokens=False)
        if len(toks) == 1:
            ids.append(int(toks[0]))
    return ids

def logit_diff(logits, io_ids, s_ids):
    last = logits[:, -1, :]
    idx = torch.arange(last.shape[0], device=last.device)
    return last[idx, io_ids] - last[idx, s_ids]


def run_stage(model, single_ids, step):
    """Run all battery arms at one checkpoint, return per-arm ΔLD."""
    rng = np.random.default_rng(SEED + 1)
    n_layers = model.cfg.n_layers

    # Accumulators per arm.
    arms = ["real_S2", "random_S2", "random_S1", "random_IO", "random_struct"]
    base_all = []
    patched_all = {a: [] for a in arms}

    for tmpl in TEMPLATES:
        ds = IOIDataset(model=model, n_prompts=PROMPTS_PER_TEMPLATE,
                        templates=[tmpl], symmetric=True, seed=SEED)
        ioi_tokens = model.to_tokens(ds.prompts).to(DEVICE)
        n, seqlen = ioi_tokens.shape

        io_ids = torch.tensor(ds.io_token_ids, dtype=torch.long, device=DEVICE)
        s_ids = torch.tensor(ds.s_token_ids, dtype=torch.long, device=DEVICE)

        # Find positions per prompt.
        s2_pos, s1_pos, io_pos, struct_pos = [], [], [], []
        for i in range(n):
            row = ioi_tokens[i].cpu()
            s2 = find_nth_position(row, ds.s_token_ids[i], 2)
            s1 = find_nth_position(row, ds.s_token_ids[i], 1)
            io = find_nth_position(row, ds.io_token_ids[i], 1)
            s2_pos.append(s2); s1_pos.append(s1); io_pos.append(io)
            # Structural token: position right after S1 (the "and"/"," token),
            # guaranteed not a name. Fall back to position 2.
            sp = s1 + 1 if (s1 > 0 and s1 + 1 < seqlen and s1 + 1 != io) else 2
            struct_pos.append(sp)

        # Build control tokens (S2 -> third name) for real_S2 arm.
        ctrl_tokens = ioi_tokens.clone()
        for i in range(n):
            io_id, s_id = int(io_ids[i]), int(s_ids[i])
            pool = [t for t in single_ids if t != io_id and t != s_id]
            if pool and s2_pos[i] > 0:
                ctrl_tokens[i, s2_pos[i]] = int(rng.choice(pool))

        names = [f"blocks.{L}.hook_resid_post" for L in PATCH_LAYERS]

        # Cache control activations (for real_S2).
        donor = {}
        def make_cap(name):
            def fn(value, hook):
                donor[name] = value.detach()
                return value
            return fn
        with torch.no_grad():
            model.run_with_hooks(ctrl_tokens, fwd_hooks=[(nm, make_cap(nm)) for nm in names])

        # Baseline + cache clean activations (for norm-matched random arms).
        clean = {}
        def make_cap_clean(name):
            def fn(value, hook):
                clean[name] = value.detach()
                return value
            return fn
        with torch.no_grad():
            base_logits = model.run_with_hooks(
                ioi_tokens, fwd_hooks=[(nm, make_cap_clean(nm)) for nm in names])
        base = logit_diff(base_logits, io_ids, s_ids).detach().cpu().numpy()
        base_all.extend(base.tolist())

        # Precompute norm-matched random replacements per layer.
        # For each arm/position, replace residual with a random Gaussian
        # scaled to the per-example norm of the clean residual at that position.
        def random_replacement(layer, positions):
            clean_act = clean[f"blocks.{layer}.hook_resid_post"]  # [n, seq, d]
            d = clean_act.shape[-1]
            repl = torch.zeros(n, d, device=DEVICE)
            for i in range(n):
                p = positions[i]
                if p < 0: continue
                norm = clean_act[i, p, :].norm()
                g = torch.randn(d, device=DEVICE)
                repl[i] = g / g.norm() * norm
            return repl

        # Pre-generate random replacements (seeded for reproducibility).
        torch.manual_seed(SEED + step)
        rand_repl = {}
        for arm, positions in [("random_S2", s2_pos), ("random_S1", s1_pos),
                               ("random_IO", io_pos), ("random_struct", struct_pos)]:
            rand_repl[arm] = {L: random_replacement(L, positions) for L in PATCH_LAYERS}

        # --- Arm hooks ---
        def real_s2_hook(layer):
            donor_act = donor[f"blocks.{layer}.hook_resid_post"]
            def fn(value, hook):
                for i in range(value.shape[0]):
                    p = s2_pos[i]
                    if p >= 0:
                        value[i, p, :] = donor_act[i, p, :]
                return value
            return fn

        def random_hook(layer, positions, arm):
            repl = rand_repl[arm][layer]
            def fn(value, hook):
                for i in range(value.shape[0]):
                    p = positions[i]
                    if p >= 0:
                        value[i, p, :] = repl[i]
                return value
            return fn

        arm_hooks = {
            "real_S2": [(f"blocks.{L}.hook_resid_post", real_s2_hook(L)) for L in PATCH_LAYERS],
            "random_S2": [(f"blocks.{L}.hook_resid_post", random_hook(L, s2_pos, "random_S2")) for L in PATCH_LAYERS],
            "random_S1": [(f"blocks.{L}.hook_resid_post", random_hook(L, s1_pos, "random_S1")) for L in PATCH_LAYERS],
            "random_IO": [(f"blocks.{L}.hook_resid_post", random_hook(L, io_pos, "random_IO")) for L in PATCH_LAYERS],
            "random_struct": [(f"blocks.{L}.hook_resid_post", random_hook(L, struct_pos, "random_struct")) for L in PATCH_LAYERS],
        }

        for arm in arms:
            with torch.no_grad():
                p_logits = model.run_with_hooks(ioi_tokens, fwd_hooks=arm_hooks[arm])
            p_ld = logit_diff(p_logits, io_ids, s_ids).detach().cpu().numpy()
            patched_all[arm].extend(p_ld.tolist())

    # Aggregate.
    base_arr = np.array(base_all)
    out = {
        "step": step,
        "base_ld_mean": float(base_arr.mean()),
        "ioi_acc": float((base_arr > 0).mean()),
        "n": len(base_all),
        "arms": {},
    }
    for arm in arms:
        p_arr = np.array(patched_all[arm])
        delta = p_arr - base_arr
        lo, hi = bootstrap_ci(delta)
        out["arms"][arm] = {
            "patched_ld_mean": float(p_arr.mean()),
            "delta_ld_mean": float(delta.mean()),
            "delta_ci": [lo, hi],
        }
    return out


def main():
    os.makedirs("results", exist_ok=True)
    results = {}
    if os.path.exists(RESULTS_PATH):
        try:
            results = json.load(open(RESULTS_PATH))
        except: pass

    single_ids = None
    for stage_name, step in STAGES.items():
        if stage_name in results:
            log(f"SKIP {stage_name}"); continue
        log("=" * 60)
        log(f"STAGE: {stage_name} (step {step})")
        log("=" * 60)
        hf = AutoModelForCausalLM.from_pretrained(MODEL, revision=f"step{step}",
                                                  torch_dtype=torch.float32)
        model = HookedTransformer.from_pretrained(
            MODEL, hf_model=hf, device=DEVICE,
            center_writing_weights=True, center_unembed=True, fold_ln=True)
        del hf; torch.cuda.empty_cache()
        if single_ids is None:
            single_ids = get_single_token_names(model.tokenizer)

        r = run_stage(model, single_ids, step)
        results[stage_name] = r
        json.dump(results, open(RESULTS_PATH, "w"), indent=2)

        log(f"  base: acc={r['ioi_acc']*100:.1f}%  LD={r['base_ld_mean']:+.3f}")
        for arm, v in r["arms"].items():
            log(f"  {arm:>14}: ΔLD={v['delta_ld_mean']:+.4f} "
                f"[{v['delta_ci'][0]:+.3f},{v['delta_ci'][1]:+.3f}]")
        del model; torch.cuda.empty_cache(); gc.collect()

    # Summary: the sign-flip table.
    log("\n" + "=" * 60)
    log("CONTROL BATTERY SUMMARY")
    log("=" * 60)
    log(f"{'Arm':>14} {'ΔLD dip':>12} {'ΔLD mature':>14} {'flips?':>8}")
    log("-" * 52)
    if "dip" in results and "mature" in results:
        for arm in results["dip"]["arms"]:
            d_dip = results["dip"]["arms"][arm]["delta_ld_mean"]
            d_mat = results["mature"]["arms"][arm]["delta_ld_mean"]
            flips = "YES" if (d_dip > 0.05 and d_mat < -0.05) else "no"
            log(f"{arm:>14} {d_dip:>+12.4f} {d_mat:>+14.4f} {flips:>8}")
    log("\nInterpretation: arms that flip => position+stage specific.")
    log("random_struct should be ~0 at both => confirms position specificity.")


if __name__ == "__main__":
    main()
