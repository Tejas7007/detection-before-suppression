"""
Consistent multi-scale sign flip (curated prompts)
===================================================
Re-runs the S2 sign flip at 160M/410M/1B using the SAME curated
10-template protocol as in emnlp_cross_model.py,
not the random IOIDataset sampling used in the overnight sweeps.

Why: random IOIDataset prompts give mature LD ~+0.1 / acc ~51%,
inconsistent with the literature (>95%) and with our earlier
+0.94 -> -4.13 numbers. The curated template set gives clean,
high-magnitude, literature-comparable results.

Protocol (per checkpoint, n=300 = 10 templates x 30 prompts):
  - Single-token names only
  - Baseline IOI logit difference + accuracy
  - Patched LD: replace S2 residual at patch layers with matched control
  - ΔLD = patched - base  (>0 at dip = removing S2 helps; <0 at maturity)
  - Bootstrap 95% CI (10,000 resamples)
  - Logit lens: per-layer LD at final position (flip layer)

Patch layers scaled to depth (~25-45%):
  160M (12L): [3,4,5]   410M (24L): [6,7,8,9,10]   1B (16L): [4,5,6,7,8]

Output: results/emnlp_consistent_signflip.json
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
RESULTS_PATH = "results/emnlp_consistent_signflip.json"

MODELS = [
    {
        "name": "pythia-160m",
        "repo": "EleutherAI/pythia-160m-deduped",
        "patch_layers": [3, 4, 5],
        "checkpoints": [0, 256, 512, 1000, 2000, 3000, 4000,
                        5000, 7000, 8000, 10000, 13000, 33000, 143000],
    },
    {
        "name": "pythia-410m",
        "repo": "EleutherAI/pythia-410m-deduped",
        "patch_layers": [6, 7, 8, 9, 10],
        "checkpoints": [0, 1000, 2000, 3000, 4000, 5000, 8000, 13000,
                        20000, 30000, 50000, 100000, 143000],
    },
    {
        "name": "pythia-1b",
        "repo": "EleutherAI/pythia-1b-deduped",
        "patch_layers": [4, 5, 6, 7, 8],
        "checkpoints": [0, 1000, 2000, 3000, 4000, 5000, 8000, 15000,
                        20000, 30000, 50000, 100000, 143000],
    },
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def bootstrap_ci(values, seed=SEED):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2: return float("nan"), float("nan")
    idx = rng.integers(0, len(arr), size=(N_BOOTSTRAP, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))

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

def logit_diff_per_prompt(logits, io_ids, s_ids):
    last = logits[:, -1, :]
    idx = torch.arange(last.shape[0], device=last.device)
    return last[idx, io_ids] - last[idx, s_ids]


def load_model(repo, step):
    hf = AutoModelForCausalLM.from_pretrained(
        repo, revision=f"step{step}", torch_dtype=torch.float32)
    model = HookedTransformer.from_pretrained(
        repo, hf_model=hf, device=DEVICE,
        center_writing_weights=True, center_unembed=True, fold_ln=True)
    del hf; torch.cuda.empty_cache()
    return model


def run_checkpoint(model, single_name_ids, patch_layers):
    """Curated protocol: 10 templates x 30 prompts, S2 patching, logit lens."""
    rng = np.random.default_rng(SEED + 1)
    base_lds, patched_lds = [], []
    n_layers = model.cfg.n_layers
    # Accumulate logit-lens LD per layer across all templates.
    ll_sums = {L: [] for L in range(n_layers)}

    for tmpl in TEMPLATES:
        ds = IOIDataset(model=model, n_prompts=PROMPTS_PER_TEMPLATE,
                        templates=[tmpl], symmetric=True, seed=SEED)
        ioi_tokens = model.to_tokens(ds.prompts).to(DEVICE)
        n = ioi_tokens.shape[0]

        s2_positions = []
        for i in range(n):
            s2_positions.append(find_s2_position(ioi_tokens[i].cpu(), ds.s_token_ids[i]))
        s2_positions = torch.tensor(s2_positions, dtype=torch.long, device=DEVICE)

        # Build control: replace S2 with a third single-token name.
        ctrl_tokens = ioi_tokens.clone()
        for i in range(n):
            io_id, s_id = int(ds.io_token_ids[i]), int(ds.s_token_ids[i])
            pool = [t for t in single_name_ids if t != io_id and t != s_id]
            p = int(s2_positions[i].item())
            if pool and p > 0:
                ctrl_tokens[i, p] = int(rng.choice(pool))

        io_ids = torch.tensor(ds.io_token_ids, dtype=torch.long, device=DEVICE)
        s_ids = torch.tensor(ds.s_token_ids, dtype=torch.long, device=DEVICE)

        # Cache control activations at patch layers.
        names = [f"blocks.{L}.hook_resid_post" for L in patch_layers]
        donor = {}
        def make_cap(name):
            def fn(value, hook):
                donor[name] = value.detach()
                return value
            return fn
        with torch.no_grad():
            model.run_with_hooks(ctrl_tokens, fwd_hooks=[(nm, make_cap(nm)) for nm in names])

        # Baseline + logit lens cache.
        ll_names = [f"blocks.{L}.hook_resid_post" for L in range(n_layers)]
        with torch.no_grad():
            base_logits, cache = model.run_with_cache(ioi_tokens, names_filter=ll_names)
        base = logit_diff_per_prompt(base_logits, io_ids, s_ids).detach().cpu().numpy()

        # Logit lens per layer.
        idx = torch.arange(n, device=DEVICE)
        for L in range(n_layers):
            resid = cache[f"blocks.{L}.hook_resid_post"][:, -1, :]
            normed = model.ln_final(resid)
            ll_logits = normed @ model.W_U + model.b_U
            ll_ld = (ll_logits[idx, io_ids] - ll_logits[idx, s_ids]).detach().cpu().numpy()
            ll_sums[L].extend(ll_ld.tolist())

        # Patched (S2 -> control).
        def make_patch(layer):
            donor_act = donor[f"blocks.{layer}.hook_resid_post"]
            def fn(value, hook):
                for i in range(value.shape[0]):
                    p = int(s2_positions[i].item())
                    if p >= 0:
                        value[i, p, :] = donor_act[i, p, :]
                return value
            return fn
        hooks = [(f"blocks.{L}.hook_resid_post", make_patch(L)) for L in patch_layers]
        with torch.no_grad():
            patched_logits = model.run_with_hooks(ioi_tokens, fwd_hooks=hooks)
        patched = logit_diff_per_prompt(patched_logits, io_ids, s_ids).detach().cpu().numpy()

        base_lds.extend(base.tolist())
        patched_lds.extend(patched.tolist())
        del cache

    base_arr = np.array(base_lds)
    patched_arr = np.array(patched_lds)
    delta = patched_arr - base_arr

    base_lo, base_hi = bootstrap_ci(base_arr)
    d_lo, d_hi = bootstrap_ci(delta)

    # Logit lens flip layer.
    logit_lens = {}
    flip_layer = None
    for L in range(n_layers):
        m = float(np.mean(ll_sums[L]))
        logit_lens[f"L{L}"] = m
        if m > 0 and flip_layer is None:
            flip_layer = L

    return {
        "n": len(base_lds),
        "base_ld_mean": float(base_arr.mean()),
        "base_ld_ci": [base_lo, base_hi],
        "ioi_acc": float((base_arr > 0).mean()),
        "patched_ld_mean": float(patched_arr.mean()),
        "delta_ld_mean": float(delta.mean()),
        "delta_ld_ci": [d_lo, d_hi],
        "logit_lens": logit_lens,
        "flip_layer": flip_layer,
    }


def main():
    os.makedirs("results", exist_ok=True)
    results = {}
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH) as f:
                results = json.load(f)
        except: pass

    t0 = time.time()
    for cfg in MODELS:
        name = cfg["name"]
        if name not in results:
            results[name] = {"repo": cfg["repo"], "patch_layers": cfg["patch_layers"],
                             "by_step": {}}
        log("=" * 60)
        log(f"{name}  (patch layers {cfg['patch_layers']})")
        log("=" * 60)

        single_ids = None
        for step in cfg["checkpoints"]:
            key = f"step_{step}"
            if key in results[name]["by_step"]:
                log(f"  SKIP step {step}"); continue
            log(f"  step {step}:")
            try:
                model = load_model(cfg["repo"], step)
                if single_ids is None:
                    single_ids = get_single_token_names(model.tokenizer)
                r = run_checkpoint(model, single_ids, cfg["patch_layers"])
                results[name]["by_step"][key] = r
                log(f"    acc={r['ioi_acc']*100:5.1f}%  base_LD={r['base_ld_mean']:+.3f}  "
                    f"ΔLD={r['delta_ld_mean']:+.4f} [{r['delta_ld_ci'][0]:+.3f},{r['delta_ld_ci'][1]:+.3f}]  "
                    f"flip_L={r['flip_layer']}")
                del model; torch.cuda.empty_cache(); gc.collect()
            except Exception as e:
                log(f"    ERROR: {str(e)[:100]}")
                results[name]["by_step"][key] = {"error": str(e)[:120]}
            with open(RESULTS_PATH, "w") as f:
                json.dump(results, f, indent=2)

    elapsed = (time.time() - t0) / 60
    log(f"\nDone. {elapsed:.1f} min. Output: {RESULTS_PATH}")

    # Summary table.
    log("\n" + "=" * 60)
    log("SIGN FLIP SUMMARY (curated prompts)")
    log("=" * 60)
    for cfg in MODELS:
        name = cfg["name"]
        by = results.get(name, {}).get("by_step", {})
        if not by: continue
        log(f"\n{name}:")
        log(f"  {'Step':>7} {'Acc':>6} {'base_LD':>8} {'ΔLD':>8} {'Flip':>5}")
        steps = sorted(by.keys(), key=lambda x: int(x.split("_")[1]))
        accs, dlds = [], []
        for sk in steps:
            r = by[sk]
            if not r or "error" in r: continue
            step = sk.split("_")[1]
            fl = f"L{r['flip_layer']}" if r['flip_layer'] is not None else "None"
            log(f"  {step:>7} {r['ioi_acc']*100:>5.1f}% {r['base_ld_mean']:>+8.3f} "
                f"{r['delta_ld_mean']:>+8.4f} {fl:>5}")
            accs.append((int(step), r['ioi_acc']))
            dlds.append((int(step), r['delta_ld_mean']))
        if accs:
            min_acc = min(accs, key=lambda x: x[1])
            flip = None
            for i in range(1, len(dlds)):
                if dlds[i-1][1] > 0.02 and dlds[i][1] < -0.02:
                    flip = dlds[i][0]; break
            log(f"  -> dip floor {min_acc[1]*100:.1f}% @ step {min_acc[0]}, "
                f"sign flip {'step '+str(flip) if flip else 'NO'}")


if __name__ == "__main__":
    main()
