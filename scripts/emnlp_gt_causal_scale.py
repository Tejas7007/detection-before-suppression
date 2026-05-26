"""
Greater-Than causal sign flip at 410M and 1B (fixed year-position bug)
=======================================================================
The overnight GT-at-scale patching returned Delta=0 everywhere because
year-position detection failed at larger scales. This fixes it by:
  1. Filtering to start-years YY where "17YY" tokenizes cleanly as
     ["17", "YY"] (verified at runtime per tokenizer).
  2. Tracking the YY token position explicitly (token right after the
     first "17"), with a verification count logged.
  3. Patching the residual at the YY position with a control prompt
     that has a DIFFERENT start year (interchange intervention).

Metric: P(>YY) - P(<=YY) over two-digit completions.
At the dip: model prefers <= (negative). At maturity: prefers > (positive).
Sign flip of the causal Delta = the GT analog of the IOI sign flip.

Runs 160M (sanity, should match +0.26->-0.66), 410M, 1B.
Output: results/emnlp_gt_causal_scale.json
"""

import os, gc, json, time, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer

DEVICE = "cuda"
SEED = 42
N_BOOTSTRAP = 10_000
N_PROMPTS = 300
RESULTS_PATH = "results/emnlp_gt_causal_scale.json"

EVENTS = ["war", "battle", "siege", "conflict", "dispute", "famine",
          "drought", "reign", "expedition", "voyage"]

MODELS = [
    {"name": "pythia-160m", "repo": "EleutherAI/pythia-160m-deduped",
     "patch_layers": [3, 4, 5],
     "checkpoints": [1000, 2000, 3000, 5000, 8000, 143000]},
    {"name": "pythia-410m", "repo": "EleutherAI/pythia-410m-deduped",
     "patch_layers": [6, 7, 8, 9, 10],
     "checkpoints": [1000, 2000, 3000, 5000, 8000, 143000]},
    {"name": "pythia-1b", "repo": "EleutherAI/pythia-1b-deduped",
     "patch_layers": [4, 5, 6, 7, 8],
     "checkpoints": [1000, 2000, 3000, 5000, 8000, 143000]},
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


def get_clean_years(tokenizer):
    """YY values where '17YY' tokenizes as ['17','YY'] (2 clean tokens)."""
    clean = []
    for yy in range(2, 99):
        ids = tokenizer.encode(f"17{yy:02d}", add_special_tokens=False)
        if len(ids) == 2:
            t0 = tokenizer.decode([ids[0]]).strip()
            t1 = tokenizer.decode([ids[1]]).strip()
            if t0 == "17" and t1 == f"{yy:02d}":
                clean.append(yy)
    return clean

def get_two_digit_tokens(tokenizer):
    """Map each two-digit value to its single token id (where it exists)."""
    m = {}
    for d in range(100):
        ids = tokenizer.encode(f"{d:02d}", add_special_tokens=False)
        if len(ids) == 1:
            m[d] = ids[0]
    return m


def build_prompts(tokenizer, clean_years, rng):
    """Build GT prompts + controls; track YY token position."""
    prompts, ctrls, start_years, ctrl_years = [], [], [], []
    for _ in range(N_PROMPTS):
        event = rng.choice(EVENTS)
        yy = int(rng.choice(clean_years))
        cy = int(rng.choice(clean_years))
        while abs(cy - yy) < 10:
            cy = int(rng.choice(clean_years))
        prompts.append(f"The {event} lasted from the year 17{yy:02d} to the year 17")
        ctrls.append(f"The {event} lasted from the year 17{cy:02d} to the year 17")
        start_years.append(yy)
        ctrl_years.append(cy)
    return prompts, ctrls, start_years, ctrl_years


def find_yy_position(model, prompt_tokens, yy, tokenizer):
    """Find position of the YY token (the one right after the FIRST '17')."""
    seq = prompt_tokens.cpu().tolist()
    tok_17 = tokenizer.encode("17", add_special_tokens=False)
    tok_yy = tokenizer.encode(f"{yy:02d}", add_special_tokens=False)
    if len(tok_17) != 1 or len(tok_yy) != 1:
        return -1
    id_17, id_yy = tok_17[0], tok_yy[0]
    # Find first '17' then check next token is YY.
    for j in range(len(seq) - 1):
        if seq[j] == id_17 and seq[j+1] == id_yy:
            return j + 1
    # Fallback: just find the YY token.
    for j in range(len(seq)):
        if seq[j] == id_yy:
            return j
    return -1


def gt_diff(logits_last, start_years, two_digit):
    """P(>YY) - P(<=YY) per prompt over two-digit completion tokens."""
    diffs = []
    for i in range(len(start_years)):
        probs = F.softmax(logits_last[i].float(), dim=-1).detach().cpu().numpy()
        yy = start_years[i]
        gt = sum(probs[two_digit[d]] for d in two_digit if d > yy)
        le = sum(probs[two_digit[d]] for d in two_digit if d <= yy)
        diffs.append(gt - le)
    return np.array(diffs)


def run_checkpoint(model, tokenizer, clean_years, two_digit, patch_layers):
    rng = np.random.default_rng(SEED)
    prompts, ctrls, start_years, ctrl_years = build_prompts(tokenizer, clean_years, rng)

    tokens = model.to_tokens(prompts).to(DEVICE)
    ctrl_tokens = model.to_tokens(ctrls).to(DEVICE)
    n = tokens.shape[0]

    # Find YY positions in both IOI and control (same structure => same pos).
    yy_pos = []
    for i in range(n):
        p = find_yy_position(model, tokens[i], start_years[i], tokenizer)
        yy_pos.append(p)
    n_found = sum(1 for p in yy_pos if p > 0)

    # Base.
    with torch.no_grad():
        base_logits = model(tokens)[:, -1, :]
    base = gt_diff(base_logits, start_years, two_digit)
    gt_acc = float((base > 0).mean())

    # Cache control activations at patch layers.
    names = [f"blocks.{L}.hook_resid_post" for L in patch_layers]
    donor = {}
    def cap(name):
        def fn(value, hook):
            donor[name] = value.detach(); return value
        return fn
    with torch.no_grad():
        model.run_with_hooks(ctrl_tokens, fwd_hooks=[(nm, cap(nm)) for nm in names])

    # Patch YY position with control's YY residual.
    def patch(layer):
        donor_act = donor[f"blocks.{layer}.hook_resid_post"]
        def fn(value, hook):
            for i in range(value.shape[0]):
                p = yy_pos[i]
                if p >= 0:
                    value[i, p, :] = donor_act[i, p, :]
            return value
        return fn
    with torch.no_grad():
        p_logits = model.run_with_hooks(
            tokens, fwd_hooks=[(f"blocks.{L}.hook_resid_post", patch(L)) for L in patch_layers]
        )[:, -1, :]
    patched = gt_diff(p_logits, start_years, two_digit)

    delta = patched - base
    lo, hi = bootstrap_ci(delta)
    return {
        "n": n, "n_yy_found": n_found,
        "gt_acc": gt_acc,
        "base_diff_mean": float(base.mean()),
        "patched_diff_mean": float(patched.mean()),
        "delta_mean": float(delta.mean()),
        "delta_ci": [lo, hi],
    }


def main():
    os.makedirs("results", exist_ok=True)
    results = {}
    if os.path.exists(RESULTS_PATH):
        try: results = json.load(open(RESULTS_PATH))
        except: pass

    for cfg in MODELS:
        name = cfg["name"]
        if name not in results:
            results[name] = {"repo": cfg["repo"], "by_step": {}}
        log("=" * 60)
        log(f"{name}  (patch layers {cfg['patch_layers']})")
        log("=" * 60)

        clean_years = two_digit = None
        for step in cfg["checkpoints"]:
            key = f"step_{step}"
            if key in results[name]["by_step"]:
                log(f"  SKIP step {step}"); continue
            log(f"  step {step}:")
            try:
                hf = AutoModelForCausalLM.from_pretrained(
                    cfg["repo"], revision=f"step{step}", torch_dtype=torch.float32)
                model = HookedTransformer.from_pretrained(
                    cfg["repo"], hf_model=hf, device=DEVICE,
                    center_writing_weights=True, center_unembed=True, fold_ln=True)
                del hf; torch.cuda.empty_cache()
                if clean_years is None:
                    clean_years = get_clean_years(model.tokenizer)
                    two_digit = get_two_digit_tokens(model.tokenizer)
                    log(f"    {len(clean_years)} clean years, {len(two_digit)} two-digit tokens")

                r = run_checkpoint(model, model.tokenizer, clean_years, two_digit, cfg["patch_layers"])
                results[name]["by_step"][key] = r
                log(f"    GT_acc={r['gt_acc']*100:5.1f}%  P(>)-P(<=)={r['base_diff_mean']:+.4f}  "
                    f"Δ={r['delta_mean']:+.4f} [{r['delta_ci'][0]:+.3f},{r['delta_ci'][1]:+.3f}]  "
                    f"(YY found {r['n_yy_found']}/{r['n']})")
                del model; torch.cuda.empty_cache(); gc.collect()
            except Exception as e:
                log(f"    ERROR: {str(e)[:100]}")
                results[name]["by_step"][key] = {"error": str(e)[:120]}
            json.dump(results, open(RESULTS_PATH, "w"), indent=2)

    # Summary.
    log("\n" + "=" * 60)
    log("GT CAUSAL SIGN FLIP SUMMARY")
    log("=" * 60)
    for cfg in MODELS:
        name = cfg["name"]
        by = results.get(name, {}).get("by_step", {})
        if not by: continue
        log(f"\n{name}:")
        steps = sorted(by.keys(), key=lambda x: int(x.split("_")[1]))
        dlds = []
        for sk in steps:
            r = by[sk]
            if not r or "error" in r: continue
            step = sk.split("_")[1]
            log(f"  step {step:>6}: GT_acc={r['gt_acc']*100:5.1f}%  "
                f"base={r['base_diff_mean']:+.3f}  Δ={r['delta_mean']:+.4f} "
                f"[{r['delta_ci'][0]:+.3f},{r['delta_ci'][1]:+.3f}]")
            dlds.append((int(step), r['delta_mean']))
        flip = None
        for i in range(1, len(dlds)):
            if dlds[i-1][1] > 0.02 and dlds[i][1] < -0.02:
                flip = dlds[i][0]; break
        log(f"  -> sign flip: {'step '+str(flip) if flip else 'check manually'}")


if __name__ == "__main__":
    main()
