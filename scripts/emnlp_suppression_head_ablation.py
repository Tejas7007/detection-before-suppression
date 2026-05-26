"""
Suppression-head ablation across training
==========================================
Closes the detection/suppression rigor asymmetry. We measure detection
(induction) causally; this gives suppression the same causal treatment.

Protocol:
  1. At maturity (step 143000), scan every head: mean-ablate it and
     measure the drop in IOI logit difference. The head with the largest
     drop is the dominant S-inhibition (suppression) head.
  2. Ablate that SAME head at every checkpoint and record the ablation
     effect (ΔLD) across training.

Prediction (detection-before-suppression):
  - At the dip, ablating the suppression head does ~nothing (it isn't
    doing suppression yet).
  - The ablation effect grows (more negative) toward maturity as the
    head takes on its suppression role.
  This is the causal complement to the induction measurement: detection
  is present at the dip; suppression's causal force appears only later.

Curated 10-template protocol, 160M and 410M.
Output: results/emnlp_suppression_head_ablation.json
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
PPT = 30
RESULTS_PATH = "results/emnlp_suppression_head_ablation.json"

MODELS = [
    {"name": "pythia-160m", "repo": "EleutherAI/pythia-160m-deduped",
     "checkpoints": [1000, 2000, 3000, 4000, 5000, 8000, 13000, 33000, 143000]},
    {"name": "pythia-410m", "repo": "EleutherAI/pythia-410m-deduped",
     "checkpoints": [1000, 2000, 3000, 5000, 8000, 13000, 30000, 143000]},
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


def build_ioi(model):
    """Build pooled IOI tokens across the 10 templates."""
    all_tokens, all_io, all_s = [], [], []
    for tmpl in TEMPLATES:
        ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl],
                        symmetric=True, seed=SEED)
        toks = model.to_tokens(ds.prompts)
        # Pad to common length handled by to_tokens per-template; eval per-template.
        all_tokens.append(toks)
        all_io.append(torch.tensor(ds.io_token_ids))
        all_s.append(torch.tensor(ds.s_token_ids))
    return all_tokens, all_io, all_s


def ld_with_optional_ablation(model, tokens_list, io_list, s_list,
                              ablate_layer=None, ablate_head=None):
    """Mean-ablate (layer,head) at hook_z and return pooled per-prompt LD."""
    lds = []

    def hook_z(value, hook):
        # value: [batch, pos, head, d_head]
        h = ablate_head
        mean_z = value[:, :, h, :].mean(dim=(0, 1), keepdim=True)
        value[:, :, h, :] = mean_z
        return value

    for toks, io_ids, s_ids in zip(tokens_list, io_list, s_list):
        toks = toks.to(DEVICE)
        io_ids = io_ids.to(DEVICE); s_ids = s_ids.to(DEVICE)
        n = toks.shape[0]; idx = torch.arange(n, device=DEVICE)
        if ablate_layer is None:
            with torch.no_grad():
                logits = model(toks)
        else:
            with torch.no_grad():
                logits = model.run_with_hooks(
                    toks, fwd_hooks=[(f"blocks.{ablate_layer}.attn.hook_z", hook_z)])
        last = logits[:, -1, :]
        ld = (last[idx, io_ids] - last[idx, s_ids]).detach().cpu().numpy()
        lds.extend(ld.tolist())
    return np.array(lds)


def find_suppression_head(model, tokens_list, io_list, s_list):
    """At maturity: scan all heads, return the one whose mean-ablation
    most REDUCES IOI LD (largest negative ΔLD)."""
    base = ld_with_optional_ablation(model, tokens_list, io_list, s_list)
    base_mean = float(base.mean())
    best = None; best_drop = 0.0; scores = {}
    for L in range(model.cfg.n_layers):
        for H in range(model.cfg.n_heads):
            abl = ld_with_optional_ablation(model, tokens_list, io_list, s_list,
                                            ablate_layer=L, ablate_head=H)
            drop = float(abl.mean()) - base_mean  # negative = ablation hurts
            scores[f"L{L}H{H}"] = drop
            if drop < best_drop:
                best_drop = drop; best = (L, H)
    return best, best_drop, base_mean, scores


def main():
    os.makedirs("results", exist_ok=True)
    results = {}
    if os.path.exists(RESULTS_PATH):
        try: results = json.load(open(RESULTS_PATH))
        except: pass

    for cfg in MODELS:
        name = cfg["name"]
        if name in results and results[name].get("by_step"):
            log(f"SKIP {name} (done)"); continue
        log("=" * 60); log(name); log("=" * 60)

        # Step 1: identify suppression head at maturity.
        log("  Identifying suppression head at maturity (step 143000)...")
        hf = AutoModelForCausalLM.from_pretrained(
            cfg["repo"], revision="step143000", torch_dtype=torch.float32)
        model = HookedTransformer.from_pretrained(
            cfg["repo"], hf_model=hf, device=DEVICE,
            center_writing_weights=True, center_unembed=True, fold_ln=True)
        del hf; torch.cuda.empty_cache()

        tokens_list, io_list, s_list = build_ioi(model)
        (sl, sh), drop, base_mature, scores = find_suppression_head(
            model, tokens_list, io_list, s_list)
        log(f"  Suppression head = L{sl}H{sh}  (ablation ΔLD={drop:+.3f} at maturity)")
        # Top 5 heads for the record.
        top5 = sorted(scores.items(), key=lambda x: x[1])[:5]
        log(f"  Top-5 suppression heads: {top5}")
        del model; torch.cuda.empty_cache(); gc.collect()

        results[name] = {
            "repo": cfg["repo"], "suppression_head": [sl, sh],
            "mature_ablation_delta": drop, "top5": top5, "by_step": {},
        }

        # Step 2: ablate that head across checkpoints.
        for step in cfg["checkpoints"]:
            log(f"  step {step}: ablating L{sl}H{sh}")
            try:
                hf = AutoModelForCausalLM.from_pretrained(
                    cfg["repo"], revision=f"step{step}", torch_dtype=torch.float32)
                model = HookedTransformer.from_pretrained(
                    cfg["repo"], hf_model=hf, device=DEVICE,
                    center_writing_weights=True, center_unembed=True, fold_ln=True)
                del hf; torch.cuda.empty_cache()
                tl, iol, sl_ = build_ioi(model)
                base = ld_with_optional_ablation(model, tl, iol, sl_)
                abl = ld_with_optional_ablation(model, tl, iol, sl_,
                                                ablate_layer=sl, ablate_head=sh)
                delta = abl - base
                lo, hi = bootstrap_ci(delta)
                results[name]["by_step"][f"step_{step}"] = {
                    "base_ld": float(base.mean()),
                    "ablated_ld": float(abl.mean()),
                    "ablation_delta": float(delta.mean()),
                    "ablation_ci": [lo, hi],
                    "ioi_acc": float((base > 0).mean()),
                }
                log(f"    base_LD={base.mean():+.3f}  ablated_LD={abl.mean():+.3f}  "
                    f"ablation ΔLD={delta.mean():+.4f} [{lo:+.3f},{hi:+.3f}]")
                del model; torch.cuda.empty_cache(); gc.collect()
            except Exception as e:
                log(f"    ERROR: {str(e)[:90]}")
                results[name]["by_step"][f"step_{step}"] = {"error": str(e)[:100]}
            json.dump(results, open(RESULTS_PATH, "w"), indent=2)

    # Summary.
    log("\n" + "=" * 60)
    log("SUPPRESSION-HEAD ABLATION ACROSS TRAINING")
    log("=" * 60)
    for cfg in MODELS:
        name = cfg["name"]
        r = results.get(name, {})
        if not r.get("by_step"): continue
        sh = r["suppression_head"]
        log(f"\n{name}  (suppression head L{sh[0]}H{sh[1]}):")
        log(f"  {'Step':>7} {'IOI%':>6} {'ablation ΔLD':>14}")
        for sk in sorted(r["by_step"], key=lambda x: int(x.split("_")[1])):
            v = r["by_step"][sk]
            if "ablation_delta" not in v: continue
            log(f"  {sk.split('_')[1]:>7} {v['ioi_acc']*100:>5.1f}% {v['ablation_delta']:>+14.4f}")
        log("  (Prediction: ~0 at dip, grows negative toward maturity)")


if __name__ == "__main__":
    main()
