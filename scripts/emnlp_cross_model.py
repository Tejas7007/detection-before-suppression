"""
EMNLP Cross-Model Sign Flip Replication
========================================
Replicates the S2 activation patching intervention on two additional
models to show the sign flip is not specific to retrained Pythia-160M:

  1. Stanford GPT-2 Small (alias seed): different architecture family
     (learned absolute position embeddings, GPT-2 tokenizer), 12 layers.
     Dip floor at step 1500 (acc ~10%), recovery by step 10000.

  2. PolyPythia seed1 (computational seed): same architecture as Pythia
     but different initialization seed. Dip floor at step 2000 (acc 32%),
     sharp recovery to acc 80% by step 3000.

Per (model, checkpoint), we measure (n=300 prompts, 10 templates x 30):
  - Baseline IOI logit difference and accuracy
  - Patched LD: replace S2 residual at layers 3-5 with control
  - Bootstrap 95% CI on ΔLD (10,000 resamples)

We deliberately patch the same layer range (3-5) used for retrained
Pythia. This is a pre-registered choice for fair comparison; if Stanford's
S-bias localizes at a different depth, the patching effect will be
attenuated, which is itself informative.

We do NOT measure L8H9 attention to S2 here. Prior analysis showed
the dominant S-inhibition head differs across architectures (Stanford's
top head is at a different (L,H) with a 2.4:1 S-suppression-to-IO-copy
ratio vs Pythia's 10.8:1), so a single per-head measurement would not
be cross-model comparable.

Output: results/emnlp_cross_model.json
Log:    results/emnlp_cross_model_log.txt
Runtime: ~10-15 minutes for 10 total checkpoints.
"""

import os
import gc
import json
import time
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer

try:
    from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES, CANDIDATE_NAMES
except ImportError:
    sys.path.insert(0, "/workspace/ioi-sign-flip/src")
    from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES, CANDIDATE_NAMES


# ----------------------------- Config -----------------------------

DEVICE = "cuda"
PATCH_LAYERS = [3, 4, 5]
TEMPLATES = ALL_TEMPLATES[:10]
PROMPTS_PER_TEMPLATE = 30
SEED = 42
N_BOOTSTRAP = 10_000

# Two cross-model configurations. Checkpoint sets chosen from existing
# sweep JSONs to span pre-dip / dip floor / mid-dip / recovery / mature.
MODELS = [
    {
        "name": "stanford_alias",
        "loader": "stanford_direct",
        "hf_repo": "stanford-crfm/alias-gpt2-small-x21",
        "revision_fmt": "checkpoint-{}",
        # Dip is wide: acc < 0.5 from step 900 to step 5000, floor at 1500.
        "checkpoints": [100, 1500, 3000, 10000, 100000],
    },
    {
        "name": "polypythia_seed1",
        "loader": "pythia_wrapped",
        "hf_repo": "EleutherAI/pythia-160m-seed1",
        "revision_fmt": "step{}",
        # Sharp dip: acc=0.323 at step 2000, snaps to 0.797 at step 3000.
        # Mirrors the Pythia sign-flip table for direct comparison.
        "checkpoints": [1000, 2000, 3000, 5000, 143000],
    },
]

RESULTS_PATH = "results/emnlp_cross_model.json"


# --------------------------- Utilities ----------------------------

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_stanford(repo, revision):
    """Stanford-CRFM models load directly into HookedTransformer."""
    return HookedTransformer.from_pretrained(
        repo, device=DEVICE, revision=revision,
    )


def load_pythia_wrapped(repo, revision):
    """PolyPythia variants: load HF model, then wrap in HookedTransformer
    using the deduped Pythia-160M architecture template."""
    hf = AutoModelForCausalLM.from_pretrained(
        repo, revision=revision, torch_dtype=torch.float32,
    )
    model = HookedTransformer.from_pretrained(
        "EleutherAI/pythia-160m-deduped",
        hf_model=hf,
        device=DEVICE,
        center_writing_weights=True,
        center_unembed=True,
        fold_ln=True,
    )
    del hf
    torch.cuda.empty_cache()
    return model


def load_model(cfg, step):
    revision = cfg["revision_fmt"].format(step)
    if cfg["loader"] == "stanford_direct":
        return load_stanford(cfg["hf_repo"], revision)
    elif cfg["loader"] == "pythia_wrapped":
        return load_pythia_wrapped(cfg["hf_repo"], revision)
    else:
        raise ValueError(f"Unknown loader: {cfg['loader']}")


def find_s2_position(token_row, s_token_id):
    seen = 0
    for j in range(1, token_row.shape[0]):  # skip BOS
        if int(token_row[j].item()) == int(s_token_id):
            seen += 1
            if seen == 2:
                return j
    return -1


def get_single_token_names(tokenizer):
    """Filter CANDIDATE_NAMES to names that tokenize to a single id with
    leading space, matching the protocol in causal_intervention.py."""
    ids = []
    for name in CANDIDATE_NAMES:
        toks = tokenizer.encode(" " + name, add_special_tokens=False)
        if len(toks) == 1:
            ids.append(int(toks[0]))
    return ids


def build_control(model, ds, single_name_ids, rng):
    """For each IOI prompt, return tokenized IOI and matched control where
    the S2 token is replaced with a third single-token name."""
    ioi_tokens = model.to_tokens(ds.prompts).to(DEVICE)
    n = ioi_tokens.shape[0]

    s2_positions = []
    for i in range(n):
        s2 = find_s2_position(ioi_tokens[i].cpu(), ds.s_token_ids[i])
        s2_positions.append(s2)
    s2_positions = torch.tensor(s2_positions, dtype=torch.long, device=DEVICE)

    ctrl_tokens = ioi_tokens.clone()
    for i in range(n):
        io_id = int(ds.io_token_ids[i])
        s_id = int(ds.s_token_ids[i])
        pool = [t for t in single_name_ids if t != io_id and t != s_id]
        if not pool:
            continue
        s2_pos = int(s2_positions[i].item())
        if s2_pos > 0:
            ctrl_tokens[i, s2_pos] = int(rng.choice(pool))

    return ioi_tokens, ctrl_tokens, s2_positions


def logit_diff_per_prompt(logits, io_ids, s_ids):
    last = logits[:, -1, :]
    idx = torch.arange(last.shape[0], device=last.device)
    return last[idx, io_ids] - last[idx, s_ids]


def s2_patch_hook_factory(donor_cache, s2_positions, layer):
    donor_act = donor_cache[f"blocks.{layer}.hook_resid_post"]

    def hook_fn(value, hook):
        for i in range(value.shape[0]):
            p = int(s2_positions[i].item())
            if p >= 0:
                value[i, p, :] = donor_act[i, p, :]
        return value

    return hook_fn


def bootstrap_ci(values, n_resamples=N_BOOTSTRAP, alpha=0.05, seed=SEED):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    n = arr.shape[0]
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = arr[idx].mean(axis=1)
    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


# --------------------------- Per-checkpoint -----------------------

def run_checkpoint(model, single_name_ids):
    """Run one model+checkpoint through the full patching protocol."""
    rng = np.random.default_rng(SEED + 1)
    base_lds = []
    patched_lds = []
    deltas = []

    for tmpl in TEMPLATES:
        ds = IOIDataset(
            model=model, n_prompts=PROMPTS_PER_TEMPLATE,
            templates=[tmpl], symmetric=True, seed=SEED,
        )
        ioi_tokens, ctrl_tokens, s2_positions = build_control(
            model, ds, single_name_ids, rng,
        )
        io_ids = torch.tensor(ds.io_token_ids, dtype=torch.long, device=DEVICE)
        s_ids = torch.tensor(ds.s_token_ids, dtype=torch.long, device=DEVICE)

        # Cache control activations at the patch layers.
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

        # Patched.
        hooks = [
            (f"blocks.{L}.hook_resid_post",
             s2_patch_hook_factory(donor, s2_positions, L))
            for L in PATCH_LAYERS
        ]
        with torch.no_grad():
            patched_logits = model.run_with_hooks(ioi_tokens, fwd_hooks=hooks)
        patched = logit_diff_per_prompt(patched_logits, io_ids, s_ids).cpu().numpy()

        base_lds.extend(base.tolist())
        patched_lds.extend(patched.tolist())
        deltas.extend((patched - base).tolist())

        del ioi_tokens, ctrl_tokens, base_logits, patched_logits, donor
        torch.cuda.empty_cache()

    base_arr = np.asarray(base_lds)
    patched_arr = np.asarray(patched_lds)
    deltas_arr = np.asarray(deltas)

    base_lo, base_hi = bootstrap_ci(base_arr)
    delta_lo, delta_hi = bootstrap_ci(deltas_arr)

    return {
        "n_prompts": int(len(base_arr)),
        "ioi_acc": float((base_arr > 0).mean()),
        "base_ld_mean": float(base_arr.mean()),
        "base_ld_ci95": [base_lo, base_hi],
        "patched_ld_mean": float(patched_arr.mean()),
        "delta_ld_mean": float(deltas_arr.mean()),
        "delta_ld_ci95": [delta_lo, delta_hi],
    }


# ------------------------------ Main ------------------------------

def main():
    os.makedirs("results", exist_ok=True)

    results = {"config": {
        "patch_layers": PATCH_LAYERS,
        "templates": len(TEMPLATES),
        "prompts_per_template": PROMPTS_PER_TEMPLATE,
        "n_total": len(TEMPLATES) * PROMPTS_PER_TEMPLATE,
        "n_bootstrap": N_BOOTSTRAP,
        "models": [{"name": m["name"], "hf_repo": m["hf_repo"],
                    "checkpoints": m["checkpoints"]} for m in MODELS],
    }, "by_model": {}}

    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH) as f:
                prev = json.load(f)
            if "by_model" in prev:
                results["by_model"] = prev["by_model"]
                done = sum(len(v) for v in results["by_model"].values())
                log(f"Resuming. {done} (model, step) pairs already complete.")
        except Exception as e:
            log(f"Could not resume from existing file: {e}")

    t0 = time.time()
    total_pairs = sum(len(m["checkpoints"]) for m in MODELS)
    completed = 0

    for cfg in MODELS:
        model_name = cfg["name"]
        results["by_model"].setdefault(model_name, {})

        log(f"=== Model: {model_name} ({cfg['hf_repo']}) ===")

        for step in cfg["checkpoints"]:
            completed += 1
            key = f"step_{step}"
            if key in results["by_model"][model_name]:
                log(f"  [{completed}/{total_pairs}] {model_name}/{step}: cached, skipping")
                continue

            log(f"  [{completed}/{total_pairs}] {model_name}/{step}: loading")
            try:
                model = load_model(cfg, step)
            except Exception as e:
                log(f"    FAILED to load: {e}")
                continue

            try:
                single_name_ids = get_single_token_names(model.tokenizer)
                row = run_checkpoint(model, single_name_ids)
            except Exception as e:
                log(f"    FAILED at run: {e}")
                del model
                torch.cuda.empty_cache()
                gc.collect()
                continue

            results["by_model"][model_name][key] = row
            log(
                f"    step={step}  acc={row['ioi_acc']:.3f}  "
                f"base_LD={row['base_ld_mean']:+.4f}  "
                f"ΔLD={row['delta_ld_mean']:+.4f} "
                f"[{row['delta_ld_ci95'][0]:+.3f}, {row['delta_ld_ci95'][1]:+.3f}]"
            )

            with open(RESULTS_PATH, "w") as f:
                json.dump(results, f, indent=2)

            del model
            torch.cuda.empty_cache()
            gc.collect()

    elapsed = (time.time() - t0) / 60.0
    log(f"Done. Total elapsed: {elapsed:.1f} min. Output: {RESULTS_PATH}")

    # Summary table for at-a-glance review.
    log("")
    log("=== Sign flip summary ===")
    for model_name, by_step in results["by_model"].items():
        log(f"{model_name}:")
        for key in sorted(by_step.keys(), key=lambda k: int(k.split('_')[1])):
            r = by_step[key]
            step = key.split('_')[1]
            sign = "+" if r["delta_ld_mean"] > 0 else "-"
            log(
                f"  step={step:>6}  acc={r['ioi_acc']:.3f}  "
                f"base_LD={r['base_ld_mean']:+.4f}  "
                f"ΔLD={r['delta_ld_mean']:+.4f}  "
                f"(sign={sign})"
            )


if __name__ == "__main__":
    main()
