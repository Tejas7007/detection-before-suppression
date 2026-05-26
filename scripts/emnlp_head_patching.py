"""
EMNLP Head-Specific Patching at S2
===================================
At a given training step, for each of 144 attention heads (12 layers x 12
heads in Pythia-160M), replace only that head's output at the S2 position
in IOI prompts with the head's output on the control prompt. Measure the
change in IOI logit difference. Heads whose patching produces a large
positive ΔLD are writing the S-promoting signal at S2 during the dip;
heads whose patching produces a large negative ΔLD are writing the
S-inhibition signal after recovery.

We run two steps:
  step 2000  - dip phase, base IOI acc ~32%; identifies S-promoting heads
  step 5000  - post-recovery, base IOI acc ~71%; identifies S-inhibition

The two-step comparison answers the mechanistic question: do the same
heads at S2 flip sign across training, or do different heads take over
the S2 computation as the circuit forms?

Per step: 144 heads x n=300 prompts each, plus baseline. Each per-head
intervention also reports a 10K-resample bootstrap 95% CI on ΔLD.

Output: results/emnlp_head_patching.json
Log:    results/emnlp_head_patching_log.txt
Runtime: ~5-10 minutes per step on A100.

Implementation notes
--------------------
1. We hook blocks.{L}.attn.hook_result, which exposes the per-head output
   AFTER W_O but BEFORE summing across heads to form the layer's
   contribution to the residual stream. Modifying value[:, p, H, :]
   changes only head (L, H)'s contribution at position p.
2. hook_result is disabled by default in TransformerLens to save memory.
   We enable it via model.set_use_attn_result(True) immediately after
   loading each checkpoint.
3. We cache the control's hook_result at all 12 layers in a single
   forward pass per template, then loop over heads doing one patched
   forward pass per head. Total: n_templates * (1 control + 1 baseline
   + 144 patched) = 1460 forward passes per step.
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
RETRAINED_REPO = "anonymous-research-sub/pythia-160m-retrained-seed42"
BASE_MODEL = "EleutherAI/pythia-160m-deduped"
N_LAYERS = 12
N_HEADS = 12
TEMPLATES = ALL_TEMPLATES[:10]
PROMPTS_PER_TEMPLATE = 30
SEED = 42
N_BOOTSTRAP = 10_000

# Both phases of the training trajectory. Step 2000 is the dip-floor
# checkpoint in the standard IOI protocol. Step 5000
# is the canonical post-recovery checkpoint where L8H9 dominates head
# ablation in prior analysis.
STEPS = [2000, 5000]

RESULTS_PATH = "results/emnlp_head_patching.json"


# --------------------------- Utilities ----------------------------

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_retrained(step):
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
    # CRITICAL: enable per-head outputs via hook_result. Without this,
    # the hook fires on an empty tensor and patching has no effect.
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


def get_single_token_names(tokenizer):
    ids = []
    for name in CANDIDATE_NAMES:
        toks = tokenizer.encode(" " + name, add_special_tokens=False)
        if len(toks) == 1:
            ids.append(int(toks[0]))
    return ids


def build_control(model, ds, single_name_ids, rng):
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


def head_patch_hook_factory(donor_result, s2_positions, target_head):
    """Return a hook that replaces head (current_layer, target_head)'s
    contribution at S2 in the IOI prompt with the donor's contribution
    from the control prompt. donor_result has shape
    [batch, seq, n_heads, d_model] from the control's forward pass.
    """
    def hook_fn(value, hook):
        # value: [batch, seq, n_heads, d_model]
        for i in range(value.shape[0]):
            p = int(s2_positions[i].item())
            if p >= 0:
                value[i, p, target_head, :] = donor_result[i, p, target_head, :]
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


# --------------------------- Per-step driver ----------------------

def run_step(model, step):
    """Run baseline + 144 per-head patches for a single checkpoint.

    Per template:
      1. Tokenize IOI and matched control.
      2. Cache control's hook_result at all 12 layers (one forward pass).
      3. Compute baseline IOI logit diffs.
      4. For each of 144 heads, run one patched forward pass and
         accumulate per-prompt LDs.

    After all templates: aggregate per-prompt arrays across templates,
    compute means, deltas, and bootstrap CIs per head.
    """
    rng = np.random.default_rng(SEED + 1)
    single_name_ids = get_single_token_names(model.tokenizer)

    # Accumulators: per-prompt LDs across all templates.
    base_lds = []
    # head_deltas[(L, H)] = list of per-prompt deltas (patched - base)
    head_deltas = {(L, H): [] for L in range(N_LAYERS) for H in range(N_HEADS)}

    n_template_done = 0
    t_template_start = time.time()

    for tmpl_idx, tmpl in enumerate(TEMPLATES):
        ds = IOIDataset(
            model=model, n_prompts=PROMPTS_PER_TEMPLATE,
            templates=[tmpl], symmetric=True, seed=SEED,
        )
        ioi_tokens, ctrl_tokens, s2_positions = build_control(
            model, ds, single_name_ids, rng,
        )
        io_ids = torch.tensor(ds.io_token_ids, dtype=torch.long, device=DEVICE)
        s_ids = torch.tensor(ds.s_token_ids, dtype=torch.long, device=DEVICE)

        # 1. Cache control's hook_result at all layers in a single pass.
        result_names = [f"blocks.{L}.attn.hook_result" for L in range(N_LAYERS)]
        donor_results = {}

        def make_cap(name):
            def fn(value, hook):
                donor_results[name] = value.detach()
                return value
            return fn

        with torch.no_grad():
            model.run_with_hooks(
                ctrl_tokens,
                fwd_hooks=[(n, make_cap(n)) for n in result_names],
            )

        # Sanity-check shape on first template: must be 4-d with N_HEADS
        # in the head dimension. If hook_result is not enabled, the
        # tensor will be the wrong shape (or zeros).
        if tmpl_idx == 0:
            first = donor_results[result_names[0]]
            if first.ndim != 4 or first.shape[2] != N_HEADS:
                raise RuntimeError(
                    f"hook_result has unexpected shape {tuple(first.shape)}. "
                    "Did set_use_attn_result(True) succeed?"
                )

        # 2. Baseline IOI LD.
        with torch.no_grad():
            base_logits = model(ioi_tokens)
        base = logit_diff_per_prompt(base_logits, io_ids, s_ids).cpu().numpy()
        base_lds.extend(base.tolist())

        # 3. For each head, patched LD.
        for L in range(N_LAYERS):
            donor_result_L = donor_results[result_names[L]]
            for H in range(N_HEADS):
                hook_fn = head_patch_hook_factory(donor_result_L, s2_positions, H)
                with torch.no_grad():
                    patched_logits = model.run_with_hooks(
                        ioi_tokens,
                        fwd_hooks=[(result_names[L], hook_fn)],
                    )
                patched = logit_diff_per_prompt(
                    patched_logits, io_ids, s_ids,
                ).cpu().numpy()
                deltas = (patched - base).tolist()
                head_deltas[(L, H)].extend(deltas)
                del patched_logits

        n_template_done += 1
        elapsed = time.time() - t_template_start
        per_tmpl = elapsed / n_template_done
        remaining = (len(TEMPLATES) - n_template_done) * per_tmpl
        log(
            f"  template {n_template_done}/{len(TEMPLATES)} done "
            f"({per_tmpl:.1f}s/template, ~{remaining:.0f}s remaining)"
        )

        del ioi_tokens, ctrl_tokens, base_logits, donor_results
        torch.cuda.empty_cache()

    # Aggregate.
    base_arr = np.asarray(base_lds)
    base_lo, base_hi = bootstrap_ci(base_arr)

    heads = []
    for L in range(N_LAYERS):
        for H in range(N_HEADS):
            d = np.asarray(head_deltas[(L, H)])
            lo, hi = bootstrap_ci(d)
            heads.append({
                "layer": L,
                "head": H,
                "n_prompts": int(d.shape[0]),
                "delta_ld_mean": float(d.mean()),
                "delta_ld_ci95": [lo, hi],
            })

    return {
        "step": step,
        "n_prompts": int(base_arr.shape[0]),
        "ioi_acc": float((base_arr > 0).mean()),
        "base_ld_mean": float(base_arr.mean()),
        "base_ld_ci95": [base_lo, base_hi],
        "heads": heads,
    }


# ------------------------------ Main ------------------------------

def main():
    os.makedirs("results", exist_ok=True)

    results = {"config": {
        "model": RETRAINED_REPO,
        "n_layers": N_LAYERS,
        "n_heads": N_HEADS,
        "templates": len(TEMPLATES),
        "prompts_per_template": PROMPTS_PER_TEMPLATE,
        "n_total": len(TEMPLATES) * PROMPTS_PER_TEMPLATE,
        "n_bootstrap": N_BOOTSTRAP,
        "steps": STEPS,
        "patch_position": "S2",
        "hook": "blocks.{L}.attn.hook_result",
    }, "by_step": {}}

    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH) as f:
                prev = json.load(f)
            if "by_step" in prev:
                results["by_step"] = prev["by_step"]
                log(f"Resuming. {len(results['by_step'])} steps already complete.")
        except Exception as e:
            log(f"Could not resume from existing file: {e}")

    t0 = time.time()
    for step in STEPS:
        key = f"step_{step}"
        if key in results["by_step"]:
            log(f"step={step}: cached, skipping.")
            continue

        log(f"=== step={step}: loading retrained checkpoint ===")
        try:
            model = load_retrained(step)
        except Exception as e:
            log(f"  FAILED to load step {step}: {e}")
            continue

        try:
            row = run_step(model, step)
        except Exception as e:
            import traceback
            log(f"  FAILED at step {step}: {e}")
            traceback.print_exc()
            del model
            torch.cuda.empty_cache()
            gc.collect()
            continue

        results["by_step"][key] = row

        # Print top-5 and bottom-5 heads inline so the user can see
        # the result without parsing the JSON.
        heads_sorted = sorted(row["heads"], key=lambda h: h["delta_ld_mean"], reverse=True)
        log(
            f"step={step}  acc={row['ioi_acc']:.3f}  "
            f"base_LD={row['base_ld_mean']:+.4f}  n={row['n_prompts']}"
        )
        log("  Top 5 S-promoting heads (positive delta = removing helps):")
        for h in heads_sorted[:5]:
            lo, hi = h["delta_ld_ci95"]
            log(
                f"    L{h['layer']:>2}H{h['head']:>2}  "
                f"delta_LD={h['delta_ld_mean']:+.4f}  "
                f"CI=[{lo:+.3f}, {hi:+.3f}]"
            )
        log("  Bottom 5 S-inhibiting heads (negative delta = removing hurts):")
        for h in heads_sorted[-5:]:
            lo, hi = h["delta_ld_ci95"]
            log(
                f"    L{h['layer']:>2}H{h['head']:>2}  "
                f"delta_LD={h['delta_ld_mean']:+.4f}  "
                f"CI=[{lo:+.3f}, {hi:+.3f}]"
            )

        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

        del model
        torch.cuda.empty_cache()
        gc.collect()

    elapsed = (time.time() - t0) / 60.0
    log(f"Done. Total elapsed: {elapsed:.1f} min. Output: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
