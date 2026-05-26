"""
Stanford GPT-2 Small IOI Analysis
=================================
Tests whether the IOI performance dip replicates in Stanford CRFM's
GPT-2 Small models (trained with different seeds than Pythia, on different data).

Uses stanford-crfm/alias-gpt2-small-x21 (609 checkpoints available).

Part 1: Accuracy sweep (37 checkpoints) — does the dip happen?
Part 2: Mechanistic dive (10 key checkpoints) — same mechanism?

Saves incrementally to results/stanford_gpt2_ioi.json
"""

import torch
import json
import os
import time
import traceback
import numpy as np
from collections import Counter

from transformer_lens import HookedTransformer

# Try importing IOIDataset
try:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
    from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES
except ImportError:
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
        from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES
    except ImportError:
        print("ERROR: Cannot import IOIDataset. Make sure local src is available.")
        raise

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "stanford-crfm/alias-gpt2-small-x21"
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

# Dense early, sparse late
SWEEP_CHECKPOINTS = [
    0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
    150, 200, 250, 300, 400, 500, 600, 700, 800, 900,
    1000, 1100, 1200, 1500, 2000, 2500, 3000,
    4000, 5000, 8000, 10000,
    20000, 50000, 100000, 200000, 400000,
]

# Key checkpoints for deep mechanistic analysis
DEEP_CHECKPOINTS = [0, 100, 500, 1000, 2000, 3000, 5000, 10000, 100000, 400000]

TEMPLATES = ALL_TEMPLATES[:15]
PPT = 20
SEED = 42
TAU = 0.02

RESULTS_FILE = "results/stanford_gpt2_ioi.json"

# ============================================================
# UTILITIES
# ============================================================

def save_results(results):
    os.makedirs("results", exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)


def load_model(step):
    revision = "checkpoint-%d" % step
    model = HookedTransformer.from_pretrained(
        MODEL_NAME,
        device=DEVICE,
        revision=revision,
    )
    return model


def empty_cache():
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    elif DEVICE == "mps":
        torch.mps.empty_cache()


# ============================================================
# PART 1: Accuracy Sweep
# ============================================================

def run_part1(results):
    print("\n" + "=" * 60)
    print("  PART 1: IOI Accuracy Sweep (%d checkpoints)" % len(SWEEP_CHECKPOINTS))
    print("  Model: %s" % MODEL_NAME)
    print("  Device: %s" % DEVICE)
    print("=" * 60)

    if "part1_sweep" not in results:
        results["part1_sweep"] = {}

    for step in SWEEP_CHECKPOINTS:
        step_key = "step_%d" % step

        # Skip if already done
        if step_key in results["part1_sweep"]:
            print("  Step %d already done, skipping" % step)
            continue

        print("\n--- Step %d ---" % step)
        try:
            model = load_model(step)
        except Exception as e:
            print("  FAILED to load: %s" % str(e))
            continue

        all_lds = []
        io_ranks = []
        s_ranks = []
        io_probs = []
        s_probs = []
        top1_is_io = 0
        top1_is_s = 0
        top1_other_tokens = []
        total = 0
        errors = 0

        for tmpl in TEMPLATES:
            try:
                ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl],
                                symmetric=True, seed=SEED)
                tokens = model.to_tokens(ds.prompts).to(DEVICE)
                io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
                s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)

                logits = model(tokens)
                last = logits[:, -1, :].float()
                probs = torch.softmax(last, dim=-1)

                for i in range(len(io_ids)):
                    total += 1
                    io_logit = last[i, io_ids[i]].item()
                    s_logit = last[i, s_ids[i]].item()
                    ld = io_logit - s_logit
                    all_lds.append(ld)

                    # Ranks
                    sorted_idx = last[i].argsort(descending=True)
                    io_rank = (sorted_idx == io_ids[i]).nonzero(as_tuple=True)[0].item()
                    s_rank = (sorted_idx == s_ids[i]).nonzero(as_tuple=True)[0].item()
                    io_ranks.append(io_rank)
                    s_ranks.append(s_rank)

                    # Probabilities
                    io_probs.append(probs[i, io_ids[i]].item())
                    s_probs.append(probs[i, s_ids[i]].item())

                    # Top-1
                    top_tok = sorted_idx[0].item()
                    if top_tok == io_ids[i].item():
                        top1_is_io += 1
                    elif top_tok == s_ids[i].item():
                        top1_is_s += 1
                    else:
                        decoded = model.tokenizer.decode([top_tok]).strip()
                        top1_other_tokens.append(decoded)

            except Exception as e:
                errors += 1
                if errors <= 3:
                    print("  Template error: %s" % str(e)[:80])
                continue

        if total == 0:
            print("  No valid examples, skipping")
            del model
            empty_cache()
            continue

        lds = np.array(all_lds)
        accuracy = float((lds > 0).mean())
        top_others = Counter(top1_other_tokens).most_common(5)

        step_result = {
            "n_examples": total,
            "accuracy": round(accuracy, 4),
            "mean_ld": round(float(lds.mean()), 4),
            "std_ld": round(float(lds.std()), 4),
            "pct_s_preferred": round(float((lds < 0).mean()), 4),
            "median_io_rank": float(np.median(io_ranks)),
            "median_s_rank": float(np.median(s_ranks)),
            "mean_io_prob": round(float(np.mean(io_probs)), 6),
            "mean_s_prob": round(float(np.mean(s_probs)), 6),
            "pct_io_top1": round(top1_is_io / total, 4),
            "pct_s_top1": round(top1_is_s / total, 4),
            "pct_other_top1": round(1 - top1_is_io/total - top1_is_s/total, 4),
            "top_other": [{"token": t, "count": c} for t, c in top_others],
        }

        results["part1_sweep"][step_key] = step_result
        save_results(results)

        print("  Acc=%.3f, LD=%.4f, IO_rank=%d, S_rank=%d, IO_prob=%.4f%%, S_prob=%.4f%%" % (
            accuracy, lds.mean(),
            np.median(io_ranks), np.median(s_ranks),
            np.mean(io_probs) * 100, np.mean(s_probs) * 100))
        print("  Top-1: IO=%.1f%%, S=%.1f%%, other=%.1f%%" % (
            top1_is_io/total*100, top1_is_s/total*100,
            (1-top1_is_io/total-top1_is_s/total)*100))
        if top_others:
            print("  Top other: %s" % str(top_others[:3]))

        del model
        empty_cache()

    print("\nPART 1 COMPLETE")


# ============================================================
# PART 2: Mechanistic Deep Dive
# ============================================================

def run_part2(results):
    print("\n" + "=" * 60)
    print("  PART 2: Mechanistic Deep Dive (%d checkpoints)" % len(DEEP_CHECKPOINTS))
    print("=" * 60)

    if "part2_mechanism" not in results:
        results["part2_mechanism"] = {}

    for step in DEEP_CHECKPOINTS:
        step_key = "step_%d" % step

        if step_key in results["part2_mechanism"]:
            print("  Step %d already done, skipping" % step)
            continue

        print("\n--- Step %d ---" % step)
        try:
            model = load_model(step)
        except Exception as e:
            print("  FAILED: %s" % str(e))
            continue

        n_layers = model.cfg.n_layers
        n_heads = model.cfg.n_heads

        # Find top head by delta_ioi (ablate each head, measure effect)
        head_deltas = {}
        baseline_acc_list = []

        # First get baseline
        all_base_lds = []
        template_data = []

        for tmpl in TEMPLATES[:10]:
            try:
                ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl],
                                symmetric=True, seed=SEED)
                tokens = model.to_tokens(ds.prompts).to(DEVICE)
                io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
                s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)

                logits = model(tokens)
                last = logits[:, -1, :]

                for i in range(len(io_ids)):
                    ld = last[i, io_ids[i]].item() - last[i, s_ids[i]].item()
                    all_base_lds.append(ld)

                template_data.append((tokens, io_ids, s_ids))
            except Exception as e:
                continue

        if not template_data:
            print("  No valid templates, skipping")
            del model
            empty_cache()
            continue

        base_lds = np.array(all_base_lds)
        base_acc = float((base_lds > 0).mean())

        # Ablate each head
        for layer in range(n_layers):
            for head in range(n_heads):
                def hook_fn(value, hook, h=head):
                    value[:, :, h, :] = 0.0
                    return value

                hook_name = "blocks.%d.attn.hook_z" % layer
                abl_lds = []

                for tokens, io_ids, s_ids in template_data:
                    logits = model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
                    last = logits[:, -1, :]
                    for i in range(len(io_ids)):
                        ld = last[i, io_ids[i]].item() - last[i, s_ids[i]].item()
                        abl_lds.append(ld)

                abl_acc = float((np.array(abl_lds) > 0).mean())
                delta = abl_acc - base_acc
                head_name = "L%dH%d" % (layer, head)
                head_deltas[head_name] = round(delta, 4)

            if (layer + 1) % 4 == 0:
                print("    Layer %d/%d done" % (layer + 1, n_layers))

        # Find top head (most negative delta = most important for IOI)
        sorted_heads = sorted(head_deltas.items(), key=lambda x: x[1])
        top_head_name = sorted_heads[0][0]
        top_head_delta = sorted_heads[0][1]
        top_layer = int(top_head_name.split("H")[0][1:])
        top_head_idx = int(top_head_name.split("H")[1])

        print("  Base acc: %.3f" % base_acc)
        print("  Top head: %s (delta=%.4f)" % (top_head_name, top_head_delta))
        print("  Top 5 heads:")
        for h, d in sorted_heads[:5]:
            print("    %s: %+.4f" % (h, d))

        # Get attention pattern of top head
        attn_to_io = []
        attn_to_s1 = []
        attn_to_s2 = []

        for tokens, io_ids, s_ids in template_data:
            _, cache = model.run_with_cache(tokens, remove_batch_dim=False)
            attn = cache["blocks.%d.attn.hook_pattern" % top_layer]
            final_pos = tokens.shape[1] - 1

            for i in range(tokens.shape[0]):
                io_tok = io_ids[i].item()
                s_tok = s_ids[i].item()
                io_pos = -1
                s1_pos = -1
                s2_pos = -1
                s_count = 0
                for j in range(1, tokens.shape[1]):
                    if tokens[i, j].item() == io_tok and io_pos == -1:
                        io_pos = j
                    if tokens[i, j].item() == s_tok:
                        s_count += 1
                        if s_count == 1: s1_pos = j
                        elif s_count == 2: s2_pos = j
                if io_pos > 0:
                    attn_to_io.append(attn[i, top_head_idx, final_pos, io_pos].item())
                if s1_pos > 0:
                    attn_to_s1.append(attn[i, top_head_idx, final_pos, s1_pos].item())
                if s2_pos > 0:
                    attn_to_s2.append(attn[i, top_head_idx, final_pos, s2_pos].item())

            del cache
            empty_cache()

        # Output projection of top head
        W_U = model.W_U
        W_O = model.W_O
        io_projections = []
        s_projections = []

        for tokens, io_ids, s_ids in template_data:
            _, cache = model.run_with_cache(tokens, remove_batch_dim=False)
            z = cache["blocks.%d.attn.hook_z" % top_layer][:, -1, top_head_idx, :]
            head_out = z @ W_O[top_layer, top_head_idx]

            for i in range(len(io_ids)):
                io_dir = W_U[:, io_ids[i].item()]
                s_dir = W_U[:, s_ids[i].item()]
                io_projections.append(torch.dot(head_out[i], io_dir).item())
                s_projections.append(torch.dot(head_out[i], s_dir).item())

            del cache
            empty_cache()

        step_result = {
            "base_accuracy": round(base_acc, 4),
            "top_head": top_head_name,
            "top_head_delta": top_head_delta,
            "top_5_heads": [{"head": h, "delta": d} for h, d in sorted_heads[:5]],
            "bottom_5_heads": [{"head": h, "delta": d} for h, d in sorted_heads[-5:]],
            "top_head_attention": {
                "to_IO": round(float(np.mean(attn_to_io)), 4) if attn_to_io else 0,
                "to_S1": round(float(np.mean(attn_to_s1)), 4) if attn_to_s1 else 0,
                "to_S2": round(float(np.mean(attn_to_s2)), 4) if attn_to_s2 else 0,
            },
            "top_head_projection": {
                "mean_io": round(float(np.mean(io_projections)), 4) if io_projections else 0,
                "mean_s": round(float(np.mean(s_projections)), 4) if s_projections else 0,
                "io_minus_s": round(float(np.mean(io_projections)) - float(np.mean(s_projections)), 4) if io_projections else 0,
            },
        }

        results["part2_mechanism"][step_key] = step_result
        save_results(results)

        if attn_to_io:
            print("  %s attention: IO=%.4f, S1=%.4f, S2=%.4f" % (
                top_head_name,
                np.mean(attn_to_io), np.mean(attn_to_s1), np.mean(attn_to_s2)))
        if io_projections:
            print("  %s projection: IO=%.4f, S=%.4f, diff=%.4f" % (
                top_head_name,
                np.mean(io_projections), np.mean(s_projections),
                np.mean(io_projections) - np.mean(s_projections)))

        del model
        empty_cache()

    print("\nPART 2 COMPLETE")


# ============================================================
# PART 3: Second seed for data ordering test
# ============================================================

def run_part3(results):
    """Run quick dip test on a SECOND Stanford GPT-2 model (different seed)."""
    print("\n" + "=" * 60)
    print("  PART 3: Second Seed (battlestar) — Data Ordering Test")
    print("=" * 60)

    model_name2 = "stanford-crfm/battlestar-gpt2-small-x49"

    if "part3_second_seed" not in results:
        results["part3_second_seed"] = {}

    # Only test key checkpoints for the second seed
    key_steps = [0, 100, 500, 1000, 2000, 3000, 5000, 10000, 100000, 400000]

    for step in key_steps:
        step_key = "step_%d" % step

        if step_key in results["part3_second_seed"]:
            print("  Step %d already done, skipping" % step)
            continue

        print("\n--- Step %d ---" % step)
        try:
            model = HookedTransformer.from_pretrained(
                model_name2, device=DEVICE,
                revision="checkpoint-%d" % step,
            )
        except Exception as e:
            print("  FAILED: %s" % str(e))
            continue

        all_lds = []
        total = 0

        for tmpl in TEMPLATES:
            try:
                ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl],
                                symmetric=True, seed=SEED)
                tokens = model.to_tokens(ds.prompts).to(DEVICE)
                io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
                s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)

                logits = model(tokens)
                last = logits[:, -1, :].float()

                for i in range(len(io_ids)):
                    total += 1
                    ld = last[i, io_ids[i]].item() - last[i, s_ids[i]].item()
                    all_lds.append(ld)
            except:
                continue

        if total == 0:
            del model
            empty_cache()
            continue

        lds = np.array(all_lds)
        accuracy = float((lds > 0).mean())

        results["part3_second_seed"][step_key] = {
            "model": model_name2,
            "accuracy": round(accuracy, 4),
            "mean_ld": round(float(lds.mean()), 4),
            "pct_s_preferred": round(float((lds < 0).mean()), 4),
            "n_examples": total,
        }
        save_results(results)

        print("  Acc=%.3f, LD=%.4f, pct_S=%.1f%%" % (accuracy, lds.mean(), (lds < 0).mean() * 100))

        del model
        empty_cache()

    print("\nPART 3 COMPLETE")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  STANFORD GPT-2 IOI ANALYSIS")
    print("  Model: %s" % MODEL_NAME)
    print("  Device: %s" % DEVICE)
    print("  Started: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        print("  Loaded existing results")
    else:
        results = {"model": MODEL_NAME, "architecture": "GPT-2 Small (12L, 12H, 124M params)"}

    t0 = time.time()

    try:
        run_part1(results)
    except Exception as e:
        print("PART 1 FAILED: %s" % str(e))
        traceback.print_exc()

    try:
        run_part2(results)
    except Exception as e:
        print("PART 2 FAILED: %s" % str(e))
        traceback.print_exc()

    try:
        run_part3(results)
    except Exception as e:
        print("PART 3 FAILED: %s" % str(e))
        traceback.print_exc()

    elapsed = time.time() - t0
    results["total_time_seconds"] = round(elapsed, 1)

    save_results(results)
    print("\n" + "=" * 60)
    print("  ALL PARTS COMPLETE")
    print("  Total time: %.0f seconds (%.1f hours)" % (elapsed, elapsed / 3600))
    print("  Results: %s" % RESULTS_FILE)
    print("=" * 60)


if __name__ == "__main__":
    main()
