"""
Head ablation and validation experiments
==============================================
Experiment A: Path patching on ORIGINAL Pythia-160M at early steps
    - Full path patching at steps 1000, 2000, 3000, 5000, 143000
    - Zero each head, measure how every downstream head's attention changes

Experiment B: Duplication detection probes
    - Binary classifier: "does this prompt contain a repeated name?"
    - IOI dataset (repeated name) vs control dataset (3 different names)
    - Probe at END position and S2 position
    - Across steps (1000, 2000, 3000, 5000, 10000, 143000) and layers (0-11)
    - Uses sklearn LogisticRegression
"""

import os
os.environ["HF_TOKEN"] = ""

import torch, json, time, numpy as np, sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES

DEVICE = "cuda"
MODEL_NAME = "EleutherAI/pythia-160m-deduped"
TEMPLATES = ALL_TEMPLATES[:15]
PPT = 20
SEED = 42
RESULTS_FILE = "results/head_ablation.json"

def empty_cache():
    torch.cuda.empty_cache()

def save_results(results):
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)

def get_positions(tokens, io_ids, s_ids):
    positions = []
    for i in range(tokens.shape[0]):
        io_tok, s_tok = io_ids[i].item(), s_ids[i].item()
        io_pos, s1_pos, s2_pos = -1, -1, -1
        s_count = 0
        for j in range(1, tokens.shape[1]):
            if tokens[i,j].item() == io_tok and io_pos == -1: io_pos = j
            if tokens[i,j].item() == s_tok:
                s_count += 1
                if s_count == 1: s1_pos = j
                elif s_count == 2: s2_pos = j
        positions.append((io_pos, s1_pos, s2_pos))
    return positions

def load_original_model(step):
    """Load original Pythia-160M at a given step."""
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision="step%d" % step)
    model = HookedTransformer.from_pretrained(
        MODEL_NAME, hf_model=hf_model, device=DEVICE,
        center_writing_weights=True, center_unembed=True, fold_ln=True)
    del hf_model
    return model


# ============================================================
# EXPERIMENT A: Path Patching on Original Pythia-160M
# ============================================================
def run_experiment_a(results):
    print("\n" + "=" * 60)
    print("  EXPERIMENT A: Path Patching on Original Pythia-160M")
    print("  Steps: 1000, 2000, 3000, 5000, 143000")
    print("=" * 60)

    if "exp_a_path_patching" not in results:
        results["exp_a_path_patching"] = {}

    patch_steps = [1000, 2000, 3000, 5000, 143000]
    n_layers = 12
    n_heads = 12

    for step in patch_steps:
        step_key = "step_%d" % step
        if step_key in results["exp_a_path_patching"]:
            print("  Step %d done, skip" % step)
            continue

        print("\n  --- Step %d ---" % step)
        model = load_original_model(step)

        # Get template data
        template_data = []
        for tmpl in TEMPLATES[:8]:
            ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl],
                            symmetric=True, seed=SEED)
            tokens = model.to_tokens(ds.prompts).to(DEVICE)
            io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
            s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)
            template_data.append((tokens, io_ids, s_ids))

        # Baseline logit diff
        base_lds = []
        for tokens, io_ids, s_ids in template_data:
            logits = model(tokens)
            last = logits[:, -1, :]
            for i in range(len(io_ids)):
                base_lds.append(last[i, io_ids[i]].item() - last[i, s_ids[i]].item())
        base_mean_ld = float(np.mean(base_lds))
        base_acc = float((np.array(base_lds) > 0).mean())
        print("  Baseline: acc=%.1f%%, mean_LD=%.4f" % (base_acc * 100, base_mean_ld))

        # For each head: zero it and measure (a) logit diff change, (b) attention changes in all downstream heads
        step_results = {"baseline_acc": round(base_acc, 4), "baseline_ld": round(base_mean_ld, 4), "heads": {}}

        for src_layer in range(n_layers):
            for src_head in range(n_heads):
                src_name = "L%dH%d" % (src_layer, src_head)

                def zero_hook(value, hook, h=src_head):
                    value[:, :, h, :] = 0.0
                    return value

                # Measure ablated logit diff
                abl_lds = []
                for tokens, io_ids, s_ids in template_data:
                    logits = model.run_with_hooks(tokens, fwd_hooks=[
                        ("blocks.%d.attn.hook_z" % src_layer, zero_hook)])
                    last = logits[:, -1, :]
                    for i in range(len(io_ids)):
                        abl_lds.append(last[i, io_ids[i]].item() - last[i, s_ids[i]].item())
                delta_ld = float(np.mean(abl_lds)) - base_mean_ld

                step_results["heads"][src_name] = {
                    "delta_ld": round(delta_ld, 4),
                }

            if (src_layer + 1) % 4 == 0:
                print("    Ablation layer %d/%d" % (src_layer + 1, n_layers))

        # Now for the TOP 5 most impactful heads, measure downstream attention changes
        sorted_heads = sorted(step_results["heads"].items(), key=lambda x: x[1]["delta_ld"])
        top5 = [h for h, _ in sorted_heads[:5]]
        print("  Top 5: %s" % top5)

        for src_name in top5:
            src_layer = int(src_name.split("H")[0][1:])
            src_head_idx = int(src_name.split("H")[1])

            def zero_hook(value, hook, h=src_head_idx):
                value[:, :, h, :] = 0.0
                return value

            # Get clean attention patterns for downstream heads
            clean_attns = {}
            for tokens, io_ids, s_ids in template_data[:4]:
                _, cache = model.run_with_cache(tokens, remove_batch_dim=False)
                positions = get_positions(tokens, io_ids, s_ids)
                final_pos = tokens.shape[1] - 1
                for layer in range(src_layer + 1, n_layers):
                    for head in range(n_heads):
                        hn = "L%dH%d" % (layer, head)
                        if hn not in clean_attns:
                            clean_attns[hn] = {"IO": [], "S2": []}
                        attn = cache["blocks.%d.attn.hook_pattern" % layer]
                        for i, (io_pos, s1_pos, s2_pos) in enumerate(positions):
                            if io_pos > 0:
                                clean_attns[hn]["IO"].append(attn[i, head, final_pos, io_pos].item())
                            if s2_pos > 0:
                                clean_attns[hn]["S2"].append(attn[i, head, final_pos, s2_pos].item())
                del cache; empty_cache()

            # Corrupted attention (source head zeroed)
            corr_attns = {}
            for tokens, io_ids, s_ids in template_data[:4]:
                model.add_hook("blocks.%d.attn.hook_z" % src_layer, zero_hook)
                _, cache = model.run_with_cache(tokens, remove_batch_dim=False)
                model.reset_hooks()
                positions = get_positions(tokens, io_ids, s_ids)
                final_pos = tokens.shape[1] - 1
                for layer in range(src_layer + 1, n_layers):
                    for head in range(n_heads):
                        hn = "L%dH%d" % (layer, head)
                        if hn not in corr_attns:
                            corr_attns[hn] = {"IO": [], "S2": []}
                        attn = cache["blocks.%d.attn.hook_pattern" % layer]
                        for i, (io_pos, s1_pos, s2_pos) in enumerate(positions):
                            if io_pos > 0:
                                corr_attns[hn]["IO"].append(attn[i, head, final_pos, io_pos].item())
                            if s2_pos > 0:
                                corr_attns[hn]["S2"].append(attn[i, head, final_pos, s2_pos].item())
                del cache; empty_cache()

            # Compute changes
            downstream_changes = []
            for hn in clean_attns:
                if hn in corr_attns:
                    d_io = float(np.mean(corr_attns[hn]["IO"]) - np.mean(clean_attns[hn]["IO"])) if clean_attns[hn]["IO"] else 0
                    d_s2 = float(np.mean(corr_attns[hn]["S2"]) - np.mean(clean_attns[hn]["S2"])) if clean_attns[hn]["S2"] else 0
                    total = abs(d_io) + abs(d_s2)
                    if total > 0.02:  # Only record meaningful changes
                        downstream_changes.append({
                            "head": hn,
                            "delta_IO": round(d_io, 4),
                            "delta_S2": round(d_s2, 4),
                            "total_change": round(total, 4),
                        })

            downstream_changes.sort(key=lambda x: -x["total_change"])
            step_results["heads"][src_name]["downstream_top10"] = downstream_changes[:10]

            print("    %s: delta_LD=%+.4f, affects %d downstream heads" % (
                src_name, step_results["heads"][src_name]["delta_ld"], len(downstream_changes)))
            for dc in downstream_changes[:3]:
                print("      %s: dIO=%+.4f dS2=%+.4f" % (dc["head"], dc["delta_IO"], dc["delta_S2"]))

        results["exp_a_path_patching"][step_key] = step_results
        save_results(results)

        del model; empty_cache()

    print("\nEXPERIMENT A COMPLETE")


# ============================================================
# EXPERIMENT B: Duplication Detection Probes
# ============================================================
def run_experiment_b(results):
    print("\n" + "=" * 60)
    print("  EXPERIMENT B: Duplication Detection Probes")
    print("  Binary: 'Does this prompt contain a repeated name?'")
    print("  Positions: END and S2")
    print("  Uses sklearn LogisticRegression")
    print("=" * 60)

    if "exp_b_duplication_probes" not in results:
        results["exp_b_duplication_probes"] = {}

    probe_steps = [1000, 2000, 3000, 5000, 10000, 143000]
    n_layers = 12

    for step in probe_steps:
        step_key = "step_%d" % step
        if step_key in results["exp_b_duplication_probes"]:
            print("  Step %d done, skip" % step)
            continue

        print("\n  --- Step %d ---" % step)
        model = load_original_model(step)

        # Collect residual stream activations for IOI (label=1, has repeated name)
        # and control (label=0, all different names)
        activations_end = {l: [] for l in range(n_layers)}
        activations_s2 = {l: [] for l in range(n_layers)}
        labels = []

        # === IOI examples (label = 1: contains repeated name) ===
        print("  Collecting IOI examples (repeated name)...")
        for tmpl in TEMPLATES[:12]:
            try:
                ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl],
                                symmetric=True, seed=SEED)
                tokens = model.to_tokens(ds.prompts).to(DEVICE)
                io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
                s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)

                _, cache = model.run_with_cache(tokens, remove_batch_dim=False)
                positions = get_positions(tokens, io_ids, s_ids)
                final_pos = tokens.shape[1] - 1

                for layer in range(n_layers):
                    resid = cache["blocks.%d.hook_resid_post" % layer]
                    for i, (io_pos, s1_pos, s2_pos) in enumerate(positions):
                        activations_end[layer].append(resid[i, final_pos, :].detach().cpu().float().numpy())
                        if s2_pos > 0:
                            activations_s2[layer].append(resid[i, s2_pos, :].detach().cpu().float().numpy())

                for i in range(len(io_ids)):
                    labels.append(1)

                del cache; empty_cache()
            except Exception as e:
                continue

        n_ioi = len(labels)
        print("    IOI examples: %d" % n_ioi)

        # === Control examples (label = 0: all different names) ===
        # Use IOI templates but replace S1 with a third name so no repetition
        print("  Collecting control examples (no repeated name)...")
        import random
        random.seed(SEED)

        # Get the name pool from the dataset
        from circuitscaling.datasets import CANDIDATE_NAMES as NAMES
        name_pool = NAMES[:50]  # Use first 50 names

        n_control = 0
        for tmpl in TEMPLATES[:12]:
            try:
                ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl],
                                symmetric=True, seed=SEED)

                # Create control prompts: replace S occurrences with different names
                control_prompts = []
                for prompt in ds.prompts:
                    # Find the IO and S names in this prompt
                    io_name = None
                    s_name = None
                    for name in name_pool:
                        if " " + name + " " in prompt:
                            if io_name is None:
                                io_name = name
                            elif name != io_name:
                                s_name = name
                                break

                    if io_name and s_name:
                        # Pick a third name different from both
                        available = [n for n in name_pool if n != io_name and n != s_name]
                        if available:
                            third_name = random.choice(available)
                            # Replace second occurrence of s_name with third_name
                            parts = prompt.split(s_name)
                            if len(parts) >= 3:
                                control = parts[0] + s_name + parts[1] + third_name + s_name.join(parts[2:])
                                control_prompts.append(control)
                            else:
                                control_prompts.append(prompt)  # fallback
                        else:
                            control_prompts.append(prompt)
                    else:
                        control_prompts.append(prompt)

                if not control_prompts:
                    continue

                tokens_ctrl = model.to_tokens(control_prompts).to(DEVICE)

                # We need to find positions in control prompts too
                # For control, use same structure but the "S2" position now has a different name
                _, cache = model.run_with_cache(tokens_ctrl, remove_batch_dim=False)
                final_pos = tokens_ctrl.shape[1] - 1

                # Use original positions as approximate (same template structure)
                io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
                s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)
                positions = get_positions(tokens_ctrl, io_ids, s_ids)

                for layer in range(n_layers):
                    resid = cache["blocks.%d.hook_resid_post" % layer]
                    for i in range(min(len(control_prompts), resid.shape[0])):
                        activations_end[layer].append(resid[i, final_pos, :].detach().cpu().float().numpy())
                        io_pos, s1_pos, s2_pos = positions[i] if i < len(positions) else (-1, -1, -1)
                        if s2_pos > 0:
                            activations_s2[layer].append(resid[i, s2_pos, :].detach().cpu().float().numpy())

                for _ in range(min(len(control_prompts), tokens_ctrl.shape[0])):
                    labels.append(0)
                    n_control += 1

                del cache; empty_cache()
            except Exception as e:
                continue

        print("    Control examples: %d" % n_control)
        print("    Total: %d (%.0f%% IOI)" % (len(labels), n_ioi / len(labels) * 100))

        if len(labels) < 20:
            print("    Too few examples, skip")
            del model; empty_cache()
            continue

        labels_arr = np.array(labels)

        # Train probes at each layer
        step_results = {"n_ioi": n_ioi, "n_control": n_control}

        for pos_name, activations in [("END", activations_end), ("S2", activations_s2)]:
            pos_results = {}
            for layer in range(n_layers):
                X = np.array(activations[layer])
                n_samples = min(len(X), len(labels_arr))
                if n_samples < 20:
                    continue
                X = X[:n_samples]
                y = labels_arr[:n_samples]

                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=SEED, stratify=y)

                # Logistic regression
                clf = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
                clf.fit(X_train, y_train)

                train_acc = accuracy_score(y_train, clf.predict(X_train))
                test_acc = accuracy_score(y_test, clf.predict(X_test))

                pos_results["layer_%d" % layer] = {
                    "train_acc": round(float(train_acc), 4),
                    "test_acc": round(float(test_acc), 4),
                }

            step_results[pos_name] = pos_results

            # Print summary
            print("    %s position:" % pos_name)
            for l in sorted(pos_results.keys(), key=lambda x: int(x.split('_')[1])):
                r = pos_results[l]
                bar = "#" * int(r["test_acc"] * 20)
                print("      %s: train=%.1f%% test=%.1f%% %s" % (
                    l, r["train_acc"] * 100, r["test_acc"] * 100, bar))

        results["exp_b_duplication_probes"][step_key] = step_results
        save_results(results)

        del model; empty_cache()

    print("\nEXPERIMENT B COMPLETE")


# ============================================================
# EXPERIMENT C: Loss Comparison (Original vs Retrained)
# ============================================================
def run_experiment_c(results):
    print("\n" + "=" * 60)
    print("  EXPERIMENT C: Loss Comparison")
    print("  Original (seed=1234) vs Retrained (seed=42)")
    print("=" * 60)

    if "exp_c_loss_comparison" in results:
        print("  Already done, skip")
        return

    from transformers import AutoTokenizer
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load some held-out Pile text for eval
    print("  Loading eval data...")
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    ds = ds.skip(100000)  # Skip ahead to get unseen data

    eval_texts = []
    for i, example in enumerate(ds):
        if i >= 100:
            break
        eval_texts.append(example["text"])

    # Tokenize
    eval_tokens = []
    for text in eval_texts:
        toks = tokenizer(text, truncation=True, max_length=2048, return_tensors="pt")["input_ids"]
        if toks.shape[1] >= 512:
            eval_tokens.append(toks[:, :512])

    print("  Eval sequences: %d" % len(eval_tokens))

    comparison_steps = [1000, 2000, 3000, 5000, 10000]
    loss_results = {}

    for step in comparison_steps:
        print("\n  --- Step %d ---" % step)

        # Original
        print("    Loading original...")
        hf_orig = AutoModelForCausalLM.from_pretrained(MODEL_NAME, revision="step%d" % step).cuda().half()
        orig_losses = []
        with torch.no_grad():
            for toks in eval_tokens[:50]:
                out = hf_orig(toks.cuda(), labels=toks.cuda())
                orig_losses.append(out.loss.item())
        orig_mean = float(np.mean(orig_losses))
        del hf_orig; empty_cache()

        # Retrained
        retrained_path = "/workspace/pythia-160m-retrain/checkpoints/step_%d" % step
        if os.path.exists(retrained_path):
            print("    Loading retrained...")
            from transformers import GPTNeoXForCausalLM
            hf_retrain = GPTNeoXForCausalLM.from_pretrained(retrained_path).cuda().half()
            retrain_losses = []
            with torch.no_grad():
                for toks in eval_tokens[:50]:
                    out = hf_retrain(toks.cuda(), labels=toks.cuda())
                    retrain_losses.append(out.loss.item())
            retrain_mean = float(np.mean(retrain_losses))
            del hf_retrain; empty_cache()
        else:
            print("    Retrained checkpoint not on disk, skip")
            retrain_mean = None

        loss_results["step_%d" % step] = {
            "original_loss": round(orig_mean, 4),
            "retrained_loss": round(retrain_mean, 4) if retrain_mean else None,
            "diff": round(retrain_mean - orig_mean, 4) if retrain_mean else None,
        }

        print("    Original: %.4f, Retrained: %s, Diff: %s" % (
            orig_mean,
            "%.4f" % retrain_mean if retrain_mean else "N/A",
            "%+.4f" % (retrain_mean - orig_mean) if retrain_mean else "N/A"))

    results["exp_c_loss_comparison"] = loss_results
    save_results(results)

    print("\nEXPERIMENT C COMPLETE")


# ============================================================
# EXPERIMENT D: L2H6 Full Attention Map (5-10 examples)
# ============================================================
def run_experiment_d(results):
    print("\n" + "=" * 60)
    print("  EXPERIMENT D: L2H6 Full Attention Map")
    print("  Manual inspection on 10 IOI examples")
    print("=" * 60)

    if "exp_d_l2h6_attention" in results:
        print("  Already done, skip")
        return

    # Load retrained model at step 10000
    retrained_path = "/workspace/pythia-160m-retrain/checkpoints/step_10000"
    if not os.path.exists(retrained_path):
        print("  Retrained checkpoint not on disk, trying original step 143000")
        model = load_original_model(143000)
        target_head = (8, 9)  # L8H9 for original
        head_label = "L8H9"
    else:
        from transformers import GPTNeoXForCausalLM
        hf = GPTNeoXForCausalLM.from_pretrained(retrained_path)
        model = HookedTransformer.from_pretrained(
            MODEL_NAME, hf_model=hf, device=DEVICE,
            center_writing_weights=True, center_unembed=True, fold_ln=True)
        del hf
        target_head = (2, 6)  # L2H6 for retrained
        head_label = "L2H6"

    examples = []
    ds = IOIDataset(model=model, n_prompts=10, templates=[TEMPLATES[0]],
                    symmetric=True, seed=SEED)
    tokens = model.to_tokens(ds.prompts).to(DEVICE)
    io_ids = torch.tensor(ds.io_token_ids, device=DEVICE)
    s_ids = torch.tensor(ds.s_token_ids, device=DEVICE)

    _, cache = model.run_with_cache(tokens, remove_batch_dim=False)
    attn = cache["blocks.%d.attn.hook_pattern" % target_head[0]]

    for i in range(min(10, tokens.shape[0])):
        final_pos = tokens.shape[1] - 1
        # Get full attention distribution from END position
        attn_dist = attn[i, target_head[1], final_pos, :final_pos+1].detach().cpu().numpy()

        # Decode each token
        token_strs = [model.tokenizer.decode([tokens[i, j].item()]).strip() for j in range(final_pos + 1)]

        # Sort by attention weight
        sorted_idx = np.argsort(-attn_dist)

        example = {
            "prompt": ds.prompts[i],
            "io_name": model.tokenizer.decode([io_ids[i].item()]).strip(),
            "s_name": model.tokenizer.decode([s_ids[i].item()]).strip(),
            "top_attended": [],
        }
        for j in sorted_idx[:10]:
            example["top_attended"].append({
                "position": int(j),
                "token": token_strs[j],
                "attention": round(float(attn_dist[j]), 4),
            })

        # Also record attention to specific positions
        positions = get_positions(tokens[i:i+1], io_ids[i:i+1], s_ids[i:i+1])[0]
        io_pos, s1_pos, s2_pos = positions
        example["named_positions"] = {
            "IO": {"pos": io_pos, "attn": round(float(attn_dist[io_pos]), 4) if io_pos > 0 else 0},
            "S1": {"pos": s1_pos, "attn": round(float(attn_dist[s1_pos]), 4) if s1_pos > 0 else 0},
            "S2": {"pos": s2_pos, "attn": round(float(attn_dist[s2_pos]), 4) if s2_pos > 0 else 0},
            "END": {"pos": final_pos, "attn": round(float(attn_dist[final_pos]), 4)},
            "BOS": {"pos": 0, "attn": round(float(attn_dist[0]), 4)},
        }

        examples.append(example)
        print("\n  Example %d: %s" % (i + 1, ds.prompts[i][:60]))
        print("    IO=%s, S=%s" % (example["io_name"], example["s_name"]))
        print("    BOS=%.3f IO=%.3f S1=%.3f S2=%.3f END=%.3f" % (
            example["named_positions"]["BOS"]["attn"],
            example["named_positions"]["IO"]["attn"],
            example["named_positions"]["S1"]["attn"],
            example["named_positions"]["S2"]["attn"],
            example["named_positions"]["END"]["attn"]))
        print("    Top 5 tokens: %s" % ", ".join(
            ["%s(%.3f)" % (t["token"], t["attention"]) for t in example["top_attended"][:5]]))

    results["exp_d_l2h6_attention"] = {"head": head_label, "examples": examples}
    save_results(results)

    del model, cache; empty_cache()
    print("\nEXPERIMENT D COMPLETE")


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("  HEAD ABLATION EXPERIMENTS")
    print("  Started: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 60)

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
    else:
        results = {}

    t0 = time.time()

    # Run in order of priority
    for name, fn in [("D (L2H6 attention)", run_experiment_d),
                     ("B (duplication probes)", run_experiment_b),
                     ("C (loss comparison)", run_experiment_c),
                     ("A (path patching)", run_experiment_a)]:
        print("\n>>> Starting Experiment %s" % name)
        try:
            fn(results)
        except Exception as e:
            print("  FAILED: %s" % str(e))
            import traceback; traceback.print_exc()

    elapsed = time.time() - t0
    save_results(results)
    print("\n" + "=" * 60)
    print("  ALL DONE. Time: %.1f hours" % (elapsed / 3600))
    print("  Results: %s" % RESULTS_FILE)
    print("=" * 60)


if __name__ == "__main__":
    main()
