"""
IOI analysis on our trained-from-scratch model.
Uses HuggingFace directly (no TransformerLens) to avoid config mismatch.
Runs after emnlp_final_overnight.py finishes training.

For each saved checkpoint:
  1. IOI accuracy + logit difference
  2. S2 patching via PyTorch hooks
  3. Per-layer logit lens

Output: results/emnlp_trained_ioi.json
"""

import os, gc, json, time, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

try:
    from circuitscaling.datasets import CANDIDATE_NAMES, ALL_TEMPLATES
except ImportError:
    sys.path.insert(0, "/workspace/ioi-sign-flip/src")
    from circuitscaling.datasets import CANDIDATE_NAMES, ALL_TEMPLATES

DEVICE = "cuda"
SEED = 42
N_BOOTSTRAP = 10_000
RESULTS_PATH = "results/emnlp_trained_ioi.json"
CKPT_DIR = "trained_model_ckpts"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def bootstrap_ci(values, seed=SEED):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2: return float("nan"), float("nan")
    idx = rng.integers(0, len(arr), size=(N_BOOTSTRAP, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# ====================================================================
# IOI PROMPT GENERATION (standalone, no TransformerLens)
# ====================================================================

IOI_TEMPLATES = [
    "When{S} and{IO} went to the store,{S} gave a drink to",
    "When{S} and{IO} went to the park,{S} gave a ball to",
    "When{S} and{IO} went to the school,{S} gave a book to",
    "When{S} and{IO} had lunch,{S} gave a sandwich to",
    "When{S} and{IO} met at the office,{S} handed a letter to",
]

PLACES = ["store", "park", "school", "office", "market",
          "library", "restaurant", "hospital", "museum", "station"]
OBJECTS = ["drink", "ball", "book", "letter", "toy",
           "gift", "flower", "key", "ticket", "phone"]

def get_single_token_names(tokenizer):
    """Get names that encode to exactly one token (with leading space)."""
    names = []
    for name in CANDIDATE_NAMES:
        toks = tokenizer.encode(" " + name, add_special_tokens=False)
        if len(toks) == 1:
            names.append(name)
    return names


def generate_ioi_data(tokenizer, n_prompts=200, seed=42):
    """Generate IOI prompts, controls, and metadata."""
    rng = np.random.default_rng(seed)
    names = get_single_token_names(tokenizer)
    if len(names) < 3:
        raise ValueError(f"Only {len(names)} single-token names found")

    prompts, ctrl_prompts = [], []
    io_names, s_names, ctrl_names = [], [], []

    for _ in range(n_prompts):
        # Pick S (repeated) and IO (indirect object) names.
        chosen = list(rng.choice(names, size=3, replace=False))
        s_name, io_name, c_name = chosen[0], chosen[1], chosen[2]

        place = rng.choice(PLACES)
        obj = rng.choice(OBJECTS)

        # ABBA template: S appears first and last.
        prompt = f"When {s_name} and {io_name} went to the {place}, {s_name} gave a {obj} to"
        ctrl_prompt = f"When {s_name} and {io_name} went to the {place}, {c_name} gave a {obj} to"

        prompts.append(prompt)
        ctrl_prompts.append(ctrl_prompt)
        io_names.append(io_name)
        s_names.append(s_name)
        ctrl_names.append(c_name)

    # Tokenize.
    io_token_ids = [tokenizer.encode(" " + n, add_special_tokens=False)[0] for n in io_names]
    s_token_ids = [tokenizer.encode(" " + n, add_special_tokens=False)[0] for n in s_names]

    return {
        "prompts": prompts, "ctrl_prompts": ctrl_prompts,
        "io_token_ids": io_token_ids, "s_token_ids": s_token_ids,
        "io_names": io_names, "s_names": s_names,
    }


def find_s2_position(token_ids, s_token_id):
    """Find position of second occurrence of s_token_id."""
    seen = 0
    for j in range(len(token_ids)):
        if token_ids[j] == s_token_id:
            seen += 1
            if seen == 2:
                return j
    return -1


# ====================================================================
# IOI ANALYSIS (using HuggingFace model + PyTorch hooks)
# ====================================================================

def analyze_checkpoint(ckpt_path, tokenizer, ioi_data, n_layers):
    """Run full IOI analysis on one checkpoint."""

    model = GPT2LMHeadModel.from_pretrained(ckpt_path).to(DEVICE)
    model.eval()

    prompts = ioi_data["prompts"]
    ctrl_prompts = ioi_data["ctrl_prompts"]
    io_ids = ioi_data["io_token_ids"]
    s_ids = ioi_data["s_token_ids"]
    N = len(prompts)

    # Tokenize (manually, handling variable lengths with padding).
    def tokenize_batch(texts):
        encoded = tokenizer(texts, return_tensors="pt", padding=True,
                           add_special_tokens=False)
        return encoded["input_ids"].to(DEVICE), encoded["attention_mask"].to(DEVICE)

    tokens, attn_mask = tokenize_batch(prompts)
    ctrl_tokens, ctrl_mask = tokenize_batch(ctrl_prompts)

    # Find last non-pad position for each sequence.
    last_pos = attn_mask.sum(dim=1) - 1  # [N]

    # Find S2 positions.
    s2_positions = []
    for i in range(N):
        seq = tokens[i].cpu().tolist()
        s2_positions.append(find_s2_position(seq, s_ids[i]))

    # === BASE FORWARD PASS ===
    with torch.no_grad():
        outputs = model(tokens, attention_mask=attn_mask)
    logits = outputs.logits  # [N, seq_len, vocab]

    # Get logits at last position.
    idx = torch.arange(N, device=DEVICE)
    last_logits = logits[idx, last_pos, :]  # [N, vocab]
    io_t = torch.tensor(io_ids, device=DEVICE)
    s_t = torch.tensor(s_ids, device=DEVICE)
    ld = (last_logits[idx, io_t] - last_logits[idx, s_t]).detach().cpu().numpy()
    acc = float((ld > 0).mean())
    ld_mean = float(ld.mean())
    ld_lo, ld_hi = bootstrap_ci(ld)

    # === S2 PATCHING ===
    # Get control's hidden states at S2 positions.
    patch_start = max(0, n_layers // 4)
    patch_end = min(n_layers, n_layers // 2 + 1)
    patch_layer_indices = list(range(patch_start, patch_end))

    # Run control to get hidden states.
    with torch.no_grad():
        ctrl_outputs = model(ctrl_tokens, attention_mask=ctrl_mask,
                            output_hidden_states=True)
    ctrl_hidden = ctrl_outputs.hidden_states  # tuple of [N, seq_len, d]
    # hidden_states[0] = embeddings, [1] = after layer 0, ..., [n_layers] = after last layer

    # Patching: replace S2 hidden states at specific layers.
    # We need to use hooks for this.
    patch_cache = {}
    for li in patch_layer_indices:
        # hidden_states index is li+1 (because [0] is embeddings)
        patch_cache[li] = ctrl_hidden[li + 1].clone()

    hooks = []
    def make_patch_hook(layer_idx):
        def hook_fn(module, input, output):
            # output is a tuple: (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hs = output[0].clone()
            else:
                hs = output.clone()
            for i in range(N):
                if s2_positions[i] > 0:
                    hs[i, s2_positions[i], :] = patch_cache[layer_idx][i, s2_positions[i], :]
            if isinstance(output, tuple):
                return (hs,) + output[1:]
            return hs
        return hook_fn

    # Register hooks on transformer blocks.
    for li in patch_layer_indices:
        h = model.transformer.h[li].register_forward_hook(make_patch_hook(li))
        hooks.append(h)

    with torch.no_grad():
        p_outputs = model(tokens, attention_mask=attn_mask)
    p_logits = p_outputs.logits
    p_last = p_logits[idx, last_pos, :]
    p_ld = (p_last[idx, io_t] - p_last[idx, s_t]).detach().cpu().numpy()
    delta_ld = p_ld - ld
    d_mean = float(delta_ld.mean())
    d_lo, d_hi = bootstrap_ci(delta_ld)

    # Remove hooks.
    for h in hooks:
        h.remove()

    # === LOGIT LENS ===
    with torch.no_grad():
        base_outputs = model(tokens, attention_mask=attn_mask,
                            output_hidden_states=True)
    all_hidden = base_outputs.hidden_states

    # Get the model's LM head (unembed).
    lm_head = model.lm_head
    ln_f = model.transformer.ln_f

    logit_lens = {}
    flip_layer = None
    for L in range(n_layers):
        hs = all_hidden[L + 1]  # after layer L
        last_hs = hs[idx, last_pos, :]  # [N, d]
        normed = ln_f(last_hs)
        ll_logits = lm_head(normed)
        ll_ld = (ll_logits[idx, io_t] - ll_logits[idx, s_t]).detach().cpu().numpy()
        ll_mean = float(ll_ld.mean())
        logit_lens[f"L{L}"] = ll_mean
        if ll_mean > 0 and flip_layer is None:
            flip_layer = L

    result = {
        "acc": acc, "ld_mean": ld_mean, "ld_ci": [ld_lo, ld_hi],
        "delta_ld": d_mean, "delta_ld_ci": [d_lo, d_hi],
        "logit_lens": logit_lens, "flip_layer": flip_layer,
        "patch_layers": patch_layer_indices,
    }

    del model, ctrl_hidden, all_hidden
    torch.cuda.empty_cache(); gc.collect()
    return result


def main():
    os.makedirs("results", exist_ok=True)
    results = {}
    if os.path.exists(RESULTS_PATH):
        try:
            with open(RESULTS_PATH) as f:
                results = json.load(f)
        except:
            pass

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Generate IOI data once (reused across checkpoints).
    N_PROMPTS = 200
    ioi_data = generate_ioi_data(tokenizer, n_prompts=N_PROMPTS, seed=SEED)
    log(f"Generated {len(ioi_data['prompts'])} IOI prompts")

    # Scan all checkpoint directories.
    all_dirs = []
    for d in sorted(os.listdir(CKPT_DIR)):
        full = os.path.join(CKPT_DIR, d)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, "config.json")):
            all_dirs.append((d, full))
    log(f"Found {len(all_dirs)} model directories: {[d[0] for d in all_dirs]}")

    for dir_name, ckpt_base in all_dirs:
        seed_key = dir_name

        # Find checkpoints.
        ckpt_steps = sorted([
            int(d.split("_")[1]) for d in os.listdir(ckpt_base)
            if d.startswith("step_") and os.path.isdir(os.path.join(ckpt_base, d))
        ])
        log(f"\n{dir_name}: {len(ckpt_steps)} checkpoints")

        if seed_key not in results:
            results[seed_key] = {"by_step": {}}

        # Get n_layers from config.
        import json as _json
        cfg_path = os.path.join(ckpt_base, "config.json")
        with open(cfg_path) as f:
            cfg = _json.load(f)
        n_layers = cfg.get("n_layer", cfg.get("num_hidden_layers", 8))
        log(f"  Model: {n_layers} layers, d={cfg.get('n_embd', cfg.get('hidden_size', '?'))}")

        for step in ckpt_steps:
            key = f"step_{step}"
            if key in results[seed_key]["by_step"]:
                cached = results[seed_key]["by_step"][key]
                if cached and "error" not in cached:
                    log(f"  SKIP step {step}")
                    continue

            ckpt_path = os.path.join(ckpt_base, f"step_{step}")
            log(f"  step {step}:")

            try:
                r = analyze_checkpoint(ckpt_path, tokenizer, ioi_data, n_layers)
                results[seed_key]["by_step"][key] = r
                log(f"    acc={r['acc']*100:5.1f}%  LD={r['ld_mean']:+.3f}  "
                    f"ΔLD={r['delta_ld']:+.4f} [{r['delta_ld_ci'][0]:+.3f},{r['delta_ld_ci'][1]:+.3f}]  "
                    f"flip_L={r['flip_layer']}")
            except Exception as e:
                log(f"    ERROR: {str(e)[:80]}")
                results[seed_key]["by_step"][key] = {"error": str(e)[:100]}

            with open(RESULTS_PATH, "w") as f:
                json.dump(results, f, indent=2)

    # Summary.
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    for seed_key in sorted(results.keys()):
        by = results[seed_key].get("by_step", {})
        if not by: continue
        steps = sorted(by.keys(), key=lambda x: int(x.split("_")[1]))
        accs = [(int(s.split("_")[1]), by[s].get("acc", None)) for s in steps
                if by[s] and "acc" in by[s]]
        dlds = [(int(s.split("_")[1]), by[s].get("delta_ld", None)) for s in steps
                if by[s] and "delta_ld" in by[s]]

        if accs:
            min_acc = min(accs, key=lambda x: x[1])
            log(f"\n  {seed_key}:")
            log(f"    min_acc = {min_acc[1]*100:.1f}% @ step {min_acc[0]}")
            log(f"    dip = {'YES' if min_acc[1] < 0.5 else 'NO'}")
            if dlds:
                max_dld = max(dlds, key=lambda x: x[1])
                min_dld = min(dlds, key=lambda x: x[1])
                log(f"    max ΔLD = {max_dld[1]:+.4f} @ step {max_dld[0]}")
                log(f"    min ΔLD = {min_dld[1]:+.4f} @ step {min_dld[0]}")
                # Sign flip detection.
                flip = None
                for i in range(1, len(dlds)):
                    if dlds[i-1][1] > 0.01 and dlds[i][1] < -0.01:
                        flip = dlds[i][0]; break
                log(f"    sign_flip = {'step ' + str(flip) if flip else 'NO'}")

    log(f"\nDone. Output: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
