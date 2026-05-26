"""
Detection-before-Suppression Timeline
======================================
The capstone mechanism experiment. Addresses three potential concerns at once:

  (A) "Is the dip just induction-head formation (Olsson 2022)?"
      -> Measure induction prefix-matching score across checkpoints.
         Prediction: duplicate-DETECTION (induction) comes online BEFORE
         IOI recovers. The dip is the window where detection exists but
         suppression does not. This SUPPORTS detection-before-suppression
         and distinguishes it from "just induction heads."

  (B) "Is the dip double descent / an optimization artifact?"
      -> Measure LM loss on held-out natural text across the SAME
         checkpoints. Prediction: LM loss decreases MONOTONICALLY while
         IOI accuracy goes below chance. Getting better at language while
         getting worse at IOI dissociates this from double descent
         (which involves a loss bump).

  (C) "The sign flip is just a corollary of the behavioral flip."
      -> Track the S-inhibition head (attention from END->S2) across
         training. Prediction: the suppression head comes online at
         RECOVERY, not at the dip. Detection and suppression are
         temporally separated, which is the mechanistic claim.

Per checkpoint (160M and 410M, dense), we measure:
  1. IOI accuracy + base LD (curated-ish prompt set)
  2. Induction prefix-matching score (mean over heads + max head id)
  3. Logit-lens LD at each layer (where suppression resolves)
  4. Top S-inhibition head: END->S2 attention, tracked across training
  5. LM loss on held-out text

Output: results/emnlp_induction_timeline.json
"""

import os, gc, json, time, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer

try:
    from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES, CANDIDATE_NAMES
except ImportError:
    sys.path.insert(0, "/workspace/ioi-sign-flip/src")
    from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES, CANDIDATE_NAMES

DEVICE = "cuda"
SEED = 42
RESULTS_PATH = "results/emnlp_induction_timeline.json"

# Held-out natural text for LM-loss monotonicity (no IOI structure).
HELDOUT_TEXT = (
    "The history of science is a record of patient observation and gradual "
    "revision. Early astronomers tracked the motion of planets across the night "
    "sky, recording their positions over many years. These measurements, taken "
    "without instruments more sophisticated than the naked eye, eventually "
    "revealed regular patterns. Later generations built telescopes that brought "
    "distant worlds into focus, and the slow accumulation of evidence reshaped "
    "humanity's understanding of its place in the cosmos. Each discovery raised "
    "new questions, and the answers often arrived only after decades of careful work."
)

MODELS = [
    {"name": "pythia-160m", "repo": "EleutherAI/pythia-160m-deduped",
     "checkpoints": [0, 256, 512, 1000, 2000, 3000, 4000, 5000, 8000,
                     13000, 33000, 143000]},
    {"name": "pythia-410m", "repo": "EleutherAI/pythia-410m-deduped",
     "checkpoints": [0, 512, 1000, 2000, 3000, 4000, 5000, 8000,
                     13000, 30000, 50000, 143000]},
]

TEMPLATES = ALL_TEMPLATES[:6]
PROMPTS_PER_TEMPLATE = 25


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

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


def induction_score(model, seq_len=48, batch=8):
    """Prefix-matching induction score per head (Olsson 2022 style).
    Random tokens repeated twice; measure attention along the induction
    stripe (offset = seq_len - 1)."""
    torch.manual_seed(SEED)
    vocab = model.cfg.d_vocab
    rand = torch.randint(100, min(vocab, 10000), (batch, seq_len), device=DEVICE)
    bos = torch.full((batch, 1), model.tokenizer.bos_token_id or 0, device=DEVICE)
    tokens = torch.cat([bos, rand, rand], dim=1)  # [batch, 1 + 2*seq_len]

    pattern_names = [f"blocks.{L}.attn.hook_pattern" for L in range(model.cfg.n_layers)]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=pattern_names)

    head_scores = {}
    # Induction stripe: query at position (1+seq_len+i) attends to key (1+i+1).
    # offset between query and key = seq_len - 1.
    offset = seq_len - 1
    for L in range(model.cfg.n_layers):
        attn = cache[f"blocks.{L}.attn.hook_pattern"]  # [batch, head, q, k]
        for H in range(model.cfg.n_heads):
            ah = attn[:, H]  # [batch, q, k]
            stripe = ah.diagonal(offset=-offset, dim1=-2, dim2=-1)  # [batch, ndiag]
            head_scores[(L, H)] = float(stripe.mean().item())

    mean_score = float(np.mean(list(head_scores.values())))
    max_head = max(head_scores, key=head_scores.get)
    max_score = head_scores[max_head]
    del cache
    return {
        "mean": mean_score, "max": max_score,
        "max_head": [int(max_head[0]), int(max_head[1])],
    }


def lm_loss(model):
    """Cross-entropy on held-out natural text."""
    tokens = model.to_tokens(HELDOUT_TEXT).to(DEVICE)
    with torch.no_grad():
        logits = model(tokens)
    logp = F.log_softmax(logits[0, :-1].float(), dim=-1)
    targets = tokens[0, 1:]
    nll = -logp[torch.arange(len(targets)), targets].mean()
    return float(nll.item())


def ioi_and_suppression(model, single_ids):
    """IOI accuracy + base LD + logit lens + top S-inhibition head
    (END->S2 attention)."""
    rng = np.random.default_rng(SEED + 1)
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    base_lds = []
    ll_acc = {L: [] for L in range(n_layers)}
    # END->S2 attention per head, accumulated.
    end_s2_attn = {(L, H): [] for L in range(n_layers) for H in range(n_heads)}

    for tmpl in TEMPLATES:
        ds = IOIDataset(model=model, n_prompts=PROMPTS_PER_TEMPLATE,
                        templates=[tmpl], symmetric=True, seed=SEED)
        tokens = model.to_tokens(ds.prompts).to(DEVICE)
        n = tokens.shape[0]
        io_ids = torch.tensor(ds.io_token_ids, dtype=torch.long, device=DEVICE)
        s_ids = torch.tensor(ds.s_token_ids, dtype=torch.long, device=DEVICE)
        idx = torch.arange(n, device=DEVICE)

        s2_pos = [find_s2_position(tokens[i].cpu(), ds.s_token_ids[i]) for i in range(n)]
        end_pos = (tokens != model.tokenizer.pad_token_id).sum(1) - 1 if model.tokenizer.pad_token_id is not None else torch.full((n,), tokens.shape[1]-1, device=DEVICE)

        resid_names = [f"blocks.{L}.hook_resid_post" for L in range(n_layers)]
        patt_names = [f"blocks.{L}.attn.hook_pattern" for L in range(n_layers)]
        with torch.no_grad():
            logits, cache = model.run_with_cache(
                tokens, names_filter=lambda nm: ("hook_resid_post" in nm or "hook_pattern" in nm))

        last = logits[:, -1, :]
        ld = (last[idx, io_ids] - last[idx, s_ids]).detach().cpu().numpy()
        base_lds.extend(ld.tolist())

        # Logit lens.
        for L in range(n_layers):
            resid = cache[f"blocks.{L}.hook_resid_post"][:, -1, :]
            normed = model.ln_final(resid)
            ll_logits = normed @ model.W_U + model.b_U
            ll_ld = (ll_logits[idx, io_ids] - ll_logits[idx, s_ids]).detach().cpu().numpy()
            ll_acc[L].extend(ll_ld.tolist())

        # END -> S2 attention per head.
        for L in range(n_layers):
            patt = cache[f"blocks.{L}.attn.hook_pattern"]  # [n, head, q, k]
            for i in range(n):
                if s2_pos[i] > 0:
                    q = patt.shape[2] - 1  # last query position
                    for H in range(n_heads):
                        end_s2_attn[(L, H)].append(float(patt[i, H, q, s2_pos[i]].item()))
        del cache

    base_arr = np.array(base_lds)
    logit_lens = {f"L{L}": float(np.mean(ll_acc[L])) for L in range(n_layers)}
    flip_layer = next((L for L in range(n_layers) if np.mean(ll_acc[L]) > 0), None)

    # Top S-inhibition head = head with highest mean END->S2 attention.
    head_attn = {k: float(np.mean(v)) for k, v in end_s2_attn.items() if v}
    top_head = max(head_attn, key=head_attn.get)
    return {
        "ioi_acc": float((base_arr > 0).mean()),
        "base_ld": float(base_arr.mean()),
        "logit_lens": logit_lens,
        "flip_layer": flip_layer,
        "top_s2_head": [int(top_head[0]), int(top_head[1])],
        "top_s2_attn": head_attn[top_head],
        "all_head_s2_attn": {f"L{k[0]}H{k[1]}": v for k, v in head_attn.items()},
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
        log("=" * 60); log(name); log("=" * 60)
        single_ids = None
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
                if single_ids is None:
                    single_ids = get_single_token_names(model.tokenizer)

                ind = induction_score(model)
                loss = lm_loss(model)
                ioi = ioi_and_suppression(model, single_ids)

                r = {
                    "ioi_acc": ioi["ioi_acc"], "base_ld": ioi["base_ld"],
                    "induction_mean": ind["mean"], "induction_max": ind["max"],
                    "induction_max_head": ind["max_head"],
                    "lm_loss": loss,
                    "flip_layer": ioi["flip_layer"],
                    "top_s2_head": ioi["top_s2_head"],
                    "top_s2_attn": ioi["top_s2_attn"],
                    "logit_lens": ioi["logit_lens"],
                    "all_head_s2_attn": ioi["all_head_s2_attn"],
                }
                results[name]["by_step"][key] = r
                log(f"    IOI_acc={r['ioi_acc']*100:5.1f}%  LD={r['base_ld']:+.3f}  "
                    f"induction(mean={r['induction_mean']:.3f},max={r['induction_max']:.3f}@L{ind['max_head'][0]}H{ind['max_head'][1]})  "
                    f"LM_loss={r['lm_loss']:.3f}  flip_L={r['flip_layer']}  "
                    f"topS2head=L{r['top_s2_head'][0]}H{r['top_s2_head'][1]}({r['top_s2_attn']:.3f})")
                del model; torch.cuda.empty_cache(); gc.collect()
            except Exception as e:
                log(f"    ERROR: {str(e)[:100]}")
                results[name]["by_step"][key] = {"error": str(e)[:120]}
            json.dump(results, open(RESULTS_PATH, "w"), indent=2)

    # Summary timeline.
    log("\n" + "=" * 60)
    log("DETECTION-BEFORE-SUPPRESSION TIMELINE")
    log("=" * 60)
    for cfg in MODELS:
        name = cfg["name"]
        by = results.get(name, {}).get("by_step", {})
        if not by: continue
        log(f"\n{name}:")
        log(f"  {'Step':>7} {'IOI_acc':>8} {'induction':>10} {'LM_loss':>8} {'flip_L':>6} {'topS2head':>10}")
        for sk in sorted(by, key=lambda x: int(x.split("_")[1])):
            r = by[sk]
            if not r or "error" in r: continue
            step = sk.split("_")[1]
            th = f"L{r['top_s2_head'][0]}H{r['top_s2_head'][1]}"
            log(f"  {step:>7} {r['ioi_acc']*100:>7.1f}% {r['induction_max']:>10.3f} "
                f"{r['lm_loss']:>8.3f} {str(r['flip_layer']):>6} {th:>10}")


if __name__ == "__main__":
    main()
