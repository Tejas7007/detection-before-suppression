"""
Mini-IOI v3: Multi-Objective Toy Model
=======================================

The previous versions failed because a model trained ONLY on [A,B,A]->B
memorizes the task immediately. There is no competing objective that
creates the shortcut.

In real Pythia, the dip arises because:
1. General language modeling teaches "predict salient/frequent tokens"
2. IOI requires "suppress the duplicate, predict the other one"
3. The general skill (1) is learned first and creates a shortcut
4. The specific circuit (2) forms later and corrects it

This version uses TWO types of training sequences:

  Type 1 (COPY, ~80%):  [X, Y, Z] where X,Y,Z all different -> target Z
    Teaches: "predict the token at the last position" = copy-last bias

  Type 2 (IOI, ~20%):   [A, B, A] where A != B -> target B
    Requires: "detect repetition, suppress A, predict B"

Training dynamics:
1. Model first learns Type 1 (majority, easier) -> develops "copy last" bias
2. On Type 2: last token is A -> model predicts A -> WRONG -> below chance
3. Model eventually learns Type 2-specific "detect repetition + suppress" circuit
4. Type 2 accuracy recovers

SIGN FLIP (patching position 2 on Type 2 sequences):
  At dip: replacing second A with C removes duplication signal -> model less
          confident in A -> logit(B)-logit(A) improves -> delta > 0
  At maturity: replacing A with C breaks suppression circuit -> model can't
               suppress A -> logit(B)-logit(A) drops -> delta < 0

Evaluation is on Type 2 sequences only, using logit(B)-logit(A) as the
primary metric (matching the real IOI experimental protocol).
"""

import os
import gc
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
N_BOOTSTRAP = 10_000
RESULTS_PATH = "results/emnlp_mini_ioi.json"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def bootstrap_ci(values, n_resamples=N_BOOTSTRAP, alpha=0.05, seed=SEED):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape[0] == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, arr.shape[0], size=(n_resamples, arr.shape[0]))
    means = arr[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)),
            float(np.quantile(means, 1 - alpha / 2)))


# ====================================================================
# MODEL
# ====================================================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_mlp):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_mlp), nn.GELU(), nn.Linear(d_mlp, d_model),
        )

    def forward(self, x, mask=None, need_weights=False):
        h = self.ln1(x)
        attn_out, attn_w = self.attn(h, h, h, attn_mask=mask, need_weights=need_weights)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, attn_w


class MiniIOITransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, d_mlp=128, n_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(3, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_mlp) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, tokens, patch_pos=None, patch_vec=None,
                return_attn=False, return_residuals=False):
        B = tokens.shape[0]
        pos = torch.arange(3, device=tokens.device).unsqueeze(0).expand(B, -1)
        h = self.embed(tokens) + self.pos_embed(pos)
        if patch_pos is not None and patch_vec is not None:
            h[:, patch_pos, :] = patch_vec
        mask = torch.triu(torch.full((3, 3), float("-inf"), device=tokens.device), diagonal=1)

        attn_all, resid_all = [], []
        for block in self.blocks:
            h, aw = block(h, mask=mask, need_weights=return_attn)
            if return_attn:
                attn_all.append(aw.detach())
            if return_residuals:
                resid_all.append(h[:, 2, :].detach().clone())

        logits = self.unembed(self.ln_final(h[:, 2, :]))
        out = {"logits": logits}
        if return_attn:
            out["attn"] = attn_all
        if return_residuals:
            out["residuals"] = resid_all
        return out

    def logit_lens(self, residual):
        return self.unembed(self.ln_final(residual))


# ====================================================================
# DATA
# ====================================================================

def make_data(vocab_size):
    """Generate Type 1 (copy) and Type 2 (IOI) sequences."""
    # Type 1: [X, Y, Z] all different, target = Z.
    type1_seqs, type1_tgts = [], []
    for x in range(vocab_size):
        for y in range(vocab_size):
            if y == x:
                continue
            for z in range(vocab_size):
                if z == x or z == y:
                    continue
                type1_seqs.append([x, y, z])
                type1_tgts.append(z)

    # Type 2: [A, B, A] with A != B, target = B.
    type2_seqs, type2_tgts = [], []
    for a in range(vocab_size):
        for b in range(vocab_size):
            if a != b:
                type2_seqs.append([a, b, a])
                type2_tgts.append(b)

    return (torch.tensor(type1_seqs), torch.tensor(type1_tgts),
            torch.tensor(type2_seqs), torch.tensor(type2_tgts))


# ====================================================================
# EVALUATION (Type 2 only — the IOI analog)
# ====================================================================

def eval_type2(model, seqs2, tgts2, vocab_size, rng, do_patch=True):
    """Evaluate on Type 2 (IOI) sequences only."""
    model.eval()
    N = seqs2.shape[0]
    device = seqs2.device

    with torch.no_grad():
        out = model(seqs2, return_attn=True, return_residuals=True)
        logits = out["logits"]
        preds = logits.argmax(-1)

        a_tokens = seqs2[:, 0]  # The repeated token A
        b_tokens = tgts2         # The correct target B

        idx = torch.arange(N, device=device)
        acc = float((preds == b_tokens).float().mean())
        pred_A = float((preds == a_tokens).float().mean())

        # Logit difference: logit(B) - logit(A) — the IOI metric.
        logit_B = logits[idx, b_tokens].float()
        logit_A = logits[idx, a_tokens].float()
        ld = (logit_B - logit_A).cpu().numpy()
        ld_mean = float(ld.mean())
        ld_lo, ld_hi = bootstrap_ci(ld)

        # Per-layer attention from pos 2.
        attn = {}
        for li, aw in enumerate(out["attn"]):
            a2 = aw[:, 2, :].cpu().numpy()
            attn[f"L{li}"] = {
                "to_0": float(a2[:, 0].mean()),
                "to_1": float(a2[:, 1].mean()),
                "to_2": float(a2[:, 2].mean()),
            }

        # Logit lens per layer.
        ll = {}
        for li, res in enumerate(out["residuals"]):
            ll_logits = model.logit_lens(res)
            ll_B = ll_logits[idx, b_tokens].float()
            ll_A = ll_logits[idx, a_tokens].float()
            ll[f"L{li}"] = float((ll_B - ll_A).mean())

        result = {
            "acc": acc, "pred_A": pred_A, "ld_mean": ld_mean,
            "ld_ci": [ld_lo, ld_hi], "attn": attn, "logit_lens": ll,
        }

        # Patching: 3 positions x 2 methods.
        if do_patch:
            patch = {}
            for pos in [0, 1, 2]:
                for method in ["mean", "rand"]:
                    if method == "mean":
                        me = model.embed.weight.mean(0)
                        pv = (me + model.pos_embed.weight[pos]).unsqueeze(0).expand(N, -1)
                    else:
                        c = torch.zeros(N, dtype=torch.long, device=device)
                        for i in range(N):
                            pool = [t for t in range(vocab_size)
                                    if t != int(a_tokens[i]) and t != int(b_tokens[i])]
                            c[i] = pool[rng.integers(0, len(pool))]
                        pv = model.embed(c) + model.pos_embed.weight[pos].unsqueeze(0)

                    p_out = model(seqs2, patch_pos=pos, patch_vec=pv)
                    p_logits = p_out["logits"]
                    p_preds = p_logits.argmax(-1)
                    p_acc = float((p_preds == b_tokens).float().mean())
                    # Logit diff after patching.
                    p_ld = (p_logits[idx, b_tokens].float() - p_logits[idx, a_tokens].float()).cpu().numpy()
                    delta_ld = p_ld - ld
                    d_lo, d_hi = bootstrap_ci(delta_ld)
                    delta_acc = p_acc - acc

                    patch[f"p{pos}_{method}"] = {
                        "acc": p_acc, "delta_acc": delta_acc,
                        "ld": float(p_ld.mean()), "delta_ld": float(delta_ld.mean()),
                        "delta_ld_ci": [d_lo, d_hi],
                    }
            result["patch"] = patch

    model.train()
    return result


# ====================================================================
# TRAINING
# ====================================================================

def run_experiment(vocab_size, seed, n_layers=2, d_model=64, n_heads=4,
                   d_mlp=128, n_steps=15000, lr=1e-3, copy_frac=0.8,
                   eval_every=50, patch_every=200):

    log(f"  V={vocab_size}, seed={seed}, L={n_layers}, copy_frac={copy_frac}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    s1, t1, s2, t2 = make_data(vocab_size)
    s1, t1 = s1.to(DEVICE), t1.to(DEVICE)
    s2, t2 = s2.to(DEVICE), t2.to(DEVICE)
    N1, N2 = s1.shape[0], s2.shape[0]
    chance = 1.0 / (vocab_size - 1)
    log(f"    Type1={N1}, Type2={N2}, chance={chance*100:.1f}%")

    model = MiniIOITransformer(vocab_size, d_model, n_heads, d_mlp, n_layers).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps)

    BATCH = 256
    n_copy = int(BATCH * copy_frac)
    n_ioi = BATCH - n_copy

    trajectory = []
    for step in range(n_steps + 1):
        do_patch = (step % patch_every == 0)
        if step % eval_every == 0:
            r = eval_type2(model, s2, t2, vocab_size, rng, do_patch=do_patch)
            r["step"] = step
            trajectory.append(r)

            if step % 1000 == 0:
                ps = ""
                if do_patch and "patch" in r:
                    d_r = r["patch"]["p2_rand"]["delta_ld"]
                    d_m = r["patch"]["p2_mean"]["delta_ld"]
                    ps = f"  ΔLD_m={d_m:+.3f} ΔLD_r={d_r:+.3f}"
                ll_s = ""
                if "logit_lens" in r:
                    ll_v = [r["logit_lens"][f"L{l}"] for l in range(n_layers)]
                    ll_s = "  LL=[" + ",".join(f"{v:+.2f}" for v in ll_v) + "]"
                log(f"    step={step:>5}  acc={r['acc']:.3f}  pred_A={r['pred_A']:.3f}  "
                    f"LD={r['ld_mean']:+.3f}{ps}{ll_s}")

        if step < n_steps:
            # Sample mixed batch: copy_frac from Type 1, rest from Type 2.
            idx1 = torch.randint(0, N1, (n_copy,), device=DEVICE)
            idx2 = torch.randint(0, N2, (n_ioi,), device=DEVICE)
            batch_seqs = torch.cat([s1[idx1], s2[idx2]], dim=0)
            batch_tgts = torch.cat([t1[idx1], t2[idx2]], dim=0)

            out = model(batch_seqs)
            loss = F.cross_entropy(out["logits"], batch_tgts)
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()

    # === Summary ===
    train_traj = [r for r in trajectory if r["step"] > 0]
    min_acc_r = min(train_traj, key=lambda r: r["acc"])
    min_ld_r = min(train_traj, key=lambda r: r["ld_mean"])
    max_pred_A = max(train_traj, key=lambda r: r["pred_A"])

    # Sign flip detection on pos2_rand patching (delta_ld).
    patch_traj = [r for r in trajectory if "patch" in r]
    if patch_traj:
        dld = [(r["step"], r["patch"]["p2_rand"]["delta_ld"]) for r in patch_traj]
        max_pos_dld = max(d for _, d in dld)
        min_neg_dld = min(d for _, d in dld)
        sign_flip = None
        for i in range(1, len(dld)):
            if dld[i-1][1] > 0.05 and dld[i][1] < -0.05:
                sign_flip = dld[i][0]
                break
    else:
        max_pos_dld = min_neg_dld = 0
        sign_flip = None

    # Position specificity.
    pos_spec = {}
    if patch_traj:
        dip_r = min(patch_traj, key=lambda r: r["ld_mean"])
        mat_r = patch_traj[-1]
        for label, pr in [("dip", dip_r), ("mature", mat_r)]:
            pos_spec[label] = {"step": pr["step"]}
            for key in ["p0_rand", "p1_rand", "p2_rand"]:
                pos_spec[label][key] = pr["patch"][key]["delta_ld"]

    summary = {
        "vocab_size": vocab_size, "seed": seed, "n_layers": n_layers,
        "copy_frac": copy_frac, "chance": chance,
        "min_acc": min_acc_r["acc"], "min_acc_step": min_acc_r["step"],
        "dip_below_chance": min_acc_r["acc"] < chance,
        "min_ld": min_ld_r["ld_mean"], "min_ld_step": min_ld_r["step"],
        "max_pred_A": max_pred_A["pred_A"], "max_pred_A_step": max_pred_A["step"],
        "max_pos_delta_ld": max_pos_dld, "min_neg_delta_ld": min_neg_dld,
        "sign_flip_step": sign_flip, "sign_flip": sign_flip is not None,
        "final_acc": trajectory[-1]["acc"], "final_ld": trajectory[-1]["ld_mean"],
        "position_specificity": pos_spec,
    }

    log(f"\n    === SUMMARY V={vocab_size} seed={seed} L={n_layers} copy={copy_frac} ===")
    log(f"    Min acc: {summary['min_acc']:.3f} @ step {summary['min_acc_step']}  "
        f"(chance={chance:.3f})  dip={'YES' if summary['dip_below_chance'] else 'NO'}")
    log(f"    Min LD: {summary['min_ld']:+.3f} @ step {summary['min_ld_step']}")
    log(f"    Max pred_A: {summary['max_pred_A']:.3f} @ step {summary['max_pred_A_step']}")
    log(f"    Patching ΔLD pos2_rand: max={max_pos_dld:+.4f}  min={min_neg_dld:+.4f}")
    log(f"    Sign flip: {'step ' + str(sign_flip) if sign_flip else 'NOT DETECTED'}")
    log(f"    Final: acc={summary['final_acc']:.3f}  LD={summary['final_ld']:+.3f}")
    if pos_spec:
        for label in ["dip", "mature"]:
            ps = pos_spec.get(label, {})
            log(f"    Pos specificity ({label}, step {ps.get('step','?')}):")
            for k in ["p0_rand", "p1_rand", "p2_rand"]:
                if k in ps:
                    log(f"      {k}: ΔLD={ps[k]:+.4f}")
    log("")

    del model, opt; torch.cuda.empty_cache()
    return {"trajectory": trajectory, "summary": summary}


# ====================================================================
# MAIN
# ====================================================================

def main():
    os.makedirs("results", exist_ok=True)

    configs = [
        # Primary: 3 seeds.
        {"vocab_size": 10, "seed": 42},
        {"vocab_size": 10, "seed": 123},
        {"vocab_size": 10, "seed": 456},
        # Scale.
        {"vocab_size": 20, "seed": 42},
        # Vary copy fraction to control shortcut strength.
        {"vocab_size": 10, "seed": 42, "copy_frac": 0.9},
        {"vocab_size": 10, "seed": 42, "copy_frac": 0.95},
        # 1 layer (can it still suppress?)
        {"vocab_size": 10, "seed": 42, "n_layers": 1},
    ]

    all_results = {}
    t0 = time.time()

    for cfg in configs:
        label = (f"V{cfg['vocab_size']}_s{cfg['seed']}"
                 f"_L{cfg.get('n_layers',2)}_c{cfg.get('copy_frac',0.8)}")
        log(f"{'='*60}")
        log(f"Running: {label}")
        log(f"{'='*60}")
        result = run_experiment(**cfg)
        all_results[label] = result
        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2)

    # Cross-seed summary.
    log("=" * 60)
    log("CROSS-SEED SUMMARY (V=10, L=2, copy=0.8)")
    log("=" * 60)
    keys = [k for k in all_results if "L2_c0.8" in k and k.startswith("V10")]
    if keys:
        dips = [all_results[k]["summary"]["dip_below_chance"] for k in keys]
        flips = [all_results[k]["summary"]["sign_flip"] for k in keys]
        min_accs = [all_results[k]["summary"]["min_acc"] for k in keys]
        min_lds = [all_results[k]["summary"]["min_ld"] for k in keys]
        log(f"  Dip below chance: {sum(dips)}/{len(dips)}")
        log(f"  Sign flip: {sum(flips)}/{len(flips)}")
        log(f"  Min acc: {min(min_accs):.3f} - {max(min_accs):.3f}")
        log(f"  Min LD: {min(min_lds):+.3f} to {max(min_lds):+.3f}")

    elapsed = (time.time() - t0) / 60.0
    log(f"\nDone. Total: {elapsed:.1f} min. Output: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
