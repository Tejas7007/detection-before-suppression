"""
Mini-IOI Scaled: Fix the 2-layer sign flip
===========================================

The V=10 mini-IOI worked for 1-layer but the 2-layer model recovered
too fast (100% by step 50). The model memorized 90 IOI sequences instantly.

Fix: scale up the vocabulary so the model CAN'T memorize. It has to
actually learn the shortcut-then-suppress pattern.

Sweep:
- V in {50, 100}: 2450-9900 IOI sequences (vs 90 before)
- d_model in {32, 64}: capacity control
- copy_frac in {0.9, 0.95}: shortcut strength
- lr in {3e-4, 1e-4}: convergence speed
- All 2-layer, 4 heads
- 3 seeds for the best config

Key metric: does the 2-LAYER model show sign_flip_detected=True?
"""

import os, json, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda"
SEED = 42
RESULTS_PATH = "results/emnlp_mini_ioi_scaled.json"

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def bootstrap_ci(values, seed=42):
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0: return float("nan"), float("nan")
    idx = rng.integers(0, len(arr), size=(10000, len(arr)))
    means = arr[idx].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


class TBlock(nn.Module):
    def __init__(self, d, nh, dm):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, nh, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, dm), nn.GELU(), nn.Linear(dm, d))
    def forward(self, x, mask):
        h = self.ln1(x)
        a, aw = self.attn(h, h, h, attn_mask=mask, need_weights=True)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, aw


class Model(nn.Module):
    def __init__(self, V, d=64, nh=4, dm=128, nl=2):
        super().__init__()
        self.V, self.d, self.nl = V, d, nl
        self.emb = nn.Embedding(V, d)
        self.pos = nn.Embedding(3, d)
        self.blocks = nn.ModuleList([TBlock(d, nh, dm) for _ in range(nl)])
        self.ln = nn.LayerNorm(d)
        self.out = nn.Linear(d, V, bias=False)

    def forward(self, tok, pp=None, pv=None):
        B = tok.shape[0]
        p = torch.arange(3, device=tok.device).unsqueeze(0).expand(B, -1)
        h = self.emb(tok) + self.pos(p)
        if pp is not None and pv is not None:
            h[:, pp, :] = pv
        mask = torch.triu(torch.full((3,3), float("-inf"), device=tok.device), diagonal=1)
        attns = []
        for b in self.blocks:
            h, aw = b(h, mask)
            attns.append(aw.detach())
        return self.out(self.ln(h[:, 2, :])), attns


def make_data(V):
    s1, t1, s2, t2 = [], [], [], []
    for x in range(V):
        for y in range(V):
            if y == x: continue
            for z in range(V):
                if z == x or z == y: continue
                s1.append([x, y, z]); t1.append(z)
    for a in range(V):
        for b in range(V):
            if a != b:
                s2.append([a, b, a]); t2.append(b)
    return (torch.tensor(s1), torch.tensor(t1),
            torch.tensor(s2), torch.tensor(t2))


def evaluate(model, s2, t2, V, rng, do_patch=True):
    model.eval()
    N = s2.shape[0]
    dev = s2.device
    with torch.no_grad():
        # Process in chunks to avoid OOM.
        CHUNK = 2048
        all_logits = []
        for i in range(0, N, CHUNK):
            lo, _ = model(s2[i:i+CHUNK])
            all_logits.append(lo)
        logits = torch.cat(all_logits, dim=0)

        preds = logits.argmax(-1)
        a_tok = s2[:, 0]
        b_tok = t2
        idx = torch.arange(N, device=dev)

        acc = float((preds == b_tok).float().mean())
        pred_A = float((preds == a_tok).float().mean())
        ld = (logits[idx, b_tok].float() - logits[idx, a_tok].float()).cpu().numpy()
        ld_mean = float(ld.mean())

        result = {"acc": acc, "pred_A": pred_A, "ld_mean": ld_mean}

        if do_patch:
            for pos in [0, 2]:
                # Random token replacement.
                c = torch.zeros(N, dtype=torch.long, device=dev)
                for i in range(N):
                    pool = [t for t in range(V) if t != int(a_tok[i]) and t != int(b_tok[i])]
                    c[i] = pool[rng.integers(0, len(pool))]
                pv = model.emb(c) + model.pos.weight[pos].unsqueeze(0)

                p_logits_all = []
                for i in range(0, N, CHUNK):
                    chunk_pv = pv[i:i+CHUNK]
                    # Need to manually patch for chunks.
                    toks = s2[i:i+CHUNK].clone()
                    Bc = toks.shape[0]
                    p_idx = torch.arange(3, device=dev).unsqueeze(0).expand(Bc, -1)
                    h = model.emb(toks) + model.pos(p_idx)
                    h[:, pos, :] = chunk_pv
                    mask = torch.triu(torch.full((3,3), float("-inf"), device=dev), diagonal=1)
                    for b in model.blocks:
                        h, _ = b(h, mask)
                    p_logits_all.append(model.out(model.ln(h[:, 2, :])))
                p_logits = torch.cat(p_logits_all, dim=0)

                p_ld = (p_logits[idx, b_tok].float() - p_logits[idx, a_tok].float()).cpu().numpy()
                delta_ld = p_ld - ld
                d_lo, d_hi = bootstrap_ci(delta_ld)
                result[f"p{pos}_rand"] = {
                    "delta_ld": float(delta_ld.mean()),
                    "ci": [d_lo, d_hi],
                }
    model.train()
    return result


def run_config(V, d_model, lr, copy_frac, seed, n_steps=50000,
               eval_every=100, patch_every=500):
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)

    s1, t1, s2, t2 = make_data(V)
    s1, t1 = s1.to(DEVICE), t1.to(DEVICE)
    s2, t2 = s2.to(DEVICE), t2.to(DEVICE)
    N1, N2 = s1.shape[0], s2.shape[0]
    chance = 1.0 / (V - 1)

    log(f"    V={V} d={d_model} lr={lr} cf={copy_frac} seed={seed}  "
        f"N1={N1} N2={N2} chance={chance*100:.1f}%")

    model = Model(V, d=d_model, nh=4, dm=d_model*2, nl=2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    BATCH = min(512, N1)
    n_copy = int(BATCH * copy_frac)
    n_ioi = BATCH - n_copy

    traj = []
    for step in range(n_steps + 1):
        do_p = (step % patch_every == 0)
        if step % eval_every == 0:
            r = evaluate(model, s2, t2, V, rng, do_patch=do_p)
            r["step"] = step
            traj.append(r)
            if step % 5000 == 0:
                ps = ""
                if do_p and "p2_rand" in r:
                    ps = f"  ΔLD_p2={r['p2_rand']['delta_ld']:+.3f}"
                log(f"      step={step:>5}  acc={r['acc']:.3f}  "
                    f"pred_A={r['pred_A']:.3f}  LD={r['ld_mean']:+.3f}{ps}")

        if step < n_steps:
            idx1 = torch.randint(0, N1, (n_copy,), device=DEVICE)
            idx2 = torch.randint(0, N2, (n_ioi,), device=DEVICE)
            seqs = torch.cat([s1[idx1], s2[idx2]])
            tgts = torch.cat([t1[idx1], t2[idx2]])
            lo, _ = model(seqs)
            loss = F.cross_entropy(lo, tgts)
            opt.zero_grad(); loss.backward(); opt.step()

    # Summary.
    tt = [r for r in traj if r["step"] > 0]
    min_acc = min(tt, key=lambda r: r["acc"])
    min_ld = min(tt, key=lambda r: r["ld_mean"])
    max_pA = max(tt, key=lambda r: r["pred_A"])

    pt = [r for r in traj if "p2_rand" in r]
    if pt:
        dlds = [(r["step"], r["p2_rand"]["delta_ld"]) for r in pt]
        mx = max(d for _, d in dlds)
        mn = min(d for _, d in dlds)
        sf = None
        for i in range(1, len(dlds)):
            if dlds[i-1][1] > 0.1 and dlds[i][1] < -0.1:
                sf = dlds[i][0]; break
    else:
        mx = mn = 0; sf = None

    summary = {
        "V": V, "d": d_model, "lr": lr, "cf": copy_frac, "seed": seed,
        "chance": chance,
        "min_acc": min_acc["acc"], "min_acc_step": min_acc["step"],
        "dip": min_acc["acc"] < chance,
        "min_ld": min_ld["ld_mean"], "min_ld_step": min_ld["step"],
        "max_pA": max_pA["pred_A"], "max_pA_step": max_pA["step"],
        "max_dld": mx, "min_dld": mn,
        "sign_flip": sf is not None, "sign_flip_step": sf,
        "final_acc": traj[-1]["acc"], "final_ld": traj[-1]["ld_mean"],
    }

    log(f"    => min_acc={summary['min_acc']:.3f}@{summary['min_acc_step']}  "
        f"dip={'YES' if summary['dip'] else 'NO'}  "
        f"max_pA={summary['max_pA']:.3f}  "
        f"ΔLD=[{mx:+.3f},{mn:+.3f}]  "
        f"flip={'step '+str(sf) if sf else 'NO'}  "
        f"final_acc={summary['final_acc']:.3f}\n")

    del model, opt; torch.cuda.empty_cache(); gc.collect()
    return {"trajectory": traj, "summary": summary}


def main():
    os.makedirs("results", exist_ok=True)
    results = {}

    # Phase 1: Coarse sweep to find configs where 2-layer shows the dip.
    log("PHASE 1: Coarse sweep")
    sweep = [
        {"V": 50,  "d_model": 32, "lr": 3e-4, "copy_frac": 0.9,  "seed": 42, "n_steps": 30000},
        {"V": 50,  "d_model": 32, "lr": 1e-4, "copy_frac": 0.95, "seed": 42, "n_steps": 30000},
        {"V": 100, "d_model": 32, "lr": 3e-4, "copy_frac": 0.9,  "seed": 42, "n_steps": 30000},
        {"V": 100, "d_model": 32, "lr": 1e-4, "copy_frac": 0.95, "seed": 42, "n_steps": 30000},
        {"V": 100, "d_model": 64, "lr": 1e-4, "copy_frac": 0.95, "seed": 42, "n_steps": 30000},
        {"V": 50,  "d_model": 32, "lr": 1e-4, "copy_frac": 0.98, "seed": 42, "n_steps": 30000},
        {"V": 50,  "d_model": 16, "lr": 1e-4, "copy_frac": 0.95, "seed": 42, "n_steps": 50000},
        {"V": 100, "d_model": 32, "lr": 3e-5, "copy_frac": 0.95, "seed": 42, "n_steps": 50000},
    ]

    for cfg in sweep:
        label = f"V{cfg['V']}_d{cfg['d_model']}_lr{cfg['lr']}_cf{cfg['copy_frac']}"
        if label in results:
            log(f"SKIP {label}")
            continue
        log(f"Running: {label}")
        r = run_config(**cfg)
        results[label] = r
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    # Phase 2: Find best config (has sign flip) and run 2 more seeds.
    log("\nPHASE 2: Multi-seed on best config")
    best = None
    for label, r in results.items():
        s = r["summary"]
        if s.get("sign_flip"):
            if best is None or s["max_dld"] > results[best]["summary"]["max_dld"]:
                best = label
    if best is None:
        # Fall back to config with largest positive delta_ld.
        best = max(results.keys(),
                   key=lambda k: results[k]["summary"]["max_dld"])
        log(f"  No sign flip found. Using config with largest max_dld: {best}")
    else:
        log(f"  Best config with sign flip: {best}")

    s = results[best]["summary"]
    for extra_seed in [123, 456]:
        label2 = f"{best}_s{extra_seed}"
        if label2 in results:
            log(f"SKIP {label2}")
            continue
        log(f"Running: {label2}")
        r = run_config(V=s["V"], d_model=s["d"], lr=s["lr"],
                       copy_frac=s["cf"], seed=extra_seed,
                       n_steps=50000 if s.get("sign_flip_step", 99999) > 20000 else 30000)
        results[label2] = r
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    # Summary.
    log("\n" + "=" * 60)
    log("FINAL SUMMARY")
    log("=" * 60)
    for label in sorted(results.keys()):
        s = results[label]["summary"]
        log(f"  {label}: dip={s['dip']}  flip={s['sign_flip']}  "
            f"min_acc={s['min_acc']:.3f}  ΔLD=[{s['max_dld']:+.3f},{s['min_dld']:+.3f}]")


if __name__ == "__main__":
    main()
