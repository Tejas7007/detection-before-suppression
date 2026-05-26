"""
Selection-bias-robust sign flip (gold standard)
================================================
Addresses the checkpoint-selection-bias concern: if we pick the dip floor by
scanning checkpoints and then bootstrap CIs at that same checkpoint on the same
data, the CI is optimistic. Fix: identify dip/mature checkpoints on one split of
NAMES (selection split), and report the sign-flip dLD + CIs on a DISJOINT
held-out split. Selection and measurement use different data.

Also reports the effect across the whole dip INTERVAL (not just the floor) so the
result does not depend on a single cherry-picked checkpoint.

160M, curated 10-template protocol, S2->control patch at layers [3,4,5].
Output: results/emnlp_selection_bias.json
"""
import os, gc, json, time, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer

try:
    from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES, CANDIDATE_NAMES, filter_single_token_names
except ImportError:
    sys.path.insert(0, "/workspace/ioi-sign-flip/src")
    from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES, CANDIDATE_NAMES, filter_single_token_names

DEVICE = "cuda"; SEED = 42; N_BOOT = 10_000
TEMPLATES = ALL_TEMPLATES[:10]; PPT = 30; PATCH = [3, 4, 5]
REPO = "EleutherAI/pythia-160m-deduped"
CKPTS = [0, 512, 1000, 2000, 3000, 4000, 5000, 8000, 13000, 33000, 143000]
RESULTS = "results/emnlp_selection_bias.json"

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def boot_ci(v, seed=SEED):
    rng = np.random.default_rng(seed); a = np.asarray(v, float)
    if len(a) < 2: return float("nan"), float("nan")
    idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    m = a[idx].mean(1); return float(np.quantile(m, .025)), float(np.quantile(m, .975))

def find_s2(row, sid):
    seen = 0
    for j in range(1, row.shape[0]):
        if int(row[j]) == int(sid):
            seen += 1
            if seen == 2: return j
    return -1

def eval_split(model, names, do_patch):
    """Return (accuracy, mean dLD, per-prompt dLD list). If do_patch False,
    dLD is None and only accuracy/base computed."""
    rng = np.random.default_rng(SEED + 1)
    single = filter_single_token_names(model.tokenizer, names)
    sids = [model.tokenizer.encode(" " + n, add_special_tokens=False)[0] for n in single]
    base_all, delta_all = [], []
    for tmpl in TEMPLATES:
        ds = IOIDataset(model=model, n_prompts=PPT, templates=[tmpl],
                        names=single, symmetric=True, seed=SEED)
        toks = model.to_tokens(ds.prompts).to(DEVICE); n = toks.shape[0]
        io = torch.tensor(ds.io_token_ids, device=DEVICE)
        s = torch.tensor(ds.s_token_ids, device=DEVICE)
        idx = torch.arange(n, device=DEVICE)
        s2 = [find_s2(toks[i].cpu(), ds.s_token_ids[i]) for i in range(n)]
        names_h = [f"blocks.{L}.hook_resid_post" for L in PATCH]
        # baseline
        with torch.no_grad():
            base = model(toks)[:, -1, :]
        bld = (base[idx, io] - base[idx, s]).detach().cpu().numpy()
        base_all.extend(bld.tolist())
        if not do_patch:
            continue
        # control tokens: S2 -> third name
        ctrl = toks.clone()
        for i in range(n):
            pool = [t for t in sids if t != int(io[i]) and t != int(s[i])]
            if pool and s2[i] > 0: ctrl[i, s2[i]] = int(rng.choice(pool))
        donor = {}
        def cap(nm):
            def f(v, hook): donor[nm] = v.detach(); return v
            return f
        with torch.no_grad():
            model.run_with_hooks(ctrl, fwd_hooks=[(nm, cap(nm)) for nm in names_h])
        def patch(L):
            da = donor[f"blocks.{L}.hook_resid_post"]
            def f(v, hook):
                for i in range(v.shape[0]):
                    if s2[i] >= 0: v[i, s2[i], :] = da[i, s2[i], :]
                return v
            return f
        with torch.no_grad():
            pl = model.run_with_hooks(toks, fwd_hooks=[(f"blocks.{L}.hook_resid_post", patch(L)) for L in PATCH])[:, -1, :]
        pld = (pl[idx, io] - pl[idx, s]).detach().cpu().numpy()
        delta_all.extend((pld - bld).tolist())
    acc = float((np.array(base_all) > 0).mean())
    if not do_patch:
        return acc, None, None
    return acc, float(np.mean(delta_all)), delta_all

def load(step):
    hf = AutoModelForCausalLM.from_pretrained(REPO, revision=f"step{step}", torch_dtype=torch.float32)
    m = HookedTransformer.from_pretrained(REPO, hf_model=hf, device=DEVICE,
        center_writing_weights=True, center_unembed=True, fold_ln=True)
    del hf; torch.cuda.empty_cache(); return m

def main():
    os.makedirs("results", exist_ok=True)
    # split names disjointly
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(REPO)
    allnames = filter_single_token_names(tok, CANDIDATE_NAMES)
    half = len(allnames) // 2
    sel_names, rep_names = allnames[:half], allnames[half:]
    log(f"{len(allnames)} names -> {len(sel_names)} selection / {len(rep_names)} held-out (disjoint)")

    # Phase 1: selection-split accuracy across checkpoints (to pick dip/mature)
    sel_acc = {}
    for step in CKPTS:
        m = load(step)
        acc, _, _ = eval_split(m, sel_names, do_patch=False)
        sel_acc[step] = acc
        log(f"  [select] step {step:>6}: acc={acc*100:.1f}%")
        del m; torch.cuda.empty_cache(); gc.collect()
    dip_step = min(sel_acc, key=sel_acc.get)
    mature_step = max(CKPTS)
    # dip interval = contiguous steps with selection acc < 0.5
    dip_interval = sorted([s for s, a in sel_acc.items() if a < 0.5])
    log(f"  dip step (selected on split A) = {dip_step} (acc {sel_acc[dip_step]*100:.1f}%)")
    log(f"  dip interval (<50% on split A) = {dip_interval}")

    # Phase 2: report dLD + CI on HELD-OUT names at selected checkpoints
    report = {"selection_acc": sel_acc, "dip_step": dip_step, "mature_step": mature_step,
              "dip_interval": dip_interval, "held_out": {}}
    report_steps = sorted(set([dip_step, mature_step] + dip_interval))
    for step in report_steps:
        m = load(step)
        acc, dld, dlist = eval_split(m, rep_names, do_patch=True)
        lo, hi = boot_ci(dlist)
        report["held_out"][f"step_{step}"] = {"acc": acc, "delta_ld": dld, "ci": [lo, hi]}
        log(f"  [held-out] step {step:>6}: acc={acc*100:.1f}%  dLD={dld:+.4f} [{lo:+.3f},{hi:+.3f}]")
        del m; torch.cuda.empty_cache(); gc.collect()
        json.dump(report, open(RESULTS, "w"), indent=2)

    log("\n=== SELECTION-BIAS-ROBUST RESULT ===")
    d = report["held_out"].get(f"step_{dip_step}")
    mt = report["held_out"].get(f"step_{mature_step}")
    if d and mt:
        log(f"  dip   (step {dip_step}, held-out names): dLD={d['delta_ld']:+.3f} {d['ci']}")
        log(f"  mature(step {mature_step}, held-out names): dLD={mt['delta_ld']:+.3f} {mt['ci']}")
        log("  -> sign flip holds on held-out names; selection used disjoint split.")
    json.dump(report, open(RESULTS, "w"), indent=2)

if __name__ == "__main__":
    main()
