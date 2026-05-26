"""
SVA causal trajectory (boundary case)
======================================
We have the SVA attractor intervention only at one checkpoint (step 512). The
proof chain needs it across training: IOI/GT FLIP sign (helpful at dip, harmful
at maturity) because the duplicate is later repurposed for suppression; SVA's
attractor is ALWAYS misleading and never repurposed, so we predict the
intervention does NOT flip -- it helps (or is neutral) at both the dip and
maturity. Confirming this turns SVA from a behavioral side-note into a confirmed
prediction of the account (and makes the falsification statement principled).

Patch the attractor noun toward the subject's number (phase5 methodology),
metric = P(correct verb) - P(attractor verb), across dip/recovery/mature.
160M. Output: results/emnlp_sva_causal_trajectory.json
"""
import os, gc, json, time, sys
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer

DEVICE = "cuda"; SEED = 42; N_BOOT = 10_000
REPO = "EleutherAI/pythia-160m-deduped"
PATCH = [3, 4, 5]
CKPTS = [256, 512, 1000, 2000, 5000, 13000, 143000]  # dip -> mature
N = 200
RESULTS = "results/emnlp_sva_causal_trajectory.json"

SING = ["key", "author", "girl", "boy", "dog", "cat", "man", "woman", "child", "king"]
PLUR = ["keys", "authors", "girls", "boys", "dogs", "cats", "men", "women", "children", "kings"]
PREP = ["near", "beside", "with", "behind"]
VERBS = [("is", "are"), ("was", "were"), ("has", "have")]

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)
def boot_ci(v, seed=SEED):
    rng = np.random.default_rng(seed); a = np.asarray(v, float)
    if len(a) < 2: return float("nan"), float("nan")
    idx = rng.integers(0, len(a), size=(N_BOOT, len(a)))
    m = a[idx].mean(1); return float(np.quantile(m, .025)), float(np.quantile(m, .975))

def load(step):
    hf = AutoModelForCausalLM.from_pretrained(REPO, revision=f"step{step}", torch_dtype=torch.float32)
    m = HookedTransformer.from_pretrained(REPO, hf_model=hf, device=DEVICE,
        center_writing_weights=True, center_unembed=True, fold_ln=True)
    del hf; torch.cuda.empty_cache(); return m

def valid_verbs(tok):
    out = []
    for s, p in VERBS:
        si = tok.encode(" " + s, add_special_tokens=False)
        pi = tok.encode(" " + p, add_special_tokens=False)
        if len(si) == 1 and len(pi) == 1:
            out.append({"singular": (s, si[0]), "plural": (p, pi[0])})
    return out

def build_pairs(tok):
    rng = np.random.default_rng(SEED + 200)
    vv = valid_verbs(tok); pairs = list(zip(SING, PLUR)); items = []
    for _ in range(N):
        subj_sing = bool(rng.integers(0, 2))
        i = int(rng.integers(0, len(pairs))); j = int(rng.integers(0, len(pairs)))
        if subj_sing:
            subj = pairs[i][0]; attr_m = pairs[j][1]; attr_c = pairs[j][0]
        else:
            subj = pairs[i][1]; attr_m = pairs[j][0]; attr_c = pairs[j][1]
        prep = rng.choice(PREP); vp = vv[int(rng.integers(0, len(vv)))]
        cid = vp["singular" if subj_sing else "plural"][1]
        aid = vp["plural" if subj_sing else "singular"][1]
        items.append({"main": f"The {subj} {prep} the {attr_m}",
                      "ctrl": f"The {subj} {prep} the {attr_c}",
                      "cid": cid, "aid": aid})
    return items

def run(model, items):
    base_d, patch_d = [], []
    names_h = [f"blocks.{L}.hook_resid_post" for L in PATCH]
    for it in items:
        mt = model.to_tokens(it["main"]).to(DEVICE)
        ct = model.to_tokens(it["ctrl"]).to(DEVICE)
        if mt.shape[1] != ct.shape[1]: continue
        dpos = -1
        for j in range(1, mt.shape[1]):
            if int(mt[0, j]) != int(ct[0, j]): dpos = j; break
        if dpos < 0: continue
        with torch.no_grad():
            bl = model(mt)[0, -1, :]
        bp = F.softmax(bl.float(), -1)
        base_d.append(float(bp[it["cid"]] - bp[it["aid"]]))
        donor = {}
        def cap(nm):
            def f(v, hook): donor[nm] = v.detach(); return v
            return f
        with torch.no_grad():
            model.run_with_hooks(ct, fwd_hooks=[(nm, cap(nm)) for nm in names_h])
        def patch(L):
            da = donor[f"blocks.{L}.hook_resid_post"]
            def f(v, hook):
                v[0, dpos, :] = da[0, dpos, :]; return v
            return f
        with torch.no_grad():
            pl = model.run_with_hooks(mt, fwd_hooks=[(f"blocks.{L}.hook_resid_post", patch(L)) for L in PATCH])[0, -1, :]
        pp = F.softmax(pl.float(), -1)
        patch_d.append(float(pp[it["cid"]] - pp[it["aid"]]))
    base = np.array(base_d); patched = np.array(patch_d)
    delta = patched - base; lo, hi = boot_ci(delta)
    return {"n": len(base), "base_diff": float(base.mean()),
            "patched_diff": float(patched.mean()), "delta": float(delta.mean()),
            "delta_ci": [lo, hi], "acc": float((base > 0).mean())}

def main():
    os.makedirs("results", exist_ok=True)
    from transformers import AutoTokenizer
    items = build_pairs(AutoTokenizer.from_pretrained(REPO))
    res = {"checkpoints": {}}
    for step in CKPTS:
        m = load(step)
        r = run(m, items)
        res["checkpoints"][f"step_{step}"] = r
        log(f"  step {step:>6}: SVA_acc={r['acc']*100:5.1f}%  base={r['base_diff']:+.4f}  "
            f"delta={r['delta']:+.4f} [{r['delta_ci'][0]:+.3f},{r['delta_ci'][1]:+.3f}]")
        del m; torch.cuda.empty_cache(); gc.collect()
        json.dump(res, open(RESULTS, "w"), indent=2)
    log("\n=== SVA BOUNDARY ===")
    deltas = [(int(k.split('_')[1]), v['delta']) for k, v in res["checkpoints"].items()]
    deltas.sort()
    signs = set(np.sign(d) for _, d in deltas if abs(d) > 0.002)
    log(f"  deltas across training: {[(s, round(d,3)) for s,d in deltas]}")
    log(f"  -> {'NO sign flip (boundary confirmed)' if len(signs)<=1 else 'sign change present (unexpected)'}")

if __name__ == "__main__":
    main()
