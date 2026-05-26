"""
Robustness: clustered CIs + layer-window stability
===================================================
Addresses two key robustness concerns at once, on the 160M headline flip (dip step 2000,
mature step 143000), curated 10-template protocol, S2->control-name patch.

(A) Clustered CIs: example-level bootstrap can overstate certainty when prompts
    share templates/names. We report dLD CIs three ways -- example-clustered,
    TEMPLATE-clustered, and NAME-PAIR-clustered -- for the primary window.
(B) Layer-window stability: dLD at dip & mature for several windows around the
    primary [3,4,5], showing the sign flip is not an artifact of one window.

Output: results/emnlp_robustness_cis_layers.json
"""
import os, gc, json, time, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

try:
    from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES, CANDIDATE_NAMES, filter_single_token_names
except ImportError:
    sys.path.insert(0, "/workspace/ioi-sign-flip/src")
    from circuitscaling.datasets import IOIDataset, ALL_TEMPLATES, CANDIDATE_NAMES, filter_single_token_names

DEVICE="cuda"; SEED=42; NB=10_000
REPO="EleutherAI/pythia-160m-deduped"
TEMPLATES=ALL_TEMPLATES[:10]; PPT=30
PRIMARY=[3,4,5]
WINDOWS=[[2,3,4],[3,4,5],[4,5,6],[5,6,7],[3,4,5,6]]
CKPTS={"dip":2000,"mature":143000}
RESULTS="results/emnlp_robustness_cis_layers.json"

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def boot(vals, seed=SEED):
    rng=np.random.default_rng(seed); a=np.asarray(vals,float)
    if len(a)<2: return [float('nan')]*2
    idx=rng.integers(0,len(a),size=(NB,len(a))); m=a[idx].mean(1)
    return [float(np.quantile(m,.025)),float(np.quantile(m,.975))]

def boot_cluster(vals, labels, seed=SEED):
    """Cluster bootstrap: resample clusters with replacement, average their obs."""
    rng=np.random.default_rng(seed); a=np.asarray(vals,float); lab=np.asarray(labels)
    clusters=np.unique(lab); by={c:a[lab==c] for c in clusters}
    means=[]
    for _ in range(NB):
        pick=rng.choice(clusters,size=len(clusters),replace=True)
        obs=np.concatenate([by[c] for c in pick])
        means.append(obs.mean())
    means=np.array(means)
    return [float(np.quantile(means,.025)),float(np.quantile(means,.975))]

def find_s2(row,sid):
    seen=0
    for j in range(1,row.shape[0]):
        if int(row[j])==int(sid):
            seen+=1
            if seen==2: return j
    return -1

def load(step):
    hf=AutoModelForCausalLM.from_pretrained(REPO,revision=f"step{step}",torch_dtype=torch.float32)
    m=HookedTransformer.from_pretrained(REPO,hf_model=hf,device=DEVICE,
        center_writing_weights=True,center_unembed=True,fold_ln=True)
    del hf; torch.cuda.empty_cache(); return m

def collect(model):
    """Per-example: base LD, donor activations cached, s2 pos, template idx,
    name-pair id. Returns structures to compute dLD per layer-window."""
    rng=np.random.default_rng(SEED+1)
    single=filter_single_token_names(model.tokenizer,CANDIDATE_NAMES)
    sids=[model.tokenizer.encode(" "+n,add_special_tokens=False)[0] for n in single]
    recs=[]  # each: dict with toks, ctrl, io, s, s2, tmpl, pair
    for ti,tmpl in enumerate(TEMPLATES):
        ds=IOIDataset(model=model,n_prompts=PPT,templates=[tmpl],names=single,symmetric=True,seed=SEED)
        toks=model.to_tokens(ds.prompts).to(DEVICE); n=toks.shape[0]
        for i in range(n):
            io=ds.io_token_ids[i]; s=ds.s_token_ids[i]
            s2=find_s2(toks[i].cpu(),s)
            ctrl=toks[i].clone()
            pool=[t for t in sids if t!=io and t!=s]
            if pool and s2>0: ctrl[s2]=int(rng.choice(pool))
            recs.append({"tok":toks[i:i+1],"ctrl":ctrl.unsqueeze(0),"io":io,"s":s,
                         "s2":s2,"tmpl":ti,"pair":f"{io}_{s}"})
    return recs

def dld_for_window(model, recs, window):
    names_h=[f"blocks.{L}.hook_resid_post" for L in window]
    deltas=[]; tmpls=[]; pairs=[]
    for r in recs:
        if r["s2"]<0: continue
        with torch.no_grad():
            base=model(r["tok"])[0,-1,:]
        bld=float(base[r["io"]]-base[r["s"]])
        donor={}
        def cap(nm):
            def f(v,hook): donor[nm]=v.detach(); return v
            return f
        with torch.no_grad():
            model.run_with_hooks(r["ctrl"],fwd_hooks=[(nm,cap(nm)) for nm in names_h])
        def patch(L):
            da=donor[f"blocks.{L}.hook_resid_post"]
            def f(v,hook): v[0,r["s2"],:]=da[0,r["s2"],:]; return v
            return f
        with torch.no_grad():
            pl=model.run_with_hooks(r["tok"],fwd_hooks=[(f"blocks.{L}.hook_resid_post",patch(L)) for L in window])[0,-1,:]
        pld=float(pl[r["io"]]-pl[r["s"]])
        deltas.append(pld-bld); tmpls.append(r["tmpl"]); pairs.append(r["pair"])
    return np.array(deltas),np.array(tmpls),np.array(pairs)

def main():
    os.makedirs("results",exist_ok=True)
    out={"primary_window":PRIMARY,"clustered_ci":{},"layer_windows":{}}
    for phase,step in CKPTS.items():
        log(f"=== {phase} (step {step}) ===")
        m=load(step); recs=collect(m)
        # (A) primary window: three CI flavors
        d,tm,pr=dld_for_window(m,recs,PRIMARY)
        out["clustered_ci"][phase]={
            "mean_dld":float(d.mean()),"n":int(len(d)),
            "ci_example":boot(d),
            "ci_template_clustered":boot_cluster(d,tm),
            "ci_namepair_clustered":boot_cluster(d,pr),
        }
        log(f"  primary {PRIMARY}: dLD={d.mean():+.3f}  "
            f"ex{out['clustered_ci'][phase]['ci_example']}  "
            f"tmpl{out['clustered_ci'][phase]['ci_template_clustered']}  "
            f"pair{out['clustered_ci'][phase]['ci_namepair_clustered']}")
        # (B) layer-window sweep
        out["layer_windows"].setdefault(phase,{})
        for w in WINDOWS:
            dw,_,_=dld_for_window(m,recs,w)
            out["layer_windows"][phase]["_".join(map(str,w))]=float(dw.mean())
            log(f"  window {w}: dLD={dw.mean():+.3f}")
        del m; torch.cuda.empty_cache(); gc.collect()
        json.dump(out,open(RESULTS,"w"),indent=2)
    log("\n=== SUMMARY ===")
    for phase in CKPTS:
        c=out["clustered_ci"][phase]
        log(f"  {phase}: dLD={c['mean_dld']:+.3f}  example{c['ci_example']}  "
            f"template{c['ci_template_clustered']}  namepair{c['ci_namepair_clustered']}")
        log(f"    windows: {out['layer_windows'][phase]}")
    json.dump(out,open(RESULTS,"w"),indent=2)

if __name__=="__main__":
    main()
