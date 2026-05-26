"""
Generate EMNLP main-text figures from the new (curated-protocol) results.
Run on the pod where ALL result JSONs exist. Outputs PNG + PDF to figures/emnlp/.

Figures:
  fig1_dip_and_signflip   : (a) IOI acc dip 3 scales  (b) ΔLD sign flip 3 scales
  fig2_mechanism_timeline : induction + suppression-head ablation + acc + LM loss (160M)
  fig3_generalization     : GT causal ΔLD sign flip 3 scales (+ SVA boundary if available)
  fig4_controlled_dip     : injection-rate-controlled dip with seed bands
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "figures/emnlp"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5,
                     "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "legend.fontsize": 9.5, "figure.dpi": 150, "savefig.dpi": 300,
                     "axes.spines.top": False, "axes.spines.right": False})
C = {"160m": "#2196F3", "410m": "#FF9800", "1b": "#4CAF50"}

def load(p):
    try:
        return json.load(open(p))
    except Exception as e:
        print(f"  [skip] {p}: {e}"); return None

def save(name):
    plt.savefig(f"{OUT}/{name}.png", bbox_inches="tight")
    plt.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    plt.close(); print(f"  wrote {name}")

def steps_vals(by_step, field):
    xs, ys = [], []
    for k in sorted(by_step, key=lambda x: int(x.split("_")[1])):
        v = by_step[k]
        if isinstance(v, dict) and field in v and "error" not in v:
            xs.append(int(k.split("_")[1])); ys.append(v[field])
    return np.array(xs), np.array(ys)

# ---------- Fig 1: the whole paper in one image (3 panels) ----------
def fig1():
    d = load("results/emnlp_consistent_signflip.json")
    if not d: return
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11.2, 3.5))
    series = [("pythia-160m", C["160m"], "160M"),
              ("pythia-410m", C["410m"], "410M"),
              ("pythia-1b", C["1b"], "1B")]
    # dip x-range from 160M for shading the below-chance window
    bs160 = d["pythia-160m"]["by_step"]
    xa, acc160 = steps_vals(bs160, "ioi_acc")
    dipx = xa[acc160 < 0.5]
    dlo, dhi = (dipx.min(), dipx.max()) if len(dipx) else (None, None)
    for m, c, lbl in series:
        if m not in d: continue
        bs = d[m]["by_step"]
        x, acc = steps_vals(bs, "ioi_acc");        a1.plot(x, acc*100, "o-", color=c, ms=4, lw=2.3, label=lbl)
        xb, bld = steps_vals(bs, "base_ld_mean");  a2.plot(xb, bld, "o-", color=c, ms=4, lw=2.3, label=lbl)
        x2, dld = steps_vals(bs, "delta_ld_mean"); a3.plot(x2, dld, "o-", color=c, ms=4, lw=2.3, label=lbl)
    for ax in (a1, a2, a3):
        ax.set_xscale("log"); ax.set_xlabel("Training step")
        if dlo: ax.axvspan(dlo, dhi, color="red", alpha=0.07)
    # (a) accuracy
    a1.axhline(50, color="0.35", ls="--", lw=1.4)
    a1.axhspan(0, 50, color="red", alpha=0.05)
    a1.set_ylabel("IOI accuracy (%)"); a1.set_ylim(0, 105)
    a1.set_title("(a) Accuracy dips below chance")
    a1.text(0.5, 0.07, "below chance", transform=a1.transAxes, ha="center",
            fontsize=9.5, color="#B23B3B", fontweight="bold")
    a1.legend(loc="center right", title="scale")
    # (b) continuous LD -> kills the metric-artifact objection
    a2.axhline(0, color="0.35", ls="--", lw=1.4)
    ylo = min(a2.get_ylim()[0], -1); a2.set_ylim(ylo, a2.get_ylim()[1])
    a2.axhspan(ylo, 0, color="red", alpha=0.05)
    a2.set_ylabel("LD = logit(IO) $-$ logit(S)")
    a2.set_title("(b) Continuous LD also goes negative")
    a2.text(0.96, 0.05, "favors wrong name", transform=a2.transAxes, ha="right", va="bottom",
            fontsize=9.5, color="#B23B3B", fontweight="bold")
    # (c) intervention sign flip -- the headline
    a3.axhline(0, color="0.35", ls="--", lw=1.4)
    a3.set_ylabel(r"$\Delta$LD (S2 perturbation)")
    a3.set_title("(c) S2 perturbation flips sign")
    a3.text(0.66, 0.95, "$\\mathbf{\\Delta LD > 0}$:\nS2 perturbation helps", transform=a3.transAxes,
            ha="center", va="top", fontsize=9, color="#2E7D46", fontweight="bold")
    a3.text(0.03, 0.06, "$\\mathbf{\\Delta LD < 0}$:\nS2 perturbation hurts", transform=a3.transAxes,
            ha="left", va="bottom", fontsize=9, color="#B23B3B", fontweight="bold")
    plt.tight_layout(); save("fig1_dip_and_signflip")

# ---------- Fig 2: mechanism timeline (160M) ----------
def fig2():
    ind = load("results/emnlp_induction_timeline.json")
    abl = load("results/emnlp_suppression_head_ablation.json")
    if not ind: return
    bs = ind["pythia-160m"]["by_step"]
    x, acc = steps_vals(bs, "ioi_acc")
    _, imax = steps_vals(bs, "induction_max")
    _, loss = steps_vals(bs, "lm_loss")
    fig, ax = plt.subplots(figsize=(5.3, 4.7))
    dip_mask = acc < 0.5
    floor_step = x[np.argmin(acc)]
    rec = x[(x > floor_step) & (acc > 0.9)]
    rec_step = rec.min() if len(rec) else None
    if dip_mask.any():
        ax.axvspan(x[dip_mask].min(), x[dip_mask].max(), color="red", alpha=0.07)
    ax.axvline(floor_step, color="#B23B3B", ls=":", lw=1.5)
    ax.text(floor_step, 1.04, "dip floor", color="#B23B3B", fontsize=11, ha="center", fontweight="bold")
    if rec_step:
        ax.axvline(rec_step, color="#2E7D46", ls=":", lw=1.5)
        ax.text(rec_step, 1.04, "recovery", color="#2E7D46", fontsize=11, ha="center", fontweight="bold")
    # LM loss as a faint normalized background line (no second axis)
    lossn = (loss - loss.min()) / (loss.max() - loss.min() + 1e-9)
    ax.plot(x, lossn, ":", color="#9C8AB0", lw=2.0, label="LM loss (normalized)")
    ax.plot(x, acc, "o-", color="#37474F", ms=5, lw=2.4, label="IOI accuracy")
    ax.plot(x, imax, "s-", color="#1976D2", ms=5, lw=2.2, label="Detection (induction)")
    if abl and "pythia-160m" in abl:
        xa, dl = steps_vals(abl["pythia-160m"]["by_step"], "ablation_delta")
        dln = -np.array(dl); dln = dln / (dln.max() + 1e-9)
        ax.plot(xa, dln, "^-", color="#C62828", ms=5, lw=2.2,
                label="Suppression (head ablation)")
    ax.axhline(0.5, color="0.5", ls="--", lw=1)
    ax.set_xscale("log"); ax.set_xlabel("Training step", fontsize=13.5)
    ax.set_ylabel("normalized value", fontsize=13.5); ax.set_ylim(-0.05, 1.14)
    ax.tick_params(labelsize=11.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2,
              fontsize=10.5, framealpha=0.95, columnspacing=1.2)
    ax.set_title("Detection before suppression (Pythia-160M)", fontsize=14)
    plt.tight_layout(); save("fig2_mechanism_timeline")

# ---------- Fig 3: GT causal sign flip ----------
def fig3():
    d = load("results/emnlp_gt_causal_scale.json")
    sva = load("results/emnlp_sva_causal_trajectory.json")
    if not d: return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.7))
    # (a) GT: causal sign flip across scale, with CI bands
    for m, c in [("pythia-160m", C["160m"]), ("pythia-410m", C["410m"]), ("pythia-1b", C["1b"])]:
        if m not in d: continue
        bs = d[m]["by_step"]
        xs = sorted(int(k.split("_")[1]) for k in bs)
        dl = [bs[f"step_{x}"]["delta_mean"] for x in xs]
        lo = [bs[f"step_{x}"]["delta_ci"][0] for x in xs]
        hi = [bs[f"step_{x}"]["delta_ci"][1] for x in xs]
        a1.plot(xs, dl, "o-", color=c, ms=4, lw=2.3, label=m.replace("pythia-", "").upper())
        a1.fill_between(xs, lo, hi, color=c, alpha=0.15)
    a1.axhline(0, color="0.35", ls="--", lw=1.4)
    a1.set_xscale("log"); a1.set_xlabel("Training step")
    a1.set_ylabel(r"$\Delta$ logit diff")
    a1.set_title("(a) Greater-than: sign flip"); a1.legend(loc="upper right", title="scale")
    a1.text(0.04, 0.93, "helps during dip", transform=a1.transAxes, ha="left", va="top",
            fontsize=9, color="#2E7D46", fontweight="bold")
    a1.text(0.96, 0.07, "harms at maturity", transform=a1.transAxes, ha="right", va="bottom",
            fontsize=9, color="#B23B3B", fontweight="bold")
    # (b) SVA: deep dip, but intervention never flips (boundary), with CI band
    if sva and "checkpoints" in sva:
        cps = sva["checkpoints"]
        xs = sorted(int(k.split("_")[1]) for k in cps)
        delta = [cps[f"step_{s}"]["delta"] for s in xs]
        lo = [cps[f"step_{s}"]["delta_ci"][0] for s in xs]
        hi = [cps[f"step_{s}"]["delta_ci"][1] for s in xs]
        acc = [cps[f"step_{s}"]["acc"] * 100 for s in xs]
        a2.plot(xs, delta, "o-", color="#6A1B9A", ms=4, lw=2.3, label=r"$\Delta$ prob. diff")
        a2.fill_between(xs, lo, hi, color="#6A1B9A", alpha=0.18)
        a2.axhline(0, color="0.35", ls="--", lw=1.4)
        a2.set_xscale("log"); a2.set_xlabel("Training step")
        a2.set_ylabel(r"$\Delta$ prob. diff (bounded)", color="#6A1B9A")
        a2.set_ylim(-0.006, 0.028)
        a2.text(0.5, 0.90, "stays positive (no flip)", transform=a2.transAxes, ha="center",
                va="top", fontsize=9, color="#2E7D46", fontweight="bold")
        a2b = a2.twinx()
        a2b.plot(xs, acc, "s--", color="#90A4AE", ms=3.5, lw=1.4, label="SVA accuracy")
        a2b.axhline(50, color="#B0BEC5", ls=":", alpha=0.7, lw=1)
        a2b.set_ylabel("SVA accuracy (%)", color="#78909C"); a2b.set_ylim(0, 105)
        h1, l1 = a2.get_legend_handles_labels(); h2, l2 = a2b.get_legend_handles_labels()
        a2.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8.5)
        a2.set_title("(b) Subject--verb agreement: no flip")
    plt.tight_layout(); save("fig3_generalization")

# ---------- Fig 4: controlled dip ----------
def fig4():
    main = load("results/emnlp_controlled_dip.json")
    seeds = load("results/emnlp_controlled_dip_seeds.json")
    if not main: return
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    rate_colors = {0.0: "#C62828", 0.05: "#F9A825", 0.15: "#2E7D32", 0.30: "#1565C0"}
    # collect per-rate trajectories across seeds
    def traj_for(rate):
        series = []
        mk = f"rate_{rate}"
        if mk in main:
            series.append(main[mk]["trajectory"])
        if seeds:
            for sd in (43, 44):
                k = f"rate_{rate}_seed_{sd}"
                if k in seeds: series.append(seeds[k]["trajectory"])
        return series
    for rate, c in rate_colors.items():
        series = traj_for(rate)
        if not series: continue
        for si, s in enumerate(series):
            steps = [x["step"] for x in s]
            acc = [x["heldout_acc"] * 100 for x in s]
            if si == 0:
                ax.plot(steps, acc, "-", color=c, lw=1.8, label=f"{int(rate*100)}% injection")
            else:
                ax.plot(steps, acc, "-", color=c, lw=0.9, alpha=0.45)
    ax.axhline(50, color="grey", ls="--", alpha=0.6, lw=1)
    ax.set_xlabel("Training step"); ax.set_ylabel("IOI accuracy (%, held-out names)")
    ax.set_title("Predictive control of the dip (from-scratch LM)")
    ax.set_ylim(0, 105); ax.legend(loc="center right")
    plt.tight_layout(); save("fig4_controlled_dip")

if __name__ == "__main__":
    print("Generating EMNLP figures ->", OUT)
    fig1(); fig2(); fig3(); fig4()
    print("done. (figures missing data are skipped with a [skip] note.)")


# ---------- Conceptual overview schematic (no data; explanatory Figure 1) ----------
def fig_overview():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(11, 4.3))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # prompt + legend
    ax.text(50, 96, 'Prompt:  "When Mary and John went to the store, John gave a drink to ___"',
            ha="center", va="center", fontsize=10, family="monospace")
    ax.text(50, 89, "IO = Mary (answer)        S1 = John (first mention)        S2 = John (repeat)",
            ha="center", va="center", fontsize=8.5, color="#666")

    # training axis
    ax.annotate("", xy=(97, 81), xytext=(3, 81),
                arrowprops=dict(arrowstyle="-|>", lw=1.4, color="#333"))
    ax.text(50, 84.5, "training", ha="center", fontsize=9, color="#333", style="italic")

    # three phase boxes (well separated; generous internal spacing)
    boxes = [
        (3,  "#E8F0FE", "Early",             ["Detection forms:", "induction head", "matches the repeat"], "model near chance",   "#333"),
        (37, "#FBE3E3", "Dip (below chance)", ["Duplicate detected", "but not suppressed", "\u2192 prefers S (wrong)"], "perturb S2:  $\\Delta$LD $>$ 0  (helps)", "#B23B3B"),
        (71, "#E4F4E9", "Mature",            ["Suppression forms:", "S-inhibition head", "down-weights S"], "perturb S2:  $\\Delta$LD $<$ 0  (hurts)", "#2E7D46"),
    ]
    w = 26
    for x, color, title, body, effect, ecol in boxes:
        ax.add_patch(FancyBboxPatch((x, 40), w, 34, boxstyle="round,pad=0.4",
                     linewidth=1.3, edgecolor="#888", facecolor=color))
        ax.text(x + w/2, 69, title, ha="center", fontsize=10.5, fontweight="bold")
        for i, line in enumerate(body):
            ax.text(x + w/2, 62 - i*4.4, line, ha="center", fontsize=8.6, color="#222")
        # effect strip near box bottom
        ax.plot([x+2, x+w-2], [49.5, 49.5], color="#aaa", lw=0.6)
        ax.text(x + w/2, 45.5, effect, ha="center", fontsize=8.7, color=ecol, fontweight="bold")

    # role-reversal arrow: lives ENTIRELY in the white band below the boxes
    # (boxes end at y=40; arrow endpoints at y=36, arc bows down to ~y=22 -> no text overlap)
    ax.annotate("", xy=(82, 36), xytext=(52, 36),
                arrowprops=dict(arrowstyle="-|>", color="#C0392B", lw=1.9,
                                connectionstyle="arc3,rad=-0.42", mutation_scale=16,
                                shrinkA=0, shrinkB=0))
    ax.text(67, 12, "the causal role of the S2 representation reverses",
            ha="center", fontsize=9.5, color="#C0392B", fontweight="bold")
    plt.tight_layout(); save("fig_overview")

if "fig_overview" not in [f.__name__ for f in []]:
    fig_overview()


# ---------- Fig 5: multi-seed PolyPythias dip (appendix robustness) ----------
def fig5_polypythias():
    pp = load("results/polypythias_ioi.json")
    if not pp: return
    fig, ax = plt.subplots(figsize=(5.8, 3.7))
    order = ["seed1","seed3","seed5","data-seed1","data-seed2","data-seed3",
             "weight-seed1","weight-seed2","weight-seed3"]
    for s in order:
        if s not in pp: continue
        cps = pp[s].get("checkpoints", {})
        rows = sorted((int(k.split("_")[1]), v["accuracy"]*100)
                      for k, v in cps.items() if "accuracy" in v)
        if not rows: continue
        xs, ys = zip(*rows)
        ax.plot(xs, ys, "-", color="#1976D2", lw=1.3, alpha=0.5)
    # overlay the mean
    allx = sorted({int(k.split("_")[1]) for s in order if s in pp
                   for k in pp[s].get("checkpoints",{}) if "accuracy" in pp[s]["checkpoints"][k]})
    mean = []
    for x in allx:
        vals = [pp[s]["checkpoints"][f"step_{x}"]["accuracy"]*100
                for s in order if s in pp and f"step_{x}" in pp[s].get("checkpoints",{})
                and "accuracy" in pp[s]["checkpoints"][f"step_{x}"]]
        mean.append(np.mean(vals))
    ax.plot(allx, mean, "o-", color="#0D2C54", lw=2.6, ms=5, label="mean over 9 seeds")
    ax.axhline(50, color="0.35", ls="--", lw=1.3)
    ax.axhspan(0, 50, color="red", alpha=0.05)
    ax.text(0.5, 0.07, "below chance", transform=ax.transAxes, ha="center",
            fontsize=10, color="#B23B3B", fontweight="bold")
    ax.set_xscale("log"); ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("IOI accuracy (%)", fontsize=12); ax.set_ylim(0, 105)
    ax.tick_params(labelsize=10.5)
    ax.legend(loc="center right", fontsize=10)
    ax.set_title("All nine PolyPythias-160M seeds dip below chance", fontsize=12)
    plt.tight_layout(); save("fig5_polypythias_seeds")

fig5_polypythias()


# ---------- Fig 6: duplicate-probe by-layer at the dip ----------
def fig6_dupprobe():
    d = load("results/duplication_probes.json")
    if not d: return
    pos = d["step_2000"]["END"]
    layers = sorted(int(k.split("_")[1]) for k in pos if "test_acc" in pos[k])
    acc = [pos[f"layer_{L}"]["test_acc"] for L in layers]
    err = [pos[f"layer_{L}"].get("test_std", 0) for L in layers]
    fig, ax = plt.subplots(figsize=(5.4, 3.5))
    # shade the layer window where detection is established
    ax.axvspan(4.5, 11.5, color="#1976D2", alpha=0.06)
    ax.axhspan(0.45, 0.50, color="red", alpha=0.06)
    ax.errorbar(layers, acc, yerr=err, fmt="o-", color="#1976D2", lw=2.4, ms=6,
                capsize=3, ecolor="#1976D2")
    ax.axhline(0.5, color="#B23B3B", ls="--", lw=1.8)
    ax.text(7.0, 0.545, "chance = 0.50 (balanced set)", color="#B23B3B",
            fontsize=10, ha="center", va="bottom", fontweight="bold")
    ax.text(1.2, 0.965, "detection\nestablished", color="#0D47A1",
            fontsize=9.5, ha="center", va="top")
    ax.set_ylim(0.45, 1.0); ax.set_xticks(layers)
    ax.set_xlabel("Layer", fontsize=12.5); ax.set_ylabel("dup.-probe held-out acc.", fontsize=12.5)
    ax.tick_params(labelsize=10.5)
    ax.set_title("Duplicate detected during the dip (Pythia-160M, step 2000)", fontsize=12)
    plt.tight_layout(); save("fig6_dupprobe_by_layer")

fig6_dupprobe()
