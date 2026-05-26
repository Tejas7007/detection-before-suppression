# Detection Before Suppression: Sequential Circuit Formation Explains a Below-Chance Dip in Language Model Training

Code, result files, and figure-generation scripts for the paper (currently under
anonymous review). This repository reproduces every figure, table, and reported
number in the paper.

> **Anonymity note.** This repository is anonymized for double-blind review. It
> contains no author names, affiliations, or links to non-anonymous resources.
> Please do not attempt to deanonymize.

## Repository structure

```
scripts/    Experiment scripts (each self-contained, single-GPU)
src/        IOI / greater-than / SVA dataset generation (no external data needed)
results/    Raw JSON outputs from every experiment (inputs to all figures/tables)
figures/    Generated figures used in the paper (main text + appendix)
```

## Requirements

```bash
pip install torch transformers transformer-lens scikit-learn matplotlib numpy
export PYTHONPATH=src:$PYTHONPATH
```

All pretrained-model analyses run on a single A100 (40 GB); each from-scratch run
trains an 8-layer, width-512 model for 40K steps. No proprietary data is used.

## Models analyzed (all public)

| Model | Source | Use in paper |
|---|---|---|
| Pythia-160M/410M/1B (deduped) | EleutherAI | Main dip + sign flip (§3–§5) |
| PolyPythias-160M (9 seeds) | EleutherAI | Multi-seed dip, Appendix E |
| GPT-2 Small (alias-gpt2-small-x21) | Stanford CRFM | Cross-architecture flip, Appendix E |
| OLMo-1B | AllenAI | Cross-family behavioral dip, Appendix E |

Author-trained checkpoints (retrained Pythia-160M; from-scratch GPT-2-style
models) are documented under **Trained checkpoints** below.

## Script → results → paper mapping

| Paper location | Script | Results file | Key verified numbers |
|---|---|---|---|
| §3 Fig 1, Table 1: dip + sign flip (3 scales) | `emnlp_consistent_signflip.py` | `emnlp_consistent_signflip.json` | floors 31.7/29.3/36.3%; ΔLD +0.95→−4.12 (160M), +0.10→−3.63 (410M), +0.43→−3.49 (1B) |
| §3 Table 2: control battery | `emnlp_control_battery.py` | `emnlp_control_battery.json` | S2 +0.95→−4.12 (flip); S1/IO/structural no flip |
| §3 / App E: 9 PolyPythias seeds | `polypythias_sweep.py` | `polypythias_ioi.json` | all 9 dip, floors 18–37% |
| §3 / App E: cross-model flip | `emnlp_cross_model.py` | `emnlp_cross_model.json` | seed-1 30.7% +0.55→−3.92; Stanford 12.0% +1.03→−2.89 |
| §3 / App E: Stanford GPT-2 sweep | `stanford_gpt2_sweep.py` | `stanford_gpt2_ioi.json` | floor 12.0% @ step 1500 |
| §4 Fig 2: detection/suppression timeline | `emnlp_induction_timeline.py`, `emnlp_suppression_head_ablation.py` | `emnlp_induction_timeline.json`, `emnlp_suppression_head_ablation.json` | induction 0.04→0.94 by 1000; L8H9 ≈0 at dip → −2.66 mature |
| §4 Fig 6 / App F: duplicate probe | `duplication_probes.py` | `duplication_probes.json` | 93% balanced held-out (chance 50%), layers 5–11 at step 2000 |
| §5 Fig 3: greater-than flip | `emnlp_gt_causal_scale.py` | `emnlp_gt_causal_scale.json` | +0.08/+0.10/+0.32 → −0.74/−0.67/−0.73 |
| §5 Fig 3: SVA boundary | `emnlp_sva_causal_trajectory.py` | `emnlp_sva_causal_trajectory.json` | stays +0.020→+0.022 (no flip) |
| §6 Fig 4: controlled dip | `emnlp_controlled_dip.py` (+ `_seeds`) | `emnlp_controlled_dip.json`, `emnlp_controlled_dip_seeds.json` | injection rate sets dip depth/recovery |
| §6: from-scratch + S2 intervention | `emnlp_trained_model_ioi.py`, `emnlp_train_openwebtext.py` | `emnlp_trained_ioi.json` | WikiText stuck below chance, ΔLD>0 throughout |
| §6: toy copy-vs-IOI sweep | `emnlp_mini_ioi.py`, `emnlp_mini_ioi_scaled.py` | `emnlp_mini_ioi*.json` | dip+flip in 9/10 configs, all 3 seeds |
| App A: clustered CIs + layer windows | `emnlp_robustness_cis_layers.py` | `emnlp_robustness_cis_layers.json` | flip survives template/name-pair clustering |
| App A: selection-bias trajectory | `emnlp_selection_bias.py` | `emnlp_selection_bias.json` | held-out +1.13 (dip) → −4.09 (mature) |
| App C: path patching (composition, public model) | `head_ablation.py` (exp_a) | `head_ablation.json` | L8H9 → downstream L9 heads (ΔIO −0.17, ΔS2 +0.17) |
| App D: subspace / cross-task / SAE controls | `projection_control.py`, `emnlp_final_lap.py`, `emnlp_tier_s.py` | `projection_controls.json`, `emnlp_final_lap.json`, `emnlp_tier_s.json` | ~107% low-rank recovery; cross-task projection fails; SAE null |
| App D: positional / grokking controls | `negative_control.py`, `emnlp_final_robustness.py` | `negative_controls.json`, `emnlp_final_robustness.json` | wrong-position ΔLD ≈ 0; grokking ablation never positive |
| App F: retrained-model heads + position scan | `emnlp_head_patching.py`, `emnlp_robustness_battery.py`, `retrain_pythia_160m.py` | `emnlp_head_patching.json`, `emnlp_robustness_battery.json`, `retrain_ioi_analysis.json` | dominant heads differ original vs retrained; S1 vs S2 position scan |
| App E: OLMo / cross-family | `emnlp_final_robustness.py`, `emnlp_additional_controls.py` | `emnlp_final_robustness.json`, `emnlp_additional_controls.json` | 44% @ 4B tokens, recovers 61–66% |
| All figures | `emnlp_generate_figures.py` | `figures/emnlp/` | reads results JSONs, writes PDFs/PNGs |

## Reproducing the paper

```bash
export PYTHONPATH=src:$PYTHONPATH

# 1. Core dip + sign flip across scales (longest run)
python scripts/emnlp_consistent_signflip.py

# 2. Mechanism: detection (induction) and suppression (head ablation) timelines
python scripts/emnlp_induction_timeline.py
python scripts/emnlp_suppression_head_ablation.py
python scripts/duplication_probes.py

# 3. Generalization and boundary
python scripts/emnlp_gt_causal_scale.py
python scripts/emnlp_sva_causal_trajectory.py

# 4. Regenerate all figures from existing results (no GPU needed)
python scripts/emnlp_generate_figures.py
```

Every number in the paper has been cross-checked against the corresponding JSON
in `results/`; the mapping above is exhaustive.

## Trained checkpoints

The following author-trained checkpoints are released on an anonymized host to
support full reproduction (the analyzed Pythia/PolyPythias/Stanford/OLMo models
are already public and are not re-hosted here):

- **Retrained Pythia-160M (seed 42)** — 103 dense checkpoints spanning steps
  0–10000 (every 50 steps to 3000, every 200 thereafter), used for the
  component-fluctuation analysis (Appendix F):
  https://huggingface.co/anonymous-research-sub/pythia-160m-retrained-seed42
- **From-scratch GPT-2-style models** (WikiText-103 / OpenWebText, §6) are not
  re-hosted: they are regenerable from `emnlp_trained_model_ioi.py`,
  `emnlp_train_openwebtext.py`, and `emnlp_controlled_dip.py`, and their outputs
  are included in `results/`.

All analyzed base models (Pythia, PolyPythias, Stanford GPT-2, OLMo) are already
public and are linked above; they are not re-hosted.

## License

Released for review under an anonymized identity; license to be finalized on
de-anonymization.
