# CISD: Causal-Invariant Stable/Spurious Decomposition for Multi-Site Clinical Risk Prediction

**One-line research statement:** We test whether explicitly decomposing a
clinical risk model's learned representation into an adversarially-purified
"stable" block and an environment-absorbing "spurious" block improves
generalization, calibration, and modality-missingness robustness under real
multi-hospital distribution shift, compared to ERM and Invariant Risk
Minimization (IRM).

## Motivation

Medical ML models are typically validated on data from the same
distribution they were trained on. In practice, deployment means a new
hospital, a new patient population, a different acquisition protocol — and
performance can collapse even when the underlying biology hasn't changed.
The failure mode is that the model learned **environment-specific
correlations** (e.g. site-specific baseline disease prevalence, protocol-
dependent measurement ranges) rather than **causally stable, biologically
grounded relationships**.

## Research Gap

Invariant Risk Minimization (Arjovsky et al., 2019) and its many variants
penalize a model for having an environment-dependent optimal classifier on
top of its representation, but they do not give the representation an
explicit place to *put* the environment-specific signal it inevitably
picks up. As a result the invariance penalty and the predictive objective
fight over the same features. To the best of our literature search
(see `Writeup.md` §2 for the sources checked), decomposition-based
approaches that combine (a) an explicit stable/spurious representation
split, (b) an adversarial (gradient-reversal) environment classifier on the
stable block, and (c) a non-adversarial auxiliary environment classifier on
the spurious block — evaluated jointly on generalization, calibration, and
missing-modality robustness on a real multi-hospital clinical dataset — are
not the standard IRM recipe. This is the gap we target. We do **not** claim
that stable/spurious decomposition itself is unprecedented (disentangled
and domain-adversarial representation learning are established ideas); the
contribution is the specific combination and the clinical evaluation
protocol.

## Proposed Method: CISD

A shared encoder maps clinical features `X` to a hidden representation `Z`,
which is split into two disjoint blocks:

```
Z = [Z_stable | Z_spurious]
```

- **Outcome head** predicts disease presence from `Z_stable` **only**.
- **Adversarial environment head** (gradient-reversal) tries to predict the
  site from `Z_stable`; the encoder is trained to *defeat* it, actively
  stripping site-identifying information out of the block that drives
  prediction.
- **Auxiliary environment head** (normal gradient) predicts the site from
  `Z_spurious`; this gives site-specific signal an explicit place to go
  instead of leaking into `Z_stable`.
- **Orthogonality penalty** decorrelates `Z_stable` and `Z_spurious` across
  the batch, discouraging the two blocks from encoding redundant
  information.

Full mathematical formulation, gradients, and the IRM baseline formulation
are in `Writeup.md`.

## Key Contribution

1. A specific, ablatable three-mechanism decomposition objective
   (adversarial purification + auxiliary absorption + orthogonality) for
   separating stable from environment-dependent signal in a clinical
   prediction model, with every gradient hand-derived and verified against
   numerical differentiation (`tests/test_gradients.py`) before any
   experiment was run.
2. An evaluation protocol that goes beyond in-distribution accuracy: Leave-
   One-Site-Out (LOSO) generalization, calibration under shift (ECE,
   Brier), and a missing-modality stress test — because clinical deployment
   risk shows up in exactly these places, not in a random train/test split.
3. An honest ablation and per-site breakdown showing where the method
   helps, where it doesn't, and why (see Results).

## Methodology / Pipeline

```
Real 4-site data → common-schema preprocessing (median impute + standardize,
fit on training sites only) → Leave-One-Site-Out split → train
{ERM, IRM, CISD} on 3 pooled sites (site id = environment) → evaluate on the
held-out 4th site → repeat over 5 seeds → aggregate → ablate CISD's three
mechanisms → stress-test all methods with each modality group zeroed at
test time.
```

## Dataset

Real, public **UCI Heart Disease** multi-site dataset — see `data/README.md`
for exact provenance, field selection, and prevalence-by-site (verified:
Cleveland 46%, Hungary 36%, VA 75%, Switzerland 93% disease-positive —
genuine label shift across sites, not constructed).

- n = 920 patients across 4 sites, 11 common clinical features.
- Modalities (feature groups) for the missing-modality test: demographic
  (age, sex), clinical/resting (cp, trestbps, chol, fbs, restecg),
  exercise-test (thalach, exang, oldpeak, slope).
- Train/test: Leave-One-Site-Out — train on 3 sites pooled, test on the 4th,
  repeated for all 4 choices of held-out site. No test-site data is ever
  seen during preprocessing fit or training.

## Baselines

- **ERM**: standard 1-hidden-layer MLP, no environment awareness.
- **IRM** (Arjovsky et al., 2019, IRMv1 penalty): same architecture, with
  the environment-invariance penalty added analytically (closed-form
  gradient for a linear head, verified against a numerical gradient check).

## Evaluation Metrics

AUROC, AUPRC, F1, Brier score, Expected Calibration Error (ECE, 10 bins).
All implemented from scratch in `src/metrics.py` (no external metrics
library was needed or used).

## Experiments (real, all actually run — see `output/`)

1. **LOSO generalization**: `output/loso_results_raw.csv`,
   `output/loso_summary_by_site.csv`, `output/loso_summary_overall.csv`.
2. **Ablation**: `output/ablation_results_raw.csv`,
   `output/ablation_summary.csv`.
3. **Missing-modality stress test**: `output/missing_modality_results_raw.csv`,
   `output/missing_modality_summary.csv`.
4. **Significance test**: `output/statistical_tests.json` (paired t-test,
   CISD vs. ERM and CISD vs. IRM, across all 4 sites × 5 seeds).

## Results (actual numbers, from `output/metrics.json`)

Overall LOSO (mean over 4 held-out sites × 5 seeds):

| Method | AUROC | ECE | Brier |
|---|---|---|---|
| ERM | 0.669 ± 0.087 | 0.316 ± 0.055 | see `output/loso_summary_overall.csv` |
| IRM | 0.677 ± 0.071 | 0.314 ± 0.040 | " |
| **CISD (ours)** | **0.704 ± 0.097** | **0.251 ± 0.097** | " |

Paired t-test (n=20: 4 sites × 5 seeds) on AUROC: CISD vs ERM mean
Δ = +0.035, t = 3.15 (p < 0.01, significant); CISD vs IRM mean Δ = +0.027,
t = 1.58 (p ≈ 0.13, **not** significant).

**Per-site breakdown reveals CISD does not win uniformly** — on Switzerland
(93% positive, the most extreme label-shift site) IRM outperforms CISD
(AUROC 0.655 vs 0.589), while CISD wins clearly on Cleveland, Hungary, and
VA. See Discussion / Limitations in `Writeup.md` §16 for why.

Missing-modality test: CISD degrades less than ERM when the exercise-test
modality is zeroed at test time (AUROC 0.636 vs 0.589) and the clinical
modality (0.691 vs 0.688, roughly tied); demographic removal is similar for
both. The largest, most consistent gap is on the exercise-test modality.

Ablation (AUROC, averaged): full CISD 0.704; removing the adversarial term
drops it to 0.675 (the largest single-component effect); removing the
orthogonality penalty *increases* it slightly to 0.717 (within noise, and
smaller than the adversarial term's effect — see Discussion); removing the
auxiliary spurious-environment head leaves it essentially unchanged (0.704).
This says the adversarial purification of `Z_stable` is doing the real
work; the orthogonality penalty and auxiliary head are not clearly pulling
their weight in this setup.

## Outputs

```
output/
├── site_summary.csv                     # real label prevalence per site
├── predictions.csv                      # raw per-patient predictions, all methods
├── loso_results_raw.csv                 # every (method, site, seed) metric row
├── loso_summary_by_site.csv / _overall.csv
├── ablation_results_raw.csv / ablation_summary.csv
├── missing_modality_results_raw.csv / _summary.csv
├── statistical_tests.json               # paired t-tests
├── metrics.json                         # everything, machine-readable
└── figures/
    ├── loso_auroc_by_site.png
    ├── ece_comparison.png
    ├── missing_modality_robustness.png
    └── ablation_study.png
```

## Reproducibility

```bash
pip install -r requirements.txt
python tests/test_gradients.py   # must print ALL GRADIENT CHECKS PASSED
python main.py                   # re-downloads nothing; reads data_raw/, writes output/
```
CPU-only, no GPU required, runs in well under a minute (pure NumPy, ~1000
patients, 2500 full-batch iterations × 4 folds × 5 seeds × 7 model variants).
Random seeds are fixed and enumerated in `main.py` (`SEEDS = [0,1,2,3,4]`).

## Limitations

- **Small n per site** (123–303), especially Switzerland — per-site standard
  deviations are large and single-site conclusions should not be
  over-interpreted.
- **Tabular, not multimodal-in-the-imaging-sense**: "modalities" here are
  clinically distinct *feature groups* from the same tabular record, not
  separate imaging/genomic/text modalities. The missing-modality test is a
  meaningful but modest proxy for true multimodal missingness.
- **CISD is not uniformly better than IRM** — see the Switzerland result.
  The overall improvement over IRM is not statistically significant at
  n=20; only the improvement over plain ERM is.
- **This is a research prototype, not a validated clinical tool.** Nothing
  here should be read as evidence that this method (or any method in this
  repo) is ready for clinical deployment, and no clinical-utility or
  regulatory claim is made.
- Architecture and hyperparameters (`H=16`, `H1=10`, penalty weights) were
  chosen once based on reasoning about scale, not tuned via a validation
  sweep due to the small-data setting; a proper nested-CV hyperparameter
  search is future work.

## Future Work

- Extend to a genuinely multimodal clinical dataset (e.g. imaging + tabular)
  where "missing modality" means a whole missing data stream.
- Investigate *why* CISD loses to IRM specifically on the most extreme
  label-shift site — plausibly the adversarial term over-strips signal that
  is legitimately prevalence-related rather than purely spurious when one
  environment is almost entirely single-class.
- Selective prediction / abstention: use disagreement between `Z_stable`-
  only and full-`Z` predictions as an uncertainty signal to abstain on
  likely-unreliable cases — a natural next step given NERVALYS-AI's existing
  focus on conformal prediction and calibration.

## Citation

- Detrano, R., et al. (1989). International application of a new
  probability algorithm for the diagnosis of coronary artery disease.
  *American Journal of Cardiology*.
- Dua, D. and Graff, C. (2019). UCI Machine Learning Repository, Heart
  Disease dataset. University of California, Irvine.
- Arjovsky, M., Bottou, L., Gulrajani, I., Lopez-Paz, D. (2019). Invariant
  Risk Minimization. arXiv:1907.02893.
