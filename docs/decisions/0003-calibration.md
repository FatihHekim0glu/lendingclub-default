# ADR-0003: Calibrate the score into a usable PD in `[0, 1]`

- **Status:** Accepted
- **Date:** 2026-06-16
- **Deciders:** lendingclub-default maintainers
- **Related:** [ADR-0002](0002-temporal-vintage-split.md) (the fold the calibration map is fit on), [ADR-0005](0005-delong-overfitting.md) (Brier reported alongside AUC)

## Context

A raw XGBoost margin (or even `predict_proba`) is a good **ranking** score but a
poor **probability**: boosted trees are systematically over-confident, so the raw
output is not a number you can call "this loan has a 29% chance of default."
Because the tool's whole point is to report a **calibrated PD**, and a
**risk decile** derived from it, the ranking score alone is not enough.

Two standard remedies exist:

- **Platt scaling**, a logistic fit on the score. Cheap, monotone, but assumes a
  sigmoidal miscalibration shape.
- **Isotonic regression**, a non-parametric monotone fit. More flexible, fits
  arbitrary monotone miscalibration, needs a bit more data and can overfit small
  folds.

Calibration must also obey the same temporal discipline as everything else: a map
fit on the test fold would leak.

## Decision

`models/calibrate.py` wraps the fitted classifier with a **calibration map
(isotonic by default, Platt available)** fit on a **later-vintage calibration
slice of the train fold**, never the held-out test vintage (ADR-0002). The
output PD is what the tool reports and what the risk decile is computed from.

Two properties are guaranteed and tested:

- the map is **monotone** (it never reorders risk, a higher raw score never maps
  to a lower PD); and
- the output lies in **`[0, 1]`**.

Both are property-tested over random knot configurations
(`property/test_invariants.py::test_calibrated_pd_is_in_unit_interval_and_monotone`),
and a parity test checks the calibrated Brier against
`CalibratedClassifierCV` to within `0.01`.

The headline keeps **Brier and log-loss** next to AUC precisely so calibration
quality is visible: AUC measures ranking, Brier measures whether the PD is
honest. The committed synthetic artifact reports Brier 0.126, a sane probability
score over a ~16% base rate, not a degenerate one.

## Consequences

- **Positive.** The reported `calibrated_pd` is a number a user can reason about,
  and the risk decile is a stable, monotone transform of it.
- **Positive.** Reporting Brier/log-loss alongside AUC keeps "well-ranked but
  badly calibrated" from passing as success.
- **Cost.** A slice of the train fold is spent on calibration rather than fitting;
  isotonic can be jumpy on small folds (mitigated by the monotone constraint and
  the later-vintage slicing).
- **Risk addressed.** "Over-confident tree scores masquerading as probabilities"
  is countered by a monotone, in-`[0,1]`, temporally-honest calibration map with a
  Brier parity oracle.
