# ADR-0002: Temporal vintage split by `issue_d`, never random K-fold

- **Status:** Accepted
- **Date:** 2026-06-16
- **Deciders:** lendingclub-default maintainers
- **Related:** [ADR-0001](0001-leakage-allowlist.md) (clean panel), [ADR-0003](0003-calibration.md) (calibration fit on the same fold discipline)

## Context

Credit data is a **time series of vintages**: loans issued in 2015 mature and
resolve before loans issued in 2018 do, and macro conditions drift across
cohorts. A random K-fold split scatters rows from every vintage into both train
and test, which leaks the future into the past in two ways:

1. **Maturity leakage.** The test set contains loans whose contemporaries the
   model already saw resolve, so the evaluation is optimistic about how the model
   generalizes to a *genuinely unseen later* cohort.
2. **Encoder leakage.** Any statistic fit across the full data (a target-encoding
   mean, a scaler's mean/variance, an imputation value) is computed partly from
   future rows, inflating in-sample fit.

A model whose only validation is random K-fold will report a number it cannot
reproduce in deployment, where it must score next quarter's applicants using only
what was known by this quarter.

## Decision

`data/split.py::temporal_split` partitions strictly by `issue_d`: **train on
vintages at or before a cutoff, test on later vintages.** No random K-fold
anywhere in the headline path.

Two reinforcing rules:

- **Held-out evaluation is the LATEST vintage**, the rows furthest in the future
 , so the reported metric is the one a deployment would actually see.
- Early-stopping and calibration slices are themselves carved as *later-vintage
  sub-folds of the train fold* (`_temporal_subsplit`), so even tuning never peeks
  past its own horizon.

`assert_temporal_order` raises `TemporalSplitError` if any train row's `issue_d`
falls after any test row's. This is asserted by a regression test, and a property
test confirms that perturbing later-vintage rows leaves train-fold transform
statistics unchanged.

## Consequences

- **Positive.** The reported AUC/PR-AUC/Brier is an honest out-of-time estimate,
  not an in-sample mirage.
- **Positive.** Combined with fit-on-train-only and out-of-fold target encoding
  (ADR-0001 clean panel, `features/pipeline.py`), there is no path for the future
  to influence the fit.
- **Cost.** Less data for training (the latest vintages are reserved) and higher
  variance in the estimate than an (invalid) K-fold would show. We accept this , 
  a wider honest interval beats a tight dishonest point.
- **Risk addressed.** "Look-ahead / maturity leakage from random K-fold" is
  eliminated by an asserted, property-tested temporal ordering.
