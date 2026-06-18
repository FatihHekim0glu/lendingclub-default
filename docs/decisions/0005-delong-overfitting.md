# ADR-0005: DeLong + Bonferroni as the overfitting / over-claim guard

- **Status:** Accepted
- **Date:** 2026-06-16
- **Deciders:** lendingclub-default maintainers
- **Related:** [ADR-0002](0002-temporal-vintage-split.md) (out-of-time folds), [ADR-0003](0003-calibration.md) (Brier reported with AUC)

## Context

The headline claim is comparative: **does the XGBoost beat the L2-logistic
baseline, or merely tie it?** Reporting two AUCs and eyeballing "0.708 < 0.714, so
they're close" is not a verdict, it ignores sampling noise and ignores
**multiplicity**: we tried a grid of configurations (depths × learning rates) and
could have reported the best of them.

This is the credit analogue of the Deflated Sharpe Ratio problem in the sister
`hrp-portfolio` project. There is no Sharpe here, so DSR is **not applicable**.
The right tool for *comparing two correlated ROC curves on the same test set* is
the **DeLong test**, which accounts for the covariance between the two models'
scores (they rank the *same* loans, so their errors are correlated).

Two ways to over-claim must be blocked:

1. **Noise.** Declaring XGB "better" when the AUC gap is within sampling error.
2. **Multiplicity.** Declaring significance after silently exploring a grid and
   under-counting the trials.

## Decision

`evaluation/delong.py::delong_auc_test` runs the **DeLong AUC-difference test**
(XGB vs logistic) on the shared held-out vintage, with **Bonferroni correction**
over the recorded `n_trials`.

- The recorded `n_trials` counts the **full explored configuration grid** (the
  depth × learning-rate sweep), and a guard asserts `n_trials ≥` the actual grid
  size, under-counting is impossible without failing a test.
- The Bonferroni-corrected p-value scales with the comparison count and is capped
  at `1.0`, property-tested for monotonicity
  (`property/test_evaluation_invariants.py`).
- DeLong's per-model AUCs are parity-tested against `roc_auc_score` to `1e-8`, and
  identical scores yield a zero gap with p ≈ 1.

On the committed synthetic artifact the result is **auc_diff = −0.005,
z = −1.68, p = 0.84** over a 9-config grid: the XGB does **not** significantly
beat the logistic, it **ties** it. That is the honest finding, and it is what the
headline reports.

A cost-matrix **threshold sweep** (`evaluation/threshold.py`) is reported for
context but is **never baked into the headline**, which stays AUC / PR-AUC / Brier
,  never accuracy, never profit.

## Consequences

- **Positive.** The "XGB vs logistic" claim survives both noise (DeLong) and
  multiplicity (Bonferroni over the full grid). The conservative verdict, "ties"
 , is the correct one and is reported as such.
- **Positive.** Counting the full grid in `n_trials`, asserted by a guard, makes
  data-snooping mechanically hard.
- **Cost.** A genuine small edge could be deflated below significance by
  Bonferroni. For an honesty-first benchmark a false "no edge" is far cheaper than
  a false "XGB wins."
- **Risk addressed.** "Over-claiming a model difference from noise or from a
  silently explored grid" is countered by DeLong + Bonferroni with a counted,
  asserted `n_trials`.
