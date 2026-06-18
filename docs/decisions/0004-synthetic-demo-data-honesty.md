# ADR-0004: The shipped demo artifact is synthetic-trained, and says so

- **Status:** Accepted
- **Date:** 2026-06-16
- **Deciders:** lendingclub-default maintainers
- **Related:** [ADR-0001](0001-leakage-allowlist.md) (the allowlist the generator exercises), [ADR-0005](0005-delong-overfitting.md) (how the honest numbers are reported)

## Context

There is **no real Kaggle LendingClub dataset on this machine and no Kaggle
credentials.** The proprietary accepted-loans dump cannot be committed to a public
repo regardless. Yet the hosted tool must run reproducibly out of the box, and a
benchmark with no shippable artifact is not a benchmark.

The tempting dishonest move is to *report a real-data AUC as if measured*, to
quote ~0.70 in the README and ship an artifact that nobody can reproduce, blurring
the line between "what this model achieved" and "what the literature expects." The
whole project's credibility rests on not doing that.

A second risk is that a synthetic generator can be too easy (AUC ~0.99, leakage by
construction) or too hard (AUC ~0.50, no signal), either of which makes the demo a
lie about what a real model would do.

## Decision

Ship a **synthetic-trained** artifact and label it as such everywhere.

- `data/synthetic.py` emits a realistic LC-schema panel: the real application-time
  columns, a `loan_status`, **and** the post-funding leakage columns (so the
  allowlist of ADR-0001 is genuinely exercised). Default probability is a noisy
  **monotone** function of fico/dti/int_rate/grade, base rate ~15%, `issue_d`
  spread across 2015 to 2018 vintages with mild regime drift. It is **seeded and
  deterministic** (reuses `_rng.py`).
- A **leakage-free** model on this panel lands at a *believable* AUC (~0.65 to 0.72),
  not 0.5 and not 0.99, verified by the golden-band regression test.
- The committed `<2MB` booster is trained by `train()` on this panel, stamped with
  a `RunManifest` (seed + config hash), under `src/lendingclub_default/artifacts/`.
- The README, DESIGN, and CLI all state plainly: **the demo model is
  synthetic-trained.** The synthetic metrics are reported as *measured on
  synthetic data*; the ~0.70 real-data figure is reported as the *expected*
  literature/plan number, **not** measured here.
- The **same** code path, leakage drop, temporal split, calibration, runs on
  real data via `train --data accepted.csv`. The synthetic generator is a fallback
  for when `--data` is omitted, not a separate, weaker pipeline.

## Consequences

- **Positive.** The tool is reproducible by anyone, with no proprietary data and
  no credentials, and the artifact stays under 2MB.
- **Positive.** The reader can never confuse the synthetic demonstration with a
  real-data claim, because every surface distinguishes the two.
- **Positive.** Because the generator embeds leakage columns and a monotone signal,
  the demo genuinely exercises the leakage guard and produces a realistic AUC.
- **Cost.** The shipped numbers are not real LendingClub performance; a user who
  wants the real figure must bring the Kaggle CSV. We consider that the honest
  trade rather than a defect.
- **Risk addressed.** "Reporting an unreproducible / fabricated real-data AUC" is
  eliminated by shipping a reproducible synthetic artifact and labelling the real
  figure as expected.
