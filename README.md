# lendingclub-default

A leakage-free, calibrated **XGBoost credit-default classifier** for
LendingClub-schema loans. It scores a loan application's probability of default
**at origination**, benchmarked honestly **out-of-time** (temporal vintage split,
never random K-fold) against a base-rate predictor and an L2-logistic baseline,
with a **DeLong** test on the AUC gap.

> **It ranks risk; it does not predict which individuals default.** The headline
> is **ROC-AUC / PR-AUC / Brier** — never accuracy, never profit/ROI.

## Honest headline

A leakage-free, calibrated model ranks default risk at **ROC-AUC ~0.70** /
**PR-AUC ~0.30–0.40** over a **~15% base rate** — it beats the base-rate
predictor and ties an L2-logistic baseline. No profit is claimed.

The **shipped demo model is trained on a synthetic LC-schema panel** so the tool
is reproducible without the proprietary Kaggle dump. The committed `<2MB` booster
artifact reports the **synthetic-data** result it actually achieves; the ~0.70
figure above is the **expected real-data** number from the literature and plan,
not a measured one. Real-data numbers require the Kaggle accepted-loans CSV:

```bash
lendingclub-default train --data accepted.csv
```

## Limitations (state these up front)

1. **Accepted-loans-only selection bias.** LendingClub only records loans it
   chose to fund. The model is trained and evaluated on accepted loans, so it
   learns default risk *conditional on acceptance*, not the population risk of
   all applicants.
2. **`int_rate` / `grade` are partly circular.** Interest rate and credit grade
   are LendingClub's *own* risk model's outputs, baked into the application data.
   Their high feature importance is therefore partly tautological — the model is
   in part re-learning LendingClub's pricing.

## Install

```bash
uv venv
uv pip install -e '.[data,viz,dev]'
```

## Quickstart

```bash
# Train on the reproducible synthetic panel (default) or a real Kaggle CSV.
lendingclub-default train                       # synthetic
lendingclub-default train --data accepted.csv   # real

# Score a single application and evaluate a held-out panel.
lendingclub-default score   ...
lendingclub-default evaluate ...
```

```python
import lendingclub_default as lcd

panel = lcd.generate_synthetic_panel()          # LC-schema synthetic loans
clean = lcd.drop_leakage(panel)                 # remove post-funding columns
labels = lcd.build_labels(panel)                # exclude in-progress loans
```

## How it works (the correctness guards)

- **No leakage.** `drop_leakage` removes every post-funding column in the frozen
  `LEAKAGE_COLS` allowlist (`recoveries`, `total_pymnt*`, `out_prncp*`,
  `last_pymnt_*`, `collection_recovery_fee`, `settlement_*`, `hardship_*`,
  `debt_settlement_flag`, `total_rec_*`, `funded_amnt*`, ...). A property test
  asserts none survive the pipeline.
- **No look-ahead.** The split is **temporal by `issue_d`**: train on vintages at
  or before a cutoff, test on later ones. Every imputer/encoder/scaler is fit on
  the train fold only; high-cardinality categoricals use out-of-fold target
  encoding.
- **Calibrated PD.** An isotonic/Platt map turns the booster's ranking score into
  a usable probability in `[0, 1]` (property-tested: in `[0, 1]` and monotone).
- **Overfitting guard.** Repeated nested CV with a recorded `n_trials`, plus a
  **DeLong** AUC-difference test (XGB vs logistic) with Bonferroni correction —
  the credit analogue of a deflated performance statistic.

## References

- LendingClub data dictionary (`LCDataDictionary`) — the source of the
  post-funding leakage column list.
- E. R. DeLong, D. M. DeLong, D. L. Clarke-Pearson (1988), "Comparing the Areas
  under Two or More Correlated Receiver Operating Characteristic Curves."

## License

MIT — see [LICENSE](LICENSE).
