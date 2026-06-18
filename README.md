# lendingclub-default

A leakage-free, calibrated **XGBoost credit-default classifier** for
LendingClub-schema loans. It scores a loan application's probability of default
**at origination**, benchmarked honestly **out-of-time** (temporal vintage split,
never random K-fold) against a base-rate predictor and an L2-logistic baseline,
with a **DeLong** test on the AUC gap.

> **It ranks risk; it does not predict which individuals default.** The headline
> is **ROC-AUC / PR-AUC / Brier**, never accuracy, never profit/ROI.

## Honest headline

On the **synthetic** held-out latest vintage, the shipped calibrated XGBoost
ranks default risk at **ROC-AUC 0.708**, **PR-AUC 0.274**, and **Brier 0.126**
over a **~16% base rate** (KS 0.331, log-loss 0.414). It beats the **0.500**
base-rate floor and **ties** the L2-logistic baseline (logistic ROC-AUC 0.714).
The **DeLong** test on the AUC gap returns **p = 0.84** (not significant after
Bonferroni over the 9-config grid). **No profit is claimed.**

> These are the numbers the committed `<2MB` artifact **actually achieves on the
> synthetic panel**, not a real-data result dressed up. On the real Kaggle
> accepted-loans dump the **expected** figure from the literature and plan is
> **ROC-AUC ~0.70 / PR-AUC ~0.30 to 0.40**; that number is *cited as expected*, not
> measured here.

The **shipped demo model is trained on a synthetic LC-schema panel** so the tool
is reproducible without the proprietary Kaggle dump. Real-data numbers require
the Kaggle accepted-loans CSV. The same leakage allowlist, temporal split, and
calibration path run on both:

```bash
lendingclub-default train --data accepted.csv
```

| Metric          | Synthetic (measured, this artifact) | Real Kaggle (expected, cited) |
| --------------- | ----------------------------------- | ----------------------------- |
| ROC-AUC (XGB)   | **0.708**                           | ~0.70                         |
| PR-AUC (XGB)    | **0.274**                           | ~0.30 to 0.40                 |
| Brier (XGB)     | **0.126**                           | n/a                           |
| KS (XGB)        | **0.331**                           | n/a                           |
| Base rate       | **~16%**                            | ~15%                          |
| Base-rate AUC   | **0.500** (floor)                   | 0.500                         |
| Logistic ROC-AUC| 0.714 (ties XGB, DeLong p=0.84)     | n/a                           |

## Limitations (state these up front)

1. **Accepted-loans-only selection bias.** LendingClub only records loans it
   chose to fund. The model is trained and evaluated on accepted loans, so it
   learns default risk *conditional on acceptance*, not the population risk of
   all applicants.
2. **`int_rate` / `grade` are partly circular.** Interest rate and credit grade
   are LendingClub's *own* risk model's outputs, baked into the application data.
   Their high feature importance is therefore partly tautological: the model is
   in part re-learning LendingClub's pricing.

## Install

```bash
uv venv
uv pip install -e '.[data,viz,dev]'
```

Optional extras (install only what you need):

| Extra    | Pulls in                                   | Use for                                  |
| -------- | ------------------------------------------ | ---------------------------------------- |
| `[data]` | `xgboost`, `scikit-learn`, `pandas`, `numpy`, `scipy` | Training and scoring (the lean API path) |
| `[viz]`  | `plotly`, `kaleido`                        | The ROC / PR / calibration figures       |
| `[dev]`  | `pytest`, `pytest-cov`, `hypothesis`, `ruff`, `mypy`, `shap` | Tests, lint, types, dev-only reason codes |

The hosted backend vendors only `[data]` (never `[viz]`/`[dev]`, never `shap`)
to keep the container lean and import-pure.

## Reproduce

Everything below is **seeded and deterministic**: the same seed yields byte-identical
artifacts, metrics, and figures.

```bash
# 1. Train the shipped artifact on the reproducible synthetic panel (the default).
lendingclub-default train                       # -> <2MB booster JSON + pipeline + manifest

# 2. Or train on the real Kaggle accepted-loans CSV (same pipeline, same guards).
lendingclub-default train --data accepted.csv

# 3. Score one application and evaluate a held-out panel.
lendingclub-default score    --application app.json
lendingclub-default evaluate --bundle artifacts

# 4. Run the full quality gate locally (ruff + strict mypy + tests at >=85% coverage).
uv run ruff check .
uv run mypy src
uv run pytest -q --cov=lendingclub_default --cov-report=term --cov-fail-under=85
```

`train` with no `--data` falls back to the synthetic generator; with `--data` it
reads the Kaggle CSV. The leakage allowlist, temporal vintage split, and
calibration map are identical on both paths.

## Validation

Each headline claim is pinned by a named test; tolerances are explicit so a
regression is mechanical, not a judgement call.

| Claim / metric                          | Tolerance                       | Enforcing test |
| --------------------------------------- | ------------------------------- | -------------- |
| ROC-AUC vs `sklearn.metrics`            | `1e-10`                         | `parity/test_metrics_parity.py::test_roc_auc_matches_sklearn` |
| PR-AUC vs `average_precision_score`     | `1e-10`                         | `parity/test_metrics_parity.py::test_pr_auc_matches_average_precision` |
| Brier / log-loss vs sklearn             | `1e-10`                         | `parity/test_metrics_parity.py::test_brier_and_log_loss_match_sklearn` |
| KS vs SciPy two-sample                  | `1e-10`                         | `parity/test_metrics_parity.py::test_ks_matches_scipy_two_sample` |
| Calibration vs `CalibratedClassifierCV` | Brier within `0.01`             | `parity/test_metrics_parity.py::test_calibration_matches_calibrated_classifier_cv` |
| DeLong AUC matches `roc_auc_score`      | `1e-8`                          | `parity/test_metrics_parity.py::test_delong_auc_values_match_roc_auc_score` |
| Golden leakage-free band                | ROC-AUC ∈ [0.60, 0.85], Brier ∈ [0.05, 0.25] | `regression/test_temporal_and_golden.py::test_golden_leakage_free_metrics_are_in_band` |
| No leakage column survives the pipeline | exact (set membership)          | `property/test_invariants.py::test_no_leakage_column_survives_the_pipeline` |
| No leakage on perturbed schema          | exact                           | `property/test_invariants.py::test_no_leakage_survives_on_perturbed_schema` |
| Calibrated PD in `[0,1]` and monotone   | exact                           | `property/test_invariants.py::test_calibrated_pd_is_in_unit_interval_and_monotone` |
| No look-ahead in train-fold stats       | exact (perturbation-invariant)  | `property/test_pipeline.py::test_later_vintage_rows_cannot_influence_train_fold_stats` |
| Prediction invariant to row permutation | exact                           | `property/test_pipeline.py::test_row_permutation_invariance` |
| Temporal order (no train after test)    | exact                           | `regression/test_temporal_and_golden.py::test_temporal_split_has_no_lookahead` |
| CLI `train`→`score` round-trip          | loadable booster JSON           | `integration/test_train_score_roundtrip.py::test_train_save_load_score_roundtrip` |
| Coverage gate                           | `>= 85%`                        | CI (`pytest --cov-fail-under=85`) |

## Library API

```python
import lendingclub_default as lcd

panel = lcd.generate_synthetic_panel()          # LC-schema synthetic loans
clean = lcd.drop_leakage(panel)                 # remove post-funding columns
labels = lcd.build_labels(panel)                # exclude in-progress loans
```

### Scoring entrypoint (what the hosted tool calls)

`train` emits a `<2MB` booster JSON + fitted pipeline + calibration map; the
shipped tool loads it **lazily** via `load_booster()` (a module-level
`_BOOSTER=None` sentinel, no training at import or per request) and scores one
application through `score_one`:

```python
import lendingclub_default as lcd

lcd.train(out_dir="artifacts")                  # synthetic-trained <2MB artifact
bundle = lcd.load_booster("artifacts")          # lazy, cached ScoredArtifacts

result = bundle.score_one({                      # one application-time row
    "loan_amnt": 15000, "term": "36 months", "int_rate": 22.5, "grade": "E",
    "sub_grade": "E3", "emp_length": 2, "home_ownership": "RENT",
    "annual_inc": 38000, "dti": 29.0, "fico_range_low": 660, "fico_range_high": 664,
    "revol_util": 78.0, "open_acc": 6, "pub_rec": 1, "purpose": "small_business",
    "addr_state": "NV", "installment": 560, "verification_status": "Not Verified",
})
# -> {"pd": 0.29, "decile": 3, "reason_codes": [...],
#     "predicted_label": "default", "threshold": 0.16, "model_auc": 0.71, ...}
```

`pd` is the calibrated probability of default in `[0, 1]`; `decile` is `1..10`
(1 = safest); `reason_codes` are container-safe adverse-action explanations from
the logistic coefficients (SHAP is dev-only and never enters the image).

> **The shipped demo model is synthetic-trained.** The committed artifact under
> `src/lendingclub_default/artifacts/` was trained by `train()` on the synthetic
> LC-schema generator, not on real LendingClub loans. It exists so the hosted
> tool runs reproducibly without the proprietary Kaggle dump. Treat the
> synthetic metrics above as a demonstration of the pipeline's behaviour, and the
> ~0.70 real-data figure as the literature-cited expectation, not a claim about
> any individual borrower.

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
  **DeLong** AUC-difference test (XGB vs logistic) with Bonferroni correction,
  the credit analogue of a deflated performance statistic.

## Design & decisions

- [`docs/DESIGN.md`](docs/DESIGN.md): layering, data flow, invariants, testing.
- ADRs in [`docs/decisions/`](docs/decisions/):
  [0001 leakage allowlist](docs/decisions/0001-leakage-allowlist.md) ·
  [0002 temporal vintage split](docs/decisions/0002-temporal-vintage-split.md) ·
  [0003 calibration](docs/decisions/0003-calibration.md) ·
  [0004 synthetic demo-data honesty](docs/decisions/0004-synthetic-demo-data-honesty.md) ·
  [0005 DeLong / overfitting guard](docs/decisions/0005-delong-overfitting.md).

## References

- LendingClub data dictionary (`LCDataDictionary`): the source of the
  post-funding leakage column list.
- E. R. DeLong, D. M. DeLong, D. L. Clarke-Pearson (1988), "Comparing the Areas
  under Two or More Correlated Receiver Operating Characteristic Curves: A
  Nonparametric Approach," *Biometrics* 44(3).

See [`CITATION.cff`](CITATION.cff) to cite this software.

## License

MIT, see [LICENSE](LICENSE).
