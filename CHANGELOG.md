# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-16

First release: a leakage-free, calibrated XGBoost credit-default classifier for
LendingClub-schema loans, benchmarked honestly out-of-time and shipped with a
reproducible synthetic-trained `<2MB` artifact.

### Added

- **Foundation.** `_constants` (status vocabularies, grades, terms,
  application columns), `_typing`, `_exceptions` (`LendingClubDefaultError` base +
  `LeakageError`, `TemporalSplitError`, `ArtifactError`, `ValidationError`,
  `InsufficientDataError`), `_validation`, `_manifest` (`RunManifest` with BLAKE2b
  config-hash), and `_rng` (seeded PCG64).
- **Data.** `data/synthetic.py` (realistic LC-schema generator with embedded
  post-funding leakage columns and a monotone default signal), `data/load.py`
  (real Kaggle CSV reader with synthetic fallback), `data/leakage.py` (the frozen
  `LEAKAGE_COLS` allowlist + `drop_leakage` / `assert_no_leakage`),
  `data/labels.py` (resolved-status labels, in-progress loans excluded),
  `data/split.py` (temporal vintage split by `issue_d`).
- **Features.** `features/pipeline.py` — fit-on-train-only `ColumnTransformer`
  with out-of-fold target encoding for high-cardinality categoricals.
- **Models.** `models/baselines.py` (base-rate + L2-logistic), `models/xgb.py`
  (XGBoost with `scale_pos_weight` and temporal early stopping),
  `models/calibrate.py` (isotonic/Platt PD calibration),
  `models/reason_codes.py` (logistic-coefficient reason codes; SHAP dev-only).
- **Evaluation.** `evaluation/metrics.py` (ROC-AUC, PR-AUC, Brier, log-loss, KS),
  `evaluation/calibration.py` (reliability curve), `evaluation/delong.py` (DeLong
  AUC-difference test with Bonferroni), `evaluation/threshold.py` (cost-matrix
  sweep, reported never baked into the headline).
- **Orchestration.** End-to-end `train` (load → drop leakage → labels → temporal
  split → fit → calibrate → evaluate latest vintage → emit `<2MB` booster +
  pipeline + manifest), Plotly figure builders (`plots`), and a Typer CLI
  (`train` / `score` / `evaluate`).
- **Shipped artifact.** A synthetic-trained `<2MB` booster JSON + fitted pipeline
  + calibration map + `RunManifest`, loaded lazily via a module-level
  `_BOOSTER=None` sentinel. Synthetic held-out metrics: ROC-AUC 0.708,
  PR-AUC 0.274, Brier 0.126, KS 0.331 over a ~16% base rate; DeLong p=0.84 vs the
  logistic baseline (a tie). Real-data ROC-AUC ~0.70 is the cited expectation.
- **Tests.** Partitioned suite (unit / parity / property / regression /
  integration) with seeded conftest fixtures (`synthetic_panel`,
  `k_vintage_fixture`, `schema_with_leakage`); metric parity vs sklearn/SciPy to
  `1e-10`, leakage/no-look-ahead/PD-in-`[0,1]` property tests, golden-band and
  temporal-order regression tests, and a CLI round-trip. Coverage gate ≥ 85%.
- **Docs & governance.** README (honest headline, two limitations, reproduce
  block, validation table), `docs/DESIGN.md` + ADRs 0001–0005, `CITATION.cff`,
  MIT `LICENSE`, and a `no-ai-attribution` CI guard.

[Unreleased]: https://github.com/FatihHekim0glu/lendingclub-default/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/FatihHekim0glu/lendingclub-default/releases/tag/v0.1.0
