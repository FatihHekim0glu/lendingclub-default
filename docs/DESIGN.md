# Design

This document explains how `lendingclub-default` is put together: the layering,
the data flow through one training run, the invariants the compute core
guarantees, and the testing strategy that keeps the honest headline honest. For
*why* individual contested choices were made, see the numbered ADRs in
[`docs/decisions/`](decisions/).

## Goals and non-goals

**Goals**

- A pure, typed (`mypy --strict`, `py.typed`), side-effect-free compute core that
  can be audited line by line and vendored into a backend without dragging UI or
  network dependencies along.
- A **leakage-free** credit-default classifier: every post-funding column is
  removed before any feature is built, and the removal is property-tested.
- An **honest out-of-time** verdict: train on early vintages, test on later ones
  (never random K-fold), with a calibrated PD in `[0, 1]` and a **DeLong** test
  on the XGB-vs-logistic AUC gap that is mechanically prevented from over-claiming.
- A reproducible `<2MB` artifact, trained on a synthetic LC-schema panel, that the
  hosted tool loads lazily — documented plainly as synthetic, with the real-data
  figure cited as *expected*, not measured.

**Non-goals**

- Predicting which individuals default. The model **ranks risk**; it reports
  AUC / PR-AUC / Brier, never accuracy and never profit/ROI.
- Beating LendingClub's own pricing. `int_rate` and `grade` are LC's own risk
  model's outputs, so part of any signal is circular (ADR-0001).
- A live underwriting system. This is a research/benchmark library.
- A generic ML toolkit. Everything exists to serve one leakage-free classifier.

## Layered architecture

The package is strictly layered; each layer imports only from the ones below it.
`src/lendingclub_default/` has **zero import-time side effects**, guarded by a
subprocess import-purity test. The shipped booster loads **lazily** via a
module-level `_BOOSTER=None` sentinel — no data load, training, or network at
import or per request.

```
            cli.py (Typer)            plots.py (Plotly)         api/ (FastAPI vendor)
                 |                          |                          |
   ┌─────────────┴──────────────────────────┴──────────────────────────┘
   │                              train.py
   │   load -> drop leakage -> labels -> temporal split -> fit -> calibrate
   │   -> evaluate latest vintage -> emit <2MB booster + pipeline + manifest
   ├────────────────────────────────────────────────────────────────────────
   │                            evaluation/
   │   metrics.py · calibration.py · delong.py · threshold.py
   │   (ROC/PR/Brier/log-loss/KS · reliability curve · DeLong+Bonferroni · cost sweep)
   ├────────────────────────────────────────────────────────────────────────
   │            models/                          features/
   │   baselines.py · xgb.py                pipeline.py
   │   calibrate.py · reason_codes.py       (ColumnTransformer, fit-on-train-only,
   │   (base-rate · L2-logistic · XGBoost    out-of-fold target encoding)
   │    · isotonic/Platt · WoE codes)
   ├────────────────────────────────────────────────────────────────────────
   │                              data/
   │   synthetic.py · load.py · leakage.py · labels.py · split.py
   │   (LC-schema generator · real-CSV reader · LEAKAGE_COLS allowlist ·
   │    resolved-status labels · temporal vintage split)
   ├────────────────────────────────────────────────────────────────────────
   │   foundation (no internal deps)
   │   _validation · _constants · _typing · _exceptions · _manifest · _rng
   └────────────────────────────────────────────────────────────────────────
```

### Foundation (`_*.py`)

- `_constants.py` — status vocabularies (`PAID_STATUSES`, `DEFAULT_STATUSES`,
  `IN_PROGRESS_STATUSES`), `VALID_GRADES`, `VALID_TERMS`, `N_RISK_DECILES`,
  `APPLICATION_COLUMNS`; one source of truth.
- `_validation.py` — input guards (`ensure_dataframe`, `ensure_series`,
  `validate_min_obs`, `coerce_dtypes`).
- `_typing.py` / `_exceptions.py` — shared aliases and the exception taxonomy
  (`LendingClubDefaultError` base, `LeakageError`, `TemporalSplitError`,
  `ArtifactError`, `ValidationError`, `InsufficientDataError`).
- `_manifest.py` / `_rng.py` — `RunManifest` (BLAKE2b config hash) plus seeded
  PCG64 substreams. The manifest makes a whole run reproducible; the same seed
  yields byte-identical artifacts, metrics, and figures.

### `data/`

`synthetic.py` emits a pandas panel with the real LC application-time schema, a
`loan_status` column, **and** the post-funding leakage columns — so the leakage
allowlist is genuinely exercised (ADR-0004). Default probability is a noisy
monotone function of fico/dti/int_rate/grade, base rate ~15%, `issue_d` spread
across 2015–2018 vintage cohorts with mild regime drift. `load.py` reads a real
Kaggle CSV when given, else falls back to the generator. `leakage.py` holds the
**frozen `LEAKAGE_COLS` allowlist** and `drop_leakage`/`assert_no_leakage`
(ADR-0001). `labels.py` maps resolved statuses to the binary target and
**excludes** in-progress loans (ADR-0001). `split.py` does the **temporal vintage
split** by `issue_d` (ADR-0002).

### `features/`

`pipeline.py` builds a sklearn `ColumnTransformer`/`Pipeline` — imputers,
encoders, scaler — that is **fit on the train fold only**. High-cardinality
categoricals (`purpose`, `addr_state`) use **out-of-fold target encoding** so no
row encodes itself. Only application-time features (post the leakage drop) ever
enter (ADR-0002).

### `models/`

`baselines.py` (a stratified base-rate predictor + L2-logistic), `xgb.py` (XGBoost
binary classifier with `scale_pos_weight` for imbalance and early stopping on a
temporal validation fold), `calibrate.py` (isotonic/Platt wrapping so the output
is a usable PD in `[0, 1]` — ADR-0003), `reason_codes.py` (weight-of-evidence /
logistic coefficients; SHAP is **dev-only** and never enters the container).

### `evaluation/`

`metrics.py` (ROC-AUC, PR-AUC, Brier, log-loss, KS), `calibration.py` (reliability
curve points), `delong.py` (the **DeLong** AUC-difference test, XGB vs logistic,
with Bonferroni — ADR-0005), `threshold.py` (a cost-matrix threshold sweep,
reported but **never baked into the headline**). The recorded `n_trials` is
carried in the `RunManifest`.

## Data flow through one training run

```
panel (synthetic or Kaggle CSV)
        │
        ▼  drop_leakage  ── removes every LEAKAGE_COLS column
   application-time columns only
        │
        ▼  build_labels  ── resolved statuses only; in-progress excluded
   (X, y)
        │
        ▼  temporal_split by issue_d  ── train ≤ cutoff,  test = later vintages
   ┌────────────── train fold ──────────────┐        held-out latest vintage
   │  fit pipeline (imputers/encoders/scaler)│              (test fold)
   │  + OOF target encoding                  │
   │  fit logistic + XGBoost (early stop on  │
   │  a later-vintage sub-slice)             │
   │  fit isotonic/Platt calibration map     │
   └─────────────────────────────────────────┘
        │
        ▼  evaluate on the held-out LATEST vintage
   metrics{XGB} · metrics{logistic} · base-rate AUC(0.5 floor)
        │
        ▼  DeLong(XGB, logistic) + Bonferroni over the recorded n_trials grid
        │
        ▼  emit  <2MB booster JSON  +  fitted pipeline  +  calibration map  +  RunManifest
```

The held-out evaluation is **always on the latest vintage** — the rows furthest
in the future — so the reported number is the one a real deployment would see.

## Key invariants

The compute core guarantees, and tests enforce:

1. **No leakage.** No column in `LEAKAGE_COLS` survives `drop_leakage` or the full
   pipeline — on the synthetic panel **and** a perturbed-schema fixture
   (`property/test_invariants.py`).
2. **No look-ahead.** Every imputer/encoder/scaler is a deterministic function of
   the train fold; perturbing later-vintage rows leaves train-fold transform
   statistics unchanged (`property/test_pipeline.py`).
3. **Temporal order.** No train row has an `issue_d` after any test row
   (`assert_temporal_order`; `regression/test_temporal_and_golden.py`).
4. **Label hygiene.** In-progress loans (`Current`, `Late`, `In Grace Period`) are
   excluded from `(X, y)`; documented and unit-tested.
5. **Calibrated PD.** The calibration map is monotone and outputs `[0, 1]`
   (`property/test_invariants.py`).
6. **Permutation invariance.** Scores do not depend on row order
   (`property/test_pipeline.py`).
7. **Honest band.** A leakage-free model lands in ROC-AUC ∈ [0.60, 0.85],
   Brier ∈ [0.05, 0.25] — above the 0.5 floor, nowhere near a fraudulent ~0.99
   (`regression/test_temporal_and_golden.py`).
8. **Multiplicity honesty.** The recorded `n_trials` ≥ the actual config grid, and
   DeLong's p-value scales with the Bonferroni comparison count (ADR-0005).
9. **Determinism.** Same `RunManifest` seed -> byte-identical artifacts.
10. **Import purity.** Importing any `src/lendingclub_default` module triggers no
    I/O, no network, no training, no RNG draw (subprocess-tested).

## Testing strategy

Tests are partitioned by intent under `tests/` (markers in `pyproject.toml`),
with seeded `conftest.py` fixtures (`synthetic_panel`, `k_vintage_fixture`,
`schema_with_leakage`) giving every layer deterministic, adversarial inputs:

- **`unit/`** — label construction, `LEAKAGE_COLS` drop completeness, the
  generator's schema/monotonicity, the temporal split's partitioning.
- **`parity/`** — golden checks against independent references: ROC-AUC / PR-AUC /
  Brier / log-loss / KS vs `sklearn.metrics` & SciPy at `1e-10`; calibration vs
  `CalibratedClassifierCV`; DeLong AUC vs `roc_auc_score` at `1e-8`.
- **`property/`** (Hypothesis) — the invariants above: no leakage survives, no
  look-ahead, PD in `[0, 1]` and monotone, prediction permutation-invariance,
  DeLong Bonferroni monotonicity.
- **`regression/`** — the honest band locked on the frozen synthetic vintage
  fixture, and the temporal-order guard.
- **`integration/`** — CLI `train`→`score`→`evaluate` round-trip producing a
  loadable booster JSON, and the public `score_one` entrypoint.

Coverage gate **≥ 85%** (CI). ruff + strict mypy clean.

## Backend & frontend boundary

The compute core is decoupled from delivery. The backend vendors
`lendingclub-default[data]` (not `[viz]`/`[dev]`, never `shap`) under
`api/lib/lendingclub_default/` and exposes
`POST /tools/lendingclub-default/run`, returning summary scalars (via
`_safe_float`) plus Plotly `{data, layout}` figures (via
`json.loads(pio.to_json(fig, validate=False))`). The route is **import-pure** and
does **no per-request training**: a module-level `_BOOSTER=None` loads the
committed booster JSON + fitted pipeline lazily at first call. Validation errors
return `422`; an artifact-load failure returns `502`. A best-effort
`platform.tool_runs` row is written but never fatal. The frontend renders the
figures and surfaces the honest caption — "Ranks risk; does not predict
individuals. Demo model trained on synthetic LC-schema data." — as the first
thing a visitor reads.
