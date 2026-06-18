"""Tests for the ``models/`` group: baselines, XGBoost, calibration, reason codes.

These cover the brief's headline model invariants:

- the base-rate predictor returns the train base rate for every loan;
- L2 logistic beats random (AUC > 0.5) on the seeded synthetic panel and matches
  ``sklearn`` to floating tolerance (parity / determinism);
- XGBoost is deterministic, fits with imbalance weighting + early stopping, and
  round-trips through the ``<2MB`` booster JSON;
- the calibration map produces a PD in ``[0, 1]`` and is monotone non-decreasing
  in the raw score (isotonic), and lowers (never raises) the Brier score;
- reason codes rank features by signed ``coef * x`` contribution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from lendingclub_default._exceptions import ArtifactError, ValidationError
from lendingclub_default.models.baselines import BaseRatePredictor, fit_logistic
from lendingclub_default.models.calibrate import CalibratedModel, fit_calibration
from lendingclub_default.models.reason_codes import ReasonCode, reason_codes_from_logit
from lendingclub_default.models.xgb import (
    XGBConfig,
    fit_xgb,
    load_booster,
    predict_proba,
    save_booster,
)

_NUMERIC_FEATURES: tuple[str, ...] = (
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "dti",
    "fico_range_low",
    "revol_util",
    "open_acc",
    "pub_rec",
    "installment",
)


def _xy(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build a leakage-free numeric design matrix + binary default target."""
    x = panel[list(_NUMERIC_FEATURES)].astype("float64").reset_index(drop=True)
    y = (panel["loan_status"] == "Charged Off").astype(int).reset_index(drop=True)
    return x, y


def _temporal_folds(
    panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split by ``issue_d`` vintage into train / valid / test (no look-ahead)."""
    vintages = sorted(panel["issue_d"].unique())
    train_v = set(vintages[:-3])
    valid_v = {vintages[-3]}
    test_v = set(vintages[-2:])
    tr = panel[panel["issue_d"].isin(train_v)]
    va = panel[panel["issue_d"].isin(valid_v)]
    te = panel[panel["issue_d"].isin(test_v)]
    x_tr, y_tr = _xy(tr)
    x_va, y_va = _xy(va)
    x_te, y_te = _xy(te)
    return x_tr, y_tr, x_va, y_va, x_te, y_te


# --------------------------------------------------------------------------- #
# Base-rate predictor                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_base_rate_predictor_returns_base_rate(synthetic_panel: pd.DataFrame) -> None:
    """BaseRatePredictor.fit records y.mean() and predicts it for every row."""
    x, y = _xy(synthetic_panel)
    model = BaseRatePredictor.fit(y)
    assert model.base_rate == pytest.approx(float(y.mean()))
    preds = model.predict_proba(x)
    assert preds.shape == (len(x),)
    assert np.allclose(preds, float(y.mean()))
    # The ~15% base rate is reproduced (sanity on the fixture).
    assert 0.05 < model.base_rate < 0.30


@pytest.mark.unit
def test_base_rate_predictor_to_dict_and_empty_guard() -> None:
    """to_dict is JSON-safe; empty target raises ValidationError."""
    payload = BaseRatePredictor(base_rate=0.15, meta={"n_train": 10}).to_dict()
    assert payload["base_rate"] == 0.15
    assert payload["meta"]["n_train"] == 10
    with pytest.raises(ValidationError):
        BaseRatePredictor.fit(pd.Series([], dtype="float64"))


# --------------------------------------------------------------------------- #
# L2 logistic baseline                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_logistic_beats_random_on_synthetic(synthetic_panel: pd.DataFrame) -> None:
    """The L2 logistic baseline ranks risk well above chance (AUC > 0.5)."""
    x_tr, y_tr, _, _, x_te, y_te = _temporal_folds(synthetic_panel)
    model = fit_logistic(x_tr, y_tr, seed=0)
    proba = model.predict_proba(x_te.to_numpy())[:, 1]
    auc = roc_auc_score(y_te.to_numpy(), proba)
    # Believable, leakage-free, application-time AUC band from the brief.
    assert 0.60 < auc < 0.85


@pytest.mark.parity
@pytest.mark.unit
def test_logistic_is_deterministic(synthetic_panel: pd.DataFrame) -> None:
    """Re-fitting with the same seed yields identical coefficients (determinism)."""
    x_tr, y_tr, _, _, _, _ = _temporal_folds(synthetic_panel)
    m1 = fit_logistic(x_tr, y_tr, seed=7)
    m2 = fit_logistic(x_tr, y_tr, seed=7)
    assert np.allclose(m1.coef_, m2.coef_, atol=1e-10)
    assert np.allclose(m1.intercept_, m2.intercept_, atol=1e-10)


@pytest.mark.unit
def test_logistic_input_guards(synthetic_panel: pd.DataFrame) -> None:
    """Empty or row-misaligned inputs raise ValidationError."""
    x_tr, y_tr, _, _, _, _ = _temporal_folds(synthetic_panel)
    with pytest.raises(ValidationError):
        fit_logistic(x_tr.iloc[:0], y_tr.iloc[:0])
    with pytest.raises(ValidationError):
        fit_logistic(x_tr, y_tr.iloc[:-5])


# --------------------------------------------------------------------------- #
# XGBoost booster                                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_xgb_config_to_dict_round_trips() -> None:
    """XGBConfig serializes its hyperparameters as JSON-safe scalars."""
    payload = XGBConfig(max_depth=5, learning_rate=0.1).to_dict()
    assert payload["max_depth"] == 5
    assert payload["learning_rate"] == 0.1
    assert payload["early_stopping_rounds"] == 30


@pytest.mark.unit
def test_xgb_fit_predict_and_beats_random(synthetic_panel: pd.DataFrame) -> None:
    """fit_xgb trains with early stopping; predictions are PDs that beat chance."""
    x_tr, y_tr, x_va, y_va, x_te, y_te = _temporal_folds(synthetic_panel)
    cfg = XGBConfig(n_estimators=120, early_stopping_rounds=20, seed=0)
    booster = fit_xgb(x_tr, y_tr, x_va, y_va, config=cfg)
    proba = predict_proba(booster, x_te)
    assert proba.shape == (len(x_te),)
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0
    auc = roc_auc_score(y_te.to_numpy(), proba)
    assert auc > 0.60


@pytest.mark.parity
@pytest.mark.unit
def test_xgb_is_deterministic(synthetic_panel: pd.DataFrame) -> None:
    """Two fits with identical config + seed give identical predictions."""
    x_tr, y_tr, x_va, y_va, x_te, _ = _temporal_folds(synthetic_panel)
    cfg = XGBConfig(n_estimators=80, seed=0)
    b1 = fit_xgb(x_tr, y_tr, x_va, y_va, config=cfg)
    b2 = fit_xgb(x_tr, y_tr, x_va, y_va, config=cfg)
    assert np.allclose(predict_proba(b1, x_te), predict_proba(b2, x_te), atol=1e-10)


@pytest.mark.unit
def test_xgb_predict_invariant_to_row_permutation(synthetic_panel: pd.DataFrame) -> None:
    """Invariant (b): permuting test rows permutes predictions identically."""
    x_tr, y_tr, x_va, y_va, x_te, _ = _temporal_folds(synthetic_panel)
    booster = fit_xgb(x_tr, y_tr, x_va, y_va, config=XGBConfig(n_estimators=60))
    base = predict_proba(booster, x_te)
    perm = np.random.default_rng(0).permutation(len(x_te))
    permuted = predict_proba(booster, x_te.iloc[perm].reset_index(drop=True))
    assert np.allclose(base[perm], permuted, atol=1e-10)


@pytest.mark.integration
def test_xgb_save_load_round_trip(synthetic_panel: pd.DataFrame, tmp_path: Path) -> None:
    """The booster JSON round-trips, stays <2MB, and reloads with equal scores."""
    x_tr, y_tr, x_va, y_va, x_te, _ = _temporal_folds(synthetic_panel)
    booster = fit_xgb(x_tr, y_tr, x_va, y_va, config=XGBConfig(n_estimators=100))
    path = tmp_path / "booster.json"
    save_booster(booster, path)
    assert path.is_file()
    assert path.stat().st_size < 2_000_000  # <2MB committed-artifact budget
    reloaded = load_booster(path, use_cache=False)
    assert np.allclose(predict_proba(booster, x_te), predict_proba(reloaded, x_te), atol=1e-10)


@pytest.mark.unit
def test_load_booster_missing_artifact_raises(tmp_path: Path) -> None:
    """A missing artifact surfaces as ArtifactError (mapped to 502 in the API)."""
    with pytest.raises(ArtifactError):
        load_booster(tmp_path / "does_not_exist.json", use_cache=False)


@pytest.mark.unit
def test_load_booster_caches_via_sentinel(synthetic_panel: pd.DataFrame, tmp_path: Path) -> None:
    """With ``use_cache``, the module ``_BOOSTER`` sentinel is populated and reused."""
    from lendingclub_default.models import xgb as xgb_mod

    x_tr, y_tr, x_va, y_va, _, _ = _temporal_folds(synthetic_panel)
    booster = fit_xgb(x_tr, y_tr, x_va, y_va, config=XGBConfig(n_estimators=40))
    path = tmp_path / "cached.json"
    save_booster(booster, path)

    prev = xgb_mod._BOOSTER
    try:
        xgb_mod._BOOSTER = None
        first = load_booster(path, use_cache=True)
        assert xgb_mod._BOOSTER is first  # sentinel populated
        # A second call ignores the path and returns the cached instance.
        second = load_booster(tmp_path / "missing.json", use_cache=True)
        assert second is first
    finally:
        xgb_mod._BOOSTER = prev


@pytest.mark.unit
def test_load_booster_corrupt_artifact_raises(tmp_path: Path) -> None:
    """A malformed booster file surfaces as ArtifactError, not a raw XGBoost error."""
    bad = tmp_path / "corrupt.json"
    bad.write_text("{ this is not a booster }", encoding="utf-8")
    with pytest.raises(ArtifactError):
        load_booster(bad, use_cache=False)


@pytest.mark.unit
def test_xgb_input_guards(synthetic_panel: pd.DataFrame) -> None:
    """Empty / mismatched-column folds raise ValidationError."""
    x_tr, y_tr, x_va, y_va, _, _ = _temporal_folds(synthetic_panel)
    with pytest.raises(ValidationError):
        fit_xgb(x_tr.iloc[:0], y_tr.iloc[:0], x_va, y_va)
    with pytest.raises(ValidationError):
        fit_xgb(x_tr, y_tr, x_va.drop(columns=["dti"]), y_va)
    booster = fit_xgb(x_tr, y_tr, x_va, y_va, config=XGBConfig(n_estimators=20))
    with pytest.raises(ValidationError):
        predict_proba(booster, x_tr.iloc[:0])


# --------------------------------------------------------------------------- #
# Calibration                                                                  #
# --------------------------------------------------------------------------- #
def _miscalibrated_scores(n: int = 1500, seed: int = 0) -> tuple[np.ndarray, pd.Series]:
    """A deliberately over-confident score / label pair for calibration tests."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    # squashing toward the extremes makes the raw "score" badly calibrated
    base = np.where(y == 1, rng.beta(2.0, 3.0, n), rng.beta(1.5, 4.0, n))
    raw = np.clip(base**0.5, 0.0, 1.0)
    return raw, pd.Series(y)


@pytest.mark.unit
@pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
def test_calibration_pd_in_unit_interval_and_lowers_brier(method: str) -> None:
    """PD stays in [0, 1] and the Brier score does not increase after calibration."""
    raw, y = _miscalibrated_scores()
    model, result = fit_calibration(raw, y, method=method)
    cal = model.calibrate(raw)
    assert cal.min() >= 0.0
    assert cal.max() <= 1.0
    assert result.method == method
    assert result.n_calibration == len(raw)
    # Fitting on the same fold cannot do worse than the raw scores in-sample.
    assert result.brier_after <= result.brier_before + 1e-9


@pytest.mark.property
@pytest.mark.unit
def test_isotonic_calibration_is_monotone() -> None:
    """Invariant (c): the isotonic map is monotone non-decreasing in the raw score."""
    raw, y = _miscalibrated_scores()
    model, _ = fit_calibration(raw, y, method="isotonic")
    grid = np.linspace(0.0, 1.0, 200)
    cal = model.calibrate(grid)
    assert np.all(np.diff(cal) >= -1e-9)


@pytest.mark.parity
@pytest.mark.unit
def test_isotonic_matches_sklearn_isotonic() -> None:
    """Calibrated PDs match sklearn IsotonicRegression to floating tolerance."""
    from sklearn.isotonic import IsotonicRegression

    raw, y = _miscalibrated_scores()
    model, _ = fit_calibration(raw, y, method="isotonic")
    ref = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ref.fit(raw, y.to_numpy().astype(float))
    grid = np.linspace(0.0, 1.0, 50)
    assert np.allclose(model.calibrate(grid), ref.predict(grid), atol=1e-9)


@pytest.mark.unit
def test_calibration_serialization_round_trip() -> None:
    """A serialized isotonic map reproduces its calibration without re-fitting."""
    raw, y = _miscalibrated_scores()
    model, _ = fit_calibration(raw, y, method="isotonic")
    rebuilt = CalibratedModel(method=model.method, params=model.to_dict()["params"])
    assert np.allclose(model.calibrate(raw), rebuilt.calibrate(raw), atol=1e-12)


@pytest.mark.unit
def test_calibration_input_guards() -> None:
    """Unknown method, empty scores, and misaligned y all raise ValidationError."""
    with pytest.raises(ValidationError):
        fit_calibration(np.array([0.1, 0.9]), pd.Series([0, 1]), method="bogus")
    with pytest.raises(ValidationError):
        fit_calibration(np.array([]), pd.Series([], dtype="int64"))
    with pytest.raises(ValidationError):
        fit_calibration(np.array([0.1, 0.2, 0.3]), pd.Series([0, 1]))
    with pytest.raises(ValidationError):
        CalibratedModel(method="bogus", params={}).calibrate(np.array([0.1]))
    # An isotonic map with no knots cannot interpolate -> guarded.
    with pytest.raises(ValidationError):
        CalibratedModel(method="isotonic", params={"knots_x": [], "knots_y": []}).calibrate(
            np.array([0.1])
        )


@pytest.mark.unit
def test_calibration_result_to_dict_is_json_safe() -> None:
    """CalibrationResult.to_dict yields float Brier scores and an int count."""
    raw, y = _miscalibrated_scores()
    _, result = fit_calibration(raw, y, method="sigmoid")
    payload = result.to_dict()
    assert isinstance(payload["brier_before"], float)
    assert isinstance(payload["brier_after"], float)
    assert isinstance(payload["n_calibration"], int)
    assert payload["method"] == "sigmoid"


# --------------------------------------------------------------------------- #
# Reason codes                                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_reason_codes_from_logit_ranks_and_signs() -> None:
    """Top contributions are coef*x, ordered by magnitude, with correct direction."""
    coef = pd.Series({"int_rate": 0.5, "fico_range_low": -0.01, "dti": 0.2})
    x_row = pd.Series({"int_rate": 20.0, "fico_range_low": 700.0, "dti": 5.0})
    codes = reason_codes_from_logit(coef, x_row, top_k=2)
    assert [c.feature for c in codes] == ["int_rate", "fico_range_low"]
    assert codes[0].direction == "increases"  # 0.5 * 20 = +10
    assert codes[1].direction == "decreases"  # -0.01 * 700 = -7
    assert codes[0].contribution == pytest.approx(10.0)
    assert codes[1].contribution == pytest.approx(-7.0)


@pytest.mark.unit
def test_reason_code_to_dict_is_json_safe() -> None:
    """ReasonCode.to_dict yields a float contribution."""
    payload = ReasonCode(feature="dti", direction="increases", contribution=0.4).to_dict()
    assert payload == {"feature": "dti", "direction": "increases", "contribution": 0.4}
    assert isinstance(payload["contribution"], float)


@pytest.mark.unit
def test_reason_codes_input_guards() -> None:
    """Non-positive top_k and disjoint feature indexes raise ValidationError."""
    coef = pd.Series({"int_rate": 0.5})
    row = pd.Series({"int_rate": 20.0})
    with pytest.raises(ValidationError):
        reason_codes_from_logit(coef, row, top_k=0)
    with pytest.raises(ValidationError):
        reason_codes_from_logit(coef, pd.Series({"other": 1.0}))


# --------------------------------------------------------------------------- #
# Reason codes - SHAP (DEV-ONLY path; SHAP lives in the [dev] extra)          #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_shap_reason_codes_dev_path(synthetic_panel: pd.DataFrame) -> None:
    """The dev-only SHAP explainer returns top-k signed, directional reason codes.

    SHAP is never imported in the shipped container; this exercises the [dev]
    helper to keep it covered and to confirm it agrees with the booster's own
    feature ordering.
    """
    shap = pytest.importorskip("shap")
    assert shap is not None

    x_tr, y_tr, x_va, y_va, x_te, _ = _temporal_folds(synthetic_panel)
    booster = fit_xgb(x_tr, y_tr, x_va, y_va, config=XGBConfig(n_estimators=40))

    from lendingclub_default.models.reason_codes import shap_reason_codes

    feature_names = list(_NUMERIC_FEATURES)
    codes = shap_reason_codes(
        booster,
        x_te.iloc[0].to_numpy(),
        feature_names,
        top_k=3,
    )
    assert len(codes) == 3
    assert all(c.feature in feature_names for c in codes)
    assert all(c.direction in {"increases", "decreases"} for c in codes)
    # ordered by absolute contribution, descending
    mags = [abs(c.contribution) for c in codes]
    assert mags == sorted(mags, reverse=True)


@pytest.mark.unit
def test_shap_reason_codes_guards(synthetic_panel: pd.DataFrame) -> None:
    """Bad top_k or a feature-name/vector mismatch raise ValidationError."""
    pytest.importorskip("shap")
    from lendingclub_default.models.reason_codes import shap_reason_codes

    x_tr, y_tr, x_va, y_va, x_te, _ = _temporal_folds(synthetic_panel)
    booster = fit_xgb(x_tr, y_tr, x_va, y_va, config=XGBConfig(n_estimators=30))
    row = x_te.iloc[0].to_numpy()

    with pytest.raises(ValidationError):
        shap_reason_codes(booster, row, list(_NUMERIC_FEATURES), top_k=0)
    with pytest.raises(ValidationError):
        # one fewer name than the vector has elements -> misalignment
        shap_reason_codes(booster, row, list(_NUMERIC_FEATURES)[:-1])
