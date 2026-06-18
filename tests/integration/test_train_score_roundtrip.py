"""Integration: ``train`` -> save -> load -> score round-trip on synthetic data.

Exercises the full orchestrator end-to-end: :func:`train` (on a small synthetic
panel) emits a ``<2MB`` booster JSON + fitted pipeline + calibration map +
manifest; :func:`load_booster` reassembles the bundle lazily; and scoring a held-
out application yields PDs in ``[0, 1]``. Also asserts the recorded ``n_trials``
is ``>=`` the swept config grid (the overfitting-honesty guard).

The orchestration in ``train.py`` is fully implemented here, but it calls into
model/calibration kernels that sibling groups fill in parallel. While any of those
upstream kernels is still a ``NotImplementedError`` stub, the round-trip is skipped
(not failed), so the suite stays green during the parallel build and these
assertions activate automatically once every group has landed.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from lendingclub_default._exceptions import ValidationError
from lendingclub_default.cli import build_app
from lendingclub_default.train import (
    ScoredArtifacts,
    TrainArtifacts,
    _reset_booster_cache,
    _swept_config_grid,
    load_artifacts,
    load_booster,
    train,
)

if TYPE_CHECKING:
    import pandas as pd


def _skip_if_upstream_stubbed(fn: Callable[[], object]) -> object:
    """Run ``fn``; skip the test if an upstream kernel is still a stub.

    Keeps the suite green while sibling groups fill in the model/calibration
    kernels the orchestrator depends on - the real assertions activate the moment
    every stub is implemented.
    """
    try:
        return fn()
    except NotImplementedError as exc:  # pragma: no cover - parallel-build guard
        pytest.skip(f"upstream kernel not yet implemented: {exc}")


@pytest.fixture
def _small_config() -> object:
    """A tiny synthetic config so the round-trip trains fast."""
    from lendingclub_default.data.synthetic import SyntheticConfig

    return SyntheticConfig(n_loans=3_000, seed=20260616)


@pytest.fixture
def _fast_xgb() -> object:
    """A shallow, few-round booster config to keep the round-trip quick."""
    from lendingclub_default.models.xgb import XGBConfig

    return XGBConfig(n_estimators=60, max_depth=3, early_stopping_rounds=10, seed=0)


@pytest.mark.integration
def test_train_entry_point_exists() -> None:
    """The end-to-end ``train`` callable is importable and invocable."""
    assert callable(train)


@pytest.mark.integration
def test_cli_builder_entry_point_exists() -> None:
    """The Typer app builder is importable; once filled it returns a Typer app."""
    try:
        app = build_app()
    except NotImplementedError:  # pragma: no cover - CLI group not yet landed
        pytest.skip("CLI build_app not yet implemented")
    assert app is not None


@pytest.mark.integration
def test_swept_config_grid_is_nonempty() -> None:
    """The overfitting-guard config grid is a non-empty tuple of dicts."""
    grid = _swept_config_grid()
    assert isinstance(grid, tuple)
    assert len(grid) >= 1
    assert all(isinstance(entry, dict) for entry in grid)


@pytest.mark.integration
@pytest.mark.slow
def test_train_save_load_score_roundtrip(
    tmp_path: Path,
    _small_config: object,
    _fast_xgb: object,
    synthetic_panel: pd.DataFrame,
) -> None:
    """End-to-end: train -> persist -> lazy-load -> score yields a usable PD."""
    _reset_booster_cache()
    out_dir = tmp_path / "artifacts"

    artifacts = _skip_if_upstream_stubbed(
        lambda: train(
            data_path=None,
            out_dir=out_dir,
            synthetic_config=_small_config,  # type: ignore[arg-type]
            xgb_config=_fast_xgb,  # type: ignore[arg-type]
            seed=20260616,
        )
    )
    assert isinstance(artifacts, TrainArtifacts)

    # --- the artifacts the API ships ------------------------------------------ #
    assert artifacts.booster_path.exists()
    assert artifacts.pipeline_path.exists()
    assert (out_dir / "calibration.json").exists()
    assert (out_dir / "manifest.json").exists()
    assert artifacts.data_source == "synthetic"

    # The committed booster must come in under the <2MB ceiling.
    assert artifacts.booster_path.stat().st_size < 2 * 1024 * 1024

    # --- honest-null + multiplicity guards ------------------------------------ #
    # n_trials must cover at least the swept config grid (overfitting honesty).
    assert artifacts.n_trials >= len(_swept_config_grid())

    # Headline metrics are real probabilities of a believable, leakage-free model.
    assert 0.0 <= artifacts.metrics.roc_auc <= 1.0
    assert 0.0 <= artifacts.metrics.brier <= 1.0
    # A leakage-free model beats the 0.5 base-rate floor (not a fraudulent ~0.99).
    assert artifacts.metrics.roc_auc > artifacts.base_rate_auc
    assert artifacts.metrics.roc_auc < 0.95

    # DeLong compares XGB vs logistic on the same held-out rows.
    assert artifacts.delong.n_comparisons == len(_swept_config_grid())
    assert 0.0 <= artifacts.delong.p_value <= 1.0

    # --- lazy load + score round-trip ----------------------------------------- #
    bundle = load_booster(out_dir)
    assert isinstance(bundle, ScoredArtifacts)
    # The lazy sentinel caches: a second call returns the SAME object.
    assert load_booster(out_dir) is bundle

    # Score the held-out style panel (leakage cols are dropped inside ``score``).
    pds = bundle.score(synthetic_panel.head(50))
    assert pds.shape == (50,)
    assert np.all(pds >= 0.0) and np.all(pds <= 1.0)
    assert np.all(np.isfinite(pds))

    _reset_booster_cache()


@pytest.mark.integration
@pytest.mark.slow
def test_load_artifacts_recovers_recorded_scalars(
    tmp_path: Path,
    _small_config: object,
    _fast_xgb: object,
) -> None:
    """A reloaded bundle carries the AUC / base-rate / data-source recorded at train."""
    out_dir = tmp_path / "artifacts"
    artifacts = _skip_if_upstream_stubbed(
        lambda: train(
            data_path=None,
            out_dir=out_dir,
            synthetic_config=_small_config,  # type: ignore[arg-type]
            xgb_config=_fast_xgb,  # type: ignore[arg-type]
            seed=20260616,
        )
    )
    assert isinstance(artifacts, TrainArtifacts)

    bundle = load_artifacts(out_dir)
    assert bundle.data_source == "synthetic"
    assert bundle.model_auc == pytest.approx(artifacts.metrics.roc_auc)
    assert 0.0 <= bundle.base_rate <= 1.0


@pytest.mark.integration
@pytest.mark.slow
def test_train_then_score_one_public_entrypoint(
    tmp_path: Path,
    _small_config: object,
    _fast_xgb: object,
    synthetic_panel: pd.DataFrame,
) -> None:
    """End-to-end: train -> ``load_booster`` -> ``score_one`` honours the public contract.

    Exercises the exact entrypoints the backend calls: train on the synthetic
    fixture, lazily load the scoring bundle, and score one application both as a
    ``dict`` and as a one-row DataFrame. Asserts the public scoring contract
    ``{pd, decile, reason_codes, predicted_label, threshold, ...}``.
    """
    _reset_booster_cache()
    out_dir = tmp_path / "artifacts"
    artifacts = _skip_if_upstream_stubbed(
        lambda: train(
            data_path=None,
            out_dir=out_dir,
            synthetic_config=_small_config,  # type: ignore[arg-type]
            xgb_config=_fast_xgb,  # type: ignore[arg-type]
            seed=20260616,
        )
    )
    assert isinstance(artifacts, TrainArtifacts)

    bundle = load_booster(out_dir)
    assert isinstance(bundle, ScoredArtifacts)
    # Reason coefficients persisted for the container-safe (SHAP-free) explanations.
    assert bundle.reason_coefficients
    assert bundle.feature_names

    application = {
        "loan_amnt": 15_000.0,
        "term": "36 months",
        "int_rate": 13.5,
        "grade": "C",
        "sub_grade": "C2",
        "emp_length": 5.0,
        "home_ownership": "MORTGAGE",
        "annual_inc": 65_000.0,
        "dti": 18.4,
        "fico_range_low": 690.0,
        "fico_range_high": 694.0,
        "revol_util": 42.7,
        "open_acc": 11.0,
        "pub_rec": 0.0,
        "purpose": "debt_consolidation",
        "addr_state": "CA",
        "installment": 509.0,
        "verification_status": "Source Verified",
    }

    result = bundle.score_one(application, top_k=4)

    # --- the public scoring contract ------------------------------------------ #
    assert set(result) >= {
        "pd",
        "decile",
        "reason_codes",
        "predicted_label",
        "threshold",
        "model_auc",
        "base_rate",
        "data_source",
    }
    assert 0.0 <= result["pd"] <= 1.0
    assert 1 <= result["decile"] <= 10
    assert result["predicted_label"] in {"default", "fully_paid"}
    assert result["data_source"] == "synthetic"

    # Reason codes: <= top_k, each a signed contribution with a direction.
    codes = result["reason_codes"]
    assert 0 < len(codes) <= 4
    for code in codes:
        assert set(code) == {"feature", "direction", "contribution"}
        assert code["direction"] in {"increases", "decreases"}
        assert np.isfinite(code["contribution"])

    # Scoring the same application as a one-row DataFrame agrees with the dict path.
    import pandas as pd

    df_result = bundle.score_one(pd.DataFrame([application]))
    assert df_result["pd"] == pytest.approx(result["pd"])

    # A non-singleton frame is rejected (exactly one application per call).
    with pytest.raises(ValidationError):
        bundle.score_one(synthetic_panel.head(3))

    _reset_booster_cache()
