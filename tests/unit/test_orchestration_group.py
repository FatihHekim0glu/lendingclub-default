"""Unit tests for the training orchestrator and the CLI command handlers.

These run the real end-to-end pipeline on a tiny synthetic panel (a full
``train`` is fast: a few thousand rows, a shallow few-round booster), so the
orchestration in ``train.py`` and the thin Typer command layer in ``cli.py`` are
exercised for real rather than mocked. They assert the honest-headline
invariants (PD in ``[0, 1]``, a leakage-free model beats the 0.5 floor and does
not hit a fraudulent ~0.99), the save/load round-trip, the public scoring
contract, the ``<2MB`` artifact-size guard, and the handled-error exit codes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from lendingclub_default._exceptions import ArtifactError, ValidationError
from lendingclub_default.cli import (
    build_app,
    evaluate_command,
    score_command,
    train_command,
)
from lendingclub_default.data.synthetic import SyntheticConfig
from lendingclub_default.models.xgb import XGBConfig
from lendingclub_default.train import (
    MAX_BOOSTER_BYTES,
    ScoredArtifacts,
    TrainArtifacts,
    _check_booster_size,
    _reset_booster_cache,
    _swept_config_grid,
    load_artifacts,
    load_booster,
    train,
)

if TYPE_CHECKING:
    import pandas as pd

_SMALL_CONFIG = SyntheticConfig(n_loans=2_500, seed=20260616)
_FAST_XGB = XGBConfig(n_estimators=40, max_depth=3, early_stopping_rounds=8, seed=0)


@pytest.fixture(scope="module")
def trained_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train the pipeline once on a tiny synthetic panel; share the artifacts.

    Scoped to the module so the (fast) full train runs a single time and every
    test in this file reads the same emitted booster / pipeline / calibration /
    manifest bundle.
    """
    out_dir = tmp_path_factory.mktemp("artifacts")
    train(
        data_path=None,
        out_dir=out_dir,
        synthetic_config=_SMALL_CONFIG,
        xgb_config=_FAST_XGB,
        seed=20260616,
    )
    return out_dir


@pytest.mark.unit
def test_train_emits_all_artifacts_and_honest_metrics(tmp_path: Path) -> None:
    """A full run writes the four artifacts and lands an honest, leakage-free AUC."""
    out_dir = tmp_path / "run"
    artifacts = train(
        data_path=None,
        out_dir=out_dir,
        synthetic_config=_SMALL_CONFIG,
        xgb_config=_FAST_XGB,
        seed=20260616,
    )

    assert isinstance(artifacts, TrainArtifacts)
    assert artifacts.booster_path.exists()
    assert artifacts.pipeline_path.exists()
    assert (out_dir / "calibration.json").exists()
    assert (out_dir / "manifest.json").exists()
    assert artifacts.data_source == "synthetic"

    # The committed booster must come in under the <2MB ceiling.
    assert artifacts.booster_path.stat().st_size < MAX_BOOSTER_BYTES

    # Honest band: above the 0.5 base-rate floor, nowhere near a fraudulent ~0.99.
    assert 0.0 <= artifacts.metrics.roc_auc <= 1.0
    assert 0.0 <= artifacts.metrics.brier <= 1.0
    assert artifacts.metrics.roc_auc > artifacts.base_rate_auc
    assert artifacts.metrics.roc_auc < 0.95

    # Multiplicity honesty: recorded trials cover the whole swept config grid.
    assert artifacts.n_trials >= len(_swept_config_grid())
    assert artifacts.delong.n_comparisons == len(_swept_config_grid())
    assert 0.0 <= artifacts.delong.p_value <= 1.0


@pytest.mark.unit
def test_train_artifacts_to_dict_is_json_serialisable(trained_dir: Path) -> None:
    """``TrainArtifacts.to_dict`` round-trips through ``json.dumps`` cleanly."""
    artifacts = train(
        data_path=None,
        out_dir=trained_dir / "again",
        synthetic_config=_SMALL_CONFIG,
        xgb_config=_FAST_XGB,
        seed=20260616,
    )
    payload = artifacts.to_dict()
    assert set(payload) >= {"booster_path", "metrics", "delong", "n_trials", "manifest"}
    # No exception => fully serialisable.
    json.loads(json.dumps(payload))


@pytest.mark.unit
def test_load_artifacts_recovers_recorded_scalars(trained_dir: Path) -> None:
    """A reloaded bundle carries the AUC / base-rate / data-source recorded at train."""
    bundle = load_artifacts(trained_dir)
    assert isinstance(bundle, ScoredArtifacts)
    assert bundle.data_source == "synthetic"
    assert 0.0 <= bundle.model_auc <= 1.0
    assert 0.0 <= bundle.base_rate <= 1.0
    assert bundle.feature_names
    assert bundle.reason_coefficients


@pytest.mark.unit
def test_load_booster_caches_the_bundle(trained_dir: Path) -> None:
    """The lazy ``_BOOSTER`` sentinel returns the same object on a second call."""
    _reset_booster_cache()
    first = load_booster(trained_dir)
    assert load_booster(trained_dir) is first
    # A fresh, uncached load is a distinct object.
    assert load_booster(trained_dir, use_cache=False) is not first
    _reset_booster_cache()


@pytest.mark.unit
def test_score_returns_probabilities_in_unit_interval(
    trained_dir: Path,
    synthetic_panel: pd.DataFrame,
) -> None:
    """Scoring a raw panel yields finite PDs in ``[0, 1]`` (leakage cols dropped)."""
    bundle = load_artifacts(trained_dir)
    pds = bundle.score(synthetic_panel.head(40))
    assert pds.shape == (40,)
    assert np.all(pds >= 0.0)
    assert np.all(pds <= 1.0)
    assert np.all(np.isfinite(pds))


@pytest.mark.unit
def test_score_one_honours_the_public_contract(trained_dir: Path) -> None:
    """``score_one`` returns the documented keys for both dict and DataFrame input."""
    import pandas as pd

    bundle = load_artifacts(trained_dir)
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
    assert 0 < len(result["reason_codes"]) <= 4
    for code in result["reason_codes"]:
        assert set(code) == {"feature", "direction", "contribution"}
        assert code["direction"] in {"increases", "decreases"}
        assert np.isfinite(code["contribution"])

    # The DataFrame path agrees with the dict path.
    df_result = bundle.score_one(pd.DataFrame([application]))
    assert df_result["pd"] == pytest.approx(result["pd"])


@pytest.mark.unit
def test_score_one_rejects_a_non_singleton_frame(
    trained_dir: Path,
    synthetic_panel: pd.DataFrame,
) -> None:
    """Exactly one application per ``score_one`` call; a multi-row frame is rejected."""
    bundle = load_artifacts(trained_dir)
    with pytest.raises(ValidationError):
        bundle.score_one(synthetic_panel.head(3))


@pytest.mark.unit
def test_check_booster_size_rejects_oversized_artifact(tmp_path: Path) -> None:
    """The <2MB guard raises ``ArtifactError`` for an oversized booster JSON."""
    big = tmp_path / "booster.json"
    big.write_bytes(b"0" * (MAX_BOOSTER_BYTES + 1))
    with pytest.raises(ArtifactError, match="ceiling"):
        _check_booster_size(big)


@pytest.mark.unit
def test_load_artifacts_missing_dir_raises_artifact_error(tmp_path: Path) -> None:
    """Loading from a directory with no artifacts raises ``ArtifactError``."""
    with pytest.raises(ArtifactError):
        load_artifacts(tmp_path / "does-not-exist")


@pytest.mark.unit
def test_build_app_registers_the_three_commands() -> None:
    """The Typer app builder returns an app exposing train/score/evaluate."""
    app = build_app()
    assert app is not None
    names = {cmd.name for cmd in app.registered_commands}
    assert {"train", "score", "evaluate"} <= names


@pytest.mark.unit
def test_train_command_runs_and_returns_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI ``train`` handler trains, prints a summary, and exits ``0``."""
    code = train_command(data=None, out_dir=str(tmp_path / "cli-art"), seed=20260616)
    assert code == 0
    out = capsys.readouterr().out
    assert "ROC-AUC" in out
    assert "ranks risk" in out


@pytest.mark.unit
def test_score_command_demo_application_round_trip(
    trained_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``score`` with no --application scores the built-in demo and exits ``0``."""
    code = score_command(artifacts_dir=str(trained_dir), application=None)
    assert code == 0
    out = capsys.readouterr().out
    assert "calibrated PD" in out
    assert "risk decile" in out


@pytest.mark.unit
def test_score_command_reads_an_application_json(
    trained_dir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``score`` reads an application JSON file and scores it."""
    app_path = tmp_path / "app.json"
    app_path.write_text(
        json.dumps(
            {
                "loan_amnt": 12_000.0,
                "term": "36 months",
                "int_rate": 11.2,
                "grade": "B",
                "sub_grade": "B3",
                "emp_length": 8.0,
                "home_ownership": "OWN",
                "annual_inc": 80_000.0,
                "dti": 12.0,
                "fico_range_low": 720.0,
                "fico_range_high": 724.0,
                "revol_util": 30.0,
                "open_acc": 9.0,
                "pub_rec": 0.0,
                "purpose": "credit_card",
                "addr_state": "TX",
                "installment": 395.0,
                "verification_status": "Verified",
            }
        ),
        encoding="utf-8",
    )
    code = score_command(artifacts_dir=str(trained_dir), application=str(app_path))
    assert code == 0
    assert "calibrated PD" in capsys.readouterr().out


@pytest.mark.unit
def test_score_command_missing_application_file_returns_one(
    trained_dir: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A non-existent --application path is a handled error: exit ``1``."""
    code = score_command(
        artifacts_dir=str(trained_dir),
        application=str(tmp_path / "nope.json"),
    )
    assert code == 1
    assert "does not exist" in capsys.readouterr().out


@pytest.mark.unit
def test_score_command_missing_artifacts_returns_one(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing trained bundle is a handled ``ArtifactError``: exit ``1``."""
    code = score_command(artifacts_dir=str(tmp_path / "empty"), application=None)
    assert code == 1
    assert "error:" in capsys.readouterr().out


@pytest.mark.unit
def test_evaluate_command_on_fresh_synthetic_panel(
    trained_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``evaluate`` scores a fresh synthetic held-out panel and prints the bundle."""
    code = evaluate_command(artifacts_dir=str(trained_dir), data=None)
    assert code == 0
    out = capsys.readouterr().out
    assert "ROC-AUC" in out
    assert "Brier" in out


@pytest.mark.unit
def test_evaluate_command_missing_artifacts_returns_one(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A missing bundle makes ``evaluate`` exit ``1`` with a handled error."""
    code = evaluate_command(artifacts_dir=str(tmp_path / "empty"), data=None)
    assert code == 1
    assert "error:" in capsys.readouterr().out
