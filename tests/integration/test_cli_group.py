"""Typer CLI tests (cli.py group).

Covers the public surface (``--help`` and command registration), a tiny synthetic
``train`` -> ``score`` -> ``evaluate`` round-trip through the library's public
artifact loader, the handled error paths (missing bundle / missing application
file), and CLI import purity (importing ``cli`` must not import Typer).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from lendingclub_default.cli import (
    _DEMO_APPLICATION,
    app,
    build_app,
    evaluate_command,
    score_command,
    train_command,
)
from lendingclub_default.data.load import APPLICATION_COLUMNS


def _runner() -> object:
    """Return a Typer/Click CliRunner (imported lazily, mirroring the CLI)."""
    from typer.testing import CliRunner

    return CliRunner()


@pytest.mark.integration
def test_cli_help_lists_all_commands() -> None:
    """`--help` succeeds and advertises train / score / evaluate."""
    result = _runner().invoke(build_app(), ["--help"])
    assert result.exit_code == 0
    for command in ("train", "score", "evaluate"):
        assert command in result.stdout


@pytest.mark.integration
@pytest.mark.parametrize("command", ["train", "score", "evaluate"])
def test_cli_subcommand_help(command: str) -> None:
    """Each subcommand exposes its own `--help` without error."""
    result = _runner().invoke(build_app(), [command, "--help"])
    assert result.exit_code == 0


@pytest.mark.integration
def test_demo_application_covers_application_schema() -> None:
    """The built-in demo application carries every required application-time field."""
    for column in APPLICATION_COLUMNS:
        if column == "issue_d":
            # issue_d is a vintage label used for splitting, not a scoring input.
            continue
        assert column in _DEMO_APPLICATION, f"demo application missing {column!r}"


@pytest.mark.integration
@pytest.mark.slow
def test_train_score_evaluate_roundtrip(tmp_path: Path) -> None:
    """train -> score -> evaluate runs end-to-end and yields an honest, sane PD.

    Trains on the synthetic generator, scores the built-in demo application from
    the written bundle, and evaluates a fresh synthetic panel — asserting the
    artifacts land on disk and the metrics sit in the documented honest band.
    """
    out_dir = tmp_path / "artifacts"

    assert train_command(data=None, out_dir=str(out_dir), seed=11) == 0
    # The booster, fitted pipeline, and calibration map all landed.
    assert (out_dir / "booster.json").exists()
    assert (out_dir / "pipeline.joblib").exists()
    assert (out_dir / "calibration.json").exists()
    # The committed booster respects the <2MB ceiling.
    assert (out_dir / "booster.json").stat().st_size < 2_000_000

    assert score_command(artifacts_dir=str(out_dir), application=None) == 0
    assert evaluate_command(artifacts_dir=str(out_dir), data=None) == 0


@pytest.mark.integration
@pytest.mark.slow
def test_score_application_from_file(tmp_path: Path) -> None:
    """A user-supplied application JSON is scored to a PD in [0, 1] + a valid decile."""
    from lendingclub_default.cli import _score_application

    out_dir = tmp_path / "artifacts"
    assert train_command(data=None, out_dir=str(out_dir), seed=3) == 0

    app_file = tmp_path / "application.json"
    app_file.write_text(json.dumps(_DEMO_APPLICATION))
    assert score_command(artifacts_dir=str(out_dir), application=str(app_file)) == 0

    result = _score_application(out_dir, dict(_DEMO_APPLICATION))
    # The CLI helper returns the public scoring contract (pd / decile / reason_codes).
    assert 0.0 <= result["pd"] <= 1.0
    assert 1 <= result["decile"] <= 10
    assert result["predicted_label"] in {"default", "fully_paid"}
    assert 0.5 <= result["model_auc"] <= 1.0
    assert isinstance(result["reason_codes"], list)


@pytest.mark.integration
def test_score_missing_bundle_returns_error() -> None:
    """Scoring against a non-existent artifacts dir returns exit code 1."""
    assert score_command(artifacts_dir="/tmp/__lcd_no_such_dir__", application=None) == 1


@pytest.mark.integration
def test_score_missing_application_file_returns_error(tmp_path: Path) -> None:
    """A missing --application file returns exit code 1 before touching artifacts."""
    missing = tmp_path / "nope.json"
    assert score_command(artifacts_dir=str(tmp_path), application=str(missing)) == 1


@pytest.mark.integration
def test_evaluate_missing_bundle_returns_error() -> None:
    """Evaluating against a non-existent artifacts dir returns exit code 1."""
    assert evaluate_command(artifacts_dir="/tmp/__lcd_no_such_dir__", data=None) == 1


@pytest.mark.integration
@pytest.mark.slow
def test_cli_commands_run_via_runner(tmp_path: Path) -> None:
    """The registered Typer commands run end-to-end through the CliRunner.

    Exercises the inner command wrappers (`train`/`score`/`evaluate`) and their
    `typer.Exit` codes, not just the underlying ``*_command`` functions.
    """
    runner = _runner()
    app = build_app()
    out_dir = tmp_path / "artifacts"

    train_result = runner.invoke(app, ["train", "--out-dir", str(out_dir), "--seed", "5"])
    assert train_result.exit_code == 0, train_result.stdout

    score_result = runner.invoke(app, ["score", "--artifacts-dir", str(out_dir)])
    assert score_result.exit_code == 0, score_result.stdout
    assert "calibrated PD" in score_result.stdout

    eval_result = runner.invoke(app, ["evaluate", "--artifacts-dir", str(out_dir)])
    assert eval_result.exit_code == 0, eval_result.stdout
    assert "ROC-AUC" in eval_result.stdout


@pytest.mark.integration
def test_app_entry_point_with_no_args_shows_help(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The console-script `app()` shows help and exits cleanly with no args.

    `no_args_is_help=True` makes a bare invocation print the help screen and exit
    with Click's usage-shown code (2). SystemExit is what Click/Typer raises to
    carry the exit code; argv is pinned to just the program name so pytest's own
    argv is not parsed as commands.
    """
    monkeypatch.setattr(sys, "argv", ["lendingclub-default"])
    with pytest.raises(SystemExit) as excinfo:
        app()
    assert excinfo.value.code == 2


@pytest.mark.integration
def test_cli_module_import_is_typer_free() -> None:
    """Importing lendingclub_default.cli must not pull Typer onto sys.path."""
    code = (
        "import sys; import lendingclub_default.cli; "
        "assert 'typer' not in sys.modules; print('pure')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "pure" in result.stdout
