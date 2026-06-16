"""Import / public-API smoke tests and fixture sanity checks.

These run today (before any compute is implemented) so the scaffold is provably
importable and the seeded fixtures are usable. The substantive unit tests (label
construction, leakage-drop completeness) are added by the data author against the
filled-in implementations.
"""

from __future__ import annotations

import pandas as pd
import pytest

import lendingclub_default as lcd
from lendingclub_default.data.leakage import LEAKAGE_COLS


@pytest.mark.unit
def test_package_imports_cleanly() -> None:
    """The top-level package imports and exposes a non-trivial public API."""
    assert lcd.__version__ == "0.1.0"
    assert "drop_leakage" in lcd.__all__
    assert "train" in lcd.__all__
    assert "delong_auc_test" in lcd.__all__


@pytest.mark.unit
def test_leakage_cols_is_frozen_and_populated() -> None:
    """LEAKAGE_COLS is a populated, frozen reference set of post-funding columns."""
    assert isinstance(LEAKAGE_COLS, frozenset)
    assert len(LEAKAGE_COLS) >= 30
    # A few canonical post-funding columns must be present.
    for col in ("recoveries", "total_pymnt", "out_prncp", "debt_settlement_flag"):
        assert col in LEAKAGE_COLS


@pytest.mark.unit
def test_synthetic_panel_fixture_is_usable(synthetic_panel: pd.DataFrame) -> None:
    """The conftest synthetic_panel is a usable LC-schema frame with the right shape."""
    assert isinstance(synthetic_panel, pd.DataFrame)
    assert synthetic_panel.shape[0] > 1_000
    # Application-time, outcome, and leakage columns are all present.
    for col in ("loan_amnt", "int_rate", "grade", "issue_d", "loan_status"):
        assert col in synthetic_panel.columns
    for col in ("recoveries", "total_pymnt", "out_prncp"):
        assert col in synthetic_panel.columns


@pytest.mark.unit
def test_synthetic_panel_default_rate_is_realistic(synthetic_panel: pd.DataFrame) -> None:
    """The synthetic default rate sits in a realistic LC band (~15%)."""
    rate = (synthetic_panel["loan_status"] == "Charged Off").mean()
    assert 0.08 <= rate <= 0.25


@pytest.mark.unit
def test_synthetic_panel_leakage_correlates_with_outcome(
    synthetic_panel: pd.DataFrame,
) -> None:
    """Leakage columns separate the outcome (so the leakage trap is real)."""
    defaulted = synthetic_panel["loan_status"] == "Charged Off"
    # Recoveries are (near-)zero for repaid loans, positive for charged-off ones.
    assert synthetic_panel.loc[~defaulted, "recoveries"].mean() < (
        synthetic_panel.loc[defaulted, "recoveries"].mean()
    )
