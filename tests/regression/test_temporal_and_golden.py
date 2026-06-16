"""Regression guards: temporal-order and golden synthetic metrics.

Filled in against the real implementations:

- golden ROC-AUC / Brier on a frozen synthetic vintage fixture (pinned values);
- temporal-order guard: no train ``issue_d`` is after any test ``issue_d``.

While the split/training kernels are stubs, this asserts the guard entry points
exist and that the seeded ``k_vintage_fixture`` exposes ordered vintages to split.
"""

from __future__ import annotations

import pandas as pd
import pytest

from lendingclub_default.data.split import assert_temporal_order, temporal_split


@pytest.mark.regression
def test_k_vintage_fixture_has_ordered_vintages(
    k_vintage_fixture: tuple[pd.DataFrame, list[str]],
) -> None:
    """The vintage fixture exposes >= 2 ordered cohorts (a split is possible)."""
    panel, vintages = k_vintage_fixture
    assert len(vintages) >= 2
    assert vintages == sorted(vintages)
    assert set(panel["issue_d"]).issubset(set(vintages))


@pytest.mark.regression
def test_temporal_split_entry_points_exist(
    k_vintage_fixture: tuple[pd.DataFrame, list[str]],
) -> None:
    """The temporal-split and order-guard entry points exist (raise until implemented)."""
    panel, _ = k_vintage_fixture
    with pytest.raises(NotImplementedError):
        temporal_split(panel)
    with pytest.raises(NotImplementedError):
        assert_temporal_order(panel, panel)
