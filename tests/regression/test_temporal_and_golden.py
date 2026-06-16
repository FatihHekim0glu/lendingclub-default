"""Regression guards: temporal-order (no look-ahead) and golden synthetic metrics.

Two pinned-behaviour guards against the real implementations:

- **temporal-order guard** — a vintage split trains on past vintages and tests on
  strictly later ones, so no train ``issue_d`` is ever after any test ``issue_d``
  (the no-look-ahead invariant), and ``assert_temporal_order`` fires when that is
  violated;
- **golden metrics** — a leakage-free model on the frozen synthetic vintage
  fixture lands in a believable, locked band (ROC-AUC beats the 0.5 floor but is
  not a fraudulent ~0.99; Brier is a sane probability), pinning the honest
  headline against silent regressions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from lendingclub_default._exceptions import TemporalSplitError
from lendingclub_default.data.split import assert_temporal_order, temporal_split
from lendingclub_default.evaluation.metrics import brier_score, roc_auc

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
def test_temporal_split_has_no_lookahead(
    k_vintage_fixture: tuple[pd.DataFrame, list[str]],
) -> None:
    """No train row's ``issue_d`` is after any test row's (the no-look-ahead law).

    The headline temporal guard: ``temporal_split`` partitions by vintage so the
    max train ``issue_d`` never exceeds the min test ``issue_d``, train vintages
    are strictly earlier than test vintages, and the folds are disjoint and total.
    """
    panel, _ = k_vintage_fixture
    split = temporal_split(panel, issue_col="issue_d", test_size=0.25)

    train = panel.loc[split.train_idx]
    test = panel.loc[split.test_idx]

    # The core no-look-ahead invariant: max(train issue_d) <= min(test issue_d).
    assert train["issue_d"].max() <= test["issue_d"].min()
    # Vintage cohorts are strictly ordered: every train vintage precedes every test one.
    assert max(split.train_vintages) < min(split.test_vintages)
    # Folds partition the panel exactly once (disjoint and total).
    assert set(split.train_idx).isdisjoint(split.test_idx)
    assert len(split.train_idx) + len(split.test_idx) == panel.shape[0]
    # The hard backstop accepts a correctly ordered split silently.
    assert_temporal_order(train, test, issue_col="issue_d")


@pytest.mark.regression
def test_assert_temporal_order_detects_lookahead(
    k_vintage_fixture: tuple[pd.DataFrame, list[str]],
) -> None:
    """Swapping the folds (train on the future) trips the look-ahead backstop."""
    panel, _ = k_vintage_fixture
    split = temporal_split(panel, issue_col="issue_d", test_size=0.25)
    train = panel.loc[split.train_idx]
    test = panel.loc[split.test_idx]
    # Train on the LATER fold and test on the EARLIER one -> look-ahead -> raise.
    with pytest.raises(TemporalSplitError):
        assert_temporal_order(test, train, issue_col="issue_d")


@pytest.mark.regression
def test_golden_leakage_free_metrics_are_in_band(
    synthetic_panel: pd.DataFrame,
) -> None:
    """A leakage-free model on the frozen fixture lands in a pinned, honest band.

    Trains an L2-logistic on application-time numeric features only (no leakage
    column ever enters), out-of-time by vintage, and asserts the held-out ROC-AUC
    and Brier sit in a locked, believable range: it beats the 0.5 base-rate floor,
    is nowhere near a fraudulent ~0.99, and the Brier is a sane probability score.
    """
    vintages = sorted(synthetic_panel["issue_d"].unique())
    cutoff = vintages[len(vintages) // 2]
    is_train = synthetic_panel["issue_d"] <= cutoff

    x = synthetic_panel[list(_NUMERIC_FEATURES)].astype("float64")
    y = (synthetic_panel["loan_status"] == "Charged Off").astype(int)

    # Standardize on TRAIN statistics only (the pipeline scales numerics for the
    # logistic baseline); this also lets lbfgs converge cleanly.
    scaler = StandardScaler().fit(x[is_train].to_numpy())
    x_train = scaler.transform(x[is_train].to_numpy())
    x_test = scaler.transform(x[~is_train].to_numpy())

    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
    model.fit(x_train, y[is_train].to_numpy())
    proba = np.asarray(model.predict_proba(x_test)[:, 1], dtype="float64")
    y_test = y[~is_train].to_numpy(dtype="float64")

    auc = roc_auc(y_test, proba)
    brier = brier_score(y_test, proba)

    # Golden band: honestly above the 0.5 floor, honestly below a leakage-grade AUC.
    assert 0.60 <= auc <= 0.85
    # Brier of a calibrated-ish PD over a ~15% base rate stays small but non-trivial.
    assert 0.05 <= brier <= 0.25
