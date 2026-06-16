"""Shared, seeded test fixtures.

Every fixture is deterministic (driven by
:func:`lendingclub_default._rng.make_rng`) and returns pandas objects, so tests
across the suite share identical synthetic data with known structure:

- ``synthetic_panel`` — a small LendingClub-schema panel carrying application-time
  columns, ``issue_d`` vintages spread across 2015-2018, post-funding LEAKAGE
  columns, and a ``loan_status`` outcome with a ~15% default rate. The default
  probability is a noisy monotone function of fico / dti / int_rate / grade, so a
  leakage-free model lands at a believable AUC (not 0.5, not 0.99). Built here
  directly (not via the stubbed generator) so it is usable from day one.
- ``k_vintage_fixture`` — the same panel plus the list of its ordered unique
  vintages, for exercising the temporal split.
- ``schema_with_leakage`` — the explicit set of leakage columns present in the
  panel, for the no-leakage drop/property tests.

Importing this module has no side effects beyond fixture registration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lendingclub_default._rng import make_rng

_SEED = 20260616

#: Ordered ``issue_d`` vintage cohort labels (2015-2018), oldest first.
_VINTAGES: tuple[str, ...] = (
    "2015-Q1",
    "2015-Q3",
    "2016-Q1",
    "2016-Q3",
    "2017-Q1",
    "2017-Q3",
    "2018-Q1",
    "2018-Q3",
)

#: Post-funding columns deliberately injected into the panel so the leakage
#: allowlist is genuinely exercised (a representative subset of LEAKAGE_COLS).
_LEAKAGE_COLUMNS: tuple[str, ...] = (
    "recoveries",
    "total_pymnt",
    "total_pymnt_inv",
    "out_prncp",
    "out_prncp_inv",
    "last_pymnt_amnt",
    "collection_recovery_fee",
    "total_rec_prncp",
    "total_rec_int",
    "debt_settlement_flag",
    "funded_amnt",
)

_GRADES: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G")
_PURPOSES: tuple[str, ...] = (
    "debt_consolidation",
    "credit_card",
    "home_improvement",
    "major_purchase",
    "small_business",
    "car",
    "medical",
    "other",
)
_STATES: tuple[str, ...] = ("CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI")
_HOME: tuple[str, ...] = ("RENT", "MORTGAGE", "OWN")
_VERIFICATION: tuple[str, ...] = ("Verified", "Source Verified", "Not Verified")


def _build_synthetic_panel(n_loans: int = 2_400, seed: int = _SEED) -> pd.DataFrame:
    """Construct a deterministic LC-schema panel with a ~15% default rate.

    The latent default log-odds is a monotone function of fico (down), dti (up),
    int_rate (up), and grade (worse grade -> up), plus idiosyncratic noise and a
    mild per-vintage regime drift. The intercept is tuned so the marginal default
    rate is ~15%.
    """
    gen = make_rng(seed)

    # Vintage assignment (roughly balanced, oldest -> newest).
    vintage_idx = gen.integers(0, len(_VINTAGES), size=n_loans)
    issue_d = np.array([_VINTAGES[i] for i in vintage_idx], dtype=object)

    # Grade as an ordinal 0..6 (A..G); worse grade is riskier.
    grade_ord = gen.integers(0, len(_GRADES), size=n_loans)
    grade = np.array([_GRADES[i] for i in grade_ord], dtype=object)
    sub_grade = np.array([f"{_GRADES[g]}{gen.integers(1, 6)}" for g in grade_ord], dtype=object)

    # Application-time numeric features (loosely realistic ranges).
    loan_amnt = gen.uniform(1_000, 35_000, size=n_loans).round(-2)
    term = np.where(gen.random(n_loans) < 0.7, "36 months", "60 months").astype(object)
    int_rate = (5.0 + grade_ord * 3.0 + gen.normal(0, 1.5, size=n_loans)).clip(5, 30)
    annual_inc = np.exp(gen.normal(11.0, 0.5, size=n_loans)).clip(15_000, 300_000)
    dti = gen.uniform(0, 40, size=n_loans)
    fico_low = gen.uniform(640, 820, size=n_loans).round()
    fico_high = fico_low + 4.0
    revol_util = gen.uniform(0, 100, size=n_loans)
    open_acc = gen.integers(2, 30, size=n_loans)
    pub_rec = gen.integers(0, 3, size=n_loans)
    emp_length = gen.integers(0, 11, size=n_loans)
    monthly_rate = int_rate / 100.0 / 12.0
    n_pay = np.where(term == "36 months", 36, 60)
    installment = (loan_amnt * monthly_rate / (1 - (1 + monthly_rate) ** (-n_pay))).round(2)

    # Latent default log-odds: noisy monotone in the risk drivers.
    z = (
        -2.6
        + 0.16 * (int_rate - 12.0)
        + 0.05 * (dti - 18.0)
        - 0.015 * (fico_low - 700.0)
        + 0.18 * grade_ord
        + 0.05 * vintage_idx  # mild regime drift: later vintages slightly riskier
        + gen.normal(0, 1.0, size=n_loans)
    )
    p_default = 1.0 / (1.0 + np.exp(-z))
    default = (gen.random(n_loans) < p_default).astype(int)
    loan_status = np.where(default == 1, "Charged Off", "Fully Paid").astype(object)

    # Inject post-funding LEAKAGE columns correlated with the outcome.
    total_pymnt = np.where(default == 1, loan_amnt * 0.4, loan_amnt * 1.1)
    recoveries = np.where(default == 1, loan_amnt * 0.1, 0.0)
    out_prncp = np.where(default == 1, loan_amnt * 0.5, 0.0)

    frame = pd.DataFrame(
        {
            # --- application-time features --------------------------------- #
            "loan_amnt": loan_amnt,
            "term": term,
            "int_rate": int_rate.round(2),
            "grade": grade,
            "sub_grade": sub_grade,
            "emp_length": emp_length,
            "home_ownership": [_HOME[i] for i in gen.integers(0, len(_HOME), size=n_loans)],
            "annual_inc": annual_inc.round(2),
            "dti": dti.round(2),
            "fico_range_low": fico_low,
            "fico_range_high": fico_high,
            "revol_util": revol_util.round(2),
            "open_acc": open_acc,
            "pub_rec": pub_rec,
            "purpose": [_PURPOSES[i] for i in gen.integers(0, len(_PURPOSES), size=n_loans)],
            "addr_state": [_STATES[i] for i in gen.integers(0, len(_STATES), size=n_loans)],
            "installment": installment,
            "verification_status": [
                _VERIFICATION[i] for i in gen.integers(0, len(_VERIFICATION), size=n_loans)
            ],
            "issue_d": issue_d,
            # --- outcome --------------------------------------------------- #
            "loan_status": loan_status,
            # --- post-funding LEAKAGE columns ------------------------------ #
            "recoveries": recoveries.round(2),
            "total_pymnt": total_pymnt.round(2),
            "total_pymnt_inv": (total_pymnt * 0.99).round(2),
            "out_prncp": out_prncp.round(2),
            "out_prncp_inv": (out_prncp * 0.99).round(2),
            "last_pymnt_amnt": (installment * gen.uniform(0.5, 1.5, n_loans)).round(2),
            "collection_recovery_fee": (recoveries * 0.1).round(2),
            "total_rec_prncp": (loan_amnt - out_prncp).round(2),
            "total_rec_int": (total_pymnt * 0.15).round(2),
            "debt_settlement_flag": np.where(default == 1, "Y", "N").astype(object),
            "funded_amnt": loan_amnt,
        }
    )
    return frame


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded PCG64 generator shared by tests that need raw randomness."""
    return make_rng(_SEED)


@pytest.fixture
def synthetic_panel() -> pd.DataFrame:
    """A small LC-schema panel with leakage cols, vintages, and a ~15% default rate.

    Shape roughly ``(2400, 31)``. Carries application-time features, ``issue_d``
    vintages across 2015-2018, post-funding leakage columns, and a ``loan_status``
    outcome whose default probability is a noisy monotone function of the risk
    drivers (so a leakage-free model achieves a believable AUC).
    """
    return _build_synthetic_panel()


@pytest.fixture
def k_vintage_fixture(synthetic_panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """The synthetic panel paired with its ordered unique vintages.

    Returns ``(panel, vintages)`` where ``vintages`` is the sorted list of unique
    ``issue_d`` cohort labels present, for exercising the temporal vintage split.
    """
    vintages = [v for v in _VINTAGES if v in set(synthetic_panel["issue_d"])]
    return synthetic_panel, vintages


@pytest.fixture
def schema_with_leakage() -> tuple[str, ...]:
    """The explicit set of post-funding leakage columns present in the panel.

    Used by the no-leakage drop/property tests to assert that none of these
    survive the preprocessing pipeline.
    """
    return _LEAKAGE_COLUMNS
