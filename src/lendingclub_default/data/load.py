"""Data loading: real Kaggle CSV when given, else the synthetic generator.

:func:`load_panel` is the single entry point used by training and the CLI. When a
path to the real LendingClub accepted-loans CSV is supplied it reads and dtype-
coerces it; when omitted it falls back to :func:`lendingclub_default.data.synthetic.generate_synthetic_panel`.
The same downstream pipeline (leakage drop -> labels -> temporal split ->
calibration) runs on either source, so the synthetic path is a faithful stand-in.

Importing this module has no side effects (no file is read at import time).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from lendingclub_default.data.synthetic import SyntheticConfig

# quantcore-candidate: new code (LC CSV loader); dtype coercion via _validation.

#: The application-time columns the loader guarantees to surface (origination
#: features the model is allowed to use). Used both to validate a real CSV and to
#: document the contract the synthetic generator must satisfy.
APPLICATION_COLUMNS: Final[tuple[str, ...]] = (
    "loan_amnt",
    "term",
    "int_rate",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "revol_util",
    "open_acc",
    "pub_rec",
    "purpose",
    "addr_state",
    "installment",
    "verification_status",
    "issue_d",
)


def load_panel(
    data_path: Path | str | None = None,
    *,
    config: SyntheticConfig | None = None,
) -> pd.DataFrame:
    """Load a loan-application panel from a CSV path, or generate one synthetically.

    Parameters
    ----------
    data_path:
        Path to the real LendingClub accepted-loans CSV. If ``None``, the
        synthetic generator is used instead (the reproducible default).
    config:
        Synthetic-generator configuration, used only when ``data_path is None``.

    Returns
    -------
    pandas.DataFrame
        The loaded (or generated) panel with at least :data:`APPLICATION_COLUMNS`
        plus ``loan_status``; a real CSV additionally carries post-funding columns
        that are dropped downstream by
        :func:`lendingclub_default.data.leakage.drop_leakage`.

    Raises
    ------
    ValidationError
        If a supplied CSV is missing required application-time columns.
    NotImplementedError
        This is a stub; the implementation is filled in by the data author.
    """
    raise NotImplementedError("load_panel is not yet implemented.")


def coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce raw LC columns to model-ready dtypes (a copy; input untouched).

    Parses percentage strings (``int_rate``, ``revol_util`` such as ``"13.5%"``)
    to floats, ``term`` to its canonical ``"36 months"``/``"60 months"`` string,
    ``emp_length`` to a numeric year count, and ``issue_d`` to a sortable
    period/datetime used by the temporal split.

    Parameters
    ----------
    df:
        The raw panel as read from CSV (or produced synthetically).

    Returns
    -------
    pandas.DataFrame
        A copy with coerced dtypes.

    Raises
    ------
    NotImplementedError
        This is a stub; the implementation is filled in by the data author.
    """
    raise NotImplementedError("coerce_dtypes is not yet implemented.")
