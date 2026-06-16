"""DeLong test for the difference between two correlated ROC-AUCs.

The credit analogue of HRP's Deflated Sharpe: with no Sharpe to deflate, the
overfitting guard is a significance test on the AUC *gap* between the headline
XGBoost and the L2-logistic baseline, evaluated on the SAME held-out vintages
(hence correlated). :func:`delong_auc_test` implements the fast DeLong (1988)
covariance estimator and returns the AUC difference, its standard error, a
z-statistic, and a (optionally Bonferroni-adjusted) two-sided p-value.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

# quantcore-candidate: new code (DeLong AUC test); reference = DeLong et al.
# (1988), Sun & Xu fast covariance.


@dataclass(frozen=True, slots=True)
class DeLongResult:
    """Immutable result of a DeLong AUC-difference test.

    Attributes
    ----------
    auc_a:
        ROC-AUC of model A (the headline XGBoost).
    auc_b:
        ROC-AUC of model B (the logistic baseline).
    auc_diff:
        ``auc_a - auc_b``.
    std_error:
        Standard error of the AUC difference (DeLong covariance).
    z:
        The z-statistic ``auc_diff / std_error``.
    p_value:
        Two-sided p-value (Bonferroni-adjusted if ``n_comparisons > 1``).
    n_comparisons:
        Number of comparisons the p-value was corrected for.
    """

    auc_a: float
    auc_b: float
    auc_diff: float
    std_error: float
    z: float
    p_value: float
    n_comparisons: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the result."""
        out = asdict(self)
        return {k: (float(v) if k != "n_comparisons" else int(v)) for k, v in out.items()}


def delong_auc_test(
    y_true: np.ndarray | pd.Series,
    score_a: np.ndarray | pd.Series,
    score_b: np.ndarray | pd.Series,
    *,
    n_comparisons: int = 1,
) -> DeLongResult:
    """Test whether two correlated ROC-AUCs differ (DeLong, 1988).

    Both score vectors are evaluated against the SAME labels on the SAME held-out
    rows, so the AUC estimates are correlated; DeLong's covariance estimator
    accounts for that correlation.

    Parameters
    ----------
    y_true:
        Binary labels (1 = default), shared by both models.
    score_a:
        Scores from model A (XGBoost).
    score_b:
        Scores from model B (logistic baseline).
    n_comparisons:
        If ``> 1``, Bonferroni-adjust the p-value for that many comparisons.

    Returns
    -------
    DeLongResult
        The two AUCs, their difference, the standard error, z, and p-value.

    Raises
    ------
    ValidationError
        If the inputs are misaligned or a class is absent.
    NotImplementedError
        This is a stub; the implementation is filled in by the evaluation author.
    """
    raise NotImplementedError("delong_auc_test is not yet implemented.")
