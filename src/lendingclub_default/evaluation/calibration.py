"""Reliability-curve points for assessing probability calibration.

Bins predicted probabilities and, for each bin, reports the mean predicted PD
against the observed default frequency. A perfectly calibrated model lies on the
diagonal. :func:`reliability_curve` returns the points that the plot layer
(:func:`lendingclub_default.plots.calibration_figure`) renders, plus the
expected calibration error.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

# quantcore-candidate: new code (reliability curve); parity vs sklearn
# calibration_curve.


@dataclass(frozen=True, slots=True)
class ReliabilityCurve:
    """Immutable reliability-curve points and summary.

    Attributes
    ----------
    mean_predicted:
        Per-bin mean predicted probability (x-axis of the reliability plot).
    observed_frequency:
        Per-bin observed default frequency (y-axis).
    bin_counts:
        Number of observations in each bin.
    expected_calibration_error:
        Count-weighted mean absolute gap between predicted and observed (ECE).
    n_bins:
        Number of (non-empty) bins.
    """

    mean_predicted: list[float]
    observed_frequency: list[float]
    bin_counts: list[int]
    expected_calibration_error: float
    n_bins: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the curve."""
        out = asdict(self)
        out["expected_calibration_error"] = float(self.expected_calibration_error)
        return out


def reliability_curve(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    *,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> ReliabilityCurve:
    """Compute reliability-curve points and the expected calibration error.

    Parameters
    ----------
    y_true:
        Binary labels (1 = default).
    y_prob:
        Calibrated probabilities in ``[0, 1]``.
    n_bins:
        Number of probability bins.
    strategy:
        ``"quantile"`` (equal-count bins) or ``"uniform"`` (equal-width bins).

    Returns
    -------
    ReliabilityCurve
        Per-bin predicted/observed points, counts, and the ECE.

    Raises
    ------
    NotImplementedError
        This is a stub; the implementation is filled in by the evaluation author.
    """
    raise NotImplementedError("reliability_curve is not yet implemented.")
