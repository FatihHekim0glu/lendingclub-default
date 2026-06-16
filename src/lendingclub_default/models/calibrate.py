"""Probability calibration so model output is a usable PD in ``[0, 1]``.

A raw booster score ranks risk well but is not a trustworthy probability.
:class:`CalibratedModel` wraps a fitted scorer with an isotonic (default) or
Platt/sigmoid calibration map fit on a held-out calibration fold, so the output
is a calibrated probability of default. Two invariants are property-tested: the
output stays in ``[0, 1]``, and the isotonic map is monotone non-decreasing in
the raw score.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

# quantcore-candidate: new code (PD calibration); parity vs sklearn
# CalibratedClassifierCV / IsotonicRegression.


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    """Immutable summary of a fitted calibration map.

    Attributes
    ----------
    method:
        ``"isotonic"`` or ``"sigmoid"`` (Platt).
    brier_before:
        Brier score of the raw scores on the calibration fold.
    brier_after:
        Brier score of the calibrated probabilities on the calibration fold
        (should be <= ``brier_before``).
    n_calibration:
        Number of rows in the calibration fold.
    """

    method: str
    brier_before: float
    brier_after: float
    n_calibration: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        return {
            "method": self.method,
            "brier_before": float(self.brier_before),
            "brier_after": float(self.brier_after),
            "n_calibration": int(self.n_calibration),
            "meta": dict(self.meta),
        }


@dataclass(frozen=True, slots=True)
class CalibratedModel:
    """A fitted scorer plus a monotone calibration map producing a PD in ``[0, 1]``.

    Attributes
    ----------
    method:
        The calibration method used (``"isotonic"`` / ``"sigmoid"``).
    params:
        Serialized calibration-map parameters (isotonic knots or Platt
        coefficients) sufficient to reproduce :meth:`calibrate` without sklearn.
    """

    method: str
    params: dict[str, Any] = field(default_factory=dict)

    def calibrate(self, raw_scores: np.ndarray) -> np.ndarray:
        """Map raw scores to calibrated probabilities in ``[0, 1]``.

        Parameters
        ----------
        raw_scores:
            Uncalibrated scores from the underlying model.

        Returns
        -------
        numpy.ndarray
            Calibrated probabilities, clipped to ``[0, 1]`` and monotone non-
            decreasing in ``raw_scores`` for the isotonic method.

        Raises
        ------
        NotImplementedError
            This is a stub; the implementation is filled in by the models author.
        """
        raise NotImplementedError("CalibratedModel.calibrate is not yet implemented.")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this calibration map."""
        return {"method": self.method, "params": dict(self.params)}


def fit_calibration(
    raw_scores: np.ndarray,
    y: pd.Series,
    *,
    method: str = "isotonic",
) -> tuple[CalibratedModel, CalibrationResult]:
    """Fit a calibration map on a held-out calibration fold.

    Parameters
    ----------
    raw_scores:
        Uncalibrated model scores on the calibration fold.
    y:
        The aligned binary target on the calibration fold.
    method:
        ``"isotonic"`` (default) or ``"sigmoid"`` (Platt scaling).

    Returns
    -------
    tuple[CalibratedModel, CalibrationResult]
        The fitted calibration map and a before/after Brier summary.

    Raises
    ------
    ValidationError
        If ``method`` is unknown or the inputs are misaligned.
    NotImplementedError
        This is a stub; the implementation is filled in by the models author.
    """
    raise NotImplementedError("fit_calibration is not yet implemented.")
