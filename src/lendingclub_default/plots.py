"""Plotly figure builders (lazy plotly import).

Each builder returns a ``plotly.graph_objects.Figure`` that serializes to a
``{data, layout}`` JSON blob the API and frontend render. Plotly is imported
LAZILY inside each function so importing this module stays pure (no plotly on the
import path of the compute library).

Figures:

- :func:`roc_figure` — ROC curve (with the 0.5 diagonal and the AUC in the title).
- :func:`pr_figure` — precision-recall curve (with the base-rate floor line).
- :func:`calibration_figure` — reliability curve vs. the perfect-calibration
  diagonal.
- :func:`score_distribution_figure` — predicted-PD histograms split by realised
  outcome (the separation the KS statistic summarizes).

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    from lendingclub_default.evaluation.calibration import ReliabilityCurve

# quantcore-candidate: new code (Plotly builders); lazy plotly import.


def roc_figure(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    *,
    title: str = "ROC curve",
) -> go.Figure:
    """Build a ROC-curve figure with the no-skill diagonal and AUC annotation.

    Parameters
    ----------
    y_true:
        Binary labels (1 = default).
    y_score:
        Predicted scores.
    title:
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        The ROC figure.

    Raises
    ------
    NotImplementedError
        This is a stub; the implementation is filled in by the plots author.
    """
    raise NotImplementedError("roc_figure is not yet implemented.")


def pr_figure(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray | pd.Series,
    *,
    title: str = "Precision-Recall curve",
) -> go.Figure:
    """Build a precision-recall figure with the base-rate (no-skill) floor.

    Parameters
    ----------
    y_true:
        Binary labels (1 = default).
    y_score:
        Predicted scores.
    title:
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        The PR figure.

    Raises
    ------
    NotImplementedError
        This is a stub; the implementation is filled in by the plots author.
    """
    raise NotImplementedError("pr_figure is not yet implemented.")


def calibration_figure(
    curve: ReliabilityCurve,
    *,
    title: str = "Calibration (reliability) curve",
) -> go.Figure:
    """Build a reliability-curve figure vs. the perfect-calibration diagonal.

    Parameters
    ----------
    curve:
        A computed :class:`lendingclub_default.evaluation.calibration.ReliabilityCurve`.
    title:
        Figure title.

    Returns
    -------
    plotly.graph_objects.Figure
        The calibration figure.

    Raises
    ------
    NotImplementedError
        This is a stub; the implementation is filled in by the plots author.
    """
    raise NotImplementedError("calibration_figure is not yet implemented.")


def score_distribution_figure(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    *,
    title: str = "Predicted PD by outcome",
    n_bins: int = 40,
) -> go.Figure:
    """Build overlaid predicted-PD histograms split by realised outcome.

    Parameters
    ----------
    y_true:
        Binary labels (1 = default).
    y_prob:
        Calibrated probabilities in ``[0, 1]``.
    title:
        Figure title.
    n_bins:
        Histogram bin count.

    Returns
    -------
    plotly.graph_objects.Figure
        The score-distribution figure.

    Raises
    ------
    NotImplementedError
        This is a stub; the implementation is filled in by the plots author.
    """
    raise NotImplementedError("score_distribution_figure is not yet implemented.")


def figure_to_dict(figure: go.Figure) -> dict[str, Any]:
    """Serialize a Plotly figure to a ``{data, layout}`` JSON-safe ``dict``.

    Mirrors the API contract ``json.loads(pio.to_json(fig, validate=False))`` so
    the same payload shape is produced in the library and the router.

    Parameters
    ----------
    figure:
        The Plotly figure to serialize.

    Returns
    -------
    dict[str, Any]
        A ``{data, layout}`` mapping.

    Raises
    ------
    NotImplementedError
        This is a stub; the implementation is filled in by the plots author.
    """
    raise NotImplementedError("figure_to_dict is not yet implemented.")
