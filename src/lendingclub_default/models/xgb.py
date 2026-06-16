"""The headline XGBoost default classifier.

A binary-logistic XGBoost booster with ``scale_pos_weight`` set for the ~15%
default imbalance and early stopping on a temporal validation fold (a slice of
the latest train vintages, never the held-out test). The shipped artifact is a
``<2MB`` booster JSON trained on the synthetic panel; the API loads it LAZILY at
first call via a module-level ``_BOOSTER=None`` sentinel — there is NO training
at import or per request.

Importing this module has no side effects (xgboost is imported lazily inside the
functions, and the sentinel stays ``None`` until first use).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    import xgboost as xgb

# quantcore-candidate: new code (XGB credit classifier); lazy booster load.

#: Module-level lazy-load sentinel for the shipped booster. The API and
#: :func:`load_booster` populate it on first call; it is NEVER trained at import.
_BOOSTER: xgb.Booster | None = None


@dataclass(frozen=True, slots=True)
class XGBConfig:
    """Immutable XGBoost hyperparameters (recorded in the RunManifest).

    Attributes
    ----------
    n_estimators:
        Maximum boosting rounds (early stopping may use fewer).
    max_depth:
        Tree depth.
    learning_rate:
        Boosting step size (``eta``).
    subsample, colsample_bytree:
        Row / column subsampling fractions for variance reduction.
    reg_lambda:
        L2 leaf regularization.
    early_stopping_rounds:
        Rounds without validation-AUC improvement before stopping.
    seed:
        Booster random seed.
    """

    n_estimators: int = 400
    max_depth: int = 4
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 30
    seed: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this config."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "early_stopping_rounds": self.early_stopping_rounds,
            "seed": self.seed,
            "meta": dict(self.meta),
        }


def fit_xgb(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_valid: pd.DataFrame,
    y_valid: pd.Series,
    *,
    config: XGBConfig | None = None,
) -> xgb.Booster:
    """Fit a binary-logistic XGBoost booster with imbalance weighting + early stopping.

    Parameters
    ----------
    x_train, y_train:
        The training design matrix and binary target.
    x_valid, y_valid:
        A *temporal* validation fold (later train vintages) for early stopping —
        NEVER the held-out test set.
    config:
        Hyperparameters; defaults to :class:`XGBConfig`.

    Returns
    -------
    xgboost.Booster
        The fitted booster (``scale_pos_weight`` derived from the train imbalance).

    Raises
    ------
    NotImplementedError
        This is a stub; the implementation is filled in by the models author.
    """
    raise NotImplementedError("fit_xgb is not yet implemented.")


def predict_proba(booster: xgb.Booster, x: pd.DataFrame) -> np.ndarray:
    """Return raw (uncalibrated) default probabilities from a fitted booster.

    Parameters
    ----------
    booster:
        A fitted XGBoost booster.
    x:
        The feature matrix to score.

    Returns
    -------
    numpy.ndarray
        A ``(n_rows,)`` array of probabilities in ``[0, 1]``.

    Raises
    ------
    NotImplementedError
        This is a stub; the implementation is filled in by the models author.
    """
    raise NotImplementedError("predict_proba is not yet implemented.")


def save_booster(booster: xgb.Booster, path: Path | str) -> None:
    """Serialize a booster to the committed ``<2MB`` JSON artifact.

    Parameters
    ----------
    booster:
        The fitted booster to persist.
    path:
        Destination path for the booster JSON.

    Raises
    ------
    NotImplementedError
        This is a stub; the implementation is filled in by the models author.
    """
    raise NotImplementedError("save_booster is not yet implemented.")


def load_booster(path: Path | str, *, use_cache: bool = True) -> xgb.Booster:
    """Lazily load (and optionally cache) the shipped booster JSON.

    Populates the module-level ``_BOOSTER`` sentinel on first call so the API can
    reuse a single in-process booster without retraining.

    Parameters
    ----------
    path:
        Path to the committed booster JSON.
    use_cache:
        If ``True``, reuse / populate the module-level ``_BOOSTER`` sentinel.

    Returns
    -------
    xgboost.Booster
        The loaded booster.

    Raises
    ------
    ArtifactError
        If the artifact is missing or cannot be parsed.
    NotImplementedError
        This is a stub; the implementation is filled in by the models author.
    """
    raise NotImplementedError("load_booster is not yet implemented.")
