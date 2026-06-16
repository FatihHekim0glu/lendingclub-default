"""Application-time feature pipeline (fit on the train fold only).

Builds an sklearn :class:`~sklearn.pipeline.Pipeline` /
:class:`~sklearn.compose.ColumnTransformer` over the origination-time columns
that survive :func:`lendingclub_default.data.leakage.drop_leakage`:

- numeric columns: median imputation + (optional) standardization;
- low-cardinality categoricals (``grade``, ``home_ownership``, ``term``,
  ``verification_status``): most-frequent imputation + one-hot encoding;
- high-cardinality categoricals (``purpose``, ``addr_state``, ``sub_grade``):
  **out-of-fold** target encoding so the encoding for a row never uses that
  row's own label.

Every transformer is fit on the TRAIN fold only; the same fitted object then
transforms the held-out later vintages. A property test asserts that later-
vintage rows cannot influence train-fold transform statistics.

Importing this module has no side effects (sklearn objects are constructed only
when :func:`build_feature_pipeline` is called).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    from sklearn.pipeline import Pipeline

# quantcore-candidate: new code (LC feature pipeline); fit-on-train-only,
# out-of-fold target encoding for high-card categoricals.

#: Numeric application-time columns passed through imputation + scaling.
NUMERIC_FEATURES: tuple[str, ...] = (
    "loan_amnt",
    "int_rate",
    "annual_inc",
    "dti",
    "fico_range_low",
    "fico_range_high",
    "revol_util",
    "open_acc",
    "pub_rec",
    "installment",
    "emp_length",
)

#: Low-cardinality categoricals -> one-hot.
LOW_CARD_FEATURES: tuple[str, ...] = (
    "term",
    "grade",
    "home_ownership",
    "verification_status",
)

#: High-cardinality categoricals -> out-of-fold target encoding.
HIGH_CARD_FEATURES: tuple[str, ...] = (
    "sub_grade",
    "purpose",
    "addr_state",
)


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Immutable description of the columns each transformer branch handles.

    Attributes
    ----------
    numeric:
        Numeric feature names (imputed + scaled).
    low_card:
        Low-cardinality categorical names (one-hot encoded).
    high_card:
        High-cardinality categorical names (out-of-fold target encoded).
    scale:
        Whether numeric features are standardized (logistic benefits; the tree
        booster does not require it).
    """

    numeric: tuple[str, ...] = NUMERIC_FEATURES
    low_card: tuple[str, ...] = LOW_CARD_FEATURES
    high_card: tuple[str, ...] = HIGH_CARD_FEATURES
    scale: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this spec."""
        return {
            "numeric": list(self.numeric),
            "low_card": list(self.low_card),
            "high_card": list(self.high_card),
            "scale": self.scale,
            "meta": dict(self.meta),
        }


def build_feature_pipeline(spec: FeatureSpec | None = None, *, seed: int = 0) -> Pipeline:
    """Construct the (unfitted) application-time feature pipeline.

    Parameters
    ----------
    spec:
        Column assignment for each transformer branch; defaults to
        :class:`FeatureSpec`.
    seed:
        Seed for any stochastic component (e.g. the out-of-fold target-encoding
        fold assignment), for reproducibility.

    Returns
    -------
    sklearn.pipeline.Pipeline
        An unfitted pipeline whose ``ColumnTransformer`` imputes/encodes/scales
        the application-time features. Caller fits it on the TRAIN fold only.

    Raises
    ------
    NotImplementedError
        This is a stub; the implementation is filled in by the features author.
    """
    raise NotImplementedError("build_feature_pipeline is not yet implemented.")


def out_of_fold_target_encode(
    df: pd.DataFrame,
    y: pd.Series,
    columns: tuple[str, ...],
    *,
    n_folds: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Out-of-fold target encoding for high-cardinality categoricals.

    Each row's encoding for a column is the mean target over the *other* folds,
    so a row never sees its own label — the standard guard against target-
    encoding leakage. Smoothed toward the global base rate for rare categories.

    Parameters
    ----------
    df:
        The (train-fold) panel.
    y:
        The aligned binary target.
    columns:
        High-cardinality categorical columns to encode.
    n_folds:
        Number of out-of-fold partitions.
    seed:
        Fold-assignment seed.

    Returns
    -------
    pandas.DataFrame
        A frame of encoded columns aligned to ``df.index``.

    Raises
    ------
    NotImplementedError
        This is a stub; the implementation is filled in by the features author.
    """
    raise NotImplementedError("out_of_fold_target_encode is not yet implemented.")
