"""Unit tests for the validation guards and feature-pipeline error paths.

These target the precondition checks that the rest of the library relies on:
the coercion helpers in ``_validation`` (ndim / empty / NaN / mismatched-index
rejection) and the out-of-fold target encoder's defensive branches (unfitted
transform, empty target, length mismatch, the ndarray input branch). They keep
the guardrails honest so a malformed input fails loudly at the boundary instead
of producing a silently wrong score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lendingclub_default._exceptions import ValidationError
from lendingclub_default._validation import (
    align_inner,
    ensure_dataframe,
    ensure_series,
)
from lendingclub_default.features.pipeline import (
    OutOfFoldTargetEncoder,
    out_of_fold_target_encode,
)


@pytest.mark.unit
def test_ensure_series_rejects_two_dimensional_array() -> None:
    """A 2-D ndarray is not a valid Series input."""
    with pytest.raises(ValidationError, match="1-dimensional"):
        ensure_series(np.zeros((2, 2)))


@pytest.mark.unit
def test_ensure_series_rejects_empty_input() -> None:
    """An empty sequence is rejected."""
    with pytest.raises(ValidationError, match="non-empty"):
        ensure_series([])


@pytest.mark.unit
def test_ensure_series_allows_nan_when_opted_in() -> None:
    """``allow_nan=True`` keeps NaN; the default rejects it."""
    s = ensure_series([1.0, np.nan], allow_nan=True)
    assert s.isna().any()
    with pytest.raises(ValidationError, match="NaN"):
        ensure_series([1.0, np.nan])


@pytest.mark.unit
def test_ensure_dataframe_rejects_three_dimensional_array() -> None:
    """A 3-D ndarray cannot become a 2-D DataFrame."""
    with pytest.raises(ValidationError, match="2-dimensional"):
        ensure_dataframe(np.zeros((2, 2, 2)))


@pytest.mark.unit
def test_ensure_dataframe_rejects_empty_frame() -> None:
    """A frame with zero columns is rejected."""
    with pytest.raises(ValidationError, match="at least one row and one column"):
        ensure_dataframe(pd.DataFrame())


@pytest.mark.unit
def test_ensure_dataframe_rejects_nan_by_default() -> None:
    """NaN in a frame is rejected unless explicitly allowed."""
    frame = pd.DataFrame({"x": [1.0, np.nan]})
    with pytest.raises(ValidationError, match="NaN"):
        ensure_dataframe(frame)
    assert ensure_dataframe(frame, allow_nan=True).isna().to_numpy().any()


@pytest.mark.unit
def test_ensure_dataframe_applies_column_labels_to_ndarray() -> None:
    """Column labels are applied when coercing an ndarray."""
    frame = ensure_dataframe(np.arange(6.0).reshape(3, 2), columns=["a", "b"])
    assert list(frame.columns) == ["a", "b"]


@pytest.mark.unit
def test_align_inner_raises_on_disjoint_indexes() -> None:
    """Two frames with no shared labels cannot be aligned."""
    left = pd.DataFrame({"a": [1.0]}, index=[0])
    right = pd.DataFrame({"b": [2.0]}, index=[9])
    with pytest.raises(ValidationError, match="no common index"):
        align_inner(left, right)


@pytest.mark.unit
def test_target_encoder_transform_before_fit_raises() -> None:
    """Transforming an unfitted encoder is rejected."""
    encoder = OutOfFoldTargetEncoder(("purpose",))
    with pytest.raises(ValidationError, match="must be fitted"):
        encoder.transform(pd.DataFrame({"purpose": ["a", "b"]}))


@pytest.mark.unit
def test_target_encoder_rejects_empty_target() -> None:
    """An empty target vector is rejected at fit."""
    encoder = OutOfFoldTargetEncoder(("purpose",))
    with pytest.raises(ValidationError, match="empty"):
        encoder.fit(pd.DataFrame({"purpose": []}), np.array([]))


@pytest.mark.unit
def test_target_encoder_rejects_length_mismatch() -> None:
    """X and y of different lengths are rejected."""
    encoder = OutOfFoldTargetEncoder(("purpose",))
    with pytest.raises(ValidationError, match="mismatched lengths"):
        encoder.fit(pd.DataFrame({"purpose": ["a", "b", "c"]}), np.array([0.0, 1.0]))


@pytest.mark.unit
def test_target_encoder_requires_a_target() -> None:
    """Fitting with ``y=None`` is rejected."""
    encoder = OutOfFoldTargetEncoder(("purpose",))
    with pytest.raises(ValidationError, match="requires a target"):
        encoder.fit(pd.DataFrame({"purpose": ["a"]}), None)


@pytest.mark.unit
def test_target_encoder_accepts_ndarray_input_and_round_trips() -> None:
    """The encoder handles a raw ndarray X (the non-DataFrame branch) and is monotone-safe."""
    rng = np.random.default_rng(0)
    cats = rng.integers(0, 4, size=200).astype(str).reshape(-1, 1)
    y = (cats.ravel().astype(int) >= 2).astype(float)
    encoder = OutOfFoldTargetEncoder(("c",), n_folds=4, seed=0)
    oof = encoder.fit_transform(cats, y)
    assert oof.shape == (200, 1)
    # Out-of-fold encodings stay within the observed target range.
    assert np.all(oof >= 0.0)
    assert np.all(oof <= 1.0)
    # The fitted full-fold map transforms held-out rows too.
    transformed = encoder.transform(cats)
    assert transformed.shape == (200, 1)
    names = encoder.get_feature_names_out()
    assert list(names) == ["c_te"]


@pytest.mark.unit
def test_out_of_fold_helper_rejects_empty_columns() -> None:
    """The functional helper rejects an empty column tuple."""
    df = pd.DataFrame({"purpose": ["a", "b"]})
    y = pd.Series([0.0, 1.0])
    with pytest.raises(ValidationError, match="no columns"):
        out_of_fold_target_encode(df, y, ())


@pytest.mark.unit
def test_out_of_fold_helper_rejects_missing_columns() -> None:
    """A column not present in the frame is reported."""
    df = pd.DataFrame({"purpose": ["a", "b"]})
    y = pd.Series([0.0, 1.0])
    with pytest.raises(ValidationError, match="missing columns"):
        out_of_fold_target_encode(df, y, ("addr_state",))


@pytest.mark.unit
def test_out_of_fold_helper_rejects_length_mismatch() -> None:
    """A frame and target of different lengths are rejected."""
    df = pd.DataFrame({"purpose": ["a", "b", "c"]})
    y = pd.Series([0.0, 1.0])
    with pytest.raises(ValidationError, match="mismatched lengths"):
        out_of_fold_target_encode(df, y, ("purpose",))
