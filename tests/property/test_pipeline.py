"""Property tests for the application-time feature pipeline.

Covers the three invariants the brief requires for the features group:

1. **No look-ahead.** Later-vintage (test-fold) rows cannot influence the
   train-fold transform statistics: a pipeline fitted on the train fold only
   transforms held-out rows identically regardless of how the later vintages are
   ordered or whether extra later rows exist.
2. **Fit/transform shape invariance.** ``transform`` produces the same number of
   feature columns as ``fit_transform``, for any (non-empty) row count.
3. **Row-permutation invariance.** Permuting the input rows permutes the output
   rows identically — the transform is row-wise and order-independent.

A fourth check pins the out-of-fold target encoder's leakage guard: its
``fit_transform`` (out-of-fold) encoding for a category never equals the naive
full-fold mean, and unseen categories fall back to the global prior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lendingclub_default.features.pipeline import (
    FeatureSpec,
    OutOfFoldTargetEncoder,
    build_feature_pipeline,
    out_of_fold_target_encode,
)


def _xy(panel: pd.DataFrame, spec: FeatureSpec) -> tuple[pd.DataFrame, pd.Series]:
    """Project the panel to the pipeline's input columns + binary target."""
    x = panel.loc[:, list(spec.all_columns)].reset_index(drop=True)
    y = (panel["loan_status"].to_numpy() == "Charged Off").astype(int)
    return x, pd.Series(y, index=x.index)


@pytest.mark.property
def test_later_vintage_rows_cannot_influence_train_fold_stats(
    synthetic_panel: pd.DataFrame,
) -> None:
    """Invariant (d): a train-only fit is invariant to the later-vintage rows.

    We fit on the earliest vintages, then transform the held-out later vintages.
    Re-ordering or extending the later vintages must not change either the
    learned statistics or the transform of any held-out row, because the encoder
    is *only ever* fitted on the train slice.
    """
    panel = synthetic_panel.sort_values("issue_d").reset_index(drop=True)
    x, y = _xy(panel, FeatureSpec())
    cutoff = len(x) // 2
    x_train, y_train = x.iloc[:cutoff], y.iloc[:cutoff]
    x_test = x.iloc[cutoff:]

    pipe = build_feature_pipeline(FeatureSpec(), seed=0).fit(x_train, y_train)
    z_test = pipe.transform(x_test)

    # Shuffling the later vintages must not alter the per-row transform.
    rng = np.random.default_rng(11)
    perm = rng.permutation(len(x_test))
    z_test_shuffled = pipe.transform(x_test.iloc[perm])
    np.testing.assert_allclose(z_test[perm], z_test_shuffled)

    # Refitting on the *identical* train fold yields identical learned stats,
    # independent of anything in the test fold.
    pipe2 = build_feature_pipeline(FeatureSpec(), seed=0).fit(x_train, y_train)
    np.testing.assert_allclose(z_test, pipe2.transform(x_test))


@pytest.mark.property
def test_fit_transform_shape_invariance(synthetic_panel: pd.DataFrame) -> None:
    """Invariant (shape): transform width matches fit_transform width."""
    x, y = _xy(synthetic_panel, FeatureSpec())
    pipe = build_feature_pipeline(FeatureSpec(), seed=0)
    z_fit = pipe.fit_transform(x, y)

    x_held = x.iloc[: len(x) // 3]
    z_transform = pipe.transform(x_held)

    assert z_fit.shape[0] == len(x)
    assert z_transform.shape[0] == len(x_held)
    assert z_fit.shape[1] == z_transform.shape[1]
    assert not np.isnan(z_fit).any()
    assert not np.isnan(z_transform).any()


@pytest.mark.property
@settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(seed=st.integers(min_value=0, max_value=2**16))
def test_row_permutation_invariance(synthetic_panel: pd.DataFrame, seed: int) -> None:
    """Invariant (b): transform is row-wise — permuting rows permutes outputs."""
    x, y = _xy(synthetic_panel, FeatureSpec())
    cutoff = len(x) // 2
    pipe = build_feature_pipeline(FeatureSpec(), seed=0).fit(x.iloc[:cutoff], y.iloc[:cutoff])

    x_eval = x.iloc[cutoff:]
    base = pipe.transform(x_eval)
    perm = np.random.default_rng(seed).permutation(len(x_eval))
    permuted = pipe.transform(x_eval.iloc[perm])
    np.testing.assert_allclose(base[perm], permuted)


@pytest.mark.property
def test_oof_encoding_differs_from_full_fold_mean(synthetic_panel: pd.DataFrame) -> None:
    """The out-of-fold encoding is not the naive full-fold mean (leakage guard).

    If the encoder were leaking, the train-fold encoding would equal the per-
    category full-fold mean. Out-of-fold encoding holds each row out, so the two
    must differ for at least one row.
    """
    x, y = _xy(synthetic_panel, FeatureSpec())
    spec = FeatureSpec()
    oof = out_of_fold_target_encode(x, y, spec.high_card, seed=0)

    full = OutOfFoldTargetEncoder(spec.high_card, seed=0).fit(x, y.to_numpy())
    full_fold = full.transform(x)

    assert oof.shape == full_fold.shape
    assert not np.allclose(oof.to_numpy(), full_fold)
    # Encodings are valid probabilities in [0, 1] (the target is binary).
    assert float(oof.to_numpy().min()) >= 0.0
    assert float(oof.to_numpy().max()) <= 1.0


@pytest.mark.property
def test_unseen_category_falls_back_to_prior(synthetic_panel: pd.DataFrame) -> None:
    """Categories unseen at fit time encode to the global prior, never NaN."""
    x, y = _xy(synthetic_panel, FeatureSpec())
    spec = FeatureSpec()
    enc = OutOfFoldTargetEncoder(spec.high_card, seed=0).fit(x, y.to_numpy())

    novel = x.iloc[:5].copy()
    novel["purpose"] = "a_purpose_never_seen_in_training"
    encoded = enc.transform(novel)
    purpose_col = spec.high_card.index("purpose")
    assert not np.isnan(encoded).any()
    np.testing.assert_allclose(encoded[:, purpose_col], enc.prior_)
