"""Hypothesis property tests for the core correctness invariants.

The headline invariants the brief mandates, asserted against the real
implementations:

(a) **No leakage column survives any pipeline** — running the *full* feature
    pipeline (which begins with ``drop_leakage``) on the synthetic panel AND on a
    perturbed-schema fixture leaves none of ``LEAKAGE_COLS`` in the engineered
    output, and ``assert_no_leakage`` passes on the post-drop frame.
(c) **The calibrated PD is in ``[0, 1]`` and monotone in the raw score** — a
    Hypothesis sweep over random isotonic knots and random raw-score vectors.

Invariants (b) prediction permutation-invariance and (d) later-vintage rows
cannot influence train-fold stats live in ``tests/property/test_pipeline.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from lendingclub_default.data.leakage import LEAKAGE_COLS, assert_no_leakage, drop_leakage
from lendingclub_default.features.pipeline import FeatureSpec, build_feature_pipeline
from lendingclub_default.models.calibrate import CalibratedModel


def _fit_pipeline_columns(panel: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop leakage, fit the feature pipeline, and return (clean frame, out names)."""
    clean = drop_leakage(panel)
    y = (panel["loan_status"].to_numpy() == "Charged Off").astype(int)
    spec = FeatureSpec()
    pipe = build_feature_pipeline(spec, seed=0)
    pipe.fit(clean.loc[:, list(spec.all_columns)], y)
    out_names = [str(name) for name in pipe.get_feature_names_out()]
    return clean, out_names


@pytest.mark.property
def test_fixture_leakage_columns_are_all_in_allowlist(
    schema_with_leakage: tuple[str, ...],
) -> None:
    """Precondition: every injected leakage column is in ``LEAKAGE_COLS``.

    Guarantees the "none survive the pipeline" assertions below have real targets
    to remove.
    """
    for col in schema_with_leakage:
        assert col.lower() in LEAKAGE_COLS


@pytest.mark.property
def test_no_leakage_column_survives_the_pipeline(
    synthetic_panel: pd.DataFrame,
    schema_with_leakage: tuple[str, ...],
) -> None:
    """Invariant (a): no leakage column survives the full feature pipeline.

    After ``drop_leakage`` + the fitted feature pipeline, none of the injected
    post-funding columns remain — neither as a surviving raw column nor as an
    engineered output feature name — and ``assert_no_leakage`` accepts the frame.
    """
    clean, out_names = _fit_pipeline_columns(synthetic_panel)

    # No leakage column survives the drop, and the backstop accepts the frame.
    surviving = {c for c in clean.columns if c.lower() in LEAKAGE_COLS}
    assert surviving == set(), f"leakage columns survived drop_leakage: {sorted(surviving)}"
    assert_no_leakage(clean)

    # No engineered output feature name is derived from a leakage column either.
    leaked_features = [
        name for name in out_names if any(lc in name.lower() for lc in schema_with_leakage)
    ]
    assert leaked_features == [], f"leakage-derived features survived: {leaked_features}"


@pytest.mark.property
def test_no_leakage_survives_on_perturbed_schema(
    synthetic_panel: pd.DataFrame,
) -> None:
    """Invariant (a), perturbed schema: extra/odd-cased leakage cols still drop.

    Perturb the panel with additional leakage columns (mixed case, plus a couple
    of LEAKAGE_COLS entries not in the base fixture) and assert the full pipeline
    still emits a leakage-free frame.
    """
    panel = synthetic_panel.copy()
    n = panel.shape[0]
    # Inject extra post-funding columns with awkward casing + new allowlist members.
    panel["Recoveries"] = np.linspace(0.0, 1.0, n)  # case perturbation
    panel["acc_now_delinq"] = np.zeros(n)
    panel["tot_coll_amt"] = np.zeros(n)
    panel["delinq_amnt"] = np.zeros(n)

    clean, out_names = _fit_pipeline_columns(panel)

    surviving = {c for c in clean.columns if c.lower() in LEAKAGE_COLS}
    assert surviving == set(), f"perturbed leakage columns survived: {sorted(surviving)}"
    assert_no_leakage(clean)
    assert all("recover" not in name.lower() for name in out_names)


@pytest.mark.property
@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    n_knots=st.integers(min_value=2, max_value=8),
    seed=st.integers(min_value=0, max_value=2**16),
)
def test_calibrated_pd_is_in_unit_interval_and_monotone(n_knots: int, seed: int) -> None:
    """Invariant (c): isotonic PD stays in ``[0, 1]`` and is monotone in raw score.

    For random monotone knots and random raw-score queries, ``calibrate`` returns
    probabilities clamped to ``[0, 1]`` that never decrease as the raw score rises.
    """
    rng = np.random.default_rng(seed)
    knots_x = np.sort(rng.uniform(0.0, 1.0, size=n_knots))
    # Distinct x knots so the piecewise-linear interpolation is well defined.
    knots_x = np.unique(knots_x)
    if knots_x.size < 2:
        knots_x = np.array([0.0, 1.0])
    knots_y = np.sort(rng.uniform(0.0, 1.0, size=knots_x.size))  # monotone non-decreasing

    model = CalibratedModel(
        method="isotonic",
        params={"knots_x": knots_x.tolist(), "knots_y": knots_y.tolist()},
    )

    queries = np.sort(rng.uniform(-0.5, 1.5, size=64))  # spans below/above the knots
    out = model.calibrate(queries)

    assert out.min() >= 0.0
    assert out.max() <= 1.0
    # Monotone non-decreasing: sorted inputs -> non-decreasing outputs.
    assert np.all(np.diff(out) >= -1e-12)
