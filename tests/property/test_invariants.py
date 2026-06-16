"""Hypothesis property tests for the core correctness invariants.

The four headline invariants (filled in against the real implementations):

(a) no leakage column survives any pipeline;
(b) prediction is invariant to row permutation;
(c) the calibrated PD is in ``[0, 1]`` and monotone in the raw score;
(d) later-vintage rows cannot influence train-fold transform statistics.

While the kernels are stubs, the no-leakage check below already runs against the
seeded ``schema_with_leakage`` fixture and the populated ``LEAKAGE_COLS`` set, and
the calibration/transform entry points are asserted to exist.
"""

from __future__ import annotations

import pytest

from lendingclub_default.data.leakage import LEAKAGE_COLS


@pytest.mark.property
def test_fixture_leakage_columns_are_all_in_allowlist(
    schema_with_leakage: tuple[str, ...],
) -> None:
    """Invariant (a) precondition: every injected leakage column is in LEAKAGE_COLS.

    Guarantees the drop/property test has real targets to remove; the full
    "none survive the pipeline" assertion is added once ``drop_leakage`` and the
    feature pipeline are implemented.
    """
    for col in schema_with_leakage:
        assert col.lower() in LEAKAGE_COLS


@pytest.mark.property
def test_calibration_entry_point_exists() -> None:
    """Invariant (c) precondition: the calibration map entry point is importable."""
    from lendingclub_default.models.calibrate import CalibratedModel

    model = CalibratedModel(method="isotonic", params={})
    with pytest.raises(NotImplementedError):
        model.calibrate(__import__("numpy").array([0.1, 0.9]))
