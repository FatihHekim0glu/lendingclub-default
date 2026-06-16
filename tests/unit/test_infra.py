"""Unit tests for the copied infra (rng, manifest, validation) and dataclasses.

These exercise the reused HRP infrastructure under its new package name and the
frozen result dataclasses' ``to_dict`` serialization, so the scaffold's own
plumbing is proven before the compute kernels land.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lendingclub_default._exceptions import (
    InsufficientDataError,
    LendingClubDefaultError,
    ValidationError,
)
from lendingclub_default._manifest import RunManifest, config_hash
from lendingclub_default._rng import make_rng, spawn_substreams
from lendingclub_default._validation import (
    align_inner,
    ensure_dataframe,
    ensure_series,
    validate_min_obs,
)


@pytest.mark.unit
def test_make_rng_is_deterministic() -> None:
    """The same seed yields identical draws; a negative seed raises."""
    a = make_rng(7).standard_normal(5)
    b = make_rng(7).standard_normal(5)
    assert np.array_equal(a, b)
    with pytest.raises(ValueError, match="non-negative"):
        make_rng(-1)


@pytest.mark.unit
def test_spawn_substreams_are_independent_and_reproducible() -> None:
    """Spawned substreams are reproducible and not all identical."""
    s1 = spawn_substreams(11, 3)
    s2 = spawn_substreams(11, 3)
    assert len(s1) == 3
    assert np.array_equal(s1[0].standard_normal(4), s2[0].standard_normal(4))
    with pytest.raises(ValueError, match="non-negative"):
        spawn_substreams(11, -2)


@pytest.mark.unit
def test_config_hash_is_order_invariant() -> None:
    """Logically-equal configs hash identically regardless of key order."""
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


@pytest.mark.unit
def test_run_manifest_capture_and_to_dict() -> None:
    """A captured manifest round-trips through to_dict with the recorded seed."""
    manifest = RunManifest.capture({"model": "xgb"}, seed=20260616)
    payload = manifest.to_dict()
    assert payload["seed"] == 20260616
    assert set(payload) >= {"git_sha", "dirty", "config_hash", "seed"}


@pytest.mark.unit
def test_ensure_series_and_dataframe_coerce_and_validate() -> None:
    """The validation helpers coerce to float64 and reject NaN / empty inputs."""
    s = ensure_series([1, 2, 3])
    assert s.dtype == np.float64
    frame = ensure_dataframe({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    assert frame.shape == (2, 2)
    with pytest.raises(ValidationError):
        ensure_series([1.0, np.nan])


@pytest.mark.unit
def test_align_inner_and_min_obs_guards() -> None:
    """align_inner intersects indices; validate_min_obs enforces a row floor."""
    left = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=[0, 1, 2])
    right = pd.DataFrame({"b": [9.0, 8.0]}, index=[1, 2])
    al, ar = align_inner(left, right)
    assert list(al.index) == [1, 2]
    assert list(ar.index) == [1, 2]
    with pytest.raises(InsufficientDataError):
        validate_min_obs(left, 10)


@pytest.mark.unit
def test_exception_hierarchy_is_rooted() -> None:
    """Every library error derives from the single base class."""
    assert issubclass(ValidationError, LendingClubDefaultError)
    assert issubclass(InsufficientDataError, ValidationError)
