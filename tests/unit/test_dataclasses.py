"""Unit tests for the frozen result dataclasses' ``to_dict`` serialization.

Every public result type is a frozen, slotted dataclass with a JSON-serializable
``to_dict`` so it crosses the API boundary cleanly. These tests pin those
contracts (real fields, JSON-safe scalars) before the compute kernels populate
them, and guard against accidental drift.
"""

from __future__ import annotations

import pytest

from lendingclub_default.data.synthetic import SyntheticConfig
from lendingclub_default.evaluation.metrics import MetricBundle
from lendingclub_default.evaluation.threshold import CostMatrix, ThresholdSweep
from lendingclub_default.features.pipeline import FeatureSpec
from lendingclub_default.models.reason_codes import ReasonCode


@pytest.mark.unit
def test_synthetic_config_to_dict_has_real_fields() -> None:
    """SyntheticConfig serializes its generator knobs."""
    payload = SyntheticConfig().to_dict()
    assert payload["base_default_rate"] == 0.15
    assert len(payload["vintages"]) >= 4


@pytest.mark.unit
def test_metric_bundle_to_dict_is_json_safe() -> None:
    """MetricBundle.to_dict yields float metrics and an int count."""
    bundle = MetricBundle(
        roc_auc=0.70,
        pr_auc=0.35,
        brier=0.12,
        log_loss=0.38,
        ks=0.30,
        base_rate=0.15,
        n=1000,
    )
    payload = bundle.to_dict()
    assert isinstance(payload["roc_auc"], float)
    assert isinstance(payload["n"], int)
    assert payload["base_rate"] == 0.15


@pytest.mark.unit
def test_cost_matrix_and_threshold_sweep_serialize() -> None:
    """CostMatrix and ThresholdSweep round-trip through to_dict."""
    assert CostMatrix().to_dict()["cost_fn"] == 1.0
    sweep = ThresholdSweep(
        thresholds=[0.1, 0.2],
        expected_cost=[0.5, 0.4],
        best_threshold=0.2,
        best_cost=0.4,
        base_rate=0.15,
    )
    payload = sweep.to_dict()
    assert payload["best_threshold"] == 0.2
    assert isinstance(payload["best_cost"], float)


@pytest.mark.unit
def test_feature_spec_and_reason_code_serialize() -> None:
    """FeatureSpec exposes its column branches; ReasonCode is JSON-safe."""
    spec = FeatureSpec().to_dict()
    assert "int_rate" in spec["numeric"]
    assert "purpose" in spec["high_card"]
    rc = ReasonCode(feature="dti", direction="increases", contribution=0.4).to_dict()
    assert rc["feature"] == "dti"
    assert isinstance(rc["contribution"], float)
