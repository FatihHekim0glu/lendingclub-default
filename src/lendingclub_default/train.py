"""End-to-end training orchestration.

Wires the whole honest pipeline together:

``load -> drop leakage -> build labels (exclude in-progress) -> temporal vintage
split -> fit feature pipeline on TRAIN only -> fit baselines + XGBoost (early stop
on a temporal validation slice) -> calibrate on a held-out fold -> evaluate on the
held-out LATEST vintage -> DeLong(XGB vs logistic) -> emit a <2MB booster JSON +
the fitted pipeline + a RunManifest``.

Runs identically on the synthetic generator (default) and a real Kaggle CSV
(``--data``). Importing this module has no side effects; training only happens
when :func:`train` is called (and the ``__main__`` demo is guarded).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lendingclub_default.data.synthetic import SyntheticConfig
    from lendingclub_default.evaluation.delong import DeLongResult
    from lendingclub_default.evaluation.metrics import MetricBundle
    from lendingclub_default.models.xgb import XGBConfig

# quantcore-candidate: new code (training orchestration); RunManifest-stamped
# <2MB synthetic-trained artifact.


@dataclass(frozen=True, slots=True)
class TrainArtifacts:
    """Immutable bundle of everything a training run produces.

    Attributes
    ----------
    booster_path:
        Path to the emitted ``<2MB`` booster JSON.
    pipeline_path:
        Path to the serialized fitted feature pipeline.
    metrics:
        Held-out (latest-vintage) headline metrics for the calibrated XGBoost.
    logistic_metrics:
        Held-out metrics for the L2-logistic baseline (for the horse race).
    base_rate_auc:
        ROC-AUC of the base-rate predictor (0.5 floor, recorded for honesty).
    delong:
        DeLong AUC-difference test, XGBoost vs logistic.
    n_trials:
        The RECORDED nested-CV trial count (guard asserts it >= the config grid).
    data_source:
        ``"synthetic"`` or ``"kaggle"`` — which panel the artifact was trained on.
    manifest:
        The run's :class:`lendingclub_default.RunManifest` as a dict.
    """

    booster_path: Path
    pipeline_path: Path
    metrics: MetricBundle
    logistic_metrics: MetricBundle
    base_rate_auc: float
    delong: DeLongResult
    n_trials: int
    data_source: str
    manifest: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable summary of the artifacts."""
        return {
            "booster_path": str(self.booster_path),
            "pipeline_path": str(self.pipeline_path),
            "metrics": self.metrics.to_dict(),
            "logistic_metrics": self.logistic_metrics.to_dict(),
            "base_rate_auc": float(self.base_rate_auc),
            "delong": self.delong.to_dict(),
            "n_trials": int(self.n_trials),
            "data_source": self.data_source,
            "manifest": dict(self.manifest),
        }


def train(
    *,
    data_path: Path | str | None = None,
    out_dir: Path | str = "artifacts",
    synthetic_config: SyntheticConfig | None = None,
    xgb_config: XGBConfig | None = None,
    seed: int = 20260616,
) -> TrainArtifacts:
    """Run the full leakage-free, temporally-split, calibrated training pipeline.

    Parameters
    ----------
    data_path:
        Real LendingClub accepted-loans CSV. If ``None``, the synthetic generator
        is used (the reproducible default; the shipped artifact is synthetic).
    out_dir:
        Directory to write the booster JSON, fitted pipeline, and manifest.
    synthetic_config:
        Synthetic-generator config (used only when ``data_path is None``).
    xgb_config:
        XGBoost hyperparameters; defaults to :class:`lendingclub_default.models.xgb.XGBConfig`.
    seed:
        Master RNG seed for the whole run (RunManifest-stamped).

    Returns
    -------
    TrainArtifacts
        Paths to the emitted artifacts plus held-out metrics, the DeLong test,
        the recorded ``n_trials``, and the run manifest.

    Raises
    ------
    NotImplementedError
        This is a stub; the implementation is filled in by the training author.
    """
    raise NotImplementedError("train is not yet implemented.")


if __name__ == "__main__":  # pragma: no cover
    # Demo entry point is intentionally guarded so importing this module is pure.
    train()
