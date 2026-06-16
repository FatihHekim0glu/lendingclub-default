"""Parity oracles vs. sklearn / scipy (filled in against the real implementations).

When the evaluation kernels are implemented, these assert AUC / Brier / log-loss /
KS agree with ``sklearn.metrics`` (and scipy KS) to ``1e-10``, and calibration
agrees with ``CalibratedClassifierCV``. While the kernels are stubs, the tests
document the contract: each entry point exists and currently raises
``NotImplementedError``.
"""

from __future__ import annotations

import numpy as np
import pytest

from lendingclub_default.evaluation.metrics import brier_score, roc_auc


@pytest.mark.parity
def test_metric_entry_points_exist_and_are_stubs() -> None:
    """The parity-checked metric entry points exist (and raise until implemented)."""
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.1, 0.9, 0.2, 0.8])
    with pytest.raises(NotImplementedError):
        roc_auc(y_true, y_score)
    with pytest.raises(NotImplementedError):
        brier_score(y_true, y_score)
