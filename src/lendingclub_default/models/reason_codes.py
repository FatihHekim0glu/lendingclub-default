"""Per-loan reason codes (adverse-action style explanations).

Returns the top contributing features for a single scored loan, with a direction
(raises / lowers default risk) and a signed contribution. The container-safe path
uses **weight-of-evidence / logistic coefficients** (no heavy dependency); SHAP is
available for richer DEV-only analysis but is NEVER imported in the shipped
container (it lives in the ``[dev]`` extra only).

Importing this module has no side effects (SHAP, if used, is imported lazily
inside the dev-only helper).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

# quantcore-candidate: new code (reason codes); WOE/logit coeffs in-container,
# SHAP dev-only (never in the API image).


@dataclass(frozen=True, slots=True)
class ReasonCode:
    """A single feature's contribution to a loan's predicted default risk.

    Attributes
    ----------
    feature:
        The (human-readable) feature name.
    direction:
        ``"increases"`` if the feature pushes PD up, ``"decreases"`` if down.
    contribution:
        The signed contribution to the score (units depend on the explainer:
        log-odds for the logistic/WOE path).
    """

    feature: str
    direction: str
    contribution: float

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this reason code."""
        out = asdict(self)
        out["contribution"] = float(self.contribution)
        return out


def reason_codes_from_logit(
    coefficients: pd.Series,
    x_row: pd.Series,
    *,
    top_k: int = 5,
) -> list[ReasonCode]:
    """Compute reason codes from logistic coefficients for one loan (in-container).

    The signed contribution of feature ``j`` is ``coef_j * x_j`` (its additive
    push on the log-odds); the top ``top_k`` by absolute contribution are returned.

    Parameters
    ----------
    coefficients:
        The fitted logistic coefficients, indexed by feature name.
    x_row:
        The single loan's engineered feature vector, indexed to match.
    top_k:
        Number of reason codes to return.

    Returns
    -------
    list[ReasonCode]
        The top contributing features with direction and signed contribution.

    Raises
    ------
    NotImplementedError
        This is a stub; the implementation is filled in by the models author.
    """
    raise NotImplementedError("reason_codes_from_logit is not yet implemented.")


def shap_reason_codes(
    booster: object,
    x_row: np.ndarray,
    feature_names: list[str],
    *,
    top_k: int = 5,
) -> list[ReasonCode]:
    """DEV-ONLY: reason codes from SHAP values for the tree booster.

    SHAP is in the ``[dev]`` extra and imported LAZILY here; this function MUST
    NOT be called from the shipped container (which has no SHAP installed).

    Parameters
    ----------
    booster:
        The fitted tree booster.
    x_row:
        The single loan's feature vector.
    feature_names:
        Names aligned to ``x_row``.
    top_k:
        Number of reason codes to return.

    Returns
    -------
    list[ReasonCode]
        Top contributing features by absolute SHAP value.

    Raises
    ------
    NotImplementedError
        This is a stub; the implementation is filled in by the models author.
    """
    raise NotImplementedError("shap_reason_codes is not yet implemented.")
