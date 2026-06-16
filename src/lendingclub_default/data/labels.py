"""Target-label construction with strict label hygiene.

The binary target is built from ``loan_status``:

- positive (1): a *resolved, defaulted* loan — ``loan_status`` in
  :data:`lendingclub_default._constants.DEFAULT_STATUSES`
  (``Charged Off`` / ``Default``);
- negative (0): a *resolved, repaid* loan — ``loan_status`` in
  :data:`lendingclub_default._constants.PAID_STATUSES` (``Fully Paid``).

In-progress loans (``Current``, ``Late ...``, ``In Grace Period``, ``Issued``)
have no known outcome and are **excluded** from the labelled set — including them
would silently bias the base rate and the calibration. This exclusion is the
documented label-hygiene rule and is unit-tested.

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

# quantcore-candidate: new code (label construction); status vocab in _constants.


@dataclass(frozen=True, slots=True)
class LabelResult:
    """Immutable result of building labels from a raw panel.

    Attributes
    ----------
    panel:
        The panel restricted to resolved loans (in-progress rows dropped),
        index-aligned with ``y``.
    y:
        The binary target Series (1 = default, 0 = fully paid), same index as
        ``panel``.
    base_rate:
        The marginal default rate over the resolved set (``y.mean()``).
    n_excluded:
        Count of in-progress rows dropped for label hygiene.
    """

    panel: pd.DataFrame
    y: pd.Series
    base_rate: float
    n_excluded: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable summary (omits the heavy frames)."""
        out = {k: v for k, v in asdict(self).items() if k not in {"panel", "y"}}
        out["n_resolved"] = int(self.panel.shape[0]) if self.panel is not None else 0
        return out


def build_labels(df: pd.DataFrame, *, status_col: str = "loan_status") -> LabelResult:
    """Build the binary default target, excluding in-progress loans.

    Parameters
    ----------
    df:
        A panel carrying a ``loan_status`` column.
    status_col:
        Name of the status column (default ``"loan_status"``).

    Returns
    -------
    LabelResult
        The resolved-only panel, its aligned binary target, the base rate, and
        the number of excluded in-progress rows.

    Raises
    ------
    ValidationError
        If ``status_col`` is absent, or no rows resolve to a usable label.
    NotImplementedError
        This is a stub; the implementation is filled in by the data author.
    """
    raise NotImplementedError("build_labels is not yet implemented.")
