"""Temporal (vintage) train/test split by ``issue_d``.

Credit models must be evaluated *out-of-time*, never with a random K-fold:
training on later vintages and testing on earlier ones leaks the future and
flatters the score. :func:`temporal_split` therefore trains on vintages at or
before a cutoff ``issue_d`` and tests on strictly later vintages. The invariant
"no train row has an ``issue_d`` after any test row" is asserted (and
regression-tested).

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

# quantcore-candidate: new code (temporal vintage split); NEVER random K-fold.


@dataclass(frozen=True, slots=True)
class TemporalSplit:
    """Immutable index-level description of a temporal vintage split.

    Attributes
    ----------
    train_idx:
        Row index labels of the training fold (vintages <= cutoff).
    test_idx:
        Row index labels of the test fold (vintages > cutoff).
    cutoff:
        The ``issue_d`` cutoff (string label or timestamp) separating the folds;
        train is at or before it, test strictly after.
    train_vintages:
        Sorted unique vintage labels in the train fold.
    test_vintages:
        Sorted unique vintage labels in the test fold.
    """

    train_idx: list[Any]
    test_idx: list[Any]
    cutoff: Any
    train_vintages: list[Any]
    test_vintages: list[Any]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` (indices as lists)."""
        return {
            "n_train": len(self.train_idx),
            "n_test": len(self.test_idx),
            "cutoff": str(self.cutoff),
            "train_vintages": [str(v) for v in self.train_vintages],
            "test_vintages": [str(v) for v in self.test_vintages],
            "meta": dict(self.meta),
        }


def temporal_split(
    df: pd.DataFrame,
    *,
    issue_col: str = "issue_d",
    cutoff: Any | None = None,
    test_size: float = 0.25,
) -> TemporalSplit:
    """Split a panel into train (<= cutoff) and test (> cutoff) by vintage.

    Parameters
    ----------
    df:
        The (resolved-only) panel carrying an ``issue_d`` vintage column.
    issue_col:
        Name of the vintage column (default ``"issue_d"``).
    cutoff:
        Explicit vintage cutoff. If ``None``, the cutoff is chosen so that
        approximately ``test_size`` of rows (the latest vintages) fall in test.
    test_size:
        Target fraction of rows in the test fold when ``cutoff`` is ``None``.

    Returns
    -------
    TemporalSplit
        The train/test index partition and the chosen cutoff.

    Raises
    ------
    TemporalSplitError
        If the chosen split leaves an empty fold, or would place a train row
        after any test row.
    NotImplementedError
        This is a stub; the implementation is filled in by the data author.
    """
    raise NotImplementedError("temporal_split is not yet implemented.")


def assert_temporal_order(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    issue_col: str = "issue_d",
) -> None:
    """Raise if any train row has an ``issue_d`` strictly after any test row.

    The hard backstop behind the no-look-ahead regression test.

    Parameters
    ----------
    train, test:
        The two folds to check, each carrying ``issue_col``.
    issue_col:
        Name of the vintage column.

    Raises
    ------
    TemporalSplitError
        If ``max(train.issue_d) > min(test.issue_d)``.
    NotImplementedError
        This is a stub; the implementation is filled in by the data author.
    """
    raise NotImplementedError("assert_temporal_order is not yet implemented.")
