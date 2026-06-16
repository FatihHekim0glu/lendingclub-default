"""Synthetic LendingClub-schema data generator.

There is no real Kaggle LendingClub dump (or credentials) on this machine, so the
shipped demo model is trained on a *synthetic* panel that mimics the real LC
application-time schema. This module emits a deterministic pandas panel with:

- the real application-time columns (``loan_amnt``, ``term``, ``int_rate``,
  ``grade``, ``sub_grade``, ``emp_length``, ``home_ownership``, ``annual_inc``,
  ``dti``, ``fico_range_low/high``, ``revol_util``, ``open_acc``, ``pub_rec``,
  ``purpose``, ``addr_state``, ``installment``, ``verification_status``,
  ``issue_d``, ...);
- a realistic ``loan_status`` outcome column;
- the post-funding LEAKAGE columns (``recoveries``, ``total_pymnt*``,
  ``out_prncp*``, ``last_pymnt_*``, ``collection_recovery_fee``,
  ``total_rec_*``, ``debt_settlement_flag``, ...) so the leakage allowlist is
  genuinely exercised end-to-end.

Design targets (so a leakage-free model lands at a *believable* AUC, not 0.5 and
not 0.99): base default rate ~15%; default probability a noisy monotone function
of ``fico``/``dti``/``int_rate``/``grade``; ``issue_d`` spread across 2015-2018
vintage cohorts with mild regime drift. All randomness flows from
:func:`lendingclub_default._rng.make_rng`, so a single seed reproduces the panel
byte-for-byte.

Importing this module has no side effects; data is only produced when
:func:`generate_synthetic_panel` is explicitly called.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

# quantcore-candidate: new code (synthetic LC-schema generator); seeded via _rng.


@dataclass(frozen=True, slots=True)
class SyntheticConfig:
    """Immutable configuration for the synthetic LC panel generator.

    Attributes
    ----------
    n_loans:
        Number of loans (rows) to emit.
    base_default_rate:
        Target marginal default rate (fraction of resolved loans that default).
        The real LC accepted-loans rate is ~15%.
    vintages:
        Ordered ``issue_d`` cohort labels (``"YYYY-MM"`` or ``"%b-%Y"``) spread
        across the modelled period (2015-2018). Each loan is assigned to one.
    regime_drift:
        Per-vintage additive drift on the latent default log-odds, modelling mild
        macro regime change across cohorts (later vintages slightly riskier).
    noise_scale:
        Standard deviation of the idiosyncratic latent noise; tuned so a
        leakage-free model achieves a believable AUC in ``[0.65, 0.72]``.
    seed:
        Master RNG seed (feeds :func:`lendingclub_default._rng.make_rng`).
    """

    n_loans: int = 20_000
    base_default_rate: float = 0.15
    vintages: tuple[str, ...] = (
        "2015-Q1",
        "2015-Q3",
        "2016-Q1",
        "2016-Q3",
        "2017-Q1",
        "2017-Q3",
        "2018-Q1",
        "2018-Q3",
    )
    regime_drift: float = 0.05
    noise_scale: float = 1.0
    seed: int = 20260616
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this config."""
        return asdict(self)


def generate_synthetic_panel(config: SyntheticConfig | None = None) -> pd.DataFrame:
    """Generate a deterministic synthetic LC-schema loan panel.

    The returned frame carries application-time columns, a ``loan_status``
    outcome, ``issue_d`` vintages, AND the post-funding leakage columns, so the
    full load -> drop-leakage -> label -> temporal-split pipeline is exercised on
    realistic data. The default probability is a noisy monotone function of
    ``fico``/``dti``/``int_rate``/``grade``; the marginal default rate matches
    ``config.base_default_rate``.

    Parameters
    ----------
    config:
        Generator configuration; defaults to :class:`SyntheticConfig` (a 20k-loan,
        ~15%-default, 2015-2018 panel).

    Returns
    -------
    pandas.DataFrame
        The synthetic panel, one row per loan.

    Raises
    ------
    NotImplementedError
        This is a stub; the full generator is filled in by the data author. (Test
        fixtures build a smaller usable frame directly via ``conftest``.)
    """
    raise NotImplementedError("generate_synthetic_panel is not yet implemented.")
