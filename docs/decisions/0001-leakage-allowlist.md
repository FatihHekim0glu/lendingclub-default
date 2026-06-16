# ADR-0001: A frozen post-funding leakage allowlist, dropped before any feature

- **Status:** Accepted
- **Date:** 2026-06-16
- **Deciders:** lendingclub-default maintainers
- **Related:** [ADR-0002](0002-temporal-vintage-split.md) (the split the clean panel feeds)

## Context

The single biggest trap in a LendingClub default model is **post-funding
leakage**. The raw LC export mixes two fundamentally different kinds of column:

- **Application-time** fields known at origination (`loan_amnt`, `int_rate`,
  `grade`, `fico_range_low`, `dti`, `annual_inc`, `purpose`, ...).
- **Post-funding outcome** fields that only exist *because the loan already ran*
  (`recoveries`, `total_pymnt*`, `out_prncp*`, `last_pymnt_*`,
  `collection_recovery_fee`, `total_rec_*`, `debt_settlement_flag`,
  `settlement_*`, `hardship_*`, `funded_amnt*`, `acc_now_delinq`, `tot_coll_amt`,
  `delinq_amnt`, ...).

A model that sees even one post-funding column learns the answer from the
outcome: AUC shoots toward ~0.99, which is not a credit model — it is a
data-leakage bug wearing a credit model's clothes. Public notebooks do this
constantly, often by `df.dropna(axis=1)` or "keep the numeric columns", which
silently retains payment history.

There are two failure modes to defend against: (a) *silently keeping* a leakage
column, and (b) a leakage column *re-appearing* downstream (a rename, a merged
schema, a perturbed input) after an early drop.

A second, subtler point belongs here: `int_rate` and `grade` are
application-time and therefore **kept**, but they are LendingClub's *own* risk
model's outputs. Their predictive power is partly circular — the model is in
part re-learning LC's pricing. We keep them (they are genuinely available at
origination) but disclose the circularity as a headline limitation.

## Decision

`data/leakage.py` defines a **single frozen `LEAKAGE_COLS` allowlist** — the
canonical list of every post-funding column from the LC data dictionary — and a
`drop_leakage(df)` that removes them **case-insensitively** before any feature is
built. `assert_no_leakage(df)` raises `LeakageError` listing any survivors.

The drop is the **first** step of `train.py`, ahead of labels, split, and the
feature pipeline. Nothing downstream may re-introduce a leakage column.

This is enforced, not assumed, by property tests:

- no `LEAKAGE_COLS` column survives the full preprocessing pipeline, on the
  synthetic panel **and** on a perturbed-schema fixture;
- `drop_leakage` is pure (does not mutate its input) and idempotent;
- application-time columns are preserved.

## Consequences

- **Positive.** The reported AUC (~0.71 synthetic) is *believable* — above the
  base-rate floor, nowhere near a leakage-grade ~0.99. The golden-band regression
  test locks this in.
- **Positive.** The allowlist is one reviewed artifact, not scattered `drop`
  calls. Adding a newly discovered post-funding column is a one-line, tested change.
- **Positive.** The `int_rate`/`grade` circularity is surfaced honestly rather
  than laundered into a "strong model".
- **Cost.** We discard genuinely predictive *outcome* signal. That is the entire
  point — at origination that signal does not exist, so using it is a lie about
  what the model can do.
- **Risk addressed.** "Post-funding leakage" — the defining failure of this
  problem — is eliminated by an exact, property-tested, fail-closed allowlist.
