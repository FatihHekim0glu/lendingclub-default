"""Data-group tests: synthetic generator, leakage drop, labels, temporal split.

Covers the data author's group (``data/synthetic.py``, ``data/leakage.py``,
``data/labels.py``, ``data/load.py``, ``data/split.py``):

- **label construction** — defaults/paid resolve to {1, 0}, in-progress excluded;
- **leakage-drop completeness** — NO ``LEAKAGE_COLS`` member survives, including
  the three borrower-status fields (``acc_now_delinq``, ``tot_coll_amt``,
  ``delinq_amnt``) added to the runtime allowlist;
- **temporal order** — no train ``issue_d`` is after any test ``issue_d``;
- **seed determinism** — the generator reproduces its panel byte-for-byte.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lendingclub_default._constants import (
    DEFAULT_STATUSES,
    IN_PROGRESS_STATUSES,
    PAID_STATUSES,
)
from lendingclub_default._exceptions import (
    LeakageError,
    TemporalSplitError,
    ValidationError,
)
from lendingclub_default.data.labels import build_labels
from lendingclub_default.data.leakage import (
    LEAKAGE_COLS,
    assert_no_leakage,
    drop_leakage,
)
from lendingclub_default.data.load import coerce_dtypes, load_panel
from lendingclub_default.data.split import (
    assert_temporal_order,
    temporal_split,
)
from lendingclub_default.data.synthetic import (
    SyntheticConfig,
    generate_synthetic_panel,
)

# --------------------------------------------------------------------------- #
# synthetic generator                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_generator_emits_full_schema() -> None:
    """The generator emits application-time, outcome, and leakage columns."""
    panel = generate_synthetic_panel(SyntheticConfig(n_loans=1_500, seed=7))
    assert panel.shape[0] == 1_500
    for col in ("loan_amnt", "int_rate", "grade", "issue_d", "loan_status"):
        assert col in panel.columns
    # Post-funding leakage columns, including the three borrower-status fields.
    for col in (
        "recoveries",
        "total_pymnt",
        "out_prncp",
        "acc_now_delinq",
        "tot_coll_amt",
        "delinq_amnt",
    ):
        assert col in panel.columns


@pytest.mark.unit
def test_generator_default_rate_matches_target() -> None:
    """The marginal default rate is tuned to the configured base rate."""
    panel = generate_synthetic_panel(
        SyntheticConfig(n_loans=8_000, base_default_rate=0.15, seed=11)
    )
    rate = (panel["loan_status"] == "Charged Off").mean()
    assert 0.12 <= rate <= 0.18


@pytest.mark.unit
def test_generator_is_seed_deterministic() -> None:
    """Same seed -> byte-identical panel; different seed -> different panel."""
    cfg = SyntheticConfig(n_loans=900, seed=123)
    a = generate_synthetic_panel(cfg)
    b = generate_synthetic_panel(cfg)
    pd.testing.assert_frame_equal(a, b)

    c = generate_synthetic_panel(SyntheticConfig(n_loans=900, seed=124))
    assert not a["loan_status"].equals(c["loan_status"])


@pytest.mark.unit
def test_generator_leakage_separates_outcome() -> None:
    """Leakage columns carry outcome signal (the leakage trap is genuine)."""
    panel = generate_synthetic_panel(SyntheticConfig(n_loans=4_000, seed=5))
    defaulted = panel["loan_status"] == "Charged Off"
    assert panel.loc[~defaulted, "recoveries"].mean() < panel.loc[defaulted, "recoveries"].mean()
    assert (
        panel.loc[~defaulted, "acc_now_delinq"].mean()
        < panel.loc[defaulted, "acc_now_delinq"].mean()
    )


@pytest.mark.unit
def test_generator_default_prob_is_monotone_in_drivers() -> None:
    """A leakage-free signal exists: worse grades default more often."""
    panel = generate_synthetic_panel(SyntheticConfig(n_loans=12_000, seed=9))
    defaulted = panel["loan_status"] == "Charged Off"
    rate_by_grade = defaulted.groupby(panel["grade"]).mean()
    # Grade A (best) should default less often than grade G (worst).
    assert rate_by_grade.get("A", 0.0) < rate_by_grade.get("G", 1.0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad_cfg",
    [
        SyntheticConfig(n_loans=0),
        SyntheticConfig(vintages=()),
        SyntheticConfig(base_default_rate=0.0),
        SyntheticConfig(base_default_rate=1.0),
    ],
)
def test_generator_rejects_bad_config(bad_cfg: SyntheticConfig) -> None:
    """Invalid configs raise a ValidationError rather than silently degrading."""
    with pytest.raises(ValidationError):
        generate_synthetic_panel(bad_cfg)


# --------------------------------------------------------------------------- #
# leakage drop                                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_drop_leakage_removes_all_leakage_columns(synthetic_panel: pd.DataFrame) -> None:
    """No LEAKAGE_COLS member survives drop_leakage on the synthetic panel."""
    cleaned = drop_leakage(synthetic_panel)
    survivors = {c for c in cleaned.columns if str(c).lower() in LEAKAGE_COLS}
    assert survivors == set()


@pytest.mark.unit
def test_drop_leakage_removes_three_borrower_status_fields() -> None:
    """The three borrower-status leak fields are dropped (explicit guard)."""
    panel = generate_synthetic_panel(SyntheticConfig(n_loans=300, seed=3))
    for col in ("acc_now_delinq", "tot_coll_amt", "delinq_amnt"):
        assert col in panel.columns  # present pre-drop ...
    cleaned = drop_leakage(panel)
    for col in ("acc_now_delinq", "tot_coll_amt", "delinq_amnt"):
        assert col not in cleaned.columns  # ... and gone post-drop.


@pytest.mark.unit
def test_drop_leakage_is_case_insensitive_and_pure() -> None:
    """Matching is case-insensitive; the input frame is never mutated."""
    df = pd.DataFrame(
        {"loan_amnt": [1.0], "RECOVERIES": [2.0], "Total_Pymnt": [3.0], "grade": ["A"]}
    )
    cleaned = drop_leakage(df)
    assert list(cleaned.columns) == ["loan_amnt", "grade"]
    # Original untouched.
    assert "RECOVERIES" in df.columns


@pytest.mark.unit
def test_drop_leakage_preserves_application_columns(synthetic_panel: pd.DataFrame) -> None:
    """Application-time columns survive the drop."""
    cleaned = drop_leakage(synthetic_panel)
    for col in ("loan_amnt", "int_rate", "grade", "dti", "issue_d"):
        assert col in cleaned.columns


@pytest.mark.unit
def test_assert_no_leakage_passes_after_drop(synthetic_panel: pd.DataFrame) -> None:
    """assert_no_leakage is silent on a cleaned frame and raises on a dirty one."""
    cleaned = drop_leakage(synthetic_panel)
    assert_no_leakage(cleaned)  # no raise
    with pytest.raises(LeakageError):
        assert_no_leakage(synthetic_panel)


@pytest.mark.unit
def test_assert_no_leakage_reports_survivors() -> None:
    """The LeakageError names the surviving leakage column(s)."""
    df = pd.DataFrame({"loan_amnt": [1.0], "recoveries": [2.0]})
    with pytest.raises(LeakageError, match="recoveries"):
        assert_no_leakage(df)


# --------------------------------------------------------------------------- #
# labels                                                                       #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_build_labels_maps_resolved_statuses() -> None:
    """Charged Off/Default -> 1, Fully Paid -> 0; base rate is the mean."""
    df = pd.DataFrame(
        {
            "loan_status": ["Charged Off", "Fully Paid", "Default", "Fully Paid"],
            "x": [1, 2, 3, 4],
        }
    )
    result = build_labels(df)
    assert list(result.y) == [1, 0, 1, 0]
    assert result.base_rate == 0.5
    assert result.n_excluded == 0
    assert result.panel.shape[0] == 4


@pytest.mark.unit
def test_build_labels_excludes_in_progress() -> None:
    """In-progress loans are dropped (label hygiene), counted in n_excluded."""
    statuses = [
        *DEFAULT_STATUSES,
        *PAID_STATUSES,
        *IN_PROGRESS_STATUSES,
    ]
    df = pd.DataFrame({"loan_status": statuses, "x": range(len(statuses))})
    result = build_labels(df)
    assert result.n_excluded == len(IN_PROGRESS_STATUSES)
    assert result.panel.shape[0] == len(DEFAULT_STATUSES) + len(PAID_STATUSES)
    # Every retained row has a 0/1 label and no in-progress status survives.
    assert set(result.y.unique()).issubset({0, 1})
    assert not result.panel["loan_status"].isin(IN_PROGRESS_STATUSES).any()


@pytest.mark.unit
def test_build_labels_index_alignment() -> None:
    """y is index-aligned with the resolved panel."""
    df = pd.DataFrame(
        {"loan_status": ["Current", "Charged Off", "Fully Paid"]},
        index=["a", "b", "c"],
    )
    result = build_labels(df)
    assert list(result.panel.index) == ["b", "c"]
    assert list(result.y.index) == ["b", "c"]
    assert result.to_dict()["n_resolved"] == 2


@pytest.mark.unit
def test_build_labels_on_synthetic_panel(synthetic_panel: pd.DataFrame) -> None:
    """The synthetic panel (all resolved) yields a ~15% base rate, nothing excluded."""
    result = build_labels(synthetic_panel)
    assert result.n_excluded == 0
    assert 0.08 <= result.base_rate <= 0.25


@pytest.mark.unit
def test_build_labels_rejects_missing_column() -> None:
    """A missing status column raises ValidationError."""
    with pytest.raises(ValidationError, match="status"):
        build_labels(pd.DataFrame({"x": [1, 2]}))


@pytest.mark.unit
def test_build_labels_rejects_all_unresolved() -> None:
    """A panel with no resolved loans raises ValidationError."""
    df = pd.DataFrame({"loan_status": list(IN_PROGRESS_STATUSES)})
    with pytest.raises(ValidationError, match="no rows resolve"):
        build_labels(df)


# --------------------------------------------------------------------------- #
# temporal split                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_temporal_split_partitions_by_vintage(
    k_vintage_fixture: tuple[pd.DataFrame, list[str]],
) -> None:
    """Train and test folds are disjoint, non-empty, and cover the panel."""
    panel, _ = k_vintage_fixture
    split = temporal_split(panel)
    assert len(split.train_idx) > 0
    assert len(split.test_idx) > 0
    assert set(split.train_idx).isdisjoint(split.test_idx)
    assert len(split.train_idx) + len(split.test_idx) == panel.shape[0]


@pytest.mark.unit
def test_temporal_split_has_no_lookahead(
    k_vintage_fixture: tuple[pd.DataFrame, list[str]],
) -> None:
    """No train issue_d is after any test issue_d (the core invariant)."""
    panel, _ = k_vintage_fixture
    split = temporal_split(panel)
    train_max = panel.loc[split.train_idx, "issue_d"].max()
    test_min = panel.loc[split.test_idx, "issue_d"].min()
    assert train_max <= test_min
    # All train vintages precede or equal the cutoff; all test vintages exceed it.
    assert all(v <= split.cutoff for v in split.train_vintages)
    assert all(v > split.cutoff for v in split.test_vintages)


@pytest.mark.unit
def test_temporal_split_respects_explicit_cutoff(
    k_vintage_fixture: tuple[pd.DataFrame, list[str]],
) -> None:
    """An explicit cutoff puts that vintage (and earlier) in train, later in test."""
    panel, vintages = k_vintage_fixture
    cutoff = vintages[len(vintages) // 2]
    split = temporal_split(panel, cutoff=cutoff)
    assert (panel.loc[split.train_idx, "issue_d"] <= cutoff).all()
    assert (panel.loc[split.test_idx, "issue_d"] > cutoff).all()


@pytest.mark.unit
def test_temporal_split_test_size_targeting() -> None:
    """The auto-cutoff lands the test fraction near the requested test_size."""
    panel = generate_synthetic_panel(SyntheticConfig(n_loans=6_000, seed=42))
    split = temporal_split(panel, test_size=0.25)
    test_frac = len(split.test_idx) / panel.shape[0]
    assert 0.10 <= test_frac <= 0.45


@pytest.mark.unit
def test_assert_temporal_order_raises_on_lookahead() -> None:
    """A train row dated after a test row raises TemporalSplitError."""
    train = pd.DataFrame({"issue_d": ["2018-Q3"]})
    test = pd.DataFrame({"issue_d": ["2015-Q1"]})
    with pytest.raises(TemporalSplitError, match="look-ahead"):
        assert_temporal_order(train, test)


@pytest.mark.unit
def test_assert_temporal_order_passes_on_valid_split() -> None:
    """A correctly ordered split passes silently."""
    train = pd.DataFrame({"issue_d": ["2015-Q1", "2016-Q1"]})
    test = pd.DataFrame({"issue_d": ["2017-Q1", "2018-Q1"]})
    assert_temporal_order(train, test)  # no raise


@pytest.mark.unit
def test_temporal_split_rejects_single_vintage() -> None:
    """A panel with a single vintage cannot be split temporally."""
    df = pd.DataFrame({"issue_d": ["2015-Q1"] * 5})
    with pytest.raises(TemporalSplitError):
        temporal_split(df)


@pytest.mark.unit
def test_temporal_split_rejects_missing_column() -> None:
    """A missing issue column raises TemporalSplitError."""
    with pytest.raises(TemporalSplitError, match="issue_d"):
        temporal_split(pd.DataFrame({"x": [1, 2, 3]}))


@pytest.mark.unit
def test_temporal_split_rejects_empty_panel() -> None:
    """An empty panel raises TemporalSplitError."""
    with pytest.raises(TemporalSplitError, match="empty"):
        temporal_split(pd.DataFrame({"issue_d": pd.Series([], dtype=object)}))


@pytest.mark.unit
def test_temporal_split_cutoff_too_early_leaves_empty_train() -> None:
    """A cutoff before every vintage leaves an empty train fold and raises."""
    df = pd.DataFrame({"issue_d": ["2016-Q1", "2017-Q1", "2018-Q1"]})
    with pytest.raises(TemporalSplitError, match="empty train fold"):
        temporal_split(df, cutoff="2014-Q1")


@pytest.mark.unit
def test_temporal_split_cutoff_too_late_leaves_empty_test() -> None:
    """A cutoff at/after the final vintage leaves an empty test fold and raises."""
    df = pd.DataFrame({"issue_d": ["2016-Q1", "2017-Q1", "2018-Q1"]})
    with pytest.raises(TemporalSplitError, match="empty test fold"):
        temporal_split(df, cutoff="2018-Q1")


@pytest.mark.unit
def test_assert_temporal_order_rejects_empty_fold() -> None:
    """An empty fold cannot be order-checked; it raises TemporalSplitError."""
    nonempty = pd.DataFrame({"issue_d": ["2015-Q1"]})
    empty = pd.DataFrame({"issue_d": pd.Series([], dtype=object)})
    with pytest.raises(TemporalSplitError, match="non-empty"):
        assert_temporal_order(empty, nonempty)


@pytest.mark.unit
def test_assert_temporal_order_rejects_missing_column() -> None:
    """A fold missing the vintage column raises TemporalSplitError."""
    good = pd.DataFrame({"issue_d": ["2015-Q1"]})
    bad = pd.DataFrame({"x": [1]})
    with pytest.raises(TemporalSplitError, match="missing"):
        assert_temporal_order(good, bad)


# --------------------------------------------------------------------------- #
# load / coerce                                                               #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_load_panel_falls_back_to_synthetic() -> None:
    """With no data_path, load_panel returns a synthetic panel."""
    panel = load_panel(None, config=SyntheticConfig(n_loans=200, seed=1))
    assert panel.shape[0] == 200
    assert "loan_status" in panel.columns


@pytest.mark.unit
def test_load_panel_reads_real_csv(tmp_path: object) -> None:
    """A real CSV is read and dtype-coerced through the same contract."""
    raw = pd.DataFrame(
        {
            "loan_amnt": [10000, 20000],
            "term": [" 36 months", " 60 months"],
            "int_rate": ["13.5%", "18.2%"],
            "grade": ["B", "D"],
            "sub_grade": ["B2", "D4"],
            "emp_length": ["10+ years", "< 1 year"],
            "home_ownership": ["RENT", "MORTGAGE"],
            "annual_inc": [60000, 90000],
            "dti": [18.0, 25.0],
            "fico_range_low": [700, 680],
            "fico_range_high": [704, 684],
            "revol_util": ["45.0%", "80.1%"],
            "open_acc": [10, 8],
            "pub_rec": [0, 1],
            "purpose": ["car", "credit_card"],
            "addr_state": ["CA", "NY"],
            "installment": [330.0, 510.0],
            "verification_status": ["Verified", "Not Verified"],
            "issue_d": ["2015-Q1", "2016-Q3"],
            "loan_status": ["Fully Paid", "Charged Off"],
            "recoveries": [0.0, 1500.0],
        }
    )
    csv_path = tmp_path / "accepted.csv"  # type: ignore[operator]
    raw.to_csv(csv_path, index=False)

    panel = load_panel(csv_path)
    assert panel.shape[0] == 2
    # Percentage strings coerced to float.
    assert np.isclose(panel["int_rate"].iloc[0], 13.5)
    assert np.isclose(panel["revol_util"].iloc[1], 80.1)
    # term canonicalised; emp_length numeric.
    assert list(panel["term"]) == ["36 months", "60 months"]
    assert panel["emp_length"].iloc[0] == 10.0
    assert panel["emp_length"].iloc[1] == 0.0


@pytest.mark.unit
def test_load_panel_rejects_missing_columns(tmp_path: object) -> None:
    """A CSV missing required application columns raises ValidationError."""
    bad = pd.DataFrame({"loan_amnt": [1], "grade": ["A"]})
    csv_path = tmp_path / "bad.csv"  # type: ignore[operator]
    bad.to_csv(csv_path, index=False)
    with pytest.raises(ValidationError, match="missing required"):
        load_panel(csv_path)


@pytest.mark.unit
def test_load_panel_rejects_missing_file() -> None:
    """A non-existent path raises ValidationError."""
    with pytest.raises(ValidationError, match="does not exist"):
        load_panel("/nonexistent/path/accepted.csv")


@pytest.mark.unit
def test_coerce_dtypes_handles_numeric_inputs() -> None:
    """Already-numeric percent / emp_length columns pass through to float64."""
    df = pd.DataFrame(
        {
            "int_rate": [13.5, 18.2],  # numeric, not "13.5%"
            "revol_util": [45.0, 80.1],  # numeric
            "emp_length": [10, 0],  # numeric, not "10+ years"
            "term": ["36 months", "60 months"],
        }
    )
    out = coerce_dtypes(df)
    assert out["int_rate"].dtype == "float64"
    assert out["emp_length"].dtype == "float64"
    assert list(out["emp_length"]) == [10.0, 0.0]


@pytest.mark.unit
def test_coerce_dtypes_is_idempotent_on_synthetic(synthetic_panel: pd.DataFrame) -> None:
    """Coercing an already-clean synthetic panel preserves it and stays pure."""
    coerced = coerce_dtypes(synthetic_panel)
    assert coerced.shape == synthetic_panel.shape
    assert list(coerced["term"].unique()) == list(synthetic_panel["term"].unique())
    # Input untouched (a copy was returned).
    assert synthetic_panel["int_rate"].dtype == coerced["int_rate"].dtype


# --------------------------------------------------------------------------- #
# end-to-end data path (load -> drop leakage -> labels -> split)              #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_full_data_path_is_leakage_free_and_ordered() -> None:
    """The full data path drops leakage AND honours temporal order together."""
    panel = load_panel(None, config=SyntheticConfig(n_loans=3_000, seed=8))
    cleaned = drop_leakage(panel)
    # issue_d survives the leakage drop (it is an application-time field).
    assert "issue_d" in cleaned.columns
    labelled = build_labels(panel)  # labels built on the panel with loan_status
    split = temporal_split(labelled.panel)
    train = labelled.panel.loc[split.train_idx]
    test = labelled.panel.loc[split.test_idx]
    assert_temporal_order(train, test)
    # And the cleaned (leakage-free) feature frame passes the no-leakage backstop.
    assert_no_leakage(drop_leakage(labelled.panel))
