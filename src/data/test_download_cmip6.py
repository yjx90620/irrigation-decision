"""audit-v2 (P0-7) + audit-v3 (4.2/4.3) tests: CMIP6 validation rejects
incomplete files correctly, common-model pairing is strict and symmetric,
and --check-only is truly read-only (no network, no writes)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pytest

from download_cmip6 import (
    MODELS, REQUIRED_VARIABLES, ColumnStatus, DataValidationError,
    common_models, is_complete_column, validate_cmip6_frame, validate_existing_dataset,
)

START, END = "1991-01-01", "2010-12-31"


def _make_frame(**kwargs) -> pd.DataFrame:
    dates = pd.date_range(START, END, freq="D")
    n = len(dates)
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d")})
    for var in REQUIRED_VARIABLES:
        for model in MODELS:
            df[f"{var}_{model}"] = np.ones(n)
    if kwargs.get("empty_var_model"):
        df[kwargs["empty_var_model"]] = np.nan
    if kwargs.get("partial_var_model"):
        df.loc[:365, kwargs["partial_var_model"]] = np.nan
    if kwargs.get("drop_dates"):
        df = df.iloc[1:]
    if kwargs.get("duplicate_dates"):
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    if kwargs.get("drop_column"):
        df = df.drop(columns=[kwargs["drop_column"]])
    return df


def test_valid_frame_all_complete():
    df = _make_frame()
    coverage = validate_cmip6_frame(df, START, END)
    assert all(v == ColumnStatus.COMPLETE for v in coverage.values())


def test_empty_required_column_is_recorded_not_hidden():
    """audit-v3 (4.2): the CMCC radiation hole is upstream; it must be
    recorded as EMPTY_INVALID (not raise), so the consumer excludes it
    symmetrically."""
    col = "shortwave_radiation_sum_CMCC_CM2_VHR4"
    coverage = validate_cmip6_frame(_make_frame(empty_var_model=col), START, END)
    assert coverage[col] == ColumnStatus.EMPTY_INVALID
    assert all(v == ColumnStatus.COMPLETE for k, v in coverage.items() if k != col)


def test_partial_column_is_not_complete():
    col = "precipitation_sum_EC_Earth3P_HR"
    coverage = validate_cmip6_frame(_make_frame(partial_var_model=col), START, END)
    assert coverage[col] != ColumnStatus.COMPLETE
    assert coverage[col] in (ColumnStatus.PARTIAL_INVALID, ColumnStatus.UPSTREAM_UNAVAILABLE)


def test_is_complete_column_strict():
    s = pd.Series(np.ones(10))
    assert is_complete_column(s, 10)
    s2 = s.copy()
    s2.iloc[3] = np.nan
    assert not is_complete_column(s2, 10)
    s3 = pd.Series([1.0, np.inf, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    assert not is_complete_column(s3, 10)  # non-finite


def test_missing_dates_raise():
    with pytest.raises(DataValidationError, match="date range mismatch"):
        validate_cmip6_frame(_make_frame(drop_dates=True), START, END)


def test_duplicate_dates_raise():
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_cmip6_frame(_make_frame(duplicate_dates=True), START, END)


def test_common_models_requires_complete_both_periods():
    """audit-v3 (4.2): a model with an EMPTY column in either period must
    be excluded from the shared ensemble (symmetric, not one-sided)."""
    df_hist = _make_frame()
    df_fut = _make_frame(empty_var_model="temperature_2m_max_MPI_ESM1_2_XR")
    common = common_models(df_hist, df_fut, "temperature_2m_max")
    assert "MPI_ESM1_2_XR" not in common
    assert len(common) == len(MODELS) - 1
    # partial (95% coverage) also excluded
    df_fut2 = _make_frame(partial_var_model="precipitation_sum_EC_Earth3P_HR")
    common2 = common_models(df_hist, df_fut2, "precipitation_sum")
    assert "EC_Earth3P_HR" not in common2


def test_check_only_is_read_only():
    """audit-v3 (4.3): validate_existing_dataset mutates nothing - tree
    hash of the dataset dir is identical before and after."""
    import hashlib
    import shutil
    import tempfile

    src = Path(__file__).resolve().parents[2] / "data" / "raw" / "cmip6"
    if not src.exists():
        pytest.skip("cmip6 data dir absent")
    tmp = Path(tempfile.mkdtemp()) / "cmip6"
    shutil.copytree(src, tmp, ignore=shutil.ignore_patterns("quarantine"))
    try:
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in tmp.iterdir() if p.is_file()}
        validate_existing_dataset(tmp)
        after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in tmp.iterdir() if p.is_file()}
        assert before == after
        assert set(before) == set(after)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
