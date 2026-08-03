"""audit-v2 (P0-7) acceptance tests: the CMIP6 downloader's validation must
reject incomplete/malformed files instead of accepting them as complete,
and historical/future pairing must require a shared model set."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pytest

from download_cmip6 import MODELS, REQUIRED_VARIABLES, DataValidationError, common_models, validate_cmip6_frame

START, END = "1991-01-01", "2010-12-31"


def _make_frame(**kwargs) -> pd.DataFrame:
    dates = pd.date_range(START, END, freq="D")
    n = len(dates)
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d")})
    for var in REQUIRED_VARIABLES:
        for model in MODELS:
            df[f"{var}_{model}"] = np.ones(n)
    if kwargs.get("empty_var_model"):
        df[f"{kwargs['empty_var_model']}"] = np.nan
    if kwargs.get("drop_dates"):
        df = df.iloc[1:]
    if kwargs.get("duplicate_dates"):
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    if kwargs.get("drop_column"):
        df = df.drop(columns=[kwargs["drop_column"]])
    return df


def test_valid_frame_passes_and_reports_coverage():
    df = _make_frame()
    coverage = validate_cmip6_frame(df, START, END)
    assert len(coverage) == len(REQUIRED_VARIABLES) * len(MODELS)
    assert all(v == 1.0 for v in coverage.values())


def test_empty_required_column_is_recorded_not_hidden():
    """audit-v2 (P0-7): an entirely-empty required variable-model column
    (the actual CMCC_CM2_VHR4 shortwave_radiation hole) does NOT hard-fail
    the file - the re-download probe proved the gap is upstream (every
    fresh API response has it), so a hard failure would kill the whole
    dataset. Instead coverage[col] must be 0.0 so the manifest can carry
    validation_status='degraded_model_gaps' and the consumer can drop the
    model per-variable while REPORTING the ensemble size."""
    col = "shortwave_radiation_sum_CMCC_CM2_VHR4"
    df = _make_frame(empty_var_model=col)
    coverage = validate_cmip6_frame(df, START, END)
    assert coverage[col] == 0.0
    assert all(v == 1.0 for k, v in coverage.items() if k != col)


def test_missing_dates_raise():
    with pytest.raises(DataValidationError, match="date range mismatch"):
        validate_cmip6_frame(_make_frame(drop_dates=True), START, END)


def test_duplicate_dates_raise():
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_cmip6_frame(_make_frame(duplicate_dates=True), START, END)


def test_missing_column_is_recorded_not_hidden():
    col = "wind_speed_10m_mean_EC_Earth3P_HR"
    df = _make_frame(drop_column=col)
    coverage = validate_cmip6_frame(df, START, END)
    assert coverage[col] == 0.0


def test_common_models_requires_both_periods():
    df_hist = _make_frame()
    df_fut = _make_frame(empty_var_model="temperature_2m_max_MPI_ESM1_2_XR")
    common = common_models(df_hist, df_fut, "temperature_2m_max")
    assert "MPI_ESM1_2_XR" not in common
    assert len(common) == len(MODELS) - 1
