"""audit-v3 (2.2): result contract - duplicate/missing keys, finiteness,
atomic writes (no partial file survives a crash)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import pytest

from result_contract import ResultContractError, ResultSpec, atomic_write_csv, validate_result_frame

SPEC = ResultSpec(
    key_columns=("policy", "site"),
    required_columns=("policy", "site", "yield_t_ha"),
    expected_keys=frozenset({("a", "s1"), ("a", "s2"), ("b", "s1")}),
    finite_columns=("yield_t_ha",),
)


def _frame():
    return pd.DataFrame([
        {"policy": "a", "site": "s1", "yield_t_ha": 1.0},
        {"policy": "a", "site": "s2", "yield_t_ha": 2.0},
        {"policy": "b", "site": "s1", "yield_t_ha": 3.0},
    ])


def test_valid_frame_passes():
    validate_result_frame(_frame(), SPEC)


def test_duplicate_keys_rejected():
    df = pd.concat([_frame(), _frame().iloc[[0]]], ignore_index=True)
    with pytest.raises(ResultContractError, match="Duplicate"):
        validate_result_frame(df, SPEC)


def test_missing_keys_rejected():
    df = _frame().iloc[:-1]  # drop ('b','s1')
    with pytest.raises(ResultContractError, match="key mismatch"):
        validate_result_frame(df, SPEC)


def test_unexpected_keys_rejected():
    df = pd.concat([_frame(), pd.DataFrame([{"policy": "z", "site": "s9", "yield_t_ha": 1.0}])],
                   ignore_index=True)
    with pytest.raises(ResultContractError, match="key mismatch"):
        validate_result_frame(df, SPEC)


def test_missing_required_column_rejected():
    with pytest.raises(ResultContractError, match="Missing columns"):
        validate_result_frame(_frame().drop(columns=["yield_t_ha"]), SPEC)


def test_non_finite_rejected():
    df = _frame()
    df.loc[0, "yield_t_ha"] = float("nan")
    with pytest.raises(ResultContractError, match="Non-finite"):
        validate_result_frame(df, SPEC)


def test_atomic_write_replaces_completely():
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    try:
        out = tmp / "result.csv"
        atomic_write_csv(_frame(), out)
        reread = pd.read_csv(out)
        validate_result_frame(reread, SPEC)
        assert len(reread) == 3
        # no temp files left behind
        leftovers = [p for p in tmp.iterdir() if p.name != "result.csv"]
        assert leftovers == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
