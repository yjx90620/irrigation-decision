"""Formal result contract (audit-v3, 任务 2.2).

A result file's existence (or even a run manifest saying "complete") is
not enough: the output must have the expected schema, exactly the
expected key set, no duplicate keys, and finite required metrics. Every
formal CSV write goes through atomic_write_csv (temp file + os.replace),
so a crash can never leave a half-written file that a completion check
would accept.
"""

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ResultSpec:
    key_columns: tuple[str, ...]
    required_columns: tuple[str, ...]
    expected_keys: frozenset[tuple]
    finite_columns: tuple[str, ...] = ()


class ResultContractError(ValueError):
    """A formal result violates its schema/keys/finiteness contract."""


def validate_result_frame(frame: pd.DataFrame, spec: ResultSpec) -> None:
    """Strict schema + key-set + finiteness validation (audit-v3 2.2)."""
    missing_columns = set(spec.required_columns) - set(frame.columns)
    if missing_columns:
        raise ResultContractError(f"Missing columns: {sorted(missing_columns)}")

    duplicated = frame.duplicated(list(spec.key_columns), keep=False)
    if duplicated.any():
        rows = frame.loc[duplicated, list(spec.key_columns)]
        raise ResultContractError(f"Duplicate result keys:\n{rows}")

    actual_keys = frozenset(
        map(tuple, frame.loc[:, list(spec.key_columns)].itertuples(index=False, name=None))
    )
    if actual_keys != spec.expected_keys:
        missing = spec.expected_keys - actual_keys
        unexpected = actual_keys - spec.expected_keys
        raise ResultContractError(
            f"Result key mismatch. Missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )

    for column in spec.finite_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if not values.notna().all():
            raise ResultContractError(f"Non-finite values in {column}")


def atomic_write_csv(frame: pd.DataFrame, path) -> Path:
    """Temp-file + os.replace atomic write (audit-v3 2.2/2.4)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            frame.to_csv(fh, index=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return path


def read_csv_checked(path, spec: ResultSpec) -> pd.DataFrame:
    """Read + validate in one step; raises on any contract violation."""
    frame = pd.read_csv(path)
    validate_result_frame(frame, spec)
    return frame
