"""Atomic file I/O and hashing helpers (audit-v2, P0-7/P0-13).

No data or result file may ever be written in place: write to a temp
sibling and atomically replace, so a crash mid-write can never leave a
half-written file that a later `if path.exists(): skip` would mistake for
a completed artifact.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def sha256_file(path) -> str:
    """SHA-256 of a file's bytes - used in manifests to pin which exact
    input/config produced a result."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
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


def atomic_write_text(path, text: str) -> Path:
    return _atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path, obj) -> Path:
    return atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def atomic_write_csv(df, path) -> Path:
    """DataFrame -> CSV via a temp file and atomic replace."""
    import pandas as pd

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            df.to_csv(fh, index=False)
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
