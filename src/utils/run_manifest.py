"""Run manifests and content-based completion checks (audit-v2, P0-13).

A result file's existence is NOT completion. Anything that writes a
result should write a manifest alongside it (status, hashes, seed,
outputs), and any skip logic must consult the manifest / the file's
content - not Path.exists() - so a failed, timeout, NaN-filled or
half-written result is never silently treated as done.

Also provides csv_result_complete(), the minimal content-based check used
to harden scripts that previously skipped on `if out_path.exists()`:
a result counts as complete only when it has the required columns, enough
rows, and no NaN in them.
"""

import json
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from atomic_io import atomic_write_json, sha256_file, sha256_text

ALLOWED_STATUSES = ("pending", "running", "complete", "failed", "timeout", "invalid", "cancelled")


def git_commit() -> str:
    """Current repo commit, if any - recorded in manifests so a result can
    always be traced back to the code that produced it."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
        return out
    except Exception:
        return ""


@dataclass
class RunManifest:
    run_id: str
    task_name: str
    code_commit: str = field(default_factory=git_commit)
    config_sha256: str = ""
    input_sha256: dict = field(default_factory=dict)  # {input_path: sha256}
    seed: int | None = None
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None
    status: str = "pending"
    error_type: str | None = None
    error_message: str | None = None
    output_files: list = field(default_factory=list)

    def __post_init__(self):
        if self.status not in ALLOWED_STATUSES:
            raise ValueError(f"invalid status '{self.status}', allowed: {ALLOWED_STATUSES}")

    def finish(self, status: str = "complete", error_type=None, error_message=None):
        self.status = status
        self.error_type = error_type
        self.error_message = error_message
        self.finished_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return asdict(self)


def save_manifest(manifest: RunManifest, path) -> None:
    atomic_write_json(path, manifest.to_dict())


def load_manifest(path) -> RunManifest | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return RunManifest(**{k: v for k, v in data.items() if k in RunManifest.__dataclass_fields__})
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def manifest_is_complete(manifest: RunManifest, code_commit: str, config_sha256: str,
                         input_hashes: dict) -> bool:
    """The skip predicate the report's rule requires: a manifest only
    satisfies it when the run finished, the code commit matches the
    current one, the config hash matches, and every declared input still
    hashes to the recorded value."""
    if manifest is None or manifest.status != "complete":
        return False
    if code_commit and manifest.code_commit != code_commit:
        return False
    if config_sha256 and manifest.config_sha256 != config_sha256:
        return False
    for key, expected in input_hashes.items():
        if manifest.input_sha256.get(key) != expected:
            return False
    return True


def csv_result_complete(path, required_columns, min_rows=1, forbid_nan=True) -> bool:
    """Content-based completion check for CSV results: the file must exist,
    contain every required column, have >= min_rows rows, and (when
    forbid_nan) no NaN in the required columns. Returns False for a
    missing, malformed, empty or NaN-filled result - the caller should
    then re-run instead of skipping."""
    import pandas as pd

    path = Path(path)
    if not path.exists():
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        return False
    if len(df) < min_rows:
        return False
    if forbid_nan and df[required_columns].isna().any().any():
        return False
    return True
