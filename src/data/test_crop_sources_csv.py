"""audit-v3 (2.5): crop_parameter_sources.csv must parse cleanly with the
declared 12-column schema and a source-type taxonomy per row."""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

CSV_PATH = Path(__file__).resolve().parents[2] / "data" / "calibration" / "crop_parameter_sources.csv"

ALLOWED_STATUS = {"literature", "calibrated", "scenario_assumption", "default"}


def test_csv_parses_with_consistent_field_counts():
    with open(CSV_PATH, encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    header = rows[0]
    assert len(header) == 12
    for i, row in enumerate(rows[1:], start=2):
        assert len(row) == len(header), f"row {i}: {len(row)} fields != {len(header)}"


def test_status_taxonomy_is_valid():
    with open(CSV_PATH, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) >= 15
    for row in rows:
        status = row["calibration_status"]
        # statuses may carry annotations: "literature/stock", "default validated",
        # "calibrated (ratio-preserving)", "literature/scenario assumption (...)"
        base = status.split()[0].split("(")[0].split("/")[0].strip()
        assert base in ALLOWED_STATUS, status


def test_summer_maize_is_scenario_assumption_not_site_calibrated():
    """audit-v3: the 105-day summer maize must NOT claim site-calibrated
    validation - it is a literature/scenario assumption."""
    with open(CSV_PATH, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        if row["parameter"] == "MaturityCD" and row["crop"] == "Maize" and row["code_value"] == "105":
            assert "scenario" in row["calibration_status"].lower(), row["calibration_status"]
            assert "calibrated" not in row["calibration_status"].lower() or "scenario" in row["calibration_status"]
