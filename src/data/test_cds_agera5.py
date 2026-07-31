"""One-off smoke test: confirm the CDS API key can authenticate and pull
a minimal AgERA5 slice (used once to validate access, not part of the
regular pipeline)."""

import cdsapi

c = cdsapi.Client()

c.retrieve(
    "sis-agrometeorological-indicators",
    {
        "version": "1_1",
        "variable": "2m_temperature",
        "statistic": "24_hour_maximum",
        "year": "2020",
        "month": "01",
        "day": ["01"],
        "area": [40, 116, 39.5, 116.5],  # small box around Beijing site
        "format": "zip",
    },
    "cds_test_output.zip",
)
print("download complete: cds_test_output.zip")
