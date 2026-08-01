"""Two-factor transfer risk predictor (paper 3's core innovation).

Prototype-stage evidence for why one factor isn't enough: Shaanxi
Guanzhong had a large environmental distance to its nearest source
(4.22, second only to Ningxia's 5.10) yet transferred *better* than any
other site (yield gap vs the rule baseline: 0.06 t/ha, smallest of all
five) - because that site's system yield barely depends on irrigation
policy quality in the first place (rainfed nearly matches full
irrigation there). Distance alone would flag it as high-risk; it isn't.

risk = distance x sensitivity

where distance is environmental distance to the nearest source site
(site_distance_matrix.csv, from src/transfer/similarity_analysis.py - the
climate features it's built from are unaffected by the crop-rotation
upgrade, so that matrix is still valid) and sensitivity is
task_sensitivity.py's (full_irrigation - rainfed) / full_irrigation gap
under the new rotation setup.

The multiplicative form encodes the mechanism directly: a source policy
transplanted to a *similar* environment should do fine regardless of how
sensitive the target is (small distance keeps risk low even at high
sensitivity), and a source policy transplanted to a *dissimilar* but
insensitive environment should also do fine (small sensitivity keeps risk
low even at high distance) - it's the combination that's dangerous.

validate_against_observed_gap() checks this against real transfer results
once leave_one_out_rotation.py has produced them - not run yet, this
module is prepared ahead of that.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def load_distance_to_nearest_source(target: str) -> float:
    dist = pd.read_csv(DATA_DIR / "site_distance_matrix.csv", index_col="site_id")
    others = dist.loc[target].drop(target)
    return others.min()


def build_risk_table() -> pd.DataFrame:
    sensitivity = pd.read_csv(DATA_DIR / "task_sensitivity.csv", index_col="site_id")
    rows = []
    for site_id in sensitivity.index:
        distance = load_distance_to_nearest_source(site_id)
        sens = sensitivity.loc[site_id, "sensitivity"]
        rows.append({
            "site_id": site_id,
            "nearest_source_distance": distance,
            "task_sensitivity": sens,
            "two_factor_risk": distance * sens,
        })
    return pd.DataFrame(rows).sort_values("two_factor_risk", ascending=False)


def validate_against_observed_gap(transfer_results_path: str, gap_column: str = "yield_gap_vs_rule"):
    """Correlate predicted risk against measured transfer gaps once
    leave_one_out_rotation.py has run. Reports both the distance-only and
    two-factor correlations so the comparison in paper 3's 对比1 (单因子 vs
    双因子) is a one-line call once the data exists."""
    risk = build_risk_table().set_index("site_id")
    observed = pd.read_csv(transfer_results_path, index_col="target_site")
    merged = risk.join(observed[gap_column])

    return {
        "distance_only_corr": merged["nearest_source_distance"].corr(merged[gap_column]),
        "two_factor_corr": merged["two_factor_risk"].corr(merged[gap_column]),
        "table": merged,
    }


if __name__ == "__main__":
    table = build_risk_table()
    print(table.round(3).to_string(index=False))
    out_path = DATA_DIR / "two_factor_risk.csv"
    table.to_csv(out_path, index=False)
    print(f"\nsaved -> {out_path}")
