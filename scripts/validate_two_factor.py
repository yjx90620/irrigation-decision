"""audit-v2: two-factor risk validation - correlation of predicted risk
with the ACTUAL transfer yield gap, vs distance alone (paper-3 core claim)."""
import numpy as np
import pandas as pd

summary = pd.read_csv("data/processed/leave_one_out_rotation_summary.csv")
risk = pd.read_csv("data/processed/two_factor_risk.csv")

risk_col = "site_id" if "site_id" in risk.columns else "target_site"
m = summary.merge(risk, left_on="target_site", right_on=risk_col).rename(columns={"target_site": "site"})
m["gap"] = m["yield_gap_vs_rule"]

print(m[["site", "nearest_source_distance", "task_sensitivity", "two_factor_risk", "gap"]].round(3).to_string(index=False))
print()

r_risk = np.corrcoef(m["gap"], m["two_factor_risk"])[0, 1]
r_dist = np.corrcoef(m["gap"], m["nearest_source_distance"])[0, 1]
r_sens = np.corrcoef(m["gap"], m["task_sensitivity"])[0, 1]
print(f"corr(gap, two_factor_risk) = {r_risk:.3f}")
print(f"corr(gap, distance)        = {r_dist:.3f}")
print(f"corr(gap, sensitivity)     = {r_sens:.3f}")
print(f"n = {len(m)} (direction-of-evidence only, per audit P1-6)")
