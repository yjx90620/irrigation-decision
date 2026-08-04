"""audit-v3 triage: two-factor correlation without ningxia + gap_days boundary."""
import numpy as np
import pandas as pd

s = pd.read_csv("data/processed/leave_one_out_rotation_summary_wq.csv")
r = pd.read_csv("data/processed/two_factor_risk.csv")
m = s.merge(r, left_on="target_site", right_on="site_id")
gap, risk = m["yield_gap_vs_rule"], m["two_factor_risk"]

full = np.corrcoef(gap, risk)[0, 1]
drop = {}
for site in m["target_site"]:
    keep = m["target_site"] != site
    drop[site] = round(np.corrcoef(gap[keep], risk[keep])[0, 1], 3)
print(f"corr(gap, risk) full: {full:.3f}")
print("leave-one-out corr:")
for k, v in drop.items():
    print(f"  drop {k}: {v}")
print(f"Spearman full: {pd.Series(gap).corr(pd.Series(risk), method='spearman'):.3f}")

# gap_days boundary check in rotation.py
import re
src = open("src/sim/rotation.py", encoding="utf-8").read()
m2 = re.search(r"gap_days < 0", src)
print(f"\nrotation.py 'gap_days < 0': {'FOUND (allows equal dates)' if m2 else 'not found'}")
m3 = re.search(r"MIN_HANDOFF_GAP_DAYS", src)
print(f"MIN_HANDOFF_GAP_DAYS: {'present' if m3 else 'absent'}")
