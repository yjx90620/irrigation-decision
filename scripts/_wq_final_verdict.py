"""audit-v2 P0-9 FINAL verdict: per_quota (reward-rebalanced) vs per_action
(primary) - balanced preference headline, site-level, and preference
conditioning check."""
import pandas as pd

per = pd.read_csv("data/processed/rotation_policy_comparison.csv")
wq = pd.read_csv("data/processed/ppo_rotation_wq_comparison.csv")
for df in (per, wq):
    df["yield"] = df["total_yield_t_ha"].astype(float)
    df["irr"] = df["total_irrigation_mm"].astype(float)

b = per[per["preference"] == "balanced"]
w = wq[wq["preference"] == "balanced"]

print("=== BALANCED preference: mean across 5 sites x 5 years x 3 seeds ===")
rows = {}
for label, df in [("per_action", b), ("per_quota_wq", w)]:
    for pol in df["policy"].unique():
        sub = df[df["policy"] == pol]
        rows[f"{label} {pol}"] = (sub["yield"].mean(), sub["irr"].mean())
t = pd.DataFrame(rows, index=["yield_t_ha", "irr_mm"]).T.round(2)
print(t.to_string())

print("\n=== site-level yields (balanced) ===")
print("--- per_action ---")
print(b.groupby(["policy", "site_id"])["yield"].mean().unstack().round(2).to_string())
print("--- per_quota_wq ---")
print(w.groupby(["policy", "site_id"])["yield"].mean().unstack().round(2).to_string())

print("\n=== preference conditioning (wq) ===")
print(wq.groupby(["preference", "policy"])[["yield", "irr"]].mean().round(2).to_string())

print("\n=== water-saving vs rules (balanced, wq) ===")
rules = b[b["policy"].isin(["quota_reserving_rule", "threshold_rule"])]
for pol in w["policy"].unique():
    sub = w[w["policy"] == pol]
    y_gap = rules[rules["policy"] == "quota_reserving_rule"]["yield"].mean() - sub["yield"].mean()
    w_save = (rules[rules["policy"] == "quota_reserving_rule"]["irr"].mean() - sub["irr"].mean()) / \
             rules[rules["policy"] == "quota_reserving_rule"]["irr"].mean()
    print(f"{pol}: yield gap vs quota_rule = {y_gap:.2f} t/ha; water saving = {w_save:.0%}")
