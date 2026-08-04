"""Early wq verdict: per_quota vs per_action direct arm (balanced preference)."""
import pandas as pd

per = pd.read_csv("data/processed/rotation_policy_comparison.csv")
wq = pd.read_csv("data/processed/ppo_rotation_wq_comparison.csv")
for df in (per, wq):
    df["yield"] = df["total_yield_t_ha"].astype(float)
    df["irr"] = df["total_irrigation_mm"].astype(float)

b = per[per["preference"] == "balanced"]
w = wq[wq["preference"] == "balanced"]
print("=== per_action (main) ===")
print(b.groupby("policy")[["yield", "irr"]].mean().round(2).to_string())
print("\n=== per_quota (wq, direct 2 seeds) ===")
print(w.groupby("policy")[["yield", "irr"]].mean().round(2).to_string())
print("\n--- site-level wq direct ---")
print(w[w["policy"] == "ppo_direct_wq"].groupby("site_id")[["yield", "irr"]].mean().round(2).to_string())
