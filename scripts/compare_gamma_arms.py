"""audit-v2 P0-9: compare the gamma=1.0 ablation arm against the primary
gamma=0.995 arm on the balanced-preference evaluation, and print the
verdict the paper needs: does removing the discounting close the yield
gap vs the rule baselines?

Reads rotation_policy_comparison.csv (gamma=0.995, primary) and
ppo_rotation_gamma1_comparison.csv (gamma=1.0, ablation). Also reports
the device asymmetry (primary trained CPU, ablation CUDA - see
docs/AUDIT_FIX_LOG.md) so the final numbers can be device-confirmed.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import pandas as pd

PRIMARY = ROOT / "data" / "processed" / "rotation_policy_comparison.csv"
ABLATION = ROOT / "data" / "processed" / "ppo_rotation_gamma1_comparison.csv"


def load(path):
    df = pd.read_csv(path)
    df = df[df["preference"] == "balanced"]
    df["yield"] = df["total_yield_t_ha"].astype(float)
    df["irr"] = df["total_irrigation_mm"].astype(float)
    return df


def main():
    primary = load(PRIMARY)
    ablation = load(ABLATION)

    def summarize(df, labels):
        g = df.groupby("policy")[["yield", "irr"]].agg(["mean", "std"]).round(2)
        return g.rename(columns=lambda c: f"{c[0]}_{c[1]}" if isinstance(c, tuple) else c)

    print("=== gamma=0.995 (primary, CPU) ===")
    print(summarize(primary, None).to_string())
    print("\n=== gamma=1.0 (ablation, CUDA) ===")
    print(summarize(ablation, None).to_string())

    # site-level yield comparison for the gamma arms
    def site_means(df):
        return df.groupby(["policy", "site_id"])["yield"].mean().unstack().round(2)

    print("\n=== site-level yields: 0.995 ===")
    print(site_means(primary).to_string())
    print("\n=== site-level yields: 1.0 ===")
    print(site_means(ablation).to_string())

    # verdict (ablation policies carry the _gamma1 label)
    p_direct = primary[primary["policy"] == "ppo_direct"]["yield"].mean()
    a_direct = ablation[ablation["policy"] == "ppo_direct_gamma1"]["yield"].mean()
    p_res = primary[primary["policy"] == "ppo_residual"]["yield"].mean()
    a_res = ablation[ablation["policy"] == "ppo_residual_gamma1"]["yield"].mean()
    rule = primary[primary["policy"] == "quota_reserving_rule"]["yield"].mean()
    print("\n=== verdict ===")
    print(f"direct:  0.995 {p_direct:.2f} -> 1.0 {a_direct:.2f}  (rule {rule:.2f})")
    print(f"residual: 0.995 {p_res:.2f} -> 1.0 {a_res:.2f}  (rule {rule:.2f})")
    gap_close_d = (a_direct - p_direct) / max(rule - p_direct, 1e-9)
    gap_close_r = (a_res - p_res) / max(rule - p_res, 1e-9)
    print(f"yield-gap-to-rule closed: direct {gap_close_d:.0%}, residual {gap_close_r:.0%}")


if __name__ == "__main__":
    main()
