"""wq site-level table for 论文二 对比三 rewrite."""
import pandas as pd

wq = pd.read_csv("data/processed/ppo_rotation_wq_comparison.csv")
per = pd.read_csv("data/processed/rotation_policy_comparison.csv")
for df in (wq, per):
    df["yield"] = df["total_yield_t_ha"].astype(float)
    df["irr"] = df["total_irrigation_mm"].astype(float)

SITE_ORDER = ["shaanxi_guanzhong", "hebei_central", "beijing_plain", "henan_north", "ningxia_irrigation"]

for name, df in [("per_action(旧主臂)", per), ("per_quota_wq(新主臂)", wq)]:
    print(f"\n=== {name} ===")
    b = df[df["preference"] == "balanced"]
    for pol in ["threshold_rule", "quota_reserving_rule", "ppo_direct", "ppo_residual"]:
        key = pol if pol in b["policy"].values else f"{pol}_wq"
        if key not in b["policy"].values:
            continue
        sub = b[b["policy"] == key]
        print(f"--- {key} ---")
        g = sub.groupby("site_id").agg(y=("yield", "mean"), s=("yield", "std"), i=("irr", "mean")).round(2)
        for site in SITE_ORDER:
            if site in g.index:
                r = g.loc[site]
                print(f"  {site:22s} {r['y']}±{r['s']} / {r['i']:.0f}mm")
