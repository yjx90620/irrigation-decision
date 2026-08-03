import pandas as pd

df = pd.read_csv("data/processed/leave_one_out_rotation_transfer.csv")
df["yield"] = df["total_yield_t_ha"].astype(float)
g = df.groupby(["target_site", "condition"])["yield"].mean().unstack().round(2)
print(g.to_string())
print()
if "threshold_rule" in g.columns:
    gap = (g["threshold_rule"] - g["zero_shot"]).round(2)
    print("zero-shot yield gap vs rule:")
    print(gap.to_string())
