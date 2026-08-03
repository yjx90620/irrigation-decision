PROTOTYPE — NOT VALID FOR FINAL SCIENTIFIC CLAIMS
=================================================

This directory (data/processed/invalidated/legacy_v1/ and
figures/invalidated/legacy_v1/) holds the OLD SINGLE-SEASON prototype
results of this project:

- single-crop simulations that started each season near field capacity
  (no rotation chain, no annual water quota, no fallow bridge),
- the pre-upgrade PPO policies and learning curves trained on that env,
- the single-season transfer (leave_one_out_transfer / instance_weighted /
  source_excl / finetuned models) and its figures (fig1-fig6).

The research premise behind them was invalidated by the system upgrade
(docs/系统升级方案.md) and by the audit-v2 fixes (docs/AUDIT_FIX_LOG.md,
P0-1/P0-2): with no rotation and no quota there was almost no irrigation
signal, and 4 of 5 sites reached 93-99% of full-irrigation yield with
zero irrigation. They are archived for reproducibility of the development
record ONLY - they must not be cited as current evidence in any paper.

Migration record with per-file SHA-256: migration_manifest.csv
