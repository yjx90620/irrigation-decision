"""audit-v3 (8): legacy isolation - the DEPRECATED single-season prototype
modules (env.py, gym_env.py, evaluate_policy.py, train_utils.py,
train_ppo.py) must never be imported by the rotation-era pipeline, and the
legacy data-producing scripts must refuse to run without --allow-legacy.
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

LEGACY_IMPORT = re.compile(r"\bfrom (env|gym_env|evaluate_policy|train_utils|train_ppo)\b|"
                           r"\bimport (env|gym_env|evaluate_policy|train_utils|train_ppo)\b")

# the rotation-era pipeline - files whose outputs feed the papers
ROTATION_ERA_FILES = [
    "src/rl/rotation_env.py",
    "src/rl/residual_gym_env.py",
    "src/rl/train_rotation_compare.py",
    "src/rl/train_rotation_utils.py",
    "src/rl/experiment_config.py",
    "src/rl/run_training_with_watchdog.py",
    "src/transfer/leave_one_out_rotation.py",
    "src/transfer/two_factor_predictor.py",
    "src/transfer/fingerprint.py",
    "src/transfer/similarity_analysis.py",
    "src/transfer/task_sensitivity.py",
    "src/sim/rotation.py",
    "src/sim/optimize_cross_season.py",
    "src/sim/optimize_algorithm_comparison.py",
    "src/sim/scan_allocation.py",
    "src/sim/marginal_water_value_v2.py",
    "src/sim/cropping_systems.py",
    "src/sim/temporal_split.py",
    "src/sim/weather.py",
    "src/sim/et0.py",
    "src/utils/result_contract.py",
]

LEGACY_DATA_SCRIPTS = [
    "src/rl/train_ppo.py",
    "src/rl/reconstruct_learning_curve.py",
    "src/transfer/leave_one_out.py",
    "src/transfer/instance_weighted.py",
]


def test_rotation_era_pipeline_has_no_legacy_imports():
    offenders = []
    for rel in ROTATION_ERA_FILES:
        path = ROOT / rel
        if not path.exists():
            offenders.append(f"{rel} (missing)")
            continue
        text = path.read_text(encoding="utf-8")
        if LEGACY_IMPORT.search(text):
            offenders.append(rel)
    assert not offenders, f"rotation-era files importing legacy modules: {offenders}"


def test_legacy_data_scripts_require_allow_legacy():
    for rel in LEGACY_DATA_SCRIPTS:
        path = ROOT / rel
        text = path.read_text(encoding="utf-8")
        assert "--allow-legacy" in text, f"{rel} has no --allow-legacy guard"
        assert "DEPRECATED" in text or "P0-12" in text, f"{rel} lacks the deprecation marker"
