"""audit-v3 (6.1): leave-one-out transfer arms - target-budget bookkeeping
for the 5-condition fold (zero_shot / finetuned / scratch /
full_target_expert / threshold_rule). Pure constant/mapping checks; the
RL training itself is covered by the full re-run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from leave_one_out_rotation import (
    EXPERT_STEPS,
    FINETUNE_STEPS,
    SOURCE_STEPS,
    _target_budget_for,
)


def test_expert_is_source_scale_upper_bound():
    # the full-target expert must never masquerade as a fair 50k-budget arm
    assert EXPERT_STEPS == SOURCE_STEPS
    assert EXPERT_STEPS > FINETUNE_STEPS


def test_target_budget_mapping():
    assert _target_budget_for("zero_shot") == 0
    assert _target_budget_for("threshold_rule") == 0
    assert _target_budget_for("finetuned") == FINETUNE_STEPS
    assert _target_budget_for("scratch") == FINETUNE_STEPS
    assert _target_budget_for("full_target_expert") == EXPERT_STEPS
