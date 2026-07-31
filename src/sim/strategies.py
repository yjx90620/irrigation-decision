"""Baseline irrigation strategies from docs/研究方案.md section 4.5.

Each entry is a zero-arg factory (not a shared instance) since
IrrigationManagement objects get mutated by the model during a run.
NSGA-II-optimized policies (研究方案 4.5 策略六) are out of scope here -
they need a proper multi-objective optimizer and get their own module.
"""

from aquacrop import IrrigationManagement

STRATEGIES = {
    "rainfed": lambda: IrrigationManagement(irrigation_method=0),
    "full_irrigation": lambda: IrrigationManagement(irrigation_method=1, SMT=[100, 100, 100, 100]),
    "fixed_interval_14d": lambda: IrrigationManagement(irrigation_method=2, IrrInterval=14),
    "threshold_30pct": lambda: IrrigationManagement(irrigation_method=1, SMT=[70, 70, 70, 70]),
    "threshold_40pct": lambda: IrrigationManagement(irrigation_method=1, SMT=[60, 60, 60, 60]),
    "threshold_50pct": lambda: IrrigationManagement(irrigation_method=1, SMT=[50, 50, 50, 50]),
    "threshold_60pct": lambda: IrrigationManagement(irrigation_method=1, SMT=[40, 40, 40, 40]),
    # AquaCrop's 4 SMT slots follow its own growth-stage split (not exactly
    # 苗期/拔节/抽雄吐丝/灌浆); this approximates 关键期灌溉 by protecting the
    # 3rd slot (covers flowering/mid-season) more than the others.
    "critical_stage": lambda: IrrigationManagement(irrigation_method=1, SMT=[40, 40, 70, 50]),
}
