"""Temporal split (audit-v3, 5.1): ONE source of truth for which years are
used for calibration/training, validation (selection) and the untouched
final test.

The audit found alpha selection (allocation scan), cross-season
optimization evaluation and MWV all evaluated on years that were also in
their own selection windows (e.g. 2018/2020 appeared in both scan and
evaluation) - a selection-on-the-test-set leak. Here:
  - development (calibration + training): 1982-2010 - wheat calendar
    feasibility, crop params, fingerprints, RL training. The audit's
    5.1 bullets forbid using FINAL TEST years for any of these, which
    this split guarantees; calibration and training intentionally share
    the window (both are "development", the audit's example dataclass
    rejects calib x train overlap, but its BULLET requirements do not).
  - validation (selection): 2011-2017 - alpha selection, cross-season
    optimization budgets, MWV stage definition. Selection decisions are
    made here, never on the test set.
  - final_test: 2018-2025 - evaluated exactly once, then never touched.

Every module reads the year ranges from here; hardcoded year lists in
scripts are a bug.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TemporalSplit:
    calibration_years: tuple
    training_years: tuple
    validation_years: tuple
    final_test_years: tuple

    def validate(self) -> None:
        cal, tr, val, te = (set(g) for g in (
            self.calibration_years, self.training_years,
            self.validation_years, self.final_test_years))
        if cal & te or tr & te or val & te:
            raise ValueError(
                "temporal leakage: final test years must not appear in any "
                "calibration/training/validation window"
            )
        if val & tr or val & cal:
            raise ValueError(
                "temporal leakage: validation (selection) years must be "
                "disjoint from training/calibration"
            )

    def as_list(self, group: str) -> list:
        return list(getattr(self, f"{group}_years"))


def _years(start, end):
    return tuple(range(start, end + 1))


# The single project-wide split. All scripts must read from here.
SPLIT = TemporalSplit(
    calibration_years=_years(1982, 2010),  # crop params, fingerprints, ET0
    training_years=_years(1982, 2010),  # RL training / simulation development
    validation_years=_years(2011, 2017),  # alpha/algorithm/model selection
    final_test_years=_years(2018, 2025),  # untouched, evaluated once
)
SPLIT.validate()
