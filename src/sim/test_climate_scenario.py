"""P1-2 acceptance tests (docs/审计修复计划.md): CMIP6 delta computation
order and model-set consistency."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))

import numpy as np
import pandas as pd

from climate_scenario import _ensemble_delta


def _synthetic_period(month, values_by_model):
    df = pd.DataFrame({"month": [month] * len(values_by_model)})
    for i, (model, val) in enumerate(values_by_model.items()):
        df[f"precip_{model}"] = val
    return df


def test_multiplicative_delta_is_mean_of_ratios_not_ratio_of_means():
    """For a ratio-based (multiplicative) variable, averaging models
    first and then dividing gives a different number than dividing each
    model's own future/historical pair first and then averaging - this
    test pins down that _ensemble_delta does the latter (matches the
    module's own documented methodology), not the former."""
    hist = pd.DataFrame({"month": [6], "precip_A": [100.0], "precip_B": [50.0]})
    fut = pd.DataFrame({"month": [6], "precip_A": [110.0], "precip_B": [100.0]})

    # patch MODELS temporarily so this only sees the two synthetic models
    import climate_scenario

    original_models = climate_scenario.MODELS
    climate_scenario.MODELS = ["A", "B"]
    try:
        mean, std, n = _ensemble_delta(hist, fut, "precip", "multiplicative")
    finally:
        climate_scenario.MODELS = original_models

    per_model_ratio_mean = np.mean([110.0 / 100.0, 100.0 / 50.0])  # 1.1, 2.0 -> mean 1.55
    ratio_of_means = (110.0 + 100.0) / (100.0 + 50.0)  # 210/150 = 1.4 - the wrong-order result
    assert n == 2
    assert abs(mean.loc[6] - per_model_ratio_mean) < 1e-9
    assert abs(mean.loc[6] - ratio_of_means) > 0.01, "delta looks like it was computed ensemble-mean-first"


def test_models_missing_from_either_period_are_excluded_from_both():
    """A model with data in only one of the two periods must not
    contribute to either side of the delta - it should be dropped
    entirely, not silently included in whichever period has it."""
    hist = pd.DataFrame({"month": [6], "precip_A": [100.0], "precip_B": [50.0]})
    fut = pd.DataFrame({"month": [6], "precip_A": [110.0]})  # B missing from future

    import climate_scenario

    original_models = climate_scenario.MODELS
    climate_scenario.MODELS = ["A", "B"]
    try:
        mean, std, n = _ensemble_delta(hist, fut, "precip", "multiplicative")
    finally:
        climate_scenario.MODELS = original_models

    assert n == 1
    assert abs(mean.loc[6] - 1.1) < 1e-9
