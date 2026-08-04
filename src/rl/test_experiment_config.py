"""audit-v3 (2.1): RLExperimentConfig - validation, hashing, strict-ablation
semantics (varying one field must not silently change others)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

from experiment_config import PRIMARY_CONFIG, RLExperimentConfig


def test_primary_config_validates():
    PRIMARY_CONFIG.validate()


def test_invalid_gamma_rejected():
    with pytest.raises(ValueError, match="gamma"):
        RLExperimentConfig(gamma=0.0)
    with pytest.raises(ValueError, match="gamma"):
        RLExperimentConfig(gamma=1.5)


def test_invalid_water_normalizer_rejected():
    with pytest.raises(ValueError, match="water_normalizer"):
        RLExperimentConfig(water_normalizer="per_quota")  # old string form is gone


def test_invalid_device_rejected():
    with pytest.raises(ValueError, match="device"):
        RLExperimentConfig(device="gpu")


def test_config_hash_is_stable_and_sensitive():
    c = PRIMARY_CONFIG
    assert c.config_hash() == c.config_hash()
    # changing any single field must change the hash
    for kw in [dict(gamma=1.0), dict(seed=1), dict(device="cuda"),
               dict(water_normalizer="max_action"), dict(shaping_mode="none")]:
        assert c.with_updates(**kw).config_hash() != c.config_hash()


def test_with_updates_changes_only_the_named_field():
    changed = PRIMARY_CONFIG.with_updates(gamma=1.0)
    assert changed.gamma == 1.0
    assert changed.seed == PRIMARY_CONFIG.seed
    assert changed.device == PRIMARY_CONFIG.device
    assert changed.water_normalizer == PRIMARY_CONFIG.water_normalizer


def test_potential_shaping_requires_valid_gamma():
    with pytest.raises(ValueError):
        RLExperimentConfig(gamma=1.5, shaping_mode="potential")


def test_to_json_roundtrip():
    import json
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    try:
        path = tmp / "config.json"
        PRIMARY_CONFIG.to_json(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["gamma"] == PRIMARY_CONFIG.gamma
        assert data["water_normalizer"] == PRIMARY_CONFIG.water_normalizer
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
