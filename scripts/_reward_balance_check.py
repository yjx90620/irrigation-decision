"""audit-v3 (3.2 fix) reward-balance check: decompose one episode's
discounted return into yield_proxy / water / cost / acute_stress channels
under the threshold rule, to confirm the per-harvest yield bonus restores
a yield signal comparable to the water penalty."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "rl"))
sys.path.insert(0, str(ROOT / "src" / "data"))
sys.path.insert(0, str(ROOT / "src" / "sim"))

from experiment_config import PRIMARY_CONFIG
from rotation_env import RotationIrrigationEnv, threshold_policy
from train_rotation_compare import BALANCED_WEIGHTS


def decompose(site_id, year, policy_fn, label):
    env = RotationIrrigationEnv(site_id, "loam", year, config=PRIMARY_CONFIG)
    state = env.reset()
    done, n = False, 0
    yp = wt = ct = ac = 0.0
    g = PRIMARY_CONFIG.gamma
    while not done:
        state, r, done, info = env.step(policy_fn(state, BALANCED_WEIGHTS))
        yp += (g ** n) * r["yield_proxy"]
        wt += (g ** n) * r["water"]
        ct += (g ** n) * r["cost"]
        ac += (g ** n) * r["acute_stress"]
        n += 1
    print(f"{label} {site_id} {year}: {n} steps; yield={info['total_yield_t_ha']:.2f} t/ha "
          f"irr={info['total_irrigation_mm']:.0f} mm")
    print(f"  discounted yield_proxy = {yp:+.3f}   water = {wt:+.3f}   cost = {ct:+.3f}   acute = {ac:+.3f}")


if __name__ == "__main__":
    for site, year in [("hebei_central", 2019), ("ningxia_irrigation", 2019), ("shaanxi_guanzhong", 2019)]:
        decompose(site, year, threshold_policy, "threshold_rule")
