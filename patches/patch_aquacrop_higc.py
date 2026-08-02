"""Apply a required bug fix to the installed aquacrop-ospy package.

Root cause behind nearly every unexplained multi-hour "hang" this project
hit (Beijing's alpha-scan failures at year 2013, task_sensitivity.py's
15+ hour rainfed run, and the RL training's repeated stalls even after
excluding Beijing): aquacrop.initialize.calculate_HIGC.calculate_HIGC()
has an unbounded `while` loop with no iteration cap. Under severe water
stress the simulated crop's calendar can collapse enough that HIstartCD
ends up after HIendCD, making crop_YldFormCD (the loop's `tHI`) negative.
That drives np.exp(-HIGC * tHI) to overflow as HIGC grows, which pins the
loop's convergence target (HIest) at exactly 0 forever - confirmed by a
direct test running the un-patched loop for 2,000,000+ iterations without
terminating. It's a real infinite loop, not just slow, and any workaround
outside the package (subprocess timeouts, safety filters, watchdogs) can
only ever be a partial mitigation for this specific mechanism.

The fix caps the loop at 200,000 iterations (HIGC up to 200 - published
crop parameters are well under 1.0, so the cap is only ever reached in
the degenerate negative-tHI case) and suppresses the resulting harmless
overflow RuntimeWarning. Verified: degenerate case now returns in ~0.2s
instead of hanging; normal crop parameters produce identical output to
the original code (both terminate in a handful of iterations well under
the cap).

.venv/ is gitignored, so this patch does not survive a fresh
`pip install -r requirements.txt` - re-run this script after any
environment (re)install. Idempotent: safe to run multiple times.
"""

import re
from pathlib import Path

import aquacrop

TARGET = Path(aquacrop.__file__).resolve().parent / "initialize" / "calculate_HIGC.py"

PATCH_MARKER = "PATCHED (see docs/aquacrop_patches.md)"

OLD_LOOP = '''    HIGC = 0.001
    HIest = 0
    while HIest <= (0.98 * crop_HI0):
        HIGC = HIGC + 0.001
        HIest = (crop_HIini * crop_HI0) / (
            crop_HIini + (crop_HI0 - crop_HIini) * np.exp(-HIGC * tHI)
        )
'''

NEW_LOOP = '''    HIGC = 0.001
    HIest = 0
    # PATCHED (see docs/aquacrop_patches.md): if a stressed crop's
    # calendar collapses enough that HIstartCD ends up after HIendCD,
    # tHI goes negative, np.exp(-HIGC * tHI) overflows to inf, and HIest
    # is permanently pinned at 0 - the original unbounded loop then spins
    # forever (confirmed: 2M+ iterations, never terminates). HIGC=200 is
    # far beyond any real crop parameter (published values are well under
    # 1.0), so hitting the cap only ever happens in this degenerate case.
    n_iter = 0
    with np.errstate(over="ignore"):
        while HIest <= (0.98 * crop_HI0) and n_iter < 200_000:
            HIGC = HIGC + 0.001
            HIest = (crop_HIini * crop_HI0) / (
                crop_HIini + (crop_HI0 - crop_HIini) * np.exp(-HIGC * tHI)
            )
            n_iter += 1
'''


def main():
    src = TARGET.read_text(encoding="utf-8")
    if PATCH_MARKER in src:
        print(f"already patched: {TARGET}")
        return
    if OLD_LOOP not in src:
        raise RuntimeError(
            f"expected unpatched loop text not found in {TARGET} - aquacrop-ospy version may have "
            "changed this function; inspect it manually before re-applying"
        )
    TARGET.write_text(src.replace(OLD_LOOP, NEW_LOOP), encoding="utf-8")
    print(f"patched: {TARGET}")


if __name__ == "__main__":
    main()
