"""audit-v3 (7.5): check every relative markdown link under docs/ (papers
and top-level docs) resolves to an existing file. Fails with exit code 1
if any link is broken."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def main() -> int:
    broken = []
    total = 0
    for f in DOCS.rglob("*.md"):
        text = f.read_text(encoding="utf-8")
        for m in re.finditer(r"\[[^\]]*\]\(([^)]+)\)", text):
            target = m.group(1).split("#")[0].strip()
            if not target or target.startswith(("http", "mailto:")):
                continue
            total += 1
            p = (f.parent / target).resolve()
            if not p.exists():
                broken.append((str(f.relative_to(ROOT)), target))
    print(f"checked {total} relative links; broken: {len(broken)}")
    for src, target in broken:
        print(f"  BROKEN: {src} -> {target}")
    return 1 if broken else 0


if __name__ == "__main__":
    raise SystemExit(main())
