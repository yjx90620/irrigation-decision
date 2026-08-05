"""audit-v3 (7.5): every relative markdown link in docs/ must resolve -
the paper drafts must never reference archived/missing files."""

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_doc_links.py"


def test_all_doc_links_resolve():
    spec = importlib.util.spec_from_file_location("check_doc_links", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert mod.main() == 0
