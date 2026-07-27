"""Root conftest: ensure repo root and src/ are on sys.path.

This allows ``import flatppl_testsuite.*`` (and, for the unified harness's
runners under ``src/flatppl_testsuite/unified/``, dynamic loading of any
``corpora/<corpus>/<test_id>/test.py``) to resolve from any test location.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_SRC = _REPO_ROOT / "src"

for _p in (_REPO_ROOT, _SRC):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
