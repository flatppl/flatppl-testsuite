"""Root conftest: ensure repo root and src/ are on sys.path.

This allows ``import corpora.hs3.tests.*`` and ``import flatppl_testsuite.*``
to resolve from any test location (tests/, tests/core/, corpora/hs3/tests/).
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
_SRC = _REPO_ROOT / "src"

for _p in (_REPO_ROOT, _SRC):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
