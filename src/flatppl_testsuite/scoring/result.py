"""CheckResult dataclass and stage-tag constants."""

from __future__ import annotations

from dataclasses import dataclass

CONVERT_SKIP = "CONVERT_SKIP"
UNSCOREABLE = "UNSCOREABLE"
NUMERIC_MISMATCH = "NUMERIC_MISMATCH"
DETERMINIZE_SKIP = "DETERMINIZE_SKIP"


@dataclass
class CheckResult:
    test_id: str
    check_id: str
    status: str          # passed | failed | skipped
    tag: str = ""        # CONVERT_SKIP | UNSCOREABLE | NUMERIC_MISMATCH | DETERMINIZE_SKIP | ""
    message: str = ""
