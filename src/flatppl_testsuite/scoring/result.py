"""CheckResult dataclass and stage-tag constants."""

from __future__ import annotations

from dataclasses import dataclass

CONVERT_SKIP = "CONVERT_SKIP"
UNSCOREABLE = "UNSCOREABLE"
NUMERIC_MISMATCH = "NUMERIC_MISMATCH"
DETERMINIZE_SKIP = "DETERMINIZE_SKIP"
EMIT_REFUSED = "EMIT_REFUSED"
# A passed point whose frozen value Stan met by rejecting it (-inf or an error record).
STAN_REJECTED = "STAN_REJECTED"
# A passed point scored by the `outputs` program because no point input binds a
# parameter of the `binding` program.
STAN_OUTPUTS = "STAN_OUTPUTS"


@dataclass
class CheckResult:
    test_id: str
    check_id: str
    status: str          # passed | failed | skipped
    tag: str = ""        # CONVERT_SKIP | UNSCOREABLE | NUMERIC_MISMATCH | DETERMINIZE_SKIP | EMIT_REFUSED | STAN_REJECTED | STAN_OUTPUTS | ""
    message: str = ""
