"""Independent oracle for a broadcast Categorical over gathered probability rows.

`idx` gathers rows at run time and `fixed` at emit time; both score the same `w`.
Spec §08 declares `p` over stdsimplex(n), so a gathered row outside the simplex
has no density (StableHLO status 9). A row that no index selects is never a
parameter and needs no check. An index outside 1..4 is out of range (§04 get),
and a category outside 1..3 is outside the support, so it scores -inf.
"""
import math

ERROR = {"error": "outside its declared value set"}
OUT_OF_RANGE = {"error": "index out of range"}
FIXED = [3, 1, 2, 3, 1]


def _simplex(row: list) -> bool:
    return all(math.isfinite(p) and p >= 0 for p in row) and abs(sum(row) - 1) <= 8 * len(row) * 2.0**-23


def _categorical(row: list, c: int) -> float:
    return math.log(row[c - 1]) if 1 <= c <= len(row) and row[c - 1] > 0 else -math.inf


def oracle(point: dict) -> float | dict:
    rows, idx, w = point["rows"], point["idx"], point["w"]
    if not all(1 <= i <= len(rows) for i in idx):
        return OUT_OF_RANGE
    if not all(_simplex(rows[i - 1]) for i in set(idx) | set(FIXED)):
        return ERROR
    return float(sum(_categorical(rows[i - 1], c) for i, c in zip(idx, w))
                 + sum(_categorical(rows[i - 1], c) for i, c in zip(FIXED, w)))
