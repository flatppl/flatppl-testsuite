"""Independent oracle for frag_trunc_out.

Outside the truncation interval, the gated density is zero -> -inf. `-inf`
cannot round-trip through standard JSON, so `test.json`'s `expected` is the
string "-inf" (parsed back to `float("-inf")` by the harness).
"""


def oracle() -> float:
    return float("-inf")
