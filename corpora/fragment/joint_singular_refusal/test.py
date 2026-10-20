"""Refusal fixture, not a scoring oracle, for frag_joint_singular_refusal.

`lawof(record(a = y, b = y))` retains the same draw `y` in both named
fields. The same draw referenced twice has no
density w.r.t. the product reference measure -- the law is carried by the
diagonal `{a = b}`, a lower-dimensional set, so `logdensityof` must REFUSE
rather than emit a product of two `Normal(0, 1)` marginals. There is
no closed-form density to freeze as `expected`; the gate is that the
determiniser exits 3 ("two fields resolve to the same draw") and the unified
harness reports `skipped`/`DETERMINIZE_SKIP`, which `allow_skip: true` in
this dir's `test.json` tolerates. `expected` is the string `"nan"` (NaN never
matches, per `scoring/compare.py`) so that IF the determiniser ever starts
LOWERING this shape instead of refusing it, Mode A's fallback numeric compare
fails loudly rather than passing on an arbitrary sentinel.

No `oracle()` function: this fixture has no frozen scalar to reproduce.
"""
