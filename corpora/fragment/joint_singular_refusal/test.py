"""Refusal fixture, not a scoring oracle, for frag_joint_singular_refusal.

`joint(a = lawof(y), b = lawof(y))` reifies the SAME draw `y` into both named
components. §06 "Singular joints": the same draw referenced twice has no
density w.r.t. the product reference measure -- the law is carried by the
diagonal `{a = b}`, a lower-dimensional set, so `logdensityof` must REFUSE
rather than emit the (wrong) product of two `Normal(0, 1)` marginals. There is
no closed-form density to freeze as `expected`; the gate is that the
determiniser exits 3 ("two fields resolve to the same draw") and the unified
harness reports `skipped`/`DETERMINIZE_SKIP`, which `allow_skip: true` in
this dir's `test.json` tolerates. `expected` is the string `"nan"` (NaN never
matches, per `scoring/compare.py`) so that IF the determiniser ever starts
LOWERING this shape instead of refusing it, Mode A's fallback numeric compare
fails loudly rather than passing on an arbitrary sentinel.

Verified live against flatppl-rust b79517a: `flatppl determinize` on this
model exits 3 with "record law has fewer distinct draws than fields ... (two
fields resolve to the same draw)".

No `oracle()` function: this fixture has no frozen scalar to reproduce.
"""
