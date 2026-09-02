"""Independent oracle for cov_out_of_window (scipy).

Every observation misses the window, so `filter` leaves an empty
vector, the derived `iid` size is 0, and the likelihood is the empty
product measure with log-density exactly 0 (§06 `iid`: "the empty sum
in the density rule for composed measures"; the zero-size-arrays
ruling). The posterior log-density is the prior term alone:

    lp(mu) = Normal(mu | 0, 2).logpdf

STATUS: the rust determiniser REFUSES the model at the `iid` node —
"iid size is not a statically-resolved 1-D count ...; only a 1-D
static size is unrolled" (re-verified live 2026-09-01 against the
record-measure-unroll branch, which does NOT reach this). So §06's own
region-restricted idiom (filter -> lengthof -> iid) does not lower at
ANY data-derived size, empty or not. `allow_skip: true`; the frozen
values are real oracle values and take over when dynamic sizes lower.

TWO determiniser blockers, not one, and neither is the unroll:

1. `lengthof(filter(...))` stays `%dynamic`. `consteval.rs`'s
   `length_observer` reads the inferred TYPE of `obs_w`, and `filter`'s
   type rule gives a dynamic length (correct in general — a filter's
   output length is data-dependent). Resolving it needs the const-eval
   table to FOLD `filter` over fixed data, which needs reals, interval
   sets, and evaluating a predicate body at a bound placeholder — the
   deferred `flatppl-interpreter` value core, not a shape widening.
2. Even at a LITERAL size the model refuses one node later:
   "normalize(truncate): closed-form Z is only implemented for an
   `interval(lo, hi)` truncation set; a named/other set is not yet
   supported" — `window` here is a named binding, and the arm wants an
   inline `interval(...)`.

NO StableHLO row, for the same reason. Re-probed live 2026-09-02 with a
`--features hs3,stablehlo` binary at rust `b08eb79` and an ABI query:
`flatppl stablehlo` exits 3 with the SAME determiniser refusal, so the
model never reaches the emitter and a stablehlo row would only add a
second skip. Nothing to work around: `crates/stablehlo` has no `filter`
either, and folding the size makes the whole data path vanish (N = 0 ->
the literal 0.0), which is also the only route by which this model could
ever execute on StableHLO.
"""
from scipy import stats


def oracle(point: dict) -> float:
    return float(stats.norm.logpdf(float(point["mu"]), 0.0, 2.0))
