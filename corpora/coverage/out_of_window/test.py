"""Independent oracle for cov_out_of_window (scipy).

Every observation misses the window, so `filter` leaves an empty
vector, the derived `iid` size is 0, and the likelihood is the empty
product measure with log-density exactly 0 (§06 `iid`: "the empty sum
in the density rule for composed measures"; the zero-size-arrays
ruling). The posterior log-density is the prior term alone:

    lp(mu) = Normal(mu | 0, 2).logpdf

STATUS: the rust determiniser REFUSES the model at the `iid` node —
"iid size is not a statically-resolved 1-D count ...; only a 1-D
static size is unrolled" (verified live 2026-09-01). So §06's own
region-restricted idiom (filter -> lengthof -> iid) does not lower at
ANY data-derived size, empty or not. `allow_skip: true`; the frozen
values are real oracle values and take over when dynamic sizes lower.
"""
from scipy import stats


def oracle(point: dict) -> float:
    return float(stats.norm.logpdf(float(point["mu"]), 0.0, 2.0))
