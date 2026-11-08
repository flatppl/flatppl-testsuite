"""Independent oracle for cov_out_of_window (scipy).

Every observation misses the window, so `filter` leaves an empty
vector, the derived `iid` size is 0, and the likelihood is the empty
product measure with log-density exactly 0 (§06 `iid`: "the empty sum
in the density rule for composed measures"; the zero-size-arrays
ruling). The posterior log-density is the prior term alone:

    lp(mu) = Normal(mu | 0, 2).logpdf

STATUS: lowers on both engines. With fixed data and a fixed predicate the
const-evaluator folds `lengthof(filter(...))` (flatppl-rust R10, a narrow
FixedValue widening: a membership test against a fixed interval, a comparison
with a fixed number, or a land/lor/lnot of those). The kept count here is 0,
which types as `%dynamic` (the zero-size ruling), and the determiniser reads
the folded 0 and emits the empty product 0.0 for the `iid`, so the data path
vanishes and StableHLO scores the prior term alone through the ABI query.

History: before R10 the determiniser refused at the `iid` node ("iid size is
not a statically-resolved shape"), and this dir was det-js only with
`allow_skip`.
"""
from scipy import stats


def oracle(point: dict) -> float:
    return float(stats.norm.logpdf(float(point["mu"]), 0.0, 2.0))
