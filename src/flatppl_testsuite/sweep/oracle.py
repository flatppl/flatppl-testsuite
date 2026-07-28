"""The independent oracle: a probe's TRUE log-density.

Transcribes §13 "Output reduction"'s enumerated rules over scipy base
densities. It walks the `Probe` — the structure the generator built — so it
needs no parser and no type inference, and it never reads the determiniser's
output. Authority order is maths > spec > code.

Each rule below cites the clause it implements. A rule with no citation is a
bug: the point of this module is that it is derivable from the spec by someone
who has never seen the determiniser.
"""
from __future__ import annotations

import math

from scipy import stats

from flatppl_testsuite.sweep.space import Base, Probe, Wrap, in_support


class OracleUnsupported(Exception):
    """This structure has no implemented closed form here.

    Raised, never silently skipped: an unsupported structure must appear in the
    verdict table as `oracle-unvalidated` so the gap is visible in the committed
    artifact.
    """


def _frozen(base: Base):
    """scipy frozen distribution for a base. Parameter ORDER follows §08, and
    the scipy translation is explicit because the conventions differ:
    §08 `Gamma(shape, rate)` is scipy `gamma(a=shape, scale=1/rate)`."""
    k, p = base.kind, base.params
    if k == "normal":
        return stats.norm(loc=p[0], scale=p[1])
    if k == "gamma":
        return stats.gamma(a=p[0], scale=1.0 / p[1])
    if k == "beta":
        return stats.beta(a=p[0], b=p[1])
    if k == "poisson":
        return stats.poisson(mu=p[0])
    raise OracleUnsupported(f"base {k}")


def _base_logpdf(base: Base, x: float) -> float:
    """Base log-density at `x`.

    **A discrete base's `x` is snapped to the nearest integer within 1e-9 first,
    and that is load-bearing, not defensive.** Inverting a wrap is a float
    round-trip: `math.sqrt(2.0) ** 2` is `2.0000000000000004`, not `2.0`. scipy's
    `poisson.logpmf` returns `-inf` on a non-integer, so without the snap a
    perfectly legitimate `poisson` + `sqrt` probe — whose preimage IS the in-support
    integer 2 — would score `-inf` here while the determiniser computed the correct
    finite `-1.4959226032237258`, and the sweep would report a false wrong-number
    against the engine. That is the worst failure available to this tool: its own
    arithmetic error, attributed to the thing it is auditing.

    **Snap ONCE and use the snapped value for both the support gate and the density
    call.** The tolerance also lives in `space.in_support`, which is correct — every
    caller recovering a discrete preimage through a float round-trip hits the same
    ULP noise, so a strict predicate would reject correctly-derived in-support
    probes. But that means the pairing has to be enforced in one place rather than
    by convention across two call sites: checking `in_support(base, x)` on the raw
    float and then rounding separately for `logpmf` is the shape that lets the two
    disagree. Compute `n` once; gate on `n`; evaluate at `n`.
    """
    d = _frozen(base)
    if _is_discrete(base):
        n = float(round(x))
        if abs(x - n) >= 1e-9:
            return -math.inf          # genuinely off-lattice: density is zero
        if not in_support(base, n):
            return -math.inf          # negative, or otherwise outside the support
        return float(d.logpmf(n))
    if not in_support(base, x):
        return -math.inf
    return float(d.logpdf(x))


def _interval(wrap: Wrap) -> tuple[float, float]:
    lo, hi = wrap.args
    return (float(lo) if lo != "-inf" else -math.inf,
            float(hi) if hi != "inf" else math.inf)


def _is_discrete(base: Base) -> bool:
    """§06 line 28: the reference measure is "Lebesgue measure for continuous
    variates, counting measure for discrete variates"."""
    return base.kind == "poisson"


# Forward log-volume elements of the `pushfwd` maps, EVALUATED AT THE PREIMAGE.
#
# §06's engine contract is
#     log densityof(pushfwd(f, M), y) = log densityof(M, f_inv(y)) - logvolume(f_inv(y))
# and §06 is explicit that `logvolume` describes the FORWARD map: "logvolume is
# the generalized log-volume-element of the forward function". So each entry
# below is `log|f'(x)|` at `x = f_inv(y)`, written in terms of `y` — and it is
# always SUBTRACTED. Getting the direction wrong is silent: it produces a
# plausible finite number, and the sweep then attributes the error to the
# determiniser.
#
# `sqrt` is the one that does not look like the others. Its forward derivative
# is `1/(2*sqrt(x))`, so at the preimage `x = y**2` the forward log-volume is
# `-log(2y)` — NEGATIVE, because `sqrt` contracts. Subtracting it ADDS `log(2y)`
# to the density. §06's own worked example annotates `log(2*_)` for the map
# `x -> x**2`, where the squaring is the forward direction; reusing that
# annotation for `sqrt` inverts the sign. Cross-checked numerically: for
# Gamma(shape = 2, rate = 1) at y = sqrt(1.5) the density is
# -0.19865515727780814, matching both a numeric derivative of `F_X(y**2)` and
# `scipy.stats.gengamma(a=2, c=2).logpdf(y)`.
_INVERSE = {
    "exp":  lambda y: math.log(y),
    "log":  lambda y: math.exp(y),
    "neg":  lambda y: -y,
    "sqrt": lambda y: y * y,
}
# Points whose preimage overflows the float range. `pushfwd(log, M)` at y > ~709
# has preimage `e^y = inf`, which raises rather than returning a density. §06's
# posreals guard means `log` is only well-formed over `gamma`/`beta`, and both
# have density 0 that far out (`beta`'s support ends at 1; `gamma`'s density
# decays like `e^-x`), so the density there is 0 -- but it has to be RETURNED,
# not raised, or the sweep dies on a probe instead of scoring it.
_PREIMAGE_OVERFLOWS = {"log": lambda y: y > 709.0}
_LOGVOLUME_AT_PREIMAGE = {
    "exp":  lambda y: math.log(y),            # |d exp/dx| at log y  = y
    "log":  lambda y: -y,                     # |d log/dx| at e^y    = e^-y
    "neg":  lambda y: 0.0,                    # volume-preserving
    # |d sqrt/dx| at y^2 = 1/(2y), which diverges as y -> 0. At y = 0 exactly the
    # forward log-volume is +inf, and subtracting it gives density 0 -- correct,
    # and the right shape for the discrete case too, where the override below
    # discards the term and leaves `pmf(0)` at the atom 0.
    "sqrt": lambda y: math.inf if y == 0.0 else -math.log(2.0 * y),
}
# The domain of the forward map: a point outside it is not in the pushed-forward
# measure's variate space at all, so the density there is 0.
_FORWARD_RANGE = {
    "exp":  lambda y: y > 0.0,
    "log":  lambda y: True,
    "neg":  lambda y: True,
    "sqrt": lambda y: y >= 0.0,
}


def true_logpdf(probe: Probe) -> float:
    """Fold the wraps, per §13's rules.

    **Wraps are peeled OUTERMOST FIRST.** `probe.wraps` is innermost-first (that
    is how `render._fold` composes them), but a density query is evaluated from
    the outside in: the outermost wrap is the one that sees the query point.
    Every wrap the space currently generates in combination is point-preserving,
    so the direction is unobservable there — but it stops being unobservable the
    moment a set-valued gate sits outside a change of variable, because
    `truncate(pushfwd(exp, M), S)` gates on `y` while `pushfwd(exp, truncate(M, S))`
    gates on `log y`. Those are different measures; peeling inward is what tells
    them apart.

    **A discrete base takes NO log-volume term.** §06's `logvolume` is the
    generalized volume element of the forward map *with respect to the reference
    measure*, and for a discrete variate that reference is the counting measure
    (§06 line 28), which a bijection does not distort. So `pushfwd(exp, Poisson(3))`
    has density `pmf(log y)` at `y` — no `- log y`. Subtracting a Lebesgue Jacobian
    there would make this oracle wrong in exactly the region the space was widened
    to probe, and it would report the error as the determiniser's.
    """
    x = probe.point
    logv = 0.0          # accumulated FORWARD log-volume, subtracted at the end
    extra = 0.0         # weight and normalization terms: point-independent shifts

    for i in range(len(probe.wraps) - 1, -1, -1):
        w = probe.wraps[i]
        k = w.kind
        if k == "identity":
            continue
        if k in ("pushfwd", "affine", "locscale"):
            # §06's pushforward form:
            #   log densityof(M, f_inv(y)) - logvolume(f_inv(y))
            if k == "pushfwd":
                op = w.args[0]
                if op not in _INVERSE:
                    raise OracleUnsupported(f"pushfwd {op}")
                if not _FORWARD_RANGE[op](x):
                    return -math.inf      # outside the forward map's range
                if _PREIMAGE_OVERFLOWS.get(op, lambda _: False)(x):
                    return -math.inf      # preimage past the float range: density 0
                logv += _LOGVOLUME_AT_PREIMAGE[op](x)
                x = _INVERSE[op](x)
            elif k == "affine":
                a, b = w.args
                logv += math.log(abs(a))  # |d(a*x + b)/dx| = |a|, constant
                x = (x - b) / a
            else:
                # §06: locscale(m, shift, scale) == pushfwd(x -> scale*x + shift, m)
                loc, sc = w.args
                logv += math.log(abs(sc))
                x = (x - loc) / sc
        elif k == "truncate":
            lo, hi = _interval(w)
            if not (lo <= x <= hi):
                return -math.inf              # §13: "gates on the truncation set"
            # §06's table: truncate does NOT normalize -- no mass correction.
        elif k == "weighted":
            extra += math.log(w.args[0])      # §13: "adds the log of the weight"
        elif k == "logweighted":
            extra += w.args[0]                # §13: "the log-weight"
        elif k == "normalize":
            extra -= _log_total_mass(probe, i)
        else:
            raise OracleUnsupported(f"wrap {k}")

    if _is_discrete(probe.base):
        logv = 0.0        # counting measure: a bijection distorts no volume
    return _base_logpdf(probe.base, x) - logv + extra


def _log_total_mass(probe: Probe, idx: int) -> float:
    """log(totalmass(M)) for the measure INSIDE the `normalize` at `probe.wraps[idx]`
    (§13: normalize "subtracts log(totalmass(M)), which must be finite and nonzero").

    Only the shapes the space generates are implemented: a bare base (§08 opens
    "the built-in distributions (i.e. probability measures)", so the mass is
    exactly 1), a truncated base (mass = the base's CDF over the interval), and a
    `weighted` base (mass = the weight). Anything else is an explicit gap, not a
    guess — `normalize(pushfwd(...))` is left unimplemented even though a
    pushforward preserves mass, because nothing generates it and an untested
    branch here is indistinguishable from a wrong one.
    """
    inner = probe.wraps[:idx]
    if len(inner) == 0:
        return 0.0
    if len(inner) == 1 and inner[0].kind == "truncate":
        lo, hi = _interval(inner[0])
        d = _frozen(probe.base)
        return math.log(float(d.cdf(hi) - d.cdf(lo)))
    if len(inner) == 1 and inner[0].kind == "weighted":
        return math.log(inner[0].args[0])
    raise OracleUnsupported(f"totalmass of {[w.kind for w in inner]}")
