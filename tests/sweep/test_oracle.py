"""The oracle is validated against hand-derived closed forms BEFORE it is
trusted on generated ground. Every expected value here is derived in the
docstring from the spec's rule, never taken from the determiniser."""
import json
import math
from pathlib import Path

import pytest
from scipy import integrate, stats

from flatppl_testsuite.scoring.compare import compare_scalar
from flatppl_testsuite.sweep.oracle import (
    OracleUnsupported,
    _frozen,
    _interval,
    _support_is_manifold,
    simplex_chart_to_hausdorff_offset,
    true_logpdf,
)
from flatppl_testsuite.sweep.space import (
    VECTOR_BASES,
    VECTOR_INNER,
    VECTOR_SUPPORT_IS_MANIFOLD,
    Base,
    Probe,
    Wrap,
    enumerate_probes,
    in_support,
    is_vector_base,
    vector_shapes,
)

REPO = Path(__file__).resolve().parents[2]


def _probe(base, wrap, point, spelling="direct"):
    return Probe(id="t", base=base, wraps=(wrap,), spelling=spelling,
                 ordering="single", consumer=False, point=point)


def test_bare_normal():
    # log phi(0.5) = -0.5*log(2pi) - 0.125
    got = true_logpdf(_probe(Base("normal", (0.0, 1.0)), Wrap("identity", ()), 0.5))
    assert got == -1.0439385332046727


def test_lognormal_via_pushfwd_exp():
    # §06: log densityof(M, log y) - log y. At y = e^0.5:
    # -1.0439385332046727 - 0.5
    got = true_logpdf(_probe(Base("normal", (0.0, 1.0)), Wrap("pushfwd", ("exp",)),
                             1.6487212707001282))
    assert abs(got - -1.5439385332046727) < 1e-12


def test_affine_pushfwd_subtracts_the_log_jacobian():
    # y = 2x + 1 over N(0,1) is N(1,2); at y = 3.0:
    # -log 2 - 0.5*log(2pi) - 0.5 = -2.112085713764618
    got = true_logpdf(_probe(Base("normal", (0.0, 1.0)), Wrap("affine", (2.0, 1.0)), 3.0))
    assert abs(got - -2.112085713764618) < 1e-12


def test_weighted_adds_the_log_weight():
    # §13: "weighted adds the log of the weight". log 0.5 + log phi(0.5)
    got = true_logpdf(_probe(Base("normal", (0.0, 1.0)), Wrap("weighted", (0.5,)), 0.5))
    assert abs(got - (math.log(0.5) - 1.0439385332046727)) < 1e-12


def test_truncate_is_unnormalized_and_gates_outside_support():
    # §06's table: truncate "does not normalize". Inside the support the density
    # is the base's, unchanged; outside it is exactly 0, i.e. -inf in logs.
    inside = true_logpdf(_probe(Base("normal", (0.0, 1.0)),
                                Wrap("truncate", (0.0, "inf")), 0.5))
    assert abs(inside - -1.0439385332046727) < 1e-12
    outside = true_logpdf(_probe(Base("normal", (0.0, 1.0)),
                                 Wrap("truncate", (0.0, "inf")), -0.5))
    assert outside == -math.inf


def test_normalize_subtracts_log_total_mass():
    # normalize(truncate(N(0,1), (0, inf))) has total mass 1/2, so the density
    # inside doubles: log phi(0.5) - log(0.5)
    p = Probe(id="t", base=Base("normal", (0.0, 1.0)),
              wraps=(Wrap("truncate", (0.0, "inf")), Wrap("normalize", ())),
              spelling="direct", ordering="single", consumer=False, point=0.5)
    assert abs(true_logpdf(p) - (-1.0439385332046727 - math.log(0.5))) < 1e-12


# --------------------------------------------------------------------------
# Rules the six cases above leave uncovered. Each is a rule the space DOES
# generate, so an untested rule here is an untested verdict on real probes.
# --------------------------------------------------------------------------

def test_pushfwd_sqrt_adds_the_log_jacobian_of_the_squaring_inverse():
    """§06's `logvolume` is the volume element of the FORWARD map.

    Forward is `sqrt`, so `logvolume(x) = log|d sqrt/dx| = -log(2*sqrt(x))`,
    and at the preimage `x = y**2` that is `-log(2y)`. §06's formula subtracts
    logvolume, so the density is `log p(y**2) + log(2y)` — the Jacobian is
    ADDED here, unlike `exp`/`affine`, because the forward map contracts.

    Cross-checked two ways independently of that algebra, for
    Gamma(shape = 2, rate = 1) at y = sqrt(1.5):
      * numeric derivative of `F_Y(y) = F_X(y**2)`  -> -0.1986551573634813
      * `scipy.stats.gengamma(a=2, c=2).logpdf(y)`  -> -0.19865515727780814
    (X ~ Gamma(a, scale=1) implies sqrt(X) ~ gengamma(a, c=2).)
    """
    y = math.sqrt(1.5)
    got = true_logpdf(_probe(Base("gamma", (2.0, 1.0)), Wrap("pushfwd", ("sqrt",)), y))
    assert abs(got - -0.19865515727780814) < 1e-12


def test_pushfwd_log_adds_the_preimage():
    """Forward is `log`, so `logvolume(x) = -log x`; at the preimage
    `x = e^y` that is `-y`, and subtracting it gives `log p(e^y) + y`.

    Gamma(shape = 2, rate = 1) has density `x * e^-x`, so at y = log(1.5)
    the pushforward density is `log(1.5 * e^-1.5) + log(1.5)`
    = `2*log(1.5) - 1.5` = -0.6890697837836712.
    """
    got = true_logpdf(_probe(Base("gamma", (2.0, 1.0)), Wrap("pushfwd", ("log",)),
                             math.log(1.5)))
    assert abs(got - (2.0 * math.log(1.5) - 1.5)) < 1e-12


def test_pushfwd_neg_is_volume_preserving():
    """`|d(-x)/dx| = 1`, so logvolume is 0: the density is the base's at -y."""
    got = true_logpdf(_probe(Base("normal", (0.0, 1.0)), Wrap("pushfwd", ("neg",)), -0.5))
    assert abs(got - -1.0439385332046727) < 1e-12


def test_logweighted_adds_the_log_weight_directly():
    # §13: "logweighted the log-weight" -- added as given, not exponentiated.
    got = true_logpdf(_probe(Base("normal", (0.0, 1.0)),
                             Wrap("logweighted", (-0.6931471805599453,)), 0.5))
    assert abs(got - (-0.6931471805599453 - 1.0439385332046727)) < 1e-12


def test_locscale_subtracts_the_log_scale():
    """§06: `locscale(m, shift, scale)` is `pushfwd(x -> scale*x + shift, m)`.
    At shift = 1, scale = 2, y = 1 + 2*0.5 = 2.0 the preimage is 0.5 and the
    log-volume is log 2: -1.0439385332046727 - log 2 = -1.737085713764618.
    """
    got = true_logpdf(_probe(Base("normal", (0.0, 1.0)), Wrap("locscale", (1.0, 2.0)), 2.0))
    assert abs(got - -1.737085713764618) < 1e-12


def test_normalize_of_a_bare_distribution_is_a_no_op():
    """§08 opens: the built-in distributions ARE probability measures, so
    `totalmass = 1` and `normalize` subtracts `log 1 = 0`. The space generates
    this shape (a bare `normalize` wrap), so it must not be an oracle gap.
    """
    got = true_logpdf(_probe(Base("normal", (0.0, 1.0)), Wrap("normalize", ()), 0.5))
    assert got == -1.0439385332046727


def test_a_discrete_base_takes_no_log_volume_term():
    """§06 line 28 makes the reference measure the COUNTING measure for a
    discrete variate, and a bijection does not distort it: the pushed-forward
    atoms carry the same masses, so `pushfwd(exp, Poisson(3))` has density
    `pmf(log y)` with no `- log y`.

    Independent argument that no Jacobian can be right here: `pushfwd` is
    defined in §06 by `(f_*M)(Y) = M(f^-1(Y))`, so it preserves total mass.
    The atoms of `exp_* Poisson(3)` are `{e^0, e^1, ...}`; their masses must
    still sum to 1, which they do only without a volume term.

    At y = e^2, `poisson.logpmf(2, 3) = -1.4959226032237258`.
    """
    got = true_logpdf(_probe(Base("poisson", (3.0,)), Wrap("pushfwd", ("exp",)),
                             math.exp(2.0)))
    assert abs(got - -1.4959226032237258) < 1e-12


def test_a_discrete_affine_pushfwd_also_takes_no_log_volume_term():
    # y = 2*2 + 1 = 5 has preimage 2; counting measure, so no `- log 2`.
    got = true_logpdf(_probe(Base("poisson", (3.0,)), Wrap("affine", (2.0, 1.0)), 5.0))
    assert abs(got - -1.4959226032237258) < 1e-12


def test_a_discrete_preimage_reached_through_a_float_round_trip_still_scores():
    """`math.sqrt(2.0) ** 2` is 2.0000000000000004, and scipy's `poisson.logpmf`
    returns -inf on a non-integer. The snap is what stops a legitimate
    poisson+sqrt probe from being reported as a determiniser defect.
    """
    y = math.sqrt(2.0)
    assert y * y != 2.0                      # the round trip really does drift
    got = true_logpdf(_probe(Base("poisson", (3.0,)), Wrap("pushfwd", ("sqrt",)), y))
    assert abs(got - -1.4959226032237258) < 1e-12


def test_a_genuinely_off_lattice_discrete_point_is_minus_inf():
    # 2.5 is not an atom of Poisson: density w.r.t. counting measure is 0.
    got = true_logpdf(_probe(Base("poisson", (3.0,)), Wrap("identity", ()), 2.5))
    assert got == -math.inf


def test_wraps_are_peeled_outermost_first():
    """`truncate(pushfwd(exp, M), S)` and `pushfwd(exp, truncate(M, S))` are
    DIFFERENT measures, because `S` lives in a different space in each. The
    fold order is what distinguishes them, so it is pinned here.

    `wraps` is innermost-first (that is how `render._fold` composes), so:
      * `(pushfwd, truncate)` = truncate(pushfwd(exp, N(0,1)), [0, inf)) --
        y = e^0.5 = 1.6487 is inside [0, inf), so the density is the
        log-normal's: -1.5439385332046727.
      * `(truncate, pushfwd)` = pushfwd(exp, truncate(N(0,1), [0, inf))) --
        the gate applies to the PREIMAGE 0.5, also inside, same value.
    Choose a point where they differ instead: y = 0.5, whose preimage is
    log 0.5 = -0.693 < 0. Outside-gate passes (0.5 >= 0) so the first is
    finite; inside-gate rejects (-0.693 < 0) so the second is -inf.
    """
    y = 0.5
    base = Base("normal", (0.0, 1.0))
    outer_gate = Probe(id="t", base=base,
                       wraps=(Wrap("pushfwd", ("exp",)), Wrap("truncate", (0.0, "inf"))),
                       spelling="direct", ordering="single", consumer=False, point=y)
    inner_gate = Probe(id="t", base=base,
                       wraps=(Wrap("truncate", (0.0, "inf")), Wrap("pushfwd", ("exp",))),
                       spelling="direct", ordering="single", consumer=False, point=y)
    # log phi(log 0.5) - log 0.5
    expected = (-0.9189385332046727 - 0.5 * math.log(0.5) ** 2) - math.log(0.5)
    assert abs(true_logpdf(outer_gate) - expected) < 1e-12
    assert true_logpdf(inner_gate) == -math.inf


def test_locscale_shift_and_scale_map_to_the_right_arguments():
    """Corroborates the `(shift, scale)` argument ORDER against scipy, not against
    a value derived by the same hand that wrote the rule.

    §06 defines `locscale(m, shift, scale)` as `pushfwd(x -> scale*x + shift, m)`,
    so over `Normal(0, 1)` it is exactly `Normal(shift, scale)` — a `loc`/`scale`
    parameterized frozen distribution, which scipy provides directly. If the two
    arguments were swapped the oracle would be scoring `Normal(2, 1)` here, and
    the mass invariant could not tell: any affine map preserves mass, so both
    orders integrate to 1.

    This is the one rule whose argument mapping the mass invariant is blind to.
    """
    loc, scale = 1.0, 2.0
    d = stats.norm(loc=loc, scale=scale)
    for y in (-3.0, -0.5, 0.0, 1.0, 2.0, 4.5):
        got = true_logpdf(_probe(Base("normal", (0.0, 1.0)),
                                 Wrap("locscale", (loc, scale)), y))
        assert got == pytest.approx(float(d.logpdf(y)), rel=1e-12), f"at y={y}"
    # And the swap really is distinguishable, so the check above has teeth.
    assert float(stats.norm(loc=2.0, scale=1.0).logpdf(1.0)) != pytest.approx(
        float(d.logpdf(1.0)), rel=1e-12)


def test_normalize_of_a_truncated_discrete_base_includes_the_boundary_atom():
    """§03: "`interval(lo, hi)` denotes the closed interval [lo, hi]", so an atom
    sitting exactly at `lo` is INSIDE the truncation set. scipy's `cdf(lo)` is
    `P(X <= lo)`, so `cdf(hi) - cdf(lo)` would drop it.

    `interval(2, inf)` over Poisson(3) makes this visible: the closed mass is
    `P(X >= 2) = 1 - cdf(1)`, whereas the open-lower reading gives
    `1 - cdf(2)` — they differ by `pmf(2)`, about 22% of the mass. At the atom
    k = 2 the normalized log-density is `logpmf(2, 3) - log(P(X >= 2))`.
    """
    d = stats.poisson(mu=3.0)
    closed_mass = float(1.0 - d.cdf(1.0))
    p = Probe(id="t", base=Base("poisson", (3.0,)),
              wraps=(Wrap("truncate", (2.0, "inf")), Wrap("normalize", ())),
              spelling="direct", ordering="single", consumer=False, point=2.0)
    expected = float(d.logpmf(2)) - math.log(closed_mass)
    assert true_logpdf(p) == pytest.approx(expected, rel=1e-12)
    # The dropped-atom reading is numerically distinct, so this test has teeth.
    assert math.log(float(1.0 - d.cdf(2.0))) != pytest.approx(math.log(closed_mass))


def test_normalize_of_a_weighted_base_divides_out_the_weight():
    """`totalmass(weighted(w, M)) = w * totalmass(M)`, and `M` is a §08 probability
    measure, so the mass is `w` and `normalize` cancels the weight exactly:
    `log w + log phi(x) - log w = log phi(x)`.
    """
    p = Probe(id="t", base=Base("normal", (0.0, 1.0)),
              wraps=(Wrap("weighted", (0.25,)), Wrap("normalize", ())),
              spelling="direct", ordering="single", consumer=False, point=0.5)
    assert true_logpdf(p) == pytest.approx(-1.0439385332046727, rel=1e-12)


def test_a_point_outside_the_forward_maps_image_has_no_preimage():
    """`exp` sends the reals to the positive reals, so a query at `y <= 0` is not
    in `pushfwd(exp, M)`'s variate space at all: density 0. Likewise `sqrt`'s
    image is the nonnegative reals, so `y < 0` is outside it.
    """
    base = Base("normal", (0.0, 1.0))
    assert true_logpdf(_probe(base, Wrap("pushfwd", ("exp",)), -1.0)) == -math.inf
    assert true_logpdf(_probe(base, Wrap("pushfwd", ("exp",)), 0.0)) == -math.inf
    assert true_logpdf(_probe(Base("gamma", (2.0, 1.0)),
                              Wrap("pushfwd", ("sqrt",)), -1.0)) == -math.inf


def test_a_preimage_past_the_float_range_returns_a_density_rather_than_raising():
    """`pushfwd(log, M)` at `y = 800` has preimage `e^800`, which overflows. It
    must RETURN `-inf`, not raise: an oracle that raises kills the sweep on a
    probe instead of scoring it. (Justified per-base — see `_PREIMAGE_OVERFLOWS`.)
    """
    for base in (Base("gamma", (2.0, 1.0)), Base("beta", (2.0, 3.0))):
        assert true_logpdf(_probe(base, Wrap("pushfwd", ("log",)), 800.0)) == -math.inf


def test_points_outside_a_base_support_are_minus_inf():
    """Both support gates in `_base_logpdf`: the continuous one, and the discrete
    one reached by a preimage that is a negative integer.
    """
    # Continuous: gamma is supported on x > 0, beta on (0, 1).
    assert true_logpdf(_probe(Base("gamma", (2.0, 1.0)), Wrap("identity", ()), -1.0)) \
        == -math.inf
    assert true_logpdf(_probe(Base("beta", (2.0, 3.0)), Wrap("identity", ()), 1.5)) \
        == -math.inf
    # Discrete: an exact integer, but outside Poisson's nonnegative support. Via
    # `pushfwd(neg, ·)` so the preimage is recovered rather than passed in.
    assert true_logpdf(_probe(Base("poisson", (3.0,)),
                              Wrap("pushfwd", ("neg",)), 2.0)) == -math.inf


def test_an_unimplemented_structure_raises_rather_than_guessing():
    with pytest.raises(OracleUnsupported):
        true_logpdf(_probe(Base("normal", (0.0, 1.0)), Wrap("superpose", ()), 0.5))
    with pytest.raises(OracleUnsupported):
        true_logpdf(_probe(Base("studentt", (3.0,)), Wrap("identity", ()), 0.5))


# --------------------------------------------------------------------------
# The total-mass invariant. This is the strongest check in the file, because it
# references NO hand-derived value: it integrates (or sums) the oracle's own
# density over the whole variate space and asserts the mass §06's algebra
# requires. A sign error in a log-volume term cannot survive it — the brief's
# `sqrt` rule, for instance, integrated to 0.32 instead of 1.
# --------------------------------------------------------------------------

# Each wrap's forward map, needed only to locate WHERE the pushed-forward measure
# lives so the quadrature limits are sane. This is the probe's own definition, not
# the density rule under test.
_FORWARD = {
    ("pushfwd", ("exp",)): math.exp,
    ("pushfwd", ("log",)): math.log,
    ("pushfwd", ("neg",)): lambda x: -x,
    ("pushfwd", ("sqrt",)): math.sqrt,
    ("affine", (2.0, 1.0)): lambda x: 2.0 * x + 1.0,
    ("locscale", (1.0, 2.0)): lambda x: 1.0 + 2.0 * x,
}


def _at(probe, y):
    return Probe(id=probe.id, base=probe.base, wraps=probe.wraps,
                 spelling=probe.spelling, ordering=probe.ordering,
                 consumer=probe.consumer, point=y)


def _expected_mass(base, wrap):
    """The mass §06/§13 require of the wrapped measure.

    `pushfwd` preserves mass by definition ((f_*M)(Y) = M(f^-1(Y))); `normalize`
    forces 1; `weighted`/`logweighted` scale it by the constant weight;
    `truncate` keeps the base's mass over the interval; `identity` leaves the
    base's own mass, which is 1 because §08's distributions are probability
    measures.
    """
    if wrap.kind == "weighted":
        return wrap.args[0]
    if wrap.kind == "logweighted":
        return math.exp(wrap.args[0])
    if wrap.kind == "truncate":
        lo, hi = _interval(wrap)
        d = _frozen(base)
        mass = float(d.cdf(hi) - d.cdf(lo))
        if base.kind == "poisson":
            # `interval(lo, hi)` is closed, and a discrete base can have an ATOM
            # at `lo` — which `cdf(hi) - cdf(lo)` excludes, because scipy's cdf is
            # `P(X <= lo)`. Poisson's atom at 0 carries 5% of the mass of
            # Poisson(3), so getting this wrong would look like a 5% oracle error.
            mass += float(d.pmf(lo))
        return mass
    return 1.0


def _breakpoints(base, wrap, fwd):
    """Quadrature breakpoints: the forward images of the base's quantile grid.

    Equal-probability sub-intervals, so each carries ~0.5% of the mass and `quad`
    converges on each — a single call over `pushfwd(exp, Gamma(2,1))`'s range of
    (1, 4e16) finds essentially nothing. Only the FORWARD map is used here (it is
    the probe's own definition); the density rule under test is not consulted.
    """
    d = _frozen(base)
    # Percentiles across the bulk, plus a per-decade refinement of both tails.
    # Without the tail refinement the outermost interval spans the whole tail --
    # for `pushfwd(exp, Gamma(2,1))` that is (764, 4e16), 14 orders of magnitude
    # with all the mass at the left edge, and `quad` reports 2e-12 instead of 0.01.
    tails = [10.0 ** -k for k in range(2, 16)]
    ps = sorted(set(tails + [i / 100.0 for i in range(1, 100)]
                    + [1.0 - t for t in tails]))
    ys = [fwd(float(d.ppf(p))) for p in ps]
    if wrap.kind == "truncate":
        # The gate is a step discontinuity; quad must not straddle it.
        lo, hi = _interval(wrap)
        ys += [b for b in (lo, hi) if math.isfinite(b) and min(ys) < b < max(ys)]
    return sorted(set(ys))


def _unique_shapes():
    """One probe per (base, wraps) the space actually generates, SCALAR bases only —
    the vector family's mass invariant is a lattice sum over a constraint surface
    and a simplex integral, neither of which this test's quadrature machinery
    expresses. Both live in
    `test_every_vector_measure_carries_the_mass_the_algebra_requires` instead."""
    shapes = {}
    for p in enumerate_probes():
        if is_vector_base(p.base):
            continue
        shapes.setdefault((p.base, p.wraps), p)
    return sorted(shapes.values(), key=lambda p: p.id)


@pytest.mark.parametrize("probe", _unique_shapes(), ids=lambda p: p.id.split(".")[0]
                         + "." + p.id.split(".")[1])
def test_every_wrapped_measure_carries_the_mass_the_algebra_requires(probe):
    base, wrap = probe.base, probe.wraps[0]
    want = _expected_mass(base, wrap)

    if base.kind == "poisson":
        # Counting measure: sum over the pushed-forward atoms, which are the
        # forward images of the base's own atoms.
        fwd = _FORWARD.get((wrap.kind, wrap.args), lambda x: x)
        got = math.fsum(math.exp(true_logpdf(_at(probe, fwd(float(n)))))
                        for n in range(0, 300))
        assert got == pytest.approx(want, abs=1e-9), (
            f"{probe.base.kind} + {wrap.kind}{wrap.args}: summed mass {got}, want {want}")
        return

    # Continuous: integrate over the forward image of the base's effective support
    # (the 1e-15 / 1-1e-15 quantiles bound tails carrying no mass), piecewise.
    fwd = _FORWARD.get((wrap.kind, wrap.args), lambda x: x)
    bps = _breakpoints(base, wrap, fwd)

    def density(y):
        return math.exp(true_logpdf(_at(probe, y)))

    got, err = 0.0, 0.0
    for a, b in zip(bps, bps[1:]):
        piece, perr = integrate.quad(density, a, b, limit=200)
        got, err = got + piece, err + perr
    # Summed over ~100 sub-intervals, so this is ~1e-9 apiece; the value
    # assertion below at rel=1e-9 is the real check.
    assert err < 1e-7, f"quadrature did not converge: error estimate {err}"
    assert got == pytest.approx(want, rel=1e-9), (
        f"{probe.base.kind} + {wrap.kind}{wrap.args}: integrated mass {got}, "
        f"want {want} over ({bps[0]}, {bps[-1]})")


# --------------------------------------------------------------------------
# The vector family. Same three gates as the scalar oracle: hand-derived closed
# forms, the total-mass invariant, and reproduction of the curated cases.
# --------------------------------------------------------------------------

_MULTINOMIAL = Base("multinomial", (5, (0.2, 0.3, 0.5)))
_DIRICHLET = Base("dirichlet", ((2.0, 3.0, 4.0),))


def _lgamma_multinomial(n, p, x):
    """§08 `Multinomial(n, p)`: "n!/prod_i x_i! prod_i p_i^{x_i} for x_i >= 0,
    sum_i x_i = n". Written out in log-gammas, independent of scipy."""
    t = math.lgamma(n + 1)
    for xi, pi in zip(x, p):
        t -= math.lgamma(xi + 1)
        t += xi * math.log(pi)
    return t


def _lgamma_dirichlet(alpha, x):
    """§08 `Dirichlet(alpha)`: "Gamma(||alpha||_1)/prod_i Gamma(alpha_i)
    prod_i x_i^{alpha_i - 1}". Written out in log-gammas, independent of scipy."""
    t = math.lgamma(sum(alpha)) - math.fsum(math.lgamma(a) for a in alpha)
    return t + math.fsum((a - 1.0) * math.log(xi) for a, xi in zip(alpha, x))


def test_bare_multinomial_matches_the_hand_derived_log_gamma_form():
    want = _lgamma_multinomial(5, (0.2, 0.3, 0.5), (1, 2, 2))
    got = true_logpdf(_probe(_MULTINOMIAL, Wrap("identity", ()), [1.0, 2.0, 2.0]))
    assert got == pytest.approx(want, abs=1e-12)
    assert got == pytest.approx(-2.0024805005437063, abs=1e-12)


def test_bare_dirichlet_matches_the_hand_derived_log_gamma_form():
    want = _lgamma_dirichlet((2.0, 3.0, 4.0), (0.2, 0.3, 0.5))
    got = true_logpdf(_probe(_DIRICHLET, Wrap("identity", ()), [0.2, 0.3, 0.5]))
    assert got == pytest.approx(want, abs=1e-12)
    assert got == pytest.approx(2.0228711901914425, abs=1e-12)


def _dirichlet_chart_mass(alpha):
    """§08's Dirichlet formula integrated over the (x1, x2) chart of stdsimplex(3)."""
    def f(x2, x1):
        x3 = 1.0 - x1 - x2
        if x3 <= 0.0:
            return 0.0
        return math.exp(_lgamma_dirichlet(alpha, (x1, x2, x3)))

    return integrate.dblquad(f, 0.0, 1.0, lambda x1: 0.0, lambda x1: 1.0 - x1,
                             epsabs=1e-12, epsrel=1e-12)


def test_sec08s_dirichlet_FORMULA_is_normalised_against_the_chart_measure():
    """What this establishes is what §08's FORMULA is normalised against — NOT what
    `Lebesgue(stdsimplex(n))` denotes. The two are different questions and the spec
    answers them incompatibly; see
    `test_sec03_and_sec06_name_the_surface_measure_which_sec08s_formula_is_not`.
    """
    mass, err = _dirichlet_chart_mass((2.0, 3.0, 4.0))
    assert err < 1e-9
    assert mass == pytest.approx(1.0, abs=1e-9), (
        f"§08's Dirichlet formula integrates to {mass} over the (x1, x2) chart")


def test_sec03_and_sec06_name_the_surface_measure_which_sec08s_formula_is_not():
    """The internal spec inconsistency, pinned as arithmetic so it cannot be argued
    away or quietly forgotten.

    §03 "Standard simplex": `Lebesgue(support = stdsimplex(n))` "measures **surface
    area** within the simplex". §06 "Lebesgue": for lower-dimensional embedded affine
    sets it is "the **intrinsic affine Lebesgue measure** on that set". Surface area
    on the embedded simplex is the (n-1)-dimensional Hausdorff measure.

    Its area element is sqrt(n) dx_1...dx_{n-1}: the tangent basis {e_i - e_n} has
    Gram matrix I + J, whose determinant is n. So §08's formula — which integrates to
    1 against the chart — integrates to sqrt(n) against the measure §03 and §06 name.
    A density cannot be normalised against both.

    The shipped rows carry the CHART reading, required for numerical parity with
    Stan/NumPyro/scipy, and are flagged `spec_wording_pending` in the verdict table.
    This test does not assert which reading is right; it asserts they differ by
    exactly log sqrt(n), which is what makes the flag necessary.
    """
    n = 3
    # The Gram determinant, computed rather than asserted from the formula.
    basis = [[1.0, 0.0, -1.0], [0.0, 1.0, -1.0]]           # e1 - e3, e2 - e3
    gram = [[sum(a * b for a, b in zip(u, v)) for v in basis] for u in basis]
    det = gram[0][0] * gram[1][1] - gram[0][1] * gram[1][0]
    assert det == pytest.approx(n, abs=1e-12), (
        f"the simplex tangent Gram determinant is {det}, expected n = {n}")
    assert simplex_chart_to_hausdorff_offset(n) == pytest.approx(
        0.5 * math.log(det), abs=1e-15)
    assert simplex_chart_to_hausdorff_offset(n) == pytest.approx(
        0.5493061443340549, abs=1e-15)

    # §08's formula integrates to sqrt(n), not 1, against the surface measure.
    chart_mass, _err = _dirichlet_chart_mass((2.0, 3.0, 4.0))
    assert chart_mass * math.sqrt(n) == pytest.approx(math.sqrt(n), abs=1e-8)

    # The two candidate values for the shipped row, both stated explicitly.
    chart = true_logpdf(_probe(_DIRICHLET, Wrap("identity", ()), (0.2, 0.3, 0.5)))
    assert chart == pytest.approx(2.0228711901914425, abs=1e-12)
    assert chart - simplex_chart_to_hausdorff_offset(n) == pytest.approx(
        1.4735650458573875, abs=1e-12)


def test_every_vector_base_declares_its_support_geometry():
    """`oracle._support_is_manifold` requires a declaration rather than defaulting,
    so a new vector base with a manifold support cannot silently take the
    ambient-Jacobian reading. Assert the two sets are equal so the failure lands here,
    at the roster, and not at whichever probe happens to run first."""
    assert set(VECTOR_SUPPORT_IS_MANIFOLD) == {b.kind for b in VECTOR_BASES}, (
        "space.VECTOR_SUPPORT_IS_MANIFOLD and VECTOR_BASES have drifted — every "
        "vector base must declare whether its support is a lower-dimensional "
        "manifold before the oracle will value a pushforward of it")


def test_an_undeclared_support_geometry_raises_rather_than_assuming_flat():
    """The inverted default, exercised. A base absent from the declaration must fail
    loudly; the old allowlist would have given it the ambient-Jacobian reading."""
    undeclared = Base("vonmisesfisher", ((1.0, 0.0, 0.0), 2.0))
    with pytest.raises(OracleUnsupported, match="support geometry"):
        _support_is_manifold(undeclared)


def test_a_zero_cell_is_inside_sec08s_dirichlet_support():
    """§08:532 gives the support inclusively — "p_i >= 0". Whether the density is
    finite at a zero cell is a separate question, decided by the exponent
    `alpha_i - 1`, and `oracle._dirichlet_logpdf` decides it per case rather than
    letting a support gate answer it (which is what made the gate wrong for
    alpha_i < 1)."""
    at_boundary = (0.0, 0.5, 0.5)
    assert in_support(_DIRICHLET, at_boundary), "§08 puts a zero cell in the support"

    # alpha_i > 1 at the zero cell: x_i^(alpha_i - 1) -> 0, so the density is 0.
    assert true_logpdf(_probe(_DIRICHLET, Wrap("identity", ()),
                              at_boundary)) == -math.inf
    # alpha_i < 1: §08's factor DIVERGES. scipy refuses the point; the oracle must
    # report +inf rather than -inf, which is what the old gate got wrong.
    heavy = Base("dirichlet", ((0.5, 3.0, 4.0),))
    assert in_support(heavy, at_boundary)
    assert true_logpdf(_probe(heavy, Wrap("identity", ()), at_boundary)) == math.inf
    # alpha_i == 1: the factor is 0^0 = 1, so the density is finite there, and equals
    # §08's product with the zero cell's factor simply omitted.
    alpha = (1.0, 3.0, 4.0)
    flat = Base("dirichlet", (alpha,))
    got = true_logpdf(_probe(flat, Wrap("identity", ()), at_boundary))
    assert math.isfinite(got), f"alpha_i == 1 at a zero cell must be finite, got {got}"
    want = (math.lgamma(sum(alpha)) - sum(math.lgamma(a) for a in alpha)
            + sum((a - 1.0) * math.log(xi)
                  for a, xi in zip(alpha, at_boundary) if xi > 0.0))
    assert got == pytest.approx(want, abs=1e-12)


def test_mixed_zero_cells_are_withheld_rather_than_resolved_by_case_order():
    """A diverging and a vanishing zero cell at once make §08's product `inf * 0`, a
    genuine indeterminate: `x = (0, 0, 1)` with `alpha = (0.5, 2, 4)` is
    `0^-0.5 * 0^1`. Whichever case the code tested first would decide the answer, so
    it withholds instead. Reachable on the simplex; reached by no shipped row."""
    mixed = Base("dirichlet", ((0.5, 2.0, 4.0),))
    point = (0.0, 0.0, 1.0)
    assert in_support(mixed, point), "the point is on §08's inclusive support"
    with pytest.raises(OracleUnsupported, match="indeterminate"):
        true_logpdf(_probe(mixed, Wrap("identity", ()), point))

    # Each limb ALONE is still answered, so the withhold is scoped to the mix.
    only_diverging = Base("dirichlet", ((0.5, 2.0, 4.0),))
    assert true_logpdf(_probe(only_diverging, Wrap("identity", ()),
                              (0.0, 0.5, 0.5))) == math.inf
    only_vanishing = Base("dirichlet", ((2.0, 3.0, 4.0),))
    assert true_logpdf(_probe(only_vanishing, Wrap("identity", ()),
                              (0.0, 0.5, 0.5))) == -math.inf


def test_a_vector_point_off_the_support_surface_is_minus_inf():
    """The support is §08's constraint SURFACE. A point in the bounding box but off
    the surface has density 0 — a gap-scan the oracle must not answer with a finite
    number, which is what scipy would raise on rather than return."""
    # sum = 4, not n = 5.
    assert true_logpdf(_probe(_MULTINOMIAL, Wrap("identity", ()),
                              [1.0, 1.0, 2.0])) == -math.inf
    # off the integer lattice
    assert true_logpdf(_probe(_MULTINOMIAL, Wrap("identity", ()),
                              [1.5, 1.5, 2.0])) == -math.inf
    # negative cell
    assert true_logpdf(_probe(_MULTINOMIAL, Wrap("identity", ()),
                              [-1.0, 4.0, 2.0])) == -math.inf
    # sum = 0.9, not 1
    assert true_logpdf(_probe(_DIRICHLET, Wrap("identity", ()),
                              (0.2, 0.3, 0.4))) == -math.inf
    # a negative cell: sums to 1 but leaves the simplex
    assert true_logpdf(_probe(_DIRICHLET, Wrap("identity", ()),
                              (-0.1, 0.6, 0.5))) == -math.inf
    # A ZERO cell is deliberately NOT here: §08's support is inclusive, so it is ON
    # the surface, and what happens there depends on `alpha_i` rather than on the
    # support. See `test_a_zero_cell_is_inside_sec08s_dirichlet_support`.


def test_a_vector_pushfwd_neg_is_volume_preserving_cell_wise():
    """`neg`'s per-cell forward log-volume is 0, so summing over cells is still 0 —
    and `neg` reflects the simplex onto a congruent copy, so the manifold reading
    agrees with the ambient one. The pushed-forward density equals the base's."""
    bare = true_logpdf(_probe(_DIRICHLET, Wrap("identity", ()), [0.2, 0.3, 0.5]))
    pushed = true_logpdf(_probe(_DIRICHLET, Wrap("pushfwd", ("neg",)),
                                [-0.2, -0.3, -0.5]))
    assert pushed == pytest.approx(bare, abs=1e-12)


def test_a_vector_pushfwd_over_a_manifold_support_withholds_a_value():
    """`pushfwd(exp, Dirichlet)` is the shape whose reference measure §06 and §08 do
    not name (see `oracle._MANIFOLD_SAFE_FORWARDS`): the ambient R^3 Jacobian, the
    2-D Hausdorff element on the image surface and the (y1, y2) chart give 1.0,
    0.6816 and 0.5 for the same volume term. The oracle must withhold rather than
    pick one — supplying a value would make it the authority for semantics nobody
    wrote down."""
    y = [math.exp(c) for c in (0.2, 0.3, 0.5)]
    with pytest.raises(OracleUnsupported):
        true_logpdf(_probe(_DIRICHLET, Wrap("pushfwd", ("exp",)), y))


def test_a_vector_pushfwd_over_a_counting_reference_takes_no_volume_term():
    """§08 gives Multinomial's density w.r.t. `iid(Counting(integers), k)`, and §06
    line 28's counting measure is not distorted by a bijection. So
    `pushfwd(exp, Multinomial)` at `y = exp(x)` is the pmf at `x`, with no
    `- sum(log y)`: a Lebesgue Jacobian here would be 5.0 (= sum x_i), which is the
    whole density's magnitude, not a correction to it.

    This shape is `_ENGINE_BLOCKED`, so no verdict-table row carries this number;
    the oracle still has to hold it, because that is the value the row will be
    checked against the day `flatppl-js` can evaluate the emitted gate."""
    x = VECTOR_INNER["multinomial"]
    y = [math.exp(c) for c in x]
    bare = true_logpdf(_probe(_MULTINOMIAL, Wrap("identity", ()), x))
    pushed = true_logpdf(_probe(_MULTINOMIAL, Wrap("pushfwd", ("exp",)), y))
    assert pushed == pytest.approx(bare, abs=1e-12)
    assert pushed == pytest.approx(-2.0024805005437063, abs=1e-12)


def test_a_scalar_truncation_over_a_vector_variate_withholds_a_value():
    """§03 makes `interval(lo, hi)` a set of REALS, so §06's ν(A) = M(A ∩ S) makes
    the restriction the zero measure over a vector variate. No §03 rule reads it
    cell-wise, so the oracle declines — exactly as it declines the record spelling.
    The determiniser REFUSES this shape, which is why it is a `spec_justified`
    refusal rather than a row with a value."""
    for base, point in ((_MULTINOMIAL, [1.0, 2.0, 2.0]),
                        (_DIRICHLET, [0.2, 0.3, 0.5])):
        with pytest.raises(OracleUnsupported):
            true_logpdf(_probe(base, Wrap("truncate", (0.0, 1.0)), point))


@pytest.mark.parametrize("base,wrap", vector_shapes(),
                         ids=lambda a: getattr(a, "kind", str(a)))
def test_every_vector_measure_carries_the_mass_the_algebra_requires(base, wrap):
    """The strongest vector check that references no hand-derived value: sum or
    integrate the oracle's own density over the whole variate space and assert the
    mass §06's algebra requires — 1 for every shape in the family, since §08's
    distributions are probability measures and `pushfwd` preserves mass by
    definition ((f_*M)(Y) = M(f^-1(Y))).

    Multinomial's variate space is the lattice {x in N_0^3 : sum x_i = n}, enumerated
    exactly — no quadrature, no truncation error. Dirichlet's is `stdsimplex(3)`,
    integrated over the (x1, x2) chart §08's formula is normalised on.

    **What this does NOT catch, stated because the earlier wording overclaimed it.**
    "A sign error in a per-cell volume term cannot survive the mass invariant" is
    FALSE for this family: every shape it reaches has a log-volume that is identically
    zero (`pushfwd(neg)`'s per-cell term is 0; `pushfwd(exp, Multinomial)` is
    counting-referenced so takes no term; `pushfwd(exp, Dirichlet)` is withheld), and
    a sign error on 0 is invisible. What these rows DO establish is that the inverse
    map is applied at all — a missing inversion evaluates Dirichlet off the simplex
    and gives -inf. The per-cell log-volume MAGNITUDE has no oracle-checked row
    anywhere in the family; an `MvNormal` base would make it checkable, since its §08
    reference is `iid(Lebesgue(reals), n)` over a full-dimensional support so the
    volume term is nonzero. Recorded as a follow-up in
    `flatppl-dev/density-sweep-notes.md`.

    The mass invariant is also blind to the `log sqrt(n)` reference-measure question
    on the Dirichlet base: it integrates against the same chart §08's formula
    normalises to, so it returns 1 under either wording. See
    `test_sec03_and_sec06_name_the_surface_measure_which_sec08s_formula_is_not`.
    """
    if wrap.kind == "truncate":
        pytest.skip("the oracle withholds a value for this shape, so there is "
                    "no density to integrate (see the test above)")

    fwd = {"exp": math.exp, "neg": lambda x: -x}.get(
        wrap.args[0] if wrap.kind == "pushfwd" else None, lambda x: x)

    def at(point):
        return true_logpdf(_probe(base, wrap, list(point)))

    if base.kind == "multinomial":
        n, _p = base.params
        total = math.fsum(
            math.exp(at([fwd(float(a)) for a in (i, j, n - i - j)]))
            for i in range(n + 1) for j in range(n + 1 - i))
        assert total == pytest.approx(1.0, abs=1e-9), (
            f"{base.kind} + {wrap.kind}{wrap.args}: summed lattice mass {total}")
        return

    # Dirichlet: integrate over the chart, pushing each chart point forward.
    def density(x2, x1):
        x3 = 1.0 - x1 - x2
        if x3 <= 0.0:
            return 0.0
        return math.exp(at([fwd(c) for c in (x1, x2, x3)]))

    mass, err = integrate.dblquad(density, 0.0, 1.0, lambda x1: 0.0,
                                  lambda x1: 1.0 - x1, epsabs=1e-11, epsrel=1e-11)
    assert err < 1e-8, f"quadrature did not converge: error estimate {err}"
    assert mass == pytest.approx(1.0, abs=1e-8), (
        f"{base.kind} + {wrap.kind}{wrap.args}: integrated mass {mass}")


def test_the_vector_oracle_has_a_curated_reproduction_gate_for_dirichlet_only():
    """State the vector family's validation coverage rather than leaving it implied.

    `corpora/stablehlo/dirichlet` carries 11 frozen `expected` values derived in its
    own `test.py` against `scipy.stats.dirichlet`, and the oracle reproduces all of
    them (`test_oracle_agrees_with_every_curated_case_it_can_express`).
    `Multinomial` appears in NO committed corpus case, so it has no reproduction
    gate: its licence is the hand-derived log-gamma form above plus the exact
    lattice-mass sum. That is a real coverage difference and it is asserted here so
    it cannot rot into an assumption.
    """
    from flatppl_testsuite.sweep.curated import curated_probes

    kinds = {p.base.kind for _n, p, _e in curated_probes()}
    assert "dirichlet" in kinds, (
        "the curated matcher no longer expresses corpora/stablehlo/dirichlet — the "
        "vector oracle just lost its only reproduction gate")
    assert "multinomial" not in kinds, (
        "a curated Multinomial case now exists; add it to the reproduction gate and "
        "update this test rather than leaving the coverage claim stale")
    n_dirichlet = sum(1 for _n, p, _e in curated_probes()
                      if p.base.kind == "dirichlet")
    assert n_dirichlet == 11, f"expected 11 curated Dirichlet cases, got {n_dirichlet}"


def test_oracle_agrees_with_every_curated_case_it_can_express():
    """The compositional oracle must reproduce the frozen `expected` of every
    curated `logdensity` case whose structure it can express.

    Those values were derived independently, by hand, in each dir's test.py.
    Agreement on that set is what licenses trusting this oracle on generated
    ground. A case the oracle cannot express is REPORTED, not skipped — an
    unvalidated oracle region must be visible.
    """
    from flatppl_testsuite.sweep.curated import curated_probes, unmatched_cases

    checked, unexpressible, wrong = 0, [], []
    for name, probe, expected in curated_probes():
        try:
            got = true_logpdf(probe)
        except OracleUnsupported as e:
            unexpressible.append((name, str(e)))
            continue
        checked += 1
        # `scoring.compare.compare_scalar` owns the +-inf and NaN rules; a
        # hand-rolled `abs(got - expected) > tol` silently PASSES a NaN, because
        # every comparison against NaN is False — so a NaN oracle result would be
        # counted as a validated case. One comparator, not two.
        try:
            compare_scalar(got, expected, {"atol": 1e-9, "rtol": 1e-9})
        except AssertionError as e:
            wrong.append((name, str(e)))

    assert not wrong, "oracle disagrees with curated cases:\n" + "\n".join(map(str, wrong))
    assert checked >= 8, (
        f"only {checked} curated cases were expressible — too few to validate the "
        f"oracle. Unexpressible: {unexpressible}"
    )
    unmatched = unmatched_cases()
    print(f"oracle validated on {checked} curated cases; "
          f"{len(unexpressible)} unexpressible, "
          f"{len(unmatched)} directories unmatched by the adapter")
    for why in unmatched:
        print(f"  unmatched: {why}")


def test_the_curated_matcher_maps_a_known_case_to_the_structure_we_expect():
    """The gate above is only as good as the matcher's fidelity: a case
    silently mapped to the WRONG structure would validate the oracle against
    the wrong value. Pin one match of each recognizer against the model text.
    """
    from flatppl_testsuite.sweep.curated import curated_probes

    by_name = {name: probe for name, probe, _ in curated_probes()}

    # corpora/fragment/norm_trunc: normalize(truncate(Normal(0,1), interval(-1,2)))
    p = by_name["fragment/norm_trunc"]
    assert p.base == Base("normal", (0.0, 1.0))
    assert p.wraps == (Wrap("truncate", (-1.0, 2.0)), Wrap("normalize", ()))
    assert p.point == 0.5

    # corpora/stablehlo/lognormal, 5th point: LogNormal(mu=0, sigma=0.75) at 0.5,
    # expressed per §08 as pushfwd(exp, Normal(mu, sigma)).
    p = by_name["stablehlo/lognormal#4"]
    assert p.base == Base("normal", (0.0, 0.75))
    assert p.wraps == (Wrap("pushfwd", ("exp",)),)
    assert p.point == 0.5


def test_every_curated_expected_value_is_the_frozen_one():
    """The adapter must not recompute an expected value — it must carry the
    committed one through. Re-read the JSON here and cross-check.
    """
    from flatppl_testsuite.sweep.curated import curated_probes

    frozen = {}
    for path in sorted((REPO / "corpora").glob("*/*/test.json")):
        d = json.loads(path.read_text())
        key = f"{path.parent.parent.name}/{path.parent.name}"
        frozen[key] = d.get("expected")

    for name, _probe, expected in curated_probes():
        stem, _, idx = name.partition("#")
        want = frozen[stem]
        if idx:
            want = want[int(idx)]
        if isinstance(want, str):
            want = float(want)
        assert expected == want, f"{name}: adapter reported {expected}, frozen is {want}"
