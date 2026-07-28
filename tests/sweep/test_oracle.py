"""The oracle is validated against hand-derived closed forms BEFORE it is
trusted on generated ground. Every expected value here is derived in the
docstring from the spec's rule, never taken from the determiniser."""
import json
import math
from pathlib import Path

import pytest
from scipy import integrate

from flatppl_testsuite.sweep.oracle import (
    OracleUnsupported,
    _frozen,
    _interval,
    true_logpdf,
)
from flatppl_testsuite.sweep.space import Base, Probe, Wrap, enumerate_probes

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
    """One probe per (base, wraps) the space actually generates."""
    shapes = {}
    for p in enumerate_probes():
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
        if got == expected:            # covers the -inf == -inf case
            continue
        if abs(got - expected) > 1e-9 + 1e-9 * abs(expected):
            wrong.append((name, got, expected))

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
