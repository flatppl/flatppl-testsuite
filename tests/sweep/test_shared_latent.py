"""The shared-latent family: its rendering, its multivariate oracle, and the
determiniser arm each shape actually reaches.

Same division of labour as `test_vector_arms.py`. The oracle tests here need no
engine: every expected value is derived in its own docstring from §06's rule or
from a hand-written covariance matrix, never from the determiniser. The arm tests
at the bottom need the binary, and they exist because **an outcome does not prove
which lowering produced it** — a row can claim to cover the shared-ancestor record
law while the emitted FlatPDL scored two independent marginals.
"""
import math
import subprocess
import tempfile
from pathlib import Path

import pytest
from scipy import stats

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.scoring.compare import compare_scalar
from flatppl_testsuite.sweep.classify import Outcome, classify
from flatppl_testsuite.sweep.oracle import (
    OracleUnsupported,
    _linear_gaussian_moments,
    _SINGULAR_EIGENVALUE_RATIO,
    true_logpdf,
)
from flatppl_testsuite.sweep.render import render
from flatppl_testsuite.sweep.space import (
    SHARED_LATENT_MU,
    SHARED_LATENT_POINTS,
    NormalNode,
    SharedLatentProbe,
    enumerate_shared_latent_probes,
    is_linear_gaussian,
    shared_latent_graph,
    shared_latent_shapes,
    shared_latent_variance,
)

TOL = {"atol": 1e-9, "rtol": 1e-9}


def _probe(shape, spelling, n=2, latent_query="none") -> SharedLatentProbe:
    return SharedLatentProbe(
        id=f"shared.{shape}.n{n}.{spelling}.{latent_query}", shape=shape, n=n,
        spelling=spelling, latent_query=latent_query, point=SHARED_LATENT_POINTS[:n])


# --------------------------------------------------------------------------
# Rendering: the source must be the model the row claims to probe
# --------------------------------------------------------------------------

# The rendered source of every (shape, spelling) pair at n = 2, pinned verbatim.
#
# Pinned as TEXT and not as a property, because the shape of the model is what the
# whole family means: a `chain` that rendered `mu = z` for its second field would
# be a `fan` wearing a chain's row id, its oracle would agree with it (both derive
# from the same graph), and the row would silently probe an arm already covered.
# The text is the one artifact that cannot drift with the graph.
_EXPECTED_SOURCE = {
    ("fan", "record_law"): """\
z = draw(Normal(mu = 0.4, sigma = 1.0))
f1 = draw(Normal(mu = z, sigma = 0.5))
f2 = draw(Normal(mu = z, sigma = 1.5))
lp = logdensityof(lawof(record(f1 = f1, f2 = f2)), record(f1 = 0.5, f2 = 0.7))
""",
    ("fan", "joint_kw"): """\
z = draw(Normal(mu = 0.4, sigma = 1.0))
f1 = draw(Normal(mu = z, sigma = 0.5))
f2 = draw(Normal(mu = z, sigma = 1.5))
lp = logdensityof(joint(f1 = lawof(f1), f2 = lawof(f2)), record(f1 = 0.5, f2 = 0.7))
""",
    ("fan", "joint_pos"): """\
z = draw(Normal(mu = 0.4, sigma = 1.0))
f1 = draw(Normal(mu = z, sigma = 0.5))
f2 = draw(Normal(mu = z, sigma = 1.5))
lp = logdensityof(joint(lawof(f1), lawof(f2)), [0.5, 0.7])
""",
    ("fan", "joint_ctor"): """\
lp = logdensityof(joint(f1 = Normal(mu = 0.4, sigma = 1.118033988749895), \
f2 = Normal(mu = 0.4, sigma = 1.8027756377319946)), record(f1 = 0.5, f2 = 0.7))
""",
    ("fan", "iid"): """\
z = draw(Normal(mu = 0.4, sigma = 1.0))
f1 = draw(Normal(mu = z, sigma = 0.5))
f2 = draw(Normal(mu = z, sigma = 1.5))
lp = logdensityof(iid(lawof(f1), 2), [0.5, 0.7])
""",
    ("chain", "record_law"): """\
z = draw(Normal(mu = 0.4, sigma = 1.0))
f1 = draw(Normal(mu = z, sigma = 0.5))
f2 = draw(Normal(mu = f1, sigma = 1.5))
lp = logdensityof(lawof(record(f1 = f1, f2 = f2)), record(f1 = 0.5, f2 = 0.7))
""",
    ("chain", "joint_kw"): """\
z = draw(Normal(mu = 0.4, sigma = 1.0))
f1 = draw(Normal(mu = z, sigma = 0.5))
f2 = draw(Normal(mu = f1, sigma = 1.5))
lp = logdensityof(joint(f1 = lawof(f1), f2 = lawof(f2)), record(f1 = 0.5, f2 = 0.7))
""",
    ("chain", "joint_pos"): """\
z = draw(Normal(mu = 0.4, sigma = 1.0))
f1 = draw(Normal(mu = z, sigma = 0.5))
f2 = draw(Normal(mu = f1, sigma = 1.5))
lp = logdensityof(joint(lawof(f1), lawof(f2)), [0.5, 0.7])
""",
    ("chain", "joint_ctor"): """\
lp = logdensityof(joint(f1 = Normal(mu = 0.4, sigma = 1.118033988749895), \
f2 = Normal(mu = 0.4, sigma = 1.8708286933869707)), record(f1 = 0.5, f2 = 0.7))
""",
    ("disjoint", "record_law"): """\
z1 = draw(Normal(mu = 0.4, sigma = 1.0))
f1 = draw(Normal(mu = z1, sigma = 0.5))
z2 = draw(Normal(mu = 0.4, sigma = 1.0))
f2 = draw(Normal(mu = z2, sigma = 1.5))
lp = logdensityof(lawof(record(f1 = f1, f2 = f2)), record(f1 = 0.5, f2 = 0.7))
""",
    ("disjoint", "joint_kw"): """\
z1 = draw(Normal(mu = 0.4, sigma = 1.0))
f1 = draw(Normal(mu = z1, sigma = 0.5))
z2 = draw(Normal(mu = 0.4, sigma = 1.0))
f2 = draw(Normal(mu = z2, sigma = 1.5))
lp = logdensityof(joint(f1 = lawof(f1), f2 = lawof(f2)), record(f1 = 0.5, f2 = 0.7))
""",
    ("disjoint", "joint_pos"): """\
z1 = draw(Normal(mu = 0.4, sigma = 1.0))
f1 = draw(Normal(mu = z1, sigma = 0.5))
z2 = draw(Normal(mu = 0.4, sigma = 1.0))
f2 = draw(Normal(mu = z2, sigma = 1.5))
lp = logdensityof(joint(lawof(f1), lawof(f2)), [0.5, 0.7])
""",
    ("disjoint", "joint_ctor"): """\
lp = logdensityof(joint(f1 = Normal(mu = 0.4, sigma = 1.118033988749895), \
f2 = Normal(mu = 0.4, sigma = 1.8027756377319946)), record(f1 = 0.5, f2 = 0.7))
""",
    ("singular", "record_law"): """\
z = draw(Normal(mu = 0.4, sigma = 1.0))
f1 = draw(Normal(mu = z, sigma = 0.5))
lp = logdensityof(lawof(record(f1 = f1, f2 = f1)), record(f1 = 0.5, f2 = 0.7))
""",
    ("singular", "joint_kw"): """\
z = draw(Normal(mu = 0.4, sigma = 1.0))
f1 = draw(Normal(mu = z, sigma = 0.5))
lp = logdensityof(joint(f1 = lawof(f1), f2 = lawof(f1)), record(f1 = 0.5, f2 = 0.7))
""",
    ("singular", "joint_pos"): """\
z = draw(Normal(mu = 0.4, sigma = 1.0))
f1 = draw(Normal(mu = z, sigma = 0.5))
lp = logdensityof(joint(lawof(f1), lawof(f1)), [0.5, 0.7])
""",
}


def test_each_spelling_renders_the_construct_it_names():
    assert set(_EXPECTED_SOURCE) == set(shared_latent_shapes()), (
        "the pinned sources and the generated (shape, spelling) pairs disagree — "
        "add the new pair's expected source rather than letting it render unpinned")
    for (shape, spelling), want in sorted(_EXPECTED_SOURCE.items()):
        got = render(_probe(shape, spelling)).source
        assert got == want, f"{shape}/{spelling} renders:\n{got}\nexpected:\n{want}"


def test_the_positional_spellings_query_a_vector_and_the_keyword_ones_a_record():
    """§06: a POSITIONAL `joint` combines component variates via `cat` (a vector),
    while the keyword form produces "a measure over a space of records". A record
    point against a vector variate is a different query, so the two must not be
    spelled alike."""
    for shape, spelling in shared_latent_shapes():
        src = render(_probe(shape, spelling)).source
        query = [ln for ln in src.splitlines() if ln.startswith("lp =")][0]
        if spelling in ("joint_pos", "iid"):
            assert "[0.5, 0.7]" in query, f"{shape}/{spelling}: {query}"
        else:
            assert "record(f1 = 0.5, f2 = 0.7)" in query, f"{shape}/{spelling}: {query}"


def test_the_constructor_joint_draws_no_field_it_never_consumes():
    """`joint_ctor`'s components are fresh constructors, so a drawn `f1` would sit
    unconsumed by the query — and `determinize` refuses a model with an unconsumed
    draw by design, which would turn every `joint_ctor` row into that refusal
    instead of the product-of-marginals arm the family needs."""
    ctor_shapes = [sh for sh, sp in shared_latent_shapes() if sp == "joint_ctor"]
    assert ctor_shapes, "no shape generates joint_ctor, so this test covers nothing"
    for shape in ctor_shapes:
        src = render(_probe(shape, "joint_ctor")).source
        assert "= draw(" not in src, f"{shape}: joint_ctor drew something\n{src}"
        # With the latent queried, the ONE draw is the latent itself -- it is consumed
        # by that query, so it is not the unconsumed-draw hazard above.
        queried = render(_probe(shape, "joint_ctor", latent_query="before")).source
        assert queried.count("= draw(") == 1, queried


def test_the_latent_query_axis_adds_a_second_query_on_the_shared_node():
    """`before`/`after` must query the LATENT, and on the side of the main query
    their name says. A pin because the ordering is the whole content of the axis:
    both variants rendering the same order would make two of the three rows
    duplicates that still read as coverage."""
    before = render(_probe("fan", "record_law", latent_query="before")).source.splitlines()
    after = render(_probe("fan", "record_law", latent_query="after")).source.splitlines()
    assert before[-2].startswith("lp_latent = logdensityof(lawof(z), 0.1)")
    assert before[-1].startswith("lp = logdensityof(")
    assert after[-2].startswith("lp = logdensityof(")
    assert after[-1].startswith("lp_latent = logdensityof(lawof(z), 0.1)")


# --------------------------------------------------------------------------
# The multivariate oracle: covariance derived by hand, not by the code
# --------------------------------------------------------------------------

# The covariance of each shape's traced law, WRITTEN OUT. With
# `SHARED_LATENT_SIGMA_Z = 1.0` and field sigmas (0.5, 1.5, 2.0):
#
#   fan       — f_i = z + eps_i, so Var(f_i) = 1 + s_i^2 and every off-diagonal is
#               Var(z) = 1. §06's own worked example: "`joint(a = lawof(a), b =
#               lawof(b))` has cross-covariance Var(z) = s^2".
#   chain     — f_1 = z + eps_1 and f_i = f_{i-1} + eps_i, so the variances
#               accumulate (1.25, 3.5, 7.5) and Cov(f_i, f_j) = Var(f_min(i,j)):
#               everything f_j adds after f_i is independent of f_i.
#   disjoint  — a private z_i per field, so Cov(f_i, f_j) = 0 and the matrix is
#               diagonal with the SAME variances as `fan`. That is what makes it a
#               control rather than a fourth shape: identical marginals, no
#               cross-terms.
#   singular  — every field is one draw, so every entry is Var(f_1) = 1.25 and the
#               matrix has rank 1.
_EXPECTED_COV = {
    ("fan", 2): [[1.25, 1.0], [1.0, 3.25]],
    ("fan", 3): [[1.25, 1.0, 1.0], [1.0, 3.25, 1.0], [1.0, 1.0, 5.0]],
    ("chain", 2): [[1.25, 1.25], [1.25, 3.5]],
    ("chain", 3): [[1.25, 1.25, 1.25], [1.25, 3.5, 3.5], [1.25, 3.5, 7.5]],
    ("disjoint", 2): [[1.25, 0.0], [0.0, 3.25]],
    ("disjoint", 3): [[1.25, 0.0, 0.0], [0.0, 3.25, 0.0], [0.0, 0.0, 5.0]],
    ("singular", 2): [[1.25, 1.25], [1.25, 1.25]],
    ("singular", 3): [[1.25, 1.25, 1.25], [1.25, 1.25, 1.25], [1.25, 1.25, 1.25]],
}


@pytest.mark.parametrize("shape,n", sorted(_EXPECTED_COV), ids=lambda v: str(v))
def test_the_covariance_matches_the_hand_derived_matrix(shape, n):
    nodes, fields = shared_latent_graph(shape, n, "record_law")
    mean, cov = _linear_gaussian_moments(nodes, fields)
    assert mean == pytest.approx([SHARED_LATENT_MU] * n, abs=1e-12), (
        "every mean in every shape here is the latent's own — a location that "
        "propagates unchanged through `mu = parent`")
    for i in range(n):
        assert cov[i] == pytest.approx(_EXPECTED_COV[(shape, n)][i], abs=1e-12), (
            f"{shape} n={n} row {i}: {cov[i]}")


def test_the_chain_covariance_matches_the_reviewer_derived_unit_sigma_law():
    """At unit sigmas the chain z -> a -> b has law MvNormal([0,0], [[2,2],[2,3]]).

    Derived independently in the wave-F1 review and quoted in that wave's brief, so
    it is a cross-check against a number this repo did not produce. Var(a) = 2,
    Cov(a,b) = Var(a) = 2, Var(b) = Var(a) + 1 = 3.
    """
    nodes = {
        "z": NormalNode("z", None, 0.0, 1.0),
        "a": NormalNode("a", "z", 0.0, 1.0),
        "b": NormalNode("b", "a", 0.0, 1.0),
    }
    mean, cov = _linear_gaussian_moments(nodes, ("a", "b"))
    assert mean == [0.0, 0.0]
    assert cov == [[2.0, 2.0], [2.0, 3.0]]


def test_the_two_variance_derivations_agree():
    """`space.shared_latent_variance` walks the scalar recursion;
    `oracle._linear_gaussian_moments` builds L L^T. They are separate code and
    `render` depends on the first (for `joint_ctor`'s matched sigmas) while every
    oracle value depends on the second, so a disagreement would put the model and
    its own audit out of step."""
    for shape in ("fan", "chain", "disjoint", "singular"):
        for n in (2, 3):
            nodes, fields = shared_latent_graph(shape, n, "record_law")
            _mean, cov = _linear_gaussian_moments(nodes, fields)
            for i, f in enumerate(fields):
                assert cov[i][i] == pytest.approx(
                    shared_latent_variance(nodes, f), abs=1e-12), (
                    f"{shape} n={n} field {f}: loading-matrix variance {cov[i][i]} "
                    f"vs recursion {shared_latent_variance(nodes, f)}")


def test_a_child_is_listed_before_its_parent_without_changing_the_moments():
    """`curated.py` builds its graph from parsed source, so declaration order is the
    model author's. The moment code resolves parents by recursion; this pins that
    rather than leaving it to the generated family, which always happens to list a
    parent first."""
    forward = {
        "z": NormalNode("z", None, 0.4, 1.0),
        "a": NormalNode("a", "z", 0.0, 0.5),
    }
    reversed_order = {
        "a": NormalNode("a", "z", 0.0, 0.5),
        "z": NormalNode("z", None, 0.4, 1.0),
    }
    assert (_linear_gaussian_moments(forward, ("a",))
            == _linear_gaussian_moments(reversed_order, ("a",)))


@pytest.mark.parametrize("shape", ["fan", "chain", "disjoint"])
@pytest.mark.parametrize("n", [2, 3])
def test_the_three_equivalent_spellings_share_one_oracle_value(shape, n):
    """§06 "Equivalent record law", verbatim: "`joint(a = lawof(a), b = lawof(b))`
    is equivalent to `lawof(record(a = a, b = b))`; the positional form is the
    corresponding `cat` law".

    One measure, three spellings — so one oracle value. The determiniser is judged
    against that equivalence too, which is a correctness signal needing no oracle at
    all: three spellings that lower to three different densities are wrong whichever
    of them the oracle happens to agree with.
    """
    values = {s: true_logpdf(_probe(shape, s, n))
              for s in ("record_law", "joint_kw", "joint_pos")}
    assert len(set(values.values())) == 1, values


@pytest.mark.parametrize("shape,n", [(s, n) for s in ("fan", "chain", "disjoint")
                                     for n in (2, 3)])
def test_the_constructor_joint_is_the_product_of_the_matched_marginals(shape, n):
    """§06, the sentence after the equivalence: "a `joint` of two constructor
    measures with the same marginals has cross-covariance 0".

    So this arm's value is the plain SUM of the scalar marginal log-densities — no
    covariance, no matrix. Computed here with `scipy.stats.norm`, which shares no
    code with the multivariate path it is checking.
    """
    nodes, fields = shared_latent_graph(shape, n, "record_law")
    want = math.fsum(
        stats.norm(loc=SHARED_LATENT_MU,
                   scale=math.sqrt(shared_latent_variance(nodes, f))).logpdf(p)
        for f, p in zip(fields, SHARED_LATENT_POINTS[:n]))
    compare_scalar(true_logpdf(_probe(shape, "joint_ctor", n)), want, TOL)


@pytest.mark.parametrize("shape,n", [(s, n) for s in ("fan", "chain") for n in (2, 3)])
def test_a_shared_node_makes_the_traced_law_differ_from_the_product(shape, n):
    """The contrast has to BITE, or `joint_ctor` is a duplicate row. With a shared
    node the cross-covariance is nonzero, so the traced law and the
    matched-marginal product are different measures and must disagree at the
    point."""
    traced = true_logpdf(_probe(shape, "record_law", n))
    product = true_logpdf(_probe(shape, "joint_ctor", n))
    assert abs(traced - product) > 1e-3, (
        f"{shape} n={n}: traced {traced} and product {product} agree, so this "
        "shape's cross-covariance is not actually exercised")


@pytest.mark.parametrize("n", [2, 3])
def test_the_disjoint_control_makes_the_traced_and_product_laws_coincide(n):
    """The disjointness control. Nothing is shared, so §06's traced law IS the
    product of the marginals, and the two spellings must agree to float precision.

    This is the row that catches a lowering which manufactures correlation — a
    `fan`-shaped closed form applied wherever two fields have any latent at all
    would show up here and nowhere else in the family.
    """
    compare_scalar(true_logpdf(_probe("disjoint", "record_law", n)),
                   true_logpdf(_probe("disjoint", "joint_ctor", n)), TOL)


@pytest.mark.parametrize("n", [2, 3])
def test_iid_does_not_pick_up_the_shared_ancestor(n):
    """§06 `iid(M, size)` is "the product measure M^(x)N", so N independent copies
    of ONE marginal — `iid(lawof(f1), n)` over a shared-latent graph is n
    independent copies of f1's law, not a correlated n-vector.

    Checked against n scalar `scipy.stats.norm` terms of f1's own marginal.
    """
    nodes, fields = shared_latent_graph("fan", n, "iid")
    v = shared_latent_variance(nodes, fields[0])
    want = math.fsum(stats.norm(loc=SHARED_LATENT_MU, scale=math.sqrt(v)).logpdf(p)
                     for p in SHARED_LATENT_POINTS[:n])
    compare_scalar(true_logpdf(_probe("fan", "iid", n)), want, TOL)


def test_the_singular_shapes_are_withheld_and_nothing_else_is():
    """§06 "Singular joints": "the joint law has no density w.r.t. the product
    reference measure ... a density query is a static error where statically
    detectable, and is otherwise refused by the engine."

    So the oracle must WITHHOLD, and the trap is that a number IS available — the
    product of the two marginals is finite and plausible and is the density of
    nothing. Per `flatppl-dev/density-sweep-notes.md`, supplying it would make this
    module the authority for semantics nobody wrote down.

    Both directions asserted: no non-singular member may be withheld either, or the
    singularity test would be quietly eating rows the family exists to score.
    """
    withheld, valued = [], []
    for p in enumerate_shared_latent_probes():
        try:
            true_logpdf(p)
            valued.append(p)
        except OracleUnsupported:
            withheld.append(p)
    assert {p.shape for p in withheld} == {"singular"}, (
        f"withheld shapes: {sorted({p.shape for p in withheld})}")
    assert all(p.shape == "singular" for p in withheld)
    assert not [p for p in valued if p.shape == "singular"], (
        "a singular probe was given a value")
    assert len(withheld) == 6, f"expected 6 singular probes, got {len(withheld)}"


def test_the_singularity_threshold_is_nowhere_near_an_admissible_probe():
    """`oracle._SINGULAR_EIGENVALUE_RATIO` claims to sit orders of magnitude away
    from every non-singular member. Asserted as arithmetic rather than left as a
    comment: a family member whose conditioning drifted toward the threshold would
    start being withheld for a reason that has nothing to do with §06.
    """
    from scipy import linalg

    worst = 1.0
    for shape in ("fan", "chain", "disjoint"):
        for n in (2, 3):
            nodes, fields = shared_latent_graph(shape, n, "record_law")
            _m, cov = _linear_gaussian_moments(nodes, fields)
            eig = sorted(float(v) for v in linalg.eigvalsh(cov))
            worst = min(worst, eig[0] / eig[-1])
    assert worst > 1e5 * _SINGULAR_EIGENVALUE_RATIO, (
        f"the worst-conditioned non-singular member sits at {worst:.3e}, too close "
        f"to the singularity threshold {_SINGULAR_EIGENVALUE_RATIO:.3e}")

    for n in (2, 3):
        nodes, fields = shared_latent_graph("singular", n, "record_law")
        _m, cov = _linear_gaussian_moments(nodes, fields)
        eig = sorted(float(v) for v in linalg.eigvalsh(cov))
        assert eig[0] / eig[-1] < _SINGULAR_EIGENVALUE_RATIO, (
            f"singular n={n} is not detected as singular: eigenvalues {eig}")


def test_the_latent_query_axis_does_not_move_the_oracle():
    """§04 makes `logdensityof` a QUERY on a measure, not a conditioning of the
    model, so a second query on the shared latent cannot change the joint's
    density. The three `latent_query` rows of one (shape, n, spelling) therefore
    carry one oracle value — and a determiniser that answers them differently is
    caught by the rows disagreeing with each other, before the oracle matters."""
    for shape, spelling in shared_latent_shapes():
        if shape == "singular":
            continue
        values = {q: true_logpdf(_probe(shape, spelling, 2, q))
                  for q in ("none", "before", "after")}
        assert len(set(values.values())) == 1, f"{shape}/{spelling}: {values}"


# --------------------------------------------------------------------------
# The reproduction gate: the frozen corpus values license this oracle
# --------------------------------------------------------------------------

def test_the_multivariate_oracle_reproduces_the_curated_shared_latent_values():
    """The multivariate path's licence, and the reason `curated.py` learned to
    express a multi-field joint at all.

    `corpora/fragment/shared_latent_{record,joint,joint_positional}` each freeze
    -2.5171832107434002, derived in their own `test.py` from
    `MvNormal(mu0 * 1, s0**2 * J + diag(sigma**2))` by hand and (per
    `flatppl-dev/density-sweep-notes.md`) cross-checked against the closed form and
    quadrature. Reproducing them is what makes this oracle something other than a
    second transcription of the determiniser's own algebra.

    Without this the family would ship in `Multinomial`'s position — a generated
    space judged by an oracle with no frozen independent value behind it.
    """
    from flatppl_testsuite.sweep.curated import curated_probes

    cases = {name: (p, exp) for name, p, exp in curated_probes()
             if is_linear_gaussian(p)}
    want = {"fragment/shared_latent_record", "fragment/shared_latent_joint",
            "fragment/shared_latent_joint_positional"}
    assert want <= set(cases), (
        f"the curated matcher stopped expressing {sorted(want - set(cases))} — the "
        "multivariate oracle just lost its reproduction gate")
    for name in sorted(want):
        probe, expected = cases[name]
        compare_scalar(true_logpdf(probe), expected, TOL)


def test_the_singular_refusal_fixture_is_not_a_reproduction_gate():
    """`corpora/fragment/joint_singular_refusal` freezes `"nan"` as a SENTINEL, not
    a value — its own test.py says "NaN never matches" — so it must stay out of
    `curated_probes()` with that reason recorded, rather than becoming a case that
    validates nothing and could report agreement the day the oracle returned NaN.
    """
    from flatppl_testsuite.sweep.curated import curated_probes, unmatched_cases

    names = {name for name, _p, _e in curated_probes()}
    assert "fragment/joint_singular_refusal" not in names
    reasons = [w for w in unmatched_cases() if "joint_singular_refusal" in w]
    assert reasons and "NaN" in reasons[0], (
        f"the exclusion is no longer recorded with its reason: {reasons}")


# --------------------------------------------------------------------------
# Arms: which lowering the determiniser actually reached
# --------------------------------------------------------------------------

pytestmark_binary = pytest.mark.skipif(
    not CONFIG.flatppl_bin.exists(), reason="needs the flatppl binary")


def _binding(emitted: str, name: str) -> str:
    """One binding's right-hand side from emitted FlatPDL, continuation lines included.

    A binding opens at column 0 as `<name> = `; the fan's `lp` spans several indented
    lines, so "the line starting with `lp =`" is not enough. Comparing whole MODULES
    instead would not work either: the latent's own binding differs across the
    `latent_query` values (`z = 0.0` unqueried versus `z = 0.1` queried), which is a
    difference in the pinned draw and not in the family's answer.
    """
    lines = emitted.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith(f"{name} = "))
    out = [lines[start]]
    for ln in lines[start + 1:]:
        if ln and not ln[0].isspace():
            break
        out.append(ln)
    return "\n".join(out).strip()


def _determinize(probe) -> tuple[int, str, str]:
    """`(exit code, emitted FlatPDL, stderr)` for one probe."""
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "m.flatppl"
        src.write_text(render(probe).source)
        out = Path(tmp) / "m.flatpdl.flatppl"
        proc = subprocess.run(
            [str(CONFIG.flatppl_bin), "determinize", str(src), "-o", str(out)],
            capture_output=True, text=True)
        return proc.returncode, (out.read_text() if out.exists() else ""), proc.stderr


@pytestmark_binary
def test_the_singular_shapes_refuse_rather_than_emitting_a_density():
    """§06 "Singular joints" makes the refusal the conformant answer, and
    `table._spec_justified` reports it as one. Asserted against the determiniser's
    own exit code, because a row that merely SAYS `spec_justified` proves nothing
    about what the binary did — the same reason `test_vector_arms.py` asserts each
    vector arm actually fires.
    """
    for spelling in ("record_law", "joint_kw", "joint_pos"):
        for n in (2, 3):
            code, emitted, stderr = _determinize(_probe("singular", spelling, n))
            assert code == 3, (
                f"singular/{spelling} n={n}: exit {code}, not the refusal §06 "
                f"requires\nemitted:\n{emitted}")
            assert "draw" in stderr.lower() or "field" in stderr.lower(), (
                f"singular/{spelling} n={n}: refusal does not name the shape: "
                f"{stderr.strip()}")


@pytestmark_binary
def test_the_shared_ancestor_arm_lowers_a_correlated_form_not_two_marginals():
    """An outcome does not prove which lowering produced it.

    A `fan` record law scored as two independent marginals would give a finite
    number, LOWERS, and a row that reads as coverage — and it would be wrong by
    exactly the cross-covariance this family exists to check. So assert the
    NEGATIVE direction too: the emitted FlatPDL for a shared-ancestor probe must
    NOT be the same text as the emitted FlatPDL for the matched-marginal
    `joint_ctor` probe, whose independence is the thing being contrasted.
    """
    code_shared, shared, stderr = _determinize(_probe("fan", "record_law"))
    if code_shared == 3:
        pytest.skip(f"this determiniser refuses the fan record law: {stderr.strip()}")
    code_ctor, ctor, _ = _determinize(_probe("fan", "joint_ctor"))
    assert code_shared == 0
    if code_ctor == 0:
        assert shared != ctor, (
            "the shared-ancestor law and the matched-marginal product emit "
            "IDENTICAL FlatPDL, so the correlated arm is not being reached")


@pytestmark_binary
@pytest.mark.parametrize("latent_query", ["before", "after"])
def test_the_latent_query_survives_into_the_emitted_flatpdl(latent_query):
    """The `latent_query` axis's arm assertion — without it the axis is unpinned
    where it matters.

    Every other test of this axis stops short of the determiniser: the render test
    checks the source text, and the oracle test asserts the three rows share one
    value. **Both would still pass if the second query were eliminated as dead
    code.** The rows would agree trivially, the table would report the axis
    covered, and this family's own rule — a row whose gate never emitted proves
    nothing — would be unmet on exactly one axis.

    So assert the `lp_latent` binding survives AND that its right-hand side is the
    latent's own scalar law. Asserted on THAT LINE, not by counting
    `builtin_logdensityof` over the whole module: the `fan` record law lowers to a
    closed-form rank-1-update expression carrying no `builtin_logdensityof` at all,
    so a global count of two is simply false here — the family's query contributes
    none and `lp_latent` contributes the only one.
    """
    code, emitted, stderr = _determinize(
        _probe("fan", "record_law", latent_query=latent_query))
    assert code == 0, f"determinize refused: {stderr.strip()}"
    line = next((ln for ln in emitted.splitlines()
                 if ln.startswith("lp_latent =")), None)
    assert line is not None, (
        f"the second query's binding was eliminated:\n{emitted}")
    # §08's `Normal(mu, sigma)` for the latent prior, scored at
    # SHARED_LATENT_LATENT_POINT. Named rather than counted, so an unrelated call
    # elsewhere in the module cannot satisfy this.
    assert "builtin_logdensityof(Normal, record(mu = 0.4, sigma = 1.0), 0.1)" in line, (
        f"lp_latent survived but is not the latent's own law: {line}")


@pytestmark_binary
def test_the_latent_query_does_not_perturb_the_familys_own_lowering():
    """The invariant the `latent_query` axis exists for, asserted on the emitted
    FlatPDL rather than on the oracle.

    §04 makes `logdensityof` a query on a measure, not a conditioning of the model,
    so a second query on the shared latent must not change the joint's density. The
    oracle enforces that by construction (it never reads `latent_query`), which is
    exactly why the oracle cannot detect a violation — a determiniser that DID
    perturb the joint would show up as three disagreeing rows, and this pins the
    stronger statement: the `lp` expression is byte-identical across all three
    values of the axis.

    Worth pinning because the determiniser binds the latent to the query's point
    (`z = 0.1` appears in the emitted module), so there IS a mechanism by which a
    second query could reach the family's own answer.
    """
    exprs = {}
    for latent_query in ("none", "before", "after"):
        code, emitted, stderr = _determinize(
            _probe("fan", "record_law", latent_query=latent_query))
        assert code == 0, f"{latent_query}: determinize refused: {stderr.strip()}"
        exprs[latent_query] = _binding(emitted, "lp")
    assert len(set(exprs.values())) == 1, (
        "the latent query changed the family's own lowering:\n"
        + "\n".join(f"{k}: {v}" for k, v in exprs.items()))


@pytestmark_binary
@pytest.mark.parametrize("shape,spelling", shared_latent_shapes(),
                         ids=lambda v: str(v))
def test_every_shared_latent_shape_reaches_a_classified_outcome(shape, spelling):
    """Every pair must land on LOWERS or REFUSES, never MALFORMED.

    MALFORMED means the determiniser exited 0 and the scorer could not evaluate
    what it emitted, which `test_gate.py` bans from the committed table — and
    unlike a refusal it is indistinguishable there from a determiniser defect.
    """
    v = classify(_probe(shape, spelling))
    assert v.outcome != Outcome.MALFORMED, f"{shape}/{spelling}: marker={v.marker}"
