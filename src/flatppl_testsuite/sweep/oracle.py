"""The independent oracle: a probe's TRUE log-density.

Transcribes a SUBSET of §13 "Output reduction"'s enumerated rules over scipy
base densities — the ones the probe space generates: `weighted`, `logweighted`,
`normalize`, `truncate`, and `pushfwd` (including its `affine`/`locscale`
spellings), plus `joint` and `iid` over the LINEAR-GAUSSIAN family only
(`linear_gaussian_logpdf`). §13's remaining rules are **not** implemented:
`superpose`, `jointchain` and `kchain` raise `OracleUnsupported`, as does any
base outside the four in `space.BASES` and the two in `space.VECTOR_BASES`, and
so does a `joint` whose components are not `Normal` draws over `Normal` parents.
That list is the module's scope, stated here rather than left to be discovered
at runtime.

The vector family (`space.VECTOR_BASES`) has its own fold, `_vector_logpdf`:
the same rules applied cell-wise, with `pushfwd` over a manifold support
withheld rather than guessed (`_MANIFOLD_SAFE_FORWARDS`).

The shared-latent family (`space.SharedLatentProbe`) is not a fold at all:
`_shared_latent_logpdf` assembles the multivariate normal the draw graph
implies, because a cross-covariance is not expressible as a composition of
scalar rules.

It walks the `Probe` — the structure the generator built — so it needs no
parser and no type inference, and it never reads the determiniser's output.
Authority order is maths > spec > code.

Each rule below cites the clause it implements. A rule with no citation is a
bug: the point of this module is that it is derivable from the spec by someone
who has never seen the determiniser.
"""
from __future__ import annotations

import math

# `scipy.linalg`, not `numpy.linalg`: `scipy` is a declared pixi dependency and
# numpy is only a transitive one, so importing numpy directly would work today and
# break the day scipy's own dependency changed shape.
from scipy import linalg, stats

from flatppl_testsuite.sweep.space import (
    SHARED_LATENT_COMPOSITION,
    VECTOR_SUPPORT_IS_MANIFOLD,
    Base,
    LinearGaussianProbe,
    NormalNode,
    Probe,
    SharedLatentProbe,
    Wrap,
    in_support,
    is_linear_gaussian,
    is_shared_latent,
    is_vector_base,
    shared_latent_graph,
)


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
    variates, counting measure for discrete variates".

    `multinomial` is the vector family's discrete member: §08 gives its density
    "w.r.t. `iid(Counting(integers), k)`", a product of counting measures.
    """
    return base.kind in ("poisson", "multinomial")


# --------------------------------------------------------------------------
# The vector family. A parallel path rather than a widening of the scalar fold
# below: the verdict table is frozen history, so the scalar arithmetic must stay
# bit-identical, and a shared fold threading `float | list[float]` through every
# branch is how a silent change to it gets in.
# --------------------------------------------------------------------------

def simplex_chart_to_hausdorff_offset(n: int) -> float:
    """`log sqrt(n)` — the factor between `Lebesgue(stdsimplex(n))`, which §03 defines
    as the COORDINATE measure, and the surface (Hausdorff) measure of the embedded
    simplex, which §03 notes "is larger by the factor sqrt(n)".

    Not applied anywhere in the density path: §03, §06 and §08 all name the coordinate
    measure, so §08's formula needs no correction. This exists so `test_oracle` can pin
    that convention as arithmetic rather than as a comment — the two measures differ by
    0.549 in log-density at n = 3, which is the size of the error the spec's wording
    now rules out.

    Derivation, not a fudge factor: parameterise the simplex by the chart
    (x_1, ..., x_{n-1}) with x_n = 1 - sum. The tangent basis is {e_i - e_n}, whose
    Gram matrix is I + J with J all-ones of size n-1. J's eigenvalues are n-1 once
    and 0 otherwise, so I + J has eigenvalues n and 1, and det = n. The surface area
    element is therefore sqrt(n) dx_1 ... dx_{n-1}, and a density normalised against
    the chart is sqrt(n) times the one normalised against surface area.
    """
    return 0.5 * math.log(n)


def _vector_base_logpdf(base: Base, x) -> float:
    """A `VECTOR_BASES` member's log-density at `x`, transcribed from §08.

    scipy's parameterisations are §08's verbatim here, unlike `Gamma` above:

    * §08 `Multinomial(n, p)` — "Density w.r.t. `iid(Counting(integers), k)`:
      n!/prod_i x_i! prod_i p_i^{x_i} for x_i >= 0, sum_i x_i = n", which is
      `scipy.stats.multinomial(n, p).logpmf(x)` argument for argument.
    * §08 `Dirichlet(alpha)` — "Density w.r.t. `Lebesgue(stdsimplex(n))`:
      Gamma(||alpha||_1)/prod_i Gamma(alpha_i) prod_i x_i^{alpha_i - 1}", which is
      `scipy.stats.dirichlet(alpha).logpdf(x)`.

    ## `Lebesgue(stdsimplex(n))` is the COORDINATE measure — settled, and normative

    The spec now says so in all three places, so this is no longer a reading to
    justify. §03 "Standard simplex":

        `Lebesgue(support = stdsimplex(n))` is the (n-1)-dimensional coordinate
        Lebesgue measure on the simplex: the image of dx_1 ... dx_{n-1} under the
        chart that appends x_n = 1 - sum_{i<n} x_i (dropping any other coordinate
        gives the same measure). ... It is not the surface (Hausdorff) measure of the
        embedded simplex, which is larger by the factor sqrt(n).

    §06 "Lebesgue" agrees: for lower-dimensional embedded affine sets such as
    `stdsimplex(n)` it is "the coordinate Lebesgue measure of the set's free
    coordinates ..., not its surface area". §08's Dirichlet entry names it under the
    formula: "The reference measure is the coordinate measure dx_1 ... dx_{n-1} of
    `Lebesgue(stdsimplex(n))`."

    So §08's formula, `scipy.stats.dirichlet` and this function all agree, and they
    agree with Stan and NumPyro — the numerical parity the project requires.
    `simplex_chart_to_hausdorff_offset` is kept only to express the sqrt(n) factor
    §03 mentions, which `test_oracle` uses to pin the convention; nothing in the
    density path applies it.

    **Consequence for this oracle's authority, still worth stating:** on the Dirichlet
    base it agrees with the engine because both transcribe §08's formula, so it cannot
    independently detect a reference-measure error in that formula. It is an
    independent check of the ALGEBRA around the base (the wraps, the support gate, the
    total mass), not a second opinion on §08 itself.

    The support gate is `space.in_support`, which is the constraint SURFACE (sum
    x_i = n; sum x_i = 1) and not a bounding box — scipy would raise on an
    off-surface point rather than return -inf, and an exception here would abort the
    sweep on a probe instead of scoring it.
    """
    if not in_support(base, x):
        return -math.inf
    if base.kind == "multinomial":
        n, p = base.params
        cells = [float(round(c)) for c in x]     # snapped: see `_base_logpdf`
        return float(stats.multinomial(n=n, p=list(p)).logpmf(cells))
    if base.kind == "dirichlet":
        (alpha,) = base.params
        return _dirichlet_logpdf(list(alpha), list(x))
    raise OracleUnsupported(f"vector base {base.kind}")


def _dirichlet_logpdf(alpha: list[float], x: list[float]) -> float:
    """§08's Dirichlet formula, with the ZERO-CELL boundary handled explicitly.

    §08's support is inclusive (`p_i >= 0`), so a zero cell is a point of the support
    (`space.in_support`), but scipy's `dirichlet.logpdf` REJECTS it — it requires
    strictly positive cells. The three cases come straight from the exponent in
    §08's `prod_i x_i^(alpha_i - 1)` at x_i = 0:

    * `alpha_i > 1` — the factor is 0, so the density is 0 and the log-density -inf;
    * `alpha_i == 1` — the factor is 0^0 = 1, so the cell contributes nothing;
    * `alpha_i < 1` — the factor DIVERGES, so the density is +inf there.

    Only the first case arises for this family's `alpha = (2, 3, 4)`, and it is what
    the previous strictly-positive support gate happened to return. The other two are
    implemented because getting them from a support gate rather than from the density
    is what made that gate wrong in general.

    **MIXED zero cells are withheld, not resolved by case order.** With a diverging
    factor and a vanishing one at once — `x = (0, 0, 1)`, `alpha = (0.5, 2, 4)` gives
    `0^-0.5 * 0^1`, i.e. `inf * 0` — the product is a genuine indeterminate, and any
    answer here is an artefact of which case the code happens to test first. §08 gives
    the density as that product and no limit convention alongside it, so this module
    withholds rather than pick a limb, exactly as it withholds elsewhere.
    """
    zeros = [i for i, xi in enumerate(x) if xi <= 0.0]
    if zeros:
        diverging = [i for i in zeros if alpha[i] < 1.0]
        vanishing = [i for i in zeros if alpha[i] > 1.0]
        if diverging and vanishing:
            raise OracleUnsupported(
                f"Dirichlet at a point with both a diverging (alpha_i < 1) and a "
                f"vanishing (alpha_i > 1) zero cell: cells {diverging} and "
                f"{vanishing} make §08's product an inf * 0 indeterminate, and §08 "
                "gives no limit convention for it")
        if diverging:
            return math.inf
        if vanishing:
            return -math.inf
        # Every zero cell has alpha_i == 1: those factors are 1, so drop them and
        # evaluate the rest of §08's product directly.
        t = math.lgamma(math.fsum(alpha)) - math.fsum(math.lgamma(a) for a in alpha)
        return t + math.fsum((a - 1.0) * math.log(xi)
                             for a, xi in zip(alpha, x) if xi > 0.0)
    return float(stats.dirichlet(alpha).logpdf(x))


# Forward maps whose log-volume element is unambiguous over a support that is a
# lower-dimensional MANIFOLD of the ambient variate space — `Dirichlet`'s
# `stdsimplex(n)` (§08), which is an (n-1)-manifold inside `cartpow(reals, n)`.
#
# **Only a map that preserves the manifold's own volume element qualifies, and of
# the maps this family generates that is `neg` alone.** `neg` is an isometry of R^n
# (it reflects the simplex onto a congruent copy), so its restricted volume element
# is 1 and the log-volume is 0 whichever reference is meant. `exp` is not, and there
# the reference measure of the IMAGE is genuinely not determined by §06 or §08:
# three defensible readings of `pushfwd(exp, Dirichlet(2, 3, 4))`'s volume term at
# (e^0.2, e^0.3, e^0.5) disagree —
#
#   * 1.0     — the ambient R^3 Jacobian log-det, `sum_i log y_i` (which on the
#               simplex is just `sum_i x_i` = 1);
#   * 0.6816  — the 2-D Hausdorff element on the image surface,
#               `0.5 * log det(U^T diag(y)^2 U)` for an orthonormal basis U of the
#               simplex's tangent space {v : sum v_i = 0};
#   * 0.5     — the (y_1, y_2) coordinate chart's `log|d(y_1,y_2)/d(x_1,x_2)|`.
#
# They differ by 0.32 and 0.5 in log-density, so no reading is a rounding detail.
# §06 says `logvolume` "generalizes the log-absolute-determinant of the Jacobian to
# mappings between spaces of different dimension"; this map is R^3 -> R^3 while the
# MEASURE lives on a 2-manifold, which is a case §06 does not name. Per
# `flatppl-dev/density-sweep-notes.md` ("the oracle must not assume semantics the
# spec does not give"), withhold: supplying one of the three would make this module
# the authority for semantics nobody wrote down, and would report a spec gap as a
# determiniser bug.
_MANIFOLD_SAFE_FORWARDS = {"neg"}


def _support_is_manifold(base: Base) -> bool:
    """Whether `base`'s support is a lower-dimensional manifold of the variate space.

    Reads `space.VECTOR_SUPPORT_IS_MANIFOLD`, where every `VECTOR_BASES` member is
    REQUIRED to declare its geometry, and raises `OracleUnsupported` for a base that
    has not. That direction is deliberate: an allowlist with a permissive default
    would give a newly added manifold-support base (`LKJ`, a von Mises-Fisher) the
    ambient-Jacobian reading and a silently wrong number. Absent means loud, never
    "probably flat". `test_oracle` asserts the declaration set equals the base set,
    so the failure lands at the roster and not at a probe.
    """
    if base.kind not in VECTOR_SUPPORT_IS_MANIFOLD:
        raise OracleUnsupported(
            f"{base.kind} does not declare its support geometry in "
            "space.VECTOR_SUPPORT_IS_MANIFOLD: a pushforward's volume element "
            "cannot be derived without knowing whether the support is a "
            "lower-dimensional manifold")
    return VECTOR_SUPPORT_IS_MANIFOLD[base.kind]


def _vector_logpdf(probe: Probe) -> float:
    """`true_logpdf` for a `VECTOR_BASES` probe. Wraps peeled OUTERMOST FIRST, as
    in the scalar fold, and every map applied CELL-WISE — §06's elementwise
    pushforward is one scalar map per cell, so its log-volume is the SUM of the
    per-cell terms.

    **A discrete base accumulates NO log-volume, decided at the accumulation site
    rather than reset afterwards.** §08 gives `Multinomial`'s density w.r.t.
    `iid(Counting(integers), k)`, and §06 line 28's counting measure is not distorted
    by a bijection, so there is no volume term to sum. Reading the reset twenty lines
    below the accumulation is how a reader ends up unsure whether a Multinomial
    pushforward carries one.

    **Coverage limitation, stated where the code is:** no probe this family generates
    exercises a NONZERO vector log-volume. `pushfwd(neg)`'s per-cell term is
    identically 0, `pushfwd(exp, Multinomial)` is counting-referenced so it takes no
    term at all, and `pushfwd(exp, Dirichlet)` is withheld below. The per-cell SUM is
    therefore not oracle-discriminated by any row: a sign error, or a
    per-cell/whole-vector confusion, would survive here. An `MvNormal` base would
    make the arm checkable — §08 gives its support as `cartpow(reals, n)` and its
    reference as `iid(Lebesgue(reals), n)`, so the volume term is nonzero and
    unambiguous. Recorded as a follow-up in `flatppl-dev/density-sweep-notes.md`.
    """
    x = list(probe.point)
    logv = 0.0
    # Read once, up front, so an undeclared base fails before any arithmetic.
    discrete = _is_discrete(probe.base)
    manifold = _support_is_manifold(probe.base)

    for i in range(len(probe.wraps) - 1, -1, -1):
        w = probe.wraps[i]
        if w.kind == "identity":
            continue
        if w.kind == "pushfwd":
            op = w.args[0]
            if op not in _INVERSE:
                raise OracleUnsupported(f"pushfwd {op}")
            if manifold and op not in _MANIFOLD_SAFE_FORWARDS:
                raise OracleUnsupported(
                    f"pushfwd({op}, {probe.base.kind}): §06 scopes `Lebesgue` to "
                    "lower-dimensional embedded AFFINE sets, and this forward's "
                    "image of the support is not affine, so no §06 rule gives the "
                    "reference measure of the image")
            if not all(_IN_FORWARD_IMAGE[op](c) for c in x):
                return -math.inf      # some cell has no preimage
            if any(_PREIMAGE_OVERFLOWS.get(op, lambda _: False)(c) for c in x):
                return -math.inf
            if not discrete:
                logv += math.fsum(_LOGVOLUME_AT_PREIMAGE[op](c) for c in x)
            x = [_INVERSE[op](c) for c in x]
        elif w.kind == "truncate":
            # A SCALAR truncation region over a VECTOR variate has no defined
            # density, for the reason the record spelling has none: §06 "Support
            # restriction" gives `truncate(M, S)` as ν(A) = M(A ∩ S), and §03 makes
            # `interval(lo, hi)` a set of REALS while the variate is a vector, so
            # A ∩ S is empty and ν is the zero measure — -inf at every point.
            #
            # There is no cell-wise reading to compute instead. §03 spells no
            # vector-valued `interval` (`cartpow(interval(lo, hi), n)` is the set of
            # vectors, and is a different second argument), and §04's auto-splat is a
            # calling convention for a callable's named arguments. The determiniser
            # REFUSES this shape rather than emitting -inf, which is why the probe is
            # a `spec_justified` refusal and not a row with a value.
            raise OracleUnsupported(
                "a scalar truncation set over a vector variate: no §06 rule")
        else:
            raise OracleUnsupported(f"wrap {w.kind} over a vector variate")

    return _vector_base_logpdf(probe.base, x) - logv


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
# has preimage `e^y = inf`, which raises rather than returning a density -- and it
# has to be RETURNED, not raised, or the sweep dies on a probe instead of scoring
# it.
#
# **The `-inf` here is justified per-base, not in general.** §06's posreals guard
# restricts `log` to `gamma`/`beta`, and both have density 0 that far out
# (`beta`'s support ends at 1; `gamma`'s decays like `e^-x`). A base added later
# whose density decays SLOWER than `1/x` would make `-inf` wrong here, because the
# `+y` volume term grows without bound: the correct limit is then not 0. If a base
# joins `space.BASES` for which `log` is well-formed, re-derive this.
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
# Whether a query point lies in the forward map's IMAGE. This is the image, not
# the domain -- `{y > 0}` is where `exp` sends things, not where it accepts them.
# The distinction matters because the query point lives in the pushed-forward
# measure's space: a `y` outside the image has no preimage, so it is not in the
# variate space at all and the density there is 0.
_IN_FORWARD_IMAGE = {
    "exp":  lambda y: y > 0.0,
    "log":  lambda y: True,
    "neg":  lambda y: True,
    "sqrt": lambda y: y >= 0.0,
}


# --------------------------------------------------------------------------
# The shared-latent family: a multivariate composition, not a scalar fold.
# --------------------------------------------------------------------------

# Relative tolerance for calling a covariance SINGULAR: the smallest eigenvalue
# measured against the largest, so the test is scale-free.
#
# A structural threshold, not a fudge factor. The family's singular shape has an
# EXACTLY zero eigenvalue in exact arithmetic (two rows of the loading matrix are
# identical), so the only thing the tolerance absorbs is the float error of
# `eigvalsh` on a 2x2 or 3x3 matrix, which is a few ULP of the largest eigenvalue.
# The nearest NON-singular member is `chain`, whose worst conditioning in this
# family is around 1e-2 -- five orders of magnitude away from this threshold, so no
# admissible probe is anywhere near it and no singular one escapes.
# `test_shared_latent` asserts both directions rather than leaving that as prose.
_SINGULAR_EIGENVALUE_RATIO = 1e-12


def _linear_gaussian_moments(nodes: dict[str, NormalNode],
                            fields: tuple[str, ...]) -> tuple[list[float], list[list[float]]]:
    """`(mean, cov)` of the named nodes, for a graph of `Normal` draws whose
    locations are their parents.

    Derived through the LOADING MATRIX, not through a covariance recursion. Give
    every node its own independent standard normal, write each node as an affine
    function of them,

        f = mu_f + sum_k L[f][k] * eps_k,      eps ~ N(0, I)

    and read off `mean[i] = mu_of(fields[i])` and `cov = L L^T`. A root contributes
    `L[root] = sigma_root * e_root`; a child inherits its parent's whole row and
    adds its own noise, `L[child] = L[parent] + sigma_child * e_child`. That is the
    definition of the model, so it needs no per-shape case analysis and it handles
    `fan`, `chain`, `disjoint` and `singular` with the same six lines — which is
    the point, because a per-shape covariance formula is where a hand-derived
    off-diagonal goes wrong unseen.

    **A repeated name in `fields` is meaningful and is not deduplicated.** `singular`
    binds every field to one draw, so two rows of `L` are identical and `cov` is
    rank-deficient. `_refuse_singular_joint` is what turns that into §06's refusal;
    silently collapsing the duplicate here would answer a question about a
    2-dimensional law with the density of a 1-dimensional one.
    """
    order: list[str] = list(nodes)
    index = {name: i for i, name in enumerate(order)}
    loading: dict[str, list[float]] = {}
    mean: dict[str, float] = {}

    def resolve(name: str) -> list[float]:
        """`name`'s loading row and mean, memoized.

        Recursive rather than a single pass over `nodes`, so a graph listing a child
        before its parent resolves the same — `curated.py` builds its graph from
        parsed source, where the declaration order is the model author's and not
        this module's to require.
        """
        if name in loading:
            return loading[name]
        node = nodes[name]
        if node.parent is not None:
            row = list(resolve(node.parent))
            mean[name] = mean[node.parent] + node.mu
        else:
            row = [0.0] * len(order)
            mean[name] = node.mu
        row[index[name]] += node.sigma
        loading[name] = row
        return row

    rows = [resolve(f) for f in fields]
    cov = [[math.fsum(a * b for a, b in zip(ri, rj)) for rj in rows] for ri in rows]
    return [mean[f] for f in fields], cov


def _refuse_singular_joint(cov: list[list[float]]) -> None:
    """Raise when `cov` has no density w.r.t. the product reference measure.

    §06 "Singular joints", verbatim:

        When one component's variate is determined by the others given the shared
        ancestors (the same draw referenced twice, a deterministic transform of
        another component), the joint law has no density w.r.t. the product
        reference measure. Sampling is well-defined; a density query is a static
        error where statically detectable, and is otherwise refused by the engine.

    So the oracle WITHHOLDS rather than returning a number, per
    `flatppl-dev/density-sweep-notes.md`'s rule ("when a probe's model is ill-typed
    or unspecified, the oracle must WITHHOLD a value"). There is a number available
    here and it is the trap: the product of the two marginals is finite, plausible,
    and not the density of anything.

    Decided from the COVARIANCE, never from `probe.shape`. Keying on the shape name
    would make the withholding a property of this family's labels instead of a
    property of the law, so a future shape that happened to be singular for a
    different reason (a deterministic transform of a field) would take the
    non-singular path and produce that plausible wrong number.

    scipy is also not asked: `multivariate_normal` raises on a singular covariance
    rather than returning `-inf`, and an exception escaping mid-sweep aborts the run
    instead of scoring a probe (the same reason `_vector_base_logpdf` gates the
    support itself).
    """
    eig = sorted(float(v) for v in linalg.eigvalsh(cov))
    if eig[-1] <= 0.0 or eig[0] / eig[-1] < _SINGULAR_EIGENVALUE_RATIO:
        raise OracleUnsupported(
            "singular joint: the covariance is rank-deficient (smallest eigenvalue "
            f"{eig[0]:.3e} against largest {eig[-1]:.3e}), so §06 'Singular joints' "
            "gives the law no density w.r.t. the product reference measure and the "
            "query is refused rather than valued")


def linear_gaussian_logpdf(nodes: dict[str, NormalNode], fields: tuple[str, ...],
                           point: tuple[float, ...], composition: str) -> float:
    """The log-density of a composed linear-Gaussian law at `point`.

    Shared by the generated family and by `curated.py`'s matcher, so the committed
    corpus's frozen shared-latent values are what license this path — the same
    relation the scalar fold has to its 56 curated cases. Taking the two apart
    would leave the multivariate oracle ungated, which is the gap
    `flatppl-dev/density-sweep-notes.md` records against `Multinomial`.

    `composition` selects §06's rule:

    * `"traced"` — §06 "Equivalent record law": `joint(a = lawof(a), b = lawof(b))`
      "is equivalent to `lawof(record(a = a, b = b))`", and a node shared between
      component traces "remains a single node of the composed trace". So the
      covariance is the graph's own, cross-terms included, and §06's worked example
      is this family's `fan`: "`joint(a = lawof(a), b = lawof(b))` has
      cross-covariance Var(z) = s^2".
    * `"iid"` — §06 `iid(M, size)`: "the product measure M^(x)N". N independent
      copies of ONE marginal, so the covariance is diagonal with that marginal's
      variance repeated, whatever ancestry the marginal itself has.

    The reference measure is the product Lebesgue measure over the fields (§06
    "Reference measure for product measures": "the reference for the product is the
    product rho_1 (x) rho_2 (x) ... on the joint variate space"), every component
    being a continuous scalar — which is exactly what
    `scipy.stats.multivariate_normal.logpdf` is normalised against.

    scipy shares no algebra with the determiniser here, so this is a genuinely
    independent check and not the same-source agreement the Dirichlet rows have:
    the determiniser lowers the record law symbolically through a Sherman-Morrison
    plus matrix-determinant-lemma expression, while this assembles L L^T and calls a
    Cholesky.
    """
    mean, cov = _linear_gaussian_moments(nodes, fields)
    if composition == "iid":
        mean = [mean[0]] * len(fields)
        v = cov[0][0]
        cov = [[v if i == j else 0.0 for j in range(len(fields))]
               for i in range(len(fields))]
    elif composition != "traced":
        raise OracleUnsupported(f"composition {composition}")
    _refuse_singular_joint(cov)
    return float(stats.multivariate_normal(mean=mean, cov=cov).logpdf(list(point)))


def _shared_latent_logpdf(probe: SharedLatentProbe) -> float:
    """`true_logpdf` for a `SharedLatentProbe`.

    The `latent_query` axis contributes NOTHING to this value, deliberately and not
    by omission. §04 makes `logdensityof` a query on a measure, not a conditioning
    of the model, so a second query on the shared latent cannot move the joint's
    density — which is the invariant the axis exists to test. The three
    `latent_query` rows of one (shape, n, spelling) therefore carry the same oracle,
    and a determiniser that answers them differently is caught by the rows
    disagreeing with each other before the oracle is even consulted.
    """
    nodes, fields = shared_latent_graph(probe.shape, probe.n, probe.spelling)
    return linear_gaussian_logpdf(
        nodes, fields, probe.point, SHARED_LATENT_COMPOSITION[probe.spelling])


def true_logpdf(probe: Probe | SharedLatentProbe | LinearGaussianProbe) -> float:
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

    A `VECTOR_BASES` probe is delegated to `_vector_logpdf`, which is the same
    algebra applied cell-wise. A `SharedLatentProbe` is delegated to
    `_shared_latent_logpdf`, which is not the same algebra at all — a joint over N
    fields is a multivariate law, and no fold over scalar §13 rules expresses a
    cross-covariance.
    """
    if is_shared_latent(probe):
        return _shared_latent_logpdf(probe)
    if is_linear_gaussian(probe):
        return linear_gaussian_logpdf(probe.graph(), probe.fields, probe.point,
                                      probe.composition)

    if is_vector_base(probe.base):
        return _vector_logpdf(probe)

    x = probe.point
    logv = 0.0          # accumulated FORWARD log-volume, subtracted at the end
    extra = 0.0         # weight and normalization terms: point-independent shifts

    # A scalar truncation region over a RECORD variate has no defined density.
    # §06 "Support restriction" gives `truncate(M, S)` as ν(A) = M(A ∩ S). Under the
    # `record` spelling the measure's variate is `record(x: real)` while `interval`
    # is a set of reals (§03), so A ∩ S is empty and ν is the zero measure —
    # -inf everywhere, at every query point.
    #
    # There is no field-wise reading to compute instead. §03 spells no
    # record-valued interval, and §04 "Calling conventions"' auto-splat is a
    # calling convention for a callable's named arguments that explicitly does not
    # apply to "a record given alongside other arguments" — `truncate(M, S)` has
    # two. So this oracle must not supply a value: asserting the field-wise answer
    # would pin unspecified semantics and report a spec gap as a determiniser bug.
    if probe.spelling == "record" and any(w.kind == "truncate" for w in probe.wraps):
        raise OracleUnsupported("truncate over a record variate: no §06 rule")

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
                if not _IN_FORWARD_IMAGE[op](x):
                    return -math.inf      # no preimage: not in the variate space
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

    Three shapes are implemented: a bare base (§08 opens "the built-in
    distributions (i.e. probability measures)", so the mass is exactly 1), a
    truncated base (the base's mass over the closed interval), and a `weighted`
    base (the weight). Anything else is an explicit gap, not a guess —
    `normalize(pushfwd(...))` is left unimplemented even though a pushforward
    preserves mass, because nothing generates it and an untested branch here is
    indistinguishable from a wrong one. Every branch that DOES exist is covered by
    a direct test, which is the same principle applied consistently.
    """
    inner = probe.wraps[:idx]
    if len(inner) == 0:
        return 0.0
    if len(inner) == 1 and inner[0].kind == "truncate":
        lo, hi = _interval(inner[0])
        d = _frozen(probe.base)
        mass = float(d.cdf(hi) - d.cdf(lo))
        if _is_discrete(probe.base):
            # §03: "`interval(lo, hi)` denotes the closed interval [lo, hi]", and
            # the `truncate` gate above is `lo <= x <= hi` to match. scipy's `cdf`
            # is `P(X <= lo)`, so `cdf(hi) - cdf(lo)` EXCLUDES an atom sitting
            # exactly at `lo` -- which a discrete base can have and a continuous
            # one cannot. Poisson(3)'s atom at 0 carries 5% of its mass, so
            # omitting it puts `normalize(truncate(Poisson(3), interval(0, inf)))`
            # at mass 0.95021 instead of 1.0: a log-density error of 0.0511
            # attributed to the determiniser.
            mass += float(d.pmf(lo))
        return math.log(mass)
    if len(inner) == 1 and inner[0].kind == "weighted":
        # totalmass(weighted(w, M)) = w * totalmass(M), and M here is a bare §08
        # distribution, so the mass is w.
        return math.log(inner[0].args[0])
    raise OracleUnsupported(f"totalmass of {[w.kind for w in inner]}")
