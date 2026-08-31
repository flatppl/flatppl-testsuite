"""The sampler sweep's probe space: base families x combinator wraps.

SHAPE OF THE SPACE. Two generated axes plus a targeted list, deliberately not
one full cross product:

* Every base family from `oracle.FAMILIES` under the identity wrap. This is the
  per-distribution roster — one row per sampleable REGISTRY entry.
* The combinator wraps. These split by whether their closed form is derivable
  from the base's moments alone:
    - BASE-AGNOSTIC wraps (`iid`, `weighted`, `normalize` of a probability
      measure, affine `pushfwd`) have an exact oracle for ANY base, because the
      algebra only moves the base's own moments around. They cross a spread of
      bases: `WRAP_BASES`.
    - COMPOSED wraps (`superpose` mixtures, `kchain`, `truncate`, and `iid` over
      those) need distinct components or a kernel, so their construction and
      oracle are written out for one Normal base. Crossing them over every
      family would mean inventing a family-specific shift for each, and the
      oracle would stop being checkable by hand.

  A full 26 x 16 cross product would be 416 rows of which most are either
  ill-typed or carry an oracle nobody can verify. The point of the space is that
  every row's expected value is independently derivable, so the space stops
  where the derivations do.

* `TARGETED`, a written-out list for constructs NEITHER axis can express: a
  scalar primitive mapped over a shaped atom batch, a `normalize` whose mass
  moves with a latent that is not the variate, and an `iid` over a `normalize`
  whose own IMPORTANCE WEIGHTS are the law. The first is a wrap over a shape the
  `WRAP_BASES` roster has no member for (a vector variate); the second needs a
  second binding the two axes never introduce (the latent); the third needs a
  variate-dependent weight, which the agnostic `normalize_prob` wrap
  deliberately excludes (its base is already a probability measure). Each entry
  states its own closed form, as `COMPOSED` does.

WHY A MIXTURE NEEDS DISTINCT COMPONENTS. `superpose` of two copies of the SAME
measure is that measure again, so its moments are unchanged no matter which
component each draw picks — a branch-pinning defect is invisible to it. The
mixture rows below use `Normal(-3, 1)` and `Normal(+3, 1)`, three sigma apart,
so pinning moves the mean by 3 and the variance by 9: the defect this class
exists to catch is thousands of sigma out, not a fraction of the tolerance.

CROSS-COORDINATE COVARIANCE. Every wrap with `k > 1` carries `cov = 0.0`: §06
"Joint composition" defines `iid(M, size)` as the product measure
$M^{\\otimes N}$, and a product measure has independent coordinates. This is the
check that catches the branch-pinning class (`iid(superpose(...), k)` selecting
one component per coordinate position), which leaves each MARGINAL looking
plausible while the coordinates are locked together.

NOT IN v1, with the reason:

* `ksuperpose` — absent from flatppl-js entirely. Verified, not assumed:
  `ksuperpose(Normal, [0.5, 0.5])(...)` raises the static diagnostic
  `Undefined variable 'ksuperpose'`, and `grep -rln ksuperpose` over the
  flatppl-js repo returns nothing. It is also not in the merged spec — §06 only
  gains it on an unmerged `ksuperpose-spec` branch. flatppl-rust refuses its
  sample lowering explicitly (`crates/determinizer/src/sample.rs`,
  `refuse_ksuperpose_sample`).
* `MvNormal`, `Wishart`, `LKJ`, `Multinomial` — not REGISTRY entries in
  `sampler-registry.ts` (the matrix families live in flatppl-rust's `src/`, per
  that repo's CLAUDE.md), so they are outside this sweep's stated roster. A
  literal-covariance `MvNormal` also does not currently materialise: `cov =
  [[2.0, 0.5], [0.5, 1.0]]` is a vector-of-vectors and §03 wants a true rank-2
  array, while the `rowstack(...)` spelling the error message suggests fails
  deeper, inside the affine bijection. That is an UN-VERIFIED observation, not a
  finding — it needs its own investigation and is recorded in the report.
* `markovchain`, `kscan` — no sample lowering exists anywhere in flatppl-js.
* Multi-seed replication. The engine's materialiser context takes a SCALAR
  `rootKey` where the contract is a two-lane PhiloxKey `[k0, k1]`, so `foldIn`
  collapses and the stream does not vary with the seed (the defect carded in
  flatppl-dev/TODO-flatppl-js.md at `worker.ts:240`). Every row is therefore ONE
  fixed sample of size `n`, not a distribution over seeds. That makes the gate
  reproducible run to run, and it means a row's tolerance must be read as
  Monte-Carlo slack on a single draw set, never as flakiness insurance.
"""
from __future__ import annotations

from dataclasses import dataclass

from flatppl_testsuite.sampler_sweep.oracle import FAMILIES, Family

# ---------------------------------------------------------------------------
# Draw counts. One value for the whole sweep keeps every tolerance derivation
# comparable; `checks.py` turns it into a per-row sigma band. 200_000 is set by
# the widest variance standard error on the roster: the mixture rows have
# mu4 = 138 and var = 10, so SE(var) = sqrt((138 - 100)/n) = 0.0138 at this n,
# and the branch-pinning defect displaced the variance by 9.0 — about 650 sigma.
# Cost at this n is ~10 s for the whole roster (measured), well inside budget.
# ---------------------------------------------------------------------------
N_DRAWS = 200_000

# Coordinate 0 is strided down to this many values for the KS test. A KS
# statistic at 200k would be dominated by its own critical value shrinking
# faster than any real defect; 20k keeps the 5-sigma-equivalent threshold
# (~0.0136, see checks.KS_CRIT) comfortably above float noise while still
# resolving a shape defect. Also the reason the driver does not ship 200k
# floats per row over stdout.
KS_SUBSAMPLE = 20_000

# The engine's fixed materialiser seed. Frozen here because the whole table is
# reproducible only against one seed (see the module docstring on the scalar
# rootKey collapse).
SEED = 0xBA5E


@dataclass(frozen=True)
class Probe:
    """One row of the sweep: a source to materialise plus its closed forms."""

    id: str
    source: str
    binding: str
    k: int
    field: str | None

    mean: float | None
    var: float | None
    fourth: float | None
    cov: float | None
    """Closed-form cov(coord 0, coord i) for i > 0. None when k == 1."""

    logtotalmass: float | None
    """Closed-form log of the measure's total mass. None means do not check it."""

    ks: tuple | None
    """KS reference, one of:
       ("dist", name, args, kwargs)             — scipy.stats.<name>(*args, **kwargs)
       ("affine", a, b, name, args, kwargs)     — that dist pushed through a*x + b
       ("mix", weights, [(name, args, kwargs)]) — a weighted mixture of them
       None                                     — no KS test on this row."""

    family: str
    wrap: str
    note: str = ""
    n_draws: int = N_DRAWS
    """Draws for this row. One value across the roster (see N_DRAWS) so every
    tolerance derivation is comparable; a field rather than a constant only so a
    future heavy-tailed row can buy precision without moving the whole sweep."""

    mean_by_coord: tuple[float, ...] | None = None
    """Per-coordinate mean oracle, overriding `mean`. `mean` alone assumes every
    coordinate has the SAME mean, which is true of every `iid` row and false of
    a pushforward carrying a vector shift -- and a shift that is equal in every
    coordinate cannot show a dropped or sign-flipped component."""

    latent: str | None = None
    """A second binding whose WEIGHTED marginal mean is checked (see
    `checks.check_latent_mean`). None on every row that needs no such check."""
    latent_mean: float | None = None
    """The latent's closed-form PRIOR mean, which §06 `normalize` makes the exact
    oracle for its marginal."""
    latent_var: float | None = None
    """The latent's closed-form prior variance, which bands the check."""
    latent_tilt: float | None = None
    """What the pooled-divisor defect gives instead: the prior tilted by Z(theta).
    Checked by nothing at run time -- it is the number the gate's teeth test
    asserts the band REJECTS, so a green row cannot mean 'landed somewhere
    plausible'."""
    latent_cov: float | None = None
    """Closed-form cov(latent, variate coordinate 0), checked WEIGHTED (see
    `checks.check_latent_cov`). The discriminating moment for a mixing weight
    that reaches the variate only through the mixture's component choice: both
    marginals stay correct when the two decouple, so neither mean can see it."""
    latent_cov_var: float | None = None
    """`n * Var(cov_hat)` for that estimator, closed form: `E[a^2 b^2] - cov^2`
    with `a`, `b` the two centred variables. It bands the check."""
    latent_cov_null: float | None = None
    """What a lift that decouples the latent from the variate gives instead --
    0, since the mixture then draws at the pooled proportion. Checked by nothing
    at run time; it is the number the gate's teeth test asserts the band
    rejects."""

    weighted_variate: bool = False
    """Take the variate's own moments under the measure's ATOM WEIGHTS, banded by
    the ensemble's effective sample size instead of `n`. Set it on a measure that
    represents its law by reweighting uniform positions -- `normalize(weighted(f,
    Q))` draws at Q's positions and carries f/Z in the weights -- where an
    unweighted moment measures Q and not the measure. A KS test cannot follow the
    weights, so such a row carries `ks=None`."""

    variate_skip_reason: str | None = None
    """Why this row's variate carries no mean/variance oracle, when the reason is
    not the default one (a law with no such moment). It is reported in the
    skipped check's detail, so a reader of the table sees WHY a row checks
    nothing there rather than assuming an oversight."""


# ---------------------------------------------------------------------------
# Base-agnostic wraps. Each is (slug, k, template, binding, moment transform).
# The template's `{M}` is the base family's surface text; `u` is a stochastic
# node holding a draw from it, so `lawof(u)` reifies the base as a measure.
#
# `xform` maps the base's (mean, var, fourth) to the wrapped measure's. It is
# exact algebra, not an approximation:
#   identity  — the wrap moves no mass:      (m, v, m4)
#   affine    — y = a x + b:                 (a m + b, a^2 v, a^4 m4)
# ---------------------------------------------------------------------------
_AFFINE_A, _AFFINE_B = 2.0, 1.0


@dataclass(frozen=True)
class AgnosticWrap:
    slug: str
    k: int
    template: str
    binding: str
    xform: str
    logtotalmass: float | None
    note: str = ""


# `math.log(2.0)`, spelled out so the table's frozen value is readable.
_LOG2 = 0.6931471805599453

AGNOSTIC_WRAPS: tuple[AgnosticWrap, ...] = (
    AgnosticWrap("iid3", 3, "b ~ iid({M}, 3)\n", "b", "identity", 0.0,
                 note="product measure: per-coordinate marginals are the base, cov 0"),
    AgnosticWrap("iid4", 4, "b ~ iid({M}, 4)\n", "b", "identity", 0.0),
    # weighted scales the MASS, not the positions: the draws are the base's, so
    # the moments are unchanged and only the total mass moves. Confirmed against
    # the engine's own note in iid-superpose-branch-freshness.test.ts: "the
    # SAMPLE positions are the mixture's, so the moments are unchanged (the
    # weight lives in the density)".
    AgnosticWrap("weighted2", 1, "u ~ {M}\nM = weighted(2.0, lawof(u))\n", "M",
                 "identity", _LOG2,
                 note="mass 2, positions unchanged: total mass is the whole check here"),
    # The base is already a probability measure, so normalize is the identity on
    # it — including on the total mass, which must come back at exactly 1.
    AgnosticWrap("normalize_prob", 1, "u ~ {M}\nM = normalize(lawof(u))\n", "M",
                 "identity", 0.0,
                 note="normalize of a probability measure is the identity, mass exactly 1"),
    AgnosticWrap("pushfwd_affine", 1,
                 f"u ~ {{M}}\nM = pushfwd(fn({_AFFINE_A} * _ + {_AFFINE_B}), lawof(u))\n",
                 "M", "affine", 0.0,
                 note=f"y = {_AFFINE_A}x + {_AFFINE_B}: mean and variance must both move"),
    AgnosticWrap("iid_pushfwd3", 3,
                 f"u ~ {{M}}\nM = pushfwd(fn({_AFFINE_A} * _ + {_AFFINE_B}), lawof(u))\n"
                 "b ~ iid(M, 3)\n", "b", "affine", 0.0,
                 note="iid over a transformed base: the composite-fallback iid path"),
)

# The bases the agnostic wraps cross. A spread over the shapes that stress
# different sampler routes: symmetric continuous, a bounded family (the
# @stdlib-defect region), a positive skewed family, a heavy tail, a discrete
# count and a two-point discrete.
WRAP_BASES: tuple[str, ...] = (
    "normal", "beta_2_2", "gamma", "studentt", "poisson", "bernoulli",
)

# ---------------------------------------------------------------------------
# Composed wraps over a Normal base. Every oracle below was verified by
# quadrature of the composed density (see the report's tolerance table); the
# mixture's fourth central moment 138.0 independently reproduces the figure the
# engine's own iid-superpose test quotes.
# ---------------------------------------------------------------------------

# Two components three sigma apart, so a pinned branch is unmistakable.
_MIX = ("u ~ Normal(mu = -3.0, sigma = 1.0)\n"
        "w ~ Normal(mu = 3.0, sigma = 1.0)\n")
_MIX_SYM = _MIX + "S = superpose(weighted(0.5, lawof(u)), weighted(0.5, lawof(w)))\n"
_MIX_ASYM = _MIX + "S = superpose(weighted(0.3, lawof(u)), weighted(0.7, lawof(w)))\n"

_KS_MIX_SYM = ("mix", (0.5, 0.5), (("norm", (-3.0, 1.0), {}), ("norm", (3.0, 1.0), {})))
_KS_MIX_ASYM = ("mix", (0.3, 0.7), (("norm", (-3.0, 1.0), {}), ("norm", (3.0, 1.0), {})))

# z ~ N(0,1); y | z ~ N(z, 1). §06 "Dependent composition": kchain "Keeps only
# the last kernel's variates, marginalizing out all intermediate variates", so
# the materialised measure is the MARGINAL of y, which is N(0, 1 + 1) = N(0, 2).
_KCHAIN = ("z ~ Normal(mu = 0.0, sigma = 1.0)\n"
           "k = kernelof(record(y = draw(Normal(mu = z, sigma = 1.0))), z = z)\n"
           "C = kchain(lawof(record(z = z)), k)\n")

# truncate(Normal(0,1), interval(-1,1)). Total mass is 2*Phi(1) - 1; the
# truncated moments are scipy.stats.truncnorm(-1, 1)'s, cross-checked against
# quadrature of the restricted density.
_TRUNC_LOGTM = -0.38171514630212616
_TRUNC_VAR = 0.291125094772793
_TRUNC_M4 = 0.16450037909117288

COMPOSED: tuple[Probe, ...] = (
    Probe("normal.superpose_sym", _MIX_SYM, "S", 1, None,
          0.0, 10.0, 138.0, None, 0.0, _KS_MIX_SYM, "normal", "superpose_sym",
          note="equal-weight mixture of N(-3,1) and N(3,1); total mass 0.5+0.5 = 1"),
    Probe("normal.superpose_asym", _MIX_ASYM, "S", 1, None,
          1.2, 8.56, 149.0592, None, 0.0, _KS_MIX_ASYM, "normal", "superpose_asym",
          note="unequal weights break the symmetry that hides a swapped branch"),
    Probe("normal.normalize_superpose", _MIX_SYM + "M = normalize(S)\n", "M", 1, None,
          0.0, 10.0, 138.0, None, 0.0, _KS_MIX_SYM, "normal", "normalize_superpose",
          note="mass is already 1, so normalize must be the identity here"),
    Probe("normal.weighted_superpose", _MIX_SYM + "M = weighted(2.0, S)\n", "M", 1, None,
          0.0, 10.0, 138.0, None, _LOG2, _KS_MIX_SYM, "normal", "weighted_superpose",
          note="mass 2 over a mixture: positions unchanged, mass doubled"),
    # THE branch-pinning rows. Per-coordinate moments AND cov(0, i) = 0.
    Probe("normal.iid_superpose3", _MIX_SYM + "b ~ iid(S, 3)\n", "b", 3, None,
          0.0, 10.0, 138.0, 0.0, 0.0, _KS_MIX_SYM, "normal", "iid_superpose3",
          note="IIDSUPER class: each coordinate must select its OWN component"),
    Probe("normal.iid_superpose4", _MIX_SYM + "b ~ iid(S, 4)\n", "b", 4, None,
          0.0, 10.0, 138.0, 0.0, 0.0, _KS_MIX_SYM, "normal", "iid_superpose4",
          note="IIDSUPER class at even k, where every coordinate was pinned"),
    Probe("normal.truncate", "u ~ Normal(mu = 0.0, sigma = 1.0)\n"
          "M = truncate(lawof(u), interval(-1.0, 1.0))\n", "M", 1, None,
          0.0, _TRUNC_VAR, _TRUNC_M4, None, _TRUNC_LOGTM,
          ("dist", "truncnorm", (-1.0, 1.0), {"loc": 0.0, "scale": 1.0}),
          "normal", "truncate",
          note="the total mass 2*Phi(1)-1 is a closed form the engine must reproduce"),
    Probe("normal.kchain", _KCHAIN, "C", 1, "y",
          0.0, 2.0, 12.0, None, 0.0, ("dist", "norm", (0.0, 2.0 ** 0.5), {}),
          "normal", "kchain",
          note="record-valued: the kept variate is the marginal y ~ N(0, 2)"),
    # WHAT THIS ROW PINS, AND WHAT IT DOES NOT.
    #
    # It pins the engine's ensemble-of-tables refusal: materialising a
    # record-valued measure at more than one atom raises "iid: sampling iid over
    # a record measure at >1 atoms (an ensemble of tables) is not supported".
    #
    # It does NOT test the defect carded in flatppl-dev/TODO-flatppl-js.md, that
    # `iid(kchain(M, K), n)` shares the chain's base draw across copies. That
    # guard trips on the SAMPLE COUNT, not on the `iid` count: the same source
    # materialises fine at n = 1, and the moment checks need n large. So this row
    # cannot observe the base-draw fix either way.
    #
    # That prediction was made against the `iid-kchain` branch head and has since
    # been confirmed by the merge. flatppl-js #164 landed as 255261f — the card in
    # flatppl-dev/TODO-flatppl-js.md is [x] FIXED 2026-08-20 — and this row still
    # REFUSES with the byte-identical message, producing no diff.
    #
    # The base-draw-sharing class is therefore UNCOVERED by this sweep, and not
    # reachable at its shape (moments need n > 1; the guard forbids n > 1 for a
    # record variate). Covering it needs a per-draw harness rather than a
    # moment sweep. Carded in the report as out of scope for v1.
    Probe("normal.iid_kchain3", _KCHAIN + "b ~ iid(C, 3)\n", "b", 3, "y",
          0.0, 2.0, 12.0, 0.0, 0.0, None, "normal", "iid_kchain3",
          note="pins the >1-atoms ensemble-of-tables refusal; the carded "
               "base-draw-sharing defect is NOT reachable at this shape"),
)


# ---------------------------------------------------------------------------
# Targeted rows. Each one is a construct the two axes above cannot express, and
# each was landed by an engine change no probe reached (the batch audit's
# finding M18, flatppl-dev/spec-audit-batch-2026-08-27.md). Every oracle here is
# closed form, computed with mpmath at 40 digits and cross-checked against a
# second derivation; none is frozen from the engine.
# ---------------------------------------------------------------------------

# A scalar primitive mapped over a SHAPED atom batch: the base is a vector
# variate and the map is scalar arithmetic, so `broadcastN` has to iterate the
# cell rather than the atom. §04 "Broadcasting", "Non-collection inputs": a
# scalar "is simply not iterated over but held constant while collection
# arguments are iterated over".
_VEC2 = "u ~ iid(Normal(mu = 0.0, sigma = 1.0), 2)\n"

# LogNormal(0, 1) moments from the raw moments E[X^k] = exp(k^2 / 2):
#   mean = e^(1/2)                                = 1.6487212707001282
#   var  = e^2 - e                                = 4.670774270471605
#   mu4  = e^8 - 4 e^(1/2) e^(9/2) + 6 e e^2 - 3 e^2
#                                                 = 2485.651403873756
# Each is the nearest double to the exact value (mpmath, 40 dps).
_LOGNORMAL_MEAN = 1.6487212707001282
_LOGNORMAL_VAR = 4.670774270471605
_LOGNORMAL_MU4 = 2485.651403873756

# Uniform(a, b) prior moments for the normalize rows: mean (a+b)/2, var
# (b-a)^2/12. Both priors below span 4, so both variances are 16/12.
_UNIFORM_SPAN4_VAR = 1.3333333333333333

# Beta(2, 5) prior moments for the mixing-weight row: mean a/(a+b) = 2/7,
# variance ab/((a+b)^2 (a+b+1)) = 5/196.
_BETA25_MEAN = 2.0 / 7.0
_BETA25_VAR = 5.0 / 196.0

TARGETED: tuple[Probe, ...] = (
    # The DOTTED spelling, and it has to be. §07 "Elementary functions": "All
    # accept scalar arguments and return scalar results", so a bare `exp` over a
    # vector variate is a static error -- flatppl-js #228 enforces exactly that,
    # and the bare spelling this row first used stopped drawing the moment it
    # landed. `exp.(_)` is §07's own elementwise form, and it is the spelling
    # #218's LogNormal pins use.
    Probe("normal.pushfwd_exp_vector", _VEC2 + "M = pushfwd(fn(exp.(_)), lawof(u))\n",
          "M", 2, None,
          _LOGNORMAL_MEAN, _LOGNORMAL_VAR, _LOGNORMAL_MU4, 0.0, 0.0,
          ("dist", "lognorm", (1.0,), {"loc": 0.0, "scale": 1.0}),
          "normal", "pushfwd_exp_vector",
          note="a NONLINEAR elementwise primitive over a vector variate: each "
               "coordinate is LogNormal(0, 1), and the coordinates stay independent"),
    # The same route with an affine map, which §06 "Engine contract for `pushfwd`
    # density evaluation" case 1 names by construction. y = 2x over a standard
    # normal is Normal(0, 2): var 4, mu4 = 3 sigma^4 = 48.
    Probe("normal.pushfwd_scalar_affine_vector", _VEC2 + "M = pushfwd(fn(2.0 * _), lawof(u))\n",
          "M", 2, None,
          0.0, 4.0, 48.0, 0.0, 0.0, ("dist", "norm", (0.0, 2.0), {}),
          "normal", "pushfwd_scalar_affine_vector",
          note="scalar-affine map over a vector variate: every coordinate is "
               "Normal(0, 2)"),
    # And with a vector SHIFT, whose components differ. A shift equal in every
    # coordinate cannot show a dropped or sign-flipped component, so this row is
    # the one that needs `mean_by_coord`: y_i = 2 x_i + b_i is Normal(b_i, 2).
    Probe("normal.pushfwd_scalar_affine_shift_vector",
          "b = [1.0, -1.0]\n" + _VEC2 + "M = pushfwd(x -> 2.0 * x + b, lawof(u))\n",
          "M", 2, None,
          None, 4.0, 48.0, 0.0, 0.0, ("dist", "norm", (1.0, 2.0), {}),
          "normal", "pushfwd_scalar_affine_shift_vector",
          note="vector shift with distinct components: coordinate i is "
               "Normal(b_i, 2), so a dropped or negated shift moves one mean",
          mean_by_coord=(1.0, -1.0)),
    # ------------------------------------------------- theta-dependent normalize
    # §06 "Normalization and mass": `normalize(M)` returns "the probability
    # measure M / Z ... On a non-nullary kernel, normalizes the output measures".
    # Every theta-slice is therefore a probability measure, so the theta-marginal
    # of the sampled joint is the PRIOR, exactly -- no quadrature enters the
    # oracle. The defect these rows exist for divides by the POOLED mass instead,
    # leaving atom i the residue Z(theta_i)/E[Z]; that hypothesis has its own
    # closed form, recorded as `latent_tilt`.
    #
    # Z(theta) = theta, theta ~ Uniform(1, 5). Tilted marginal
    #   E[theta^2] / E[theta] = (124/12) / 3 = 31/9 = 3.4444444444444446,
    # confirmed by quadrature. The measure itself is Normal(0, 1) at every theta
    # (the factor divides straight out), so the variate's own moments are the
    # standard normal's and the weights come back uniform.
    Probe("normal.normalize_theta_weighted",
          "theta ~ Uniform(interval(1.0, 5.0))\n"
          "m = normalize(weighted(theta, Normal(mu = 0.0, sigma = 1.0)))\n"
          "y ~ m\n",
          "y", 1, None,
          0.0, 1.0, 3.0, None, 0.0, ("dist", "norm", (0.0, 1.0), {}),
          "normal", "normalize_theta_weighted",
          note="a scalar theta-dependent mass factor over a probability leaf; "
               "the theta-marginal must be the prior, not the Z-tilted 31/9",
          latent="theta", latent_mean=3.0, latent_var=_UNIFORM_SPAN4_VAR,
          latent_tilt=3.4444444444444446),
    # The same factor in LOG space, which reaches the other arm of the sampler's
    # per-atom divisor. Z(theta) = e^theta over Uniform(1, 5), so the tilted
    # marginal is
    #   int theta e^theta / int e^theta = 4 e^5 / (e^5 - e) = 4.074629441455096,
    # closed form and quadrature agreeing to 20 digits. The correct value is the
    # same prior mean as above -- both divisors are EXACT, so both rows leave the
    # weights uniform and report the same number; what differs is the route.
    Probe("normal.normalize_theta_logweighted",
          "theta ~ Uniform(interval(1.0, 5.0))\n"
          "m = normalize(logweighted(theta, Normal(mu = 0.0, sigma = 1.0)))\n"
          "y ~ m\n",
          "y", 1, None,
          0.0, 1.0, 3.0, None, 0.0, ("dist", "norm", (0.0, 1.0), {}),
          "normal", "normalize_theta_logweighted",
          note="the same mass factor in log space; the tilted hypothesis is "
               "4 e^5 / (e^5 - e), a full nat above the prior",
          latent="theta", latent_mean=3.0, latent_var=_UNIFORM_SPAN4_VAR,
          latent_tilt=4.074629441455096),
    # The weighted-box witness. f = exp(theta x) on x in [0, 1], theta ~
    # Uniform(0, 4), so Z(theta) = (e^theta - 1) / theta and the tilted marginal
    # is
    #   int_0^4 (e^theta - 1) dtheta / int_0^4 (e^theta - 1)/theta dtheta
    #     = 2.8073315740022866
    # (mpmath quadrature at 40 dps, cross-checked against the theta-marginal
    # form). The prior mean is 2.0.
    #
    # THE VARIATE CHECKS ARE OFF HERE, and that is the point of
    # `variate_skip_reason`: unlike the two rows above, the weights do NOT come
    # back uniform (n_eff is about 144k of 200k), so an UNWEIGHTED moment of the
    # variate measures the proposal and not the measure. Pinning it would freeze
    # an implementation detail as if it were spec.
    Probe("normal.normalize_theta_weighted_box",
          "theta ~ Uniform(interval(0.0, 4.0))\n"
          "m = normalize(weighted(fn(exp(theta * _)), "
          "Lebesgue(support = interval(0.0, 1.0))))\n"
          "y ~ m\n",
          "y", 1, None,
          None, None, None, None, 0.0, None,
          "normal", "normalize_theta_weighted_box",
          note="the weighted-box witness: a latent inside the weight of a "
               "Lebesgue box, whose per-theta mass is a 128-point CRN estimate",
          latent="theta", latent_mean=2.0, latent_var=_UNIFORM_SPAN4_VAR,
          latent_tilt=2.8073315740022866,
          variate_skip_reason="the atoms are importance-weighted and NOT "
                              "equally weighted here, so an unweighted moment "
                              "of the variate measures the proposal"),
    # ------------------------------------- iid over an importance-weighted normalize
    # The dropped-weight witness (flatppl-js #232). §06 `normalize` returns "the
    # probability measure M / Z" and §06 `iid` "the product measure
    # $M^{\\otimes N}$", so this measure is EXACTLY Normal(1, 1)^{otimes 3} --
    # e^x times the standard normal density is e^(1/2) times the Normal(1, 1)
    # density, a conjugate tilt with Z = e^(1/2). Per coordinate: mean 1,
    # variance 1, fourth central moment 3 sigma^4 = 3, and cov 0 across
    # coordinates.
    #
    # `iid`'s composite fallback re-materialised the inner measure at N*k atoms
    # and then rebuilt the output WITHOUT the per-position weights, so every
    # coordinate came back at the unnormalized base's mean of 0 -- a full sigma
    # out, and silent. `rand` refuses the identical weight-drop loudly, which is
    # what made the gap visible.
    #
    # WEIGHTED MOMENTS, and this row is the reason the flag exists. The atoms sit
    # at Normal(0, 1)'s positions and the whole reweighting rides in the atom
    # weight, so an UNWEIGHTED moment here measures the proposal and reports 0 --
    # bit-identical before and after the fix, exactly as the weighted-box row's
    # `variate_skip_reason` describes. Weighted, the same three coordinates are
    # the oracle's, and the covariance check is what separates a correct
    # per-coordinate product from a mis-folded single-position weight (which
    # leaves coordinate 0 at 1 and the rest at 0).
    #
    # No KS row: the subsample the driver ships cannot carry the weights.
    #
    # THE ONE ROW THAT BUYS PRECISION. The atom weight is a product of k = 3
    # lognormal(0, 1) factors, so its log has variance 3 and the ensemble's
    # ESS/n is about e^(-3) = 5% -- by far the thinnest on the roster. At the
    # roster's 200k that leaves an effective count near 9.7k, an order of
    # magnitude under every other row's, and the bands are computed from the
    # effective count. 600k restores an effective count comparable to what the
    # unweighted rows actually have. Derived from the weight's own distribution,
    # not fitted to an observed sigma.
    Probe("normal.iid_normalize_weighted_variate",
          "m = normalize(weighted(fn(exp(_)), Normal(mu = 0.0, sigma = 1.0)))\n"
          "y ~ iid(m, 3)\n",
          "y", 3, None,
          1.0, 1.0, 3.0, 0.0, 0.0, None,
          "normal", "iid_normalize_weighted_variate",
          note="iid over a normalize whose weights ARE the law: every coordinate "
               "is Normal(1, 1) by conjugacy, and the weights must fold as a "
               "product over the k coordinates",
          n_draws=600_000, weighted_variate=True),
    # ------------------------------------------------- a LATENT mixing weight
    # §06 "Normalization and mass", the `normalize` entry's own recommended
    # mixture spelling: "To build a normalized mixture distribution, use
    # `normalize(superpose(weighted(w1, M1), weighted(w2, M2)))`". With a LATENT
    # in those weights the mixing PROPORTION is what moves per atom, and §06
    # `superpose` makes the mass "nu(A) = M1(A) + M2(A) + ..." -- here
    # p + (1 - p) = 1, so the mass is CONSTANT and the proportion is the only
    # thing at stake.
    #
    # WHY NO EXISTING CHECK REACHED IT. Atom i must mix at p_i, giving
    # E[y | p] = 10(1 - p). Both marginals are then correct whether or not the
    # per-atom proportion survives: E[p] is the Beta(2, 5) prior 2/7, and E[y] is
    # LINEAR in p, so it is 10*E[1 - p] = 50/7 at E[p] just as it is atom by
    # atom. flatppl-js pooled the proportion into E[p] and both means stayed
    # clean; `latent_mean` read 0.285793 against 2/7 and the variate mean 7.147
    # against 50/7. Only the JOINT moment separates the two, which is what
    # `latent_cov` adds.
    #
    # THE ORACLE, closed form and exact:
    #   cov(p, y) = cov(p, 10(1 - p)) = -10*Var(p) = -25/98 = -0.2551020408
    # for Beta(2, 5)'s variance 5/196. The failing hypothesis has its own closed
    # form -- 0, since a decoupled proportion leaves p independent of y -- and it
    # is `latent_cov_null`, which the gate asserts the band rejects. Measured
    # -0.003046 pre-fix, 0.98 sigma from zero at 60k.
    #
    # THE BAND is derived, not fitted. For iid pairs
    #   n*Var(cov_hat) = E[a^2 b^2] - cov^2  with a = p - E[p], b = y - E[y],
    # and conditioning on p gives E[b^2 | p] = 2549/49 - (300/7)(1 - p), so
    #   E[a^2 b^2] = (2549/49)*Var(p) - (300/7)*(Var(p)(1 - E[p]) - mu3(p))
    #              = 6245/9604,
    # hence n*Var(cov_hat) = 1405/2401 = 0.5851728446480633 exactly, with
    # mu3(Beta(2,5)) = 5/2058. Cross-checked against 400 replicate ensembles of
    # the closed-form generative model at 60k: sd 0.0031301 against the formula's
    # 0.0031230, agreeing to 0.2%.
    #
    # The variate's own moments are NOT pinned. y's mean is 50/7 either side, so
    # it discriminates nothing here, and its variance is the mixture's -- a
    # number the row would freeze without any defect to catch.
    Probe("normal.normalize_superpose_latent_mixing",
          "p ~ Beta(alpha = 2.0, beta = 5.0)\n"
          "q = 1.0 - p\n"
          "m = normalize(superpose(weighted(p, Normal(mu = 0.0, sigma = 1.0)), "
          "weighted(q, Normal(mu = 10.0, sigma = 1.0))))\n"
          "y ~ m\n",
          "y", 1, None,
          None, None, None, None, 0.0, None,
          "normal", "normalize_superpose_latent_mixing",
          note="§06 normalize's own mixture spelling with a LATENT mixing "
               "weight: atom i must mix at p_i, and only cov(p, y) can see it",
          latent="p", latent_mean=_BETA25_MEAN, latent_var=_BETA25_VAR,
          latent_cov=-10.0 * _BETA25_VAR,
          latent_cov_var=1405.0 / 2401.0, latent_cov_null=0.0,
          variate_skip_reason="the variate's mean is 50/7 whether or not the "
                              "per-atom proportion survives (it is linear in p) "
                              "and its variance pins no defect; cov(p, y) is the "
                              "discriminating moment and it is checked"),
)


def _apply(xform: str, m: float | None, v: float | None, m4: float | None):
    if xform == "identity":
        return m, v, m4
    if xform == "affine":
        a, b = _AFFINE_A, _AFFINE_B
        return (None if m is None else a * m + b,
                None if v is None else a * a * v,
                None if m4 is None else a ** 4 * m4)
    raise ValueError(f"unknown moment transform {xform!r}")


def _ks_for(fam: Family, xform: str):
    if fam.scipy is None:
        return None
    name, args, kwargs = fam.scipy
    if fam.discrete:
        # A KS test needs a continuous cdf; a discrete family gets moments only.
        return None
    if xform == "identity":
        return ("dist", name, args, kwargs)
    if xform == "affine":
        return ("affine", _AFFINE_A, _AFFINE_B, name, args, kwargs)
    raise ValueError(f"unknown moment transform {xform!r}")


def enumerate_probes() -> list[Probe]:
    """The whole space, in a stable order."""
    out: list[Probe] = []

    # Axis 1 — every sampleable family, identity wrap.
    for fam in FAMILIES:
        out.append(Probe(
            id=f"{fam.slug}.identity", source=f"x ~ {fam.measure}\n", binding="x",
            k=1, field=None, mean=fam.mean, var=fam.var, fourth=fam.fourth,
            cov=None, logtotalmass=0.0, ks=_ks_for(fam, "identity"),
            family=fam.slug, wrap="identity", note=fam.note,
        ))

    # Axis 2 — base-agnostic wraps over a spread of bases.
    by_slug = {f.slug: f for f in FAMILIES}
    for base in WRAP_BASES:
        fam = by_slug[base]
        for wrap in AGNOSTIC_WRAPS:
            m, v, m4 = _apply(wrap.xform, fam.mean, fam.var, fam.fourth)
            out.append(Probe(
                id=f"{fam.slug}.{wrap.slug}",
                source=wrap.template.format(M=fam.measure),
                binding=wrap.binding, k=wrap.k, field=None,
                mean=m, var=v, fourth=m4,
                cov=0.0 if wrap.k > 1 else None,
                logtotalmass=wrap.logtotalmass, ks=_ks_for(fam, wrap.xform),
                family=fam.slug, wrap=wrap.slug, note=wrap.note,
            ))

    # Axis 3 — composed wraps, Normal base, oracles written out per row.
    out.extend(COMPOSED)
    # Axis 4 — targeted rows for constructs no axis above reaches.
    out.extend(TARGETED)
    return out


def probe_count() -> int:
    return len(enumerate_probes())
