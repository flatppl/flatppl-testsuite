"""The sampler sweep's probe space: base families x combinator wraps.

SHAPE OF THE SPACE. Two axes, deliberately not one full cross product:

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
    # Carded OPEN in flatppl-dev/TODO-flatppl-js.md: `iid(kchain(M, K), n)`
    # shares the chain's base draw across copies. Kept in the roster so its
    # status is asserted rather than remembered; whichever way it lands is a
    # recorded outcome, not a surprise.
    Probe("normal.iid_kchain3", _KCHAIN + "b ~ iid(C, 3)\n", "b", 3, "y",
          0.0, 2.0, 12.0, 0.0, 0.0, None, "normal", "iid_kchain3",
          note="carded OPEN in TODO-flatppl-js.md: iid over a kchain shares the base draw"),
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
    return out


def probe_count() -> int:
    return len(enumerate_probes())
