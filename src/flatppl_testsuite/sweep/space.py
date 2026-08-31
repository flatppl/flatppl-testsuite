"""The probe space: pure data.

Axes are derived from the spec, not invented. Bases span the SUPPORTS that
drive §06 case 1's domain guards (a `log` map is only defined over a positive
base, so the guard is support-dependent). Wraps are §13 "Output reduction"'s own
enumerated list, one normative density rule each. Spellings are restricted to
pairs §06/§04 declare EQUIVALENT, which gives a correctness signal needing no
oracle: two spellings of one measure that lower to different densities are wrong
even if neither has been scored. Orderings exist because the pinned-by-a-later-
query case is where two silent wrong densities were found by hand.

## The vector family

`VECTOR_BASES` is a SECOND, targeted family, not a fifth base on the axes above.
It exists for two determiniser gate arms a scalar `Probe.point` cannot reach:

1. the discrete lattice snap, `iszero(sum(abs(y - f(round(f_inv(y))))))`, whose
   `sum(abs(.))` reduction only appears over a vector variate
   (`determinizer/src/density.rs`, `lattice_test`);
2. the `cartpow` membership gate, `in cartpow(S, n)`, which `forward_image`
   emits in place of the scalar `in S` when the variate is a vector
   (`determinizer/src/invert.rs`, `Image::vector_condition`).

It is deliberately NOT crossed with `WRAPS`/`SPELLINGS`/`ORDERINGS` in full: a
full cross-product would be hundreds of probes covering no further arm. **A new
vector gate arm in the determiniser obliges a new member of this family** — the
coverage invariant is one oracle-checked row per emitted vector arm, and
`tests/sweep/test_vector_arms.py` asserts each targeted arm actually FIRES in
the probe that claims it. A row whose gate never emitted proves nothing.

## The shared-latent family

`SHARED_LATENT_SHAPES` × `SHARED_LATENT_SPELLINGS` is a THIRD targeted family,
for the joint constructs. It exists because the two families above cannot reach
them at all: the scalar `record` spelling wraps exactly ONE bare draw, so no
probe there has two fields, let alone two fields sharing a stochastic ancestor.
That blind spot was measured, not assumed — the repin to flatppl-rust `ebed88b`
(#131, which taught the record path to lower the shared-latent record law) left
all 780 rows byte-identical, so a green sweep said nothing whatever about it
(`flatppl-dev/density-sweep-notes.md`, "A determiniser feature the probe space
cannot reach at all").

Every member is linear-Gaussian, so its true law is a multivariate normal whose
mean and covariance follow from the draw graph (`NormalNode`,
`shared_latent_graph`) — `oracle._linear_gaussian_moments` derives them from the
loading matrix, and `scipy.stats.multivariate_normal` scores the point. That is
the "multivariate composition" this family needs and the scalar §13 fold cannot
express.

The axes, and what each one buys:

* **shape** — the ancestry graph. `fan` (z → f_i for every i) is #131's own arm.
  `chain` (z → f_1 → f_2 → …) is the composed-affine case flatppl-js #134 taught
  the engine's recogniser; the determiniser is a separate question, which is the
  point of probing it. `disjoint` (a private z_i per field) is the DISJOINTNESS
  CONTROL: nothing is shared, so the traced law and the product of the marginals
  must coincide, and a lowering that manufactures correlation is caught by the
  same rows that would otherwise prove nothing. `singular` references ONE draw
  from every field, which §06 "Singular joints" says has no density.
* **spelling** — §06 "Equivalent record law" declares `record_law`, `joint_kw`
  and `joint_pos` three spellings of ONE measure, which is the same
  oracle-independent correctness signal the scalar `SPELLINGS` axis gives: three
  spellings that lower to different densities are wrong even before a value is
  compared. `joint_ctor` is §06's own contrast in the sentence right after —
  "a `joint` of two constructor measures with the same marginals has
  cross-covariance 0" — so it is the product-of-marginals arm, and it must
  DIFFER from `record_law` wherever a node is shared and AGREE with it on
  `disjoint`. `ctor_shared_kw` / `ctor_shared_pos` are the OTHER constructor
  case: components sharing the latent through a stochastic PARAMETER rather
  than through a reified law, which §06 keeps as one node of the composed
  trace — so they must agree with `record_law`, not with `joint_ctor`. That
  pair is what flatppl-rust #156 taught the determiniser to lower, and before
  it they refused. `iid` is independent by construction (§06 `iid`), so it must
  not pick up the shared ancestor either.
* **n** — 2 and 3 fields. `n = 3` is what makes the off-diagonal STRUCTURE
  checkable rather than a single number: `fan`'s off-diagonals are all `Var(z)`
  while `chain`'s are nested (`cov[i][j] = Var(f_min(i,j))`), and at n = 2 those
  two are indistinguishable in shape.
* **latent_query** — a SECOND `logdensityof`, on the shared latent itself,
  before or after the family's own query. `logdensityof` is a query and not a
  conditioning, so the joint's density must not move; the scalar family's
  `ORDERINGS` axis exists because two silent wrong densities were found by hand
  in exactly this two-query region, and here the second query lands on the node
  the whole family shares.

**Not crossed in full, for the vector family's reason.** `latent_query` is
crossed only at `n = 2`, and `n = 3` runs at `latent_query = "none"`: the
ordering hazard is a property of the two-query lowering and does not interact
with how many fields the record has. `iid` runs on `fan` alone, because its
independence is a property of the construct rather than of the ancestry graph.
`singular` runs at `latent_query = "none"`, because a shape that has no density
has no ordering behaviour to probe.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Base:
    kind: str          # "normal" | "gamma" | "beta" | "poisson" | "dirichlet" | "multinomial"
    # Positional, in §08's own order, e.g. (0.0, 1.0). A VECTOR-valued parameter
    # is a nested `tuple`, not a `list`: `Base` is frozen and is used as a dict
    # key (`test_oracle._unique_shapes`), so every component has to be hashable.
    params: tuple


@dataclass(frozen=True)
class Wrap:
    kind: str          # "pushfwd" | "truncate" | "weighted" | ...
    args: tuple        # wrap-specific, e.g. ("exp",) or (0.5,)


@dataclass(frozen=True)
class Probe:
    id: str
    base: Base
    wraps: tuple[Wrap, ...]
    spelling: str      # "direct" | "stochastic_node" | "record"
    ordering: str      # "single" | "pinned_earlier" | "pinned_later"
    consumer: bool
    # A scalar query point, or one cell per component for a `VECTOR_BASES` probe.
    # A TUPLE for the vector case, not a list: `Probe` is frozen, so a mutable
    # field would make an otherwise-immutable record only conditionally hashable.
    # `render._value_src` renders either sequence type.
    point: float | tuple[float, ...]


# Support-distinct bases. `point` per base is chosen INSIDE the base's support
# so an unwrapped probe is always scoreable; wraps that move the support carry
# their own point override in `_point_for`.
BASES = [
    Base("normal", (0.0, 1.0)),      # reals
    Base("gamma", (2.0, 1.0)),       # posreals  (shape, rate — §08 order)
    Base("beta", (2.0, 3.0)),        # (0, 1)
    Base("poisson", (3.0,)),         # nonneg integers
]

# The point inside EACH base's own support that every wrapped point is
# derived from (see `_point_for`) — never hardcode a wrapped point directly,
# or the next base added shares another base's constant by accident (that
# was exactly the bug in fix round 1: `pushfwd_log`/`pushfwd_sqrt` reused
# `gamma`'s inner point for `beta`, landing outside `beta`'s support).
INNER = {"normal": 0.5, "gamma": 1.5, "beta": 0.4, "poisson": 2.0}

# Forward maps for the `pushfwd` wraps, keyed the same as their `args[0]`.
_FORWARD = {
    "exp": math.exp,
    "log": math.log,
    "neg": lambda x: -x,
    "sqrt": math.sqrt,
}

# §13 "Output reduction"'s list. Each entry is one normative rule.
WRAPS = [
    Wrap("identity", ()),
    Wrap("pushfwd", ("exp",)),
    Wrap("pushfwd", ("log",)),
    Wrap("pushfwd", ("neg",)),
    Wrap("pushfwd", ("sqrt",)),
    # "affine" is this sweep's OWN label, not a FlatPPL construct -- there is
    # no such combinator in the spec or either engine's catalogue. It names
    # the LAMBDA-PUSHFWD spelling of an affine map (render.py emits
    # `pushfwd(x -> 2.0 * x + 1.0, ·)`), covering the same §06 rule as
    # `locscale` below through a different spelling -- §06 gives `locscale`
    # verbatim as `pushfwd(x -> scale * x + shift, m)`, so the two exist here
    # to check the spellings agree, not because they are two rules.
    Wrap("affine", (2.0, 1.0)),          # 2*x + 1
    Wrap("truncate", (0.0, "inf")),
    Wrap("weighted", (0.5,)),
    Wrap("logweighted", (-0.6931471805599453,)),
    Wrap("normalize", ()),
    Wrap("locscale", (1.0, 2.0)),
]

SPELLINGS = ["direct", "stochastic_node", "record"]
ORDERINGS = ["single", "pinned_earlier", "pinned_later"]


# --------------------------------------------------------------------------
# The vector family (see the module docstring)
# --------------------------------------------------------------------------

# §08 "Multivariate distributions" gives both parameterisations verbatim:
# `Multinomial(n, p)` with `n = elementof(posintegers)` and
# `p = elementof(stdsimplex(k))`, and `Dirichlet(alpha)` with `alpha` an array of
# positive reals. The parameter ORDER here is §08's own.
#
# Kept out of `BASES` on purpose. `BASES` is an AXIS, crossed with every wrap,
# spelling and ordering; these two are a targeted pair, one per reference measure:
# `Multinomial`'s density is w.r.t. `iid(Counting(integers), k)` (§08), which is
# what puts a pushforward of it on the determiniser's lattice-snap arm, and
# `Dirichlet`'s is w.r.t. `Lebesgue(stdsimplex(n))`, which is not.
VECTOR_BASES = [
    Base("multinomial", (5, (0.2, 0.3, 0.5))),   # §08 order: (n, p)
    Base("dirichlet", ((2.0, 3.0, 4.0),)),        # §08: (alpha)
]

# The point inside each vector base's own SUPPORT, played by `INNER`'s role for the
# scalar family. Both are on the support's constraint surface, not merely in the
# bounding box: §08 gives Multinomial's support as {x in N_0^k : sum x_i = n}, so
# (1, 2, 2) is chosen to sum to n = 5, and Dirichlet's as `stdsimplex(n)`, so
# (0.2, 0.3, 0.5) sums to 1. A point off the surface has density 0, which would
# make every probe in the family a -inf row proving nothing.
#
# Tuples, not lists, for the reason `Base.params` uses tuples: these are module
# constants and a probe's `point` derives from them, and `Probe` is frozen.
VECTOR_INNER = {
    "multinomial": (1.0, 2.0, 2.0),
    "dirichlet": (0.2, 0.3, 0.5),
}

# Whether each vector base's SUPPORT is a lower-dimensional manifold of the variate
# space, which decides whether a pushforward's volume element is well defined
# (`oracle._MANIFOLD_SAFE_FORWARDS`).
#
# **Every `VECTOR_BASES` member must appear here, and
# `test_oracle.test_every_vector_base_declares_its_support_geometry` asserts the two
# sets are equal.** The declaration is mandatory rather than an allowlist with a
# permissive default: a new manifold-support base (`LKJ`, a von Mises-Fisher) that
# someone forgot to declare would otherwise silently take the ambient-Jacobian
# reading and produce a wrong number, which is the one failure mode this module is
# built to prevent. Absent = loud `OracleUnsupported`, never "probably flat".
VECTOR_SUPPORT_IS_MANIFOLD = {
    # §08: support `stdsimplex(n)`, an (n-1)-manifold inside `cartpow(reals, n)`.
    "dirichlet": True,
    # §08: support {x in N_0^k : sum x_i = n}. A counting reference (§08 gives the
    # density w.r.t. `iid(Counting(integers), k)`), which a bijection does not
    # distort whatever shape the support has, so the manifold question never arises.
    "multinomial": False,
}

# The curated wrap list. Four entries, each for a stated reason:
#
# * `identity`   — the bare vector base density, the row every other member is
#                  read against.
# * `pushfwd(neg)` — the only elementwise pushforward in the family whose volume
#                  element is unambiguous over BOTH supports (see
#                  `oracle._MANIFOLD_SAFE_FORWARDS`), so it is the one that
#                  oracle-checks the vector change-of-variables end to end.
# * `pushfwd(exp)` — the member that reaches the `cartpow` image gate, and over
#                  `Multinomial` the lattice snap as well. Generated and
#                  oracle-checked over `Multinomial`; held out over `Dirichlet`,
#                  where the oracle has no value to check against (`_HELD_OUT`).
# * `truncate`   — a SCALAR `interval` against a vector variate: the set-kind
#                  mismatch the determiniser refuses. A conformant refusal, not a
#                  capability gap (`table._spec_justified`).
VECTOR_WRAPS = [
    Wrap("identity", ()),
    Wrap("pushfwd", ("neg",)),
    Wrap("pushfwd", ("exp",)),
    Wrap("truncate", (0.0, 1.0)),
]

# `record` is excluded: wrapping a vector in a one-field record changes the
# variate KIND, which is the scalar family's `record` axis, not a vector arm.
VECTOR_SPELLINGS = ["direct", "stochastic_node"]

# (base kind, wrap) shapes deliberately kept OUT of the generated family, each with
# the CATEGORY of reason and the reason itself. `_supported` is where ill-formed
# combinations go; this is for shapes that are well formed and simply cannot yield a
# checkable row.
#
# Two categories have existed, and the distinction is the whole point — they retire
# on different events:
#
# * `"engine"` — the emitted FlatPDL is correct but `flatppl-js` cannot evaluate it,
#   so the probe classifies `MALFORMED`, which is banned from the verdict table
#   (`test_gate.py::test_the_table_records_no_malformed_and_no_wrong_numbers`) and is
#   indistinguishable there from a determiniser defect. **This category is now
#   EMPTY.** It held `multinomial + pushfwd(neg)` and `multinomial + pushfwd(exp)`
#   until flatppl-js `e9803b6` taught `real` to accept integer arrays and gave `in` a
#   `cartpow` branch; both are now generated and oracle-checked, which is exactly
#   what the failing-when-fixed pins were for.
#
# * `"oracle"` — the engine evaluates the shape and returns a number, but no §06 rule
#   gives the reference measure of the result, so THIS SWEEP HAS NOTHING TO CHECK IT
#   AGAINST. Generating it would add a `LOWERS` row with `oracle = None`, which no
#   gate compares, i.e. a row that looks covered and is not.
#
# Each reason records what this harness OBSERVES, because that is what a human
# triaging it will grep for.
_HELD_OUT = {
    ("dirichlet", Wrap("pushfwd", ("exp",))): (
        "oracle",
        "the engine now scores this shape -- 1.022871190191443, which is the bare "
        "law's 2.0228711901914425 minus an ambient-R^3 Jacobian of exactly 1.0 "
        "(`sum(log y)`, which on the simplex is `sum(x)` = 1). But §06 scopes "
        "`Lebesgue` to lower-dimensional embedded AFFINE sets and `exp`'s image of "
        "`stdsimplex(n)` is a CURVED 2-manifold, so no §06 rule says which measure "
        "that number is a density against: the ambient Jacobian, the Hausdorff area "
        "element and the coordinate chart give 1.0, 0.6816 and 0.5 for the same "
        "volume term. The oracle withholds (`oracle._MANIFOLD_SAFE_FORWARDS`), so a "
        "generated row here would carry no checkable value. Tracked as an open spec "
        "question in flatppl-dev/measure-algebra-audit.md; this hold-out retires when "
        "that question is ruled on, NOT when an engine changes -- "
        "`tests/sweep/test_vector_arms.py` fails if either the emitted value or the "
        "oracle's withhold moves",
    ),
}

# The engine-gap category, kept as a named empty set rather than deleted: it records
# that the category exists and what retires it, so the next engine gap has somewhere
# obvious to go instead of being bolted onto the oracle one.
_ENGINE_BLOCKED = {k: v[1] for k, v in _HELD_OUT.items() if v[0] == "engine"}


# --------------------------------------------------------------------------
# The shared-latent family (see the module docstring)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class NormalNode:
    """One `Normal` draw in a linear-Gaussian graph.

    `parent` names the node this one's `mu` references, or is `None` for an
    ancestor-free prior. `mu` is the LITERAL location: the prior's own mean when
    `parent is None`, and an offset added to the parent otherwise. Every shape this
    family generates uses offset 0, so `mu` is only nonzero at the root — the field
    is kept general because the moment recursion is no harder for it and a
    hardcoded 0 would read as an invariant of the maths rather than of this family.

    `sigma` is a literal, never a reference: a stochastic SCALE is a different
    construct (it is not linear-Gaussian, so no multivariate normal is its law) and
    the oracle would have nothing to say about it.
    """
    name: str
    parent: str | None
    mu: float
    sigma: float


@dataclass(frozen=True)
class SharedLatentProbe:
    """A joint/record probe over N >= 2 fields of one linear-Gaussian graph.

    A separate type from `Probe` rather than more fields on it. `Probe` is a base
    plus a wrap stack, and this family has neither: its measure is composed from
    several DRAWS, and what varies is the ancestry graph between them. Threading
    `base: Base | None` and `wraps: () ` through `Probe` would make every existing
    reader of those two fields conditional, and the verdict table is frozen
    history keyed on `probe_id` — the scalar and vector rows have to keep coming
    out bit-identical.
    """
    id: str
    shape: str            # one of SHARED_LATENT_SHAPES
    n: int                # number of fields, >= 2
    spelling: str         # one of SHARED_LATENT_SPELLINGS
    latent_query: str     # "none" | "before" | "after"
    # One cell per field. A TUPLE for `Probe.point`'s reason: the record is frozen
    # and hashable only if every field is.
    point: tuple[float, ...]


# The latent prior, and the per-field conditional scales. Distinct sigmas on
# purpose: with one shared sigma the covariance is invariant under permuting the
# fields, so a lowering that transposed or mis-paired two fields would produce the
# right number anyway. `mu` is nonzero for the same reason applied to the location
# — every mean in every shape here is `SHARED_LATENT_MU`, so a dropped mean shifts
# every row rather than cancelling.
SHARED_LATENT_MU = 0.4
SHARED_LATENT_SIGMA_Z = 1.0
SHARED_LATENT_FIELD_SIGMAS = (0.5, 1.5, 2.0)

# Query points, one per field, and the point for the second (latent) query.
# Asymmetric and not all positive, so a sign or an index slip does not cancel.
SHARED_LATENT_POINTS = (0.5, 0.7, -0.3)
SHARED_LATENT_LATENT_POINT = 0.1

# Field NAMES are the record's, and they are what the query point is keyed by. They
# are distinct even where the underlying NODES are not (`singular` binds every
# field to one draw), which is what makes the singular shape expressible at all.
SHARED_LATENT_FIELD_NAMES = ("f1", "f2", "f3")

SHARED_LATENT_SHAPES = ["fan", "chain", "disjoint", "singular"]
SHARED_LATENT_SPELLINGS = ["record_law", "joint_kw", "joint_pos", "joint_ctor",
                           "ctor_shared_kw", "ctor_shared_pos", "iid"]
SHARED_LATENT_QUERIES = ["none", "before", "after"]
SHARED_LATENT_NS = [2, 3]

# How each spelling composes its components, which is what decides the oracle's
# covariance and is NOT derivable from the spelling's syntax alone:
#
# * "traced"  — §06 "Equivalent record law": the shared stochastic node stays a
#   single node of the composed trace, so the covariance is the graph's own. All
#   three of `record_law`, `joint_kw` and `joint_pos` are this, which is exactly
#   what makes them an equivalence check.
#   `joint_ctor` is ALSO "traced" and needs no product-measure special case: its
#   graph (see `shared_latent_graph`) is n ancestor-free priors, so the traced
#   covariance of that graph is already diagonal. One code path, no branch that
#   could disagree with the branch beside it.
# * "iid" — §06 `iid`: the product measure M^{(x)N}, so N independent copies of ONE
#   marginal whatever ancestry that marginal has.
SHARED_LATENT_COMPOSITION = {
    "record_law": "traced",
    "joint_kw": "traced",
    "joint_pos": "traced",
    "joint_ctor": "traced",
    # The CONSTRUCTOR components that share a node through their PARAMETERS -- §06's
    # "a stochastic node shared between component traces (through a reified component
    # ... or a stochastic constructor parameter) remains a single node of the composed
    # trace". So the composed law is the compound one, which is the record law of two
    # fresh draws, which is this family's `fan` graph exactly. Hence "traced" over the
    # SAME graph `record_law` uses -- not a fourth composition rule.
    "ctor_shared_kw": "traced",
    "ctor_shared_pos": "traced",
    "iid": "iid",
}


def _latent_name(shape: str) -> str:
    """The node the `latent_query` axis queries.

    `disjoint` has one private latent per field and no shared one; its first,
    `z1`, is queried, which is the honest analogue — the axis asks whether a
    second query on an ANCESTOR of a field perturbs the answer, and `z1` is one.
    """
    return "z1" if shape == "disjoint" else "z"


def shared_latent_variance(nodes: dict[str, NormalNode], name: str) -> float:
    """`Var(name)` from the graph, by the scalar recursion: a root's variance is
    its own `sigma**2`, and a child adds `sigma**2` to its parent's.

    Deliberately a SECOND derivation of what `oracle._linear_gaussian_moments`
    produces on the diagonal — that one goes through the loading matrix, this one
    through the recursion, and
    `test_shared_latent.test_the_two_variance_derivations_agree` asserts they
    match. `render` needs the marginal variance to spell `joint_ctor`'s matched
    constructors, so a copy exists either way; making it an independently checked
    copy is cheaper than making the oracle import from the renderer's needs.
    """
    node = nodes[name]
    own = node.sigma ** 2
    return own if node.parent is None else own + shared_latent_variance(nodes, node.parent)


def shared_latent_graph(shape: str, n: int, spelling: str,
                        ) -> tuple[dict[str, NormalNode], tuple[str, ...]]:
    """`(nodes, field_nodes)` for one probe: the draw graph, and the NODE each
    record field resolves to, in field order.

    `field_nodes` may repeat a name — that is the `singular` shape, and it is why
    the fields are addressed by `SHARED_LATENT_FIELD_NAMES` rather than by node.

    `joint_ctor` gets its OWN graph: n ancestor-free priors whose variances are the
    other spellings' MARGINAL variances. That is §06's contrast verbatim — "a
    `joint` of two constructor measures with the same marginals has
    cross-covariance 0" — so the arm is only the contrast it claims to be if the
    marginals really do match, which is what deriving them from the traced graph
    guarantees.

    `ctor_shared_kw` / `ctor_shared_pos` deliberately fall through to the shape
    branches and get the SAME graph as `record_law`. Their components are
    constructors whose `mu` names the latent, and §06 keeps a node shared through a
    stochastic constructor parameter as one node of the composed trace — so the
    composed law IS the record law of two fresh draws, and giving them their own
    graph would be a second way to spell one thing.
    """
    sig = SHARED_LATENT_FIELD_SIGMAS
    fields = SHARED_LATENT_FIELD_NAMES[:n]

    if spelling == "joint_ctor":
        traced, traced_fields = shared_latent_graph(shape, n, "record_law")
        return ({f: NormalNode(f, None, SHARED_LATENT_MU,
                               math.sqrt(shared_latent_variance(traced, node)))
                 for f, node in zip(fields, traced_fields)}, fields)

    if shape == "fan":
        nodes = {"z": NormalNode("z", None, SHARED_LATENT_MU, SHARED_LATENT_SIGMA_Z)}
        nodes.update({f: NormalNode(f, "z", 0.0, sig[i]) for i, f in enumerate(fields)})
        return nodes, fields

    if shape == "chain":
        nodes = {"z": NormalNode("z", None, SHARED_LATENT_MU, SHARED_LATENT_SIGMA_Z)}
        parent = "z"
        for i, f in enumerate(fields):
            nodes[f] = NormalNode(f, parent, 0.0, sig[i])
            parent = f
        return nodes, fields

    if shape == "disjoint":
        nodes: dict[str, NormalNode] = {}
        for i, f in enumerate(fields):
            z = f"z{i + 1}"
            nodes[z] = NormalNode(z, None, SHARED_LATENT_MU, SHARED_LATENT_SIGMA_Z)
            nodes[f] = NormalNode(f, z, 0.0, sig[i])
        return nodes, fields

    if shape == "singular":
        # ONE draw, bound to every field. §06 "Singular joints": "the same draw
        # referenced twice" — the law is carried by the diagonal {f1 = f2 = ...},
        # so it has no density w.r.t. the product reference measure.
        nodes = {
            "z": NormalNode("z", None, SHARED_LATENT_MU, SHARED_LATENT_SIGMA_Z),
            "f1": NormalNode("f1", "z", 0.0, sig[0]),
        }
        return nodes, ("f1",) * n

    raise ValueError(f"unknown shared-latent shape: {shape}")


# (shape, spelling) pairs held OUT of the generated family, each with the CATEGORY of
# reason and the reason itself — the same machinery, and the same two categories, as
# the vector family's `_HELD_OUT`. Separate from `shared_latent_supported`'s
# well-formedness rules below: those are shapes that DO NOT EXIST, this is for shapes
# that exist and cannot currently yield a usable row.
#
# `"engine"` — the probe is well formed as a probe, but the toolchain's behaviour on it
# makes the row unusable (a `MALFORMED` verdict, which `test_gate.py` bans from the
# table and which is indistinguishable there from a determiniser defect). Retires on an
# upstream change.
_SHARED_LATENT_HELD_OUT = {
    # Empty since the chain + ctor_shared_* retirement. Those two pairs sat here while
    # the toolchain MISLOWERED their source (an unbound `f1` was absorbed as `%fixed`
    # and determinize emitted a free variable — observed at flatppl-rust 9eefb43,
    # pinned on all three legs). flatppl-rust 499f39c makes an unresolvable name a
    # static error, so the pin reddened as designed and the entries were retired per
    # their own disposition: the refusal tests SCOPING, not the joint algebra, so the
    # pairs left the supported tuple (below) rather than becoming density probes.
    # `tests/sweep/test_shared_latent.py::test_the_chain_constructor_source_is_refused`
    # keeps the scoping observation pinned.
}

# The engine-gap category, derived rather than hand-maintained, for `_ENGINE_BLOCKED`'s
# reason: it records that the category exists and what retires it.
_SHARED_LATENT_ENGINE_BLOCKED = {
    k: v[1] for k, v in _SHARED_LATENT_HELD_OUT.items() if v[0] == "engine"
}


def shared_latent_supported(shape: str, spelling: str) -> bool:
    """Which (shape, spelling) pairs the family generates — see the module
    docstring's "not crossed in full" paragraph.

    `iid` on `fan` only: `iid`'s independence is a property of the construct (§06
    gives it as the product measure M^{(x)N}), so repeating it under `chain` and
    `disjoint` would add rows whose oracle value is the same product of the same
    marginal, differing only in a sibling definition the query never reaches.

    `joint_ctor` and `iid` are excluded from `singular`: neither is singular.
    `joint_ctor` composes fresh constructors, which share no node with anything,
    and `iid` is a product by construction — writing "the singular spelling" of
    either would be a probe of a shape that does not exist.
    """
    if spelling == "iid":
        return shape == "fan"
    if spelling in ("ctor_shared_kw", "ctor_shared_pos"):
        # `fan` is the shape this spelling IS -- every component's `mu` names the one
        # latent. `disjoint` is its control: distinct latents per component, so §06
        # gives back the independent product and a lowering that correlates any two
        # stochastic-parameter constructors is caught.
        #
        # `chain` has no constructor spelling, because a component's parameter can only
        # name a BINDING and a sibling component of the same `joint` is not one. The
        # source a renderer would emit for it references an unbound `f1`, which the
        # toolchain REFUSES since flatppl-rust 499f39c (it mislowered before — the
        # retired `_SHARED_LATENT_HELD_OUT` entries above record that history). The
        # refusal is a SCOPING verdict, not a joint-algebra one, so the pair stays out
        # of the family rather than becoming a density probe;
        # `test_the_chain_constructor_source_is_refused` pins the refusal. (An earlier
        # revision of this comment claimed the model itself was inexpressible, which is
        # false and is exactly the kind of unverified premise a hold-out reason is
        # supposed to stop.)
        #
        # `singular` does not apply: two fresh draws with identical parameters are
        # conditionally independent given the latent, never the same draw, so §06's
        # singular case is not reachable through constructors at all.
        if (shape, spelling) in _SHARED_LATENT_HELD_OUT:
            return False
        return shape in ("fan", "disjoint")
    if shape == "singular":
        return spelling in ("record_law", "joint_kw", "joint_pos")
    return True


def shared_latent_shapes() -> list[tuple[str, str]]:
    """The (shape, spelling) pairs the family generates.

    Separate from `enumerate_shared_latent_probes` for `vector_shapes`' reason:
    `table._slice_probes` takes one probe per pair and must not re-derive which
    pairs exist.
    """
    return [(shape, spelling)
            for shape in SHARED_LATENT_SHAPES
            for spelling in SHARED_LATENT_SPELLINGS
            if shared_latent_supported(shape, spelling)]


def _shared_latent_queries(shape: str, n: int) -> list[str]:
    """Which `latent_query` values a (shape, n) runs — the trimming the module
    docstring justifies. `n = 3` and `singular` run the unqueried case only."""
    if n != 2 or shape == "singular":
        return ["none"]
    return SHARED_LATENT_QUERIES


def enumerate_shared_latent_probes() -> list[SharedLatentProbe]:
    out: list[SharedLatentProbe] = []
    for shape, spelling in shared_latent_shapes():
        for n in SHARED_LATENT_NS:
            for latent_query in _shared_latent_queries(shape, n):
                pid = f"shared.{shape}.n{n}.{spelling}.{latent_query}"
                out.append(SharedLatentProbe(
                    id=pid, shape=shape, n=n, spelling=spelling,
                    latent_query=latent_query, point=SHARED_LATENT_POINTS[:n],
                ))
    return out


@dataclass(frozen=True)
class LinearGaussianProbe:
    """An EXPLICIT linear-Gaussian graph plus the rule composing its fields.

    What `SharedLatentProbe` becomes once the (shape, n, spelling) labels are gone:
    it names the graph directly instead of deriving it. That is what a PARSED model
    gives you — `curated.py` reads a corpus dir whose graph is whatever its author
    wrote, with no shape label to key off — and it is why the two types are not one.
    A generated probe must be identified by its axes (the verdict table's ids and
    `table._spec_justified` are keyed on them); a curated one has no axes at all.

    Both reach the same oracle (`oracle.linear_gaussian_logpdf`), which is the whole
    point: the frozen corpus values then license the code the generated family runs
    on, rather than each side validating itself.

    `nodes` is a tuple and not a dict so the record stays frozen and hashable, for
    `Probe`'s reason. Node order carries no meaning —
    `oracle._linear_gaussian_moments` resolves parents by recursion, so a child may
    be listed before its parent.
    """
    id: str
    nodes: tuple[NormalNode, ...]
    fields: tuple[str, ...]
    composition: str          # a SHARED_LATENT_COMPOSITION value
    point: tuple[float, ...]

    def graph(self) -> dict[str, NormalNode]:
        return {node.name: node for node in self.nodes}


# --------------------------------------------------------------------------
# The literal family
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class LiteralProbe:
    """A probe whose source and oracle are written out, not generated.

    The three families above derive a source from axes and an oracle from the
    §13 fold. That buys a cross product, and it costs every construct the axes
    cannot express: `metricsum` is not a measure at all, and a scalar-affine map
    over an `iid` vector base is not `WRAPS` x `BASES`. Both are load-bearing and
    neither is in the table, so this family carries them as text plus a number
    derived by hand.

    Read `oracle` as the same contract the generated families meet: a value
    derived from the spec and an independent computation, NEVER a value read off
    the engine. Each entry's `note` states the derivation.

    `refusal_spec_justified` decides `table._spec_justified` for this family,
    which cannot key off `wraps`/`shape` here. False means a refusal is a
    tracked capability gap, exactly as it does for the generated families.
    """
    id: str
    source: str
    binding: str
    oracle: float | None
    note: str
    refusal_spec_justified: bool = False


# The metric both metricsum entries use: `g^{ij} = diag(2, 3)`, NOT `eye`.
# §04 "Equivalence to `aggregate` under identity metric" makes `metricsum(eye(n),
# ...)` a plain `aggregate(sum, ...)`, so an identity metric cannot tell a
# dropped `inv(metric)` insertion from a correct one -- the defect class the
# batch audit's finding M2 names. Under `diag(2, 3)` the two spellings differ.
_MS_METRIC = "g = rowstack([[2.0, 0.0], [0.0, 3.0]])\n"

LITERAL_PROBES: tuple[LiteralProbe, ...] = (
    # ---------------------------------------------------------------- metricsum
    # §04 "Static checks" scopes its pairing rule to "Every REPEATED non-output
    # index", so a non-output index occurring ONCE is legal and is summed. This
    # is the spelling flatppl-js #219 and flatppl-rust dd2b39f settled, and no
    # probe reached it.
    #
    # ORACLE. §04 "Lowering to `aggregate`": "Each `_` (lower-variance) axis name
    # in `expr` becomes an `inv(metric)` contraction". With `A^{mu,nu} = [[1, 2],
    # [3, 4]]` and `inv(g) = diag(1/2, 1/3)`, the empty output sums both indices:
    #     sum_{mu,nu} sum_alpha A^{mu,alpha} inv(g)_{alpha,nu}
    #   = sum_alpha (sum_mu A^{mu,alpha}) (sum_nu inv(g)_{alpha,nu})
    #   = 4 * (1/2) + 6 * (1/3) = 4.0
    # exactly. Under `eye(2)` the same body gives 10, so the metric is live in
    # the number. The scalar feeds `Normal`'s `mu`, and the row's value is that
    # normal's log-density at 0.5:
    #     -log(sqrt(2 pi)) - (0.5 - 4)^2 / 2 = -7.043938533204672742
    # (mpmath, 40 dps; the mean is derived above, not measured).
    LiteralProbe(
        id="metricsum.unpaired_lower.direct.single.noconsumer",
        source=_MS_METRIC
        + "A = rowstack([[1.0, 2.0], [3.0, 4.0]])\n"
        + "g: s[] := A[.mu^, .nu_]\n"
        + "m = Normal(mu = s, sigma = 1.0)\n"
        + "lp = logdensityof(m, 0.5)\n",
        binding="lp",
        oracle=-7.043938533204672742,
        note="single unpaired non-output index, one upper and one lower; "
             "metricsum mean 4.0 under g = diag(2, 3), 10.0 under eye(2)",
    ),
    # The paired contraction the same clause governs from the other side: `.i`
    # occurs twice, once upper and once lower, so the metric enters exactly once.
    #     p^i p_i = sum_i p^i sum_alpha inv(g)_{i,alpha} p^alpha
    #             = 9 * (1/2) + 4 * (1/3) = 35/6
    # and the log-density at 0.5 is -15.141160755426894964 (mpmath, 40 dps).
    LiteralProbe(
        id="metricsum.paired_contraction.direct.single.noconsumer",
        source=_MS_METRIC
        + "p = [3.0, 2.0]\n"
        + "g: s[] := p[.i^] * p[.i_]\n"
        + "m = Normal(mu = s, sigma = 1.0)\n"
        + "lp = logdensityof(m, 0.5)\n",
        binding="lp",
        oracle=-15.141160755426894964,
        note="paired upper/lower contraction of one vector; metricsum mean 35/6 "
             "under g = diag(2, 3), 13.0 under eye(2)",
    ),
    # ------------------------------------------------- scalar-affine pushfwd
    # §06 "Engine contract for `pushfwd` density evaluation" case 1 requires a
    # conforming engine to recognize "affine maps composed from
    # `add`/`sub`/`neg`/`mul`/`divide` (with positive scaling)" by name and score
    # them analytically. flatppl-js #224 taught its own density route the scalar
    # spelling over a vector base; the determiniser this table drives still
    # refuses it, so the row reads REFUSES and is NOT spec-justified -- a tracked
    # capability gap against that clause, carrying the value it should have.
    #
    # ORACLE. The pushforward of `iid(Normal(0, 1), 2)` through `x -> 2 x` is a
    # pair of independent `Normal(0, 2)`, so
    #     sum_i [ log phi(y_i / 2) - log 2 ]  at  y = [1.5, -0.5]
    #   = -3.536671427529236102
    # (mpmath, 40 dps, from the closed-form normal log-density).
    LiteralProbe(
        id="pushfwd.scalar_affine_vector.direct.single.noconsumer",
        source="m = pushfwd(x -> 2.0 * x, iid(Normal(mu = 0.0, sigma = 1.0), 2))\n"
               "lp = logdensityof(m, [1.5, -0.5])\n",
        binding="lp",
        oracle=-3.536671427529236102,
        note="scalar-affine map over an iid vector base; §06 case 1 names affine "
             "maps as analytically scoreable, so a refusal is a capability gap",
        refusal_spec_justified=False,
    ),
    # ------------------------------------------ a superposition component's mass
    # `oracle.py` raises `OracleUnsupported` on `superpose`, so the generated
    # families carry NO superposition row at all and this construct reaches the
    # table only by hand. It is load-bearing: a mixture is §06 `normalize`'s own
    # recommended spelling, "To build a normalized mixture distribution, use
    # `normalize(superpose(weighted(w1, M1), weighted(w2, M2)))`".
    #
    # ORACLE. §06 `superpose` is "ν(A) = M₁(A) + M₂(A) + …", so the mass is
    # Σᵢ wᵢ·totalmass(Mᵢ), and §06 `truncate` "restricts the support of measure M
    # to the set S: ν(A) = M(A ∩ S). Does not normalize automatically", so the
    # truncated component keeps Z_t = 2 Phi(1) - 1 = 0.6826894921370859. Then
    #     Z = 0.3 Z_t + 0.7 = 0.9048068476411258
    # and §06 `normalize` shifts the density by -log Z:
    #     log[(0.3 phi(0.5) + 0.7 phi(0.5 - 10)) / Z] = -2.147877551448541081
    # (mpmath, 50 dps, from the closed-form normal density; the second component
    # contributes at 1e-20 and is kept rather than dropped).
    #
    # THE DEFECT THIS PINS. flatppl-js's matSuperpose read only each component's
    # per-atom weights, which carry the weighting events introduced along that
    # component's own chain; matTruncate keeps uniform weights and records the
    # accept rate on `logTotalmass` alone. Z_t never reached the mixture, so
    # `normalize` divided by 0.3 + 0.7 = 1 and every point scored exactly
    # log Z = -0.1000337860820677 low — the same offset at y = 0.5 and y = 10.0,
    # which is what identified it as a missing divisor rather than a mis-scored
    # component.
    #
    # THIS ROW READS `REFUSES` TODAY, AND IT IS THE DETERMINISER THAT REFUSES:
    #     determinize: refuse normalize (node NodeId(16)): normalize of an
    #     unnormalized measure needs a closed-form mass rule; `totalmass` is not
    #     FlatPDL
    # The same table drives `flatppl determinize` before the JS engine, so the
    # engine's own density number never reaches it — flatppl-js scores this
    # source at -2.1478775514485413 directly, and that check lives in
    # flatppl-js's `normalize-pooled-divisor.test.ts`. So a green verdict here
    # does NOT pin the engine fix; it pins the determiniser gap, and the row
    # flips to LOWERS with a value check the moment that gap closes. With
    # `weighted(w, <leaf>)` components the determiniser lowers instead, and
    # rightly emits no divisor at all, since Sigma w = 1 there.
    LiteralProbe(
        id="superpose.component_mass_truncate.direct.single.noconsumer",
        source="m = normalize(superpose(weighted(0.3, truncate("
               "Normal(mu = 0.0, sigma = 1.0), interval(-1.0, 1.0))), "
               "weighted(0.7, Normal(mu = 10.0, sigma = 1.0))))\n"
               "lp = logdensityof(m, 0.5)\n",
        binding="lp",
        oracle=-2.147877551448541081,
        note="a truncated superposition component enters at w*Z_t; the mass-"
             "ignored divisor of 1 scores -2.2479113375 instead. REFUSES today "
             "in the DETERMINISER, not the engine",
        refusal_spec_justified=False,
    ),
    # ------------------------------- a weighted probability leaf's normalizer
    # The generated families carry no `weighted` wrap over a probability leaf
    # with a FUNCTION weight, so this construct reaches the table only by hand.
    # It is §06 `weighted`'s own reading -- "produces the measure ν(A) = ∫_A f(x)
    # dM(x)" -- composed with §06 `normalize`, and its Z is an integral against
    # the base rather than an algebraic factor.
    #
    # ORACLE. The Gaussian exponential tilt is closed-form: e^x φ(x) =
    # e^{1/2} φ(x − 1), so Z = ∫ e^x dΦ = e^{1/2} exactly and the normalized
    # measure is Normal(1, 1). §06 `normalize` shifts by −log Z, giving
    #     log φ(0.5) + 0.5 − 0.5 = −1.043938533204672742
    # (mpmath, 40 dps, from the closed-form normal density; log Z = 0.5 is the
    # Gaussian moment-generating function at 1, derived and not measured).
    #
    # THE DEFECT THIS PINS. flatppl-js had no deterministic arm for this shape.
    # It fell through to `mat-density.resolveNormalizeMasses`'s materialise
    # fallback, which bakes −log Ẑ from the inner measure's tracked
    # `logTotalmass` — for a `weighted` parent, log((1/N) Σᵢ w(xᵢ)) over the
    # N = `sampleCount` atoms of the base's own ensemble. So the divisor moved
    # with the session seed and the sample count: the implied log Ẑ ran 1.099 at
    # N = 1 to 0.494 at N = 100000 against the exact 0.5, and no
    # `marginalizationCount` changed it.
    #
    # THIS ROW READS `REFUSES` TODAY, AND IT IS THE DETERMINISER THAT REFUSES,
    # for the same reason the superposition row above does:
    #     determinize: refuse normalize: normalize of an unnormalized measure
    #     needs a closed-form mass rule; `totalmass` is not FlatPDL
    # So a green verdict here does NOT pin the engine fix; it pins the same
    # determiniser gap, and the row flips to LOWERS with a value check the moment
    # that gap closes. The engine-side check lives in flatppl-js's
    # `weighted-leaf-quad-z.test.ts`, which scores this source at
    # −1.0439385332046156 and holds it fixed across four seeds and every sample
    # count.
    LiteralProbe(
        id="normalize.weighted_leaf_function.direct.single.noconsumer",
        source="m = normalize(weighted(fn(exp(_)), "
               "Normal(mu = 0.0, sigma = 1.0)))\n"
               "lp = logdensityof(m, 0.5)\n",
        binding="lp",
        oracle=-1.043938533204672742,
        note="a function weight over a probability leaf: Z = ∫ e^x dΦ = e^{1/2}. "
             "The seeded ensemble divisor scored -1.6433810714 at N = 1. REFUSES "
             "today in the DETERMINISER, not the engine",
        refusal_spec_justified=False,
    ),
)


def is_literal(probe) -> bool:
    """Whether `probe` is a `LiteralProbe` (see `is_shared_latent`)."""
    return isinstance(probe, LiteralProbe)


def is_linear_gaussian(probe) -> bool:
    """Whether `probe` is a `LinearGaussianProbe` (see `is_shared_latent`)."""
    return isinstance(probe, LinearGaussianProbe)


def is_shared_latent(probe) -> bool:
    """Whether `probe` is a `SharedLatentProbe`.

    A named predicate rather than a bare `isinstance` at every call site: the
    union `Probe | SharedLatentProbe` is threaded through `render`, `oracle` and
    `table`, and each of those reads fields the other type does not have.
    """
    return isinstance(probe, SharedLatentProbe)


def _supported(base: Base, wrap: Wrap) -> bool:
    """Skip combinations that are ill-formed rather than merely refused.

    §06 case 1's well-formedness guard keys on DOMAIN CONTAINMENT, not on
    continuity or discreteness. `log` is restricted to `posreals`, so it is
    ill-formed wherever the base's support isn't a subset of `posreals` —
    `normal` (all reals) and `poisson` (an atom at 0) both fail that, leaving
    only `gamma`/`beta`. `sqrt` is restricted to `nonnegreals`: `poisson`'s
    support `{0,1,2,...}` IS a subset of `nonnegreals`, so `sqrt` is
    well-formed there too (a bijection on a discrete support doesn't distort
    the counting measure — see §06 line 28); only `normal` fails `sqrt`.
    Every other wrap in `WRAPS` carries no domain restriction at all.
    """
    if wrap.kind == "pushfwd" and wrap.args[0] == "log":
        return base.kind in ("gamma", "beta")
    if wrap.kind == "pushfwd" and wrap.args[0] == "sqrt":
        return base.kind in ("gamma", "beta", "poisson")
    return True


def _point_for(base: Base, wrap: Wrap) -> float:
    """A point inside the WRAPPED measure's support.

    Derived from `INNER[base.kind]`, never hardcoded: the preimage of a
    derived point is `INNER[base.kind]` BY CONSTRUCTION, which is in
    `base`'s own support by construction too, so the invariant holds for
    every base without per-base casework here.
    """
    inner = INNER[base.kind]
    if wrap.kind == "pushfwd":
        return _FORWARD[wrap.args[0]](inner)
    if wrap.kind == "affine":
        return wrap.args[0] * inner + wrap.args[1]
    if wrap.kind == "locscale":
        return wrap.args[0] + wrap.args[1] * inner
    return inner


def is_vector_base(base: Base) -> bool:
    """Whether `base`'s variate is a VECTOR — one of `VECTOR_BASES`.

    Keyed on the kind, not on `params`' shape: `Multinomial`'s first parameter is
    the scalar `n`, so "has a tuple parameter" would be an accident of position.
    """
    return base.kind in {b.kind for b in VECTOR_BASES}


def in_support(base: Base, x: float | tuple[float, ...] | list[float]) -> bool:
    """Whether `x` lies in `base`'s own (unwrapped) support.

    Public because Task 2's oracle needs the same predicate to decide
    whether a density is even defined at a given point — one copy, not two
    that can drift.

    A vector base's support is a CONSTRAINT SURFACE, not a box, and both surfaces
    are what §08 says they are:

    * `Multinomial(n, p)` — "{x in N_0^k : sum_i x_i = n}". Every cell nonnegative
      AND integral AND the cells summing to `n`. The same 1e-9 lattice tolerance as
      `poisson` below, and for the same reason: a cell recovered through a float
      round-trip need not land exactly on its integer.
    * `Dirichlet(alpha)` — `stdsimplex(n)`, i.e. "sum p_i = 1, p_i >= 0". Cells are
      required STRICTLY positive: §08's density carries `x_i^(alpha_i - 1)`, which
      diverges at 0 for `alpha_i < 1`, so 0 is not a point where the density is
      defined for every admissible `alpha`.
    """
    if base.kind == "multinomial":
        n, _p = base.params
        cells = list(x)
        if any(abs(c - round(c)) >= 1e-9 or c < -1e-9 for c in cells):
            return False
        return sum(round(c) for c in cells) == n
    if base.kind == "dirichlet":
        # §08's support is INCLUSIVE: "{p in R^n : sum p_i = 1, p_i >= 0}". A zero
        # cell is IN the support, and whether the density is finite there is a
        # separate question the density function answers (`oracle` handles the three
        # cases of `x_i^(alpha_i - 1)` at x_i = 0). Excluding 0 here would conflate
        # "the density is not defined there" with "the point is outside the support",
        # and it would be observably wrong for alpha_i < 1, where §08's density
        # DIVERGES at a zero cell rather than vanishing.
        cells = list(x)
        return all(c >= 0.0 for c in cells) and abs(sum(cells) - 1.0) < 1e-9
    if base.kind == "normal":
        return True
    if base.kind == "gamma":
        return x > 0.0
    if base.kind == "beta":
        return 0.0 < x < 1.0
    if base.kind == "poisson":
        # Tolerant of float round-trip noise (e.g. `math.sqrt(2.0) ** 2`
        # lands at 2.0000000000000004, not exactly 2.0) rather than an exact
        # `.is_integer()`, which would reject a genuinely-integer point for
        # a reason that has nothing to do with support.
        return x >= -1e-9 and abs(x - round(x)) < 1e-9
    raise ValueError(f"unknown base kind: {base.kind}")


def _vector_point_for(base: Base, wrap: Wrap) -> tuple[float, ...]:
    """`_point_for`'s vector counterpart: the forward map applied CELL-WISE.

    Derived from `VECTOR_INNER[base.kind]` for exactly the reason `_point_for` is
    derived from `INNER` — the preimage of a derived point is the inner point by
    construction, so it lies on the support's constraint surface by construction too.
    """
    inner = VECTOR_INNER[base.kind]
    if wrap.kind == "pushfwd":
        f = _FORWARD[wrap.args[0]]
        return tuple(f(c) for c in inner)
    return tuple(inner)


def _wrap_name(wrap: Wrap) -> str:
    """A wrap's slug in a probe id. Joins ALL args (see `enumerate_probes`)."""
    return wrap.kind + ("_" + "_".join(str(a) for a in wrap.args) if wrap.args else "")


def vector_shapes() -> list[tuple[Base, Wrap]]:
    """The (base, wrap) shapes the vector family generates, blocked ones removed.

    Separate from `enumerate_vector_probes` so `table._slice_probes` can take one
    probe per shape without re-deriving which shapes exist.
    """
    return [(b, w) for b in VECTOR_BASES for w in VECTOR_WRAPS
            if (b.kind, w) not in _HELD_OUT]


def enumerate_vector_probes() -> list[Probe]:
    out: list[Probe] = []
    for base, wrap in vector_shapes():
        for spelling in VECTOR_SPELLINGS:
            for ordering in ORDERINGS:
                pid = (f"{base.kind}.{_wrap_name(wrap)}.{spelling}.{ordering}."
                       f"noconsumer")
                out.append(Probe(
                    id=pid, base=base, wraps=(wrap,), spelling=spelling,
                    ordering=ordering, consumer=False,
                    point=_vector_point_for(base, wrap),
                ))
    return out


def enumerate_probes() -> list[Probe | SharedLatentProbe]:
    """The scalar axes' full cross-product, then each targeted family.

    The targeted families are APPENDED, and the scalar loop below is untouched:
    the verdict table is frozen history keyed on `probe_id`, so every existing id
    and its `expected` value has to survive this addition byte-identically.

    This returns THE WHOLE SPACE, mixed types included, rather than the scalar
    space with the families reachable through separate calls. A reader who
    enumerates the space and quietly misses a family is the failure this shape
    prevents; the cost is that a caller touching `Probe`-only fields has to skip
    the shared-latent rows explicitly (`is_shared_latent`), which is a visible
    conditional rather than a silent omission.
    """
    out: list[Probe] = []
    for base in BASES:
        for wrap in WRAPS:
            if not _supported(base, wrap):
                continue
            for spelling in SPELLINGS:
                for ordering in ORDERINGS:
                    for consumer in (False, True):
                        # Join ALL args, not just the first — ids are the
                        # verdict table's primary key, so e.g. truncate's
                        # upper bound and locscale's scale must both appear
                        # or two distinct wraps of the same kind could collide.
                        wname = _wrap_name(wrap)
                        pid = (f"{base.kind}.{wname}.{spelling}.{ordering}."
                               f"{'consumer' if consumer else 'noconsumer'}")
                        out.append(Probe(
                            id=pid, base=base, wraps=(wrap,), spelling=spelling,
                            ordering=ordering, consumer=consumer,
                            point=_point_for(base, wrap),
                        ))
    return (out + enumerate_vector_probes() + enumerate_shared_latent_probes()
            + list(LITERAL_PROBES))
