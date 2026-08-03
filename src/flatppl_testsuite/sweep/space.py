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
    point: float | list[float]


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
VECTOR_INNER = {
    "multinomial": [1.0, 2.0, 2.0],
    "dirichlet": [0.2, 0.3, 0.5],
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
#                  `Multinomial` the lattice snap as well. Currently unevaluable
#                  (`_ENGINE_BLOCKED`), pinned in `tests/sweep/test_vector_arms.py`.
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

# (base kind, wrap) shapes whose emitted FlatPDL is well-formed but which the
# `flatppl-js` scorer cannot evaluate, so they classify `MALFORMED` — a frozen
# `MALFORMED` row is banned from the verdict table
# (`tests/sweep/test_gate.py::test_the_table_records_no_malformed_and_no_wrong_numbers`)
# and would be indistinguishable from a determiniser defect.
#
# **This is an ENGINE gap list, not a well-formedness gate** — `_supported` is
# where ill-formed combinations go. Each entry is pinned by its own test in
# `tests/sweep/test_vector_arms.py`, which asserts the arm fires in the emitted
# text AND asserts the current crash. When `flatppl-js` learns the missing op that
# test fails, which is what forces the probe back in rather than leaving the arm
# silently uncovered.
_ENGINE_BLOCKED = {
    ("multinomial", Wrap("pushfwd", ("neg",))):
        "flatppl-js `real` rejects an integer array: the determiniser's lattice "
        "snap emits `real(round.(v))` over a vector variate, and the scorer throws "
        "`real: arg 1 expects complex, got array of integer`",
    ("multinomial", Wrap("pushfwd", ("exp",))):
        "the same `real` gap as pushfwd(neg), reached first: this shape emits BOTH "
        "the lattice snap and the `cartpow` image gate, and flatppl-js supports "
        "neither `real` over an integer array nor `in` over a `cartpow` set",
    ("dirichlet", Wrap("pushfwd", ("exp",))):
        "flatppl-js `in` does not handle a `cartpow` set: the emitted image gate "
        "`y in cartpow(posreals, 3)` throws `Cannot read properties of undefined "
        "(reading 'length')`. The oracle would withhold a value here in any case "
        "-- see `oracle._MANIFOLD_SAFE_FORWARDS`",
}


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


def in_support(base: Base, x: float | list[float]) -> bool:
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
        cells = list(x)
        return all(c > 0.0 for c in cells) and abs(sum(cells) - 1.0) < 1e-9
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


def _vector_point_for(base: Base, wrap: Wrap) -> list[float]:
    """`_point_for`'s vector counterpart: the forward map applied CELL-WISE.

    Derived from `VECTOR_INNER[base.kind]` for exactly the reason `_point_for` is
    derived from `INNER` — the preimage of a derived point is the inner point by
    construction, so it lies on the support's constraint surface by construction too.
    """
    inner = VECTOR_INNER[base.kind]
    if wrap.kind == "pushfwd":
        f = _FORWARD[wrap.args[0]]
        return [f(c) for c in inner]
    return list(inner)


def _wrap_name(wrap: Wrap) -> str:
    """A wrap's slug in a probe id. Joins ALL args (see `enumerate_probes`)."""
    return wrap.kind + ("_" + "_".join(str(a) for a in wrap.args) if wrap.args else "")


def vector_shapes() -> list[tuple[Base, Wrap]]:
    """The (base, wrap) shapes the vector family generates, blocked ones removed.

    Separate from `enumerate_vector_probes` so `table._slice_probes` can take one
    probe per shape without re-deriving which shapes exist.
    """
    return [(b, w) for b in VECTOR_BASES for w in VECTOR_WRAPS
            if (b.kind, w) not in _ENGINE_BLOCKED]


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


def enumerate_probes() -> list[Probe]:
    """The scalar axes' full cross-product, then the targeted vector family.

    The vector probes are APPENDED, and the scalar loop below is untouched: the
    verdict table is frozen history keyed on `probe_id`, so every existing id and
    its `expected` value has to survive this addition byte-identically.
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
    return out + enumerate_vector_probes()
