"""The probe space: pure data.

Axes are derived from the spec, not invented. Bases span the SUPPORTS that
drive §06 case 1's domain guards (a `log` map is only defined over a positive
base, so the guard is support-dependent). Wraps are §13 "Output reduction"'s own
enumerated list, one normative density rule each. Spellings are restricted to
pairs §06/§04 declare EQUIVALENT, which gives a correctness signal needing no
oracle: two spellings of one measure that lower to different densities are wrong
even if neither has been scored. Orderings exist because the pinned-by-a-later-
query case is where two silent wrong densities were found by hand.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Base:
    kind: str          # "normal" | "gamma" | "beta" | "poisson"
    params: tuple      # positional, e.g. (0.0, 1.0)


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
    point: float


# Support-distinct bases. `point` per base is chosen INSIDE the base's support
# so an unwrapped probe is always scoreable; wraps that move the support carry
# their own point override in `_point_for`.
BASES = [
    Base("normal", (0.0, 1.0)),      # reals
    Base("gamma", (2.0, 1.0)),       # posreals  (shape, rate — §08 order)
    Base("beta", (2.0, 3.0)),        # (0, 1)
    Base("poisson", (3.0,)),         # nonneg integers
]

# §13 "Output reduction"'s list. Each entry is one normative rule.
WRAPS = [
    Wrap("identity", ()),
    Wrap("pushfwd", ("exp",)),
    Wrap("pushfwd", ("log",)),
    Wrap("pushfwd", ("neg",)),
    Wrap("pushfwd", ("sqrt",)),
    Wrap("affine", (2.0, 1.0)),          # 2*x + 1
    Wrap("truncate", (0.0, "inf")),
    Wrap("weighted", (0.5,)),
    Wrap("logweighted", (-0.6931471805599453,)),
    Wrap("normalize", ()),
    Wrap("locscale", (1.0, 2.0)),
]

SPELLINGS = ["direct", "stochastic_node", "record"]
ORDERINGS = ["single", "pinned_earlier", "pinned_later"]


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
    """A point inside the WRAPPED measure's support."""
    inner = {"normal": 0.5, "gamma": 1.5, "beta": 0.4, "poisson": 2.0}[base.kind]
    if wrap.kind == "pushfwd":
        return {"exp": 1.6487212707001282, "log": 0.4054651081081644,
                "neg": -inner, "sqrt": 1.224744871391589}[wrap.args[0]]
    if wrap.kind == "affine":
        return wrap.args[0] * inner + wrap.args[1]
    if wrap.kind == "locscale":
        return wrap.args[0] + wrap.args[1] * inner
    return inner


def enumerate_probes() -> list[Probe]:
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
                        wname = wrap.kind + ("_" + "_".join(str(a) for a in wrap.args)
                                             if wrap.args else "")
                        pid = (f"{base.kind}.{wname}.{spelling}.{ordering}."
                               f"{'consumer' if consumer else 'noconsumer'}")
                        out.append(Probe(
                            id=pid, base=base, wraps=(wrap,), spelling=spelling,
                            ordering=ordering, consumer=consumer,
                            point=_point_for(base, wrap),
                        ))
    return out
