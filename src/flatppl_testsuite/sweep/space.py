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

    A `log` pushforward over a base with mass on the negatives is not a
    determiniser question — the map is undefined there, so §06's domain
    restriction makes a refusal CORRECT and the probe carries no information.
    Excluding it keeps the space's refusals meaningful. `sqrt` is the same at
    `nonnegreals`; `discrete` bases take no continuous map at all.
    """
    if base.kind == "poisson":
        return wrap.kind in ("identity", "weighted", "logweighted")
    if wrap.kind == "pushfwd" and wrap.args[0] in ("log", "sqrt"):
        return base.kind in ("gamma", "beta")
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
                        wname = wrap.kind + ("_" + str(wrap.args[0]) if wrap.args else "")
                        pid = (f"{base.kind}.{wname}.{spelling}.{ordering}."
                               f"{'consumer' if consumer else 'noconsumer'}")
                        out.append(Probe(
                            id=pid, base=base, wraps=(wrap,), spelling=spelling,
                            ordering=ordering, consumer=consumer,
                            point=_point_for(base, wrap),
                        ))
    return out
