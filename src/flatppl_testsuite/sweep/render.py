"""Probe -> FlatPPL source. Pure.

Separate from `classify` so the whole space can be rendered and parse-checked
with no determiniser present, and so the oracle never sees source text — it
walks the Probe, which is the structure we built.
"""
from __future__ import annotations

from dataclasses import dataclass

from flatppl_testsuite.sweep.space import Base, Probe, Wrap

# Parameter names are §08's, and they have to be exactly right: a wrong one is
# silent everywhere downstream. `Poisson(lambda = 3.0)` parses, converts to
# `(%kwarg lambda 3.0)`, and determinizes with no diagnostic, so no test in the
# toolchain guards these strings except
# `test_base_constructors_use_the_parameter_names_the_corpus_uses`, which pins them
# against the committed corpus models.
_CTOR = {
    "normal": "Normal(mu = {0}, sigma = {1})",
    "gamma": "Gamma(shape = {0}, rate = {1})",
    "beta": "Beta(alpha = {0}, beta = {1})",
    "poisson": "Poisson(rate = {0})",
}


@dataclass(frozen=True)
class RenderedProbe:
    source: str
    binding: str


def _base_src(base: Base) -> str:
    return _CTOR[base.kind].format(*base.params)


def _wrap_src(wrap: Wrap, inner: str) -> str:
    k = wrap.kind
    if k == "identity":
        return inner
    if k == "pushfwd":
        return f"pushfwd({wrap.args[0]}, {inner})"
    if k == "affine":
        return f"pushfwd(x -> {wrap.args[0]} * x + {wrap.args[1]}, {inner})"
    if k == "truncate":
        return f"truncate({inner}, interval({wrap.args[0]}, {wrap.args[1]}))"
    if k == "weighted":
        return f"weighted({wrap.args[0]}, {inner})"
    if k == "logweighted":
        return f"logweighted({wrap.args[0]}, {inner})"
    if k == "normalize":
        return f"normalize({inner})"
    if k == "locscale":
        return f"locscale({inner}, {wrap.args[0]}, {wrap.args[1]})"
    raise ValueError(f"unrendered wrap kind: {k}")


def _fold(wraps: tuple[Wrap, ...], inner: str) -> str:
    """Compose every wrap onto `inner`, in order."""
    for w in wraps:
        inner = _wrap_src(w, inner)
    return inner


def render(probe: Probe) -> RenderedProbe:
    lines: list[str] = []
    if probe.spelling == "direct":
        lines.append(f"m = {_fold(probe.wraps, _base_src(probe.base))}")
        query_measure = "m"
    elif probe.spelling == "stochastic_node":
        # §06 declares this equivalent to the direct spelling.
        lines.append(f"mb = {_base_src(probe.base)}")
        lines.append("x = draw(mb)")
        lines.append(f"m = {_fold(probe.wraps, 'lawof(x)')}")
        query_measure = "m"
    elif probe.spelling == "record":
        # Same equivalence, spelled through a record law rather than a bare
        # scalar law — every wrap folds onto the record law exactly as the
        # `direct` spelling folds them onto the base constructor.
        lines.append(f"mb = {_base_src(probe.base)}")
        lines.append("x = draw(mb)")
        lines.append(f"m = {_fold(probe.wraps, 'lawof(record(x = x))')}")
        query_measure = "m"
    else:
        raise ValueError(f"unrendered spelling: {probe.spelling}")

    if probe.consumer:
        lines.append("w_consumer = 1.0")

    pt = probe.point
    if probe.spelling == "record":
        query = f"lp = logdensityof({query_measure}, record(x = {pt}))"
    else:
        query = f"lp = logdensityof({query_measure}, {pt})"

    if probe.ordering == "pinned_earlier":
        lines.insert(0, "z_pin = draw(Normal(mu = 0.0, sigma = 1.0))")
        lines.append("lp_pin = logdensityof(lawof(z_pin), 0.3)")
        lines.append(query)
    elif probe.ordering == "pinned_later":
        lines.insert(0, "z_pin = draw(Normal(mu = 0.0, sigma = 1.0))")
        lines.append(query)
        lines.append("lp_pin = logdensityof(lawof(z_pin), 0.3)")
    else:
        lines.append(query)

    return RenderedProbe(source="\n".join(lines) + "\n", binding="lp")
