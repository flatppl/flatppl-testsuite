"""Probe -> FlatPPL source. Pure.

Separate from `classify` so the whole space can be rendered and parse-checked
with no determiniser present, and so the oracle never sees source text — it
walks the Probe, which is the structure we built.
"""
from __future__ import annotations

from dataclasses import dataclass

from flatppl_testsuite.sweep.space import (
    SHARED_LATENT_FIELD_NAMES,
    SHARED_LATENT_LATENT_POINT,
    Base,
    LiteralProbe,
    NormalNode,
    Probe,
    SharedLatentProbe,
    Wrap,
    _latent_name,
    is_literal,
    is_shared_latent,
    shared_latent_graph,
)

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
    # §08 "Multivariate distributions": `Multinomial(n, p)` and `Dirichlet(alpha)`.
    # `Multinomial`'s `n` is `elementof(posintegers)` per §08, so it is rendered as
    # an integer literal (`n = 5`), not `5.0`.
    "multinomial": "Multinomial(n = {0}, p = {1})",
    "dirichlet": "Dirichlet(alpha = {0})",
}


@dataclass(frozen=True)
class RenderedProbe:
    source: str
    binding: str


def _value_src(v) -> str:
    """A FlatPPL value literal: a scalar as itself, a vector as `[a, b, c]` (§05's
    array literal). Vector parameters arrive as tuples and vector points as lists,
    so both sequence types render the same way — `repr` on a tuple would emit
    `(0.2, 0.3, 0.5)`, which is not FlatPPL syntax at all."""
    if isinstance(v, (tuple, list)):
        return "[" + ", ".join(_value_src(c) for c in v) + "]"
    return str(v)


def _base_src(base: Base) -> str:
    return _CTOR[base.kind].format(*(_value_src(p) for p in base.params))


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


def _normal_src(node: NormalNode) -> str:
    """`Normal(mu = ..., sigma = ...)` for one graph node. `mu` is the parent's
    name when there is one, which is what makes the graph shared rather than
    replicated — a literal `mu` at a child would be a different model with the
    same marginals, i.e. the `joint_ctor` arm."""
    mu = node.parent if node.parent is not None else _value_src(node.mu)
    return f"Normal(mu = {mu}, sigma = {_value_src(node.sigma)})"


def _record_src(pairs: list[tuple[str, str]]) -> str:
    return "record(" + ", ".join(f"{k} = {v}" for k, v in pairs) + ")"


def _shared_measure_src(probe: SharedLatentProbe, nodes: dict[str, NormalNode],
                        field_nodes: tuple[str, ...]) -> str:
    """The measure expression a shared-latent probe queries, per §06's spellings."""
    labels = SHARED_LATENT_FIELD_NAMES[:probe.n]
    s = probe.spelling
    if s == "record_law":
        return f"lawof({_record_src(list(zip(labels, field_nodes)))})"
    if s == "joint_kw":
        return "joint(" + ", ".join(
            f"{lab} = lawof({node})" for lab, node in zip(labels, field_nodes)) + ")"
    if s == "joint_pos":
        return "joint(" + ", ".join(f"lawof({node})" for node in field_nodes) + ")"
    if s == "joint_ctor":
        return "joint(" + ", ".join(
            f"{lab} = {_normal_src(nodes[lab])}" for lab in labels) + ")"
    if s in ("ctor_shared_kw", "ctor_shared_pos"):
        # Each component is the CONSTRUCTOR of the corresponding field's draw, so its
        # `mu` names that field's parent -- the latent. §06 keeps a node shared through
        # a stochastic constructor parameter as one node of the composed trace, which
        # is what makes this the compound law rather than a product.
        ctors = [_normal_src(nodes[node]) for node in field_nodes]
        if s == "ctor_shared_pos":
            return "joint(" + ", ".join(ctors) + ")"
        return "joint(" + ", ".join(
            f"{lab} = {c}" for lab, c in zip(labels, ctors)) + ")"
    if s == "iid":
        # §06 `iid(M, size)`: the product measure. `size` is "an integer (1-D
        # length)", so it is emitted as an integer literal, not `2.0`.
        return f"iid(lawof({field_nodes[0]}), {probe.n})"
    raise ValueError(f"unrendered shared-latent spelling: {s}")


def _render_shared_latent(probe: SharedLatentProbe) -> RenderedProbe:
    """A shared-latent probe's source: the draw graph, then one `logdensityof`.

    The graph is emitted from `space.shared_latent_graph` in insertion order,
    which is dependency order by construction — a node's `mu` names its parent, so
    a parent emitted after its child would not even parse, and that is a stronger
    guard than a topological sort with nothing to check it.

    `joint_ctor` emits NO field draws. Its components are fresh constructors, so a
    drawn `f1` would sit in the model unconsumed by the query, and an unconsumed
    draw is a determiniser refusal in its own right (`determinize` refuses them by
    design) — the row would then report that refusal rather than the
    product-of-marginals arm it exists for. The latent is still emitted when the
    `latent_query` axis asks for it, because that query consumes it.
    """
    nodes, field_nodes = shared_latent_graph(probe.shape, probe.n, probe.spelling)
    latent = _latent_name(probe.shape)
    lines: list[str] = []

    if probe.spelling == "joint_ctor":
        if probe.latent_query != "none":
            traced, _ = shared_latent_graph(probe.shape, probe.n, "record_law")
            lines.append(f"{latent} = draw({_normal_src(traced[latent])})")
    elif probe.spelling in ("ctor_shared_kw", "ctor_shared_pos"):
        # Emit the LATENTS only. The fields are the joint's own constructor
        # components -- fresh draws made inside the measure -- so drawing them as
        # bindings too would both change the model and leave an unconsumed draw. Every
        # latent IS consumed, by the constructor parameter that names it.
        for name, node in nodes.items():
            if name not in field_nodes:
                lines.append(f"{name} = draw({_normal_src(node)})")
    else:
        for name, node in nodes.items():
            lines.append(f"{name} = draw({_normal_src(node)})")

    labels = SHARED_LATENT_FIELD_NAMES[:probe.n]
    measure = _shared_measure_src(probe, nodes, field_nodes)
    if probe.spelling in ("joint_pos", "iid", "ctor_shared_pos"):
        # §06: a POSITIONAL `joint` combines the component variates via `cat`, and
        # `iid` over a scalar law is an array — both are vector variates, so the
        # query point is an array literal and not a record.
        point = _value_src(list(probe.point))
    else:
        point = _record_src([(lab, _value_src(v))
                             for lab, v in zip(labels, probe.point)])
    query = f"lp = logdensityof({measure}, {point})"

    latent_query = (f"lp_latent = logdensityof(lawof({latent}), "
                    f"{_value_src(SHARED_LATENT_LATENT_POINT)})")
    if probe.latent_query == "before":
        lines += [latent_query, query]
    elif probe.latent_query == "after":
        lines += [query, latent_query]
    else:
        lines.append(query)

    return RenderedProbe(source="\n".join(lines) + "\n", binding="lp")


def render(probe: Probe | SharedLatentProbe | LiteralProbe) -> RenderedProbe:
    if is_literal(probe):
        # Already source text -- there is nothing to compose. The family exists
        # because its constructs are not `BASES` x `WRAPS` (see `space`).
        return RenderedProbe(source=probe.source, binding=probe.binding)
    if is_shared_latent(probe):
        return _render_shared_latent(probe)

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

    pt = _value_src(probe.point)
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
