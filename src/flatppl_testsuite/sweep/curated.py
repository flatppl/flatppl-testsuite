"""Express the committed `logdensity` corpus cases as `Probe`s, where possible.

This is the oracle's validation gate. Each `corpora/<corpus>/<dir>/` carries a
frozen `expected` that was derived independently, by hand, in that dir's own
`test.py` — several of them against a *different* closed form than the oracle
uses (`scipy.stats.lognorm` for the pushforward-of-exp rule, `truncnorm` for
normalize-of-truncate). Reproducing those values is what licenses trusting the
compositional oracle on generated ground.

The matcher is deliberately **fail-closed**: any line, construct, keyword, or
point shape it does not explicitly recognize makes the whole directory
unexpressible. A false match would validate the oracle against the wrong value,
which is strictly worse than a smaller validated set — so a non-match is a
reported gap (`unmatched_cases`), never a guess.

Two model shapes are recognized:

1. **Inline** (`corpora/fragment/*`): one model file whose `logdensityof` query
   names a numeric point directly.
2. **Parameterized** (`corpora/stablehlo/*`): a `model.flatppl` of
   `elementof` declarations plus one `draw`, a `query.flatppl` holding the
   `logdensityof`, and a `points` list binding every declared parameter. Each
   point becomes its own case, suffixed `#<index>`. These need no ABI or engine
   — the oracle walks structure, and a point dict is exactly the binding it
   needs.

A case whose query is a MULTI-FIELD joint or record law is expressed as a
`space.LinearGaussianProbe` instead of a `Probe` (`_joint_measure`), because those
are the `oracle.linear_gaussian_logpdf` path and not the scalar §13 fold. Without this,
the multivariate oracle would ship with no curated reproduction gate at all —
the gap `flatppl-dev/density-sweep-notes.md` records against `Multinomial`, which
is licensed by "the hand-derived log-gamma form plus the exact lattice-mass sum,
not a frozen independent value". Three corpus dirs supply that gate here
(`fragment/shared_latent_record`, `fragment/shared_latent_joint`,
`fragment/shared_latent_joint_positional`), each frozen against a `test.py` that
derives `MvNormal(mu0 * 1, s0**2 * J + diag(sigma**2))` by hand.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from flatppl_testsuite.sweep.space import (
    Base,
    LinearGaussianProbe,
    NormalNode,
    Probe,
    Wrap,
)

REPO = Path(__file__).resolve().parents[3]

_IDENT = r"[A-Za-z_][A-Za-z0-9_]*"


class _NoMatch(Exception):
    """This corpus directory is not expressible as a Probe."""


# §08's parameter names and their positional order. Keyword spellings are matched
# STRICTLY: §08 gives `Poisson(rate)`, so a model spelling it `Poisson(lambda =
# ...)` is not a Poisson as far as this matcher is concerned. Being strict here is
# the point — a matcher that accepted any keyword would silently bind a parameter
# to the wrong slot.
_BASE_SPECS = {
    "Normal": ("normal", ("mu", "sigma")),
    "Gamma": ("gamma", ("shape", "rate")),
    "Beta": ("beta", ("alpha", "beta")),
    "Poisson": ("poisson", ("rate",)),
    # §08 "Multivariate distributions". `Dirichlet(alpha)`'s single parameter is a
    # VECTOR, so it is listed in `_VECTOR_PARAMS` below as well.
    "Dirichlet": ("dirichlet", ("alpha",)),
}

# Which constructor parameters are vector-valued, by (constructor, parameter name).
# A parameter resolved through the wrong one of `_Ctx.number`/`_Ctx.vector` would
# either raise or bind a length-1 vector, so the split is explicit rather than
# inferred from whatever the point happens to hold.
_VECTOR_PARAMS = {("Dirichlet", "alpha")}

# `pushfwd` forward maps the oracle implements, by every spelling the corpus uses.
_FORWARD_NAMES = {
    "exp": "exp", "log": "log", "neg": "neg", "sqrt": "sqrt",
    "fn(exp(_))": "exp", "fn(log(_))": "log", "fn(sqrt(_))": "sqrt",
}


# --------------------------------------------------------------------------
# Minimal expression tools. Not a FlatPPL parser -- just enough to peel a
# nest of calls, and strict enough to refuse anything else.
# --------------------------------------------------------------------------

def _split_top(s: str) -> list[str]:
    """Split on top-level commas (depth 0 w.r.t. `(` and `[`)."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return [a.strip() for a in out]


def _as_call(expr: str) -> tuple[str, str] | None:
    """`f(a, b)` -> `("f", "a, b")`, but only when the whole expression IS the
    call: `f(a) + g(b)` returns None rather than pretending to be `f`."""
    expr = expr.strip()
    m = re.match(rf"^({_IDENT})\(", expr)
    if not m or not expr.endswith(")"):
        return None
    depth = 0
    for i, ch in enumerate(expr):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return (m.group(1), expr[m.end():-1]) if i == len(expr) - 1 else None
    return None


def _args(argstr: str) -> tuple[list[str], dict[str, str]]:
    """Positional and keyword arguments, unevaluated."""
    pos, kw = [], {}
    for a in _split_top(argstr):
        m = re.match(rf"^({_IDENT})\s*=\s*(.+)$", a, re.DOTALL)
        if m:
            kw[m.group(1)] = m.group(2).strip()
        else:
            pos.append(a)
    return pos, kw


def _literal(tok: str) -> float | None:
    try:
        return float(tok.strip())
    except ValueError:
        return None


class _Ctx:
    """One directory's bindings, plus the point substitution for one case."""

    def __init__(self, bindings: dict[str, str], aliases: dict[str, str],
                 params: set[str], point: dict[str, float] | None):
        self.bindings = bindings
        self.aliases = aliases
        self.params = params
        self.point = point or {}

    def number(self, tok: str) -> float:
        """A numeric literal, or a declared parameter bound by this case's point."""
        lit = _literal(tok)
        if lit is not None:
            return lit
        name = tok.strip()
        if name in self.params and name in self.point:
            value = self.point[name]
            if isinstance(value, list):
                raise _NoMatch(f"{name} is vector-valued in this case's point")
            return float(value)
        raise _NoMatch(f"unresolved numeric argument {tok!r}")

    def vector(self, tok: str) -> tuple[float, ...]:
        """A vector: an `[a, b, c]` literal (§05), or a declared parameter this
        case's point binds to a JSON list. Returned as a tuple because `Base` is
        frozen and hashable (`space.Base.params`)."""
        tok = tok.strip()
        if tok.startswith("[") and tok.endswith("]"):
            cells = [c for c in _split_top(tok[1:-1]) if c]
            if not cells:
                raise _NoMatch("empty vector literal")
            return tuple(self.number(c) for c in cells)
        if tok in self.params and tok in self.point:
            value = self.point[tok]
            if not isinstance(value, list):
                raise _NoMatch(f"{tok} is not vector-valued in this case's point")
            return tuple(float(c) for c in value)
        if tok in self.bindings and tok not in self.params:
            return self.vector(self.bindings[tok])
        raise _NoMatch(f"unresolved vector argument {tok!r}")


def _forward(tok: str, ctx: _Ctx) -> Wrap:
    """The `pushfwd` function argument, as a Wrap."""
    tok = re.sub(r"\s+", "", tok.strip())
    tok = re.sub(r"\s+", "", ctx.aliases.get(tok, tok))
    if tok in _FORWARD_NAMES:
        return Wrap("pushfwd", (_FORWARD_NAMES[tok],))
    # Affine lambdas, in the spellings the corpus uses. `x -> mul(2.0, x)` and
    # `x -> 2.0 * x` both denote `y = a*x + b` with b defaulting to 0.
    for pat, (ai, bi) in (
        (rf"^{_IDENT}->mul\(([^,]+),{_IDENT}\)$", (1, None)),
        (rf"^{_IDENT}->([-\d.eE+]+)\*{_IDENT}$", (1, None)),
        (rf"^{_IDENT}->([-\d.eE+]+)\*{_IDENT}\+([-\d.eE+]+)$", (1, 2)),
    ):
        m = re.match(pat, tok)
        if m:
            a = ctx.number(m.group(ai))
            b = ctx.number(m.group(bi)) if bi else 0.0
            return Wrap("affine", (a, b))
    raise _NoMatch(f"unrecognized pushfwd map {tok!r}")


def _peel(expr: str, ctx: _Ctx) -> tuple[Base, tuple[Wrap, ...]]:
    """Measure expression -> (base, wraps innermost-first).

    Wrap order matches `render._fold`: the outermost construct in the source is
    the LAST element, which is the order `oracle.true_logpdf` expects.
    """
    expr = expr.strip()
    call = _as_call(expr)
    if call is None:
        name = expr
        if re.fullmatch(_IDENT, name) and name in ctx.bindings:
            return _peel(ctx.bindings[name], ctx)
        raise _NoMatch(f"not a measure expression: {expr!r}")

    head, argstr = call
    pos, kw = _args(argstr)

    if head == "draw":
        if kw or len(pos) != 1:
            raise _NoMatch("draw arity")
        return _peel(pos[0], ctx)

    if head == "lawof":
        # §04: `lawof(x)` is the law of the drawn variable, so its density is the
        # density of the measure `x` was drawn from. A one-field record law has
        # the same density as that field's scalar law (§06's product rule over a
        # single component); more fields would be a `joint`, which is not a Probe.
        if kw or len(pos) != 1:
            raise _NoMatch("lawof arity")
        rec = _as_call(pos[0])
        if rec and rec[0] == "record":
            rpos, rkw = _args(rec[1])
            if rpos or len(rkw) != 1:
                raise _NoMatch("record law is not single-field")
            return _peel(next(iter(rkw.values())), ctx)
        return _peel(pos[0], ctx)

    if head == "normalize":
        if kw or len(pos) != 1:
            raise _NoMatch("normalize arity")
        base, wraps = _peel(pos[0], ctx)
        return base, wraps + (Wrap("normalize", ()),)

    if head == "truncate":
        if kw or len(pos) != 2:
            raise _NoMatch("truncate arity")
        iv = _as_call(pos[1])
        if not iv or iv[0] != "interval":
            raise _NoMatch(f"truncation set is not an interval: {pos[1]!r}")
        ipos, ikw = _args(iv[1])
        if ikw or len(ipos) != 2:
            raise _NoMatch("interval arity")
        bounds = tuple("inf" if b == "inf" else "-inf" if b == "-inf"
                       else ctx.number(b) for b in ipos)
        base, wraps = _peel(pos[0], ctx)
        return base, wraps + (Wrap("truncate", bounds),)

    if head in ("weighted", "logweighted"):
        if kw or len(pos) != 2:
            raise _NoMatch(f"{head} arity")
        base, wraps = _peel(pos[1], ctx)
        return base, wraps + (Wrap(head, (ctx.number(pos[0]),)),)

    if head == "locscale":
        if kw or len(pos) != 3:
            raise _NoMatch("locscale arity")
        base, wraps = _peel(pos[0], ctx)
        return base, wraps + (Wrap("locscale", (ctx.number(pos[1]),
                                               ctx.number(pos[2]))),)

    if head == "pushfwd":
        if kw or len(pos) != 2:
            raise _NoMatch("pushfwd arity")
        wrap = _forward(pos[0], ctx)
        base, wraps = _peel(pos[1], ctx)
        return base, wraps + (wrap,)

    if head in _BASE_SPECS:
        kind, names = _BASE_SPECS[head]
        return Base(kind, _ctor_params(head, names, pos, kw, ctx)), ()

    if head == "LogNormal":
        # §08: "LogNormal(mu, sigma) is equivalent to pushfwd(exp, Normal(mu, sigma))".
        params = _ctor_params(head, ("mu", "sigma"), pos, kw, ctx)
        return Base("normal", params), (Wrap("pushfwd", ("exp",)),)

    if head == "Exponential":
        # §08: `Exponential(rate)` has density `rate * exp(-rate*x)`, which is
        # `Gamma(shape = 1, rate = rate)`'s density exactly.
        (rate,) = _ctor_params(head, ("rate",), pos, kw, ctx)
        return Base("gamma", (1.0, rate)), ()

    raise _NoMatch(f"unsupported construct {head!r}")


def _ctor_params(head: str, names: tuple[str, ...], pos: list[str],
                 kw: dict[str, str], ctx: _Ctx) -> tuple:
    """Constructor arguments in §08's positional order. Keywords must be exactly
    §08's names; a mix of positional and keyword, or any extra key, is refused.

    A parameter in `_VECTOR_PARAMS` is resolved as a vector, so the result is a
    tuple of scalars with a nested tuple in that slot — exactly `space.Base.params`'
    shape.
    """
    def one(name: str, tok: str):
        return ctx.vector(tok) if (head, name) in _VECTOR_PARAMS else ctx.number(tok)

    if pos and kw:
        raise _NoMatch(f"{head}: mixed positional and keyword arguments")
    if kw:
        if set(kw) != set(names):
            raise _NoMatch(f"{head}: expected parameters {names}, got {sorted(kw)}")
        return tuple(one(n, kw[n]) for n in names)
    if len(pos) != len(names):
        raise _NoMatch(f"{head}: expected {len(names)} arguments, got {len(pos)}")
    return tuple(one(n, p) for n, p in zip(names, pos))


# --------------------------------------------------------------------------
# The multi-field joint recognizer -> LinearGaussianProbe
# --------------------------------------------------------------------------

def _joint_shaped(expr: str, ctx: _Ctx) -> bool:
    """Whether `expr` is a MULTI-FIELD joint/record law, i.e. this recognizer's
    business rather than `_peel`'s.

    Dispatch is on the construct, not on "did `_peel` fail". Falling back on
    `_peel`'s failure would report `_peel`'s reason for a case that is plainly a
    `joint` — the matcher is fail-closed to keep its reasons truthful, and a
    misattributed reason is the same defect in a quieter form.
    """
    call = _as_call(expr.strip())
    if not call:
        return False
    head, argstr = call
    if head in ("joint", "iid"):
        return True
    if head == "lawof":
        pos, kw = _args(argstr)
        if kw or len(pos) != 1:
            return False
        rec = _as_call(pos[0])
        if rec and rec[0] == "record":
            rpos, rkw = _args(rec[1])
            return not rpos and len(rkw) >= 2
    return False


def _normal_draw(name: str, ctx: _Ctx, nodes: dict[str, NormalNode]) -> None:
    """Resolve `name` to a `NormalNode` and record it, recursing on its parent.

    Accepts exactly `name = draw(Normal(mu = <parent-or-literal>, sigma = <literal>))`
    — §08's own keyword names, matched strictly for `_BASE_SPECS`' reason. Anything
    else is `_NoMatch`: a non-`Normal` draw is not linear-Gaussian, a stochastic
    `sigma` is not either, and an affine `mu` (`2.0 * z`) is a different model whose
    loading `NormalNode` cannot express. Each of those has a defensible law and none
    of them has THIS law, so guessing would validate the oracle against the wrong
    value — the one failure this module is built to avoid.
    """
    if name in nodes:
        return
    expr = ctx.bindings.get(name)
    if expr is None:
        raise _NoMatch(f"joint component {name!r} is not a binding")
    call = _as_call(expr)
    if not call or call[0] != "draw":
        raise _NoMatch(f"joint component {name!r} is not a draw: {expr!r}")
    dpos, dkw = _args(call[1])
    if dkw or len(dpos) != 1:
        raise _NoMatch("draw arity")
    ctor = _as_call(dpos[0])
    if not ctor or ctor[0] != "Normal":
        raise _NoMatch(f"joint component {name!r} is not a Normal draw: {dpos[0]!r}")
    cpos, ckw = _args(ctor[1])
    if cpos or set(ckw) != {"mu", "sigma"}:
        raise _NoMatch(f"{name}: Normal must be spelled Normal(mu = ..., sigma = ...)")
    sigma = _literal(ckw["sigma"])
    if sigma is None:
        raise _NoMatch(f"{name}: a non-literal sigma is not linear-Gaussian")
    mu_lit = _literal(ckw["mu"])
    if mu_lit is not None:
        nodes[name] = NormalNode(name, None, mu_lit, sigma)
        return
    parent = ckw["mu"].strip()
    if not re.fullmatch(_IDENT, parent):
        raise _NoMatch(f"{name}: mu {ckw['mu']!r} is neither a literal nor a node")
    _normal_draw(parent, ctx, nodes)
    # A child's own `mu` is the parent alone: `NormalNode.mu` is an OFFSET here, and
    # `Normal(mu = z, ...)` adds none.
    nodes[name] = NormalNode(name, parent, 0.0, sigma)


def _lawof_arg(expr: str) -> str:
    """`lawof(x)` -> `"x"`. Anything else is not a reified component."""
    call = _as_call(expr.strip())
    if not call or call[0] != "lawof":
        raise _NoMatch(f"joint component is not a lawof: {expr!r}")
    pos, kw = _args(call[1])
    if kw or len(pos) != 1 or not re.fullmatch(_IDENT, pos[0].strip()):
        raise _NoMatch(f"lawof argument is not a plain name: {expr!r}")
    return pos[0].strip()


def _joint_measure(expr: str, ctx: _Ctx,
                   ) -> tuple[dict[str, NormalNode], tuple[str, ...], str, bool]:
    """`(nodes, fields, composition, point_is_record)` for a multi-field joint.

    The three spellings §06 "Equivalent record law" declares equivalent, plus `iid`:

    * `lawof(record(a = x, b = y))` and `joint(a = lawof(x), b = lawof(y))` — a
      RECORD variate, so the query point is a record;
    * `joint(lawof(x), lawof(y))` — §06 combines positional component variates via
      `cat`, so the variate is a vector and the point is an array;
    * `iid(lawof(x), n)` — §06's product measure, so `composition` is `"iid"` and the
      variate is an array.

    A `joint` mixing reified components with constructors is refused: §06 makes a
    constructor share no stochastic node, so the law is a product of one traced
    marginal and one fresh measure, and `linear_gaussian_logpdf` has no composition
    for that mixture. Nothing in the corpus spells it; refusing is how it stays
    that way rather than being answered wrongly.
    """
    call = _as_call(expr.strip())
    head, argstr = call
    pos, kw = _args(argstr)
    nodes: dict[str, NormalNode] = {}

    if head == "lawof":
        rec = _as_call(pos[0])
        rpos, rkw = _args(rec[1])
        names = [_name_of(v) for v in rkw.values()]
        for n in names:
            _normal_draw(n, ctx, nodes)
        return nodes, tuple(names), "traced", True

    if head == "iid":
        if kw or len(pos) != 2:
            raise _NoMatch("iid arity")
        size = _literal(pos[1])
        if size is None or size != int(size) or int(size) < 2:
            raise _NoMatch(f"iid size is not an integer >= 2: {pos[1]!r}")
        name = _lawof_arg(pos[0])
        _normal_draw(name, ctx, nodes)
        return nodes, (name,) * int(size), "iid", False

    # `joint`, keyword or positional.
    if pos and kw:
        raise _NoMatch("joint: mixed positional and keyword components")
    components = list(kw.values()) if kw else pos
    if len(components) < 2:
        raise _NoMatch("joint with fewer than two components")
    names = [_lawof_arg(c) for c in components]
    for n in names:
        _normal_draw(n, ctx, nodes)
    return nodes, tuple(names), "traced", bool(kw)


def _name_of(tok: str) -> str:
    tok = tok.strip()
    if not re.fullmatch(_IDENT, tok):
        raise _NoMatch(f"record field value is not a plain name: {tok!r}")
    return tok


def _joint_point(expr: str, ctx: _Ctx, arity: int, is_record: bool,
                 ) -> tuple[float, ...]:
    """The query point of a multi-field joint, in field order.

    A record point's fields are taken IN THE ORDER WRITTEN, matching the order
    `_joint_measure` reads the components — both come from `_args`, which preserves
    source order. A record whose field order differed from the measure's would bind
    a cell to the wrong field, so the arity check below is not the only guard that
    matters: `test_the_curated_matcher_maps_a_known_case_to_the_structure_we_expect`
    pins the mapping against the model text.
    """
    rec = _as_call(expr.strip())
    if is_record:
        if not rec or rec[0] != "record":
            raise _NoMatch(f"a record variate needs a record point: {expr!r}")
        rpos, rkw = _args(rec[1])
        if rpos:
            raise _NoMatch("record point mixes positional and keyword fields")
        cells = [ctx.number(v) for v in rkw.values()]
    else:
        tok = expr.strip()
        if not (tok.startswith("[") and tok.endswith("]")):
            raise _NoMatch(f"a vector variate needs an array point: {expr!r}")
        cells = list(ctx.vector(tok))
    if len(cells) != arity:
        raise _NoMatch(f"point has {len(cells)} cells for {arity} components")
    return tuple(cells)


def _query_point(expr: str, ctx: _Ctx) -> float | tuple[float, ...]:
    """The second argument of `logdensityof`, as a scalar or vector point.

    A vector is returned as a tuple, matching `space.Probe.point`.
    """
    rec = _as_call(expr)
    if rec and rec[0] == "record":
        rpos, rkw = _args(rec[1])
        if rpos or len(rkw) != 1:
            raise _NoMatch("query point is not a single-field record")
        expr = next(iter(rkw.values()))
    if expr.strip().startswith("["):
        return ctx.vector(expr)
    return ctx.number(expr)


# --------------------------------------------------------------------------
# Directory loading
# --------------------------------------------------------------------------

def _bindings(text: str) -> tuple[dict[str, str], dict[str, str], set[str],
                                 list[tuple[str, str]]]:
    """Split model source into bindings, bijection aliases, `elementof` params,
    and `logdensityof` queries. A line that is not `name = expr` fails the
    directory outright — `~` sugar, `.~`, and multi-line constructs are all
    outside what this matcher claims to understand."""
    bindings: dict[str, str] = {}
    aliases: dict[str, str] = {}
    params: set[str] = set()
    queries: list[tuple[str, str]] = []
    # §05: `#` starts a line comment and `###` alone on a line opens a block
    # comment; doc-comments are "lexically symmetric to plain comments
    # (`%` <-> `#`, `%%%` <-> `###`)", and a block opener may carry a markup tag
    # with no space after the marker (`%%%markdown`). `%` is comment-only in
    # FlatPPL — §07 spells modulo as the function `mod`, so there is no `%`
    # operator a strip could corrupt.
    #
    # Stripping only `#` made 14 `examples/*` dirs report
    # `unparsed line: '%%%'` instead of the substantive construct that actually
    # defeats the matcher: fail-closed, but misleading about why.
    in_block: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if in_block is not None:
            if line == in_block:
                in_block = None
            continue
        opener = next((m for m in ("###", "%%%") if line.startswith(m)), None)
        if opener:
            in_block = opener
            continue
        for marker in ("#", "%"):
            line = line.split(marker, 1)[0].strip()
        if not line:
            continue
        m = re.match(rf"^({_IDENT})\s*=\s*(.+)$", line)
        if not m:
            raise _NoMatch(f"unparsed line: {line!r}")
        name, expr = m.group(1), m.group(2).strip()
        call = _as_call(expr)
        if call and call[0] == "elementof":
            params.add(name)
        elif call and call[0] == "bijection":
            bpos, bkw = _args(call[1])
            if bkw or len(bpos) != 3:
                raise _NoMatch("bijection arity")
            aliases[name] = bpos[0]
        elif call and call[0] == "logdensityof":
            queries.append((name, call[1]))
        bindings[name] = expr
    return bindings, aliases, params, queries


def _case(directory: Path, text: str, binding: str,
          point: dict[str, float] | None) -> Probe | LinearGaussianProbe:
    bindings, aliases, params, queries = _bindings(text)
    matching = [q for q in queries if q[0] == binding]
    if not matching:
        raise _NoMatch(f"no `logdensityof` bound to {binding!r} "
                       "(a `densityof` query is not one)")
    if len(matching) != 1:
        raise _NoMatch(f"{len(matching)} queries bound to {binding!r}")
    qargs = _split_top(matching[0][1])
    if len(qargs) != 2:
        raise _NoMatch("logdensityof arity")
    ctx = _Ctx(bindings, aliases, params, point)

    if _joint_shaped(qargs[0], ctx):
        nodes, fields, composition, is_record = _joint_measure(qargs[0], ctx)
        return LinearGaussianProbe(
            id=directory.name, nodes=tuple(nodes.values()), fields=fields,
            composition=composition,
            point=_joint_point(qargs[1], ctx, len(fields), is_record))

    base, wraps = _peel(qargs[0], ctx)
    pt = _query_point(qargs[1], ctx)
    return Probe(id=directory.name, base=base, wraps=wraps, spelling="curated",
                 ordering="single", consumer=False, point=pt)


def _load(directory: Path) -> tuple[list[tuple[str, Probe | LinearGaussianProbe, float]], str | None]:
    """One directory -> its expressible cases, or a reason it has none."""
    meta = json.loads((directory / "test.json").read_text())
    if meta.get("test_type") != "logdensity":
        return [], None                       # not this gate's business at all
    name = f"{directory.parent.name}/{directory.name}"
    binding = meta.get("binding", "lp")

    try:
        if "points" in meta:
            model, query = directory / "model.flatppl", directory / "query.flatppl"
            if not (model.exists() and query.exists()):
                raise _NoMatch("parameterized case without model.flatppl + query.flatppl")
            text = model.read_text() + "\n" + query.read_text()
            expected = meta["expected"]
            if not isinstance(expected, list) or len(expected) != len(meta["points"]):
                raise _NoMatch("`expected` does not align with `points`")
            out = []
            for i, pt in enumerate(meta["points"]):
                probe = _case(directory, text, binding, pt)
                out.append((f"{name}#{i}", probe, float(expected[i])))
            return out, None

        model_name = meta.get("model")
        if not model_name:
            raise _NoMatch("no `model` in test.json")
        expected = meta["expected"]
        if isinstance(expected, str):
            expected = float(expected)        # "-inf" cannot round-trip through JSON
        if not isinstance(expected, (int, float)):
            raise _NoMatch(f"non-scalar expected: {type(expected).__name__}")
        if isinstance(expected, float) and expected != expected:
            # A NaN `expected` is a REFUSAL fixture's sentinel, not a value: it is
            # chosen precisely because "NaN never matches" (see
            # corpora/fragment/joint_singular_refusal/test.py), so the dir freezes
            # no number for this gate to reproduce. Admitting it would add a case
            # that validates nothing AND could report agreement the day the oracle
            # itself returned NaN, since a NaN-vs-NaN comparison is never equal.
            raise _NoMatch("expected is NaN: a refusal fixture freezes no value to "
                           "reproduce")
        probe = _case(directory, (directory / model_name).read_text(), binding, None)
        return [(name, probe, float(expected))], None
    except _NoMatch as e:
        return [], f"{name}: {e}"


def _dirs() -> list[Path]:
    return sorted(p.parent for p in (REPO / "corpora").glob("*/*/test.json"))


def curated_probes() -> list[tuple[str, Probe | LinearGaussianProbe, float]]:
    """Every curated `logdensity` case expressible as a Probe, with its FROZEN
    expected value carried through unchanged (never recomputed here)."""
    out: list[tuple[str, Probe | LinearGaussianProbe, float]] = []
    for d in _dirs():
        cases, _ = _load(d)
        out.extend(cases)
    return out


def unmatched_cases() -> list[str]:
    """Curated `logdensity` cases the matcher could NOT express, with the reason.

    Reported rather than dropped: an unvalidated region of the oracle has to be
    visible, and the honest way to say "this case does not validate anything" is
    to name it.
    """
    reasons = []
    for d in _dirs():
        _, why = _load(d)
        if why:
            reasons.append(why)
    return reasons


if __name__ == "__main__":                       # pragma: no cover - debugging aid
    for _name, _probe, _expected in curated_probes():
        print(f"{_name:34} {_probe.base} "
              f"{[(w.kind, w.args) for w in _probe.wraps]} "
              f"@ {_probe.point} -> {_expected}")
    for _why in unmatched_cases():
        print("UNMATCHED", _why)
