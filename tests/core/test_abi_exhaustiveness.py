"""No handwritten corpus ABI query leaves an unaccounted `elementof` behind.

This is a deliberate HOUSE RULE, stricter than the spec -- read the next
paragraph before treating a failure here as a conformance bug.

The normative rule (flatppl-design "Determinization" -> "Signature: `inputs` and
`outputs`") is:

    `inputs` must list every `elementof` leaf that an output depends on
    (otherwise the module is ill-formed); an `elementof` that no output reaches
    is eliminated like any other unreached binding.

So an UNREACHED `elementof` is explicitly well-formed and simply eliminated, and
`flatppl stablehlo` accepts such a module (exit 0). This test additionally
forbids it in handwritten fixtures, where a parameterized param that no output
reaches is an authoring slip -- a query that meant to feed it and does not, or a
leftover shadowing duplicate of a binding the model already declares. Catching
that early is worth a rule the spec does not impose; it is not evidence of
non-conformance. Explicit `functionof`/`kernelof` boundary references also
account for their leaves: those parameters belong to the callable rather than
the query ABI. This is an accounting check, not semantic liveness analysis;
it does not establish whether a reified helper is used.

Full vendored model snapshots follow the spec instead. Their reified helpers
can leave unused top-level leaves, and the source hash requires unchanged model
bytes. Apply this house rule to their query additions only; executed corpus
tests still enforce the compiler's reachable-input rule for the whole module.

The reached free case IS enforced by the compiler, and loudly -- an `elementof`
that an output depends on but that `inputs` omits is refused with exit 3
("elementof parameter `x` is not listed in `inputs`"), so for that case this test
is a fast local echo of a check that already exists. What it uniquely covers is
the CONCATENATION: a test dir's emitted module is `model.flatppl` + `query.flatppl`
(see `unified/runners/logdensity_stablehlo.py::_concat`), so a query introducing
its own params while the model declares params of its own has to account for both,
and that combined view is not visible to either file alone.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_CORPORA = Path(__file__).resolve().parents[2] / "corpora"

# `name = elementof(...)` at the start of a line (top-level binding).
_ELEMENTOF = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*elementof\s*\(", re.M)
# `inputs = x` or `inputs = (x, y, ...)`
_INPUTS = re.compile(r"^\s*inputs\s*=\s*(.+?)\s*$", re.M)
# `name = load_data(...)` at the start of a line (top-level binding).
_LOAD_DATA = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*load_data\s*\(", re.M)
# `key = src.field` anywhere -- used to spot a param PINNED to a load_data
# column (`x_data = data.x`), the one feeding route that is not `inputs`.
_FIELD_PIN = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\.[A-Za-z_]"
)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Keep strings and punctuation intact so their contents cannot become arguments.
_TOKENS = re.compile(
    r'\#\#\#[ \t]*\n[\s\S]*?^[ \t]*\#\#\#[ \t]*(?=\n|$)'
    r'|%%%[^\n]*\n[\s\S]*?^[ \t]*%%%[ \t]*(?=\n|$)'
    r'|[\#%][^\n;]*|"(?:\\.|[^"\\])*"|[A-Za-z_][A-Za-z0-9_]*|\S',
    re.M,
)


def _abi_dirs() -> list[Path]:
    return sorted(p.parent for p in _CORPORA.rglob("query.flatppl"))


def _model_path(dir: Path) -> Path:
    """The file the query is concatenated onto. The fragment and
    bayesian_inference corpora name the model after the test id, and their
    StableHLO row names it in the engine block, so the name is read off
    `test.json` rather than assumed to be `model.flatppl`."""
    body = json.loads((dir / "test.json").read_text())
    for source in (body.get("stablehlo") or {}, body):
        name = source.get("model")
        if name:
            return dir / name
    return dir / "model.flatppl"


def _declared_inputs(query_src: str) -> list[str]:
    m = _INPUTS.search(query_src)
    if m is None:
        return []
    rhs = m.group(1).strip()
    if rhs.startswith("("):
        rhs = rhs[1:rhs.rindex(")")] if ")" in rhs else rhs[1:]
    return [t.strip() for t in rhs.split(",") if t.strip()]


def _reified_parameters(source: str) -> set[str]:
    """Account only for explicit bare references at a callable boundary."""
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    tokens = [t for t in _TOKENS.findall(source) if not t.startswith(("#", "%"))]
    bound = set()
    for i, token in enumerate(tokens[:-1]):
        if token not in ("functionof", "kernelof") or tokens[i + 1] != "(":
            continue
        if i and tokens[i - 1] == ".":
            continue
        depth = 0
        arguments = [[]]
        for token in tokens[i + 2:]:
            if token == ")" and depth == 0:
                break
            if token == "," and depth == 0:
                arguments.append([])
                continue
            if token in ("(", "[", "{"):
                depth += 1
            elif token in (")", "]", "}"):
                depth -= 1
            arguments[-1].append(token)
        for argument in arguments[1:]:
            if (len(argument) == 3 and argument[1] == "="
                    and _IDENTIFIER.fullmatch(argument[0])
                    and _IDENTIFIER.fullmatch(argument[2])):
                bound.add(argument[2])
    return bound


_IDS = [str(d.relative_to(_CORPORA)) for d in _abi_dirs()]


@pytest.mark.parametrize("dir", _abi_dirs(), ids=_IDS)
def test_inputs_lists_every_parameterized_param(dir: Path):
    query = (dir / "query.flatppl").read_text()
    model = _model_path(dir).read_text()
    source = json.loads((dir / "test.json").read_text()).get("source", {})

    params = set(_ELEMENTOF.findall(query))
    if not isinstance(source, dict) or "sha256" not in source:
        params.update(_ELEMENTOF.findall(model))
    listed = set(_declared_inputs(query))

    # A param pinned to a load_data column (`x_data = data.x` at the density
    # point, with `data = load_data(...)` in `inputs`) is fed and substituted
    # away, not dead.
    # Scoped to load_data sources so an ordinary shadowing slip still fails.
    load_data_names = set(_LOAD_DATA.findall(query))
    pinned = {
        key
        for key, src in _FIELD_PIN.findall(query)
        if src in load_data_names
    }

    bound = _reified_parameters(model + "\n" + query)
    unlisted = sorted(params - listed - pinned - bound)
    assert not unlisted, (
        f"{dir.name}: parameterized elementof binding(s) {unlisted} are not listed "
        f"in `inputs` ({sorted(listed)}), pinned or explicitly reified. "
        "If an output depends freely on it the module is "
        "ill-formed per the spec; if nothing reaches it the spec would eliminate it, "
        "but this corpus forbids a dead param anyway (see this module's docstring). "
        "Either list it, or have the query reuse the model's own binding instead of "
        "shadowing it with a duplicate."
    )


@pytest.mark.parametrize(("source", "expected"), [
    ("f = functionof(body,\n renamed = original, y = y)", {"original", "y"}),
    ("k = kernelof(Normal(mu=x, sigma=s), location=x)", {"x"}),
    ('f = functionof(body, label="x=x", x=x+1, y=model.y)', set()),
    ('# functionof(body, a=a)\n% kernelof(body, b=b)\n'
     '###\nAn inline ### is not a fence.\nfunctionof(body, c=c)\n###\n'
     '%%%md\nAn inline %%% is not a fence.\nkernelof(body, d=d)\n%%%\n'
     'label="functionof(body, e=e)"\nf=functionof(body, x=x)', {"x"}),
    ("### a line comment; f=functionof(body, x=x)\r"
     "% another comment; k=kernelof(body, y=y)", {"x", "y"}),
    ("f = model.functionof(body, x=x)", set()),
    ("dead = elementof(reals)\nf = functionof(body, x=x)", {"x"}),
])
def test_explicit_reification_boundaries(source: str, expected: set[str]):
    assert _reified_parameters(source) == expected


def test_the_guard_sees_the_corpus():
    """Guard against the parametrization silently collecting nothing."""
    assert _abi_dirs(), "no query.flatppl found -- this guard is vacuous"
