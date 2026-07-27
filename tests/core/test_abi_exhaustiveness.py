"""Every ABI query module in the corpus must list ALL its parameterized params.

The compilation ABI (flatppl-design §12, "Compilation ABI: `inputs` and
`outputs`") makes `inputs` authoritative and exhaustive:

    every parameterized `elementof` binding in the module must appear in it,
    otherwise the declaration is ill-formed -- a parameter with no argument slot.

A test dir's emitted module is `model.flatppl` + `query.flatppl` concatenated
(see `unified/runners/logdensity_stablehlo.py::_concat`), so the rule applies to
the CONCATENATION: a query that introduces its own `t_<field>` params while the
model already declares `elementof` bindings of its own must account for both.

This is a local guard rather than a redundant one: `flatppl stablehlo` does not
currently enforce exhaustiveness (verified 2026-07-27 -- an unlisted
`elementof` emits a func with one fewer argument and exits 0), so an
ill-formed query here would score correctly and silently until the emitter
starts refusing it.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_CORPORA = Path(__file__).resolve().parents[2] / "corpora"

# `name = elementof(...)` at the start of a line (top-level binding).
_ELEMENTOF = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*elementof\s*\(", re.M)
# `inputs = x` or `inputs = (x, y, ...)`
_INPUTS = re.compile(r"^\s*inputs\s*=\s*(.+?)\s*$", re.M)


def _abi_dirs() -> list[Path]:
    return sorted(p.parent for p in _CORPORA.rglob("query.flatppl"))


def _declared_inputs(query_src: str) -> list[str]:
    m = _INPUTS.search(query_src)
    if m is None:
        return []
    rhs = m.group(1).strip()
    if rhs.startswith("("):
        rhs = rhs[1:rhs.rindex(")")] if ")" in rhs else rhs[1:]
    return [t.strip() for t in rhs.split(",") if t.strip()]


_IDS = [str(d.relative_to(_CORPORA)) for d in _abi_dirs()]


@pytest.mark.parametrize("dir", _abi_dirs(), ids=_IDS)
def test_inputs_lists_every_parameterized_param(dir: Path):
    model = (dir / "model.flatppl").read_text()
    query = (dir / "query.flatppl").read_text()

    params = set(_ELEMENTOF.findall(model)) | set(_ELEMENTOF.findall(query))
    listed = set(_declared_inputs(query))

    unlisted = sorted(params - listed)
    assert not unlisted, (
        f"{dir.name}: parameterized elementof binding(s) {unlisted} are not listed "
        f"in `inputs` ({sorted(listed)}) -- ill-formed per the compilation ABI. "
        "Either list them, or have the query reuse the model's own binding instead "
        "of shadowing it with a duplicate."
    )


def test_the_guard_sees_the_corpus():
    """Guard against the parametrization silently collecting nothing."""
    assert _abi_dirs(), "no query.flatppl found -- this guard is vacuous"
