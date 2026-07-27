"""No corpus ABI query leaves a dead `elementof` behind.

This is a deliberate HOUSE RULE, stricter than the spec -- read the next
paragraph before treating a failure here as a conformance bug.

The normative rule (flatppl-design "Determinization" -> "Signature: `inputs` and
`outputs`") is:

    `inputs` must list every `elementof` leaf that an output depends on
    (otherwise the module is ill-formed); an `elementof` that no output reaches
    is eliminated like any other unreached binding.

So an UNREACHED `elementof` is explicitly well-formed and simply eliminated, and
`flatppl stablehlo` accepts such a module (exit 0). This test additionally
forbids it, because in THIS corpus a parameterized param that no output reaches
is always an authoring slip -- a query that meant to feed it and does not, or a
leftover shadowing duplicate of a binding the model already declares. Catching
that early is worth a rule the spec does not impose; it is not evidence of
non-conformance.

The reached case IS enforced by the emitter, and loudly -- an `elementof` that an
output depends on but that `inputs` omits is refused with exit 3
("elementof parameter `x` is not listed in `inputs`"), so for that case this test
is a fast local echo of a check that already exists. What it uniquely covers is
the CONCATENATION: a test dir's emitted module is `model.flatppl` + `query.flatppl`
(see `unified/runners/logdensity_stablehlo.py::_concat`), so a query introducing
its own params while the model declares params of its own has to account for both,
and that combined view is not visible to either file alone.
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
        f"in `inputs` ({sorted(listed)}). If an output depends on it the module is "
        "ill-formed per the spec; if nothing reaches it the spec would eliminate it, "
        "but this corpus forbids a dead param anyway (see this module's docstring). "
        "Either list it, or have the query reuse the model's own binding instead of "
        "shadowing it with a duplicate."
    )


def test_the_guard_sees_the_corpus():
    """Guard against the parametrization silently collecting nothing."""
    assert _abi_dirs(), "no query.flatppl found -- this guard is vacuous"
