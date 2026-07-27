"""`assemble` must reference the converter's pdf binding BY NAME.

The engine accepts a relabel'd measure inside `iid`/`truncate`/`normalize`, so
`assemble` composes the scoreable module by NAME rather than by parsing and
re-expanding the pdf's right-hand side. That is a real structural contract: a
regression to RHS-expansion would very likely still produce the correct numbers
on the vendored fixtures -- the composed measure is equivalent -- and so would
pass every numeric check while silently reintroducing the RHS parsing the
by-name design removed.

`corpora/hs3/tests/test_assemble_multiline.py` pinned this and was deleted with
the legacy gates, leaving `assemble` with no structural coverage at all. These
are those tests, restored (the converted-golden check is restored alongside them,
since that was lost in the same sweep).
"""
from __future__ import annotations

from pathlib import Path

from flatppl_testsuite.formats.hs3.importer import assemble, convert

_CORPORA = Path(__file__).resolve().parents[2] / "corpora"
RF101 = _CORPORA / "hs3" / "fixtures" / "rf101_basics"

# Minimal converted source mimicking the rf101 converter output (a relabel'd
# Normal), so the structural tests stay self-contained and fast.
_RF101_LIKE_SRC = (
    "mean = elementof(reals)\n"
    "sigma = elementof(posreals)\n"
    'gauss = relabel(Normal(mu = mean, sigma = sigma), ["x"])\n'
    "gaussData = table(x = [1.0, -2.0, 3.0])\n"
    "default_domain = cartprod(\n"
    "  mean = interval(-10.0, 10.0),\n"
    "  sigma = interval(0.1, 10.0),\n"
    "  x = interval(-10.0, 10.0),\n"
    ")\n"
    "default_values = record(x = 0.0, mean = 1.0, sigma = 3.0)\n"
)


def _m_line(scoreable: str) -> str:
    return next(ln for ln in scoreable.splitlines() if ln.startswith("__M__"))


def test_assemble_references_pdf_by_name_with_interval():
    scoreable, binding = assemble(_RF101_LIKE_SRC, "gauss", "gaussData", "x", {"x"})
    m_line = _m_line(scoreable)
    # The pdf NAME must be the truncate argument -- never expanded to the RHS.
    assert "gauss" in m_line
    assert "relabel(" not in m_line, f"pdf RHS was expanded: {m_line}"
    assert "Normal(" not in m_line, f"pdf RHS was expanded: {m_line}"
    assert "normalize(truncate(gauss," in m_line or "normalize(truncate(gauss ," in m_line
    assert "interval(" in m_line
    assert m_line.count("(") == m_line.count(")"), f"unbalanced parens: {m_line}"
    assert binding == "__L__"
    assert "__L__ = likelihoodof(iid(__M__" in scoreable


def test_assemble_no_interval_emits_bare_pdf():
    """With no interval in scope the measure is a bare reference, not a
    degenerate `normalize(truncate(...))` wrapper."""
    src_no_range = (
        "mean = elementof(reals)\n"
        "sigma = elementof(posreals)\n"
        'gauss = relabel(Normal(mu = mean, sigma = sigma), ["x"])\n'
        "gaussData = table(x = [1.0, 2.0])\n"
    )
    scoreable, binding = assemble(src_no_range, "gauss", "gaussData", "x", {"x"})
    assert _m_line(scoreable) == "__M__ = gauss"
    assert binding == "__L__"


def test_assemble_rf101_by_name_uses_real_converter():
    """The real converter output composes by name too, not just the stub above."""
    src = convert(RF101 / "hs3.json")
    m_line = _m_line(assemble(src, "gauss", "gaussData", "x", {"x"})[0])
    assert "gauss" in m_line
    assert "relabel(" not in m_line and "Normal(" not in m_line, (
        f"real converter path expanded the pdf RHS: {m_line}"
    )


def test_converter_still_reproduces_the_committed_golden():
    """`flatppl convert` output must still equal the committed `model.flatppl`.

    The legacy suite compared converter output against every committed golden
    (`test_conversions.py`, `test_fixture_converted_flatppl`); the unified
    `convert` runner converts in memory and never compares to the golden, so a
    converter change that alters the emitted FlatPPL while preserving the
    numbers would go unnoticed. This restores the check for rf101."""
    got = _strip_generated_header(convert(RF101 / "hs3.json"))
    want = _strip_generated_header((RF101 / "model.flatppl").read_text())
    assert got.strip() == want.strip(), (
        "converter output no longer matches the committed model.flatppl golden; "
        "if the change is intended, re-pin the golden"
    )


def _strip_generated_header(src: str) -> str:
    """Drop a leading `# AUTOMATICALLY GENERATED` provenance comment.

    `convert()` emits that header; the committed goldens do not carry it (the
    vendored `model.flatppl` files are the converter's output without it). The
    header is provenance, not semantics, so comparing modulo it is what makes
    the golden check about the emitted MODEL -- verified: for rf101 the two agree
    on all 2016 remaining lines and differ only by that header."""
    lines = src.splitlines()
    while lines and (
        not lines[0].strip() or lines[0].lstrip().startswith("# AUTOMATICALLY GENERATED")
    ):
        lines.pop(0)
    return "\n".join(lines)
