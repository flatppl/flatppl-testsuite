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

import pytest

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


# --- golden conversion, over the WHOLE hs3 roster -------------------------------
#
# The unified `convert` runner converts in memory and never compares to the
# committed golden, so a converter change that alters the emitted FlatPPL while
# preserving the numbers goes unnoticed -- exactly the RHS-expansion class of
# regression the structural tests above exist for. The legacy suite compared every
# committed golden (`test_conversions.py`, `test_fixture_converted_flatppl`); this
# restores that over all 8 dirs, not just rf101.
#
# Two dir shapes:
#   fixtures/<n>/     hs3.json          -> model.flatppl
#   conversions/<n>/  <n>.hs3.json      -> <n>.flatppl, whose tail after a
#                                          `# === scoring ===` marker is
#                                          HAND-WRITTEN (the scoring bindings the
#                                          converter does not emit), so only the
#                                          part before the marker is compared.
_SCORING_MARKER = "# === scoring ==="


def _golden_cases() -> list[tuple[str, Path, Path]]:
    cases = []
    for d in sorted((_CORPORA / "hs3" / "fixtures").iterdir()):
        if (d / "hs3.json").exists() and (d / "model.flatppl").exists():
            cases.append((f"fixtures/{d.name}", d / "hs3.json", d / "model.flatppl"))
    for d in sorted((_CORPORA / "hs3" / "conversions").iterdir()):
        hs3, gold = d / f"{d.name}.hs3.json", d / f"{d.name}.flatppl"
        if hs3.exists() and gold.exists():
            cases.append((f"conversions/{d.name}", hs3, gold))
    return cases


_GOLDEN = _golden_cases()


def test_the_golden_roster_is_complete():
    """All 12 hs3 dirs must be covered; a silently-shrinking list defeats this.

    9 fixtures + 3 conversions. The four rf30x conditional fixtures joined in
    the same change that vendored them: their `model.flatppl` is compared
    against live converter output here, which is the ONE place a fixture dir's
    golden is asserted -- the `fixture` arm of the runner reads only
    `hs3.json`, so nothing else would notice a stale golden.
    """
    assert len(_GOLDEN) == 12, f"expected 12 golden cases, found {[c[0] for c in _GOLDEN]}"


@pytest.mark.parametrize("name,hs3,gold", _GOLDEN, ids=[c[0] for c in _GOLDEN])
def test_converter_still_reproduces_the_committed_golden(name, hs3: Path, gold: Path):
    got = _strip_generated_header(convert(hs3))
    want = _strip_generated_header(gold.read_text())
    # A conversions golden carries a hand-written scoring tail the converter
    # never emits; compare only the converted part.
    if _SCORING_MARKER in want:
        want = want.split(_SCORING_MARKER)[0]
    assert got.strip() == want.strip(), (
        f"{name}: converter output no longer matches the committed golden; "
        "if the change is intended, re-pin the golden"
    )


def _strip_generated_header(src: str) -> str:
    """Drop a leading `# AUTOMATICALLY GENERATED` provenance comment.

    `convert()` emits that header; the committed goldens do not carry it. The
    header is provenance, not semantics, so comparing modulo it keeps the check
    about the emitted MODEL -- verified: for rf101 the two agree on all 2016
    remaining lines and differ only by that header."""
    lines = src.splitlines()
    while lines and (
        not lines[0].strip() or lines[0].lstrip().startswith("# AUTOMATICALLY GENERATED")
    ):
        lines.pop(0)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# `assemble` consults the prenormalized notion PER FACTOR inside a joint, and
# binds the observation by the converter's observable NAME rather than by the
# caller's column position. Both are structural: on the vendored fixtures the
# two binding routes agree, and the mis-wrapped factor showed up only as a
# determiniser refusal, so neither regression would move a number.
# ---------------------------------------------------------------------------

# A joint mixing a raw factor with an already-normalize-headed one -- rf305's
# shape, reduced to the parts `assemble` reads.
_MIXED_JOINT_SRC = (
    "sigma = elementof(posreals)\n"
    "obs_domain = cartprod(y = interval(-5.0, 5.0), x = interval(-5.0, 5.0))\n"
    "model = joint(y = gaussy, x = gaussx)\n"
    "gaussy = Normal(mu = 0.0, sigma = 3.0)\n"
    "gaussx = normalize(logweighted((y, x) -> 0.0, Lebesgue(support = obs_domain)))\n"
    'd = table(y = [1.0], x = [2.0])\n'
)


def test_a_normalize_headed_joint_factor_is_not_re_wrapped():
    """The whole point of the per-factor check.

    Wrapping an already-normalized factor makes a `normalize` node the base of
    a `truncate`, which the determiniser refuses for want of a closed-form Z
    over a multivariate base. Consulting the notion once for the head binding
    cannot see this: the head is a `joint(...)`, which is not normalize-headed.
    """
    src, binding = assemble(_MIXED_JOINT_SRC, "model", "d", "y",
                            {"x", "y"})
    assert binding == "__L__"
    assert "truncate(gaussx" not in src, (
        "the normalize-headed factor was re-wrapped; `assemble` is consulting "
        "the prenormalized notion for the product rather than per factor"
    )
    # ... while the RAW factor still gets its range normalization.
    assert "normalize(truncate(gaussy, interval(-5.0, 5.0)))" in src
    # Each factor is observed on its own axis, by label.
    assert 'get(d, "y")' in src and 'get(d, "x")' in src


def test_the_raw_joint_factors_are_still_wrapped():
    """A joint of two raw dists must not change -- rf304_uncorrprod's shape."""
    src_in = (
        "obs_domain = cartprod(x = interval(-5.0, 5.0), y = interval(-5.0, 5.0))\n"
        "gaussxy = joint(x = gx, y = gy)\n"
        "gx = Normal(mu = 0.0, sigma = 1.0)\n"
        "gy = Normal(mu = 0.0, sigma = 2.0)\n"
        "d = table(x = [1.0], y = [2.0])\n"
    )
    src, _ = assemble(src_in, "gaussxy", "d", "x", {"x", "y"})
    assert "normalize(truncate(gx, interval(-5.0, 5.0)))" in src
    assert "normalize(truncate(gy, interval(-5.0, 5.0)))" in src


def test_the_observation_is_bound_by_the_declared_observable_not_the_column():
    """A `% observable:` annotation wins over the caller's column argument.

    Position is not identity. Upstream reordered the rf30x datasets' axes after
    their ROOT vectors were frozen, so a positional pick follows the reordering
    silently; the converter's own annotation does not.
    """
    src_in = (
        "% observable: x\n"
        "gauss = Normal(mu = 0.0, sigma = 1.0)\n"
        "obs_domain = cartprod(x = interval(-5.0, 5.0))\n"
        'd = table(y = [9.0], x = [1.0])\n'
    )
    # The caller passes the WRONG column, as `data_columns(...)[0]` would after
    # an axis reorder. The annotation must override it.
    src, _ = assemble(src_in, "gauss", "d", "y", {"x"})
    assert 'get(d, "x")' in src
    assert 'get(d, "y")' not in src


def test_the_callers_column_stands_when_no_observable_is_declared():
    """No annotation for this pdf -> behaviour is exactly what it was."""
    src_in = (
        "model = normalize(superpose(weighted(f, gx), weighted(1.0 - f, px)))\n"
        'd = table(x = [1.0])\n'
    )
    src, _ = assemble(src_in, "model", "d", "x", {"x"}, prenormalized=True)
    assert 'get(d, "x")' in src


def test_the_prenormalized_predicate_has_one_definition():
    """`suites.hs3_import` keeps the old private name as a delegate.

    Two copies of this predicate would drift, and the joint branch now depends
    on it, so the importer owns it and the old import site forwards.
    """
    from flatppl_testsuite.formats.hs3.importer import binding_is_prenormalized
    from flatppl_testsuite.suites.hs3_import import _binding_is_prenormalized

    src = "m = normalize(weighted(f, g))\n"
    assert binding_is_prenormalized(src, "m") is True
    assert _binding_is_prenormalized(src, "m") is True
    assert binding_is_prenormalized("g = Normal(mu = 0.0, sigma = 1.0)\n", "g") is False
    assert _binding_is_prenormalized("g = Normal(mu = 0.0, sigma = 1.0)\n", "g") is False
