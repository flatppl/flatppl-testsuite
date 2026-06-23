"""assemble references the converter's pdf binding BY NAME (the engine accepts
relabel'd measures in iid/truncate/normalize since PR #38), so no RHS parsing
is required."""

from flatppl_testsuite.formats.hs3.importer import assemble, convert
from flatppl_testsuite.suites.hs3_import import HS3_CORPUS

RF101 = HS3_CORPUS / "fixtures" / "rf101_basics"

# Minimal converted source that mimics the rf101 converter output (a relabel'd
# Normal).  Used to keep unit tests self-contained and fast.
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


def test_assemble_references_pdf_by_name_with_interval():
    scoreable, binding = assemble(_RF101_LIKE_SRC, "gauss", "gaussData", "x", {"x"})
    m_line = next(ln for ln in scoreable.splitlines() if ln.startswith("__M__"))
    # The pdf name "gauss" must appear as the truncate argument — not expanded
    # to relabel(...) or Normal(...).
    assert "gauss" in m_line
    assert "normalize(truncate(gauss," in m_line or "normalize(truncate(gauss ," in m_line
    assert "interval(" in m_line
    # Parens are balanced.
    assert m_line.count("(") == m_line.count(")")
    assert binding == "__L__"
    assert "__L__ = likelihoodof(iid(__M__" in scoreable


def test_assemble_no_interval_emits_bare_pdf():
    src_no_range = (
        "mean = elementof(reals)\n"
        "sigma = elementof(posreals)\n"
        'gauss = relabel(Normal(mu = mean, sigma = sigma), ["x"])\n'
        "gaussData = table(x = [1.0, 2.0])\n"
    )
    scoreable, binding = assemble(src_no_range, "gauss", "gaussData", "x", {"x"})
    m_line = next(ln for ln in scoreable.splitlines() if ln.startswith("__M__"))
    # No interval in scope → bare reference.
    assert m_line == "__M__ = gauss"
    assert binding == "__L__"


def test_assemble_rf101_by_name_uses_real_converter():
    """The real converter output also produces a by-name measure reference."""
    src = convert(RF101 / "hs3.json")
    scoreable, binding = assemble(src, "gauss", "gaussData", "x", {"x"})
    m_line = next(ln for ln in scoreable.splitlines() if ln.startswith("__M__"))
    # The binding name "gauss" must appear literally as the truncate argument.
    assert "gauss" in m_line
    assert "relabel(" not in m_line
    assert "Normal(" not in m_line
    assert m_line.count("(") == m_line.count(")")
    assert binding == "__L__"
