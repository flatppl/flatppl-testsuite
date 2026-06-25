from flatppl_testsuite.formats.hs3.importer import _strip_provenance


def test_strips_block_doc_comment_fences_and_content():
    src = (
        "model = normalize(superpose(weighted(f, gx), weighted(1.0 - f, px)))\n"
        "%%%\n"
        "HS3 polynomial_dist → normalize(weighted(...))\n"
        "observable: x\n"
        "%%%\n"
        "px = Normal(mu = 1.0, sigma = 1.0)\n"
    )
    out = _strip_provenance(src)
    assert "→" not in out                      # block content gone, no orphaned arrow
    assert "polynomial_dist" not in out
    assert "model = normalize(superpose" in out
    assert "px = Normal(mu = 1.0, sigma = 1.0)" in out


def test_strips_line_comments():
    assert _strip_provenance("% lead\nx = 1\n") == "x = 1"
