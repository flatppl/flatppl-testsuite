from flatppl_testsuite.suites.hs3_import import _binding_is_prenormalized


def test_normalize_binding_is_prenormalized():
    src = "model = normalize(superpose(weighted(f, gx), weighted(1.0 - f, px)))\n"
    assert _binding_is_prenormalized(src, "model") is True


def test_plain_dist_binding_is_not_prenormalized():
    assert _binding_is_prenormalized("gx = Normal(mu = mu, sigma = 1.0)\n", "gx") is False
