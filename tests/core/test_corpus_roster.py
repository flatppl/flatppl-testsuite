"""The corpus inventory is pinned, so a test directory cannot vanish silently.

`tests/test_unified.py` collects by `discover_test_dirs`, i.e. `rglob("test.json")`.
That is convenient -- dropping in a directory is the whole point of the layout --
but it means DELETING one, renaming it, or removing its `test.json` reduces the
collected count with **no failing test**. Coverage silently shrinks and the suite
still reports green.

The legacy per-corpus gates guarded exactly this with literal roster assertions
(`test_all_fragments_are_gated`, `test_all_posteriors_are_gated`,
`test_all_examples_are_gated`, `test_all_sample_models_are_gated`,
`test_conversions_are_gated`, and the `cov_y1_y2` structural check). Those were
deleted with the gates; this module is their replacement, in one place.

Updating these literals is intended when coverage genuinely changes -- adding a
test dir should be a deliberate one-line edit here, not an invisible drift.
"""
from __future__ import annotations

import json
from pathlib import Path

_CORPORA = Path(__file__).resolve().parents[2] / "corpora"

# corpus -> number of test directories it must contain.
EXPECTED_COUNTS = {
    "bayesian_inference": 5,
    "coverage": 12,
    "examples": 15,
    "fragment": 21,
    "hs3": 8,
    "sample": 1,
    "stablehlo": 25,
    "stablehlo-gradient": 18,
    "stablehlo-sample": 18,
}
EXPECTED_TOTAL = 123

# corpus -> the engine set EVERY dir in it must declare.
#
# Pinning directory names is not enough: dropping `"stablehlo"` from the 14
# examples dirs removes 14 StableHLO cases with a fully green run, since the
# harness parametrizes over whatever `engines` each dir happens to declare. That
# is the same silent coverage loss the examples corpus already suffered once when
# its StableHLO gate was retired -- so the engine list is pinned too, not just
# the roster.
EXPECTED_ENGINES = {
    "bayesian_inference": {"det-js"},
    "coverage": {"det-js"},
    "examples": {"det-js", "stablehlo"},
    "fragment": {"det-js"},
    "hs3": {"det-js"},
    "sample": {"det-js"},
    "stablehlo": {"stablehlo"},
    "stablehlo-gradient": {"stablehlo"},
    "stablehlo-sample": {"stablehlo"},
}
# Total (dir, engine) pairs the harness must collect -- the number that actually
# determines how many cases run.
EXPECTED_CASES = 138

# The rosters whose individual membership the legacy gates pinned by name.
EXPECTED_EXAMPLES = {
    "ex_bayesian_inference_1", "ex_bayesian_inference_2", "ex_best_estimation",
    "ex_capture_recapture", "ex_dissimilar_mixture", "ex_eight_schools",
    "ex_gamma_reparam", "ex_hierarchical_logistic", "ex_linear_regression",
    "ex_partial_pooling", "ex_poisson_glm_link", "ex_poisson_model",
    "ex_rasch_1pl", "ex_signal_background_counting", "ex_zero_inflated_binomial",
}
EXPECTED_HS3 = {
    "conversions/gaussian", "conversions/histfactory", "conversions/product",
    "fixtures/rf101_basics", "fixtures/rf103_interprfuncs", "fixtures/rf203_ranges",
    "fixtures/rf207_comptools", "fixtures/rf304_uncorrprod",
}
# Examples deliberately NOT given a test dir (recorded when the legacy
# manifest.json that listed them was deleted).
EXCLUDED_EXAMPLES = {
    "minimal", "aggregates", "bayesian_inference_common",
    "bayesian_inference_priors", "bayesian_inference_3", "bayesian_inference_4",
}


def _dirs_by_corpus() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for f in _CORPORA.rglob("test.json"):
        rel = f.parent.relative_to(_CORPORA)
        out.setdefault(rel.parts[0], set()).add(str(Path(*rel.parts[1:])))
    return out


def test_every_corpus_has_its_expected_number_of_test_dirs():
    actual = {k: len(v) for k, v in _dirs_by_corpus().items()}
    assert actual == EXPECTED_COUNTS, (
        "corpus inventory changed. If this is intentional, update EXPECTED_COUNTS "
        f"(and EXPECTED_TOTAL) in this file.\n  expected: {EXPECTED_COUNTS}\n"
        f"  actual:   {actual}"
    )


def test_total_test_dir_count():
    total = sum(len(v) for v in _dirs_by_corpus().values())
    assert total == EXPECTED_TOTAL, (
        f"expected {EXPECTED_TOTAL} test dirs, found {total}"
    )


def test_examples_roster_by_name():
    assert _dirs_by_corpus().get("examples", set()) == EXPECTED_EXAMPLES


def test_hs3_roster_by_name():
    assert _dirs_by_corpus().get("hs3", set()) == EXPECTED_HS3


def test_excluded_examples_have_no_test_dir():
    """The 6 deliberately-excluded examples must not acquire one silently."""
    present = _dirs_by_corpus().get("examples", set())
    leaked = sorted(EXCLUDED_EXAMPLES & {p.removeprefix("ex_") for p in present})
    assert not leaked, f"excluded example(s) gained a test dir: {leaked}"


def test_sample_corpus_still_pins_the_covariance_check():
    """`cov_y1_y2` is the check the sample corpus exists for -- a shared-ancestor
    hierarchical model induces covariance between its leaves, and 100.0 is the
    closed-form value. The legacy suite pinned its id, fields and expected value
    structurally so the check could not be quietly dropped or renamed."""
    body = json.loads((_CORPORA / "sample" / "hier_normal" / "test.json").read_text())
    checks = {c["id"]: c for c in body["checks"]}
    assert "cov_y1_y2" in checks, f"cov_y1_y2 check gone (have: {sorted(checks)})"
    cov = checks["cov_y1_y2"]
    assert cov.get("fields") == ["y1", "y2"], f"unexpected fields: {cov.get('fields')}"
    assert float(cov["expected"]) == 100.0, f"unexpected expected: {cov['expected']}"


def _engines_by_dir() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for f in _CORPORA.rglob("test.json"):
        rel = f.parent.relative_to(_CORPORA)
        out[str(rel)] = json.loads(f.read_text())["engines"]
    return out


def test_every_dir_declares_the_expected_engines():
    """A dir quietly dropping an engine removes its cases with a green run."""
    wrong = {}
    for rel, engines in _engines_by_dir().items():
        corpus = rel.split("/")[0]
        want = EXPECTED_ENGINES.get(corpus)
        if want is None:
            wrong[rel] = f"corpus {corpus!r} not in EXPECTED_ENGINES"
        elif set(engines) != want:
            wrong[rel] = f"declares {sorted(engines)}, expected {sorted(want)}"
    assert not wrong, (
        "engine coverage changed. If intentional, update EXPECTED_ENGINES "
        f"(and EXPECTED_CASES):\n" + "\n".join(f"  {k}: {v}" for k, v in sorted(wrong.items()))
    )


def test_total_collected_case_count():
    """(dir, engine) pairs -- the number that decides how many cases actually run."""
    cases = sum(len(v) for v in _engines_by_dir().values())
    assert cases == EXPECTED_CASES, (
        f"expected {EXPECTED_CASES} (dir, engine) cases, found {cases}"
    )
