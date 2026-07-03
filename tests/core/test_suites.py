"""Suite registry and HS3ImportSuite tests (Task 5).

test_hs3_suite_registered: get_suites() must include an HS3ImportSuite instance.
test_hs3_suite_runs_rf101: HS3ImportSuite.run(selected={"rf101_basics"}) must return a
    passing twice_delta_nll_scan result.
test_fragment_suite_registered: get_suites() must also include a
    FragmentGateSuite instance (the fragment numeric gate, see
    flatppl_testsuite/suites/fragment_gate.py); its own scoring behaviour
    is exercised end-to-end by corpora/fragment/tests/test_fragment_gate.py.
"""

from flatppl_testsuite.suites.base import Suite, get_suites
from flatppl_testsuite.suites.hs3_import import HS3ImportSuite
from flatppl_testsuite.suites.fragment_gate import FragmentGateSuite


def test_hs3_suite_registered():
    suites = get_suites()
    assert any(isinstance(s, HS3ImportSuite) for s in suites)


def test_hs3_suite_runs_rf101():
    suite = HS3ImportSuite()
    results = suite.run(selected={"rf101_basics"})
    nll = [r for r in results if r.check_id == "twice_delta_nll_scan"]
    assert nll and nll[0].status == "passed"


def test_fragment_suite_registered():
    suites = get_suites()
    assert any(isinstance(s, FragmentGateSuite) for s in suites)
