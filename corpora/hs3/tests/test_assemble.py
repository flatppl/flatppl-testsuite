from flatppl_testsuite.formats.hs3.importer import observations, assemble, convert
from flatppl_testsuite.scoring.flatppl_engine import twice_delta_nll
from flatppl_testsuite.suites.hs3_import import HS3_CORPUS

RF101 = HS3_CORPUS / "fixtures" / "rf101_basics"


def test_observations_count():
    obs = observations(RF101 / "hs3.json", "gaussData")
    assert len(obs) == 2000


def test_rf101_reproduces_frozen_expected(tmp_path):
    src = convert(RF101 / "hs3.json")
    # Observe the converter's embedded `gaussData = table(x = [...])` column.
    scoreable, binding = assemble(src, "gauss", "gaussData", "x", {"x"})
    model = tmp_path / "rf101.flatppl"
    model.write_text(scoreable)
    vec = twice_delta_nll(model, binding, "mean",
                          [-1.0, 0.0, 1.0, 2.0, 3.0], {"mean": 1.0, "sigma": 3.0})
    expected = [888.0456117195517, 224.26202740723056, 0.0,
                213.06114567253644, 856.1426350606562]
    for got, want in zip(vec, expected):
        assert abs(got - want) <= 1e-7 + 1e-8 * abs(want)
