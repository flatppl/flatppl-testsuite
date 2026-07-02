import subprocess, pathlib, pytest
from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.suites.hs3_import import HS3_CORPUS

HERE = pathlib.Path(__file__).parents[1] / "conversions"
MODELS = ["gaussian", "product", "histfactory"]

def _strip_header(text: str) -> str:
    # Drop provenance comment lines (start with %) and blank padding.
    body = [ln for ln in text.splitlines() if not ln.lstrip().startswith("%")]
    return "\n".join(ln for ln in body if ln.strip()) + "\n"


SCORING_MARKER = "# === scoring ==="


def _conversion_part(text: str) -> str:
    # The golden ends with a mechanically-appended scoring section (regen.py);
    # the converter only emits the part above the marker.
    return text.split(SCORING_MARKER, 1)[0]

@pytest.mark.parametrize("model", MODELS)
def test_known_good_conversion(model, tmp_path):
    hs3 = HERE / model / f"{model}.hs3.json"
    expected = _strip_header(_conversion_part((HERE / model / f"{model}.flatppl").read_text()))
    out = tmp_path / f"{model}.flatppl"
    subprocess.run([str(CONFIG.flatppl_bin), "convert", "--from", "hs3",
                    str(hs3), str(out), "--no-header"], check=True)
    assert _strip_header(out.read_text()) == expected


# Each vendored fixture ships its converted FlatPPL (model.flatppl); assert the
# converter still reproduces it (header ignored).
_FIXTURES = sorted(
    p.parent.name
    for p in (HS3_CORPUS / "fixtures").glob("*/model.flatppl")
)

@pytest.mark.parametrize("fixture", _FIXTURES)
def test_fixture_converted_flatppl(fixture, tmp_path):
    fdir = HS3_CORPUS / "fixtures" / fixture
    expected = _strip_header((fdir / "model.flatppl").read_text())
    out = tmp_path / "model.flatppl"
    subprocess.run([str(CONFIG.flatppl_bin), "convert", "--from", "hs3",
                    str(fdir / "hs3.json"), str(out), "--no-header"], check=True)
    assert _strip_header(out.read_text()) == expected
