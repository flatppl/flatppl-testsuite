"""Tests for the convert stage: hs3 -> flatppl conversion and error classification."""
import json

import pytest
from flatppl_testsuite.formats.hs3.importer import convert, SkipUnimplemented
from flatppl_testsuite.suites.hs3_import import HS3_CORPUS

FIX = HS3_CORPUS / "fixtures"


def test_convert_gaussian_emits_likelihood():
    src = convert(FIX / "rf101_basics" / "hs3.json")
    assert "Normal(" in src


def test_convert_unimplemented_raises(tmp_path):
    # A genuinely-unmapped HS3 distribution type triggers SkipUnimplemented.
    # (landau_dist is implemented now; fft_conv_pdf — convolution — is not.)
    doc = {"distributions": [
        {"name": "c", "type": "fft_conv_pdf", "x": "obs"}]}
    p = tmp_path / "hs3.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(SkipUnimplemented) as e:
        convert(p)
    assert e.value.hs3_type == "fft_conv_pdf"


def test_convert_landau_emits_hepphys(tmp_path):
    # landau_dist → hepphys.Landau(loc, scale), reading HS3 mean/sigma.
    doc = {
        "distributions": [
            {"name": "lx", "type": "landau_dist",
             "mean": "ml", "sigma": "sl", "x": "obs"}],
        "parameter_points": [
            {"name": "nominal", "entries": [
                {"name": "ml", "value": 0.0},
                {"name": "sl", "value": 1.0}]}],
    }
    p = tmp_path / "hs3.json"
    p.write_text(json.dumps(doc))
    src = convert(p)
    assert "hepphys.Landau(ml, sl)" in src
