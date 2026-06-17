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
    # Synthetic input: an unimplemented HS3 distribution type triggers
    # SkipUnimplemented, independent of any vendored fixture.
    doc = {"distributions": [
        {"name": "c", "type": "chebychev_dist", "x": "obs",
         "coefficients": ["a1", "a2"]}]}
    p = tmp_path / "hs3.json"
    p.write_text(json.dumps(doc))
    with pytest.raises(SkipUnimplemented) as e:
        convert(p)
    assert e.value.hs3_type == "chebychev_dist"
