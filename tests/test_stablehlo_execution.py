"""Execute declared ABI types without silently changing input values."""

from pathlib import Path

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("invalid", [1.5, 2**31])
def test_integer_abi_accepts_integral_spelling_but_rejects_loss(invalid):
    _gate_engine("stablehlo")
    directory = Path(__file__).resolve().parents[1] / "corpora/coverage/typed_abi_inputs"
    src = ex.emit_concat(directory, "logdensity")
    assert ex.value(src, [[1.0, 2.0, 3.0], 1]) == pytest.approx(-2.112085713764618, abs=1e-6)
    with pytest.raises(ValueError, match="integer ABI"):
        ex.value(src, [[invalid, 2, 3], 1])


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("key", [(2**63 + 1, 0), (2**64 - 1, 0)])
def test_full_width_tuple_key_matches_the_same_uint64_array(key):
    _gate_engine("stablehlo")
    directory = Path(__file__).resolve().parents[1] / "corpora/stablehlo-sample/normal"
    src = ex.emit_concat(directory, "sample")
    tuple_result = ex.sample_call(src, key, [0.0, 1.0])
    array_result = ex.sample_call(src, np.asarray(key, dtype=np.uint64), [0.0, 1.0])
    for actual, expected in zip(tuple_result, array_result):
        np.testing.assert_array_equal(actual, expected)
