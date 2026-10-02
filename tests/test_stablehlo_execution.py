"""Execute declared ABI types without silently changing input values."""

from pathlib import Path

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
def test_masked_loop_preserves_values_and_adjoints():
    _gate_engine("stablehlo")
    jax, jnp, hlo_call = ex._jax()
    # The adjoint materializes a non-splat 87x2 boolean mask. Its dense
    # encoding must survive the Enzyme/JAX MLIR boundary.
    source = """module {
      func.func @main(%seed: tensor<2xf64>, %factors: tensor<87x2xf64>) -> tensor<2xf64> {
        %zero = stablehlo.constant dense<0> : tensor<i64>
        %one = stablehlo.constant dense<1> : tensor<i64>
        %limit = stablehlo.constant dense<87> : tensor<i64>
        %lengths = stablehlo.constant dense<[43, 87]> : tensor<2xi64>
        %zeros = stablehlo.constant dense<0.0> : tensor<2xf64>
        %ones = stablehlo.constant dense<1.0> : tensor<2xf64>
        %loop:2 = stablehlo.while(%i = %zero, %acc = %seed) : tensor<i64>, tensor<2xf64>
        cond {
          %continue = stablehlo.compare LT, %i, %limit : (tensor<i64>, tensor<i64>) -> tensor<i1>
          stablehlo.return %continue : tensor<i1>
        } do {
          %row = stablehlo.dynamic_slice %factors, %i, %zero, sizes = [1, 2] : (tensor<87x2xf64>, tensor<i64>, tensor<i64>) -> tensor<1x2xf64>
          %factor = stablehlo.reshape %row : (tensor<1x2xf64>) -> tensor<2xf64>
          %stages = stablehlo.broadcast_in_dim %i, dims = [] : (tensor<i64>) -> tensor<2xi64>
          %active = stablehlo.compare LT, %stages, %lengths : (tensor<2xi64>, tensor<2xi64>) -> tensor<2xi1>
          %lhs = stablehlo.select %active, %acc, %zeros : (tensor<2xi1>, tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
          %rhs = stablehlo.select %active, %factor, %ones : (tensor<2xi1>, tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
          %product = stablehlo.multiply %lhs, %rhs : tensor<2xf64>
          %next = stablehlo.select %active, %product, %acc : (tensor<2xi1>, tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
          %next_i = stablehlo.add %i, %one : tensor<i64>
          stablehlo.return %next_i, %next : tensor<i64>, tensor<2xf64>
        }
        return %loop#1 : tensor<2xf64>
      }
    }"""
    weights = jnp.array([0.25, -0.5])

    def score(seed, factors):
        return jnp.sum(weights * hlo_call(seed, factors, source=source)[0])

    seed = jnp.array([2.0, -3.0])
    factors = jnp.ones((87, 2))
    value, (dseed, dfactors) = jax.jit(jax.value_and_grad(score, argnums=(0, 1)))(seed, factors)
    expected = np.where(np.arange(87)[:, None] < [43, 87], np.asarray(weights * seed), 0)
    assert value == 2.0
    np.testing.assert_array_equal(dseed, weights)
    np.testing.assert_array_equal(dfactors, expected)


@pytest.mark.stablehlo_only
def test_continued_poisson_zero_variate_retains_the_rate_derivative():
    _gate_engine("stablehlo")
    directory = Path(__file__).resolve().parents[1] / "corpora/stablehlo/continued_poisson"
    src = ex.emit_concat(directory, "logdensity")
    # At x=0, §09 gives log-density = -rate, including rate=0.
    for rate in (0.0, 1.0):
        assert ex.gradient(src, [0.0, rate], [1]) == pytest.approx([-1.0])


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
