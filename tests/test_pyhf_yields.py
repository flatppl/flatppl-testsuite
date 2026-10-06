"""Yield helpers preserve model axes, empty identities and zero-factor adjoints."""

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.fixture
def yield_query(tmp_path):
    _gate_engine("stablehlo")
    (tmp_path / "reference.flatppl").write_text(
        'flatppl_compat = "0.1"\n'
        "sample_yields(nominal, shifts, factors) =\n"
        "  (nominal .+ aggregate(sum, [.s, .b], shifts[.s, .a, .b] / 1.0)) .*\n"
        "  aggregate(prod, [.s, .b], factors[.s, .m, .b] / 1.0)\n"
        "expected_counts(samples) = aggregate(sum, [.b], samples[.s, .b] / 1.0)\n"
    )
    def write(body, reference=False):
        module = 'load_module("reference.flatppl")' if reference else 'standard_module("pyhf_helpers", "0.1")'
        path = tmp_path / "query.flatppl"
        path.write_text(f'flatppl_compat = "0.1"\npyhf = {module}\n{body}')
        return path
    return write


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("dtype, tolerance", [("f64", 1e-11), ("f32", 2e-4)])
def test_yield_model_axes_and_empty_identities(yield_query, dtype, tolerance):
    jax, _, _ = ex._jax()
    device = jax.devices()[0]
    numeric = np.float64 if dtype == "f64" else np.float32
    for samples, additive, factors in [(2, 2, 3), (2, 0, 3), (2, 2, 0), (2, 0, 0), (0, 2, 3)]:
        # Select empty axes from typed runtime arrays, retaining bins and batches.
        shapes = [(max(samples, 1), 3), (max(samples, 1), max(additive, 1), 3),
                  (max(samples, 1), max(factors, 1), 3)]
        arrays = [np.arange(2 * np.prod(shape), dtype=numeric).reshape((2, *shape)) / 10 + 1
                  for shape in shapes]
        arrays[2].flat[0] = 0
        expected = ((arrays[0][:, :samples] + arrays[1][:, :samples, :additive].sum(axis=2))
                    * arrays[2][:, :samples, :factors].prod(axis=2)).sum(axis=1)
        declarations = [f"{name} = elementof(cartpow(cartpow(reals, {list(shape)}), 2))"
                        for name, shape in zip(("n", "a", "m"), shapes)]
        s, a, m = (":" if size else "ix" for size in (samples, additive, factors))
        body = "\n".join(declarations) + "\ninputs = (n, a, m)\nix = fill(1, get([0], 1))\n"
        body += f"block(n, a, m) = pyhf.sample_yields(n[{s}, :], a[{s}, {a}, :], m[{s}, {m}, :])\n"
        body += "outputs = pyhf.expected_counts.(block.(n, a, m))\n"
        for reference in (False, True):
            for mode in (False, True):
                source = ex.emit(yield_query(body, reference), "logdensity", dtype=dtype, restrict_enzyme_compatible=mode)
                executable = device.client.compile_and_load(source, [device])
                actual = executable.execute([jax.device_put(value) for value in arrays])[0]
                np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("dtype, tolerance", [("f64", 1e-11), ("f32", 2e-4)])
def test_yield_zero_factor_adjoints_accumulate_shared_batches(yield_query, dtype, tolerance):
    jax, jnp, hlo_call = ex._jax()
    numeric = np.float64 if dtype == "f64" else np.float32
    nominal = np.array([[[10, 20]]], dtype=numeric)
    shifts = np.array([[[[1, 2], [3, 4]]]], dtype=numeric)
    factors = np.array([[[[0, 2], [3, 0], [5, 0]]], [[[1, 1], [1, 1], [1, 1]]]], dtype=numeric)
    weights = np.array([[2, 7], [17, 19]], dtype=numeric)
    shifted = nominal + shifts.sum(axis=2)
    dn = (factors.prod(axis=2) * weights[:, None, :]).sum(axis=0, keepdims=True)
    da = np.broadcast_to(dn[:, :, None, :], shifts.shape)
    # The derivative multiplies the OTHER factors. It never divides by zero.
    dm = np.stack([np.prod(np.delete(factors, index, axis=2), axis=2) * shifted * weights[:, None, :]
                   for index in range(factors.shape[2])], axis=2)
    body = (
        "n = elementof(cartpow(cartpow(reals, [1, 2]), 1))\n"
        "a = elementof(cartpow(cartpow(reals, [1, 2, 2]), 1))\n"
        "m = elementof(cartpow(cartpow(reals, [1, 3, 2]), 2))\n"
        "inputs = (n, a, m)\n"
        "yields = pyhf.sample_yields.(nominal = n, factors = m, shifts = a)\n"
        "outputs = pyhf.expected_counts.(yields)\n"
    )
    for reference in (False, True):
        source = ex.emit(yield_query(body, reference), "logdensity", dtype=dtype, restrict_enzyme_compatible=True)
        evaluate = jax.jit(lambda n, a, m: jnp.sum(hlo_call(n, a, m, source=source)[0] * weights))
        np.testing.assert_allclose(evaluate(nominal, shifts, factors), 732, rtol=tolerance, atol=tolerance)
        gradient = jax.jit(jax.grad(evaluate, argnums=(0, 1, 2)))
        for actual, expected in zip(gradient(nominal, shifts, factors), (dn, da, dm)):
            np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("dtype, integer, power", [("f64", np.int64, 62), ("f32", np.int32, 30)])
def test_yield_inputs_promote_before_sum_and_product(yield_query, dtype, integer, power):
    jax, _, _ = ex._jax()
    device = jax.devices()[0]
    arrays = [np.array([[1]], dtype=integer), np.array([[[0]]], dtype=integer),
              np.full((1, 2, 1), 2**power, dtype=integer)]
    body = (
        "n = elementof(cartpow(integers, [1, 1]))\n"
        "a = elementof(cartpow(integers, [1, 1, 1]))\n"
        "m = elementof(cartpow(integers, [1, 2, 1]))\ninputs = (n, a, m)\n"
        "outputs = (pyhf.sample_yields(record(factors = m, shifts = a, nominal = n)),\n"
        "  pyhf.expected_counts(m[1, :, :]))\n"
    )
    for reference in (False, True):
        source = ex.emit(yield_query(body, reference), "logdensity", dtype=dtype)
        executable = device.client.compile_and_load(source, [device])
        product, total = executable.execute([jax.device_put(value) for value in arrays])
        assert np.asarray(product).dtype == (np.float64 if dtype == "f64" else np.float32)
        np.testing.assert_array_equal(product, [[float(2**(2 * power))]])
        np.testing.assert_array_equal(total, [float(2**(power + 1))])
