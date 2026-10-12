"""Rowwise scalar laws preserve joined axes, atom support and RNG threading."""

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("precision", ["f32", "f64"])
@pytest.mark.parametrize("law,field,point", [
    ("StudentT", "nu", [1.3, -.5, .75, 2.5, 1.2, 8, 12, 20, 0]),
    ("Bernoulli", "p", [1.3, -.5, -2, .75, 1.2, .1, .5, .9, 0]),
])
def test_rowwise_scalar_laws_keep_cell_laws_and_successor_keys(
    tmp_path, precision, law, field, point,
):
    _gate_engine("stablehlo")
    jax, jnp, hlo_call = ex._jax()
    dtype = np.float32 if precision == "f32" else np.float64
    prefix = (
        "p = external(cartpow(reals, 9))\n"
        "mu = addaxes([p[1], p[2]], 0, 1)\n"
        "sigma = addaxes([p[3], p[4], p[5]], 1, 0)\n"
        "nu = addaxes([p[6], p[7], p[8]], 1, 0)\n"
    )
    suffix = (
        "after, final_state = rand(Normal(0, 1), state)\n"
        "inputs = p\noutputs = (value, state, after, final_state)\n"
    )

    def sampler(body):
        query = tmp_path / "query.flatppl"
        query.write_text(prefix + body + suffix)
        module = ex.emit(query, "sample", dtype=precision, restrict_enzyme_compatible=False)
        return lambda key, params: hlo_call(key, params, source=module)

    row_parameter = "d" if law == "StudentT" else "q"
    draw = sampler(
        f"row = (l, s, {row_parameter}) -> locscale({law}({row_parameter}), l, s)\n"
        "value, state = rand(row.(mu, sigma, nu), rnginit(23))\n"
    )
    reference = jax.jit(sampler(
        f"z, state = builtin_sample(rnginit(23), broadcasted({law}), "
        f"record({field} = nu .+ fill(0, [2, 1])))\n"
        "value = sigma .* z .+ mu\n"
    ))
    sample = jax.jit(draw)
    params = np.asarray(point, dtype=dtype)
    reflected = params.copy()
    reflected[[2, 4]] *= -1
    points = jnp.asarray(np.stack([params, reflected]))
    keys = jnp.asarray([[23, 0], [0x8000000000000123, 0]], dtype=jnp.uint64)
    singles = []
    for key, values in zip(keys, points, strict=True):
        actual = jax.block_until_ready(sample(key, values))
        expected = jax.block_until_ready(reference(key, values))
        assert actual[0].shape == (2, 3) and actual[0].dtype == dtype
        for left, right in zip(actual, expected, strict=True):
            assert left.devices() == {jax.devices()[0]}
            np.testing.assert_array_equal(left, right)
        assert not np.array_equal(actual[1], key)
        assert not np.array_equal(actual[-1], actual[1])
        singles.append(actual)

    # This exercises executor batching; Rust BatchSpec compilation is separate.
    batched = jax.block_until_ready(jax.jit(jax.vmap(draw))(keys, points))
    for index, actual in enumerate(batched):
        np.testing.assert_array_equal(actual, np.stack([row[index] for row in singles]))

    def collect(key):
        def step(state, _):
            value, advanced, _, _ = draw(state, jnp.asarray(params))
            return advanced, value

        return jax.lax.scan(step, key, None, length=2048)

    _, values = jax.block_until_ready(jax.jit(collect)(keys[-1]))
    z = (np.asarray(values, dtype=np.float64) - params[None, :2, None]) / params[None, None, 2:5]
    assert np.all(np.isfinite(z))
    if law == "Bernoulli":
        np.testing.assert_allclose(z, np.round(z), atol=1e-6)
        z = np.round(z)
        assert np.all((z == 0) | (z == 1))
        mean = np.broadcast_to(params[None, 5:8].astype(np.float64), (2, 3))
        variance = mean * (1 - mean)
        fourth = variance * ((1 - mean)**3 + mean**3)
    else:
        mean = np.zeros((2, 3))
        df = params[None, 5:8].astype(np.float64)
        variance = df / (df - 2)
        fourth = 3 * df**2 / ((df - 2) * (df - 4))
    # Eight Monte Carlo standard errors from independent analytic moments.
    assert np.all(abs(z.mean(axis=0) - mean) < 8 * np.sqrt(variance / len(z)))
    second = ((z - mean)**2).mean(axis=0)
    assert np.all(abs(second - variance) <= 8 * np.sqrt((fourth - variance**2) / len(z)))
    correlation = np.corrcoef(z.reshape(len(z), -1), rowvar=False)
    assert np.max(abs(correlation[np.triu_indices(6, k=1)])) < 8 / np.sqrt(len(z))
