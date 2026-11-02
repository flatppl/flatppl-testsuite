"""A BridgeStan box served by flatppl-bridge, queried from FlatPPL through flatppl-python.

The Stan model in ``tests/bridge/normal.stan`` has a closed-form Normal
posterior. Each check compares the compiled FlatPPL query with BridgeStan
called directly and with scipy. Run it in the ``bridge`` environment after
``pixi run -e bridge bridge-setup``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

pytestmark = [
    pytest.mark.bridge_only,
    # The server's plugin and the direct oracle load the same model library.
    pytest.mark.filterwarnings("ignore:Loading a shared object"),
]

jax = pytest.importorskip("jax")
flatppl = pytest.importorskip("flatppl")
fb = pytest.importorskip("flatppl_bridge")
bridgestan = pytest.importorskip("bridgestan")

HERE = Path(__file__).parent / "bridge"
BUILD = Path(os.environ.get("FLATPPL_BRIDGE_BUILD", Path(__file__).parents[1] / ".bridge"))
DATA = json.loads((HERE / "normal.data.json").read_text())


def _built(name: str) -> Path:
    path = BUILD / name
    if not path.exists():
        pytest.skip(f"{path} is missing: run `pixi run -e bridge bridge-setup`")
    return path


@pytest.fixture(scope="module")
def server():
    plugin = _built("plugin")
    model = _built("normal_model.so")
    library = next(plugin.glob("libfpb_bridgestan.*"))
    config = json.dumps({"model": str(model), "data": str(HERE / "normal.data.json")})
    box = fb.Box()
    box.measure(
        "std",
        logdensity=lambda x: stats.norm.logpdf(x),
        sample=_standard_normal,
        variate="reals",
        normalized=True,
    )
    box.kernel(
        "K",
        logdensity=lambda p, x: stats.norm.logpdf(x, p["mu"], p["sigma"]),
        gradient=_kernel_gradient,
        input="cartprod(mu = reals, sigma = posreals)",
        variate="reals",
        normalized=True,
    )
    with fb.Server({"py": box}, libraries={"stan": (str(library), config)}) as running:
        yield running


def _kernel_gradient(p, x):
    z = (x - p["mu"]) / p["sigma"]
    d_input = {"mu": z / p["sigma"], "sigma": (z**2 - 1.0) / p["sigma"]}
    return d_input, -z / p["sigma"]


def _standard_normal(size, rng):
    return rng.normal(size=size)


@pytest.fixture(scope="module")
def stan_model():
    return bridgestan.StanModel(str(_built("normal_model.so")), str(HERE / "normal.data.json"))


def _scipy(theta: np.ndarray) -> float:
    total = stats.norm.logpdf(theta, 1.5, 2.0).sum()
    total += stats.norm.logpdf(DATA["y1"], theta[0], 1.0).sum()
    total += stats.norm.logpdf(DATA["y2"], theta[1], 1.0).sum()
    return float(total)


def _scipy_gradient(theta: np.ndarray) -> np.ndarray:
    y = [np.asarray(DATA["y1"]), np.asarray(DATA["y2"])]
    return np.array([-(t - 1.5) / 4.0 + np.sum(y[i] - t) for i, t in enumerate(theta)])


def _compile(server, query: str, **options):
    module = flatppl.load(HERE / query)
    boxes = {
        "stan": flatppl.Box(server.url, "stan"),
        "py": flatppl.Box(server.url, "py"),
    }
    used = {alias: box for alias, box in boxes.items() if f'{alias} = load_module' in module.source}
    return module.compile(boxes=used, **options)


POINTS = np.array([[0.3, -0.2], [1.0, 2.0], [-1.5, 3.5], [0.9, 2.4]])


def test_log_density_and_gradient_match_bridgestan_and_scipy(server, stan_model):
    with jax.enable_x64():
        density = _compile(server, "posterior.flatppl", dtype="float64")
        for theta in POINTS:
            value, gradient = jax.value_and_grad(density)(jax.numpy.asarray(theta))
            direct, direct_gradient = stan_model.log_density_gradient(
                theta, propto=False, jacobian=True
            )
            np.testing.assert_allclose(value, direct, rtol=1e-12)
            np.testing.assert_allclose(gradient, direct_gradient, rtol=1e-12)
            np.testing.assert_allclose(value, _scipy(theta), rtol=1e-12)
            np.testing.assert_allclose(gradient, _scipy_gradient(theta), rtol=1e-10, atol=1e-12)


def test_hmc_gradient_loop_under_jit_and_vmap(server):
    jnp = jax.numpy
    with jax.enable_x64():
        density = _compile(server, "posterior.flatppl", dtype="float64")

        def reference(theta):
            prior = jax.scipy.stats.norm.logpdf(theta, 1.5, 2.0).sum()
            y1 = jax.scipy.stats.norm.logpdf(jnp.asarray(DATA["y1"]), theta[0], 1.0).sum()
            y2 = jax.scipy.stats.norm.logpdf(jnp.asarray(DATA["y2"]), theta[1], 1.0).sum()
            return prior + y1 + y2

        def trajectory(logdensity):
            def leapfrog(state, _):
                q, p = state
                p = p + 0.025 * jax.grad(logdensity)(q)
                q = q + 0.05 * p
                p = p + 0.025 * jax.grad(logdensity)(q)
                return (q, p), logdensity(q)

            def run(q, p):
                return jax.lax.scan(leapfrog, (q, p), length=8)

            return jax.jit(jax.vmap(run))

        momenta = jnp.asarray(np.random.default_rng(7).normal(size=POINTS.shape))
        actual = trajectory(density)(jnp.asarray(POINTS), momenta)
        expected = trajectory(reference)(jnp.asarray(POINTS), momenta)
        for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
            np.testing.assert_allclose(a, e, rtol=1e-10, atol=1e-12)


def test_a_draw_from_a_normalized_box_measure_uses_the_box_sampler(server):
    with jax.enable_x64():
        draw = _compile(server, "draw.flatppl", dtype="float64", autodiff=False)
        key = jax.numpy.array([1729, 5], dtype=jax.numpy.uint64)
        value, _ = draw(key)
        seed = np.asarray(key).astype("<u8").view(np.uint8)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(list(seed))))
        np.testing.assert_allclose(value, _standard_normal((), rng), rtol=1e-15)


def test_iid_of_a_record_input_kernel_differentiates_under_vmap(server):
    observed = np.asarray(DATA["y1"])
    with jax.enable_x64():
        density = _compile(server, "kernel.flatppl", dtype="float64")
        mus = jax.numpy.array([-0.5, 0.4, 1.3])
        values = jax.jit(jax.vmap(density))(mus)
        gradients = jax.jit(jax.vmap(jax.grad(density)))(mus)
    for mu, value, gradient in zip(np.asarray(mus), values, gradients, strict=True):
        np.testing.assert_allclose(value, stats.norm.logpdf(observed, mu, 1.5).sum(), rtol=1e-12)
        np.testing.assert_allclose(gradient, np.sum(observed - mu) / 1.5**2, rtol=1e-12)
