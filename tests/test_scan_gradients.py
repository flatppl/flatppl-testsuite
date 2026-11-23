"""A nonlinear scan preserves each state, its pullback and runtime index errors."""

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("dtype,tol", [("f32", 3e-6), ("f64", 2e-13)])
def test_nonlinear_scan_trajectory_and_adjoint(tmp_path, dtype, tol):
    _gate_engine("stablehlo")
    from jax.experimental import checkify

    n = 35
    query = tmp_path / "query.flatppl"
    query.write_text(
        "initial = elementof(reals)\nrho = elementof(reals)\n"
        f"x = elementof(cartpow(reals, {n}))\nw = elementof(cartpow(reals, {n}))\n"
        f"order = elementof(cartpow(integers, {n}))\n"
        "step(s, i, p) = tanh(p.rho * s + p.x[i])\n"
        "states = scan(step, initial, order, record(rho=rho, x=x))\n"
        "inputs = (initial, rho, x, w, order)\noutputs = sum(states .* w)\n"
    )
    source = ex.emit(query, "logdensity", dtype=dtype, target="gpu")
    initial, rho = 0.3, 0.7
    x = np.linspace(-0.5, 1.0, n)
    w = np.linspace(-0.4, 0.8, n)
    order = np.arange(n) * 11 % n
    order[-2:] = order[:2]
    args = [initial, rho, x, w, order + 1]

    state = [initial]
    for i in order:
        state.append(np.tanh(rho * state[-1] + x[i]))
    dx = np.zeros_like(x)
    carry = drho = 0.0
    for t in reversed(range(n)):
        dz = (carry + w[t]) * (1 - state[t + 1] ** 2)
        dx[order[t]] += dz
        drho += dz * state[t]
        carry = rho * dz

    assert ex.value(source, args) == pytest.approx(w @ state[1:], rel=tol, abs=tol)
    d_initial, d_rho, d_x = ex.gradient(source, args, [0, 1, 2])
    np.testing.assert_allclose([d_initial, d_rho], [carry, drho], rtol=tol, atol=tol)
    np.testing.assert_allclose(d_x, dx, rtol=tol, atol=tol)

    args[-1][-1] = n + 1
    with pytest.raises(checkify.JaxRuntimeError, match="index out of range"):
        ex.value(source, args)
    with pytest.raises(checkify.JaxRuntimeError, match="index out of range"):
        ex.gradient(source, args, [0, 1, 2])
