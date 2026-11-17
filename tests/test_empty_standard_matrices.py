"""Data-derived empty matrix cells retain their shapes and zero adjoints."""

import numpy as np
import pytest

from flatppl_testsuite.unified import stablehlo_exec as ex
from tests.test_unified import _gate_engine


@pytest.mark.stablehlo_only
@pytest.mark.parametrize("member", ["lu", "kron", "matexp"])
@pytest.mark.parametrize("complex_input", [False, True])
def test_empty_standard_matrix_operations(tmp_path, member, complex_input):
    _gate_engine("stablehlo")
    query = tmp_path / "empty.flatppl"
    cols = "2" if member == "kron" else "zero"
    matrix = "complex.(Ar, Ar)" if complex_input else "Ar"
    if member == "lu":
        expression = "e.lu(A)"
        outputs = ("C.P, real.(C.L), imag.(C.L), real.(C.U), imag.(C.U)"
                   if complex_input else "C.P, C.L, C.U")
        count, expected_shape = 5 if complex_input else 3, (0, 0)
    else:
        expression = "e.kron(A, fill(1.0, [3, 4]))" if member == "kron" else "e.matexp(A)"
        outputs = "real.(C), imag.(C)" if complex_input else "C"
        count, expected_shape = 2 if complex_input else 1, (0, 8) if member == "kron" else (0, 0)
    query.write_text(
        'e = standard_module("ext-linear-algebra", "0.1")\n'
        "zero = lengthof(filter(fn(_ > 2), [1]))\nt = external(reals)\n"
        f"Ar = fill(t, [zero, {cols}])\nA = {matrix}\nC = {expression}\n"
        f"inputs = t\noutputs = ({outputs})\n"
    )
    source = ex.emit(query, "logdensity", dtype="f64")
    jax, jnp, _ = ex._jax()
    call = ex.call_with_status(source)
    actual = ex.expect_valid(jax.jit(call)(jnp.float64(2)))
    assert len(actual) == count
    for value in actual:
        assert value.shape == expected_shape
        np.testing.assert_array_equal(value, np.zeros(expected_shape))

    def total(t):
        values, _ = call(t)
        return sum(jnp.sum(value) for value in values)

    assert jax.jit(jax.grad(total))(jnp.float64(2)) == 0
