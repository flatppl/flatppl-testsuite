"""Evaluate converted pyhf likelihoods as float64 StableHLO on IREE CPU."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from flatppl_testsuite.unified.detjs_exec import PointScore
from flatppl_testsuite.unified.stablehlo_exec import emit


def log_density_points(model: Path, binding: str, points: list[dict]) -> list[PointScore]:
    """Compile and invoke once, broadcasting over runtime parameter records."""
    import iree.compiler as compiler
    import iree.runtime as runtime

    if not points:
        return []
    fields = list(points[0])
    if any(set(point) != set(fields) for point in points):
        raise ValueError("all parameter points must contain the same fields")
    # Pack the parameter ABI so each field does not need a separate buffer.
    shapes = {field: np.asarray(points[0][field]).shape for field in fields}
    if any(
        np.asarray(point[field]).shape != shapes[field]
        for point in points for field in fields
    ):
        raise ValueError("parameter shapes must match across points")
    entries = []
    offset = 0
    for field, shape in shapes.items():
        if len(shape) > 1:
            raise ValueError("pyhf parameters must be scalars or one-dimensional vectors")
        size = shape[0] if shape else 1
        indices = ", ".join(str(i) for i in range(offset + 1, offset + size + 1))
        selector = f"[{indices}]" if shape else indices
        entries.append(f"{field} = get(__point__, {selector})")
        offset += size
    record = ", ".join(entries)
    query = (
        f"__model__ = load_module({json.dumps(model.name)})\n"
        f"__target__ = __model__.{binding}\n"
        f"__score__(__point__) = logdensityof(__target__, record({record}))\n"
        f"__theta__ = elementof(cartpow(cartpow(reals, {offset}), {len(points)}))\n"
        "inputs = (__theta__)\noutputs = __score__.(__theta__)\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".flatppl", dir=model.parent) as source:
        source.write(query)
        source.flush()
        mlir = emit(Path(source.name), "logdensity", dtype="f64")

    binary = compiler.compile_str(
        mlir, input_type="stablehlo", target_backends=["llvm-cpu"],
        extra_args=[
            "--iree-llvmcpu-target-cpu=host",
            # IREE otherwise demotes f64 input programs to f32 by default.
            "--iree-input-demote-f64-to-f32=false",
            "--iree-vm-target-extension-f64=true",
            # f64 transcendental operations need the host's libm symbols.
            "--iree-llvmcpu-link-embedded=false",
            # The constant-evaluation JIT still uses embedded linking.
            "--iree-opt-const-eval=false",
            # Compute constant tensors once at module load, not per likelihood.
            "--iree-opt-const-expr-hoisting=true",
            # Coalesce proven shared allocations, including dispatch outputs.
            "--iree-stream-resource-alias-mutable-bindings=true",
        ],
    )
    config = runtime.Config("local-sync")
    module = runtime.VmModule.copy_buffer(config.vm_instance, binary)
    context = runtime.VmContext(
        instance=config.vm_instance, modules=[*config.default_vm_modules, module],
    )
    function = module.lookup_function("main")
    theta = np.asarray([
        [value for field in fields for value in np.atleast_1d(point[field])]
        for point in points
    ], dtype=np.float64)
    arguments = runtime.VmVariantList(1)
    arguments.push_ref(config.device.allocator.allocate_buffer_copy(
        memory_type=runtime.MemoryType.DEVICE_LOCAL,
        allowed_usage=runtime.BufferUsage.DEFAULT, device=config.device,
        buffer=theta, element_type=runtime.HalElementType.FLOAT_64,
    ))
    returns = runtime.VmVariantList(1)
    context.invoke(function, arguments, returns)
    result = returns.get_as_object(0, runtime.HalBufferView)
    if (
        result.shape != [len(points)]
        or result.element_type != int(runtime.HalElementType.FLOAT_64)
    ):
        raise ValueError(
            f"expected {len(points)} f64 scores, "
            f"got shape {result.shape}, type {result.element_type}"
        )
    # Use the buffer protocol: IREE 3.11's ndarray helper leaks mapped views.
    values = np.frombuffer(result.map(), dtype=np.float64, count=len(points)).tolist()
    return [PointScore(value=value) for value in values]
