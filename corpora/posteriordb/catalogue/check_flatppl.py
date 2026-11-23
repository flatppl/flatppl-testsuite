"""Score the published record queries in an installed FlatPPL Python environment."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import numpy as np
from flatppl import load
from jax.experimental import checkify
from regenerate import original_data, verified_digest, verify_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "measurements", type=Path, help="public dashboard measurements.json"
    )
    parser.add_argument(
        "database",
        type=Path,
        help="posteriorDB checkout's posterior_database directory",
    )
    parser.add_argument("fixtures", type=Path, nargs="+")
    parser.add_argument("--target", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != args.target:
        raise RuntimeError(f"select the {args.target} JAX backend to score this target")
    catalogue = json.loads(args.measurements.read_bytes())["catalogue"]
    query = Path(__file__).with_name("score.flatppl").read_bytes()
    for fixture in args.fixtures:
        spec = json.loads(fixture.read_text())
        p = spec["provenance"]
        entry = catalogue["sources"][fixture.stem]
        source = entry["source"].encode()
        verified_digest(source, p["flatppl_sha256"])
        verified_digest(query, p["query_sha256"])
        if (
            entry["query_sha256"] != p["query_sha256"]
            or catalogue["posteriors"][fixture.stem]["data_sha256"] != p["data_sha256"]
        ):
            raise ValueError(
                f"the published query or dataset changed for {fixture.stem}"
            )
        verify_points(spec["points"], p["point_hashes"])
        with TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "model.flatppl").write_bytes(source)
            (root / "data.json").write_bytes(original_data(args.database, p).encode())
            (root / "query.flatppl").write_bytes(query)
            score = load(root / "query.flatppl").compile(
                dtype="float64", target=args.target, checks="raise"
            )
            value = jax.jit(checkify.checkify(score))
            value_and_grad = jax.jit(checkify.checkify(jax.value_and_grad(score)))
            for i, point in enumerate(spec["points"]):
                record = {k: np.asarray(v, dtype=np.float64) for k, v in point.items()}
                status, scalar = jax.device_get(value(record))
                status.throw()
                status, (primal, grad) = jax.device_get(value_and_grad(record))
                status.throw()
                np.testing.assert_allclose(
                    scalar, spec["expected"][i], atol=1e-8, rtol=1e-8
                )
                ad_primal = (
                    p["stan_log_density_with_gradient"][i] + p["log_density_offset"]
                )
                np.testing.assert_allclose(primal, ad_primal, atol=1e-8, rtol=1e-8)
                expected = spec["expected_grad"][i]
                if grad.keys() != expected.keys():
                    raise ValueError(
                        f"missing or extra gradient fields in {fixture.stem}"
                    )
                for name, actual in grad.items():
                    np.testing.assert_allclose(
                        actual, expected[name], atol=1e-8, rtol=1e-8, strict=True
                    )
        print(fixture.stem, flush=True)
        jax.clear_caches()


if __name__ == "__main__":
    main()
