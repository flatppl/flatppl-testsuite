"""On-demand ROOT (PyROOT) and pyHS3 oracles, reused from HS3TestSuite.

Each oracle delegates directly to the suite's own backend class, run inside its
pixi environment via a subprocess so the heavy dependencies (ROOT, pytensor) do
not pollute the base environment:

  ROOT   pixi run --frozen -e root   python -c "..."  (cwd = HS3TestSuite checkout)
  pyHS3  pixi run --frozen -e pyhs3  python -c "..."  (cwd = HS3TestSuite checkout)

The subprocess imports the suite backend, loads the workspace, calls
``run_twice_delta_nll_scan``, and prints the result as a JSON array on stdout.
This avoids reimplementing the scoring and keeps the oracle behaviour in sync
with the suite as it evolves.

ROOT is the reference backend that produced the frozen expected values, so the
ROOT oracle is a live recompute of ground truth; pyHS3 is an independent second
opinion. Used to debug divergences, not on the default path.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

from ..base import ForeignEngine

# The oracle subprocesses HS3TestSuite's OWN `python -m hs3suite` backend, which
# requires the upstream HS3TestSuite checkout (the vendored corpus is data-only).
# Defaults to a sibling checkout; HS3SUITE overrides. If absent, the subprocess
# fails and the oracle is reported unavailable (callers skip).
_HS3TESTSUITE = Path(os.environ.get(
    "HS3SUITE", Path(__file__).resolve().parents[5] / "HS3TestSuite"))

# Map harness backend names to the suite's backend names (currently identical,
# but kept explicit so we can rename without touching call sites).
_BACKEND_MAP: dict[str, str] = {
    "roofit": "roofit",
    "pyhs3": "pyhs3",
}

# Map harness backend names to the pixi environment that provides the deps.
_ENV_MAP: dict[str, str] = {
    "roofit": "root",
    "pyhs3": "pyhs3",
}


def _oracle_script(suite_backend: str, test_id: str) -> str:
    """Return a self-contained Python script that prints the 2ΔNLL vector as JSON.

    The script is executed inside the oracle pixi environment where the suite's
    heavy dependencies (ROOT / pyhs3+pytensor) are available.  It imports the
    suite backend directly so the scoring logic stays in one place.
    """
    return textwrap.dedent(f"""\
        import json, sys
        from pathlib import Path

        # Locate the fixture relative to the suite root (cwd).
        root = Path(".")
        manifest_path = root / "manifest.json"
        with manifest_path.open() as fh:
            manifest = json.load(fh)

        fixture = next(
            (f for f in manifest["fixtures"] if f["test_id"] == {test_id!r}),
            None,
        )
        if fixture is None:
            print(json.dumps({{"error": "test_id not found: {test_id}"}}))
            sys.exit(1)

        expected_path = root / fixture["path"] / "expected.json"
        with expected_path.open() as fh:
            expected = json.load(fh)

        check = next(
            (c for c in expected["checks"] if c["kind"] == "twice_delta_nll_scan"),
            None,
        )
        if check is None:
            print(json.dumps({{"error": "no twice_delta_nll_scan check in {test_id}"}}))
            sys.exit(1)

        hs3_path = root / fixture["path"] / "hs3.json"

        from hs3suite.backends import build_backend
        backend = build_backend({suite_backend!r})
        workspace = backend.load_workspace(hs3_path)
        vector = backend.run_twice_delta_nll_scan(workspace, check)
        print(json.dumps(vector))
    """)


def run_oracle(backend: str, test_id: str) -> list[float]:
    """Run one suite backend on one fixture and return its 2ΔNll vector.

    Parameters
    ----------
    backend:
        ``"roofit"`` (ROOT/PyROOT) or ``"pyhs3"``.
    test_id:
        Suite fixture identifier, e.g. ``"rf101_basics"``.

    Returns
    -------
    list[float]
        The 2ΔNLL values at each scan point, in order.

    Raises
    ------
    RuntimeError
        If ``pixi`` is not on PATH, the oracle environment is not provisioned,
        or the required backend dependencies are not importable.  Callers should
        treat this as a skip signal rather than a test failure.
    ValueError
        If ``backend`` is not ``"roofit"`` or ``"pyhs3"``.
    """
    if backend not in _BACKEND_MAP:
        raise ValueError(
            f"unsupported oracle backend {backend!r}; "
            f"choose from {sorted(_BACKEND_MAP)}"
        )

    if shutil.which("pixi") is None:
        raise RuntimeError("pixi not found on PATH; oracle unavailable")

    suite_backend = _BACKEND_MAP[backend]
    pixi_env = _ENV_MAP[backend]
    script = _oracle_script(suite_backend, test_id)

    # Use bare 'python' so pixi selects the environment's own interpreter;
    # sys.executable would use the base-environment interpreter instead.
    cmd = ["pixi", "run", "--frozen", "-e", pixi_env, "python", "-c", script]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(_HS3TESTSUITE),
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError:
        raise RuntimeError("pixi not found on PATH; oracle unavailable") from None
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"oracle subprocess timed out for {backend}/{test_id}") from None

    if result.returncode != 0:
        # Distinguish "env not provisioned" from other errors so callers can
        # skip cleanly.  pixi exits with a distinctive message when an env is
        # not yet solved/installed.
        stderr = result.stderr or ""
        if any(
            phrase in stderr.lower()
            for phrase in (
                "no environment",
                "not installed",
                "could not find",
                "cannot find",
                "environment does not exist",
                "please install",
                "solve",
            )
        ):
            raise RuntimeError(
                f"oracle env '{pixi_env}' not provisioned "
                f"(pixi exit {result.returncode}): {stderr.strip()[:200]}"
            )
        raise RuntimeError(
            f"oracle subprocess failed (exit {result.returncode}) "
            f"for {backend}/{test_id}:\n"
            f"stdout: {result.stdout.strip()[:400]}\n"
            f"stderr: {stderr.strip()[:400]}"
        )

    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError(
            f"oracle produced no output for {backend}/{test_id}; "
            f"stderr: {(result.stderr or '').strip()[:400]}"
        )

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"oracle output is not valid JSON for {backend}/{test_id}: "
            f"{stdout[:200]}"
        ) from exc

    if isinstance(payload, dict) and "error" in payload:
        raise RuntimeError(
            f"oracle script error for {backend}/{test_id}: {payload['error']}"
        )

    if not isinstance(payload, list):
        raise RuntimeError(
            f"expected a JSON array from oracle, got {type(payload).__name__}"
        )

    return [float(v) for v in payload]


# ---------------------------------------------------------------------------
# ABC-conforming face
# ---------------------------------------------------------------------------


class HS3ForeignEngine(ForeignEngine):
    """Run an HS3 model in a foreign engine (roofit | pyhs3) via the suite backend."""

    def __init__(self, backend: str):
        self.backend = backend

    def twice_delta_nll(self, model, target, scan_param, scan_points, reference):
        # The HS3 suite backend reads the whole check; test_id identifies the fixture.
        return run_oracle(self.backend, target)
