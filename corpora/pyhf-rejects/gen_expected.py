#!/usr/bin/env python3
"""Freeze each ``corpora/pyhf-rejects/<doc>/test.json``'s converter outcome.

One document per pyhf validation-failure class, plus the handful pyhf and the
converter disagree about. There is no log-density to freeze: an exit-1 document
has none. What each row holds is the converter's **exit code** and, for a
refusal, a substring of its message — so the converter cannot start accepting a
document pyhf rejects, or drift to a reason naming a different defect, with a
green run.

Both sides are MEASURED here, not copied:

* pyhf's own verdict comes from building the workspace and catching what it
  raises. It is recorded as ``pyhf_error`` (null when pyhf accepts).
* the converter's exit code and stderr come from running ``FLATPPL_BIN``.

``pyhf_agrees`` is then derived: true when both refuse or both accept, false
when they disagree. A false is a KNOWN, reasoned mismatch — the README lists
all of them and why none can produce a wrong number — not a defect. The
generator prints the mismatch set at the end so a NEW one cannot slip in
unnoticed.

The refusal substring is the message minus its ``flatppl: pyhf: `` prefix, so
the row survives a change to the CLI's diagnostic framing while still pinning
the defect the converter names.

Run in the pyhf pixi env:

    FLATPPL_BIN=/path/to/flatppl pixi run -e pyhf gen-pyhf-rejects
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

_PREFIXES = ("flatppl: pyhf: ", "flatppl: hs3: ", "flatppl: ")


def flatppl_bin() -> str:
    b = os.environ.get("FLATPPL_BIN")
    if not b:
        sys.exit("FLATPPL_BIN is unset; this generator runs the converter.")
    if not Path(b).exists():
        sys.exit(f"FLATPPL_BIN={b} does not exist")
    return b


def converter_outcome(workspace: Path) -> tuple[int, str]:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "m.flatppl"
        proc = subprocess.run(
            [flatppl_bin(), "convert", "--from", "pyhf", str(workspace), str(out)],
            capture_output=True, text=True,
        )
        return proc.returncode, proc.stderr.strip()


def defect_substring(stderr: str) -> str:
    """The refusal's own sentence, with the CLI's diagnostic prefix removed."""
    line = next((ln.strip() for ln in stderr.splitlines() if ln.strip()), "")
    for p in _PREFIXES:
        if line.startswith(p):
            return line[len(p):]
    return line


def pyhf_verdict(spec: dict) -> str | None:
    """What pyhf itself does with the document: an error string, or None.

    The probe runs all the way to a `logpdf` call, not just `Workspace` +
    `model`. pyhf defers some validation to evaluation -- an observation of the
    wrong length passes `Workspace.data` and only raises as `InvalidPdfData`
    from `logpdf` -- so stopping at model construction would record pyhf as
    ACCEPTING a document it rejects, and manufacture a mismatch that is not one.
    """
    import pyhf

    try:
        ws = pyhf.Workspace(spec)
        model = ws.model()
        data = ws.data(model)
        model.logpdf(model.config.suggested_init(), data)
    except Exception as e:  # noqa: BLE001 - any refusal, whatever its class
        detail = " ".join(str(e).split())
        return f"{type(e).__name__}: {detail}"
    return None


def generate(dir: Path) -> tuple[str, bool]:
    import pyhf

    source = dir / "pyhf.json"
    spec = json.loads(source.read_text())

    error = pyhf_verdict(spec)
    code, stderr = converter_outcome(source)
    agrees = (error is not None) == (code != 0)

    test_path = dir / "test.json"
    body = json.loads(test_path.read_text()) if test_path.exists() else {}

    check = {
        "id": "convert_outcome",
        "kind": "convert_outcome",
        "expect_exit": code,
    }
    if code != 0:
        check["stderr_contains"] = defect_substring(stderr)

    body = {
        **body,
        "test_type": "convert",
        "engines": ["det-js"],
        "fixture_kind": "pyhf_reject",
        "source": "pyhf.json",
        "oracle": {
            "tool": "pyhf",
            "version": pyhf.__version__,
            "generator": "corpora/pyhf-rejects/gen_expected.py",
            "quantity": "the converter's exit code and refusal message, beside pyhf's own verdict",
        },
        "pyhf_error": error,
        "pyhf_agrees": agrees,
        "checks": [
            {
                "id": "static_integrity",
                "kind": "static_integrity",
                "canonical_sha256": hashlib.sha256(
                    json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            },
            check,
        ],
    }
    test_path.write_text(json.dumps(body, indent=2) + "\n")
    return dir.name, agrees


def main() -> None:
    dirs = sorted(p.parent for p in HERE.rglob("pyhf.json"))
    if not dirs:
        sys.exit(f"no rejection documents under {HERE}")
    mismatches = []
    for dir in dirs:
        name, agrees = generate(dir)
        body = json.loads((dir / "test.json").read_text())
        code = body["checks"][1]["expect_exit"]
        print(f"{name}: converter exit {code}, pyhf "
              f"{'refuses' if body['pyhf_error'] else 'accepts'}"
              f"{'' if agrees else '   <- MISMATCH'}")
        if not agrees:
            mismatches.append(name)
    print(f"\nfroze {len(dirs)} documents; {len(mismatches)} known mismatch(es):")
    for m in mismatches:
        print(f"  {m}")


if __name__ == "__main__":
    main()
