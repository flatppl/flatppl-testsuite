#!/usr/bin/env python
"""Regenerate the HS3 conversion examples and append a scoring line.

For each ``*.hs3`` under a target directory (default: this ``conversions/`` dir;
pass another such as ``flatppl-examples/examples/hs3``):

  1. convert ``<model>.hs3`` → ``<model>.flatppl`` via the flatppl converter
     (honours ``FLATPPL_BIN``);
  2. append a scoring section that evaluates the root likelihood at the model's
     nominal parameter point:

         % === scoring ===
         log_likelihood = logdensityof(<L>, record(<free params at nominal>))

Everything above the ``% === scoring ===`` marker is exactly the converter's
output, so ``test_known_good_conversion`` re-pins against it; the scoring part
is derived mechanically — the root likelihood binding and the nominal θ — so no
hand editing is required.

    pixi run python corpora/hs3/conversions/regen.py                  # this dir
    pixi run python corpora/hs3/conversions/regen.py <dir-of-hs3>      # e.g. examples
    FLATPPL_BIN=/path/to/flatppl pixi run python .../regen.py [<dir>]
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

# Allow `python regen.py` outside an installed env; pixi already has the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from flatppl_testsuite.config import CONFIG  # noqa: E402
from flatppl_testsuite.scoring.engine import render_record  # noqa: E402

HERE = Path(__file__).resolve().parent
MARKER = "% === scoring ==="


def root_likelihood(src: str) -> str:
    """The top-level likelihood binding that nothing else references.

    A model's full likelihood is the `likelihoodof`/`joint_likelihood` binding
    that is not itself consumed by another binding (histfactory's combined
    `joint_likelihood` over its constraint terms; the lone `likelihoodof`
    otherwise).
    """
    cands = re.findall(r"(?m)^([A-Za-z_]\w*) = (?:joint_likelihood|likelihoodof)\(", src)
    if not cands:
        raise ValueError("no likelihood binding in converted source")

    def referenced(name: str) -> bool:
        pat = re.compile(rf"\b{re.escape(name)}\b")
        return any(pat.search(ln) for ln in src.splitlines()
                   if not ln.startswith(f"{name} ="))

    roots = [c for c in cands if not referenced(c)]
    return roots[-1] if roots else cands[-1]


def free_params(src: str) -> list[tuple[str, int]]:
    """Each `<name> = elementof(<set>)` → (name, size); size>1 for a cartpow vector."""
    out: list[tuple[str, int]] = []
    for name, setexpr in re.findall(r"(?m)^([A-Za-z_]\w*) = elementof\((.*)\)\s*$", src):
        m = re.search(r"cartpow\([^,]+,\s*(\d+)\)", setexpr)
        out.append((name, int(m.group(1)) if m else 1))
    return out


def nominal_values(hs3: dict) -> dict[str, float]:
    """Flat name→value map from the `default_values` parameter point (else first)."""
    pts = hs3.get("parameter_points", [])
    pt = next((p for p in pts if p.get("name") == "default_values"), pts[0] if pts else {})
    return {e["name"]: float(e["value"]) for e in pt.get("parameters", pt.get("entries", []))}


def theta(src: str, hs3: dict) -> dict:
    """Nominal θ: each free param at its parameter-point value (vectors collapsed)."""
    vals = nominal_values(hs3)
    out: dict = {}
    for name, size in free_params(src):
        if size == 1:
            out[name] = vals.get(name, 0.0)
        else:
            # A cartpow vector param's components are stored split (`mcstat_0`, …).
            out[name] = [vals.get(f"{name}_{i}", 1.0) for i in range(size)]
    return out


def regen(hs3_path: Path) -> None:
    out_path = hs3_path.with_suffix(".flatppl")
    subprocess.run(
        [str(CONFIG.flatppl_bin), "convert", "--from", "hs3",
         str(hs3_path), str(out_path), "--no-header"],
        check=True,
    )
    src = out_path.read_text().rstrip()
    hs3 = json.loads(hs3_path.read_text())
    binding = root_likelihood(src)
    rec = render_record(theta(src, hs3))
    out_path.write_text(
        f"{src}\n\n{MARKER}\n"
        f"% Evaluate the root likelihood at the nominal parameter point.\n"
        f"log_likelihood = logdensityof({binding}, {rec})\n"
    )
    print(f"{hs3_path.stem}: log_likelihood = logdensityof({binding}, {rec})")


if __name__ == "__main__":
    # Scan a target directory for `*.hs3` (default: this conversions/ dir).
    # Works for both the per-model-subdir layout here and the flat layout in
    # flatppl-examples/examples/hs3.
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE
    hs3_files = sorted(root.rglob("*.hs3"))
    if not hs3_files:
        raise SystemExit(f"no .hs3 files under {root}")
    for hs3_path in hs3_files:
        regen(hs3_path)
