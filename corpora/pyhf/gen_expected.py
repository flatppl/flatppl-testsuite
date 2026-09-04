#!/usr/bin/env python3
"""Freeze each ``corpora/pyhf/<fixture>/test.json``'s absolute logpdf vector.

The oracle is **pyhf itself** (``pyhf.Model.logpdf``). The rust converter is
the artifact under test and the FlatPPL engine scores its output, so neither
can stand in as the reference number -- pyhf's own log-density is the only
authority for a pyhf workspace.

Unlike ``corpora/hs3/conversions/gen_expected.py`` this freezes an **absolute**
log-density per point, not an offset-invariant Delta against a reference point.
pyhf and the FlatPPL lowering both carry the full Poisson normalization, so the
absolute values must agree; and they have to be compared that way, because a
constraint read with the wrong form (the staterror defect the import audit
found) moves the normalization, which a Delta inside one model would partly
cancel.

Points are ``suggested_init()`` plus five drawn uniformly inside
``suggested_bounds()`` clipped to init +/- 2.5, with every
``suggested_fixed()`` component held at its init. ``SEED`` is fixed and the
draw is one vectorised ``rng.uniform`` per point, so the whole corpus's point
sets reproduce from scratch -- these are the same sets the import audit
measured. A fixture whose ``test.json`` already carries points REUSES them, so
an ordinary regen is a pure re-measurement whose diff shows only moved values.

The FlatPPL record shape per parameter is read off the converter's **own**
emitted ``elementof`` declaration, so a frozen point carries the shape the
engine will be handed rather than a guess: pyhf's per-bin kinds (shapesys,
staterror, shapefactor) become a vector even at one bin, while normfactor,
normsys, histosys and lumi stay scalar. That needs ``FLATPPL_BIN``, and a
workspace the converter refuses fails loudly here instead of being skipped.

Run in the pyhf pixi env:

    FLATPPL_BIN=/path/to/flatppl pixi run -e pyhf gen-pyhf
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Measured over all 1002 points: worst absolute difference 1.819e-12, on
# `sw3_norm_norm_shap_shap_stat`, whose log-density is about -1.9e+3 (a relative
# difference of 1e-15, one or two ulp of a double). This band is ~550x that
# worst case, so ordinary float reassociation between pyhf's numpy reduction
# order and the engine's cannot reach it, while the smallest defect the audit
# found (1.276e+0) is nine orders of magnitude outside it.
#
# `rtol` stays 0 deliberately. The corpus's deepest log-density is -4.66e+3, so
# an rtol of 1e-12 would admit 4.7e-9 there -- five times looser than atol, on
# exactly the rows where a constant offset from a wrong normalization hides
# best. An absolute band is what makes such an offset fail.
TOLERANCE = {"atol": 1e-9, "rtol": 0.0}

# Fixed so the point sets reproduce from scratch, not only re-measure. This is
# the audit's seed; changing it re-draws all 167 fixtures.
SEED = 137
N_DRAWN_POINTS = 5
CLIP = 2.5

# `name = elementof(cartpow(<set>, <n>))` -- a per-bin parameter, so the record
# field is a vector of length n. Any other `elementof` RHS is a scalar.
_ELEMENTOF = re.compile(
    r"^\s*([A-Za-z_]\w*)\s*=\s*elementof\(\s*(cartpow\(\s*\w+\s*,\s*(\d+)\s*\)|\w+)\s*\)\s*$",
    re.M,
)

# `likelihood = <rhs>` -- the converter's top-level likelihood binding.
_TOP = re.compile(r"^\s*likelihood\s*=\s*(.+?)\s*$", re.M)
# A bare identifier RHS, i.e. an alias rather than a likelihood-forming call.
_BARE_NAME = re.compile(r"^[A-Za-z_]\w*$")


def score_binding_name(src: str) -> tuple[str, str | None]:
    """The binding to score, and a note if it is not `likelihood` itself.

    A workspace with no constrained parameter has exactly one likelihood term,
    so the converter emits `likelihood = <channel>_likelihood` -- a bare alias.
    `flatppl determinize` refuses to score through such an alias ("expected
    likelihoodof"), which is a determiniser gap and has nothing to do with the
    conversion: a hand-written `top = lik; logdensityof(top, ...)` is refused
    the same way. Resolving the alias here keeps those fixtures scoring the
    same measure against pyhf, instead of letting that gap delete whole
    modifier kinds from the corpus. The note records the substitution in
    `test.json`.
    """
    m = _TOP.search(src)
    if m is None:
        raise RuntimeError("emitted module has no top-level `likelihood` binding")
    rhs = m.group(1)
    if not _BARE_NAME.match(rhs):
        return "likelihood", None
    return rhs, (
        f"`likelihood = {rhs}` is a bare alias, which `flatppl determinize` "
        f"refuses to score through (expected likelihoodof). Scoring {rhs} "
        f"directly -- the same measure, since the alias carries no arithmetic."
    )


def flatppl_bin() -> str:
    b = os.environ.get("FLATPPL_BIN")
    if not b:
        sys.exit(
            "FLATPPL_BIN is unset. The record shape per parameter is read off the "
            "converter's emitted `elementof` declaration, so this generator needs "
            "the same `flatppl` binary the harness scores with."
        )
    if not Path(b).exists():
        sys.exit(f"FLATPPL_BIN={b} does not exist")
    return b


def convert_source(workspace: Path) -> str:
    """The FlatPPL the converter emits for `workspace`."""
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "m.flatppl"
        proc = subprocess.run(
            [flatppl_bin(), "convert", "--from", "pyhf", str(workspace), str(out)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"convert failed for {workspace}: {proc.stderr.strip()}")
        return out.read_text()


def param_shapes(src: str) -> dict[str, int | None]:
    """Each free parameter's record shape: `None` scalar, an int vector length."""
    return {
        m.group(1): (int(m.group(3)) if m.group(3) else None)
        for m in _ELEMENTOF.finditer(src)
    }


def draw_points(model) -> list[list[float]]:
    """`suggested_init()` plus N seeded points inside the clipped bounds.

    One vectorised `rng.uniform` per point over the whole parameter array, so
    the draw order matches the audit's generator exactly. A
    `suggested_fixed()` component never moves: pyhf's own fit would hold it,
    and a degenerate bin's parameter is fixed precisely because its constraint
    carries no information.
    """
    import numpy as np

    rng = np.random.default_rng(SEED)
    init = np.array(model.config.suggested_init(), dtype=float)
    bounds = np.array(model.config.suggested_bounds(), dtype=float)
    fixed = np.array(model.config.suggested_fixed(), dtype=bool)
    lo = np.maximum(bounds[:, 0], init - CLIP)
    hi = np.minimum(bounds[:, 1], init + CLIP)
    points = [init] + [
        np.where(fixed, init, rng.uniform(lo, hi)) for _ in range(N_DRAWN_POINTS)
    ]
    return [[float(x) for x in p] for p in points]


def as_record(model, pars: list[float], shapes: dict[str, int | None]) -> dict:
    """One pyhf parameter vector as the FlatPPL `record(...)` field map."""
    rec: dict[str, object] = {}
    for pname in model.config.par_order:
        if pname not in shapes:
            raise RuntimeError(
                f"pyhf parameter {pname!r} has no emitted `elementof` binding; "
                "the converter and this generator disagree about the free parameters"
            )
        sl = model.config.par_slice(pname)
        values = [float(v) for v in pars[sl]]
        want = shapes[pname]
        if want is None:
            if len(values) != 1:
                raise RuntimeError(
                    f"{pname!r} is emitted as a scalar but pyhf gives it "
                    f"{len(values)} components"
                )
            rec[pname] = values[0]
        else:
            if len(values) != want:
                raise RuntimeError(
                    f"{pname!r} is emitted as a length-{want} vector but pyhf "
                    f"gives it {len(values)} components"
                )
            rec[pname] = values
    extra = set(shapes) - set(model.config.par_order)
    if extra:
        raise RuntimeError(
            f"the converter emits free parameters pyhf does not declare: {sorted(extra)}"
        )
    return rec


def record_to_pars(model, rec: dict) -> list[float]:
    """The inverse of `as_record`: a frozen record back to a pyhf parameter vector."""
    pars = [0.0] * model.config.npars
    for pname in model.config.par_order:
        v = rec[pname]
        values = [float(x) for x in v] if isinstance(v, list) else [float(v)]
        sl = model.config.par_slice(pname)
        if len(values) != sl.stop - sl.start:
            raise RuntimeError(
                f"frozen record field {pname!r} has {len(values)} components but "
                f"pyhf's paramset has {sl.stop - sl.start}"
            )
        pars[sl] = values
    return pars


def generate(dir: Path) -> tuple[str, list[float]]:
    import pyhf

    spec = json.loads((dir / "pyhf.json").read_text())
    ws = pyhf.Workspace(spec)
    model = ws.model()
    data = ws.data(model)

    src = convert_source(dir / "pyhf.json")
    shapes = param_shapes(src)
    binding, binding_note = score_binding_name(src)

    test_path = dir / "test.json"
    body = json.loads(test_path.read_text()) if test_path.exists() else {}
    check = next(
        (c for c in body.get("checks", []) if c.get("kind") == "logpdf_points"), None
    )

    if check is not None and check.get("points"):
        records = check["points"]
        par_points = [record_to_pars(model, r) for r in records]
    else:
        par_points = draw_points(model)
        records = [as_record(model, p, shapes) for p in par_points]

    expected = [float(model.logpdf(p, data)[0]) for p in par_points]

    envelope = {
        "test_type": "convert",
        "engines": ["det-js"],
        "fixture_kind": "pyhf",
        "source": "pyhf.json",
        "oracle": {
            "tool": "pyhf",
            "version": pyhf.__version__,
            "backend": f"{pyhf.tensorlib.name} {pyhf.tensorlib.precision}",
            "generator": "corpora/pyhf/gen_expected.py",
            "seed": SEED,
            "quantity": "pyhf.Model.logpdf(pars, workspace.data(model)) -- ABSOLUTE, no reference point subtracted",
        },
    }
    integrity = {
        "id": "static_integrity",
        "kind": "static_integrity",
        "canonical_sha256": hashlib.sha256(
            json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    logpdf_check = {
        "id": "logpdf_points",
        "kind": "logpdf_points",
        "binding": binding,
        **({"binding_note": binding_note} if binding_note else {}),
        "comparison": {
            "type": "pointwise_comp",
            "rule": "|evaluated - expected| <= atol + rtol * |expected|",
        },
        "points": records,
        "expected": expected,
        "tolerance": (check or {}).get("tolerance", TOLERANCE),
    }
    body = {**body, **envelope, "checks": [integrity, logpdf_check]}
    test_path.write_text(json.dumps(body, indent=2) + "\n")
    return dir.name, expected


def main() -> None:
    import pyhf

    pyhf.set_backend("numpy", precision="64b")
    dirs = sorted(p.parent for p in HERE.rglob("pyhf.json"))
    if not dirs:
        sys.exit(f"no pyhf fixtures under {HERE}")
    for dir in dirs:
        name, expected = generate(dir)
        print(f"{name}: {len(expected)} points, logpdf[0] = {expected[0]!r}")
    print(f"\nfroze {len(dirs)} fixtures (pyhf {pyhf.__version__})")


if __name__ == "__main__":
    main()
