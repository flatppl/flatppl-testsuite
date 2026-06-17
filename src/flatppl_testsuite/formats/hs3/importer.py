"""Stage 1: convert an HS3 fixture to FlatPPL via the flatppl-rust CLI.

Classifies the converter's exit into one of:
  - success: returns the emitted FlatPPL source.
  - SkipUnimplemented: the converter refused an unimplemented HS3 construct;
    the harness reports this as SKIP, naming the construct.

Unimplemented-type detection is based on the observed converter stderr:
    flatppl: hs3: unsupported HS3 distribution type: <typename>

Also provides Stage 2: assemble a scoreable FlatPPL source from converter
output + a check (observations, assemble).
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ...config import CONFIG
from ..base import Importer

# Substring that identifies an out-of-scope HS3 construct in the converter's
# stderr (observed from flatppl convert on rf207_comptools/hs3.json).
_UNIMPL_MARKER = "unsupported HS3 distribution type:"

# Captures the HS3 type name following the marker.
_TYPE_RE = re.compile(r"unsupported HS3 distribution type:\s+(\S+)")


@dataclass
class SkipUnimplemented(Exception):
    """The converter does not yet implement an HS3 construct in this fixture."""

    hs3_type: str
    detail: str = ""


def convert(hs3_json: Path) -> str:
    """Run `flatppl convert --from hs3` and return the emitted FlatPPL source.

    Raises SkipUnimplemented when the converter reports an out-of-scope
    construct, so the runner can mark the fixture SKIP with the named type.
    """
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "m.flatppl"
        proc = subprocess.run(
            [str(CONFIG.flatppl_bin), "convert", "--from", "hs3",
             str(hs3_json), str(out)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            stderr = proc.stderr
            if _UNIMPL_MARKER in stderr:
                m = _TYPE_RE.search(stderr)
                raise SkipUnimplemented(
                    hs3_type=m.group(1) if m else "unknown",
                    detail=stderr.strip(),
                )
            raise RuntimeError(f"convert failed: {stderr.strip()}")
        # The converter can emit an `error[...]`-level diagnostic (e.g. an
        # unresolved reference) yet still exit 0 and write a file. That file is
        # not trustworthy — treat an emitted error as a conversion failure so the
        # runner tags it UNSCOREABLE at the convert stage with the real reason,
        # rather than letting it resurface as a confusing downstream score error.
        if "error[" in proc.stderr:
            first = next((ln.strip() for ln in proc.stderr.splitlines()
                          if "error[" in ln), proc.stderr.strip())
            raise RuntimeError(f"convert emitted an error diagnostic: {first}")
        return out.read_text()


# ---------------------------------------------------------------------------
# Stage 2: assemble a scoreable FlatPPL source from converter output + a check.
#
# Takes the converter's emitted module and a check `target` (pdf, data), reads
# the observed dataset from the HS3 fixture, and appends an iid likelihood over
# the converter's emitted pdf measure:
#
#     __obs__ = [ ... ]                     # observations from the HS3 data block
#     __M__   = normalize(truncate(<pdf>, interval(lo, hi)))
#     __L__   = likelihoodof(iid(__M__, lengthof(__obs__)), __obs__)
#
# The pdf binding is referenced BY NAME; the engine accepts relabel'd measures
# directly in truncate/iid/normalize (PR #38). No RHS extraction is required.
#
# Scoring `logdensityof(__L__, record(<free params>))` reproduces the suite's
# frozen 2DeltaNLL vector.
# ---------------------------------------------------------------------------


def observations(hs3_json: Path, data_name: str) -> list[float]:
    """Read the unbinned observation values for `data_name` from the HS3 file."""
    doc = json.loads(Path(hs3_json).read_text())
    for d in doc.get("data", []):
        if d.get("name") == data_name:
            return [float(e[0]) for e in d["entries"]]
    raise KeyError(f"dataset {data_name!r} not in {hs3_json}")


def _strip_provenance(src: str) -> str:
    """Drop the converter's leading `%`-comment provenance header."""
    return "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("%"))


# The observable's declared range lives in a `cartprod(..., x = interval(lo, hi))`
# domain binding. Match the interval keyed by the observable label.
def _observable_interval(src: str, observable: str) -> tuple[float, float] | None:
    m = re.search(
        rf"{re.escape(observable)}\s*=\s*interval\(\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)",
        src)
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def assemble(flatppl_src: str, pdf: str, obs: list[float],
             observables: set[str], prenormalized: bool = False) -> tuple[str, str]:
    """Return (scoreable_source, binding_name) for an iid likelihood over `pdf`.

    `pdf` is the name of the converter's emitted pdf binding; the engine
    accepts relabel'd measures directly in `truncate`/`iid`/`normalize`, so the
    binding is referenced by name rather than by extracting its RHS.

    A raw distribution (e.g. gaussian) is range-normalized here —
    `normalize(truncate(pdf, interval))` over the observable's declared domain —
    to reproduce the HS3 pdf. A `prenormalized` pdf (generic-family dists, which
    the importer already emits as `normalize(truncate(weighted(...), interval))`)
    is iid'd directly; re-wrapping it would double-normalize and the engine
    cannot resolve a `normalize(...)` node as a truncate base.
    """
    body = _strip_provenance(flatppl_src)

    if prenormalized:
        measure = pdf
    else:
        # Find the declared interval for the single observable that has one.
        interval = None
        for obs_name in observables:
            iv = _observable_interval(body, obs_name)
            if iv is not None:
                interval = iv
                break
        if interval is not None:
            lo, hi = interval
            measure = f"normalize(truncate({pdf}, interval({lo!r}, {hi!r})))"
        else:
            # No declared range → score the bare (full-support) measure.
            measure = pdf

    arr = "[" + ", ".join(repr(float(v)) for v in obs) + "]"

    extra = (
        f"\n__obs__ = {arr}"
        f"\n__M__ = {measure}"
        f"\n__L__ = likelihoodof(iid(__M__, lengthof(__obs__)), __obs__)\n"
    )
    return body + extra, "__L__"


# ---------------------------------------------------------------------------
# ABC-conforming face
# ---------------------------------------------------------------------------


class HS3Importer(Importer):
    """HS3 -> FlatPPL via the flatppl converter CLI."""

    def import_(self, source: Path | str) -> str:
        return convert(Path(source))
