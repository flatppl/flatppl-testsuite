"""Determinize a probe and classify the outcome. The only unit here that runs
subprocesses.

Three outcomes, because two are not enough: a determiniser can exit 0 and
still emit something that is not FlatPDL (a surviving `(%call ...)`), and no
existing gate rejects that. Scoring it would report a tolerance failure for
the wrong reason, or worse, a pass.

`MALFORMED` covers two shapes now, both "exit 0, not actually evaluable":

1. A residual `%call` in the printed output (see the known-limitation note
   below -- structurally unreachable through the CLI, kept anyway).
2. The determinizer emits FlatPDL that type-checks syntactically but is
   wrongly typed for the builtin/op it feeds (e.g. `log` applied to a whole
   record instead of a scalar) -- the JS scorer throws evaluating it.
   `score_binding` wraps that as a `RuntimeError("score_flatpdl failed: ...")`.
   That is NOT the "a scorer crash fails the run loudly" infrastructure case
   (missing binary, unresolvable engine, missing Node) -- the scorer started
   fine and ran; it choked on the SHAPE of what determinize handed it. Routing
   it into a bare `RuntimeError` re-raise would abort the entire sweep on a
   single bad probe. Distinguished by message prefix: `score_flatpdl failed`
   means the scorer subprocess ran and threw; anything else re-raises.

**Known limitation, not a bug in this module**: `_RESIDUAL_CALL` scans the
*printed* `-o` file, i.e. surface FlatPPL text re-derived from the
determinized module. Per `flatppl-dev/TODO-flatppl-rust.md`
("How NOT to bound that change: a CLI text comparison cannot see it"), the
surface printer renders a residual `CallHead::User` exactly like any other
call -- verified there end to end: a FlatPIR `(%call log 0.5)` prints as
`v = log(0.5)` and RE-PARSES to a builtin call, so the `%call` marker does
not survive the round trip. That means this regex can never fire against
real CLI output; it is not "currently unexercised", it is structurally blind
to its own namesake bug when read through `determinize -o`. It is kept
anyway -- harmless, matches the interface, and a future output path (a
different print mode, an error message) could yet make it live -- but no
Python-side text scan over `flatppl`'s CLI output can currently observe a
surviving user call. Seeing one requires the in-memory `Module` (the gates
`is_flatpdl`/`infer` already use inside flatppl-rust's own test suite), which
is out of reach from here.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.scoring.engine import DeterminizeRefused, score_binding
from flatppl_testsuite.sweep.render import render
from flatppl_testsuite.sweep.space import Probe


class Outcome(str, Enum):
    LOWERS = "LOWERS"
    REFUSES = "REFUSES"
    MALFORMED = "MALFORMED"


# A residual user call. FlatPDL is deterministic ops plus the six `builtin_*`
# primitives; a `%call` is neither. See the module docstring: this can only
# ever match synthetic text, not real `determinize -o` output.
_RESIDUAL_CALL = re.compile(r"\(%call\s")


@dataclass(frozen=True)
class Verdict:
    probe_id: str
    outcome: str
    value: float | None = None
    marker: str | None = None


def _marker(stderr: str) -> str:
    """A stable slice of a refusal reason, for the table to diff on.

    The full reason is prose and will be reworded; the leading `refuse <head>`
    plus the first few significant words is stable enough to detect a CHANGED
    refusal without churning on every message edit.
    """
    line = stderr.strip().splitlines()[0] if stderr.strip() else ""
    line = line.replace("determinize: ", "")
    head = re.match(r"refuse (\S+)", line)
    words = re.findall(r"[a-z]{4,}", line.lower())[:6]
    return f"{head.group(1) if head else '?'}:{'-'.join(words)}"


def _crash_marker(message: str) -> str:
    """A stable slice of a scorer-crash message, for the table to diff on.

    Same shape as `_marker` (a short word-list, not full prose), but for the
    JS scorer throwing on emitted FlatPDL it can't evaluate -- there is no
    `refuse <head>` line here, so this always keys off the literal `crash`
    prefix rather than a parsed head.

    **Residual risk, stated rather than hidden**: this cannot by itself tell
    a determinizer defect (wrong-shaped output) apart from a `flatppl-js` bug
    on genuinely valid FlatPDL. The marker narrows a human's triage; it does
    not replace it.
    """
    line = message.strip().splitlines()[0] if message.strip() else ""
    for prefix in ("score_flatpdl failed: ", "score_flatpdl: "):
        if line.startswith(prefix):
            line = line[len(prefix):]
    words = re.findall(r"[a-z]{4,}", line.lower())[:6]
    return f"crash:{'-'.join(words)}"


def classify_source(source: str, binding: str) -> Verdict:
    """Determinize `source` and classify what came back.

    `probe_id` is left empty here (the caller doesn't know one); `classify`
    fills it in.
    """
    with tempfile.TemporaryDirectory() as tmp:
        model = Path(tmp) / "probe.flatppl"
        model.write_text(source)
        out = Path(tmp) / "probe.flatpdl.flatppl"
        det = subprocess.run(
            [str(CONFIG.flatppl_bin), "determinize", str(model), "-o", str(out)],
            capture_output=True, text=True,
        )
        if det.returncode == 3:
            return Verdict("", Outcome.REFUSES, None, _marker(det.stderr))
        if det.returncode != 0:
            # Not a probe verdict: the tool itself failed. Fail loudly.
            raise RuntimeError(
                f"determinize exited {det.returncode} (neither 0 nor 3): "
                f"{det.stderr.strip()}"
            )
        emitted = out.read_text() if out.exists() else det.stdout
        if _RESIDUAL_CALL.search(emitted):
            return Verdict("", Outcome.MALFORMED, None, "residual-user-call")
        try:
            value = score_binding(model, binding)
        except DeterminizeRefused as e:
            return Verdict("", Outcome.REFUSES, None, _marker(str(e)))
        except RuntimeError as e:
            msg = str(e)
            if not msg.startswith("score_flatpdl failed"):
                raise  # "determinize failed": infrastructure, fail loudly
            return Verdict("", Outcome.MALFORMED, None, _crash_marker(msg))
        return Verdict("", Outcome.LOWERS, float(value), None)


def classify(probe: Probe) -> Verdict:
    r = render(probe)
    v = classify_source(r.source, r.binding)
    return Verdict(probe.id, v.outcome, v.value, v.marker)
