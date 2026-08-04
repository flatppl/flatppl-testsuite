"""The vector family's coverage invariant, and the two arms it cannot yet close.

The verdict table records a probe's OUTCOME. It does not record which gate the
determiniser emitted to produce that outcome, so a row can claim to cover the
`cartpow` membership gate while the emitted FlatPDL contains no gate at all —
that row proves nothing. This module closes the gap from the other side: for each
targeted vector arm it asserts, against the determiniser's own emitted text, that
the arm FIRES in the probe that claims it, and that a probe which should NOT reach
an arm does not.

It also pins the two arms `flatppl-js` cannot evaluate. Those shapes are held out
of the probe family (`space._ENGINE_BLOCKED`) because a `MALFORMED` row is banned
from the committed table and is indistinguishable there from a determiniser
defect. Holding them out silently would leave the arms uncovered with nothing
saying so, so each is pinned here with its emitted arm, its derived oracle value
where one exists, and the exact crash. **When `flatppl-js` gains the missing op
the crash assertion FAILS**, which is what forces the shape back into the family
rather than letting the gap outlive its cause.
"""
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.scoring.engine import score_binding
from flatppl_testsuite.sweep.classify import Outcome, classify
from flatppl_testsuite.sweep.oracle import OracleUnsupported, true_logpdf
from flatppl_testsuite.sweep.render import render
from flatppl_testsuite.sweep.space import (
    VECTOR_BASES,
    VECTOR_INNER,
    VECTOR_SPELLINGS,
    Probe,
    Wrap,
    _ENGINE_BLOCKED,
    _vector_point_for,
    enumerate_vector_probes,
)

pytestmark = pytest.mark.skipif(
    not CONFIG.flatppl_bin.exists(), reason="needs the flatppl binary"
)

_BY_KIND = {b.kind: b for b in VECTOR_BASES}


def _probe(base_kind: str, wrap: Wrap, spelling: str = "direct") -> Probe:
    base = _BY_KIND[base_kind]
    return Probe(id=f"{base_kind}.arm", base=base, wraps=(wrap,), spelling=spelling,
                 ordering="single", consumer=False, point=_vector_point_for(base, wrap))


def _emitted(probe: Probe) -> str:
    """The determinized FlatPDL, whitespace-collapsed.

    The surface printer wraps long expressions across lines, so `iszero(sum(abs(`
    only appears contiguously once the layout is normalised — matching the raw text
    would make these assertions depend on the printer's line width rather than on
    which gate was emitted.
    """
    source = render(probe).source
    with tempfile.TemporaryDirectory() as tmp:
        model = Path(tmp) / "probe.flatppl"
        model.write_text(source)
        out = Path(tmp) / "probe.flatpdl.flatppl"
        proc = subprocess.run(
            [str(CONFIG.flatppl_bin), "determinize", str(model), "-o", str(out)],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, (
            f"{probe.id}: determinize exited {proc.returncode}: {proc.stderr.strip()}")
        return re.sub(r"\s+", "", out.read_text())


# The emitted signature of each targeted arm, as the determiniser builds it.
#
# * `cartpow_gate` — `invert.rs::Image::vector_condition`: a set-valued image over a
#   vector variate is spelled `in cartpow(S, n)` (§03), where the scalar form is
#   `in S`. `exp`'s image is `posreals`, and the length is the variate's static one.
# * `lattice_snap` — `density.rs::lattice_test`: over a vector variate the per-cell
#   differences are reduced before the exact-zero test, so the arm is
#   `iszero(sum(abs(...)))`. The scalar form has no `sum(abs(` at all.
# * `broadcast_round` — `density.rs::snap_to_lattice`: the vector spelling of the
#   snap is `real(round.(x))`, a BROADCAST round wrapped in §07 `real`.
def _cartpow_gate(base_kind: str) -> str:
    """The image gate's emitted text, with the length DERIVED from the base's own
    variate. Hardcoding `3` meant that changing a vector's length failed the positive
    assertions with "the gate did not fire", pointing at the determiniser instead of
    at a stale literal in this file."""
    return f"incartpow(posreals,{len(VECTOR_INNER[base_kind])})"


# Since determinizer 48899b0 the image comes from the BASE'S SUPPORT, not from a
# static per-op table. Multinomial's support resolves (nonneg-integer hull), so
# exp's image tightens to ge(y, 1.0), reduced over the vector as a minimum —
# the cartpow spelling survives only where support resolution falls back to the
# map's own domain (Dirichlet: no per-cell constructor support).
_GE_ONE_GATE = ("land(minimum(", ")>=1.0")


_ARM = {
    "lattice_snap": "iszero(sum(abs(",
    "broadcast_round": "real(round.(",
}


@pytest.mark.parametrize("spelling", VECTOR_SPELLINGS)
def test_the_cartpow_membership_gate_fires_where_it_is_claimed(spelling):
    """`pushfwd(exp, ·)` over a vector variate is the family's route to the gate:
    `exp`'s image is a §03 SET (`posreals`), which over a vector becomes
    `cartpow(posreals, n)`. Asserted for both bases, because the gate comes off the
    forward map's image and must not depend on the base's reference measure, and over
    every spelling in the family, because the emitted-text invariant should not be
    pinned for only the rows that happen to use `direct`.

    Multinomial's claim moved with the support-aware images (determinizer 48899b0):
    its resolved support tightens exp's image to ge(y, 1.0), emitted as a
    minimum-reduce, so the cartpow spelling now fires only for Dirichlet."""
    text = _emitted(_probe("dirichlet", Wrap("pushfwd", ("exp",)), spelling))
    assert _cartpow_gate("dirichlet") in text, (
        f"dirichlet + pushfwd(exp) [{spelling}]: the cartpow membership gate did "
        f"not fire, so nothing here covers it:\n{text}")
    text = _emitted(_probe("multinomial", Wrap("pushfwd", ("exp",)), spelling))
    for part in _GE_ONE_GATE:
        assert part in text, (
            f"multinomial + pushfwd(exp) [{spelling}]: the support-tightened ge(1.0) "
            f"image gate did not fire, so nothing here covers it:\n{text}")
    assert _cartpow_gate("multinomial") not in text, (
        f"multinomial + pushfwd(exp) [{spelling}]: both the cartpow and the ge(1.0) "
        f"gate fired — the claimed-arm map in this file is stale:\n{text}")


@pytest.mark.parametrize("spelling", VECTOR_SPELLINGS)
def test_the_lattice_snap_fires_only_over_a_counting_reference(spelling):
    """The snap is the DISCRETE arm: §08 gives Multinomial's density w.r.t.
    `iid(Counting(integers), k)`, and §06 line 28's counting measure is what puts a
    pushforward of it on the snap path. Dirichlet's Lebesgue reference must not
    reach it — that direction is what makes this assertion discriminating rather
    than trivially true."""
    text = _emitted(_probe("multinomial", Wrap("pushfwd", ("exp",)), spelling))
    assert _ARM["lattice_snap"] in text, f"lattice snap absent:\n{text}"
    assert _ARM["broadcast_round"] in text, f"broadcast round absent:\n{text}"

    text = _emitted(_probe("dirichlet", Wrap("pushfwd", ("exp",)), spelling))
    assert _ARM["lattice_snap"] not in text, (
        f"a Lebesgue-reference base reached the lattice snap:\n{text}")


@pytest.mark.parametrize("spelling", VECTOR_SPELLINGS)
def test_a_volume_preserving_pushfwd_emits_no_cartpow_gate(spelling):
    """`neg` is onto, so `invert.rs::forward_image` derives no image for it and the
    gate is correctly absent. Without this the gate assertions above could pass on a
    determiniser that gated every vector pushforward regardless of image."""
    text = _emitted(_probe("dirichlet", Wrap("pushfwd", ("neg",)), spelling))
    assert _cartpow_gate("dirichlet") not in text, (
        f"an onto forward map emitted an image gate:\n{text}")


@pytest.mark.parametrize("spelling", VECTOR_SPELLINGS)
def test_the_bare_vector_laws_emit_a_gateless_builtin_density(spelling):
    """The family's oracle-checked baseline: no gate, one `builtin_logdensityof`
    against the §08 constructor. A gate appearing here would mean the row's value is
    a gated arm's, not the base density's."""
    for kind in ("dirichlet", "multinomial"):
        text = _emitted(_probe(kind, Wrap("identity", ()), spelling))
        ctor = "Dirichlet" if kind == "dirichlet" else "Multinomial"
        assert f"builtin_logdensityof({ctor}," in text, f"{kind}:\n{text}"
        assert "ifelse(" not in text, f"{kind}: unexpected gate:\n{text}"


# --------------------------------------------------------------------------
# The held-out shapes
# --------------------------------------------------------------------------

# The crash each `_ENGINE_BLOCKED` shape currently produces, as `classify._crash_marker`
# renders it, and HOW MUCH that marker actually discriminates.
#
# The two `real` markers name the failing op, so they do discriminate: a different
# crash changes the marker and fails the pin. `crash:derivation` does NOT — it is
# what `classify._crash_marker` produces for ANY "no derivation for X" failure, with
# no diagnostic line to narrow it. For that shape the pin's guarantee is only
# "MALFORMED, and still by a bare derivation failure"; the CAUSE is pinned separately
# by `test_the_cartpow_membership_gate_is_what_flatppl_js_cannot_evaluate`, which
# scores a direct membership model where the engine does emit an identifying throw.
# Saying so here rather than letting the module docstring overclaim it.
_BLOCKED_MARKERS = {
    ("multinomial", Wrap("pushfwd", ("neg",))):
        "crash:diagnostic-real-expects-complex-array-integer",
    ("multinomial", Wrap("pushfwd", ("exp",))):
        "crash:diagnostic-real-expects-complex-array-integer",
    ("dirichlet", Wrap("pushfwd", ("exp",))):
        "crash:derivation",
}

# Markers that identify their cause, versus markers that only say "it failed".
_DISCRIMINATING_MARKERS = {
    ("multinomial", Wrap("pushfwd", ("neg",))),
    ("multinomial", Wrap("pushfwd", ("exp",))),
}


def test_the_cartpow_membership_gate_is_what_flatppl_js_cannot_evaluate():
    """Pin J2's MECHANISM somewhere the harness can actually observe it.

    Through a probe, `dirichlet + pushfwd(exp)` only ever reports
    `score_flatpdl: no derivation for 'lp'` — generic, and it names nothing. The
    membership gate in isolation DOES throw identifiably, so score that instead: a
    model whose only unusual construct is `in cartpow(posreals, n)`. If this starts
    passing, `in` has learned `cartpow` and the `dirichlet + pushfwd(exp)` hold-out
    should be re-examined even though its own generic marker cannot tell.
    """
    n = len(VECTOR_INNER["dirichlet"])
    cells = ", ".join("1.0" for _ in range(n))
    source = (f"s = cartpow(posreals, {n})\n"
              f"y = [{cells}]\n"
              "b = ifelse(y in s, 1.0, -1.0)\n")
    with tempfile.TemporaryDirectory() as tmp:
        model = Path(tmp) / "membership.flatppl"
        model.write_text(source)
        with pytest.raises(RuntimeError) as exc:
            score_binding(model, "b")
    message = str(exc.value)
    assert "score_flatpdl failed" in message, (
        f"expected a scorer failure, got: {message}")
    # Match most of the throw, not just "length": J1's own diagnostic ends in
    # `(length 3)`, so a bare substring could be satisfied by an unrelated failure.
    assert "Cannot read properties of undefined (reading 'length')" in message, (
        "flatppl-js no longer fails `in cartpow(...)` the recorded way. If it now "
        "EVALUATES the gate, J2 is fixed: re-examine the dirichlet + pushfwd(exp) "
        f"hold-out in space._ENGINE_BLOCKED. Got: {message}")


def test_a_generic_blocked_marker_is_labelled_as_such():
    """The `_DISCRIMINATING_MARKERS` bookkeeping must stay honest: a marker claimed to
    discriminate has to name something beyond the bare failure mode."""
    for shape, marker in _BLOCKED_MARKERS.items():
        if shape in _DISCRIMINATING_MARKERS:
            assert marker != "crash:derivation", (
                f"{shape} is listed as discriminating but its marker is the generic "
                "crash:derivation")
        else:
            assert marker == "crash:derivation", (
                f"{shape} is not listed as discriminating, so its marker was expected "
                f"to be the generic crash:derivation, not {marker!r} — if it now "
                "names a cause, add it to _DISCRIMINATING_MARKERS")


def test_the_blocked_list_and_the_pinned_crashes_are_the_same_set():
    assert set(_BLOCKED_MARKERS) == set(_ENGINE_BLOCKED), (
        "space._ENGINE_BLOCKED and this module's pinned crashes have drifted — a "
        "shape held out of the family with no pinned crash is an uncovered arm with "
        "nothing saying so")


@pytest.mark.parametrize(
    "shape",
    sorted(_BLOCKED_MARKERS, key=lambda s: (s[0], s[1].kind, s[1].args)),
    ids=lambda s: f"{s[0]}.{s[1].kind}_{s[1].args[0]}")
def test_a_blocked_shape_still_fails_the_way_its_recorded_reason_says(shape):
    """Every held-out shape must still be MALFORMED for the pinned reason.

    This test failing is GOOD NEWS and an action item, not a regression: it means
    `flatppl-js` can now evaluate the emitted arm, so the shape belongs back in
    `space.VECTOR_WRAPS`' generated family, with a real verdict-table row checked
    against the oracle value the next test derives.
    """
    kind, wrap = shape
    verdict = classify(_probe(kind, wrap))
    assert verdict.outcome == Outcome.MALFORMED, (
        f"{kind} + {wrap.kind}{wrap.args} now classifies {verdict.outcome} "
        f"(value={verdict.value}). Remove it from space._ENGINE_BLOCKED, regenerate "
        f"the verdict table, and delete its entry here.\n"
        f"Recorded reason: {_ENGINE_BLOCKED[shape]}")
    assert verdict.marker == _BLOCKED_MARKERS[shape], (
        f"{kind} + {wrap.kind}{wrap.args} fails differently now: marker "
        f"{verdict.marker!r}, pinned {_BLOCKED_MARKERS[shape]!r}. The recorded "
        f"reason no longer describes the failure.\n"
        f"Recorded reason: {_ENGINE_BLOCKED[shape]}")


def test_the_blocked_multinomial_arms_already_have_their_oracle_value():
    """The cartpow gate and the lattice snap are only ORACLE-blocked over Dirichlet
    (a manifold support, see `oracle._MANIFOLD_SAFE_FORWARDS`). Over Multinomial the
    reference is a counting measure, so §08's pmf at the preimage IS the value, with
    no volume term — derivable today, and held here so the number is not re-derived
    under time pressure the day the engine gap closes.
    """
    x = VECTOR_INNER["multinomial"]
    bare = true_logpdf(_probe("multinomial", Wrap("identity", ())))
    for op in ("exp", "neg"):
        got = true_logpdf(_probe("multinomial", Wrap("pushfwd", (op,))))
        assert got == pytest.approx(bare, abs=1e-12), (
            f"pushfwd({op}, Multinomial) at the forward image of {x} must equal the "
            f"pmf at {x}: a bijection does not distort a counting measure")

    with pytest.raises(OracleUnsupported):
        true_logpdf(_probe("dirichlet", Wrap("pushfwd", ("exp",))))


def test_no_blocked_shape_reached_the_generated_family():
    """The family and the held-out list must partition the wrap list, or a
    `MALFORMED` row lands in the committed table and the gate fails on regen with a
    message about a determiniser defect that does not exist."""
    generated = {(p.base.kind, p.wraps[0]) for p in enumerate_vector_probes()}
    overlap = generated & set(_ENGINE_BLOCKED)
    assert not overlap, f"blocked shapes are being generated: {sorted(overlap)}"
