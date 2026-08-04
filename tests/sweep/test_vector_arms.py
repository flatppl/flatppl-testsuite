"""The vector family's coverage invariant, and the one arm still open.

The verdict table records a probe's OUTCOME. It does not record which gate the
determiniser emitted to produce that outcome, so a row can claim to cover the
`cartpow` membership gate while the emitted FlatPDL contains no gate at all —
that row proves nothing. This module closes the gap from the other side: for each
targeted vector arm it asserts, against the determiniser's own emitted text, that
the arm FIRES in the probe that claims it, and that a probe which should NOT reach
an arm does not.

**Both target arms are now oracle-checked.** The `cartpow` image gate and the
discrete lattice snap were held out while `flatppl-js` could evaluate neither
`real` over an integer array nor `in` over a `cartpow` set; `e9803b6` fixed both,
the failing-when-fixed crash pins fired as designed, and the two
`multinomial + pushfwd` shapes are back in the generated family with real
verdict-table rows checked against §08's pmf.

One hold-out remains, and its cause is NOT an engine gap:
`dirichlet + pushfwd(exp)` scores fine — the sweep has nothing to check the number
against, because §06 scopes `Lebesgue` to lower-dimensional embedded **affine**
sets and `exp`'s image of the simplex is curved. It retires on a spec ruling
(`flatppl-dev/measure-algebra-audit.md`), not on an engine release, and the pin
here fails if the emitted value, the oracle's withhold, or the recorded category
moves. The distinction matters: an engine-gap hold-out and an oracle-gap hold-out
retire on different events, so `space._HELD_OUT` records the category per shape.
"""
import math
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from flatppl_testsuite.config import CONFIG
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
    _HELD_OUT,
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
# The reinstated shapes: both target arms now score
# --------------------------------------------------------------------------

# The value every `Multinomial` probe in the family must produce, whatever wrap it
# carries. §08 gives the density w.r.t. `iid(Counting(integers), k)`, and §06 line 28's
# counting measure is not distorted by a bijection, so a pushforward carries NO volume
# term and lands on the bare pmf at the preimage. Independently: the log-gamma closed
# form log(5!/(1!2!2!) * 0.2 * 0.3^2 * 0.5^2) = log(0.135) = -2.0024805005437063.
_MULTINOMIAL_LOGPMF = -2.0024805005437063


@pytest.mark.parametrize("op", ["neg", "exp"])
@pytest.mark.parametrize("spelling", VECTOR_SPELLINGS)
def test_a_multinomial_pushfwd_scores_the_counting_reference_value(op, spelling):
    """The two arms this family exists for, now checked against a number rather than
    against a crash.

    `pushfwd(exp, Multinomial)` emits BOTH the `cartpow` image gate and the discrete
    lattice snap; `pushfwd(neg, Multinomial)` emits the snap alone (`neg` is onto, so
    it has no image gate). Both were held out while `flatppl-js` could evaluate
    neither `real` over an integer array nor `in` over a `cartpow` set. Both now score,
    and the value is the bare pmf — which is the whole point of the counting
    reference, and the thing a spurious volume term would break: subtracting
    `sum(log y)` here would be -5.0, not a rounding error.
    """
    probe = _probe("multinomial", Wrap("pushfwd", (op,)), spelling)
    verdict = classify(probe)
    assert verdict.outcome == Outcome.LOWERS, (
        f"multinomial + pushfwd({op}) [{spelling}] no longer scores: "
        f"{verdict.outcome} (marker={verdict.marker})")
    assert verdict.value == pytest.approx(_MULTINOMIAL_LOGPMF, abs=1e-12)
    # And the oracle agrees, derived from §08 rather than from the engine.
    assert true_logpdf(probe) == pytest.approx(_MULTINOMIAL_LOGPMF, abs=1e-12)


def test_the_reinstated_arms_are_in_the_generated_family():
    """The reinstatement is what makes the arms *covered*; the assertions above would
    still pass with the shapes held out, so pin membership too."""
    generated = {(p.base.kind, p.wraps[0]) for p in enumerate_vector_probes()}
    for op in ("neg", "exp"):
        assert ("multinomial", Wrap("pushfwd", (op,))) in generated, (
            f"multinomial + pushfwd({op}) is not generated, so no verdict-table row "
            "covers its arm")


def test_the_engine_gap_category_is_empty():
    """`_ENGINE_BLOCKED` held both Multinomial pushforwards until flatppl-js gained
    `real` over integer arrays and `in` over `cartpow`. Nothing should be blocked on
    the engine now. A future entry here is fine — but it must be a deliberate
    addition, not a silent regression of these two."""
    assert _ENGINE_BLOCKED == {}, (
        "a shape is held out on an ENGINE gap again: "
        f"{sorted(_ENGINE_BLOCKED)}. If that is intended, pin its crash the way the "
        "Multinomial shapes used to be pinned, so it retires when the engine is fixed")


# --------------------------------------------------------------------------
# The one remaining hold-out, and why it is NOT an engine gap
# --------------------------------------------------------------------------

# The bare Dirichlet log-density at the family's point, under §08's formula.
_DIRICHLET_BARE = 2.0228711901914425
# The ambient-R^3 Jacobian the determiniser subtracts for `pushfwd(exp, ·)` over a
# vector variate: `sum(log y)` at `y = exp(x)` is `sum(x)`, and on `stdsimplex(n)`
# that is exactly 1. So the emitted value is the bare law minus 1, to the bit.
_DIRICHLET_AMBIENT_LOGVOL = 1.0
_DIRICHLET_PUSHFWD_EXP_EMITTED = _DIRICHLET_BARE - _DIRICHLET_AMBIENT_LOGVOL


def test_the_dirichlet_pushfwd_exp_holdout_is_an_oracle_gap_not_an_engine_gap():
    """The hold-out that survived wave J, re-pinned on its actual cause.

    It is no longer blocked on the engine: `flatppl-js` scores this shape now. It is
    held out because §06 scopes `Lebesgue` to lower-dimensional embedded **affine**
    sets, and `exp`'s image of `stdsimplex(n)` is a CURVED 2-manifold — so no §06 rule
    says which measure the emitted number is a density against. The ambient Jacobian,
    the Hausdorff area element and the coordinate chart give 1.0, 0.6816 and 0.5 for
    the same volume term. Generating the probe would add a `LOWERS` row with
    `oracle = None`: a row no gate compares, which looks covered and is not.

    Three assertions, so the pin fails on any of the three ways this can move:

    1. the engine still evaluates it, and to the AMBIENT reading (a different reading
       would mean the determiniser changed its mind about the volume term);
    2. the oracle still withholds (if it stops, the spec question was ruled on and the
       shape must be reinstated with a real oracle value);
    3. the shape is still out of the generated family.

    It retires on a SPEC RULING, recorded in
    `flatppl-dev/measure-algebra-audit.md`, not on an engine release.
    """
    probe = _probe("dirichlet", Wrap("pushfwd", ("exp",)))

    verdict = classify(probe)
    assert verdict.outcome == Outcome.LOWERS, (
        f"the engine stopped scoring this shape ({verdict.outcome}, "
        f"marker={verdict.marker}). It scored at flatppl-js e9803b6; a regression "
        "here is an engine bug, not a reason to re-record the hold-out")
    assert verdict.value == pytest.approx(_DIRICHLET_PUSHFWD_EXP_EMITTED, abs=1e-12), (
        f"the determiniser now emits {verdict.value} for pushfwd(exp, Dirichlet), not "
        f"the ambient-Jacobian {_DIRICHLET_PUSHFWD_EXP_EMITTED}. If it moved to the "
        "Hausdorff (0.6816) or chart (0.5) volume term, the manifold question has "
        "been decided somewhere -- check flatppl-dev/measure-algebra-audit.md and "
        "reinstate this shape with the matching oracle rule")

    with pytest.raises(OracleUnsupported, match="affine"):
        true_logpdf(probe)

    generated = {(p.base.kind, p.wraps[0]) for p in enumerate_vector_probes()}
    assert ("dirichlet", Wrap("pushfwd", ("exp",))) not in generated

    category, reason = _HELD_OUT[("dirichlet", Wrap("pushfwd", ("exp",)))]
    assert category == "oracle", (
        "the recorded category still says the engine is at fault; it is not, the "
        "engine scores this shape")
    assert "affine" in reason.lower(), (
        "the reason no longer names the §06 affine scoping that justifies the "
        "withhold")
    assert "measure-algebra-audit" in reason, (
        "the reason no longer points at where the open spec question is tracked")


def test_the_emitted_ambient_volume_term_is_exactly_one_on_the_simplex():
    """Why the pinned value is `bare - 1.0` rather than an opaque constant: for
    `pushfwd(exp, ·)` the per-cell forward log-volume at the preimage is `log y_i`, so
    the sum is `sum(x_i)`, and every point of `stdsimplex(n)` sums to 1. Derived here
    so the literal above cannot drift from its own justification."""
    x = VECTOR_INNER["dirichlet"]
    assert sum(x) == pytest.approx(1.0, abs=1e-15)
    ambient = math.fsum(math.log(math.exp(c)) for c in x)
    assert ambient == pytest.approx(_DIRICHLET_AMBIENT_LOGVOL, abs=1e-12)
    bare = true_logpdf(_probe("dirichlet", Wrap("identity", ())))
    assert bare == pytest.approx(_DIRICHLET_BARE, abs=1e-12)


def test_no_held_out_shape_reached_the_generated_family():
    """The family and the hold-out list must partition the wrap list, or an unchecked
    row lands in the committed table."""
    generated = {(p.base.kind, p.wraps[0]) for p in enumerate_vector_probes()}
    overlap = generated & set(_HELD_OUT)
    assert not overlap, f"held-out shapes are being generated: {sorted(overlap)}"


def test_every_held_out_shape_records_a_known_category_and_a_reason():
    for shape, entry in _HELD_OUT.items():
        category, reason = entry
        assert category in ("engine", "oracle"), (
            f"{shape}: unknown hold-out category {category!r}")
        assert reason and len(reason) > 80, (
            f"{shape}: a hold-out needs a reason a triager can act on")
