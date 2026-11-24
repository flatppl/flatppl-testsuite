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

No hold-out remains. `dirichlet + pushfwd(exp)` was held out while no §06 rule
said which measure its number was a density against. By §03 and §06 a law
on `stdsimplex(n)` keeps a density only under a uniform affine map. `exp`'s image
of the simplex is curved, so the law has no density, the determiniser refuses,
and the shape is a generated row with a conformant refusal. The
`in cartpow(posreals, n)` image gate is probed through the full-dimensional
`MvNormal` base instead: Multinomial's support tightens `exp`'s image to `ge(1.0)`.
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
from flatppl_testsuite.sweep.table import _spec_justified

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
    "lattice_snap": "iszero(sum(abs.(",
    "broadcast_round": "real.(round.(",
}


@pytest.mark.parametrize("spelling", VECTOR_SPELLINGS)
def test_the_image_gate_fires_where_it_is_claimed(spelling):
    """`pushfwd(fn(exp.(_)), ·)` over a vector variate is the family's route to the
    image gate. Asserted over every spelling in the family, because the emitted-text
    invariant should not be pinned for only the rows that happen to use `direct`.

    MvNormal's support is `cartpow(reals, n)` (§08), so exp's image is the set
    `cartpow(posreals, n)` and the membership gate fires. Multinomial's resolved
    support tightens exp's image to ge(y, 1.0), emitted as a minimum-reduce
    (determinizer 48899b0), so the `cartpow` spelling does not fire there.
    Dirichlet's map has no density and refuses
    (`test_a_non_affine_map_of_a_simplex_law_refuses`)."""
    text = _emitted(_probe("mvnormal", Wrap("pushfwd", ("exp",)), spelling))
    assert _cartpow_gate("mvnormal") in text, (
        f"mvnormal + pushfwd(exp) [{spelling}]: the cartpow membership gate did not "
        f"fire, so nothing here covers it:\n{text}")
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
    than trivially true. The Dirichlet side uses `neg`, the one map of a simplex law
    that keeps a density."""
    text = _emitted(_probe("multinomial", Wrap("pushfwd", ("exp",)), spelling))
    assert _ARM["lattice_snap"] in text, f"lattice snap absent:\n{text}"
    assert _ARM["broadcast_round"] in text, f"broadcast round absent:\n{text}"

    text = _emitted(_probe("dirichlet", Wrap("pushfwd", ("neg",)), spelling))
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
    ctors = {"dirichlet": "Dirichlet", "multinomial": "Multinomial", "mvnormal": "MvNormal"}
    for kind, ctor in ctors.items():
        text = _emitted(_probe(kind, Wrap("identity", ()), spelling))
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

    `pushfwd(fn(exp.(_)), Multinomial)` emits BOTH the image gate and the discrete
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


# scipy 1.18.1 `multivariate_normal([0.1, -0.2, 0.3], cov).logpdf([0.4, -0.1, 0.2])`
# for the family's MvNormal base.
_MVNORMAL_LOGPDF = -3.303072655578398


@pytest.mark.parametrize("spelling", VECTOR_SPELLINGS)
def test_the_mvnormal_exp_row_carries_a_nonzero_cell_wise_volume(spelling):
    """§08 gives MvNormal's support as `cartpow(reals, n)`, so `fn(exp.(_))` has
    the per-cell volume Σ log y, subtracted. At y = exp(0.4, −0.1, 0.2) that is 0.5,
    nonzero, so a sign error or a per-cell/whole-vector confusion in the sum
    changes the value. Both the determiniser's value and the oracle must equal
    log MvNormal(log y) − Σ log y."""
    probe = _probe("mvnormal", Wrap("pushfwd", ("exp",)), spelling)
    volume = sum(math.log(c) for c in probe.point)
    assert abs(volume - 0.5) < 1e-12
    expected = _MVNORMAL_LOGPDF - volume
    assert true_logpdf(probe) == pytest.approx(expected, abs=1e-12)
    verdict = classify(probe)
    assert verdict.outcome == Outcome.LOWERS, f"{verdict.outcome} (marker={verdict.marker})"
    assert verdict.value == pytest.approx(expected, abs=1e-12)


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
# A non-affine map of a simplex law: no density
# --------------------------------------------------------------------------

# The determiniser's refusal for the image of a law on a lower-dimensional support.
_NO_DENSITY = "the image of a law on a lower-dimensional support has no density"


def _determinize(probe: Probe) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as tmp:
        model = Path(tmp) / "probe.flatppl"
        model.write_text(render(probe).source)
        return subprocess.run(
            [str(CONFIG.flatppl_bin), "determinize", str(model),
             "-o", str(Path(tmp) / "out.flatppl")],
            capture_output=True, text=True,
        )


@pytest.mark.parametrize("spelling", VECTOR_SPELLINGS)
def test_a_non_affine_map_of_a_simplex_law_refuses(spelling):
    """`exp` maps `stdsimplex(n)` onto a curved surface. §06 gives `Lebesgue` on an
    embedded set only for an affine one, and the curved image is singular against
    ambient Lebesgue, so the law has no density (§06 "Singular laws"). The engine
    used to score it as the bare law minus an ambient-R^3 Jacobian of 1.0, while the
    Hausdorff element and the (y1, y2) chart give 0.6816 and 0.5 for the same term.

    Four assertions: the determiniser refuses with the lower-dimensional text (exit 3,
    a refusal rather than an inference error), the oracle withholds, the shape is in
    the generated family, and its refusal row is conformant."""
    probe = _probe("dirichlet", Wrap("pushfwd", ("exp",)), spelling)
    proc = _determinize(probe)
    assert proc.returncode == 3 and _NO_DENSITY in proc.stderr, (
        f"dirichlet + pushfwd(exp) [{spelling}] no longer refuses as having no "
        f"density: exit {proc.returncode}: {proc.stderr.strip()}")

    with pytest.raises(OracleUnsupported, match="affine"):
        true_logpdf(probe)

    generated = {(p.base.kind, p.wraps[0]) for p in enumerate_vector_probes()}
    assert ("dirichlet", Wrap("pushfwd", ("exp",))) in generated

    assert _spec_justified(probe, Outcome.REFUSES) is True


def test_no_shape_is_held_out():
    """Both hold-out categories are empty. A future entry must be a deliberate
    addition with a category and a reason."""
    assert _HELD_OUT == {}, f"held-out shapes: {sorted(_HELD_OUT)}"


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
