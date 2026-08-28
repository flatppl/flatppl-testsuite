"""The probe space is pure data, so it is testable with no engine present."""
import math
import re
import subprocess
import tempfile
from pathlib import Path

import pytest

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.sweep.curated import _args, _as_call
from flatppl_testsuite.sweep.space import (
    BASES,
    INNER,
    VECTOR_BASES,
    VECTOR_INNER,
    Base,
    Probe,
    Wrap,
    enumerate_probes,
    in_support,
    is_literal,
    is_shared_latent,
    is_vector_base,
)
from flatppl_testsuite.sweep.render import render


def test_enumeration_is_deterministic_and_ids_are_unique():
    a = enumerate_probes()
    b = enumerate_probes()
    assert [p.id for p in a] == [p.id for p in b], "enumeration must be stable"
    assert len({p.id for p in a}) == len(a), "probe ids must be unique"
    assert len(a) >= 200, f"space too small to be a sweep: {len(a)}"


def test_every_probe_renders_to_a_binding_it_defines():
    for p in enumerate_probes():
        r = render(p)
        assert f"{r.binding} =" in r.source, f"{p.id}: binding absent from source"
        assert "logdensityof" in r.source, f"{p.id}: not a density query"


def test_a_known_probe_renders_the_expected_source():
    p = Probe(
        id="normal.pushfwd_exp.direct.single.noconsumer",
        base=Base("normal", (0.0, 1.0)),
        wraps=(Wrap("pushfwd", ("exp",)),),
        spelling="direct",
        ordering="single",
        consumer=False,
        point=1.6487212707001282,
    )
    r = render(p)
    assert r.binding == "lp"
    assert "pushfwd(exp, Normal(mu = 0.0, sigma = 1.0))" in r.source
    assert "logdensityof(m, 1.6487212707001282)" in r.source


def _corpus_ctor_keywords() -> dict[str, tuple[str, ...]]:
    """`{constructor: (keyword, ...)}` as spelled in the committed corpus models."""
    out: dict[str, tuple[str, ...]] = {}
    root = Path(__file__).resolve().parents[2] / "corpora"
    for model in sorted(root.glob("*/*/*.flatppl")):
        for raw in model.read_text().splitlines():
            m = re.match(r"^[A-Za-z_]\w*\s*=\s*draw\((.+)\)$", raw.strip())
            if not m:
                continue
            call = _as_call(m.group(1))
            if not call:
                continue
            _pos, kw = _args(call[1])
            if kw:
                out.setdefault(call[0], tuple(sorted(kw)))
    return out


def test_base_constructors_use_the_parameter_names_the_corpus_uses():
    """A wrong distribution parameter name is caught by NOTHING else in the
    toolchain: `Poisson(lambda = 3.0)` parses, converts to `(%kwarg lambda 3.0)`,
    and determinizes with no diagnostic at all — the emitted keyword simply has to
    be right. So pin `render`'s keywords against the committed corpus models,
    which carry §08's names and whose numbers are gated on frozen oracles.

    §08's parameter names for the four bases: `Normal(mu, sigma)`,
    `Gamma(shape, rate)`, `Beta(alpha, beta)`, `Poisson(rate)`.
    """
    corpus = _corpus_ctor_keywords()
    assert corpus, "found no corpus constructors to compare against"

    checked = 0
    for base in BASES:
        probe = Probe(id="t", base=base, wraps=(Wrap("identity", ()),),
                      spelling="direct", ordering="single", consumer=False,
                      point=INNER[base.kind])
        rhs = render(probe).source.splitlines()[0].split("=", 1)[1].strip()
        call = _as_call(rhs)
        assert call, f"{base.kind}: rendered constructor did not parse: {rhs!r}"
        head, argstr = call
        _pos, kw = _args(argstr)
        assert head in corpus, (
            f"{head} appears in no corpus model, so this test cannot verify its "
            "parameter names — add a corpus case or pin them another way")
        assert tuple(sorted(kw)) == corpus[head], (
            f"{head}: render emits {tuple(sorted(kw))}, the corpus uses "
            f"{corpus[head]} — a wrong parameter name is silent all the way "
            "through the determiniser")
        checked += 1
    assert checked == len(BASES)


# §08 "Multivariate distributions"' parameter names for the vector family, quoted
# from the entries themselves: `Dirichlet(alpha)` and `Multinomial(n, p)`. Only
# `Dirichlet` appears in the committed corpus (`corpora/stablehlo/dirichlet`), so
# `Multinomial`'s names are pinned against §08 directly — there is no corpus model
# to compare them with, and the test above would otherwise silently not cover it.
_VECTOR_CTOR_KEYWORDS = {
    "Dirichlet": ("alpha",),
    "Multinomial": ("n", "p"),
}


def test_vector_base_constructors_use_sec08s_parameter_names():
    corpus = _corpus_ctor_keywords()
    checked = 0
    for base in VECTOR_BASES:
        probe = Probe(id="t", base=base, wraps=(Wrap("identity", ()),),
                      spelling="direct", ordering="single", consumer=False,
                      point=VECTOR_INNER[base.kind])
        rhs = render(probe).source.splitlines()[0].split("=", 1)[1].strip()
        call = _as_call(rhs)
        assert call, f"{base.kind}: rendered constructor did not parse: {rhs!r}"
        head, argstr = call
        _pos, kw = _args(argstr)
        assert head in _VECTOR_CTOR_KEYWORDS, f"{head}: no §08 names recorded"
        assert tuple(sorted(kw)) == tuple(sorted(_VECTOR_CTOR_KEYWORDS[head])), (
            f"{head}: render emits {tuple(sorted(kw))}, §08 gives "
            f"{_VECTOR_CTOR_KEYWORDS[head]}")
        if head in corpus:
            # Where a corpus model does exist, it must agree with §08 too —
            # otherwise one of the two is wrong and this test would hide it.
            assert tuple(sorted(kw)) == corpus[head], (
                f"{head}: §08 names {tuple(sorted(kw))} disagree with the corpus's "
                f"{corpus[head]}")
        checked += 1
    assert checked == len(VECTOR_BASES)


def test_every_vector_probe_point_is_on_its_supports_constraint_surface():
    """A vector base's support is a surface, not a box (§08: Multinomial's is
    {x in N_0^k : sum x_i = n}, Dirichlet's is `stdsimplex(n)`). A point off it has
    density 0, so a probe generated there would be a -inf row proving nothing about
    the arm it claims to cover.

    Checked through the PREIMAGE, which is what makes it "derived, not hardcoded":
    every wrap in the family is either point-preserving or an invertible elementwise
    map, so the preimage must be `VECTOR_INNER[base.kind]` exactly.
    """
    inverse = {"exp": math.log, "neg": lambda y: -y}
    for p in enumerate_probes():
        if is_shared_latent(p) or is_literal(p) or not is_vector_base(p.base):
            continue
        w = p.wraps[0]
        pre = list(p.point)
        if w.kind == "pushfwd":
            pre = [inverse[w.args[0]](c) for c in pre]
        assert pre == pytest.approx(VECTOR_INNER[p.base.kind], abs=1e-9), (
            f"{p.id}: preimage {pre} != VECTOR_INNER[{p.base.kind}] "
            f"({VECTOR_INNER[p.base.kind]}) — point looks hardcoded, not derived")
        assert in_support(p.base, pre), (
            f"{p.id}: preimage {pre} not in {p.base.kind}'s support")


def _wrap_marker(wrap: Wrap) -> str:
    """A substring that only appears in the rendered source if this wrap's
    operator was actually applied — used to catch a spelling that silently
    drops a wrap, the general form of the bug where `record` built its law
    from the unwrapped base while the query point stayed wrap-adjusted."""
    if wrap.kind == "pushfwd":
        return f"pushfwd({wrap.args[0]}"
    if wrap.kind == "affine":
        return "x ->"
    return f"{wrap.kind}("


def test_every_wrap_is_present_across_all_three_spellings():
    """The spelling axis exists to assert direct/stochastic_node/record denote
    the SAME measure, so a wrap that one spelling composes and another drops
    is a correctness bug in the sweep itself, not a cosmetic gap — it would
    manufacture false "the determiniser is wrong" findings once Tasks 2-4
    start comparing spellings against each other and the oracle.

    The shared-latent family has no wrap axis; its own equivalent — that every
    spelling of one measure renders the measure it claims — is
    `test_shared_latent.test_each_spelling_renders_the_construct_it_names`.
    """
    for p in enumerate_probes():
        if is_shared_latent(p) or is_literal(p):
            continue
        w = p.wraps[0]
        if w.kind == "identity":
            continue
        r = render(p)
        marker = _wrap_marker(w)
        assert marker in r.source, (
            f"{p.id}: wrap operator {marker!r} missing from the "
            f"{p.spelling!r} spelling's source"
        )


# Inverse of each `pushfwd` forward map, for recovering the preimage of a
# probe's query point.
_INVERSE = {
    "exp": math.log,
    "log": math.exp,
    "neg": lambda y: -y,
    "sqrt": lambda y: y * y,
}


def test_pushfwd_points_are_derived_not_hardcoded():
    """A `pushfwd` probe's point must be `forward(INNER[base])` -- derived,
    not a per-operator constant that happens to land in some base's support
    by luck (that was fix round 1's bug: `log`/`sqrt` reused `gamma`'s inner
    point for `beta`, landing outside `beta`'s support).

    Checking `in_support` alone would not catch a wrong-but-in-support
    constant, so also pin the preimage to `INNER[base]` exactly -- that is
    what forces "derived", not just "happens to work"."""
    for p in enumerate_probes():
        if is_shared_latent(p) or is_literal(p):
            continue          # no wrap axis, so no pushforward point to derive
        w = p.wraps[0]
        if w.kind != "pushfwd" or is_vector_base(p.base):
            # The vector family has its own version of this check, cell-wise:
            # `test_every_vector_probe_point_is_on_its_supports_constraint_surface`.
            continue
        preimage = _INVERSE[w.args[0]](p.point)
        assert in_support(p.base, preimage), (
            f"{p.id}: preimage {preimage} not in {p.base.kind}'s support"
        )
        assert preimage == pytest.approx(INNER[p.base.kind], abs=1e-9), (
            f"{p.id}: preimage {preimage} != INNER[{p.base.kind}] "
            f"({INNER[p.base.kind]}) -- point looks hardcoded, not derived"
        )


@pytest.mark.skipif(not CONFIG.flatppl_bin.exists(), reason="needs the flatppl binary")
def test_every_rendered_model_parses():
    """A probe whose source does not parse measures the sweep's bug, not the
    determiniser's. `convert` to the same format canonicalizes, so a zero exit
    is a parse."""
    bad = []
    for p in enumerate_probes():
        r = render(p)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "m.flatppl"
            src.write_text(r.source)
            out = Path(tmp) / "out.flatppl"
            proc = subprocess.run(
                [str(CONFIG.flatppl_bin), "convert", str(src), str(out)],
                capture_output=True, text=True,
            )
            if proc.returncode != 0:
                bad.append((p.id, proc.stderr.strip().splitlines()[:1]))
    assert not bad, "probes that do not parse:\n" + "\n".join(map(str, bad))
