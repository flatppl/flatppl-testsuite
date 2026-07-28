"""The probe space is pure data, so it is testable with no engine present."""
import subprocess
import tempfile
from pathlib import Path

import pytest

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.sweep.space import Base, Probe, Wrap, enumerate_probes
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
    start comparing spellings against each other and the oracle."""
    for p in enumerate_probes():
        w = p.wraps[0]
        if w.kind == "identity":
            continue
        r = render(p)
        marker = _wrap_marker(w)
        assert marker in r.source, (
            f"{p.id}: wrap operator {marker!r} missing from the "
            f"{p.spelling!r} spelling's source"
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
