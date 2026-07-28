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
