"""The persistent Node score worker: one process serves every request, scores as
the one-shot scripts score, keeps requests apart, and survives a bad request, a
hung request and a dead process."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.scoring import worker

pytestmark = pytest.mark.skipif(
    not (CONFIG.flatppl_js_dir / "packages" / "engine" / "index.ts").exists()
    or shutil.which(CONFIG.node_bin) is None and not Path(CONFIG.node_bin).exists(),
    reason="requires Node and a flatppl-js checkout",
)

_ENGINE = str(CONFIG.flatppl_js_dir / "packages" / "engine")
_NORMAL = 'flatppl_compat = "0.1"\noutputs = {x} * 2.0 + 1.0\n'
_LOGPDF = 1.0  # outputs at x = 0.0


def _one_shot(script: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([CONFIG.node_bin, str(script), *args, "--engine", _ENGINE],
                          capture_output=True, text=True)


@pytest.fixture()
def private_worker():
    w = worker.ScoreWorker()
    yield w
    w.close()


def _flatpdl(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


def test_one_process_serves_every_request(tmp_path, private_worker):
    model = _flatpdl(tmp_path, "m.flatppl", _NORMAL.format(x="0.0"))
    for _ in range(5):
        assert worker.score_binding_file(model, "outputs", private_worker) == _LOGPDF
        rows = worker.score_batch_rows([_NORMAL.format(x="1.0")], "outputs", None, private_worker)
        assert rows == [{"ok": True, "value": 3.0}]
    assert private_worker.started == 1


def test_the_worker_scores_a_binding_as_the_one_shot_script_does(tmp_path, private_worker):
    model = _flatpdl(tmp_path, "m.flatppl", _NORMAL.format(x="0.3"))
    one_shot = _one_shot(CONFIG.flatpdl_scorer, str(model), "outputs")
    assert one_shot.returncode == 0
    assert worker.score_binding_file(model, "outputs", private_worker) == float(one_shot.stdout)


def test_a_failing_binding_reports_the_one_shot_scripts_stderr(tmp_path, private_worker):
    model = _flatpdl(tmp_path, "m.flatppl", _NORMAL.format(x="0.3"))
    one_shot = _one_shot(CONFIG.flatpdl_scorer, str(model), "missing")
    assert one_shot.returncode != 0
    with pytest.raises(RuntimeError) as raised:
        worker.score_binding_file(model, "missing", private_worker)
    assert str(raised.value) == f"score_flatpdl failed: {one_shot.stderr.strip()}"
    # The worker is still the same live process and still scores.
    assert worker.score_binding_file(model, "outputs", private_worker) == 0.3 * 2.0 + 1.0
    assert private_worker.started == 1


def test_a_source_with_an_error_diagnostic_names_it_first_like_the_script(tmp_path, private_worker):
    model = _flatpdl(tmp_path, "m.flatppl", "outputs = nope(1.0)\n")
    one_shot = _one_shot(CONFIG.flatpdl_scorer, str(model), "outputs")
    with pytest.raises(RuntimeError) as raised:
        worker.score_binding_file(model, "outputs", private_worker)
    assert str(raised.value) == f"score_flatpdl failed: {one_shot.stderr.strip()}"


def test_score_js_matches_the_one_shot_script_and_its_errors(tmp_path, private_worker):
    model = _flatpdl(tmp_path, "m.flatppl", "m = Normal(mu = 0.0, sigma = 1.0)\n")
    one_shot = _one_shot(CONFIG.scorer, str(model), "m", "0.5")
    assert one_shot.returncode == 0, one_shot.stderr
    assert worker.score_js_file(model, "m", "0.5", private_worker) == float(one_shot.stdout)
    one_shot = _one_shot(CONFIG.scorer, str(model), "missing", "0.5")
    assert one_shot.returncode != 0
    with pytest.raises(RuntimeError) as raised:
        worker.score_js_file(model, "missing", "0.5", private_worker)
    assert str(raised.value) == f"score_js failed: {one_shot.stderr.strip()}"
    assert private_worker.started == 1


def test_requests_do_not_leak_state_into_each_other(private_worker):
    a, b = _NORMAL.format(x="0.5"), _NORMAL.format(x="2.0")
    first = worker.score_batch_rows([a], "outputs", None, private_worker)
    other = worker.score_batch_rows([b], "outputs", None, private_worker)
    assert worker.score_batch_rows([a], "outputs", None, private_worker) == first
    assert first != other
    bad = worker.score_batch_rows(["outputs = nope(1.0)\n"], "outputs", None, private_worker)
    assert bad[0]["ok"] is False
    assert worker.score_batch_rows([a, b], "outputs", None, private_worker) == first + other


def _determinized_sample_model(tmp_path: Path) -> Path:
    source = _flatpdl(tmp_path, "rand.flatppl", (
        "s = rnginit([42, 0, 0, 0])\n"
        "first, s2 = rand(Normal(0.0, 1.0), s)\n"
        "second, s3 = rand(Normal(0.0, 1.0), s2)\n"))
    out = tmp_path / "rand.flatpdl.flatppl"
    subprocess.run([str(CONFIG.flatppl_bin), "determinize", str(source), "-o", str(out)],
                   check=True, capture_output=True)
    return out


@pytest.mark.skipif(not CONFIG.flatppl_bin.exists(), reason="requires a determinize-capable flatppl binary")
def test_sample_sweep_matches_the_one_shot_script_and_rejects_bad_arguments(tmp_path, private_worker):
    model = _determinized_sample_model(tmp_path)
    one_shot = _one_shot(CONFIG.sample_sweep_scorer, str(model), "5", "first,second", "--base", "7")
    assert one_shot.returncode == 0, one_shot.stderr
    assert worker.sample_sweep_rows(model, 5, ["first", "second"], 7, private_worker) == json.loads(one_shot.stdout)
    for n, base in ((0, 0), (1.5, 0), (1, -1), (1, 1 << 32), (2, (1 << 32) - 1)):
        with pytest.raises(RuntimeError, match="sample_sweep failed: sample_sweep: .*\nusage:"):
            worker.sample_sweep_rows(model, n, ["first"], base, private_worker)
    with pytest.raises(RuntimeError, match="no rnginit"):
        worker.sample_sweep_rows(_flatpdl(tmp_path, "plain.flatppl", "x = 1.0\n"), 1, ["x"], 0, private_worker)
    assert private_worker.started == 1


def test_a_dead_worker_is_replaced_for_the_next_request(tmp_path, private_worker):
    model = _flatpdl(tmp_path, "m.flatppl", _NORMAL.format(x="0.0"))
    assert worker.score_binding_file(model, "outputs", private_worker) == _LOGPDF
    os.kill(private_worker._process.pid, signal.SIGKILL)
    private_worker._process.wait()
    assert worker.score_binding_file(model, "outputs", private_worker) == _LOGPDF
    assert private_worker.started == 2


def test_a_worker_that_dies_mid_request_fails_that_request_only(tmp_path):
    script = _flatpdl(tmp_path, "die.cjs", (
        "require('readline').createInterface({input: process.stdin}).on('line', () => process.exit(3));\n"
        "process.stdout.write(JSON.stringify({ready: true}) + '\\n');\n"))
    dying = worker.ScoreWorker(script=script)
    try:
        with pytest.raises(worker.ScoreWorkerFailed, match="score_flatpdl failed: score worker exited with code 3"):
            dying.request("binding", "score_flatpdl", model="m", binding="b")
    finally:
        dying.close()


def test_a_hung_request_times_out_and_does_not_hang_the_run(tmp_path):
    script = _flatpdl(tmp_path, "hang.cjs", (
        "process.stdout.write(JSON.stringify({ready: true}) + '\\n');\n"
        "require('readline').createInterface({input: process.stdin}).on('line', () => {});\n"
        "setInterval(() => {}, 1000);\n"))
    hanging = worker.ScoreWorker(script=script)
    started = time.monotonic()
    try:
        with pytest.raises(worker.ScoreWorkerFailed, match="timed out"):
            hanging.request("binding", "score_flatpdl", timeout=1.0, model="m", binding="b")
        assert time.monotonic() - started < 30
        assert hanging._process is None  # killed, so the next request starts clean
    finally:
        hanging.close()


def test_a_worker_over_the_memory_limit_is_recycled(tmp_path, monkeypatch, private_worker):
    model = _flatpdl(tmp_path, "m.flatppl", _NORMAL.format(x="0.0"))
    monkeypatch.setattr(worker, "_RSS_LIMIT", 1)
    assert worker.score_binding_file(model, "outputs", private_worker) == _LOGPDF
    assert worker.score_binding_file(model, "outputs", private_worker) == _LOGPDF
    assert private_worker.started == 2


def test_the_shared_worker_is_one_process_across_scoring_calls(tmp_path):
    model = _flatpdl(tmp_path, "m.flatppl", _NORMAL.format(x="0.0"))
    shared = worker.shared()
    assert worker.score_binding_file(model, "outputs") == _LOGPDF
    assert shared.loaded_engine == _ENGINE  # the engine the configuration names
    before = shared.started
    for _ in range(3):
        assert worker.score_binding_file(model, "outputs") == _LOGPDF
    assert shared.started - before <= 1
