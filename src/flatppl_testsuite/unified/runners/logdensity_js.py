"""Score the same ABI query directly with the JS engine, without determinizing."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from flatppl_testsuite.config import CONFIG
from flatppl_testsuite.scoring import worker
from flatppl_testsuite.scoring.engine import temporary_source_dir
from flatppl_testsuite.scoring.result import UNSCOREABLE, CheckResult
from flatppl_testsuite.unified import detjs_exec
from flatppl_testsuite.unified.loader import TestSpec
from flatppl_testsuite.unified.runners import logdensity_detjs


def engine_available() -> bool:
    return (shutil.which(CONFIG.node_bin) is not None
            and CONFIG.score_worker.exists()
            and (CONFIG.flatppl_js_dir / "packages" / "engine" / "index.ts").exists())


def score_abi_points(model: Path, query: Path, fields: list[str], points: list[dict]):
    source = model.read_text().rstrip() + "\n" + query.read_text().lstrip()
    sources = detjs_exec.bind_abi_sources(detjs_exec.strip_docs(source), fields, points)
    scores = []
    for src in sources:
        with tempfile.NamedTemporaryFile(
            "w", suffix=".flatppl", prefix=".abi-", dir=temporary_source_dir(model, src)
        ) as file:
            file.write(src)
            file.flush()
            try:
                scores.append(detjs_exec.PointScore(worker.score_binding_file(Path(file.name), "outputs")))
            except RuntimeError as error:
                scores.append(detjs_exec.PointScore(None, str(error)))
    return scores


def run(spec: TestSpec, dir: Path) -> list[CheckResult]:
    if not (dir / "query.flatppl").exists() or spec.body.get("points") is None:
        return [CheckResult(dir.name, "logdensity", "failed", UNSCOREABLE,
                            "native JS requires an inputs/outputs query and points")]
    return logdensity_detjs.run(spec, dir, abi_scorer=score_abi_points)
