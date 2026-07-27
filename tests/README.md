# Tests

The pytest entry point. Collection is pinned here (`pytest.ini` → `testpaths =
tests`), so everything runs from:

```sh
pixi run test          # python -m pytest -q  (tests/core/ + the unified harness, det-js cases)
pixi run unified        # the unified harness alone
```

`conftest.py` at the repo root puts `src/` and the repo root on the path, so tests
can `import flatppl_testsuite`.

## Layout

- **`core/`** — toolkit tests for the corpus-agnostic machinery: the engine seam
  (`test_engine.py`), the layer-dependency guard (`test_architecture.py`), the
  format ABCs (`test_formats_base.py`), the unified `test.json` loader
  (`test_unified_loader.py`), and the det-js engine end-to-end path
  (`test_det_js_engine.py`).
- **`test_unified.py`** — the unified per-test-directory harness: discovers every
  `corpora/**/test.json` directory (`unified/loader.py::discover_test_dirs`) and
  parametrizes one pytest case per `(dir, engine)` pair. This is where every
  corpus's actual test cases live now — there is no per-corpus test file or shim;
  add a case by adding a `corpora/<corpus>/<test_id>/` directory (`test.json` +
  `test.py`), not by touching `tests/`.
- **`test_prenormalized_structural.py`**, **`test_skip_classify.py`**,
  **`test_strip_provenance.py`** — unit tests of specific HS3-importer internals
  (prenormalized-binding detection, the stderr skip-classifier, provenance-comment
  stripping), independent of any one test directory.

A few tests need the converter binary; it lands at `.pixi-bin/bin/flatppl` after
`pixi run setup` (and is on `PATH` inside the pixi environment). StableHLO cases
need the separate `stablehlo` pixi env and a `flatppl` built with that feature —
see the root README's "Use" section.
