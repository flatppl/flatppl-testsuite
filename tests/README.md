# Tests

The pytest entry point. Collection is pinned here (`pytest.ini` → `testpaths =
tests`), so everything runs from:

```sh
pixi run test          # python -m pytest -q
```

`conftest.py` at the repo root puts `src/` and the repo root on the path, so tests
can `import flatppl_testsuite` and `import corpora.<name>.tests.*`.

## Layout

- **`core/`** — toolkit tests for the corpus-agnostic machinery: the engine seam
  (`test_engine.py`), the layer-dependency guard (`test_architecture.py`), and the
  format/suite ABCs.
- **`test_<corpus>.py`** — one shim per corpus. Each is a few lines that re-export
  the corpus's own test definitions (e.g. `test_hs3.py` pulls in
  `corpora/hs3/tests/*`). The tests themselves live with the corpus they exercise;
  the shim is only what makes pytest discover them from here.

Add a corpus → add `tests/test_<corpus>.py` re-exporting `corpora/<corpus>/tests`.

A few tests need the converter binary; it lands at `.pixi-bin/bin/flatppl` after
`pixi run setup` (and is on `PATH` inside the pixi environment). The numeric
comparison tables are a separate, non-pytest path: `pixi run hs3`.
