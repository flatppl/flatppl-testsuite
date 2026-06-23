# flatppl-testsuite

Tests FlatPPL engines against corpora of converted foreign models.

Each case starts as a foreign model — an HS3 fixture, say. The `flatppl` converter
(`flatppl-rust`) imports it to FlatPPL, a FlatPPL engine scores the result, and the
score is checked against a frozen expected value. The loop runs in service of
co-development: a case that won't convert or won't score is a signal to fix the
converter or the engine and re-pin, not to work around.

HS3 is the first corpus, but nothing in the core knows that — corpora and engines
both plug in.

## Layout

```
pixi.toml                        single entry point (envs, tasks)
pytest.ini                       collection pinned to tests/
scripts/setup.sh                 installs the pinned flatppl converter
src/flatppl_testsuite/
  config.py                      generic config (binaries, scorer path)
  scoring/                       engine seam + 2DeltaNLL + compare
    engine.py                    FlatpplEngine ABC + registry + selector
    score_js.cjs                 the flatppl-js single-point scorer
  formats/                       Importer / Exporter / ForeignEngine ABCs
    hs3/                         HS3 importer + oracles
  suites/                        Suite ABC + registry (hs3_import)
  runner.py                      thin orchestration
corpora/hs3/                     vendored, self-contained HS3 corpus
  manifest.json  ATTRIBUTION.md
  fixtures/<id>/                 hs3.json, expected.json, metadata.json, model.flatppl
  conversions/<model>/           <model>.hs3.json, .flatppl, _root.py
  run_comparisons.py             formatted comparison tables
  tests/                         the corpus's own pytest definitions
tests/
  core/                          toolkit tests (engine seam, architecture, ABCs)
  test_hs3.py                    shim: re-exports corpora/hs3/tests
```

## Extending

**A new corpus** lives under `corpora/<name>/` with its own `tests/`. Add a `Suite`
in `suites/` to drive it and a one-line `tests/test_<name>.py` shim that re-exports
the corpus tests. The core stays corpus-agnostic.

**A new FlatPPL engine** subclasses `FlatpplEngine` in `scoring/engine.py` and calls
`register_engine(...)`. Select it with `FLATPPL_ENGINE` (default `js`); suites,
comparisons, and the 2DeltaNLL math are all engine-agnostic.

## Dependencies

Everything runs through pixi.

`pixi run setup` provisions both. The **converter** (`flatppl-rust`) is installed
via `cargo install --git …/flatppl-rust --branch <ref> --features hs3 flatppl-cli`;
bump `FLATPPL_RUST_REF` and re-run to pull changes. The default **engine**
(`flatppl-js`) is resolved at scoring time from `FLATPPL_JS_DIR` (default: sibling
`../flatppl-js`), whose `.ts` Node 24 loads directly. Setup clones that location at
`FLATPPL_JS_REF` when it is missing and leaves an existing checkout untouched — so a
co-development checkout with your own engine changes is never clobbered. The **oracles**, ROOT (conda-forge `root`) and
pyHS3 (pip `pyhs3`), live in their own pixi environments (`-e root`, `-e pyhs3`) and
subprocess the upstream HS3TestSuite checkout named by `HS3SUITE`.

## Use

```sh
pixi run setup                 # install the pinned converter
pixi run test                  # pytest: toolkit + every corpus via its shim
pixi run harness               # numeric harness over the default suite
pixi run hs3                   # run the HS3 corpus, print comparison tables
```
