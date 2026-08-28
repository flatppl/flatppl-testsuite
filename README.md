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
  suites/
    hs3_import.py                shared HS3 helper library (convert/assemble/score
                                  helpers reused by the unified convert runner)
  unified/                       the unified per-test-directory harness
    loader.py                    load + validate a test.json
    harness.py                   run_test_dir: dispatch to the right runner
    regen.py                     offline oracle -> frozen expected/stat
    runners/                     one runner per (test_type, engine) pair
corpora/<name>/<test_id>/        one directory per test case, everywhere:
  test.json                       test_type, engines, frozen expected/stat, tolerances
  test.py                         independent oracle (oracle()/grad_oracle()/stat())
  model.flatppl (+ query.flatppl / hs3.json / ...)   per test_type
corpora/hs3/                      the one corpus with a foreign-format source
  ATTRIBUTION.md                  source, commit, license
  fixtures/<id>/, conversions/<model>/   test dirs (test_type "convert")
  conversions/gen_expected.py     regenerates the conversions' frozen ROOT
                                  vectors from the ROOT/RooFit oracle (root env)
tests/
  core/                           toolkit tests (engine seam, architecture, ABCs,
                                   the unified loader)
  test_unified.py                 discovers + runs every corpora/**/test.json dir
  test_prenormalized_structural.py, test_skip_classify.py, test_strip_provenance.py
                                   converter-internals unit tests
```

## Extending

**A new test case** is a new `corpora/<corpus>/<test_id>/` directory: a `test.json`
(`test_type`, `engines`, frozen `expected`/`stat`, tolerances) and a `test.py`
independent oracle. `tests/test_unified.py` discovers it automatically — no
registration step. `pixi run regen <dir>` (`-e stablehlo` for stablehlo-only dirs)
freezes the oracle's value into `test.json`.

**A new `test_type` or engine** gets its own runner under
`src/flatppl_testsuite/unified/runners/`, wired into `unified/harness.py`'s dispatch.

**A new FlatPPL engine** subclasses `FlatpplEngine` in `scoring/engine.py` and calls
`register_engine(...)`. Select it with `FLATPPL_ENGINE` (default `js`); the unified
harness and the 2DeltaNLL math are all engine-agnostic.

## Dependencies

Everything runs through pixi.

`pixi run setup` provisions both. The **converter** (`flatppl-rust`) is installed
via `cargo install --git …/flatppl-rust --features hs3,stablehlo flatppl-cli` at
`FLATPPL_RUST_REF` (a branch or a 40-hex commit); bump the ref and re-run to pull
changes. The default **engine**
(`flatppl-js`) is resolved at scoring time from `FLATPPL_JS_DIR` (default: sibling
`../flatppl-js`), whose `.ts` Node 24 loads directly. Setup clones that location at
`FLATPPL_JS_REF` when it is missing and leaves an existing checkout untouched — so a
co-development checkout with your own engine changes is never clobbered. The **oracles**, ROOT (conda-forge `root`) and
pyHS3 (pip `pyhs3`), live in their own pixi environments (`-e root`, `-e pyhs3`) and
subprocess the upstream HS3TestSuite checkout named by `HS3SUITE`.

## Use

```sh
pixi run setup                        # install the pinned converter + resolve the JS engine
pixi run test                         # pytest: tests/core/ + the unified harness (det-js cases)
pixi run unified                      # the unified harness alone
pixi run convert <in.json> <out.flatppl>   # convert one fixture (debugging aid)

# StableHLO cases (separate env; needs a `flatppl` built with --features stablehlo):
FLATPPL_BIN=/path/to/flatppl PYTHONPATH=$PWD/src pixi run -e stablehlo unified
# Regenerate a test dir's frozen expected/stat from its test.py oracle:
pixi run -e stablehlo regen corpora/stablehlo/linear_regression
```

## Engine pins

The two sweep tables under `verdicts/` are frozen against one determiniser
commit (`density-sweep.json`, `metadata.determinizer_commit`) and one engine
commit (`sampler-sweep.json`, `metadata.engine_commit`), and their gates assert
that the build under test is that one. CI reads both pins with
`scripts/engine-pins.py` and builds the engines at them, so provenance holds by
construction and a merge in `flatppl-js` or `flatppl-rust` cannot redden this
repo.

```sh
pixi run setup     # only when the DETERMINISER pin is the one to move
pixi run repin     # re-pin both tables to the engines now configured
```

`repin` regenerates both tables and commits only when no verdict moved; a moved
verdict is printed and nothing is committed, because a re-pin must never be the
way a behavioural change gets frozen. Locally, a branch determiniser or engine
fails the two provenance gates by construction — regenerate, read the diff,
then discard the regenerated table (`git checkout -- verdicts/`).
