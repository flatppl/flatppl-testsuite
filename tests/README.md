# Tests

## `conversion/` — conversion tests that must not fail

Drop conversion tests here (the set that should always pass). They are
discovered and run by:

```sh
pixi run test
```

which runs `pytest -q` from the repo root. `PYTHONPATH` already points at
`src/`, so tests can `import flatppl_testsuite`.

These assert that specific HS3 fixtures convert to the expected FlatPPL — the
green baseline the harness must never regress. Numeric scoring of the converted
models (the `twice_delta_nll_scan` path) is a separate layer driven by
`python -m flatppl_testsuite`; see `docs/` for the design and plan.

If a test needs the converter binary, it is at `.pixi-bin/bin/flatppl` after
`pixi run setup` (also on `PATH` inside the pixi environment).
