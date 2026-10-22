#!/usr/bin/env node
'use strict';
// Score a FlatPPL binding's log-density at a point, using the flatppl-js engine.
// The JS-engine counterpart to the `*_root.py` oracles in this directory.
//
// Usage:
//   node score_js.cjs <model.flatppl> <binding> <theta> [--engine <dir>] [--count N]
//
//   <binding>  a measure / likelihood binding in the model (e.g. `obs`, `likelihood`).
//   <theta>    a FlatPPL record literal of the binding's free parameters, e.g.
//              'record(mu = 0.0, sigma = 1.0)'. The script appends
//              `__score__ = logdensityof(<binding>, <theta>)` to the model source
//              and prints the resulting per-atom log-density to stdout.
//
// Engine resolution (first that exists): --engine <dir>, then $FLATPPL_JS_DIR
// (its packages/engine), then the cache clone at ~/.cache/flatppl-js. Requires
// Node >= 24 — the engine modules are TypeScript, loaded via Node's native type
// stripping. The scoring itself lives in score_core.cjs, shared with the
// persistent worker (score_worker.cjs).
//
// ---------------------------------------------------------------------------
// Reproduce the ROOT / closed-form comparison for the bundled HS3 examples.
// Each line shows the FlatPPL JS-engine value and the oracle from the matching
// *_root.py.
//
//   # gaussian (HS3 paper A.1) — agrees with ROOT on the absolute log-density
//   node score_js.cjs gaussian.flatppl obs 'record(mu = 0.0, sigma = 1.0)'
//   #   -> -1.7253885332   (python gaussian_root.py: logL @ mu=0 = -1.7253885332)
//
//   # histfactory (A.3) — Δ(logL) matches ROOT; the absolute differs by a
//   # parameter-independent constant (ROOT's extended NLL drops the log(n!) term)
//   node score_js.cjs histfactory.flatppl L_model_channel1 \
//        'record(mu = 1.0, syst1 = 0.0, syst2 = 0.0, syst3 = 0.0, mcstat = [1.0, 1.0])'   # -> -16.5292632794
//   node score_js.cjs histfactory.flatppl L_model_channel1 \
//        'record(mu = 1.5, syst1 = 0.5, syst2 = 0.0, syst3 = 0.0, mcstat = [1.1, 1.0])'   # -> -19.8529754084
//   #   Δ = -3.3237121290   (python histfactory_root.py: Δ(logL) = -3.3237121290)
//
//   # product_dist (A.2) — exact analytic Gaussian-product normalizer
//   node score_js.cjs product.flatppl likelihood \
//        'record(mu1 = 0.0, sigma1 = 1.0, mu2 = 1.0, sigma2 = 2.0)'
//   #   -> -13.9458491571  (python product_root.py: -13.9458508897; ROOT integrates
//   #                       the normalizer numerically, hence the ~1e-6 difference)
// ---------------------------------------------------------------------------

const core = require('./score_core.cjs');

function usage(msg) {
  if (msg) process.stderr.write('score_js: ' + msg + '\n');
  process.stderr.write(
    'usage: node score_js.cjs <model.flatppl> <binding> <theta> [--engine <dir>] [--count N]\n');
  process.exit(2);
}

function parseArgs(argv) {
  const pos = [];
  let engineDir = null;
  let count = 1;
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--engine') { engineDir = argv[++i]; }
    else if (a === '--count') { count = parseInt(argv[++i], 10); }
    else if (a.startsWith('--')) { usage('unknown flag ' + a); }
    else { pos.push(a); }
  }
  if (pos.length !== 3) usage('expected <model.flatppl> <binding> <theta>');
  return { model: pos[0], binding: pos[1], theta: pos[2], engineDir, count };
}

async function main() {
  const { model, binding, theta, engineDir, count } = parseArgs(process.argv.slice(2));
  let dir;
  try { dir = core.resolveEngine(engineDir); } catch (e) {
    if (e instanceof core.EngineNotFound) usage(e.message);
    throw e;
  }
  const engine = core.loadEngine(dir);
  const value = await core.scoreJs(engine, { model, binding, theta, count }, (line) => process.stderr.write(line));
  process.stdout.write(value + '\n');
}

main().catch((e) => {
  process.stderr.write('score_js: ' + (e && e.message ? e.message : e) + '\n');
  process.exit(1);
});
