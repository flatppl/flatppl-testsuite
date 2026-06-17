#!/usr/bin/env node
'use strict';
// Full JS-engine reproduction of the three HS3 examples.
// The JS-engine counterpart to running each *_root.py individually.
//
// Usage:
//   node repro_hs3_js.cjs [--engine <dir>] [--model gaussian|histfactory|product]
//
// Without --model, runs all three. repro_hs3.sh drives one model at a time so
// each FlatPPL table is printed alongside the matching ROOT oracle table.
//
// Engine resolution (first that exists):
//   1. --engine <dir>
//   2. $FLATPPL_JS_DIR (its packages/engine sub-directory)
//   3. ~/.cache/flatppl-js  (cloned from https://github.com/flatppl/flatppl-js
//      by repro_hs3.sh on first run)
//
// Requires Node >= 24 (native TypeScript type-stripping).
//
// Each row prints the FlatPPL JS log-density. Rows tagged PASS are checked
// against the ROOT oracle (gaussian: tol 1e-8; histfact Δ: 1e-6; product: 2e-4
// for ROOT's numeric integrator). Untagged rows are intermediate values shown
// for reference — the histfact absolute log-densities differ from ROOT by a
// constant Σlog(obs_k!) offset (~1049) that ROOT drops by HEP convention, so
// only Δ(logL) is directly comparable.

const fs   = require('fs');
const path = require('path');

// ---------------------------------------------------------------------------
// Engine resolution
// ---------------------------------------------------------------------------

function parseArgs(argv) {
  let engineDir = null;
  let model = null;
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--engine') engineDir = argv[++i];
    else if (argv[i] === '--model') model = argv[++i];
  }
  return { engineDir, model };
}

function resolveEngine(explicit) {
  const candidates = [
    explicit,
    process.env.FLATPPL_JS_DIR &&
      path.join(process.env.FLATPPL_JS_DIR, 'packages', 'engine'),
    // GitHub clone cached at ~/.cache/flatppl-js
    path.join(require('os').homedir(), '.cache', 'flatppl-js', 'packages', 'engine'),
  ].filter(Boolean);
  for (const c of candidates) {
    if (fs.existsSync(path.join(c, 'index.ts'))) return c;
  }
  process.stderr.write(
    'repro_hs3_js: flatppl-js engine not found.\n' +
    'Pass --engine <dir>, set FLATPPL_JS_DIR, or run repro_hs3.sh to clone it.\n' +
    '(looked in: ' + candidates.join(', ') + ')\n',
  );
  process.exit(2);
}

// ---------------------------------------------------------------------------
// Scoring helper — appends `__score__ = logdensityof(<binding>, <theta>)` and
// evaluates it. Returns the scalar log-density.
// ---------------------------------------------------------------------------

async function score(engineDir, modelSrc, binding, thetaLiteral) {
  const { processSource, orchestrator, materialiser } =
    require(path.join(engineDir, 'index.ts'));
  const { createWorkerHandler } = require(path.join(engineDir, 'worker.ts'));

  const src = modelSrc + `\n__score__ = logdensityof(${binding}, ${thetaLiteral})\n`;
  const proc = processSource(src);
  for (const d of proc.diagnostics || []) {
    if (d.severity === 'error' && !d.message.includes('cartprod()'))
      process.stderr.write('engine diagnostic: ' + d.message + '\n');
  }

  const built = orchestrator.buildDerivations(proc.bindings);
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 3 });
  const cache = new Map();
  const ctx = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    sampleCount: 1,
    rootKey: 3,
    rootSeed: 3,
    marginalizationCount: 32,
    moduleRegistry: proc.loweredModule && proc.loweredModule.moduleRegistry,
    getMeasure(n) {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, m);
      return m;
    },
    sendWorker: (m) => Promise.resolve(w.handle(m)),
  };

  const measure = await ctx.getMeasure('__score__');
  if (!measure || !measure.samples || measure.samples.length === 0)
    throw new Error('no density produced for `' + binding + '`');
  return measure.samples[0];
}

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

const COL_MODEL   = 10;
const COL_LABEL   = 36;
const COL_VALUE   = 16;

function pad(s, n) { return String(s).padEnd(n); }
function lpad(s, n) { return String(s).padStart(n); }

let passed = 0, failed = 0;

function show(model, label, got) {
  console.log(pad(model, COL_MODEL) + pad(label, COL_LABEL) + lpad(got.toFixed(10), COL_VALUE));
}

function check(model, label, got, expected, tol) {
  const diff = Math.abs(got - expected);
  const ok   = diff <= tol;
  const mark = ok ? 'PASS' : 'FAIL';
  console.log(
    pad(model, COL_MODEL) +
    pad(label, COL_LABEL) +
    lpad(got.toFixed(10), COL_VALUE) + '   ' + mark +
    (ok ? '' : `  (expected ${expected.toFixed(10)}, |Δ|=${diff.toExponential(2)})`),
  );
  if (ok) passed++; else failed++;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function runGaussian(engine, hs3Dir) {
  const src = fs.readFileSync(path.join(hs3Dir, 'gaussian', 'gaussian.flatppl'), 'utf8');
  for (const [mu, expected] of [
    [0.0,  -1.7253885332],
    [0.5,  -1.2153885332],
    [1.27, -0.9189385332],
  ]) {
    const got = await score(engine, src, 'obs', `record(mu = ${mu}, sigma = 1.0)`);
    check('gaussian', `obs @ mu=${mu},sigma=1.0`, got, expected, 1e-8);
  }
}

async function runHistfact(engine, hs3Dir) {
  const src = fs.readFileSync(path.join(hs3Dir, 'histfactory', 'histfactory.flatppl'), 'utf8');
  const HF_POINTS = [
    ['record(mu = 1.0, syst1 = 0.0,  syst2 = 0.0,  syst3 = 0.0,  mcstat = [1.0, 1.0])', -16.5292632794],
    ['record(mu = 1.5, syst1 = 0.5,  syst2 = 0.0,  syst3 = 0.0,  mcstat = [1.1, 1.0])', -19.8529754084],
    ['record(mu = 0.5, syst1 = -0.3, syst2 = 0.2,  syst3 = 0.0,  mcstat = [0.9, 1.1])', -21.3094646795],
    ['record(mu = 2.0, syst1 = 1.0,  syst2 = -1.0, syst3 = 0.5,  mcstat = [1.2, 0.8])', -31.3969487578],
    ['record(mu = 0.0, syst1 = 0.0,  syst2 = 0.0,  syst3 = 0.0,  mcstat = [1.0, 1.0])', -19.4472333464],
  ];
  const HF_ROOT_DELTA = [0, -3.3237121291, -4.7802014001, -14.8676854784, -2.9179700670];
  const scores = [];
  for (const [theta] of HF_POINTS) {
    const v = await score(engine, src, 'L_model_channel1', theta);
    scores.push(v);
    const label = theta.replace('record(', '').replace(')', '').slice(0, 34);
    show('histfact', label, v);
  }
  for (let i = 1; i < scores.length; i++) {
    check('histfact', `Δ(pt0→pt${i}) vs ROOT`, scores[i] - scores[0], HF_ROOT_DELTA[i], 1e-6);
  }
}

async function runProduct(engine, hs3Dir) {
  const src = fs.readFileSync(path.join(hs3Dir, 'product', 'product.flatppl'), 'utf8');
  const def  = 'record(mu1 = 0.0, sigma1 = 1.0, mu2 = 1.0, sigma2 = 2.0)';
  const pert = 'record(mu1 = 0.5, sigma1 = 1.0, mu2 = 1.0, sigma2 = 2.0)';
  const v0 = await score(engine, src, 'likelihood', def);
  const v1 = await score(engine, src, 'likelihood', pert);
  check('product', 'likelihood @ default',  v0,      -13.9458508897, 2e-4);
  check('product', 'likelihood @ mu1=0.5',  v1,      -12.6372687739, 2e-4);
  check('product', 'Δ(default→mu1=0.5)',    v1 - v0,   1.3085821158, 2e-4);
}

const RUNNERS = { gaussian: runGaussian, histfactory: runHistfact, product: runProduct };

async function main() {
  const { engineDir: explicit, model } = parseArgs(process.argv.slice(2));
  const engine = resolveEngine(explicit);
  const hs3Dir = __dirname;

  if (model && !RUNNERS[model]) {
    process.stderr.write(`repro_hs3_js: unknown model '${model}'. Choose: ${Object.keys(RUNNERS).join(', ')}\n`);
    process.exit(2);
  }

  const SEP = '-'.repeat(COL_MODEL + COL_LABEL + COL_VALUE + 10);
  console.log(pad('model', COL_MODEL) + pad('test point', COL_LABEL) + lpad('log-density', COL_VALUE) + '   result');
  console.log(SEP);

  const toRun = model ? [RUNNERS[model]] : Object.values(RUNNERS);
  for (const run of toRun) await run(engine, hs3Dir);

  console.log(SEP);
  process.exit(failed > 0 ? 1 : 0);
}

main().catch((e) => {
  process.stderr.write('repro_hs3_js: ' + (e && e.message ? e.message : e) + '\n');
  process.exit(1);
});
