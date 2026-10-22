#!/usr/bin/env node
'use strict';
// Seed-sweep a determinized FlatPDL model that ends in a `builtin_sample`
// chain (the output of `flatppl determinize` on a model using
// `rand(lawof(...), rng)`) to build an empirical sample set for a numeric
// statistical gate — the sibling of score_flatpdl.cjs, which materialises
// exactly ONE deterministic binding at the model's single fixed seed.
//
// A determinized FlatPDL model is a closed-form function of the RNG bytes
// literal in its leading `rnginit([...])` binding: one seed gives ONE
// realization. To get a DISTRIBUTION of realizations (so a suite can check
// empirical mean/var/cov against a closed-form oracle), this script
// string-substitutes the `rnginit([...])` byte-vector for N distinct seeds
// and re-materialises the requested bindings each time, in ONE Node
// process — it does NOT re-invoke `flatppl determinize` or spawn a
// subprocess per seed (3N subprocesses over N=thousands would be far too
// slow). The engine module is loaded once; only `processSource` +
// `orchestrator.buildDerivations` + `materialiser.materialiseMeasure` repeat
// per seed, each with a fresh context (a fresh binding graph and a fresh
// per-seed sample cache — the whole point is that these are INDEPENDENT
// realizations, so nothing may carry over between seeds).
//
// Usage:
//   node sample_sweep.cjs <model.flatpdl.flatppl> <N> <binding,binding,...> [--engine <dir>] [--base <i0>]
//
//   <model.flatpdl.flatppl>  a determinized FlatPDL model containing exactly
//                            one `rnginit([b0, b1, b2, b3])` binding.
//   <N>                      number of seeds to sweep.
//   <binding,binding,...>    comma-separated deterministic bindings to
//                            materialise at every seed (e.g. `mu,y1,y2`).
//   --base <i0>              first seed index (default 0); seeds are
//                            `i0 .. i0+N-1`, encoded injectively as four
//                            little-endian bytes. Every index must be an
//                            integer in [0, 2^32); the range must not wrap.
//
// Prints a JSON array of N objects, one per seed, each keyed by the
// requested binding names, to stdout:
//   [{"mu": -18.386..., "y1": -18.854..., "y2": -19.263...}, ...]
//
// Engine resolution is shared with score_flatpdl.cjs (score_core.cjs), as is the
// sweep itself, which the persistent worker (score_worker.cjs) also runs.

const core = require('./score_core.cjs');

function usage(msg) {
  if (msg) process.stderr.write('sample_sweep: ' + msg + '\n');
  process.stderr.write(
    'usage: node sample_sweep.cjs <model.flatpdl.flatppl> <N> <binding,binding,...> '
    + '[--engine <dir>] [--base <i0>]\n');
  process.exit(2);
}

function parseArgs(argv) {
  const pos = [];
  let engineDir = null;
  let base = 0;
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--engine') { engineDir = argv[++i]; }
    else if (a === '--base') { base = Number(argv[++i]); }
    else if (a.startsWith('--')) { usage('unknown flag ' + a); }
    else { pos.push(a); }
  }
  if (pos.length !== 3) usage('expected <model.flatpdl.flatppl> <N> <bindings>');
  const n = Number(pos[1]);
  let bindings;
  try { bindings = core.checkSweepArgs(n, base, [pos[2]]); } catch (e) {
    if (e instanceof core.SweepUsageError) usage(e.message);
    throw e;
  }
  return { model: pos[0], n, bindings, engineDir, base };
}

async function main() {
  const { model, n, bindings, engineDir, base } = parseArgs(process.argv.slice(2));
  let dir;
  try { dir = core.resolveEngine(engineDir); } catch (e) {
    if (e instanceof core.EngineNotFound) usage(e.message);
    throw e;
  }
  const engine = core.loadEngine(dir);
  let results;
  try {
    results = await core.sampleSweep(engine, { model, n, bindings, base });
  } catch (e) {
    // A refused model or seed fails the sweep with its message, no stack.
    if (e instanceof core.SweepError) {
      if (e instanceof core.SweepUsageError) usage(e.message);
      process.stderr.write('sample_sweep: ' + e.message + '\n');
      process.exit(1);
    }
    throw e;
  }
  process.stdout.write(JSON.stringify(results) + '\n');
}

main().catch((e) => {
  process.stderr.write('sample_sweep: ' + (e && e.stack ? e.stack : e) + '\n');
  process.exit(1);
});
