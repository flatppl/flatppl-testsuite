#!/usr/bin/env node
'use strict';
// Evaluate a NAMED deterministic binding across MANY already-determinized
// FlatPDL sources, in ONE Node process -- the ABI-point counterpart of
// sample_sweep.cjs (which sweeps rnginit seeds instead of full sources).
// score_flatpdl.cjs pays flatppl-js's ~0.3s module-load cost once per POINT;
// unified/detjs_exec.py's score_abi_points() already determinizes an ABI
// query module once and only needs each point's inputs re-substituted as
// literals, so this pays that load cost once per TEST DIR instead of once
// per point.
//
// Usage:
//   node score_flatpdl_batch.cjs <sources.json> <binding> [--engine <dir>] [--vector-size N]
// The unified harness no longer starts this script per case: it sends the same
// request to the persistent worker (score_worker.cjs), which runs the shared core
// (score_core.cjs) this script runs.
//
//   <sources.json>  a JSON array of FlatPDL source strings, one per point, in
//                   order. Each is independently processSource'd/materialised
//                   with a FRESH derivation graph and sample cache -- no state
//                   carries between points, matching the one-process-per-point
//                   behaviour this replaces.
//   <binding>       the deterministic binding to evaluate in each source (the
//                   ABI query's `outputs` binding).
//   --vector-size N  evaluate ONE source returning an N-element score vector,
//                   and emit one result row per element, in order.
//
// Prints a JSON array to stdout, one entry per source IN ORDER:
//   {"ok": true, "value": <number>}     -- binding materialised to a value
//   {"ok": false, "error": "<message>"} -- this point failed
// A per-point failure never aborts the batch or the process -- the caller
// (score_abi_points) turns an {"ok": false, ...} entry into that point's own
// CheckResult instead of failing every point in the test dir.
// Vector mode instead reports a whole-vector failure if its one evaluation
// throws. Nonfinite numeric elements remain independent score values.
//
// Engine resolution is shared with score_flatpdl.cjs (score_core.cjs).

const fs = require('fs');
const core = require('./score_core.cjs');

function usage(msg) {
  if (msg) process.stderr.write('score_flatpdl_batch: ' + msg + '\n');
  process.stderr.write(
    'usage: node score_flatpdl_batch.cjs <sources.json> <binding> [--engine <dir>] [--vector-size N]\n');
  process.exit(2);
}

function parseArgs(argv) {
  const pos = [];
  let engineDir = null;
  let vectorSize = null;
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--engine') { engineDir = argv[++i]; }
    else if (a === '--vector-size') {
      vectorSize = Number(argv[++i]);
      if (!Number.isSafeInteger(vectorSize) || vectorSize < 1) usage('invalid vector size');
    }
    else if (a.startsWith('--')) { usage('unknown flag ' + a); }
    else { pos.push(a); }
  }
  if (pos.length !== 2) usage('expected <sources.json> <binding>');
  return { sourcesPath: pos[0], binding: pos[1], engineDir, vectorSize };
}

async function main() {
  const { sourcesPath, binding, engineDir, vectorSize } = parseArgs(process.argv.slice(2));
  let dir;
  try { dir = core.resolveEngine(engineDir); } catch (e) {
    if (e instanceof core.EngineNotFound) usage(e.message);
    throw e;
  }
  const engine = core.loadEngine(dir);
  const sources = JSON.parse(fs.readFileSync(sourcesPath, 'utf8'));
  process.stdout.write(JSON.stringify(await core.scoreBatch(engine, { sources, binding, vectorSize })) + '\n');
}

main().catch((e) => {
  process.stderr.write('score_flatpdl_batch: ' + (e && e.stack ? e.stack : e) + '\n');
  process.exit(1);
});
