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
//   node score_flatpdl_batch.cjs <sources.json> <binding> [--engine <dir>]
//
//   <sources.json>  a JSON array of FlatPDL source strings, one per point, in
//                   order. Each is independently processSource'd/materialised
//                   with a FRESH derivation graph and sample cache -- no state
//                   carries between points, matching the one-process-per-point
//                   behaviour this replaces.
//   <binding>       the deterministic binding to evaluate in each source (the
//                   ABI query's `outputs` binding).
//
// Prints a JSON array to stdout, one entry per source IN ORDER:
//   {"ok": true, "value": <number>}     -- binding materialised to a value
//   {"ok": false, "error": "<message>"} -- this point failed
// A per-point failure never aborts the batch or the process -- the caller
// (score_abi_points) turns an {"ok": false, ...} entry into that point's own
// CheckResult instead of failing every point in the test dir.
//
// Engine resolution mirrors score_flatpdl.cjs exactly (--engine, then
// $FLATPPL_JS_DIR, then the ~/.cache/flatppl-js clone); keep the two in sync
// if the engine API changes.

const fs = require('fs');
const path = require('path');

function usage(msg) {
  if (msg) process.stderr.write('score_flatpdl_batch: ' + msg + '\n');
  process.stderr.write(
    'usage: node score_flatpdl_batch.cjs <sources.json> <binding> [--engine <dir>]\n');
  process.exit(2);
}

function parseArgs(argv) {
  const pos = [];
  let engineDir = null;
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--engine') { engineDir = argv[++i]; }
    else if (a.startsWith('--')) { usage('unknown flag ' + a); }
    else { pos.push(a); }
  }
  if (pos.length !== 2) usage('expected <sources.json> <binding>');
  return { sourcesPath: pos[0], binding: pos[1], engineDir };
}

function resolveEngine(explicit) {
  const candidates = [
    explicit,
    process.env.FLATPPL_JS_DIR && path.join(process.env.FLATPPL_JS_DIR, 'packages', 'engine'),
    path.join(require('os').homedir(), '.cache', 'flatppl-js', 'packages', 'engine'),
  ].filter(Boolean);
  for (const c of candidates) {
    if (fs.existsSync(path.join(c, 'index.ts'))) return c;
  }
  usage('flatppl-js engine not found — pass --engine <dir> or set FLATPPL_JS_DIR '
    + '(looked in: ' + candidates.join(', ') + ')');
}

// One point: a fresh derivation graph and sample cache, sharing only the
// already-loaded engine module and the worker handler -- mirrors the isolation
// score_flatpdl.cjs gets for free by being a fresh process per point.
async function scoreOne(src, binding, processSource, orchestrator, materialiser, w) {
  const proc = processSource(src);
  for (const d of proc.diagnostics || []) {
    if (d.severity === 'error') throw new Error('diagnostic: ' + d.message);
  }

  const built = orchestrator.buildDerivations(proc.bindings);
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
    getMeasure: (n) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, m);
      return m;
    },
    sendWorker: (m) => Promise.resolve(w.handle(m)),
  };

  const measure = await ctx.getMeasure(binding);
  if (measure && measure.value && measure.value.data) return measure.value.data[0];
  if (measure && measure.samples && measure.samples.length) return measure.samples[0];
  throw new Error('no value for binding ' + binding);
}

async function main() {
  const { sourcesPath, binding, engineDir } = parseArgs(process.argv.slice(2));
  const engine = resolveEngine(engineDir);
  const { processSource, orchestrator, materialiser } = require(path.join(engine, 'index.ts'));
  const { createWorkerHandler } = require(path.join(engine, 'worker.ts'));

  const sources = JSON.parse(fs.readFileSync(sourcesPath, 'utf8'));
  if (!Array.isArray(sources) || !sources.length) {
    usage('sources.json must be a non-empty JSON array of source strings');
  }

  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 3 });

  const results = new Array(sources.length);
  for (let i = 0; i < sources.length; i++) {
    try {
      const value = await scoreOne(sources[i], binding, processSource, orchestrator, materialiser, w);
      // JSON has no encoding for NaN/±Infinity (JSON.stringify(-Infinity) is
      // null) -- an out-of-support point's log-density is exactly -inf, and
      // this corpus carries that shape (score_abi_points' caller compares
      // against frozen "-inf"/"nan" strings). Send those through as strings;
      // Python's float() already parses "Infinity"/"-Infinity"/"NaN" natively.
      results[i] = { ok: true, value: Number.isFinite(value) ? value : String(value) };
    } catch (e) {
      results[i] = { ok: false, error: e && e.message ? e.message : String(e) };
    }
  }

  process.stdout.write(JSON.stringify(results) + '\n');
}

main().catch((e) => {
  process.stderr.write('score_flatpdl_batch: ' + (e && e.stack ? e.stack : e) + '\n');
  process.exit(1);
});
