#!/usr/bin/env node
'use strict';
// Evaluate a NAMED deterministic binding in an already-determinized FlatPDL
// model, using the flatppl-js engine. The FlatPDL counterpart to score_js.cjs:
// that script scores a *measure* by appending `logdensityof(binding, theta)`;
// this one scores a *value* binding that a prior `flatppl determinize` pass
// has already reduced to a closed-form expression (e.g. `__score__` bound to
// a `builtin_logdensityof(...)` call) — no measure layer left to query.
//
// Usage:
//   node score_flatpdl.cjs <model.flatppl> <binding> [--engine <dir>]
//
//   <binding>  the name of a deterministic binding in the model (e.g. the
//              `__score__` binding appended before determinizing).
//
// Engine resolution (first that exists): --engine <dir>, then $FLATPPL_JS_DIR
// (its packages/engine), then the cache clone at ~/.cache/flatppl-js. Requires
// Node >= 24 — the engine modules are TypeScript, loaded via Node's native type
// stripping. Mirrors score_js.cjs's engine wiring (resolveEngine, processSource,
// orchestrator.buildDerivations, createWorkerHandler, materialiser) exactly;
// keep the two in sync if the engine API changes.

const fs = require('fs');
const path = require('path');

function usage(msg) {
  if (msg) process.stderr.write('score_flatpdl: ' + msg + '\n');
  process.stderr.write(
    'usage: node score_flatpdl.cjs <model.flatppl> <binding> [--engine <dir>]\n');
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
  if (pos.length !== 2) usage('expected <model.flatppl> <binding>');
  return { model: pos[0], binding: pos[1], engineDir };
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

async function main() {
  const { model, binding, engineDir } = parseArgs(process.argv.slice(2));
  const engine = resolveEngine(engineDir);
  const { processSource, orchestrator, materialiser } = require(path.join(engine, 'index.ts'));
  const { createWorkerHandler } = require(path.join(engine, 'worker.ts'));

  const src = fs.readFileSync(model, 'utf8');

  const proc = processSource(src);
  for (const d of proc.diagnostics || []) {
    if (d.severity === 'error') process.stderr.write('diagnostic: ' + d.message + '\n');
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
    getMeasure: (n) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, m);
      return m;
    },
    sendWorker: (m) => Promise.resolve(w.handle(m)),
  };

  const measure = await ctx.getMeasure(binding);
  if (measure && measure.value && measure.value.data) {
    process.stdout.write(measure.value.data[0] + '\n');
  } else if (measure && measure.samples && measure.samples.length) {
    process.stdout.write(measure.samples[0] + '\n');
  } else {
    process.stderr.write('score_flatpdl: no value for binding ' + binding + '\n');
    process.exit(1);
  }
}

main().catch((e) => {
  process.stderr.write('score_flatpdl: ' + (e && e.message ? e.message : e) + '\n');
  process.exit(1);
});
