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
// keep the two in sync if the engine API changes. The scoring itself lives in
// score_core.cjs, shared with the persistent worker (score_worker.cjs).

const core = require('./score_core.cjs');

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

async function main() {
  const { model, binding, engineDir } = parseArgs(process.argv.slice(2));
  let dir;
  try { dir = core.resolveEngine(engineDir); } catch (e) {
    if (e instanceof core.EngineNotFound) usage(e.message);
    throw e;
  }
  const engine = core.loadEngine(dir);
  const value = await core.scoreBinding(engine, model, binding, (line) => process.stderr.write(line));
  process.stdout.write(value + '\n');
}

main().catch((e) => {
  process.stderr.write('score_flatpdl: ' + (e && e.message ? e.message : e) + '\n');
  process.exit(1);
});
