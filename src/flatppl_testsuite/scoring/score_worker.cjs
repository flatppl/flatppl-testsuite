#!/usr/bin/env node
'use strict';
// Persistent scoring worker: loads the flatppl-js engine ONCE and answers one
// JSON request per line on stdin with one JSON reply per line on stdout. The
// harness starts one per pytest process instead of one Node process per case.
//
// Usage:
//   node score_worker.cjs [--engine <dir>]      (EOF on stdin ends the worker)
//
// Requests ({"id": N, "op": ..., ...}), each run by the same core the one-shot
// scripts run (score_core.cjs), so a request scores exactly as its script would:
//   binding       {model, binding}               == score_flatpdl.cjs
//   js            {model, binding, theta, count} == score_js.cjs
//   batch         {sources, binding, vectorSize} == score_flatpdl_batch.cjs
//   sample_sweep  {model, n, bindings, base}     == sample_sweep.cjs
//
// Replies: {"id": N, "ok": true, "value": "<number as text>" | "rows": [...],
// "rss": <bytes>} or {"id": N, "ok": false, "stderr": "<what the one-shot script
// would have written to stderr>"}. A failing request never ends the worker.
//
// Isolation: every request builds its own worker handler, RNG seed, derivation
// graph and measure cache (see score_core.cjs); only the loaded engine code is
// shared. Anything the engine writes to stdout or stderr while a request runs is
// captured into that request's `stderr` text, so it can neither corrupt the
// protocol stream nor leak into another request. A fatal error answers the request
// in flight and exits, so the harness restarts a fresh worker.

const fs = require('fs');
const core = require('./score_core.cjs');

function parseArgs(argv) {
  let engineDir = null;
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--engine') engineDir = argv[++i];
    else { process.stderr.write('score_worker: unknown argument ' + argv[i] + '\n'); process.exit(2); }
  }
  return { engineDir };
}

const SWEEP_USAGE = 'usage: node sample_sweep.cjs <model.flatpdl.flatppl> <N> <binding,binding,...> '
  + '[--engine <dir>] [--base <i0>]\n';

const realStdoutWrite = process.stdout.write.bind(process.stdout);
const realStderrWrite = process.stderr.write.bind(process.stderr);
let captured = null;
let inFlight = null;

function capture(chunk, encoding, callback) {
  if (captured !== null) captured.push(typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString());
  else realStderrWrite(chunk, encoding, callback);
  const cb = typeof encoding === 'function' ? encoding : callback;
  if (typeof cb === 'function') cb();
  return true;
}
process.stdout.write = capture;
process.stderr.write = capture;

function reply(obj) {
  realStdoutWrite(JSON.stringify(obj) + '\n');
}

function fatal(error) {
  const text = (captured ? captured.join('') : '') + 'score_worker: ' + (error && error.stack ? error.stack : error) + '\n';
  if (inFlight !== null) {
    try { fs.writeSync(1, JSON.stringify({ id: inFlight, ok: false, stderr: text }) + '\n'); } catch (_) { /* pipe closed */ }
  } else {
    try { fs.writeSync(2, text); } catch (_) { /* nothing to report to */ }
  }
  process.exit(70);
}
process.on('uncaughtException', fatal);
process.on('unhandledRejection', fatal);

async function handle(engine, req) {
  switch (req.op) {
    case 'binding': {
      try {
        const value = await core.scoreBinding(engine, req.model, req.binding, (line) => process.stderr.write(line));
        return { ok: true, value: String(value) };
      } catch (e) {
        process.stderr.write('score_flatpdl: ' + (e && e.message ? e.message : e) + '\n');
        return { ok: false };
      }
    }
    case 'js': {
      try {
        const value = await core.scoreJs(engine, { model: req.model, binding: req.binding,
          theta: req.theta, count: req.count === undefined ? 1 : req.count }, (line) => process.stderr.write(line));
        return { ok: true, value: String(value) };
      } catch (e) {
        process.stderr.write('score_js: ' + (e && e.message ? e.message : e) + '\n');
        return { ok: false };
      }
    }
    case 'batch': {
      try {
        return { ok: true, rows: await core.scoreBatch(engine, { sources: req.sources, binding: req.binding,
          vectorSize: req.vectorSize === undefined ? null : req.vectorSize }) };
      } catch (e) {
        process.stderr.write('score_flatpdl_batch: ' + (e && e.message ? e.message : e) + '\n');
        return { ok: false };
      }
    }
    case 'sample_sweep': {
      try {
        return { ok: true, rows: await core.sampleSweep(engine, {
          model: req.model, n: req.n, bindings: req.bindings, base: req.base }) };
      } catch (e) {
        if (e instanceof core.SweepError) {
          process.stderr.write('sample_sweep: ' + e.message + '\n');
          if (e instanceof core.SweepUsageError) process.stderr.write(SWEEP_USAGE);
        } else {
          process.stderr.write('sample_sweep: ' + (e && e.stack ? e.stack : e) + '\n');
        }
        return { ok: false };
      }
    }
    default:
      process.stderr.write('score_worker: unknown op ' + JSON.stringify(req.op) + '\n');
      return { ok: false };
  }
}

async function main() {
  const { engineDir } = parseArgs(process.argv.slice(2));
  let dir;
  try { dir = core.resolveEngine(engineDir); } catch (e) {
    if (e instanceof core.EngineNotFound) { realStderrWrite('score_worker: ' + e.message + '\n'); process.exit(2); }
    throw e;
  }
  const engine = core.loadEngine(dir);
  reply({ ready: true, engine: dir });

  const lines = require('readline').createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of lines) {
    if (!line.trim()) continue;
    let req;
    try { req = JSON.parse(line); } catch (e) {
      reply({ id: null, ok: false, stderr: 'score_worker: invalid request: ' + e.message + '\n' });
      continue;
    }
    inFlight = req.id === undefined ? null : req.id;
    captured = [];
    let result;
    try { result = await handle(engine, req); } catch (e) {
      captured.push('score_worker: ' + (e && e.stack ? e.stack : e) + '\n');
      result = { ok: false };
    }
    const stderr = captured.join('');
    captured = null;
    if (!result.ok) result.stderr = stderr;
    result.id = inFlight;
    result.rss = process.memoryUsage().rss;
    inFlight = null;
    reply(result);
  }
}

main().catch(fatal);
