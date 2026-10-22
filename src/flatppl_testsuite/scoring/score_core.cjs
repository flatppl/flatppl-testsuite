'use strict';
// Shared scoring cores for the one-shot scripts (score_flatpdl.cjs,
// score_flatpdl_batch.cjs, sample_sweep.cjs) and the persistent worker
// (score_worker.cjs). One code path, so a request scored by the worker is
// scored exactly as the one-shot script scores it.
//
// Isolation rule: every call builds its own worker handler (math kernel +
// RNG seed), derivation graph and measure cache from the source. The only state
// shared between calls is the engine's loaded code.

const fs = require('fs');
const path = require('path');

class EngineNotFound extends Error {}

// Engine resolution order: an explicit directory, then $FLATPPL_JS_DIR's
// packages/engine, then the ~/.cache/flatppl-js clone.
function resolveEngine(explicit) {
  const candidates = [
    explicit,
    process.env.FLATPPL_JS_DIR && path.join(process.env.FLATPPL_JS_DIR, 'packages', 'engine'),
    path.join(require('os').homedir(), '.cache', 'flatppl-js', 'packages', 'engine'),
  ].filter(Boolean);
  for (const c of candidates) {
    if (fs.existsSync(path.join(c, 'index.ts'))) return c;
  }
  throw new EngineNotFound('flatppl-js engine not found — pass --engine <dir> or set FLATPPL_JS_DIR '
    + '(looked in: ' + candidates.join(', ') + ')');
}

function loadEngine(dir) {
  const index = require(path.join(dir, 'index.ts'));
  const { createWorkerHandler } = require(path.join(dir, 'worker.ts'));
  return {
    processSource: index.processSource,
    resolveBundleBytes: index.resolveBundleBytes,
    orchestrator: index.orchestrator,
    materialiser: index.materialiser,
    createWorkerHandler,
  };
}

function withBuildDiagnostics(e, built) {
  const errors = (built.diagnostics || []).filter((d) => d.severity === 'error');
  const message = e && e.message ? e.message : String(e);
  if (!errors.length) return e;
  return new Error(message + ' (' + errors.map((d) => 'diagnostic: ' + d.message).join('; ') + ')');
}

function makeCtx(engine, built, loweredRegistry, w, sampleCount = 1) {
  const cache = new Map();
  const ctx = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    sampleCount,
    rootKey: 3,
    rootSeed: 3,
    marginalizationCount: 32,
    moduleRegistry: loweredRegistry,
    getMeasure: (n) => {
      if (cache.has(n)) return cache.get(n);
      const m = engine.materialiser.materialiseMeasure(n, ctx);
      cache.set(n, m);
      return m;
    },
    sendWorker: (m) => Promise.resolve(w.handle(m)),
  };
  return ctx;
}

function freshWorker(engine) {
  const w = engine.createWorkerHandler();
  w.handle({ type: 'init', seed: 3 });
  return w;
}

// score_flatpdl: one deterministic binding of one FlatPDL file, with module and
// data resolution through the file's bundle. `diag(line)` receives each error
// diagnostic as the one-shot script writes it to stderr.
async function scoreBinding(engine, model, binding, diag) {
  const bundle = await engine.resolveBundleBytes(path.resolve(model), fs.readFileSync(model),
    (file) => file.startsWith('flatppl-zip:') ? null : fs.promises.readFile(file), { strict: false });
  const proc = engine.processSource(bundle.primarySource, { path: bundle.primaryPath, bundle });
  for (const d of proc.diagnostics || []) {
    if (d.severity === 'error') diag('diagnostic: ' + d.message + '\n');
  }

  const built = engine.orchestrator.buildDerivations(proc.linkedBindings, {
    moduleRegistry: proc.linkedModuleRegistry,
  });
  const w = freshWorker(engine);
  const ctx = makeCtx(engine, built, proc.linkedModuleRegistry, w);

  // A fixed binding that fails to evaluate gets no derivation. Its real cause is
  // a build diagnostic, so name it instead of the materialiser's "no derivation".
  const measure = await ctx.getMeasure(binding).catch((e) => {
    throw withBuildDiagnostics(e, built);
  });
  if (measure && measure.value && measure.value.data) return measure.value.data[0];
  if (measure && measure.samples && measure.samples.length) return measure.samples[0];
  throw new Error('no value for binding ' + binding);
}

// score_js: a measure binding's log-density at a point, straight from FlatPPL
// source through the JS measure layer (no determinize step). Appends
// `__score__ = logdensityof(binding, theta)` to the model, as score_js.cjs does.
async function scoreJs(engine, { model, binding, theta, count = 1 }, diag) {
  const bundle = await engine.resolveBundleBytes(path.resolve(model), fs.readFileSync(model),
    (file) => file.startsWith('flatppl-zip:') ? null : fs.promises.readFile(file), { strict: false });
  const src = bundle.primarySource + `\n__score__ = logdensityof(${binding}, ${theta})\n`;
  const proc = engine.processSource(src, { path: bundle.primaryPath, bundle });
  // Surface diagnostics, but do NOT abort on them: a diagnostic on a binding
  // off the scored measure's dependency path (e.g. a single-factor `cartprod`
  // domain the engine flags) does not affect the density. If a diagnostic does
  // break the scored binding, materialising it below throws and is reported.
  for (const d of proc.diagnostics || []) {
    if (d.severity === 'error') diag('diagnostic: ' + d.message + '\n');
  }
  const built = engine.orchestrator.buildDerivations(proc.linkedBindings, {
    moduleRegistry: proc.linkedModuleRegistry,
  });
  // The standard-module registry (e.g. `hepphys`) must be threaded into the
  // worker session env, or cross-module refs in the body fail to resolve.
  const ctx = makeCtx(engine, built, proc.linkedModuleRegistry, freshWorker(engine), count);
  const measure = await ctx.getMeasure('__score__');
  if (!measure || !measure.samples || measure.samples.length === 0) {
    throw new Error('no density produced for `' + binding + '`');
  }
  return measure.samples[0];
}

// score_flatpdl_batch: one source's binding value. A fresh derivation graph and
// sample cache per source, sharing only the loaded engine and the worker handler.
async function scoreOne(engine, src, binding, w, vectorSize) {
  const proc = engine.processSource(src);
  for (const d of proc.diagnostics || []) {
    if (d.severity === 'error') throw new Error('diagnostic: ' + d.message);
  }

  const built = engine.orchestrator.buildDerivations(proc.bindings);
  const ctx = makeCtx(engine, built, proc.loweredModule && proc.loweredModule.moduleRegistry, w);

  const measure = await ctx.getMeasure(binding).catch((e) => {
    throw withBuildDiagnostics(e, built);
  });
  if (vectorSize !== null) {
    const value = measure && measure.value;
    if (!value || !value.shape || value.shape.length !== 1
        || value.shape[0] !== vectorSize || !value.data || value.data.length !== vectorSize) {
      throw new Error('expected a score vector of length ' + vectorSize + ' for ' + binding);
    }
    return Array.from(value.data);
  }
  if (measure && measure.value && measure.value.data) return measure.value.data[0];
  if (measure && measure.samples && measure.samples.length) return measure.samples[0];
  throw new Error('no value for binding ' + binding);
}

async function scoreBatch(engine, { sources, binding, vectorSize = null }) {
  if (!Array.isArray(sources) || !sources.length) {
    throw new Error('sources must be a non-empty JSON array of source strings');
  }
  if (vectorSize !== null && (!Number.isSafeInteger(vectorSize) || vectorSize < 1 || sources.length !== 1)) {
    throw new Error('vector mode requires one source and a positive vector size');
  }
  // Match one-shot state: share loaded code, never a previous request's env/RNG/cache.
  const w = freshWorker(engine);
  const results = [];
  for (let i = 0; i < sources.length; i++) {
    try {
      const value = await scoreOne(engine, sources[i], binding, w, vectorSize);
      // JSON cannot encode nonfinite values. Python's float() accepts these strings.
      for (const score of vectorSize === null ? [value] : value) {
        results.push({ ok: true, value: Number.isFinite(score) ? score : String(score) });
      }
    } catch (e) {
      for (let j = 0; j < (vectorSize === null ? 1 : vectorSize); j++) {
        results.push({ ok: false, error: e && e.message ? e.message : String(e) });
      }
    }
  }
  return results;
}

const RNGINIT_RE = /rnginit\(\s*\[[^\]]*\]\s*\)/;

// Injective seed -> 4-byte mapping (see sample_sweep.cjs, `--base`).
function seedBytes(i) {
  return [i & 255, (i >>> 8) & 255, (i >>> 16) & 255, (i >>> 24) & 255];
}

class SweepError extends Error {}
// A refused argument or model: the one-shot script prints its usage after the message.
class SweepUsageError extends SweepError {}

// The sweep's argument rules, shared by the CLI and the worker.
function checkSweepArgs(n, base, bindings) {
  if (!Number.isSafeInteger(n) || n <= 0) throw new SweepUsageError('N must be a positive integer');
  if (!Number.isSafeInteger(base) || base < 0 || base >= 0x100000000) {
    throw new SweepUsageError('base must be an integer in [0, 2^32)');
  }
  if (n > 0x100000000 - base) throw new SweepUsageError('seed range exceeds the four-byte domain');
  const names = bindings.join(',').split(',').map((s) => s.trim()).filter(Boolean);
  if (!names.length) throw new SweepUsageError('need at least one binding');
  return names;
}

// sample_sweep: N realizations of a determinized sample-path model, one
// string-substituted `rnginit([...])` per seed, each with a fresh context.
async function sampleSweep(engine, { model, n, bindings, base }) {
  bindings = checkSweepArgs(n, base, bindings);
  const rawSrc = fs.readFileSync(model, 'utf8');
  if (!RNGINIT_RE.test(rawSrc)) {
    throw new SweepUsageError('model has no rnginit([...]) binding to seed-sweep: ' + model);
  }

  // The worker is a stateless math kernel (see worker.ts's header comment); its
  // `init` seed and the ctx's rootKey/sampleCount are fixed across the sweep.
  // The per-seed randomness comes entirely from the `rnginit([...])` bytes
  // substituted into the source.
  const w = freshWorker(engine);

  const results = new Array(n);
  for (let i = 0; i < n; i++) {
    const [b0, b1, b2, b3] = seedBytes(base + i);
    const src = rawSrc.replace(RNGINIT_RE, `rnginit([${b0}, ${b1}, ${b2}, ${b3}])`);

    const proc = engine.processSource(src);
    for (const d of proc.diagnostics || []) {
      if (d.severity === 'error') {
        throw new SweepError('diagnostic at seed ' + i + ': ' + d.message);
      }
    }
    const built = engine.orchestrator.buildDerivations(proc.bindings);
    const ctx = makeCtx(engine, built, proc.loweredModule && proc.loweredModule.moduleRegistry, w);

    const row = {};
    for (const binding of bindings) {
      const measure = await ctx.getMeasure(binding);
      if (measure && measure.value && measure.value.data) {
        row[binding] = measure.value.data[0];
      } else if (measure && measure.samples && measure.samples.length) {
        row[binding] = measure.samples[0];
      } else {
        throw new SweepError('no value for binding ' + binding + ' at seed ' + i);
      }
    }
    results[i] = row;
  }
  return results;
}

module.exports = {
  EngineNotFound, SweepError, SweepUsageError, checkSweepArgs, resolveEngine, loadEngine, scoreBinding, scoreJs, scoreBatch, sampleSweep,
};
