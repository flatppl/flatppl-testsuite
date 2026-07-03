#!/usr/bin/env node
'use strict';
// Seed-sweep a determinized FlatPDL model that ends in a `builtin_sample`
// chain (the output of `flatppl determinize` on a model using
// `rand(rng, lawof(...))`) to build an empirical sample set for a numeric
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
//                            `i0 .. i0+N-1`, mapped injectively to 4 bytes
//                            via `[i & 255, (i >> 8) & 255, 0, 0]` — distinct
//                            for i in [0, 65536), ample for the sweep sizes
//                            this gate uses.
//
// Prints a JSON array of N objects, one per seed, each keyed by the
// requested binding names, to stdout:
//   [{"mu": -18.386..., "y1": -18.854..., "y2": -19.263...}, ...]
//
// Engine resolution mirrors score_flatpdl.cjs exactly (--engine, then
// $FLATPPL_JS_DIR, then the ~/.cache/flatppl-js clone); keep the two in sync
// if the engine API changes.

const fs = require('fs');
const path = require('path');

const RNGINIT_RE = /rnginit\(\s*\[[^\]]*\]\s*\)/;

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
    else if (a === '--base') { base = parseInt(argv[++i], 10); }
    else if (a.startsWith('--')) { usage('unknown flag ' + a); }
    else { pos.push(a); }
  }
  if (pos.length !== 3) usage('expected <model.flatpdl.flatppl> <N> <bindings>');
  const n = parseInt(pos[1], 10);
  if (!Number.isInteger(n) || n <= 0) usage('N must be a positive integer');
  const bindings = pos[2].split(',').map((s) => s.trim()).filter(Boolean);
  if (!bindings.length) usage('need at least one binding');
  return { model: pos[0], n, bindings, engineDir, base };
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

// Injective seed -> 4-byte mapping (see the --base doc comment above).
function seedBytes(i) {
  return [i & 255, (i >> 8) & 255, 0, 0];
}

async function main() {
  const { model, n, bindings, engineDir, base } = parseArgs(process.argv.slice(2));
  const engine = resolveEngine(engineDir);
  const { processSource, orchestrator, materialiser } = require(path.join(engine, 'index.ts'));
  const { createWorkerHandler } = require(path.join(engine, 'worker.ts'));

  const rawSrc = fs.readFileSync(model, 'utf8');
  if (!RNGINIT_RE.test(rawSrc)) {
    usage('model has no rnginit([...]) binding to seed-sweep: ' + model);
  }

  // The worker is a stateless math kernel (see worker.ts's header comment);
  // its `init` seed and the ctx's rootKey/sampleCount below are fixed across
  // the whole sweep, exactly as in score_flatpdl.cjs. The per-seed
  // randomness driving the swept realizations comes entirely from the
  // `rnginit([...])` bytes substituted into the source below — verified by
  // construction: the same fixed worker/rootKey config reproduces distinct,
  // oracle-matching realizations per distinct rnginit byte-vector.
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 3 });

  const results = new Array(n);
  for (let i = 0; i < n; i++) {
    const [b0, b1, b2, b3] = seedBytes(base + i);
    const src = rawSrc.replace(RNGINIT_RE, `rnginit([${b0}, ${b1}, ${b2}, ${b3}])`);

    const proc = processSource(src);
    for (const d of proc.diagnostics || []) {
      if (d.severity === 'error') {
        process.stderr.write('sample_sweep: diagnostic at seed ' + i + ': ' + d.message + '\n');
        process.exit(1);
      }
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
      getMeasure: (name) => {
        if (cache.has(name)) return cache.get(name);
        const m = materialiser.materialiseMeasure(name, ctx);
        cache.set(name, m);
        return m;
      },
      sendWorker: (m) => Promise.resolve(w.handle(m)),
    };

    const row = {};
    for (const binding of bindings) {
      const measure = await ctx.getMeasure(binding);
      if (measure && measure.value && measure.value.data) {
        row[binding] = measure.value.data[0];
      } else if (measure && measure.samples && measure.samples.length) {
        row[binding] = measure.samples[0];
      } else {
        process.stderr.write(
          'sample_sweep: no value for binding ' + binding + ' at seed ' + i + '\n');
        process.exit(1);
      }
    }
    results[i] = row;
  }

  process.stdout.write(JSON.stringify(results) + '\n');
}

main().catch((e) => {
  process.stderr.write('sample_sweep: ' + (e && e.stack ? e.stack : e) + '\n');
  process.exit(1);
});
