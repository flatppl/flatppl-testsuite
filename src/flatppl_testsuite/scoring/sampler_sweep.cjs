#!/usr/bin/env node
'use strict';
// Batch-draw every probe of the SAMPLER sweep in ONE Node process and return
// sufficient statistics — the sample-side counterpart of the density sweep's
// score_flatpdl_batch.cjs.
//
// WHY THIS DOES NOT DETERMINIZE. Its sibling sample_sweep.cjs seed-sweeps a
// *determinized FlatPDL* model, so it can only reach constructs the rust
// determiniser lowers on the sample path — which is `iid` and measure-position
// `pushfwd`, and nothing else: `weighted`/`logweighted`/`bayesupdate` are
// refused as intractable (flatppl-rust crates/determinizer/src/sample.rs, in
// `refuse_weighted_family`), `superpose`/`kchain`/`jointchain` are refused as
// deferred (`refuse_deferred_combinator`), and draw-position `pushfwd` is
// refused separately. Every one of those IS sampled by the flatppl-js
// materialiser (matSuperpose, matClm, matPushfwd, the conditional normalize
// path). Going through the determiniser would therefore make the sweep blind
// to the whole combinator surface it exists to cover — including
// `iid(superpose(...))`, where the branch-pinning defect actually lived. So
// this driver loads the JS engine and materialises the probe's binding
// straight from FlatPPL source, exactly as the engine's own tests do via
// test/_ctx-factory.ts.
//
// WHY SUFFICIENT STATISTICS AND NOT THE DRAWS. A row at n = 200000 with k = 4
// coordinates is 800k floats; the roster is ~70 rows. Shipping the raw draws
// over stdout would dominate the runtime and the memory. Instead the moment
// and covariance accumulators are formed here, in the pass that already has
// the samples in cache, and only a fixed-size subsample survives for the KS
// test (which needs a scipy cdf that exists only on the Python side).
//
// Usage:
//   node sampler_sweep.cjs <job.json> [--engine <dir>]
//
// job.json:
//   { "seed": <int>,
//     "ksSubsample": <int>,
//     "probes": [ { "id":     <string>,
//                   "source": <FlatPPL source text>,
//                   "binding":<binding name to materialise>,
//                   "n":      <draw count>,
//                   "k":      <coordinates per draw>,
//                   "field":  <record field name, or null for a plain measure>,
//                   "latent": <a SECOND binding whose weighted marginal is
//                              reported, or null>,
//                   "weightedVariate": <true to accumulate the variate's own
//                              moments under the measure's atom weights> },
//                 ... ] }
//
// WHY A LATENT, AND WHY ITS MOMENT IS WEIGHTED. A `normalize` whose mass moves
// with a latent that is NOT the variate is corrected in the WEIGHTS, not in the
// atom positions: the pooled-divisor defect leaves atom i the residue
// Z(theta_i)/E[Z] while every atom keeps the position it had. An UNWEIGHTED
// moment of the latent is therefore identical before and after the fix, and
// blind to the whole class. So a probe naming a `latent` gets the
// self-normalised weighted mean of that binding's samples under the TARGET
// measure's `logWeights`, plus the effective sample size the same weights give
// (`nEff` = 1 / sum of squared normalised weights), which is what the caller
// bands with.
//
// stdout (one JSON object):
//   { "results": [ { "id", "status", ... } ] }
//
// `status` is "DRAWS" (finite draws obtained), "THREW" (the engine raised —
// Python classifies the message into REFUSES vs MALFORMED, so the pattern list
// and its rationale live in one reviewable place), or "NONFINITE" (draws came
// back but not all finite; always a defect).
//
// A DRAWS row carries, per coordinate i in 0..k-1: `sum[i]`, `sumsq[i]` and
// `cross[i]` = sum over atoms of x_0 * x_i (raw, uncentred — the caller
// centres, so no second pass is needed). Plus `logTotalmass` when the engine
// reported one, and `ksSample`, a strided subsample of coordinate 0.
//
// WHY A VARIATE MOMENT IS SOMETIMES WEIGHTED TOO. The same argument as the
// latent's, one step over: `normalize(weighted(f, Q))` draws at Q's positions
// and carries f/Z in the weights, so `iid` of it has the whole law in the ATOM
// weight and an unweighted moment of the variate measures Q. A probe setting
// `weightedVariate` gets the self-normalised weighted accumulators instead, and
// `momentDenom` (1 rather than n) plus `variateNEff` tell the caller how to read
// and band them. A KS test cannot follow: it needs the draws themselves, and
// only a resample would turn a weighted ensemble into them.
//
// Engine resolution mirrors score_flatpdl.cjs (--engine, then $FLATPPL_JS_DIR,
// then the ~/.cache/flatppl-js clone); keep them in sync if the API moves.

const fs = require('fs');
const path = require('path');

function usage(msg) {
  if (msg) process.stderr.write('sampler_sweep: ' + msg + '\n');
  process.stderr.write('usage: node sampler_sweep.cjs <job.json> [--engine <dir>]\n');
  process.exit(2);
}

function parseArgs(argv) {
  const pos = [];
  let engineDir = null;
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--engine') engineDir = argv[++i];
    else if (argv[i].startsWith('--')) usage('unknown flag ' + argv[i]);
    else pos.push(argv[i]);
  }
  if (pos.length !== 1) usage('expected exactly one <job.json>');
  return { job: pos[0], engineDir };
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
  const { job, engineDir } = parseArgs(process.argv.slice(2));
  const engine = resolveEngine(engineDir);
  const { processSource, orchestrator, materialiser } = require(path.join(engine, 'index.ts'));
  const { createWorkerHandler } = require(path.join(engine, 'worker.ts'));

  const spec = JSON.parse(fs.readFileSync(job, 'utf8'));
  const seed = spec.seed;
  const ksSub = spec.ksSubsample || 20000;

  // One worker for the whole batch. It is a stateless math kernel, and each
  // probe gets a fresh binding graph and a fresh sample cache below, so
  // nothing carries between probes.
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed });

  const results = [];

  for (const p of spec.probes) {
    const t0 = Date.now();
    const row = { id: p.id, n: p.n, k: p.k };
    try {
      const proc = processSource(p.source);
      const errs = (proc.diagnostics || []).filter((d) => d.severity === 'error');
      if (errs.length) throw new Error('static diagnostic: ' + errs.map((d) => d.message).join(' | '));

      const built = orchestrator.buildDerivations(proc.bindings);
      const cache = new Map();
      const ctx = {
        derivations: built.derivations,
        bindings: built.bindings,
        fixedValues: built.fixedValues || new Map(),
        sampleCount: p.n,
        rootKey: seed,
        rootSeed: seed,
        marginalizationCount: 64,
        moduleRegistry: proc.loweredModule && proc.loweredModule.moduleRegistry,
        getMeasure: (name) => {
          if (cache.has(name)) return cache.get(name);
          const m = materialiser.materialiseMeasure(name, ctx);
          cache.set(name, m);
          return m;
        },
        sendWorker: (m) => {
          const r = w.handle(m);
          return r && r.type === 'error'
            ? Promise.reject(new Error(r.message))
            : Promise.resolve(r);
        },
      };

      results.push(await drawRow(row, ctx, p, ksSub, t0));
    } catch (e) {
      row.status = 'THREW';
      row.error = String((e && e.message) || e).replace(/\s+/g, ' ').trim();
      row.ms = Date.now() - t0;
      results.push(row);
    }
  }

  process.stdout.write(JSON.stringify({ results }) + '\n');
}

// Reduce one materialised measure to sufficient statistics.
//
// `await` rather than a bare call: the materialiser returns a settled object for
// a leaf distribution but a Promise for the composite paths (matClm, matIid over
// a record, …), and a REFUSAL on those arrives as a rejected promise rather than
// a synchronous throw. Awaiting puts both shapes through the caller's try/catch;
// without it a refusing composite row crashes the whole batch as an unhandled
// rejection instead of being recorded as that row's outcome.
async function drawRow(row, ctx, p, ksSub, t0) {
  const measure = await ctx.getMeasure(p.binding);
  if (!measure) throw new Error('materialiser returned no measure for ' + p.binding);

  const flat = await extractSamples(measure, p.field);
  const n = p.n;
  const k = p.k;
  if (flat.length !== n * k) {
    throw new Error(`sample layout: got ${flat.length} values, expected n*k = ${n * k}`);
  }

  let allFinite = true;
  for (let i = 0; i < flat.length; i++) {
    if (!Number.isFinite(flat[i])) { allFinite = false; break; }
  }
  if (!allFinite) {
    row.status = 'NONFINITE';
    row.ms = Date.now() - t0;
    return row;
  }

  // Atom-major layout: atom a, coordinate i lives at flat[a*k + i]. Pinned by
  // the engine's own iid tests (packages/engine/test/
  // iid-superpose-branch-freshness.test.ts reads `flat[a * k + i]`).
  //
  // WEIGHTED VARIATE MOMENTS. A measure that represents its law by REWEIGHTING
  // uniform positions -- `normalize(weighted(f, Q))`, whose atoms sit at Q's
  // positions -- has an unweighted moment that measures Q, not the measure. The
  // atom weights are the whole law there, so such a probe asks for the
  // self-normalised WEIGHTED accumulators and the caller bands them with the
  // ensemble's effective sample size rather than n. `momentDenom` carries the
  // divisor the caller must use: n for a raw sum, 1 for an already-normalised
  // weighted one.
  const wts = p.weightedVariate ? normalisedWeights(measure, n) : null;
  const sum = new Array(k).fill(0);
  const sumsq = new Array(k).fill(0);
  const cross = new Array(k).fill(0);
  for (let a = 0; a < n; a++) {
    const x0 = flat[a * k];
    const wa = wts ? wts.w[a] : 1;
    for (let i = 0; i < k; i++) {
      const x = flat[a * k + i];
      sum[i] += wa * x;
      sumsq[i] += wa * x * x;
      cross[i] += wa * x0 * x;
    }
  }

  row.status = 'DRAWS';
  row.sum = sum;
  row.sumsq = sumsq;
  row.cross = cross;
  row.momentDenom = wts ? 1 : n;
  if (wts) row.variateNEff = wts.nEff;

  // Coordinate 0, strided down to at most ksSub values, for the scipy KS test.
  // A stride (not a head slice) keeps the subsample spread over the whole
  // stream rather than over its first block.
  const stride = Math.max(1, Math.floor(n / ksSub));
  const ks = [];
  for (let a = 0; a < n && ks.length < ksSub; a += stride) ks.push(flat[a * k]);
  row.ksSample = ks;

  if (measure.logTotalmass !== undefined && measure.logTotalmass !== null) {
    row.logTotalmass = Number(measure.logTotalmass);
  }

  if (p.latent) {
    const lat = await ctx.getMeasure(p.latent);
    if (!lat || !lat.samples) throw new Error('no samples for latent binding ' + p.latent);
    const ts = Array.from(lat.samples);
    if (ts.length !== n) {
      throw new Error(`latent layout: got ${ts.length} values, expected n = ${n}`);
    }
    const { w, nEff } = normalisedWeights(measure, n);
    let et = 0;
    for (let i = 0; i < n; i++) et += w[i] * ts[i];
    if (!Number.isFinite(et)) {
      throw new Error('latent weighted mean is not finite for ' + p.latent);
    }
    row.latentMean = et;
    row.latentNEff = nEff;
  }

  row.ms = Date.now() - t0;
  return row;
}

// A measure's atom weights, normalised to sum to one, plus their effective
// sample size (1 / sum of squared normalised weights). Absent weights means an
// equally-weighted ensemble, which is the uniform log-weight -- not an error,
// and the correct reading of a measure the engine reports no weights for.
function normalisedWeights(measure, n) {
  const lw = measure.logWeights
    ? Array.from(measure.logWeights)
    : new Array(n).fill(0);
  if (lw.length !== n) {
    throw new Error(`weight layout: got ${lw.length} weights, expected n = ${n}`);
  }
  let mx = -Infinity;
  for (const v of lw) if (v > mx) mx = v;
  let z = 0;
  for (const v of lw) z += Math.exp(v - mx);
  const w = new Array(n);
  let sw2 = 0;
  for (let i = 0; i < n; i++) {
    w[i] = Math.exp(lw[i] - mx) / z;
    sw2 += w[i] * w[i];
  }
  if (!Number.isFinite(sw2) || sw2 <= 0) {
    throw new Error('measure weights do not normalise to a finite ensemble');
  }
  return { w, nEff: 1 / sw2 };
}

// A plain measure exposes `samples`; a record-valued one (joint, kchain, …)
// exposes `fields[name]`, each field being a measure in its own right.
async function extractSamples(measure, field) {
  if (field) {
    if (!measure.fields) {
      throw new Error('probe names field ' + field + ' but the measure has no fields (keys: '
        + Object.keys(measure).join(',') + ')');
    }
    const f = await measure.fields[field];
    if (!f) {
      throw new Error('no field ' + field + ' (have: ' + Object.keys(measure.fields).join(',') + ')');
    }
    return Array.from(f.samples || []).map(Number);
  }
  if (measure.samples) return Array.from(measure.samples).map(Number);
  if (measure.fields) {
    throw new Error('measure is record-valued (fields: ' + Object.keys(measure.fields).join(',')
      + ') but the probe names no field');
  }
  throw new Error('measure exposes no samples (keys: ' + Object.keys(measure).join(',') + ')');
}

main().catch((e) => {
  process.stderr.write('sampler_sweep: ' + (e && e.stack ? e.stack : e) + '\n');
  process.exit(1);
});
