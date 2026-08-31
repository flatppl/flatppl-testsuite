"""The sampler sweep's CI gate.

Five signals, mirroring the density gate's (`tests/sweep/test_gate.py`) but on
the sample path:

    DRAWS but a check failed      -> a wrong number (the class this exists to find)
    DRAWS where the table REFUSES -> newly admitted; needs an oracle verdict
    REFUSES where the table DRAWS -> a regression, or an over-refusal
    a changed refusal marker      -> the reason moved; re-read it
    MALFORMED anywhere            -> always a defect

Plus two guards the density gate has no equivalent of, both specific to a
STATISTICAL gate:

* THE TEETH TESTS. A statistical gate whose bands are too wide passes
  everything and proves nothing, and that failure mode is invisible from a green
  run. So the tolerance arithmetic is tested directly against the magnitudes of
  the two defects this sweep was built for — the `@stdlib` symmetric-Beta
  variance bias and the `iid(superpose(...))` branch pinning — asserting each
  would be caught, and by how many sigma. These need no engine and run in
  milliseconds.

* PROVENANCE IS NOT ONE OF THE FIVE SIGNALS. An engine commit that differs from
  the table's pin is metadata skew, and it is the NORMAL state: CI clones
  flatppl-js main every run, so the pin sits behind it from the next merge
  onward. The provenance test is marked `provenance` and deselected from
  `pixi run test` by pytest.ini; `pixi run provenance` reports it, and CI runs
  that in a step that annotates and exits 0. Only the five signals above -- all
  of them draws, outcomes or check statuses -- make a run red. `pixi run repin`
  moves the pin.

* ROSTER COMPLETENESS. The roster claims to cover every sampleable REGISTRY
  entry. That claim decays silently the moment a new distribution lands in
  flatppl-js, so it is asserted against `sampler-registry.ts` itself rather than
  maintained by hand.
"""
from __future__ import annotations

import dataclasses
import math
import re
from types import SimpleNamespace

import pytest

from flatppl_testsuite.sampler_sweep import checks as C
from flatppl_testsuite.sampler_sweep import engine, space, table
from flatppl_testsuite.sampler_sweep.oracle import DENSITY_ONLY, FAMILIES

N = space.N_DRAWS


def _engine_available() -> bool:
    try:
        engine.resolve_engine_dir()
    except RuntimeError:
        return False
    return True


# ---------------------------------------------------------------------------
# Teeth. No engine, no table — pure tolerance arithmetic against known defects.
# ---------------------------------------------------------------------------

# From flatppl-dev/TODO-flatppl-js.md: `@stdlib/random-base-beta@0.2.2` draws
# every symmetric `Beta(a, a)` with `a > 1.5` with a variance biased LOW, by
# 3.09% at a = 2. flatppl-js routes that region through two gammas
# (`randBetaFixed` in sampler-registry.ts) to avoid it.
BETA_22_VAR = 0.05
BETA_22_MU4 = 0.005357142857
BETA_DEFECT_REL = 0.0309

# The mixture the IIDSUPER rows use: Normal(-3,1) / Normal(+3,1), equal weights.
MIX_VAR = 10.0
MIX_MU4 = 138.0


def test_the_beta_vendor_variance_defect_would_be_caught():
    biased = BETA_22_VAR * (1 - BETA_DEFECT_REL)
    chk = C.check_var(0, biased, BETA_22_VAR, BETA_22_MU4, N)
    assert chk.status == "failed", f"the Beta(2,2) vendor defect would pass: {chk.detail}"
    assert chk.sigma > 10, f"caught, but only at {chk.sigma:.1f} sigma: {chk.detail}"


def test_the_beta_row_resolves_a_variance_bias_well_under_the_defect():
    """The band must be tight enough that the defect is not a borderline catch."""
    se = math.sqrt((BETA_22_MU4 - BETA_22_VAR ** 2) / N)
    resolvable_rel = C.SIGMA * se / BETA_22_VAR
    assert resolvable_rel < BETA_DEFECT_REL / 2, (
        f"the row resolves only a {100 * resolvable_rel:.2f}% variance bias, "
        f"which leaves no margin under the {100 * BETA_DEFECT_REL:.2f}% defect")


def test_iid_superpose_branch_pinning_would_be_caught_on_covariance():
    """The check that actually catches pinning.

    A pinned coordinate locks to one component, so cov(0, i) goes to the mixture
    variance instead of 0. This is the signal that does NOT depend on a marginal
    looking wrong — see the next test for why that matters.
    """
    chk = C.check_cov(1, MIX_VAR, 0.0, MIX_VAR, N)
    assert chk.status == "failed"
    assert chk.sigma > 100, f"only {chk.sigma:.1f} sigma: {chk.detail}"


def test_a_pinned_coordinate_can_have_a_plausible_marginal_variance():
    """Why the covariance check is not redundant with the moment checks.

    The historically observed pinning left the MIDDLE coordinate's variance at
    9.9860 against an oracle of 10.0 — 1.0 sigma, a clean pass. The defect was
    only visible across coordinates. A gate with moment checks alone would have
    missed it on that coordinate.
    """
    marginal = C.check_var(1, 9.9860, MIX_VAR, MIX_MU4, N)
    assert marginal.status == "passed", (
        "this test encodes the historical observation that a pinned coordinate's "
        "own variance looked fine; if it now fails, re-read the premise")
    cross = C.check_cov(1, MIX_VAR, 0.0, MIX_VAR, N)
    assert cross.status == "failed", "the cross-coordinate check must catch what the marginal missed"


def test_the_carded_residual_covariance_would_be_caught():
    """TODO-flatppl-js.md still cards `iid(kchain(superpose, K), n)` with a
    residual covariance of 0.4791 against an oracle of 0. Far smaller than full
    pinning, so it is the real sensitivity floor for this class."""
    chk = C.check_cov(1, 0.4791, 0.0, MIX_VAR, N)
    assert chk.status == "failed", f"the carded residual would pass: {chk.detail}"
    assert chk.sigma > 10, f"only {chk.sigma:.1f} sigma: {chk.detail}"


# The pooled-divisor defect these rows exist for. A theta-dependent `normalize`
# divided by the POOLED mass leaves atom i the residue Z(theta_i)/E[Z], which
# tilts the theta-marginal from the prior to E[theta Z]/E[Z]. Every probe
# carrying a `latent_tilt` records that closed form.
#
# The band is `5 * sqrt(prior var / n_eff)`, and `n_eff` comes from the run. This
# test uses a PESSIMISTIC quarter of the draw count, well below the lowest ESS
# any of these rows produces (about 144k of 200k on the weighted-box row), so a
# green result here does not depend on the weights staying near-uniform.
_PESSIMISTIC_N_EFF = N / 4


def test_the_pooled_normalize_divisor_would_be_caught_on_every_theta_row():
    rows = [p for p in space.enumerate_probes() if p.latent_tilt is not None]
    assert rows, "no theta-dependent normalize rows in the space"
    for p in rows:
        chk = C.check_latent_mean(p.latent_tilt, p.latent_mean, p.latent_var,
                                  _PESSIMISTIC_N_EFF)
        assert chk.status == "failed", \
            f"{p.id}: the Z-tilted marginal would pass: {chk.detail}"
        assert chk.sigma > 20, \
            f"{p.id}: caught, but only at {chk.sigma:.1f} sigma: {chk.detail}"


def test_the_theta_rows_do_not_fire_on_the_prior_itself():
    """The other half: the band must accept the value the spec requires."""
    for p in space.enumerate_probes():
        if p.latent_tilt is None:
            continue
        chk = C.check_latent_mean(p.latent_mean, p.latent_mean, p.latent_var,
                                  _PESSIMISTIC_N_EFF)
        assert chk.status == "passed", f"{p.id}: {chk.detail}"


# The mixing-weight row's teeth (flatppl-js #233). `matSuperpose` SIR-resampled a
# parent whose per-atom weights were non-uniform to equal weights carrying the
# POOLED total mass, so §06 normalize's own mixture spelling mixed every atom at
# E[p] instead of p_i and the latent came out INDEPENDENT of the variate. The
# failing hypothesis is therefore cov = 0, recorded as `latent_cov_null`.
#
# BOTH marginals stay correct under that defect -- E[p] is the prior and E[y] is
# linear in p -- so this is also the one row where `check_latent_mean` proves
# nothing, which is why the covariance check exists.


def test_a_pooled_mixing_proportion_would_be_caught_on_the_covariance():
    rows = [p for p in space.enumerate_probes() if p.latent_cov_null is not None]
    assert rows, "no latent mixing-weight rows in the space"
    for p in rows:
        chk = C.check_latent_cov(p.latent_cov_null, p.latent_cov,
                                 p.latent_cov_var, _PESSIMISTIC_N_EFF)
        assert chk.status == "failed", \
            f"{p.id}: a decoupled mixing proportion would pass: {chk.detail}"
        assert chk.sigma > 20, \
            f"{p.id}: caught, but only at {chk.sigma:.1f} sigma: {chk.detail}"


def test_the_mixing_row_does_not_fire_on_the_closed_form_covariance():
    """The other half: the band must accept the value §06 requires."""
    for p in space.enumerate_probes():
        if p.latent_cov_null is None:
            continue
        chk = C.check_latent_cov(p.latent_cov, p.latent_cov,
                                 p.latent_cov_var, _PESSIMISTIC_N_EFF)
        assert chk.status == "passed", f"{p.id}: {chk.detail}"


def test_the_mixing_rows_marginals_prove_nothing_on_their_own():
    """The trap, asserted. A latent mixing weight leaves the latent's own
    marginal exactly on its prior, so `check_latent_mean` PASSES at the defect.
    A future reader must not read that pass as coverage of this shape."""
    for p in space.enumerate_probes():
        if p.latent_cov_null is None:
            continue
        chk = C.check_latent_mean(p.latent_mean, p.latent_mean, p.latent_var,
                                  _PESSIMISTIC_N_EFF)
        assert chk.status == "passed", \
            f"{p.id}: the latent marginal is the prior under the defect too"
        assert p.latent_tilt is None, \
            f"{p.id}: the mass is constant here, so there is no Z-tilt to record"


def test_a_dropped_importance_weight_would_be_caught_on_every_weighted_row():
    """The teeth of every `weighted_variate` row.

    Both rows carry the SAME failing hypothesis, because both represent their law
    by reweighting `Normal(0, 1)` positions through the conjugate tilt e^x: drop
    the weights and the reported mean is the unnormalized base's 0 where the
    oracle is 1.

    `iid`'s composite fallback dropped the inner measure's per-position weights
    (flatppl-js #232). A MIS-FOLD is the other failure that row must catch:
    reading one position's weight instead of the block's product leaves
    coordinate 0 right and the rest at the base's mean, which is why every
    coordinate is checked and not a pooled mean.

    `matSample` dropped a PARAMETER measure's weights, which leaves the drawn
    variate's ensemble at the same 0. There the other failure is a DOUBLE count
    of the one stream, which reports 2 -- equally far from the oracle, so this
    band rejects it too and the test asserts that explicitly.

    Each row bands at a pessimistic quarter of ITS OWN effective count, derived
    from `weight_log_var` (the closed-form variance of the atom weight's log), so
    the teeth do not rest on the ESS a run happens to report and no row borrows
    another's weight distribution.
    """
    rows = [p for p in space.enumerate_probes() if p.weighted_variate]
    assert rows, "no weighted-variate rows in the space"
    for p in rows:
        assert p.weight_log_var is not None, \
            f"{p.id}: a weighted_variate row must record its weight's log variance"
        n_eff = p.n_draws / 4 * math.exp(-p.weight_log_var)
        # 0 is the dropped-weight reading; 2 * p.mean is the doubled-stream one,
        # the same distance the other way.
        for wrong, why in ((0.0, "the unnormalized base's mean"),
                           (2.0 * p.mean, "a doubled weight stream")):
            for i in range(p.k):
                chk = C.check_mean(i, wrong, p.mean, p.var, n_eff)
                assert chk.status == "failed", \
                    f"{p.id} coord {i}: {why} would pass: {chk.detail}"
                assert chk.sigma > 20, \
                    f"{p.id} coord {i}: {why} caught, but only at {chk.sigma:.1f} sigma"


def test_a_per_coordinate_parameter_redraw_would_be_caught_on_the_covariance():
    """The teeth of every row whose coordinates share a stochastic parameter.

    §06 `iid` makes the coordinates independent GIVEN the parameter, so they
    share its whole variance. A fix that re-drew the parameter per coordinate
    would leave every mean and variance right and read cov = 0 instead, so the
    covariance is the only check that can see it.
    """
    rows = [p for p in space.enumerate_probes() if p.cov not in (None, 0.0)]
    assert rows, "no shared-parameter rows in the space"
    for p in rows:
        for i in range(1, p.k):
            chk = C.check_cov(i, 0.0, p.cov, p.var, _PESSIMISTIC_N_EFF)
            assert chk.status == "failed", \
                f"{p.id} coord {i}: a per-coordinate re-draw would pass: {chk.detail}"
            assert chk.sigma > 20, \
                f"{p.id} coord {i}: caught, but only at {chk.sigma:.1f} sigma"


def test_a_dropped_or_negated_vector_shift_would_be_caught():
    """Why `mean_by_coord` exists.

    A shift whose components are equal is invisible to a dropped or sign-flipped
    component. The one row that carries distinct components pins each coordinate
    against its own mean, so both failure modes move a coordinate by the full
    shift.
    """
    rows = [p for p in space.enumerate_probes() if p.mean_by_coord is not None]
    assert rows, "no per-coordinate-mean rows in the space"
    for p in rows:
        for i, want in enumerate(p.mean_by_coord):
            for wrong in (0.0, -want):           # shift dropped, shift negated
                if wrong == want:
                    continue
                chk = C.check_mean(i, wrong, want, p.var, N)
                assert chk.status == "failed", \
                    f"{p.id} coord {i}: {wrong} would pass against {want}: {chk.detail}"


def test_the_bands_do_not_fire_on_true_null_noise():
    """A 5-sigma band must not cost false positives, or the gate becomes noise
    everyone learns to ignore. 4000 true-null draws of each estimator."""
    import numpy as np

    rng = np.random.default_rng(0)
    mean_se = math.sqrt(MIX_VAR / N)
    var_se = math.sqrt((MIX_MU4 - MIX_VAR ** 2) / N)
    mean_fp = sum(1 for _ in range(4000)
                  if C.check_mean(0, rng.normal(0.0, mean_se), 0.0, MIX_VAR, N).status == "failed")
    var_fp = sum(1 for _ in range(4000)
                 if C.check_var(0, MIX_VAR + rng.normal(0.0, var_se),
                                MIX_VAR, MIX_MU4, N).status == "failed")
    assert mean_fp == 0, f"{mean_fp}/4000 false positives on the mean band"
    assert var_fp == 0, f"{var_fp}/4000 false positives on the variance band"


def test_ks_catches_a_shape_defect_the_moments_would_miss():
    """A mis-weighted mixture keeps a plausible spread but the wrong shape."""
    import numpy as np
    from scipy import stats

    spec = ("mix", (0.5, 0.5), (("norm", (-3.0, 1.0), {}), ("norm", (3.0, 1.0), {})))
    # True 0.5/0.5 mixture: passes.
    good = np.concatenate([stats.norm(-3, 1).rvs(10000, random_state=2),
                           stats.norm(3, 1).rvs(10000, random_state=3)])
    assert C.check_ks(list(good), spec, 20000).status == "passed"
    # 0.4/0.6 instead: caught.
    bad = np.concatenate([stats.norm(-3, 1).rvs(8000, random_state=4),
                          stats.norm(3, 1).rvs(12000, random_state=5)])
    chk = C.check_ks(list(bad), spec, 20000)
    assert chk.status == "failed", f"a 0.4/0.6 mis-weighting would pass KS: {chk.detail}"
    # A pinned coordinate: caught by a wide margin.
    pinned = stats.norm(-3, 1).rvs(20000, random_state=1)
    assert C.check_ks(list(pinned), spec, 20000).status == "failed"


def test_totalmass_catches_a_mass_confusion():
    """`weighted(2.0, M)` has mass 2, not 1. The band is float-precision, since
    the engine computes this in closed form rather than estimating it."""
    log2 = 0.6931471805599453
    assert C.check_totalmass(log2, log2).status == "passed"
    assert C.check_totalmass(0.0, log2).status == "failed", "mass 1 vs 2 must not pass"
    assert C.check_totalmass(log2 * (1 + 1e-6), log2).status == "failed", \
        "a 1e-6 relative mass error must not pass"


# ---------------------------------------------------------------------------
# Roster completeness — asserted against the engine, not maintained by hand.
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _engine_available(), reason="no flatppl-js checkout found anywhere")
def test_the_roster_covers_every_sampleable_registry_entry():
    # `resolve_engine_dir`, not `CONFIG.flatppl_js_dir`: from a testsuite worktree
    # the CONFIG sibling is the parked stale flatppl-js clone, so this assertion
    # would validate the roster against a months-old registry and report green.
    root, _why = engine.resolve_engine_dir()
    registry = root / "packages" / "engine" / "sampler-registry.ts"
    if not registry.exists():
        pytest.skip(f"no flatppl-js engine at {registry}")
    src = registry.read_text()
    body = src[src.index("const REGISTRY = {"):]
    body = body[:body.index("\n};")]

    # Top-level entries are `  Name: {` at exactly two spaces of indent; an
    # entry is density-only when its block carries `densityOnly: true`.
    entries = re.findall(r"^  ([A-Za-z0-9_]+): \{$", body, re.M)
    assert len(entries) > 20, f"parsed only {len(entries)} REGISTRY entries — did the file move?"

    blocks = re.split(r"^  (?=[A-Za-z0-9_]+: \{$)", body, flags=re.M)
    density_only = set()
    for b in blocks:
        m = re.match(r"([A-Za-z0-9_]+): \{", b)
        if m and "densityOnly: true" in b:
            density_only.add(m.group(1))

    sampleable = set(entries) - density_only
    covered = {f.measure.split("(")[0] for f in FAMILIES}

    assert density_only == {name for name, _why in DENSITY_ONLY}, (
        "oracle.DENSITY_ONLY disagrees with sampler-registry.ts:\n"
        f"  registry says: {sorted(density_only)}\n"
        f"  oracle says:   {sorted(name for name, _ in DENSITY_ONLY)}")
    assert sampleable == covered, (
        "the roster no longer matches the engine's sampleable REGISTRY:\n"
        f"  in the registry but not the roster: {sorted(sampleable - covered)}\n"
        f"  in the roster but not the registry: {sorted(covered - sampleable)}")


def test_engine_resolution_does_not_pick_a_checkout_parked_under_worktrees(monkeypatch):
    """A flatppl-js clone inside `flatppl-testsuite/.worktrees/` is another
    repo's tree parked in this repo's worktree directory, and is never the engine
    a testsuite worktree means to load. One is parked there today at a commit
    months behind main, and it is a REAL checkout, so only this rule excludes it.

    This guards the DEFAULT resolution: a deliberate `FLATPPL_JS_DIR` override
    may legitimately name a worktree (a branch landing gate, a bisect). The
    override lives in `CONFIG.flatppl_js_dir`, snapshotted at import — clearing
    the environment variable is too late — so the CONFIG attribute is reset to
    the computed default here.
    """
    monkeypatch.delenv("FLATPPL_JS_DIR", raising=False)
    monkeypatch.setattr(
        engine, "CONFIG",
        dataclasses.replace(
            engine.CONFIG, flatppl_js_dir=str(engine._config_default_path())))
    root, why = engine.resolve_engine_dir()
    assert ".worktrees" not in root.resolve().parts, (
        f"resolved the engine to {root} ({why}), which sits under a .worktrees "
        f"directory — the sweep would draw from a parked, probably stale checkout")


def test_a_named_engine_checkout_that_does_not_exist_is_an_error(monkeypatch, tmp_path):
    """A typo'd `FLATPPL_JS_DIR` must not resolve to a different engine.

    Falling through to the workspace root hands the operator a tree they did not
    name, and the sweep then reports green against it. The path is non-default, so
    it is a deliberate choice, and a deliberate choice is never substituted.
    """
    missing = tmp_path / "nope-js"
    monkeypatch.setattr(engine, "CONFIG", SimpleNamespace(flatppl_js_dir=missing))
    with pytest.raises(RuntimeError) as excinfo:
        engine.resolve_engine_dir()
    assert str(missing) in str(excinfo.value), \
        "the error must name the checkout the operator asked for"


def test_the_accidental_default_still_falls_back_instead_of_erroring(monkeypatch):
    """The N3 error must not swallow the H1 filtering.

    A default-valued path parked under `.worktrees/` is an accident, not a
    choice, so it still falls back to the workspace root rather than raising.
    """
    monkeypatch.setattr(
        engine, "CONFIG", SimpleNamespace(flatppl_js_dir=engine._config_default_path()))
    root, why = engine.resolve_engine_dir()
    assert ".worktrees" not in root.resolve().parts, f"resolved to {root} ({why})"


def test_the_frozen_metadata_carries_no_machine_local_path():
    """The table is tracked, so an absolute path in it churns per machine and
    diverges on CI. The engine commit identifies the engine; `engine_resolved_by`
    records a resolution KIND, not a path."""
    meta, rows = table.load()
    if not rows:
        pytest.skip("no committed sampler table")
    local = {k: v for k, v in meta.items() if isinstance(v, str) and v.startswith("/")}
    assert not local, f"absolute local paths frozen into the tracked table: {local}"


def test_a_changed_refusal_reason_diffs_even_when_the_marker_is_unchanged():
    """M1: distinct engine guards share a coarse marker, so the gate compares
    the full normalised message.

    The premise is not one pair of guards but the density of the collision. Four
    unrelated `is not supported` messages live in `packages/engine/*.ts` today
    (`materialiser.ts:517`, `mat-density.ts:851`, `mat-broadcast.ts:1116`,
    `sampler-aggregate.ts:930`), and `_marker` maps every one of them to
    `is-not-supported`. Any refusal moving between two of them keeps the marker,
    so only a verbatim message compare can show that the reason changed.

    Both strings below are copied verbatim from the engine, so a reworded guard
    breaks this test's premise assertion rather than letting it pass on
    strings the test authored itself.
    """
    ensemble = ("iid: sampling iid over a record measure at >1 atoms "
                "(an ensemble of tables) is not supported; sample a single dataset (1 atom)")
    truncate_cap = ("density: normalize(truncate(M, cartprod)) — region dimension 4 "
                    "exceeds the quadrature cap of 3; higher-dimensional truncation "
                    "is not supported")
    assert table._marker(ensemble) == table._marker(truncate_cap) == "is-not-supported", (
        "premise of this test: two unrelated real guards share one marker")

    def row(err):
        return table.Row(probe_id="p", family="f", wrap="w", outcome="REFUSES",
                         n=1, k=1, marker=table._marker(err), error=err)

    problems = table.diff({"p": row(ensemble)}, {"p": row(truncate_cap)})
    assert problems, "a changed refusal reason produced no diff"
    assert "the reason changed" in problems[0]
    # And an unchanged reason must stay quiet.
    assert not table.diff({"p": row(ensemble)}, {"p": row(ensemble)})


def test_every_probe_has_at_least_one_checkable_oracle():
    """A row that checks nothing is a row that proves nothing.

    A `refusal_only_reason` row is the one exception, and it is not an escape
    hatch: its oracle is the frozen `REFUSES` outcome plus the verbatim error
    `table.diff` compares, which is the only honest oracle for a shape the spec
    makes ill-typed. Inventing a moment for such a shape would pin a number no
    correct engine produces.
    """
    for p in space.enumerate_probes():
        checkable = (p.mean is not None or p.mean_by_coord is not None
                     or p.var is not None
                     or p.cov is not None or p.ks is not None
                     or p.logtotalmass is not None
                     or p.latent_mean is not None
                     or p.latent_cov is not None)
        assert checkable or p.refusal_only_reason, (
            f"{p.id} carries no oracle of any kind")


def test_a_refusal_only_row_actually_refuses():
    """`refusal_only_reason` says the shape has no correct draw, so the table
    must record it REFUSING. A row that declares the exemption and then DRAWS is
    either mis-declared or a live over-refusal that healed, and either way it is
    now a row with no oracle at all."""
    _meta, table_rows = table.load()
    if not table_rows:
        pytest.skip("no committed sampler table — run `pixi run sampler-sweep-regen`")
    for p in space.enumerate_probes():
        if not p.refusal_only_reason:
            continue
        row = table_rows.get(p.id)
        assert row is not None, f"{p.id} is not in the committed table"
        assert row.outcome == table.Outcome.REFUSES.value, (
            f"{p.id} declares refusal_only_reason but the table records "
            f"{row.outcome} — it now checks nothing")
        assert row.error, f"{p.id} refuses with no message to freeze"


def test_replicated_wraps_all_carry_a_covariance_oracle():
    """§06 makes `iid` a product measure, so every k > 1 row must pin the
    cross-coordinate covariance. A k > 1 row without one is the IIDSUPER blind
    spot re-opened.

    The oracle is 0 whenever the coordinates share no stochastic parameter.
    `iid` makes them independent GIVEN such a parameter, so a row that shares
    one pins the parameter's variance instead, and it must name the parameter in
    `latent` so a reader sees why the oracle is not 0.
    """
    for p in space.enumerate_probes():
        if p.k > 1:
            assert p.cov is not None, f"{p.id} has k={p.k} but no cov oracle"
            if p.cov != 0.0:
                assert p.latent is not None, \
                    f"{p.id} pins cov={p.cov} but names no shared parameter"


# The margin the gate's stability rests on. `diff` compares each check's
# STATUS, never its estimate, so a live engine that re-plumbs which RNG stream
# feeds which distribution moves the digits without moving a verdict -- measured
# 2026-08-28 by re-running the whole roster at a different seed, which is a
# strictly stronger perturbation than any re-plumbing: 353 of 457 estimates
# moved, `diff` reported nothing. What that argument needs is HEADROOM. A row
# frozen at 4.9 sigma passes today and flips to red on the next reseed, which
# would read as engine drift and is really an eroded band.
#
# The worst check measured 2.50 sigma as frozen and 2.72 at the other seed, so
# 4.0 leaves room for ordinary movement while still catching erosion. Free: it
# reads the committed table and calls no engine.
_MARGIN_SIGMA = 4.0


def test_no_frozen_check_sits_near_its_band():
    _meta, rows = table.load()
    if not rows:
        pytest.skip("no committed sampler table")
    near = []
    for r in rows.values():
        for c in r.checks:
            if c["status"] == "skipped" or c.get("sigma") is None:
                continue
            if c["sigma"] > _MARGIN_SIGMA:
                near.append(f"{r.probe_id} {c['name']}: {c['detail']}")
    assert not near, (
        f"checks frozen above {_MARGIN_SIGMA} sigma of their {C.SIGMA}-sigma band. "
        "Either a real bias the band nearly catches, or a margin thin enough that "
        "an unrelated engine change flips it to red:\n  " + "\n  ".join(near))


# ---------------------------------------------------------------------------
# The live diff against the frozen table.
# ---------------------------------------------------------------------------

@pytest.mark.provenance
@pytest.mark.skipif(not _engine_available(), reason="no flatppl-js checkout found anywhere")
def test_the_running_engine_matches_the_table_it_is_compared_against():
    """Reported, never blocking -- see this module's docstring.

    Still worth reporting, and it still FAILS rather than skips when run: the
    engine moving is the single most likely reason for this sweep's numbers to
    change, so a reader looking at row drift needs to know whether the two sides
    are the same engine. An unknown commit on either side reports too, rather
    than reading as agreement.
    """
    problem = table.check_provenance()
    assert problem is None, problem


@pytest.mark.skipif(not _engine_available(), reason="no flatppl-js checkout found anywhere")
def test_the_provenance_gate_rejects_a_stale_unknown_or_missing_commit():
    """The provenance gate must actually fire — including on UNKNOWN.

    Exercised by mutating a copy of the frozen table rather than by moving the
    engine, so the shared flatppl-js checkout is never touched.
    """
    import json
    import tempfile
    from pathlib import Path

    payload = json.loads(table.DEFAULT_PATH.read_text())
    cases = {
        "stale": "e9803b6afdf9e183f9e0616697fc4523ac700e68",
        "unknown": "unknown",
        "missing": None,
    }
    for label, commit in cases.items():
        mutated = json.loads(json.dumps(payload))
        if commit is None:
            mutated["metadata"].pop("engine_commit", None)
        else:
            mutated["metadata"]["engine_commit"] = commit
        path = Path(tempfile.mkdtemp()) / "mutated.json"
        path.write_text(json.dumps(mutated))
        problem = table.check_provenance(path)
        assert problem is not None, f"a {label} engine_commit was accepted as provenance"
        assert "provenance mismatch" in problem
        assert table.engine_commit()[:12] in problem, \
            "the message must name the commit actually running"


def test_the_frozen_table_exists_and_is_self_consistent():
    meta, rows = table.load()
    if not rows:
        pytest.skip("no committed sampler table — run `pixi run sampler-sweep-regen`")
    assert meta["probe_count"] == len(rows)
    assert meta["n_draws"] == space.N_DRAWS, (
        "the table was frozen at a different draw count, so its tolerances are not "
        "this code's tolerances — refreeze")
    assert meta["seed"] == space.SEED, "the table was frozen at a different seed — refreeze"
    assert set(rows) == {p.id for p in space.enumerate_probes()}, \
        "the frozen table's probe set differs from the current space — refreeze"


def test_no_malformed_row_is_frozen():
    """MALFORMED is always a defect, so none may be baked in as expected."""
    _meta, rows = table.load()
    if not rows:
        pytest.skip("no committed sampler table")
    bad = {pid: r.error for pid, r in rows.items() if r.outcome == "MALFORMED"}
    assert not bad, f"MALFORMED rows frozen in the table: {bad}"


def test_no_failing_check_is_frozen_without_being_listed():
    """A failing row may exist — it is a finding — but it must be declared in the
    metadata, so a NEW one cannot hide among the old ones."""
    meta, rows = table.load()
    if not rows:
        pytest.skip("no committed sampler table")
    failing = sorted(pid for pid, r in rows.items() if r.failed)
    assert failing == sorted(meta.get("failing_rows", [])), (
        "rows with a failing check do not match metadata.failing_rows — refreeze")


@pytest.mark.skipif(not _engine_available(), reason="no flatppl-js checkout found anywhere")
def test_live_sweep_matches_the_committed_table():
    """The diff itself.

    Runs even on a provenance mismatch: the dedicated provenance test above
    already names that cause, and seeing the actual damage is more useful than
    suppressing it. Two red tests, one of which explains the other.
    """
    _meta, expected = table.load()
    if not expected:
        pytest.skip("no committed sampler table — run `pixi run sampler-sweep-regen`")
    actual = {r.probe_id: r for r in table.sweep()}
    problems = table.diff(expected, actual)
    assert not problems, "sampler sweep diverged from the committed table:\n" + "\n".join(problems)
