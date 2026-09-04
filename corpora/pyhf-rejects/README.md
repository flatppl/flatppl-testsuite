# pyhf rejection corpus

31 pyhf documents with no log-density to freeze, so what each row holds is the
converter's **outcome**: its exit code and, for a refusal, a substring of its
message. The refusal half of `corpora/pyhf/`, from the same import audit
(`flatppl-dev/audit-fix-pyhf.md`).

One document per pyhf validation-failure class, plus the six where the
converter and pyhf disagree. A `logpdf_points` row cannot cover this: a
document that exits 1 has nothing to score, and the property under test is that
the converter refuses it *for the same reason pyhf does*.

## The `convert_outcome` check

```json
{
  "id": "convert_outcome",
  "kind": "convert_outcome",
  "expect_exit": 1,
  "stderr_contains": "channel `c` has no samples, so it has no expected counts"
}
```

The substring is the refusal's own sentence with the CLI's `flatppl: pyhf: `
prefix stripped, so the row survives a change to the diagnostic framing while
still pinning the defect the converter names. A row with `expect_exit: 0`
carries no substring: there is no message to match.

Each `test.json` also records what pyhf itself does with the document, in
`pyhf_error`, and derives `pyhf_agrees`. Both sides are **measured**, not
copied: the generator builds the workspace under pyhf and catches what it
raises, and runs `FLATPPL_BIN` for the converter's side.

```sh
FLATPPL_BIN=/path/to/flatppl pixi run -e pyhf gen-pyhf-rejects
```

The pyhf probe runs all the way to a `logpdf` call, not just `Workspace` plus
`model`. pyhf defers some validation to evaluation — an observation of the
wrong length passes `Workspace.data` and only raises `InvalidPdfData` from
`logpdf` — so stopping at model construction would record pyhf as *accepting* a
document it rejects and manufacture a mismatch that is not one.

## Outcomes

23 both refuse, 2 both accept and agree, 6 disagree.

| Document | pyhf | converter | agree | The defect the converter names |
|---|---|---|:--:|---|
| `rej_bad_param_name` | accepts | exit 1 | **no** | parameter name `mu-1` contains `-`, so it cannot be a FlatPPL binding |
| `rej_channel_no_samples` | refuses | exit 1 | yes | channel `c` has no samples, so it has no expected counts |
| `rej_duplicate_channel_name` | refuses | exit 1 | yes | channel name `c` appears twice |
| `rej_duplicate_sample_name` | refuses | exit 1 | yes | channel `c` has two samples named `b` |
| `rej_empty_channels` | refuses | exit 1 | yes | workspace has no channels |
| `rej_empty_sample_data` | refuses | exit 1 | yes | channel `c`: samples have zero bins |
| `rej_histosys_length_mismatch` | refuses | exit 1 | yes | histosys `h` lo.contents has 1 bins but the nominal has 2 |
| `rej_histosys_sigmas_override` | refuses | exit 1 | yes | a `histosys` parameter does not use `sigmas` |
| `rej_lumi_without_config` | refuses | exit 1 | yes | lumi modifier with no `lumi` entry in the measurement config |
| `rej_lumi_wrong_modifier_name` | refuses | exit 1 | yes | same message: the `lumi` entry is what is missing |
| `rej_measurement_missing_poi` | refuses | exit 0 | **no** | — |
| `rej_modifier_extra_key` | refuses | exit 0 | **no** | — |
| `rej_modifier_missing_type` | refuses | exit 1 | yes | missing field `type` |
| `rej_name_reuse_across_types` | refuses | exit 1 | yes | name `t` used by a normfactor and a normsys |
| `rej_negative_nominal` | accepts | exit 0 | yes | both accept; both give NaN |
| `rej_no_measurements` | refuses | exit 0 | **no** | — |
| `rej_no_observation_for_channel` | refuses | exit 1 | yes | no observation data for channel `c` |
| `rej_normsys_sigmas_override` | refuses | exit 1 | yes | a `normsys` parameter does not use `sigmas` |
| `rej_obs_length_mismatch` | refuses | exit 1 | yes | observed data has 1 bins but samples have 2 |
| `rej_ragged_samples` | refuses | exit 1 | yes | sample `b` has 1 bins but expected 2 |
| `rej_reserved_param_name` | accepts | exit 0 | yes | a parameter named `record` converts through `base.record` |
| `rej_sample_missing_data` | refuses | exit 1 | yes | missing field `data` |
| `rej_shapesys_length_mismatch` | refuses | exit 1 | yes | shapesys `g` has 1 per-bin error but the sample has 2 bins |
| `rej_shapesys_shared_across_channels` | refuses | exit 1 | yes | a `shapesys` parameter is per-channel |
| `rej_staterror_factors` | refuses | exit 1 | yes | a `staterror` parameter does not use `factors` |
| `rej_staterror_length_mismatch` | refuses | exit 1 | yes | staterror `st` has 1 bins but the channel has 2 |
| `rej_staterror_siglen` | refuses | exit 1 | yes | 1 `sigmas` value on a parameter that has 2 |
| `rej_undeclared_poi` | refuses | exit 1 | yes | POI `mu` that no modifier declares |
| `rej_unknown_modifier_type` | refuses | exit 1 | yes | unsupported histfactory modifier: nosuchmod |
| `shapefactor_shared_diff_bins` | accepts | exit 1 | **no** | per-bin name shared across channels of unequal bin counts |
| `staterror_shared_across_channels` | accepts | exit 1 | **no** | a `staterror` name spanning channels |

Two documents deliberately share a message. `rej_lumi_wrong_modifier_name` uses
a lumi modifier under a non-constant name; pyhf's schema requires the literal
name `lumi`, and the converter reaches the same conclusion by the route it
takes, so it reports the missing config entry.

## The six mismatches

Each carries a `mismatch_reason` in its `test.json` and is pinned as a
mismatch, not treated as a defect. `tests/core/test_corpus_roster.py` asserts
the mismatch set is exactly these six and that each has a reason, in both
directions: a new mismatch is a finding that needs understanding before it is
pinned, and a mismatch that quietly *resolves* means the converter changed
behaviour on a document whose divergence was deliberate.

**The converter is stricter.** `rej_bad_param_name` — `mu-1` is not a FlatPPL
`Name` (spec §05), so no binding can hold it, and an importer must not rename a
parameter behind the user's back.

**Schema pedantry the converter does not mirror.**
`rej_measurement_missing_poi` (pyhf's schema requires the `poi` key; an *empty*
`poi` string is a deliberate converter accept, so requiring the key would break
that), `rej_no_measurements` (a non-empty `measurements` array; the emitted
module simply has no POI record), and `rej_modifier_extra_key` (pyhf sets
`additionalProperties: false`; serde ignores unknown keys, and
`deny_unknown_fields` is unavailable because `model::Modifier` is shared with
the native HS3 path, which legitimately carries `parameter`, `constraint`,
`constraint_type` and `interpolation`). None can produce a wrong number.

**Refused by design, where pyhf returns something.**

- `shapefactor_shared_diff_bins` is a **pyhf bug**, refused permanently. For a
  per-bin name shared across channels of unequal bin counts pyhf builds a
  model, then reads past the end of its own two-component paramset and takes
  the third channel-B component from the preceding parameter, so it returns a
  number that is not the document's.
- `staterror_shared_across_channels` is a **tripwire**. pyhf accepts a
  staterror name shared across channels and gives it one paramset spanning
  every channel's bins, each channel masking its own slice. The converter
  refuses rather than emit the correlated shape it used to. A follow-up rust
  branch implements the spanning lowering, and when it lands **this row fails
  on the exit code** — that is the signal to move the document into
  `corpora/pyhf/` as a scoring fixture. pyhf's numbers for it are already in
  `flatppl-dev/audit-fix-pyhf.md`: -5.269013574690845 at init and
  -32.9547729941094 at `[1.1, 0.9, 1.2, 0.8]`.

## The gate bites

Run against a `flatppl` built from rust `main` at `0b5fc1c`, the immediate
parent of the first pyhf import fix, **11 of the 31 rows fail**: six documents
the converter then accepted (`rej_duplicate_channel_name`,
`rej_duplicate_sample_name`, `rej_histosys_sigmas_override`,
`rej_normsys_sigmas_override`, `rej_staterror_factors`, `rej_staterror_siglen`)
plus the two cross-channel shapes it had not yet refused, and three whose
message did not name the right defect:

- `rej_channel_no_samples` and `rej_empty_channels` refused for an unrelated
  reason — `measurement 'm' names parameter of interest 'mu', which no modifier
  declares` — reaching an undeclared-POI check before noticing the empty
  channel or the empty sample list. The exit code was right and the reason was
  not, which is exactly what a substring pin catches and a plain exit-code
  assertion would not.
- `rej_histosys_length_mismatch` named the modifier as `histosys '?'` rather
  than `histosys 'h'`.
