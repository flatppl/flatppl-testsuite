# LHCb amplitude models

Shared files for the `corpora/stablehlo/lhcb_*` fixtures. This directory has no
`test.json`, so the harness does not run it.

## Models

The four `.flatppl` files are verbatim copies from
[statsmodel-rosetta-stone](https://github.com/Democratizing-Models/statsmodel-rosetta-stone),
commit `31c309d593018c04b14c06c58940548c4bfcd566`, directory
`src/physics/hadron/amplitude-serialization/`. Their licence is in
`LICENSE-statsmodel-rosetta-stone` (MIT).

| File | Decay | Translated from |
|---|---|---|
| `lc2ppik-lhcb-2683025.flatppl` | Λc+ → p π+ K− ([arXiv:2208.03262](https://arxiv.org/abs/2208.03262)) | amplitude-serialization `models/lc2ppik-lhcb-2683025.json` |
| `lb2pkg-lhcb-2765817.flatppl` | Λb0 → p K− γ ([arXiv:2403.03710](https://arxiv.org/abs/2403.03710)) | amplitude-serialization `models/lb2pkg-lhcb-2765817.json` |
| `b2dxdk-lhcb-2794793-Cminus.flatppl` | B+ → D0 π+ D− K+, model C− ([arXiv:2406.03156](https://arxiv.org/abs/2406.03156)) | B2DxDK.jl `data/b2dxdk_amplitude_model_Cminus.json` |
| `b2dxdk-lhcb-2794793-Cplus.flatppl` | B+ → D0 π+ D− K+, model C+ | B2DxDK.jl `data/b2dxdk_amplitude_model_Cplus.json` |

## Reference values

`references.json` holds the validation points and checksums of the source JSON
files, with each file's commit and SHA-256:

- [amplitude-serialization](https://github.com/RUB-EP1/amplitude-serialization)
  commit `7f01d662226ca0d4c3c7206dc240b7c10b7edefb`, for the Λc and Λb models:
  the intensity at the validation points, and every lineshape at its own σ.
  Licence: `LICENSE-amplitude-serialization` (MIT).
- [B2DxDK.jl](https://github.com/RUB-EP1/B2DxDK.jl) by Alexander Kazatsky and
  Mikhail Mikhasenko (the authors in its `Project.toml`), for the B+ models:
  only reference numbers, from `data/b2dxdk_amplitude_model_{Cminus,Cplus}.json`.
  Commit `85af4cc1e0d244adffeaf2c24dc7be0796da36dd` (`translated_from`) is the
  commit the B+ models were translated from, and their `checked` values are its
  checksums. Commit `9d78e672421b8efe7cfd7b396877c2e2b2b4264c` (`later_points`)
  keeps validation point 1 and replaces points 2 and 3. The model is the same,
  and the fixtures check both sets.

## Oracles

`oracle.py` follows the text of spec §09 for `breakup_momentum`,
`blatt_weisskopf` (through spherical Bessel functions) and
`resonance_breitwigner`. For a density it uses the model's own
`phase_space_density` binding. Each fixture's `test.py` documents its oracle.
Regenerate the frozen values with
`pixi run regen corpora/stablehlo/lhcb_*`.
