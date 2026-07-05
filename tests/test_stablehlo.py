"""Shim: run the StableHLO numeric-execution gate tests (defined in
corpora/stablehlo/tests). Skips unless the `stablehlo` pixi env (jax +
enzyme_ad) and a `stablehlo`-capable `flatppl` binary are present."""
from corpora.stablehlo.tests.test_stablehlo_gate import *  # noqa: F401,F403
