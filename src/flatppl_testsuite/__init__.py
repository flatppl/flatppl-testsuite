"""flatppl-hs3-harness — test FlatPPL autoconversion from HS3 against the HS3 suite.

The harness is a co-development feedback loop over three repos: it converts
frozen HS3 fixtures with the flatppl-rust converter, scores the result with the
flatppl-js engine, and compares against the suite's frozen expected values. A
failure to convert or score is a signal to change the converter or the engine,
not something the harness works around.

See docs/ for the design spec and implementation plan. Modules here are
scaffolding; their bodies are filled in by the implementation plan.
"""

__version__ = "0.0.0"
