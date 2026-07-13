"""Suite contract + registry. A Suite is a self-contained orchestrator that
uses the toolkit (scoring + formats) and owns its own corpus and check kinds."""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..scoring.result import CheckResult

REGISTRY: dict[str, "Suite"] = {}


class Suite(ABC):
    name: str

    @abstractmethod
    def run(self, selected: set[str] | None = None,
            oracles: tuple[str, ...] = ()) -> list[CheckResult]: ...


def register(suite: "Suite") -> None:
    REGISTRY[suite.name] = suite


def get_suites(names: set[str] | None = None) -> list["Suite"]:
    import flatppl_testsuite.suites.hs3_import  # noqa: F401  (ensure registration)
    import flatppl_testsuite.suites.fragment_gate  # noqa: F401  (ensure registration)
    import flatppl_testsuite.suites.sample_gate  # noqa: F401  (ensure registration)
    import flatppl_testsuite.suites.bayesian_inference_gate  # noqa: F401  (ensure registration)
    import flatppl_testsuite.suites.examples_gate  # noqa: F401  (ensure registration)
    return [s for n, s in REGISTRY.items() if names is None or n in names]
