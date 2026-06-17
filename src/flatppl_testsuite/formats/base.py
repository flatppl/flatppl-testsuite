"""Abstract contracts for format converters and foreign engines.

Import direction: Importer (foreign -> FlatPPL), scored by the FlatPPL engine.
Export direction: Exporter (FlatPPL -> foreign), scored by a ForeignEngine.
These are the seams; concrete impls live under formats/<fmt>/.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class Importer(ABC):
    @abstractmethod
    def import_(self, source: Path | str) -> str:
        """Convert a foreign-format model to FlatPPL source. May raise a
        format-specific skip signal for unimplemented constructs."""


class Exporter(ABC):
    @abstractmethod
    def export(self, flatppl_src: str) -> str:
        """Convert FlatPPL source to a foreign-format model. No concrete
        implementation exists yet; subclasses are the future seam."""


class ForeignEngine(ABC):
    @abstractmethod
    def twice_delta_nll(self, model, target, scan_param,
                        scan_points, reference) -> list[float]:
        """Run a foreign model in its native engine and return the 2DeltaNLL
        vector over scan_points relative to the reference point."""
