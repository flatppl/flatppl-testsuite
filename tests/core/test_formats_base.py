import pytest
from flatppl_testsuite.formats.base import Importer, Exporter, ForeignEngine


def test_abcs_cannot_instantiate():
    for cls in (Importer, Exporter, ForeignEngine):
        with pytest.raises(TypeError):
            cls()


def test_unimplemented_exporter_is_a_clear_seam():
    # A concrete Exporter subclass that hasn't implemented export raises clearly.
    class HS3Exporter(Exporter):
        def export(self, flatppl_src):
            raise NotImplementedError("FlatPPL->HS3 export not implemented yet")
    with pytest.raises(NotImplementedError):
        HS3Exporter().export("mu = elementof(reals)\n")
