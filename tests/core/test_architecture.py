"""Architecture guard: toolkit layers must not import suites.

scoring/ and formats/ are lower-level toolkit modules.  suites/ is a
higher-level consumer.  The dependency is one-directional: suites may import
scoring and formats, but scoring and formats must never import suites.
"""

import ast
import pathlib

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "flatppl_testsuite"


def _imports(pyfile):
    tree = ast.parse(pyfile.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            yield node.module
        elif isinstance(node, ast.Import):
            for n in node.names:
                yield n.name


def test_toolkit_does_not_import_suites():
    for layer in ("scoring", "formats"):
        for pyfile in (SRC / layer).rglob("*.py"):
            for mod in _imports(pyfile):
                assert "suites" not in mod, f"{pyfile} imports {mod}"
