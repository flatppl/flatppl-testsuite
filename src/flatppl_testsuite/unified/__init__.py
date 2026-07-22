"""Unified per-test-directory harness (Buffy #358).

One self-contained directory per test — model.flatppl + query.flatppl +
test.json + test.py — loaded and run by `run_test_dir`. Replaces the
per-corpus gate*.py + manifest.json + gen_expected.py + oracle.py spread.
"""
