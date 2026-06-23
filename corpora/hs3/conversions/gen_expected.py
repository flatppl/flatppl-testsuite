#!/usr/bin/env python3
"""Generate ``conversions/<model>/expected.json`` from the ROOT/RooFit oracle.

For each HS3 paper example, evaluate the converted likelihood's 2DeltaNLL over a
set of theta points with ROOT (the authority) and freeze it as ``expected.json``
(check kind ``twice_delta_nll_points``). The harness then scores the committed
``<model>.flatppl`` with the FlatPPL engine and compares against this frozen ROOT
vector — the same score+compare loop as the fixtures, with no hand-copied
numbers (which is what let ``repro_hs3_js.cjs`` silently rot when the converter
output changed).

Each point is given in two views:

  - ``root``   — flat RooFit variable names (what ``ws.var(...).setVal`` takes);
                 HistFactory's per-bin MC-stat factors are ``mcstat_0``/``mcstat_1``.
  - ``record`` — the FlatPPL ``record(...)`` passed to ``logdensityof`` (vector
                 params grouped, e.g. ``mcstat = [g0, g1]``).

``points[0]`` is the reference; ``expected[0]`` is 0 by construction. The
2DeltaNLL difference is offset-invariant, so HistFactory's ROOT
``Sum log(n_k!)`` convention offset drops out and the absolute log-densities
need not agree.

Run in the ROOT pixi env:

    pixi run -e root python corpora/hs3/conversions/gen_expected.py
"""
from __future__ import annotations

import json
from pathlib import Path

import ROOT

HERE = Path(__file__).resolve().parent

MODELS = {
    "gaussian": {
        "binding": "obs",
        "pdf": "gauss_x",
        "data": "obs_gaussian_channel",
        "globals": [],
        # sigma is a fixed constant in the ROOT workspace, so it is not a ROOT
        # var to set — only mu moves; the FlatPPL record carries sigma = 1.0.
        "points": [
            {"root": {"mu": 0.0},  "record": {"mu": 0.0,  "sigma": 1.0}},
            {"root": {"mu": 0.5},  "record": {"mu": 0.5,  "sigma": 1.0}},
            {"root": {"mu": 1.27}, "record": {"mu": 1.27, "sigma": 1.0}},
        ],
        "tolerance": {"atol": 1e-7, "rtol": 1e-8},
    },
    "product": {
        "binding": "likelihood",
        "pdf": "prod",
        "data": "toy",
        "globals": [],
        "points": [
            {"root": {"mu1": 0.0, "sigma1": 1.0, "mu2": 1.0, "sigma2": 2.0},
             "record": {"mu1": 0.0, "sigma1": 1.0, "mu2": 1.0, "sigma2": 2.0}},
            {"root": {"mu1": 0.5, "sigma1": 1.0, "mu2": 1.0, "sigma2": 2.0},
             "record": {"mu1": 0.5, "sigma1": 1.0, "mu2": 1.0, "sigma2": 2.0}},
        ],
        # ROOT normalizes the product pdf with a numeric integral, which sets
        # the achievable agreement.
        "tolerance": {"atol": 2e-4, "rtol": 1e-4},
    },
    "histfactory": {
        "binding": "likelihood",
        "pdf": "model_channel1",
        "data": "observed_channel1",
        "globals": ["mu", "syst1", "syst2", "syst3", "mcstat_0", "mcstat_1"],
        "points": [
            {"root": {"mu": 1.0, "syst1": 0.0, "syst2": 0.0, "syst3": 0.0, "mcstat_0": 1.0, "mcstat_1": 1.0},
             "record": {"mu": 1.0, "syst1": 0.0, "syst2": 0.0, "syst3": 0.0, "mcstat": [1.0, 1.0]}},
            {"root": {"mu": 1.5, "syst1": 0.5, "syst2": 0.0, "syst3": 0.0, "mcstat_0": 1.1, "mcstat_1": 1.0},
             "record": {"mu": 1.5, "syst1": 0.5, "syst2": 0.0, "syst3": 0.0, "mcstat": [1.1, 1.0]}},
            {"root": {"mu": 0.5, "syst1": -0.3, "syst2": 0.2, "syst3": 0.0, "mcstat_0": 0.9, "mcstat_1": 1.1},
             "record": {"mu": 0.5, "syst1": -0.3, "syst2": 0.2, "syst3": 0.0, "mcstat": [0.9, 1.1]}},
            {"root": {"mu": 2.0, "syst1": 1.0, "syst2": -1.0, "syst3": 0.5, "mcstat_0": 1.2, "mcstat_1": 0.8},
             "record": {"mu": 2.0, "syst1": 1.0, "syst2": -1.0, "syst3": 0.5, "mcstat": [1.2, 0.8]}},
            {"root": {"mu": 0.0, "syst1": 0.0, "syst2": 0.0, "syst3": 0.0, "mcstat_0": 1.0, "mcstat_1": 1.0},
             "record": {"mu": 0.0, "syst1": 0.0, "syst2": 0.0, "syst3": 0.0, "mcstat": [1.0, 1.0]}},
        ],
        "tolerance": {"atol": 1e-6, "rtol": 1e-6},
    },
}


def root_logL(hs3: Path, pdf_name: str, data_name: str, globals_: list[str]):
    """Return a `logL(point)` closure over the imported ROOT workspace."""
    ROOT.gROOT.SetBatch(True)
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
    ws = ROOT.RooWorkspace("ws")
    if not ROOT.RooJSONFactoryWSTool(ws).importJSON(str(hs3)):
        raise SystemExit(f"failed to import {hs3}")
    pdf = ws.pdf(pdf_name)
    data = ws.data(data_name)
    if globals_:
        glob = ROOT.RooArgSet(*[ws.var(f"nom_{p}") for p in globals_ if ws.var(f"nom_{p}")])
        nll = pdf.createNLL(data, ROOT.RooFit.GlobalObservables(glob))
    else:
        nll = pdf.createNLL(data)

    def logL(point: dict[str, float]) -> float:
        for name, value in point.items():
            ws.var(name).setVal(value)
        return -nll.getVal()

    return logL


def gen(model: str, cfg: dict) -> None:
    hs3 = HERE / model / f"{model}.hs3"
    logL = root_logL(hs3, cfg["pdf"], cfg["data"], cfg["globals"])
    ref = logL(cfg["points"][0]["root"])
    # `+ 0.0` normalises the reference point's -0.0 to 0.0 for a clean diff.
    expected = [-2.0 * (logL(p["root"]) - ref) + 0.0 for p in cfg["points"]]
    doc = {
        "schema_version": 1,
        "test_id": f"conv_{model}",
        "model": f"{model}.flatppl",
        "reference_backend": f"root {ROOT.gROOT.GetVersion()}",
        "checks": [
            {
                "id": "twice_delta_nll_points",
                "kind": "twice_delta_nll_points",
                "binding": cfg["binding"],
                "reference_point": cfg["points"][0]["record"],
                "points": [p["record"] for p in cfg["points"]],
                "expected": expected,
                "tolerance": cfg["tolerance"],
            }
        ],
    }
    out = HERE / model / "expected.json"
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"{model}: expected={expected}")


if __name__ == "__main__":
    for name, config in MODELS.items():
        gen(name, config)
