#!/usr/bin/env python3
"""Evaluate histfactory.hs3 with ROOT/RooFit.

Requires ROOT >= 6.30 with RooFit JSON support.
"""
import os
import sys
from pathlib import Path

import ROOT

HS3 = str(Path(__file__).resolve().with_name("histfactory.hs3"))

PARAMS = ["mu", "syst1", "syst2", "syst3", "mcstat_0", "mcstat_1"]

COL_MODEL, COL_LABEL, COL_VALUE = 10, 36, 16
SEP = '-' * (COL_MODEL + COL_LABEL + COL_VALUE + 10)


def row(model, label, value, tag='ROOT'):
    return f"{model:<{COL_MODEL}}{label:<{COL_LABEL}}{value:>{COL_VALUE}.10f}   {tag}"

def show(model, label, value):
    return f"{model:<{COL_MODEL}}{label:<{COL_LABEL}}{value:>{COL_VALUE}.10f}"


def main() -> None:
    ROOT.gROOT.SetBatch(True)
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
    ws = ROOT.RooWorkspace("ws")
    if not ROOT.RooJSONFactoryWSTool(ws).importJSON(HS3):
        raise SystemExit(f"failed to import {HS3}")

    pdf = ws.pdf("model_channel1")
    data = ws.data("observed_channel1")
    glob = ROOT.RooArgSet(*[ws.var(f"nom_{p}") for p in PARAMS if ws.var(f"nom_{p}")])
    nll = pdf.createNLL(data, ROOT.RooFit.GlobalObservables(glob))

    def logpdf(**point):
        for name, value in point.items():
            ws.var(name).setVal(value)
        return -nll.getVal()

    POINTS = [
        dict(mu=1.0, syst1=0.0,  syst2=0.0,  syst3=0.0,  mcstat_0=1.0, mcstat_1=1.0),
        dict(mu=1.5, syst1=0.5,  syst2=0.0,  syst3=0.0,  mcstat_0=1.1, mcstat_1=1.0),
        dict(mu=0.5, syst1=-0.3, syst2=0.2,  syst3=0.0,  mcstat_0=0.9, mcstat_1=1.1),
        dict(mu=2.0, syst1=1.0,  syst2=-1.0, syst3=0.5,  mcstat_0=1.2, mcstat_1=0.8),
        dict(mu=0.0, syst1=0.0,  syst2=0.0,  syst3=0.0,  mcstat_0=1.0, mcstat_1=1.0),
    ]

    vals = [logpdf(**pt) for pt in POINTS]
    base = vals[0]

    print(f"{'model':<{COL_MODEL}}{'test point':<{COL_LABEL}}{'log-density':>{COL_VALUE}}   result")
    print(SEP)
    for i, (pt, v) in enumerate(zip(POINTS, vals)):
        label = ', '.join(f'{k} = {vv}' for k, vv in pt.items())[:COL_LABEL - 1]
        print(show('histfact', label, v))
    for i in range(1, len(vals)):
        print(row('histfact', f'Δ(pt0→pt{i})', vals[i] - base))
    print(SEP)

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
