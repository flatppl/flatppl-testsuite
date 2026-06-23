#!/usr/bin/env python3
"""Evaluate gaussian.hs3.json with ROOT/RooFit.

Requires ROOT >= 6.30 with RooFit JSON support.
"""
import os
import sys
from pathlib import Path

import ROOT

HS3 = str(Path(__file__).resolve().with_name("gaussian.hs3.json"))

COL_MODEL, COL_LABEL, COL_VALUE = 10, 36, 16
SEP = '-' * (COL_MODEL + COL_LABEL + COL_VALUE + 10)


def row(model, label, value):
    return f"{model:<{COL_MODEL}}{label:<{COL_LABEL}}{value:>{COL_VALUE}.10f}   ROOT"


def main() -> None:
    ROOT.gROOT.SetBatch(True)
    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.ERROR)
    ws = ROOT.RooWorkspace("ws")
    if not ROOT.RooJSONFactoryWSTool(ws).importJSON(HS3):
        raise SystemExit(f"failed to import {HS3}")

    pdf = ws.pdf("gauss_x")
    data = ws.data("obs_gaussian_channel")
    nll = pdf.createNLL(data)

    def logpdf(mu):
        ws.var("mu").setVal(mu)
        return -nll.getVal()

    print(f"{'model':<{COL_MODEL}}{'test point':<{COL_LABEL}}{'log-density':>{COL_VALUE}}   result")
    print(SEP)
    for mu in (0.0, 0.5, 1.27):
        print(row('gaussian', f'obs @ mu={mu},sigma=1.0', logpdf(mu)))
    print(SEP)

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
