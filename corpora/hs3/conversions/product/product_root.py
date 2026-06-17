#!/usr/bin/env python3
"""Evaluate product.hs3 with ROOT/RooFit.

Requires ROOT >= 6.30 with RooFit JSON support.
"""
import os
import sys
from pathlib import Path

import ROOT

HS3 = str(Path(__file__).resolve().with_name("product.hs3"))

DEFAULT = dict(mu1=0.0, sigma1=1.0, mu2=1.0, sigma2=2.0)
PERT    = dict(mu1=0.5, sigma1=1.0, mu2=1.0, sigma2=2.0)

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

    pdf = ws.pdf("prod")
    data = ws.data("toy")
    nll = pdf.createNLL(data)

    def logpdf(**point):
        for name, value in point.items():
            ws.var(name).setVal(value)
        return -nll.getVal()

    v0 = logpdf(**DEFAULT)
    v1 = logpdf(**PERT)

    print(f"{'model':<{COL_MODEL}}{'test point':<{COL_LABEL}}{'log-density':>{COL_VALUE}}   result")
    print(SEP)
    print(row('product', 'likelihood @ default',  v0))
    print(row('product', 'likelihood @ mu1=0.5',  v1))
    print(row('product', 'Δ(default→mu1=0.5)',    v1 - v0))
    print(SEP)

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
