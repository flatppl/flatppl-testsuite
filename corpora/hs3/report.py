#!/usr/bin/env python3
"""Render the HS3 corpus scoring results as a self-contained HTML report.

Scores every numeric check in the manifest — the `fixtures/` 2DeltaNLL scans and
the `conversions/` point clouds — with the FlatPPL engine, compares against the
frozen ROOT vectors, and emits a single standalone HTML file: per check, the
2DeltaNLL trace (expected curve vs the engine's points) beside the numbers.

    pixi run report                      # write corpora/hs3/report.html
    pixi run report -o /tmp/r.html       # choose the output path
    pixi run report --open               # open it afterwards (macOS `open`)
"""
from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path

HS3_ROOT = Path(__file__).resolve().parent       # corpora/hs3
REPO = HS3_ROOT.parents[1]                        # repo root
sys.path.insert(0, str(REPO / "src"))

from flatppl_testsuite.suites.hs3_import import (  # noqa: E402
    score_scan, score_points, HS3_CORPUS, HS3_MANIFEST)


# ---------------------------------------------------------------------------
# Gather: score each numeric check, keep the full vectors for plotting.
# ---------------------------------------------------------------------------

def _within(got: float, exp: float, tol: dict) -> bool:
    return abs(got - exp) <= tol["atol"] + tol["rtol"] * abs(exp)


def gather() -> tuple[dict, list[dict]]:
    manifest = json.loads(HS3_MANIFEST.read_text())
    rows: list[dict] = []

    for fx in manifest.get("fixtures", []):
        fdir = HS3_CORPUS / fx["path"]
        hs3_path = fdir / "hs3.json"
        hs3_doc = json.loads(hs3_path.read_text())
        expected_doc = json.loads((fdir / "expected.json").read_text())
        for check in expected_doc["checks"]:
            if check["kind"] != "twice_delta_nll_scan":
                continue
            row = _row("fixtures", fx["test_id"], check,
                       axis=check["scan_parameter"], xs=check["scan_points"])
            try:
                row["got"] = score_scan(hs3_doc, hs3_path, check)
            except Exception as e:  # noqa: BLE001
                row["error"] = str(e)
            rows.append(row)

    for cv in manifest.get("conversions", []):
        cdir = HS3_CORPUS / cv["path"]
        expected_doc = json.loads((cdir / "expected.json").read_text())
        model = cdir / expected_doc["model"]
        for check in expected_doc["checks"]:
            if check["kind"] != "twice_delta_nll_points":
                continue
            row = _row("conversions", cv["test_id"], check,
                       axis=check["binding"], xs=list(range(len(check["expected"]))))
            try:
                row["got"] = score_points(model, check)
            except Exception as e:  # noqa: BLE001
                row["error"] = str(e)
            rows.append(row)

    return manifest, rows


def _row(corpus: str, test_id: str, check: dict, axis: str, xs: list) -> dict:
    return {
        "corpus": corpus,
        "test_id": test_id,
        "check_id": check["id"],
        "axis": axis,
        "xs": xs,
        "expected": check["expected"],
        "tol": check["tolerance"],
        "got": None,
        "error": None,
    }


# ---------------------------------------------------------------------------
# SVG: the 2DeltaNLL trace — expected polyline, engine points as markers.
# ---------------------------------------------------------------------------

W, H, PAD = 360, 168, 34


def _scale(vals, lo, hi, a, b):
    span = (hi - lo) or 1.0
    return [a + (v - lo) / span * (b - a) for v in vals]


def svg_trace(row: dict) -> str:
    xs = [float(x) for x in row["xs"]]
    exp = [float(v) for v in row["expected"]]
    got = [float(v) for v in row["got"]] if row["got"] else exp
    ys = exp + got + [0.0]
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    px = _scale(xs, xlo, xhi, PAD, W - PAD)
    pexp = _scale(exp, ylo, yhi, H - PAD, PAD)
    pgot = _scale(got, ylo, yhi, H - PAD, PAD)
    y0 = _scale([0.0], ylo, yhi, H - PAD, PAD)[0]

    order = sorted(range(len(xs)), key=lambda i: px[i])
    poly = " ".join(f"{px[i]:.1f},{pexp[i]:.1f}" for i in order)

    markers = []
    for i in range(len(xs)):
        ok = row["got"] is not None and _within(got[i], exp[i], row["tol"])
        c = "var(--accent)" if ok else "var(--fail)"
        markers.append(
            f'<circle cx="{px[i]:.1f}" cy="{pgot[i]:.1f}" r="3.4" '
            f'fill="{c}" stroke="var(--ground)" stroke-width="1.5"/>'
        )
    marks = "".join(markers)

    baseline = ""
    if ylo <= 0.0 <= yhi:
        baseline = (f'<line x1="{PAD}" y1="{y0:.1f}" x2="{W - PAD}" y2="{y0:.1f}" '
                    f'stroke="var(--line)" stroke-dasharray="3 4"/>')

    return f"""<svg viewBox="0 0 {W} {H}" class="trace" role="img"
   aria-label="2 delta NLL over {html.escape(row['axis'])}">
  <rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="6"
        fill="none" stroke="var(--line)"/>
  {baseline}
  <polyline points="{poly}" fill="none" stroke="var(--muted)"
            stroke-width="1.4" stroke-linejoin="round"/>
  {marks}
  <text x="{PAD}" y="{H - 12}" class="axn">{html.escape(row['axis'])} &#8594;</text>
  <text x="{PAD}" y="20" class="axn">2&#916;NLL</text>
</svg>"""


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

STYLE = """
:root{
  --ground:#0E141F; --panel:#16202E; --text:#CFD8E6; --muted:#6B7A92;
  --accent:#46E0B8; --fail:#FF6B6B; --line:#243245;
  --mono:ui-monospace,"SF Mono",SFMono-Regular,"Cascadia Code",Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--text);font-family:var(--mono);
  font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:48px 28px 80px}
.eyebrow{color:var(--accent);letter-spacing:.32em;text-transform:uppercase;
  font-size:11px;margin:0 0 14px}
h1{font-size:30px;font-weight:600;letter-spacing:-.01em;margin:0 0 8px}
.sub{color:var(--muted);margin:0 0 32px;max-width:60ch}
.readout{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 40px}
.chip{border:1px solid var(--line);background:var(--panel);border-radius:8px;
  padding:12px 16px;min-width:104px}
.chip .k{color:var(--muted);font-size:10.5px;letter-spacing:.16em;text-transform:uppercase}
.chip .v{font-size:22px;font-weight:600;margin-top:2px}
.chip.ok .v{color:var(--accent)} .chip.bad .v{color:var(--fail)}
.sec{margin:44px 0 16px;display:flex;align-items:baseline;gap:14px;
  border-bottom:1px solid var(--line);padding-bottom:10px}
.sec h2{font-size:13px;letter-spacing:.22em;text-transform:uppercase;margin:0;font-weight:600}
.sec .n{color:var(--muted);font-size:11.5px}
.card{border:1px solid var(--line);background:var(--panel);border-radius:12px;
  padding:18px 20px;margin:16px 0;display:grid;
  grid-template-columns:360px 1fr;gap:24px;align-items:center}
@media(max-width:760px){.card{grid-template-columns:1fr}}
.trace{width:100%;height:auto;background:#0B1019;border-radius:8px}
.axn{fill:var(--muted);font-size:10px;letter-spacing:.08em}
.head{display:flex;align-items:center;gap:12px;margin-bottom:12px}
.head .id{font-weight:600}
.head .meta{color:var(--muted);font-size:11.5px}
.tag{margin-left:auto;font-size:10.5px;letter-spacing:.14em;text-transform:uppercase;
  padding:4px 10px;border-radius:999px;font-weight:600}
.tag.ok{color:var(--ground);background:var(--accent)}
.tag.bad{color:var(--ground);background:var(--fail)}
table{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:5px 8px;white-space:nowrap}
th{color:var(--muted);font-weight:500;letter-spacing:.08em;text-transform:uppercase;
  font-size:10px;border-bottom:1px solid var(--line)}
td:first-child,th:first-child{text-align:left}
tr td{border-bottom:1px solid rgba(36,50,69,.5)}
tr:last-child td{border-bottom:none}
.bar{display:inline-block;height:7px;border-radius:4px;background:var(--accent);
  vertical-align:middle;min-width:2px}
.bar.bad{background:var(--fail)}
.dim{color:var(--muted)}
.delta{color:var(--accent)} .delta.bad{color:var(--fail)}
footer{color:var(--muted);font-size:11.5px;margin-top:48px;border-top:1px solid var(--line);
  padding-top:18px}
"""


def _fmt(v: float) -> str:
    return f"{v:.10g}"


def check_table(row: dict) -> str:
    exp, got, tol = row["expected"], row["got"], row["tol"]
    head = ("<tr><th>pt</th><th>expected</th><th>got</th><th>|&#916;|</th>"
            "<th>tol</th><th>headroom</th></tr>")
    body = []
    for i, (x, e) in enumerate(zip(row["xs"], exp)):
        g = got[i]
        d = abs(g - e)
        t = tol["atol"] + tol["rtol"] * abs(e)
        ok = d <= t
        frac = 0.0 if t == 0 else min(d / t, 1.0)
        w = max(2, round(frac * 70))
        bar = f'<span class="bar{"" if ok else " bad"}" style="width:{w}px"></span>'
        body.append(
            f'<tr><td class="dim">{_fmt(float(x))}</td>'
            f'<td>{_fmt(e)}</td><td>{_fmt(g)}</td>'
            f'<td class="delta{"" if ok else " bad"}">{d:.2e}</td>'
            f'<td class="dim">{t:.1e}</td><td>{bar}</td></tr>'
        )
    return f"<table>{head}{''.join(body)}</table>"


def card(row: dict) -> str:
    if row["error"]:
        body = f'<p class="delta bad">UNSCOREABLE — {html.escape(row["error"])}</p>'
        ok = False
        plot = ""
    else:
        ok = all(_within(g, e, row["tol"])
                 for g, e in zip(row["got"], row["expected"]))
        body = check_table(row)
        plot = svg_trace(row)
    tag = ('<span class="tag ok">pass</span>' if ok
           else '<span class="tag bad">fail</span>')
    npts = len(row["expected"])
    return f"""<div class="card">
  <div>{plot}</div>
  <div>
    <div class="head">
      <span class="id">{html.escape(row['test_id'])}</span>
      <span class="meta">{html.escape(row['check_id'])} &middot; {npts} pts &middot;
        atol {row['tol']['atol']:g} / rtol {row['tol']['rtol']:g}</span>
      {tag}
    </div>
    {body}
  </div>
</div>"""


def render_inner(manifest: dict, rows: list[dict]) -> str:
    passed = sum(
        1 for r in rows
        if not r["error"] and all(_within(g, e, r["tol"])
                                  for g, e in zip(r["got"], r["expected"]))
    )
    total = len(rows)
    failed = total - passed
    n_fix = len(manifest.get("fixtures", []))
    n_conv = len(manifest.get("conversions", []))
    backend = manifest.get("reference_backend", "ROOT/RooFit")
    all_ok = failed == 0

    def section(corpus: str, title: str) -> str:
        sub = [r for r in rows if r["corpus"] == corpus]
        if not sub:
            return ""
        cards = "".join(card(r) for r in sub)
        return (f'<div class="sec"><h2>{title}</h2>'
                f'<span class="n">{len(sub)} checks</span></div>{cards}')

    status_chip = (f'<div class="chip {"ok" if all_ok else "bad"}">'
                   f'<div class="k">status</div>'
                   f'<div class="v">{"NOMINAL" if all_ok else "FAIL"}</div></div>')

    return f"""<style>{STYLE}</style>
<div class="wrap">
  <p class="eyebrow">FlatPPL &times; HS3 &middot; conversion + scoring</p>
  <h1>2&#916;NLL conformance report</h1>
  <p class="sub">Every converted model, scored by the FlatPPL engine and
     compared point-by-point against the frozen RooFit / ROOT 2&#916;NLL vector.
     The trace is the likelihood scan; markers are the engine's values.</p>
  <div class="readout">
    {status_chip}
    <div class="chip"><div class="k">checks</div><div class="v">{total}</div></div>
    <div class="chip ok"><div class="k">pass</div><div class="v">{passed}</div></div>
    <div class="chip {"bad" if failed else ""}"><div class="k">fail</div>
      <div class="v">{failed}</div></div>
    <div class="chip"><div class="k">fixtures</div><div class="v">{n_fix}</div></div>
    <div class="chip"><div class="k">conversions</div><div class="v">{n_conv}</div></div>
  </div>
  {section("fixtures", "Fixtures — RooFit tutorials")}
  {section("conversions", "Conversions — HS3 paper appendix")}
  <footer>Oracle: {html.escape(str(backend))} &middot;
    engine: flatppl-js &middot; generated by corpora/hs3/report.py</footer>
</div>"""


def render_doc(manifest: dict, rows: list[dict]) -> str:
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>FlatPPL &times; HS3 — 2&#916;NLL report</title></head>'
            f'<body>{render_inner(manifest, rows)}</body></html>')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-o", "--out", default=str(HS3_ROOT / "report.html"),
                    help="output HTML path (default: corpora/hs3/report.html)")
    ap.add_argument("--open", action="store_true", help="open the report afterwards")
    args = ap.parse_args()

    manifest, rows = gather()
    out = Path(args.out)
    out.write_text(render_doc(manifest, rows))

    failed = [r for r in rows
              if r["error"] or not all(_within(g, e, r["tol"])
                                       for g, e in zip(r["got"], r["expected"]))]
    print(f"wrote {out}  ({len(rows) - len(failed)}/{len(rows)} checks pass)")
    if args.open:
        subprocess.run(["open", str(out)], check=False)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
