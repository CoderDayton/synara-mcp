"""HTML/SVG report rendering for the self-learning simulation.

Pure presentation: takes a list of ``Snapshot`` records, a header tuple,
a meta line, and a list of synthesis strings; returns a self-contained
HTML string (inline CSS + SVG, no external assets). Kept in a sibling
module so the simulation script stays focused on sim mechanics.
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass

_CW, _CH = 440, 240


@dataclass(frozen=True, slots=True)
class Snapshot:
    """One longitudinal sample. Order of fields matches the printed table."""

    day: int
    epis: int
    sem: int
    p_edges: int
    habits: int
    max_hits: int
    sr_rows: int
    sr_edges: int
    cons: int
    dreams: int
    ep_per_sch: float
    sr_dens: float
    edge_conc: float

    def as_row(self) -> tuple[object, ...]:
        return (
            self.day,
            self.epis,
            self.sem,
            self.p_edges,
            self.habits,
            self.max_hits,
            self.sr_rows,
            self.sr_edges,
            self.cons,
            self.dreams,
            self.ep_per_sch,
            self.sr_dens,
            self.edge_conc,
        )


_BIG = 100
_MID = 10


def _fmt_num(v: float) -> str:
    """Pretty-print numbers for chart axes and tooltips."""
    if v == int(v):
        return f"{int(v):,}"
    if abs(v) >= _BIG:
        return f"{v:,.0f}"
    if abs(v) >= _MID:
        return f"{v:.1f}"
    return f"{v:.2f}"


def _fmt_cell(v: object) -> str:
    if isinstance(v, float):
        return f"{v:,.2f}" if v != int(v) else f"{int(v):,}"
    if isinstance(v, int):
        return f"{v:,}"
    return _html.escape(str(v))


def _svg_chart(
    title: str,
    days: list[int],
    ys: list[float],
    *,
    color: str = "#2f9e44",
) -> str:
    """One self-contained SVG line chart with gridlines + 3-tick axes."""
    pad_l, pad_r, pad_t, pad_b = 56, 18, 38, 34
    iw, ih = _CW - pad_l - pad_r, _CH - pad_t - pad_b
    xmin, xmax = days[0], days[-1]
    ymax_raw = max(ys) if ys else 0.0
    ymax = ymax_raw if ymax_raw > 0 else 1.0

    def sx(d: float) -> float:
        return pad_l + (d - xmin) / ((xmax - xmin) or 1) * iw

    def sy(v: float) -> float:
        return pad_t + ih - (v / ymax) * ih

    pts_xy = list(zip(days, ys, strict=True))
    pts = " ".join(f"{sx(d):.1f},{sy(v):.1f}" for d, v in pts_xy)
    area = f"{sx(xmin):.1f},{sy(0):.1f} {pts} {sx(xmax):.1f},{sy(0):.1f}"

    # Y gridlines + labels at 0, ymax/2, ymax
    yticks = [(0.0, "0"), (ymax / 2, _fmt_num(ymax / 2)), (ymax, _fmt_num(ymax))]
    grid = "".join(
        f'<line x1="{pad_l}" y1="{sy(y):.1f}" x2="{pad_l + iw}" y2="{sy(y):.1f}" class="gl"/>'
        for y, _ in yticks
    )
    ylab = "".join(
        f'<text x="{pad_l - 8}" y="{sy(y) + 3.5:.1f}" class="tk" '
        f'text-anchor="end">{_html.escape(lbl)}</text>'
        for y, lbl in yticks
    )

    # X labels at first, mid, last
    xmid = (xmin + xmax) / 2
    xticks = [(xmin, xmin), (xmid, xmid), (xmax, xmax)]
    xlab = "".join(
        f'<text x="{sx(x):.1f}" y="{_CH - 12}" class="tk" text-anchor="middle">d{round(d)}</text>'
        for x, d in xticks
    )

    dots = "".join(
        f'<circle cx="{sx(d):.1f}" cy="{sy(v):.1f}" r="3" '
        f'fill="{color}" stroke="#fff" stroke-width="1.2">'
        f"<title>day {d}: {_fmt_num(v)}</title></circle>"
        for d, v in pts_xy
    )

    return (
        f'<svg viewBox="0 0 {_CW} {_CH}" class="chart" '
        'xmlns="http://www.w3.org/2000/svg">'
        f'<text x="{pad_l}" y="22" class="ct">{_html.escape(title)}</text>'
        f"{grid}{ylab}{xlab}"
        f'<polygon points="{area}" fill="{color}" fill-opacity="0.10"/>'
        f'<polyline points="{pts}" fill="none" stroke="{color}" '
        'stroke-width="2.2" stroke-linecap="round" '
        'stroke-linejoin="round"/>'
        f"{dots}</svg>"
    )


_CSS = """
:root{
  --bg:#f6f7f9;--surface:#fff;--text:#1a1d23;--dim:#6c757d;
  --border:#e7eaef;--border-strong:#d6dbe2;--accent:#4c6ef5;
  --shadow:0 1px 2px rgba(0,0,0,.04),0 1px 6px rgba(0,0,0,.03);
  --pad:10px;
}
*{box-sizing:border-box}
html,body{height:100%;margin:0;overflow:hidden}
body{font:13px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;
  background:var(--bg);color:var(--text);
  font-feature-settings:"tnum" 1,"cv11" 1;
  display:grid;grid-template-rows:auto 1fr;height:100dvh}
header{padding:8px 14px;border-bottom:1px solid var(--border);
  background:var(--surface);display:flex;align-items:baseline;
  gap:14px;flex-wrap:wrap}
h1{font-size:17px;margin:0;letter-spacing:-0.012em;font-weight:700}
h1 .pill{display:inline-block;font-size:10.5px;font-weight:600;
  letter-spacing:.04em;padding:2px 7px;border-radius:999px;
  background:#e8edff;color:#3247a8;margin-left:8px;vertical-align:middle}
.caption{color:var(--dim);margin:0;font-size:12px;flex:1;min-width:0;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
main{display:grid;gap:var(--pad);padding:var(--pad);min-height:0;
  grid-template-columns:minmax(0,1fr) clamp(360px,26vw,520px);
  grid-template-rows:auto minmax(0,1fr);
  grid-template-areas:"cards table" "charts table"}
@media (max-width:1280px){
  main{grid-template-columns:1fr;
    grid-template-rows:auto minmax(0,1.4fr) minmax(0,1fr);
    grid-template-areas:"cards" "charts" "table"}
}
@media (min-width:2400px){
  main{grid-template-columns:minmax(0,1fr) clamp(440px,22vw,640px)}
}
.cards{grid-area:cards;display:grid;gap:6px;min-width:0;
  grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
  grid-auto-rows:1fr;align-content:start}
.card{background:var(--surface);border:1px solid var(--border);
  border-left:3px solid var(--accent);border-radius:7px;
  padding:5px 9px;box-shadow:var(--shadow);min-width:0;overflow:hidden}
.card-label{font-size:9px;text-transform:uppercase;letter-spacing:.06em;
  color:var(--dim);font-weight:700;margin-bottom:1px}
.card-body{font-size:11px;line-height:1.3;color:var(--text);
  display:-webkit-box;-webkit-box-orient:vertical;
  -webkit-line-clamp:2;overflow:hidden}
.card:hover{z-index:10;position:relative}
.card:hover .card-body{-webkit-line-clamp:unset;overflow:visible}
.charts{grid-area:charts;display:grid;gap:8px;min-height:0;min-width:0;
  grid-template-columns:repeat(auto-fit,minmax(clamp(240px,12.5%,480px),1fr));
  grid-auto-rows:auto;align-content:center;justify-content:stretch;
  overflow:hidden}
.chart{background:var(--surface);border:1px solid var(--border);
  border-radius:8px;box-shadow:var(--shadow);display:block;
  width:100%;height:auto;aspect-ratio:11/6;max-height:100%;
  min-width:0}
.ct{font-size:13px;font-weight:600;fill:var(--text)}
.gl{stroke:var(--border);stroke-width:1;stroke-dasharray:2,3}
.tk{font-size:10.5px;fill:var(--dim)}
.table-wrap{grid-area:table;background:var(--surface);
  border:1px solid var(--border);border-radius:8px;
  box-shadow:var(--shadow);overflow:auto;min-height:0;min-width:0}
table{width:100%;border-collapse:collapse;font-size:11.5px;
  font-variant-numeric:tabular-nums}
thead th{background:#f1f3f7;padding:6px 8px;text-align:right;
  font-weight:600;color:var(--dim);
  border-bottom:1px solid var(--border-strong);position:sticky;top:0;
  font-size:10px;text-transform:uppercase;letter-spacing:.04em;
  z-index:1}
thead th:first-child{text-align:left;position:sticky;left:0;
  background:#f1f3f7;z-index:2}
tbody td{padding:4px 8px;text-align:right;
  border-top:1px solid var(--border);font-variant-numeric:tabular-nums;
  white-space:nowrap}
tbody td:first-child{text-align:left;color:var(--dim);font-weight:600;
  position:sticky;left:0;background:inherit}
tbody tr{background:var(--surface)}
tbody tr:nth-child(even){background:#fafbfc}
tbody tr:hover{background:#eef2ff}
"""


def _synthesis_cards(synthesis: list[str]) -> str:
    out: list[str] = []
    for s in synthesis:
        if ":" in s:
            label, _, body = s.partition(":")
            out.append(
                '<div class="card">'
                f'<div class="card-label">{_html.escape(label.strip())}</div>'
                f'<div class="card-body">{_html.escape(body.strip())}</div>'
                "</div>"
            )
        else:
            out.append(f'<div class="card"><div class="card-body">{_html.escape(s)}</div></div>')
    return "".join(out)


def render_html(
    rows: list[Snapshot],
    headers: tuple[str, ...],
    meta: str,
    synthesis: list[str],
) -> str:
    """Assemble a dependency-free HTML report (inline SVG + table)."""
    days = [r.day for r in rows]
    specs: list[tuple[str, str, str]] = [
        ("Episodic store", "epis", "#1971c2"),
        ("Semantic schemas", "sem", "#9c36b5"),
        ("Plasticity edges", "p_edges", "#2f9e44"),
        ("Habit edges", "habits", "#e8590c"),
        ("Max edge hits", "max_hits", "#c2255c"),
        ("SR transition rows", "sr_rows", "#0c8599"),
        ("SR graph edges", "sr_edges", "#5c940d"),
        ("Dream replays (cum.)", "dreams", "#862e9c"),
        ("Episodes / schema (compression)", "ep_per_sch", "#d6336c"),
        ("SR fan-out / node (densification)", "sr_dens", "#1098ad"),
        ("Hits / edge (concentration)", "edge_conc", "#f08c00"),
    ]
    charts = "".join(
        _svg_chart(t, days, [float(getattr(r, attr)) for r in rows], color=col)
        for t, attr, col in specs
    )
    head = "".join(f"<th>{_html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_fmt_cell(v)}</td>" for v in r.as_row()) + "</tr>" for r in rows
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>synara self-learning sim</title>"
        f"<style>{_CSS}</style></head><body>"
        "<header>"
        '<h1>synara<span class="pill">self-learning sim</span></h1>'
        f'<p class="caption" title="{_html.escape(meta)}">{_html.escape(meta)}</p>'
        "</header>"
        "<main>"
        f'<section class="cards" aria-label="Highlights">'
        f"{_synthesis_cards(synthesis)}</section>"
        f'<section class="charts" aria-label="Trends">{charts}</section>'
        '<section class="table-wrap" aria-label="Snapshots">'
        f"<table><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table></section>"
        "</main></body></html>"
    )
