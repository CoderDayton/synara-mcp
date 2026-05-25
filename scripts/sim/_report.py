"""HTML/SVG report rendering for the self-learning simulation.

Pure presentation: takes a list of ``Snapshot`` records, a header tuple,
a meta line, and a list of synthesis strings; returns a self-contained
HTML string (inline CSS + SVG, no external assets). Kept in a sibling
module so the simulation script stays focused on sim mechanics.
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass

_CW, _CH = 460, 260


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
    color: str = "#4f46e5",
    grad_id: str = "g0",
) -> str:
    """One self-contained SVG line chart with gradient under-fill, 3-tick
    axes, and an inline last-value annotation."""
    pad_l, pad_r, pad_t, pad_b = 46, 56, 38, 30
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

    yticks = [(0.0, "0"), (ymax / 2, _fmt_num(ymax / 2)), (ymax, _fmt_num(ymax))]
    grid = "".join(
        f'<line x1="{pad_l}" y1="{sy(y):.1f}" x2="{pad_l + iw}" y2="{sy(y):.1f}" class="gl"/>'
        for y, _ in yticks
    )
    ylab = "".join(
        f'<text x="{pad_l - 6}" y="{sy(y) + 3.5:.1f}" class="tk" '
        f'text-anchor="end">{_html.escape(lbl)}</text>'
        for y, lbl in yticks
    )

    xmid = (xmin + xmax) / 2
    xticks = [(xmin, xmin), (xmid, xmid), (xmax, xmax)]
    xlab = "".join(
        f'<text x="{sx(x):.1f}" y="{_CH - 10}" class="tk" text-anchor="middle">d{round(d)}</text>'
        for x, d in xticks
    )

    dots = "".join(
        f'<circle cx="{sx(d):.1f}" cy="{sy(v):.1f}" r="2.6" '
        f'fill="{color}" stroke="#fff" stroke-width="1.2">'
        f"<title>day {d}: {_fmt_num(v)}</title></circle>"
        for d, v in pts_xy
    )

    last_d, last_v = pts_xy[-1]
    last_label = _fmt_num(last_v)
    lx = sx(last_d) + 6
    ly = sy(last_v) + 3.5

    return (
        f'<svg viewBox="0 0 {_CW} {_CH}" class="chart" '
        'preserveAspectRatio="xMidYMid meet" '
        'xmlns="http://www.w3.org/2000/svg">'
        "<defs>"
        f'<linearGradient id="{grad_id}" x1="0" x2="0" y1="0" y2="1">'
        f'<stop offset="0%" stop-color="{color}" stop-opacity="0.28"/>'
        f'<stop offset="100%" stop-color="{color}" stop-opacity="0"/>'
        "</linearGradient></defs>"
        f'<text x="{pad_l}" y="20" class="ct">{_html.escape(title)}</text>'
        f"{grid}{ylab}{xlab}"
        f'<polygon points="{area}" fill="url(#{grad_id})"/>'
        f'<polyline points="{pts}" fill="none" stroke="{color}" '
        'stroke-width="2.25" stroke-linecap="round" '
        'stroke-linejoin="round"/>'
        f"{dots}"
        f'<text x="{lx:.1f}" y="{ly:.1f}" class="lv" fill="{color}">'
        f"{_html.escape(last_label)}</text>"
        "</svg>"
    )


_CSS = """
:root{
  --bg:#f5f6f8;--surface:#ffffff;--panel:#fbfbfd;
  --ink:#0b1220;--ink-2:#1f2937;--muted:#64748b;--muted-2:#94a3b8;
  --line:#e5e7eb;--line-strong:#d1d5db;
  --accent:#4f46e5;--accent-tint:#eef2ff;--accent-ink:#3730a3;
  --r:10px;--r-sm:8px;--pad:14px;
  --shadow-sm:0 1px 0 rgba(15,23,42,.04),0 1px 3px rgba(15,23,42,.04);
  --shadow:0 1px 0 rgba(15,23,42,.04),0 4px 14px rgba(15,23,42,.06);
}
*{box-sizing:border-box}
html,body{height:100%;margin:0;overflow:hidden}
body{
  font:13px/1.45 "Inter","SF Pro Text",-apple-system,system-ui,
    "Segoe UI",Roboto,sans-serif;
  background:
    radial-gradient(1200px 600px at 8% -10%,#eef2ff 0%,transparent 55%),
    radial-gradient(900px 500px at 105% 110%,#ecfeff 0%,transparent 55%),
    var(--bg);
  color:var(--ink);
  font-feature-settings:"tnum" 1,"ss01" 1,"cv11" 1;
  -webkit-font-smoothing:antialiased;
  letter-spacing:-0.005em;
  display:grid;grid-template-rows:auto 1fr;height:100dvh;
}
header{
  padding:11px 18px;
  background:linear-gradient(180deg,#ffffff 0%,#fafbff 100%);
  border-bottom:1px solid var(--line);
  display:flex;align-items:center;gap:14px;flex-wrap:wrap;
  box-shadow:0 1px 0 rgba(15,23,42,.02);
}
.brand{display:flex;align-items:center;gap:10px;min-width:0}
.brand-mark{
  width:26px;height:26px;border-radius:8px;
  background:conic-gradient(from 220deg,#4f46e5,#0ea5e9,#10b981,#4f46e5);
  box-shadow:inset 0 0 0 1px rgba(255,255,255,.5),
    0 1px 2px rgba(79,70,229,.3);
  position:relative;flex:none;
}
.brand-mark::after{
  content:"";position:absolute;inset:5px;border-radius:5px;
  background:radial-gradient(circle at 30% 30%,#fff,rgba(255,255,255,.7));
  mix-blend-mode:overlay;
}
h1{
  font-size:16px;margin:0;font-weight:700;letter-spacing:-0.018em;
  color:var(--ink);white-space:nowrap;
}
h1 .sub{
  display:inline-block;font-size:11.5px;font-weight:500;
  color:var(--muted);margin-left:8px;letter-spacing:0;
}
.meta{
  flex:1;min-width:0;color:var(--muted);font-size:12px;
  text-align:right;white-space:nowrap;overflow:hidden;
  text-overflow:ellipsis;font-weight:500;
  font-feature-settings:"tnum" 1;
}
.meta b{color:var(--ink-2);font-weight:600}
.meta .dot{display:inline-block;margin:0 8px;color:var(--muted-2)}
main{
  display:grid;gap:var(--pad);padding:var(--pad);min-height:0;
  grid-template-columns:1fr;
  grid-template-rows:auto minmax(0,1.5fr) minmax(0,1fr);
  grid-template-areas:"cards" "charts" "table";
}
@media (min-width:1720px){
  main{
    grid-template-columns:minmax(0,1fr) clamp(480px,28vw,600px);
    grid-template-rows:auto minmax(0,1fr);
    grid-template-areas:"cards table" "charts table";
  }
}
@media (min-width:2600px){
  main{grid-template-columns:minmax(0,1fr) clamp(520px,24vw,720px)}
}
.cards{
  grid-area:cards;display:grid;gap:8px;min-width:0;
  grid-template-columns:repeat(var(--n,7),minmax(0,1fr));
  grid-auto-rows:auto;align-items:stretch;
}
@media (max-width:1100px){
  .cards{grid-template-columns:repeat(auto-fit,minmax(min(180px,100%),1fr))}
}
.card{
  position:relative;background:var(--surface);
  border:1px solid var(--line);border-radius:var(--r);
  padding:10px 14px 10px 16px;box-shadow:var(--shadow-sm);
  min-width:0;overflow:hidden;
  transition:transform .15s ease,box-shadow .15s ease;
}
.card::before{
  content:"";position:absolute;left:0;top:10px;bottom:10px;width:3px;
  border-radius:0 3px 3px 0;background:var(--c,var(--accent));
}
.card:hover{transform:translateY(-1px);box-shadow:var(--shadow)}
.card-label{
  font-size:9.5px;text-transform:uppercase;letter-spacing:.08em;
  color:var(--muted);font-weight:700;margin:0 0 3px;
}
.card-body{
  font-size:12px;line-height:1.45;color:var(--ink-2);
  word-wrap:break-word;overflow-wrap:anywhere;
}
.charts{
  grid-area:charts;display:grid;gap:8px;min-height:0;min-width:0;
  grid-template-columns:repeat(auto-fit,minmax(clamp(200px,14vh,420px),1fr));
  grid-auto-rows:auto;align-content:center;justify-content:stretch;
  overflow:hidden;
}
.chart{
  background:var(--surface);border:1px solid var(--line);
  border-radius:var(--r);box-shadow:var(--shadow-sm);display:block;
  width:100%;height:auto;aspect-ratio:11/6.2;
  max-height:100%;min-width:0;
}
.ct{font-size:13.5px;font-weight:600;fill:var(--ink);
  letter-spacing:-0.005em}
.gl{stroke:var(--line);stroke-width:1;stroke-dasharray:2,3}
.tk{font-size:10.5px;fill:var(--muted);font-weight:500;
  font-feature-settings:"tnum" 1}
.lv{font-size:11px;font-weight:700;
  font-feature-settings:"tnum" 1;dominant-baseline:middle}
.table-wrap{
  grid-area:table;background:var(--surface);
  border:1px solid var(--line);border-radius:var(--r);
  box-shadow:var(--shadow-sm);overflow:auto;min-height:0;min-width:0;
  display:flex;flex-direction:column;
}
.table-head{
  padding:9px 14px;border-bottom:1px solid var(--line);
  display:flex;align-items:baseline;gap:10px;
  font-size:11px;color:var(--muted);font-weight:600;
  text-transform:uppercase;letter-spacing:.08em;flex:none;
  background:linear-gradient(180deg,#fcfcfe,#f8f9fc);
}
.table-head .count{
  color:var(--ink-2);background:var(--panel);
  border:1px solid var(--line);border-radius:999px;
  padding:1px 8px;font-size:10px;
}
.table-scroll{flex:1;overflow:auto;min-height:0}
table{
  width:100%;border-collapse:collapse;
  font-size:11px;font-variant-numeric:tabular-nums;
  table-layout:auto;
}
thead th{
  background:#fbfbfd;padding:6px 5px;text-align:right;
  font-weight:600;color:var(--muted);
  border-bottom:1px solid var(--line-strong);
  position:sticky;top:0;font-size:9.5px;
  text-transform:uppercase;letter-spacing:.04em;z-index:1;
}
thead th:first-child{
  text-align:left;padding-left:12px;position:sticky;left:0;
  background:#fbfbfd;z-index:2;
}
tbody td{
  padding:4px 5px;text-align:right;
  border-top:1px solid var(--line);
  font-variant-numeric:tabular-nums;color:var(--ink-2);
  white-space:nowrap;
}
tbody td:first-child{padding-left:12px}
tbody td:first-child{
  text-align:left;color:var(--accent-ink);font-weight:700;
  position:sticky;left:0;background:inherit;
}
tbody tr{background:var(--surface)}
tbody tr:nth-child(even){background:#fbfbfd}
tbody tr:hover{background:var(--accent-tint)}
tbody tr:last-child td{font-weight:600;color:var(--ink)}
"""


_CARD_ACCENTS = (
    "#4f46e5",
    "#0891b2",
    "#10b981",
    "#f59e0b",
    "#e11d48",
    "#7c3aed",
    "#0ea5e9",
)


def _synthesis_cards(synthesis: list[str]) -> str:
    out: list[str] = []
    for i, s in enumerate(synthesis):
        color = _CARD_ACCENTS[i % len(_CARD_ACCENTS)]
        style = f'style="--c:{color}"'
        if ":" in s:
            label, _, body = s.partition(":")
            out.append(
                f'<div class="card" {style}>'
                f'<div class="card-label">{_html.escape(label.strip())}</div>'
                f'<div class="card-body">{_html.escape(body.strip())}</div>'
                "</div>"
            )
        else:
            out.append(
                f'<div class="card" {style}><div class="card-body">{_html.escape(s)}</div></div>'
            )
    return "".join(out)


def _meta_line(meta: str) -> str:
    """Render the meta line as a single prose caption.

    Segments split on '; ' become bullet-separated, and bare ``k=v``
    tokens get their key highlighted. Result truncates with ellipsis
    at narrow widths; the full text is available via ``title=``.
    """
    parts = [p.strip() for p in meta.split(";") if p.strip()]
    rendered: list[str] = []
    for part in parts:
        toks: list[str] = []
        for word in part.split():
            had_comma = word.endswith(",")
            bare = word.rstrip(",")
            if "=" in bare:
                k, _, v = bare.partition("=")
                tok = f"<b>{_html.escape(k)}</b>={_html.escape(v)}"
            else:
                tok = _html.escape(bare)
            if had_comma:
                tok = tok + ","
            toks.append(tok)
        rendered.append(" ".join(toks))
    sep = '<span class="dot">·</span>'
    return sep.join(rendered)


def render_html(
    rows: list[Snapshot],
    headers: tuple[str, ...],
    meta: str,
    synthesis: list[str],
) -> str:
    """Assemble a dependency-free HTML report (inline SVG + table)."""
    days = [r.day for r in rows]
    specs: list[tuple[str, str, str]] = [
        ("Episodic store", "epis", "#4f46e5"),
        ("Semantic schemas", "sem", "#7c3aed"),
        ("Plasticity edges", "p_edges", "#059669"),
        ("Habit edges", "habits", "#ea580c"),
        ("Max edge hits", "max_hits", "#e11d48"),
        ("SR transition rows", "sr_rows", "#0891b2"),
        ("SR graph edges", "sr_edges", "#65a30d"),
        ("Dream replays (cum.)", "dreams", "#9333ea"),
        ("Episodes / schema", "ep_per_sch", "#db2777"),
        ("SR fan-out / node", "sr_dens", "#0284c7"),
        ("Hits / edge", "edge_conc", "#d97706"),
    ]
    charts = "".join(
        _svg_chart(
            t,
            days,
            [float(getattr(r, attr)) for r in rows],
            color=col,
            grad_id=f"grad{i}",
        )
        for i, (t, attr, col) in enumerate(specs)
    )
    head = "".join(f"<th>{_html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{_fmt_cell(v)}</td>" for v in r.as_row()) + "</tr>" for r in rows
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>synara · self-learning sim</title>"
        f"<style>{_CSS}</style></head><body>"
        "<header>"
        '<div class="brand"><div class="brand-mark" aria-hidden="true"></div>'
        '<h1>synara<span class="sub">self-learning simulation</span></h1>'
        "</div>"
        f'<div class="meta" title="{_html.escape(meta)}">'
        f"{_meta_line(meta)}</div>"
        "</header>"
        "<main>"
        f'<section class="cards" aria-label="Highlights" '
        f'style="--n:{len(synthesis)}">'
        f"{_synthesis_cards(synthesis)}</section>"
        f'<section class="charts" aria-label="Trends">{charts}</section>'
        '<section class="table-wrap" aria-label="Snapshots">'
        f'<div class="table-head">Snapshots'
        f'<span class="count">{len(rows)} samples · {len(headers)} fields</span>'
        "</div>"
        '<div class="table-scroll">'
        f"<table><thead><tr>{head}</tr></thead>"
        f"<tbody>{body}</tbody></table></div>"
        "</section>"
        "</main></body></html>"
    )
