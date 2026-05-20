"""HTML/SVG report rendering for the self-learning simulation.

Pure presentation: takes a list of ``Snapshot`` records, a header tuple,
a meta line, and a list of synthesis strings; returns a self-contained
HTML string (inline CSS + SVG, no external assets). Kept in a sibling
module so the simulation script stays focused on sim mechanics.
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass

_CW, _CH = 380, 210


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


def _svg_chart(
    title: str,
    days: list[int],
    ys: list[float],
    *,
    color: str = "#2f9e44",
) -> str:
    """One self-contained SVG line chart, autoscaled from 0 to max(ys)."""
    pad_l, pad_r, pad_t, pad_b = 46, 14, 30, 26
    iw, ih = _CW - pad_l - pad_r, _CH - pad_t - pad_b
    xmin, xmax = days[0], days[-1]
    ymax = max(ys) if ys else 1.0
    ymax = ymax if ymax > 0 else 1.0

    def sx(d: float) -> float:
        return pad_l + (d - xmin) / ((xmax - xmin) or 1) * iw

    def sy(v: float) -> float:
        return pad_t + ih - (v / ymax) * ih

    pts = " ".join(f"{sx(d):.1f},{sy(v):.1f}" for d, v in zip(days, ys, strict=True))
    dots = "".join(
        f'<circle cx="{sx(d):.1f}" cy="{sy(v):.1f}" r="2.6" '
        f'fill="{color}"><title>day {d}: {v:g}</title></circle>'
        for d, v in zip(days, ys, strict=True)
    )
    ax = pad_t + ih
    return (
        f'<svg viewBox="0 0 {_CW} {_CH}" class="chart" '
        'xmlns="http://www.w3.org/2000/svg">'
        f'<text x="{pad_l}" y="18" class="ct">{_html.escape(title)}</text>'
        f'<line x1="{pad_l}" y1="{ax}" x2="{pad_l + iw}" y2="{ax}" '
        'class="axis"/>'
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{ax}" '
        'class="axis"/>'
        f'<text x="{pad_l - 6}" y="{pad_t + 4}" class="tk" '
        f'text-anchor="end">{ymax:g}</text>'
        f'<text x="{pad_l - 6}" y="{ax}" class="tk" text-anchor="end">0</text>'
        f'<text x="{pad_l}" y="{_CH - 8}" class="tk">d{xmin}</text>'
        f'<text x="{pad_l + iw}" y="{_CH - 8}" class="tk" '
        f'text-anchor="end">d{xmax}</text>'
        f'<polyline points="{pts}" fill="none" stroke="{color}" '
        'stroke-width="2"/>'
        f"{dots}</svg>"
    )


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
    syn_li = "".join(f"<li>{_html.escape(s)}</li>" for s in synthesis)
    head = "".join(f"<th>{_html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in r.as_row()) + "</tr>" for r in rows)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        "<title>synara self-learning sim</title><style>"
        "body{font:14px/1.5 system-ui,sans-serif;margin:24px;"
        "background:#fafafa;color:#212529}"
        "h1{font-size:20px;margin:0 0 4px}"
        ".meta{color:#666;margin:0 0 16px}"
        "ul{margin:0 0 20px;padding-left:18px}li{margin:2px 0}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,"
        "minmax(360px,1fr));gap:14px}"
        ".chart{background:#fff;border:1px solid #e0e0e0;border-radius:8px}"
        ".ct{font-size:13px;font-weight:600;fill:#212529}"
        ".axis{stroke:#adb5bd;stroke-width:1}"
        ".tk{font-size:10px;fill:#868e96}"
        "table{border-collapse:collapse;margin-top:22px;font-size:12px}"
        "th,td{border:1px solid #dee2e6;padding:3px 8px;text-align:right}"
        "th{background:#f1f3f5}"
        "</style></head><body>"
        "<h1>synara &mdash; runtime self-learning simulation</h1>"
        f'<p class="meta">{_html.escape(meta)}</p>'
        f"<ul>{syn_li}</ul>"
        f'<div class="grid">{charts}</div>'
        f"<table><thead><tr>{head}</tr></thead><tbody>{body}"
        "</tbody></table></body></html>"
    )
