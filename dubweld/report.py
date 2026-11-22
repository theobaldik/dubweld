"""Self-contained HTML report: offset plot, residuals, segments, cuts and verification."""

from __future__ import annotations

import html
from collections.abc import Callable
from pathlib import Path

import numpy as np

from dubweld.plan import Plan, mmss
from dubweld.verify import DIALOGUE, NEAR_ZERO, VerifyReport

W, H = 960, 300
ML, MR, MT, MB = 64, 16, 40, 36
# Score bins -> sequential blue steps (light, dark); the lowest bin is the "junk" gray
SCORE_BINS = [
    (0.0, "s0", "below 0.3"),
    (0.3, "s1", "0.3-0.4"),
    (0.4, "s2", "0.4-0.5"),
    (0.5, "s3", "0.5-0.65"),
    (0.65, "s4", "0.65+"),
]

CSS = """
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink: #0b0b0b; --ink-2: #52514e; --muted: #898781;
  --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --gap: rgba(137,135,129,0.14);
  --s0: #c3c2b7; --s1: #b7d3f6; --s2: #6da7ec; --s3: #2a78d6; --s4: #104281;
  --level: #eb6834; --cut: #52514e; --good: #0ca30c; --warn: #fab219; --bad: #d03b3b; --band: rgba(12,163,12,0.10);
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --gap: rgba(137,135,129,0.18);
    --s0: #383835; --s1: #184f95; --s2: #256abf; --s3: #5598e7; --s4: #b7d3f6;
    --level: #d95926; --cut: #c3c2b7; --band: rgba(12,163,12,0.16);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink: #f0efec; --ink-2: #c3c2b7; --muted: #898781;
  --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --gap: rgba(137,135,129,0.18);
  --s0: #383835; --s1: #184f95; --s2: #256abf; --s3: #5598e7; --s4: #b7d3f6;
  --level: #d95926; --cut: #c3c2b7; --band: rgba(12,163,12,0.16);
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--ink);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1000px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 22px; margin: 0 0 4px; font-weight: 600; }
h2 { font-size: 16px; margin: 32px 0 8px; font-weight: 600; }
p.sub { color: var(--ink-2); margin: 0 0 16px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 12px; }
svg { width: 100%; height: auto; display: block; }
svg text { fill: var(--muted); font: 11px system-ui, -apple-system, "Segoe UI", sans-serif;
  font-variant-numeric: tabular-nums; }
svg .grid { stroke: var(--grid); stroke-width: 1; }
svg .axis { stroke: var(--axis); stroke-width: 1; }
svg .level { stroke: var(--level); stroke-width: 2; stroke-linecap: round; }
svg .cut { stroke: var(--cut); stroke-width: 1; opacity: 0.6; }
svg .cutlabel { fill: var(--ink-2); }
svg .gap { fill: var(--gap); }
svg .band { fill: var(--band); }
svg circle { stroke: var(--surface); stroke-width: 1.5; }
svg circle:hover { stroke: var(--ink); }
.s0 { fill: var(--s0); } .s1 { fill: var(--s1); } .s2 { fill: var(--s2); } .s3 { fill: var(--s3); }
.s4 { fill: var(--s4); }
.legend .s0 { background: var(--s0); } .legend .s1 { background: var(--s1); } .legend .s2 { background: var(--s2); }
.legend .s3 { background: var(--s3); } .legend .s4 { background: var(--s4); }
.legend { display: flex; flex-wrap: wrap; gap: 4px 16px; color: var(--ink-2); font-size: 12px; margin: 8px 4px 0; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
.line { width: 16px; height: 2px; display: inline-block; background: var(--level); }
.vline { width: 1px; height: 12px; display: inline-block; background: var(--cut); }
.swatch { width: 14px; height: 10px; display: inline-block; background: var(--gap); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--grid); vertical-align: top; }
th { color: var(--ink-2); font-weight: 600; }
.pill { display: inline-flex; align-items: center; gap: 6px; font-weight: 600; }
.pill::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
.pass, .high { color: var(--good); } .warn, .medium { color: var(--warn); } .fail, .low { color: var(--bad); }
.pill.pass::after { content: " ✓"; } .pill.fail::after { content: " ✕"; } .pill.warn::after { content: " !"; }
ul.notes { color: var(--ink-2); padding-left: 20px; }
code { font-size: 12px; }
"""


def _esc(text: object) -> str:
    return html.escape(str(text))


class _Axes:
    def __init__(self, x0: float, x1: float, y0: float, y1: float, height: int = H) -> None:
        self.x0, self.x1, self.y0, self.y1, self.h = x0, x1, y0, y1, height

    def x(self, v: float) -> float:
        return ML + (v - self.x0) / (self.x1 - self.x0) * (W - ML - MR)

    def y(self, v: float) -> float:
        return MT + (self.y1 - v) / (self.y1 - self.y0) * (self.h - MT - MB)


def _ticks(lo: float, hi: float, target: int = 6) -> list[float]:
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / target
    mag = 10 ** np.floor(np.log10(raw))
    step = min((m * mag for m in (1, 2, 5, 10) if m * mag >= raw), default=raw)
    start = np.ceil(lo / step) * step
    return [float(v) for v in np.arange(start, hi + step / 2, step) if v <= hi + 1e-9]


def _time_ticks(lo: float, hi: float) -> list[float]:
    for step in (30, 60, 120, 300, 600, 900, 1800):
        if (hi - lo) / step <= 10:
            break
    return [float(v) for v in np.arange(np.ceil(lo / step) * step, hi + 1e-6, step)]


def _frame(ax: _Axes, yticks: list[float], yfmt: Callable[[float], str], ylabel: str) -> list[str]:
    out = []
    for v in yticks:
        y = ax.y(v)
        out.append(f'<line class="grid" x1="{ML}" x2="{W - MR}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(f'<text x="{ML - 6}" y="{y + 4:.1f}" text-anchor="end">{_esc(yfmt(v))}</text>')
    base = ax.h - MB
    out.append(f'<line class="axis" x1="{ML}" x2="{W - MR}" y1="{base}" y2="{base}"/>')
    for v in _time_ticks(ax.x0, ax.x1):
        x = ax.x(v)
        out.append(f'<text x="{x:.1f}" y="{base + 16}" text-anchor="middle">{mmss(v, 0)}</text>')
    out.append(f'<text x="12" y="{MT - 22}">{_esc(ylabel)}</text>')
    return out


def _score_class(s: float) -> str:
    cls = "s0"
    for lo, name, _ in SCORE_BINS:
        if s >= lo:
            cls = name
    return cls


def _segment_offset_at(plan: Plan, b_stretched: float) -> float:
    best = plan.segments[0]
    for seg in plan.segments:
        if seg.b_start * plan.ratio <= b_stretched:
            best = seg
    return best.offset


def offsets_svg(plan: Plan) -> tuple[str, int]:
    """Window offsets against target time; returns the SVG and how many windows fell off the plotted range."""
    w = plan.windows
    t = np.asarray(w.get("times", []), dtype=float)
    o = np.asarray(w.get("offsets", []), dtype=float)
    s = np.asarray(w.get("scores", []), dtype=float)
    offs = [seg.offset for seg in plan.segments]
    y0, y1 = min(offs) - 0.75, max(offs) + 0.5
    ax = _Axes(0, plan.a.duration, y0, y1)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Offset of every analysis window over target time">']
    for g in plan.gaps(0.05):
        out.append(
            f'<rect class="gap" x="{ax.x(g.a_start):.1f}" y="{MT}" width="{max(1.0, ax.x(g.a_end) - ax.x(g.a_start)):.1f}" height="{H - MT - MB}"/>'
        )
    out += _frame(ax, _ticks(y0, y1), lambda v: f"{v:+.1f} s", "offset (s)")
    for seg in plan.segments:
        y = ax.y(seg.offset)
        out.append(
            f'<line class="level" x1="{ax.x(seg.a_start):.1f}" x2="{ax.x(seg.a_end):.1f}" y1="{y:.1f}" y2="{y:.1f}"/>'
        )
    off_range = 0
    order = np.argsort(s)
    for i in order:
        if not y0 <= o[i] <= y1:
            off_range += 1
            continue
        at = t[i] + _segment_offset_at(plan, t[i])
        r = 2.5 if s[i] < 0.3 else 4
        out.append(
            f'<circle class="{_score_class(s[i])}" cx="{ax.x(at):.1f}" cy="{ax.y(o[i]):.1f}" r="{r}">'
            f"<title>{mmss(at)}  offset {o[i]:+.4f} s  score {s[i]:.2f}</title></circle>"
        )
    for c in plan.cuts:
        x = ax.x(c.a_time)
        out.append(f'<line class="cut" x1="{x:.1f}" x2="{x:.1f}" y1="{MT}" y2="{H - MB}"/>')
        right = x > W - MR - 60
        anchor, dx = ("end", -4) if right else ("start", 4)
        out.append(
            f'<text class="cutlabel" x="{x + dx:.1f}" y="{MT + 10}" text-anchor="{anchor}">{mmss(c.a_time)}</text>'
        )
    out.append("</svg>")
    return "\n".join(out), off_range


def residual_svg(plan: Plan) -> str:
    """Window offsets minus their segment's offset, in ms: the crisp line at 0 vs the loose dialogue cluster."""
    w = plan.windows
    t = np.asarray(w.get("times", []), dtype=float)
    o = np.asarray(w.get("offsets", []), dtype=float)
    s = np.asarray(w.get("scores", []), dtype=float)
    y0, y1 = -600.0, 200.0
    h = 240
    ax = _Axes(0, plan.a.duration, y0, y1, h)
    out = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="Residual of each window against its segment offset">']
    out += _frame(ax, _ticks(y0, y1, 8), lambda v: f"{v:+.0f}", "window offset - segment offset (ms)")
    for i in np.argsort(s):
        off = _segment_offset_at(plan, t[i])
        r = (o[i] - off) * 1000
        if not y0 <= r <= y1 or s[i] < 0.3:
            continue
        at = t[i] + off
        out.append(
            f'<circle class="{_score_class(s[i])}" cx="{ax.x(at):.1f}" cy="{ax.y(r):.1f}" r="4">'
            f"<title>{mmss(at)}  {r:+.1f} ms  score {s[i]:.2f}</title></circle>"
        )
    out.append("</svg>")
    return "\n".join(out)


def verify_svg(report: VerifyReport, duration: float) -> str:
    """Residual at offset 0 of the rendered dub, with the ±40 ms lip-sync band."""
    t = np.asarray(report.times)
    r = np.asarray(report.residuals) * 1000
    s = np.asarray(report.scores)
    y0, y1 = -600.0, 200.0
    h = 240
    ax = _Axes(0, duration, y0, y1, h)
    out = [f'<svg viewBox="0 0 {W} {h}" role="img" aria-label="Residual offset of the rendered dub">']
    band = NEAR_ZERO * 1000
    out.append(
        f'<rect class="band" x="{ML}" y="{ax.y(band):.1f}" width="{W - ML - MR}" height="{ax.y(-band) - ax.y(band):.1f}"/>'
    )
    out += _frame(ax, _ticks(y0, y1, 8), lambda v: f"{v:+.0f}", "residual at offset 0 (ms; negative = dub late)")
    for i in np.argsort(s):
        if not y0 <= r[i] <= y1 or s[i] < 0.3:
            continue
        out.append(
            f'<circle class="{_score_class(s[i])}" cx="{ax.x(t[i]):.1f}" cy="{ax.y(r[i]):.1f}" r="4">'
            f"<title>{mmss(t[i])}  {r[i]:+.1f} ms  score {s[i]:.2f}</title></circle>"
        )
    out.append("</svg>")
    return "\n".join(out)


def _legend(extra: str = "") -> str:
    dots = "".join(f'<span><i class="dot {c}"></i>score {label}</span>' for _, c, label in SCORE_BINS)
    return f'<div class="legend">{dots}{extra}</div>'


def build_html(plan: Plan, report: VerifyReport | None = None) -> str:
    """The whole report page as one HTML string (no external resources)."""
    a_name, b_name = Path(plan.a.path).name, Path(plan.b.path).name
    plot, off_range = offsets_svg(plan)
    parts = [
        "<!doctype html><html lang=en><head><meta charset=utf-8>",
        '<meta name=viewport content="width=device-width, initial-scale=1">',
        f"<title>dubweld report</title><style>{CSS}</style></head><body><main>",
        "<h1>dubweld sync report</h1>",
        f'<p class="sub">Dub from <code>{_esc(b_name)}</code> onto <code>{_esc(a_name)}</code>. Speed ratio '
        f"{plan.ratio:.10f} ({_esc(plan.ratio_name)}, {_esc(plan.ratio_method)}). Times are target times.</p>",
    ]
    if report is not None:
        parts.append(
            f'<div class="card"><span class="pill {report.verdict}">Verification {report.verdict}</span> '
            f"&nbsp; median {report.median_ms:.1f} ms, max {report.max_ms:.1f} ms, bias {report.bias_ms:+.1f} ms over "
            f"{report.confident} confident windows; {report.matched} of {report.active} audible windows match at "
            f"offset 0.</div>"
        )
    parts += [
        "<h2>Offsets</h2>",
        '<div class="card">',
        plot,
        _legend(
            '<span><i class="line"></i>segment offset</span><span><i class="vline"></i>cut</span>'
            '<span><i class="swatch"></i>no dub (gap)</span>'
        ),
        "</div>",
    ]
    if off_range:
        parts.append(
            f'<p class="sub">{off_range} windows with offsets outside the plotted range (mostly junk '
            f"matches below score 0.5) are not drawn.</p>"
        )
    parts += [
        "<h2>Residuals against the segment offset</h2>",
        '<p class="sub">The music and effects match sits on a flat line at 0. Dubbed dialogue roughly lining up '
        "with the original forms the loose cloud 0.1-0.5 s below; it is ignored.</p>",
        f'<div class="card">{residual_svg(plan)}{_legend()}</div>',
    ]
    if report is not None:
        parts += [
            "<h2>Verification of the rendered dub</h2>",
            '<p class="sub">Each window of the rendered dub re-matched against the target. Green band: within '
            f"±{NEAR_ZERO * 1000:.0f} ms. Readings between {DIALOGUE[1] * 1000:.0f} and {DIALOGUE[0] * 1000:.0f} ms "
            "are the loose dialogue match.</p>",
            f'<div class="card">{verify_svg(report, plan.a.duration)}{_legend()}</div>',
        ]
        notes = [
            f"{mmss(a)} - {mmss(b)} ({b - a:.0f} s) has no match to check; listen there" for a, b in report.uncovered
        ]
        notes += [
            f"OUT OF SYNC {mmss(a)} - {mmss(b)}: {n} windows agree on {r * 1000:+.0f} ms"
            for a, b, r, n in report.offsync
        ]
        notes += [
            f"{mmss(a)} - {mmss(b)}: {n} windows agree on {r * 1000:+.0f} ms where the dub matches nearby; probably "
            "dialogue, listen"
            for a, b, r, n in report.doubtful
        ]
        notes += [f"{mmss(t)}: isolated reading {r * 1000:+.0f} ms (score {s:.2f})" for t, r, s in report.outliers]
        notes += report.messages
        if notes:
            parts.append('<ul class="notes">' + "".join(f"<li>{_esc(n)}</li>" for n in notes) + "</ul>")

    rows = "".join(
        f"<tr><td>{i}</td><td>{mmss(s.a_start)}</td><td>{mmss(s.a_end)}</td><td>{s.offset:+.4f} s</td>"
        f"<td>{mmss(s.b_start)} - {mmss(s.b_end)}</td><td>{s.windows}</td>"
        f"<td>{'' if s.video_offset is None else f'{s.offset - s.video_offset:+.3f} s'}</td></tr>"
        for i, s in enumerate(plan.segments, 1)
    )
    parts += [
        "<h2>Segments</h2>",
        '<div class="card scroll"><table><tr><th>#</th><th>from</th><th>to</th><th>offset</th><th>donor span</th>'
        f"<th>windows</th><th>audio - video</th></tr>{rows}</table></div>",
    ]
    if plan.cuts:
        rows = ""
        for c in plan.cuts:
            evidence = "<br>".join(f"{_esc(e.kind)}: {_esc(e.detail)}" for e in c.evidence)
            what = f"+{c.jump:.2f} s gap" if c.jump >= 0 else f"{c.jump:.2f} s of donor dropped"
            rows += (
                f"<tr><td>{mmss(c.a_time)}</td><td>{c.offset_before:+.3f} → {c.offset_after:+.3f}</td>"
                f'<td>{what}</td><td><span class="pill {c.confidence}">{c.confidence}</span></td><td>{evidence}</td></tr>'
            )
        parts += [
            "<h2>Cuts</h2>",
            '<div class="card scroll"><table><tr><th>at</th><th>offset</th><th>change</th><th>confidence</th>'
            f"<th>evidence</th></tr>{rows}</table></div>",
        ]
    if plan.candidates:
        rows = "".join(
            f"<tr><td>{_esc(c['name'])}</td><td>{c['ratio']:.10f}</td><td>{c['flat']}</td><td>{c['confident']}</td>"
            f"<td>{c['slope']:+.1e}</td></tr>"
            for c in plan.candidates
        )
        parts += [
            "<h2>Speed ratios tried</h2>",
            '<div class="card scroll"><table><tr><th>name</th><th>ratio</th><th>flat windows</th>'
            f"<th>confident windows</th><th>drift</th></tr>{rows}</table></div>",
        ]
    if plan.warnings:
        parts.append(
            '<h2>Warnings</h2><ul class="notes">' + "".join(f"<li>{_esc(w)}</li>" for w in plan.warnings) + "</ul>"
        )
    parts.append("</main></body></html>")
    return "\n".join(parts)


def write_report(path: Path, plan: Plan, report: VerifyReport | None = None) -> None:
    """Write the HTML report to `path`."""
    Path(path).write_text(build_html(plan, report), encoding="utf-8")
