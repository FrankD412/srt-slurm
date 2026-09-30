# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Standalone HTML visualizer for the power collector's ``scrape-timings.jsonl`` sidecar.

Reads the diagnostic file written next to ``power/samples.csv`` (see
``docs/power-telemetry.md`` § "Diagnosing slow scrapes") and emits one
self-contained HTML page with inline SVG charts:

1. slot-grid timeline (one lane per host, one bar per scrape request),
2. schedule lag per host against ``scrape_seq``,
3. per-cycle cost decomposition (slowest request, slowest parse, writer lock
   wait, sample write),
4. write-health strip from the ``cycle_write`` records,
5. host × seq coverage heatmap of ``row_count``.

A ``(hostname, scrape_seq)`` pair with no ``scrape`` record is an endpoint
abandoned at the cycle deadline; it is rendered as a hatched slot rather than
dropped. Stdlib only. The page defaults to dark mode with a light/dark toggle,
and every mark carries a hover tooltip (HTML panel with JS, SVG <title> without).

Usage::

    python3 src/srtctl/analysis/scrape_timings_viz.py power/scrape-timings.jsonl -o out.html
"""

from __future__ import annotations

import argparse
import html
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SVG_W = 1200
PAD_L = 130
PAD_R = 24
PAD_T = 12
PAD_B = 34
PLOT_W = SVG_W - PAD_L - PAD_R

# Outcome colours are CSS classes so they can follow the theme; the hues here
# feed the cost-stack segments and the heatmap ramp, which are inline.
HOST_HUES = [212, 18, 158, 280, 45, 340, 95, 190]


@dataclass
class Scrape:
    hostname: str
    seq: int
    started: float
    finished: float
    duration: float
    parse: float | None
    lag: float | None
    sample_ts: float | None
    http_status: int | None
    error_type: str | None
    row_count: int
    reason_codes: list[str]

    @property
    def is_bracket(self) -> bool:
        return self.lag is None

    @property
    def outcome(self) -> str:
        if self.error_type and "Timeout" in self.error_type:
            return "timeout"
        if self.error_type:
            return "http_error" if self.http_status is not None else "other_error"
        if self.http_status is not None and self.http_status != 200:
            return "http_error"
        return "ok"

    @property
    def css_class(self) -> str:
        return {"ok": "o-ok", "http_error": "o-http", "timeout": "o-timeout", "other_error": "o-other"}[self.outcome]


@dataclass
class CycleWrite:
    seq: int
    scheduled_at: float | None  # unix; null for bracket/manual cycles
    row_count: int
    lock_wait: float
    write_seconds: float
    completed: bool
    error: str | None


@dataclass
class Timings:
    job_id: str | None = None
    run_name: str | None = None
    scrapes: list[Scrape] = field(default_factory=list)
    writes: dict[int, CycleWrite] = field(default_factory=dict)
    dropped_records: int | None = None
    unknown_events: int = 0
    bad_lines: int = 0

    @property
    def hosts(self) -> list[str]:
        return sorted({s.hostname for s in self.scrapes})

    @property
    def seqs(self) -> list[int]:
        return sorted({s.seq for s in self.scrapes} | set(self.writes))

    def by_slot(self) -> dict[tuple[str, int], Scrape]:
        return {(s.hostname, s.seq): s for s in self.scrapes}

    def by_seq(self) -> dict[int, list[Scrape]]:
        out: dict[int, list[Scrape]] = defaultdict(list)
        for s in self.scrapes:
            out[s.seq].append(s)
        return out

    def inferred_interval(self) -> tuple[float, str] | None:
        """Sample interval and how it was obtained.

        Prefer the median gap between consecutive ``cycle_write.scheduled_at_unix``
        slots (exact: the collector's own cadence). Fall back to the median gap
        between cycles' earliest request start for files written before that
        field existed.
        """
        scheduled = sorted(w.scheduled_at for w in self.writes.values() if w.scheduled_at is not None)
        if len(scheduled) >= 2:
            gaps = [scheduled[i + 1] - scheduled[i] for i in range(len(scheduled) - 1)]
            return statistics.median(gaps), "median gap between cycle_write.scheduled_at_unix slots"
        starts = sorted(
            min(s.started for s in group) for group in self.by_seq().values() if any(not s.is_bracket for s in group)
        )
        gaps = [starts[i + 1] - starts[i] for i in range(len(starts) - 1)]
        return (statistics.median(gaps), "inferred: median gap between cycle request starts") if gaps else None


def load_timings(path: Path) -> Timings:
    t = Timings()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                t.bad_lines += 1
                continue
            event = rec.get("event")
            if event == "scrape":
                t.job_id = t.job_id or rec.get("job_id")
                t.run_name = t.run_name or rec.get("run_name")
                t.scrapes.append(
                    Scrape(
                        hostname=str(rec["hostname"]),
                        seq=int(rec["scrape_seq"]),
                        started=float(rec["request_started_at_unix"]),
                        finished=float(rec["request_finished_at_unix"]),
                        duration=float(rec["request_duration_seconds"]),
                        parse=rec.get("parse_seconds"),
                        lag=rec.get("schedule_lag_seconds"),
                        sample_ts=rec.get("sample_timestamp_unix"),
                        http_status=rec.get("http_status"),
                        error_type=rec.get("error_type"),
                        row_count=int(rec.get("row_count", 0)),
                        reason_codes=list(rec.get("reason_codes") or []),
                    )
                )
            elif event == "cycle_write":
                t.job_id = t.job_id or rec.get("job_id")
                t.run_name = t.run_name or rec.get("run_name")
                seq = int(rec["scrape_seq"])
                t.writes[seq] = CycleWrite(
                    seq=seq,
                    scheduled_at=rec.get("scheduled_at_unix"),
                    row_count=int(rec.get("row_count", 0)),
                    lock_wait=float(rec.get("writer_lock_wait_seconds") or 0.0),
                    write_seconds=float(rec.get("sample_write_seconds") or 0.0),
                    completed=bool(rec.get("sample_write_completed")),
                    error=rec.get("sample_write_error"),
                )
            elif event == "diagnostic_summary":
                t.dropped_records = int(rec.get("dropped_records", 0))
            else:
                t.unknown_events += 1
    return t


# --------------------------------------------------------------------------- SVG helpers


def _esc(s: object) -> str:
    return html.escape(str(s), quote=True)


def _fmt_s(seconds: float) -> str:
    if seconds >= 1:
        return f"{seconds:.2f} s"
    return f"{seconds * 1000:.1f} ms"


def _nice_ticks(lo: float, hi: float, n: int = 6) -> list[float]:
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    first = math.ceil(lo / step) * step
    ticks = []
    v = first
    while v <= hi + 1e-12:
        ticks.append(round(v, 10))
        v += step
    return ticks


def _fmt_ms(seconds: float | None) -> str:
    return "—" if seconds is None else _fmt_s(seconds)


def _fmt_clock(unix: float) -> str:
    dt = datetime.fromtimestamp(unix, tz=timezone.utc)
    return dt.strftime("%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _tip(title: str, rows: Sequence[tuple[str, object]], note: str | None = None) -> tuple[str, str]:
    """Attributes for a hoverable element: JSON rows for the HTML tooltip + a <title> fallback.

    Returns ``data-tip="..."`` and a ``<title>`` element; callers place the title
    inside the element. The page script removes the <title>s so both never show.
    """
    payload = {"title": title, "rows": [[k, str(v)] for k, v in rows], "note": note}
    plain = title + " · " + " · ".join(f"{k} {v}" for k, v in rows) + (f" · {note}" if note else "")
    return f'data-tip="{_esc(json.dumps(payload, separators=(",", ":")))}"', f"<title>{_esc(plain)}</title>"


def _fmt_tick(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) >= 100:
        return f"{v:.0f}"
    return f"{v:.3g}"


def _svg_open(height: int, title: str) -> str:
    return (
        f'<svg class="chart" viewBox="0 0 {SVG_W} {height}" role="img" aria-label="{_esc(title)}" '
        f'preserveAspectRatio="xMidYMid meet" style="aspect-ratio:{SVG_W}/{height}">'
        '<defs><pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
        '<line x1="0" y1="0" x2="0" y2="6" class="hatch-line" stroke-width="1.5"/></pattern></defs>'
    )


def _x_axis(y: float, x_of, lo: float, hi: float, label: str, *, integer: bool = False) -> str:
    parts = [f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{PAD_L + PLOT_W}" y2="{y:.1f}" class="ax"/>']
    ticks = list(range(int(lo), int(hi) + 1)) if integer else _nice_ticks(lo, hi)
    if integer and len(ticks) > 30:
        stride = (len(ticks) + 29) // 30
        ticks = ticks[::stride]
    for v in ticks:
        x = x_of(v)
        parts.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x:.1f}" y2="{y + 4:.1f}" class="ax"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{y + 15:.1f}" text-anchor="middle" class="tick">{_fmt_tick(v) if not integer else v}</text>'
        )
    parts.append(
        f'<text x="{PAD_L + PLOT_W / 2:.1f}" y="{y + 29:.1f}" text-anchor="middle" class="axis">{_esc(label)}</text>'
    )
    return "".join(parts)


def _y_axis(y_of, lo: float, hi: float, top: float, bottom: float, label: str) -> str:
    parts = [f'<line x1="{PAD_L}" y1="{top:.1f}" x2="{PAD_L}" y2="{bottom:.1f}" class="ax"/>']
    for v in _nice_ticks(lo, hi, 5):
        y = y_of(v)
        parts.append(f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{PAD_L + PLOT_W}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{PAD_L - 6}" y="{y + 4:.1f}" text-anchor="end" class="tick">{_fmt_tick(v)}</text>')
    mid = (top + bottom) / 2
    parts.append(
        f'<text x="14" y="{mid:.1f}" text-anchor="middle" class="axis" transform="rotate(-90 14 {mid:.1f})">{_esc(label)}</text>'
    )
    return "".join(parts)


def _host_color(index: int) -> str:
    return f"hsl({HOST_HUES[index % len(HOST_HUES)]} 62% 52%)"


# --------------------------------------------------------------------------- charts


_OUTCOME_LABEL = {
    "ok": "OK — HTTP 200, body parsed",
    "http_error": "HTTP error",
    "timeout": "Request timeout",
    "other_error": "Request exception",
}


def _scrape_tip(s: Scrape, w: CycleWrite | None, t0: float) -> tuple[str, list[tuple[str, object]], str | None]:
    """Title, rows and note for one settled scrape record (shared by timeline, lag, coverage)."""
    rows: list[tuple[str, object]] = [
        ("host", s.hostname),
        ("scrape_seq", s.seq),
        ("started", f"{_fmt_clock(s.started)}  (+{s.started - t0:.3f} s)"),
        ("finished", _fmt_clock(s.finished)),
        ("request duration", _fmt_s(s.duration)),
        ("parse", _fmt_ms(s.parse)),
        ("HTTP status", s.http_status if s.http_status is not None else "—"),
    ]
    if s.error_type:
        rows.append(("exception", s.error_type))
    if s.is_bracket:
        rows.append(("schedule lag", "n/a — bracket/manual scrape"))
    else:
        rows.append(("schedule lag", _fmt_ms(s.lag)))
        if w is not None and w.scheduled_at is not None:
            rows.append(("scheduled slot", _fmt_clock(w.scheduled_at)))
    rows.append(("GPU rows", s.row_count))
    if s.sample_ts is not None:
        rows.append(("sample timestamp", _fmt_clock(s.sample_ts)))
    if s.reason_codes:
        rows.append(("reason codes", ", ".join(s.reason_codes)))
    note = None
    if s.outcome == "timeout":
        note = "requests raised a Timeout; the cycle deadline had not yet expired so a record exists."
    elif s.outcome == "http_error":
        note = "Non-2xx response; no rows were parsed and no power sample was invented."
    elif s.row_count == 0:
        note = "Settled with 0 rows — body parsed but yielded no GPU readings."
    return _OUTCOME_LABEL[s.outcome], rows, note


def chart_timeline(t: Timings) -> str:
    hosts, seqs, slots, by_seq = t.hosts, t.seqs, t.by_slot(), t.by_seq()
    lane_h = 26
    height = PAD_T + lane_h * len(hosts) + PAD_B
    t0 = min([s.started for s in t.scrapes] + [w.scheduled_at for w in t.writes.values() if w.scheduled_at is not None])
    t1 = max(s.finished for s in t.scrapes)
    span = max(t1 - t0, 1e-6)

    def x_of(sec: float) -> float:
        return PAD_L + (sec / span) * PLOT_W

    # Cycle spans from the settled scrapes of each seq. A cycle with no scrape
    # record at all (every endpoint abandoned) is anchored at its
    # cycle_write.scheduled_at_unix when present and runs to the next cycle's
    # scheduled slot; only files predating that field fall back to interpolating
    # between neighbours.
    cycle_span: dict[int, tuple[float, float]] = {}
    for seq in seqs:
        group = by_seq.get(seq)
        if group:
            cycle_span[seq] = (min(s.started for s in group) - t0, max(s.finished for s in group) - t0)
    for i, seq in enumerate(seqs):
        if seq in cycle_span:
            continue
        w = t.writes.get(seq)
        if w is not None and w.scheduled_at is not None:
            a = w.scheduled_at - t0
            nxt_sched = next(
                (
                    t.writes[s].scheduled_at
                    for s in seqs[i + 1 :]
                    if s in t.writes and t.writes[s].scheduled_at is not None
                ),
                None,
            )
            b = (nxt_sched - t0) if nxt_sched is not None else min(span, a + 0.02)
            cycle_span[seq] = (a, max(b, a + 0.005))
            continue
        prev = next((cycle_span[s] for s in reversed(seqs[:i]) if s in cycle_span), None)
        nxt = next((cycle_span[s] for s in seqs[i + 1 :] if s in cycle_span), None)
        if prev and nxt:
            cycle_span[seq] = (prev[1], nxt[0])
        elif prev:
            cycle_span[seq] = (prev[1], min(span, prev[1] + (prev[1] - prev[0]) + 0.01))
        elif nxt:
            cycle_span[seq] = (max(0.0, nxt[0] - (nxt[1] - nxt[0]) - 0.01), nxt[0])

    parts = [_svg_open(height, "Scrape request timeline per host")]
    lanes_bottom = PAD_T + lane_h * len(hosts)
    for hi_, host in enumerate(hosts):
        y = PAD_T + hi_ * lane_h
        if hi_ % 2:
            parts.append(f'<rect x="{PAD_L}" y="{y}" width="{PLOT_W}" height="{lane_h}" class="band"/>')
        parts.append(
            f'<text x="{PAD_L - 8}" y="{y + lane_h / 2 + 4:.1f}" text-anchor="end" class="lane">{_esc(host)}</text>'
        )
        for seq in seqs:
            s = slots.get((host, seq))
            if s is None:
                if seq not in cycle_span:
                    continue
                a, b = cycle_span[seq]
                x, w = x_of(a), max(x_of(b) - x_of(a), 4)
                wr = t.writes.get(seq)
                rows: list[tuple[str, object]] = [("host", host), ("scrape_seq", seq)]
                if wr is not None and wr.scheduled_at is not None:
                    rows.append(("scheduled at", _fmt_clock(wr.scheduled_at)))
                rows.append(("slot span", f"{_fmt_s(a)} → {_fmt_s(b)} after t₀"))
                attr, title = _tip(
                    "Abandoned — no scrape record",
                    rows,
                    "The request had not settled when the cycle deadline expired; the collector counted an "
                    "endpoint_timeout and wrote no timing record for this (host, seq).",
                )
                parts.append(
                    f'<rect x="{x:.1f}" y="{y + 4}" width="{w:.1f}" height="{lane_h - 8}" fill="url(#hatch)" '
                    f'class="missing" stroke-dasharray="2 2" {attr}>{title}</rect>'
                )
                continue
            x = x_of(s.started - t0)
            w = max(x_of(s.finished - t0) - x, 2.5)
            attr, title = _tip(*_scrape_tip(s, t.writes.get(seq), t0))
            cls = s.css_class + (" bracket" if s.is_bracket else "")
            parts.append(
                f'<rect x="{x:.1f}" y="{y + 5}" width="{w:.1f}" height="{lane_h - 10}" class="{cls}" rx="1.5" {attr}>'
                f"{title}</rect>"
            )
    # Scheduled slot ticks (cycle_write.scheduled_at_unix): the gap from a tick
    # to the bars that follow it is the schedule lag, drawn on the wall clock.
    for seq in seqs:
        w = t.writes.get(seq)
        if w is None or w.scheduled_at is None:
            continue
        x = x_of(w.scheduled_at - t0)
        group = by_seq.get(seq, [])
        first_start = min((s.started for s in group), default=None)
        rows = [
            ("scrape_seq", seq),
            ("scheduled at", _fmt_clock(w.scheduled_at)),
            ("offset", f"+{w.scheduled_at - t0:.3f} s"),
        ]
        if first_start is not None:
            rows.append(("first request started", f"+{_fmt_s(max(0.0, first_start - w.scheduled_at))} after slot"))
        attr, title = _tip("Scheduled slot", rows)
        parts.append(
            f'<line x1="{x:.1f}" y1="{PAD_T}" x2="{x:.1f}" y2="{lanes_bottom}" class="sched" '
            f'stroke-dasharray="1 3" {attr}>{title}</line>'
        )
    parts.append(_x_axis(lanes_bottom, x_of, 0, span, "wall-clock seconds since the first scheduled slot / request"))
    parts.append("</svg>")
    return "".join(parts)


def chart_lag(t: Timings) -> str:
    hosts, seqs = t.hosts, t.seqs
    plot_h = 200
    height = PAD_T + plot_h + PAD_B
    lags = [s.lag for s in t.scrapes if s.lag is not None]
    if not lags:
        return '<p class="empty">No scheduled scrapes carry a schedule_lag_seconds value.</p>'
    lo, hi = 0.0, max(lags) * 1.08 or 0.01
    seq_lo, seq_hi = seqs[0], seqs[-1]

    def x_of(seq: float) -> float:
        return PAD_L + ((seq - seq_lo) / max(seq_hi - seq_lo, 1)) * PLOT_W

    def y_of(v: float) -> float:
        return PAD_T + plot_h - (v / hi) * plot_h

    parts = [_svg_open(height, "Schedule lag per host by scrape sequence")]
    parts.append(_y_axis(y_of, lo, hi, PAD_T, PAD_T + plot_h, "schedule lag (s)"))
    slots, by_seq = t.by_slot(), t.by_seq()
    for i, host in enumerate(hosts):
        # Break the line at any seq this host has no lag for (abandoned or bracket),
        # so a gap never reads as an interpolated value.
        runs: list[list[tuple[int, float]]] = [[]]
        for seq in seqs:
            s = slots.get((host, seq))
            if s is not None and s.lag is not None:
                runs[-1].append((seq, s.lag))
            elif runs[-1]:
                runs.append([])
        pts = [p for run in runs for p in run]
        if not pts:
            continue
        for run in runs:
            if len(run) < 2:
                continue
            d = " ".join(f"{x_of(seq):.1f},{y_of(lag):.1f}" for seq, lag in run)
            parts.append(f'<polyline points="{d}" fill="none" stroke="{_host_color(i)}" stroke-width="1.8"/>')
        for seq, lag in pts:
            s = slots[(host, seq)]
            w = t.writes.get(seq)
            rows = [("host", host), ("scrape_seq", seq), ("schedule lag", _fmt_s(lag))]
            if w is not None and w.scheduled_at is not None:
                rows.append(("scheduled slot", _fmt_clock(w.scheduled_at)))
            rows.append(("request started", _fmt_clock(s.started)))
            rows.append(("request duration", _fmt_s(s.duration)))
            others = [o.lag for o in by_seq.get(seq, []) if o.lag is not None and o.hostname != host]
            if others:
                rows.append(("other hosts' lag (this seq)", f"{_fmt_s(min(others))} – {_fmt_s(max(others))}"))
            attr, title = _tip("Schedule lag", rows)
            parts.append(
                f'<circle cx="{x_of(seq):.1f}" cy="{y_of(lag):.1f}" r="3" fill="{_host_color(i)}" {attr}>{title}</circle>'
            )
        # gaps in the line = missing (abandoned) or bracket slots
        for seq in seqs:
            if (host, seq) not in slots:
                attr, title = _tip(
                    "Abandoned — no scrape record",
                    [("host", host), ("scrape_seq", seq)],
                    "No lag can be measured: the request never settled before the cycle deadline.",
                )
                parts.append(
                    f'<line x1="{x_of(seq):.1f}" y1="{PAD_T}" x2="{x_of(seq):.1f}" y2="{PAD_T + plot_h}" '
                    f'stroke="{_host_color(i)}" stroke-dasharray="2 3" opacity="0.6" stroke-width="3" {attr}>{title}</line>'
                )
    parts.append(_x_axis(PAD_T + plot_h, x_of, seq_lo, seq_hi, "scrape_seq", integer=True))
    parts.append("</svg>")
    legend = "".join(
        f'<span class="key"><i style="background:{_host_color(i)}"></i>{_esc(host)}</span>'
        for i, host in enumerate(hosts)
    )
    return f'<div class="legend">{legend}</div>' + "".join(parts)


def chart_cycle_cost(t: Timings) -> str:
    seqs, by_seq = t.seqs, t.by_seq()
    plot_h = 220
    height = PAD_T + plot_h + PAD_B
    seq_lo, seq_hi = seqs[0], seqs[-1]
    inferred = t.inferred_interval()
    interval, interval_how = inferred if inferred is not None else (None, "")
    stacks: dict[int, list[tuple[str, float, str]]] = {}
    for seq in seqs:
        group = by_seq.get(seq, [])
        w = t.writes.get(seq)
        stacks[seq] = [
            ("slowest request", max((s.duration for s in group), default=0.0), "hsl(212 62% 55%)"),
            ("slowest parse", max((s.parse for s in group if s.parse is not None), default=0.0), "hsl(158 50% 48%)"),
            ("writer lock wait", w.lock_wait if w else 0.0, "hsl(45 80% 55%)"),
            ("sample write", w.write_seconds if w else 0.0, "hsl(18 70% 55%)"),
        ]
    hi = max([sum(v for _, v, _ in st) for st in stacks.values()] + [interval or 0.0]) * 1.1 or 0.01

    def x_of(seq: float) -> float:
        return PAD_L + ((seq - seq_lo + 0.5) / (seq_hi - seq_lo + 1)) * PLOT_W

    def y_of(v: float) -> float:
        return PAD_T + plot_h - (v / hi) * plot_h

    bar_w = max(2.0, PLOT_W / (seq_hi - seq_lo + 1) * 0.7)
    parts = [_svg_open(height, "Per-cycle cost decomposition")]
    parts.append(_y_axis(y_of, 0.0, hi, PAD_T, PAD_T + plot_h, "seconds in the cycle"))
    for seq, st in stacks.items():
        base = 0.0
        x = x_of(seq) - bar_w / 2
        group = by_seq.get(seq, [])
        w = t.writes.get(seq)
        slowest = max(group, key=lambda s: s.duration, default=None)
        total = sum(v for _, v, _ in st)
        rows: list[tuple[str, object]] = [("scrape_seq", seq), ("total", _fmt_s(total))]
        rows += [(name, _fmt_s(v)) for name, v, _ in st]
        if slowest is not None:
            rows.append(("slowest host", f"{slowest.hostname} ({_fmt_s(slowest.duration)}, {slowest.outcome})"))
        rows.append(("hosts settled", f"{len(group)} of {len(t.hosts)}"))
        if interval is not None:
            rows.append(("vs interval", f"{100 * total / interval:.0f}% of {_fmt_s(interval)}"))
        note = None
        if w is None:
            note = "No cycle_write record for this seq: lock-wait and write terms are unknown (shown as 0)."
        elif interval is not None and total > interval:
            note = "Cycle cost exceeded the sample interval — the next slot starts late and lag accumulates."
        attr, title = _tip("Cycle cost", rows, note)
        # One invisible hit-rect spanning the full column so the tooltip works on thin bars too.
        parts.append(
            f'<rect x="{x:.1f}" y="{PAD_T}" width="{bar_w:.1f}" height="{plot_h}" fill="transparent" class="hit" {attr}>'
            f"{title}</rect>"
        )
        for _name, v, color in st:
            if v <= 0:
                continue
            y_top, y_bot = y_of(base + v), y_of(base)
            parts.append(
                f'<rect x="{x:.1f}" y="{y_top:.1f}" width="{bar_w:.1f}" height="{max(y_bot - y_top, 0.5):.1f}" '
                f'fill="{color}" pointer-events="none"/>'
            )
            base += v
        if w is None:
            parts.append(
                f'<text x="{x_of(seq):.1f}" y="{PAD_T + plot_h - 3:.1f}" text-anchor="middle" class="tick warn" '
                f'pointer-events="none">?</text>'
            )
    if interval is not None:
        y = y_of(interval)
        # Put the label over whichever end of the plot has the shorter bars,
        # checking only the bars the label actually covers (~7 px per char).
        how_short = "from scheduled slots" if "scheduled_at_unix" in interval_how else "inferred from request starts"
        label = f"sample interval ≈ {_fmt_s(interval)} ({how_short})"
        covered = min(len(seqs), max(1, math.ceil(len(label) * 7.0 / (PLOT_W / len(seqs)))))
        totals = {seq: sum(v for _, v, _ in st) for seq, st in stacks.items()}
        left_max = max(totals[s] for s in seqs[:covered])
        right_max = max(totals[s] for s in seqs[-covered:])
        if left_max <= right_max:
            lx, anchor = PAD_L + 6, "start"
        else:
            lx, anchor = PAD_L + PLOT_W - 4, "end"
        parts.append(
            f'<line x1="{PAD_L}" y1="{y:.1f}" x2="{PAD_L + PLOT_W}" y2="{y:.1f}" class="ref" stroke-dasharray="6 3"/>'
            f'<text x="{lx}" y="{y - 4:.1f}" text-anchor="{anchor}" class="tick ref-label">{_esc(label)}</text>'
        )
    parts.append(_x_axis(PAD_T + plot_h, x_of, seq_lo, seq_hi, "scrape_seq", integer=True))
    parts.append("</svg>")
    legend = "".join(f'<span class="key"><i style="background:{c}"></i>{n}</span>' for n, _, c in stacks[seqs[0]])
    return f'<div class="legend">{legend}</div>' + "".join(parts)


def chart_write_health(t: Timings) -> str:
    seqs = t.seqs
    strip_h = 28
    height = PAD_T + strip_h + PAD_B
    seq_lo, seq_hi = seqs[0], seqs[-1]

    def x_of(seq: float) -> float:
        return PAD_L + ((seq - seq_lo + 0.5) / (seq_hi - seq_lo + 1)) * PLOT_W

    cell_w = max(2.0, PLOT_W / (seq_hi - seq_lo + 1) * 0.85)
    parts = [_svg_open(height, "Sample batch write health per cycle")]
    parts.append(
        f'<text x="{PAD_L - 8}" y="{PAD_T + strip_h / 2 + 4:.1f}" text-anchor="end" class="lane">cycle_write</text>'
    )
    failures: list[str] = []
    for seq in seqs:
        w = t.writes.get(seq)
        x = x_of(seq) - cell_w / 2
        if w is None:
            fill, cls = 'fill="url(#hatch)"', "missing"
            attr, title = _tip(
                "No cycle_write record",
                [("scrape_seq", seq)],
                "Either the diagnostics queue dropped it (see dropped_records) or the file was cut off.",
            )
        else:
            rows = [
                ("scrape_seq", seq),
                (
                    "scheduled slot",
                    _fmt_clock(w.scheduled_at) if w.scheduled_at is not None else "n/a — bracket/manual",
                ),
                ("rows attempted", w.row_count),
                ("writer lock wait", _fmt_s(w.lock_wait)),
                ("append + flush", _fmt_s(w.write_seconds)),
                ("completed", "yes" if w.completed else "no"),
            ]
            if w.completed:
                fill, cls = "", "o-ok cell"
                attr, title = _tip("Batch written", rows)
            else:
                fill, cls = "", "o-http cell"
                why = w.error or "refused (session finalizing)"
                rows.append(("reason", why))
                attr, title = _tip(
                    "Batch NOT written",
                    rows,
                    "sample_write_error names the exception class; null means the session had already disabled "
                    "artifact mutation and refused the append.",
                )
                failures.append(f"seq {seq}: {why}")
        parts.append(
            f'<rect x="{x:.1f}" y="{PAD_T + 4}" width="{cell_w:.1f}" height="{strip_h - 8}" {fill} class="{cls}" {attr}>'
            f"{title}</rect>"
        )
    parts.append(_x_axis(PAD_T + strip_h, x_of, seq_lo, seq_hi, "scrape_seq", integer=True))
    parts.append("</svg>")
    note = (
        f'<p class="note warn">{len(failures)} failed batch write(s): {_esc("; ".join(failures))}</p>'
        if failures
        else '<p class="note">Every cycle_write record reports sample_write_completed = true.</p>'
    )
    return "".join(parts) + note


def chart_coverage(t: Timings) -> str:
    hosts, seqs, slots = t.hosts, t.seqs, t.by_slot()
    row_h = 22
    height = PAD_T + row_h * len(hosts) + PAD_B
    seq_lo, seq_hi = seqs[0], seqs[-1]
    max_rows = max((s.row_count for s in t.scrapes), default=1) or 1
    t0 = min(s.started for s in t.scrapes)

    def x_of(seq: float) -> float:
        return PAD_L + ((seq - seq_lo + 0.5) / (seq_hi - seq_lo + 1)) * PLOT_W

    cell_w = PLOT_W / (seq_hi - seq_lo + 1)
    parts = [_svg_open(height, "GPU rows per host per scrape sequence")]
    for i, host in enumerate(hosts):
        y = PAD_T + i * row_h
        parts.append(
            f'<text x="{PAD_L - 8}" y="{y + row_h / 2 + 4:.1f}" text-anchor="end" class="lane">{_esc(host)}</text>'
        )
        for seq in seqs:
            s = slots.get((host, seq))
            x = x_of(seq) - cell_w / 2
            if s is None:
                attrs = 'fill="url(#hatch)" class="missing"'
                attr, title = _tip(
                    "Abandoned — no scrape record",
                    [("host", host), ("scrape_seq", seq)],
                    "Unsettled at the cycle deadline; no rows and no timing record.",
                )
            else:
                if s.row_count == 0:
                    attrs = 'class="o-http cell"'
                else:
                    # Ramp on the ok hue: full rows = the ok colour, fewer rows fade toward the surface.
                    alpha = 0.3 + 0.7 * (s.row_count / max_rows)
                    attrs = f'class="o-ok cell" fill-opacity="{alpha:.2f}"'
                heading, rows, note = _scrape_tip(s, t.writes.get(seq), t0)
                rows.insert(2, ("rows vs max seen", f"{s.row_count} / {max_rows}"))
                attr, title = _tip(heading, rows, note)
            parts.append(
                f'<rect x="{x + 0.5:.1f}" y="{y + 1}" width="{max(cell_w - 1, 1):.1f}" height="{row_h - 2}" {attrs} {attr}>'
                f"{title}</rect>"
            )
    parts.append(_x_axis(PAD_T + row_h * len(hosts), x_of, seq_lo, seq_hi, "scrape_seq", integer=True))
    parts.append("</svg>")
    return "".join(parts)


# --------------------------------------------------------------------------- page

# Tokens and both palettes mirror src/srtctl/analysis/power_report_html.py (_CSS,
# _SLOT_LIGHT / _SLOT_DARK) so the two reports read as one family. Unlike the
# power report, this page defaults to dark and exposes an explicit toggle stored
# in localStorage; the OS preference is not consulted.
_CSS = """
:root[data-theme=dark] { color-scheme: dark;
  --page: #0d0d0d; --surface: #1a1a19; --ink-primary: #ffffff; --ink-secondary: #c3c2b7;
  --ink-muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --slot-0: #3987e5; --slot-1: #d95926; --slot-2: #199e70;
  --ok: #199e70; --err: #e5484d; --timeout: #d95926; --other: #b07cd8; --warn: #f0a04b; }
:root[data-theme=light] { color-scheme: light;
  --page: #f9f9f7; --surface: #fcfcfb; --ink-primary: #0b0b0b; --ink-secondary: #52514e;
  --ink-muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --slot-0: #2a78d6; --slot-1: #eb6834; --slot-2: #1baf7a;
  --ok: #1baf7a; --err: #d33b3b; --timeout: #eb6834; --other: #8e4fc2; --warn: #b8741a; }
body { margin: 0; padding: 24px; background: var(--page); color: var(--ink-primary);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
.page-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 0 0 6px; }
.subtitle { color: var(--ink-secondary); margin: 0 0 20px; }
.theme-toggle { appearance: none; font: inherit; font-size: 12px; font-weight: 600; color: var(--ink-secondary);
  background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 5px 10px; cursor: pointer;
  white-space: nowrap; }
.theme-toggle:hover { color: var(--ink-primary); border-color: var(--ink-muted); }
.stat-cards { display: flex; gap: 12px; margin: 4px 0 24px; flex-wrap: wrap; }
.stat-card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px 18px; min-width: 110px; }
.stat-card-num { font-size: 22px; font-weight: 700; margin: 0; font-variant-numeric: tabular-nums; }
.stat-card-label { color: var(--ink-secondary); font-size: 12px; margin: 2px 0 0; }
.stat-card.warn .stat-card-num { color: var(--warn); }
.chart-panel { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px 16px 8px; margin-bottom: 16px; }
.chart-panel p.blurb { margin: 0 0 8px; color: var(--ink-secondary); font-size: 13px; }
svg.chart { display: block; width: 100%; height: auto; }
svg text { fill: var(--ink-primary); font-family: inherit; }
svg text.tick { font-size: 11px; fill: var(--ink-muted); }
svg text.axis { font-size: 11px; font-weight: 600; letter-spacing: .02em; fill: var(--ink-secondary); }
svg text.lane { font-size: 12px; fill: var(--ink-secondary); }
svg text.warn { fill: var(--err); font-weight: 700; }
svg text.ref-label { fill: var(--ink-secondary); }
svg .ax { stroke: var(--axis); }
svg .grid { stroke: var(--grid); }
svg .band { fill: var(--ink-primary); fill-opacity: .035; }
svg .ref { stroke: var(--ink-secondary); }
svg .sched { stroke: var(--ink-primary); stroke-opacity: .55; }
svg .hatch-line { stroke: var(--ink-muted); }
svg .missing { stroke: var(--ink-muted); }
svg .o-ok { fill: var(--ok); } svg .o-http { fill: var(--err); } svg .o-timeout { fill: var(--timeout); } svg .o-other { fill: var(--other); }
svg .bracket { stroke: var(--slot-0); stroke-width: 2; stroke-dasharray: 3 2; }
svg .cell { stroke: var(--grid); }
.legend { display: flex; gap: 6px 14px; flex-wrap: wrap; margin: 4px 0 8px; font-size: 12px; color: var(--ink-secondary); }
.legend .key { display: inline-flex; align-items: center; gap: 6px; padding: 3px 8px; border: 1px solid var(--border); border-radius: 4px; }
.legend .key i { display: inline-block; width: 12px; height: 12px; border-radius: 2px; border: 1px solid var(--border); }
.legend .key i.ok { background: var(--ok); } .legend .key i.err { background: var(--err); }
.legend .key i.timeout { background: var(--timeout); } .legend .key i.other { background: var(--other); }
.legend .key i.hatch { background: repeating-linear-gradient(45deg, var(--ink-muted) 0 1.5px, transparent 1.5px 6px); }
.legend .key i.bracket { background: var(--ok); outline: 2px dashed var(--slot-0); outline-offset: -2px; }
.legend .key i.sched { width: 0; border: 0; border-left: 1.5px dotted var(--ink-primary); border-radius: 0; height: 14px; }
.note, .empty { margin: 6px 0 0; color: var(--ink-muted); font-size: 12px; }
.note.warn { color: var(--warn); }
footer { color: var(--ink-muted); font-size: 12px; margin-top: 24px; }
svg [data-tip] { cursor: help; }
svg [data-tip]:hover { filter: brightness(1.25); }
svg .hit:hover { fill: var(--ink-primary); fill-opacity: .06; }
.tooltip { position: fixed; pointer-events: none; background: var(--surface); border: 1px solid var(--border);
  border-radius: 4px; padding: 7px 10px; font-size: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.25); opacity: 0;
  z-index: 10; max-width: 420px; transition: opacity .08s; }
.tooltip.on { opacity: 1; }
.tooltip .t-title { font-weight: 700; margin-bottom: 4px; color: var(--ink-primary); }
.tooltip .t-row { display: flex; justify-content: space-between; gap: 14px; line-height: 1.45; }
.tooltip .t-key { color: var(--ink-muted); white-space: nowrap; }
.tooltip .t-val { font-weight: 600; font-variant-numeric: tabular-nums; text-align: right; }
.tooltip .t-note { margin-top: 5px; padding-top: 5px; border-top: 1px solid var(--grid); color: var(--ink-secondary);
  font-size: 11.5px; white-space: normal; }
"""

# Tooltip: one fixed-position panel fed from data-tip JSON on hover. The SVG
# <title> fallbacks are removed at load so the browser's native tooltip does not
# appear alongside it; without JS they remain and still work.
_TOOLTIP_JS = """
document.addEventListener('DOMContentLoaded', function () {
  var tip = document.createElement('div'); tip.className = 'tooltip'; document.body.appendChild(tip);
  document.querySelectorAll('svg [data-tip] > title').forEach(function (t) { t.remove(); });
  function node(tag, cls, text, parent) {
    var el = document.createElement(tag); el.className = cls; el.textContent = text; parent.appendChild(el); return el;
  }
  function show(el, ev) {
    var d; try { d = JSON.parse(el.getAttribute('data-tip')); } catch (e) { return; }
    tip.textContent = '';
    node('div', 't-title', d.title, tip);
    d.rows.forEach(function (r) {
      var row = node('div', 't-row', '', tip);
      node('span', 't-key', r[0], row);
      node('span', 't-val', r[1], row);
    });
    if (d.note) node('div', 't-note', d.note, tip);
    tip.classList.add('on'); move(ev);
  }
  function move(ev) {
    var pad = 14, w = tip.offsetWidth, hgt = tip.offsetHeight;
    var x = ev.clientX + pad, y = ev.clientY + pad;
    if (x + w > window.innerWidth - 8) x = ev.clientX - w - pad;
    if (y + hgt > window.innerHeight - 8) y = ev.clientY - hgt - pad;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }
  document.addEventListener('pointerover', function (ev) {
    var el = ev.target.closest ? ev.target.closest('svg [data-tip]') : null;
    if (el) show(el, ev);
  });
  document.addEventListener('pointermove', function (ev) { if (tip.classList.contains('on')) move(ev); });
  document.addEventListener('pointerout', function (ev) {
    var el = ev.target.closest ? ev.target.closest('svg [data-tip]') : null;
    if (el && !(ev.relatedTarget && el.contains(ev.relatedTarget))) tip.classList.remove('on');
  });
});
"""

# Applied before first paint so a stored light preference never flashes dark.
_THEME_JS = """
(function () {
  var KEY = "scrape-timings-theme";
  var root = document.documentElement;
  try { var saved = localStorage.getItem(KEY); if (saved === "light" || saved === "dark") root.dataset.theme = saved; } catch (e) {}
  function label(btn) { btn.textContent = root.dataset.theme === "dark" ? "Switch to light mode" : "Switch to dark mode"; }
  document.addEventListener("DOMContentLoaded", function () {
    var btn = document.querySelector(".theme-toggle");
    if (!btn) return;
    label(btn);
    btn.addEventListener("click", function () {
      root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
      try { localStorage.setItem(KEY, root.dataset.theme); } catch (e) {}
      label(btn);
    });
  });
})();
"""


def _timeline_legend() -> str:
    keys = [
        ("ok", "HTTP 200, parsed"),
        ("err", "HTTP error (non-200 / HTTPError)"),
        ("timeout", "request timeout"),
        ("other", "other request exception"),
        ("hatch", "no scrape record — abandoned at the cycle deadline"),
        ("bracket", "bracket / manual scrape (no scheduled slot, no lag)"),
        ("sched", "scheduled slot (cycle_write.scheduled_at_unix)"),
    ]
    out = "".join(f'<span class="key"><i class="{cls}"></i>{label}</span>' for cls, label in keys)
    return f'<div class="legend">{out}</div>'


def _stat_card(num: object, label: str, *, warn: bool = False) -> str:
    cls = "stat-card warn" if warn else "stat-card"
    return (
        f'<div class="{cls}"><p class="stat-card-num">{_esc(num)}</p><p class="stat-card-label">{_esc(label)}</p></div>'
    )


def render_html(t: Timings, source: Path) -> str:
    hosts, seqs = t.hosts, t.seqs
    if not t.scrapes:
        raise SystemExit(f"{source}: no scrape records found")
    scheduled = [s for s in t.scrapes if not s.is_bracket]
    brackets = [s for s in t.scrapes if s.is_bracket]
    missing = sum(1 for h in hosts for q in seqs if (h, q) not in t.by_slot())
    failed = sum(1 for s in t.scrapes if s.outcome != "ok")
    write_failures = sum(1 for w in t.writes.values() if not w.completed)
    cards = [
        _stat_card(f"{seqs[0]} – {seqs[-1]}", f"scrape_seq range ({len(seqs)} cycles)"),
        _stat_card(len(hosts), "hosts seen"),
        _stat_card(f"{len(scheduled)} + {len(brackets)}", "scrape records: scheduled + bracket"),
        _stat_card(failed, "failed requests", warn=failed > 0),
        _stat_card(missing, "abandoned slots (host × seq, no record)", warn=missing > 0),
        _stat_card(
            f"{len(t.writes)} / {write_failures}", "cycle_write records / failed writes", warn=write_failures > 0
        ),
    ]
    if t.dropped_records is None:
        cards.append(_stat_card("missing", "diagnostic_summary — file may be cut off", warn=True))
    else:
        cards.append(_stat_card(t.dropped_records, "dropped diagnostic records", warn=t.dropped_records > 0))
    if t.bad_lines:
        cards.append(_stat_card(t.bad_lines, "unparseable lines", warn=True))
    if t.unknown_events:
        cards.append(_stat_card(t.unknown_events, "unknown event kinds", warn=True))

    def panel(title: str, blurb: str, body: str) -> str:
        return f'<section class="chart-panel"><h2>{_esc(title)}</h2><p class="blurb">{_esc(blurb)}</p>{body}</section>'

    body = "".join(
        [
            panel(
                "Scrape request timeline — one lane per host, one bar per request (request_started_at_unix → request_finished_at_unix)",
                "Bar colour is the request outcome; hatched slots are (host, scrape_seq) pairs with no scrape record, i.e. the "
                "endpoint was still unsettled at the cycle deadline. Dashed blue outline = bracket/manual scrape. Dotted vertical "
                "ticks are each cycle's scheduled slot; the gap from a tick to its bars is the schedule lag. Hover a bar for details.",
                _timeline_legend() + chart_timeline(t),
            ),
            panel(
                "Schedule lag per host by scrape_seq — request start minus the cycle's scheduled slot (schedule_lag_seconds)",
                "Steady growth means cycle overrun is accumulating; a spike on every host at one seq means one slow endpoint stalled "
                "that batch. Vertical dashed ticks mark seqs where that host has no record.",
                chart_lag(t),
            ),
            panel(
                "Per-cycle cost decomposition by scrape_seq — slowest request, slowest parse, writer lock wait, sample write",
                "The cycle waits for its slowest endpoint, so the max request_duration_seconds across hosts is the dominant term. "
                "Dashed line = sample interval, taken as the median gap between cycle_write.scheduled_at_unix slots (the setting "
                "itself is not in the file).",
                chart_cycle_cost(t),
            ),
            panel(
                "Sample batch write health per scrape_seq — cycle_write.sample_write_completed",
                "Green = batch appended and flushed; red = not written (sample_write_error names the exception, or the session was "
                "already finalizing and refused the batch); hatched = no cycle_write record for that seq.",
                chart_write_health(t),
            ),
            panel(
                "GPU rows returned per host per scrape_seq — scrape.row_count",
                "Stronger green = more rows; red = a settled request that produced 0 rows (failure); hatched = abandoned (no record).",
                chart_coverage(t),
            ),
        ]
    )
    title = f"{_esc(t.run_name or '—')}" + (f" · job {_esc(t.job_id)}" if t.job_id else "")
    return (
        '<!DOCTYPE html><html lang="en" data-theme="dark"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>scrape-timings — {_esc(t.run_name or source.name)}</title><style>{_CSS}</style>"
        f"<script>{_THEME_JS}</script><script>{_TOOLTIP_JS}</script></head><body>"
        '<div class="page-head"><div>'
        f"<h1>Power collector scrape timings — {title}</h1>"
        f'<p class="subtitle">Source: {_esc(source)} · diagnostic sidecar written by the power collector; '
        "not publication-validation evidence.</p></div>"
        '<button type="button" class="theme-toggle">Switch to light mode</button></div>'
        f'<div class="stat-cards">{"".join(cards)}</div>{body}'
        "<footer>Generated by srtctl.analysis.scrape_timings_viz · every value is read from the JSONL; "
        "the sample interval is the only derived quantity.</footer></body></html>"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render scrape-timings.jsonl as a standalone HTML page.")
    parser.add_argument("timings", type=Path, help="path to power/scrape-timings.jsonl")
    parser.add_argument("-o", "--output", type=Path, required=True, help="output .html path")
    args = parser.parse_args(argv)
    timings = load_timings(args.timings)
    args.output.write_text(render_html(timings, args.timings), encoding="utf-8")
    print(f"wrote {args.output} ({len(timings.scrapes)} scrape records, {len(timings.writes)} cycles)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
