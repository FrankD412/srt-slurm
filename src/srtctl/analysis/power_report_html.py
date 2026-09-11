# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-contained HTML power report: a concurrency-level stats table plus
GPU/CPU power-over-time charts, built from the same artifacts as
``power_energy_report.py``.

Reuses that module's CSV-loading (``load_gpu_samples``, ``load_cpu_samples``,
``build_reports``) rather than re-parsing the power CSVs, so the two reports never
disagree about what a device's series looks like. CSV *discovery* is its own local
helper here (see ``_discover_power_csvs``): unlike the JSON report, chart-only mode
must work even without a recognized benchmark.out, e.g. mid-run or serve-only jobs.

Charts are faceted by node (one panel per hostname) because a run can carry more
GPUs than a line chart can color distinctly: only the first three categorical
slots in the project's validated palette (see the ``dataviz`` skill's
``references/palette.md``) clear the all-pairs CVD/contrast gates that apply
whenever multiple lines can sit side by side. A fourth device per node reuses the
first slot with a dashed stroke; every line also carries a direct end-of-line
label, so identity never depends on hue alone.

No external JS/CSS: everything (styles, downsampled series data, the crosshair
tooltip) is inlined so the file opens standalone from a browser or an artifact
store with no network access.
"""

from __future__ import annotations

import html
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from srtctl.analysis.power_energy_report import (
    CPU_SAMPLES_DIRNAMES,
    PowerReportError,
    build_reports,
    load_cpu_samples,
    load_gpu_roles,
    load_gpu_samples,
    report_to_dict,
)

if TYPE_CHECKING:
    from srtctl.core.runtime import RuntimeContext

logger = logging.getLogger(__name__)

HTML_FILENAME = "power_report.html"

# Only the first three slots of the project's default categorical palette clear
# the all-pairs CVD/contrast gates in both light and dark mode (see palette.md).
# A fourth series per facet reuses slot 1 with a dashed stroke instead of a new hue.
_SLOT_LIGHT = ("#2a78d6", "#eb6834", "#1baf7a")
_SLOT_DARK = ("#3987e5", "#d95926", "#199e70")

# CSS stroke-dasharray per pattern index: solid, dashed, dotted. A facet with more
# than 9 devices (3 hues x 3 patterns) starts repeating combinations; unusual for
# a single node and degrades to relying on the legend + direct labels only.
_STROKE_PATTERNS = ("", "5 4", "1.5 3")

# Cap plotted points per series: two per bucket (min + max), so short spikes in an
# otherwise-smooth power trace survive downsampling instead of being averaged away.
_MAX_BUCKETS_PER_SERIES = 1200


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------


def _downsample_minmax(
    times: np.ndarray, watts: np.ndarray, max_buckets: int = _MAX_BUCKETS_PER_SERIES
) -> tuple[np.ndarray, np.ndarray]:
    """Bucket into ``max_buckets`` chunks, keeping each bucket's min and max sample.

    Preserves spikes that a plain stride/average downsample would smooth away,
    while keeping the embedded payload bounded regardless of the run's duration
    or sample rate.
    """
    n = len(times)
    if n <= max_buckets * 2:
        return times, watts
    edges = np.linspace(0, n, max_buckets + 1).astype(int)
    out_t: list[float] = []
    out_w: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        if hi <= lo:
            continue
        seg_t, seg_w = times[lo:hi], watts[lo:hi]
        i_min, i_max = int(np.argmin(seg_w)), int(np.argmax(seg_w))
        for i in sorted({i_min, i_max}):
            out_t.append(float(seg_t[i]))
            out_w.append(float(seg_w[i]))
    return np.array(out_t), np.array(out_w)


# ---------------------------------------------------------------------------
# Per-facet series preparation
# ---------------------------------------------------------------------------


def _series_stats(watts: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(watts.mean()),
        "min": float(watts.min()),
        "p50": float(np.percentile(watts, 50)),
        "p95": float(np.percentile(watts, 95)),
        "max": float(watts.max()),
    }


def _build_facets(
    per_device: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]],
    *,
    label_fmt: str,
) -> list[dict]:
    """One facet per hostname, its devices sorted by index, series downsampled and
    time-shifted to seconds since the earliest sample across every facet (a shared
    x-axis so facets stay visually comparable)."""
    if not per_device:
        return []
    global_start = min(times[0] for times, _ in per_device.values() if len(times))

    by_host: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for (host, index), (times, watts) in per_device.items():
        if len(times) == 0:
            continue
        by_host.setdefault(host, []).append((index, times, watts))

    facets = []
    for host in sorted(by_host):
        devices = sorted(by_host[host], key=lambda d: d[0])
        series = []
        for position, (index, times, watts) in enumerate(devices):
            ds_times, ds_watts = _downsample_minmax(times, watts)
            rel_times = (ds_times - global_start).round(2)
            # Cycle through the 3 validated hues; each cycle ("generation") past the
            # first reuses a hue with the next stroke pattern so no two devices in a
            # facet share both a color and a pattern until >9 devices in one facet.
            slot = position % len(_SLOT_LIGHT)
            generation = position // len(_SLOT_LIGHT)
            series.append(
                {
                    "label": label_fmt.format(index=index),
                    "slot": slot,
                    "pattern": generation % len(_STROKE_PATTERNS),
                    "t": rel_times.tolist(),
                    "w": [round(v, 2) for v in ds_watts.tolist()],
                    "stats": _series_stats(watts),
                }
            )
        facets.append({"host": host, "series": series})
    return facets


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------

_LIGHT_SLOT_VARS = f"--slot-0: {_SLOT_LIGHT[0]}; --slot-1: {_SLOT_LIGHT[1]}; --slot-2: {_SLOT_LIGHT[2]};"
_DARK_SLOT_VARS = f"--slot-0: {_SLOT_DARK[0]}; --slot-1: {_SLOT_DARK[1]}; --slot-2: {_SLOT_DARK[2]};"

_CSS = """
:root, .light { color-scheme: light; }
body {
  margin: 0; padding: 24px; background: var(--page); color: var(--ink-primary);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  --page: #f9f9f7; --surface: #fcfcfb; --ink-primary: #0b0b0b; --ink-secondary: #52514e;
  --ink-muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  __LIGHT_SLOT_VARS__
}
@media (prefers-color-scheme: dark) {
  body {
    --page: #0d0d0d; --surface: #1a1a19; --ink-primary: #ffffff; --ink-secondary: #c3c2b7;
    --ink-muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    __DARK_SLOT_VARS__
  }
}
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 32px 0 12px; }
.subtitle { color: var(--ink-secondary); margin: 0 0 24px; }
table { border-collapse: collapse; width: 100%; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
th, td { text-align: right; padding: 6px 10px; font-variant-numeric: tabular-nums; border-bottom: 1px solid var(--grid); }
th:first-child, td:first-child { text-align: left; font-variant-numeric: normal; }
th { color: var(--ink-secondary); font-weight: 600; font-size: 12px; text-transform: uppercase; letter-spacing: .02em; background: var(--page); }
tr:last-child td { border-bottom: none; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; margin: 4px 0 10px; font-size: 12px; color: var(--ink-secondary); }
.legend-key { display: inline-flex; align-items: center; gap: 6px; }
.legend-swatch { width: 16px; height: 2px; border-radius: 1px; }
.facet { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px 16px 4px; margin-bottom: 16px; }
.facet-title { font-weight: 600; font-size: 13px; margin: 0 0 4px; }
.chart-wrap { position: relative; }
svg.chart { width: 100%; height: 220px; display: block; overflow: visible; }
.gridline { stroke: var(--grid); stroke-width: 1; }
.axis-text { fill: var(--ink-muted); font-size: 10px; }
.end-label { font-size: 10px; fill: var(--ink-secondary); }
.crosshair { stroke: var(--axis); stroke-width: 1; pointer-events: none; opacity: 0; }
.tooltip {
  position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border);
  border-radius: 4px; padding: 6px 8px; font-size: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.15);
  opacity: 0; transform: translate(-50%, -110%); white-space: nowrap; z-index: 10;
}
.tooltip .t-time { color: var(--ink-muted); margin-bottom: 4px; }
.tooltip .t-row { display: flex; gap: 8px; align-items: center; }
.tooltip .t-key { width: 12px; height: 2px; flex: none; }
.tooltip .t-val { font-weight: 600; font-variant-numeric: tabular-nums; }
.stats-toggle { color: var(--ink-muted); font-size: 11px; cursor: pointer; user-select: none; }
.stats-table { margin: 8px 0 12px; font-size: 12px; }
footer { color: var(--ink-muted); font-size: 12px; margin-top: 32px; }
code { background: var(--page); padding: 1px 4px; border-radius: 3px; }
.tabs { display: flex; gap: 4px; border-bottom: 1px solid var(--border); margin-bottom: 20px; }
.tab-btn {
  appearance: none; background: none; border: none; color: var(--ink-secondary); cursor: pointer;
  font: inherit; font-size: 13px; font-weight: 600; padding: 8px 4px 10px; margin-bottom: -1px;
  border-bottom: 2px solid transparent;
}
.tab-btn.active { color: var(--ink-primary); border-bottom-color: var(--slot-0); }
.pareto-grid { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: 16px; align-items: start; }
.pareto-card { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px 16px 16px; }
.pareto-card h3 { font-size: 14px; margin: 0 0 4px; }
.pareto-subtitle { color: var(--ink-secondary); font-size: 12px; margin: 0 0 12px; }
.pareto-root { position: relative; }
svg.pareto-svg { width: 100%; height: 400px; display: block; overflow: visible; }
.pareto-point { fill: var(--slot-0); stroke: var(--surface); stroke-width: 2; cursor: pointer; }
.pareto-point.selected { fill: var(--slot-1); r: 7; }
.pareto-panel { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 16px; font-size: 13px; }
.pareto-panel .stat-label { color: var(--ink-muted); font-size: 11px; text-transform: uppercase; letter-spacing: .02em; margin: 0 0 2px; }
.pareto-panel .stat-value { font-weight: 600; font-variant-numeric: tabular-nums; }
.pareto-panel .pareto-empty { color: var(--ink-muted); grid-column: 1 / -1; }
"""
_CSS = _CSS.replace("__LIGHT_SLOT_VARS__", _LIGHT_SLOT_VARS).replace("__DARK_SLOT_VARS__", _DARK_SLOT_VARS)

_JS = """
function fmtT(sec) {
  sec = Math.max(0, Math.round(sec));
  const m = Math.floor(sec / 60), s = sec % 60;
  return m + ":" + String(s).padStart(2, "0");
}

function nearestIndex(arr, target) {
  let lo = 0, hi = arr.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (arr[mid] < target) lo = mid + 1; else hi = mid;
  }
  if (lo > 0 && Math.abs(arr[lo - 1] - target) <= Math.abs(arr[lo] - target)) return lo - 1;
  return lo;
}

const STROKE_PATTERNS = ["", "5 4", "1.5 3"];

function initChart(root) {
  const data = JSON.parse(root.dataset.series);
  const svg = root.querySelector("svg.chart");
  const overlay = root.querySelector(".overlay");
  const crosshair = root.querySelector(".crosshair");
  const tooltip = root.querySelector(".tooltip");
  const W = 900, H = 220, padL = 46, padR = 12, padT = 10, padB = 22;
  const plotW = W - padL - padR, plotH = H - padT - padB;

  let tMin = Infinity, tMax = -Infinity, wMax = 0;
  data.forEach(s => {
    s.t.forEach(v => { if (v < tMin) tMin = v; if (v > tMax) tMax = v; });
    s.w.forEach(v => { if (v > wMax) wMax = v; });
  });
  if (!isFinite(tMin)) return;
  wMax = wMax <= 0 ? 1 : wMax * 1.08;

  const x = t => padL + (tMax > tMin ? (t - tMin) / (tMax - tMin) : 0) * plotW;
  const y = w => padT + plotH - (w / wMax) * plotH;

  overlay.setAttribute("x", padL); overlay.setAttribute("y", padT);
  overlay.setAttribute("width", plotW); overlay.setAttribute("height", plotH);
  crosshair.setAttribute("y1", padT); crosshair.setAttribute("y2", padT + plotH);

  overlay.addEventListener("pointermove", (ev) => {
    const rect = svg.getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / rect.width) * W;
    const targetT = tMin + ((px - padL) / plotW) * (tMax - tMin);
    crosshair.style.opacity = 1;
    crosshair.setAttribute("x1", x(targetT));
    crosshair.setAttribute("x2", x(targetT));

    let snappedT = targetT;
    const rows = data.map(s => {
      const idx = nearestIndex(s.t, targetT);
      snappedT = s.t[idx];
      return { label: s.label, slot: s.slot, pattern: s.pattern, value: s.w[idx] };
    });
    tooltip.innerHTML = "";
    const timeEl = document.createElement("div");
    timeEl.className = "t-time";
    timeEl.textContent = fmtT(snappedT);
    tooltip.appendChild(timeEl);
    rows.forEach(r => {
      const row = document.createElement("div");
      row.className = "t-row";
      const key = document.createElement("span");
      key.className = "t-key";
      key.style.background = "var(--slot-" + r.slot + ")";
      if (r.pattern > 0) key.style.backgroundImage =
        "repeating-linear-gradient(90deg, var(--slot-" + r.slot + ") 0 3px, transparent 3px 6px)";
      const label = document.createElement("span");
      label.textContent = r.label;
      const val = document.createElement("span");
      val.className = "t-val";
      val.style.marginLeft = "auto";
      val.textContent = r.value.toFixed(1) + " W";
      row.appendChild(key); row.appendChild(label); row.appendChild(val);
      tooltip.appendChild(row);
    });
    tooltip.style.opacity = 1;
    tooltip.style.left = ((ev.clientX - rect.left)) + "px";
    tooltip.style.top = "0px";
  });
  overlay.addEventListener("pointerleave", () => {
    crosshair.style.opacity = 0;
    tooltip.style.opacity = 0;
  });

  // Static geometry: gridlines, lines, end-dots, end-labels.
  const niceMax = wMax;
  [0, 0.25, 0.5, 0.75, 1].forEach(f => {
    const gy = padT + plotH - f * plotH;
    const gl = document.createElementNS("http://www.w3.org/2000/svg", "line");
    gl.setAttribute("class", "gridline");
    gl.setAttribute("x1", padL); gl.setAttribute("x2", W - padR);
    gl.setAttribute("y1", gy); gl.setAttribute("y2", gy);
    svg.appendChild(gl);
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("class", "axis-text");
    label.setAttribute("x", padL - 6); label.setAttribute("y", gy + 3);
    label.setAttribute("text-anchor", "end");
    label.textContent = Math.round(f * niceMax).toLocaleString();
    svg.appendChild(label);
  });
  [0, 0.5, 1].forEach(f => {
    const gx = padL + f * plotW;
    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("class", "axis-text");
    label.setAttribute("x", gx); label.setAttribute("y", H - 4);
    label.setAttribute("text-anchor", f === 0 ? "start" : f === 1 ? "end" : "middle");
    label.textContent = fmtT(tMin + f * (tMax - tMin));
    svg.appendChild(label);
  });

  data.forEach(s => {
    if (!s.t.length) return;
    const points = s.t.map((t, i) => x(t) + "," + y(s.w[i])).join(" ");
    const poly = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    poly.setAttribute("points", points);
    poly.setAttribute("fill", "none");
    poly.setAttribute("stroke", "var(--slot-" + s.slot + ")");
    poly.setAttribute("stroke-width", "2");
    poly.setAttribute("stroke-linejoin", "round");
    poly.setAttribute("stroke-linecap", "round");
    if (STROKE_PATTERNS[s.pattern]) poly.setAttribute("stroke-dasharray", STROKE_PATTERNS[s.pattern]);
    svg.insertBefore(poly, overlay);

    const lastT = s.t[s.t.length - 1], lastW = s.w[s.w.length - 1];
    const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    dot.setAttribute("cx", x(lastT)); dot.setAttribute("cy", y(lastW)); dot.setAttribute("r", 4);
    dot.setAttribute("fill", "var(--slot-" + s.slot + ")");
    dot.setAttribute("stroke", "var(--surface)"); dot.setAttribute("stroke-width", "2");
    svg.insertBefore(dot, overlay);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("class", "end-label");
    label.setAttribute("x", Math.min(x(lastT) + 6, W - padR - 2));
    label.setAttribute("y", y(lastW) + 3);
    label.textContent = s.label;
    svg.appendChild(label);
  });
}

document.querySelectorAll(".chart-root").forEach(initChart);
document.querySelectorAll(".stats-toggle").forEach(t => {
  t.addEventListener("click", () => {
    const table = t.nextElementSibling;
    table.style.display = table.style.display === "none" ? "" : "none";
  });
});

document.querySelectorAll(".tabs").forEach(tabs => {
  const container = tabs.parentElement;
  tabs.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      tabs.querySelectorAll(".tab-btn").forEach(b => b.classList.toggle("active", b === btn));
      container.querySelectorAll(".tab-panel").forEach(panel => {
        panel.hidden = panel.dataset.tabPanel !== btn.dataset.tab;
      });
    });
  });
});

function initPareto(root) {
  const points = JSON.parse(root.dataset.points);
  if (!points.length) return;
  const svg = root.querySelector("svg.pareto-svg");
  const panel = root.parentElement.querySelector(".pareto-panel");
  const W = 560, H = 400, padL = 50, padR = 16, padT = 12, padB = 30;
  const plotW = W - padL - padR, plotH = H - padT - padB;

  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  points.forEach(p => {
    if (p.x < xMin) xMin = p.x; if (p.x > xMax) xMax = p.x;
    if (p.y < yMin) yMin = p.y; if (p.y > yMax) yMax = p.y;
  });
  const xPad = (xMax - xMin) * 0.1 || 1, yPad = (yMax - yMin) * 0.1 || 1;
  xMin -= xPad; xMax += xPad; yMin = Math.max(0, yMin - yPad); yMax += yPad;

  const x = v => padL + ((v - xMin) / (xMax - xMin)) * plotW;
  const y = v => padT + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

  [0, 0.5, 1].forEach(f => {
    const gy = padT + plotH - f * plotH;
    const gl = document.createElementNS("http://www.w3.org/2000/svg", "line");
    gl.setAttribute("class", "gridline");
    gl.setAttribute("x1", padL); gl.setAttribute("x2", W - padR);
    gl.setAttribute("y1", gy); gl.setAttribute("y2", gy);
    svg.appendChild(gl);
    const gx = padL + f * plotW;
    const xLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
    xLabel.setAttribute("class", "axis-text");
    xLabel.setAttribute("x", gx); xLabel.setAttribute("y", H - 6);
    xLabel.setAttribute("text-anchor", f === 0 ? "start" : f === 1 ? "end" : "middle");
    xLabel.textContent = Math.round(xMin + f * (xMax - xMin)).toLocaleString();
    svg.appendChild(xLabel);
    const yLabel = document.createElementNS("http://www.w3.org/2000/svg", "text");
    yLabel.setAttribute("class", "axis-text");
    yLabel.setAttribute("x", padL - 6); yLabel.setAttribute("y", gy + 3);
    yLabel.setAttribute("text-anchor", "end");
    yLabel.textContent = Math.round(yMin + f * (yMax - yMin)).toLocaleString();
    svg.appendChild(yLabel);
  });

  function renderPanel(p) {
    panel.innerHTML = "";
    p.fields.forEach(([label, value]) => {
      const cell = document.createElement("div");
      const l = document.createElement("p");
      l.className = "stat-label";
      l.textContent = label;
      const v = document.createElement("p");
      v.className = "stat-value";
      v.textContent = value;
      cell.appendChild(l); cell.appendChild(v);
      panel.appendChild(cell);
    });
  }

  const circles = points.map((p, i) => {
    const c = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    c.setAttribute("class", "pareto-point");
    c.setAttribute("cx", x(p.x)); c.setAttribute("cy", y(p.y)); c.setAttribute("r", 5);
    c.addEventListener("click", () => select(i));
    svg.appendChild(c);
    return c;
  });

  function select(i) {
    circles.forEach((c, j) => c.classList.toggle("selected", j === i));
    renderPanel(points[i]);
  }
  select(0);
}

document.querySelectorAll(".pareto-root").forEach(initPareto);
"""


def _fmt(value: float | None, decimals: int = 2) -> str:
    return "n/a" if value is None else f"{value:,.{decimals}f}"


_SUMMARY_TABLE_HEADER = (
    "<tr><th>Run</th><th>Output tok/s</th><th>Tok/s/GPU</th><th>TPOT p50 (ms)</th><th>TPOT p90 (ms)</th>"
    "<th>Avg GPU W</th><th>Avg CPU W</th><th>GPU-only tok/s/W</th><th>CPU-only tok/s/W</th>"
    "<th>Combined tok/s/W</th></tr>"
)


def _summary_rows_html(reports: list[dict], *, run_label: str | None = None) -> str:
    """``<tr>``s for one run's per-concurrency reports.

    ``run_label`` prefixes the Run cell -- set only in the combined multi-directory
    report, where rows from different runs are interleaved in one table and need
    to say which run they came from.
    """
    rows = []
    for r in reports:
        w = r
        ppw = r["perf_per_watt"]
        run_cell = (
            f"{run_label} · {r['benchmark_type']} c={r['concurrency']}"
            if run_label
            else (f"{r['benchmark_type']} c={r['concurrency']}")
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(run_cell)}</td>"
            f"<td>{_fmt(ppw['output_tokens_per_second'])}</td>"
            f"<td>{_fmt(ppw['output_tokens_per_second_per_gpu'])} ({ppw['num_gpus']} gpu)</td>"
            f"<td>{_fmt(w['tpot_p50_ms'])}</td>"
            f"<td>{_fmt(w['tpot_p90_ms'])}</td>"
            f"<td>{_fmt(ppw['gpu_avg_power_w'])}</td>"
            f"<td>{_fmt(ppw['cpu_avg_power_w'])}</td>"
            f"<td>{_fmt(ppw['output_tokens_per_second_per_gpu_watt'], 4)}</td>"
            f"<td>{_fmt(ppw['output_tokens_per_second_per_cpu_watt'], 4)}</td>"
            f"<td>{_fmt(ppw['output_tokens_per_second_per_combined_watt'], 4)}</td>"
            "</tr>"
        )
    return "".join(rows)


def _summary_table_html(reports: list[dict]) -> str:
    return f"<table><thead>{_SUMMARY_TABLE_HEADER}</thead><tbody>{_summary_rows_html(reports)}</tbody></table>"


def _stats_table_html(series: list[dict]) -> str:
    rows = []
    for s in series:
        st = s["stats"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(s['label'])}</td>"
            f"<td>{_fmt(st['mean'])}</td><td>{_fmt(st['min'])}</td><td>{_fmt(st['p50'])}</td>"
            f"<td>{_fmt(st['p95'])}</td><td>{_fmt(st['max'])}</td>"
            "</tr>"
        )
    header = "<tr><th>Device</th><th>Mean W</th><th>Min W</th><th>P50 W</th><th>P95 W</th><th>Max W</th></tr>"
    return f'<table class="stats-table"><thead>{header}</thead><tbody>{"".join(rows)}</tbody></table>'


def _legend_html(series: list[dict]) -> str:
    keys = []
    for s in series:
        style = f"background: var(--slot-{s['slot']})"
        if s["pattern"]:
            style += (
                "; background-image: repeating-linear-gradient("
                f"90deg, var(--slot-{s['slot']}) 0 3px, transparent 3px 6px); background-color: transparent"
            )
        keys.append(
            f'<span class="legend-key"><span class="legend-swatch" style="{style}"></span>{html.escape(s["label"])}</span>'
        )
    return f'<div class="legend">{"".join(keys)}</div>'


def _pareto_points(reports: list[dict], *, run_label: str | None = None) -> list[dict]:
    """Scatter points (output TPS vs. TPS/active GPU) for the Pareto view tab.

    One point per summary-table row, reusing the same fields ``_summary_rows_html``
    renders. Rows with no resolvable GPU count (``output_tokens_per_second_per_gpu``
    is ``None``) can't be plotted and are dropped.
    """
    points = []
    for r in reports:
        ppw = r["perf_per_watt"]
        x, y = ppw["output_tokens_per_second"], ppw["output_tokens_per_second_per_gpu"]
        if x is None or y is None:
            continue
        run_cell = f"{r['benchmark_type']} c={r['concurrency']}"
        label = f"{run_label} · {run_cell}" if run_label else run_cell
        points.append(
            {
                "label": label,
                "x": x,
                "y": y,
                "fields": [
                    ("Concurrency", str(r["concurrency"])),
                    ("Output throughput", f"{_fmt(x)} tok/s"),
                    ("TPS / active GPU", f"{_fmt(y)} ({ppw['num_gpus']} gpu)"),
                    ("P90 TPOT", f"{_fmt(r['tpot_p90_ms'])} ms"),
                    ("Average GPU power", f"{_fmt(ppw['gpu_avg_power_w'])} W"),
                    ("Average CPU power", f"{_fmt(ppw['cpu_avg_power_w'])} W"),
                    ("TPS / GPU watt", _fmt(ppw["output_tokens_per_second_per_gpu_watt"], 4)),
                    ("Profile duration", f"{_fmt(r['timing']['computed']['duration_seconds'])} s"),
                ],
            }
        )
    return points


def _pareto_tab_html(points: list[dict]) -> str:
    points_json = html.escape(json.dumps(points), quote=True)
    return f"""
<div class="pareto-grid">
  <div class="pareto-card">
    <h3>Pareto view</h3>
    <p class="pareto-subtitle">X-axis: output tokens/s. Y-axis: output TPS per active GPU.</p>
    <div class="pareto-root" data-points="{points_json}">
      <svg class="pareto-svg" viewBox="0 0 560 400" preserveAspectRatio="none">
        <rect class="overlay" fill="transparent"></rect>
      </svg>
    </div>
  </div>
  <div class="pareto-card">
    <h3>Selected run</h3>
    <div class="pareto-panel"></div>
  </div>
</div>
"""


def _tabs_html(overview_html: str, pareto_html: str) -> str:
    return f"""
<div class="tabs">
  <button class="tab-btn active" type="button" data-tab="overview">Overview</button>
  <button class="tab-btn" type="button" data-tab="pareto">Pareto view</button>
</div>
<div class="tab-panel" data-tab-panel="overview">{overview_html}</div>
<div class="tab-panel" data-tab-panel="pareto" hidden>{pareto_html}</div>
"""


def _facet_html(facet: dict) -> str:
    series_json = html.escape(json.dumps(facet["series"]), quote=True)
    return f"""
<div class="facet">
  <p class="facet-title">{html.escape(facet["host"])}</p>
  {_legend_html(facet["series"])}
  <div class="chart-root chart-wrap" data-series="{series_json}">
    <svg class="chart" viewBox="0 0 900 220" preserveAspectRatio="none">
      <rect class="overlay" fill="transparent"></rect>
      <line class="crosshair"></line>
    </svg>
    <div class="tooltip"></div>
  </div>
  <span class="stats-toggle">show device stats table</span>
  {_stats_table_html(facet["series"])}
</div>
"""


def _page_html(*, title: str, body_parts: list[str]) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>{_CSS}</style>
</head>
<body>
{"".join(body_parts)}
<script>{_JS}</script>
</body>
</html>
"""


def _sources_footer_html(sources: list[tuple[str, Path | None]]) -> str:
    parts = [f"{label}: <code>{html.escape(str(path))}</code>" for label, path in sources if path is not None]
    return (
        "<footer>Time series are downsampled (min/max per bucket) for display; "
        f"full-resolution samples: {' &middot; '.join(parts) if parts else 'n/a'}.</footer>"
    )


def render_html(
    *,
    run_label: str,
    reports: list[dict],
    gpu_facets: list[dict],
    cpu_facets: list[dict],
    gpu_source: Path | None,
    cpu_source: Path | None,
) -> str:
    header_parts = [
        "<h1>Power &amp; performance report</h1>",
        f'<p class="subtitle">{html.escape(run_label)}</p>',
    ]

    overview_parts = []
    if reports:
        overview_parts.append("<h2>Throughput &amp; power by concurrency</h2>")
        overview_parts.append(_summary_table_html(reports))
    else:
        overview_parts.append("<p>No concurrency-level benchmark windows were found for this run.</p>")

    if gpu_facets:
        overview_parts.append("<h2>GPU power over time</h2>")
        for facet in gpu_facets:
            overview_parts.append(_facet_html(facet))
    if cpu_facets:
        overview_parts.append("<h2>CPU socket power over time</h2>")
        for facet in cpu_facets:
            overview_parts.append(_facet_html(facet))

    overview_parts.append(_sources_footer_html([("GPU", gpu_source), ("CPU", cpu_source)]))

    points = _pareto_points(reports)
    if len(points) >= 2:
        body_parts = [*header_parts, _tabs_html("".join(overview_parts), _pareto_tab_html(points))]
    else:
        body_parts = [*header_parts, *overview_parts]
    return _page_html(title=f"{run_label} — power report", body_parts=body_parts)


def _dedupe_labels(labels: list[str]) -> list[str]:
    """Disambiguate repeated run labels (e.g. two dirs sharing a parent folder name)."""
    seen: dict[str, int] = {}
    out = []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        out.append(label if seen[label] == 1 else f"{label} ({seen[label]})")
    return out


def render_combined_html(bundles: list[dict]) -> str:
    """Combined comparison report for two or more run bundles (see ``_build_run_bundle``).

    One shared throughput/power table (a row per run x concurrency), then each
    run's GPU/CPU power-over-time sections in full, still faceted by node --
    overlaying raw power traces from different runs on one axis isn't meaningful
    since each run covers a different wall-clock window.
    """
    labels = _dedupe_labels([b["label"] for b in bundles])

    header_parts = [
        "<h1>Power &amp; performance report</h1>",
        f'<p class="subtitle">{len(bundles)} runs: {html.escape(", ".join(labels))}</p>',
    ]

    overview_parts = []
    any_reports = any(b["reports"] for b in bundles)
    if any_reports:
        rows = "".join(
            _summary_rows_html(b["reports"], run_label=label) for b, label in zip(bundles, labels, strict=True)
        )
        overview_parts.append("<h2>Throughput &amp; power by concurrency</h2>")
        overview_parts.append(f"<table><thead>{_SUMMARY_TABLE_HEADER}</thead><tbody>{rows}</tbody></table>")
    else:
        overview_parts.append("<p>No concurrency-level benchmark windows were found for any of these runs.</p>")

    sources: list[tuple[str, Path | None]] = []
    for bundle, label in zip(bundles, labels, strict=True):
        if bundle["gpu_facets"]:
            overview_parts.append(f"<h2>GPU power over time — {html.escape(label)}</h2>")
            overview_parts.extend(_facet_html(facet) for facet in bundle["gpu_facets"])
        if bundle["cpu_facets"]:
            overview_parts.append(f"<h2>CPU socket power over time — {html.escape(label)}</h2>")
            overview_parts.extend(_facet_html(facet) for facet in bundle["cpu_facets"])
        sources.append((f"{label} GPU", bundle["gpu_source"]))
        sources.append((f"{label} CPU", bundle["cpu_source"]))

    overview_parts.append(_sources_footer_html(sources))

    points = [
        p
        for bundle, label in zip(bundles, labels, strict=True)
        for p in _pareto_points(bundle["reports"], run_label=label)
    ]
    if len(points) >= 2:
        body_parts = [*header_parts, _tabs_html("".join(overview_parts), _pareto_tab_html(points))]
    else:
        body_parts = [*header_parts, *overview_parts]
    return _page_html(title=f"Power report — {len(bundles)} runs", body_parts=body_parts)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def _discover_power_csvs(
    log_dir: Path, *, cpu_samples_csv: Path | None = None
) -> tuple[Path | None, Path | None, Path | None]:
    """Locate the CPU/GPU ``samples.csv`` legs and the GPU manifest, independent of
    ``power_energy_report.discover_run`` -- that helper also requires a benchmark.out
    with a recognized engine and a matched concurrency window, which a serve-only or
    still-running job may not have. The charts here only need the CSVs themselves.

    Returns ``(cpu_samples_csv, gpu_samples_csv, gpu_manifest)``.
    """
    all_matches = sorted(log_dir.rglob("samples.csv"))
    cpu_matches = [p for p in all_matches if p.parent.name in CPU_SAMPLES_DIRNAMES]
    gpu_matches = [p for p in all_matches if p.parent.name not in CPU_SAMPLES_DIRNAMES]

    if cpu_samples_csv is not None:
        if not cpu_samples_csv.is_file():
            raise PowerReportError(f"{cpu_samples_csv}: not found")
    elif len(cpu_matches) > 1:
        listing = ", ".join(str(p) for p in cpu_matches)
        raise PowerReportError(
            f"multiple CPU power samples.csv found below {log_dir}: {listing}; pick one with --cpu-samples"
        )
    else:
        cpu_samples_csv = cpu_matches[0] if cpu_matches else None

    if len(gpu_matches) > 1:
        raise PowerReportError(f"multiple GPU power samples.csv found below {log_dir}: {gpu_matches}")
    gpu_samples_csv = gpu_matches[0] if gpu_matches else None

    gpu_manifest = None
    if gpu_samples_csv is not None:
        candidate = gpu_samples_csv.with_name("manifest.json")
        gpu_manifest = candidate if candidate.is_file() else None

    return cpu_samples_csv, gpu_samples_csv, gpu_manifest


def _build_run_bundle(log_dir: Path, *, cpu_samples_csv: Path | None = None) -> dict:
    """Everything one directory contributes to a report: its stats, its facets, its
    label. Raises ``PowerReportError`` if the directory has nothing to report.

    Shared by :func:`build_report` (one directory -> one page) and
    :func:`build_combined_report` (several directories -> one comparison page), so
    a single directory's report and its row in a combined report are always built
    from identical data.
    """
    cpu_csv, gpu_csv, gpu_manifest = _discover_power_csvs(log_dir, cpu_samples_csv=cpu_samples_csv)

    reports: list[dict] = []
    try:
        reports = [report_to_dict(r) for r in build_reports(log_dir, cpu_samples_csv=cpu_samples_csv)]
    except PowerReportError as e:
        logger.info("power report: concurrency stats unavailable for %s (%s); charts only", log_dir, e)

    gpu_facets: list[dict] = []
    if gpu_csv is not None:
        roles = load_gpu_roles(gpu_manifest) if gpu_manifest else None
        gpu_samples = load_gpu_samples(gpu_csv, roles)
        gpu_facets = _build_facets(gpu_samples.per_device, label_fmt="gpu{index}")

    cpu_facets: list[dict] = []
    if cpu_csv is not None:
        cpu_samples = load_cpu_samples(cpu_csv)
        cpu_facets = _build_facets(cpu_samples.per_socket, label_fmt="socket{index}")

    if not reports and not gpu_facets and not cpu_facets:
        raise PowerReportError(f"{log_dir}: no benchmark stats or power samples to report")

    return {
        "label": log_dir.parent.name or str(log_dir),
        "reports": reports,
        "gpu_facets": gpu_facets,
        "cpu_facets": cpu_facets,
        "gpu_source": gpu_csv,
        "cpu_source": cpu_csv,
    }


def build_report(log_dir: Path, *, cpu_samples_csv: Path | None = None) -> str:
    """Build the report HTML for ``log_dir``. Raises ``PowerReportError`` if nothing applies."""
    bundle = _build_run_bundle(log_dir, cpu_samples_csv=cpu_samples_csv)
    return render_html(
        run_label=bundle["label"],
        reports=bundle["reports"],
        gpu_facets=bundle["gpu_facets"],
        cpu_facets=bundle["cpu_facets"],
        gpu_source=bundle["gpu_source"],
        cpu_source=bundle["cpu_source"],
    )


def build_combined_report(log_dirs: list[Path], *, cpu_samples_csv: Path | None = None) -> str:
    """Build one comparison report across several run directories.

    A directory with nothing to report is skipped (logged as a warning) rather
    than failing the whole rollup; ``PowerReportError`` is only raised if *none*
    of the directories had anything.
    """
    bundles: list[dict] = []
    skipped: list[str] = []
    for log_dir in log_dirs:
        try:
            bundles.append(_build_run_bundle(log_dir, cpu_samples_csv=cpu_samples_csv))
        except PowerReportError as e:
            logger.warning("power report: skipping %s: %s", log_dir, e)
            skipped.append(str(e))

    if not bundles:
        raise PowerReportError(f"none of the {len(log_dirs)} directories had a report to build: {'; '.join(skipped)}")
    return render_combined_html(bundles)


def build(log_dir: Path, *, cpu_samples_csv: Path | None = None, output_path: Path | None = None) -> Path | None:
    """Write the report next to the run's other power artifacts. Returns the path, or None on failure."""
    try:
        content = build_report(log_dir, cpu_samples_csv=cpu_samples_csv)
    except PowerReportError as e:
        logger.debug("power report: skipped: %s", e)
        return None

    out = output_path or (log_dir / HTML_FILENAME)
    out.write_text(content)
    logger.info("power report: %s", out)
    return out


def build_combined(log_dirs: list[Path], *, output_path: Path, cpu_samples_csv: Path | None = None) -> Path | None:
    """Write a combined comparison report for several run directories. Returns the path, or None on failure."""
    try:
        content = build_combined_report(log_dirs, cpu_samples_csv=cpu_samples_csv)
    except PowerReportError as e:
        logger.debug("power report: combined build skipped: %s", e)
        return None

    output_path.write_text(content)
    logger.info("power report (combined, %d run(s)): %s", len(log_dirs), output_path)
    return output_path


def try_build(runtime: RuntimeContext) -> Path | None:
    """Best-effort entry point for :class:`PostProcessStageMixin`.

    Mirrors :func:`srtctl.analysis.perf_dashboard.try_build`'s contract: never
    raises, so a rendering bug cannot affect the outcome of a benchmark that has
    already produced its results.
    """
    try:
        return build(Path(runtime.log_dir))
    except Exception as e:  # noqa: BLE001 - visualisation is never fatal
        logger.warning("power report: skipped after error: %s", e)
        return None


DEFAULT_COMBINED_FILENAME = "power_report_combined.html"


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "log_dirs",
        type=Path,
        nargs="+",
        metavar="LOG_DIR",
        help=(
            "One or more run logs/ directories. A single directory renders its own "
            "power_report.html; more than one rolls them up into a single "
            "combined comparison report instead."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help=(
            "Output HTML path. Default: <log_dir>/power_report.html for a single "
            f"directory, ./{DEFAULT_COMBINED_FILENAME} for several."
        ),
    )
    parser.add_argument("--cpu-samples", type=Path, help="Explicit CPU samples.csv, see power_energy_report --help")
    args = parser.parse_args(argv)

    try:
        if len(args.log_dirs) == 1:
            path = build(args.log_dirs[0], cpu_samples_csv=args.cpu_samples, output_path=args.output)
        else:
            output_path = args.output or (Path.cwd() / DEFAULT_COMBINED_FILENAME)
            path = build_combined(args.log_dirs, output_path=output_path, cpu_samples_csv=args.cpu_samples)
    except Exception as e:  # noqa: BLE001 - CLI surface, report and exit non-zero
        print(f"error: {e}", file=sys.stderr)
        return 1
    if path is None:
        print("error: no report could be built (see logs)", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
