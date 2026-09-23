"""Timing diagram for PR #480 scrape_seq semantics (main vs per-endpoint slot grid).

Simulates the two schedulers with deterministic latencies and renders an SVG.
Usage: python pr480_timing_diagram.py <out.html>
"""

import html
import sys
import textwrap

INTERVAL = 1.0
T_END = 6.6
PX = 230
X0 = 250
W = X0 + int(T_END * PX) + 80
LANE_H = 88
WRAP = 112


def sim_old(lat, t_end):
    """Pre-PR: one lock-step cycle; the next cycle starts at max(next_cycle, all settled)."""
    t = 0.0
    seq = 0
    ev = {h: [] for h in lat}
    while t < t_end:
        settle = t + max(lat.values())
        for h, latency in lat.items():
            ev[h].append(dict(seq=seq, start=t, end=t + latency, row=True))
        seq += 1
        t = max(t + INTERVAL, settle)
    return ev, []


def sim_new(lat_fn, t_end, *, late_fire=False, t_stop=None, clock_bracket=False, crashed=()):
    """Per-endpoint slot grid.

    late_fire=False: PR #480 -- any overrun skips int(overrun/interval)+1 slots (slot fires only at its grid time).
    late_fire=True : prototype -- slot N fires any time before slot N+1 is due; only fully elapsed slots are skipped.
    t_stop         : if set, stop fires at this instant and every thread ends with one bracket poll.
    clock_bracket  : False -- PR #480: bracket at whatever seq the thread holds (in-loop skip suppressed once _stop is set).
                     True  -- prototype: bracket at B = floor(t_stop/interval)+1, filling counter..B-1 as a missed range.
    crashed        : hosts whose thread raised at t_stop -- they set _stop but take no bracket poll themselves.
                     Their last event is marked crash=True.
    """
    ev = {}
    missed = []
    bracket_seq = int(t_stop / INTERVAL) + 1 if (t_stop is not None and clock_bracket) else None
    for h, fn in lat_fn.items():
        seq = 0
        nxt = 0.0
        ev[h] = []
        busy_until = 0.0
        while nxt < t_end:
            if t_stop is not None and nxt >= t_stop:
                break
            start = max(nxt, busy_until) if late_fire else nxt
            latency, ok = fn(seq)
            end = start + latency
            busy_until = end
            ev[h].append(dict(seq=seq, start=start, end=end, row=ok))
            if not ok:
                missed.append((h, seq, seq, "endpoint_timeout"))
            seq += 1
            nxt += INTERVAL
            if t_stop is not None and end >= t_stop:
                break  # stop fired while this request was in flight; skip block is suppressed
            if late_fire:
                if nxt + INTERVAL <= end:
                    n = int((end - nxt) / INTERVAL)
                    missed.append((h, seq, seq + n - 1, "sample_schedule_overrun"))
                    seq += n
                    nxt += n * INTERVAL
            elif nxt <= end:
                n = int((end - nxt) / INTERVAL) + 1
                missed.append((h, seq, seq + n - 1, "sample_schedule_overrun"))
                seq += n
                nxt += n * INTERVAL
        if h in crashed:
            if ev[h]:
                ev[h][-1]["crash"] = True
            continue
        if t_stop is not None:
            b = bracket_seq if bracket_seq is not None else seq
            if b > seq:
                missed.append((h, seq, b - 1, "sample_schedule_overrun"))
            start = max(t_stop, busy_until)
            latency, ok = fn(b)
            ev[h].append(dict(seq=b, start=start, end=start + latency, row=ok, bracket=True))
    return ev, missed


def coalesce(m):
    return ", ".join(f"{h} seq {a}" + (f"–{b}" if b != a else "") + f" ({r})" for h, a, b, r in m)


def _host_ranges(missed, host):
    """Merge one host's missed entries into contiguous (first, last, reason) spans for drawing."""
    spans = sorted((a, b, r) for h, a, b, r in missed if h == host)
    out = []
    for a, b, r in spans:
        if out and out[-1][2] == r and out[-1][1] + 1 == a:
            out[-1] = (out[-1][0], b, r)
        else:
            out.append((a, b, r))
    return out


def rows_in(ev, host):
    return sum(e["row"] and e["end"] <= T_END for e in ev[host])


def wrap(lines, width=WRAP):
    out = []
    for n in lines:
        out.extend(textwrap.wrap(n, width))
    return out


def panel(title, subtitle, ev, missed, y0, notes, hosts=("node-a", "node-b"), t_stop=None, stop_label=None):
    k = LANE_H / 66  # vertical scale relative to the original 66 px lane
    lh = round(21 * k)  # line height for sub/note text
    out = [f'<text x="{X0}" y="{y0 + round(24 * k)}" class="title">{html.escape(title)}</text>']
    sub = wrap([subtitle])
    for i, s in enumerate(sub):
        out.append(f'<text x="{X0}" y="{y0 + round(48 * k) + i * lh}" class="sub">{html.escape(s)}</text>')
    gy = y0 + round(70 * k) + lh * (len(sub) - 1) + (round(26 * k) if t_stop is not None else 0)
    lane_h = LANE_H
    H = lane_h * len(hosts)
    gap_y = gy + H + round(14 * k)
    tick_y = gy + H + round(50 * k)
    for kk in range(int(T_END) + 1):
        x = X0 + kk * PX
        out.append(f'<line x1="{x}" y1="{gy}" x2="{x}" y2="{gy + H}" class="grid"/>')
        out.append(f'<text x="{x}" y="{tick_y}" class="tick" text-anchor="middle">{kk}s</text>')
    out.append(
        f'<text x="{X0 + int(T_END * PX)}" y="{tick_y + lh}" class="tick" text-anchor="end">'
        f"head-node clock (interval = {INTERVAL:g} s)</text>"
    )
    if t_stop is not None:
        xs = X0 + t_stop * PX
        label = stop_label or f"stop @ {t_stop:g} s → bracket seq B = {int(t_stop / INTERVAL) + 1}"
        out.append(f'<line x1="{xs:.1f}" y1="{gy - 6}" x2="{xs:.1f}" y2="{gy + H + 4}" class="stop"/>')
        out.append(f'<text x="{xs + 6:.1f}" y="{gy - 10}" class="stop-lbl">{html.escape(label)}</text>')
    for i, h in enumerate(hosts):
        ly = gy + i * lane_h
        out.append(f'<text x="{X0 - 14}" y="{ly + lane_h / 2 + 6}" class="host" text-anchor="end">{h}</text>')
        for e in ev[h]:
            x1 = X0 + e["start"] * PX
            x2 = X0 + min(e["end"], T_END) * PX
            cls = "req" if e["row"] else "req-fail"
            if e.get("bracket"):
                cls += " bracket"
            if e.get("crash"):
                cls += " crash"
            out.append(
                f'<rect x="{x1:.1f}" y="{ly + round(24 * k)}" width="{max(3, x2 - x1):.1f}" height="{lane_h - round(36 * k)}" rx="4" class="{cls}"/>'
            )
            label = f"seq {e['seq']}" + (" (B)" if e.get("bracket") else "") + (" ✗ raised" if e.get("crash") else "")
            out.append(f'<text x="{x1:.1f}" y="{ly + round(18 * k)}" class="seq">{label}</text>')
            if e["row"] and e["end"] <= T_END and not e.get("crash"):
                mid = X0 + (e["start"] + e["end"]) / 2 * PX
                out.append(f'<line x1="{mid:.1f}" y1="{ly + round(20 * k)}" x2="{mid:.1f}" y2="{ly + lane_h - round(8 * k)}" class="row"/>')
        # Missed-slot boxes are drawn after the bars so a long in-flight request cannot hide them.
        # Contiguous same-cause slots get one box and one label; per-slot labels only when the slot is wide enough.
        for a, b, r in _host_ranges(missed, h):
            xa = X0 + a * PX
            xb = X0 + min(b + 1, T_END) * PX
            if xa >= X0 + T_END * PX:
                continue
            cls = "missed-overrun" if r == "sample_schedule_overrun" else "missed-timeout"
            out.append(f'<rect x="{xa + 1}" y="{ly + round(21 * k)}" width="{xb - xa - 2:.1f}" height="{lane_h - round(30 * k)}" class="{cls}"/>')
            if PX >= 150:
                for s in range(a, b + 1):
                    out.append(
                        f'<text x="{X0 + s * PX + PX / 2}" y="{ly + lane_h / 2 + round(10 * k)}" class="miss-lbl" text-anchor="middle">'
                        f"seq {s} · missed</text>"
                    )
            else:
                span = f"seq {a}" if a == b else f"seq {a}–{b}"
                out.append(
                    f'<text x="{(xa + xb) / 2:.1f}" y="{ly + lane_h / 2 + round(10 * k)}" class="miss-lbl" text-anchor="middle">'
                    f"{span} · missed</text>"
                )
    rows = sorted((e["start"] + e["end"]) / 2 for e in ev[hosts[-1]] if e["row"] and e["end"] <= T_END and not e.get("crash"))
    if len(rows) > 1:
        g, a, b = max((b - a, a, b) for a, b in zip(rows, rows[1:]))
        out.append(
            f'<line x1="{X0 + a * PX:.1f}" y1="{gap_y}" x2="{X0 + b * PX:.1f}" y2="{gap_y}" class="gap" '
            'marker-start="url(#tick)" marker-end="url(#tick)"/>'
        )
        out.append(
            f'<text x="{X0 + (a + b) / 2 * PX:.1f}" y="{gap_y + round(19 * k)}" class="gap-lbl" text-anchor="middle">'
            f"largest {hosts[-1]} row gap = {g:.1f} s (row timestamps = request midpoints)</text>"
        )
    ny = tick_y + round(52 * k)
    for n in wrap(notes):
        out.append(f'<text x="{X0}" y="{ny}" class="note">{html.escape(n)}</text>')
        ny += lh + 1
    return "\n".join(out), ny + round(18 * k)


STYLE = """
 text{fill:var(--foreground);font-size:23px;font-family:-apple-system,"SF Pro Text","Helvetica Neue",Helvetica,Arial,sans-serif}
 .title{font-size:28px;font-weight:600}
 .sub,.note,.tick{fill:var(--muted-foreground);font-size:21px}
 .host{font-size:23px;font-weight:600}
 .seq{font-size:17px;fill:var(--foreground)}
 .grid{stroke:var(--border);stroke-width:2}
 .req{fill:var(--accent);fill-opacity:.85;stroke:var(--foreground);stroke-opacity:.35}
 .req-fail{fill:none;stroke:#e5484d;stroke-width:2.5;stroke-dasharray:8 5}
 .row{stroke:var(--foreground);stroke-width:4}
 .missed-overrun{fill:#e5a100;fill-opacity:.18;stroke:#e5a100;stroke-width:2;stroke-dasharray:5 5}
 .missed-timeout{fill:#e5484d;fill-opacity:.15;stroke:#e5484d;stroke-width:2;stroke-dasharray:5 5}
 .miss-lbl{font-size:17px;fill:#f0f0f0;paint-order:stroke;stroke:#0f1115;stroke-width:4px;stroke-linejoin:round}
 .gap{stroke:var(--foreground);stroke-width:2}
 .gap-lbl{font-size:20px}
 .stop{stroke:#ff6b6b;stroke-width:3;stroke-dasharray:10 6}
 .stop-lbl{font-size:19px;fill:#ff6b6b;font-weight:600}
 .bracket{stroke:#7ee787;stroke-width:3;stroke-opacity:1}
 .crash{fill:#ff6b6b;fill-opacity:.55;stroke:#ff6b6b;stroke-width:3;stroke-opacity:1}
"""


def _largest_gap(ev, host):
    rows = sorted((e["start"] + e["end"]) / 2 for e in ev[host] if e["row"] and e["end"] <= T_END)
    return max((b - a for a, b in zip(rows, rows[1:])), default=0.0)


def build():
    fast = lambda s: (0.2, True)  # noqa: E731
    slow = lambda s: (1.2, True)  # noqa: E731
    hang1 = lambda s: (4.0, False) if s == 1 else (0.2, True)  # noqa: E731
    T_STOP = 4.5  # lands while node-b's 1.0-5.0 s hang is still in flight
    B = int(T_STOP / INTERVAL) + 1

    ev_b_pr, m_b_pr = sim_new({"node-a": fast, "node-b": slow}, T_END)
    ev_b_lf, m_b_lf = sim_new({"node-a": fast, "node-b": slow}, T_END, late_fire=True)
    ev_c_pr, m_c_pr = sim_new({"node-a": fast, "node-b": hang1}, T_END, t_stop=T_STOP)
    ev_c_lf, m_c_lf = sim_new({"node-a": fast, "node-b": hang1}, T_END, late_fire=True, t_stop=T_STOP, clock_bracket=True)

    def bracket(ev, host):
        e = next(e for e in ev[host] if e.get("bracket"))
        return e["seq"], (e["start"] + e["end"]) / 2

    ba_pr, bb_pr = bracket(ev_c_pr, "node-a"), bracket(ev_c_pr, "node-b")
    ba_lf, bb_lf = bracket(ev_c_lf, "node-a"), bracket(ev_c_lf, "node-b")

    body = []
    y = 10
    p, y = panel(
        "B1 · PR #480 as written — node-b answers in 1.2 s (20 % over the interval)",
        "Skip rule: after a poll, if next_cycle ≤ now → forfeit int(overrun/interval)+1 slots. A slot fires only exactly at its grid time.",
        ev_b_pr, m_b_pr, y,
        [
            f"node-b keeps {rows_in(ev_b_pr, 'node-b')} of node-a's {rows_in(ev_b_pr, 'node-a')} slots (50 %). Largest node-b row gap {_largest_gap(ev_b_pr, 'node-b'):.1f} s.",
            f"missed_sample_ranges: {coalesce(m_b_pr)}.",
        ],
    )
    body.append(p)
    p, y = panel(
        "B2 · Prototype (late-fire) — same 1.2 s node-b",
        "Skip rule: forfeit only if next_cycle + interval ≤ now → int(overrun/interval) slots. Slot N may start any time before slot N+1 is due.",
        ev_b_lf, m_b_lf, y,
        [
            f"node-b keeps {rows_in(ev_b_lf, 'node-b')} of {rows_in(ev_b_lf, 'node-a')} slots (physical ceiling 1/1.2 = 83 %). Largest row gap {_largest_gap(ev_b_lf, 'node-b'):.1f} s = the request latency.",
            "seq is still the slot index (start + seq·interval); node-b's row for seq N is simply taken later inside slot N — bounded by < 1 interval, re-synced on every skip (sawtooth, no drift).",
            f"missed_sample_ranges: {coalesce(m_b_lf)}.",
        ],
        )
    body.append(p)
    p, y = panel(
        f"C1 · PR #480 as written — node-b's seq 1 hangs 4 s (2 × timeout); stop fires at {T_STOP:g} s mid-flight",
        "Stop sets _stop; each thread leaves its loop and does one bracket poll at whatever seq it holds. The in-loop skip block is guarded by `not _stop.is_set()`.",
        ev_c_pr, m_c_pr, y,
        [
            f"node-a was idle waiting for slot {ba_pr[0]} → brackets at seq {ba_pr[0]} (t ≈ {ba_pr[1]:.1f}). node-b's hung request settles after stop → skip block suppressed → brackets at seq {bb_pr[0]} at t ≈ {bb_pr[1]:.1f}: {ba_pr[0] - bb_pr[0]} slots behind, same wall-clock.",
            f"Slots {bb_pr[0]}–{ba_pr[0] - 1} on node-b are never recorded: missed_sample_ranges = {coalesce(m_c_pr)}. The undercount is silent — the validator only checks per-host consistency.",
        ],
        t_stop=T_STOP,
    )
    body.append(p)
    p, y = panel(
        f"C2 · Prototype — same hang, bracket seq derived from the clock: B = floor(t_stop / interval) + 1 = {B}",
        "Computed once in stop_and_finalize. Each thread: if B > its counter → record counter…B−1 as sample_schedule_overrun; then poll seq B with scheduled_at = grid time of B.",
        ev_c_lf, m_c_lf, y,
        [
            f"Both hosts bracket at seq {B}. node-b fills the gap as one coalesced range: {coalesce(m_c_lf)}.",
            "Every thread's counter is ≤ S+1 at stop (it cannot have polled a slot that has not started) and an idle thread holds exactly S+1 — so B never collides with an existing row and never goes backwards.",
            f"Bracket row timestamps: node-a ≈ {ba_lf[1]:.1f} (fires at stop), node-b ≈ {bb_lf[1]:.1f} (fires when its hung request settles). Same seq, skew bounded by one request latency; the validator only needs a row with t ≥ window end.",
        ],
        t_stop=T_STOP,
    )
    body.append(p)
    return _wrap_svg(body, y, bracket_legend=True)


def _wrap_svg(body, y, *, bracket_legend: bool, crash_legend: bool = False) -> str:
    rows = [
        ('<rect x="0" y="{y}" width="74" height="24" rx="5" class="req"/>', "request in flight; row timestamp = midpoint (┃)"),
        ('<rect x="0" y="{y}" width="74" height="24" rx="5" class="req-fail"/>', "request failed / timed out — no row (endpoint_timeout)"),
        ('<rect x="0" y="{y}" width="74" height="24" class="missed-overrun"/>', "slot forfeited — sample_schedule_overrun"),
    ]
    if bracket_legend:
        rows.append(('<rect x="0" y="{y}" width="74" height="24" rx="5" class="req bracket"/>', 'bracket poll after stop (seq marked "(B)")'))
    if crash_legend:
        rows.append(('<rect x="0" y="{y}" width="74" height="24" rx="5" class="req crash"/>', "request whose thread raised — sets _stop, no bracket poll"))
    rows.append(('<line x1="0" y1="{yl}" x2="74" y2="{yl}" class="stop"/>', "_stop set"))
    parts = []
    for i, (shape, text) in enumerate(rows):
        yy = i * 40
        parts.append(shape.format(y=yy, yl=yy + 12) + f'<text x="88" y="{yy + 20}" class="note">{text}</text>')
    legend = f'<g transform="translate({X0},{y})">' + "".join(parts) + "</g>"
    h_total = y + 40 * len(rows) + 30
    width = X0 + int(T_END * PX) + 80
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{h_total}" viewBox="0 0 {width} {h_total}">
<defs><marker id="tick" markerWidth="3" markerHeight="12" refX="1.5" refY="6" orient="auto"><line x1="1.5" y1="0" x2="1.5" y2="12" stroke="var(--foreground)" stroke-width="2"/></marker></defs>
<style>svg{{--foreground:#e6e6e6;--muted-foreground:#a0a4ab;--accent:#4c8dff;--border:#3a3f47}}{STYLE}</style>
<rect width="100%" height="100%" fill="#0f1115"/>
{"".join(body)}
{legend}
</svg>"""


def build_crash():
    """Endpoint-thread crash while a sibling is mid-hang: 104f4bcd vs _request_stop."""
    fast = lambda s: (0.2, True)  # noqa: E731
    # node-b hangs on slot 2 for 8 intervals (settles at 10.2); node-a raises on slot 6.
    hang2 = lambda s: (8.2, True) if s == 2 else (0.2, True)  # noqa: E731
    crash6 = lambda s: (0.2, True)  # noqa: E731
    T_CRASH = 6.1  # node-a's slot-6 request raises 0.1 s in
    B = int(T_CRASH / INTERVAL) + 1

    global T_END, PX
    saved = (T_END, PX)
    T_END = 12.6
    PX = int(saved[1] * saved[0] / T_END)  # keep the figure the same overall width as the main diagram
    try:
        ev_v3, m_v3 = sim_new({"node-a": crash6, "node-b": hang2}, T_END, late_fire=True, t_stop=T_CRASH, crashed={"node-a"})
        ev_fx, m_fx = sim_new(
            {"node-a": crash6, "node-b": hang2}, T_END, late_fire=True, t_stop=T_CRASH, clock_bracket=True, crashed={"node-a"}
        )
        b_v3 = next(e for e in ev_v3["node-b"] if e.get("bracket"))
        b_fx = next(e for e in ev_fx["node-b"] if e.get("bracket"))
        body = []
        y = 10
        p, y = panel(
            "D1 · 104f4bcd — node-a's slot-6 request raises while node-b is mid-hang on slot 2",
            "The endpoint except block calls _stop.set() directly (L553). stop_and_finalize has not run, so _shutdown_bracket is still None when node-b's hang settles.",
            ev_v3, m_v3, y,
            [
                "node-a's thread is gone after the raise: no bracket poll from it, its last slot is 5 — correct, and COLLECTOR_EXCEPTION says why.",
                f"node-b leaves its loop, reads _shutdown_bracket → None → takes the fallback (L524-526): brackets at its own counter, seq {b_v3['seq']}, at t ≈ {(b_v3['start'] + b_v3['end']) / 2:.1f} s.",
                f"Slots 3–{B - 1} on node-b have no row and no missed range: missed_sample_ranges = {coalesce(m_v3) or '(empty)'}. The manifest reads as if node-b was fine until it stopped at slot {b_v3['seq']}.",
            ],
            t_stop=T_CRASH,
            stop_label=f"node-a raises @ {T_CRASH:g} s → _stop set, bracket NOT armed",
        )
        body.append(p)
        p, y = panel(
            f"D2 · with _request_stop — same crash, every _stop path arms the bracket first: B = floor({T_CRASH:g}/1)+1 = {B}",
            "The except block calls _request_stop(now) instead of _stop.set(); it computes the shared bracket under the state lock, then sets _stop. The None fallback becomes unreachable and raises.",
            ev_fx, m_fx, y,
            [
                f"node-b now sees B = {B} on exit: files 3–{B - 1} as one sample_schedule_overrun range, then polls seq {B} at t ≈ {(b_fx['start'] + b_fx['end']) / 2:.1f} s.",
                f"missed_sample_ranges = {coalesce(m_fx)}. Every node-b slot up to B is a row or a range; the only unexplained gap is node-a's, and COLLECTOR_EXCEPTION is on the manifest.",
                "publication_valid is False in both panels (COLLECTOR_EXCEPTION is fatal). The difference is whether the manifest describes the outage or hides it.",
            ],
            t_stop=T_CRASH,
            stop_label=f"node-a raises @ {T_CRASH:g} s → _request_stop → B = {B}",
        )
        body.append(p)
        return _wrap_svg(body, y, bracket_legend=True, crash_legend=True)
    finally:
        T_END, PX = saved


if __name__ == "__main__":
    import re

    out = sys.argv[1]
    scenario = sys.argv[2] if len(sys.argv) > 2 else "main"
    svg = {"main": build, "crash": build_crash}[scenario]()
    m = re.search(r'width="(\d+)" height="(\d+)"', svg)
    # Inline-preview variant: the SVG scales to the frame width and stays vector-crisp.
    fluid = svg.replace(f'width="{m.group(1)}" height="{m.group(2)}"', 'width="100%"', 1)
    dark = (
        "<!doctype html><html><body style='margin:0;background:#0f1115;--foreground:#e6e6e6;"
        f"--muted-foreground:#a0a4ab;--accent:#4c8dff;--border:#3a3f47'>{fluid}</body></html>"
    )
    with open(out, "w") as f:
        f.write(dark)
    with open(out.replace(".html", ".svg"), "w") as f:
        f.write(svg)
    print(out, m.group(1), m.group(2))
