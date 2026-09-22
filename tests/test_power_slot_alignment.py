# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-slot alignment table over the collector's slot grid."""

from __future__ import annotations

import pytest

from srtctl.core.power.contract import Reason
from srtctl.core.power.samples import SampleRow
from srtctl.core.power.slot_alignment import (
    ANCHOR_CALIBRATED,
    ANCHOR_RECORDED,
    STATUS_LATE,
    STATUS_MISSED,
    STATUS_OK,
    STATUS_UNACCOUNTED,
    compute_slot_alignment,
    render_slot_alignment,
)

START = 1_000.0
INTERVAL = 1.0


def _row(host, seq, t, gpu=0):
    return SampleRow(
        timestamp_unix=t, scrape_seq=seq, hostname=host, gpu_index=gpu, gpu_uuid=f"GPU-{host}-{gpu}", power_w=1.0
    )


def _manifest(*, scrape_count, missed=(), recorded=True):
    m = {
        "sample_interval_seconds": INTERVAL,
        "scrape_count": scrape_count,
        "missed_sample_ranges": [
            {"hostname": h, "first_scrape_seq": a, "last_scrape_seq": b, "reason_codes": [r]} for h, a, b, r in missed
        ],
    }
    if recorded:
        m["slot_grid_started_at_unix"] = START
    else:
        m["started_at_unix"] = START - 25.0  # legacy: session construction, well before slot 0
    return m


def _slow_host_rows(host, latency, slots):
    """Late-fire rule: slot N starts at max(grid, previous settled); row stamp = midpoint."""
    rows = []
    busy = START
    seq = 0
    while seq < slots:
        start = max(START + seq * INTERVAL, busy)
        end = start + latency
        rows.append(_row(host, seq, (start + end) / 2))
        busy = end
        seq += 1
        # forfeit slots that fully elapsed while the request ran
        while START + (seq + 1) * INTERVAL <= end:
            seq += 1
    return rows


class TestClassification:
    def test_on_time_rows_are_ok_with_small_offsets(self):
        rows = [_row("a", s, START + s + 0.1) for s in range(4)]
        al = compute_slot_alignment(rows, _manifest(scrape_count=4))
        assert al is not None
        assert al.slot_count == 4
        assert all(al.cells[s]["a"].status == STATUS_OK for s in range(4))
        assert al.per_host[0].max_offset_seconds == pytest.approx(0.1)
        assert al.per_host[0].late == 0

    def test_late_fire_spill_is_late_not_missed(self):
        # 1.2x latency: slots 0..4 fire, slot 5 is forfeited, slot 6 resumes on-grid.
        rows = _slow_host_rows("b", 1.2, 7) + [_row("a", s, START + s + 0.1) for s in range(7)]
        m = _manifest(scrape_count=7, missed=[("b", 5, 5, Reason.SAMPLE_SCHEDULE_OVERRUN)])
        al = compute_slot_alignment(rows, m)
        assert al is not None
        b = {s: al.cells[s]["b"] for s in range(7)}
        assert [b[s].status for s in range(7)] == [
            STATUS_OK,  # 0.6 offset
            STATUS_OK,  # 1.8 -> offset 0.8
            STATUS_LATE,  # 3.0 -> offset 1.0 (>= interval)
            STATUS_LATE,  # 4.2 -> 1.2
            STATUS_LATE,  # 5.4 -> 1.4
            STATUS_MISSED,
            STATUS_OK,  # resumed on grid
        ]
        assert b[5].reason == "overrun"
        host_b = next(h for h in al.per_host if h.hostname == "b")
        assert host_b.rows == 6 and host_b.late == 3 and host_b.missed == {"overrun": 1} and host_b.unaccounted == 0
        # offsets are bounded by < interval + latency/2 and never accumulate
        assert host_b.max_offset_seconds is not None and host_b.max_offset_seconds < INTERVAL + 1.2 / 2

    def test_hung_bracket_at_stale_seq_leaves_unaccounted_slots(self):
        """PR #480 as written: node-b brackets at seq 2 while node-a is at seq 5; slots 2-4 have no evidence."""
        rows = [_row("a", s, START + s + 0.1) for s in range(5)] + [_row("a", 5, START + 4.6)]
        rows += [_row("b", 0, START + 0.1), _row("b", 2, START + 5.1)]
        m = _manifest(scrape_count=6, missed=[("b", 1, 1, Reason.ENDPOINT_TIMEOUT)])
        al = compute_slot_alignment(rows, m)
        assert al is not None
        assert al.cells[1]["b"].label == "missed:timeout"
        # seq 2 carries a row whose stamp is 3 slots late -> classified late, not unaccounted
        assert al.cells[2]["b"].status == STATUS_LATE
        assert al.cells[3]["b"].status == STATUS_UNACCOUNTED
        assert al.cells[4]["b"].status == STATUS_UNACCOUNTED
        assert al.cells[5]["b"].status == STATUS_UNACCOUNTED
        host_b = next(h for h in al.per_host if h.hostname == "b")
        assert host_b.unaccounted == 3

    def test_clock_derived_bracket_accounts_for_every_slot(self):
        """Prototype: both hosts bracket at B=5, node-b files 2-4 as one overrun range."""
        rows = [_row("a", s, START + s + 0.1) for s in range(5)] + [_row("a", 5, START + 4.6)]
        rows += [_row("b", 0, START + 0.1), _row("b", 5, START + 5.1)]
        m = _manifest(
            scrape_count=6,
            missed=[("b", 1, 1, Reason.ENDPOINT_TIMEOUT), ("b", 2, 4, Reason.SAMPLE_SCHEDULE_OVERRUN)],
        )
        al = compute_slot_alignment(rows, m)
        assert al is not None
        host_b = next(h for h in al.per_host if h.hostname == "b")
        assert host_b.unaccounted == 0
        assert host_b.missed == {"timeout": 1, "overrun": 3}
        assert al.cells[5]["a"].status == STATUS_OK and al.cells[5]["b"].status == STATUS_OK
        # same seq, skew bounded by request latency, not by an interval
        assert host_b.max_skew_seconds is not None and abs(host_b.max_skew_seconds - 0.5) < 1e-9

    def test_manifest_without_interval_yields_none(self):
        rows = [_row("a", 0, START)]
        assert compute_slot_alignment(rows, {"scrape_count": 1}) is None
        assert compute_slot_alignment(rows, {"slot_grid_started_at_unix": START, "sample_interval_seconds": 0}) is None
        assert compute_slot_alignment([], _manifest(scrape_count=1, recorded=False)) is None

    def test_legacy_manifest_calibrates_anchor_from_on_time_rows(self):
        """Pre-slot-grid producers only record session construction time; the grid is recovered from the data."""
        rows = [_row(h, s, START + s + 0.12) for h in ("a", "b") for s in range(6)]
        rows += [_row("c", s, START + s + 0.9) for s in range(6)]  # one persistently late host
        al = compute_slot_alignment(rows, _manifest(scrape_count=6, recorded=False))
        assert al is not None
        assert al.anchor_source == ANCHOR_CALIBRATED
        # median residual is the on-time hosts' 0.12 s, so they read as on-grid and c as late-in-slot
        assert al.started_at_unix == pytest.approx(START + 0.12)
        assert all(al.cells[s]["a"].status == STATUS_OK for s in range(6))
        c = next(h for h in al.per_host if h.hostname == "c")
        assert c.late == 0 and c.max_offset_seconds == pytest.approx(0.78)

    def test_recorded_anchor_wins_over_calibration(self):
        rows = [_row("a", s, START + s + 0.9) for s in range(4)]
        al = compute_slot_alignment(rows, _manifest(scrape_count=4))
        assert al is not None
        assert al.anchor_source == ANCHOR_RECORDED and al.started_at_unix == START
        assert al.per_host[0].max_offset_seconds == pytest.approx(0.9)

    def test_scrape_count_extends_grid_past_last_row(self):
        rows = [_row("a", 0, START)]
        al = compute_slot_alignment(rows, _manifest(scrape_count=3))
        assert al is not None
        assert al.slot_count == 3
        assert al.cells[2]["a"].status == STATUS_UNACCOUNTED


class TestRender:
    def test_folded_table_collapses_all_ok_runs_and_shows_summary(self):
        rows = [_row(h, s, START + s + 0.1) for h in ("a", "b") for s in range(10)]
        rows = [r for r in rows if not (r.hostname == "b" and r.scrape_seq == 4)]
        m = _manifest(scrape_count=10, missed=[("b", 4, 4, Reason.SAMPLE_SCHEDULE_OVERRUN)])
        al = compute_slot_alignment(rows, m)
        assert al is not None
        text = render_slot_alignment(al)
        assert "slots=10 hosts=2" in text
        assert "0–3" in text and "all 2 hosts ok (4 slots)" in text
        assert "5–9" in text and "(5 slots)" in text
        assert "missed:overrun" in text
        # only the defective slot is expanded: host a's ok cell next to host b's miss
        assert text.count("ok +0.10s") == 1
        assert "     4         4.0  ok +0.10s" in text

    def test_full_table_lists_every_slot(self):
        rows = [_row("a", s, START + s) for s in range(3)]
        al = compute_slot_alignment(rows, _manifest(scrape_count=3))
        assert al is not None
        text = render_slot_alignment(al, full=True)
        assert text.count("ok +0.00s") == 3
        assert "all 1 hosts ok" not in text
