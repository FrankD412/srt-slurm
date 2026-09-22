# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-slot, per-host alignment of persisted power samples against the collector's slot grid.

The collector schedules slot ``N`` on every endpoint at
``started_at_unix + N * sample_interval_seconds`` and stamps the rows it
produces with ``scrape_seq == N``. Both anchors live in the manifest, so the
grid can be rebuilt offline without a schema change to ``samples.csv``.

For every slot and host this module classifies the outcome:

``ok``
    Rows exist and the row timestamp (request midpoint) fell inside the slot.
``late``
    Rows exist but the request ran long enough that its midpoint spilled past
    the slot's end. Under the late-fire rule this is the expected shape for a
    persistently slow endpoint.
``missed:<reason>``
    No rows; the manifest's ``missed_sample_ranges`` accounts for the slot.
``unaccounted``
    No rows and no missed range. The collector lost track of the slot -- the
    shape a hung request across shutdown produces when the bracket poll reuses
    a stale sequence number.

Nothing here feeds ``publication_valid``; coverage is judged on timestamps in
:mod:`srtctl.core.power.windows`. This is reviewer evidence for *why* a gap
exists and whether the manifest's accounting is complete.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from srtctl.core.power.contract import Reason
from srtctl.core.power.samples import SampleRow

STATUS_OK = "ok"
STATUS_LATE = "late"
STATUS_MISSED = "missed"
STATUS_UNACCOUNTED = "unaccounted"

ANCHOR_RECORDED = "recorded"  # manifest.slot_grid_started_at_unix
ANCHOR_CALIBRATED = "calibrated"  # median residual of on-disk rows (legacy manifests)

_REASON_ABBREV = {
    Reason.SAMPLE_SCHEDULE_OVERRUN: "overrun",
    Reason.ENDPOINT_TIMEOUT: "timeout",
    Reason.ENDPOINT_HTTP_ERROR: "http",
    Reason.ENDPOINT_PARSE_ERROR: "parse",
}


@dataclass(frozen=True)
class SlotCell:
    """One host's outcome for one slot."""

    status: str
    offset_seconds: float | None = None  # row timestamp minus slot start; None when no rows
    reason: str | None = None  # abbreviated cause for a missed slot

    @property
    def label(self) -> str:
        if self.status == STATUS_MISSED:
            return f"{STATUS_MISSED}:{self.reason}"
        return self.status


@dataclass(frozen=True)
class HostAlignment:
    hostname: str
    rows: int
    late: int
    missed: dict[str, int]
    unaccounted: int
    max_offset_seconds: float | None
    max_skew_seconds: float | None  # max |t_host - t_reference| over shared slots

    def to_dict(self) -> dict[str, Any]:
        return {
            "hostname": self.hostname,
            "rows": self.rows,
            "late": self.late,
            "missed": dict(self.missed),
            "unaccounted": self.unaccounted,
            "max_offset_seconds": self.max_offset_seconds,
            "max_skew_seconds": self.max_skew_seconds,
        }


@dataclass(frozen=True)
class SlotAlignment:
    started_at_unix: float
    anchor_source: str  # ANCHOR_RECORDED | ANCHOR_CALIBRATED
    interval_seconds: float
    hosts: tuple[str, ...]
    slot_count: int
    cells: dict[int, dict[str, SlotCell]] = field(repr=False)
    per_host: tuple[HostAlignment, ...]
    reference_host: str | None

    def scheduled_at(self, seq: int) -> float:
        return self.started_at_unix + seq * self.interval_seconds

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at_unix": self.started_at_unix,
            "anchor_source": self.anchor_source,
            "interval_seconds": self.interval_seconds,
            "slot_count": self.slot_count,
            "reference_host": self.reference_host,
            "hosts": [host.to_dict() for host in self.per_host],
        }


def compute_slot_alignment(rows: Sequence[SampleRow], manifest: Mapping[str, Any]) -> SlotAlignment | None:
    """Bucket every persisted row and missed range onto the manifest's slot grid.

    The grid anchor is ``slot_grid_started_at_unix`` when the producer recorded
    it. Older manifests only carry ``started_at_unix`` (session construction,
    which precedes the first poll by exporter startup time), so for those the
    anchor is *calibrated* from the data: the median of ``t_row - seq*interval``
    over every on-grid host, which makes the table read "relative to the grid
    the collector actually ran" rather than reporting every slot as late.
    ``anchor_source`` says which happened.

    Returns ``None`` when the manifest lacks a usable interval or there are no
    samples, so callers can skip the table instead of inventing one.
    """
    interval = manifest.get("sample_interval_seconds")
    if not _positive_number(interval):
        return None
    assert isinstance(interval, (int, float))  # narrowed by _positive_number
    interval = float(interval)

    # One timestamp per (host, seq): every GPU row from one scrape shares it, but
    # take the min defensively so a mixed file cannot inflate the offset.
    stamp: dict[str, dict[int, float]] = {}
    for row in rows:
        per_host = stamp.setdefault(row.hostname, {})
        per_host[row.scrape_seq] = min(per_host.get(row.scrape_seq, row.timestamp_unix), row.timestamp_unix)

    recorded = manifest.get("slot_grid_started_at_unix")
    if isinstance(recorded, (int, float)) and not isinstance(recorded, bool):
        started, anchor_source = float(recorded), ANCHOR_RECORDED
    else:
        started = _calibrate_anchor(stamp, interval)
        if started is None:
            return None
        anchor_source = ANCHOR_CALIBRATED

    missed: dict[str, dict[int, str]] = {}
    for entry in _missed_ranges(manifest):
        host_missed = missed.setdefault(entry["hostname"], {})
        for seq in range(entry["first_scrape_seq"], entry["last_scrape_seq"] + 1):
            host_missed[seq] = entry["reason"]

    hosts = tuple(sorted(set(stamp) | set(missed)))
    if not hosts:
        return None
    last_seq = max(
        max((seq for per_host in stamp.values() for seq in per_host), default=-1),
        max((seq for per_host in missed.values() for seq in per_host), default=-1),
    )
    scrape_count = manifest.get("scrape_count")
    slot_count = max(
        last_seq + 1, scrape_count if isinstance(scrape_count, int) and not isinstance(scrape_count, bool) else 0
    )

    cells: dict[int, dict[str, SlotCell]] = {}
    for seq in range(slot_count):
        slot_start = started + seq * interval
        per_slot: dict[str, SlotCell] = {}
        for host in hosts:
            t = stamp.get(host, {}).get(seq)
            if t is not None:
                offset = t - slot_start
                per_slot[host] = SlotCell(STATUS_LATE if offset >= interval else STATUS_OK, offset_seconds=offset)
            elif seq in missed.get(host, {}):
                per_slot[host] = SlotCell(STATUS_MISSED, reason=missed[host][seq])
            else:
                per_slot[host] = SlotCell(STATUS_UNACCOUNTED)
        cells[seq] = per_slot

    # Reference host for skew: the one with the most rows (ties -> first by name).
    reference = max(hosts, key=lambda h: (len(stamp.get(h, {})), -hosts.index(h))) if stamp else None
    per_host = tuple(_summarise_host(host, cells, stamp, reference) for host in hosts)
    return SlotAlignment(
        started_at_unix=started,
        anchor_source=anchor_source,
        interval_seconds=interval,
        hosts=hosts,
        slot_count=slot_count,
        cells=cells,
        per_host=per_host,
        reference_host=reference,
    )


def _calibrate_anchor(stamp: Mapping[str, Mapping[int, float]], interval: float) -> float | None:
    """Median of ``t - seq*interval`` across every (host, seq) row: the grid the collector actually ran.

    A slow host's late rows pull the estimate upward, but the median is robust
    to that as long as most rows on most hosts are on time.
    """
    residuals = sorted(t - seq * interval for per_host in stamp.values() for seq, t in per_host.items())
    if not residuals:
        return None
    mid = len(residuals) // 2
    return residuals[mid] if len(residuals) % 2 else (residuals[mid - 1] + residuals[mid]) / 2


def render_slot_alignment(alignment: SlotAlignment, *, full: bool = False) -> str:
    """Text table: per-host summary, then one line per slot with all-``ok`` runs folded.

    ``full`` prints every slot instead of folding.
    """
    lines = [
        f"slot grid: start={alignment.started_at_unix:.3f} ({alignment.anchor_source}) "
        f"interval={alignment.interval_seconds:g}s slots={alignment.slot_count} hosts={len(alignment.hosts)}"
        + (f" skew_reference={alignment.reference_host}" if alignment.reference_host else ""),
    ]
    name_w = max(len(h) for h in alignment.hosts)
    lines.append(f"  {'host':<{name_w}}  rows  late  {'missed':<24} unacc  max_offset  max_skew")
    for host in alignment.per_host:
        missed = ", ".join(f"{k}={v}" for k, v in sorted(host.missed.items())) or "-"
        lines.append(
            f"  {host.hostname:<{name_w}}  {host.rows:>4}  {host.late:>4}  {missed:<24} {host.unaccounted:>5}  "
            f"{_fmt(host.max_offset_seconds):>10}  {_fmt(host.max_skew_seconds):>8}"
        )

    lines.append("")
    col_w = max(name_w, 16)
    # Enough decimals to tell adjacent slots apart at sub-second cadences.
    decimals = 1 if alignment.interval_seconds >= 1 else max(1, -math.floor(math.log10(alignment.interval_seconds)) + 1)
    header = f"  {'seq':>6}  {'sched(+s)':>10}  " + "  ".join(f"{h:<{col_w}}" for h in alignment.hosts)
    lines.append(header)
    for seq_a, seq_b in _fold_runs(alignment, full=full):
        if seq_b > seq_a:
            span = f"{seq_a}–{seq_b}"
            sched = f"{seq_a * alignment.interval_seconds:.{decimals}f}…"
            status = alignment.cells[seq_a][alignment.hosts[0]].label
            body = f"all {len(alignment.hosts)} hosts {status} ({seq_b - seq_a + 1} slots)"
            lines.append(f"  {span:>6}  {sched:>10}  {body}")
            continue
        per_slot = alignment.cells[seq_a]
        body = "  ".join(f"{_cell_text(per_slot[h]):<{col_w}}" for h in alignment.hosts)
        lines.append(f"  {seq_a:>6}  {seq_a * alignment.interval_seconds:>10.{decimals}f}  {body}")
    return "\n".join(lines)


def _summarise_host(
    host: str,
    cells: Mapping[int, Mapping[str, SlotCell]],
    stamp: Mapping[str, Mapping[int, float]],
    reference: str | None,
) -> HostAlignment:
    missed: dict[str, int] = {}
    late = 0
    unaccounted = 0
    offsets: list[float] = []
    for per_slot in cells.values():
        cell = per_slot[host]
        if cell.status == STATUS_MISSED:
            missed[cell.reason or "?"] = missed.get(cell.reason or "?", 0) + 1
        elif cell.status == STATUS_UNACCOUNTED:
            unaccounted += 1
        else:
            offsets.append(cell.offset_seconds or 0.0)
            if cell.status == STATUS_LATE:
                late += 1
    skew: float | None = None
    if reference is not None and reference != host:
        mine, theirs = stamp.get(host, {}), stamp.get(reference, {})
        shared = [abs(mine[s] - theirs[s]) for s in mine.keys() & theirs.keys()]
        skew = max(shared) if shared else None
    elif reference == host:
        skew = 0.0
    return HostAlignment(
        hostname=host,
        rows=len(offsets),
        late=late,
        missed=missed,
        unaccounted=unaccounted,
        max_offset_seconds=max(offsets) if offsets else None,
        max_skew_seconds=skew,
    )


def _fold_runs(alignment: SlotAlignment, *, full: bool) -> Iterable[tuple[int, int]]:
    """Yield ``(first, last)`` slot spans.

    Consecutive slots where every host has the same non-``late`` status
    (all ``ok``, or all ``unaccounted`` before the first poll) fold into one
    span unless ``full``. Slots with any per-host difference are always listed.
    """
    seq = 0
    while seq < alignment.slot_count:
        key = _uniform_key(alignment.cells[seq])
        if full or key is None:
            yield seq, seq
            seq += 1
            continue
        end = seq
        while end + 1 < alignment.slot_count and _uniform_key(alignment.cells[end + 1]) == key:
            end += 1
        yield seq, end
        seq = end + 1


def _uniform_key(per_slot: Mapping[str, SlotCell]) -> str | None:
    """The shared label when every host has the same foldable status, else ``None``."""
    labels = {cell.label for cell in per_slot.values()}
    if len(labels) != 1:
        return None
    label = next(iter(labels))
    return label if label in (STATUS_OK, STATUS_UNACCOUNTED) else None


def _cell_text(cell: SlotCell) -> str:
    if cell.offset_seconds is not None:
        offset = 0.0 if abs(cell.offset_seconds) < 0.005 else cell.offset_seconds
        return f"{cell.status} {offset:+.2f}s"
    return cell.label


def _missed_ranges(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    entries = manifest.get("missed_sample_ranges")
    if not isinstance(entries, list):
        return out
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        host, first, last = entry.get("hostname"), entry.get("first_scrape_seq"), entry.get("last_scrape_seq")
        if not isinstance(host, str) or not isinstance(first, int) or not isinstance(last, int):
            continue
        if isinstance(first, bool) or isinstance(last, bool) or first < 0 or last < first:
            continue
        reasons = entry.get("reason_codes")
        primary = reasons[0] if isinstance(reasons, list) and reasons and isinstance(reasons[0], str) else "?"
        out.append(
            {
                "hostname": host,
                "first_scrape_seq": first,
                "last_scrape_seq": last,
                "reason": _REASON_ABBREV.get(primary, primary),
            }
        )
    return out


def _positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _fmt(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}s"
