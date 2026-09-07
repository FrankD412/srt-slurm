# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Incremental, per-case power/energy emission during a live benchmark.

Best-effort by design and strictly additive: the terminal
``power_energy_report.json`` remains authoritative and unchanged. Nothing here
may affect the job exit code or block the sweep.

The design is a periodic idempotent rescan -- each poll re-runs the same
``discover_run`` / ``build_concurrency_report`` code the terminal report uses,
and writes out any case that is complete and not yet emitted. That makes it
benchmark-agnostic by construction: whatever the terminal report can discover,
this discovers.
"""

from __future__ import annotations

import io
import json
import logging
import time
from pathlib import Path

from srtctl.analysis.power_energy_report import (
    ConcurrencyReport,
    CpuSamples,
    GpuSamples,
    PowerReportError,
    RunPaths,
    aiperf_window,
    build_concurrency_report,
    discover_run,
    load_cpu_samples_from,
    load_gpu_roles,
    load_gpu_samples_from,
    report_to_dict,
    sa_bench_window,
)
from srtctl.core.power.contract import atomic_write_json

logger = logging.getLogger(__name__)


def read_csv_tolerantly(path: Path) -> io.StringIO:
    """Read a CSV that another thread may be appending to, dropping a torn tail.

    The power collectors flush ``samples.csv`` every cycle, so a read taken
    mid-run can land between a row's bytes. Anything after the last newline is
    discarded; a file with no newline at all yields an empty stream, which the
    sample loaders turn into empty series. ``IncrementalPowerEmitter`` treats a
    discovered-but-empty sample series as not-ready-yet (see its emptiness
    guard) rather than relying on the integrator to reject it, since an empty
    series integrates to zero rather than raising.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.endswith("\n"):
        cut = text.rfind("\n")
        text = text[: cut + 1] if cut >= 0 else ""
    return io.StringIO(text)


INDEX_FILENAME = "power_energy_report.jsonl"
CO_LOCATED_FILENAME_TEMPLATE = "power_energy_c{concurrency}.json"
INCREMENTAL_SCHEMA_VERSION = 1

# Every exception a not-yet-complete artifact can plausibly raise. A case that
# trips any of these is simply retried on the next poll.
_NOT_READY = (
    PowerReportError,
    json.JSONDecodeError,
    OSError,
    KeyError,
    ValueError,
    IndexError,
    AttributeError,
    TypeError,
)


class IncrementalPowerEmitter:
    """Writes each benchmark case's energy result as soon as that case completes.

    Idempotent: ``poll()`` emits a given concurrency exactly once. Never
    raises -- an incomplete artifact, an uncovered window or an unreadable file
    all mean "not ready yet, retry next tick".
    """

    def __init__(self, log_dir: Path):
        self._log_dir = log_dir
        self._emitted: set[int] = set()

    @property
    def index_path(self) -> Path:
        return self._log_dir / INDEX_FILENAME

    def poll(self) -> tuple[int, ...]:
        """Emit every case that is complete and not yet written. Returns what it emitted."""
        try:
            paths = discover_run(self._log_dir)
        except _NOT_READY as exc:
            logger.debug("Incremental power: run not discoverable yet: %s", exc)
            return ()

        pending = sorted((c, s) for c, s in paths.concurrency_sources if c not in self._emitted)
        if not pending:
            return ()

        try:
            cpu_samples, gpu_samples = self._load_samples(paths)
        except _NOT_READY as exc:
            logger.debug("Incremental power: samples not readable yet: %s", exc)
            return ()

        emitted: list[int] = []
        for concurrency, source in pending:
            report = self._try_build(concurrency, source, cpu_samples, gpu_samples)
            if report is None:
                continue
            try:
                self._write(concurrency, source, report)
            except Exception:
                logger.warning("Incremental power: failed writing concurrency %d", concurrency, exc_info=True)
                continue
            self._emitted.add(concurrency)
            emitted.append(concurrency)

        if emitted:
            logger.info("Incremental power: emitted concurrency point(s) %s", emitted)
        return tuple(emitted)

    def _load_samples(self, paths: RunPaths) -> tuple[CpuSamples | None, GpuSamples | None]:
        """Load whichever sample series discovery found.

        A source that discovery never found (``None``) is legitimate -- a
        CPU-only or GPU-only run -- and is left as ``None``. But a source that
        *was* found and yet loaded with no series at all (e.g. the collector
        never started, or died before writing any rows) must NOT be treated as
        an empty-but-valid series: ``build_concurrency_report`` integrates
        empty series to 0 joules rather than raising, so we have to catch this
        here rather than relying on it to reject the case.
        """
        cpu_samples = None
        if paths.cpu_samples_csv is not None:
            cpu_samples = load_cpu_samples_from(read_csv_tolerantly(paths.cpu_samples_csv))
            if not cpu_samples.per_socket and not cpu_samples.per_node:
                raise PowerReportError(f"no CPU power samples yet in {paths.cpu_samples_csv}")

        gpu_samples = None
        if paths.gpu_samples_csv is not None:
            roles = load_gpu_roles(paths.gpu_manifest) if paths.gpu_manifest else None
            gpu_samples = load_gpu_samples_from(read_csv_tolerantly(paths.gpu_samples_csv), roles)
            if not gpu_samples.per_device and not gpu_samples.per_node:
                raise PowerReportError(f"no GPU power samples yet in {paths.gpu_samples_csv}")

        return cpu_samples, gpu_samples

    def _try_build(
        self,
        concurrency: int,
        source: Path,
        cpu_samples: CpuSamples | None,
        gpu_samples: GpuSamples | None,
    ) -> ConcurrencyReport | None:
        """Build the case's report, or None if it is not ready yet.

        ``build_concurrency_report`` integrates every socket, device and node in
        the window, and ``windowed_energy`` refuses any window whose nearest
        sample is more than MAX_SAMPLE_GAP_SECONDS from a boundary. So a case
        that is not yet fully bracketed by samples raises here and is withheld
        -- which is exactly the completeness guarantee we want.
        """
        try:
            if source.name == "profile_export.jsonl":
                window = aiperf_window(concurrency, source)
            else:
                window = sa_bench_window(concurrency, source)
            return build_concurrency_report(window, cpu_samples, gpu_samples)
        except _NOT_READY as exc:
            logger.debug("Incremental power: concurrency %d not ready: %s", concurrency, exc)
            return None

    def _write(self, concurrency: int, source: Path, report: ConcurrencyReport) -> None:
        payload = {
            **report_to_dict(report),
            "schema_version": INCREMENTAL_SCHEMA_VERSION,
            "emitted_at_unix": time.time(),
        }
        co_located = source.parent / CO_LOCATED_FILENAME_TEMPLATE.format(concurrency=concurrency)
        atomic_write_json(co_located, payload)
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
            handle.flush()
