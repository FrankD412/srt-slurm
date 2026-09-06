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
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_csv_tolerantly(path: Path) -> io.StringIO:
    """Read a CSV that another thread may be appending to, dropping a torn tail.

    The power collectors flush ``samples.csv`` every cycle, so a read taken
    mid-run can land between a row's bytes. Anything after the last newline is
    discarded; a file with no newline at all yields an empty stream, which the
    sample loaders turn into empty series and the integrator then refuses to
    integrate.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.endswith("\n"):
        cut = text.rfind("\n")
        text = text[: cut + 1] if cut >= 0 else ""
    return io.StringIO(text)
