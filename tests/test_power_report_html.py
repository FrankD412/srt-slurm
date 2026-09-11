# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the self-contained HTML power/perf report."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from srtctl.analysis.power_energy_report import PowerReportError
from srtctl.analysis.power_report_html import (
    _build_facets,
    _dedupe_labels,
    _downsample_minmax,
    _pareto_points,
    build,
    build_combined,
    build_combined_report,
    build_report,
    main,
)

# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------


def test_downsample_passes_through_short_series_unchanged() -> None:
    times = np.arange(10, dtype=float)
    watts = np.arange(10, dtype=float)

    out_t, out_w = _downsample_minmax(times, watts, max_buckets=100)

    assert list(out_t) == list(times)
    assert list(out_w) == list(watts)


def test_downsample_caps_point_count_and_keeps_the_peak() -> None:
    n = 10_000
    times = np.arange(n, dtype=float)
    watts = np.zeros(n)
    watts[n // 2] = 999.0  # a lone spike a stride/average sample would likely miss

    out_t, out_w = _downsample_minmax(times, watts, max_buckets=50)

    assert len(out_t) <= 100
    assert out_w.max() == pytest.approx(999.0)


# ---------------------------------------------------------------------------
# Facet building
# ---------------------------------------------------------------------------


def _series(watts: list[float]) -> tuple[np.ndarray, np.ndarray]:
    return np.arange(len(watts), dtype=float), np.array(watts, dtype=float)


def test_build_facets_groups_by_host_sorted_by_index() -> None:
    per_device = {
        ("node-b", 0): _series([10.0, 12.0]),
        ("node-a", 1): _series([20.0, 22.0]),
        ("node-a", 0): _series([30.0, 32.0]),
    }

    facets = _build_facets(per_device, label_fmt="gpu{index}")

    assert [f["host"] for f in facets] == ["node-a", "node-b"]
    node_a = facets[0]
    assert [s["label"] for s in node_a["series"]] == ["gpu0", "gpu1"]


def test_build_facets_assigns_slot_and_pattern_cycling_past_three_devices() -> None:
    per_device = {("node-a", i): _series([float(i)] * 3) for i in range(4)}

    facets = _build_facets(per_device, label_fmt="gpu{index}")

    slots = [s["slot"] for s in facets[0]["series"]]
    patterns = [s["pattern"] for s in facets[0]["series"]]
    assert slots == [0, 1, 2, 0]
    assert patterns == [0, 0, 0, 1]  # the 4th device reuses slot 0 with the next stroke pattern


def test_build_facets_time_shifts_to_the_earliest_sample_across_all_devices() -> None:
    per_device = {
        ("node-a", 0): (np.array([100.0, 101.0]), np.array([1.0, 2.0])),
        ("node-a", 1): (np.array([105.0, 106.0]), np.array([3.0, 4.0])),
    }

    facets = _build_facets(per_device, label_fmt="gpu{index}")

    series = facets[0]["series"]
    assert series[0]["t"] == [0.0, 1.0]
    assert series[1]["t"] == [5.0, 6.0]


def test_build_facets_empty_input_returns_no_facets() -> None:
    assert _build_facets({}, label_fmt="gpu{index}") == []


# ---------------------------------------------------------------------------
# Pareto points
# ---------------------------------------------------------------------------


def _report_dict(
    *,
    concurrency: int,
    output_tps: float | None,
    tps_per_gpu: float | None,
    num_gpus: int = 2,
) -> dict:
    return {
        "benchmark_type": "aiperf",
        "concurrency": concurrency,
        "tpot_p90_ms": 12.0,
        "timing": {"computed": {"duration_seconds": 60.0}},
        "perf_per_watt": {
            "output_tokens_per_second": output_tps,
            "output_tokens_per_second_per_gpu": tps_per_gpu,
            "num_gpus": num_gpus,
            "gpu_avg_power_w": 300.0,
            "cpu_avg_power_w": 40.0,
            "output_tokens_per_second_per_gpu_watt": 0.16,
        },
    }


def test_pareto_points_extracts_xy_and_label() -> None:
    reports = [_report_dict(concurrency=4, output_tps=100.0, tps_per_gpu=50.0)]

    points = _pareto_points(reports, run_label="runA")

    assert len(points) == 1
    assert points[0]["label"] == "runA · aiperf c=4"
    assert points[0]["x"] == 100.0
    assert points[0]["y"] == 50.0


def test_pareto_points_skips_rows_with_no_gpu_count() -> None:
    reports = [
        _report_dict(concurrency=4, output_tps=100.0, tps_per_gpu=50.0),
        _report_dict(concurrency=8, output_tps=None, tps_per_gpu=None, num_gpus=0),
    ]

    points = _pareto_points(reports, run_label="runA")

    assert len(points) == 1
    assert points[0]["label"] == "runA · aiperf c=4"


def test_pareto_points_omits_run_label_prefix_when_none() -> None:
    reports = [_report_dict(concurrency=4, output_tps=100.0, tps_per_gpu=50.0)]

    points = _pareto_points(reports)

    assert points[0]["label"] == "aiperf c=4"


def test_pareto_points_includes_panel_fields() -> None:
    reports = [_report_dict(concurrency=4, output_tps=100.0, tps_per_gpu=50.0)]

    points = _pareto_points(reports, run_label="runA")

    field_names = [name for name, _ in points[0]["fields"]]
    assert field_names == [
        "Concurrency",
        "Output throughput",
        "TPS / active GPU",
        "P90 TPOT",
        "Average GPU power",
        "Average CPU power",
        "TPS / GPU watt",
        "Profile duration",
    ]


# ---------------------------------------------------------------------------
# End-to-end report build
# ---------------------------------------------------------------------------


def _write_gpu_csv(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["schema_version", "timestamp_unix", "scrape_seq", "hostname", "gpu_index", "gpu_uuid", "power_w"]
        )
        writer.writerows(rows)


def _write_cpu_csv(path: Path, rows: list[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "schema_version",
                "timestamp_unix",
                "hostname",
                "source",
                "sensor",
                "socket_id",
                "power_w",
                "total_power_w",
            ]
        )
        writer.writerows(rows)


def _write_aiperf_conc(log_dir: Path, concurrency: int) -> None:
    conc_dir = log_dir / "agentic" / f"conc_{concurrency}" / "aiperf_artifacts"
    conc_dir.mkdir(parents=True)
    with (conc_dir / "profile_export.jsonl").open("w") as handle:
        handle.write(
            json.dumps(
                {
                    "metadata": {
                        "benchmark_phase": "profiling",
                        "request_start_ns": 10_000_000_000,
                        "request_end_ns": 20_000_000_000,
                    }
                }
            )
            + "\n"
        )
    (conc_dir / "profile_export_aiperf.json").write_text(
        json.dumps(
            {"total_osl": {"avg": 5.0}, "total_isl": {"avg": 2.0}, "inter_token_latency": {"p50": 8.0, "p90": 12.0}}
        )
    )


def _write_aiperf_run(log_dir: Path, *, concurrencies: tuple[int, ...] = (4,)) -> None:
    (log_dir / "benchmark.out").write_text(
        "17:59:31.680 NOTICE   Phase profiling (profiling) started (runner.py:593)\n"
        "19:00:01.681 NOTICE   Phase profiling (profiling) complete (runner.py:1162)\n"
    )
    for concurrency in concurrencies:
        _write_aiperf_conc(log_dir, concurrency)


def test_build_report_renders_summary_table_and_charts(tmp_path: Path) -> None:
    log_dir = tmp_path
    _write_aiperf_run(log_dir)
    _write_gpu_csv(
        log_dir / "power" / "samples.csv",
        [
            (1, 9.0, 1, "node-a", 0, "GPU-a", 100.0),
            (1, 15.0, 2, "node-a", 0, "GPU-a", 110.0),
            (1, 21.0, 3, "node-a", 0, "GPU-a", 105.0),
        ],
    )
    _write_cpu_csv(
        log_dir / "power" / "cpu" / "samples.csv",
        [
            (2, 9.0, "node-a", "acpi", "CPU0:cpuPowerUsageW", 0, 40.0, 40.0),
            (2, 15.0, "node-a", "acpi", "CPU0:cpuPowerUsageW", 0, 44.0, 44.0),
            (2, 21.0, "node-a", "acpi", "CPU0:cpuPowerUsageW", 0, 42.0, 42.0),
        ],
    )

    content = build_report(log_dir)

    assert "Throughput &amp; power by concurrency" in content
    assert "GPU power over time" in content
    assert "CPU socket power over time" in content
    assert "node-a" in content
    assert "gpu0" in content
    assert "socket0" in content


def test_build_report_omits_pareto_tab_with_a_single_concurrency_point(tmp_path: Path) -> None:
    log_dir = tmp_path
    _write_aiperf_run(log_dir)
    _write_gpu_csv(
        log_dir / "power" / "samples.csv",
        [(1, 9.0, 1, "node-a", 0, "GPU-a", 100.0), (1, 21.0, 3, "node-a", 0, "GPU-a", 105.0)],
    )

    content = build_report(log_dir)

    assert "Pareto view" not in content


def test_build_report_includes_pareto_tab_with_multiple_concurrency_points(tmp_path: Path) -> None:
    log_dir = tmp_path
    _write_aiperf_run(log_dir, concurrencies=(4, 8))
    _write_gpu_csv(
        log_dir / "power" / "samples.csv",
        [(1, 9.0, 1, "node-a", 0, "GPU-a", 100.0), (1, 21.0, 3, "node-a", 0, "GPU-a", 105.0)],
    )

    content = build_report(log_dir)

    assert "Pareto view" in content
    assert "Selected run" in content
    assert "aiperf c=4" in content
    assert "aiperf c=8" in content


def test_build_report_charts_only_without_benchmark_windows(tmp_path: Path) -> None:
    """No benchmark.out at all: still renders the power charts, just no stats table."""
    log_dir = tmp_path
    _write_gpu_csv(
        log_dir / "power" / "samples.csv",
        [
            (1, 9.0, 1, "node-a", 0, "GPU-a", 100.0),
            (1, 15.0, 2, "node-a", 0, "GPU-a", 110.0),
        ],
    )

    content = build_report(log_dir)

    assert "No concurrency-level benchmark windows" in content
    assert "GPU power over time" in content


def test_build_report_raises_when_nothing_applies(tmp_path: Path) -> None:
    with pytest.raises(PowerReportError):
        build_report(tmp_path)


def test_build_writes_html_file_next_to_the_run(tmp_path: Path) -> None:
    log_dir = tmp_path
    _write_gpu_csv(log_dir / "power" / "samples.csv", [(1, 9.0, 1, "node-a", 0, "GPU-a", 100.0)])

    out = build(log_dir)

    assert out == log_dir / "power_report.html"
    assert out.is_file()


def test_build_returns_none_when_nothing_applies(tmp_path: Path) -> None:
    assert build(tmp_path) is None


# ---------------------------------------------------------------------------
# Multi-directory rollup
# ---------------------------------------------------------------------------


def test_dedupe_labels_suffixes_repeats() -> None:
    assert _dedupe_labels(["a", "b", "a", "a", "c"]) == ["a", "b", "a (2)", "a (3)", "c"]


def _write_run(log_dir: Path, *, gpu_watts: float) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    _write_aiperf_run(log_dir)
    _write_gpu_csv(
        log_dir / "power" / "samples.csv",
        [
            (1, 9.0, 1, "node-a", 0, "GPU-a", gpu_watts),
            (1, 15.0, 2, "node-a", 0, "GPU-a", gpu_watts + 10.0),
            (1, 21.0, 3, "node-a", 0, "GPU-a", gpu_watts + 5.0),
        ],
    )


def test_build_combined_report_merges_rows_from_every_run(tmp_path: Path) -> None:
    run_a = tmp_path / "runA" / "logs"
    run_b = tmp_path / "runB" / "logs"
    _write_run(run_a, gpu_watts=100.0)
    _write_run(run_b, gpu_watts=200.0)

    content = build_combined_report([run_a, run_b])

    assert "2 runs: runA, runB" in content
    assert content.count("<td>runA · aiperf c=4</td>") == 1
    assert content.count("<td>runB · aiperf c=4</td>") == 1
    assert "GPU power over time — runA" in content
    assert "GPU power over time — runB" in content
    # one shared table, not one table per run
    assert content.count("Throughput &amp; power by concurrency") == 1
    assert "Pareto view" in content
    assert "runA · aiperf c=4" in content
    assert "runB · aiperf c=4" in content


def test_build_combined_report_dedupes_identical_labels(tmp_path: Path) -> None:
    run_a = tmp_path / "same" / "logs"
    run_b = tmp_path / "same_copy" / "same" / "logs"
    _write_run(run_a, gpu_watts=100.0)
    _write_run(run_b, gpu_watts=150.0)

    content = build_combined_report([run_a, run_b])

    assert "same, same (2)" in content
    assert "<td>same · aiperf c=4</td>" in content
    assert "<td>same (2) · aiperf c=4</td>" in content


def test_build_combined_report_skips_empty_dirs_but_keeps_the_rest(tmp_path: Path) -> None:
    run_a = tmp_path / "runA" / "logs"
    empty_dir = tmp_path / "empty" / "logs"
    empty_dir.mkdir(parents=True)
    _write_run(run_a, gpu_watts=100.0)

    content = build_combined_report([run_a, empty_dir])

    assert "1 runs: runA" in content


def test_build_combined_report_raises_when_every_dir_is_empty(tmp_path: Path) -> None:
    empty_a = tmp_path / "a"
    empty_b = tmp_path / "b"
    empty_a.mkdir()
    empty_b.mkdir()

    with pytest.raises(PowerReportError):
        build_combined_report([empty_a, empty_b])


def test_build_combined_writes_to_the_given_output_path(tmp_path: Path) -> None:
    run_a = tmp_path / "runA" / "logs"
    run_b = tmp_path / "runB" / "logs"
    _write_run(run_a, gpu_watts=100.0)
    _write_run(run_b, gpu_watts=200.0)
    out_path = tmp_path / "combined.html"

    out = build_combined([run_a, run_b], output_path=out_path)

    assert out == out_path
    assert out_path.is_file()


def test_build_combined_returns_none_when_nothing_applies(tmp_path: Path) -> None:
    empty_a = tmp_path / "a"
    empty_b = tmp_path / "b"
    empty_a.mkdir()
    empty_b.mkdir()

    assert build_combined([empty_a, empty_b], output_path=tmp_path / "out.html") is None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_main_single_dir_writes_into_the_run_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    log_dir = tmp_path
    _write_gpu_csv(log_dir / "power" / "samples.csv", [(1, 9.0, 1, "node-a", 0, "GPU-a", 100.0)])

    exit_code = main([str(log_dir)])

    assert exit_code == 0
    assert (log_dir / "power_report.html").is_file()
    assert str(log_dir / "power_report.html") in capsys.readouterr().out


def test_main_multi_dir_writes_combined_report_to_explicit_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_a = tmp_path / "runA" / "logs"
    run_b = tmp_path / "runB" / "logs"
    _write_run(run_a, gpu_watts=100.0)
    _write_run(run_b, gpu_watts=200.0)
    out_path = tmp_path / "combined.html"

    exit_code = main([str(run_a), str(run_b), "-o", str(out_path)])

    assert exit_code == 0
    assert out_path.is_file()
    assert "2 runs" in out_path.read_text()


def test_main_multi_dir_defaults_output_to_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    run_a = tmp_path / "runA" / "logs"
    run_b = tmp_path / "runB" / "logs"
    _write_run(run_a, gpu_watts=100.0)
    _write_run(run_b, gpu_watts=200.0)
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    exit_code = main([str(run_a), str(run_b)])

    assert exit_code == 0
    assert (cwd / "power_report_combined.html").is_file()


def test_main_returns_nonzero_when_nothing_applies(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main([str(tmp_path)])

    assert exit_code == 1
    assert "error" in capsys.readouterr().err
