# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""IncrementalPowerEmitter against synthetic sa-bench and aiperf log dirs."""

import json

from srtctl.analysis.incremental_power import IncrementalPowerEmitter

GPU_HEADER = "schema_version,timestamp_unix,scrape_seq,hostname,gpu_index,gpu_uuid,power_w\n"


def _gpu_rows(timestamps):
    return "".join(f"1,{t},{i},node-a,0,GPU-aaa,100.0\n" for i, t in enumerate(timestamps))


def _make_sa_bench_log_dir(
    tmp_path,
    *,
    concurrency=4,
    sample_times=(999.5, 1002.0, 1005.0, 1008.0, 1010.5),
    cases=None,
):
    """A log dir shaped like a real sa-bench run: benchmark.out + results + power CSV.

    ``cases``, when given, overrides ``concurrency`` and is an iterable of
    ``(concurrency, start_unix, end_unix)`` tuples, one results file each --
    letting a test model a real sweep's sequential, non-overlapping windows.
    ``sample_times`` still governs the single shared power/samples.csv, so
    callers can control which case(s) are actually bracketed.
    """
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "benchmark.out").write_text("Successful requests: 100\n")

    power_dir = log_dir / "power"
    power_dir.mkdir()
    (power_dir / "samples.csv").write_text(GPU_HEADER + _gpu_rows(sample_times))

    results_dir = log_dir / "sa-bench_isl_128_osl_128"
    results_dir.mkdir()
    for conc, start, end in cases if cases is not None else ((concurrency, 1000.0, 1010.0),):
        (results_dir / f"results_concurrency_{conc}_gpus_8.json").write_text(
            json.dumps(
                {
                    "benchmark_start_time_unix": start,
                    "benchmark_end_time_unix": end,
                    "total_input_tokens": 1000,
                    "total_output_tokens": 2000,
                }
            )
        )
    return log_dir


def test_emits_a_completed_case(tmp_path):
    log_dir = _make_sa_bench_log_dir(tmp_path)
    emitter = IncrementalPowerEmitter(log_dir)

    assert emitter.poll() == (4,)

    co_located = log_dir / "sa-bench_isl_128_osl_128" / "power_energy_c4.json"
    assert co_located.is_file()
    payload = json.loads(co_located.read_text())
    assert payload["concurrency"] == 4
    assert payload["gpu_total_joules"] > 0
    assert payload["schema_version"] == 1
    assert "emitted_at_unix" in payload

    lines = (log_dir / "power_energy_report.jsonl").read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["concurrency"] == 4


def test_poll_is_idempotent(tmp_path):
    log_dir = _make_sa_bench_log_dir(tmp_path)
    emitter = IncrementalPowerEmitter(log_dir)

    assert emitter.poll() == (4,)
    assert emitter.poll() == ()

    lines = (log_dir / "power_energy_report.jsonl").read_text().splitlines()
    assert len(lines) == 1


def test_case_is_withheld_until_samples_bracket_the_window(tmp_path):
    # Samples stop at 1004, but the window ends at 1010 -- a 6s gap, beyond
    # MAX_SAMPLE_GAP_SECONDS, so integration must be refused.
    log_dir = _make_sa_bench_log_dir(tmp_path, sample_times=(999.5, 1002.0, 1004.0))
    emitter = IncrementalPowerEmitter(log_dir)

    assert emitter.poll() == ()
    assert not (log_dir / "power_energy_report.jsonl").exists()

    # The collector catches up; now the window is bracketed.
    (log_dir / "power" / "samples.csv").write_text(GPU_HEADER + _gpu_rows((999.5, 1002.0, 1004.0, 1007.0, 1010.5)))
    assert emitter.poll() == (4,)


def test_a_partially_written_result_json_is_treated_as_not_ready(tmp_path):
    log_dir = _make_sa_bench_log_dir(tmp_path)
    result = log_dir / "sa-bench_isl_128_osl_128" / "results_concurrency_4_gpus_8.json"
    result.write_text('{"benchmark_start_time_unix": 1000.0, "benchmark_end')
    emitter = IncrementalPowerEmitter(log_dir)

    assert emitter.poll() == ()  # must not raise

    result.write_text(
        json.dumps(
            {
                "benchmark_start_time_unix": 1000.0,
                "benchmark_end_time_unix": 1010.0,
                "total_input_tokens": 1000,
                "total_output_tokens": 2000,
            }
        )
    )
    assert emitter.poll() == (4,)


def test_a_torn_samples_csv_tail_is_tolerated(tmp_path):
    log_dir = _make_sa_bench_log_dir(tmp_path)
    samples = log_dir / "power" / "samples.csv"
    samples.write_text(samples.read_text() + "1,1011.0,5,node-a,0,GPU-")
    emitter = IncrementalPowerEmitter(log_dir)

    assert emitter.poll() == (4,)


def test_a_header_only_samples_csv_withholds_rather_than_emits_zero_joules(tmp_path):
    # power/samples.csv exists (discovery finds it) but has no data rows yet --
    # e.g. the collector hasn't started or died before writing a sample. This
    # must be withheld, not emitted with 0 joules.
    log_dir = _make_sa_bench_log_dir(tmp_path)
    samples = log_dir / "power" / "samples.csv"
    samples.write_text(GPU_HEADER)
    emitter = IncrementalPowerEmitter(log_dir)

    assert emitter.poll() == ()
    assert not (log_dir / "power_energy_report.jsonl").exists()
    assert not (log_dir / "sa-bench_isl_128_osl_128" / "power_energy_c4.json").exists()

    # Once real sample rows arrive, the case is emitted -- proving withheld,
    # not permanently rejected.
    samples.write_text(GPU_HEADER + _gpu_rows((999.5, 1002.0, 1005.0, 1008.0, 1010.5)))
    assert emitter.poll() == (4,)


def test_an_empty_log_dir_emits_nothing(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    emitter = IncrementalPowerEmitter(log_dir)

    assert emitter.poll() == ()  # must not raise


def _make_aiperf_log_dir(tmp_path):
    """A log dir shaped like a real aiperf run: conc_<N>/aiperf_artifacts/ per case."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "benchmark.out").write_text("Phase profiling (profiling) complete\n")

    power_dir = log_dir / "power"
    power_dir.mkdir()
    (power_dir / "samples.csv").write_text(GPU_HEADER + _gpu_rows((999.5, 1002.0, 1005.0, 1008.0, 1010.5)))

    artifacts = log_dir / "conc_8" / "aiperf_artifacts"
    artifacts.mkdir(parents=True)
    records = [
        {
            "metadata": {
                "benchmark_phase": "profiling",
                "request_start_ns": 1_000_000_000_000,
                "request_end_ns": 1_010_000_000_000,
            }
        }
    ]
    (artifacts / "profile_export.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (artifacts / "profile_export_aiperf.json").write_text(
        json.dumps({"total_osl": {"avg": 2000}, "total_isl": {"avg": 1000}})
    )
    return log_dir


def test_emits_an_aiperf_case_beside_its_own_source(tmp_path):
    log_dir = _make_aiperf_log_dir(tmp_path)
    emitter = IncrementalPowerEmitter(log_dir)

    assert emitter.poll() == (8,)

    # Placement rule is "next to the source artifact", so for aiperf that is
    # inside aiperf_artifacts/, not the conc_8/ directory above it.
    assert (log_dir / "conc_8" / "aiperf_artifacts" / "power_energy_c8.json").is_file()
    assert not (log_dir / "conc_8" / "power_energy_c8.json").exists()


def test_matches_the_terminal_report_for_the_same_case(tmp_path):
    """The incremental row and the terminal report must agree exactly."""
    from srtctl.analysis.power_energy_report import build_reports, report_to_dict

    log_dir = _make_sa_bench_log_dir(tmp_path)
    emitter = IncrementalPowerEmitter(log_dir)
    assert emitter.poll() == (4,)

    incremental = json.loads((log_dir / "sa-bench_isl_128_osl_128" / "power_energy_c4.json").read_text())
    terminal = report_to_dict(build_reports(log_dir)[0])

    for key in terminal:
        assert incremental[key] == terminal[key], key
