# CPU Power Telemetry

Host-side CPU power collection for NVIDIA Grace nodes, run alongside GPU DCGM
power telemetry as an independent, best-effort leg.

## Table of Contents

- [Overview](#overview)
- [Enabling It](#enabling-it)
- [How It Starts](#how-it-starts)
- [Collection Sources](#collection-sources)
- [Output Format](#output-format)
- [Computing Total Energy Over a Run](#computing-total-energy-over-a-run)
- [Relationship to GPU Power Telemetry](#relationship-to-gpu-power-telemetry)

---

## Overview

CPU power collection is a scrape-based, head-node-orchestrated design, not an
in-job daemon per node writing its own files. On each worker node, srtctl
launches a small HTTP exporter process directly on the bare host (outside the
model container, so it can read host power interfaces) that serves a
Prometheus `/metrics` endpoint. A single collector thread on the head node
(`CpuPowerCollector`, `src/srtctl/core/power/cpu_session.py`) polls every
worker's exporter on a fixed interval, parses the scrape body, and appends
rows to one shared `cpu/samples.csv` for the whole run.

The leg is fully decoupled from the GPU DCGM power pipeline's lifecycle and is
always best-effort: an unresolvable node, an unreachable exporter, a malformed
scrape, or a wedged collector thread is absorbed and logged, never raised into
the benchmark. There is no `required` flag for CPU power — gaps in
`samples.csv` are the visible cost of a failure, not a blocked run.

## Enabling It

Presence of `telemetry.cpu_power_exporter` (not a separate `enabled` flag)
turns CPU power collection on:

```yaml
telemetry:
  enabled: true                # master switch; also gates GPU DCGM power telemetry
  cpu_power_exporter:
    port: 9405                 # default; exporter listen port / collector scrape port
    source: auto                # "auto" | "acpi" | "dcgm", passed to the exporter binary
```

`telemetry.enabled: true` no longer requires `dcgm_exporter` — a recipe may
configure `cpu_power_exporter` alone with no DCGM leg at all. The sampling
cadence and timeouts (`collect_interval_ms`, `request_timeout_seconds`,
`startup_timeout_seconds`, `collector_join_timeout_seconds`) are shared with
the DCGM leg on `TelemetryConfig`; there is no separate CPU-specific set.

Config lives in `CpuPowerExporterConfig` (`src/srtctl/core/schema.py`), nested
under `TelemetryConfig.cpu_power_exporter`. Port-collision validation
(against `dcgm_exporter`, `observability.tachometer`'s exporters, and any
Dynamo system port) happens in `SrtConfig._validate_cpu_power_exporter`.

## How It Starts

`start_cpu_power_telemetry()` in `src/srtctl/cli/mixins/telemetry_stage.py` is
called from the sweep startup path alongside the tachometer and GPU DCGM
exporter. It resolves the exporter binary, then launches one `srun` task per
worker node (or per het-group chunk, `use_bash_wrapper=False` — bare host, no
container):

```bash
srun --nodes=<N> --ntasks=<N> --nodelist=<nodes> \
     --output=<log_dir>/telemetry_cpu_power_exporter.%N.out \
     [--het-group=<id>] \
     <cpu-power-exporter binary> --port 9405 --source auto
```

The binary is resolved via `_resolve_bundled_binary("cpu-power-exporter")` — a
Rust binary installed by `make setup`. When that binary is absent or not
executable, srtctl falls back to a Python stdlib exporter
(`python3 -m srtctl.core.cpu_power_exporter`), which is ACPI-only and has no
`--source` flag; a non-`auto` `source` request logs a warning in that case
instead of being silently dropped.

Once the exporter tasks are launched, `CpuPowerCollector.start()` resolves
each worker's IP (`get_hostname_ip`, respecting `runtime.network_interface`),
opens the `cpu/samples.csv` writer, and starts a background thread that polls
every endpoint's `/metrics` every `collect_interval_ms` and appends parsed rows.
Any launch failure for the exporter tasks themselves is caught and logged; the
collector object is still returned (with whatever endpoints did resolve) so
the caller doesn't have to special-case a partial launch.

At job teardown, `stop_and_finalize()` stops the collector thread, closes the
CSV writer, and writes `cpu_manifest.json` (non-authoritative: per-node
scrape/error counts and the resolved source mode, for debugging — the CSV is
the source of truth).

## Collection Sources

The exporter binary itself decides ACPI vs. DCGM per its own `--source` flag:

- **`acpi`** — reads Linux ACPI `power_meter` hwmon sysfs channels. Reports
  per-channel detail: `cpu`, `sysio`, and (where firmware exposes it)
  `grace`-kind rails per socket.
- **`dcgm`** — reads DCGM CPU entity power directly, one already-aggregated
  value per socket.
- **`auto`** (default) — tries DCGM first, falls back to ACPI when DCGM is
  unavailable or reports no CPU entities.

The exporter resolves this once at process startup and serves only one metric
family (`cpu_power_dcgm_watts` or `cpu_power_acpi_watts`) for its lifetime.
Client-side parsing (`src/srtctl/core/power/cpu_parser.py`) prefers ACPI
readings if a scrape body ever contained both, since ACPI carries more detail.

## Output Format

`samples.csv` under `<log_dir>/<telemetry.storage_subdir>/cpu/` has header:

```
schema_version, timestamp_unix, hostname, source, sensor, socket_id, power_w, total_power_w
```

- **`power_w`** — one sensor's power reading for that scrape. `sensor` names
  look like `CPU0:cpuPowerUsageW` (ACPI) or a DCGM field label; granularity is
  per-socket.
- **`total_power_w`** — the node-level total for that scrape, duplicated on
  every sensor row at the same `(hostname, timestamp_unix)`. In DCGM mode this
  is the sum of the per-socket DCGM values. In ACPI mode it is **not** a sum of
  the `cpu`- and `sysio`-kind rails: whenever a `grace`-kind channel exists for
  a socket, that channel alone is the total. Real hardware traces show `grace`
  at roughly 93-104W against `cpu`+`sysio` combined at roughly 53-58W for the
  same socket — `grace` measures the whole Grace SoC power boundary, not
  literally `cpu + sysio`. **When no `grace` channel is present for a scrape,
  `total_power_w` is left blank** for every row from that scrape rather than
  guessed from the component rails; per-sensor `power_w` values are still
  populated. Consumers reading this CSV (e.g.
  `srtctl.analysis.power_energy_report.load_cpu_samples`) must skip blank
  `total_power_w` rows rather than treat them as `0`.

`cpu_manifest.json` alongside it is non-authoritative debugging metadata:
per-node scrape/error counts and the resolved source mode, plus start/stop
timestamps and the producer's git commit.

## Computing Total Energy Over a Run

The collector intentionally never integrates power into energy — same
philosophy as the GPU power artifact contract
(`src/srtctl/core/power/contract.py`: it never integrates power into energy;
that belongs to consumers of the artifact contract). To get run-total energy:

```python
import pandas as pd
import numpy as np

df = pd.read_csv("samples.csv")
df = df[df["total_power_w"] != ""]  # skip scrapes with no grace channel

# total_power_w repeats across every sensor row for the same (hostname, timestamp);
# dedupe before integrating or sockets get double-counted.
per_node_ts = (
    df[["hostname", "timestamp_unix", "total_power_w"]]
    .drop_duplicates(subset=["hostname", "timestamp_unix"])
    .sort_values(["hostname", "timestamp_unix"])
)

def energy_joules(group: pd.DataFrame) -> float:
    return float(np.trapezoid(group["total_power_w"].astype(float), x=group["timestamp_unix"]))

energy_per_node_j = per_node_ts.groupby("hostname").apply(energy_joules)
run_total_wh = energy_per_node_j.sum() / 3600
```

Use trapezoidal integration (`np.trapezoid`; `np.trapz` was removed in numpy
2.0), not `mean(power) * duration` — the scrape loop is not perfectly uniform,
and scrape failures leave gaps. For per-sensor energy instead of per-node,
group by `(hostname, sensor)` (or `(hostname, socket_id)`) on `power_w`
instead of `total_power_w`.

## Relationship to GPU Power Telemetry

GPU power telemetry (`start_gpu_power_telemetry`, same mixin) works
similarly in shape — an exporter process per worker node scraped by a
head-node collector — but the exporter is a containerized DCGM exporter
sidecar (`telemetry.dcgm_exporter`, launched via `_start_exporter_container`)
rather than a bare-host process, and it is not best-effort by default:
`telemetry.required` (which applies to the DCGM leg) can fail the benchmark
stage if publishable GPU power artifacts can't be produced. CPU power has no
equivalent `required` semantics; it is always best-effort.

GPU power *limits* (apply/restore audited caps, `src/srtctl/core/gpu_power_limit.py`)
are a separate, unrelated top-level config (`gpu_power_limits`) — not part of
`telemetry.cpu_power_exporter`.
