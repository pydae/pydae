# modbus test suite

Structured experiments for the Modbus PV-plant cosim. Each test is a
self-contained folder with a single `run.py` driver that configures the PPC
over its HTTP API, captures telemetry, and writes **CSV + static SVG +
interactive Plotly HTML** into its own `csv/` and `plots/`.

| Test | Purpose |
|---|---|
| [`open_loop_step/`](open_loop_step/) | Plant + comms time response (gains=0, feedforward only). Exposes the plant DC gain. |
| [`closed_loop_step/`](closed_loop_step/) | PI gain-design step response (configured gains). |
| [`frequency_response/`](frequency_response/) | Sinusoidal sweeps → Bode. Modes: P-only, Q-only, P+Q at 90°. |
| [`mitm_attack/`](mitm_attack/) | Modbus man-in-the-middle: spoofed PPC reading vs true plant. |

## Prerequisites

Cosim running and the PPC API reachable at `http://localhost:8090` (the socat
relay), per the top-level `../README.md`:

```bash
cd ..            # modbus/
sudo python3 run.py            # Terminal A  (add --mitm for the attack test)
```

Then from Terminal B:

```bash
PY=/home/jmmmacubu/miniforge3/envs/pydae_dev/bin/python3
$PY tests/open_loop_step/run.py
$PY tests/closed_loop_step/run.py
$PY tests/frequency_response/run.py --mode p
$PY tests/frequency_response/run.py --mode q
$PY tests/frequency_response/run.py --mode pq90
```

## Shared library — `lib/`

| Module | What it provides |
|---|---|
| `common.py` | Paths, PPC URL, interpreter, plant-config helpers (`inverter_ids`, `baseline_gains`). |
| `ppc_api.py` | `status()`, `set_setpoints()`, `set_sine()`, `get_sine()`. |
| `capture.py` | `pre_settle()`, `drive_pq_step()`, `Recorder` (background `colinker.recorder`, bypassing the `record_ctl start` readline hang). |
| `fit.py` | `fit_sine()` / `gain_phase()` for Bode extraction. |
| `plots.py` | `step_figure()`, `bode_figure()`, `sweep_timeseries_figure()` — each emits SVG **and** Plotly HTML. |

Everything is config-driven: `common.py` auto-discovers the plant JSON in the
parent dir (the one with a `ppc` block) and reads the inverter list and gains
from it, so the same drivers work for any M×N plant. This suite is scaffolded
by `pv_designer.write_suite()`.
