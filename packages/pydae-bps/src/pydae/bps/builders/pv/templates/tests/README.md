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
| [`mitm_attack/`](mitm_attack/) | Modbus man-in-the-middle: falsified PPC reading vs true plant (additive offset by default, fixed spoof optional). |

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

## MITM attack — falsification modes

The attack (`colinker.attacks.modbus_mitm`) is armed when the cosim is launched
with `--mitm`; the `mitm_attack/run.py` driver only records a window. Two
policies, both rewriting the POI active-power FC03 response in flight:

- **offset (default)** — bias the *real* POI reading by `cur·(1+frac)+abs`. The
  PPC sees a believable, slowly-diverging value and **settles at a small
  steady-state error without saturating**. The sign of `--mitm-offset-frac`
  picks the direction:
  - `+frac` → PPC reads *more* than real → plant runs **below** target
    (under-production). Default is `+0.05` (+5%).
  - `−frac` → PPC reads *less* than real → plant runs **above** target
    (**over-production**, e.g. silently breaching a grid-code power cap).
- **fixed (`--mitm-value-mw`)** — pin the readback at a constant. The PI error
  never closes, so the controller **rails to saturation** (the original
  over-power demo, loud and obvious).

```bash
# Terminal A — relaunch with the attack armed
sudo python3 run.py --no-cli --mitm                                   # default +5% offset
sudo python3 run.py --no-cli --mitm --mitm-offset-frac=-0.0909        # over-cap: real = PPC×1.10
sudo python3 run.py --no-cli --mitm --mitm-value-mw 5.0               # fixed spoof → saturation

# Terminal B — record a window that brackets arm (+start-after) → disarm
$PY tests/mitm_attack/run.py --duration 110
```

Example (Spanish 12 MW cap, `target_mw=12`, `--mitm-offset-frac=-0.0909`): the
PPC believes **12.0 MW** (compliant) while the plant injects **13.2 MW** (+10%,
over the cap) — no saturation, no alarm. See [`mitm_attack/`](mitm_attack/).

## Packet capture — wire-level forensics

The falsification is visible on the wire: in `--explicit-redirect` mode the PPC
talks to the MITM host (which forwards to the real POI), so the **same FC03
transaction carries different payload bytes at different capture points**. Add
`--pcap-dir` to run tcpdump in the PPC / POI / MITM namespaces:

```bash
# Terminal A
sudo python3 run.py --no-cli --mitm --mitm-offset-frac=-0.0909 \
     --mitm-start-after 40 --pcap-dir results_overpower/pcaps
# …let the attack window pass, then Ctrl-C to flush the pcaps.

# Decode + diff the captures (Modbus on tcp/502 here, not the 5000 default):
python -m colinker.pcap_report --pcap-dir results_overpower/pcaps \
    --register 1370 --dtype float32 --fmt CDAB --modbus-port 502 --scale-mw
```

`pcap_report` emits a markdown report (per-host PDU counts, per-host distinct
values, and a **same-tid / different-payload** attack table), a timeline SVG,
and a matched-response CSV. The POI capture shows the real value (13.2 MW)
while the PPC capture only ever carries the spoof (12.0 MW). The raw `.pcap`s
also open in **Wireshark** (filter `modbus`; compare `mbtcp.trans_id` across the
PPC and POI files).

## Shared library — `lib/`

| Module | What it provides |
|---|---|
| `common.py` | Paths, PPC URL (`PPC_URL`), interpreter, plant-config helpers (`inverter_ids`, `baseline_gains`). |
| `ppc_api.py` | `status()`, `set_setpoints()`, `set_sine()`, `get_sine()`. |
| `capture.py` | `pre_settle()`, `drive_pq_step()`, `Recorder` (background `colinker.recorder`, bypassing the `record_ctl start` readline hang). Honors `PPC_URL`/`SIM_URL` (defaults `:8090`/`:8080`) so the suite also runs against the no-Mininet `local/` case (`:5500`/`:8001`). |
| `fit.py` | `fit_sine()` / `gain_phase()` for Bode extraction. |
| `plots.py` | `step_figure()`, `bode_figure()`, `sweep_timeseries_figure()` — each emits SVG **and** Plotly HTML. |

Everything is config-driven: `common.py` auto-discovers the plant JSON in the
parent dir (the one with a `ppc` block) and reads the inverter list and gains
from it, so the same drivers work for any M×N plant. This suite is scaffolded
by `pv_designer.write_suite()`; the same `tests/` suite is emitted for both
`modbus/` and the no-Mininet `local/` case.
