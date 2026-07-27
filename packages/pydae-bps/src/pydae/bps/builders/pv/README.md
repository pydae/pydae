# `pydae.bps.builders.pv`

Generator for **utility-scale PV plant** models — produces:

1. A pydae-ready plant JSON (electrical + comms + cosim configs),
2. Optionally, a full **benchmark suite** folder ready to be driven by
   [`colinker`](https://github.com/pydae/colinker): four networked SCADA
   stacks (`modbus` / `iec104` / `dnp3` / `dds`) plus a no-Mininet `local`
   case — each with the structured `tests/` suite — and a `common/` folder
   with the `run_all_tests.py` orchestrator (+ monitor / plot / metrics), and
3. A topology diagram (PNG / SVG / PDF) of the plant and its comms network.

The pydae steady-state seed (`xy_0.json`) is **not** shipped in the scaffold —
`colinker.pydae_sim` writes it at model-build time. `make_xy_0(plant)` is still
available to build a seed in-memory (e.g. for a standalone `model.ini`).

The plant topology, cable sizing, and per-inverter REEC_B control mode
are all derived from a few high-level parameters: number of feeders,
inverters per feeder, voltage levels, inverter rating, and PQ-mode
setpoints.

---

## Module layout

```
pydae/bps/builders/pv/
├── __init__.py                              re-exports public API
├── README.md                                this file
├── pv_designer.py                           electrical + comms + suite scaffold
├── topology.py                              diagram renderer
├── template_mininet_local_modbus.json       comms-plane template (hosts/switches/links/configs)
└── templates/                               stamped into each suite: run_py/<proto>.py,
                                             common/*, local_run.sh, README_top.md, tests/
```

Public API:

| Function | What it does |
|---|---|
| `designer(M, N, …)` | Build the plant dict, write a flat `<output_file>.json`, return the dict |
| `make_xy_0(plant)` | Build the pydae Newton-solver seed for a plant dict |
| `write_suite(out_dir, plant, name, M, N)` | Drop the full benchmark folder tree (4 stacks + local + common/) |
| `draw_topology(json_or_dict, out=…)` | Render the topology to SVG / PNG / PDF |

All four are also importable directly from `pydae.bps.builders`.

---

## Quick start

```python
from pydae.bps.builders.pv import designer, write_suite, draw_topology

# 1. Build the plant dict (writes a flat copv_5_10.json next to wherever
#    you ran from, and returns the dict for downstream use).
plant = designer(
    M=5, N=10,
    output_file="copv_5_10.json",
    P_inv_max=1e6,                 # per-inverter nameplate (W)
    U_lv=800, U_mv=30e3, U_hv=132e3,
    Pref_pq=0.5, Qext_pq=0.1,      # PQ-mode setpoints (pu of S_n)
)

# 2. Scaffold the benchmark suite (4 stacks + local + common/run_all_tests) around the plant.
write_suite("./copv_5_10_wecc", plant, name="copv_5_10", M=5, N=10)

# 3. Render a topology diagram.
draw_topology("./copv_5_10_wecc/modbus/copv_5_10.json",
              "./copv_5_10_wecc/copv_5_10.svg")
```

Or run the module as a script — its `__main__` does the same three
steps with the parameters baked at the top of the block:

```bash
python -m pydae.bps.builders.pv.pv_designer
```

---

## Plant model

### Electrical topology

```
GRID  (HV slack — genape source)
  |    POI<->GRID line: X = 0.02 pu, R = 0.002 pu  (in S_trafo_poi base)
POI   (HV)
  |    plant step-up POIMV<->POI: X = 0.1 pu, R = 0  (own base)
  |    S_trafo_poi = 1.1 · M · N · P_inv_max
POIMV (MV main bus)
  |    per-feeder MV cable, daisy-chained
  |    cable sized from short-circuit current + thermal-tube rating
MV<F><N> ... MV<F><1>     (one row per feeder F = 1..M)
  |    per-inverter step-up: S = 1.2 · P_inv_max,
  |                           X = 0.05 pu, R = 0.01 pu  (own base)
LV<F><N>
  |    regc_a + REEC_B inverter (see "PQ mode" below)
```

Bus IDs follow `<level><FF><NN>` where `FF` is the feeder index and
`NN` is the inverter index inside the feeder (both 1-based, two
digits). For instance: `LV0312` is the LV bus of inverter 12 on
feeder 3.

`S_base` is a free parameter (`100 MVA` by default) — decoupled from
plant nameplate so the per-unit reactances of the POIMV-POI step-up and
the POI-GRID line stay constant as the plant grows.

### Cable sizing

The MV cable is automatically selected from a six-row aluminium catalogue
(1×95 … 1×630). For each plant, `designer()` computes:

- **Short-circuit current** at POIMV from the plant step-up impedance,
- **Nominal current** in the feeder from M·P_inv_max,

and picks the smallest cross-section that meets both. The chosen cable
is printed at generation time, e.g.:

```
Short circuit designed cable (12.86 kA): 1X150
Nominal current designed cable  (192.45 A): 1X95
Cable: 1X150
```

### PQ mode

Each inverter runs as a fixed **PQ generator** under the REEC_B convention
documented in
[`docs/pydae-bps/models/weccs/reec_b.md`](../../../../../../docs/pydae-bps/models/weccs/reec_b.md):

```
Vflag = 0    # Q-PI engaged (Kqp/Kqi track Qext)
Qflag = 1    # inner V-PI engaged (Kvp/Kvi -> Iqcmd)
PFflag = 0   # Qext is the Q setpoint (not power factor)
Pqflag = 1   # P priority on the current limit
Kqv   = 0    # VRT supplementary injection disabled
Ipcmd = Pref = Pref_pq        # match for consistent PF init
Iqcmd = Qext = Qext_pq
```

The other three supported REEC_B mode combinations
(`Vflag=1/Qflag=1` voltage regulator, `Vflag=0/Qflag=1/PFflag=1`
constant-PF, `Vflag=0/Qflag=0` rank-deficient — see the doc) require
swapping the wecc block by hand.

### Comms plane

The comms-side scaffold lives in `template_mininet_local_modbus.json`
and contributes:

- Infrastructure hosts: `SCADA`, `PPC`, `POI`, `MITM`, `CPN`.
- Core switches: `sPOI`, `sEEMU` (eth0 to host), `sKALI` (eth1 to host),
  `sPPC` (eth2 to host).
- The default `configs` block, which carries **both** modbus
  (`register` / `type` / `format`) and IEC 104
  (`io_register` / `type_iec104`) fields on every measurement and
  setpoint — so the same file works for both protocols without
  rewriting.

`designer()` then adds, per inverter:

- one host `LV<id>` on `10.10.<F>.<N>/16`,
- one per-device switch `sLV<id>`,
- a chain link `s<previous> → s<this>` per feeder so the topology
  daisies into `sPOI`,
- a device entry under `devices[]` with `modbus_ip`, `modbus_port=502`,
  `reg_0` (1000-spaced), and the modbus-default `protocol` field.

---

## Output: the benchmark suite

`write_suite(out_dir, plant, name, M, N)` mirrors the layout of
`benchmarks_colinker/hoyas_grandes_wecc/`:

```
<out_dir>/
├── README.md                              suite overview + workflow + run_all_tests
├── common/
│   ├── run_all_tests.py                   drives every test for every stack (sudo)
│   ├── monitor.py                         live POI P/Q dashboard (read-only)
│   ├── plot_pq_compare.py                 cross-protocol P-Q overlay
│   └── compute_closed_loop_metrics.py     settling time + overshoot table
├── modbus/
│   ├── <name>.json                        plant JSON (protocol="modbus" on each device)
│   ├── run.py                             sudo-friendly Mininet launcher (--mitm-serve, --pcap-dir)
│   ├── README.md                          stack-specific notes
│   └── tests/                             open_loop_step, closed_loop_step,
│       │                                  frequency_response, mitm_attack (+ lib/)
│       └── */{csv,plots}/                 populated at runtime
├── iec104/   ... same shape, protocol="iec104"
├── dnp3/     ... same shape, protocol="dnp3"
├── dds/      ... same shape, protocol="dds"
└── local/    direct pydae-HTTP (run.sh, no Mininet) + same tests/ suite
```

The pydae seed (`xy_0.json`) is not shipped — `pydae_sim` writes it at
model-build time.  Post-run analysis artifacts (`packet_analysis.md`,
`closed_loop_metrics.md`, …) are produced by running the suite, not scaffolded.

The plant JSON is identical across protocols; only the per-device
`protocol` field varies. This works because the shared `configs` block
already carries both modbus and IEC 104 wire-level fields.

### Driving the suite

The whole matrix (every test for every stack + `local`) runs in one shot from
the Linux VM:

```bash
cd <out_dir>
sudo python3 common/run_all_tests.py            # all stacks
sudo python3 common/run_all_tests.py --only-mitm # just the over-power MITM phase
python3 common/monitor.py                        # live POI P/Q dashboard (separate terminal)
```

Or drive a single stack by hand:

```bash
# Terminal A — launch the cosim (stays attached to mininet>)
cd <out_dir>/<protocol>/ && sudo python3 run.py

# Terminal B — one test against it
cd <out_dir>/<protocol>/ && $PY tests/closed_loop_step/run.py
```

See the generated `<out_dir>/README.md` for the full workflow (MITM serve mode,
pcap capture, outputs, and how to modify parameters).

---

## Parameters of `designer()`

| Parameter | Default | What it controls |
|---|---|---|
| `M` | — | Number of feeders |
| `N` | — | Inverters per feeder |
| `output_file` | `'.'` (writes `pv_<M>_<N>.json`) | Path of the flat plant JSON |
| `S_base` | `100e6` (100 MVA) | pydae system MVA base — arbitrary, decoupled from nameplate |
| `P_inv_max` | `3e6` | Per-inverter nameplate (W) |
| `U_lv` | `800` | Inverter LV voltage (V) |
| `U_mv` | `20e3` | Feeder MV voltage (V) |
| `U_hv` | `132e3` | Transmission HV voltage (V) — POI and GRID |
| `ip_prefix` | `'10.10'` | SCADA plane IP prefix (inverters at `10.10.N.F/16`) |
| `Pref_pq` | `0.5` | REEC_B active-power setpoint (pu of S_n) |
| `Qext_pq` | `0.1` | REEC_B reactive-power setpoint (pu of S_n) |

---

## `make_xy_0(plant)`

Returns a dict of state seeds matching the REEC_B PQ-mode init recipe:

- Bus voltages `V=1.0`, `theta=0.0`.
- Line / transformer flows `±0.1 pu` per endpoint pair (placeholder —
  refined by Newton).
- `omega_coi = 1.0`; for every `genape` source: `omega_<bus>=1.0`,
  `v_ref_filtered_<bus>=1.0`.
- Per WECC `<id>`:
  - `Ip_<id> = Ipcmd`, `Iq_<id> = −Iqcmd` (sign per regc_a convention),
  - `x_Pe = P_flt = Ipcmd_<id> = Ipcmd`,
  - **`x_Kqi_<id> = 1.0`** when `Vflag = 0` (the Q-PI integrator output
    equals `V_t0`, which is the `V_ref_ctrl` the inner V-PI matches),
  - `x_Kqi_<id> = 0` when `Vflag = 1` (Q-PI is decoupled),
  - `x_Kvi_<id> = Vl_flt = Iqcmd_<id> = Iqcmd`.
- Per PLL `<bus>`: `theta_pll=0`, `xi_pll=0`, `omega_pll_f=1`,
  `rocof_pll_f=0`, `rocof_pll=0`.

The `x_Kqi` seed is the most consequential one — seeding it to `0` in
Q-control mode leaves the model far from steady state and the PF init
either NaNs out or settles on `Q ≈ 0` instead of `Q = Qext`.

---

## `topology.py`

Renders a side-by-side view: top panel = electrical (buses, lines,
transformers, source, WECC PV triangles), bottom panel = comms (hosts,
switches, links). Reads position fields (`pos_x` / `pos_y`) from the
JSON when present and falls back to algorithmic layout otherwise.

CLI:

```bash
python -m pydae.bps.builders.pv.topology --json plant.json --out plant.svg
```

Options:

| Flag | Default | Effect |
|---|---|---|
| `--json` | required | Input plant JSON |
| `--out` | `<basename>.svg` next to the JSON | Output path (SVG / PNG / PDF) |
| `--show` | off | Also open the figure interactively |

Programmatic:

```python
from pydae.bps.builders.pv import draw_topology

fig = draw_topology("plant.json", out_path="plant.png")
# fig is the matplotlib Figure — caller may close it.
```

---

## Sanity-checking the plant

The `__main__` block of `pv_designer.py` finishes by running the plant
through pydae as a smoke test:

```python
from pydae.bps import BpsBuilder
from pydae.core.builder.casadi_builder import CasadiBuilder
from pydae.core.model.casadi_model import CasadiModel
from pydae.bps.utils.reporter import report_all

grid  = BpsBuilder(plant_json, use_casadi=True)
grid.construct(plant_json)
bld   = CasadiBuilder(grid.sys_dict).build()
model = CasadiModel(bld)
model.ini({}, xy_0_json)
report_all(model, plant_json)
```

If the rootfinder NaNs out (`rf:g failed: NaN detected …`) the most
common culprits are:

1. `reec.Pref` ≠ `regc_a.Ipcmd` or `reec.Qext` ≠ `regc_a.Iqcmd` — the
   PF init starts far from steady state.
2. `Imax` / `Iqhl` clamps hugging the operating point — loosen them
   (`Imax=2.0`, `Iqhl=±2.0`) for init headroom.
3. `Vflag=0/Qflag=0` (rank-deficient — REEC_B's `x_Kvi` has no
   defining equation in this mode). Build aborts with
   `sprank(J) < n`.

See `docs/pydae-bps/models/weccs/reec_b.md` for the full mode matrix
and the supported flag combinations.
