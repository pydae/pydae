"""Canonical PV-plant designer for pydae + colinker.

Produces:

- a single pydae-compatible plant JSON (electrical + comms + cosim configs)
- a full benchmark folder tree ready to be driven by colinker: four networked
  SCADA stacks (``modbus`` / ``iec104`` / ``dnp3`` / ``dds``) plus a no-Mininet
  ``local`` case.  Every stack carries the structured ``tests/`` suite
  (open-/closed-loop step, frequency-response Bode, over-power MITM) copied from
  ``templates/tests/``, and the tree carries a ``common/`` folder with the
  ``run_all_tests.py`` orchestrator (+ monitor / plot / metrics).  Run scripts,
  common tooling and the top README are stamped from files under ``templates/``.

The pydae Newton seed (``xy_0.json``) is *not* shipped in the scaffold —
``colinker.pydae_sim`` writes it at model-build time.

This module is the consolidated successor of the earlier
``pv_plant.py`` (templates only) and ``utils/pv_builder/pv_designer.py``
(electrical model + cable sizing). Everything lives in one place now.

Usage::

    from pydae.bps.builders.pv_designer import designer, write_suite, make_xy_0

    plant = designer(M=5, N=10, output_file='copv_5_10.json',
                     P_inv_max=1e6, U_mv=30e3, U_hv=132e3,
                     Pref_pq=0.5, Qext_pq=0.1)
    write_suite('./copv_5_10_wecc', plant, name='copv_5_10', M=5, N=10)

Or just::

    python -m pydae.bps.builders.pv_designer
"""

import glob
import json
import os
import shutil
import sys
import tempfile

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))


# Must mirror INV_HOP in _apply_link_delays; lifted here so
# _modbus_timeout_for_hops can size the Modbus client timeout from the same
# per-hop latency budget.
_INV_HOP_MS = [(5, 2), (8, 3), (10, 3), (15, 5), (18, 5),
               (20, 6), (22, 6), (25, 7)]


def _pydae_chunk_ms_for_size(M: int, N: int) -> int:
    """Pick a pydae simulator chunk size (ms) that keeps the integrator
    above real time as the plant grows.  Baseline: 20 ms per 25-inverter
    "unit"; larger plants get proportionally bigger chunks so IDAS setup cost
    stays amortised.  Concrete tiers:

      M×N ≤ 25 → 20 ms   (unchanged — matches all pre-existing plants)
      26–50    → 40 ms
      51–75    → 60 ms
      76+      → 80 ms
    """
    n_inv = max(1, M * N)
    return 20 * ((n_inv - 1) // 25 + 1)


def _modbus_timeout_for_hops(N: int, with_link_delays: bool) -> float:
    """Recommend a Modbus TCP client timeout (seconds) that covers TCP connect
    through the deepest inverter in a branch.  ~3x the accumulated peak RTT
    (one-way delay + 2σ jitter) to the N-th inverter, floored at 0.15 s.
    When ``with_link_delays`` is False the topology is instantaneous, so the
    old 0.15 s default suffices."""
    if not with_link_delays:
        return 0.15
    n = min(max(N, 1), len(_INV_HOP_MS))
    trunk_ms   = 3.0                                   # PPC↔sPOI + host last-mile
    one_way_ms = trunk_ms + sum(d for d, _ in _INV_HOP_MS[:n])
    jitter_ms  = sum(j for _, j in _INV_HOP_MS[:n])
    peak_rtt_s = 2.0 * (one_way_ms + 2.0 * jitter_ms) / 1000.0
    return round(max(0.15, 3.0 * peak_rtt_s), 3)


# ---------------------------------------------------------------------------
# Electrical + comms generator
# ---------------------------------------------------------------------------

def designer(M, N, output_file='.', S_base=100e6, P_inv_max=3e6,
             U_lv=800, U_mv=20e3, U_hv=132e3, ip_prefix='10.10',
             Pref_pq=0.5, Qext_pq=0.1, with_link_delays=False):
    """Build the plant dict, write it to ``output_file``, and return it.

    Topology (per the user-supplied spec):

        GRID  (HV slack, ``genape`` source)
          |  POI-GRID line: X = 0.02 pu, R = 0.002 pu (in S_trafo_poi base)
          |  S_trafo_poi = 1.1 * M * N * P_inv_max
        POI   (HV)
          |  plant step-up POIMV<->POI: X = 0.1 pu, R = 0 (in S_trafo_poi base)
        POIMV (MV main)
          |  per-feeder MV cable sized from short-circuit current and
          |  thermal-tube rating (see cable table below)
        MV<id> ... MV<id>  (daisy-chain along the feeder)
          |  per-inverter step-up trafo LV<id><->MV<id>:
          |    S = 1.2 * P_inv_max, X = 0.05 pu, R = 0.01 pu (own base)
        LV<id>
          |  regc_a + REEC_B inverter in PQ mode
          |    Vflag=0 (Q-PI engaged), Qflag=1 (inner V-PI engaged)
          |    Ipcmd/Iqcmd match reec.Pref/Qext
    """
    S_inv_max = P_inv_max
    S_gens    = S_inv_max * N * M
    S_plant   = S_gens

    Irrad_max         = 1000     # W / m^2
    Area_form_factor  = 1.0
    PV_efficiency     = 0.1

    Z_trafo_poi_pu = 0.1
    S_trafo_poi    = M * N * S_inv_max * 1.1

    Area_inv = P_inv_max / (Irrad_max * PV_efficiency)
    Long_m   = np.sqrt(Area_inv / Area_form_factor)
    Long_n   = Area_form_factor * Long_m

    S_grid_line = S_plant

    data = {
        "system": {
            "name": f"pv_{M}_{N}",
            "S_base": S_base,
            "K_p_agc": 0.0,
            "K_i_agc": 0.0,
            "K_xif": 0.01,
        },
        "pydae_sim": {
            "chunk_ms":    _pydae_chunk_ms_for_size(M, N),
            "ramp_chunks": 2,
        },
        "buses": [],
        "lines": [
            {"bus_j": "POI", "bus_k": "GRID",
             "X_pu": 0.02, "R_pu": 0.002, "Bs_pu": 0.0,
             "S_mva": S_grid_line / 1e6,
             "sym": True, "monitor": True},
        ],
        "transformers": [
            {"bus_j": "POIMV", "bus_k": "POI",
             "X_pu": 0.1, "R_pu": 0.0, "Bs_pu": 0.0,
             "S_mva": S_grid_line / 1e6},
        ],
        "weccs": [],
        "sources": [{
            "type": "genape", "bus": "GRID",
            "S_n": S_plant * 100, "F_n": 50.0,
            "X_v": 0.001, "R_v": 0.0,
            "K_delta": 0.001, "K_alpha": 1e-6,
        }],
    }

    cables = [
        {'name':  '1X95',  'R_dc_km': 0.320,  'R_ac_km': 0.403,  'X_km': 0.128, 'muC_km': 0.187, 'R_0_km': 1.050, 'X_0_km': 0.391, 'muC_0_km': 0.187,  'I_max_aire': 255, 'I_max_direct': 205, 'I_max_tube': 190, 'I_cc_max':  8930, 'I_cc_0_max': 3140},
        {'name':  '1X150', 'R_dc_km': 0.206,  'R_ac_km': 0.262,  'X_km': 0.119, 'muC_km': 0.216, 'R_0_km': 0.890, 'X_0_km': 0.341, 'muC_0_km': 0.216,  'I_max_aire': 335, 'I_max_direct': 260, 'I_max_tube': 245, 'I_cc_max': 14100, 'I_cc_0_max': 3470},
        {'name':  '1X240', 'R_dc_km': 0.125,  'R_ac_km': 0.161,  'X_km': 0.109, 'muC_km': 0.260, 'R_0_km': 0.768, 'X_0_km': 0.297, 'muC_0_km': 0.260,  'I_max_aire': 455, 'I_max_direct': 345, 'I_max_tube': 320, 'I_cc_max': 22600, 'I_cc_0_max': 3810},
        {'name':  '1X400', 'R_dc_km': 0.0778, 'R_ac_km': 0.102,  'X_km': 0.102, 'muC_km': 0.313, 'R_0_km': 0.650, 'X_0_km': 0.237, 'muC_0_km': 0.313,  'I_max_aire': 610, 'I_max_direct': 445, 'I_max_tube': 415, 'I_cc_max': 37600, 'I_cc_0_max': 4300},
        {'name':  '1X500', 'R_dc_km': 0.0605, 'R_ac_km': 0.084,  'X_km': 0.099, 'muC_km': 0.329, 'R_0_km': 0.618, 'X_0_km': 0.225, 'muC_0_km': 0.329,  'I_max_aire': 715, 'I_max_direct': 505, 'I_max_tube': 480, 'I_cc_max': 47000, 'I_cc_0_max': 4810},
        {'name':  '1X630', 'R_dc_km': 0.0469, 'R_ac_km': 0.0636, 'X_km': 0.095, 'muC_km': 0.396, 'R_0_km': 0.561, 'X_0_km': 0.195, 'muC_0_km': 0.396,  'I_max_aire': 830, 'I_max_direct': 575, 'I_max_tube': 545, 'I_cc_max': 59200, 'I_cc_0_max': 5140},
    ]

    Z_base_trafo_poi = U_mv ** 2 / S_trafo_poi
    Z_trafo_poi      = Z_base_trafo_poi * Z_trafo_poi_pu
    I_cc_max         = U_mv / (np.sqrt(3) * Z_trafo_poi)

    for it, item in enumerate(cables):
        if item['I_cc_max'] > I_cc_max:
            print(f"Short circuit designed cable ({I_cc_max/1e3:0.2f} kA): "
                  f"{item['name']}")
            idx_cc = it
            break

    I_max = 1.0 * N * S_inv_max / (np.sqrt(3) * U_mv)
    for it, item in enumerate(cables):
        if item['I_max_tube'] > I_max:
            print(f"Nominal current designed cable  ({I_max:0.2f} A): "
                  f"{item['name']}")
            idx_n = it
            break

    idx_cable = idx_cc if idx_cc > idx_n else idx_n
    print(f"Cable: {cables[idx_cable]['name']}")

    R_km_cable = cables[idx_cable]['R_ac_km']
    X_km_cable = cables[idx_cable]['X_km']
    B_km_cable = 2 * np.pi * 50 * cables[idx_cable]['muC_km'] * 1e-6

    pos_mv_poi_m_pu = 0.0
    pos_mv_poi_m    = pos_mv_poi_m_pu * (M - 1) * Long_m
    pos_mv_poi_n    = Long_n * (N - 1) / 2

    data['buses'].append({
        "name": "POIMV", "P_W": 0.0, "Q_var": 0.0, "U_kV": U_mv / 1e3,
        "pos_x": pos_mv_poi_m,        "pos_y": pos_mv_poi_n,
    })
    data['buses'].append({
        "name": "POI",   "P_W": 0.0, "Q_var": 0.0, "U_kV": U_hv / 1e3,
        "pos_x": pos_mv_poi_m - 40,   "pos_y": pos_mv_poi_n,
    })
    data['buses'].append({
        "name": "GRID",  "P_W": 0.0, "Q_var": 0.0, "U_kV": U_hv / 1e3,
        "pos_x": pos_mv_poi_m - 200,  "pos_y": pos_mv_poi_n,
    })

    for i_m in range(1, M + 1):
        name_j = "POIMV"
        monitor = True
        pos_m = (i_m - 1) * Long_m

        for i_n in range(1, N + 1):
            name = f"{i_m}".zfill(2) + f"{i_n}".zfill(2)
            name_k = 'MV' + name

            if i_n == 1:
                Long = np.abs(pos_mv_poi_m - pos_m) + Long_n * 0.5
            else:
                Long = Long_n

            pos_n = (i_n - 1) * Long_n + Long_n / 2

            data['buses'].append({
                "name": f"LV{name}", "P_W": 0.0, "Q_var": 0.0,
                "U_kV": U_lv / 1e3, "pos_y": pos_m + 10, "pos_x": pos_n,
            })
            data['buses'].append({
                "name": f"MV{name}", "P_W": 0.0, "Q_var": 0.0,
                "U_kV": U_mv / 1e3, "pos_y": pos_m, "pos_x": pos_n,
            })

            print(f'pos_m = {pos_m:5.1f}, pos_n = {pos_n:5.1f}')

            # LV-MV trafo
            S_trafo_n = 1.2 * S_inv_max
            data['lines'].append({
                "bus_j": f"LV{name}", "bus_k": f"MV{name}",
                "X_pu": 0.05, "R_pu": 0.01, "Bs_pu": 0.0,
                "S_mva": S_trafo_n / 1e6, "monitor": False,
            })

            # MV cable
            Z_b = U_mv ** 2 / S_base
            R_cable_pu = R_km_cable * Long / 1e3 / Z_b
            X_cable_pu = X_km_cable * Long / 1e3 / Z_b
            B_cable_pu = B_km_cable * Long / 1e3 * Z_b
            data['lines'].append({
                "bus_j": name_k, "bus_k": name_j,
                "X_pu": X_cable_pu, "R_pu": R_cable_pu, "Bs_pu": B_cable_pu,
                "S_mva": S_base / 1e6, "monitor": monitor,
            })
            name_j = name_k

            # WECC inverter in PQ-generator mode per the REEC_B docs:
            # Vflag=0 (Q-PI engaged) + Qflag=1 (inner V-PI engaged). The outer
            # regc_a Ipcmd / Iqcmd must agree with reec.Pref / reec.Qext or PF
            # init will sit far from steady state and Newton may NaN.
            wecc = {
                "type":   "regc_a",
                "name":   f"LV{name}",
                "bus":    f"LV{name}",
                "S_n":    S_inv_max,
                "Tg":     0.02,
                "Tfltr":  0.02,
                "Lvplsw": 1,
                "Lvpl1":  1.22,
                "Zerox":  0.4,
                "Brkpt":  0.9,
                "rrpwr":  10.0,
                "lvpnt0": 0.4,
                "lvpnt1": 0.8,
                "Volim":  1.2,
                "Iolim": -1.3,
                "Khv":    0.7,
                "Iqrmax":  999.0,
                "Iqrmin": -999.0,
                "Ipcmd":  Pref_pq,        # match reec.Pref
                "Iqcmd":  Qext_pq,        # match reec.Qext
                "V_ini":  1.0,
                "reec": {
                    "Vflag":  0,           # Q control (Kqp/Kqi engaged)
                    "Qflag":  1,           # inner V-PI engaged
                    "PFflag": 0,           # Qext as Q setpoint (not PF*P)
                    "Pqflag": 1,           # P priority on the current limit
                    "Trv":    0.02,
                    "Tp":     0.02,
                    "Tiq":    0.02,
                    "Tpord":  0.1,
                    "Kvp":    1.0,
                    "Kvi":   40.0,
                    "Kqp":    1.0,
                    "Kqi":   10.0,
                    "Vref0":  1.0,
                    "dbd1":  -0.99,
                    "dbd2":   0.99,
                    "Kqv":    0.0,
                    "Iqhl":   2.0,
                    "Iqll":  -2.0,
                    "Vmax":   1.1,
                    "Vmin":   0.9,
                    "Imax":   2.0,
                    "Pmax":   1.0,
                    "Pmin":  -0.1,
                    "dPmax":  999.0,
                    "dPmin": -999.0,
                    "Pref":   Pref_pq,
                    "Qext":   Qext_pq,
                },
            }
            data['weccs'].append(wecc)
            monitor = False

    # Merge in the comms-side template (hosts / switches / links / configs).
    with open(os.path.join(_HERE, 'template_mininet_local_modbus.json'),
              'r') as fobj:
        com_dev_data = json.loads(fobj.read())

    # Every device is its own Modbus outstation (own IP:port), so their
    # register spaces are isolated — reg_0 does NOT need to be globally
    # unique.  Using a fixed low value keeps ``reg_abs = reg_0 + register``
    # inside the 16-bit Modbus wire address space regardless of M×N; the
    # earlier ``reg_0 += 1000`` scheme silently overflowed uint16 past ~63
    # devices, causing pymodbus to reject every write with
    # ``'H' format requires 0 <= number <= 65535``.
    reg_0 = 1000
    com_dev_data['devices'] += [{
        "ing_id": "poi", "ing_type": "poi", "emec_id": "POI",
        "modbus_ip": "10.10.0.5", "modbus_port": 502,
        "iec104_ip": "10.10.0.5", "iec104_port": 2404,
        "config": "poi", "reg_0": reg_0, "linker_Dt": 0.1,
        "protocol": "modbus", "common_address": 40,
    }]

    for i_m in range(1, M + 1):
        for i_n in range(1, N + 1):
            name = f"{i_m}".zfill(2) + f"{i_n}".zfill(2)
            com_dev_data['devices'] += [{
                "ing_id": f"inv{name}", "ing_type": "inverter",
                "emec_id": f"LV{name}",
                "modbus_ip": f"{ip_prefix}.{i_n}.{i_m}", "modbus_port": 502,
                "iec104_ip": f"{ip_prefix}.{i_n}.{i_m}", "iec104_port": 2404,
                "config": "inv", "reg_0": reg_0, "linker_Dt": 0.1,
                "protocol": "modbus", "common_address": 41,
            }]
            i_m_00, i_n_00 = f"{i_m}".zfill(2), f"{i_n}".zfill(2)
            com_dev_data['hosts'] += [{
                "name": f"LV{name}",
                "ip":  f"10.10.{i_m}.{i_n}/16",
                "mac": f"00:00:00:00:{i_m_00}:{i_n_00}",
                "ip_2": f"10.20.{i_m}.{i_n}/16",
                "run_linker": True,
            }]
            com_dev_data['switches'] += [{"name": f"sLV{name}"}]
            com_dev_data['links'] += [
                {"node_j": f"LV{name}", "node_k": f"sLV{name}"},
                {"node_j": f"LV{name}", "node_k":  "sEEMU"},
            ]

    for i_m in range(1, M + 1):
        node_j = 'sPOI'
        for i_n in range(1, N + 1):
            i_m_00, i_n_00 = f"{i_m}".zfill(2), f"{i_n}".zfill(2)
            node_k = f"sLV{i_m_00}{i_n_00}"
            com_dev_data['links'] += [{"node_j": node_j, "node_k": node_k}]
            node_j = node_k

    # PPC controller block — generalized N-inverter PI loop wiring.
    pv_ids   = [f"LV{f'{i_m}'.zfill(2)}{f'{i_n}'.zfill(2)}"
                for i_m in range(1, M + 1) for i_n in range(1, N + 1)]
    target_mw_default = 0.5 * (M * N) * (P_inv_max / 1e6)
    com_dev_data['ppc'] = {
        "ip":          "10.10.0.4",
        "api_ip":      "10.20.0.4",
        "api_port":    5500,
        "poi_id":      "POI",
        "pv_ids":      pv_ids,
        "target_mw":   target_mw_default,
        "target_mvar": 0.0,
        # Stable defaults validated against the Modbus cosim (10× WECC inverters,
        # ~100 ms/write transport delay).  The aggressive K_ip=2.0 / ctrl_dt=0.02
        # set oscillates at cold start through that delay; these settle cleanly.
        "ctrl_dt_s":   0.1,
        "modbus_timeout_s": _modbus_timeout_for_hops(N, with_link_delays),
        "p_max_mw":    P_inv_max / 1e6,
        "p_min_mw":    0.0,
        "q_max_mvar":  0.5 * P_inv_max / 1e6,
        "q_min_mvar": -0.5 * P_inv_max / 1e6,
        "K_pp":        0.1,
        "K_ip":        0.35,
        "K_pq":        0.1,
        "K_iq":        0.35,
    }

    if with_link_delays:
        _apply_link_delays(com_dev_data, M, N)

    data.update(com_dev_data)
    formatted_json = json.dumps(data, indent=4, separators=(',', ':'))

    if output_file == '.':
        output_file = f'pv_{M}_{N}.json'
    out_parent = os.path.dirname(output_file)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    print(f'Writing file: {output_file}')
    with open(output_file, 'w') as fobj:
        fobj.write(formatted_json)
    return data


def _apply_link_delays(com_dev_data, M, N):
    """Decorate Mininet links with realistic SCADA-grade delay + jitter.

    Profile (mirrors the ecn benchmark):
      - PPC ↔ sPOI:    2 ms ± 0.5 ms (control cabinet trunk)
      - PPC ↔ sPPC:    1 ms ± 0.2 ms (management plane)
      - POI ↔ sPOI:    1 ms ± 0.2 ms (POI host last-mile)
      - MITM/CPN/SCADA ↔ sPOI: 1 ms ± 0.2 ms
      - LV<id> ↔ sLV<id>:      1 ms ± 0.2 ms (NIC + Ethernet)
      - sPOI ↔ first sLV<id_i_n=1> on each feeder, then sLV chain:
            delay grows with inverter index ``i_n`` so that the lag
            difference between closest and furthest converter is
            ~17 ms — large enough to see distinct arrival patterns.
      - *↔ sEEMU:   no delay/jitter — emulation side, not SCADA plane.
    """
    HOST_HOP   = ("1ms",  "0.2ms")
    PPC_TRUNK  = ("2ms",  "0.5ms")
    # Per-inverter index → (delay, jitter) within a feeder
    INV_HOP = {
        1: ("5ms",  "2ms"),
        2: ("8ms",  "3ms"),
        3: ("10ms", "3ms"),
        4: ("15ms", "5ms"),
        5: ("18ms", "5ms"),
        6: ("20ms", "6ms"),
        7: ("22ms", "6ms"),
        8: ("25ms", "7ms"),
    }

    def _sLV_i_n(name):
        # sLV<i_m_00><i_n_00> → i_n
        try:
            return int(name[5:7])
        except Exception:
            return None

    for lnk in com_dev_data['links']:
        j, k = lnk['node_j'], lnk['node_k']

        # sEEMU side is the power-system emulation plane — no delay or jitter
        if j == "sEEMU" or k == "sEEMU":
            continue

        # PPC ↔ sPOI trunk
        if {j, k} == {"PPC", "sPOI"}:
            lnk['delay'], lnk['jitter'], lnk['loss'] = PPC_TRUNK[0], PPC_TRUNK[1], 0
            continue

        # sPOI ↔ sLV0X01 (first feeder hop): use INV_HOP[1]
        if {j, k} == {"sPOI", j if j.startswith("sLV") else k} and \
           (j.startswith("sLV") or k.startswith("sLV")):
            lv_name = j if j.startswith("sLV") else k
            i_n = _sLV_i_n(lv_name)
            if i_n is not None:
                d, jit = INV_HOP.get(i_n, INV_HOP[1])
                lnk['delay'], lnk['jitter'], lnk['loss'] = d, jit, 0
                continue

        # sLV<a> ↔ sLV<b> (intra-feeder chain hop): per-position delay
        # of the *higher-index* end so the cumulative budget grows with i_n.
        if j.startswith("sLV") and k.startswith("sLV"):
            in_j = _sLV_i_n(j)
            in_k = _sLV_i_n(k)
            if in_j is not None and in_k is not None:
                i_n = max(in_j, in_k)
                d, jit = INV_HOP.get(i_n, INV_HOP[1])
                lnk['delay'], lnk['jitter'], lnk['loss'] = d, jit, 0
                continue

        # All other host last-mile / mgmt links
        lnk['delay'], lnk['jitter'], lnk['loss'] = HOST_HOP[0], HOST_HOP[1], 0


# ---------------------------------------------------------------------------
# pydae steady-state seed
# ---------------------------------------------------------------------------

def make_xy_0(plant):
    """pydae Newton-solver seed for a ``designer()`` plant.

    Per the REEC_B PQ-mode initialization documented in
    ``docs/pydae-bps/models/weccs/reec_b.md``:

    - Bus voltages V = 1.0, theta = 0.0
    - Line / transformer flows +-0.1 placeholder per endpoint pair
    - omega_coi = 1.0; genape source: omega_<bus> = v_ref_filtered_<bus> = 1.0
    - regc_a / reec_b states derived from each WECC's Ipcmd / Iqcmd:
        * Ip = Ipcmd, Iq = -Iqcmd  (sign per regc_a convention)
        * x_Pe = P_flt = Ipcmd, Ipcmd_<id> = Ipcmd
        * x_Kqi = V_t0 = 1.0 when Vflag = 0 (Q-PI is the upstream regulator)
                 = 0   when Vflag = 1 (Q-PI decoupled)
        * x_Kvi = Vl_flt = Iqcmd_<id> = Iqcmd
    """
    out = {}
    for b in plant.get("buses", []):
        out[f"V_{b['name']}"]     = 1.0
        out[f"theta_{b['name']}"] = 0.0
    for ln in (*plant.get("lines", []), *plant.get("transformers", [])):
        j, k = ln["bus_j"], ln["bus_k"]
        out[f"p_line_pu_{j}_{k}"] =  0.1
        out[f"p_line_pu_{k}_{j}"] = -0.1
        out[f"I_{j}_{k}"]         =  0.1
        out[f"I_{k}_{j}"]         = -0.1
    out["omega_coi"] = 1.0
    for s in plant.get("sources", []):
        bus = s["bus"]
        if s.get("type") == "genape":
            out[f"omega_{bus}"]          = 1.0
            out[f"v_ref_filtered_{bus}"] = 1.0
    for w in plant.get("weccs", []):
        nm    = w["name"]
        Ipcmd = float(w.get("Ipcmd", 0.0))
        Iqcmd = float(w.get("Iqcmd", 0.0))
        reec  = w.get("reec", {})
        Vflag = int(reec.get("Vflag", 1))
        # x_Kqi is the Q-PI integrator output. In Q-control mode (Vflag=0) it
        # IS the V_ref_ctrl that the inner V-PI matches to V_t at steady
        # state -- seed to V_t0 = 1.0, NOT 0, or the PF init will either NaN
        # or settle on Q ~ 0 instead of Q = Qext.
        x_Kqi_seed = 1.0 if Vflag == 0 else 0.0
        out[f"V_flt_{nm}"]  = 1.0
        out[f"Vt_flt_{nm}"] = 1.0
        out[f"Vl_flt_{nm}"] = Iqcmd
        out[f"Ip_{nm}"]     = Ipcmd
        out[f"Iq_{nm}"]     = -Iqcmd
        out[f"Ipcmd_{nm}"]  = Ipcmd
        out[f"Iqcmd_{nm}"]  = Iqcmd
        out[f"x_Pe_{nm}"]   = Ipcmd
        out[f"P_flt_{nm}"]  = Ipcmd
        out[f"x_Kqi_{nm}"]  = x_Kqi_seed
        out[f"x_Kvi_{nm}"]  = Iqcmd
    for pll in plant.get("plls", []):
        bus = pll["bus"]
        out[f"theta_pll_{bus}"]   = 0.0
        out[f"xi_pll_{bus}"]      = 0.0
        out[f"omega_pll_f_{bus}"] = 1.0
        out[f"rocof_pll_f_{bus}"] = 0.0
        out[f"rocof_pll_{bus}"]   = 0.0
    return out


# ---------------------------------------------------------------------------
# Four-protocol benchmark suite scaffold
# ---------------------------------------------------------------------------
#
# Mirrors the layout of ``benchmarks_colinker/hoyas_grandes_wecc/``::
#
#     <out_dir>/
#         README.md
#         modbus/  -> <name>.json, xy_0.json, <name>_xy_0.json,
#                     run.py, run_step.sh, README.md, results/
#         iec104/  -> ...
#         dnp3/    -> ...
#         dds/     -> ...
#         reporter/report.ipynb
#
# Run.py targets the standard Linux VM layout (pydae_dev conda env,
# COLINKER_PYTHON, sudo-rs). Edit the constants inside the emitted run.py (or
# the templates/run_py/<proto>.py sources) if your paths differ.

# IEC 104 runs the PPC control loop slower than the other stacks (which use the
# designer default 0.1 s): its event-driven c104 link cannot sustain a 10 Hz
# loop without saturating / oscillating.  See iec104_saturation.md.
_IEC104_CTRL_DT_S = 1.0

# Protocols whose proxies put engineering units on the wire (wire = pydae *
# emec_scale) and whose PPC reads them directly — power measurement points
# therefore need emec_scale=1e-6 (W→MW).  Modbus keeps raw W (scale 1.0) and
# divides in software, so it is excluded.
_ENGINEERING_UNIT_WIRE_PROTOCOLS = ("iec104", "dnp3")

# ---------------------------------------------------------------------------
# Emission templates live as files under templates/ (run_py/<proto>.py,
# common/*, local_run.sh, tests/*), synced from the known-good case folder and
# stamped by the builder — no giant embedded strings.
# ---------------------------------------------------------------------------
_TEMPLATES    = os.path.join(_HERE, "templates")
_RUN_PY_FILES = {p: os.path.join(_TEMPLATES, "run_py", f"{p}.py")
                 for p in ("modbus", "iec104", "dnp3", "dds")}
_MODBUS_REG_BASE = 1000   # POI device reg_0; wire register = base + point register


def _load(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _stamp(text, **subs):
    for k, v in subs.items():
        text = text.replace(k, str(v))
    return text


def _write_common(out_dir, name, py_interp, mitm_cap_mw):
    """Emit the common/ orchestration folder: run_all_tests.py (stamped with the
    interpreter path + this plant's MITM cap) plus monitor.py, plot_pq_compare.py
    and compute_closed_loop_metrics.py (all name-agnostic — they derive ROOT from
    their own location and discover the plant JSON)."""
    dst  = os.path.join(out_dir, "common")
    cdir = os.path.join(_TEMPLATES, "common")
    os.makedirs(dst, exist_ok=True)
    for fn in ("monitor.py", "plot_pq_compare.py", "compute_closed_loop_metrics.py"):
        shutil.copy(os.path.join(cdir, fn), os.path.join(dst, fn))
    rat = _stamp(_load(os.path.join(cdir, "run_all_tests.py")),
                 __PY_INTERP__=py_interp, __MITM_CAP_MW__=repr(float(mitm_cap_mw)))
    with open(os.path.join(dst, "run_all_tests.py"), "w", encoding="utf-8") as f:
        f.write(rat)
    for fn in ("run_all_tests.py", "monitor.py", "plot_pq_compare.py",
               "compute_closed_loop_metrics.py"):
        os.chmod(os.path.join(dst, fn), 0o755)


def _readme_top(name: str, M: int, N: int) -> str:
    """Top-level suite README (workflow, run_all_tests.py, MITM/pcap, outputs,
    how to modify parameters) — loaded from templates/README_top.md and stamped
    with this plant's name / M / N."""
    return _stamp(_load(os.path.join(_TEMPLATES, "README_top.md")),
                  __NAME__=name, __M__=M, __N__=N, __MN__=M * N)


def _readme_modbus(M: int, N: int) -> str:
    return f'''# {{name}} — Modbus TCP benchmark

Pymodbus 3.13 (`ModbusTcpServer` + `SimDevice` / `SimData`).

## Modules launched

- `colinker.pydae_sim` (rtsim)
- `colinker.poi_proxy` (POI)
- `colinker.pv_proxy --name LV<F><N>` (one per inverter)
- `colinker.modbus_ppc` (PPC) — N-inverter PI loop with reference
  feedforward and the `/status` + `/setpoints` + `/sine` HTTP API.

## Run the cosim

```bash
sudo python3 run.py        # Terminal A — PPC API at http://localhost:8090
```

## Tests — `tests/`

Structured experiments, each a self-contained folder emitting CSV + SVG +
interactive Plotly HTML (see `tests/README.md`):

| Test | What it measures |
|---|---|
| `tests/open_loop_step/` | Plant + comms time response (gains=0, feedforward only). |
| `tests/closed_loop_step/` | Closed-loop PI step response — gain-design workspace. |
| `tests/frequency_response/` | Sinusoidal Bode sweeps: P-only, Q-only, P+Q at 90°. |
| `tests/mitm_attack/` | Modbus MITM — falsified PPC reading vs true plant (offset bias by default, fixed spoof optional). |

```bash
PY=/home/jmmmacubu/miniforge3/envs/pydae_dev/bin/python3   # Terminal B
$PY tests/closed_loop_step/run.py
$PY tests/frequency_response/run.py --mode p   # also --mode q, --mode pq90
```
'''


def _readme_iec104(M: int, N: int) -> str:
    return f'''# {{name}} — IEC 60870-5-104 benchmark

Same plant + topology as the Modbus variant, with IEC 60870-5-104
between field devices and the PPC. Uses the `c104` library (v2.2.1+).

## Required

```bash
pip install c104          # or: uv sync --extra iec104
```

## Modules launched

- `colinker.pydae_sim` (rtsim)
- `colinker.iec104_poi_proxy --emec-id POI` (POI)
- `colinker.iec104_pv_proxy --name LV<F><N>` (one per inverter)
- `colinker.iec104_ppc` (PPC) — c104 client driving the N-inverter PI
  loop

## Run the cosim

```bash
sudo python3 run.py        # Terminal A — PPC API at http://localhost:8090
```

## Tests — `tests/`

Same structured suite as the Modbus variant (shared `colinker.ppc_core`
control API), each folder emitting CSV + SVG + interactive Plotly HTML:

| Test | What it measures |
|---|---|
| `tests/open_loop_step/` | Plant + comms time response (gains=0, feedforward only). |
| `tests/closed_loop_step/` | Closed-loop PI step response — gain-design workspace. |
| `tests/frequency_response/` | Sinusoidal Bode sweeps: P-only, Q-only, P+Q at 90°. |
| `tests/mitm_attack/` | IEC 104 MITM — falsified PPC reading vs true plant (offset bias by default, fixed spoof optional). |

```bash
PY=/home/jmmmacubu/miniforge3/envs/pydae_dev/bin/python3   # Terminal B
$PY tests/closed_loop_step/run.py
$PY tests/frequency_response/run.py --mode p   # also --mode q, --mode pq90
```

## MITM attack + pcap forensics

```bash
sudo python3 run.py --no-cli --mitm \\           # default: +5% offset
     --pcap-dir results_overpower/pcaps          # capture PPC/POI/MITM (tcp/2404)
$PY tests/mitm_attack/run.py                      # record believed-vs-true
python -m colinker.iec104_pcap_report --pcap-dir results_overpower/pcaps \\
     --ioa 1001 --common-address 47 --scale-mw            # decode the wire
```

The attack rewrites the POI active-power `M_ME_NC_1` short float in flight,
keyed by IOA + common address. The pcap report shows the same IOA carrying
the real value on the POI link and the spoofed value on the PPC link (the
MITM vantage carries both legs).

## IEC 104 specifics (vs Modbus)

- **Slower control loop** — `ppc.ctrl_dt_s = {_IEC104_CTRL_DT_S}` s (Modbus runs 0.1 s).
  The event-driven c104 link can't sustain a 10 Hz sample/actuate without
  saturating the client thread and driving the (same-gain) loop into a limit
  cycle. Same PI gains, protocol-appropriate sample time. See
  `iec104_saturation.md`.
- **Wire carries engineering units** — power measurement points use
  `emec_scale = 1e-6` (W→MW on the wire), so MITM `--value`/`--offset` are in
  MW (no 1e6 factor, unlike the Modbus register path).
- **Common address** is fixed at 47 by `colinker.iec104_ppc` /
  `iec104_poi_proxy` (`COMMON_ADDR`), not the JSON device field.

## Gotcha — c104 callback annotations

`c104` inspects setpoint-callback signatures at runtime via
`__annotations__` and rejects stringified type hints. If you fork the
proxy modules and add `from __future__ import annotations`, setpoint
commands silently stop working.
'''


def _readme_dnp3(M: int, N: int) -> str:
    n_inv = M * N
    return f'''# {{name}} — DNP3 benchmark

Same plant + topology as the Modbus variant, with DNP3 between field
devices and the PPC. **No external Python DNP3 binding is required** —
`colinker.dnp3_codec` implements just the subset of the protocol the
control loop needs (~300 LOC).

Outstation addresses (master = 1): POI = 10, inverters = 11..{10 + n_inv}
(linear over feeder-major order).

## Modules launched

- `colinker.pydae_sim` (rtsim)
- `colinker.dnp3_poi_proxy --name POI --addr 10`
- `colinker.dnp3_pv_proxy --name LV<F><N> --addr <11+global_idx>` per inverter
- `colinker.dnp3_ppc` — one persistent TCP connection per outstation

## Run

```bash
sudo python3 run.py                         # launch the cosim
PY=/home/jmmmacubu/miniforge3/envs/pydae_dev/bin/python3
$PY tests/closed_loop_step/run.py           # structured tests/ suite
```

## Tests (`tests/`)

Same config-driven suite as `modbus/` and `iec104/` (open-/closed-loop step,
frequency-response Bode, MITM), all driving the shared `ppc_core` API.

| Test | Purpose |
|---|---|
| `tests/open_loop_step/` | Plant + comms time response (gains 0, feedforward only). |
| `tests/closed_loop_step/` | PI gain-design step response. |
| `tests/frequency_response/` | Sinusoidal sweeps → Bode (P / Q / PQ90). |
| `tests/mitm_attack/` | DNP3 MITM — falsified PPC reading vs true plant. |

## MITM attack

```bash
# over-power: plant driven ~10% above target, readback stays on-target
sudo python3 run.py --no-cli --mitm --mitm-offset-frac -0.0909 \\
     --mitm-start-after 30 --mitm-duration 120
$PY tests/mitm_attack/run.py                # self-timing; auto-detects the window
```

`colinker.attacks.dnp3_mitm` rewrites the POI active-power **Analog Input float
(Group 30 Var 5, object index 1)** in flight, then **re-frames the response with
fresh DLL CRCs** — a plain byte patch would corrupt the block CRC and black-hole
the POI reading (the PPC would see NaN, not a spoof). Wire carries engineering
units (`emec_scale = 1e-6`, W→MW), so MITM `--value`/`--offset` are in MW.
'''


def _readme_dds(M: int, N: int) -> str:
    return f'''# {{name}} — DDS benchmark

Same plant + topology as the Modbus variant, with DDS over UDP
multicast between field devices and the PPC. Uses `cyclonedds`.

```{{warning}}
**Incomplete stack.** Only the PPC half (`colinker.dds_ppc`) is in
tree. There is no `dds_poi_proxy` / `dds_pv_proxy` yet, so a full
end-to-end run does not work today — the PPC will start, discover no
`openfmb/PoiReading` publisher, and log "no PoiReading yet" until
something publishes the topic.
```

## Required

```bash
pip install cyclonedds          # or: uv sync --extra dds
```

## Mininet caveat — multicast

DDS discovery uses UDP multicast. If endpoints discover each other
(`*_matched` callbacks fire) but data never flows, try disabling IGMP
snooping on each SCADA-plane switch:

```bash
ovs-vsctl set Bridge <sSwitch> mcast_snooping_enable=false
```

## Run

```bash
sudo python3 run.py
./run_step.sh
```
'''


_README_FN = {
    "modbus": _readme_modbus,
    "iec104": _readme_iec104,
    "dnp3":   _readme_dnp3,
    "dds":    _readme_dds,
}


# ---------------------------------------------------------------------------
# local case — the controller talks straight to pydae over HTTP (no Mininet,
# no SCADA wire stack).  Same control core as the protocol PPCs, exercised
# against the bare plant model.  Emitted alongside the four protocol folders.
# ---------------------------------------------------------------------------

def _readme_local(M: int, N: int) -> str:
    return f'''# {{name}} — local (direct pydae HTTP, no Mininet)

A fifth "protocol" that isn't a SCADA wire stack at all: the controller talks
**straight to the pydae real-time simulator over HTTP**, with no proxies, no
Modbus/IEC104/DNP3/DDS, and **no Mininet**. It's for exercising and tuning PPC
behaviour against the bare plant model, fast, without the network cosim.

## What runs

- `colinker.pydae_sim` — the pydae real-time simulator, serving
  `GET /cosim/measurements` and `POST /cosim/setpoints` (default `:8000`).
- `colinker.pydae_ppc` — the controller. Same control core as every protocol
  PPC (`colinker.ppc_core`: velocity-form PI + reference feedforward +
  sinusoid, and the `/status` + `/setpoints` + `/sine` HTTP API), but its I/O
  reads POI power from `/cosim/measurements` (W → MW) and writes inverter
  setpoints to `/cosim/setpoints` (`pu = MW · 1e6 · emec_scale`, scale from the
  plant JSON).

Because there is no wire protocol or netns hop, this path has **no transport
delay** — it isolates the controller + plant dynamics from the SCADA comms.

## Run

```bash
# Terminal A — bring up pydae_sim + pydae_ppc
./run.sh                       # pydae on :8001, PPC API on :5500

# or by hand:
PY=/home/jmmmacubu/miniforge3/envs/pydae_dev/bin/python3
$PY -m colinker.pydae_sim --json {{name}}.json --port 8001
$PY -m colinker.pydae_ppc --json {{name}}.json \\
    --pydae-url http://localhost:8001 --api-host 0.0.0.0 --api-port 5500
```

The PPC management API is then at `http://localhost:5500/` (`/status`,
`/setpoints`, `/sine`) — identical surface to the protocol PPCs.

## Debugging model init

If the cosim fails to boot (`Connection refused` on the pydae port) run:

```bash
./ini.py                       # builds the pydae model and calls model.ini({{}})
```

which reproduces the pydae initialisation step in isolation — no FastAPI, no
proxies. On success it prints `OK — initialisation converged`; on failure it
exits non-zero with the IDAS / Newton traceback so you can see which algebraic
constraint the solver choked on.

## Tests

This folder carries its own copy of the [`tests/`](tests/) suite (same
config-driven drivers as `modbus/tests/`; `lib/common.py` auto-discovers
`{{name}}.json` here). Point them at the PPC via `PPC_URL`:

```bash
# Terminal B
PY=/home/jmmmacubu/miniforge3/envs/pydae_dev/bin/python3
PPC_URL=http://localhost:5500 $PY tests/frequency_response/run.py --mode p
PPC_URL=http://localhost:5500 $PY tests/closed_loop_step/run.py
PPC_URL=http://localhost:5500 $PY tests/open_loop_step/run.py
```

The `open_loop_step` / `closed_loop_step` drivers also record pydae ground
truth through `colinker.recorder`; in direct mode point the recorder's sim URL
at this `pydae_sim` via `SIM_URL=http://localhost:8001`.

## Notes

- Same `{{name}}.json` plant as the other folders; `pydae_ppc` only uses the
  electrical model + the `pydae_key`/`emec_scale` from the configs, so the
  Modbus device fields in the JSON are simply ignored.
- `--api-host` defaults to `0.0.0.0`; the JSON's `api_ip` is a Mininet-namespace
  address and is not used here.
- `mitm_attack/` is carried for parity with the protocol suites but is an
  on-the-wire attack — it has no effect in this no-wire path.
'''


def _write_local_case(out_dir, plant, name, M, N, json_filename, mitm_cap_mw):
    """Emit the local/ folder: direct pydae-HTTP controller (pydae_ppc), a
    run.sh launching pydae_sim + pydae_ppc, the plant JSON, the structured
    tests/ suite, and a README.  No Mininet, no wire protocol (the device-level
    protocol field is left as-is; pydae_ppc only reads the electrical model +
    pydae_key/emec_scale)."""
    sub = os.path.join(out_dir, "local")
    os.makedirs(sub, exist_ok=True)
    with open(os.path.join(sub, json_filename), "w", encoding="utf-8") as f:
        json.dump(plant, f, indent=2)
    run_sh = _stamp(_load(os.path.join(_TEMPLATES, "local_run.sh")),
                    __JSON_FILENAME__=json_filename)
    run_sh_path = os.path.join(sub, "run.sh")
    with open(run_sh_path, "w", encoding="utf-8") as f:
        f.write(run_sh)
    os.chmod(run_sh_path, 0o755)
    ini_py = _stamp(_load(os.path.join(_TEMPLATES, "local_ini.py")),
                    __JSON_FILENAME__=json_filename)
    ini_py_path = os.path.join(sub, "ini.py")
    with open(ini_py_path, "w", encoding="utf-8") as f:
        f.write(ini_py)
    os.chmod(ini_py_path, 0o755)
    _write_tests_suite(sub, mitm_cap_mw)
    readme = _readme_local(M, N).replace("{name}", name)
    with open(os.path.join(sub, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)


def _write_tests_suite(sub, mitm_cap_mw):
    """Copy the structured tests/ suite (lib + 4 drivers + READMEs) from
    templates/tests/ into a folder and create per-test csv/ + plots/.  The
    drivers are config-driven (tests/lib/common.py auto-discovers the plant
    JSON), so the same files work for any M×N plant; only the mitm_attack
    driver's default --cap-mw is stamped for this plant's nameplate."""
    src = os.path.join(_TEMPLATES, "tests")
    dst = os.path.join(sub, "tests")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".DS_Store"))
    mitm = os.path.join(dst, "mitm_attack", "run.py")
    with open(mitm, encoding="utf-8") as f:
        txt = f.read()
    with open(mitm, "w", encoding="utf-8") as f:
        f.write(_stamp(txt, __MITM_CAP_MW__=repr(float(mitm_cap_mw))))
    for tst in ("open_loop_step", "closed_loop_step",
                "frequency_response", "mitm_attack"):
        os.makedirs(os.path.join(dst, tst, "csv"), exist_ok=True)
        os.makedirs(os.path.join(dst, tst, "plots"), exist_ok=True)
    for run_py in glob.glob(os.path.join(dst, "*", "run.py")):
        os.chmod(run_py, 0o755)


def write_suite(out_dir, plant, name, M, N):
    """Emit the benchmark folder tree around ``plant``: four networked protocol
    stacks (modbus/iec104/dnp3/dds) + a no-Mininet ``local`` case, each with the
    structured ``tests/`` suite, plus the cross-protocol ``common/`` orchestration
    (``run_all_tests.py`` + monitor / plot / metrics).  The plant JSON is
    identical across protocols; only the device-level ``protocol`` field and a
    few wire-scale fields vary."""
    os.makedirs(out_dir, exist_ok=True)
    json_filename = f"{name}.json"
    py_interp = sys.executable

    pv_hosts = [f"LV{i_m:02d}{i_n:02d}"
                for i_m in range(1, M + 1) for i_n in range(1, N + 1)]
    pv_hosts_literal = "[" + ", ".join(f'"{h}"' for h in pv_hosts) + "]"

    # Plant nameplate (MW) and the MITM operating cap.  0.8 x nameplate leaves
    # headroom so the +10% over-power attack (real = cap/(1+offset)) stays under
    # the nameplate -> no inverter saturation, so the breach is stealthy.
    p_max_total_mw = sum(w.get("S_n", 1e6) for w in plant.get("weccs", [])) / 1e6
    mitm_cap_mw    = round(0.8 * p_max_total_mw, 1)

    # POI wire coordinates of the active-power point (the MITM target), read from
    # the poi config.  Modbus register = base + point register; IEC 104 IOA =
    # io_register; DNP3 index = position (or explicit dnp3_index).  Ports/common
    # address come from the poi device entry.  Fall back to copv_3_5 defaults.
    poi_modbus_port, poi_iec104_port, poi_dnp3_port = 502, 2404, 20000
    # IEC 104 common address is fixed by the proxy (colinker.iec104_pv_proxy
    # hardcodes COMMON_ADDR = 47), not the JSON common_address field — so the
    # MITM must target 47 to match the wire.
    poi_common_address = 47
    for d in plant.get("devices", []):
        if d.get("config") == "poi":
            poi_modbus_port    = int(d.get("modbus_port", 502))
            poi_iec104_port    = int(d.get("iec104_port", 2404))
            poi_dnp3_port      = int(d.get("dnp3_port", 20000))
            break
    poi_p_register, poi_p_ioa, poi_p_index = 1370, 1001, 1
    poi_cfg = (plant.get("configs", {}) or {}).get("poi", {})
    for i, m in enumerate(poi_cfg.get("measurements", [])):
        if m.get("ing_name") == "ActivePower" or (m.get("emec_name") or "").startswith("p_line"):
            poi_p_register = _MODBUS_REG_BASE + int(m.get("register", 370))
            poi_p_ioa      = int(m.get("io_register", 1001))
            poi_p_index    = int(m.get("dnp3_index", i))
            break

    run_subs = dict(
        __JSON_FILENAME__=json_filename,
        __PV_HOSTS_LITERAL__=pv_hosts_literal,
        __POI_MODBUS_PORT__=poi_modbus_port,
        __POI_IEC104_PORT__=poi_iec104_port,
        __POI_DNP3_PORT__=poi_dnp3_port,
        __POI_P_REGISTER__=poi_p_register,
        __POI_P_IOA__=poi_p_ioa,
        __POI_COMMON_ADDRESS__=poi_common_address,
        __POI_P_INDEX__=poi_p_index,
    )

    for protocol in ("modbus", "iec104", "dnp3", "dds"):
        sub = os.path.join(out_dir, protocol)
        os.makedirs(sub, exist_ok=True)

        # Per-protocol plant with the device-level protocol field stamped.
        plant_p = json.loads(json.dumps(plant))
        for d in plant_p.get("devices", []):
            d["protocol"] = protocol
        # IEC 104's event-driven link can't sustain the 0.1 s loop the others
        # use (see iec104_saturation.md): same PI gains, slower sample time.
        if protocol == "iec104" and "ppc" in plant_p:
            plant_p["ppc"]["ctrl_dt_s"] = _IEC104_CTRL_DT_S
        # IEC 104 / DNP3 carry the measured value already in engineering units on
        # the wire (emec_scale=1e-6, W->MW); Modbus keeps raw W (1.0) and the PPC
        # divides.  Power points identified by ing_name / emec_name.
        if protocol in _ENGINEERING_UNIT_WIRE_PROTOCOLS:
            for cfg in (plant_p.get("configs", {}) or {}).values():
                for m in cfg.get("measurements", []):
                    nm = m.get("emec_name") or ""
                    if m.get("ing_name") in ("ActivePower", "ReactivePower") \
                            or nm.startswith(("p_", "q_")):
                        m["emec_scale"] = 1e-6
        with open(os.path.join(sub, json_filename), "w", encoding="utf-8") as f:
            json.dump(plant_p, f, indent=2)

        run_py = _stamp(_load(_RUN_PY_FILES[protocol]), **run_subs)
        run_py_path = os.path.join(sub, "run.py")
        with open(run_py_path, "w", encoding="utf-8") as f:
            f.write(run_py)
        os.chmod(run_py_path, 0o755)

        _write_tests_suite(sub, mitm_cap_mw)

        readme = _README_FN[protocol](M, N).replace("{name}", name)
        with open(os.path.join(sub, "README.md"), "w", encoding="utf-8") as f:
            f.write(readme)

    # local case (no Mininet) + cross-protocol common/ + top README.
    _write_local_case(out_dir, plant, name, M, N, json_filename, mitm_cap_mw)
    _write_common(out_dir, name, py_interp, mitm_cap_mw)
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(_readme_top(name, M, N))

    print(f"wrote suite at {out_dir}: M={M} N={N} ({M * N} inverters), "
          f"nameplate {p_max_total_mw:.1f} MW, MITM cap {mitm_cap_mw} MW")


__all__ = ["designer", "make_xy_0", "write_suite"]


if __name__ == "__main__":
    # Plant parameters
    M, N = 5, 10
    NAME = f"copv_{M}_{N}"
    OUT_DIR = f"/home/jmmmacubu/mac_shared/benchmarks_colinker/{NAME}_wecc"

    # 1. Build the plant dict (and write a flat <NAME>.json next to this file).
    plant = designer(
        M, N,
        output_file=os.path.join(_HERE, f"{NAME}.json"),
        P_inv_max=1e6, U_lv=800, U_mv=30e3, U_hv=132e3,
        Pref_pq=0.5, Qext_pq=0.1,
    )

    # 2. Scaffold the four-protocol + local suite (with common/run_all_tests.py).
    write_suite(OUT_DIR, plant, name=NAME, M=M, N=N)

    # 3. Sanity-check: build and init the model via pydae.  The scaffold does
    #    not ship xy_0.json (pydae_sim.construct() writes it at runtime), so
    #    seed the init from a temp file here.
    from pydae.bps import BpsBuilder
    from pydae.core.builder.casadi_builder import CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel
    from pydae.bps.utils.reporter import report_all

    plant_json = os.path.join(OUT_DIR, "modbus", f"{NAME}.json")
    xy_0_json  = os.path.join(tempfile.gettempdir(), f"{NAME}_xy_0.json")
    with open(xy_0_json, "w", encoding="utf-8") as f:
        json.dump(make_xy_0(plant), f, indent=4)
    print(f"\n[sanity] BpsBuilder({plant_json})")
    grid = BpsBuilder(plant_json, use_casadi=True)
    grid.construct(plant_json)
    bld = CasadiBuilder(grid.sys_dict).build()

    model = CasadiModel(bld)
    model.ini({}, xy_0_json)
    report_all(model, plant_json)
