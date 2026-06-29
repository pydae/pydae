"""Canonical PV-plant designer for pydae + colinker.

Produces:

- a single pydae-compatible plant JSON (electrical + comms + cosim configs)
- a pydae Newton-solver seed (``xy_0.json``)
- optionally, a four-protocol benchmark folder tree (``modbus`` /
  ``iec104`` / ``dnp3`` / ``dds``) ready to be driven by colinker.  The
  ``modbus`` and ``iec104`` folders carry a structured ``tests/`` suite
  (open-/closed-loop step, frequency-response Bode, MITM) copied from
  ``templates/tests/``.

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

import numpy as np


_HERE = os.path.dirname(os.path.abspath(__file__))


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
            reg_0 += 1000
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
# Run.py / run_step.sh target the standard Linux VM layout (pydae_dev conda
# env, COLINKER_PYTHON, sudo-rs). Edit the constants inside the emitted
# run.py if your paths differ.

_PROTOCOL_LABEL = {
    "modbus": "Modbus TCP stack",
    "iec104": "IEC 60870-5-104 stack",
    "dnp3":   "DNP3 stack",
    "dds":    "DDS stack",
}

# IEC 104 runs the PPC control loop slower than the other stacks (which use the
# designer default 0.1 s): its event-driven c104 link cannot sustain a 10 Hz
# loop without saturating / oscillating.  See iec104_saturation.md.
_IEC104_CTRL_DT_S = 1.0

# Protocols that carry the structured tests/ suite (open-/closed-loop step,
# frequency-response Bode, MITM) instead of the flat run_step.sh + results/.
# All run the shared ppc_core control API, so the same drivers apply.  dnp3's
# mitm_attack drives colinker.attacks.dnp3_mitm (a DNP3-aware MITM that
# re-frames the falsified AI response with fresh DLL CRCs).
_SUITE_PROTOCOLS = ("modbus", "iec104", "dnp3")

# Protocols whose proxies put engineering units on the wire (wire = pydae *
# emec_scale) and whose PPC reads them directly — power measurement points
# therefore need emec_scale=1e-6 (W→MW).  Modbus keeps raw W (scale 1.0) and
# divides in software, so it is excluded.
_ENGINEERING_UNIT_WIRE_PROTOCOLS = ("iec104", "dnp3")

_RUN_PY_TEMPLATE = '''#!/usr/bin/env python3
"""Launch the co-simulation (Mininet + pydae + proxies + PPC).

Reads the plant JSON sitting next to this file and delegates to
``colinker.mn.mininet_topo.setup()``. Requires root.

Usage::

    sudo python3 run.py [--no-cli] [--no-ppc]
    sudo python3 run.py --no-cli --arrivals-dir results_delays/arrivals
    sudo python3 run.py --no-cli --mitm                 # default: +5% offset
    sudo python3 run.py --no-cli --mitm --mitm-offset-frac 0.05 \\
                        --arrivals-dir results_overpower/arrivals \\
                        --pcap-dir     results_overpower/pcaps
    sudo python3 run.py --no-cli --mitm --mitm-value-mw __MITM_DEFAULT_VALUE_MW__   # fixed-spoof
"""
import argparse
import os
import sys


_HERE         = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_JSON = os.path.join(_HERE, "__JSON_FILENAME__")
_PYTHON       = "/home/jmmmacubu/miniforge3/envs/pydae_dev/bin/python3"
_COLINKER_SRC = "/home/jmmmacubu/mac_shared/colinker/src"

# Pcap default capture set: PPC, POI, MITM, plus one host per inverter.
_PV_HOSTS = __PV_HOSTS_LITERAL__


def _check_root() -> None:
    if os.geteuid() != 0:
        print("error: Mininet requires root. Re-run with:")
        print(f"  sudo python3 {sys.argv[0]}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the co-simulation (__PROTOCOL_LABEL__)"
    )
    parser.add_argument("--json", "-j", default=_DEFAULT_JSON,
                        help=f"Plant JSON path (default: {_DEFAULT_JSON})")
    parser.add_argument("--no-cli", action="store_true",
                        help="Skip Mininet CLI; block on SIGINT (headless)")
    parser.add_argument("--no-ppc", action="store_true",
                        help="Skip auto-starting the PPC")
    parser.add_argument("--arrivals-dir", default=None,
                        help="If set, each pv_proxy writes a 50 ms-cadence CSV "
                             "of the wire-level setpoint values that arrived "
                             "at it (one file per inverter, named "
                             "<emec_id>_arrivals.csv).")
    parser.add_argument("--mitm", action="store_true",
                        help="Arm the Modbus MITM attack from the MITM host. "
                             "Default: spoof POI active-power reading.")
    parser.add_argument("--mitm-args", default=None,
                        help="Override the full argv passed to "
                             "colinker.attacks.modbus_mitm (when --mitm is set). "
                             "Implies --mitm.")
    parser.add_argument("--mitm-value-mw", type=float, default=None,
                        help="FIXED-spoof mode: pin the POI active-power "
                             "readback at this constant (MW) while armed. The "
                             "PI error never closes, so the plant drives to "
                             "saturation. Mutually exclusive with "
                             "--mitm-offset-frac / --mitm-offset-mw.")
    parser.add_argument("--mitm-offset-frac", type=float, default=0.05,
                        help="OFFSET-spoof mode (default): add this fraction of "
                             "the REAL POI reading before the PPC sees it "
                             "(0.05 = +5%%). The PPC settles at a small, "
                             "non-saturating steady-state error — the plant "
                             "ends up ~this fraction off target.")
    parser.add_argument("--mitm-offset-mw", type=float, default=0.0,
                        help="OFFSET-spoof mode: also add this absolute bias "
                             "(MW) to the real POI reading. Combined with "
                             "--mitm-offset-frac.")
    parser.add_argument("--mitm-start-after", type=float, default=30.0,
                        help="Seconds after launch before the attack arms.")
    parser.add_argument("--mitm-duration", type=float, default=60.0,
                        help="Seconds the attack holds, then disarms.")
    parser.add_argument("--pcap-dir", default=None,
                        help="If set, run tcpdump in every host's netns "
                             "(PPC, POI, all PV hosts, MITM) capturing all "
                             "Modbus (tcp/5000 + tcp/502) and ARP traffic to "
                             "<PCAP_DIR>/<HOST>.pcap.")
    args = parser.parse_args()

    _check_root()

    if not os.path.exists(args.json):
        print(f"error: JSON not found: {args.json}")
        sys.exit(1)

    # sudo-rs blocks env-var forwarding even with -E, so set inline.
    os.environ.setdefault("COLINKER_PYTHON", _PYTHON)

    if args.arrivals_dir:
        os.makedirs(args.arrivals_dir, exist_ok=True)
        os.environ["COLINKER_ARRIVALS_DIR"] = os.path.abspath(args.arrivals_dir)
        print(f"[run] arrivals CSVs -> {os.environ['COLINKER_ARRIVALS_DIR']}/<name>_arrivals.csv")

    if args.pcap_dir:
        os.makedirs(args.pcap_dir, exist_ok=True)
        os.environ["COLINKER_PCAP_DIR"] = os.path.abspath(args.pcap_dir)
        # PPC/POI/MITM are the three vantage points needed to see the attack
        # signature (same Modbus tid, different payload bytes).  Per-inverter
        # captures are optional — drop them to save memory on small VMs.
        pcap_hosts = ["PPC", "POI", "MITM"]
        if os.environ.get("COLINKER_PCAP_INCLUDE_PV", "0") not in ("0", "", "false"):
            pcap_hosts += list(_PV_HOSTS)
        os.environ["COLINKER_PCAP_HOSTS"] = ",".join(pcap_hosts)
        print(f"[run] pcap captures -> {os.environ['COLINKER_PCAP_DIR']}/<HOST>.pcap")

    if args.mitm or args.mitm_args:
        # Default attack: falsify the POI active-power register
        # (p_line_POI_GRID, float32 CDAB, raw W) the PPC reads.
        #
        #   offset mode (default): add a small bias to the REAL reading
        #     (cur*(1+frac)+abs) → the PPC settles ~frac off target without
        #     saturating — stealthy, realistic falsification.
        #   fixed mode (--mitm-value-mw): pin the readback at a constant →
        #     the PI rails to saturation (the original over-power demo).
        #
        # --explicit-redirect mode: the PPC is told via --device-ip POI=<MITM>
        # to connect directly to MITM:5000; MITM forwards to the real POI.
        # Deterministic test path — the ARP-spoof + iptables NAT path is
        # documented but not used here (conntrack flakiness inside netns).
        MITM_IP          = "10.10.0.9"
        # Listen on the same port the POI exposes so the PPC's
        # `--device-ip POI=<MITM>` override (IP only, not port) reaches us.
        POI_PORT         = __POI_MODBUS_PORT__
        MITM_LISTEN_PORT = POI_PORT
        # POI register holding p_line_POI_GRID (offset 370 from reg_0=1000).
        POI_P_REGISTER   = 1370
        if args.mitm_value_mw is not None:
            if args.mitm_offset_mw or args.mitm_offset_frac != 0.05:
                print("[run] WARNING: --mitm-value-mw (fixed) overrides "
                      "--mitm-offset-* (offset)")
            inject_arg = f"--value {args.mitm_value_mw * 1e6}"
        else:
            # offset mode: bias real reading by frac (+ abs MW → W)
            inject_arg = (f"--offset-frac {args.mitm_offset_frac} "
                          f"--offset {args.mitm_offset_mw * 1e6}")
        default_args = (
            "--iface mitm-eth0 "
            "--victim-ip 10.10.0.4   --victim-mac 00:00:00:00:00:04 "
            "--target-ip 10.10.0.5   --target-mac 00:00:00:00:00:05 "
            f"--target-port {POI_PORT}  --listen-port {MITM_LISTEN_PORT} "
            f"--register {POI_P_REGISTER}  --n-regs 2  --dtype float32  --fmt CDAB "
            f"{inject_arg} "
            f"--start-after {args.mitm_start_after}  --duration {args.mitm_duration} "
            "--explicit-redirect"
        )
        os.environ["COLINKER_MITM_ARGS"] = args.mitm_args or default_args
        if "--explicit-redirect" in os.environ["COLINKER_MITM_ARGS"]:
            os.environ["COLINKER_PPC_POI_IP_OVERRIDE"] = MITM_IP
            print(f"[run] PPC POI device-ip overridden to MITM ({MITM_IP})")
        print(f"[run] MITM attack ARMED — args:\\n    {os.environ['COLINKER_MITM_ARGS']}")

    if _COLINKER_SRC not in sys.path:
        sys.path.insert(0, _COLINKER_SRC)

    from mininet.log import setLogLevel
    from colinker.mn.mininet_topo import setup

    setLogLevel("info")
    setup(args.json,
          interactive=not args.no_cli,
          ppc_auto=not args.no_ppc)


if __name__ == "__main__":
    main()
'''


# IEC 60870-5-104 variant of run.py.  Same skeleton as the Modbus template but
# the MITM + pcap wiring differs: the attack rewrites an ``M_ME_NC_1`` short
# float keyed by IOA + common address (not a Modbus register), the value/offset
# are already in MW on the wire (``emec_scale=1e-6``, no W↔MW factor), the proxy
# server speaks tcp/2404, and the MITM module is ``colinker.attacks.iec104_mitm``.
_RUN_PY_TEMPLATE_IEC104 = '''#!/usr/bin/env python3
"""Launch the co-simulation (Mininet + pydae + proxies + PPC).

Reads the plant JSON sitting next to this file and delegates to
``colinker.mn.mininet_topo.setup()``. Requires root.

Usage::

    sudo python3 run.py [--no-cli] [--no-ppc]
    sudo python3 run.py --no-cli --arrivals-dir results_delays/arrivals
    sudo python3 run.py --no-cli --mitm                 # default: +5% offset
    sudo python3 run.py --no-cli --mitm --mitm-offset-frac 0.05 \\
                        --arrivals-dir results_overpower/arrivals \\
                        --pcap-dir     results_overpower/pcaps
    sudo python3 run.py --no-cli --mitm --mitm-value-mw __MITM_DEFAULT_VALUE_MW__   # fixed-spoof
"""
import argparse
import os
import sys


_HERE         = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_JSON = os.path.join(_HERE, "__JSON_FILENAME__")
_PYTHON       = "/home/jmmmacubu/miniforge3/envs/pydae_dev/bin/python3"
_COLINKER_SRC = "/home/jmmmacubu/mac_shared/colinker/src"

# Pcap default capture set: PPC, POI, MITM, plus one host per inverter.
_PV_HOSTS = __PV_HOSTS_LITERAL__


def _check_root() -> None:
    if os.geteuid() != 0:
        print("error: Mininet requires root. Re-run with:")
        print(f"  sudo python3 {sys.argv[0]}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the co-simulation (__PROTOCOL_LABEL__)"
    )
    parser.add_argument("--json", "-j", default=_DEFAULT_JSON,
                        help=f"Plant JSON path (default: {_DEFAULT_JSON})")
    parser.add_argument("--no-cli", action="store_true",
                        help="Skip Mininet CLI; block on SIGINT (headless)")
    parser.add_argument("--no-ppc", action="store_true",
                        help="Skip auto-starting the PPC")
    parser.add_argument("--arrivals-dir", default=None,
                        help="If set, each pv_proxy writes a 50 ms-cadence CSV "
                             "of the wire-level setpoint values that arrived "
                             "at it (one file per inverter, named "
                             "<emec_id>_arrivals.csv).")
    parser.add_argument("--mitm", action="store_true",
                        help="Arm the IEC 104 MITM attack from the MITM host. "
                             "Default: offset the POI active-power value.")
    parser.add_argument("--mitm-args", default=None,
                        help="Override the full argv passed to "
                             "colinker.attacks.iec104_mitm (when --mitm is set). "
                             "Implies --mitm.")
    parser.add_argument("--mitm-value-mw", type=float, default=None,
                        help="FIXED-spoof mode: pin the POI active-power "
                             "readback at this constant (MW) while armed. The "
                             "PI error never closes, so the plant drives to "
                             "saturation. Mutually exclusive with "
                             "--mitm-offset-frac / --mitm-offset-mw.")
    parser.add_argument("--mitm-offset-frac", type=float, default=0.05,
                        help="OFFSET-spoof mode (default): add this fraction of "
                             "the REAL POI reading before the PPC sees it "
                             "(0.05 = +5%%). The PPC settles at a small, "
                             "non-saturating steady-state error — the plant "
                             "ends up ~this fraction off target.")
    parser.add_argument("--mitm-offset-mw", type=float, default=0.0,
                        help="OFFSET-spoof mode: also add this absolute bias "
                             "(MW) to the real POI reading. Combined with "
                             "--mitm-offset-frac.")
    parser.add_argument("--mitm-start-after", type=float, default=30.0,
                        help="Seconds after launch before the attack arms.")
    parser.add_argument("--mitm-duration", type=float, default=60.0,
                        help="Seconds the attack holds, then disarms.")
    parser.add_argument("--pcap-dir", default=None,
                        help="If set, run tcpdump in every host's netns "
                             "(PPC, POI, all PV hosts, MITM) capturing all "
                             "IEC 104 (tcp/2404) and ARP traffic to "
                             "<PCAP_DIR>/<HOST>.pcap.")
    args = parser.parse_args()

    _check_root()

    if not os.path.exists(args.json):
        print(f"error: JSON not found: {args.json}")
        sys.exit(1)

    # sudo-rs blocks env-var forwarding even with -E, so set inline.
    os.environ.setdefault("COLINKER_PYTHON", _PYTHON)

    if args.arrivals_dir:
        os.makedirs(args.arrivals_dir, exist_ok=True)
        os.environ["COLINKER_ARRIVALS_DIR"] = os.path.abspath(args.arrivals_dir)
        print(f"[run] arrivals CSVs -> {os.environ['COLINKER_ARRIVALS_DIR']}/<name>_arrivals.csv")

    if args.pcap_dir:
        os.makedirs(args.pcap_dir, exist_ok=True)
        os.environ["COLINKER_PCAP_DIR"] = os.path.abspath(args.pcap_dir)
        # PPC/POI/MITM are the three vantage points needed to see the attack
        # signature (same IOA, different M_ME_NC_1 value bytes).  Per-inverter
        # captures are optional — drop them to save memory on small VMs.
        pcap_hosts = ["PPC", "POI", "MITM"]
        if os.environ.get("COLINKER_PCAP_INCLUDE_PV", "0") not in ("0", "", "false"):
            pcap_hosts += list(_PV_HOSTS)
        os.environ["COLINKER_PCAP_HOSTS"] = ",".join(pcap_hosts)
        # IEC 104 runs on tcp/2404, not the Modbus default (5000/502).
        os.environ.setdefault("COLINKER_PCAP_PORTS", str(__POI_IEC104_PORT__))
        print(f"[run] pcap captures (tcp/{os.environ['COLINKER_PCAP_PORTS']}) -> "
              f"{os.environ['COLINKER_PCAP_DIR']}/<HOST>.pcap")

    if args.mitm or args.mitm_args:
        # Default attack: falsify the POI active-power measured value
        # (p_line_POI_GRID, M_ME_NC_1 short float at IOA __POI_P_IOA__) the PPC
        # reads over IEC 60870-5-104.
        #
        #   offset mode (default): add a small bias to the REAL value
        #     (cur*(1+frac)+abs) → the PPC settles ~frac off target without
        #     saturating — stealthy, realistic falsification.
        #   fixed mode (--mitm-value-mw): pin the readback at a constant →
        #     the PI rails to saturation (the original over-power demo).
        #
        # --explicit-redirect mode: the PPC is told via the POI device-ip
        # override to connect directly to MITM:2404; MITM forwards to the real
        # POI.  Deterministic test path — the ARP-spoof + NAT path is documented
        # but not used here (conntrack flakiness inside netns).
        MITM_IP          = "10.10.0.9"
        # Listen on the same port the POI server exposes so the PPC's POI
        # device-ip override (IP only, not port) reaches us.
        POI_PORT         = __POI_IEC104_PORT__
        MITM_LISTEN_PORT = POI_PORT
        # POI information-object address holding p_line_POI_GRID, and the ASDU
        # common (station) address.  CA is fixed by the colinker iec104 stack
        # (colinker.iec104_ppc.COMMON_ADDR / iec104_poi_proxy.COMMON_ADDR), not
        # the JSON device field — keep them in sync if that constant changes.
        POI_P_IOA        = __POI_P_IOA__
        COMMON_ADDRESS   = 47
        # The IEC 104 wire carries the value already scaled to MW
        # (proxy: pydae_W * emec_scale, emec_scale=1e-6), so the absolute
        # value/offset are in MW here (no 1e6 factor — unlike the Modbus
        # path where the register holds raw W). offset-frac is unitless.
        if args.mitm_value_mw is not None:
            if args.mitm_offset_mw or args.mitm_offset_frac != 0.05:
                print("[run] WARNING: --mitm-value-mw (fixed) overrides "
                      "--mitm-offset-* (offset)")
            inject_arg = f"--value {args.mitm_value_mw}"
        else:
            inject_arg = (f"--offset-frac {args.mitm_offset_frac} "
                          f"--offset {args.mitm_offset_mw}")
        default_args = (
            "--iface mitm-eth0 "
            "--victim-ip 10.10.0.4   --victim-mac 00:00:00:00:00:04 "
            "--target-ip 10.10.0.5   --target-mac 00:00:00:00:00:05 "
            f"--target-port {POI_PORT}  --listen-port {MITM_LISTEN_PORT} "
            f"--ioa {POI_P_IOA}  --common-address {COMMON_ADDRESS} "
            f"{inject_arg} "
            f"--start-after {args.mitm_start_after}  --duration {args.mitm_duration} "
            "--explicit-redirect"
        )
        os.environ["COLINKER_MITM_MODULE"] = "colinker.attacks.iec104_mitm"
        os.environ["COLINKER_MITM_ARGS"] = args.mitm_args or default_args
        if "--explicit-redirect" in os.environ["COLINKER_MITM_ARGS"]:
            os.environ["COLINKER_PPC_POI_IP_OVERRIDE"] = MITM_IP
            print(f"[run] PPC POI device-ip overridden to MITM ({MITM_IP})")
        print(f"[run] IEC 104 MITM attack ARMED — args:\\n    {os.environ['COLINKER_MITM_ARGS']}")

    if _COLINKER_SRC not in sys.path:
        sys.path.insert(0, _COLINKER_SRC)

    from mininet.log import setLogLevel
    from colinker.mn.mininet_topo import setup

    setLogLevel("info")
    setup(args.json,
          interactive=not args.no_cli,
          ppc_auto=not args.no_ppc)


if __name__ == "__main__":
    main()
'''

# DNP3 variant of run.py.  Same skeleton as the IEC 104 template but the MITM
# rewrites a DNP3 Analog Input float (Group 30 Var 5) keyed by object index
# (not an IOA/common-address or a Modbus register); the proxy server speaks
# tcp/20000; and the MITM module is ``colinker.attacks.dnp3_mitm``, which
# re-frames the falsified response with fresh DLL CRCs (a plain byte patch
# would corrupt the block CRC and black-hole the POI reading).  Value/offset
# are in MW on the wire (``emec_scale=1e-6``), as for IEC 104.
_RUN_PY_TEMPLATE_DNP3 = '''#!/usr/bin/env python3
"""Launch the co-simulation (Mininet + pydae + proxies + PPC).

Reads the plant JSON sitting next to this file and delegates to
``colinker.mn.mininet_topo.setup()``. Requires root.

Usage::

    sudo python3 run.py [--no-cli] [--no-ppc]
    sudo python3 run.py --no-cli --arrivals-dir results_delays/arrivals
    sudo python3 run.py --no-cli --mitm                 # default: +5% offset
    sudo python3 run.py --no-cli --mitm --mitm-offset-frac 0.05 \\
                        --arrivals-dir results_overpower/arrivals \\
                        --pcap-dir     results_overpower/pcaps
    sudo python3 run.py --no-cli --mitm --mitm-value-mw __MITM_DEFAULT_VALUE_MW__   # fixed-spoof
"""
import argparse
import os
import sys


_HERE         = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_JSON = os.path.join(_HERE, "__JSON_FILENAME__")
_PYTHON       = "/home/jmmmacubu/miniforge3/envs/pydae_dev/bin/python3"
_COLINKER_SRC = "/home/jmmmacubu/mac_shared/colinker/src"

# Pcap default capture set: PPC, POI, MITM, plus one host per inverter.
_PV_HOSTS = __PV_HOSTS_LITERAL__


def _check_root() -> None:
    if os.geteuid() != 0:
        print("error: Mininet requires root. Re-run with:")
        print(f"  sudo python3 {sys.argv[0]}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the co-simulation (__PROTOCOL_LABEL__)"
    )
    parser.add_argument("--json", "-j", default=_DEFAULT_JSON,
                        help=f"Plant JSON path (default: {_DEFAULT_JSON})")
    parser.add_argument("--no-cli", action="store_true",
                        help="Skip Mininet CLI; block on SIGINT (headless)")
    parser.add_argument("--no-ppc", action="store_true",
                        help="Skip auto-starting the PPC")
    parser.add_argument("--arrivals-dir", default=None,
                        help="If set, each pv_proxy writes a 50 ms-cadence CSV "
                             "of the wire-level setpoint values that arrived "
                             "at it (one file per inverter, named "
                             "<emec_id>_arrivals.csv).")
    parser.add_argument("--mitm", action="store_true",
                        help="Arm the DNP3 MITM attack from the MITM host. "
                             "Default: offset the POI active-power value.")
    parser.add_argument("--mitm-args", default=None,
                        help="Override the full argv passed to "
                             "colinker.attacks.dnp3_mitm (when --mitm is set). "
                             "Implies --mitm.")
    parser.add_argument("--mitm-value-mw", type=float, default=None,
                        help="FIXED-spoof mode: pin the POI active-power "
                             "readback at this constant (MW) while armed. The "
                             "PI error never closes, so the plant drives to "
                             "saturation. Mutually exclusive with "
                             "--mitm-offset-frac / --mitm-offset-mw.")
    parser.add_argument("--mitm-offset-frac", type=float, default=0.05,
                        help="OFFSET-spoof mode (default): add this fraction of "
                             "the REAL POI reading before the PPC sees it "
                             "(0.05 = +5%%). The PPC settles at a small, "
                             "non-saturating steady-state error — the plant "
                             "ends up ~this fraction off target.")
    parser.add_argument("--mitm-offset-mw", type=float, default=0.0,
                        help="OFFSET-spoof mode: also add this absolute bias "
                             "(MW) to the real POI reading. Combined with "
                             "--mitm-offset-frac.")
    parser.add_argument("--mitm-start-after", type=float, default=30.0,
                        help="Seconds after launch before the attack arms.")
    parser.add_argument("--mitm-duration", type=float, default=60.0,
                        help="Seconds the attack holds, then disarms.")
    parser.add_argument("--pcap-dir", default=None,
                        help="If set, run tcpdump in every host's netns "
                             "(PPC, POI, all PV hosts, MITM) capturing all "
                             "DNP3 (tcp/__POI_DNP3_PORT__) and ARP traffic to "
                             "<PCAP_DIR>/<HOST>.pcap.")
    args = parser.parse_args()

    _check_root()

    if not os.path.exists(args.json):
        print(f"error: JSON not found: {args.json}")
        sys.exit(1)

    # sudo-rs blocks env-var forwarding even with -E, so set inline.
    os.environ.setdefault("COLINKER_PYTHON", _PYTHON)

    if args.arrivals_dir:
        os.makedirs(args.arrivals_dir, exist_ok=True)
        os.environ["COLINKER_ARRIVALS_DIR"] = os.path.abspath(args.arrivals_dir)
        print(f"[run] arrivals CSVs -> {os.environ['COLINKER_ARRIVALS_DIR']}/<name>_arrivals.csv")

    if args.pcap_dir:
        os.makedirs(args.pcap_dir, exist_ok=True)
        os.environ["COLINKER_PCAP_DIR"] = os.path.abspath(args.pcap_dir)
        # PPC/POI/MITM are the three vantage points needed to see the attack
        # signature (same AI object index, different float value bytes).
        # Per-inverter captures are optional — drop them to save memory.
        pcap_hosts = ["PPC", "POI", "MITM"]
        if os.environ.get("COLINKER_PCAP_INCLUDE_PV", "0") not in ("0", "", "false"):
            pcap_hosts += list(_PV_HOSTS)
        os.environ["COLINKER_PCAP_HOSTS"] = ",".join(pcap_hosts)
        # DNP3 runs on tcp/__POI_DNP3_PORT__, not the Modbus default (5000/502).
        os.environ.setdefault("COLINKER_PCAP_PORTS", str(__POI_DNP3_PORT__))
        print(f"[run] pcap captures (tcp/{os.environ['COLINKER_PCAP_PORTS']}) -> "
              f"{os.environ['COLINKER_PCAP_DIR']}/<HOST>.pcap")

    if args.mitm or args.mitm_args:
        # Default attack: falsify the POI active-power measured value
        # (p_line_POI_GRID, DNP3 Analog Input G30V5 at object index
        # __POI_P_INDEX__) the PPC reads over DNP3.  The MITM re-frames the
        # response with fresh DLL CRCs after rewriting the float, so the link
        # stays in sync (a raw byte patch would corrupt the block CRC and
        # black-hole the POI reading).
        #
        #   offset mode (default): add a small bias to the REAL value
        #     (cur*(1+frac)+abs) → the PPC settles ~frac off target without
        #     saturating — stealthy, realistic falsification.
        #   fixed mode (--mitm-value-mw): pin the readback at a constant →
        #     the PI rails to saturation (the original over-power demo).
        #
        # --explicit-redirect mode: the PPC is told via the POI device-ip
        # override to connect directly to MITM:__POI_DNP3_PORT__; MITM forwards
        # to the real POI.  Deterministic test path — the ARP-spoof + NAT path
        # is documented but not used here (conntrack flakiness inside netns).
        MITM_IP          = "10.10.0.9"
        # Listen on the same port the POI server exposes so the PPC's POI
        # device-ip override (IP only, not port) reaches us.
        POI_PORT         = __POI_DNP3_PORT__
        MITM_LISTEN_PORT = POI_PORT
        # POI Analog Input object index holding p_line_POI_GRID.
        POI_P_INDEX      = __POI_P_INDEX__
        # The DNP3 wire carries the value already scaled to MW (proxy:
        # pydae_W * emec_scale, emec_scale=1e-6), so the absolute value/offset
        # are in MW here (no 1e6 factor — unlike the Modbus register path's
        # raw W). offset-frac is unitless.
        if args.mitm_value_mw is not None:
            if args.mitm_offset_mw or args.mitm_offset_frac != 0.05:
                print("[run] WARNING: --mitm-value-mw (fixed) overrides "
                      "--mitm-offset-* (offset)")
            inject_arg = f"--value {args.mitm_value_mw}"
        else:
            inject_arg = (f"--offset-frac {args.mitm_offset_frac} "
                          f"--offset {args.mitm_offset_mw}")
        default_args = (
            "--iface mitm-eth0 "
            "--victim-ip 10.10.0.4   --victim-mac 00:00:00:00:00:04 "
            "--target-ip 10.10.0.5   --target-mac 00:00:00:00:00:05 "
            f"--target-port {POI_PORT}  --listen-port {MITM_LISTEN_PORT} "
            f"--ai-index {POI_P_INDEX} "
            f"{inject_arg} "
            f"--start-after {args.mitm_start_after}  --duration {args.mitm_duration} "
            "--explicit-redirect"
        )
        os.environ["COLINKER_MITM_MODULE"] = "colinker.attacks.dnp3_mitm"
        os.environ["COLINKER_MITM_ARGS"] = args.mitm_args or default_args
        if "--explicit-redirect" in os.environ["COLINKER_MITM_ARGS"]:
            os.environ["COLINKER_PPC_POI_IP_OVERRIDE"] = MITM_IP
            print(f"[run] PPC POI device-ip overridden to MITM ({MITM_IP})")
        print(f"[run] DNP3 MITM attack ARMED — args:\\n    {os.environ['COLINKER_MITM_ARGS']}")

    if _COLINKER_SRC not in sys.path:
        sys.path.insert(0, _COLINKER_SRC)

    from mininet.log import setLogLevel
    from colinker.mn.mininet_topo import setup

    setLogLevel("info")
    setup(args.json,
          interactive=not args.no_cli,
          ppc_auto=not args.no_ppc)


if __name__ == "__main__":
    main()
'''

# Per-protocol run.py template override.  Protocols not listed fall back to the
# Modbus-style _RUN_PY_TEMPLATE (placeholders that don't apply are simply left
# unreplaced/unused — dds doesn't yet have its own MITM wiring).
_RUN_PY_TEMPLATE_BY_PROTOCOL = {
    "iec104": _RUN_PY_TEMPLATE_IEC104,
    "dnp3":   _RUN_PY_TEMPLATE_DNP3,
}

_RUN_STEP_SH = '''#!/usr/bin/env bash
# PQ step-response benchmark.  Assumes the cosim is already running
# (sudo python3 run.py in a separate terminal) and the PPC HTTP API is
# reachable via the socat relay at http://localhost:8090.
#
# Usage:   ./run_step.sh [out.csv]
set -euo pipefail

OUT="${1:-step.csv}"
PYTHON="/home/jmmmacubu/miniforge3/envs/pydae_dev/bin/python3"

echo "[bench] recorder start -> $OUT  (auto-arm on |dtarget_mw| >= 0.1, 3 s pre, 10 s post)"
"$PYTHON" -m colinker.record_ctl start --out "$OUT" \\
    --trigger-mw 0.1 --pre 3 --post 10

echo "[bench] PQ step:  P 2.0 -> 3.0 MW @ t=1 s   Q 0 -> 1.0 Mvar @ t=5 s   (end @ t=15 s)"
"$PYTHON" -m colinker.poi_step_test \\
    --p-initial 2.0 --p-step 3.0 --q-step 1.0 --t-end 15

echo "[bench] recorder stop"
"$PYTHON" -m colinker.record_ctl stop

SVG="${OUT%.csv}.svg"
echo "[bench] plot -> $SVG"
"$PYTHON" -m colinker.report --csv "$OUT" --out "$SVG"

echo "[bench] done."
echo "  CSV:  $OUT"
echo "  SVG:  $SVG"
'''


_PLOT_FLOW_TEMPLATE = '''#!/usr/bin/env python3
"""Plot the three-stage signal flow for this Modbus benchmark.

Layout: (1 + N_inv) rows × 2 cols.

Row 0 — POI active / reactive overlay
        (target → PPC reading → pydae ground truth).
Rows 1..N — one row per inverter, each with active (left) / reactive (right):
  * PPC sent — from ``step.csv`` ``pv{i}_sp_{mw,mvar}``.
  * PV received — from ``arrivals/<LVid>_arrivals.csv`` (``Pref_LVid_pu``).
  * Inverter output — pydae ground-truth ``sim_p_g_LV<id>`` from step.csv.

Usage::

    python plot_flow.py                              # results_delays/flow.svg
    python plot_flow.py --results results_overpower/
    python plot_flow.py --out /tmp/x.svg
"""
import argparse
import csv
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


P_NOMINAL_MW   = 1.0   # 1.0 pu = 1 MW per inverter (S_n = 1 MVA)
Q_NOMINAL_MVAR = 1.0   # 1.0 pu = 1 Mvar (= S_n)

INV_IDS = __PV_IDS_LITERAL__


def _parse_iso(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def _load_step(path):
    cols = {}
    t_wall = []
    with open(path) as f:
        for row in csv.DictReader(f):
            t_wall.append(_parse_iso(row["ts_iso"]))
            for k, v in row.items():
                if k == "ts_iso":
                    continue
                try:
                    cols.setdefault(k, []).append(float(v))
                except ValueError:
                    cols.setdefault(k, []).append(float("nan"))
    return t_wall, cols


def _load_arrivals(path, p_col, q_col):
    t, p, q = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            t.append(float(row["t_wall"]))
            p.append(float(row[p_col]))
            q.append(float(row[q_col]))
    return t, p, q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--results", default=os.path.join(here, "results_delays"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    step_path = os.path.join(args.results, "step.csv")
    out_path  = args.out or os.path.join(args.results, "flow.svg")

    step_t, step = _load_step(step_path)
    t0 = step_t[0]
    step_rel = [t - t0 for t in step_t]
    t_max = step_rel[-1]

    arrivals = {}
    for inv_id in INV_IDS:
        path = os.path.join(args.results, "arrivals", f"{inv_id}_arrivals.csv")
        if not os.path.exists(path):
            arrivals[inv_id] = None
            continue
        t, p_pu, q_pu = _load_arrivals(path,
                                        f"Pref_{inv_id}_pu",
                                        f"Qext_{inv_id}_pu")
        t_rel = [x - t0 for x in t]
        keep = [i for i, x in enumerate(t_rel) if 0.0 <= x <= t_max + 0.5]
        arrivals[inv_id] = (
            [t_rel[i] for i in keep],
            [p_pu[i]   for i in keep],
            [q_pu[i]   for i in keep],
        )

    n_rows = 1 + len(INV_IDS)
    fig, axes = plt.subplots(n_rows, 2, figsize=(11, 2.0 + 1.7 * n_rows),
                             sharex=True)
    fig.suptitle("Modbus + realistic link delays — signal flow POI → inverters\\n"
                 "(reference → PPC reading via wire → pydae ground truth, "
                 "then PPC → inverter)",
                 fontsize=11)

    # Row 0 — POI summary
    ax = axes[0][0]
    ax.plot(step_rel, step["target_mw"], "--", color="#888888", lw=1.5,
            label="target_mw (reference)")
    ax.plot(step_rel, step["poi_mw"], "-", color="#1f77b4", lw=1.5,
            label="poi_mw (PPC reading)")
    if "sim_p_line_POI_GRID" in step:
        ax.plot(step_rel, [v / 1e6 for v in step["sim_p_line_POI_GRID"]],
                "-", color="#2ca02c", lw=1.2,
                label="pydae ground truth (p_line_POI_GRID)")
    ax.set_title("POI — active power")
    ax.set_ylabel("MW")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[0][1]
    ax.plot(step_rel, step["target_mvar"], "--", color="#888888", lw=1.5,
            label="target_mvar (reference)")
    ax.plot(step_rel, step["poi_mvar"], "-", color="#1f77b4", lw=1.5,
            label="poi_mvar (PPC reading)")
    if "sim_q_line_POI_GRID" in step:
        ax.plot(step_rel, [v / 1e6 for v in step["sim_q_line_POI_GRID"]],
                "-", color="#2ca02c", lw=1.2,
                label="pydae ground truth (q_line_POI_GRID)")
    ax.set_title("POI — reactive power")
    ax.set_ylabel("Mvar")
    ax.grid(True, alpha=0.3)

    # Rows 1..N: per-inverter
    for row_idx, inv_id in enumerate(INV_IDS, start=1):
        sp_p  = f"pv{row_idx}_sp_mw"
        sp_q  = f"pv{row_idx}_sp_mvar"
        sim_p = f"sim_p_g_{inv_id}"
        sim_q = f"sim_q_g_{inv_id}"

        # Active (col 0)
        ax = axes[row_idx][0]
        if sp_p in step:
            ax.plot(step_rel, step[sp_p], "-", color="#1f77b4", lw=1.5,
                    label="PPC sent (MW)")
        if arrivals[inv_id] is not None:
            arr_t, arr_p_pu, _ = arrivals[inv_id]
            ax.plot(arr_t, [v * P_NOMINAL_MW for v in arr_p_pu],
                    "-", color="#ff7f0e", lw=1.2, label="PV received")
        if sim_p in step:
            ax.plot(step_rel, [v * P_NOMINAL_MW for v in step[sim_p]],
                    "-", color="#2ca02c", lw=1.2, label="Inverter output")
        ax.set_title(f"INV {inv_id} — active")
        ax.set_ylabel("MW")
        ax.grid(True, alpha=0.3)
        if row_idx == 1:
            ax.legend(loc="lower right", fontsize=8)

        # Reactive (col 1)
        ax = axes[row_idx][1]
        if sp_q in step:
            ax.plot(step_rel, step[sp_q], "-", color="#1f77b4", lw=1.5,
                    label="PPC sent (Mvar)")
        if arrivals[inv_id] is not None:
            arr_t, _, arr_q_pu = arrivals[inv_id]
            ax.plot(arr_t, [v * Q_NOMINAL_MVAR for v in arr_q_pu],
                    "-", color="#ff7f0e", lw=1.2, label="PV received")
        if sim_q in step:
            ax.plot(step_rel, [v * Q_NOMINAL_MVAR for v in step[sim_q]],
                    "-", color="#2ca02c", lw=1.2, label="Inverter output")
        ax.set_title(f"INV {inv_id} — reactive")
        ax.set_ylabel("Mvar")
        ax.grid(True, alpha=0.3)

    for ax in axes[-1]:
        ax.set_xlabel("t (s, from recorder start)")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(out_path)
    print(f"[plot_flow] wrote {out_path}")


if __name__ == "__main__":
    main()
'''


def _readme_top(name: str, M: int, N: int) -> str:
    n_inv = M * N
    return f'''# {name} — protocol benchmark suite

Auto-generated by ``pydae.bps.builders.pv_designer``. Parallel
co-simulations of the **{name}** plant ({M} feeder(s) × {N} inverter(s)
per feeder = {n_inv} inverters total) — one per SCADA stack supported
by ``colinker``, plus a no-Mininet ``local`` case. Same electrical model,
same PI control law, same step magnitudes. Only the wire layer changes.

## Layout

```
{name}/
├── modbus/    # pymodbus 3.13
├── iec104/    # c104 v2.2.x  (M_ME_NC_1 / C_SE_NC_1)
├── dnp3/      # in-tree pure-Python codec, AI G30v5 / AO G41v3
├── dds/       # cyclonedds  (OpenFMB-style profiles)
└── local/     # direct pydae HTTP — pydae_ppc, no Mininet, no wire stack
```

Each subdir carries `{name}.json` and a stack-specific `README.md`. The four
protocol folders carry `run.py` (Mininet cosim); `local/` carries `run.sh`
(pydae_sim + pydae_ppc). `modbus/`, `iec104/` and `local/` additionally carry
the structured `tests/` suite (open-loop / closed-loop step, frequency
response, MITM); `dnp3/`/`dds/` use a flat `run_step.sh` + `results/`.

## Workflow (any protocol)

```bash
# Terminal A — launch the cosim (stays attached to mininet>)
cd <protocol>/
sudo python3 run.py
```

Wait for the topology banner `PPC API (macOS): http://localhost:8090/`.

```bash
# Terminal B — drive the step, record, plot
cd modbus/      && PY=/home/jmmmacubu/miniforge3/envs/pydae_dev/bin/python3 \\
                && $PY tests/closed_loop_step/run.py   # modbus/iec104: tests/ suite
cd dnp3/        && ./run_step.sh                        # dnp3/dds: flat run_step.sh
```

Tear down: `exit` at the `mininet>` prompt in Terminal A.
'''


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

_LOCAL_RUN_SH = '''#!/usr/bin/env bash
# Direct (no-Mininet) plant: pydae_sim + pydae_ppc, talking over plain HTTP.
#
# Brings up the bare pydae model and the controller in one terminal; the PPC
# management API is then reachable at http://localhost:<API_PORT>/ for the test
# drivers (Terminal B).  Ctrl-C stops the controller; the simulator is torn
# down on exit.
#
#   ./run.sh                 # defaults below
#   PYDAE_PORT=8001 API_PORT=5500 ./run.sh
set -euo pipefail

_HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-/home/jmmmacubu/miniforge3/envs/pydae_dev/bin/python3}"
JSON="$_HERE/__JSON_FILENAME__"
PYDAE_PORT="${PYDAE_PORT:-8001}"
API_PORT="${API_PORT:-5500}"

echo "[local] launching pydae_sim on :$PYDAE_PORT (building model) …"
"$PY" -m colinker.pydae_sim --json "$JSON" --port "$PYDAE_PORT" &
SIM_PID=$!
trap 'echo; echo "[local] stopping pydae_sim ($SIM_PID)"; kill "$SIM_PID" 2>/dev/null || true' EXIT

echo "[local] waiting for pydae_sim …"
until curl -s -o /dev/null --max-time 2 "http://localhost:$PYDAE_PORT/cosim/measurements"; do
    kill -0 "$SIM_PID" 2>/dev/null || { echo "[local] ERROR: pydae_sim died"; exit 1; }
    sleep 1
done

echo "[local] pydae_ppc -> http://localhost:$PYDAE_PORT   PPC API on :$API_PORT"
echo "[local] (Terminal B:  PPC_URL=http://localhost:$API_PORT \\$PY tests/frequency_response/run.py --mode p)"
exec "$PY" -m colinker.pydae_ppc --json "$JSON" \\
    --pydae-url "http://localhost:$PYDAE_PORT" \\
    --api-host 0.0.0.0 --api-port "$API_PORT"
'''


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


def _write_local_case(out_dir, plant, name, M, N, json_filename):
    """Emit the ``local/`` folder: direct pydae-HTTP controller (pydae_ppc),
    a run.sh launching pydae_sim + pydae_ppc, the plant JSON + xy_0 seeds,
    the structured tests/ suite, and a README.

    No Mininet, no wire protocol — the device-level ``protocol`` field is left
    as modbus (pydae_ppc reads the modbus_codec ``pydae_key``/``emec_scale``
    fields and ignores the rest)."""
    sub = os.path.join(out_dir, "local")
    os.makedirs(sub, exist_ok=True)

    with open(os.path.join(sub, json_filename), "w", encoding="utf-8") as f:
        json.dump(plant, f, indent=2)

    xy_0_json = json.dumps(make_xy_0(plant), indent=4)
    for fname in (f"{name}_xy_0.json", "xy_0.json"):
        with open(os.path.join(sub, fname), "w", encoding="utf-8") as f:
            f.write(xy_0_json)

    run_sh = _LOCAL_RUN_SH.replace("__JSON_FILENAME__", json_filename)
    run_sh_path = os.path.join(sub, "run.sh")
    with open(run_sh_path, "w", encoding="utf-8") as f:
        f.write(run_sh)
    os.chmod(run_sh_path, 0o755)

    _write_tests_suite(sub)

    readme = _readme_local(M, N).replace("{name}", name)
    with open(os.path.join(sub, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)


def _report_notebook(name: str) -> str:
    """Return a Jupyter notebook (nbformat 4) that overlays the step.csv
    from each protocol's results/ folder for direct comparison."""
    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {},
             "source": [
                f"# {name} — protocol comparison\n",
                "\n",
                "After running `./run_step.sh` inside each `<protocol>/` "
                "directory, the step-response CSVs live at "
                "`../<protocol>/results/step.csv`. This notebook overlays "
                "the four traces for direct comparison."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None,
             "outputs": [], "source": [
                "import os\n",
                "import pandas as pd\n",
                "import plotly.express as px\n",
                "\n",
                "PROTOCOLS = ['modbus', 'iec104', 'dnp3', 'dds']\n",
                "ROOT = os.path.abspath(os.path.join(os.getcwd(), '..'))\n",
                "\n",
                "frames = []\n",
                "for proto in PROTOCOLS:\n",
                "    path = os.path.join(ROOT, proto, 'results', 'step.csv')\n",
                "    if not os.path.exists(path):\n",
                "        print(f'missing: {path}')\n",
                "        continue\n",
                "    df = pd.read_csv(path, parse_dates=['ts_iso'])\n",
                "    df['protocol'] = proto\n",
                "    df['t_s'] = (df['ts_iso'] - df['ts_iso'].iloc[0])"
                ".dt.total_seconds()\n",
                "    frames.append(df)\n",
                "\n",
                "data = pd.concat(frames, ignore_index=True) "
                "if frames else pd.DataFrame()\n",
                "data.head()"]},
            {"cell_type": "markdown", "metadata": {},
             "source": ["## Active power at POI — all protocols"]},
            {"cell_type": "code", "metadata": {}, "execution_count": None,
             "outputs": [], "source": [
                "fig = px.line(\n",
                "    data, x='t_s', y='poi_mw', color='protocol',\n",
                "    title='POI active power (MW) — step response',\n",
                "    labels={'t_s': 'Time (s)', 'poi_mw': 'Power (MW)'},\n",
                "    template='plotly_white',\n",
                ")\n",
                "fig.add_scatter(x=data['t_s'], y=data['target_mw'], "
                "mode='lines', name='target', line=dict(dash='dot', "
                "color='black'))\n",
                "fig.update_layout(hovermode='x unified')\n",
                "fig.show()"]},
            {"cell_type": "markdown", "metadata": {},
             "source": ["## Reactive power at POI — all protocols"]},
            {"cell_type": "code", "metadata": {}, "execution_count": None,
             "outputs": [], "source": [
                "fig = px.line(\n",
                "    data, x='t_s', y='poi_mvar', color='protocol',\n",
                "    title='POI reactive power (Mvar) — step response',\n",
                "    labels={'t_s': 'Time (s)', 'poi_mvar': 'Q (Mvar)'},\n",
                "    template='plotly_white',\n",
                ")\n",
                "fig.add_scatter(x=data['t_s'], y=data['target_mvar'], "
                "mode='lines', name='target', line=dict(dash='dot', "
                "color='black'))\n",
                "fig.update_layout(hovermode='x unified')\n",
                "fig.show()"]},
            {"cell_type": "markdown", "metadata": {},
             "source": [
                "## Per-inverter setpoints (one protocol at a time)\n",
                "\n",
                "Pick a protocol from the dropdown above the chart to "
                "see how the PPC fanned out the POI command across the "
                "M*N inverters."]},
            {"cell_type": "code", "metadata": {}, "execution_count": None,
             "outputs": [], "source": [
                "SHOW = 'modbus'  # change to iec104 / dnp3 / dds\n",
                "sub = data[data['protocol'] == SHOW]\n",
                "sp_cols = sorted([c for c in sub.columns "
                "if c.startswith('pv') and c.endswith('_sp_mw')])\n",
                "if sub.empty or not sp_cols:\n",
                "    print(f'no data for {SHOW}')\n",
                "else:\n",
                "    fig = px.line(\n",
                "        sub, x='t_s', y=sp_cols,\n",
                "        title=f'Per-inverter active setpoints ({SHOW})',\n",
                "        labels={'t_s': 'Time (s)', 'value': 'P_sp (MW)'},\n",
                "        template='plotly_white',\n",
                "    )\n",
                "    fig.update_layout(hovermode='x unified')\n",
                "    fig.show()"]},
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"},
            "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"cell-{i:02d}"
    return json.dumps(nb, indent=1)


def _write_tests_suite(sub):
    """Copy the structured `tests/` suite (lib + 4 experiment drivers + READMEs)
    from ``templates/tests/`` into a protocol folder, and create the per-test
    ``csv/`` and ``plots/`` output dirs.

    The drivers are name-agnostic — ``tests/lib/common.py`` auto-discovers the
    plant JSON (the one carrying a ``ppc`` block) and reads the inverter list
    and gains from it — so the same files work for any M×N plant.  Emitted for
    every protocol in ``_SUITE_PROTOCOLS`` — they all expose the same shared
    ``ppc_core`` ``/sine`` + feedforward control API the suite drives."""
    src = os.path.join(_HERE, "templates", "tests")
    dst = os.path.join(sub, "tests")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(
        "__pycache__", "*.pyc", ".DS_Store"))
    for t in ("open_loop_step", "closed_loop_step",
              "frequency_response", "mitm_attack"):
        os.makedirs(os.path.join(dst, t, "csv"), exist_ok=True)
        os.makedirs(os.path.join(dst, t, "plots"), exist_ok=True)
    for run_py in glob.glob(os.path.join(dst, "*", "run.py")):
        os.chmod(run_py, 0o755)


def write_suite(out_dir, plant, name, M, N):
    """Emit the four-protocol benchmark folder tree around ``plant``.

    The plant JSON is identical across protocols; only the ``protocol``
    field on each device entry varies, since ``configs`` in the template
    already carries both modbus (``register``/``format``/``type``) and
    iec104 (``io_register``/``type_iec104``) fields per point.
    """
    os.makedirs(out_dir, exist_ok=True)
    json_filename = f"{name}.json"

    # PV hosts list for the modbus run.py (one entry per inverter).
    pv_hosts = [f"LV{f'{i_m}'.zfill(2)}{f'{i_n}'.zfill(2)}"
                for i_m in range(1, M + 1) for i_n in range(1, N + 1)]
    pv_hosts_literal = "[" + ", ".join(f'"{h}"' for h in pv_hosts) + "]"
    # Default MITM spoof = 0.5 × (total plant nameplate) so the over-power
    # demo reads "below limit" while the real plant runs above it.
    p_max_total_mw  = sum(w.get("S_n", 1e6) for w in plant.get("weccs", [])) / 1e6
    mitm_default_mw = round(0.5 * p_max_total_mw, 2)
    # POI ports from the devices block (first poi device); fall back to defaults.
    poi_modbus_port = 502
    poi_iec104_port = 2404
    poi_dnp3_port   = 20000
    for d in plant.get("devices", []):
        if d.get("config") == "poi":
            poi_modbus_port = int(d.get("modbus_port", 502))
            poi_iec104_port = int(d.get("iec104_port", 2404))
            poi_dnp3_port   = int(d.get("dnp3_port", 20000))
            break
    # IOA / DNP3 object index of the POI active-power measurement (the MITM
    # target), read from the poi config's ActivePower point.  The DNP3 index
    # defaults to the point's position within the measurements list (matching
    # colinker.dnp3_codec.load_dnp3_points), unless an explicit dnp3_index is
    # set; the IEC 104 IOA comes from io_register.  Fall back to 1001 / 1.
    poi_p_ioa   = 1001
    poi_p_index = 1
    poi_cfg = (plant.get("configs", {}) or {}).get("poi", {})
    for i, m in enumerate(poi_cfg.get("measurements", [])):
        if m.get("ing_name") == "ActivePower" or (m.get("emec_name") or "").startswith("p_line"):
            poi_p_ioa   = int(m.get("io_register", 1001))
            poi_p_index = int(m.get("dnp3_index", i))
            break

    for protocol in ("modbus", "iec104", "dnp3", "dds"):
        sub = os.path.join(out_dir, protocol)
        os.makedirs(sub, exist_ok=True)
        # modbus + iec104 carry the structured tests/ suite (each with its own
        # per-test csv/ + plots/); dnp3/dds keep the flat results/ + run_step.sh.
        has_suite = protocol in _SUITE_PROTOCOLS
        if not has_suite:
            os.makedirs(os.path.join(sub, "results"), exist_ok=True)

        # Per-protocol copy with the device-level protocol field stamped.
        plant_p = json.loads(json.dumps(plant))
        for d in plant_p.get("devices", []):
            d["protocol"] = protocol

        # Protocol-appropriate control period.  IEC 104's event-driven c104
        # link can't sustain the 10 Hz (0.1 s) loop the other stacks use:
        # sampling/actuating faster than it can read+deliver saturates the
        # single client thread and, once telemetry is made fresh, drives the
        # loop (same PI gains) into a limit cycle.  A slower loop both cuts the
        # command rate and detunes the per-cycle velocity-form integral.  The
        # benchmark invariant is "same PI gains, protocol-appropriate sample
        # time".  See copv_*/iec104/iec104_saturation.md.
        if protocol == "iec104" and "ppc" in plant_p:
            plant_p["ppc"]["ctrl_dt_s"] = _IEC104_CTRL_DT_S

        # IEC 104 and DNP3 carry the measured value already scaled to
        # engineering units on the wire (proxy: wire = pydae_W * emec_scale)
        # and the PPC reads it directly, so power points must use
        # emec_scale=1e-6 (W→MW).  Modbus instead stores raw W and divides by
        # 1e6 in the PPC, so it keeps 1.0.  Voltage points are ~1 pu and stay
        # at their scale.  Power is identified by ing_name / emec_name.
        if protocol in _ENGINEERING_UNIT_WIRE_PROTOCOLS:
            for cfg in (plant_p.get("configs", {}) or {}).values():
                for m in cfg.get("measurements", []):
                    nm = m.get("emec_name") or ""
                    if m.get("ing_name") in ("ActivePower", "ReactivePower") \
                            or nm.startswith(("p_", "q_")):
                        m["emec_scale"] = 1e-6

        with open(os.path.join(sub, json_filename), "w",
                  encoding="utf-8") as f:
            json.dump(plant_p, f, indent=2)

        xy_0      = make_xy_0(plant_p)
        xy_0_json = json.dumps(xy_0, indent=4)
        for fname in (f"{name}_xy_0.json", "xy_0.json"):
            with open(os.path.join(sub, fname), "w",
                      encoding="utf-8") as f:
                f.write(xy_0_json)

        run_py = (_RUN_PY_TEMPLATE_BY_PROTOCOL.get(protocol, _RUN_PY_TEMPLATE)
                  .replace("__PROTOCOL_LABEL__", _PROTOCOL_LABEL[protocol])
                  .replace("__JSON_FILENAME__", json_filename)
                  .replace("__PV_HOSTS_LITERAL__", pv_hosts_literal)
                  .replace("__MITM_DEFAULT_VALUE_MW__", str(mitm_default_mw))
                  .replace("__POI_MODBUS_PORT__", str(poi_modbus_port))
                  .replace("__POI_IEC104_PORT__", str(poi_iec104_port))
                  .replace("__POI_DNP3_PORT__", str(poi_dnp3_port))
                  .replace("__POI_P_IOA__", str(poi_p_ioa))
                  .replace("__POI_P_INDEX__", str(poi_p_index)))
        run_py_path = os.path.join(sub, "run.py")
        with open(run_py_path, "w", encoding="utf-8") as f:
            f.write(run_py)
        os.chmod(run_py_path, 0o755)

        if not has_suite:
            run_sh_path = os.path.join(sub, "run_step.sh")
            with open(run_sh_path, "w", encoding="utf-8") as f:
                f.write(_RUN_STEP_SH)
            os.chmod(run_sh_path, 0o755)

        # The structured tests/ suite (shared ppc_core /sine + feedforward API)
        # is emitted for every suite protocol.  plot_flow.py is Modbus-specific.
        if has_suite:
            _write_tests_suite(sub)

        if protocol == "modbus":
            plot_flow = _PLOT_FLOW_TEMPLATE.replace(
                "__PV_IDS_LITERAL__", pv_hosts_literal
            )
            plot_path = os.path.join(sub, "plot_flow.py")
            with open(plot_path, "w", encoding="utf-8") as f:
                f.write(plot_flow)
            os.chmod(plot_path, 0o755)

        readme = _README_FN[protocol](M, N).replace("{name}", name)
        with open(os.path.join(sub, "README.md"), "w",
                  encoding="utf-8") as f:
            f.write(readme)

    # local case — direct pydae-HTTP controller, no Mininet (uses the
    # unstamped plant so pydae_ppc reads the modbus_codec key/scale fields).
    _write_local_case(out_dir, plant, name, M, N, json_filename)

    with open(os.path.join(out_dir, "README.md"), "w",
              encoding="utf-8") as f:
        f.write(_readme_top(name, M, N))

    reporter_dir = os.path.join(out_dir, "reporter")
    os.makedirs(reporter_dir, exist_ok=True)
    with open(os.path.join(reporter_dir, "report.ipynb"), "w",
              encoding="utf-8") as f:
        f.write(_report_notebook(name))

    print(f"wrote suite at {out_dir}: M={M} N={N} ({M * N} inverters)")


__all__ = ["designer", "make_xy_0", "write_suite"]


if __name__ == "__main__":
    # Plant parameters
    M, N = 5, 10
    NAME = f"copv_{M}_{N}"
    OUT_DIR = f"/Users/jmmauricio/workspace/benchmarks_colinker/{NAME}_wecc"

    # 1. Build the plant dict (and write a flat <NAME>.json next to this file)
    plant = designer(
        M, N,
        output_file=os.path.join(_HERE, f"{NAME}.json"),
        P_inv_max=1e6, U_lv=800, U_mv=30e3, U_hv=132e3,
        Pref_pq=0.5, Qext_pq=0.1,
    )

    # 2. Scaffold the four-protocol suite around it.
    write_suite(OUT_DIR, plant, name=NAME, M=M, N=N)

    # 3. Sanity-check: build and init the model via pydae.
    from pydae.bps import BpsBuilder
    from pydae.core.builder.casadi_builder import CasadiBuilder
    from pydae.core.model.casadi_model import CasadiModel
    from pydae.bps.utils.reporter import report_all

    plant_json = os.path.join(OUT_DIR, "modbus", f"{NAME}.json")
    xy_0_json  = os.path.join(OUT_DIR, "modbus", "xy_0.json")
    print(f"\n[sanity] BpsBuilder({plant_json})")
    grid = BpsBuilder(plant_json, use_casadi=True)
    grid.construct(plant_json)
    bld = CasadiBuilder(grid.sys_dict).build()

    model = CasadiModel(bld)
    model.ini({}, xy_0_json)
    report_all(model, plant_json)
