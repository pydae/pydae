"""
Compare four bus-frequency estimators on a 2-bus / 1-genape system.

System
======

    bus 1  ────  line  ────  bus 2
      │                         │
      │                       genape (S_n = 1 GVA, F_n = 50 Hz)
      ▼
    pll, sogi, sogi_fll, sogi_pll
    (all four attached at bus 1)

Disturbance
===========

The genape's ``alpha_2`` input is the rate-of-change of frequency in pu.
We drive a 1 Hz/s ramp for 1 s (alpha = 1/F_n = 0.02 pu/s for 1 s), then
release. The grid frequency therefore goes from 50 Hz at t = 0.5 s to
51 Hz at t = 1.5 s and stays there.

Phasor-frame caveat
===================

The PI ``pll`` and the phasor-frame ``sogi`` read the bus phasor
(V_1, theta_1) directly and track the frequency without any external
driving. The textbook ``sogi_fll`` and ``sogi_pll`` need an
*instantaneous* AC voltage sample at ``v_grid``; this script feeds them
a lab-frame reconstruction ``V_1 * cos(theta_lab + theta_1)`` where
``theta_lab`` is integrated from the COI frequency on the Python side.
The small Dt (= 1e-4 s) resolves the 50 Hz carrier.
"""
import os
import sys
from pathlib import Path

import numpy as np

from pydae.bps import BpsBuilder
from pydae.core import Builder, Model

HERE       = Path(__file__).resolve().parent
HJSON_PATH = HERE / "pll_compare.hjson"
MODEL_NAME = "pll_compare"

F_N      = 50.0           # Hz
OMEGA_B  = 2 * np.pi * F_N
DT       = 1.0e-4         # s — fine enough to resolve the 50 Hz carrier
T_END    = 3.0            # s
T_RAMP_0 = 0.5            # s — ramp starts
T_RAMP_1 = 1.5            # s — ramp ends
ALPHA_RAMP = 1.0 / F_N    # pu/s == 1 Hz/s


def build():
    """Compile the system."""
    os.chdir(HERE)
    grid = BpsBuilder(str(HJSON_PATH))
    grid.checker()
    grid.uz_jacs = False
    grid.construct(MODEL_NAME)
    Builder(grid.sys_dict, target="ctypes", sparse=False).build()


def simulate():
    """Run the ramp scenario and return the populated model."""
    os.chdir(HERE)
    model = Model(MODEL_NAME)
    model.Dt         = DT
    # Internal buffer is allocated to 10_000 rows at construction; using
    # decimation = 10 stores every 10th step so we fit 30_000 sim steps.
    model.decimation = 10

    # Seed every state explicitly — pydae's model.ini does not apply the
    # sys_dict xy_0_dict defaults unless a file is loaded, so any state
    # we skip would default to zero (singular for V/omega).
    omega_0 = OMEGA_B
    xy0 = {
        # Bus phasors
        "V_1": 1.0, "theta_1": 0.0,
        "V_2": 1.0, "theta_2": 0.0,
        # System COI
        "omega_coi": 1.0,
        # genape (states + algebraic omega)
        "delta_2": 0.0, "Domega_2": 0.0, "Dv_2": 0.0,
        "v_ref_filtered_2": 1.0, "omega_2": 1.0,
        # pll
        "theta_pll_1": 0.0, "xi_pll_1": 0.0,
        "omega_pll_f_1": 1.0, "rocof_pll_f_1": 0.0, "rocof_pll_1": 0.0,
        # sogi
        "x_v_sogi_sogi1": 0.0, "x_qv_sogi_sogi1": 0.0,
        "omega_sogi_sogi1": 1.0, "theta_sogi_sogi1": 0.0,
        "omega_sogi_f_sogi1": 1.0, "rocof_sogi_f_sogi1": 0.0,
        "rocof_sogi_sogi1": 0.0,
        # sogi_fll (omega_est is rad/s here)
        "v_p_sogi_fll_fll1": 0.0, "qv_p_sogi_fll_fll1": 0.0,
        "omega_est_sogi_fll_fll1": omega_0,
        "err_v_sogi_fll_fll1": 0.0,
        # sogi_pll
        "v_p_sogi_pll_pll1": 0.0, "qv_p_sogi_pll_pll1": 0.0,
        "x_pi_sogi_pll_pll1": 0.0, "theta_est_sogi_pll_pll1": 0.0,
        "err_v_sogi_pll_pll1": 0.0,
        "v_d_sogi_pll_pll1": 0.0, "v_q_sogi_pll_pll1": 0.0,
        "omega_est_sogi_pll_pll1": omega_0,
    }
    model.ini({}, xy0)

    # Stepwise loop: at each Dt, set alpha and feed a lab-frame v_grid
    # sample to the textbook SOGI variants.
    n_steps = int(round(T_END / DT))
    theta_lab = 0.0

    for k in range(1, n_steps + 1):
        t = k * DT

        # 1 Hz/s ramp on alpha during [T_RAMP_0, T_RAMP_1].
        alpha_val = ALPHA_RAMP if (T_RAMP_0 <= t <= T_RAMP_1) else 0.0

        # Lab-frame AC reconstruction: integrate omega_coi (in pu) into
        # a lab-frame angle, project the current bus phasor onto it.
        omega_coi_pu = model.get_value("omega_coi") or 1.0
        theta_lab   += omega_coi_pu * OMEGA_B * DT
        V_1         = model.get_value("V_1") or 1.0
        theta_1     = model.get_value("theta_1") or 0.0
        v_grid_val  = V_1 * np.cos(theta_lab + theta_1)

        model.run(t, {
            "alpha_2":               alpha_val,
            "v_grid_sogi_fll_fll1":  v_grid_val,
            "v_grid_sogi_pll_pll1":  v_grid_val,
        })

    model.post()
    return model


def plot(model):
    """Build the Plotly comparison figure and save as HTML."""
    import plotly.graph_objects as go

    t = model.Time

    # Ground truth: COI angular speed -> Hz.
    f_true = model.get_values("omega_coi") * F_N

    # Each estimator's Hz output.
    f_pll      = model.get_values("frequency_pll_1")
    f_sogi     = model.get_values("frequency_sogi_sogi1")
    f_sogi_fll = model.get_values("f_est_sogi_fll_fll1")
    f_sogi_pll = model.get_values("f_est_sogi_pll_pll1")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=f_true,
                             name="omega_coi · F_n  (ground truth)",
                             line=dict(color="black", width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=t, y=f_pll,      name="pll  (PI, phasor)"))
    fig.add_trace(go.Scatter(x=t, y=f_sogi,     name="sogi (FLL, phasor)"))
    fig.add_trace(go.Scatter(x=t, y=f_sogi_fll, name="sogi_fll (textbook, EMT)"))
    fig.add_trace(go.Scatter(x=t, y=f_sogi_pll, name="sogi_pll (textbook, EMT)"))

    fig.update_layout(
        title=("Bus-frequency estimators on a 2-bus / genape system — "
               "1 Hz/s ramp from t = 0.5 s for 1 s"),
        xaxis_title="time [s]",
        yaxis_title="frequency [Hz]",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1.0),
    )

    # Shade the ramp window.
    fig.add_vrect(x0=T_RAMP_0, x1=T_RAMP_1,
                  fillcolor="orange", opacity=0.10, line_width=0,
                  annotation_text="alpha = 1 Hz/s",
                  annotation_position="top left")

    out_path = HERE / "pll_compare.html"
    fig.write_html(str(out_path))
    print(f"Plot written: {out_path}")
    return fig


if __name__ == "__main__":
    print("Building model (compiles C code on first run)...")
    build()
    print("Simulating 3 s at Dt = 1e-4 s (this takes a moment)...")
    model = simulate()
    print("Final values:")
    print(f"  f_true     = {model.get_values('omega_coi')[-1] * F_N:.4f} Hz")
    print(f"  pll        = {model.get_values('frequency_pll_1')[-1]:.4f} Hz")
    print(f"  sogi       = {model.get_values('frequency_sogi_sogi1')[-1]:.4f} Hz")
    print(f"  sogi_fll   = {model.get_values('f_est_sogi_fll_fll1')[-1]:.4f} Hz")
    print(f"  sogi_pll   = {model.get_values('f_est_sogi_pll_pll1')[-1]:.4f} Hz")
    plot(model)
