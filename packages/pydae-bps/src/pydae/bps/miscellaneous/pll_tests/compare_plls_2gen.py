"""
Compare four bus-frequency estimators on a 2-machine system (CasADi backend).

System
======

    bus 1  ─── line (X=0.5, R=0.05) ─── bus 2
      │                                  │
    syn G1 (milano4ord, H=3)        syn G2 (milano4ord, H=6)
    + AVR sexs, GOV tgov1, LC      + AVR sexs, GOV tgov1, LC
    + ZIP load (50 MW, 10 Mvar)
    + pll, sogi, sogi_fll, sogi_pll

Disturbance
===========

At t = 1.0 s we step the constant-power load at bus 1 by +20 MW (pu p_p
from 0.5 to 0.7 on system base) for 1 s, then release. The local machine
G1 (small H) absorbs more transient impact than the remote machine G2
(large H, weak tie), so ``omega_G1`` and ``omega_G2`` split — exactly
the situation where the PLL semantics ("track local rotor speed at
bus 1") become non-trivial.

Truth signal
============

PLLs sit at bus 1 → truth is ``omega_G1``, the rotor speed of the local
generator. ``omega_coi`` is plotted for context but it is the
inertia-weighted average and lags G1's swing.

Phasor-frame caveat
===================

``sogi_fll`` / ``sogi_pll`` are EMT blocks that want an instantaneous AC
sample. We feed them ``V_1 * cos(theta_lab + theta_1)`` where
``theta_lab`` is integrated from ``omega_coi`` at the simulation step.
"""
import os
from pathlib import Path

import numpy as np

from pydae.bps import BpsBuilder
from pydae.core.builder.casadi_builder import CasadiBuilder
from pydae.core.model.casadi_model import CasadiModel

HERE       = Path(__file__).resolve().parent
HJSON_PATH = HERE / "pll_compare_2gen.hjson"
MODEL_NAME = "pll_compare_2gen"

F_N      = 50.0
OMEGA_B  = 2 * np.pi * F_N
DT       = 2.0e-4
T_END    = 5.0
T_STEP_0 = 1.0
T_STEP_1 = 2.0

# ZIP-load run inputs are p_p / q_p (constant-power component, pu on
# S_base). Initial value at ini = 0.5 (50 MW / 100 MVA system base).
P_P_BASE = 50.0e6 / 100e6
Q_P_BASE = 10.0e6 / 100e6
P_P_STEP = P_P_BASE + 0.20   # +20 MW
Q_P_STEP = Q_P_BASE


def build():
    """Compile the system using the CasADi backend."""
    os.chdir(HERE)
    grid = BpsBuilder(str(HJSON_PATH), use_casadi=True)
    grid.checker()
    grid.uz_jacs = False
    grid.construct(MODEL_NAME)
    bld = CasadiBuilder(grid.sys_dict).build()
    return bld


def simulate(bld):
    """Run the load step scenario."""
    os.chdir(HERE)
    model = CasadiModel(bld)
    model.Dt = DT

    # The construct() step writes a default xy_0 file alongside the data
    # JSON; use it as the Newton initial guess.
    xy_0_path = HERE / f"{MODEL_NAME}_xy_0.json"
    model.ini({}, xy_0=str(xy_0_path), newton_tol=1e-10)

    print("Steady-state after ini:")
    for k in ("V_1", "V_2", "omega_G1", "omega_G2", "omega_coi",
              "p_g_G1", "p_g_G2"):
        print(f"  {k:12s} = {model.get_value(k):+.5f}")

    n_steps   = int(round(T_END / DT))
    theta_lab = 0.0

    for k in range(1, n_steps + 1):
        t = k * DT

        if T_STEP_0 <= t <= T_STEP_1:
            p_p, q_p = P_P_STEP, Q_P_STEP
        else:
            p_p, q_p = P_P_BASE, Q_P_BASE

        # Lab-frame AC reconstruction for the EMT-style PLLs.
        omega_coi_pu = model.get_value("omega_coi") or 1.0
        theta_lab   += omega_coi_pu * OMEGA_B * DT
        V_1          = model.get_value("V_1") or 1.0
        theta_1      = model.get_value("theta_1") or 0.0
        v_grid_val   = V_1 * np.cos(theta_lab + theta_1)

        model.run(t, {
            "p_p_L1":                p_p,
            "q_p_L1":                q_p,
            "v_grid_sogi_fll_fll1":  v_grid_val,
            "v_grid_sogi_pll_pll1":  v_grid_val,
        })

    model.post()
    return model


def plot(model):
    """Plotly comparison figure."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    t = model.Time

    f_omega_G1  = model.get_values("omega_G1")  * F_N
    f_omega_G2  = model.get_values("omega_G2")  * F_N
    f_omega_coi = model.get_values("omega_coi") * F_N

    f_pll      = model.get_values("frequency_pll_1")
    f_sogi     = model.get_values("frequency_sogi_sogi1")
    f_sogi_fll = model.get_values("f_est_sogi_fll_fll1")
    f_sogi_pll = model.get_values("f_est_sogi_pll_pll1")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        subplot_titles=("Frequency estimates vs local rotor speed at bus 1",
                        "Estimation error: f_est − F_n·omega_G1"))

    fig.add_trace(go.Scatter(x=t, y=f_omega_G1,
                             name="F_n·omega_G1 (truth at bus 1)",
                             line=dict(color="black", width=1.8, dash="dash")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=f_omega_G2,
                             name="F_n·omega_G2",
                             line=dict(color="gray", width=1.0, dash="dot")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=f_omega_coi,
                             name="F_n·omega_coi",
                             line=dict(color="lightgray", width=1.0, dash="dot")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=f_pll,      name="pll  (PI, phasor)"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=f_sogi,     name="sogi (FLL, phasor)"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=f_sogi_fll, name="sogi_fll (EMT, recon. v)"),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=f_sogi_pll, name="sogi_pll (EMT, recon. v)"),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=t, y=f_pll      - f_omega_G1, name="err pll",
                             showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=f_sogi     - f_omega_G1, name="err sogi",
                             showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=f_sogi_fll - f_omega_G1, name="err sogi_fll",
                             showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=t, y=f_sogi_pll - f_omega_G1, name="err sogi_pll",
                             showlegend=False), row=2, col=1)

    fig.update_yaxes(title_text="frequency [Hz]", row=1, col=1)
    fig.update_yaxes(title_text="error [Hz]",      row=2, col=1)
    fig.update_xaxes(title_text="time [s]",        row=2, col=1)

    fig.update_layout(
        title=("Bus-frequency estimators on a 2-machine system — "
               f"+20 MW load step at bus 1, t ∈ "
               f"[{T_STEP_0}, {T_STEP_1}] s"),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.06,
                    xanchor="right", x=1.0),
        height=750,
    )

    fig.add_vrect(x0=T_STEP_0, x1=T_STEP_1,
                  fillcolor="orange", opacity=0.10, line_width=0,
                  annotation_text="+20 MW load",
                  annotation_position="top left",
                  row=1, col=1)
    fig.add_vrect(x0=T_STEP_0, x1=T_STEP_1,
                  fillcolor="orange", opacity=0.10, line_width=0,
                  row=2, col=1)

    out_path = HERE / "pll_compare_2gen.html"
    fig.write_html(str(out_path))
    print(f"Plot written: {out_path}")
    return fig


if __name__ == "__main__":
    print("Building model (CasADi backend)...")
    bld = build()
    print(f"Simulating {T_END:.1f} s at Dt = {DT:.0e} s (IDAS)...")
    model = simulate(bld)
    print("\nFinal values:")
    print(f"  omega_G1·F_n  = {model.get_values('omega_G1')[-1]   * F_N:.4f} Hz")
    print(f"  omega_G2·F_n  = {model.get_values('omega_G2')[-1]   * F_N:.4f} Hz")
    print(f"  omega_coi·F_n = {model.get_values('omega_coi')[-1] * F_N:.4f} Hz")
    print(f"  pll           = {model.get_values('frequency_pll_1')[-1]:.4f} Hz")
    print(f"  sogi          = {model.get_values('frequency_sogi_sogi1')[-1]:.4f} Hz")
    print(f"  sogi_fll      = {model.get_values('f_est_sogi_fll_fll1')[-1]:.4f} Hz")
    print(f"  sogi_pll      = {model.get_values('f_est_sogi_pll_pll1')[-1]:.4f} Hz")
    plot(model)
