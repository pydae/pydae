"""Shared plotting — every figure is written twice: a static SVG (matplotlib)
and an interactive HTML (Plotly).  Used by all four tests.
"""
from __future__ import annotations

import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .common import SIM_P_COL, SIM_Q_COL


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------

def load_recorder_csv(path: str) -> dict:
    """Load a recorder CSV → arrays.  Ground truth converted W→MW / var→Mvar."""
    rows = list(csv.DictReader(open(path)))
    def col(name, scale=1.0, default=np.nan):
        return np.array([float(r.get(name, default) or default) * scale
                         for r in rows])
    return {
        "t":     col("t_elapsed"),
        "tgt_p": col("target_mw"),  "poi_p": col("poi_mw"),
        "tgt_q": col("target_mvar"), "poi_q": col("poi_mvar"),
        "sim_p": col(SIM_P_COL, 1e-6), "sim_q": col(SIM_Q_COL, 1e-6),
    }


# ---------------------------------------------------------------------------
# Step response (P + Q panels)
# ---------------------------------------------------------------------------

def step_figure(csv_path: str, out_base: str, title: str) -> dict:
    """Two stacked panels (P, Q): reference, PPC readback, pydae ground truth.
    Writes ``out_base.svg`` and ``out_base.html``; returns the loaded data."""
    d = load_recorder_csv(csv_path)
    t = d["t"]

    # matplotlib SVG
    fig, (ap, aq) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ap.plot(t, d["tgt_p"], "k--", lw=1, label="reference")
    ap.plot(t, d["sim_p"], lw=1.8, color="tab:green", label="plant (pydae)")
    ap.plot(t, d["poi_p"], lw=1.0, color="tab:red", alpha=0.8, label="PPC readback")
    ap.set_ylabel("POI P (MW)"); ap.grid(alpha=0.3); ap.legend(fontsize=8)
    ap.set_title(title)
    aq.plot(t, d["tgt_q"], "k--", lw=1, label="reference")
    aq.plot(t, d["sim_q"], lw=1.8, color="tab:green", label="plant (pydae)")
    aq.plot(t, d["poi_q"], lw=1.0, color="tab:red", alpha=0.8, label="PPC readback")
    aq.set_ylabel("POI Q (Mvar)"); aq.set_xlabel("t (s)"); aq.grid(alpha=0.3)
    aq.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_base + ".svg"); plt.close(fig)

    # Plotly HTML
    pf = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       subplot_titles=("Active power (MW)", "Reactive power (Mvar)"))
    for col, c, name in (("tgt_p", "black", "reference"),
                         ("sim_p", "green", "plant (pydae)"),
                         ("poi_p", "red", "PPC readback")):
        pf.add_trace(go.Scatter(x=t, y=d[col], name=name,
                                line=dict(color=c)), row=1, col=1)
    for col, c, name in (("tgt_q", "black", "reference"),
                         ("sim_q", "green", "plant (pydae)"),
                         ("poi_q", "red", "PPC readback")):
        pf.add_trace(go.Scatter(x=t, y=d[col], name=name, showlegend=False,
                                line=dict(color=c)), row=2, col=1)
    pf.update_layout(title=title, template="plotly_white", height=700)
    pf.update_xaxes(title_text="t (s)", row=2, col=1)
    pf.write_html(out_base + ".html", include_plotlyjs="cdn")
    return d


# ---------------------------------------------------------------------------
# Bode (magnitude + phase)
# ---------------------------------------------------------------------------

def bode_figure(freqs, gains, phases_deg, out_base: str, title: str) -> None:
    """Magnitude (dB) + phase (deg, unwrapped) vs frequency → SVG + HTML."""
    f = np.asarray(freqs, float)
    g_db = 20.0 * np.log10(np.asarray(gains, float))
    ph = np.degrees(np.unwrap(np.radians(np.asarray(phases_deg, float))))

    fig, (am, ax) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    am.plot(f, g_db, "o-", color="tab:blue")
    am.axhline(-3, ls="--", color="grey", lw=0.8, label="-3 dB")
    am.set_ylabel("Magnitude (dB)"); am.grid(alpha=0.3); am.legend(fontsize=8)
    am.set_title(title)
    ax.plot(f, ph, "o-", color="tab:red")
    ax.set_ylabel("Phase (deg)"); ax.set_xlabel("Frequency (Hz)"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_base + ".svg"); plt.close(fig)

    pf = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       subplot_titles=("Magnitude (dB)", "Phase (deg)"))
    pf.add_trace(go.Scatter(x=f, y=g_db, mode="lines+markers", name="|G|"),
                 row=1, col=1)
    pf.add_trace(go.Scatter(x=f, y=ph, mode="lines+markers", name="phase",
                            line=dict(color="red")), row=2, col=1)
    pf.update_layout(title=title, template="plotly_white", height=600)
    pf.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    pf.write_html(out_base + ".html", include_plotlyjs="cdn")


# ---------------------------------------------------------------------------
# Sweep time series (one trace per frequency)
# ---------------------------------------------------------------------------

def sweep_timeseries_figure(runs, out_base: str, title: str, ylabel: str) -> None:
    """``runs`` is a list of ``(freq, t, ref, meas)`` arrays → SVG + HTML."""
    n = len(runs)
    fig, axes = plt.subplots(n, 1, figsize=(9, 2.0 * n))
    axes = np.atleast_1d(axes)
    for ax, (f, t, ref, meas) in zip(axes, runs):
        ax.plot(t, ref, color="tab:blue", lw=1.4, label="reference")
        ax.plot(t, meas, color="tab:red", lw=1.0, label="POI")
        ax.set_title(f"f = {f:.2f} Hz", fontsize=9, loc="left")
        ax.set_ylabel(ylabel); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    axes[-1].set_xlabel("t (s)")
    fig.suptitle(title); fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(out_base + ".svg"); plt.close(fig)

    pf = make_subplots(rows=n, cols=1, shared_xaxes=False,
                       subplot_titles=[f"f = {f:.2f} Hz" for f, *_ in runs])
    for i, (f, t, ref, meas) in enumerate(runs, start=1):
        pf.add_trace(go.Scatter(x=t, y=ref, name="reference", line=dict(color="blue"),
                                showlegend=(i == 1)), row=i, col=1)
        pf.add_trace(go.Scatter(x=t, y=meas, name="POI", line=dict(color="red"),
                                showlegend=(i == 1)), row=i, col=1)
    pf.update_layout(title=title, template="plotly_white", height=220 * n)
    pf.write_html(out_base + ".html", include_plotlyjs="cdn")
