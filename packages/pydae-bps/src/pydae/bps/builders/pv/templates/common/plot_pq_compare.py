#!/usr/bin/env python3
"""Comparative P-Q step response across SCADA protocols.

Overlays the POI active/reactive power response of every protocol stack
(local, modbus, iec104, dnp3, dds) for one of the structured step tests, so the
plant + transport behaviour can be compared wire layer against wire layer.

Each protocol carries its own
``<proto>/tests/<test>/csv/<test>.csv`` (written by that folder's ``run.py``).
This script reads them all and produces two stacked panels — POI P (MW) and
POI Q (Mvar) — with the commanded reference drawn once as a dashed black line.

    python3 common/plot_pq_compare.py                          # open-loop (default)
    python3 common/plot_pq_compare.py --test closed_loop_step  # closed-loop
    python3 common/plot_pq_compare.py --exclude dds            # omit a protocol

Outputs (in ``common/plots/``):
    <open_loop|closed_loop>_pq_compare.svg   — static (matplotlib)
    <open_loop|closed_loop>_pq_compare.html  — interactive (plotly)
"""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- layout ----------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # copv_3_5_wecc/
OUT_DIR = os.path.join(HERE, "plots")

# Protocols in display order: (folder, colour, legend label).
PROTOCOLS = [
    ("local",  "tab:gray",   "local"),            # direct pydae, no wire stack
    ("modbus", "tab:blue",   "modbus"),
    ("iec104", "tab:orange", "iec104"),
    ("dnp3",   "tab:green",  "dnp3"),
    ("dds",    "tab:purple", "dds"),
]
# Plotly needs CSS colours; map the matplotlib tab: names.
PLOTLY_COLOR = {
    "tab:gray": "gray", "tab:blue": "royalblue", "tab:orange": "darkorange",
    "tab:green": "seagreen", "tab:purple": "mediumpurple",
}

# Ground-truth POI flow columns (W / var -> MW / Mvar).
SIM_P_COL = "sim_p_line_POI_GRID"
SIM_Q_COL = "sim_q_line_POI_GRID"

TESTS = {
    "open_loop_step":   ("Open-loop",   "open_loop_pq_compare"),
    "closed_loop_step": ("Closed-loop", "closed_loop_pq_compare"),
}


def load_csv(path: str) -> dict:
    """Load a recorder CSV -> arrays (MW / Mvar)."""
    rows = list(csv.DictReader(open(path)))

    def col(name, scale=1.0):
        return np.array([float(r.get(name) or "nan") * scale for r in rows])

    return {
        "t":     col("t_elapsed"),
        "tgt_p": col("target_mw"),   "tgt_q": col("target_mvar"),
        "sim_p": col(SIM_P_COL, 1e-6), "sim_q": col(SIM_Q_COL, 1e-6),
    }


def main() -> None:
    ap_ = argparse.ArgumentParser(description=__doc__)
    ap_.add_argument("--test", default="open_loop_step", choices=list(TESTS),
                     help="which step test to compare (default: open_loop_step)")
    ap_.add_argument("--exclude", default="", metavar="P1,P2",
                     help="comma-separated protocol folders to omit (e.g. dds)")
    args = ap_.parse_args()
    excl = {x.strip() for x in args.exclude.split(",") if x.strip()}

    label_kind, out_base = TESTS[args.test]
    csv_rel = os.path.join("tests", args.test, "csv", f"{args.test}.csv")

    data = []
    for name, color, label in PROTOCOLS:
        if name in excl:
            print(f"  exclude {label} (requested)")
            continue
        path = os.path.join(ROOT, name, csv_rel)
        if not os.path.isfile(path):
            print(f"  skip {label}: no CSV at {path}")
            continue
        data.append((label, color, load_csv(path)))
        print(f"  loaded {label}: {path}")

    if not data:
        raise SystemExit(f"No {args.test} CSVs found — run the tests first.")

    os.makedirs(OUT_DIR, exist_ok=True)
    title = f"{label_kind} P-Q step — protocol comparison (POI ground truth)"

    # Reference: take it from the first dataset (identical schedule everywhere).
    _, _, ref = data[0]

    # ---- matplotlib SVG ----------------------------------------------------
    fig, (ap, aq) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ap.plot(ref["t"], ref["tgt_p"], "k--", lw=1.2, label="reference")
    aq.plot(ref["t"], ref["tgt_q"], "k--", lw=1.2, label="reference")
    for name, color, d in data:
        ap.plot(d["t"], d["sim_p"], lw=1.6, color=color, alpha=0.9, label=name)
        aq.plot(d["t"], d["sim_q"], lw=1.6, color=color, alpha=0.9, label=name)
    ap.set_ylabel("POI P (MW)"); ap.grid(alpha=0.3); ap.legend(fontsize=8, ncol=2)
    ap.set_title(title)
    aq.set_ylabel("POI Q (Mvar)"); aq.set_xlabel("t (s)"); aq.grid(alpha=0.3)
    aq.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    svg = os.path.join(OUT_DIR, f"{out_base}.svg")
    fig.savefig(svg); plt.close(fig)

    # ---- plotly HTML -------------------------------------------------------
    pf = make_subplots(rows=2, cols=1, shared_xaxes=True,
                       subplot_titles=("POI active power (MW)",
                                       "POI reactive power (Mvar)"))
    pf.add_trace(go.Scatter(x=ref["t"], y=ref["tgt_p"], name="reference",
                            line=dict(color="black", dash="dash")), row=1, col=1)
    pf.add_trace(go.Scatter(x=ref["t"], y=ref["tgt_q"], name="reference",
                            line=dict(color="black", dash="dash"),
                            showlegend=False), row=2, col=1)
    for name, color, d in data:
        c = PLOTLY_COLOR.get(color, color)
        pf.add_trace(go.Scatter(x=d["t"], y=d["sim_p"], name=name,
                                legendgroup=name, line=dict(color=c)),
                     row=1, col=1)
        pf.add_trace(go.Scatter(x=d["t"], y=d["sim_q"], name=name,
                                legendgroup=name, showlegend=False,
                                line=dict(color=c)), row=2, col=1)
    pf.update_layout(title=title, template="plotly_white", height=800)
    pf.update_xaxes(title_text="t (s)", row=2, col=1)
    html = os.path.join(OUT_DIR, f"{out_base}.html")
    pf.write_html(html, include_plotlyjs="cdn")

    print(f"\nwrote {svg}\n      {html}")


if __name__ == "__main__":
    main()
