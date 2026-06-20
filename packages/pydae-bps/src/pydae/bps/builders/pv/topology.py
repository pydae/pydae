"""Draw the electrical + comms topology of a generated plant JSON.

Reads a plant JSON produced by ``pv_designer.designer`` and renders
two stacked subplots:

- Top:    electrical (buses, lines, transformers, sources, WECCs)
- Bottom: comms (hosts, switches, links — positions taken from the JSON
  ``x`` / ``y`` fields written by the builder)

Usage::

    python -m pydae.bps.builders.topology --json plant.json
    python -m pydae.bps.builders.topology --json plant.json --out diagram.svg
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle


_INV_RE = re.compile(r"^SSf(\d+)n(\d+)$")
_LV_RE  = re.compile(r"^SSf(\d+)n(\d+)LV$")
_FDR_RE = re.compile(r"^FDR(\d+)$")


# ---------- helpers ----------

def _infer_MN(plant: dict[str, Any]) -> tuple[int, int]:
    M = N = 0
    for d in plant.get("devices", []):
        m = _INV_RE.match(d.get("emec_id", ""))
        if m:
            M = max(M, int(m.group(1)))
            N = max(N, int(m.group(2)))
    return M, N


_EL_Y_FDR = 3.0   # vertical pitch between feeders (MV-row to MV-row)


def _bus_pos(name: str, M: int, N: int) -> tuple[float, float]:
    """Algorithmic position for an electrical bus.

    Coordinate space (Y-up, X-right). New topology:
        GRID (HV) - POI (HV) - POIMV (MV) - per-feeder daisy - inverters
    """
    x_poimv = N * 2.0 + 2.0
    x_poi   = x_poimv + 2.5
    x_grid  = x_poi   + 2.5
    y_top   = (M - 1) * _EL_Y_FDR
    y_mid   = y_top / 2.0 + 0.5    # vertical center of feeder rows
    if name == "GRID":
        return (x_grid, y_mid)
    if name == "POI":
        return (x_poi, y_mid)
    if name == "POIMV":
        return (x_poimv, y_mid)
    if m := _INV_RE.match(name):  # MV side
        F, n = int(m.group(1)), int(m.group(2))
        return ((n - 1) * 2.0, (M - F) * _EL_Y_FDR + 0.5)
    if m := _LV_RE.match(name):
        F, n = int(m.group(1)), int(m.group(2))
        return ((n - 1) * 2.0, (M - F) * _EL_Y_FDR - 1.0)
    return (0.0, 0.0)


def _midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


# ---------- electrical ----------

def _draw_electrical(ax: plt.Axes, plant: dict[str, Any]) -> None:
    M, N = _infer_MN(plant)
    if M == 0:
        ax.text(0.5, 0.5, "no inverters found in 'devices'",
                transform=ax.transAxes, ha="center", va="center")
        return

    bus_pos = {b["name"]: _bus_pos(b["name"], M, N) for b in plant["buses"]}

    # lines
    for ln in plant.get("lines", []):
        a = bus_pos.get(ln["bus_j"])
        b = bus_pos.get(ln["bus_k"])
        if a is None or b is None:
            continue
        ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=1.4, zorder=1)

    # transformers: short bar between bus_j and bus_k, two stacked circles
    for tx in plant.get("transformers", []):
        a = bus_pos.get(tx["bus_j"])
        b = bus_pos.get(tx["bus_k"])
        if a is None or b is None:
            continue
        ax.plot([a[0], b[0]], [a[1], b[1]], color="black", lw=1.4,
                linestyle="--", zorder=1)
        mx, my = _midpoint(a, b)
        ax.add_patch(Circle((mx, my + 0.10), 0.18, fill=False,
                            ec="black", lw=1.2, zorder=2))
        ax.add_patch(Circle((mx, my - 0.10), 0.18, fill=False,
                            ec="black", lw=1.2, zorder=2))

    # buses
    for b in plant["buses"]:
        x, y = bus_pos[b["name"]]
        is_lv = "LV" in b["name"]
        color = "#cfe8ff" if is_lv else "#ffd9a8"
        ax.add_patch(Rectangle((x - 0.55, y - 0.18), 1.10, 0.36,
                               facecolor=color, edgecolor="black",
                               linewidth=1.0, zorder=3))
        ax.text(x, y, b["name"], ha="center", va="center", fontsize=7,
                zorder=4)

    # WECCs (PV inverters): triangle next to LV bus
    for w in plant.get("weccs", []):
        bus = w.get("bus")
        a = bus_pos.get(bus)
        if a is None:
            continue
        x, y = a[0], a[1] - 1.1
        tri = Polygon([(x - 0.3, y - 0.25), (x + 0.3, y - 0.25), (x, y + 0.30)],
                      facecolor="#f8e16c", edgecolor="black", lw=1.0, zorder=3)
        ax.add_patch(tri)
        ax.plot([a[0], x], [a[1] - 0.18, y - 0.30], color="black", lw=1.0,
                zorder=2)
        ax.text(x, y - 0.55, "PV", ha="center", va="top", fontsize=6,
                zorder=4)

    # Sources (genape). If the source bus is already labeled "GRID"
    # don't draw a separate circle — the bus rectangle is enough.
    for s in plant.get("sources", []):
        bus = s.get("bus")
        a = bus_pos.get(bus)
        if a is None or bus == "GRID":
            continue
        gx, gy = a[0], a[1] + 1.0
        ax.add_patch(Circle((gx, gy), 0.32, facecolor="#ffd6d6",
                            edgecolor="black", lw=1.0, zorder=3))
        ax.text(gx, gy, "GRID", ha="center", va="center", fontsize=6, zorder=4)
        ax.plot([a[0], gx], [a[1] + 0.18, gy - 0.32], color="black", lw=1.0,
                zorder=2)

    ax.set_title(f"Electrical  —  {M} feeder(s) x {N} inverter(s)  "
                 f"(M*N = {M * N})", fontsize=10)
    ax.set_aspect("equal")
    ax.set_axis_off()
    _autoscale(ax, bus_pos.values(), margin=2.0)


# ---------- comms ----------

_HOST_COLOR = {
    "PPC":   "#ffd6d6",
    "POI":   "#cfe8ff",
    "SCADA": "#d6f5d6",
    "UI":    "#e8d6ff",
    "FDII":  "#e8d6ff",
    "MITM":  "#ff9b9b",
    "CPN":   "#f7d68c",
    "BESS":  "#cfe8ff",
}


def _comm_pos(plant: dict[str, Any]) -> dict[str, tuple[float, float]]:
    pos: dict[str, tuple[float, float]] = {}
    for h in plant.get("hosts", []):
        if "x" in h and "y" in h:
            pos[h["name"]] = (float(h["x"]), float(h["y"]))
    for s in plant.get("switches", []):
        if "x" in s and "y" in s:
            pos[s["name"]] = (float(s["x"]), float(s["y"]))
    return pos


def _draw_comms(ax: plt.Axes, plant: dict[str, Any]) -> None:
    pos = _comm_pos(plant)
    if not pos:
        ax.text(0.5, 0.5, "no x/y on hosts/switches — regenerate JSON",
                transform=ax.transAxes, ha="center", va="center")
        return

    # links first (so nodes draw on top)
    for ln in plant.get("links", []):
        a = pos.get(ln["node_j"])
        b = pos.get(ln["node_k"])
        if a is None or b is None:
            continue
        is_trunk = ln["node_j"].startswith("s") and ln["node_k"].startswith("s")
        impaired = "delay" in ln or "loss" in ln
        ax.plot([a[0], b[0]], [a[1], b[1]],
                color=("#c44" if impaired else "#888"),
                lw=(1.6 if is_trunk else 0.9),
                linestyle=("-" if is_trunk else "--"),
                zorder=1, alpha=0.85)

    # switches: rounded squares
    for s in plant.get("switches", []):
        if s["name"] not in pos:
            continue
        x, y = pos[s["name"]]
        ax.add_patch(FancyBboxPatch(
            (x - 0.36, y - 0.22), 0.72, 0.44,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor="#e6e6e6", edgecolor="black", lw=1.1, zorder=3,
        ))
        ax.text(x, y, s["name"], ha="center", va="center",
                fontsize=6, zorder=4)

    # hosts: circles
    for h in plant.get("hosts", []):
        if h["name"] not in pos:
            continue
        x, y = pos[h["name"]]
        color = _HOST_COLOR.get(h["name"], "#cfe8ff")  # default = inverter
        ax.add_patch(Circle((x, y), 0.32, facecolor=color,
                            edgecolor="black", lw=1.1, zorder=3))
        ax.text(x, y, h["name"], ha="center", va="center",
                fontsize=5.5, zorder=4)

    # legend
    legend_handles = [
        Line2D([0], [0], color="#c44", lw=1.6, label="impaired trunk"),
        Line2D([0], [0], color="#888", lw=1.6, label="trunk"),
        Line2D([0], [0], color="#888", lw=0.9, linestyle="--",
               label="host link"),
    ]
    ax.legend(handles=legend_handles, loc="lower right",
              fontsize=7, framealpha=0.9)

    ax.set_title("Comms (switches in series along the feeder)", fontsize=10)
    ax.set_aspect("equal")
    ax.set_axis_off()
    _autoscale(ax, pos.values(), margin=1.5)


# ---------- public API ----------

def _autoscale(ax: plt.Axes, points, *, margin: float) -> None:
    pts = list(points)
    if not pts:
        return
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)


def draw_topology(
    plant_or_path: dict[str, Any] | str,
    out_path: str | None = None,
    *,
    show: bool = False,
) -> plt.Figure:
    """Render the electrical + comms topology of a plant.

    ``plant_or_path`` accepts a parsed dict or a JSON file path.
    Returns the matplotlib Figure (caller may close it).
    """
    if isinstance(plant_or_path, str):
        with open(plant_or_path, encoding="utf-8") as f:
            plant = json.load(f)
        default_title = os.path.basename(plant_or_path)
    else:
        plant = plant_or_path
        default_title = plant.get("system", {}).get("name", "plant")

    # Size figure proportionally to the plant so labels stay legible.
    M, N = _infer_MN(plant)
    width  = max(10.0, 2.4 * N + 7.0)
    height = max(8.0,  2.0 * M + 6.0)
    fig, (ax_e, ax_c) = plt.subplots(2, 1, figsize=(width, height),
                                     gridspec_kw={"height_ratios": [1.0, 1.0]})
    fig.suptitle(default_title, fontsize=12, y=0.995)

    _draw_electrical(ax_e, plant)
    _draw_comms(ax_c, plant)
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    if out_path:
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    return fig


def main() -> None:
    p = argparse.ArgumentParser(
        prog="python -m pydae.bps.builders.topology",
        description="Render the electrical + comms topology of a plant JSON.",
    )
    p.add_argument("--json", required=True, help="Input plant JSON path")
    p.add_argument("--out", default=None,
                   help="Output image path (svg/pdf/png). Default: "
                        "<json basename>.svg next to the JSON.")
    p.add_argument("--show", action="store_true",
                   help="Display the figure interactively in addition to "
                        "writing it.")
    args = p.parse_args()

    out = args.out
    if out is None:
        base = os.path.splitext(args.json)[0]
        out = base + ".svg"

    draw_topology(args.json, out, show=args.show)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
