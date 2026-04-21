"""
Milano 2nd-order synchronous machine — mechanical-power step response
======================================================================

Builds a 2nd-order Milano synchronous machine with the ``pydae.bps``
Balanced Power Systems builder, runs to steady state, then applies a
step in mechanical power ``p_m_1`` together with a large damping
coefficient ``D_1``. The rotor speed ``omega_1`` is plotted to SVG.

Adapted from ``tests/...`` to use the public package API:
    from pydae.bps  import BpsBuilder
    from pydae.core import Builder, Model
"""

import os
from pathlib import Path

# Force matplotlib's native Agg extension to load BEFORE any pydae
# compiled model DLL. On Windows this avoids a CRT-heap collision
# (STATUS_HEAP_CORRUPTION 0xC0000374) that silently kills the process.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# plt.figure(); plt.close("all")

import numpy as np
import sympy as sym

from pydae.core import Builder, Model
from pydae.bps import BpsBuilder

# Make all relative paths resolve inside this examples/ folder regardless
# of the caller's working directory.
os.chdir(Path(__file__).parent)


MODEL_NAME = "milano2ord"
HJSON_PATH = Path(__file__).parent / "milano2ord.hjson"


def build():
    """Construct the system from milano2ord.hjson and compile it."""
    grid = BpsBuilder(str(HJSON_PATH))
    grid.checker()
    grid.uz_jacs = False
    grid.construct(MODEL_NAME)

    bld = Builder(grid.sys_dict, target="ctypes", sparse=False)
    bld.build()
    return bld


def simulate():
    """Initialize, step, and post-process the compiled model."""
    model = Model(MODEL_NAME)

    # Initialize at p_m_1 = 0.5 pu
    model.ini({'p_m_1':0.5,'e1q_1':1.5}, "xy_0.json")

    model.report_u()
    model.report_y()

    print("Simulating step response...")


    model.report_u()
    model.report_y()

    # Step change: p_m_1 -> 1.0 pu, raise damping to D_1 = 20
    model.run(1.0, {})
    model.run(10.0, {"p_m_1": 1.0, "D_1":20.0})    
    model.post()

    print("Simulation ended...")

    print("Post done")

    return model


if __name__ == "__main__":
    print("Building Milano 2nd-order machine model (compiles C code)...")
    build()

    print("Simulating step response...")
    model = simulate()

    try:
        print(model.Time)
        print(model.get_values('omega_1'))
        # Plot omega_1 and save as SVG next to this script
        out_path = Path(__file__).parent / "milano2ord.svg"
        print(f"  plot to be saved at: {out_path}")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(model.Time, model.get_values('omega_1'), color="tab:blue", linewidth=1.2)
        ax.set_xlabel("time [s]")
        ax.set_ylabel(r"$\theta$ [deg]")
        ax.set_title("Pendulum — angle vs time")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()


        fig.savefig(out_path)
        # plt.close(fig)
        
    except AttributeError:
        print("  (Model has finished — inspect attributes interactively)")