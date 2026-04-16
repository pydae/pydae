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
import matplotlib.pyplot as plt  # noqa: E402

# Force the Agg backend's native .pyd to load now, into a clean heap
plt.figure()
plt.close("all")

from pydae.bps import BpsBuilder  # noqa: E402
from pydae.core import Builder, Model  # noqa: E402

# Make all relative paths (./build, xy_0.json, output SVG, etc.) resolve
# inside this examples/ folder, regardless of where the script is invoked
# from (repo root, examples/, an IDE, etc.).
os.chdir(Path(__file__).parent)

MODEL_NAME = "temp_m2"
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
    # Hold steady for 1 s
    model.run(0.1, {})

    print("Simulation ended...")

    model.report_u()
    model.report_y()

    # Step change: p_m_1 -> 1.0 pu, raise damping to D_1 = 20
    model.run(1.0, {"p_m_1": 1.0, "D_1": 20.0})

    model.post()

    print("Post done")

    return model


def plot(model) -> None:
    """Plot omega_1 vs time and save to SVG."""

    for it, item in enumerate(model.Time):
        print(it, item, model.get_values("omega_1")[it])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(model.Time, model.get_values("omega_1"), label=r"$\omega_1$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\omega_1$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('milano2ord_pm.svg')
    plt.close(fig)


if __name__ == "__main__":
    print("Building Milano 2nd-order machine model (compiles C code)...")
    build()

    print("Simulating step response...")
    model = simulate()

    out_path = Path(__file__).parent / "milano2ord_pm.svg"
    print(f"Saving plot to: {out_path}")
    plot(model)
    #print(f"Done.