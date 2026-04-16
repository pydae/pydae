"""
Milano 4th-order synchronous machine — saturation step response
================================================================

Builds a 4th-order Milano synchronous-machine model with the
``pydae.bps`` (Balanced Power Systems) builder, runs it to steady
state, then applies a step in mechanical power and voltage reference
and plots the resulting speed, q-axis transient voltage, and
saturation factor to an SVG file.

Adapted from ``tests/...`` to use only the public package API:
    from pydae.bps  import BpsBuilder
    from pydae.core import Builder, Model
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt

from pydae.bps import BpsBuilder
from pydae.core import Builder, Model

# Make all relative paths resolve inside this examples/ folder regardless
# of the caller's working directory.
os.chdir(Path(__file__).parent)

MODEL_NAME = "temp_m4"
HJSON_PATH = Path(__file__).parent / "milano4ord.hjson"


def build():
    """Construct the system from milano4ord.hjson and compile it."""
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

    # Initialization with steady-state mechanical power and voltage reference
    model.ini({"p_m_1": 0.5, "v_ref_1": 1.5}, "xy_0.json")

    # Run 1 s at the initial operating point
    model.run(1.0, {})

    # Step: increase mechanical power, hold voltage reference
    model.run(10.0, {"p_m_1": 1.0, "v_ref_1": 2})

    model.post()
    return model


def plot(model, out_path: Path) -> None:
    """Plot omega, e'_q and saturation factor S_at and save to SVG."""
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(model.Time, model.get_values("omega_1"),
                 label=r"$\omega$ (pu)", color="b")
    axes[0].set_ylabel("Speed")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(model.Time, model.get_values("e1q_1"),
                 label=r"$e'_q$ (pu)", color="r")
    axes[1].set_ylabel("q-axis transient V")
    axes[1].legend()
    axes[1].grid(True)

    # axes[2].plot(model.Time, model.get_values("S_at_1"),
    #              label=r"$S_{at}$ (pu)", color="purple")
    # axes[2].set_xlabel("Time (s)")
    # axes[2].set_ylabel("Saturation factor")
    # axes[2].legend()
    # axes[2].grid(True)

    fig.tight_layout()
    fig.savefig("milano4ord_saturation.svg")
    plt.close(fig)


if __name__ == "__main__":
    print("Building Milano 4th-order machine model (compiles C code)...")
    build()

    print("Simulating step response...")
    model = simulate()

    out_path = Path(__file__).parent / "milano4ord_saturation.svg"
    plot(model, out_path)
    print(f"Done. Plot saved to: {out_path}")