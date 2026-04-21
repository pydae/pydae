"""
Milano 4th-order machine — small-signal analysis example
=========================================================

Companion to ``tests/bps/syns/test_syns.py::test_milano4ord_ssa``.
Walks through the SSA workflow in three separate functions so each
stage can be read in isolation:

    1. build()        -- compile the DAE system to C.
    2. initialize()   -- Newton-Raphson solve of the operating point
                         (no time stepping).
    3. analyze()      -- linearize at the operating point and produce
                         the damping / eigenvalue report.

Run with:

    uv run python examples/milano4ord_ssa_example.py
"""

import os
from pathlib import Path

import numpy as np

# Pre-import tabulate + wcwidth before any CFFI extension is loaded.
# On Windows, cffi can corrupt Python's heap; a later lazy import of
# tabulate via pandas.DataFrame.to_markdown() then segfaults inside
# wcwidth -> importlib.metadata. Loading them up front sidesteps that.
import tabulate  # noqa: F401
import wcwidth   # noqa: F401

from pydae.bps import BpsBuilder
from pydae.core import Builder, Model
from pydae.ssa import A_eval, damp_report


# Resolve everything relative to this file's directory so the example
# can be launched from any working directory.
HERE = Path(__file__).parent
os.chdir(HERE)

MODEL_NAME = "milano4ord_ssa_ex"
HJSON_PATH = HERE / "milano4ord.hjson"


# -----------------------------------------------------------------------------
# 1) Build stage: symbolic -> C -> shared library
# -----------------------------------------------------------------------------
def build():
    """Parse the HJSON network, assemble the DAE system, and compile it.

    This is the expensive step (SymPy + C compiler). Re-run only when
    the network description or a component model changes.
    """
    grid = BpsBuilder(str(HJSON_PATH))
    grid.checker()
    grid.uz_jacs = False
    grid.construct(MODEL_NAME)

    bld = Builder(grid.sys_dict, target="cffi", sparse=False)
    bld.build()
    return bld


# -----------------------------------------------------------------------------
# 2) Initialization stage: Newton-Raphson to steady state
# -----------------------------------------------------------------------------
def initialize():
    """Load the compiled model and solve the ini problem (no time stepping).

    `xy_0=1` seeds every dynamic and algebraic variable at 1.0, which is
    a reasonable starting guess for a machine near nominal voltage and
    speed. The operating-point inputs are mechanical power and (because
    the hjson attaches an AVR) the voltage reference instead of v_f.
    """
    model = Model(MODEL_NAME)

    inputs = {"p_m_1": 0.5, "v_ref_1": 1.0}
    success = model.ini(inputs, xy_0=1)
    assert success, "ini did not converge"

    # Quick sanity check on the operating point.
    print(f"omega_1 = {model.get_value('omega_1'):.6f} pu")
    print(f"V_1     = {model.get_value('V_1'):.6f} pu")
    print(f"p_m_1   = {model.get_value('p_m_1'):.6f} pu")
    return model


# -----------------------------------------------------------------------------
# 3) SSA stage: linearize + eigen-decompose
# -----------------------------------------------------------------------------
def analyze(model):
    """Build the reduced state matrix A and print a damping report.

    No time integration is needed: SSA uses only the Jacobian blocks
    captured at the ini operating point (Fx, Fy, Gx, Gy). The reduced
    A = Fx - Fy * Gy^{-1} * Gx is the Schur complement.
    """
    # A_eval stores the result on the model as `model.A` (needed by
    # damp_report, which reads it for the eigen-decomposition).
    A = A_eval(model)
    print(f"\nA shape: {A.shape}")
    print(f"max |A|: {np.max(np.abs(A)):.4g}")

    # damp_report returns a DataFrame with Real / Imag / Freq. / Damp /
    # Participation per mode, and also writes a markdown file to the
    # current working directory. The to_markdown() call can trigger a
    # known Windows heap-corruption crash via tabulate's lazy import;
    # we print the DataFrame before writing it out to guarantee the
    # modes are shown even if the crash fires.
    df = damp_report(model, sparse=False, tol_part=0.2)
    print("\nModes (sorted by damping):")
    print(df.sort_values("Damp"))
    return df


if __name__ == "__main__":
    print("[1/3] Building Milano 4th-order machine...")
    build()

    print("\n[2/3] Initializing to steady state...")
    model = initialize()

    print("\n[3/3] Running small-signal analysis...")
    analyze(model)

    print("\nDone.")
