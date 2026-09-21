"""
Batched parameter sweeps on the MLX (Apple Silicon) backend
===========================================================

Two studies that would be slow one case at a time, run as a single batched
solve on the GPU:

1. **Capability curve** — reactive power at fixed excitation for 10,000
   different mechanical powers. This is a *steady-state* sweep: 10,000
   independent initializations, solved together in about a second.

2. **Inertia study** — 200 transients in lockstep, one per value of the
   machine inertia `H`, after a step in mechanical power.

The model is the `genrou` single-machine-infinite-bus fixture shipped with
`pydae-bps` (IEEE 1110 model 2.2 round-rotor machine, 6th order).

Requires an Apple Silicon machine with `mlx` installed::

    pip install pydae[mlx]

Run it with::

    python examples/mlx_sweep.py
"""

import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pydae.bps import BpsBuilder

# Keep every build artefact (xy_0.json, the SVGs) next to this script.
os.chdir(Path(__file__).parent)

GENROU = Path(__file__).resolve().parents[1] / (
    "packages/pydae-bps/src/pydae/bps/syns/genrou.hjson")


def build():
    """Build the model on the MLX backend and solve the nominal operating point.

    The nominal solve matters twice over: it is the sanity check that the model
    is well posed, and it is the seed every case in a sweep starts from.
    """
    # build_mlx() = construct() + MlxBuilder(...).build(). float32 keeps the
    # whole solve on the Metal GPU, which is what makes batching pay;
    # dtype='float64' also works but runs on the CPU stream.
    model = BpsBuilder(str(GENROU)).build_mlx("genrou_mlx", dtype="float32", Dt=0.01)
    model.ini({})                      # seeded from sys_dict['xy_0_dict']
    return model


def capability_curve(model, n_cases=10_000):
    """Q(P) at fixed excitation, for `n_cases` dispatch points at once."""
    print(f"\n--- capability curve: {n_cases:,} initializations ---")

    start = time.time()
    sweep = model.sweep({"p_m_1": np.linspace(0.05, 0.95, n_cases)})
    sweep.ini()
    elapsed = time.time() - start

    print(f"solved in {elapsed:.2f} s "
          f"({1e3 * elapsed / n_cases:.3f} ms per case)")

    # Cases past the machine's steady-state limit have no solution at all.
    # They are reported, not raised, and read back as NaN.
    lo, hi = sweep.feasible_range("p_m_1")
    analytic = 1.5 / 2.135                      # E*V/X_d, the pull-out limit
    print(f"feasible p_m : {lo:.4f} .. {hi:.4f} pu")
    print(f"analytic limit E*V/X_d = {analytic:.4f} pu")

    print("\n" + sweep.to_frame(["q_g_1", "V_1"]).iloc[::n_cases // 6]
          .to_string(index=False))
    return sweep


def inertia_study(model, n_cases=200):
    """`n_cases` transients in lockstep: the same power step at different H."""
    print(f"\n--- inertia study: {n_cases} transients in lockstep ---")

    sweep = model.sweep({"H_1": np.linspace(3.0, 9.0, n_cases)})
    sweep.ini()

    start = time.time()
    sweep.run(1.0, record=["omega_1"])           # settle at the initial point
    sweep.run(6.0, {"p_m_1": 0.65})              # step the mechanical power
    elapsed = time.time() - start

    omega = sweep.get_series("omega_1")          # (n_samples, n_cases)
    print(f"integrated {len(sweep.Time)} steps x {n_cases} cases "
          f"in {elapsed:.2f} s")
    print(f"peak speed: H=3 -> {omega[:, 0].max():.5f} pu, "
          f"H=9 -> {omega[:, -1].max():.5f} pu")
    return sweep, omega


def eigenvalue_locus(model, n_cases=60):
    """Track the slowest oscillatory mode as the operating point moves.

    `sweep.eigenvalues()` linearises every case, so a damping locus costs one
    batched solve rather than `n_cases` sequential ones.
    """
    print(f"\n--- eigenvalue locus over {n_cases} operating points ---")

    sweep = model.sweep({"p_m_1": np.linspace(0.10, 0.65, n_cases)})
    sweep.ini()
    eigs = sweep.eigenvalues()                   # (n_cases, N_x)

    # Pick the complex pair with the lowest frequency. Masking on the imaginary
    # part itself (not on a complex NaN, whose imaginary part is 0) keeps the
    # purely real poles out of the selection.
    imag = np.abs(eigs.imag)
    idx = np.argmin(np.where(imag > 1e-9, imag, np.inf), axis=1)
    mode = eigs[np.arange(len(idx)), idx]
    damping = -mode.real / np.abs(mode)
    freq = np.abs(mode.imag) / (2 * np.pi)

    print(f"frequency: {np.nanmin(freq) * 1e3:.2f} .. {np.nanmax(freq) * 1e3:.2f} mHz")
    print(f"damping  : {np.nanmin(damping):.4f} .. {np.nanmax(damping):.4f}")
    return sweep.cases["p_m_1"], freq, damping


def plot(sweep, omega_sweep, omega, locus):
    """One figure per study, saved as SVG next to this script."""
    p_m = sweep.cases["p_m_1"]
    q_g = sweep.get_values("q_g_1")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(p_m, q_g, color="tab:blue", linewidth=1.4)
    limit = sweep.feasible_range("p_m_1")[1]
    ax.axvspan(limit, p_m.max(), color="tab:red", alpha=0.12)
    ax.text(limit + 0.01, np.nanmax(q_g) * 0.8, "no steady state\n(pull-out limit)",
            color="tab:red", fontsize=9, va="top")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel(r"$p_m$ [pu]")
    ax.set_ylabel(r"$q_g$ [pu]")
    ax.set_title(f"Reactive power at fixed excitation — {len(p_m):,} cases")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("mlx_sweep_capability.svg")

    fig, ax = plt.subplots(figsize=(8, 4))
    inertias = omega_sweep.cases["H_1"]
    for j in range(0, len(inertias), max(1, len(inertias) // 8)):
        ax.plot(omega_sweep.Time, omega[:, j], linewidth=1.0,
                label=f"H = {inertias[j]:.1f} s")
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"$\omega$ [pu]")
    ax.set_title(f"Speed after a power step — {len(inertias)} inertias in lockstep")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("mlx_sweep_inertia.svg")

    p_locus, freq, damping = locus
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    ax1.plot(p_locus, freq * 1e3, color="tab:green", linewidth=1.4)
    ax1.set_ylabel("frequency [mHz]")
    ax1.grid(True, alpha=0.3)
    ax2.plot(p_locus, damping, color="tab:purple", linewidth=1.4)
    ax2.set_ylabel("damping ratio")
    ax2.set_xlabel(r"$p_m$ [pu]")
    ax2.grid(True, alpha=0.3)
    ax1.set_title("Slowest oscillatory mode vs operating point")
    fig.tight_layout()
    fig.savefig("mlx_sweep_locus.svg")

    print("\nplots saved: mlx_sweep_capability.svg, mlx_sweep_inertia.svg, "
          "mlx_sweep_locus.svg")


if __name__ == "__main__":
    print("Building the genrou model on the MLX backend...")
    model = build()

    sweep = capability_curve(model)
    omega_sweep, omega = inertia_study(model)
    locus = eigenvalue_locus(model)
    plot(sweep, omega_sweep, omega, locus)
