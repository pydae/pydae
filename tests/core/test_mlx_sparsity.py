"""Structural sparsity of the MLX Jacobians (Apple Silicon only).

The compressed Jacobian has one job: produce exactly what the dense one
produces, from far fewer AD passes.  So these tests are equivalence tests
against ``_jacobian``, plus checks that the compression is actually engaged --
a pattern that silently degenerated to one colour per column would still be
*correct*, and would quietly give back all the speed.

Run with:
    uv run pytest tests/core/test_mlx_sparsity.py -v
"""

import os
import platform
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core", reason="mlx is an Apple Silicon extra")
pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(platform.machine() != "arm64", reason="MLX needs Apple Silicon"),
]

GENROU_HJSON = "packages/pydae-bps/src/pydae/bps/syns/genrou.hjson"
K12P6_HJSON = "packages/pydae-bps/src/pydae/bps/miscellaneous/k12p6.hjson"


# ---------------------------------------------------------------------------
# a synthetic multi-machine model with a genuine dense row
# ---------------------------------------------------------------------------
def coi_sys_dict(N_machines):
    r"""``N_machines`` classical machines coupled through a centre-of-inertia row.

    Built here rather than taken from ``pydae-bps`` because it is the one
    structure that breaks plain column colouring: ``omega_coi`` contains every
    machine speed, so every pair of columns shares that row and the colouring
    degenerates to one colour per column unless the row is pulled out and given
    to reverse mode.  Scaling ``N_machines`` scales the row's density.
    """
    import sympy as sp

    f_list, g_list, x_list, y_list = [], [], [], []
    params, u_ini, u_run, xy_0 = {}, {}, {}, {}
    Omega_b = sp.Symbol("Omega_b", real=True)
    omega_coi = sp.Symbol("omega_coi", real=True)
    params["Omega_b"] = 314.159265
    numerator, denominator = 0, 0

    for i in range(1, N_machines + 1):
        delta, omega = sp.symbols(f"delta_{i} omega_{i}", real=True)
        p_e = sp.Symbol(f"p_e_{i}", real=True)
        H, D_damp, X, E, V, Theta, P_m = sp.symbols(
            f"H_{i} D_{i} X_{i} E_{i} V_{i} Theta_{i} P_m_{i}", real=True)
        f_list += [Omega_b * (omega - omega_coi),
                   (P_m - p_e - D_damp * (omega - omega_coi)) / (2 * H)]
        x_list += [delta, omega]
        g_list.append(-p_e + E * V * sp.sin(delta - Theta) / X)
        y_list.append(p_e)
        inertia = 3.0 + 0.1 * i
        params.update({f"H_{i}": inertia, f"D_{i}": 1.0, f"X_{i}": 0.4,
                       f"E_{i}": 1.1, f"V_{i}": 1.0, f"Theta_{i}": 0.0})
        u_ini[f"P_m_{i}"] = 0.5
        u_run[f"P_m_{i}"] = 0.5
        xy_0.update({f"delta_{i}": 0.2, f"omega_{i}": 1.0, f"p_e_{i}": 0.5})
        numerator += inertia * omega
        denominator += inertia

    g_list.append(-omega_coi + numerator / denominator)
    y_list.append(omega_coi)
    xy_0["omega_coi"] = 1.0
    return {"name": f"coi{N_machines}", "f_list": f_list, "g_list": g_list,
            "x_list": x_list, "y_ini_list": list(y_list),
            "y_run_list": list(y_list), "u_ini_dict": u_ini, "u_run_dict": u_run,
            "params_dict": params, "h_dict": {"omega_coi_out": omega_coi},
            "xy_0_dict": xy_0}


def _sys_dict_from(hjson_path, name, tmp_path_factory):
    from pydae.bps import BpsBuilder

    # BpsBuilder.construct() drops xy_0.json in the working directory.
    hjson = str(Path(hjson_path).resolve())
    cwd = os.getcwd()
    os.chdir(tmp_path_factory.mktemp(name))
    try:
        grid = BpsBuilder(hjson)
        grid.construct(name)
    finally:
        os.chdir(cwd)
    return grid.sys_dict


@pytest.fixture(scope="module")
def genrou_sys(tmp_path_factory):
    return _sys_dict_from(GENROU_HJSON, "sparsity_genrou", tmp_path_factory)


@pytest.fixture(scope="module")
def k12p6_sys(tmp_path_factory):
    return _sys_dict_from(K12P6_HJSON, "sparsity_k12p6", tmp_path_factory)


def _ini_point(model):
    """``(primal, rest)`` for the initialization Jacobian at the seed state."""
    return (model._arr(np.concatenate([model.x, model.y_ini])),
            (model._arr(model.p_vals), model._arr(model.u_ini_vals),
             model._arr(model.c_ini_vals)))


def on_stream(model):
    """Every MLX op of a float64 model must be issued on the CPU stream.

    The builder wraps its own entry points with ``_on_stream``; a test poking
    at ``_jacobian`` or ``compressed_jacobian`` directly has to do the same.
    """
    return mx.stream(model.stream)


# ---------------------------------------------------------------------------
# the pattern itself
# ---------------------------------------------------------------------------
def test_pattern_covers_every_numerically_nonzero_entry(genrou_sys):
    """The structural pattern may over-count, but must never miss an entry."""
    from pydae.core.builder import MlxBuilder
    from pydae.core.builder.mlx_builder import _jacobian

    model = MlxBuilder(genrou_sys, dtype="float64", verbose=False).build()
    model.ini({})
    with on_stream(model):
        primal, rest = _ini_point(model)
        dense = np.array(
            _jacobian(model._fg_ini, model.N_f + model.N_g)(primal, *rest))

    pattern = model.patterns["ini"]
    structural = np.zeros((pattern.n_rows, pattern.n_cols), dtype=bool)
    structural[pattern.rows, pattern.cols] = True
    missed = (np.abs(dense) > 1e-12) & ~structural
    assert not missed.any(), f"{missed.sum()} nonzero entries outside the pattern"


def test_pattern_has_no_duplicate_entries(k12p6_sys):
    """Recovery accumulates, so a repeated (row, col) would double that entry.

    The trapezoidal pattern is where this bites: a state appearing in its own
    derivative lands on the same diagonal slot as the step's explicit ``x_n``.
    """
    from pydae.core.builder import MlxBuilder

    model = MlxBuilder(k12p6_sys, dtype="float32", verbose=False).build()
    for label, pattern in model.patterns.items():
        slots = pattern.rows * pattern.n_cols + pattern.cols
        assert len(np.unique(slots)) == len(slots), f"duplicates in '{label}'"


def test_compression_actually_engages(k12p6_sys):
    """A correct pattern that never compresses would pass every other test."""
    from pydae.core.builder import MlxBuilder

    model = MlxBuilder(k12p6_sys, dtype="float32", verbose=False).build()
    pattern = model.patterns["ini"]
    assert pattern.n_rows == 87
    # Measured: 87 columns need 9 colours or fewer.
    assert pattern.n_passes <= 12, f"expected heavy compression, got {pattern!r}"
    assert pattern.worthwhile


def test_dense_row_is_routed_to_reverse_mode():
    """The COI row must be pulled out, or the colouring degenerates."""
    from pydae.core.builder.mlx_sparsity import JacobianPattern

    model_dict = coi_sys_dict(20)
    n_machines = 20
    # One row touching every speed column, against an otherwise banded pattern.
    rows = list(range(n_machines)) + [n_machines] * n_machines
    cols = list(range(n_machines)) + list(range(n_machines))
    pattern = JacobianPattern(rows, cols, n_machines + 1, n_machines)
    assert len(pattern.dense_rows) == 1
    assert pattern.dense_rows[0] == n_machines
    assert pattern.n_passes < n_machines, repr(pattern)
    assert model_dict["name"] == "coi20"


# ---------------------------------------------------------------------------
# equivalence with the dense Jacobian
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("width", [None, 1, 8])
def test_compressed_matches_dense_jacobian(k12p6_sys, width):
    """Same values as plain reverse-mode AD, scalar and batched layouts alike."""
    from pydae.core.builder import MlxBuilder
    from pydae.core.builder.mlx_builder import _jacobian
    from pydae.core.builder.mlx_sparsity import compressed_jacobian

    model = MlxBuilder(k12p6_sys, dtype="float64", verbose=False).build()
    model.ini({})
    with on_stream(model):
        primal, rest = _ini_point(model)
        if width is not None:
            primal = mx.broadcast_to(primal[:, None],
                                     (primal.shape[0], width)) + 0.0
            rest = tuple(
                mx.broadcast_to(a[:, None], (a.shape[0], width)) + 0.0
                if a.shape[0] else a for a in rest)

        dense = np.array(
            _jacobian(model._fg_ini, model.N_f + model.N_g)(primal, *rest))
        sparse = np.array(compressed_jacobian(
            model._fg_ini, model.patterns["ini"], model.dtype,
            model.stream)(primal, *rest))
    assert sparse.shape == dense.shape
    assert np.abs(sparse - dense).max() < 1e-10


def test_compressed_matches_dense_with_a_dense_row():
    """Equivalence on the one structure that needs both AD modes at once."""
    from pydae.core.builder import MlxBuilder
    from pydae.core.builder.mlx_builder import _jacobian
    from pydae.core.builder.mlx_sparsity import compressed_jacobian

    model = MlxBuilder(coi_sys_dict(12), dtype="float64", verbose=False).build()
    model.ini({})
    pattern = model.patterns["ini"]
    assert len(pattern.dense_rows) >= 1, "the COI row should go to reverse mode"

    with on_stream(model):
        primal, rest = _ini_point(model)
        dense = np.array(
            _jacobian(model._fg_ini, model.N_f + model.N_g)(primal, *rest))
        sparse = np.array(compressed_jacobian(
            model._fg_ini, pattern, model.dtype, model.stream)(primal, *rest))
    assert np.abs(sparse - dense).max() < 1e-10


# ---------------------------------------------------------------------------
# equivalence of the solutions the Jacobians produce
# ---------------------------------------------------------------------------
def test_ini_and_run_agree_with_the_dense_path(k12p6_sys):
    """The whole point: same operating point, same trajectory, fewer passes."""
    from pydae.core.builder import MlxBuilder

    results = {}
    for sparse in (False, True):
        model = MlxBuilder(k12p6_sys, dtype="float64", sparse=sparse,
                           verbose=False).build()
        model.ini({})
        model.ini2run()
        model.run(0.5, {"p_m_1": 0.7})
        model.post()
        results[sparse] = (np.array(model.x), np.array(model.y_run))

    for dense_block, sparse_block in zip(results[False], results[True]):
        assert np.abs(dense_block - sparse_block).max() < 1e-8


def test_sweep_agrees_with_the_dense_path(genrou_sys):
    """Batched solves are where compression pays; they must still agree."""
    from pydae.core.builder import MlxBuilder

    values = np.linspace(0.2, 0.6, 32)
    results = {}
    for sparse in (False, True):
        model = MlxBuilder(genrou_sys, dtype="float64", sparse=sparse,
                           verbose=False).build()
        model.ini({})
        sweep = model.sweep({"p_m_1": values})
        sweep.ini()
        results[sparse] = (sweep.get_values("q_g_1"), sweep.converged.copy())

    assert np.array_equal(results[False][1], results[True][1])
    assert results[True][1].all()
    assert np.abs(results[False][0] - results[True][0]).max() < 1e-8


def test_state_matrix_agrees_with_the_dense_path(k12p6_sys):
    """Fx/Fy/Gx/Gy are compressed too, so A and its spectrum must be unchanged."""
    from pydae.core.builder import MlxBuilder

    spectra = {}
    for sparse in (False, True):
        model = MlxBuilder(k12p6_sys, dtype="float64", sparse=sparse,
                           verbose=False).build()
        model.ini({})
        model.ini2run()
        spectra[sparse] = np.sort_complex(np.linalg.eigvals(model.A_eval()))

    assert np.abs(spectra[False] - spectra[True]).max() < 1e-6


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------
def test_sparse_false_uses_plain_reverse_mode(genrou_sys):
    from pydae.core.builder import MlxBuilder

    model = MlxBuilder(genrou_sys, dtype="float64", sparse=False,
                       verbose=False).build()
    assert model._sparse_used == {}
    model.ini({})
    assert np.isfinite(model.x).all()


def test_a_dense_pattern_falls_back_rather_than_compressing():
    """Nothing to gain on a dense system, so the compression must stand aside."""
    from pydae.core.builder.mlx_sparsity import JacobianPattern

    size = 6
    rows = np.repeat(np.arange(size), size)
    cols = np.tile(np.arange(size), size)
    pattern = JacobianPattern(rows, cols, size, size)
    assert pattern.n_passes >= size
    assert not pattern.worthwhile


def test_both_front_ends_give_the_same_pattern(tmp_path_factory):
    """Symbol extraction differs per front end; the pattern must not.

    SymPy is scanned with ``free_symbols`` and CasADi with ``ca.symvar``, so
    this is the only check that the two agree on what a Jacobian's structure is.
    """
    from pydae.bps import BpsBuilder
    from pydae.core.builder import MlxBuilder

    results = {}
    hjson = str(Path(GENROU_HJSON).resolve())
    for use_casadi in (False, True):
        cwd = os.getcwd()
        os.chdir(tmp_path_factory.mktemp(f"frontend_{use_casadi}"))
        try:
            grid = BpsBuilder(hjson, use_casadi=use_casadi)
            grid.construct("genrou_frontend")
        finally:
            os.chdir(cwd)
        model = MlxBuilder(grid.sys_dict, dtype="float64", verbose=False).build()
        assert model.frontend == ("casadi" if use_casadi else "sympy")
        model.ini({})
        pattern = model.patterns["ini"]
        results[use_casadi] = (pattern.nnz, pattern.n_passes, np.array(model.x))

    sympy_side, casadi_side = results[False], results[True]
    assert sympy_side[:2] == casadi_side[:2]
    assert np.abs(sympy_side[2] - casadi_side[2]).max() < 1e-8


def test_verify_pattern_catches_a_corrupted_recovery(genrou_sys):
    """The build-time check has to actually fail on a wrong Jacobian."""
    from pydae.core.builder import MlxBuilder
    from pydae.core.builder.mlx_sparsity import verify_pattern

    model = MlxBuilder(genrou_sys, dtype="float64", verbose=False).build()
    model.ini({})

    def corrupted(vector, *args):
        return model._jac_ini(vector, *args) * 1.5

    with on_stream(model):
        primal, rest = _ini_point(model)
        _, ok = verify_pattern(model._fg_ini, corrupted, primal, rest, 1e-9)
    assert not ok
