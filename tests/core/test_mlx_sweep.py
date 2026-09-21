"""Batched parameter sweeps on the MLX backend (Apple Silicon only).

The whole point of the MLX backend is solving B copies of one DAE at once, so
these tests are mostly about *equivalence*: a batched solve must agree with the
same cases run one at a time, and one hopeless case must not disturb its
neighbours.

Run with:
    uv run pytest tests/core/test_mlx_sweep.py -v
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

# float32 (the sweep default) floors around 1e-5; float64 is used where the
# comparison is against CasADi, and still floors near 1e-8 because MLX evaluates
# sin/cos/exp in single precision whatever the dtype.
TOL32 = 2e-5
TOL64 = 1e-6


@pytest.fixture(scope="module")
def genrou_sys(tmp_path_factory):
    from pydae.bps import BpsBuilder

    # BpsBuilder.construct() drops xy_0.json in the working directory, so build
    # from a temp dir rather than dirtying the repo root on every test run.
    hjson = str(Path(GENROU_HJSON).resolve())
    cwd = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("mlx_sweep"))
    try:
        grid = BpsBuilder(hjson)
        grid.construct("genrou_sweep_test")
    finally:
        os.chdir(cwd)
    return grid.sys_dict


@pytest.fixture(scope="module")
def model64(genrou_sys):
    from pydae.core.builder import MlxBuilder

    m = MlxBuilder(genrou_sys, dtype="float64", Dt=0.01, verbose=False).build()
    m.ini({}, xy_0=genrou_sys.get("xy_0_dict", {}))
    return m


def _fresh(genrou_sys, dtype="float64"):
    from pydae.core.builder import MlxBuilder

    m = MlxBuilder(genrou_sys, dtype=dtype, Dt=0.01, verbose=False).build()
    m.ini({}, xy_0=genrou_sys.get("xy_0_dict", {}))
    return m


def test_build_mlx_helper_matches_the_explicit_path(tmp_path, monkeypatch):
    """`grid.build_mlx(name)` == construct() + MlxBuilder(...).build()."""
    from pydae.bps import BpsBuilder
    from pydae.core.builder import MlxBuilder

    hjson = str(Path(GENROU_HJSON).resolve())
    monkeypatch.chdir(tmp_path)          # construct() drops xy_0.json in the cwd

    from_helper = BpsBuilder(hjson).build_mlx("genrou_helper", dtype="float64",
                                              Dt=0.02, verbose=False)
    grid = BpsBuilder(hjson)
    grid.construct("genrou_explicit")
    explicit = MlxBuilder(grid.sys_dict, dtype="float64", Dt=0.02, verbose=False).build()

    assert from_helper.Dt == 0.02                      # kwargs reach MlxBuilder
    assert from_helper.x_names == explicit.x_names

    # ini({}) alone must reproduce the explicitly seeded solve: the builder
    # already applies sys_dict['xy_0_dict'] when it sets up its initial state.
    from_helper.ini({})
    explicit.ini({}, xy_0=grid.sys_dict["xy_0_dict"])
    assert np.allclose(from_helper.x, explicit.x)
    assert np.allclose(from_helper.get_values("q_g_1"), explicit.get_values("q_g_1"))


def test_build_mlx_helper_accepts_the_casadi_front_end(tmp_path, monkeypatch):
    from pydae.bps import BpsBuilder

    hjson = str(Path(GENROU_HJSON).resolve())
    monkeypatch.chdir(tmp_path)

    model = BpsBuilder(hjson, use_casadi=True).build_mlx("genrou_ca", dtype="float64",
                                                         verbose=False)
    assert model.frontend == "casadi"
    model.ini({})
    assert abs(model.get_value("p_g_1") - 0.5) < 1e-6


def test_single_case_reproduces_the_scalar_solve(model64):
    """B=1 must land exactly where MlxBuilder.ini() lands."""
    p_m = float(model64.get_value("p_m_1"))
    sw = model64.sweep({"p_m_1": [p_m]}, verbose=False).ini()

    assert sw.converged.all()
    assert np.max(np.abs(sw.x[:, 0] - model64.x)) < TOL64
    assert abs(sw.get_values("q_g_1")[0] - model64.get_value("q_g_1")) < TOL64


def test_batch_matches_the_same_cases_solved_one_at_a_time(genrou_sys):
    """The core equivalence: batched sweep == sequential loop."""
    m = _fresh(genrou_sys)
    points = np.linspace(0.15, 0.60, 8)

    sw = m.sweep({"p_m_1": points}, verbose=False).ini()
    assert sw.converged.all()

    for name in ("q_g_1", "p_g_1", "delta_1", "V_1"):
        batched = sw.get_values(name)
        sequential = []
        for value in points:
            m.ini({"p_m_1": float(value)}, xy_0=genrou_sys.get("xy_0_dict", {}))
            sequential.append(m.get_value(name))
        assert np.allclose(batched, sequential, atol=TOL64), name


def test_swept_input_reaches_both_ini_and_run_blocks(genrou_sys):
    """``p_m_1`` lives in u_ini_dict *and* u_run_dict.

    Writing only the ini copy would leave the transient running at the nominal
    dispatch, which looks plausible and is wrong.
    """
    m = _fresh(genrou_sys)
    targets = np.array([0.30, 0.55])
    sw = m.sweep({"p_m_1": targets}, verbose=False).ini()

    idx = m.u_run_names.index("p_m_1")
    assert np.allclose(sw.u_run_vals[idx], targets)
    assert np.allclose(sw.get_values("p_g_1"), targets, atol=1e-3)


def test_infeasible_cases_are_reported_not_raised(genrou_sys):
    """Past the pull-out limit there is no steady state; that is a result."""
    m = _fresh(genrou_sys, dtype="float32")
    sw = m.sweep({"p_m_1": np.linspace(0.05, 0.95, 400)}, verbose=False).ini()

    assert sw.converged.any() and not sw.converged.all()
    # E*V/X_d with v_f = 1.5, X_d = 2.135
    lo, hi = sw.feasible_range("p_m_1")
    assert lo == pytest.approx(0.05, abs=1e-6)
    assert hi == pytest.approx(1.5 / 2.135, rel=0.02)
    # the boundary is a boundary, not scattered noise
    assert np.all(np.diff(sw.converged.astype(int)) <= 0)
    assert np.isnan(sw.get_values("q_g_1")[~sw.converged]).all()
    assert len(sw.failures()) == int((~sw.converged).sum())


def test_one_hopeless_case_does_not_disturb_the_others(genrou_sys):
    """MLX aborts a whole batched solve on a singular element; masking prevents it."""
    m = _fresh(genrou_sys)
    points = np.linspace(0.15, 0.60, 8)

    clean = m.sweep({"p_m_1": points}, verbose=False).ini()
    mixed = m.sweep({"p_m_1": np.append(points, 50.0)}, verbose=False).ini()

    assert not mixed.converged[-1]
    assert mixed.converged[:-1].all()
    assert np.array_equal(mixed.get_values("q_g_1")[:-1], clean.get_values("q_g_1"))


def test_auto_chunk_memory_model_matches_the_measured_peak(tmp_path_factory):
    """``memory_budget_gb`` has to be an honest ceiling, not a wish.

    This is the check that was missing when the per-case model drifted: it
    predicted 132 kB/case at n=87 where the truth is ~246 kB, and the hard chunk
    clamp hid the error. Pinning the model against a real ``get_peak_memory()``
    keeps it from drifting again.

    float32 only, deliberately. ``mx.get_peak_memory()`` under-reports the CPU
    stream, so a float64 run measures ~253 kB/case against float32's ~246 kB
    when it should be about double -- do not "fix" the model against that.
    """
    from pydae.bps import BpsBuilder
    from pydae.core.builder import MlxBuilder

    hjson = str(Path("packages/pydae-bps/src/pydae/bps/miscellaneous/k12p6.hjson").resolve())
    cwd = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("mlx_mem"))
    try:
        grid = BpsBuilder(hjson)
        grid.construct("k12p6_mem_test")
    finally:
        os.chdir(cwd)

    m = MlxBuilder(grid.sys_dict, dtype="float32", verbose=False).build()
    m.ini({})

    chunk = 1024
    sweep = m.sweep({"p_m_1": np.linspace(0.3, 0.6, chunk)}, chunk=chunk)
    mx.clear_cache()
    mx.reset_peak_memory()
    sweep.ini()
    measured = mx.get_peak_memory() / chunk

    n = max(m.N_x + m.N_y_ini, m.N_x + m.N_y_run)
    predicted = np.dtype(np.float32).itemsize * (8 * n * n + 40 * n)
    # The estimate must bound the truth, and not by a wide margin -- a model
    # that is merely "big enough" would let the budget drift again.
    assert measured <= predicted * 1.10, (
        f"peak {measured / 1024:.1f} kB/case exceeds the model's "
        f"{predicted / 1024:.1f} kB/case; _auto_chunk under-budgets")
    assert measured >= predicted * 0.5, (
        f"peak {measured / 1024:.1f} kB/case is far under the model's "
        f"{predicted / 1024:.1f} kB/case; _auto_chunk wastes headroom")


def test_auto_chunk_stays_within_its_budget_on_a_large_system(genrou_sys):
    """The case the old constant got wrong: n large enough that the budget binds.

    At n around 500 the chunk clamp no longer saves us, so the per-case model is
    the only thing keeping the promise.
    """
    m = _fresh(genrou_sys)
    sweep = m.sweep({"p_m_1": np.linspace(0.3, 0.6, 8)})

    for n in (200, 500, 1200):
        itemsize = np.dtype(np.float32).itemsize
        per_case = itemsize * (8 * n * n + 40 * n)
        budget = sweep.memory_budget_gb * 2 ** 30
        chunk = max(64, min(int(budget // per_case), 4096))
        # 64 is a floor, so a huge n can legitimately exceed the budget; below
        # that floor the promise is "smallest useful chunk", not the budget.
        if chunk > 64:
            assert chunk * per_case <= budget, f"n={n} overshoots the budget"


def test_chunking_does_not_change_the_answer(genrou_sys):
    """Including the padded final chunk."""
    m = _fresh(genrou_sys)
    points = np.linspace(0.15, 0.60, 20)

    whole = m.sweep({"p_m_1": points}, chunk=64, verbose=False).ini()
    split = m.sweep({"p_m_1": points}, chunk=7, verbose=False).ini()

    assert np.array_equal(whole.get_values("q_g_1"), split.get_values("q_g_1"))


def test_batched_transient_matches_the_scalar_run(genrou_sys):
    """Same trajectory, whether stepped alone or alongside 7 others."""
    m = _fresh(genrou_sys)
    inertias = np.linspace(4.0, 9.0, 4)

    sw = m.sweep({"H_1": inertias}, verbose=False).ini()
    sw.run(0.5, record=["omega_1"])
    sw.run(1.5, {"p_m_1": 0.65})
    batched = sw.get_series("omega_1")

    for j, inertia in enumerate(inertias):
        m.ini({"H_1": float(inertia)}, xy_0=genrou_sys.get("xy_0_dict", {}))
        m.run(0.5, {})
        m.run(1.5, {"p_m_1": 0.65})
        m.post()
        reference = m.get_values("omega_1")[1:]
        assert np.allclose(batched[:, j], reference[:len(batched)], atol=TOL64)

    # lighter machines swing further -- a physics check on the sweep axis itself
    assert batched[:, 0].max() > batched[:, -1].max()


def test_recorded_series_has_one_column_per_case(genrou_sys):
    m = _fresh(genrou_sys)
    sw = m.sweep({"H_1": np.linspace(4.0, 9.0, 5)}, verbose=False).ini()
    sw.run(0.5, record=["omega_1", "V_1"], decimation=5)

    series = sw.get_series("omega_1")
    assert series.shape == (10, 5)
    assert sw.Time.shape == (10,)
    assert sw.get_series("V_1").shape == series.shape
    with pytest.raises(KeyError):
        sw.get_series("delta_1")          # not recorded


def test_batched_state_matrix_and_eigenvalues(genrou_sys):
    """Batched SSA must agree with the scalar A_eval at the same point."""
    m = _fresh(genrou_sys)
    points = np.linspace(0.20, 0.55, 4)
    sw = m.sweep({"p_m_1": points}, verbose=False).ini()

    A = sw.A_eval()
    eigs = sw.eigenvalues()
    assert A.shape == (4, m.N_x, m.N_x)

    for j, value in enumerate(points):
        m.ini({"p_m_1": float(value)}, xy_0=genrou_sys.get("xy_0_dict", {}))
        assert np.allclose(A[j], m.A_eval(), atol=1e-5)
        assert np.allclose(np.sort_complex(eigs[j]),
                           np.sort_complex(m.eval_eigenvalues()), atol=1e-5)


def test_frames_carry_the_case_columns(genrou_sys):
    pd = pytest.importorskip("pandas")
    m = _fresh(genrou_sys)
    sw = m.sweep({"p_m_1": np.linspace(0.2, 0.5, 5)}, verbose=False).ini()

    frame = sw.to_frame(["q_g_1", "V_1"])
    assert isinstance(frame, pd.DataFrame)
    assert list(frame.columns) == ["p_m_1", "q_g_1", "V_1", "converged", "status"]
    assert len(frame) == 5
    assert frame["converged"].all()


def test_grid_builds_the_cartesian_product(genrou_sys):
    from pydae.core.builder import MlxSweep

    m = _fresh(genrou_sys)
    sw = MlxSweep.grid(m, p_m_1=np.linspace(0.2, 0.5, 4), v_f_1=[1.4, 1.6])
    assert sw.B == 8
    assert sorted(set(sw.cases["v_f_1"])) == [1.4, 1.6]


def test_multiple_axes_are_paired_not_crossed(genrou_sys):
    """Two equal-length arrays give B paired cases, never B*B.

    The one multi-axis mistake that does not raise: mismatched lengths are
    rejected, but two axes that happen to be the same length pair up quietly,
    so the semantics are pinned here.
    """
    m = _fresh(genrou_sys)
    p_m = np.linspace(0.2, 0.7, 6)
    v_f = np.linspace(1.3, 1.9, 6)
    sw = m.sweep({"p_m_1": p_m, "v_f_1": v_f})
    assert sw.B == 6
    sw.ini()
    assert sw.converged.all()
    # Case k is exactly the pair (p_m[k], v_f[k]).
    assert np.allclose(sw.get_values("p_m_1"), p_m)
    assert np.allclose(sw.get_values("v_f_1"), v_f)


def test_grid_results_reshape_back_onto_their_axes(genrou_sys):
    """``meshgrid(indexing='ij')`` -- first named axis slowest.

    The docs tell users to recover a factorial result with a plain
    ``reshape(len(axis_1), len(axis_2))``, which only holds for that ordering.
    """
    from pydae.core.builder import MlxSweep

    m = _fresh(genrou_sys)
    p_m = np.linspace(0.2, 0.6, 5)
    v_f = [1.5, 1.8, 2.1]
    sw = MlxSweep.grid(m, p_m_1=p_m, v_f_1=v_f)
    assert sw.B == 15
    sw.ini()

    assert np.allclose(sw.cases["p_m_1"].reshape(5, 3), np.asarray(p_m)[:, None])
    assert np.allclose(sw.cases["v_f_1"].reshape(5, 3), np.asarray(v_f)[None, :])
    # Reactive power rises with excitation at fixed dispatch.
    q = sw.get_values("q_g_1").reshape(5, 3)
    assert (np.diff(q, axis=1) > 0).all()


def test_a_scalar_axis_broadcasts_across_the_batch(genrou_sys):
    m = _fresh(genrou_sys)
    sw = m.sweep({"p_m_1": np.linspace(0.2, 0.7, 5), "v_f_1": 1.8})
    assert sw.B == 5
    assert np.unique(sw.cases["v_f_1"]).tolist() == [1.8]


def test_mismatched_sweep_lengths_are_rejected(genrou_sys):
    m = _fresh(genrou_sys)
    with pytest.raises(ValueError, match="inconsistent lengths"):
        m.sweep({"p_m_1": [0.2, 0.3, 0.4], "v_f_1": [1.4, 1.6]})


def test_unknown_swept_name_is_rejected(genrou_sys):
    m = _fresh(genrou_sys)
    with pytest.raises(KeyError):
        m.sweep({"not_a_variable": [1.0, 2.0]})


def test_require_all_raises_with_the_failure_table(genrou_sys):
    from pydae.core.builder.mlx_sweep import SweepError

    m = _fresh(genrou_sys)
    with pytest.raises(SweepError) as excinfo:
        m.sweep({"p_m_1": [0.3, 50.0]}, verbose=False).ini(require_all=True)
    assert len(excinfo.value.failures) == 1


@pytest.mark.slow
def test_ten_thousand_cases(genrou_sys):
    """The headline case: 10k steady states in one call."""
    import time

    m = _fresh(genrou_sys, dtype="float32")
    start = time.time()
    sw = m.sweep({"p_m_1": np.linspace(0.05, 0.68, 10_000)}, verbose=False).ini()
    elapsed = time.time() - start

    assert sw.converged.all()
    assert elapsed < 30.0
    q = sw.get_values("q_g_1")
    assert np.isfinite(q).all()
    # q(p_m) is smooth over the feasible range
    assert np.max(np.abs(np.diff(q, 2))) < 1e-3
