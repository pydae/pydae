"""Sparse small-signal analysis on the MLX backend (Apple Silicon only).

The reference these tests measure against is the dense reduced state matrix
``A = Fx - Fy Gy^-1 Gx``. The point of the module under test is to reach the
same answer *without* forming it, so most of these are equivalence tests.

Run with:
    uv run pytest tests/core/test_mlx_ssa.py -v
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


@pytest.fixture(scope="module")
def genrou_sys(tmp_path_factory):
    from pydae.bps import BpsBuilder

    hjson = str(Path(GENROU_HJSON).resolve())
    cwd = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("mlx_ssa"))
    try:
        grid = BpsBuilder(hjson)
        grid.construct("genrou_ssa_test")
    finally:
        os.chdir(cwd)
    return grid.sys_dict


def _model(genrou_sys, dtype="float64"):
    from pydae.core.builder import MlxBuilder

    model = MlxBuilder(genrou_sys, dtype=dtype, verbose=False).build()
    model.ini({}, xy_0=genrou_sys.get("xy_0_dict", {}))
    return model


@pytest.fixture(scope="module")
def dense_reference(genrou_sys):
    """Eigen-decomposition of the reduced A, the thing we must reproduce."""
    model = _model(genrou_sys)
    state_matrix = np.array(model.A_eval())
    values, right = np.linalg.eig(state_matrix)
    left = np.linalg.inv(right).T
    participation = np.abs(right * left)
    participation = participation / participation.sum(axis=0)
    return values, right, participation


def test_pencil_reproduces_the_reduced_state_matrix(genrou_sys, dense_reference):
    """The finite eigenvalues of (Lambda_xi, E) are exactly the eigenvalues of A."""
    model = _model(genrou_sys)
    reference_values = dense_reference[0]

    result = model.small_signal(Shift=0.0, N_modes=model.N_x, matrix_free=False)

    assert len(result.eigenvalues) == model.N_x
    for value in result.eigenvalues:
        assert np.min(np.abs(reference_values - value)) < 1e-8


def test_matrix_free_agrees_with_the_factorised_path(genrou_sys):
    """GMRES-driven shift-invert must match the LU-driven one."""
    model = _model(genrou_sys)

    free = model.small_signal(Shift=0.0, N_modes=4, matrix_free=True)
    factorised = model.small_signal(Shift=0.0, N_modes=4, matrix_free=False)

    for value in free.eigenvalues:
        assert np.min(np.abs(factorised.eigenvalues - value)) < 1e-6


def test_eigenvectors_satisfy_the_generalized_problem(genrou_sys):
    """Every returned mode must actually solve Lambda phi = lambda E phi."""
    model = _model(genrou_sys)
    result = model.small_signal(Shift=0.0, N_modes=5)

    assert result.right_eigenvectors.shape[0] == model.N_x + model.N_y_run
    assert np.max(result.mode_residual) < 1e-7


def test_participation_matches_the_dense_computation(genrou_sys, dense_reference):
    reference_values, _, reference_participation = dense_reference
    model = _model(genrou_sys)
    result = model.small_signal(Shift=0.0, N_modes=model.N_x)

    assert result.participation.shape == (model.N_x, len(result.eigenvalues))
    # each column is a normalised distribution over the dynamic states
    assert np.allclose(result.participation.sum(axis=0), 1.0)

    for column, value in enumerate(result.eigenvalues):
        match = int(np.argmin(np.abs(reference_values - value)))
        assert np.allclose(result.participation[:, column],
                           reference_participation[:, match], atol=1e-8)


def test_left_eigenvectors_are_normalised_against_the_mass_matrix(genrou_sys):
    """psi^T E phi = 1 is what makes the participation factors meaningful.

    The pairing is the plain transpose. Using the conjugate transpose pairs a
    complex mode with its own conjugate, which is biorthogonal to it, so the
    product collapses to ~0 and normalising against it divides by noise.
    """
    model = _model(genrou_sys)
    result = model.small_signal(Shift=0.0, N_modes=4)
    n_x = model.N_x

    for column in range(result.right_eigenvectors.shape[1]):
        pairing = (result.left_eigenvectors[:n_x, column]
                   * result.right_eigenvectors[:n_x, column]).sum()
        assert abs(pairing - 1.0) < 1e-6


def test_shift_selects_the_modes_nearest_to_it(genrou_sys, dense_reference):
    """Shift-invert is the whole reason to use Arnoldi here."""
    reference_values = dense_reference[0]
    model = _model(genrou_sys)

    slow = model.small_signal(Shift=0.0, N_modes=2)
    fast = model.small_signal(Shift=-57.0, N_modes=2)

    # Compare distances to the shift rather than the values themselves: the two
    # members of a conjugate pair are equidistant, so which one Arnoldi returns
    # when only part of the pair fits in N_modes is arbitrary.
    def distances(values, shift):
        return np.sort(np.abs(values - shift))

    assert np.allclose(distances(slow.eigenvalues, 0.0),
                       distances(reference_values, 0.0)[:2], atol=1e-6)
    assert np.allclose(distances(fast.eigenvalues, -57.0),
                       distances(reference_values, -57.0)[:2], atol=1e-4)


def test_ghost_ritz_pairs_are_rejected_in_float32(genrou_sys, dense_reference):
    """Regression: a float32 Krylov basis loses orthogonality and invents modes.

    Selecting by |mu| alone let a spurious pair displace three genuine
    eigenvalues; selection is on the Ritz residual for exactly this reason.
    """
    reference_values = dense_reference[0]
    model = _model(genrou_sys, dtype="float32")

    result = model.small_signal(Shift=0.0, N_modes=4, matrix_free=True)

    assert np.max(result.mode_residual) < 1e-2
    for value in result.eigenvalues:
        assert np.min(np.abs(reference_values - value)) < 1e-2


def test_requesting_fewer_modes_returns_the_closest_ones(genrou_sys):
    model = _model(genrou_sys)
    result = model.small_signal(Shift=0.0, N_modes=3)

    assert len(result.eigenvalues) == 3
    distances = np.abs(result.eigenvalues - 0.0)
    assert np.all(np.diff(distances) >= -1e-12)      # ordered by distance to Shift


def test_damping_and_frequency_are_reported(genrou_sys):
    pytest.importorskip("pandas")
    model = _model(genrou_sys)
    result = model.small_signal(Shift=0.0, N_modes=model.N_x)

    oscillatory = np.abs(result.eigenvalues.imag) > 1e-6
    assert oscillatory.any()
    assert np.all(result.damping <= 1.0 + 1e-12)
    assert np.allclose(result.frequency, np.abs(result.eigenvalues.imag) / (2 * np.pi))

    frame = result.damp_frame()
    assert list(frame.columns) == ["Real", "Imag", "Freq.", "Damp", "Residual", "Converged"]
    assert len(frame) == len(result.eigenvalues)

    table = result.participation_frame()
    assert list(table.index) == model.x_names


def test_skipping_left_vectors_still_returns_eigenvalues(genrou_sys, dense_reference):
    model = _model(genrou_sys)
    result = model.small_signal(Shift=0.0, N_modes=3, left_vectors=False)

    for value in result.eigenvalues:
        assert np.min(np.abs(dense_reference[0] - value)) < 1e-8
