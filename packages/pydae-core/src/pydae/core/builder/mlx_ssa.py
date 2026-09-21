r"""Sparse small-signal analysis for the MLX backend (Apple Silicon).

This module linearises a converged DAE operating point and extracts eigenvalues
and eigenvectors **without ever forming the reduced state matrix**
:math:`A = F_x - F_y G_y^{-1} G_x`.  Inverting :math:`G_y` destroys the sparsity
of the network and, on a large system, the memory with it.

Augmented formulation
---------------------
The dynamic and algebraic states are stacked into one vector, and the whole DAE
is evaluated as a single residual function:

.. math::

    \xi = \begin{bmatrix} x \\ y \end{bmatrix}, \qquad
    \lambda(\xi) = \begin{bmatrix} f(\xi) \\ g(\xi) \end{bmatrix}, \qquad
    0 = \lambda(\xi^\star)

Automatic differentiation of :math:`\lambda` at the equilibrium gives the
augmented Jacobian, and the small-signal problem becomes a *generalized*
eigenvalue problem against a singular mass matrix:

.. math::

    \Lambda_\xi = \begin{bmatrix} F_x & F_y \\ G_x & G_y \end{bmatrix},
    \qquad
    \Lambda_\xi\,\phi = \lambda\,E\,\phi,
    \qquad
    E = \begin{bmatrix} I & 0 \\ 0 & 0 \end{bmatrix}

The pencil :math:`(\Lambda_\xi, E)` has exactly ``N_x`` finite eigenvalues --
identical to the eigenvalues of :math:`A`, verified here to 1e-13 -- plus
``N_y`` infinite ones contributed by the algebraic block.  Shift-invert maps the
infinite ones to zero, so they filter themselves out.

Shift-invert Arnoldi, matrix free
---------------------------------
Arnoldi is run on

.. math::

    M = (\Lambda_\xi - \sigma E)^{-1} E,
    \qquad
    \mu = \frac{1}{\lambda - \sigma}
    \;\Longleftrightarrow\;
    \lambda = \sigma + \frac{1}{\mu}

so the modes nearest the shift :math:`\sigma` converge first.  Nothing of size
``n x n`` is ever built:

* :math:`\Lambda_\xi v` comes from a forward-mode ``mx.jvp`` pass over
  :math:`\lambda(\xi)` -- a Jacobian-vector product, not a Jacobian;
* :math:`\Lambda_\xi^{\mathsf T} w` comes from a reverse-mode ``mx.vjp`` pass,
  which is what yields the left eigenvectors;
* :math:`E v` zeroes the algebraic tail, an elementwise kernel;
* the shift-invert solve is matrix-free GMRES, built only from those products.

Both AD passes are wrapped in ``mx.compile``, so each one dispatches as a fused
Metal kernel graph.

What is *not* on the GPU
------------------------
MLX 0.32 has **no GPU linear algebra at all** -- ``solve``, ``qr``, ``eig``,
``lu_factor`` and ``cholesky`` every one raise "not yet supported on the GPU".
That is precisely why the solve here is an iterative Krylov method: it is built
from matrix-vector products, which *are* GPU kernels.  The only work that leaves
the device is the projected ``Krylov_dim x Krylov_dim`` Hessenberg
eigenproblem, with ``Krylov_dim`` on the order of 30 -- independent of the
system size, and microseconds either way.

Naming
------
Following the caller's convention: constants and parameters take an initial
capital (``Shift``, ``Tolerance``, ``Krylov_dim``), runtime variables are
lowercase (``eigenvalues``, ``krylov_basis``, ``residual_vector``), no
single-letter names are used anywhere, and subindices use ``r`` for a reference
and ``c`` for a limit.
"""

from __future__ import annotations

import warnings

import numpy as np
from pydae.core.builder.mlx_builder import _mx, _on_stream

__all__ = ["MlxSmallSignal", "small_signal"]


class MlxSmallSignal:
    """Eigenvalues, eigenvectors and participation factors of one operating point.

    Attributes
    ----------
    eigenvalues : ndarray, shape (n_modes,)
        Finite eigenvalues of the pencil, :math:`\\lambda = \\sigma_r \\pm j\\omega`,
        ordered by distance from the shift.
    right_eigenvectors : ndarray, shape (n_total, n_modes)
        Columns :math:`\\phi_i` solving :math:`\\Lambda_\\xi \\phi_i = \\lambda_i E \\phi_i`.
        The first ``N_x`` rows are the dynamic part, the rest the algebraic part.
    left_eigenvectors : ndarray, shape (n_total, n_modes)
        Columns :math:`\\psi_i` solving :math:`\\Lambda_\\xi^{\\mathsf T} \\psi_i =
        \\lambda_i E^{\\mathsf T} \\psi_i`, normalised so
        :math:`\\psi_i^{\\mathsf T} E \\phi_i = 1` (plain transpose -- the
        conjugate transpose pairs a mode with its complex conjugate and vanishes).
    participation : ndarray, shape (N_x, n_modes)
        Kundur-normalised participation factors over the dynamic states.
    damping, frequency : ndarray, shape (n_modes,)
        Damping ratio and frequency in Hz of each mode.
    mode_converged : ndarray of bool, shape (n_modes,)
        Whether each mode met ``Mode_tolerance``. A mode that did not is still
        returned, but it is not an eigenvalue -- check this before trusting it.
    mode_residual : ndarray, shape (n_modes,)
        :math:`\\|\\Lambda_\\xi \\phi_i - \\lambda_i E \\phi_i\\|_\\infty`, the honest
        check on each converged mode.
    """

    def __init__(self, eigenvalues, right_eigenvectors, left_eigenvectors,
                 participation, mode_residual, mode_converged, state_names,
                 Shift, solver_residual, matrix_free):
        self.eigenvalues = eigenvalues
        self.right_eigenvectors = right_eigenvectors
        self.left_eigenvectors = left_eigenvectors
        self.participation = participation
        self.mode_residual = mode_residual
        self.mode_converged = mode_converged
        self.state_names = state_names
        self.Shift = Shift
        self.solver_residual = solver_residual
        self.matrix_free = matrix_free

    @property
    def damping(self):
        magnitude = np.abs(self.eigenvalues)
        return np.where(magnitude > 0, -self.eigenvalues.real / np.maximum(magnitude, 1e-300), 1.0)

    @property
    def frequency(self):
        return np.abs(self.eigenvalues.imag) / (2.0 * np.pi)

    def damp_frame(self):
        """Damping table, one row per mode (mirrors :func:`pydae.ssa.ssa.damp`)."""
        import pandas as pd
        return pd.DataFrame({
            "Real": self.eigenvalues.real,
            "Imag": self.eigenvalues.imag,
            "Freq.": self.frequency,
            "Damp": self.damping,
            "Residual": self.mode_residual,
            "Converged": self.mode_converged,
        }, index=[f"Mode {index + 1}" for index in range(len(self.eigenvalues))])

    def participation_frame(self, mode=None):
        """Participation factors as a DataFrame indexed by dynamic state name."""
        import pandas as pd
        columns = [f"Mode {index + 1}" for index in range(self.participation.shape[1])]
        frame = pd.DataFrame(self.participation, index=self.state_names, columns=columns)
        return frame if mode is None else frame[f"Mode {mode + 1}"].sort_values(ascending=False)

    def __repr__(self):
        return (f"<MlxSmallSignal {len(self.eigenvalues)} modes near Shift={self.Shift:g}, "
                f"worst mode residual {np.max(self.mode_residual):.1e}>")


# ---------------------------------------------------------------------------
# matrix-free operators
# ---------------------------------------------------------------------------
def _build_operators(builder, Shift):
    """Compiled, matrix-free actions of the augmented Jacobian and mass matrix.

    Returns ``(augmented_residual, apply_mass, apply_shifted,
    apply_shifted_transpose, equilibrium)``. None of them ever materialises a
    matrix: the Jacobian only ever appears through ``jvp`` / ``vjp`` products.
    """
    mx = _mx()
    n_x, n_y = builder.N_x, builder.N_y_run

    parameters = builder._arr(builder.p_vals)
    inputs = builder._arr(builder.u_run_vals)
    frozen = builder._arr(builder.c_run_vals)
    equilibrium = mx.concatenate([builder._arr(builder.x), builder._arr(builder.y_run)])

    def augmented_residual(xi):
        """lambda(xi) = [f(xi); g(xi)] -- the whole DAE as one vector function."""
        return mx.concatenate([
            builder._f_run(xi[:n_x], xi[n_x:], parameters, inputs, frozen),
            builder._g_run(xi[:n_x], xi[n_x:], parameters, inputs, frozen)])

    def apply_mass(vector):
        """E @ vector: keep the dynamic block, annihilate the algebraic tail."""
        return mx.concatenate([vector[:n_x], mx.zeros((n_y,), vector.dtype)])

    @mx.compile
    def apply_jacobian(vector):
        """Lambda_xi @ vector, one forward-mode AD pass."""
        return mx.jvp(augmented_residual, (equilibrium,), (vector,))[1][0]

    @mx.compile
    def apply_jacobian_transpose(vector):
        """Lambda_xi.T @ vector, one reverse-mode AD pass."""
        return mx.vjp(augmented_residual, (equilibrium,), (vector,))[1][0]

    def apply_shifted(vector):
        return apply_jacobian(vector) - Shift * apply_mass(vector)

    def apply_shifted_transpose(vector):
        # E is symmetric, so (Lambda - Shift E).T = Lambda.T - Shift E.
        return apply_jacobian_transpose(vector) - Shift * apply_mass(vector)

    return (augmented_residual, apply_mass, apply_jacobian, apply_shifted,
            apply_shifted_transpose, equilibrium)


def _generalized_residual(builder, apply_jacobian, apply_mass, eigenvector, eigenvalue):
    """Relative ``||Lambda phi - lambda E phi||`` for one candidate mode.

    The model residual is real, so a complex eigenvector is pushed through the
    operators as two real passes and recombined -- no complex tracing through
    the lowered equations, which contain comparisons and piecewise branches.
    """
    real_part = builder._arr(eigenvector.real)
    imag_part = builder._arr(eigenvector.imag)
    jacobian_action = (np.array(apply_jacobian(real_part), dtype=np.float64)
                       + 1j * np.array(apply_jacobian(imag_part), dtype=np.float64))
    mass_action = (np.array(apply_mass(real_part), dtype=np.float64)
                   + 1j * np.array(apply_mass(imag_part), dtype=np.float64))
    deviation = jacobian_action - eigenvalue * mass_action
    scale = max(float(np.max(np.abs(jacobian_action))), 1e-30)
    return float(np.max(np.abs(deviation))) / scale


def _norm(vector):
    return _mx().sqrt(_mx().sum(vector * vector))


def _gmres(apply_operator, right_hand_side, Restart, Tolerance, Max_cycles):
    """Restarted GMRES using only matrix-vector products.

    Projections are done as two matmuls against the stacked basis (classical
    Gram-Schmidt, repeated once for stability) rather than a loop of dot
    products.  That matters far more than it looks: a per-coefficient
    formulation forces one host synchronisation per inner iteration, which on
    these vector lengths costs more than the arithmetic by an order of magnitude.
    """
    mx = _mx()
    dtype = right_hand_side.dtype
    n_total = right_hand_side.shape[0]
    Restart = int(min(Restart, n_total))

    solution = mx.zeros((n_total,), dtype=dtype)
    reference_norm = float(_norm(right_hand_side))
    if reference_norm == 0.0:
        return solution, 0.0

    relative_residual = 1.0
    previous_residual = np.inf
    for _ in range(Max_cycles):
        residual_vector = right_hand_side - apply_operator(solution)
        beta = float(_norm(residual_vector))
        relative_residual = beta / reference_norm
        if relative_residual < Tolerance:
            break
        # Stop once a cycle no longer buys anything. In float32 the requested
        # tolerance is often below what the arithmetic can reach, and without
        # this every solve would burn its full cycle budget making no progress.
        if relative_residual > 0.9 * previous_residual:
            break
        previous_residual = relative_residual

        krylov_basis = [residual_vector / beta]
        hessenberg = np.zeros((Restart + 1, Restart))
        used = Restart
        for step in range(Restart):
            candidate = apply_operator(krylov_basis[step])
            basis_matrix = mx.stack(krylov_basis, axis=1)          # (n_total, step+1)
            coefficients = basis_matrix.T @ candidate
            candidate = candidate - basis_matrix @ coefficients
            correction = basis_matrix.T @ candidate                # re-orthogonalise
            candidate = candidate - basis_matrix @ correction
            coefficients = coefficients + correction
            next_norm = _norm(candidate)
            mx.eval(coefficients, next_norm)
            hessenberg[:step + 1, step] = np.array(coefficients)
            next_norm = float(next_norm)
            hessenberg[step + 1, step] = next_norm
            if next_norm < 1e-13 * max(beta, 1.0):
                used = step + 1
                break
            krylov_basis.append(candidate / next_norm)

        projected_rhs = np.zeros(used + 1)
        projected_rhs[0] = beta
        weights, *_ = np.linalg.lstsq(hessenberg[:used + 1, :used], projected_rhs, rcond=None)
        basis_matrix = mx.stack(krylov_basis[:used], axis=1)
        solution = solution + basis_matrix @ mx.array(weights.astype(np.float32)
                                                      if dtype == mx.float32 else weights)

    final_residual = right_hand_side - apply_operator(solution)
    return solution, float(_norm(final_residual)) / reference_norm


def _arnoldi(apply_operator, start_vector, Krylov_dim):
    """Arnoldi factorisation of an operator given only its action.

    Returns the basis as an ``(n_total, steps)`` device array and the projected
    Hessenberg as a host array.
    """
    mx = _mx()
    krylov_basis = [start_vector / _norm(start_vector)]
    hessenberg = np.zeros((Krylov_dim + 1, Krylov_dim))
    steps = Krylov_dim

    for step in range(Krylov_dim):
        candidate = apply_operator(krylov_basis[step])
        basis_matrix = mx.stack(krylov_basis, axis=1)
        coefficients = basis_matrix.T @ candidate
        candidate = candidate - basis_matrix @ coefficients
        correction = basis_matrix.T @ candidate
        candidate = candidate - basis_matrix @ correction
        coefficients = coefficients + correction
        next_norm = _norm(candidate)
        mx.eval(coefficients, next_norm)
        hessenberg[:step + 1, step] = np.array(coefficients)
        next_norm = float(next_norm)
        hessenberg[step + 1, step] = next_norm
        if next_norm < 1e-12:
            steps = step + 1
            break
        krylov_basis.append(candidate / next_norm)

    return mx.stack(krylov_basis[:steps], axis=1), hessenberg[:steps, :steps], steps


def _pair_left_with_right(left_values, left_vectors, right_values):
    """Match the transposed problem's modes to the direct problem's, by value."""
    paired = np.zeros((left_vectors.shape[0], len(right_values)), dtype=complex)
    if len(left_values) == 0:
        return paired
    available = list(range(len(left_values)))
    for column, value in enumerate(right_values):
        distances = [abs(left_values[index] - value) for index in available]
        best = available[int(np.argmin(distances))]
        paired[:, column] = left_vectors[:, best]
        if len(available) > 1:
            available.remove(best)
    return paired


@_on_stream
def small_signal(builder, Shift=-0.5, N_modes=None, Krylov_dim=None,
                 Tolerance=None, Max_cycles=20, Restart=None,
                 Mode_tolerance=None, matrix_free=True, left_vectors=True):
    """Eigenvalues and eigenvectors of the pencil ``(Lambda_xi, E)``.

    Call after :meth:`MlxBuilder.ini` -- the linearisation is taken at whatever
    operating point the model currently holds.

    Parameters
    ----------
    Shift : float
        Shift-invert target :math:`\\sigma`.  Modes closest to it converge first,
        so pick it near the dynamics of interest: ``0.0`` or a small negative
        value for the slow electromechanical modes, a large negative value for
        fast converter modes.
    N_modes : int, optional
        How many modes to return (default: all ``N_x`` finite ones).
    Krylov_dim : int, optional
        Arnoldi subspace dimension.  Defaults to ``min(2*N_modes + 12, n_total)``.
        Larger is more robust and costs one shift-invert solve per extra step.
    Tolerance, Max_cycles, Restart
        Inner GMRES controls (relative residual, restart cycles, cycle length).
    Mode_tolerance : float, optional
        A Ritz pair is accepted only if its relative residual
        :math:`\\|\\Lambda_\\xi\\phi - \\lambda E\\phi\\| / \\|\\Lambda_\\xi\\phi\\|`
        falls below this.  Defaults to 1e-8 in float64 and 1e-2 in float32,
        which is what it takes to reject the ghost Ritz pairs that a float32
        Krylov basis produces once it loses orthogonality.
    matrix_free : bool
        ``True`` (default) never forms a matrix: the shift-invert solve is GMRES
        driven by ``jvp``/``vjp`` products, entirely on the GPU.  ``False``
        forms ``Lambda_xi`` once in a single AD pass and factorises it on the
        CPU -- far faster below a few hundred states, and still never touches
        ``A`` or :math:`G_y^{-1}`.
    left_vectors : bool
        Also run the transposed Arnoldi to obtain left eigenvectors and
        participation factors.  Roughly doubles the cost.

    Returns
    -------
    MlxSmallSignal
    """
    mx = _mx()
    builder._require_built()
    n_x, n_y = builder.N_x, builder.N_y_run
    n_total = n_x + n_y
    N_modes = int(N_modes or n_x)
    N_modes = max(1, min(N_modes, n_x))
    Krylov_dim = int(Krylov_dim or min(2 * N_modes + 12, n_total))
    Krylov_dim = max(N_modes + 2, min(Krylov_dim, n_total))
    # Full GMRES by default. Restarting below the system order stagnates badly
    # on an augmented power-system Jacobian -- measured on the Kundur two-area
    # system (n = 87): Restart=60 flatlines at a relative residual of 4e-1,
    # while full GMRES reaches 5e-12 in two cycles. Restricting Restart trades
    # Krylov memory for convergence and needs a preconditioner to be safe.
    Restart = int(Restart or n_total)

    (augmented_residual, apply_mass, apply_jacobian, apply_shifted,
     apply_shifted_transpose, equilibrium) = _build_operators(builder, Shift)
    double_precision = builder.np_dtype is np.float64
    if Mode_tolerance is None:
        Mode_tolerance = 1e-8 if double_precision else 1e-2
    if Tolerance is None:
        Tolerance = 1e-10 if double_precision else 1e-6

    worst_solver_residual = 0.0

    if matrix_free:
        def solve_shifted(vector):
            nonlocal worst_solver_residual
            solution, relative = _gmres(apply_shifted, vector, Restart, Tolerance, Max_cycles)
            worst_solver_residual = max(worst_solver_residual, relative)
            return solution

        def solve_shifted_transpose(vector):
            nonlocal worst_solver_residual
            solution, relative = _gmres(apply_shifted_transpose, vector,
                                        Restart, Tolerance, Max_cycles)
            worst_solver_residual = max(worst_solver_residual, relative)
            return solution
    else:
        # One AD pass for the whole augmented Jacobian, then a reusable
        # factorisation of (Lambda - Shift E). Still no A, still no inverse of Gy.
        import scipy.linalg as linalg
        # Compressed by the augmented block's colouring: a handful of AD
        # passes instead of one per equation (9 rather than 87 on k12p6).
        jacobian_dense = builder._make_jacobian(augmented_residual, n_total, "aug")
        jacobian_host = np.array(jacobian_dense(equilibrium), dtype=np.float64)
        mass_host = np.zeros((n_total, n_total))
        mass_host[:n_x, :n_x] = np.eye(n_x)
        factorisation = linalg.lu_factor(jacobian_host - Shift * mass_host)

        def solve_shifted(vector):
            return builder._arr(linalg.lu_solve(factorisation, np.array(vector, dtype=np.float64)))

        def solve_shifted_transpose(vector):
            return builder._arr(linalg.lu_solve(factorisation,
                                                np.array(vector, dtype=np.float64), trans=1))

    def apply_shift_invert(vector):
        """M @ vector = (Lambda - Shift E)^-1 E @ vector."""
        return solve_shifted(apply_mass(vector))

    def apply_shift_invert_transpose(vector):
        """M.T @ vector = E (Lambda - Shift E)^-T @ vector."""
        return apply_mass(solve_shifted_transpose(vector))

    generator = np.random.default_rng(0)
    start_vector = builder._arr(generator.standard_normal(n_total))

    basis, hessenberg, steps = _arnoldi(apply_shift_invert, start_vector, Krylov_dim)
    ritz_values, ritz_vectors = np.linalg.eig(hessenberg)
    basis_host = np.array(basis, dtype=np.float64)

    # mu = 1/(lambda - Shift), so the pencil's infinite eigenvalues collapse onto
    # mu = 0 and the finite ones are lambda = Shift + 1/mu.
    finite = np.abs(ritz_values) > 1e-12
    ritz_values, ritz_vectors = ritz_values[finite], ritz_vectors[:, finite]
    candidate_values = Shift + 1.0 / ritz_values
    candidate_vectors = basis_host @ ritz_vectors

    # Select on the Ritz residual, not on |mu|. In float32 the Krylov basis
    # loses orthogonality and throws off ghost Ritz pairs with plausible-looking
    # |mu|; they are betrayed by a residual orders of magnitude worse than the
    # converged modes, whereas ranking by |mu| alone would let one displace a
    # genuine mode.
    candidate_residual = np.array([
        _generalized_residual(builder, apply_jacobian, apply_mass,
                              candidate_vectors[:, column], candidate_values[column])
        for column in range(len(candidate_values))])

    accepted = np.where(candidate_residual < Mode_tolerance)[0]
    if len(accepted) == 0:
        # Returning the least-bad Ritz pairs unannounced would be the worst
        # possible failure mode: they look like eigenvalues. Keep them, but say so.
        accepted = np.argsort(candidate_residual)[:N_modes]
        warnings.warn(
            f"no Ritz pair reached Mode_tolerance={Mode_tolerance:.1e}; the best "
            f"residual was {candidate_residual.min():.1e}. The returned modes are "
            "NOT converged -- raise Krylov_dim, move Shift closer to the modes of "
            "interest, or switch to matrix_free=False.",
            RuntimeWarning, stacklevel=2)
    nearest_first = accepted[np.argsort(np.abs(candidate_values[accepted] - Shift))]
    selected = nearest_first[:N_modes]

    eigenvalues = candidate_values[selected]
    right_eigenvectors = candidate_vectors[:, selected]
    mode_residual = candidate_residual[selected]
    mode_converged = mode_residual < Mode_tolerance
    if not mode_converged.all():
        warnings.warn(
            f"{int((~mode_converged).sum())} of {len(mode_converged)} returned modes "
            f"exceed Mode_tolerance={Mode_tolerance:.1e} (worst "
            f"{mode_residual.max():.1e}); check MlxSmallSignal.mode_converged.",
            RuntimeWarning, stacklevel=2)

    if left_vectors:
        left_basis, left_hessenberg, _ = _arnoldi(apply_shift_invert_transpose,
                                                  start_vector, Krylov_dim)
        left_ritz_values, left_ritz_vectors = np.linalg.eig(left_hessenberg)
        keep = np.abs(left_ritz_values) > 1e-10
        left_values = Shift + 1.0 / left_ritz_values[keep]
        left_host = np.array(left_basis, dtype=np.float64) @ left_ritz_vectors[:, keep]
        left_eigenvectors = _pair_left_with_right(left_values, left_host, eigenvalues)
    else:
        left_eigenvectors = np.zeros_like(right_eigenvectors)

    # Normalise so psi_i^T E phi_i = 1, the scaling participation factors need.
    # The pairing is the plain transpose, NOT the conjugate transpose: for a
    # complex mode, conj(psi_i) is the left eigenvector of the conjugate
    # eigenvalue, so psi_i^H E phi_i is ~0 by biorthogonality and normalising
    # against it divides by numerical noise.
    for column in range(right_eigenvectors.shape[1]):
        pairing = (left_eigenvectors[:n_x, column] * right_eigenvectors[:n_x, column]).sum()
        if abs(pairing) > 1e-30:
            left_eigenvectors[:, column] /= pairing

    participation = np.abs(left_eigenvectors[:n_x, :] * right_eigenvectors[:n_x, :])
    column_sums = participation.sum(axis=0)
    participation = participation / np.where(column_sums > 0, column_sums, 1.0)

    mx.eval(equilibrium)
    return MlxSmallSignal(eigenvalues, right_eigenvectors, left_eigenvectors,
                          participation, mode_residual, mode_converged,
                          list(builder.x_names), Shift, worst_solver_residual,
                          matrix_free)
