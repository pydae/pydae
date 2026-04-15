# pydae/core/diagnostics/dae_check.py
"""
Numerical diagnostic tool for DAE solver failures.

Works with both dense and sparse Jacobians. When the solver is compiled with
a sparse backend (KLU, PARDISO, Accelerate), the Jacobian flat array contains
only the NNZ packed values — this module reconstructs the full dense matrix
from the sparsity pattern before running the analysis.

Improvements over the original version
=======================================
* Sparse support via (Ap, Ai) reconstruction.
* Zero-row / zero-column detection (equations or variables disconnected).
* Diagonal dominance check (useful for implicit integrators).
* Near-zero pivot detection along the diagonal.
* Newton convergence estimate from residual + condition number.
* Colour-coded terminal output (OK / WARNING / CRITICAL).
* Heatmap saved to file instead of blocking on ``plt.show()``.
* Summary verdict at the end with actionable next steps.
"""

import numpy as np
import logging

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# ANSI colour helpers (degrade gracefully on Windows without colorama)
# ---------------------------------------------------------------------------
try:
    _BOLD  = '\033[1m'
    _RED   = '\033[91m'
    _YEL   = '\033[93m'
    _GRN   = '\033[92m'
    _RST   = '\033[0m'
except Exception:
    _BOLD = _RED = _YEL = _GRN = _RST = ''

def _ok(msg):   return f"{_GRN}OK{_RST}      {msg}"
def _warn(msg): return f"{_YEL}WARNING{_RST}  {msg}"
def _crit(msg): return f"{_RED}CRITICAL{_RST} {msg}"


# ---------------------------------------------------------------------------
# Sparse → Dense reconstruction
# ---------------------------------------------------------------------------
def _sparse_to_dense(jac_values, Ap, Ai, N, backend):
    """
    Reconstruct a dense N×N matrix from packed sparse values + structure.

    Parameters
    ----------
    jac_values : ndarray (NNZ,)
        Packed nonzero values as written by jac_*_eval.
    Ap : array-like
        Column pointers (CSC, 0-based) for KLU / Accelerate, or
        row pointers (CSR, 1-based) for PARDISO.
    Ai : array-like
        Row indices (CSC) or column indices (CSR).
    N : int
        Matrix dimension.
    backend : str
        One of 'klu', 'accelerate', 'pardiso'.
    """
    J = np.zeros((N, N), dtype=np.float64)

    if backend in ('klu', 'accelerate'):
        # CSC: Ap[j] .. Ap[j+1]-1 are the row indices for column j
        for col in range(N):
            for idx in range(Ap[col], Ap[col + 1]):
                row = Ai[idx]
                J[row, col] = jac_values[idx]
    elif backend == 'pardiso':
        # CSR with 1-based indexing: Ap[i]-1 .. Ap[i+1]-2 for row i
        for row in range(N):
            for idx in range(Ap[row] - 1, Ap[row + 1] - 1):
                col = Ai[idx] - 1
                J[row, col] = jac_values[idx]
    else:
        raise ValueError(f"Unknown sparse backend: {backend}")

    return J


# ---------------------------------------------------------------------------
# Main diagnostic entry point
# ---------------------------------------------------------------------------
def diagnose_dae_model(jac_flat, fg, Nx, Ny,
                       x_names=None, y_names=None,
                       sparse_backend=None, Ap=None, Ai=None,
                       save_figure='jacobian_diagnostic.png'):
    """
    Perform a numerical health check on the DAE Jacobian and residuals.

    Parameters
    ----------
    jac_flat : ndarray
        Flat Jacobian array from the C solver.
        Dense: length N², Sparse: length NNZ.
    fg : ndarray
        Residual vector [f; g], length N.
    Nx, Ny : int
        Number of differential / algebraic variables.
    x_names, y_names : list of str, optional
        Human-readable variable names.
    sparse_backend : str or None
        ``None`` for dense, or ``'klu'`` / ``'pardiso'`` / ``'accelerate'``.
    Ap, Ai : array-like or None
        Sparsity structure arrays (required when ``sparse_backend`` is set).
    save_figure : str or None
        Path to save the heatmap image. ``None`` to skip.
    """
    N = Nx + Ny

    # ------------------------------------------------------------------
    # 0. Reconstruct dense Jacobian
    # ------------------------------------------------------------------
    if sparse_backend is not None:
        if Ap is None or Ai is None:
            raise ValueError("Ap and Ai must be provided for sparse diagnostics")
        J = _sparse_to_dense(jac_flat, Ap, Ai, N, sparse_backend)
    else:
        J = jac_flat.reshape((N, N))

    # Variable names
    if x_names is None: x_names = [f"x[{i}]" for i in range(Nx)]
    if y_names is None: y_names = [f"y[{i}]" for i in range(Ny)]
    var_names = list(x_names) + list(y_names)

    issues = []  # Collect verdicts for the summary

    print(f"\n{'='*60}")
    print(f"{'DAE SOLVER DIAGNOSTIC REPORT':^60}")
    print(f"{'='*60}")
    print(f"  System size: {Nx} differential + {Ny} algebraic = {N} total")
    if sparse_backend:
        print(f"  Sparse backend: {sparse_backend}  (NNZ = {len(jac_flat)})")
    else:
        print(f"  Dense Jacobian ({N}×{N})")

    # ------------------------------------------------------------------
    # 1. Residual analysis
    # ------------------------------------------------------------------
    print(f"\n{_BOLD}[1] RESIDUAL MAGNITUDES{_RST}")
    has_nan = False
    has_large = False
    for i in range(N):
        val = fg[i]
        if np.isnan(val):
            status = f"{_RED}NaN{_RST}"
            has_nan = True
        elif abs(val) > 1e3:
            status = f"{_RED}VERY LARGE{_RST}"
            has_large = True
        elif abs(val) > 1.0:
            status = f"{_YEL}LARGE{_RST}"
            has_large = True
        else:
            status = f"{_GRN}OK{_RST}"
        eq_type = "f" if i < Nx else "g"
        eq_idx = i if i < Nx else i - Nx
        print(f"    {eq_type}[{eq_idx}] ({var_names[i]:>12}): {val:12.4e}  [{status}]")

    if has_nan:
        issues.append(_crit("NaN in residuals — likely division by zero at initial guess."))
    elif has_large:
        issues.append(_warn("Large residuals — initial guess may be far from solution."))

    # ------------------------------------------------------------------
    # 2. Zero row / zero column detection
    # ------------------------------------------------------------------
    print(f"\n{_BOLD}[2] STRUCTURAL CHECKS{_RST}")
    row_norms = np.linalg.norm(J, axis=1)
    col_norms = np.linalg.norm(J, axis=0)

    zero_rows = np.where(row_norms < 1e-15)[0]
    zero_cols = np.where(col_norms < 1e-15)[0]

    if len(zero_rows) > 0:
        for r in zero_rows:
            print(f"    {_RED}Zero row {r}{_RST} → equation for {var_names[r]} has no dependence on any variable")
        issues.append(_crit(f"Zero rows: {[var_names[r] for r in zero_rows]} — disconnected equations."))
    else:
        print(f"    All rows nonzero  {_GRN}✓{_RST}")

    if len(zero_cols) > 0:
        for c in zero_cols:
            print(f"    {_RED}Zero col {c}{_RST} → variable {var_names[c]} does not appear in any equation")
        issues.append(_crit(f"Zero columns: {[var_names[c] for c in zero_cols]} — disconnected variables."))
    else:
        print(f"    All columns nonzero  {_GRN}✓{_RST}")

    # ------------------------------------------------------------------
    # 3. Row scaling (equation balancing)
    # ------------------------------------------------------------------
    print(f"\n{_BOLD}[3] ROW SCALING{_RST}")
    nonzero_rows = row_norms[row_norms > 0]
    if len(nonzero_rows) > 0:
        max_r = np.max(nonzero_rows)
        min_r = np.min(nonzero_rows)
        ratio = max_r / (min_r + 1e-30)
        print(f"    Max row norm: {max_r:.2e}")
        print(f"    Min row norm: {min_r:.2e}")
        print(f"    Ratio:        {ratio:.2e}")
        if ratio > 1e9:
            issues.append(_crit(f"Row scaling ratio {ratio:.0e} — equations have wildly different magnitudes."))
        elif ratio > 1e6:
            issues.append(_warn(f"Row scaling ratio {ratio:.0e} — consider scaling equations."))
        else:
            print(f"    {_GRN}Scaling looks reasonable.{_RST}")

    # ------------------------------------------------------------------
    # 4. Diagonal analysis (near-zero pivots)
    # ------------------------------------------------------------------
    print(f"\n{_BOLD}[4] DIAGONAL ANALYSIS{_RST}")
    diag = np.abs(np.diag(J))
    small_pivots = np.where(diag < 1e-12)[0]
    if len(small_pivots) > 0:
        for idx in small_pivots:
            print(f"    {_YEL}Near-zero diagonal [{idx}]{_RST}: {var_names[idx]}  (|J[{idx},{idx}]| = {diag[idx]:.2e})")
        issues.append(_warn(f"Near-zero diagonals at: {[var_names[i] for i in small_pivots]}"))
    else:
        print(f"    All diagonal entries nonzero  {_GRN}✓{_RST}")

    # Diagonal dominance (informational)
    dd_count = 0
    for i in range(N):
        off_diag_sum = np.sum(np.abs(J[i, :])) - np.abs(J[i, i])
        if np.abs(J[i, i]) >= off_diag_sum:
            dd_count += 1
    print(f"    Diagonally dominant rows: {dd_count}/{N}")

    # ------------------------------------------------------------------
    # 5. Condition number
    # ------------------------------------------------------------------
    print(f"\n{_BOLD}[5] CONDITION NUMBER{_RST}")
    try:
        c_num = np.linalg.cond(J)
        print(f"    cond(J) = {c_num:.2e}")
        if c_num > 1e14:
            issues.append(_crit(f"Condition number {c_num:.0e} — effectively singular."))
        elif c_num > 1e8:
            issues.append(_warn(f"Condition number {c_num:.0e} — ill-conditioned."))
        else:
            print(f"    {_GRN}Well-conditioned.{_RST}")
    except Exception:
        c_num = np.inf
        issues.append(_crit("Condition number computation failed — matrix may be singular."))

    # ------------------------------------------------------------------
    # 6. SVD — identify the broken variable combination
    # ------------------------------------------------------------------
    print(f"\n{_BOLD}[6] SINGULAR VALUE DECOMPOSITION{_RST}")
    try:
        U, S, Vh = np.linalg.svd(J)
        print(f"    Largest  σ: {S[0]:.2e}")
        print(f"    Smallest σ: {S[-1]:.2e}")

        # Report all near-zero singular values, not just the smallest
        near_zero = np.where(S < 1e-10)[0]
        if len(near_zero) > 0:
            print(f"    {_RED}{len(near_zero)} near-zero singular value(s) detected{_RST}")
            for sv_idx in near_zero[:5]:  # Show up to 5
                problem_vec = Vh[sv_idx, :]
                top3 = np.argsort(np.abs(problem_vec))[::-1][:3]
                contributors = ", ".join(
                    f"{var_names[k]} ({problem_vec[k]:+.4f})" for k in top3
                )
                print(f"      σ[{sv_idx}] = {S[sv_idx]:.2e}  →  {contributors}")
            issues.append(_crit(f"SVD found {len(near_zero)} near-zero singular value(s)."))
        else:
            print(f"    {_GRN}No near-zero singular values.{_RST}")

            # Still report the smallest for informational purposes
            min_idx = np.argmin(S)
            problem_vec = Vh[min_idx, :]
            top3 = np.argsort(np.abs(problem_vec))[::-1][:3]
            contributors = ", ".join(
                f"{var_names[k]} ({problem_vec[k]:+.4f})" for k in top3
            )
            print(f"    Weakest direction: σ[{min_idx}] = {S[min_idx]:.2e}  →  {contributors}")
    except Exception as e:
        issues.append(_crit(f"SVD computation failed: {e}"))

    # ------------------------------------------------------------------
    # 7. Newton convergence estimate
    # ------------------------------------------------------------------
    print(f"\n{_BOLD}[7] NEWTON CONVERGENCE ESTIMATE{_RST}")
    res_norm = np.linalg.norm(fg)
    print(f"    ||residual|| = {res_norm:.2e}")
    if c_num < np.inf and res_norm > 0:
        # Estimated correction step size: ||Δxy|| ≈ cond(J) × ||fg|| / ||J||
        J_norm = np.linalg.norm(J)
        est_step = res_norm / (J_norm + 1e-30)
        print(f"    Estimated ||Δxy|| ≈ {est_step:.2e}")
        if est_step > 1e6:
            issues.append(_warn(f"Newton step estimate {est_step:.0e} — far from convergence."))

    # ------------------------------------------------------------------
    # 8. Sparsity statistics (sparse mode only)
    # ------------------------------------------------------------------
    if sparse_backend is not None:
        nnz = len(jac_flat)
        density = nnz / (N * N) * 100
        print(f"\n{_BOLD}[8] SPARSITY STATISTICS{_RST}")
        print(f"    NNZ = {nnz} / {N*N}  ({density:.1f}% fill)")
        if density > 50:
            print(f"    {_YEL}High fill — dense solver may be faster for this system size.{_RST}")

    # ------------------------------------------------------------------
    # 9. Heatmap
    # ------------------------------------------------------------------
    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7), gridspec_kw={'width_ratios': [3, 1]})

        # Jacobian heatmap
        ax1 = axes[0]
        log_J = np.log10(np.abs(J) + 1e-15)
        im = ax1.imshow(log_J, cmap='viridis', interpolation='nearest', aspect='auto')
        fig.colorbar(im, ax=ax1, label='|J| (log₁₀)', shrink=0.8)
        ax1.set_xticks(range(N)); ax1.set_xticklabels(var_names, rotation=90, fontsize=8)
        ax1.set_yticks(range(N)); ax1.set_yticklabels(var_names, fontsize=8)
        ax1.set_title('Jacobian Structure & Magnitude')

        # Residual bar chart
        ax2 = axes[1]
        colors = []
        for i in range(N):
            if np.isnan(fg[i]):
                colors.append('red')
            elif abs(fg[i]) > 1.0:
                colors.append('orange')
            else:
                colors.append('green')
        fg_plot = np.where(np.isnan(fg), 0, fg)
        ax2.barh(range(N), np.abs(fg_plot) + 1e-15, color=colors, log=True)
        ax2.set_yticks(range(N)); ax2.set_yticklabels(var_names, fontsize=8)
        ax2.set_xlabel('|residual| (log scale)')
        ax2.set_title('Residual Magnitudes')
        ax2.invert_yaxis()

        fig.tight_layout()
        if save_figure:
            fig.savefig(save_figure, dpi=150, bbox_inches='tight')
            print(f"\n    Diagnostic figure saved to: {save_figure}")
        plt.show()

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"{'SUMMARY':^60}")
    print(f"{'='*60}")
    if len(issues) == 0:
        print(f"  {_GRN}No issues detected. System looks healthy.{_RST}")
    else:
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

    print(f"\n{'SUGGESTED ACTIONS':^60}")
    print(f"{'-'*60}")
    if has_nan:
        print("  → Check initial guesses: ensure no variable in a denominator is 0.")
        print("  → Inspect the generated C code for division by zero patterns.")
    if len(zero_rows) > 0 or len(zero_cols) > 0:
        print("  → Check for redundant conservation laws or disconnected components.")
        print("  → Verify that every algebraic variable appears in at least one equation.")
    if c_num > 1e8 and not has_nan:
        print("  → Scale equations to similar magnitudes (e.g., per-unit system).")
        print("  → Try a better initial guess closer to the expected steady state.")
    if len(issues) == 0:
        print("  → If convergence still fails, try increasing max_it or relaxing itol.")
    print(f"{'='*60}\n")

    return J  # Return the dense Jacobian for further inspection if needed


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    Nx, Ny = 4, 2
    N = Nx + Ny

    jac_ini = np.array([
        -1.91938577e-01,  3.34996016e-05,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00, -9.47507845e-03,  1.04200002e+01,
         0.00000000e+00,  0.00000000e+00, -1.00000000e-06,  0.00000000e+00,
        -0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  0.00000000e+00, -0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        -0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -3.33333333e-05,
        -3.47333328e-01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        -3.33333333e-05,  0.00000000e+00,  1.74532927e-04,  3.33333333e-02
    ])

    fg = np.array([
        -0.00000000e+00, -0.00000000e+00, 8.90765703e-10,
         5.10371148e-06, -7.34345917e-12, -0.00000000e+00
    ])

    # Test dense mode
    print("=== DENSE MODE TEST ===")
    diagnose_dae_model(jac_ini, fg, Nx, Ny,
                       save_figure='diag_dense.png')

    # Test sparse mode (simulate CSC from the same matrix)
    print("\n=== SPARSE (KLU) MODE TEST ===")
    J_full = jac_ini.reshape((N, N))
    from scipy.sparse import csc_matrix
    J_csc = csc_matrix(J_full)
    diagnose_dae_model(
        J_csc.data, fg, Nx, Ny,
        sparse_backend='klu',
        Ap=J_csc.indptr.tolist(),
        Ai=J_csc.indices.tolist(),
        save_figure='diag_sparse.png'
    )
