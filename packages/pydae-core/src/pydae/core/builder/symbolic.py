# pydae/core/builder/symbolic.py
import logging
import sympy as sym
from sympy.matrices.sparsetools import _doktocsr
from sympy import SparseMatrix

def sym_jac(f, x):
    """
    Computes the symbolic Jacobian of vector 'f' with respect to vector 'x'.
    """
    N_f = len(f)
    N_x = len(x)
    J = sym.MutableSparseMatrix(N_f, N_x, {})
    
    for irow in range(N_f):
        str_f = str(f[irow])
        for icol in range(N_x):
            if str(x[icol]) in str_f:
                J[irow, icol] = f[irow].diff(x[icol])
    return J


def compute_base_jacobians(sys, uz_jacs=True):
    """
    Computes the fundamental sub-Jacobians (Fx, Fy, Gx, Gy, etc.).
    """
    logging.info('Computing base symbolic Jacobians...')
    
    sys['Fx'] = sym_jac(sys['f'], sys['x'])
    sys['Fy_ini'] = sym_jac(sys['f'], sys['y_ini'])
    sys['Fy_run'] = sym_jac(sys['f'], sys['y_run'])
    sys['Gx'] = sym_jac(sys['g'], sys['x'])
    sys['Gy_ini'] = sym_jac(sys['g'], sys['y_ini'])
    sys['Gy_run'] = sym_jac(sys['g'], sys['y_run'])

    if uz_jacs:
        sys['Fu_ini'] = sym_jac(sys['f'], sys['u_ini'])
        sys['Fu_run'] = sym_jac(sys['f'], sys['u_run'])
        sys['Gu_ini'] = sym_jac(sys['g'], sys['u_ini'])
        sys['Gu_run'] = sym_jac(sys['g'], sys['u_run'])
        
        if 'h' in sys:
            sys['Hx'] = sym_jac(sys['h'], sys['x'])
            sys['Hy_ini'] = sym_jac(sys['h'], sys['y_ini'])
            sys['Hy_run'] = sym_jac(sys['h'], sys['y_run'])
            sys['Hu_ini'] = sym_jac(sys['h'], sys['u_ini'])
            sys['Hu_run'] = sym_jac(sys['h'], sys['u_run'])
            
    return sys


def build_large_jacobians(builder_obj):
    """
    Assembles the large global Jacobians (ini and trap) and extracts
    nonzero elements into lists for C code generation.
    
    Each Jacobian gets its OWN independent sparsity pattern (Ap, Ai).
    This avoids the bug where y_ini_list != y_run_list causes entries
    to land in wrong positions when a single shared pattern is used.
    """
    logging.info('Building large global Jacobians...')
    sys = builder_obj.sys
    is_sparse = getattr(builder_obj, 'sparse', False)
    
    N_x = len(sys['x'])
    N_y = len(sys['y_ini'])
    N_total = N_x + N_y

    # --- 1. Initialization Jacobian ---
    ini_top = sys['Fx'].row_join(sys['Fy_ini'])
    ini_bot = sys['Gx'].row_join(sys['Gy_ini'])
    jac_ini = ini_top.col_join(ini_bot)
    
    # --- 2. Trapezoidal Run Jacobian ---
    Dt = sym.Symbol('Dt', real=True)
    alpha = sym.Symbol('alpha_solver', real=True)
    eye_sparse = sym.eye(N_x)
    
    trap_top = (eye_sparse - alpha * Dt * sys['Fx']).row_join(-alpha * Dt * sys['Fy_run'])
    trap_bot = sys['Gx'].row_join(sys['Gy_run'])
    jac_trap = trap_top.col_join(trap_bot)

    sys['jac_ini'] = jac_ini
    sys['jac_trap'] = jac_trap

    logging.info(f'Converting Jacobians to target format (Sparse: {is_sparse})')

    if is_sparse:
        builder_obj.jac_ini_sp  = _process_sparse(jac_ini, builder_obj.jac_ini_list, N_total, 'ini')
        builder_obj.jac_trap_sp = _process_sparse(jac_trap, builder_obj.jac_trap_list, N_total, 'trap')
    else:
        builder_obj.jac_ini_sp  = _extract_dense_matrix(jac_ini, builder_obj.jac_ini_list, N_total)
        builder_obj.jac_trap_sp = _extract_dense_matrix(jac_trap, builder_obj.jac_trap_list, N_total)

    # ------------------------------------------------------------------
    # UZ Jacobians (for Small Signal Analysis)
    # ------------------------------------------------------------------
    if getattr(builder_obj, 'uz_jacs', False):
        logging.info('Extracting UZ Jacobians...')
        
        # Dimensions
        N_x = len(sys['x'])
        N_y_ini = len(sys['y_ini'])
        N_y_run = len(sys['y_run'])
        N_u_ini = len(sys['u_ini'])
        N_u_run = len(sys['u_run'])

        # Helper to call extraction
        def extract_if_exists(sym_key, target_list, stride):
            if sym_key in sys:
                _extract_dense_matrix(sys[sym_key], target_list, stride)
            else:
                logging.warning(f'Symbolic key {sym_key} not found, skipping.')

        extract_if_exists('Fu_ini', builder_obj.Fu_ini_list, N_u_ini)
        extract_if_exists('Fu_run', builder_obj.Fu_run_list, N_u_run)
        extract_if_exists('Gu_ini', builder_obj.Gu_ini_list, N_u_ini)
        extract_if_exists('Gu_run', builder_obj.Gu_run_list, N_u_run)
        extract_if_exists('Hx', builder_obj.Hx_list, N_x)
        extract_if_exists('Hy_ini', builder_obj.Hy_ini_list, N_y_ini)
        extract_if_exists('Hy_run', builder_obj.Hy_run_list, N_y_run)
        extract_if_exists('Hu_ini', builder_obj.Hu_ini_list, N_u_ini)
        extract_if_exists('Hu_run', builder_obj.Hu_run_list, N_u_run)


# ---------------------------------------------------------------------------
# Dense extraction
# ---------------------------------------------------------------------------
def _extract_dense_matrix(matrix, target_list, stride):
    """Extract nonzero entries with flat dense indexing (row*stride+col)."""
    dok = dict(SparseMatrix(matrix).todok())
    
    for (row, col), expr in sorted(dok.items()):
        target_list.append({
            'sym': expr,
            'de_idx': row * stride + col,
            'ij': (row, col),
        })
    
    # Return CSR tuple for compatibility (not used in dense mode)
    if dok:
        sp = _doktocsr(SparseMatrix(matrix))
    else:
        sp = ([], [], [0] * (matrix.rows + 1))
    return sp


# ---------------------------------------------------------------------------
# Sparse extraction — independent pattern per Jacobian
# ---------------------------------------------------------------------------
def _process_sparse(matrix, target_list, N_total, label):
    """
    Build a CSC sparsity pattern from a single Jacobian matrix and map
    its symbolic entries into packed sp_idx positions.
    
    Returns (data_placeholder, Ai, Ap) in the same tuple format the
    builders expect from [:3] slicing.
    """
    dok = dict(SparseMatrix(matrix).todok())
    positions = sorted(dok.keys())
    
    logging.info(f'  jac_{label}: {len(positions)} nonzeros')
    
    # Build CSC: Ap (column pointers), Ai (row indices)
    Ap = [0] * (N_total + 1)
    Ai = []
    
    col_to_rows = {}
    for (row, col) in positions:
        col_to_rows.setdefault(col, []).append(row)
    
    idx = 0
    for col in range(N_total):
        rows = sorted(col_to_rows.get(col, []))
        Ai.extend(rows)
        idx += len(rows)
        Ap[col + 1] = idx
    
    # Lookup: (row, col) -> packed index
    pos_to_sp_idx = {}
    for col in range(N_total):
        for sp_idx in range(Ap[col], Ap[col + 1]):
            row = Ai[sp_idx]
            pos_to_sp_idx[(row, col)] = sp_idx
    
    # Map entries
    for (row, col), expr in sorted(dok.items()):
        sp_idx = pos_to_sp_idx[(row, col)]
        target_list.append({
            'sym': expr,
            'sp_idx': sp_idx,
            'de_idx': row * N_total + col,
            'ij': (row, col),
        })
    
    # Sort by sp_idx so C function fills Ax[] in order
    target_list.sort(key=lambda x: x['sp_idx'])
    
    return ([], Ai, Ap)
