# symbolic.py
import logging
import numpy as np
import sympy as sym
from sympy.matrices.sparsetools import _doktocsr
from sympy import SparseMatrix

def sym_jac(f, x):
    """
    Computes the symbolic Jacobian of vector 'f' with respect to vector 'x'.
    It includes an optimization that checks if the string representation of 
    the variable exists in the equation string before applying .diff(), 
    which drastically reduces computation time for large sparse systems.
    
    Parameters
    ----------
    f : sym.Matrix
        Vector of symbolic functions (equations).
    x : sym.Matrix
        Vector of symbolic variables (states/algebraic variables).
        
    Returns
    -------
    J : sym.MutableSparseMatrix
        The analytical Jacobian matrix.
    """
    N_f = len(f)
    N_x = len(x)

    # Initialize an empty sparse matrix
    J = sym.MutableSparseMatrix(N_f, N_x, {})
    
    for irow in range(N_f):
        # Convert the current equation to a string for fast substring matching
        str_f = str(f[irow])
        for icol in range(N_x):
            # Optimization: Only compute the derivative if the variable name 
            # actually appears in the equation string.
            if str(x[icol]) in str_f:
                J[irow, icol] = f[irow].diff(x[icol])
                
    return J

def spidx2ij(csr_matrix_list):
    """
    Converts SymPy CSR (Compressed Sparse Row) matrix data into (row, col) 
    tuple indices and flattened 1D array indices.
    
    Parameters
    ----------
    csr_matrix_list : tuple/list
        A structure containing [data, indices, indptr, shape] from a CSR matrix.
        
    Returns
    -------
    ij_indices : list of tuples
        List containing (row, col) coordinates for non-zero elements.
    dense_indices : list of ints
        List containing the flattened 1D array index for non-zero elements.
    """
    indices = csr_matrix_list[1]
    indptr = csr_matrix_list[2]
    shape = csr_matrix_list[3]
    
    N_rows, N_cols = shape
    ij_indices = []
    dense_indices = []
    
    # Iterate through the rows to map non-zero elements
    for i in range(shape[0]):
        row_col_start = indptr[i]
        row_col_end = indptr[i+1]
        
        # Extract the column index 'j' for each non-zero element in row 'i'
        for j in indices[row_col_start:row_col_end]:
            ij_indices.append((i, j))
            dense_indices.append(i * N_cols + j)
            
    return ij_indices, dense_indices

def compute_base_jacobians(sys, inirun):
    """
    Computes the fundamental sub-Jacobian matrices (Fx, Fy, Gx, Gy, Hx, etc.) 
    for the DAE system and updates the 'sys' dictionary.
    """
    # Extract lists of symbolic equations
    f = sys['f']
    g = sys['g']
    h = sys['h']
    
    # Extract lists of symbolic variables
    x = sys['x']
    y_ini = sys['y_ini']
    y_run = sys['y_run']
    u_run = sys['u_run']

    # 1. Compute dynamic equations Jacobians (run stage)
    logging.info('Computing jacobians Fx_run, Fy_run')
    sys['Fx'] = sym_jac(f, x)
    sys['Fy_run'] = sym_jac(f, y_run)
    
    # 2. Compute algebraic equations Jacobians (run stage)
    logging.info('Computing jacobians Gx_run, Gy_run')
    sys['Gx'] = sym_jac(g, x)
    sys['Gy_run'] = sym_jac(g, y_run)
    
    # 3. Compute input-related Jacobians (run stage)
    logging.info('Computing jacobians Fu_run, Gu_run')
    sys['Fu_run'] = sym_jac(f, u_run)
    sys['Gu_run'] = sym_jac(g, u_run)

    # 4. Compute Jacobians for the initialization stage if required
    if inirun: 
        logging.info('Computing jacobians Fx_ini, Fy_ini, Gy_ini')
        sys['Fy_ini'] = sym_jac(f, y_ini)
        sys['Gy_ini'] = sym_jac(g, y_ini) 
    else:
        # If no separate init stage, use the run Jacobians
        sys['Fy_ini'] = sys['Fy_run']
        sys['Gy_ini'] = sys['Gy_run']

    # 5. Compute output equations Jacobians
    logging.info('Computing jacobians Hx_run, Hy_run, Hu_run')
    sys['Hx_run'] = sym_jac(h, x)
    sys['Hy_run'] = sym_jac(h, y_run)
    sys['Hu_run'] = sym_jac(h, u_run)

    return sys

def build_large_jacobians(builder_obj):
    """
    Constructs the full system Jacobians (jac_ini, jac_run, jac_trap) safely 
    using native SparseMatrix join operations to avoid flattening bugs.
    """
    sys = builder_obj.sys
    N_x = sys['N_x']
    
    logging.info('Assembling full jac_ini and jac_run matrices')
    
    # Unir sub-matrices usando row_join y col_join (100% seguro para SparseMatrices)
    jac_ini_top = sys['Fx'].row_join(sys['Fy_ini'])
    jac_ini_bot = sys['Gx'].row_join(sys['Gy_ini'])
    jac_ini = jac_ini_top.col_join(jac_ini_bot)
    
    jac_trap_top = sys['Fx'].row_join(sys['Fy_run'])
    jac_trap_bot = sys['Gx'].row_join(sys['Gy_run'])
    jac_trap = jac_trap_top.col_join(jac_trap_bot)
    
    logging.info('Assembling full jac_trap matrix')
    eye_sparse = sym.SparseMatrix(sym.eye(N_x, real=True))
    Dt = sym.Symbol('Dt', real=True)
    
    trap_top_left = eye_sparse - 0.5 * Dt * sys['Fx']
    trap_top_right = -0.5 * Dt * sys['Fy_run']
    jac_trap_top = trap_top_left.row_join(trap_top_right)
    jac_trap = jac_trap_top.col_join(jac_trap_bot)

    sys['jac_ini'] = jac_ini
    # sys['jac_run'] = jac_run   
    sys['jac_trap'] = jac_trap
    
    # --- Helper para procesar las coordenadas Sparse ---
    def process_sparse_matrix(matrix, target_list):
        sparse_mat = _doktocsr(SparseMatrix(matrix))
        ij_indices, dense_indices = spidx2ij(sparse_mat)
        
        for it, item in enumerate(sparse_mat[0]):
            target_list.append({
                'sym': item, 
                'ij': ij_indices[it], 
                'de_idx': dense_indices[it]
            })
        return sparse_mat

    logging.info('Converting Jacobians to sparse CSR format')
    
    builder_obj.jac_ini_sp = process_sparse_matrix(jac_ini, builder_obj.jac_ini_list)
    # builder_obj.jac_run_sp = process_sparse_matrix(jac_run, builder_obj.jac_run_list)
    builder_obj.jac_trap_sp = process_sparse_matrix(jac_trap, builder_obj.jac_trap_list)
        
    if getattr(builder_obj, 'uz_jacs', False):
        logging.info('Processing Input/Output (UZ) sparse Jacobians')
        builder_obj.Fu_run_sp = process_sparse_matrix(sys['Fu_run'], builder_obj.Fu_list)
        builder_obj.Gu_run_sp = process_sparse_matrix(sys['Gu_run'], builder_obj.Gu_list)
        builder_obj.Hx_run_sp = process_sparse_matrix(sys['Hx_run'], builder_obj.Hx_list)