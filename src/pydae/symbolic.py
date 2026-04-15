# symbolic.py
import logging
import numpy as np
import sympy as sym
from sympy.matrices.sparsetools import _doktocsr
from sympy import SparseMatrix

def sym_jac(f, x):
    """
    Computes the symbolic Jacobian of vector 'f' with respect to vector 'x'.
    It optimizes the calculation by checking if the string representation of 
    the variable exists in the equation string before applying .diff().
    
    Parameters
    ----------
    f : sym.Matrix
        Column vector of symbolic functions (equations).
    x : sym.Matrix
        Column vector of symbolic variables (states/algebraic variables).
        
    Returns
    -------
    J : sym.MutableSparseMatrix
        The analytical Jacobian matrix.
    """
    # Get the number of equations and variables
    N_f = len(f)
    N_x = len(x)

    # Initialize an empty sparse matrix
    J = sym.MutableSparseMatrix(N_f, N_x, {})
    
    for irow in range(N_f):
        # Convert the current equation to a string for fast substring matching
        str_f = str(f[irow])
        for icol in range(N_x):
            # String matching optimization: 
            # Only compute the derivative if the variable name appears in the equation
            if str(x[icol]) in str_f:
                J[irow, icol] = f[irow].diff(x[icol])
                
    return J

def spidx2ij(csr_matrix_list):
    """
    Converts SymPy CSR (Compressed Sparse Row) matrix data into (i, j) 
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
    # Extract CSR structure components
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
            ij_indices += [(i, j)]
            dense_indices += [i * N_cols + j]
            
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
    logging.debug('Computing jacobians Fx_run, Fy_run')
    sys['Fx'] = sym_jac(f, x)
    sys['Fy_run'] = sym_jac(f, y_run)
    
    # 2. Compute algebraic equations Jacobians (run stage)
    logging.debug('Computing jacobians Gx_run, Gy_run')
    sys['Gx'] = sym_jac(g, x)
    sys['Gy_run'] = sym_jac(g, y_run)
    
    # 3. Compute input-related Jacobians (run stage)
    logging.debug('Computing jacobians Fu_run, Gu_run')
    sys['Fu_run'] = sym_jac(f, u_run)
    sys['Gu_run'] = sym_jac(g, u_run)

    # 4. Compute Jacobians for the initialization stage if required
    if inirun: 
        logging.debug('Computing jacobians Fx_ini, Fy_ini, Gy_ini')
        sys['Fy_ini'] = sym_jac(f, y_ini)
        sys['Gy_ini'] = sym_jac(g, y_ini) 
    else:
        # If no separate init stage, use the run Jacobians
        sys['Fy_ini'] = sys['Fy_run']
        sys['Gy_ini'] = sys['Gy_run']

    # 5. Compute output equations Jacobians
    logging.debug('Computing jacobians Hx_run, Hy_run, Hu_run')
    sys['Hx_run'] = sym_jac(h, x)
    sys['Hy_run'] = sym_jac(h, y_run)
    sys['Hu_run'] = sym_jac(h, u_run)

    return sys

def build_large_jacobians(builder_obj):
    """
    Constructs the full system Jacobians (jac_ini, jac_run, jac_trap)
    from the previously computed sub-matrices. Converts them to sparse 
    format and populates the builder's element lists for code generation.
    """
    sys = builder_obj.sys
    N_x = sys['N_x']
    
    # 1. Assemble the full initialization Jacobian matrix
    logging.debug('Computing full jac_ini matrix')
    jac_ini = sym.Matrix([[sys['Fx'], sys['Fy_ini']], 
                          [sys['Gx'], sys['Gy_ini']]]) 
    
    # 2. Assemble the full run Jacobian matrix
    logging.debug('Computing full jac_run matrix')
    jac_run = sym.Matrix([[sys['Fx'], sys['Fy_run']], 
                          [sys['Gx'], sys['Gy_run']]]) 
    
    # 3. Assemble the implicit trapezoidal integration Jacobian matrix
    logging.debug('Computing full jac_trap matrix')
    eye = sym.eye(N_x, real=True)
    Dt = sym.Symbol('Dt', real=True)
    jac_trap = sym.Matrix([[eye - 0.5 * Dt * sys['Fx'], -0.5 * Dt * sys['Fy_run']], 
                           [sys['Gx'], sys['Gy_run']]])    

    # Store full dense matrices in the system dictionary
    sys['jac_ini'] = jac_ini
    sys['jac_run'] = jac_run   
    sys['jac_trap'] = jac_trap
    
    logging.debug('End of large jacobians computation')

    # --- Sparse Processing: Initialization Jacobian ---
    # Convert to DOK (Dictionary of Keys) and then to CSR format
    builder_obj.jac_ini_sp = _doktocsr(SparseMatrix(jac_ini))
    sys['sp_jac_ini_list'] = builder_obj.jac_ini_sp
    
    # Extract coordinates and 1D indices for the non-zero elements
    ij_indices, dense_indices = spidx2ij(builder_obj.jac_ini_sp)
    
    # Store elements in a list for the code generator (C/Go/Julia)
    for it, item in enumerate(builder_obj.jac_ini_sp[0]):
        builder_obj.jac_ini_list.append({'sym': item, 'ij': ij_indices[it], 'de_idx': dense_indices[it]})

    # --- Sparse Processing: Run Jacobian ---
    builder_obj.jac_run_sp = _doktocsr(SparseMatrix(jac_run))
    sys['sp_jac_run_list'] = builder_obj.jac_run_sp
    ij_indices, dense_indices = spidx2ij(builder_obj.jac_run_sp)
    
    for it, item in enumerate(builder_obj.jac_run_sp[0]):
        builder_obj.jac_run_list.append({'sym': item, 'ij': ij_indices[it], 'de_idx': dense_indices[it]})

    # --- Sparse Processing: Trapezoidal Jacobian ---
    builder_obj.jac_trap_sp = _doktocsr(SparseMatrix(jac_trap))
    sys['sp_jac_trap_list'] = builder_obj.jac_trap_sp
    ij_indices, dense_indices = spidx2ij(builder_obj.jac_trap_sp)
    
    for it, item in enumerate(builder_obj.jac_trap_sp[0]):
        builder_obj.jac_trap_list.append({'sym': item, 'ij': ij_indices[it], 'de_idx': dense_indices[it]})
        
    # Process additional sparse Jacobians if the 'uz_jacs' flag is enabled
    if builder_obj.uz_jacs:
        _process_uz_jacobians(builder_obj)

def _process_uz_jacobians(builder_obj):
    """
    Helper method to process and store additional sparse Jacobians 
    related to inputs (u) and outputs (z).
    """
    sys = builder_obj.sys
    
    # Process Jacobian of dynamic equations with respect to inputs (Fu)
    builder_obj.Fu_run_sp = _doktocsr(SparseMatrix(sys['Fu_run']))
    for item in builder_obj.Fu_run_sp[0]:
        builder_obj.Fu_list.append({'sym': item})
        
    # Process Jacobian of algebraic equations with respect to inputs (Gu)
    builder_obj.Gu_run_sp = _doktocsr(SparseMatrix(sys['Gu_run']))
    for item in builder_obj.Gu_run_sp[0]:
        builder_obj.Gu_list.append({'sym': item})
        
    # Process Jacobian of output equations with respect to dynamic states (Hx)
    builder_obj.Hx_run_sp = _doktocsr(SparseMatrix(sys['Hx_run']))
    for item in builder_obj.Hx_run_sp[0]:
        builder_obj.Hx_list.append({'sym': item})
        
    # (Note: Additional processing for Hy and Hu can be added here following the same pattern)