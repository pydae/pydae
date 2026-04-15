import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import cond, svd, norm

def diagnose_dae_model(jac_flat, fg, Nx, Ny, x_names=None, y_names=None):
    """
    Performs a numerical health check on the DAE Jacobian and Residuals.
    
    This function analyzes the conditioning, scaling, and singularity of the 
    system to help identify which specific equations or variables are 
    causing solver failure.

    How to interpret the results:
    ----------------------------
    [1] Condition Number > $10^{14}$:
        Your equations are mathematically redundant. Check for redundant 
        conservation laws (e.g., KCL at both ends of an ideal wire).
        
    [2] Residual is NaN:
        Check 'temp_ctypes.c' for potential division by zero. This typically 
        occurs if a variable (like "Voltage" or "Speed") is in a denominator 
        and the initial guess is 0.0.
        
    [3] Max/Min scaling ratio is high (> $10^9$):
        The system needs pre-conditioning or scaling. If one equation is in 
        Volts ($10^5$) and another in Nanofarads ($10^{-9}$), multiply the 
        smaller equation by $10^9$ to align their numerical magnitudes.
        
    [4] Singular Value Analysis (SVD):
        The "Smoking Gun." If SVD points to a specific variable (e.g., y[1]), 
        the associated equation likely has no derivative or the variable 
        has "disconnected" from the system logic.
        
    Heatmap Visualization:
        - Dark purple row (near zero): The equation is not contributing to 
          the solution.
        - Dark purple column (near zero): The variable has no effect on any 
          equation, rendering the matrix singular.

    Parameters:
    -----------
    jac_flat : ndarray
        The flat Jacobian array returned from the C solver.
    fg : ndarray
        The residual vector (f and g).
    Nx : int
        Number of differential variables.
    Ny : int
        Number of algebraic variables.
    x_names : list, optional
        Names for differential variables for better reporting.
    y_names : list, optional
        Names for algebraic variables for better reporting.
    """

    N = Nx + Ny
    J = jac_flat.reshape((N, N))
    
    # Generate default names if none provided
    if x_names is None: x_names = [f"x[{i}]" for i in range(Nx)]
    if y_names is None: y_names = [f"y[{i}]" for i in range(Ny)]
    var_names = x_names + y_names
    
    print("\n" + "="*50)
    print("       DAE SOLVER DIAGNOSTIC REPORT")
    print("="*50)


    # 2. Residual Analysis (Equation Health)
    print("\n[2] RESIDUAL MAGNITUDES (f and g):")
    for i in range(N):
        status = "OK" if abs(fg[i]) < 1.0 else "LARGE"
        if np.isnan(fg[i]): status = "NaN"
        print(f"    Eq {i:2d} ({var_names[i]:>10}): {fg[i]:12.4e} [{status}]")

    # 3. Row Scaling Analysis (Equation Balancing)
    row_norms = norm(J, axis=1)
    print("\n[3] ROW SCALING (Equation Coefficients):")
    max_r = np.max(row_norms)
    min_r = np.min(row_norms[row_norms > 0]) if any(row_norms > 0) else 0
    print(f"    Max Row Norm: {max_r:.2e}")
    print(f"    Min Row Norm: {min_r:.2e}")
    if max_r / (min_r + 1e-15) > 1e6:
        print("    WARNING: Vastly different scales detected between equations.")

    # 1. Condition Number
    c_num = cond(J)
    print(f"\n[1] CONDITION NUMBER: {c_num:.2e}")
    if c_num > 1e12:
        print("    CRITICAL: Matrix is ill-conditioned. Results are likely noise.")
    elif c_num > 1e7:
        print("    WARNING: Matrix is moderately stiff.")


    # 4. SVD - Identifying the "Broken" Variable Combination
    U, S, Vh = svd(J)
    min_sv_idx = np.argmin(S)
    print(f"\n[4] SINGULAR VALUE ANALYSIS:")
    print(f"    Smallest Singular Value: {S[min_sv_idx]:.2e}")
    
    # The right-singular vector (Vh) corresponding to the smallest S
    # tells us which variables are involved in the redundancy/singularity.
    problem_vec = Vh[min_sv_idx, :]
    top_contributors = np.argsort(np.abs(problem_vec))[::-1][:3]
    
    print("    Variables most involved in the singularity:")
    for idx in top_contributors:
        weight = problem_vec[idx]
        print(f"    -> {var_names[idx]} (Weight: {weight:.4f})")

    # 5. Visual Sparsity and Heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(np.log10(np.abs(J) + 1e-15), cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Magnitude (log10)')
    plt.title("Jacobian Heatmap (log scale)")
    plt.xticks(range(N), var_names, rotation=90)
    plt.yticks(range(N), var_names)
    plt.tight_layout()
    plt.show()

    print("="*50 + "\n")

# --- EXAMPLE INTEGRATION ---
# res_ini = solver_lib.ini(...)
# if res_ini != 0 or np.isnan(xy).any():
#     diagnose_dae_model(jac, fg_work, N_x, N_y)


if __name__ == '__main__':
    N_x = 4
    N_y = 2
    N_xy = N_x + N_y

    jac_ini = np.array([-1.91938577e-01,  3.34996016e-05,  0.00000000e+00,  0.00000000e+00,
                        0.00000000e+00,  0.00000000e+00, -9.47507845e-03,  1.04200002e+01,
                        0.00000000e+00,  0.00000000e+00, -1.00000000e-06,  0.00000000e+00,
                        -0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  0.00000000e+00,
                        0.00000000e+00,  0.00000000e+00, -0.00000000e+00,  0.00000000e+00,
                        0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                        -0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -3.33333333e-05,
                        -3.47333328e-01,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
                        -3.33333333e-05,  0.00000000e+00,  1.74532927e-04,  3.33333333e-02]).reshape(N_xy,N_xy)
    
             
    fg = np.array([-0.00000000e+00, -0.00000000e+00,  8.90765703e-10,  5.10371148e-06, -7.34345917e-12, -0.00000000e+00])

  
    diagnose_dae_model(jac_ini.flatten(), fg, N_x, N_y, x_names=None, y_names=None)