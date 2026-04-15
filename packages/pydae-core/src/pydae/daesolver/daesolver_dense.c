#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "daesolver_dense.h"

/* * PORTABLE DENSE LINEAR SOLVER: LU Decomposition with Partial Pivoting
 * flag = 0: Factorize matrix A in-place into LU, populate pivots
 * flag = 1: Solve Ax = b using the existing in-place LU factorization
 */
int solve_dense(double *A, int N, int *pivots, double *b, double *x, int flag) {
    if (flag == 0) {
        for (int i = 0; i < N; i++) pivots[i] = i;
        for (int i = 0; i < N; i++) {
            double max_el = fabs(A[i * N + i]);
            int max_row = i;
            for (int k = i + 1; k < N; k++) {
                if (fabs(A[k * N + i]) > max_el) {
                    max_el = fabs(A[k * N + i]);
                    max_row = k;
                }
            }
            if (max_row != i) {
                int tmp_piv = pivots[i];
                pivots[i] = pivots[max_row];
                pivots[max_row] = tmp_piv;
                for (int k = 0; k < N; k++) {
                    double tmp_val = A[i * N + k];
                    A[i * N + k] = A[max_row * N + k];
                    A[max_row * N + k] = tmp_val;
                }
            }
            if (fabs(A[i * N + i]) < 1e-14) return -1; // Singular matrix
            for (int k = i + 1; k < N; k++) {
                A[k * N + i] /= A[i * N + i];
                for (int j = i + 1; j < N; j++) {
                    A[k * N + j] -= A[k * N + i] * A[i * N + j];
                }
            }
        }
    } else if (flag == 1) {
        for (int i = 0; i < N; i++) {
            x[i] = b[pivots[i]];
            for (int k = 0; k < i; k++) x[i] -= A[i * N + k] * x[k];
        }
        for (int i = N - 1; i >= 0; i--) {
            for (int k = i + 1; k < N; k++) x[i] -= A[i * N + k] * x[k];
            x[i] /= A[i * N + i];
        }
    }
    return 0;
}

/*
 * STEADY-STATE INITIALIZATION (Newton-Raphson)
 */
int ini(double *jac_ini, int *pivots, double *x, double *y, double *xy, double *Dxy, 
        double *u, double *p, int N_x, int N_y, int max_it, double itol, double *z, 
        double *inidblparams, int *iniintparams, double *f, double *g, double *fg) {
    
    int i, k, it;
    int N = N_x + N_y;
    double Dt = 0.0; 

    // --- DIAGNOSTIC MODE ---
    if (iniintparams[4] == 1) {
        f_ini_eval(f, x, y, u, p, Dt);
        g_ini_eval(g, x, y, u, p, Dt);

        for (i = 0; i < N_x; i++) fg[i] = -f[i];
        for (i = 0; i < N_y; i++) fg[i + N_x] = -g[i];

        for (k = 0; k < N * N; k++) jac_ini[k] = 0.0;
        jac_ini_eval(jac_ini, x, y, u, p, Dt); // Restored!

        return 0; 
    }

    // --- REGULAR NEWTON-RAPHSON LOOP ---
    for (it = 0; it < max_it; it++) {
        f_ini_eval(f, x, y, u, p, Dt);
        g_ini_eval(g, x, y, u, p, Dt);

        for (i = 0; i < N_x; i++) fg[i] = -f[i];
        for (i = 0; i < N_y; i++) fg[i + N_x] = -g[i];

        if (iniintparams[0] == 0 || (iniintparams[0] == 1 && it == 0)) {
            for (k = 0; k < N * N; k++) jac_ini[k] = 0.0;
            jac_ini_eval(jac_ini, x, y, u, p, Dt); // Restored!
            if (solve_dense(jac_ini, N, pivots, fg, Dxy, 0) != 0) return -1;
            iniintparams[3] += 1; 
        }

        solve_dense(jac_ini, N, pivots, fg, Dxy, 1);

        for (i = 0; i < N; i++) xy[i] += Dxy[i];
        for (i = 0; i < N_x; i++) x[i] = xy[i];
        for (i = 0; i < N_y; i++) y[i] = xy[i + N_x];

        double norma = 0.0;
        for (i = 0; i < N; i++) norma += fg[i] * fg[i];
        if (norma < itol) break;
    }
    
    iniintparams[2] = it;
    if (iniintparams[1] == 0) h_eval(z, x, y, u, p, Dt);
    return 0;
}



/*
 * TIME INTEGRATION (Generalized Alpha-Method with Diagnostics & Extraction)
 */
int run(double t, double t_end, double *jac_trap, int *pivots, double *x, double *y, double *xy, 
        double *u, double *p, int N_x, int N_y, int max_it, double itol, int *its, double Dt, 
        double *z, double *dblparams, int *intparams, double *Time, double *X, double *Y, 
        double *Z, int N_z, int N_store, double *f, double *g, double *fg) {
    
    int i, j, k, it;
    double norma;
    int N = N_x + N_y;

    // --- 1. SOLVER SELECTION VIA DBLPARAMS ---
    // dblparams[0] = 0.5  -> Implicit Trapezoidal (No numerical damping)
    // dblparams[0] = 1.0  -> Backward Euler (Maximum numerical damping)
    double alpha = dblparams[0]; 
    double k_jac = 2.0 * alpha; 
    
    // --- 2. DECIMATION LOGIC ---
    // intparams[6] = Decimation factor (e.g., save every 10 steps)
    // intparams[7] = Global step counter
    int decimation = intparams[6];
    if (decimation < 1) decimation = 1; // Prevent modulo by zero or negative

    // --- MODE 1: JAC_RUN EXTRACTION ---
    // If intparams[4] == 1, calculate jac_run from jac_trap and exit immediately
    if (intparams[4] == 1) {
        for (k = 0; k < N * N; k++) jac_trap[k] = 0.0;
        jac_trap_eval(jac_trap, x, y, u, p, Dt);

        for (i = 0; i < N_x; i++) {
            for (j = 0; j < N; j++) {
                int idx = i * N + j;
                if (i == j) {
                    jac_trap[idx] = (-2.0 / Dt) * (jac_trap[idx] - 1.0);
                } else {
                    jac_trap[idx] = (-2.0 / Dt) * jac_trap[idx];
                }
            }
        }
        return 0; // Exit successfully
    }

    // --- MODE 2: DIAGNOSTIC MODE ---
    // If intparams[5] == 1, compute Jacobian and the current dynamic residuals, then exit
    if (intparams[5] == 1) {
        f_run_eval(f, x, y, u, p, Dt);
        g_run_eval(g, x, y, u, p, Dt);

        // Compute residuals assuming we try to hold the state exactly where it is
        for (i = 0; i < N_x; i++) fg[i] = Dt * f[i];
        for (i = 0; i < N_y; i++) fg[i + N_x] = -g[i];

        for (k = 0; k < N * N; k++) jac_trap[k] = 0.0;
        jac_trap_eval(jac_trap, x, y, u, p, Dt);
        
        // Transform Jacobian based on the selected Alpha method
        if (alpha != 0.5) {
            for (i = 0; i < N_x; i++) {
                for (j = 0; j < N; j++) {
                    int idx = i * N + j;
                    if (i == j) jac_trap[idx] = k_jac * jac_trap[idx] + (1.0 - k_jac);
                    else jac_trap[idx] = k_jac * jac_trap[idx];
                }
            }
        }
        return 0; // Exit successfully
    }

    // --- MODE 3: REGULAR TIME INTEGRATION ---
    double* x_0 = (double*)malloc(N_x * sizeof(double));
    double* f_0 = (double*)malloc(N_x * sizeof(double));
    double* Dxy = (double*)malloc(N * sizeof(double));

    f_run_eval(f, x, y, u, p, Dt);
    g_run_eval(g, x, y, u, p, Dt);

    // Initial storage step (t=0). Only store if this is the very first step globally.
    if (intparams[1] == 0 && its[0] < N_store && intparams[7] == 0) {
        h_eval(z, x, y, u, p, Dt);
        Time[its[0]] = t;     
        for (i = 0; i < N_x; i++) X[its[0] * N_x + i] = x[i];     
        for (i = 0; i < N_y; i++) Y[its[0] * N_y + i] = y[i];     
        for (i = 0; i < N_z; i++) Z[its[0] * N_z + i] = z[i];     
        its[0] += 1;
    }

    while (t < t_end - 1e-9) {    
        t += Dt;
        intparams[7] += 1; // Increment global step counter
        
        for (i = 0; i < N_x; i++) {
            f_0[i] = f[i];
            x_0[i] = x[i];
        }
        
        // Newton-Raphson Loop
        for (it = 0; it < max_it; it++) {
            f_run_eval(f, x, y, u, p, Dt);
            g_run_eval(g, x, y, u, p, Dt);

            for (i = 0; i < N_x; i++) {
                fg[i] = -(x[i] - x_0[i] - Dt * (alpha * f[i] + (1.0 - alpha) * f_0[i]));
            }
            for (i = 0; i < N_y; i++) {
                fg[i + N_x] = -g[i]; 
            }

            if (intparams[0] == 0 || (intparams[0] == 1 && it == 0)) { 
                for (k = 0; k < N * N; k++) jac_trap[k] = 0.0;
                
                jac_trap_eval(jac_trap, x, y, u, p, Dt);

                // Apply alpha method transformation to Jacobian if needed
                if (alpha != 0.5) {
                    for (i = 0; i < N_x; i++) {
                        for (j = 0; j < N; j++) {
                            int idx = i * N + j;
                            if (i == j) jac_trap[idx] = k_jac * jac_trap[idx] + (1.0 - k_jac);
                            else jac_trap[idx] = k_jac * jac_trap[idx];
                        }
                    }
                }

                if (solve_dense(jac_trap, N, pivots, fg, Dxy, 0) != 0) return -1;
            }
            
            solve_dense(jac_trap, N, pivots, fg, Dxy, 1);

            for (i = 0; i < N; i++) xy[i] += Dxy[i];         
            for (i = 0; i < N_x; i++) x[i] = xy[i]; 
            for (i = 0; i < N_y; i++) y[i] = xy[i + N_x]; 

            norma = 0.0;
            for (i = 0; i < N; i++) norma += fg[i] * fg[i];
            if (norma < itol) break; // Convergence achieved
        }

        // DECIMATED STORAGE: Only store if global step matches decimation frequency
        if (intparams[1] == 0 && its[0] < N_store && (intparams[7] % decimation == 0)) {
            h_eval(z, x, y, u, p, Dt);
            Time[its[0]] = t;     
            for (i = 0; i < N_x; i++) X[its[0] * N_x + i] = x[i];     
            for (i = 0; i < N_y; i++) Y[its[0] * N_y + i] = y[i];     
            for (i = 0; i < N_z; i++) Z[its[0] * N_z + i] = z[i];     
            its[0] += 1;
        }
    }    

    intparams[2] = it; // Pass final iteration count back to Python
    free(x_0); 
    free(f_0); 
    free(Dxy);
    return 0;
}