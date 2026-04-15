#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dae_run_dense.h"

// --------------------------------------------------------------------
// Portable Dense Linear Solver: LU Decomposition with Partial Pivoting
// flag = 0: Factorize matrix A in-place into LU, populate pivots
// flag = 1: Solve Ax = b using the existing in-place LU factorization
// --------------------------------------------------------------------
int solve_dense(double *A, int N, int *pivots, double *b, double *x, int flag) {
    if (flag == 0) {
        // LU Factorization with partial pivoting
        for (int i = 0; i < N; i++) pivots[i] = i;

        for (int i = 0; i < N; i++) {
            // Find pivot
            double max_el = fabs(A[i * N + i]);
            int max_row = i;
            for (int k = i + 1; k < N; k++) {
                if (fabs(A[k * N + i]) > max_el) {
                    max_el = fabs(A[k * N + i]);
                    max_row = k;
                }
            }

            // Swap rows if necessary
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

            // Check for singularity
            if (fabs(A[i * N + i]) < 1e-14) {
                printf("Error: Singular matrix in dense solver.\n");
                return -1; 
            }

            // Eliminate
            for (int k = i + 1; k < N; k++) {
                A[k * N + i] /= A[i * N + i];
                for (int j = i + 1; j < N; j++) {
                    A[k * N + j] -= A[k * N + i] * A[i * N + j];
                }
            }
        }
    } 
    else if (flag == 1) {
        // Forward substitution for L y = P b (solving in-place into x)
        for (int i = 0; i < N; i++) {
            x[i] = b[pivots[i]];
            for (int k = 0; k < i; k++) {
                x[i] -= A[i * N + k] * x[k];
            }
        }
        // Backward substitution for U x = y
        for (int i = N - 1; i >= 0; i--) {
            for (int k = i + 1; k < N; k++) {
                x[i] -= A[i * N + k] * x[k];
            }
            x[i] /= A[i * N + i];
        }
    }
    return 0;
}

// --------------------------------------------------------------------
// DAE Step Function (Trapezoidal Implicit)
// --------------------------------------------------------------------
int step(double t, double t_end, double *jac_trap, int *pivots, double *x, double *y, double *xy, double *u, double *p, int N_x, int N_y, int max_it, double itol, int *its, double Dt, double *z, double *dblparams, int *intparams) {
    int i, k;
    double norma;
    int N = N_x + N_y;
    int it = 0;

    double* f = (double*)malloc(N_x * sizeof(double));
    double* g = (double*)malloc(N_y * sizeof(double));
    double* fg = (double*)malloc(N * sizeof(double));
    double* x_0 = (double*)malloc(N_x * sizeof(double));
    double* f_0 = (double*)malloc(N_x * sizeof(double));
    double* Dxy = (double*)malloc(N * sizeof(double));

    f_run_eval(f, x, y, u, p, Dt);
    g_run_eval(g, x, y, u, p, Dt);

    while (t < t_end) {    
        its[0] += 1; 
        t += Dt;

        for (i = 0; i < N_x; i++) {
            f_0[i] = f[i];
            x_0[i] = x[i];
        }
        
        // Algebraic Loop (Newton Iterations)
        for (it = 0; it < max_it; it++) {
            f_run_eval(f, x, y, u, p, Dt);
            g_run_eval(g, x, y, u, p, Dt);

            for (i = 0; i < N_x; i++) {
                fg[i] = -(x[i] - x_0[i] - 0.5 * Dt * (f[i] + f_0[i]));
            }
            for (i = 0; i < N_y; i++) {
                fg[i + N_x] = -g[i];
            } 

            // Jacobian Evaluation and Factorization Logic
            if (intparams[0] == 0 || (intparams[0] == 1 && it == 0)) { 
                // Zero out the dense Jacobian matrix before evaluation
                for (k = 0; k < N * N; k++) {
                    jac_trap[k] = 0.0;
                }
                
                // Call the single unified Jacobian function
                jac_trap_eval(jac_trap, x, y, u, p, Dt);
                
                // Factorize
                solve_dense(jac_trap, N, pivots, fg, Dxy, 0); 
            }

            // Linear System Solve
            solve_dense(jac_trap, N, pivots, fg, Dxy, 1);

            // State Update
            for (i = 0; i < N; i++) {
                xy[i] += Dxy[i];
            }         
            for (i = 0; i < N_x; i++) {
                x[i] = xy[i];
            } 
            for (i = 0; i < N_y; i++) {
                y[i] = xy[i + N_x];
            } 

            // Convergence Check
            norma = 0.0;
            for (i = 0; i < N; i++) {
                norma += fg[i] * fg[i];
            } 
            if (norma < itol) {     
                break;
            }
        }
    }    

    intparams[2] = it;
    if (intparams[1] == 0) {
        h_eval(z, x, y, u, p, Dt);
    }

    free(f); free(g); free(fg);
    free(x_0); free(f_0); free(Dxy);

    return 0;
}

// --------------------------------------------------------------------
// DAE Run Function (Trapezoidal Implicit with history storage)
// --------------------------------------------------------------------
int run(double t, double t_end, double *jac_trap, int *pivots, double *x, double *y, double *xy, double *u, double *p, int N_x, int N_y, int max_it, double itol, int *its, double Dt, double *z, double *dblparams, int *intparams, double *Time, double *X, double *Y, double *Z, int N_z, int N_store) {
    int i, k;
    double norma;
    int N = N_x + N_y;
    int it = 0;

    double* f = (double*)malloc(N_x * sizeof(double));
    double* g = (double*)malloc(N_y * sizeof(double));
    double* fg = (double*)malloc(N * sizeof(double));
    double* x_0 = (double*)malloc(N_x * sizeof(double));
    double* f_0 = (double*)malloc(N_x * sizeof(double));
    double* Dxy = (double*)malloc(N * sizeof(double));

    f_run_eval(f, x, y, u, p, Dt);
    g_run_eval(g, x, y, u, p, Dt);

    while (t < t_end) {    
        t += Dt;

        for (i = 0; i < N_x; i++) {
            f_0[i] = f[i];
            x_0[i] = x[i];
        }
        
        // Algebraic Loop
        for (it = 0; it < max_it; it++) {
            f_run_eval(f, x, y, u, p, Dt);
            g_run_eval(g, x, y, u, p, Dt);

            for (i = 0; i < N_x; i++) {
                fg[i] = -(x[i] - x_0[i] - 0.5 * Dt * (f[i] + f_0[i]));
            }
            for (i = 0; i < N_y; i++) {
                fg[i + N_x] = -g[i];
            } 

            // Jacobian Evaluation and Factorization Logic
            if (intparams[0] == 0 || (intparams[0] == 1 && it == 0)) { 
                // Zero out the dense Jacobian matrix before evaluation
                for (k = 0; k < N * N; k++) {
                    jac_trap[k] = 0.0;
                }
                
                // Call the single unified Jacobian function
                jac_trap_eval(jac_trap, x, y, u, p, Dt);
                
                // Factorize
                solve_dense(jac_trap, N, pivots, fg, Dxy, 0); 
            }

            // Linear System Solve
            solve_dense(jac_trap, N, pivots, fg, Dxy, 1);

            for (i = 0; i < N; i++) {
                xy[i] += Dxy[i];
            }         
            for (i = 0; i < N_x; i++) {
                x[i] = xy[i];
            } 
            for (i = 0; i < N_y; i++) {
                y[i] = xy[i + N_x];
            } 

            norma = 0.0;
            for (i = 0; i < N; i++) {
                norma += fg[i] * fg[i];
            } 
            if (norma < itol) {     
                break;
            }
        }

        if (intparams[1] == 0) {
            h_eval(z, x, y, u, p, Dt);
            Time[its[0]] = t;     

            for (i = 0; i < N_x; i++) X[its[0] * N_x + i] = x[i];     
            for (i = 0; i < N_y; i++) Y[its[0] * N_y + i] = y[i];     
            for (i = 0; i < N_z; i++) Z[its[0] * N_z + i] = z[i];     
        }
        its[0] += 1;
    }    

    intparams[2] = it;

    free(f); free(g); free(fg);
    free(x_0); free(f_0); free(Dxy);

    return 0;
}