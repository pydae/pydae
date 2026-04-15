#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "daesolver_run_lapack.h"

// LAPACK Fortran function declarations
extern void dgetrf_(int* M, int* N, double* A, int* lda, int* IPIV, int* INFO);
extern void dgetrs_(char* TRANS, int* N, int* NRHS, double* A, int* lda, int* IPIV, double* B, int* ldb, int* INFO);

// --------------------------------------------------------------------
// Portable LAPACK Wrapper for Dense Linear Solver
// flag = 0: Factorize matrix A using dgetrf_
// flag = 1: Solve Ax = b using dgetrs_
// --------------------------------------------------------------------
int solve_dense(double *A, int N, int *pivots, double *b, double *x, int flag) {
    int info = 0;
    int nrhs = 1;

    if (flag == 0) {
        // LU Factorization
        // Note: A is conceptually transposed due to row-major/col-major mismatch.
        dgetrf_(&N, &N, A, &N, pivots, &info);
        
        if (info < 0) {
            printf("LAPACK dgetrf_ error: Argument %d had an illegal value.\n", -info);
            return -1; 
        } else if (info > 0) {
            printf("LAPACK dgetrf_ error: Matrix is singular (U(%d,%d) is 0).\n", info, info);
            return -1;
        }
    } 
    else if (flag == 1) {
        // LAPACK dgetrs_ overwrites the right-hand side (B) with the solution (X).
        // So we copy our RHS `b` into our output array `x`, and pass `x` to LAPACK.
        for (int i = 0; i < N; i++) {
            x[i] = b[i];
        }

        // We use 'T' for Transpose to reverse the fact that our C-array 
        // was read by Fortran as Column-Major (which implicitly transposed it).
        char trans = 'T'; 
        
        dgetrs_(&trans, &N, &nrhs, A, &N, pivots, x, &N, &info);
        
        if (info != 0) {
            printf("LAPACK dgetrs_ error: Info = %d\n", info);
            return -1;
        }
    }
    return 0;
}

// --------------------------------------------------------------------
// DAE Step Function 
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
        
        for (it = 0; it < max_it; it++) {
            f_run_eval(f, x, y, u, p, Dt);
            g_run_eval(g, x, y, u, p, Dt);

            for (i = 0; i < N_x; i++) {
                fg[i] = -(x[i] - x_0[i] - 0.5 * Dt * (f[i] + f_0[i]));
            }
            for (i = 0; i < N_y; i++) {
                fg[i + N_x] = -g[i];
            } 

            if (intparams[0] == 0 || (intparams[0] == 1 && it == 0)) { 
                for (k = 0; k < N * N; k++) jac_trap[k] = 0.0;
                jac_trap_eval(jac_trap, x, y, u, p, Dt);
                solve_dense(jac_trap, N, pivots, fg, Dxy, 0); 
            }

            solve_dense(jac_trap, N, pivots, fg, Dxy, 1);

            for (i = 0; i < N; i++) xy[i] += Dxy[i];         
            for (i = 0; i < N_x; i++) x[i] = xy[i]; 
            for (i = 0; i < N_y; i++) y[i] = xy[i + N_x]; 

            norma = 0.0;
            for (i = 0; i < N; i++) norma += fg[i] * fg[i];
            if (norma < itol) break;
        }
    }    

    intparams[2] = it;
    if (intparams[1] == 0) h_eval(z, x, y, u, p, Dt);

    free(f); free(g); free(fg);
    free(x_0); free(f_0); free(Dxy);
    return 0;
}

// --------------------------------------------------------------------
// DAE Run Function 
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

    if (intparams[1] == 0 && its[0] < N_store) {
        h_eval(z, x, y, u, p, Dt);
        Time[its[0]] = t;     
        for (i = 0; i < N_x; i++) X[its[0] * N_x + i] = x[i];     
        for (i = 0; i < N_y; i++) Y[its[0] * N_y + i] = y[i];     
        for (i = 0; i < N_z; i++) Z[its[0] * N_z + i] = z[i];     
        its[0] += 1;
    }

    while (t < t_end - 1e-9) {    
        t += Dt;

        for (i = 0; i < N_x; i++) {
            f_0[i] = f[i];
            x_0[i] = x[i];
        }
        
        for (it = 0; it < max_it; it++) {
            f_run_eval(f, x, y, u, p, Dt);
            g_run_eval(g, x, y, u, p, Dt);

            for (i = 0; i < N_x; i++) fg[i] = -(x[i] - x_0[i] - 0.5 * Dt * (f[i] + f_0[i]));
            for (i = 0; i < N_y; i++) fg[i + N_x] = -g[i]; 

            if (intparams[0] == 0 || (intparams[0] == 1 && it == 0)) { 
                for (k = 0; k < N * N; k++) jac_trap[k] = 0.0;
                jac_trap_eval(jac_trap, x, y, u, p, Dt);
                solve_dense(jac_trap, N, pivots, fg, Dxy, 0); 
            }

            solve_dense(jac_trap, N, pivots, fg, Dxy, 1);

            for (i = 0; i < N; i++) xy[i] += Dxy[i];         
            for (i = 0; i < N_x; i++) x[i] = xy[i]; 
            for (i = 0; i < N_y; i++) y[i] = xy[i + N_x]; 

            norma = 0.0;
            for (i = 0; i < N; i++) norma += fg[i] * fg[i];
            if (norma < itol) break;
        }

        if (intparams[1] == 0 && its[0] < N_store) {
            h_eval(z, x, y, u, p, Dt);
            Time[its[0]] = t;     
            for (i = 0; i < N_x; i++) X[its[0] * N_x + i] = x[i];     
            for (i = 0; i < N_y; i++) Y[its[0] * N_y + i] = y[i];     
            for (i = 0; i < N_z; i++) Z[its[0] * N_z + i] = z[i];     
            its[0] += 1;
        }
    }    

    intparams[2] = it;

    free(f); free(g); free(fg);
    free(x_0); free(f_0); free(Dxy);
    return 0;
}