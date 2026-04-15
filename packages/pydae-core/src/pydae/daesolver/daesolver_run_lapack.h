#ifndef DAESOLVER_RUN_LAPACK_H
#define DAESOLVER_RUN_LAPACK_H

// Core linear solver wrapper using LAPACK
int solve_dense(double *A, int N, int *pivots, double *b, double *x, int flag);

// Evaluation function signatures
void f_run_eval(double *f, double *x, double *y, double *u, double *p, double Dt);
void g_run_eval(double *g, double *x, double *y, double *u, double *p, double Dt);
void h_eval(double *data, double *x, double *y, double *u, double *p, double Dt);
void jac_trap_eval(double *out, double *x, double *y, double *u, double *p, double Dt);

// DAE solver steps
int step(double t, double t_end, double *jac_trap, int *pivots, double *x, double *y, double *xy, double *u, double *p, int N_x, int N_y, int max_it, double itol, int *its, double Dt, double *z, double *dblparams, int *intparams);
int run(double t, double t_end, double *jac_trap, int *pivots, double *x, double *y, double *xy, double *u, double *p, int N_x, int N_y, int max_it, double itol, int *its, double Dt, double *z, double *dblparams, int *intparams, double *Time, double *X, double *Y, double *Z, int N_z, int N_store);

#endif // DAESOLVER_RUN_LAPACK_H