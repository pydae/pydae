#ifndef DAESOLVER_DENSE_H
#define DAESOLVER_DENSE_H

// --- Core Linear Solver ---
int solve_dense(double *A, int N, int *pivots, double *b, double *x, int flag);

// --- Equation Signatures (Implemented in build/temp_ctypes.c) ---
void f_ini_eval(double *data, double *x, double *y, double *u, double *p, double Dt);
void g_ini_eval(double *data, double *x, double *y, double *u, double *p, double Dt);
void f_run_eval(double *data, double *x, double *y, double *u, double *p, double Dt);
void g_run_eval(double *data, double *x, double *y, double *u, double *p, double Dt);
void h_eval(double *data, double *x, double *y, double *u, double *p, double Dt);
void jac_ini_eval(double *data, double *x, double *y, double *u, double *p, double Dt);
void jac_trap_eval(double *data, double *x, double *y, double *u, double *p, double Dt);

// --- High-Level Solver Functions ---
int ini(double *jac_ini, int *pivots, double *x, double *y, double *xy, double *Dxy, double *u, double *p, int N_x, int N_y, int max_it, double itol, double *z, double *inidblparams, int *iniintparams, double *f, double *g, double *fg);
int run(double t, double t_end, double *jac_trap, int *pivots, double *x, double *y, double *xy, 
        double *u, double *p, int N_x, int N_y, int max_it, double itol, int *its, double Dt, 
        double *z, double *dblparams, int *intparams, double *Time, double *X, double *Y, 
        double *Z, int N_z, int N_store, double *f, double *g, double *fg);
        
        
#endif