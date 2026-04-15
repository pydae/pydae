//int solve(int * pt,double * a, int * ia, int * ja, int n, double * b, double * x, int flag);
//int step(int * pt,double t, double t_end, double *jac_trap,int *indptr,int *indices,double *x,double *y,double *xy,double *u,double *p,int N_x,int N_y,int max_it, double itol, int its, double Dt, double *z, double *dblparams, int *intparams);
void f_run_eval(double *f,double *x,double *y,double *u,double *p,double Dt);
void g_run_eval(double *g,double *x,double *y,double *u,double *p,double Dt);
void sp_jac_trap_num_eval(double *out,double *x,double *y,double *u,double *p,double Dt);
void sp_jac_trap_up_eval(double *out,double *x,double *y,double *u,double *p,double Dt);
void sp_jac_trap_xy_eval(double *out,double *x,double *y,double *u,double *p,double Dt);
void h_eval(double *data,double *x,double *y,double *u,double *p,double Dt);
//int step2(int * pt,double t, double t_end, double *jac_trap,int *indptr,int *indices,double *x,double *y,double *xy,double *u,double *p,int N_x,int N_y,int max_it, double itol, int its, double Dt, double *z, double *dblparams, int *intparams);
