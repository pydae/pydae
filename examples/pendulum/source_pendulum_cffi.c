void f_ini_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = x[2];
out[1] = x[3];
out[2] = (-p[3]*x[2] + y[1] - 2*y[0]*x[0])/p[2];
out[3] = (-p[1]*p[2] - p[3]*x[3] - 2*y[0]*x[1])/p[2];

}
void g_ini_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = -p[4]*y[0] - pow(p[0], 2) + pow(x[0], 2) + pow(x[1], 2);
out[1] = -u[0] + atan2(x[0], -x[1]);

}
void f_run_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = x[2];
out[1] = x[3];
out[2] = (-p[3]*x[2] + u[0] - 2*y[0]*x[0])/p[2];
out[3] = (-p[1]*p[2] - p[3]*x[3] - 2*y[0]*x[1])/p[2];

}
void g_run_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = -p[4]*y[0] - pow(p[0], 2) + pow(x[0], 2) + pow(x[1], 2);
out[1] = -y[1] + atan2(x[0], -x[1]);

}
void h_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = -p[4]*y[0] - pow(p[0], 2) + pow(x[0], 2) + pow(x[1], 2);
out[1] = p[1]*p[2]*x[1];
out[2] = 0.5*p[2]*(pow(x[2], 2) + pow(x[3], 2));
out[3] = y[1];

}
void de_jac_ini_xy_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[12] = -2*y[0]/p[2];
out[16] = -2*x[0]/p[2];
out[19] = -2*y[0]/p[2];
out[22] = -2*x[1]/p[2];
out[24] = 2*x[0];
out[25] = 2*x[1];
out[30] = -x[1]/(pow(x[0], 2) + pow(x[1], 2));
out[31] = x[0]/(pow(x[0], 2) + pow(x[1], 2));

}

void de_jac_ini_up_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[14] = -p[3]/p[2];
out[17] = 1.0/p[2];
out[21] = -p[3]/p[2];
out[28] = -p[4];

}

void de_jac_ini_num_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[2] = 1;
out[9] = 1;

}

void sp_jac_ini_xy_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[2] = -2*y[0]/p[2];
out[4] = -2*x[0]/p[2];
out[6] = -2*y[0]/p[2];
out[8] = -2*x[1]/p[2];
out[9] = 2*x[0];
out[10] = 2*x[1];
out[12] = -x[1]/(pow(x[0], 2) + pow(x[1], 2));
out[13] = x[0]/(pow(x[0], 2) + pow(x[1], 2));

}

void sp_jac_ini_up_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[3] = -p[3]/p[2];
out[5] = 1.0/p[2];
out[7] = -p[3]/p[2];
out[11] = -p[4];

}

void sp_jac_ini_num_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = 1;
out[1] = 1;

}

void de_jac_run_xy_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[12] = -2*y[0]/p[2];
out[16] = -2*x[0]/p[2];
out[19] = -2*y[0]/p[2];
out[22] = -2*x[1]/p[2];
out[24] = 2*x[0];
out[25] = 2*x[1];
out[30] = -x[1]/(pow(x[0], 2) + pow(x[1], 2));
out[31] = x[0]/(pow(x[0], 2) + pow(x[1], 2));

}

void de_jac_run_up_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[14] = -p[3]/p[2];
out[21] = -p[3]/p[2];
out[28] = -p[4];

}

void de_jac_run_num_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[2] = 1;
out[9] = 1;
out[35] = -1;

}

void sp_jac_run_xy_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[2] = -2*y[0]/p[2];
out[4] = -2*x[0]/p[2];
out[5] = -2*y[0]/p[2];
out[7] = -2*x[1]/p[2];
out[8] = 2*x[0];
out[9] = 2*x[1];
out[11] = -x[1]/(pow(x[0], 2) + pow(x[1], 2));
out[12] = x[0]/(pow(x[0], 2) + pow(x[1], 2));

}

void sp_jac_run_up_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[3] = -p[3]/p[2];
out[6] = -p[3]/p[2];
out[10] = -p[4];

}

void sp_jac_run_num_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = 1;
out[1] = 1;
out[13] = -1;

}

void de_jac_trap_xy_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[12] = 1.0*Dt*y[0]/p[2];
out[16] = 1.0*Dt*x[0]/p[2];
out[19] = 1.0*Dt*y[0]/p[2];
out[22] = 1.0*Dt*x[1]/p[2];
out[24] = 2*x[0];
out[25] = 2*x[1];
out[30] = -x[1]/(pow(x[0], 2) + pow(x[1], 2));
out[31] = x[0]/(pow(x[0], 2) + pow(x[1], 2));

}

void de_jac_trap_up_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[2] = -0.5*Dt;
out[9] = -0.5*Dt;
out[14] = 0.5*Dt*p[3]/p[2] + 1;
out[21] = 0.5*Dt*p[3]/p[2] + 1;
out[28] = -p[4];

}

void de_jac_trap_num_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = 1;
out[7] = 1;
out[35] = -1;

}

void sp_jac_trap_xy_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[4] = 1.0*Dt*y[0]/p[2];
out[6] = 1.0*Dt*x[0]/p[2];
out[7] = 1.0*Dt*y[0]/p[2];
out[9] = 1.0*Dt*x[1]/p[2];
out[10] = 2*x[0];
out[11] = 2*x[1];
out[13] = -x[1]/(pow(x[0], 2) + pow(x[1], 2));
out[14] = x[0]/(pow(x[0], 2) + pow(x[1], 2));

}

void sp_jac_trap_up_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[1] = -0.5*Dt;
out[3] = -0.5*Dt;
out[5] = 0.5*Dt*p[3]/p[2] + 1;
out[8] = 0.5*Dt*p[3]/p[2] + 1;
out[12] = -p[4];

}

void sp_jac_trap_num_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = 1;
out[2] = 1;
out[15] = -1;

}

