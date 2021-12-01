void f_ini_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = -p[1]*(p[0]*x[2] - x[0]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) + x[1]*u[3] + u[0];
out[1] = -p[1]*(p[0]*x[3] - x[1]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) - x[0]*u[3] + u[1];
out[2] = -p[3]*(p[0]*x[0] - x[2]*(p[2] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) + x[3]*(-p[5]*x[4] + u[3]);
out[3] = -p[3]*(p[0]*x[1] - x[3]*(p[2] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) - x[2]*(-p[5]*x[4] + u[3]);
out[4] = (-p[7]*x[4] - u[2] + y[0])/p[6];

}
void g_ini_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = 1.5*p[5]*(x[0]*(p[0]*x[3] - x[1]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) - x[1]*(p[0]*x[2] - x[0]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]))) - y[0];

}
void f_run_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = -p[1]*(p[0]*x[2] - x[0]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) + x[1]*u[3] + u[0];
out[1] = -p[1]*(p[0]*x[3] - x[1]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) - x[0]*u[3] + u[1];
out[2] = -p[3]*(p[0]*x[0] - x[2]*(p[2] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) + x[3]*(-p[5]*x[4] + u[3]);
out[3] = -p[3]*(p[0]*x[1] - x[3]*(p[2] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) - x[2]*(-p[5]*x[4] + u[3]);
out[4] = (-p[7]*x[4] - u[2] + y[0])/p[6];

}
void h_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = (p[0]*x[2] - x[0]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[1] = (p[0]*x[3] - x[1]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[2] = (p[0]*x[0] - x[2]*(p[2] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[3] = (p[0]*x[1] - x[3]*(p[2] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[4] = sqrt(pow(p[0]*x[2] - x[0]*(p[4] + p[0]), 2)/pow(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]), 2) + pow(p[0]*x[3] - x[1]*(p[4] + p[0]), 2)/pow(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]), 2));
out[5] = p[7]*x[4];
out[6] = u[2];
out[7] = 1.5*u[0]*(p[0]*x[2] - x[0]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) + 1.5*u[1]*(p[0]*x[3] - x[1]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[8] = -1.5*u[0]*(p[0]*x[3] - x[1]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) + 1.5*u[1]*(p[0]*x[2] - x[0]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));

}
void g_run_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = 1.5*p[5]*(x[0]*(p[0]*x[3] - x[1]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) - x[1]*(p[0]*x[2] - x[0]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]))) - y[0];

}
void de_jac_ini_xy_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[15] = -p[5]*x[4] + u[3];
out[16] = -p[5]*x[3];
out[20] = p[5]*x[4] - u[3];
out[22] = p[5]*x[2];
out[30] = 1.5*p[5]*(-x[1]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) + (p[0]*x[3] - x[1]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])));
out[31] = 1.5*p[5]*(x[0]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) - (p[0]*x[2] - x[0]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])));
out[32] = -1.5*p[0]*p[5]*x[1]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[33] = 1.5*p[0]*p[5]*x[0]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));

}

void de_jac_ini_up_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = -p[1]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[1] = u[3];
out[2] = -p[0]*p[1]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[6] = -u[3];
out[7] = -p[1]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[9] = -p[0]*p[1]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[12] = -p[0]*p[3]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[14] = -p[3]*(-p[2] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[19] = -p[0]*p[3]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[21] = -p[3]*(-p[2] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[28] = -p[7]/p[6];
out[29] = 1.0/p[6];

}

void de_jac_ini_num_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[35] = -1;

}

void sp_jac_ini_xy_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[8] = -p[5]*x[4] + u[3];
out[9] = -p[5]*x[3];
out[11] = p[5]*x[4] - u[3];
out[13] = p[5]*x[2];
out[16] = 1.5*p[5]*(-x[1]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) + (p[0]*x[3] - x[1]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])));
out[17] = 1.5*p[5]*(x[0]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) - (p[0]*x[2] - x[0]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])));
out[18] = -1.5*p[0]*p[5]*x[1]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[19] = 1.5*p[0]*p[5]*x[0]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));

}

void sp_jac_ini_up_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = -p[1]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[1] = u[3];
out[2] = -p[0]*p[1]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[3] = -u[3];
out[4] = -p[1]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[5] = -p[0]*p[1]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[6] = -p[0]*p[3]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[7] = -p[3]*(-p[2] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[10] = -p[0]*p[3]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[12] = -p[3]*(-p[2] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[14] = -p[7]/p[6];
out[15] = 1.0/p[6];

}

void sp_jac_ini_num_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[20] = -1;

}

void de_jac_trap_xy_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[15] = -0.5*Dt*(-p[5]*x[4] + u[3]);
out[16] = 0.5*Dt*p[5]*x[3];
out[20] = -0.5*Dt*(p[5]*x[4] - u[3]);
out[22] = -0.5*Dt*p[5]*x[2];
out[30] = 1.5*p[5]*(-x[1]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) + (p[0]*x[3] - x[1]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])));
out[31] = 1.5*p[5]*(x[0]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) - (p[0]*x[2] - x[0]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])));
out[32] = -1.5*p[0]*p[5]*x[1]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[33] = 1.5*p[0]*p[5]*x[0]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));

}

void de_jac_trap_up_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = 0.5*Dt*p[1]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) + 1;
out[1] = -0.5*Dt*u[3];
out[2] = 0.5*Dt*p[0]*p[1]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[6] = 0.5*Dt*u[3];
out[7] = 0.5*Dt*p[1]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) + 1;
out[9] = 0.5*Dt*p[0]*p[1]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[12] = 0.5*Dt*p[0]*p[3]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[14] = 0.5*Dt*p[3]*(-p[2] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) + 1;
out[19] = 0.5*Dt*p[0]*p[3]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[21] = 0.5*Dt*p[3]*(-p[2] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) + 1;
out[28] = 0.5*p[7]*Dt/p[6] + 1;
out[29] = -0.5*Dt/p[6];

}

void de_jac_trap_num_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[35] = -1;

}

void sp_jac_trap_xy_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[8] = -0.5*Dt*(-p[5]*x[4] + u[3]);
out[9] = 0.5*Dt*p[5]*x[3];
out[11] = -0.5*Dt*(p[5]*x[4] - u[3]);
out[13] = -0.5*Dt*p[5]*x[2];
out[16] = 1.5*p[5]*(-x[1]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) + (p[0]*x[3] - x[1]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])));
out[17] = 1.5*p[5]*(x[0]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) - (p[0]*x[2] - x[0]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])));
out[18] = -1.5*p[0]*p[5]*x[1]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[19] = 1.5*p[0]*p[5]*x[0]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));

}

void sp_jac_trap_up_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = 0.5*Dt*p[1]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) + 1;
out[1] = -0.5*Dt*u[3];
out[2] = 0.5*Dt*p[0]*p[1]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[3] = 0.5*Dt*u[3];
out[4] = 0.5*Dt*p[1]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) + 1;
out[5] = 0.5*Dt*p[0]*p[1]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[6] = 0.5*Dt*p[0]*p[3]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[7] = 0.5*Dt*p[3]*(-p[2] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) + 1;
out[10] = 0.5*Dt*p[0]*p[3]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[12] = 0.5*Dt*p[3]*(-p[2] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) + 1;
out[14] = 0.5*p[7]*Dt/p[6] + 1;
out[15] = -0.5*Dt/p[6];

}

void sp_jac_trap_num_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[20] = -1;

}

