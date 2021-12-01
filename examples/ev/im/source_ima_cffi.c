void f_ini_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = (-p[7]*x[0] - u[2] + y[0])/p[6];

}
void g_ini_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = 1.5*p[5]*(y[1]*(p[0]*y[4] - y[2]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) - y[2]*(p[0]*y[3] - y[1]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]))) - y[0];
out[1] = -p[1]*(p[0]*y[3] - y[1]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) + y[2]*u[3] + u[0];
out[2] = -p[1]*(p[0]*y[4] - y[2]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) - y[1]*u[3] + u[1];
out[3] = -p[3]*(p[0]*y[1] - y[3]*(p[2] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) + y[4]*(-p[5]*x[0] + u[3]);
out[4] = -p[3]*(p[0]*y[2] - y[4]*(p[2] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) - y[3]*(-p[5]*x[0] + u[3]);

}
void f_run_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = (-p[7]*x[0] - u[2] + y[0])/p[6];

}
void h_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = (p[0]*y[3] - y[1]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[1] = (p[0]*y[4] - y[2]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[2] = (p[0]*y[1] - y[3]*(p[2] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[3] = (p[0]*y[2] - y[4]*(p[2] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[4] = sqrt(pow(p[0]*y[3] - y[1]*(p[4] + p[0]), 2)/pow(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]), 2) + pow(p[0]*y[4] - y[2]*(p[4] + p[0]), 2)/pow(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]), 2));
out[5] = p[7]*x[0];
out[6] = u[2];
out[7] = 1.5*u[0]*(p[0]*y[3] - y[1]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) + 1.5*u[1]*(p[0]*y[4] - y[2]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[8] = -1.5*u[0]*(p[0]*y[4] - y[2]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) + 1.5*u[1]*(p[0]*y[3] - y[1]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));

}
void g_run_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = 1.5*p[5]*(y[1]*(p[0]*y[4] - y[2]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) - y[2]*(p[0]*y[3] - y[1]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]))) - y[0];
out[1] = -p[1]*(p[0]*y[3] - y[1]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) + y[2]*u[3] + u[0];
out[2] = -p[1]*(p[0]*y[4] - y[2]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) - y[1]*u[3] + u[1];
out[3] = -p[3]*(p[0]*y[1] - y[3]*(p[2] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) + y[4]*(-p[5]*x[0] + u[3]);
out[4] = -p[3]*(p[0]*y[2] - y[4]*(p[2] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) - y[3]*(-p[5]*x[0] + u[3]);

}
void de_jac_ini_xy_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[8] = 1.5*p[5]*(-y[2]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) + (p[0]*y[4] - y[2]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])));
out[9] = 1.5*p[5]*(y[1]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) - (p[0]*y[3] - y[1]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])));
out[10] = -1.5*p[0]*p[5]*y[2]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[11] = 1.5*p[0]*p[5]*y[1]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[24] = -p[5]*y[4];
out[29] = -p[5]*x[0] + u[3];
out[30] = p[5]*y[3];
out[34] = p[5]*x[0] - u[3];

}

void de_jac_ini_up_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = -p[7]/p[6];
out[1] = 1.0/p[6];
out[14] = -p[1]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[15] = u[3];
out[16] = -p[0]*p[1]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[20] = -u[3];
out[21] = -p[1]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[23] = -p[0]*p[1]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[26] = -p[0]*p[3]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[28] = -p[3]*(-p[2] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[33] = -p[0]*p[3]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[35] = -p[3]*(-p[2] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));

}

void de_jac_ini_num_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[7] = -1;

}

void sp_jac_ini_xy_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[3] = 1.5*p[5]*(-y[2]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) + (p[0]*y[4] - y[2]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])));
out[4] = 1.5*p[5]*(y[1]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) - (p[0]*y[3] - y[1]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])));
out[5] = -1.5*p[0]*p[5]*y[2]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[6] = 1.5*p[0]*p[5]*y[1]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[13] = -p[5]*y[4];
out[16] = -p[5]*x[0] + u[3];
out[17] = p[5]*y[3];
out[19] = p[5]*x[0] - u[3];

}

void sp_jac_ini_up_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = -p[7]/p[6];
out[1] = 1.0/p[6];
out[7] = -p[1]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[8] = u[3];
out[9] = -p[0]*p[1]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[10] = -u[3];
out[11] = -p[1]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[12] = -p[0]*p[1]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[14] = -p[0]*p[3]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[15] = -p[3]*(-p[2] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[18] = -p[0]*p[3]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[20] = -p[3]*(-p[2] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));

}

void sp_jac_ini_num_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[2] = -1;

}

void de_jac_trap_xy_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[8] = 1.5*p[5]*(-y[2]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) + (p[0]*y[4] - y[2]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])));
out[9] = 1.5*p[5]*(y[1]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) - (p[0]*y[3] - y[1]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])));
out[10] = -1.5*p[0]*p[5]*y[2]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[11] = 1.5*p[0]*p[5]*y[1]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[24] = -p[5]*y[4];
out[29] = -p[5]*x[0] + u[3];
out[30] = p[5]*y[3];
out[34] = p[5]*x[0] - u[3];

}

void de_jac_trap_up_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = 0.5*p[7]*Dt/p[6] + 1;
out[1] = -0.5*Dt/p[6];
out[14] = -p[1]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[15] = u[3];
out[16] = -p[0]*p[1]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[20] = -u[3];
out[21] = -p[1]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[23] = -p[0]*p[1]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[26] = -p[0]*p[3]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[28] = -p[3]*(-p[2] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[33] = -p[0]*p[3]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[35] = -p[3]*(-p[2] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));

}

void de_jac_trap_num_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[7] = -1;

}

void sp_jac_trap_xy_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[3] = 1.5*p[5]*(-y[2]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) + (p[0]*y[4] - y[2]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])));
out[4] = 1.5*p[5]*(y[1]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])) - (p[0]*y[3] - y[1]*(p[4] + p[0]))/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0])));
out[5] = -1.5*p[0]*p[5]*y[2]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[6] = 1.5*p[0]*p[5]*y[1]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[13] = -p[5]*y[4];
out[16] = -p[5]*x[0] + u[3];
out[17] = p[5]*y[3];
out[19] = p[5]*x[0] - u[3];

}

void sp_jac_trap_up_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = 0.5*p[7]*Dt/p[6] + 1;
out[1] = -0.5*Dt/p[6];
out[7] = -p[1]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[8] = u[3];
out[9] = -p[0]*p[1]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[10] = -u[3];
out[11] = -p[1]*(-p[4] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[12] = -p[0]*p[1]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[14] = -p[0]*p[3]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[15] = -p[3]*(-p[2] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[18] = -p[0]*p[3]/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));
out[20] = -p[3]*(-p[2] - p[0])/(pow(p[0], 2) - (p[4] + p[0])*(p[2] + p[0]));

}

void sp_jac_trap_num_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[2] = -1;

}

