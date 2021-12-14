void f_ini_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = (-p[4]*p[1]*p[2]*y[1] - 0.5*p[7]*p[5]*p[6]*pow(x[0], 2)*y[1] - p[1]*p[2]*sin(u[1]) + u[0]/(p[3]*p[0]))/p[2];
out[1] = x[0] - 0.0001*x[1];

}
void g_ini_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = -y[1] - 1 + 2/(1 + exp(-p[8]*x[0]));
out[1] = -y[0] + x[0]/(p[3]*p[0]);

}
void f_run_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = (-p[4]*p[1]*p[2]*y[1] - 0.5*p[7]*p[5]*p[6]*pow(x[0], 2)*y[1] - p[1]*p[2]*sin(u[1]) + u[0]/(p[3]*p[0]))/p[2];
out[1] = x[0] - 0.0001*x[1];

}
void g_run_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = -y[1] - 1 + 2/(1 + exp(-p[8]*x[0]));
out[1] = -y[0] + x[0]/(p[3]*p[0]);

}
void h_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = u[0]/(p[3]*p[0]);
out[1] = 0.5*p[7]*p[5]*p[6]*pow(x[0], 2)*y[1];
out[2] = p[4]*p[1]*p[2]*y[1];
out[3] = u[0];
out[4] = y[0]*u[0];
out[5] = 3.6000000000000001*x[0];

}
void de_jac_ini_xy_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = -1.0*p[7]*p[5]*p[6]*x[0]*y[1]/p[2];
out[3] = (-p[4]*p[1]*p[2] - 0.5*p[7]*p[5]*p[6]*pow(x[0], 2))/p[2];
out[8] = 2*p[8]*exp(-p[8]*x[0])/pow(1 + exp(-p[8]*x[0]), 2);

}

void de_jac_ini_up_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[12] = 1/(p[3]*p[0]);

}

void de_jac_ini_num_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[4] = 1;
out[5] = -0.0001;
out[11] = -1;
out[14] = -1;

}

void sp_jac_ini_xy_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = -1.0*p[7]*p[5]*p[6]*x[0]*y[1]/p[2];
out[1] = (-p[4]*p[1]*p[2] - 0.5*p[7]*p[5]*p[6]*pow(x[0], 2))/p[2];
out[4] = 2*p[8]*exp(-p[8]*x[0])/pow(1 + exp(-p[8]*x[0]), 2);

}

void sp_jac_ini_up_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[6] = 1/(p[3]*p[0]);

}

void sp_jac_ini_num_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[2] = 1;
out[3] = -0.0001;
out[5] = -1;
out[7] = -1;

}

void de_jac_run_xy_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = -1.0*p[7]*p[5]*p[6]*x[0]*y[1]/p[2];
out[3] = (-p[4]*p[1]*p[2] - 0.5*p[7]*p[5]*p[6]*pow(x[0], 2))/p[2];
out[8] = 2*p[8]*exp(-p[8]*x[0])/pow(1 + exp(-p[8]*x[0]), 2);

}

void de_jac_run_up_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[12] = 1/(p[3]*p[0]);

}

void de_jac_run_num_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[4] = 1;
out[5] = -0.0001;
out[11] = -1;
out[14] = -1;

}

void sp_jac_run_xy_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = -1.0*p[7]*p[5]*p[6]*x[0]*y[1]/p[2];
out[1] = (-p[4]*p[1]*p[2] - 0.5*p[7]*p[5]*p[6]*pow(x[0], 2))/p[2];
out[4] = 2*p[8]*exp(-p[8]*x[0])/pow(1 + exp(-p[8]*x[0]), 2);

}

void sp_jac_run_up_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[6] = 1/(p[3]*p[0]);

}

void sp_jac_run_num_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[2] = 1;
out[3] = -0.0001;
out[5] = -1;
out[7] = -1;

}

void de_jac_trap_xy_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = 0.5*p[7]*Dt*p[5]*p[6]*x[0]*y[1]/p[2] + 1;
out[3] = -0.5*Dt*(-p[4]*p[1]*p[2] - 0.5*p[7]*p[5]*p[6]*pow(x[0], 2))/p[2];
out[8] = 2*p[8]*exp(-p[8]*x[0])/pow(1 + exp(-p[8]*x[0]), 2);

}

void de_jac_trap_up_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[4] = -0.5*Dt;
out[5] = 5.0000000000000002e-5*Dt + 1;
out[12] = 1/(p[3]*p[0]);

}

void de_jac_trap_num_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[11] = -1;
out[14] = -1;

}

void sp_jac_trap_xy_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[0] = 0.5*p[7]*Dt*p[5]*p[6]*x[0]*y[1]/p[2] + 1;
out[1] = -0.5*Dt*(-p[4]*p[1]*p[2] - 0.5*p[7]*p[5]*p[6]*pow(x[0], 2))/p[2];
out[4] = 2*p[8]*exp(-p[8]*x[0])/pow(1 + exp(-p[8]*x[0]), 2);

}

void sp_jac_trap_up_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[2] = -0.5*Dt;
out[3] = 5.0000000000000002e-5*Dt + 1;
out[6] = 1/(p[3]*p[0]);

}

void sp_jac_trap_num_eval(double *out,double *x,double *y,double *u,double *p,double Dt){

out[5] = -1;
out[7] = -1;

}

