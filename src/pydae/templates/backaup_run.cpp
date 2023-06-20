int run(int * pt,double t, double t_end, double *jac_trap,int *indptr,int *indices,double *x,double *y,double *xy,double *u,double *p,int N_x,int N_y,int max_it, double itol, int * its, double Dt, double *z, double *dblparams, int *intparams, double * Time, double * X, double * Y, double * Z, int N_z, int N_store)
{
    mkl_verbose(0);
    int i;
    double norma;
    int N;
    int flag;
    int it = 0;
    N = N_x+N_y;

    double* f = (double*)malloc(N_x * sizeof(double));
    double* g = (double*)malloc(N_y * sizeof(double));
    double* fg = (double*)malloc(N * sizeof(double));
    double* x_0 = (double*)malloc(N_x * sizeof(double));
    double* f_0 = (double*)malloc(N_x * sizeof(double));
    double* Dxy = (double*)malloc(N * sizeof(double));

    sp_jac_trap_num_eval(jac_trap,x,y,u,p,Dt);
    sp_jac_trap_up_eval(jac_trap,x,y,u,p,Dt);

    f_run_eval(f,x,y,u,p,Dt);
    g_run_eval(g,x,y,u,p,Dt);

    while (t<t_end) // time loop
    {    
        
        t += Dt;

        // f_run_eval(f,x,y,u,p,Dt);
        // g_run_eval(g,x,y,u,p,Dt);

        for (i = 0; i < N_x; i++)
        {
            f_0[i] = f[i];
            x_0[i] = x[i];
        }
        
        // algebraic loop 
        for  (it = 0; it < max_it; it++)
        {

            f_run_eval(f,x,y,u,p,Dt);
            g_run_eval(g,x,y,u,p,Dt);
            sp_jac_trap_xy_eval(jac_trap,x,y,u,p,Dt); 

            for (i = 0; i < N_x; i++) //f_n_i = x - x_0 - 0.5*Dt*(f+f_0) 
            {
                fg[i] = -(x[i]-x_0[i] - 0.5*Dt*(f[i]+f_0[i]));
            }
            for (i = 0; i < N_y; i++)
            {
                fg[i+N_x] =-g[i];
            } 

            if (intparams[0] == 0) { // factorization is always computed
            flag = 0;
            solve(pt,jac_trap, indptr, indices, N, fg,Dxy, flag); 
            }

            if (intparams[0] == 1) // factorization is only computed in the first iteration
            { 
                if (it == 0) 
                { // factorization is only computed in the first iteration
                    flag = 0;
                    solve(pt,jac_trap, indptr, indices, N, fg,Dxy, flag); 
                }
            }


            flag = 1; // linear system solution
            solve(pt,jac_trap, indptr, indices, N, fg,Dxy, flag);


            for (i = 0; i < (N_y+N_x); i++)
            {
                xy[i] += Dxy[i];
                
            }         

            for (i = 0; i < N_x; i++)
            {
                x[i] = xy[i];
            } 
            for (i = 0; i < N_y; i++)
            {
                y[i] = xy[i+N_x];
            } 

            norma = 0.0;
            for (i = 0; i < (N_y+N_x); i++)
            {
                norma += fg[i]*fg[i];
            } 
            if (norma < itol) {     
                
                break;
                
            }


        }
    if (intparams[1] == 0)
    {
        h_eval(z,x,y,u,p,Dt);

        Time[its[0]] = t;     

        for (i = 0; i < N_x; i++)
        {
            X[its[0]*N_x+i] = x[i];     
        }    
        for (i = 0; i < N_y; i++)
        {
            Y[its[0]*N_y+i] = y[i];     
        }  
        for (i = 0; i < N_z; i++)
        {
            Z[its[0]*N_z+i] = z[i];     
        }   
    }
    its[0] += 1;
    }    

    free(f);
    free(g);
    free(fg);
    free(x_0);
    free(f_0);
    free(Dxy);

    intparams[2] = it;


    return 0;
}