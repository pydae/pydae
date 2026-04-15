#include <stdio.h>
#include <stdlib.h>
#include "daesolver_run.h"

#include "C:\Users\jmmau\anaconda3\Library\include/mkl_pardiso.h"
#include "C:\Users\jmmau\anaconda3\Library\include/mkl.h"

// cl example.c /I "C:\Users\jmmau\anaconda3\Library\include\" /link /libpath:"C:\Users\jmmau\anaconda3\pkgs\mkl-devel-2021.4.0-hb8b2395_640\Library\lib\" "C:\Users\jmmau\anaconda3\pkgs\mkl-devel-2021.4.0-hb8b2395_640\Library\lib\mkl_intel_lp64_dll.lib" "C:\Users\jmmau\anaconda3\pkgs\mkl-devel-2021.4.0-hb8b2395_640\Library\lib\mkl_sequential_dll.lib"

int main() {
    //solve(100);

    // // Solve linear system
    // phase = 22;
    // PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, NULL, &nrhs, iparm, &msglvl, b, x, &error);
    // if (error != 0) {
    //     printf("Error in PARDISO: %d\n", error);
    //     exit(1);
    // }

    // // Print solution
    // printf("Solution: ");
    // for (int i = 0; i < n; i++)
    //     printf("%lf ", x[i]);
    // printf("\n");

    // // Release memory
    // phase = -1;
    // PARDISO(pt, &maxfct, &mnum, &mtype, &phase, &n, a, ia, ja, NULL, &nrhs, iparm, &msglvl, b, x, &error);

    return 0;
}



int solve(int * pt,double * a, int * ia, int * ja, int n, double * b, double * x, int flag)
{
    mkl_verbose(0);

    //printf ("\n mkl_verbose: %d", mkl_verbose(0));

    // PARDISO control parameters
    int phase, maxfct, mnum, nrhs, iparm[64], msglvl, error;
    //void *pt[64];
    msglvl = 0; 

    // Initialize PARDISO
    phase = 11;
    maxfct = 1;
    mnum = 1;
    MKL_INT mtype = 11;       /* Real unsymmetric matrix */
    nrhs = 1;


    // pardisoinit(pt, &mtype, iparm); /* Default iparm initialization */
    // printf("Init: ");
    // for (int i = 0; i < 35; i++)
    //     printf("        iparm[%d] = %d; \n", i, iparm[i]);
    // printf("\n");

        // set default parameters
        iparm[0] = 1;  // tell pardiso to not reset these values on the first call
        iparm[1] = 2;  // The nested dissection algorithm from the METIS
        iparm[3] = 0;  // The factorization is always computed as required by phase.
        iparm[4] = 0;  // fill perm with computed permutation vector
        iparm[5] = 0;  // The array x contains the solution; right-hand side vector b is kept unchanged.
        iparm[7] = 0;  // The solver automatically performs two steps of iterative refinement when perterbed pivots are obtained
        iparm[9] = 13;
        iparm[10] = 1;
        iparm[11] = 0;  // Solve a linear system AX = B (as opposed to A.T or A.H)
        iparm[12] = 1;
        iparm[17] =-1;  // Return the number of non-zeros in this value after first call
        iparm[18] =-1;  // do not report flop count
        iparm[20] = 0; //if matrix_type in [-2, -4, 6] else 0
        iparm[23] =10;  // classic (not parallel) factorization
        iparm[24] = 0;  // default behavoir of parallel solving
        iparm[26] = 0;  // Do not check the input matrix
        //set precision
        iparm[27] = 0;
        iparm[30] = 0;  // this would be used to enable sparse input/output for solves
        iparm[33] = 0;  // optimal number of thread for CNR mode
        iparm[34] = 1;  // zero based indexing
        iparm[35] = 0;  // Do not compute schur complement
        iparm[36] = 0;  // use CSR storage format
        iparm[38] = 0;  // Do not use low rank update
        iparm[42] = 0;  // Do not compute the diagonal of the inverse
        iparm[55] = 0;  // Internal function used to work with pivot and calculation of diagonal arrays turned off.
        iparm[59] = 0;  // operate in-core mode


        // iparm[34] = 1;  /* zero based indexing */ 
        // iparm[17] = 1;
        // iparm[18] = 1;


    // Print solution
    // printf("Init: ");
    // for (int i = 0; i < 35; i++)
    //     printf("iparm[%d] = %d \n", i, iparm[i]);
    // printf("\n");

    MKL_INT i;
    double ddum;          /* Double dummy */
    MKL_INT idum;         /* Integer dummy. */


if (flag == 0){  // flag == 0: initialization and symbolic factorization
/* -------------------------------------------------------------------- */
/* .. Initialize the internal solver memory pointer. This is only */
/* necessary for the FIRST call of the PARDISO solver. */
/* -------------------------------------------------------------------- */
    for ( i = 0; i < 64; i++ )
    {
        pt[i] = 0;
    }
/* -------------------------------------------------------------------- */
/* .. Reordering and Symbolic Factorization. This step also allocates */
/* all memory that is necessary for the factorization. */
/* -------------------------------------------------------------------- */

        phase = 11; // Analysis

        // printf ("\n Start solving");

        pardiso (pt, &maxfct, &mnum, &mtype, &phase,
                &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
        if ( error != 0 )
        {
            printf ("\nERROR during symbolic factorization: %d", error);
            exit (1);
        }
        // printf ("\nReordering completed ... ");
        // printf ("\nNumber of nonzeros in factors = %d", iparm[17]);
        // printf ("\nNumber of factorization MFLOPS = %d", iparm[18]);
    }


 if (flag == 1){  // flag == 1: numeric factorization and solving
   
/* -------------------------------------------------------------------- */
/* .. Numerical factorization. */
/* -------------------------------------------------------------------- */

    phase = 22; // Numerical factorization
    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, &ddum, &ddum, &error);
    if ( error != 0 )
    {
        printf ("\nERROR during numerical factorization: %d", error);
        exit (2);
    }
    // printf ("\nFactorization completed ... ");

/* -------------------------------------------------------------------- */
/* .. Back substitution and iterative refinement. */
/* -------------------------------------------------------------------- */
    phase = 33; // Solve, iterative refinement
    iparm[7] = 2;         /* Max numbers of iterative refinement steps. */

    pardiso (pt, &maxfct, &mnum, &mtype, &phase,
             &n, a, ia, ja, &idum, &nrhs, iparm, &msglvl, b, x, &error);
    if ( error != 0 )
    {
        printf ("\nERROR during solution: %d", error);
        exit (3);
    }
    // printf ("\nSolve completed ... ");
    // printf ("\nThe solution of the system is: ");
    // for ( i = 0; i < n; i++ )
    // {
    //     printf ("\n x [%d] = % f", i, x[i]);
    // }
    // printf ("\n");

 }

 if (flag == 10){  // flag == 10: termination

/* -------------------------------------------------------------------- */
/* .. Termination and release of memory. */
/* -------------------------------------------------------------------- */
    phase = -1;  /* Release all internal memory for all matrices */
    PARDISO (pt, &maxfct, &mnum, &mtype, &phase,
             &n, &ddum, ia, ja, &idum, &nrhs,
             iparm, &msglvl, &ddum, &ddum, &error);
 }
    return 0;
 
}

// int step( int * pt,double t, double t_end, double *jac_trap,int *indptr,int *indices,double *x,double *y,double *xy,double *u,double *p,int N_x,int N_y,int max_it, double itol, int * its, double Dt, double *z, double *dblparams, int *intparams)
// {
    // mkl_verbose(0);
    // int i;
    // double norma;
    // int N;
    // int flag;
    // int it = 0;
    // N = N_x+N_y;

    // double* f = (double*)malloc(N_x * sizeof(double));
    // double* g = (double*)malloc(N_y * sizeof(double));
    // double* fg = (double*)malloc(N * sizeof(double));
    // double* x_0 = (double*)malloc(N_x * sizeof(double));
    // double* f_0 = (double*)malloc(N_x * sizeof(double));
    // double* Dxy = (double*)malloc(N * sizeof(double));
    // sp_jac_trap_num_eval(jac_trap,x,y,u,p,Dt);
    // sp_jac_trap_up_eval(jac_trap,x,y,u,p,Dt);

    // f_run_eval(f,x,y,u,p,Dt);
    // g_run_eval(g,x,y,u,p,Dt);    


    // while (t<t_end) // time loop
    // {    
    //     its += 1;
    //     t += Dt;

    //     // f_run_eval(f,x,y,u,p,Dt);
    //     // g_run_eval(g,x,y,u,p,Dt);

    //     for (i = 0; i < N_x; i++)
    //     {
    //         f_0[i] = f[i];
    //         x_0[i] = x[i];
    //     }
        
    //     // algebraic loop 
    //     for  (it = 0; it < max_it; it++)
    //     {

    //         f_run_eval(f,x,y,u,p,Dt);
    //         g_run_eval(g,x,y,u,p,Dt);
    //         sp_jac_trap_xy_eval(jac_trap,x,y,u,p,Dt); 

    //         for (i = 0; i < N_x; i++) //f_n_i = x - x_0 - 0.5*Dt*(f+f_0) 
    //         {
    //             fg[i] = -(x[i]-x_0[i] - 0.5*Dt*(f[i]+f_0[i]));
    //         }
    //         for (i = 0; i < N_y; i++)
    //         {
    //             fg[i+N_x] =-g[i];
    //         } 

    //         if (intparams[0] == 0) { // factorization is always computed
    //         flag = 0;
    //         solve(pt,jac_trap, indptr, indices, N, fg,Dxy, flag); 
    //         }

    //         if (intparams[0] == 1) // factorization is only computed in the first iteration
    //         { 
    //             if (it == 0) 
    //             { // factorization is only computed in the first iteration
    //                 flag = 0;
    //                 solve(pt,jac_trap, indptr, indices, N, fg,Dxy, flag); 
    //             }
    //         }


    //         flag = 1; // linear system solution
    //         solve(pt,jac_trap, indptr, indices, N, fg,Dxy, flag);

    //         if (intparams[0] == 0) { // factorization is always computed
    //         flag = 10;
    //         solve(pt,jac_trap, indptr, indices, N, fg,Dxy, flag); 
    //         }

    //         for (i = 0; i < (N_y+N_x); i++)
    //         {
    //             xy[i] += Dxy[i];
                
    //         }         

    //         for (i = 0; i < N_x; i++)
    //         {
    //             x[i] = xy[i];
    //         } 
    //         for (i = 0; i < N_y; i++)
    //         {
    //             y[i] = xy[i+N_x];
    //         } 

    //         norma = 0.0;
    //         for (i = 0; i < (N_y+N_x); i++)
    //         {
    //             norma += fg[i]*fg[i];
    //         } 
    //         if (norma < itol) {     
                
    //             break;
                
    //         }


    //     }
    //     //printf ("\n N_it[%d]", it); 


    // }    

    // free(f);
    // free(g);
    // free(fg);
    // free(x_0);
    // free(f_0);
    // free(Dxy);

    // intparams[2] = it;
    // if (intparams[1] == 0)
    // {
    //     h_eval(z,x,y,u,p,Dt);
    // }

    // if (intparams[0] == 1) // factorization is only computed in the first iteration
    // { 
    //     flag = 10;
    //     solve(pt,jac_trap, indptr, indices, N, fg,Dxy, flag);
    // }

//     return 2;
// }

int run( int * pt,double t, double t_end, double *jac_trap,int *indptr,int *indices,double *x,double *y,double *xy,double *u,double *p,int N_x,int N_y,int max_it, double itol, int * its, double Dt, double *z, double *dblparams, int *intparams, double * Time, double * X, double * Y, double * Z, int N_z, int N_store)
{
    mkl_verbose(0);
    int i;
    double norma;
    double norm_max;
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

            if (intparams[0] == 0) { // factorization is always computed
            flag = 10;
            solve(pt,jac_trap, indptr, indices, N, fg,Dxy, flag); 
            }
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
            norm_max = 0.0;
            for (i = 0; i < (N_y+N_x); i++)
            {
                norma = fg[i]*fg[i];
                if(norma>norm_max)
                {norm_max = norma;}
            } 
            if (norm_max < itol) {     
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

    if (intparams[0] == 1) // factorization is only computed in the first iteration
    { 
        flag = 10;
        solve(pt,jac_trap, indptr, indices, N, fg,Dxy, flag); 
    }

    intparams[2] = it;
    dblparams[0] = t;

    return 0;
}

int step3( int * pt,double t, double t_end, double *jac_trap,int *indptr,int *indices,double *x,double *y,double *xy,double *u,double *p,int N_x,int N_y,int max_it, double itol, int * its, double Dt, double *z, double *dblparams, int *intparams, double * Time, double * X, double * Y, double * Z, int N_z, int N_store)
{
    mkl_verbose(0);
    int i;
    double norma;
    double norm_max;
    int N;
    int flag;
    int it = 0;
    int it_max = 0; // maximum number of algebraic iterations
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

            if (intparams[0] == 0) { // factorization is always computed
            flag = 10;
            solve(pt,jac_trap, indptr, indices, N, fg,Dxy, flag); 
            }
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
            norm_max = 0.0;
            for (i = 0; i < (N_y+N_x); i++)
            {
                norma = fg[i]*fg[i];
                if(norma>norm_max)
                {norm_max = norma;}
            } 
            if (norm_max < itol) {     
                break; 
            }
        }

    if (it > it_max)
    {it_max = it;}
    
    if (intparams[1] == 0)
    {
        h_eval(z,x,y,u,p,Dt);

        // Time[its[0]] = t;     

        // for (i = 0; i < N_x; i++)
        // {
        //     X[its[0]*N_x+i] = x[i];     
        // }    
        // for (i = 0; i < N_y; i++)
        // {
        //     Y[its[0]*N_y+i] = y[i];     
        // }  
        // for (i = 0; i < N_z; i++)
        // {
        //     Z[its[0]*N_z+i] = z[i];     
        // }   
    }
    its[0] += 0;
    }    

    free(f);
    free(g);
    free(fg);
    free(x_0);
    free(f_0);
    free(Dxy);

    if (intparams[0] == 1) // factorization is only computed in the first iteration
    { 
        flag = 10;
        solve(pt,jac_trap, indptr, indices, N, fg,Dxy, flag); 
    }

    intparams[2] = it;
    intparams[5] = it_max;
    dblparams[0] = t;

    return 0;
}