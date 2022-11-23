#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 20:43:46 2018

@author: jmmauricio
"""

import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import csc_matrix

from matplotlib.patches import Circle, Wedge, Polygon, Rectangle


def A_eval(system):
    system.jac_run_eval()
    N_x = system.N_x
    Fx = system.jac_run[:N_x,:N_x]
    Fy = system.jac_run[:N_x,N_x:]
    Gx = system.jac_run[N_x:,:N_x]
    Gy = system.jac_run[N_x:,N_x:]
    
    A = Fx - Fy @ np.linalg.solve(Gy,Gx)
    
    system.A = A
    
    return A


def ss_eval(model):
    '''
    
    Parameters
    ----------
    system : system class
        object.

    Returns
    -------
    A : np.array
        System A matrix.

    # DAE system        
    dx = f(x,y,u)
     0 = g(x,y,u)
     z = h(x,y,u)
     
    # system linealization
    Δdx = Fx*Δx + Fy*Δy + Fu*Δu
      0 = Gx*Δx + Gy*Δy + Gu*Δu
     Δz = Hx*Δx + Hy*Δy + Hu*Δu
    
    Δy = -inv(Gy)*Gx*Dx - inv(Gy)*Gu*Du
                                     
    Δdx = Fx*Dx - Fy*inv(Gy*Gx)*Δx - Fy*inv(Gy)*Gu*Δu + Fu*Δu           
    Δdx = (Fx - Fy*inv(Gy*Gx))*Δx + (Fu - Fy*inv(Gy)*Gu)*Δu
    

    Δz = Hx*Dx + Hy*Δy + Hu*Δu
    Δz = Hx*Dx - Hy*inv(Gy)*(Gx*Δx) - Hy*inv(Gy)*Gu*Du + Hu*Δu
    Δz = (Hx - Hy*inv(Gy)*(Gx))*Δx + (Hu - Hy*inv(Gy)*Gu)*Δu


    '''
    
    model.full_jacs_eval()

    Fx = model.Fx
    Fy = model.Fy
    Gx = csc_matrix(model.Gx)
    Gy = csc_matrix(model.Gy)

    Fu = model.Fu
    Gu = csc_matrix(model.Gu)  

    Hx = model.Hx
    Hy = model.Hy  
    Hu = model.Hu 

    A = Fx - Fy @ spsolve(Gy,Gx)
    B = Fu - Fy @ spsolve(Gy,Gu)
    C = Hx - Hy @ spsolve(Gy,Gx)
    D = Hu - Hy @ spsolve(Gy,Gu)

    model.A = A
    model.B = B
    model.C = C
    model.D = D

    return A



def eval_A_ini(system):
    
    Fx = system.struct[0].Fx_ini
    Fy = system.struct[0].Fy_ini
    Gx = system.struct[0].Gx_ini
    Gy = system.struct[0].Gy_ini
    
    A = Fx - Fy @ np.linalg.solve(Gy,Gx)
     
    system.A = A
    
    return A


def lqr(A,B,Q,R):
    """Solve the continuous time lqr controller.
     
    dx/dt = A x + B u
     
    cost = integral x.T*Q*x + u.T*R*u
    
    https://www.mwm.im/lqr-controllers-with-python/
    """
    #ref Bertsekas, p.151
     
    #first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_continuous_are(A, B, Q, R))
     
    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(R)*(B.T*X))
     
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
     
    return K, X, eigVals



    
def dlqr(A,B,Q,R):
    """Solve the discrete time lqr controller.
     
    x[k+1] = A x[k] + B u[k]
     
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    
    https://www.mwm.im/lqr-controllers-with-python/
    """
    #ref Bertsekas, p.151
     
    #first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
     
    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A))
     
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
     
    return K, X, eigVals


def discretise_time(A, B, dt):
    '''Compute the exact discretization of the continuous system A,B.
    
    Goes from a description 
     d/dt x(t) = A*x(t) + B*u(t)
     u(t)  = ud[k] for t in [k*dt, (k+1)*dt)
    to the description
     xd[k+1] = Ad*xd[k] + Bd*ud[k]
    where
     xd[k] := x(k*dt)
     
    Returns: Ad, Bd
    
    from: https://github.com/markwmuller/controlpy/blob/master/controlpy/analysis.py
    '''
    
    nstates = A.shape[0]
    ninputs = B.shape[1]

    M = np.matrix(np.zeros([nstates+ninputs,nstates+ninputs]))
    M[:nstates,:nstates] = A
    M[:nstates, nstates:] = B
    
    Md = scipy.linalg.expm(M*dt)
    Ad = Md[:nstates, :nstates]
    Bd = Md[:nstates, nstates:]

    return Ad, Bd


def acker(A,B,poles):
    '''
    This function is a copy from the original in: https://github.com/python-control/python-control
    but it allows to work with complex A and B matrices. It is experimental and the original should be
    considered
    
    
    ----------
    A : numpy array_like (complex can be used)
        Dynamics amatrix of the system
    B : numpy array_like (complex can be used)
        Input matrix of the system
    poles : numpy array_like
        Desired eigenvalue locations.

    Returns
    -------
    K : numpy array_like
        Gain such that A - B K has eigenvalues given in p.

    '''
    
    
    N_x = np.shape(A)[0]
    
    ctrb = np.hstack([B] + [np.dot(np.linalg.matrix_power(A, i), B)
                                             for i in range(1, N_x)])

    # Compute the desired characteristic polynomial
    p = np.real(np.poly(poles))

    n = np.size(p)
    pmat = p[n-1] * np.linalg.matrix_power(A, 0)
    for i in np.arange(1,n):
        pmat = pmat + np.dot(p[n-i-1], np.linalg.matrix_power(A, i))
    K = np.linalg.solve(ctrb, pmat)

    K = K[-1][:]                # Extract the last row            # Extract the last row
    
    return K    

def right2df(system):
    
    eig,V = np.linalg.eig(system.A)
    W = np.linalg.inv(V)
    N_x = W.shape[0]

    modules = np.abs(V)
    degs = np.rad2deg(np.angle(V))
    W_row = []
    W_list =[]
    N_x = W.shape[0]
    for irow in range(N_x):
        W_row = []
        for icol in range(N_x):
            W_row += [f'{modules[icol,irow]:0.2f}∠{degs[icol,irow]:5.1f}']
        W_list += [W_row]

    modes = [f'Mode {it+1}' for it in range(N_x)]

    df_report = pd.DataFrame(data=W_list,index=modes, columns=system.x_list)
    df = pd.DataFrame(data=modules*np.exp(1j*np.angle(W)),index=modes, columns=system.x_list)
    
    system.shape_modules = modules
    system.shape_degs = degs
    system.modes_id = modes
    system.df = df
    
    return df_report

def left2df(system):
    
    eig,V = np.linalg.eig(system.A)
    W = np.linalg.inv(V)
    N_x = W.shape[0]

    modules = np.abs(W)
    degs = np.rad2deg(np.angle(W))
    W_row = []
    W_list =[]
    N_x = W.shape[0]
    for irow in range(N_x):
        W_row = []
        for icol in range(N_x):
            W_row += [f'{modules[irow,icol]:0.2f}∠{degs[irow,icol]:5.1f}']
        W_list += [W_row]

    modes = [f'Mode {it+1}' for it in range(N_x)]

    df_report = pd.DataFrame(data=W_list,index=modes, columns=system.x_list)
    df = pd.DataFrame(data=modules*np.exp(1j*np.angle(W)),index=modes, columns=system.x_list)
    
    system.shape_modules = modules
    system.shape_degs = degs
    system.modes_id = modes
    system.df = df
    
    return df_report


def pi_design(a,b,zeta,omega):
    '''
    G = 1/(a*s + b)    
    PI = (K_p + K_i/s) = (K_p*s + K_i)/s -> tf([K_p,K_i],[1,0])
    K_i = K_p/T_p
    T_p = K_p/K_i
    
    Example:
    
    L = 4e-3
    R = 0.1

    K_p,K_i = pi_design(L,R,1.0,2*np.pi*10)

    G_planta = ctrl.tf([1], [L,R])
    G_pi = ctrl.tf([K_p,K_i],[1,0])

    H = G_planta*G_pi/(1 + G_planta*G_pi)
    ctrl.bode_plot(H);
    
    '''
    
    K_p = 2*zeta*omega*a-b
    K_i = a*omega**2
    
    return K_p,K_i

def lead_design(angle,omega, verbose=False):
    '''
    Desing lead lag compensator: G = (T_1*s + 1)/(T_2*s + 1)
    
    alpha = (1+np.sin(angle))/(1-np.sin(angle))
    
    T_2 = 1/(omega*np.sqrt(alpha))
    T_1 = alpha*T_2
    

    G_ll = ctrl.tf([T_1,1],[T_2,1])

    ctrl.bode_plot(G_ll);
    
    '''
    
    alpha = (1+np.sin(angle))/(1-np.sin(angle))
    T_2 = 1/(omega*np.sqrt(alpha))
    T_1 = alpha*T_2
    
    if verbose:
        G_omega = (T_1*1j*omega+1)/(T_2*1j*omega+1)
        print(f'Gain at omega: {np.abs(G_omega)}')
    
    return T_1,T_2
    
     
if __name__ == "__main__":
    
    class sys_class():
        
        def __init__(self): 
     
            
            self.A = np.array([[ 0.00000000e+00,  3.14159265e+02,  0.00000000e+00,
                 0.00000000e+00,  0.00000000e+00],
               [-4.43027144e-01, -1.42857143e-01, -1.44867645e-01,
                 1.69999669e-01,  0.00000000e+00],
               [-1.80455968e-01,  0.00000000e+00, -6.64265905e-01,
                -2.31113959e-03, -2.50000000e+01],
               [ 1.32845727e+00,  0.00000000e+00,  1.35913375e-02,
                -2.58565604e+00,  0.00000000e+00],
               [-3.90435895e-01,  0.00000000e+00,  2.69427382e+00,
                 4.81098835e-01, -2.00000000e+01]])


            self.A = np.array([[      0,  314.1593,        0,         0,         0,         0],
                               [-0.1594,         0,        0,         0,   -0.1730,    0.0600],
                               [-0.2258,         0,  -0.0857,         0,   -0.2669,   -0.0011],
                               [ 0.4611,         0,        0,   -1.0000,    0.0063,   -1.4894],
                               [-2.9728,         0,  33.3333,         0,  -36.8474,   -0.0145],
                               [ 2.6216,         0,        0,   14.2857,    0.0360,  -22.7542]])
            
            self.A = np.array([[0,314.16,0,0,0,0],
                                [-0.15935,0,0,0,-0.17302,0.060008],
                                [-0.22579,0,-0.085744,0,-0.26689,-0.0011041],
                                [0.46108,0,0,-1,0.0063364,-1.4894],
                                [-2.9728,0,33.333,0,-36.847,-0.014537],
                                [2.6216,0,0,14.286,0.036027,-22.754]])
            
            self.x_list = ['delta_Syn_1','omega_Syn_1','e1q_Syn_1','e1d_Syn_1','e2q_Syn_1','e2d_Syn_1']

            
            # Example 12.4 (AVR+PSS):
            self.A = np.array([[     0, -0.1092, -0.1236, 0, 0, 0],
                               [376.99,       0,       0, 0, 0, 0],
                               [     0, -0.1938, -0.4229, -27.3172, 0, 27.3172],
                               [      0,-7.3125,   20.8391, -50.0, 0,0],
                               [     0,-1.0372, -1.1738,0, -0.7143,0],
                               [     0,-4.8404, -5.4777,0,26.9697, -30.3030]])
           
            # Example 12.4 (AVR+PSS):
            self.x_list = ['Domega_r','Ddelta','Dpsi_fd','Dv_1','Dv_2','Dv_s']
            

    sys = sys_class()
    print(damp_report(sys))
    PF = participation(sys)
    df_shape = shape2df(sys)

    
'''    
STATE MATRIX EIGENVALUES

Eigevalue      Most Associated States      Real part      Imag. Part     Pseudo-Freq.   Frequency      

Eig As #1      e2q_Syn_1                  -36.4944        0              0              0             
Eig As #2      e2d_Syn_1                  -21.6378        0              0              0             
Eig As #3      omega_Syn_1, delta_Syn_1   -0.25606        6.633          1.0557         1.0565        
Eig As #4      omega_Syn_1, delta_Syn_1   -0.25606       -6.633          1.0557         1.0565        
Eig As #5      e1d_Syn_1                  -1.9568         0              0              0             
Eig As #6      e1q_Syn_1                  -0.08604        0              0              0             

PARTICIPATION FACTORS (Euclidean norm)

               delta_Syn_1    omega_Syn_1    e1q_Syn_1      e1d_Syn_1      e2q_Syn_1      

Eig As #1      0.00283        0.00283        0.00641        0              0.98792       
Eig As #2      0.00344        0.00344        1e-05          0.04673        0.0001        
Eig As #3      0.4805         0.4805         0.01663        0.00586        0.0091        
Eig As #4      0.4805         0.4805         0.01663        0.00586        0.0091        
Eig As #5      0.0023         0.0023         0.00474        0.94518        7e-05         
Eig As #6      0.0005         0.0005         0.99273        0.00576        1e-05         

PARTICIPATION FACTORS (Euclidean norm)

               e2d_Syn_1      

Eig As #1      2e-05         
Eig As #2      0.94629       
Eig As #3      0.00741       
Eig As #4      0.00741       
Eig As #5      0.04542       
Eig As #6      0.00051       

STATISTICS

DYNAMIC ORDER                 6             
# OF EIGS WITH Re(mu) < 0     6             
# OF EIGS WITH Re(mu) > 0     0             
# OF REAL EIGS                4             
# OF COMPLEX PAIRS            1             
# OF ZERO EIGS                0 
'''

['delta_Syn_1','omega_Syn_1','e1q_Syn_1','e1d_Syn_1','e2q_Syn_1','e2d_Syn_1']
