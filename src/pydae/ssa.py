#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 20:43:46 2018

@author: jmmauricio
"""

import numpy as np
import pandas as pd
import scipy

def eval_A(system):
    
    Fx = system.struct[0].Fx
    Fy = system.struct[0].Fy
    Gx = system.struct[0].Gx
    Gy = system.struct[0].Gy
    
    A = Fx - Fy @ np.linalg.solve(Gy,Gx)
    
    system.A = A
    
    return A

def eval_ss(system):
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
    Ddx = Fx*Dx + Fy*Dy + Fu*Du
      0 = Gx*Dx + Gy*Dy + Gu*Du
      z = Hx*Dx + Hy*Dy + Hu*Du
    
    Dy = -inv(Gy)*Gx*Dx - inv(Gy)*Gu*Du
                                     
    Ddx = Fx*Dx - Fy*inv(Gy*Gx)*Dx - Fy*inv(Gy)*Gu*Du + Fu*Du           
    Ddx = (Fx - Fy*inv(Gy*Gx))*Dx + (Fu - Fy*inv(Gy)*Gu)*Du
    

    Dz = Hx*Dx + Hy*Dy + Hu*Du
    Dz = Hx*Dx - Hy*inv(Gy)*(Gx*Dx) - Hy*inv(Gy)*Gu*Du + Hu*Du
    Dz = (Hx - Hy*inv(Gy)*(Gx))*Dx + (Hu - Hy*inv(Gy)*Gu)*Du


    '''
    Fx = system.struct[0].Fx
    Fy = system.struct[0].Fy
    Gx = system.struct[0].Gx
    Gy = system.struct[0].Gy
    
    Fu = system.struct[0].Fu
    Gu = system.struct[0].Gu    
    
    Hx = system.struct[0].Hx
    Hy = system.struct[0].Hy  
    Hu = system.struct[0].Hu 
    
    A = Fx - Fy @ np.linalg.solve(Gy,Gx)
    B = Fu - Fy @ np.linalg.solve(Gy,Gu)
    C = Hx - Hy @ np.linalg.solve(Gy,Gx)
    D = Hu - Hy @ np.linalg.solve(Gy,Gu)
    
    system.A = A
    system.B = B
    system.C = C
    system.D = D
    
    return A



def eval_A_ini(system):
    
    Fx = system.struct[0].Fx_ini
    Fy = system.struct[0].Fy_ini
    Gx = system.struct[0].Gx_ini
    Gy = system.struct[0].Gy_ini
    
    A = Fx - Fy @ np.linalg.solve(Gy,Gx)
     
    system.A = A
    
    return A


def damp_report(system):
    eig,eigv = np.linalg.eig(system.A)
    omegas = eig.imag
    sigmas = eig.real

    freqs = np.abs(omegas/(2*np.pi))
    zetas = -sigmas/np.sqrt(sigmas**2+omegas**2)
    
    string = ''
    string += f' Mode'.ljust(10, ' ') 
    string += f' Real'.ljust(10, ' ') 
    string += f' Imag'.ljust(10, ' ') 
    string += f' Freq.'.ljust(10, ' ') 
    string += f' Damp'.ljust(10, ' ') 
    string += '\n'

    N_x = len(eig)
    for it in range(N_x):
        r = eig[it].real
        i = eig[it].imag
        string += f'{it+1:0d}  '.rjust(10, ' ') 
        string += f'{r:0.4f}  '.rjust(10, ' ') 
        string += f'{i:0.4f}j'.rjust(10, ' ') 
        string += f'{freqs[it]:0.4f}'.rjust(10, ' ') 
        string += f'{zetas[it]:0.4f}'.rjust(10, ' ') 

        string += '\n'
        #'\t{i:0.4f}\t{freqs[it]:0.3f}\t{zetas[it]:0.4f}\n'
    
    columns = ['Real','Imag','Freq.','Damp']     
    modes = [f'Mode {it+1}' for it in range(N_x)]
    eig_df = pd.DataFrame(data={'Real':eig.real,'Imag':eig.imag,  'Freq.':freqs,     'Damp':zetas},index=modes)
    
    return eig_df
 

def participation(system, method='kundur'):
    '''
    As in PSAT but with the notation from Kundur
    
    ''' 
    
    eig,V = np.linalg.eig(system.A)
    W = np.linalg.inv(V)
    
    N_x = len(eig)
    
    participation_matrix = np.zeros((N_x,N_x))
    
    if method=='milano':
        for i in range(N_x):
            for j in range(N_x):
    
                den = np.sum(np.abs(W[j,:])*np.abs(V[:,j]))
                participation_matrix[i,j] =  np.abs(W[i,j])*np.abs(V[j,i])/den
                
        
        modes = [f'Mode {it+1}' for it in range(N_x)]
        participation_df = pd.DataFrame(data=participation_matrix,index=modes, columns=system.x_list)

    if method=='milano_transpose':
        for i in range(N_x):
            for j in range(N_x):
    
                den = np.sum(np.abs(W[j,:])*np.abs(V[:,j]))
                participation_matrix[j,i] =  np.abs(W[i,j])*np.abs(V[j,i])/den
                
        
        modes = [f'Mode {it+1}' for it in range(N_x)]
        participation_df = pd.DataFrame(data=participation_matrix,index=system.x_list, columns=modes)

    if method=='kundur':
        Phi = V
        Psi = W
        
        participation_matrix =  Phi * Psi.T
                
        
        modes = [f'Mode {it+1}' for it in range(N_x)]
        participation_df = pd.DataFrame(data=participation_matrix,index=system.x_list, columns=modes)

    
    return participation_df


def shape2df(system):
    
    eig,V = np.linalg.eig(system.A)
    W = np.linalg.inv(V)
    N_x = W.shape[0]

    participation_matrix = np.zeros((N_x,N_x))

    for i in range(N_x):
        for j in range(N_x):
            den = np.sum(np.abs(W[j,:])*np.abs(V[:,j]))
            participation_matrix[i,j] =  np.abs(W[i,j])*np.abs(V[j,i])/den
            
    modules = np.abs(participation_matrix)
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

    df = pd.DataFrame(data=W_list,index=modes, columns=system.x_list)
    
    return df
    # for it in range(N_x):
    #     r = eig[it].real
    #     i = eig[it].imag
    #     string += f'{r:0.4f}  '.rjust(10, ' ') 
    #     string += f'{i:0.4f}j'.rjust(10, ' ') 
    #     string += f'{freqs[it]:0.4f}'.rjust(10, ' ') 
    #     string += f'{zetas[it]:0.4f}'.rjust(10, ' ') 

    #     string += '\n'
    #     #'\t{i:0.4f}\t{freqs[it]:0.3f}\t{zetas[it]:0.4f}\n'
        
    # return string
    

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
