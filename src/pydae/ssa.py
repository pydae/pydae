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

from matplotlib.patches import Circle, Wedge, Polygon, Rectangle

def eval_A(system):
    
    Fx = system.struct[0].Fx
    Fy = system.struct[0].Fy
    Gx = system.struct[0].Gx
    Gy = system.struct[0].Gy
    
    A = Fx - Fy @ np.linalg.solve(Gy,Gx)
    
    system.A = A
    
    return A

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
    
    system.eigvalues = eig
    system.eigvectors = eigv
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

    df_shapes_report = pd.DataFrame(data=W_list,index=modes, columns=system.x_list)
    df_shapes = pd.DataFrame(data=modules*np.exp(1j*np.angle(W)),index=modes, columns=system.x_list)
    
    system.participation_matrix = participation_matrix
    system.shape_modules = modules
    system.shape_degs = degs
    system.modes_id = modes
    system.df_shapes = df_shapes
    
    return df_shapes_report
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



def plot_eig(grid, x_min='',x_max='',y_min='',y_max=''):
    fig,axes = plt.subplots()

    
    pi2 = 2*np.pi

    #damping zones
    ## damping zone > 10%
    e_real = -10000.0
    damp = 0.1
    e_imag = np.sqrt((e_real/damp)**2-e_real**2)/pi2
    poly = Polygon([(0,0),(e_real,0),(e_real,e_imag)], facecolor='#e1f2e1',zorder=0)
    axes.add_patch(poly)
    #axes.text(-0.85, 0.5, '$\zeta>10\%$', fontsize=12)

    ## damping zone from  5% to 10%
    axes.plot([0,e_real],[0, e_imag],'--', color='k',zorder=0)
    axes.plot([0,e_real],[0,-e_imag],'--', color='k',zorder=0)
    poly = Polygon([(0,0),(e_real,e_imag),(0,e_imag)], facecolor='#fae8ce',zorder=0)
    axes.add_patch(poly)
    #axes.text(-0.87, 9.0/pi2, '$10\% <\zeta<5\%$', fontsize=12)

    ## damping zone <5%
    e_real = -100000.0
    damp = 0.05
    e_imag = np.sqrt((e_real/damp)**2-e_real**2)/pi2
    axes.plot([0,e_real],[0, e_imag],'--', color='k',zorder=0)
    axes.plot([0,e_real],[0,-e_imag],'--', color='k',zorder=0)
    poly = Polygon([(0,0),(e_real,e_imag),(0,e_imag)], facecolor='#f7dcda',zorder=0)
    axes.add_patch(poly)
    #axes.text(-0.37, 9.0/pi2, '$\zeta<$5 %', fontsize=12)



    axes.plot([0,0],[-20,20],'-', color='k', lw=4)

    if x_min == '': x_min = np.min(grid.eigvalues.real)*1.1
    if x_max == '': x_max = np.max(grid.eigvalues.real)
    if y_min == '': y_min = np.min(grid.eigvalues.imag)/(2*np.pi)*1.1
    if y_max == '': y_max = np.max(grid.eigvalues.imag)/(2*np.pi)*1.1
        
    axes.set_xlim((x_min,x_max))
    axes.set_ylim((y_min,y_max))
    axes.grid(True)
    axes.set_xlabel('Real')
    axes.set_ylabel('Imag$/2\pi$ (Hz)')

    axes.plot(grid.eigvalues.real,grid.eigvalues.imag/(2*np.pi),'o')
    fig.tight_layout()
    
    return fig
    
def add_arrow(string,name='arrow',scale=1,angle=0,center_x=0,center_y=0,color="#337ab7"):
    trans_x = center_x*(1-scale)
    trans_y = center_y*(1-scale)
    string += f'<path d="M {center_x},{center_y} h 1.32292 v -52.916665 h 1.32291 l -2.64583,-5.291666 -2.64583,5.291666 h 1.32291 v 52.916665 z" ' 
    string += f'fill="{color}" id="arrow_1" class="tooltip-trigger" data-tooltip-text="{name}"  fill-opacity="0.5" '
    string += f'stroke-width="0" transform="translate({trans_x},{trans_y}) scale({scale})  rotate({angle},{center_x},{center_y})" '   
    string += f'/>' 
    return string


def plot_shapes(grid,mode='Mode 13',states=['omega_1','omega_1','omega_3','omega_4'], plot_scale=3):
    svg_arrows = ''
    shapes = grid.df_shapes.loc[mode][states]

    max_module = shapes.abs().max()
    for shape,name in zip(shapes,shapes.index):
        module = np.abs(shape)
        angle = np.angle(shape,deg=True)
        svg_arrows = add_arrow(svg_arrows,name=name,scale=plot_scale*module/max_module,angle=angle,center_x=200,center_y=200)
        
    # from: https://www.petercollingridge.co.uk/tutorials/svg/interactive/tooltip/    
    svg_string = ''' 
        <svg xmlns="http://www.w3.org/2000/svg" 
        height="400"  width="400"
        id="tooltip-svg-5">
            <style>
                #tooltip {
                    dominant-baseline: hanging; 
                }
            </style>
            
            <circle cx="200" cy="200" r="190" stroke="#888888" stroke-width="3" fill="#DDDDDD" />

        {arrows}    
        
        <g id="tooltip" visibility="hidden" >

                <rect width="80" height="24" fill="white" rx="2" ry="2"/>
                <text x="3" y="6">Tooltip</text>
            </g>


            <script type="text/ecmascript"><![CDATA[
                (function() {
                    var svg = document.getElementById('tooltip-svg-5');
                    var tooltip = svg.getElementById('tooltip');
                    var tooltipText = tooltip.getElementsByTagName('text')[0].firstChild;
                    var triggers = svg.getElementsByClassName('tooltip-trigger');

                    for (var i = 0; i < triggers.length; i++) {
                        triggers[i].addEventListener('mousemove', showTooltip);
                        triggers[i].addEventListener('mouseout', hideTooltip);
                    }

                    function showTooltip(evt) {
                        var CTM = svg.getScreenCTM();
                        var x = (evt.clientX - CTM.e + 6) / CTM.a;
                        var y = (evt.clientY - CTM.f + 20) / CTM.d;
                        tooltip.setAttributeNS(null, "transform", "translate(" + x + " " + y + ")");
                        tooltip.setAttributeNS(null, "visibility", "visible");
                        tooltipText.data = evt.target.getAttributeNS(null, "data-tooltip-text");
                    }

                    function hideTooltip(evt) {
                        tooltip.setAttributeNS(null, "visibility", "hidden");
                    }
                })()
            ]]></script>
        </svg>
        '''
    
    svg_string = svg_string.replace('{arrows}',svg_arrows)
    
    return svg_string
    

def plot_vectors(vectors,names,colors, plot_scale=3):
    svg_arrows = ''

    for vector,name,color in zip(vectors,names,colors):
        module = np.abs(vector)
        angle = np.angle(vector,deg=True)
        svg_arrows = add_arrow(svg_arrows,name=name,scale=plot_scale*module,angle=angle,center_x=200,center_y=200,color=color)
        
    # from: https://www.petercollingridge.co.uk/tutorials/svg/interactive/tooltip/    
    svg_string = ''' 
        <svg xmlns="http://www.w3.org/2000/svg" 
        height="400"  width="400"
        id="tooltip-svg-5">
            <style>
                #tooltip {
                    dominant-baseline: hanging; 
                }
            </style>
            
            <circle cx="200" cy="200" r="190" stroke="#888888" stroke-width="3" fill="#DDDDDD" />

        {arrows}    
        
        <g id="tooltip" visibility="hidden" >

                <rect width="80" height="24" fill="white" rx="2" ry="2"/>
                <text x="3" y="6">Tooltip</text>
            </g>


            <script type="text/ecmascript"><![CDATA[
                (function() {
                    var svg = document.getElementById('tooltip-svg-5');
                    var tooltip = svg.getElementById('tooltip');
                    var tooltipText = tooltip.getElementsByTagName('text')[0].firstChild;
                    var triggers = svg.getElementsByClassName('tooltip-trigger');

                    for (var i = 0; i < triggers.length; i++) {
                        triggers[i].addEventListener('mousemove', showTooltip);
                        triggers[i].addEventListener('mouseout', hideTooltip);
                    }

                    function showTooltip(evt) {
                        var CTM = svg.getScreenCTM();
                        var x = (evt.clientX - CTM.e + 6) / CTM.a;
                        var y = (evt.clientY - CTM.f + 20) / CTM.d;
                        tooltip.setAttributeNS(null, "transform", "translate(" + x + " " + y + ")");
                        tooltip.setAttributeNS(null, "visibility", "visible");
                        tooltipText.data = evt.target.getAttributeNS(null, "data-tooltip-text");
                    }

                    function hideTooltip(evt) {
                        tooltip.setAttributeNS(null, "visibility", "hidden");
                    }
                })()
            ]]></script>
        </svg>
        '''
    
    svg_string = svg_string.replace('{arrows}',svg_arrows)
    
    return svg_string

    
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
