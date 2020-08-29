#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 11:53:28 2018

@author: jmmauricio

bug: if there is not jacobian add pass to the if mode == 12:
"""
import numpy as np
import sympy as sym
import os
from collections import deque 
import pkgutil

def sym_gen_str():

    str = '''\
import sympy as sym

if 'params' in sys_vars: params = sys_vars['params']
if 'params_dict' in sys_vars: params = sys_vars['params_dict']
u_list = sys_vars['u_list']
x_list = sys_vars['x_list']
y_list = sys_vars['y_list']
for item in params:
    exec(f"{item} = sym.Symbol('{item}')", globals())
for item in u_list:
    exec(f"{item} = sym.Symbol('{item}')", globals())
for item in x_list:
    exec(f"{item} = sym.Symbol('{item}')", globals())
for item in y_list:
    exec(f"{item} = sym.Symbol('{item}')", globals())  
sym_func_list = ['sin','cos']
for item in sym_func_list:
    exec(f"{item} = sym.{item}", globals()) 
    '''
    return str

def sym_gen(sys_vars):
    """Generates global sympy symbolic variables from a dictionary.
    
    Parameters
    ----------
    sys_vars : A ``dict`` with the following keys:
    
    params: ``dict``: Dictionary of the parameters and their values.
    u_list: ``list(str)``: List of input variables as strings for the running problem. 
    x_list: ``list(str)``: List of dynamical states as strings for the running problem.
    y_list: ``list(str)``: List of algebraic states as strings for the running problem.   

    R = 0.1
    L = 0.01
    T_pi = L/R
    K_p = L/0.01

    params= {'R' : 0.1, 'L':0.01, 'V_max':10.0, 'V_min':0.0, 'K_p':1.0, 'T_pi':0.01, 'K_aw':1.0}
    u_ini_dict = { 'i_ref':1.0}  # for the initialization problem
    u_run_dict = { 'i_ref':1.0}  # for the running problem (here initialization and running problem are the same)

    x_list = ['i','xi']
    y_ini_list = ['y','v']
    y_run_list = ['y','v']

    sys_vars = {'params':params,
                'u_list':u_run_dict,
                'x_list':x_list,
                'y_run_list':y_run_list}   
    """    

    params = sys_vars['params_dict']
    u_list = sys_vars['u_list']
    x_list = sys_vars['x_list']
    y_list = sys_vars['y_list']

    for item in params:
        exec(f"{item} = sym.Symbol('{item}', real=True)", globals())
    for item in u_list:
        exec(f"{item} = sym.Symbol('{item}', real=True)", globals())
    for item in x_list:
        exec(f"{item} = sym.Symbol('{item}', real=True)", globals())
    for item in y_list:
        exec(f"{item} = sym.Symbol('{item}', real=True)", globals())  
    sym_func_list = ['sin','cos']
    for item in sym_func_list:
        exec(f"{item} = sym.{item}", globals()) 


    return sys_vars

    
def check_system(sys):
    
    if len(sys['f_list']) == 0:
        print('system without dynamic equations, adding dummy dynamic equation')
        x_dummy,u_dummy = sym.symbols('x_dummy,u_dummy')
        sys['x_list'] = ['x_dummy']
        sys['f_list'] = [u_dummy-x_dummy]
        sys['u_ini_dict'].update({'u_dummy':1.0})
        sys['u_run_dict'].update({'u_dummy':1.0})

    if len(sys['g_list']) == 0:
        print('system without algebraic equations, adding dummy algebraic equation')
        y_dummy,u_dummy = sym.symbols('y_dummy,u_dummy')
        
        sys['g_list'] = [u_dummy-y_dummy]
        sys['y_ini_list'] = ['y_dummy']
        sys['y_run_list'] = ['y_dummy']
        sys['u_ini_dict'].update({'u_dummy':1.0})
        sys['u_run_dict'].update({'u_dummy':1.0})

       
def system(sys):
    '''
    

    Parameters
    ----------
    sys : dict
           {'name':sys_name,
            'params_dict':params_dict,
            'f_list':f_list,
            'g_list':g_list,
            'x_list':x_list,
            'y_ini_list':y_ini_list,
            'y_run_list':y_run_list,
            'u_run_dict':u_run_dict,
            'u_ini_dict':u_ini_dict,
            'h_dict':h_dict}

    Returns
    -------
    sys : TYPE
        DESCRIPTION.

    '''
    check_system(sys)
    
    f = sym.Matrix(sys['f_list']).T
    g = sym.Matrix(sys['g_list']).T
    x = sym.Matrix(sys['x_list']).T
    y_ini = sym.Matrix(sys['y_ini_list']).T
    y_run = sym.Matrix(sys['y_run_list']).T
    u_ini = sym.Matrix(list(sys['u_ini_dict'].keys()), real=True)
    
    u_run_list = [sym.Symbol(item,real=True) for item in list(sys['u_run_dict'].keys())]
    u_run = sym.Matrix(u_run_list).T 
    h =  sym.Matrix(list(sys['h_dict'].values())).T     

    Fx_run = f.jacobian(x)
    Fy_run = f.jacobian(y_run)
    Gx_run = g.jacobian(x)
    Gy_run = g.jacobian(y_run)

    Fu_run = f.jacobian(u_run)
    Gu_run = g.jacobian(u_run)
    
    Fx_ini = f.jacobian(x)
    Fy_ini = f.jacobian(y_ini)
    Gx_ini = g.jacobian(x)
    Gy_ini = g.jacobian(y_ini)
    
    Hx_run = h.jacobian(x)
    Hy_run = h.jacobian(y_run)   
    Hu_run = h.jacobian(u_run)

    sys['f']= f
    sys['g']= g 
    sys['x']= x
    sys['y_ini']= y_ini  
    sys['y_run']= y_run      
    sys['u_ini']= u_ini 
    sys['u_run']= u_run
    sys['h']= h 
    
    sys['Fx_run'] = Fx_run
    sys['Fy_run'] = Fy_run
    sys['Gx_run'] = Gx_run
    sys['Gy_run'] = Gy_run

    sys['Fx_ini'] = Fx_ini
    sys['Fy_ini'] = Fy_ini
    sys['Gx_ini'] = Gx_ini
    sys['Gy_ini'] = Gy_ini
    
    sys['Fu_run'] = Fu_run
    sys['Gu_run'] = Gu_run

    sys['Hx_run'] = Hx_run
    sys['Hy_run'] = Hy_run
    sys['Hu_run'] = Hu_run
    
    return sys


def getIndex(s, i): 
  
    # If input is invalid. 
    if s[i] != '(':
        return -1
  
    # Create a deque to use it as a stack. 
    d = deque() 
  
    # Traverse through all elements 
    # starting from i. 
    for k in range(i, len(s)): 
  
        # Pop a starting bracket 
        # for every closing bracket 
        if s[k] == ')': 
            d.popleft() 
  
        # Push all starting brackets 
        elif s[k] == '(': 
            d.append(s[i]) 
  
        # If deque becomes empty 
        if not d: 
            return k 
  
    return -1
  
def arg2np(string,function_name):
    idx_end = 0
    for it in range(3):
        if function_name in string[idx_end:]:
            #print(string[idx_end:])
            idx_ini = string.find(f'{function_name}(',idx_end)+len(f'{function_name}(')
            idx_end = getIndex(string, idx_ini-1)
            string = string.replace(string[idx_ini:idx_end],f'np.array([{string[idx_ini:idx_end]}])')
    else: pass
    return string    

def sys2num(sys):
    
    params = sys['params_dict']
    h_dict = sys['h_dict']
        
    x = sys['x']
    y_ini = sys['y_ini']
    y_run = sys['y_run']

    u_ini = sys['u_ini']
    u_run = sys['u_run']

    f = sys['f']
    g = sys['g']   
    h =  sys['h']    

    Fx_run = sys['Fx_run']
    Fy_run = sys['Fy_run']
    Gx_run = sys['Gx_run']
    Gy_run = sys['Gy_run']

    Fx_ini = sys['Fx_ini']
    Fy_ini = sys['Fy_ini']
    Gx_ini = sys['Gx_ini']
    Gy_ini = sys['Gy_ini']

    Fu_run = sys['Fu_run'] 
    Gu_run = sys['Gu_run'] 
    
    Hx_run = sys['Hx_run'] 
    Hy_run = sys['Hy_run'] 
    Hu_run = sys['Hu_run']

    
    N_x = len(x)
    N_y = len(y_run)
    N_u = len(u_run)
    N_z = len(h)

    N_params = len(params)
    N_inputs = len(u_run)

    name = sys['name']

    run_fun = ''
    ini_fun = ''
    run_nn_fun = ''
    ini_nn_fun = ''
    
    numba_enable = True
    tab = '    '

    if numba_enable: run_fun += f'@numba.njit(cache=True)\n'
    run_fun += f'def run(t,struct,mode):\n\n'

    if numba_enable: ini_fun += f'@numba.njit(cache=True)\n'
    ini_fun += f'def ini(struct,mode):\n\n'

    run_fun += f'{tab}# Parameters:\n'
    ini_fun += f'{tab}# Parameters:\n'
    for item in params:
        run_fun += f'{tab}{item} = struct[0].{item}\n'
        ini_fun += f'{tab}{item} = struct[0].{item}\n'
    run_fun += f'{tab}\n'
    ini_fun += f'{tab}\n'

    run_fun += f'{tab}# Inputs:\n'
    ini_fun += f'{tab}# Inputs:\n'
    for item in u_run:
        run_fun += f'{tab}{item} = struct[0].{item}\n'

    for item in u_ini:    
        ini_fun += f'{tab}{item} = struct[0].{item}\n'    
    run_fun += f'{tab}\n'
    ini_fun += f'{tab}\n'

    run_fun += f'{tab}# Dynamical states:\n'
    ini_fun += f'{tab}# Dynamical states:\n'    
    for irow in range(N_x):
        run_fun += f'{tab}{x[irow]} = struct[0].x[{irow},0]\n'
        ini_fun += f'{tab}{x[irow]} = struct[0].x[{irow},0]\n'
    run_fun += f'{tab}\n'
    ini_fun += f'{tab}\n'
    
    run_fun += f'{tab}# Algebraic states:\n'
    ini_fun += f'{tab}# Algebraic states:\n' 
    for irow in range(N_y):
        run_fun += f'{tab}{y_run[irow]} = struct[0].y_run[{irow},0]\n'
        ini_fun += f'{tab}{y_ini[irow]} = struct[0].y_ini[{irow},0]\n'
    run_fun += f'{tab}\n'
    ini_fun += f'{tab}\n'

    # f: differential equations
    run_fun += f'{tab}# Differential equations:\n'
    run_fun += f'{tab}if mode == 2:\n\n'
    run_fun += f'\n'
    ini_fun += f'{tab}# Differential equations:\n'
    ini_fun += f'{tab}if mode == 2:\n\n'
    ini_fun += f'\n'
    for irow in range(N_x):
        string = f'{f[irow]}'
        run_fun += f'{2*tab}struct[0].f[{irow},0] = {arg2np(string,"Piecewise")}\n'
        ini_fun += f'{2*tab}struct[0].f[{irow},0] = {arg2np(string,"Piecewise")}\n'
    run_fun += f'{tab}\n'
    ini_fun += f'{tab}\n'

    ## g
    run_fun += f'{tab}# Algebraic equations:\n'
    run_fun += f'{tab}if mode == 3:\n\n'
    run_fun += f'\n'
    ini_fun += f'{tab}# Algebraic equations:\n'
    ini_fun += f'{tab}if mode == 3:\n\n'
    ini_fun += f'\n'
    for irow in range(N_y):
        string = f'{g[irow]}'
        run_fun += f'{2*tab}struct[0].g[{irow},0] = {arg2np(string,"Piecewise")}\n'
        string = f'{g[irow]}'
        ini_fun += f'{2*tab}struct[0].g[{irow},0] = {arg2np(string,"Piecewise")}\n'    
    run_fun += f'{tab}\n'
    ini_fun += f'{tab}\n'

    ## outputs
    run_fun += f'{tab}# Outputs:\n'
    run_fun += f'{tab}if mode == 3:\n'
    run_fun += f'\n'
    ini_fun += f'{tab}# Outputs:\n'
    ini_fun += f'{tab}if mode == 3:\n'
    ini_fun += f'\n'
    for irow in range(N_z):
        string = f'{h[irow]}'
        run_fun += f'{2*tab}struct[0].h[{irow},0] = {arg2np(string,"Piecewise")}\n'
        string = f'{h[irow]}'
        ini_fun += f'{2*tab}struct[0].h[{irow},0] = {arg2np(string,"Piecewise")}\n'    
    run_fun += f'{tab}\n'
    ini_fun += f'{tab}\n'
    

    ini_nn_fun = ini_fun    
    run_nn_fun = run_fun

    # jacobians
    ## Fx
    run_fun += f'\n'
    run_fun += f'{tab}if mode == 10:\n\n'
    ini_fun += f'\n'
    ini_fun += f'{tab}if mode == 10:\n\n'
    for irow in range(N_x):
        for icol in range(N_x):
            if not Fx_run[irow,icol].is_number:  # Fx_run = Fx_ini
                string = arg2np(f'{Fx_run[irow,icol]}',"Piecewise")
                run_fun += f'{2*tab}struct[0].Fx[{irow},{icol}] = {string}\n'            
                string = arg2np(f'{Fx_run[irow,icol]}',"Piecewise")  
                ini_fun += f'{2*tab}struct[0].Fx_ini[{irow},{icol}] = {string}\n'

    ## Fy
    run_fun += f'\n'
    run_fun += f'{tab}if mode == 11:\n\n'
    ini_fun += f'\n'
    ini_fun += f'{tab}if mode == 11:\n\n'
    for irow in range(N_x):
        for icol in range(N_y):
            if not Fy_ini[irow,icol]==0:  # Fy_ini
                string = arg2np(f'{Fy_ini[irow,icol]}',"Piecewise")
                ini_fun += f'{2*tab}struct[0].Fy_ini[{irow},{icol}] = {string} \n'
            if not Fy_run[irow,icol]==0:  # Fy_run 
                string = arg2np(f'{Fy_run[irow,icol]}',"Piecewise")
                run_fun += f'{2*tab}struct[0].Fy[{irow},{icol}] = {string}\n'

    ## Gx
    run_fun += f'\n'
    ini_fun += f'\n'
    for irow in range(N_y):
        for icol in range(N_x):
            if not Gx_run[irow,icol]==0: # Gx_run = Gx_ini
                string = arg2np(f'{Gx_run[irow,icol]}',"Piecewise")
                run_fun += f'{2*tab}struct[0].Gx[{irow},{icol}] = {string}\n'
                string = arg2np(f'{Gx_ini[irow,icol]}',"Piecewise")
                ini_fun += f'{2*tab}struct[0].Gx_ini[{irow},{icol}] = {string}\n'

    ## Gy
    run_fun += f'\n'
    ini_fun += f'\n'
    for irow in range(N_y):
        for icol in range(N_y):  
            if not Gy_run[irow,icol].is_number:  # Gy_run
                string = f'{Gy_run[irow,icol]}'
                run_fun += f'{2*tab}struct[0].Gy[{irow},{icol}] = {arg2np(string,"Piecewise")}\n'
            if not Gy_ini[irow,icol].is_number:  # Gy_ini
                string = f'{Gy_ini[irow,icol]}'
                ini_fun += f'{2*tab}struct[0].Gy_ini[{irow},{icol}] = {arg2np(string,"Piecewise")}\n'


    # Fu and Gu
    nonzero = 0
    run_fun += f'\n'
    run_fun += f'{tab}if mode > 12:\n\n'
    for irow in range(N_x):
        for icol in range(N_u):
            if not Fu_run[irow,icol]==0:
                string = arg2np(f'{Fu_run[irow,icol]}',"Piecewise")
                run_fun += f'{2*tab}struct[0].Fu[{irow},{icol}] = {string}\n'
                nonzero += 1
             
                
    run_fun += f'\n'
    for irow in range(N_y):
        for icol in range(N_u):
            if not Gu_run[irow,icol]==0:
                string = arg2np(f'{Gu_run[irow,icol]}',"Piecewise")
                run_fun += f'{2*tab}struct[0].Gu[{irow},{icol}] = {string}\n'
                nonzero += 1

    # Hx, Hy and Hu
    run_fun += f'\n'
    for irow in range(N_z):
        for icol in range(N_x):
            if not Hx_run[irow,icol]==0:
                string = arg2np(f'{Hx_run[irow,icol]}',"Piecewise")
                run_fun += f'{2*tab}struct[0].Hx[{irow},{icol}] = {string}\n'
                nonzero += 1
                
    run_fun += f'\n'
    for irow in range(N_z):
        for icol in range(N_y):
            if not Hy_run[irow,icol]==0:
                string = arg2np(f'{Hy_run[irow,icol]}',"Piecewise")
                run_fun += f'{2*tab}struct[0].Hy[{irow},{icol}] = {string}\n'
                nonzero += 1
                                
    run_fun += f'\n'
    for irow in range(N_z):
        for icol in range(N_u):
            if not Hu_run[irow,icol]==0:
                string = arg2np(f'{Hu_run[irow,icol]}',"Piecewise")
                run_fun += f'{2*tab}struct[0].Hu[{irow},{icol}] = {string}\n'
                nonzero += 1              
 


    if nonzero==0: 
        print('jacobians respect u = 0')
        
        run_fun += f'{2*tab}pass\n'                 
                
                
    #with open('./class_dae_template.py','r') as fobj:
    #    class_template = fobj.read()
    class_template = pkgutil.get_data(__name__, "templates/class_dae_template.py").decode().replace('\r\n','\n') 
    functions_template = pkgutil.get_data(__name__, "templates/functions_template.py").decode().replace('\r\n','\n') 
    solver_template = pkgutil.get_data(__name__, "templates/solver_template_v2.py").decode().replace('\r\n','\n') 

    class_template = class_template.replace('{name}',str(name))
    class_template = class_template.replace('{N_x}',str(N_x)).replace('{N_y}',str(N_y)).replace('{N_z}',str(N_z))
    class_template = class_template.replace('{x_list}', str([str(item) for item in x]))
    class_template = class_template.replace('{y_run_list}', str([str(item) for item in y_run]))
    class_template = class_template.replace('{y_ini_list}', str([str(item) for item in y_ini]))
    class_template = class_template.replace('{params_list}', str([str(item) for item in params]))
    class_template = class_template.replace('{params_values_list}', str([(params[item]) for item in params]))
    class_template = class_template.replace('{inputs_ini_list}', str([str(item) for item in u_ini]))
    class_template = class_template.replace('{inputs_run_list}', str([str(item) for item in u_run]))
    class_template = class_template.replace('{inputs_ini_values_list}', str([(sys['u_ini_dict'][item]) for item in sys['u_ini_dict']]))
    class_template = class_template.replace('{inputs_run_values_list}', str([(sys['u_run_dict'][item]) for item in sys['u_run_dict']]))
    class_template = class_template.replace('{outputs_list}', str([str(item) for item in h_dict]))

    module = class_template
    module += '\n'*3
    module += run_fun
    module += '\n'*3
    module += ini_fun
    module += '\n'*3
    module += functions_template
    module += '\n'*3
    module += solver_template

    with open(f'{name}.py','w') as fobj:
        fobj.write(module)

    
    
if __name__ == "__main__":

    
    params_dict = {'X_d':1.81,'X1d':0.3,'T1d0':8.0,
                   'X_q':1.76,'X1q':0.65,'T1q0':1.0,
                   'R_a':0.003,'X_l': 0.05, 
                   'H':3.5,'D':1.0,
                   'Omega_b':2*np.pi*50,'omega_s':1.0,
                   'v_0':0.9008,'theta_0':0.0}
    
    
    u_ini_dict = {'P_t':0.8, 'Q_t':0.2}  # for the initialization problem
    u_run_dict = {'p_m':0.8,'e1q':1.0}  # for the running problem (here initialization and running problem are the same)
    
    
    x_list = ['delta','omega']    # [inductor current, PI integrator]
    y_ini_list = ['i_d','i_q','v_1','theta_1','p_m','e1q'] # for the initialization problem
    y_run_list = ['i_d','i_q','v_1','theta_1','P_t','Q_t'] # for the running problem (here initialization and running problem are the same)
    
    sys_vars = {'params':params_dict,
                'u_list':u_run_dict,
                'x_list':x_list,
                'y_list':y_run_list}
    
    exec(sym_gen_str())  # exec to generate the required symbolic varables and constants
    

