#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 11:53:28 2018

@author: jmmauricio

bug: if there is not jacobian add pass to the if mode == 12:
"""
import numpy as np
import sympy as sym
from collections import deque 
import pkgutil
import re
from sympy.matrices.sparsetools import _doktocsr
from sympy import SparseMatrix
import time
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
        
    y_ini_set = set(sys['y_ini_list'])
    contains_duplicates = len(sys['y_ini_list']) != len(y_ini_set)
    if contains_duplicates:
        print('error: y_ini contains duplicates')

    y_set = set(sys['y_run_list'])
    contains_duplicates = len(sys['y_run_list']) != len(y_set)
    if contains_duplicates:
        print('error: y_run contains duplicates')
       
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
    
    
    Fx_ini_rows,Fx_ini_cols = np.nonzero(Fx_ini)
    Fy_ini_rows,Fy_ini_cols = np.nonzero(Fy_ini)
    Gx_ini_rows,Gx_ini_cols = np.nonzero(Gx_ini)
    Gy_ini_rows,Gy_ini_cols = np.nonzero(Gy_ini)
    
    sys['Fx_ini_rows'] = Fx_ini_rows 
    sys['Fx_ini_cols'] = Fx_ini_cols 
    sys['Fy_ini_rows'] = Fy_ini_rows 
    sys['Fy_ini_cols'] = Fy_ini_cols 
    sys['Gx_ini_rows'] = Gx_ini_rows 
    sys['Gx_ini_cols'] = Gx_ini_cols 
    sys['Gy_ini_rows'] = Gy_ini_rows 
    sys['Gy_ini_cols'] = Gy_ini_cols 

    Fx_run_rows,Fx_run_cols = np.nonzero(Fx_run)
    Fy_run_rows,Fy_run_cols = np.nonzero(Fy_run)
    Gx_run_rows,Gx_run_cols = np.nonzero(Gx_run)
    Gy_run_rows,Gy_run_cols = np.nonzero(Gy_run)
    
    sys['Fx_run_rows'] = Fx_run_rows 
    sys['Fx_run_cols'] = Fx_run_cols 
    sys['Fy_run_rows'] = Fy_run_rows 
    sys['Fy_run_cols'] = Fy_run_cols 
    sys['Gx_run_rows'] = Gx_run_rows 
    sys['Gx_run_cols'] = Gx_run_cols 
    sys['Gy_run_rows'] = Gy_run_rows 
    sys['Gy_run_cols'] = Gy_run_cols 

    
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

def replace(eq, x_list, y_list, u_list, params_list):
    eq_str = str(eq)
    it = 0
    for item in x_list:
        x_sym = sym.Symbol(f'x_{it:04}',real=True)
        eq = eq.replace(item,x_sym)
        eq_str = str(eq)
        eq_str = eq_str.replace(str(x_sym), f'x[{it}]')
        it +=1
    it = 0
    for item in y_list:
        y_sym = sym.Symbol(f'y_{it:04}',real=True)
        eq = eq.replace(item,y_sym)
        eq_str = str(eq)
        eq_str = eq_str.replace(str(y_sym), f'y[{it}]')
        it +=1
    it = 0
    for item in u_list:
        u_sym = sym.Symbol(f'u_{it:04}')
        eq = eq.subs(str(item),u_sym)
        eq_str = str(eq)
        eq_str = eq_str.replace(str(u_sym), f'u[{it}]')
        it +=1
    it = 0
    for item in params_list:
        item_sym = sym.Symbol(item)
        p_sym = sym.Symbol(f'p_{it:04}',real=True)
        eq = eq.replace(item_sym,p_sym)
        eq_str = str(eq)
        eq_str = eq_str.replace(str(p_sym), f'p[{it}]')
        it +=1    

    eq_str = str(eq)
    
    it = 0
    for item in x_list:
        x_sym = sym.Symbol(f'x_{it:04}',real=True)        
        eq_str = eq_str.replace(str(x_sym), f'x[{it}]')
        it +=1
    it = 0
    for item in y_list:
        y_sym = sym.Symbol(f'y_{it:04}',real=True)
        eq_str = eq_str.replace(str(y_sym), f'y[{it}]')
        it +=1
    it = 0
    for item in u_list:
        u_sym = sym.Symbol(f'u_{it:04}')
        eq_str = eq_str.replace(str(u_sym), f'u[{it}]')
        it +=1
    it = 0
    for item in params_list:
        item_sym = sym.Symbol(item)
        p_sym = sym.Symbol(f'p_{it:04}',real=True)
        eq_str = eq_str.replace(str(p_sym), f'p[{it}]')
        it +=1   
        
    return eq_str

def mreplace(string,replacements):
    for item in replacements:
        string = str(string).replace(item,replacements[item])
    return string

def sym2str(sym_exp,x,y,u,p,multi_eval=False):
    
    matrix_str = str(sym_exp)
    
    mev = ''
    if multi_eval:
        mev = 'i,'
    
    if sym_exp == 0:
        matrix_str = 0
        return matrix_str
        

    for it in range(len(x)):
        name = str(x[it])
        matrix_str = re.sub(r'\b' + name + r'\b',f'x[{mev}{it}]',matrix_str)
    for it in range(len(y)):
        name = str(y[it])
        matrix_str = re.sub(r'\b' + name + r'\b',f'y[{mev}{it}]',matrix_str)
    for it in range(len(u)):
        name = str(u[it])
        matrix_str = re.sub(r'\b' + name + r'\b',f'u[{mev}{it}]',matrix_str)
    it = 0
    for item in p:
        name = str(item)
        matrix_str = re.sub(r'\b' + name + r'\b',f'p[{mev}{it}]',matrix_str)
        it+=1
        
    return matrix_str   

def vector2string(vector,function_header,vector_name,x,y,u,p,aux={},multi_eval=False):
    string = ''
    string_i = ''
    N  = len(vector)
    
    for item in aux:
        str_element = sym2str(aux[item],x,y,u,p,multi_eval=multi_eval)
        string_i += f'{" "*4}{item} = {str_element}\n'
     
    string_i += '\n'
    
    for it in range(N):
        str_element = sym2str(vector[it],x,y,u,p,multi_eval=multi_eval)
        str_element = arg2np(str_element,"Piecewise")

        if multi_eval:
            string_i += f'{" "*4}{vector_name}[i,{it}] = {str_element}\n'
        else:
            string_i += f'{" "*4}{vector_name}[{it}] = {str_element}\n'            

    string += '\n'
    string += function_header
    string += string_i
    string += '\n'
    
    return string

def matrix2string(matrix,function_header,matrix_name,x,y,u,p):
    string = ''
    string_xy = ''
    string_up = ''
    string_num = ''
    N_row,N_col = matrix.shape
    
    for irow in range(N_row):
        for icol in range(N_col):
            element = matrix[irow,icol]
            
            if element.is_number:
                str_element = str(element)
            else:
                str_element = sym2str(element,x,y,u,p)
            
            if str_element != '0':
                str_element = arg2np(str_element,"Piecewise")
                if 'x' in str_element or 'y' in str_element:
                    string_xy += f'{" "*4}{matrix_name}[{irow},{icol}] = {str_element}\n'
                elif 'u' in str_element or 'p' in str_element or 'Dt' in str_element:
                    string_up += f'{" "*4}{matrix_name}[{irow},{icol}] = {str_element}\n'
                else:
                    string_num += f'{" "*4}{matrix_name}[{irow},{icol}] = {str_element}\n'

                    

    string += '\n'
    string += '@numba.njit(cache=True)\n'
    string += function_header.replace('(','_xy(')
    string += string_xy
    string += '\n'
    string += '@numba.njit(cache=True)\n'
    string += function_header.replace('(','_up(')
    string += string_up
    string += '\n'
    string += function_header.replace('(','_num(')
    string += string_num
    string += '\n'
    
    return string

def spmatrix2string(spmatrix_list,function_header,matrix_name,x,y,u,p):
    
    data = spmatrix_list[0]
    string = ''
    string_xy = ''
    string_up = ''
    string_num = ''
    
    for irow in range(len(data)):

        element = data[irow]
        
        if element.is_number:
            str_element = str(element)
        else:
            str_element = sym2str(element,x,y,u,p)        
        
        if str_element != 0:
            str_element = arg2np(str_element,"Piecewise")
            if 'x' in str_element or 'y' in str_element:
                string_xy += f'{" "*4}{matrix_name}[{irow}] = {str_element}\n'
            elif 'u' in str_element or 'p' in str_element or 'Dt' in str_element:
                string_up += f'{" "*4}{matrix_name}[{irow}] = {str_element}\n'
            else:
                string_num+= f'{" "*4}{matrix_name}[{irow}] = {str_element}\n'
                

    string += '\n'
    string += '@numba.njit(cache=True)\n'
    string += function_header.replace('(','_xy(')
    string += string_xy
    string += '\n'
    string += '@numba.njit(cache=True)\n'
    string += function_header.replace('(','_up(')
    string += string_up
    string += '\n'
    string += function_header.replace('(','_num(')
    string += string_num
    string += '\n'
    
    string += f'def {matrix_name}_vectors():\n\n'
    string += f'{" "*4}{matrix_name}_ia = ' + str(spmatrix_list[1]) + '\n'
    string += f'{" "*4}{matrix_name}_ja = ' + str(spmatrix_list[2]) + '\n'    
    string += f'{" "*4}{matrix_name}_nia = ' + str(spmatrix_list[3][0])  + '\n'
    string += f'{" "*4}{matrix_name}_nja = ' + str(spmatrix_list[3][1])  + '\n'
    string += f'{" "*4}return {matrix_name}_ia, {matrix_name}_ja, {matrix_name}_nia, {matrix_name}_nja \n'
    
    return string


def sys2num(sys, verbose=False):
    
    params = sys['params_dict']
    h_dict = sys['h_dict']
    
    if 'xy0_dict' in sys:
        xy0_dict = sys['xy0_dict']
        
    x = sys['x']
    y_ini = sys['y_ini']
    y_run = sys['y_run']

    u_ini = sys['u_ini']
    u_run = sys['u_run']

    substitutions = {}
    replacements = {}    
    substitutions_ini = {}
    replacements_ini = {}
    substitutions_run = {}
    replacements_run = {}
    it = 0
    for item in sys['params_dict']:
        p     = sym.Symbol(f'p_{it:05}',real=True)
        param = sym.Symbol(item,real=True)
        substitutions.update({param:p})
        replacements.update({f'p_{it:05}':f'p[{it}]'})
        it+=1
    for item in sys['params_dict']:
        p     = sym.Symbol(f'p_{it:05}',real=True)
        param = sym.Symbol(item)
        substitutions.update({param:p})
        replacements.update({f'p_{it:05}':f'p[{it}]'})
        it+=1


    it = 0
    for item in sys['x']:
        x_new = sym.Symbol(f'x_{it:05}',real=True)
        param = sym.Symbol(str(item),real=True)
        substitutions.update({param:x_new})
        replacements.update({f'x_{it:05}':f'x[{it}]'})
        it+=1   
        
    
    it = 0
    for item in sys['y_ini']:
        
        new = sym.Symbol(f'y_ini_{it:05}',real=True)
        param = sym.Symbol(str(item),real=True)
        substitutions_ini.update({param:new})
        replacements_ini.update({f'y_ini_{it:05}':f'y[{it}]'})
        it+=1  
    it = 0
    for item in sys['u_ini']:
        new = sym.Symbol(f'u_ini_{it:05}',real=True)
        param = sym.Symbol(str(item),real=True)
        substitutions_ini.update({param:new})
        replacements_ini.update({f'u_ini_{it:05}':f'u[{it}]'})
        it+=1   
    it = 0
    for item in sys['y_run']:
        new = sym.Symbol(f'y_run_{it:05}',real=True)
        param = sym.Symbol(str(item),real=True)
        substitutions_run.update({param:new})
        replacements_run.update({f'y_run_{it:05}':f'y[{it}]'})
        it+=1  
    it = 0
    for item in sys['u_run']:
        new = sym.Symbol(f'u_run_{it:05}',real=True)
        param = sym.Symbol(str(item),real=True)
        substitutions_run.update({param:new})
        replacements_run.update({f'u_run_{it:05}':f'u[{it}]'})
        it+=1  
        
    
        
    
    f_ini = sys['f']
    g_ini = sys['g'] 
    
    f_run = sys['f']
    g_run = sys['g'] 
    
    h = sys['h'] 

    Fx_run = sys['Fx_run']
    Fy_run = sys['Fy_run']
    Gx_run = sys['Gx_run']
    Gy_run = sys['Gy_run']
    
    Fx_ini = sys['Fx_run']
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
    N_z = len(h_dict)
    
    t_0 = time.time()
    if verbose: print(f'computing jac_ini (time: {time.time()-t_0}')
    jac_ini = sym.Matrix([[Fx_ini,Fy_ini],[Gx_ini,Gy_ini]]) 
    if verbose: print(f'computing jac_run (time: {time.time()-t_0}')
    jac_run = sym.Matrix([[Fx_run,Fy_run],[Gx_run,Gy_run]]) 
    
    if verbose: print(f'computing jac_trap (time: {time.time()-t_0}')
    eye = sym.eye(N_x, real=True)
    Dt = sym.Symbol('Dt',real=True)
    jac_trap = sym.Matrix([[eye - 0.5*Dt*Fx_run, -0.5*Dt*Fy_run],[Gx_run,Gy_run]])    

    sys['jac_ini']  = jac_ini
    sys['jac_run']  = jac_run   
    sys['jac_trap'] = jac_trap
    

    name = sys['name']
    
    if 'aux_dict' in sys:
        aux_dict = sys['aux_dict']
    else:
        aux_dict = {}
        

    f_ini_eval = ''
    f_run_eval = ''
    
    
    numba_enable = True
    tab = ' '*4

    if verbose: print(f'generating f_ini_eval string (time: {time.time()-t_0}')    
    # f: differential equations for backward problem
    function_header  = '@numba.njit(cache=True)\n'
    function_header += 'def f_ini_eval(f_ini,x,y,u,p,xyup = 0):\n\n'
    x,y,u,p = sys['x'],sys['y_ini'],sys['u_ini'],sys['params_dict'].keys()
    f_ini_eval += vector2string(sys['f'],function_header,'f_ini',x,y,u,p,aux=aux_dict,multi_eval=False)

    if verbose: print(f'generating f_run_eval string (time: {time.time()-t_0}')        
    # f: differential equations for foreward problem
    function_header  = '@numba.njit(cache=True)\n'
    function_header += 'def f_run_eval(f_run,x,y,u,p,xyup = 0):\n\n'
    x,y,u,p = sys['x'],sys['y_run'],sys['u_run'],sys['params_dict'].keys()
    f_run_eval += vector2string(sys['f'],function_header,'f_run',x,y,u,p,aux=aux_dict,multi_eval=False)


    if verbose: print(f'generating g_ini_eval string (time: {time.time()-t_0}')        
    # g: algebraic equations for backward problem
    function_header  = '@numba.njit(cache=True)\n'
    function_header += 'def g_ini_eval(g_ini,x,y,u,p,xyup = 0):\n\n'
    x,y,u,p = sys['x'],sys['y_ini'],sys['u_ini'],sys['params_dict'].keys()
    g_ini_eval = vector2string(sys['g'],function_header,'g_ini',x,y,u,p,aux=aux_dict,multi_eval=False)

    if verbose: print(f'generating g_run_eval string (time: {time.time()-t_0}')           
    # g: algebraic equations for forward problem
    function_header  = '@numba.njit(cache=True)\n'
    function_header += 'def g_run_eval(g_run,x,y,u,p,xyup = 1):\n\n'
    x,y,u,p = sys['x'],sys['y_run'],sys['u_run'],sys['params_dict'].keys()
    g_run_eval = vector2string(sys['g'],function_header,'g_run',x,y,u,p,aux=aux_dict,multi_eval=False)

    if verbose: print(f'generating h_run_eval string (time: {time.time()-t_0}')               
    # h: outputs for the foreward problem
    function_header  = '@numba.njit(cache=True)\n'
    function_header += 'def h_eval(h_run,x,y,u,p,xyup = 1):\n\n'
    x,y,u,p = sys['x'],sys['y_run'],sys['u_run'],sys['params_dict'].keys()
    h_run_eval = vector2string(sys['h'],function_header,'h_run',x,y,u,p,aux=aux_dict,multi_eval=False)

    if verbose: print(f'generating f_run_gpu string (time: {time.time()-t_0}')      
    # GPU f: differential equations for foreward problem
    function_header  = '@cuda.jit(device=True)\n'
    function_header += 'def f_run_gpu(f_run,x,u,p):\n\n'
    function_header += '    sin = math.sin\n'
    function_header += '    cos = math.cos\n'
    function_header += '    sqrt = math.sqrt\n'
    function_header += '    abs = math.fabs\n'
    x,y,u,p = sys['x'],sys['y_run'],sys['u_run'],sys['params_dict'].keys()
    f_run_gpu = vector2string(sys['f'],function_header,'f_run',x,y,u,p,aux=aux_dict,multi_eval=False)

    if verbose: print(f'generating h_run_gpu string (time: {time.time()-t_0}')      
    # GPU h: outputs
    function_header  = '@cuda.jit(device=True)\n'
    function_header += 'def h_eval_gpu(z,x,u,p):\n\n'
    function_header += '    sin = math.sin\n'
    function_header += '    cos = math.cos\n'
    function_header += '    sqrt = math.sqrt\n'
    function_header += '    abs = math.fabs\n'
    x,y,u,p = sys['x'],sys['y_run'],sys['u_run'],sys['params_dict'].keys()
    h_run_gpu = vector2string(sys['h'],function_header,'z',x,y,u,p,aux=aux_dict,multi_eval=False)



  
    xy0_eval = ''
    if 'xy0_dict' in sys:    
        if numba_enable: xy0_eval += '@numba.njit(cache=True)\n'
        xy0_eval += 'def xy0_eval(x,y,u,p):\n\n'
        for item in xy0_dict:
            if type(xy0_dict[item]) == float:
                xy0_sym_ = sym.Symbol(str(xy0_dict[item]))
            else:
                xy0_sym_ = xy0_dict[item]
            x,y,u,p = sys['x'],sys['y_ini'],sys['u_ini'],sys['params_dict'].keys()
            rh = sym2str(xy0_sym_,x,y,u,p,multi_eval=False)
            item_sym = sym.Symbol(item)
            lh = sym2str(item_sym,x,y,u,p,multi_eval=False)
            xy0_eval += f'{tab}{lh} = {arg2np(rh,"Piecewise")}\n'
        xy0_eval += f'{tab}\n'    
        
    
    if verbose: print(f'generating jac_ini_ss_eval string (time: {time.time()-t_0}')      
### jacobian steady state backward   
    function_header = 'def jac_ini_ss_eval(jac_ini,x,y,u,p,xyup = 1):\n\n'
    matrix_name = 'jac_ini'
    x,y,u,p = sys['x'],sys['y_ini'],sys['u_ini'],sys['params_dict'].keys()
    jac_ini_ss_eval = matrix2string(jac_ini,function_header,matrix_name,x,y,u,p)

    if verbose: print(f'generating sp_jac_ini_eval string (time: {time.time()-t_0}')      
    
## sparse jacobian ini 
    spmatrix_list = _doktocsr(SparseMatrix(jac_ini))

    function_header =  '@numba.njit(cache=True)\n'
    function_header = 'def sp_jac_ini_eval(sp_jac_ini,x,y,u,p,Dt,xyup = 1):\n\n'
    matrix_name = 'sp_jac_ini'
    x,y,u,p = sys['x'],sys['y_run'],sys['u_run'],sys['params_dict'].keys()
    sp_jac_ini_eval = spmatrix2string(spmatrix_list,function_header,matrix_name,x,y,u,p)
    
    
    
    if verbose: print(f'generating jac_ss_eval string (time: {time.time()-t_0}')      
## jacobian steady state forward
    function_header =  '@numba.njit(cache=True)\n'
    function_header = 'def jac_run_ss_eval(jac_run,x,y,u,p,xyup = 1):\n\n'
    matrix_name = 'jac_run'
    x,y,u,p = sys['x'],sys['y_run'],sys['u_run'],sys['params_dict'].keys()
    jac_ss_eval = matrix2string(jac_run,function_header,matrix_name,x,y,u,p)


    if verbose: print(f'generating jac_trap_eval string (time: {time.time()-t_0}')      
## jacobian trapezoidal 
    function_header =  '@numba.njit(cache=True)\n'
    function_header = 'def jac_trap_eval(jac_trap,x,y,u,p,Dt,xyup = 1):\n\n'
    matrix_name = 'jac_trap'
    x,y,u,p = sys['x'],sys['y_run'],sys['u_run'],sys['params_dict'].keys()
    jac_trap_eval = matrix2string(jac_trap,function_header,matrix_name,x,y,u,p)

## sparse jacobian trapezoidal 
    spmatrix_list = _doktocsr(SparseMatrix(jac_trap))
    function_header =  '@numba.njit(cache=True)\n'
    function_header = 'def sp_jac_trap_eval(sp_jac_trap,x,y,u,p,Dt,xyup = 1):\n\n'
    matrix_name = 'sp_jac_trap'
    x,y,u,p = sys['x'],sys['y_run'],sys['u_run'],sys['params_dict'].keys()
    sp_jac_trap_eval = spmatrix2string(spmatrix_list,function_header,matrix_name,x,y,u,p)
    


    class_template = pkgutil.get_data(__name__, "templates/class_dae_fast_template.py").decode().replace('\r\n','\n') 
    functions_template = pkgutil.get_data(__name__, "templates/functions_template.py").decode().replace('\r\n','\n') 
#     solver_template = pkgutil.get_data(__name__, "templates/solver_template_v2.py").decode().replace('\r\n','\n') 

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

    if verbose: print(f'writting file (time: {time.time()-t_0}')      
    module = class_template
    module += '\n'*3
    module += f_ini_eval
    module += '\n'*3
    module += f_run_eval
    module += '\n'*3
    module += g_ini_eval
    module += '\n'*3
    module += g_run_eval
    module += '\n'*3    
    module += f_run_gpu
    module += '\n'*3
    module += h_run_gpu
    module += '\n'*3  

    module += jac_ini_ss_eval
    module += '\n'*3
    module += sp_jac_ini_eval
    module += '\n'*3
    module += jac_ss_eval
    module += '\n'*3
    module += jac_trap_eval
    module += '\n'*3
    module += h_run_eval
    module += '\n'*3
    module += functions_template
    module += '\n'*3
    module += xy0_eval
    module += '\n'*3
    module += sp_jac_trap_eval
    module += '\n'*3    
    module += 'def nonzeros():\n'
    module += f"    Fx_ini_rows = {sys['Fx_ini_rows'].tolist()}\n\n"
    module += f"    Fx_ini_cols = {sys['Fx_ini_cols'].tolist()}\n\n"   
    module += f"    Fy_ini_rows = {sys['Fy_ini_rows'].tolist()}\n\n"
    module += f"    Fy_ini_cols = {sys['Fy_ini_cols'].tolist()}\n\n"  
    module += f"    Gx_ini_rows = {sys['Gx_ini_rows'].tolist()}\n\n"
    module += f"    Gx_ini_cols = {sys['Gx_ini_cols'].tolist()}\n\n"  
    module += f"    Gy_ini_rows = {sys['Gy_ini_rows'].tolist()}\n\n"
    module += f"    Gy_ini_cols = {sys['Gy_ini_cols'].tolist()}\n\n"    
    module += f"    return Fx_ini_rows,Fx_ini_cols,Fy_ini_rows,Fy_ini_cols,Gx_ini_rows,Gx_ini_cols,Gy_ini_rows,Gy_ini_cols"       

    
    with open(f'{name}.py','w') as fobj:
        fobj.write(module)
   





    
    
if __name__ == "__main__":

    
    params_dict = {'X_d':1.81,'X1d':0.3, 'T1d0':8.0,  # synnchronous machine d-axis parameters
                   'X_q':1.76,'X1q':0.65,'T1q0':1.0,  # synnchronous machine q-axis parameters
                   'R_a':0.003,
                   'X_l': 0.02, 
                   'H':3.5,'D':0.0,
                   'Omega_b':2*np.pi*50,'omega_s':1.0,
                   'V_0':1.0,'Theta_0':0.0}
    
    
    u_ini_dict = {'p_m':0.8,'v_f':1.0}  # for the initialization problem
    u_run_dict = {'p_m':0.8,'v_f':1.0}  # for the running problem (here initialization and running problem are the same)
    
    
    x_list = ['delta','omega','e1q','e1d']    # dynamic states
    y_ini_list = ['i_d','i_q','p_t','q_t','v_t','theta_t']   
    y_run_list = ['i_d','i_q','p_t','q_t','v_t','theta_t']   
    
    sys_vars = {'params':params_dict,
                'u_list':u_run_dict,
                'x_list':x_list,
                'y_list':y_run_list}
    
    exec(sym_gen_str())  # exec to generate the required symbolic varables and constants

    # auxiliar equations
    v_d = v_t*sin(delta - theta_t)  # park
    v_q = v_t*cos(delta - theta_t)  # park
    
    p_e = i_d*(v_d + R_a*i_d) + i_q*(v_q + R_a*i_q) # electromagnetic power
    
    # dynamic equations
    ddelta = Omega_b*(omega - omega_s) - 1e-4*delta  # load angle
    domega = 1/(2*H)*(p_m - p_e - D*(omega - omega_s)) # speed
    de1q = 1/T1d0*(-e1q - (X_d - X1d)*i_d + v_f)
    de1d = 1/T1q0*(-e1d + (X_q - X1q)*i_q)
    
    # algrbraic equations
    g_1 = v_q + R_a*i_q + X1d*i_d - e1q # stator
    g_2 = v_d + R_a*i_d - X1q*i_q - e1d # stator
    g_3 = i_d*v_d + i_q*v_q - p_t # active power 
    g_4 = i_d*v_q - i_q*v_d - q_t # reactive power
    g_5 = p_t - (v_t*V_0*sin(theta_t - Theta_0))/X_l  # network equation (p)
    g_6 = q_t + (v_t*V_0*cos(theta_t - Theta_0))/X_l - v_t**2/X_l  # network equation (q) 

    sys_dict = {'name':'build_test',
           'params_dict':params,
           'f_list':[ddelta,domega,de1q,de1d],
           'g_list':[g_1,g_2,g_3,g_4,g_5,g_6],
           'x_list':x_list,
           'y_ini_list':y_ini_list,
           'y_run_list':y_run_list,
           'u_run_dict':u_run_dict,
           'u_ini_dict':u_ini_dict,
           'h_dict':{'p_m':p_m,'p_e':p_e, 'v_f':v_f}}
    
    sys_dict = system(sys_dict)
   # sys2num(sys_dict)                                                             
