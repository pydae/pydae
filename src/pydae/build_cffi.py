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
from sympy.codegen.ast import Assignment
from scipy.sparse import csr_matrix,save_npz
import os


import cffi

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

class builder():   

    def __init__(self,sys,verbose=False):

        self.verbose = verbose
        self.sys = sys
        self.jac_num = True
        self.name = self.sys['name']
        self.save_sources = False
        self.string_u2z = ''
        self.u2z = '\#'

        if not os.path.exists('build'):
            os.makedirs('build')

        self.matrices_folder = 'build'


    def check_system(self):
        sys = self.sys
        
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
        
            
    def sym_jac(self,f,x):
        
        N_f = len(f)
        N_x = len(x)
        J = sym.zeros(N_f,N_x)
        
        for irow in range(N_f):
            str_f = str(f[irow])
            for icol in range(N_x):
                if str(x[icol]) in str_f:
                    J[irow,icol] = f[irow].diff(x[icol])
        
        return J

    def system(self):
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

        sys = self.sys
        
        t_0 = time.time()
        if self.verbose: print(f'check_system (time: {time.time()-t_0})')
        self.check_system()
        
        f = sym.Matrix(sys['f_list']).T
        g = sym.Matrix(sys['g_list']).T
        x = sym.Matrix(sys['x_list']).T
        y_ini = sym.Matrix(sys['y_ini_list']).T
        y_run = sym.Matrix(sys['y_run_list']).T
        u_ini = sym.Matrix(list(sys['u_ini_dict'].keys()), real=True)
        
        u_run_list = [sym.Symbol(item,real=True) for item in list(sys['u_run_dict'].keys())]
        u_run = sym.Matrix(u_run_list).T 
        h =  sym.Matrix(list(sys['h_dict'].values())).T   
        

        if self.verbose: print(f'computing jacobians Fx_run,Fy_run  (time: {time.time()-t_0:0.3f} s)')
        Fx_run = f.jacobian(x)
        Fy_run = f.jacobian(y_run)
        if self.verbose: print(f'computing jacobians Gx_run,Gy_run  (time: {time.time()-t_0:0.3f} s)')
        Gx_run = g.jacobian(x)
        
        #Gy_run = g.jacobian(y_run): 
        Gy_run = self.sym_jac(g,y_run)

                

        if self.verbose: print(f'computing jacobians Fu_run,Gu_run  (time: {time.time()-t_0:0.3f} s)')
        Fu_run = f.jacobian(u_run)
    

        
        #Gu_run = g.jacobian(u_run):
        Gu_run = self.sym_jac(g,u_run)

        if self.verbose: print(f'computing jacobians Fx_ini,Fy_ini  (time: {time.time()-t_0:0.3f} s)')   
        Fx_ini = f.jacobian(x)
        Fy_ini = f.jacobian(y_ini)
        if self.verbose: print(f'computing jacobians Gx_ini,Gy_ini  (time: {time.time()-t_0:0.3f} s)')
        Gx_ini = g.jacobian(x)
        
        #Gy_ini = g.jacobian(y_ini)
        Gy_ini =  self.sym_jac(g,y_ini) 
                    
        if self.verbose: print(f'computing jacobians Hx_run,Hy_run,Hu_run  (time: {time.time()-t_0} s)')   
        Hx_run = h.jacobian(x)
        
        #Hy_run = h.jacobian(y_run) 
        Hy_run = self.sym_jac(h,y_run)
        
        
        #Hu_run = h.jacobian(u_run)
        Hu_run = self.sym_jac(h,u_run)


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
        
        sys['N_x'] = len(x)
        sys['N_y'] = len(y_run)
        sys['N_u'] = len(u_run)
        sys['N_z'] = len(h)

        if self.verbose: print(f'end system  (time: {time.time()-t_0:0.3f} s)')   
        
        return sys


    def jacobians(self):
        '''
        

        Parameters
        ----------
        sys : dict
            

        Returns
        -------
        sys : TYPE
            DESCRIPTION.

        '''
        
        sys = self.sys
        
        t_0 = time.time()
        
        N_x = sys['N_x']
        N_y = sys['N_y']
        N_z = sys['N_z']
        N_u = sys['N_u']
    
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
        
        if self.verbose: print(f'computing jac_ini (time: {time.time()-t_0})')
        jac_ini = sym.Matrix([[Fx_ini,Fy_ini],[Gx_ini,Gy_ini]]) 
        
        if self.verbose: print(f'computing jac_run (time: {time.time()-t_0})')
        jac_run = sym.Matrix([[Fx_run,Fy_run],[Gx_run,Gy_run]]) 
        
        if self.verbose: print(f'computing jac_trap (time: {time.time()-t_0})')
        eye = sym.eye(N_x, real=True)
        Dt = sym.Symbol('Dt',real=True)
        jac_trap = sym.Matrix([[eye - 0.5*Dt*Fx_run, -0.5*Dt*Fy_run],[Gx_run,Gy_run]])    

        sys['jac_ini']  = jac_ini
        sys['jac_run']  = jac_run   
        sys['jac_trap'] = jac_trap
        

        if self.verbose: print(f'end of jacobians computation (time: {time.time()-t_0:0.3f})')



    def getIndex(self,s, i): 
    
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
    
    def arg2np(self,string,function_name):
        idx_end = 0
        for it in range(3):
            if function_name in string[idx_end:]:
                #print(string[idx_end:])
                idx_ini = string.find(f'{function_name}(',idx_end)+len(f'{function_name}(')
                idx_end = self.getIndex(string, idx_ini-1)
                string = string.replace(string[idx_ini:idx_end],f'np.array([{string[idx_ini:idx_end]}])')
        else: pass
        return string    

    def replace(self,eq, x_list, y_list, u_list, params_list):
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

    # def sym2str(sym_exp,x,y,u,p,multi_eval=False):
        
    #     matrix_str = str(sym_exp)
        
    #     mev = ''
    #     if multi_eval:
    #         mev = 'i,'
        
    #     if sym_exp == 0:
    #         matrix_str = 0
    #         return matrix_str
            

    #     for it in range(len(x)):
    #         name = str(x[it])
    #         matrix_str = re.sub(r'\b' + name + r'\b',f'x[{mev}{it}]',matrix_str)
    #     for it in range(len(y)):
    #         name = str(y[it])
    #         matrix_str = re.sub(r'\b' + name + r'\b',f'y[{mev}{it}]',matrix_str)
    #     for it in range(len(u)):
    #         name = str(u[it])
    #         matrix_str = re.sub(r'\b' + name + r'\b',f'u[{mev}{it}]',matrix_str)
    #     it = 0
    #     for item in p:
    #         name = str(item)
    #         matrix_str = re.sub(r'\b' + name + r'\b',f'p[{mev}{it}]',matrix_str)
    #         it+=1
            
    #     return matrix_str   

    def vector2string(self,vector,function_header,vector_name,x,y,u,p,aux={},multi_eval=False):
        string = ''
        string_i = ''
        N  = len(vector)
        
        for item in aux:
            str_element = self.sym2str(aux[item],x,y,u,p,multi_eval=multi_eval)
            string_i += f'{" "*4}{item} = {str_element}\n'
        
        string_i += '\n'
        
        for it in range(N):
            str_element = self.sym2str(vector[it],x,y,u,p,multi_eval=multi_eval)
            str_element = self.arg2np(str_element,"Piecewise")

            if multi_eval:
                string_i += f'{" "*4}{vector_name}[i,{it}] = {str_element}\n'
            else:
                string_i += f'{" "*4}{vector_name}[{it}] = {str_element}\n'            

        string += '\n'
        string += function_header
        string += string_i
        string += '\n'
        
        return string

    def matrix2string(self,matrix,function_header,matrix_name,x,y,u,p):
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
                    str_element = self.sym2str(element,x,y,u,p)
                
                if str_element != '0':
                    str_element = self.arg2np(str_element,"Piecewise")
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

    def spmatrix2def(self,spmatrix_list,matrix_name):
        
        string = ''
        string += f'def {matrix_name}_vectors():\n\n'
        string += f'{" "*4}{matrix_name}_ia = ' + str(spmatrix_list[1]) + '\n'
        string += f'{" "*4}{matrix_name}_ja = ' + str(spmatrix_list[2]) + '\n'    
        string += f'{" "*4}{matrix_name}_nia = ' + str(spmatrix_list[3][0])  + '\n'
        string += f'{" "*4}{matrix_name}_nja = ' + str(spmatrix_list[3][1])  + '\n'
        string += f'{" "*4}return {matrix_name}_ia, {matrix_name}_ja, {matrix_name}_nia, {matrix_name}_nja \n'
        
        return string


    def sys2num(self):
        sys = self.sys

        t_0 = time.time()
        
        name = sys['name']
        N_x = sys['N_x']
        N_y = sys['N_y']
        N_z = sys['N_z']
        N_u = sys['N_u']



        class_template = pkgutil.get_data(__name__, "templates/class_dae_cffi_template.py").decode().replace('\r\n','\n') 
        functions_template = pkgutil.get_data(__name__, "templates/functions_template.py").decode().replace('\r\n','\n') 
    #     solver_template = pkgutil.get_data(__name__, "templates/solver_template_v2.py").decode().replace('\r\n','\n') 

        class_template = class_template.replace('{name}',str(name))
        class_template = class_template.replace('{u2z_jacobians}',self.string_u2z)
        class_template = class_template.replace('{u2z_comment}',self.u2z_comment)

        class_template = class_template.replace('{N_x}',str(N_x)).replace('{N_y}',str(N_y)).replace('{N_z}',str(N_z))
        class_template = class_template.replace('{x_list}', str([str(item) for item in sys['x']]))
        class_template = class_template.replace('{y_run_list}', str([str(item) for item in sys['y_run']]))
        class_template = class_template.replace('{y_ini_list}', str([str(item) for item in sys['y_ini']]))
        class_template = class_template.replace('{params_list}', str([str(item) for item in sys['params_dict']]))
        class_template = class_template.replace('{params_values_list}', str([(sys['params_dict'][item]) for item in sys['params_dict']]))
        class_template = class_template.replace('{inputs_ini_list}', str([str(item) for item in sys['u_ini']]))
        class_template = class_template.replace('{inputs_run_list}', str([str(item) for item in sys['u_run']]))
        class_template = class_template.replace('{inputs_ini_values_list}', str([(sys['u_ini_dict'][item]) for item in sys['u_ini_dict']]))
        class_template = class_template.replace('{inputs_run_values_list}', str([(sys['u_run_dict'][item]) for item in sys['u_run_dict']]))
        class_template = class_template.replace('{outputs_list}', str([str(item) for item in sys['h_dict']]))

        if 'enviroment' in sys:
            class_template = class_template.replace('{enviroment_name}',str(sys['enviroment']))
            class_template = class_template.replace('{dae_file_mode}',"'enviroment'")
        elif 'colab' in sys:
            class_template = class_template.replace('{enviroment_name}',str(sys['colab']))
            class_template = class_template.replace('{dae_file_mode}',"'colab'")            
        else:
            class_template = class_template.replace('{enviroment_name}','no_enviroment')
            class_template = class_template.replace('{dae_file_mode}',"'local'")
            
            
        module = class_template
        
        module += '\n'*2
        module += self.spmatrix2def(sys['sp_jac_ini_list'],'sp_jac_ini')
        module += '\n'
        module += self.spmatrix2def(sys['sp_jac_run_list'],'sp_jac_run')
        module += '\n'
        module += self.spmatrix2def(sys['sp_jac_trap_list'],'sp_jac_trap')
        
        

           
        with open(f'{name}.py','w') as fobj:
            fobj.write(module)
    
        if self.verbose: print(f'sys2num (time: {time.time()-t_0:0.3f})')                 
            




    def sym2xyup(self,string,sys,inirun):
        i = 0
        for item in sys['x']:
            string = re.sub(f"\\b{item}\\b"  ,f'x[{i}]',string)
            i+=1

        i = 0
        for item in sys[f'y_{inirun}']:
            string = re.sub(f"\\b{item}\\b"  ,f'y[{i}]',string)
            i+=1

        i = 0
        for item in sys[f'u_{inirun}']:
            string = re.sub(f"\\b{item}\\b"  ,f'u[{i}]',string)
            i+=1

        i = 0
        for item in sys['params_dict']:
            string = re.sub(f"\\b{item}\\b"  ,f'p[{i}]',string)
            i+=1
            
        return string

    def sym2str(self,fun_name,vector,sys,inirun):
        LHS = sym.Symbol('LHS')

        string_sym = ''
        for irow in range(len(vector)):
            string_sym += sym.ccode(Assignment(LHS,vector[irow])).replace('LHS',f'out[{irow}]') +'\n#'
            
        string_src = self.sym2xyup(string_sym,sys,inirun=inirun)
        string_split = string_src.split('#')
        string = ''
        for irow in range(len(vector)):
            string_sym = sym.ccode(Assignment(LHS,vector[irow])).replace('LHS',f'out[{irow}]') +'\n'
            string += string_split[irow]

        defs   = f'void {fun_name}_eval(double *out,double *x,double *y,double *u,double *p,double Dt);' + '\n' 
        source = f'void {fun_name}_eval(double *out,double *x,double *y,double *u,double *p,double Dt)'  + '{'+ '\n'*2
        source += string + '\n}\n'

        self.defs = defs
        self.source = source
        return defs,source
            
            
                    
    def sym2rhs(self,data,indices,indptr,shape,sys,inirun):
        '''
        Takes a sparse symbolic CRS matrix (data,indices,indptr) and convert it as a
        string replacing symbolic names by x,y,u,p names.
        
        
        '''

        
        rhs_list = []
        
        for irow in range(len(indptr)-1):
            for k in range(indptr[irow],indptr[irow+1]):
                icol = indices[k]
                data_k = data[k]
                if not data[irow] == 0:
                    string_sym = sym.ccode(data_k) + ';\n'
                    string_i = self.sym2xyup(string_sym,sys,inirun)

                    tipo = 'num'
                    if 'x[' in string_i or 'y[' in string_i:
                        tipo = 'xy' 
                    elif 'p[' in string_i or 'u[' in string_i or 'Dt' in string_i:
                        tipo = 'up' 
                
                    rhs_list += [(string_i,tipo,irow,icol)]
                    
        return rhs_list

    def sym2rhs2(self,data,indices,indptr,shape,sys,inirun):


        rhs_list = []
        
        string_full_sym = ''
        
        for irow in range(len(indptr)-1):
            for k in range(indptr[irow],indptr[irow+1]):
                icol = indices[k]
                data_k = data[k]
                if not data_k == 0:
                    string_sym = sym.ccode(sym.N(data_k)) + ';\n#'
                    string_full_sym += string_sym
        

        string_full = self.sym2xyup(string_full_sym,sys,inirun)
        
        string_split = string_full.split('#')  
        for irow in range(len(indptr)-1):
            for k in range(indptr[irow],indptr[irow+1]):
                icol = indices[k]
                data_k = string_split[k]
                if not data_k == 0:
                    string_i = data_k
                    
                    tipo = 'num'
                    if 'x[' in string_i or 'y[' in string_i:
                        tipo = 'xy' 
                    elif 'p[' in string_i or 'u[' in string_i or 'Dt' in string_i:
                        tipo = 'up' 
                
                    rhs_list += [(string_i,tipo,irow,icol)]

                    
        return rhs_list


    def rhs2str(self,rhs_list,lhs_name,num_matrix,shape,mode='crs'):
        string_xy = ''
        string_up = ''
        string_num = ''
        N_col = shape[1]

        k = 0    
        for data_i,tipo,irow,icol in rhs_list:
            if mode == 'crs':
                idx = k
            if mode == 'dense':
                idx = irow*N_col+icol
            if mode == '2d':
                idx = f'{irow},{icol}'
                
            if tipo == 'xy':
                string_xy += f'{lhs_name}[{idx}] = {data_i}'
            if tipo == 'up':
                string_up += f'{lhs_name}[{idx}] = {data_i}'
            if tipo == 'num':
                string_num += f'{lhs_name}[{idx}] = {data_i}'  
                num_matrix[k] = float(data_i[:-2])
            k+=1
                    
        return string_xy,string_up,string_num
            

    def str2src(self,fun_name,string_xy,string_up,string_num,matrix_name='out'):
    

        defs = ''
        defs += f'void {fun_name}_xy_eval(double *{matrix_name},double *x,double *y,double *u,double *p,double Dt);' + '\n' 
        defs += f'void {fun_name}_up_eval(double *{matrix_name},double *x,double *y,double *u,double *p,double Dt);' + '\n' 
        defs += f'void {fun_name}_num_eval(double *{matrix_name},double *x,double *y,double *u,double *p,double Dt);' + '\n' 
            
        source = '' 
        source += f'void {fun_name}_xy_eval(double *{matrix_name},double *x,double *y,double *u,double *p,double Dt)' + '{' +'\n'*2
        source += string_xy
        source += '\n}\n\n'
        source += f'void {fun_name}_up_eval(double *{matrix_name},double *x,double *y,double *u,double *p,double Dt)'  + '{'+ '\n'*2
        source += string_up
        source += '\n}\n\n'
        source += f'void {fun_name}_num_eval(double *{matrix_name},double *x,double *y,double *u,double *p,double Dt)'  + '{'+ '\n'*2
        source += string_num
        source += '\n}\n\n'
        
        return defs,source
        

    def sym2src(self):
        sys = self.sys
        
        t_0 = time.time()
        
        ## jac_run ################################################################
        if self.verbose: print(f'writting f_ini and g_ini code (time: {time.time()-t_0:0.3f} s)')

        defs_f_ini,source_f_ini = self.sym2str('f_ini',sys['f'],sys,'ini')
        defs_g_ini,source_g_ini = self.sym2str('g_ini',sys['g'],sys,'ini')

        if self.verbose: print(f'writting f_run and g_run code (time: {time.time()-t_0:0.3f} s)')
        
        defs_f_run,source_f_run = self.sym2str('f_run',sys['f'],sys,'run')
        defs_g_run,source_g_run = self.sym2str('g_run',sys['g'],sys,'run')

        if self.verbose: print(f'writting h_run code (time: {time.time()-t_0:0.3f} s)')   
        defs_h,source_h = self.sym2str('h',sys['h'],sys,'run')
        
        ## jac_ini ################################################################
        if self.verbose: print(f'converting jac_ini to sp_jac_ini  (time: {time.time()-t_0:0.3f} s)')
        
        jac_ini=sys['jac_ini']

        sp_jac_ini_list = _doktocsr(SparseMatrix(jac_ini))
        sys['sp_jac_ini_list'] = sp_jac_ini_list

        
        data = sp_jac_ini_list[0]
        indices = sp_jac_ini_list[1]
        indptr = sp_jac_ini_list[2]
        shape = sp_jac_ini_list[3]
        
        sp_jac_ini_num_matrix = csr_matrix((np.zeros(len(data)),indices,indptr),shape=shape)
        
        if self.verbose: print(f'running sym2rhs for sp_jac_ini (time: {time.time()-t_0:0.3f} s)')

        rhs_list = self.sym2rhs2(data,indices,indptr,shape,sys,'ini')   
        
        string_xy,string_up,string_num = self.rhs2str(rhs_list,'out',sp_jac_ini_num_matrix.data,shape,mode='dense')  
        
        if self.jac_num:
            defs_de_ini,source_de_ini = self.str2src('de_jac_ini',string_xy,string_up,string_num,matrix_name='out')
        else:
            defs_de_ini,source_de_ini = self.str2src('de_jac_ini',string_xy,string_up,'\n',matrix_name='out')
            

        string_xy,string_up,string_num = self.rhs2str(rhs_list,'out',sp_jac_ini_num_matrix.data,shape,mode='crs')
        
        if self.jac_num:
            defs_sp_ini,source_sp_ini = self.str2src('sp_jac_ini',string_xy,string_up,string_num,matrix_name='out')
        else:
            defs_sp_ini,source_sp_ini = self.str2src('sp_jac_ini',string_xy,string_up,'\n',matrix_name='out')

               
        ## jac_run ################################################################
        if self.verbose: print(f'converting jac_run to sp_jac_run  (time: {time.time()-t_0:0.3f} s)')
        
        jac_run=sys['jac_run']

        sp_jac_run_list = _doktocsr(SparseMatrix(jac_run))
        
        data = sp_jac_run_list[0]
        indices = sp_jac_run_list[1]
        indptr = sp_jac_run_list[2]
        shape = sp_jac_run_list[3]
        
        sp_jac_run_num_matrix = csr_matrix((np.zeros(len(data)),indices,indptr),shape=shape)

        
        if self.verbose: print(f'running sym2rhs for sp_jac_run (time: {time.time()-t_0:0.3f} s)')
        #sym2rhs(data,indices,indptr,shape,sys,inirun)
        #rhs_list = sym2rhs2(data,indices,indptr,shape,sys,'run')       


        sys['sp_jac_run_list'] = sp_jac_run_list
        
        data = sp_jac_run_list[0]
        indices = sp_jac_run_list[1]
        indptr = sp_jac_run_list[2]
        shape = sp_jac_run_list[3]
        
        rhs_list = self.sym2rhs2(data,indices,indptr,shape,sys,'run')   
        
        
        string_xy,string_up,string_num = self.rhs2str(rhs_list,'out',sp_jac_run_num_matrix.data,shape,mode='dense')
        
        if self.jac_num:  
            defs_de_run,source_de_run = self.str2src('de_jac_run',string_xy,string_up,string_num,matrix_name='out')
        else:
            defs_de_run,source_de_run = self.str2src('de_jac_run',string_xy,string_up,'\n',matrix_name='out')

        
        string_xy,string_up,string_num = self.rhs2str(rhs_list,'out',sp_jac_run_num_matrix.data,shape,mode='crs')
        if self.jac_num:
            defs_sp_run,source_sp_run = self.str2src('sp_jac_run',string_xy,string_up,string_num,matrix_name='out')
        else:    
            defs_sp_run,source_sp_run = self.str2src('sp_jac_run',string_xy,string_up,'\n',matrix_name='out')
        
        
        ## jac_trap ###############################################################
        if self.verbose: print(f'converting jac_trap to sp_jac_trap  (time: {time.time()-t_0:0.3f} s)')  
        
        jac_trap=sys['jac_trap']

        sp_jac_trap_list = _doktocsr(SparseMatrix(jac_trap))

        sys['sp_jac_trap_list'] = sp_jac_trap_list
        data = sp_jac_trap_list[0]
        indices = sp_jac_trap_list[1]
        indptr = sp_jac_trap_list[2]
        
        sp_jac_trap_num_matrix = csr_matrix((np.zeros(len(data)),indices,indptr),shape=shape)

        
        if self.verbose: print(f'running sym2rhs for sp_jac_trap (time: {time.time()-t_0:0.3f} s)')
        rhs_list = self.sym2rhs2(data,indices,indptr,shape,sys,'run')       

        if self.verbose: print(f'wrtting  de_jac_trap code (time: {time.time()-t_0:0.3f} s)')   
        string_xy,string_up,string_num = self.rhs2str(rhs_list,'out',sp_jac_trap_num_matrix.data,shape,mode='dense')
        
        if self.jac_num:
            defs_de_trap,source_de_trap = self.str2src('de_jac_trap',string_xy,string_up,string_num,matrix_name='out')
        else:
            defs_de_trap,source_de_trap = self.str2src('de_jac_trap',string_xy,string_up,'\n',matrix_name='out')
            

        if self.verbose: print(f'writting sp_jac_trap code (time: {time.time()-t_0:0.3f} s)')      
        string_xy,string_up,string_num = self.rhs2str(rhs_list,'out',sp_jac_trap_num_matrix.data,shape,mode='crs')
        if self.jac_num:
            defs_sp_trap,source_sp_trap = self.str2src('sp_jac_trap',string_xy,string_up,string_num,matrix_name='out')
        else:
            defs_sp_trap,source_sp_trap = self.str2src('sp_jac_trap',string_xy,string_up,'\n',matrix_name='out')
            

        ## Fu_run and Gu_run ###############################################################
        defs_de_uz   = ''
        source_de_uz = ''
        defs_sp_uz   = ''
        source_sp_uz = ''

        if not 'uz_jacs' in sys:
            self.uz_jacs = False
            self.string_u2z = '\n'
            self.u2z_comment = '#'
        else:
            self.uz_jacs = sys['uz_jacs'] 
            self.u2z_comment = ''


        if self.uz_jacs:
            self.string_u2z = 'sp_Fu_run_up_eval = jacs.lib.sp_Fu_run_up_eval\n'
            self.string_u2z+= 'sp_Gu_run_up_eval = jacs.lib.sp_Gu_run_up_eval\n'
            self.string_u2z+= 'sp_Hx_run_up_eval = jacs.lib.sp_Hx_run_up_eval\n'
            self.string_u2z+= 'sp_Hy_run_up_eval = jacs.lib.sp_Hy_run_up_eval\n'
            self.string_u2z+= 'sp_Hu_run_up_eval = jacs.lib.sp_Hu_run_up_eval\n'
            self.string_u2z+= 'sp_Fu_run_xy_eval = jacs.lib.sp_Fu_run_xy_eval\n'
            self.string_u2z+= 'sp_Gu_run_xy_eval = jacs.lib.sp_Gu_run_xy_eval\n'
            self.string_u2z+= 'sp_Hx_run_xy_eval = jacs.lib.sp_Hx_run_xy_eval\n'
            self.string_u2z+= 'sp_Hy_run_xy_eval = jacs.lib.sp_Hy_run_xy_eval\n'
            self.string_u2z+= 'sp_Hu_run_xy_eval = jacs.lib.sp_Hu_run_xy_eval\n'

            for item in ['Fu_run','Gu_run','Hx_run','Hy_run','Hu_run']:
                if self.verbose: print(f'converting {item} to sparse (time: {time.time()-t_0:0.3f} s)')  
                sp_jac_list = _doktocsr(SparseMatrix(sys[item]))  
                data = sp_jac_list[0]
                indices = sp_jac_list[1]
                indptr = sp_jac_list[2]
                shape = sp_jac_list[3]
            
                sp_jac_num_matrix = csr_matrix((np.zeros(len(data)),indices,indptr),shape=shape)
            
            
                if self.verbose: print(f'running sym2rhs for {item} (time: {time.time()-t_0:0.3f} s)')
                rhs_list = self.sym2rhs2(data,indices,indptr,shape,sys,'run')       

                if self.verbose: print(f'writting  {item} code (time: {time.time()-t_0:0.3f} s)')   
                string_xy,string_up,string_num = self.rhs2str(rhs_list,'out',sp_jac_num_matrix.data,shape,mode='dense')
            
                if self.jac_num:
                    defs_de_jac,source_de_jac = self.str2src(f'de_{item}',string_xy,string_up,string_num,matrix_name='out')
                else:
                    defs_de_jac,source_de_jac = self.str2src(f'de_{item}',string_xy,string_up,'\n',matrix_name='out')
                    

                if self.verbose: print(f'writting {item} code (time: {time.time()-t_0:0.3f} s)')      
                string_xy,string_up,string_num = self.rhs2str(rhs_list,'out',sp_jac_num_matrix.data,shape,mode='crs')
                if self.jac_num:
                    defs_sp_jac,source_sp_jac = self.str2src(f'sp_{item}',string_xy,string_up,string_num,matrix_name='out')
                else:
                    defs_sp_jac,source_sp_jac = self.str2src(f'sp_{item}',string_xy,string_up,'\n',matrix_name='out')

                defs_de_uz   += defs_de_jac
                source_de_uz += source_de_jac
                defs_sp_uz   += defs_sp_jac
                source_sp_uz += source_sp_jac
                
                if self.matrices_folder == '':
                    save_npz(f"./{sys['name']}_{item}_num.npz",sp_jac_num_matrix, compressed=True)
                else:
                    save_npz(f"./{self.matrices_folder}/{sys['name']}_{item}_num.npz",sp_jac_num_matrix, compressed=True)


        ## C sources ##############################################################
        if self.verbose: print(f'writting full source (time: {time.time()-t_0:0.3f} s)')   
        
        defs = defs_f_ini + defs_g_ini 
        defs += defs_f_run + defs_g_run  
        defs += defs_h  
        defs += defs_de_ini + defs_sp_ini 
        defs += defs_de_run + defs_sp_run 
        defs += defs_de_trap + defs_sp_trap 
        defs += defs_de_uz + defs_sp_uz
            
        source = source_f_ini + source_g_ini 
        source += source_f_run + source_g_run
        source += source_h
        source += source_de_ini + source_sp_ini 
        source += source_de_run + source_sp_run
        source += source_de_trap + source_sp_trap
        source += source_de_uz + source_sp_uz

        save_npz( f"./{self.matrices_folder}/{sys['name']}_sp_jac_ini_num.npz", sp_jac_ini_num_matrix, compressed=True)
        save_npz( f"./{self.matrices_folder}/{sys['name']}_sp_jac_run_num.npz", sp_jac_run_num_matrix, compressed=True)
        save_npz( f"./{self.matrices_folder}/{sys['name']}_sp_jac_trap_num.npz",sp_jac_trap_num_matrix, compressed=True)

        if self.verbose: print(f'Code wrote in {time.time()-t_0:0.3f} s')

        self.defs = defs
        self.source = source

        return defs,source

    def compile_module(self):
        name = self.name
        defs = self.defs
        source = self.source
        module_name = self.name + '_cffi'
        
        ffi = cffi.FFI()
        ffi.cdef(defs, override=True)
        ffi.set_source(module_name=module_name,source=source)
        t_0 = time.time()
        ffi.compile()
        if self.verbose: print(f'Compilation time: {time.time()-t_0:0.2f} s')

    def compile_module_files(self):
        
        ffi = cffi.FFI()
        name = self.name

        with open(f'./build/defs_{name}_cffi.h', 'r') as f:
            defs = f.read()
        with open(f'./build/source_{name}_cffi.c', 'r') as f:
            source = f.read()
            
        ffi.cdef(defs, override=True)
        ffi.set_source(module_name=f"{name}",source=source)
        t_0 = time.time()
        ffi.compile()
        print(f'Compilation time: {time.time()-t_0:0.2f} s')
        
    def build(self):

        sys = self.sys
        name = self.name
        sys = self.system()
        self.jacobians()
        defs,source = self.sym2src()
        if self.save_sources:
            with open(f'./build/defs_{name}_cffi.h', 'w') as f:
                f.write(defs)
            with open(f'./build/source_{name}_cffi.c', 'w') as f:
                f.write(source)
        self.compile_module()
        self.sys2num()
            
    
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
    
    # algebraic equations
    g_1 = v_q + R_a*i_q + X1d*i_d - e1q # stator
    g_2 = v_d + R_a*i_d - X1q*i_q - e1d # stator
    g_3 = i_d*v_d + i_q*v_q - p_t # active power 
    g_4 = i_d*v_q - i_q*v_d - q_t # reactive power
    g_5 = p_t - (v_t*V_0*sin(theta_t - Theta_0))/X_l  # network equation (p)
    g_6 = q_t + (v_t*V_0*cos(theta_t - Theta_0))/X_l - v_t**2/X_l  # network equation (q) 

    sys_dict = {'name':'build_test',
           'params_dict':params_dict,
           'f_list':[ddelta,domega,de1q,de1d],
           'g_list':[g_1,g_2,g_3,g_4,g_5,g_6],
           'x_list':x_list,
           'y_ini_list':y_ini_list,
           'y_run_list':y_run_list,
           'u_run_dict':u_run_dict,
           'u_ini_dict':u_ini_dict,
           'h_dict':{'p_m':p_m,'p_e':p_e, 'v_f':v_f}}
    
    bd = builder(sys_dict)
    bd.build()
    #sys2num('build_test',sys, verbose=True)
   # sys2num(sys_dict)                                                             


    
