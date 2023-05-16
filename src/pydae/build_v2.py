#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 11:53:28 2018

@author: jmmauricio

todo:
    - save matrices 
    - add dense matrices
    - add (i,j) to lists
    - add f_ini and f_run

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
import logging
import multiprocessing
from functools import partial




class builder():   

    def __init__(self,sys,verbose=False):

        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

        self.verbose = verbose
        self.sys = sys
        self.jac_num = True
        self.name = self.sys['name']
        self.save_sources = False
        self.string_u2z = ''
        self.u2z = '\#'
        self.inirun = True
        self.sparse = False


        if not os.path.exists('build'):
            os.makedirs('build')

        self.matrices_folder = 'build'

        self.f_ini_list = []
        self.f_run_list = []
        self.g_ini_list = []
        self.g_run_list = []
        self.h_list = []

        self.jac_ini_list = []
        self.jac_run_list = []
        self.jac_trap_list = []

        self.Fu_list = []
        self.Gu_list = []
        self.Hx_list = []
        self.Hy_list = []
        self.Hu_list = []

        if not 'uz_jacs' in sys:
            self.uz_jacs = False
            self.string_u2z = '\n'
            self.u2z_comment = '#'
        else:
            if sys['uz_jacs'] == True:
                self.uz_jacs = True 
                self.u2z_comment = ''
            else:
                self.uz_jacs = False 
                self.string_u2z = '\n'
                self.u2z_comment = '#'

    def dict2system(self):
        '''
        Converts input dictionary of DAE system to sympy matrices of the DAE system

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
                'u_ini_dict':u_ini_dict,
                'u_run_dict':u_run_dict,
                'h_dict':h_dict}

        Returns
        -------
        sys : TYPE
            DESCRIPTION.

        '''

        sys = self.sys
        
        t_0 = time.time()
        logging.debug('check_system')
        self.check_system()
        
        # from dict to sympy matrices and vectors
        f = sym.Matrix(sys['f_list']).T
        g = sym.Matrix(sys['g_list']).T
        x = sym.Matrix(sys['x_list']).T
        y_ini = sym.Matrix(sys['y_ini_list']).T
        y_run = sym.Matrix(sys['y_run_list']).T
        u_ini = sym.Matrix(list(sys['u_ini_dict'].keys()), real=True)
        u_run_list = [sym.Symbol(item,real=True) for item in list(sys['u_run_dict'].keys())]
        u_run = sym.Matrix(u_run_list).T 
        h =  sym.Matrix(list(sys['h_dict'].values())).T   
        

        # jacobians computation
        logging.debug('computing jacobians Fx_run,Fy_run')
        Fx = sym_jac(f,x) 
        Fy_run = sym_jac(f,y_run)
        logging.debug('computing jacobians Gx_run,Gy_run')
        Gx = sym_jac(g,x)

        Gy_run = sym_jac(g,y_run)
        logging.debug('computing jacobians Fu_run,Gu_run')
        Fu_run = sym_jac(f,u_run)
        Gu_run = sym_jac(g,u_run)

        if self.inirun: # do not compute ini jacobians if ini system is equal to run system
            logging.debug('computing jacobians Fx_ini,Fy_ini')
            Fy_ini = sym_jac(f,y_ini)
            logging.debug('computing jacobians Gy_ini')
            Gy_ini = sym_jac(g,y_ini) 
        else:
            Fy_ini = Fy_run
            Gy_ini = Gy_run

        logging.debug('computing jacobians Hx_run,Hy_run,Hu_run')
        Hx_run = sym_jac(h,x) 
        Hy_run = sym_jac(h,y_run)
        Hu_run = sym_jac(h,u_run)

        # dictionary update
        sys['f']= f
        sys['g']= g 
        sys['x']= x
        sys['y_ini']= y_ini  
        sys['y_run']= y_run      
        sys['u_ini']= u_ini 
        sys['u_run']= u_run
        sys['h']= h 
        
        sys['Fx'] = Fx
        sys['Fy_run'] = Fy_run
        sys['Gy_run'] = Gy_run

        sys['Fx'] = Fx
        sys['Fy_ini'] = Fy_ini
        sys['Gx'] = Gx
        sys['Gy_ini'] = Gy_ini
        
        sys['Fu_run'] = Fu_run
        sys['Gu_run'] = Gu_run

        sys['Hx_run'] = Hx_run
        sys['Hy_run'] = Hy_run
        sys['Hu_run'] = Hu_run
        
        Fx_ini_rows,Fx_ini_cols = np.nonzero(Fx)
        Fy_ini_rows,Fy_ini_cols = np.nonzero(Fy_ini)
        Gx_ini_rows,Gx_ini_cols = np.nonzero(Gx)
        Gy_ini_rows,Gy_ini_cols = np.nonzero(Gy_ini)
        
        sys['Fx_ini_rows'] = Fx_ini_rows 
        sys['Fx_ini_cols'] = Fx_ini_cols 
        sys['Fy_ini_rows'] = Fy_ini_rows 
        sys['Fy_ini_cols'] = Fy_ini_cols 
        sys['Gx_ini_rows'] = Gx_ini_rows 
        sys['Gx_ini_cols'] = Gx_ini_cols 
        sys['Gy_ini_rows'] = Gy_ini_rows 
        sys['Gy_ini_cols'] = Gy_ini_cols 

        Fx_run_rows,Fx_run_cols = np.nonzero(Fx)
        Fy_run_rows,Fy_run_cols = np.nonzero(Fy_run)
        Gx_run_rows,Gx_run_cols = np.nonzero(Gx)
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

        logging.debug('end checking and computing jacobians')

        self.sys = sys

    def check_system(self):
        
        if len(self.sys['f_list']) == 0:
            print('system without dynamic equations, adding dummy dynamic equation')
            x_dummy,u_dummy = sym.symbols('x_dummy,u_dummy')
            self.sys['x_list'] = ['x_dummy']
            self.sys['f_list'] = [u_dummy-x_dummy]
            self.sys['u_ini_dict'].update({'u_dummy':1.0})
            self.sys['u_run_dict'].update({'u_dummy':1.0})

        if len(self.sys['g_list']) == 0:
            print('system without algebraic equations, adding dummy algebraic equation')
            y_dummy,u_dummy = sym.symbols('y_dummy,u_dummy')
            
            self.sys['g_list'] = [u_dummy-y_dummy]
            self.sys['y_ini_list'] = ['y_dummy']
            self.sys['y_run_list'] = ['y_dummy']
            self.sys['u_ini_dict'].update({'u_dummy':1.0})
            self.sys['u_run_dict'].update({'u_dummy':1.0})
            
        y_ini_set = set(self.sys['y_ini_list'])
        contains_duplicates = len(self.sys['y_ini_list']) != len(y_ini_set)
        if contains_duplicates:
            print('error: y_ini contains duplicates')

        y_set = set(self.sys['y_run_list'])
        contains_duplicates = len(self.sys['y_run_list']) != len(y_set)
        if contains_duplicates:
            print('error: y_run contains duplicates')

        if sys['y_run_list'] == ['y_ini_list']:
            self.inirun = False

    def functions(self):
        
        logging.debug('f_ini symbolic to c and xyup')
        f = self.sys['f']
        for item in f:
            self.f_ini_list += [{'sym':item}]
        sym2c(self.f_ini_list)
        sym2xyup(self.sys,self.f_ini_list,'ini')
        logging.debug('end f_ini symbolic to c and xyup')

        logging.debug('f_run symbolic to c and xyup')
        f = self.sys['f']
        for item in f:
            self.f_run_list += [{'sym':item}]
        sym2c(self.f_run_list)
        sym2xyup(self.sys,self.f_run_list,'run')
        logging.debug('end f symbolic to c and xyup')

        logging.debug('g_ini symbolic to c and xyup')
        for item in self.sys['g']:
            self.g_ini_list += [{'sym':item}]
        sym2c(self.g_ini_list)
        sym2xyup(self.sys,self.g_ini_list,'ini')
        logging.debug('end g_ini symbolic to c and xyup')

        logging.debug('g_run symbolic to c and xyup')
        for item in self.sys['g']:
            self.g_run_list += [{'sym':item}]
        sym2c(self.g_run_list)
        sym2xyup(self.sys,self.g_run_list,'run')
        logging.debug('end g_run symbolic to c and xyup')

        logging.debug('h symbolic to c and xyup')
        for item in self.sys['h']:
            self.h_list += [{'sym':item}]
        sym2c(self.h_list)
        sym2xyup(self.sys,self.h_list,'run')
        logging.debug('end h symbolic to c and xyup')

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
                
        N_x = self.sys['N_x']
    
        Fx_run = self.sys['Fx']
        Fy_run = self.sys['Fy_run']
        Gx_run = self.sys['Gx']
        Gy_run = self.sys['Gy_run']
        
        Fx_ini = self.sys['Fx']
        Fy_ini = self.sys['Fy_ini']
        Gx_ini = self.sys['Gx']
        Gy_ini = self.sys['Gy_ini']

        Fu_run = self.sys['Fu_run']
        Gu_run = self.sys['Gu_run']  

        Hx_run = self.sys['Hx_run']
        Hy_run = self.sys['Hy_run']
        Hu_run = self.sys['Hu_run']

        logging.debug('computing jac_ini')
        jac_ini = sym.Matrix([[Fx_ini,Fy_ini],[Gx_ini,Gy_ini]]) 
        
        logging.debug('computing jac_run')
        jac_run = sym.Matrix([[Fx_run,Fy_run],[Gx_run,Gy_run]]) 
        
        logging.debug('computing jac_trap')
        eye = sym.eye(N_x, real=True)
        Dt = sym.Symbol('Dt',real=True)
        jac_trap = sym.Matrix([[eye - 0.5*Dt*Fx_run, -0.5*Dt*Fy_run],[Gx_run,Gy_run]])    

        self.sys['jac_ini']  = jac_ini
        self.sys['jac_run']  = jac_run   
        self.sys['jac_trap'] = jac_trap
        
        logging.debug('end of large jacobians computation')

        ## jac_ini
        self.jac_ini_sp = _doktocsr(SparseMatrix(jac_ini))
        self.sys['sp_jac_ini_list'] = self.jac_ini_sp
        ij_indices,dense_indices = spidx2ij(self.jac_ini_sp)
        logging.debug('jac_ini symbolic to c')
        for it,item in enumerate(self.jac_ini_sp[0]):
            self.jac_ini_list += [{'sym':item,'ij':ij_indices[it],'de_idx':dense_indices[it]}]

        sym2c(self.jac_ini_list)
        logging.debug('end jac_ini symbolic to c')

        logging.debug('jac_ini c to c xyup')
        sym2xyup(self.sys,self.jac_ini_list,'ini')
        logging.debug('end jac_ini c to c xyup')

        ## jac_run
        self.jac_run_sp = _doktocsr(SparseMatrix(jac_run))
        self.sys['sp_jac_run_list'] = self.jac_run_sp
        ij_indices,dense_indices = spidx2ij(self.jac_run_sp)

        logging.debug('jac_run symbolic to c')
        for it,item in enumerate(self.jac_run_sp[0]):
            self.jac_run_list += [{'sym':item,'ij':ij_indices[it],'de_idx':dense_indices[it]}]

        sym2c(self.jac_run_list)
        logging.debug('end jac_run symbolic to c')

        logging.debug('jac_run c to c xyup')
        sym2xyup(self.sys,self.jac_run_list,'ini')
        logging.debug('end jac_run c to c xyup')

        ## jac_trap
        self.jac_trap_sp = _doktocsr(SparseMatrix(jac_trap))
        self.sys['sp_jac_trap_list'] = self.jac_trap_sp
        ij_indices,dense_indices = spidx2ij(self.jac_trap_sp)

        logging.debug('jac_trap symbolic to c')
        for it,item in enumerate(self.jac_trap_sp[0]):
            self.jac_trap_list += [{'sym':item,'ij':ij_indices[it],'de_idx':dense_indices[it]}]

        sym2c(self.jac_trap_list)
        logging.debug('end jac_trap symbolic to c')

        logging.debug('jac_trap c to c xyup')
        sym2xyup(self.sys,self.jac_trap_list,'run')
        logging.debug('end jac_trap c to c xyup')  

        if self.uz_jacs:
            ## Fu_run
            logging.debug('Fu_run_sp symbolic to c')
            self.Fu_run_sp = _doktocsr(SparseMatrix(Fu_run))
            for item in self.Fu_run_sp[0]:
                self.Fu_list += [{'sym':item}]
            sym2c(self.Fu_list)
            logging.debug('end Fu_run_sp symbolic to c')
            logging.debug('Fu_run_sp c to c xyup')
            sym2xyup(self.sys,self.Fu_list,'run')
            logging.debug('end Fu_run_sp c to c xyup')

            ## Gu_run
            self.Gu_run_sp = _doktocsr(SparseMatrix(Gu_run))
            for item in self.Gu_run_sp[0]:
                self.Gu_list += [{'sym':item}]
            sym2c(self.Gu_list)
            sym2xyup(self.sys,self.Gu_list,'run')

            ## Hx_run
            self.Hx_run_sp = _doktocsr(SparseMatrix(Hx_run))
            for item in self.Hx_run_sp[0]:
                self.Hx_list += [{'sym':item}]
            sym2c(self.Hx_list)
            sym2xyup(self.sys,self.Hx_list,'run')

            ## Hy_run
            self.Hy_run_sp = _doktocsr(SparseMatrix(Hy_run))
            for item in self.Hy_run_sp[0]:
                self.Hy_list += [{'sym':item}]
            sym2c(self.Hy_list)
            sym2xyup(self.sys,self.Hy_list,'run')

            ## Hu_run
            self.Hu_run_sp = _doktocsr(SparseMatrix(Hu_run))
            for item in self.Hu_run_sp[0]:
                self.Hu_list += [{'sym':item}]
            sym2c(self.Hu_list)
            sym2xyup(self.sys,self.Hu_list,'run')

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

        if sys['y_run_list'] == ['y_ini_list']:
            self.inirun = False
  
    def cwrite(self): 

        defs_ini = ''
        source_ini = '' 

        defs_run = ''
        source_run = '' 

        defs_trap = ''
        source_trap = '' 

        defs_ini += f'void f_ini_eval(double *data,double *x,double *y,double *u,double *p,double Dt);\n'
        source_ini += f'void f_ini_eval(double *data,double *x,double *y,double *u,double *p,double Dt)' + '{' +'\n'*2
        for it,item in enumerate(self.f_ini_list):
            source_ini += f"data[{it}] = {item['xyup']}; \n"
        source_ini += '\n}\n\n'

        defs_run += f'void f_run_eval(double *data,double *x,double *y,double *u,double *p,double Dt);\n'
        source_run += f'void f_run_eval(double *data,double *x,double *y,double *u,double *p,double Dt)' + '{' +'\n'*2
        for it,item in enumerate(self.f_run_list):
            source_run += f"data[{it}] = {item['xyup']}; \n"
        source_run += '\n}\n\n'

        defs_ini += f'void g_ini_eval(double *data,double *x,double *y,double *u,double *p,double Dt);\n'
        source_ini += f'void g_ini_eval(double *data,double *x,double *y,double *u,double *p,double Dt)' + '{' +'\n'*2
        for it,item in enumerate(self.g_ini_list):
            source_ini += f"data[{it}] = {item['xyup']}; \n"
        source_ini += '\n}\n\n'

        defs_run += f'void g_run_eval(double *data,double *x,double *y,double *u,double *p,double Dt);\n'
        source_run += f'void g_run_eval(double *data,double *x,double *y,double *u,double *p,double Dt)' + '{' +'\n'*2
        for it,item in enumerate(self.g_run_list):
            source_run += f"data[{it}] = {item['xyup']}; \n"
        source_run += '\n}\n\n'

        defs_ini += f'void h_eval(double *data,double *x,double *y,double *u,double *p,double Dt);\n'
        source_ini += f'void h_eval(double *data,double *x,double *y,double *u,double *p,double Dt)' + '{' +'\n'*2
        for it,item in enumerate(self.h_list):
            source_ini += f"data[{it}] = {item['xyup']}; \n"
        source_ini += '\n}\n\n'

        # jac_ini dense
        defs_ini += f'void de_jac_ini_up_eval(double *data,double *x,double *y,double *u,double *p,double Dt);\n'
        source_ini += f'void de_jac_ini_up_eval(double *data,double *x,double *y,double *u,double *p,double Dt)' + '{' +'\n'*2
        for it,item in enumerate(self.jac_ini_list):
            if item['tipo'] == 'up':
                source_ini += f"data[{item['de_idx']}] = {item['xyup']}; \n"
        source_ini += '\n}\n\n'

        defs_ini += f'void de_jac_ini_xy_eval(double *data,double *x,double *y,double *u,double *p,double Dt);\n'
        source_ini += f'void de_jac_ini_xy_eval(double *data,double *x,double *y,double *u,double *p,double Dt)' + '{' +'\n'*2
        for it,item in enumerate(self.jac_ini_list):
            if item['tipo'] == 'xy':
                source_ini += f"data[{item['de_idx']}] = {item['xyup']}; \n"
        source_ini += '\n}\n\n'

        defs_ini += f'void de_jac_ini_num_eval(double *data,double *x,double *y,double *u,double *p,double Dt);\n'
        source_ini += f'void de_jac_ini_num_eval(double *data,double *x,double *y,double *u,double *p,double Dt)' + '{' +'\n'*2
        for it,item in enumerate(self.jac_ini_list):
            if item['tipo'] == 'num':   
                source_ini += f"data[{item['de_idx']}] = {item['xyup']}; \n"
        source_ini += '\n}\n\n'

        if self.sparse:
            # jac_ini sparse
            defs += f'void sp_jac_ini_up_eval(double *data,double *x,double *y,double *u,double *p,double Dt);\n'
            source += f'void sp_jac_ini_up_eval(double *data,double *x,double *y,double *u,double *p,double Dt)' + '{' +'\n'*2
            for it,item in enumerate(self.jac_ini_list):
                if item['tipo'] == 'up':
                    source += f"data[{it}] = {item['xyup']}; \n"
            source += '\n}\n\n'

            defs += f'void sp_jac_ini_xy_eval(double *data,double *x,double *y,double *u,double *p,double Dt);\n'
            source += f'void sp_jac_ini_xy_eval(double *data,double *x,double *y,double *u,double *p,double Dt)' + '{' +'\n'*2
            for it,item in enumerate(self.jac_ini_list):
                if item['tipo'] == 'xy':
                    source += f"data[{it}] = {item['xyup']}; \n"
            source += '\n}\n\n'

            defs += f'void sp_jac_ini_num_eval(double *data,double *x,double *y,double *u,double *p,double Dt);\n'
            source += f'void sp_jac_ini_num_eval(double *data,double *x,double *y,double *u,double *p,double Dt)' + '{' +'\n'*2
            for it,item in enumerate(self.jac_ini_list):
                if item['tipo'] == 'num':
                    source += f"data[{it}] = {item['xyup']}; \n"
            source += '\n}\n\n'

        # jac_run dense
        defs_run += f'void de_jac_run_up_eval(double *data,double *x,double *y,double *u,double *p,double Dt);\n'
        source_run += f'void de_jac_run_up_eval(double *data,double *x,double *y,double *u,double *p,double Dt)' + '{' +'\n'*2
        for it,item in enumerate(self.jac_run_list):
            if item['tipo'] == 'up':
                source_run += f"data[{item['de_idx']}] = {item['xyup']}; \n"
        source_run += '\n}\n\n'

        defs_run += f'void de_jac_run_xy_eval(double *data,double *x,double *y,double *u,double *p,double Dt);\n'
        source_run += f'void de_jac_run_xy_eval(double *data,double *x,double *y,double *u,double *p,double Dt)' + '{' +'\n'*2
        for it,item in enumerate(self.jac_run_list):
            if item['tipo'] == 'xy':
                source_run += f"data[{item['de_idx']}] = {item['xyup']}; \n"
        source_run += '\n}\n\n'

        defs_run += f'void de_jac_run_num_eval(double *data,double *x,double *y,double *u,double *p,double Dt);\n'
        source_run += f'void de_jac_run_num_eval(double *data,double *x,double *y,double *u,double *p,double Dt)' + '{' +'\n'*2
        for it,item in enumerate(self.jac_run_list):
            if item['tipo'] == 'num':
                source_run += f"data[{item['de_idx']}] = {item['xyup']}; \n"
        source_run += '\n}\n\n'

        if self.sparse:
            # jac_run sparse
            defs += f'void sp_jac_run_up_eval(double *data,double *x,double *y,double *u,double *p,double Dt);\n'
            source += f'void sp_jac_run_up_eval(double *data,double *x,double *y,double *u,double *p,double Dt)' + '{' +'\n'*2
            for it,item in enumerate(self.jac_run_list):
                if item['tipo'] == 'up':
                    source += f"data[{it}] = {item['xyup']}; \n"
            source += '\n}\n\n'

            defs += f'void sp_jac_run_xy_eval(double *data,double *x,double *y,double *u,double *p,double Dt);\n'
            source += f'void sp_jac_run_xy_eval(double *data,double *x,double *y,double *u,double *p,double Dt)' + '{' +'\n'*2
            for it,item in enumerate(self.jac_run_list):
                if item['tipo'] == 'xy':
                    source += f"data[{it}] = {item['xyup']}; \n"
            source += '\n}\n\n'

            defs += f'void sp_jac_run_num_eval(double *data,double *x,double *y,double *u,double *p,double Dt);\n'
            source += f'void sp_jac_run_num_eval(double *data,double *x,double *y,double *u,double *p,double Dt)' + '{' +'\n'*2
            for it,item in enumerate(self.jac_run_list):
                if item['tipo'] == 'num':
                    source += f"data[{it}] = {item['xyup']}; \n"
            source += '\n}\n\n'

        # jac_trap dense
        defs_trap += f'void de_jac_trap_up_eval(double *data,double *x,double *y,double *u,double *p,double Dt);\n'
        source_trap += f'void de_jac_trap_up_eval(double *data,double *x,double *y,double *u,double *p,double Dt)' + '{' +'\n'*2
        for it,item in enumerate(self.jac_trap_list):
            if item['tipo'] == 'up':
                source_trap += f"data[{item['de_idx']}] = {item['xyup']}; \n"
        source_trap += '\n}\n\n'

        defs_trap += f'void de_jac_trap_xy_eval(double *data,double *x,double *y,double *u,double *p,double Dt);\n'
        source_trap += f'void de_jac_trap_xy_eval(double *data,double *x,double *y,double *u,double *p,double Dt)' + '{' +'\n'*2
        for it,item in enumerate(self.jac_trap_list):
            if item['tipo'] == 'xy':
                source_trap += f"data[{item['de_idx']}] = {item['xyup']}; \n"
        source_trap += '\n}\n\n'

        defs_trap += f'void de_jac_trap_num_eval(double *data,double *x,double *y,double *u,double *p,double Dt);\n'
        source_trap += f'void de_jac_trap_num_eval(double *data,double *x,double *y,double *u,double *p,double Dt)' + '{' +'\n'*2
        for it,item in enumerate(self.jac_trap_list):
            if item['tipo'] == 'num':
                source_trap += f"data[{item['de_idx']}] = {item['xyup']}; \n"
        source_trap += '\n}\n\n'

        if self.sparse:
            # jac_trap sparse
            defs += f'void sp_jac_trap_up_eval(double *data,double *x,double *y,double *u,double *p,double Dt);\n'
            source += f'void sp_jac_trap_up_eval(double *data,double *x,double *y,double *u,double *p,double Dt)' + '{' +'\n'*2
            for it,item in enumerate(self.jac_trap_list):
                if item['tipo'] == 'up':
                    source += f"data[{it}] = {item['xyup']}; \n"
            source += '\n}\n\n'

            defs += f'void sp_jac_trap_xy_eval(double *data,double *x,double *y,double *u,double *p,double Dt);\n'
            source += f'void sp_jac_trap_xy_eval(double *data,double *x,double *y,double *u,double *p,double Dt)' + '{' +'\n'*2
            for it,item in enumerate(self.jac_trap_list):
                if item['tipo'] == 'xy':
                    source += f"data[{it}] = {item['xyup']}; \n"
            source += '\n}\n\n'

            defs += f'void sp_jac_trap_num_eval(double *data,double *x,double *y,double *u,double *p,double Dt);\n'
            source += f'void sp_jac_trap_num_eval(double *data,double *x,double *y,double *u,double *p,double Dt)' + '{' +'\n'*2
            for it,item in enumerate(self.jac_trap_list):
                if item['tipo'] == 'num':
                    source += f"data[{it}] = {item['xyup']}; \n"
            source += '\n}\n\n'

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

            # Fu, Gu, Hx, Hy, Hu sparse
            matrices = [{'name':'Fu_run','lista':self.Fu_list,'matrix':self.Fu_run_sp,'de_sp':['sp'],'ini_run':['run']},
                        {'name':'Gu_run','lista':self.Gu_list,'matrix':self.Gu_run_sp,'de_sp':['sp'],'ini_run':['run']},
                        {'name':'Hx_run','lista':self.Hx_list,'matrix':self.Hx_run_sp,'de_sp':['sp'],'ini_run':['run']},
                        {'name':'Hy_run','lista':self.Hy_list,'matrix':self.Hy_run_sp,'de_sp':['sp'],'ini_run':['run']},
                        {'name':'Hu_run','lista':self.Hu_list,'matrix':self.Hu_run_sp,'de_sp':['sp'],'ini_run':['run']},]

            for item in matrices:
                for de_sp in item['de_sp']:
                    for xyup in ['up','xy','num']:
                        defs += f'void {de_sp}_{item["name"]}_{xyup}_eval(double *data,double *x,double *y,double *u,double *p,double Dt);\n'
                        source += f'void {de_sp}_{item["name"]}_{xyup}_eval(double *data,double *x,double *y,double *u,double *p,double Dt)' + '{' +'\n'*2
                        for it,item2 in enumerate(item["lista"]):
                            if item2['tipo'] == xyup:
                                source += f"data[{it}] = {item2['xyup']}; \n"
                        source += '\n}\n\n' 
                matrix = csr_matrix((np.zeros(len(item['matrix'][0])),item['matrix'][1],item['matrix'][2]),shape=item['matrix'][3])
                save_npz( f"./{self.matrices_folder}/{self.sys['name']}_{item['name']}_num.npz", matrix, compressed=True)

        sp_jac_ini_num_matrix = csr_matrix((np.zeros(len(self.jac_ini_sp[0])),self.jac_ini_sp[1],self.jac_ini_sp[2]),shape=self.jac_ini_sp[3])
        sp_jac_run_num_matrix = csr_matrix((np.zeros(len(self.jac_run_sp[0])),self.jac_run_sp[1],self.jac_run_sp[2]),shape=self.jac_run_sp[3])
        sp_jac_trap_num_matrix = csr_matrix((np.zeros(len(self.jac_trap_sp[0])),self.jac_trap_sp[1],self.jac_trap_sp[2]),shape=self.jac_trap_sp[3])

        save_npz( f"./{self.matrices_folder}/{self.sys['name']}_sp_jac_ini_num.npz", sp_jac_ini_num_matrix, compressed=True)
        save_npz( f"./{self.matrices_folder}/{self.sys['name']}_sp_jac_run_num.npz", sp_jac_run_num_matrix, compressed=True)
        save_npz( f"./{self.matrices_folder}/{self.sys['name']}_sp_jac_trap_num.npz",sp_jac_trap_num_matrix, compressed=True)


        self.defs_ini = defs_ini
        self.source_ini = source_ini
        self.defs_run = defs_run
        self.source_run = source_run
        self.defs_trap = defs_trap
        self.source_trap = source_trap

        with open(f'./build/defs_ini_{self.name}_cffi.h', 'w') as fobj:
            fobj.write(self.defs_ini)
        with open(f'./build/source_ini_{self.name}_cffi.c', 'w') as fobj:
            fobj.write(self.source_ini)
        with open(f'./build/defs_run_{self.name}_cffi.h', 'w') as fobj:
            fobj.write(self.defs_run)
        with open(f'./build/source_run_{self.name}_cffi.c', 'w') as fobj:
            fobj.write(self.source_run)
        with open(f'./build/defs_trap_{self.name}_cffi.h', 'w') as fobj:
            fobj.write(self.defs_trap)
        with open(f'./build/source_trap_{self.name}_cffi.c', 'w') as fobj:
            fobj.write(self.source_trap)

    def compile(self):
        
        logging.debug('start compiling ini module')
        ffi_ini = cffi.FFI()
        ffi_ini.cdef(self.defs_ini, override=True)
        ffi_ini.set_source(module_name=f"{self.name}_ini_cffi",source=self.source_ini)
        ffi_ini.compile()
        logging.debug('end compiling ini module')

        logging.debug('start compiling run module')
        ffi_run = cffi.FFI()
        ffi_run.cdef(self.defs_run, override=True)
        ffi_run.set_source(module_name=f"{self.name}_run_cffi",source=self.source_run)
        ffi_run.compile()
        logging.debug('end compiling run module')

        logging.debug('start compiling trap module')
        ffi_trap = cffi.FFI()
        ffi_trap.cdef(self.defs_trap, override=True)
        ffi_trap.set_source(module_name=f"{self.name}_trap_cffi",source=self.source_trap)
        ffi_trap.compile()
        logging.debug('end compiling trap module')


    def template(self):
        sys = self.sys

        t_0 = time.time()
        
        name = sys['name']
        N_x = sys['N_x']
        N_y = sys['N_y']
        N_z = sys['N_z']
        N_u = sys['N_u']

        class_template = pkgutil.get_data(__name__, "templates/class_dae_template_v2.py").decode().replace('\r\n','\n') 
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
    
        
    def spmatrix2def(self,spmatrix_list,matrix_name):
        
        string = ''
        string += f'def {matrix_name}_vectors():\n\n'
        string += f'{" "*4}{matrix_name}_ia = ' + str(spmatrix_list[1]) + '\n'
        string += f'{" "*4}{matrix_name}_ja = ' + str(spmatrix_list[2]) + '\n'    
        string += f'{" "*4}{matrix_name}_nia = ' + str(spmatrix_list[3][0])  + '\n'
        string += f'{" "*4}{matrix_name}_nja = ' + str(spmatrix_list[3][1])  + '\n'
        string += f'{" "*4}return {matrix_name}_ia, {matrix_name}_ja, {matrix_name}_nia, {matrix_name}_nja \n'
        
        return string   

def sym_jac(f,x):
    
    N_f = len(f)
    N_x = len(x)

    J = sym.MutableSparseMatrix(N_f, N_x, {})
    
    for irow in range(N_f):
        str_f = str(f[irow])
        for icol in range(N_x):
            if str(x[icol]) in str_f:
                J[irow,icol] = f[irow].diff(x[icol])

    return J

def spidx2ij(csr_matrix_list):

        data    = csr_matrix_list[0]
        indices = csr_matrix_list[1]
        indptr  = csr_matrix_list[2]
        shape   = csr_matrix_list[3]
        N_rows,N_cols = shape

        ij_indices = []
        dense_indices = []
        for i in range(shape[0]):
            row_col_start = indptr[i]
            row_col_end   = indptr[i+1]
            for j in indices[row_col_start:row_col_end]:
                ij_indices += [(i,j)]
                dense_indices += [i*N_cols+j]

        return ij_indices,dense_indices

def sym2c(full_list):
    '''
    Converts sympy symbolic expressions to c code
    '''

    for item in full_list:
        sym_expr = item['sym']
        c_expr = sym.ccode(sym_expr)
        item.update({'ccode':c_expr})


def sym2xyup(sys,full_list,inirun):
    '''
    Converts symbols to vectors x, y, u and p
    '''

    full_string = '#'.join([element['ccode'] for element in full_list])
    i = 0
    for symbol in sys['x']:
        full_string = re.sub(f"\\b{symbol}\\b"  ,f'x[{i}]',full_string)
        i+=1

    i = 0
    for symbol in sys[f'y_{inirun}']:
        full_string = re.sub(f"\\b{symbol}\\b"  ,f'y[{i}]',full_string)
        i+=1

    i = 0
    for symbol in sys[f'u_{inirun}']:
        full_string = re.sub(f"\\b{symbol}\\b"  ,f'u[{i}]',full_string)
        i+=1

    i = 0
    for symbol in sys['params_dict']:
        full_string = re.sub(f"\\b{symbol}\\b"  ,f'p[{i}]',full_string)
        i+=1

    for item,string in zip(full_list,full_string.split('#')):
        tipo = 'num'
        if 'x[' in string or 'y[' in string:
            tipo = 'xy' 
        elif 'p[' in string or 'u[' in string or 'Dt' in string:
            tipo = 'up' 
        item.update({'xyup':string,'tipo':tipo})


def sym2xyup_process(sys,inirun,element):
    '''
    Converts symbols to vectors x, y, u and p
    '''

    string = element['ccode']
    i = 0
    for symbol in sys['x']:
        string = re.sub(f"\\b{symbol}\\b"  ,f'x[{i}]',string)
        i+=1

    i = 0
    for symbol in sys[f'y_{inirun}']:
        string = re.sub(f"\\b{symbol}\\b"  ,f'y[{i}]',string)
        i+=1

    i = 0
    for symbol in sys[f'u_{inirun}']:
        string = re.sub(f"\\b{symbol}\\b"  ,f'u[{i}]',string)
        i+=1

    i = 0
    for symbol in sys['params_dict']:
        string = re.sub(f"\\b{symbol}\\b"  ,f'p[{i}]',string)
        i+=1

    tipo = 'num'
    if 'x[' in string or 'y[' in string:
        tipo = 'xy' 
    elif 'p[' in string or 'u[' in string or 'Dt' in string:
        tipo = 'up' 

    return string,tipo 

def sym2xyup_mp(sys,full_list,inirun):

    pool = multiprocessing.Pool()

    # use the pool to process each string in parallel
    sym2xyup_p = partial(sym2xyup_process,sys,inirun)
    
    logging.debug('start sym2xyup pool')
    xyup_tipo_list = pool.map(sym2xyup_p, full_list)

    pool.close()
    pool.join()
    logging.debug('end sym2xyup pool')

    logging.debug('start full_list update with xyup and tipo')
    for item,xyup_tipo in zip(full_list,xyup_tipo_list):
        xyup,tipo = xyup_tipo
        item.update({'xyup':xyup,'tipo':tipo})
    logging.debug('end full_list update with xyup and tipo')





if __name__ == '__main__':
    from pydae.bmapu.bmapu_builder import  bmapu
    grid = bmapu(r"./smib.json")
    grid.construct('smib')

    b = builder(grid.sys_dict)
    b.dict2system()
    b.functions()
    b.jacobians()
    b.cwrite()
    #b.compile()

