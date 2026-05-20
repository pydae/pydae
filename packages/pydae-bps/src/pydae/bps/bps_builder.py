# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 23:52:55 2021

@author: jmmau
"""

import numpy as np
import sympy as sym
import json
import os
import hjson
from pydae.core.builder.casadi_builder import MathBackend
from pydae.core.builder.sympy_builder import Builder as db
from pydae.bps.syns.syns import add_syns
from pydae.bps.vscs.vscs import add_vscs
from pydae.bps.vsgs.vsgs import add_vsgs
from pydae.bps.wecs.wecs import add_wecs
from pydae.bps.weccs.weccs import add_weccs
from pydae.bps.ppcs.ppcs import add_ppcs
from pydae.bps.pvs.pvs import add_pvs
from pydae.bps.loads.loads import add_loads
from pydae.bps.sources.sources import add_sources
from pydae.bps.miscellaneous.miscellaneous import add_miscellaneous
from pydae.bps.pods.pods import add_pods
from pydae.bps.miscellaneous.banks import add_banks
from pydae.bps.miscellaneous.agc import add_agc
from pydae.bps.lines.lines import add_lines

# todo:
    # S_base can't be modified becuase impedances in element base are computed
    # in S_base only in the build
class BpsBuilder:
    '''
    

    Parameters
    ----------
    data_input : string or dict
        File path to the system data information or dictionary with the information.
    use_casadi : bool
        If True, use CasADi SX symbols instead of SymPy symbols for the DAE.
        Enables the CasADi backend (no C compiler needed).

    Returns
    -------
    dict
        Dictionary with the equations for pydae. 
        
    {
     'sys':{'name':'pf_1','S_base':100e6},       
     'buses':[{'name':'GRI','P_W':0.0,'Q_var':0.0,'U_kV':66.0, 'type':'slack'},
              {'name':'POI','P_W':0.0,'Q_var':0.0,'U_kV':66.0},
              {'name':'PMV','P_W':0.0,'Q_var':0.0,'U_kV':20.0}],
     'lines':[{'bus_j':'GRI','bus_k':'POI','X_km':0.4,'R_km':0.12,'km':20},
              {'bus_j':'POI','bus_k':'PMV','X_pu':0.04,'R_pu':0.01, 'S_mva':50.0}]
    }
        

    '''

    def __init__(self, data_input='', testing=False, use_casadi=False):
        
        if type(data_input) == str:
            if 'http' in data_input:
                url = data_input
                resp = requests.get(url)
                data = hjson.loads(resp.text)
            else:
                if os.path.splitext(data_input)[1] == '.json':
                    with open(data_input, 'r', encoding='utf-8') as fobj:
                        data = json.loads(fobj.read().replace("'", '"'))
                if os.path.splitext(data_input)[1] == '.hjson':
                    with open(data_input, 'r', encoding='utf-8') as fobj:
                        data = hjson.loads(fobj.read().replace("'", '"'))
        elif isinstance(data_input, dict):
            data = data_input
        else:
            raise ValueError(f"data_input must be a str (file path or URL) or dict, got {type(data_input)}")
            
        self.data = data

        if not 'lines' in self.data:
            self.data['lines'] = []
        if not 'shunts' in self.data:
            self.data['shunts'] = []
        if not 'transformers' in self.data:
            self.data['transformers'] = []
        if not 'loads' in self.data:
            self.data['loads'] = []

        self.system = data['system']
        self.buses = data['buses']
        self.lines = data['lines']
        self.shunts = data['shunts']
        self.transformers = data['transformers']
        self.loads = data['loads']

        self.use_casadi = use_casadi
        self.backend = MathBackend(use_casadi)

        self.S_base = self.backend.symbols("S_base")
        self.N_bus = len(self.buses)
        self.N_branch = 3*len(self.lines) + len(self.shunts) + 2*len(self.transformers)

        self.dae = {'f':[],'g':[],'x':[],'y_ini':[],'y_run':[],
                    'u_ini_dict':{},'u_run_dict':{},'params_dict':{},
                    'h_dict':{},'xy_0_dict':{}}

        self.dae['params_dict'].update({'S_base': self.system['S_base']})

        self.uz_jacs = True
        self.verbose = False
        self.testing = testing
                
    def contruct_grid(self):
        
        N_branch = self.N_branch
        N_bus = self.N_bus
        sys = self.system
        
        S_base = self.backend.symbols('S_base')

        self.A = self.backend.zeros(N_branch, N_bus)
        self.G_primitive = self.backend.zeros(N_branch, N_branch)
        self.B_primitive = self.backend.zeros(N_branch, N_branch)
        self.buses_list = [bus['name'] for bus in self.buses]
        self.it = 0

        add_lines(self) 
        # for line in self.lines:
    
        #     bus_j = line['bus_j']
        #     bus_k = line['bus_k']
    
        #     idx_j = self.buses_list.index(bus_j)
        #     idx_k = self.buses_list.index(bus_k)    
    
        #     A[self.self.it,idx_j] = 1
        #     A[self.self.it,idx_k] =-1   
        #     A[self.self.it+1,idx_j] = 1
        #     A[self.self.it+2,idx_k] = 1   
            
        #     line_name = f"{bus_j}_{bus_k}"
        #     g_jk = sym.Symbol(f"g_{line_name}", real=True) 
        #     b_jk = sym.Symbol(f"b_{line_name}", real=True) 
        #     bs_jk = sym.Symbol(f"bs_{line_name}", real=True) 
        #     G_primitive[self.it,self.it] = g_jk
        #     B_primitive[self.it,self.it] = b_jk
        #     B_primitive[self.it+1,self.it+1] = bs_jk/2
        #     B_primitive[self.it+2,self.it+2] = bs_jk/2

        #     if not 'thermal' in line:
        #         line.update({'thermal':False})      

        #     if 'X_pu' in line:
        #         if 'S_mva' in line: S_line = 1e6*line['S_mva']
        #         R = line['R_pu']*sys['S_base']/S_line  # in pu of the system base
        #         X = line['X_pu']*sys['S_base']/S_line  # in pu of the system base
        #         G =  R/(R**2+X**2)
        #         B = -X/(R**2+X**2)
        #         self.dae['params_dict'].update({f"g_{line_name}":G})
        #         self.dae['params_dict'].update({f'b_{line_name}':B})
    
        #     if 'X' in line:
        #         bus_idx = self.buses_list.index(line['bus_j'])
        #         U_base = self.buses[bus_idx]['U_kV']*1000
        #         Z_base = U_base**2/sys['S_base']
        #         R = line['R']/Z_base  # in pu of the system base
        #         X = line['X']/Z_base  # in pu of the system base
        #         G =  R/(R**2+X**2)
        #         B = -X/(R**2+X**2)
        #         self.dae['params_dict'].update({f"g_{line_name}":G})
        #         self.dae['params_dict'].update({f'b_{line_name}':B})
    
        #     if 'X_km' in line:
        #         bus_idx = self.buses_list.index(line['bus_j'])
        #         U_base = self.buses[bus_idx]['U_kV']*1000
        #         Z_base = U_base**2/sys['S_base']
        #         if line['thermal']:
        #             R = sym.Symbol(f"R_{line_name}", real=True)
        #             R_N = line['R_km']*line['km']/Z_base  # in pu of the system base
        #             self.dae['u_ini_dict'].update({str(R):R_N})
        #         else:    
        #             R = line['R_km']*line['km']/Z_base  # in pu of the system base

        #         X = line['X_km']*line['km']/Z_base  # in pu of the system base
        #         G =  R/(R**2+X**2)
        #         B = -X/(R**2+X**2)
        #         self.dae['params_dict'].update({f"g_{line_name}":G})
        #         self.dae['params_dict'].update({f'b_{line_name}':B})        
    
        #     self.dae['params_dict'].update({f'bs_{line_name}':0.0})
        #     if 'Bs_pu' in line:
        #         if 'S_mva' in line: S_line = 1e6*line['S_mva']
        #         Bs = line['Bs_pu']*S_line/sys['S_base']  # in pu of the system base
        #         bs = Bs
        #         self.dae['params_dict'][f'bs_{line_name}'] = bs
     
        #     if 'Bs_km' in line:
        #         bus_idx = self.buses_list.index(line['bus_j'])
        #         U_base = self.buses[bus_idx]['U_kV']*1000
        #         Z_base = U_base**2/sys['S_base']
        #         Y_base = 1.0/Z_base
        #         Bs = line['Bs_km']*line['km']/Y_base # in pu of the system base
        #         bs = Bs 
        #         self.dae['params_dict'][f'bs_{line_name}'] = bs
                
        #     self.it += 3

        for trafo in self.transformers:
    
            bus_j = trafo['bus_j']
            bus_k = trafo['bus_k']
    
            idx_j = self.buses_list.index(bus_j)
            idx_k = self.buses_list.index(bus_k)    
    
            self.A[self.it,idx_j] = 1
            self.A[self.it+1,idx_k] = 1  
            
            trafo_name = f"{bus_j}_{bus_k}"
            g_jk = self.backend.symbols(f"g_cc_{trafo_name}")
            b_jk = self.backend.symbols(f"b_cc_{trafo_name}")
            tap = self.backend.symbols(f"tap_{trafo_name}")
            ang = self.backend.symbols(f"ang_{trafo_name}")
            a_s = tap * self.backend.cos(ang)
            b_s = tap * self.backend.sin(ang)

            if self.backend.use_casadi:
                denom = a_s**2 + b_s**2
                Y_cc_re = g_jk
                Y_cc_im = b_jk
                Yp_00_re = Y_cc_re / denom
                Yp_00_im = Y_cc_im / denom
                Yp_01_re = -(Y_cc_re * a_s + Y_cc_im * b_s) / denom
                Yp_01_im = -(Y_cc_im * a_s - Y_cc_re * b_s) / denom
                Yp_10_re = -(Y_cc_re * a_s - Y_cc_im * b_s) / denom
                Yp_10_im = -(Y_cc_im * a_s + Y_cc_re * b_s) / denom
                Yp_11_re = Y_cc_re
                Yp_11_im = Y_cc_im
                self.G_primitive[self.it:self.it+2, self.it:self.it+2] = self.backend.Matrix([
                    [Yp_00_re, Yp_01_re],
                    [Yp_10_re, Yp_11_re],
                ])
                self.B_primitive[self.it:self.it+2, self.it:self.it+2] = self.backend.Matrix([
                    [Yp_00_im, Yp_01_im],
                    [Yp_10_im, Yp_11_im],
                ])
            else:
                Y_cc = g_jk + self.backend.I * b_jk

                Y_primitive = self.backend.Matrix([
                    [Y_cc / (a_s**2 + b_s**2), -Y_cc / (a_s - self.backend.I * b_s)],
                    [-Y_cc / (a_s + self.backend.I * b_s), Y_cc]
                ])

                self.G_primitive[self.it:self.it+2, self.it:self.it+2] = self.backend.re(Y_primitive)
                self.B_primitive[self.it:self.it+2, self.it:self.it+2] = self.backend.im(Y_primitive)


            if 'X_pu' in trafo:
                if 'S_mva' in trafo: S_trafo = 1e6*trafo['S_mva']
                R = trafo['R_pu']*sys['S_base']/S_trafo  # in pu of the system base
                X = trafo['X_pu']*sys['S_base']/S_trafo  # in pu of the system base
                G =  R/(R**2+X**2)
                B = -X/(R**2+X**2)
                self.dae['params_dict'].update({f"g_cc_{trafo_name}":G})
                self.dae['params_dict'].update({f'b_cc_{trafo_name}':B})
                tap_m = 1.0
                if 'tap_m' in trafo:
                    tap_m = trafo['tap_m']
                self.dae['params_dict'].update({f'tap_{trafo_name}':tap_m})
                self.dae['params_dict'].update({f'ang_{trafo_name}':0.0})

    
            if 'X' in trafo:
                bus_idx = self.buses_list.index(trafo['bus_j'])
                U_base = self.buses[bus_idx]['U_kV']*1000
                Z_base = U_base**2/sys['S_base']
                R = trafo['R']/Z_base  # in pu of the system base
                X = trafo['X']/Z_base  # in pu of the system base
                G =  R/(R**2+X**2)
                B = -X/(R**2+X**2)
                self.dae['params_dict'].update({f"g_cc_{trafo_name}":G})
                self.dae['params_dict'].update({f'b_cc_{trafo_name}':B})
                self.dae['params_dict'].update({f'tap_{trafo_name}':1.0})
                self.dae['params_dict'].update({f'ang_{trafo_name}':0.0})
     
            self.it += 2


        for shunt in self.shunts:
    
            bus_j = shunt['bus']   
            idx_j = self.buses_list.index(bus_j)
    
            self.A[self.it,idx_j] = 1
             
            shunt_name = f"{bus_j}"
            g_j = self.backend.symbols(f"g_shunt_{shunt_name}")
            b_j = self.backend.symbols(f"b_shunt_{shunt_name}") 
            self.G_primitive[self.it,self.it] = g_j
            self.B_primitive[self.it,self.it] = b_j

            
            if 'X_pu' in shunt:
                if 'S_mva' in shunt: S_line = 1e6*shunt['S_mva']
                R = shunt['R_pu']*sys['S_base']/S_line  # in pu of the system base
                X = shunt['X_pu']*sys['S_base']/S_line  # in pu of the system base
                G =  R/(R**2+X**2)
                B = -X/(R**2+X**2)
                self.dae['params_dict'].update({f"g_shunt_{shunt_name}":G})
                self.dae['params_dict'].update({f'b_shunt_{shunt_name}':B})
    
            if 'X' in shunt:
                U_base = self.buses[idx_j]['U_kV']*1000
                Z_base = U_base**2/sys['S_base']
                R = shunt['R']/Z_base  # in pu of the system base
                X = shunt['X']/Z_base  # in pu of the system base
                G =  R/(R**2+X**2)
                B = -X/(R**2+X**2)
                self.dae['params_dict'].update({f"g_shunt_{shunt_name}":G})
                self.dae['params_dict'].update({f'b_shunt_{shunt_name}':B})
                
            self.it += 1    
    
        G = self.A.T @ self.G_primitive @ self.A
        B = self.A.T @ self.B_primitive @ self.A

        g = self.backend.zeros(2 * N_bus, 1)

        for j in range(N_bus):
            bus_j_name = self.buses_list[j]
            P_j = self.backend.symbols(f"P_{bus_j_name}")
            Q_j = self.backend.symbols(f"Q_{bus_j_name}")
            g[2*j]   = -P_j / S_base
            g[2*j+1] = -Q_j / S_base
            for k in range(N_bus):

                bus_k_name = self.buses_list[k]
                V_j = self.backend.symbols(f"V_{bus_j_name}")
                V_k = self.backend.symbols(f"V_{bus_k_name}")
                theta_j = self.backend.symbols(f"theta_{bus_j_name}")
                theta_k = self.backend.symbols(f"theta_{bus_k_name}")
                g[2*j]   += V_j * V_k * (G[j,k] * self.backend.cos(theta_j - theta_k) + B[j,k] * self.backend.sin(theta_j - theta_k))
                g[2*j+1] += V_j * V_k * (G[j,k] * self.backend.sin(theta_j - theta_k) - B[j,k] * self.backend.cos(theta_j - theta_k))
                self.dae['h_dict'].update({f"V_{bus_j_name}": V_j})
            bus = self.buses[j]
            bus_name = bus['name']
            if 'type' in bus:
                if bus['type'] == 'slack':
                    self.dae['y_ini'] += [P_j]
                    self.dae['y_ini'] += [Q_j]
                    self.dae['y_run'] += [P_j]
                    self.dae['y_run'] += [Q_j]
                    self.dae['u_ini_dict'].update({f"V_{bus_name}": 1.0})
                    self.dae['u_ini_dict'].update({f"theta_{bus_name}": 0.0})
                    self.dae['u_run_dict'].update({f"V_{bus_name}": 1.0})
                    self.dae['u_run_dict'].update({f"theta_{bus_name}": 0.0})
            else:
                self.dae['y_ini'] += [V_j]
                self.dae['y_ini'] += [theta_j]
                self.dae['y_run'] += [V_j]
                self.dae['y_run'] += [theta_j]
                self.dae['u_ini_dict'].update({f"P_{bus_name}": bus['P_W']})
                self.dae['u_ini_dict'].update({f"Q_{bus_name}": bus['Q_var']})
                self.dae['u_run_dict'].update({f"P_{bus_name}": bus['P_W']})
                self.dae['u_run_dict'].update({f"Q_{bus_name}": bus['Q_var']})
                self.dae['xy_0_dict'].update({str(V_j): 1.0, str(theta_j): 0.0})
                
            self.dae['params_dict'].update({f'U_{bus_name}_n': bus['U_kV'] * 1000})

        if self.use_casadi:
            self.dae['g'] += [g[i] for i in range(g.size1())]
        else:
            self.dae['g'] += list(g)     
    
        if False:
            v_sym_list = []
            for bus in self.buses_list:
                V_m = sym.Symbol(f'V_{bus}',real=True)
                V_a = sym.Symbol(f'theta_{bus}',real=True)
                v_sym_list += [V_m*sym.exp(sym.I*V_a)]
    
            sym.Matrix(v_sym_list)
    
            I_lines = (G_primitive+1j*B_primitive) * A * sym.Matrix(v_sym_list)
    
            self.it = 0
            for line in self.lines:
                I_jk_r = sym.Symbol(f"I_{line['bus_j']}_{line['bus_k']}_r", real=True)
                I_jk_i = sym.Symbol(f"I_{line['bus_j']}_{line['bus_k']}_i", real=True)
                self.dae['g'] += [-I_jk_r + sym.re(I_lines[self.it])]
                self.dae['g'] += [-I_jk_i + sym.im(I_lines[self.it])]
                self.dae['y_ini'] += [I_jk_r]
                self.dae['y_ini'] += [I_jk_i]
                self.dae['y_run'] += [I_jk_r]
                self.dae['y_run'] += [I_jk_i]
                self.it += 1
        
        ### Lines monitoring
        for line in self.lines:
    
            bus_j = line['bus_j']
            bus_k = line['bus_k']

            line_name = f"{bus_j}_{bus_k}"
    
            idx_j = self.buses_list.index(bus_j)
            idx_k = self.buses_list.index(bus_k)  

            V_j = self.backend.symbols(f"V_{bus_j}")
            V_k = self.backend.symbols(f"V_{bus_k}")
            theta_j = self.backend.symbols(f"theta_{bus_j}")
            theta_k = self.backend.symbols(f"theta_{bus_k}")

            b_ij_p = 0.0
            if f'bs_{line_name}' in self.dae['params_dict']:
                b_ij_p = self.dae['params_dict'][f'bs_{line_name}']


            if not 'monitor' in line: line.update({'monitor':False})
            if 'monitor' in line or 'dtr' in line:
                if line['monitor'] or line['dtr']:
                    # self.dae['h_dict'].update({f"p_line_{bus_j}_{bus_k}":P_line_to})
                    # self.dae['h_dict'].update({f"q_line_{bus_j}_{bus_k}":Q_line_to}) 
                    # self.dae['h_dict'].update({f"p_line_{bus_k}_{bus_j}":P_line_from})
                    # self.dae['h_dict'].update({f"q_line_{bus_k}_{bus_j}":Q_line_from}) 


                    G_jk = G[idx_j,idx_k] 
                    B_jk = B[idx_j,idx_k] 
                    theta_jk = theta_j - theta_k
                    P_line_to   = V_j*V_k*(G_jk*self.backend.cos(theta_jk) + B_jk*self.backend.sin(theta_jk)) - V_j**2*(G_jk)
                    Q_line_to   = V_j*V_k*(G_jk*self.backend.sin(theta_jk) - B_jk*self.backend.cos(theta_jk)) + V_j**2*(B_jk)
                    P_line_from = V_j*V_k*(G_jk*self.backend.cos(-theta_jk) + B_jk*self.backend.sin(-theta_jk)) - V_k**2*(G_jk)
                    Q_line_from = V_j*V_k*(G_jk*self.backend.sin(-theta_jk) - B_jk*self.backend.cos(-theta_jk)) + V_k**2*(B_jk)

                    p_line_to_pu, q_line_to_pu = self.backend.symbols(f"p_line_pu_{bus_j}_{bus_k}, q_line_pu_{bus_j}_{bus_k}")
                    p_line_from_pu, q_line_from_pu = self.backend.symbols(f"p_line_pu_{bus_k}_{bus_j}, q_line_pu_{bus_k}_{bus_j}")

                    self.dae['g'] += [p_line_to_pu - P_line_to]
                    self.dae['g'] += [q_line_to_pu - Q_line_to]
                    self.dae['g'] += [p_line_from_pu - P_line_from]
                    self.dae['g'] += [q_line_from_pu - Q_line_from]

                    self.dae['y_ini'] += [p_line_to_pu,q_line_to_pu,p_line_from_pu,q_line_from_pu]
                    self.dae['y_run'] += [p_line_to_pu,q_line_to_pu,p_line_from_pu,q_line_from_pu]
                    
                    U_base = self.buses[idx_j]['U_kV']*1000
                    I_base = S_base/(np.sqrt(3)*U_base)

                    self.dae['h_dict'].update({f'p_line_{bus_j}_{bus_k}':p_line_to_pu*S_base})
                    self.dae['h_dict'].update({f'q_line_{bus_j}_{bus_k}':q_line_to_pu*S_base})
                    self.dae['h_dict'].update({f'p_line_{bus_k}_{bus_j}':p_line_from_pu*S_base})
                    self.dae['h_dict'].update({f'q_line_{bus_k}_{bus_j}':q_line_from_pu*S_base})

                    I_j_k, I_k_j = self.backend.symbols(f"I_{bus_j}_{bus_k}, I_{bus_k}_{bus_j}")

                    I_j_k_eq = (  p_line_to_pu**2 +   q_line_to_pu**2)**0.5/V_j*I_base
                    I_k_j_eq = (p_line_from_pu**2 + q_line_from_pu**2)**0.5/V_k*I_base

                    self.dae['g'] += [I_j_k_eq - I_j_k]

                    self.dae['y_ini'] += [I_j_k]
                    self.dae['y_run'] += [I_j_k]


                    self.dae['g'] += [I_k_j_eq - I_k_j]

                    self.dae['y_ini'] += [I_k_j]
                    self.dae['y_run'] += [I_k_j]

                    self.dae['h_dict'].update({f'I_line_{bus_j}_{bus_k}':I_j_k})
                    self.dae['h_dict'].update({f'I_line_{bus_k}_{bus_j}':I_k_j})

                    self.dae['xy_0_dict'].update({str(p_line_to_pu):0.1})
                    self.dae['xy_0_dict'].update({str(p_line_from_pu):-0.1})
                    self.dae['xy_0_dict'].update({str(I_j_k):0.1})
                    self.dae['xy_0_dict'].update({str(I_k_j):-0.1})

        ### Transformer monitoring
        # Tap-aware branch flows for transformers carrying `monitor: true`.
        # Mirrors the line monitoring loop above so downstream code that
        # references p_line_pu_{j}_{k} / q_line_pu_{j}_{k} works for tap
        # branches as well. Tap is on the bus_j (primary) side; with tap=1
        # the formulas reduce to the line equations exactly.
        for trafo in self.transformers:
            if not trafo.get('monitor', False):
                continue

            bus_j = trafo['bus_j']
            bus_k = trafo['bus_k']
            trafo_name = f"{bus_j}_{bus_k}"

            V_j = self.backend.symbols(f"V_{bus_j}")
            V_k = self.backend.symbols(f"V_{bus_k}")
            theta_j = self.backend.symbols(f"theta_{bus_j}")
            theta_k = self.backend.symbols(f"theta_{bus_k}")
            G_t = self.backend.symbols(f"g_cc_{trafo_name}")
            B_t = self.backend.symbols(f"b_cc_{trafo_name}")
            tap = self.backend.symbols(f"tap_{trafo_name}")

            theta_jk = theta_j - theta_k
            cos_jk = self.backend.cos(theta_jk)
            sin_jk = self.backend.sin(theta_jk)

            # P_j = V_j^2 G / a^2 - V_j V_k (G cos + B sin)/a
            # Q_j = -V_j^2 B / a^2 + V_j V_k (B cos - G sin)/a
            # P_k = V_k^2 G - V_k V_j (G cos - B sin)/a
            # Q_k = -V_k^2 B + V_k V_j (B cos + G sin)/a
            P_to   = V_j**2 * G_t / tap**2 - V_j * V_k * (G_t * cos_jk + B_t * sin_jk) / tap
            Q_to   = -V_j**2 * B_t / tap**2 + V_j * V_k * (B_t * cos_jk - G_t * sin_jk) / tap
            P_from = V_k**2 * G_t - V_k * V_j * (G_t * cos_jk - B_t * sin_jk) / tap
            Q_from = -V_k**2 * B_t + V_k * V_j * (B_t * cos_jk + G_t * sin_jk) / tap

            p_to_pu, q_to_pu = self.backend.symbols(
                f"p_line_pu_{bus_j}_{bus_k}, q_line_pu_{bus_j}_{bus_k}")
            p_from_pu, q_from_pu = self.backend.symbols(
                f"p_line_pu_{bus_k}_{bus_j}, q_line_pu_{bus_k}_{bus_j}")

            self.dae['g'] += [p_to_pu   - P_to,
                              q_to_pu   - Q_to,
                              p_from_pu - P_from,
                              q_from_pu - Q_from]

            self.dae['y_ini'] += [p_to_pu, q_to_pu, p_from_pu, q_from_pu]
            self.dae['y_run'] += [p_to_pu, q_to_pu, p_from_pu, q_from_pu]

            U_base_j = self.buses[self.buses_list.index(bus_j)]['U_kV'] * 1000
            U_base_k = self.buses[self.buses_list.index(bus_k)]['U_kV'] * 1000
            I_base_j = sys['S_base'] / (np.sqrt(3) * U_base_j)
            I_base_k = sys['S_base'] / (np.sqrt(3) * U_base_k)

            self.dae['h_dict'].update({
                f'p_line_{bus_j}_{bus_k}': p_to_pu   * sys['S_base'],
                f'q_line_{bus_j}_{bus_k}': q_to_pu   * sys['S_base'],
                f'p_line_{bus_k}_{bus_j}': p_from_pu * sys['S_base'],
                f'q_line_{bus_k}_{bus_j}': q_from_pu * sys['S_base'],
            })

            I_j_k, I_k_j = self.backend.symbols(
                f"I_{bus_j}_{bus_k}, I_{bus_k}_{bus_j}")
            self.dae['g'] += [
                (p_to_pu**2   + q_to_pu**2)**0.5  / V_j * I_base_j - I_j_k,
                (p_from_pu**2 + q_from_pu**2)**0.5 / V_k * I_base_k - I_k_j,
            ]
            self.dae['y_ini'] += [I_j_k, I_k_j]
            self.dae['y_run'] += [I_j_k, I_k_j]
            self.dae['h_dict'].update({
                f'I_line_{bus_j}_{bus_k}': I_j_k,
                f'I_line_{bus_k}_{bus_j}': I_k_j,
            })

            self.dae['xy_0_dict'].update({
                str(p_to_pu):   0.1,
                str(p_from_pu): -0.1,
                str(I_j_k):     0.1,
                str(I_k_j):     -0.1,
            })

        for bus in self.buses:
            if 'monitor' in bus:
                if bus['monitor']:
                    U_base = bus['U_kV']*1000
                    V = self.backend.symbols(f"V_{bus['name']}") 
                    self.dae['h_dict'].update({f"U_{bus['name']}":V*U_base})
 


        # self.dae['f'] += []
        #self.dae['g'] += g_grid
        # self.dae['x'] += []
        # self.dae['y_ini'] += self.dae['y_ini']
        # self.dae['y_run'] += self.dae['y_ini']
        # self.dae['u_ini_dict'].update(self.dae['u_ini_dict'])
        # self.dae['u_run_dict'].update(self.dae['u_ini_dict'])
        # self.dae['h_dict'].update(self.dae['h_dict'])
        # self.dae['params_dict'].update(self.dae['params_dict'])

        self.A_incidence = self.A
        self.G_primitive = self.G_primitive
        self.B_primitive = self.B_primitive


        self.N_syn = 0
        self.N_gformers = 0
        
        self.generators_list = []
        self.generators_id_list = []
        
        # COI 
        omega_coi = self.backend.symbols("omega_coi")
        
        self.H_total = 0
        self.omega_coi_numerator = 0.0
        self.omega_coi_denominator = 0.0

        self.dae['xy_0_dict'].update({str(omega_coi):1.0})  

        self.buses = self.data['buses']
        self.buses_list = [bus['name'] for bus in self.buses]  
        

    def construct(self, name):
        
        self.contruct_grid()   
        
#        omega_coi = sym.Symbol("omega_coi", real=True)  
#        
#        self.dae['g'] += [ -omega_coi + self.omega_coi_numerator/self.omega_coi_denominator]
#        self.dae['y_ini'] += [ omega_coi]
#        self.dae['y_run'] += [ omega_coi]        
        


        #grid = BpsBuilder(data)
        if 'syns' in self.data:
            add_syns(self)
        if 'vscs' in self.data:
            add_vscs(self)
        if 'vsgs' in self.data:
            add_vsgs(self)
        if 'sources' in  self.data:
            add_sources(self)
        if 'wecs' in  self.data:
            add_wecs(self)
        if 'weccs' in self.data:
            add_weccs(self)
        if 'ppcs' in self.data:
            add_ppcs(self)
        if 'pvs' in  self.data:
            add_pvs(self)
        if 'loads' in  self.data:
            add_loads(self)
        if 'pods' in  self.data:
            add_pods(self)
        if 'banks' in  self.data:
            add_banks(self)
        if 'agc' in self.data:
            add_agc(self)

        add_miscellaneous(self)

        omega_coi = self.backend.symbols("omega_coi")

        self.dae['g'] += [-omega_coi + self.omega_coi_numerator / self.omega_coi_denominator]
        self.dae['y_ini'] += [omega_coi]
        self.dae['y_run'] += [omega_coi]

        # # secondary frequency control
        # xi_freq = sym.Symbol("xi_freq", real=True) 
        # p_agc = sym.Symbol("p_agc", real=True)  
        # K_p_agc = sym.Symbol("K_p_agc", real=True) 
        # K_i_agc = sym.Symbol("K_i_agc", real=True) 
        # K_xif  = sym.Symbol("K_xif", real=True)

        # epsilon_freq = 1-omega_coi
        # g_agc = [ -p_agc + K_p_agc*epsilon_freq + K_i_agc*xi_freq ]
        # y_agc = [  p_agc]
        # x_agc = [ xi_freq]
        # f_agc = [epsilon_freq - K_xif*xi_freq]

        # self.dae['g'] += g_agc
        # self.dae['y_ini'] += y_agc
        # self.dae['y_run'] += y_agc
        # self.dae['f'] += f_agc
        # self.dae['x'] += x_agc
        # self.dae['params_dict'].update({'K_p_agc':self.system['K_p_agc'],'K_i_agc':self.system['K_i_agc']})

        # if 'K_xif' in self.system:
        #     self.dae['params_dict'].update({'K_xif':self.system['K_xif']})
        # else:
        #     self.dae['params_dict'].update({'K_xif':0.0})

         
        with open('xy_0.json','w') as fobj:
            fobj.write(json.dumps(self.dae['xy_0_dict'],indent=4))

        with open(f'{name}_xy_0.json','w') as fobj:
            fobj.write(json.dumps(self.dae['xy_0_dict'],indent=4))

        self.sys_dict = {'name': name, 'uz_jacs': self.uz_jacs,
                         'params_dict': self.dae['params_dict'],
                         'f_list': self.dae['f'],
                         'g_list': self.dae['g'],
                         'x_list': self.dae['x'],
                         'y_ini_list': self.dae['y_ini'],
                         'y_run_list': self.dae['y_run'],
                         'u_run_dict': self.dae['u_run_dict'],
                         'u_ini_dict': self.dae['u_ini_dict'],
                         'h_dict': self.dae['h_dict'],
                         'xy_0_dict': self.dae['xy_0_dict']}
        if self.testing:
            self.sys_dict.update({'testing':True})


    def compile(self, API=False):

        bldr = db(self.sys_dict, verbose=self.verbose)
        bldr.build()    

    def compile_mkl(self, name):

        b = db(self.sys_dict, verbose=self.verbose)
        b.sparse = True
        b.mkl = True
        b.uz_jacs = False
        b.dict2system()
        b.functions()
        b.jacobians()
        b.cwrite()
        b.template()
        b.compile_mkl()  

    def build(self, name ='', API=False):
        if name == '':
            print('Error: name is not provided.')
        self.construct(name)    
        self.compile(API=False)  

    def build_mkl_win(self, name =''):
        if name == '':
            print('Error: name is not provided.')
        self.construct(name)    
        self.compile_mkl(name)

    def checker(self):
        
        if not 'syns' in self.data: self.data.update({'syns':[]})
        if not 'vscs' in self.data: self.data.update({'vscs':[]})
        if not 'vsgs' in self.data: self.data.update({'vsgs':[]})
        if not 'genapes' in self.data: self.data.update({'genapes':[]})
        if not 'sources' in self.data: self.data.update({'sources':[]})
                                                         
                                                         
        K_deltas_n = 0
        for item in self.data['syns']:
            if item['K_delta'] > 0.0:
                K_deltas_n += 1
        for item in self.data['vscs']:
            if item['K_delta'] > 0.0:
                K_deltas_n += 1
        for item in self.data['vsgs']:
            if item['K_delta'] > 0.0:
                K_deltas_n += 1
        for item in self.data['genapes']:
            if item['K_delta'] > 0.0:
                K_deltas_n += 1      
        for item in self.data['sources']:
            if item['type'] > 'vsource':
                K_deltas_n += 1   

        if  K_deltas_n == 0:
            print('One generator must have K_delta > 0.0')
        if  K_deltas_n > 1:
            print('Only one generator must have K_delta > 0.0')                                          
                                                      
        if len(self.data['genapes']) > 0:
            if self.data['system']['K_p_agc'] != 0.0:
                print('With a genape in the system K_p_agc must be set to 0.0')
            if self.data['system']['K_i_agc'] != 0.0:
                print('With a genape in the system K_i_agc must be set to 0.0')  
            if not self.data['system']['K_xif'] > 0.0:
                print('With a genape in the system K_xif must be set larger than 0.0')     


        for item in self.data['syns']:
            if 'gov' in item:
                if 'K_imw' in item['gov']:
                    if item['gov']['p_c'] > 0.0 and item['gov']['K_imw'] == 0.0:
                        print(f"Synchornous machine at bus {item['bus']} has p_c > 0, therefore K_imw should be larger than 0")                
            
        
        for item in self.data['syns']:
            if 'avr' in item:
                if 'sexs' in item['avr']:
                    if item['avr']['K_ai'] <= 0.0:
                        print(f"AVR of a synchornous machine at bus {item['bus']} must have constant K_ai larger than 0")                
            
               
            

if __name__ == "__main__":

    data = {
        "system":{"name":"k12p6","S_base":100e6, "K_p_agc":0.01,"K_i_agc":0.01},       
        "buses":[{"name":"1", "P_W":0.0,"Q_var":0.0,"U_kV":20.0},
                 {"name":"2", "P_W":0.0,"Q_var":0.0,"U_kV":20.0}],
        "lines":[{"bus_j":"1", "bus_k":"2", "X_pu":0.15,"R_pu":0.0, "S_mva":900.0}]
        }

    grid = BpsBuilder(data)

    from syns.syns import add_syns
    add_syns(grid,'hola')
    
    # data = {
    #     "sys":{"name":"k12p6","S_base":100e6, "K_p_agc":0.01,"K_i_agc":0.01},       
    #     "buses":[{"name":"1", "P_W":0.0,"Q_var":0.0,"U_kV":20.0},
    #              {"name":"2", "P_W":0.0,"Q_var":0.0,"U_kV":20.0},
    #              {"name":"3", "P_W":0.0,"Q_var":0.0,"U_kV":20.0},
    #              {"name":"4", "P_W":0.0,"Q_var":0.0,"U_kV":20.0},
    #              {"name":"5", "P_W":0.0,"Q_var":0.0,"U_kV":230.0},
    #              {"name":"6", "P_W":0.0,"Q_var":0.0,"U_kV":230.0},
    #              {"name":"7", "P_W":-967e6,"Q_var":100e6,"U_kV":230.0},
    #              {"name":"8", "P_W":0.0,"Q_var":0.0,"U_kV":230.0},
    #              {"name":"9", "P_W":-1767e6,"Q_var":250e6,"U_kV":230.0},
    #              {"name":"10","P_W":0.0,"Q_var":0.0,"U_kV":230.0},
    #              {"name":"11","P_W":0.0,"Q_var":0.0,"U_kV":230.0}      
    #             ],
    #     "lines":[{"bus_j":"1", "bus_k":"5", "X_pu":0.15,"R_pu":0.0, "S_mva":900.0},
    #              {"bus_j":"2", "bus_k":"6", "X_pu":0.15,"R_pu":0.0, "S_mva":900.0},
    #              {"bus_j":"3", "bus_k":"11","X_pu":0.15,"R_pu":0.0, "S_mva":900.0},
    #              {"bus_j":"4", "bus_k":"10","X_pu":0.15,"R_pu":0.0, "S_mva":900.0},
    #              {"bus_j":"5", "bus_k":"6", "X_km":0.529,"R_km":0.0529,"Bs_km":2.1e-6,"km":25},
    #              {"bus_j":"6", "bus_k":"7", "X_km":0.529,"R_km":0.0529,"Bs_km":2.1e-6,"km":10},
    #              {"bus_j":"7", "bus_k":"8", "X_km":0.529,"R_km":0.0529,"Bs_km":2.1e-6,"km":110},
    #              {"bus_j":"7", "bus_k":"8", "X_km":0.529,"R_km":0.0529,"Bs_km":2.1e-6,"km":110},
    #              {"bus_j":"8", "bus_k":"9", "X_km":0.529,"R_km":0.0529,"Bs_km":2.1e-6,"km":110},
    #              {"bus_j":"8", "bus_k":"9", "X_km":0.529,"R_km":0.0529,"Bs_km":2.1e-6,"km":110},
    #              {"bus_j":"9", "bus_k":"10","X_km":0.529,"R_km":0.0529,"Bs_km":2.1e-6,"km":10},
    #              {"bus_j":"10","bus_k":"11","X_km":0.529,"R_km":0.0529,"Bs_km":2.1e-6,"km":25}]
    #     }
    
    # bpu_obj = bpu(data_input=data)
    