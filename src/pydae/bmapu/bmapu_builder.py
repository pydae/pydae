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
from pydae.bmapu.syns.syns import add_syns
from pydae.bmapu.vscs.vscs import add_vscs
from pydae.bmapu.vsgs.vsgs import add_vsgs
from pydae.bmapu.wecs.wecs import add_wecs
from pydae.bmapu.pvs.pvs import add_pvs
from pydae.bmapu.loads.loads import add_loads
from pydae.bmapu.sources.sources import add_sources
from pydae.bmapu.miscellaneous.miscellaneous import add_miscellaneous
from pydae.bmapu.pods.pods import add_pods
import pydae.build_cffi as db
from pydae.build_v2 import builder

import requests

# todo:
    # S_base can't be modified becuase impedances in element base are computed
    # in S_base only in the build
class bmapu:
    '''
    

    Parameters
    ----------
    data_input : string or dict
        File path to the system data information or dictionary with the information.

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

    def __init__(self,data_input='',testing=False):
        
        if type(data_input) == str:
            if 'http' in data_input:
                url = data_input
                resp = requests.get(url)
                data = hjson.loads(resp.text)
            else:
                if os.path.splitext(data_input)[1] == '.json':
                    with open(data_input,'r') as fobj:
                        data = json.loads(fobj.read().replace("'",'"'))
                if os.path.splitext(data_input)[1] == '.hjson':
                    with open(data_input,'r') as fobj:
                        data = hjson.loads(fobj.read().replace("'",'"'))
        elif type(data_input) == dict:
            data = data_input
            
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

        self.x_grid = []
        self.f_grid = []
    
        self.params_grid = {'S_base':self.system['S_base']}
        self.S_base = sym.Symbol("S_base", real=True) 
        self.N_bus = len(self.buses)
        self.N_branch = 3*len(self.lines) + len(self.shunts) + 2*len(self.transformers)
 
        self.dae = {'f':[],'g':[],'x':[],'y_ini':[],'y_run':[],
                    'u_ini_dict':{},'u_run_dict':{},'params_dict':{},
                    'h_dict':{},'xy_0_dict':{}}

        self.uz_jacs = True     
        self.verbose = False   
        self.testing = testing
                
    def contruct_grid(self):
        
        N_branch = self.N_branch
        N_bus = self.N_bus
        sys = self.system
        
        S_base = sym.Symbol('S_base', real=True)
        
        xy_0_dict_grid = {}
        u_grid = {}
        h_grid = {}
        
        A = sym.zeros(N_branch,N_bus)
        G_primitive = sym.zeros(N_branch,N_branch)
        B_primitive = sym.zeros(N_branch,N_branch)
        buses_list = [bus['name'] for bus in self.buses]
        it = 0
        for line in self.lines:
    
            bus_j = line['bus_j']
            bus_k = line['bus_k']
    
            idx_j = buses_list.index(bus_j)
            idx_k = buses_list.index(bus_k)    
    
            A[it,idx_j] = 1
            A[it,idx_k] =-1   
            A[it+1,idx_j] = 1
            A[it+2,idx_k] = 1   
            
            line_name = f"{bus_j}_{bus_k}"
            g_jk = sym.Symbol(f"g_{line_name}", real=True) 
            b_jk = sym.Symbol(f"b_{line_name}", real=True) 
            bs_jk = sym.Symbol(f"bs_{line_name}", real=True) 
            G_primitive[it,it] = g_jk
            B_primitive[it,it] = b_jk
            B_primitive[it+1,it+1] = bs_jk/2
            B_primitive[it+2,it+2] = bs_jk/2

            
            if 'X_pu' in line:
                if 'S_mva' in line: S_line = 1e6*line['S_mva']
                R = line['R_pu']*sys['S_base']/S_line  # in pu of the system base
                X = line['X_pu']*sys['S_base']/S_line  # in pu of the system base
                G =  R/(R**2+X**2)
                B = -X/(R**2+X**2)
                self.params_grid.update({f"g_{line_name}":G})
                self.params_grid.update({f'b_{line_name}':B})
    
            if 'X' in line:
                bus_idx = buses_list.index(line['bus_j'])
                U_base = self.buses[bus_idx]['U_kV']*1000
                Z_base = U_base**2/sys['S_base']
                R = line['R']/Z_base  # in pu of the system base
                X = line['X']/Z_base  # in pu of the system base
                G =  R/(R**2+X**2)
                B = -X/(R**2+X**2)
                self.params_grid.update({f"g_{line_name}":G})
                self.params_grid.update({f'b_{line_name}':B})
    
            if 'X_km' in line:
                bus_idx = buses_list.index(line['bus_j'])
                U_base = self.buses[bus_idx]['U_kV']*1000
                Z_base = U_base**2/sys['S_base']
                R = line['R_km']*line['km']/Z_base  # in pu of the system base
                X = line['X_km']*line['km']/Z_base  # in pu of the system base
                G =  R/(R**2+X**2)
                B = -X/(R**2+X**2)
                self.params_grid.update({f"g_{line_name}":G})
                self.params_grid.update({f'b_{line_name}':B})        
    
            self.params_grid.update({f'bs_{line_name}':0.0})
            if 'Bs_pu' in line:
                if 'S_mva' in line: S_line = 1e6*line['S_mva']
                Bs = line['Bs_pu']*S_line/sys['S_base']  # in pu of the system base
                bs = Bs
                self.params_grid[f'bs_{line_name}'] = bs
     
            if 'Bs_km' in line:
                bus_idx = buses_list.index(line['bus_j'])
                U_base = self.buses[bus_idx]['U_kV']*1000
                Z_base = U_base**2/sys['S_base']
                Y_base = 1.0/Z_base
                Bs = line['Bs_km']*line['km']/Y_base # in pu of the system base
                bs = Bs 
                self.params_grid[f'bs_{line_name}'] = bs
                
            it += 3

        for trafo in self.transformers:
    
            bus_j = trafo['bus_j']
            bus_k = trafo['bus_k']
    
            idx_j = buses_list.index(bus_j)
            idx_k = buses_list.index(bus_k)    
    
            A[it,idx_j] = 1
            A[it+1,idx_k] = 1  
            
            trafo_name = f"{bus_j}_{bus_k}"
            g_jk = sym.Symbol(f"g_cc_{trafo_name}", real=True) 
            b_jk = sym.Symbol(f"b_cc_{trafo_name}", real=True) 
            tap = sym.Symbol(f"tap_{trafo_name}", real=True) 
            ang = sym.Symbol(f"ang_{trafo_name}", real=True) 
            a_s = tap*sym.cos(ang)
            b_s = tap*sym.sin(ang)

            Y_cc = g_jk + sym.I*b_jk

            
            Y_primitive =sym.Matrix([[ Y_cc/(a_s**2+b_s**2),-Y_cc/(a_s-sym.I*b_s)],
                                     [-Y_cc/(a_s+sym.I*b_s),                 Y_cc]])


            G_primitive[it:it+2,it:it+2] = sym.re(Y_primitive)
            B_primitive[it:it+2,it:it+2] = sym.im(Y_primitive)


            if 'X_pu' in trafo:
                if 'S_mva' in trafo: S_trafo = 1e6*trafo['S_mva']
                R = trafo['R_pu']*sys['S_base']/S_trafo  # in pu of the system base
                X = trafo['X_pu']*sys['S_base']/S_trafo  # in pu of the system base
                G =  R/(R**2+X**2)
                B = -X/(R**2+X**2)
                self.params_grid.update({f"g_cc_{trafo_name}":G})
                self.params_grid.update({f'b_cc_{trafo_name}':B})
                tap_m = 1.0
                if 'tap_m' in trafo:
                    tap_m = trafo['tap_m']
                self.params_grid.update({f'tap_{trafo_name}':tap_m})
                self.params_grid.update({f'ang_{trafo_name}':0.0})

    
            if 'X' in trafo:
                bus_idx = buses_list.index(trafo['bus_j'])
                U_base = self.buses[bus_idx]['U_kV']*1000
                Z_base = U_base**2/sys['S_base']
                R = trafo['R']/Z_base  # in pu of the system base
                X = trafo['X']/Z_base  # in pu of the system base
                G =  R/(R**2+X**2)
                B = -X/(R**2+X**2)
                self.params_grid.update({f"g_cc_{trafo_name}":G})
                self.params_grid.update({f'b_cc_{trafo_name}':B})
                self.params_grid.update({f'tap_{trafo_name}':1.0})
                self.params_grid.update({f'ang_{trafo_name}':0.0})
     
            it += 2


        for shunt in self.shunts:
    
            bus_j = shunt['bus']   
            idx_j = buses_list.index(bus_j)
    
            A[it,idx_j] = 1
             
            shunt_name = f"{bus_j}"
            g_j = sym.Symbol(f"g_shunt_{shunt_name}", real=True) 
            b_j = sym.Symbol(f"b_shunt_{shunt_name}", real=True) 
            G_primitive[it,it] = g_j
            B_primitive[it,it] = b_j

            
            if 'X_pu' in shunt:
                if 'S_mva' in shunt: S_line = 1e6*shunt['S_mva']
                R = shunt['R_pu']*sys['S_base']/S_line  # in pu of the system base
                X = shunt['X_pu']*sys['S_base']/S_line  # in pu of the system base
                G =  R/(R**2+X**2)
                B = -X/(R**2+X**2)
                self.params_grid.update({f"g_shunt_{shunt_name}":G})
                self.params_grid.update({f'b_shunt_{shunt_name}':B})
    
            if 'X' in shunt:
                U_base = self.buses[idx_j]['U_kV']*1000
                Z_base = U_base**2/sys['S_base']
                R = shunt['R']/Z_base  # in pu of the system base
                X = shunt['X']/Z_base  # in pu of the system base
                G =  R/(R**2+X**2)
                B = -X/(R**2+X**2)
                self.params_grid.update({f"g_shunt_{shunt_name}":G})
                self.params_grid.update({f'b_shunt_{shunt_name}':B})
                
            it += 1    
    
        G = A.T * G_primitive * A
        B = A.T * B_primitive * A    
        
        sin = sym.sin
        cos = sym.cos
        y_grid = []
        g = sym.zeros(2*N_bus,1)

        for j in range(N_bus):
            bus_j_name = buses_list[j]
            P_j = sym.Symbol(f"P_{bus_j_name}", real=True)
            Q_j = sym.Symbol(f"Q_{bus_j_name}", real=True)
            g[2*j]   = -P_j/S_base
            g[2*j+1] = -Q_j/S_base
            for k in range(N_bus): 
    
                bus_k_name = buses_list[k]
                V_j = sym.Symbol(f"V_{bus_j_name}", real=True) 
                V_k = sym.Symbol(f"V_{bus_k_name}", real=True) 
                theta_j = sym.Symbol(f"theta_{bus_j_name}", real=True) 
                theta_k = sym.Symbol(f"theta_{bus_k_name}", real=True) 
                g[2*j]   += V_j*V_k*(G[j,k]*cos(theta_j - theta_k) + B[j,k]*sin(theta_j - theta_k)) 
                g[2*j+1] += V_j*V_k*(G[j,k]*sin(theta_j - theta_k) - B[j,k]*cos(theta_j - theta_k))        
                h_grid.update({f"V_{bus_j_name}":V_j})
            bus = self.buses[j]
            bus_name = bus['name']
            if 'type' in bus:
                if bus['type'] == 'slack':
                    y_grid += [P_j]
                    y_grid += [Q_j]
                    u_grid.update({f"V_{bus_name}":1.0})
                    u_grid.update({f"theta_{bus_name}":0.0})  
            else:
                y_grid += [V_j]
                y_grid += [theta_j]        
                u_grid.update({f"P_{bus_name}":bus['P_W']})
                u_grid.update({f"Q_{bus_name}":bus['Q_var']})    
                xy_0_dict_grid.update({str(V_j):1.0,str(theta_j):0.0})
                
            self.params_grid.update({f'U_{bus_name}_n':bus['U_kV']*1000})
        g_grid = list(g)     
    
        if False:
            v_sym_list = []
            for bus in buses_list:
                V_m = sym.Symbol(f'V_{bus}',real=True)
                V_a = sym.Symbol(f'theta_{bus}',real=True)
                v_sym_list += [V_m*sym.exp(sym.I*V_a)]
    
            sym.Matrix(v_sym_list)
    
            I_lines = (G_primitive+1j*B_primitive) * A * sym.Matrix(v_sym_list)
    
            it = 0
            for line in self.lines:
                I_jk_r = sym.Symbol(f"I_{line['bus_j']}_{line['bus_k']}_r", real=True)
                I_jk_i = sym.Symbol(f"I_{line['bus_j']}_{line['bus_k']}_i", real=True)
                g_grid += [-I_jk_r + sym.re(I_lines[it])]
                g_grid += [-I_jk_i + sym.im(I_lines[it])]
                y_grid += [I_jk_r]
                y_grid += [I_jk_i]
                it += 1
        
        for line in self.lines:
    
            bus_j = line['bus_j']
            bus_k = line['bus_k']

            line_name = f"{bus_j}_{bus_k}"
    
            idx_j = buses_list.index(bus_j)
            idx_k = buses_list.index(bus_k)  

            V_j = sym.Symbol(f"V_{bus_j}", real=True) 
            V_k = sym.Symbol(f"V_{bus_k}", real=True) 
            theta_j = sym.Symbol(f"theta_{bus_j}", real=True) 
            theta_k = sym.Symbol(f"theta_{bus_k}", real=True)

            b_ij_p = 0.0
            if f'bs_{line_name}' in self.params_grid:
                b_ij_p = self.params_grid[f'bs_{line_name}']

            G_jk = G[idx_j,idx_k] 
            B_jk = B[idx_j,idx_k] 
            theta_jk = theta_j - theta_k
            P_line_to   = V_j*V_k*(G_jk*sym.cos(theta_jk) + B_jk*sym.sin(theta_jk)) - V_j**2*(G_jk) 
            Q_line_to   = V_j*V_k*(G_jk*sym.sin(theta_jk) - B_jk*sym.cos(theta_jk)) + V_j**2*(B_jk) 
            P_line_from = V_j*V_k*(G_jk*sym.cos(-theta_jk) + B_jk*sym.sin(-theta_jk)) - V_k**2*(G_jk) 
            Q_line_from = V_j*V_k*(G_jk*sym.sin(-theta_jk) - B_jk*sym.cos(-theta_jk)) + V_k**2*(B_jk) 

            if 'monitor' in line:
                if line['monitor']:
                    h_grid.update({f"p_line_{bus_j}_{bus_k}":P_line_to})
                    h_grid.update({f"q_line_{bus_j}_{bus_k}":Q_line_to}) 
                    h_grid.update({f"p_line_{bus_k}_{bus_j}":P_line_from})
                    h_grid.update({f"q_line_{bus_k}_{bus_j}":Q_line_from}) 

        self.dae['f'] += []
        self.dae['g'] += g_grid
        self.dae['x'] += []
        self.dae['y_ini'] += y_grid
        self.dae['y_run'] += y_grid
        self.dae['u_ini_dict'].update(u_grid)
        self.dae['u_run_dict'].update(u_grid)
        self.dae['h_dict'].update(h_grid)
        self.dae['params_dict'].update(self.params_grid)
        self.dae['xy_0_dict'].update(xy_0_dict_grid)
        
        self.A_incidence = A
        self.G_primitive = G_primitive
        self.B_primitive = B_primitive


        self.N_syn = 0
        self.N_gformers = 0
        
        self.generators_list = []
        self.generators_id_list = []
        
        # COI 
        omega_coi = sym.Symbol("omega_coi", real=True)  
        
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
        


        #grid = bmapu_builder.bmapu(data)
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
        if 'pvs' in  self.data:
            add_pvs(self)
        if 'loads' in  self.data:
            add_loads(self)
        if 'pods' in  self.data:
            add_pods(self)

        add_miscellaneous(self)

        #add_vsgs(grid)
        omega_coi = sym.Symbol("omega_coi", real=True)  

        self.dae['g'] += [ -omega_coi + self.omega_coi_numerator/self.omega_coi_denominator]
        self.dae['y_ini'] += [ omega_coi]
        self.dae['y_run'] += [ omega_coi]

        # secondary frequency control
        xi_freq = sym.Symbol("xi_freq", real=True) 
        p_agc = sym.Symbol("p_agc", real=True)  
        K_p_agc = sym.Symbol("K_p_agc", real=True) 
        K_i_agc = sym.Symbol("K_i_agc", real=True) 
        K_xif  = sym.Symbol("K_xif", real=True)

        epsilon_freq = 1-omega_coi
        g_agc = [ -p_agc + K_p_agc*epsilon_freq + K_i_agc*xi_freq ]
        y_agc = [  p_agc]
        x_agc = [ xi_freq]
        f_agc = [epsilon_freq - K_xif*xi_freq]

        self.dae['g'] += g_agc
        self.dae['y_ini'] += y_agc
        self.dae['y_run'] += y_agc
        self.dae['f'] += f_agc
        self.dae['x'] += x_agc
        self.dae['params_dict'].update({'K_p_agc':self.system['K_p_agc'],'K_i_agc':self.system['K_i_agc']})

        if 'K_xif' in self.system:
            self.dae['params_dict'].update({'K_xif':self.system['K_xif']})
        else:
            self.dae['params_dict'].update({'K_xif':0.0})

         
        with open('xy_0.json','w') as fobj:
            fobj.write(json.dumps(self.dae['xy_0_dict'],indent=4))

        with open(f'{name}_xy_0.json','w') as fobj:
            fobj.write(json.dumps(self.dae['xy_0_dict'],indent=4))

        self.sys_dict = {'name':name,'uz_jacs':self.uz_jacs,
                'params_dict':self.dae['params_dict'],
                'f_list':self.dae['f'],
                'g_list':self.dae['g'] ,
                'x_list':self.dae['x'],
                'y_ini_list':self.dae['y_ini'],
                'y_run_list':self.dae['y_run'],
                'u_run_dict':self.dae['u_run_dict'],
                'u_ini_dict':self.dae['u_ini_dict'],
                'h_dict':self.dae['h_dict']}
        if self.testing:
            self.sys_dict.update({'testing':True})


    def compile(self):

        bldr = db.builder(self.sys_dict,verbose=self.verbose);
        bldr.build()    

    def compile_mkl(self, name):

        b = builder(self.sys_dict,verbose=self.verbose)
        b.sparse = True
        b.mkl = True
        b.uz_jacs = False
        b.dict2system()
        b.functions()
        b.jacobians()
        b.cwrite()
        b.template()
        b.compile_mkl()  

    def build(self, name =''):
        if name == '':
            print('Error: name is not provided.')
        self.construct(name)    
        self.compile()  

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
        "sys":{"name":"k12p6","S_base":100e6, "K_p_agc":0.01,"K_i_agc":0.01},       
        "buses":[{"name":"1", "P_W":0.0,"Q_var":0.0,"U_kV":20.0},
                 {"name":"2", "P_W":0.0,"Q_var":0.0,"U_kV":20.0}],
        "lines":[{"bus_j":"1", "bus_k":"2", "X_pu":0.15,"R_pu":0.0, "S_mva":900.0}]
        }

    grid = bmapu(data)

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
    