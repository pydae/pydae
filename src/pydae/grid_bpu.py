# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 23:52:55 2021

@author: jmmau
"""

import numpy as np
import sympy as sym
import json

# todo:
    # S_base can't be modified becuase impedances in element base are computed
    # in S_base only in the build
class bpu:
    '''
    

    Parameters
    ----------
    data_input : string or dict
        File path to the system data information or dictionar y with the information.

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

    def __init__(self,data_input=''):
        
        if type(data_input) == str:
            with open(data_input,'r') as fobj:
                data = json.loads(fobj.read().replace("'",'"'))
        elif type(data_input) == dict:
            data = data_input
            
        self.data = data
    
        self.sys = data['sys']
        self.buses = data['buses']
        self.lines = data['lines']
        self.x_grid = []
        self.f_grid = []
    
        self.params_grid = {'S_base':self.sys['S_base']}
        self.S_base = sym.Symbol("S_base", real=True) 
        self.N_bus = len(self.buses)
        self.N_branch = len(self.lines)
 
        self.dae = {'f':[],'g':[],'x':[],'y_ini':[],'y_run':[],
                    'u_ini_dict':{},'u_run_dict':{},'params_dict':{},
                    'h_dict':{},'xy_0_dict':{}}
        
        self.contruct_grid()
        self.contruct_generators()
        
        with open('xy_0_2.json','w') as fobj:
            fobj.write(json.dumps(self.dae['xy_0_dict'],indent=4))
                
    def contruct_grid(self):
        
        N_branch = self.N_branch
        N_bus = self.N_bus
        sys = self.sys
        
        S_base = sym.Symbol('S_base', real=True)
        
        xy_0_dict_grid = {}
        
        A = sym.zeros(3*N_branch,N_bus)
        G_primitive = sym.zeros(3*N_branch,3*N_branch)
        B_primitive = sym.zeros(3*N_branch,3*N_branch)
        buses_list = [bus['name'] for bus in self.buses]
        it = 0
        for line in self.lines:
    
            bus_j = line['bus_j']
            bus_k = line['bus_k']
    
            idx_j = buses_list.index(bus_j)
            idx_k = buses_list.index(bus_k)    
    
            A[3*it,idx_j] = 1
            A[3*it,idx_k] =-1   
            A[3*it+1,idx_j] = 1
            A[3*it+2,idx_k] = 1   
            
            line_name = f"{bus_j}_{bus_k}"
            g_jk = sym.Symbol(f"g_{line_name}", real=True) 
            b_jk = sym.Symbol(f"b_{line_name}", real=True) 
            bs_jk = sym.Symbol(f"bs_{line_name}", real=True) 
            G_primitive[3*it,3*it] = g_jk
            B_primitive[3*it,3*it] = b_jk
            B_primitive[3*it+1,3*it+1] = bs_jk/2
            B_primitive[3*it+2,3*it+2] = bs_jk/2
            
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
                bs = -Bs/2.0
                self.params_grid[f'bs_{line_name}'] = bs
     
            if 'Bs_km' in line:
                bus_idx = buses_list.index(line['bus_j'])
                U_base = self.buses[bus_idx]['U_kV']*1000
                Z_base = U_base**2/sys['S_base']
                Y_base = 1.0/Z_base
                Bs = line['Bs_km']*line['km']/Y_base # in pu of the system base
                bs = Bs 
                self.params_grid[f'bs_{line_name}'] = bs
                
            it += 1
    
    
        G = A.T * G_primitive * A
        B = A.T * B_primitive * A    
        
        sin = sym.sin
        cos = sym.cos
        y_grid = []
        g = sym.zeros(2*N_bus,1)
        u_grid = {}
        h_grid = {}
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
        self.B_primitive = B_primitive
        
 
    def contruct_generators(self):
        
        self.N_syn = 0
        self.N_gformers = 0
        
        self.generators_list = []
        self.generators_id_list = []
        
        self.omega_coi_n = 0
        self.V_media_n = 0
        
        omega_coi = sym.Symbol("omega_coi", real=True)  
        V_media = sym.Symbol("V_media", real=True) 
        
        self.H_total = 0
        self.omega_coi_numerator = 0.0
        self.omega_coi_denominator = 0.0
    
        if 'syns' in self.data:    
            self.add_syns()
            
        if 'pq_sat' in self.data:
            self.add_pq_sat()
  
        if 'gformers_droop_z' in self.data:    
            gformer_droop_z_add()
            
        if 'vsgs' in self.data:
            self.add_vsgs()
            
        if 'genapes' in self.data:  
            for genape_data in self.data['genapes']:
                self.add_genape(genape_data) 
            
         
        self.dae['g'] += [ -omega_coi + self.omega_coi_numerator/self.omega_coi_denominator]
        self.dae['y_ini'] += [ omega_coi]
        self.dae['y_run'] += [ omega_coi]
        
        #     # omega COI
        #     H_total = 0.0
        #     for syn in data['syns']:
        #         bus_name = syn['bus']
        #         omega = sym.Symbol(f"omega_{bus_name}", real=True)    
        #         S_n = sym.Symbol(f"S_n_{bus_name}", real=True)            
        #         H = sym.Symbol(f"H_{bus_name}", real=True)            
    
        #         omega_coi_n += omega*H*S_n
        #         H_total += H*S_n
        #         N_syn += 1
    
        #     # secondary frequency control
        #     xi_freq = sym.Symbol("xi_freq", real=True) 
        #     for syn in data['syns']:
        #         bus_name = syn['bus']
        #         p_r = sym.Symbol(f"p_r_{bus_name}", real=True)            
        #         K_sec = sym.Symbol(f"K_sec_{bus_name}", real=True) 
                
        #         y_grid += [  p_r]
        #         g_grid += [ -p_r + K_sec*xi_freq/(N_syn+N_gformers)]
        #         params_grid.update({str(K_sec):syn['K_sec']})
        #     x_grid += [ xi_freq]
        #     f_grid += [ 1-omega_coi]  
            
    

            
        #     # omega COI
        #     for gformer_droop_z in data['gformers_droop_z']:
        #         bus_name = gformer_droop_z['bus']
        #         omega_v = sym.Symbol(f"omega_v_{bus_name}", real=True)            
        #         omega_coi_n += omega_v 
        #         N_gformers += 1
    
        #     # v media
        #     for gformer_droop_z in data['gformers_droop_z']:
        #         bus_name = gformer_droop_z['bus']
        #         V = sym.Symbol(f"V_{bus_name}", real=True)            
        #         V_media_n += V 
                
                
        # secondary frequency control
        xi_freq = sym.Symbol("xi_freq", real=True) 
        p_agc = sym.Symbol("p_agc", real=True)  
        K_p_agc = sym.Symbol("K_p_agc", real=True) 
        K_i_agc = sym.Symbol("K_i_agc", real=True) 
        
        epsilon_freq = 1-omega_coi
        g_agc = [ -p_agc + K_p_agc*epsilon_freq + K_i_agc*xi_freq ]
        y_agc = [  p_agc]
        x_agc = [ xi_freq]
        f_agc = [epsilon_freq]
        
        self.dae['g'] += g_agc
        self.dae['y_ini'] += y_agc
        self.dae['y_run'] += y_agc
        self.dae['f'] += f_agc
        self.dae['x'] += x_agc
        self.dae['params_dict'].update({'K_p_agc':self.sys['K_p_agc'],'K_i_agc':self.sys['K_i_agc']})
        
        #for gformer_droop_z in data['gformers_droop_z']:
        #         bus_name = gformer_droop_z['bus']
        #         p_r = sym.Symbol(f"p_r_{bus_name}", real=True)            
        #         K_sec = sym.Symbol(f"K_sec_{bus_name}", real=True) 
                
        #         y_grid += [  p_r]
        #         g_grid += [ -p_r + K_sec*xi_freq/(N_syn+N_gformers)]
        #         params_grid.update({str(K_sec):gformer_droop_z['K_sec']})
        #     x_grid += [ xi_freq]
        #     f_grid += [ 1-omega_coi]  
            
        #     # secondary voltage control
        #     xi_v = sym.Symbol("xi_v", real=True) 
        #     for gformer_droop_z in data['gformers_droop_z']:
        #         bus_name = gformer_droop_z['bus']
        #         q_r = sym.Symbol(f"q_r_{bus_name}", real=True)            
        #         K_sec_v = sym.Symbol(f"K_sec_v_{bus_name}", real=True) 
                
        #         y_grid += [  q_r]
        #         g_grid += [ -q_r + K_sec_v*xi_v/(N_syn+N_gformers)]
        #         params_grid.update({str(K_sec_v):gformer_droop_z['K_sec_v']})
        #     x_grid += [ xi_v]
        #     f_grid += [ 1-V_media]  
    
            

            
        #     # omega COI
        #     for gformer_z in data['gformers_z']:
        #         bus_name = gformer_z['bus']
        #         omega = sym.Symbol(f"omega_v_{bus_name}", real=True)
        #         omega_coi_n += omega 
        #         N_gformers += 1
            
        # if 'gformers' in data:    
        #     #syns_add(grid)
            
        #     # omega COI
        #     for gformer in data['gformers']:
        #         bus_name = gformer['bus']
        #         omega = sym.Symbol(f"omega_{bus_name}", real=True)
        #         V = sym.Symbol(f"V_{bus_name}", real=True)
        #         V_ref = sym.Symbol(f"V_{bus_name}_ref", real=True)
        #         theta_ref = sym.Symbol(f"theta_{bus_name}_ref", real=True)
        #         params_grid.update({f"omega_{bus_name}":1.0}) 
        #         params_grid.update({f"V_{bus_name}_ref":1.0})  
        #         params_grid.update({f"theta_{bus_name}_ref":0.0}) 
        #         y_grid += [ V]
        #         g_grid += [-omega_coi + omega_coi_n/(N_syn+N_gformers)]
        #         omega_coi_n += omega 
        #         N_gformers += 1
           
        # y_grid += [ omega_coi]
        # g_grid += [-omega_coi + omega_coi_n/H_total]
            
        # y_grid += [ V_media]
        # g_grid += [-V_media + V_media_n/(N_syn+N_gformers)]
        
        # return grid
    

    def add_syns(self):
        sin = sym.sin
        cos = sym.cos
        buses = self.data['buses']
        buses_list = [bus['name'] for bus in buses]
        
        for syn_data in self.data['syns']:
            
            bus_name = syn_data['bus']
            
            if 'name' in syn_data:
                name = syn_data['name']
            else:
                name = bus_name
                
            for gen_id in range(100):
                if name not in self.generators_id_list:
                    self.generators_id_list += [name]
                    break
                else:
                    name = name + f'_{gen_id}'
                    
            syn_data['name'] = name
                              
            idx_bus = buses_list.index(bus_name) # get the number of the bus where the syn is connected
            if not 'idx_powers' in buses[idx_bus]: buses[idx_bus].update({'idx_powers':0})
            buses[idx_bus]['idx_powers'] += 1
            
            # inputs
            V = sym.Symbol(f"V_{bus_name}", real=True)
            theta = sym.Symbol(f"theta_{bus_name}", real=True)
            p_m = sym.Symbol(f"p_m_{name}", real=True)
            v_f = sym.Symbol(f"v_f_{name}", real=True)  
            omega_coi = sym.Symbol("omega_coi", real=True)   
               
            # dynamic states
            delta = sym.Symbol(f"delta_{name}", real=True)
            omega = sym.Symbol(f"omega_{name}", real=True)
            e1q = sym.Symbol(f"e1q_{name}", real=True)
            e1d = sym.Symbol(f"e1d_{name}", real=True)

            # algebraic states
            i_d = sym.Symbol(f"i_d_{name}", real=True)
            i_q = sym.Symbol(f"i_q_{name}", real=True)            
            p_g = sym.Symbol(f"p_g_{name}", real=True)
            q_g = sym.Symbol(f"q_g_{name}", real=True)

            # parameters
            S_n = sym.Symbol(f"S_n_{name}", real=True)
            Omega_b = sym.Symbol(f"Omega_b_{name}", real=True)            
            H = sym.Symbol(f"H_{name}", real=True)
            T1d0 = sym.Symbol(f"T1d0_{name}", real=True)
            T1q0 = sym.Symbol(f"T1q0_{name}", real=True)
            X_d = sym.Symbol(f"X_d_{name}", real=True)
            X_q = sym.Symbol(f"X_q_{name}", real=True)
            X1d = sym.Symbol(f"X1d_{name}", real=True)
            X1q = sym.Symbol(f"X1q_{name}", real=True)
            D = sym.Symbol(f"D_{name}", real=True)
            R_a = sym.Symbol(f"R_a_{name}", real=True)
            K_delta = sym.Symbol(f"K_delta_{name}", real=True)
            params_list = ['S_n','Omega_b','H','T1d0','T1q0','X_d','X_q','X1d','X1q','D','R_a','K_delta','K_sec']
            
            # auxiliar
            v_d = V*sin(delta - theta) 
            v_q = V*cos(delta - theta) 
            p_e = i_d*(v_d + R_a*i_d) + i_q*(v_q + R_a*i_q)     
            omega_s = omega_coi
                        
            # dynamic equations            
            ddelta = Omega_b*(omega - omega_s) - K_delta*delta
            domega = 1/(2*H)*(p_m - p_e - D*(omega - omega_s))
            de1q = 1/T1d0*(-e1q - (X_d - X1d)*i_d + v_f)
            de1d = 1/T1q0*(-e1d + (X_q - X1q)*i_q)
    
            # algebraic equations   
            g_i_d  = v_q + R_a*i_q + X1d*i_d - e1q
            g_i_q  = v_d + R_a*i_d - X1q*i_q - e1d
            g_p_g  = i_d*v_d + i_q*v_q - p_g  
            g_q_g  = i_d*v_q - i_q*v_d - q_g 
            
            # dae 
            f_syn = [ddelta,domega,de1q,de1d]
            x_syn = [ delta, omega, e1q, e1d]
            g_syn = [g_i_d,g_i_q,g_p_g,g_q_g]
            y_syn = [  i_d,  i_q,  p_g,  q_g]
            
            self.H_total += H
            self.omega_coi_numerator += omega*H*S_n
            self.omega_coi_denominator += H*S_n
    
            self.dae['f'] += f_syn
            self.dae['x'] += x_syn
            self.dae['g'] += g_syn
            self.dae['y_ini'] += y_syn  
            self.dae['y_run'] += y_syn  
            
            if 'v_f' in syn_data:
                self.dae['u_ini_dict'].update({f'{v_f}':{syn_data['v_f']}})
                self.dae['u_run_dict'].update({f'{v_f}':{syn_data['v_f']}})
            else:
                self.dae['u_ini_dict'].update({f'{v_f}':1.0})
                self.dae['u_run_dict'].update({f'{v_f}':1.0})

            if 'p_m' in syn_data:
                self.dae['u_ini_dict'].update({f'{p_m}':{syn_data['p_m']}})
                self.dae['u_run_dict'].update({f'{p_m}':{syn_data['p_m']}})
            else:
                self.dae['u_ini_dict'].update({f'{p_m}':1.0})
                self.dae['u_run_dict'].update({f'{p_m}':1.0})
    
            self.dae['xy_0_dict'].update({str(omega):1.0})
            
            # grid power injection
            S_base = sym.Symbol('S_base', real = True)
            self.dae['g'][idx_bus*2]   += -p_g*S_n/S_base
            self.dae['g'][idx_bus*2+1] += -q_g*S_n/S_base
            
            # outputs
            self.dae['h_dict'].update({f"p_e_{name}":p_e})
            
            for item in params_list:       
                self.dae['params_dict'].update({f"{item}_{name}":syn_data[item]})
            
            if 'avr' in syn_data:
                add_avr(self.dae,syn_data)
                self.dae['u_ini_dict'].pop(str(v_f))
                self.dae['u_run_dict'].pop(str(v_f))
                self.dae['xy_0_dict'].update({str(v_f):1.5})
            if 'gov' in syn_data:
                add_gov(self.dae,syn_data)  
                self.dae['u_ini_dict'].pop(str(p_m))
                self.dae['u_run_dict'].pop(str(p_m))
                self.dae['xy_0_dict'].update({str(p_m):0.5})
            if 'pss' in syn_data:
                add_pss(self.dae,syn_data)          

    def add_pq_sat(self):
        
        for data in self.data['pq_sat']:
            
            bus_name = data['bus']
            name = bus_name
            
            buses = self.data['buses']
            buses_list = [bus['name'] for bus in buses]            
            idx_bus = buses_list.index(bus_name) # get the number of the bus where the syn is connected

                                         
            # inputs
            p_in = sym.Symbol(f"p_in_{name}", real=True)
            Dp_r = sym.Symbol(f"Dp_r_{name}", real=True)
            Dq_r = sym.Symbol(f"Dq_r_{name}", real=True)
 
               
            # dynamic states
            p_out = sym.Symbol(f"p_out_{name}", real=True)
            q_out = sym.Symbol(f"q_out_{name}", real=True)

            # algebraic states


            # parameters
            S_n = sym.Symbol(f"S_n_{name}", real=True)
            
            # auxiliar
            
                        
            # dynamic equations            

    
            # algebraic equations   
            p_out_sat = sym.Piecewise((0.0,p_in + Dp_r<0.0),(p_in,p_in + Dp_r>p_in),(p_in + Dp_r,True))         
            q_out_max = (S_n**2 - p_out_sat**2)**0.5
            q_out_sat = sym.Piecewise((-q_out_max,Dq_r<-q_out_max),(q_out_max,Dq_r>q_out_max),(Dq_r,True))     
            g_p_out = -p_out + p_out_sat
            g_q_out = -q_out + q_out_sat

            
            # dae 

    
            self.dae['f'] += []
            self.dae['x'] += []
            self.dae['g'] += [g_p_out,g_q_out]
            self.dae['y_ini'] += [p_out,q_out]  
            self.dae['y_run'] += [p_out,q_out]  
            
            self.dae['u_ini_dict'].update({f'{p_in}':data['p_in']})
            self.dae['u_run_dict'].update({f'{p_in}':data['p_in']})
            self.dae['u_ini_dict'].update({f'{Dp_r}':0.0})
            self.dae['u_run_dict'].update({f'{Dp_r}':0.0})
            self.dae['u_ini_dict'].update({f'{Dq_r}':0.0})
            self.dae['u_run_dict'].update({f'{Dq_r}':0.0})            
            
           
            # grid power injection
            S_base = sym.Symbol('S_base', real = True)
            self.dae['g'][idx_bus*2]   += -p_out/S_base
            self.dae['g'][idx_bus*2+1] += -q_out/S_base
            
            # outputs
            self.dae['h_dict'].update({f"{p_in}":p_in})
            self.dae['h_dict'].update({f"{Dp_r}":Dp_r})
            self.dae['h_dict'].update({f"{Dq_r}":Dp_r})
            
            # parameters            
            self.dae['params_dict'].update({f"{S_n}":data['S_n']})

                
    def add_vsgs(self):
        
        for vsg_data in self.data['vsgs']:
            if vsg_data['type'] == 'wo_vsg':
                self.add_vsg_wo(vsg_data)
            if vsg_data['type'] == 'pi_vsg':
                self.add_vsg_pi(vsg_data)
            if vsg_data['type'] == 'uvsg':
                self.add_uvsg(vsg_data)  
            if vsg_data['type'] == 'uvsg_high':
                self.add_uvsg_high(vsg_data) 
            if vsg_data['type'] == 'vsg_co':
                self.add_vsg_co(vsg_data) 
                
    def add_vsg_wo(self,vsg_data):
        sin = sym.sin
        cos = sym.cos
        buses = self.data['buses']
        buses_list = [bus['name'] for bus in buses]
                   
        bus_name = vsg_data['bus']
        
        if 'name' in vsg_data:
            name = vsg_data['name']
        else:
            name = bus_name
            
        for gen_id in range(100):
            if name not in self.generators_id_list:
                self.generators_id_list += [name]
                break
            else:
                name = name + f'_{gen_id}'
                
        vsg_data['name'] = name
                          
        idx_bus = buses_list.index(bus_name) # get the number of the bus where the syn is connected
        if not 'idx_powers' in buses[idx_bus]: buses[idx_bus].update({'idx_powers':0})
        buses[idx_bus]['idx_powers'] += 1
        
        # inputs
        V = sym.Symbol(f"V_{bus_name}", real=True)
        theta = sym.Symbol(f"theta_{bus_name}", real=True)
        p_m = sym.Symbol(f"p_m_{name}", real=True)
        v_ref = sym.Symbol(f"v_ref_{name}", real=True)  
        omega_coi = sym.Symbol("omega_coi", real=True)   
        p_c = sym.Symbol(f"p_c_{name}", real=True)
        p_agc = sym.Symbol("p_agc", real=True)
        omega_ref = sym.Symbol(f"omega_ref_{name}", real=True)
           
        # dynamic states
        delta = sym.Symbol(f"delta_{name}", real=True)
        omega = sym.Symbol(f"omega_{name}", real=True)
        e_qv = sym.Symbol(f"e_qv_{name}", real=True)
        x_w = sym.Symbol(f"x_w_{name}", real=True)

        # algebraic states
        i_d = sym.Symbol(f"i_d_{name}", real=True)
        i_q = sym.Symbol(f"i_q_{name}", real=True)            
        p_g = sym.Symbol(f"p_g_{name}", real=True)
        q_g = sym.Symbol(f"q_g_{name}", real=True)
        omega_w  = sym.Symbol(f"omega_w_{name}", real=True)

        # parameters
        S_n = sym.Symbol(f"S_n_{name}", real=True)
        Omega_b = sym.Symbol(f"Omega_b_{name}", real=True)            
        H = sym.Symbol(f"H_{name}", real=True)
        D = sym.Symbol(f"D_{name}", real=True) 
        T_v = sym.Symbol(f"T_v_{name}", real=True)
        X_v = sym.Symbol(f"X_v_{name}", real=True)
        R_v = sym.Symbol(f"R_v_{name}", real=True)
        K_delta = sym.Symbol(f"K_delta_{name}", real=True)
        T_w = sym.Symbol(f"T_w_{name}", real=True) 
        Droop = sym.Symbol(f"Droop_{name}", real=True) 
        K_sec = sym.Symbol(f"K_sec_{name}", real=True) 
        params_list = ['S_n','Omega_b','H','T_v','X_v','D','R_v','K_delta','K_sec','Droop','K_sec','T_w']
        
        # auxiliar
        v_d = V*sin(delta - theta) 
        v_q = V*cos(delta - theta) 
        p_e = i_d*(v_d + R_v*i_d) + i_q*(v_q + R_v*i_q)     
        omega_s = omega_coi
        e_dv = 0
        p_r = K_sec*p_agc
                    
        
        # dynamic equations            
        ddelta = Omega_b*(omega - omega_s) - K_delta*delta
        domega = 1/(2*H)*(p_m - p_e - D*omega_w)
        de_qv = 1/T_v*(v_ref - e_qv)
        dx_w = (omega - x_w)/T_w

        # algebraic equations   
        g_i_d  = -R_v*i_d - X_v*i_q - v_d + e_dv 
        g_i_q  = -R_v*i_q + X_v*i_d - v_q + e_qv
        g_p_g  = i_d*v_d + i_q*v_q - p_g  
        g_q_g  = i_d*v_q - i_q*v_d - q_g 
        g_p_m  = -p_m + p_c + p_r - 1/Droop*(omega - omega_ref)
        g_omega_w =  (omega - x_w) - omega_w 
        
        # dae 
        f_vsg = [ddelta,domega,de_qv,dx_w]
        x_vsg = [ delta, omega, e_qv, x_w]
        g_vsg = [g_i_d,g_i_q,g_p_g,g_q_g,g_p_m,g_omega_w]
        y_vsg = [  i_d,  i_q,  p_g,  q_g,  p_m,  omega_w]
        
        self.H_total += H
        self.omega_coi_numerator += omega*H*S_n
        self.omega_coi_denominator += H*S_n

        self.dae['f'] += f_vsg
        self.dae['x'] += x_vsg
        self.dae['g'] += g_vsg
        self.dae['y_ini'] += y_vsg  
        self.dae['y_run'] += y_vsg  
        
        if 'v_ref' in vsg_data:
            self.dae['u_ini_dict'].update({f'{str(v_ref)}':vsg_data['v_ref']})
            self.dae['u_run_dict'].update({f'{str(v_ref)}':vsg_data['v_ref']})
        else:
            self.dae['u_ini_dict'].update({f'{str(v_ref)}':1.0})
            self.dae['u_run_dict'].update({f'{str(v_ref)}':1.0})

        if 'p_m' in vsg_data:
            self.dae['u_ini_dict'].update({f'{str(p_m)}':vsg_data['p_m']})
            self.dae['u_run_dict'].update({f'{str(p_m)}':vsg_data['p_m']})
        else:
            self.dae['u_ini_dict'].update({f'{str(p_m)}':1.0})
            self.dae['u_run_dict'].update({f'{str(p_m)}':1.0})
            
        self.dae['u_ini_dict'].update({f'{str(p_c)}':vsg_data['p_c']})
        self.dae['u_run_dict'].update({f'{str(p_c)}':vsg_data['p_c']})

        self.dae['u_ini_dict'].update({f'{str(omega_ref)}':1.0})
        self.dae['u_run_dict'].update({f'{str(omega_ref)}':1.0})
        
    
        self.dae['xy_0_dict'].update({str(omega):1.0})
        
        # grid power injection
        S_base = sym.Symbol('S_base', real = True)
        self.dae['g'][idx_bus*2]   += -p_g*S_n/S_base
        self.dae['g'][idx_bus*2+1] += -q_g*S_n/S_base
        
        # outputs
        self.dae['h_dict'].update({f"p_e_{name}":p_e})
        
        for item in params_list:       
            self.dae['params_dict'].update({f"{item}_{name}":vsg_data[item]})

    def add_vsg_pi(self,vsg_data):
        sin = sym.sin
        cos = sym.cos
        buses = self.data['buses']
        buses_list = [bus['name'] for bus in buses]
                   
        bus_name = vsg_data['bus']
        
        if 'name' in vsg_data:
            name = vsg_data['name']
        else:
            name = bus_name
            
        for gen_id in range(100):
            if name not in self.generators_id_list:
                self.generators_id_list += [name]
                break
            else:
                name = name + f'_{gen_id}'
                
        vsg_data['name'] = name
                          
        idx_bus = buses_list.index(bus_name) # get the number of the bus where the syn is connected
        if not 'idx_powers' in buses[idx_bus]: buses[idx_bus].update({'idx_powers':0})
        buses[idx_bus]['idx_powers'] += 1
        
        # inputs
        V = sym.Symbol(f"V_{bus_name}", real=True)
        theta = sym.Symbol(f"theta_{bus_name}", real=True)
        p_m = sym.Symbol(f"p_m_{name}", real=True)
        v_ref = sym.Symbol(f"v_ref_{name}", real=True)  
        omega_coi = sym.Symbol("omega_coi", real=True)   
        p_c = sym.Symbol(f"p_c_{name}", real=True)
        p_agc = sym.Symbol("p_agc", real=True)
        omega_ref = sym.Symbol(f"omega_ref_{name}", real=True)
           
        # dynamic states
        delta = sym.Symbol(f"delta_{name}", real=True)
        e_qv = sym.Symbol(f"e_qv_{name}", real=True)
        xi_p = sym.Symbol(f"xi_p_{name}", real=True)

        # algebraic states
        omega = sym.Symbol(f"omega_{name}", real=True)
        i_d = sym.Symbol(f"i_d_{name}", real=True)
        i_q = sym.Symbol(f"i_q_{name}", real=True)            
        p_g = sym.Symbol(f"p_g_{name}", real=True)
        q_g = sym.Symbol(f"q_g_{name}", real=True)


        # parameters
        S_n = sym.Symbol(f"S_n_{name}", real=True)
        Omega_b = sym.Symbol(f"Omega_b_{name}", real=True)            
        K_p = sym.Symbol(f"K_p_{name}", real=True)        
        T_p = sym.Symbol(f"T_p_{name}", real=True)
        X_v = sym.Symbol(f"X_v_{name}", real=True)
        T_v = sym.Symbol(f"T_v_{name}", real=True)
        X_v = sym.Symbol(f"X_v_{name}", real=True)
        R_v = sym.Symbol(f"R_v_{name}", real=True)
        K_delta = sym.Symbol(f"K_delta_{name}", real=True)
        Droop = sym.Symbol(f"Droop_{name}", real=True) 
        K_sec = sym.Symbol(f"K_sec_{name}", real=True) 
        params_list = ['S_n','Omega_b','K_p','T_p','T_v','X_v','R_v','K_delta','K_sec','Droop','K_sec']
        
        # auxiliar
        v_d = V*sin(delta - theta) 
        v_q = V*cos(delta - theta) 
        p_e = i_d*(v_d + R_v*i_d) + i_q*(v_q + R_v*i_q)     
        omega_s = omega_coi
        e_dv = 0
        p_r = K_sec*p_agc
        epsilon_p = p_m - p_e
                            
        
        # dynamic equations            
        ddelta = Omega_b*(omega - omega_s) - K_delta*delta
        dxi_p = epsilon_p
        de_qv = 1/T_v*(v_ref - e_qv)

        # algebraic equations   
        g_omega = -omega + K_p*(epsilon_p + xi_p/T_p) + 1
        g_i_d  = -R_v*i_d - X_v*i_q - v_d + e_dv 
        g_i_q  = -R_v*i_q + X_v*i_d - v_q + e_qv
        g_p_g  = i_d*v_d + i_q*v_q - p_g  
        g_q_g  = i_d*v_q - i_q*v_d - q_g 
        g_p_m  = -p_m + p_c + p_r - 1/Droop*(omega - omega_ref)
        
        # dae 
        f_vsg = [ddelta,dxi_p,de_qv]
        x_vsg = [ delta, xi_p, e_qv]
        g_vsg = [g_omega,g_i_d,g_i_q,g_p_g,g_q_g,g_p_m]
        y_vsg = [  omega,  i_d,  i_q,  p_g,  q_g,  p_m]
        
        # T_p = K_p*2*H
        H = T_p/(2*K_p)
        self.H_total += H
        self.omega_coi_numerator += omega*H*S_n
        self.omega_coi_denominator += H*S_n

        self.dae['f'] += f_vsg
        self.dae['x'] += x_vsg
        self.dae['g'] += g_vsg
        self.dae['y_ini'] += y_vsg  
        self.dae['y_run'] += y_vsg  
        
        if 'v_ref' in vsg_data:
            self.dae['u_ini_dict'].update({f'{str(v_ref)}':vsg_data['v_ref']})
            self.dae['u_run_dict'].update({f'{str(v_ref)}':vsg_data['v_ref']})
        else:
            self.dae['u_ini_dict'].update({f'{str(v_ref)}':1.0})
            self.dae['u_run_dict'].update({f'{str(v_ref)}':1.0})

        if 'p_m' in vsg_data:
            self.dae['u_ini_dict'].update({f'{str(p_m)}':vsg_data['p_m']})
            self.dae['u_run_dict'].update({f'{str(p_m)}':vsg_data['p_m']})
        else:
            self.dae['u_ini_dict'].update({f'{str(p_m)}':1.0})
            self.dae['u_run_dict'].update({f'{str(p_m)}':1.0})
            
        self.dae['u_ini_dict'].update({f'{str(p_c)}':vsg_data['p_c']})
        self.dae['u_run_dict'].update({f'{str(p_c)}':vsg_data['p_c']})

        self.dae['u_ini_dict'].update({f'{str(omega_ref)}':1.0})
        self.dae['u_run_dict'].update({f'{str(omega_ref)}':1.0})
        
    
        self.dae['xy_0_dict'].update({str(omega):1.0})
        
        # grid power injection
        S_base = sym.Symbol('S_base', real = True)
        self.dae['g'][idx_bus*2]   += -p_g*S_n/S_base
        self.dae['g'][idx_bus*2+1] += -q_g*S_n/S_base
        
        # outputs
        self.dae['h_dict'].update({f"p_e_{name}":p_e})
        
        for item in params_list:       
            self.dae['params_dict'].update({f"{item}_{name}":vsg_data[item]})


    def add_vsg_co(self,vsg_data):
        sin = sym.sin
        cos = sym.cos
        buses = self.data['buses']
        buses_list = [bus['name'] for bus in buses]
                   
        bus_name = vsg_data['bus']
        
        if 'name' in vsg_data:
            name = vsg_data['name']
        else:
            name = bus_name
            
        for gen_id in range(100):
            if name not in self.generators_id_list:
                self.generators_id_list += [name]
                break
            else:
                name = name + f'_{gen_id}'
                
        vsg_data['name'] = name
                          
        idx_bus = buses_list.index(bus_name) # get the number of the bus where the syn is connected
        if not 'idx_powers' in buses[idx_bus]: buses[idx_bus].update({'idx_powers':0})
        buses[idx_bus]['idx_powers'] += 1
        
        # inputs
        V = sym.Symbol(f"V_{bus_name}", real=True)
        theta = sym.Symbol(f"theta_{bus_name}", real=True)
        p_m = sym.Symbol(f"p_m_{name}", real=True)
        v_ref = sym.Symbol(f"v_ref_{name}", real=True)  
        omega_coi = sym.Symbol("omega_coi", real=True)   
        p_c = sym.Symbol(f"p_c_{name}", real=True)
        p_agc = sym.Symbol("p_agc", real=True)
        omega_ref = sym.Symbol(f"omega_ref_{name}", real=True)
        q_ref = sym.Symbol(f"q_ref_{name}", real=True) 
           
        # dynamic states
        delta = sym.Symbol(f"delta_{name}", real=True)
        e_qv = sym.Symbol(f"e_qv_{name}", real=True)
        xi_p = sym.Symbol(f"xi_p_{name}", real=True)

        # algebraic states
        omega = sym.Symbol(f"omega_{name}", real=True)
        i_d = sym.Symbol(f"i_d_{name}", real=True)
        i_q = sym.Symbol(f"i_q_{name}", real=True)            
        p_g = sym.Symbol(f"p_g_{name}", real=True)
        q_g = sym.Symbol(f"q_g_{name}", real=True)


        # parameters
        S_n = sym.Symbol(f"S_n_{name}", real=True)
        Omega_b = sym.Symbol(f"Omega_b_{name}", real=True)            
        K_p = sym.Symbol(f"K_p_{name}", real=True)    
        K_q = sym.Symbol(f"K_q_{name}", real=True)   
        T_p = sym.Symbol(f"T_p_{name}", real=True)
        X_v = sym.Symbol(f"X_v_{name}", real=True)
        T_v = sym.Symbol(f"T_v_{name}", real=True)
        X_v = sym.Symbol(f"X_v_{name}", real=True)
        R_v = sym.Symbol(f"R_v_{name}", real=True)
        K_delta = sym.Symbol(f"K_delta_{name}", real=True)
        Droop = sym.Symbol(f"Droop_{name}", real=True) 
        K_sec = sym.Symbol(f"K_sec_{name}", real=True) 
        params_list = ['S_n','Omega_b','K_p','T_p','K_q','T_v','X_v','R_v','K_delta','K_sec','Droop','K_sec']
        
        # auxiliar
        v_d = V*sin(delta - theta) 
        v_q = V*cos(delta - theta) 
        p_e = i_d*(v_d + R_v*i_d) + i_q*(v_q + R_v*i_q)     
        omega_s = omega_coi
        e_dv = 0
        p_r = K_sec*p_agc
        epsilon_p = p_m - p_e
        epsilon_q = q_ref - q_g
        
                            
        
        # dynamic equations            
        ddelta = Omega_b*(omega - omega_s) - K_delta*delta
        dxi_p = epsilon_p
        de_qv = 1/T_v*(v_ref + K_q*epsilon_q - e_qv)

        # algebraic equations   
        g_omega = -omega + K_p*(epsilon_p + xi_p/T_p) + 1
        g_i_d  = -R_v*i_d + X_v*i_q - v_d + e_dv 
        g_i_q  = -R_v*i_q - X_v*i_d - v_q + e_qv
        g_p_g  = i_d*v_d + i_q*v_q - p_g  
        g_q_g  = i_d*v_q - i_q*v_d - q_g 
        g_p_m  = -p_m + p_c + p_r - 1/Droop*(omega - omega_ref)
        
        # dae 
        f_vsg = [ddelta,dxi_p,de_qv]
        x_vsg = [ delta, xi_p, e_qv]
        g_vsg = [g_omega,g_i_d,g_i_q,g_p_g,g_q_g,g_p_m]
        y_vsg = [  omega,  i_d,  i_q,  p_g,  q_g,  p_m]
        
        # T_p = K_p*2*H
        H = T_p/(2*K_p)
        self.H_total += H
        self.omega_coi_numerator += omega*H*S_n
        self.omega_coi_denominator += H*S_n

        self.dae['f'] += f_vsg
        self.dae['x'] += x_vsg
        self.dae['g'] += g_vsg
        self.dae['y_ini'] += y_vsg  
        self.dae['y_run'] += y_vsg  
        
        if 'v_ref' in vsg_data:
            self.dae['u_ini_dict'].update({f'{str(v_ref)}':vsg_data['v_ref']})
            self.dae['u_run_dict'].update({f'{str(v_ref)}':vsg_data['v_ref']})
        else:
            self.dae['u_ini_dict'].update({f'{str(v_ref)}':1.0})
            self.dae['u_run_dict'].update({f'{str(v_ref)}':1.0})

        if 'p_m' in vsg_data:
            self.dae['u_ini_dict'].update({f'{str(p_m)}':vsg_data['p_m']})
            self.dae['u_run_dict'].update({f'{str(p_m)}':vsg_data['p_m']})
        else:
            self.dae['u_ini_dict'].update({f'{str(p_m)}':1.0})
            self.dae['u_run_dict'].update({f'{str(p_m)}':1.0})
            
        self.dae['u_ini_dict'].update({f'{str(p_c)}':vsg_data['p_c']})
        self.dae['u_run_dict'].update({f'{str(p_c)}':vsg_data['p_c']})

        self.dae['u_ini_dict'].update({f'{str(omega_ref)}':1.0})
        self.dae['u_run_dict'].update({f'{str(omega_ref)}':1.0})

        self.dae['u_ini_dict'].update({f'{str(q_ref)}':0.0})
        self.dae['u_run_dict'].update({f'{str(q_ref)}':0.0})
        
    
        self.dae['xy_0_dict'].update({str(omega):1.0})
        
        # grid power injection
        S_base = sym.Symbol('S_base', real = True)
        self.dae['g'][idx_bus*2]   += -p_g*S_n/S_base
        self.dae['g'][idx_bus*2+1] += -q_g*S_n/S_base
        
        # outputs
        self.dae['h_dict'].update({f"p_e_{name}":p_e})
        
        for item in params_list:       
            self.dae['params_dict'].update({f"{item}_{name}":vsg_data[item]})
            
            
    def add_uvsg(self,vsg_data):
        sin = sym.sin
        cos = sym.cos
        buses = self.data['buses']
        buses_list = [bus['name'] for bus in buses]
                   
        bus_name = vsg_data['bus']
        
        if 'name' in vsg_data:
            name = vsg_data['name']
        else:
            name = bus_name
            
        for gen_id in range(100):
            if name not in self.generators_id_list:
                self.generators_id_list += [name]
                break
            else:
                name = name + f'_{gen_id}'
                
        vsg_data['name'] = name
                          
        idx_bus = buses_list.index(bus_name) # get the number of the bus where the syn is connected
        if not 'idx_powers' in buses[idx_bus]: buses[idx_bus].update({'idx_powers':0})
        buses[idx_bus]['idx_powers'] += 1
        
        # inputs
        V = sym.Symbol(f"V_{bus_name}", real=True)
        theta = sym.Symbol(f"theta_{bus_name}", real=True)
        omega_coi = sym.Symbol("omega_coi", real=True)   
        p_g = sym.Symbol(f"p_g_{name}", real=True)
        q_s_ref = sym.Symbol(f"q_s_ref_{name}", real=True)
        e_ref = sym.Symbol(f"e_ref_{name}", real=True) 
           
        # dynamic states
        delta = sym.Symbol(f"delta_{name}", real=True)
        e_qv = sym.Symbol(f"e_qv_{name}", real=True)
        xi_p = sym.Symbol(f"xi_p_{name}", real=True)
        xi_q = sym.Symbol(f"xi_q_{name}", real=True)
        e = sym.Symbol(f"e_{name}", real=True)

        # algebraic states
        omega = sym.Symbol(f"omega_{name}", real=True)
        i_d = sym.Symbol(f"i_d_{name}", real=True)
        i_q = sym.Symbol(f"i_q_{name}", real=True)            
        p_s = sym.Symbol(f"p_s_{name}", real=True)
        q_s = sym.Symbol(f"q_s_{name}", real=True)
        p_m = sym.Symbol(f"p_m_{name}", real=True)
        p_t = sym.Symbol(f"p_t_{name}", real=True)
        p_u = sym.Symbol(f"p_u_{name}", real=True)

        # parameters
        S_n = sym.Symbol(f"S_n_{name}", real=True)
        Omega_b = sym.Symbol(f"Omega_b_{name}", real=True)            
        K_p = sym.Symbol(f"K_p_{name}", real=True)        
        T_p = sym.Symbol(f"T_p_{name}", real=True)
        K_q = sym.Symbol(f"K_q_{name}", real=True)        
        T_q = sym.Symbol(f"T_q_{name}", real=True)
        X_v = sym.Symbol(f"X_v_{name}", real=True)
        R_v = sym.Symbol(f"R_v_{name}", real=True)
        R_s = sym.Symbol(f"R_s_{name}", real=True)
        T_e = sym.Symbol(f"T_e_{name}", real=True)
        K_e = sym.Symbol(f"K_e_{name}", real=True)
        params_list = ['S_n','Omega_b','K_p','T_p','K_q','T_q','X_v','R_v','R_s','T_e','K_e']
        
        # auxiliar
        v_d = V*sin(delta - theta) 
        v_q = V*cos(delta - theta) 
        omega_s = omega_coi
        e_dv = 0
        epsilon_p = p_m - p_s
        epsilon_q = q_s_ref - q_s
                            
        
        # dynamic equations            
        ddelta = Omega_b*(omega - omega_s)
        dxi_p = epsilon_p
        dxi_q = epsilon_q
        de = 1/T_e*(p_g-p_t)

        # algebraic equations   
        g_omega = -omega + K_p*(epsilon_p + xi_p/T_p)
        g_e_qv = -e_qv   + K_q*(epsilon_q + xi_q/T_q)
        g_i_d  = -R_v*i_d - X_v*i_q - v_d + e_dv 
        g_i_q  = -R_v*i_q + X_v*i_d - v_q + e_qv
        g_p_s  = i_d*v_d + i_q*v_q - p_s  
        g_q_s  = i_d*v_q - i_q*v_d - q_s 
        g_p_m  = -p_m + p_g + p_u
        g_p_t  = -p_t + i_d*(v_d + R_s*i_d) + i_q*(v_q + R_s*i_q)
        g_p_u  = -p_u + K_e*(e_ref - e)
        
        
        # dae 
        f_vsg = [ddelta,dxi_p,dxi_q,de]
        x_vsg = [ delta, xi_p, xi_q, e]
        g_vsg = [g_omega,g_e_qv,g_i_d,g_i_q,g_p_s,g_q_s,g_p_m,g_p_t,g_p_u]
        y_vsg = [  omega,  e_qv,  i_d,  i_q,  p_s,  q_s,  p_m,  p_t,  p_u]
        
        # T_p = K_p*2*H
        H = T_p/(2*K_p)
        self.H_total += H
        self.omega_coi_numerator += omega*H*S_n
        self.omega_coi_denominator += H*S_n

        self.dae['f'] += f_vsg
        self.dae['x'] += x_vsg
        self.dae['g'] += g_vsg
        self.dae['y_ini'] += y_vsg  
        self.dae['y_run'] += y_vsg  
        
        if 'q_s_ref' in vsg_data:
            self.dae['u_ini_dict'].update({f'{str(q_s_ref)}':vsg_data['q_s_ref']})
            self.dae['u_run_dict'].update({f'{str(q_s_ref)}':vsg_data['q_s_ref']})
        else:
            self.dae['u_ini_dict'].update({f'{str(q_s_ref)}':0.0})
            self.dae['u_run_dict'].update({f'{str(q_s_ref)}':0.0})
           
        self.dae['u_ini_dict'].update({f'{str(p_g)}':vsg_data['p_g']})
        self.dae['u_run_dict'].update({f'{str(p_g)}':vsg_data['p_g']})

        self.dae['u_ini_dict'].update({f'{str(e_ref)}':vsg_data['e_ref']})
        self.dae['u_run_dict'].update({f'{str(e_ref)}':vsg_data['e_ref']})
        
        self.dae['xy_0_dict'].update({str(omega):1.0})
        
        # grid power injection
        S_base = sym.Symbol('S_base', real = True)
        self.dae['g'][idx_bus*2]   += -p_s*S_n/S_base
        self.dae['g'][idx_bus*2+1] += -q_s*S_n/S_base
        
        # outputs
        #self.dae['h_dict'].update({f"p_e_{name}":p_e})
        
        for item in params_list:       
            self.dae['params_dict'].update({f"{item}_{name}":vsg_data[item]})

    def add_uvsg_high(self,vsg_data):
        sin = sym.sin
        cos = sym.cos
        buses = self.data['buses']
        buses_list = [bus['name'] for bus in buses]
                   
        bus_name = vsg_data['bus']
        
        if 'name' in vsg_data:
            name = vsg_data['name']
        else:
            name = bus_name
            
        for gen_id in range(100):
            if name not in self.generators_id_list:
                self.generators_id_list += [name]
                break
            else:
                name = name + f'_{gen_id}'
                
        vsg_data['name'] = name
                          
        idx_bus = buses_list.index(bus_name) # get the number of the bus where the syn is connected
        if not 'idx_powers' in buses[idx_bus]: buses[idx_bus].update({'idx_powers':0})
        buses[idx_bus]['idx_powers'] += 1
        
        # inputs
        V = sym.Symbol(f"V_{bus_name}", real=True)
        theta = sym.Symbol(f"theta_{bus_name}", real=True)
        omega_coi = sym.Symbol("omega_coi", real=True)   
        p_gin_0 = sym.Symbol(f"p_gin_0_{name}", real=True)
        p_g_ref = sym.Symbol(f"p_g_ref_{name}", real=True)
        q_s_ref = sym.Symbol(f"q_s_ref_{name}", real=True)
        v_u_ref = sym.Symbol(f"v_u_ref_{name}", real=True) 
        omega_ref = sym.Symbol(f"omega_ref_{name}", real=True) 
        ramp_p_gin = sym.Symbol(f"ramp_p_gin_{name}", real=True) 
                      
        # dynamic states
        delta = sym.Symbol(f"delta_{name}", real=True)
        e_qv = sym.Symbol(f"e_qv_{name}", real=True)
        xi_p = sym.Symbol(f"xi_p_{name}", real=True)
        xi_q = sym.Symbol(f"xi_q_{name}", real=True)
        e_u = sym.Symbol(f"e_u_{name}", real=True)
        p_ghr = sym.Symbol(f"p_ghr_{name}", real=True)
        k_cur = sym.Symbol(f"k_cur_{name}", real=True)
        inc_p_gin = sym.Symbol(f"inc_p_gin_{name}", real=True)
        xi_pll = sym.Symbol(f"xi_pll_{name}", real=True)
        theta_pll = sym.Symbol(f"theta_pll_{name}", real=True)
        
        # algebraic states
        omega = sym.Symbol(f"omega_{name}", real=True)
        i_d = sym.Symbol(f"i_d_{name}", real=True)
        i_q = sym.Symbol(f"i_q_{name}", real=True)            
        p_s = sym.Symbol(f"p_s_{name}", real=True)
        q_s = sym.Symbol(f"q_s_{name}", real=True)
        p_m = sym.Symbol(f"p_m_{name}", real=True)
        p_t = sym.Symbol(f"p_t_{name}", real=True)
        p_u = sym.Symbol(f"p_u_{name}", real=True)
        v_u = sym.Symbol(f"v_u_{name}", real=True)
        k_u = sym.Symbol(f"k_u_{name}", real=True)
        p_gou = sym.Symbol(f"p_gou_{name}", real=True)
        p_f = sym.Symbol(f"p_f_{name}", real=True)
        k_cur_sat = sym.Symbol(f"k_cur_sat_{name}", real=True)
        r_lim  = sym.Symbol(f"r_lim_{name}", real=True)
        omega_pll  = sym.Symbol(f"omega_pll_{name}", real=True)

        # parameters
        S_n = sym.Symbol(f"S_n_{name}", real=True)
        Omega_b = sym.Symbol(f"Omega_b_{name}", real=True)            
        K_p = sym.Symbol(f"K_p_{name}", real=True)        
        T_p = sym.Symbol(f"T_p_{name}", real=True)
        K_q = sym.Symbol(f"K_q_{name}", real=True)        
        T_q = sym.Symbol(f"T_q_{name}", real=True)
        X_v = sym.Symbol(f"X_v_{name}", real=True)
        R_v = sym.Symbol(f"R_v_{name}", real=True)
        R_s = sym.Symbol(f"R_s_{name}", real=True)
        C_u = sym.Symbol(f"C_u_{name}", real=True)
        K_u_0 = sym.Symbol(f"K_u_0_{name}", real=True)
        K_u_max = sym.Symbol(f"K_u_max_{name}", real=True)
        V_u_min = sym.Symbol(f"V_u_min_{name}", real=True)
        V_u_max = sym.Symbol(f"V_u_max_{name}", real=True)
        R_uc= sym.Symbol(f"R_uc_{name}", real=True)
        K_h = sym.Symbol(f"K_h_{name}", real=True)
        T_cur= sym.Symbol(f"T_cur_{name}", real=True)
        R_lim= sym.Symbol(f"R_lim_{name}", real=True)
        V_u_lt = sym.Symbol(f"V_u_lt_{name}", real=True)
        V_u_ht = sym.Symbol(f"V_u_ht_{name}", real=True)
        Droop = sym.Symbol(f"Droop_{name}", real=True)
        DB = sym.Symbol(f"DB_{name}", real=True)
        R_lim_max = sym.Symbol(f"R_lim_max_{name}", real=True)
        K_fpfr = sym.Symbol(f"K_fpfr_{name}", real=True)
        P_f_min = sym.Symbol(f"P_f_min_{name}", real=True)
        P_f_max = sym.Symbol(f"P_f_max_{name}", real=True)
        K_p_pll = sym.Symbol(f"K_p_pll_{name}", real=True)
        K_i_pll = sym.Symbol(f"K_i_pll_{name}", real=True)
        K_speed = sym.Symbol(f"K_speed_{name}", real=True)
       
        params_list = ['S_n','Omega_b','K_p','T_p','K_q','T_q',
                       'X_v','R_v','R_s','C_u','K_u_0',
                       'K_u_max','V_u_min','V_u_max','R_uc','K_h','R_lim',
                       'V_u_lt','V_u_ht','Droop','DB','T_cur'
                      ]
        
        # auxiliar
        v_d = V*sin(delta - theta) 
        v_q = V*cos(delta - theta) 
        omega_s = omega_coi
        e_dv = 0
        epsilon_p = p_m - p_s
        epsilon_q = q_s_ref - q_s
        epsilon_gh = K_h*(p_gou - p_ghr)           
        epsilon_gh_sat = sym.Piecewise((-r_lim,epsilon_gh<-r_lim),(r_lim,epsilon_gh>r_lim),(epsilon_gh,True))         
        i_u = (p_gou-p_t)*S_n/(v_u+0.1)
        p_l = p_t - p_s
        p_gin = p_gin_0 + inc_p_gin
        p_f_sat = sym.Piecewise((P_f_min,p_f<P_f_min),(P_f_max,p_f>P_f_max),(p_f,True))
        k_cur_ref = p_g_ref/p_gin + p_f_sat/p_gin
        soc = (e_u**2 - V_u_min**2)/(V_u_max**2 - V_u_min**2)
        p_fpfr = K_fpfr*(p_f_sat)
        v_Qh =  V*cos(theta) 
        v_Dh =  V*sin(theta) 
        v_dl = v_Dh*cos(theta_pll) - v_Qh*sin(theta_pll)
        omega_f = K_speed*omega_pll + (1-K_speed)*omega 
        #p_fpfr = K_fpfr*(p_f - ((p_g_ref + p_f)*k_cur-p_ghr))

        # dynamic equations            
        ddelta = Omega_b*(omega - omega_s)
        dxi_p = epsilon_p
        dxi_q = epsilon_q
        de_u = 1/C_u*(i_u)
        dp_ghr = epsilon_gh_sat
        dk_cur = 1/T_cur*(k_cur_ref - k_cur)
        dinc_p_gin = ramp_p_gin - 0.001*inc_p_gin
        dtheta_pll = K_p_pll*v_dl + K_i_pll*xi_pll - Omega_b*omega_s
        dxi_pll = v_dl


        
        # algebraic equations   
        g_omega = -omega + K_p*(epsilon_p + xi_p/T_p)
        g_e_qv = -e_qv   + K_q*(epsilon_q + xi_q/T_q)
        g_i_d  = -R_v*i_d + X_v*i_q - v_d + e_dv 
        g_i_q  = -R_v*i_q - X_v*i_d - v_q + e_qv
        g_p_s  = i_d*v_d + i_q*v_q - p_s  
        g_q_s  = i_d*v_q - i_q*v_d - q_s 
        g_p_m  = -p_m + p_ghr + p_u - p_l + p_fpfr
        g_p_t  = -p_t + i_d*(v_d + R_s*i_d) + i_q*(v_q + R_s*i_q)
        g_p_u  = -p_u - k_u*(v_u_ref**2 - v_u**2)/V_u_max**2
        g_v_u  = -v_u + e_u + R_uc*i_u   
        g_k_u  = -k_u + sym.Piecewise((K_u_max,v_u<V_u_min),
                                      ((K_u_max-K_u_0)*(v_u-V_u_lt)/(V_u_min-V_u_lt)+K_u_0,v_u<V_u_lt),
                                      ((K_u_max-K_u_0)*(v_u-V_u_ht)/(V_u_max-V_u_ht)+K_u_0,v_u>V_u_ht),
                                      (K_u_max,v_u>V_u_max),
                                      (K_u_0,True))
        g_k_cur_sat = -k_cur_sat +  sym.Piecewise((0.0001,k_cur<0.0001),(1,k_cur>1),(k_cur,True))
        g_p_gou = -p_gou + k_cur_sat*p_gin 
        g_p_f = -p_f - sym.Piecewise((1/Droop*(omega_f-(omega_ref-DB/2.0)),omega_f<(omega_ref-DB/2.0)),
                                     (1/Droop*(omega_f-(omega_ref+DB/2.0)),omega_f>(omega_ref+DB/2.0)),
                                     (0.0,True))
        g_r_lim = -r_lim + sym.Piecewise(((R_lim_max-R_lim)*(v_u-V_u_lt)/(V_u_min-V_u_lt)+R_lim,v_u<V_u_lt),
                                         ((R_lim_max-R_lim)*(v_u-V_u_ht)/(V_u_max-V_u_ht)+R_lim,v_u>V_u_ht),
                                         (R_lim,True)) + sym.Piecewise((R_lim_max,omega<(omega_ref-DB/2.0)),
                                         (R_lim_max,omega>(omega_ref+DB/2.0)),
                                         (0.0,True)) 
        g_omega_pll = -omega_pll + (K_p_pll*v_dl + K_i_pll*xi_pll)/Omega_b                                                                
                
        # dae 
        f_vsg = [ddelta,dxi_p,dxi_q,de_u,dp_ghr,dk_cur,dinc_p_gin,dtheta_pll,dxi_pll]
        x_vsg = [ delta, xi_p, xi_q, e_u, p_ghr, k_cur, inc_p_gin, theta_pll, xi_pll]
        g_vsg = [g_omega,g_e_qv,g_i_d,g_i_q,g_p_s,g_q_s,g_p_m,g_p_t,g_p_u,g_v_u,g_k_u,g_k_cur_sat,g_p_gou,g_p_f,g_r_lim,g_omega_pll]
        y_vsg = [  omega,  e_qv,  i_d,  i_q,  p_s,  q_s,  p_m,  p_t,  p_u,  v_u,  k_u,  k_cur_sat,  p_gou,  p_f,  r_lim,  omega_pll]
        
        # T_p = K_p*2*H
        H = T_p/(2*K_p)
        self.H_total += H
        self.omega_coi_numerator += omega*H*S_n
        self.omega_coi_denominator += H*S_n

        # DAE update
        self.dae['f'] += f_vsg
        self.dae['x'] += x_vsg
        self.dae['g'] += g_vsg
        self.dae['y_ini'] += y_vsg  
        self.dae['y_run'] += y_vsg  
        
        # inputs and their default values
        if 'q_s_ref' in vsg_data:
            self.dae['u_ini_dict'].update({f'{str(q_s_ref)}':vsg_data['q_s_ref']})
            self.dae['u_run_dict'].update({f'{str(q_s_ref)}':vsg_data['q_s_ref']})
        else:
            self.dae['u_ini_dict'].update({f'{str(q_s_ref)}':0.0})
            self.dae['u_run_dict'].update({f'{str(q_s_ref)}':0.0})
           
        self.dae['u_ini_dict'].update({f'{str(v_u_ref)}':vsg_data['v_u_ref']})
        self.dae['u_run_dict'].update({f'{str(v_u_ref)}':vsg_data['v_u_ref']})
        
        self.dae['u_ini_dict'].update({f'{str(omega_ref)}':1.0})
        self.dae['u_run_dict'].update({f'{str(omega_ref)}':1.0})

        self.dae['u_ini_dict'].update({f'{str(p_gin_0)}':vsg_data['p_gin_0']})
        self.dae['u_run_dict'].update({f'{str(p_gin_0)}':vsg_data['p_gin_0']})
        
        self.dae['u_ini_dict'].update({f'{str(p_g_ref)}':vsg_data['p_g_ref']})
        self.dae['u_run_dict'].update({f'{str(p_g_ref)}':vsg_data['p_g_ref']})

        self.dae['u_ini_dict'].update({f'{str(ramp_p_gin)}':0.0})
        self.dae['u_run_dict'].update({f'{str(ramp_p_gin)}':0.0})
        
        # initial guess (experimental)
        self.dae['xy_0_dict'].update({str(omega):1.0})
        self.dae['xy_0_dict'].update({str(v_u):vsg_data['v_u_ref']})
        self.dae['xy_0_dict'].update({str(e_u):vsg_data['v_u_ref']})
                                      
        # grid power injection
        S_base = sym.Symbol('S_base', real = True)
        self.dae['g'][idx_bus*2]   += -p_s*S_n/S_base
        self.dae['g'][idx_bus*2+1] += -q_s*S_n/S_base
        
        # outputs
        self.dae['h_dict'].update({f"p_gin_{name}":p_gin})
        self.dae['h_dict'].update({f"p_g_ref_{name}":p_g_ref})
        self.dae['h_dict'].update({f"p_l_{name}":p_l})
        self.dae['h_dict'].update({f"soc_{name}":soc})
        self.dae['h_dict'].update({f"p_fpfr_{name}":p_fpfr})
        self.dae['h_dict'].update({f"p_f_sat_{name}":p_f_sat})
        
        for item in params_list:       
            self.dae['params_dict'].update({f"{item}_{name}":vsg_data[item]})
        
        self.dae['params_dict'].update({f"{'R_lim_max'}_{name}":100.0})            
        self.dae['params_dict'].update({f"{'K_fpfr'}_{name}":0.0})
        self.dae['params_dict'].update({f"{'P_f_min'}_{name}":-1.0})
        self.dae['params_dict'].update({f"{'P_f_max'}_{name}": 1.0})
        self.dae['params_dict'].update({f"{'K_p_pll'}_{name}": 126.0})
        self.dae['params_dict'].update({f"{'K_i_pll'}_{name}": 3948.0})
        self.dae['params_dict'].update({f"{'K_speed'}_{name}": 1.0})

        
    def add_genape(self,vsg_data):
        sin = sym.sin
        cos = sym.cos
        buses = self.data['buses']
        buses_list = [bus['name'] for bus in buses]
                   
        bus_name = vsg_data['bus']
        
        if 'name' in vsg_data:
            name = vsg_data['name']
        else:
            name = bus_name
            
        for gen_id in range(100):
            if name not in self.generators_id_list:
                self.generators_id_list += [name]
                break
            else:
                name = name + f'_{gen_id}'
                
        vsg_data['name'] = name
                          
        idx_bus = buses_list.index(bus_name) # get the number of the bus where the syn is connected
        if not 'idx_powers' in buses[idx_bus]: buses[idx_bus].update({'idx_powers':0})
        buses[idx_bus]['idx_powers'] += 1
        
        # inputs
        V = sym.Symbol(f"V_{bus_name}", real=True)
        theta = sym.Symbol(f"theta_{bus_name}", real=True)
        omega_coi = sym.Symbol("omega_coi", real=True)   
        alpha = sym.Symbol(f"alpha_{name}", real=True)
        e_qv = sym.Symbol(f"e_qv_{name}", real=True)
        omega_ref = sym.Symbol(f"omega_ref_{name}", real=True) 
        
        # dynamic states
        delta = sym.Symbol(f"delta_{name}", real=True)
        omega = sym.Symbol(f"omega_{name}", real=True)

        # algebraic states
        Domega = sym.Symbol(f"Domega_{name}", real=True)
        i_d = sym.Symbol(f"i_d_{name}", real=True)
        i_q = sym.Symbol(f"i_q_{name}", real=True)            
        p_s = sym.Symbol(f"p_s_{name}", real=True)
        q_s = sym.Symbol(f"q_s_{name}", real=True)
        
        # parameters
        S_n = sym.Symbol(f"S_n_{name}", real=True)
        Omega_b = sym.Symbol(f"Omega_b_{name}", real=True)            
        X_v = sym.Symbol(f"X_v_{name}", real=True)
        R_v = sym.Symbol(f"R_v_{name}", real=True)
        K_delta = sym.Symbol(f"K_delta_{name}", real=True)
        K_alpha = sym.Symbol(f"K_alpha_{name}", real=True)
        
        params_list = ['S_n','Omega_b','X_v','R_v','K_delta']
        
        # auxiliar
        v_d = V*sin(delta - theta) 
        v_q = V*cos(delta - theta) 
        omega_s = omega_coi
        e_dv = 0                           
        
        # dynamic equations            
        ddelta = Omega_b*(omega - omega_s) - K_delta*delta
        dDomega = alpha - K_alpha*Domega

        # algebraic equations   
        g_omega = -omega + Domega + omega_ref
        g_i_d  = -R_v*i_d + X_v*i_q - v_d + e_dv 
        g_i_q  = -R_v*i_q - X_v*i_d - v_q + e_qv
        g_p_s  = i_d*v_d + i_q*v_q - p_s  
        g_q_s  = i_d*v_q - i_q*v_d - q_s 
        
        
        # dae 
        f_vsg = [ddelta,dDomega]
        x_vsg = [ delta, Domega]
        g_vsg = [g_omega,g_i_d,g_i_q,g_p_s,g_q_s]
        y_vsg = [  omega,  i_d,  i_q,  p_s,  q_s]
        
        H = 1e6
        self.H_total += H
        self.omega_coi_numerator += omega*H*S_n
        self.omega_coi_denominator += H*S_n

        self.dae['f'] += f_vsg
        self.dae['x'] += x_vsg
        self.dae['g'] += g_vsg
        self.dae['y_ini'] += y_vsg  
        self.dae['y_run'] += y_vsg  
        
          
        self.dae['u_ini_dict'].update({f'{str(alpha)}':0})
        self.dae['u_run_dict'].update({f'{str(alpha)}':0})

        self.dae['u_ini_dict'].update({f'{str(e_qv)}':1})
        self.dae['u_run_dict'].update({f'{str(e_qv)}':1})

        self.dae['u_ini_dict'].update({f'{str(omega_ref)}':1})
        self.dae['u_run_dict'].update({f'{str(omega_ref)}':1})
        
        self.dae['xy_0_dict'].update({str(omega):1.0})
        
        # grid power injection
        S_base = sym.Symbol('S_base', real = True)
        self.dae['g'][idx_bus*2]   += -p_s*S_n/S_base
        self.dae['g'][idx_bus*2+1] += -q_s*S_n/S_base
        
        # outputs
        self.dae['h_dict'].update({f"alpha_{name}":alpha})
        
        for item in params_list:       
            self.dae['params_dict'].update({f"{item}_{name}":vsg_data[item]})
        self.dae['params_dict'].update({f"{'K_alpha'}_{name}":1e-6})    
        

                
def add_avr(dae,syn_data):
    
    if syn_data['avr']['type'] == 'sexs':
        sexs(dae,syn_data)

def add_gov(dae,syn_data):
    
    if syn_data['gov']['type'] == 'tgov1':
        tgov1(dae,syn_data)

    if syn_data['gov']['type'] == 'agov1':
        agov1(dae,syn_data)

    if syn_data['gov']['type'] == 'hygov':
        hygov(dae,syn_data)

def add_pss(dae,syn_data):
    
    if syn_data['pss']['type'] == 'pss_kundur':
        pss_kundur(dae,syn_data)

def tgov1(syn_data):
    '''
    Governor TGOV1 like in PSS/E
    
    '''
    bus_name = syn_data['bus']
    gov_data = syn_data['gov']
    
    # inpunts
    omega = sym.Symbol(f"omega_{bus_name}", real=True)
    p_r = sym.Symbol(f"p_r_{bus_name}", real=True)
    p_c = sym.Symbol(f"p_c_{bus_name}", real=True) 

    # dynamic states
    x_gov_1 = sym.Symbol(f"x_gov_1_{bus_name}", real=True)
    x_gov_2 = sym.Symbol(f"x_gov_2_{bus_name}", real=True)  

    # algebraic states
    p_m = sym.Symbol(f"p_m_{bus_name}", real=True)  
    p_m_ref = sym.Symbol(f"p_m_ref_{bus_name}", real=True)  

    # parameters
    T_1 = sym.Symbol(f"T_gov_1_{bus_name}", real=True)  # 1
    T_2 = sym.Symbol(f"T_gov_2_{bus_name}", real=True)  # 2
    T_3 = sym.Symbol(f"T_gov_3_{bus_name}", real=True)  # 10

    Droop = sym.Symbol(f"Droop_{bus_name}", real=True)  # 0.05
    
    omega_ref = sym.Symbol(f"omega_ref_{bus_name}", real=True)

    # differential equations
    dx_gov_1 =   (p_m_ref - x_gov_1)/T_1
    dx_gov_2 =   (x_gov_1 - x_gov_2)/T_3

    g_p_m_ref  = -p_m_ref + p_c + p_r - 1/Droop*(omega - omega_ref)
    g_p_m = (x_gov_1 - x_gov_2)*T_2/T_3 + x_gov_2 - p_m

    
    grid['f'] += [dx_gov_1,dx_gov_2]
    grid['x'] += [ x_gov_1, x_gov_2]
    grid['g'] += [g_p_m_ref,g_p_m]
    grid['y'] += [  p_m_ref,  p_m]  
    grid['params'].update({str(Droop):gov_data['Droop']})
    grid['params'].update({str(T_1):gov_data['T_1']})
    grid['params'].update({str(T_2):gov_data['T_2']})
    grid['params'].update({str(T_3):gov_data['T_3']})
    grid['params'].update({str(omega_ref):1.0})

    grid['u'].update({str(p_c):gov_data['p_c']})

def agov1(grid,syn_data,bus_i = ''):

    gov_data = syn_data['gov']
    name = syn_data['name']
    
    # inpunts
    omega = sym.Symbol(f"omega_{name}", real=True)
    p_c = sym.Symbol(f"p_c_{name}", real=True) 
    p_g = sym.Symbol(f"p_g_{name}", real=True)
    p_agc = sym.Symbol("p_agc", real=True)
    p_r = sym.Symbol(f"p_r_{name}", real=True)
    
    # dynamic states
    x_gov_1 = sym.Symbol(f"x_gov_1_{name}", real=True)
    x_gov_2 = sym.Symbol(f"x_gov_2_{name}", real=True)  
    xi_imw  = sym.Symbol(f"xi_imw_{name}", real=True)  

    # algebraic states
    p_m = sym.Symbol(f"p_m_{name}", real=True)  
    p_m_ref = sym.Symbol(f"p_m_ref_{name}", real=True)  

    # parameters
    T_1 = sym.Symbol(f"T_gov_1_{name}", real=True)  # 1
    T_2 = sym.Symbol(f"T_gov_2_{name}", real=True)  # 2
    T_3 = sym.Symbol(f"T_gov_3_{name}", real=True)  # 10
    Droop = sym.Symbol(f"Droop_{name}", real=True)  # 0.05
    K_imw = sym.Symbol(f"K_imw_{name}", real=True)  # 0.0
    K_sec = sym.Symbol(f"K_sec_{name}", real=True)  # 0.0    
    omega_ref = sym.Symbol(f"omega_ref_{name}", real=True)
    
    # auxiliar

    # differential equations
    dx_gov_1 =   (p_m_ref - x_gov_1)/T_1
    dx_gov_2 =   (x_gov_1 - x_gov_2)/T_3
    dxi_imw =   K_imw*(p_c - p_g) - 1e-6*xi_imw

    g_p_m_ref  = -p_m_ref + xi_imw + p_r + K_sec*p_agc - 1/Droop*(omega - omega_ref) 
    g_p_m = (x_gov_1 - x_gov_2)*T_2/T_3 + x_gov_2 - p_m

    
    grid['f'] += [dx_gov_1,dx_gov_2,dxi_imw]
    grid['x'] += [ x_gov_1, x_gov_2, xi_imw]
    grid['g'] += [g_p_m_ref,g_p_m]
    grid['y_ini'] += [  p_m_ref,  p_m]  
    grid['y_run'] += [  p_m_ref,  p_m] 
    grid['params_dict'].update({str(Droop):gov_data['Droop']})
    grid['params_dict'].update({str(T_1):gov_data['T_1']})
    grid['params_dict'].update({str(T_2):gov_data['T_2']})
    grid['params_dict'].update({str(T_3):gov_data['T_3']})
    grid['params_dict'].update({str(K_imw):gov_data['K_imw']})
    grid['params_dict'].update({str(K_sec):syn_data['K_sec']})
    grid['params_dict'].update({str(omega_ref):1.0})

    grid['u_ini_dict'].update({str(p_c):gov_data['p_c']})
    grid['u_run_dict'].update({str(p_c):gov_data['p_c']})
    grid['u_ini_dict'].update({str(p_r):0.0})
    grid['u_run_dict'].update({str(p_r):0.0})
    
def type1(grid,syn_data,bus_i = ''):
    bus_name = syn_data['bus']
    gov_data = syn_data['gov']
    
    # inpunts
    omega = sym.Symbol(f"omega_{bus_name}", real=True)
    p_r = sym.Symbol(f"p_r_{bus_name}", real=True)
    p_c = sym.Symbol(f"p_c_{bus_name}", real=True) 
    p_g = sym.Symbol(f"p_g_{bus_name}_{bus_i}", real=True)

    # dynamic states
    x_gov_1 = sym.Symbol(f"x_gov_1_{bus_name}", real=True)
    x_gov_2 = sym.Symbol(f"x_gov_2_{bus_name}", real=True)  
    x_gov_2 = sym.Symbol(f"x_gov_2_{bus_name}", real=True)  

    xi_imw  = sym.Symbol(f"xi_imw_{bus_name}", real=True)  

    # algebraic states
    p_m = sym.Symbol(f"p_m_{bus_name}", real=True)  
    p_m_ref = sym.Symbol(f"p_m_ref_{bus_name}", real=True)  

    # parameters
    T_1 = sym.Symbol(f"T_gov_1_{bus_name}", real=True)  # 1
    T_2 = sym.Symbol(f"T_gov_2_{bus_name}", real=True)  # 2
    T_3 = sym.Symbol(f"T_gov_3_{bus_name}", real=True)  # 10

    Droop = sym.Symbol(f"Droop_{bus_name}", real=True)  # 0.05
    K_imw = sym.Symbol(f"K_imw_{bus_name}", real=True)  # 0.0

    omega_ref = sym.Symbol(f"omega_ref_{bus_name}", real=True)

    # differential equations
    dx_gov_1 =   (p_m_ref - x_gov_1)/T_1
    dx_gov_2 =   (x_gov_1 - x_gov_2)/T_3
    dxi_imw =   K_imw*(p_c - p_g) - 1e-6*xi_imw

    g_p_m_ref  = -p_m_ref + p_c + xi_imw + p_r - 1/Droop*(omega - omega_ref) 
    g_p_m = (x_gov_1 - x_gov_2)*T_2/T_3 + x_gov_2 - p_m

    
    grid['f'] += [dx_gov_1,dx_gov_2,dxi_imw]
    grid['x'] += [ x_gov_1, x_gov_2, xi_imw]
    grid['g'] += [g_p_m_ref,g_p_m]
    grid['y'] += [  p_m_ref,  p_m]  
    grid['params'].update({str(Droop):gov_data['Droop']})
    grid['params'].update({str(T_1):gov_data['T_1']})
    grid['params'].update({str(T_2):gov_data['T_2']})
    grid['params'].update({str(T_3):gov_data['T_3']})
    grid['params'].update({str(K_imw):gov_data['K_imw']})
    grid['params'].update({str(omega_ref):1.0})

def hygov(grid,syn_data):
    bus_name = syn_data['bus']
    gov_data = syn_data['gov']
    
    p_r = sym.Symbol(f"p_r_{bus_name}", real=True)
    omega_ref = sym.Symbol(f"omega_ref_{bus_name}", real=True)
    omega = sym.Symbol(f"omega_{bus_name}", real=True)

    # dynamic states:
    xi_omega = sym.Symbol(f"xi_omega_{bus_name}", real=True)
    servo = sym.Symbol(f"servo_{bus_name}", real=True)
    pos = sym.Symbol(f"pos_{bus_name}", real=True)
    flow = sym.Symbol(f"flow_{bus_name}", real=True)
    # algebraic states:    
    servo_u = sym.Symbol(f"servo_u_{bus_name}", real=True)
    gate = sym.Symbol(f"gate_{bus_name}", real=True)
    head = sym.Symbol(f"head_{bus_name}", real=True)
    p_m = sym.Symbol(f"p_m_{bus_name}", real=True)
    # parameters:
    Droop = sym.Symbol(f"Droop_{bus_name}", real=True) 
    K_p_gov = sym.Symbol(f"K_p_gov_{bus_name}", real=True)
    K_i_gov = sym.Symbol(f"K_i_gov_{bus_name}", real=True) 
    K_servo = sym.Symbol(f"K_servo_{bus_name}", real=True)  
    T_servo = sym.Symbol(f"T_servo_{bus_name}", real=True) 
    V_gate_max = sym.Symbol(f"V_gate_max_{bus_name}", real=True)
    Gate_max = sym.Symbol(f"Gate_max_{bus_name}", real=True)
    T_w = sym.Symbol(f"T_w_{bus_name}", real=True) 
    Flow_nl = sym.Symbol(f"Flow_nl_{bus_name}", real=True) 
    A_t = sym.Symbol(f"A_t_{bus_name}", real=True)     
    
    epsilon_omega = omega_ref - omega #- Droop*gate + p_r
    dxi_omega = epsilon_omega
    g_servo_u = -servo_u + K_p_gov*epsilon_omega + K_i_gov*xi_omega
    dservo = (K_servo*(servo_u - gate) - servo)/T_servo
    dpos = servo #sym.Piecewise((V_gate_max,servo>V_gate_max),(-V_gate_max,servo<-V_gate_max),(servo,True))
    g_gate = -gate + servo_u # + pos #sym.Piecewise((Gate_max,pos>Gate_max),(1e-6,pos<1e-6),(pos,True))
    dflow =   (gate - flow)/T_w
    g_head = -(gate - flow)/T_w + flow - head  
    g_p_m = -p_m + A_t*head




    grid['f'] += [dxi_omega, dflow] #[dxi_omega, dservo, dpos, dflow]
    grid['x'] += [ xi_omega,  flow]
    grid['g'] += [g_servo_u, g_gate, g_head, g_p_m] # [, g_gate, g_head, g_p_m]
    grid['y'] += [  servo_u,   gate,   head, p_m] # [  servo_u,   gate,   head,   p_m]
    grid['params'].update({str(Droop):gov_data['Droop'],str(K_p_gov):gov_data['K_p_gov'],str(K_i_gov):gov_data['K_i_gov']})
    grid['params'].update({str(K_servo):gov_data['K_servo'],str(T_servo):gov_data['T_servo']})
    grid['params'].update({str(V_gate_max):gov_data['V_gate_max'],str(Gate_max):gov_data['Gate_max']})
    grid['params'].update({str(T_w):gov_data['T_w'],str(Flow_nl):gov_data['Flow_nl']})
    grid['params'].update({str(A_t):gov_data['A_t']})
    grid['u'].update({str(omega_ref):gov_data['omega_ref']})

def sexs(dae,syn_data):
    name = syn_data['name']
    bus_name = syn_data['bus']
    avr_data = syn_data['avr']
    
    v_t = sym.Symbol(f"V_{name}", real=True)   
    v_c = sym.Symbol(f"v_c_{name}", real=True)  
    xi_v  = sym.Symbol(f"xi_v_{name}", real=True)
    v_f = sym.Symbol(f"v_f_{name}", real=True)  
    T_r = sym.Symbol(f"T_r_{name}", real=True) 
    K_a = sym.Symbol(f"K_a_{name}", real=True)
    K_ai = sym.Symbol(f"K_ai_{name}", real=True)
    V_min = sym.Symbol(f"V_min_{name}", real=True)
    V_max = sym.Symbol(f"V_max_{name}", real=True)
    K_aw = sym.Symbol(f"K_aw_{name}", real=True)   
    
    v_ref = sym.Symbol(f"v_ref_{name}", real=True) 
    v_pss = sym.Symbol(f"v_pss_{name}", real=True) 
    
    epsilon_v = v_ref - v_c + v_pss
    v_f_nosat = K_a*epsilon_v + K_ai*xi_v
    
    dv_c =   (v_t - v_c)/T_r
    dxi_v =   epsilon_v  - K_aw*(v_f_nosat - v_f) 
    
    g_v_f  =   sym.Piecewise((V_min, v_f_nosat<V_min),(V_max,v_f_nosat>V_max),(v_f_nosat,True)) - v_f 
  #  g_v_f  =   v_f_nosat - v_f 
  
  
    
    dae['f'] += [dv_c,dxi_v]
    dae['x'] += [ v_c, xi_v]
    dae['g'] += [g_v_f]
    dae['y_ini'] += [v_f] 
    dae['y_run'] += [v_f]  
    dae['params_dict'].update({str(K_a):avr_data['K_a']})
    dae['params_dict'].update({str(K_ai):avr_data['K_ai']})
    dae['params_dict'].update({str(T_r):avr_data['T_r']})  
    dae['params_dict'].update({str(V_min):avr_data['V_min']})  
    dae['params_dict'].update({str(V_max):avr_data['V_max']})  
    dae['params_dict'].update({str(K_aw):avr_data['K_aw']}) 
    dae['u_ini_dict'].update({str(v_ref):avr_data['v_ref']})
    dae['u_ini_dict'].update({str(v_pss):avr_data['v_pss']})
    dae['u_run_dict'].update({str(v_ref):avr_data['v_ref']})
    dae['u_run_dict'].update({str(v_pss):avr_data['v_pss']})
    dae['xy_0_dict'].update({str(xi_v):1})
    
def pss_kundur(grid,syn_data):
    bus_name = syn_data['bus']
    pss_data = syn_data['pss']
    
    omega = sym.Symbol(f"omega_{bus_name}", real=True)   
       
    x_wo  = sym.Symbol(f"x_wo_{bus_name}", real=True)
    x_lead  = sym.Symbol(f"x_lead_{bus_name}", real=True)

    z_wo  = sym.Symbol(f"z_wo_{bus_name}", real=True)
    x_lead  = sym.Symbol(f"x_lead_{bus_name}", real=True)
    
    T_wo = sym.Symbol(f"T_wo_{bus_name}", real=True)  
    T_1 = sym.Symbol(f"T_1_{bus_name}", real=True) 
    T_2 = sym.Symbol(f"T_2_{bus_name}", real=True)
    K_stab = sym.Symbol(f"K_stab_{bus_name}", real=True)
    V_lim = sym.Symbol(f"V_lim_{bus_name}", real=True)
    v_pss = sym.Symbol(f"v_pss_{bus_name}", real=True) 
    
    
    u_wo = omega - 1.0
    v_pss_nosat = K_stab*((z_wo - x_lead)*T_1/T_2 + x_lead)
    
    dx_wo =   (u_wo - x_wo)/T_wo  # washout state
    dx_lead =  (z_wo - x_lead)/T_2      # lead compensator state
    
    g_z_wo =  (u_wo - x_wo) - z_wo  
    g_v_pss = -v_pss + sym.Piecewise((-V_lim,v_pss_nosat<-V_lim),(V_lim,v_pss_nosat>V_lim),(v_pss_nosat,True))  
    
    
    grid['f'] += [dx_wo,dx_lead]
    grid['x'] += [ x_wo, x_lead]
    grid['g'] += [g_z_wo,g_v_pss]
    grid['y_ini'] += [  z_wo, v_pss]  
    grid['y_run'] += [  z_wo, v_pss] 
    grid['params_dict'].update({str(T_wo):pss_data['T_wo']})
    grid['params_dict'].update({str(T_1):pss_data['T_1']})
    grid['params_dict'].update({str(T_2):pss_data['T_2']})
    grid['params_dict'].update({str(K_stab):pss_data['K_stab']})
    grid['params_dict'].update({str(V_lim):0.1})

def hygov_original(grid,syn_data):
    bus_name = syn_data['bus']
    gov_data = syn_data['gov']
    
    p_r = sym.Symbol(f"p_r_{bus_name}", real=True)
    omega_ref = sym.Symbol(f"omega_ref_{bus_name}", real=True)
    omega = sym.Symbol(f"omega_{bus_name}", real=True)

    # dynamic states:
    xi_omega = sym.Symbol(f"xi_omega_{bus_name}", real=True)
    servo = sym.Symbol(f"servo_{bus_name}", real=True)
    pos = sym.Symbol(f"pos_{bus_name}", real=True)
    flow = sym.Symbol(f"flow_{bus_name}", real=True)
    # algebraic states:    
    servo_u = sym.Symbol(f"servo_u_{bus_name}", real=True)
    gate = sym.Symbol(f"gate_{bus_name}", real=True)
    head = sym.Symbol(f"head_{bus_name}", real=True)
    p_m = sym.Symbol(f"p_m_{bus_name}", real=True)
    # parameters:
    Droop = sym.Symbol(f"Droop_{bus_name}", real=True) 
    K_p_gov = sym.Symbol(f"K_p_gov_{bus_name}", real=True)
    K_i_gov = sym.Symbol(f"K_i_gov_{bus_name}", real=True) 
    K_servo = sym.Symbol(f"K_servo_{bus_name}", real=True)  
    T_servo = sym.Symbol(f"T_servo_{bus_name}", real=True) 
    V_gate_max = sym.Symbol(f"V_gate_max_{bus_name}", real=True)
    Gate_max = sym.Symbol(f"Gate_max_{bus_name}", real=True)
    T_w = sym.Symbol(f"T_w_{bus_name}", real=True) 
    Flow_nl = sym.Symbol(f"Flow_nl_{bus_name}", real=True) 
    A_t = sym.Symbol(f"A_t_{bus_name}", real=True)     
    
    epsilon_omega = omega_ref - omega - Droop*gate + p_r
    dxi_omega = epsilon_omega
    g_servo_u = -servo_u + K_p_gov*epsilon_omega + K_i_gov*xi_omega
    dservo = (K_servo*(servo_u-gate) - servo)/T_servo
    dpos = servo #sym.Piecewise((V_gate_max,servo>V_gate_max),(-V_gate_max,servo<-V_gate_max),(servo,True))
    g_gate = -gate + K_p_gov*epsilon_omega # + pos #sym.Piecewise((Gate_max,pos>Gate_max),(1e-6,pos<1e-6),(pos,True))
    dflow =  (1-head)/T_w
    g_head = -head + (gate/(flow+1e-6))**2 
    g_p_m = -p_m + A_t*head*(flow - Flow_nl)

    grid['f'] += [dxi_omega, dservo, dpos, dflow] #[]
    grid['x'] += [ xi_omega,  servo,  pos,  flow]
    grid['g'] += [g_servo_u, g_gate, g_head, g_p_m] # [g_servo_u, g_gate, g_head, g_p_m]
    grid['y'] += [  servo_u,   gate,   head,   p_m] # [  servo_u,   gate,   head,   p_m]
    grid['params'].update({str(Droop):gov_data['Droop'],str(K_p_gov):gov_data['K_p_gov'],str(K_i_gov):gov_data['K_i_gov']})
    grid['params'].update({str(K_servo):gov_data['K_servo'],str(T_servo):gov_data['T_servo']})
    grid['params'].update({str(V_gate_max):gov_data['V_gate_max'],str(Gate_max):gov_data['Gate_max']})
    grid['params'].update({str(T_w):gov_data['T_w'],str(Flow_nl):gov_data['Flow_nl']})
    grid['params'].update({str(A_t):gov_data['A_t']})
    grid['u'].update({str(omega_ref):gov_data['omega_ref']})

def vsgs_add(grid):
    sin = sym.sin
    cos = sym.cos
    buses = grid['data']['buses']
    buses_list = [bus['name'] for bus in buses]
    for vsg in grid['data']['vsgs']:

        bus_name = vsg['bus']
        idx_bus = buses_list.index(bus_name) # get the number of the bus where the syn is connected
        if not 'idx_powers' in buses[idx_bus]: buses[idx_bus].update({'idx_powers':0})
        buses[idx_bus]['idx_powers'] += 1

        p_g = sym.Symbol(f"p_g_{bus_name}_{buses[idx_bus]['idx_powers']}", real=True) # inyected active power (m-pu)
        q_g = sym.Symbol(f"q_g_{bus_name}_{buses[idx_bus]['idx_powers']}", real=True) # inyected reactive power (m-pu)
        V = sym.Symbol(f"V_{bus_name}", real=True)    # bus voltage module (pu)
        theta = sym.Symbol(f"theta_{bus_name}", real=True) # bus voltage angle (rad)
        i_d = sym.Symbol(f"i_d_{bus_name}", real=True)  # d-axe current (pu)
        i_q = sym.Symbol(f"i_q_{bus_name}", real=True)  # q-axe current (pu)
        delta = sym.Symbol(f"delta_{bus_name}", real=True)
        omega_v = sym.Symbol(f"omega_v_{bus_name}", real=True)
        x_wo = sym.Symbol(f"x_wo_{bus_name}", real=True)
        p_m = sym.Symbol(f"p_m_{bus_name}", real=True)
        e_v = sym.Symbol(f"e_v_{bus_name}", real=True)
        p_d2 = sym.Symbol(f"p_d2_{bus_name}", real=True)
        i_d_ref = sym.Symbol(f"i_d_ref_{bus_name}", real=True)
        i_q_ref = sym.Symbol(f"i_q_ref_{bus_name}", real=True)
        q_ref = sym.Symbol(f"q_ref_{bus_name}", real=True)
        xi_q = sym.Symbol(f"xi_q_{bus_name}", real=True)
        p_src = sym.Symbol(f"p_src_{bus_name}", real=True)
        soc_ref = sym.Symbol(f"soc_ref_{bus_name}", real=True)
        p_sto = sym.Symbol(f"p_sto_{bus_name}", real=True)
        Dp_ref = sym.Symbol(f"Dp_ref_{bus_name}", real=True)
        soc = sym.Symbol(f"soc_{bus_name}", real=True)
        xi_soc = sym.Symbol(f"xi_soc_{bus_name}", real=True)
        
        omega_coi = sym.Symbol("omega_coi", real=True)

        v_d = V*sin(delta - theta) 
        v_q = V*cos(delta - theta) 
        #S_n,H,Omega_b,R_v,X_v,D,K_delta = 0,0,0,0,0,0,0
        for item in ['S_n','R_s','H','Omega_b','R_v','X_v','D1','D2','D3','K_delta','T_wo','T_i','K_q','T_q','H_s','K_p_soc','K_i_soc']:
            string = f"{item}=sym.Symbol('{item}_{bus_name}', real = True)" 
            exec(string,globals())
            grid['params'].update({f'{item}_{bus_name}':vsg[item]})

        p_t = i_d*(v_d + R_s*i_d) + i_q*(v_q + R_s*i_q) 

        omega_s = omega_coi
        
        u_wo = omega_v - 1.0
        epsilon_q = q_ref - q_g
        
        dx_wo  =  (u_wo - x_wo)/T_wo 
        g_p_d2 =  D2*(u_wo - x_wo) - p_d2  
        

        p_d1 = D1*(omega_v - 1)

        #  D3 = K_p*Omega_b
        ddelta   = Omega_b*(omega_v - omega_s) + D3*(p_m- p_g) - K_delta*delta
        domega_v = 1/(2*H)*(p_m - p_g - p_d1 - p_d2)
        di_d = 1/T_i*(i_d_ref - i_d)
        di_q = 1/T_i*(i_q_ref - i_q)
        dxi_q = epsilon_q
        dsoc = -p_sto/H_s
        dxi_soc = (soc_ref - soc)
        
        p_soc_ref = K_p_soc*(soc_ref - soc) + K_i_soc*xi_soc
        p_soc = sym.Piecewise((p_soc_ref - (Dp_ref + p_src),(soc<0.0) & (p_sto>0.0)),(p_soc_ref - (Dp_ref + p_src),(soc>1.0) & (p_sto<0.0)), (p_soc_ref, True))
        
        g_i_d_ref  = v_q + R_v*i_q_ref + X_v*i_d_ref - e_v
        g_i_q_ref  = v_d + R_v*i_d_ref - X_v*i_q_ref - 0
        g_p_g  = i_d*v_d + i_q*v_q - p_g  
        g_q_g  = i_d*v_q - i_q*v_d - q_g 
        g_e_v = -e_v +  K_q*(epsilon_q + xi_q/T_q)
        g_p_sto = -p_t + p_sto + p_src 
        g_p_m = -p_m + p_soc + Dp_ref + p_src
        #dv_dc = 1/C_dc*(-i_t + i_src + i_sto)
        #dv_dc = 1/(v_dc*C_dc)*(-p_t + p_src + p_sto)
        
        f_syn = [ddelta,domega_v,dx_wo,di_d,di_q,dxi_q, dsoc,dxi_soc]
        x_syn = [ delta, omega_v, x_wo, i_d, i_q, xi_q,  soc, xi_soc]
        g_syn = [g_i_d_ref,g_i_q_ref,g_p_g,g_q_g,g_p_d2,g_e_v, g_p_sto,g_p_m]
        y_syn = [  i_d_ref,  i_q_ref,  p_g,  q_g,  p_d2,  e_v,   p_sto,  p_m]
        
        if 'f' not in grid: grid.update({'f':[]})
        if 'x' not in grid: grid.update({'x':[]})
        grid['f'] += f_syn
        grid['x'] += x_syn
        grid['g'] += g_syn
        grid['y'] += y_syn  
        
        S_base = sym.Symbol('S_base', real = True)
        grid['u'].update({f'p_in_{bus_name}':0.0})
        grid['u'].update({f'Dp_ref_{bus_name}':0.0})
        grid['u'].update({f'q_ref_{bus_name}':0.0})
        grid['u'].update({f'p_src_{bus_name}':vsg['p_src']})
        grid['u'].update({f'soc_ref_{bus_name}':vsg['soc_ref']})
        grid['g'][idx_bus*2]   += -p_g*S_n/S_base
        grid['g'][idx_bus*2+1] += -q_g*S_n/S_base
        grid['h'].update({f"p_t_{bus_name}":p_t})
        grid['h'].update({f"p_soc_{bus_name}":p_soc})

def uvsgs_add(grid):
    sin = sym.sin
    cos = sym.cos
    buses = grid['data']['buses']
    buses_list = [bus['name'] for bus in buses]
    for uvsg in grid['data']['uvsgs']:

        bus_name = uvsg['bus']
        idx_bus = buses_list.index(bus_name) # get the number of the bus where the syn is connected
        if not 'idx_powers' in buses[idx_bus]: buses[idx_bus].update({'idx_powers':0})
        buses[idx_bus]['idx_powers'] += 1

        p_g = sym.Symbol(f"p_g_{bus_name}_{buses[idx_bus]['idx_powers']}", real=True)
        q_g = sym.Symbol(f"q_g_{bus_name}_{buses[idx_bus]['idx_powers']}", real=True)
        V = sym.Symbol(f"V_{bus_name}", real=True)
        theta = sym.Symbol(f"theta_{bus_name}", real=True)
        i_d = sym.Symbol(f"i_d_{bus_name}", real=True)
        i_q = sym.Symbol(f"i_q_{bus_name}", real=True)
        delta = sym.Symbol(f"delta_{bus_name}", real=True)
        omega_v = sym.Symbol(f"omega_v_{bus_name}", real=True)
        p_m = sym.Symbol(f"p_m_{bus_name}", real=True)
        e_v = sym.Symbol(f"e_v_{bus_name}", real=True)
        i_d_ref = sym.Symbol(f"i_d_ref_{bus_name}", real=True)
        i_q_ref = sym.Symbol(f"i_q_ref_{bus_name}", real=True)
        q_ref = sym.Symbol(f"q_ref_{bus_name}", real=True)
        xi_p = sym.Symbol(f"xi_p_{bus_name}", real=True)
        xi_q = sym.Symbol(f"xi_q_{bus_name}", real=True)
      
        omega_coi = sym.Symbol("omega_coi", real=True)

        v_d = V*sin(delta - theta) 
        v_q = V*cos(delta - theta) 
        #S_n,H,Omega_b,R_v,X_v,D,K_delta = 0,0,0,0,0,0,0
        for item in ['S_n','Omega_b','R_v','X_v','K_delta','T_i','K_q','T_q','K_p','K_i']:
            string = f"{item}=sym.Symbol('{item}_{bus_name}', real = True)" 
            exec(string,globals())
            grid['params'].update({f'{item}_{bus_name}':uvsg[item]})

        p_e = i_d*(v_d + R_v*i_d) + i_q*(v_q + R_v*i_q) 

        omega_s = omega_coi
        
        epsilon_p = p_m - p_g
        epsilon_q = q_ref - q_g
        
        # digsilent: omega_v = K_p*(epsilon_p + xi_p/T_p) + 1.0  ! UVSG PI             
        # digsilent: phi_v. =  2.0*pi()*50.0*(omega_v-fref)  
   

        omega_v = (K_p*epsilon_p + K_i*xi_p)  + 1.0

        ddelta   = Omega_b*(omega_v - omega_s) - K_delta*delta
        dxi_p = epsilon_p
        di_d = 1/T_i*(i_d_ref - i_d)
        di_q = 1/T_i*(i_q_ref - i_q)
        dxi_q = epsilon_q
        
        
        g_i_d_ref  = v_q + R_v*i_q_ref + X_v*i_d_ref - e_v
        g_i_q_ref  = v_d + R_v*i_d_ref - X_v*i_q_ref - 0
        g_p_g  = i_d*v_d + i_q*v_q - p_g  
        g_q_g  = i_d*v_q - i_q*v_d - q_g 
        g_e_v = -e_v +  K_q*(epsilon_q + xi_q/T_q)
        
        f_syn = [ddelta,dxi_p,di_d,di_q,dxi_q]
        x_syn = [ delta, xi_p, i_d, i_q, xi_q]
        g_syn = [g_i_d_ref,g_i_q_ref,g_p_g,g_q_g,g_e_v]
        y_syn = [  i_d_ref,  i_q_ref,  p_g,  q_g,  e_v]
        
        if 'f' not in grid: grid.update({'f':[]})
        if 'x' not in grid: grid.update({'x':[]})
        grid['f'] += f_syn
        grid['x'] += x_syn
        grid['g'] += g_syn
        grid['y'] += y_syn  
        
        S_base = sym.Symbol('S_base', real = True)
        grid['u'].update({f'q_ref_{bus_name}':0.0})
        grid['u'].update({f'p_m_{bus_name}':uvsg['p_m']})
        grid['g'][idx_bus*2]   += -p_g*S_n/S_base
        grid['g'][idx_bus*2+1] += -q_g*S_n/S_base
        grid['h'].update({f"p_e_{bus_name}":p_e})
        grid['h'].update({f"omega_v_{bus_name}":omega_v})

def gformer_z_add(grid):
    sin = sym.sin
    cos = sym.cos
    buses = grid['data']['buses']
    buses_list = [bus['name'] for bus in buses]
    for gformer_z in grid['data']['gformers_z']:

        bus_name = gformer_z['bus']
        idx_bus = buses_list.index(bus_name) # get the number of the bus where the syn is connected
        if not 'idx_powers' in buses[idx_bus]: buses[idx_bus].update({'idx_powers':0})
        buses[idx_bus]['idx_powers'] += 1

        p_g = sym.Symbol(f"p_g_{bus_name}_{buses[idx_bus]['idx_powers']}", real=True) # inyected active power (m-pu)
        q_g = sym.Symbol(f"q_g_{bus_name}_{buses[idx_bus]['idx_powers']}", real=True) # inyected reactive power (m-pu)
        V = sym.Symbol(f"V_{bus_name}", real=True)    # bus voltage module (pu)
        theta   = sym.Symbol(f"theta_{bus_name}", real=True) # bus voltage angle (rad)
        theta_v_0 = sym.Symbol(f"theta_v_0_{bus_name}", real=True) # bus voltage angle (rad)

        i_d = sym.Symbol(f"i_d_{bus_name}", real=True)  # d-axe current (pu)
        i_q = sym.Symbol(f"i_q_{bus_name}", real=True)  # q-axe current (pu)
        delta = sym.Symbol(f"delta_{bus_name}", real=True)
        omega_v = sym.Symbol(f"omega_v_{bus_name}", real=True)
        x_wo = sym.Symbol(f"x_wo_{bus_name}", real=True)
        p_m = sym.Symbol(f"p_m_{bus_name}", real=True)
        e_v = sym.Symbol(f"e_v_{bus_name}", real=True)
        p_d2 = sym.Symbol(f"p_d2_{bus_name}", real=True)
        i_d_ref = sym.Symbol(f"i_d_ref_{bus_name}", real=True)
        i_q_ref = sym.Symbol(f"i_q_ref_{bus_name}", real=True)
        q_ref = sym.Symbol(f"q_ref_{bus_name}", real=True)
        xi_q = sym.Symbol(f"xi_q_{bus_name}", real=True)
        p_src = sym.Symbol(f"p_src_{bus_name}", real=True)
        soc_ref = sym.Symbol(f"soc_ref_{bus_name}", real=True)
        p_sto = sym.Symbol(f"p_sto_{bus_name}", real=True)
        Dp_ref = sym.Symbol(f"Dp_ref_{bus_name}", real=True)
        soc = sym.Symbol(f"soc_{bus_name}", real=True)
        xi_soc = sym.Symbol(f"xi_soc_{bus_name}", real=True)
        
        omega_coi = sym.Symbol("omega_coi", real=True)

        v_d = V*sin(delta - theta) 
        v_q = V*cos(delta - theta) 
        #S_n,H,Omega_b,R_v,X_v,D,K_delta = 0,0,0,0,0,0,0
        for item in ['S_n','Omega_b','R_v','X_v','T_i','K_delta']:
            string = f"{item}=sym.Symbol('{item}_{bus_name}', real = True)" 
            exec(string,globals())
            grid['params'].update({f'{item}_{bus_name}':gformer_z[item]})

        omega_s = omega_coi
        
        ddelta   = Omega_b*(omega_v - omega_s)  - K_delta*delta
        di_d = 1/T_i*(i_d_ref - i_d)
        di_q = 1/T_i*(i_q_ref - i_q)
          
        g_i_d_ref  = v_q + R_v*i_q_ref + X_v*i_d_ref - e_v
        g_i_q_ref  = v_d + R_v*i_d_ref - X_v*i_q_ref - 0
        g_p_g  = i_d*v_d + i_q*v_q - p_g  
        g_q_g  = i_d*v_q - i_q*v_d - q_g 
        
        f_syn = [ ddelta,di_d,di_q]
        x_syn = [  delta, i_d, i_q]
        g_syn = [g_i_d_ref,g_i_q_ref,g_p_g,g_q_g]
        y_syn = [  i_d_ref,  i_q_ref,  p_g,  q_g]
        
        if 'f' not in grid: grid.update({'f':[]})
        if 'x' not in grid: grid.update({'x':[]})
        grid['f'] += f_syn
        grid['x'] += x_syn
        grid['g'] += g_syn
        grid['y'] += y_syn  
        
        S_base = sym.Symbol('S_base', real = True)
        grid['u'].update({f'e_v_{bus_name}':1.0})
        grid['u'].update({f'omega_v_{bus_name}':1.0})
        grid['g'][idx_bus*2]   += -p_g*S_n/S_base
        grid['g'][idx_bus*2+1] += -q_g*S_n/S_base


def gformer_droop_z_add(grid):
    sin = sym.sin
    cos = sym.cos
    buses = grid['data']['buses']
    buses_list = [bus['name'] for bus in buses]
    for gformer_droop_z in grid['data']['gformers_droop_z']:

        bus_name = gformer_droop_z['bus']
        idx_bus = buses_list.index(bus_name) # get the number of the bus where the syn is connected
        if not 'idx_powers' in buses[idx_bus]: buses[idx_bus].update({'idx_powers':0})
        buses[idx_bus]['idx_powers'] += 1

        p_g = sym.Symbol(f"p_g_{bus_name}_{buses[idx_bus]['idx_powers']}", real=True) # inyected active power (m-pu)
        q_g = sym.Symbol(f"q_g_{bus_name}_{buses[idx_bus]['idx_powers']}", real=True) # inyected reactive power (m-pu)
        V = sym.Symbol(f"V_{bus_name}", real=True)    # bus voltage module (pu)
        theta   = sym.Symbol(f"theta_{bus_name}", real=True) # bus voltage angle (rad)

        p_c = sym.Symbol(f"p_c_{bus_name}", real=True)
        p_r = sym.Symbol(f"p_r_{bus_name}", real=True)

        q_c = sym.Symbol(f"q_c_{bus_name}", real=True)
        q_r = sym.Symbol(f"q_r_{bus_name}", real=True)
        
        i_d = sym.Symbol(f"i_d_{bus_name}", real=True)  # d-axe current (pu)
        i_q = sym.Symbol(f"i_q_{bus_name}", real=True)  # q-axe current (pu)
        delta = sym.Symbol(f"delta_{bus_name}", real=True)
        omega_v = sym.Symbol(f"omega_v_{bus_name}", real=True)
        e_v = sym.Symbol(f"e_v_{bus_name}", real=True)
        i_d_ref = sym.Symbol(f"i_d_ref_{bus_name}", real=True)
        i_q_ref = sym.Symbol(f"i_q_ref_{bus_name}", real=True)

        omega_v_0 = sym.Symbol(f"omega_v_0_{bus_name}", real=True)
        e_v_0 = sym.Symbol(f"e_v_0_{bus_name}", real=True)
        

        S_n = sym.Symbol(f"S_n_{bus_name}", real=True)
        Omega_b = sym.Symbol(f"Omega_b_{bus_name}", real=True)
        R_v = sym.Symbol(f"R_v_{bus_name}", real=True)
        X_v = sym.Symbol(f"X_v_{bus_name}", real=True)
        K_delta = sym.Symbol(f"K_delta_{bus_name}", real=True)
        K_droop_p = sym.Symbol(f"K_droop_p_{bus_name}", real=True)
        K_droop_q = sym.Symbol(f"K_droop_q_{bus_name}", real=True)
        T_i = sym.Symbol(f"T_i_{bus_name}", real=True)        
    
        for item in ['S_n','Omega_b','R_v','X_v','K_droop_p','K_droop_q','T_i','K_delta']:
            grid['params'].update({f'{item}_{bus_name}':gformer_droop_z[item]})

        
        omega_coi = sym.Symbol("omega_coi", real=True)

        v_d = V*sin(delta - theta) 
        v_q = V*cos(delta - theta) 
        
        omega_s = omega_coi
        
        g_omega_v = -omega_v + omega_v_0 + K_droop_p*(p_c - p_g + p_r)
        g_e_v     =     -e_v +     e_v_0 + K_droop_q*(q_c - q_g + q_r)
        
        ddelta   = Omega_b*(omega_v - omega_s)  - K_delta*delta
        di_d = 1/T_i*(i_d_ref - i_d)
        di_q = 1/T_i*(i_q_ref - i_q)
          
        g_i_d_ref  = v_q + R_v*i_q_ref + X_v*i_d_ref - e_v
        g_i_q_ref  = v_d + R_v*i_d_ref - X_v*i_q_ref - 0
        g_p_g  = i_d*v_d + i_q*v_q - p_g  
        g_q_g  = i_d*v_q - i_q*v_d - q_g 
        
        f_syn = [ ddelta,di_d,di_q]
        x_syn = [  delta, i_d, i_q]
        g_syn = [g_omega_v,g_e_v,g_i_d_ref,g_i_q_ref,g_p_g,g_q_g]
        y_syn = [  omega_v, e_v, i_d_ref,  i_q_ref,  p_g,  q_g]
        
        if 'f' not in grid: grid.update({'f':[]})
        if 'x' not in grid: grid.update({'x':[]})
        grid['f'] += f_syn
        grid['x'] += x_syn
        grid['g'] += g_syn
        grid['y'] += y_syn  
        
        S_base = sym.Symbol('S_base', real = True)
        grid['u'].update({f'e_v_0_{bus_name}':1.0})
        grid['u'].update({f'omega_v_0_{bus_name}':1.0})
        grid['u'].update({f'p_c_{bus_name}':0.0})
        grid['u'].update({f'q_c_{bus_name}':0.0})
        
        grid['g'][idx_bus*2]   += -p_g*S_n/S_base
        grid['g'][idx_bus*2+1] += -q_g*S_n/S_base
        
        
if __name__ == "__main__":
    
    data = {
        "sys":{"name":"k12p6","S_base":100e6, "K_p_agc":0.01,"K_i_agc":0.01},       
        "buses":[{"name":"1", "P_W":0.0,"Q_var":0.0,"U_kV":20.0},
                 {"name":"2", "P_W":0.0,"Q_var":0.0,"U_kV":20.0},
                 {"name":"3", "P_W":0.0,"Q_var":0.0,"U_kV":20.0},
                 {"name":"4", "P_W":0.0,"Q_var":0.0,"U_kV":20.0},
                 {"name":"5", "P_W":0.0,"Q_var":0.0,"U_kV":230.0},
                 {"name":"6", "P_W":0.0,"Q_var":0.0,"U_kV":230.0},
                 {"name":"7", "P_W":-967e6,"Q_var":100e6,"U_kV":230.0},
                 {"name":"8", "P_W":0.0,"Q_var":0.0,"U_kV":230.0},
                 {"name":"9", "P_W":-1767e6,"Q_var":250e6,"U_kV":230.0},
                 {"name":"10","P_W":0.0,"Q_var":0.0,"U_kV":230.0},
                 {"name":"11","P_W":0.0,"Q_var":0.0,"U_kV":230.0}      
                ],
        "lines":[{"bus_j":"1", "bus_k":"5", "X_pu":0.15,"R_pu":0.0, "S_mva":900.0},
                 {"bus_j":"2", "bus_k":"6", "X_pu":0.15,"R_pu":0.0, "S_mva":900.0},
                 {"bus_j":"3", "bus_k":"11","X_pu":0.15,"R_pu":0.0, "S_mva":900.0},
                 {"bus_j":"4", "bus_k":"10","X_pu":0.15,"R_pu":0.0, "S_mva":900.0},
                 {"bus_j":"5", "bus_k":"6", "X_km":0.529,"R_km":0.0529,"Bs_km":2.1e-6,"km":25},
                 {"bus_j":"6", "bus_k":"7", "X_km":0.529,"R_km":0.0529,"Bs_km":2.1e-6,"km":10},
                 {"bus_j":"7", "bus_k":"8", "X_km":0.529,"R_km":0.0529,"Bs_km":2.1e-6,"km":110},
                 {"bus_j":"7", "bus_k":"8", "X_km":0.529,"R_km":0.0529,"Bs_km":2.1e-6,"km":110},
                 {"bus_j":"8", "bus_k":"9", "X_km":0.529,"R_km":0.0529,"Bs_km":2.1e-6,"km":110},
                 {"bus_j":"8", "bus_k":"9", "X_km":0.529,"R_km":0.0529,"Bs_km":2.1e-6,"km":110},
                 {"bus_j":"9", "bus_k":"10","X_km":0.529,"R_km":0.0529,"Bs_km":2.1e-6,"km":10},
                 {"bus_j":"10","bus_k":"11","X_km":0.529,"R_km":0.0529,"Bs_km":2.1e-6,"km":25}],
        "syns":[{"bus":"1","S_n":900e6,
                "X_d":1.8,"X1d":0.3, "T1d0":8.0,    
                "X_q":1.7,"X1q":0.55,"T1q0":0.4,  
                "R_a":0.0025,"X_l": 0.2, 
                "H":6.5,"D":1.0,
                "Omega_b":314.1592653589793,"omega_s":1.0,"K_sec":0.0,
                "avr":{"type":"sexs","K_a":300, "T_r":0.02, 'K_ai':1e-6,"V_min":-10000.0,"V_max":5.0, "v_pss":0.0, "v_ref":1.03,"K_aw":10},
                "gov":{"type":"agov1","Droop":0.05,"T_1":1.0,"T_2":2.0,"T_3":10.0, "p_c":700/900,"omega_ref":1.0, "K_imw":0.01},
                "pss":{"type":"pss_kundur","T_wo":10.0,"T_1":0.1, 'T_2':0.1, 'K_stab':1.0},
                "K_delta":0.001},
              {"bus":"2","S_n":900e6,
                "X_d":1.8,"X1d":0.3, "T1d0":8.0,    
                "X_q":1.7,"X1q":0.55,"T1q0":0.4,  
                "R_a":0.0025,"X_l": 0.2, 
                "H":6.5,"D":1.0,
                "Omega_b":314.1592653589793,"omega_s":1.0,"K_sec":0.0,
                "avr":{"type":"sexs","K_a":300, "T_r":0.02, 'K_ai':1e-6,"V_min":-10000.0,"V_max":5.0,  "v_pss":0.0, "v_ref":1.01,"K_aw":10},
                "gov":{"type":"agov1","Droop":0.05,"T_1":1.0,"T_2":2.0,"T_3":10.0, "p_c":700/900,"omega_ref":1.0, "K_imw":0.01},
                "pss":{"type":"pss_kundur","T_wo":10.0,"T_1":0.1, 'T_2':0.1, 'K_stab':1.0},
                "K_delta":0.0},
              {"bus":"3","S_n":900e6,
                "X_d":1.8,"X1d":0.3, "T1d0":8.0,    
                "X_q":1.7,"X1q":0.55,"T1q0":0.4,  
                "R_a":0.0025,"X_l": 0.2, 
                "H":6.175,"D":1.0,
                "Omega_b":314.1592653589793,"omega_s":1.0,"K_sec":0.01,
                "avr":{"type":"sexs","K_a":300, "T_r":0.02, 'K_ai':1e-6,"V_min":-10000.0,"V_max":5.0, "v_pss":0.0, "v_ref":1.03,"K_aw":10},
                "gov":{"type":"agov1","Droop":0.05,"T_1":1.0,"T_2":2.0,"T_3":10.0, "p_c":700/900,"omega_ref":1.0, "K_imw":0.0},
                "pss":{"type":"pss_kundur","T_wo":10.0,"T_1":0.1, 'T_2':0.1, 'K_stab':1.0},
                "K_delta":0.0},
              {"bus":"4","S_n":900e6,
                "X_d":1.8,"X1d":0.3, "T1d0":8.0,    
                "X_q":1.7,"X1q":0.55,"T1q0":0.4,  
                "R_a":0.0025,"X_l": 0.2, 
                "H":6.175,"D":1.0,
                "Omega_b":314.1592653589793,"omega_s":1.0,"K_sec":0.0,
                "avr":{"type":"sexs","K_a":300, "T_r":0.02, 'K_ai':1e-6,"V_min":-10000.0,"V_max":5.0, "v_pss":0.0, "v_ref":1.01,"K_aw":10},
                "gov":{"type":"agov1","Droop":0.05,"T_1":1.0,"T_2":2.0,"T_3":10.0, "p_c":700/900,"omega_ref":1.0, "K_imw":0.01},
                "pss":{"type":"pss_kundur","T_wo":10.0,"T_1":0.1, 'T_2':0.1, 'K_stab':1.0},
                "K_delta":0.0}
                 ]
        }
    
    bpu_obj = bpu(data_input=data)
    import pydae.build as db
    
    params_dict = bpu_obj.dae['params_dict']
    
    theta_1 = sym.Symbol('theta_1',real=True)

    g_list = bpu_obj.dae['g'] 
    h_dict = bpu_obj.dae['h_dict']
    f_list = bpu_obj.dae['f']
    x_list = bpu_obj.dae['x']
    
    sys = {'name':'k12p6_pss',
           'params_dict':params_dict,
           'f_list':f_list,
           'g_list':g_list,
           'x_list':x_list,
           'y_ini_list':bpu_obj.dae['y_ini'],
           'y_run_list':bpu_obj.dae['y_run'],
           'u_run_dict':bpu_obj.dae['u_run_dict'],
           'u_ini_dict':bpu_obj.dae['u_ini_dict'],
           'h_dict':h_dict}
    
    #sys = db.system(sys)
    #db.sys2num(sys)
    
    from k12p6_pss import k12p6_pss_class
    
    k12p6 = k12p6_pss_class()
    K_sec = 0.01
    K_a = 300.0
    T_r = 0.02
    D =1
    params = {        'P_7':-967e6,'P_9':-1_767e6,'Q_7':100e6,'Q_9':250e6,
                      'K_delta_1':0,'K_delta_2':0,
                      'K_delta_3':0.01,'K_delta_4':0,
                      "V_min_1":0.0,"V_max_1":3.0,
                      "V_min_2":0.0,"V_max_2":3.0,
                      "V_min_3":0.0,"V_max_3":3.0,
                      "V_min_4":0.0,"V_max_4":3.0,
                      'K_a_1':K_a,'K_a_2':K_a,'K_a_3':K_a,'K_a_4':K_a,
                      'K_ai_1':0.0001,'K_ai_2':0.0001,'K_ai_3':0.0001,'K_ai_4':0.0001,
                      'T_r_1':T_r,'T_r_2':T_r,'T_r_3':T_r,'T_r_4':T_r,
                      'p_c_1':700/900,'p_c_2':700/900,'p_c_4':700/900,
                      'v_ref_1':1.03,'v_ref_2':1.01,'v_ref_3':1.03,'v_ref_4':1.01,
                      'v_pss_1':0,'v_pss_3':0,
                      'K_imw_1':0.001,'K_imw_2':0.001,'K_imw_3':0.0,'K_imw_4':0.001,
                      'K_sec_1':0.0,'K_sec_2':0.0,'K_sec_3':K_sec,'K_sec_4':0.0,
                      'K_p_agc':0.001,'K_i_agc':0.001,
                      'D_1':D,'D_2':D,'D_3':D,'D_4':D}
    #k12p6.load_params(params)
    #k12p6.load_0('xy_0_3.json')

    #k12p6.ss()


    grid = k12p6_pss_class()
    
    xy_0_dict = {
        "V_1":1.0,"V_2":1.0,"V_3":1.0,"V_4":1.0,
        "V_5":1.0,"V_6":1.0,"V_7":1.0,"V_8":1.0,
        "V_9":1.0,"V_10":1.0,"V_11":1.0,
        "v_f_1":2,"v_f_2":2,"v_f_3":2,"v_f_4":2,
        "omega_1":1.0,"omega_2":1.0,"omega_3":1.0,"omega_4":1.0,
        "omega_coi":1.0,
    "p_g_1":1.0,"q_g_1":0.0,
    "p_g_2":1.0,"q_g_2":0.0,
    "p_g_3":1.0,"q_g_3":0.0,
    "p_g_4":1.0,"q_g_4":0.0,
    "xi_freq":0.0    
        }
    

    grid.initialize([params],
                       'xy_0.json',compile=True)
    
    grid.run([{'t_end':1.0}])
    grid.run([{'t_end':2.0,'Q_7':-100.0e6}])
    grid.post()
    #grid.save_0('xy_0.json')
    #grid.ss()