import numpy as np
import numba
import scipy.optimize as sopt
import scipy.sparse as sspa
from scipy.sparse.linalg import spsolve,spilu,splu
from numba import cuda
import cffi
import numba.core.typing.cffi_utils as cffi_support

ffi = cffi.FFI()

import cigre_eu_mv_lab_cffi as jacs

cffi_support.register_module(jacs)
f_ini_eval = jacs.lib.f_ini_eval
g_ini_eval = jacs.lib.g_ini_eval
f_run_eval = jacs.lib.f_run_eval
g_run_eval = jacs.lib.g_run_eval
h_eval  = jacs.lib.h_eval

de_jac_ini_xy_eval = jacs.lib.de_jac_ini_xy_eval
de_jac_ini_up_eval = jacs.lib.de_jac_ini_up_eval
de_jac_ini_num_eval = jacs.lib.de_jac_ini_num_eval

sp_jac_ini_xy_eval = jacs.lib.sp_jac_ini_xy_eval
sp_jac_ini_up_eval = jacs.lib.sp_jac_ini_up_eval
sp_jac_ini_num_eval = jacs.lib.sp_jac_ini_num_eval

de_jac_run_xy_eval = jacs.lib.de_jac_run_xy_eval
de_jac_run_up_eval = jacs.lib.de_jac_run_up_eval
de_jac_run_num_eval = jacs.lib.de_jac_run_num_eval

sp_jac_run_xy_eval = jacs.lib.sp_jac_run_xy_eval
sp_jac_run_up_eval = jacs.lib.sp_jac_run_up_eval
sp_jac_run_num_eval = jacs.lib.sp_jac_run_num_eval

de_jac_trap_xy_eval= jacs.lib.de_jac_trap_xy_eval            
de_jac_trap_up_eval= jacs.lib.de_jac_trap_up_eval        
de_jac_trap_num_eval= jacs.lib.de_jac_trap_num_eval

sp_jac_trap_xy_eval= jacs.lib.sp_jac_trap_xy_eval            
sp_jac_trap_up_eval= jacs.lib.sp_jac_trap_up_eval        
sp_jac_trap_num_eval= jacs.lib.sp_jac_trap_num_eval

import json

sin = np.sin
cos = np.cos
atan2 = np.arctan2
sqrt = np.sqrt 
sign = np.sign 
exp = np.exp


class cigre_eu_mv_lab_class: 

    def __init__(self): 

        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 12
        self.N_y = 37 
        self.N_z = 14 
        self.N_store = 10000 
        self.params_list = ['S_base', 'g_N00_N02', 'b_N00_N02', 'bs_N00_N02', 'g_N02_N03', 'b_N02_N03', 'bs_N02_N03', 'g_N03_N04', 'b_N03_N04', 'bs_N03_N04', 'g_N04_N05', 'b_N04_N05', 'bs_N04_N05', 'g_N05_N06', 'b_N05_N06', 'bs_N05_N06', 'g_N06_N07', 'b_N06_N07', 'bs_N06_N07', 'g_N07_N08', 'b_N07_N08', 'bs_N07_N08', 'g_N08_N09', 'b_N08_N09', 'bs_N08_N09', 'g_N09_N10', 'b_N09_N10', 'bs_N09_N10', 'g_N10_N11', 'b_N10_N11', 'bs_N10_N11', 'g_N11_N04', 'b_N11_N04', 'bs_N11_N04', 'g_N03_N08', 'b_N03_N08', 'bs_N03_N08', 'g_N00_N13', 'b_N00_N13', 'bs_N00_N13', 'g_N13_N14', 'b_N13_N14', 'bs_N13_N14', 'U_N00_n', 'U_N02_n', 'U_N03_n', 'U_N04_n', 'U_N05_n', 'U_N06_n', 'U_N07_n', 'U_N08_n', 'U_N09_n', 'U_N10_n', 'U_N11_n', 'U_N13_n', 'U_N14_n', 'S_n_N00', 'Omega_b_N00', 'H_N00', 'T1d0_N00', 'T1q0_N00', 'X_d_N00', 'X_q_N00', 'X1d_N00', 'X1q_N00', 'D_N00', 'R_a_N00', 'K_delta_N00', 'K_sec_N00', 'K_a_N00', 'K_ai_N00', 'T_r_N00', 'V_min_N00', 'V_max_N00', 'K_aw_N00', 'Droop_N00', 'T_gov_1_N00', 'T_gov_2_N00', 'T_gov_3_N00', 'K_imw_N00', 'omega_ref_N00', 'T_wo_N00', 'T_1_N00', 'T_2_N00', 'K_stab_N00', 'V_lim_N00', 'K_p_agc', 'K_i_agc'] 
        self.params_values_list  = [10000000.0, 0.1867499908813481, -0.12216561903488188, 0.0, 0.4581573484143461, -0.28863912950103804, 0.0, 2.249560632688928, -1.4622144112478033, 0.0, 1.447145279819107, -0.4703222159412097, 0.0, 0.5838883313566281, -0.1810053827205547, 0.0, 2.9357798165137616, -0.8807339449541285, 0.0, 0.5838883313566281, -0.1810053827205547, 0.0, 2.9357798165137616, -0.8807339449541285, 0.0, 2.249560632688928, -1.4622144112478033, 0.0, 1.56479217603912, -0.23471882640586797, 0.0, 0.6293637525813748, -0.08181728783557872, 0.0, 1.248780487804878, -1.5609756097560976, 0.0, 0.08827586206896551, -0.2206896551724138, 0.0, 0.2929732204165713, -0.9228656443121996, 0.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 100000000.0, 314.1592653589793, 6.5, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.001, 1.0, 300, 1e-06, 0.02, -10000.0, 500.0, 10, 0.05, 1.0, 1.0, 1.0, 0.01, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 0.01, 0.01] 
        self.inputs_ini_list = ['P_N00', 'Q_N00', 'P_N02', 'Q_N02', 'P_N03', 'Q_N03', 'P_N04', 'Q_N04', 'P_N05', 'Q_N05', 'P_N06', 'Q_N06', 'P_N07', 'Q_N07', 'P_N08', 'Q_N08', 'P_N09', 'Q_N09', 'P_N10', 'Q_N10', 'P_N11', 'Q_N11', 'P_N13', 'Q_N13', 'P_N14', 'Q_N14', 'v_ref_N00', 'v_pss_N00', 'p_c_N00', 'p_r_N00'] 
        self.inputs_ini_values_list  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0] 
        self.inputs_run_list = ['P_N00', 'Q_N00', 'P_N02', 'Q_N02', 'P_N03', 'Q_N03', 'P_N04', 'Q_N04', 'P_N05', 'Q_N05', 'P_N06', 'Q_N06', 'P_N07', 'Q_N07', 'P_N08', 'Q_N08', 'P_N09', 'Q_N09', 'P_N10', 'Q_N10', 'P_N11', 'Q_N11', 'P_N13', 'Q_N13', 'P_N14', 'Q_N14', 'v_ref_N00', 'v_pss_N00', 'p_c_N00', 'p_r_N00'] 
        self.inputs_run_values_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0] 
        self.outputs_list = ['V_N00', 'V_N02', 'V_N03', 'V_N04', 'V_N05', 'V_N06', 'V_N07', 'V_N08', 'V_N09', 'V_N10', 'V_N11', 'V_N13', 'V_N14', 'p_e_N00'] 
        self.x_list = ['delta_N00', 'omega_N00', 'e1q_N00', 'e1d_N00', 'v_c_N00', 'xi_v_N00', 'x_gov_1_N00', 'x_gov_2_N00', 'xi_imw_N00', 'x_wo_N00', 'x_lead_N00', 'xi_freq'] 
        self.y_run_list = ['V_N00', 'theta_N00', 'V_N02', 'theta_N02', 'V_N03', 'theta_N03', 'V_N04', 'theta_N04', 'V_N05', 'theta_N05', 'V_N06', 'theta_N06', 'V_N07', 'theta_N07', 'V_N08', 'theta_N08', 'V_N09', 'theta_N09', 'V_N10', 'theta_N10', 'V_N11', 'theta_N11', 'V_N13', 'theta_N13', 'V_N14', 'theta_N14', 'i_d_N00', 'i_q_N00', 'p_g_N00', 'q_g_N00', 'v_f_N00', 'p_m_ref_N00', 'p_m_N00', 'z_wo_N00', 'v_pss_N00', 'omega_coi', 'p_agc'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['V_N00', 'theta_N00', 'V_N02', 'theta_N02', 'V_N03', 'theta_N03', 'V_N04', 'theta_N04', 'V_N05', 'theta_N05', 'V_N06', 'theta_N06', 'V_N07', 'theta_N07', 'V_N08', 'theta_N08', 'V_N09', 'theta_N09', 'V_N10', 'theta_N10', 'V_N11', 'theta_N11', 'V_N13', 'theta_N13', 'V_N14', 'theta_N14', 'i_d_N00', 'i_q_N00', 'p_g_N00', 'q_g_N00', 'v_f_N00', 'p_m_ref_N00', 'p_m_N00', 'z_wo_N00', 'v_pss_N00', 'omega_coi', 'p_agc'] 
        self.xy_ini_list = self.x_list + self.y_ini_list 
        self.t = 0.0
        self.it = 0
        self.it_store = 0
        self.xy_prev = np.zeros((self.N_x+self.N_y,1))
        self.initialization_tol = 1e-6
        self.N_u = len(self.inputs_run_list) 
        self.sopt_root_method='hybr'
        self.sopt_root_jac=True
        self.u_ini_list = self.inputs_ini_list
        self.u_ini_values_list = self.inputs_ini_values_list
        self.u_run_list = self.inputs_run_list
        self.u_run_values_list = self.inputs_run_values_list
        self.N_u = len(self.u_run_list)
        self.u_ini = np.array(self.inputs_ini_values_list)
        self.p = np.array(self.params_values_list)
        self.xy_0 = np.zeros((self.N_x+self.N_y,))
        self.xy = np.zeros((self.N_x+self.N_y,))
        self.z = np.zeros((self.N_z,))
        
        # numerical elements of jacobians computing:
        x = self.xy[:self.N_x]
        y = self.xy[self.N_x:]
        
        self.yini2urun = list(set(self.u_run_list).intersection(set(self.y_ini_list)))
        self.uini2yrun = list(set(self.y_run_list).intersection(set(self.u_ini_list)))
        self.Time = np.zeros(self.N_store)
        self.X = np.zeros((self.N_store,self.N_x))
        self.Y = np.zeros((self.N_store,self.N_y))
        self.Z = np.zeros((self.N_store,self.N_z))
        self.iters = np.zeros(self.N_store) 
        self.u_run = np.array(self.u_run_values_list,dtype=np.float64)
 
        ## jac_ini
        self.jac_ini = np.zeros((self.N_x+self.N_y,self.N_x+self.N_y))
        self.sp_jac_ini_ia, self.sp_jac_ini_ja, self.sp_jac_ini_nia, self.sp_jac_ini_nja = sp_jac_ini_vectors()
        data = np.array(self.sp_jac_ini_ia,dtype=np.float64)
        self.sp_jac_ini = sspa.csr_matrix((data, self.sp_jac_ini_ia, self.sp_jac_ini_ja), shape=(self.sp_jac_ini_nia,self.sp_jac_ini_nja))
        self.J_ini_d = np.array(self.sp_jac_ini_ia)*0.0
        self.J_ini_i = np.array(self.sp_jac_ini_ia)
        self.J_ini_p = np.array(self.sp_jac_ini_ja)
        de_jac_ini_eval(self.jac_ini,x,y,self.u_ini,self.p,self.Dt)
        sp_jac_ini_eval(self.J_ini_d,x,y,self.u_ini,self.p,self.Dt) 
        self.fill_factor_ini,self.drop_tol_ini,self.drop_rule_ini = 100,1e-10,'basic'       


        ## jac_run
        self.jac_run = np.zeros((self.N_x+self.N_y,self.N_x+self.N_y))
        self.sp_jac_run_ia, self.sp_jac_run_ja, self.sp_jac_run_nia, self.sp_jac_run_nja = sp_jac_run_vectors()
        data = np.array(self.sp_jac_run_ia,dtype=np.float64)
        self.sp_jac_run = sspa.csr_matrix((data, self.sp_jac_run_ia, self.sp_jac_run_ja), shape=(self.sp_jac_run_nia,self.sp_jac_run_nja))
        self.J_run_d = np.array(self.sp_jac_run_ia)*0.0
        self.J_run_i = np.array(self.sp_jac_run_ia)
        self.J_run_p = np.array(self.sp_jac_run_ja)
        de_jac_run_eval(self.jac_run,x,y,self.u_run,self.p,self.Dt)
        sp_jac_run_eval(self.J_run_d,x,y,self.u_run,self.p,self.Dt)
        
        ## jac_trap
        self.jac_trap = np.zeros((self.N_x+self.N_y,self.N_x+self.N_y))
        self.sp_jac_trap_ia, self.sp_jac_trap_ja, self.sp_jac_trap_nia, self.sp_jac_trap_nja = sp_jac_trap_vectors()
        data = np.array(self.sp_jac_trap_ia,dtype=np.float64)
        self.sp_jac_trap = sspa.csr_matrix((data, self.sp_jac_trap_ia, self.sp_jac_trap_ja), shape=(self.sp_jac_trap_nia,self.sp_jac_trap_nja))
        self.J_trap_d = np.array(self.sp_jac_trap_ia)*0.0
        self.J_trap_i = np.array(self.sp_jac_trap_ia)
        self.J_trap_p = np.array(self.sp_jac_trap_ja)
        de_jac_trap_eval(self.jac_trap,x,y,self.u_run,self.p,self.Dt)
        sp_jac_trap_eval(self.J_trap_d,x,y,self.u_run,self.p,self.Dt)
        self.fill_factor_trap,self.drop_tol_trap,self.drop_rule_trap = 100,1e-10,'basic' 
   

        

        
        self.max_it,self.itol,self.store = 50,1e-8,1 
        self.lmax_it,self.ltol,self.ldamp=50,1e-8,1.0
        self.mode = 0 

        self.lmax_it_ini,self.ltol_ini,self.ldamp_ini=50,1e-8,1.0

        
 
        



        
    def update(self):

        self.Time = np.zeros(self.N_store)
        self.X = np.zeros((self.N_store,self.N_x))
        self.Y = np.zeros((self.N_store,self.N_y))
        self.Z = np.zeros((self.N_store,self.N_z))
        self.iters = np.zeros(self.N_store)
        
    def ss_ini(self):

        xy_ini,it = sstate(self.xy_0,self.u_ini,self.p,self.jac_ini,self.N_x,self.N_y)
        self.xy_ini = xy_ini
        self.N_iters = it
        
        return xy_ini
    
    # def ini(self,up_dict,xy_0={}):

    #     for item in up_dict:
    #         self.set_value(item,up_dict[item])
            
    #     self.xy_ini = self.ss_ini()
    #     self.ini2run()
    #     jac_run_ss_eval_xy(self.jac_run,self.x,self.y_run,self.u_run,self.p)
    #     jac_run_ss_eval_up(self.jac_run,self.x,self.y_run,self.u_run,self.p)
        
    def jac_run_eval(self):
        de_jac_run_eval(self.jac_run,self.x,self.y_run,self.u_run,self.p,self.Dt)
      
    
    def run(self,t_end,up_dict):
        for item in up_dict:
            self.set_value(item,up_dict[item])
            
        t = self.t
        p = self.p
        it = self.it
        it_store = self.it_store
        xy = self.xy
        u = self.u_run
        
        t,it,it_store,xy = daesolver(t,t_end,it,it_store,xy,u,p,
                                  self.jac_trap,
                                  self.Time,
                                  self.X,
                                  self.Y,
                                  self.Z,
                                  self.iters,
                                  self.Dt,
                                  self.N_x,
                                  self.N_y,
                                  self.N_z,
                                  self.decimation,
                                  max_it=self.max_it,itol=self.itol,store=self.store)
        
        self.t = t
        self.it = it
        self.it_store = it_store
        self.xy = xy
 
    def runsp(self,t_end,up_dict):
        for item in up_dict:
            self.set_value(item,up_dict[item])
            
        t = self.t
        p = self.p
        it = self.it
        it_store = self.it_store
        xy = self.xy
        u = self.u_run
        
        t,it,it_store,xy = daesolver_sp(t,t_end,it,it_store,xy,u,p,
                                  self.sp_jac_trap,
                                  self.Time,
                                  self.X,
                                  self.Y,
                                  self.Z,
                                  self.iters,
                                  self.Dt,
                                  self.N_x,
                                  self.N_y,
                                  self.N_z,
                                  self.decimation,
                                  max_it=50,itol=1e-8,store=1)
        
        self.t = t
        self.it = it
        self.it_store = it_store
        self.xy = xy
        
    def post(self):
        
        self.Time = self.Time[:self.it_store]
        self.X = self.X[:self.it_store]
        self.Y = self.Y[:self.it_store]
        self.Z = self.Z[:self.it_store]
        
    def ini2run(self):
        
        ## y_ini to y_run
        self.y_ini = self.xy_ini[self.N_x:]
        self.y_run = np.copy(self.y_ini)
        self.u_run = np.copy(self.u_ini)
        
        ## y_ini to u_run
        for item in self.yini2urun:
            self.u_run[self.u_run_list.index(item)] = self.y_ini[self.y_ini_list.index(item)]
                
        ## u_ini to y_run
        for item in self.uini2yrun:
            self.y_run[self.y_run_list.index(item)] = self.u_ini[self.u_ini_list.index(item)]
            
        
        self.x = self.xy_ini[:self.N_x]
        self.xy[:self.N_x] = self.x
        self.xy[self.N_x:] = self.y_run
        c_h_eval(self.z,self.x,self.y_run,self.u_ini,self.p,self.Dt)
        

        
    def get_value(self,name):
        
        if name in self.inputs_run_list:
            value = self.u_run[self.inputs_run_list.index(name)]
            return value
            
        if name in self.x_list:
            idx = self.x_list.index(name)
            value = self.xy[idx]
            return value
            
        if name in self.y_run_list:
            idy = self.y_run_list.index(name)
            value = self.xy[self.N_x+idy]
            return value
        
        if name in self.params_list:
            idp = self.params_list.index(name)
            value = self.p[idp]
            return value
            
        if name in self.outputs_list:
            idz = self.outputs_list.index(name)
            value = self.z[idz]
            return value

    def get_values(self,name):
        if name in self.x_list:
            values = self.X[:,self.x_list.index(name)]
        if name in self.y_run_list:
            values = self.Y[:,self.y_run_list.index(name)]
        if name in self.outputs_list:
            values = self.Z[:,self.outputs_list.index(name)]
                        
        return values

    def get_mvalue(self,names):
        '''

        Parameters
        ----------
        names : list
            list of variables names to return each value.

        Returns
        -------
        mvalue : TYPE
            list of value of each variable.

        '''
        mvalue = []
        for name in names:
            mvalue += [self.get_value(name)]
                        
        return mvalue
    
    def set_value(self,name_,value):
        if name_ in self.inputs_ini_list or name_ in self.inputs_run_list:
            if name_ in self.inputs_ini_list:
                self.u_ini[self.inputs_ini_list.index(name_)] = value
            if name_ in self.inputs_run_list:
                self.u_run[self.inputs_run_list.index(name_)] = value
            return
        elif name_ in self.params_list:
            self.p[self.params_list.index(name_)] = value
            return
        else:
            print(f'Input or parameter {name_} not found.')
 
    def report_x(self,value_format='5.2f'):
        for item in self.x_list:
            print(f'{item:5s} = {self.get_value(item):5.2f}')

    def report_y(self,value_format='5.2f'):
        for item in self.y_run_list:
            print(f'{item:5s} = {self.get_value(item):5.2f}')
            
    def report_u(self,value_format='5.2f'):
        for item in self.inputs_run_list:
            print(f'{item:5s} = {self.get_value(item):5.2f}')

    def report_z(self,value_format='5.2f'):
        for item in self.outputs_list:
            print(f'{item:5s} = {self.get_value(item):5.2f}')

    def report_params(self,value_format='5.2f'):
        for item in self.params_list:
            print(f'{item:5s} = {self.get_value(item):5.2f}')
            
    def ini(self,up_dict,xy_0={}):
        '''
        Find the steady state of the initialization problem:
            
               0 = f(x,y,u,p) 
               0 = g(x,y,u,p) 

        Parameters
        ----------
        up_dict : dict
            dictionary with all the parameters p and inputs u new values.
        xy_0: if scalar, all the x and y values initial guess are set to the scalar.
              if dict, the initial guesses are applied for the x and y that are in the dictionary
              if string, the initial guess considers a json file with the x and y names and their initial values

        Returns
        -------
        mvalue : TYPE
            list of value of each variable.

        '''
        
        self.it = 0
        self.it_store = 0
        self.t = 0.0
    
        for item in up_dict:
            self.set_value(item,up_dict[item])
            
        if type(xy_0) == dict:
            xy_0_dict = xy_0
            self.dict2xy0(xy_0_dict)
            
        if type(xy_0) == str:
            if xy_0 == 'eval':
                N_x = self.N_x
                self.xy_0_new = np.copy(self.xy_0)*0
                xy0_eval(self.xy_0_new[:N_x],self.xy_0_new[N_x:],self.u_ini,self.p)
                self.xy_0_evaluated = np.copy(self.xy_0_new)
                self.xy_0 = np.copy(self.xy_0_new)
            else:
                self.load_xy_0(file_name = xy_0)
                
        if type(xy_0) == float or type(xy_0) == int:
            self.xy_0 = np.ones(self.N_x+self.N_y,dtype=np.float64)*xy_0

        xy_ini,it = sstate(self.xy_0,self.u_ini,self.p,
                           self.jac_ini,
                           self.N_x,self.N_y,
                           max_it=self.max_it,tol=self.itol)
        
        if it < self.max_it:
            
            self.xy_ini = xy_ini
            self.N_iters = it

            self.ini2run()
            
            self.ini_convergence = True
            
        if it >= self.max_it:
            print(f'Maximum number of iterations (max_it = {self.max_it}) reached without convergence.')
            self.ini_convergence = False
            
        return self.ini_convergence
            
        


    
    def dict2xy0(self,xy_0_dict):
    
        for item in xy_0_dict:
            if item in self.x_list:
                self.xy_0[self.x_list.index(item)] = xy_0_dict[item]
            if item in self.y_ini_list:
                self.xy_0[self.y_ini_list.index(item) + self.N_x] = xy_0_dict[item]
        
    
    def save_xy_0(self,file_name = 'xy_0.json'):
        xy_0_dict = {}
        for item in self.x_list:
            xy_0_dict.update({item:self.get_value(item)})
        for item in self.y_ini_list:
            xy_0_dict.update({item:self.get_value(item)})
    
        xy_0_str = json.dumps(xy_0_dict, indent=4)
        with open(file_name,'w') as fobj:
            fobj.write(xy_0_str)
    
    def load_xy_0(self,file_name = 'xy_0.json'):
        with open(file_name) as fobj:
            xy_0_str = fobj.read()
        xy_0_dict = json.loads(xy_0_str)
    
        for item in xy_0_dict:
            if item in self.x_list:
                self.xy_0[self.x_list.index(item)] = xy_0_dict[item]
            if item in self.y_ini_list:
                self.xy_0[self.y_ini_list.index(item)+self.N_x] = xy_0_dict[item]            

    def load_params(self,data_input):

        if type(data_input) == str:
            json_file = data_input
            self.json_file = json_file
            self.json_data = open(json_file).read().replace("'",'"')
            data = json.loads(self.json_data)
        elif type(data_input) == dict:
            data = data_input

        self.data = data
        for item in self.data:
            self.struct[0][item] = self.data[item]
            if item in self.params_list:
                self.params_values_list[self.params_list.index(item)] = self.data[item]
            elif item in self.inputs_ini_list:
                self.inputs_ini_values_list[self.inputs_ini_list.index(item)] = self.data[item]
            elif item in self.inputs_run_list:
                self.inputs_run_values_list[self.inputs_run_list.index(item)] = self.data[item]
            else: 
                print(f'parameter or input {item} not found')

    def save_params(self,file_name = 'parameters.json'):
        params_dict = {}
        for item in self.params_list:
            params_dict.update({item:self.get_value(item)})

        params_dict_str = json.dumps(params_dict, indent=4)
        with open(file_name,'w') as fobj:
            fobj.write(params_dict_str)

    def save_inputs_ini(self,file_name = 'inputs_ini.json'):
        inputs_ini_dict = {}
        for item in self.inputs_ini_list:
            inputs_ini_dict.update({item:self.get_value(item)})

        inputs_ini_dict_str = json.dumps(inputs_ini_dict, indent=4)
        with open(file_name,'w') as fobj:
            fobj.write(inputs_ini_dict_str)

    def eval_preconditioner_ini(self):
    
        sp_jac_ini_eval(self.sp_jac_ini.data,self.x,self.y_run,self.u_run,self.p,self.Dt)
    
        csc_sp_jac_ini = sspa.csc_matrix(self.sp_jac_ini)
        P_slu = spilu(csc_sp_jac_ini,
                  fill_factor=self.fill_factor_ini,
                  drop_tol=self.drop_tol_ini,
                  drop_rule = self.drop_rule_ini)
    
        self.P_slu = P_slu
        P_d,P_i,P_p,perm_r,perm_c = slu2pydae(P_slu)   
        self.P_d = P_d
        self.P_i = P_i
        self.P_p = P_p
    
        self.perm_r = perm_r
        self.perm_c = perm_c
            
    
    def eval_preconditioner_trap(self):
    
        sp_jac_trap_eval(self.J_trap_d,self.x,self.y_run,self.u_run,self.p,self.Dt)
    
        self.sp_jac_trap.data = self.J_trap_d 
        
        csc_sp_jac_trap = sspa.csc_matrix(self.sp_jac_trap)


        P_slu_trap = spilu(csc_sp_jac_trap,
                          fill_factor=self.fill_factor_trap,
                          drop_tol=self.drop_tol_trap,
                          drop_rule = self.drop_rule_trap)
    
        self.P_slu_trap = P_slu_trap
        P_d,P_i,P_p,perm_r,perm_c = slu2pydae(P_slu_trap)   
        self.P_trap_d = P_d
        self.P_trap_i = P_i
        self.P_trap_p = P_p
    
        self.perm_trap_r = perm_r
        self.perm_trap_c = perm_c
        
    def sprun(self,t_end,up_dict):
        
        for item in up_dict:
            self.set_value(item,up_dict[item])
    
        t = self.t
        p = self.p
        it = self.it
        it_store = self.it_store
        xy = self.xy
        u = self.u_run
        self.iparams_run = np.zeros(10,dtype=np.float64)
    
        t,it,it_store,xy = spdaesolver(t,t_end,it,it_store,xy,u,p,
                                  self.jac_trap,
                                  self.J_trap_d,self.J_trap_i,self.J_trap_p,
                                  self.P_trap_d,self.P_trap_i,self.P_trap_p,self.perm_trap_r,self.perm_trap_c,
                                  self.Time,
                                  self.X,
                                  self.Y,
                                  self.Z,
                                  self.iters,
                                  self.Dt,
                                  self.N_x,
                                  self.N_y,
                                  self.N_z,
                                  self.decimation,
                                  self.iparams_run,
                                  max_it=self.max_it,itol=self.max_it,store=self.store,
                                  lmax_it=self.lmax_it,ltol=self.ltol,ldamp=self.ldamp,mode=self.mode)
    
        self.t = t
        self.it = it
        self.it_store = it_store
        self.xy = xy
            
    def spini(self,up_dict,xy_0={}):
    
        self.it = 0
        self.it_store = 0
        self.t = 0.0
    
        for item in up_dict:
            self.set_value(item,up_dict[item])
    
        if type(xy_0) == dict:
            xy_0_dict = xy_0
            self.dict2xy0(xy_0_dict)
    
        if type(xy_0) == str:
            if xy_0 == 'eval':
                N_x = self.N_x
                self.xy_0_new = np.copy(self.xy_0)*0
                xy0_eval(self.xy_0_new[:N_x],self.xy_0_new[N_x:],self.u_ini,self.p)
                self.xy_0_evaluated = np.copy(self.xy_0_new)
                self.xy_0 = np.copy(self.xy_0_new)
            else:
                self.load_xy_0(file_name = xy_0)
    
        self.xy_ini = self.spss_ini()
        self.ini2run()
        #jac_run_ss_eval_xy(self.jac_run,self.x,self.y_run,self.u_run,self.p)
        #jac_run_ss_eval_up(self.jac_run,self.x,self.y_run,self.u_run,self.p)

        
    def spss_ini(self):
        J_d,J_i,J_p = csr2pydae(self.sp_jac_ini)
        
        xy_ini,it,iparams = spsstate(self.xy,self.u_ini,self.p,
                 J_d,J_i,J_p,
                 self.P_d,self.P_i,self.P_p,self.perm_r,self.perm_c,
                 self.N_x,self.N_y,
                 max_it=self.max_it,tol=self.itol,
                 lmax_it=self.lmax_it_ini,
                 ltol=self.ltol_ini,
                 ldamp=self.ldamp)

 
        self.xy_ini = xy_ini
        self.N_iters = it
        self.iparams = iparams
    
        return xy_ini

    #def import_cffi(self):
        

        

           
            



def daesolver_sp(t,t_end,it,it_store,xy,u,p,sp_jac_trap,T,X,Y,Z,iters,Dt,N_x,N_y,N_z,decimation,max_it=50,itol=1e-8,store=1): 

    fg = np.zeros((N_x+N_y,1),dtype=np.float64)
    fg_i = np.zeros((N_x+N_y),dtype=np.float64)
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    h = np.zeros((N_z),dtype=np.float64)
    sp_jac_trap_eval_up(sp_jac_trap.data,x,y,u,p,Dt,xyup=1)
    
    if it == 0:
        f_run_eval(f,x,y,u,p)
        h_eval(h,x,y,u,p)
        it_store = 0  
        T[0] = t 
        X[0,:] = x  
        Y[0,:] = y  
        Z[0,:] = h  

    while t<t_end: 
        it += 1
        t += Dt

        f_run_eval(f,x,y,u,p)
        g_run_eval(g,x,y,u,p)

        x_0 = np.copy(x) 
        y_0 = np.copy(y) 
        f_0 = np.copy(f) 
        g_0 = np.copy(g) 
            
        for iti in range(max_it):
            f_run_eval(f,x,y,u,p)
            g_run_eval(g,x,y,u,p)
            sp_jac_trap_eval(sp_jac_trap.data,x,y,u,p,Dt,xyup=1)            

            f_n_i = x - x_0 - 0.5*Dt*(f+f_0) 

            fg_i[:N_x] = f_n_i
            fg_i[N_x:] = g
            
            Dxy_i = spsolve(sp_jac_trap,-fg_i) 

            x = x + Dxy_i[:N_x]
            y = y + Dxy_i[N_x:]              

            # iteration stop
            max_relative = 0.0
            for it_var in range(N_x+N_y):
                abs_value = np.abs(xy[it_var])
                if abs_value < 0.001:
                    abs_value = 0.001
                relative_error = np.abs(Dxy_i[it_var])/abs_value

                if relative_error > max_relative: max_relative = relative_error

            if max_relative<itol:
                break
                
        h_eval(h,x,y,u,p)
        xy[:N_x] = x
        xy[N_x:] = y
        
        # store in channels 
        if store == 1:
            if it >= it_store*decimation: 
                T[it_store+1] = t 
                X[it_store+1,:] = x 
                Y[it_store+1,:] = y
                Z[it_store+1,:] = h
                iters[it_store+1] = iti
                it_store += 1 

    return t,it,it_store,xy




@numba.njit()
def sprichardson(A_d,A_i,A_p,b,P_d,P_i,P_p,perm_r,perm_c,x,iparams,damp=1.0,max_it=100,tol=1e-3):
    N_A = A_p.shape[0]-1
    f = np.zeros(N_A)
    for it in range(max_it):
        spMvmul(N_A,A_d,A_i,A_p,x,f) 
        f -= b                          # A@x-b
        x = x - damp*splu_solve(P_d,P_i,P_p,perm_r,perm_c,f)   
        if np.linalg.norm(f,2) < tol: break
    iparams[0] = it
    return x
    
    

@numba.njit()
def spsstate(xy,u,p,
             J_d,J_i,J_p,
             P_d,P_i,P_p,perm_r,perm_c,
             N_x,N_y,
             max_it=50,tol=1e-8,
             lmax_it=20,ltol=1e-8,ldamp=1.0):
    
   
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    iparams = np.array([0],dtype=np.int64)    
    
    f_c_ptr=ffi.from_buffer(np.ascontiguousarray(f))
    g_c_ptr=ffi.from_buffer(np.ascontiguousarray(g))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))
    J_d_ptr=ffi.from_buffer(np.ascontiguousarray(J_d))

    sp_jac_ini_num_eval(J_d_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
    sp_jac_ini_up_eval(J_d_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
    
    #sp_jac_ini_eval_up(J_d,x,y,u,p,0.0)

    Dxy = np.zeros(N_x + N_y)
    for it in range(max_it):
        
        x = xy[:N_x]
        y = xy[N_x:]   
       
        sp_jac_ini_xy_eval(J_d_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)

        
        f_ini_eval(f_c_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
        g_ini_eval(g_c_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
        
        #f_ini_eval(f,x,y,u,p)
        #g_ini_eval(g,x,y,u,p)
        
        fg[:N_x] = f
        fg[N_x:] = g
               
        Dxy = sprichardson(J_d,J_i,J_p,-fg,P_d,P_i,P_p,perm_r,perm_c,Dxy,iparams,damp=ldamp,max_it=lmax_it,tol=ltol)
   
        xy += Dxy
        #if np.max(np.abs(fg))<tol: break
        if np.linalg.norm(fg,np.inf)<tol: break

    return xy,it,iparams


    
@numba.njit() 
def daesolver(t,t_end,it,it_store,xy,u,p,jac_trap,T,X,Y,Z,iters,Dt,N_x,N_y,N_z,decimation,max_it=50,itol=1e-8,store=1): 


    fg = np.zeros((N_x+N_y,1),dtype=np.float64)
    fg_i = np.zeros((N_x+N_y),dtype=np.float64)
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    h = np.zeros((N_z),dtype=np.float64)
    
    f_ptr=ffi.from_buffer(np.ascontiguousarray(f))
    g_ptr=ffi.from_buffer(np.ascontiguousarray(g))
    h_ptr=ffi.from_buffer(np.ascontiguousarray(h))
    x_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    jac_trap_ptr=ffi.from_buffer(np.ascontiguousarray(jac_trap))
    
    de_jac_trap_num_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)    
    de_jac_trap_up_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    de_jac_trap_xy_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    
    if it == 0:
        f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        h_eval(h_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        it_store = 0  
        T[0] = t 
        X[0,:] = x  
        Y[0,:] = y  
        Z[0,:] = h  

    while t<t_end: 
        it += 1
        t += Dt

        f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)

        x_0 = np.copy(x) 
        y_0 = np.copy(y) 
        f_0 = np.copy(f) 
        g_0 = np.copy(g) 
            
        for iti in range(max_it):
            f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
            g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
            de_jac_trap_xy_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 

            f_n_i = x - x_0 - 0.5*Dt*(f+f_0) 

            fg_i[:N_x] = f_n_i
            fg_i[N_x:] = g
            
            Dxy_i = np.linalg.solve(-jac_trap,fg_i) 

            x += Dxy_i[:N_x]
            y += Dxy_i[N_x:] 
            
            #print(Dxy_i)

            # iteration stop
            max_relative = 0.0
            for it_var in range(N_x+N_y):
                abs_value = np.abs(xy[it_var])
                if abs_value < 0.001:
                    abs_value = 0.001
                relative_error = np.abs(Dxy_i[it_var])/abs_value

                if relative_error > max_relative: max_relative = relative_error

            if max_relative<itol:
                break
                
        h_eval(h_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        xy[:N_x] = x
        xy[N_x:] = y
        
        # store in channels 
        if store == 1:
            if it >= it_store*decimation: 
                T[it_store+1] = t 
                X[it_store+1,:] = x 
                Y[it_store+1,:] = y
                Z[it_store+1,:] = h
                iters[it_store+1] = iti
                it_store += 1 

    return t,it,it_store,xy
    
@numba.njit() 
def spdaesolver(t,t_end,it,it_store,xy,u,p,jac_trap,
                J_d,J_i,J_p,
                P_d,P_i,P_p,perm_r,perm_c,
                T,X,Y,Z,iters,Dt,N_x,N_y,N_z,decimation,
                iparams,
                max_it=50,itol=1e-8,store=1,
                lmax_it=20,ltol=1e-4,ldamp=1.0,mode=0):

    fg = np.zeros((N_x+N_y,1),dtype=np.float64)
    fg_i = np.zeros((N_x+N_y),dtype=np.float64)
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    h = np.zeros((N_z),dtype=np.float64)
    Dxy_i_0 = np.zeros(N_x+N_y,dtype=np.float64) 
    f_ptr=ffi.from_buffer(np.ascontiguousarray(f))
    g_ptr=ffi.from_buffer(np.ascontiguousarray(g))
    h_ptr=ffi.from_buffer(np.ascontiguousarray(h))
    x_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    J_d_ptr=ffi.from_buffer(np.ascontiguousarray(J_d))
    
    sp_jac_trap_num_eval(J_d_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)    
    sp_jac_trap_up_eval( J_d_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    sp_jac_trap_xy_eval( J_d_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    
    if it == 0:
        f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        h_eval(h_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        it_store = 0  
        T[0] = t 
        X[0,:] = x  
        Y[0,:] = y  
        Z[0,:] = h  

    while t<t_end: 
        it += 1
        t += Dt

        f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)

        x_0 = np.copy(x) 
        y_0 = np.copy(y) 
        f_0 = np.copy(f) 
        g_0 = np.copy(g) 
            
        for iti in range(max_it):
            f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
            g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
            sp_jac_trap_xy_eval(J_d_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 

            f_n_i = x - x_0 - 0.5*Dt*(f+f_0) 

            fg_i[:N_x] = f_n_i
            fg_i[N_x:] = g
            
            #Dxy_i = np.linalg.solve(-jac_trap,fg_i) 
            Dxy_i = sprichardson(J_d,J_i,J_p,-fg_i,P_d,P_i,P_p,perm_r,perm_c,
                                 Dxy_i_0,iparams,damp=ldamp,max_it=lmax_it,tol=ltol)

            x += Dxy_i[:N_x]
            y += Dxy_i[N_x:] 
            
            #print(Dxy_i)

            # iteration stop
            max_relative = 0.0
            for it_var in range(N_x+N_y):
                abs_value = np.abs(xy[it_var])
                if abs_value < 0.001:
                    abs_value = 0.001
                relative_error = np.abs(Dxy_i[it_var])/abs_value

                if relative_error > max_relative: max_relative = relative_error

            if max_relative<itol:
                break
                
        h_eval(h_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        xy[:N_x] = x
        xy[N_x:] = y
        
        # store in channels 
        if store == 1:
            if it >= it_store*decimation: 
                T[it_store+1] = t 
                X[it_store+1,:] = x 
                Y[it_store+1,:] = y
                Z[it_store+1,:] = h
                iters[it_store+1] = iti
                it_store += 1 

    return t,it,it_store,xy


@cuda.jit()
def ode_solve(x,u,p,f_run,u_idxs,z_i,z,sim):

    N_i,N_j,N_x,N_z,Dt = sim

    # index of thread on GPU:
    i = cuda.grid(1)

    if i < x.size:
        for j in range(N_j):
            f_run_eval(f_run[i,:],x[i,:],u[i,u_idxs[j],:],p[i,:])
            for k in range(N_x):
              x[i,k] +=  Dt*f_run[i,k]

            # outputs in time range
            #z[i,j] = u[i,idxs[j],0]
            z[i,j] = x[i,1]
        h_eval(z_i[i,:],x[i,:],u[i,u_idxs[j],:],p[i,:])
        
def csr2pydae(A_csr):
    '''
    From scipy CSR to the three vectors:
    
    - data
    - indices
    - indptr
    
    '''
    
    A_d = A_csr.data
    A_i = A_csr.indices
    A_p = A_csr.indptr
    
    return A_d,A_i,A_p
    
def slu2pydae(P_slu):
    '''
    From SupderLU matrix to the three vectors:
    
    - data
    - indices
    - indptr
    
    and the premutation vectors:
    
    - perm_r
    - perm_c
    
    '''
    N = P_slu.shape[0]
    #P_slu_full = P_slu.L.A - sspa.eye(N,format='csr') + P_slu.U.A
    P_slu_full = P_slu.L - sspa.eye(N,format='csc') + P_slu.U
    perm_r = P_slu.perm_r
    perm_c = P_slu.perm_c
    P_csr = sspa.csr_matrix(P_slu_full)
    
    P_d = P_csr.data
    P_i = P_csr.indices
    P_p = P_csr.indptr
    
    return P_d,P_i,P_p,perm_r,perm_c

@numba.njit(cache=True)
def spMvmul(N,A_data,A_indices,A_indptr,x,y):
    '''
    y = A @ x
    
    with A in sparse CRS form
    '''
    #y = np.zeros(x.shape[0])
    for i in range(N):
        y[i] = 0.0
        for j in range(A_indptr[i],A_indptr[i + 1]):
            y[i] = y[i] + A_data[j]*x[A_indices[j]]
            
            
@numba.njit(cache=True)
def splu_solve(LU_d,LU_i,LU_p,perm_r,perm_c,b):
    N = len(b)
    y = np.zeros(N)
    x = np.zeros(N)
    z = np.zeros(N)
    bp = np.zeros(N)
    
    for i in range(N): 
        bp[perm_r[i]] = b[i]
        
    for i in range(N): 
        y[i] = bp[i]
        for j in range(LU_p[i],LU_p[i+1]):
            if LU_i[j]>i-1: break
            y[i] -= LU_d[j] * y[LU_i[j]]

    for i in range(N-1,-1,-1): #(int i = N - 1; i >= 0; i--) 
        z[i] = y[i]
        den = 0.0
        for j in range(LU_p[i],LU_p[i+1]): #(int k = i + 1; k < N; k++)
            if LU_i[j] > i:
                z[i] -= LU_d[j] * z[LU_i[j]]
            if LU_i[j] == i: den = LU_d[j]
        z[i] = z[i]/den
 
    for i in range(N):
        x[i] = z[perm_c[i]]
        
    return x



@numba.njit("float64[:,:](float64[:,:],float64[:],float64[:],float64[:],float64[:],float64)")
def de_jac_ini_eval(de_jac_ini,x,y,u,p,Dt):   
    '''
    Computes the dense full initialization jacobian:
    
    jac_ini = [[Fx_ini, Fy_ini],
               [Gx_ini, Gy_ini]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    de_jac_ini : (N, N) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (ini problem).
    u : (N_u,) array_like
        Vector with inputs (ini problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with N = N_x+N_y
 
    Returns
    -------
    
    de_jac_ini : (N, N) array_like
                  Updated matrix.    
    
    '''
    
    de_jac_ini_ptr=ffi.from_buffer(np.ascontiguousarray(de_jac_ini))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    de_jac_ini_num_eval(de_jac_ini_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    de_jac_ini_up_eval( de_jac_ini_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    de_jac_ini_xy_eval( de_jac_ini_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    return de_jac_ini

@numba.njit("float64[:,:](float64[:,:],float64[:],float64[:],float64[:],float64[:],float64)")
def de_jac_run_eval(de_jac_run,x,y,u,p,Dt):   
    '''
    Computes the dense full initialization jacobian:
    
    jac_run = [[Fx_run, Fy_run],
               [Gx_run, Gy_run]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    de_jac_run : (N, N) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (ini problem).
    u : (N_u,) array_like
        Vector with inputs (ini problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with N = N_x+N_y
 
    Returns
    -------
    
    de_jac_ini : (N, N) array_like
                  Updated matrix.    
    
    '''
    
    de_jac_run_ptr=ffi.from_buffer(np.ascontiguousarray(de_jac_run))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    de_jac_run_num_eval(de_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    de_jac_run_up_eval( de_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    de_jac_run_xy_eval( de_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    return de_jac_run

@numba.njit("float64[:,:](float64[:,:],float64[:],float64[:],float64[:],float64[:],float64)")
def de_jac_trap_eval(de_jac_trap,x,y,u,p,Dt):   
    '''
    Computes the dense full trapezoidal jacobian:
    
    jac_trap = [[eye - 0.5*Dt*Fx_run, -0.5*Dt*Fy_run],
                [             Gx_run,         Gy_run]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    de_jac_trap : (N, N) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (run problem).
    u : (N_u,) array_like
        Vector with inputs (run problem). 
    p : (N_p,) array_like
        Vector with parameters. 
 
    Returns
    -------
    
    de_jac_trap : (N, N) array_like
                  Updated matrix.    
    
    '''
        
    de_jac_trap_ptr = ffi.from_buffer(np.ascontiguousarray(de_jac_trap))
    x_c_ptr = ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr = ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr = ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr = ffi.from_buffer(np.ascontiguousarray(p))

    de_jac_trap_num_eval(de_jac_trap_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    de_jac_trap_up_eval( de_jac_trap_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    de_jac_trap_xy_eval( de_jac_trap_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    return de_jac_trap


@numba.njit("float64[:](float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
def sp_jac_run_eval(sp_jac_run,x,y,u,p,Dt):   
    '''
    Computes the sparse full trapezoidal jacobian:
    
    jac_trap = [[eye - 0.5*Dt*Fx_run, -0.5*Dt*Fy_run],
                [             Gx_run,         Gy_run]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    sp_jac_trap : (Nnz,) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (run problem).
    u : (N_u,) array_like
        Vector with inputs (run problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with Nnz the number of non-zeros elements in the jacobian.
 
    Returns
    -------
    
    sp_jac_trap : (Nnz,) array_like
                  Updated matrix.    
    
    '''        
    sp_jac_run_ptr=ffi.from_buffer(np.ascontiguousarray(sp_jac_run))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    sp_jac_run_num_eval(sp_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_jac_run_up_eval( sp_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_jac_run_xy_eval( sp_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    return sp_jac_run

@numba.njit("float64[:](float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
def sp_jac_trap_eval(sp_jac_trap,x,y,u,p,Dt):   
    '''
    Computes the sparse full trapezoidal jacobian:
    
    jac_trap = [[eye - 0.5*Dt*Fx_run, -0.5*Dt*Fy_run],
                [             Gx_run,         Gy_run]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    sp_jac_trap : (Nnz,) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (run problem).
    u : (N_u,) array_like
        Vector with inputs (run problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with Nnz the number of non-zeros elements in the jacobian.
 
    Returns
    -------
    
    sp_jac_trap : (Nnz,) array_like
                  Updated matrix.    
    
    '''        
    sp_jac_trap_ptr=ffi.from_buffer(np.ascontiguousarray(sp_jac_trap))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    sp_jac_trap_num_eval(sp_jac_trap_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_jac_trap_up_eval( sp_jac_trap_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_jac_trap_xy_eval( sp_jac_trap_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    return sp_jac_trap

@numba.njit("float64[:](float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
def sp_jac_ini_eval(sp_jac_ini,x,y,u,p,Dt):   
    '''
    Computes the SPARSE full initialization jacobian:
    
    jac_ini = [[Fx_ini, Fy_ini],
               [Gx_ini, Gy_ini]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    de_jac_ini : (N, N) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (ini problem).
    u : (N_u,) array_like
        Vector with inputs (ini problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with N = N_x+N_y
 
    Returns
    -------
    
    de_jac_ini : (N, N) array_like
                  Updated matrix.    
    
    '''
    
    sp_jac_ini_ptr=ffi.from_buffer(np.ascontiguousarray(sp_jac_ini))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    sp_jac_ini_num_eval(sp_jac_ini_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_jac_ini_up_eval( sp_jac_ini_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_jac_ini_xy_eval( sp_jac_ini_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    return sp_jac_ini


@numba.njit()
def sstate(xy,u,p,jac_ini_ss,N_x,N_y,max_it=50,tol=1e-8):
    
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]

    f_c_ptr=ffi.from_buffer(np.ascontiguousarray(f))
    g_c_ptr=ffi.from_buffer(np.ascontiguousarray(g))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))
    jac_ini_ss_ptr=ffi.from_buffer(np.ascontiguousarray(jac_ini_ss))

    de_jac_ini_num_eval(jac_ini_ss_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
    de_jac_ini_up_eval(jac_ini_ss_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)

    for it in range(max_it):
        de_jac_ini_xy_eval(jac_ini_ss_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
        f_ini_eval(f_c_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
        g_ini_eval(g_c_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
        fg[:N_x] = f
        fg[N_x:] = g
        xy += np.linalg.solve(jac_ini_ss,-fg)
        if np.max(np.abs(fg))<tol: break

    return xy,it


@numba.njit("float64[:](float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
def c_h_eval(z,x,y,u,p,Dt):   
    '''
    Computes the SPARSE full initialization jacobian:
    
    jac_ini = [[Fx_ini, Fy_ini],
               [Gx_ini, Gy_ini]]
                
    for the given x,y,u,p vectors and Dt time increment.
    
    Parameters
    ----------
    de_jac_ini : (N, N) array_like
                  Input data.
    x : (N_x,) array_like
        Vector with dynamical states.
    y : (N_y,) array_like
        Vector with algebraic states (ini problem).
    u : (N_u,) array_like
        Vector with inputs (ini problem). 
    p : (N_p,) array_like
        Vector with parameters. 
        
    with N = N_x+N_y
 
    Returns
    -------
    
    de_jac_ini : (N, N) array_like
                  Updated matrix.    
    
    '''
    
    z_c_ptr=ffi.from_buffer(np.ascontiguousarray(z))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    h_eval(z_c_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    return z

def sp_jac_ini_vectors():

    sp_jac_ini_ia = [0, 1, 47, 0, 1, 12, 13, 38, 39, 44, 47, 2, 38, 42, 3, 39, 4, 12, 4, 5, 42, 46, 6, 43, 6, 7, 8, 40, 1, 9, 10, 45, 47, 12, 13, 14, 15, 34, 35, 40, 12, 13, 14, 15, 34, 35, 41, 12, 13, 14, 15, 16, 17, 12, 13, 14, 15, 16, 17, 14, 15, 16, 17, 18, 19, 26, 27, 14, 15, 16, 17, 18, 19, 26, 27, 16, 17, 18, 19, 20, 21, 32, 33, 16, 17, 18, 19, 20, 21, 32, 33, 18, 19, 20, 21, 22, 23, 18, 19, 20, 21, 22, 23, 20, 21, 22, 23, 24, 25, 20, 21, 22, 23, 24, 25, 22, 23, 24, 25, 26, 27, 22, 23, 24, 25, 26, 27, 16, 17, 24, 25, 26, 27, 28, 29, 16, 17, 24, 25, 26, 27, 28, 29, 26, 27, 28, 29, 30, 31, 26, 27, 28, 29, 30, 31, 28, 29, 30, 31, 32, 33, 28, 29, 30, 31, 32, 33, 18, 19, 30, 31, 32, 33, 18, 19, 30, 31, 32, 33, 12, 13, 34, 35, 36, 37, 12, 13, 34, 35, 36, 37, 34, 35, 36, 37, 34, 35, 36, 37, 0, 2, 12, 13, 38, 39, 0, 3, 12, 13, 38, 39, 0, 12, 13, 38, 39, 40, 0, 12, 13, 38, 39, 41, 4, 5, 42, 46, 1, 8, 43, 48, 6, 7, 44, 1, 9, 45, 10, 45, 46, 1, 47, 11, 47, 48]
    sp_jac_ini_ja = [0, 3, 11, 14, 16, 18, 22, 24, 26, 28, 30, 32, 33, 40, 47, 53, 59, 67, 75, 83, 91, 97, 103, 109, 115, 121, 127, 135, 143, 149, 155, 161, 167, 173, 179, 185, 191, 195, 199, 205, 211, 217, 223, 227, 231, 234, 237, 240, 242, 245]
    sp_jac_ini_nia = 49
    sp_jac_ini_nja = 49
    return sp_jac_ini_ia, sp_jac_ini_ja, sp_jac_ini_nia, sp_jac_ini_nja 

def sp_jac_run_vectors():

    sp_jac_run_ia = [0, 1, 47, 0, 1, 12, 13, 38, 39, 44, 47, 2, 38, 42, 3, 39, 4, 12, 4, 5, 42, 46, 6, 43, 6, 7, 8, 40, 1, 9, 10, 45, 47, 12, 13, 14, 15, 34, 35, 40, 12, 13, 14, 15, 34, 35, 41, 12, 13, 14, 15, 16, 17, 12, 13, 14, 15, 16, 17, 14, 15, 16, 17, 18, 19, 26, 27, 14, 15, 16, 17, 18, 19, 26, 27, 16, 17, 18, 19, 20, 21, 32, 33, 16, 17, 18, 19, 20, 21, 32, 33, 18, 19, 20, 21, 22, 23, 18, 19, 20, 21, 22, 23, 20, 21, 22, 23, 24, 25, 20, 21, 22, 23, 24, 25, 22, 23, 24, 25, 26, 27, 22, 23, 24, 25, 26, 27, 16, 17, 24, 25, 26, 27, 28, 29, 16, 17, 24, 25, 26, 27, 28, 29, 26, 27, 28, 29, 30, 31, 26, 27, 28, 29, 30, 31, 28, 29, 30, 31, 32, 33, 28, 29, 30, 31, 32, 33, 18, 19, 30, 31, 32, 33, 18, 19, 30, 31, 32, 33, 12, 13, 34, 35, 36, 37, 12, 13, 34, 35, 36, 37, 34, 35, 36, 37, 34, 35, 36, 37, 0, 2, 12, 13, 38, 39, 0, 3, 12, 13, 38, 39, 0, 12, 13, 38, 39, 40, 0, 12, 13, 38, 39, 41, 4, 5, 42, 46, 1, 8, 43, 48, 6, 7, 44, 1, 9, 45, 10, 45, 46, 1, 47, 11, 47, 48]
    sp_jac_run_ja = [0, 3, 11, 14, 16, 18, 22, 24, 26, 28, 30, 32, 33, 40, 47, 53, 59, 67, 75, 83, 91, 97, 103, 109, 115, 121, 127, 135, 143, 149, 155, 161, 167, 173, 179, 185, 191, 195, 199, 205, 211, 217, 223, 227, 231, 234, 237, 240, 242, 245]
    sp_jac_run_nia = 49
    sp_jac_run_nja = 49
    return sp_jac_run_ia, sp_jac_run_ja, sp_jac_run_nia, sp_jac_run_nja 

def sp_jac_trap_vectors():

    sp_jac_trap_ia = [0, 1, 47, 0, 1, 12, 13, 38, 39, 44, 47, 2, 38, 42, 3, 39, 4, 12, 4, 5, 42, 46, 6, 43, 6, 7, 8, 40, 1, 9, 10, 45, 11, 47, 12, 13, 14, 15, 34, 35, 40, 12, 13, 14, 15, 34, 35, 41, 12, 13, 14, 15, 16, 17, 12, 13, 14, 15, 16, 17, 14, 15, 16, 17, 18, 19, 26, 27, 14, 15, 16, 17, 18, 19, 26, 27, 16, 17, 18, 19, 20, 21, 32, 33, 16, 17, 18, 19, 20, 21, 32, 33, 18, 19, 20, 21, 22, 23, 18, 19, 20, 21, 22, 23, 20, 21, 22, 23, 24, 25, 20, 21, 22, 23, 24, 25, 22, 23, 24, 25, 26, 27, 22, 23, 24, 25, 26, 27, 16, 17, 24, 25, 26, 27, 28, 29, 16, 17, 24, 25, 26, 27, 28, 29, 26, 27, 28, 29, 30, 31, 26, 27, 28, 29, 30, 31, 28, 29, 30, 31, 32, 33, 28, 29, 30, 31, 32, 33, 18, 19, 30, 31, 32, 33, 18, 19, 30, 31, 32, 33, 12, 13, 34, 35, 36, 37, 12, 13, 34, 35, 36, 37, 34, 35, 36, 37, 34, 35, 36, 37, 0, 2, 12, 13, 38, 39, 0, 3, 12, 13, 38, 39, 0, 12, 13, 38, 39, 40, 0, 12, 13, 38, 39, 41, 4, 5, 42, 46, 1, 8, 43, 48, 6, 7, 44, 1, 9, 45, 10, 45, 46, 1, 47, 11, 47, 48]
    sp_jac_trap_ja = [0, 3, 11, 14, 16, 18, 22, 24, 26, 28, 30, 32, 34, 41, 48, 54, 60, 68, 76, 84, 92, 98, 104, 110, 116, 122, 128, 136, 144, 150, 156, 162, 168, 174, 180, 186, 192, 196, 200, 206, 212, 218, 224, 228, 232, 235, 238, 241, 243, 246]
    sp_jac_trap_nia = 49
    sp_jac_trap_nja = 49
    return sp_jac_trap_ia, sp_jac_trap_ja, sp_jac_trap_nia, sp_jac_trap_nja 
