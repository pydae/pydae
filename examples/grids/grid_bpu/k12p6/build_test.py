import numpy as np
import numba
import scipy.optimize as sopt
import scipy.sparse as sspa
from scipy.sparse.linalg import spsolve,spilu,splu
from numba import cuda
import cffi
import numba.core.typing.cffi_utils as cffi_support

ffi = cffi.FFI()

import build_test_cffi as jacs

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


class build_test_class: 

    def __init__(self): 

        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 45
        self.N_y = 60 
        self.N_z = 15 
        self.N_store = 10000 
        self.params_list = ['S_base', 'g_1_5', 'b_1_5', 'bs_1_5', 'g_2_6', 'b_2_6', 'bs_2_6', 'g_3_11', 'b_3_11', 'bs_3_11', 'g_4_10', 'b_4_10', 'bs_4_10', 'g_5_6', 'b_5_6', 'bs_5_6', 'g_6_7', 'b_6_7', 'bs_6_7', 'g_7_8', 'b_7_8', 'bs_7_8', 'g_8_9', 'b_8_9', 'bs_8_9', 'g_9_10', 'b_9_10', 'bs_9_10', 'g_10_11', 'b_10_11', 'bs_10_11', 'U_1_n', 'U_2_n', 'U_3_n', 'U_4_n', 'U_5_n', 'U_6_n', 'U_7_n', 'U_8_n', 'U_9_n', 'U_10_n', 'U_11_n', 'S_n_1', 'Omega_b_1', 'H_1', 'T1d0_1', 'T1q0_1', 'X_d_1', 'X_q_1', 'X1d_1', 'X1q_1', 'D_1', 'R_a_1', 'K_delta_1', 'K_sec_1', 'K_a_1', 'K_ai_1', 'T_r_1', 'V_min_1', 'V_max_1', 'K_aw_1', 'Droop_1', 'T_gov_1_1', 'T_gov_2_1', 'T_gov_3_1', 'K_imw_1', 'omega_ref_1', 'T_wo_1', 'T_1_1', 'T_2_1', 'K_stab_1', 'V_lim_1', 'S_n_2', 'Omega_b_2', 'H_2', 'T1d0_2', 'T1q0_2', 'X_d_2', 'X_q_2', 'X1d_2', 'X1q_2', 'D_2', 'R_a_2', 'K_delta_2', 'K_sec_2', 'K_a_2', 'K_ai_2', 'T_r_2', 'V_min_2', 'V_max_2', 'K_aw_2', 'Droop_2', 'T_gov_1_2', 'T_gov_2_2', 'T_gov_3_2', 'K_imw_2', 'omega_ref_2', 'T_wo_2', 'T_1_2', 'T_2_2', 'K_stab_2', 'V_lim_2', 'S_n_3', 'Omega_b_3', 'H_3', 'T1d0_3', 'T1q0_3', 'X_d_3', 'X_q_3', 'X1d_3', 'X1q_3', 'D_3', 'R_a_3', 'K_delta_3', 'K_sec_3', 'K_a_3', 'K_ai_3', 'T_r_3', 'V_min_3', 'V_max_3', 'K_aw_3', 'Droop_3', 'T_gov_1_3', 'T_gov_2_3', 'T_gov_3_3', 'K_imw_3', 'omega_ref_3', 'T_wo_3', 'T_1_3', 'T_2_3', 'K_stab_3', 'V_lim_3', 'S_n_4', 'Omega_b_4', 'H_4', 'T1d0_4', 'T1q0_4', 'X_d_4', 'X_q_4', 'X1d_4', 'X1q_4', 'D_4', 'R_a_4', 'K_delta_4', 'K_sec_4', 'K_a_4', 'K_ai_4', 'T_r_4', 'V_min_4', 'V_max_4', 'K_aw_4', 'Droop_4', 'T_gov_1_4', 'T_gov_2_4', 'T_gov_3_4', 'K_imw_4', 'omega_ref_4', 'T_wo_4', 'T_1_4', 'T_2_4', 'K_stab_4', 'V_lim_4', 'K_p_agc', 'K_i_agc'] 
        self.params_values_list  = [100000000.0, 0.0, -60.0, 0.0, 0.0, -60.0, 0.0, 0.0, -60.0, 0.0, 0.0, -60.0, 0.0, 3.96039603960396, -39.603960396039604, 0.027772499999999995, 9.900990099009901, -99.00990099009901, 0.011108999999999999, 0.9000900090008999, -9.000900090009, 0.12219899999999999, 0.9000900090008999, -9.000900090009, 0.12219899999999999, 9.900990099009901, -99.00990099009901, 0.011108999999999999, 3.96039603960396, -39.603960396039604, 0.027772499999999995, 20000.0, 20000.0, 20000.0, 20000.0, 230000.0, 230000.0, 230000.0, 230000.0, 230000.0, 230000.0, 230000.0, 900000000.0, 314.1592653589793, 6.5, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.001, 0.0, 300, 1e-06, 0.02, -10000.0, 5.0, 10, 0.05, 1.0, 2.0, 10.0, 0.01, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 900000000.0, 314.1592653589793, 6.5, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.0, 0.0, 300, 1e-06, 0.02, -10000.0, 5.0, 10, 0.05, 1.0, 2.0, 10.0, 0.01, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 900000000.0, 314.1592653589793, 6.175, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.0, 0.01, 300, 1e-06, 0.02, -10000.0, 5.0, 10, 0.05, 1.0, 2.0, 10.0, 0.0, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 900000000.0, 314.1592653589793, 6.175, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.0, 0.0, 300, 1e-06, 0.02, -10000.0, 5.0, 10, 0.05, 1.0, 2.0, 10.0, 0.01, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 0.01, 0.01] 
        self.inputs_ini_list = ['P_1', 'Q_1', 'P_2', 'Q_2', 'P_3', 'Q_3', 'P_4', 'Q_4', 'P_5', 'Q_5', 'P_6', 'Q_6', 'P_7', 'Q_7', 'P_8', 'Q_8', 'P_9', 'Q_9', 'P_10', 'Q_10', 'P_11', 'Q_11', 'v_ref_1', 'v_pss_1', 'p_c_1', 'p_r_1', 'v_ref_2', 'v_pss_2', 'p_c_2', 'p_r_2', 'v_ref_3', 'v_pss_3', 'p_c_3', 'p_r_3', 'v_ref_4', 'v_pss_4', 'p_c_4', 'p_r_4'] 
        self.inputs_ini_values_list  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -967000000.0, 100000000.0, 0.0, 0.0, -1767000000.0, 250000000.0, 0.0, 0.0, 0.0, 0.0, 1.03, 0.0, 0.778, 0.0, 1.01, 0.0, 0.778, 0.0, 1.03, 0.0, 0.778, 0.0, 1.01, 0.0, 0.778, 0.0] 
        self.inputs_run_list = ['P_1', 'Q_1', 'P_2', 'Q_2', 'P_3', 'Q_3', 'P_4', 'Q_4', 'P_5', 'Q_5', 'P_6', 'Q_6', 'P_7', 'Q_7', 'P_8', 'Q_8', 'P_9', 'Q_9', 'P_10', 'Q_10', 'P_11', 'Q_11', 'v_ref_1', 'v_pss_1', 'p_c_1', 'p_r_1', 'v_ref_2', 'v_pss_2', 'p_c_2', 'p_r_2', 'v_ref_3', 'v_pss_3', 'p_c_3', 'p_r_3', 'v_ref_4', 'v_pss_4', 'p_c_4', 'p_r_4'] 
        self.inputs_run_values_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -967000000.0, 100000000.0, 0.0, 0.0, -1767000000.0, 250000000.0, 0.0, 0.0, 0.0, 0.0, 1.03, 0.0, 0.778, 0.0, 1.01, 0.0, 0.778, 0.0, 1.03, 0.0, 0.778, 0.0, 1.01, 0.0, 0.778, 0.0] 
        self.outputs_list = ['V_1', 'V_2', 'V_3', 'V_4', 'V_5', 'V_6', 'V_7', 'V_8', 'V_9', 'V_10', 'V_11', 'p_e_1', 'p_e_2', 'p_e_3', 'p_e_4'] 
        self.x_list = ['delta_1', 'omega_1', 'e1q_1', 'e1d_1', 'v_c_1', 'xi_v_1', 'x_gov_1_1', 'x_gov_2_1', 'xi_imw_1', 'x_wo_1', 'x_lead_1', 'delta_2', 'omega_2', 'e1q_2', 'e1d_2', 'v_c_2', 'xi_v_2', 'x_gov_1_2', 'x_gov_2_2', 'xi_imw_2', 'x_wo_2', 'x_lead_2', 'delta_3', 'omega_3', 'e1q_3', 'e1d_3', 'v_c_3', 'xi_v_3', 'x_gov_1_3', 'x_gov_2_3', 'xi_imw_3', 'x_wo_3', 'x_lead_3', 'delta_4', 'omega_4', 'e1q_4', 'e1d_4', 'v_c_4', 'xi_v_4', 'x_gov_1_4', 'x_gov_2_4', 'xi_imw_4', 'x_wo_4', 'x_lead_4', 'xi_freq'] 
        self.y_run_list = ['V_1', 'theta_1', 'V_2', 'theta_2', 'V_3', 'theta_3', 'V_4', 'theta_4', 'V_5', 'theta_5', 'V_6', 'theta_6', 'V_7', 'theta_7', 'V_8', 'theta_8', 'V_9', 'theta_9', 'V_10', 'theta_10', 'V_11', 'theta_11', 'i_d_1', 'i_q_1', 'p_g_1', 'q_g_1', 'v_f_1', 'p_m_ref_1', 'p_m_1', 'z_wo_1', 'v_pss_1', 'i_d_2', 'i_q_2', 'p_g_2', 'q_g_2', 'v_f_2', 'p_m_ref_2', 'p_m_2', 'z_wo_2', 'v_pss_2', 'i_d_3', 'i_q_3', 'p_g_3', 'q_g_3', 'v_f_3', 'p_m_ref_3', 'p_m_3', 'z_wo_3', 'v_pss_3', 'i_d_4', 'i_q_4', 'p_g_4', 'q_g_4', 'v_f_4', 'p_m_ref_4', 'p_m_4', 'z_wo_4', 'v_pss_4', 'omega_coi', 'p_agc'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['V_1', 'theta_1', 'V_2', 'theta_2', 'V_3', 'theta_3', 'V_4', 'theta_4', 'V_5', 'theta_5', 'V_6', 'theta_6', 'V_7', 'theta_7', 'V_8', 'theta_8', 'V_9', 'theta_9', 'V_10', 'theta_10', 'V_11', 'theta_11', 'i_d_1', 'i_q_1', 'p_g_1', 'q_g_1', 'v_f_1', 'p_m_ref_1', 'p_m_1', 'z_wo_1', 'v_pss_1', 'i_d_2', 'i_q_2', 'p_g_2', 'q_g_2', 'v_f_2', 'p_m_ref_2', 'p_m_2', 'z_wo_2', 'v_pss_2', 'i_d_3', 'i_q_3', 'p_g_3', 'q_g_3', 'v_f_3', 'p_m_ref_3', 'p_m_3', 'z_wo_3', 'v_pss_3', 'i_d_4', 'i_q_4', 'p_g_4', 'q_g_4', 'v_f_4', 'p_m_ref_4', 'p_m_4', 'z_wo_4', 'v_pss_4', 'omega_coi', 'p_agc'] 
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
        
        self.jac_ini = np.zeros((self.N_x+self.N_y,self.N_x+self.N_y))
        self.jac_run = np.zeros((self.N_x+self.N_y,self.N_x+self.N_y))
        self.jac_trap = np.zeros((self.N_x+self.N_y,self.N_x+self.N_y))
        
        self.yini2urun = list(set(self.u_run_list).intersection(set(self.y_ini_list)))
        self.uini2yrun = list(set(self.y_run_list).intersection(set(self.u_ini_list)))
        self.Time = np.zeros(self.N_store)
        self.X = np.zeros((self.N_store,self.N_x))
        self.Y = np.zeros((self.N_store,self.N_y))
        self.Z = np.zeros((self.N_store,self.N_z))
        self.iters = np.zeros(self.N_store) 
        self.u_run = np.array(self.u_run_values_list)
        
        self.sp_jac_trap_ia, self.sp_jac_trap_ja, self.sp_jac_trap_nia, self.sp_jac_trap_nja = sp_jac_trap_vectors()
        data = np.array(self.sp_jac_trap_ia,dtype=np.float64)
        self.sp_jac_trap = sspa.csr_matrix((data, self.sp_jac_trap_ia, self.sp_jac_trap_ja), shape=(self.sp_jac_trap_nia,self.sp_jac_trap_nja))

        self.J_run_d = np.array(self.sp_jac_trap_ia)*0.0
        self.J_run_i = np.array(self.sp_jac_trap_ia)
        self.J_run_p = np.array(self.sp_jac_trap_ja)
        
        self.J_trap_d = np.array(self.sp_jac_trap_ia)*0.0
        self.J_trap_i = np.array(self.sp_jac_trap_ia)
        self.J_trap_p = np.array(self.sp_jac_trap_ja)
        
        self.sp_jac_ini_ia, self.sp_jac_ini_ja, self.sp_jac_ini_nia, self.sp_jac_ini_nja = sp_jac_ini_vectors()
        data = np.array(self.sp_jac_ini_ia,dtype=np.float64)
        self.sp_jac_ini = sspa.csr_matrix((data, self.sp_jac_ini_ia, self.sp_jac_ini_ja), shape=(self.sp_jac_ini_nia,self.sp_jac_ini_nja))

        self.J_ini_d = np.array(self.sp_jac_ini_ia)*0.0
        self.J_ini_i = np.array(self.sp_jac_ini_ia)
        self.J_ini_p = np.array(self.sp_jac_ini_ja)
        

        
        self.max_it,self.itol,self.store = 50,1e-8,1 
        self.lmax_it,self.ltol,self.ldamp=50,1e-8,1.1
        self.mode = 0 

        self.lmax_it_ini,self.ltol_ini,self.ldamp_ini=50,1e-8,1.1

        self.fill_factor_ini,self.drop_tol_ini,self.drop_rule_ini = 10,0.001,'column'       
        self.fill_factor_run,self.drop_tol_run,self.drop_rule_run = 10,0.001,'column' 
        
        # numerical elements of jacobians computing:
        x = self.xy[:self.N_x]
        y = self.xy[self.N_x:]
        
        de_jac_ini_eval(self.jac_ini,x,y,self.u_ini,self.p,self.Dt)
        de_jac_trap_eval(self.jac_trap,x,y,self.u_run,self.p,self.Dt)
   
        sp_jac_ini_eval(self.J_ini_d,x,y,self.u_ini,self.p,self.Dt)
        sp_jac_trap_eval(self.J_trap_d,x,y,self.u_run,self.p,self.Dt)
   


        
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
                                  max_it=50,itol=1e-8,store=1)
        
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
            value = self.x[idx]
            return value
            
        if name in self.y_run_list:
            idy = self.y_run_list.index(name)
            value = self.y_run[idy]
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
                
        self.xy_ini = self.ss_ini()
        self.ini2run()
        #jac_run_ss_eval_xy(self.jac_run,self.x,self.y_run,self.u_run,self.p)
        #jac_run_ss_eval_up(self.jac_run,self.x,self.y_run,self.u_run,self.p)
    
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
    
        sp_jac_trap_eval(self.sp_jac_ini.data,self.x,self.y_run,self.u_run,self.p,self.Dt)
    
        P_slu = spilu(self.sp_jac_ini,
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
            
    
    def eval_preconditioner_run(self):
    
        sp_jac_trap_eval(self.J_run_d,self.x,self.y_run,self.u_run,self.p,self.Dt)
    
        self.sp_jac_trap.data = self.J_run_d 
        P_slu_run = spilu(self.sp_jac_trap,
                          fill_factor=self.fill_factor_run,
                          drop_tol=self.drop_tol_run,
                          drop_rule = self.drop_rule_run)
    
        self.P_slu_run = P_slu_run
        P_d,P_i,P_p,perm_r,perm_c = slu2pydae(P_slu_run)   
        self.P_run_d = P_d
        self.P_run_i = P_i
        self.P_run_p = P_p
    
        self.perm_run_r = perm_r
        self.perm_run_c = perm_c
        
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
                                  self.J_run_d,self.J_run_i,self.J_run_p,
                                  self.P_run_d,self.P_run_i,self.P_run_p,self.perm_run_r,self.perm_run_c,
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
    sp_jac_trap_up_eval(J_d_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    sp_jac_trap_xy_eval(J_d_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    
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
    P_slu_full = P_slu.L.A - sspa.eye(N,format='csr') + P_slu.U.A
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

    sp_jac_ini_ia = [0, 1, 103, 0, 1, 45, 46, 67, 68, 73, 103, 2, 67, 71, 3, 68, 4, 45, 4, 5, 71, 75, 6, 72, 6, 7, 8, 69, 1, 9, 10, 74, 11, 12, 103, 11, 12, 47, 48, 76, 77, 82, 103, 13, 76, 80, 14, 77, 15, 47, 15, 16, 80, 84, 17, 81, 17, 18, 19, 78, 12, 20, 21, 83, 22, 23, 103, 22, 23, 49, 50, 85, 86, 91, 103, 24, 85, 89, 25, 86, 26, 49, 26, 27, 89, 93, 28, 90, 28, 29, 30, 87, 23, 31, 32, 92, 33, 34, 103, 33, 34, 51, 52, 94, 95, 100, 103, 35, 94, 98, 36, 95, 37, 51, 37, 38, 98, 102, 39, 99, 39, 40, 41, 96, 34, 42, 43, 101, 103, 45, 46, 53, 54, 69, 45, 46, 53, 54, 70, 47, 48, 55, 56, 78, 47, 48, 55, 56, 79, 49, 50, 65, 66, 87, 49, 50, 65, 66, 88, 51, 52, 63, 64, 96, 51, 52, 63, 64, 97, 45, 46, 53, 54, 55, 56, 45, 46, 53, 54, 55, 56, 47, 48, 53, 54, 55, 56, 57, 58, 47, 48, 53, 54, 55, 56, 57, 58, 55, 56, 57, 58, 59, 60, 55, 56, 57, 58, 59, 60, 57, 58, 59, 60, 61, 62, 57, 58, 59, 60, 61, 62, 59, 60, 61, 62, 63, 64, 59, 60, 61, 62, 63, 64, 51, 52, 61, 62, 63, 64, 65, 66, 51, 52, 61, 62, 63, 64, 65, 66, 49, 50, 63, 64, 65, 66, 49, 50, 63, 64, 65, 66, 0, 2, 45, 46, 67, 68, 0, 3, 45, 46, 67, 68, 0, 45, 46, 67, 68, 69, 0, 45, 46, 67, 68, 70, 4, 5, 71, 75, 1, 8, 72, 104, 6, 7, 73, 1, 9, 74, 10, 74, 75, 11, 13, 47, 48, 76, 77, 11, 14, 47, 48, 76, 77, 11, 47, 48, 76, 77, 78, 11, 47, 48, 76, 77, 79, 15, 16, 80, 84, 12, 19, 81, 104, 17, 18, 82, 12, 20, 83, 21, 83, 84, 22, 24, 49, 50, 85, 86, 22, 25, 49, 50, 85, 86, 22, 49, 50, 85, 86, 87, 22, 49, 50, 85, 86, 88, 26, 27, 89, 93, 23, 30, 90, 104, 28, 29, 91, 23, 31, 92, 32, 92, 93, 33, 35, 51, 52, 94, 95, 33, 36, 51, 52, 94, 95, 33, 51, 52, 94, 95, 96, 33, 51, 52, 94, 95, 97, 37, 38, 98, 102, 34, 41, 99, 104, 39, 40, 100, 34, 42, 101, 43, 101, 102, 1, 12, 23, 34, 103, 44, 103, 104]
    sp_jac_ini_ja = [0, 3, 11, 14, 16, 18, 22, 24, 26, 28, 30, 32, 35, 43, 46, 48, 50, 54, 56, 58, 60, 62, 64, 67, 75, 78, 80, 82, 86, 88, 90, 92, 94, 96, 99, 107, 110, 112, 114, 118, 120, 122, 124, 126, 128, 129, 134, 139, 144, 149, 154, 159, 164, 169, 175, 181, 189, 197, 203, 209, 215, 221, 227, 233, 241, 249, 255, 261, 267, 273, 279, 285, 289, 293, 296, 299, 302, 308, 314, 320, 326, 330, 334, 337, 340, 343, 349, 355, 361, 367, 371, 375, 378, 381, 384, 390, 396, 402, 408, 412, 416, 419, 422, 425, 430, 433]
    sp_jac_ini_nia = 105
    sp_jac_ini_nja = 105
    return sp_jac_ini_ia, sp_jac_ini_ja, sp_jac_ini_nia, sp_jac_ini_nja 

def sp_jac_trap_vectors():

    sp_jac_trap_ia = [0, 1, 103, 0, 1, 45, 46, 67, 68, 73, 103, 2, 67, 71, 3, 68, 4, 45, 4, 5, 71, 75, 6, 72, 6, 7, 8, 69, 1, 9, 10, 74, 11, 12, 103, 11, 12, 47, 48, 76, 77, 82, 103, 13, 76, 80, 14, 77, 15, 47, 15, 16, 80, 84, 17, 81, 17, 18, 19, 78, 12, 20, 21, 83, 22, 23, 103, 22, 23, 49, 50, 85, 86, 91, 103, 24, 85, 89, 25, 86, 26, 49, 26, 27, 89, 93, 28, 90, 28, 29, 30, 87, 23, 31, 32, 92, 33, 34, 103, 33, 34, 51, 52, 94, 95, 100, 103, 35, 94, 98, 36, 95, 37, 51, 37, 38, 98, 102, 39, 99, 39, 40, 41, 96, 34, 42, 43, 101, 44, 103, 45, 46, 53, 54, 69, 45, 46, 53, 54, 70, 47, 48, 55, 56, 78, 47, 48, 55, 56, 79, 49, 50, 65, 66, 87, 49, 50, 65, 66, 88, 51, 52, 63, 64, 96, 51, 52, 63, 64, 97, 45, 46, 53, 54, 55, 56, 45, 46, 53, 54, 55, 56, 47, 48, 53, 54, 55, 56, 57, 58, 47, 48, 53, 54, 55, 56, 57, 58, 55, 56, 57, 58, 59, 60, 55, 56, 57, 58, 59, 60, 57, 58, 59, 60, 61, 62, 57, 58, 59, 60, 61, 62, 59, 60, 61, 62, 63, 64, 59, 60, 61, 62, 63, 64, 51, 52, 61, 62, 63, 64, 65, 66, 51, 52, 61, 62, 63, 64, 65, 66, 49, 50, 63, 64, 65, 66, 49, 50, 63, 64, 65, 66, 0, 2, 45, 46, 67, 68, 0, 3, 45, 46, 67, 68, 0, 45, 46, 67, 68, 69, 0, 45, 46, 67, 68, 70, 4, 5, 71, 75, 1, 8, 72, 104, 6, 7, 73, 1, 9, 74, 10, 74, 75, 11, 13, 47, 48, 76, 77, 11, 14, 47, 48, 76, 77, 11, 47, 48, 76, 77, 78, 11, 47, 48, 76, 77, 79, 15, 16, 80, 84, 12, 19, 81, 104, 17, 18, 82, 12, 20, 83, 21, 83, 84, 22, 24, 49, 50, 85, 86, 22, 25, 49, 50, 85, 86, 22, 49, 50, 85, 86, 87, 22, 49, 50, 85, 86, 88, 26, 27, 89, 93, 23, 30, 90, 104, 28, 29, 91, 23, 31, 92, 32, 92, 93, 33, 35, 51, 52, 94, 95, 33, 36, 51, 52, 94, 95, 33, 51, 52, 94, 95, 96, 33, 51, 52, 94, 95, 97, 37, 38, 98, 102, 34, 41, 99, 104, 39, 40, 100, 34, 42, 101, 43, 101, 102, 1, 12, 23, 34, 103, 44, 103, 104]
    sp_jac_trap_ja = [0, 3, 11, 14, 16, 18, 22, 24, 26, 28, 30, 32, 35, 43, 46, 48, 50, 54, 56, 58, 60, 62, 64, 67, 75, 78, 80, 82, 86, 88, 90, 92, 94, 96, 99, 107, 110, 112, 114, 118, 120, 122, 124, 126, 128, 130, 135, 140, 145, 150, 155, 160, 165, 170, 176, 182, 190, 198, 204, 210, 216, 222, 228, 234, 242, 250, 256, 262, 268, 274, 280, 286, 290, 294, 297, 300, 303, 309, 315, 321, 327, 331, 335, 338, 341, 344, 350, 356, 362, 368, 372, 376, 379, 382, 385, 391, 397, 403, 409, 413, 417, 420, 423, 426, 431, 434]
    sp_jac_trap_nia = 105
    sp_jac_trap_nja = 105
    return sp_jac_trap_ia, sp_jac_trap_ja, sp_jac_trap_nia, sp_jac_trap_nja 
