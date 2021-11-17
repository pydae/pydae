import numpy as np
import numba
import scipy.optimize as sopt
import scipy.sparse as sspa
from scipy.sparse.linalg import spsolve,spilu,splu
from numba import cuda


import json

sin = np.sin
cos = np.cos
atan2 = np.arctan2
sqrt = np.sqrt 
sign = np.sign 
exp = np.exp


class k12p6_pss_class: 

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
        y = self.xy[:self.N_x]
        
        jac_ini_ss_eval_num(self.jac_ini,x,y,self.u_ini,self.p)
        jac_run_ss_eval_num(self.jac_run,x,y,self.u_run,self.p)
        jac_trap_eval_num(self.jac_trap,x,y,self.u_run,self.p,self.Dt)


        
        sp_jac_ini_eval_num(self.sp_jac_ini.data,x,y,self.u_ini,self.p,self.Dt)
        sp_jac_trap_eval_num(self.J_run_d,x,y,self.u_run,self.p,self.Dt)


        
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
        h_eval(self.z,self.x,self.y_run,self.u_ini,self.p)
        

        
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
        jac_run_ss_eval_xy(self.jac_run,self.x,self.y_run,self.u_run,self.p)
        jac_run_ss_eval_up(self.jac_run,self.x,self.y_run,self.u_run,self.p)
    
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
        
        sp_jac_ini_eval_up(self.sp_jac_ini.data,
                        self.x,self.y_ini,self.u_ini,self.p,self.Dt)
        sp_jac_ini_eval_xy(self.sp_jac_ini.data,
                        self.x,self.y_ini,self.u_ini,self.p,self.Dt)
        
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
        
        #print(f'Preconditioner Nnz = {P_slu.nnz}')
        
    def eval_preconditioner_run(self):
    
        sp_jac_trap_eval_up(self.J_run_d,self.x,self.y_ini,self.u_ini,self.p,self.Dt,xyup = 1)
        sp_jac_trap_eval_xy(self.J_run_d,self.x,self.y_ini,self.u_ini,self.p,self.Dt,xyup = 1)

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
        jac_run_ss_eval_xy(self.jac_run,self.x,self.y_run,self.u_run,self.p)
        jac_run_ss_eval_up(self.jac_run,self.x,self.y_run,self.u_run,self.p)

        
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



@numba.njit(cache=True)
def sstate(xy,u,p,jac_ini_ss,N_x,N_y,max_it=50,tol=1e-8):
    
    fg = np.zeros((N_x+N_y,1),dtype=np.float64)
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    
    jac_ini_ss_eval_up(jac_ini_ss,x,y,u,p)


    for it in range(max_it):
        jac_ini_ss_eval_xy(jac_ini_ss,x,y,u,p)
        f_ini_eval(f,x,y,u,p)
        g_ini_eval(g,x,y,u,p)
        fg[:N_x] = f
        fg[N_x:] = g
        xy += np.linalg.solve(jac_ini_ss,-fg)
        if np.max(np.abs(fg))<tol: break

    return xy,it

            
            
@numba.njit(cache=True) 
def daesolver(t,t_end,it,it_store,xy,u,p,jac_trap,T,X,Y,Z,iters,Dt,N_x,N_y,N_z,decimation,max_it=50,itol=1e-8,store=1): 


    fg = np.zeros((N_x+N_y,1),dtype=np.float64)
    fg_i = np.zeros((N_x+N_y),dtype=np.float64)
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    h = np.zeros((N_z),dtype=np.float64)
    jac_trap_eval_up(jac_trap,x,y,u,p,Dt,xyup=1) 
    jac_trap_eval_xy(jac_trap,x,y,u,p,Dt,xyup=1) 
    
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
            jac_trap_eval_xy(jac_trap,x,y,u,p,Dt,xyup=0)             

            f_n_i = x - x_0 - 0.5*Dt*(f+f_0) 

            fg_i[:N_x] = f_n_i
            fg_i[N_x:] = g
            
            Dxy_i = np.linalg.solve(-jac_trap,fg_i) 

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

@numba.njit(cache=True)
def spdaesolver(t,t_end,it,it_store,xy,u,p,jac_trap,
                J_d,J_i,J_p,
                P_d,P_i,P_p,perm_r,perm_c,
                T,X,Y,Z,iters,Dt,N_x,N_y,N_z,decimation,max_it=50,itol=1e-8,store=1,
                lmax_it=20,ltol=1e-4,ldamp=1.0,mode=0):


    fg = np.zeros((N_x+N_y,1),dtype=np.float64)
    fg_i = np.zeros((N_x+N_y),dtype=np.float64)
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    h = np.zeros((N_z),dtype=np.float64)
    jac_trap_eval_up(jac_trap,x,y,u,p,Dt,xyup=1) 
    jac_trap_eval_xy(jac_trap,x,y,u,p,Dt,xyup=1) 
    iparams = np.array([0],dtype=np.int64)    

    
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
            
        Dxy_i = xy*0.0
        for iti in range(max_it):
            f_run_eval(f,x,y,u,p)
            g_run_eval(g,x,y,u,p)
            
            if mode == 0:
                jac_trap_eval_xy(jac_trap,x,y,u,p,Dt,xyup=0)   
            if mode == 1:
                sp_jac_trap_eval_xy(J_d,x,y,u,p,Dt,xyup=0) 

            f_n_i = x - x_0 - 0.5*Dt*(f+f_0) 

            fg_i[:N_x] = f_n_i
            fg_i[N_x:] = g
            
            if mode == 0:
                Dxy_i = np.linalg.solve(-jac_trap,fg_i) 
            if mode == 1:
                Dxy_i = sprichardson(J_d,J_i,J_p,-fg_i,P_d,P_i,P_p,perm_r,perm_c,0*Dxy_i,iparams,damp=ldamp,max_it=lmax_it,tol=ltol)

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
    sp_jac_ini_eval_up(J_d,x,y,u,p,0.0)

    Dxy = np.zeros(N_x + N_y)
    for it in range(max_it):
        
        x = xy[:N_x]
        y = xy[N_x:]   
       
        sp_jac_ini_eval_xy(J_d,x,y,u,p,0.0)

        f_ini_eval(f,x,y,u,p)
        g_ini_eval(g,x,y,u,p)
        
        fg[:N_x] = f
        fg[N_x:] = g
               
        Dxy = sprichardson(J_d,J_i,J_p,-fg,P_d,P_i,P_p,perm_r,perm_c,Dxy,iparams,damp=ldamp,max_it=lmax_it,tol=ltol)
   
        xy += Dxy
        #if np.max(np.abs(fg))<tol: break
        if np.linalg.norm(fg,np.inf)<tol: break

    return xy,it,iparams


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



@numba.njit(cache=True)
def f_ini_eval(f_ini,x,y,u,p,xyup = 0):


    f_ini[0] = -p[53]*x[0] + p[43]*(x[1] - y[58])
    f_ini[1] = (-p[51]*(x[1] - y[58]) - y[22]*(p[52]*y[22] + y[0]*sin(x[0] - y[1])) - y[23]*(p[52]*y[23] + y[0]*cos(x[0] - y[1])) + y[28])/(2*p[44])
    f_ini[2] = (-x[2] - y[22]*(-p[49] + p[47]) + y[26])/p[45]
    f_ini[3] = (-x[3] + y[23]*(-p[50] + p[48]))/p[46]
    f_ini[4] = (y[0] - x[4])/p[57]
    f_ini[5] = -p[60]*(p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5] - y[26]) - x[4] + y[30] + u[22]
    f_ini[6] = (y[27] - x[6])/p[62]
    f_ini[7] = (x[6] - x[7])/p[64]
    f_ini[8] = p[65]*(u[24] - y[24]) - 1.0e-6*x[8]
    f_ini[9] = (x[1] - x[9] - 1.0)/p[67]
    f_ini[10] = (-x[10] + y[29])/p[69]
    f_ini[11] = -p[83]*x[11] + p[73]*(x[12] - y[58])
    f_ini[12] = (-p[81]*(x[12] - y[58]) - y[31]*(p[82]*y[31] + y[2]*sin(x[11] - y[3])) - y[32]*(p[82]*y[32] + y[2]*cos(x[11] - y[3])) + y[37])/(2*p[74])
    f_ini[13] = (-x[13] - y[31]*(-p[79] + p[77]) + y[35])/p[75]
    f_ini[14] = (-x[14] + y[32]*(-p[80] + p[78]))/p[76]
    f_ini[15] = (y[2] - x[15])/p[87]
    f_ini[16] = -p[90]*(p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16] - y[35]) - x[15] + y[39] + u[26]
    f_ini[17] = (y[36] - x[17])/p[92]
    f_ini[18] = (x[17] - x[18])/p[94]
    f_ini[19] = p[95]*(u[28] - y[33]) - 1.0e-6*x[19]
    f_ini[20] = (x[12] - x[20] - 1.0)/p[97]
    f_ini[21] = (-x[21] + y[38])/p[99]
    f_ini[22] = -p[113]*x[22] + p[103]*(x[23] - y[58])
    f_ini[23] = (-p[111]*(x[23] - y[58]) - y[40]*(p[112]*y[40] + y[4]*sin(x[22] - y[5])) - y[41]*(p[112]*y[41] + y[4]*cos(x[22] - y[5])) + y[46])/(2*p[104])
    f_ini[24] = (-x[24] - y[40]*(-p[109] + p[107]) + y[44])/p[105]
    f_ini[25] = (-x[25] + y[41]*(-p[110] + p[108]))/p[106]
    f_ini[26] = (y[4] - x[26])/p[117]
    f_ini[27] = -p[120]*(p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27] - y[44]) - x[26] + y[48] + u[30]
    f_ini[28] = (y[45] - x[28])/p[122]
    f_ini[29] = (x[28] - x[29])/p[124]
    f_ini[30] = p[125]*(u[32] - y[42]) - 1.0e-6*x[30]
    f_ini[31] = (x[23] - x[31] - 1.0)/p[127]
    f_ini[32] = (-x[32] + y[47])/p[129]
    f_ini[33] = -p[143]*x[33] + p[133]*(x[34] - y[58])
    f_ini[34] = (-p[141]*(x[34] - y[58]) - y[49]*(p[142]*y[49] + y[6]*sin(x[33] - y[7])) - y[50]*(p[142]*y[50] + y[6]*cos(x[33] - y[7])) + y[55])/(2*p[134])
    f_ini[35] = (-x[35] - y[49]*(-p[139] + p[137]) + y[53])/p[135]
    f_ini[36] = (-x[36] + y[50]*(-p[140] + p[138]))/p[136]
    f_ini[37] = (y[6] - x[37])/p[147]
    f_ini[38] = -p[150]*(p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38] - y[53]) - x[37] + y[57] + u[34]
    f_ini[39] = (y[54] - x[39])/p[152]
    f_ini[40] = (x[39] - x[40])/p[154]
    f_ini[41] = p[155]*(u[36] - y[51]) - 1.0e-6*x[41]
    f_ini[42] = (x[34] - x[42] - 1.0)/p[157]
    f_ini[43] = (-x[43] + y[56])/p[159]
    f_ini[44] = 1 - y[58]





@numba.njit(cache=True)
def f_run_eval(f_run,x,y,u,p,xyup = 0):


    f_run[0] = -p[53]*x[0] + p[43]*(x[1] - y[58])
    f_run[1] = (-p[51]*(x[1] - y[58]) - y[22]*(p[52]*y[22] + y[0]*sin(x[0] - y[1])) - y[23]*(p[52]*y[23] + y[0]*cos(x[0] - y[1])) + y[28])/(2*p[44])
    f_run[2] = (-x[2] - y[22]*(-p[49] + p[47]) + y[26])/p[45]
    f_run[3] = (-x[3] + y[23]*(-p[50] + p[48]))/p[46]
    f_run[4] = (y[0] - x[4])/p[57]
    f_run[5] = -p[60]*(p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5] - y[26]) - x[4] + y[30] + u[22]
    f_run[6] = (y[27] - x[6])/p[62]
    f_run[7] = (x[6] - x[7])/p[64]
    f_run[8] = p[65]*(u[24] - y[24]) - 1.0e-6*x[8]
    f_run[9] = (x[1] - x[9] - 1.0)/p[67]
    f_run[10] = (-x[10] + y[29])/p[69]
    f_run[11] = -p[83]*x[11] + p[73]*(x[12] - y[58])
    f_run[12] = (-p[81]*(x[12] - y[58]) - y[31]*(p[82]*y[31] + y[2]*sin(x[11] - y[3])) - y[32]*(p[82]*y[32] + y[2]*cos(x[11] - y[3])) + y[37])/(2*p[74])
    f_run[13] = (-x[13] - y[31]*(-p[79] + p[77]) + y[35])/p[75]
    f_run[14] = (-x[14] + y[32]*(-p[80] + p[78]))/p[76]
    f_run[15] = (y[2] - x[15])/p[87]
    f_run[16] = -p[90]*(p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16] - y[35]) - x[15] + y[39] + u[26]
    f_run[17] = (y[36] - x[17])/p[92]
    f_run[18] = (x[17] - x[18])/p[94]
    f_run[19] = p[95]*(u[28] - y[33]) - 1.0e-6*x[19]
    f_run[20] = (x[12] - x[20] - 1.0)/p[97]
    f_run[21] = (-x[21] + y[38])/p[99]
    f_run[22] = -p[113]*x[22] + p[103]*(x[23] - y[58])
    f_run[23] = (-p[111]*(x[23] - y[58]) - y[40]*(p[112]*y[40] + y[4]*sin(x[22] - y[5])) - y[41]*(p[112]*y[41] + y[4]*cos(x[22] - y[5])) + y[46])/(2*p[104])
    f_run[24] = (-x[24] - y[40]*(-p[109] + p[107]) + y[44])/p[105]
    f_run[25] = (-x[25] + y[41]*(-p[110] + p[108]))/p[106]
    f_run[26] = (y[4] - x[26])/p[117]
    f_run[27] = -p[120]*(p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27] - y[44]) - x[26] + y[48] + u[30]
    f_run[28] = (y[45] - x[28])/p[122]
    f_run[29] = (x[28] - x[29])/p[124]
    f_run[30] = p[125]*(u[32] - y[42]) - 1.0e-6*x[30]
    f_run[31] = (x[23] - x[31] - 1.0)/p[127]
    f_run[32] = (-x[32] + y[47])/p[129]
    f_run[33] = -p[143]*x[33] + p[133]*(x[34] - y[58])
    f_run[34] = (-p[141]*(x[34] - y[58]) - y[49]*(p[142]*y[49] + y[6]*sin(x[33] - y[7])) - y[50]*(p[142]*y[50] + y[6]*cos(x[33] - y[7])) + y[55])/(2*p[134])
    f_run[35] = (-x[35] - y[49]*(-p[139] + p[137]) + y[53])/p[135]
    f_run[36] = (-x[36] + y[50]*(-p[140] + p[138]))/p[136]
    f_run[37] = (y[6] - x[37])/p[147]
    f_run[38] = -p[150]*(p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38] - y[53]) - x[37] + y[57] + u[34]
    f_run[39] = (y[54] - x[39])/p[152]
    f_run[40] = (x[39] - x[40])/p[154]
    f_run[41] = p[155]*(u[36] - y[51]) - 1.0e-6*x[41]
    f_run[42] = (x[34] - x[42] - 1.0)/p[157]
    f_run[43] = (-x[43] + y[56])/p[159]
    f_run[44] = 1 - y[58]





@numba.njit(cache=True)
def g_ini_eval(g_ini,x,y,u,p,xyup = 0):


    g_ini[0] = -u[0]/p[0] + y[0]**2*p[1] + y[0]*y[8]*(-p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9])) - p[42]*y[24]/p[0]
    g_ini[1] = -u[1]/p[0] + y[0]**2*(-p[2] - p[3]/2) + y[0]*y[8]*(p[2]*cos(y[1] - y[9]) - p[1]*sin(y[1] - y[9])) - p[42]*y[25]/p[0]
    g_ini[2] = -u[2]/p[0] + y[2]**2*p[4] + y[2]*y[10]*(-p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11])) - p[72]*y[33]/p[0]
    g_ini[3] = -u[3]/p[0] + y[2]**2*(-p[5] - p[6]/2) + y[2]*y[10]*(p[5]*cos(y[3] - y[11]) - p[4]*sin(y[3] - y[11])) - p[72]*y[34]/p[0]
    g_ini[4] = -u[4]/p[0] + y[20]*y[4]*(p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5])) + y[4]**2*p[7] - p[102]*y[42]/p[0]
    g_ini[5] = -u[5]/p[0] + y[20]*y[4]*(p[8]*cos(y[21] - y[5]) + p[7]*sin(y[21] - y[5])) + y[4]**2*(-p[8] - p[9]/2) - p[102]*y[43]/p[0]
    g_ini[6] = -u[6]/p[0] + y[18]*y[6]*(p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7])) + y[6]**2*p[10] - p[132]*y[51]/p[0]
    g_ini[7] = -u[7]/p[0] + y[18]*y[6]*(p[11]*cos(y[19] - y[7]) + p[10]*sin(y[19] - y[7])) + y[6]**2*(-p[11] - p[12]/2) - p[132]*y[52]/p[0]
    g_ini[8] = -u[8]/p[0] + y[0]*y[8]*(p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9])) + y[8]**2*(p[1] + p[13]) + y[8]*y[10]*(-p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11]))
    g_ini[9] = -u[9]/p[0] + y[0]*y[8]*(p[2]*cos(y[1] - y[9]) + p[1]*sin(y[1] - y[9])) + y[8]**2*(-p[2] - p[14] - p[3]/2 - p[15]/2) + y[8]*y[10]*(p[14]*cos(y[9] - y[11]) - p[13]*sin(y[9] - y[11]))
    g_ini[10] = -u[10]/p[0] + y[2]*y[10]*(p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11])) + y[8]*y[10]*(p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11])) + y[10]**2*(p[4] + p[13] + p[16]) + y[10]*y[12]*(-p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13]))
    g_ini[11] = -u[11]/p[0] + y[2]*y[10]*(p[5]*cos(y[3] - y[11]) + p[4]*sin(y[3] - y[11])) + y[8]*y[10]*(p[14]*cos(y[9] - y[11]) + p[13]*sin(y[9] - y[11])) + y[10]**2*(-p[5] - p[14] - p[17] - p[6]/2 - p[15]/2 - p[18]/2) + y[10]*y[12]*(p[17]*cos(y[11] - y[13]) - p[16]*sin(y[11] - y[13]))
    g_ini[12] = -u[12]/p[0] + y[10]*y[12]*(p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13])) + y[12]**2*(p[16] + 2*p[19]) + y[12]*y[14]*(-2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15]))
    g_ini[13] = -u[13]/p[0] + y[10]*y[12]*(p[17]*cos(y[11] - y[13]) + p[16]*sin(y[11] - y[13])) + y[12]**2*(-p[17] - 2*p[20] - p[18]/2 - p[21]) + y[12]*y[14]*(2*p[20]*cos(y[13] - y[15]) - 2*p[19]*sin(y[13] - y[15]))
    g_ini[14] = -u[14]/p[0] + y[12]*y[14]*(2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15])) + y[14]**2*(2*p[19] + 2*p[22]) + y[14]*y[16]*(-2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17]))
    g_ini[15] = -u[15]/p[0] + y[12]*y[14]*(2*p[20]*cos(y[13] - y[15]) + 2*p[19]*sin(y[13] - y[15])) + y[14]**2*(-2*p[20] - 2*p[23] - p[21] - p[24]) + y[14]*y[16]*(2*p[23]*cos(y[15] - y[17]) - 2*p[22]*sin(y[15] - y[17]))
    g_ini[16] = -u[16]/p[0] + y[18]*y[16]*(p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17])) + y[14]*y[16]*(2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17])) + y[16]**2*(2*p[22] + p[25])
    g_ini[17] = -u[17]/p[0] + y[18]*y[16]*(p[26]*cos(y[19] - y[17]) + p[25]*sin(y[19] - y[17])) + y[14]*y[16]*(2*p[23]*cos(y[15] - y[17]) + 2*p[22]*sin(y[15] - y[17])) + y[16]**2*(-2*p[23] - p[26] - p[24] - p[27]/2)
    g_ini[18] = -u[18]/p[0] + y[18]**2*(p[28] + p[10] + p[25]) + y[18]*y[20]*(-p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21])) + y[18]*y[6]*(-p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7])) + y[18]*y[16]*(-p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17]))
    g_ini[19] = -u[19]/p[0] + y[18]**2*(-p[29] - p[11] - p[26] - p[30]/2 - p[12]/2 - p[27]/2) + y[18]*y[20]*(p[29]*cos(y[19] - y[21]) - p[28]*sin(y[19] - y[21])) + y[18]*y[6]*(p[11]*cos(y[19] - y[7]) - p[10]*sin(y[19] - y[7])) + y[18]*y[16]*(p[26]*cos(y[19] - y[17]) - p[25]*sin(y[19] - y[17]))
    g_ini[20] = -u[20]/p[0] + y[18]*y[20]*(p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21])) + y[20]**2*(p[28] + p[7]) + y[20]*y[4]*(-p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5]))
    g_ini[21] = -u[21]/p[0] + y[18]*y[20]*(p[29]*cos(y[19] - y[21]) + p[28]*sin(y[19] - y[21])) + y[20]**2*(-p[29] - p[8] - p[30]/2 - p[9]/2) + y[20]*y[4]*(p[8]*cos(y[21] - y[5]) - p[7]*sin(y[21] - y[5]))
    g_ini[22] = p[52]*y[23] + y[0]*cos(x[0] - y[1]) + p[49]*y[22] - x[2]
    g_ini[23] = p[52]*y[22] + y[0]*sin(x[0] - y[1]) - p[50]*y[23] - x[3]
    g_ini[24] = y[0]*y[22]*sin(x[0] - y[1]) + y[0]*y[23]*cos(x[0] - y[1]) - y[24]
    g_ini[25] = y[0]*y[22]*cos(x[0] - y[1]) - y[0]*y[23]*sin(x[0] - y[1]) - y[25]
    g_ini[26] = -y[26] + Piecewise(np.array([(p[58], p[58] > p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5]), (p[59], p[59] < p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5]), (p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5], True)]))
    g_ini[27] = p[54]*y[59] - y[27] + u[25] + x[8] - (x[1] - p[66])/p[61]
    g_ini[28] = p[63]*(x[6] - x[7])/p[64] - y[28] + x[7]
    g_ini[29] = x[1] - x[9] - y[29] - 1.0
    g_ini[30] = -y[30] + Piecewise(np.array([(-p[71], p[71] < -p[70]*(p[68]*(-x[10] + y[29])/p[69] + x[10])), (p[71], p[71] < p[70]*(p[68]*(-x[10] + y[29])/p[69] + x[10])), (p[70]*(p[68]*(-x[10] + y[29])/p[69] + x[10]), True)]))
    g_ini[31] = p[82]*y[32] + y[2]*cos(x[11] - y[3]) + p[79]*y[31] - x[13]
    g_ini[32] = p[82]*y[31] + y[2]*sin(x[11] - y[3]) - p[80]*y[32] - x[14]
    g_ini[33] = y[2]*y[31]*sin(x[11] - y[3]) + y[2]*y[32]*cos(x[11] - y[3]) - y[33]
    g_ini[34] = y[2]*y[31]*cos(x[11] - y[3]) - y[2]*y[32]*sin(x[11] - y[3]) - y[34]
    g_ini[35] = -y[35] + Piecewise(np.array([(p[88], p[88] > p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16]), (p[89], p[89] < p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16]), (p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16], True)]))
    g_ini[36] = p[84]*y[59] - y[36] + u[29] + x[19] - (x[12] - p[96])/p[91]
    g_ini[37] = p[93]*(x[17] - x[18])/p[94] - y[37] + x[18]
    g_ini[38] = x[12] - x[20] - y[38] - 1.0
    g_ini[39] = -y[39] + Piecewise(np.array([(-p[101], p[101] < -p[100]*(p[98]*(-x[21] + y[38])/p[99] + x[21])), (p[101], p[101] < p[100]*(p[98]*(-x[21] + y[38])/p[99] + x[21])), (p[100]*(p[98]*(-x[21] + y[38])/p[99] + x[21]), True)]))
    g_ini[40] = p[112]*y[41] + y[4]*cos(x[22] - y[5]) + p[109]*y[40] - x[24]
    g_ini[41] = p[112]*y[40] + y[4]*sin(x[22] - y[5]) - p[110]*y[41] - x[25]
    g_ini[42] = y[4]*y[40]*sin(x[22] - y[5]) + y[4]*y[41]*cos(x[22] - y[5]) - y[42]
    g_ini[43] = y[4]*y[40]*cos(x[22] - y[5]) - y[4]*y[41]*sin(x[22] - y[5]) - y[43]
    g_ini[44] = -y[44] + Piecewise(np.array([(p[118], p[118] > p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27]), (p[119], p[119] < p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27]), (p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27], True)]))
    g_ini[45] = p[114]*y[59] - y[45] + u[33] + x[30] - (x[23] - p[126])/p[121]
    g_ini[46] = p[123]*(x[28] - x[29])/p[124] - y[46] + x[29]
    g_ini[47] = x[23] - x[31] - y[47] - 1.0
    g_ini[48] = -y[48] + Piecewise(np.array([(-p[131], p[131] < -p[130]*(p[128]*(-x[32] + y[47])/p[129] + x[32])), (p[131], p[131] < p[130]*(p[128]*(-x[32] + y[47])/p[129] + x[32])), (p[130]*(p[128]*(-x[32] + y[47])/p[129] + x[32]), True)]))
    g_ini[49] = p[142]*y[50] + y[6]*cos(x[33] - y[7]) + p[139]*y[49] - x[35]
    g_ini[50] = p[142]*y[49] + y[6]*sin(x[33] - y[7]) - p[140]*y[50] - x[36]
    g_ini[51] = y[6]*y[49]*sin(x[33] - y[7]) + y[6]*y[50]*cos(x[33] - y[7]) - y[51]
    g_ini[52] = y[6]*y[49]*cos(x[33] - y[7]) - y[6]*y[50]*sin(x[33] - y[7]) - y[52]
    g_ini[53] = -y[53] + Piecewise(np.array([(p[148], p[148] > p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38]), (p[149], p[149] < p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38]), (p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38], True)]))
    g_ini[54] = p[144]*y[59] - y[54] + u[37] + x[41] - (x[34] - p[156])/p[151]
    g_ini[55] = p[153]*(x[39] - x[40])/p[154] - y[55] + x[40]
    g_ini[56] = x[34] - x[42] - y[56] - 1.0
    g_ini[57] = -y[57] + Piecewise(np.array([(-p[161], p[161] < -p[160]*(p[158]*(-x[43] + y[56])/p[159] + x[43])), (p[161], p[161] < p[160]*(p[158]*(-x[43] + y[56])/p[159] + x[43])), (p[160]*(p[158]*(-x[43] + y[56])/p[159] + x[43]), True)]))
    g_ini[58] = -y[58] + (p[44]*p[42]*x[1] + p[74]*p[72]*x[12] + p[104]*p[102]*x[23] + p[134]*p[132]*x[34])/(p[44]*p[42] + p[74]*p[72] + p[104]*p[102] + p[134]*p[132])
    g_ini[59] = p[163]*x[44] + p[162]*(1 - y[58]) - y[59]





@numba.njit(cache=True)
def g_run_eval(g_run,x,y,u,p,xyup = 1):


    g_run[0] = -u[0]/p[0] + y[0]**2*p[1] + y[0]*y[8]*(-p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9])) - p[42]*y[24]/p[0]
    g_run[1] = -u[1]/p[0] + y[0]**2*(-p[2] - p[3]/2) + y[0]*y[8]*(p[2]*cos(y[1] - y[9]) - p[1]*sin(y[1] - y[9])) - p[42]*y[25]/p[0]
    g_run[2] = -u[2]/p[0] + y[2]**2*p[4] + y[2]*y[10]*(-p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11])) - p[72]*y[33]/p[0]
    g_run[3] = -u[3]/p[0] + y[2]**2*(-p[5] - p[6]/2) + y[2]*y[10]*(p[5]*cos(y[3] - y[11]) - p[4]*sin(y[3] - y[11])) - p[72]*y[34]/p[0]
    g_run[4] = -u[4]/p[0] + y[20]*y[4]*(p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5])) + y[4]**2*p[7] - p[102]*y[42]/p[0]
    g_run[5] = -u[5]/p[0] + y[20]*y[4]*(p[8]*cos(y[21] - y[5]) + p[7]*sin(y[21] - y[5])) + y[4]**2*(-p[8] - p[9]/2) - p[102]*y[43]/p[0]
    g_run[6] = -u[6]/p[0] + y[18]*y[6]*(p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7])) + y[6]**2*p[10] - p[132]*y[51]/p[0]
    g_run[7] = -u[7]/p[0] + y[18]*y[6]*(p[11]*cos(y[19] - y[7]) + p[10]*sin(y[19] - y[7])) + y[6]**2*(-p[11] - p[12]/2) - p[132]*y[52]/p[0]
    g_run[8] = -u[8]/p[0] + y[0]*y[8]*(p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9])) + y[8]**2*(p[1] + p[13]) + y[8]*y[10]*(-p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11]))
    g_run[9] = -u[9]/p[0] + y[0]*y[8]*(p[2]*cos(y[1] - y[9]) + p[1]*sin(y[1] - y[9])) + y[8]**2*(-p[2] - p[14] - p[3]/2 - p[15]/2) + y[8]*y[10]*(p[14]*cos(y[9] - y[11]) - p[13]*sin(y[9] - y[11]))
    g_run[10] = -u[10]/p[0] + y[2]*y[10]*(p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11])) + y[8]*y[10]*(p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11])) + y[10]**2*(p[4] + p[13] + p[16]) + y[10]*y[12]*(-p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13]))
    g_run[11] = -u[11]/p[0] + y[2]*y[10]*(p[5]*cos(y[3] - y[11]) + p[4]*sin(y[3] - y[11])) + y[8]*y[10]*(p[14]*cos(y[9] - y[11]) + p[13]*sin(y[9] - y[11])) + y[10]**2*(-p[5] - p[14] - p[17] - p[6]/2 - p[15]/2 - p[18]/2) + y[10]*y[12]*(p[17]*cos(y[11] - y[13]) - p[16]*sin(y[11] - y[13]))
    g_run[12] = -u[12]/p[0] + y[10]*y[12]*(p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13])) + y[12]**2*(p[16] + 2*p[19]) + y[12]*y[14]*(-2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15]))
    g_run[13] = -u[13]/p[0] + y[10]*y[12]*(p[17]*cos(y[11] - y[13]) + p[16]*sin(y[11] - y[13])) + y[12]**2*(-p[17] - 2*p[20] - p[18]/2 - p[21]) + y[12]*y[14]*(2*p[20]*cos(y[13] - y[15]) - 2*p[19]*sin(y[13] - y[15]))
    g_run[14] = -u[14]/p[0] + y[12]*y[14]*(2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15])) + y[14]**2*(2*p[19] + 2*p[22]) + y[14]*y[16]*(-2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17]))
    g_run[15] = -u[15]/p[0] + y[12]*y[14]*(2*p[20]*cos(y[13] - y[15]) + 2*p[19]*sin(y[13] - y[15])) + y[14]**2*(-2*p[20] - 2*p[23] - p[21] - p[24]) + y[14]*y[16]*(2*p[23]*cos(y[15] - y[17]) - 2*p[22]*sin(y[15] - y[17]))
    g_run[16] = -u[16]/p[0] + y[18]*y[16]*(p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17])) + y[14]*y[16]*(2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17])) + y[16]**2*(2*p[22] + p[25])
    g_run[17] = -u[17]/p[0] + y[18]*y[16]*(p[26]*cos(y[19] - y[17]) + p[25]*sin(y[19] - y[17])) + y[14]*y[16]*(2*p[23]*cos(y[15] - y[17]) + 2*p[22]*sin(y[15] - y[17])) + y[16]**2*(-2*p[23] - p[26] - p[24] - p[27]/2)
    g_run[18] = -u[18]/p[0] + y[18]**2*(p[28] + p[10] + p[25]) + y[18]*y[20]*(-p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21])) + y[18]*y[6]*(-p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7])) + y[18]*y[16]*(-p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17]))
    g_run[19] = -u[19]/p[0] + y[18]**2*(-p[29] - p[11] - p[26] - p[30]/2 - p[12]/2 - p[27]/2) + y[18]*y[20]*(p[29]*cos(y[19] - y[21]) - p[28]*sin(y[19] - y[21])) + y[18]*y[6]*(p[11]*cos(y[19] - y[7]) - p[10]*sin(y[19] - y[7])) + y[18]*y[16]*(p[26]*cos(y[19] - y[17]) - p[25]*sin(y[19] - y[17]))
    g_run[20] = -u[20]/p[0] + y[18]*y[20]*(p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21])) + y[20]**2*(p[28] + p[7]) + y[20]*y[4]*(-p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5]))
    g_run[21] = -u[21]/p[0] + y[18]*y[20]*(p[29]*cos(y[19] - y[21]) + p[28]*sin(y[19] - y[21])) + y[20]**2*(-p[29] - p[8] - p[30]/2 - p[9]/2) + y[20]*y[4]*(p[8]*cos(y[21] - y[5]) - p[7]*sin(y[21] - y[5]))
    g_run[22] = p[52]*y[23] + y[0]*cos(x[0] - y[1]) + p[49]*y[22] - x[2]
    g_run[23] = p[52]*y[22] + y[0]*sin(x[0] - y[1]) - p[50]*y[23] - x[3]
    g_run[24] = y[0]*y[22]*sin(x[0] - y[1]) + y[0]*y[23]*cos(x[0] - y[1]) - y[24]
    g_run[25] = y[0]*y[22]*cos(x[0] - y[1]) - y[0]*y[23]*sin(x[0] - y[1]) - y[25]
    g_run[26] = -y[26] + Piecewise(np.array([(p[58], p[58] > p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5]), (p[59], p[59] < p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5]), (p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5], True)]))
    g_run[27] = p[54]*y[59] - y[27] + u[25] + x[8] - (x[1] - p[66])/p[61]
    g_run[28] = p[63]*(x[6] - x[7])/p[64] - y[28] + x[7]
    g_run[29] = x[1] - x[9] - y[29] - 1.0
    g_run[30] = -y[30] + Piecewise(np.array([(-p[71], p[71] < -p[70]*(p[68]*(-x[10] + y[29])/p[69] + x[10])), (p[71], p[71] < p[70]*(p[68]*(-x[10] + y[29])/p[69] + x[10])), (p[70]*(p[68]*(-x[10] + y[29])/p[69] + x[10]), True)]))
    g_run[31] = p[82]*y[32] + y[2]*cos(x[11] - y[3]) + p[79]*y[31] - x[13]
    g_run[32] = p[82]*y[31] + y[2]*sin(x[11] - y[3]) - p[80]*y[32] - x[14]
    g_run[33] = y[2]*y[31]*sin(x[11] - y[3]) + y[2]*y[32]*cos(x[11] - y[3]) - y[33]
    g_run[34] = y[2]*y[31]*cos(x[11] - y[3]) - y[2]*y[32]*sin(x[11] - y[3]) - y[34]
    g_run[35] = -y[35] + Piecewise(np.array([(p[88], p[88] > p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16]), (p[89], p[89] < p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16]), (p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16], True)]))
    g_run[36] = p[84]*y[59] - y[36] + u[29] + x[19] - (x[12] - p[96])/p[91]
    g_run[37] = p[93]*(x[17] - x[18])/p[94] - y[37] + x[18]
    g_run[38] = x[12] - x[20] - y[38] - 1.0
    g_run[39] = -y[39] + Piecewise(np.array([(-p[101], p[101] < -p[100]*(p[98]*(-x[21] + y[38])/p[99] + x[21])), (p[101], p[101] < p[100]*(p[98]*(-x[21] + y[38])/p[99] + x[21])), (p[100]*(p[98]*(-x[21] + y[38])/p[99] + x[21]), True)]))
    g_run[40] = p[112]*y[41] + y[4]*cos(x[22] - y[5]) + p[109]*y[40] - x[24]
    g_run[41] = p[112]*y[40] + y[4]*sin(x[22] - y[5]) - p[110]*y[41] - x[25]
    g_run[42] = y[4]*y[40]*sin(x[22] - y[5]) + y[4]*y[41]*cos(x[22] - y[5]) - y[42]
    g_run[43] = y[4]*y[40]*cos(x[22] - y[5]) - y[4]*y[41]*sin(x[22] - y[5]) - y[43]
    g_run[44] = -y[44] + Piecewise(np.array([(p[118], p[118] > p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27]), (p[119], p[119] < p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27]), (p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27], True)]))
    g_run[45] = p[114]*y[59] - y[45] + u[33] + x[30] - (x[23] - p[126])/p[121]
    g_run[46] = p[123]*(x[28] - x[29])/p[124] - y[46] + x[29]
    g_run[47] = x[23] - x[31] - y[47] - 1.0
    g_run[48] = -y[48] + Piecewise(np.array([(-p[131], p[131] < -p[130]*(p[128]*(-x[32] + y[47])/p[129] + x[32])), (p[131], p[131] < p[130]*(p[128]*(-x[32] + y[47])/p[129] + x[32])), (p[130]*(p[128]*(-x[32] + y[47])/p[129] + x[32]), True)]))
    g_run[49] = p[142]*y[50] + y[6]*cos(x[33] - y[7]) + p[139]*y[49] - x[35]
    g_run[50] = p[142]*y[49] + y[6]*sin(x[33] - y[7]) - p[140]*y[50] - x[36]
    g_run[51] = y[6]*y[49]*sin(x[33] - y[7]) + y[6]*y[50]*cos(x[33] - y[7]) - y[51]
    g_run[52] = y[6]*y[49]*cos(x[33] - y[7]) - y[6]*y[50]*sin(x[33] - y[7]) - y[52]
    g_run[53] = -y[53] + Piecewise(np.array([(p[148], p[148] > p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38]), (p[149], p[149] < p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38]), (p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38], True)]))
    g_run[54] = p[144]*y[59] - y[54] + u[37] + x[41] - (x[34] - p[156])/p[151]
    g_run[55] = p[153]*(x[39] - x[40])/p[154] - y[55] + x[40]
    g_run[56] = x[34] - x[42] - y[56] - 1.0
    g_run[57] = -y[57] + Piecewise(np.array([(-p[161], p[161] < -p[160]*(p[158]*(-x[43] + y[56])/p[159] + x[43])), (p[161], p[161] < p[160]*(p[158]*(-x[43] + y[56])/p[159] + x[43])), (p[160]*(p[158]*(-x[43] + y[56])/p[159] + x[43]), True)]))
    g_run[58] = -y[58] + (p[44]*p[42]*x[1] + p[74]*p[72]*x[12] + p[104]*p[102]*x[23] + p[134]*p[132]*x[34])/(p[44]*p[42] + p[74]*p[72] + p[104]*p[102] + p[134]*p[132])
    g_run[59] = p[163]*x[44] + p[162]*(1 - y[58]) - y[59]





@cuda.jit(device=True)
def f_run_gpu(f_run,x,u,p):

    sin = math.sin
    cos = math.cos
    sqrt = math.sqrt
    abs = math.fabs

    f_run[0] = -p[53]*x[0] + p[43]*(x[1] - y[58])
    f_run[1] = (-p[51]*(x[1] - y[58]) - y[22]*(p[52]*y[22] + y[0]*sin(x[0] - y[1])) - y[23]*(p[52]*y[23] + y[0]*cos(x[0] - y[1])) + y[28])/(2*p[44])
    f_run[2] = (-x[2] - y[22]*(-p[49] + p[47]) + y[26])/p[45]
    f_run[3] = (-x[3] + y[23]*(-p[50] + p[48]))/p[46]
    f_run[4] = (y[0] - x[4])/p[57]
    f_run[5] = -p[60]*(p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5] - y[26]) - x[4] + y[30] + u[22]
    f_run[6] = (y[27] - x[6])/p[62]
    f_run[7] = (x[6] - x[7])/p[64]
    f_run[8] = p[65]*(u[24] - y[24]) - 1.0e-6*x[8]
    f_run[9] = (x[1] - x[9] - 1.0)/p[67]
    f_run[10] = (-x[10] + y[29])/p[69]
    f_run[11] = -p[83]*x[11] + p[73]*(x[12] - y[58])
    f_run[12] = (-p[81]*(x[12] - y[58]) - y[31]*(p[82]*y[31] + y[2]*sin(x[11] - y[3])) - y[32]*(p[82]*y[32] + y[2]*cos(x[11] - y[3])) + y[37])/(2*p[74])
    f_run[13] = (-x[13] - y[31]*(-p[79] + p[77]) + y[35])/p[75]
    f_run[14] = (-x[14] + y[32]*(-p[80] + p[78]))/p[76]
    f_run[15] = (y[2] - x[15])/p[87]
    f_run[16] = -p[90]*(p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16] - y[35]) - x[15] + y[39] + u[26]
    f_run[17] = (y[36] - x[17])/p[92]
    f_run[18] = (x[17] - x[18])/p[94]
    f_run[19] = p[95]*(u[28] - y[33]) - 1.0e-6*x[19]
    f_run[20] = (x[12] - x[20] - 1.0)/p[97]
    f_run[21] = (-x[21] + y[38])/p[99]
    f_run[22] = -p[113]*x[22] + p[103]*(x[23] - y[58])
    f_run[23] = (-p[111]*(x[23] - y[58]) - y[40]*(p[112]*y[40] + y[4]*sin(x[22] - y[5])) - y[41]*(p[112]*y[41] + y[4]*cos(x[22] - y[5])) + y[46])/(2*p[104])
    f_run[24] = (-x[24] - y[40]*(-p[109] + p[107]) + y[44])/p[105]
    f_run[25] = (-x[25] + y[41]*(-p[110] + p[108]))/p[106]
    f_run[26] = (y[4] - x[26])/p[117]
    f_run[27] = -p[120]*(p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27] - y[44]) - x[26] + y[48] + u[30]
    f_run[28] = (y[45] - x[28])/p[122]
    f_run[29] = (x[28] - x[29])/p[124]
    f_run[30] = p[125]*(u[32] - y[42]) - 1.0e-6*x[30]
    f_run[31] = (x[23] - x[31] - 1.0)/p[127]
    f_run[32] = (-x[32] + y[47])/p[129]
    f_run[33] = -p[143]*x[33] + p[133]*(x[34] - y[58])
    f_run[34] = (-p[141]*(x[34] - y[58]) - y[49]*(p[142]*y[49] + y[6]*sin(x[33] - y[7])) - y[50]*(p[142]*y[50] + y[6]*cos(x[33] - y[7])) + y[55])/(2*p[134])
    f_run[35] = (-x[35] - y[49]*(-p[139] + p[137]) + y[53])/p[135]
    f_run[36] = (-x[36] + y[50]*(-p[140] + p[138]))/p[136]
    f_run[37] = (y[6] - x[37])/p[147]
    f_run[38] = -p[150]*(p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38] - y[53]) - x[37] + y[57] + u[34]
    f_run[39] = (y[54] - x[39])/p[152]
    f_run[40] = (x[39] - x[40])/p[154]
    f_run[41] = p[155]*(u[36] - y[51]) - 1.0e-6*x[41]
    f_run[42] = (x[34] - x[42] - 1.0)/p[157]
    f_run[43] = (-x[43] + y[56])/p[159]
    f_run[44] = 1 - y[58]





@cuda.jit(device=True)
def h_eval_gpu(z,x,u,p):

    sin = math.sin
    cos = math.cos
    sqrt = math.sqrt
    abs = math.fabs

    z[0] = y[0]
    z[1] = y[2]
    z[2] = y[4]
    z[3] = y[6]
    z[4] = y[8]
    z[5] = y[10]
    z[6] = y[12]
    z[7] = y[14]
    z[8] = y[16]
    z[9] = y[18]
    z[10] = y[20]
    z[11] = y[22]*(p[52]*y[22] + y[0]*sin(x[0] - y[1])) + y[23]*(p[52]*y[23] + y[0]*cos(x[0] - y[1]))
    z[12] = y[31]*(p[82]*y[31] + y[2]*sin(x[11] - y[3])) + y[32]*(p[82]*y[32] + y[2]*cos(x[11] - y[3]))
    z[13] = y[40]*(p[112]*y[40] + y[4]*sin(x[22] - y[5])) + y[41]*(p[112]*y[41] + y[4]*cos(x[22] - y[5]))
    z[14] = y[49]*(p[142]*y[49] + y[6]*sin(x[33] - y[7])) + y[50]*(p[142]*y[50] + y[6]*cos(x[33] - y[7]))





@numba.njit(cache=True)
def jac_ini_ss_eval_xy(jac_ini,x,y,u,p,xyup = 1):

    jac_ini[1,0] = (-y[0]*y[22]*cos(x[0] - y[1]) + y[0]*y[23]*sin(x[0] - y[1]))/(2*p[44])
    jac_ini[1,45] = (-y[22]*sin(x[0] - y[1]) - y[23]*cos(x[0] - y[1]))/(2*p[44])
    jac_ini[1,46] = (y[0]*y[22]*cos(x[0] - y[1]) - y[0]*y[23]*sin(x[0] - y[1]))/(2*p[44])
    jac_ini[1,67] = (-2*p[52]*y[22] - y[0]*sin(x[0] - y[1]))/(2*p[44])
    jac_ini[1,68] = (-2*p[52]*y[23] - y[0]*cos(x[0] - y[1]))/(2*p[44])
    jac_ini[12,11] = (-y[2]*y[31]*cos(x[11] - y[3]) + y[2]*y[32]*sin(x[11] - y[3]))/(2*p[74])
    jac_ini[12,47] = (-y[31]*sin(x[11] - y[3]) - y[32]*cos(x[11] - y[3]))/(2*p[74])
    jac_ini[12,48] = (y[2]*y[31]*cos(x[11] - y[3]) - y[2]*y[32]*sin(x[11] - y[3]))/(2*p[74])
    jac_ini[12,76] = (-2*p[82]*y[31] - y[2]*sin(x[11] - y[3]))/(2*p[74])
    jac_ini[12,77] = (-2*p[82]*y[32] - y[2]*cos(x[11] - y[3]))/(2*p[74])
    jac_ini[23,22] = (-y[4]*y[40]*cos(x[22] - y[5]) + y[4]*y[41]*sin(x[22] - y[5]))/(2*p[104])
    jac_ini[23,49] = (-y[40]*sin(x[22] - y[5]) - y[41]*cos(x[22] - y[5]))/(2*p[104])
    jac_ini[23,50] = (y[4]*y[40]*cos(x[22] - y[5]) - y[4]*y[41]*sin(x[22] - y[5]))/(2*p[104])
    jac_ini[23,85] = (-2*p[112]*y[40] - y[4]*sin(x[22] - y[5]))/(2*p[104])
    jac_ini[23,86] = (-2*p[112]*y[41] - y[4]*cos(x[22] - y[5]))/(2*p[104])
    jac_ini[34,33] = (-y[6]*y[49]*cos(x[33] - y[7]) + y[6]*y[50]*sin(x[33] - y[7]))/(2*p[134])
    jac_ini[34,51] = (-y[49]*sin(x[33] - y[7]) - y[50]*cos(x[33] - y[7]))/(2*p[134])
    jac_ini[34,52] = (y[6]*y[49]*cos(x[33] - y[7]) - y[6]*y[50]*sin(x[33] - y[7]))/(2*p[134])
    jac_ini[34,94] = (-2*p[142]*y[49] - y[6]*sin(x[33] - y[7]))/(2*p[134])
    jac_ini[34,95] = (-2*p[142]*y[50] - y[6]*cos(x[33] - y[7]))/(2*p[134])
    jac_ini[45,45] = 2*y[0]*p[1] + y[8]*(-p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9]))
    jac_ini[45,46] = y[0]*y[8]*(-p[2]*cos(y[1] - y[9]) + p[1]*sin(y[1] - y[9]))
    jac_ini[45,53] = y[0]*(-p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9]))
    jac_ini[45,54] = y[0]*y[8]*(p[2]*cos(y[1] - y[9]) - p[1]*sin(y[1] - y[9]))
    jac_ini[46,45] = 2*y[0]*(-p[2] - p[3]/2) + y[8]*(p[2]*cos(y[1] - y[9]) - p[1]*sin(y[1] - y[9]))
    jac_ini[46,46] = y[0]*y[8]*(-p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9]))
    jac_ini[46,53] = y[0]*(p[2]*cos(y[1] - y[9]) - p[1]*sin(y[1] - y[9]))
    jac_ini[46,54] = y[0]*y[8]*(p[2]*sin(y[1] - y[9]) + p[1]*cos(y[1] - y[9]))
    jac_ini[47,47] = 2*y[2]*p[4] + y[10]*(-p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11]))
    jac_ini[47,48] = y[2]*y[10]*(-p[5]*cos(y[3] - y[11]) + p[4]*sin(y[3] - y[11]))
    jac_ini[47,55] = y[2]*(-p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11]))
    jac_ini[47,56] = y[2]*y[10]*(p[5]*cos(y[3] - y[11]) - p[4]*sin(y[3] - y[11]))
    jac_ini[48,47] = 2*y[2]*(-p[5] - p[6]/2) + y[10]*(p[5]*cos(y[3] - y[11]) - p[4]*sin(y[3] - y[11]))
    jac_ini[48,48] = y[2]*y[10]*(-p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11]))
    jac_ini[48,55] = y[2]*(p[5]*cos(y[3] - y[11]) - p[4]*sin(y[3] - y[11]))
    jac_ini[48,56] = y[2]*y[10]*(p[5]*sin(y[3] - y[11]) + p[4]*cos(y[3] - y[11]))
    jac_ini[49,49] = y[20]*(p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5])) + 2*y[4]*p[7]
    jac_ini[49,50] = y[20]*y[4]*(-p[8]*cos(y[21] - y[5]) - p[7]*sin(y[21] - y[5]))
    jac_ini[49,65] = y[4]*(p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5]))
    jac_ini[49,66] = y[20]*y[4]*(p[8]*cos(y[21] - y[5]) + p[7]*sin(y[21] - y[5]))
    jac_ini[50,49] = y[20]*(p[8]*cos(y[21] - y[5]) + p[7]*sin(y[21] - y[5])) + 2*y[4]*(-p[8] - p[9]/2)
    jac_ini[50,50] = y[20]*y[4]*(p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5]))
    jac_ini[50,65] = y[4]*(p[8]*cos(y[21] - y[5]) + p[7]*sin(y[21] - y[5]))
    jac_ini[50,66] = y[20]*y[4]*(-p[8]*sin(y[21] - y[5]) + p[7]*cos(y[21] - y[5]))
    jac_ini[51,51] = y[18]*(p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7])) + 2*y[6]*p[10]
    jac_ini[51,52] = y[18]*y[6]*(-p[11]*cos(y[19] - y[7]) - p[10]*sin(y[19] - y[7]))
    jac_ini[51,63] = y[6]*(p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7]))
    jac_ini[51,64] = y[18]*y[6]*(p[11]*cos(y[19] - y[7]) + p[10]*sin(y[19] - y[7]))
    jac_ini[52,51] = y[18]*(p[11]*cos(y[19] - y[7]) + p[10]*sin(y[19] - y[7])) + 2*y[6]*(-p[11] - p[12]/2)
    jac_ini[52,52] = y[18]*y[6]*(p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7]))
    jac_ini[52,63] = y[6]*(p[11]*cos(y[19] - y[7]) + p[10]*sin(y[19] - y[7]))
    jac_ini[52,64] = y[18]*y[6]*(-p[11]*sin(y[19] - y[7]) + p[10]*cos(y[19] - y[7]))
    jac_ini[53,45] = y[8]*(p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9]))
    jac_ini[53,46] = y[0]*y[8]*(p[2]*cos(y[1] - y[9]) + p[1]*sin(y[1] - y[9]))
    jac_ini[53,53] = y[0]*(p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9])) + 2*y[8]*(p[1] + p[13]) + y[10]*(-p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11]))
    jac_ini[53,54] = y[0]*y[8]*(-p[2]*cos(y[1] - y[9]) - p[1]*sin(y[1] - y[9])) + y[8]*y[10]*(-p[14]*cos(y[9] - y[11]) + p[13]*sin(y[9] - y[11]))
    jac_ini[53,55] = y[8]*(-p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11]))
    jac_ini[53,56] = y[8]*y[10]*(p[14]*cos(y[9] - y[11]) - p[13]*sin(y[9] - y[11]))
    jac_ini[54,45] = y[8]*(p[2]*cos(y[1] - y[9]) + p[1]*sin(y[1] - y[9]))
    jac_ini[54,46] = y[0]*y[8]*(-p[2]*sin(y[1] - y[9]) + p[1]*cos(y[1] - y[9]))
    jac_ini[54,53] = y[0]*(p[2]*cos(y[1] - y[9]) + p[1]*sin(y[1] - y[9])) + 2*y[8]*(-p[2] - p[14] - p[3]/2 - p[15]/2) + y[10]*(p[14]*cos(y[9] - y[11]) - p[13]*sin(y[9] - y[11]))
    jac_ini[54,54] = y[0]*y[8]*(p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9])) + y[8]*y[10]*(-p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11]))
    jac_ini[54,55] = y[8]*(p[14]*cos(y[9] - y[11]) - p[13]*sin(y[9] - y[11]))
    jac_ini[54,56] = y[8]*y[10]*(p[14]*sin(y[9] - y[11]) + p[13]*cos(y[9] - y[11]))
    jac_ini[55,47] = y[10]*(p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11]))
    jac_ini[55,48] = y[2]*y[10]*(p[5]*cos(y[3] - y[11]) + p[4]*sin(y[3] - y[11]))
    jac_ini[55,53] = y[10]*(p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11]))
    jac_ini[55,54] = y[8]*y[10]*(p[14]*cos(y[9] - y[11]) + p[13]*sin(y[9] - y[11]))
    jac_ini[55,55] = y[2]*(p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11])) + y[8]*(p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11])) + 2*y[10]*(p[4] + p[13] + p[16]) + y[12]*(-p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13]))
    jac_ini[55,56] = y[2]*y[10]*(-p[5]*cos(y[3] - y[11]) - p[4]*sin(y[3] - y[11])) + y[8]*y[10]*(-p[14]*cos(y[9] - y[11]) - p[13]*sin(y[9] - y[11])) + y[10]*y[12]*(-p[17]*cos(y[11] - y[13]) + p[16]*sin(y[11] - y[13]))
    jac_ini[55,57] = y[10]*(-p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13]))
    jac_ini[55,58] = y[10]*y[12]*(p[17]*cos(y[11] - y[13]) - p[16]*sin(y[11] - y[13]))
    jac_ini[56,47] = y[10]*(p[5]*cos(y[3] - y[11]) + p[4]*sin(y[3] - y[11]))
    jac_ini[56,48] = y[2]*y[10]*(-p[5]*sin(y[3] - y[11]) + p[4]*cos(y[3] - y[11]))
    jac_ini[56,53] = y[10]*(p[14]*cos(y[9] - y[11]) + p[13]*sin(y[9] - y[11]))
    jac_ini[56,54] = y[8]*y[10]*(-p[14]*sin(y[9] - y[11]) + p[13]*cos(y[9] - y[11]))
    jac_ini[56,55] = y[2]*(p[5]*cos(y[3] - y[11]) + p[4]*sin(y[3] - y[11])) + y[8]*(p[14]*cos(y[9] - y[11]) + p[13]*sin(y[9] - y[11])) + 2*y[10]*(-p[5] - p[14] - p[17] - p[6]/2 - p[15]/2 - p[18]/2) + y[12]*(p[17]*cos(y[11] - y[13]) - p[16]*sin(y[11] - y[13]))
    jac_ini[56,56] = y[2]*y[10]*(p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11])) + y[8]*y[10]*(p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11])) + y[10]*y[12]*(-p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13]))
    jac_ini[56,57] = y[10]*(p[17]*cos(y[11] - y[13]) - p[16]*sin(y[11] - y[13]))
    jac_ini[56,58] = y[10]*y[12]*(p[17]*sin(y[11] - y[13]) + p[16]*cos(y[11] - y[13]))
    jac_ini[57,55] = y[12]*(p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13]))
    jac_ini[57,56] = y[10]*y[12]*(p[17]*cos(y[11] - y[13]) + p[16]*sin(y[11] - y[13]))
    jac_ini[57,57] = y[10]*(p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13])) + 2*y[12]*(p[16] + 2*p[19]) + y[14]*(-2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15]))
    jac_ini[57,58] = y[10]*y[12]*(-p[17]*cos(y[11] - y[13]) - p[16]*sin(y[11] - y[13])) + y[12]*y[14]*(-2*p[20]*cos(y[13] - y[15]) + 2*p[19]*sin(y[13] - y[15]))
    jac_ini[57,59] = y[12]*(-2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15]))
    jac_ini[57,60] = y[12]*y[14]*(2*p[20]*cos(y[13] - y[15]) - 2*p[19]*sin(y[13] - y[15]))
    jac_ini[58,55] = y[12]*(p[17]*cos(y[11] - y[13]) + p[16]*sin(y[11] - y[13]))
    jac_ini[58,56] = y[10]*y[12]*(-p[17]*sin(y[11] - y[13]) + p[16]*cos(y[11] - y[13]))
    jac_ini[58,57] = y[10]*(p[17]*cos(y[11] - y[13]) + p[16]*sin(y[11] - y[13])) + 2*y[12]*(-p[17] - 2*p[20] - p[18]/2 - p[21]) + y[14]*(2*p[20]*cos(y[13] - y[15]) - 2*p[19]*sin(y[13] - y[15]))
    jac_ini[58,58] = y[10]*y[12]*(p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13])) + y[12]*y[14]*(-2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15]))
    jac_ini[58,59] = y[12]*(2*p[20]*cos(y[13] - y[15]) - 2*p[19]*sin(y[13] - y[15]))
    jac_ini[58,60] = y[12]*y[14]*(2*p[20]*sin(y[13] - y[15]) + 2*p[19]*cos(y[13] - y[15]))
    jac_ini[59,57] = y[14]*(2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15]))
    jac_ini[59,58] = y[12]*y[14]*(2*p[20]*cos(y[13] - y[15]) + 2*p[19]*sin(y[13] - y[15]))
    jac_ini[59,59] = y[12]*(2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15])) + 2*y[14]*(2*p[19] + 2*p[22]) + y[16]*(-2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17]))
    jac_ini[59,60] = y[12]*y[14]*(-2*p[20]*cos(y[13] - y[15]) - 2*p[19]*sin(y[13] - y[15])) + y[14]*y[16]*(-2*p[23]*cos(y[15] - y[17]) + 2*p[22]*sin(y[15] - y[17]))
    jac_ini[59,61] = y[14]*(-2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17]))
    jac_ini[59,62] = y[14]*y[16]*(2*p[23]*cos(y[15] - y[17]) - 2*p[22]*sin(y[15] - y[17]))
    jac_ini[60,57] = y[14]*(2*p[20]*cos(y[13] - y[15]) + 2*p[19]*sin(y[13] - y[15]))
    jac_ini[60,58] = y[12]*y[14]*(-2*p[20]*sin(y[13] - y[15]) + 2*p[19]*cos(y[13] - y[15]))
    jac_ini[60,59] = y[12]*(2*p[20]*cos(y[13] - y[15]) + 2*p[19]*sin(y[13] - y[15])) + 2*y[14]*(-2*p[20] - 2*p[23] - p[21] - p[24]) + y[16]*(2*p[23]*cos(y[15] - y[17]) - 2*p[22]*sin(y[15] - y[17]))
    jac_ini[60,60] = y[12]*y[14]*(2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15])) + y[14]*y[16]*(-2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17]))
    jac_ini[60,61] = y[14]*(2*p[23]*cos(y[15] - y[17]) - 2*p[22]*sin(y[15] - y[17]))
    jac_ini[60,62] = y[14]*y[16]*(2*p[23]*sin(y[15] - y[17]) + 2*p[22]*cos(y[15] - y[17]))
    jac_ini[61,59] = y[16]*(2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17]))
    jac_ini[61,60] = y[14]*y[16]*(2*p[23]*cos(y[15] - y[17]) + 2*p[22]*sin(y[15] - y[17]))
    jac_ini[61,61] = y[18]*(p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17])) + y[14]*(2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17])) + 2*y[16]*(2*p[22] + p[25])
    jac_ini[61,62] = y[18]*y[16]*(-p[26]*cos(y[19] - y[17]) - p[25]*sin(y[19] - y[17])) + y[14]*y[16]*(-2*p[23]*cos(y[15] - y[17]) - 2*p[22]*sin(y[15] - y[17]))
    jac_ini[61,63] = y[16]*(p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17]))
    jac_ini[61,64] = y[18]*y[16]*(p[26]*cos(y[19] - y[17]) + p[25]*sin(y[19] - y[17]))
    jac_ini[62,59] = y[16]*(2*p[23]*cos(y[15] - y[17]) + 2*p[22]*sin(y[15] - y[17]))
    jac_ini[62,60] = y[14]*y[16]*(-2*p[23]*sin(y[15] - y[17]) + 2*p[22]*cos(y[15] - y[17]))
    jac_ini[62,61] = y[18]*(p[26]*cos(y[19] - y[17]) + p[25]*sin(y[19] - y[17])) + y[14]*(2*p[23]*cos(y[15] - y[17]) + 2*p[22]*sin(y[15] - y[17])) + 2*y[16]*(-2*p[23] - p[26] - p[24] - p[27]/2)
    jac_ini[62,62] = y[18]*y[16]*(p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17])) + y[14]*y[16]*(2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17]))
    jac_ini[62,63] = y[16]*(p[26]*cos(y[19] - y[17]) + p[25]*sin(y[19] - y[17]))
    jac_ini[62,64] = y[18]*y[16]*(-p[26]*sin(y[19] - y[17]) + p[25]*cos(y[19] - y[17]))
    jac_ini[63,51] = y[18]*(-p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7]))
    jac_ini[63,52] = y[18]*y[6]*(p[11]*cos(y[19] - y[7]) - p[10]*sin(y[19] - y[7]))
    jac_ini[63,61] = y[18]*(-p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17]))
    jac_ini[63,62] = y[18]*y[16]*(p[26]*cos(y[19] - y[17]) - p[25]*sin(y[19] - y[17]))
    jac_ini[63,63] = 2*y[18]*(p[28] + p[10] + p[25]) + y[20]*(-p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21])) + y[6]*(-p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7])) + y[16]*(-p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17]))
    jac_ini[63,64] = y[18]*y[20]*(-p[29]*cos(y[19] - y[21]) + p[28]*sin(y[19] - y[21])) + y[18]*y[6]*(-p[11]*cos(y[19] - y[7]) + p[10]*sin(y[19] - y[7])) + y[18]*y[16]*(-p[26]*cos(y[19] - y[17]) + p[25]*sin(y[19] - y[17]))
    jac_ini[63,65] = y[18]*(-p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21]))
    jac_ini[63,66] = y[18]*y[20]*(p[29]*cos(y[19] - y[21]) - p[28]*sin(y[19] - y[21]))
    jac_ini[64,51] = y[18]*(p[11]*cos(y[19] - y[7]) - p[10]*sin(y[19] - y[7]))
    jac_ini[64,52] = y[18]*y[6]*(p[11]*sin(y[19] - y[7]) + p[10]*cos(y[19] - y[7]))
    jac_ini[64,61] = y[18]*(p[26]*cos(y[19] - y[17]) - p[25]*sin(y[19] - y[17]))
    jac_ini[64,62] = y[18]*y[16]*(p[26]*sin(y[19] - y[17]) + p[25]*cos(y[19] - y[17]))
    jac_ini[64,63] = 2*y[18]*(-p[29] - p[11] - p[26] - p[30]/2 - p[12]/2 - p[27]/2) + y[20]*(p[29]*cos(y[19] - y[21]) - p[28]*sin(y[19] - y[21])) + y[6]*(p[11]*cos(y[19] - y[7]) - p[10]*sin(y[19] - y[7])) + y[16]*(p[26]*cos(y[19] - y[17]) - p[25]*sin(y[19] - y[17]))
    jac_ini[64,64] = y[18]*y[20]*(-p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21])) + y[18]*y[6]*(-p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7])) + y[18]*y[16]*(-p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17]))
    jac_ini[64,65] = y[18]*(p[29]*cos(y[19] - y[21]) - p[28]*sin(y[19] - y[21]))
    jac_ini[64,66] = y[18]*y[20]*(p[29]*sin(y[19] - y[21]) + p[28]*cos(y[19] - y[21]))
    jac_ini[65,49] = y[20]*(-p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5]))
    jac_ini[65,50] = y[20]*y[4]*(p[8]*cos(y[21] - y[5]) - p[7]*sin(y[21] - y[5]))
    jac_ini[65,63] = y[20]*(p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21]))
    jac_ini[65,64] = y[18]*y[20]*(p[29]*cos(y[19] - y[21]) + p[28]*sin(y[19] - y[21]))
    jac_ini[65,65] = y[18]*(p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21])) + 2*y[20]*(p[28] + p[7]) + y[4]*(-p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5]))
    jac_ini[65,66] = y[18]*y[20]*(-p[29]*cos(y[19] - y[21]) - p[28]*sin(y[19] - y[21])) + y[20]*y[4]*(-p[8]*cos(y[21] - y[5]) + p[7]*sin(y[21] - y[5]))
    jac_ini[66,49] = y[20]*(p[8]*cos(y[21] - y[5]) - p[7]*sin(y[21] - y[5]))
    jac_ini[66,50] = y[20]*y[4]*(p[8]*sin(y[21] - y[5]) + p[7]*cos(y[21] - y[5]))
    jac_ini[66,63] = y[20]*(p[29]*cos(y[19] - y[21]) + p[28]*sin(y[19] - y[21]))
    jac_ini[66,64] = y[18]*y[20]*(-p[29]*sin(y[19] - y[21]) + p[28]*cos(y[19] - y[21]))
    jac_ini[66,65] = y[18]*(p[29]*cos(y[19] - y[21]) + p[28]*sin(y[19] - y[21])) + 2*y[20]*(-p[29] - p[8] - p[30]/2 - p[9]/2) + y[4]*(p[8]*cos(y[21] - y[5]) - p[7]*sin(y[21] - y[5]))
    jac_ini[66,66] = y[18]*y[20]*(p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21])) + y[20]*y[4]*(-p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5]))
    jac_ini[67,0] = -y[0]*sin(x[0] - y[1])
    jac_ini[67,45] = cos(x[0] - y[1])
    jac_ini[67,46] = y[0]*sin(x[0] - y[1])
    jac_ini[68,0] = y[0]*cos(x[0] - y[1])
    jac_ini[68,45] = sin(x[0] - y[1])
    jac_ini[68,46] = -y[0]*cos(x[0] - y[1])
    jac_ini[69,0] = y[0]*y[22]*cos(x[0] - y[1]) - y[0]*y[23]*sin(x[0] - y[1])
    jac_ini[69,45] = y[22]*sin(x[0] - y[1]) + y[23]*cos(x[0] - y[1])
    jac_ini[69,46] = -y[0]*y[22]*cos(x[0] - y[1]) + y[0]*y[23]*sin(x[0] - y[1])
    jac_ini[69,67] = y[0]*sin(x[0] - y[1])
    jac_ini[69,68] = y[0]*cos(x[0] - y[1])
    jac_ini[70,0] = -y[0]*y[22]*sin(x[0] - y[1]) - y[0]*y[23]*cos(x[0] - y[1])
    jac_ini[70,45] = y[22]*cos(x[0] - y[1]) - y[23]*sin(x[0] - y[1])
    jac_ini[70,46] = y[0]*y[22]*sin(x[0] - y[1]) + y[0]*y[23]*cos(x[0] - y[1])
    jac_ini[70,67] = y[0]*cos(x[0] - y[1])
    jac_ini[70,68] = -y[0]*sin(x[0] - y[1])
    jac_ini[71,4] = Piecewise(np.array([(0, (p[58] > p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5]) | (p[59] < p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5])), (-p[55], True)]))
    jac_ini[71,5] = Piecewise(np.array([(0, (p[58] > p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5]) | (p[59] < p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5])), (p[56], True)]))
    jac_ini[71,75] = Piecewise(np.array([(0, (p[58] > p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5]) | (p[59] < p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5])), (p[55], True)]))
    jac_ini[75,10] = Piecewise(np.array([(0, (p[71] < p[70]*(p[68]*(-x[10] + y[29])/p[69] + x[10])) | (p[71] < -p[70]*(p[68]*(-x[10] + y[29])/p[69] + x[10]))), (p[70]*(-p[68]/p[69] + 1), True)]))
    jac_ini[75,74] = Piecewise(np.array([(0, (p[71] < p[70]*(p[68]*(-x[10] + y[29])/p[69] + x[10])) | (p[71] < -p[70]*(p[68]*(-x[10] + y[29])/p[69] + x[10]))), (p[70]*p[68]/p[69], True)]))
    jac_ini[76,11] = -y[2]*sin(x[11] - y[3])
    jac_ini[76,47] = cos(x[11] - y[3])
    jac_ini[76,48] = y[2]*sin(x[11] - y[3])
    jac_ini[77,11] = y[2]*cos(x[11] - y[3])
    jac_ini[77,47] = sin(x[11] - y[3])
    jac_ini[77,48] = -y[2]*cos(x[11] - y[3])
    jac_ini[78,11] = y[2]*y[31]*cos(x[11] - y[3]) - y[2]*y[32]*sin(x[11] - y[3])
    jac_ini[78,47] = y[31]*sin(x[11] - y[3]) + y[32]*cos(x[11] - y[3])
    jac_ini[78,48] = -y[2]*y[31]*cos(x[11] - y[3]) + y[2]*y[32]*sin(x[11] - y[3])
    jac_ini[78,76] = y[2]*sin(x[11] - y[3])
    jac_ini[78,77] = y[2]*cos(x[11] - y[3])
    jac_ini[79,11] = -y[2]*y[31]*sin(x[11] - y[3]) - y[2]*y[32]*cos(x[11] - y[3])
    jac_ini[79,47] = y[31]*cos(x[11] - y[3]) - y[32]*sin(x[11] - y[3])
    jac_ini[79,48] = y[2]*y[31]*sin(x[11] - y[3]) + y[2]*y[32]*cos(x[11] - y[3])
    jac_ini[79,76] = y[2]*cos(x[11] - y[3])
    jac_ini[79,77] = -y[2]*sin(x[11] - y[3])
    jac_ini[80,15] = Piecewise(np.array([(0, (p[88] > p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16]) | (p[89] < p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16])), (-p[85], True)]))
    jac_ini[80,16] = Piecewise(np.array([(0, (p[88] > p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16]) | (p[89] < p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16])), (p[86], True)]))
    jac_ini[80,84] = Piecewise(np.array([(0, (p[88] > p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16]) | (p[89] < p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16])), (p[85], True)]))
    jac_ini[84,21] = Piecewise(np.array([(0, (p[101] < p[100]*(p[98]*(-x[21] + y[38])/p[99] + x[21])) | (p[101] < -p[100]*(p[98]*(-x[21] + y[38])/p[99] + x[21]))), (p[100]*(-p[98]/p[99] + 1), True)]))
    jac_ini[84,83] = Piecewise(np.array([(0, (p[101] < p[100]*(p[98]*(-x[21] + y[38])/p[99] + x[21])) | (p[101] < -p[100]*(p[98]*(-x[21] + y[38])/p[99] + x[21]))), (p[100]*p[98]/p[99], True)]))
    jac_ini[85,22] = -y[4]*sin(x[22] - y[5])
    jac_ini[85,49] = cos(x[22] - y[5])
    jac_ini[85,50] = y[4]*sin(x[22] - y[5])
    jac_ini[86,22] = y[4]*cos(x[22] - y[5])
    jac_ini[86,49] = sin(x[22] - y[5])
    jac_ini[86,50] = -y[4]*cos(x[22] - y[5])
    jac_ini[87,22] = y[4]*y[40]*cos(x[22] - y[5]) - y[4]*y[41]*sin(x[22] - y[5])
    jac_ini[87,49] = y[40]*sin(x[22] - y[5]) + y[41]*cos(x[22] - y[5])
    jac_ini[87,50] = -y[4]*y[40]*cos(x[22] - y[5]) + y[4]*y[41]*sin(x[22] - y[5])
    jac_ini[87,85] = y[4]*sin(x[22] - y[5])
    jac_ini[87,86] = y[4]*cos(x[22] - y[5])
    jac_ini[88,22] = -y[4]*y[40]*sin(x[22] - y[5]) - y[4]*y[41]*cos(x[22] - y[5])
    jac_ini[88,49] = y[40]*cos(x[22] - y[5]) - y[41]*sin(x[22] - y[5])
    jac_ini[88,50] = y[4]*y[40]*sin(x[22] - y[5]) + y[4]*y[41]*cos(x[22] - y[5])
    jac_ini[88,85] = y[4]*cos(x[22] - y[5])
    jac_ini[88,86] = -y[4]*sin(x[22] - y[5])
    jac_ini[89,26] = Piecewise(np.array([(0, (p[118] > p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27]) | (p[119] < p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27])), (-p[115], True)]))
    jac_ini[89,27] = Piecewise(np.array([(0, (p[118] > p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27]) | (p[119] < p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27])), (p[116], True)]))
    jac_ini[89,93] = Piecewise(np.array([(0, (p[118] > p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27]) | (p[119] < p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27])), (p[115], True)]))
    jac_ini[93,32] = Piecewise(np.array([(0, (p[131] < p[130]*(p[128]*(-x[32] + y[47])/p[129] + x[32])) | (p[131] < -p[130]*(p[128]*(-x[32] + y[47])/p[129] + x[32]))), (p[130]*(-p[128]/p[129] + 1), True)]))
    jac_ini[93,92] = Piecewise(np.array([(0, (p[131] < p[130]*(p[128]*(-x[32] + y[47])/p[129] + x[32])) | (p[131] < -p[130]*(p[128]*(-x[32] + y[47])/p[129] + x[32]))), (p[130]*p[128]/p[129], True)]))
    jac_ini[94,33] = -y[6]*sin(x[33] - y[7])
    jac_ini[94,51] = cos(x[33] - y[7])
    jac_ini[94,52] = y[6]*sin(x[33] - y[7])
    jac_ini[95,33] = y[6]*cos(x[33] - y[7])
    jac_ini[95,51] = sin(x[33] - y[7])
    jac_ini[95,52] = -y[6]*cos(x[33] - y[7])
    jac_ini[96,33] = y[6]*y[49]*cos(x[33] - y[7]) - y[6]*y[50]*sin(x[33] - y[7])
    jac_ini[96,51] = y[49]*sin(x[33] - y[7]) + y[50]*cos(x[33] - y[7])
    jac_ini[96,52] = -y[6]*y[49]*cos(x[33] - y[7]) + y[6]*y[50]*sin(x[33] - y[7])
    jac_ini[96,94] = y[6]*sin(x[33] - y[7])
    jac_ini[96,95] = y[6]*cos(x[33] - y[7])
    jac_ini[97,33] = -y[6]*y[49]*sin(x[33] - y[7]) - y[6]*y[50]*cos(x[33] - y[7])
    jac_ini[97,51] = y[49]*cos(x[33] - y[7]) - y[50]*sin(x[33] - y[7])
    jac_ini[97,52] = y[6]*y[49]*sin(x[33] - y[7]) + y[6]*y[50]*cos(x[33] - y[7])
    jac_ini[97,94] = y[6]*cos(x[33] - y[7])
    jac_ini[97,95] = -y[6]*sin(x[33] - y[7])
    jac_ini[98,37] = Piecewise(np.array([(0, (p[148] > p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38]) | (p[149] < p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38])), (-p[145], True)]))
    jac_ini[98,38] = Piecewise(np.array([(0, (p[148] > p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38]) | (p[149] < p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38])), (p[146], True)]))
    jac_ini[98,102] = Piecewise(np.array([(0, (p[148] > p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38]) | (p[149] < p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38])), (p[145], True)]))
    jac_ini[102,43] = Piecewise(np.array([(0, (p[161] < p[160]*(p[158]*(-x[43] + y[56])/p[159] + x[43])) | (p[161] < -p[160]*(p[158]*(-x[43] + y[56])/p[159] + x[43]))), (p[160]*(-p[158]/p[159] + 1), True)]))
    jac_ini[102,101] = Piecewise(np.array([(0, (p[161] < p[160]*(p[158]*(-x[43] + y[56])/p[159] + x[43])) | (p[161] < -p[160]*(p[158]*(-x[43] + y[56])/p[159] + x[43]))), (p[160]*p[158]/p[159], True)]))

@numba.njit(cache=True)
def jac_ini_ss_eval_up(jac_ini,x,y,u,p,xyup = 1):

    jac_ini[0,0] = -p[53]
    jac_ini[0,1] = p[43]
    jac_ini[0,103] = -p[43]
    jac_ini[1,1] = -p[51]/(2*p[44])
    jac_ini[1,73] = 1/(2*p[44])
    jac_ini[1,103] = p[51]/(2*p[44])
    jac_ini[2,2] = -1/p[45]
    jac_ini[2,67] = (p[49] - p[47])/p[45]
    jac_ini[2,71] = 1/p[45]
    jac_ini[3,3] = -1/p[46]
    jac_ini[3,68] = (-p[50] + p[48])/p[46]
    jac_ini[4,4] = -1/p[57]
    jac_ini[4,45] = 1/p[57]
    jac_ini[5,4] = p[55]*p[60] - 1
    jac_ini[5,5] = -p[56]*p[60]
    jac_ini[5,71] = p[60]
    jac_ini[5,75] = -p[55]*p[60] + 1
    jac_ini[6,6] = -1/p[62]
    jac_ini[6,72] = 1/p[62]
    jac_ini[7,6] = 1/p[64]
    jac_ini[7,7] = -1/p[64]
    jac_ini[8,69] = -p[65]
    jac_ini[9,1] = 1/p[67]
    jac_ini[9,9] = -1/p[67]
    jac_ini[10,10] = -1/p[69]
    jac_ini[10,74] = 1/p[69]
    jac_ini[11,11] = -p[83]
    jac_ini[11,12] = p[73]
    jac_ini[11,103] = -p[73]
    jac_ini[12,12] = -p[81]/(2*p[74])
    jac_ini[12,82] = 1/(2*p[74])
    jac_ini[12,103] = p[81]/(2*p[74])
    jac_ini[13,13] = -1/p[75]
    jac_ini[13,76] = (p[79] - p[77])/p[75]
    jac_ini[13,80] = 1/p[75]
    jac_ini[14,14] = -1/p[76]
    jac_ini[14,77] = (-p[80] + p[78])/p[76]
    jac_ini[15,15] = -1/p[87]
    jac_ini[15,47] = 1/p[87]
    jac_ini[16,15] = p[85]*p[90] - 1
    jac_ini[16,16] = -p[86]*p[90]
    jac_ini[16,80] = p[90]
    jac_ini[16,84] = -p[85]*p[90] + 1
    jac_ini[17,17] = -1/p[92]
    jac_ini[17,81] = 1/p[92]
    jac_ini[18,17] = 1/p[94]
    jac_ini[18,18] = -1/p[94]
    jac_ini[19,78] = -p[95]
    jac_ini[20,12] = 1/p[97]
    jac_ini[20,20] = -1/p[97]
    jac_ini[21,21] = -1/p[99]
    jac_ini[21,83] = 1/p[99]
    jac_ini[22,22] = -p[113]
    jac_ini[22,23] = p[103]
    jac_ini[22,103] = -p[103]
    jac_ini[23,23] = -p[111]/(2*p[104])
    jac_ini[23,91] = 1/(2*p[104])
    jac_ini[23,103] = p[111]/(2*p[104])
    jac_ini[24,24] = -1/p[105]
    jac_ini[24,85] = (p[109] - p[107])/p[105]
    jac_ini[24,89] = 1/p[105]
    jac_ini[25,25] = -1/p[106]
    jac_ini[25,86] = (-p[110] + p[108])/p[106]
    jac_ini[26,26] = -1/p[117]
    jac_ini[26,49] = 1/p[117]
    jac_ini[27,26] = p[115]*p[120] - 1
    jac_ini[27,27] = -p[116]*p[120]
    jac_ini[27,89] = p[120]
    jac_ini[27,93] = -p[115]*p[120] + 1
    jac_ini[28,28] = -1/p[122]
    jac_ini[28,90] = 1/p[122]
    jac_ini[29,28] = 1/p[124]
    jac_ini[29,29] = -1/p[124]
    jac_ini[30,87] = -p[125]
    jac_ini[31,23] = 1/p[127]
    jac_ini[31,31] = -1/p[127]
    jac_ini[32,32] = -1/p[129]
    jac_ini[32,92] = 1/p[129]
    jac_ini[33,33] = -p[143]
    jac_ini[33,34] = p[133]
    jac_ini[33,103] = -p[133]
    jac_ini[34,34] = -p[141]/(2*p[134])
    jac_ini[34,100] = 1/(2*p[134])
    jac_ini[34,103] = p[141]/(2*p[134])
    jac_ini[35,35] = -1/p[135]
    jac_ini[35,94] = (p[139] - p[137])/p[135]
    jac_ini[35,98] = 1/p[135]
    jac_ini[36,36] = -1/p[136]
    jac_ini[36,95] = (-p[140] + p[138])/p[136]
    jac_ini[37,37] = -1/p[147]
    jac_ini[37,51] = 1/p[147]
    jac_ini[38,37] = p[145]*p[150] - 1
    jac_ini[38,38] = -p[146]*p[150]
    jac_ini[38,98] = p[150]
    jac_ini[38,102] = -p[145]*p[150] + 1
    jac_ini[39,39] = -1/p[152]
    jac_ini[39,99] = 1/p[152]
    jac_ini[40,39] = 1/p[154]
    jac_ini[40,40] = -1/p[154]
    jac_ini[41,96] = -p[155]
    jac_ini[42,34] = 1/p[157]
    jac_ini[42,42] = -1/p[157]
    jac_ini[43,43] = -1/p[159]
    jac_ini[43,101] = 1/p[159]
    jac_ini[45,69] = -p[42]/p[0]
    jac_ini[46,70] = -p[42]/p[0]
    jac_ini[47,78] = -p[72]/p[0]
    jac_ini[48,79] = -p[72]/p[0]
    jac_ini[49,87] = -p[102]/p[0]
    jac_ini[50,88] = -p[102]/p[0]
    jac_ini[51,96] = -p[132]/p[0]
    jac_ini[52,97] = -p[132]/p[0]
    jac_ini[67,67] = p[49]
    jac_ini[67,68] = p[52]
    jac_ini[68,67] = p[52]
    jac_ini[68,68] = -p[50]
    jac_ini[72,1] = -1/p[61]
    jac_ini[72,104] = p[54]
    jac_ini[73,6] = p[63]/p[64]
    jac_ini[73,7] = -p[63]/p[64] + 1
    jac_ini[76,76] = p[79]
    jac_ini[76,77] = p[82]
    jac_ini[77,76] = p[82]
    jac_ini[77,77] = -p[80]
    jac_ini[81,12] = -1/p[91]
    jac_ini[81,104] = p[84]
    jac_ini[82,17] = p[93]/p[94]
    jac_ini[82,18] = -p[93]/p[94] + 1
    jac_ini[85,85] = p[109]
    jac_ini[85,86] = p[112]
    jac_ini[86,85] = p[112]
    jac_ini[86,86] = -p[110]
    jac_ini[90,23] = -1/p[121]
    jac_ini[90,104] = p[114]
    jac_ini[91,28] = p[123]/p[124]
    jac_ini[91,29] = -p[123]/p[124] + 1
    jac_ini[94,94] = p[139]
    jac_ini[94,95] = p[142]
    jac_ini[95,94] = p[142]
    jac_ini[95,95] = -p[140]
    jac_ini[99,34] = -1/p[151]
    jac_ini[99,104] = p[144]
    jac_ini[100,39] = p[153]/p[154]
    jac_ini[100,40] = -p[153]/p[154] + 1
    jac_ini[103,1] = p[44]*p[42]/(p[44]*p[42] + p[74]*p[72] + p[104]*p[102] + p[134]*p[132])
    jac_ini[103,12] = p[74]*p[72]/(p[44]*p[42] + p[74]*p[72] + p[104]*p[102] + p[134]*p[132])
    jac_ini[103,23] = p[104]*p[102]/(p[44]*p[42] + p[74]*p[72] + p[104]*p[102] + p[134]*p[132])
    jac_ini[103,34] = p[134]*p[132]/(p[44]*p[42] + p[74]*p[72] + p[104]*p[102] + p[134]*p[132])
    jac_ini[104,44] = p[163]
    jac_ini[104,103] = -p[162]

def jac_ini_ss_eval_num(jac_ini,x,y,u,p,xyup = 1):

    jac_ini[8,8] = -1.00000000000000e-6
    jac_ini[19,19] = -1.00000000000000e-6
    jac_ini[30,30] = -1.00000000000000e-6
    jac_ini[41,41] = -1.00000000000000e-6
    jac_ini[44,103] = -1
    jac_ini[67,2] = -1
    jac_ini[68,3] = -1
    jac_ini[69,69] = -1
    jac_ini[70,70] = -1
    jac_ini[71,71] = -1
    jac_ini[72,8] = 1
    jac_ini[72,72] = -1
    jac_ini[73,73] = -1
    jac_ini[74,1] = 1
    jac_ini[74,9] = -1
    jac_ini[74,74] = -1
    jac_ini[75,75] = -1
    jac_ini[76,13] = -1
    jac_ini[77,14] = -1
    jac_ini[78,78] = -1
    jac_ini[79,79] = -1
    jac_ini[80,80] = -1
    jac_ini[81,19] = 1
    jac_ini[81,81] = -1
    jac_ini[82,82] = -1
    jac_ini[83,12] = 1
    jac_ini[83,20] = -1
    jac_ini[83,83] = -1
    jac_ini[84,84] = -1
    jac_ini[85,24] = -1
    jac_ini[86,25] = -1
    jac_ini[87,87] = -1
    jac_ini[88,88] = -1
    jac_ini[89,89] = -1
    jac_ini[90,30] = 1
    jac_ini[90,90] = -1
    jac_ini[91,91] = -1
    jac_ini[92,23] = 1
    jac_ini[92,31] = -1
    jac_ini[92,92] = -1
    jac_ini[93,93] = -1
    jac_ini[94,35] = -1
    jac_ini[95,36] = -1
    jac_ini[96,96] = -1
    jac_ini[97,97] = -1
    jac_ini[98,98] = -1
    jac_ini[99,41] = 1
    jac_ini[99,99] = -1
    jac_ini[100,100] = -1
    jac_ini[101,34] = 1
    jac_ini[101,42] = -1
    jac_ini[101,101] = -1
    jac_ini[102,102] = -1
    jac_ini[103,103] = -1
    jac_ini[104,104] = -1





@numba.njit(cache=True)
def sp_jac_ini_eval_xy(sp_jac_ini,x,y,u,p,Dt,xyup = 1):

    sp_jac_ini[3] = (-y[0]*y[22]*cos(x[0] - y[1]) + y[0]*y[23]*sin(x[0] - y[1]))/(2*p[44])
    sp_jac_ini[5] = (-y[22]*sin(x[0] - y[1]) - y[23]*cos(x[0] - y[1]))/(2*p[44])
    sp_jac_ini[6] = (y[0]*y[22]*cos(x[0] - y[1]) - y[0]*y[23]*sin(x[0] - y[1]))/(2*p[44])
    sp_jac_ini[7] = (-2*p[52]*y[22] - y[0]*sin(x[0] - y[1]))/(2*p[44])
    sp_jac_ini[8] = (-2*p[52]*y[23] - y[0]*cos(x[0] - y[1]))/(2*p[44])
    sp_jac_ini[35] = (-y[2]*y[31]*cos(x[11] - y[3]) + y[2]*y[32]*sin(x[11] - y[3]))/(2*p[74])
    sp_jac_ini[37] = (-y[31]*sin(x[11] - y[3]) - y[32]*cos(x[11] - y[3]))/(2*p[74])
    sp_jac_ini[38] = (y[2]*y[31]*cos(x[11] - y[3]) - y[2]*y[32]*sin(x[11] - y[3]))/(2*p[74])
    sp_jac_ini[39] = (-2*p[82]*y[31] - y[2]*sin(x[11] - y[3]))/(2*p[74])
    sp_jac_ini[40] = (-2*p[82]*y[32] - y[2]*cos(x[11] - y[3]))/(2*p[74])
    sp_jac_ini[67] = (-y[4]*y[40]*cos(x[22] - y[5]) + y[4]*y[41]*sin(x[22] - y[5]))/(2*p[104])
    sp_jac_ini[69] = (-y[40]*sin(x[22] - y[5]) - y[41]*cos(x[22] - y[5]))/(2*p[104])
    sp_jac_ini[70] = (y[4]*y[40]*cos(x[22] - y[5]) - y[4]*y[41]*sin(x[22] - y[5]))/(2*p[104])
    sp_jac_ini[71] = (-2*p[112]*y[40] - y[4]*sin(x[22] - y[5]))/(2*p[104])
    sp_jac_ini[72] = (-2*p[112]*y[41] - y[4]*cos(x[22] - y[5]))/(2*p[104])
    sp_jac_ini[99] = (-y[6]*y[49]*cos(x[33] - y[7]) + y[6]*y[50]*sin(x[33] - y[7]))/(2*p[134])
    sp_jac_ini[101] = (-y[49]*sin(x[33] - y[7]) - y[50]*cos(x[33] - y[7]))/(2*p[134])
    sp_jac_ini[102] = (y[6]*y[49]*cos(x[33] - y[7]) - y[6]*y[50]*sin(x[33] - y[7]))/(2*p[134])
    sp_jac_ini[103] = (-2*p[142]*y[49] - y[6]*sin(x[33] - y[7]))/(2*p[134])
    sp_jac_ini[104] = (-2*p[142]*y[50] - y[6]*cos(x[33] - y[7]))/(2*p[134])
    sp_jac_ini[129] = 2*y[0]*p[1] + y[8]*(-p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9]))
    sp_jac_ini[130] = y[0]*y[8]*(-p[2]*cos(y[1] - y[9]) + p[1]*sin(y[1] - y[9]))
    sp_jac_ini[131] = y[0]*(-p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9]))
    sp_jac_ini[132] = y[0]*y[8]*(p[2]*cos(y[1] - y[9]) - p[1]*sin(y[1] - y[9]))
    sp_jac_ini[134] = 2*y[0]*(-p[2] - p[3]/2) + y[8]*(p[2]*cos(y[1] - y[9]) - p[1]*sin(y[1] - y[9]))
    sp_jac_ini[135] = y[0]*y[8]*(-p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9]))
    sp_jac_ini[136] = y[0]*(p[2]*cos(y[1] - y[9]) - p[1]*sin(y[1] - y[9]))
    sp_jac_ini[137] = y[0]*y[8]*(p[2]*sin(y[1] - y[9]) + p[1]*cos(y[1] - y[9]))
    sp_jac_ini[139] = 2*y[2]*p[4] + y[10]*(-p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11]))
    sp_jac_ini[140] = y[2]*y[10]*(-p[5]*cos(y[3] - y[11]) + p[4]*sin(y[3] - y[11]))
    sp_jac_ini[141] = y[2]*(-p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11]))
    sp_jac_ini[142] = y[2]*y[10]*(p[5]*cos(y[3] - y[11]) - p[4]*sin(y[3] - y[11]))
    sp_jac_ini[144] = 2*y[2]*(-p[5] - p[6]/2) + y[10]*(p[5]*cos(y[3] - y[11]) - p[4]*sin(y[3] - y[11]))
    sp_jac_ini[145] = y[2]*y[10]*(-p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11]))
    sp_jac_ini[146] = y[2]*(p[5]*cos(y[3] - y[11]) - p[4]*sin(y[3] - y[11]))
    sp_jac_ini[147] = y[2]*y[10]*(p[5]*sin(y[3] - y[11]) + p[4]*cos(y[3] - y[11]))
    sp_jac_ini[149] = y[20]*(p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5])) + 2*y[4]*p[7]
    sp_jac_ini[150] = y[20]*y[4]*(-p[8]*cos(y[21] - y[5]) - p[7]*sin(y[21] - y[5]))
    sp_jac_ini[151] = y[4]*(p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5]))
    sp_jac_ini[152] = y[20]*y[4]*(p[8]*cos(y[21] - y[5]) + p[7]*sin(y[21] - y[5]))
    sp_jac_ini[154] = y[20]*(p[8]*cos(y[21] - y[5]) + p[7]*sin(y[21] - y[5])) + 2*y[4]*(-p[8] - p[9]/2)
    sp_jac_ini[155] = y[20]*y[4]*(p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5]))
    sp_jac_ini[156] = y[4]*(p[8]*cos(y[21] - y[5]) + p[7]*sin(y[21] - y[5]))
    sp_jac_ini[157] = y[20]*y[4]*(-p[8]*sin(y[21] - y[5]) + p[7]*cos(y[21] - y[5]))
    sp_jac_ini[159] = y[18]*(p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7])) + 2*y[6]*p[10]
    sp_jac_ini[160] = y[18]*y[6]*(-p[11]*cos(y[19] - y[7]) - p[10]*sin(y[19] - y[7]))
    sp_jac_ini[161] = y[6]*(p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7]))
    sp_jac_ini[162] = y[18]*y[6]*(p[11]*cos(y[19] - y[7]) + p[10]*sin(y[19] - y[7]))
    sp_jac_ini[164] = y[18]*(p[11]*cos(y[19] - y[7]) + p[10]*sin(y[19] - y[7])) + 2*y[6]*(-p[11] - p[12]/2)
    sp_jac_ini[165] = y[18]*y[6]*(p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7]))
    sp_jac_ini[166] = y[6]*(p[11]*cos(y[19] - y[7]) + p[10]*sin(y[19] - y[7]))
    sp_jac_ini[167] = y[18]*y[6]*(-p[11]*sin(y[19] - y[7]) + p[10]*cos(y[19] - y[7]))
    sp_jac_ini[169] = y[8]*(p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9]))
    sp_jac_ini[170] = y[0]*y[8]*(p[2]*cos(y[1] - y[9]) + p[1]*sin(y[1] - y[9]))
    sp_jac_ini[171] = y[0]*(p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9])) + 2*y[8]*(p[1] + p[13]) + y[10]*(-p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11]))
    sp_jac_ini[172] = y[0]*y[8]*(-p[2]*cos(y[1] - y[9]) - p[1]*sin(y[1] - y[9])) + y[8]*y[10]*(-p[14]*cos(y[9] - y[11]) + p[13]*sin(y[9] - y[11]))
    sp_jac_ini[173] = y[8]*(-p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11]))
    sp_jac_ini[174] = y[8]*y[10]*(p[14]*cos(y[9] - y[11]) - p[13]*sin(y[9] - y[11]))
    sp_jac_ini[175] = y[8]*(p[2]*cos(y[1] - y[9]) + p[1]*sin(y[1] - y[9]))
    sp_jac_ini[176] = y[0]*y[8]*(-p[2]*sin(y[1] - y[9]) + p[1]*cos(y[1] - y[9]))
    sp_jac_ini[177] = y[0]*(p[2]*cos(y[1] - y[9]) + p[1]*sin(y[1] - y[9])) + 2*y[8]*(-p[2] - p[14] - p[3]/2 - p[15]/2) + y[10]*(p[14]*cos(y[9] - y[11]) - p[13]*sin(y[9] - y[11]))
    sp_jac_ini[178] = y[0]*y[8]*(p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9])) + y[8]*y[10]*(-p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11]))
    sp_jac_ini[179] = y[8]*(p[14]*cos(y[9] - y[11]) - p[13]*sin(y[9] - y[11]))
    sp_jac_ini[180] = y[8]*y[10]*(p[14]*sin(y[9] - y[11]) + p[13]*cos(y[9] - y[11]))
    sp_jac_ini[181] = y[10]*(p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11]))
    sp_jac_ini[182] = y[2]*y[10]*(p[5]*cos(y[3] - y[11]) + p[4]*sin(y[3] - y[11]))
    sp_jac_ini[183] = y[10]*(p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11]))
    sp_jac_ini[184] = y[8]*y[10]*(p[14]*cos(y[9] - y[11]) + p[13]*sin(y[9] - y[11]))
    sp_jac_ini[185] = y[2]*(p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11])) + y[8]*(p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11])) + 2*y[10]*(p[4] + p[13] + p[16]) + y[12]*(-p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13]))
    sp_jac_ini[186] = y[2]*y[10]*(-p[5]*cos(y[3] - y[11]) - p[4]*sin(y[3] - y[11])) + y[8]*y[10]*(-p[14]*cos(y[9] - y[11]) - p[13]*sin(y[9] - y[11])) + y[10]*y[12]*(-p[17]*cos(y[11] - y[13]) + p[16]*sin(y[11] - y[13]))
    sp_jac_ini[187] = y[10]*(-p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13]))
    sp_jac_ini[188] = y[10]*y[12]*(p[17]*cos(y[11] - y[13]) - p[16]*sin(y[11] - y[13]))
    sp_jac_ini[189] = y[10]*(p[5]*cos(y[3] - y[11]) + p[4]*sin(y[3] - y[11]))
    sp_jac_ini[190] = y[2]*y[10]*(-p[5]*sin(y[3] - y[11]) + p[4]*cos(y[3] - y[11]))
    sp_jac_ini[191] = y[10]*(p[14]*cos(y[9] - y[11]) + p[13]*sin(y[9] - y[11]))
    sp_jac_ini[192] = y[8]*y[10]*(-p[14]*sin(y[9] - y[11]) + p[13]*cos(y[9] - y[11]))
    sp_jac_ini[193] = y[2]*(p[5]*cos(y[3] - y[11]) + p[4]*sin(y[3] - y[11])) + y[8]*(p[14]*cos(y[9] - y[11]) + p[13]*sin(y[9] - y[11])) + 2*y[10]*(-p[5] - p[14] - p[17] - p[6]/2 - p[15]/2 - p[18]/2) + y[12]*(p[17]*cos(y[11] - y[13]) - p[16]*sin(y[11] - y[13]))
    sp_jac_ini[194] = y[2]*y[10]*(p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11])) + y[8]*y[10]*(p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11])) + y[10]*y[12]*(-p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13]))
    sp_jac_ini[195] = y[10]*(p[17]*cos(y[11] - y[13]) - p[16]*sin(y[11] - y[13]))
    sp_jac_ini[196] = y[10]*y[12]*(p[17]*sin(y[11] - y[13]) + p[16]*cos(y[11] - y[13]))
    sp_jac_ini[197] = y[12]*(p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13]))
    sp_jac_ini[198] = y[10]*y[12]*(p[17]*cos(y[11] - y[13]) + p[16]*sin(y[11] - y[13]))
    sp_jac_ini[199] = y[10]*(p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13])) + 2*y[12]*(p[16] + 2*p[19]) + y[14]*(-2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15]))
    sp_jac_ini[200] = y[10]*y[12]*(-p[17]*cos(y[11] - y[13]) - p[16]*sin(y[11] - y[13])) + y[12]*y[14]*(-2*p[20]*cos(y[13] - y[15]) + 2*p[19]*sin(y[13] - y[15]))
    sp_jac_ini[201] = y[12]*(-2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15]))
    sp_jac_ini[202] = y[12]*y[14]*(2*p[20]*cos(y[13] - y[15]) - 2*p[19]*sin(y[13] - y[15]))
    sp_jac_ini[203] = y[12]*(p[17]*cos(y[11] - y[13]) + p[16]*sin(y[11] - y[13]))
    sp_jac_ini[204] = y[10]*y[12]*(-p[17]*sin(y[11] - y[13]) + p[16]*cos(y[11] - y[13]))
    sp_jac_ini[205] = y[10]*(p[17]*cos(y[11] - y[13]) + p[16]*sin(y[11] - y[13])) + 2*y[12]*(-p[17] - 2*p[20] - p[18]/2 - p[21]) + y[14]*(2*p[20]*cos(y[13] - y[15]) - 2*p[19]*sin(y[13] - y[15]))
    sp_jac_ini[206] = y[10]*y[12]*(p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13])) + y[12]*y[14]*(-2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15]))
    sp_jac_ini[207] = y[12]*(2*p[20]*cos(y[13] - y[15]) - 2*p[19]*sin(y[13] - y[15]))
    sp_jac_ini[208] = y[12]*y[14]*(2*p[20]*sin(y[13] - y[15]) + 2*p[19]*cos(y[13] - y[15]))
    sp_jac_ini[209] = y[14]*(2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15]))
    sp_jac_ini[210] = y[12]*y[14]*(2*p[20]*cos(y[13] - y[15]) + 2*p[19]*sin(y[13] - y[15]))
    sp_jac_ini[211] = y[12]*(2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15])) + 2*y[14]*(2*p[19] + 2*p[22]) + y[16]*(-2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17]))
    sp_jac_ini[212] = y[12]*y[14]*(-2*p[20]*cos(y[13] - y[15]) - 2*p[19]*sin(y[13] - y[15])) + y[14]*y[16]*(-2*p[23]*cos(y[15] - y[17]) + 2*p[22]*sin(y[15] - y[17]))
    sp_jac_ini[213] = y[14]*(-2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17]))
    sp_jac_ini[214] = y[14]*y[16]*(2*p[23]*cos(y[15] - y[17]) - 2*p[22]*sin(y[15] - y[17]))
    sp_jac_ini[215] = y[14]*(2*p[20]*cos(y[13] - y[15]) + 2*p[19]*sin(y[13] - y[15]))
    sp_jac_ini[216] = y[12]*y[14]*(-2*p[20]*sin(y[13] - y[15]) + 2*p[19]*cos(y[13] - y[15]))
    sp_jac_ini[217] = y[12]*(2*p[20]*cos(y[13] - y[15]) + 2*p[19]*sin(y[13] - y[15])) + 2*y[14]*(-2*p[20] - 2*p[23] - p[21] - p[24]) + y[16]*(2*p[23]*cos(y[15] - y[17]) - 2*p[22]*sin(y[15] - y[17]))
    sp_jac_ini[218] = y[12]*y[14]*(2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15])) + y[14]*y[16]*(-2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17]))
    sp_jac_ini[219] = y[14]*(2*p[23]*cos(y[15] - y[17]) - 2*p[22]*sin(y[15] - y[17]))
    sp_jac_ini[220] = y[14]*y[16]*(2*p[23]*sin(y[15] - y[17]) + 2*p[22]*cos(y[15] - y[17]))
    sp_jac_ini[221] = y[16]*(2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17]))
    sp_jac_ini[222] = y[14]*y[16]*(2*p[23]*cos(y[15] - y[17]) + 2*p[22]*sin(y[15] - y[17]))
    sp_jac_ini[223] = y[18]*(p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17])) + y[14]*(2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17])) + 2*y[16]*(2*p[22] + p[25])
    sp_jac_ini[224] = y[18]*y[16]*(-p[26]*cos(y[19] - y[17]) - p[25]*sin(y[19] - y[17])) + y[14]*y[16]*(-2*p[23]*cos(y[15] - y[17]) - 2*p[22]*sin(y[15] - y[17]))
    sp_jac_ini[225] = y[16]*(p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17]))
    sp_jac_ini[226] = y[18]*y[16]*(p[26]*cos(y[19] - y[17]) + p[25]*sin(y[19] - y[17]))
    sp_jac_ini[227] = y[16]*(2*p[23]*cos(y[15] - y[17]) + 2*p[22]*sin(y[15] - y[17]))
    sp_jac_ini[228] = y[14]*y[16]*(-2*p[23]*sin(y[15] - y[17]) + 2*p[22]*cos(y[15] - y[17]))
    sp_jac_ini[229] = y[18]*(p[26]*cos(y[19] - y[17]) + p[25]*sin(y[19] - y[17])) + y[14]*(2*p[23]*cos(y[15] - y[17]) + 2*p[22]*sin(y[15] - y[17])) + 2*y[16]*(-2*p[23] - p[26] - p[24] - p[27]/2)
    sp_jac_ini[230] = y[18]*y[16]*(p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17])) + y[14]*y[16]*(2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17]))
    sp_jac_ini[231] = y[16]*(p[26]*cos(y[19] - y[17]) + p[25]*sin(y[19] - y[17]))
    sp_jac_ini[232] = y[18]*y[16]*(-p[26]*sin(y[19] - y[17]) + p[25]*cos(y[19] - y[17]))
    sp_jac_ini[233] = y[18]*(-p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7]))
    sp_jac_ini[234] = y[18]*y[6]*(p[11]*cos(y[19] - y[7]) - p[10]*sin(y[19] - y[7]))
    sp_jac_ini[235] = y[18]*(-p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17]))
    sp_jac_ini[236] = y[18]*y[16]*(p[26]*cos(y[19] - y[17]) - p[25]*sin(y[19] - y[17]))
    sp_jac_ini[237] = 2*y[18]*(p[28] + p[10] + p[25]) + y[20]*(-p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21])) + y[6]*(-p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7])) + y[16]*(-p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17]))
    sp_jac_ini[238] = y[18]*y[20]*(-p[29]*cos(y[19] - y[21]) + p[28]*sin(y[19] - y[21])) + y[18]*y[6]*(-p[11]*cos(y[19] - y[7]) + p[10]*sin(y[19] - y[7])) + y[18]*y[16]*(-p[26]*cos(y[19] - y[17]) + p[25]*sin(y[19] - y[17]))
    sp_jac_ini[239] = y[18]*(-p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21]))
    sp_jac_ini[240] = y[18]*y[20]*(p[29]*cos(y[19] - y[21]) - p[28]*sin(y[19] - y[21]))
    sp_jac_ini[241] = y[18]*(p[11]*cos(y[19] - y[7]) - p[10]*sin(y[19] - y[7]))
    sp_jac_ini[242] = y[18]*y[6]*(p[11]*sin(y[19] - y[7]) + p[10]*cos(y[19] - y[7]))
    sp_jac_ini[243] = y[18]*(p[26]*cos(y[19] - y[17]) - p[25]*sin(y[19] - y[17]))
    sp_jac_ini[244] = y[18]*y[16]*(p[26]*sin(y[19] - y[17]) + p[25]*cos(y[19] - y[17]))
    sp_jac_ini[245] = 2*y[18]*(-p[29] - p[11] - p[26] - p[30]/2 - p[12]/2 - p[27]/2) + y[20]*(p[29]*cos(y[19] - y[21]) - p[28]*sin(y[19] - y[21])) + y[6]*(p[11]*cos(y[19] - y[7]) - p[10]*sin(y[19] - y[7])) + y[16]*(p[26]*cos(y[19] - y[17]) - p[25]*sin(y[19] - y[17]))
    sp_jac_ini[246] = y[18]*y[20]*(-p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21])) + y[18]*y[6]*(-p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7])) + y[18]*y[16]*(-p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17]))
    sp_jac_ini[247] = y[18]*(p[29]*cos(y[19] - y[21]) - p[28]*sin(y[19] - y[21]))
    sp_jac_ini[248] = y[18]*y[20]*(p[29]*sin(y[19] - y[21]) + p[28]*cos(y[19] - y[21]))
    sp_jac_ini[249] = y[20]*(-p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5]))
    sp_jac_ini[250] = y[20]*y[4]*(p[8]*cos(y[21] - y[5]) - p[7]*sin(y[21] - y[5]))
    sp_jac_ini[251] = y[20]*(p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21]))
    sp_jac_ini[252] = y[18]*y[20]*(p[29]*cos(y[19] - y[21]) + p[28]*sin(y[19] - y[21]))
    sp_jac_ini[253] = y[18]*(p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21])) + 2*y[20]*(p[28] + p[7]) + y[4]*(-p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5]))
    sp_jac_ini[254] = y[18]*y[20]*(-p[29]*cos(y[19] - y[21]) - p[28]*sin(y[19] - y[21])) + y[20]*y[4]*(-p[8]*cos(y[21] - y[5]) + p[7]*sin(y[21] - y[5]))
    sp_jac_ini[255] = y[20]*(p[8]*cos(y[21] - y[5]) - p[7]*sin(y[21] - y[5]))
    sp_jac_ini[256] = y[20]*y[4]*(p[8]*sin(y[21] - y[5]) + p[7]*cos(y[21] - y[5]))
    sp_jac_ini[257] = y[20]*(p[29]*cos(y[19] - y[21]) + p[28]*sin(y[19] - y[21]))
    sp_jac_ini[258] = y[18]*y[20]*(-p[29]*sin(y[19] - y[21]) + p[28]*cos(y[19] - y[21]))
    sp_jac_ini[259] = y[18]*(p[29]*cos(y[19] - y[21]) + p[28]*sin(y[19] - y[21])) + 2*y[20]*(-p[29] - p[8] - p[30]/2 - p[9]/2) + y[4]*(p[8]*cos(y[21] - y[5]) - p[7]*sin(y[21] - y[5]))
    sp_jac_ini[260] = y[18]*y[20]*(p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21])) + y[20]*y[4]*(-p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5]))
    sp_jac_ini[261] = -y[0]*sin(x[0] - y[1])
    sp_jac_ini[263] = cos(x[0] - y[1])
    sp_jac_ini[264] = y[0]*sin(x[0] - y[1])
    sp_jac_ini[267] = y[0]*cos(x[0] - y[1])
    sp_jac_ini[269] = sin(x[0] - y[1])
    sp_jac_ini[270] = -y[0]*cos(x[0] - y[1])
    sp_jac_ini[273] = y[0]*y[22]*cos(x[0] - y[1]) - y[0]*y[23]*sin(x[0] - y[1])
    sp_jac_ini[274] = y[22]*sin(x[0] - y[1]) + y[23]*cos(x[0] - y[1])
    sp_jac_ini[275] = -y[0]*y[22]*cos(x[0] - y[1]) + y[0]*y[23]*sin(x[0] - y[1])
    sp_jac_ini[276] = y[0]*sin(x[0] - y[1])
    sp_jac_ini[277] = y[0]*cos(x[0] - y[1])
    sp_jac_ini[279] = -y[0]*y[22]*sin(x[0] - y[1]) - y[0]*y[23]*cos(x[0] - y[1])
    sp_jac_ini[280] = y[22]*cos(x[0] - y[1]) - y[23]*sin(x[0] - y[1])
    sp_jac_ini[281] = y[0]*y[22]*sin(x[0] - y[1]) + y[0]*y[23]*cos(x[0] - y[1])
    sp_jac_ini[282] = y[0]*cos(x[0] - y[1])
    sp_jac_ini[283] = -y[0]*sin(x[0] - y[1])
    sp_jac_ini[285] = Piecewise(np.array([(0, (p[58] > p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5]) | (p[59] < p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5])), (-p[55], True)]))
    sp_jac_ini[286] = Piecewise(np.array([(0, (p[58] > p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5]) | (p[59] < p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5])), (p[56], True)]))
    sp_jac_ini[288] = Piecewise(np.array([(0, (p[58] > p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5]) | (p[59] < p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5])), (p[55], True)]))
    sp_jac_ini[299] = Piecewise(np.array([(0, (p[71] < p[70]*(p[68]*(-x[10] + y[29])/p[69] + x[10])) | (p[71] < -p[70]*(p[68]*(-x[10] + y[29])/p[69] + x[10]))), (p[70]*(-p[68]/p[69] + 1), True)]))
    sp_jac_ini[300] = Piecewise(np.array([(0, (p[71] < p[70]*(p[68]*(-x[10] + y[29])/p[69] + x[10])) | (p[71] < -p[70]*(p[68]*(-x[10] + y[29])/p[69] + x[10]))), (p[70]*p[68]/p[69], True)]))
    sp_jac_ini[302] = -y[2]*sin(x[11] - y[3])
    sp_jac_ini[304] = cos(x[11] - y[3])
    sp_jac_ini[305] = y[2]*sin(x[11] - y[3])
    sp_jac_ini[308] = y[2]*cos(x[11] - y[3])
    sp_jac_ini[310] = sin(x[11] - y[3])
    sp_jac_ini[311] = -y[2]*cos(x[11] - y[3])
    sp_jac_ini[314] = y[2]*y[31]*cos(x[11] - y[3]) - y[2]*y[32]*sin(x[11] - y[3])
    sp_jac_ini[315] = y[31]*sin(x[11] - y[3]) + y[32]*cos(x[11] - y[3])
    sp_jac_ini[316] = -y[2]*y[31]*cos(x[11] - y[3]) + y[2]*y[32]*sin(x[11] - y[3])
    sp_jac_ini[317] = y[2]*sin(x[11] - y[3])
    sp_jac_ini[318] = y[2]*cos(x[11] - y[3])
    sp_jac_ini[320] = -y[2]*y[31]*sin(x[11] - y[3]) - y[2]*y[32]*cos(x[11] - y[3])
    sp_jac_ini[321] = y[31]*cos(x[11] - y[3]) - y[32]*sin(x[11] - y[3])
    sp_jac_ini[322] = y[2]*y[31]*sin(x[11] - y[3]) + y[2]*y[32]*cos(x[11] - y[3])
    sp_jac_ini[323] = y[2]*cos(x[11] - y[3])
    sp_jac_ini[324] = -y[2]*sin(x[11] - y[3])
    sp_jac_ini[326] = Piecewise(np.array([(0, (p[88] > p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16]) | (p[89] < p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16])), (-p[85], True)]))
    sp_jac_ini[327] = Piecewise(np.array([(0, (p[88] > p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16]) | (p[89] < p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16])), (p[86], True)]))
    sp_jac_ini[329] = Piecewise(np.array([(0, (p[88] > p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16]) | (p[89] < p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16])), (p[85], True)]))
    sp_jac_ini[340] = Piecewise(np.array([(0, (p[101] < p[100]*(p[98]*(-x[21] + y[38])/p[99] + x[21])) | (p[101] < -p[100]*(p[98]*(-x[21] + y[38])/p[99] + x[21]))), (p[100]*(-p[98]/p[99] + 1), True)]))
    sp_jac_ini[341] = Piecewise(np.array([(0, (p[101] < p[100]*(p[98]*(-x[21] + y[38])/p[99] + x[21])) | (p[101] < -p[100]*(p[98]*(-x[21] + y[38])/p[99] + x[21]))), (p[100]*p[98]/p[99], True)]))
    sp_jac_ini[343] = -y[4]*sin(x[22] - y[5])
    sp_jac_ini[345] = cos(x[22] - y[5])
    sp_jac_ini[346] = y[4]*sin(x[22] - y[5])
    sp_jac_ini[349] = y[4]*cos(x[22] - y[5])
    sp_jac_ini[351] = sin(x[22] - y[5])
    sp_jac_ini[352] = -y[4]*cos(x[22] - y[5])
    sp_jac_ini[355] = y[4]*y[40]*cos(x[22] - y[5]) - y[4]*y[41]*sin(x[22] - y[5])
    sp_jac_ini[356] = y[40]*sin(x[22] - y[5]) + y[41]*cos(x[22] - y[5])
    sp_jac_ini[357] = -y[4]*y[40]*cos(x[22] - y[5]) + y[4]*y[41]*sin(x[22] - y[5])
    sp_jac_ini[358] = y[4]*sin(x[22] - y[5])
    sp_jac_ini[359] = y[4]*cos(x[22] - y[5])
    sp_jac_ini[361] = -y[4]*y[40]*sin(x[22] - y[5]) - y[4]*y[41]*cos(x[22] - y[5])
    sp_jac_ini[362] = y[40]*cos(x[22] - y[5]) - y[41]*sin(x[22] - y[5])
    sp_jac_ini[363] = y[4]*y[40]*sin(x[22] - y[5]) + y[4]*y[41]*cos(x[22] - y[5])
    sp_jac_ini[364] = y[4]*cos(x[22] - y[5])
    sp_jac_ini[365] = -y[4]*sin(x[22] - y[5])
    sp_jac_ini[367] = Piecewise(np.array([(0, (p[118] > p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27]) | (p[119] < p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27])), (-p[115], True)]))
    sp_jac_ini[368] = Piecewise(np.array([(0, (p[118] > p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27]) | (p[119] < p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27])), (p[116], True)]))
    sp_jac_ini[370] = Piecewise(np.array([(0, (p[118] > p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27]) | (p[119] < p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27])), (p[115], True)]))
    sp_jac_ini[381] = Piecewise(np.array([(0, (p[131] < p[130]*(p[128]*(-x[32] + y[47])/p[129] + x[32])) | (p[131] < -p[130]*(p[128]*(-x[32] + y[47])/p[129] + x[32]))), (p[130]*(-p[128]/p[129] + 1), True)]))
    sp_jac_ini[382] = Piecewise(np.array([(0, (p[131] < p[130]*(p[128]*(-x[32] + y[47])/p[129] + x[32])) | (p[131] < -p[130]*(p[128]*(-x[32] + y[47])/p[129] + x[32]))), (p[130]*p[128]/p[129], True)]))
    sp_jac_ini[384] = -y[6]*sin(x[33] - y[7])
    sp_jac_ini[386] = cos(x[33] - y[7])
    sp_jac_ini[387] = y[6]*sin(x[33] - y[7])
    sp_jac_ini[390] = y[6]*cos(x[33] - y[7])
    sp_jac_ini[392] = sin(x[33] - y[7])
    sp_jac_ini[393] = -y[6]*cos(x[33] - y[7])
    sp_jac_ini[396] = y[6]*y[49]*cos(x[33] - y[7]) - y[6]*y[50]*sin(x[33] - y[7])
    sp_jac_ini[397] = y[49]*sin(x[33] - y[7]) + y[50]*cos(x[33] - y[7])
    sp_jac_ini[398] = -y[6]*y[49]*cos(x[33] - y[7]) + y[6]*y[50]*sin(x[33] - y[7])
    sp_jac_ini[399] = y[6]*sin(x[33] - y[7])
    sp_jac_ini[400] = y[6]*cos(x[33] - y[7])
    sp_jac_ini[402] = -y[6]*y[49]*sin(x[33] - y[7]) - y[6]*y[50]*cos(x[33] - y[7])
    sp_jac_ini[403] = y[49]*cos(x[33] - y[7]) - y[50]*sin(x[33] - y[7])
    sp_jac_ini[404] = y[6]*y[49]*sin(x[33] - y[7]) + y[6]*y[50]*cos(x[33] - y[7])
    sp_jac_ini[405] = y[6]*cos(x[33] - y[7])
    sp_jac_ini[406] = -y[6]*sin(x[33] - y[7])
    sp_jac_ini[408] = Piecewise(np.array([(0, (p[148] > p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38]) | (p[149] < p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38])), (-p[145], True)]))
    sp_jac_ini[409] = Piecewise(np.array([(0, (p[148] > p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38]) | (p[149] < p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38])), (p[146], True)]))
    sp_jac_ini[411] = Piecewise(np.array([(0, (p[148] > p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38]) | (p[149] < p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38])), (p[145], True)]))
    sp_jac_ini[422] = Piecewise(np.array([(0, (p[161] < p[160]*(p[158]*(-x[43] + y[56])/p[159] + x[43])) | (p[161] < -p[160]*(p[158]*(-x[43] + y[56])/p[159] + x[43]))), (p[160]*(-p[158]/p[159] + 1), True)]))
    sp_jac_ini[423] = Piecewise(np.array([(0, (p[161] < p[160]*(p[158]*(-x[43] + y[56])/p[159] + x[43])) | (p[161] < -p[160]*(p[158]*(-x[43] + y[56])/p[159] + x[43]))), (p[160]*p[158]/p[159], True)]))

@numba.njit(cache=True)
def sp_jac_ini_eval_up(sp_jac_ini,x,y,u,p,Dt,xyup = 1):

    sp_jac_ini[0] = -p[53]
    sp_jac_ini[1] = p[43]
    sp_jac_ini[2] = -p[43]
    sp_jac_ini[4] = -p[51]/(2*p[44])
    sp_jac_ini[9] = 1/(2*p[44])
    sp_jac_ini[10] = p[51]/(2*p[44])
    sp_jac_ini[11] = -1/p[45]
    sp_jac_ini[12] = (p[49] - p[47])/p[45]
    sp_jac_ini[13] = 1/p[45]
    sp_jac_ini[14] = -1/p[46]
    sp_jac_ini[15] = (-p[50] + p[48])/p[46]
    sp_jac_ini[16] = -1/p[57]
    sp_jac_ini[17] = 1/p[57]
    sp_jac_ini[18] = p[55]*p[60] - 1
    sp_jac_ini[19] = -p[56]*p[60]
    sp_jac_ini[20] = p[60]
    sp_jac_ini[21] = -p[55]*p[60] + 1
    sp_jac_ini[22] = -1/p[62]
    sp_jac_ini[23] = 1/p[62]
    sp_jac_ini[24] = 1/p[64]
    sp_jac_ini[25] = -1/p[64]
    sp_jac_ini[27] = -p[65]
    sp_jac_ini[28] = 1/p[67]
    sp_jac_ini[29] = -1/p[67]
    sp_jac_ini[30] = -1/p[69]
    sp_jac_ini[31] = 1/p[69]
    sp_jac_ini[32] = -p[83]
    sp_jac_ini[33] = p[73]
    sp_jac_ini[34] = -p[73]
    sp_jac_ini[36] = -p[81]/(2*p[74])
    sp_jac_ini[41] = 1/(2*p[74])
    sp_jac_ini[42] = p[81]/(2*p[74])
    sp_jac_ini[43] = -1/p[75]
    sp_jac_ini[44] = (p[79] - p[77])/p[75]
    sp_jac_ini[45] = 1/p[75]
    sp_jac_ini[46] = -1/p[76]
    sp_jac_ini[47] = (-p[80] + p[78])/p[76]
    sp_jac_ini[48] = -1/p[87]
    sp_jac_ini[49] = 1/p[87]
    sp_jac_ini[50] = p[85]*p[90] - 1
    sp_jac_ini[51] = -p[86]*p[90]
    sp_jac_ini[52] = p[90]
    sp_jac_ini[53] = -p[85]*p[90] + 1
    sp_jac_ini[54] = -1/p[92]
    sp_jac_ini[55] = 1/p[92]
    sp_jac_ini[56] = 1/p[94]
    sp_jac_ini[57] = -1/p[94]
    sp_jac_ini[59] = -p[95]
    sp_jac_ini[60] = 1/p[97]
    sp_jac_ini[61] = -1/p[97]
    sp_jac_ini[62] = -1/p[99]
    sp_jac_ini[63] = 1/p[99]
    sp_jac_ini[64] = -p[113]
    sp_jac_ini[65] = p[103]
    sp_jac_ini[66] = -p[103]
    sp_jac_ini[68] = -p[111]/(2*p[104])
    sp_jac_ini[73] = 1/(2*p[104])
    sp_jac_ini[74] = p[111]/(2*p[104])
    sp_jac_ini[75] = -1/p[105]
    sp_jac_ini[76] = (p[109] - p[107])/p[105]
    sp_jac_ini[77] = 1/p[105]
    sp_jac_ini[78] = -1/p[106]
    sp_jac_ini[79] = (-p[110] + p[108])/p[106]
    sp_jac_ini[80] = -1/p[117]
    sp_jac_ini[81] = 1/p[117]
    sp_jac_ini[82] = p[115]*p[120] - 1
    sp_jac_ini[83] = -p[116]*p[120]
    sp_jac_ini[84] = p[120]
    sp_jac_ini[85] = -p[115]*p[120] + 1
    sp_jac_ini[86] = -1/p[122]
    sp_jac_ini[87] = 1/p[122]
    sp_jac_ini[88] = 1/p[124]
    sp_jac_ini[89] = -1/p[124]
    sp_jac_ini[91] = -p[125]
    sp_jac_ini[92] = 1/p[127]
    sp_jac_ini[93] = -1/p[127]
    sp_jac_ini[94] = -1/p[129]
    sp_jac_ini[95] = 1/p[129]
    sp_jac_ini[96] = -p[143]
    sp_jac_ini[97] = p[133]
    sp_jac_ini[98] = -p[133]
    sp_jac_ini[100] = -p[141]/(2*p[134])
    sp_jac_ini[105] = 1/(2*p[134])
    sp_jac_ini[106] = p[141]/(2*p[134])
    sp_jac_ini[107] = -1/p[135]
    sp_jac_ini[108] = (p[139] - p[137])/p[135]
    sp_jac_ini[109] = 1/p[135]
    sp_jac_ini[110] = -1/p[136]
    sp_jac_ini[111] = (-p[140] + p[138])/p[136]
    sp_jac_ini[112] = -1/p[147]
    sp_jac_ini[113] = 1/p[147]
    sp_jac_ini[114] = p[145]*p[150] - 1
    sp_jac_ini[115] = -p[146]*p[150]
    sp_jac_ini[116] = p[150]
    sp_jac_ini[117] = -p[145]*p[150] + 1
    sp_jac_ini[118] = -1/p[152]
    sp_jac_ini[119] = 1/p[152]
    sp_jac_ini[120] = 1/p[154]
    sp_jac_ini[121] = -1/p[154]
    sp_jac_ini[123] = -p[155]
    sp_jac_ini[124] = 1/p[157]
    sp_jac_ini[125] = -1/p[157]
    sp_jac_ini[126] = -1/p[159]
    sp_jac_ini[127] = 1/p[159]
    sp_jac_ini[133] = -p[42]/p[0]
    sp_jac_ini[138] = -p[42]/p[0]
    sp_jac_ini[143] = -p[72]/p[0]
    sp_jac_ini[148] = -p[72]/p[0]
    sp_jac_ini[153] = -p[102]/p[0]
    sp_jac_ini[158] = -p[102]/p[0]
    sp_jac_ini[163] = -p[132]/p[0]
    sp_jac_ini[168] = -p[132]/p[0]
    sp_jac_ini[265] = p[49]
    sp_jac_ini[266] = p[52]
    sp_jac_ini[271] = p[52]
    sp_jac_ini[272] = -p[50]
    sp_jac_ini[289] = -1/p[61]
    sp_jac_ini[292] = p[54]
    sp_jac_ini[293] = p[63]/p[64]
    sp_jac_ini[294] = -p[63]/p[64] + 1
    sp_jac_ini[306] = p[79]
    sp_jac_ini[307] = p[82]
    sp_jac_ini[312] = p[82]
    sp_jac_ini[313] = -p[80]
    sp_jac_ini[330] = -1/p[91]
    sp_jac_ini[333] = p[84]
    sp_jac_ini[334] = p[93]/p[94]
    sp_jac_ini[335] = -p[93]/p[94] + 1
    sp_jac_ini[347] = p[109]
    sp_jac_ini[348] = p[112]
    sp_jac_ini[353] = p[112]
    sp_jac_ini[354] = -p[110]
    sp_jac_ini[371] = -1/p[121]
    sp_jac_ini[374] = p[114]
    sp_jac_ini[375] = p[123]/p[124]
    sp_jac_ini[376] = -p[123]/p[124] + 1
    sp_jac_ini[388] = p[139]
    sp_jac_ini[389] = p[142]
    sp_jac_ini[394] = p[142]
    sp_jac_ini[395] = -p[140]
    sp_jac_ini[412] = -1/p[151]
    sp_jac_ini[415] = p[144]
    sp_jac_ini[416] = p[153]/p[154]
    sp_jac_ini[417] = -p[153]/p[154] + 1
    sp_jac_ini[425] = p[44]*p[42]/(p[44]*p[42] + p[74]*p[72] + p[104]*p[102] + p[134]*p[132])
    sp_jac_ini[426] = p[74]*p[72]/(p[44]*p[42] + p[74]*p[72] + p[104]*p[102] + p[134]*p[132])
    sp_jac_ini[427] = p[104]*p[102]/(p[44]*p[42] + p[74]*p[72] + p[104]*p[102] + p[134]*p[132])
    sp_jac_ini[428] = p[134]*p[132]/(p[44]*p[42] + p[74]*p[72] + p[104]*p[102] + p[134]*p[132])
    sp_jac_ini[430] = p[163]
    sp_jac_ini[431] = -p[162]

def sp_jac_ini_eval_num(sp_jac_ini,x,y,u,p,Dt,xyup = 1):

    sp_jac_ini[26] = -1.00000000000000e-6
    sp_jac_ini[58] = -1.00000000000000e-6
    sp_jac_ini[90] = -1.00000000000000e-6
    sp_jac_ini[122] = -1.00000000000000e-6
    sp_jac_ini[128] = -1
    sp_jac_ini[262] = -1
    sp_jac_ini[268] = -1
    sp_jac_ini[278] = -1
    sp_jac_ini[284] = -1
    sp_jac_ini[287] = -1
    sp_jac_ini[290] = 1
    sp_jac_ini[291] = -1
    sp_jac_ini[295] = -1
    sp_jac_ini[296] = 1
    sp_jac_ini[297] = -1
    sp_jac_ini[298] = -1
    sp_jac_ini[301] = -1
    sp_jac_ini[303] = -1
    sp_jac_ini[309] = -1
    sp_jac_ini[319] = -1
    sp_jac_ini[325] = -1
    sp_jac_ini[328] = -1
    sp_jac_ini[331] = 1
    sp_jac_ini[332] = -1
    sp_jac_ini[336] = -1
    sp_jac_ini[337] = 1
    sp_jac_ini[338] = -1
    sp_jac_ini[339] = -1
    sp_jac_ini[342] = -1
    sp_jac_ini[344] = -1
    sp_jac_ini[350] = -1
    sp_jac_ini[360] = -1
    sp_jac_ini[366] = -1
    sp_jac_ini[369] = -1
    sp_jac_ini[372] = 1
    sp_jac_ini[373] = -1
    sp_jac_ini[377] = -1
    sp_jac_ini[378] = 1
    sp_jac_ini[379] = -1
    sp_jac_ini[380] = -1
    sp_jac_ini[383] = -1
    sp_jac_ini[385] = -1
    sp_jac_ini[391] = -1
    sp_jac_ini[401] = -1
    sp_jac_ini[407] = -1
    sp_jac_ini[410] = -1
    sp_jac_ini[413] = 1
    sp_jac_ini[414] = -1
    sp_jac_ini[418] = -1
    sp_jac_ini[419] = 1
    sp_jac_ini[420] = -1
    sp_jac_ini[421] = -1
    sp_jac_ini[424] = -1
    sp_jac_ini[429] = -1
    sp_jac_ini[432] = -1

def sp_jac_ini_vectors():

    sp_jac_ini_ia = [0, 1, 103, 0, 1, 45, 46, 67, 68, 73, 103, 2, 67, 71, 3, 68, 4, 45, 4, 5, 71, 75, 6, 72, 6, 7, 8, 69, 1, 9, 10, 74, 11, 12, 103, 11, 12, 47, 48, 76, 77, 82, 103, 13, 76, 80, 14, 77, 15, 47, 15, 16, 80, 84, 17, 81, 17, 18, 19, 78, 12, 20, 21, 83, 22, 23, 103, 22, 23, 49, 50, 85, 86, 91, 103, 24, 85, 89, 25, 86, 26, 49, 26, 27, 89, 93, 28, 90, 28, 29, 30, 87, 23, 31, 32, 92, 33, 34, 103, 33, 34, 51, 52, 94, 95, 100, 103, 35, 94, 98, 36, 95, 37, 51, 37, 38, 98, 102, 39, 99, 39, 40, 41, 96, 34, 42, 43, 101, 103, 45, 46, 53, 54, 69, 45, 46, 53, 54, 70, 47, 48, 55, 56, 78, 47, 48, 55, 56, 79, 49, 50, 65, 66, 87, 49, 50, 65, 66, 88, 51, 52, 63, 64, 96, 51, 52, 63, 64, 97, 45, 46, 53, 54, 55, 56, 45, 46, 53, 54, 55, 56, 47, 48, 53, 54, 55, 56, 57, 58, 47, 48, 53, 54, 55, 56, 57, 58, 55, 56, 57, 58, 59, 60, 55, 56, 57, 58, 59, 60, 57, 58, 59, 60, 61, 62, 57, 58, 59, 60, 61, 62, 59, 60, 61, 62, 63, 64, 59, 60, 61, 62, 63, 64, 51, 52, 61, 62, 63, 64, 65, 66, 51, 52, 61, 62, 63, 64, 65, 66, 49, 50, 63, 64, 65, 66, 49, 50, 63, 64, 65, 66, 0, 2, 45, 46, 67, 68, 0, 3, 45, 46, 67, 68, 0, 45, 46, 67, 68, 69, 0, 45, 46, 67, 68, 70, 4, 5, 71, 75, 1, 8, 72, 104, 6, 7, 73, 1, 9, 74, 10, 74, 75, 11, 13, 47, 48, 76, 77, 11, 14, 47, 48, 76, 77, 11, 47, 48, 76, 77, 78, 11, 47, 48, 76, 77, 79, 15, 16, 80, 84, 12, 19, 81, 104, 17, 18, 82, 12, 20, 83, 21, 83, 84, 22, 24, 49, 50, 85, 86, 22, 25, 49, 50, 85, 86, 22, 49, 50, 85, 86, 87, 22, 49, 50, 85, 86, 88, 26, 27, 89, 93, 23, 30, 90, 104, 28, 29, 91, 23, 31, 92, 32, 92, 93, 33, 35, 51, 52, 94, 95, 33, 36, 51, 52, 94, 95, 33, 51, 52, 94, 95, 96, 33, 51, 52, 94, 95, 97, 37, 38, 98, 102, 34, 41, 99, 104, 39, 40, 100, 34, 42, 101, 43, 101, 102, 1, 12, 23, 34, 103, 44, 103, 104]
    sp_jac_ini_ja = [0, 3, 11, 14, 16, 18, 22, 24, 26, 28, 30, 32, 35, 43, 46, 48, 50, 54, 56, 58, 60, 62, 64, 67, 75, 78, 80, 82, 86, 88, 90, 92, 94, 96, 99, 107, 110, 112, 114, 118, 120, 122, 124, 126, 128, 129, 134, 139, 144, 149, 154, 159, 164, 169, 175, 181, 189, 197, 203, 209, 215, 221, 227, 233, 241, 249, 255, 261, 267, 273, 279, 285, 289, 293, 296, 299, 302, 308, 314, 320, 326, 330, 334, 337, 340, 343, 349, 355, 361, 367, 371, 375, 378, 381, 384, 390, 396, 402, 408, 412, 416, 419, 422, 425, 430, 433]
    sp_jac_ini_nia = 105
    sp_jac_ini_nja = 105
    return sp_jac_ini_ia, sp_jac_ini_ja, sp_jac_ini_nia, sp_jac_ini_nja 




@numba.njit(cache=True)
def jac_run_ss_eval_xy(jac_run,x,y,u,p,xyup = 1):

    jac_run[1,0] = (-y[0]*y[22]*cos(x[0] - y[1]) + y[0]*y[23]*sin(x[0] - y[1]))/(2*p[44])
    jac_run[1,45] = (-y[22]*sin(x[0] - y[1]) - y[23]*cos(x[0] - y[1]))/(2*p[44])
    jac_run[1,46] = (y[0]*y[22]*cos(x[0] - y[1]) - y[0]*y[23]*sin(x[0] - y[1]))/(2*p[44])
    jac_run[1,67] = (-2*p[52]*y[22] - y[0]*sin(x[0] - y[1]))/(2*p[44])
    jac_run[1,68] = (-2*p[52]*y[23] - y[0]*cos(x[0] - y[1]))/(2*p[44])
    jac_run[12,11] = (-y[2]*y[31]*cos(x[11] - y[3]) + y[2]*y[32]*sin(x[11] - y[3]))/(2*p[74])
    jac_run[12,47] = (-y[31]*sin(x[11] - y[3]) - y[32]*cos(x[11] - y[3]))/(2*p[74])
    jac_run[12,48] = (y[2]*y[31]*cos(x[11] - y[3]) - y[2]*y[32]*sin(x[11] - y[3]))/(2*p[74])
    jac_run[12,76] = (-2*p[82]*y[31] - y[2]*sin(x[11] - y[3]))/(2*p[74])
    jac_run[12,77] = (-2*p[82]*y[32] - y[2]*cos(x[11] - y[3]))/(2*p[74])
    jac_run[23,22] = (-y[4]*y[40]*cos(x[22] - y[5]) + y[4]*y[41]*sin(x[22] - y[5]))/(2*p[104])
    jac_run[23,49] = (-y[40]*sin(x[22] - y[5]) - y[41]*cos(x[22] - y[5]))/(2*p[104])
    jac_run[23,50] = (y[4]*y[40]*cos(x[22] - y[5]) - y[4]*y[41]*sin(x[22] - y[5]))/(2*p[104])
    jac_run[23,85] = (-2*p[112]*y[40] - y[4]*sin(x[22] - y[5]))/(2*p[104])
    jac_run[23,86] = (-2*p[112]*y[41] - y[4]*cos(x[22] - y[5]))/(2*p[104])
    jac_run[34,33] = (-y[6]*y[49]*cos(x[33] - y[7]) + y[6]*y[50]*sin(x[33] - y[7]))/(2*p[134])
    jac_run[34,51] = (-y[49]*sin(x[33] - y[7]) - y[50]*cos(x[33] - y[7]))/(2*p[134])
    jac_run[34,52] = (y[6]*y[49]*cos(x[33] - y[7]) - y[6]*y[50]*sin(x[33] - y[7]))/(2*p[134])
    jac_run[34,94] = (-2*p[142]*y[49] - y[6]*sin(x[33] - y[7]))/(2*p[134])
    jac_run[34,95] = (-2*p[142]*y[50] - y[6]*cos(x[33] - y[7]))/(2*p[134])
    jac_run[45,45] = 2*y[0]*p[1] + y[8]*(-p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9]))
    jac_run[45,46] = y[0]*y[8]*(-p[2]*cos(y[1] - y[9]) + p[1]*sin(y[1] - y[9]))
    jac_run[45,53] = y[0]*(-p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9]))
    jac_run[45,54] = y[0]*y[8]*(p[2]*cos(y[1] - y[9]) - p[1]*sin(y[1] - y[9]))
    jac_run[46,45] = 2*y[0]*(-p[2] - p[3]/2) + y[8]*(p[2]*cos(y[1] - y[9]) - p[1]*sin(y[1] - y[9]))
    jac_run[46,46] = y[0]*y[8]*(-p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9]))
    jac_run[46,53] = y[0]*(p[2]*cos(y[1] - y[9]) - p[1]*sin(y[1] - y[9]))
    jac_run[46,54] = y[0]*y[8]*(p[2]*sin(y[1] - y[9]) + p[1]*cos(y[1] - y[9]))
    jac_run[47,47] = 2*y[2]*p[4] + y[10]*(-p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11]))
    jac_run[47,48] = y[2]*y[10]*(-p[5]*cos(y[3] - y[11]) + p[4]*sin(y[3] - y[11]))
    jac_run[47,55] = y[2]*(-p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11]))
    jac_run[47,56] = y[2]*y[10]*(p[5]*cos(y[3] - y[11]) - p[4]*sin(y[3] - y[11]))
    jac_run[48,47] = 2*y[2]*(-p[5] - p[6]/2) + y[10]*(p[5]*cos(y[3] - y[11]) - p[4]*sin(y[3] - y[11]))
    jac_run[48,48] = y[2]*y[10]*(-p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11]))
    jac_run[48,55] = y[2]*(p[5]*cos(y[3] - y[11]) - p[4]*sin(y[3] - y[11]))
    jac_run[48,56] = y[2]*y[10]*(p[5]*sin(y[3] - y[11]) + p[4]*cos(y[3] - y[11]))
    jac_run[49,49] = y[20]*(p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5])) + 2*y[4]*p[7]
    jac_run[49,50] = y[20]*y[4]*(-p[8]*cos(y[21] - y[5]) - p[7]*sin(y[21] - y[5]))
    jac_run[49,65] = y[4]*(p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5]))
    jac_run[49,66] = y[20]*y[4]*(p[8]*cos(y[21] - y[5]) + p[7]*sin(y[21] - y[5]))
    jac_run[50,49] = y[20]*(p[8]*cos(y[21] - y[5]) + p[7]*sin(y[21] - y[5])) + 2*y[4]*(-p[8] - p[9]/2)
    jac_run[50,50] = y[20]*y[4]*(p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5]))
    jac_run[50,65] = y[4]*(p[8]*cos(y[21] - y[5]) + p[7]*sin(y[21] - y[5]))
    jac_run[50,66] = y[20]*y[4]*(-p[8]*sin(y[21] - y[5]) + p[7]*cos(y[21] - y[5]))
    jac_run[51,51] = y[18]*(p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7])) + 2*y[6]*p[10]
    jac_run[51,52] = y[18]*y[6]*(-p[11]*cos(y[19] - y[7]) - p[10]*sin(y[19] - y[7]))
    jac_run[51,63] = y[6]*(p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7]))
    jac_run[51,64] = y[18]*y[6]*(p[11]*cos(y[19] - y[7]) + p[10]*sin(y[19] - y[7]))
    jac_run[52,51] = y[18]*(p[11]*cos(y[19] - y[7]) + p[10]*sin(y[19] - y[7])) + 2*y[6]*(-p[11] - p[12]/2)
    jac_run[52,52] = y[18]*y[6]*(p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7]))
    jac_run[52,63] = y[6]*(p[11]*cos(y[19] - y[7]) + p[10]*sin(y[19] - y[7]))
    jac_run[52,64] = y[18]*y[6]*(-p[11]*sin(y[19] - y[7]) + p[10]*cos(y[19] - y[7]))
    jac_run[53,45] = y[8]*(p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9]))
    jac_run[53,46] = y[0]*y[8]*(p[2]*cos(y[1] - y[9]) + p[1]*sin(y[1] - y[9]))
    jac_run[53,53] = y[0]*(p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9])) + 2*y[8]*(p[1] + p[13]) + y[10]*(-p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11]))
    jac_run[53,54] = y[0]*y[8]*(-p[2]*cos(y[1] - y[9]) - p[1]*sin(y[1] - y[9])) + y[8]*y[10]*(-p[14]*cos(y[9] - y[11]) + p[13]*sin(y[9] - y[11]))
    jac_run[53,55] = y[8]*(-p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11]))
    jac_run[53,56] = y[8]*y[10]*(p[14]*cos(y[9] - y[11]) - p[13]*sin(y[9] - y[11]))
    jac_run[54,45] = y[8]*(p[2]*cos(y[1] - y[9]) + p[1]*sin(y[1] - y[9]))
    jac_run[54,46] = y[0]*y[8]*(-p[2]*sin(y[1] - y[9]) + p[1]*cos(y[1] - y[9]))
    jac_run[54,53] = y[0]*(p[2]*cos(y[1] - y[9]) + p[1]*sin(y[1] - y[9])) + 2*y[8]*(-p[2] - p[14] - p[3]/2 - p[15]/2) + y[10]*(p[14]*cos(y[9] - y[11]) - p[13]*sin(y[9] - y[11]))
    jac_run[54,54] = y[0]*y[8]*(p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9])) + y[8]*y[10]*(-p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11]))
    jac_run[54,55] = y[8]*(p[14]*cos(y[9] - y[11]) - p[13]*sin(y[9] - y[11]))
    jac_run[54,56] = y[8]*y[10]*(p[14]*sin(y[9] - y[11]) + p[13]*cos(y[9] - y[11]))
    jac_run[55,47] = y[10]*(p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11]))
    jac_run[55,48] = y[2]*y[10]*(p[5]*cos(y[3] - y[11]) + p[4]*sin(y[3] - y[11]))
    jac_run[55,53] = y[10]*(p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11]))
    jac_run[55,54] = y[8]*y[10]*(p[14]*cos(y[9] - y[11]) + p[13]*sin(y[9] - y[11]))
    jac_run[55,55] = y[2]*(p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11])) + y[8]*(p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11])) + 2*y[10]*(p[4] + p[13] + p[16]) + y[12]*(-p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13]))
    jac_run[55,56] = y[2]*y[10]*(-p[5]*cos(y[3] - y[11]) - p[4]*sin(y[3] - y[11])) + y[8]*y[10]*(-p[14]*cos(y[9] - y[11]) - p[13]*sin(y[9] - y[11])) + y[10]*y[12]*(-p[17]*cos(y[11] - y[13]) + p[16]*sin(y[11] - y[13]))
    jac_run[55,57] = y[10]*(-p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13]))
    jac_run[55,58] = y[10]*y[12]*(p[17]*cos(y[11] - y[13]) - p[16]*sin(y[11] - y[13]))
    jac_run[56,47] = y[10]*(p[5]*cos(y[3] - y[11]) + p[4]*sin(y[3] - y[11]))
    jac_run[56,48] = y[2]*y[10]*(-p[5]*sin(y[3] - y[11]) + p[4]*cos(y[3] - y[11]))
    jac_run[56,53] = y[10]*(p[14]*cos(y[9] - y[11]) + p[13]*sin(y[9] - y[11]))
    jac_run[56,54] = y[8]*y[10]*(-p[14]*sin(y[9] - y[11]) + p[13]*cos(y[9] - y[11]))
    jac_run[56,55] = y[2]*(p[5]*cos(y[3] - y[11]) + p[4]*sin(y[3] - y[11])) + y[8]*(p[14]*cos(y[9] - y[11]) + p[13]*sin(y[9] - y[11])) + 2*y[10]*(-p[5] - p[14] - p[17] - p[6]/2 - p[15]/2 - p[18]/2) + y[12]*(p[17]*cos(y[11] - y[13]) - p[16]*sin(y[11] - y[13]))
    jac_run[56,56] = y[2]*y[10]*(p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11])) + y[8]*y[10]*(p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11])) + y[10]*y[12]*(-p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13]))
    jac_run[56,57] = y[10]*(p[17]*cos(y[11] - y[13]) - p[16]*sin(y[11] - y[13]))
    jac_run[56,58] = y[10]*y[12]*(p[17]*sin(y[11] - y[13]) + p[16]*cos(y[11] - y[13]))
    jac_run[57,55] = y[12]*(p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13]))
    jac_run[57,56] = y[10]*y[12]*(p[17]*cos(y[11] - y[13]) + p[16]*sin(y[11] - y[13]))
    jac_run[57,57] = y[10]*(p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13])) + 2*y[12]*(p[16] + 2*p[19]) + y[14]*(-2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15]))
    jac_run[57,58] = y[10]*y[12]*(-p[17]*cos(y[11] - y[13]) - p[16]*sin(y[11] - y[13])) + y[12]*y[14]*(-2*p[20]*cos(y[13] - y[15]) + 2*p[19]*sin(y[13] - y[15]))
    jac_run[57,59] = y[12]*(-2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15]))
    jac_run[57,60] = y[12]*y[14]*(2*p[20]*cos(y[13] - y[15]) - 2*p[19]*sin(y[13] - y[15]))
    jac_run[58,55] = y[12]*(p[17]*cos(y[11] - y[13]) + p[16]*sin(y[11] - y[13]))
    jac_run[58,56] = y[10]*y[12]*(-p[17]*sin(y[11] - y[13]) + p[16]*cos(y[11] - y[13]))
    jac_run[58,57] = y[10]*(p[17]*cos(y[11] - y[13]) + p[16]*sin(y[11] - y[13])) + 2*y[12]*(-p[17] - 2*p[20] - p[18]/2 - p[21]) + y[14]*(2*p[20]*cos(y[13] - y[15]) - 2*p[19]*sin(y[13] - y[15]))
    jac_run[58,58] = y[10]*y[12]*(p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13])) + y[12]*y[14]*(-2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15]))
    jac_run[58,59] = y[12]*(2*p[20]*cos(y[13] - y[15]) - 2*p[19]*sin(y[13] - y[15]))
    jac_run[58,60] = y[12]*y[14]*(2*p[20]*sin(y[13] - y[15]) + 2*p[19]*cos(y[13] - y[15]))
    jac_run[59,57] = y[14]*(2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15]))
    jac_run[59,58] = y[12]*y[14]*(2*p[20]*cos(y[13] - y[15]) + 2*p[19]*sin(y[13] - y[15]))
    jac_run[59,59] = y[12]*(2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15])) + 2*y[14]*(2*p[19] + 2*p[22]) + y[16]*(-2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17]))
    jac_run[59,60] = y[12]*y[14]*(-2*p[20]*cos(y[13] - y[15]) - 2*p[19]*sin(y[13] - y[15])) + y[14]*y[16]*(-2*p[23]*cos(y[15] - y[17]) + 2*p[22]*sin(y[15] - y[17]))
    jac_run[59,61] = y[14]*(-2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17]))
    jac_run[59,62] = y[14]*y[16]*(2*p[23]*cos(y[15] - y[17]) - 2*p[22]*sin(y[15] - y[17]))
    jac_run[60,57] = y[14]*(2*p[20]*cos(y[13] - y[15]) + 2*p[19]*sin(y[13] - y[15]))
    jac_run[60,58] = y[12]*y[14]*(-2*p[20]*sin(y[13] - y[15]) + 2*p[19]*cos(y[13] - y[15]))
    jac_run[60,59] = y[12]*(2*p[20]*cos(y[13] - y[15]) + 2*p[19]*sin(y[13] - y[15])) + 2*y[14]*(-2*p[20] - 2*p[23] - p[21] - p[24]) + y[16]*(2*p[23]*cos(y[15] - y[17]) - 2*p[22]*sin(y[15] - y[17]))
    jac_run[60,60] = y[12]*y[14]*(2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15])) + y[14]*y[16]*(-2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17]))
    jac_run[60,61] = y[14]*(2*p[23]*cos(y[15] - y[17]) - 2*p[22]*sin(y[15] - y[17]))
    jac_run[60,62] = y[14]*y[16]*(2*p[23]*sin(y[15] - y[17]) + 2*p[22]*cos(y[15] - y[17]))
    jac_run[61,59] = y[16]*(2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17]))
    jac_run[61,60] = y[14]*y[16]*(2*p[23]*cos(y[15] - y[17]) + 2*p[22]*sin(y[15] - y[17]))
    jac_run[61,61] = y[18]*(p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17])) + y[14]*(2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17])) + 2*y[16]*(2*p[22] + p[25])
    jac_run[61,62] = y[18]*y[16]*(-p[26]*cos(y[19] - y[17]) - p[25]*sin(y[19] - y[17])) + y[14]*y[16]*(-2*p[23]*cos(y[15] - y[17]) - 2*p[22]*sin(y[15] - y[17]))
    jac_run[61,63] = y[16]*(p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17]))
    jac_run[61,64] = y[18]*y[16]*(p[26]*cos(y[19] - y[17]) + p[25]*sin(y[19] - y[17]))
    jac_run[62,59] = y[16]*(2*p[23]*cos(y[15] - y[17]) + 2*p[22]*sin(y[15] - y[17]))
    jac_run[62,60] = y[14]*y[16]*(-2*p[23]*sin(y[15] - y[17]) + 2*p[22]*cos(y[15] - y[17]))
    jac_run[62,61] = y[18]*(p[26]*cos(y[19] - y[17]) + p[25]*sin(y[19] - y[17])) + y[14]*(2*p[23]*cos(y[15] - y[17]) + 2*p[22]*sin(y[15] - y[17])) + 2*y[16]*(-2*p[23] - p[26] - p[24] - p[27]/2)
    jac_run[62,62] = y[18]*y[16]*(p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17])) + y[14]*y[16]*(2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17]))
    jac_run[62,63] = y[16]*(p[26]*cos(y[19] - y[17]) + p[25]*sin(y[19] - y[17]))
    jac_run[62,64] = y[18]*y[16]*(-p[26]*sin(y[19] - y[17]) + p[25]*cos(y[19] - y[17]))
    jac_run[63,51] = y[18]*(-p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7]))
    jac_run[63,52] = y[18]*y[6]*(p[11]*cos(y[19] - y[7]) - p[10]*sin(y[19] - y[7]))
    jac_run[63,61] = y[18]*(-p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17]))
    jac_run[63,62] = y[18]*y[16]*(p[26]*cos(y[19] - y[17]) - p[25]*sin(y[19] - y[17]))
    jac_run[63,63] = 2*y[18]*(p[28] + p[10] + p[25]) + y[20]*(-p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21])) + y[6]*(-p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7])) + y[16]*(-p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17]))
    jac_run[63,64] = y[18]*y[20]*(-p[29]*cos(y[19] - y[21]) + p[28]*sin(y[19] - y[21])) + y[18]*y[6]*(-p[11]*cos(y[19] - y[7]) + p[10]*sin(y[19] - y[7])) + y[18]*y[16]*(-p[26]*cos(y[19] - y[17]) + p[25]*sin(y[19] - y[17]))
    jac_run[63,65] = y[18]*(-p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21]))
    jac_run[63,66] = y[18]*y[20]*(p[29]*cos(y[19] - y[21]) - p[28]*sin(y[19] - y[21]))
    jac_run[64,51] = y[18]*(p[11]*cos(y[19] - y[7]) - p[10]*sin(y[19] - y[7]))
    jac_run[64,52] = y[18]*y[6]*(p[11]*sin(y[19] - y[7]) + p[10]*cos(y[19] - y[7]))
    jac_run[64,61] = y[18]*(p[26]*cos(y[19] - y[17]) - p[25]*sin(y[19] - y[17]))
    jac_run[64,62] = y[18]*y[16]*(p[26]*sin(y[19] - y[17]) + p[25]*cos(y[19] - y[17]))
    jac_run[64,63] = 2*y[18]*(-p[29] - p[11] - p[26] - p[30]/2 - p[12]/2 - p[27]/2) + y[20]*(p[29]*cos(y[19] - y[21]) - p[28]*sin(y[19] - y[21])) + y[6]*(p[11]*cos(y[19] - y[7]) - p[10]*sin(y[19] - y[7])) + y[16]*(p[26]*cos(y[19] - y[17]) - p[25]*sin(y[19] - y[17]))
    jac_run[64,64] = y[18]*y[20]*(-p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21])) + y[18]*y[6]*(-p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7])) + y[18]*y[16]*(-p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17]))
    jac_run[64,65] = y[18]*(p[29]*cos(y[19] - y[21]) - p[28]*sin(y[19] - y[21]))
    jac_run[64,66] = y[18]*y[20]*(p[29]*sin(y[19] - y[21]) + p[28]*cos(y[19] - y[21]))
    jac_run[65,49] = y[20]*(-p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5]))
    jac_run[65,50] = y[20]*y[4]*(p[8]*cos(y[21] - y[5]) - p[7]*sin(y[21] - y[5]))
    jac_run[65,63] = y[20]*(p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21]))
    jac_run[65,64] = y[18]*y[20]*(p[29]*cos(y[19] - y[21]) + p[28]*sin(y[19] - y[21]))
    jac_run[65,65] = y[18]*(p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21])) + 2*y[20]*(p[28] + p[7]) + y[4]*(-p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5]))
    jac_run[65,66] = y[18]*y[20]*(-p[29]*cos(y[19] - y[21]) - p[28]*sin(y[19] - y[21])) + y[20]*y[4]*(-p[8]*cos(y[21] - y[5]) + p[7]*sin(y[21] - y[5]))
    jac_run[66,49] = y[20]*(p[8]*cos(y[21] - y[5]) - p[7]*sin(y[21] - y[5]))
    jac_run[66,50] = y[20]*y[4]*(p[8]*sin(y[21] - y[5]) + p[7]*cos(y[21] - y[5]))
    jac_run[66,63] = y[20]*(p[29]*cos(y[19] - y[21]) + p[28]*sin(y[19] - y[21]))
    jac_run[66,64] = y[18]*y[20]*(-p[29]*sin(y[19] - y[21]) + p[28]*cos(y[19] - y[21]))
    jac_run[66,65] = y[18]*(p[29]*cos(y[19] - y[21]) + p[28]*sin(y[19] - y[21])) + 2*y[20]*(-p[29] - p[8] - p[30]/2 - p[9]/2) + y[4]*(p[8]*cos(y[21] - y[5]) - p[7]*sin(y[21] - y[5]))
    jac_run[66,66] = y[18]*y[20]*(p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21])) + y[20]*y[4]*(-p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5]))
    jac_run[67,0] = -y[0]*sin(x[0] - y[1])
    jac_run[67,45] = cos(x[0] - y[1])
    jac_run[67,46] = y[0]*sin(x[0] - y[1])
    jac_run[68,0] = y[0]*cos(x[0] - y[1])
    jac_run[68,45] = sin(x[0] - y[1])
    jac_run[68,46] = -y[0]*cos(x[0] - y[1])
    jac_run[69,0] = y[0]*y[22]*cos(x[0] - y[1]) - y[0]*y[23]*sin(x[0] - y[1])
    jac_run[69,45] = y[22]*sin(x[0] - y[1]) + y[23]*cos(x[0] - y[1])
    jac_run[69,46] = -y[0]*y[22]*cos(x[0] - y[1]) + y[0]*y[23]*sin(x[0] - y[1])
    jac_run[69,67] = y[0]*sin(x[0] - y[1])
    jac_run[69,68] = y[0]*cos(x[0] - y[1])
    jac_run[70,0] = -y[0]*y[22]*sin(x[0] - y[1]) - y[0]*y[23]*cos(x[0] - y[1])
    jac_run[70,45] = y[22]*cos(x[0] - y[1]) - y[23]*sin(x[0] - y[1])
    jac_run[70,46] = y[0]*y[22]*sin(x[0] - y[1]) + y[0]*y[23]*cos(x[0] - y[1])
    jac_run[70,67] = y[0]*cos(x[0] - y[1])
    jac_run[70,68] = -y[0]*sin(x[0] - y[1])
    jac_run[71,4] = Piecewise(np.array([(0, (p[58] > p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5]) | (p[59] < p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5])), (-p[55], True)]))
    jac_run[71,5] = Piecewise(np.array([(0, (p[58] > p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5]) | (p[59] < p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5])), (p[56], True)]))
    jac_run[71,75] = Piecewise(np.array([(0, (p[58] > p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5]) | (p[59] < p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5])), (p[55], True)]))
    jac_run[75,10] = Piecewise(np.array([(0, (p[71] < p[70]*(p[68]*(-x[10] + y[29])/p[69] + x[10])) | (p[71] < -p[70]*(p[68]*(-x[10] + y[29])/p[69] + x[10]))), (p[70]*(-p[68]/p[69] + 1), True)]))
    jac_run[75,74] = Piecewise(np.array([(0, (p[71] < p[70]*(p[68]*(-x[10] + y[29])/p[69] + x[10])) | (p[71] < -p[70]*(p[68]*(-x[10] + y[29])/p[69] + x[10]))), (p[70]*p[68]/p[69], True)]))
    jac_run[76,11] = -y[2]*sin(x[11] - y[3])
    jac_run[76,47] = cos(x[11] - y[3])
    jac_run[76,48] = y[2]*sin(x[11] - y[3])
    jac_run[77,11] = y[2]*cos(x[11] - y[3])
    jac_run[77,47] = sin(x[11] - y[3])
    jac_run[77,48] = -y[2]*cos(x[11] - y[3])
    jac_run[78,11] = y[2]*y[31]*cos(x[11] - y[3]) - y[2]*y[32]*sin(x[11] - y[3])
    jac_run[78,47] = y[31]*sin(x[11] - y[3]) + y[32]*cos(x[11] - y[3])
    jac_run[78,48] = -y[2]*y[31]*cos(x[11] - y[3]) + y[2]*y[32]*sin(x[11] - y[3])
    jac_run[78,76] = y[2]*sin(x[11] - y[3])
    jac_run[78,77] = y[2]*cos(x[11] - y[3])
    jac_run[79,11] = -y[2]*y[31]*sin(x[11] - y[3]) - y[2]*y[32]*cos(x[11] - y[3])
    jac_run[79,47] = y[31]*cos(x[11] - y[3]) - y[32]*sin(x[11] - y[3])
    jac_run[79,48] = y[2]*y[31]*sin(x[11] - y[3]) + y[2]*y[32]*cos(x[11] - y[3])
    jac_run[79,76] = y[2]*cos(x[11] - y[3])
    jac_run[79,77] = -y[2]*sin(x[11] - y[3])
    jac_run[80,15] = Piecewise(np.array([(0, (p[88] > p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16]) | (p[89] < p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16])), (-p[85], True)]))
    jac_run[80,16] = Piecewise(np.array([(0, (p[88] > p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16]) | (p[89] < p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16])), (p[86], True)]))
    jac_run[80,84] = Piecewise(np.array([(0, (p[88] > p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16]) | (p[89] < p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16])), (p[85], True)]))
    jac_run[84,21] = Piecewise(np.array([(0, (p[101] < p[100]*(p[98]*(-x[21] + y[38])/p[99] + x[21])) | (p[101] < -p[100]*(p[98]*(-x[21] + y[38])/p[99] + x[21]))), (p[100]*(-p[98]/p[99] + 1), True)]))
    jac_run[84,83] = Piecewise(np.array([(0, (p[101] < p[100]*(p[98]*(-x[21] + y[38])/p[99] + x[21])) | (p[101] < -p[100]*(p[98]*(-x[21] + y[38])/p[99] + x[21]))), (p[100]*p[98]/p[99], True)]))
    jac_run[85,22] = -y[4]*sin(x[22] - y[5])
    jac_run[85,49] = cos(x[22] - y[5])
    jac_run[85,50] = y[4]*sin(x[22] - y[5])
    jac_run[86,22] = y[4]*cos(x[22] - y[5])
    jac_run[86,49] = sin(x[22] - y[5])
    jac_run[86,50] = -y[4]*cos(x[22] - y[5])
    jac_run[87,22] = y[4]*y[40]*cos(x[22] - y[5]) - y[4]*y[41]*sin(x[22] - y[5])
    jac_run[87,49] = y[40]*sin(x[22] - y[5]) + y[41]*cos(x[22] - y[5])
    jac_run[87,50] = -y[4]*y[40]*cos(x[22] - y[5]) + y[4]*y[41]*sin(x[22] - y[5])
    jac_run[87,85] = y[4]*sin(x[22] - y[5])
    jac_run[87,86] = y[4]*cos(x[22] - y[5])
    jac_run[88,22] = -y[4]*y[40]*sin(x[22] - y[5]) - y[4]*y[41]*cos(x[22] - y[5])
    jac_run[88,49] = y[40]*cos(x[22] - y[5]) - y[41]*sin(x[22] - y[5])
    jac_run[88,50] = y[4]*y[40]*sin(x[22] - y[5]) + y[4]*y[41]*cos(x[22] - y[5])
    jac_run[88,85] = y[4]*cos(x[22] - y[5])
    jac_run[88,86] = -y[4]*sin(x[22] - y[5])
    jac_run[89,26] = Piecewise(np.array([(0, (p[118] > p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27]) | (p[119] < p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27])), (-p[115], True)]))
    jac_run[89,27] = Piecewise(np.array([(0, (p[118] > p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27]) | (p[119] < p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27])), (p[116], True)]))
    jac_run[89,93] = Piecewise(np.array([(0, (p[118] > p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27]) | (p[119] < p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27])), (p[115], True)]))
    jac_run[93,32] = Piecewise(np.array([(0, (p[131] < p[130]*(p[128]*(-x[32] + y[47])/p[129] + x[32])) | (p[131] < -p[130]*(p[128]*(-x[32] + y[47])/p[129] + x[32]))), (p[130]*(-p[128]/p[129] + 1), True)]))
    jac_run[93,92] = Piecewise(np.array([(0, (p[131] < p[130]*(p[128]*(-x[32] + y[47])/p[129] + x[32])) | (p[131] < -p[130]*(p[128]*(-x[32] + y[47])/p[129] + x[32]))), (p[130]*p[128]/p[129], True)]))
    jac_run[94,33] = -y[6]*sin(x[33] - y[7])
    jac_run[94,51] = cos(x[33] - y[7])
    jac_run[94,52] = y[6]*sin(x[33] - y[7])
    jac_run[95,33] = y[6]*cos(x[33] - y[7])
    jac_run[95,51] = sin(x[33] - y[7])
    jac_run[95,52] = -y[6]*cos(x[33] - y[7])
    jac_run[96,33] = y[6]*y[49]*cos(x[33] - y[7]) - y[6]*y[50]*sin(x[33] - y[7])
    jac_run[96,51] = y[49]*sin(x[33] - y[7]) + y[50]*cos(x[33] - y[7])
    jac_run[96,52] = -y[6]*y[49]*cos(x[33] - y[7]) + y[6]*y[50]*sin(x[33] - y[7])
    jac_run[96,94] = y[6]*sin(x[33] - y[7])
    jac_run[96,95] = y[6]*cos(x[33] - y[7])
    jac_run[97,33] = -y[6]*y[49]*sin(x[33] - y[7]) - y[6]*y[50]*cos(x[33] - y[7])
    jac_run[97,51] = y[49]*cos(x[33] - y[7]) - y[50]*sin(x[33] - y[7])
    jac_run[97,52] = y[6]*y[49]*sin(x[33] - y[7]) + y[6]*y[50]*cos(x[33] - y[7])
    jac_run[97,94] = y[6]*cos(x[33] - y[7])
    jac_run[97,95] = -y[6]*sin(x[33] - y[7])
    jac_run[98,37] = Piecewise(np.array([(0, (p[148] > p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38]) | (p[149] < p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38])), (-p[145], True)]))
    jac_run[98,38] = Piecewise(np.array([(0, (p[148] > p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38]) | (p[149] < p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38])), (p[146], True)]))
    jac_run[98,102] = Piecewise(np.array([(0, (p[148] > p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38]) | (p[149] < p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38])), (p[145], True)]))
    jac_run[102,43] = Piecewise(np.array([(0, (p[161] < p[160]*(p[158]*(-x[43] + y[56])/p[159] + x[43])) | (p[161] < -p[160]*(p[158]*(-x[43] + y[56])/p[159] + x[43]))), (p[160]*(-p[158]/p[159] + 1), True)]))
    jac_run[102,101] = Piecewise(np.array([(0, (p[161] < p[160]*(p[158]*(-x[43] + y[56])/p[159] + x[43])) | (p[161] < -p[160]*(p[158]*(-x[43] + y[56])/p[159] + x[43]))), (p[160]*p[158]/p[159], True)]))

@numba.njit(cache=True)
def jac_run_ss_eval_up(jac_run,x,y,u,p,xyup = 1):

    jac_run[0,0] = -p[53]
    jac_run[0,1] = p[43]
    jac_run[0,103] = -p[43]
    jac_run[1,1] = -p[51]/(2*p[44])
    jac_run[1,73] = 1/(2*p[44])
    jac_run[1,103] = p[51]/(2*p[44])
    jac_run[2,2] = -1/p[45]
    jac_run[2,67] = (p[49] - p[47])/p[45]
    jac_run[2,71] = 1/p[45]
    jac_run[3,3] = -1/p[46]
    jac_run[3,68] = (-p[50] + p[48])/p[46]
    jac_run[4,4] = -1/p[57]
    jac_run[4,45] = 1/p[57]
    jac_run[5,4] = p[55]*p[60] - 1
    jac_run[5,5] = -p[56]*p[60]
    jac_run[5,71] = p[60]
    jac_run[5,75] = -p[55]*p[60] + 1
    jac_run[6,6] = -1/p[62]
    jac_run[6,72] = 1/p[62]
    jac_run[7,6] = 1/p[64]
    jac_run[7,7] = -1/p[64]
    jac_run[8,69] = -p[65]
    jac_run[9,1] = 1/p[67]
    jac_run[9,9] = -1/p[67]
    jac_run[10,10] = -1/p[69]
    jac_run[10,74] = 1/p[69]
    jac_run[11,11] = -p[83]
    jac_run[11,12] = p[73]
    jac_run[11,103] = -p[73]
    jac_run[12,12] = -p[81]/(2*p[74])
    jac_run[12,82] = 1/(2*p[74])
    jac_run[12,103] = p[81]/(2*p[74])
    jac_run[13,13] = -1/p[75]
    jac_run[13,76] = (p[79] - p[77])/p[75]
    jac_run[13,80] = 1/p[75]
    jac_run[14,14] = -1/p[76]
    jac_run[14,77] = (-p[80] + p[78])/p[76]
    jac_run[15,15] = -1/p[87]
    jac_run[15,47] = 1/p[87]
    jac_run[16,15] = p[85]*p[90] - 1
    jac_run[16,16] = -p[86]*p[90]
    jac_run[16,80] = p[90]
    jac_run[16,84] = -p[85]*p[90] + 1
    jac_run[17,17] = -1/p[92]
    jac_run[17,81] = 1/p[92]
    jac_run[18,17] = 1/p[94]
    jac_run[18,18] = -1/p[94]
    jac_run[19,78] = -p[95]
    jac_run[20,12] = 1/p[97]
    jac_run[20,20] = -1/p[97]
    jac_run[21,21] = -1/p[99]
    jac_run[21,83] = 1/p[99]
    jac_run[22,22] = -p[113]
    jac_run[22,23] = p[103]
    jac_run[22,103] = -p[103]
    jac_run[23,23] = -p[111]/(2*p[104])
    jac_run[23,91] = 1/(2*p[104])
    jac_run[23,103] = p[111]/(2*p[104])
    jac_run[24,24] = -1/p[105]
    jac_run[24,85] = (p[109] - p[107])/p[105]
    jac_run[24,89] = 1/p[105]
    jac_run[25,25] = -1/p[106]
    jac_run[25,86] = (-p[110] + p[108])/p[106]
    jac_run[26,26] = -1/p[117]
    jac_run[26,49] = 1/p[117]
    jac_run[27,26] = p[115]*p[120] - 1
    jac_run[27,27] = -p[116]*p[120]
    jac_run[27,89] = p[120]
    jac_run[27,93] = -p[115]*p[120] + 1
    jac_run[28,28] = -1/p[122]
    jac_run[28,90] = 1/p[122]
    jac_run[29,28] = 1/p[124]
    jac_run[29,29] = -1/p[124]
    jac_run[30,87] = -p[125]
    jac_run[31,23] = 1/p[127]
    jac_run[31,31] = -1/p[127]
    jac_run[32,32] = -1/p[129]
    jac_run[32,92] = 1/p[129]
    jac_run[33,33] = -p[143]
    jac_run[33,34] = p[133]
    jac_run[33,103] = -p[133]
    jac_run[34,34] = -p[141]/(2*p[134])
    jac_run[34,100] = 1/(2*p[134])
    jac_run[34,103] = p[141]/(2*p[134])
    jac_run[35,35] = -1/p[135]
    jac_run[35,94] = (p[139] - p[137])/p[135]
    jac_run[35,98] = 1/p[135]
    jac_run[36,36] = -1/p[136]
    jac_run[36,95] = (-p[140] + p[138])/p[136]
    jac_run[37,37] = -1/p[147]
    jac_run[37,51] = 1/p[147]
    jac_run[38,37] = p[145]*p[150] - 1
    jac_run[38,38] = -p[146]*p[150]
    jac_run[38,98] = p[150]
    jac_run[38,102] = -p[145]*p[150] + 1
    jac_run[39,39] = -1/p[152]
    jac_run[39,99] = 1/p[152]
    jac_run[40,39] = 1/p[154]
    jac_run[40,40] = -1/p[154]
    jac_run[41,96] = -p[155]
    jac_run[42,34] = 1/p[157]
    jac_run[42,42] = -1/p[157]
    jac_run[43,43] = -1/p[159]
    jac_run[43,101] = 1/p[159]
    jac_run[45,69] = -p[42]/p[0]
    jac_run[46,70] = -p[42]/p[0]
    jac_run[47,78] = -p[72]/p[0]
    jac_run[48,79] = -p[72]/p[0]
    jac_run[49,87] = -p[102]/p[0]
    jac_run[50,88] = -p[102]/p[0]
    jac_run[51,96] = -p[132]/p[0]
    jac_run[52,97] = -p[132]/p[0]
    jac_run[67,67] = p[49]
    jac_run[67,68] = p[52]
    jac_run[68,67] = p[52]
    jac_run[68,68] = -p[50]
    jac_run[72,1] = -1/p[61]
    jac_run[72,104] = p[54]
    jac_run[73,6] = p[63]/p[64]
    jac_run[73,7] = -p[63]/p[64] + 1
    jac_run[76,76] = p[79]
    jac_run[76,77] = p[82]
    jac_run[77,76] = p[82]
    jac_run[77,77] = -p[80]
    jac_run[81,12] = -1/p[91]
    jac_run[81,104] = p[84]
    jac_run[82,17] = p[93]/p[94]
    jac_run[82,18] = -p[93]/p[94] + 1
    jac_run[85,85] = p[109]
    jac_run[85,86] = p[112]
    jac_run[86,85] = p[112]
    jac_run[86,86] = -p[110]
    jac_run[90,23] = -1/p[121]
    jac_run[90,104] = p[114]
    jac_run[91,28] = p[123]/p[124]
    jac_run[91,29] = -p[123]/p[124] + 1
    jac_run[94,94] = p[139]
    jac_run[94,95] = p[142]
    jac_run[95,94] = p[142]
    jac_run[95,95] = -p[140]
    jac_run[99,34] = -1/p[151]
    jac_run[99,104] = p[144]
    jac_run[100,39] = p[153]/p[154]
    jac_run[100,40] = -p[153]/p[154] + 1
    jac_run[103,1] = p[44]*p[42]/(p[44]*p[42] + p[74]*p[72] + p[104]*p[102] + p[134]*p[132])
    jac_run[103,12] = p[74]*p[72]/(p[44]*p[42] + p[74]*p[72] + p[104]*p[102] + p[134]*p[132])
    jac_run[103,23] = p[104]*p[102]/(p[44]*p[42] + p[74]*p[72] + p[104]*p[102] + p[134]*p[132])
    jac_run[103,34] = p[134]*p[132]/(p[44]*p[42] + p[74]*p[72] + p[104]*p[102] + p[134]*p[132])
    jac_run[104,44] = p[163]
    jac_run[104,103] = -p[162]

def jac_run_ss_eval_num(jac_run,x,y,u,p,xyup = 1):

    jac_run[8,8] = -1.00000000000000e-6
    jac_run[19,19] = -1.00000000000000e-6
    jac_run[30,30] = -1.00000000000000e-6
    jac_run[41,41] = -1.00000000000000e-6
    jac_run[44,103] = -1
    jac_run[67,2] = -1
    jac_run[68,3] = -1
    jac_run[69,69] = -1
    jac_run[70,70] = -1
    jac_run[71,71] = -1
    jac_run[72,8] = 1
    jac_run[72,72] = -1
    jac_run[73,73] = -1
    jac_run[74,1] = 1
    jac_run[74,9] = -1
    jac_run[74,74] = -1
    jac_run[75,75] = -1
    jac_run[76,13] = -1
    jac_run[77,14] = -1
    jac_run[78,78] = -1
    jac_run[79,79] = -1
    jac_run[80,80] = -1
    jac_run[81,19] = 1
    jac_run[81,81] = -1
    jac_run[82,82] = -1
    jac_run[83,12] = 1
    jac_run[83,20] = -1
    jac_run[83,83] = -1
    jac_run[84,84] = -1
    jac_run[85,24] = -1
    jac_run[86,25] = -1
    jac_run[87,87] = -1
    jac_run[88,88] = -1
    jac_run[89,89] = -1
    jac_run[90,30] = 1
    jac_run[90,90] = -1
    jac_run[91,91] = -1
    jac_run[92,23] = 1
    jac_run[92,31] = -1
    jac_run[92,92] = -1
    jac_run[93,93] = -1
    jac_run[94,35] = -1
    jac_run[95,36] = -1
    jac_run[96,96] = -1
    jac_run[97,97] = -1
    jac_run[98,98] = -1
    jac_run[99,41] = 1
    jac_run[99,99] = -1
    jac_run[100,100] = -1
    jac_run[101,34] = 1
    jac_run[101,42] = -1
    jac_run[101,101] = -1
    jac_run[102,102] = -1
    jac_run[103,103] = -1
    jac_run[104,104] = -1





@numba.njit(cache=True)
def jac_trap_eval_xy(jac_trap,x,y,u,p,Dt,xyup = 1):

    jac_trap[1,0] = -0.25*Dt*(-y[0]*y[22]*cos(x[0] - y[1]) + y[0]*y[23]*sin(x[0] - y[1]))/p[44]
    jac_trap[1,45] = -0.25*Dt*(-y[22]*sin(x[0] - y[1]) - y[23]*cos(x[0] - y[1]))/p[44]
    jac_trap[1,46] = -0.25*Dt*(y[0]*y[22]*cos(x[0] - y[1]) - y[0]*y[23]*sin(x[0] - y[1]))/p[44]
    jac_trap[1,67] = -0.25*Dt*(-2*p[52]*y[22] - y[0]*sin(x[0] - y[1]))/p[44]
    jac_trap[1,68] = -0.25*Dt*(-2*p[52]*y[23] - y[0]*cos(x[0] - y[1]))/p[44]
    jac_trap[12,11] = -0.25*Dt*(-y[2]*y[31]*cos(x[11] - y[3]) + y[2]*y[32]*sin(x[11] - y[3]))/p[74]
    jac_trap[12,47] = -0.25*Dt*(-y[31]*sin(x[11] - y[3]) - y[32]*cos(x[11] - y[3]))/p[74]
    jac_trap[12,48] = -0.25*Dt*(y[2]*y[31]*cos(x[11] - y[3]) - y[2]*y[32]*sin(x[11] - y[3]))/p[74]
    jac_trap[12,76] = -0.25*Dt*(-2*p[82]*y[31] - y[2]*sin(x[11] - y[3]))/p[74]
    jac_trap[12,77] = -0.25*Dt*(-2*p[82]*y[32] - y[2]*cos(x[11] - y[3]))/p[74]
    jac_trap[23,22] = -0.25*Dt*(-y[4]*y[40]*cos(x[22] - y[5]) + y[4]*y[41]*sin(x[22] - y[5]))/p[104]
    jac_trap[23,49] = -0.25*Dt*(-y[40]*sin(x[22] - y[5]) - y[41]*cos(x[22] - y[5]))/p[104]
    jac_trap[23,50] = -0.25*Dt*(y[4]*y[40]*cos(x[22] - y[5]) - y[4]*y[41]*sin(x[22] - y[5]))/p[104]
    jac_trap[23,85] = -0.25*Dt*(-2*p[112]*y[40] - y[4]*sin(x[22] - y[5]))/p[104]
    jac_trap[23,86] = -0.25*Dt*(-2*p[112]*y[41] - y[4]*cos(x[22] - y[5]))/p[104]
    jac_trap[34,33] = -0.25*Dt*(-y[6]*y[49]*cos(x[33] - y[7]) + y[6]*y[50]*sin(x[33] - y[7]))/p[134]
    jac_trap[34,51] = -0.25*Dt*(-y[49]*sin(x[33] - y[7]) - y[50]*cos(x[33] - y[7]))/p[134]
    jac_trap[34,52] = -0.25*Dt*(y[6]*y[49]*cos(x[33] - y[7]) - y[6]*y[50]*sin(x[33] - y[7]))/p[134]
    jac_trap[34,94] = -0.25*Dt*(-2*p[142]*y[49] - y[6]*sin(x[33] - y[7]))/p[134]
    jac_trap[34,95] = -0.25*Dt*(-2*p[142]*y[50] - y[6]*cos(x[33] - y[7]))/p[134]
    jac_trap[45,45] = 2*y[0]*p[1] + y[8]*(-p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9]))
    jac_trap[45,46] = y[0]*y[8]*(-p[2]*cos(y[1] - y[9]) + p[1]*sin(y[1] - y[9]))
    jac_trap[45,53] = y[0]*(-p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9]))
    jac_trap[45,54] = y[0]*y[8]*(p[2]*cos(y[1] - y[9]) - p[1]*sin(y[1] - y[9]))
    jac_trap[46,45] = 2*y[0]*(-p[2] - p[3]/2) + y[8]*(p[2]*cos(y[1] - y[9]) - p[1]*sin(y[1] - y[9]))
    jac_trap[46,46] = y[0]*y[8]*(-p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9]))
    jac_trap[46,53] = y[0]*(p[2]*cos(y[1] - y[9]) - p[1]*sin(y[1] - y[9]))
    jac_trap[46,54] = y[0]*y[8]*(p[2]*sin(y[1] - y[9]) + p[1]*cos(y[1] - y[9]))
    jac_trap[47,47] = 2*y[2]*p[4] + y[10]*(-p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11]))
    jac_trap[47,48] = y[2]*y[10]*(-p[5]*cos(y[3] - y[11]) + p[4]*sin(y[3] - y[11]))
    jac_trap[47,55] = y[2]*(-p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11]))
    jac_trap[47,56] = y[2]*y[10]*(p[5]*cos(y[3] - y[11]) - p[4]*sin(y[3] - y[11]))
    jac_trap[48,47] = 2*y[2]*(-p[5] - p[6]/2) + y[10]*(p[5]*cos(y[3] - y[11]) - p[4]*sin(y[3] - y[11]))
    jac_trap[48,48] = y[2]*y[10]*(-p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11]))
    jac_trap[48,55] = y[2]*(p[5]*cos(y[3] - y[11]) - p[4]*sin(y[3] - y[11]))
    jac_trap[48,56] = y[2]*y[10]*(p[5]*sin(y[3] - y[11]) + p[4]*cos(y[3] - y[11]))
    jac_trap[49,49] = y[20]*(p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5])) + 2*y[4]*p[7]
    jac_trap[49,50] = y[20]*y[4]*(-p[8]*cos(y[21] - y[5]) - p[7]*sin(y[21] - y[5]))
    jac_trap[49,65] = y[4]*(p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5]))
    jac_trap[49,66] = y[20]*y[4]*(p[8]*cos(y[21] - y[5]) + p[7]*sin(y[21] - y[5]))
    jac_trap[50,49] = y[20]*(p[8]*cos(y[21] - y[5]) + p[7]*sin(y[21] - y[5])) + 2*y[4]*(-p[8] - p[9]/2)
    jac_trap[50,50] = y[20]*y[4]*(p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5]))
    jac_trap[50,65] = y[4]*(p[8]*cos(y[21] - y[5]) + p[7]*sin(y[21] - y[5]))
    jac_trap[50,66] = y[20]*y[4]*(-p[8]*sin(y[21] - y[5]) + p[7]*cos(y[21] - y[5]))
    jac_trap[51,51] = y[18]*(p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7])) + 2*y[6]*p[10]
    jac_trap[51,52] = y[18]*y[6]*(-p[11]*cos(y[19] - y[7]) - p[10]*sin(y[19] - y[7]))
    jac_trap[51,63] = y[6]*(p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7]))
    jac_trap[51,64] = y[18]*y[6]*(p[11]*cos(y[19] - y[7]) + p[10]*sin(y[19] - y[7]))
    jac_trap[52,51] = y[18]*(p[11]*cos(y[19] - y[7]) + p[10]*sin(y[19] - y[7])) + 2*y[6]*(-p[11] - p[12]/2)
    jac_trap[52,52] = y[18]*y[6]*(p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7]))
    jac_trap[52,63] = y[6]*(p[11]*cos(y[19] - y[7]) + p[10]*sin(y[19] - y[7]))
    jac_trap[52,64] = y[18]*y[6]*(-p[11]*sin(y[19] - y[7]) + p[10]*cos(y[19] - y[7]))
    jac_trap[53,45] = y[8]*(p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9]))
    jac_trap[53,46] = y[0]*y[8]*(p[2]*cos(y[1] - y[9]) + p[1]*sin(y[1] - y[9]))
    jac_trap[53,53] = y[0]*(p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9])) + 2*y[8]*(p[1] + p[13]) + y[10]*(-p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11]))
    jac_trap[53,54] = y[0]*y[8]*(-p[2]*cos(y[1] - y[9]) - p[1]*sin(y[1] - y[9])) + y[8]*y[10]*(-p[14]*cos(y[9] - y[11]) + p[13]*sin(y[9] - y[11]))
    jac_trap[53,55] = y[8]*(-p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11]))
    jac_trap[53,56] = y[8]*y[10]*(p[14]*cos(y[9] - y[11]) - p[13]*sin(y[9] - y[11]))
    jac_trap[54,45] = y[8]*(p[2]*cos(y[1] - y[9]) + p[1]*sin(y[1] - y[9]))
    jac_trap[54,46] = y[0]*y[8]*(-p[2]*sin(y[1] - y[9]) + p[1]*cos(y[1] - y[9]))
    jac_trap[54,53] = y[0]*(p[2]*cos(y[1] - y[9]) + p[1]*sin(y[1] - y[9])) + 2*y[8]*(-p[2] - p[14] - p[3]/2 - p[15]/2) + y[10]*(p[14]*cos(y[9] - y[11]) - p[13]*sin(y[9] - y[11]))
    jac_trap[54,54] = y[0]*y[8]*(p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9])) + y[8]*y[10]*(-p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11]))
    jac_trap[54,55] = y[8]*(p[14]*cos(y[9] - y[11]) - p[13]*sin(y[9] - y[11]))
    jac_trap[54,56] = y[8]*y[10]*(p[14]*sin(y[9] - y[11]) + p[13]*cos(y[9] - y[11]))
    jac_trap[55,47] = y[10]*(p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11]))
    jac_trap[55,48] = y[2]*y[10]*(p[5]*cos(y[3] - y[11]) + p[4]*sin(y[3] - y[11]))
    jac_trap[55,53] = y[10]*(p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11]))
    jac_trap[55,54] = y[8]*y[10]*(p[14]*cos(y[9] - y[11]) + p[13]*sin(y[9] - y[11]))
    jac_trap[55,55] = y[2]*(p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11])) + y[8]*(p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11])) + 2*y[10]*(p[4] + p[13] + p[16]) + y[12]*(-p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13]))
    jac_trap[55,56] = y[2]*y[10]*(-p[5]*cos(y[3] - y[11]) - p[4]*sin(y[3] - y[11])) + y[8]*y[10]*(-p[14]*cos(y[9] - y[11]) - p[13]*sin(y[9] - y[11])) + y[10]*y[12]*(-p[17]*cos(y[11] - y[13]) + p[16]*sin(y[11] - y[13]))
    jac_trap[55,57] = y[10]*(-p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13]))
    jac_trap[55,58] = y[10]*y[12]*(p[17]*cos(y[11] - y[13]) - p[16]*sin(y[11] - y[13]))
    jac_trap[56,47] = y[10]*(p[5]*cos(y[3] - y[11]) + p[4]*sin(y[3] - y[11]))
    jac_trap[56,48] = y[2]*y[10]*(-p[5]*sin(y[3] - y[11]) + p[4]*cos(y[3] - y[11]))
    jac_trap[56,53] = y[10]*(p[14]*cos(y[9] - y[11]) + p[13]*sin(y[9] - y[11]))
    jac_trap[56,54] = y[8]*y[10]*(-p[14]*sin(y[9] - y[11]) + p[13]*cos(y[9] - y[11]))
    jac_trap[56,55] = y[2]*(p[5]*cos(y[3] - y[11]) + p[4]*sin(y[3] - y[11])) + y[8]*(p[14]*cos(y[9] - y[11]) + p[13]*sin(y[9] - y[11])) + 2*y[10]*(-p[5] - p[14] - p[17] - p[6]/2 - p[15]/2 - p[18]/2) + y[12]*(p[17]*cos(y[11] - y[13]) - p[16]*sin(y[11] - y[13]))
    jac_trap[56,56] = y[2]*y[10]*(p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11])) + y[8]*y[10]*(p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11])) + y[10]*y[12]*(-p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13]))
    jac_trap[56,57] = y[10]*(p[17]*cos(y[11] - y[13]) - p[16]*sin(y[11] - y[13]))
    jac_trap[56,58] = y[10]*y[12]*(p[17]*sin(y[11] - y[13]) + p[16]*cos(y[11] - y[13]))
    jac_trap[57,55] = y[12]*(p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13]))
    jac_trap[57,56] = y[10]*y[12]*(p[17]*cos(y[11] - y[13]) + p[16]*sin(y[11] - y[13]))
    jac_trap[57,57] = y[10]*(p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13])) + 2*y[12]*(p[16] + 2*p[19]) + y[14]*(-2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15]))
    jac_trap[57,58] = y[10]*y[12]*(-p[17]*cos(y[11] - y[13]) - p[16]*sin(y[11] - y[13])) + y[12]*y[14]*(-2*p[20]*cos(y[13] - y[15]) + 2*p[19]*sin(y[13] - y[15]))
    jac_trap[57,59] = y[12]*(-2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15]))
    jac_trap[57,60] = y[12]*y[14]*(2*p[20]*cos(y[13] - y[15]) - 2*p[19]*sin(y[13] - y[15]))
    jac_trap[58,55] = y[12]*(p[17]*cos(y[11] - y[13]) + p[16]*sin(y[11] - y[13]))
    jac_trap[58,56] = y[10]*y[12]*(-p[17]*sin(y[11] - y[13]) + p[16]*cos(y[11] - y[13]))
    jac_trap[58,57] = y[10]*(p[17]*cos(y[11] - y[13]) + p[16]*sin(y[11] - y[13])) + 2*y[12]*(-p[17] - 2*p[20] - p[18]/2 - p[21]) + y[14]*(2*p[20]*cos(y[13] - y[15]) - 2*p[19]*sin(y[13] - y[15]))
    jac_trap[58,58] = y[10]*y[12]*(p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13])) + y[12]*y[14]*(-2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15]))
    jac_trap[58,59] = y[12]*(2*p[20]*cos(y[13] - y[15]) - 2*p[19]*sin(y[13] - y[15]))
    jac_trap[58,60] = y[12]*y[14]*(2*p[20]*sin(y[13] - y[15]) + 2*p[19]*cos(y[13] - y[15]))
    jac_trap[59,57] = y[14]*(2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15]))
    jac_trap[59,58] = y[12]*y[14]*(2*p[20]*cos(y[13] - y[15]) + 2*p[19]*sin(y[13] - y[15]))
    jac_trap[59,59] = y[12]*(2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15])) + 2*y[14]*(2*p[19] + 2*p[22]) + y[16]*(-2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17]))
    jac_trap[59,60] = y[12]*y[14]*(-2*p[20]*cos(y[13] - y[15]) - 2*p[19]*sin(y[13] - y[15])) + y[14]*y[16]*(-2*p[23]*cos(y[15] - y[17]) + 2*p[22]*sin(y[15] - y[17]))
    jac_trap[59,61] = y[14]*(-2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17]))
    jac_trap[59,62] = y[14]*y[16]*(2*p[23]*cos(y[15] - y[17]) - 2*p[22]*sin(y[15] - y[17]))
    jac_trap[60,57] = y[14]*(2*p[20]*cos(y[13] - y[15]) + 2*p[19]*sin(y[13] - y[15]))
    jac_trap[60,58] = y[12]*y[14]*(-2*p[20]*sin(y[13] - y[15]) + 2*p[19]*cos(y[13] - y[15]))
    jac_trap[60,59] = y[12]*(2*p[20]*cos(y[13] - y[15]) + 2*p[19]*sin(y[13] - y[15])) + 2*y[14]*(-2*p[20] - 2*p[23] - p[21] - p[24]) + y[16]*(2*p[23]*cos(y[15] - y[17]) - 2*p[22]*sin(y[15] - y[17]))
    jac_trap[60,60] = y[12]*y[14]*(2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15])) + y[14]*y[16]*(-2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17]))
    jac_trap[60,61] = y[14]*(2*p[23]*cos(y[15] - y[17]) - 2*p[22]*sin(y[15] - y[17]))
    jac_trap[60,62] = y[14]*y[16]*(2*p[23]*sin(y[15] - y[17]) + 2*p[22]*cos(y[15] - y[17]))
    jac_trap[61,59] = y[16]*(2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17]))
    jac_trap[61,60] = y[14]*y[16]*(2*p[23]*cos(y[15] - y[17]) + 2*p[22]*sin(y[15] - y[17]))
    jac_trap[61,61] = y[18]*(p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17])) + y[14]*(2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17])) + 2*y[16]*(2*p[22] + p[25])
    jac_trap[61,62] = y[18]*y[16]*(-p[26]*cos(y[19] - y[17]) - p[25]*sin(y[19] - y[17])) + y[14]*y[16]*(-2*p[23]*cos(y[15] - y[17]) - 2*p[22]*sin(y[15] - y[17]))
    jac_trap[61,63] = y[16]*(p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17]))
    jac_trap[61,64] = y[18]*y[16]*(p[26]*cos(y[19] - y[17]) + p[25]*sin(y[19] - y[17]))
    jac_trap[62,59] = y[16]*(2*p[23]*cos(y[15] - y[17]) + 2*p[22]*sin(y[15] - y[17]))
    jac_trap[62,60] = y[14]*y[16]*(-2*p[23]*sin(y[15] - y[17]) + 2*p[22]*cos(y[15] - y[17]))
    jac_trap[62,61] = y[18]*(p[26]*cos(y[19] - y[17]) + p[25]*sin(y[19] - y[17])) + y[14]*(2*p[23]*cos(y[15] - y[17]) + 2*p[22]*sin(y[15] - y[17])) + 2*y[16]*(-2*p[23] - p[26] - p[24] - p[27]/2)
    jac_trap[62,62] = y[18]*y[16]*(p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17])) + y[14]*y[16]*(2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17]))
    jac_trap[62,63] = y[16]*(p[26]*cos(y[19] - y[17]) + p[25]*sin(y[19] - y[17]))
    jac_trap[62,64] = y[18]*y[16]*(-p[26]*sin(y[19] - y[17]) + p[25]*cos(y[19] - y[17]))
    jac_trap[63,51] = y[18]*(-p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7]))
    jac_trap[63,52] = y[18]*y[6]*(p[11]*cos(y[19] - y[7]) - p[10]*sin(y[19] - y[7]))
    jac_trap[63,61] = y[18]*(-p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17]))
    jac_trap[63,62] = y[18]*y[16]*(p[26]*cos(y[19] - y[17]) - p[25]*sin(y[19] - y[17]))
    jac_trap[63,63] = 2*y[18]*(p[28] + p[10] + p[25]) + y[20]*(-p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21])) + y[6]*(-p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7])) + y[16]*(-p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17]))
    jac_trap[63,64] = y[18]*y[20]*(-p[29]*cos(y[19] - y[21]) + p[28]*sin(y[19] - y[21])) + y[18]*y[6]*(-p[11]*cos(y[19] - y[7]) + p[10]*sin(y[19] - y[7])) + y[18]*y[16]*(-p[26]*cos(y[19] - y[17]) + p[25]*sin(y[19] - y[17]))
    jac_trap[63,65] = y[18]*(-p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21]))
    jac_trap[63,66] = y[18]*y[20]*(p[29]*cos(y[19] - y[21]) - p[28]*sin(y[19] - y[21]))
    jac_trap[64,51] = y[18]*(p[11]*cos(y[19] - y[7]) - p[10]*sin(y[19] - y[7]))
    jac_trap[64,52] = y[18]*y[6]*(p[11]*sin(y[19] - y[7]) + p[10]*cos(y[19] - y[7]))
    jac_trap[64,61] = y[18]*(p[26]*cos(y[19] - y[17]) - p[25]*sin(y[19] - y[17]))
    jac_trap[64,62] = y[18]*y[16]*(p[26]*sin(y[19] - y[17]) + p[25]*cos(y[19] - y[17]))
    jac_trap[64,63] = 2*y[18]*(-p[29] - p[11] - p[26] - p[30]/2 - p[12]/2 - p[27]/2) + y[20]*(p[29]*cos(y[19] - y[21]) - p[28]*sin(y[19] - y[21])) + y[6]*(p[11]*cos(y[19] - y[7]) - p[10]*sin(y[19] - y[7])) + y[16]*(p[26]*cos(y[19] - y[17]) - p[25]*sin(y[19] - y[17]))
    jac_trap[64,64] = y[18]*y[20]*(-p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21])) + y[18]*y[6]*(-p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7])) + y[18]*y[16]*(-p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17]))
    jac_trap[64,65] = y[18]*(p[29]*cos(y[19] - y[21]) - p[28]*sin(y[19] - y[21]))
    jac_trap[64,66] = y[18]*y[20]*(p[29]*sin(y[19] - y[21]) + p[28]*cos(y[19] - y[21]))
    jac_trap[65,49] = y[20]*(-p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5]))
    jac_trap[65,50] = y[20]*y[4]*(p[8]*cos(y[21] - y[5]) - p[7]*sin(y[21] - y[5]))
    jac_trap[65,63] = y[20]*(p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21]))
    jac_trap[65,64] = y[18]*y[20]*(p[29]*cos(y[19] - y[21]) + p[28]*sin(y[19] - y[21]))
    jac_trap[65,65] = y[18]*(p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21])) + 2*y[20]*(p[28] + p[7]) + y[4]*(-p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5]))
    jac_trap[65,66] = y[18]*y[20]*(-p[29]*cos(y[19] - y[21]) - p[28]*sin(y[19] - y[21])) + y[20]*y[4]*(-p[8]*cos(y[21] - y[5]) + p[7]*sin(y[21] - y[5]))
    jac_trap[66,49] = y[20]*(p[8]*cos(y[21] - y[5]) - p[7]*sin(y[21] - y[5]))
    jac_trap[66,50] = y[20]*y[4]*(p[8]*sin(y[21] - y[5]) + p[7]*cos(y[21] - y[5]))
    jac_trap[66,63] = y[20]*(p[29]*cos(y[19] - y[21]) + p[28]*sin(y[19] - y[21]))
    jac_trap[66,64] = y[18]*y[20]*(-p[29]*sin(y[19] - y[21]) + p[28]*cos(y[19] - y[21]))
    jac_trap[66,65] = y[18]*(p[29]*cos(y[19] - y[21]) + p[28]*sin(y[19] - y[21])) + 2*y[20]*(-p[29] - p[8] - p[30]/2 - p[9]/2) + y[4]*(p[8]*cos(y[21] - y[5]) - p[7]*sin(y[21] - y[5]))
    jac_trap[66,66] = y[18]*y[20]*(p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21])) + y[20]*y[4]*(-p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5]))
    jac_trap[67,0] = -y[0]*sin(x[0] - y[1])
    jac_trap[67,45] = cos(x[0] - y[1])
    jac_trap[67,46] = y[0]*sin(x[0] - y[1])
    jac_trap[68,0] = y[0]*cos(x[0] - y[1])
    jac_trap[68,45] = sin(x[0] - y[1])
    jac_trap[68,46] = -y[0]*cos(x[0] - y[1])
    jac_trap[69,0] = y[0]*y[22]*cos(x[0] - y[1]) - y[0]*y[23]*sin(x[0] - y[1])
    jac_trap[69,45] = y[22]*sin(x[0] - y[1]) + y[23]*cos(x[0] - y[1])
    jac_trap[69,46] = -y[0]*y[22]*cos(x[0] - y[1]) + y[0]*y[23]*sin(x[0] - y[1])
    jac_trap[69,67] = y[0]*sin(x[0] - y[1])
    jac_trap[69,68] = y[0]*cos(x[0] - y[1])
    jac_trap[70,0] = -y[0]*y[22]*sin(x[0] - y[1]) - y[0]*y[23]*cos(x[0] - y[1])
    jac_trap[70,45] = y[22]*cos(x[0] - y[1]) - y[23]*sin(x[0] - y[1])
    jac_trap[70,46] = y[0]*y[22]*sin(x[0] - y[1]) + y[0]*y[23]*cos(x[0] - y[1])
    jac_trap[70,67] = y[0]*cos(x[0] - y[1])
    jac_trap[70,68] = -y[0]*sin(x[0] - y[1])
    jac_trap[71,4] = Piecewise(np.array([(0, (p[58] > p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5]) | (p[59] < p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5])), (-p[55], True)]))
    jac_trap[71,5] = Piecewise(np.array([(0, (p[58] > p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5]) | (p[59] < p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5])), (p[56], True)]))
    jac_trap[71,75] = Piecewise(np.array([(0, (p[58] > p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5]) | (p[59] < p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5])), (p[55], True)]))
    jac_trap[75,10] = Piecewise(np.array([(0, (p[71] < p[70]*(p[68]*(-x[10] + y[29])/p[69] + x[10])) | (p[71] < -p[70]*(p[68]*(-x[10] + y[29])/p[69] + x[10]))), (p[70]*(-p[68]/p[69] + 1), True)]))
    jac_trap[75,74] = Piecewise(np.array([(0, (p[71] < p[70]*(p[68]*(-x[10] + y[29])/p[69] + x[10])) | (p[71] < -p[70]*(p[68]*(-x[10] + y[29])/p[69] + x[10]))), (p[70]*p[68]/p[69], True)]))
    jac_trap[76,11] = -y[2]*sin(x[11] - y[3])
    jac_trap[76,47] = cos(x[11] - y[3])
    jac_trap[76,48] = y[2]*sin(x[11] - y[3])
    jac_trap[77,11] = y[2]*cos(x[11] - y[3])
    jac_trap[77,47] = sin(x[11] - y[3])
    jac_trap[77,48] = -y[2]*cos(x[11] - y[3])
    jac_trap[78,11] = y[2]*y[31]*cos(x[11] - y[3]) - y[2]*y[32]*sin(x[11] - y[3])
    jac_trap[78,47] = y[31]*sin(x[11] - y[3]) + y[32]*cos(x[11] - y[3])
    jac_trap[78,48] = -y[2]*y[31]*cos(x[11] - y[3]) + y[2]*y[32]*sin(x[11] - y[3])
    jac_trap[78,76] = y[2]*sin(x[11] - y[3])
    jac_trap[78,77] = y[2]*cos(x[11] - y[3])
    jac_trap[79,11] = -y[2]*y[31]*sin(x[11] - y[3]) - y[2]*y[32]*cos(x[11] - y[3])
    jac_trap[79,47] = y[31]*cos(x[11] - y[3]) - y[32]*sin(x[11] - y[3])
    jac_trap[79,48] = y[2]*y[31]*sin(x[11] - y[3]) + y[2]*y[32]*cos(x[11] - y[3])
    jac_trap[79,76] = y[2]*cos(x[11] - y[3])
    jac_trap[79,77] = -y[2]*sin(x[11] - y[3])
    jac_trap[80,15] = Piecewise(np.array([(0, (p[88] > p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16]) | (p[89] < p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16])), (-p[85], True)]))
    jac_trap[80,16] = Piecewise(np.array([(0, (p[88] > p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16]) | (p[89] < p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16])), (p[86], True)]))
    jac_trap[80,84] = Piecewise(np.array([(0, (p[88] > p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16]) | (p[89] < p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16])), (p[85], True)]))
    jac_trap[84,21] = Piecewise(np.array([(0, (p[101] < p[100]*(p[98]*(-x[21] + y[38])/p[99] + x[21])) | (p[101] < -p[100]*(p[98]*(-x[21] + y[38])/p[99] + x[21]))), (p[100]*(-p[98]/p[99] + 1), True)]))
    jac_trap[84,83] = Piecewise(np.array([(0, (p[101] < p[100]*(p[98]*(-x[21] + y[38])/p[99] + x[21])) | (p[101] < -p[100]*(p[98]*(-x[21] + y[38])/p[99] + x[21]))), (p[100]*p[98]/p[99], True)]))
    jac_trap[85,22] = -y[4]*sin(x[22] - y[5])
    jac_trap[85,49] = cos(x[22] - y[5])
    jac_trap[85,50] = y[4]*sin(x[22] - y[5])
    jac_trap[86,22] = y[4]*cos(x[22] - y[5])
    jac_trap[86,49] = sin(x[22] - y[5])
    jac_trap[86,50] = -y[4]*cos(x[22] - y[5])
    jac_trap[87,22] = y[4]*y[40]*cos(x[22] - y[5]) - y[4]*y[41]*sin(x[22] - y[5])
    jac_trap[87,49] = y[40]*sin(x[22] - y[5]) + y[41]*cos(x[22] - y[5])
    jac_trap[87,50] = -y[4]*y[40]*cos(x[22] - y[5]) + y[4]*y[41]*sin(x[22] - y[5])
    jac_trap[87,85] = y[4]*sin(x[22] - y[5])
    jac_trap[87,86] = y[4]*cos(x[22] - y[5])
    jac_trap[88,22] = -y[4]*y[40]*sin(x[22] - y[5]) - y[4]*y[41]*cos(x[22] - y[5])
    jac_trap[88,49] = y[40]*cos(x[22] - y[5]) - y[41]*sin(x[22] - y[5])
    jac_trap[88,50] = y[4]*y[40]*sin(x[22] - y[5]) + y[4]*y[41]*cos(x[22] - y[5])
    jac_trap[88,85] = y[4]*cos(x[22] - y[5])
    jac_trap[88,86] = -y[4]*sin(x[22] - y[5])
    jac_trap[89,26] = Piecewise(np.array([(0, (p[118] > p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27]) | (p[119] < p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27])), (-p[115], True)]))
    jac_trap[89,27] = Piecewise(np.array([(0, (p[118] > p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27]) | (p[119] < p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27])), (p[116], True)]))
    jac_trap[89,93] = Piecewise(np.array([(0, (p[118] > p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27]) | (p[119] < p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27])), (p[115], True)]))
    jac_trap[93,32] = Piecewise(np.array([(0, (p[131] < p[130]*(p[128]*(-x[32] + y[47])/p[129] + x[32])) | (p[131] < -p[130]*(p[128]*(-x[32] + y[47])/p[129] + x[32]))), (p[130]*(-p[128]/p[129] + 1), True)]))
    jac_trap[93,92] = Piecewise(np.array([(0, (p[131] < p[130]*(p[128]*(-x[32] + y[47])/p[129] + x[32])) | (p[131] < -p[130]*(p[128]*(-x[32] + y[47])/p[129] + x[32]))), (p[130]*p[128]/p[129], True)]))
    jac_trap[94,33] = -y[6]*sin(x[33] - y[7])
    jac_trap[94,51] = cos(x[33] - y[7])
    jac_trap[94,52] = y[6]*sin(x[33] - y[7])
    jac_trap[95,33] = y[6]*cos(x[33] - y[7])
    jac_trap[95,51] = sin(x[33] - y[7])
    jac_trap[95,52] = -y[6]*cos(x[33] - y[7])
    jac_trap[96,33] = y[6]*y[49]*cos(x[33] - y[7]) - y[6]*y[50]*sin(x[33] - y[7])
    jac_trap[96,51] = y[49]*sin(x[33] - y[7]) + y[50]*cos(x[33] - y[7])
    jac_trap[96,52] = -y[6]*y[49]*cos(x[33] - y[7]) + y[6]*y[50]*sin(x[33] - y[7])
    jac_trap[96,94] = y[6]*sin(x[33] - y[7])
    jac_trap[96,95] = y[6]*cos(x[33] - y[7])
    jac_trap[97,33] = -y[6]*y[49]*sin(x[33] - y[7]) - y[6]*y[50]*cos(x[33] - y[7])
    jac_trap[97,51] = y[49]*cos(x[33] - y[7]) - y[50]*sin(x[33] - y[7])
    jac_trap[97,52] = y[6]*y[49]*sin(x[33] - y[7]) + y[6]*y[50]*cos(x[33] - y[7])
    jac_trap[97,94] = y[6]*cos(x[33] - y[7])
    jac_trap[97,95] = -y[6]*sin(x[33] - y[7])
    jac_trap[98,37] = Piecewise(np.array([(0, (p[148] > p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38]) | (p[149] < p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38])), (-p[145], True)]))
    jac_trap[98,38] = Piecewise(np.array([(0, (p[148] > p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38]) | (p[149] < p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38])), (p[146], True)]))
    jac_trap[98,102] = Piecewise(np.array([(0, (p[148] > p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38]) | (p[149] < p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38])), (p[145], True)]))
    jac_trap[102,43] = Piecewise(np.array([(0, (p[161] < p[160]*(p[158]*(-x[43] + y[56])/p[159] + x[43])) | (p[161] < -p[160]*(p[158]*(-x[43] + y[56])/p[159] + x[43]))), (p[160]*(-p[158]/p[159] + 1), True)]))
    jac_trap[102,101] = Piecewise(np.array([(0, (p[161] < p[160]*(p[158]*(-x[43] + y[56])/p[159] + x[43])) | (p[161] < -p[160]*(p[158]*(-x[43] + y[56])/p[159] + x[43]))), (p[160]*p[158]/p[159], True)]))

@numba.njit(cache=True)
def jac_trap_eval_up(jac_trap,x,y,u,p,Dt,xyup = 1):

    jac_trap[0,0] = 0.5*Dt*p[53] + 1
    jac_trap[0,1] = -0.5*Dt*p[43]
    jac_trap[0,103] = 0.5*Dt*p[43]
    jac_trap[1,1] = 0.25*p[51]*Dt/p[44] + 1
    jac_trap[1,73] = -0.25*Dt/p[44]
    jac_trap[1,103] = -0.25*p[51]*Dt/p[44]
    jac_trap[2,2] = 0.5*Dt/p[45] + 1
    jac_trap[2,67] = -0.5*Dt*(p[49] - p[47])/p[45]
    jac_trap[2,71] = -0.5*Dt/p[45]
    jac_trap[3,3] = 0.5*Dt/p[46] + 1
    jac_trap[3,68] = -0.5*Dt*(-p[50] + p[48])/p[46]
    jac_trap[4,4] = 0.5*Dt/p[57] + 1
    jac_trap[4,45] = -0.5*Dt/p[57]
    jac_trap[5,4] = -0.5*Dt*(p[55]*p[60] - 1)
    jac_trap[5,5] = 0.5*Dt*p[56]*p[60] + 1
    jac_trap[5,71] = -0.5*Dt*p[60]
    jac_trap[5,75] = -0.5*Dt*(-p[55]*p[60] + 1)
    jac_trap[6,6] = 0.5*Dt/p[62] + 1
    jac_trap[6,72] = -0.5*Dt/p[62]
    jac_trap[7,6] = -0.5*Dt/p[64]
    jac_trap[7,7] = 0.5*Dt/p[64] + 1
    jac_trap[8,8] = 5.0e-7*Dt + 1
    jac_trap[8,69] = 0.5*Dt*p[65]
    jac_trap[9,1] = -0.5*Dt/p[67]
    jac_trap[9,9] = 0.5*Dt/p[67] + 1
    jac_trap[10,10] = 0.5*Dt/p[69] + 1
    jac_trap[10,74] = -0.5*Dt/p[69]
    jac_trap[11,11] = 0.5*Dt*p[83] + 1
    jac_trap[11,12] = -0.5*Dt*p[73]
    jac_trap[11,103] = 0.5*Dt*p[73]
    jac_trap[12,12] = 0.25*p[81]*Dt/p[74] + 1
    jac_trap[12,82] = -0.25*Dt/p[74]
    jac_trap[12,103] = -0.25*p[81]*Dt/p[74]
    jac_trap[13,13] = 0.5*Dt/p[75] + 1
    jac_trap[13,76] = -0.5*Dt*(p[79] - p[77])/p[75]
    jac_trap[13,80] = -0.5*Dt/p[75]
    jac_trap[14,14] = 0.5*Dt/p[76] + 1
    jac_trap[14,77] = -0.5*Dt*(-p[80] + p[78])/p[76]
    jac_trap[15,15] = 0.5*Dt/p[87] + 1
    jac_trap[15,47] = -0.5*Dt/p[87]
    jac_trap[16,15] = -0.5*Dt*(p[85]*p[90] - 1)
    jac_trap[16,16] = 0.5*Dt*p[86]*p[90] + 1
    jac_trap[16,80] = -0.5*Dt*p[90]
    jac_trap[16,84] = -0.5*Dt*(-p[85]*p[90] + 1)
    jac_trap[17,17] = 0.5*Dt/p[92] + 1
    jac_trap[17,81] = -0.5*Dt/p[92]
    jac_trap[18,17] = -0.5*Dt/p[94]
    jac_trap[18,18] = 0.5*Dt/p[94] + 1
    jac_trap[19,19] = 5.0e-7*Dt + 1
    jac_trap[19,78] = 0.5*Dt*p[95]
    jac_trap[20,12] = -0.5*Dt/p[97]
    jac_trap[20,20] = 0.5*Dt/p[97] + 1
    jac_trap[21,21] = 0.5*Dt/p[99] + 1
    jac_trap[21,83] = -0.5*Dt/p[99]
    jac_trap[22,22] = 0.5*Dt*p[113] + 1
    jac_trap[22,23] = -0.5*Dt*p[103]
    jac_trap[22,103] = 0.5*Dt*p[103]
    jac_trap[23,23] = 0.25*p[111]*Dt/p[104] + 1
    jac_trap[23,91] = -0.25*Dt/p[104]
    jac_trap[23,103] = -0.25*p[111]*Dt/p[104]
    jac_trap[24,24] = 0.5*Dt/p[105] + 1
    jac_trap[24,85] = -0.5*Dt*(p[109] - p[107])/p[105]
    jac_trap[24,89] = -0.5*Dt/p[105]
    jac_trap[25,25] = 0.5*Dt/p[106] + 1
    jac_trap[25,86] = -0.5*Dt*(-p[110] + p[108])/p[106]
    jac_trap[26,26] = 0.5*Dt/p[117] + 1
    jac_trap[26,49] = -0.5*Dt/p[117]
    jac_trap[27,26] = -0.5*Dt*(p[115]*p[120] - 1)
    jac_trap[27,27] = 0.5*Dt*p[116]*p[120] + 1
    jac_trap[27,89] = -0.5*Dt*p[120]
    jac_trap[27,93] = -0.5*Dt*(-p[115]*p[120] + 1)
    jac_trap[28,28] = 0.5*Dt/p[122] + 1
    jac_trap[28,90] = -0.5*Dt/p[122]
    jac_trap[29,28] = -0.5*Dt/p[124]
    jac_trap[29,29] = 0.5*Dt/p[124] + 1
    jac_trap[30,30] = 5.0e-7*Dt + 1
    jac_trap[30,87] = 0.5*Dt*p[125]
    jac_trap[31,23] = -0.5*Dt/p[127]
    jac_trap[31,31] = 0.5*Dt/p[127] + 1
    jac_trap[32,32] = 0.5*Dt/p[129] + 1
    jac_trap[32,92] = -0.5*Dt/p[129]
    jac_trap[33,33] = 0.5*Dt*p[143] + 1
    jac_trap[33,34] = -0.5*Dt*p[133]
    jac_trap[33,103] = 0.5*Dt*p[133]
    jac_trap[34,34] = 0.25*p[141]*Dt/p[134] + 1
    jac_trap[34,100] = -0.25*Dt/p[134]
    jac_trap[34,103] = -0.25*p[141]*Dt/p[134]
    jac_trap[35,35] = 0.5*Dt/p[135] + 1
    jac_trap[35,94] = -0.5*Dt*(p[139] - p[137])/p[135]
    jac_trap[35,98] = -0.5*Dt/p[135]
    jac_trap[36,36] = 0.5*Dt/p[136] + 1
    jac_trap[36,95] = -0.5*Dt*(-p[140] + p[138])/p[136]
    jac_trap[37,37] = 0.5*Dt/p[147] + 1
    jac_trap[37,51] = -0.5*Dt/p[147]
    jac_trap[38,37] = -0.5*Dt*(p[145]*p[150] - 1)
    jac_trap[38,38] = 0.5*Dt*p[146]*p[150] + 1
    jac_trap[38,98] = -0.5*Dt*p[150]
    jac_trap[38,102] = -0.5*Dt*(-p[145]*p[150] + 1)
    jac_trap[39,39] = 0.5*Dt/p[152] + 1
    jac_trap[39,99] = -0.5*Dt/p[152]
    jac_trap[40,39] = -0.5*Dt/p[154]
    jac_trap[40,40] = 0.5*Dt/p[154] + 1
    jac_trap[41,41] = 5.0e-7*Dt + 1
    jac_trap[41,96] = 0.5*Dt*p[155]
    jac_trap[42,34] = -0.5*Dt/p[157]
    jac_trap[42,42] = 0.5*Dt/p[157] + 1
    jac_trap[43,43] = 0.5*Dt/p[159] + 1
    jac_trap[43,101] = -0.5*Dt/p[159]
    jac_trap[44,103] = 0.5*Dt
    jac_trap[45,69] = -p[42]/p[0]
    jac_trap[46,70] = -p[42]/p[0]
    jac_trap[47,78] = -p[72]/p[0]
    jac_trap[48,79] = -p[72]/p[0]
    jac_trap[49,87] = -p[102]/p[0]
    jac_trap[50,88] = -p[102]/p[0]
    jac_trap[51,96] = -p[132]/p[0]
    jac_trap[52,97] = -p[132]/p[0]
    jac_trap[67,67] = p[49]
    jac_trap[67,68] = p[52]
    jac_trap[68,67] = p[52]
    jac_trap[68,68] = -p[50]
    jac_trap[72,1] = -1/p[61]
    jac_trap[72,104] = p[54]
    jac_trap[73,6] = p[63]/p[64]
    jac_trap[73,7] = -p[63]/p[64] + 1
    jac_trap[76,76] = p[79]
    jac_trap[76,77] = p[82]
    jac_trap[77,76] = p[82]
    jac_trap[77,77] = -p[80]
    jac_trap[81,12] = -1/p[91]
    jac_trap[81,104] = p[84]
    jac_trap[82,17] = p[93]/p[94]
    jac_trap[82,18] = -p[93]/p[94] + 1
    jac_trap[85,85] = p[109]
    jac_trap[85,86] = p[112]
    jac_trap[86,85] = p[112]
    jac_trap[86,86] = -p[110]
    jac_trap[90,23] = -1/p[121]
    jac_trap[90,104] = p[114]
    jac_trap[91,28] = p[123]/p[124]
    jac_trap[91,29] = -p[123]/p[124] + 1
    jac_trap[94,94] = p[139]
    jac_trap[94,95] = p[142]
    jac_trap[95,94] = p[142]
    jac_trap[95,95] = -p[140]
    jac_trap[99,34] = -1/p[151]
    jac_trap[99,104] = p[144]
    jac_trap[100,39] = p[153]/p[154]
    jac_trap[100,40] = -p[153]/p[154] + 1
    jac_trap[103,1] = p[44]*p[42]/(p[44]*p[42] + p[74]*p[72] + p[104]*p[102] + p[134]*p[132])
    jac_trap[103,12] = p[74]*p[72]/(p[44]*p[42] + p[74]*p[72] + p[104]*p[102] + p[134]*p[132])
    jac_trap[103,23] = p[104]*p[102]/(p[44]*p[42] + p[74]*p[72] + p[104]*p[102] + p[134]*p[132])
    jac_trap[103,34] = p[134]*p[132]/(p[44]*p[42] + p[74]*p[72] + p[104]*p[102] + p[134]*p[132])
    jac_trap[104,44] = p[163]
    jac_trap[104,103] = -p[162]

def jac_trap_eval_num(jac_trap,x,y,u,p,Dt,xyup = 1):

    jac_trap[44,44] = 1
    jac_trap[67,2] = -1
    jac_trap[68,3] = -1
    jac_trap[69,69] = -1
    jac_trap[70,70] = -1
    jac_trap[71,71] = -1
    jac_trap[72,8] = 1
    jac_trap[72,72] = -1
    jac_trap[73,73] = -1
    jac_trap[74,1] = 1
    jac_trap[74,9] = -1
    jac_trap[74,74] = -1
    jac_trap[75,75] = -1
    jac_trap[76,13] = -1
    jac_trap[77,14] = -1
    jac_trap[78,78] = -1
    jac_trap[79,79] = -1
    jac_trap[80,80] = -1
    jac_trap[81,19] = 1
    jac_trap[81,81] = -1
    jac_trap[82,82] = -1
    jac_trap[83,12] = 1
    jac_trap[83,20] = -1
    jac_trap[83,83] = -1
    jac_trap[84,84] = -1
    jac_trap[85,24] = -1
    jac_trap[86,25] = -1
    jac_trap[87,87] = -1
    jac_trap[88,88] = -1
    jac_trap[89,89] = -1
    jac_trap[90,30] = 1
    jac_trap[90,90] = -1
    jac_trap[91,91] = -1
    jac_trap[92,23] = 1
    jac_trap[92,31] = -1
    jac_trap[92,92] = -1
    jac_trap[93,93] = -1
    jac_trap[94,35] = -1
    jac_trap[95,36] = -1
    jac_trap[96,96] = -1
    jac_trap[97,97] = -1
    jac_trap[98,98] = -1
    jac_trap[99,41] = 1
    jac_trap[99,99] = -1
    jac_trap[100,100] = -1
    jac_trap[101,34] = 1
    jac_trap[101,42] = -1
    jac_trap[101,101] = -1
    jac_trap[102,102] = -1
    jac_trap[103,103] = -1
    jac_trap[104,104] = -1





@numba.njit(cache=True)
def h_eval(h_run,x,y,u,p,xyup = 1):


    h_run[0] = y[0]
    h_run[1] = y[2]
    h_run[2] = y[4]
    h_run[3] = y[6]
    h_run[4] = y[8]
    h_run[5] = y[10]
    h_run[6] = y[12]
    h_run[7] = y[14]
    h_run[8] = y[16]
    h_run[9] = y[18]
    h_run[10] = y[20]
    h_run[11] = y[22]*(p[52]*y[22] + y[0]*sin(x[0] - y[1])) + y[23]*(p[52]*y[23] + y[0]*cos(x[0] - y[1]))
    h_run[12] = y[31]*(p[82]*y[31] + y[2]*sin(x[11] - y[3])) + y[32]*(p[82]*y[32] + y[2]*cos(x[11] - y[3]))
    h_run[13] = y[40]*(p[112]*y[40] + y[4]*sin(x[22] - y[5])) + y[41]*(p[112]*y[41] + y[4]*cos(x[22] - y[5]))
    h_run[14] = y[49]*(p[142]*y[49] + y[6]*sin(x[33] - y[7])) + y[50]*(p[142]*y[50] + y[6]*cos(x[33] - y[7]))






@numba.njit(cache=True)
def Piecewise(arg):
    out = arg[0][1]
    N = len(arg)
    for it in range(N-1,-1,-1):
        if arg[it][1]: out = arg[it][0]
    return out

@numba.njit(cache=True)
def ITE(arg):
    out = arg[0][1]
    N = len(arg)
    for it in range(N-1,-1,-1):
        if arg[it][1]: out = arg[it][0]
    return out


@numba.njit(cache=True)
def Abs(x):
    return np.abs(x)


@numba.njit(cache=True)
def ini_dae_jacobian_numba(struct,x):
    N_x = struct[0].N_x
    N_y = struct[0].N_y
    struct[0].x[:,0] = x[0:N_x]
    struct[0].y_ini[:,0] = x[N_x:(N_x+N_y)]

    ini(struct,10)
    ini(struct,11) 

    for row,col in zip(struct[0].Fx_ini_rows,struct[0].Fx_ini_cols):
        struct[0].Ac_ini[row,col] = struct[0].Fx_ini[row,col]
    for row,col in zip(struct[0].Fy_ini_rows,struct[0].Fy_ini_cols):
        struct[0].Ac_ini[row,col+N_x] = struct[0].Fy_ini[row,col]
    for row,col in zip(struct[0].Gx_ini_rows,struct[0].Gx_ini_cols):
        struct[0].Ac_ini[row+N_x,col] = struct[0].Gx_ini[row,col]
    for row,col in zip(struct[0].Gy_ini_rows,struct[0].Gy_ini_cols):
        struct[0].Ac_ini[row+N_x,col+N_x] = struct[0].Gy_ini[row,col]
        

@numba.njit(cache=True)
def ini_dae_problem(struct,x):
    N_x = struct[0].N_x
    N_y = struct[0].N_y
    struct[0].x[:,0] = x[0:N_x]
    struct[0].y_ini[:,0] = x[N_x:(N_x+N_y)]

    ini(struct,2)
    ini(struct,3) 
    struct[0].fg[:N_x,:] = struct[0].f[:]
    struct[0].fg[N_x:,:] = struct[0].g[:]    
        
@numba.njit(cache=True)
def ssate(struct,xy):
    for it in range(100):
        ini_dae_jacobian_numba(struct,xy[:,0])
        ini_dae_problem(struct,xy[:,0])
        xy[:] += np.linalg.solve(struct[0].Ac_ini,-struct[0].fg)
        if np.max(np.abs(struct[0].fg[:,0]))<1e-8: break
    N_x = struct[0].N_x
    struct[0].x[:,0] = xy[:N_x,0]
    struct[0].y_ini[:,0] = xy[N_x:,0]
    return xy,it






@numba.njit(cache=True)
def sp_jac_trap_eval_xy(sp_jac_trap,x,y,u,p,Dt,xyup = 1):

    sp_jac_trap[3] = -0.25*Dt*(-y[0]*y[22]*cos(x[0] - y[1]) + y[0]*y[23]*sin(x[0] - y[1]))/p[44]
    sp_jac_trap[5] = -0.25*Dt*(-y[22]*sin(x[0] - y[1]) - y[23]*cos(x[0] - y[1]))/p[44]
    sp_jac_trap[6] = -0.25*Dt*(y[0]*y[22]*cos(x[0] - y[1]) - y[0]*y[23]*sin(x[0] - y[1]))/p[44]
    sp_jac_trap[7] = -0.25*Dt*(-2*p[52]*y[22] - y[0]*sin(x[0] - y[1]))/p[44]
    sp_jac_trap[8] = -0.25*Dt*(-2*p[52]*y[23] - y[0]*cos(x[0] - y[1]))/p[44]
    sp_jac_trap[35] = -0.25*Dt*(-y[2]*y[31]*cos(x[11] - y[3]) + y[2]*y[32]*sin(x[11] - y[3]))/p[74]
    sp_jac_trap[37] = -0.25*Dt*(-y[31]*sin(x[11] - y[3]) - y[32]*cos(x[11] - y[3]))/p[74]
    sp_jac_trap[38] = -0.25*Dt*(y[2]*y[31]*cos(x[11] - y[3]) - y[2]*y[32]*sin(x[11] - y[3]))/p[74]
    sp_jac_trap[39] = -0.25*Dt*(-2*p[82]*y[31] - y[2]*sin(x[11] - y[3]))/p[74]
    sp_jac_trap[40] = -0.25*Dt*(-2*p[82]*y[32] - y[2]*cos(x[11] - y[3]))/p[74]
    sp_jac_trap[67] = -0.25*Dt*(-y[4]*y[40]*cos(x[22] - y[5]) + y[4]*y[41]*sin(x[22] - y[5]))/p[104]
    sp_jac_trap[69] = -0.25*Dt*(-y[40]*sin(x[22] - y[5]) - y[41]*cos(x[22] - y[5]))/p[104]
    sp_jac_trap[70] = -0.25*Dt*(y[4]*y[40]*cos(x[22] - y[5]) - y[4]*y[41]*sin(x[22] - y[5]))/p[104]
    sp_jac_trap[71] = -0.25*Dt*(-2*p[112]*y[40] - y[4]*sin(x[22] - y[5]))/p[104]
    sp_jac_trap[72] = -0.25*Dt*(-2*p[112]*y[41] - y[4]*cos(x[22] - y[5]))/p[104]
    sp_jac_trap[99] = -0.25*Dt*(-y[6]*y[49]*cos(x[33] - y[7]) + y[6]*y[50]*sin(x[33] - y[7]))/p[134]
    sp_jac_trap[101] = -0.25*Dt*(-y[49]*sin(x[33] - y[7]) - y[50]*cos(x[33] - y[7]))/p[134]
    sp_jac_trap[102] = -0.25*Dt*(y[6]*y[49]*cos(x[33] - y[7]) - y[6]*y[50]*sin(x[33] - y[7]))/p[134]
    sp_jac_trap[103] = -0.25*Dt*(-2*p[142]*y[49] - y[6]*sin(x[33] - y[7]))/p[134]
    sp_jac_trap[104] = -0.25*Dt*(-2*p[142]*y[50] - y[6]*cos(x[33] - y[7]))/p[134]
    sp_jac_trap[130] = 2*y[0]*p[1] + y[8]*(-p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9]))
    sp_jac_trap[131] = y[0]*y[8]*(-p[2]*cos(y[1] - y[9]) + p[1]*sin(y[1] - y[9]))
    sp_jac_trap[132] = y[0]*(-p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9]))
    sp_jac_trap[133] = y[0]*y[8]*(p[2]*cos(y[1] - y[9]) - p[1]*sin(y[1] - y[9]))
    sp_jac_trap[135] = 2*y[0]*(-p[2] - p[3]/2) + y[8]*(p[2]*cos(y[1] - y[9]) - p[1]*sin(y[1] - y[9]))
    sp_jac_trap[136] = y[0]*y[8]*(-p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9]))
    sp_jac_trap[137] = y[0]*(p[2]*cos(y[1] - y[9]) - p[1]*sin(y[1] - y[9]))
    sp_jac_trap[138] = y[0]*y[8]*(p[2]*sin(y[1] - y[9]) + p[1]*cos(y[1] - y[9]))
    sp_jac_trap[140] = 2*y[2]*p[4] + y[10]*(-p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11]))
    sp_jac_trap[141] = y[2]*y[10]*(-p[5]*cos(y[3] - y[11]) + p[4]*sin(y[3] - y[11]))
    sp_jac_trap[142] = y[2]*(-p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11]))
    sp_jac_trap[143] = y[2]*y[10]*(p[5]*cos(y[3] - y[11]) - p[4]*sin(y[3] - y[11]))
    sp_jac_trap[145] = 2*y[2]*(-p[5] - p[6]/2) + y[10]*(p[5]*cos(y[3] - y[11]) - p[4]*sin(y[3] - y[11]))
    sp_jac_trap[146] = y[2]*y[10]*(-p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11]))
    sp_jac_trap[147] = y[2]*(p[5]*cos(y[3] - y[11]) - p[4]*sin(y[3] - y[11]))
    sp_jac_trap[148] = y[2]*y[10]*(p[5]*sin(y[3] - y[11]) + p[4]*cos(y[3] - y[11]))
    sp_jac_trap[150] = y[20]*(p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5])) + 2*y[4]*p[7]
    sp_jac_trap[151] = y[20]*y[4]*(-p[8]*cos(y[21] - y[5]) - p[7]*sin(y[21] - y[5]))
    sp_jac_trap[152] = y[4]*(p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5]))
    sp_jac_trap[153] = y[20]*y[4]*(p[8]*cos(y[21] - y[5]) + p[7]*sin(y[21] - y[5]))
    sp_jac_trap[155] = y[20]*(p[8]*cos(y[21] - y[5]) + p[7]*sin(y[21] - y[5])) + 2*y[4]*(-p[8] - p[9]/2)
    sp_jac_trap[156] = y[20]*y[4]*(p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5]))
    sp_jac_trap[157] = y[4]*(p[8]*cos(y[21] - y[5]) + p[7]*sin(y[21] - y[5]))
    sp_jac_trap[158] = y[20]*y[4]*(-p[8]*sin(y[21] - y[5]) + p[7]*cos(y[21] - y[5]))
    sp_jac_trap[160] = y[18]*(p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7])) + 2*y[6]*p[10]
    sp_jac_trap[161] = y[18]*y[6]*(-p[11]*cos(y[19] - y[7]) - p[10]*sin(y[19] - y[7]))
    sp_jac_trap[162] = y[6]*(p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7]))
    sp_jac_trap[163] = y[18]*y[6]*(p[11]*cos(y[19] - y[7]) + p[10]*sin(y[19] - y[7]))
    sp_jac_trap[165] = y[18]*(p[11]*cos(y[19] - y[7]) + p[10]*sin(y[19] - y[7])) + 2*y[6]*(-p[11] - p[12]/2)
    sp_jac_trap[166] = y[18]*y[6]*(p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7]))
    sp_jac_trap[167] = y[6]*(p[11]*cos(y[19] - y[7]) + p[10]*sin(y[19] - y[7]))
    sp_jac_trap[168] = y[18]*y[6]*(-p[11]*sin(y[19] - y[7]) + p[10]*cos(y[19] - y[7]))
    sp_jac_trap[170] = y[8]*(p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9]))
    sp_jac_trap[171] = y[0]*y[8]*(p[2]*cos(y[1] - y[9]) + p[1]*sin(y[1] - y[9]))
    sp_jac_trap[172] = y[0]*(p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9])) + 2*y[8]*(p[1] + p[13]) + y[10]*(-p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11]))
    sp_jac_trap[173] = y[0]*y[8]*(-p[2]*cos(y[1] - y[9]) - p[1]*sin(y[1] - y[9])) + y[8]*y[10]*(-p[14]*cos(y[9] - y[11]) + p[13]*sin(y[9] - y[11]))
    sp_jac_trap[174] = y[8]*(-p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11]))
    sp_jac_trap[175] = y[8]*y[10]*(p[14]*cos(y[9] - y[11]) - p[13]*sin(y[9] - y[11]))
    sp_jac_trap[176] = y[8]*(p[2]*cos(y[1] - y[9]) + p[1]*sin(y[1] - y[9]))
    sp_jac_trap[177] = y[0]*y[8]*(-p[2]*sin(y[1] - y[9]) + p[1]*cos(y[1] - y[9]))
    sp_jac_trap[178] = y[0]*(p[2]*cos(y[1] - y[9]) + p[1]*sin(y[1] - y[9])) + 2*y[8]*(-p[2] - p[14] - p[3]/2 - p[15]/2) + y[10]*(p[14]*cos(y[9] - y[11]) - p[13]*sin(y[9] - y[11]))
    sp_jac_trap[179] = y[0]*y[8]*(p[2]*sin(y[1] - y[9]) - p[1]*cos(y[1] - y[9])) + y[8]*y[10]*(-p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11]))
    sp_jac_trap[180] = y[8]*(p[14]*cos(y[9] - y[11]) - p[13]*sin(y[9] - y[11]))
    sp_jac_trap[181] = y[8]*y[10]*(p[14]*sin(y[9] - y[11]) + p[13]*cos(y[9] - y[11]))
    sp_jac_trap[182] = y[10]*(p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11]))
    sp_jac_trap[183] = y[2]*y[10]*(p[5]*cos(y[3] - y[11]) + p[4]*sin(y[3] - y[11]))
    sp_jac_trap[184] = y[10]*(p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11]))
    sp_jac_trap[185] = y[8]*y[10]*(p[14]*cos(y[9] - y[11]) + p[13]*sin(y[9] - y[11]))
    sp_jac_trap[186] = y[2]*(p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11])) + y[8]*(p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11])) + 2*y[10]*(p[4] + p[13] + p[16]) + y[12]*(-p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13]))
    sp_jac_trap[187] = y[2]*y[10]*(-p[5]*cos(y[3] - y[11]) - p[4]*sin(y[3] - y[11])) + y[8]*y[10]*(-p[14]*cos(y[9] - y[11]) - p[13]*sin(y[9] - y[11])) + y[10]*y[12]*(-p[17]*cos(y[11] - y[13]) + p[16]*sin(y[11] - y[13]))
    sp_jac_trap[188] = y[10]*(-p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13]))
    sp_jac_trap[189] = y[10]*y[12]*(p[17]*cos(y[11] - y[13]) - p[16]*sin(y[11] - y[13]))
    sp_jac_trap[190] = y[10]*(p[5]*cos(y[3] - y[11]) + p[4]*sin(y[3] - y[11]))
    sp_jac_trap[191] = y[2]*y[10]*(-p[5]*sin(y[3] - y[11]) + p[4]*cos(y[3] - y[11]))
    sp_jac_trap[192] = y[10]*(p[14]*cos(y[9] - y[11]) + p[13]*sin(y[9] - y[11]))
    sp_jac_trap[193] = y[8]*y[10]*(-p[14]*sin(y[9] - y[11]) + p[13]*cos(y[9] - y[11]))
    sp_jac_trap[194] = y[2]*(p[5]*cos(y[3] - y[11]) + p[4]*sin(y[3] - y[11])) + y[8]*(p[14]*cos(y[9] - y[11]) + p[13]*sin(y[9] - y[11])) + 2*y[10]*(-p[5] - p[14] - p[17] - p[6]/2 - p[15]/2 - p[18]/2) + y[12]*(p[17]*cos(y[11] - y[13]) - p[16]*sin(y[11] - y[13]))
    sp_jac_trap[195] = y[2]*y[10]*(p[5]*sin(y[3] - y[11]) - p[4]*cos(y[3] - y[11])) + y[8]*y[10]*(p[14]*sin(y[9] - y[11]) - p[13]*cos(y[9] - y[11])) + y[10]*y[12]*(-p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13]))
    sp_jac_trap[196] = y[10]*(p[17]*cos(y[11] - y[13]) - p[16]*sin(y[11] - y[13]))
    sp_jac_trap[197] = y[10]*y[12]*(p[17]*sin(y[11] - y[13]) + p[16]*cos(y[11] - y[13]))
    sp_jac_trap[198] = y[12]*(p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13]))
    sp_jac_trap[199] = y[10]*y[12]*(p[17]*cos(y[11] - y[13]) + p[16]*sin(y[11] - y[13]))
    sp_jac_trap[200] = y[10]*(p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13])) + 2*y[12]*(p[16] + 2*p[19]) + y[14]*(-2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15]))
    sp_jac_trap[201] = y[10]*y[12]*(-p[17]*cos(y[11] - y[13]) - p[16]*sin(y[11] - y[13])) + y[12]*y[14]*(-2*p[20]*cos(y[13] - y[15]) + 2*p[19]*sin(y[13] - y[15]))
    sp_jac_trap[202] = y[12]*(-2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15]))
    sp_jac_trap[203] = y[12]*y[14]*(2*p[20]*cos(y[13] - y[15]) - 2*p[19]*sin(y[13] - y[15]))
    sp_jac_trap[204] = y[12]*(p[17]*cos(y[11] - y[13]) + p[16]*sin(y[11] - y[13]))
    sp_jac_trap[205] = y[10]*y[12]*(-p[17]*sin(y[11] - y[13]) + p[16]*cos(y[11] - y[13]))
    sp_jac_trap[206] = y[10]*(p[17]*cos(y[11] - y[13]) + p[16]*sin(y[11] - y[13])) + 2*y[12]*(-p[17] - 2*p[20] - p[18]/2 - p[21]) + y[14]*(2*p[20]*cos(y[13] - y[15]) - 2*p[19]*sin(y[13] - y[15]))
    sp_jac_trap[207] = y[10]*y[12]*(p[17]*sin(y[11] - y[13]) - p[16]*cos(y[11] - y[13])) + y[12]*y[14]*(-2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15]))
    sp_jac_trap[208] = y[12]*(2*p[20]*cos(y[13] - y[15]) - 2*p[19]*sin(y[13] - y[15]))
    sp_jac_trap[209] = y[12]*y[14]*(2*p[20]*sin(y[13] - y[15]) + 2*p[19]*cos(y[13] - y[15]))
    sp_jac_trap[210] = y[14]*(2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15]))
    sp_jac_trap[211] = y[12]*y[14]*(2*p[20]*cos(y[13] - y[15]) + 2*p[19]*sin(y[13] - y[15]))
    sp_jac_trap[212] = y[12]*(2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15])) + 2*y[14]*(2*p[19] + 2*p[22]) + y[16]*(-2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17]))
    sp_jac_trap[213] = y[12]*y[14]*(-2*p[20]*cos(y[13] - y[15]) - 2*p[19]*sin(y[13] - y[15])) + y[14]*y[16]*(-2*p[23]*cos(y[15] - y[17]) + 2*p[22]*sin(y[15] - y[17]))
    sp_jac_trap[214] = y[14]*(-2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17]))
    sp_jac_trap[215] = y[14]*y[16]*(2*p[23]*cos(y[15] - y[17]) - 2*p[22]*sin(y[15] - y[17]))
    sp_jac_trap[216] = y[14]*(2*p[20]*cos(y[13] - y[15]) + 2*p[19]*sin(y[13] - y[15]))
    sp_jac_trap[217] = y[12]*y[14]*(-2*p[20]*sin(y[13] - y[15]) + 2*p[19]*cos(y[13] - y[15]))
    sp_jac_trap[218] = y[12]*(2*p[20]*cos(y[13] - y[15]) + 2*p[19]*sin(y[13] - y[15])) + 2*y[14]*(-2*p[20] - 2*p[23] - p[21] - p[24]) + y[16]*(2*p[23]*cos(y[15] - y[17]) - 2*p[22]*sin(y[15] - y[17]))
    sp_jac_trap[219] = y[12]*y[14]*(2*p[20]*sin(y[13] - y[15]) - 2*p[19]*cos(y[13] - y[15])) + y[14]*y[16]*(-2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17]))
    sp_jac_trap[220] = y[14]*(2*p[23]*cos(y[15] - y[17]) - 2*p[22]*sin(y[15] - y[17]))
    sp_jac_trap[221] = y[14]*y[16]*(2*p[23]*sin(y[15] - y[17]) + 2*p[22]*cos(y[15] - y[17]))
    sp_jac_trap[222] = y[16]*(2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17]))
    sp_jac_trap[223] = y[14]*y[16]*(2*p[23]*cos(y[15] - y[17]) + 2*p[22]*sin(y[15] - y[17]))
    sp_jac_trap[224] = y[18]*(p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17])) + y[14]*(2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17])) + 2*y[16]*(2*p[22] + p[25])
    sp_jac_trap[225] = y[18]*y[16]*(-p[26]*cos(y[19] - y[17]) - p[25]*sin(y[19] - y[17])) + y[14]*y[16]*(-2*p[23]*cos(y[15] - y[17]) - 2*p[22]*sin(y[15] - y[17]))
    sp_jac_trap[226] = y[16]*(p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17]))
    sp_jac_trap[227] = y[18]*y[16]*(p[26]*cos(y[19] - y[17]) + p[25]*sin(y[19] - y[17]))
    sp_jac_trap[228] = y[16]*(2*p[23]*cos(y[15] - y[17]) + 2*p[22]*sin(y[15] - y[17]))
    sp_jac_trap[229] = y[14]*y[16]*(-2*p[23]*sin(y[15] - y[17]) + 2*p[22]*cos(y[15] - y[17]))
    sp_jac_trap[230] = y[18]*(p[26]*cos(y[19] - y[17]) + p[25]*sin(y[19] - y[17])) + y[14]*(2*p[23]*cos(y[15] - y[17]) + 2*p[22]*sin(y[15] - y[17])) + 2*y[16]*(-2*p[23] - p[26] - p[24] - p[27]/2)
    sp_jac_trap[231] = y[18]*y[16]*(p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17])) + y[14]*y[16]*(2*p[23]*sin(y[15] - y[17]) - 2*p[22]*cos(y[15] - y[17]))
    sp_jac_trap[232] = y[16]*(p[26]*cos(y[19] - y[17]) + p[25]*sin(y[19] - y[17]))
    sp_jac_trap[233] = y[18]*y[16]*(-p[26]*sin(y[19] - y[17]) + p[25]*cos(y[19] - y[17]))
    sp_jac_trap[234] = y[18]*(-p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7]))
    sp_jac_trap[235] = y[18]*y[6]*(p[11]*cos(y[19] - y[7]) - p[10]*sin(y[19] - y[7]))
    sp_jac_trap[236] = y[18]*(-p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17]))
    sp_jac_trap[237] = y[18]*y[16]*(p[26]*cos(y[19] - y[17]) - p[25]*sin(y[19] - y[17]))
    sp_jac_trap[238] = 2*y[18]*(p[28] + p[10] + p[25]) + y[20]*(-p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21])) + y[6]*(-p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7])) + y[16]*(-p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17]))
    sp_jac_trap[239] = y[18]*y[20]*(-p[29]*cos(y[19] - y[21]) + p[28]*sin(y[19] - y[21])) + y[18]*y[6]*(-p[11]*cos(y[19] - y[7]) + p[10]*sin(y[19] - y[7])) + y[18]*y[16]*(-p[26]*cos(y[19] - y[17]) + p[25]*sin(y[19] - y[17]))
    sp_jac_trap[240] = y[18]*(-p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21]))
    sp_jac_trap[241] = y[18]*y[20]*(p[29]*cos(y[19] - y[21]) - p[28]*sin(y[19] - y[21]))
    sp_jac_trap[242] = y[18]*(p[11]*cos(y[19] - y[7]) - p[10]*sin(y[19] - y[7]))
    sp_jac_trap[243] = y[18]*y[6]*(p[11]*sin(y[19] - y[7]) + p[10]*cos(y[19] - y[7]))
    sp_jac_trap[244] = y[18]*(p[26]*cos(y[19] - y[17]) - p[25]*sin(y[19] - y[17]))
    sp_jac_trap[245] = y[18]*y[16]*(p[26]*sin(y[19] - y[17]) + p[25]*cos(y[19] - y[17]))
    sp_jac_trap[246] = 2*y[18]*(-p[29] - p[11] - p[26] - p[30]/2 - p[12]/2 - p[27]/2) + y[20]*(p[29]*cos(y[19] - y[21]) - p[28]*sin(y[19] - y[21])) + y[6]*(p[11]*cos(y[19] - y[7]) - p[10]*sin(y[19] - y[7])) + y[16]*(p[26]*cos(y[19] - y[17]) - p[25]*sin(y[19] - y[17]))
    sp_jac_trap[247] = y[18]*y[20]*(-p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21])) + y[18]*y[6]*(-p[11]*sin(y[19] - y[7]) - p[10]*cos(y[19] - y[7])) + y[18]*y[16]*(-p[26]*sin(y[19] - y[17]) - p[25]*cos(y[19] - y[17]))
    sp_jac_trap[248] = y[18]*(p[29]*cos(y[19] - y[21]) - p[28]*sin(y[19] - y[21]))
    sp_jac_trap[249] = y[18]*y[20]*(p[29]*sin(y[19] - y[21]) + p[28]*cos(y[19] - y[21]))
    sp_jac_trap[250] = y[20]*(-p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5]))
    sp_jac_trap[251] = y[20]*y[4]*(p[8]*cos(y[21] - y[5]) - p[7]*sin(y[21] - y[5]))
    sp_jac_trap[252] = y[20]*(p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21]))
    sp_jac_trap[253] = y[18]*y[20]*(p[29]*cos(y[19] - y[21]) + p[28]*sin(y[19] - y[21]))
    sp_jac_trap[254] = y[18]*(p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21])) + 2*y[20]*(p[28] + p[7]) + y[4]*(-p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5]))
    sp_jac_trap[255] = y[18]*y[20]*(-p[29]*cos(y[19] - y[21]) - p[28]*sin(y[19] - y[21])) + y[20]*y[4]*(-p[8]*cos(y[21] - y[5]) + p[7]*sin(y[21] - y[5]))
    sp_jac_trap[256] = y[20]*(p[8]*cos(y[21] - y[5]) - p[7]*sin(y[21] - y[5]))
    sp_jac_trap[257] = y[20]*y[4]*(p[8]*sin(y[21] - y[5]) + p[7]*cos(y[21] - y[5]))
    sp_jac_trap[258] = y[20]*(p[29]*cos(y[19] - y[21]) + p[28]*sin(y[19] - y[21]))
    sp_jac_trap[259] = y[18]*y[20]*(-p[29]*sin(y[19] - y[21]) + p[28]*cos(y[19] - y[21]))
    sp_jac_trap[260] = y[18]*(p[29]*cos(y[19] - y[21]) + p[28]*sin(y[19] - y[21])) + 2*y[20]*(-p[29] - p[8] - p[30]/2 - p[9]/2) + y[4]*(p[8]*cos(y[21] - y[5]) - p[7]*sin(y[21] - y[5]))
    sp_jac_trap[261] = y[18]*y[20]*(p[29]*sin(y[19] - y[21]) - p[28]*cos(y[19] - y[21])) + y[20]*y[4]*(-p[8]*sin(y[21] - y[5]) - p[7]*cos(y[21] - y[5]))
    sp_jac_trap[262] = -y[0]*sin(x[0] - y[1])
    sp_jac_trap[264] = cos(x[0] - y[1])
    sp_jac_trap[265] = y[0]*sin(x[0] - y[1])
    sp_jac_trap[268] = y[0]*cos(x[0] - y[1])
    sp_jac_trap[270] = sin(x[0] - y[1])
    sp_jac_trap[271] = -y[0]*cos(x[0] - y[1])
    sp_jac_trap[274] = y[0]*y[22]*cos(x[0] - y[1]) - y[0]*y[23]*sin(x[0] - y[1])
    sp_jac_trap[275] = y[22]*sin(x[0] - y[1]) + y[23]*cos(x[0] - y[1])
    sp_jac_trap[276] = -y[0]*y[22]*cos(x[0] - y[1]) + y[0]*y[23]*sin(x[0] - y[1])
    sp_jac_trap[277] = y[0]*sin(x[0] - y[1])
    sp_jac_trap[278] = y[0]*cos(x[0] - y[1])
    sp_jac_trap[280] = -y[0]*y[22]*sin(x[0] - y[1]) - y[0]*y[23]*cos(x[0] - y[1])
    sp_jac_trap[281] = y[22]*cos(x[0] - y[1]) - y[23]*sin(x[0] - y[1])
    sp_jac_trap[282] = y[0]*y[22]*sin(x[0] - y[1]) + y[0]*y[23]*cos(x[0] - y[1])
    sp_jac_trap[283] = y[0]*cos(x[0] - y[1])
    sp_jac_trap[284] = -y[0]*sin(x[0] - y[1])
    sp_jac_trap[286] = Piecewise(np.array([(0, (p[58] > p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5]) | (p[59] < p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5])), (-p[55], True)]))
    sp_jac_trap[287] = Piecewise(np.array([(0, (p[58] > p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5]) | (p[59] < p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5])), (p[56], True)]))
    sp_jac_trap[289] = Piecewise(np.array([(0, (p[58] > p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5]) | (p[59] < p[55]*(-x[4] + y[30] + u[22]) + p[56]*x[5])), (p[55], True)]))
    sp_jac_trap[300] = Piecewise(np.array([(0, (p[71] < p[70]*(p[68]*(-x[10] + y[29])/p[69] + x[10])) | (p[71] < -p[70]*(p[68]*(-x[10] + y[29])/p[69] + x[10]))), (p[70]*(-p[68]/p[69] + 1), True)]))
    sp_jac_trap[301] = Piecewise(np.array([(0, (p[71] < p[70]*(p[68]*(-x[10] + y[29])/p[69] + x[10])) | (p[71] < -p[70]*(p[68]*(-x[10] + y[29])/p[69] + x[10]))), (p[70]*p[68]/p[69], True)]))
    sp_jac_trap[303] = -y[2]*sin(x[11] - y[3])
    sp_jac_trap[305] = cos(x[11] - y[3])
    sp_jac_trap[306] = y[2]*sin(x[11] - y[3])
    sp_jac_trap[309] = y[2]*cos(x[11] - y[3])
    sp_jac_trap[311] = sin(x[11] - y[3])
    sp_jac_trap[312] = -y[2]*cos(x[11] - y[3])
    sp_jac_trap[315] = y[2]*y[31]*cos(x[11] - y[3]) - y[2]*y[32]*sin(x[11] - y[3])
    sp_jac_trap[316] = y[31]*sin(x[11] - y[3]) + y[32]*cos(x[11] - y[3])
    sp_jac_trap[317] = -y[2]*y[31]*cos(x[11] - y[3]) + y[2]*y[32]*sin(x[11] - y[3])
    sp_jac_trap[318] = y[2]*sin(x[11] - y[3])
    sp_jac_trap[319] = y[2]*cos(x[11] - y[3])
    sp_jac_trap[321] = -y[2]*y[31]*sin(x[11] - y[3]) - y[2]*y[32]*cos(x[11] - y[3])
    sp_jac_trap[322] = y[31]*cos(x[11] - y[3]) - y[32]*sin(x[11] - y[3])
    sp_jac_trap[323] = y[2]*y[31]*sin(x[11] - y[3]) + y[2]*y[32]*cos(x[11] - y[3])
    sp_jac_trap[324] = y[2]*cos(x[11] - y[3])
    sp_jac_trap[325] = -y[2]*sin(x[11] - y[3])
    sp_jac_trap[327] = Piecewise(np.array([(0, (p[88] > p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16]) | (p[89] < p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16])), (-p[85], True)]))
    sp_jac_trap[328] = Piecewise(np.array([(0, (p[88] > p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16]) | (p[89] < p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16])), (p[86], True)]))
    sp_jac_trap[330] = Piecewise(np.array([(0, (p[88] > p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16]) | (p[89] < p[85]*(-x[15] + y[39] + u[26]) + p[86]*x[16])), (p[85], True)]))
    sp_jac_trap[341] = Piecewise(np.array([(0, (p[101] < p[100]*(p[98]*(-x[21] + y[38])/p[99] + x[21])) | (p[101] < -p[100]*(p[98]*(-x[21] + y[38])/p[99] + x[21]))), (p[100]*(-p[98]/p[99] + 1), True)]))
    sp_jac_trap[342] = Piecewise(np.array([(0, (p[101] < p[100]*(p[98]*(-x[21] + y[38])/p[99] + x[21])) | (p[101] < -p[100]*(p[98]*(-x[21] + y[38])/p[99] + x[21]))), (p[100]*p[98]/p[99], True)]))
    sp_jac_trap[344] = -y[4]*sin(x[22] - y[5])
    sp_jac_trap[346] = cos(x[22] - y[5])
    sp_jac_trap[347] = y[4]*sin(x[22] - y[5])
    sp_jac_trap[350] = y[4]*cos(x[22] - y[5])
    sp_jac_trap[352] = sin(x[22] - y[5])
    sp_jac_trap[353] = -y[4]*cos(x[22] - y[5])
    sp_jac_trap[356] = y[4]*y[40]*cos(x[22] - y[5]) - y[4]*y[41]*sin(x[22] - y[5])
    sp_jac_trap[357] = y[40]*sin(x[22] - y[5]) + y[41]*cos(x[22] - y[5])
    sp_jac_trap[358] = -y[4]*y[40]*cos(x[22] - y[5]) + y[4]*y[41]*sin(x[22] - y[5])
    sp_jac_trap[359] = y[4]*sin(x[22] - y[5])
    sp_jac_trap[360] = y[4]*cos(x[22] - y[5])
    sp_jac_trap[362] = -y[4]*y[40]*sin(x[22] - y[5]) - y[4]*y[41]*cos(x[22] - y[5])
    sp_jac_trap[363] = y[40]*cos(x[22] - y[5]) - y[41]*sin(x[22] - y[5])
    sp_jac_trap[364] = y[4]*y[40]*sin(x[22] - y[5]) + y[4]*y[41]*cos(x[22] - y[5])
    sp_jac_trap[365] = y[4]*cos(x[22] - y[5])
    sp_jac_trap[366] = -y[4]*sin(x[22] - y[5])
    sp_jac_trap[368] = Piecewise(np.array([(0, (p[118] > p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27]) | (p[119] < p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27])), (-p[115], True)]))
    sp_jac_trap[369] = Piecewise(np.array([(0, (p[118] > p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27]) | (p[119] < p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27])), (p[116], True)]))
    sp_jac_trap[371] = Piecewise(np.array([(0, (p[118] > p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27]) | (p[119] < p[115]*(-x[26] + y[48] + u[30]) + p[116]*x[27])), (p[115], True)]))
    sp_jac_trap[382] = Piecewise(np.array([(0, (p[131] < p[130]*(p[128]*(-x[32] + y[47])/p[129] + x[32])) | (p[131] < -p[130]*(p[128]*(-x[32] + y[47])/p[129] + x[32]))), (p[130]*(-p[128]/p[129] + 1), True)]))
    sp_jac_trap[383] = Piecewise(np.array([(0, (p[131] < p[130]*(p[128]*(-x[32] + y[47])/p[129] + x[32])) | (p[131] < -p[130]*(p[128]*(-x[32] + y[47])/p[129] + x[32]))), (p[130]*p[128]/p[129], True)]))
    sp_jac_trap[385] = -y[6]*sin(x[33] - y[7])
    sp_jac_trap[387] = cos(x[33] - y[7])
    sp_jac_trap[388] = y[6]*sin(x[33] - y[7])
    sp_jac_trap[391] = y[6]*cos(x[33] - y[7])
    sp_jac_trap[393] = sin(x[33] - y[7])
    sp_jac_trap[394] = -y[6]*cos(x[33] - y[7])
    sp_jac_trap[397] = y[6]*y[49]*cos(x[33] - y[7]) - y[6]*y[50]*sin(x[33] - y[7])
    sp_jac_trap[398] = y[49]*sin(x[33] - y[7]) + y[50]*cos(x[33] - y[7])
    sp_jac_trap[399] = -y[6]*y[49]*cos(x[33] - y[7]) + y[6]*y[50]*sin(x[33] - y[7])
    sp_jac_trap[400] = y[6]*sin(x[33] - y[7])
    sp_jac_trap[401] = y[6]*cos(x[33] - y[7])
    sp_jac_trap[403] = -y[6]*y[49]*sin(x[33] - y[7]) - y[6]*y[50]*cos(x[33] - y[7])
    sp_jac_trap[404] = y[49]*cos(x[33] - y[7]) - y[50]*sin(x[33] - y[7])
    sp_jac_trap[405] = y[6]*y[49]*sin(x[33] - y[7]) + y[6]*y[50]*cos(x[33] - y[7])
    sp_jac_trap[406] = y[6]*cos(x[33] - y[7])
    sp_jac_trap[407] = -y[6]*sin(x[33] - y[7])
    sp_jac_trap[409] = Piecewise(np.array([(0, (p[148] > p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38]) | (p[149] < p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38])), (-p[145], True)]))
    sp_jac_trap[410] = Piecewise(np.array([(0, (p[148] > p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38]) | (p[149] < p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38])), (p[146], True)]))
    sp_jac_trap[412] = Piecewise(np.array([(0, (p[148] > p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38]) | (p[149] < p[145]*(-x[37] + y[57] + u[34]) + p[146]*x[38])), (p[145], True)]))
    sp_jac_trap[423] = Piecewise(np.array([(0, (p[161] < p[160]*(p[158]*(-x[43] + y[56])/p[159] + x[43])) | (p[161] < -p[160]*(p[158]*(-x[43] + y[56])/p[159] + x[43]))), (p[160]*(-p[158]/p[159] + 1), True)]))
    sp_jac_trap[424] = Piecewise(np.array([(0, (p[161] < p[160]*(p[158]*(-x[43] + y[56])/p[159] + x[43])) | (p[161] < -p[160]*(p[158]*(-x[43] + y[56])/p[159] + x[43]))), (p[160]*p[158]/p[159], True)]))

@numba.njit(cache=True)
def sp_jac_trap_eval_up(sp_jac_trap,x,y,u,p,Dt,xyup = 1):

    sp_jac_trap[0] = 0.5*Dt*p[53] + 1
    sp_jac_trap[1] = -0.5*Dt*p[43]
    sp_jac_trap[2] = 0.5*Dt*p[43]
    sp_jac_trap[4] = 0.25*p[51]*Dt/p[44] + 1
    sp_jac_trap[9] = -0.25*Dt/p[44]
    sp_jac_trap[10] = -0.25*p[51]*Dt/p[44]
    sp_jac_trap[11] = 0.5*Dt/p[45] + 1
    sp_jac_trap[12] = -0.5*Dt*(p[49] - p[47])/p[45]
    sp_jac_trap[13] = -0.5*Dt/p[45]
    sp_jac_trap[14] = 0.5*Dt/p[46] + 1
    sp_jac_trap[15] = -0.5*Dt*(-p[50] + p[48])/p[46]
    sp_jac_trap[16] = 0.5*Dt/p[57] + 1
    sp_jac_trap[17] = -0.5*Dt/p[57]
    sp_jac_trap[18] = -0.5*Dt*(p[55]*p[60] - 1)
    sp_jac_trap[19] = 0.5*Dt*p[56]*p[60] + 1
    sp_jac_trap[20] = -0.5*Dt*p[60]
    sp_jac_trap[21] = -0.5*Dt*(-p[55]*p[60] + 1)
    sp_jac_trap[22] = 0.5*Dt/p[62] + 1
    sp_jac_trap[23] = -0.5*Dt/p[62]
    sp_jac_trap[24] = -0.5*Dt/p[64]
    sp_jac_trap[25] = 0.5*Dt/p[64] + 1
    sp_jac_trap[26] = 5.0e-7*Dt + 1
    sp_jac_trap[27] = 0.5*Dt*p[65]
    sp_jac_trap[28] = -0.5*Dt/p[67]
    sp_jac_trap[29] = 0.5*Dt/p[67] + 1
    sp_jac_trap[30] = 0.5*Dt/p[69] + 1
    sp_jac_trap[31] = -0.5*Dt/p[69]
    sp_jac_trap[32] = 0.5*Dt*p[83] + 1
    sp_jac_trap[33] = -0.5*Dt*p[73]
    sp_jac_trap[34] = 0.5*Dt*p[73]
    sp_jac_trap[36] = 0.25*p[81]*Dt/p[74] + 1
    sp_jac_trap[41] = -0.25*Dt/p[74]
    sp_jac_trap[42] = -0.25*p[81]*Dt/p[74]
    sp_jac_trap[43] = 0.5*Dt/p[75] + 1
    sp_jac_trap[44] = -0.5*Dt*(p[79] - p[77])/p[75]
    sp_jac_trap[45] = -0.5*Dt/p[75]
    sp_jac_trap[46] = 0.5*Dt/p[76] + 1
    sp_jac_trap[47] = -0.5*Dt*(-p[80] + p[78])/p[76]
    sp_jac_trap[48] = 0.5*Dt/p[87] + 1
    sp_jac_trap[49] = -0.5*Dt/p[87]
    sp_jac_trap[50] = -0.5*Dt*(p[85]*p[90] - 1)
    sp_jac_trap[51] = 0.5*Dt*p[86]*p[90] + 1
    sp_jac_trap[52] = -0.5*Dt*p[90]
    sp_jac_trap[53] = -0.5*Dt*(-p[85]*p[90] + 1)
    sp_jac_trap[54] = 0.5*Dt/p[92] + 1
    sp_jac_trap[55] = -0.5*Dt/p[92]
    sp_jac_trap[56] = -0.5*Dt/p[94]
    sp_jac_trap[57] = 0.5*Dt/p[94] + 1
    sp_jac_trap[58] = 5.0e-7*Dt + 1
    sp_jac_trap[59] = 0.5*Dt*p[95]
    sp_jac_trap[60] = -0.5*Dt/p[97]
    sp_jac_trap[61] = 0.5*Dt/p[97] + 1
    sp_jac_trap[62] = 0.5*Dt/p[99] + 1
    sp_jac_trap[63] = -0.5*Dt/p[99]
    sp_jac_trap[64] = 0.5*Dt*p[113] + 1
    sp_jac_trap[65] = -0.5*Dt*p[103]
    sp_jac_trap[66] = 0.5*Dt*p[103]
    sp_jac_trap[68] = 0.25*p[111]*Dt/p[104] + 1
    sp_jac_trap[73] = -0.25*Dt/p[104]
    sp_jac_trap[74] = -0.25*p[111]*Dt/p[104]
    sp_jac_trap[75] = 0.5*Dt/p[105] + 1
    sp_jac_trap[76] = -0.5*Dt*(p[109] - p[107])/p[105]
    sp_jac_trap[77] = -0.5*Dt/p[105]
    sp_jac_trap[78] = 0.5*Dt/p[106] + 1
    sp_jac_trap[79] = -0.5*Dt*(-p[110] + p[108])/p[106]
    sp_jac_trap[80] = 0.5*Dt/p[117] + 1
    sp_jac_trap[81] = -0.5*Dt/p[117]
    sp_jac_trap[82] = -0.5*Dt*(p[115]*p[120] - 1)
    sp_jac_trap[83] = 0.5*Dt*p[116]*p[120] + 1
    sp_jac_trap[84] = -0.5*Dt*p[120]
    sp_jac_trap[85] = -0.5*Dt*(-p[115]*p[120] + 1)
    sp_jac_trap[86] = 0.5*Dt/p[122] + 1
    sp_jac_trap[87] = -0.5*Dt/p[122]
    sp_jac_trap[88] = -0.5*Dt/p[124]
    sp_jac_trap[89] = 0.5*Dt/p[124] + 1
    sp_jac_trap[90] = 5.0e-7*Dt + 1
    sp_jac_trap[91] = 0.5*Dt*p[125]
    sp_jac_trap[92] = -0.5*Dt/p[127]
    sp_jac_trap[93] = 0.5*Dt/p[127] + 1
    sp_jac_trap[94] = 0.5*Dt/p[129] + 1
    sp_jac_trap[95] = -0.5*Dt/p[129]
    sp_jac_trap[96] = 0.5*Dt*p[143] + 1
    sp_jac_trap[97] = -0.5*Dt*p[133]
    sp_jac_trap[98] = 0.5*Dt*p[133]
    sp_jac_trap[100] = 0.25*p[141]*Dt/p[134] + 1
    sp_jac_trap[105] = -0.25*Dt/p[134]
    sp_jac_trap[106] = -0.25*p[141]*Dt/p[134]
    sp_jac_trap[107] = 0.5*Dt/p[135] + 1
    sp_jac_trap[108] = -0.5*Dt*(p[139] - p[137])/p[135]
    sp_jac_trap[109] = -0.5*Dt/p[135]
    sp_jac_trap[110] = 0.5*Dt/p[136] + 1
    sp_jac_trap[111] = -0.5*Dt*(-p[140] + p[138])/p[136]
    sp_jac_trap[112] = 0.5*Dt/p[147] + 1
    sp_jac_trap[113] = -0.5*Dt/p[147]
    sp_jac_trap[114] = -0.5*Dt*(p[145]*p[150] - 1)
    sp_jac_trap[115] = 0.5*Dt*p[146]*p[150] + 1
    sp_jac_trap[116] = -0.5*Dt*p[150]
    sp_jac_trap[117] = -0.5*Dt*(-p[145]*p[150] + 1)
    sp_jac_trap[118] = 0.5*Dt/p[152] + 1
    sp_jac_trap[119] = -0.5*Dt/p[152]
    sp_jac_trap[120] = -0.5*Dt/p[154]
    sp_jac_trap[121] = 0.5*Dt/p[154] + 1
    sp_jac_trap[122] = 5.0e-7*Dt + 1
    sp_jac_trap[123] = 0.5*Dt*p[155]
    sp_jac_trap[124] = -0.5*Dt/p[157]
    sp_jac_trap[125] = 0.5*Dt/p[157] + 1
    sp_jac_trap[126] = 0.5*Dt/p[159] + 1
    sp_jac_trap[127] = -0.5*Dt/p[159]
    sp_jac_trap[129] = 0.5*Dt
    sp_jac_trap[134] = -p[42]/p[0]
    sp_jac_trap[139] = -p[42]/p[0]
    sp_jac_trap[144] = -p[72]/p[0]
    sp_jac_trap[149] = -p[72]/p[0]
    sp_jac_trap[154] = -p[102]/p[0]
    sp_jac_trap[159] = -p[102]/p[0]
    sp_jac_trap[164] = -p[132]/p[0]
    sp_jac_trap[169] = -p[132]/p[0]
    sp_jac_trap[266] = p[49]
    sp_jac_trap[267] = p[52]
    sp_jac_trap[272] = p[52]
    sp_jac_trap[273] = -p[50]
    sp_jac_trap[290] = -1/p[61]
    sp_jac_trap[293] = p[54]
    sp_jac_trap[294] = p[63]/p[64]
    sp_jac_trap[295] = -p[63]/p[64] + 1
    sp_jac_trap[307] = p[79]
    sp_jac_trap[308] = p[82]
    sp_jac_trap[313] = p[82]
    sp_jac_trap[314] = -p[80]
    sp_jac_trap[331] = -1/p[91]
    sp_jac_trap[334] = p[84]
    sp_jac_trap[335] = p[93]/p[94]
    sp_jac_trap[336] = -p[93]/p[94] + 1
    sp_jac_trap[348] = p[109]
    sp_jac_trap[349] = p[112]
    sp_jac_trap[354] = p[112]
    sp_jac_trap[355] = -p[110]
    sp_jac_trap[372] = -1/p[121]
    sp_jac_trap[375] = p[114]
    sp_jac_trap[376] = p[123]/p[124]
    sp_jac_trap[377] = -p[123]/p[124] + 1
    sp_jac_trap[389] = p[139]
    sp_jac_trap[390] = p[142]
    sp_jac_trap[395] = p[142]
    sp_jac_trap[396] = -p[140]
    sp_jac_trap[413] = -1/p[151]
    sp_jac_trap[416] = p[144]
    sp_jac_trap[417] = p[153]/p[154]
    sp_jac_trap[418] = -p[153]/p[154] + 1
    sp_jac_trap[426] = p[44]*p[42]/(p[44]*p[42] + p[74]*p[72] + p[104]*p[102] + p[134]*p[132])
    sp_jac_trap[427] = p[74]*p[72]/(p[44]*p[42] + p[74]*p[72] + p[104]*p[102] + p[134]*p[132])
    sp_jac_trap[428] = p[104]*p[102]/(p[44]*p[42] + p[74]*p[72] + p[104]*p[102] + p[134]*p[132])
    sp_jac_trap[429] = p[134]*p[132]/(p[44]*p[42] + p[74]*p[72] + p[104]*p[102] + p[134]*p[132])
    sp_jac_trap[431] = p[163]
    sp_jac_trap[432] = -p[162]

def sp_jac_trap_eval_num(sp_jac_trap,x,y,u,p,Dt,xyup = 1):

    sp_jac_trap[128] = 1
    sp_jac_trap[263] = -1
    sp_jac_trap[269] = -1
    sp_jac_trap[279] = -1
    sp_jac_trap[285] = -1
    sp_jac_trap[288] = -1
    sp_jac_trap[291] = 1
    sp_jac_trap[292] = -1
    sp_jac_trap[296] = -1
    sp_jac_trap[297] = 1
    sp_jac_trap[298] = -1
    sp_jac_trap[299] = -1
    sp_jac_trap[302] = -1
    sp_jac_trap[304] = -1
    sp_jac_trap[310] = -1
    sp_jac_trap[320] = -1
    sp_jac_trap[326] = -1
    sp_jac_trap[329] = -1
    sp_jac_trap[332] = 1
    sp_jac_trap[333] = -1
    sp_jac_trap[337] = -1
    sp_jac_trap[338] = 1
    sp_jac_trap[339] = -1
    sp_jac_trap[340] = -1
    sp_jac_trap[343] = -1
    sp_jac_trap[345] = -1
    sp_jac_trap[351] = -1
    sp_jac_trap[361] = -1
    sp_jac_trap[367] = -1
    sp_jac_trap[370] = -1
    sp_jac_trap[373] = 1
    sp_jac_trap[374] = -1
    sp_jac_trap[378] = -1
    sp_jac_trap[379] = 1
    sp_jac_trap[380] = -1
    sp_jac_trap[381] = -1
    sp_jac_trap[384] = -1
    sp_jac_trap[386] = -1
    sp_jac_trap[392] = -1
    sp_jac_trap[402] = -1
    sp_jac_trap[408] = -1
    sp_jac_trap[411] = -1
    sp_jac_trap[414] = 1
    sp_jac_trap[415] = -1
    sp_jac_trap[419] = -1
    sp_jac_trap[420] = 1
    sp_jac_trap[421] = -1
    sp_jac_trap[422] = -1
    sp_jac_trap[425] = -1
    sp_jac_trap[430] = -1
    sp_jac_trap[433] = -1

def sp_jac_trap_vectors():

    sp_jac_trap_ia = [0, 1, 103, 0, 1, 45, 46, 67, 68, 73, 103, 2, 67, 71, 3, 68, 4, 45, 4, 5, 71, 75, 6, 72, 6, 7, 8, 69, 1, 9, 10, 74, 11, 12, 103, 11, 12, 47, 48, 76, 77, 82, 103, 13, 76, 80, 14, 77, 15, 47, 15, 16, 80, 84, 17, 81, 17, 18, 19, 78, 12, 20, 21, 83, 22, 23, 103, 22, 23, 49, 50, 85, 86, 91, 103, 24, 85, 89, 25, 86, 26, 49, 26, 27, 89, 93, 28, 90, 28, 29, 30, 87, 23, 31, 32, 92, 33, 34, 103, 33, 34, 51, 52, 94, 95, 100, 103, 35, 94, 98, 36, 95, 37, 51, 37, 38, 98, 102, 39, 99, 39, 40, 41, 96, 34, 42, 43, 101, 44, 103, 45, 46, 53, 54, 69, 45, 46, 53, 54, 70, 47, 48, 55, 56, 78, 47, 48, 55, 56, 79, 49, 50, 65, 66, 87, 49, 50, 65, 66, 88, 51, 52, 63, 64, 96, 51, 52, 63, 64, 97, 45, 46, 53, 54, 55, 56, 45, 46, 53, 54, 55, 56, 47, 48, 53, 54, 55, 56, 57, 58, 47, 48, 53, 54, 55, 56, 57, 58, 55, 56, 57, 58, 59, 60, 55, 56, 57, 58, 59, 60, 57, 58, 59, 60, 61, 62, 57, 58, 59, 60, 61, 62, 59, 60, 61, 62, 63, 64, 59, 60, 61, 62, 63, 64, 51, 52, 61, 62, 63, 64, 65, 66, 51, 52, 61, 62, 63, 64, 65, 66, 49, 50, 63, 64, 65, 66, 49, 50, 63, 64, 65, 66, 0, 2, 45, 46, 67, 68, 0, 3, 45, 46, 67, 68, 0, 45, 46, 67, 68, 69, 0, 45, 46, 67, 68, 70, 4, 5, 71, 75, 1, 8, 72, 104, 6, 7, 73, 1, 9, 74, 10, 74, 75, 11, 13, 47, 48, 76, 77, 11, 14, 47, 48, 76, 77, 11, 47, 48, 76, 77, 78, 11, 47, 48, 76, 77, 79, 15, 16, 80, 84, 12, 19, 81, 104, 17, 18, 82, 12, 20, 83, 21, 83, 84, 22, 24, 49, 50, 85, 86, 22, 25, 49, 50, 85, 86, 22, 49, 50, 85, 86, 87, 22, 49, 50, 85, 86, 88, 26, 27, 89, 93, 23, 30, 90, 104, 28, 29, 91, 23, 31, 92, 32, 92, 93, 33, 35, 51, 52, 94, 95, 33, 36, 51, 52, 94, 95, 33, 51, 52, 94, 95, 96, 33, 51, 52, 94, 95, 97, 37, 38, 98, 102, 34, 41, 99, 104, 39, 40, 100, 34, 42, 101, 43, 101, 102, 1, 12, 23, 34, 103, 44, 103, 104]
    sp_jac_trap_ja = [0, 3, 11, 14, 16, 18, 22, 24, 26, 28, 30, 32, 35, 43, 46, 48, 50, 54, 56, 58, 60, 62, 64, 67, 75, 78, 80, 82, 86, 88, 90, 92, 94, 96, 99, 107, 110, 112, 114, 118, 120, 122, 124, 126, 128, 130, 135, 140, 145, 150, 155, 160, 165, 170, 176, 182, 190, 198, 204, 210, 216, 222, 228, 234, 242, 250, 256, 262, 268, 274, 280, 286, 290, 294, 297, 300, 303, 309, 315, 321, 327, 331, 335, 338, 341, 344, 350, 356, 362, 368, 372, 376, 379, 382, 385, 391, 397, 403, 409, 413, 417, 420, 423, 426, 431, 434]
    sp_jac_trap_nia = 105
    sp_jac_trap_nja = 105
    return sp_jac_trap_ia, sp_jac_trap_ja, sp_jac_trap_nia, sp_jac_trap_nja 



def nonzeros():
    Fx_ini_rows = [0, 0, 1, 1, 2, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 12, 13, 14, 15, 16, 16, 17, 18, 18, 19, 20, 20, 21, 22, 22, 23, 23, 24, 25, 26, 27, 27, 28, 29, 29, 30, 31, 31, 32, 33, 33, 34, 34, 35, 36, 37, 38, 38, 39, 40, 40, 41, 42, 42, 43]

    Fx_ini_cols = [0, 1, 0, 1, 2, 3, 4, 4, 5, 6, 6, 7, 8, 1, 9, 10, 11, 12, 11, 12, 13, 14, 15, 15, 16, 17, 17, 18, 19, 12, 20, 21, 22, 23, 22, 23, 24, 25, 26, 26, 27, 28, 28, 29, 30, 23, 31, 32, 33, 34, 33, 34, 35, 36, 37, 37, 38, 39, 39, 40, 41, 34, 42, 43]

    Fy_ini_rows = [0, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4, 5, 5, 6, 8, 10, 11, 12, 12, 12, 12, 12, 12, 13, 13, 14, 15, 16, 16, 17, 19, 21, 22, 23, 23, 23, 23, 23, 23, 24, 24, 25, 26, 27, 27, 28, 30, 32, 33, 34, 34, 34, 34, 34, 34, 35, 35, 36, 37, 38, 38, 39, 41, 43, 44]

    Fy_ini_cols = [58, 0, 1, 22, 23, 28, 58, 22, 26, 23, 0, 26, 30, 27, 24, 29, 58, 2, 3, 31, 32, 37, 58, 31, 35, 32, 2, 35, 39, 36, 33, 38, 58, 4, 5, 40, 41, 46, 58, 40, 44, 41, 4, 44, 48, 45, 42, 47, 58, 6, 7, 49, 50, 55, 58, 49, 53, 50, 6, 53, 57, 54, 51, 56, 58]

    Gx_ini_rows = [22, 22, 23, 23, 24, 25, 26, 26, 27, 27, 28, 28, 29, 29, 30, 31, 31, 32, 32, 33, 34, 35, 35, 36, 36, 37, 37, 38, 38, 39, 40, 40, 41, 41, 42, 43, 44, 44, 45, 45, 46, 46, 47, 47, 48, 49, 49, 50, 50, 51, 52, 53, 53, 54, 54, 55, 55, 56, 56, 57, 58, 58, 58, 58, 59]

    Gx_ini_cols = [0, 2, 0, 3, 0, 0, 4, 5, 1, 8, 6, 7, 1, 9, 10, 11, 13, 11, 14, 11, 11, 15, 16, 12, 19, 17, 18, 12, 20, 21, 22, 24, 22, 25, 22, 22, 26, 27, 23, 30, 28, 29, 23, 31, 32, 33, 35, 33, 36, 33, 33, 37, 38, 34, 41, 39, 40, 34, 42, 43, 1, 12, 23, 34, 44]

    Gy_ini_rows = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 26, 26, 27, 27, 28, 29, 30, 30, 31, 31, 31, 31, 32, 32, 32, 32, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 35, 35, 36, 36, 37, 38, 39, 39, 40, 40, 40, 40, 41, 41, 41, 41, 42, 42, 42, 42, 42, 43, 43, 43, 43, 43, 44, 44, 45, 45, 46, 47, 48, 48, 49, 49, 49, 49, 50, 50, 50, 50, 51, 51, 51, 51, 51, 52, 52, 52, 52, 52, 53, 53, 54, 54, 55, 56, 57, 57, 58, 59, 59]

    Gy_ini_cols = [0, 1, 8, 9, 24, 0, 1, 8, 9, 25, 2, 3, 10, 11, 33, 2, 3, 10, 11, 34, 4, 5, 20, 21, 42, 4, 5, 20, 21, 43, 6, 7, 18, 19, 51, 6, 7, 18, 19, 52, 0, 1, 8, 9, 10, 11, 0, 1, 8, 9, 10, 11, 2, 3, 8, 9, 10, 11, 12, 13, 2, 3, 8, 9, 10, 11, 12, 13, 10, 11, 12, 13, 14, 15, 10, 11, 12, 13, 14, 15, 12, 13, 14, 15, 16, 17, 12, 13, 14, 15, 16, 17, 14, 15, 16, 17, 18, 19, 14, 15, 16, 17, 18, 19, 6, 7, 16, 17, 18, 19, 20, 21, 6, 7, 16, 17, 18, 19, 20, 21, 4, 5, 18, 19, 20, 21, 4, 5, 18, 19, 20, 21, 0, 1, 22, 23, 0, 1, 22, 23, 0, 1, 22, 23, 24, 0, 1, 22, 23, 25, 26, 30, 27, 59, 28, 29, 29, 30, 2, 3, 31, 32, 2, 3, 31, 32, 2, 3, 31, 32, 33, 2, 3, 31, 32, 34, 35, 39, 36, 59, 37, 38, 38, 39, 4, 5, 40, 41, 4, 5, 40, 41, 4, 5, 40, 41, 42, 4, 5, 40, 41, 43, 44, 48, 45, 59, 46, 47, 47, 48, 6, 7, 49, 50, 6, 7, 49, 50, 6, 7, 49, 50, 51, 6, 7, 49, 50, 52, 53, 57, 54, 59, 55, 56, 56, 57, 58, 58, 59]

    return Fx_ini_rows,Fx_ini_cols,Fy_ini_rows,Fy_ini_cols,Gx_ini_rows,Gx_ini_cols,Gy_ini_rows,Gy_ini_cols