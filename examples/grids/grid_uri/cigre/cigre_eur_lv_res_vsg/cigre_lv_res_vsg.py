import numpy as np
import numba
import scipy.optimize as sopt
import json

sin = np.sin
cos = np.cos
atan2 = np.arctan2
sqrt = np.sqrt 
sign = np.sign 
exp = np.exp


class cigre_lv_res_vsg_class: 

    def __init__(self): 

        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 25
        self.N_y = 140 
        self.N_z = 4 
        self.N_store = 10000 
        self.params_list = ['R_R00R01', 'L_R00R01', 'C_R00R01', 'R_R02R01', 'L_R02R01', 'C_R02R01', 'R_R02R03', 'L_R02R03', 'C_R02R03', 'R_R03R04', 'L_R03R04', 'C_R03R04', 'R_R04R05', 'L_R04R05', 'C_R04R05', 'R_R04R12', 'L_R04R12', 'C_R04R12', 'R_R05R06', 'L_R05R06', 'C_R05R06', 'R_R06R07', 'L_R06R07', 'C_R06R07', 'R_R07R08', 'L_R07R08', 'C_R07R08', 'R_R08R09', 'L_R08R09', 'C_R08R09', 'R_R09R10', 'L_R09R10', 'C_R09R10', 'R_R09R17', 'L_R09R17', 'C_R09R17', 'R_R11R03', 'L_R11R03', 'C_R11R03', 'R_R12R13', 'L_R12R13', 'C_R12R13', 'R_R13R14', 'L_R13R14', 'C_R13R14', 'R_R14R15', 'L_R14R15', 'C_R14R15', 'R_R16R06', 'L_R16R06', 'C_R16R06', 'R_R18R10', 'L_R18R10', 'C_R18R10', 'i_R00_D', 'i_R00_Q', 'i_R02_D', 'i_R02_Q', 'i_R03_D', 'i_R03_Q', 'i_R04_D', 'i_R04_Q', 'i_R05_D', 'i_R05_Q', 'i_R06_D', 'i_R06_Q', 'i_R07_D', 'i_R07_Q', 'i_R08_D', 'i_R08_Q', 'i_R09_D', 'i_R09_Q', 'i_R12_D', 'i_R12_Q', 'i_R13_D', 'i_R13_Q', 'omega', 'L_t_G10', 'R_t_G10', 'C_m_G10', 'L_s_G10', 'R_s_G10', 'omega_G10', 'G_d_G10', 'K_p_G10', 'T_p_G10', 'K_q_G10', 'T_q_G10', 'R_v_G10', 'X_v_G10', 'S_b_kVA_G10', 'U_b_G10', 'K_phi_G10', 'H_G10', 'D_G10', 'T_vpoi_G10', 'K_vpoi_G10', 'T_f_G10', 'K_f_G10', 'L_t_G14', 'R_t_G14', 'C_m_G14', 'L_s_G14', 'R_s_G14', 'omega_G14', 'G_d_G14', 'K_p_G14', 'T_p_G14', 'K_q_G14', 'T_q_G14', 'R_v_G14', 'X_v_G14', 'S_b_kVA_G14', 'U_b_G14', 'K_phi_G14', 'H_G14', 'D_G14', 'T_vpoi_G14', 'K_vpoi_G14', 'T_f_G14', 'K_f_G14', 'K_f_sec'] 
        self.params_values_list  = [0.0032, 4.074366543152521e-05, 0.0, 0.0056721, 9.038568373131828e-06, 0.0, 0.0056721, 9.038568373131828e-06, 0.0, 0.0056721, 9.038568373131828e-06, 0.0, 0.0056721, 9.038568373131828e-06, 0.0, 0.0287735, 9.496457144407212e-06, 0.0, 0.0056721, 9.038568373131828e-06, 0.0, 0.0056721, 9.038568373131828e-06, 0.0, 0.0056721, 9.038568373131828e-06, 0.0, 0.0056721, 9.038568373131828e-06, 0.0, 0.0056721, 9.038568373131828e-06, 0.0, 0.024663, 8.139820409491894e-06, 0.0, 0.024663, 8.139820409491894e-06, 0.0, 0.0287735, 9.496457144407212e-06, 0.0, 0.0287735, 9.496457144407212e-06, 0.0, 0.0287735, 9.496457144407212e-06, 0.0, 0.024663, 8.139820409491894e-06, 0.0, 0.024663, 8.139820409491894e-06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 314.1592653589793, 0.00125, 0.039269908169872414, 4e-06, 0.00125, 0.039269908169872414, 314.1592653589793, 0.01, 0.01, 0.1, 0.1, 0.1, 0.01, 0.05, 200, 400.0, 0.0, 5.0, 5.0, 0.1, 10, 0.1, 20.0, 0.00125, 0.039269908169872414, 4e-06, 0.00125, 0.039269908169872414, 314.1592653589793, 0.01, 0.01, 0.1, 0.1, 0.1, 0.01, 0.05, 200, 400.0, 0.0, 5.0, 5.0, 0.1, 10, 0.1, 20.0, 0.001] 
        self.inputs_ini_list = ['T_i_R01', 'I_max_R01', 'p_R01_ref', 'q_R01_ref', 'T_i_R11', 'I_max_R11', 'p_R11_ref', 'q_R11_ref', 'T_i_R15', 'I_max_R15', 'p_R15_ref', 'q_R15_ref', 'T_i_R16', 'I_max_R16', 'p_R16_ref', 'q_R16_ref', 'T_i_R17', 'I_max_R17', 'p_R17_ref', 'q_R17_ref', 'T_i_R18', 'I_max_R18', 'p_R18_ref', 'q_R18_ref', 'v_dc_G10', 'p_m_ref_G10', 'q_s_ref_G10', 'v_s_ref_G10', 'omega_ref_G10', 'p_r_G10', 'q_r_G10', 'v_dc_G14', 'p_m_ref_G14', 'q_s_ref_G14', 'v_s_ref_G14', 'omega_ref_G14', 'p_r_G14', 'q_r_G14'] 
        self.inputs_ini_values_list  = [0.01, 434.78, 190000.0, 62449.979983984005, 0.01, 32.6, 14250.0, 4683.7484987988, 0.01, 120, 49400.0, 16236.99479583584, 0.01, 120, 52250.0, 17173.7444955956, 0.01, 120, 33250.0, 10928.7464971972, 0.01, 120, 44650.0, 14675.74529623624, 750, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 750, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0] 
        self.inputs_run_list = ['T_i_R01', 'I_max_R01', 'p_R01_ref', 'q_R01_ref', 'T_i_R11', 'I_max_R11', 'p_R11_ref', 'q_R11_ref', 'T_i_R15', 'I_max_R15', 'p_R15_ref', 'q_R15_ref', 'T_i_R16', 'I_max_R16', 'p_R16_ref', 'q_R16_ref', 'T_i_R17', 'I_max_R17', 'p_R17_ref', 'q_R17_ref', 'T_i_R18', 'I_max_R18', 'p_R18_ref', 'q_R18_ref', 'v_dc_G10', 'p_m_ref_G10', 'q_s_ref_G10', 'v_s_ref_G10', 'omega_ref_G10', 'p_r_G10', 'q_r_G10', 'v_dc_G14', 'p_m_ref_G14', 'q_s_ref_G14', 'v_s_ref_G14', 'omega_ref_G14', 'p_r_G14', 'q_r_G14'] 
        self.inputs_run_values_list = [0.01, 434.78, 190000.0, 62449.979983984005, 0.01, 32.6, 14250.0, 4683.7484987988, 0.01, 120, 49400.0, 16236.99479583584, 0.01, 120, 52250.0, 17173.7444955956, 0.01, 120, 33250.0, 10928.7464971972, 0.01, 120, 44650.0, 14675.74529623624, 750, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 750, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0] 
        self.outputs_list = ['i_sD_G10', 'i_sQ_G10', 'i_sD_G14', 'i_sQ_G14'] 
        self.x_list = ['i_R01_D', 'i_R01_Q', 'i_R11_D', 'i_R11_Q', 'i_R15_D', 'i_R15_Q', 'i_R16_D', 'i_R16_Q', 'i_R17_D', 'i_R17_Q', 'i_R18_D', 'i_R18_Q', 'phi_G10', 'omega_v_G10', 'xi_q_G10', 'omega_rads_G10', 'omega_v_filt_G10', 'v_s_filt_G10', 'phi_G14', 'omega_v_G14', 'xi_q_G14', 'omega_rads_G14', 'omega_v_filt_G14', 'v_s_filt_G14', 'xi_f_sec'] 
        self.y_run_list = ['i_l_R00R01_D', 'i_l_R00R01_Q', 'i_l_R02R01_D', 'i_l_R02R01_Q', 'i_l_R02R03_D', 'i_l_R02R03_Q', 'i_l_R03R04_D', 'i_l_R03R04_Q', 'i_l_R04R05_D', 'i_l_R04R05_Q', 'i_l_R04R12_D', 'i_l_R04R12_Q', 'i_l_R05R06_D', 'i_l_R05R06_Q', 'i_l_R06R07_D', 'i_l_R06R07_Q', 'i_l_R07R08_D', 'i_l_R07R08_Q', 'i_l_R08R09_D', 'i_l_R08R09_Q', 'i_l_R09R10_D', 'i_l_R09R10_Q', 'i_l_R09R17_D', 'i_l_R09R17_Q', 'i_l_R11R03_D', 'i_l_R11R03_Q', 'i_l_R12R13_D', 'i_l_R12R13_Q', 'i_l_R13R14_D', 'i_l_R13R14_Q', 'i_l_R14R15_D', 'i_l_R14R15_Q', 'i_l_R16R06_D', 'i_l_R16R06_Q', 'i_l_R18R10_D', 'i_l_R18R10_Q', 'v_R00_D', 'v_R00_Q', 'v_R01_D', 'v_R01_Q', 'v_R02_D', 'v_R02_Q', 'v_R03_D', 'v_R03_Q', 'v_R04_D', 'v_R04_Q', 'v_R05_D', 'v_R05_Q', 'v_R06_D', 'v_R06_Q', 'v_R07_D', 'v_R07_Q', 'v_R08_D', 'v_R08_Q', 'v_R09_D', 'v_R09_Q', 'v_R10_D', 'v_R10_Q', 'v_R11_D', 'v_R11_Q', 'v_R12_D', 'v_R12_Q', 'v_R13_D', 'v_R13_Q', 'v_R14_D', 'v_R14_Q', 'v_R15_D', 'v_R15_Q', 'v_R16_D', 'v_R16_Q', 'v_R17_D', 'v_R17_Q', 'v_R18_D', 'v_R18_Q', 'i_R01_d_ref', 'i_R01_q_ref', 'i_R11_d_ref', 'i_R11_q_ref', 'i_R15_d_ref', 'i_R15_q_ref', 'i_R16_d_ref', 'i_R16_q_ref', 'i_R17_d_ref', 'i_R17_q_ref', 'i_R18_d_ref', 'i_R18_q_ref', 'i_tD_G10', 'i_tQ_G10', 'v_mD_G10', 'v_mQ_G10', 'i_sD_G10', 'i_sQ_G10', 'i_R10_D', 'i_R10_Q', 'v_sD_G10', 'v_sQ_G10', 'eta_d_G10', 'eta_q_G10', 'eta_D_G10', 'eta_Q_G10', 'v_md_G10', 'v_mq_G10', 'v_sd_G10', 'v_sq_G10', 'i_td_G10', 'i_tq_G10', 'i_sd_G10', 'i_sq_G10', 'DV_sat_G10', 'p_s_pu_G10', 'q_s_pu_G10', 'p_m_ref_G10', 'q_s_ref_G10', 'i_tD_G14', 'i_tQ_G14', 'v_mD_G14', 'v_mQ_G14', 'i_sD_G14', 'i_sQ_G14', 'i_R14_D', 'i_R14_Q', 'v_sD_G14', 'v_sQ_G14', 'eta_d_G14', 'eta_q_G14', 'eta_D_G14', 'eta_Q_G14', 'v_md_G14', 'v_mq_G14', 'v_sd_G14', 'v_sq_G14', 'i_td_G14', 'i_tq_G14', 'i_sd_G14', 'i_sq_G14', 'DV_sat_G14', 'p_s_pu_G14', 'q_s_pu_G14', 'p_m_ref_G14', 'q_s_ref_G14'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['i_l_R00R01_D', 'i_l_R00R01_Q', 'i_l_R02R01_D', 'i_l_R02R01_Q', 'i_l_R02R03_D', 'i_l_R02R03_Q', 'i_l_R03R04_D', 'i_l_R03R04_Q', 'i_l_R04R05_D', 'i_l_R04R05_Q', 'i_l_R04R12_D', 'i_l_R04R12_Q', 'i_l_R05R06_D', 'i_l_R05R06_Q', 'i_l_R06R07_D', 'i_l_R06R07_Q', 'i_l_R07R08_D', 'i_l_R07R08_Q', 'i_l_R08R09_D', 'i_l_R08R09_Q', 'i_l_R09R10_D', 'i_l_R09R10_Q', 'i_l_R09R17_D', 'i_l_R09R17_Q', 'i_l_R11R03_D', 'i_l_R11R03_Q', 'i_l_R12R13_D', 'i_l_R12R13_Q', 'i_l_R13R14_D', 'i_l_R13R14_Q', 'i_l_R14R15_D', 'i_l_R14R15_Q', 'i_l_R16R06_D', 'i_l_R16R06_Q', 'i_l_R18R10_D', 'i_l_R18R10_Q', 'v_R00_D', 'v_R00_Q', 'v_R01_D', 'v_R01_Q', 'v_R02_D', 'v_R02_Q', 'v_R03_D', 'v_R03_Q', 'v_R04_D', 'v_R04_Q', 'v_R05_D', 'v_R05_Q', 'v_R06_D', 'v_R06_Q', 'v_R07_D', 'v_R07_Q', 'v_R08_D', 'v_R08_Q', 'v_R09_D', 'v_R09_Q', 'v_R10_D', 'v_R10_Q', 'v_R11_D', 'v_R11_Q', 'v_R12_D', 'v_R12_Q', 'v_R13_D', 'v_R13_Q', 'v_R14_D', 'v_R14_Q', 'v_R15_D', 'v_R15_Q', 'v_R16_D', 'v_R16_Q', 'v_R17_D', 'v_R17_Q', 'v_R18_D', 'v_R18_Q', 'i_R01_d_ref', 'i_R01_q_ref', 'i_R11_d_ref', 'i_R11_q_ref', 'i_R15_d_ref', 'i_R15_q_ref', 'i_R16_d_ref', 'i_R16_q_ref', 'i_R17_d_ref', 'i_R17_q_ref', 'i_R18_d_ref', 'i_R18_q_ref', 'i_tD_G10', 'i_tQ_G10', 'v_mD_G10', 'v_mQ_G10', 'i_sD_G10', 'i_sQ_G10', 'i_R10_D', 'i_R10_Q', 'v_sD_G10', 'v_sQ_G10', 'eta_d_G10', 'eta_q_G10', 'eta_D_G10', 'eta_Q_G10', 'v_md_G10', 'v_mq_G10', 'v_sd_G10', 'v_sq_G10', 'i_td_G10', 'i_tq_G10', 'i_sd_G10', 'i_sq_G10', 'DV_sat_G10', 'p_s_pu_G10', 'q_s_pu_G10', 'p_m_ref_G10', 'q_s_ref_G10', 'i_tD_G14', 'i_tQ_G14', 'v_mD_G14', 'v_mQ_G14', 'i_sD_G14', 'i_sQ_G14', 'i_R14_D', 'i_R14_Q', 'v_sD_G14', 'v_sQ_G14', 'eta_d_G14', 'eta_q_G14', 'eta_D_G14', 'eta_Q_G14', 'v_md_G14', 'v_mq_G14', 'v_sd_G14', 'v_sq_G14', 'i_td_G14', 'i_tq_G14', 'i_sd_G14', 'i_sq_G14', 'DV_sat_G14', 'p_s_pu_G14', 'q_s_pu_G14', 'p_m_ref_G14', 'q_s_ref_G14'] 
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
        Fx_ini_rows,Fx_ini_cols,Fy_ini_rows,Fy_ini_cols,Gx_ini_rows,Gx_ini_cols,Gy_ini_rows,Gy_ini_cols = nonzeros()

        self.Fx_ini_rows = np.array(Fx_ini_rows) 
        if len(Fx_ini_rows) == 1: 
            self.Fx_ini_rows = np.array([[Fx_ini_rows]]).reshape(1,) 
            self.Fx_ini_cols = np.array([[Fx_ini_cols]]).reshape(1,)  
            
        self.Fx_ini_cols = np.array(Fx_ini_cols)
        self.Fy_ini_rows = np.array(Fy_ini_rows)        
        self.Fy_ini_cols = np.array(Fy_ini_cols)
        self.Gx_ini_rows = np.array(Gx_ini_rows)        
        self.Gx_ini_cols = np.array(Gx_ini_cols)
        self.Gy_ini_rows = np.array(Gy_ini_rows)        
        self.Gy_ini_cols = np.array(Gy_ini_cols)
        
        
        self.yini2urun = list(set(self.inputs_run_list).intersection(set(self.y_ini_list)))
        self.uini2yrun = list(set(self.y_run_list).intersection(set(self.inputs_ini_list)))

        self.update() 
                
    def update(self): 

        self.N_steps = int(np.ceil(self.t_end/self.Dt)) 
        dt = [  
              ('t_end', np.float64),
              ('Dt', np.float64),
              ('decimation', np.float64),
              ('itol', np.float64),
              ('Dt_max', np.float64),
              ('Dt_min', np.float64),
              ('solvern', np.int64),
              ('imax', np.int64),
              ('N_steps', np.int64),
              ('N_store', np.int64),
              ('N_x', np.int64),
              ('N_y', np.int64),
              ('N_z', np.int64),
              ('t', np.float64),
              ('it', np.int64),
              ('it_store', np.int64),
              ('idx', np.int64),
              ('idy', np.int64),
              ('f', np.float64, (self.N_x,1)),
              ('x', np.float64, (self.N_x,1)),
              ('x_0', np.float64, (self.N_x,1)),
              ('g', np.float64, (self.N_y,1)),
              ('y_run', np.float64, (self.N_y,1)),
              ('y_ini', np.float64, (self.N_y,1)),
              ('u_run', np.float64, (self.N_u,1)),
              ('y_0', np.float64, (self.N_y,1)),
              ('h', np.float64, (self.N_z,1)),
              ('Fx', np.float64, (self.N_x,self.N_x)),
              ('Fy', np.float64, (self.N_x,self.N_y)),
              ('Gx', np.float64, (self.N_y,self.N_x)),
              ('Gy', np.float64, (self.N_y,self.N_y)),
              ('Fu', np.float64, (self.N_x,self.N_u)),
              ('Gu', np.float64, (self.N_y,self.N_u)),
              ('Hx', np.float64, (self.N_z,self.N_x)),
              ('Hy', np.float64, (self.N_z,self.N_y)),
              ('Hu', np.float64, (self.N_z,self.N_u)),
              ('Fx_ini', np.float64, (self.N_x,self.N_x)),
              ('Fy_ini', np.float64, (self.N_x,self.N_y)),
              ('Gx_ini', np.float64, (self.N_y,self.N_x)),
              ('Gy_ini', np.float64, (self.N_y,self.N_y)),
              ('T', np.float64, (self.N_store+1,1)),
              ('X', np.float64, (self.N_store+1,self.N_x)),
              ('Y', np.float64, (self.N_store+1,self.N_y)),
              ('Z', np.float64, (self.N_store+1,self.N_z)),
              ('iters', np.float64, (self.N_store+1,1)),
              ('store', np.int64),
              ('Fx_ini_rows', np.int64, self.Fx_ini_rows.shape),
              ('Fx_ini_cols', np.int64, self.Fx_ini_cols.shape),
              ('Fy_ini_rows', np.int64, self.Fy_ini_rows.shape),
              ('Fy_ini_cols', np.int64, self.Fy_ini_cols.shape),
              ('Gx_ini_rows', np.int64, self.Gx_ini_rows.shape),
              ('Gx_ini_cols', np.int64, self.Gx_ini_cols.shape),
              ('Gy_ini_rows', np.int64, self.Gy_ini_rows.shape),
              ('Gy_ini_cols', np.int64, self.Gy_ini_cols.shape),
              ('Ac_ini', np.float64, ((self.N_x+self.N_y,self.N_x+self.N_y))),   
              ('fg', np.float64, ((self.N_x+self.N_y,1))),  
             ]



        
        
        values = [
                self.t_end,                          
                self.Dt,
                self.decimation,
                self.itol,
                self.Dt_max,
                self.Dt_min,
                self.solvern,
                self.imax,
                self.N_steps,
                self.N_store,
                self.N_x,
                self.N_y,
                self.N_z,
                self.t,
                self.it,
                self.it_store,
                0,                                     # idx
                0,                                     # idy
                np.zeros((self.N_x,1)),                # f
                np.zeros((self.N_x,1)),                # x
                np.zeros((self.N_x,1)),                # x_0
                np.zeros((self.N_y,1)),                # g
                np.zeros((self.N_y,1)),                # y_run
                np.zeros((self.N_y,1)),                # y_ini
                np.zeros((self.N_u,1)),                # u_run
                np.zeros((self.N_y,1)),                # y_0
                np.zeros((self.N_z,1)),                # h
                np.zeros((self.N_x,self.N_x)),         # Fx   
                np.zeros((self.N_x,self.N_y)),         # Fy 
                np.zeros((self.N_y,self.N_x)),         # Gx 
                np.zeros((self.N_y,self.N_y)),         # Fy
                np.zeros((self.N_x,self.N_u)),         # Fu 
                np.zeros((self.N_y,self.N_u)),         # Gu 
                np.zeros((self.N_z,self.N_x)),         # Hx 
                np.zeros((self.N_z,self.N_y)),         # Hy 
                np.zeros((self.N_z,self.N_u)),         # Hu 
                np.zeros((self.N_x,self.N_x)),         # Fx_ini  
                np.zeros((self.N_x,self.N_y)),         # Fy_ini 
                np.zeros((self.N_y,self.N_x)),         # Gx_ini 
                np.zeros((self.N_y,self.N_y)),         # Fy_ini 
                np.zeros((self.N_store+1,1)),          # T
                np.zeros((self.N_store+1,self.N_x)),   # X
                np.zeros((self.N_store+1,self.N_y)),   # Y
                np.zeros((self.N_store+1,self.N_z)),   # Z
                np.zeros((self.N_store+1,1)),          # iters
                1,
                self.Fx_ini_rows,       
                self.Fx_ini_cols,
                self.Fy_ini_rows,       
                self.Fy_ini_cols,
                self.Gx_ini_rows,        
                self.Gx_ini_cols,
                self.Gy_ini_rows,       
                self.Gy_ini_cols,
                np.zeros((self.N_x+self.N_y,self.N_x+self.N_y)),  
                np.zeros((self.N_x+self.N_y,1)),
                ]  

        dt += [(item,np.float64) for item in self.params_list]
        values += [item for item in self.params_values_list]

        for item_id,item_val in zip(self.inputs_ini_list,self.inputs_ini_values_list):
            if item_id in self.inputs_run_list: continue
            dt += [(item_id,np.float64)]
            values += [item_val]

        dt += [(item,np.float64) for item in self.inputs_run_list]
        values += [item for item in self.inputs_run_values_list]

        self.struct = np.rec.array([tuple(values)], dtype=np.dtype(dt))
        
        xy0 = np.zeros((self.N_x+self.N_y,))
        self.ini_dae_jacobian_nn(xy0)
        self.run_dae_jacobian_nn(xy0)
        


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

    def ini_problem(self,x):
        self.struct[0].x[:,0] = x[0:self.N_x]
        self.struct[0].y_ini[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        if self.compile:
            ini(self.struct,2)
            ini(self.struct,3)       
        else:
            ini.py_func(self.struct,2)
            ini.py_func(self.struct,3)                   
        fg = np.vstack((self.struct[0].f,self.struct[0].g))[:,0]
        return fg

    def run_problem(self,x):
        t = self.struct[0].t
        self.struct[0].x[:,0] = x[0:self.N_x]
        self.struct[0].y_run[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        
        if self.compile:
            run(t,self.struct,2)
            run(t,self.struct,3)
            run(t,self.struct,10)
            run(t,self.struct,11)
            run(t,self.struct,12)
            run(t,self.struct,13)
        else:
            run.py_func(t,self.struct,2)
            run.py_func(t,self.struct,3)
            run.py_func(t,self.struct,10)
            run.py_func(t,self.struct,11)
            run.py_func(t,self.struct,12)
            run.py_func(t,self.struct,13)            
        
        fg = np.vstack((self.struct[0].f,self.struct[0].g))[:,0]
        return fg
    

    def run_dae_jacobian(self,x):
        self.struct[0].x[:,0] = x[0:self.N_x]
        self.struct[0].y_run[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        run(0.0,self.struct,10)
        run(0.0,self.struct,11)     
        run(0.0,self.struct,12)
        run(0.0,self.struct,13)
        A_c = np.block([[self.struct[0].Fx,self.struct[0].Fy],
                        [self.struct[0].Gx,self.struct[0].Gy]])
        return A_c

    def run_dae_jacobian_nn(self,x):
        self.struct[0].x[:,0] = x[0:self.N_x]
        self.struct[0].y_run[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        run_nn(0.0,self.struct,10)
        run_nn(0.0,self.struct,11)     
        run_nn(0.0,self.struct,12)
        run_nn(0.0,self.struct,13)
 

    
    def eval_jacobians(self):

        run(0.0,self.struct,10)
        run(0.0,self.struct,11)  
        run(0.0,self.struct,12) 

        return 1


    def ini_dae_jacobian(self,x):
        self.struct[0].x[:,0] = x[0:self.N_x]
        self.struct[0].y_ini[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        if self.compile:
            ini(self.struct,10)
            ini(self.struct,11) 
        else:
            ini.py_func(self.struct,10)
            ini.py_func(self.struct,11)             
        A_c = np.block([[self.struct[0].Fx_ini,self.struct[0].Fy_ini],
                        [self.struct[0].Gx_ini,self.struct[0].Gy_ini]])
        return A_c

    def ini_dae_jacobian_nn(self,x):
        self.struct[0].x[:,0] = x[0:self.N_x]
        self.struct[0].y_ini[:,0] = x[self.N_x:(self.N_x+self.N_y)]
        ini_nn(self.struct,10)
        ini_nn(self.struct,11)       
 

    def f_ode(self,x):
        self.struct[0].x[:,0] = x
        run(self.struct,1)
        return self.struct[0].f[:,0]

    def f_odeint(self,x,t):
        self.struct[0].x[:,0] = x
        run(self.struct,1)
        return self.struct[0].f[:,0]

    def f_ivp(self,t,x):
        self.struct[0].x[:,0] = x
        run(self.struct,1)
        return self.struct[0].f[:,0]

    def Fx_ode(self,x):
        self.struct[0].x[:,0] = x
        run(self.struct,10)
        return self.struct[0].Fx

    def eval_A(self):
        
        Fx = self.struct[0].Fx
        Fy = self.struct[0].Fy
        Gx = self.struct[0].Gx
        Gy = self.struct[0].Gy
        
        A = Fx - Fy @ np.linalg.solve(Gy,Gx)
        
        self.A = A
        
        return A

    def eval_A_ini(self):
        
        Fx = self.struct[0].Fx_ini
        Fy = self.struct[0].Fy_ini
        Gx = self.struct[0].Gx_ini
        Gy = self.struct[0].Gy_ini
        
        A = Fx - Fy @ np.linalg.solve(Gy,Gx)
        
        
        return A
    
    def reset(self):
        for param,param_value in zip(self.params_list,self.params_values_list):
            self.struct[0][param] = param_value
        for input_name,input_value in zip(self.inputs_ini_list,self.inputs_ini_values_list):
            self.struct[0][input_name] = input_value   
        for input_name,input_value in zip(self.inputs_run_list,self.inputs_run_values_list):
            self.struct[0][input_name] = input_value  

    def simulate(self,events,xy0=0):
        
        # initialize both the ini and the run system
        self.initialize(events,xy0=xy0)
        
        # simulation run
        for event in events:  
            # make all the desired changes
            self.run([event]) 
            
        # post process
        T,X,Y,Z = self.post()
        
        return T,X,Y,Z
    

    
    def run(self,events):
        

        # simulation run
        for event in events:  
            # make all the desired changes
            for item in event:
                self.struct[0][item] = event[item]
            daesolver(self.struct)    # run until next event
            
        return 1
 
    def rtrun(self,events):
        

        # simulation run
        for event in events:  
            # make all the desired changes
            for item in event:
                self.struct[0][item] = event[item]
            self.struct[0].it_store = self.struct[0].N_store-1
            daesolver(self.struct)    # run until next event
            
            
        return 1
    
    def post(self):
        
        # post process result    
        T = self.struct[0]['T'][:self.struct[0].it_store]
        X = self.struct[0]['X'][:self.struct[0].it_store,:]
        Y = self.struct[0]['Y'][:self.struct[0].it_store,:]
        Z = self.struct[0]['Z'][:self.struct[0].it_store,:]
        iters = self.struct[0]['iters'][:self.struct[0].it_store,:]
    
        self.T = T
        self.X = X
        self.Y = Y
        self.Z = Z
        self.iters = iters
        
        return T,X,Y,Z
        
    def save_0(self,file_name = 'xy_0.json'):
        xy_0_dict = {}
        for item in self.x_list:
            xy_0_dict.update({item:self.get_value(item)})
        for item in self.y_ini_list:
            xy_0_dict.update({item:self.get_value(item)})
    
        xy_0_str = json.dumps(xy_0_dict, indent=4)
        with open(file_name,'w') as fobj:
            fobj.write(xy_0_str)

    def load_0(self,file_name = 'xy_0.json'):
        with open(file_name) as fobj:
            xy_0_str = fobj.read()
        xy_0_dict = json.loads(xy_0_str)
    
        for item in xy_0_dict:
            if item in self.x_list:
                self.xy_prev[self.x_list.index(item)] = xy_0_dict[item]
            if item in self.y_ini_list:
                self.xy_prev[self.y_ini_list.index(item)+self.N_x] = xy_0_dict[item]
                
            
    def initialize(self,events=[{}],xy0=0,compile=True):
        '''
        

        Parameters
        ----------
        events : dictionary 
            Dictionary with at least 't_end' and all inputs and parameters 
            that need to be changed.
        xy0 : float or string, optional
            0 means all states should be zero as initial guess. 
            If not zero all the states initial guess are the given input.
            If 'prev' it uses the last known initialization result as initial guess.

        Returns
        -------
        T : TYPE
            DESCRIPTION.
        X : TYPE
            DESCRIPTION.
        Y : TYPE
            DESCRIPTION.
        Z : TYPE
            DESCRIPTION.

        '''
        
        self.compile = compile
        
        # simulation parameters
        self.struct[0].it = 0       # set time step to zero
        self.struct[0].it_store = 0 # set storage to zero
        self.struct[0].t = 0.0      # set time to zero
                    
        # initialization
        it_event = 0
        event = events[it_event]
        for item in event:
            self.struct[0][item] = event[item]
            
        
        ## compute initial conditions using x and y_ini 
        if type(xy0) == str:
            if xy0 == 'prev':
                xy0 = self.xy_prev
            else:
                self.load_0(xy0)
                xy0 = self.xy_prev
        elif type(xy0) == dict:
            with open('xy_0.json','w') as fobj:
                fobj.write(json.dumps(xy0))
            self.load_0('xy_0.json')
            xy0 = self.xy_prev            
        else:
            if xy0 == 0:
                xy0 = np.zeros(self.N_x+self.N_y)
            elif xy0 == 1:
                xy0 = np.ones(self.N_x+self.N_y)
            else:
                xy0 = xy0*np.ones(self.N_x+self.N_y)

        #xy = sopt.fsolve(self.ini_problem,xy0, jac=self.ini_dae_jacobian )

        
        if self.sopt_root_jac:
            sol = sopt.root(self.ini_problem, xy0, 
                            jac=self.ini_dae_jacobian, 
                            method=self.sopt_root_method, tol=self.initialization_tol)
        else:
            sol = sopt.root(self.ini_problem, xy0, method=self.sopt_root_method)

        self.initialization_ok = True
        if sol.success == False:
            print('initialization not found!')
            self.initialization_ok = False

            T = self.struct[0]['T'][:self.struct[0].it_store]
            X = self.struct[0]['X'][:self.struct[0].it_store,:]
            Y = self.struct[0]['Y'][:self.struct[0].it_store,:]
            Z = self.struct[0]['Z'][:self.struct[0].it_store,:]
            iters = self.struct[0]['iters'][:self.struct[0].it_store,:]

        if self.initialization_ok:
            xy = sol.x
            self.xy_prev = xy
            self.struct[0].x[:,0] = xy[0:self.N_x]
            self.struct[0].y_run[:,0] = xy[self.N_x:]

            ## y_ini to u_run
            for item in self.inputs_run_list:
                if item in self.y_ini_list:
                    self.struct[0][item] = self.struct[0].y_ini[self.y_ini_list.index(item)]

            ## u_ini to y_run
            for item in self.inputs_ini_list:
                if item in self.y_run_list:
                    self.struct[0].y_run[self.y_run_list.index(item)] = self.struct[0][item]


            #xy = sopt.fsolve(self.ini_problem,xy0, jac=self.ini_dae_jacobian )
            if self.sopt_root_jac:
                sol = sopt.root(self.run_problem, xy0, 
                                jac=self.run_dae_jacobian, 
                                method=self.sopt_root_method, tol=self.initialization_tol)
            else:
                sol = sopt.root(self.run_problem, xy0, method=self.sopt_root_method)

            if self.compile:
                # evaluate f and g
                run(0.0,self.struct,2)
                run(0.0,self.struct,3)                
    
                # evaluate run jacobians 
                run(0.0,self.struct,10)
                run(0.0,self.struct,11)                
                run(0.0,self.struct,12) 
                run(0.0,self.struct,14) 
                
            else:
                # evaluate f and g
                run.py_func(0.0,self.struct,2)
                run.py_func(0.0,self.struct,3)                
    
                # evaluate run jacobians 
                run.py_func(0.0,self.struct,10)
                run.py_func(0.0,self.struct,11)                
                run.py_func(0.0,self.struct,12) 
                run.py_func(0.0,self.struct,14)                 
                
             
            # post process result    
            T = self.struct[0]['T'][:self.struct[0].it_store]
            X = self.struct[0]['X'][:self.struct[0].it_store,:]
            Y = self.struct[0]['Y'][:self.struct[0].it_store,:]
            Z = self.struct[0]['Z'][:self.struct[0].it_store,:]
            iters = self.struct[0]['iters'][:self.struct[0].it_store,:]
        
            self.T = T
            self.X = X
            self.Y = Y
            self.Z = Z
            self.iters = iters
            
        return self.initialization_ok
    
    
    def get_value(self,name):
        if name in self.inputs_run_list:
            value = self.struct[0][name]
        if name in self.x_list:
            idx = self.x_list.index(name)
            value = self.struct[0].x[idx,0]
        if name in self.y_run_list:
            idy = self.y_run_list.index(name)
            value = self.struct[0].y_run[idy,0]
        if name in self.params_list:
            value = self.struct[0][name]
        if name in self.outputs_list:
            value = self.struct[0].h[self.outputs_list.index(name),0] 

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
        if name_ in self.inputs_run_list:
            self.struct[0][name_] = value
            return
        elif name_ in self.params_list:
            self.struct[0][name_] = value
            return
        elif name_ in self.inputs_ini_list:
            self.struct[0][name_] = value
            return 
        else:
            print(f'Input or parameter {name_} not found.')

    def set_values(self,dictionary):
        
        for item in dictionary:
            self.set_value(item,dictionary[item])
            
            
    def report_x(self,value_format='5.2f', decimals=2):
        for item in self.x_list:
            print(f'{item:5s} = {self.get_value(item):5.{decimals}f}')

    def report_y(self,value_format='5.2f', decimals=2):
        for item in self.y_run_list:
            print(f'{item:5s} = {self.get_value(item):5.{decimals}f}')
            
    def report_u(self,value_format='5.2f', decimals=2):
        for item in self.inputs_run_list:
            print(f'{item:5s} = {self.get_value(item):5.{decimals}f}')

    def report_z(self,value_format='5.2f', decimals=2):
        for item in self.outputs_list:
            print(f'{item:5s} = {self.get_value(item):5.{decimals}f}')

    def report_params(self,value_format='5.2f', decimals=2):
        for item in self.params_list:
            print(f'{item:5s} = {self.get_value(item):5.{decimals}f}')
            
    def get_x(self):
        return self.struct[0].x
    
    def ss(self):
        
        ssate(self.struct,self.xy_prev.reshape(len(self.xy_prev),1))
        
        ## y_ini to y_run
        self.struct[0].y_run = self.struct[0].y_ini
        
        ## y_ini to u_run
        for item in self.yini2urun:
            self.struct[0][item] = self.struct[0].y_ini[self.y_ini_list.index(item)]
                
        ## u_ini to y_run
        for item in self.uini2yrun:
            self.struct[0].y_run[self.y_run_list.index(item)] = self.struct[0][item]






@numba.njit(cache=True)
def ini(struct,mode):

    # Parameters:
    R_R00R01 = struct[0].R_R00R01
    L_R00R01 = struct[0].L_R00R01
    C_R00R01 = struct[0].C_R00R01
    R_R02R01 = struct[0].R_R02R01
    L_R02R01 = struct[0].L_R02R01
    C_R02R01 = struct[0].C_R02R01
    R_R02R03 = struct[0].R_R02R03
    L_R02R03 = struct[0].L_R02R03
    C_R02R03 = struct[0].C_R02R03
    R_R03R04 = struct[0].R_R03R04
    L_R03R04 = struct[0].L_R03R04
    C_R03R04 = struct[0].C_R03R04
    R_R04R05 = struct[0].R_R04R05
    L_R04R05 = struct[0].L_R04R05
    C_R04R05 = struct[0].C_R04R05
    R_R04R12 = struct[0].R_R04R12
    L_R04R12 = struct[0].L_R04R12
    C_R04R12 = struct[0].C_R04R12
    R_R05R06 = struct[0].R_R05R06
    L_R05R06 = struct[0].L_R05R06
    C_R05R06 = struct[0].C_R05R06
    R_R06R07 = struct[0].R_R06R07
    L_R06R07 = struct[0].L_R06R07
    C_R06R07 = struct[0].C_R06R07
    R_R07R08 = struct[0].R_R07R08
    L_R07R08 = struct[0].L_R07R08
    C_R07R08 = struct[0].C_R07R08
    R_R08R09 = struct[0].R_R08R09
    L_R08R09 = struct[0].L_R08R09
    C_R08R09 = struct[0].C_R08R09
    R_R09R10 = struct[0].R_R09R10
    L_R09R10 = struct[0].L_R09R10
    C_R09R10 = struct[0].C_R09R10
    R_R09R17 = struct[0].R_R09R17
    L_R09R17 = struct[0].L_R09R17
    C_R09R17 = struct[0].C_R09R17
    R_R11R03 = struct[0].R_R11R03
    L_R11R03 = struct[0].L_R11R03
    C_R11R03 = struct[0].C_R11R03
    R_R12R13 = struct[0].R_R12R13
    L_R12R13 = struct[0].L_R12R13
    C_R12R13 = struct[0].C_R12R13
    R_R13R14 = struct[0].R_R13R14
    L_R13R14 = struct[0].L_R13R14
    C_R13R14 = struct[0].C_R13R14
    R_R14R15 = struct[0].R_R14R15
    L_R14R15 = struct[0].L_R14R15
    C_R14R15 = struct[0].C_R14R15
    R_R16R06 = struct[0].R_R16R06
    L_R16R06 = struct[0].L_R16R06
    C_R16R06 = struct[0].C_R16R06
    R_R18R10 = struct[0].R_R18R10
    L_R18R10 = struct[0].L_R18R10
    C_R18R10 = struct[0].C_R18R10
    i_R00_D = struct[0].i_R00_D
    i_R00_Q = struct[0].i_R00_Q
    i_R02_D = struct[0].i_R02_D
    i_R02_Q = struct[0].i_R02_Q
    i_R03_D = struct[0].i_R03_D
    i_R03_Q = struct[0].i_R03_Q
    i_R04_D = struct[0].i_R04_D
    i_R04_Q = struct[0].i_R04_Q
    i_R05_D = struct[0].i_R05_D
    i_R05_Q = struct[0].i_R05_Q
    i_R06_D = struct[0].i_R06_D
    i_R06_Q = struct[0].i_R06_Q
    i_R07_D = struct[0].i_R07_D
    i_R07_Q = struct[0].i_R07_Q
    i_R08_D = struct[0].i_R08_D
    i_R08_Q = struct[0].i_R08_Q
    i_R09_D = struct[0].i_R09_D
    i_R09_Q = struct[0].i_R09_Q
    i_R12_D = struct[0].i_R12_D
    i_R12_Q = struct[0].i_R12_Q
    i_R13_D = struct[0].i_R13_D
    i_R13_Q = struct[0].i_R13_Q
    omega = struct[0].omega
    L_t_G10 = struct[0].L_t_G10
    R_t_G10 = struct[0].R_t_G10
    C_m_G10 = struct[0].C_m_G10
    L_s_G10 = struct[0].L_s_G10
    R_s_G10 = struct[0].R_s_G10
    omega_G10 = struct[0].omega_G10
    G_d_G10 = struct[0].G_d_G10
    K_p_G10 = struct[0].K_p_G10
    T_p_G10 = struct[0].T_p_G10
    K_q_G10 = struct[0].K_q_G10
    T_q_G10 = struct[0].T_q_G10
    R_v_G10 = struct[0].R_v_G10
    X_v_G10 = struct[0].X_v_G10
    S_b_kVA_G10 = struct[0].S_b_kVA_G10
    U_b_G10 = struct[0].U_b_G10
    K_phi_G10 = struct[0].K_phi_G10
    H_G10 = struct[0].H_G10
    D_G10 = struct[0].D_G10
    T_vpoi_G10 = struct[0].T_vpoi_G10
    K_vpoi_G10 = struct[0].K_vpoi_G10
    T_f_G10 = struct[0].T_f_G10
    K_f_G10 = struct[0].K_f_G10
    L_t_G14 = struct[0].L_t_G14
    R_t_G14 = struct[0].R_t_G14
    C_m_G14 = struct[0].C_m_G14
    L_s_G14 = struct[0].L_s_G14
    R_s_G14 = struct[0].R_s_G14
    omega_G14 = struct[0].omega_G14
    G_d_G14 = struct[0].G_d_G14
    K_p_G14 = struct[0].K_p_G14
    T_p_G14 = struct[0].T_p_G14
    K_q_G14 = struct[0].K_q_G14
    T_q_G14 = struct[0].T_q_G14
    R_v_G14 = struct[0].R_v_G14
    X_v_G14 = struct[0].X_v_G14
    S_b_kVA_G14 = struct[0].S_b_kVA_G14
    U_b_G14 = struct[0].U_b_G14
    K_phi_G14 = struct[0].K_phi_G14
    H_G14 = struct[0].H_G14
    D_G14 = struct[0].D_G14
    T_vpoi_G14 = struct[0].T_vpoi_G14
    K_vpoi_G14 = struct[0].K_vpoi_G14
    T_f_G14 = struct[0].T_f_G14
    K_f_G14 = struct[0].K_f_G14
    K_f_sec = struct[0].K_f_sec
    
    # Inputs:
    T_i_R01 = struct[0].T_i_R01
    I_max_R01 = struct[0].I_max_R01
    p_R01_ref = struct[0].p_R01_ref
    q_R01_ref = struct[0].q_R01_ref
    T_i_R11 = struct[0].T_i_R11
    I_max_R11 = struct[0].I_max_R11
    p_R11_ref = struct[0].p_R11_ref
    q_R11_ref = struct[0].q_R11_ref
    T_i_R15 = struct[0].T_i_R15
    I_max_R15 = struct[0].I_max_R15
    p_R15_ref = struct[0].p_R15_ref
    q_R15_ref = struct[0].q_R15_ref
    T_i_R16 = struct[0].T_i_R16
    I_max_R16 = struct[0].I_max_R16
    p_R16_ref = struct[0].p_R16_ref
    q_R16_ref = struct[0].q_R16_ref
    T_i_R17 = struct[0].T_i_R17
    I_max_R17 = struct[0].I_max_R17
    p_R17_ref = struct[0].p_R17_ref
    q_R17_ref = struct[0].q_R17_ref
    T_i_R18 = struct[0].T_i_R18
    I_max_R18 = struct[0].I_max_R18
    p_R18_ref = struct[0].p_R18_ref
    q_R18_ref = struct[0].q_R18_ref
    v_dc_G10 = struct[0].v_dc_G10
    p_m_ref_G10 = struct[0].p_m_ref_G10
    q_s_ref_G10 = struct[0].q_s_ref_G10
    v_s_ref_G10 = struct[0].v_s_ref_G10
    omega_ref_G10 = struct[0].omega_ref_G10
    p_r_G10 = struct[0].p_r_G10
    q_r_G10 = struct[0].q_r_G10
    v_dc_G14 = struct[0].v_dc_G14
    p_m_ref_G14 = struct[0].p_m_ref_G14
    q_s_ref_G14 = struct[0].q_s_ref_G14
    v_s_ref_G14 = struct[0].v_s_ref_G14
    omega_ref_G14 = struct[0].omega_ref_G14
    p_r_G14 = struct[0].p_r_G14
    q_r_G14 = struct[0].q_r_G14
    
    # Dynamical states:
    i_R01_D = struct[0].x[0,0]
    i_R01_Q = struct[0].x[1,0]
    i_R11_D = struct[0].x[2,0]
    i_R11_Q = struct[0].x[3,0]
    i_R15_D = struct[0].x[4,0]
    i_R15_Q = struct[0].x[5,0]
    i_R16_D = struct[0].x[6,0]
    i_R16_Q = struct[0].x[7,0]
    i_R17_D = struct[0].x[8,0]
    i_R17_Q = struct[0].x[9,0]
    i_R18_D = struct[0].x[10,0]
    i_R18_Q = struct[0].x[11,0]
    phi_G10 = struct[0].x[12,0]
    omega_v_G10 = struct[0].x[13,0]
    xi_q_G10 = struct[0].x[14,0]
    omega_rads_G10 = struct[0].x[15,0]
    omega_v_filt_G10 = struct[0].x[16,0]
    v_s_filt_G10 = struct[0].x[17,0]
    phi_G14 = struct[0].x[18,0]
    omega_v_G14 = struct[0].x[19,0]
    xi_q_G14 = struct[0].x[20,0]
    omega_rads_G14 = struct[0].x[21,0]
    omega_v_filt_G14 = struct[0].x[22,0]
    v_s_filt_G14 = struct[0].x[23,0]
    xi_f_sec = struct[0].x[24,0]
    
    # Algebraic states:
    i_l_R00R01_D = struct[0].y_ini[0,0]
    i_l_R00R01_Q = struct[0].y_ini[1,0]
    i_l_R02R01_D = struct[0].y_ini[2,0]
    i_l_R02R01_Q = struct[0].y_ini[3,0]
    i_l_R02R03_D = struct[0].y_ini[4,0]
    i_l_R02R03_Q = struct[0].y_ini[5,0]
    i_l_R03R04_D = struct[0].y_ini[6,0]
    i_l_R03R04_Q = struct[0].y_ini[7,0]
    i_l_R04R05_D = struct[0].y_ini[8,0]
    i_l_R04R05_Q = struct[0].y_ini[9,0]
    i_l_R04R12_D = struct[0].y_ini[10,0]
    i_l_R04R12_Q = struct[0].y_ini[11,0]
    i_l_R05R06_D = struct[0].y_ini[12,0]
    i_l_R05R06_Q = struct[0].y_ini[13,0]
    i_l_R06R07_D = struct[0].y_ini[14,0]
    i_l_R06R07_Q = struct[0].y_ini[15,0]
    i_l_R07R08_D = struct[0].y_ini[16,0]
    i_l_R07R08_Q = struct[0].y_ini[17,0]
    i_l_R08R09_D = struct[0].y_ini[18,0]
    i_l_R08R09_Q = struct[0].y_ini[19,0]
    i_l_R09R10_D = struct[0].y_ini[20,0]
    i_l_R09R10_Q = struct[0].y_ini[21,0]
    i_l_R09R17_D = struct[0].y_ini[22,0]
    i_l_R09R17_Q = struct[0].y_ini[23,0]
    i_l_R11R03_D = struct[0].y_ini[24,0]
    i_l_R11R03_Q = struct[0].y_ini[25,0]
    i_l_R12R13_D = struct[0].y_ini[26,0]
    i_l_R12R13_Q = struct[0].y_ini[27,0]
    i_l_R13R14_D = struct[0].y_ini[28,0]
    i_l_R13R14_Q = struct[0].y_ini[29,0]
    i_l_R14R15_D = struct[0].y_ini[30,0]
    i_l_R14R15_Q = struct[0].y_ini[31,0]
    i_l_R16R06_D = struct[0].y_ini[32,0]
    i_l_R16R06_Q = struct[0].y_ini[33,0]
    i_l_R18R10_D = struct[0].y_ini[34,0]
    i_l_R18R10_Q = struct[0].y_ini[35,0]
    v_R00_D = struct[0].y_ini[36,0]
    v_R00_Q = struct[0].y_ini[37,0]
    v_R01_D = struct[0].y_ini[38,0]
    v_R01_Q = struct[0].y_ini[39,0]
    v_R02_D = struct[0].y_ini[40,0]
    v_R02_Q = struct[0].y_ini[41,0]
    v_R03_D = struct[0].y_ini[42,0]
    v_R03_Q = struct[0].y_ini[43,0]
    v_R04_D = struct[0].y_ini[44,0]
    v_R04_Q = struct[0].y_ini[45,0]
    v_R05_D = struct[0].y_ini[46,0]
    v_R05_Q = struct[0].y_ini[47,0]
    v_R06_D = struct[0].y_ini[48,0]
    v_R06_Q = struct[0].y_ini[49,0]
    v_R07_D = struct[0].y_ini[50,0]
    v_R07_Q = struct[0].y_ini[51,0]
    v_R08_D = struct[0].y_ini[52,0]
    v_R08_Q = struct[0].y_ini[53,0]
    v_R09_D = struct[0].y_ini[54,0]
    v_R09_Q = struct[0].y_ini[55,0]
    v_R10_D = struct[0].y_ini[56,0]
    v_R10_Q = struct[0].y_ini[57,0]
    v_R11_D = struct[0].y_ini[58,0]
    v_R11_Q = struct[0].y_ini[59,0]
    v_R12_D = struct[0].y_ini[60,0]
    v_R12_Q = struct[0].y_ini[61,0]
    v_R13_D = struct[0].y_ini[62,0]
    v_R13_Q = struct[0].y_ini[63,0]
    v_R14_D = struct[0].y_ini[64,0]
    v_R14_Q = struct[0].y_ini[65,0]
    v_R15_D = struct[0].y_ini[66,0]
    v_R15_Q = struct[0].y_ini[67,0]
    v_R16_D = struct[0].y_ini[68,0]
    v_R16_Q = struct[0].y_ini[69,0]
    v_R17_D = struct[0].y_ini[70,0]
    v_R17_Q = struct[0].y_ini[71,0]
    v_R18_D = struct[0].y_ini[72,0]
    v_R18_Q = struct[0].y_ini[73,0]
    i_R01_d_ref = struct[0].y_ini[74,0]
    i_R01_q_ref = struct[0].y_ini[75,0]
    i_R11_d_ref = struct[0].y_ini[76,0]
    i_R11_q_ref = struct[0].y_ini[77,0]
    i_R15_d_ref = struct[0].y_ini[78,0]
    i_R15_q_ref = struct[0].y_ini[79,0]
    i_R16_d_ref = struct[0].y_ini[80,0]
    i_R16_q_ref = struct[0].y_ini[81,0]
    i_R17_d_ref = struct[0].y_ini[82,0]
    i_R17_q_ref = struct[0].y_ini[83,0]
    i_R18_d_ref = struct[0].y_ini[84,0]
    i_R18_q_ref = struct[0].y_ini[85,0]
    i_tD_G10 = struct[0].y_ini[86,0]
    i_tQ_G10 = struct[0].y_ini[87,0]
    v_mD_G10 = struct[0].y_ini[88,0]
    v_mQ_G10 = struct[0].y_ini[89,0]
    i_sD_G10 = struct[0].y_ini[90,0]
    i_sQ_G10 = struct[0].y_ini[91,0]
    i_R10_D = struct[0].y_ini[92,0]
    i_R10_Q = struct[0].y_ini[93,0]
    v_sD_G10 = struct[0].y_ini[94,0]
    v_sQ_G10 = struct[0].y_ini[95,0]
    eta_d_G10 = struct[0].y_ini[96,0]
    eta_q_G10 = struct[0].y_ini[97,0]
    eta_D_G10 = struct[0].y_ini[98,0]
    eta_Q_G10 = struct[0].y_ini[99,0]
    v_md_G10 = struct[0].y_ini[100,0]
    v_mq_G10 = struct[0].y_ini[101,0]
    v_sd_G10 = struct[0].y_ini[102,0]
    v_sq_G10 = struct[0].y_ini[103,0]
    i_td_G10 = struct[0].y_ini[104,0]
    i_tq_G10 = struct[0].y_ini[105,0]
    i_sd_G10 = struct[0].y_ini[106,0]
    i_sq_G10 = struct[0].y_ini[107,0]
    DV_sat_G10 = struct[0].y_ini[108,0]
    p_s_pu_G10 = struct[0].y_ini[109,0]
    q_s_pu_G10 = struct[0].y_ini[110,0]
    p_m_ref_G10 = struct[0].y_ini[111,0]
    q_s_ref_G10 = struct[0].y_ini[112,0]
    i_tD_G14 = struct[0].y_ini[113,0]
    i_tQ_G14 = struct[0].y_ini[114,0]
    v_mD_G14 = struct[0].y_ini[115,0]
    v_mQ_G14 = struct[0].y_ini[116,0]
    i_sD_G14 = struct[0].y_ini[117,0]
    i_sQ_G14 = struct[0].y_ini[118,0]
    i_R14_D = struct[0].y_ini[119,0]
    i_R14_Q = struct[0].y_ini[120,0]
    v_sD_G14 = struct[0].y_ini[121,0]
    v_sQ_G14 = struct[0].y_ini[122,0]
    eta_d_G14 = struct[0].y_ini[123,0]
    eta_q_G14 = struct[0].y_ini[124,0]
    eta_D_G14 = struct[0].y_ini[125,0]
    eta_Q_G14 = struct[0].y_ini[126,0]
    v_md_G14 = struct[0].y_ini[127,0]
    v_mq_G14 = struct[0].y_ini[128,0]
    v_sd_G14 = struct[0].y_ini[129,0]
    v_sq_G14 = struct[0].y_ini[130,0]
    i_td_G14 = struct[0].y_ini[131,0]
    i_tq_G14 = struct[0].y_ini[132,0]
    i_sd_G14 = struct[0].y_ini[133,0]
    i_sq_G14 = struct[0].y_ini[134,0]
    DV_sat_G14 = struct[0].y_ini[135,0]
    p_s_pu_G14 = struct[0].y_ini[136,0]
    q_s_pu_G14 = struct[0].y_ini[137,0]
    p_m_ref_G14 = struct[0].y_ini[138,0]
    q_s_ref_G14 = struct[0].y_ini[139,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -100.0*i_R01_D + 100.0*Piecewise(np.array([(-I_max_R01, I_max_R01 < -i_R01_d_ref), (I_max_R01, I_max_R01 < i_R01_d_ref), (i_R01_d_ref, True)]))
        struct[0].f[1,0] = -100.0*i_R01_Q + 100.0*Piecewise(np.array([(-I_max_R01, I_max_R01 < -i_R01_q_ref), (I_max_R01, I_max_R01 < i_R01_q_ref), (i_R01_q_ref, True)]))
        struct[0].f[2,0] = -100.0*i_R11_D + 100.0*Piecewise(np.array([(-I_max_R11, I_max_R11 < -i_R11_d_ref), (I_max_R11, I_max_R11 < i_R11_d_ref), (i_R11_d_ref, True)]))
        struct[0].f[3,0] = -100.0*i_R11_Q + 100.0*Piecewise(np.array([(-I_max_R11, I_max_R11 < -i_R11_q_ref), (I_max_R11, I_max_R11 < i_R11_q_ref), (i_R11_q_ref, True)]))
        struct[0].f[4,0] = -100.0*i_R15_D + 100.0*Piecewise(np.array([(-I_max_R15, I_max_R15 < -i_R15_d_ref), (I_max_R15, I_max_R15 < i_R15_d_ref), (i_R15_d_ref, True)]))
        struct[0].f[5,0] = -100.0*i_R15_Q + 100.0*Piecewise(np.array([(-I_max_R15, I_max_R15 < -i_R15_q_ref), (I_max_R15, I_max_R15 < i_R15_q_ref), (i_R15_q_ref, True)]))
        struct[0].f[6,0] = -100.0*i_R16_D + 100.0*Piecewise(np.array([(-I_max_R16, I_max_R16 < -i_R16_d_ref), (I_max_R16, I_max_R16 < i_R16_d_ref), (i_R16_d_ref, True)]))
        struct[0].f[7,0] = -100.0*i_R16_Q + 100.0*Piecewise(np.array([(-I_max_R16, I_max_R16 < -i_R16_q_ref), (I_max_R16, I_max_R16 < i_R16_q_ref), (i_R16_q_ref, True)]))
        struct[0].f[8,0] = -100.0*i_R17_D + 100.0*Piecewise(np.array([(-I_max_R17, I_max_R17 < -i_R17_d_ref), (I_max_R17, I_max_R17 < i_R17_d_ref), (i_R17_d_ref, True)]))
        struct[0].f[9,0] = -100.0*i_R17_Q + 100.0*Piecewise(np.array([(-I_max_R17, I_max_R17 < -i_R17_q_ref), (I_max_R17, I_max_R17 < i_R17_q_ref), (i_R17_q_ref, True)]))
        struct[0].f[10,0] = -100.0*i_R18_D + 100.0*Piecewise(np.array([(-I_max_R18, I_max_R18 < -i_R18_d_ref), (I_max_R18, I_max_R18 < i_R18_d_ref), (i_R18_d_ref, True)]))
        struct[0].f[11,0] = -100.0*i_R18_Q + 100.0*Piecewise(np.array([(-I_max_R18, I_max_R18 < -i_R18_q_ref), (I_max_R18, I_max_R18 < i_R18_q_ref), (i_R18_q_ref, True)]))
        struct[0].f[12,0] = -K_phi_G10*phi_G10 + 314.159265358979*omega_v_G10 - 314.159265358979*(H_G10*omega_v_G10 + H_G14*omega_v_G14)/(H_G10 + H_G14)
        struct[0].f[13,0] = (-D_G10*(omega_v_G10 - 1.0) + p_m_ref_G10 - p_s_pu_G10)/(2*H_G10)
        struct[0].f[14,0] = -q_s_pu_G10 + q_s_ref_G10
        struct[0].f[15,0] = -1.0*omega_rads_G10 + 314.159265358979*omega_v_G10
        struct[0].f[16,0] = (omega_v_G10 - omega_v_filt_G10)/T_f_G10
        struct[0].f[17,0] = (-v_s_filt_G10 + 0.00306186217847897*(v_sd_G10**2 + v_sq_G10**2)**0.5)/T_vpoi_G10
        struct[0].f[18,0] = -K_phi_G14*phi_G14 + 314.159265358979*omega_v_G14 - 314.159265358979*(H_G10*omega_v_G10 + H_G14*omega_v_G14)/(H_G10 + H_G14)
        struct[0].f[19,0] = (-D_G14*(omega_v_G14 - 1.0) + p_m_ref_G14 - p_s_pu_G14)/(2*H_G14)
        struct[0].f[20,0] = -q_s_pu_G14 + q_s_ref_G14
        struct[0].f[21,0] = -1.0*omega_rads_G14 + 314.159265358979*omega_v_G14
        struct[0].f[22,0] = (omega_v_G14 - omega_v_filt_G14)/T_f_G14
        struct[0].f[23,0] = (-v_s_filt_G14 + 0.00306186217847897*(v_sd_G14**2 + v_sq_G14**2)**0.5)/T_vpoi_G14
        struct[0].f[24,0] = 1 - (H_G10*omega_v_G10 + H_G14*omega_v_G14)/(H_G10 + H_G14)
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy_ini) @ np.ascontiguousarray(struct[0].y_ini)

        struct[0].g[36,0] = C_R00R01*omega*v_R00_Q/2 + i_R00_D - i_l_R00R01_D
        struct[0].g[37,0] = -C_R00R01*omega*v_R00_D/2 + i_R00_Q - i_l_R00R01_Q
        struct[0].g[38,0] = i_R01_D + i_l_R00R01_D + i_l_R02R01_D + omega*v_R01_Q*(C_R00R01/2 + C_R02R01/2)
        struct[0].g[39,0] = i_R01_Q + i_l_R00R01_Q + i_l_R02R01_Q - omega*v_R01_D*(C_R00R01/2 + C_R02R01/2)
        struct[0].g[40,0] = i_R02_D - i_l_R02R01_D - i_l_R02R03_D + omega*v_R02_Q*(C_R02R01/2 + C_R02R03/2)
        struct[0].g[41,0] = i_R02_Q - i_l_R02R01_Q - i_l_R02R03_Q - omega*v_R02_D*(C_R02R01/2 + C_R02R03/2)
        struct[0].g[42,0] = i_R03_D + i_l_R02R03_D - i_l_R03R04_D + i_l_R11R03_D + omega*v_R03_Q*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].g[43,0] = i_R03_Q + i_l_R02R03_Q - i_l_R03R04_Q + i_l_R11R03_Q - omega*v_R03_D*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].g[44,0] = i_R04_D + i_l_R03R04_D - i_l_R04R05_D - i_l_R04R12_D + omega*v_R04_Q*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].g[45,0] = i_R04_Q + i_l_R03R04_Q - i_l_R04R05_Q - i_l_R04R12_Q - omega*v_R04_D*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].g[46,0] = i_R05_D + i_l_R04R05_D - i_l_R05R06_D + omega*v_R05_Q*(C_R04R05/2 + C_R05R06/2)
        struct[0].g[47,0] = i_R05_Q + i_l_R04R05_Q - i_l_R05R06_Q - omega*v_R05_D*(C_R04R05/2 + C_R05R06/2)
        struct[0].g[48,0] = i_R06_D + i_l_R05R06_D - i_l_R06R07_D + i_l_R16R06_D + omega*v_R06_Q*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].g[49,0] = i_R06_Q + i_l_R05R06_Q - i_l_R06R07_Q + i_l_R16R06_Q - omega*v_R06_D*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].g[50,0] = i_R07_D + i_l_R06R07_D - i_l_R07R08_D + omega*v_R07_Q*(C_R06R07/2 + C_R07R08/2)
        struct[0].g[51,0] = i_R07_Q + i_l_R06R07_Q - i_l_R07R08_Q - omega*v_R07_D*(C_R06R07/2 + C_R07R08/2)
        struct[0].g[52,0] = i_R08_D + i_l_R07R08_D - i_l_R08R09_D + omega*v_R08_Q*(C_R07R08/2 + C_R08R09/2)
        struct[0].g[53,0] = i_R08_Q + i_l_R07R08_Q - i_l_R08R09_Q - omega*v_R08_D*(C_R07R08/2 + C_R08R09/2)
        struct[0].g[54,0] = i_R09_D + i_l_R08R09_D - i_l_R09R10_D - i_l_R09R17_D + omega*v_R09_Q*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].g[55,0] = i_R09_Q + i_l_R08R09_Q - i_l_R09R10_Q - i_l_R09R17_Q - omega*v_R09_D*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].g[58,0] = C_R11R03*omega*v_R11_Q/2 + i_R11_D - i_l_R11R03_D
        struct[0].g[59,0] = -C_R11R03*omega*v_R11_D/2 + i_R11_Q - i_l_R11R03_Q
        struct[0].g[60,0] = i_R12_D + i_l_R04R12_D - i_l_R12R13_D + omega*v_R12_Q*(C_R04R12/2 + C_R12R13/2)
        struct[0].g[61,0] = i_R12_Q + i_l_R04R12_Q - i_l_R12R13_Q - omega*v_R12_D*(C_R04R12/2 + C_R12R13/2)
        struct[0].g[62,0] = i_R13_D + i_l_R12R13_D - i_l_R13R14_D + omega*v_R13_Q*(C_R12R13/2 + C_R13R14/2)
        struct[0].g[63,0] = i_R13_Q + i_l_R12R13_Q - i_l_R13R14_Q - omega*v_R13_D*(C_R12R13/2 + C_R13R14/2)
        struct[0].g[66,0] = C_R14R15*omega*v_R15_Q/2 + i_R15_D + i_l_R14R15_D
        struct[0].g[67,0] = -C_R14R15*omega*v_R15_D/2 + i_R15_Q + i_l_R14R15_Q
        struct[0].g[68,0] = C_R16R06*omega*v_R16_Q/2 + i_R16_D - i_l_R16R06_D
        struct[0].g[69,0] = -C_R16R06*omega*v_R16_D/2 + i_R16_Q - i_l_R16R06_Q
        struct[0].g[70,0] = C_R09R17*omega*v_R17_Q/2 + i_R17_D + i_l_R09R17_D
        struct[0].g[71,0] = -C_R09R17*omega*v_R17_D/2 + i_R17_Q + i_l_R09R17_Q
        struct[0].g[72,0] = C_R18R10*omega*v_R18_Q/2 + i_R18_D - i_l_R18R10_D
        struct[0].g[73,0] = -C_R18R10*omega*v_R18_D/2 + i_R18_Q - i_l_R18R10_Q
        struct[0].g[74,0] = -i_R01_d_ref + (-0.666666666666667*p_R01_ref*v_R01_D - 0.666666666666667*q_R01_ref*v_R01_Q)*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)]))
        struct[0].g[75,0] = -i_R01_q_ref - (0.666666666666667*p_R01_ref*v_R01_Q - 0.666666666666667*q_R01_ref*v_R01_D)*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)]))
        struct[0].g[76,0] = -i_R11_d_ref + (-0.666666666666667*p_R11_ref*v_R11_D - 0.666666666666667*q_R11_ref*v_R11_Q)*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)]))
        struct[0].g[77,0] = -i_R11_q_ref - (0.666666666666667*p_R11_ref*v_R11_Q - 0.666666666666667*q_R11_ref*v_R11_D)*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)]))
        struct[0].g[78,0] = -i_R15_d_ref + (-0.666666666666667*p_R15_ref*v_R15_D - 0.666666666666667*q_R15_ref*v_R15_Q)*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)]))
        struct[0].g[79,0] = -i_R15_q_ref - (0.666666666666667*p_R15_ref*v_R15_Q - 0.666666666666667*q_R15_ref*v_R15_D)*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)]))
        struct[0].g[80,0] = -i_R16_d_ref + (-0.666666666666667*p_R16_ref*v_R16_D - 0.666666666666667*q_R16_ref*v_R16_Q)*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)]))
        struct[0].g[81,0] = -i_R16_q_ref - (0.666666666666667*p_R16_ref*v_R16_Q - 0.666666666666667*q_R16_ref*v_R16_D)*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)]))
        struct[0].g[82,0] = -i_R17_d_ref + (-0.666666666666667*p_R17_ref*v_R17_D - 0.666666666666667*q_R17_ref*v_R17_Q)*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)]))
        struct[0].g[83,0] = -i_R17_q_ref - (0.666666666666667*p_R17_ref*v_R17_Q - 0.666666666666667*q_R17_ref*v_R17_D)*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)]))
        struct[0].g[84,0] = -i_R18_d_ref + (-0.666666666666667*p_R18_ref*v_R18_D - 0.666666666666667*q_R18_ref*v_R18_Q)*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)]))
        struct[0].g[85,0] = -i_R18_q_ref - (0.666666666666667*p_R18_ref*v_R18_Q - 0.666666666666667*q_R18_ref*v_R18_D)*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)]))
        struct[0].g[86,0] = (L_t_G10*i_tQ_G10*omega_G10 - R_t_G10*i_tD_G10 + eta_D_G10*v_dc_G10/2 - v_mD_G10)/L_t_G10
        struct[0].g[87,0] = (-L_t_G10*i_tD_G10*omega_G10 - R_t_G10*i_tQ_G10 + eta_Q_G10*v_dc_G10/2 - v_mQ_G10)/L_t_G10
        struct[0].g[88,0] = (C_m_G10*omega_G10*v_mQ_G10 - G_d_G10*v_mD_G10 - i_sD_G10 + i_tD_G10)/C_m_G10
        struct[0].g[89,0] = (-C_m_G10*omega_G10*v_mD_G10 - G_d_G10*v_mQ_G10 - i_sQ_G10 + i_tQ_G10)/C_m_G10
        struct[0].g[90,0] = (L_s_G10*i_sQ_G10*omega_G10 - R_s_G10*i_sD_G10 + v_mD_G10 - v_sD_G10)/L_s_G10
        struct[0].g[91,0] = (-L_s_G10*i_sD_G10*omega_G10 - R_s_G10*i_sQ_G10 + v_mQ_G10 - v_sQ_G10)/L_s_G10
        struct[0].g[96,0] = eta_d_G10 - 2*(-0.8*R_v_G10*i_sd_G10 + 0.8*X_v_G10*i_sq_G10)/v_dc_G10
        struct[0].g[97,0] = eta_q_G10 - 2*(326.59863237109*DV_sat_G10 - 0.8*R_v_G10*i_sq_G10 - 0.8*X_v_G10*i_sd_G10 + 326.59863237109)/v_dc_G10
        struct[0].g[108,0] = DV_sat_G10 - K_q_G10*(-q_s_pu_G10 + q_s_ref_G10 + xi_q_G10/T_q_G10)
        struct[0].g[109,0] = 7.5e-6*i_sd_G10*v_sd_G10 + 7.5e-6*i_sq_G10*v_sq_G10 - p_s_pu_G10
        struct[0].g[110,0] = 7.5e-6*i_sd_G10*v_sq_G10 - 7.5e-6*i_sq_G10*v_sd_G10 - q_s_pu_G10
        struct[0].g[111,0] = K_f_G10*(omega_ref_G10 - omega_v_filt_G10) + K_f_sec*xi_f_sec/2 - p_m_ref_G10 + p_r_G10
        struct[0].g[112,0] = K_vpoi_G10*(-v_s_filt_G10 + v_s_ref_G10) + q_r_G10 - q_s_ref_G10
        struct[0].g[113,0] = (L_t_G14*i_tQ_G14*omega_G14 - R_t_G14*i_tD_G14 + eta_D_G14*v_dc_G14/2 - v_mD_G14)/L_t_G14
        struct[0].g[114,0] = (-L_t_G14*i_tD_G14*omega_G14 - R_t_G14*i_tQ_G14 + eta_Q_G14*v_dc_G14/2 - v_mQ_G14)/L_t_G14
        struct[0].g[115,0] = (C_m_G14*omega_G14*v_mQ_G14 - G_d_G14*v_mD_G14 - i_sD_G14 + i_tD_G14)/C_m_G14
        struct[0].g[116,0] = (-C_m_G14*omega_G14*v_mD_G14 - G_d_G14*v_mQ_G14 - i_sQ_G14 + i_tQ_G14)/C_m_G14
        struct[0].g[117,0] = (L_s_G14*i_sQ_G14*omega_G14 - R_s_G14*i_sD_G14 + v_mD_G14 - v_sD_G14)/L_s_G14
        struct[0].g[118,0] = (-L_s_G14*i_sD_G14*omega_G14 - R_s_G14*i_sQ_G14 + v_mQ_G14 - v_sQ_G14)/L_s_G14
        struct[0].g[123,0] = eta_d_G14 - 2*(-0.8*R_v_G14*i_sd_G14 + 0.8*X_v_G14*i_sq_G14)/v_dc_G14
        struct[0].g[124,0] = eta_q_G14 - 2*(326.59863237109*DV_sat_G14 - 0.8*R_v_G14*i_sq_G14 - 0.8*X_v_G14*i_sd_G14 + 326.59863237109)/v_dc_G14
        struct[0].g[135,0] = DV_sat_G14 - K_q_G14*(-q_s_pu_G14 + q_s_ref_G14 + xi_q_G14/T_q_G14)
        struct[0].g[136,0] = 7.5e-6*i_sd_G14*v_sd_G14 + 7.5e-6*i_sq_G14*v_sq_G14 - p_s_pu_G14
        struct[0].g[137,0] = 7.5e-6*i_sd_G14*v_sq_G14 - 7.5e-6*i_sq_G14*v_sd_G14 - q_s_pu_G14
        struct[0].g[138,0] = K_f_G14*(omega_ref_G14 - omega_v_filt_G14) + K_f_sec*xi_f_sec/2 - p_m_ref_G14 + p_r_G14
        struct[0].g[139,0] = K_vpoi_G14*(-v_s_filt_G14 + v_s_ref_G14) + q_r_G14 - q_s_ref_G14
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = i_sD_G10
        struct[0].h[1,0] = i_sQ_G10
        struct[0].h[2,0] = i_sD_G14
        struct[0].h[3,0] = i_sQ_G14
    

    if mode == 10:

        struct[0].Fx_ini[12,12] = -K_phi_G10
        struct[0].Fx_ini[12,13] = -314.159265358979*H_G10/(H_G10 + H_G14) + 314.159265358979
        struct[0].Fx_ini[12,19] = -314.159265358979*H_G14/(H_G10 + H_G14)
        struct[0].Fx_ini[13,13] = -D_G10/(2*H_G10)
        struct[0].Fx_ini[16,13] = 1/T_f_G10
        struct[0].Fx_ini[16,16] = -1/T_f_G10
        struct[0].Fx_ini[17,17] = -1/T_vpoi_G10
        struct[0].Fx_ini[18,13] = -314.159265358979*H_G10/(H_G10 + H_G14)
        struct[0].Fx_ini[18,18] = -K_phi_G14
        struct[0].Fx_ini[18,19] = -314.159265358979*H_G14/(H_G10 + H_G14) + 314.159265358979
        struct[0].Fx_ini[19,19] = -D_G14/(2*H_G14)
        struct[0].Fx_ini[22,19] = 1/T_f_G14
        struct[0].Fx_ini[22,22] = -1/T_f_G14
        struct[0].Fx_ini[23,23] = -1/T_vpoi_G14
        struct[0].Fx_ini[24,13] = -H_G10/(H_G10 + H_G14)
        struct[0].Fx_ini[24,19] = -H_G14/(H_G10 + H_G14)

    if mode == 11:

        struct[0].Fy_ini[0,74] = 100.0*Piecewise(np.array([(0, (I_max_R01 < i_R01_d_ref) | (I_max_R01 < -i_R01_d_ref)), (1, True)])) 
        struct[0].Fy_ini[1,75] = 100.0*Piecewise(np.array([(0, (I_max_R01 < i_R01_q_ref) | (I_max_R01 < -i_R01_q_ref)), (1, True)])) 
        struct[0].Fy_ini[2,76] = 100.0*Piecewise(np.array([(0, (I_max_R11 < i_R11_d_ref) | (I_max_R11 < -i_R11_d_ref)), (1, True)])) 
        struct[0].Fy_ini[3,77] = 100.0*Piecewise(np.array([(0, (I_max_R11 < i_R11_q_ref) | (I_max_R11 < -i_R11_q_ref)), (1, True)])) 
        struct[0].Fy_ini[4,78] = 100.0*Piecewise(np.array([(0, (I_max_R15 < i_R15_d_ref) | (I_max_R15 < -i_R15_d_ref)), (1, True)])) 
        struct[0].Fy_ini[5,79] = 100.0*Piecewise(np.array([(0, (I_max_R15 < i_R15_q_ref) | (I_max_R15 < -i_R15_q_ref)), (1, True)])) 
        struct[0].Fy_ini[6,80] = 100.0*Piecewise(np.array([(0, (I_max_R16 < i_R16_d_ref) | (I_max_R16 < -i_R16_d_ref)), (1, True)])) 
        struct[0].Fy_ini[7,81] = 100.0*Piecewise(np.array([(0, (I_max_R16 < i_R16_q_ref) | (I_max_R16 < -i_R16_q_ref)), (1, True)])) 
        struct[0].Fy_ini[8,82] = 100.0*Piecewise(np.array([(0, (I_max_R17 < i_R17_d_ref) | (I_max_R17 < -i_R17_d_ref)), (1, True)])) 
        struct[0].Fy_ini[9,83] = 100.0*Piecewise(np.array([(0, (I_max_R17 < i_R17_q_ref) | (I_max_R17 < -i_R17_q_ref)), (1, True)])) 
        struct[0].Fy_ini[10,84] = 100.0*Piecewise(np.array([(0, (I_max_R18 < i_R18_d_ref) | (I_max_R18 < -i_R18_d_ref)), (1, True)])) 
        struct[0].Fy_ini[11,85] = 100.0*Piecewise(np.array([(0, (I_max_R18 < i_R18_q_ref) | (I_max_R18 < -i_R18_q_ref)), (1, True)])) 
        struct[0].Fy_ini[13,109] = -1/(2*H_G10) 
        struct[0].Fy_ini[13,111] = 1/(2*H_G10) 
        struct[0].Fy_ini[14,110] = -1 
        struct[0].Fy_ini[14,112] = 1 
        struct[0].Fy_ini[17,102] = 0.00306186217847897*v_sd_G10/(T_vpoi_G10*(v_sd_G10**2 + v_sq_G10**2)**0.5) 
        struct[0].Fy_ini[17,103] = 0.00306186217847897*v_sq_G10/(T_vpoi_G10*(v_sd_G10**2 + v_sq_G10**2)**0.5) 
        struct[0].Fy_ini[19,136] = -1/(2*H_G14) 
        struct[0].Fy_ini[19,138] = 1/(2*H_G14) 
        struct[0].Fy_ini[20,137] = -1 
        struct[0].Fy_ini[20,139] = 1 
        struct[0].Fy_ini[23,129] = 0.00306186217847897*v_sd_G14/(T_vpoi_G14*(v_sd_G14**2 + v_sq_G14**2)**0.5) 
        struct[0].Fy_ini[23,130] = 0.00306186217847897*v_sq_G14/(T_vpoi_G14*(v_sd_G14**2 + v_sq_G14**2)**0.5) 

        struct[0].Gx_ini[38,0] = 1
        struct[0].Gx_ini[39,1] = 1
        struct[0].Gx_ini[58,2] = 1
        struct[0].Gx_ini[59,3] = 1
        struct[0].Gx_ini[66,4] = 1
        struct[0].Gx_ini[67,5] = 1
        struct[0].Gx_ini[68,6] = 1
        struct[0].Gx_ini[69,7] = 1
        struct[0].Gx_ini[70,8] = 1
        struct[0].Gx_ini[71,9] = 1
        struct[0].Gx_ini[72,10] = 1
        struct[0].Gx_ini[73,11] = 1
        struct[0].Gx_ini[98,12] = -eta_d_G10*sin(phi_G10) - eta_q_G10*cos(phi_G10)
        struct[0].Gx_ini[99,12] = eta_d_G10*cos(phi_G10) - eta_q_G10*sin(phi_G10)
        struct[0].Gx_ini[100,12] = -v_mD_G10*sin(phi_G10) + v_mQ_G10*cos(phi_G10)
        struct[0].Gx_ini[101,12] = -v_mD_G10*cos(phi_G10) - v_mQ_G10*sin(phi_G10)
        struct[0].Gx_ini[102,12] = -v_sD_G10*sin(phi_G10) + v_sQ_G10*cos(phi_G10)
        struct[0].Gx_ini[103,12] = -v_sD_G10*cos(phi_G10) - v_sQ_G10*sin(phi_G10)
        struct[0].Gx_ini[104,12] = -i_tD_G10*sin(phi_G10) + i_tQ_G10*cos(phi_G10)
        struct[0].Gx_ini[105,12] = -i_tD_G10*cos(phi_G10) - i_tQ_G10*sin(phi_G10)
        struct[0].Gx_ini[106,12] = -i_sD_G10*sin(phi_G10) + i_sQ_G10*cos(phi_G10)
        struct[0].Gx_ini[107,12] = -i_sD_G10*cos(phi_G10) - i_sQ_G10*sin(phi_G10)
        struct[0].Gx_ini[108,14] = -K_q_G10/T_q_G10
        struct[0].Gx_ini[111,16] = -K_f_G10
        struct[0].Gx_ini[111,24] = K_f_sec/2
        struct[0].Gx_ini[112,17] = -K_vpoi_G10
        struct[0].Gx_ini[125,18] = -eta_d_G14*sin(phi_G14) - eta_q_G14*cos(phi_G14)
        struct[0].Gx_ini[126,18] = eta_d_G14*cos(phi_G14) - eta_q_G14*sin(phi_G14)
        struct[0].Gx_ini[127,18] = -v_mD_G14*sin(phi_G14) + v_mQ_G14*cos(phi_G14)
        struct[0].Gx_ini[128,18] = -v_mD_G14*cos(phi_G14) - v_mQ_G14*sin(phi_G14)
        struct[0].Gx_ini[129,18] = -v_sD_G14*sin(phi_G14) + v_sQ_G14*cos(phi_G14)
        struct[0].Gx_ini[130,18] = -v_sD_G14*cos(phi_G14) - v_sQ_G14*sin(phi_G14)
        struct[0].Gx_ini[131,18] = -i_tD_G14*sin(phi_G14) + i_tQ_G14*cos(phi_G14)
        struct[0].Gx_ini[132,18] = -i_tD_G14*cos(phi_G14) - i_tQ_G14*sin(phi_G14)
        struct[0].Gx_ini[133,18] = -i_sD_G14*sin(phi_G14) + i_sQ_G14*cos(phi_G14)
        struct[0].Gx_ini[134,18] = -i_sD_G14*cos(phi_G14) - i_sQ_G14*sin(phi_G14)
        struct[0].Gx_ini[135,20] = -K_q_G14/T_q_G14
        struct[0].Gx_ini[138,22] = -K_f_G14
        struct[0].Gx_ini[138,24] = K_f_sec/2
        struct[0].Gx_ini[139,23] = -K_vpoi_G14

        struct[0].Gy_ini[0,0] = -R_R00R01
        struct[0].Gy_ini[0,1] = L_R00R01*omega
        struct[0].Gy_ini[1,0] = -L_R00R01*omega
        struct[0].Gy_ini[1,1] = -R_R00R01
        struct[0].Gy_ini[2,2] = -R_R02R01
        struct[0].Gy_ini[2,3] = L_R02R01*omega
        struct[0].Gy_ini[3,2] = -L_R02R01*omega
        struct[0].Gy_ini[3,3] = -R_R02R01
        struct[0].Gy_ini[4,4] = -R_R02R03
        struct[0].Gy_ini[4,5] = L_R02R03*omega
        struct[0].Gy_ini[5,4] = -L_R02R03*omega
        struct[0].Gy_ini[5,5] = -R_R02R03
        struct[0].Gy_ini[6,6] = -R_R03R04
        struct[0].Gy_ini[6,7] = L_R03R04*omega
        struct[0].Gy_ini[7,6] = -L_R03R04*omega
        struct[0].Gy_ini[7,7] = -R_R03R04
        struct[0].Gy_ini[8,8] = -R_R04R05
        struct[0].Gy_ini[8,9] = L_R04R05*omega
        struct[0].Gy_ini[9,8] = -L_R04R05*omega
        struct[0].Gy_ini[9,9] = -R_R04R05
        struct[0].Gy_ini[10,10] = -R_R04R12
        struct[0].Gy_ini[10,11] = L_R04R12*omega
        struct[0].Gy_ini[11,10] = -L_R04R12*omega
        struct[0].Gy_ini[11,11] = -R_R04R12
        struct[0].Gy_ini[12,12] = -R_R05R06
        struct[0].Gy_ini[12,13] = L_R05R06*omega
        struct[0].Gy_ini[13,12] = -L_R05R06*omega
        struct[0].Gy_ini[13,13] = -R_R05R06
        struct[0].Gy_ini[14,14] = -R_R06R07
        struct[0].Gy_ini[14,15] = L_R06R07*omega
        struct[0].Gy_ini[15,14] = -L_R06R07*omega
        struct[0].Gy_ini[15,15] = -R_R06R07
        struct[0].Gy_ini[16,16] = -R_R07R08
        struct[0].Gy_ini[16,17] = L_R07R08*omega
        struct[0].Gy_ini[17,16] = -L_R07R08*omega
        struct[0].Gy_ini[17,17] = -R_R07R08
        struct[0].Gy_ini[18,18] = -R_R08R09
        struct[0].Gy_ini[18,19] = L_R08R09*omega
        struct[0].Gy_ini[19,18] = -L_R08R09*omega
        struct[0].Gy_ini[19,19] = -R_R08R09
        struct[0].Gy_ini[20,20] = -R_R09R10
        struct[0].Gy_ini[20,21] = L_R09R10*omega
        struct[0].Gy_ini[21,20] = -L_R09R10*omega
        struct[0].Gy_ini[21,21] = -R_R09R10
        struct[0].Gy_ini[22,22] = -R_R09R17
        struct[0].Gy_ini[22,23] = L_R09R17*omega
        struct[0].Gy_ini[23,22] = -L_R09R17*omega
        struct[0].Gy_ini[23,23] = -R_R09R17
        struct[0].Gy_ini[24,24] = -R_R11R03
        struct[0].Gy_ini[24,25] = L_R11R03*omega
        struct[0].Gy_ini[25,24] = -L_R11R03*omega
        struct[0].Gy_ini[25,25] = -R_R11R03
        struct[0].Gy_ini[26,26] = -R_R12R13
        struct[0].Gy_ini[26,27] = L_R12R13*omega
        struct[0].Gy_ini[27,26] = -L_R12R13*omega
        struct[0].Gy_ini[27,27] = -R_R12R13
        struct[0].Gy_ini[28,28] = -R_R13R14
        struct[0].Gy_ini[28,29] = L_R13R14*omega
        struct[0].Gy_ini[29,28] = -L_R13R14*omega
        struct[0].Gy_ini[29,29] = -R_R13R14
        struct[0].Gy_ini[30,30] = -R_R14R15
        struct[0].Gy_ini[30,31] = L_R14R15*omega
        struct[0].Gy_ini[31,30] = -L_R14R15*omega
        struct[0].Gy_ini[31,31] = -R_R14R15
        struct[0].Gy_ini[32,32] = -R_R16R06
        struct[0].Gy_ini[32,33] = L_R16R06*omega
        struct[0].Gy_ini[33,32] = -L_R16R06*omega
        struct[0].Gy_ini[33,33] = -R_R16R06
        struct[0].Gy_ini[34,34] = -R_R18R10
        struct[0].Gy_ini[34,35] = L_R18R10*omega
        struct[0].Gy_ini[35,34] = -L_R18R10*omega
        struct[0].Gy_ini[35,35] = -R_R18R10
        struct[0].Gy_ini[36,37] = C_R00R01*omega/2
        struct[0].Gy_ini[37,36] = -C_R00R01*omega/2
        struct[0].Gy_ini[38,39] = omega*(C_R00R01/2 + C_R02R01/2)
        struct[0].Gy_ini[39,38] = -omega*(C_R00R01/2 + C_R02R01/2)
        struct[0].Gy_ini[40,41] = omega*(C_R02R01/2 + C_R02R03/2)
        struct[0].Gy_ini[41,40] = -omega*(C_R02R01/2 + C_R02R03/2)
        struct[0].Gy_ini[42,43] = omega*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].Gy_ini[43,42] = -omega*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].Gy_ini[44,45] = omega*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].Gy_ini[45,44] = -omega*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].Gy_ini[46,47] = omega*(C_R04R05/2 + C_R05R06/2)
        struct[0].Gy_ini[47,46] = -omega*(C_R04R05/2 + C_R05R06/2)
        struct[0].Gy_ini[48,49] = omega*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].Gy_ini[49,48] = -omega*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].Gy_ini[50,51] = omega*(C_R06R07/2 + C_R07R08/2)
        struct[0].Gy_ini[51,50] = -omega*(C_R06R07/2 + C_R07R08/2)
        struct[0].Gy_ini[52,53] = omega*(C_R07R08/2 + C_R08R09/2)
        struct[0].Gy_ini[53,52] = -omega*(C_R07R08/2 + C_R08R09/2)
        struct[0].Gy_ini[54,55] = omega*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].Gy_ini[55,54] = -omega*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].Gy_ini[56,57] = omega*(C_R09R10/2 + C_R18R10/2)
        struct[0].Gy_ini[57,56] = -omega*(C_R09R10/2 + C_R18R10/2)
        struct[0].Gy_ini[58,59] = C_R11R03*omega/2
        struct[0].Gy_ini[59,58] = -C_R11R03*omega/2
        struct[0].Gy_ini[60,61] = omega*(C_R04R12/2 + C_R12R13/2)
        struct[0].Gy_ini[61,60] = -omega*(C_R04R12/2 + C_R12R13/2)
        struct[0].Gy_ini[62,63] = omega*(C_R12R13/2 + C_R13R14/2)
        struct[0].Gy_ini[63,62] = -omega*(C_R12R13/2 + C_R13R14/2)
        struct[0].Gy_ini[64,65] = omega*(C_R13R14/2 + C_R14R15/2)
        struct[0].Gy_ini[65,64] = -omega*(C_R13R14/2 + C_R14R15/2)
        struct[0].Gy_ini[66,67] = C_R14R15*omega/2
        struct[0].Gy_ini[67,66] = -C_R14R15*omega/2
        struct[0].Gy_ini[68,69] = C_R16R06*omega/2
        struct[0].Gy_ini[69,68] = -C_R16R06*omega/2
        struct[0].Gy_ini[70,71] = C_R09R17*omega/2
        struct[0].Gy_ini[71,70] = -C_R09R17*omega/2
        struct[0].Gy_ini[72,73] = C_R18R10*omega/2
        struct[0].Gy_ini[73,72] = -C_R18R10*omega/2
        struct[0].Gy_ini[74,38] = -0.666666666666667*p_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_D - 0.666666666666667*q_R01_ref*v_R01_Q)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_D/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy_ini[74,39] = -0.666666666666667*q_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_D - 0.666666666666667*q_R01_ref*v_R01_Q)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_Q/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy_ini[75,38] = 0.666666666666667*q_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_Q + 0.666666666666667*q_R01_ref*v_R01_D)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_D/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy_ini[75,39] = -0.666666666666667*p_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_Q + 0.666666666666667*q_R01_ref*v_R01_D)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_Q/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy_ini[76,58] = -0.666666666666667*p_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_D - 0.666666666666667*q_R11_ref*v_R11_Q)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_D/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy_ini[76,59] = -0.666666666666667*q_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_D - 0.666666666666667*q_R11_ref*v_R11_Q)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_Q/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy_ini[77,58] = 0.666666666666667*q_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_Q + 0.666666666666667*q_R11_ref*v_R11_D)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_D/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy_ini[77,59] = -0.666666666666667*p_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_Q + 0.666666666666667*q_R11_ref*v_R11_D)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_Q/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy_ini[78,66] = -0.666666666666667*p_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_D - 0.666666666666667*q_R15_ref*v_R15_Q)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_D/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy_ini[78,67] = -0.666666666666667*q_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_D - 0.666666666666667*q_R15_ref*v_R15_Q)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_Q/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy_ini[79,66] = 0.666666666666667*q_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_Q + 0.666666666666667*q_R15_ref*v_R15_D)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_D/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy_ini[79,67] = -0.666666666666667*p_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_Q + 0.666666666666667*q_R15_ref*v_R15_D)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_Q/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy_ini[80,68] = -0.666666666666667*p_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_D - 0.666666666666667*q_R16_ref*v_R16_Q)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_D/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy_ini[80,69] = -0.666666666666667*q_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_D - 0.666666666666667*q_R16_ref*v_R16_Q)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_Q/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy_ini[81,68] = 0.666666666666667*q_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_Q + 0.666666666666667*q_R16_ref*v_R16_D)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_D/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy_ini[81,69] = -0.666666666666667*p_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_Q + 0.666666666666667*q_R16_ref*v_R16_D)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_Q/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy_ini[82,70] = -0.666666666666667*p_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_D - 0.666666666666667*q_R17_ref*v_R17_Q)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_D/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy_ini[82,71] = -0.666666666666667*q_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_D - 0.666666666666667*q_R17_ref*v_R17_Q)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_Q/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy_ini[83,70] = 0.666666666666667*q_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_Q + 0.666666666666667*q_R17_ref*v_R17_D)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_D/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy_ini[83,71] = -0.666666666666667*p_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_Q + 0.666666666666667*q_R17_ref*v_R17_D)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_Q/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy_ini[84,72] = -0.666666666666667*p_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_D - 0.666666666666667*q_R18_ref*v_R18_Q)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_D/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy_ini[84,73] = -0.666666666666667*q_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_D - 0.666666666666667*q_R18_ref*v_R18_Q)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_Q/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy_ini[85,72] = 0.666666666666667*q_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_Q + 0.666666666666667*q_R18_ref*v_R18_D)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_D/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy_ini[85,73] = -0.666666666666667*p_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_Q + 0.666666666666667*q_R18_ref*v_R18_D)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_Q/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy_ini[86,86] = -R_t_G10/L_t_G10
        struct[0].Gy_ini[86,87] = omega_G10
        struct[0].Gy_ini[86,88] = -1/L_t_G10
        struct[0].Gy_ini[86,98] = v_dc_G10/(2*L_t_G10)
        struct[0].Gy_ini[87,86] = -omega_G10
        struct[0].Gy_ini[87,87] = -R_t_G10/L_t_G10
        struct[0].Gy_ini[87,89] = -1/L_t_G10
        struct[0].Gy_ini[87,99] = v_dc_G10/(2*L_t_G10)
        struct[0].Gy_ini[88,86] = 1/C_m_G10
        struct[0].Gy_ini[88,88] = -G_d_G10/C_m_G10
        struct[0].Gy_ini[88,89] = omega_G10
        struct[0].Gy_ini[88,90] = -1/C_m_G10
        struct[0].Gy_ini[89,87] = 1/C_m_G10
        struct[0].Gy_ini[89,88] = -omega_G10
        struct[0].Gy_ini[89,89] = -G_d_G10/C_m_G10
        struct[0].Gy_ini[89,91] = -1/C_m_G10
        struct[0].Gy_ini[90,88] = 1/L_s_G10
        struct[0].Gy_ini[90,90] = -R_s_G10/L_s_G10
        struct[0].Gy_ini[90,91] = omega_G10
        struct[0].Gy_ini[90,94] = -1/L_s_G10
        struct[0].Gy_ini[91,89] = 1/L_s_G10
        struct[0].Gy_ini[91,90] = -omega_G10
        struct[0].Gy_ini[91,91] = -R_s_G10/L_s_G10
        struct[0].Gy_ini[91,95] = -1/L_s_G10
        struct[0].Gy_ini[96,106] = 1.6*R_v_G10/v_dc_G10
        struct[0].Gy_ini[96,107] = -1.6*X_v_G10/v_dc_G10
        struct[0].Gy_ini[97,106] = 1.6*X_v_G10/v_dc_G10
        struct[0].Gy_ini[97,107] = 1.6*R_v_G10/v_dc_G10
        struct[0].Gy_ini[97,108] = -653.197264742181/v_dc_G10
        struct[0].Gy_ini[98,96] = cos(phi_G10)
        struct[0].Gy_ini[98,97] = -sin(phi_G10)
        struct[0].Gy_ini[99,96] = sin(phi_G10)
        struct[0].Gy_ini[99,97] = cos(phi_G10)
        struct[0].Gy_ini[100,88] = cos(phi_G10)
        struct[0].Gy_ini[100,89] = sin(phi_G10)
        struct[0].Gy_ini[101,88] = -sin(phi_G10)
        struct[0].Gy_ini[101,89] = cos(phi_G10)
        struct[0].Gy_ini[102,94] = cos(phi_G10)
        struct[0].Gy_ini[102,95] = sin(phi_G10)
        struct[0].Gy_ini[103,94] = -sin(phi_G10)
        struct[0].Gy_ini[103,95] = cos(phi_G10)
        struct[0].Gy_ini[104,86] = cos(phi_G10)
        struct[0].Gy_ini[104,87] = sin(phi_G10)
        struct[0].Gy_ini[105,86] = -sin(phi_G10)
        struct[0].Gy_ini[105,87] = cos(phi_G10)
        struct[0].Gy_ini[106,90] = cos(phi_G10)
        struct[0].Gy_ini[106,91] = sin(phi_G10)
        struct[0].Gy_ini[107,90] = -sin(phi_G10)
        struct[0].Gy_ini[107,91] = cos(phi_G10)
        struct[0].Gy_ini[108,110] = K_q_G10
        struct[0].Gy_ini[108,112] = -K_q_G10
        struct[0].Gy_ini[109,102] = 7.5e-6*i_sd_G10
        struct[0].Gy_ini[109,103] = 7.5e-6*i_sq_G10
        struct[0].Gy_ini[109,106] = 7.5e-6*v_sd_G10
        struct[0].Gy_ini[109,107] = 7.5e-6*v_sq_G10
        struct[0].Gy_ini[110,102] = -7.5e-6*i_sq_G10
        struct[0].Gy_ini[110,103] = 7.5e-6*i_sd_G10
        struct[0].Gy_ini[110,106] = 7.5e-6*v_sq_G10
        struct[0].Gy_ini[110,107] = -7.5e-6*v_sd_G10
        struct[0].Gy_ini[113,113] = -R_t_G14/L_t_G14
        struct[0].Gy_ini[113,114] = omega_G14
        struct[0].Gy_ini[113,115] = -1/L_t_G14
        struct[0].Gy_ini[113,125] = v_dc_G14/(2*L_t_G14)
        struct[0].Gy_ini[114,113] = -omega_G14
        struct[0].Gy_ini[114,114] = -R_t_G14/L_t_G14
        struct[0].Gy_ini[114,116] = -1/L_t_G14
        struct[0].Gy_ini[114,126] = v_dc_G14/(2*L_t_G14)
        struct[0].Gy_ini[115,113] = 1/C_m_G14
        struct[0].Gy_ini[115,115] = -G_d_G14/C_m_G14
        struct[0].Gy_ini[115,116] = omega_G14
        struct[0].Gy_ini[115,117] = -1/C_m_G14
        struct[0].Gy_ini[116,114] = 1/C_m_G14
        struct[0].Gy_ini[116,115] = -omega_G14
        struct[0].Gy_ini[116,116] = -G_d_G14/C_m_G14
        struct[0].Gy_ini[116,118] = -1/C_m_G14
        struct[0].Gy_ini[117,115] = 1/L_s_G14
        struct[0].Gy_ini[117,117] = -R_s_G14/L_s_G14
        struct[0].Gy_ini[117,118] = omega_G14
        struct[0].Gy_ini[117,121] = -1/L_s_G14
        struct[0].Gy_ini[118,116] = 1/L_s_G14
        struct[0].Gy_ini[118,117] = -omega_G14
        struct[0].Gy_ini[118,118] = -R_s_G14/L_s_G14
        struct[0].Gy_ini[118,122] = -1/L_s_G14
        struct[0].Gy_ini[123,133] = 1.6*R_v_G14/v_dc_G14
        struct[0].Gy_ini[123,134] = -1.6*X_v_G14/v_dc_G14
        struct[0].Gy_ini[124,133] = 1.6*X_v_G14/v_dc_G14
        struct[0].Gy_ini[124,134] = 1.6*R_v_G14/v_dc_G14
        struct[0].Gy_ini[124,135] = -653.197264742181/v_dc_G14
        struct[0].Gy_ini[125,123] = cos(phi_G14)
        struct[0].Gy_ini[125,124] = -sin(phi_G14)
        struct[0].Gy_ini[126,123] = sin(phi_G14)
        struct[0].Gy_ini[126,124] = cos(phi_G14)
        struct[0].Gy_ini[127,115] = cos(phi_G14)
        struct[0].Gy_ini[127,116] = sin(phi_G14)
        struct[0].Gy_ini[128,115] = -sin(phi_G14)
        struct[0].Gy_ini[128,116] = cos(phi_G14)
        struct[0].Gy_ini[129,121] = cos(phi_G14)
        struct[0].Gy_ini[129,122] = sin(phi_G14)
        struct[0].Gy_ini[130,121] = -sin(phi_G14)
        struct[0].Gy_ini[130,122] = cos(phi_G14)
        struct[0].Gy_ini[131,113] = cos(phi_G14)
        struct[0].Gy_ini[131,114] = sin(phi_G14)
        struct[0].Gy_ini[132,113] = -sin(phi_G14)
        struct[0].Gy_ini[132,114] = cos(phi_G14)
        struct[0].Gy_ini[133,117] = cos(phi_G14)
        struct[0].Gy_ini[133,118] = sin(phi_G14)
        struct[0].Gy_ini[134,117] = -sin(phi_G14)
        struct[0].Gy_ini[134,118] = cos(phi_G14)
        struct[0].Gy_ini[135,137] = K_q_G14
        struct[0].Gy_ini[135,139] = -K_q_G14
        struct[0].Gy_ini[136,129] = 7.5e-6*i_sd_G14
        struct[0].Gy_ini[136,130] = 7.5e-6*i_sq_G14
        struct[0].Gy_ini[136,133] = 7.5e-6*v_sd_G14
        struct[0].Gy_ini[136,134] = 7.5e-6*v_sq_G14
        struct[0].Gy_ini[137,129] = -7.5e-6*i_sq_G14
        struct[0].Gy_ini[137,130] = 7.5e-6*i_sd_G14
        struct[0].Gy_ini[137,133] = 7.5e-6*v_sq_G14
        struct[0].Gy_ini[137,134] = -7.5e-6*v_sd_G14



@numba.njit(cache=True)
def run(t,struct,mode):

    # Parameters:
    R_R00R01 = struct[0].R_R00R01
    L_R00R01 = struct[0].L_R00R01
    C_R00R01 = struct[0].C_R00R01
    R_R02R01 = struct[0].R_R02R01
    L_R02R01 = struct[0].L_R02R01
    C_R02R01 = struct[0].C_R02R01
    R_R02R03 = struct[0].R_R02R03
    L_R02R03 = struct[0].L_R02R03
    C_R02R03 = struct[0].C_R02R03
    R_R03R04 = struct[0].R_R03R04
    L_R03R04 = struct[0].L_R03R04
    C_R03R04 = struct[0].C_R03R04
    R_R04R05 = struct[0].R_R04R05
    L_R04R05 = struct[0].L_R04R05
    C_R04R05 = struct[0].C_R04R05
    R_R04R12 = struct[0].R_R04R12
    L_R04R12 = struct[0].L_R04R12
    C_R04R12 = struct[0].C_R04R12
    R_R05R06 = struct[0].R_R05R06
    L_R05R06 = struct[0].L_R05R06
    C_R05R06 = struct[0].C_R05R06
    R_R06R07 = struct[0].R_R06R07
    L_R06R07 = struct[0].L_R06R07
    C_R06R07 = struct[0].C_R06R07
    R_R07R08 = struct[0].R_R07R08
    L_R07R08 = struct[0].L_R07R08
    C_R07R08 = struct[0].C_R07R08
    R_R08R09 = struct[0].R_R08R09
    L_R08R09 = struct[0].L_R08R09
    C_R08R09 = struct[0].C_R08R09
    R_R09R10 = struct[0].R_R09R10
    L_R09R10 = struct[0].L_R09R10
    C_R09R10 = struct[0].C_R09R10
    R_R09R17 = struct[0].R_R09R17
    L_R09R17 = struct[0].L_R09R17
    C_R09R17 = struct[0].C_R09R17
    R_R11R03 = struct[0].R_R11R03
    L_R11R03 = struct[0].L_R11R03
    C_R11R03 = struct[0].C_R11R03
    R_R12R13 = struct[0].R_R12R13
    L_R12R13 = struct[0].L_R12R13
    C_R12R13 = struct[0].C_R12R13
    R_R13R14 = struct[0].R_R13R14
    L_R13R14 = struct[0].L_R13R14
    C_R13R14 = struct[0].C_R13R14
    R_R14R15 = struct[0].R_R14R15
    L_R14R15 = struct[0].L_R14R15
    C_R14R15 = struct[0].C_R14R15
    R_R16R06 = struct[0].R_R16R06
    L_R16R06 = struct[0].L_R16R06
    C_R16R06 = struct[0].C_R16R06
    R_R18R10 = struct[0].R_R18R10
    L_R18R10 = struct[0].L_R18R10
    C_R18R10 = struct[0].C_R18R10
    i_R00_D = struct[0].i_R00_D
    i_R00_Q = struct[0].i_R00_Q
    i_R02_D = struct[0].i_R02_D
    i_R02_Q = struct[0].i_R02_Q
    i_R03_D = struct[0].i_R03_D
    i_R03_Q = struct[0].i_R03_Q
    i_R04_D = struct[0].i_R04_D
    i_R04_Q = struct[0].i_R04_Q
    i_R05_D = struct[0].i_R05_D
    i_R05_Q = struct[0].i_R05_Q
    i_R06_D = struct[0].i_R06_D
    i_R06_Q = struct[0].i_R06_Q
    i_R07_D = struct[0].i_R07_D
    i_R07_Q = struct[0].i_R07_Q
    i_R08_D = struct[0].i_R08_D
    i_R08_Q = struct[0].i_R08_Q
    i_R09_D = struct[0].i_R09_D
    i_R09_Q = struct[0].i_R09_Q
    i_R12_D = struct[0].i_R12_D
    i_R12_Q = struct[0].i_R12_Q
    i_R13_D = struct[0].i_R13_D
    i_R13_Q = struct[0].i_R13_Q
    omega = struct[0].omega
    L_t_G10 = struct[0].L_t_G10
    R_t_G10 = struct[0].R_t_G10
    C_m_G10 = struct[0].C_m_G10
    L_s_G10 = struct[0].L_s_G10
    R_s_G10 = struct[0].R_s_G10
    omega_G10 = struct[0].omega_G10
    G_d_G10 = struct[0].G_d_G10
    K_p_G10 = struct[0].K_p_G10
    T_p_G10 = struct[0].T_p_G10
    K_q_G10 = struct[0].K_q_G10
    T_q_G10 = struct[0].T_q_G10
    R_v_G10 = struct[0].R_v_G10
    X_v_G10 = struct[0].X_v_G10
    S_b_kVA_G10 = struct[0].S_b_kVA_G10
    U_b_G10 = struct[0].U_b_G10
    K_phi_G10 = struct[0].K_phi_G10
    H_G10 = struct[0].H_G10
    D_G10 = struct[0].D_G10
    T_vpoi_G10 = struct[0].T_vpoi_G10
    K_vpoi_G10 = struct[0].K_vpoi_G10
    T_f_G10 = struct[0].T_f_G10
    K_f_G10 = struct[0].K_f_G10
    L_t_G14 = struct[0].L_t_G14
    R_t_G14 = struct[0].R_t_G14
    C_m_G14 = struct[0].C_m_G14
    L_s_G14 = struct[0].L_s_G14
    R_s_G14 = struct[0].R_s_G14
    omega_G14 = struct[0].omega_G14
    G_d_G14 = struct[0].G_d_G14
    K_p_G14 = struct[0].K_p_G14
    T_p_G14 = struct[0].T_p_G14
    K_q_G14 = struct[0].K_q_G14
    T_q_G14 = struct[0].T_q_G14
    R_v_G14 = struct[0].R_v_G14
    X_v_G14 = struct[0].X_v_G14
    S_b_kVA_G14 = struct[0].S_b_kVA_G14
    U_b_G14 = struct[0].U_b_G14
    K_phi_G14 = struct[0].K_phi_G14
    H_G14 = struct[0].H_G14
    D_G14 = struct[0].D_G14
    T_vpoi_G14 = struct[0].T_vpoi_G14
    K_vpoi_G14 = struct[0].K_vpoi_G14
    T_f_G14 = struct[0].T_f_G14
    K_f_G14 = struct[0].K_f_G14
    K_f_sec = struct[0].K_f_sec
    
    # Inputs:
    T_i_R01 = struct[0].T_i_R01
    I_max_R01 = struct[0].I_max_R01
    p_R01_ref = struct[0].p_R01_ref
    q_R01_ref = struct[0].q_R01_ref
    T_i_R11 = struct[0].T_i_R11
    I_max_R11 = struct[0].I_max_R11
    p_R11_ref = struct[0].p_R11_ref
    q_R11_ref = struct[0].q_R11_ref
    T_i_R15 = struct[0].T_i_R15
    I_max_R15 = struct[0].I_max_R15
    p_R15_ref = struct[0].p_R15_ref
    q_R15_ref = struct[0].q_R15_ref
    T_i_R16 = struct[0].T_i_R16
    I_max_R16 = struct[0].I_max_R16
    p_R16_ref = struct[0].p_R16_ref
    q_R16_ref = struct[0].q_R16_ref
    T_i_R17 = struct[0].T_i_R17
    I_max_R17 = struct[0].I_max_R17
    p_R17_ref = struct[0].p_R17_ref
    q_R17_ref = struct[0].q_R17_ref
    T_i_R18 = struct[0].T_i_R18
    I_max_R18 = struct[0].I_max_R18
    p_R18_ref = struct[0].p_R18_ref
    q_R18_ref = struct[0].q_R18_ref
    v_dc_G10 = struct[0].v_dc_G10
    p_m_ref_G10 = struct[0].p_m_ref_G10
    q_s_ref_G10 = struct[0].q_s_ref_G10
    v_s_ref_G10 = struct[0].v_s_ref_G10
    omega_ref_G10 = struct[0].omega_ref_G10
    p_r_G10 = struct[0].p_r_G10
    q_r_G10 = struct[0].q_r_G10
    v_dc_G14 = struct[0].v_dc_G14
    p_m_ref_G14 = struct[0].p_m_ref_G14
    q_s_ref_G14 = struct[0].q_s_ref_G14
    v_s_ref_G14 = struct[0].v_s_ref_G14
    omega_ref_G14 = struct[0].omega_ref_G14
    p_r_G14 = struct[0].p_r_G14
    q_r_G14 = struct[0].q_r_G14
    
    # Dynamical states:
    i_R01_D = struct[0].x[0,0]
    i_R01_Q = struct[0].x[1,0]
    i_R11_D = struct[0].x[2,0]
    i_R11_Q = struct[0].x[3,0]
    i_R15_D = struct[0].x[4,0]
    i_R15_Q = struct[0].x[5,0]
    i_R16_D = struct[0].x[6,0]
    i_R16_Q = struct[0].x[7,0]
    i_R17_D = struct[0].x[8,0]
    i_R17_Q = struct[0].x[9,0]
    i_R18_D = struct[0].x[10,0]
    i_R18_Q = struct[0].x[11,0]
    phi_G10 = struct[0].x[12,0]
    omega_v_G10 = struct[0].x[13,0]
    xi_q_G10 = struct[0].x[14,0]
    omega_rads_G10 = struct[0].x[15,0]
    omega_v_filt_G10 = struct[0].x[16,0]
    v_s_filt_G10 = struct[0].x[17,0]
    phi_G14 = struct[0].x[18,0]
    omega_v_G14 = struct[0].x[19,0]
    xi_q_G14 = struct[0].x[20,0]
    omega_rads_G14 = struct[0].x[21,0]
    omega_v_filt_G14 = struct[0].x[22,0]
    v_s_filt_G14 = struct[0].x[23,0]
    xi_f_sec = struct[0].x[24,0]
    
    # Algebraic states:
    i_l_R00R01_D = struct[0].y_run[0,0]
    i_l_R00R01_Q = struct[0].y_run[1,0]
    i_l_R02R01_D = struct[0].y_run[2,0]
    i_l_R02R01_Q = struct[0].y_run[3,0]
    i_l_R02R03_D = struct[0].y_run[4,0]
    i_l_R02R03_Q = struct[0].y_run[5,0]
    i_l_R03R04_D = struct[0].y_run[6,0]
    i_l_R03R04_Q = struct[0].y_run[7,0]
    i_l_R04R05_D = struct[0].y_run[8,0]
    i_l_R04R05_Q = struct[0].y_run[9,0]
    i_l_R04R12_D = struct[0].y_run[10,0]
    i_l_R04R12_Q = struct[0].y_run[11,0]
    i_l_R05R06_D = struct[0].y_run[12,0]
    i_l_R05R06_Q = struct[0].y_run[13,0]
    i_l_R06R07_D = struct[0].y_run[14,0]
    i_l_R06R07_Q = struct[0].y_run[15,0]
    i_l_R07R08_D = struct[0].y_run[16,0]
    i_l_R07R08_Q = struct[0].y_run[17,0]
    i_l_R08R09_D = struct[0].y_run[18,0]
    i_l_R08R09_Q = struct[0].y_run[19,0]
    i_l_R09R10_D = struct[0].y_run[20,0]
    i_l_R09R10_Q = struct[0].y_run[21,0]
    i_l_R09R17_D = struct[0].y_run[22,0]
    i_l_R09R17_Q = struct[0].y_run[23,0]
    i_l_R11R03_D = struct[0].y_run[24,0]
    i_l_R11R03_Q = struct[0].y_run[25,0]
    i_l_R12R13_D = struct[0].y_run[26,0]
    i_l_R12R13_Q = struct[0].y_run[27,0]
    i_l_R13R14_D = struct[0].y_run[28,0]
    i_l_R13R14_Q = struct[0].y_run[29,0]
    i_l_R14R15_D = struct[0].y_run[30,0]
    i_l_R14R15_Q = struct[0].y_run[31,0]
    i_l_R16R06_D = struct[0].y_run[32,0]
    i_l_R16R06_Q = struct[0].y_run[33,0]
    i_l_R18R10_D = struct[0].y_run[34,0]
    i_l_R18R10_Q = struct[0].y_run[35,0]
    v_R00_D = struct[0].y_run[36,0]
    v_R00_Q = struct[0].y_run[37,0]
    v_R01_D = struct[0].y_run[38,0]
    v_R01_Q = struct[0].y_run[39,0]
    v_R02_D = struct[0].y_run[40,0]
    v_R02_Q = struct[0].y_run[41,0]
    v_R03_D = struct[0].y_run[42,0]
    v_R03_Q = struct[0].y_run[43,0]
    v_R04_D = struct[0].y_run[44,0]
    v_R04_Q = struct[0].y_run[45,0]
    v_R05_D = struct[0].y_run[46,0]
    v_R05_Q = struct[0].y_run[47,0]
    v_R06_D = struct[0].y_run[48,0]
    v_R06_Q = struct[0].y_run[49,0]
    v_R07_D = struct[0].y_run[50,0]
    v_R07_Q = struct[0].y_run[51,0]
    v_R08_D = struct[0].y_run[52,0]
    v_R08_Q = struct[0].y_run[53,0]
    v_R09_D = struct[0].y_run[54,0]
    v_R09_Q = struct[0].y_run[55,0]
    v_R10_D = struct[0].y_run[56,0]
    v_R10_Q = struct[0].y_run[57,0]
    v_R11_D = struct[0].y_run[58,0]
    v_R11_Q = struct[0].y_run[59,0]
    v_R12_D = struct[0].y_run[60,0]
    v_R12_Q = struct[0].y_run[61,0]
    v_R13_D = struct[0].y_run[62,0]
    v_R13_Q = struct[0].y_run[63,0]
    v_R14_D = struct[0].y_run[64,0]
    v_R14_Q = struct[0].y_run[65,0]
    v_R15_D = struct[0].y_run[66,0]
    v_R15_Q = struct[0].y_run[67,0]
    v_R16_D = struct[0].y_run[68,0]
    v_R16_Q = struct[0].y_run[69,0]
    v_R17_D = struct[0].y_run[70,0]
    v_R17_Q = struct[0].y_run[71,0]
    v_R18_D = struct[0].y_run[72,0]
    v_R18_Q = struct[0].y_run[73,0]
    i_R01_d_ref = struct[0].y_run[74,0]
    i_R01_q_ref = struct[0].y_run[75,0]
    i_R11_d_ref = struct[0].y_run[76,0]
    i_R11_q_ref = struct[0].y_run[77,0]
    i_R15_d_ref = struct[0].y_run[78,0]
    i_R15_q_ref = struct[0].y_run[79,0]
    i_R16_d_ref = struct[0].y_run[80,0]
    i_R16_q_ref = struct[0].y_run[81,0]
    i_R17_d_ref = struct[0].y_run[82,0]
    i_R17_q_ref = struct[0].y_run[83,0]
    i_R18_d_ref = struct[0].y_run[84,0]
    i_R18_q_ref = struct[0].y_run[85,0]
    i_tD_G10 = struct[0].y_run[86,0]
    i_tQ_G10 = struct[0].y_run[87,0]
    v_mD_G10 = struct[0].y_run[88,0]
    v_mQ_G10 = struct[0].y_run[89,0]
    i_sD_G10 = struct[0].y_run[90,0]
    i_sQ_G10 = struct[0].y_run[91,0]
    i_R10_D = struct[0].y_run[92,0]
    i_R10_Q = struct[0].y_run[93,0]
    v_sD_G10 = struct[0].y_run[94,0]
    v_sQ_G10 = struct[0].y_run[95,0]
    eta_d_G10 = struct[0].y_run[96,0]
    eta_q_G10 = struct[0].y_run[97,0]
    eta_D_G10 = struct[0].y_run[98,0]
    eta_Q_G10 = struct[0].y_run[99,0]
    v_md_G10 = struct[0].y_run[100,0]
    v_mq_G10 = struct[0].y_run[101,0]
    v_sd_G10 = struct[0].y_run[102,0]
    v_sq_G10 = struct[0].y_run[103,0]
    i_td_G10 = struct[0].y_run[104,0]
    i_tq_G10 = struct[0].y_run[105,0]
    i_sd_G10 = struct[0].y_run[106,0]
    i_sq_G10 = struct[0].y_run[107,0]
    DV_sat_G10 = struct[0].y_run[108,0]
    p_s_pu_G10 = struct[0].y_run[109,0]
    q_s_pu_G10 = struct[0].y_run[110,0]
    p_m_ref_G10 = struct[0].y_run[111,0]
    q_s_ref_G10 = struct[0].y_run[112,0]
    i_tD_G14 = struct[0].y_run[113,0]
    i_tQ_G14 = struct[0].y_run[114,0]
    v_mD_G14 = struct[0].y_run[115,0]
    v_mQ_G14 = struct[0].y_run[116,0]
    i_sD_G14 = struct[0].y_run[117,0]
    i_sQ_G14 = struct[0].y_run[118,0]
    i_R14_D = struct[0].y_run[119,0]
    i_R14_Q = struct[0].y_run[120,0]
    v_sD_G14 = struct[0].y_run[121,0]
    v_sQ_G14 = struct[0].y_run[122,0]
    eta_d_G14 = struct[0].y_run[123,0]
    eta_q_G14 = struct[0].y_run[124,0]
    eta_D_G14 = struct[0].y_run[125,0]
    eta_Q_G14 = struct[0].y_run[126,0]
    v_md_G14 = struct[0].y_run[127,0]
    v_mq_G14 = struct[0].y_run[128,0]
    v_sd_G14 = struct[0].y_run[129,0]
    v_sq_G14 = struct[0].y_run[130,0]
    i_td_G14 = struct[0].y_run[131,0]
    i_tq_G14 = struct[0].y_run[132,0]
    i_sd_G14 = struct[0].y_run[133,0]
    i_sq_G14 = struct[0].y_run[134,0]
    DV_sat_G14 = struct[0].y_run[135,0]
    p_s_pu_G14 = struct[0].y_run[136,0]
    q_s_pu_G14 = struct[0].y_run[137,0]
    p_m_ref_G14 = struct[0].y_run[138,0]
    q_s_ref_G14 = struct[0].y_run[139,0]
    
    struct[0].u_run[0,0] = T_i_R01
    struct[0].u_run[1,0] = I_max_R01
    struct[0].u_run[2,0] = p_R01_ref
    struct[0].u_run[3,0] = q_R01_ref
    struct[0].u_run[4,0] = T_i_R11
    struct[0].u_run[5,0] = I_max_R11
    struct[0].u_run[6,0] = p_R11_ref
    struct[0].u_run[7,0] = q_R11_ref
    struct[0].u_run[8,0] = T_i_R15
    struct[0].u_run[9,0] = I_max_R15
    struct[0].u_run[10,0] = p_R15_ref
    struct[0].u_run[11,0] = q_R15_ref
    struct[0].u_run[12,0] = T_i_R16
    struct[0].u_run[13,0] = I_max_R16
    struct[0].u_run[14,0] = p_R16_ref
    struct[0].u_run[15,0] = q_R16_ref
    struct[0].u_run[16,0] = T_i_R17
    struct[0].u_run[17,0] = I_max_R17
    struct[0].u_run[18,0] = p_R17_ref
    struct[0].u_run[19,0] = q_R17_ref
    struct[0].u_run[20,0] = T_i_R18
    struct[0].u_run[21,0] = I_max_R18
    struct[0].u_run[22,0] = p_R18_ref
    struct[0].u_run[23,0] = q_R18_ref
    struct[0].u_run[24,0] = v_dc_G10
    struct[0].u_run[25,0] = p_m_ref_G10
    struct[0].u_run[26,0] = q_s_ref_G10
    struct[0].u_run[27,0] = v_s_ref_G10
    struct[0].u_run[28,0] = omega_ref_G10
    struct[0].u_run[29,0] = p_r_G10
    struct[0].u_run[30,0] = q_r_G10
    struct[0].u_run[31,0] = v_dc_G14
    struct[0].u_run[32,0] = p_m_ref_G14
    struct[0].u_run[33,0] = q_s_ref_G14
    struct[0].u_run[34,0] = v_s_ref_G14
    struct[0].u_run[35,0] = omega_ref_G14
    struct[0].u_run[36,0] = p_r_G14
    struct[0].u_run[37,0] = q_r_G14
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -100.0*i_R01_D + 100.0*Piecewise(np.array([(-I_max_R01, I_max_R01 < -i_R01_d_ref), (I_max_R01, I_max_R01 < i_R01_d_ref), (i_R01_d_ref, True)]))
        struct[0].f[1,0] = -100.0*i_R01_Q + 100.0*Piecewise(np.array([(-I_max_R01, I_max_R01 < -i_R01_q_ref), (I_max_R01, I_max_R01 < i_R01_q_ref), (i_R01_q_ref, True)]))
        struct[0].f[2,0] = -100.0*i_R11_D + 100.0*Piecewise(np.array([(-I_max_R11, I_max_R11 < -i_R11_d_ref), (I_max_R11, I_max_R11 < i_R11_d_ref), (i_R11_d_ref, True)]))
        struct[0].f[3,0] = -100.0*i_R11_Q + 100.0*Piecewise(np.array([(-I_max_R11, I_max_R11 < -i_R11_q_ref), (I_max_R11, I_max_R11 < i_R11_q_ref), (i_R11_q_ref, True)]))
        struct[0].f[4,0] = -100.0*i_R15_D + 100.0*Piecewise(np.array([(-I_max_R15, I_max_R15 < -i_R15_d_ref), (I_max_R15, I_max_R15 < i_R15_d_ref), (i_R15_d_ref, True)]))
        struct[0].f[5,0] = -100.0*i_R15_Q + 100.0*Piecewise(np.array([(-I_max_R15, I_max_R15 < -i_R15_q_ref), (I_max_R15, I_max_R15 < i_R15_q_ref), (i_R15_q_ref, True)]))
        struct[0].f[6,0] = -100.0*i_R16_D + 100.0*Piecewise(np.array([(-I_max_R16, I_max_R16 < -i_R16_d_ref), (I_max_R16, I_max_R16 < i_R16_d_ref), (i_R16_d_ref, True)]))
        struct[0].f[7,0] = -100.0*i_R16_Q + 100.0*Piecewise(np.array([(-I_max_R16, I_max_R16 < -i_R16_q_ref), (I_max_R16, I_max_R16 < i_R16_q_ref), (i_R16_q_ref, True)]))
        struct[0].f[8,0] = -100.0*i_R17_D + 100.0*Piecewise(np.array([(-I_max_R17, I_max_R17 < -i_R17_d_ref), (I_max_R17, I_max_R17 < i_R17_d_ref), (i_R17_d_ref, True)]))
        struct[0].f[9,0] = -100.0*i_R17_Q + 100.0*Piecewise(np.array([(-I_max_R17, I_max_R17 < -i_R17_q_ref), (I_max_R17, I_max_R17 < i_R17_q_ref), (i_R17_q_ref, True)]))
        struct[0].f[10,0] = -100.0*i_R18_D + 100.0*Piecewise(np.array([(-I_max_R18, I_max_R18 < -i_R18_d_ref), (I_max_R18, I_max_R18 < i_R18_d_ref), (i_R18_d_ref, True)]))
        struct[0].f[11,0] = -100.0*i_R18_Q + 100.0*Piecewise(np.array([(-I_max_R18, I_max_R18 < -i_R18_q_ref), (I_max_R18, I_max_R18 < i_R18_q_ref), (i_R18_q_ref, True)]))
        struct[0].f[12,0] = -K_phi_G10*phi_G10 + 314.159265358979*omega_v_G10 - 314.159265358979*(H_G10*omega_v_G10 + H_G14*omega_v_G14)/(H_G10 + H_G14)
        struct[0].f[13,0] = (-D_G10*(omega_v_G10 - 1.0) + p_m_ref_G10 - p_s_pu_G10)/(2*H_G10)
        struct[0].f[14,0] = -q_s_pu_G10 + q_s_ref_G10
        struct[0].f[15,0] = -1.0*omega_rads_G10 + 314.159265358979*omega_v_G10
        struct[0].f[16,0] = (omega_v_G10 - omega_v_filt_G10)/T_f_G10
        struct[0].f[17,0] = (-v_s_filt_G10 + 0.00306186217847897*(v_sd_G10**2 + v_sq_G10**2)**0.5)/T_vpoi_G10
        struct[0].f[18,0] = -K_phi_G14*phi_G14 + 314.159265358979*omega_v_G14 - 314.159265358979*(H_G10*omega_v_G10 + H_G14*omega_v_G14)/(H_G10 + H_G14)
        struct[0].f[19,0] = (-D_G14*(omega_v_G14 - 1.0) + p_m_ref_G14 - p_s_pu_G14)/(2*H_G14)
        struct[0].f[20,0] = -q_s_pu_G14 + q_s_ref_G14
        struct[0].f[21,0] = -1.0*omega_rads_G14 + 314.159265358979*omega_v_G14
        struct[0].f[22,0] = (omega_v_G14 - omega_v_filt_G14)/T_f_G14
        struct[0].f[23,0] = (-v_s_filt_G14 + 0.00306186217847897*(v_sd_G14**2 + v_sq_G14**2)**0.5)/T_vpoi_G14
        struct[0].f[24,0] = 1 - (H_G10*omega_v_G10 + H_G14*omega_v_G14)/(H_G10 + H_G14)
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy) @ np.ascontiguousarray(struct[0].y_run) + np.ascontiguousarray(struct[0].Gu) @ np.ascontiguousarray(struct[0].u_run)

        struct[0].g[36,0] = C_R00R01*omega*v_R00_Q/2 + i_R00_D - i_l_R00R01_D
        struct[0].g[37,0] = -C_R00R01*omega*v_R00_D/2 + i_R00_Q - i_l_R00R01_Q
        struct[0].g[38,0] = i_R01_D + i_l_R00R01_D + i_l_R02R01_D + omega*v_R01_Q*(C_R00R01/2 + C_R02R01/2)
        struct[0].g[39,0] = i_R01_Q + i_l_R00R01_Q + i_l_R02R01_Q - omega*v_R01_D*(C_R00R01/2 + C_R02R01/2)
        struct[0].g[40,0] = i_R02_D - i_l_R02R01_D - i_l_R02R03_D + omega*v_R02_Q*(C_R02R01/2 + C_R02R03/2)
        struct[0].g[41,0] = i_R02_Q - i_l_R02R01_Q - i_l_R02R03_Q - omega*v_R02_D*(C_R02R01/2 + C_R02R03/2)
        struct[0].g[42,0] = i_R03_D + i_l_R02R03_D - i_l_R03R04_D + i_l_R11R03_D + omega*v_R03_Q*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].g[43,0] = i_R03_Q + i_l_R02R03_Q - i_l_R03R04_Q + i_l_R11R03_Q - omega*v_R03_D*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].g[44,0] = i_R04_D + i_l_R03R04_D - i_l_R04R05_D - i_l_R04R12_D + omega*v_R04_Q*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].g[45,0] = i_R04_Q + i_l_R03R04_Q - i_l_R04R05_Q - i_l_R04R12_Q - omega*v_R04_D*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].g[46,0] = i_R05_D + i_l_R04R05_D - i_l_R05R06_D + omega*v_R05_Q*(C_R04R05/2 + C_R05R06/2)
        struct[0].g[47,0] = i_R05_Q + i_l_R04R05_Q - i_l_R05R06_Q - omega*v_R05_D*(C_R04R05/2 + C_R05R06/2)
        struct[0].g[48,0] = i_R06_D + i_l_R05R06_D - i_l_R06R07_D + i_l_R16R06_D + omega*v_R06_Q*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].g[49,0] = i_R06_Q + i_l_R05R06_Q - i_l_R06R07_Q + i_l_R16R06_Q - omega*v_R06_D*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].g[50,0] = i_R07_D + i_l_R06R07_D - i_l_R07R08_D + omega*v_R07_Q*(C_R06R07/2 + C_R07R08/2)
        struct[0].g[51,0] = i_R07_Q + i_l_R06R07_Q - i_l_R07R08_Q - omega*v_R07_D*(C_R06R07/2 + C_R07R08/2)
        struct[0].g[52,0] = i_R08_D + i_l_R07R08_D - i_l_R08R09_D + omega*v_R08_Q*(C_R07R08/2 + C_R08R09/2)
        struct[0].g[53,0] = i_R08_Q + i_l_R07R08_Q - i_l_R08R09_Q - omega*v_R08_D*(C_R07R08/2 + C_R08R09/2)
        struct[0].g[54,0] = i_R09_D + i_l_R08R09_D - i_l_R09R10_D - i_l_R09R17_D + omega*v_R09_Q*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].g[55,0] = i_R09_Q + i_l_R08R09_Q - i_l_R09R10_Q - i_l_R09R17_Q - omega*v_R09_D*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].g[58,0] = C_R11R03*omega*v_R11_Q/2 + i_R11_D - i_l_R11R03_D
        struct[0].g[59,0] = -C_R11R03*omega*v_R11_D/2 + i_R11_Q - i_l_R11R03_Q
        struct[0].g[60,0] = i_R12_D + i_l_R04R12_D - i_l_R12R13_D + omega*v_R12_Q*(C_R04R12/2 + C_R12R13/2)
        struct[0].g[61,0] = i_R12_Q + i_l_R04R12_Q - i_l_R12R13_Q - omega*v_R12_D*(C_R04R12/2 + C_R12R13/2)
        struct[0].g[62,0] = i_R13_D + i_l_R12R13_D - i_l_R13R14_D + omega*v_R13_Q*(C_R12R13/2 + C_R13R14/2)
        struct[0].g[63,0] = i_R13_Q + i_l_R12R13_Q - i_l_R13R14_Q - omega*v_R13_D*(C_R12R13/2 + C_R13R14/2)
        struct[0].g[66,0] = C_R14R15*omega*v_R15_Q/2 + i_R15_D + i_l_R14R15_D
        struct[0].g[67,0] = -C_R14R15*omega*v_R15_D/2 + i_R15_Q + i_l_R14R15_Q
        struct[0].g[68,0] = C_R16R06*omega*v_R16_Q/2 + i_R16_D - i_l_R16R06_D
        struct[0].g[69,0] = -C_R16R06*omega*v_R16_D/2 + i_R16_Q - i_l_R16R06_Q
        struct[0].g[70,0] = C_R09R17*omega*v_R17_Q/2 + i_R17_D + i_l_R09R17_D
        struct[0].g[71,0] = -C_R09R17*omega*v_R17_D/2 + i_R17_Q + i_l_R09R17_Q
        struct[0].g[72,0] = C_R18R10*omega*v_R18_Q/2 + i_R18_D - i_l_R18R10_D
        struct[0].g[73,0] = -C_R18R10*omega*v_R18_D/2 + i_R18_Q - i_l_R18R10_Q
        struct[0].g[74,0] = -i_R01_d_ref + (-0.666666666666667*p_R01_ref*v_R01_D - 0.666666666666667*q_R01_ref*v_R01_Q)*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)]))
        struct[0].g[75,0] = -i_R01_q_ref - (0.666666666666667*p_R01_ref*v_R01_Q - 0.666666666666667*q_R01_ref*v_R01_D)*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)]))
        struct[0].g[76,0] = -i_R11_d_ref + (-0.666666666666667*p_R11_ref*v_R11_D - 0.666666666666667*q_R11_ref*v_R11_Q)*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)]))
        struct[0].g[77,0] = -i_R11_q_ref - (0.666666666666667*p_R11_ref*v_R11_Q - 0.666666666666667*q_R11_ref*v_R11_D)*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)]))
        struct[0].g[78,0] = -i_R15_d_ref + (-0.666666666666667*p_R15_ref*v_R15_D - 0.666666666666667*q_R15_ref*v_R15_Q)*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)]))
        struct[0].g[79,0] = -i_R15_q_ref - (0.666666666666667*p_R15_ref*v_R15_Q - 0.666666666666667*q_R15_ref*v_R15_D)*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)]))
        struct[0].g[80,0] = -i_R16_d_ref + (-0.666666666666667*p_R16_ref*v_R16_D - 0.666666666666667*q_R16_ref*v_R16_Q)*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)]))
        struct[0].g[81,0] = -i_R16_q_ref - (0.666666666666667*p_R16_ref*v_R16_Q - 0.666666666666667*q_R16_ref*v_R16_D)*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)]))
        struct[0].g[82,0] = -i_R17_d_ref + (-0.666666666666667*p_R17_ref*v_R17_D - 0.666666666666667*q_R17_ref*v_R17_Q)*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)]))
        struct[0].g[83,0] = -i_R17_q_ref - (0.666666666666667*p_R17_ref*v_R17_Q - 0.666666666666667*q_R17_ref*v_R17_D)*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)]))
        struct[0].g[84,0] = -i_R18_d_ref + (-0.666666666666667*p_R18_ref*v_R18_D - 0.666666666666667*q_R18_ref*v_R18_Q)*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)]))
        struct[0].g[85,0] = -i_R18_q_ref - (0.666666666666667*p_R18_ref*v_R18_Q - 0.666666666666667*q_R18_ref*v_R18_D)*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)]))
        struct[0].g[86,0] = (L_t_G10*i_tQ_G10*omega_G10 - R_t_G10*i_tD_G10 + eta_D_G10*v_dc_G10/2 - v_mD_G10)/L_t_G10
        struct[0].g[87,0] = (-L_t_G10*i_tD_G10*omega_G10 - R_t_G10*i_tQ_G10 + eta_Q_G10*v_dc_G10/2 - v_mQ_G10)/L_t_G10
        struct[0].g[88,0] = (C_m_G10*omega_G10*v_mQ_G10 - G_d_G10*v_mD_G10 - i_sD_G10 + i_tD_G10)/C_m_G10
        struct[0].g[89,0] = (-C_m_G10*omega_G10*v_mD_G10 - G_d_G10*v_mQ_G10 - i_sQ_G10 + i_tQ_G10)/C_m_G10
        struct[0].g[90,0] = (L_s_G10*i_sQ_G10*omega_G10 - R_s_G10*i_sD_G10 + v_mD_G10 - v_sD_G10)/L_s_G10
        struct[0].g[91,0] = (-L_s_G10*i_sD_G10*omega_G10 - R_s_G10*i_sQ_G10 + v_mQ_G10 - v_sQ_G10)/L_s_G10
        struct[0].g[96,0] = eta_d_G10 - 2*(-0.8*R_v_G10*i_sd_G10 + 0.8*X_v_G10*i_sq_G10)/v_dc_G10
        struct[0].g[97,0] = eta_q_G10 - 2*(326.59863237109*DV_sat_G10 - 0.8*R_v_G10*i_sq_G10 - 0.8*X_v_G10*i_sd_G10 + 326.59863237109)/v_dc_G10
        struct[0].g[108,0] = DV_sat_G10 - K_q_G10*(-q_s_pu_G10 + q_s_ref_G10 + xi_q_G10/T_q_G10)
        struct[0].g[109,0] = 7.5e-6*i_sd_G10*v_sd_G10 + 7.5e-6*i_sq_G10*v_sq_G10 - p_s_pu_G10
        struct[0].g[110,0] = 7.5e-6*i_sd_G10*v_sq_G10 - 7.5e-6*i_sq_G10*v_sd_G10 - q_s_pu_G10
        struct[0].g[111,0] = K_f_G10*(omega_ref_G10 - omega_v_filt_G10) + K_f_sec*xi_f_sec/2 - p_m_ref_G10 + p_r_G10
        struct[0].g[112,0] = K_vpoi_G10*(-v_s_filt_G10 + v_s_ref_G10) + q_r_G10 - q_s_ref_G10
        struct[0].g[113,0] = (L_t_G14*i_tQ_G14*omega_G14 - R_t_G14*i_tD_G14 + eta_D_G14*v_dc_G14/2 - v_mD_G14)/L_t_G14
        struct[0].g[114,0] = (-L_t_G14*i_tD_G14*omega_G14 - R_t_G14*i_tQ_G14 + eta_Q_G14*v_dc_G14/2 - v_mQ_G14)/L_t_G14
        struct[0].g[115,0] = (C_m_G14*omega_G14*v_mQ_G14 - G_d_G14*v_mD_G14 - i_sD_G14 + i_tD_G14)/C_m_G14
        struct[0].g[116,0] = (-C_m_G14*omega_G14*v_mD_G14 - G_d_G14*v_mQ_G14 - i_sQ_G14 + i_tQ_G14)/C_m_G14
        struct[0].g[117,0] = (L_s_G14*i_sQ_G14*omega_G14 - R_s_G14*i_sD_G14 + v_mD_G14 - v_sD_G14)/L_s_G14
        struct[0].g[118,0] = (-L_s_G14*i_sD_G14*omega_G14 - R_s_G14*i_sQ_G14 + v_mQ_G14 - v_sQ_G14)/L_s_G14
        struct[0].g[123,0] = eta_d_G14 - 2*(-0.8*R_v_G14*i_sd_G14 + 0.8*X_v_G14*i_sq_G14)/v_dc_G14
        struct[0].g[124,0] = eta_q_G14 - 2*(326.59863237109*DV_sat_G14 - 0.8*R_v_G14*i_sq_G14 - 0.8*X_v_G14*i_sd_G14 + 326.59863237109)/v_dc_G14
        struct[0].g[135,0] = DV_sat_G14 - K_q_G14*(-q_s_pu_G14 + q_s_ref_G14 + xi_q_G14/T_q_G14)
        struct[0].g[136,0] = 7.5e-6*i_sd_G14*v_sd_G14 + 7.5e-6*i_sq_G14*v_sq_G14 - p_s_pu_G14
        struct[0].g[137,0] = 7.5e-6*i_sd_G14*v_sq_G14 - 7.5e-6*i_sq_G14*v_sd_G14 - q_s_pu_G14
        struct[0].g[138,0] = K_f_G14*(omega_ref_G14 - omega_v_filt_G14) + K_f_sec*xi_f_sec/2 - p_m_ref_G14 + p_r_G14
        struct[0].g[139,0] = K_vpoi_G14*(-v_s_filt_G14 + v_s_ref_G14) + q_r_G14 - q_s_ref_G14
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = i_sD_G10
        struct[0].h[1,0] = i_sQ_G10
        struct[0].h[2,0] = i_sD_G14
        struct[0].h[3,0] = i_sQ_G14
    

    if mode == 10:

        struct[0].Fx[12,12] = -K_phi_G10
        struct[0].Fx[12,13] = -314.159265358979*H_G10/(H_G10 + H_G14) + 314.159265358979
        struct[0].Fx[12,19] = -314.159265358979*H_G14/(H_G10 + H_G14)
        struct[0].Fx[13,13] = -D_G10/(2*H_G10)
        struct[0].Fx[16,13] = 1/T_f_G10
        struct[0].Fx[16,16] = -1/T_f_G10
        struct[0].Fx[17,17] = -1/T_vpoi_G10
        struct[0].Fx[18,13] = -314.159265358979*H_G10/(H_G10 + H_G14)
        struct[0].Fx[18,18] = -K_phi_G14
        struct[0].Fx[18,19] = -314.159265358979*H_G14/(H_G10 + H_G14) + 314.159265358979
        struct[0].Fx[19,19] = -D_G14/(2*H_G14)
        struct[0].Fx[22,19] = 1/T_f_G14
        struct[0].Fx[22,22] = -1/T_f_G14
        struct[0].Fx[23,23] = -1/T_vpoi_G14
        struct[0].Fx[24,13] = -H_G10/(H_G10 + H_G14)
        struct[0].Fx[24,19] = -H_G14/(H_G10 + H_G14)

    if mode == 11:

        struct[0].Fy[0,74] = 100.0*Piecewise(np.array([(0, (I_max_R01 < i_R01_d_ref) | (I_max_R01 < -i_R01_d_ref)), (1, True)]))
        struct[0].Fy[1,75] = 100.0*Piecewise(np.array([(0, (I_max_R01 < i_R01_q_ref) | (I_max_R01 < -i_R01_q_ref)), (1, True)]))
        struct[0].Fy[2,76] = 100.0*Piecewise(np.array([(0, (I_max_R11 < i_R11_d_ref) | (I_max_R11 < -i_R11_d_ref)), (1, True)]))
        struct[0].Fy[3,77] = 100.0*Piecewise(np.array([(0, (I_max_R11 < i_R11_q_ref) | (I_max_R11 < -i_R11_q_ref)), (1, True)]))
        struct[0].Fy[4,78] = 100.0*Piecewise(np.array([(0, (I_max_R15 < i_R15_d_ref) | (I_max_R15 < -i_R15_d_ref)), (1, True)]))
        struct[0].Fy[5,79] = 100.0*Piecewise(np.array([(0, (I_max_R15 < i_R15_q_ref) | (I_max_R15 < -i_R15_q_ref)), (1, True)]))
        struct[0].Fy[6,80] = 100.0*Piecewise(np.array([(0, (I_max_R16 < i_R16_d_ref) | (I_max_R16 < -i_R16_d_ref)), (1, True)]))
        struct[0].Fy[7,81] = 100.0*Piecewise(np.array([(0, (I_max_R16 < i_R16_q_ref) | (I_max_R16 < -i_R16_q_ref)), (1, True)]))
        struct[0].Fy[8,82] = 100.0*Piecewise(np.array([(0, (I_max_R17 < i_R17_d_ref) | (I_max_R17 < -i_R17_d_ref)), (1, True)]))
        struct[0].Fy[9,83] = 100.0*Piecewise(np.array([(0, (I_max_R17 < i_R17_q_ref) | (I_max_R17 < -i_R17_q_ref)), (1, True)]))
        struct[0].Fy[10,84] = 100.0*Piecewise(np.array([(0, (I_max_R18 < i_R18_d_ref) | (I_max_R18 < -i_R18_d_ref)), (1, True)]))
        struct[0].Fy[11,85] = 100.0*Piecewise(np.array([(0, (I_max_R18 < i_R18_q_ref) | (I_max_R18 < -i_R18_q_ref)), (1, True)]))
        struct[0].Fy[13,109] = -1/(2*H_G10)
        struct[0].Fy[13,111] = 1/(2*H_G10)
        struct[0].Fy[14,110] = -1
        struct[0].Fy[14,112] = 1
        struct[0].Fy[17,102] = 0.00306186217847897*v_sd_G10/(T_vpoi_G10*(v_sd_G10**2 + v_sq_G10**2)**0.5)
        struct[0].Fy[17,103] = 0.00306186217847897*v_sq_G10/(T_vpoi_G10*(v_sd_G10**2 + v_sq_G10**2)**0.5)
        struct[0].Fy[19,136] = -1/(2*H_G14)
        struct[0].Fy[19,138] = 1/(2*H_G14)
        struct[0].Fy[20,137] = -1
        struct[0].Fy[20,139] = 1
        struct[0].Fy[23,129] = 0.00306186217847897*v_sd_G14/(T_vpoi_G14*(v_sd_G14**2 + v_sq_G14**2)**0.5)
        struct[0].Fy[23,130] = 0.00306186217847897*v_sq_G14/(T_vpoi_G14*(v_sd_G14**2 + v_sq_G14**2)**0.5)

        struct[0].Gx[38,0] = 1
        struct[0].Gx[39,1] = 1
        struct[0].Gx[58,2] = 1
        struct[0].Gx[59,3] = 1
        struct[0].Gx[66,4] = 1
        struct[0].Gx[67,5] = 1
        struct[0].Gx[68,6] = 1
        struct[0].Gx[69,7] = 1
        struct[0].Gx[70,8] = 1
        struct[0].Gx[71,9] = 1
        struct[0].Gx[72,10] = 1
        struct[0].Gx[73,11] = 1
        struct[0].Gx[98,12] = -eta_d_G10*sin(phi_G10) - eta_q_G10*cos(phi_G10)
        struct[0].Gx[99,12] = eta_d_G10*cos(phi_G10) - eta_q_G10*sin(phi_G10)
        struct[0].Gx[100,12] = -v_mD_G10*sin(phi_G10) + v_mQ_G10*cos(phi_G10)
        struct[0].Gx[101,12] = -v_mD_G10*cos(phi_G10) - v_mQ_G10*sin(phi_G10)
        struct[0].Gx[102,12] = -v_sD_G10*sin(phi_G10) + v_sQ_G10*cos(phi_G10)
        struct[0].Gx[103,12] = -v_sD_G10*cos(phi_G10) - v_sQ_G10*sin(phi_G10)
        struct[0].Gx[104,12] = -i_tD_G10*sin(phi_G10) + i_tQ_G10*cos(phi_G10)
        struct[0].Gx[105,12] = -i_tD_G10*cos(phi_G10) - i_tQ_G10*sin(phi_G10)
        struct[0].Gx[106,12] = -i_sD_G10*sin(phi_G10) + i_sQ_G10*cos(phi_G10)
        struct[0].Gx[107,12] = -i_sD_G10*cos(phi_G10) - i_sQ_G10*sin(phi_G10)
        struct[0].Gx[108,14] = -K_q_G10/T_q_G10
        struct[0].Gx[111,16] = -K_f_G10
        struct[0].Gx[111,24] = K_f_sec/2
        struct[0].Gx[112,17] = -K_vpoi_G10
        struct[0].Gx[125,18] = -eta_d_G14*sin(phi_G14) - eta_q_G14*cos(phi_G14)
        struct[0].Gx[126,18] = eta_d_G14*cos(phi_G14) - eta_q_G14*sin(phi_G14)
        struct[0].Gx[127,18] = -v_mD_G14*sin(phi_G14) + v_mQ_G14*cos(phi_G14)
        struct[0].Gx[128,18] = -v_mD_G14*cos(phi_G14) - v_mQ_G14*sin(phi_G14)
        struct[0].Gx[129,18] = -v_sD_G14*sin(phi_G14) + v_sQ_G14*cos(phi_G14)
        struct[0].Gx[130,18] = -v_sD_G14*cos(phi_G14) - v_sQ_G14*sin(phi_G14)
        struct[0].Gx[131,18] = -i_tD_G14*sin(phi_G14) + i_tQ_G14*cos(phi_G14)
        struct[0].Gx[132,18] = -i_tD_G14*cos(phi_G14) - i_tQ_G14*sin(phi_G14)
        struct[0].Gx[133,18] = -i_sD_G14*sin(phi_G14) + i_sQ_G14*cos(phi_G14)
        struct[0].Gx[134,18] = -i_sD_G14*cos(phi_G14) - i_sQ_G14*sin(phi_G14)
        struct[0].Gx[135,20] = -K_q_G14/T_q_G14
        struct[0].Gx[138,22] = -K_f_G14
        struct[0].Gx[138,24] = K_f_sec/2
        struct[0].Gx[139,23] = -K_vpoi_G14

        struct[0].Gy[0,0] = -R_R00R01
        struct[0].Gy[0,1] = L_R00R01*omega
        struct[0].Gy[1,0] = -L_R00R01*omega
        struct[0].Gy[1,1] = -R_R00R01
        struct[0].Gy[2,2] = -R_R02R01
        struct[0].Gy[2,3] = L_R02R01*omega
        struct[0].Gy[3,2] = -L_R02R01*omega
        struct[0].Gy[3,3] = -R_R02R01
        struct[0].Gy[4,4] = -R_R02R03
        struct[0].Gy[4,5] = L_R02R03*omega
        struct[0].Gy[5,4] = -L_R02R03*omega
        struct[0].Gy[5,5] = -R_R02R03
        struct[0].Gy[6,6] = -R_R03R04
        struct[0].Gy[6,7] = L_R03R04*omega
        struct[0].Gy[7,6] = -L_R03R04*omega
        struct[0].Gy[7,7] = -R_R03R04
        struct[0].Gy[8,8] = -R_R04R05
        struct[0].Gy[8,9] = L_R04R05*omega
        struct[0].Gy[9,8] = -L_R04R05*omega
        struct[0].Gy[9,9] = -R_R04R05
        struct[0].Gy[10,10] = -R_R04R12
        struct[0].Gy[10,11] = L_R04R12*omega
        struct[0].Gy[11,10] = -L_R04R12*omega
        struct[0].Gy[11,11] = -R_R04R12
        struct[0].Gy[12,12] = -R_R05R06
        struct[0].Gy[12,13] = L_R05R06*omega
        struct[0].Gy[13,12] = -L_R05R06*omega
        struct[0].Gy[13,13] = -R_R05R06
        struct[0].Gy[14,14] = -R_R06R07
        struct[0].Gy[14,15] = L_R06R07*omega
        struct[0].Gy[15,14] = -L_R06R07*omega
        struct[0].Gy[15,15] = -R_R06R07
        struct[0].Gy[16,16] = -R_R07R08
        struct[0].Gy[16,17] = L_R07R08*omega
        struct[0].Gy[17,16] = -L_R07R08*omega
        struct[0].Gy[17,17] = -R_R07R08
        struct[0].Gy[18,18] = -R_R08R09
        struct[0].Gy[18,19] = L_R08R09*omega
        struct[0].Gy[19,18] = -L_R08R09*omega
        struct[0].Gy[19,19] = -R_R08R09
        struct[0].Gy[20,20] = -R_R09R10
        struct[0].Gy[20,21] = L_R09R10*omega
        struct[0].Gy[21,20] = -L_R09R10*omega
        struct[0].Gy[21,21] = -R_R09R10
        struct[0].Gy[22,22] = -R_R09R17
        struct[0].Gy[22,23] = L_R09R17*omega
        struct[0].Gy[23,22] = -L_R09R17*omega
        struct[0].Gy[23,23] = -R_R09R17
        struct[0].Gy[24,24] = -R_R11R03
        struct[0].Gy[24,25] = L_R11R03*omega
        struct[0].Gy[25,24] = -L_R11R03*omega
        struct[0].Gy[25,25] = -R_R11R03
        struct[0].Gy[26,26] = -R_R12R13
        struct[0].Gy[26,27] = L_R12R13*omega
        struct[0].Gy[27,26] = -L_R12R13*omega
        struct[0].Gy[27,27] = -R_R12R13
        struct[0].Gy[28,28] = -R_R13R14
        struct[0].Gy[28,29] = L_R13R14*omega
        struct[0].Gy[29,28] = -L_R13R14*omega
        struct[0].Gy[29,29] = -R_R13R14
        struct[0].Gy[30,30] = -R_R14R15
        struct[0].Gy[30,31] = L_R14R15*omega
        struct[0].Gy[31,30] = -L_R14R15*omega
        struct[0].Gy[31,31] = -R_R14R15
        struct[0].Gy[32,32] = -R_R16R06
        struct[0].Gy[32,33] = L_R16R06*omega
        struct[0].Gy[33,32] = -L_R16R06*omega
        struct[0].Gy[33,33] = -R_R16R06
        struct[0].Gy[34,34] = -R_R18R10
        struct[0].Gy[34,35] = L_R18R10*omega
        struct[0].Gy[35,34] = -L_R18R10*omega
        struct[0].Gy[35,35] = -R_R18R10
        struct[0].Gy[36,37] = C_R00R01*omega/2
        struct[0].Gy[37,36] = -C_R00R01*omega/2
        struct[0].Gy[38,39] = omega*(C_R00R01/2 + C_R02R01/2)
        struct[0].Gy[39,38] = -omega*(C_R00R01/2 + C_R02R01/2)
        struct[0].Gy[40,41] = omega*(C_R02R01/2 + C_R02R03/2)
        struct[0].Gy[41,40] = -omega*(C_R02R01/2 + C_R02R03/2)
        struct[0].Gy[42,43] = omega*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].Gy[43,42] = -omega*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].Gy[44,45] = omega*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].Gy[45,44] = -omega*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].Gy[46,47] = omega*(C_R04R05/2 + C_R05R06/2)
        struct[0].Gy[47,46] = -omega*(C_R04R05/2 + C_R05R06/2)
        struct[0].Gy[48,49] = omega*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].Gy[49,48] = -omega*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].Gy[50,51] = omega*(C_R06R07/2 + C_R07R08/2)
        struct[0].Gy[51,50] = -omega*(C_R06R07/2 + C_R07R08/2)
        struct[0].Gy[52,53] = omega*(C_R07R08/2 + C_R08R09/2)
        struct[0].Gy[53,52] = -omega*(C_R07R08/2 + C_R08R09/2)
        struct[0].Gy[54,55] = omega*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].Gy[55,54] = -omega*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].Gy[56,57] = omega*(C_R09R10/2 + C_R18R10/2)
        struct[0].Gy[57,56] = -omega*(C_R09R10/2 + C_R18R10/2)
        struct[0].Gy[58,59] = C_R11R03*omega/2
        struct[0].Gy[59,58] = -C_R11R03*omega/2
        struct[0].Gy[60,61] = omega*(C_R04R12/2 + C_R12R13/2)
        struct[0].Gy[61,60] = -omega*(C_R04R12/2 + C_R12R13/2)
        struct[0].Gy[62,63] = omega*(C_R12R13/2 + C_R13R14/2)
        struct[0].Gy[63,62] = -omega*(C_R12R13/2 + C_R13R14/2)
        struct[0].Gy[64,65] = omega*(C_R13R14/2 + C_R14R15/2)
        struct[0].Gy[65,64] = -omega*(C_R13R14/2 + C_R14R15/2)
        struct[0].Gy[66,67] = C_R14R15*omega/2
        struct[0].Gy[67,66] = -C_R14R15*omega/2
        struct[0].Gy[68,69] = C_R16R06*omega/2
        struct[0].Gy[69,68] = -C_R16R06*omega/2
        struct[0].Gy[70,71] = C_R09R17*omega/2
        struct[0].Gy[71,70] = -C_R09R17*omega/2
        struct[0].Gy[72,73] = C_R18R10*omega/2
        struct[0].Gy[73,72] = -C_R18R10*omega/2
        struct[0].Gy[74,38] = -0.666666666666667*p_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_D - 0.666666666666667*q_R01_ref*v_R01_Q)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_D/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy[74,39] = -0.666666666666667*q_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_D - 0.666666666666667*q_R01_ref*v_R01_Q)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_Q/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy[75,38] = 0.666666666666667*q_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_Q + 0.666666666666667*q_R01_ref*v_R01_D)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_D/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy[75,39] = -0.666666666666667*p_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_Q + 0.666666666666667*q_R01_ref*v_R01_D)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_Q/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy[76,58] = -0.666666666666667*p_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_D - 0.666666666666667*q_R11_ref*v_R11_Q)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_D/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy[76,59] = -0.666666666666667*q_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_D - 0.666666666666667*q_R11_ref*v_R11_Q)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_Q/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy[77,58] = 0.666666666666667*q_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_Q + 0.666666666666667*q_R11_ref*v_R11_D)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_D/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy[77,59] = -0.666666666666667*p_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_Q + 0.666666666666667*q_R11_ref*v_R11_D)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_Q/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy[78,66] = -0.666666666666667*p_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_D - 0.666666666666667*q_R15_ref*v_R15_Q)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_D/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy[78,67] = -0.666666666666667*q_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_D - 0.666666666666667*q_R15_ref*v_R15_Q)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_Q/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy[79,66] = 0.666666666666667*q_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_Q + 0.666666666666667*q_R15_ref*v_R15_D)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_D/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy[79,67] = -0.666666666666667*p_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_Q + 0.666666666666667*q_R15_ref*v_R15_D)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_Q/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy[80,68] = -0.666666666666667*p_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_D - 0.666666666666667*q_R16_ref*v_R16_Q)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_D/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy[80,69] = -0.666666666666667*q_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_D - 0.666666666666667*q_R16_ref*v_R16_Q)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_Q/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy[81,68] = 0.666666666666667*q_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_Q + 0.666666666666667*q_R16_ref*v_R16_D)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_D/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy[81,69] = -0.666666666666667*p_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_Q + 0.666666666666667*q_R16_ref*v_R16_D)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_Q/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy[82,70] = -0.666666666666667*p_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_D - 0.666666666666667*q_R17_ref*v_R17_Q)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_D/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy[82,71] = -0.666666666666667*q_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_D - 0.666666666666667*q_R17_ref*v_R17_Q)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_Q/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy[83,70] = 0.666666666666667*q_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_Q + 0.666666666666667*q_R17_ref*v_R17_D)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_D/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy[83,71] = -0.666666666666667*p_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_Q + 0.666666666666667*q_R17_ref*v_R17_D)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_Q/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy[84,72] = -0.666666666666667*p_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_D - 0.666666666666667*q_R18_ref*v_R18_Q)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_D/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy[84,73] = -0.666666666666667*q_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_D - 0.666666666666667*q_R18_ref*v_R18_Q)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_Q/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy[85,72] = 0.666666666666667*q_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_Q + 0.666666666666667*q_R18_ref*v_R18_D)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_D/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy[85,73] = -0.666666666666667*p_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_Q + 0.666666666666667*q_R18_ref*v_R18_D)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_Q/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy[86,86] = -R_t_G10/L_t_G10
        struct[0].Gy[86,87] = omega_G10
        struct[0].Gy[86,88] = -1/L_t_G10
        struct[0].Gy[86,98] = v_dc_G10/(2*L_t_G10)
        struct[0].Gy[87,86] = -omega_G10
        struct[0].Gy[87,87] = -R_t_G10/L_t_G10
        struct[0].Gy[87,89] = -1/L_t_G10
        struct[0].Gy[87,99] = v_dc_G10/(2*L_t_G10)
        struct[0].Gy[88,86] = 1/C_m_G10
        struct[0].Gy[88,88] = -G_d_G10/C_m_G10
        struct[0].Gy[88,89] = omega_G10
        struct[0].Gy[88,90] = -1/C_m_G10
        struct[0].Gy[89,87] = 1/C_m_G10
        struct[0].Gy[89,88] = -omega_G10
        struct[0].Gy[89,89] = -G_d_G10/C_m_G10
        struct[0].Gy[89,91] = -1/C_m_G10
        struct[0].Gy[90,88] = 1/L_s_G10
        struct[0].Gy[90,90] = -R_s_G10/L_s_G10
        struct[0].Gy[90,91] = omega_G10
        struct[0].Gy[90,94] = -1/L_s_G10
        struct[0].Gy[91,89] = 1/L_s_G10
        struct[0].Gy[91,90] = -omega_G10
        struct[0].Gy[91,91] = -R_s_G10/L_s_G10
        struct[0].Gy[91,95] = -1/L_s_G10
        struct[0].Gy[96,106] = 1.6*R_v_G10/v_dc_G10
        struct[0].Gy[96,107] = -1.6*X_v_G10/v_dc_G10
        struct[0].Gy[97,106] = 1.6*X_v_G10/v_dc_G10
        struct[0].Gy[97,107] = 1.6*R_v_G10/v_dc_G10
        struct[0].Gy[97,108] = -653.197264742181/v_dc_G10
        struct[0].Gy[98,96] = cos(phi_G10)
        struct[0].Gy[98,97] = -sin(phi_G10)
        struct[0].Gy[99,96] = sin(phi_G10)
        struct[0].Gy[99,97] = cos(phi_G10)
        struct[0].Gy[100,88] = cos(phi_G10)
        struct[0].Gy[100,89] = sin(phi_G10)
        struct[0].Gy[101,88] = -sin(phi_G10)
        struct[0].Gy[101,89] = cos(phi_G10)
        struct[0].Gy[102,94] = cos(phi_G10)
        struct[0].Gy[102,95] = sin(phi_G10)
        struct[0].Gy[103,94] = -sin(phi_G10)
        struct[0].Gy[103,95] = cos(phi_G10)
        struct[0].Gy[104,86] = cos(phi_G10)
        struct[0].Gy[104,87] = sin(phi_G10)
        struct[0].Gy[105,86] = -sin(phi_G10)
        struct[0].Gy[105,87] = cos(phi_G10)
        struct[0].Gy[106,90] = cos(phi_G10)
        struct[0].Gy[106,91] = sin(phi_G10)
        struct[0].Gy[107,90] = -sin(phi_G10)
        struct[0].Gy[107,91] = cos(phi_G10)
        struct[0].Gy[108,110] = K_q_G10
        struct[0].Gy[108,112] = -K_q_G10
        struct[0].Gy[109,102] = 7.5e-6*i_sd_G10
        struct[0].Gy[109,103] = 7.5e-6*i_sq_G10
        struct[0].Gy[109,106] = 7.5e-6*v_sd_G10
        struct[0].Gy[109,107] = 7.5e-6*v_sq_G10
        struct[0].Gy[110,102] = -7.5e-6*i_sq_G10
        struct[0].Gy[110,103] = 7.5e-6*i_sd_G10
        struct[0].Gy[110,106] = 7.5e-6*v_sq_G10
        struct[0].Gy[110,107] = -7.5e-6*v_sd_G10
        struct[0].Gy[113,113] = -R_t_G14/L_t_G14
        struct[0].Gy[113,114] = omega_G14
        struct[0].Gy[113,115] = -1/L_t_G14
        struct[0].Gy[113,125] = v_dc_G14/(2*L_t_G14)
        struct[0].Gy[114,113] = -omega_G14
        struct[0].Gy[114,114] = -R_t_G14/L_t_G14
        struct[0].Gy[114,116] = -1/L_t_G14
        struct[0].Gy[114,126] = v_dc_G14/(2*L_t_G14)
        struct[0].Gy[115,113] = 1/C_m_G14
        struct[0].Gy[115,115] = -G_d_G14/C_m_G14
        struct[0].Gy[115,116] = omega_G14
        struct[0].Gy[115,117] = -1/C_m_G14
        struct[0].Gy[116,114] = 1/C_m_G14
        struct[0].Gy[116,115] = -omega_G14
        struct[0].Gy[116,116] = -G_d_G14/C_m_G14
        struct[0].Gy[116,118] = -1/C_m_G14
        struct[0].Gy[117,115] = 1/L_s_G14
        struct[0].Gy[117,117] = -R_s_G14/L_s_G14
        struct[0].Gy[117,118] = omega_G14
        struct[0].Gy[117,121] = -1/L_s_G14
        struct[0].Gy[118,116] = 1/L_s_G14
        struct[0].Gy[118,117] = -omega_G14
        struct[0].Gy[118,118] = -R_s_G14/L_s_G14
        struct[0].Gy[118,122] = -1/L_s_G14
        struct[0].Gy[123,133] = 1.6*R_v_G14/v_dc_G14
        struct[0].Gy[123,134] = -1.6*X_v_G14/v_dc_G14
        struct[0].Gy[124,133] = 1.6*X_v_G14/v_dc_G14
        struct[0].Gy[124,134] = 1.6*R_v_G14/v_dc_G14
        struct[0].Gy[124,135] = -653.197264742181/v_dc_G14
        struct[0].Gy[125,123] = cos(phi_G14)
        struct[0].Gy[125,124] = -sin(phi_G14)
        struct[0].Gy[126,123] = sin(phi_G14)
        struct[0].Gy[126,124] = cos(phi_G14)
        struct[0].Gy[127,115] = cos(phi_G14)
        struct[0].Gy[127,116] = sin(phi_G14)
        struct[0].Gy[128,115] = -sin(phi_G14)
        struct[0].Gy[128,116] = cos(phi_G14)
        struct[0].Gy[129,121] = cos(phi_G14)
        struct[0].Gy[129,122] = sin(phi_G14)
        struct[0].Gy[130,121] = -sin(phi_G14)
        struct[0].Gy[130,122] = cos(phi_G14)
        struct[0].Gy[131,113] = cos(phi_G14)
        struct[0].Gy[131,114] = sin(phi_G14)
        struct[0].Gy[132,113] = -sin(phi_G14)
        struct[0].Gy[132,114] = cos(phi_G14)
        struct[0].Gy[133,117] = cos(phi_G14)
        struct[0].Gy[133,118] = sin(phi_G14)
        struct[0].Gy[134,117] = -sin(phi_G14)
        struct[0].Gy[134,118] = cos(phi_G14)
        struct[0].Gy[135,137] = K_q_G14
        struct[0].Gy[135,139] = -K_q_G14
        struct[0].Gy[136,129] = 7.5e-6*i_sd_G14
        struct[0].Gy[136,130] = 7.5e-6*i_sq_G14
        struct[0].Gy[136,133] = 7.5e-6*v_sd_G14
        struct[0].Gy[136,134] = 7.5e-6*v_sq_G14
        struct[0].Gy[137,129] = -7.5e-6*i_sq_G14
        struct[0].Gy[137,130] = 7.5e-6*i_sd_G14
        struct[0].Gy[137,133] = 7.5e-6*v_sq_G14
        struct[0].Gy[137,134] = -7.5e-6*v_sd_G14

    if mode > 12:

        struct[0].Fu[0,1] = 100.0*Piecewise(np.array([(-1, I_max_R01 < -i_R01_d_ref), (1, I_max_R01 < i_R01_d_ref), (0, True)]))
        struct[0].Fu[1,1] = 100.0*Piecewise(np.array([(-1, I_max_R01 < -i_R01_q_ref), (1, I_max_R01 < i_R01_q_ref), (0, True)]))
        struct[0].Fu[2,5] = 100.0*Piecewise(np.array([(-1, I_max_R11 < -i_R11_d_ref), (1, I_max_R11 < i_R11_d_ref), (0, True)]))
        struct[0].Fu[3,5] = 100.0*Piecewise(np.array([(-1, I_max_R11 < -i_R11_q_ref), (1, I_max_R11 < i_R11_q_ref), (0, True)]))
        struct[0].Fu[4,9] = 100.0*Piecewise(np.array([(-1, I_max_R15 < -i_R15_d_ref), (1, I_max_R15 < i_R15_d_ref), (0, True)]))
        struct[0].Fu[5,9] = 100.0*Piecewise(np.array([(-1, I_max_R15 < -i_R15_q_ref), (1, I_max_R15 < i_R15_q_ref), (0, True)]))
        struct[0].Fu[6,13] = 100.0*Piecewise(np.array([(-1, I_max_R16 < -i_R16_d_ref), (1, I_max_R16 < i_R16_d_ref), (0, True)]))
        struct[0].Fu[7,13] = 100.0*Piecewise(np.array([(-1, I_max_R16 < -i_R16_q_ref), (1, I_max_R16 < i_R16_q_ref), (0, True)]))
        struct[0].Fu[8,17] = 100.0*Piecewise(np.array([(-1, I_max_R17 < -i_R17_d_ref), (1, I_max_R17 < i_R17_d_ref), (0, True)]))
        struct[0].Fu[9,17] = 100.0*Piecewise(np.array([(-1, I_max_R17 < -i_R17_q_ref), (1, I_max_R17 < i_R17_q_ref), (0, True)]))
        struct[0].Fu[10,21] = 100.0*Piecewise(np.array([(-1, I_max_R18 < -i_R18_d_ref), (1, I_max_R18 < i_R18_d_ref), (0, True)]))
        struct[0].Fu[11,21] = 100.0*Piecewise(np.array([(-1, I_max_R18 < -i_R18_q_ref), (1, I_max_R18 < i_R18_q_ref), (0, True)]))
        struct[0].Fu[13,25] = 1/(2*H_G10)
        struct[0].Fu[14,26] = 1
        struct[0].Fu[19,32] = 1/(2*H_G14)
        struct[0].Fu[20,33] = 1

        struct[0].Gu[74,2] = -0.666666666666667*v_R01_D*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)]))
        struct[0].Gu[74,3] = -0.666666666666667*v_R01_Q*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)]))
        struct[0].Gu[75,2] = -0.666666666666667*v_R01_Q*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)]))
        struct[0].Gu[75,3] = 0.666666666666667*v_R01_D*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)]))
        struct[0].Gu[76,6] = -0.666666666666667*v_R11_D*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)]))
        struct[0].Gu[76,7] = -0.666666666666667*v_R11_Q*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)]))
        struct[0].Gu[77,6] = -0.666666666666667*v_R11_Q*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)]))
        struct[0].Gu[77,7] = 0.666666666666667*v_R11_D*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)]))
        struct[0].Gu[78,10] = -0.666666666666667*v_R15_D*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)]))
        struct[0].Gu[78,11] = -0.666666666666667*v_R15_Q*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)]))
        struct[0].Gu[79,10] = -0.666666666666667*v_R15_Q*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)]))
        struct[0].Gu[79,11] = 0.666666666666667*v_R15_D*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)]))
        struct[0].Gu[80,14] = -0.666666666666667*v_R16_D*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)]))
        struct[0].Gu[80,15] = -0.666666666666667*v_R16_Q*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)]))
        struct[0].Gu[81,14] = -0.666666666666667*v_R16_Q*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)]))
        struct[0].Gu[81,15] = 0.666666666666667*v_R16_D*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)]))
        struct[0].Gu[82,18] = -0.666666666666667*v_R17_D*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)]))
        struct[0].Gu[82,19] = -0.666666666666667*v_R17_Q*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)]))
        struct[0].Gu[83,18] = -0.666666666666667*v_R17_Q*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)]))
        struct[0].Gu[83,19] = 0.666666666666667*v_R17_D*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)]))
        struct[0].Gu[84,22] = -0.666666666666667*v_R18_D*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)]))
        struct[0].Gu[84,23] = -0.666666666666667*v_R18_Q*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)]))
        struct[0].Gu[85,22] = -0.666666666666667*v_R18_Q*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)]))
        struct[0].Gu[85,23] = 0.666666666666667*v_R18_D*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)]))
        struct[0].Gu[86,24] = eta_D_G10/(2*L_t_G10)
        struct[0].Gu[87,24] = eta_Q_G10/(2*L_t_G10)
        struct[0].Gu[96,24] = 2*(-0.8*R_v_G10*i_sd_G10 + 0.8*X_v_G10*i_sq_G10)/v_dc_G10**2
        struct[0].Gu[97,24] = 2*(326.59863237109*DV_sat_G10 - 0.8*R_v_G10*i_sq_G10 - 0.8*X_v_G10*i_sd_G10 + 326.59863237109)/v_dc_G10**2
        struct[0].Gu[108,26] = -K_q_G10
        struct[0].Gu[111,28] = K_f_G10
        struct[0].Gu[112,27] = K_vpoi_G10
        struct[0].Gu[113,31] = eta_D_G14/(2*L_t_G14)
        struct[0].Gu[114,31] = eta_Q_G14/(2*L_t_G14)
        struct[0].Gu[123,31] = 2*(-0.8*R_v_G14*i_sd_G14 + 0.8*X_v_G14*i_sq_G14)/v_dc_G14**2
        struct[0].Gu[124,31] = 2*(326.59863237109*DV_sat_G14 - 0.8*R_v_G14*i_sq_G14 - 0.8*X_v_G14*i_sd_G14 + 326.59863237109)/v_dc_G14**2
        struct[0].Gu[135,33] = -K_q_G14
        struct[0].Gu[138,35] = K_f_G14
        struct[0].Gu[139,34] = K_vpoi_G14


        struct[0].Hy[0,90] = 1
        struct[0].Hy[1,91] = 1
        struct[0].Hy[2,117] = 1
        struct[0].Hy[3,118] = 1




def ini_nn(struct,mode):

    # Parameters:
    R_R00R01 = struct[0].R_R00R01
    L_R00R01 = struct[0].L_R00R01
    C_R00R01 = struct[0].C_R00R01
    R_R02R01 = struct[0].R_R02R01
    L_R02R01 = struct[0].L_R02R01
    C_R02R01 = struct[0].C_R02R01
    R_R02R03 = struct[0].R_R02R03
    L_R02R03 = struct[0].L_R02R03
    C_R02R03 = struct[0].C_R02R03
    R_R03R04 = struct[0].R_R03R04
    L_R03R04 = struct[0].L_R03R04
    C_R03R04 = struct[0].C_R03R04
    R_R04R05 = struct[0].R_R04R05
    L_R04R05 = struct[0].L_R04R05
    C_R04R05 = struct[0].C_R04R05
    R_R04R12 = struct[0].R_R04R12
    L_R04R12 = struct[0].L_R04R12
    C_R04R12 = struct[0].C_R04R12
    R_R05R06 = struct[0].R_R05R06
    L_R05R06 = struct[0].L_R05R06
    C_R05R06 = struct[0].C_R05R06
    R_R06R07 = struct[0].R_R06R07
    L_R06R07 = struct[0].L_R06R07
    C_R06R07 = struct[0].C_R06R07
    R_R07R08 = struct[0].R_R07R08
    L_R07R08 = struct[0].L_R07R08
    C_R07R08 = struct[0].C_R07R08
    R_R08R09 = struct[0].R_R08R09
    L_R08R09 = struct[0].L_R08R09
    C_R08R09 = struct[0].C_R08R09
    R_R09R10 = struct[0].R_R09R10
    L_R09R10 = struct[0].L_R09R10
    C_R09R10 = struct[0].C_R09R10
    R_R09R17 = struct[0].R_R09R17
    L_R09R17 = struct[0].L_R09R17
    C_R09R17 = struct[0].C_R09R17
    R_R11R03 = struct[0].R_R11R03
    L_R11R03 = struct[0].L_R11R03
    C_R11R03 = struct[0].C_R11R03
    R_R12R13 = struct[0].R_R12R13
    L_R12R13 = struct[0].L_R12R13
    C_R12R13 = struct[0].C_R12R13
    R_R13R14 = struct[0].R_R13R14
    L_R13R14 = struct[0].L_R13R14
    C_R13R14 = struct[0].C_R13R14
    R_R14R15 = struct[0].R_R14R15
    L_R14R15 = struct[0].L_R14R15
    C_R14R15 = struct[0].C_R14R15
    R_R16R06 = struct[0].R_R16R06
    L_R16R06 = struct[0].L_R16R06
    C_R16R06 = struct[0].C_R16R06
    R_R18R10 = struct[0].R_R18R10
    L_R18R10 = struct[0].L_R18R10
    C_R18R10 = struct[0].C_R18R10
    i_R00_D = struct[0].i_R00_D
    i_R00_Q = struct[0].i_R00_Q
    i_R02_D = struct[0].i_R02_D
    i_R02_Q = struct[0].i_R02_Q
    i_R03_D = struct[0].i_R03_D
    i_R03_Q = struct[0].i_R03_Q
    i_R04_D = struct[0].i_R04_D
    i_R04_Q = struct[0].i_R04_Q
    i_R05_D = struct[0].i_R05_D
    i_R05_Q = struct[0].i_R05_Q
    i_R06_D = struct[0].i_R06_D
    i_R06_Q = struct[0].i_R06_Q
    i_R07_D = struct[0].i_R07_D
    i_R07_Q = struct[0].i_R07_Q
    i_R08_D = struct[0].i_R08_D
    i_R08_Q = struct[0].i_R08_Q
    i_R09_D = struct[0].i_R09_D
    i_R09_Q = struct[0].i_R09_Q
    i_R12_D = struct[0].i_R12_D
    i_R12_Q = struct[0].i_R12_Q
    i_R13_D = struct[0].i_R13_D
    i_R13_Q = struct[0].i_R13_Q
    omega = struct[0].omega
    L_t_G10 = struct[0].L_t_G10
    R_t_G10 = struct[0].R_t_G10
    C_m_G10 = struct[0].C_m_G10
    L_s_G10 = struct[0].L_s_G10
    R_s_G10 = struct[0].R_s_G10
    omega_G10 = struct[0].omega_G10
    G_d_G10 = struct[0].G_d_G10
    K_p_G10 = struct[0].K_p_G10
    T_p_G10 = struct[0].T_p_G10
    K_q_G10 = struct[0].K_q_G10
    T_q_G10 = struct[0].T_q_G10
    R_v_G10 = struct[0].R_v_G10
    X_v_G10 = struct[0].X_v_G10
    S_b_kVA_G10 = struct[0].S_b_kVA_G10
    U_b_G10 = struct[0].U_b_G10
    K_phi_G10 = struct[0].K_phi_G10
    H_G10 = struct[0].H_G10
    D_G10 = struct[0].D_G10
    T_vpoi_G10 = struct[0].T_vpoi_G10
    K_vpoi_G10 = struct[0].K_vpoi_G10
    T_f_G10 = struct[0].T_f_G10
    K_f_G10 = struct[0].K_f_G10
    L_t_G14 = struct[0].L_t_G14
    R_t_G14 = struct[0].R_t_G14
    C_m_G14 = struct[0].C_m_G14
    L_s_G14 = struct[0].L_s_G14
    R_s_G14 = struct[0].R_s_G14
    omega_G14 = struct[0].omega_G14
    G_d_G14 = struct[0].G_d_G14
    K_p_G14 = struct[0].K_p_G14
    T_p_G14 = struct[0].T_p_G14
    K_q_G14 = struct[0].K_q_G14
    T_q_G14 = struct[0].T_q_G14
    R_v_G14 = struct[0].R_v_G14
    X_v_G14 = struct[0].X_v_G14
    S_b_kVA_G14 = struct[0].S_b_kVA_G14
    U_b_G14 = struct[0].U_b_G14
    K_phi_G14 = struct[0].K_phi_G14
    H_G14 = struct[0].H_G14
    D_G14 = struct[0].D_G14
    T_vpoi_G14 = struct[0].T_vpoi_G14
    K_vpoi_G14 = struct[0].K_vpoi_G14
    T_f_G14 = struct[0].T_f_G14
    K_f_G14 = struct[0].K_f_G14
    K_f_sec = struct[0].K_f_sec
    
    # Inputs:
    T_i_R01 = struct[0].T_i_R01
    I_max_R01 = struct[0].I_max_R01
    p_R01_ref = struct[0].p_R01_ref
    q_R01_ref = struct[0].q_R01_ref
    T_i_R11 = struct[0].T_i_R11
    I_max_R11 = struct[0].I_max_R11
    p_R11_ref = struct[0].p_R11_ref
    q_R11_ref = struct[0].q_R11_ref
    T_i_R15 = struct[0].T_i_R15
    I_max_R15 = struct[0].I_max_R15
    p_R15_ref = struct[0].p_R15_ref
    q_R15_ref = struct[0].q_R15_ref
    T_i_R16 = struct[0].T_i_R16
    I_max_R16 = struct[0].I_max_R16
    p_R16_ref = struct[0].p_R16_ref
    q_R16_ref = struct[0].q_R16_ref
    T_i_R17 = struct[0].T_i_R17
    I_max_R17 = struct[0].I_max_R17
    p_R17_ref = struct[0].p_R17_ref
    q_R17_ref = struct[0].q_R17_ref
    T_i_R18 = struct[0].T_i_R18
    I_max_R18 = struct[0].I_max_R18
    p_R18_ref = struct[0].p_R18_ref
    q_R18_ref = struct[0].q_R18_ref
    v_dc_G10 = struct[0].v_dc_G10
    p_m_ref_G10 = struct[0].p_m_ref_G10
    q_s_ref_G10 = struct[0].q_s_ref_G10
    v_s_ref_G10 = struct[0].v_s_ref_G10
    omega_ref_G10 = struct[0].omega_ref_G10
    p_r_G10 = struct[0].p_r_G10
    q_r_G10 = struct[0].q_r_G10
    v_dc_G14 = struct[0].v_dc_G14
    p_m_ref_G14 = struct[0].p_m_ref_G14
    q_s_ref_G14 = struct[0].q_s_ref_G14
    v_s_ref_G14 = struct[0].v_s_ref_G14
    omega_ref_G14 = struct[0].omega_ref_G14
    p_r_G14 = struct[0].p_r_G14
    q_r_G14 = struct[0].q_r_G14
    
    # Dynamical states:
    i_R01_D = struct[0].x[0,0]
    i_R01_Q = struct[0].x[1,0]
    i_R11_D = struct[0].x[2,0]
    i_R11_Q = struct[0].x[3,0]
    i_R15_D = struct[0].x[4,0]
    i_R15_Q = struct[0].x[5,0]
    i_R16_D = struct[0].x[6,0]
    i_R16_Q = struct[0].x[7,0]
    i_R17_D = struct[0].x[8,0]
    i_R17_Q = struct[0].x[9,0]
    i_R18_D = struct[0].x[10,0]
    i_R18_Q = struct[0].x[11,0]
    phi_G10 = struct[0].x[12,0]
    omega_v_G10 = struct[0].x[13,0]
    xi_q_G10 = struct[0].x[14,0]
    omega_rads_G10 = struct[0].x[15,0]
    omega_v_filt_G10 = struct[0].x[16,0]
    v_s_filt_G10 = struct[0].x[17,0]
    phi_G14 = struct[0].x[18,0]
    omega_v_G14 = struct[0].x[19,0]
    xi_q_G14 = struct[0].x[20,0]
    omega_rads_G14 = struct[0].x[21,0]
    omega_v_filt_G14 = struct[0].x[22,0]
    v_s_filt_G14 = struct[0].x[23,0]
    xi_f_sec = struct[0].x[24,0]
    
    # Algebraic states:
    i_l_R00R01_D = struct[0].y_ini[0,0]
    i_l_R00R01_Q = struct[0].y_ini[1,0]
    i_l_R02R01_D = struct[0].y_ini[2,0]
    i_l_R02R01_Q = struct[0].y_ini[3,0]
    i_l_R02R03_D = struct[0].y_ini[4,0]
    i_l_R02R03_Q = struct[0].y_ini[5,0]
    i_l_R03R04_D = struct[0].y_ini[6,0]
    i_l_R03R04_Q = struct[0].y_ini[7,0]
    i_l_R04R05_D = struct[0].y_ini[8,0]
    i_l_R04R05_Q = struct[0].y_ini[9,0]
    i_l_R04R12_D = struct[0].y_ini[10,0]
    i_l_R04R12_Q = struct[0].y_ini[11,0]
    i_l_R05R06_D = struct[0].y_ini[12,0]
    i_l_R05R06_Q = struct[0].y_ini[13,0]
    i_l_R06R07_D = struct[0].y_ini[14,0]
    i_l_R06R07_Q = struct[0].y_ini[15,0]
    i_l_R07R08_D = struct[0].y_ini[16,0]
    i_l_R07R08_Q = struct[0].y_ini[17,0]
    i_l_R08R09_D = struct[0].y_ini[18,0]
    i_l_R08R09_Q = struct[0].y_ini[19,0]
    i_l_R09R10_D = struct[0].y_ini[20,0]
    i_l_R09R10_Q = struct[0].y_ini[21,0]
    i_l_R09R17_D = struct[0].y_ini[22,0]
    i_l_R09R17_Q = struct[0].y_ini[23,0]
    i_l_R11R03_D = struct[0].y_ini[24,0]
    i_l_R11R03_Q = struct[0].y_ini[25,0]
    i_l_R12R13_D = struct[0].y_ini[26,0]
    i_l_R12R13_Q = struct[0].y_ini[27,0]
    i_l_R13R14_D = struct[0].y_ini[28,0]
    i_l_R13R14_Q = struct[0].y_ini[29,0]
    i_l_R14R15_D = struct[0].y_ini[30,0]
    i_l_R14R15_Q = struct[0].y_ini[31,0]
    i_l_R16R06_D = struct[0].y_ini[32,0]
    i_l_R16R06_Q = struct[0].y_ini[33,0]
    i_l_R18R10_D = struct[0].y_ini[34,0]
    i_l_R18R10_Q = struct[0].y_ini[35,0]
    v_R00_D = struct[0].y_ini[36,0]
    v_R00_Q = struct[0].y_ini[37,0]
    v_R01_D = struct[0].y_ini[38,0]
    v_R01_Q = struct[0].y_ini[39,0]
    v_R02_D = struct[0].y_ini[40,0]
    v_R02_Q = struct[0].y_ini[41,0]
    v_R03_D = struct[0].y_ini[42,0]
    v_R03_Q = struct[0].y_ini[43,0]
    v_R04_D = struct[0].y_ini[44,0]
    v_R04_Q = struct[0].y_ini[45,0]
    v_R05_D = struct[0].y_ini[46,0]
    v_R05_Q = struct[0].y_ini[47,0]
    v_R06_D = struct[0].y_ini[48,0]
    v_R06_Q = struct[0].y_ini[49,0]
    v_R07_D = struct[0].y_ini[50,0]
    v_R07_Q = struct[0].y_ini[51,0]
    v_R08_D = struct[0].y_ini[52,0]
    v_R08_Q = struct[0].y_ini[53,0]
    v_R09_D = struct[0].y_ini[54,0]
    v_R09_Q = struct[0].y_ini[55,0]
    v_R10_D = struct[0].y_ini[56,0]
    v_R10_Q = struct[0].y_ini[57,0]
    v_R11_D = struct[0].y_ini[58,0]
    v_R11_Q = struct[0].y_ini[59,0]
    v_R12_D = struct[0].y_ini[60,0]
    v_R12_Q = struct[0].y_ini[61,0]
    v_R13_D = struct[0].y_ini[62,0]
    v_R13_Q = struct[0].y_ini[63,0]
    v_R14_D = struct[0].y_ini[64,0]
    v_R14_Q = struct[0].y_ini[65,0]
    v_R15_D = struct[0].y_ini[66,0]
    v_R15_Q = struct[0].y_ini[67,0]
    v_R16_D = struct[0].y_ini[68,0]
    v_R16_Q = struct[0].y_ini[69,0]
    v_R17_D = struct[0].y_ini[70,0]
    v_R17_Q = struct[0].y_ini[71,0]
    v_R18_D = struct[0].y_ini[72,0]
    v_R18_Q = struct[0].y_ini[73,0]
    i_R01_d_ref = struct[0].y_ini[74,0]
    i_R01_q_ref = struct[0].y_ini[75,0]
    i_R11_d_ref = struct[0].y_ini[76,0]
    i_R11_q_ref = struct[0].y_ini[77,0]
    i_R15_d_ref = struct[0].y_ini[78,0]
    i_R15_q_ref = struct[0].y_ini[79,0]
    i_R16_d_ref = struct[0].y_ini[80,0]
    i_R16_q_ref = struct[0].y_ini[81,0]
    i_R17_d_ref = struct[0].y_ini[82,0]
    i_R17_q_ref = struct[0].y_ini[83,0]
    i_R18_d_ref = struct[0].y_ini[84,0]
    i_R18_q_ref = struct[0].y_ini[85,0]
    i_tD_G10 = struct[0].y_ini[86,0]
    i_tQ_G10 = struct[0].y_ini[87,0]
    v_mD_G10 = struct[0].y_ini[88,0]
    v_mQ_G10 = struct[0].y_ini[89,0]
    i_sD_G10 = struct[0].y_ini[90,0]
    i_sQ_G10 = struct[0].y_ini[91,0]
    i_R10_D = struct[0].y_ini[92,0]
    i_R10_Q = struct[0].y_ini[93,0]
    v_sD_G10 = struct[0].y_ini[94,0]
    v_sQ_G10 = struct[0].y_ini[95,0]
    eta_d_G10 = struct[0].y_ini[96,0]
    eta_q_G10 = struct[0].y_ini[97,0]
    eta_D_G10 = struct[0].y_ini[98,0]
    eta_Q_G10 = struct[0].y_ini[99,0]
    v_md_G10 = struct[0].y_ini[100,0]
    v_mq_G10 = struct[0].y_ini[101,0]
    v_sd_G10 = struct[0].y_ini[102,0]
    v_sq_G10 = struct[0].y_ini[103,0]
    i_td_G10 = struct[0].y_ini[104,0]
    i_tq_G10 = struct[0].y_ini[105,0]
    i_sd_G10 = struct[0].y_ini[106,0]
    i_sq_G10 = struct[0].y_ini[107,0]
    DV_sat_G10 = struct[0].y_ini[108,0]
    p_s_pu_G10 = struct[0].y_ini[109,0]
    q_s_pu_G10 = struct[0].y_ini[110,0]
    p_m_ref_G10 = struct[0].y_ini[111,0]
    q_s_ref_G10 = struct[0].y_ini[112,0]
    i_tD_G14 = struct[0].y_ini[113,0]
    i_tQ_G14 = struct[0].y_ini[114,0]
    v_mD_G14 = struct[0].y_ini[115,0]
    v_mQ_G14 = struct[0].y_ini[116,0]
    i_sD_G14 = struct[0].y_ini[117,0]
    i_sQ_G14 = struct[0].y_ini[118,0]
    i_R14_D = struct[0].y_ini[119,0]
    i_R14_Q = struct[0].y_ini[120,0]
    v_sD_G14 = struct[0].y_ini[121,0]
    v_sQ_G14 = struct[0].y_ini[122,0]
    eta_d_G14 = struct[0].y_ini[123,0]
    eta_q_G14 = struct[0].y_ini[124,0]
    eta_D_G14 = struct[0].y_ini[125,0]
    eta_Q_G14 = struct[0].y_ini[126,0]
    v_md_G14 = struct[0].y_ini[127,0]
    v_mq_G14 = struct[0].y_ini[128,0]
    v_sd_G14 = struct[0].y_ini[129,0]
    v_sq_G14 = struct[0].y_ini[130,0]
    i_td_G14 = struct[0].y_ini[131,0]
    i_tq_G14 = struct[0].y_ini[132,0]
    i_sd_G14 = struct[0].y_ini[133,0]
    i_sq_G14 = struct[0].y_ini[134,0]
    DV_sat_G14 = struct[0].y_ini[135,0]
    p_s_pu_G14 = struct[0].y_ini[136,0]
    q_s_pu_G14 = struct[0].y_ini[137,0]
    p_m_ref_G14 = struct[0].y_ini[138,0]
    q_s_ref_G14 = struct[0].y_ini[139,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -100.0*i_R01_D + 100.0*Piecewise(np.array([(-I_max_R01, I_max_R01 < -i_R01_d_ref), (I_max_R01, I_max_R01 < i_R01_d_ref), (i_R01_d_ref, True)]))
        struct[0].f[1,0] = -100.0*i_R01_Q + 100.0*Piecewise(np.array([(-I_max_R01, I_max_R01 < -i_R01_q_ref), (I_max_R01, I_max_R01 < i_R01_q_ref), (i_R01_q_ref, True)]))
        struct[0].f[2,0] = -100.0*i_R11_D + 100.0*Piecewise(np.array([(-I_max_R11, I_max_R11 < -i_R11_d_ref), (I_max_R11, I_max_R11 < i_R11_d_ref), (i_R11_d_ref, True)]))
        struct[0].f[3,0] = -100.0*i_R11_Q + 100.0*Piecewise(np.array([(-I_max_R11, I_max_R11 < -i_R11_q_ref), (I_max_R11, I_max_R11 < i_R11_q_ref), (i_R11_q_ref, True)]))
        struct[0].f[4,0] = -100.0*i_R15_D + 100.0*Piecewise(np.array([(-I_max_R15, I_max_R15 < -i_R15_d_ref), (I_max_R15, I_max_R15 < i_R15_d_ref), (i_R15_d_ref, True)]))
        struct[0].f[5,0] = -100.0*i_R15_Q + 100.0*Piecewise(np.array([(-I_max_R15, I_max_R15 < -i_R15_q_ref), (I_max_R15, I_max_R15 < i_R15_q_ref), (i_R15_q_ref, True)]))
        struct[0].f[6,0] = -100.0*i_R16_D + 100.0*Piecewise(np.array([(-I_max_R16, I_max_R16 < -i_R16_d_ref), (I_max_R16, I_max_R16 < i_R16_d_ref), (i_R16_d_ref, True)]))
        struct[0].f[7,0] = -100.0*i_R16_Q + 100.0*Piecewise(np.array([(-I_max_R16, I_max_R16 < -i_R16_q_ref), (I_max_R16, I_max_R16 < i_R16_q_ref), (i_R16_q_ref, True)]))
        struct[0].f[8,0] = -100.0*i_R17_D + 100.0*Piecewise(np.array([(-I_max_R17, I_max_R17 < -i_R17_d_ref), (I_max_R17, I_max_R17 < i_R17_d_ref), (i_R17_d_ref, True)]))
        struct[0].f[9,0] = -100.0*i_R17_Q + 100.0*Piecewise(np.array([(-I_max_R17, I_max_R17 < -i_R17_q_ref), (I_max_R17, I_max_R17 < i_R17_q_ref), (i_R17_q_ref, True)]))
        struct[0].f[10,0] = -100.0*i_R18_D + 100.0*Piecewise(np.array([(-I_max_R18, I_max_R18 < -i_R18_d_ref), (I_max_R18, I_max_R18 < i_R18_d_ref), (i_R18_d_ref, True)]))
        struct[0].f[11,0] = -100.0*i_R18_Q + 100.0*Piecewise(np.array([(-I_max_R18, I_max_R18 < -i_R18_q_ref), (I_max_R18, I_max_R18 < i_R18_q_ref), (i_R18_q_ref, True)]))
        struct[0].f[12,0] = -K_phi_G10*phi_G10 + 314.159265358979*omega_v_G10 - 314.159265358979*(H_G10*omega_v_G10 + H_G14*omega_v_G14)/(H_G10 + H_G14)
        struct[0].f[13,0] = (-D_G10*(omega_v_G10 - 1.0) + p_m_ref_G10 - p_s_pu_G10)/(2*H_G10)
        struct[0].f[14,0] = -q_s_pu_G10 + q_s_ref_G10
        struct[0].f[15,0] = -1.0*omega_rads_G10 + 314.159265358979*omega_v_G10
        struct[0].f[16,0] = (omega_v_G10 - omega_v_filt_G10)/T_f_G10
        struct[0].f[17,0] = (-v_s_filt_G10 + 0.00306186217847897*(v_sd_G10**2 + v_sq_G10**2)**0.5)/T_vpoi_G10
        struct[0].f[18,0] = -K_phi_G14*phi_G14 + 314.159265358979*omega_v_G14 - 314.159265358979*(H_G10*omega_v_G10 + H_G14*omega_v_G14)/(H_G10 + H_G14)
        struct[0].f[19,0] = (-D_G14*(omega_v_G14 - 1.0) + p_m_ref_G14 - p_s_pu_G14)/(2*H_G14)
        struct[0].f[20,0] = -q_s_pu_G14 + q_s_ref_G14
        struct[0].f[21,0] = -1.0*omega_rads_G14 + 314.159265358979*omega_v_G14
        struct[0].f[22,0] = (omega_v_G14 - omega_v_filt_G14)/T_f_G14
        struct[0].f[23,0] = (-v_s_filt_G14 + 0.00306186217847897*(v_sd_G14**2 + v_sq_G14**2)**0.5)/T_vpoi_G14
        struct[0].f[24,0] = 1 - (H_G10*omega_v_G10 + H_G14*omega_v_G14)/(H_G10 + H_G14)
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = L_R00R01*i_l_R00R01_Q*omega - R_R00R01*i_l_R00R01_D + v_R00_D - v_R01_D
        struct[0].g[1,0] = -L_R00R01*i_l_R00R01_D*omega - R_R00R01*i_l_R00R01_Q + v_R00_Q - v_R01_Q
        struct[0].g[2,0] = L_R02R01*i_l_R02R01_Q*omega - R_R02R01*i_l_R02R01_D - v_R01_D + v_R02_D
        struct[0].g[3,0] = -L_R02R01*i_l_R02R01_D*omega - R_R02R01*i_l_R02R01_Q - v_R01_Q + v_R02_Q
        struct[0].g[4,0] = L_R02R03*i_l_R02R03_Q*omega - R_R02R03*i_l_R02R03_D + v_R02_D - v_R03_D
        struct[0].g[5,0] = -L_R02R03*i_l_R02R03_D*omega - R_R02R03*i_l_R02R03_Q + v_R02_Q - v_R03_Q
        struct[0].g[6,0] = L_R03R04*i_l_R03R04_Q*omega - R_R03R04*i_l_R03R04_D + v_R03_D - v_R04_D
        struct[0].g[7,0] = -L_R03R04*i_l_R03R04_D*omega - R_R03R04*i_l_R03R04_Q + v_R03_Q - v_R04_Q
        struct[0].g[8,0] = L_R04R05*i_l_R04R05_Q*omega - R_R04R05*i_l_R04R05_D + v_R04_D - v_R05_D
        struct[0].g[9,0] = -L_R04R05*i_l_R04R05_D*omega - R_R04R05*i_l_R04R05_Q + v_R04_Q - v_R05_Q
        struct[0].g[10,0] = L_R04R12*i_l_R04R12_Q*omega - R_R04R12*i_l_R04R12_D + v_R04_D - v_R12_D
        struct[0].g[11,0] = -L_R04R12*i_l_R04R12_D*omega - R_R04R12*i_l_R04R12_Q + v_R04_Q - v_R12_Q
        struct[0].g[12,0] = L_R05R06*i_l_R05R06_Q*omega - R_R05R06*i_l_R05R06_D + v_R05_D - v_R06_D
        struct[0].g[13,0] = -L_R05R06*i_l_R05R06_D*omega - R_R05R06*i_l_R05R06_Q + v_R05_Q - v_R06_Q
        struct[0].g[14,0] = L_R06R07*i_l_R06R07_Q*omega - R_R06R07*i_l_R06R07_D + v_R06_D - v_R07_D
        struct[0].g[15,0] = -L_R06R07*i_l_R06R07_D*omega - R_R06R07*i_l_R06R07_Q + v_R06_Q - v_R07_Q
        struct[0].g[16,0] = L_R07R08*i_l_R07R08_Q*omega - R_R07R08*i_l_R07R08_D + v_R07_D - v_R08_D
        struct[0].g[17,0] = -L_R07R08*i_l_R07R08_D*omega - R_R07R08*i_l_R07R08_Q + v_R07_Q - v_R08_Q
        struct[0].g[18,0] = L_R08R09*i_l_R08R09_Q*omega - R_R08R09*i_l_R08R09_D + v_R08_D - v_R09_D
        struct[0].g[19,0] = -L_R08R09*i_l_R08R09_D*omega - R_R08R09*i_l_R08R09_Q + v_R08_Q - v_R09_Q
        struct[0].g[20,0] = L_R09R10*i_l_R09R10_Q*omega - R_R09R10*i_l_R09R10_D + v_R09_D - v_R10_D
        struct[0].g[21,0] = -L_R09R10*i_l_R09R10_D*omega - R_R09R10*i_l_R09R10_Q + v_R09_Q - v_R10_Q
        struct[0].g[22,0] = L_R09R17*i_l_R09R17_Q*omega - R_R09R17*i_l_R09R17_D + v_R09_D - v_R17_D
        struct[0].g[23,0] = -L_R09R17*i_l_R09R17_D*omega - R_R09R17*i_l_R09R17_Q + v_R09_Q - v_R17_Q
        struct[0].g[24,0] = L_R11R03*i_l_R11R03_Q*omega - R_R11R03*i_l_R11R03_D - v_R03_D + v_R11_D
        struct[0].g[25,0] = -L_R11R03*i_l_R11R03_D*omega - R_R11R03*i_l_R11R03_Q - v_R03_Q + v_R11_Q
        struct[0].g[26,0] = L_R12R13*i_l_R12R13_Q*omega - R_R12R13*i_l_R12R13_D + v_R12_D - v_R13_D
        struct[0].g[27,0] = -L_R12R13*i_l_R12R13_D*omega - R_R12R13*i_l_R12R13_Q + v_R12_Q - v_R13_Q
        struct[0].g[28,0] = L_R13R14*i_l_R13R14_Q*omega - R_R13R14*i_l_R13R14_D + v_R13_D - v_R14_D
        struct[0].g[29,0] = -L_R13R14*i_l_R13R14_D*omega - R_R13R14*i_l_R13R14_Q + v_R13_Q - v_R14_Q
        struct[0].g[30,0] = L_R14R15*i_l_R14R15_Q*omega - R_R14R15*i_l_R14R15_D + v_R14_D - v_R15_D
        struct[0].g[31,0] = -L_R14R15*i_l_R14R15_D*omega - R_R14R15*i_l_R14R15_Q + v_R14_Q - v_R15_Q
        struct[0].g[32,0] = L_R16R06*i_l_R16R06_Q*omega - R_R16R06*i_l_R16R06_D - v_R06_D + v_R16_D
        struct[0].g[33,0] = -L_R16R06*i_l_R16R06_D*omega - R_R16R06*i_l_R16R06_Q - v_R06_Q + v_R16_Q
        struct[0].g[34,0] = L_R18R10*i_l_R18R10_Q*omega - R_R18R10*i_l_R18R10_D - v_R10_D + v_R18_D
        struct[0].g[35,0] = -L_R18R10*i_l_R18R10_D*omega - R_R18R10*i_l_R18R10_Q - v_R10_Q + v_R18_Q
        struct[0].g[36,0] = C_R00R01*omega*v_R00_Q/2 + i_R00_D - i_l_R00R01_D
        struct[0].g[37,0] = -C_R00R01*omega*v_R00_D/2 + i_R00_Q - i_l_R00R01_Q
        struct[0].g[38,0] = i_R01_D + i_l_R00R01_D + i_l_R02R01_D + omega*v_R01_Q*(C_R00R01/2 + C_R02R01/2)
        struct[0].g[39,0] = i_R01_Q + i_l_R00R01_Q + i_l_R02R01_Q - omega*v_R01_D*(C_R00R01/2 + C_R02R01/2)
        struct[0].g[40,0] = i_R02_D - i_l_R02R01_D - i_l_R02R03_D + omega*v_R02_Q*(C_R02R01/2 + C_R02R03/2)
        struct[0].g[41,0] = i_R02_Q - i_l_R02R01_Q - i_l_R02R03_Q - omega*v_R02_D*(C_R02R01/2 + C_R02R03/2)
        struct[0].g[42,0] = i_R03_D + i_l_R02R03_D - i_l_R03R04_D + i_l_R11R03_D + omega*v_R03_Q*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].g[43,0] = i_R03_Q + i_l_R02R03_Q - i_l_R03R04_Q + i_l_R11R03_Q - omega*v_R03_D*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].g[44,0] = i_R04_D + i_l_R03R04_D - i_l_R04R05_D - i_l_R04R12_D + omega*v_R04_Q*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].g[45,0] = i_R04_Q + i_l_R03R04_Q - i_l_R04R05_Q - i_l_R04R12_Q - omega*v_R04_D*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].g[46,0] = i_R05_D + i_l_R04R05_D - i_l_R05R06_D + omega*v_R05_Q*(C_R04R05/2 + C_R05R06/2)
        struct[0].g[47,0] = i_R05_Q + i_l_R04R05_Q - i_l_R05R06_Q - omega*v_R05_D*(C_R04R05/2 + C_R05R06/2)
        struct[0].g[48,0] = i_R06_D + i_l_R05R06_D - i_l_R06R07_D + i_l_R16R06_D + omega*v_R06_Q*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].g[49,0] = i_R06_Q + i_l_R05R06_Q - i_l_R06R07_Q + i_l_R16R06_Q - omega*v_R06_D*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].g[50,0] = i_R07_D + i_l_R06R07_D - i_l_R07R08_D + omega*v_R07_Q*(C_R06R07/2 + C_R07R08/2)
        struct[0].g[51,0] = i_R07_Q + i_l_R06R07_Q - i_l_R07R08_Q - omega*v_R07_D*(C_R06R07/2 + C_R07R08/2)
        struct[0].g[52,0] = i_R08_D + i_l_R07R08_D - i_l_R08R09_D + omega*v_R08_Q*(C_R07R08/2 + C_R08R09/2)
        struct[0].g[53,0] = i_R08_Q + i_l_R07R08_Q - i_l_R08R09_Q - omega*v_R08_D*(C_R07R08/2 + C_R08R09/2)
        struct[0].g[54,0] = i_R09_D + i_l_R08R09_D - i_l_R09R10_D - i_l_R09R17_D + omega*v_R09_Q*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].g[55,0] = i_R09_Q + i_l_R08R09_Q - i_l_R09R10_Q - i_l_R09R17_Q - omega*v_R09_D*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].g[56,0] = i_R10_D + i_l_R09R10_D + i_l_R18R10_D + omega*v_R10_Q*(C_R09R10/2 + C_R18R10/2)
        struct[0].g[57,0] = i_R10_Q + i_l_R09R10_Q + i_l_R18R10_Q - omega*v_R10_D*(C_R09R10/2 + C_R18R10/2)
        struct[0].g[58,0] = C_R11R03*omega*v_R11_Q/2 + i_R11_D - i_l_R11R03_D
        struct[0].g[59,0] = -C_R11R03*omega*v_R11_D/2 + i_R11_Q - i_l_R11R03_Q
        struct[0].g[60,0] = i_R12_D + i_l_R04R12_D - i_l_R12R13_D + omega*v_R12_Q*(C_R04R12/2 + C_R12R13/2)
        struct[0].g[61,0] = i_R12_Q + i_l_R04R12_Q - i_l_R12R13_Q - omega*v_R12_D*(C_R04R12/2 + C_R12R13/2)
        struct[0].g[62,0] = i_R13_D + i_l_R12R13_D - i_l_R13R14_D + omega*v_R13_Q*(C_R12R13/2 + C_R13R14/2)
        struct[0].g[63,0] = i_R13_Q + i_l_R12R13_Q - i_l_R13R14_Q - omega*v_R13_D*(C_R12R13/2 + C_R13R14/2)
        struct[0].g[64,0] = i_R14_D + i_l_R13R14_D - i_l_R14R15_D + omega*v_R14_Q*(C_R13R14/2 + C_R14R15/2)
        struct[0].g[65,0] = i_R14_Q + i_l_R13R14_Q - i_l_R14R15_Q - omega*v_R14_D*(C_R13R14/2 + C_R14R15/2)
        struct[0].g[66,0] = C_R14R15*omega*v_R15_Q/2 + i_R15_D + i_l_R14R15_D
        struct[0].g[67,0] = -C_R14R15*omega*v_R15_D/2 + i_R15_Q + i_l_R14R15_Q
        struct[0].g[68,0] = C_R16R06*omega*v_R16_Q/2 + i_R16_D - i_l_R16R06_D
        struct[0].g[69,0] = -C_R16R06*omega*v_R16_D/2 + i_R16_Q - i_l_R16R06_Q
        struct[0].g[70,0] = C_R09R17*omega*v_R17_Q/2 + i_R17_D + i_l_R09R17_D
        struct[0].g[71,0] = -C_R09R17*omega*v_R17_D/2 + i_R17_Q + i_l_R09R17_Q
        struct[0].g[72,0] = C_R18R10*omega*v_R18_Q/2 + i_R18_D - i_l_R18R10_D
        struct[0].g[73,0] = -C_R18R10*omega*v_R18_D/2 + i_R18_Q - i_l_R18R10_Q
        struct[0].g[74,0] = -i_R01_d_ref + (-0.666666666666667*p_R01_ref*v_R01_D - 0.666666666666667*q_R01_ref*v_R01_Q)*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)]))
        struct[0].g[75,0] = -i_R01_q_ref - (0.666666666666667*p_R01_ref*v_R01_Q - 0.666666666666667*q_R01_ref*v_R01_D)*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)]))
        struct[0].g[76,0] = -i_R11_d_ref + (-0.666666666666667*p_R11_ref*v_R11_D - 0.666666666666667*q_R11_ref*v_R11_Q)*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)]))
        struct[0].g[77,0] = -i_R11_q_ref - (0.666666666666667*p_R11_ref*v_R11_Q - 0.666666666666667*q_R11_ref*v_R11_D)*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)]))
        struct[0].g[78,0] = -i_R15_d_ref + (-0.666666666666667*p_R15_ref*v_R15_D - 0.666666666666667*q_R15_ref*v_R15_Q)*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)]))
        struct[0].g[79,0] = -i_R15_q_ref - (0.666666666666667*p_R15_ref*v_R15_Q - 0.666666666666667*q_R15_ref*v_R15_D)*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)]))
        struct[0].g[80,0] = -i_R16_d_ref + (-0.666666666666667*p_R16_ref*v_R16_D - 0.666666666666667*q_R16_ref*v_R16_Q)*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)]))
        struct[0].g[81,0] = -i_R16_q_ref - (0.666666666666667*p_R16_ref*v_R16_Q - 0.666666666666667*q_R16_ref*v_R16_D)*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)]))
        struct[0].g[82,0] = -i_R17_d_ref + (-0.666666666666667*p_R17_ref*v_R17_D - 0.666666666666667*q_R17_ref*v_R17_Q)*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)]))
        struct[0].g[83,0] = -i_R17_q_ref - (0.666666666666667*p_R17_ref*v_R17_Q - 0.666666666666667*q_R17_ref*v_R17_D)*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)]))
        struct[0].g[84,0] = -i_R18_d_ref + (-0.666666666666667*p_R18_ref*v_R18_D - 0.666666666666667*q_R18_ref*v_R18_Q)*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)]))
        struct[0].g[85,0] = -i_R18_q_ref - (0.666666666666667*p_R18_ref*v_R18_Q - 0.666666666666667*q_R18_ref*v_R18_D)*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)]))
        struct[0].g[86,0] = (L_t_G10*i_tQ_G10*omega_G10 - R_t_G10*i_tD_G10 + eta_D_G10*v_dc_G10/2 - v_mD_G10)/L_t_G10
        struct[0].g[87,0] = (-L_t_G10*i_tD_G10*omega_G10 - R_t_G10*i_tQ_G10 + eta_Q_G10*v_dc_G10/2 - v_mQ_G10)/L_t_G10
        struct[0].g[88,0] = (C_m_G10*omega_G10*v_mQ_G10 - G_d_G10*v_mD_G10 - i_sD_G10 + i_tD_G10)/C_m_G10
        struct[0].g[89,0] = (-C_m_G10*omega_G10*v_mD_G10 - G_d_G10*v_mQ_G10 - i_sQ_G10 + i_tQ_G10)/C_m_G10
        struct[0].g[90,0] = (L_s_G10*i_sQ_G10*omega_G10 - R_s_G10*i_sD_G10 + v_mD_G10 - v_sD_G10)/L_s_G10
        struct[0].g[91,0] = (-L_s_G10*i_sD_G10*omega_G10 - R_s_G10*i_sQ_G10 + v_mQ_G10 - v_sQ_G10)/L_s_G10
        struct[0].g[92,0] = -i_R10_D + i_sD_G10
        struct[0].g[93,0] = -i_R10_Q + i_sQ_G10
        struct[0].g[94,0] = -v_R10_D + v_sD_G10
        struct[0].g[95,0] = -v_R10_Q + v_sQ_G10
        struct[0].g[96,0] = eta_d_G10 - 2*(-0.8*R_v_G10*i_sd_G10 + 0.8*X_v_G10*i_sq_G10)/v_dc_G10
        struct[0].g[97,0] = eta_q_G10 - 2*(326.59863237109*DV_sat_G10 - 0.8*R_v_G10*i_sq_G10 - 0.8*X_v_G10*i_sd_G10 + 326.59863237109)/v_dc_G10
        struct[0].g[98,0] = -eta_D_G10 + eta_d_G10*cos(phi_G10) - eta_q_G10*sin(phi_G10)
        struct[0].g[99,0] = -eta_Q_G10 + eta_d_G10*sin(phi_G10) + eta_q_G10*cos(phi_G10)
        struct[0].g[100,0] = v_mD_G10*cos(phi_G10) + v_mQ_G10*sin(phi_G10) - v_md_G10
        struct[0].g[101,0] = -v_mD_G10*sin(phi_G10) + v_mQ_G10*cos(phi_G10) - v_mq_G10
        struct[0].g[102,0] = v_sD_G10*cos(phi_G10) + v_sQ_G10*sin(phi_G10) - v_sd_G10
        struct[0].g[103,0] = -v_sD_G10*sin(phi_G10) + v_sQ_G10*cos(phi_G10) - v_sq_G10
        struct[0].g[104,0] = i_tD_G10*cos(phi_G10) + i_tQ_G10*sin(phi_G10) - i_td_G10
        struct[0].g[105,0] = -i_tD_G10*sin(phi_G10) + i_tQ_G10*cos(phi_G10) - i_tq_G10
        struct[0].g[106,0] = i_sD_G10*cos(phi_G10) + i_sQ_G10*sin(phi_G10) - i_sd_G10
        struct[0].g[107,0] = -i_sD_G10*sin(phi_G10) + i_sQ_G10*cos(phi_G10) - i_sq_G10
        struct[0].g[108,0] = DV_sat_G10 - K_q_G10*(-q_s_pu_G10 + q_s_ref_G10 + xi_q_G10/T_q_G10)
        struct[0].g[109,0] = 7.5e-6*i_sd_G10*v_sd_G10 + 7.5e-6*i_sq_G10*v_sq_G10 - p_s_pu_G10
        struct[0].g[110,0] = 7.5e-6*i_sd_G10*v_sq_G10 - 7.5e-6*i_sq_G10*v_sd_G10 - q_s_pu_G10
        struct[0].g[111,0] = K_f_G10*(omega_ref_G10 - omega_v_filt_G10) + K_f_sec*xi_f_sec/2 - p_m_ref_G10 + p_r_G10
        struct[0].g[112,0] = K_vpoi_G10*(-v_s_filt_G10 + v_s_ref_G10) + q_r_G10 - q_s_ref_G10
        struct[0].g[113,0] = (L_t_G14*i_tQ_G14*omega_G14 - R_t_G14*i_tD_G14 + eta_D_G14*v_dc_G14/2 - v_mD_G14)/L_t_G14
        struct[0].g[114,0] = (-L_t_G14*i_tD_G14*omega_G14 - R_t_G14*i_tQ_G14 + eta_Q_G14*v_dc_G14/2 - v_mQ_G14)/L_t_G14
        struct[0].g[115,0] = (C_m_G14*omega_G14*v_mQ_G14 - G_d_G14*v_mD_G14 - i_sD_G14 + i_tD_G14)/C_m_G14
        struct[0].g[116,0] = (-C_m_G14*omega_G14*v_mD_G14 - G_d_G14*v_mQ_G14 - i_sQ_G14 + i_tQ_G14)/C_m_G14
        struct[0].g[117,0] = (L_s_G14*i_sQ_G14*omega_G14 - R_s_G14*i_sD_G14 + v_mD_G14 - v_sD_G14)/L_s_G14
        struct[0].g[118,0] = (-L_s_G14*i_sD_G14*omega_G14 - R_s_G14*i_sQ_G14 + v_mQ_G14 - v_sQ_G14)/L_s_G14
        struct[0].g[119,0] = -i_R14_D + i_sD_G14
        struct[0].g[120,0] = -i_R14_Q + i_sQ_G14
        struct[0].g[121,0] = -v_R14_D + v_sD_G14
        struct[0].g[122,0] = -v_R14_Q + v_sQ_G14
        struct[0].g[123,0] = eta_d_G14 - 2*(-0.8*R_v_G14*i_sd_G14 + 0.8*X_v_G14*i_sq_G14)/v_dc_G14
        struct[0].g[124,0] = eta_q_G14 - 2*(326.59863237109*DV_sat_G14 - 0.8*R_v_G14*i_sq_G14 - 0.8*X_v_G14*i_sd_G14 + 326.59863237109)/v_dc_G14
        struct[0].g[125,0] = -eta_D_G14 + eta_d_G14*cos(phi_G14) - eta_q_G14*sin(phi_G14)
        struct[0].g[126,0] = -eta_Q_G14 + eta_d_G14*sin(phi_G14) + eta_q_G14*cos(phi_G14)
        struct[0].g[127,0] = v_mD_G14*cos(phi_G14) + v_mQ_G14*sin(phi_G14) - v_md_G14
        struct[0].g[128,0] = -v_mD_G14*sin(phi_G14) + v_mQ_G14*cos(phi_G14) - v_mq_G14
        struct[0].g[129,0] = v_sD_G14*cos(phi_G14) + v_sQ_G14*sin(phi_G14) - v_sd_G14
        struct[0].g[130,0] = -v_sD_G14*sin(phi_G14) + v_sQ_G14*cos(phi_G14) - v_sq_G14
        struct[0].g[131,0] = i_tD_G14*cos(phi_G14) + i_tQ_G14*sin(phi_G14) - i_td_G14
        struct[0].g[132,0] = -i_tD_G14*sin(phi_G14) + i_tQ_G14*cos(phi_G14) - i_tq_G14
        struct[0].g[133,0] = i_sD_G14*cos(phi_G14) + i_sQ_G14*sin(phi_G14) - i_sd_G14
        struct[0].g[134,0] = -i_sD_G14*sin(phi_G14) + i_sQ_G14*cos(phi_G14) - i_sq_G14
        struct[0].g[135,0] = DV_sat_G14 - K_q_G14*(-q_s_pu_G14 + q_s_ref_G14 + xi_q_G14/T_q_G14)
        struct[0].g[136,0] = 7.5e-6*i_sd_G14*v_sd_G14 + 7.5e-6*i_sq_G14*v_sq_G14 - p_s_pu_G14
        struct[0].g[137,0] = 7.5e-6*i_sd_G14*v_sq_G14 - 7.5e-6*i_sq_G14*v_sd_G14 - q_s_pu_G14
        struct[0].g[138,0] = K_f_G14*(omega_ref_G14 - omega_v_filt_G14) + K_f_sec*xi_f_sec/2 - p_m_ref_G14 + p_r_G14
        struct[0].g[139,0] = K_vpoi_G14*(-v_s_filt_G14 + v_s_ref_G14) + q_r_G14 - q_s_ref_G14
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = i_sD_G10
        struct[0].h[1,0] = i_sQ_G10
        struct[0].h[2,0] = i_sD_G14
        struct[0].h[3,0] = i_sQ_G14
    

    if mode == 10:

        struct[0].Fx_ini[0,0] = -100.000000000000
        struct[0].Fx_ini[1,1] = -100.000000000000
        struct[0].Fx_ini[2,2] = -100.000000000000
        struct[0].Fx_ini[3,3] = -100.000000000000
        struct[0].Fx_ini[4,4] = -100.000000000000
        struct[0].Fx_ini[5,5] = -100.000000000000
        struct[0].Fx_ini[6,6] = -100.000000000000
        struct[0].Fx_ini[7,7] = -100.000000000000
        struct[0].Fx_ini[8,8] = -100.000000000000
        struct[0].Fx_ini[9,9] = -100.000000000000
        struct[0].Fx_ini[10,10] = -100.000000000000
        struct[0].Fx_ini[11,11] = -100.000000000000
        struct[0].Fx_ini[12,12] = -K_phi_G10
        struct[0].Fx_ini[12,13] = -314.159265358979*H_G10/(H_G10 + H_G14) + 314.159265358979
        struct[0].Fx_ini[12,19] = -314.159265358979*H_G14/(H_G10 + H_G14)
        struct[0].Fx_ini[13,13] = -D_G10/(2*H_G10)
        struct[0].Fx_ini[15,13] = 314.159265358979
        struct[0].Fx_ini[15,15] = -1.00000000000000
        struct[0].Fx_ini[16,13] = 1/T_f_G10
        struct[0].Fx_ini[16,16] = -1/T_f_G10
        struct[0].Fx_ini[17,17] = -1/T_vpoi_G10
        struct[0].Fx_ini[18,13] = -314.159265358979*H_G10/(H_G10 + H_G14)
        struct[0].Fx_ini[18,18] = -K_phi_G14
        struct[0].Fx_ini[18,19] = -314.159265358979*H_G14/(H_G10 + H_G14) + 314.159265358979
        struct[0].Fx_ini[19,19] = -D_G14/(2*H_G14)
        struct[0].Fx_ini[21,19] = 314.159265358979
        struct[0].Fx_ini[21,21] = -1.00000000000000
        struct[0].Fx_ini[22,19] = 1/T_f_G14
        struct[0].Fx_ini[22,22] = -1/T_f_G14
        struct[0].Fx_ini[23,23] = -1/T_vpoi_G14
        struct[0].Fx_ini[24,13] = -H_G10/(H_G10 + H_G14)
        struct[0].Fx_ini[24,19] = -H_G14/(H_G10 + H_G14)

    if mode == 11:

        struct[0].Fy_ini[0,74] = 100.0*Piecewise(np.array([(0, (I_max_R01 < i_R01_d_ref) | (I_max_R01 < -i_R01_d_ref)), (1, True)])) 
        struct[0].Fy_ini[1,75] = 100.0*Piecewise(np.array([(0, (I_max_R01 < i_R01_q_ref) | (I_max_R01 < -i_R01_q_ref)), (1, True)])) 
        struct[0].Fy_ini[2,76] = 100.0*Piecewise(np.array([(0, (I_max_R11 < i_R11_d_ref) | (I_max_R11 < -i_R11_d_ref)), (1, True)])) 
        struct[0].Fy_ini[3,77] = 100.0*Piecewise(np.array([(0, (I_max_R11 < i_R11_q_ref) | (I_max_R11 < -i_R11_q_ref)), (1, True)])) 
        struct[0].Fy_ini[4,78] = 100.0*Piecewise(np.array([(0, (I_max_R15 < i_R15_d_ref) | (I_max_R15 < -i_R15_d_ref)), (1, True)])) 
        struct[0].Fy_ini[5,79] = 100.0*Piecewise(np.array([(0, (I_max_R15 < i_R15_q_ref) | (I_max_R15 < -i_R15_q_ref)), (1, True)])) 
        struct[0].Fy_ini[6,80] = 100.0*Piecewise(np.array([(0, (I_max_R16 < i_R16_d_ref) | (I_max_R16 < -i_R16_d_ref)), (1, True)])) 
        struct[0].Fy_ini[7,81] = 100.0*Piecewise(np.array([(0, (I_max_R16 < i_R16_q_ref) | (I_max_R16 < -i_R16_q_ref)), (1, True)])) 
        struct[0].Fy_ini[8,82] = 100.0*Piecewise(np.array([(0, (I_max_R17 < i_R17_d_ref) | (I_max_R17 < -i_R17_d_ref)), (1, True)])) 
        struct[0].Fy_ini[9,83] = 100.0*Piecewise(np.array([(0, (I_max_R17 < i_R17_q_ref) | (I_max_R17 < -i_R17_q_ref)), (1, True)])) 
        struct[0].Fy_ini[10,84] = 100.0*Piecewise(np.array([(0, (I_max_R18 < i_R18_d_ref) | (I_max_R18 < -i_R18_d_ref)), (1, True)])) 
        struct[0].Fy_ini[11,85] = 100.0*Piecewise(np.array([(0, (I_max_R18 < i_R18_q_ref) | (I_max_R18 < -i_R18_q_ref)), (1, True)])) 
        struct[0].Fy_ini[13,109] = -1/(2*H_G10) 
        struct[0].Fy_ini[13,111] = 1/(2*H_G10) 
        struct[0].Fy_ini[14,110] = -1 
        struct[0].Fy_ini[14,112] = 1 
        struct[0].Fy_ini[17,102] = 0.00306186217847897*v_sd_G10/(T_vpoi_G10*(v_sd_G10**2 + v_sq_G10**2)**0.5) 
        struct[0].Fy_ini[17,103] = 0.00306186217847897*v_sq_G10/(T_vpoi_G10*(v_sd_G10**2 + v_sq_G10**2)**0.5) 
        struct[0].Fy_ini[19,136] = -1/(2*H_G14) 
        struct[0].Fy_ini[19,138] = 1/(2*H_G14) 
        struct[0].Fy_ini[20,137] = -1 
        struct[0].Fy_ini[20,139] = 1 
        struct[0].Fy_ini[23,129] = 0.00306186217847897*v_sd_G14/(T_vpoi_G14*(v_sd_G14**2 + v_sq_G14**2)**0.5) 
        struct[0].Fy_ini[23,130] = 0.00306186217847897*v_sq_G14/(T_vpoi_G14*(v_sd_G14**2 + v_sq_G14**2)**0.5) 

        struct[0].Gy_ini[0,0] = -R_R00R01
        struct[0].Gy_ini[0,1] = L_R00R01*omega
        struct[0].Gy_ini[0,36] = 1
        struct[0].Gy_ini[0,38] = -1
        struct[0].Gy_ini[1,0] = -L_R00R01*omega
        struct[0].Gy_ini[1,1] = -R_R00R01
        struct[0].Gy_ini[1,37] = 1
        struct[0].Gy_ini[1,39] = -1
        struct[0].Gy_ini[2,2] = -R_R02R01
        struct[0].Gy_ini[2,3] = L_R02R01*omega
        struct[0].Gy_ini[2,38] = -1
        struct[0].Gy_ini[2,40] = 1
        struct[0].Gy_ini[3,2] = -L_R02R01*omega
        struct[0].Gy_ini[3,3] = -R_R02R01
        struct[0].Gy_ini[3,39] = -1
        struct[0].Gy_ini[3,41] = 1
        struct[0].Gy_ini[4,4] = -R_R02R03
        struct[0].Gy_ini[4,5] = L_R02R03*omega
        struct[0].Gy_ini[4,40] = 1
        struct[0].Gy_ini[4,42] = -1
        struct[0].Gy_ini[5,4] = -L_R02R03*omega
        struct[0].Gy_ini[5,5] = -R_R02R03
        struct[0].Gy_ini[5,41] = 1
        struct[0].Gy_ini[5,43] = -1
        struct[0].Gy_ini[6,6] = -R_R03R04
        struct[0].Gy_ini[6,7] = L_R03R04*omega
        struct[0].Gy_ini[6,42] = 1
        struct[0].Gy_ini[6,44] = -1
        struct[0].Gy_ini[7,6] = -L_R03R04*omega
        struct[0].Gy_ini[7,7] = -R_R03R04
        struct[0].Gy_ini[7,43] = 1
        struct[0].Gy_ini[7,45] = -1
        struct[0].Gy_ini[8,8] = -R_R04R05
        struct[0].Gy_ini[8,9] = L_R04R05*omega
        struct[0].Gy_ini[8,44] = 1
        struct[0].Gy_ini[8,46] = -1
        struct[0].Gy_ini[9,8] = -L_R04R05*omega
        struct[0].Gy_ini[9,9] = -R_R04R05
        struct[0].Gy_ini[9,45] = 1
        struct[0].Gy_ini[9,47] = -1
        struct[0].Gy_ini[10,10] = -R_R04R12
        struct[0].Gy_ini[10,11] = L_R04R12*omega
        struct[0].Gy_ini[10,44] = 1
        struct[0].Gy_ini[10,60] = -1
        struct[0].Gy_ini[11,10] = -L_R04R12*omega
        struct[0].Gy_ini[11,11] = -R_R04R12
        struct[0].Gy_ini[11,45] = 1
        struct[0].Gy_ini[11,61] = -1
        struct[0].Gy_ini[12,12] = -R_R05R06
        struct[0].Gy_ini[12,13] = L_R05R06*omega
        struct[0].Gy_ini[12,46] = 1
        struct[0].Gy_ini[12,48] = -1
        struct[0].Gy_ini[13,12] = -L_R05R06*omega
        struct[0].Gy_ini[13,13] = -R_R05R06
        struct[0].Gy_ini[13,47] = 1
        struct[0].Gy_ini[13,49] = -1
        struct[0].Gy_ini[14,14] = -R_R06R07
        struct[0].Gy_ini[14,15] = L_R06R07*omega
        struct[0].Gy_ini[14,48] = 1
        struct[0].Gy_ini[14,50] = -1
        struct[0].Gy_ini[15,14] = -L_R06R07*omega
        struct[0].Gy_ini[15,15] = -R_R06R07
        struct[0].Gy_ini[15,49] = 1
        struct[0].Gy_ini[15,51] = -1
        struct[0].Gy_ini[16,16] = -R_R07R08
        struct[0].Gy_ini[16,17] = L_R07R08*omega
        struct[0].Gy_ini[16,50] = 1
        struct[0].Gy_ini[16,52] = -1
        struct[0].Gy_ini[17,16] = -L_R07R08*omega
        struct[0].Gy_ini[17,17] = -R_R07R08
        struct[0].Gy_ini[17,51] = 1
        struct[0].Gy_ini[17,53] = -1
        struct[0].Gy_ini[18,18] = -R_R08R09
        struct[0].Gy_ini[18,19] = L_R08R09*omega
        struct[0].Gy_ini[18,52] = 1
        struct[0].Gy_ini[18,54] = -1
        struct[0].Gy_ini[19,18] = -L_R08R09*omega
        struct[0].Gy_ini[19,19] = -R_R08R09
        struct[0].Gy_ini[19,53] = 1
        struct[0].Gy_ini[19,55] = -1
        struct[0].Gy_ini[20,20] = -R_R09R10
        struct[0].Gy_ini[20,21] = L_R09R10*omega
        struct[0].Gy_ini[20,54] = 1
        struct[0].Gy_ini[20,56] = -1
        struct[0].Gy_ini[21,20] = -L_R09R10*omega
        struct[0].Gy_ini[21,21] = -R_R09R10
        struct[0].Gy_ini[21,55] = 1
        struct[0].Gy_ini[21,57] = -1
        struct[0].Gy_ini[22,22] = -R_R09R17
        struct[0].Gy_ini[22,23] = L_R09R17*omega
        struct[0].Gy_ini[22,54] = 1
        struct[0].Gy_ini[22,70] = -1
        struct[0].Gy_ini[23,22] = -L_R09R17*omega
        struct[0].Gy_ini[23,23] = -R_R09R17
        struct[0].Gy_ini[23,55] = 1
        struct[0].Gy_ini[23,71] = -1
        struct[0].Gy_ini[24,24] = -R_R11R03
        struct[0].Gy_ini[24,25] = L_R11R03*omega
        struct[0].Gy_ini[24,42] = -1
        struct[0].Gy_ini[24,58] = 1
        struct[0].Gy_ini[25,24] = -L_R11R03*omega
        struct[0].Gy_ini[25,25] = -R_R11R03
        struct[0].Gy_ini[25,43] = -1
        struct[0].Gy_ini[25,59] = 1
        struct[0].Gy_ini[26,26] = -R_R12R13
        struct[0].Gy_ini[26,27] = L_R12R13*omega
        struct[0].Gy_ini[26,60] = 1
        struct[0].Gy_ini[26,62] = -1
        struct[0].Gy_ini[27,26] = -L_R12R13*omega
        struct[0].Gy_ini[27,27] = -R_R12R13
        struct[0].Gy_ini[27,61] = 1
        struct[0].Gy_ini[27,63] = -1
        struct[0].Gy_ini[28,28] = -R_R13R14
        struct[0].Gy_ini[28,29] = L_R13R14*omega
        struct[0].Gy_ini[28,62] = 1
        struct[0].Gy_ini[28,64] = -1
        struct[0].Gy_ini[29,28] = -L_R13R14*omega
        struct[0].Gy_ini[29,29] = -R_R13R14
        struct[0].Gy_ini[29,63] = 1
        struct[0].Gy_ini[29,65] = -1
        struct[0].Gy_ini[30,30] = -R_R14R15
        struct[0].Gy_ini[30,31] = L_R14R15*omega
        struct[0].Gy_ini[30,64] = 1
        struct[0].Gy_ini[30,66] = -1
        struct[0].Gy_ini[31,30] = -L_R14R15*omega
        struct[0].Gy_ini[31,31] = -R_R14R15
        struct[0].Gy_ini[31,65] = 1
        struct[0].Gy_ini[31,67] = -1
        struct[0].Gy_ini[32,32] = -R_R16R06
        struct[0].Gy_ini[32,33] = L_R16R06*omega
        struct[0].Gy_ini[32,48] = -1
        struct[0].Gy_ini[32,68] = 1
        struct[0].Gy_ini[33,32] = -L_R16R06*omega
        struct[0].Gy_ini[33,33] = -R_R16R06
        struct[0].Gy_ini[33,49] = -1
        struct[0].Gy_ini[33,69] = 1
        struct[0].Gy_ini[34,34] = -R_R18R10
        struct[0].Gy_ini[34,35] = L_R18R10*omega
        struct[0].Gy_ini[34,56] = -1
        struct[0].Gy_ini[34,72] = 1
        struct[0].Gy_ini[35,34] = -L_R18R10*omega
        struct[0].Gy_ini[35,35] = -R_R18R10
        struct[0].Gy_ini[35,57] = -1
        struct[0].Gy_ini[35,73] = 1
        struct[0].Gy_ini[36,0] = -1
        struct[0].Gy_ini[36,37] = C_R00R01*omega/2
        struct[0].Gy_ini[37,1] = -1
        struct[0].Gy_ini[37,36] = -C_R00R01*omega/2
        struct[0].Gy_ini[38,0] = 1
        struct[0].Gy_ini[38,2] = 1
        struct[0].Gy_ini[38,39] = omega*(C_R00R01/2 + C_R02R01/2)
        struct[0].Gy_ini[39,1] = 1
        struct[0].Gy_ini[39,3] = 1
        struct[0].Gy_ini[39,38] = -omega*(C_R00R01/2 + C_R02R01/2)
        struct[0].Gy_ini[40,2] = -1
        struct[0].Gy_ini[40,4] = -1
        struct[0].Gy_ini[40,41] = omega*(C_R02R01/2 + C_R02R03/2)
        struct[0].Gy_ini[41,3] = -1
        struct[0].Gy_ini[41,5] = -1
        struct[0].Gy_ini[41,40] = -omega*(C_R02R01/2 + C_R02R03/2)
        struct[0].Gy_ini[42,4] = 1
        struct[0].Gy_ini[42,6] = -1
        struct[0].Gy_ini[42,24] = 1
        struct[0].Gy_ini[42,43] = omega*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].Gy_ini[43,5] = 1
        struct[0].Gy_ini[43,7] = -1
        struct[0].Gy_ini[43,25] = 1
        struct[0].Gy_ini[43,42] = -omega*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].Gy_ini[44,6] = 1
        struct[0].Gy_ini[44,8] = -1
        struct[0].Gy_ini[44,10] = -1
        struct[0].Gy_ini[44,45] = omega*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].Gy_ini[45,7] = 1
        struct[0].Gy_ini[45,9] = -1
        struct[0].Gy_ini[45,11] = -1
        struct[0].Gy_ini[45,44] = -omega*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].Gy_ini[46,8] = 1
        struct[0].Gy_ini[46,12] = -1
        struct[0].Gy_ini[46,47] = omega*(C_R04R05/2 + C_R05R06/2)
        struct[0].Gy_ini[47,9] = 1
        struct[0].Gy_ini[47,13] = -1
        struct[0].Gy_ini[47,46] = -omega*(C_R04R05/2 + C_R05R06/2)
        struct[0].Gy_ini[48,12] = 1
        struct[0].Gy_ini[48,14] = -1
        struct[0].Gy_ini[48,32] = 1
        struct[0].Gy_ini[48,49] = omega*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].Gy_ini[49,13] = 1
        struct[0].Gy_ini[49,15] = -1
        struct[0].Gy_ini[49,33] = 1
        struct[0].Gy_ini[49,48] = -omega*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].Gy_ini[50,14] = 1
        struct[0].Gy_ini[50,16] = -1
        struct[0].Gy_ini[50,51] = omega*(C_R06R07/2 + C_R07R08/2)
        struct[0].Gy_ini[51,15] = 1
        struct[0].Gy_ini[51,17] = -1
        struct[0].Gy_ini[51,50] = -omega*(C_R06R07/2 + C_R07R08/2)
        struct[0].Gy_ini[52,16] = 1
        struct[0].Gy_ini[52,18] = -1
        struct[0].Gy_ini[52,53] = omega*(C_R07R08/2 + C_R08R09/2)
        struct[0].Gy_ini[53,17] = 1
        struct[0].Gy_ini[53,19] = -1
        struct[0].Gy_ini[53,52] = -omega*(C_R07R08/2 + C_R08R09/2)
        struct[0].Gy_ini[54,18] = 1
        struct[0].Gy_ini[54,20] = -1
        struct[0].Gy_ini[54,22] = -1
        struct[0].Gy_ini[54,55] = omega*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].Gy_ini[55,19] = 1
        struct[0].Gy_ini[55,21] = -1
        struct[0].Gy_ini[55,23] = -1
        struct[0].Gy_ini[55,54] = -omega*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].Gy_ini[56,20] = 1
        struct[0].Gy_ini[56,34] = 1
        struct[0].Gy_ini[56,57] = omega*(C_R09R10/2 + C_R18R10/2)
        struct[0].Gy_ini[56,92] = 1
        struct[0].Gy_ini[57,21] = 1
        struct[0].Gy_ini[57,35] = 1
        struct[0].Gy_ini[57,56] = -omega*(C_R09R10/2 + C_R18R10/2)
        struct[0].Gy_ini[57,93] = 1
        struct[0].Gy_ini[58,24] = -1
        struct[0].Gy_ini[58,59] = C_R11R03*omega/2
        struct[0].Gy_ini[59,25] = -1
        struct[0].Gy_ini[59,58] = -C_R11R03*omega/2
        struct[0].Gy_ini[60,10] = 1
        struct[0].Gy_ini[60,26] = -1
        struct[0].Gy_ini[60,61] = omega*(C_R04R12/2 + C_R12R13/2)
        struct[0].Gy_ini[61,11] = 1
        struct[0].Gy_ini[61,27] = -1
        struct[0].Gy_ini[61,60] = -omega*(C_R04R12/2 + C_R12R13/2)
        struct[0].Gy_ini[62,26] = 1
        struct[0].Gy_ini[62,28] = -1
        struct[0].Gy_ini[62,63] = omega*(C_R12R13/2 + C_R13R14/2)
        struct[0].Gy_ini[63,27] = 1
        struct[0].Gy_ini[63,29] = -1
        struct[0].Gy_ini[63,62] = -omega*(C_R12R13/2 + C_R13R14/2)
        struct[0].Gy_ini[64,28] = 1
        struct[0].Gy_ini[64,30] = -1
        struct[0].Gy_ini[64,65] = omega*(C_R13R14/2 + C_R14R15/2)
        struct[0].Gy_ini[64,119] = 1
        struct[0].Gy_ini[65,29] = 1
        struct[0].Gy_ini[65,31] = -1
        struct[0].Gy_ini[65,64] = -omega*(C_R13R14/2 + C_R14R15/2)
        struct[0].Gy_ini[65,120] = 1
        struct[0].Gy_ini[66,30] = 1
        struct[0].Gy_ini[66,67] = C_R14R15*omega/2
        struct[0].Gy_ini[67,31] = 1
        struct[0].Gy_ini[67,66] = -C_R14R15*omega/2
        struct[0].Gy_ini[68,32] = -1
        struct[0].Gy_ini[68,69] = C_R16R06*omega/2
        struct[0].Gy_ini[69,33] = -1
        struct[0].Gy_ini[69,68] = -C_R16R06*omega/2
        struct[0].Gy_ini[70,22] = 1
        struct[0].Gy_ini[70,71] = C_R09R17*omega/2
        struct[0].Gy_ini[71,23] = 1
        struct[0].Gy_ini[71,70] = -C_R09R17*omega/2
        struct[0].Gy_ini[72,34] = -1
        struct[0].Gy_ini[72,73] = C_R18R10*omega/2
        struct[0].Gy_ini[73,35] = -1
        struct[0].Gy_ini[73,72] = -C_R18R10*omega/2
        struct[0].Gy_ini[74,38] = -0.666666666666667*p_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_D - 0.666666666666667*q_R01_ref*v_R01_Q)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_D/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy_ini[74,39] = -0.666666666666667*q_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_D - 0.666666666666667*q_R01_ref*v_R01_Q)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_Q/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy_ini[74,74] = -1
        struct[0].Gy_ini[75,38] = 0.666666666666667*q_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_Q + 0.666666666666667*q_R01_ref*v_R01_D)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_D/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy_ini[75,39] = -0.666666666666667*p_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_Q + 0.666666666666667*q_R01_ref*v_R01_D)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_Q/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy_ini[75,75] = -1
        struct[0].Gy_ini[76,58] = -0.666666666666667*p_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_D - 0.666666666666667*q_R11_ref*v_R11_Q)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_D/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy_ini[76,59] = -0.666666666666667*q_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_D - 0.666666666666667*q_R11_ref*v_R11_Q)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_Q/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy_ini[76,76] = -1
        struct[0].Gy_ini[77,58] = 0.666666666666667*q_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_Q + 0.666666666666667*q_R11_ref*v_R11_D)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_D/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy_ini[77,59] = -0.666666666666667*p_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_Q + 0.666666666666667*q_R11_ref*v_R11_D)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_Q/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy_ini[77,77] = -1
        struct[0].Gy_ini[78,66] = -0.666666666666667*p_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_D - 0.666666666666667*q_R15_ref*v_R15_Q)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_D/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy_ini[78,67] = -0.666666666666667*q_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_D - 0.666666666666667*q_R15_ref*v_R15_Q)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_Q/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy_ini[78,78] = -1
        struct[0].Gy_ini[79,66] = 0.666666666666667*q_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_Q + 0.666666666666667*q_R15_ref*v_R15_D)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_D/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy_ini[79,67] = -0.666666666666667*p_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_Q + 0.666666666666667*q_R15_ref*v_R15_D)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_Q/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy_ini[79,79] = -1
        struct[0].Gy_ini[80,68] = -0.666666666666667*p_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_D - 0.666666666666667*q_R16_ref*v_R16_Q)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_D/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy_ini[80,69] = -0.666666666666667*q_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_D - 0.666666666666667*q_R16_ref*v_R16_Q)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_Q/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy_ini[80,80] = -1
        struct[0].Gy_ini[81,68] = 0.666666666666667*q_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_Q + 0.666666666666667*q_R16_ref*v_R16_D)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_D/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy_ini[81,69] = -0.666666666666667*p_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_Q + 0.666666666666667*q_R16_ref*v_R16_D)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_Q/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy_ini[81,81] = -1
        struct[0].Gy_ini[82,70] = -0.666666666666667*p_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_D - 0.666666666666667*q_R17_ref*v_R17_Q)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_D/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy_ini[82,71] = -0.666666666666667*q_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_D - 0.666666666666667*q_R17_ref*v_R17_Q)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_Q/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy_ini[82,82] = -1
        struct[0].Gy_ini[83,70] = 0.666666666666667*q_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_Q + 0.666666666666667*q_R17_ref*v_R17_D)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_D/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy_ini[83,71] = -0.666666666666667*p_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_Q + 0.666666666666667*q_R17_ref*v_R17_D)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_Q/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy_ini[83,83] = -1
        struct[0].Gy_ini[84,72] = -0.666666666666667*p_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_D - 0.666666666666667*q_R18_ref*v_R18_Q)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_D/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy_ini[84,73] = -0.666666666666667*q_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_D - 0.666666666666667*q_R18_ref*v_R18_Q)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_Q/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy_ini[84,84] = -1
        struct[0].Gy_ini[85,72] = 0.666666666666667*q_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_Q + 0.666666666666667*q_R18_ref*v_R18_D)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_D/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy_ini[85,73] = -0.666666666666667*p_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_Q + 0.666666666666667*q_R18_ref*v_R18_D)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_Q/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy_ini[85,85] = -1
        struct[0].Gy_ini[86,86] = -R_t_G10/L_t_G10
        struct[0].Gy_ini[86,87] = omega_G10
        struct[0].Gy_ini[86,88] = -1/L_t_G10
        struct[0].Gy_ini[86,98] = v_dc_G10/(2*L_t_G10)
        struct[0].Gy_ini[87,86] = -omega_G10
        struct[0].Gy_ini[87,87] = -R_t_G10/L_t_G10
        struct[0].Gy_ini[87,89] = -1/L_t_G10
        struct[0].Gy_ini[87,99] = v_dc_G10/(2*L_t_G10)
        struct[0].Gy_ini[88,86] = 1/C_m_G10
        struct[0].Gy_ini[88,88] = -G_d_G10/C_m_G10
        struct[0].Gy_ini[88,89] = omega_G10
        struct[0].Gy_ini[88,90] = -1/C_m_G10
        struct[0].Gy_ini[89,87] = 1/C_m_G10
        struct[0].Gy_ini[89,88] = -omega_G10
        struct[0].Gy_ini[89,89] = -G_d_G10/C_m_G10
        struct[0].Gy_ini[89,91] = -1/C_m_G10
        struct[0].Gy_ini[90,88] = 1/L_s_G10
        struct[0].Gy_ini[90,90] = -R_s_G10/L_s_G10
        struct[0].Gy_ini[90,91] = omega_G10
        struct[0].Gy_ini[90,94] = -1/L_s_G10
        struct[0].Gy_ini[91,89] = 1/L_s_G10
        struct[0].Gy_ini[91,90] = -omega_G10
        struct[0].Gy_ini[91,91] = -R_s_G10/L_s_G10
        struct[0].Gy_ini[91,95] = -1/L_s_G10
        struct[0].Gy_ini[92,90] = 1
        struct[0].Gy_ini[92,92] = -1
        struct[0].Gy_ini[93,91] = 1
        struct[0].Gy_ini[93,93] = -1
        struct[0].Gy_ini[94,56] = -1
        struct[0].Gy_ini[94,94] = 1
        struct[0].Gy_ini[95,57] = -1
        struct[0].Gy_ini[95,95] = 1
        struct[0].Gy_ini[96,96] = 1
        struct[0].Gy_ini[96,106] = 1.6*R_v_G10/v_dc_G10
        struct[0].Gy_ini[96,107] = -1.6*X_v_G10/v_dc_G10
        struct[0].Gy_ini[97,97] = 1
        struct[0].Gy_ini[97,106] = 1.6*X_v_G10/v_dc_G10
        struct[0].Gy_ini[97,107] = 1.6*R_v_G10/v_dc_G10
        struct[0].Gy_ini[97,108] = -653.197264742181/v_dc_G10
        struct[0].Gy_ini[98,96] = cos(phi_G10)
        struct[0].Gy_ini[98,97] = -sin(phi_G10)
        struct[0].Gy_ini[98,98] = -1
        struct[0].Gy_ini[99,96] = sin(phi_G10)
        struct[0].Gy_ini[99,97] = cos(phi_G10)
        struct[0].Gy_ini[99,99] = -1
        struct[0].Gy_ini[100,88] = cos(phi_G10)
        struct[0].Gy_ini[100,89] = sin(phi_G10)
        struct[0].Gy_ini[100,100] = -1
        struct[0].Gy_ini[101,88] = -sin(phi_G10)
        struct[0].Gy_ini[101,89] = cos(phi_G10)
        struct[0].Gy_ini[101,101] = -1
        struct[0].Gy_ini[102,94] = cos(phi_G10)
        struct[0].Gy_ini[102,95] = sin(phi_G10)
        struct[0].Gy_ini[102,102] = -1
        struct[0].Gy_ini[103,94] = -sin(phi_G10)
        struct[0].Gy_ini[103,95] = cos(phi_G10)
        struct[0].Gy_ini[103,103] = -1
        struct[0].Gy_ini[104,86] = cos(phi_G10)
        struct[0].Gy_ini[104,87] = sin(phi_G10)
        struct[0].Gy_ini[104,104] = -1
        struct[0].Gy_ini[105,86] = -sin(phi_G10)
        struct[0].Gy_ini[105,87] = cos(phi_G10)
        struct[0].Gy_ini[105,105] = -1
        struct[0].Gy_ini[106,90] = cos(phi_G10)
        struct[0].Gy_ini[106,91] = sin(phi_G10)
        struct[0].Gy_ini[106,106] = -1
        struct[0].Gy_ini[107,90] = -sin(phi_G10)
        struct[0].Gy_ini[107,91] = cos(phi_G10)
        struct[0].Gy_ini[107,107] = -1
        struct[0].Gy_ini[108,108] = 1
        struct[0].Gy_ini[108,110] = K_q_G10
        struct[0].Gy_ini[108,112] = -K_q_G10
        struct[0].Gy_ini[109,102] = 7.5e-6*i_sd_G10
        struct[0].Gy_ini[109,103] = 7.5e-6*i_sq_G10
        struct[0].Gy_ini[109,106] = 7.5e-6*v_sd_G10
        struct[0].Gy_ini[109,107] = 7.5e-6*v_sq_G10
        struct[0].Gy_ini[109,109] = -1
        struct[0].Gy_ini[110,102] = -7.5e-6*i_sq_G10
        struct[0].Gy_ini[110,103] = 7.5e-6*i_sd_G10
        struct[0].Gy_ini[110,106] = 7.5e-6*v_sq_G10
        struct[0].Gy_ini[110,107] = -7.5e-6*v_sd_G10
        struct[0].Gy_ini[110,110] = -1
        struct[0].Gy_ini[111,111] = -1
        struct[0].Gy_ini[112,112] = -1
        struct[0].Gy_ini[113,113] = -R_t_G14/L_t_G14
        struct[0].Gy_ini[113,114] = omega_G14
        struct[0].Gy_ini[113,115] = -1/L_t_G14
        struct[0].Gy_ini[113,125] = v_dc_G14/(2*L_t_G14)
        struct[0].Gy_ini[114,113] = -omega_G14
        struct[0].Gy_ini[114,114] = -R_t_G14/L_t_G14
        struct[0].Gy_ini[114,116] = -1/L_t_G14
        struct[0].Gy_ini[114,126] = v_dc_G14/(2*L_t_G14)
        struct[0].Gy_ini[115,113] = 1/C_m_G14
        struct[0].Gy_ini[115,115] = -G_d_G14/C_m_G14
        struct[0].Gy_ini[115,116] = omega_G14
        struct[0].Gy_ini[115,117] = -1/C_m_G14
        struct[0].Gy_ini[116,114] = 1/C_m_G14
        struct[0].Gy_ini[116,115] = -omega_G14
        struct[0].Gy_ini[116,116] = -G_d_G14/C_m_G14
        struct[0].Gy_ini[116,118] = -1/C_m_G14
        struct[0].Gy_ini[117,115] = 1/L_s_G14
        struct[0].Gy_ini[117,117] = -R_s_G14/L_s_G14
        struct[0].Gy_ini[117,118] = omega_G14
        struct[0].Gy_ini[117,121] = -1/L_s_G14
        struct[0].Gy_ini[118,116] = 1/L_s_G14
        struct[0].Gy_ini[118,117] = -omega_G14
        struct[0].Gy_ini[118,118] = -R_s_G14/L_s_G14
        struct[0].Gy_ini[118,122] = -1/L_s_G14
        struct[0].Gy_ini[119,117] = 1
        struct[0].Gy_ini[119,119] = -1
        struct[0].Gy_ini[120,118] = 1
        struct[0].Gy_ini[120,120] = -1
        struct[0].Gy_ini[121,64] = -1
        struct[0].Gy_ini[121,121] = 1
        struct[0].Gy_ini[122,65] = -1
        struct[0].Gy_ini[122,122] = 1
        struct[0].Gy_ini[123,123] = 1
        struct[0].Gy_ini[123,133] = 1.6*R_v_G14/v_dc_G14
        struct[0].Gy_ini[123,134] = -1.6*X_v_G14/v_dc_G14
        struct[0].Gy_ini[124,124] = 1
        struct[0].Gy_ini[124,133] = 1.6*X_v_G14/v_dc_G14
        struct[0].Gy_ini[124,134] = 1.6*R_v_G14/v_dc_G14
        struct[0].Gy_ini[124,135] = -653.197264742181/v_dc_G14
        struct[0].Gy_ini[125,123] = cos(phi_G14)
        struct[0].Gy_ini[125,124] = -sin(phi_G14)
        struct[0].Gy_ini[125,125] = -1
        struct[0].Gy_ini[126,123] = sin(phi_G14)
        struct[0].Gy_ini[126,124] = cos(phi_G14)
        struct[0].Gy_ini[126,126] = -1
        struct[0].Gy_ini[127,115] = cos(phi_G14)
        struct[0].Gy_ini[127,116] = sin(phi_G14)
        struct[0].Gy_ini[127,127] = -1
        struct[0].Gy_ini[128,115] = -sin(phi_G14)
        struct[0].Gy_ini[128,116] = cos(phi_G14)
        struct[0].Gy_ini[128,128] = -1
        struct[0].Gy_ini[129,121] = cos(phi_G14)
        struct[0].Gy_ini[129,122] = sin(phi_G14)
        struct[0].Gy_ini[129,129] = -1
        struct[0].Gy_ini[130,121] = -sin(phi_G14)
        struct[0].Gy_ini[130,122] = cos(phi_G14)
        struct[0].Gy_ini[130,130] = -1
        struct[0].Gy_ini[131,113] = cos(phi_G14)
        struct[0].Gy_ini[131,114] = sin(phi_G14)
        struct[0].Gy_ini[131,131] = -1
        struct[0].Gy_ini[132,113] = -sin(phi_G14)
        struct[0].Gy_ini[132,114] = cos(phi_G14)
        struct[0].Gy_ini[132,132] = -1
        struct[0].Gy_ini[133,117] = cos(phi_G14)
        struct[0].Gy_ini[133,118] = sin(phi_G14)
        struct[0].Gy_ini[133,133] = -1
        struct[0].Gy_ini[134,117] = -sin(phi_G14)
        struct[0].Gy_ini[134,118] = cos(phi_G14)
        struct[0].Gy_ini[134,134] = -1
        struct[0].Gy_ini[135,135] = 1
        struct[0].Gy_ini[135,137] = K_q_G14
        struct[0].Gy_ini[135,139] = -K_q_G14
        struct[0].Gy_ini[136,129] = 7.5e-6*i_sd_G14
        struct[0].Gy_ini[136,130] = 7.5e-6*i_sq_G14
        struct[0].Gy_ini[136,133] = 7.5e-6*v_sd_G14
        struct[0].Gy_ini[136,134] = 7.5e-6*v_sq_G14
        struct[0].Gy_ini[136,136] = -1
        struct[0].Gy_ini[137,129] = -7.5e-6*i_sq_G14
        struct[0].Gy_ini[137,130] = 7.5e-6*i_sd_G14
        struct[0].Gy_ini[137,133] = 7.5e-6*v_sq_G14
        struct[0].Gy_ini[137,134] = -7.5e-6*v_sd_G14
        struct[0].Gy_ini[137,137] = -1
        struct[0].Gy_ini[138,138] = -1
        struct[0].Gy_ini[139,139] = -1



def run_nn(t,struct,mode):

    # Parameters:
    R_R00R01 = struct[0].R_R00R01
    L_R00R01 = struct[0].L_R00R01
    C_R00R01 = struct[0].C_R00R01
    R_R02R01 = struct[0].R_R02R01
    L_R02R01 = struct[0].L_R02R01
    C_R02R01 = struct[0].C_R02R01
    R_R02R03 = struct[0].R_R02R03
    L_R02R03 = struct[0].L_R02R03
    C_R02R03 = struct[0].C_R02R03
    R_R03R04 = struct[0].R_R03R04
    L_R03R04 = struct[0].L_R03R04
    C_R03R04 = struct[0].C_R03R04
    R_R04R05 = struct[0].R_R04R05
    L_R04R05 = struct[0].L_R04R05
    C_R04R05 = struct[0].C_R04R05
    R_R04R12 = struct[0].R_R04R12
    L_R04R12 = struct[0].L_R04R12
    C_R04R12 = struct[0].C_R04R12
    R_R05R06 = struct[0].R_R05R06
    L_R05R06 = struct[0].L_R05R06
    C_R05R06 = struct[0].C_R05R06
    R_R06R07 = struct[0].R_R06R07
    L_R06R07 = struct[0].L_R06R07
    C_R06R07 = struct[0].C_R06R07
    R_R07R08 = struct[0].R_R07R08
    L_R07R08 = struct[0].L_R07R08
    C_R07R08 = struct[0].C_R07R08
    R_R08R09 = struct[0].R_R08R09
    L_R08R09 = struct[0].L_R08R09
    C_R08R09 = struct[0].C_R08R09
    R_R09R10 = struct[0].R_R09R10
    L_R09R10 = struct[0].L_R09R10
    C_R09R10 = struct[0].C_R09R10
    R_R09R17 = struct[0].R_R09R17
    L_R09R17 = struct[0].L_R09R17
    C_R09R17 = struct[0].C_R09R17
    R_R11R03 = struct[0].R_R11R03
    L_R11R03 = struct[0].L_R11R03
    C_R11R03 = struct[0].C_R11R03
    R_R12R13 = struct[0].R_R12R13
    L_R12R13 = struct[0].L_R12R13
    C_R12R13 = struct[0].C_R12R13
    R_R13R14 = struct[0].R_R13R14
    L_R13R14 = struct[0].L_R13R14
    C_R13R14 = struct[0].C_R13R14
    R_R14R15 = struct[0].R_R14R15
    L_R14R15 = struct[0].L_R14R15
    C_R14R15 = struct[0].C_R14R15
    R_R16R06 = struct[0].R_R16R06
    L_R16R06 = struct[0].L_R16R06
    C_R16R06 = struct[0].C_R16R06
    R_R18R10 = struct[0].R_R18R10
    L_R18R10 = struct[0].L_R18R10
    C_R18R10 = struct[0].C_R18R10
    i_R00_D = struct[0].i_R00_D
    i_R00_Q = struct[0].i_R00_Q
    i_R02_D = struct[0].i_R02_D
    i_R02_Q = struct[0].i_R02_Q
    i_R03_D = struct[0].i_R03_D
    i_R03_Q = struct[0].i_R03_Q
    i_R04_D = struct[0].i_R04_D
    i_R04_Q = struct[0].i_R04_Q
    i_R05_D = struct[0].i_R05_D
    i_R05_Q = struct[0].i_R05_Q
    i_R06_D = struct[0].i_R06_D
    i_R06_Q = struct[0].i_R06_Q
    i_R07_D = struct[0].i_R07_D
    i_R07_Q = struct[0].i_R07_Q
    i_R08_D = struct[0].i_R08_D
    i_R08_Q = struct[0].i_R08_Q
    i_R09_D = struct[0].i_R09_D
    i_R09_Q = struct[0].i_R09_Q
    i_R12_D = struct[0].i_R12_D
    i_R12_Q = struct[0].i_R12_Q
    i_R13_D = struct[0].i_R13_D
    i_R13_Q = struct[0].i_R13_Q
    omega = struct[0].omega
    L_t_G10 = struct[0].L_t_G10
    R_t_G10 = struct[0].R_t_G10
    C_m_G10 = struct[0].C_m_G10
    L_s_G10 = struct[0].L_s_G10
    R_s_G10 = struct[0].R_s_G10
    omega_G10 = struct[0].omega_G10
    G_d_G10 = struct[0].G_d_G10
    K_p_G10 = struct[0].K_p_G10
    T_p_G10 = struct[0].T_p_G10
    K_q_G10 = struct[0].K_q_G10
    T_q_G10 = struct[0].T_q_G10
    R_v_G10 = struct[0].R_v_G10
    X_v_G10 = struct[0].X_v_G10
    S_b_kVA_G10 = struct[0].S_b_kVA_G10
    U_b_G10 = struct[0].U_b_G10
    K_phi_G10 = struct[0].K_phi_G10
    H_G10 = struct[0].H_G10
    D_G10 = struct[0].D_G10
    T_vpoi_G10 = struct[0].T_vpoi_G10
    K_vpoi_G10 = struct[0].K_vpoi_G10
    T_f_G10 = struct[0].T_f_G10
    K_f_G10 = struct[0].K_f_G10
    L_t_G14 = struct[0].L_t_G14
    R_t_G14 = struct[0].R_t_G14
    C_m_G14 = struct[0].C_m_G14
    L_s_G14 = struct[0].L_s_G14
    R_s_G14 = struct[0].R_s_G14
    omega_G14 = struct[0].omega_G14
    G_d_G14 = struct[0].G_d_G14
    K_p_G14 = struct[0].K_p_G14
    T_p_G14 = struct[0].T_p_G14
    K_q_G14 = struct[0].K_q_G14
    T_q_G14 = struct[0].T_q_G14
    R_v_G14 = struct[0].R_v_G14
    X_v_G14 = struct[0].X_v_G14
    S_b_kVA_G14 = struct[0].S_b_kVA_G14
    U_b_G14 = struct[0].U_b_G14
    K_phi_G14 = struct[0].K_phi_G14
    H_G14 = struct[0].H_G14
    D_G14 = struct[0].D_G14
    T_vpoi_G14 = struct[0].T_vpoi_G14
    K_vpoi_G14 = struct[0].K_vpoi_G14
    T_f_G14 = struct[0].T_f_G14
    K_f_G14 = struct[0].K_f_G14
    K_f_sec = struct[0].K_f_sec
    
    # Inputs:
    T_i_R01 = struct[0].T_i_R01
    I_max_R01 = struct[0].I_max_R01
    p_R01_ref = struct[0].p_R01_ref
    q_R01_ref = struct[0].q_R01_ref
    T_i_R11 = struct[0].T_i_R11
    I_max_R11 = struct[0].I_max_R11
    p_R11_ref = struct[0].p_R11_ref
    q_R11_ref = struct[0].q_R11_ref
    T_i_R15 = struct[0].T_i_R15
    I_max_R15 = struct[0].I_max_R15
    p_R15_ref = struct[0].p_R15_ref
    q_R15_ref = struct[0].q_R15_ref
    T_i_R16 = struct[0].T_i_R16
    I_max_R16 = struct[0].I_max_R16
    p_R16_ref = struct[0].p_R16_ref
    q_R16_ref = struct[0].q_R16_ref
    T_i_R17 = struct[0].T_i_R17
    I_max_R17 = struct[0].I_max_R17
    p_R17_ref = struct[0].p_R17_ref
    q_R17_ref = struct[0].q_R17_ref
    T_i_R18 = struct[0].T_i_R18
    I_max_R18 = struct[0].I_max_R18
    p_R18_ref = struct[0].p_R18_ref
    q_R18_ref = struct[0].q_R18_ref
    v_dc_G10 = struct[0].v_dc_G10
    p_m_ref_G10 = struct[0].p_m_ref_G10
    q_s_ref_G10 = struct[0].q_s_ref_G10
    v_s_ref_G10 = struct[0].v_s_ref_G10
    omega_ref_G10 = struct[0].omega_ref_G10
    p_r_G10 = struct[0].p_r_G10
    q_r_G10 = struct[0].q_r_G10
    v_dc_G14 = struct[0].v_dc_G14
    p_m_ref_G14 = struct[0].p_m_ref_G14
    q_s_ref_G14 = struct[0].q_s_ref_G14
    v_s_ref_G14 = struct[0].v_s_ref_G14
    omega_ref_G14 = struct[0].omega_ref_G14
    p_r_G14 = struct[0].p_r_G14
    q_r_G14 = struct[0].q_r_G14
    
    # Dynamical states:
    i_R01_D = struct[0].x[0,0]
    i_R01_Q = struct[0].x[1,0]
    i_R11_D = struct[0].x[2,0]
    i_R11_Q = struct[0].x[3,0]
    i_R15_D = struct[0].x[4,0]
    i_R15_Q = struct[0].x[5,0]
    i_R16_D = struct[0].x[6,0]
    i_R16_Q = struct[0].x[7,0]
    i_R17_D = struct[0].x[8,0]
    i_R17_Q = struct[0].x[9,0]
    i_R18_D = struct[0].x[10,0]
    i_R18_Q = struct[0].x[11,0]
    phi_G10 = struct[0].x[12,0]
    omega_v_G10 = struct[0].x[13,0]
    xi_q_G10 = struct[0].x[14,0]
    omega_rads_G10 = struct[0].x[15,0]
    omega_v_filt_G10 = struct[0].x[16,0]
    v_s_filt_G10 = struct[0].x[17,0]
    phi_G14 = struct[0].x[18,0]
    omega_v_G14 = struct[0].x[19,0]
    xi_q_G14 = struct[0].x[20,0]
    omega_rads_G14 = struct[0].x[21,0]
    omega_v_filt_G14 = struct[0].x[22,0]
    v_s_filt_G14 = struct[0].x[23,0]
    xi_f_sec = struct[0].x[24,0]
    
    # Algebraic states:
    i_l_R00R01_D = struct[0].y_run[0,0]
    i_l_R00R01_Q = struct[0].y_run[1,0]
    i_l_R02R01_D = struct[0].y_run[2,0]
    i_l_R02R01_Q = struct[0].y_run[3,0]
    i_l_R02R03_D = struct[0].y_run[4,0]
    i_l_R02R03_Q = struct[0].y_run[5,0]
    i_l_R03R04_D = struct[0].y_run[6,0]
    i_l_R03R04_Q = struct[0].y_run[7,0]
    i_l_R04R05_D = struct[0].y_run[8,0]
    i_l_R04R05_Q = struct[0].y_run[9,0]
    i_l_R04R12_D = struct[0].y_run[10,0]
    i_l_R04R12_Q = struct[0].y_run[11,0]
    i_l_R05R06_D = struct[0].y_run[12,0]
    i_l_R05R06_Q = struct[0].y_run[13,0]
    i_l_R06R07_D = struct[0].y_run[14,0]
    i_l_R06R07_Q = struct[0].y_run[15,0]
    i_l_R07R08_D = struct[0].y_run[16,0]
    i_l_R07R08_Q = struct[0].y_run[17,0]
    i_l_R08R09_D = struct[0].y_run[18,0]
    i_l_R08R09_Q = struct[0].y_run[19,0]
    i_l_R09R10_D = struct[0].y_run[20,0]
    i_l_R09R10_Q = struct[0].y_run[21,0]
    i_l_R09R17_D = struct[0].y_run[22,0]
    i_l_R09R17_Q = struct[0].y_run[23,0]
    i_l_R11R03_D = struct[0].y_run[24,0]
    i_l_R11R03_Q = struct[0].y_run[25,0]
    i_l_R12R13_D = struct[0].y_run[26,0]
    i_l_R12R13_Q = struct[0].y_run[27,0]
    i_l_R13R14_D = struct[0].y_run[28,0]
    i_l_R13R14_Q = struct[0].y_run[29,0]
    i_l_R14R15_D = struct[0].y_run[30,0]
    i_l_R14R15_Q = struct[0].y_run[31,0]
    i_l_R16R06_D = struct[0].y_run[32,0]
    i_l_R16R06_Q = struct[0].y_run[33,0]
    i_l_R18R10_D = struct[0].y_run[34,0]
    i_l_R18R10_Q = struct[0].y_run[35,0]
    v_R00_D = struct[0].y_run[36,0]
    v_R00_Q = struct[0].y_run[37,0]
    v_R01_D = struct[0].y_run[38,0]
    v_R01_Q = struct[0].y_run[39,0]
    v_R02_D = struct[0].y_run[40,0]
    v_R02_Q = struct[0].y_run[41,0]
    v_R03_D = struct[0].y_run[42,0]
    v_R03_Q = struct[0].y_run[43,0]
    v_R04_D = struct[0].y_run[44,0]
    v_R04_Q = struct[0].y_run[45,0]
    v_R05_D = struct[0].y_run[46,0]
    v_R05_Q = struct[0].y_run[47,0]
    v_R06_D = struct[0].y_run[48,0]
    v_R06_Q = struct[0].y_run[49,0]
    v_R07_D = struct[0].y_run[50,0]
    v_R07_Q = struct[0].y_run[51,0]
    v_R08_D = struct[0].y_run[52,0]
    v_R08_Q = struct[0].y_run[53,0]
    v_R09_D = struct[0].y_run[54,0]
    v_R09_Q = struct[0].y_run[55,0]
    v_R10_D = struct[0].y_run[56,0]
    v_R10_Q = struct[0].y_run[57,0]
    v_R11_D = struct[0].y_run[58,0]
    v_R11_Q = struct[0].y_run[59,0]
    v_R12_D = struct[0].y_run[60,0]
    v_R12_Q = struct[0].y_run[61,0]
    v_R13_D = struct[0].y_run[62,0]
    v_R13_Q = struct[0].y_run[63,0]
    v_R14_D = struct[0].y_run[64,0]
    v_R14_Q = struct[0].y_run[65,0]
    v_R15_D = struct[0].y_run[66,0]
    v_R15_Q = struct[0].y_run[67,0]
    v_R16_D = struct[0].y_run[68,0]
    v_R16_Q = struct[0].y_run[69,0]
    v_R17_D = struct[0].y_run[70,0]
    v_R17_Q = struct[0].y_run[71,0]
    v_R18_D = struct[0].y_run[72,0]
    v_R18_Q = struct[0].y_run[73,0]
    i_R01_d_ref = struct[0].y_run[74,0]
    i_R01_q_ref = struct[0].y_run[75,0]
    i_R11_d_ref = struct[0].y_run[76,0]
    i_R11_q_ref = struct[0].y_run[77,0]
    i_R15_d_ref = struct[0].y_run[78,0]
    i_R15_q_ref = struct[0].y_run[79,0]
    i_R16_d_ref = struct[0].y_run[80,0]
    i_R16_q_ref = struct[0].y_run[81,0]
    i_R17_d_ref = struct[0].y_run[82,0]
    i_R17_q_ref = struct[0].y_run[83,0]
    i_R18_d_ref = struct[0].y_run[84,0]
    i_R18_q_ref = struct[0].y_run[85,0]
    i_tD_G10 = struct[0].y_run[86,0]
    i_tQ_G10 = struct[0].y_run[87,0]
    v_mD_G10 = struct[0].y_run[88,0]
    v_mQ_G10 = struct[0].y_run[89,0]
    i_sD_G10 = struct[0].y_run[90,0]
    i_sQ_G10 = struct[0].y_run[91,0]
    i_R10_D = struct[0].y_run[92,0]
    i_R10_Q = struct[0].y_run[93,0]
    v_sD_G10 = struct[0].y_run[94,0]
    v_sQ_G10 = struct[0].y_run[95,0]
    eta_d_G10 = struct[0].y_run[96,0]
    eta_q_G10 = struct[0].y_run[97,0]
    eta_D_G10 = struct[0].y_run[98,0]
    eta_Q_G10 = struct[0].y_run[99,0]
    v_md_G10 = struct[0].y_run[100,0]
    v_mq_G10 = struct[0].y_run[101,0]
    v_sd_G10 = struct[0].y_run[102,0]
    v_sq_G10 = struct[0].y_run[103,0]
    i_td_G10 = struct[0].y_run[104,0]
    i_tq_G10 = struct[0].y_run[105,0]
    i_sd_G10 = struct[0].y_run[106,0]
    i_sq_G10 = struct[0].y_run[107,0]
    DV_sat_G10 = struct[0].y_run[108,0]
    p_s_pu_G10 = struct[0].y_run[109,0]
    q_s_pu_G10 = struct[0].y_run[110,0]
    p_m_ref_G10 = struct[0].y_run[111,0]
    q_s_ref_G10 = struct[0].y_run[112,0]
    i_tD_G14 = struct[0].y_run[113,0]
    i_tQ_G14 = struct[0].y_run[114,0]
    v_mD_G14 = struct[0].y_run[115,0]
    v_mQ_G14 = struct[0].y_run[116,0]
    i_sD_G14 = struct[0].y_run[117,0]
    i_sQ_G14 = struct[0].y_run[118,0]
    i_R14_D = struct[0].y_run[119,0]
    i_R14_Q = struct[0].y_run[120,0]
    v_sD_G14 = struct[0].y_run[121,0]
    v_sQ_G14 = struct[0].y_run[122,0]
    eta_d_G14 = struct[0].y_run[123,0]
    eta_q_G14 = struct[0].y_run[124,0]
    eta_D_G14 = struct[0].y_run[125,0]
    eta_Q_G14 = struct[0].y_run[126,0]
    v_md_G14 = struct[0].y_run[127,0]
    v_mq_G14 = struct[0].y_run[128,0]
    v_sd_G14 = struct[0].y_run[129,0]
    v_sq_G14 = struct[0].y_run[130,0]
    i_td_G14 = struct[0].y_run[131,0]
    i_tq_G14 = struct[0].y_run[132,0]
    i_sd_G14 = struct[0].y_run[133,0]
    i_sq_G14 = struct[0].y_run[134,0]
    DV_sat_G14 = struct[0].y_run[135,0]
    p_s_pu_G14 = struct[0].y_run[136,0]
    q_s_pu_G14 = struct[0].y_run[137,0]
    p_m_ref_G14 = struct[0].y_run[138,0]
    q_s_ref_G14 = struct[0].y_run[139,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = -100.0*i_R01_D + 100.0*Piecewise(np.array([(-I_max_R01, I_max_R01 < -i_R01_d_ref), (I_max_R01, I_max_R01 < i_R01_d_ref), (i_R01_d_ref, True)]))
        struct[0].f[1,0] = -100.0*i_R01_Q + 100.0*Piecewise(np.array([(-I_max_R01, I_max_R01 < -i_R01_q_ref), (I_max_R01, I_max_R01 < i_R01_q_ref), (i_R01_q_ref, True)]))
        struct[0].f[2,0] = -100.0*i_R11_D + 100.0*Piecewise(np.array([(-I_max_R11, I_max_R11 < -i_R11_d_ref), (I_max_R11, I_max_R11 < i_R11_d_ref), (i_R11_d_ref, True)]))
        struct[0].f[3,0] = -100.0*i_R11_Q + 100.0*Piecewise(np.array([(-I_max_R11, I_max_R11 < -i_R11_q_ref), (I_max_R11, I_max_R11 < i_R11_q_ref), (i_R11_q_ref, True)]))
        struct[0].f[4,0] = -100.0*i_R15_D + 100.0*Piecewise(np.array([(-I_max_R15, I_max_R15 < -i_R15_d_ref), (I_max_R15, I_max_R15 < i_R15_d_ref), (i_R15_d_ref, True)]))
        struct[0].f[5,0] = -100.0*i_R15_Q + 100.0*Piecewise(np.array([(-I_max_R15, I_max_R15 < -i_R15_q_ref), (I_max_R15, I_max_R15 < i_R15_q_ref), (i_R15_q_ref, True)]))
        struct[0].f[6,0] = -100.0*i_R16_D + 100.0*Piecewise(np.array([(-I_max_R16, I_max_R16 < -i_R16_d_ref), (I_max_R16, I_max_R16 < i_R16_d_ref), (i_R16_d_ref, True)]))
        struct[0].f[7,0] = -100.0*i_R16_Q + 100.0*Piecewise(np.array([(-I_max_R16, I_max_R16 < -i_R16_q_ref), (I_max_R16, I_max_R16 < i_R16_q_ref), (i_R16_q_ref, True)]))
        struct[0].f[8,0] = -100.0*i_R17_D + 100.0*Piecewise(np.array([(-I_max_R17, I_max_R17 < -i_R17_d_ref), (I_max_R17, I_max_R17 < i_R17_d_ref), (i_R17_d_ref, True)]))
        struct[0].f[9,0] = -100.0*i_R17_Q + 100.0*Piecewise(np.array([(-I_max_R17, I_max_R17 < -i_R17_q_ref), (I_max_R17, I_max_R17 < i_R17_q_ref), (i_R17_q_ref, True)]))
        struct[0].f[10,0] = -100.0*i_R18_D + 100.0*Piecewise(np.array([(-I_max_R18, I_max_R18 < -i_R18_d_ref), (I_max_R18, I_max_R18 < i_R18_d_ref), (i_R18_d_ref, True)]))
        struct[0].f[11,0] = -100.0*i_R18_Q + 100.0*Piecewise(np.array([(-I_max_R18, I_max_R18 < -i_R18_q_ref), (I_max_R18, I_max_R18 < i_R18_q_ref), (i_R18_q_ref, True)]))
        struct[0].f[12,0] = -K_phi_G10*phi_G10 + 314.159265358979*omega_v_G10 - 314.159265358979*(H_G10*omega_v_G10 + H_G14*omega_v_G14)/(H_G10 + H_G14)
        struct[0].f[13,0] = (-D_G10*(omega_v_G10 - 1.0) + p_m_ref_G10 - p_s_pu_G10)/(2*H_G10)
        struct[0].f[14,0] = -q_s_pu_G10 + q_s_ref_G10
        struct[0].f[15,0] = -1.0*omega_rads_G10 + 314.159265358979*omega_v_G10
        struct[0].f[16,0] = (omega_v_G10 - omega_v_filt_G10)/T_f_G10
        struct[0].f[17,0] = (-v_s_filt_G10 + 0.00306186217847897*(v_sd_G10**2 + v_sq_G10**2)**0.5)/T_vpoi_G10
        struct[0].f[18,0] = -K_phi_G14*phi_G14 + 314.159265358979*omega_v_G14 - 314.159265358979*(H_G10*omega_v_G10 + H_G14*omega_v_G14)/(H_G10 + H_G14)
        struct[0].f[19,0] = (-D_G14*(omega_v_G14 - 1.0) + p_m_ref_G14 - p_s_pu_G14)/(2*H_G14)
        struct[0].f[20,0] = -q_s_pu_G14 + q_s_ref_G14
        struct[0].f[21,0] = -1.0*omega_rads_G14 + 314.159265358979*omega_v_G14
        struct[0].f[22,0] = (omega_v_G14 - omega_v_filt_G14)/T_f_G14
        struct[0].f[23,0] = (-v_s_filt_G14 + 0.00306186217847897*(v_sd_G14**2 + v_sq_G14**2)**0.5)/T_vpoi_G14
        struct[0].f[24,0] = 1 - (H_G10*omega_v_G10 + H_G14*omega_v_G14)/(H_G10 + H_G14)
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = L_R00R01*i_l_R00R01_Q*omega - R_R00R01*i_l_R00R01_D + v_R00_D - v_R01_D
        struct[0].g[1,0] = -L_R00R01*i_l_R00R01_D*omega - R_R00R01*i_l_R00R01_Q + v_R00_Q - v_R01_Q
        struct[0].g[2,0] = L_R02R01*i_l_R02R01_Q*omega - R_R02R01*i_l_R02R01_D - v_R01_D + v_R02_D
        struct[0].g[3,0] = -L_R02R01*i_l_R02R01_D*omega - R_R02R01*i_l_R02R01_Q - v_R01_Q + v_R02_Q
        struct[0].g[4,0] = L_R02R03*i_l_R02R03_Q*omega - R_R02R03*i_l_R02R03_D + v_R02_D - v_R03_D
        struct[0].g[5,0] = -L_R02R03*i_l_R02R03_D*omega - R_R02R03*i_l_R02R03_Q + v_R02_Q - v_R03_Q
        struct[0].g[6,0] = L_R03R04*i_l_R03R04_Q*omega - R_R03R04*i_l_R03R04_D + v_R03_D - v_R04_D
        struct[0].g[7,0] = -L_R03R04*i_l_R03R04_D*omega - R_R03R04*i_l_R03R04_Q + v_R03_Q - v_R04_Q
        struct[0].g[8,0] = L_R04R05*i_l_R04R05_Q*omega - R_R04R05*i_l_R04R05_D + v_R04_D - v_R05_D
        struct[0].g[9,0] = -L_R04R05*i_l_R04R05_D*omega - R_R04R05*i_l_R04R05_Q + v_R04_Q - v_R05_Q
        struct[0].g[10,0] = L_R04R12*i_l_R04R12_Q*omega - R_R04R12*i_l_R04R12_D + v_R04_D - v_R12_D
        struct[0].g[11,0] = -L_R04R12*i_l_R04R12_D*omega - R_R04R12*i_l_R04R12_Q + v_R04_Q - v_R12_Q
        struct[0].g[12,0] = L_R05R06*i_l_R05R06_Q*omega - R_R05R06*i_l_R05R06_D + v_R05_D - v_R06_D
        struct[0].g[13,0] = -L_R05R06*i_l_R05R06_D*omega - R_R05R06*i_l_R05R06_Q + v_R05_Q - v_R06_Q
        struct[0].g[14,0] = L_R06R07*i_l_R06R07_Q*omega - R_R06R07*i_l_R06R07_D + v_R06_D - v_R07_D
        struct[0].g[15,0] = -L_R06R07*i_l_R06R07_D*omega - R_R06R07*i_l_R06R07_Q + v_R06_Q - v_R07_Q
        struct[0].g[16,0] = L_R07R08*i_l_R07R08_Q*omega - R_R07R08*i_l_R07R08_D + v_R07_D - v_R08_D
        struct[0].g[17,0] = -L_R07R08*i_l_R07R08_D*omega - R_R07R08*i_l_R07R08_Q + v_R07_Q - v_R08_Q
        struct[0].g[18,0] = L_R08R09*i_l_R08R09_Q*omega - R_R08R09*i_l_R08R09_D + v_R08_D - v_R09_D
        struct[0].g[19,0] = -L_R08R09*i_l_R08R09_D*omega - R_R08R09*i_l_R08R09_Q + v_R08_Q - v_R09_Q
        struct[0].g[20,0] = L_R09R10*i_l_R09R10_Q*omega - R_R09R10*i_l_R09R10_D + v_R09_D - v_R10_D
        struct[0].g[21,0] = -L_R09R10*i_l_R09R10_D*omega - R_R09R10*i_l_R09R10_Q + v_R09_Q - v_R10_Q
        struct[0].g[22,0] = L_R09R17*i_l_R09R17_Q*omega - R_R09R17*i_l_R09R17_D + v_R09_D - v_R17_D
        struct[0].g[23,0] = -L_R09R17*i_l_R09R17_D*omega - R_R09R17*i_l_R09R17_Q + v_R09_Q - v_R17_Q
        struct[0].g[24,0] = L_R11R03*i_l_R11R03_Q*omega - R_R11R03*i_l_R11R03_D - v_R03_D + v_R11_D
        struct[0].g[25,0] = -L_R11R03*i_l_R11R03_D*omega - R_R11R03*i_l_R11R03_Q - v_R03_Q + v_R11_Q
        struct[0].g[26,0] = L_R12R13*i_l_R12R13_Q*omega - R_R12R13*i_l_R12R13_D + v_R12_D - v_R13_D
        struct[0].g[27,0] = -L_R12R13*i_l_R12R13_D*omega - R_R12R13*i_l_R12R13_Q + v_R12_Q - v_R13_Q
        struct[0].g[28,0] = L_R13R14*i_l_R13R14_Q*omega - R_R13R14*i_l_R13R14_D + v_R13_D - v_R14_D
        struct[0].g[29,0] = -L_R13R14*i_l_R13R14_D*omega - R_R13R14*i_l_R13R14_Q + v_R13_Q - v_R14_Q
        struct[0].g[30,0] = L_R14R15*i_l_R14R15_Q*omega - R_R14R15*i_l_R14R15_D + v_R14_D - v_R15_D
        struct[0].g[31,0] = -L_R14R15*i_l_R14R15_D*omega - R_R14R15*i_l_R14R15_Q + v_R14_Q - v_R15_Q
        struct[0].g[32,0] = L_R16R06*i_l_R16R06_Q*omega - R_R16R06*i_l_R16R06_D - v_R06_D + v_R16_D
        struct[0].g[33,0] = -L_R16R06*i_l_R16R06_D*omega - R_R16R06*i_l_R16R06_Q - v_R06_Q + v_R16_Q
        struct[0].g[34,0] = L_R18R10*i_l_R18R10_Q*omega - R_R18R10*i_l_R18R10_D - v_R10_D + v_R18_D
        struct[0].g[35,0] = -L_R18R10*i_l_R18R10_D*omega - R_R18R10*i_l_R18R10_Q - v_R10_Q + v_R18_Q
        struct[0].g[36,0] = C_R00R01*omega*v_R00_Q/2 + i_R00_D - i_l_R00R01_D
        struct[0].g[37,0] = -C_R00R01*omega*v_R00_D/2 + i_R00_Q - i_l_R00R01_Q
        struct[0].g[38,0] = i_R01_D + i_l_R00R01_D + i_l_R02R01_D + omega*v_R01_Q*(C_R00R01/2 + C_R02R01/2)
        struct[0].g[39,0] = i_R01_Q + i_l_R00R01_Q + i_l_R02R01_Q - omega*v_R01_D*(C_R00R01/2 + C_R02R01/2)
        struct[0].g[40,0] = i_R02_D - i_l_R02R01_D - i_l_R02R03_D + omega*v_R02_Q*(C_R02R01/2 + C_R02R03/2)
        struct[0].g[41,0] = i_R02_Q - i_l_R02R01_Q - i_l_R02R03_Q - omega*v_R02_D*(C_R02R01/2 + C_R02R03/2)
        struct[0].g[42,0] = i_R03_D + i_l_R02R03_D - i_l_R03R04_D + i_l_R11R03_D + omega*v_R03_Q*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].g[43,0] = i_R03_Q + i_l_R02R03_Q - i_l_R03R04_Q + i_l_R11R03_Q - omega*v_R03_D*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].g[44,0] = i_R04_D + i_l_R03R04_D - i_l_R04R05_D - i_l_R04R12_D + omega*v_R04_Q*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].g[45,0] = i_R04_Q + i_l_R03R04_Q - i_l_R04R05_Q - i_l_R04R12_Q - omega*v_R04_D*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].g[46,0] = i_R05_D + i_l_R04R05_D - i_l_R05R06_D + omega*v_R05_Q*(C_R04R05/2 + C_R05R06/2)
        struct[0].g[47,0] = i_R05_Q + i_l_R04R05_Q - i_l_R05R06_Q - omega*v_R05_D*(C_R04R05/2 + C_R05R06/2)
        struct[0].g[48,0] = i_R06_D + i_l_R05R06_D - i_l_R06R07_D + i_l_R16R06_D + omega*v_R06_Q*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].g[49,0] = i_R06_Q + i_l_R05R06_Q - i_l_R06R07_Q + i_l_R16R06_Q - omega*v_R06_D*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].g[50,0] = i_R07_D + i_l_R06R07_D - i_l_R07R08_D + omega*v_R07_Q*(C_R06R07/2 + C_R07R08/2)
        struct[0].g[51,0] = i_R07_Q + i_l_R06R07_Q - i_l_R07R08_Q - omega*v_R07_D*(C_R06R07/2 + C_R07R08/2)
        struct[0].g[52,0] = i_R08_D + i_l_R07R08_D - i_l_R08R09_D + omega*v_R08_Q*(C_R07R08/2 + C_R08R09/2)
        struct[0].g[53,0] = i_R08_Q + i_l_R07R08_Q - i_l_R08R09_Q - omega*v_R08_D*(C_R07R08/2 + C_R08R09/2)
        struct[0].g[54,0] = i_R09_D + i_l_R08R09_D - i_l_R09R10_D - i_l_R09R17_D + omega*v_R09_Q*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].g[55,0] = i_R09_Q + i_l_R08R09_Q - i_l_R09R10_Q - i_l_R09R17_Q - omega*v_R09_D*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].g[56,0] = i_R10_D + i_l_R09R10_D + i_l_R18R10_D + omega*v_R10_Q*(C_R09R10/2 + C_R18R10/2)
        struct[0].g[57,0] = i_R10_Q + i_l_R09R10_Q + i_l_R18R10_Q - omega*v_R10_D*(C_R09R10/2 + C_R18R10/2)
        struct[0].g[58,0] = C_R11R03*omega*v_R11_Q/2 + i_R11_D - i_l_R11R03_D
        struct[0].g[59,0] = -C_R11R03*omega*v_R11_D/2 + i_R11_Q - i_l_R11R03_Q
        struct[0].g[60,0] = i_R12_D + i_l_R04R12_D - i_l_R12R13_D + omega*v_R12_Q*(C_R04R12/2 + C_R12R13/2)
        struct[0].g[61,0] = i_R12_Q + i_l_R04R12_Q - i_l_R12R13_Q - omega*v_R12_D*(C_R04R12/2 + C_R12R13/2)
        struct[0].g[62,0] = i_R13_D + i_l_R12R13_D - i_l_R13R14_D + omega*v_R13_Q*(C_R12R13/2 + C_R13R14/2)
        struct[0].g[63,0] = i_R13_Q + i_l_R12R13_Q - i_l_R13R14_Q - omega*v_R13_D*(C_R12R13/2 + C_R13R14/2)
        struct[0].g[64,0] = i_R14_D + i_l_R13R14_D - i_l_R14R15_D + omega*v_R14_Q*(C_R13R14/2 + C_R14R15/2)
        struct[0].g[65,0] = i_R14_Q + i_l_R13R14_Q - i_l_R14R15_Q - omega*v_R14_D*(C_R13R14/2 + C_R14R15/2)
        struct[0].g[66,0] = C_R14R15*omega*v_R15_Q/2 + i_R15_D + i_l_R14R15_D
        struct[0].g[67,0] = -C_R14R15*omega*v_R15_D/2 + i_R15_Q + i_l_R14R15_Q
        struct[0].g[68,0] = C_R16R06*omega*v_R16_Q/2 + i_R16_D - i_l_R16R06_D
        struct[0].g[69,0] = -C_R16R06*omega*v_R16_D/2 + i_R16_Q - i_l_R16R06_Q
        struct[0].g[70,0] = C_R09R17*omega*v_R17_Q/2 + i_R17_D + i_l_R09R17_D
        struct[0].g[71,0] = -C_R09R17*omega*v_R17_D/2 + i_R17_Q + i_l_R09R17_Q
        struct[0].g[72,0] = C_R18R10*omega*v_R18_Q/2 + i_R18_D - i_l_R18R10_D
        struct[0].g[73,0] = -C_R18R10*omega*v_R18_D/2 + i_R18_Q - i_l_R18R10_Q
        struct[0].g[74,0] = -i_R01_d_ref + (-0.666666666666667*p_R01_ref*v_R01_D - 0.666666666666667*q_R01_ref*v_R01_Q)*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)]))
        struct[0].g[75,0] = -i_R01_q_ref - (0.666666666666667*p_R01_ref*v_R01_Q - 0.666666666666667*q_R01_ref*v_R01_D)*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)]))
        struct[0].g[76,0] = -i_R11_d_ref + (-0.666666666666667*p_R11_ref*v_R11_D - 0.666666666666667*q_R11_ref*v_R11_Q)*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)]))
        struct[0].g[77,0] = -i_R11_q_ref - (0.666666666666667*p_R11_ref*v_R11_Q - 0.666666666666667*q_R11_ref*v_R11_D)*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)]))
        struct[0].g[78,0] = -i_R15_d_ref + (-0.666666666666667*p_R15_ref*v_R15_D - 0.666666666666667*q_R15_ref*v_R15_Q)*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)]))
        struct[0].g[79,0] = -i_R15_q_ref - (0.666666666666667*p_R15_ref*v_R15_Q - 0.666666666666667*q_R15_ref*v_R15_D)*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)]))
        struct[0].g[80,0] = -i_R16_d_ref + (-0.666666666666667*p_R16_ref*v_R16_D - 0.666666666666667*q_R16_ref*v_R16_Q)*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)]))
        struct[0].g[81,0] = -i_R16_q_ref - (0.666666666666667*p_R16_ref*v_R16_Q - 0.666666666666667*q_R16_ref*v_R16_D)*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)]))
        struct[0].g[82,0] = -i_R17_d_ref + (-0.666666666666667*p_R17_ref*v_R17_D - 0.666666666666667*q_R17_ref*v_R17_Q)*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)]))
        struct[0].g[83,0] = -i_R17_q_ref - (0.666666666666667*p_R17_ref*v_R17_Q - 0.666666666666667*q_R17_ref*v_R17_D)*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)]))
        struct[0].g[84,0] = -i_R18_d_ref + (-0.666666666666667*p_R18_ref*v_R18_D - 0.666666666666667*q_R18_ref*v_R18_Q)*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)]))
        struct[0].g[85,0] = -i_R18_q_ref - (0.666666666666667*p_R18_ref*v_R18_Q - 0.666666666666667*q_R18_ref*v_R18_D)*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)]))
        struct[0].g[86,0] = (L_t_G10*i_tQ_G10*omega_G10 - R_t_G10*i_tD_G10 + eta_D_G10*v_dc_G10/2 - v_mD_G10)/L_t_G10
        struct[0].g[87,0] = (-L_t_G10*i_tD_G10*omega_G10 - R_t_G10*i_tQ_G10 + eta_Q_G10*v_dc_G10/2 - v_mQ_G10)/L_t_G10
        struct[0].g[88,0] = (C_m_G10*omega_G10*v_mQ_G10 - G_d_G10*v_mD_G10 - i_sD_G10 + i_tD_G10)/C_m_G10
        struct[0].g[89,0] = (-C_m_G10*omega_G10*v_mD_G10 - G_d_G10*v_mQ_G10 - i_sQ_G10 + i_tQ_G10)/C_m_G10
        struct[0].g[90,0] = (L_s_G10*i_sQ_G10*omega_G10 - R_s_G10*i_sD_G10 + v_mD_G10 - v_sD_G10)/L_s_G10
        struct[0].g[91,0] = (-L_s_G10*i_sD_G10*omega_G10 - R_s_G10*i_sQ_G10 + v_mQ_G10 - v_sQ_G10)/L_s_G10
        struct[0].g[92,0] = -i_R10_D + i_sD_G10
        struct[0].g[93,0] = -i_R10_Q + i_sQ_G10
        struct[0].g[94,0] = -v_R10_D + v_sD_G10
        struct[0].g[95,0] = -v_R10_Q + v_sQ_G10
        struct[0].g[96,0] = eta_d_G10 - 2*(-0.8*R_v_G10*i_sd_G10 + 0.8*X_v_G10*i_sq_G10)/v_dc_G10
        struct[0].g[97,0] = eta_q_G10 - 2*(326.59863237109*DV_sat_G10 - 0.8*R_v_G10*i_sq_G10 - 0.8*X_v_G10*i_sd_G10 + 326.59863237109)/v_dc_G10
        struct[0].g[98,0] = -eta_D_G10 + eta_d_G10*cos(phi_G10) - eta_q_G10*sin(phi_G10)
        struct[0].g[99,0] = -eta_Q_G10 + eta_d_G10*sin(phi_G10) + eta_q_G10*cos(phi_G10)
        struct[0].g[100,0] = v_mD_G10*cos(phi_G10) + v_mQ_G10*sin(phi_G10) - v_md_G10
        struct[0].g[101,0] = -v_mD_G10*sin(phi_G10) + v_mQ_G10*cos(phi_G10) - v_mq_G10
        struct[0].g[102,0] = v_sD_G10*cos(phi_G10) + v_sQ_G10*sin(phi_G10) - v_sd_G10
        struct[0].g[103,0] = -v_sD_G10*sin(phi_G10) + v_sQ_G10*cos(phi_G10) - v_sq_G10
        struct[0].g[104,0] = i_tD_G10*cos(phi_G10) + i_tQ_G10*sin(phi_G10) - i_td_G10
        struct[0].g[105,0] = -i_tD_G10*sin(phi_G10) + i_tQ_G10*cos(phi_G10) - i_tq_G10
        struct[0].g[106,0] = i_sD_G10*cos(phi_G10) + i_sQ_G10*sin(phi_G10) - i_sd_G10
        struct[0].g[107,0] = -i_sD_G10*sin(phi_G10) + i_sQ_G10*cos(phi_G10) - i_sq_G10
        struct[0].g[108,0] = DV_sat_G10 - K_q_G10*(-q_s_pu_G10 + q_s_ref_G10 + xi_q_G10/T_q_G10)
        struct[0].g[109,0] = 7.5e-6*i_sd_G10*v_sd_G10 + 7.5e-6*i_sq_G10*v_sq_G10 - p_s_pu_G10
        struct[0].g[110,0] = 7.5e-6*i_sd_G10*v_sq_G10 - 7.5e-6*i_sq_G10*v_sd_G10 - q_s_pu_G10
        struct[0].g[111,0] = K_f_G10*(omega_ref_G10 - omega_v_filt_G10) + K_f_sec*xi_f_sec/2 - p_m_ref_G10 + p_r_G10
        struct[0].g[112,0] = K_vpoi_G10*(-v_s_filt_G10 + v_s_ref_G10) + q_r_G10 - q_s_ref_G10
        struct[0].g[113,0] = (L_t_G14*i_tQ_G14*omega_G14 - R_t_G14*i_tD_G14 + eta_D_G14*v_dc_G14/2 - v_mD_G14)/L_t_G14
        struct[0].g[114,0] = (-L_t_G14*i_tD_G14*omega_G14 - R_t_G14*i_tQ_G14 + eta_Q_G14*v_dc_G14/2 - v_mQ_G14)/L_t_G14
        struct[0].g[115,0] = (C_m_G14*omega_G14*v_mQ_G14 - G_d_G14*v_mD_G14 - i_sD_G14 + i_tD_G14)/C_m_G14
        struct[0].g[116,0] = (-C_m_G14*omega_G14*v_mD_G14 - G_d_G14*v_mQ_G14 - i_sQ_G14 + i_tQ_G14)/C_m_G14
        struct[0].g[117,0] = (L_s_G14*i_sQ_G14*omega_G14 - R_s_G14*i_sD_G14 + v_mD_G14 - v_sD_G14)/L_s_G14
        struct[0].g[118,0] = (-L_s_G14*i_sD_G14*omega_G14 - R_s_G14*i_sQ_G14 + v_mQ_G14 - v_sQ_G14)/L_s_G14
        struct[0].g[119,0] = -i_R14_D + i_sD_G14
        struct[0].g[120,0] = -i_R14_Q + i_sQ_G14
        struct[0].g[121,0] = -v_R14_D + v_sD_G14
        struct[0].g[122,0] = -v_R14_Q + v_sQ_G14
        struct[0].g[123,0] = eta_d_G14 - 2*(-0.8*R_v_G14*i_sd_G14 + 0.8*X_v_G14*i_sq_G14)/v_dc_G14
        struct[0].g[124,0] = eta_q_G14 - 2*(326.59863237109*DV_sat_G14 - 0.8*R_v_G14*i_sq_G14 - 0.8*X_v_G14*i_sd_G14 + 326.59863237109)/v_dc_G14
        struct[0].g[125,0] = -eta_D_G14 + eta_d_G14*cos(phi_G14) - eta_q_G14*sin(phi_G14)
        struct[0].g[126,0] = -eta_Q_G14 + eta_d_G14*sin(phi_G14) + eta_q_G14*cos(phi_G14)
        struct[0].g[127,0] = v_mD_G14*cos(phi_G14) + v_mQ_G14*sin(phi_G14) - v_md_G14
        struct[0].g[128,0] = -v_mD_G14*sin(phi_G14) + v_mQ_G14*cos(phi_G14) - v_mq_G14
        struct[0].g[129,0] = v_sD_G14*cos(phi_G14) + v_sQ_G14*sin(phi_G14) - v_sd_G14
        struct[0].g[130,0] = -v_sD_G14*sin(phi_G14) + v_sQ_G14*cos(phi_G14) - v_sq_G14
        struct[0].g[131,0] = i_tD_G14*cos(phi_G14) + i_tQ_G14*sin(phi_G14) - i_td_G14
        struct[0].g[132,0] = -i_tD_G14*sin(phi_G14) + i_tQ_G14*cos(phi_G14) - i_tq_G14
        struct[0].g[133,0] = i_sD_G14*cos(phi_G14) + i_sQ_G14*sin(phi_G14) - i_sd_G14
        struct[0].g[134,0] = -i_sD_G14*sin(phi_G14) + i_sQ_G14*cos(phi_G14) - i_sq_G14
        struct[0].g[135,0] = DV_sat_G14 - K_q_G14*(-q_s_pu_G14 + q_s_ref_G14 + xi_q_G14/T_q_G14)
        struct[0].g[136,0] = 7.5e-6*i_sd_G14*v_sd_G14 + 7.5e-6*i_sq_G14*v_sq_G14 - p_s_pu_G14
        struct[0].g[137,0] = 7.5e-6*i_sd_G14*v_sq_G14 - 7.5e-6*i_sq_G14*v_sd_G14 - q_s_pu_G14
        struct[0].g[138,0] = K_f_G14*(omega_ref_G14 - omega_v_filt_G14) + K_f_sec*xi_f_sec/2 - p_m_ref_G14 + p_r_G14
        struct[0].g[139,0] = K_vpoi_G14*(-v_s_filt_G14 + v_s_ref_G14) + q_r_G14 - q_s_ref_G14
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = i_sD_G10
        struct[0].h[1,0] = i_sQ_G10
        struct[0].h[2,0] = i_sD_G14
        struct[0].h[3,0] = i_sQ_G14
    

    if mode == 10:

        struct[0].Fx[0,0] = -100.000000000000
        struct[0].Fx[1,1] = -100.000000000000
        struct[0].Fx[2,2] = -100.000000000000
        struct[0].Fx[3,3] = -100.000000000000
        struct[0].Fx[4,4] = -100.000000000000
        struct[0].Fx[5,5] = -100.000000000000
        struct[0].Fx[6,6] = -100.000000000000
        struct[0].Fx[7,7] = -100.000000000000
        struct[0].Fx[8,8] = -100.000000000000
        struct[0].Fx[9,9] = -100.000000000000
        struct[0].Fx[10,10] = -100.000000000000
        struct[0].Fx[11,11] = -100.000000000000
        struct[0].Fx[12,12] = -K_phi_G10
        struct[0].Fx[12,13] = -314.159265358979*H_G10/(H_G10 + H_G14) + 314.159265358979
        struct[0].Fx[12,19] = -314.159265358979*H_G14/(H_G10 + H_G14)
        struct[0].Fx[13,13] = -D_G10/(2*H_G10)
        struct[0].Fx[15,13] = 314.159265358979
        struct[0].Fx[15,15] = -1.00000000000000
        struct[0].Fx[16,13] = 1/T_f_G10
        struct[0].Fx[16,16] = -1/T_f_G10
        struct[0].Fx[17,17] = -1/T_vpoi_G10
        struct[0].Fx[18,13] = -314.159265358979*H_G10/(H_G10 + H_G14)
        struct[0].Fx[18,18] = -K_phi_G14
        struct[0].Fx[18,19] = -314.159265358979*H_G14/(H_G10 + H_G14) + 314.159265358979
        struct[0].Fx[19,19] = -D_G14/(2*H_G14)
        struct[0].Fx[21,19] = 314.159265358979
        struct[0].Fx[21,21] = -1.00000000000000
        struct[0].Fx[22,19] = 1/T_f_G14
        struct[0].Fx[22,22] = -1/T_f_G14
        struct[0].Fx[23,23] = -1/T_vpoi_G14
        struct[0].Fx[24,13] = -H_G10/(H_G10 + H_G14)
        struct[0].Fx[24,19] = -H_G14/(H_G10 + H_G14)

    if mode == 11:

        struct[0].Fy[0,74] = 100.0*Piecewise(np.array([(0, (I_max_R01 < i_R01_d_ref) | (I_max_R01 < -i_R01_d_ref)), (1, True)]))
        struct[0].Fy[1,75] = 100.0*Piecewise(np.array([(0, (I_max_R01 < i_R01_q_ref) | (I_max_R01 < -i_R01_q_ref)), (1, True)]))
        struct[0].Fy[2,76] = 100.0*Piecewise(np.array([(0, (I_max_R11 < i_R11_d_ref) | (I_max_R11 < -i_R11_d_ref)), (1, True)]))
        struct[0].Fy[3,77] = 100.0*Piecewise(np.array([(0, (I_max_R11 < i_R11_q_ref) | (I_max_R11 < -i_R11_q_ref)), (1, True)]))
        struct[0].Fy[4,78] = 100.0*Piecewise(np.array([(0, (I_max_R15 < i_R15_d_ref) | (I_max_R15 < -i_R15_d_ref)), (1, True)]))
        struct[0].Fy[5,79] = 100.0*Piecewise(np.array([(0, (I_max_R15 < i_R15_q_ref) | (I_max_R15 < -i_R15_q_ref)), (1, True)]))
        struct[0].Fy[6,80] = 100.0*Piecewise(np.array([(0, (I_max_R16 < i_R16_d_ref) | (I_max_R16 < -i_R16_d_ref)), (1, True)]))
        struct[0].Fy[7,81] = 100.0*Piecewise(np.array([(0, (I_max_R16 < i_R16_q_ref) | (I_max_R16 < -i_R16_q_ref)), (1, True)]))
        struct[0].Fy[8,82] = 100.0*Piecewise(np.array([(0, (I_max_R17 < i_R17_d_ref) | (I_max_R17 < -i_R17_d_ref)), (1, True)]))
        struct[0].Fy[9,83] = 100.0*Piecewise(np.array([(0, (I_max_R17 < i_R17_q_ref) | (I_max_R17 < -i_R17_q_ref)), (1, True)]))
        struct[0].Fy[10,84] = 100.0*Piecewise(np.array([(0, (I_max_R18 < i_R18_d_ref) | (I_max_R18 < -i_R18_d_ref)), (1, True)]))
        struct[0].Fy[11,85] = 100.0*Piecewise(np.array([(0, (I_max_R18 < i_R18_q_ref) | (I_max_R18 < -i_R18_q_ref)), (1, True)]))
        struct[0].Fy[13,109] = -1/(2*H_G10)
        struct[0].Fy[13,111] = 1/(2*H_G10)
        struct[0].Fy[14,110] = -1
        struct[0].Fy[14,112] = 1
        struct[0].Fy[17,102] = 0.00306186217847897*v_sd_G10/(T_vpoi_G10*(v_sd_G10**2 + v_sq_G10**2)**0.5)
        struct[0].Fy[17,103] = 0.00306186217847897*v_sq_G10/(T_vpoi_G10*(v_sd_G10**2 + v_sq_G10**2)**0.5)
        struct[0].Fy[19,136] = -1/(2*H_G14)
        struct[0].Fy[19,138] = 1/(2*H_G14)
        struct[0].Fy[20,137] = -1
        struct[0].Fy[20,139] = 1
        struct[0].Fy[23,129] = 0.00306186217847897*v_sd_G14/(T_vpoi_G14*(v_sd_G14**2 + v_sq_G14**2)**0.5)
        struct[0].Fy[23,130] = 0.00306186217847897*v_sq_G14/(T_vpoi_G14*(v_sd_G14**2 + v_sq_G14**2)**0.5)

        struct[0].Gy[0,0] = -R_R00R01
        struct[0].Gy[0,1] = L_R00R01*omega
        struct[0].Gy[0,36] = 1
        struct[0].Gy[0,38] = -1
        struct[0].Gy[1,0] = -L_R00R01*omega
        struct[0].Gy[1,1] = -R_R00R01
        struct[0].Gy[1,37] = 1
        struct[0].Gy[1,39] = -1
        struct[0].Gy[2,2] = -R_R02R01
        struct[0].Gy[2,3] = L_R02R01*omega
        struct[0].Gy[2,38] = -1
        struct[0].Gy[2,40] = 1
        struct[0].Gy[3,2] = -L_R02R01*omega
        struct[0].Gy[3,3] = -R_R02R01
        struct[0].Gy[3,39] = -1
        struct[0].Gy[3,41] = 1
        struct[0].Gy[4,4] = -R_R02R03
        struct[0].Gy[4,5] = L_R02R03*omega
        struct[0].Gy[4,40] = 1
        struct[0].Gy[4,42] = -1
        struct[0].Gy[5,4] = -L_R02R03*omega
        struct[0].Gy[5,5] = -R_R02R03
        struct[0].Gy[5,41] = 1
        struct[0].Gy[5,43] = -1
        struct[0].Gy[6,6] = -R_R03R04
        struct[0].Gy[6,7] = L_R03R04*omega
        struct[0].Gy[6,42] = 1
        struct[0].Gy[6,44] = -1
        struct[0].Gy[7,6] = -L_R03R04*omega
        struct[0].Gy[7,7] = -R_R03R04
        struct[0].Gy[7,43] = 1
        struct[0].Gy[7,45] = -1
        struct[0].Gy[8,8] = -R_R04R05
        struct[0].Gy[8,9] = L_R04R05*omega
        struct[0].Gy[8,44] = 1
        struct[0].Gy[8,46] = -1
        struct[0].Gy[9,8] = -L_R04R05*omega
        struct[0].Gy[9,9] = -R_R04R05
        struct[0].Gy[9,45] = 1
        struct[0].Gy[9,47] = -1
        struct[0].Gy[10,10] = -R_R04R12
        struct[0].Gy[10,11] = L_R04R12*omega
        struct[0].Gy[10,44] = 1
        struct[0].Gy[10,60] = -1
        struct[0].Gy[11,10] = -L_R04R12*omega
        struct[0].Gy[11,11] = -R_R04R12
        struct[0].Gy[11,45] = 1
        struct[0].Gy[11,61] = -1
        struct[0].Gy[12,12] = -R_R05R06
        struct[0].Gy[12,13] = L_R05R06*omega
        struct[0].Gy[12,46] = 1
        struct[0].Gy[12,48] = -1
        struct[0].Gy[13,12] = -L_R05R06*omega
        struct[0].Gy[13,13] = -R_R05R06
        struct[0].Gy[13,47] = 1
        struct[0].Gy[13,49] = -1
        struct[0].Gy[14,14] = -R_R06R07
        struct[0].Gy[14,15] = L_R06R07*omega
        struct[0].Gy[14,48] = 1
        struct[0].Gy[14,50] = -1
        struct[0].Gy[15,14] = -L_R06R07*omega
        struct[0].Gy[15,15] = -R_R06R07
        struct[0].Gy[15,49] = 1
        struct[0].Gy[15,51] = -1
        struct[0].Gy[16,16] = -R_R07R08
        struct[0].Gy[16,17] = L_R07R08*omega
        struct[0].Gy[16,50] = 1
        struct[0].Gy[16,52] = -1
        struct[0].Gy[17,16] = -L_R07R08*omega
        struct[0].Gy[17,17] = -R_R07R08
        struct[0].Gy[17,51] = 1
        struct[0].Gy[17,53] = -1
        struct[0].Gy[18,18] = -R_R08R09
        struct[0].Gy[18,19] = L_R08R09*omega
        struct[0].Gy[18,52] = 1
        struct[0].Gy[18,54] = -1
        struct[0].Gy[19,18] = -L_R08R09*omega
        struct[0].Gy[19,19] = -R_R08R09
        struct[0].Gy[19,53] = 1
        struct[0].Gy[19,55] = -1
        struct[0].Gy[20,20] = -R_R09R10
        struct[0].Gy[20,21] = L_R09R10*omega
        struct[0].Gy[20,54] = 1
        struct[0].Gy[20,56] = -1
        struct[0].Gy[21,20] = -L_R09R10*omega
        struct[0].Gy[21,21] = -R_R09R10
        struct[0].Gy[21,55] = 1
        struct[0].Gy[21,57] = -1
        struct[0].Gy[22,22] = -R_R09R17
        struct[0].Gy[22,23] = L_R09R17*omega
        struct[0].Gy[22,54] = 1
        struct[0].Gy[22,70] = -1
        struct[0].Gy[23,22] = -L_R09R17*omega
        struct[0].Gy[23,23] = -R_R09R17
        struct[0].Gy[23,55] = 1
        struct[0].Gy[23,71] = -1
        struct[0].Gy[24,24] = -R_R11R03
        struct[0].Gy[24,25] = L_R11R03*omega
        struct[0].Gy[24,42] = -1
        struct[0].Gy[24,58] = 1
        struct[0].Gy[25,24] = -L_R11R03*omega
        struct[0].Gy[25,25] = -R_R11R03
        struct[0].Gy[25,43] = -1
        struct[0].Gy[25,59] = 1
        struct[0].Gy[26,26] = -R_R12R13
        struct[0].Gy[26,27] = L_R12R13*omega
        struct[0].Gy[26,60] = 1
        struct[0].Gy[26,62] = -1
        struct[0].Gy[27,26] = -L_R12R13*omega
        struct[0].Gy[27,27] = -R_R12R13
        struct[0].Gy[27,61] = 1
        struct[0].Gy[27,63] = -1
        struct[0].Gy[28,28] = -R_R13R14
        struct[0].Gy[28,29] = L_R13R14*omega
        struct[0].Gy[28,62] = 1
        struct[0].Gy[28,64] = -1
        struct[0].Gy[29,28] = -L_R13R14*omega
        struct[0].Gy[29,29] = -R_R13R14
        struct[0].Gy[29,63] = 1
        struct[0].Gy[29,65] = -1
        struct[0].Gy[30,30] = -R_R14R15
        struct[0].Gy[30,31] = L_R14R15*omega
        struct[0].Gy[30,64] = 1
        struct[0].Gy[30,66] = -1
        struct[0].Gy[31,30] = -L_R14R15*omega
        struct[0].Gy[31,31] = -R_R14R15
        struct[0].Gy[31,65] = 1
        struct[0].Gy[31,67] = -1
        struct[0].Gy[32,32] = -R_R16R06
        struct[0].Gy[32,33] = L_R16R06*omega
        struct[0].Gy[32,48] = -1
        struct[0].Gy[32,68] = 1
        struct[0].Gy[33,32] = -L_R16R06*omega
        struct[0].Gy[33,33] = -R_R16R06
        struct[0].Gy[33,49] = -1
        struct[0].Gy[33,69] = 1
        struct[0].Gy[34,34] = -R_R18R10
        struct[0].Gy[34,35] = L_R18R10*omega
        struct[0].Gy[34,56] = -1
        struct[0].Gy[34,72] = 1
        struct[0].Gy[35,34] = -L_R18R10*omega
        struct[0].Gy[35,35] = -R_R18R10
        struct[0].Gy[35,57] = -1
        struct[0].Gy[35,73] = 1
        struct[0].Gy[36,0] = -1
        struct[0].Gy[36,37] = C_R00R01*omega/2
        struct[0].Gy[37,1] = -1
        struct[0].Gy[37,36] = -C_R00R01*omega/2
        struct[0].Gy[38,0] = 1
        struct[0].Gy[38,2] = 1
        struct[0].Gy[38,39] = omega*(C_R00R01/2 + C_R02R01/2)
        struct[0].Gy[39,1] = 1
        struct[0].Gy[39,3] = 1
        struct[0].Gy[39,38] = -omega*(C_R00R01/2 + C_R02R01/2)
        struct[0].Gy[40,2] = -1
        struct[0].Gy[40,4] = -1
        struct[0].Gy[40,41] = omega*(C_R02R01/2 + C_R02R03/2)
        struct[0].Gy[41,3] = -1
        struct[0].Gy[41,5] = -1
        struct[0].Gy[41,40] = -omega*(C_R02R01/2 + C_R02R03/2)
        struct[0].Gy[42,4] = 1
        struct[0].Gy[42,6] = -1
        struct[0].Gy[42,24] = 1
        struct[0].Gy[42,43] = omega*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].Gy[43,5] = 1
        struct[0].Gy[43,7] = -1
        struct[0].Gy[43,25] = 1
        struct[0].Gy[43,42] = -omega*(C_R02R03/2 + C_R03R04/2 + C_R11R03/2)
        struct[0].Gy[44,6] = 1
        struct[0].Gy[44,8] = -1
        struct[0].Gy[44,10] = -1
        struct[0].Gy[44,45] = omega*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].Gy[45,7] = 1
        struct[0].Gy[45,9] = -1
        struct[0].Gy[45,11] = -1
        struct[0].Gy[45,44] = -omega*(C_R03R04/2 + C_R04R05/2 + C_R04R12/2)
        struct[0].Gy[46,8] = 1
        struct[0].Gy[46,12] = -1
        struct[0].Gy[46,47] = omega*(C_R04R05/2 + C_R05R06/2)
        struct[0].Gy[47,9] = 1
        struct[0].Gy[47,13] = -1
        struct[0].Gy[47,46] = -omega*(C_R04R05/2 + C_R05R06/2)
        struct[0].Gy[48,12] = 1
        struct[0].Gy[48,14] = -1
        struct[0].Gy[48,32] = 1
        struct[0].Gy[48,49] = omega*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].Gy[49,13] = 1
        struct[0].Gy[49,15] = -1
        struct[0].Gy[49,33] = 1
        struct[0].Gy[49,48] = -omega*(C_R05R06/2 + C_R06R07/2 + C_R16R06/2)
        struct[0].Gy[50,14] = 1
        struct[0].Gy[50,16] = -1
        struct[0].Gy[50,51] = omega*(C_R06R07/2 + C_R07R08/2)
        struct[0].Gy[51,15] = 1
        struct[0].Gy[51,17] = -1
        struct[0].Gy[51,50] = -omega*(C_R06R07/2 + C_R07R08/2)
        struct[0].Gy[52,16] = 1
        struct[0].Gy[52,18] = -1
        struct[0].Gy[52,53] = omega*(C_R07R08/2 + C_R08R09/2)
        struct[0].Gy[53,17] = 1
        struct[0].Gy[53,19] = -1
        struct[0].Gy[53,52] = -omega*(C_R07R08/2 + C_R08R09/2)
        struct[0].Gy[54,18] = 1
        struct[0].Gy[54,20] = -1
        struct[0].Gy[54,22] = -1
        struct[0].Gy[54,55] = omega*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].Gy[55,19] = 1
        struct[0].Gy[55,21] = -1
        struct[0].Gy[55,23] = -1
        struct[0].Gy[55,54] = -omega*(C_R08R09/2 + C_R09R10/2 + C_R09R17/2)
        struct[0].Gy[56,20] = 1
        struct[0].Gy[56,34] = 1
        struct[0].Gy[56,57] = omega*(C_R09R10/2 + C_R18R10/2)
        struct[0].Gy[56,92] = 1
        struct[0].Gy[57,21] = 1
        struct[0].Gy[57,35] = 1
        struct[0].Gy[57,56] = -omega*(C_R09R10/2 + C_R18R10/2)
        struct[0].Gy[57,93] = 1
        struct[0].Gy[58,24] = -1
        struct[0].Gy[58,59] = C_R11R03*omega/2
        struct[0].Gy[59,25] = -1
        struct[0].Gy[59,58] = -C_R11R03*omega/2
        struct[0].Gy[60,10] = 1
        struct[0].Gy[60,26] = -1
        struct[0].Gy[60,61] = omega*(C_R04R12/2 + C_R12R13/2)
        struct[0].Gy[61,11] = 1
        struct[0].Gy[61,27] = -1
        struct[0].Gy[61,60] = -omega*(C_R04R12/2 + C_R12R13/2)
        struct[0].Gy[62,26] = 1
        struct[0].Gy[62,28] = -1
        struct[0].Gy[62,63] = omega*(C_R12R13/2 + C_R13R14/2)
        struct[0].Gy[63,27] = 1
        struct[0].Gy[63,29] = -1
        struct[0].Gy[63,62] = -omega*(C_R12R13/2 + C_R13R14/2)
        struct[0].Gy[64,28] = 1
        struct[0].Gy[64,30] = -1
        struct[0].Gy[64,65] = omega*(C_R13R14/2 + C_R14R15/2)
        struct[0].Gy[64,119] = 1
        struct[0].Gy[65,29] = 1
        struct[0].Gy[65,31] = -1
        struct[0].Gy[65,64] = -omega*(C_R13R14/2 + C_R14R15/2)
        struct[0].Gy[65,120] = 1
        struct[0].Gy[66,30] = 1
        struct[0].Gy[66,67] = C_R14R15*omega/2
        struct[0].Gy[67,31] = 1
        struct[0].Gy[67,66] = -C_R14R15*omega/2
        struct[0].Gy[68,32] = -1
        struct[0].Gy[68,69] = C_R16R06*omega/2
        struct[0].Gy[69,33] = -1
        struct[0].Gy[69,68] = -C_R16R06*omega/2
        struct[0].Gy[70,22] = 1
        struct[0].Gy[70,71] = C_R09R17*omega/2
        struct[0].Gy[71,23] = 1
        struct[0].Gy[71,70] = -C_R09R17*omega/2
        struct[0].Gy[72,34] = -1
        struct[0].Gy[72,73] = C_R18R10*omega/2
        struct[0].Gy[73,35] = -1
        struct[0].Gy[73,72] = -C_R18R10*omega/2
        struct[0].Gy[74,38] = -0.666666666666667*p_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_D - 0.666666666666667*q_R01_ref*v_R01_Q)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_D/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy[74,39] = -0.666666666666667*q_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_D - 0.666666666666667*q_R01_ref*v_R01_Q)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_Q/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy[74,74] = -1
        struct[0].Gy[75,38] = 0.666666666666667*q_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_Q + 0.666666666666667*q_R01_ref*v_R01_D)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_D/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy[75,39] = -0.666666666666667*p_R01_ref*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)])) + (-0.666666666666667*p_R01_ref*v_R01_Q + 0.666666666666667*q_R01_ref*v_R01_D)*Piecewise(np.array([(0, (v_R01_D**2 + v_R01_Q**2 > 1000000000000.0) | (v_R01_D**2 + v_R01_Q**2 < 0.01)), (-2*v_R01_Q/(v_R01_D**2 + v_R01_Q**2)**2, True)]))
        struct[0].Gy[75,75] = -1
        struct[0].Gy[76,58] = -0.666666666666667*p_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_D - 0.666666666666667*q_R11_ref*v_R11_Q)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_D/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy[76,59] = -0.666666666666667*q_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_D - 0.666666666666667*q_R11_ref*v_R11_Q)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_Q/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy[76,76] = -1
        struct[0].Gy[77,58] = 0.666666666666667*q_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_Q + 0.666666666666667*q_R11_ref*v_R11_D)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_D/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy[77,59] = -0.666666666666667*p_R11_ref*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)])) + (-0.666666666666667*p_R11_ref*v_R11_Q + 0.666666666666667*q_R11_ref*v_R11_D)*Piecewise(np.array([(0, (v_R11_D**2 + v_R11_Q**2 > 1000000000000.0) | (v_R11_D**2 + v_R11_Q**2 < 0.01)), (-2*v_R11_Q/(v_R11_D**2 + v_R11_Q**2)**2, True)]))
        struct[0].Gy[77,77] = -1
        struct[0].Gy[78,66] = -0.666666666666667*p_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_D - 0.666666666666667*q_R15_ref*v_R15_Q)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_D/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy[78,67] = -0.666666666666667*q_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_D - 0.666666666666667*q_R15_ref*v_R15_Q)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_Q/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy[78,78] = -1
        struct[0].Gy[79,66] = 0.666666666666667*q_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_Q + 0.666666666666667*q_R15_ref*v_R15_D)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_D/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy[79,67] = -0.666666666666667*p_R15_ref*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)])) + (-0.666666666666667*p_R15_ref*v_R15_Q + 0.666666666666667*q_R15_ref*v_R15_D)*Piecewise(np.array([(0, (v_R15_D**2 + v_R15_Q**2 > 1000000000000.0) | (v_R15_D**2 + v_R15_Q**2 < 0.01)), (-2*v_R15_Q/(v_R15_D**2 + v_R15_Q**2)**2, True)]))
        struct[0].Gy[79,79] = -1
        struct[0].Gy[80,68] = -0.666666666666667*p_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_D - 0.666666666666667*q_R16_ref*v_R16_Q)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_D/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy[80,69] = -0.666666666666667*q_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_D - 0.666666666666667*q_R16_ref*v_R16_Q)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_Q/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy[80,80] = -1
        struct[0].Gy[81,68] = 0.666666666666667*q_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_Q + 0.666666666666667*q_R16_ref*v_R16_D)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_D/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy[81,69] = -0.666666666666667*p_R16_ref*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)])) + (-0.666666666666667*p_R16_ref*v_R16_Q + 0.666666666666667*q_R16_ref*v_R16_D)*Piecewise(np.array([(0, (v_R16_D**2 + v_R16_Q**2 > 1000000000000.0) | (v_R16_D**2 + v_R16_Q**2 < 0.01)), (-2*v_R16_Q/(v_R16_D**2 + v_R16_Q**2)**2, True)]))
        struct[0].Gy[81,81] = -1
        struct[0].Gy[82,70] = -0.666666666666667*p_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_D - 0.666666666666667*q_R17_ref*v_R17_Q)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_D/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy[82,71] = -0.666666666666667*q_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_D - 0.666666666666667*q_R17_ref*v_R17_Q)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_Q/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy[82,82] = -1
        struct[0].Gy[83,70] = 0.666666666666667*q_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_Q + 0.666666666666667*q_R17_ref*v_R17_D)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_D/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy[83,71] = -0.666666666666667*p_R17_ref*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)])) + (-0.666666666666667*p_R17_ref*v_R17_Q + 0.666666666666667*q_R17_ref*v_R17_D)*Piecewise(np.array([(0, (v_R17_D**2 + v_R17_Q**2 > 1000000000000.0) | (v_R17_D**2 + v_R17_Q**2 < 0.01)), (-2*v_R17_Q/(v_R17_D**2 + v_R17_Q**2)**2, True)]))
        struct[0].Gy[83,83] = -1
        struct[0].Gy[84,72] = -0.666666666666667*p_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_D - 0.666666666666667*q_R18_ref*v_R18_Q)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_D/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy[84,73] = -0.666666666666667*q_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_D - 0.666666666666667*q_R18_ref*v_R18_Q)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_Q/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy[84,84] = -1
        struct[0].Gy[85,72] = 0.666666666666667*q_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_Q + 0.666666666666667*q_R18_ref*v_R18_D)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_D/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy[85,73] = -0.666666666666667*p_R18_ref*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)])) + (-0.666666666666667*p_R18_ref*v_R18_Q + 0.666666666666667*q_R18_ref*v_R18_D)*Piecewise(np.array([(0, (v_R18_D**2 + v_R18_Q**2 > 1000000000000.0) | (v_R18_D**2 + v_R18_Q**2 < 0.01)), (-2*v_R18_Q/(v_R18_D**2 + v_R18_Q**2)**2, True)]))
        struct[0].Gy[85,85] = -1
        struct[0].Gy[86,86] = -R_t_G10/L_t_G10
        struct[0].Gy[86,87] = omega_G10
        struct[0].Gy[86,88] = -1/L_t_G10
        struct[0].Gy[86,98] = v_dc_G10/(2*L_t_G10)
        struct[0].Gy[87,86] = -omega_G10
        struct[0].Gy[87,87] = -R_t_G10/L_t_G10
        struct[0].Gy[87,89] = -1/L_t_G10
        struct[0].Gy[87,99] = v_dc_G10/(2*L_t_G10)
        struct[0].Gy[88,86] = 1/C_m_G10
        struct[0].Gy[88,88] = -G_d_G10/C_m_G10
        struct[0].Gy[88,89] = omega_G10
        struct[0].Gy[88,90] = -1/C_m_G10
        struct[0].Gy[89,87] = 1/C_m_G10
        struct[0].Gy[89,88] = -omega_G10
        struct[0].Gy[89,89] = -G_d_G10/C_m_G10
        struct[0].Gy[89,91] = -1/C_m_G10
        struct[0].Gy[90,88] = 1/L_s_G10
        struct[0].Gy[90,90] = -R_s_G10/L_s_G10
        struct[0].Gy[90,91] = omega_G10
        struct[0].Gy[90,94] = -1/L_s_G10
        struct[0].Gy[91,89] = 1/L_s_G10
        struct[0].Gy[91,90] = -omega_G10
        struct[0].Gy[91,91] = -R_s_G10/L_s_G10
        struct[0].Gy[91,95] = -1/L_s_G10
        struct[0].Gy[92,90] = 1
        struct[0].Gy[92,92] = -1
        struct[0].Gy[93,91] = 1
        struct[0].Gy[93,93] = -1
        struct[0].Gy[94,56] = -1
        struct[0].Gy[94,94] = 1
        struct[0].Gy[95,57] = -1
        struct[0].Gy[95,95] = 1
        struct[0].Gy[96,96] = 1
        struct[0].Gy[96,106] = 1.6*R_v_G10/v_dc_G10
        struct[0].Gy[96,107] = -1.6*X_v_G10/v_dc_G10
        struct[0].Gy[97,97] = 1
        struct[0].Gy[97,106] = 1.6*X_v_G10/v_dc_G10
        struct[0].Gy[97,107] = 1.6*R_v_G10/v_dc_G10
        struct[0].Gy[97,108] = -653.197264742181/v_dc_G10
        struct[0].Gy[98,96] = cos(phi_G10)
        struct[0].Gy[98,97] = -sin(phi_G10)
        struct[0].Gy[98,98] = -1
        struct[0].Gy[99,96] = sin(phi_G10)
        struct[0].Gy[99,97] = cos(phi_G10)
        struct[0].Gy[99,99] = -1
        struct[0].Gy[100,88] = cos(phi_G10)
        struct[0].Gy[100,89] = sin(phi_G10)
        struct[0].Gy[100,100] = -1
        struct[0].Gy[101,88] = -sin(phi_G10)
        struct[0].Gy[101,89] = cos(phi_G10)
        struct[0].Gy[101,101] = -1
        struct[0].Gy[102,94] = cos(phi_G10)
        struct[0].Gy[102,95] = sin(phi_G10)
        struct[0].Gy[102,102] = -1
        struct[0].Gy[103,94] = -sin(phi_G10)
        struct[0].Gy[103,95] = cos(phi_G10)
        struct[0].Gy[103,103] = -1
        struct[0].Gy[104,86] = cos(phi_G10)
        struct[0].Gy[104,87] = sin(phi_G10)
        struct[0].Gy[104,104] = -1
        struct[0].Gy[105,86] = -sin(phi_G10)
        struct[0].Gy[105,87] = cos(phi_G10)
        struct[0].Gy[105,105] = -1
        struct[0].Gy[106,90] = cos(phi_G10)
        struct[0].Gy[106,91] = sin(phi_G10)
        struct[0].Gy[106,106] = -1
        struct[0].Gy[107,90] = -sin(phi_G10)
        struct[0].Gy[107,91] = cos(phi_G10)
        struct[0].Gy[107,107] = -1
        struct[0].Gy[108,108] = 1
        struct[0].Gy[108,110] = K_q_G10
        struct[0].Gy[108,112] = -K_q_G10
        struct[0].Gy[109,102] = 7.5e-6*i_sd_G10
        struct[0].Gy[109,103] = 7.5e-6*i_sq_G10
        struct[0].Gy[109,106] = 7.5e-6*v_sd_G10
        struct[0].Gy[109,107] = 7.5e-6*v_sq_G10
        struct[0].Gy[109,109] = -1
        struct[0].Gy[110,102] = -7.5e-6*i_sq_G10
        struct[0].Gy[110,103] = 7.5e-6*i_sd_G10
        struct[0].Gy[110,106] = 7.5e-6*v_sq_G10
        struct[0].Gy[110,107] = -7.5e-6*v_sd_G10
        struct[0].Gy[110,110] = -1
        struct[0].Gy[111,111] = -1
        struct[0].Gy[112,112] = -1
        struct[0].Gy[113,113] = -R_t_G14/L_t_G14
        struct[0].Gy[113,114] = omega_G14
        struct[0].Gy[113,115] = -1/L_t_G14
        struct[0].Gy[113,125] = v_dc_G14/(2*L_t_G14)
        struct[0].Gy[114,113] = -omega_G14
        struct[0].Gy[114,114] = -R_t_G14/L_t_G14
        struct[0].Gy[114,116] = -1/L_t_G14
        struct[0].Gy[114,126] = v_dc_G14/(2*L_t_G14)
        struct[0].Gy[115,113] = 1/C_m_G14
        struct[0].Gy[115,115] = -G_d_G14/C_m_G14
        struct[0].Gy[115,116] = omega_G14
        struct[0].Gy[115,117] = -1/C_m_G14
        struct[0].Gy[116,114] = 1/C_m_G14
        struct[0].Gy[116,115] = -omega_G14
        struct[0].Gy[116,116] = -G_d_G14/C_m_G14
        struct[0].Gy[116,118] = -1/C_m_G14
        struct[0].Gy[117,115] = 1/L_s_G14
        struct[0].Gy[117,117] = -R_s_G14/L_s_G14
        struct[0].Gy[117,118] = omega_G14
        struct[0].Gy[117,121] = -1/L_s_G14
        struct[0].Gy[118,116] = 1/L_s_G14
        struct[0].Gy[118,117] = -omega_G14
        struct[0].Gy[118,118] = -R_s_G14/L_s_G14
        struct[0].Gy[118,122] = -1/L_s_G14
        struct[0].Gy[119,117] = 1
        struct[0].Gy[119,119] = -1
        struct[0].Gy[120,118] = 1
        struct[0].Gy[120,120] = -1
        struct[0].Gy[121,64] = -1
        struct[0].Gy[121,121] = 1
        struct[0].Gy[122,65] = -1
        struct[0].Gy[122,122] = 1
        struct[0].Gy[123,123] = 1
        struct[0].Gy[123,133] = 1.6*R_v_G14/v_dc_G14
        struct[0].Gy[123,134] = -1.6*X_v_G14/v_dc_G14
        struct[0].Gy[124,124] = 1
        struct[0].Gy[124,133] = 1.6*X_v_G14/v_dc_G14
        struct[0].Gy[124,134] = 1.6*R_v_G14/v_dc_G14
        struct[0].Gy[124,135] = -653.197264742181/v_dc_G14
        struct[0].Gy[125,123] = cos(phi_G14)
        struct[0].Gy[125,124] = -sin(phi_G14)
        struct[0].Gy[125,125] = -1
        struct[0].Gy[126,123] = sin(phi_G14)
        struct[0].Gy[126,124] = cos(phi_G14)
        struct[0].Gy[126,126] = -1
        struct[0].Gy[127,115] = cos(phi_G14)
        struct[0].Gy[127,116] = sin(phi_G14)
        struct[0].Gy[127,127] = -1
        struct[0].Gy[128,115] = -sin(phi_G14)
        struct[0].Gy[128,116] = cos(phi_G14)
        struct[0].Gy[128,128] = -1
        struct[0].Gy[129,121] = cos(phi_G14)
        struct[0].Gy[129,122] = sin(phi_G14)
        struct[0].Gy[129,129] = -1
        struct[0].Gy[130,121] = -sin(phi_G14)
        struct[0].Gy[130,122] = cos(phi_G14)
        struct[0].Gy[130,130] = -1
        struct[0].Gy[131,113] = cos(phi_G14)
        struct[0].Gy[131,114] = sin(phi_G14)
        struct[0].Gy[131,131] = -1
        struct[0].Gy[132,113] = -sin(phi_G14)
        struct[0].Gy[132,114] = cos(phi_G14)
        struct[0].Gy[132,132] = -1
        struct[0].Gy[133,117] = cos(phi_G14)
        struct[0].Gy[133,118] = sin(phi_G14)
        struct[0].Gy[133,133] = -1
        struct[0].Gy[134,117] = -sin(phi_G14)
        struct[0].Gy[134,118] = cos(phi_G14)
        struct[0].Gy[134,134] = -1
        struct[0].Gy[135,135] = 1
        struct[0].Gy[135,137] = K_q_G14
        struct[0].Gy[135,139] = -K_q_G14
        struct[0].Gy[136,129] = 7.5e-6*i_sd_G14
        struct[0].Gy[136,130] = 7.5e-6*i_sq_G14
        struct[0].Gy[136,133] = 7.5e-6*v_sd_G14
        struct[0].Gy[136,134] = 7.5e-6*v_sq_G14
        struct[0].Gy[136,136] = -1
        struct[0].Gy[137,129] = -7.5e-6*i_sq_G14
        struct[0].Gy[137,130] = 7.5e-6*i_sd_G14
        struct[0].Gy[137,133] = 7.5e-6*v_sq_G14
        struct[0].Gy[137,134] = -7.5e-6*v_sd_G14
        struct[0].Gy[137,137] = -1
        struct[0].Gy[138,138] = -1
        struct[0].Gy[139,139] = -1

        struct[0].Gu[74,2] = -0.666666666666667*v_R01_D*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)]))
        struct[0].Gu[74,3] = -0.666666666666667*v_R01_Q*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)]))
        struct[0].Gu[75,2] = -0.666666666666667*v_R01_Q*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)]))
        struct[0].Gu[75,3] = 0.666666666666667*v_R01_D*Piecewise(np.array([(100.0, v_R01_D**2 + v_R01_Q**2 < 0.01), (1.0e-12, v_R01_D**2 + v_R01_Q**2 > 1000000000000.0), (1/(v_R01_D**2 + v_R01_Q**2), True)]))
        struct[0].Gu[76,6] = -0.666666666666667*v_R11_D*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)]))
        struct[0].Gu[76,7] = -0.666666666666667*v_R11_Q*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)]))
        struct[0].Gu[77,6] = -0.666666666666667*v_R11_Q*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)]))
        struct[0].Gu[77,7] = 0.666666666666667*v_R11_D*Piecewise(np.array([(100.0, v_R11_D**2 + v_R11_Q**2 < 0.01), (1.0e-12, v_R11_D**2 + v_R11_Q**2 > 1000000000000.0), (1/(v_R11_D**2 + v_R11_Q**2), True)]))
        struct[0].Gu[78,10] = -0.666666666666667*v_R15_D*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)]))
        struct[0].Gu[78,11] = -0.666666666666667*v_R15_Q*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)]))
        struct[0].Gu[79,10] = -0.666666666666667*v_R15_Q*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)]))
        struct[0].Gu[79,11] = 0.666666666666667*v_R15_D*Piecewise(np.array([(100.0, v_R15_D**2 + v_R15_Q**2 < 0.01), (1.0e-12, v_R15_D**2 + v_R15_Q**2 > 1000000000000.0), (1/(v_R15_D**2 + v_R15_Q**2), True)]))
        struct[0].Gu[80,14] = -0.666666666666667*v_R16_D*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)]))
        struct[0].Gu[80,15] = -0.666666666666667*v_R16_Q*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)]))
        struct[0].Gu[81,14] = -0.666666666666667*v_R16_Q*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)]))
        struct[0].Gu[81,15] = 0.666666666666667*v_R16_D*Piecewise(np.array([(100.0, v_R16_D**2 + v_R16_Q**2 < 0.01), (1.0e-12, v_R16_D**2 + v_R16_Q**2 > 1000000000000.0), (1/(v_R16_D**2 + v_R16_Q**2), True)]))
        struct[0].Gu[82,18] = -0.666666666666667*v_R17_D*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)]))
        struct[0].Gu[82,19] = -0.666666666666667*v_R17_Q*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)]))
        struct[0].Gu[83,18] = -0.666666666666667*v_R17_Q*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)]))
        struct[0].Gu[83,19] = 0.666666666666667*v_R17_D*Piecewise(np.array([(100.0, v_R17_D**2 + v_R17_Q**2 < 0.01), (1.0e-12, v_R17_D**2 + v_R17_Q**2 > 1000000000000.0), (1/(v_R17_D**2 + v_R17_Q**2), True)]))
        struct[0].Gu[84,22] = -0.666666666666667*v_R18_D*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)]))
        struct[0].Gu[84,23] = -0.666666666666667*v_R18_Q*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)]))
        struct[0].Gu[85,22] = -0.666666666666667*v_R18_Q*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)]))
        struct[0].Gu[85,23] = 0.666666666666667*v_R18_D*Piecewise(np.array([(100.0, v_R18_D**2 + v_R18_Q**2 < 0.01), (1.0e-12, v_R18_D**2 + v_R18_Q**2 > 1000000000000.0), (1/(v_R18_D**2 + v_R18_Q**2), True)]))
        struct[0].Gu[86,24] = eta_D_G10/(2*L_t_G10)
        struct[0].Gu[87,24] = eta_Q_G10/(2*L_t_G10)
        struct[0].Gu[96,24] = 2*(-0.8*R_v_G10*i_sd_G10 + 0.8*X_v_G10*i_sq_G10)/v_dc_G10**2
        struct[0].Gu[97,24] = 2*(326.59863237109*DV_sat_G10 - 0.8*R_v_G10*i_sq_G10 - 0.8*X_v_G10*i_sd_G10 + 326.59863237109)/v_dc_G10**2
        struct[0].Gu[108,26] = -K_q_G10
        struct[0].Gu[111,25] = -1
        struct[0].Gu[111,28] = K_f_G10
        struct[0].Gu[111,29] = 1
        struct[0].Gu[112,26] = -1
        struct[0].Gu[112,27] = K_vpoi_G10
        struct[0].Gu[112,30] = 1
        struct[0].Gu[113,31] = eta_D_G14/(2*L_t_G14)
        struct[0].Gu[114,31] = eta_Q_G14/(2*L_t_G14)
        struct[0].Gu[123,31] = 2*(-0.8*R_v_G14*i_sd_G14 + 0.8*X_v_G14*i_sq_G14)/v_dc_G14**2
        struct[0].Gu[124,31] = 2*(326.59863237109*DV_sat_G14 - 0.8*R_v_G14*i_sq_G14 - 0.8*X_v_G14*i_sd_G14 + 326.59863237109)/v_dc_G14**2
        struct[0].Gu[135,33] = -K_q_G14
        struct[0].Gu[138,32] = -1
        struct[0].Gu[138,35] = K_f_G14
        struct[0].Gu[138,36] = 1
        struct[0].Gu[139,33] = -1
        struct[0].Gu[139,34] = K_vpoi_G14
        struct[0].Gu[139,37] = 1





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
def daesolver(struct): 
    sin = np.sin
    cos = np.cos
    sqrt = np.sqrt
    i = 0 
    
    Dt = struct[i].Dt 

    N_x = struct[i].N_x
    N_y = struct[i].N_y
    N_z = struct[i].N_z

    decimation = struct[i].decimation 
    eye = np.eye(N_x)
    t = struct[i].t 
    t_end = struct[i].t_end 
    if struct[i].it == 0:
        run(t,struct, 1) 
        struct[i].it_store = 0  
        struct[i]['T'][0] = t 
        struct[i].X[0,:] = struct[i].x[:,0]  
        struct[i].Y[0,:] = struct[i].y_run[:,0]  
        struct[i].Z[0,:] = struct[i].h[:,0]  

    solver = struct[i].solvern 
    while t<t_end: 
        struct[i].it += 1
        struct[i].t += Dt
        
        t = struct[i].t


            
        if solver == 5: # Teapezoidal DAE as in Milano's book

            run(t,struct, 2) 
            run(t,struct, 3) 

            x = np.copy(struct[i].x[:]) 
            y = np.copy(struct[i].y_run[:]) 
            f = np.copy(struct[i].f[:]) 
            g = np.copy(struct[i].g[:]) 
            
            for iter in range(struct[i].imax):
                run(t,struct, 2) 
                run(t,struct, 3) 
                run(t,struct,10) 
                run(t,struct,11) 
                
                x_i = struct[i].x[:] 
                y_i = struct[i].y_run[:]  
                f_i = struct[i].f[:] 
                g_i = struct[i].g[:]                 
                F_x_i = struct[i].Fx[:,:]
                F_y_i = struct[i].Fy[:,:] 
                G_x_i = struct[i].Gx[:,:] 
                G_y_i = struct[i].Gy[:,:]                

                A_c_i = np.vstack((np.hstack((eye-0.5*Dt*F_x_i, -0.5*Dt*F_y_i)),
                                   np.hstack((G_x_i,         G_y_i))))
                     
                f_n_i = x_i - x - 0.5*Dt*(f_i+f) 
                # print(t,iter,g_i)
                Dxy_i = np.linalg.solve(-A_c_i,np.vstack((f_n_i,g_i))) 
                
                x_i = x_i + Dxy_i[0:N_x]
                y_i = y_i + Dxy_i[N_x:(N_x+N_y)]

                struct[i].x[:] = x_i
                struct[i].y_run[:] = y_i

        # [f_i,g_i,F_x_i,F_y_i,G_x_i,G_y_i] =  smib_transient(x_i,y_i,u);
        
        # A_c_i = [[eye(N_x)-0.5*Dt*F_x_i, -0.5*Dt*F_y_i],
        #          [                G_x_i,         G_y_i]];
             
        # f_n_i = x_i - x - 0.5*Dt*(f_i+f);
        
        # Dxy_i = -A_c_i\[f_n_i.',g_i.'].';
        
        # x_i = x_i + Dxy_i(1:N_x);
        # y_i = y_i + Dxy_i(N_x+1:N_x+N_y);
                
                xy = np.vstack((x_i,y_i))
                max_relative = 0.0
                for it_var in range(N_x+N_y):
                    abs_value = np.abs(xy[it_var,0])
                    if abs_value < 0.001:
                        abs_value = 0.001
                                             
                    relative_error = np.abs(Dxy_i[it_var,0])/abs_value
                    
                    if relative_error > max_relative: max_relative = relative_error
                    
                if max_relative<struct[i].itol:
                    
                    break
                
                # if iter>struct[i].imax-2:
                    
                #     print('Convergence problem')

            struct[i].x[:] = x_i
            struct[i].y_run[:] = y_i
                
        # channels 
        if struct[i].store == 1:
            it_store = struct[i].it_store
            if struct[i].it >= it_store*decimation: 
                struct[i]['T'][it_store+1] = t 
                struct[i].X[it_store+1,:] = struct[i].x[:,0] 
                struct[i].Y[it_store+1,:] = struct[i].y_run[:,0]
                struct[i].Z[it_store+1,:] = struct[i].h[:,0]
                struct[i].iters[it_store+1,0] = iter
                struct[i].it_store += 1 
            
    struct[i].t = t

    return t





def nonzeros():
    Fx_ini_rows = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 12, 13, 15, 15, 16, 16, 17, 18, 18, 18, 19, 21, 21, 22, 22, 23, 24, 24]

    Fx_ini_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 19, 13, 13, 15, 13, 16, 17, 13, 18, 19, 19, 19, 21, 19, 22, 23, 13, 19]

    Fy_ini_rows = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 13, 14, 14, 17, 17, 19, 19, 20, 20, 23, 23]

    Fy_ini_cols = [74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 109, 111, 110, 112, 102, 103, 136, 138, 137, 139, 129, 130]

    Gx_ini_rows = [38, 39, 58, 59, 66, 67, 68, 69, 70, 71, 72, 73, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 111, 111, 112, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 138, 138, 139]

    Gx_ini_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 14, 16, 24, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 20, 22, 24, 23]

    Gy_ini_rows = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14, 15, 15, 15, 15, 16, 16, 16, 16, 17, 17, 17, 17, 18, 18, 18, 18, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 24, 25, 25, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27, 28, 28, 28, 28, 29, 29, 29, 29, 30, 30, 30, 30, 31, 31, 31, 31, 32, 32, 32, 32, 33, 33, 33, 33, 34, 34, 34, 34, 35, 35, 35, 35, 36, 36, 37, 37, 38, 38, 38, 39, 39, 39, 40, 40, 40, 41, 41, 41, 42, 42, 42, 42, 43, 43, 43, 43, 44, 44, 44, 44, 45, 45, 45, 45, 46, 46, 46, 47, 47, 47, 48, 48, 48, 48, 49, 49, 49, 49, 50, 50, 50, 51, 51, 51, 52, 52, 52, 53, 53, 53, 54, 54, 54, 54, 55, 55, 55, 55, 56, 56, 56, 56, 57, 57, 57, 57, 58, 58, 59, 59, 60, 60, 60, 61, 61, 61, 62, 62, 62, 63, 63, 63, 64, 64, 64, 64, 65, 65, 65, 65, 66, 66, 67, 67, 68, 68, 69, 69, 70, 70, 71, 71, 72, 72, 73, 73, 74, 74, 74, 75, 75, 75, 76, 76, 76, 77, 77, 77, 78, 78, 78, 79, 79, 79, 80, 80, 80, 81, 81, 81, 82, 82, 82, 83, 83, 83, 84, 84, 84, 85, 85, 85, 86, 86, 86, 86, 87, 87, 87, 87, 88, 88, 88, 88, 89, 89, 89, 89, 90, 90, 90, 90, 91, 91, 91, 91, 92, 92, 93, 93, 94, 94, 95, 95, 96, 96, 96, 97, 97, 97, 97, 98, 98, 98, 99, 99, 99, 100, 100, 100, 101, 101, 101, 102, 102, 102, 103, 103, 103, 104, 104, 104, 105, 105, 105, 106, 106, 106, 107, 107, 107, 108, 108, 108, 109, 109, 109, 109, 109, 110, 110, 110, 110, 110, 111, 112, 113, 113, 113, 113, 114, 114, 114, 114, 115, 115, 115, 115, 116, 116, 116, 116, 117, 117, 117, 117, 118, 118, 118, 118, 119, 119, 120, 120, 121, 121, 122, 122, 123, 123, 123, 124, 124, 124, 124, 125, 125, 125, 126, 126, 126, 127, 127, 127, 128, 128, 128, 129, 129, 129, 130, 130, 130, 131, 131, 131, 132, 132, 132, 133, 133, 133, 134, 134, 134, 135, 135, 135, 136, 136, 136, 136, 136, 137, 137, 137, 137, 137, 138, 139]

    Gy_ini_cols = [0, 1, 36, 38, 0, 1, 37, 39, 2, 3, 38, 40, 2, 3, 39, 41, 4, 5, 40, 42, 4, 5, 41, 43, 6, 7, 42, 44, 6, 7, 43, 45, 8, 9, 44, 46, 8, 9, 45, 47, 10, 11, 44, 60, 10, 11, 45, 61, 12, 13, 46, 48, 12, 13, 47, 49, 14, 15, 48, 50, 14, 15, 49, 51, 16, 17, 50, 52, 16, 17, 51, 53, 18, 19, 52, 54, 18, 19, 53, 55, 20, 21, 54, 56, 20, 21, 55, 57, 22, 23, 54, 70, 22, 23, 55, 71, 24, 25, 42, 58, 24, 25, 43, 59, 26, 27, 60, 62, 26, 27, 61, 63, 28, 29, 62, 64, 28, 29, 63, 65, 30, 31, 64, 66, 30, 31, 65, 67, 32, 33, 48, 68, 32, 33, 49, 69, 34, 35, 56, 72, 34, 35, 57, 73, 0, 37, 1, 36, 0, 2, 39, 1, 3, 38, 2, 4, 41, 3, 5, 40, 4, 6, 24, 43, 5, 7, 25, 42, 6, 8, 10, 45, 7, 9, 11, 44, 8, 12, 47, 9, 13, 46, 12, 14, 32, 49, 13, 15, 33, 48, 14, 16, 51, 15, 17, 50, 16, 18, 53, 17, 19, 52, 18, 20, 22, 55, 19, 21, 23, 54, 20, 34, 57, 92, 21, 35, 56, 93, 24, 59, 25, 58, 10, 26, 61, 11, 27, 60, 26, 28, 63, 27, 29, 62, 28, 30, 65, 119, 29, 31, 64, 120, 30, 67, 31, 66, 32, 69, 33, 68, 22, 71, 23, 70, 34, 73, 35, 72, 38, 39, 74, 38, 39, 75, 58, 59, 76, 58, 59, 77, 66, 67, 78, 66, 67, 79, 68, 69, 80, 68, 69, 81, 70, 71, 82, 70, 71, 83, 72, 73, 84, 72, 73, 85, 86, 87, 88, 98, 86, 87, 89, 99, 86, 88, 89, 90, 87, 88, 89, 91, 88, 90, 91, 94, 89, 90, 91, 95, 90, 92, 91, 93, 56, 94, 57, 95, 96, 106, 107, 97, 106, 107, 108, 96, 97, 98, 96, 97, 99, 88, 89, 100, 88, 89, 101, 94, 95, 102, 94, 95, 103, 86, 87, 104, 86, 87, 105, 90, 91, 106, 90, 91, 107, 108, 110, 112, 102, 103, 106, 107, 109, 102, 103, 106, 107, 110, 111, 112, 113, 114, 115, 125, 113, 114, 116, 126, 113, 115, 116, 117, 114, 115, 116, 118, 115, 117, 118, 121, 116, 117, 118, 122, 117, 119, 118, 120, 64, 121, 65, 122, 123, 133, 134, 124, 133, 134, 135, 123, 124, 125, 123, 124, 126, 115, 116, 127, 115, 116, 128, 121, 122, 129, 121, 122, 130, 113, 114, 131, 113, 114, 132, 117, 118, 133, 117, 118, 134, 135, 137, 139, 129, 130, 133, 134, 136, 129, 130, 133, 134, 137, 138, 139]

    return Fx_ini_rows,Fx_ini_cols,Fy_ini_rows,Fy_ini_cols,Gx_ini_rows,Gx_ini_cols,Gy_ini_rows,Gy_ini_cols