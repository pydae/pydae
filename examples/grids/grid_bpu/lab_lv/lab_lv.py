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


class lab_lv_class: 

    def __init__(self): 

        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 24
        self.N_y = 76 
        self.N_z = 31 
        self.N_store = 10000 
        self.params_list = ['S_base', 'g_01_02', 'b_01_02', 'bs_01_02', 'g_02_03', 'b_02_03', 'bs_02_03', 'g_03_04', 'b_03_04', 'bs_03_04', 'g_04_05', 'b_04_05', 'bs_04_05', 'g_05_06', 'b_05_06', 'bs_05_06', 'g_06_07', 'b_06_07', 'bs_06_07', 'g_07_08', 'b_07_08', 'bs_07_08', 'g_06_09', 'b_06_09', 'bs_06_09', 'g_05_10', 'b_05_10', 'bs_05_10', 'g_04_11', 'b_04_11', 'bs_04_11', 'g_03_12', 'b_03_12', 'bs_03_12', 'U_01_n', 'U_02_n', 'U_03_n', 'U_04_n', 'U_05_n', 'U_06_n', 'U_07_n', 'U_08_n', 'U_09_n', 'U_10_n', 'U_11_n', 'U_12_n', 'S_n_08', 'Omega_b_08', 'K_p_08', 'T_p_08', 'K_q_08', 'T_q_08', 'X_v_08', 'R_v_08', 'R_s_08', 'C_u_08', 'K_u_0_08', 'K_u_max_08', 'V_u_min_08', 'V_u_max_08', 'R_uc_08', 'K_h_08', 'R_lim_08', 'V_u_lt_08', 'V_u_ht_08', 'Droop_08', 'DB_08', 'T_cur_08', 'R_lim_max_08', 'K_fpfr_08', 'P_f_min_08', 'P_f_max_08', 'S_n_09', 'Omega_b_09', 'K_p_09', 'T_p_09', 'K_q_09', 'T_q_09', 'X_v_09', 'R_v_09', 'R_s_09', 'C_u_09', 'K_u_0_09', 'K_u_max_09', 'V_u_min_09', 'V_u_max_09', 'R_uc_09', 'K_h_09', 'R_lim_09', 'V_u_lt_09', 'V_u_ht_09', 'Droop_09', 'DB_09', 'T_cur_09', 'R_lim_max_09', 'K_fpfr_09', 'P_f_min_09', 'P_f_max_09', 'S_n_10', 'Omega_b_10', 'K_p_10', 'T_p_10', 'K_q_10', 'T_q_10', 'X_v_10', 'R_v_10', 'R_s_10', 'C_u_10', 'K_u_0_10', 'K_u_max_10', 'V_u_min_10', 'V_u_max_10', 'R_uc_10', 'K_h_10', 'R_lim_10', 'V_u_lt_10', 'V_u_ht_10', 'Droop_10', 'DB_10', 'T_cur_10', 'R_lim_max_10', 'K_fpfr_10', 'P_f_min_10', 'P_f_max_10', 'S_n_01', 'Omega_b_01', 'X_v_01', 'R_v_01', 'K_delta_01', 'K_alpha_01', 'K_p_agc', 'K_i_agc'] 
        self.params_values_list  = [100000.0, 64.70588235294117, -258.8235294117647, 0.0, 12.131762250617438, -7.801776366956552, 0.0, 24.929645136934585, -16.03404488983458, 0.0, 13.092406546203273, -7.469203734601868, 0.0, 7.524731931424944, -4.834827759261077, 0.0, 25.903011891971992, -16.656170118178785, 0.0, 9.286163883367264, -1.2568153246401323, 0.0, 8.942349014359865, -1.2113142091164757, 0.0, 8.890876379647214, -1.207589321514053, 0.0, 2.9405790658137216, -0.39846139690232724, 0.0, 7.947033843480717, -1.0772913859650137, 0.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 20000.0, 314.1592653589793, 0.01, 0.1, 0.1, 0.1, 0.1, 0.01, 0.02, 5.0, 0.005, 0.1, 80, 160, 0.1, 1.0, 0.2, 85, 155, 0.05, 0.001, 10.0, 100.0, 0.0, -1.0, 1.0, 20000.0, 314.1592653589793, 0.01, 0.1, 0.1, 0.1, 0.1, 0.01, 0.02, 5.0, 0.005, 0.1, 80, 160, 0.1, 1.0, 0.2, 85, 155, 0.05, 0.001, 10.0, 100.0, 0.0, -1.0, 1.0, 20000.0, 314.1592653589793, 0.01, 0.1, 0.1, 0.1, 0.1, 0.01, 0.02, 5.0, 0.005, 0.1, 80, 160, 0.1, 1.0, 0.2, 85, 155, 0.05, 0.001, 10.0, 100.0, 0.0, -1.0, 1.0, 100000.0, 314.1592653589793, 0.001, 0.001, 0.001, 1e-06, 0.01, 0.01] 
        self.inputs_ini_list = ['P_01', 'Q_01', 'P_02', 'Q_02', 'P_03', 'Q_03', 'P_04', 'Q_04', 'P_05', 'Q_05', 'P_06', 'Q_06', 'P_07', 'Q_07', 'P_08', 'Q_08', 'P_09', 'Q_09', 'P_10', 'Q_10', 'P_11', 'Q_11', 'P_12', 'Q_12', 'q_s_ref_08', 'v_u_ref_08', 'omega_ref_08', 'p_gin_0_08', 'p_g_ref_08', 'ramp_p_gin_08', 'q_s_ref_09', 'v_u_ref_09', 'omega_ref_09', 'p_gin_0_09', 'p_g_ref_09', 'ramp_p_gin_09', 'q_s_ref_10', 'v_u_ref_10', 'omega_ref_10', 'p_gin_0_10', 'p_g_ref_10', 'ramp_p_gin_10', 'alpha_01', 'e_qv_01', 'omega_ref_01'] 
        self.inputs_ini_values_list  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 126.0, 1.0, 0.6, 0.4, 0.0, 0.0, 126.0, 1.0, 0.6, 0.4, 0.0, 0.0, 126.0, 1.0, 0.6, 0.4, 0.0, 0, 1, 1] 
        self.inputs_run_list = ['P_01', 'Q_01', 'P_02', 'Q_02', 'P_03', 'Q_03', 'P_04', 'Q_04', 'P_05', 'Q_05', 'P_06', 'Q_06', 'P_07', 'Q_07', 'P_08', 'Q_08', 'P_09', 'Q_09', 'P_10', 'Q_10', 'P_11', 'Q_11', 'P_12', 'Q_12', 'q_s_ref_08', 'v_u_ref_08', 'omega_ref_08', 'p_gin_0_08', 'p_g_ref_08', 'ramp_p_gin_08', 'q_s_ref_09', 'v_u_ref_09', 'omega_ref_09', 'p_gin_0_09', 'p_g_ref_09', 'ramp_p_gin_09', 'q_s_ref_10', 'v_u_ref_10', 'omega_ref_10', 'p_gin_0_10', 'p_g_ref_10', 'ramp_p_gin_10', 'alpha_01', 'e_qv_01', 'omega_ref_01'] 
        self.inputs_run_values_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 126.0, 1.0, 0.6, 0.4, 0.0, 0.0, 126.0, 1.0, 0.6, 0.4, 0.0, 0.0, 126.0, 1.0, 0.6, 0.4, 0.0, 0, 1, 1] 
        self.outputs_list = ['V_01', 'V_02', 'V_03', 'V_04', 'V_05', 'V_06', 'V_07', 'V_08', 'V_09', 'V_10', 'V_11', 'V_12', 'p_gin_08', 'p_g_ref_08', 'p_l_08', 'soc_08', 'p_fpfr_08', 'p_f_sat_08', 'p_gin_09', 'p_g_ref_09', 'p_l_09', 'soc_09', 'p_fpfr_09', 'p_f_sat_09', 'p_gin_10', 'p_g_ref_10', 'p_l_10', 'soc_10', 'p_fpfr_10', 'p_f_sat_10', 'alpha_01'] 
        self.x_list = ['delta_08', 'xi_p_08', 'xi_q_08', 'e_u_08', 'p_ghr_08', 'k_cur_08', 'inc_p_gin_08', 'delta_09', 'xi_p_09', 'xi_q_09', 'e_u_09', 'p_ghr_09', 'k_cur_09', 'inc_p_gin_09', 'delta_10', 'xi_p_10', 'xi_q_10', 'e_u_10', 'p_ghr_10', 'k_cur_10', 'inc_p_gin_10', 'delta_01', 'Domega_01', 'xi_freq'] 
        self.y_run_list = ['V_01', 'theta_01', 'V_02', 'theta_02', 'V_03', 'theta_03', 'V_04', 'theta_04', 'V_05', 'theta_05', 'V_06', 'theta_06', 'V_07', 'theta_07', 'V_08', 'theta_08', 'V_09', 'theta_09', 'V_10', 'theta_10', 'V_11', 'theta_11', 'V_12', 'theta_12', 'omega_08', 'e_qv_08', 'i_d_08', 'i_q_08', 'p_s_08', 'q_s_08', 'p_m_08', 'p_t_08', 'p_u_08', 'v_u_08', 'k_u_08', 'k_cur_sat_08', 'p_gou_08', 'p_f_08', 'r_lim_08', 'omega_09', 'e_qv_09', 'i_d_09', 'i_q_09', 'p_s_09', 'q_s_09', 'p_m_09', 'p_t_09', 'p_u_09', 'v_u_09', 'k_u_09', 'k_cur_sat_09', 'p_gou_09', 'p_f_09', 'r_lim_09', 'omega_10', 'e_qv_10', 'i_d_10', 'i_q_10', 'p_s_10', 'q_s_10', 'p_m_10', 'p_t_10', 'p_u_10', 'v_u_10', 'k_u_10', 'k_cur_sat_10', 'p_gou_10', 'p_f_10', 'r_lim_10', 'omega_01', 'i_d_01', 'i_q_01', 'p_s_01', 'q_s_01', 'omega_coi', 'p_agc'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['V_01', 'theta_01', 'V_02', 'theta_02', 'V_03', 'theta_03', 'V_04', 'theta_04', 'V_05', 'theta_05', 'V_06', 'theta_06', 'V_07', 'theta_07', 'V_08', 'theta_08', 'V_09', 'theta_09', 'V_10', 'theta_10', 'V_11', 'theta_11', 'V_12', 'theta_12', 'omega_08', 'e_qv_08', 'i_d_08', 'i_q_08', 'p_s_08', 'q_s_08', 'p_m_08', 'p_t_08', 'p_u_08', 'v_u_08', 'k_u_08', 'k_cur_sat_08', 'p_gou_08', 'p_f_08', 'r_lim_08', 'omega_09', 'e_qv_09', 'i_d_09', 'i_q_09', 'p_s_09', 'q_s_09', 'p_m_09', 'p_t_09', 'p_u_09', 'v_u_09', 'k_u_09', 'k_cur_sat_09', 'p_gou_09', 'p_f_09', 'r_lim_09', 'omega_10', 'e_qv_10', 'i_d_10', 'i_q_10', 'p_s_10', 'q_s_10', 'p_m_10', 'p_t_10', 'p_u_10', 'v_u_10', 'k_u_10', 'k_cur_sat_10', 'p_gou_10', 'p_f_10', 'r_lim_10', 'omega_01', 'i_d_01', 'i_q_01', 'p_s_01', 'q_s_01', 'omega_coi', 'p_agc'] 
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
    S_base = struct[0].S_base
    g_01_02 = struct[0].g_01_02
    b_01_02 = struct[0].b_01_02
    bs_01_02 = struct[0].bs_01_02
    g_02_03 = struct[0].g_02_03
    b_02_03 = struct[0].b_02_03
    bs_02_03 = struct[0].bs_02_03
    g_03_04 = struct[0].g_03_04
    b_03_04 = struct[0].b_03_04
    bs_03_04 = struct[0].bs_03_04
    g_04_05 = struct[0].g_04_05
    b_04_05 = struct[0].b_04_05
    bs_04_05 = struct[0].bs_04_05
    g_05_06 = struct[0].g_05_06
    b_05_06 = struct[0].b_05_06
    bs_05_06 = struct[0].bs_05_06
    g_06_07 = struct[0].g_06_07
    b_06_07 = struct[0].b_06_07
    bs_06_07 = struct[0].bs_06_07
    g_07_08 = struct[0].g_07_08
    b_07_08 = struct[0].b_07_08
    bs_07_08 = struct[0].bs_07_08
    g_06_09 = struct[0].g_06_09
    b_06_09 = struct[0].b_06_09
    bs_06_09 = struct[0].bs_06_09
    g_05_10 = struct[0].g_05_10
    b_05_10 = struct[0].b_05_10
    bs_05_10 = struct[0].bs_05_10
    g_04_11 = struct[0].g_04_11
    b_04_11 = struct[0].b_04_11
    bs_04_11 = struct[0].bs_04_11
    g_03_12 = struct[0].g_03_12
    b_03_12 = struct[0].b_03_12
    bs_03_12 = struct[0].bs_03_12
    U_01_n = struct[0].U_01_n
    U_02_n = struct[0].U_02_n
    U_03_n = struct[0].U_03_n
    U_04_n = struct[0].U_04_n
    U_05_n = struct[0].U_05_n
    U_06_n = struct[0].U_06_n
    U_07_n = struct[0].U_07_n
    U_08_n = struct[0].U_08_n
    U_09_n = struct[0].U_09_n
    U_10_n = struct[0].U_10_n
    U_11_n = struct[0].U_11_n
    U_12_n = struct[0].U_12_n
    S_n_08 = struct[0].S_n_08
    Omega_b_08 = struct[0].Omega_b_08
    K_p_08 = struct[0].K_p_08
    T_p_08 = struct[0].T_p_08
    K_q_08 = struct[0].K_q_08
    T_q_08 = struct[0].T_q_08
    X_v_08 = struct[0].X_v_08
    R_v_08 = struct[0].R_v_08
    R_s_08 = struct[0].R_s_08
    C_u_08 = struct[0].C_u_08
    K_u_0_08 = struct[0].K_u_0_08
    K_u_max_08 = struct[0].K_u_max_08
    V_u_min_08 = struct[0].V_u_min_08
    V_u_max_08 = struct[0].V_u_max_08
    R_uc_08 = struct[0].R_uc_08
    K_h_08 = struct[0].K_h_08
    R_lim_08 = struct[0].R_lim_08
    V_u_lt_08 = struct[0].V_u_lt_08
    V_u_ht_08 = struct[0].V_u_ht_08
    Droop_08 = struct[0].Droop_08
    DB_08 = struct[0].DB_08
    T_cur_08 = struct[0].T_cur_08
    R_lim_max_08 = struct[0].R_lim_max_08
    K_fpfr_08 = struct[0].K_fpfr_08
    P_f_min_08 = struct[0].P_f_min_08
    P_f_max_08 = struct[0].P_f_max_08
    S_n_09 = struct[0].S_n_09
    Omega_b_09 = struct[0].Omega_b_09
    K_p_09 = struct[0].K_p_09
    T_p_09 = struct[0].T_p_09
    K_q_09 = struct[0].K_q_09
    T_q_09 = struct[0].T_q_09
    X_v_09 = struct[0].X_v_09
    R_v_09 = struct[0].R_v_09
    R_s_09 = struct[0].R_s_09
    C_u_09 = struct[0].C_u_09
    K_u_0_09 = struct[0].K_u_0_09
    K_u_max_09 = struct[0].K_u_max_09
    V_u_min_09 = struct[0].V_u_min_09
    V_u_max_09 = struct[0].V_u_max_09
    R_uc_09 = struct[0].R_uc_09
    K_h_09 = struct[0].K_h_09
    R_lim_09 = struct[0].R_lim_09
    V_u_lt_09 = struct[0].V_u_lt_09
    V_u_ht_09 = struct[0].V_u_ht_09
    Droop_09 = struct[0].Droop_09
    DB_09 = struct[0].DB_09
    T_cur_09 = struct[0].T_cur_09
    R_lim_max_09 = struct[0].R_lim_max_09
    K_fpfr_09 = struct[0].K_fpfr_09
    P_f_min_09 = struct[0].P_f_min_09
    P_f_max_09 = struct[0].P_f_max_09
    S_n_10 = struct[0].S_n_10
    Omega_b_10 = struct[0].Omega_b_10
    K_p_10 = struct[0].K_p_10
    T_p_10 = struct[0].T_p_10
    K_q_10 = struct[0].K_q_10
    T_q_10 = struct[0].T_q_10
    X_v_10 = struct[0].X_v_10
    R_v_10 = struct[0].R_v_10
    R_s_10 = struct[0].R_s_10
    C_u_10 = struct[0].C_u_10
    K_u_0_10 = struct[0].K_u_0_10
    K_u_max_10 = struct[0].K_u_max_10
    V_u_min_10 = struct[0].V_u_min_10
    V_u_max_10 = struct[0].V_u_max_10
    R_uc_10 = struct[0].R_uc_10
    K_h_10 = struct[0].K_h_10
    R_lim_10 = struct[0].R_lim_10
    V_u_lt_10 = struct[0].V_u_lt_10
    V_u_ht_10 = struct[0].V_u_ht_10
    Droop_10 = struct[0].Droop_10
    DB_10 = struct[0].DB_10
    T_cur_10 = struct[0].T_cur_10
    R_lim_max_10 = struct[0].R_lim_max_10
    K_fpfr_10 = struct[0].K_fpfr_10
    P_f_min_10 = struct[0].P_f_min_10
    P_f_max_10 = struct[0].P_f_max_10
    S_n_01 = struct[0].S_n_01
    Omega_b_01 = struct[0].Omega_b_01
    X_v_01 = struct[0].X_v_01
    R_v_01 = struct[0].R_v_01
    K_delta_01 = struct[0].K_delta_01
    K_alpha_01 = struct[0].K_alpha_01
    K_p_agc = struct[0].K_p_agc
    K_i_agc = struct[0].K_i_agc
    
    # Inputs:
    P_01 = struct[0].P_01
    Q_01 = struct[0].Q_01
    P_02 = struct[0].P_02
    Q_02 = struct[0].Q_02
    P_03 = struct[0].P_03
    Q_03 = struct[0].Q_03
    P_04 = struct[0].P_04
    Q_04 = struct[0].Q_04
    P_05 = struct[0].P_05
    Q_05 = struct[0].Q_05
    P_06 = struct[0].P_06
    Q_06 = struct[0].Q_06
    P_07 = struct[0].P_07
    Q_07 = struct[0].Q_07
    P_08 = struct[0].P_08
    Q_08 = struct[0].Q_08
    P_09 = struct[0].P_09
    Q_09 = struct[0].Q_09
    P_10 = struct[0].P_10
    Q_10 = struct[0].Q_10
    P_11 = struct[0].P_11
    Q_11 = struct[0].Q_11
    P_12 = struct[0].P_12
    Q_12 = struct[0].Q_12
    q_s_ref_08 = struct[0].q_s_ref_08
    v_u_ref_08 = struct[0].v_u_ref_08
    omega_ref_08 = struct[0].omega_ref_08
    p_gin_0_08 = struct[0].p_gin_0_08
    p_g_ref_08 = struct[0].p_g_ref_08
    ramp_p_gin_08 = struct[0].ramp_p_gin_08
    q_s_ref_09 = struct[0].q_s_ref_09
    v_u_ref_09 = struct[0].v_u_ref_09
    omega_ref_09 = struct[0].omega_ref_09
    p_gin_0_09 = struct[0].p_gin_0_09
    p_g_ref_09 = struct[0].p_g_ref_09
    ramp_p_gin_09 = struct[0].ramp_p_gin_09
    q_s_ref_10 = struct[0].q_s_ref_10
    v_u_ref_10 = struct[0].v_u_ref_10
    omega_ref_10 = struct[0].omega_ref_10
    p_gin_0_10 = struct[0].p_gin_0_10
    p_g_ref_10 = struct[0].p_g_ref_10
    ramp_p_gin_10 = struct[0].ramp_p_gin_10
    alpha_01 = struct[0].alpha_01
    e_qv_01 = struct[0].e_qv_01
    omega_ref_01 = struct[0].omega_ref_01
    
    # Dynamical states:
    delta_08 = struct[0].x[0,0]
    xi_p_08 = struct[0].x[1,0]
    xi_q_08 = struct[0].x[2,0]
    e_u_08 = struct[0].x[3,0]
    p_ghr_08 = struct[0].x[4,0]
    k_cur_08 = struct[0].x[5,0]
    inc_p_gin_08 = struct[0].x[6,0]
    delta_09 = struct[0].x[7,0]
    xi_p_09 = struct[0].x[8,0]
    xi_q_09 = struct[0].x[9,0]
    e_u_09 = struct[0].x[10,0]
    p_ghr_09 = struct[0].x[11,0]
    k_cur_09 = struct[0].x[12,0]
    inc_p_gin_09 = struct[0].x[13,0]
    delta_10 = struct[0].x[14,0]
    xi_p_10 = struct[0].x[15,0]
    xi_q_10 = struct[0].x[16,0]
    e_u_10 = struct[0].x[17,0]
    p_ghr_10 = struct[0].x[18,0]
    k_cur_10 = struct[0].x[19,0]
    inc_p_gin_10 = struct[0].x[20,0]
    delta_01 = struct[0].x[21,0]
    Domega_01 = struct[0].x[22,0]
    xi_freq = struct[0].x[23,0]
    
    # Algebraic states:
    V_01 = struct[0].y_ini[0,0]
    theta_01 = struct[0].y_ini[1,0]
    V_02 = struct[0].y_ini[2,0]
    theta_02 = struct[0].y_ini[3,0]
    V_03 = struct[0].y_ini[4,0]
    theta_03 = struct[0].y_ini[5,0]
    V_04 = struct[0].y_ini[6,0]
    theta_04 = struct[0].y_ini[7,0]
    V_05 = struct[0].y_ini[8,0]
    theta_05 = struct[0].y_ini[9,0]
    V_06 = struct[0].y_ini[10,0]
    theta_06 = struct[0].y_ini[11,0]
    V_07 = struct[0].y_ini[12,0]
    theta_07 = struct[0].y_ini[13,0]
    V_08 = struct[0].y_ini[14,0]
    theta_08 = struct[0].y_ini[15,0]
    V_09 = struct[0].y_ini[16,0]
    theta_09 = struct[0].y_ini[17,0]
    V_10 = struct[0].y_ini[18,0]
    theta_10 = struct[0].y_ini[19,0]
    V_11 = struct[0].y_ini[20,0]
    theta_11 = struct[0].y_ini[21,0]
    V_12 = struct[0].y_ini[22,0]
    theta_12 = struct[0].y_ini[23,0]
    omega_08 = struct[0].y_ini[24,0]
    e_qv_08 = struct[0].y_ini[25,0]
    i_d_08 = struct[0].y_ini[26,0]
    i_q_08 = struct[0].y_ini[27,0]
    p_s_08 = struct[0].y_ini[28,0]
    q_s_08 = struct[0].y_ini[29,0]
    p_m_08 = struct[0].y_ini[30,0]
    p_t_08 = struct[0].y_ini[31,0]
    p_u_08 = struct[0].y_ini[32,0]
    v_u_08 = struct[0].y_ini[33,0]
    k_u_08 = struct[0].y_ini[34,0]
    k_cur_sat_08 = struct[0].y_ini[35,0]
    p_gou_08 = struct[0].y_ini[36,0]
    p_f_08 = struct[0].y_ini[37,0]
    r_lim_08 = struct[0].y_ini[38,0]
    omega_09 = struct[0].y_ini[39,0]
    e_qv_09 = struct[0].y_ini[40,0]
    i_d_09 = struct[0].y_ini[41,0]
    i_q_09 = struct[0].y_ini[42,0]
    p_s_09 = struct[0].y_ini[43,0]
    q_s_09 = struct[0].y_ini[44,0]
    p_m_09 = struct[0].y_ini[45,0]
    p_t_09 = struct[0].y_ini[46,0]
    p_u_09 = struct[0].y_ini[47,0]
    v_u_09 = struct[0].y_ini[48,0]
    k_u_09 = struct[0].y_ini[49,0]
    k_cur_sat_09 = struct[0].y_ini[50,0]
    p_gou_09 = struct[0].y_ini[51,0]
    p_f_09 = struct[0].y_ini[52,0]
    r_lim_09 = struct[0].y_ini[53,0]
    omega_10 = struct[0].y_ini[54,0]
    e_qv_10 = struct[0].y_ini[55,0]
    i_d_10 = struct[0].y_ini[56,0]
    i_q_10 = struct[0].y_ini[57,0]
    p_s_10 = struct[0].y_ini[58,0]
    q_s_10 = struct[0].y_ini[59,0]
    p_m_10 = struct[0].y_ini[60,0]
    p_t_10 = struct[0].y_ini[61,0]
    p_u_10 = struct[0].y_ini[62,0]
    v_u_10 = struct[0].y_ini[63,0]
    k_u_10 = struct[0].y_ini[64,0]
    k_cur_sat_10 = struct[0].y_ini[65,0]
    p_gou_10 = struct[0].y_ini[66,0]
    p_f_10 = struct[0].y_ini[67,0]
    r_lim_10 = struct[0].y_ini[68,0]
    omega_01 = struct[0].y_ini[69,0]
    i_d_01 = struct[0].y_ini[70,0]
    i_q_01 = struct[0].y_ini[71,0]
    p_s_01 = struct[0].y_ini[72,0]
    q_s_01 = struct[0].y_ini[73,0]
    omega_coi = struct[0].y_ini[74,0]
    p_agc = struct[0].y_ini[75,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = Omega_b_08*(omega_08 - omega_coi)
        struct[0].f[1,0] = p_m_08 - p_s_08
        struct[0].f[2,0] = -q_s_08 + q_s_ref_08
        struct[0].f[3,0] = S_n_08*(p_gou_08 - p_t_08)/(C_u_08*(v_u_08 + 0.1))
        struct[0].f[4,0] = Piecewise(np.array([(-r_lim_08, r_lim_08 < -K_h_08*(-p_ghr_08 + p_gou_08)), (r_lim_08, r_lim_08 < K_h_08*(-p_ghr_08 + p_gou_08)), (K_h_08*(-p_ghr_08 + p_gou_08), True)]))
        struct[0].f[5,0] = (-k_cur_08 + p_g_ref_08/(inc_p_gin_08 + p_gin_0_08) + Piecewise(np.array([(P_f_min_08, P_f_min_08 > p_f_08), (P_f_max_08, P_f_max_08 < p_f_08), (p_f_08, True)]))/(inc_p_gin_08 + p_gin_0_08))/T_cur_08
        struct[0].f[6,0] = -0.001*inc_p_gin_08 + ramp_p_gin_08
        struct[0].f[7,0] = Omega_b_09*(omega_09 - omega_coi)
        struct[0].f[8,0] = p_m_09 - p_s_09
        struct[0].f[9,0] = -q_s_09 + q_s_ref_09
        struct[0].f[10,0] = S_n_09*(p_gou_09 - p_t_09)/(C_u_09*(v_u_09 + 0.1))
        struct[0].f[11,0] = Piecewise(np.array([(-r_lim_09, r_lim_09 < -K_h_09*(-p_ghr_09 + p_gou_09)), (r_lim_09, r_lim_09 < K_h_09*(-p_ghr_09 + p_gou_09)), (K_h_09*(-p_ghr_09 + p_gou_09), True)]))
        struct[0].f[12,0] = (-k_cur_09 + p_g_ref_09/(inc_p_gin_09 + p_gin_0_09) + Piecewise(np.array([(P_f_min_09, P_f_min_09 > p_f_09), (P_f_max_09, P_f_max_09 < p_f_09), (p_f_09, True)]))/(inc_p_gin_09 + p_gin_0_09))/T_cur_09
        struct[0].f[13,0] = -0.001*inc_p_gin_09 + ramp_p_gin_09
        struct[0].f[14,0] = Omega_b_10*(omega_10 - omega_coi)
        struct[0].f[15,0] = p_m_10 - p_s_10
        struct[0].f[16,0] = -q_s_10 + q_s_ref_10
        struct[0].f[17,0] = S_n_10*(p_gou_10 - p_t_10)/(C_u_10*(v_u_10 + 0.1))
        struct[0].f[18,0] = Piecewise(np.array([(-r_lim_10, r_lim_10 < -K_h_10*(-p_ghr_10 + p_gou_10)), (r_lim_10, r_lim_10 < K_h_10*(-p_ghr_10 + p_gou_10)), (K_h_10*(-p_ghr_10 + p_gou_10), True)]))
        struct[0].f[19,0] = (-k_cur_10 + p_g_ref_10/(inc_p_gin_10 + p_gin_0_10) + Piecewise(np.array([(P_f_min_10, P_f_min_10 > p_f_10), (P_f_max_10, P_f_max_10 < p_f_10), (p_f_10, True)]))/(inc_p_gin_10 + p_gin_0_10))/T_cur_10
        struct[0].f[20,0] = -0.001*inc_p_gin_10 + ramp_p_gin_10
        struct[0].f[21,0] = -K_delta_01*delta_01 + Omega_b_01*(omega_01 - omega_coi)
        struct[0].f[22,0] = -Domega_01*K_alpha_01 + alpha_01
        struct[0].f[23,0] = 1 - omega_coi
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy_ini) @ np.ascontiguousarray(struct[0].y_ini)

        struct[0].g[0,0] = -P_01/S_base + V_01**2*g_01_02 + V_01*V_02*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) - S_n_01*p_s_01/S_base
        struct[0].g[1,0] = -Q_01/S_base + V_01**2*(-b_01_02 - bs_01_02/2) + V_01*V_02*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02)) - S_n_01*q_s_01/S_base
        struct[0].g[2,0] = -P_02/S_base + V_01*V_02*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) + V_02**2*(g_01_02 + g_02_03) + V_02*V_03*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].g[3,0] = -Q_02/S_base + V_01*V_02*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02)) + V_02**2*(-b_01_02 - b_02_03 - bs_01_02/2 - bs_02_03/2) + V_02*V_03*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].g[4,0] = -P_03/S_base + V_02*V_03*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03)) + V_03**2*(g_02_03 + g_03_04 + g_03_12) + V_03*V_04*(-b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04)) + V_03*V_12*(-b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12))
        struct[0].g[5,0] = -Q_03/S_base + V_02*V_03*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03)) + V_03**2*(-b_02_03 - b_03_04 - b_03_12 - bs_02_03/2 - bs_03_04/2 - bs_03_12/2) + V_03*V_04*(b_03_04*cos(theta_03 - theta_04) - g_03_04*sin(theta_03 - theta_04)) + V_03*V_12*(b_03_12*cos(theta_03 - theta_12) - g_03_12*sin(theta_03 - theta_12))
        struct[0].g[6,0] = -P_04/S_base + V_03*V_04*(b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04)) + V_04**2*(g_03_04 + g_04_05 + g_04_11) + V_04*V_05*(-b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05)) + V_04*V_11*(-b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11))
        struct[0].g[7,0] = -Q_04/S_base + V_03*V_04*(b_03_04*cos(theta_03 - theta_04) + g_03_04*sin(theta_03 - theta_04)) + V_04**2*(-b_03_04 - b_04_05 - b_04_11 - bs_03_04/2 - bs_04_05/2 - bs_04_11/2) + V_04*V_05*(b_04_05*cos(theta_04 - theta_05) - g_04_05*sin(theta_04 - theta_05)) + V_04*V_11*(b_04_11*cos(theta_04 - theta_11) - g_04_11*sin(theta_04 - theta_11))
        struct[0].g[8,0] = -P_05/S_base + V_04*V_05*(b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05)) + V_05**2*(g_04_05 + g_05_06 + g_05_10) + V_05*V_06*(-b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06)) + V_05*V_10*(-b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10))
        struct[0].g[9,0] = -Q_05/S_base + V_04*V_05*(b_04_05*cos(theta_04 - theta_05) + g_04_05*sin(theta_04 - theta_05)) + V_05**2*(-b_04_05 - b_05_06 - b_05_10 - bs_04_05/2 - bs_05_06/2 - bs_05_10/2) + V_05*V_06*(b_05_06*cos(theta_05 - theta_06) - g_05_06*sin(theta_05 - theta_06)) + V_05*V_10*(b_05_10*cos(theta_05 - theta_10) - g_05_10*sin(theta_05 - theta_10))
        struct[0].g[10,0] = -P_06/S_base + V_05*V_06*(b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06)) + V_06**2*(g_05_06 + g_06_07 + g_06_09) + V_06*V_07*(-b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07)) + V_06*V_09*(-b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09))
        struct[0].g[11,0] = -Q_06/S_base + V_05*V_06*(b_05_06*cos(theta_05 - theta_06) + g_05_06*sin(theta_05 - theta_06)) + V_06**2*(-b_05_06 - b_06_07 - b_06_09 - bs_05_06/2 - bs_06_07/2 - bs_06_09/2) + V_06*V_07*(b_06_07*cos(theta_06 - theta_07) - g_06_07*sin(theta_06 - theta_07)) + V_06*V_09*(b_06_09*cos(theta_06 - theta_09) - g_06_09*sin(theta_06 - theta_09))
        struct[0].g[12,0] = -P_07/S_base + V_06*V_07*(b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07)) + V_07**2*(g_06_07 + g_07_08) + V_07*V_08*(-b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08))
        struct[0].g[13,0] = -Q_07/S_base + V_06*V_07*(b_06_07*cos(theta_06 - theta_07) + g_06_07*sin(theta_06 - theta_07)) + V_07**2*(-b_06_07 - b_07_08 - bs_06_07/2 - bs_07_08/2) + V_07*V_08*(b_07_08*cos(theta_07 - theta_08) - g_07_08*sin(theta_07 - theta_08))
        struct[0].g[14,0] = -P_08/S_base + V_07*V_08*(b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08)) + V_08**2*g_07_08 - S_n_08*p_s_08/S_base
        struct[0].g[15,0] = -Q_08/S_base + V_07*V_08*(b_07_08*cos(theta_07 - theta_08) + g_07_08*sin(theta_07 - theta_08)) + V_08**2*(-b_07_08 - bs_07_08/2) - S_n_08*q_s_08/S_base
        struct[0].g[16,0] = -P_09/S_base + V_06*V_09*(b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09)) + V_09**2*g_06_09 - S_n_09*p_s_09/S_base
        struct[0].g[17,0] = -Q_09/S_base + V_06*V_09*(b_06_09*cos(theta_06 - theta_09) + g_06_09*sin(theta_06 - theta_09)) + V_09**2*(-b_06_09 - bs_06_09/2) - S_n_09*q_s_09/S_base
        struct[0].g[18,0] = -P_10/S_base + V_05*V_10*(b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10)) + V_10**2*g_05_10 - S_n_10*p_s_10/S_base
        struct[0].g[19,0] = -Q_10/S_base + V_05*V_10*(b_05_10*cos(theta_05 - theta_10) + g_05_10*sin(theta_05 - theta_10)) + V_10**2*(-b_05_10 - bs_05_10/2) - S_n_10*q_s_10/S_base
        struct[0].g[20,0] = -P_11/S_base + V_04*V_11*(b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11)) + V_11**2*g_04_11
        struct[0].g[21,0] = -Q_11/S_base + V_04*V_11*(b_04_11*cos(theta_04 - theta_11) + g_04_11*sin(theta_04 - theta_11)) + V_11**2*(-b_04_11 - bs_04_11/2)
        struct[0].g[22,0] = -P_12/S_base + V_03*V_12*(b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12)) + V_12**2*g_03_12
        struct[0].g[23,0] = -Q_12/S_base + V_03*V_12*(b_03_12*cos(theta_03 - theta_12) + g_03_12*sin(theta_03 - theta_12)) + V_12**2*(-b_03_12 - bs_03_12/2)
        struct[0].g[24,0] = K_p_08*(p_m_08 - p_s_08 + xi_p_08/T_p_08) - omega_08
        struct[0].g[25,0] = K_q_08*(-q_s_08 + q_s_ref_08 + xi_q_08/T_q_08) - e_qv_08
        struct[0].g[26,0] = -R_v_08*i_d_08 - V_08*sin(delta_08 - theta_08) + X_v_08*i_q_08
        struct[0].g[27,0] = -R_v_08*i_q_08 - V_08*cos(delta_08 - theta_08) - X_v_08*i_d_08 + e_qv_08
        struct[0].g[28,0] = V_08*i_d_08*sin(delta_08 - theta_08) + V_08*i_q_08*cos(delta_08 - theta_08) - p_s_08
        struct[0].g[29,0] = V_08*i_d_08*cos(delta_08 - theta_08) - V_08*i_q_08*sin(delta_08 - theta_08) - q_s_08
        struct[0].g[30,0] = K_fpfr_08*Piecewise(np.array([(P_f_min_08, P_f_min_08 > p_f_08), (P_f_max_08, P_f_max_08 < p_f_08), (p_f_08, True)])) + p_ghr_08 - p_m_08 + p_s_08 - p_t_08 + p_u_08
        struct[0].g[31,0] = i_d_08*(R_s_08*i_d_08 + V_08*sin(delta_08 - theta_08)) + i_q_08*(R_s_08*i_q_08 + V_08*cos(delta_08 - theta_08)) - p_t_08
        struct[0].g[32,0] = -p_u_08 - k_u_08*(-v_u_08**2 + v_u_ref_08**2)/V_u_max_08**2
        struct[0].g[33,0] = R_uc_08*S_n_08*(p_gou_08 - p_t_08)/(v_u_08 + 0.1) + e_u_08 - v_u_08
        struct[0].g[34,0] = -k_u_08 + Piecewise(np.array([(K_u_max_08, V_u_min_08 > v_u_08), (K_u_0_08 + (-K_u_0_08 + K_u_max_08)*(-V_u_lt_08 + v_u_08)/(-V_u_lt_08 + V_u_min_08), V_u_lt_08 > v_u_08), (K_u_0_08 + (-K_u_0_08 + K_u_max_08)*(-V_u_ht_08 + v_u_08)/(-V_u_ht_08 + V_u_max_08), V_u_ht_08 < v_u_08), (K_u_max_08, V_u_max_08 < v_u_08), (K_u_0_08, True)]))
        struct[0].g[35,0] = -k_cur_sat_08 + Piecewise(np.array([(0.0001, k_cur_08 < 0.0001), (1, k_cur_08 > 1), (k_cur_08, True)]))
        struct[0].g[37,0] = -p_f_08 - Piecewise(np.array([((0.5*DB_08 + omega_08 - omega_ref_08)/Droop_08, omega_08 < -0.5*DB_08 + omega_ref_08), ((-0.5*DB_08 + omega_08 - omega_ref_08)/Droop_08, omega_08 > 0.5*DB_08 + omega_ref_08), (0.0, True)]))
        struct[0].g[38,0] = -r_lim_08 + Piecewise(np.array([(R_lim_max_08, (omega_08 > 0.5*DB_08 + omega_ref_08) | (omega_08 < -0.5*DB_08 + omega_ref_08)), (0.0, True)])) + Piecewise(np.array([(R_lim_08 + (-R_lim_08 + R_lim_max_08)*(-V_u_lt_08 + v_u_08)/(-V_u_lt_08 + V_u_min_08), V_u_lt_08 > v_u_08), (R_lim_08 + (-R_lim_08 + R_lim_max_08)*(-V_u_ht_08 + v_u_08)/(-V_u_ht_08 + V_u_max_08), V_u_ht_08 < v_u_08), (R_lim_08, True)]))
        struct[0].g[39,0] = K_p_09*(p_m_09 - p_s_09 + xi_p_09/T_p_09) - omega_09
        struct[0].g[40,0] = K_q_09*(-q_s_09 + q_s_ref_09 + xi_q_09/T_q_09) - e_qv_09
        struct[0].g[41,0] = -R_v_09*i_d_09 - V_09*sin(delta_09 - theta_09) + X_v_09*i_q_09
        struct[0].g[42,0] = -R_v_09*i_q_09 - V_09*cos(delta_09 - theta_09) - X_v_09*i_d_09 + e_qv_09
        struct[0].g[43,0] = V_09*i_d_09*sin(delta_09 - theta_09) + V_09*i_q_09*cos(delta_09 - theta_09) - p_s_09
        struct[0].g[44,0] = V_09*i_d_09*cos(delta_09 - theta_09) - V_09*i_q_09*sin(delta_09 - theta_09) - q_s_09
        struct[0].g[45,0] = K_fpfr_09*Piecewise(np.array([(P_f_min_09, P_f_min_09 > p_f_09), (P_f_max_09, P_f_max_09 < p_f_09), (p_f_09, True)])) + p_ghr_09 - p_m_09 + p_s_09 - p_t_09 + p_u_09
        struct[0].g[46,0] = i_d_09*(R_s_09*i_d_09 + V_09*sin(delta_09 - theta_09)) + i_q_09*(R_s_09*i_q_09 + V_09*cos(delta_09 - theta_09)) - p_t_09
        struct[0].g[47,0] = -p_u_09 - k_u_09*(-v_u_09**2 + v_u_ref_09**2)/V_u_max_09**2
        struct[0].g[48,0] = R_uc_09*S_n_09*(p_gou_09 - p_t_09)/(v_u_09 + 0.1) + e_u_09 - v_u_09
        struct[0].g[49,0] = -k_u_09 + Piecewise(np.array([(K_u_max_09, V_u_min_09 > v_u_09), (K_u_0_09 + (-K_u_0_09 + K_u_max_09)*(-V_u_lt_09 + v_u_09)/(-V_u_lt_09 + V_u_min_09), V_u_lt_09 > v_u_09), (K_u_0_09 + (-K_u_0_09 + K_u_max_09)*(-V_u_ht_09 + v_u_09)/(-V_u_ht_09 + V_u_max_09), V_u_ht_09 < v_u_09), (K_u_max_09, V_u_max_09 < v_u_09), (K_u_0_09, True)]))
        struct[0].g[50,0] = -k_cur_sat_09 + Piecewise(np.array([(0.0001, k_cur_09 < 0.0001), (1, k_cur_09 > 1), (k_cur_09, True)]))
        struct[0].g[52,0] = -p_f_09 - Piecewise(np.array([((0.5*DB_09 + omega_09 - omega_ref_09)/Droop_09, omega_09 < -0.5*DB_09 + omega_ref_09), ((-0.5*DB_09 + omega_09 - omega_ref_09)/Droop_09, omega_09 > 0.5*DB_09 + omega_ref_09), (0.0, True)]))
        struct[0].g[53,0] = -r_lim_09 + Piecewise(np.array([(R_lim_max_09, (omega_09 > 0.5*DB_09 + omega_ref_09) | (omega_09 < -0.5*DB_09 + omega_ref_09)), (0.0, True)])) + Piecewise(np.array([(R_lim_09 + (-R_lim_09 + R_lim_max_09)*(-V_u_lt_09 + v_u_09)/(-V_u_lt_09 + V_u_min_09), V_u_lt_09 > v_u_09), (R_lim_09 + (-R_lim_09 + R_lim_max_09)*(-V_u_ht_09 + v_u_09)/(-V_u_ht_09 + V_u_max_09), V_u_ht_09 < v_u_09), (R_lim_09, True)]))
        struct[0].g[54,0] = K_p_10*(p_m_10 - p_s_10 + xi_p_10/T_p_10) - omega_10
        struct[0].g[55,0] = K_q_10*(-q_s_10 + q_s_ref_10 + xi_q_10/T_q_10) - e_qv_10
        struct[0].g[56,0] = -R_v_10*i_d_10 - V_10*sin(delta_10 - theta_10) + X_v_10*i_q_10
        struct[0].g[57,0] = -R_v_10*i_q_10 - V_10*cos(delta_10 - theta_10) - X_v_10*i_d_10 + e_qv_10
        struct[0].g[58,0] = V_10*i_d_10*sin(delta_10 - theta_10) + V_10*i_q_10*cos(delta_10 - theta_10) - p_s_10
        struct[0].g[59,0] = V_10*i_d_10*cos(delta_10 - theta_10) - V_10*i_q_10*sin(delta_10 - theta_10) - q_s_10
        struct[0].g[60,0] = K_fpfr_10*Piecewise(np.array([(P_f_min_10, P_f_min_10 > p_f_10), (P_f_max_10, P_f_max_10 < p_f_10), (p_f_10, True)])) + p_ghr_10 - p_m_10 + p_s_10 - p_t_10 + p_u_10
        struct[0].g[61,0] = i_d_10*(R_s_10*i_d_10 + V_10*sin(delta_10 - theta_10)) + i_q_10*(R_s_10*i_q_10 + V_10*cos(delta_10 - theta_10)) - p_t_10
        struct[0].g[62,0] = -p_u_10 - k_u_10*(-v_u_10**2 + v_u_ref_10**2)/V_u_max_10**2
        struct[0].g[63,0] = R_uc_10*S_n_10*(p_gou_10 - p_t_10)/(v_u_10 + 0.1) + e_u_10 - v_u_10
        struct[0].g[64,0] = -k_u_10 + Piecewise(np.array([(K_u_max_10, V_u_min_10 > v_u_10), (K_u_0_10 + (-K_u_0_10 + K_u_max_10)*(-V_u_lt_10 + v_u_10)/(-V_u_lt_10 + V_u_min_10), V_u_lt_10 > v_u_10), (K_u_0_10 + (-K_u_0_10 + K_u_max_10)*(-V_u_ht_10 + v_u_10)/(-V_u_ht_10 + V_u_max_10), V_u_ht_10 < v_u_10), (K_u_max_10, V_u_max_10 < v_u_10), (K_u_0_10, True)]))
        struct[0].g[65,0] = -k_cur_sat_10 + Piecewise(np.array([(0.0001, k_cur_10 < 0.0001), (1, k_cur_10 > 1), (k_cur_10, True)]))
        struct[0].g[67,0] = -p_f_10 - Piecewise(np.array([((0.5*DB_10 + omega_10 - omega_ref_10)/Droop_10, omega_10 < -0.5*DB_10 + omega_ref_10), ((-0.5*DB_10 + omega_10 - omega_ref_10)/Droop_10, omega_10 > 0.5*DB_10 + omega_ref_10), (0.0, True)]))
        struct[0].g[68,0] = -r_lim_10 + Piecewise(np.array([(R_lim_max_10, (omega_10 > 0.5*DB_10 + omega_ref_10) | (omega_10 < -0.5*DB_10 + omega_ref_10)), (0.0, True)])) + Piecewise(np.array([(R_lim_10 + (-R_lim_10 + R_lim_max_10)*(-V_u_lt_10 + v_u_10)/(-V_u_lt_10 + V_u_min_10), V_u_lt_10 > v_u_10), (R_lim_10 + (-R_lim_10 + R_lim_max_10)*(-V_u_ht_10 + v_u_10)/(-V_u_ht_10 + V_u_max_10), V_u_ht_10 < v_u_10), (R_lim_10, True)]))
        struct[0].g[69,0] = Domega_01 - omega_01 + omega_ref_01
        struct[0].g[70,0] = -R_v_01*i_d_01 - V_01*sin(delta_01 - theta_01) + X_v_01*i_q_01
        struct[0].g[71,0] = -R_v_01*i_q_01 - V_01*cos(delta_01 - theta_01) - X_v_01*i_d_01 + e_qv_01
        struct[0].g[72,0] = V_01*i_d_01*sin(delta_01 - theta_01) + V_01*i_q_01*cos(delta_01 - theta_01) - p_s_01
        struct[0].g[73,0] = V_01*i_d_01*cos(delta_01 - theta_01) - V_01*i_q_01*sin(delta_01 - theta_01) - q_s_01
        struct[0].g[74,0] = -omega_coi + (1000000.0*S_n_01*omega_01 + S_n_10*T_p_10*omega_10/(2*K_p_10) + S_n_09*T_p_09*omega_09/(2*K_p_09) + S_n_08*T_p_08*omega_08/(2*K_p_08))/(1000000.0*S_n_01 + S_n_10*T_p_10/(2*K_p_10) + S_n_09*T_p_09/(2*K_p_09) + S_n_08*T_p_08/(2*K_p_08))
        struct[0].g[75,0] = K_i_agc*xi_freq + K_p_agc*(1 - omega_coi) - p_agc
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_01
        struct[0].h[1,0] = V_02
        struct[0].h[2,0] = V_03
        struct[0].h[3,0] = V_04
        struct[0].h[4,0] = V_05
        struct[0].h[5,0] = V_06
        struct[0].h[6,0] = V_07
        struct[0].h[7,0] = V_08
        struct[0].h[8,0] = V_09
        struct[0].h[9,0] = V_10
        struct[0].h[10,0] = V_11
        struct[0].h[11,0] = V_12
        struct[0].h[12,0] = inc_p_gin_08 + p_gin_0_08
        struct[0].h[13,0] = p_g_ref_08
        struct[0].h[14,0] = -p_s_08 + p_t_08
        struct[0].h[15,0] = (-V_u_min_08**2 + e_u_08**2)/(V_u_max_08**2 - V_u_min_08**2)
        struct[0].h[16,0] = K_fpfr_08*Piecewise(np.array([(P_f_min_08, P_f_min_08 > p_f_08), (P_f_max_08, P_f_max_08 < p_f_08), (p_f_08, True)]))
        struct[0].h[17,0] = Piecewise(np.array([(P_f_min_08, P_f_min_08 > p_f_08), (P_f_max_08, P_f_max_08 < p_f_08), (p_f_08, True)]))
        struct[0].h[18,0] = inc_p_gin_09 + p_gin_0_09
        struct[0].h[19,0] = p_g_ref_09
        struct[0].h[20,0] = -p_s_09 + p_t_09
        struct[0].h[21,0] = (-V_u_min_09**2 + e_u_09**2)/(V_u_max_09**2 - V_u_min_09**2)
        struct[0].h[22,0] = K_fpfr_09*Piecewise(np.array([(P_f_min_09, P_f_min_09 > p_f_09), (P_f_max_09, P_f_max_09 < p_f_09), (p_f_09, True)]))
        struct[0].h[23,0] = Piecewise(np.array([(P_f_min_09, P_f_min_09 > p_f_09), (P_f_max_09, P_f_max_09 < p_f_09), (p_f_09, True)]))
        struct[0].h[24,0] = inc_p_gin_10 + p_gin_0_10
        struct[0].h[25,0] = p_g_ref_10
        struct[0].h[26,0] = -p_s_10 + p_t_10
        struct[0].h[27,0] = (-V_u_min_10**2 + e_u_10**2)/(V_u_max_10**2 - V_u_min_10**2)
        struct[0].h[28,0] = K_fpfr_10*Piecewise(np.array([(P_f_min_10, P_f_min_10 > p_f_10), (P_f_max_10, P_f_max_10 < p_f_10), (p_f_10, True)]))
        struct[0].h[29,0] = Piecewise(np.array([(P_f_min_10, P_f_min_10 > p_f_10), (P_f_max_10, P_f_max_10 < p_f_10), (p_f_10, True)]))
        struct[0].h[30,0] = alpha_01
    

    if mode == 10:

        struct[0].Fx_ini[4,4] = Piecewise(np.array([(0, (r_lim_08 < K_h_08*(-p_ghr_08 + p_gou_08)) | (r_lim_08 < -K_h_08*(-p_ghr_08 + p_gou_08))), (-K_h_08, True)]))
        struct[0].Fx_ini[5,5] = -1/T_cur_08
        struct[0].Fx_ini[5,6] = (-p_g_ref_08/(inc_p_gin_08 + p_gin_0_08)**2 - Piecewise(np.array([(P_f_min_08, P_f_min_08 > p_f_08), (P_f_max_08, P_f_max_08 < p_f_08), (p_f_08, True)]))/(inc_p_gin_08 + p_gin_0_08)**2)/T_cur_08
        struct[0].Fx_ini[11,11] = Piecewise(np.array([(0, (r_lim_09 < K_h_09*(-p_ghr_09 + p_gou_09)) | (r_lim_09 < -K_h_09*(-p_ghr_09 + p_gou_09))), (-K_h_09, True)]))
        struct[0].Fx_ini[12,12] = -1/T_cur_09
        struct[0].Fx_ini[12,13] = (-p_g_ref_09/(inc_p_gin_09 + p_gin_0_09)**2 - Piecewise(np.array([(P_f_min_09, P_f_min_09 > p_f_09), (P_f_max_09, P_f_max_09 < p_f_09), (p_f_09, True)]))/(inc_p_gin_09 + p_gin_0_09)**2)/T_cur_09
        struct[0].Fx_ini[18,18] = Piecewise(np.array([(0, (r_lim_10 < K_h_10*(-p_ghr_10 + p_gou_10)) | (r_lim_10 < -K_h_10*(-p_ghr_10 + p_gou_10))), (-K_h_10, True)]))
        struct[0].Fx_ini[19,19] = -1/T_cur_10
        struct[0].Fx_ini[19,20] = (-p_g_ref_10/(inc_p_gin_10 + p_gin_0_10)**2 - Piecewise(np.array([(P_f_min_10, P_f_min_10 > p_f_10), (P_f_max_10, P_f_max_10 < p_f_10), (p_f_10, True)]))/(inc_p_gin_10 + p_gin_0_10)**2)/T_cur_10
        struct[0].Fx_ini[21,21] = -K_delta_01
        struct[0].Fx_ini[22,22] = -K_alpha_01

    if mode == 11:

        struct[0].Fy_ini[0,24] = Omega_b_08 
        struct[0].Fy_ini[0,74] = -Omega_b_08 
        struct[0].Fy_ini[1,28] = -1 
        struct[0].Fy_ini[1,30] = 1 
        struct[0].Fy_ini[2,29] = -1 
        struct[0].Fy_ini[3,31] = -S_n_08/(C_u_08*(v_u_08 + 0.1)) 
        struct[0].Fy_ini[3,33] = -S_n_08*(p_gou_08 - p_t_08)/(C_u_08*(v_u_08 + 0.1)**2) 
        struct[0].Fy_ini[3,36] = S_n_08/(C_u_08*(v_u_08 + 0.1)) 
        struct[0].Fy_ini[4,36] = Piecewise(np.array([(0, (r_lim_08 < K_h_08*(-p_ghr_08 + p_gou_08)) | (r_lim_08 < -K_h_08*(-p_ghr_08 + p_gou_08))), (K_h_08, True)])) 
        struct[0].Fy_ini[4,38] = Piecewise(np.array([(-1, r_lim_08 < -K_h_08*(-p_ghr_08 + p_gou_08)), (1, r_lim_08 < K_h_08*(-p_ghr_08 + p_gou_08)), (0, True)])) 
        struct[0].Fy_ini[5,37] = Piecewise(np.array([(0, (P_f_min_08 > p_f_08) | (P_f_max_08 < p_f_08)), (1, True)]))/(T_cur_08*(inc_p_gin_08 + p_gin_0_08)) 
        struct[0].Fy_ini[7,39] = Omega_b_09 
        struct[0].Fy_ini[7,74] = -Omega_b_09 
        struct[0].Fy_ini[8,43] = -1 
        struct[0].Fy_ini[8,45] = 1 
        struct[0].Fy_ini[9,44] = -1 
        struct[0].Fy_ini[10,46] = -S_n_09/(C_u_09*(v_u_09 + 0.1)) 
        struct[0].Fy_ini[10,48] = -S_n_09*(p_gou_09 - p_t_09)/(C_u_09*(v_u_09 + 0.1)**2) 
        struct[0].Fy_ini[10,51] = S_n_09/(C_u_09*(v_u_09 + 0.1)) 
        struct[0].Fy_ini[11,51] = Piecewise(np.array([(0, (r_lim_09 < K_h_09*(-p_ghr_09 + p_gou_09)) | (r_lim_09 < -K_h_09*(-p_ghr_09 + p_gou_09))), (K_h_09, True)])) 
        struct[0].Fy_ini[11,53] = Piecewise(np.array([(-1, r_lim_09 < -K_h_09*(-p_ghr_09 + p_gou_09)), (1, r_lim_09 < K_h_09*(-p_ghr_09 + p_gou_09)), (0, True)])) 
        struct[0].Fy_ini[12,52] = Piecewise(np.array([(0, (P_f_min_09 > p_f_09) | (P_f_max_09 < p_f_09)), (1, True)]))/(T_cur_09*(inc_p_gin_09 + p_gin_0_09)) 
        struct[0].Fy_ini[14,54] = Omega_b_10 
        struct[0].Fy_ini[14,74] = -Omega_b_10 
        struct[0].Fy_ini[15,58] = -1 
        struct[0].Fy_ini[15,60] = 1 
        struct[0].Fy_ini[16,59] = -1 
        struct[0].Fy_ini[17,61] = -S_n_10/(C_u_10*(v_u_10 + 0.1)) 
        struct[0].Fy_ini[17,63] = -S_n_10*(p_gou_10 - p_t_10)/(C_u_10*(v_u_10 + 0.1)**2) 
        struct[0].Fy_ini[17,66] = S_n_10/(C_u_10*(v_u_10 + 0.1)) 
        struct[0].Fy_ini[18,66] = Piecewise(np.array([(0, (r_lim_10 < K_h_10*(-p_ghr_10 + p_gou_10)) | (r_lim_10 < -K_h_10*(-p_ghr_10 + p_gou_10))), (K_h_10, True)])) 
        struct[0].Fy_ini[18,68] = Piecewise(np.array([(-1, r_lim_10 < -K_h_10*(-p_ghr_10 + p_gou_10)), (1, r_lim_10 < K_h_10*(-p_ghr_10 + p_gou_10)), (0, True)])) 
        struct[0].Fy_ini[19,67] = Piecewise(np.array([(0, (P_f_min_10 > p_f_10) | (P_f_max_10 < p_f_10)), (1, True)]))/(T_cur_10*(inc_p_gin_10 + p_gin_0_10)) 
        struct[0].Fy_ini[21,69] = Omega_b_01 
        struct[0].Fy_ini[21,74] = -Omega_b_01 
        struct[0].Fy_ini[23,74] = -1 

        struct[0].Gx_ini[24,1] = K_p_08/T_p_08
        struct[0].Gx_ini[25,2] = K_q_08/T_q_08
        struct[0].Gx_ini[26,0] = -V_08*cos(delta_08 - theta_08)
        struct[0].Gx_ini[27,0] = V_08*sin(delta_08 - theta_08)
        struct[0].Gx_ini[28,0] = V_08*i_d_08*cos(delta_08 - theta_08) - V_08*i_q_08*sin(delta_08 - theta_08)
        struct[0].Gx_ini[29,0] = -V_08*i_d_08*sin(delta_08 - theta_08) - V_08*i_q_08*cos(delta_08 - theta_08)
        struct[0].Gx_ini[30,4] = 1
        struct[0].Gx_ini[31,0] = V_08*i_d_08*cos(delta_08 - theta_08) - V_08*i_q_08*sin(delta_08 - theta_08)
        struct[0].Gx_ini[33,3] = 1
        struct[0].Gx_ini[35,5] = Piecewise(np.array([(0, (k_cur_08 > 1) | (k_cur_08 < 0.0001)), (1, True)]))
        struct[0].Gx_ini[36,6] = k_cur_sat_08
        struct[0].Gx_ini[39,8] = K_p_09/T_p_09
        struct[0].Gx_ini[40,9] = K_q_09/T_q_09
        struct[0].Gx_ini[41,7] = -V_09*cos(delta_09 - theta_09)
        struct[0].Gx_ini[42,7] = V_09*sin(delta_09 - theta_09)
        struct[0].Gx_ini[43,7] = V_09*i_d_09*cos(delta_09 - theta_09) - V_09*i_q_09*sin(delta_09 - theta_09)
        struct[0].Gx_ini[44,7] = -V_09*i_d_09*sin(delta_09 - theta_09) - V_09*i_q_09*cos(delta_09 - theta_09)
        struct[0].Gx_ini[45,11] = 1
        struct[0].Gx_ini[46,7] = V_09*i_d_09*cos(delta_09 - theta_09) - V_09*i_q_09*sin(delta_09 - theta_09)
        struct[0].Gx_ini[48,10] = 1
        struct[0].Gx_ini[50,12] = Piecewise(np.array([(0, (k_cur_09 > 1) | (k_cur_09 < 0.0001)), (1, True)]))
        struct[0].Gx_ini[51,13] = k_cur_sat_09
        struct[0].Gx_ini[54,15] = K_p_10/T_p_10
        struct[0].Gx_ini[55,16] = K_q_10/T_q_10
        struct[0].Gx_ini[56,14] = -V_10*cos(delta_10 - theta_10)
        struct[0].Gx_ini[57,14] = V_10*sin(delta_10 - theta_10)
        struct[0].Gx_ini[58,14] = V_10*i_d_10*cos(delta_10 - theta_10) - V_10*i_q_10*sin(delta_10 - theta_10)
        struct[0].Gx_ini[59,14] = -V_10*i_d_10*sin(delta_10 - theta_10) - V_10*i_q_10*cos(delta_10 - theta_10)
        struct[0].Gx_ini[60,18] = 1
        struct[0].Gx_ini[61,14] = V_10*i_d_10*cos(delta_10 - theta_10) - V_10*i_q_10*sin(delta_10 - theta_10)
        struct[0].Gx_ini[63,17] = 1
        struct[0].Gx_ini[65,19] = Piecewise(np.array([(0, (k_cur_10 > 1) | (k_cur_10 < 0.0001)), (1, True)]))
        struct[0].Gx_ini[66,20] = k_cur_sat_10
        struct[0].Gx_ini[69,22] = 1
        struct[0].Gx_ini[70,21] = -V_01*cos(delta_01 - theta_01)
        struct[0].Gx_ini[71,21] = V_01*sin(delta_01 - theta_01)
        struct[0].Gx_ini[72,21] = V_01*i_d_01*cos(delta_01 - theta_01) - V_01*i_q_01*sin(delta_01 - theta_01)
        struct[0].Gx_ini[73,21] = -V_01*i_d_01*sin(delta_01 - theta_01) - V_01*i_q_01*cos(delta_01 - theta_01)
        struct[0].Gx_ini[75,23] = K_i_agc

        struct[0].Gy_ini[0,0] = 2*V_01*g_01_02 + V_02*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy_ini[0,1] = V_01*V_02*(-b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy_ini[0,2] = V_01*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy_ini[0,3] = V_01*V_02*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy_ini[0,72] = -S_n_01/S_base
        struct[0].Gy_ini[1,0] = 2*V_01*(-b_01_02 - bs_01_02/2) + V_02*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy_ini[1,1] = V_01*V_02*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy_ini[1,2] = V_01*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy_ini[1,3] = V_01*V_02*(b_01_02*sin(theta_01 - theta_02) + g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy_ini[1,73] = -S_n_01/S_base
        struct[0].Gy_ini[2,0] = V_02*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy_ini[2,1] = V_01*V_02*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy_ini[2,2] = V_01*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) + 2*V_02*(g_01_02 + g_02_03) + V_03*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy_ini[2,3] = V_01*V_02*(-b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02)) + V_02*V_03*(-b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy_ini[2,4] = V_02*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy_ini[2,5] = V_02*V_03*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy_ini[3,0] = V_02*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy_ini[3,1] = V_01*V_02*(-b_01_02*sin(theta_01 - theta_02) + g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy_ini[3,2] = V_01*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02)) + 2*V_02*(-b_01_02 - b_02_03 - bs_01_02/2 - bs_02_03/2) + V_03*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy_ini[3,3] = V_01*V_02*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) + V_02*V_03*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy_ini[3,4] = V_02*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy_ini[3,5] = V_02*V_03*(b_02_03*sin(theta_02 - theta_03) + g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy_ini[4,2] = V_03*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy_ini[4,3] = V_02*V_03*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy_ini[4,4] = V_02*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03)) + 2*V_03*(g_02_03 + g_03_04 + g_03_12) + V_04*(-b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04)) + V_12*(-b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12))
        struct[0].Gy_ini[4,5] = V_02*V_03*(-b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03)) + V_03*V_04*(-b_03_04*cos(theta_03 - theta_04) + g_03_04*sin(theta_03 - theta_04)) + V_03*V_12*(-b_03_12*cos(theta_03 - theta_12) + g_03_12*sin(theta_03 - theta_12))
        struct[0].Gy_ini[4,6] = V_03*(-b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04))
        struct[0].Gy_ini[4,7] = V_03*V_04*(b_03_04*cos(theta_03 - theta_04) - g_03_04*sin(theta_03 - theta_04))
        struct[0].Gy_ini[4,22] = V_03*(-b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12))
        struct[0].Gy_ini[4,23] = V_03*V_12*(b_03_12*cos(theta_03 - theta_12) - g_03_12*sin(theta_03 - theta_12))
        struct[0].Gy_ini[5,2] = V_03*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy_ini[5,3] = V_02*V_03*(-b_02_03*sin(theta_02 - theta_03) + g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy_ini[5,4] = V_02*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03)) + 2*V_03*(-b_02_03 - b_03_04 - b_03_12 - bs_02_03/2 - bs_03_04/2 - bs_03_12/2) + V_04*(b_03_04*cos(theta_03 - theta_04) - g_03_04*sin(theta_03 - theta_04)) + V_12*(b_03_12*cos(theta_03 - theta_12) - g_03_12*sin(theta_03 - theta_12))
        struct[0].Gy_ini[5,5] = V_02*V_03*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03)) + V_03*V_04*(-b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04)) + V_03*V_12*(-b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12))
        struct[0].Gy_ini[5,6] = V_03*(b_03_04*cos(theta_03 - theta_04) - g_03_04*sin(theta_03 - theta_04))
        struct[0].Gy_ini[5,7] = V_03*V_04*(b_03_04*sin(theta_03 - theta_04) + g_03_04*cos(theta_03 - theta_04))
        struct[0].Gy_ini[5,22] = V_03*(b_03_12*cos(theta_03 - theta_12) - g_03_12*sin(theta_03 - theta_12))
        struct[0].Gy_ini[5,23] = V_03*V_12*(b_03_12*sin(theta_03 - theta_12) + g_03_12*cos(theta_03 - theta_12))
        struct[0].Gy_ini[6,4] = V_04*(b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04))
        struct[0].Gy_ini[6,5] = V_03*V_04*(b_03_04*cos(theta_03 - theta_04) + g_03_04*sin(theta_03 - theta_04))
        struct[0].Gy_ini[6,6] = V_03*(b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04)) + 2*V_04*(g_03_04 + g_04_05 + g_04_11) + V_05*(-b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05)) + V_11*(-b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11))
        struct[0].Gy_ini[6,7] = V_03*V_04*(-b_03_04*cos(theta_03 - theta_04) - g_03_04*sin(theta_03 - theta_04)) + V_04*V_05*(-b_04_05*cos(theta_04 - theta_05) + g_04_05*sin(theta_04 - theta_05)) + V_04*V_11*(-b_04_11*cos(theta_04 - theta_11) + g_04_11*sin(theta_04 - theta_11))
        struct[0].Gy_ini[6,8] = V_04*(-b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05))
        struct[0].Gy_ini[6,9] = V_04*V_05*(b_04_05*cos(theta_04 - theta_05) - g_04_05*sin(theta_04 - theta_05))
        struct[0].Gy_ini[6,20] = V_04*(-b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11))
        struct[0].Gy_ini[6,21] = V_04*V_11*(b_04_11*cos(theta_04 - theta_11) - g_04_11*sin(theta_04 - theta_11))
        struct[0].Gy_ini[7,4] = V_04*(b_03_04*cos(theta_03 - theta_04) + g_03_04*sin(theta_03 - theta_04))
        struct[0].Gy_ini[7,5] = V_03*V_04*(-b_03_04*sin(theta_03 - theta_04) + g_03_04*cos(theta_03 - theta_04))
        struct[0].Gy_ini[7,6] = V_03*(b_03_04*cos(theta_03 - theta_04) + g_03_04*sin(theta_03 - theta_04)) + 2*V_04*(-b_03_04 - b_04_05 - b_04_11 - bs_03_04/2 - bs_04_05/2 - bs_04_11/2) + V_05*(b_04_05*cos(theta_04 - theta_05) - g_04_05*sin(theta_04 - theta_05)) + V_11*(b_04_11*cos(theta_04 - theta_11) - g_04_11*sin(theta_04 - theta_11))
        struct[0].Gy_ini[7,7] = V_03*V_04*(b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04)) + V_04*V_05*(-b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05)) + V_04*V_11*(-b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11))
        struct[0].Gy_ini[7,8] = V_04*(b_04_05*cos(theta_04 - theta_05) - g_04_05*sin(theta_04 - theta_05))
        struct[0].Gy_ini[7,9] = V_04*V_05*(b_04_05*sin(theta_04 - theta_05) + g_04_05*cos(theta_04 - theta_05))
        struct[0].Gy_ini[7,20] = V_04*(b_04_11*cos(theta_04 - theta_11) - g_04_11*sin(theta_04 - theta_11))
        struct[0].Gy_ini[7,21] = V_04*V_11*(b_04_11*sin(theta_04 - theta_11) + g_04_11*cos(theta_04 - theta_11))
        struct[0].Gy_ini[8,6] = V_05*(b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05))
        struct[0].Gy_ini[8,7] = V_04*V_05*(b_04_05*cos(theta_04 - theta_05) + g_04_05*sin(theta_04 - theta_05))
        struct[0].Gy_ini[8,8] = V_04*(b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05)) + 2*V_05*(g_04_05 + g_05_06 + g_05_10) + V_06*(-b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06)) + V_10*(-b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10))
        struct[0].Gy_ini[8,9] = V_04*V_05*(-b_04_05*cos(theta_04 - theta_05) - g_04_05*sin(theta_04 - theta_05)) + V_05*V_06*(-b_05_06*cos(theta_05 - theta_06) + g_05_06*sin(theta_05 - theta_06)) + V_05*V_10*(-b_05_10*cos(theta_05 - theta_10) + g_05_10*sin(theta_05 - theta_10))
        struct[0].Gy_ini[8,10] = V_05*(-b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06))
        struct[0].Gy_ini[8,11] = V_05*V_06*(b_05_06*cos(theta_05 - theta_06) - g_05_06*sin(theta_05 - theta_06))
        struct[0].Gy_ini[8,18] = V_05*(-b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10))
        struct[0].Gy_ini[8,19] = V_05*V_10*(b_05_10*cos(theta_05 - theta_10) - g_05_10*sin(theta_05 - theta_10))
        struct[0].Gy_ini[9,6] = V_05*(b_04_05*cos(theta_04 - theta_05) + g_04_05*sin(theta_04 - theta_05))
        struct[0].Gy_ini[9,7] = V_04*V_05*(-b_04_05*sin(theta_04 - theta_05) + g_04_05*cos(theta_04 - theta_05))
        struct[0].Gy_ini[9,8] = V_04*(b_04_05*cos(theta_04 - theta_05) + g_04_05*sin(theta_04 - theta_05)) + 2*V_05*(-b_04_05 - b_05_06 - b_05_10 - bs_04_05/2 - bs_05_06/2 - bs_05_10/2) + V_06*(b_05_06*cos(theta_05 - theta_06) - g_05_06*sin(theta_05 - theta_06)) + V_10*(b_05_10*cos(theta_05 - theta_10) - g_05_10*sin(theta_05 - theta_10))
        struct[0].Gy_ini[9,9] = V_04*V_05*(b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05)) + V_05*V_06*(-b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06)) + V_05*V_10*(-b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10))
        struct[0].Gy_ini[9,10] = V_05*(b_05_06*cos(theta_05 - theta_06) - g_05_06*sin(theta_05 - theta_06))
        struct[0].Gy_ini[9,11] = V_05*V_06*(b_05_06*sin(theta_05 - theta_06) + g_05_06*cos(theta_05 - theta_06))
        struct[0].Gy_ini[9,18] = V_05*(b_05_10*cos(theta_05 - theta_10) - g_05_10*sin(theta_05 - theta_10))
        struct[0].Gy_ini[9,19] = V_05*V_10*(b_05_10*sin(theta_05 - theta_10) + g_05_10*cos(theta_05 - theta_10))
        struct[0].Gy_ini[10,8] = V_06*(b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06))
        struct[0].Gy_ini[10,9] = V_05*V_06*(b_05_06*cos(theta_05 - theta_06) + g_05_06*sin(theta_05 - theta_06))
        struct[0].Gy_ini[10,10] = V_05*(b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06)) + 2*V_06*(g_05_06 + g_06_07 + g_06_09) + V_07*(-b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07)) + V_09*(-b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09))
        struct[0].Gy_ini[10,11] = V_05*V_06*(-b_05_06*cos(theta_05 - theta_06) - g_05_06*sin(theta_05 - theta_06)) + V_06*V_07*(-b_06_07*cos(theta_06 - theta_07) + g_06_07*sin(theta_06 - theta_07)) + V_06*V_09*(-b_06_09*cos(theta_06 - theta_09) + g_06_09*sin(theta_06 - theta_09))
        struct[0].Gy_ini[10,12] = V_06*(-b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07))
        struct[0].Gy_ini[10,13] = V_06*V_07*(b_06_07*cos(theta_06 - theta_07) - g_06_07*sin(theta_06 - theta_07))
        struct[0].Gy_ini[10,16] = V_06*(-b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09))
        struct[0].Gy_ini[10,17] = V_06*V_09*(b_06_09*cos(theta_06 - theta_09) - g_06_09*sin(theta_06 - theta_09))
        struct[0].Gy_ini[11,8] = V_06*(b_05_06*cos(theta_05 - theta_06) + g_05_06*sin(theta_05 - theta_06))
        struct[0].Gy_ini[11,9] = V_05*V_06*(-b_05_06*sin(theta_05 - theta_06) + g_05_06*cos(theta_05 - theta_06))
        struct[0].Gy_ini[11,10] = V_05*(b_05_06*cos(theta_05 - theta_06) + g_05_06*sin(theta_05 - theta_06)) + 2*V_06*(-b_05_06 - b_06_07 - b_06_09 - bs_05_06/2 - bs_06_07/2 - bs_06_09/2) + V_07*(b_06_07*cos(theta_06 - theta_07) - g_06_07*sin(theta_06 - theta_07)) + V_09*(b_06_09*cos(theta_06 - theta_09) - g_06_09*sin(theta_06 - theta_09))
        struct[0].Gy_ini[11,11] = V_05*V_06*(b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06)) + V_06*V_07*(-b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07)) + V_06*V_09*(-b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09))
        struct[0].Gy_ini[11,12] = V_06*(b_06_07*cos(theta_06 - theta_07) - g_06_07*sin(theta_06 - theta_07))
        struct[0].Gy_ini[11,13] = V_06*V_07*(b_06_07*sin(theta_06 - theta_07) + g_06_07*cos(theta_06 - theta_07))
        struct[0].Gy_ini[11,16] = V_06*(b_06_09*cos(theta_06 - theta_09) - g_06_09*sin(theta_06 - theta_09))
        struct[0].Gy_ini[11,17] = V_06*V_09*(b_06_09*sin(theta_06 - theta_09) + g_06_09*cos(theta_06 - theta_09))
        struct[0].Gy_ini[12,10] = V_07*(b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07))
        struct[0].Gy_ini[12,11] = V_06*V_07*(b_06_07*cos(theta_06 - theta_07) + g_06_07*sin(theta_06 - theta_07))
        struct[0].Gy_ini[12,12] = V_06*(b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07)) + 2*V_07*(g_06_07 + g_07_08) + V_08*(-b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08))
        struct[0].Gy_ini[12,13] = V_06*V_07*(-b_06_07*cos(theta_06 - theta_07) - g_06_07*sin(theta_06 - theta_07)) + V_07*V_08*(-b_07_08*cos(theta_07 - theta_08) + g_07_08*sin(theta_07 - theta_08))
        struct[0].Gy_ini[12,14] = V_07*(-b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08))
        struct[0].Gy_ini[12,15] = V_07*V_08*(b_07_08*cos(theta_07 - theta_08) - g_07_08*sin(theta_07 - theta_08))
        struct[0].Gy_ini[13,10] = V_07*(b_06_07*cos(theta_06 - theta_07) + g_06_07*sin(theta_06 - theta_07))
        struct[0].Gy_ini[13,11] = V_06*V_07*(-b_06_07*sin(theta_06 - theta_07) + g_06_07*cos(theta_06 - theta_07))
        struct[0].Gy_ini[13,12] = V_06*(b_06_07*cos(theta_06 - theta_07) + g_06_07*sin(theta_06 - theta_07)) + 2*V_07*(-b_06_07 - b_07_08 - bs_06_07/2 - bs_07_08/2) + V_08*(b_07_08*cos(theta_07 - theta_08) - g_07_08*sin(theta_07 - theta_08))
        struct[0].Gy_ini[13,13] = V_06*V_07*(b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07)) + V_07*V_08*(-b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08))
        struct[0].Gy_ini[13,14] = V_07*(b_07_08*cos(theta_07 - theta_08) - g_07_08*sin(theta_07 - theta_08))
        struct[0].Gy_ini[13,15] = V_07*V_08*(b_07_08*sin(theta_07 - theta_08) + g_07_08*cos(theta_07 - theta_08))
        struct[0].Gy_ini[14,12] = V_08*(b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08))
        struct[0].Gy_ini[14,13] = V_07*V_08*(b_07_08*cos(theta_07 - theta_08) + g_07_08*sin(theta_07 - theta_08))
        struct[0].Gy_ini[14,14] = V_07*(b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08)) + 2*V_08*g_07_08
        struct[0].Gy_ini[14,15] = V_07*V_08*(-b_07_08*cos(theta_07 - theta_08) - g_07_08*sin(theta_07 - theta_08))
        struct[0].Gy_ini[14,28] = -S_n_08/S_base
        struct[0].Gy_ini[15,12] = V_08*(b_07_08*cos(theta_07 - theta_08) + g_07_08*sin(theta_07 - theta_08))
        struct[0].Gy_ini[15,13] = V_07*V_08*(-b_07_08*sin(theta_07 - theta_08) + g_07_08*cos(theta_07 - theta_08))
        struct[0].Gy_ini[15,14] = V_07*(b_07_08*cos(theta_07 - theta_08) + g_07_08*sin(theta_07 - theta_08)) + 2*V_08*(-b_07_08 - bs_07_08/2)
        struct[0].Gy_ini[15,15] = V_07*V_08*(b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08))
        struct[0].Gy_ini[15,29] = -S_n_08/S_base
        struct[0].Gy_ini[16,10] = V_09*(b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09))
        struct[0].Gy_ini[16,11] = V_06*V_09*(b_06_09*cos(theta_06 - theta_09) + g_06_09*sin(theta_06 - theta_09))
        struct[0].Gy_ini[16,16] = V_06*(b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09)) + 2*V_09*g_06_09
        struct[0].Gy_ini[16,17] = V_06*V_09*(-b_06_09*cos(theta_06 - theta_09) - g_06_09*sin(theta_06 - theta_09))
        struct[0].Gy_ini[16,43] = -S_n_09/S_base
        struct[0].Gy_ini[17,10] = V_09*(b_06_09*cos(theta_06 - theta_09) + g_06_09*sin(theta_06 - theta_09))
        struct[0].Gy_ini[17,11] = V_06*V_09*(-b_06_09*sin(theta_06 - theta_09) + g_06_09*cos(theta_06 - theta_09))
        struct[0].Gy_ini[17,16] = V_06*(b_06_09*cos(theta_06 - theta_09) + g_06_09*sin(theta_06 - theta_09)) + 2*V_09*(-b_06_09 - bs_06_09/2)
        struct[0].Gy_ini[17,17] = V_06*V_09*(b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09))
        struct[0].Gy_ini[17,44] = -S_n_09/S_base
        struct[0].Gy_ini[18,8] = V_10*(b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10))
        struct[0].Gy_ini[18,9] = V_05*V_10*(b_05_10*cos(theta_05 - theta_10) + g_05_10*sin(theta_05 - theta_10))
        struct[0].Gy_ini[18,18] = V_05*(b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10)) + 2*V_10*g_05_10
        struct[0].Gy_ini[18,19] = V_05*V_10*(-b_05_10*cos(theta_05 - theta_10) - g_05_10*sin(theta_05 - theta_10))
        struct[0].Gy_ini[18,58] = -S_n_10/S_base
        struct[0].Gy_ini[19,8] = V_10*(b_05_10*cos(theta_05 - theta_10) + g_05_10*sin(theta_05 - theta_10))
        struct[0].Gy_ini[19,9] = V_05*V_10*(-b_05_10*sin(theta_05 - theta_10) + g_05_10*cos(theta_05 - theta_10))
        struct[0].Gy_ini[19,18] = V_05*(b_05_10*cos(theta_05 - theta_10) + g_05_10*sin(theta_05 - theta_10)) + 2*V_10*(-b_05_10 - bs_05_10/2)
        struct[0].Gy_ini[19,19] = V_05*V_10*(b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10))
        struct[0].Gy_ini[19,59] = -S_n_10/S_base
        struct[0].Gy_ini[20,6] = V_11*(b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11))
        struct[0].Gy_ini[20,7] = V_04*V_11*(b_04_11*cos(theta_04 - theta_11) + g_04_11*sin(theta_04 - theta_11))
        struct[0].Gy_ini[20,20] = V_04*(b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11)) + 2*V_11*g_04_11
        struct[0].Gy_ini[20,21] = V_04*V_11*(-b_04_11*cos(theta_04 - theta_11) - g_04_11*sin(theta_04 - theta_11))
        struct[0].Gy_ini[21,6] = V_11*(b_04_11*cos(theta_04 - theta_11) + g_04_11*sin(theta_04 - theta_11))
        struct[0].Gy_ini[21,7] = V_04*V_11*(-b_04_11*sin(theta_04 - theta_11) + g_04_11*cos(theta_04 - theta_11))
        struct[0].Gy_ini[21,20] = V_04*(b_04_11*cos(theta_04 - theta_11) + g_04_11*sin(theta_04 - theta_11)) + 2*V_11*(-b_04_11 - bs_04_11/2)
        struct[0].Gy_ini[21,21] = V_04*V_11*(b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11))
        struct[0].Gy_ini[22,4] = V_12*(b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12))
        struct[0].Gy_ini[22,5] = V_03*V_12*(b_03_12*cos(theta_03 - theta_12) + g_03_12*sin(theta_03 - theta_12))
        struct[0].Gy_ini[22,22] = V_03*(b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12)) + 2*V_12*g_03_12
        struct[0].Gy_ini[22,23] = V_03*V_12*(-b_03_12*cos(theta_03 - theta_12) - g_03_12*sin(theta_03 - theta_12))
        struct[0].Gy_ini[23,4] = V_12*(b_03_12*cos(theta_03 - theta_12) + g_03_12*sin(theta_03 - theta_12))
        struct[0].Gy_ini[23,5] = V_03*V_12*(-b_03_12*sin(theta_03 - theta_12) + g_03_12*cos(theta_03 - theta_12))
        struct[0].Gy_ini[23,22] = V_03*(b_03_12*cos(theta_03 - theta_12) + g_03_12*sin(theta_03 - theta_12)) + 2*V_12*(-b_03_12 - bs_03_12/2)
        struct[0].Gy_ini[23,23] = V_03*V_12*(b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12))
        struct[0].Gy_ini[24,28] = -K_p_08
        struct[0].Gy_ini[24,30] = K_p_08
        struct[0].Gy_ini[25,29] = -K_q_08
        struct[0].Gy_ini[26,14] = -sin(delta_08 - theta_08)
        struct[0].Gy_ini[26,15] = V_08*cos(delta_08 - theta_08)
        struct[0].Gy_ini[26,26] = -R_v_08
        struct[0].Gy_ini[26,27] = X_v_08
        struct[0].Gy_ini[27,14] = -cos(delta_08 - theta_08)
        struct[0].Gy_ini[27,15] = -V_08*sin(delta_08 - theta_08)
        struct[0].Gy_ini[27,26] = -X_v_08
        struct[0].Gy_ini[27,27] = -R_v_08
        struct[0].Gy_ini[28,14] = i_d_08*sin(delta_08 - theta_08) + i_q_08*cos(delta_08 - theta_08)
        struct[0].Gy_ini[28,15] = -V_08*i_d_08*cos(delta_08 - theta_08) + V_08*i_q_08*sin(delta_08 - theta_08)
        struct[0].Gy_ini[28,26] = V_08*sin(delta_08 - theta_08)
        struct[0].Gy_ini[28,27] = V_08*cos(delta_08 - theta_08)
        struct[0].Gy_ini[29,14] = i_d_08*cos(delta_08 - theta_08) - i_q_08*sin(delta_08 - theta_08)
        struct[0].Gy_ini[29,15] = V_08*i_d_08*sin(delta_08 - theta_08) + V_08*i_q_08*cos(delta_08 - theta_08)
        struct[0].Gy_ini[29,26] = V_08*cos(delta_08 - theta_08)
        struct[0].Gy_ini[29,27] = -V_08*sin(delta_08 - theta_08)
        struct[0].Gy_ini[30,37] = K_fpfr_08*Piecewise(np.array([(0, (P_f_min_08 > p_f_08) | (P_f_max_08 < p_f_08)), (1, True)]))
        struct[0].Gy_ini[31,14] = i_d_08*sin(delta_08 - theta_08) + i_q_08*cos(delta_08 - theta_08)
        struct[0].Gy_ini[31,15] = -V_08*i_d_08*cos(delta_08 - theta_08) + V_08*i_q_08*sin(delta_08 - theta_08)
        struct[0].Gy_ini[31,26] = 2*R_s_08*i_d_08 + V_08*sin(delta_08 - theta_08)
        struct[0].Gy_ini[31,27] = 2*R_s_08*i_q_08 + V_08*cos(delta_08 - theta_08)
        struct[0].Gy_ini[32,33] = 2*k_u_08*v_u_08/V_u_max_08**2
        struct[0].Gy_ini[32,34] = -(-v_u_08**2 + v_u_ref_08**2)/V_u_max_08**2
        struct[0].Gy_ini[33,31] = -R_uc_08*S_n_08/(v_u_08 + 0.1)
        struct[0].Gy_ini[33,33] = -R_uc_08*S_n_08*(p_gou_08 - p_t_08)/(v_u_08 + 0.1)**2 - 1
        struct[0].Gy_ini[33,36] = R_uc_08*S_n_08/(v_u_08 + 0.1)
        struct[0].Gy_ini[34,33] = Piecewise(np.array([(0, V_u_min_08 > v_u_08), ((-K_u_0_08 + K_u_max_08)/(-V_u_lt_08 + V_u_min_08), V_u_lt_08 > v_u_08), ((-K_u_0_08 + K_u_max_08)/(-V_u_ht_08 + V_u_max_08), V_u_ht_08 < v_u_08), (0, True)]))
        struct[0].Gy_ini[36,35] = inc_p_gin_08 + p_gin_0_08
        struct[0].Gy_ini[37,24] = -Piecewise(np.array([(1/Droop_08, (omega_08 > 0.5*DB_08 + omega_ref_08) | (omega_08 < -0.5*DB_08 + omega_ref_08)), (0, True)]))
        struct[0].Gy_ini[38,33] = Piecewise(np.array([((-R_lim_08 + R_lim_max_08)/(-V_u_lt_08 + V_u_min_08), V_u_lt_08 > v_u_08), ((-R_lim_08 + R_lim_max_08)/(-V_u_ht_08 + V_u_max_08), V_u_ht_08 < v_u_08), (0, True)]))
        struct[0].Gy_ini[39,43] = -K_p_09
        struct[0].Gy_ini[39,45] = K_p_09
        struct[0].Gy_ini[40,44] = -K_q_09
        struct[0].Gy_ini[41,16] = -sin(delta_09 - theta_09)
        struct[0].Gy_ini[41,17] = V_09*cos(delta_09 - theta_09)
        struct[0].Gy_ini[41,41] = -R_v_09
        struct[0].Gy_ini[41,42] = X_v_09
        struct[0].Gy_ini[42,16] = -cos(delta_09 - theta_09)
        struct[0].Gy_ini[42,17] = -V_09*sin(delta_09 - theta_09)
        struct[0].Gy_ini[42,41] = -X_v_09
        struct[0].Gy_ini[42,42] = -R_v_09
        struct[0].Gy_ini[43,16] = i_d_09*sin(delta_09 - theta_09) + i_q_09*cos(delta_09 - theta_09)
        struct[0].Gy_ini[43,17] = -V_09*i_d_09*cos(delta_09 - theta_09) + V_09*i_q_09*sin(delta_09 - theta_09)
        struct[0].Gy_ini[43,41] = V_09*sin(delta_09 - theta_09)
        struct[0].Gy_ini[43,42] = V_09*cos(delta_09 - theta_09)
        struct[0].Gy_ini[44,16] = i_d_09*cos(delta_09 - theta_09) - i_q_09*sin(delta_09 - theta_09)
        struct[0].Gy_ini[44,17] = V_09*i_d_09*sin(delta_09 - theta_09) + V_09*i_q_09*cos(delta_09 - theta_09)
        struct[0].Gy_ini[44,41] = V_09*cos(delta_09 - theta_09)
        struct[0].Gy_ini[44,42] = -V_09*sin(delta_09 - theta_09)
        struct[0].Gy_ini[45,52] = K_fpfr_09*Piecewise(np.array([(0, (P_f_min_09 > p_f_09) | (P_f_max_09 < p_f_09)), (1, True)]))
        struct[0].Gy_ini[46,16] = i_d_09*sin(delta_09 - theta_09) + i_q_09*cos(delta_09 - theta_09)
        struct[0].Gy_ini[46,17] = -V_09*i_d_09*cos(delta_09 - theta_09) + V_09*i_q_09*sin(delta_09 - theta_09)
        struct[0].Gy_ini[46,41] = 2*R_s_09*i_d_09 + V_09*sin(delta_09 - theta_09)
        struct[0].Gy_ini[46,42] = 2*R_s_09*i_q_09 + V_09*cos(delta_09 - theta_09)
        struct[0].Gy_ini[47,48] = 2*k_u_09*v_u_09/V_u_max_09**2
        struct[0].Gy_ini[47,49] = -(-v_u_09**2 + v_u_ref_09**2)/V_u_max_09**2
        struct[0].Gy_ini[48,46] = -R_uc_09*S_n_09/(v_u_09 + 0.1)
        struct[0].Gy_ini[48,48] = -R_uc_09*S_n_09*(p_gou_09 - p_t_09)/(v_u_09 + 0.1)**2 - 1
        struct[0].Gy_ini[48,51] = R_uc_09*S_n_09/(v_u_09 + 0.1)
        struct[0].Gy_ini[49,48] = Piecewise(np.array([(0, V_u_min_09 > v_u_09), ((-K_u_0_09 + K_u_max_09)/(-V_u_lt_09 + V_u_min_09), V_u_lt_09 > v_u_09), ((-K_u_0_09 + K_u_max_09)/(-V_u_ht_09 + V_u_max_09), V_u_ht_09 < v_u_09), (0, True)]))
        struct[0].Gy_ini[51,50] = inc_p_gin_09 + p_gin_0_09
        struct[0].Gy_ini[52,39] = -Piecewise(np.array([(1/Droop_09, (omega_09 > 0.5*DB_09 + omega_ref_09) | (omega_09 < -0.5*DB_09 + omega_ref_09)), (0, True)]))
        struct[0].Gy_ini[53,48] = Piecewise(np.array([((-R_lim_09 + R_lim_max_09)/(-V_u_lt_09 + V_u_min_09), V_u_lt_09 > v_u_09), ((-R_lim_09 + R_lim_max_09)/(-V_u_ht_09 + V_u_max_09), V_u_ht_09 < v_u_09), (0, True)]))
        struct[0].Gy_ini[54,58] = -K_p_10
        struct[0].Gy_ini[54,60] = K_p_10
        struct[0].Gy_ini[55,59] = -K_q_10
        struct[0].Gy_ini[56,18] = -sin(delta_10 - theta_10)
        struct[0].Gy_ini[56,19] = V_10*cos(delta_10 - theta_10)
        struct[0].Gy_ini[56,56] = -R_v_10
        struct[0].Gy_ini[56,57] = X_v_10
        struct[0].Gy_ini[57,18] = -cos(delta_10 - theta_10)
        struct[0].Gy_ini[57,19] = -V_10*sin(delta_10 - theta_10)
        struct[0].Gy_ini[57,56] = -X_v_10
        struct[0].Gy_ini[57,57] = -R_v_10
        struct[0].Gy_ini[58,18] = i_d_10*sin(delta_10 - theta_10) + i_q_10*cos(delta_10 - theta_10)
        struct[0].Gy_ini[58,19] = -V_10*i_d_10*cos(delta_10 - theta_10) + V_10*i_q_10*sin(delta_10 - theta_10)
        struct[0].Gy_ini[58,56] = V_10*sin(delta_10 - theta_10)
        struct[0].Gy_ini[58,57] = V_10*cos(delta_10 - theta_10)
        struct[0].Gy_ini[59,18] = i_d_10*cos(delta_10 - theta_10) - i_q_10*sin(delta_10 - theta_10)
        struct[0].Gy_ini[59,19] = V_10*i_d_10*sin(delta_10 - theta_10) + V_10*i_q_10*cos(delta_10 - theta_10)
        struct[0].Gy_ini[59,56] = V_10*cos(delta_10 - theta_10)
        struct[0].Gy_ini[59,57] = -V_10*sin(delta_10 - theta_10)
        struct[0].Gy_ini[60,67] = K_fpfr_10*Piecewise(np.array([(0, (P_f_min_10 > p_f_10) | (P_f_max_10 < p_f_10)), (1, True)]))
        struct[0].Gy_ini[61,18] = i_d_10*sin(delta_10 - theta_10) + i_q_10*cos(delta_10 - theta_10)
        struct[0].Gy_ini[61,19] = -V_10*i_d_10*cos(delta_10 - theta_10) + V_10*i_q_10*sin(delta_10 - theta_10)
        struct[0].Gy_ini[61,56] = 2*R_s_10*i_d_10 + V_10*sin(delta_10 - theta_10)
        struct[0].Gy_ini[61,57] = 2*R_s_10*i_q_10 + V_10*cos(delta_10 - theta_10)
        struct[0].Gy_ini[62,63] = 2*k_u_10*v_u_10/V_u_max_10**2
        struct[0].Gy_ini[62,64] = -(-v_u_10**2 + v_u_ref_10**2)/V_u_max_10**2
        struct[0].Gy_ini[63,61] = -R_uc_10*S_n_10/(v_u_10 + 0.1)
        struct[0].Gy_ini[63,63] = -R_uc_10*S_n_10*(p_gou_10 - p_t_10)/(v_u_10 + 0.1)**2 - 1
        struct[0].Gy_ini[63,66] = R_uc_10*S_n_10/(v_u_10 + 0.1)
        struct[0].Gy_ini[64,63] = Piecewise(np.array([(0, V_u_min_10 > v_u_10), ((-K_u_0_10 + K_u_max_10)/(-V_u_lt_10 + V_u_min_10), V_u_lt_10 > v_u_10), ((-K_u_0_10 + K_u_max_10)/(-V_u_ht_10 + V_u_max_10), V_u_ht_10 < v_u_10), (0, True)]))
        struct[0].Gy_ini[66,65] = inc_p_gin_10 + p_gin_0_10
        struct[0].Gy_ini[67,54] = -Piecewise(np.array([(1/Droop_10, (omega_10 > 0.5*DB_10 + omega_ref_10) | (omega_10 < -0.5*DB_10 + omega_ref_10)), (0, True)]))
        struct[0].Gy_ini[68,63] = Piecewise(np.array([((-R_lim_10 + R_lim_max_10)/(-V_u_lt_10 + V_u_min_10), V_u_lt_10 > v_u_10), ((-R_lim_10 + R_lim_max_10)/(-V_u_ht_10 + V_u_max_10), V_u_ht_10 < v_u_10), (0, True)]))
        struct[0].Gy_ini[70,0] = -sin(delta_01 - theta_01)
        struct[0].Gy_ini[70,1] = V_01*cos(delta_01 - theta_01)
        struct[0].Gy_ini[70,70] = -R_v_01
        struct[0].Gy_ini[70,71] = X_v_01
        struct[0].Gy_ini[71,0] = -cos(delta_01 - theta_01)
        struct[0].Gy_ini[71,1] = -V_01*sin(delta_01 - theta_01)
        struct[0].Gy_ini[71,70] = -X_v_01
        struct[0].Gy_ini[71,71] = -R_v_01
        struct[0].Gy_ini[72,0] = i_d_01*sin(delta_01 - theta_01) + i_q_01*cos(delta_01 - theta_01)
        struct[0].Gy_ini[72,1] = -V_01*i_d_01*cos(delta_01 - theta_01) + V_01*i_q_01*sin(delta_01 - theta_01)
        struct[0].Gy_ini[72,70] = V_01*sin(delta_01 - theta_01)
        struct[0].Gy_ini[72,71] = V_01*cos(delta_01 - theta_01)
        struct[0].Gy_ini[73,0] = i_d_01*cos(delta_01 - theta_01) - i_q_01*sin(delta_01 - theta_01)
        struct[0].Gy_ini[73,1] = V_01*i_d_01*sin(delta_01 - theta_01) + V_01*i_q_01*cos(delta_01 - theta_01)
        struct[0].Gy_ini[73,70] = V_01*cos(delta_01 - theta_01)
        struct[0].Gy_ini[73,71] = -V_01*sin(delta_01 - theta_01)
        struct[0].Gy_ini[74,24] = S_n_08*T_p_08/(2*K_p_08*(1000000.0*S_n_01 + S_n_10*T_p_10/(2*K_p_10) + S_n_09*T_p_09/(2*K_p_09) + S_n_08*T_p_08/(2*K_p_08)))
        struct[0].Gy_ini[74,39] = S_n_09*T_p_09/(2*K_p_09*(1000000.0*S_n_01 + S_n_10*T_p_10/(2*K_p_10) + S_n_09*T_p_09/(2*K_p_09) + S_n_08*T_p_08/(2*K_p_08)))
        struct[0].Gy_ini[74,54] = S_n_10*T_p_10/(2*K_p_10*(1000000.0*S_n_01 + S_n_10*T_p_10/(2*K_p_10) + S_n_09*T_p_09/(2*K_p_09) + S_n_08*T_p_08/(2*K_p_08)))
        struct[0].Gy_ini[74,69] = 1000000.0*S_n_01/(1000000.0*S_n_01 + S_n_10*T_p_10/(2*K_p_10) + S_n_09*T_p_09/(2*K_p_09) + S_n_08*T_p_08/(2*K_p_08))
        struct[0].Gy_ini[75,74] = -K_p_agc



@numba.njit(cache=True)
def run(t,struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_01_02 = struct[0].g_01_02
    b_01_02 = struct[0].b_01_02
    bs_01_02 = struct[0].bs_01_02
    g_02_03 = struct[0].g_02_03
    b_02_03 = struct[0].b_02_03
    bs_02_03 = struct[0].bs_02_03
    g_03_04 = struct[0].g_03_04
    b_03_04 = struct[0].b_03_04
    bs_03_04 = struct[0].bs_03_04
    g_04_05 = struct[0].g_04_05
    b_04_05 = struct[0].b_04_05
    bs_04_05 = struct[0].bs_04_05
    g_05_06 = struct[0].g_05_06
    b_05_06 = struct[0].b_05_06
    bs_05_06 = struct[0].bs_05_06
    g_06_07 = struct[0].g_06_07
    b_06_07 = struct[0].b_06_07
    bs_06_07 = struct[0].bs_06_07
    g_07_08 = struct[0].g_07_08
    b_07_08 = struct[0].b_07_08
    bs_07_08 = struct[0].bs_07_08
    g_06_09 = struct[0].g_06_09
    b_06_09 = struct[0].b_06_09
    bs_06_09 = struct[0].bs_06_09
    g_05_10 = struct[0].g_05_10
    b_05_10 = struct[0].b_05_10
    bs_05_10 = struct[0].bs_05_10
    g_04_11 = struct[0].g_04_11
    b_04_11 = struct[0].b_04_11
    bs_04_11 = struct[0].bs_04_11
    g_03_12 = struct[0].g_03_12
    b_03_12 = struct[0].b_03_12
    bs_03_12 = struct[0].bs_03_12
    U_01_n = struct[0].U_01_n
    U_02_n = struct[0].U_02_n
    U_03_n = struct[0].U_03_n
    U_04_n = struct[0].U_04_n
    U_05_n = struct[0].U_05_n
    U_06_n = struct[0].U_06_n
    U_07_n = struct[0].U_07_n
    U_08_n = struct[0].U_08_n
    U_09_n = struct[0].U_09_n
    U_10_n = struct[0].U_10_n
    U_11_n = struct[0].U_11_n
    U_12_n = struct[0].U_12_n
    S_n_08 = struct[0].S_n_08
    Omega_b_08 = struct[0].Omega_b_08
    K_p_08 = struct[0].K_p_08
    T_p_08 = struct[0].T_p_08
    K_q_08 = struct[0].K_q_08
    T_q_08 = struct[0].T_q_08
    X_v_08 = struct[0].X_v_08
    R_v_08 = struct[0].R_v_08
    R_s_08 = struct[0].R_s_08
    C_u_08 = struct[0].C_u_08
    K_u_0_08 = struct[0].K_u_0_08
    K_u_max_08 = struct[0].K_u_max_08
    V_u_min_08 = struct[0].V_u_min_08
    V_u_max_08 = struct[0].V_u_max_08
    R_uc_08 = struct[0].R_uc_08
    K_h_08 = struct[0].K_h_08
    R_lim_08 = struct[0].R_lim_08
    V_u_lt_08 = struct[0].V_u_lt_08
    V_u_ht_08 = struct[0].V_u_ht_08
    Droop_08 = struct[0].Droop_08
    DB_08 = struct[0].DB_08
    T_cur_08 = struct[0].T_cur_08
    R_lim_max_08 = struct[0].R_lim_max_08
    K_fpfr_08 = struct[0].K_fpfr_08
    P_f_min_08 = struct[0].P_f_min_08
    P_f_max_08 = struct[0].P_f_max_08
    S_n_09 = struct[0].S_n_09
    Omega_b_09 = struct[0].Omega_b_09
    K_p_09 = struct[0].K_p_09
    T_p_09 = struct[0].T_p_09
    K_q_09 = struct[0].K_q_09
    T_q_09 = struct[0].T_q_09
    X_v_09 = struct[0].X_v_09
    R_v_09 = struct[0].R_v_09
    R_s_09 = struct[0].R_s_09
    C_u_09 = struct[0].C_u_09
    K_u_0_09 = struct[0].K_u_0_09
    K_u_max_09 = struct[0].K_u_max_09
    V_u_min_09 = struct[0].V_u_min_09
    V_u_max_09 = struct[0].V_u_max_09
    R_uc_09 = struct[0].R_uc_09
    K_h_09 = struct[0].K_h_09
    R_lim_09 = struct[0].R_lim_09
    V_u_lt_09 = struct[0].V_u_lt_09
    V_u_ht_09 = struct[0].V_u_ht_09
    Droop_09 = struct[0].Droop_09
    DB_09 = struct[0].DB_09
    T_cur_09 = struct[0].T_cur_09
    R_lim_max_09 = struct[0].R_lim_max_09
    K_fpfr_09 = struct[0].K_fpfr_09
    P_f_min_09 = struct[0].P_f_min_09
    P_f_max_09 = struct[0].P_f_max_09
    S_n_10 = struct[0].S_n_10
    Omega_b_10 = struct[0].Omega_b_10
    K_p_10 = struct[0].K_p_10
    T_p_10 = struct[0].T_p_10
    K_q_10 = struct[0].K_q_10
    T_q_10 = struct[0].T_q_10
    X_v_10 = struct[0].X_v_10
    R_v_10 = struct[0].R_v_10
    R_s_10 = struct[0].R_s_10
    C_u_10 = struct[0].C_u_10
    K_u_0_10 = struct[0].K_u_0_10
    K_u_max_10 = struct[0].K_u_max_10
    V_u_min_10 = struct[0].V_u_min_10
    V_u_max_10 = struct[0].V_u_max_10
    R_uc_10 = struct[0].R_uc_10
    K_h_10 = struct[0].K_h_10
    R_lim_10 = struct[0].R_lim_10
    V_u_lt_10 = struct[0].V_u_lt_10
    V_u_ht_10 = struct[0].V_u_ht_10
    Droop_10 = struct[0].Droop_10
    DB_10 = struct[0].DB_10
    T_cur_10 = struct[0].T_cur_10
    R_lim_max_10 = struct[0].R_lim_max_10
    K_fpfr_10 = struct[0].K_fpfr_10
    P_f_min_10 = struct[0].P_f_min_10
    P_f_max_10 = struct[0].P_f_max_10
    S_n_01 = struct[0].S_n_01
    Omega_b_01 = struct[0].Omega_b_01
    X_v_01 = struct[0].X_v_01
    R_v_01 = struct[0].R_v_01
    K_delta_01 = struct[0].K_delta_01
    K_alpha_01 = struct[0].K_alpha_01
    K_p_agc = struct[0].K_p_agc
    K_i_agc = struct[0].K_i_agc
    
    # Inputs:
    P_01 = struct[0].P_01
    Q_01 = struct[0].Q_01
    P_02 = struct[0].P_02
    Q_02 = struct[0].Q_02
    P_03 = struct[0].P_03
    Q_03 = struct[0].Q_03
    P_04 = struct[0].P_04
    Q_04 = struct[0].Q_04
    P_05 = struct[0].P_05
    Q_05 = struct[0].Q_05
    P_06 = struct[0].P_06
    Q_06 = struct[0].Q_06
    P_07 = struct[0].P_07
    Q_07 = struct[0].Q_07
    P_08 = struct[0].P_08
    Q_08 = struct[0].Q_08
    P_09 = struct[0].P_09
    Q_09 = struct[0].Q_09
    P_10 = struct[0].P_10
    Q_10 = struct[0].Q_10
    P_11 = struct[0].P_11
    Q_11 = struct[0].Q_11
    P_12 = struct[0].P_12
    Q_12 = struct[0].Q_12
    q_s_ref_08 = struct[0].q_s_ref_08
    v_u_ref_08 = struct[0].v_u_ref_08
    omega_ref_08 = struct[0].omega_ref_08
    p_gin_0_08 = struct[0].p_gin_0_08
    p_g_ref_08 = struct[0].p_g_ref_08
    ramp_p_gin_08 = struct[0].ramp_p_gin_08
    q_s_ref_09 = struct[0].q_s_ref_09
    v_u_ref_09 = struct[0].v_u_ref_09
    omega_ref_09 = struct[0].omega_ref_09
    p_gin_0_09 = struct[0].p_gin_0_09
    p_g_ref_09 = struct[0].p_g_ref_09
    ramp_p_gin_09 = struct[0].ramp_p_gin_09
    q_s_ref_10 = struct[0].q_s_ref_10
    v_u_ref_10 = struct[0].v_u_ref_10
    omega_ref_10 = struct[0].omega_ref_10
    p_gin_0_10 = struct[0].p_gin_0_10
    p_g_ref_10 = struct[0].p_g_ref_10
    ramp_p_gin_10 = struct[0].ramp_p_gin_10
    alpha_01 = struct[0].alpha_01
    e_qv_01 = struct[0].e_qv_01
    omega_ref_01 = struct[0].omega_ref_01
    
    # Dynamical states:
    delta_08 = struct[0].x[0,0]
    xi_p_08 = struct[0].x[1,0]
    xi_q_08 = struct[0].x[2,0]
    e_u_08 = struct[0].x[3,0]
    p_ghr_08 = struct[0].x[4,0]
    k_cur_08 = struct[0].x[5,0]
    inc_p_gin_08 = struct[0].x[6,0]
    delta_09 = struct[0].x[7,0]
    xi_p_09 = struct[0].x[8,0]
    xi_q_09 = struct[0].x[9,0]
    e_u_09 = struct[0].x[10,0]
    p_ghr_09 = struct[0].x[11,0]
    k_cur_09 = struct[0].x[12,0]
    inc_p_gin_09 = struct[0].x[13,0]
    delta_10 = struct[0].x[14,0]
    xi_p_10 = struct[0].x[15,0]
    xi_q_10 = struct[0].x[16,0]
    e_u_10 = struct[0].x[17,0]
    p_ghr_10 = struct[0].x[18,0]
    k_cur_10 = struct[0].x[19,0]
    inc_p_gin_10 = struct[0].x[20,0]
    delta_01 = struct[0].x[21,0]
    Domega_01 = struct[0].x[22,0]
    xi_freq = struct[0].x[23,0]
    
    # Algebraic states:
    V_01 = struct[0].y_run[0,0]
    theta_01 = struct[0].y_run[1,0]
    V_02 = struct[0].y_run[2,0]
    theta_02 = struct[0].y_run[3,0]
    V_03 = struct[0].y_run[4,0]
    theta_03 = struct[0].y_run[5,0]
    V_04 = struct[0].y_run[6,0]
    theta_04 = struct[0].y_run[7,0]
    V_05 = struct[0].y_run[8,0]
    theta_05 = struct[0].y_run[9,0]
    V_06 = struct[0].y_run[10,0]
    theta_06 = struct[0].y_run[11,0]
    V_07 = struct[0].y_run[12,0]
    theta_07 = struct[0].y_run[13,0]
    V_08 = struct[0].y_run[14,0]
    theta_08 = struct[0].y_run[15,0]
    V_09 = struct[0].y_run[16,0]
    theta_09 = struct[0].y_run[17,0]
    V_10 = struct[0].y_run[18,0]
    theta_10 = struct[0].y_run[19,0]
    V_11 = struct[0].y_run[20,0]
    theta_11 = struct[0].y_run[21,0]
    V_12 = struct[0].y_run[22,0]
    theta_12 = struct[0].y_run[23,0]
    omega_08 = struct[0].y_run[24,0]
    e_qv_08 = struct[0].y_run[25,0]
    i_d_08 = struct[0].y_run[26,0]
    i_q_08 = struct[0].y_run[27,0]
    p_s_08 = struct[0].y_run[28,0]
    q_s_08 = struct[0].y_run[29,0]
    p_m_08 = struct[0].y_run[30,0]
    p_t_08 = struct[0].y_run[31,0]
    p_u_08 = struct[0].y_run[32,0]
    v_u_08 = struct[0].y_run[33,0]
    k_u_08 = struct[0].y_run[34,0]
    k_cur_sat_08 = struct[0].y_run[35,0]
    p_gou_08 = struct[0].y_run[36,0]
    p_f_08 = struct[0].y_run[37,0]
    r_lim_08 = struct[0].y_run[38,0]
    omega_09 = struct[0].y_run[39,0]
    e_qv_09 = struct[0].y_run[40,0]
    i_d_09 = struct[0].y_run[41,0]
    i_q_09 = struct[0].y_run[42,0]
    p_s_09 = struct[0].y_run[43,0]
    q_s_09 = struct[0].y_run[44,0]
    p_m_09 = struct[0].y_run[45,0]
    p_t_09 = struct[0].y_run[46,0]
    p_u_09 = struct[0].y_run[47,0]
    v_u_09 = struct[0].y_run[48,0]
    k_u_09 = struct[0].y_run[49,0]
    k_cur_sat_09 = struct[0].y_run[50,0]
    p_gou_09 = struct[0].y_run[51,0]
    p_f_09 = struct[0].y_run[52,0]
    r_lim_09 = struct[0].y_run[53,0]
    omega_10 = struct[0].y_run[54,0]
    e_qv_10 = struct[0].y_run[55,0]
    i_d_10 = struct[0].y_run[56,0]
    i_q_10 = struct[0].y_run[57,0]
    p_s_10 = struct[0].y_run[58,0]
    q_s_10 = struct[0].y_run[59,0]
    p_m_10 = struct[0].y_run[60,0]
    p_t_10 = struct[0].y_run[61,0]
    p_u_10 = struct[0].y_run[62,0]
    v_u_10 = struct[0].y_run[63,0]
    k_u_10 = struct[0].y_run[64,0]
    k_cur_sat_10 = struct[0].y_run[65,0]
    p_gou_10 = struct[0].y_run[66,0]
    p_f_10 = struct[0].y_run[67,0]
    r_lim_10 = struct[0].y_run[68,0]
    omega_01 = struct[0].y_run[69,0]
    i_d_01 = struct[0].y_run[70,0]
    i_q_01 = struct[0].y_run[71,0]
    p_s_01 = struct[0].y_run[72,0]
    q_s_01 = struct[0].y_run[73,0]
    omega_coi = struct[0].y_run[74,0]
    p_agc = struct[0].y_run[75,0]
    
    struct[0].u_run[0,0] = P_01
    struct[0].u_run[1,0] = Q_01
    struct[0].u_run[2,0] = P_02
    struct[0].u_run[3,0] = Q_02
    struct[0].u_run[4,0] = P_03
    struct[0].u_run[5,0] = Q_03
    struct[0].u_run[6,0] = P_04
    struct[0].u_run[7,0] = Q_04
    struct[0].u_run[8,0] = P_05
    struct[0].u_run[9,0] = Q_05
    struct[0].u_run[10,0] = P_06
    struct[0].u_run[11,0] = Q_06
    struct[0].u_run[12,0] = P_07
    struct[0].u_run[13,0] = Q_07
    struct[0].u_run[14,0] = P_08
    struct[0].u_run[15,0] = Q_08
    struct[0].u_run[16,0] = P_09
    struct[0].u_run[17,0] = Q_09
    struct[0].u_run[18,0] = P_10
    struct[0].u_run[19,0] = Q_10
    struct[0].u_run[20,0] = P_11
    struct[0].u_run[21,0] = Q_11
    struct[0].u_run[22,0] = P_12
    struct[0].u_run[23,0] = Q_12
    struct[0].u_run[24,0] = q_s_ref_08
    struct[0].u_run[25,0] = v_u_ref_08
    struct[0].u_run[26,0] = omega_ref_08
    struct[0].u_run[27,0] = p_gin_0_08
    struct[0].u_run[28,0] = p_g_ref_08
    struct[0].u_run[29,0] = ramp_p_gin_08
    struct[0].u_run[30,0] = q_s_ref_09
    struct[0].u_run[31,0] = v_u_ref_09
    struct[0].u_run[32,0] = omega_ref_09
    struct[0].u_run[33,0] = p_gin_0_09
    struct[0].u_run[34,0] = p_g_ref_09
    struct[0].u_run[35,0] = ramp_p_gin_09
    struct[0].u_run[36,0] = q_s_ref_10
    struct[0].u_run[37,0] = v_u_ref_10
    struct[0].u_run[38,0] = omega_ref_10
    struct[0].u_run[39,0] = p_gin_0_10
    struct[0].u_run[40,0] = p_g_ref_10
    struct[0].u_run[41,0] = ramp_p_gin_10
    struct[0].u_run[42,0] = alpha_01
    struct[0].u_run[43,0] = e_qv_01
    struct[0].u_run[44,0] = omega_ref_01
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = Omega_b_08*(omega_08 - omega_coi)
        struct[0].f[1,0] = p_m_08 - p_s_08
        struct[0].f[2,0] = -q_s_08 + q_s_ref_08
        struct[0].f[3,0] = S_n_08*(p_gou_08 - p_t_08)/(C_u_08*(v_u_08 + 0.1))
        struct[0].f[4,0] = Piecewise(np.array([(-r_lim_08, r_lim_08 < -K_h_08*(-p_ghr_08 + p_gou_08)), (r_lim_08, r_lim_08 < K_h_08*(-p_ghr_08 + p_gou_08)), (K_h_08*(-p_ghr_08 + p_gou_08), True)]))
        struct[0].f[5,0] = (-k_cur_08 + p_g_ref_08/(inc_p_gin_08 + p_gin_0_08) + Piecewise(np.array([(P_f_min_08, P_f_min_08 > p_f_08), (P_f_max_08, P_f_max_08 < p_f_08), (p_f_08, True)]))/(inc_p_gin_08 + p_gin_0_08))/T_cur_08
        struct[0].f[6,0] = -0.001*inc_p_gin_08 + ramp_p_gin_08
        struct[0].f[7,0] = Omega_b_09*(omega_09 - omega_coi)
        struct[0].f[8,0] = p_m_09 - p_s_09
        struct[0].f[9,0] = -q_s_09 + q_s_ref_09
        struct[0].f[10,0] = S_n_09*(p_gou_09 - p_t_09)/(C_u_09*(v_u_09 + 0.1))
        struct[0].f[11,0] = Piecewise(np.array([(-r_lim_09, r_lim_09 < -K_h_09*(-p_ghr_09 + p_gou_09)), (r_lim_09, r_lim_09 < K_h_09*(-p_ghr_09 + p_gou_09)), (K_h_09*(-p_ghr_09 + p_gou_09), True)]))
        struct[0].f[12,0] = (-k_cur_09 + p_g_ref_09/(inc_p_gin_09 + p_gin_0_09) + Piecewise(np.array([(P_f_min_09, P_f_min_09 > p_f_09), (P_f_max_09, P_f_max_09 < p_f_09), (p_f_09, True)]))/(inc_p_gin_09 + p_gin_0_09))/T_cur_09
        struct[0].f[13,0] = -0.001*inc_p_gin_09 + ramp_p_gin_09
        struct[0].f[14,0] = Omega_b_10*(omega_10 - omega_coi)
        struct[0].f[15,0] = p_m_10 - p_s_10
        struct[0].f[16,0] = -q_s_10 + q_s_ref_10
        struct[0].f[17,0] = S_n_10*(p_gou_10 - p_t_10)/(C_u_10*(v_u_10 + 0.1))
        struct[0].f[18,0] = Piecewise(np.array([(-r_lim_10, r_lim_10 < -K_h_10*(-p_ghr_10 + p_gou_10)), (r_lim_10, r_lim_10 < K_h_10*(-p_ghr_10 + p_gou_10)), (K_h_10*(-p_ghr_10 + p_gou_10), True)]))
        struct[0].f[19,0] = (-k_cur_10 + p_g_ref_10/(inc_p_gin_10 + p_gin_0_10) + Piecewise(np.array([(P_f_min_10, P_f_min_10 > p_f_10), (P_f_max_10, P_f_max_10 < p_f_10), (p_f_10, True)]))/(inc_p_gin_10 + p_gin_0_10))/T_cur_10
        struct[0].f[20,0] = -0.001*inc_p_gin_10 + ramp_p_gin_10
        struct[0].f[21,0] = -K_delta_01*delta_01 + Omega_b_01*(omega_01 - omega_coi)
        struct[0].f[22,0] = -Domega_01*K_alpha_01 + alpha_01
        struct[0].f[23,0] = 1 - omega_coi
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy) @ np.ascontiguousarray(struct[0].y_run) + np.ascontiguousarray(struct[0].Gu) @ np.ascontiguousarray(struct[0].u_run)

        struct[0].g[0,0] = -P_01/S_base + V_01**2*g_01_02 + V_01*V_02*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) - S_n_01*p_s_01/S_base
        struct[0].g[1,0] = -Q_01/S_base + V_01**2*(-b_01_02 - bs_01_02/2) + V_01*V_02*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02)) - S_n_01*q_s_01/S_base
        struct[0].g[2,0] = -P_02/S_base + V_01*V_02*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) + V_02**2*(g_01_02 + g_02_03) + V_02*V_03*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].g[3,0] = -Q_02/S_base + V_01*V_02*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02)) + V_02**2*(-b_01_02 - b_02_03 - bs_01_02/2 - bs_02_03/2) + V_02*V_03*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].g[4,0] = -P_03/S_base + V_02*V_03*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03)) + V_03**2*(g_02_03 + g_03_04 + g_03_12) + V_03*V_04*(-b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04)) + V_03*V_12*(-b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12))
        struct[0].g[5,0] = -Q_03/S_base + V_02*V_03*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03)) + V_03**2*(-b_02_03 - b_03_04 - b_03_12 - bs_02_03/2 - bs_03_04/2 - bs_03_12/2) + V_03*V_04*(b_03_04*cos(theta_03 - theta_04) - g_03_04*sin(theta_03 - theta_04)) + V_03*V_12*(b_03_12*cos(theta_03 - theta_12) - g_03_12*sin(theta_03 - theta_12))
        struct[0].g[6,0] = -P_04/S_base + V_03*V_04*(b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04)) + V_04**2*(g_03_04 + g_04_05 + g_04_11) + V_04*V_05*(-b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05)) + V_04*V_11*(-b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11))
        struct[0].g[7,0] = -Q_04/S_base + V_03*V_04*(b_03_04*cos(theta_03 - theta_04) + g_03_04*sin(theta_03 - theta_04)) + V_04**2*(-b_03_04 - b_04_05 - b_04_11 - bs_03_04/2 - bs_04_05/2 - bs_04_11/2) + V_04*V_05*(b_04_05*cos(theta_04 - theta_05) - g_04_05*sin(theta_04 - theta_05)) + V_04*V_11*(b_04_11*cos(theta_04 - theta_11) - g_04_11*sin(theta_04 - theta_11))
        struct[0].g[8,0] = -P_05/S_base + V_04*V_05*(b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05)) + V_05**2*(g_04_05 + g_05_06 + g_05_10) + V_05*V_06*(-b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06)) + V_05*V_10*(-b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10))
        struct[0].g[9,0] = -Q_05/S_base + V_04*V_05*(b_04_05*cos(theta_04 - theta_05) + g_04_05*sin(theta_04 - theta_05)) + V_05**2*(-b_04_05 - b_05_06 - b_05_10 - bs_04_05/2 - bs_05_06/2 - bs_05_10/2) + V_05*V_06*(b_05_06*cos(theta_05 - theta_06) - g_05_06*sin(theta_05 - theta_06)) + V_05*V_10*(b_05_10*cos(theta_05 - theta_10) - g_05_10*sin(theta_05 - theta_10))
        struct[0].g[10,0] = -P_06/S_base + V_05*V_06*(b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06)) + V_06**2*(g_05_06 + g_06_07 + g_06_09) + V_06*V_07*(-b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07)) + V_06*V_09*(-b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09))
        struct[0].g[11,0] = -Q_06/S_base + V_05*V_06*(b_05_06*cos(theta_05 - theta_06) + g_05_06*sin(theta_05 - theta_06)) + V_06**2*(-b_05_06 - b_06_07 - b_06_09 - bs_05_06/2 - bs_06_07/2 - bs_06_09/2) + V_06*V_07*(b_06_07*cos(theta_06 - theta_07) - g_06_07*sin(theta_06 - theta_07)) + V_06*V_09*(b_06_09*cos(theta_06 - theta_09) - g_06_09*sin(theta_06 - theta_09))
        struct[0].g[12,0] = -P_07/S_base + V_06*V_07*(b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07)) + V_07**2*(g_06_07 + g_07_08) + V_07*V_08*(-b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08))
        struct[0].g[13,0] = -Q_07/S_base + V_06*V_07*(b_06_07*cos(theta_06 - theta_07) + g_06_07*sin(theta_06 - theta_07)) + V_07**2*(-b_06_07 - b_07_08 - bs_06_07/2 - bs_07_08/2) + V_07*V_08*(b_07_08*cos(theta_07 - theta_08) - g_07_08*sin(theta_07 - theta_08))
        struct[0].g[14,0] = -P_08/S_base + V_07*V_08*(b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08)) + V_08**2*g_07_08 - S_n_08*p_s_08/S_base
        struct[0].g[15,0] = -Q_08/S_base + V_07*V_08*(b_07_08*cos(theta_07 - theta_08) + g_07_08*sin(theta_07 - theta_08)) + V_08**2*(-b_07_08 - bs_07_08/2) - S_n_08*q_s_08/S_base
        struct[0].g[16,0] = -P_09/S_base + V_06*V_09*(b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09)) + V_09**2*g_06_09 - S_n_09*p_s_09/S_base
        struct[0].g[17,0] = -Q_09/S_base + V_06*V_09*(b_06_09*cos(theta_06 - theta_09) + g_06_09*sin(theta_06 - theta_09)) + V_09**2*(-b_06_09 - bs_06_09/2) - S_n_09*q_s_09/S_base
        struct[0].g[18,0] = -P_10/S_base + V_05*V_10*(b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10)) + V_10**2*g_05_10 - S_n_10*p_s_10/S_base
        struct[0].g[19,0] = -Q_10/S_base + V_05*V_10*(b_05_10*cos(theta_05 - theta_10) + g_05_10*sin(theta_05 - theta_10)) + V_10**2*(-b_05_10 - bs_05_10/2) - S_n_10*q_s_10/S_base
        struct[0].g[20,0] = -P_11/S_base + V_04*V_11*(b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11)) + V_11**2*g_04_11
        struct[0].g[21,0] = -Q_11/S_base + V_04*V_11*(b_04_11*cos(theta_04 - theta_11) + g_04_11*sin(theta_04 - theta_11)) + V_11**2*(-b_04_11 - bs_04_11/2)
        struct[0].g[22,0] = -P_12/S_base + V_03*V_12*(b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12)) + V_12**2*g_03_12
        struct[0].g[23,0] = -Q_12/S_base + V_03*V_12*(b_03_12*cos(theta_03 - theta_12) + g_03_12*sin(theta_03 - theta_12)) + V_12**2*(-b_03_12 - bs_03_12/2)
        struct[0].g[24,0] = K_p_08*(p_m_08 - p_s_08 + xi_p_08/T_p_08) - omega_08
        struct[0].g[25,0] = K_q_08*(-q_s_08 + q_s_ref_08 + xi_q_08/T_q_08) - e_qv_08
        struct[0].g[26,0] = -R_v_08*i_d_08 - V_08*sin(delta_08 - theta_08) + X_v_08*i_q_08
        struct[0].g[27,0] = -R_v_08*i_q_08 - V_08*cos(delta_08 - theta_08) - X_v_08*i_d_08 + e_qv_08
        struct[0].g[28,0] = V_08*i_d_08*sin(delta_08 - theta_08) + V_08*i_q_08*cos(delta_08 - theta_08) - p_s_08
        struct[0].g[29,0] = V_08*i_d_08*cos(delta_08 - theta_08) - V_08*i_q_08*sin(delta_08 - theta_08) - q_s_08
        struct[0].g[30,0] = K_fpfr_08*Piecewise(np.array([(P_f_min_08, P_f_min_08 > p_f_08), (P_f_max_08, P_f_max_08 < p_f_08), (p_f_08, True)])) + p_ghr_08 - p_m_08 + p_s_08 - p_t_08 + p_u_08
        struct[0].g[31,0] = i_d_08*(R_s_08*i_d_08 + V_08*sin(delta_08 - theta_08)) + i_q_08*(R_s_08*i_q_08 + V_08*cos(delta_08 - theta_08)) - p_t_08
        struct[0].g[32,0] = -p_u_08 - k_u_08*(-v_u_08**2 + v_u_ref_08**2)/V_u_max_08**2
        struct[0].g[33,0] = R_uc_08*S_n_08*(p_gou_08 - p_t_08)/(v_u_08 + 0.1) + e_u_08 - v_u_08
        struct[0].g[34,0] = -k_u_08 + Piecewise(np.array([(K_u_max_08, V_u_min_08 > v_u_08), (K_u_0_08 + (-K_u_0_08 + K_u_max_08)*(-V_u_lt_08 + v_u_08)/(-V_u_lt_08 + V_u_min_08), V_u_lt_08 > v_u_08), (K_u_0_08 + (-K_u_0_08 + K_u_max_08)*(-V_u_ht_08 + v_u_08)/(-V_u_ht_08 + V_u_max_08), V_u_ht_08 < v_u_08), (K_u_max_08, V_u_max_08 < v_u_08), (K_u_0_08, True)]))
        struct[0].g[35,0] = -k_cur_sat_08 + Piecewise(np.array([(0.0001, k_cur_08 < 0.0001), (1, k_cur_08 > 1), (k_cur_08, True)]))
        struct[0].g[36,0] = k_cur_sat_08*(inc_p_gin_08 + p_gin_0_08) - p_gou_08
        struct[0].g[37,0] = -p_f_08 - Piecewise(np.array([((0.5*DB_08 + omega_08 - omega_ref_08)/Droop_08, omega_08 < -0.5*DB_08 + omega_ref_08), ((-0.5*DB_08 + omega_08 - omega_ref_08)/Droop_08, omega_08 > 0.5*DB_08 + omega_ref_08), (0.0, True)]))
        struct[0].g[38,0] = -r_lim_08 + Piecewise(np.array([(R_lim_max_08, (omega_08 > 0.5*DB_08 + omega_ref_08) | (omega_08 < -0.5*DB_08 + omega_ref_08)), (0.0, True)])) + Piecewise(np.array([(R_lim_08 + (-R_lim_08 + R_lim_max_08)*(-V_u_lt_08 + v_u_08)/(-V_u_lt_08 + V_u_min_08), V_u_lt_08 > v_u_08), (R_lim_08 + (-R_lim_08 + R_lim_max_08)*(-V_u_ht_08 + v_u_08)/(-V_u_ht_08 + V_u_max_08), V_u_ht_08 < v_u_08), (R_lim_08, True)]))
        struct[0].g[39,0] = K_p_09*(p_m_09 - p_s_09 + xi_p_09/T_p_09) - omega_09
        struct[0].g[40,0] = K_q_09*(-q_s_09 + q_s_ref_09 + xi_q_09/T_q_09) - e_qv_09
        struct[0].g[41,0] = -R_v_09*i_d_09 - V_09*sin(delta_09 - theta_09) + X_v_09*i_q_09
        struct[0].g[42,0] = -R_v_09*i_q_09 - V_09*cos(delta_09 - theta_09) - X_v_09*i_d_09 + e_qv_09
        struct[0].g[43,0] = V_09*i_d_09*sin(delta_09 - theta_09) + V_09*i_q_09*cos(delta_09 - theta_09) - p_s_09
        struct[0].g[44,0] = V_09*i_d_09*cos(delta_09 - theta_09) - V_09*i_q_09*sin(delta_09 - theta_09) - q_s_09
        struct[0].g[45,0] = K_fpfr_09*Piecewise(np.array([(P_f_min_09, P_f_min_09 > p_f_09), (P_f_max_09, P_f_max_09 < p_f_09), (p_f_09, True)])) + p_ghr_09 - p_m_09 + p_s_09 - p_t_09 + p_u_09
        struct[0].g[46,0] = i_d_09*(R_s_09*i_d_09 + V_09*sin(delta_09 - theta_09)) + i_q_09*(R_s_09*i_q_09 + V_09*cos(delta_09 - theta_09)) - p_t_09
        struct[0].g[47,0] = -p_u_09 - k_u_09*(-v_u_09**2 + v_u_ref_09**2)/V_u_max_09**2
        struct[0].g[48,0] = R_uc_09*S_n_09*(p_gou_09 - p_t_09)/(v_u_09 + 0.1) + e_u_09 - v_u_09
        struct[0].g[49,0] = -k_u_09 + Piecewise(np.array([(K_u_max_09, V_u_min_09 > v_u_09), (K_u_0_09 + (-K_u_0_09 + K_u_max_09)*(-V_u_lt_09 + v_u_09)/(-V_u_lt_09 + V_u_min_09), V_u_lt_09 > v_u_09), (K_u_0_09 + (-K_u_0_09 + K_u_max_09)*(-V_u_ht_09 + v_u_09)/(-V_u_ht_09 + V_u_max_09), V_u_ht_09 < v_u_09), (K_u_max_09, V_u_max_09 < v_u_09), (K_u_0_09, True)]))
        struct[0].g[50,0] = -k_cur_sat_09 + Piecewise(np.array([(0.0001, k_cur_09 < 0.0001), (1, k_cur_09 > 1), (k_cur_09, True)]))
        struct[0].g[51,0] = k_cur_sat_09*(inc_p_gin_09 + p_gin_0_09) - p_gou_09
        struct[0].g[52,0] = -p_f_09 - Piecewise(np.array([((0.5*DB_09 + omega_09 - omega_ref_09)/Droop_09, omega_09 < -0.5*DB_09 + omega_ref_09), ((-0.5*DB_09 + omega_09 - omega_ref_09)/Droop_09, omega_09 > 0.5*DB_09 + omega_ref_09), (0.0, True)]))
        struct[0].g[53,0] = -r_lim_09 + Piecewise(np.array([(R_lim_max_09, (omega_09 > 0.5*DB_09 + omega_ref_09) | (omega_09 < -0.5*DB_09 + omega_ref_09)), (0.0, True)])) + Piecewise(np.array([(R_lim_09 + (-R_lim_09 + R_lim_max_09)*(-V_u_lt_09 + v_u_09)/(-V_u_lt_09 + V_u_min_09), V_u_lt_09 > v_u_09), (R_lim_09 + (-R_lim_09 + R_lim_max_09)*(-V_u_ht_09 + v_u_09)/(-V_u_ht_09 + V_u_max_09), V_u_ht_09 < v_u_09), (R_lim_09, True)]))
        struct[0].g[54,0] = K_p_10*(p_m_10 - p_s_10 + xi_p_10/T_p_10) - omega_10
        struct[0].g[55,0] = K_q_10*(-q_s_10 + q_s_ref_10 + xi_q_10/T_q_10) - e_qv_10
        struct[0].g[56,0] = -R_v_10*i_d_10 - V_10*sin(delta_10 - theta_10) + X_v_10*i_q_10
        struct[0].g[57,0] = -R_v_10*i_q_10 - V_10*cos(delta_10 - theta_10) - X_v_10*i_d_10 + e_qv_10
        struct[0].g[58,0] = V_10*i_d_10*sin(delta_10 - theta_10) + V_10*i_q_10*cos(delta_10 - theta_10) - p_s_10
        struct[0].g[59,0] = V_10*i_d_10*cos(delta_10 - theta_10) - V_10*i_q_10*sin(delta_10 - theta_10) - q_s_10
        struct[0].g[60,0] = K_fpfr_10*Piecewise(np.array([(P_f_min_10, P_f_min_10 > p_f_10), (P_f_max_10, P_f_max_10 < p_f_10), (p_f_10, True)])) + p_ghr_10 - p_m_10 + p_s_10 - p_t_10 + p_u_10
        struct[0].g[61,0] = i_d_10*(R_s_10*i_d_10 + V_10*sin(delta_10 - theta_10)) + i_q_10*(R_s_10*i_q_10 + V_10*cos(delta_10 - theta_10)) - p_t_10
        struct[0].g[62,0] = -p_u_10 - k_u_10*(-v_u_10**2 + v_u_ref_10**2)/V_u_max_10**2
        struct[0].g[63,0] = R_uc_10*S_n_10*(p_gou_10 - p_t_10)/(v_u_10 + 0.1) + e_u_10 - v_u_10
        struct[0].g[64,0] = -k_u_10 + Piecewise(np.array([(K_u_max_10, V_u_min_10 > v_u_10), (K_u_0_10 + (-K_u_0_10 + K_u_max_10)*(-V_u_lt_10 + v_u_10)/(-V_u_lt_10 + V_u_min_10), V_u_lt_10 > v_u_10), (K_u_0_10 + (-K_u_0_10 + K_u_max_10)*(-V_u_ht_10 + v_u_10)/(-V_u_ht_10 + V_u_max_10), V_u_ht_10 < v_u_10), (K_u_max_10, V_u_max_10 < v_u_10), (K_u_0_10, True)]))
        struct[0].g[65,0] = -k_cur_sat_10 + Piecewise(np.array([(0.0001, k_cur_10 < 0.0001), (1, k_cur_10 > 1), (k_cur_10, True)]))
        struct[0].g[66,0] = k_cur_sat_10*(inc_p_gin_10 + p_gin_0_10) - p_gou_10
        struct[0].g[67,0] = -p_f_10 - Piecewise(np.array([((0.5*DB_10 + omega_10 - omega_ref_10)/Droop_10, omega_10 < -0.5*DB_10 + omega_ref_10), ((-0.5*DB_10 + omega_10 - omega_ref_10)/Droop_10, omega_10 > 0.5*DB_10 + omega_ref_10), (0.0, True)]))
        struct[0].g[68,0] = -r_lim_10 + Piecewise(np.array([(R_lim_max_10, (omega_10 > 0.5*DB_10 + omega_ref_10) | (omega_10 < -0.5*DB_10 + omega_ref_10)), (0.0, True)])) + Piecewise(np.array([(R_lim_10 + (-R_lim_10 + R_lim_max_10)*(-V_u_lt_10 + v_u_10)/(-V_u_lt_10 + V_u_min_10), V_u_lt_10 > v_u_10), (R_lim_10 + (-R_lim_10 + R_lim_max_10)*(-V_u_ht_10 + v_u_10)/(-V_u_ht_10 + V_u_max_10), V_u_ht_10 < v_u_10), (R_lim_10, True)]))
        struct[0].g[69,0] = Domega_01 - omega_01 + omega_ref_01
        struct[0].g[70,0] = -R_v_01*i_d_01 - V_01*sin(delta_01 - theta_01) + X_v_01*i_q_01
        struct[0].g[71,0] = -R_v_01*i_q_01 - V_01*cos(delta_01 - theta_01) - X_v_01*i_d_01 + e_qv_01
        struct[0].g[72,0] = V_01*i_d_01*sin(delta_01 - theta_01) + V_01*i_q_01*cos(delta_01 - theta_01) - p_s_01
        struct[0].g[73,0] = V_01*i_d_01*cos(delta_01 - theta_01) - V_01*i_q_01*sin(delta_01 - theta_01) - q_s_01
        struct[0].g[74,0] = -omega_coi + (1000000.0*S_n_01*omega_01 + S_n_10*T_p_10*omega_10/(2*K_p_10) + S_n_09*T_p_09*omega_09/(2*K_p_09) + S_n_08*T_p_08*omega_08/(2*K_p_08))/(1000000.0*S_n_01 + S_n_10*T_p_10/(2*K_p_10) + S_n_09*T_p_09/(2*K_p_09) + S_n_08*T_p_08/(2*K_p_08))
        struct[0].g[75,0] = K_i_agc*xi_freq + K_p_agc*(1 - omega_coi) - p_agc
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_01
        struct[0].h[1,0] = V_02
        struct[0].h[2,0] = V_03
        struct[0].h[3,0] = V_04
        struct[0].h[4,0] = V_05
        struct[0].h[5,0] = V_06
        struct[0].h[6,0] = V_07
        struct[0].h[7,0] = V_08
        struct[0].h[8,0] = V_09
        struct[0].h[9,0] = V_10
        struct[0].h[10,0] = V_11
        struct[0].h[11,0] = V_12
        struct[0].h[12,0] = inc_p_gin_08 + p_gin_0_08
        struct[0].h[13,0] = p_g_ref_08
        struct[0].h[14,0] = -p_s_08 + p_t_08
        struct[0].h[15,0] = (-V_u_min_08**2 + e_u_08**2)/(V_u_max_08**2 - V_u_min_08**2)
        struct[0].h[16,0] = K_fpfr_08*Piecewise(np.array([(P_f_min_08, P_f_min_08 > p_f_08), (P_f_max_08, P_f_max_08 < p_f_08), (p_f_08, True)]))
        struct[0].h[17,0] = Piecewise(np.array([(P_f_min_08, P_f_min_08 > p_f_08), (P_f_max_08, P_f_max_08 < p_f_08), (p_f_08, True)]))
        struct[0].h[18,0] = inc_p_gin_09 + p_gin_0_09
        struct[0].h[19,0] = p_g_ref_09
        struct[0].h[20,0] = -p_s_09 + p_t_09
        struct[0].h[21,0] = (-V_u_min_09**2 + e_u_09**2)/(V_u_max_09**2 - V_u_min_09**2)
        struct[0].h[22,0] = K_fpfr_09*Piecewise(np.array([(P_f_min_09, P_f_min_09 > p_f_09), (P_f_max_09, P_f_max_09 < p_f_09), (p_f_09, True)]))
        struct[0].h[23,0] = Piecewise(np.array([(P_f_min_09, P_f_min_09 > p_f_09), (P_f_max_09, P_f_max_09 < p_f_09), (p_f_09, True)]))
        struct[0].h[24,0] = inc_p_gin_10 + p_gin_0_10
        struct[0].h[25,0] = p_g_ref_10
        struct[0].h[26,0] = -p_s_10 + p_t_10
        struct[0].h[27,0] = (-V_u_min_10**2 + e_u_10**2)/(V_u_max_10**2 - V_u_min_10**2)
        struct[0].h[28,0] = K_fpfr_10*Piecewise(np.array([(P_f_min_10, P_f_min_10 > p_f_10), (P_f_max_10, P_f_max_10 < p_f_10), (p_f_10, True)]))
        struct[0].h[29,0] = Piecewise(np.array([(P_f_min_10, P_f_min_10 > p_f_10), (P_f_max_10, P_f_max_10 < p_f_10), (p_f_10, True)]))
        struct[0].h[30,0] = alpha_01
    

    if mode == 10:

        struct[0].Fx[4,4] = Piecewise(np.array([(0, (r_lim_08 < K_h_08*(-p_ghr_08 + p_gou_08)) | (r_lim_08 < -K_h_08*(-p_ghr_08 + p_gou_08))), (-K_h_08, True)]))
        struct[0].Fx[5,5] = -1/T_cur_08
        struct[0].Fx[5,6] = (-p_g_ref_08/(inc_p_gin_08 + p_gin_0_08)**2 - Piecewise(np.array([(P_f_min_08, P_f_min_08 > p_f_08), (P_f_max_08, P_f_max_08 < p_f_08), (p_f_08, True)]))/(inc_p_gin_08 + p_gin_0_08)**2)/T_cur_08
        struct[0].Fx[11,11] = Piecewise(np.array([(0, (r_lim_09 < K_h_09*(-p_ghr_09 + p_gou_09)) | (r_lim_09 < -K_h_09*(-p_ghr_09 + p_gou_09))), (-K_h_09, True)]))
        struct[0].Fx[12,12] = -1/T_cur_09
        struct[0].Fx[12,13] = (-p_g_ref_09/(inc_p_gin_09 + p_gin_0_09)**2 - Piecewise(np.array([(P_f_min_09, P_f_min_09 > p_f_09), (P_f_max_09, P_f_max_09 < p_f_09), (p_f_09, True)]))/(inc_p_gin_09 + p_gin_0_09)**2)/T_cur_09
        struct[0].Fx[18,18] = Piecewise(np.array([(0, (r_lim_10 < K_h_10*(-p_ghr_10 + p_gou_10)) | (r_lim_10 < -K_h_10*(-p_ghr_10 + p_gou_10))), (-K_h_10, True)]))
        struct[0].Fx[19,19] = -1/T_cur_10
        struct[0].Fx[19,20] = (-p_g_ref_10/(inc_p_gin_10 + p_gin_0_10)**2 - Piecewise(np.array([(P_f_min_10, P_f_min_10 > p_f_10), (P_f_max_10, P_f_max_10 < p_f_10), (p_f_10, True)]))/(inc_p_gin_10 + p_gin_0_10)**2)/T_cur_10
        struct[0].Fx[21,21] = -K_delta_01
        struct[0].Fx[22,22] = -K_alpha_01

    if mode == 11:

        struct[0].Fy[0,24] = Omega_b_08
        struct[0].Fy[0,74] = -Omega_b_08
        struct[0].Fy[1,28] = -1
        struct[0].Fy[1,30] = 1
        struct[0].Fy[2,29] = -1
        struct[0].Fy[3,31] = -S_n_08/(C_u_08*(v_u_08 + 0.1))
        struct[0].Fy[3,33] = -S_n_08*(p_gou_08 - p_t_08)/(C_u_08*(v_u_08 + 0.1)**2)
        struct[0].Fy[3,36] = S_n_08/(C_u_08*(v_u_08 + 0.1))
        struct[0].Fy[4,36] = Piecewise(np.array([(0, (r_lim_08 < K_h_08*(-p_ghr_08 + p_gou_08)) | (r_lim_08 < -K_h_08*(-p_ghr_08 + p_gou_08))), (K_h_08, True)]))
        struct[0].Fy[4,38] = Piecewise(np.array([(-1, r_lim_08 < -K_h_08*(-p_ghr_08 + p_gou_08)), (1, r_lim_08 < K_h_08*(-p_ghr_08 + p_gou_08)), (0, True)]))
        struct[0].Fy[5,37] = Piecewise(np.array([(0, (P_f_min_08 > p_f_08) | (P_f_max_08 < p_f_08)), (1, True)]))/(T_cur_08*(inc_p_gin_08 + p_gin_0_08))
        struct[0].Fy[7,39] = Omega_b_09
        struct[0].Fy[7,74] = -Omega_b_09
        struct[0].Fy[8,43] = -1
        struct[0].Fy[8,45] = 1
        struct[0].Fy[9,44] = -1
        struct[0].Fy[10,46] = -S_n_09/(C_u_09*(v_u_09 + 0.1))
        struct[0].Fy[10,48] = -S_n_09*(p_gou_09 - p_t_09)/(C_u_09*(v_u_09 + 0.1)**2)
        struct[0].Fy[10,51] = S_n_09/(C_u_09*(v_u_09 + 0.1))
        struct[0].Fy[11,51] = Piecewise(np.array([(0, (r_lim_09 < K_h_09*(-p_ghr_09 + p_gou_09)) | (r_lim_09 < -K_h_09*(-p_ghr_09 + p_gou_09))), (K_h_09, True)]))
        struct[0].Fy[11,53] = Piecewise(np.array([(-1, r_lim_09 < -K_h_09*(-p_ghr_09 + p_gou_09)), (1, r_lim_09 < K_h_09*(-p_ghr_09 + p_gou_09)), (0, True)]))
        struct[0].Fy[12,52] = Piecewise(np.array([(0, (P_f_min_09 > p_f_09) | (P_f_max_09 < p_f_09)), (1, True)]))/(T_cur_09*(inc_p_gin_09 + p_gin_0_09))
        struct[0].Fy[14,54] = Omega_b_10
        struct[0].Fy[14,74] = -Omega_b_10
        struct[0].Fy[15,58] = -1
        struct[0].Fy[15,60] = 1
        struct[0].Fy[16,59] = -1
        struct[0].Fy[17,61] = -S_n_10/(C_u_10*(v_u_10 + 0.1))
        struct[0].Fy[17,63] = -S_n_10*(p_gou_10 - p_t_10)/(C_u_10*(v_u_10 + 0.1)**2)
        struct[0].Fy[17,66] = S_n_10/(C_u_10*(v_u_10 + 0.1))
        struct[0].Fy[18,66] = Piecewise(np.array([(0, (r_lim_10 < K_h_10*(-p_ghr_10 + p_gou_10)) | (r_lim_10 < -K_h_10*(-p_ghr_10 + p_gou_10))), (K_h_10, True)]))
        struct[0].Fy[18,68] = Piecewise(np.array([(-1, r_lim_10 < -K_h_10*(-p_ghr_10 + p_gou_10)), (1, r_lim_10 < K_h_10*(-p_ghr_10 + p_gou_10)), (0, True)]))
        struct[0].Fy[19,67] = Piecewise(np.array([(0, (P_f_min_10 > p_f_10) | (P_f_max_10 < p_f_10)), (1, True)]))/(T_cur_10*(inc_p_gin_10 + p_gin_0_10))
        struct[0].Fy[21,69] = Omega_b_01
        struct[0].Fy[21,74] = -Omega_b_01
        struct[0].Fy[23,74] = -1

        struct[0].Gx[24,1] = K_p_08/T_p_08
        struct[0].Gx[25,2] = K_q_08/T_q_08
        struct[0].Gx[26,0] = -V_08*cos(delta_08 - theta_08)
        struct[0].Gx[27,0] = V_08*sin(delta_08 - theta_08)
        struct[0].Gx[28,0] = V_08*i_d_08*cos(delta_08 - theta_08) - V_08*i_q_08*sin(delta_08 - theta_08)
        struct[0].Gx[29,0] = -V_08*i_d_08*sin(delta_08 - theta_08) - V_08*i_q_08*cos(delta_08 - theta_08)
        struct[0].Gx[30,4] = 1
        struct[0].Gx[31,0] = V_08*i_d_08*cos(delta_08 - theta_08) - V_08*i_q_08*sin(delta_08 - theta_08)
        struct[0].Gx[33,3] = 1
        struct[0].Gx[35,5] = Piecewise(np.array([(0, (k_cur_08 > 1) | (k_cur_08 < 0.0001)), (1, True)]))
        struct[0].Gx[36,6] = k_cur_sat_08
        struct[0].Gx[39,8] = K_p_09/T_p_09
        struct[0].Gx[40,9] = K_q_09/T_q_09
        struct[0].Gx[41,7] = -V_09*cos(delta_09 - theta_09)
        struct[0].Gx[42,7] = V_09*sin(delta_09 - theta_09)
        struct[0].Gx[43,7] = V_09*i_d_09*cos(delta_09 - theta_09) - V_09*i_q_09*sin(delta_09 - theta_09)
        struct[0].Gx[44,7] = -V_09*i_d_09*sin(delta_09 - theta_09) - V_09*i_q_09*cos(delta_09 - theta_09)
        struct[0].Gx[45,11] = 1
        struct[0].Gx[46,7] = V_09*i_d_09*cos(delta_09 - theta_09) - V_09*i_q_09*sin(delta_09 - theta_09)
        struct[0].Gx[48,10] = 1
        struct[0].Gx[50,12] = Piecewise(np.array([(0, (k_cur_09 > 1) | (k_cur_09 < 0.0001)), (1, True)]))
        struct[0].Gx[51,13] = k_cur_sat_09
        struct[0].Gx[54,15] = K_p_10/T_p_10
        struct[0].Gx[55,16] = K_q_10/T_q_10
        struct[0].Gx[56,14] = -V_10*cos(delta_10 - theta_10)
        struct[0].Gx[57,14] = V_10*sin(delta_10 - theta_10)
        struct[0].Gx[58,14] = V_10*i_d_10*cos(delta_10 - theta_10) - V_10*i_q_10*sin(delta_10 - theta_10)
        struct[0].Gx[59,14] = -V_10*i_d_10*sin(delta_10 - theta_10) - V_10*i_q_10*cos(delta_10 - theta_10)
        struct[0].Gx[60,18] = 1
        struct[0].Gx[61,14] = V_10*i_d_10*cos(delta_10 - theta_10) - V_10*i_q_10*sin(delta_10 - theta_10)
        struct[0].Gx[63,17] = 1
        struct[0].Gx[65,19] = Piecewise(np.array([(0, (k_cur_10 > 1) | (k_cur_10 < 0.0001)), (1, True)]))
        struct[0].Gx[66,20] = k_cur_sat_10
        struct[0].Gx[69,22] = 1
        struct[0].Gx[70,21] = -V_01*cos(delta_01 - theta_01)
        struct[0].Gx[71,21] = V_01*sin(delta_01 - theta_01)
        struct[0].Gx[72,21] = V_01*i_d_01*cos(delta_01 - theta_01) - V_01*i_q_01*sin(delta_01 - theta_01)
        struct[0].Gx[73,21] = -V_01*i_d_01*sin(delta_01 - theta_01) - V_01*i_q_01*cos(delta_01 - theta_01)
        struct[0].Gx[75,23] = K_i_agc

        struct[0].Gy[0,0] = 2*V_01*g_01_02 + V_02*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy[0,1] = V_01*V_02*(-b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy[0,2] = V_01*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy[0,3] = V_01*V_02*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy[0,72] = -S_n_01/S_base
        struct[0].Gy[1,0] = 2*V_01*(-b_01_02 - bs_01_02/2) + V_02*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy[1,1] = V_01*V_02*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy[1,2] = V_01*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy[1,3] = V_01*V_02*(b_01_02*sin(theta_01 - theta_02) + g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy[1,73] = -S_n_01/S_base
        struct[0].Gy[2,0] = V_02*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy[2,1] = V_01*V_02*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy[2,2] = V_01*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) + 2*V_02*(g_01_02 + g_02_03) + V_03*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy[2,3] = V_01*V_02*(-b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02)) + V_02*V_03*(-b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy[2,4] = V_02*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy[2,5] = V_02*V_03*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy[3,0] = V_02*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy[3,1] = V_01*V_02*(-b_01_02*sin(theta_01 - theta_02) + g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy[3,2] = V_01*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02)) + 2*V_02*(-b_01_02 - b_02_03 - bs_01_02/2 - bs_02_03/2) + V_03*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy[3,3] = V_01*V_02*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) + V_02*V_03*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy[3,4] = V_02*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy[3,5] = V_02*V_03*(b_02_03*sin(theta_02 - theta_03) + g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy[4,2] = V_03*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy[4,3] = V_02*V_03*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy[4,4] = V_02*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03)) + 2*V_03*(g_02_03 + g_03_04 + g_03_12) + V_04*(-b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04)) + V_12*(-b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12))
        struct[0].Gy[4,5] = V_02*V_03*(-b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03)) + V_03*V_04*(-b_03_04*cos(theta_03 - theta_04) + g_03_04*sin(theta_03 - theta_04)) + V_03*V_12*(-b_03_12*cos(theta_03 - theta_12) + g_03_12*sin(theta_03 - theta_12))
        struct[0].Gy[4,6] = V_03*(-b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04))
        struct[0].Gy[4,7] = V_03*V_04*(b_03_04*cos(theta_03 - theta_04) - g_03_04*sin(theta_03 - theta_04))
        struct[0].Gy[4,22] = V_03*(-b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12))
        struct[0].Gy[4,23] = V_03*V_12*(b_03_12*cos(theta_03 - theta_12) - g_03_12*sin(theta_03 - theta_12))
        struct[0].Gy[5,2] = V_03*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy[5,3] = V_02*V_03*(-b_02_03*sin(theta_02 - theta_03) + g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy[5,4] = V_02*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03)) + 2*V_03*(-b_02_03 - b_03_04 - b_03_12 - bs_02_03/2 - bs_03_04/2 - bs_03_12/2) + V_04*(b_03_04*cos(theta_03 - theta_04) - g_03_04*sin(theta_03 - theta_04)) + V_12*(b_03_12*cos(theta_03 - theta_12) - g_03_12*sin(theta_03 - theta_12))
        struct[0].Gy[5,5] = V_02*V_03*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03)) + V_03*V_04*(-b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04)) + V_03*V_12*(-b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12))
        struct[0].Gy[5,6] = V_03*(b_03_04*cos(theta_03 - theta_04) - g_03_04*sin(theta_03 - theta_04))
        struct[0].Gy[5,7] = V_03*V_04*(b_03_04*sin(theta_03 - theta_04) + g_03_04*cos(theta_03 - theta_04))
        struct[0].Gy[5,22] = V_03*(b_03_12*cos(theta_03 - theta_12) - g_03_12*sin(theta_03 - theta_12))
        struct[0].Gy[5,23] = V_03*V_12*(b_03_12*sin(theta_03 - theta_12) + g_03_12*cos(theta_03 - theta_12))
        struct[0].Gy[6,4] = V_04*(b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04))
        struct[0].Gy[6,5] = V_03*V_04*(b_03_04*cos(theta_03 - theta_04) + g_03_04*sin(theta_03 - theta_04))
        struct[0].Gy[6,6] = V_03*(b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04)) + 2*V_04*(g_03_04 + g_04_05 + g_04_11) + V_05*(-b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05)) + V_11*(-b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11))
        struct[0].Gy[6,7] = V_03*V_04*(-b_03_04*cos(theta_03 - theta_04) - g_03_04*sin(theta_03 - theta_04)) + V_04*V_05*(-b_04_05*cos(theta_04 - theta_05) + g_04_05*sin(theta_04 - theta_05)) + V_04*V_11*(-b_04_11*cos(theta_04 - theta_11) + g_04_11*sin(theta_04 - theta_11))
        struct[0].Gy[6,8] = V_04*(-b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05))
        struct[0].Gy[6,9] = V_04*V_05*(b_04_05*cos(theta_04 - theta_05) - g_04_05*sin(theta_04 - theta_05))
        struct[0].Gy[6,20] = V_04*(-b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11))
        struct[0].Gy[6,21] = V_04*V_11*(b_04_11*cos(theta_04 - theta_11) - g_04_11*sin(theta_04 - theta_11))
        struct[0].Gy[7,4] = V_04*(b_03_04*cos(theta_03 - theta_04) + g_03_04*sin(theta_03 - theta_04))
        struct[0].Gy[7,5] = V_03*V_04*(-b_03_04*sin(theta_03 - theta_04) + g_03_04*cos(theta_03 - theta_04))
        struct[0].Gy[7,6] = V_03*(b_03_04*cos(theta_03 - theta_04) + g_03_04*sin(theta_03 - theta_04)) + 2*V_04*(-b_03_04 - b_04_05 - b_04_11 - bs_03_04/2 - bs_04_05/2 - bs_04_11/2) + V_05*(b_04_05*cos(theta_04 - theta_05) - g_04_05*sin(theta_04 - theta_05)) + V_11*(b_04_11*cos(theta_04 - theta_11) - g_04_11*sin(theta_04 - theta_11))
        struct[0].Gy[7,7] = V_03*V_04*(b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04)) + V_04*V_05*(-b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05)) + V_04*V_11*(-b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11))
        struct[0].Gy[7,8] = V_04*(b_04_05*cos(theta_04 - theta_05) - g_04_05*sin(theta_04 - theta_05))
        struct[0].Gy[7,9] = V_04*V_05*(b_04_05*sin(theta_04 - theta_05) + g_04_05*cos(theta_04 - theta_05))
        struct[0].Gy[7,20] = V_04*(b_04_11*cos(theta_04 - theta_11) - g_04_11*sin(theta_04 - theta_11))
        struct[0].Gy[7,21] = V_04*V_11*(b_04_11*sin(theta_04 - theta_11) + g_04_11*cos(theta_04 - theta_11))
        struct[0].Gy[8,6] = V_05*(b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05))
        struct[0].Gy[8,7] = V_04*V_05*(b_04_05*cos(theta_04 - theta_05) + g_04_05*sin(theta_04 - theta_05))
        struct[0].Gy[8,8] = V_04*(b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05)) + 2*V_05*(g_04_05 + g_05_06 + g_05_10) + V_06*(-b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06)) + V_10*(-b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10))
        struct[0].Gy[8,9] = V_04*V_05*(-b_04_05*cos(theta_04 - theta_05) - g_04_05*sin(theta_04 - theta_05)) + V_05*V_06*(-b_05_06*cos(theta_05 - theta_06) + g_05_06*sin(theta_05 - theta_06)) + V_05*V_10*(-b_05_10*cos(theta_05 - theta_10) + g_05_10*sin(theta_05 - theta_10))
        struct[0].Gy[8,10] = V_05*(-b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06))
        struct[0].Gy[8,11] = V_05*V_06*(b_05_06*cos(theta_05 - theta_06) - g_05_06*sin(theta_05 - theta_06))
        struct[0].Gy[8,18] = V_05*(-b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10))
        struct[0].Gy[8,19] = V_05*V_10*(b_05_10*cos(theta_05 - theta_10) - g_05_10*sin(theta_05 - theta_10))
        struct[0].Gy[9,6] = V_05*(b_04_05*cos(theta_04 - theta_05) + g_04_05*sin(theta_04 - theta_05))
        struct[0].Gy[9,7] = V_04*V_05*(-b_04_05*sin(theta_04 - theta_05) + g_04_05*cos(theta_04 - theta_05))
        struct[0].Gy[9,8] = V_04*(b_04_05*cos(theta_04 - theta_05) + g_04_05*sin(theta_04 - theta_05)) + 2*V_05*(-b_04_05 - b_05_06 - b_05_10 - bs_04_05/2 - bs_05_06/2 - bs_05_10/2) + V_06*(b_05_06*cos(theta_05 - theta_06) - g_05_06*sin(theta_05 - theta_06)) + V_10*(b_05_10*cos(theta_05 - theta_10) - g_05_10*sin(theta_05 - theta_10))
        struct[0].Gy[9,9] = V_04*V_05*(b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05)) + V_05*V_06*(-b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06)) + V_05*V_10*(-b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10))
        struct[0].Gy[9,10] = V_05*(b_05_06*cos(theta_05 - theta_06) - g_05_06*sin(theta_05 - theta_06))
        struct[0].Gy[9,11] = V_05*V_06*(b_05_06*sin(theta_05 - theta_06) + g_05_06*cos(theta_05 - theta_06))
        struct[0].Gy[9,18] = V_05*(b_05_10*cos(theta_05 - theta_10) - g_05_10*sin(theta_05 - theta_10))
        struct[0].Gy[9,19] = V_05*V_10*(b_05_10*sin(theta_05 - theta_10) + g_05_10*cos(theta_05 - theta_10))
        struct[0].Gy[10,8] = V_06*(b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06))
        struct[0].Gy[10,9] = V_05*V_06*(b_05_06*cos(theta_05 - theta_06) + g_05_06*sin(theta_05 - theta_06))
        struct[0].Gy[10,10] = V_05*(b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06)) + 2*V_06*(g_05_06 + g_06_07 + g_06_09) + V_07*(-b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07)) + V_09*(-b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09))
        struct[0].Gy[10,11] = V_05*V_06*(-b_05_06*cos(theta_05 - theta_06) - g_05_06*sin(theta_05 - theta_06)) + V_06*V_07*(-b_06_07*cos(theta_06 - theta_07) + g_06_07*sin(theta_06 - theta_07)) + V_06*V_09*(-b_06_09*cos(theta_06 - theta_09) + g_06_09*sin(theta_06 - theta_09))
        struct[0].Gy[10,12] = V_06*(-b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07))
        struct[0].Gy[10,13] = V_06*V_07*(b_06_07*cos(theta_06 - theta_07) - g_06_07*sin(theta_06 - theta_07))
        struct[0].Gy[10,16] = V_06*(-b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09))
        struct[0].Gy[10,17] = V_06*V_09*(b_06_09*cos(theta_06 - theta_09) - g_06_09*sin(theta_06 - theta_09))
        struct[0].Gy[11,8] = V_06*(b_05_06*cos(theta_05 - theta_06) + g_05_06*sin(theta_05 - theta_06))
        struct[0].Gy[11,9] = V_05*V_06*(-b_05_06*sin(theta_05 - theta_06) + g_05_06*cos(theta_05 - theta_06))
        struct[0].Gy[11,10] = V_05*(b_05_06*cos(theta_05 - theta_06) + g_05_06*sin(theta_05 - theta_06)) + 2*V_06*(-b_05_06 - b_06_07 - b_06_09 - bs_05_06/2 - bs_06_07/2 - bs_06_09/2) + V_07*(b_06_07*cos(theta_06 - theta_07) - g_06_07*sin(theta_06 - theta_07)) + V_09*(b_06_09*cos(theta_06 - theta_09) - g_06_09*sin(theta_06 - theta_09))
        struct[0].Gy[11,11] = V_05*V_06*(b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06)) + V_06*V_07*(-b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07)) + V_06*V_09*(-b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09))
        struct[0].Gy[11,12] = V_06*(b_06_07*cos(theta_06 - theta_07) - g_06_07*sin(theta_06 - theta_07))
        struct[0].Gy[11,13] = V_06*V_07*(b_06_07*sin(theta_06 - theta_07) + g_06_07*cos(theta_06 - theta_07))
        struct[0].Gy[11,16] = V_06*(b_06_09*cos(theta_06 - theta_09) - g_06_09*sin(theta_06 - theta_09))
        struct[0].Gy[11,17] = V_06*V_09*(b_06_09*sin(theta_06 - theta_09) + g_06_09*cos(theta_06 - theta_09))
        struct[0].Gy[12,10] = V_07*(b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07))
        struct[0].Gy[12,11] = V_06*V_07*(b_06_07*cos(theta_06 - theta_07) + g_06_07*sin(theta_06 - theta_07))
        struct[0].Gy[12,12] = V_06*(b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07)) + 2*V_07*(g_06_07 + g_07_08) + V_08*(-b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08))
        struct[0].Gy[12,13] = V_06*V_07*(-b_06_07*cos(theta_06 - theta_07) - g_06_07*sin(theta_06 - theta_07)) + V_07*V_08*(-b_07_08*cos(theta_07 - theta_08) + g_07_08*sin(theta_07 - theta_08))
        struct[0].Gy[12,14] = V_07*(-b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08))
        struct[0].Gy[12,15] = V_07*V_08*(b_07_08*cos(theta_07 - theta_08) - g_07_08*sin(theta_07 - theta_08))
        struct[0].Gy[13,10] = V_07*(b_06_07*cos(theta_06 - theta_07) + g_06_07*sin(theta_06 - theta_07))
        struct[0].Gy[13,11] = V_06*V_07*(-b_06_07*sin(theta_06 - theta_07) + g_06_07*cos(theta_06 - theta_07))
        struct[0].Gy[13,12] = V_06*(b_06_07*cos(theta_06 - theta_07) + g_06_07*sin(theta_06 - theta_07)) + 2*V_07*(-b_06_07 - b_07_08 - bs_06_07/2 - bs_07_08/2) + V_08*(b_07_08*cos(theta_07 - theta_08) - g_07_08*sin(theta_07 - theta_08))
        struct[0].Gy[13,13] = V_06*V_07*(b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07)) + V_07*V_08*(-b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08))
        struct[0].Gy[13,14] = V_07*(b_07_08*cos(theta_07 - theta_08) - g_07_08*sin(theta_07 - theta_08))
        struct[0].Gy[13,15] = V_07*V_08*(b_07_08*sin(theta_07 - theta_08) + g_07_08*cos(theta_07 - theta_08))
        struct[0].Gy[14,12] = V_08*(b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08))
        struct[0].Gy[14,13] = V_07*V_08*(b_07_08*cos(theta_07 - theta_08) + g_07_08*sin(theta_07 - theta_08))
        struct[0].Gy[14,14] = V_07*(b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08)) + 2*V_08*g_07_08
        struct[0].Gy[14,15] = V_07*V_08*(-b_07_08*cos(theta_07 - theta_08) - g_07_08*sin(theta_07 - theta_08))
        struct[0].Gy[14,28] = -S_n_08/S_base
        struct[0].Gy[15,12] = V_08*(b_07_08*cos(theta_07 - theta_08) + g_07_08*sin(theta_07 - theta_08))
        struct[0].Gy[15,13] = V_07*V_08*(-b_07_08*sin(theta_07 - theta_08) + g_07_08*cos(theta_07 - theta_08))
        struct[0].Gy[15,14] = V_07*(b_07_08*cos(theta_07 - theta_08) + g_07_08*sin(theta_07 - theta_08)) + 2*V_08*(-b_07_08 - bs_07_08/2)
        struct[0].Gy[15,15] = V_07*V_08*(b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08))
        struct[0].Gy[15,29] = -S_n_08/S_base
        struct[0].Gy[16,10] = V_09*(b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09))
        struct[0].Gy[16,11] = V_06*V_09*(b_06_09*cos(theta_06 - theta_09) + g_06_09*sin(theta_06 - theta_09))
        struct[0].Gy[16,16] = V_06*(b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09)) + 2*V_09*g_06_09
        struct[0].Gy[16,17] = V_06*V_09*(-b_06_09*cos(theta_06 - theta_09) - g_06_09*sin(theta_06 - theta_09))
        struct[0].Gy[16,43] = -S_n_09/S_base
        struct[0].Gy[17,10] = V_09*(b_06_09*cos(theta_06 - theta_09) + g_06_09*sin(theta_06 - theta_09))
        struct[0].Gy[17,11] = V_06*V_09*(-b_06_09*sin(theta_06 - theta_09) + g_06_09*cos(theta_06 - theta_09))
        struct[0].Gy[17,16] = V_06*(b_06_09*cos(theta_06 - theta_09) + g_06_09*sin(theta_06 - theta_09)) + 2*V_09*(-b_06_09 - bs_06_09/2)
        struct[0].Gy[17,17] = V_06*V_09*(b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09))
        struct[0].Gy[17,44] = -S_n_09/S_base
        struct[0].Gy[18,8] = V_10*(b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10))
        struct[0].Gy[18,9] = V_05*V_10*(b_05_10*cos(theta_05 - theta_10) + g_05_10*sin(theta_05 - theta_10))
        struct[0].Gy[18,18] = V_05*(b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10)) + 2*V_10*g_05_10
        struct[0].Gy[18,19] = V_05*V_10*(-b_05_10*cos(theta_05 - theta_10) - g_05_10*sin(theta_05 - theta_10))
        struct[0].Gy[18,58] = -S_n_10/S_base
        struct[0].Gy[19,8] = V_10*(b_05_10*cos(theta_05 - theta_10) + g_05_10*sin(theta_05 - theta_10))
        struct[0].Gy[19,9] = V_05*V_10*(-b_05_10*sin(theta_05 - theta_10) + g_05_10*cos(theta_05 - theta_10))
        struct[0].Gy[19,18] = V_05*(b_05_10*cos(theta_05 - theta_10) + g_05_10*sin(theta_05 - theta_10)) + 2*V_10*(-b_05_10 - bs_05_10/2)
        struct[0].Gy[19,19] = V_05*V_10*(b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10))
        struct[0].Gy[19,59] = -S_n_10/S_base
        struct[0].Gy[20,6] = V_11*(b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11))
        struct[0].Gy[20,7] = V_04*V_11*(b_04_11*cos(theta_04 - theta_11) + g_04_11*sin(theta_04 - theta_11))
        struct[0].Gy[20,20] = V_04*(b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11)) + 2*V_11*g_04_11
        struct[0].Gy[20,21] = V_04*V_11*(-b_04_11*cos(theta_04 - theta_11) - g_04_11*sin(theta_04 - theta_11))
        struct[0].Gy[21,6] = V_11*(b_04_11*cos(theta_04 - theta_11) + g_04_11*sin(theta_04 - theta_11))
        struct[0].Gy[21,7] = V_04*V_11*(-b_04_11*sin(theta_04 - theta_11) + g_04_11*cos(theta_04 - theta_11))
        struct[0].Gy[21,20] = V_04*(b_04_11*cos(theta_04 - theta_11) + g_04_11*sin(theta_04 - theta_11)) + 2*V_11*(-b_04_11 - bs_04_11/2)
        struct[0].Gy[21,21] = V_04*V_11*(b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11))
        struct[0].Gy[22,4] = V_12*(b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12))
        struct[0].Gy[22,5] = V_03*V_12*(b_03_12*cos(theta_03 - theta_12) + g_03_12*sin(theta_03 - theta_12))
        struct[0].Gy[22,22] = V_03*(b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12)) + 2*V_12*g_03_12
        struct[0].Gy[22,23] = V_03*V_12*(-b_03_12*cos(theta_03 - theta_12) - g_03_12*sin(theta_03 - theta_12))
        struct[0].Gy[23,4] = V_12*(b_03_12*cos(theta_03 - theta_12) + g_03_12*sin(theta_03 - theta_12))
        struct[0].Gy[23,5] = V_03*V_12*(-b_03_12*sin(theta_03 - theta_12) + g_03_12*cos(theta_03 - theta_12))
        struct[0].Gy[23,22] = V_03*(b_03_12*cos(theta_03 - theta_12) + g_03_12*sin(theta_03 - theta_12)) + 2*V_12*(-b_03_12 - bs_03_12/2)
        struct[0].Gy[23,23] = V_03*V_12*(b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12))
        struct[0].Gy[24,28] = -K_p_08
        struct[0].Gy[24,30] = K_p_08
        struct[0].Gy[25,29] = -K_q_08
        struct[0].Gy[26,14] = -sin(delta_08 - theta_08)
        struct[0].Gy[26,15] = V_08*cos(delta_08 - theta_08)
        struct[0].Gy[26,26] = -R_v_08
        struct[0].Gy[26,27] = X_v_08
        struct[0].Gy[27,14] = -cos(delta_08 - theta_08)
        struct[0].Gy[27,15] = -V_08*sin(delta_08 - theta_08)
        struct[0].Gy[27,26] = -X_v_08
        struct[0].Gy[27,27] = -R_v_08
        struct[0].Gy[28,14] = i_d_08*sin(delta_08 - theta_08) + i_q_08*cos(delta_08 - theta_08)
        struct[0].Gy[28,15] = -V_08*i_d_08*cos(delta_08 - theta_08) + V_08*i_q_08*sin(delta_08 - theta_08)
        struct[0].Gy[28,26] = V_08*sin(delta_08 - theta_08)
        struct[0].Gy[28,27] = V_08*cos(delta_08 - theta_08)
        struct[0].Gy[29,14] = i_d_08*cos(delta_08 - theta_08) - i_q_08*sin(delta_08 - theta_08)
        struct[0].Gy[29,15] = V_08*i_d_08*sin(delta_08 - theta_08) + V_08*i_q_08*cos(delta_08 - theta_08)
        struct[0].Gy[29,26] = V_08*cos(delta_08 - theta_08)
        struct[0].Gy[29,27] = -V_08*sin(delta_08 - theta_08)
        struct[0].Gy[30,37] = K_fpfr_08*Piecewise(np.array([(0, (P_f_min_08 > p_f_08) | (P_f_max_08 < p_f_08)), (1, True)]))
        struct[0].Gy[31,14] = i_d_08*sin(delta_08 - theta_08) + i_q_08*cos(delta_08 - theta_08)
        struct[0].Gy[31,15] = -V_08*i_d_08*cos(delta_08 - theta_08) + V_08*i_q_08*sin(delta_08 - theta_08)
        struct[0].Gy[31,26] = 2*R_s_08*i_d_08 + V_08*sin(delta_08 - theta_08)
        struct[0].Gy[31,27] = 2*R_s_08*i_q_08 + V_08*cos(delta_08 - theta_08)
        struct[0].Gy[32,33] = 2*k_u_08*v_u_08/V_u_max_08**2
        struct[0].Gy[32,34] = -(-v_u_08**2 + v_u_ref_08**2)/V_u_max_08**2
        struct[0].Gy[33,31] = -R_uc_08*S_n_08/(v_u_08 + 0.1)
        struct[0].Gy[33,33] = -R_uc_08*S_n_08*(p_gou_08 - p_t_08)/(v_u_08 + 0.1)**2 - 1
        struct[0].Gy[33,36] = R_uc_08*S_n_08/(v_u_08 + 0.1)
        struct[0].Gy[34,33] = Piecewise(np.array([(0, V_u_min_08 > v_u_08), ((-K_u_0_08 + K_u_max_08)/(-V_u_lt_08 + V_u_min_08), V_u_lt_08 > v_u_08), ((-K_u_0_08 + K_u_max_08)/(-V_u_ht_08 + V_u_max_08), V_u_ht_08 < v_u_08), (0, True)]))
        struct[0].Gy[36,35] = inc_p_gin_08 + p_gin_0_08
        struct[0].Gy[37,24] = -Piecewise(np.array([(1/Droop_08, (omega_08 > 0.5*DB_08 + omega_ref_08) | (omega_08 < -0.5*DB_08 + omega_ref_08)), (0, True)]))
        struct[0].Gy[38,33] = Piecewise(np.array([((-R_lim_08 + R_lim_max_08)/(-V_u_lt_08 + V_u_min_08), V_u_lt_08 > v_u_08), ((-R_lim_08 + R_lim_max_08)/(-V_u_ht_08 + V_u_max_08), V_u_ht_08 < v_u_08), (0, True)]))
        struct[0].Gy[39,43] = -K_p_09
        struct[0].Gy[39,45] = K_p_09
        struct[0].Gy[40,44] = -K_q_09
        struct[0].Gy[41,16] = -sin(delta_09 - theta_09)
        struct[0].Gy[41,17] = V_09*cos(delta_09 - theta_09)
        struct[0].Gy[41,41] = -R_v_09
        struct[0].Gy[41,42] = X_v_09
        struct[0].Gy[42,16] = -cos(delta_09 - theta_09)
        struct[0].Gy[42,17] = -V_09*sin(delta_09 - theta_09)
        struct[0].Gy[42,41] = -X_v_09
        struct[0].Gy[42,42] = -R_v_09
        struct[0].Gy[43,16] = i_d_09*sin(delta_09 - theta_09) + i_q_09*cos(delta_09 - theta_09)
        struct[0].Gy[43,17] = -V_09*i_d_09*cos(delta_09 - theta_09) + V_09*i_q_09*sin(delta_09 - theta_09)
        struct[0].Gy[43,41] = V_09*sin(delta_09 - theta_09)
        struct[0].Gy[43,42] = V_09*cos(delta_09 - theta_09)
        struct[0].Gy[44,16] = i_d_09*cos(delta_09 - theta_09) - i_q_09*sin(delta_09 - theta_09)
        struct[0].Gy[44,17] = V_09*i_d_09*sin(delta_09 - theta_09) + V_09*i_q_09*cos(delta_09 - theta_09)
        struct[0].Gy[44,41] = V_09*cos(delta_09 - theta_09)
        struct[0].Gy[44,42] = -V_09*sin(delta_09 - theta_09)
        struct[0].Gy[45,52] = K_fpfr_09*Piecewise(np.array([(0, (P_f_min_09 > p_f_09) | (P_f_max_09 < p_f_09)), (1, True)]))
        struct[0].Gy[46,16] = i_d_09*sin(delta_09 - theta_09) + i_q_09*cos(delta_09 - theta_09)
        struct[0].Gy[46,17] = -V_09*i_d_09*cos(delta_09 - theta_09) + V_09*i_q_09*sin(delta_09 - theta_09)
        struct[0].Gy[46,41] = 2*R_s_09*i_d_09 + V_09*sin(delta_09 - theta_09)
        struct[0].Gy[46,42] = 2*R_s_09*i_q_09 + V_09*cos(delta_09 - theta_09)
        struct[0].Gy[47,48] = 2*k_u_09*v_u_09/V_u_max_09**2
        struct[0].Gy[47,49] = -(-v_u_09**2 + v_u_ref_09**2)/V_u_max_09**2
        struct[0].Gy[48,46] = -R_uc_09*S_n_09/(v_u_09 + 0.1)
        struct[0].Gy[48,48] = -R_uc_09*S_n_09*(p_gou_09 - p_t_09)/(v_u_09 + 0.1)**2 - 1
        struct[0].Gy[48,51] = R_uc_09*S_n_09/(v_u_09 + 0.1)
        struct[0].Gy[49,48] = Piecewise(np.array([(0, V_u_min_09 > v_u_09), ((-K_u_0_09 + K_u_max_09)/(-V_u_lt_09 + V_u_min_09), V_u_lt_09 > v_u_09), ((-K_u_0_09 + K_u_max_09)/(-V_u_ht_09 + V_u_max_09), V_u_ht_09 < v_u_09), (0, True)]))
        struct[0].Gy[51,50] = inc_p_gin_09 + p_gin_0_09
        struct[0].Gy[52,39] = -Piecewise(np.array([(1/Droop_09, (omega_09 > 0.5*DB_09 + omega_ref_09) | (omega_09 < -0.5*DB_09 + omega_ref_09)), (0, True)]))
        struct[0].Gy[53,48] = Piecewise(np.array([((-R_lim_09 + R_lim_max_09)/(-V_u_lt_09 + V_u_min_09), V_u_lt_09 > v_u_09), ((-R_lim_09 + R_lim_max_09)/(-V_u_ht_09 + V_u_max_09), V_u_ht_09 < v_u_09), (0, True)]))
        struct[0].Gy[54,58] = -K_p_10
        struct[0].Gy[54,60] = K_p_10
        struct[0].Gy[55,59] = -K_q_10
        struct[0].Gy[56,18] = -sin(delta_10 - theta_10)
        struct[0].Gy[56,19] = V_10*cos(delta_10 - theta_10)
        struct[0].Gy[56,56] = -R_v_10
        struct[0].Gy[56,57] = X_v_10
        struct[0].Gy[57,18] = -cos(delta_10 - theta_10)
        struct[0].Gy[57,19] = -V_10*sin(delta_10 - theta_10)
        struct[0].Gy[57,56] = -X_v_10
        struct[0].Gy[57,57] = -R_v_10
        struct[0].Gy[58,18] = i_d_10*sin(delta_10 - theta_10) + i_q_10*cos(delta_10 - theta_10)
        struct[0].Gy[58,19] = -V_10*i_d_10*cos(delta_10 - theta_10) + V_10*i_q_10*sin(delta_10 - theta_10)
        struct[0].Gy[58,56] = V_10*sin(delta_10 - theta_10)
        struct[0].Gy[58,57] = V_10*cos(delta_10 - theta_10)
        struct[0].Gy[59,18] = i_d_10*cos(delta_10 - theta_10) - i_q_10*sin(delta_10 - theta_10)
        struct[0].Gy[59,19] = V_10*i_d_10*sin(delta_10 - theta_10) + V_10*i_q_10*cos(delta_10 - theta_10)
        struct[0].Gy[59,56] = V_10*cos(delta_10 - theta_10)
        struct[0].Gy[59,57] = -V_10*sin(delta_10 - theta_10)
        struct[0].Gy[60,67] = K_fpfr_10*Piecewise(np.array([(0, (P_f_min_10 > p_f_10) | (P_f_max_10 < p_f_10)), (1, True)]))
        struct[0].Gy[61,18] = i_d_10*sin(delta_10 - theta_10) + i_q_10*cos(delta_10 - theta_10)
        struct[0].Gy[61,19] = -V_10*i_d_10*cos(delta_10 - theta_10) + V_10*i_q_10*sin(delta_10 - theta_10)
        struct[0].Gy[61,56] = 2*R_s_10*i_d_10 + V_10*sin(delta_10 - theta_10)
        struct[0].Gy[61,57] = 2*R_s_10*i_q_10 + V_10*cos(delta_10 - theta_10)
        struct[0].Gy[62,63] = 2*k_u_10*v_u_10/V_u_max_10**2
        struct[0].Gy[62,64] = -(-v_u_10**2 + v_u_ref_10**2)/V_u_max_10**2
        struct[0].Gy[63,61] = -R_uc_10*S_n_10/(v_u_10 + 0.1)
        struct[0].Gy[63,63] = -R_uc_10*S_n_10*(p_gou_10 - p_t_10)/(v_u_10 + 0.1)**2 - 1
        struct[0].Gy[63,66] = R_uc_10*S_n_10/(v_u_10 + 0.1)
        struct[0].Gy[64,63] = Piecewise(np.array([(0, V_u_min_10 > v_u_10), ((-K_u_0_10 + K_u_max_10)/(-V_u_lt_10 + V_u_min_10), V_u_lt_10 > v_u_10), ((-K_u_0_10 + K_u_max_10)/(-V_u_ht_10 + V_u_max_10), V_u_ht_10 < v_u_10), (0, True)]))
        struct[0].Gy[66,65] = inc_p_gin_10 + p_gin_0_10
        struct[0].Gy[67,54] = -Piecewise(np.array([(1/Droop_10, (omega_10 > 0.5*DB_10 + omega_ref_10) | (omega_10 < -0.5*DB_10 + omega_ref_10)), (0, True)]))
        struct[0].Gy[68,63] = Piecewise(np.array([((-R_lim_10 + R_lim_max_10)/(-V_u_lt_10 + V_u_min_10), V_u_lt_10 > v_u_10), ((-R_lim_10 + R_lim_max_10)/(-V_u_ht_10 + V_u_max_10), V_u_ht_10 < v_u_10), (0, True)]))
        struct[0].Gy[70,0] = -sin(delta_01 - theta_01)
        struct[0].Gy[70,1] = V_01*cos(delta_01 - theta_01)
        struct[0].Gy[70,70] = -R_v_01
        struct[0].Gy[70,71] = X_v_01
        struct[0].Gy[71,0] = -cos(delta_01 - theta_01)
        struct[0].Gy[71,1] = -V_01*sin(delta_01 - theta_01)
        struct[0].Gy[71,70] = -X_v_01
        struct[0].Gy[71,71] = -R_v_01
        struct[0].Gy[72,0] = i_d_01*sin(delta_01 - theta_01) + i_q_01*cos(delta_01 - theta_01)
        struct[0].Gy[72,1] = -V_01*i_d_01*cos(delta_01 - theta_01) + V_01*i_q_01*sin(delta_01 - theta_01)
        struct[0].Gy[72,70] = V_01*sin(delta_01 - theta_01)
        struct[0].Gy[72,71] = V_01*cos(delta_01 - theta_01)
        struct[0].Gy[73,0] = i_d_01*cos(delta_01 - theta_01) - i_q_01*sin(delta_01 - theta_01)
        struct[0].Gy[73,1] = V_01*i_d_01*sin(delta_01 - theta_01) + V_01*i_q_01*cos(delta_01 - theta_01)
        struct[0].Gy[73,70] = V_01*cos(delta_01 - theta_01)
        struct[0].Gy[73,71] = -V_01*sin(delta_01 - theta_01)
        struct[0].Gy[74,24] = S_n_08*T_p_08/(2*K_p_08*(1000000.0*S_n_01 + S_n_10*T_p_10/(2*K_p_10) + S_n_09*T_p_09/(2*K_p_09) + S_n_08*T_p_08/(2*K_p_08)))
        struct[0].Gy[74,39] = S_n_09*T_p_09/(2*K_p_09*(1000000.0*S_n_01 + S_n_10*T_p_10/(2*K_p_10) + S_n_09*T_p_09/(2*K_p_09) + S_n_08*T_p_08/(2*K_p_08)))
        struct[0].Gy[74,54] = S_n_10*T_p_10/(2*K_p_10*(1000000.0*S_n_01 + S_n_10*T_p_10/(2*K_p_10) + S_n_09*T_p_09/(2*K_p_09) + S_n_08*T_p_08/(2*K_p_08)))
        struct[0].Gy[74,69] = 1000000.0*S_n_01/(1000000.0*S_n_01 + S_n_10*T_p_10/(2*K_p_10) + S_n_09*T_p_09/(2*K_p_09) + S_n_08*T_p_08/(2*K_p_08))
        struct[0].Gy[75,74] = -K_p_agc

    if mode > 12:

        struct[0].Fu[2,24] = 1
        struct[0].Fu[5,27] = (-p_g_ref_08/(inc_p_gin_08 + p_gin_0_08)**2 - Piecewise(np.array([(P_f_min_08, P_f_min_08 > p_f_08), (P_f_max_08, P_f_max_08 < p_f_08), (p_f_08, True)]))/(inc_p_gin_08 + p_gin_0_08)**2)/T_cur_08
        struct[0].Fu[5,28] = 1/(T_cur_08*(inc_p_gin_08 + p_gin_0_08))
        struct[0].Fu[6,29] = 1
        struct[0].Fu[9,30] = 1
        struct[0].Fu[12,33] = (-p_g_ref_09/(inc_p_gin_09 + p_gin_0_09)**2 - Piecewise(np.array([(P_f_min_09, P_f_min_09 > p_f_09), (P_f_max_09, P_f_max_09 < p_f_09), (p_f_09, True)]))/(inc_p_gin_09 + p_gin_0_09)**2)/T_cur_09
        struct[0].Fu[12,34] = 1/(T_cur_09*(inc_p_gin_09 + p_gin_0_09))
        struct[0].Fu[13,35] = 1
        struct[0].Fu[16,36] = 1
        struct[0].Fu[19,39] = (-p_g_ref_10/(inc_p_gin_10 + p_gin_0_10)**2 - Piecewise(np.array([(P_f_min_10, P_f_min_10 > p_f_10), (P_f_max_10, P_f_max_10 < p_f_10), (p_f_10, True)]))/(inc_p_gin_10 + p_gin_0_10)**2)/T_cur_10
        struct[0].Fu[19,40] = 1/(T_cur_10*(inc_p_gin_10 + p_gin_0_10))
        struct[0].Fu[20,41] = 1
        struct[0].Fu[22,42] = 1

        struct[0].Gu[0,0] = -1/S_base
        struct[0].Gu[1,1] = -1/S_base
        struct[0].Gu[2,2] = -1/S_base
        struct[0].Gu[3,3] = -1/S_base
        struct[0].Gu[4,4] = -1/S_base
        struct[0].Gu[5,5] = -1/S_base
        struct[0].Gu[6,6] = -1/S_base
        struct[0].Gu[7,7] = -1/S_base
        struct[0].Gu[8,8] = -1/S_base
        struct[0].Gu[9,9] = -1/S_base
        struct[0].Gu[10,10] = -1/S_base
        struct[0].Gu[11,11] = -1/S_base
        struct[0].Gu[12,12] = -1/S_base
        struct[0].Gu[13,13] = -1/S_base
        struct[0].Gu[14,14] = -1/S_base
        struct[0].Gu[15,15] = -1/S_base
        struct[0].Gu[16,16] = -1/S_base
        struct[0].Gu[17,17] = -1/S_base
        struct[0].Gu[18,18] = -1/S_base
        struct[0].Gu[19,19] = -1/S_base
        struct[0].Gu[20,20] = -1/S_base
        struct[0].Gu[21,21] = -1/S_base
        struct[0].Gu[22,22] = -1/S_base
        struct[0].Gu[23,23] = -1/S_base
        struct[0].Gu[25,24] = K_q_08
        struct[0].Gu[32,25] = -2*k_u_08*v_u_ref_08/V_u_max_08**2
        struct[0].Gu[36,27] = k_cur_sat_08
        struct[0].Gu[37,26] = -Piecewise(np.array([(-1/Droop_08, (omega_08 > 0.5*DB_08 + omega_ref_08) | (omega_08 < -0.5*DB_08 + omega_ref_08)), (0, True)]))
        struct[0].Gu[40,30] = K_q_09
        struct[0].Gu[47,31] = -2*k_u_09*v_u_ref_09/V_u_max_09**2
        struct[0].Gu[51,33] = k_cur_sat_09
        struct[0].Gu[52,32] = -Piecewise(np.array([(-1/Droop_09, (omega_09 > 0.5*DB_09 + omega_ref_09) | (omega_09 < -0.5*DB_09 + omega_ref_09)), (0, True)]))
        struct[0].Gu[55,36] = K_q_10
        struct[0].Gu[62,37] = -2*k_u_10*v_u_ref_10/V_u_max_10**2
        struct[0].Gu[66,39] = k_cur_sat_10
        struct[0].Gu[67,38] = -Piecewise(np.array([(-1/Droop_10, (omega_10 > 0.5*DB_10 + omega_ref_10) | (omega_10 < -0.5*DB_10 + omega_ref_10)), (0, True)]))

        struct[0].Hx[12,6] = 1
        struct[0].Hx[15,3] = 2*e_u_08/(V_u_max_08**2 - V_u_min_08**2)
        struct[0].Hx[18,13] = 1
        struct[0].Hx[21,10] = 2*e_u_09/(V_u_max_09**2 - V_u_min_09**2)
        struct[0].Hx[24,20] = 1
        struct[0].Hx[27,17] = 2*e_u_10/(V_u_max_10**2 - V_u_min_10**2)

        struct[0].Hy[0,0] = 1
        struct[0].Hy[1,2] = 1
        struct[0].Hy[2,4] = 1
        struct[0].Hy[3,6] = 1
        struct[0].Hy[4,8] = 1
        struct[0].Hy[5,10] = 1
        struct[0].Hy[6,12] = 1
        struct[0].Hy[7,14] = 1
        struct[0].Hy[8,16] = 1
        struct[0].Hy[9,18] = 1
        struct[0].Hy[10,20] = 1
        struct[0].Hy[11,22] = 1
        struct[0].Hy[14,28] = -1
        struct[0].Hy[14,31] = 1
        struct[0].Hy[16,37] = K_fpfr_08*Piecewise(np.array([(0, (P_f_min_08 > p_f_08) | (P_f_max_08 < p_f_08)), (1, True)]))
        struct[0].Hy[17,37] = Piecewise(np.array([(0, (P_f_min_08 > p_f_08) | (P_f_max_08 < p_f_08)), (1, True)]))
        struct[0].Hy[20,43] = -1
        struct[0].Hy[20,46] = 1
        struct[0].Hy[22,52] = K_fpfr_09*Piecewise(np.array([(0, (P_f_min_09 > p_f_09) | (P_f_max_09 < p_f_09)), (1, True)]))
        struct[0].Hy[23,52] = Piecewise(np.array([(0, (P_f_min_09 > p_f_09) | (P_f_max_09 < p_f_09)), (1, True)]))
        struct[0].Hy[26,58] = -1
        struct[0].Hy[26,61] = 1
        struct[0].Hy[28,67] = K_fpfr_10*Piecewise(np.array([(0, (P_f_min_10 > p_f_10) | (P_f_max_10 < p_f_10)), (1, True)]))
        struct[0].Hy[29,67] = Piecewise(np.array([(0, (P_f_min_10 > p_f_10) | (P_f_max_10 < p_f_10)), (1, True)]))

        struct[0].Hu[12,27] = 1
        struct[0].Hu[13,28] = 1
        struct[0].Hu[18,33] = 1
        struct[0].Hu[19,34] = 1
        struct[0].Hu[24,39] = 1
        struct[0].Hu[25,40] = 1
        struct[0].Hu[30,42] = 1



def ini_nn(struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_01_02 = struct[0].g_01_02
    b_01_02 = struct[0].b_01_02
    bs_01_02 = struct[0].bs_01_02
    g_02_03 = struct[0].g_02_03
    b_02_03 = struct[0].b_02_03
    bs_02_03 = struct[0].bs_02_03
    g_03_04 = struct[0].g_03_04
    b_03_04 = struct[0].b_03_04
    bs_03_04 = struct[0].bs_03_04
    g_04_05 = struct[0].g_04_05
    b_04_05 = struct[0].b_04_05
    bs_04_05 = struct[0].bs_04_05
    g_05_06 = struct[0].g_05_06
    b_05_06 = struct[0].b_05_06
    bs_05_06 = struct[0].bs_05_06
    g_06_07 = struct[0].g_06_07
    b_06_07 = struct[0].b_06_07
    bs_06_07 = struct[0].bs_06_07
    g_07_08 = struct[0].g_07_08
    b_07_08 = struct[0].b_07_08
    bs_07_08 = struct[0].bs_07_08
    g_06_09 = struct[0].g_06_09
    b_06_09 = struct[0].b_06_09
    bs_06_09 = struct[0].bs_06_09
    g_05_10 = struct[0].g_05_10
    b_05_10 = struct[0].b_05_10
    bs_05_10 = struct[0].bs_05_10
    g_04_11 = struct[0].g_04_11
    b_04_11 = struct[0].b_04_11
    bs_04_11 = struct[0].bs_04_11
    g_03_12 = struct[0].g_03_12
    b_03_12 = struct[0].b_03_12
    bs_03_12 = struct[0].bs_03_12
    U_01_n = struct[0].U_01_n
    U_02_n = struct[0].U_02_n
    U_03_n = struct[0].U_03_n
    U_04_n = struct[0].U_04_n
    U_05_n = struct[0].U_05_n
    U_06_n = struct[0].U_06_n
    U_07_n = struct[0].U_07_n
    U_08_n = struct[0].U_08_n
    U_09_n = struct[0].U_09_n
    U_10_n = struct[0].U_10_n
    U_11_n = struct[0].U_11_n
    U_12_n = struct[0].U_12_n
    S_n_08 = struct[0].S_n_08
    Omega_b_08 = struct[0].Omega_b_08
    K_p_08 = struct[0].K_p_08
    T_p_08 = struct[0].T_p_08
    K_q_08 = struct[0].K_q_08
    T_q_08 = struct[0].T_q_08
    X_v_08 = struct[0].X_v_08
    R_v_08 = struct[0].R_v_08
    R_s_08 = struct[0].R_s_08
    C_u_08 = struct[0].C_u_08
    K_u_0_08 = struct[0].K_u_0_08
    K_u_max_08 = struct[0].K_u_max_08
    V_u_min_08 = struct[0].V_u_min_08
    V_u_max_08 = struct[0].V_u_max_08
    R_uc_08 = struct[0].R_uc_08
    K_h_08 = struct[0].K_h_08
    R_lim_08 = struct[0].R_lim_08
    V_u_lt_08 = struct[0].V_u_lt_08
    V_u_ht_08 = struct[0].V_u_ht_08
    Droop_08 = struct[0].Droop_08
    DB_08 = struct[0].DB_08
    T_cur_08 = struct[0].T_cur_08
    R_lim_max_08 = struct[0].R_lim_max_08
    K_fpfr_08 = struct[0].K_fpfr_08
    P_f_min_08 = struct[0].P_f_min_08
    P_f_max_08 = struct[0].P_f_max_08
    S_n_09 = struct[0].S_n_09
    Omega_b_09 = struct[0].Omega_b_09
    K_p_09 = struct[0].K_p_09
    T_p_09 = struct[0].T_p_09
    K_q_09 = struct[0].K_q_09
    T_q_09 = struct[0].T_q_09
    X_v_09 = struct[0].X_v_09
    R_v_09 = struct[0].R_v_09
    R_s_09 = struct[0].R_s_09
    C_u_09 = struct[0].C_u_09
    K_u_0_09 = struct[0].K_u_0_09
    K_u_max_09 = struct[0].K_u_max_09
    V_u_min_09 = struct[0].V_u_min_09
    V_u_max_09 = struct[0].V_u_max_09
    R_uc_09 = struct[0].R_uc_09
    K_h_09 = struct[0].K_h_09
    R_lim_09 = struct[0].R_lim_09
    V_u_lt_09 = struct[0].V_u_lt_09
    V_u_ht_09 = struct[0].V_u_ht_09
    Droop_09 = struct[0].Droop_09
    DB_09 = struct[0].DB_09
    T_cur_09 = struct[0].T_cur_09
    R_lim_max_09 = struct[0].R_lim_max_09
    K_fpfr_09 = struct[0].K_fpfr_09
    P_f_min_09 = struct[0].P_f_min_09
    P_f_max_09 = struct[0].P_f_max_09
    S_n_10 = struct[0].S_n_10
    Omega_b_10 = struct[0].Omega_b_10
    K_p_10 = struct[0].K_p_10
    T_p_10 = struct[0].T_p_10
    K_q_10 = struct[0].K_q_10
    T_q_10 = struct[0].T_q_10
    X_v_10 = struct[0].X_v_10
    R_v_10 = struct[0].R_v_10
    R_s_10 = struct[0].R_s_10
    C_u_10 = struct[0].C_u_10
    K_u_0_10 = struct[0].K_u_0_10
    K_u_max_10 = struct[0].K_u_max_10
    V_u_min_10 = struct[0].V_u_min_10
    V_u_max_10 = struct[0].V_u_max_10
    R_uc_10 = struct[0].R_uc_10
    K_h_10 = struct[0].K_h_10
    R_lim_10 = struct[0].R_lim_10
    V_u_lt_10 = struct[0].V_u_lt_10
    V_u_ht_10 = struct[0].V_u_ht_10
    Droop_10 = struct[0].Droop_10
    DB_10 = struct[0].DB_10
    T_cur_10 = struct[0].T_cur_10
    R_lim_max_10 = struct[0].R_lim_max_10
    K_fpfr_10 = struct[0].K_fpfr_10
    P_f_min_10 = struct[0].P_f_min_10
    P_f_max_10 = struct[0].P_f_max_10
    S_n_01 = struct[0].S_n_01
    Omega_b_01 = struct[0].Omega_b_01
    X_v_01 = struct[0].X_v_01
    R_v_01 = struct[0].R_v_01
    K_delta_01 = struct[0].K_delta_01
    K_alpha_01 = struct[0].K_alpha_01
    K_p_agc = struct[0].K_p_agc
    K_i_agc = struct[0].K_i_agc
    
    # Inputs:
    P_01 = struct[0].P_01
    Q_01 = struct[0].Q_01
    P_02 = struct[0].P_02
    Q_02 = struct[0].Q_02
    P_03 = struct[0].P_03
    Q_03 = struct[0].Q_03
    P_04 = struct[0].P_04
    Q_04 = struct[0].Q_04
    P_05 = struct[0].P_05
    Q_05 = struct[0].Q_05
    P_06 = struct[0].P_06
    Q_06 = struct[0].Q_06
    P_07 = struct[0].P_07
    Q_07 = struct[0].Q_07
    P_08 = struct[0].P_08
    Q_08 = struct[0].Q_08
    P_09 = struct[0].P_09
    Q_09 = struct[0].Q_09
    P_10 = struct[0].P_10
    Q_10 = struct[0].Q_10
    P_11 = struct[0].P_11
    Q_11 = struct[0].Q_11
    P_12 = struct[0].P_12
    Q_12 = struct[0].Q_12
    q_s_ref_08 = struct[0].q_s_ref_08
    v_u_ref_08 = struct[0].v_u_ref_08
    omega_ref_08 = struct[0].omega_ref_08
    p_gin_0_08 = struct[0].p_gin_0_08
    p_g_ref_08 = struct[0].p_g_ref_08
    ramp_p_gin_08 = struct[0].ramp_p_gin_08
    q_s_ref_09 = struct[0].q_s_ref_09
    v_u_ref_09 = struct[0].v_u_ref_09
    omega_ref_09 = struct[0].omega_ref_09
    p_gin_0_09 = struct[0].p_gin_0_09
    p_g_ref_09 = struct[0].p_g_ref_09
    ramp_p_gin_09 = struct[0].ramp_p_gin_09
    q_s_ref_10 = struct[0].q_s_ref_10
    v_u_ref_10 = struct[0].v_u_ref_10
    omega_ref_10 = struct[0].omega_ref_10
    p_gin_0_10 = struct[0].p_gin_0_10
    p_g_ref_10 = struct[0].p_g_ref_10
    ramp_p_gin_10 = struct[0].ramp_p_gin_10
    alpha_01 = struct[0].alpha_01
    e_qv_01 = struct[0].e_qv_01
    omega_ref_01 = struct[0].omega_ref_01
    
    # Dynamical states:
    delta_08 = struct[0].x[0,0]
    xi_p_08 = struct[0].x[1,0]
    xi_q_08 = struct[0].x[2,0]
    e_u_08 = struct[0].x[3,0]
    p_ghr_08 = struct[0].x[4,0]
    k_cur_08 = struct[0].x[5,0]
    inc_p_gin_08 = struct[0].x[6,0]
    delta_09 = struct[0].x[7,0]
    xi_p_09 = struct[0].x[8,0]
    xi_q_09 = struct[0].x[9,0]
    e_u_09 = struct[0].x[10,0]
    p_ghr_09 = struct[0].x[11,0]
    k_cur_09 = struct[0].x[12,0]
    inc_p_gin_09 = struct[0].x[13,0]
    delta_10 = struct[0].x[14,0]
    xi_p_10 = struct[0].x[15,0]
    xi_q_10 = struct[0].x[16,0]
    e_u_10 = struct[0].x[17,0]
    p_ghr_10 = struct[0].x[18,0]
    k_cur_10 = struct[0].x[19,0]
    inc_p_gin_10 = struct[0].x[20,0]
    delta_01 = struct[0].x[21,0]
    Domega_01 = struct[0].x[22,0]
    xi_freq = struct[0].x[23,0]
    
    # Algebraic states:
    V_01 = struct[0].y_ini[0,0]
    theta_01 = struct[0].y_ini[1,0]
    V_02 = struct[0].y_ini[2,0]
    theta_02 = struct[0].y_ini[3,0]
    V_03 = struct[0].y_ini[4,0]
    theta_03 = struct[0].y_ini[5,0]
    V_04 = struct[0].y_ini[6,0]
    theta_04 = struct[0].y_ini[7,0]
    V_05 = struct[0].y_ini[8,0]
    theta_05 = struct[0].y_ini[9,0]
    V_06 = struct[0].y_ini[10,0]
    theta_06 = struct[0].y_ini[11,0]
    V_07 = struct[0].y_ini[12,0]
    theta_07 = struct[0].y_ini[13,0]
    V_08 = struct[0].y_ini[14,0]
    theta_08 = struct[0].y_ini[15,0]
    V_09 = struct[0].y_ini[16,0]
    theta_09 = struct[0].y_ini[17,0]
    V_10 = struct[0].y_ini[18,0]
    theta_10 = struct[0].y_ini[19,0]
    V_11 = struct[0].y_ini[20,0]
    theta_11 = struct[0].y_ini[21,0]
    V_12 = struct[0].y_ini[22,0]
    theta_12 = struct[0].y_ini[23,0]
    omega_08 = struct[0].y_ini[24,0]
    e_qv_08 = struct[0].y_ini[25,0]
    i_d_08 = struct[0].y_ini[26,0]
    i_q_08 = struct[0].y_ini[27,0]
    p_s_08 = struct[0].y_ini[28,0]
    q_s_08 = struct[0].y_ini[29,0]
    p_m_08 = struct[0].y_ini[30,0]
    p_t_08 = struct[0].y_ini[31,0]
    p_u_08 = struct[0].y_ini[32,0]
    v_u_08 = struct[0].y_ini[33,0]
    k_u_08 = struct[0].y_ini[34,0]
    k_cur_sat_08 = struct[0].y_ini[35,0]
    p_gou_08 = struct[0].y_ini[36,0]
    p_f_08 = struct[0].y_ini[37,0]
    r_lim_08 = struct[0].y_ini[38,0]
    omega_09 = struct[0].y_ini[39,0]
    e_qv_09 = struct[0].y_ini[40,0]
    i_d_09 = struct[0].y_ini[41,0]
    i_q_09 = struct[0].y_ini[42,0]
    p_s_09 = struct[0].y_ini[43,0]
    q_s_09 = struct[0].y_ini[44,0]
    p_m_09 = struct[0].y_ini[45,0]
    p_t_09 = struct[0].y_ini[46,0]
    p_u_09 = struct[0].y_ini[47,0]
    v_u_09 = struct[0].y_ini[48,0]
    k_u_09 = struct[0].y_ini[49,0]
    k_cur_sat_09 = struct[0].y_ini[50,0]
    p_gou_09 = struct[0].y_ini[51,0]
    p_f_09 = struct[0].y_ini[52,0]
    r_lim_09 = struct[0].y_ini[53,0]
    omega_10 = struct[0].y_ini[54,0]
    e_qv_10 = struct[0].y_ini[55,0]
    i_d_10 = struct[0].y_ini[56,0]
    i_q_10 = struct[0].y_ini[57,0]
    p_s_10 = struct[0].y_ini[58,0]
    q_s_10 = struct[0].y_ini[59,0]
    p_m_10 = struct[0].y_ini[60,0]
    p_t_10 = struct[0].y_ini[61,0]
    p_u_10 = struct[0].y_ini[62,0]
    v_u_10 = struct[0].y_ini[63,0]
    k_u_10 = struct[0].y_ini[64,0]
    k_cur_sat_10 = struct[0].y_ini[65,0]
    p_gou_10 = struct[0].y_ini[66,0]
    p_f_10 = struct[0].y_ini[67,0]
    r_lim_10 = struct[0].y_ini[68,0]
    omega_01 = struct[0].y_ini[69,0]
    i_d_01 = struct[0].y_ini[70,0]
    i_q_01 = struct[0].y_ini[71,0]
    p_s_01 = struct[0].y_ini[72,0]
    q_s_01 = struct[0].y_ini[73,0]
    omega_coi = struct[0].y_ini[74,0]
    p_agc = struct[0].y_ini[75,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = Omega_b_08*(omega_08 - omega_coi)
        struct[0].f[1,0] = p_m_08 - p_s_08
        struct[0].f[2,0] = -q_s_08 + q_s_ref_08
        struct[0].f[3,0] = S_n_08*(p_gou_08 - p_t_08)/(C_u_08*(v_u_08 + 0.1))
        struct[0].f[4,0] = Piecewise(np.array([(-r_lim_08, r_lim_08 < -K_h_08*(-p_ghr_08 + p_gou_08)), (r_lim_08, r_lim_08 < K_h_08*(-p_ghr_08 + p_gou_08)), (K_h_08*(-p_ghr_08 + p_gou_08), True)]))
        struct[0].f[5,0] = (-k_cur_08 + p_g_ref_08/(inc_p_gin_08 + p_gin_0_08) + Piecewise(np.array([(P_f_min_08, P_f_min_08 > p_f_08), (P_f_max_08, P_f_max_08 < p_f_08), (p_f_08, True)]))/(inc_p_gin_08 + p_gin_0_08))/T_cur_08
        struct[0].f[6,0] = -0.001*inc_p_gin_08 + ramp_p_gin_08
        struct[0].f[7,0] = Omega_b_09*(omega_09 - omega_coi)
        struct[0].f[8,0] = p_m_09 - p_s_09
        struct[0].f[9,0] = -q_s_09 + q_s_ref_09
        struct[0].f[10,0] = S_n_09*(p_gou_09 - p_t_09)/(C_u_09*(v_u_09 + 0.1))
        struct[0].f[11,0] = Piecewise(np.array([(-r_lim_09, r_lim_09 < -K_h_09*(-p_ghr_09 + p_gou_09)), (r_lim_09, r_lim_09 < K_h_09*(-p_ghr_09 + p_gou_09)), (K_h_09*(-p_ghr_09 + p_gou_09), True)]))
        struct[0].f[12,0] = (-k_cur_09 + p_g_ref_09/(inc_p_gin_09 + p_gin_0_09) + Piecewise(np.array([(P_f_min_09, P_f_min_09 > p_f_09), (P_f_max_09, P_f_max_09 < p_f_09), (p_f_09, True)]))/(inc_p_gin_09 + p_gin_0_09))/T_cur_09
        struct[0].f[13,0] = -0.001*inc_p_gin_09 + ramp_p_gin_09
        struct[0].f[14,0] = Omega_b_10*(omega_10 - omega_coi)
        struct[0].f[15,0] = p_m_10 - p_s_10
        struct[0].f[16,0] = -q_s_10 + q_s_ref_10
        struct[0].f[17,0] = S_n_10*(p_gou_10 - p_t_10)/(C_u_10*(v_u_10 + 0.1))
        struct[0].f[18,0] = Piecewise(np.array([(-r_lim_10, r_lim_10 < -K_h_10*(-p_ghr_10 + p_gou_10)), (r_lim_10, r_lim_10 < K_h_10*(-p_ghr_10 + p_gou_10)), (K_h_10*(-p_ghr_10 + p_gou_10), True)]))
        struct[0].f[19,0] = (-k_cur_10 + p_g_ref_10/(inc_p_gin_10 + p_gin_0_10) + Piecewise(np.array([(P_f_min_10, P_f_min_10 > p_f_10), (P_f_max_10, P_f_max_10 < p_f_10), (p_f_10, True)]))/(inc_p_gin_10 + p_gin_0_10))/T_cur_10
        struct[0].f[20,0] = -0.001*inc_p_gin_10 + ramp_p_gin_10
        struct[0].f[21,0] = -K_delta_01*delta_01 + Omega_b_01*(omega_01 - omega_coi)
        struct[0].f[22,0] = -Domega_01*K_alpha_01 + alpha_01
        struct[0].f[23,0] = 1 - omega_coi
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = -P_01/S_base + V_01**2*g_01_02 + V_01*V_02*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) - S_n_01*p_s_01/S_base
        struct[0].g[1,0] = -Q_01/S_base + V_01**2*(-b_01_02 - bs_01_02/2) + V_01*V_02*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02)) - S_n_01*q_s_01/S_base
        struct[0].g[2,0] = -P_02/S_base + V_01*V_02*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) + V_02**2*(g_01_02 + g_02_03) + V_02*V_03*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].g[3,0] = -Q_02/S_base + V_01*V_02*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02)) + V_02**2*(-b_01_02 - b_02_03 - bs_01_02/2 - bs_02_03/2) + V_02*V_03*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].g[4,0] = -P_03/S_base + V_02*V_03*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03)) + V_03**2*(g_02_03 + g_03_04 + g_03_12) + V_03*V_04*(-b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04)) + V_03*V_12*(-b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12))
        struct[0].g[5,0] = -Q_03/S_base + V_02*V_03*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03)) + V_03**2*(-b_02_03 - b_03_04 - b_03_12 - bs_02_03/2 - bs_03_04/2 - bs_03_12/2) + V_03*V_04*(b_03_04*cos(theta_03 - theta_04) - g_03_04*sin(theta_03 - theta_04)) + V_03*V_12*(b_03_12*cos(theta_03 - theta_12) - g_03_12*sin(theta_03 - theta_12))
        struct[0].g[6,0] = -P_04/S_base + V_03*V_04*(b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04)) + V_04**2*(g_03_04 + g_04_05 + g_04_11) + V_04*V_05*(-b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05)) + V_04*V_11*(-b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11))
        struct[0].g[7,0] = -Q_04/S_base + V_03*V_04*(b_03_04*cos(theta_03 - theta_04) + g_03_04*sin(theta_03 - theta_04)) + V_04**2*(-b_03_04 - b_04_05 - b_04_11 - bs_03_04/2 - bs_04_05/2 - bs_04_11/2) + V_04*V_05*(b_04_05*cos(theta_04 - theta_05) - g_04_05*sin(theta_04 - theta_05)) + V_04*V_11*(b_04_11*cos(theta_04 - theta_11) - g_04_11*sin(theta_04 - theta_11))
        struct[0].g[8,0] = -P_05/S_base + V_04*V_05*(b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05)) + V_05**2*(g_04_05 + g_05_06 + g_05_10) + V_05*V_06*(-b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06)) + V_05*V_10*(-b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10))
        struct[0].g[9,0] = -Q_05/S_base + V_04*V_05*(b_04_05*cos(theta_04 - theta_05) + g_04_05*sin(theta_04 - theta_05)) + V_05**2*(-b_04_05 - b_05_06 - b_05_10 - bs_04_05/2 - bs_05_06/2 - bs_05_10/2) + V_05*V_06*(b_05_06*cos(theta_05 - theta_06) - g_05_06*sin(theta_05 - theta_06)) + V_05*V_10*(b_05_10*cos(theta_05 - theta_10) - g_05_10*sin(theta_05 - theta_10))
        struct[0].g[10,0] = -P_06/S_base + V_05*V_06*(b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06)) + V_06**2*(g_05_06 + g_06_07 + g_06_09) + V_06*V_07*(-b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07)) + V_06*V_09*(-b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09))
        struct[0].g[11,0] = -Q_06/S_base + V_05*V_06*(b_05_06*cos(theta_05 - theta_06) + g_05_06*sin(theta_05 - theta_06)) + V_06**2*(-b_05_06 - b_06_07 - b_06_09 - bs_05_06/2 - bs_06_07/2 - bs_06_09/2) + V_06*V_07*(b_06_07*cos(theta_06 - theta_07) - g_06_07*sin(theta_06 - theta_07)) + V_06*V_09*(b_06_09*cos(theta_06 - theta_09) - g_06_09*sin(theta_06 - theta_09))
        struct[0].g[12,0] = -P_07/S_base + V_06*V_07*(b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07)) + V_07**2*(g_06_07 + g_07_08) + V_07*V_08*(-b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08))
        struct[0].g[13,0] = -Q_07/S_base + V_06*V_07*(b_06_07*cos(theta_06 - theta_07) + g_06_07*sin(theta_06 - theta_07)) + V_07**2*(-b_06_07 - b_07_08 - bs_06_07/2 - bs_07_08/2) + V_07*V_08*(b_07_08*cos(theta_07 - theta_08) - g_07_08*sin(theta_07 - theta_08))
        struct[0].g[14,0] = -P_08/S_base + V_07*V_08*(b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08)) + V_08**2*g_07_08 - S_n_08*p_s_08/S_base
        struct[0].g[15,0] = -Q_08/S_base + V_07*V_08*(b_07_08*cos(theta_07 - theta_08) + g_07_08*sin(theta_07 - theta_08)) + V_08**2*(-b_07_08 - bs_07_08/2) - S_n_08*q_s_08/S_base
        struct[0].g[16,0] = -P_09/S_base + V_06*V_09*(b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09)) + V_09**2*g_06_09 - S_n_09*p_s_09/S_base
        struct[0].g[17,0] = -Q_09/S_base + V_06*V_09*(b_06_09*cos(theta_06 - theta_09) + g_06_09*sin(theta_06 - theta_09)) + V_09**2*(-b_06_09 - bs_06_09/2) - S_n_09*q_s_09/S_base
        struct[0].g[18,0] = -P_10/S_base + V_05*V_10*(b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10)) + V_10**2*g_05_10 - S_n_10*p_s_10/S_base
        struct[0].g[19,0] = -Q_10/S_base + V_05*V_10*(b_05_10*cos(theta_05 - theta_10) + g_05_10*sin(theta_05 - theta_10)) + V_10**2*(-b_05_10 - bs_05_10/2) - S_n_10*q_s_10/S_base
        struct[0].g[20,0] = -P_11/S_base + V_04*V_11*(b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11)) + V_11**2*g_04_11
        struct[0].g[21,0] = -Q_11/S_base + V_04*V_11*(b_04_11*cos(theta_04 - theta_11) + g_04_11*sin(theta_04 - theta_11)) + V_11**2*(-b_04_11 - bs_04_11/2)
        struct[0].g[22,0] = -P_12/S_base + V_03*V_12*(b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12)) + V_12**2*g_03_12
        struct[0].g[23,0] = -Q_12/S_base + V_03*V_12*(b_03_12*cos(theta_03 - theta_12) + g_03_12*sin(theta_03 - theta_12)) + V_12**2*(-b_03_12 - bs_03_12/2)
        struct[0].g[24,0] = K_p_08*(p_m_08 - p_s_08 + xi_p_08/T_p_08) - omega_08
        struct[0].g[25,0] = K_q_08*(-q_s_08 + q_s_ref_08 + xi_q_08/T_q_08) - e_qv_08
        struct[0].g[26,0] = -R_v_08*i_d_08 - V_08*sin(delta_08 - theta_08) + X_v_08*i_q_08
        struct[0].g[27,0] = -R_v_08*i_q_08 - V_08*cos(delta_08 - theta_08) - X_v_08*i_d_08 + e_qv_08
        struct[0].g[28,0] = V_08*i_d_08*sin(delta_08 - theta_08) + V_08*i_q_08*cos(delta_08 - theta_08) - p_s_08
        struct[0].g[29,0] = V_08*i_d_08*cos(delta_08 - theta_08) - V_08*i_q_08*sin(delta_08 - theta_08) - q_s_08
        struct[0].g[30,0] = K_fpfr_08*Piecewise(np.array([(P_f_min_08, P_f_min_08 > p_f_08), (P_f_max_08, P_f_max_08 < p_f_08), (p_f_08, True)])) + p_ghr_08 - p_m_08 + p_s_08 - p_t_08 + p_u_08
        struct[0].g[31,0] = i_d_08*(R_s_08*i_d_08 + V_08*sin(delta_08 - theta_08)) + i_q_08*(R_s_08*i_q_08 + V_08*cos(delta_08 - theta_08)) - p_t_08
        struct[0].g[32,0] = -p_u_08 - k_u_08*(-v_u_08**2 + v_u_ref_08**2)/V_u_max_08**2
        struct[0].g[33,0] = R_uc_08*S_n_08*(p_gou_08 - p_t_08)/(v_u_08 + 0.1) + e_u_08 - v_u_08
        struct[0].g[34,0] = -k_u_08 + Piecewise(np.array([(K_u_max_08, V_u_min_08 > v_u_08), (K_u_0_08 + (-K_u_0_08 + K_u_max_08)*(-V_u_lt_08 + v_u_08)/(-V_u_lt_08 + V_u_min_08), V_u_lt_08 > v_u_08), (K_u_0_08 + (-K_u_0_08 + K_u_max_08)*(-V_u_ht_08 + v_u_08)/(-V_u_ht_08 + V_u_max_08), V_u_ht_08 < v_u_08), (K_u_max_08, V_u_max_08 < v_u_08), (K_u_0_08, True)]))
        struct[0].g[35,0] = -k_cur_sat_08 + Piecewise(np.array([(0.0001, k_cur_08 < 0.0001), (1, k_cur_08 > 1), (k_cur_08, True)]))
        struct[0].g[36,0] = k_cur_sat_08*(inc_p_gin_08 + p_gin_0_08) - p_gou_08
        struct[0].g[37,0] = -p_f_08 - Piecewise(np.array([((0.5*DB_08 + omega_08 - omega_ref_08)/Droop_08, omega_08 < -0.5*DB_08 + omega_ref_08), ((-0.5*DB_08 + omega_08 - omega_ref_08)/Droop_08, omega_08 > 0.5*DB_08 + omega_ref_08), (0.0, True)]))
        struct[0].g[38,0] = -r_lim_08 + Piecewise(np.array([(R_lim_max_08, (omega_08 > 0.5*DB_08 + omega_ref_08) | (omega_08 < -0.5*DB_08 + omega_ref_08)), (0.0, True)])) + Piecewise(np.array([(R_lim_08 + (-R_lim_08 + R_lim_max_08)*(-V_u_lt_08 + v_u_08)/(-V_u_lt_08 + V_u_min_08), V_u_lt_08 > v_u_08), (R_lim_08 + (-R_lim_08 + R_lim_max_08)*(-V_u_ht_08 + v_u_08)/(-V_u_ht_08 + V_u_max_08), V_u_ht_08 < v_u_08), (R_lim_08, True)]))
        struct[0].g[39,0] = K_p_09*(p_m_09 - p_s_09 + xi_p_09/T_p_09) - omega_09
        struct[0].g[40,0] = K_q_09*(-q_s_09 + q_s_ref_09 + xi_q_09/T_q_09) - e_qv_09
        struct[0].g[41,0] = -R_v_09*i_d_09 - V_09*sin(delta_09 - theta_09) + X_v_09*i_q_09
        struct[0].g[42,0] = -R_v_09*i_q_09 - V_09*cos(delta_09 - theta_09) - X_v_09*i_d_09 + e_qv_09
        struct[0].g[43,0] = V_09*i_d_09*sin(delta_09 - theta_09) + V_09*i_q_09*cos(delta_09 - theta_09) - p_s_09
        struct[0].g[44,0] = V_09*i_d_09*cos(delta_09 - theta_09) - V_09*i_q_09*sin(delta_09 - theta_09) - q_s_09
        struct[0].g[45,0] = K_fpfr_09*Piecewise(np.array([(P_f_min_09, P_f_min_09 > p_f_09), (P_f_max_09, P_f_max_09 < p_f_09), (p_f_09, True)])) + p_ghr_09 - p_m_09 + p_s_09 - p_t_09 + p_u_09
        struct[0].g[46,0] = i_d_09*(R_s_09*i_d_09 + V_09*sin(delta_09 - theta_09)) + i_q_09*(R_s_09*i_q_09 + V_09*cos(delta_09 - theta_09)) - p_t_09
        struct[0].g[47,0] = -p_u_09 - k_u_09*(-v_u_09**2 + v_u_ref_09**2)/V_u_max_09**2
        struct[0].g[48,0] = R_uc_09*S_n_09*(p_gou_09 - p_t_09)/(v_u_09 + 0.1) + e_u_09 - v_u_09
        struct[0].g[49,0] = -k_u_09 + Piecewise(np.array([(K_u_max_09, V_u_min_09 > v_u_09), (K_u_0_09 + (-K_u_0_09 + K_u_max_09)*(-V_u_lt_09 + v_u_09)/(-V_u_lt_09 + V_u_min_09), V_u_lt_09 > v_u_09), (K_u_0_09 + (-K_u_0_09 + K_u_max_09)*(-V_u_ht_09 + v_u_09)/(-V_u_ht_09 + V_u_max_09), V_u_ht_09 < v_u_09), (K_u_max_09, V_u_max_09 < v_u_09), (K_u_0_09, True)]))
        struct[0].g[50,0] = -k_cur_sat_09 + Piecewise(np.array([(0.0001, k_cur_09 < 0.0001), (1, k_cur_09 > 1), (k_cur_09, True)]))
        struct[0].g[51,0] = k_cur_sat_09*(inc_p_gin_09 + p_gin_0_09) - p_gou_09
        struct[0].g[52,0] = -p_f_09 - Piecewise(np.array([((0.5*DB_09 + omega_09 - omega_ref_09)/Droop_09, omega_09 < -0.5*DB_09 + omega_ref_09), ((-0.5*DB_09 + omega_09 - omega_ref_09)/Droop_09, omega_09 > 0.5*DB_09 + omega_ref_09), (0.0, True)]))
        struct[0].g[53,0] = -r_lim_09 + Piecewise(np.array([(R_lim_max_09, (omega_09 > 0.5*DB_09 + omega_ref_09) | (omega_09 < -0.5*DB_09 + omega_ref_09)), (0.0, True)])) + Piecewise(np.array([(R_lim_09 + (-R_lim_09 + R_lim_max_09)*(-V_u_lt_09 + v_u_09)/(-V_u_lt_09 + V_u_min_09), V_u_lt_09 > v_u_09), (R_lim_09 + (-R_lim_09 + R_lim_max_09)*(-V_u_ht_09 + v_u_09)/(-V_u_ht_09 + V_u_max_09), V_u_ht_09 < v_u_09), (R_lim_09, True)]))
        struct[0].g[54,0] = K_p_10*(p_m_10 - p_s_10 + xi_p_10/T_p_10) - omega_10
        struct[0].g[55,0] = K_q_10*(-q_s_10 + q_s_ref_10 + xi_q_10/T_q_10) - e_qv_10
        struct[0].g[56,0] = -R_v_10*i_d_10 - V_10*sin(delta_10 - theta_10) + X_v_10*i_q_10
        struct[0].g[57,0] = -R_v_10*i_q_10 - V_10*cos(delta_10 - theta_10) - X_v_10*i_d_10 + e_qv_10
        struct[0].g[58,0] = V_10*i_d_10*sin(delta_10 - theta_10) + V_10*i_q_10*cos(delta_10 - theta_10) - p_s_10
        struct[0].g[59,0] = V_10*i_d_10*cos(delta_10 - theta_10) - V_10*i_q_10*sin(delta_10 - theta_10) - q_s_10
        struct[0].g[60,0] = K_fpfr_10*Piecewise(np.array([(P_f_min_10, P_f_min_10 > p_f_10), (P_f_max_10, P_f_max_10 < p_f_10), (p_f_10, True)])) + p_ghr_10 - p_m_10 + p_s_10 - p_t_10 + p_u_10
        struct[0].g[61,0] = i_d_10*(R_s_10*i_d_10 + V_10*sin(delta_10 - theta_10)) + i_q_10*(R_s_10*i_q_10 + V_10*cos(delta_10 - theta_10)) - p_t_10
        struct[0].g[62,0] = -p_u_10 - k_u_10*(-v_u_10**2 + v_u_ref_10**2)/V_u_max_10**2
        struct[0].g[63,0] = R_uc_10*S_n_10*(p_gou_10 - p_t_10)/(v_u_10 + 0.1) + e_u_10 - v_u_10
        struct[0].g[64,0] = -k_u_10 + Piecewise(np.array([(K_u_max_10, V_u_min_10 > v_u_10), (K_u_0_10 + (-K_u_0_10 + K_u_max_10)*(-V_u_lt_10 + v_u_10)/(-V_u_lt_10 + V_u_min_10), V_u_lt_10 > v_u_10), (K_u_0_10 + (-K_u_0_10 + K_u_max_10)*(-V_u_ht_10 + v_u_10)/(-V_u_ht_10 + V_u_max_10), V_u_ht_10 < v_u_10), (K_u_max_10, V_u_max_10 < v_u_10), (K_u_0_10, True)]))
        struct[0].g[65,0] = -k_cur_sat_10 + Piecewise(np.array([(0.0001, k_cur_10 < 0.0001), (1, k_cur_10 > 1), (k_cur_10, True)]))
        struct[0].g[66,0] = k_cur_sat_10*(inc_p_gin_10 + p_gin_0_10) - p_gou_10
        struct[0].g[67,0] = -p_f_10 - Piecewise(np.array([((0.5*DB_10 + omega_10 - omega_ref_10)/Droop_10, omega_10 < -0.5*DB_10 + omega_ref_10), ((-0.5*DB_10 + omega_10 - omega_ref_10)/Droop_10, omega_10 > 0.5*DB_10 + omega_ref_10), (0.0, True)]))
        struct[0].g[68,0] = -r_lim_10 + Piecewise(np.array([(R_lim_max_10, (omega_10 > 0.5*DB_10 + omega_ref_10) | (omega_10 < -0.5*DB_10 + omega_ref_10)), (0.0, True)])) + Piecewise(np.array([(R_lim_10 + (-R_lim_10 + R_lim_max_10)*(-V_u_lt_10 + v_u_10)/(-V_u_lt_10 + V_u_min_10), V_u_lt_10 > v_u_10), (R_lim_10 + (-R_lim_10 + R_lim_max_10)*(-V_u_ht_10 + v_u_10)/(-V_u_ht_10 + V_u_max_10), V_u_ht_10 < v_u_10), (R_lim_10, True)]))
        struct[0].g[69,0] = Domega_01 - omega_01 + omega_ref_01
        struct[0].g[70,0] = -R_v_01*i_d_01 - V_01*sin(delta_01 - theta_01) + X_v_01*i_q_01
        struct[0].g[71,0] = -R_v_01*i_q_01 - V_01*cos(delta_01 - theta_01) - X_v_01*i_d_01 + e_qv_01
        struct[0].g[72,0] = V_01*i_d_01*sin(delta_01 - theta_01) + V_01*i_q_01*cos(delta_01 - theta_01) - p_s_01
        struct[0].g[73,0] = V_01*i_d_01*cos(delta_01 - theta_01) - V_01*i_q_01*sin(delta_01 - theta_01) - q_s_01
        struct[0].g[74,0] = -omega_coi + (1000000.0*S_n_01*omega_01 + S_n_10*T_p_10*omega_10/(2*K_p_10) + S_n_09*T_p_09*omega_09/(2*K_p_09) + S_n_08*T_p_08*omega_08/(2*K_p_08))/(1000000.0*S_n_01 + S_n_10*T_p_10/(2*K_p_10) + S_n_09*T_p_09/(2*K_p_09) + S_n_08*T_p_08/(2*K_p_08))
        struct[0].g[75,0] = K_i_agc*xi_freq + K_p_agc*(1 - omega_coi) - p_agc
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_01
        struct[0].h[1,0] = V_02
        struct[0].h[2,0] = V_03
        struct[0].h[3,0] = V_04
        struct[0].h[4,0] = V_05
        struct[0].h[5,0] = V_06
        struct[0].h[6,0] = V_07
        struct[0].h[7,0] = V_08
        struct[0].h[8,0] = V_09
        struct[0].h[9,0] = V_10
        struct[0].h[10,0] = V_11
        struct[0].h[11,0] = V_12
        struct[0].h[12,0] = inc_p_gin_08 + p_gin_0_08
        struct[0].h[13,0] = p_g_ref_08
        struct[0].h[14,0] = -p_s_08 + p_t_08
        struct[0].h[15,0] = (-V_u_min_08**2 + e_u_08**2)/(V_u_max_08**2 - V_u_min_08**2)
        struct[0].h[16,0] = K_fpfr_08*Piecewise(np.array([(P_f_min_08, P_f_min_08 > p_f_08), (P_f_max_08, P_f_max_08 < p_f_08), (p_f_08, True)]))
        struct[0].h[17,0] = Piecewise(np.array([(P_f_min_08, P_f_min_08 > p_f_08), (P_f_max_08, P_f_max_08 < p_f_08), (p_f_08, True)]))
        struct[0].h[18,0] = inc_p_gin_09 + p_gin_0_09
        struct[0].h[19,0] = p_g_ref_09
        struct[0].h[20,0] = -p_s_09 + p_t_09
        struct[0].h[21,0] = (-V_u_min_09**2 + e_u_09**2)/(V_u_max_09**2 - V_u_min_09**2)
        struct[0].h[22,0] = K_fpfr_09*Piecewise(np.array([(P_f_min_09, P_f_min_09 > p_f_09), (P_f_max_09, P_f_max_09 < p_f_09), (p_f_09, True)]))
        struct[0].h[23,0] = Piecewise(np.array([(P_f_min_09, P_f_min_09 > p_f_09), (P_f_max_09, P_f_max_09 < p_f_09), (p_f_09, True)]))
        struct[0].h[24,0] = inc_p_gin_10 + p_gin_0_10
        struct[0].h[25,0] = p_g_ref_10
        struct[0].h[26,0] = -p_s_10 + p_t_10
        struct[0].h[27,0] = (-V_u_min_10**2 + e_u_10**2)/(V_u_max_10**2 - V_u_min_10**2)
        struct[0].h[28,0] = K_fpfr_10*Piecewise(np.array([(P_f_min_10, P_f_min_10 > p_f_10), (P_f_max_10, P_f_max_10 < p_f_10), (p_f_10, True)]))
        struct[0].h[29,0] = Piecewise(np.array([(P_f_min_10, P_f_min_10 > p_f_10), (P_f_max_10, P_f_max_10 < p_f_10), (p_f_10, True)]))
        struct[0].h[30,0] = alpha_01
    

    if mode == 10:

        struct[0].Fx_ini[4,4] = Piecewise(np.array([(0, (r_lim_08 < K_h_08*(-p_ghr_08 + p_gou_08)) | (r_lim_08 < -K_h_08*(-p_ghr_08 + p_gou_08))), (-K_h_08, True)]))
        struct[0].Fx_ini[5,5] = -1/T_cur_08
        struct[0].Fx_ini[5,6] = (-p_g_ref_08/(inc_p_gin_08 + p_gin_0_08)**2 - Piecewise(np.array([(P_f_min_08, P_f_min_08 > p_f_08), (P_f_max_08, P_f_max_08 < p_f_08), (p_f_08, True)]))/(inc_p_gin_08 + p_gin_0_08)**2)/T_cur_08
        struct[0].Fx_ini[6,6] = -0.00100000000000000
        struct[0].Fx_ini[11,11] = Piecewise(np.array([(0, (r_lim_09 < K_h_09*(-p_ghr_09 + p_gou_09)) | (r_lim_09 < -K_h_09*(-p_ghr_09 + p_gou_09))), (-K_h_09, True)]))
        struct[0].Fx_ini[12,12] = -1/T_cur_09
        struct[0].Fx_ini[12,13] = (-p_g_ref_09/(inc_p_gin_09 + p_gin_0_09)**2 - Piecewise(np.array([(P_f_min_09, P_f_min_09 > p_f_09), (P_f_max_09, P_f_max_09 < p_f_09), (p_f_09, True)]))/(inc_p_gin_09 + p_gin_0_09)**2)/T_cur_09
        struct[0].Fx_ini[13,13] = -0.00100000000000000
        struct[0].Fx_ini[18,18] = Piecewise(np.array([(0, (r_lim_10 < K_h_10*(-p_ghr_10 + p_gou_10)) | (r_lim_10 < -K_h_10*(-p_ghr_10 + p_gou_10))), (-K_h_10, True)]))
        struct[0].Fx_ini[19,19] = -1/T_cur_10
        struct[0].Fx_ini[19,20] = (-p_g_ref_10/(inc_p_gin_10 + p_gin_0_10)**2 - Piecewise(np.array([(P_f_min_10, P_f_min_10 > p_f_10), (P_f_max_10, P_f_max_10 < p_f_10), (p_f_10, True)]))/(inc_p_gin_10 + p_gin_0_10)**2)/T_cur_10
        struct[0].Fx_ini[20,20] = -0.00100000000000000
        struct[0].Fx_ini[21,21] = -K_delta_01
        struct[0].Fx_ini[22,22] = -K_alpha_01

    if mode == 11:

        struct[0].Fy_ini[0,24] = Omega_b_08 
        struct[0].Fy_ini[0,74] = -Omega_b_08 
        struct[0].Fy_ini[1,28] = -1 
        struct[0].Fy_ini[1,30] = 1 
        struct[0].Fy_ini[2,29] = -1 
        struct[0].Fy_ini[3,31] = -S_n_08/(C_u_08*(v_u_08 + 0.1)) 
        struct[0].Fy_ini[3,33] = -S_n_08*(p_gou_08 - p_t_08)/(C_u_08*(v_u_08 + 0.1)**2) 
        struct[0].Fy_ini[3,36] = S_n_08/(C_u_08*(v_u_08 + 0.1)) 
        struct[0].Fy_ini[4,36] = Piecewise(np.array([(0, (r_lim_08 < K_h_08*(-p_ghr_08 + p_gou_08)) | (r_lim_08 < -K_h_08*(-p_ghr_08 + p_gou_08))), (K_h_08, True)])) 
        struct[0].Fy_ini[4,38] = Piecewise(np.array([(-1, r_lim_08 < -K_h_08*(-p_ghr_08 + p_gou_08)), (1, r_lim_08 < K_h_08*(-p_ghr_08 + p_gou_08)), (0, True)])) 
        struct[0].Fy_ini[5,37] = Piecewise(np.array([(0, (P_f_min_08 > p_f_08) | (P_f_max_08 < p_f_08)), (1, True)]))/(T_cur_08*(inc_p_gin_08 + p_gin_0_08)) 
        struct[0].Fy_ini[7,39] = Omega_b_09 
        struct[0].Fy_ini[7,74] = -Omega_b_09 
        struct[0].Fy_ini[8,43] = -1 
        struct[0].Fy_ini[8,45] = 1 
        struct[0].Fy_ini[9,44] = -1 
        struct[0].Fy_ini[10,46] = -S_n_09/(C_u_09*(v_u_09 + 0.1)) 
        struct[0].Fy_ini[10,48] = -S_n_09*(p_gou_09 - p_t_09)/(C_u_09*(v_u_09 + 0.1)**2) 
        struct[0].Fy_ini[10,51] = S_n_09/(C_u_09*(v_u_09 + 0.1)) 
        struct[0].Fy_ini[11,51] = Piecewise(np.array([(0, (r_lim_09 < K_h_09*(-p_ghr_09 + p_gou_09)) | (r_lim_09 < -K_h_09*(-p_ghr_09 + p_gou_09))), (K_h_09, True)])) 
        struct[0].Fy_ini[11,53] = Piecewise(np.array([(-1, r_lim_09 < -K_h_09*(-p_ghr_09 + p_gou_09)), (1, r_lim_09 < K_h_09*(-p_ghr_09 + p_gou_09)), (0, True)])) 
        struct[0].Fy_ini[12,52] = Piecewise(np.array([(0, (P_f_min_09 > p_f_09) | (P_f_max_09 < p_f_09)), (1, True)]))/(T_cur_09*(inc_p_gin_09 + p_gin_0_09)) 
        struct[0].Fy_ini[14,54] = Omega_b_10 
        struct[0].Fy_ini[14,74] = -Omega_b_10 
        struct[0].Fy_ini[15,58] = -1 
        struct[0].Fy_ini[15,60] = 1 
        struct[0].Fy_ini[16,59] = -1 
        struct[0].Fy_ini[17,61] = -S_n_10/(C_u_10*(v_u_10 + 0.1)) 
        struct[0].Fy_ini[17,63] = -S_n_10*(p_gou_10 - p_t_10)/(C_u_10*(v_u_10 + 0.1)**2) 
        struct[0].Fy_ini[17,66] = S_n_10/(C_u_10*(v_u_10 + 0.1)) 
        struct[0].Fy_ini[18,66] = Piecewise(np.array([(0, (r_lim_10 < K_h_10*(-p_ghr_10 + p_gou_10)) | (r_lim_10 < -K_h_10*(-p_ghr_10 + p_gou_10))), (K_h_10, True)])) 
        struct[0].Fy_ini[18,68] = Piecewise(np.array([(-1, r_lim_10 < -K_h_10*(-p_ghr_10 + p_gou_10)), (1, r_lim_10 < K_h_10*(-p_ghr_10 + p_gou_10)), (0, True)])) 
        struct[0].Fy_ini[19,67] = Piecewise(np.array([(0, (P_f_min_10 > p_f_10) | (P_f_max_10 < p_f_10)), (1, True)]))/(T_cur_10*(inc_p_gin_10 + p_gin_0_10)) 
        struct[0].Fy_ini[21,69] = Omega_b_01 
        struct[0].Fy_ini[21,74] = -Omega_b_01 
        struct[0].Fy_ini[23,74] = -1 

        struct[0].Gy_ini[0,0] = 2*V_01*g_01_02 + V_02*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy_ini[0,1] = V_01*V_02*(-b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy_ini[0,2] = V_01*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy_ini[0,3] = V_01*V_02*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy_ini[0,72] = -S_n_01/S_base
        struct[0].Gy_ini[1,0] = 2*V_01*(-b_01_02 - bs_01_02/2) + V_02*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy_ini[1,1] = V_01*V_02*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy_ini[1,2] = V_01*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy_ini[1,3] = V_01*V_02*(b_01_02*sin(theta_01 - theta_02) + g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy_ini[1,73] = -S_n_01/S_base
        struct[0].Gy_ini[2,0] = V_02*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy_ini[2,1] = V_01*V_02*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy_ini[2,2] = V_01*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) + 2*V_02*(g_01_02 + g_02_03) + V_03*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy_ini[2,3] = V_01*V_02*(-b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02)) + V_02*V_03*(-b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy_ini[2,4] = V_02*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy_ini[2,5] = V_02*V_03*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy_ini[3,0] = V_02*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy_ini[3,1] = V_01*V_02*(-b_01_02*sin(theta_01 - theta_02) + g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy_ini[3,2] = V_01*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02)) + 2*V_02*(-b_01_02 - b_02_03 - bs_01_02/2 - bs_02_03/2) + V_03*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy_ini[3,3] = V_01*V_02*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) + V_02*V_03*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy_ini[3,4] = V_02*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy_ini[3,5] = V_02*V_03*(b_02_03*sin(theta_02 - theta_03) + g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy_ini[4,2] = V_03*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy_ini[4,3] = V_02*V_03*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy_ini[4,4] = V_02*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03)) + 2*V_03*(g_02_03 + g_03_04 + g_03_12) + V_04*(-b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04)) + V_12*(-b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12))
        struct[0].Gy_ini[4,5] = V_02*V_03*(-b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03)) + V_03*V_04*(-b_03_04*cos(theta_03 - theta_04) + g_03_04*sin(theta_03 - theta_04)) + V_03*V_12*(-b_03_12*cos(theta_03 - theta_12) + g_03_12*sin(theta_03 - theta_12))
        struct[0].Gy_ini[4,6] = V_03*(-b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04))
        struct[0].Gy_ini[4,7] = V_03*V_04*(b_03_04*cos(theta_03 - theta_04) - g_03_04*sin(theta_03 - theta_04))
        struct[0].Gy_ini[4,22] = V_03*(-b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12))
        struct[0].Gy_ini[4,23] = V_03*V_12*(b_03_12*cos(theta_03 - theta_12) - g_03_12*sin(theta_03 - theta_12))
        struct[0].Gy_ini[5,2] = V_03*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy_ini[5,3] = V_02*V_03*(-b_02_03*sin(theta_02 - theta_03) + g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy_ini[5,4] = V_02*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03)) + 2*V_03*(-b_02_03 - b_03_04 - b_03_12 - bs_02_03/2 - bs_03_04/2 - bs_03_12/2) + V_04*(b_03_04*cos(theta_03 - theta_04) - g_03_04*sin(theta_03 - theta_04)) + V_12*(b_03_12*cos(theta_03 - theta_12) - g_03_12*sin(theta_03 - theta_12))
        struct[0].Gy_ini[5,5] = V_02*V_03*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03)) + V_03*V_04*(-b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04)) + V_03*V_12*(-b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12))
        struct[0].Gy_ini[5,6] = V_03*(b_03_04*cos(theta_03 - theta_04) - g_03_04*sin(theta_03 - theta_04))
        struct[0].Gy_ini[5,7] = V_03*V_04*(b_03_04*sin(theta_03 - theta_04) + g_03_04*cos(theta_03 - theta_04))
        struct[0].Gy_ini[5,22] = V_03*(b_03_12*cos(theta_03 - theta_12) - g_03_12*sin(theta_03 - theta_12))
        struct[0].Gy_ini[5,23] = V_03*V_12*(b_03_12*sin(theta_03 - theta_12) + g_03_12*cos(theta_03 - theta_12))
        struct[0].Gy_ini[6,4] = V_04*(b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04))
        struct[0].Gy_ini[6,5] = V_03*V_04*(b_03_04*cos(theta_03 - theta_04) + g_03_04*sin(theta_03 - theta_04))
        struct[0].Gy_ini[6,6] = V_03*(b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04)) + 2*V_04*(g_03_04 + g_04_05 + g_04_11) + V_05*(-b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05)) + V_11*(-b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11))
        struct[0].Gy_ini[6,7] = V_03*V_04*(-b_03_04*cos(theta_03 - theta_04) - g_03_04*sin(theta_03 - theta_04)) + V_04*V_05*(-b_04_05*cos(theta_04 - theta_05) + g_04_05*sin(theta_04 - theta_05)) + V_04*V_11*(-b_04_11*cos(theta_04 - theta_11) + g_04_11*sin(theta_04 - theta_11))
        struct[0].Gy_ini[6,8] = V_04*(-b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05))
        struct[0].Gy_ini[6,9] = V_04*V_05*(b_04_05*cos(theta_04 - theta_05) - g_04_05*sin(theta_04 - theta_05))
        struct[0].Gy_ini[6,20] = V_04*(-b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11))
        struct[0].Gy_ini[6,21] = V_04*V_11*(b_04_11*cos(theta_04 - theta_11) - g_04_11*sin(theta_04 - theta_11))
        struct[0].Gy_ini[7,4] = V_04*(b_03_04*cos(theta_03 - theta_04) + g_03_04*sin(theta_03 - theta_04))
        struct[0].Gy_ini[7,5] = V_03*V_04*(-b_03_04*sin(theta_03 - theta_04) + g_03_04*cos(theta_03 - theta_04))
        struct[0].Gy_ini[7,6] = V_03*(b_03_04*cos(theta_03 - theta_04) + g_03_04*sin(theta_03 - theta_04)) + 2*V_04*(-b_03_04 - b_04_05 - b_04_11 - bs_03_04/2 - bs_04_05/2 - bs_04_11/2) + V_05*(b_04_05*cos(theta_04 - theta_05) - g_04_05*sin(theta_04 - theta_05)) + V_11*(b_04_11*cos(theta_04 - theta_11) - g_04_11*sin(theta_04 - theta_11))
        struct[0].Gy_ini[7,7] = V_03*V_04*(b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04)) + V_04*V_05*(-b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05)) + V_04*V_11*(-b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11))
        struct[0].Gy_ini[7,8] = V_04*(b_04_05*cos(theta_04 - theta_05) - g_04_05*sin(theta_04 - theta_05))
        struct[0].Gy_ini[7,9] = V_04*V_05*(b_04_05*sin(theta_04 - theta_05) + g_04_05*cos(theta_04 - theta_05))
        struct[0].Gy_ini[7,20] = V_04*(b_04_11*cos(theta_04 - theta_11) - g_04_11*sin(theta_04 - theta_11))
        struct[0].Gy_ini[7,21] = V_04*V_11*(b_04_11*sin(theta_04 - theta_11) + g_04_11*cos(theta_04 - theta_11))
        struct[0].Gy_ini[8,6] = V_05*(b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05))
        struct[0].Gy_ini[8,7] = V_04*V_05*(b_04_05*cos(theta_04 - theta_05) + g_04_05*sin(theta_04 - theta_05))
        struct[0].Gy_ini[8,8] = V_04*(b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05)) + 2*V_05*(g_04_05 + g_05_06 + g_05_10) + V_06*(-b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06)) + V_10*(-b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10))
        struct[0].Gy_ini[8,9] = V_04*V_05*(-b_04_05*cos(theta_04 - theta_05) - g_04_05*sin(theta_04 - theta_05)) + V_05*V_06*(-b_05_06*cos(theta_05 - theta_06) + g_05_06*sin(theta_05 - theta_06)) + V_05*V_10*(-b_05_10*cos(theta_05 - theta_10) + g_05_10*sin(theta_05 - theta_10))
        struct[0].Gy_ini[8,10] = V_05*(-b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06))
        struct[0].Gy_ini[8,11] = V_05*V_06*(b_05_06*cos(theta_05 - theta_06) - g_05_06*sin(theta_05 - theta_06))
        struct[0].Gy_ini[8,18] = V_05*(-b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10))
        struct[0].Gy_ini[8,19] = V_05*V_10*(b_05_10*cos(theta_05 - theta_10) - g_05_10*sin(theta_05 - theta_10))
        struct[0].Gy_ini[9,6] = V_05*(b_04_05*cos(theta_04 - theta_05) + g_04_05*sin(theta_04 - theta_05))
        struct[0].Gy_ini[9,7] = V_04*V_05*(-b_04_05*sin(theta_04 - theta_05) + g_04_05*cos(theta_04 - theta_05))
        struct[0].Gy_ini[9,8] = V_04*(b_04_05*cos(theta_04 - theta_05) + g_04_05*sin(theta_04 - theta_05)) + 2*V_05*(-b_04_05 - b_05_06 - b_05_10 - bs_04_05/2 - bs_05_06/2 - bs_05_10/2) + V_06*(b_05_06*cos(theta_05 - theta_06) - g_05_06*sin(theta_05 - theta_06)) + V_10*(b_05_10*cos(theta_05 - theta_10) - g_05_10*sin(theta_05 - theta_10))
        struct[0].Gy_ini[9,9] = V_04*V_05*(b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05)) + V_05*V_06*(-b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06)) + V_05*V_10*(-b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10))
        struct[0].Gy_ini[9,10] = V_05*(b_05_06*cos(theta_05 - theta_06) - g_05_06*sin(theta_05 - theta_06))
        struct[0].Gy_ini[9,11] = V_05*V_06*(b_05_06*sin(theta_05 - theta_06) + g_05_06*cos(theta_05 - theta_06))
        struct[0].Gy_ini[9,18] = V_05*(b_05_10*cos(theta_05 - theta_10) - g_05_10*sin(theta_05 - theta_10))
        struct[0].Gy_ini[9,19] = V_05*V_10*(b_05_10*sin(theta_05 - theta_10) + g_05_10*cos(theta_05 - theta_10))
        struct[0].Gy_ini[10,8] = V_06*(b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06))
        struct[0].Gy_ini[10,9] = V_05*V_06*(b_05_06*cos(theta_05 - theta_06) + g_05_06*sin(theta_05 - theta_06))
        struct[0].Gy_ini[10,10] = V_05*(b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06)) + 2*V_06*(g_05_06 + g_06_07 + g_06_09) + V_07*(-b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07)) + V_09*(-b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09))
        struct[0].Gy_ini[10,11] = V_05*V_06*(-b_05_06*cos(theta_05 - theta_06) - g_05_06*sin(theta_05 - theta_06)) + V_06*V_07*(-b_06_07*cos(theta_06 - theta_07) + g_06_07*sin(theta_06 - theta_07)) + V_06*V_09*(-b_06_09*cos(theta_06 - theta_09) + g_06_09*sin(theta_06 - theta_09))
        struct[0].Gy_ini[10,12] = V_06*(-b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07))
        struct[0].Gy_ini[10,13] = V_06*V_07*(b_06_07*cos(theta_06 - theta_07) - g_06_07*sin(theta_06 - theta_07))
        struct[0].Gy_ini[10,16] = V_06*(-b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09))
        struct[0].Gy_ini[10,17] = V_06*V_09*(b_06_09*cos(theta_06 - theta_09) - g_06_09*sin(theta_06 - theta_09))
        struct[0].Gy_ini[11,8] = V_06*(b_05_06*cos(theta_05 - theta_06) + g_05_06*sin(theta_05 - theta_06))
        struct[0].Gy_ini[11,9] = V_05*V_06*(-b_05_06*sin(theta_05 - theta_06) + g_05_06*cos(theta_05 - theta_06))
        struct[0].Gy_ini[11,10] = V_05*(b_05_06*cos(theta_05 - theta_06) + g_05_06*sin(theta_05 - theta_06)) + 2*V_06*(-b_05_06 - b_06_07 - b_06_09 - bs_05_06/2 - bs_06_07/2 - bs_06_09/2) + V_07*(b_06_07*cos(theta_06 - theta_07) - g_06_07*sin(theta_06 - theta_07)) + V_09*(b_06_09*cos(theta_06 - theta_09) - g_06_09*sin(theta_06 - theta_09))
        struct[0].Gy_ini[11,11] = V_05*V_06*(b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06)) + V_06*V_07*(-b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07)) + V_06*V_09*(-b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09))
        struct[0].Gy_ini[11,12] = V_06*(b_06_07*cos(theta_06 - theta_07) - g_06_07*sin(theta_06 - theta_07))
        struct[0].Gy_ini[11,13] = V_06*V_07*(b_06_07*sin(theta_06 - theta_07) + g_06_07*cos(theta_06 - theta_07))
        struct[0].Gy_ini[11,16] = V_06*(b_06_09*cos(theta_06 - theta_09) - g_06_09*sin(theta_06 - theta_09))
        struct[0].Gy_ini[11,17] = V_06*V_09*(b_06_09*sin(theta_06 - theta_09) + g_06_09*cos(theta_06 - theta_09))
        struct[0].Gy_ini[12,10] = V_07*(b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07))
        struct[0].Gy_ini[12,11] = V_06*V_07*(b_06_07*cos(theta_06 - theta_07) + g_06_07*sin(theta_06 - theta_07))
        struct[0].Gy_ini[12,12] = V_06*(b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07)) + 2*V_07*(g_06_07 + g_07_08) + V_08*(-b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08))
        struct[0].Gy_ini[12,13] = V_06*V_07*(-b_06_07*cos(theta_06 - theta_07) - g_06_07*sin(theta_06 - theta_07)) + V_07*V_08*(-b_07_08*cos(theta_07 - theta_08) + g_07_08*sin(theta_07 - theta_08))
        struct[0].Gy_ini[12,14] = V_07*(-b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08))
        struct[0].Gy_ini[12,15] = V_07*V_08*(b_07_08*cos(theta_07 - theta_08) - g_07_08*sin(theta_07 - theta_08))
        struct[0].Gy_ini[13,10] = V_07*(b_06_07*cos(theta_06 - theta_07) + g_06_07*sin(theta_06 - theta_07))
        struct[0].Gy_ini[13,11] = V_06*V_07*(-b_06_07*sin(theta_06 - theta_07) + g_06_07*cos(theta_06 - theta_07))
        struct[0].Gy_ini[13,12] = V_06*(b_06_07*cos(theta_06 - theta_07) + g_06_07*sin(theta_06 - theta_07)) + 2*V_07*(-b_06_07 - b_07_08 - bs_06_07/2 - bs_07_08/2) + V_08*(b_07_08*cos(theta_07 - theta_08) - g_07_08*sin(theta_07 - theta_08))
        struct[0].Gy_ini[13,13] = V_06*V_07*(b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07)) + V_07*V_08*(-b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08))
        struct[0].Gy_ini[13,14] = V_07*(b_07_08*cos(theta_07 - theta_08) - g_07_08*sin(theta_07 - theta_08))
        struct[0].Gy_ini[13,15] = V_07*V_08*(b_07_08*sin(theta_07 - theta_08) + g_07_08*cos(theta_07 - theta_08))
        struct[0].Gy_ini[14,12] = V_08*(b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08))
        struct[0].Gy_ini[14,13] = V_07*V_08*(b_07_08*cos(theta_07 - theta_08) + g_07_08*sin(theta_07 - theta_08))
        struct[0].Gy_ini[14,14] = V_07*(b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08)) + 2*V_08*g_07_08
        struct[0].Gy_ini[14,15] = V_07*V_08*(-b_07_08*cos(theta_07 - theta_08) - g_07_08*sin(theta_07 - theta_08))
        struct[0].Gy_ini[14,28] = -S_n_08/S_base
        struct[0].Gy_ini[15,12] = V_08*(b_07_08*cos(theta_07 - theta_08) + g_07_08*sin(theta_07 - theta_08))
        struct[0].Gy_ini[15,13] = V_07*V_08*(-b_07_08*sin(theta_07 - theta_08) + g_07_08*cos(theta_07 - theta_08))
        struct[0].Gy_ini[15,14] = V_07*(b_07_08*cos(theta_07 - theta_08) + g_07_08*sin(theta_07 - theta_08)) + 2*V_08*(-b_07_08 - bs_07_08/2)
        struct[0].Gy_ini[15,15] = V_07*V_08*(b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08))
        struct[0].Gy_ini[15,29] = -S_n_08/S_base
        struct[0].Gy_ini[16,10] = V_09*(b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09))
        struct[0].Gy_ini[16,11] = V_06*V_09*(b_06_09*cos(theta_06 - theta_09) + g_06_09*sin(theta_06 - theta_09))
        struct[0].Gy_ini[16,16] = V_06*(b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09)) + 2*V_09*g_06_09
        struct[0].Gy_ini[16,17] = V_06*V_09*(-b_06_09*cos(theta_06 - theta_09) - g_06_09*sin(theta_06 - theta_09))
        struct[0].Gy_ini[16,43] = -S_n_09/S_base
        struct[0].Gy_ini[17,10] = V_09*(b_06_09*cos(theta_06 - theta_09) + g_06_09*sin(theta_06 - theta_09))
        struct[0].Gy_ini[17,11] = V_06*V_09*(-b_06_09*sin(theta_06 - theta_09) + g_06_09*cos(theta_06 - theta_09))
        struct[0].Gy_ini[17,16] = V_06*(b_06_09*cos(theta_06 - theta_09) + g_06_09*sin(theta_06 - theta_09)) + 2*V_09*(-b_06_09 - bs_06_09/2)
        struct[0].Gy_ini[17,17] = V_06*V_09*(b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09))
        struct[0].Gy_ini[17,44] = -S_n_09/S_base
        struct[0].Gy_ini[18,8] = V_10*(b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10))
        struct[0].Gy_ini[18,9] = V_05*V_10*(b_05_10*cos(theta_05 - theta_10) + g_05_10*sin(theta_05 - theta_10))
        struct[0].Gy_ini[18,18] = V_05*(b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10)) + 2*V_10*g_05_10
        struct[0].Gy_ini[18,19] = V_05*V_10*(-b_05_10*cos(theta_05 - theta_10) - g_05_10*sin(theta_05 - theta_10))
        struct[0].Gy_ini[18,58] = -S_n_10/S_base
        struct[0].Gy_ini[19,8] = V_10*(b_05_10*cos(theta_05 - theta_10) + g_05_10*sin(theta_05 - theta_10))
        struct[0].Gy_ini[19,9] = V_05*V_10*(-b_05_10*sin(theta_05 - theta_10) + g_05_10*cos(theta_05 - theta_10))
        struct[0].Gy_ini[19,18] = V_05*(b_05_10*cos(theta_05 - theta_10) + g_05_10*sin(theta_05 - theta_10)) + 2*V_10*(-b_05_10 - bs_05_10/2)
        struct[0].Gy_ini[19,19] = V_05*V_10*(b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10))
        struct[0].Gy_ini[19,59] = -S_n_10/S_base
        struct[0].Gy_ini[20,6] = V_11*(b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11))
        struct[0].Gy_ini[20,7] = V_04*V_11*(b_04_11*cos(theta_04 - theta_11) + g_04_11*sin(theta_04 - theta_11))
        struct[0].Gy_ini[20,20] = V_04*(b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11)) + 2*V_11*g_04_11
        struct[0].Gy_ini[20,21] = V_04*V_11*(-b_04_11*cos(theta_04 - theta_11) - g_04_11*sin(theta_04 - theta_11))
        struct[0].Gy_ini[21,6] = V_11*(b_04_11*cos(theta_04 - theta_11) + g_04_11*sin(theta_04 - theta_11))
        struct[0].Gy_ini[21,7] = V_04*V_11*(-b_04_11*sin(theta_04 - theta_11) + g_04_11*cos(theta_04 - theta_11))
        struct[0].Gy_ini[21,20] = V_04*(b_04_11*cos(theta_04 - theta_11) + g_04_11*sin(theta_04 - theta_11)) + 2*V_11*(-b_04_11 - bs_04_11/2)
        struct[0].Gy_ini[21,21] = V_04*V_11*(b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11))
        struct[0].Gy_ini[22,4] = V_12*(b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12))
        struct[0].Gy_ini[22,5] = V_03*V_12*(b_03_12*cos(theta_03 - theta_12) + g_03_12*sin(theta_03 - theta_12))
        struct[0].Gy_ini[22,22] = V_03*(b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12)) + 2*V_12*g_03_12
        struct[0].Gy_ini[22,23] = V_03*V_12*(-b_03_12*cos(theta_03 - theta_12) - g_03_12*sin(theta_03 - theta_12))
        struct[0].Gy_ini[23,4] = V_12*(b_03_12*cos(theta_03 - theta_12) + g_03_12*sin(theta_03 - theta_12))
        struct[0].Gy_ini[23,5] = V_03*V_12*(-b_03_12*sin(theta_03 - theta_12) + g_03_12*cos(theta_03 - theta_12))
        struct[0].Gy_ini[23,22] = V_03*(b_03_12*cos(theta_03 - theta_12) + g_03_12*sin(theta_03 - theta_12)) + 2*V_12*(-b_03_12 - bs_03_12/2)
        struct[0].Gy_ini[23,23] = V_03*V_12*(b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12))
        struct[0].Gy_ini[24,24] = -1
        struct[0].Gy_ini[24,28] = -K_p_08
        struct[0].Gy_ini[24,30] = K_p_08
        struct[0].Gy_ini[25,25] = -1
        struct[0].Gy_ini[25,29] = -K_q_08
        struct[0].Gy_ini[26,14] = -sin(delta_08 - theta_08)
        struct[0].Gy_ini[26,15] = V_08*cos(delta_08 - theta_08)
        struct[0].Gy_ini[26,26] = -R_v_08
        struct[0].Gy_ini[26,27] = X_v_08
        struct[0].Gy_ini[27,14] = -cos(delta_08 - theta_08)
        struct[0].Gy_ini[27,15] = -V_08*sin(delta_08 - theta_08)
        struct[0].Gy_ini[27,25] = 1
        struct[0].Gy_ini[27,26] = -X_v_08
        struct[0].Gy_ini[27,27] = -R_v_08
        struct[0].Gy_ini[28,14] = i_d_08*sin(delta_08 - theta_08) + i_q_08*cos(delta_08 - theta_08)
        struct[0].Gy_ini[28,15] = -V_08*i_d_08*cos(delta_08 - theta_08) + V_08*i_q_08*sin(delta_08 - theta_08)
        struct[0].Gy_ini[28,26] = V_08*sin(delta_08 - theta_08)
        struct[0].Gy_ini[28,27] = V_08*cos(delta_08 - theta_08)
        struct[0].Gy_ini[28,28] = -1
        struct[0].Gy_ini[29,14] = i_d_08*cos(delta_08 - theta_08) - i_q_08*sin(delta_08 - theta_08)
        struct[0].Gy_ini[29,15] = V_08*i_d_08*sin(delta_08 - theta_08) + V_08*i_q_08*cos(delta_08 - theta_08)
        struct[0].Gy_ini[29,26] = V_08*cos(delta_08 - theta_08)
        struct[0].Gy_ini[29,27] = -V_08*sin(delta_08 - theta_08)
        struct[0].Gy_ini[29,29] = -1
        struct[0].Gy_ini[30,28] = 1
        struct[0].Gy_ini[30,30] = -1
        struct[0].Gy_ini[30,31] = -1
        struct[0].Gy_ini[30,32] = 1
        struct[0].Gy_ini[30,37] = K_fpfr_08*Piecewise(np.array([(0, (P_f_min_08 > p_f_08) | (P_f_max_08 < p_f_08)), (1, True)]))
        struct[0].Gy_ini[31,14] = i_d_08*sin(delta_08 - theta_08) + i_q_08*cos(delta_08 - theta_08)
        struct[0].Gy_ini[31,15] = -V_08*i_d_08*cos(delta_08 - theta_08) + V_08*i_q_08*sin(delta_08 - theta_08)
        struct[0].Gy_ini[31,26] = 2*R_s_08*i_d_08 + V_08*sin(delta_08 - theta_08)
        struct[0].Gy_ini[31,27] = 2*R_s_08*i_q_08 + V_08*cos(delta_08 - theta_08)
        struct[0].Gy_ini[31,31] = -1
        struct[0].Gy_ini[32,32] = -1
        struct[0].Gy_ini[32,33] = 2*k_u_08*v_u_08/V_u_max_08**2
        struct[0].Gy_ini[32,34] = -(-v_u_08**2 + v_u_ref_08**2)/V_u_max_08**2
        struct[0].Gy_ini[33,31] = -R_uc_08*S_n_08/(v_u_08 + 0.1)
        struct[0].Gy_ini[33,33] = -R_uc_08*S_n_08*(p_gou_08 - p_t_08)/(v_u_08 + 0.1)**2 - 1
        struct[0].Gy_ini[33,36] = R_uc_08*S_n_08/(v_u_08 + 0.1)
        struct[0].Gy_ini[34,33] = Piecewise(np.array([(0, V_u_min_08 > v_u_08), ((-K_u_0_08 + K_u_max_08)/(-V_u_lt_08 + V_u_min_08), V_u_lt_08 > v_u_08), ((-K_u_0_08 + K_u_max_08)/(-V_u_ht_08 + V_u_max_08), V_u_ht_08 < v_u_08), (0, True)]))
        struct[0].Gy_ini[34,34] = -1
        struct[0].Gy_ini[35,35] = -1
        struct[0].Gy_ini[36,35] = inc_p_gin_08 + p_gin_0_08
        struct[0].Gy_ini[36,36] = -1
        struct[0].Gy_ini[37,24] = -Piecewise(np.array([(1/Droop_08, (omega_08 > 0.5*DB_08 + omega_ref_08) | (omega_08 < -0.5*DB_08 + omega_ref_08)), (0, True)]))
        struct[0].Gy_ini[37,37] = -1
        struct[0].Gy_ini[38,33] = Piecewise(np.array([((-R_lim_08 + R_lim_max_08)/(-V_u_lt_08 + V_u_min_08), V_u_lt_08 > v_u_08), ((-R_lim_08 + R_lim_max_08)/(-V_u_ht_08 + V_u_max_08), V_u_ht_08 < v_u_08), (0, True)]))
        struct[0].Gy_ini[38,38] = -1
        struct[0].Gy_ini[39,39] = -1
        struct[0].Gy_ini[39,43] = -K_p_09
        struct[0].Gy_ini[39,45] = K_p_09
        struct[0].Gy_ini[40,40] = -1
        struct[0].Gy_ini[40,44] = -K_q_09
        struct[0].Gy_ini[41,16] = -sin(delta_09 - theta_09)
        struct[0].Gy_ini[41,17] = V_09*cos(delta_09 - theta_09)
        struct[0].Gy_ini[41,41] = -R_v_09
        struct[0].Gy_ini[41,42] = X_v_09
        struct[0].Gy_ini[42,16] = -cos(delta_09 - theta_09)
        struct[0].Gy_ini[42,17] = -V_09*sin(delta_09 - theta_09)
        struct[0].Gy_ini[42,40] = 1
        struct[0].Gy_ini[42,41] = -X_v_09
        struct[0].Gy_ini[42,42] = -R_v_09
        struct[0].Gy_ini[43,16] = i_d_09*sin(delta_09 - theta_09) + i_q_09*cos(delta_09 - theta_09)
        struct[0].Gy_ini[43,17] = -V_09*i_d_09*cos(delta_09 - theta_09) + V_09*i_q_09*sin(delta_09 - theta_09)
        struct[0].Gy_ini[43,41] = V_09*sin(delta_09 - theta_09)
        struct[0].Gy_ini[43,42] = V_09*cos(delta_09 - theta_09)
        struct[0].Gy_ini[43,43] = -1
        struct[0].Gy_ini[44,16] = i_d_09*cos(delta_09 - theta_09) - i_q_09*sin(delta_09 - theta_09)
        struct[0].Gy_ini[44,17] = V_09*i_d_09*sin(delta_09 - theta_09) + V_09*i_q_09*cos(delta_09 - theta_09)
        struct[0].Gy_ini[44,41] = V_09*cos(delta_09 - theta_09)
        struct[0].Gy_ini[44,42] = -V_09*sin(delta_09 - theta_09)
        struct[0].Gy_ini[44,44] = -1
        struct[0].Gy_ini[45,43] = 1
        struct[0].Gy_ini[45,45] = -1
        struct[0].Gy_ini[45,46] = -1
        struct[0].Gy_ini[45,47] = 1
        struct[0].Gy_ini[45,52] = K_fpfr_09*Piecewise(np.array([(0, (P_f_min_09 > p_f_09) | (P_f_max_09 < p_f_09)), (1, True)]))
        struct[0].Gy_ini[46,16] = i_d_09*sin(delta_09 - theta_09) + i_q_09*cos(delta_09 - theta_09)
        struct[0].Gy_ini[46,17] = -V_09*i_d_09*cos(delta_09 - theta_09) + V_09*i_q_09*sin(delta_09 - theta_09)
        struct[0].Gy_ini[46,41] = 2*R_s_09*i_d_09 + V_09*sin(delta_09 - theta_09)
        struct[0].Gy_ini[46,42] = 2*R_s_09*i_q_09 + V_09*cos(delta_09 - theta_09)
        struct[0].Gy_ini[46,46] = -1
        struct[0].Gy_ini[47,47] = -1
        struct[0].Gy_ini[47,48] = 2*k_u_09*v_u_09/V_u_max_09**2
        struct[0].Gy_ini[47,49] = -(-v_u_09**2 + v_u_ref_09**2)/V_u_max_09**2
        struct[0].Gy_ini[48,46] = -R_uc_09*S_n_09/(v_u_09 + 0.1)
        struct[0].Gy_ini[48,48] = -R_uc_09*S_n_09*(p_gou_09 - p_t_09)/(v_u_09 + 0.1)**2 - 1
        struct[0].Gy_ini[48,51] = R_uc_09*S_n_09/(v_u_09 + 0.1)
        struct[0].Gy_ini[49,48] = Piecewise(np.array([(0, V_u_min_09 > v_u_09), ((-K_u_0_09 + K_u_max_09)/(-V_u_lt_09 + V_u_min_09), V_u_lt_09 > v_u_09), ((-K_u_0_09 + K_u_max_09)/(-V_u_ht_09 + V_u_max_09), V_u_ht_09 < v_u_09), (0, True)]))
        struct[0].Gy_ini[49,49] = -1
        struct[0].Gy_ini[50,50] = -1
        struct[0].Gy_ini[51,50] = inc_p_gin_09 + p_gin_0_09
        struct[0].Gy_ini[51,51] = -1
        struct[0].Gy_ini[52,39] = -Piecewise(np.array([(1/Droop_09, (omega_09 > 0.5*DB_09 + omega_ref_09) | (omega_09 < -0.5*DB_09 + omega_ref_09)), (0, True)]))
        struct[0].Gy_ini[52,52] = -1
        struct[0].Gy_ini[53,48] = Piecewise(np.array([((-R_lim_09 + R_lim_max_09)/(-V_u_lt_09 + V_u_min_09), V_u_lt_09 > v_u_09), ((-R_lim_09 + R_lim_max_09)/(-V_u_ht_09 + V_u_max_09), V_u_ht_09 < v_u_09), (0, True)]))
        struct[0].Gy_ini[53,53] = -1
        struct[0].Gy_ini[54,54] = -1
        struct[0].Gy_ini[54,58] = -K_p_10
        struct[0].Gy_ini[54,60] = K_p_10
        struct[0].Gy_ini[55,55] = -1
        struct[0].Gy_ini[55,59] = -K_q_10
        struct[0].Gy_ini[56,18] = -sin(delta_10 - theta_10)
        struct[0].Gy_ini[56,19] = V_10*cos(delta_10 - theta_10)
        struct[0].Gy_ini[56,56] = -R_v_10
        struct[0].Gy_ini[56,57] = X_v_10
        struct[0].Gy_ini[57,18] = -cos(delta_10 - theta_10)
        struct[0].Gy_ini[57,19] = -V_10*sin(delta_10 - theta_10)
        struct[0].Gy_ini[57,55] = 1
        struct[0].Gy_ini[57,56] = -X_v_10
        struct[0].Gy_ini[57,57] = -R_v_10
        struct[0].Gy_ini[58,18] = i_d_10*sin(delta_10 - theta_10) + i_q_10*cos(delta_10 - theta_10)
        struct[0].Gy_ini[58,19] = -V_10*i_d_10*cos(delta_10 - theta_10) + V_10*i_q_10*sin(delta_10 - theta_10)
        struct[0].Gy_ini[58,56] = V_10*sin(delta_10 - theta_10)
        struct[0].Gy_ini[58,57] = V_10*cos(delta_10 - theta_10)
        struct[0].Gy_ini[58,58] = -1
        struct[0].Gy_ini[59,18] = i_d_10*cos(delta_10 - theta_10) - i_q_10*sin(delta_10 - theta_10)
        struct[0].Gy_ini[59,19] = V_10*i_d_10*sin(delta_10 - theta_10) + V_10*i_q_10*cos(delta_10 - theta_10)
        struct[0].Gy_ini[59,56] = V_10*cos(delta_10 - theta_10)
        struct[0].Gy_ini[59,57] = -V_10*sin(delta_10 - theta_10)
        struct[0].Gy_ini[59,59] = -1
        struct[0].Gy_ini[60,58] = 1
        struct[0].Gy_ini[60,60] = -1
        struct[0].Gy_ini[60,61] = -1
        struct[0].Gy_ini[60,62] = 1
        struct[0].Gy_ini[60,67] = K_fpfr_10*Piecewise(np.array([(0, (P_f_min_10 > p_f_10) | (P_f_max_10 < p_f_10)), (1, True)]))
        struct[0].Gy_ini[61,18] = i_d_10*sin(delta_10 - theta_10) + i_q_10*cos(delta_10 - theta_10)
        struct[0].Gy_ini[61,19] = -V_10*i_d_10*cos(delta_10 - theta_10) + V_10*i_q_10*sin(delta_10 - theta_10)
        struct[0].Gy_ini[61,56] = 2*R_s_10*i_d_10 + V_10*sin(delta_10 - theta_10)
        struct[0].Gy_ini[61,57] = 2*R_s_10*i_q_10 + V_10*cos(delta_10 - theta_10)
        struct[0].Gy_ini[61,61] = -1
        struct[0].Gy_ini[62,62] = -1
        struct[0].Gy_ini[62,63] = 2*k_u_10*v_u_10/V_u_max_10**2
        struct[0].Gy_ini[62,64] = -(-v_u_10**2 + v_u_ref_10**2)/V_u_max_10**2
        struct[0].Gy_ini[63,61] = -R_uc_10*S_n_10/(v_u_10 + 0.1)
        struct[0].Gy_ini[63,63] = -R_uc_10*S_n_10*(p_gou_10 - p_t_10)/(v_u_10 + 0.1)**2 - 1
        struct[0].Gy_ini[63,66] = R_uc_10*S_n_10/(v_u_10 + 0.1)
        struct[0].Gy_ini[64,63] = Piecewise(np.array([(0, V_u_min_10 > v_u_10), ((-K_u_0_10 + K_u_max_10)/(-V_u_lt_10 + V_u_min_10), V_u_lt_10 > v_u_10), ((-K_u_0_10 + K_u_max_10)/(-V_u_ht_10 + V_u_max_10), V_u_ht_10 < v_u_10), (0, True)]))
        struct[0].Gy_ini[64,64] = -1
        struct[0].Gy_ini[65,65] = -1
        struct[0].Gy_ini[66,65] = inc_p_gin_10 + p_gin_0_10
        struct[0].Gy_ini[66,66] = -1
        struct[0].Gy_ini[67,54] = -Piecewise(np.array([(1/Droop_10, (omega_10 > 0.5*DB_10 + omega_ref_10) | (omega_10 < -0.5*DB_10 + omega_ref_10)), (0, True)]))
        struct[0].Gy_ini[67,67] = -1
        struct[0].Gy_ini[68,63] = Piecewise(np.array([((-R_lim_10 + R_lim_max_10)/(-V_u_lt_10 + V_u_min_10), V_u_lt_10 > v_u_10), ((-R_lim_10 + R_lim_max_10)/(-V_u_ht_10 + V_u_max_10), V_u_ht_10 < v_u_10), (0, True)]))
        struct[0].Gy_ini[68,68] = -1
        struct[0].Gy_ini[69,69] = -1
        struct[0].Gy_ini[70,0] = -sin(delta_01 - theta_01)
        struct[0].Gy_ini[70,1] = V_01*cos(delta_01 - theta_01)
        struct[0].Gy_ini[70,70] = -R_v_01
        struct[0].Gy_ini[70,71] = X_v_01
        struct[0].Gy_ini[71,0] = -cos(delta_01 - theta_01)
        struct[0].Gy_ini[71,1] = -V_01*sin(delta_01 - theta_01)
        struct[0].Gy_ini[71,70] = -X_v_01
        struct[0].Gy_ini[71,71] = -R_v_01
        struct[0].Gy_ini[72,0] = i_d_01*sin(delta_01 - theta_01) + i_q_01*cos(delta_01 - theta_01)
        struct[0].Gy_ini[72,1] = -V_01*i_d_01*cos(delta_01 - theta_01) + V_01*i_q_01*sin(delta_01 - theta_01)
        struct[0].Gy_ini[72,70] = V_01*sin(delta_01 - theta_01)
        struct[0].Gy_ini[72,71] = V_01*cos(delta_01 - theta_01)
        struct[0].Gy_ini[72,72] = -1
        struct[0].Gy_ini[73,0] = i_d_01*cos(delta_01 - theta_01) - i_q_01*sin(delta_01 - theta_01)
        struct[0].Gy_ini[73,1] = V_01*i_d_01*sin(delta_01 - theta_01) + V_01*i_q_01*cos(delta_01 - theta_01)
        struct[0].Gy_ini[73,70] = V_01*cos(delta_01 - theta_01)
        struct[0].Gy_ini[73,71] = -V_01*sin(delta_01 - theta_01)
        struct[0].Gy_ini[73,73] = -1
        struct[0].Gy_ini[74,24] = S_n_08*T_p_08/(2*K_p_08*(1000000.0*S_n_01 + S_n_10*T_p_10/(2*K_p_10) + S_n_09*T_p_09/(2*K_p_09) + S_n_08*T_p_08/(2*K_p_08)))
        struct[0].Gy_ini[74,39] = S_n_09*T_p_09/(2*K_p_09*(1000000.0*S_n_01 + S_n_10*T_p_10/(2*K_p_10) + S_n_09*T_p_09/(2*K_p_09) + S_n_08*T_p_08/(2*K_p_08)))
        struct[0].Gy_ini[74,54] = S_n_10*T_p_10/(2*K_p_10*(1000000.0*S_n_01 + S_n_10*T_p_10/(2*K_p_10) + S_n_09*T_p_09/(2*K_p_09) + S_n_08*T_p_08/(2*K_p_08)))
        struct[0].Gy_ini[74,69] = 1000000.0*S_n_01/(1000000.0*S_n_01 + S_n_10*T_p_10/(2*K_p_10) + S_n_09*T_p_09/(2*K_p_09) + S_n_08*T_p_08/(2*K_p_08))
        struct[0].Gy_ini[74,74] = -1
        struct[0].Gy_ini[75,74] = -K_p_agc
        struct[0].Gy_ini[75,75] = -1



def run_nn(t,struct,mode):

    # Parameters:
    S_base = struct[0].S_base
    g_01_02 = struct[0].g_01_02
    b_01_02 = struct[0].b_01_02
    bs_01_02 = struct[0].bs_01_02
    g_02_03 = struct[0].g_02_03
    b_02_03 = struct[0].b_02_03
    bs_02_03 = struct[0].bs_02_03
    g_03_04 = struct[0].g_03_04
    b_03_04 = struct[0].b_03_04
    bs_03_04 = struct[0].bs_03_04
    g_04_05 = struct[0].g_04_05
    b_04_05 = struct[0].b_04_05
    bs_04_05 = struct[0].bs_04_05
    g_05_06 = struct[0].g_05_06
    b_05_06 = struct[0].b_05_06
    bs_05_06 = struct[0].bs_05_06
    g_06_07 = struct[0].g_06_07
    b_06_07 = struct[0].b_06_07
    bs_06_07 = struct[0].bs_06_07
    g_07_08 = struct[0].g_07_08
    b_07_08 = struct[0].b_07_08
    bs_07_08 = struct[0].bs_07_08
    g_06_09 = struct[0].g_06_09
    b_06_09 = struct[0].b_06_09
    bs_06_09 = struct[0].bs_06_09
    g_05_10 = struct[0].g_05_10
    b_05_10 = struct[0].b_05_10
    bs_05_10 = struct[0].bs_05_10
    g_04_11 = struct[0].g_04_11
    b_04_11 = struct[0].b_04_11
    bs_04_11 = struct[0].bs_04_11
    g_03_12 = struct[0].g_03_12
    b_03_12 = struct[0].b_03_12
    bs_03_12 = struct[0].bs_03_12
    U_01_n = struct[0].U_01_n
    U_02_n = struct[0].U_02_n
    U_03_n = struct[0].U_03_n
    U_04_n = struct[0].U_04_n
    U_05_n = struct[0].U_05_n
    U_06_n = struct[0].U_06_n
    U_07_n = struct[0].U_07_n
    U_08_n = struct[0].U_08_n
    U_09_n = struct[0].U_09_n
    U_10_n = struct[0].U_10_n
    U_11_n = struct[0].U_11_n
    U_12_n = struct[0].U_12_n
    S_n_08 = struct[0].S_n_08
    Omega_b_08 = struct[0].Omega_b_08
    K_p_08 = struct[0].K_p_08
    T_p_08 = struct[0].T_p_08
    K_q_08 = struct[0].K_q_08
    T_q_08 = struct[0].T_q_08
    X_v_08 = struct[0].X_v_08
    R_v_08 = struct[0].R_v_08
    R_s_08 = struct[0].R_s_08
    C_u_08 = struct[0].C_u_08
    K_u_0_08 = struct[0].K_u_0_08
    K_u_max_08 = struct[0].K_u_max_08
    V_u_min_08 = struct[0].V_u_min_08
    V_u_max_08 = struct[0].V_u_max_08
    R_uc_08 = struct[0].R_uc_08
    K_h_08 = struct[0].K_h_08
    R_lim_08 = struct[0].R_lim_08
    V_u_lt_08 = struct[0].V_u_lt_08
    V_u_ht_08 = struct[0].V_u_ht_08
    Droop_08 = struct[0].Droop_08
    DB_08 = struct[0].DB_08
    T_cur_08 = struct[0].T_cur_08
    R_lim_max_08 = struct[0].R_lim_max_08
    K_fpfr_08 = struct[0].K_fpfr_08
    P_f_min_08 = struct[0].P_f_min_08
    P_f_max_08 = struct[0].P_f_max_08
    S_n_09 = struct[0].S_n_09
    Omega_b_09 = struct[0].Omega_b_09
    K_p_09 = struct[0].K_p_09
    T_p_09 = struct[0].T_p_09
    K_q_09 = struct[0].K_q_09
    T_q_09 = struct[0].T_q_09
    X_v_09 = struct[0].X_v_09
    R_v_09 = struct[0].R_v_09
    R_s_09 = struct[0].R_s_09
    C_u_09 = struct[0].C_u_09
    K_u_0_09 = struct[0].K_u_0_09
    K_u_max_09 = struct[0].K_u_max_09
    V_u_min_09 = struct[0].V_u_min_09
    V_u_max_09 = struct[0].V_u_max_09
    R_uc_09 = struct[0].R_uc_09
    K_h_09 = struct[0].K_h_09
    R_lim_09 = struct[0].R_lim_09
    V_u_lt_09 = struct[0].V_u_lt_09
    V_u_ht_09 = struct[0].V_u_ht_09
    Droop_09 = struct[0].Droop_09
    DB_09 = struct[0].DB_09
    T_cur_09 = struct[0].T_cur_09
    R_lim_max_09 = struct[0].R_lim_max_09
    K_fpfr_09 = struct[0].K_fpfr_09
    P_f_min_09 = struct[0].P_f_min_09
    P_f_max_09 = struct[0].P_f_max_09
    S_n_10 = struct[0].S_n_10
    Omega_b_10 = struct[0].Omega_b_10
    K_p_10 = struct[0].K_p_10
    T_p_10 = struct[0].T_p_10
    K_q_10 = struct[0].K_q_10
    T_q_10 = struct[0].T_q_10
    X_v_10 = struct[0].X_v_10
    R_v_10 = struct[0].R_v_10
    R_s_10 = struct[0].R_s_10
    C_u_10 = struct[0].C_u_10
    K_u_0_10 = struct[0].K_u_0_10
    K_u_max_10 = struct[0].K_u_max_10
    V_u_min_10 = struct[0].V_u_min_10
    V_u_max_10 = struct[0].V_u_max_10
    R_uc_10 = struct[0].R_uc_10
    K_h_10 = struct[0].K_h_10
    R_lim_10 = struct[0].R_lim_10
    V_u_lt_10 = struct[0].V_u_lt_10
    V_u_ht_10 = struct[0].V_u_ht_10
    Droop_10 = struct[0].Droop_10
    DB_10 = struct[0].DB_10
    T_cur_10 = struct[0].T_cur_10
    R_lim_max_10 = struct[0].R_lim_max_10
    K_fpfr_10 = struct[0].K_fpfr_10
    P_f_min_10 = struct[0].P_f_min_10
    P_f_max_10 = struct[0].P_f_max_10
    S_n_01 = struct[0].S_n_01
    Omega_b_01 = struct[0].Omega_b_01
    X_v_01 = struct[0].X_v_01
    R_v_01 = struct[0].R_v_01
    K_delta_01 = struct[0].K_delta_01
    K_alpha_01 = struct[0].K_alpha_01
    K_p_agc = struct[0].K_p_agc
    K_i_agc = struct[0].K_i_agc
    
    # Inputs:
    P_01 = struct[0].P_01
    Q_01 = struct[0].Q_01
    P_02 = struct[0].P_02
    Q_02 = struct[0].Q_02
    P_03 = struct[0].P_03
    Q_03 = struct[0].Q_03
    P_04 = struct[0].P_04
    Q_04 = struct[0].Q_04
    P_05 = struct[0].P_05
    Q_05 = struct[0].Q_05
    P_06 = struct[0].P_06
    Q_06 = struct[0].Q_06
    P_07 = struct[0].P_07
    Q_07 = struct[0].Q_07
    P_08 = struct[0].P_08
    Q_08 = struct[0].Q_08
    P_09 = struct[0].P_09
    Q_09 = struct[0].Q_09
    P_10 = struct[0].P_10
    Q_10 = struct[0].Q_10
    P_11 = struct[0].P_11
    Q_11 = struct[0].Q_11
    P_12 = struct[0].P_12
    Q_12 = struct[0].Q_12
    q_s_ref_08 = struct[0].q_s_ref_08
    v_u_ref_08 = struct[0].v_u_ref_08
    omega_ref_08 = struct[0].omega_ref_08
    p_gin_0_08 = struct[0].p_gin_0_08
    p_g_ref_08 = struct[0].p_g_ref_08
    ramp_p_gin_08 = struct[0].ramp_p_gin_08
    q_s_ref_09 = struct[0].q_s_ref_09
    v_u_ref_09 = struct[0].v_u_ref_09
    omega_ref_09 = struct[0].omega_ref_09
    p_gin_0_09 = struct[0].p_gin_0_09
    p_g_ref_09 = struct[0].p_g_ref_09
    ramp_p_gin_09 = struct[0].ramp_p_gin_09
    q_s_ref_10 = struct[0].q_s_ref_10
    v_u_ref_10 = struct[0].v_u_ref_10
    omega_ref_10 = struct[0].omega_ref_10
    p_gin_0_10 = struct[0].p_gin_0_10
    p_g_ref_10 = struct[0].p_g_ref_10
    ramp_p_gin_10 = struct[0].ramp_p_gin_10
    alpha_01 = struct[0].alpha_01
    e_qv_01 = struct[0].e_qv_01
    omega_ref_01 = struct[0].omega_ref_01
    
    # Dynamical states:
    delta_08 = struct[0].x[0,0]
    xi_p_08 = struct[0].x[1,0]
    xi_q_08 = struct[0].x[2,0]
    e_u_08 = struct[0].x[3,0]
    p_ghr_08 = struct[0].x[4,0]
    k_cur_08 = struct[0].x[5,0]
    inc_p_gin_08 = struct[0].x[6,0]
    delta_09 = struct[0].x[7,0]
    xi_p_09 = struct[0].x[8,0]
    xi_q_09 = struct[0].x[9,0]
    e_u_09 = struct[0].x[10,0]
    p_ghr_09 = struct[0].x[11,0]
    k_cur_09 = struct[0].x[12,0]
    inc_p_gin_09 = struct[0].x[13,0]
    delta_10 = struct[0].x[14,0]
    xi_p_10 = struct[0].x[15,0]
    xi_q_10 = struct[0].x[16,0]
    e_u_10 = struct[0].x[17,0]
    p_ghr_10 = struct[0].x[18,0]
    k_cur_10 = struct[0].x[19,0]
    inc_p_gin_10 = struct[0].x[20,0]
    delta_01 = struct[0].x[21,0]
    Domega_01 = struct[0].x[22,0]
    xi_freq = struct[0].x[23,0]
    
    # Algebraic states:
    V_01 = struct[0].y_run[0,0]
    theta_01 = struct[0].y_run[1,0]
    V_02 = struct[0].y_run[2,0]
    theta_02 = struct[0].y_run[3,0]
    V_03 = struct[0].y_run[4,0]
    theta_03 = struct[0].y_run[5,0]
    V_04 = struct[0].y_run[6,0]
    theta_04 = struct[0].y_run[7,0]
    V_05 = struct[0].y_run[8,0]
    theta_05 = struct[0].y_run[9,0]
    V_06 = struct[0].y_run[10,0]
    theta_06 = struct[0].y_run[11,0]
    V_07 = struct[0].y_run[12,0]
    theta_07 = struct[0].y_run[13,0]
    V_08 = struct[0].y_run[14,0]
    theta_08 = struct[0].y_run[15,0]
    V_09 = struct[0].y_run[16,0]
    theta_09 = struct[0].y_run[17,0]
    V_10 = struct[0].y_run[18,0]
    theta_10 = struct[0].y_run[19,0]
    V_11 = struct[0].y_run[20,0]
    theta_11 = struct[0].y_run[21,0]
    V_12 = struct[0].y_run[22,0]
    theta_12 = struct[0].y_run[23,0]
    omega_08 = struct[0].y_run[24,0]
    e_qv_08 = struct[0].y_run[25,0]
    i_d_08 = struct[0].y_run[26,0]
    i_q_08 = struct[0].y_run[27,0]
    p_s_08 = struct[0].y_run[28,0]
    q_s_08 = struct[0].y_run[29,0]
    p_m_08 = struct[0].y_run[30,0]
    p_t_08 = struct[0].y_run[31,0]
    p_u_08 = struct[0].y_run[32,0]
    v_u_08 = struct[0].y_run[33,0]
    k_u_08 = struct[0].y_run[34,0]
    k_cur_sat_08 = struct[0].y_run[35,0]
    p_gou_08 = struct[0].y_run[36,0]
    p_f_08 = struct[0].y_run[37,0]
    r_lim_08 = struct[0].y_run[38,0]
    omega_09 = struct[0].y_run[39,0]
    e_qv_09 = struct[0].y_run[40,0]
    i_d_09 = struct[0].y_run[41,0]
    i_q_09 = struct[0].y_run[42,0]
    p_s_09 = struct[0].y_run[43,0]
    q_s_09 = struct[0].y_run[44,0]
    p_m_09 = struct[0].y_run[45,0]
    p_t_09 = struct[0].y_run[46,0]
    p_u_09 = struct[0].y_run[47,0]
    v_u_09 = struct[0].y_run[48,0]
    k_u_09 = struct[0].y_run[49,0]
    k_cur_sat_09 = struct[0].y_run[50,0]
    p_gou_09 = struct[0].y_run[51,0]
    p_f_09 = struct[0].y_run[52,0]
    r_lim_09 = struct[0].y_run[53,0]
    omega_10 = struct[0].y_run[54,0]
    e_qv_10 = struct[0].y_run[55,0]
    i_d_10 = struct[0].y_run[56,0]
    i_q_10 = struct[0].y_run[57,0]
    p_s_10 = struct[0].y_run[58,0]
    q_s_10 = struct[0].y_run[59,0]
    p_m_10 = struct[0].y_run[60,0]
    p_t_10 = struct[0].y_run[61,0]
    p_u_10 = struct[0].y_run[62,0]
    v_u_10 = struct[0].y_run[63,0]
    k_u_10 = struct[0].y_run[64,0]
    k_cur_sat_10 = struct[0].y_run[65,0]
    p_gou_10 = struct[0].y_run[66,0]
    p_f_10 = struct[0].y_run[67,0]
    r_lim_10 = struct[0].y_run[68,0]
    omega_01 = struct[0].y_run[69,0]
    i_d_01 = struct[0].y_run[70,0]
    i_q_01 = struct[0].y_run[71,0]
    p_s_01 = struct[0].y_run[72,0]
    q_s_01 = struct[0].y_run[73,0]
    omega_coi = struct[0].y_run[74,0]
    p_agc = struct[0].y_run[75,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = Omega_b_08*(omega_08 - omega_coi)
        struct[0].f[1,0] = p_m_08 - p_s_08
        struct[0].f[2,0] = -q_s_08 + q_s_ref_08
        struct[0].f[3,0] = S_n_08*(p_gou_08 - p_t_08)/(C_u_08*(v_u_08 + 0.1))
        struct[0].f[4,0] = Piecewise(np.array([(-r_lim_08, r_lim_08 < -K_h_08*(-p_ghr_08 + p_gou_08)), (r_lim_08, r_lim_08 < K_h_08*(-p_ghr_08 + p_gou_08)), (K_h_08*(-p_ghr_08 + p_gou_08), True)]))
        struct[0].f[5,0] = (-k_cur_08 + p_g_ref_08/(inc_p_gin_08 + p_gin_0_08) + Piecewise(np.array([(P_f_min_08, P_f_min_08 > p_f_08), (P_f_max_08, P_f_max_08 < p_f_08), (p_f_08, True)]))/(inc_p_gin_08 + p_gin_0_08))/T_cur_08
        struct[0].f[6,0] = -0.001*inc_p_gin_08 + ramp_p_gin_08
        struct[0].f[7,0] = Omega_b_09*(omega_09 - omega_coi)
        struct[0].f[8,0] = p_m_09 - p_s_09
        struct[0].f[9,0] = -q_s_09 + q_s_ref_09
        struct[0].f[10,0] = S_n_09*(p_gou_09 - p_t_09)/(C_u_09*(v_u_09 + 0.1))
        struct[0].f[11,0] = Piecewise(np.array([(-r_lim_09, r_lim_09 < -K_h_09*(-p_ghr_09 + p_gou_09)), (r_lim_09, r_lim_09 < K_h_09*(-p_ghr_09 + p_gou_09)), (K_h_09*(-p_ghr_09 + p_gou_09), True)]))
        struct[0].f[12,0] = (-k_cur_09 + p_g_ref_09/(inc_p_gin_09 + p_gin_0_09) + Piecewise(np.array([(P_f_min_09, P_f_min_09 > p_f_09), (P_f_max_09, P_f_max_09 < p_f_09), (p_f_09, True)]))/(inc_p_gin_09 + p_gin_0_09))/T_cur_09
        struct[0].f[13,0] = -0.001*inc_p_gin_09 + ramp_p_gin_09
        struct[0].f[14,0] = Omega_b_10*(omega_10 - omega_coi)
        struct[0].f[15,0] = p_m_10 - p_s_10
        struct[0].f[16,0] = -q_s_10 + q_s_ref_10
        struct[0].f[17,0] = S_n_10*(p_gou_10 - p_t_10)/(C_u_10*(v_u_10 + 0.1))
        struct[0].f[18,0] = Piecewise(np.array([(-r_lim_10, r_lim_10 < -K_h_10*(-p_ghr_10 + p_gou_10)), (r_lim_10, r_lim_10 < K_h_10*(-p_ghr_10 + p_gou_10)), (K_h_10*(-p_ghr_10 + p_gou_10), True)]))
        struct[0].f[19,0] = (-k_cur_10 + p_g_ref_10/(inc_p_gin_10 + p_gin_0_10) + Piecewise(np.array([(P_f_min_10, P_f_min_10 > p_f_10), (P_f_max_10, P_f_max_10 < p_f_10), (p_f_10, True)]))/(inc_p_gin_10 + p_gin_0_10))/T_cur_10
        struct[0].f[20,0] = -0.001*inc_p_gin_10 + ramp_p_gin_10
        struct[0].f[21,0] = -K_delta_01*delta_01 + Omega_b_01*(omega_01 - omega_coi)
        struct[0].f[22,0] = -Domega_01*K_alpha_01 + alpha_01
        struct[0].f[23,0] = 1 - omega_coi
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = -P_01/S_base + V_01**2*g_01_02 + V_01*V_02*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) - S_n_01*p_s_01/S_base
        struct[0].g[1,0] = -Q_01/S_base + V_01**2*(-b_01_02 - bs_01_02/2) + V_01*V_02*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02)) - S_n_01*q_s_01/S_base
        struct[0].g[2,0] = -P_02/S_base + V_01*V_02*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) + V_02**2*(g_01_02 + g_02_03) + V_02*V_03*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].g[3,0] = -Q_02/S_base + V_01*V_02*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02)) + V_02**2*(-b_01_02 - b_02_03 - bs_01_02/2 - bs_02_03/2) + V_02*V_03*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].g[4,0] = -P_03/S_base + V_02*V_03*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03)) + V_03**2*(g_02_03 + g_03_04 + g_03_12) + V_03*V_04*(-b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04)) + V_03*V_12*(-b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12))
        struct[0].g[5,0] = -Q_03/S_base + V_02*V_03*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03)) + V_03**2*(-b_02_03 - b_03_04 - b_03_12 - bs_02_03/2 - bs_03_04/2 - bs_03_12/2) + V_03*V_04*(b_03_04*cos(theta_03 - theta_04) - g_03_04*sin(theta_03 - theta_04)) + V_03*V_12*(b_03_12*cos(theta_03 - theta_12) - g_03_12*sin(theta_03 - theta_12))
        struct[0].g[6,0] = -P_04/S_base + V_03*V_04*(b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04)) + V_04**2*(g_03_04 + g_04_05 + g_04_11) + V_04*V_05*(-b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05)) + V_04*V_11*(-b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11))
        struct[0].g[7,0] = -Q_04/S_base + V_03*V_04*(b_03_04*cos(theta_03 - theta_04) + g_03_04*sin(theta_03 - theta_04)) + V_04**2*(-b_03_04 - b_04_05 - b_04_11 - bs_03_04/2 - bs_04_05/2 - bs_04_11/2) + V_04*V_05*(b_04_05*cos(theta_04 - theta_05) - g_04_05*sin(theta_04 - theta_05)) + V_04*V_11*(b_04_11*cos(theta_04 - theta_11) - g_04_11*sin(theta_04 - theta_11))
        struct[0].g[8,0] = -P_05/S_base + V_04*V_05*(b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05)) + V_05**2*(g_04_05 + g_05_06 + g_05_10) + V_05*V_06*(-b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06)) + V_05*V_10*(-b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10))
        struct[0].g[9,0] = -Q_05/S_base + V_04*V_05*(b_04_05*cos(theta_04 - theta_05) + g_04_05*sin(theta_04 - theta_05)) + V_05**2*(-b_04_05 - b_05_06 - b_05_10 - bs_04_05/2 - bs_05_06/2 - bs_05_10/2) + V_05*V_06*(b_05_06*cos(theta_05 - theta_06) - g_05_06*sin(theta_05 - theta_06)) + V_05*V_10*(b_05_10*cos(theta_05 - theta_10) - g_05_10*sin(theta_05 - theta_10))
        struct[0].g[10,0] = -P_06/S_base + V_05*V_06*(b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06)) + V_06**2*(g_05_06 + g_06_07 + g_06_09) + V_06*V_07*(-b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07)) + V_06*V_09*(-b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09))
        struct[0].g[11,0] = -Q_06/S_base + V_05*V_06*(b_05_06*cos(theta_05 - theta_06) + g_05_06*sin(theta_05 - theta_06)) + V_06**2*(-b_05_06 - b_06_07 - b_06_09 - bs_05_06/2 - bs_06_07/2 - bs_06_09/2) + V_06*V_07*(b_06_07*cos(theta_06 - theta_07) - g_06_07*sin(theta_06 - theta_07)) + V_06*V_09*(b_06_09*cos(theta_06 - theta_09) - g_06_09*sin(theta_06 - theta_09))
        struct[0].g[12,0] = -P_07/S_base + V_06*V_07*(b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07)) + V_07**2*(g_06_07 + g_07_08) + V_07*V_08*(-b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08))
        struct[0].g[13,0] = -Q_07/S_base + V_06*V_07*(b_06_07*cos(theta_06 - theta_07) + g_06_07*sin(theta_06 - theta_07)) + V_07**2*(-b_06_07 - b_07_08 - bs_06_07/2 - bs_07_08/2) + V_07*V_08*(b_07_08*cos(theta_07 - theta_08) - g_07_08*sin(theta_07 - theta_08))
        struct[0].g[14,0] = -P_08/S_base + V_07*V_08*(b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08)) + V_08**2*g_07_08 - S_n_08*p_s_08/S_base
        struct[0].g[15,0] = -Q_08/S_base + V_07*V_08*(b_07_08*cos(theta_07 - theta_08) + g_07_08*sin(theta_07 - theta_08)) + V_08**2*(-b_07_08 - bs_07_08/2) - S_n_08*q_s_08/S_base
        struct[0].g[16,0] = -P_09/S_base + V_06*V_09*(b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09)) + V_09**2*g_06_09 - S_n_09*p_s_09/S_base
        struct[0].g[17,0] = -Q_09/S_base + V_06*V_09*(b_06_09*cos(theta_06 - theta_09) + g_06_09*sin(theta_06 - theta_09)) + V_09**2*(-b_06_09 - bs_06_09/2) - S_n_09*q_s_09/S_base
        struct[0].g[18,0] = -P_10/S_base + V_05*V_10*(b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10)) + V_10**2*g_05_10 - S_n_10*p_s_10/S_base
        struct[0].g[19,0] = -Q_10/S_base + V_05*V_10*(b_05_10*cos(theta_05 - theta_10) + g_05_10*sin(theta_05 - theta_10)) + V_10**2*(-b_05_10 - bs_05_10/2) - S_n_10*q_s_10/S_base
        struct[0].g[20,0] = -P_11/S_base + V_04*V_11*(b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11)) + V_11**2*g_04_11
        struct[0].g[21,0] = -Q_11/S_base + V_04*V_11*(b_04_11*cos(theta_04 - theta_11) + g_04_11*sin(theta_04 - theta_11)) + V_11**2*(-b_04_11 - bs_04_11/2)
        struct[0].g[22,0] = -P_12/S_base + V_03*V_12*(b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12)) + V_12**2*g_03_12
        struct[0].g[23,0] = -Q_12/S_base + V_03*V_12*(b_03_12*cos(theta_03 - theta_12) + g_03_12*sin(theta_03 - theta_12)) + V_12**2*(-b_03_12 - bs_03_12/2)
        struct[0].g[24,0] = K_p_08*(p_m_08 - p_s_08 + xi_p_08/T_p_08) - omega_08
        struct[0].g[25,0] = K_q_08*(-q_s_08 + q_s_ref_08 + xi_q_08/T_q_08) - e_qv_08
        struct[0].g[26,0] = -R_v_08*i_d_08 - V_08*sin(delta_08 - theta_08) + X_v_08*i_q_08
        struct[0].g[27,0] = -R_v_08*i_q_08 - V_08*cos(delta_08 - theta_08) - X_v_08*i_d_08 + e_qv_08
        struct[0].g[28,0] = V_08*i_d_08*sin(delta_08 - theta_08) + V_08*i_q_08*cos(delta_08 - theta_08) - p_s_08
        struct[0].g[29,0] = V_08*i_d_08*cos(delta_08 - theta_08) - V_08*i_q_08*sin(delta_08 - theta_08) - q_s_08
        struct[0].g[30,0] = K_fpfr_08*Piecewise(np.array([(P_f_min_08, P_f_min_08 > p_f_08), (P_f_max_08, P_f_max_08 < p_f_08), (p_f_08, True)])) + p_ghr_08 - p_m_08 + p_s_08 - p_t_08 + p_u_08
        struct[0].g[31,0] = i_d_08*(R_s_08*i_d_08 + V_08*sin(delta_08 - theta_08)) + i_q_08*(R_s_08*i_q_08 + V_08*cos(delta_08 - theta_08)) - p_t_08
        struct[0].g[32,0] = -p_u_08 - k_u_08*(-v_u_08**2 + v_u_ref_08**2)/V_u_max_08**2
        struct[0].g[33,0] = R_uc_08*S_n_08*(p_gou_08 - p_t_08)/(v_u_08 + 0.1) + e_u_08 - v_u_08
        struct[0].g[34,0] = -k_u_08 + Piecewise(np.array([(K_u_max_08, V_u_min_08 > v_u_08), (K_u_0_08 + (-K_u_0_08 + K_u_max_08)*(-V_u_lt_08 + v_u_08)/(-V_u_lt_08 + V_u_min_08), V_u_lt_08 > v_u_08), (K_u_0_08 + (-K_u_0_08 + K_u_max_08)*(-V_u_ht_08 + v_u_08)/(-V_u_ht_08 + V_u_max_08), V_u_ht_08 < v_u_08), (K_u_max_08, V_u_max_08 < v_u_08), (K_u_0_08, True)]))
        struct[0].g[35,0] = -k_cur_sat_08 + Piecewise(np.array([(0.0001, k_cur_08 < 0.0001), (1, k_cur_08 > 1), (k_cur_08, True)]))
        struct[0].g[36,0] = k_cur_sat_08*(inc_p_gin_08 + p_gin_0_08) - p_gou_08
        struct[0].g[37,0] = -p_f_08 - Piecewise(np.array([((0.5*DB_08 + omega_08 - omega_ref_08)/Droop_08, omega_08 < -0.5*DB_08 + omega_ref_08), ((-0.5*DB_08 + omega_08 - omega_ref_08)/Droop_08, omega_08 > 0.5*DB_08 + omega_ref_08), (0.0, True)]))
        struct[0].g[38,0] = -r_lim_08 + Piecewise(np.array([(R_lim_max_08, (omega_08 > 0.5*DB_08 + omega_ref_08) | (omega_08 < -0.5*DB_08 + omega_ref_08)), (0.0, True)])) + Piecewise(np.array([(R_lim_08 + (-R_lim_08 + R_lim_max_08)*(-V_u_lt_08 + v_u_08)/(-V_u_lt_08 + V_u_min_08), V_u_lt_08 > v_u_08), (R_lim_08 + (-R_lim_08 + R_lim_max_08)*(-V_u_ht_08 + v_u_08)/(-V_u_ht_08 + V_u_max_08), V_u_ht_08 < v_u_08), (R_lim_08, True)]))
        struct[0].g[39,0] = K_p_09*(p_m_09 - p_s_09 + xi_p_09/T_p_09) - omega_09
        struct[0].g[40,0] = K_q_09*(-q_s_09 + q_s_ref_09 + xi_q_09/T_q_09) - e_qv_09
        struct[0].g[41,0] = -R_v_09*i_d_09 - V_09*sin(delta_09 - theta_09) + X_v_09*i_q_09
        struct[0].g[42,0] = -R_v_09*i_q_09 - V_09*cos(delta_09 - theta_09) - X_v_09*i_d_09 + e_qv_09
        struct[0].g[43,0] = V_09*i_d_09*sin(delta_09 - theta_09) + V_09*i_q_09*cos(delta_09 - theta_09) - p_s_09
        struct[0].g[44,0] = V_09*i_d_09*cos(delta_09 - theta_09) - V_09*i_q_09*sin(delta_09 - theta_09) - q_s_09
        struct[0].g[45,0] = K_fpfr_09*Piecewise(np.array([(P_f_min_09, P_f_min_09 > p_f_09), (P_f_max_09, P_f_max_09 < p_f_09), (p_f_09, True)])) + p_ghr_09 - p_m_09 + p_s_09 - p_t_09 + p_u_09
        struct[0].g[46,0] = i_d_09*(R_s_09*i_d_09 + V_09*sin(delta_09 - theta_09)) + i_q_09*(R_s_09*i_q_09 + V_09*cos(delta_09 - theta_09)) - p_t_09
        struct[0].g[47,0] = -p_u_09 - k_u_09*(-v_u_09**2 + v_u_ref_09**2)/V_u_max_09**2
        struct[0].g[48,0] = R_uc_09*S_n_09*(p_gou_09 - p_t_09)/(v_u_09 + 0.1) + e_u_09 - v_u_09
        struct[0].g[49,0] = -k_u_09 + Piecewise(np.array([(K_u_max_09, V_u_min_09 > v_u_09), (K_u_0_09 + (-K_u_0_09 + K_u_max_09)*(-V_u_lt_09 + v_u_09)/(-V_u_lt_09 + V_u_min_09), V_u_lt_09 > v_u_09), (K_u_0_09 + (-K_u_0_09 + K_u_max_09)*(-V_u_ht_09 + v_u_09)/(-V_u_ht_09 + V_u_max_09), V_u_ht_09 < v_u_09), (K_u_max_09, V_u_max_09 < v_u_09), (K_u_0_09, True)]))
        struct[0].g[50,0] = -k_cur_sat_09 + Piecewise(np.array([(0.0001, k_cur_09 < 0.0001), (1, k_cur_09 > 1), (k_cur_09, True)]))
        struct[0].g[51,0] = k_cur_sat_09*(inc_p_gin_09 + p_gin_0_09) - p_gou_09
        struct[0].g[52,0] = -p_f_09 - Piecewise(np.array([((0.5*DB_09 + omega_09 - omega_ref_09)/Droop_09, omega_09 < -0.5*DB_09 + omega_ref_09), ((-0.5*DB_09 + omega_09 - omega_ref_09)/Droop_09, omega_09 > 0.5*DB_09 + omega_ref_09), (0.0, True)]))
        struct[0].g[53,0] = -r_lim_09 + Piecewise(np.array([(R_lim_max_09, (omega_09 > 0.5*DB_09 + omega_ref_09) | (omega_09 < -0.5*DB_09 + omega_ref_09)), (0.0, True)])) + Piecewise(np.array([(R_lim_09 + (-R_lim_09 + R_lim_max_09)*(-V_u_lt_09 + v_u_09)/(-V_u_lt_09 + V_u_min_09), V_u_lt_09 > v_u_09), (R_lim_09 + (-R_lim_09 + R_lim_max_09)*(-V_u_ht_09 + v_u_09)/(-V_u_ht_09 + V_u_max_09), V_u_ht_09 < v_u_09), (R_lim_09, True)]))
        struct[0].g[54,0] = K_p_10*(p_m_10 - p_s_10 + xi_p_10/T_p_10) - omega_10
        struct[0].g[55,0] = K_q_10*(-q_s_10 + q_s_ref_10 + xi_q_10/T_q_10) - e_qv_10
        struct[0].g[56,0] = -R_v_10*i_d_10 - V_10*sin(delta_10 - theta_10) + X_v_10*i_q_10
        struct[0].g[57,0] = -R_v_10*i_q_10 - V_10*cos(delta_10 - theta_10) - X_v_10*i_d_10 + e_qv_10
        struct[0].g[58,0] = V_10*i_d_10*sin(delta_10 - theta_10) + V_10*i_q_10*cos(delta_10 - theta_10) - p_s_10
        struct[0].g[59,0] = V_10*i_d_10*cos(delta_10 - theta_10) - V_10*i_q_10*sin(delta_10 - theta_10) - q_s_10
        struct[0].g[60,0] = K_fpfr_10*Piecewise(np.array([(P_f_min_10, P_f_min_10 > p_f_10), (P_f_max_10, P_f_max_10 < p_f_10), (p_f_10, True)])) + p_ghr_10 - p_m_10 + p_s_10 - p_t_10 + p_u_10
        struct[0].g[61,0] = i_d_10*(R_s_10*i_d_10 + V_10*sin(delta_10 - theta_10)) + i_q_10*(R_s_10*i_q_10 + V_10*cos(delta_10 - theta_10)) - p_t_10
        struct[0].g[62,0] = -p_u_10 - k_u_10*(-v_u_10**2 + v_u_ref_10**2)/V_u_max_10**2
        struct[0].g[63,0] = R_uc_10*S_n_10*(p_gou_10 - p_t_10)/(v_u_10 + 0.1) + e_u_10 - v_u_10
        struct[0].g[64,0] = -k_u_10 + Piecewise(np.array([(K_u_max_10, V_u_min_10 > v_u_10), (K_u_0_10 + (-K_u_0_10 + K_u_max_10)*(-V_u_lt_10 + v_u_10)/(-V_u_lt_10 + V_u_min_10), V_u_lt_10 > v_u_10), (K_u_0_10 + (-K_u_0_10 + K_u_max_10)*(-V_u_ht_10 + v_u_10)/(-V_u_ht_10 + V_u_max_10), V_u_ht_10 < v_u_10), (K_u_max_10, V_u_max_10 < v_u_10), (K_u_0_10, True)]))
        struct[0].g[65,0] = -k_cur_sat_10 + Piecewise(np.array([(0.0001, k_cur_10 < 0.0001), (1, k_cur_10 > 1), (k_cur_10, True)]))
        struct[0].g[66,0] = k_cur_sat_10*(inc_p_gin_10 + p_gin_0_10) - p_gou_10
        struct[0].g[67,0] = -p_f_10 - Piecewise(np.array([((0.5*DB_10 + omega_10 - omega_ref_10)/Droop_10, omega_10 < -0.5*DB_10 + omega_ref_10), ((-0.5*DB_10 + omega_10 - omega_ref_10)/Droop_10, omega_10 > 0.5*DB_10 + omega_ref_10), (0.0, True)]))
        struct[0].g[68,0] = -r_lim_10 + Piecewise(np.array([(R_lim_max_10, (omega_10 > 0.5*DB_10 + omega_ref_10) | (omega_10 < -0.5*DB_10 + omega_ref_10)), (0.0, True)])) + Piecewise(np.array([(R_lim_10 + (-R_lim_10 + R_lim_max_10)*(-V_u_lt_10 + v_u_10)/(-V_u_lt_10 + V_u_min_10), V_u_lt_10 > v_u_10), (R_lim_10 + (-R_lim_10 + R_lim_max_10)*(-V_u_ht_10 + v_u_10)/(-V_u_ht_10 + V_u_max_10), V_u_ht_10 < v_u_10), (R_lim_10, True)]))
        struct[0].g[69,0] = Domega_01 - omega_01 + omega_ref_01
        struct[0].g[70,0] = -R_v_01*i_d_01 - V_01*sin(delta_01 - theta_01) + X_v_01*i_q_01
        struct[0].g[71,0] = -R_v_01*i_q_01 - V_01*cos(delta_01 - theta_01) - X_v_01*i_d_01 + e_qv_01
        struct[0].g[72,0] = V_01*i_d_01*sin(delta_01 - theta_01) + V_01*i_q_01*cos(delta_01 - theta_01) - p_s_01
        struct[0].g[73,0] = V_01*i_d_01*cos(delta_01 - theta_01) - V_01*i_q_01*sin(delta_01 - theta_01) - q_s_01
        struct[0].g[74,0] = -omega_coi + (1000000.0*S_n_01*omega_01 + S_n_10*T_p_10*omega_10/(2*K_p_10) + S_n_09*T_p_09*omega_09/(2*K_p_09) + S_n_08*T_p_08*omega_08/(2*K_p_08))/(1000000.0*S_n_01 + S_n_10*T_p_10/(2*K_p_10) + S_n_09*T_p_09/(2*K_p_09) + S_n_08*T_p_08/(2*K_p_08))
        struct[0].g[75,0] = K_i_agc*xi_freq + K_p_agc*(1 - omega_coi) - p_agc
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = V_01
        struct[0].h[1,0] = V_02
        struct[0].h[2,0] = V_03
        struct[0].h[3,0] = V_04
        struct[0].h[4,0] = V_05
        struct[0].h[5,0] = V_06
        struct[0].h[6,0] = V_07
        struct[0].h[7,0] = V_08
        struct[0].h[8,0] = V_09
        struct[0].h[9,0] = V_10
        struct[0].h[10,0] = V_11
        struct[0].h[11,0] = V_12
        struct[0].h[12,0] = inc_p_gin_08 + p_gin_0_08
        struct[0].h[13,0] = p_g_ref_08
        struct[0].h[14,0] = -p_s_08 + p_t_08
        struct[0].h[15,0] = (-V_u_min_08**2 + e_u_08**2)/(V_u_max_08**2 - V_u_min_08**2)
        struct[0].h[16,0] = K_fpfr_08*Piecewise(np.array([(P_f_min_08, P_f_min_08 > p_f_08), (P_f_max_08, P_f_max_08 < p_f_08), (p_f_08, True)]))
        struct[0].h[17,0] = Piecewise(np.array([(P_f_min_08, P_f_min_08 > p_f_08), (P_f_max_08, P_f_max_08 < p_f_08), (p_f_08, True)]))
        struct[0].h[18,0] = inc_p_gin_09 + p_gin_0_09
        struct[0].h[19,0] = p_g_ref_09
        struct[0].h[20,0] = -p_s_09 + p_t_09
        struct[0].h[21,0] = (-V_u_min_09**2 + e_u_09**2)/(V_u_max_09**2 - V_u_min_09**2)
        struct[0].h[22,0] = K_fpfr_09*Piecewise(np.array([(P_f_min_09, P_f_min_09 > p_f_09), (P_f_max_09, P_f_max_09 < p_f_09), (p_f_09, True)]))
        struct[0].h[23,0] = Piecewise(np.array([(P_f_min_09, P_f_min_09 > p_f_09), (P_f_max_09, P_f_max_09 < p_f_09), (p_f_09, True)]))
        struct[0].h[24,0] = inc_p_gin_10 + p_gin_0_10
        struct[0].h[25,0] = p_g_ref_10
        struct[0].h[26,0] = -p_s_10 + p_t_10
        struct[0].h[27,0] = (-V_u_min_10**2 + e_u_10**2)/(V_u_max_10**2 - V_u_min_10**2)
        struct[0].h[28,0] = K_fpfr_10*Piecewise(np.array([(P_f_min_10, P_f_min_10 > p_f_10), (P_f_max_10, P_f_max_10 < p_f_10), (p_f_10, True)]))
        struct[0].h[29,0] = Piecewise(np.array([(P_f_min_10, P_f_min_10 > p_f_10), (P_f_max_10, P_f_max_10 < p_f_10), (p_f_10, True)]))
        struct[0].h[30,0] = alpha_01
    

    if mode == 10:

        struct[0].Fx[4,4] = Piecewise(np.array([(0, (r_lim_08 < K_h_08*(-p_ghr_08 + p_gou_08)) | (r_lim_08 < -K_h_08*(-p_ghr_08 + p_gou_08))), (-K_h_08, True)]))
        struct[0].Fx[5,5] = -1/T_cur_08
        struct[0].Fx[5,6] = (-p_g_ref_08/(inc_p_gin_08 + p_gin_0_08)**2 - Piecewise(np.array([(P_f_min_08, P_f_min_08 > p_f_08), (P_f_max_08, P_f_max_08 < p_f_08), (p_f_08, True)]))/(inc_p_gin_08 + p_gin_0_08)**2)/T_cur_08
        struct[0].Fx[6,6] = -0.00100000000000000
        struct[0].Fx[11,11] = Piecewise(np.array([(0, (r_lim_09 < K_h_09*(-p_ghr_09 + p_gou_09)) | (r_lim_09 < -K_h_09*(-p_ghr_09 + p_gou_09))), (-K_h_09, True)]))
        struct[0].Fx[12,12] = -1/T_cur_09
        struct[0].Fx[12,13] = (-p_g_ref_09/(inc_p_gin_09 + p_gin_0_09)**2 - Piecewise(np.array([(P_f_min_09, P_f_min_09 > p_f_09), (P_f_max_09, P_f_max_09 < p_f_09), (p_f_09, True)]))/(inc_p_gin_09 + p_gin_0_09)**2)/T_cur_09
        struct[0].Fx[13,13] = -0.00100000000000000
        struct[0].Fx[18,18] = Piecewise(np.array([(0, (r_lim_10 < K_h_10*(-p_ghr_10 + p_gou_10)) | (r_lim_10 < -K_h_10*(-p_ghr_10 + p_gou_10))), (-K_h_10, True)]))
        struct[0].Fx[19,19] = -1/T_cur_10
        struct[0].Fx[19,20] = (-p_g_ref_10/(inc_p_gin_10 + p_gin_0_10)**2 - Piecewise(np.array([(P_f_min_10, P_f_min_10 > p_f_10), (P_f_max_10, P_f_max_10 < p_f_10), (p_f_10, True)]))/(inc_p_gin_10 + p_gin_0_10)**2)/T_cur_10
        struct[0].Fx[20,20] = -0.00100000000000000
        struct[0].Fx[21,21] = -K_delta_01
        struct[0].Fx[22,22] = -K_alpha_01

    if mode == 11:

        struct[0].Fy[0,24] = Omega_b_08
        struct[0].Fy[0,74] = -Omega_b_08
        struct[0].Fy[1,28] = -1
        struct[0].Fy[1,30] = 1
        struct[0].Fy[2,29] = -1
        struct[0].Fy[3,31] = -S_n_08/(C_u_08*(v_u_08 + 0.1))
        struct[0].Fy[3,33] = -S_n_08*(p_gou_08 - p_t_08)/(C_u_08*(v_u_08 + 0.1)**2)
        struct[0].Fy[3,36] = S_n_08/(C_u_08*(v_u_08 + 0.1))
        struct[0].Fy[4,36] = Piecewise(np.array([(0, (r_lim_08 < K_h_08*(-p_ghr_08 + p_gou_08)) | (r_lim_08 < -K_h_08*(-p_ghr_08 + p_gou_08))), (K_h_08, True)]))
        struct[0].Fy[4,38] = Piecewise(np.array([(-1, r_lim_08 < -K_h_08*(-p_ghr_08 + p_gou_08)), (1, r_lim_08 < K_h_08*(-p_ghr_08 + p_gou_08)), (0, True)]))
        struct[0].Fy[5,37] = Piecewise(np.array([(0, (P_f_min_08 > p_f_08) | (P_f_max_08 < p_f_08)), (1, True)]))/(T_cur_08*(inc_p_gin_08 + p_gin_0_08))
        struct[0].Fy[7,39] = Omega_b_09
        struct[0].Fy[7,74] = -Omega_b_09
        struct[0].Fy[8,43] = -1
        struct[0].Fy[8,45] = 1
        struct[0].Fy[9,44] = -1
        struct[0].Fy[10,46] = -S_n_09/(C_u_09*(v_u_09 + 0.1))
        struct[0].Fy[10,48] = -S_n_09*(p_gou_09 - p_t_09)/(C_u_09*(v_u_09 + 0.1)**2)
        struct[0].Fy[10,51] = S_n_09/(C_u_09*(v_u_09 + 0.1))
        struct[0].Fy[11,51] = Piecewise(np.array([(0, (r_lim_09 < K_h_09*(-p_ghr_09 + p_gou_09)) | (r_lim_09 < -K_h_09*(-p_ghr_09 + p_gou_09))), (K_h_09, True)]))
        struct[0].Fy[11,53] = Piecewise(np.array([(-1, r_lim_09 < -K_h_09*(-p_ghr_09 + p_gou_09)), (1, r_lim_09 < K_h_09*(-p_ghr_09 + p_gou_09)), (0, True)]))
        struct[0].Fy[12,52] = Piecewise(np.array([(0, (P_f_min_09 > p_f_09) | (P_f_max_09 < p_f_09)), (1, True)]))/(T_cur_09*(inc_p_gin_09 + p_gin_0_09))
        struct[0].Fy[14,54] = Omega_b_10
        struct[0].Fy[14,74] = -Omega_b_10
        struct[0].Fy[15,58] = -1
        struct[0].Fy[15,60] = 1
        struct[0].Fy[16,59] = -1
        struct[0].Fy[17,61] = -S_n_10/(C_u_10*(v_u_10 + 0.1))
        struct[0].Fy[17,63] = -S_n_10*(p_gou_10 - p_t_10)/(C_u_10*(v_u_10 + 0.1)**2)
        struct[0].Fy[17,66] = S_n_10/(C_u_10*(v_u_10 + 0.1))
        struct[0].Fy[18,66] = Piecewise(np.array([(0, (r_lim_10 < K_h_10*(-p_ghr_10 + p_gou_10)) | (r_lim_10 < -K_h_10*(-p_ghr_10 + p_gou_10))), (K_h_10, True)]))
        struct[0].Fy[18,68] = Piecewise(np.array([(-1, r_lim_10 < -K_h_10*(-p_ghr_10 + p_gou_10)), (1, r_lim_10 < K_h_10*(-p_ghr_10 + p_gou_10)), (0, True)]))
        struct[0].Fy[19,67] = Piecewise(np.array([(0, (P_f_min_10 > p_f_10) | (P_f_max_10 < p_f_10)), (1, True)]))/(T_cur_10*(inc_p_gin_10 + p_gin_0_10))
        struct[0].Fy[21,69] = Omega_b_01
        struct[0].Fy[21,74] = -Omega_b_01
        struct[0].Fy[23,74] = -1

        struct[0].Gy[0,0] = 2*V_01*g_01_02 + V_02*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy[0,1] = V_01*V_02*(-b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy[0,2] = V_01*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy[0,3] = V_01*V_02*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy[0,72] = -S_n_01/S_base
        struct[0].Gy[1,0] = 2*V_01*(-b_01_02 - bs_01_02/2) + V_02*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy[1,1] = V_01*V_02*(-b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy[1,2] = V_01*(b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy[1,3] = V_01*V_02*(b_01_02*sin(theta_01 - theta_02) + g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy[1,73] = -S_n_01/S_base
        struct[0].Gy[2,0] = V_02*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy[2,1] = V_01*V_02*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy[2,2] = V_01*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) + 2*V_02*(g_01_02 + g_02_03) + V_03*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy[2,3] = V_01*V_02*(-b_01_02*cos(theta_01 - theta_02) - g_01_02*sin(theta_01 - theta_02)) + V_02*V_03*(-b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy[2,4] = V_02*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy[2,5] = V_02*V_03*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy[3,0] = V_02*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02))
        struct[0].Gy[3,1] = V_01*V_02*(-b_01_02*sin(theta_01 - theta_02) + g_01_02*cos(theta_01 - theta_02))
        struct[0].Gy[3,2] = V_01*(b_01_02*cos(theta_01 - theta_02) + g_01_02*sin(theta_01 - theta_02)) + 2*V_02*(-b_01_02 - b_02_03 - bs_01_02/2 - bs_02_03/2) + V_03*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy[3,3] = V_01*V_02*(b_01_02*sin(theta_01 - theta_02) - g_01_02*cos(theta_01 - theta_02)) + V_02*V_03*(-b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy[3,4] = V_02*(b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy[3,5] = V_02*V_03*(b_02_03*sin(theta_02 - theta_03) + g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy[4,2] = V_03*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy[4,3] = V_02*V_03*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy[4,4] = V_02*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03)) + 2*V_03*(g_02_03 + g_03_04 + g_03_12) + V_04*(-b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04)) + V_12*(-b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12))
        struct[0].Gy[4,5] = V_02*V_03*(-b_02_03*cos(theta_02 - theta_03) - g_02_03*sin(theta_02 - theta_03)) + V_03*V_04*(-b_03_04*cos(theta_03 - theta_04) + g_03_04*sin(theta_03 - theta_04)) + V_03*V_12*(-b_03_12*cos(theta_03 - theta_12) + g_03_12*sin(theta_03 - theta_12))
        struct[0].Gy[4,6] = V_03*(-b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04))
        struct[0].Gy[4,7] = V_03*V_04*(b_03_04*cos(theta_03 - theta_04) - g_03_04*sin(theta_03 - theta_04))
        struct[0].Gy[4,22] = V_03*(-b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12))
        struct[0].Gy[4,23] = V_03*V_12*(b_03_12*cos(theta_03 - theta_12) - g_03_12*sin(theta_03 - theta_12))
        struct[0].Gy[5,2] = V_03*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03))
        struct[0].Gy[5,3] = V_02*V_03*(-b_02_03*sin(theta_02 - theta_03) + g_02_03*cos(theta_02 - theta_03))
        struct[0].Gy[5,4] = V_02*(b_02_03*cos(theta_02 - theta_03) + g_02_03*sin(theta_02 - theta_03)) + 2*V_03*(-b_02_03 - b_03_04 - b_03_12 - bs_02_03/2 - bs_03_04/2 - bs_03_12/2) + V_04*(b_03_04*cos(theta_03 - theta_04) - g_03_04*sin(theta_03 - theta_04)) + V_12*(b_03_12*cos(theta_03 - theta_12) - g_03_12*sin(theta_03 - theta_12))
        struct[0].Gy[5,5] = V_02*V_03*(b_02_03*sin(theta_02 - theta_03) - g_02_03*cos(theta_02 - theta_03)) + V_03*V_04*(-b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04)) + V_03*V_12*(-b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12))
        struct[0].Gy[5,6] = V_03*(b_03_04*cos(theta_03 - theta_04) - g_03_04*sin(theta_03 - theta_04))
        struct[0].Gy[5,7] = V_03*V_04*(b_03_04*sin(theta_03 - theta_04) + g_03_04*cos(theta_03 - theta_04))
        struct[0].Gy[5,22] = V_03*(b_03_12*cos(theta_03 - theta_12) - g_03_12*sin(theta_03 - theta_12))
        struct[0].Gy[5,23] = V_03*V_12*(b_03_12*sin(theta_03 - theta_12) + g_03_12*cos(theta_03 - theta_12))
        struct[0].Gy[6,4] = V_04*(b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04))
        struct[0].Gy[6,5] = V_03*V_04*(b_03_04*cos(theta_03 - theta_04) + g_03_04*sin(theta_03 - theta_04))
        struct[0].Gy[6,6] = V_03*(b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04)) + 2*V_04*(g_03_04 + g_04_05 + g_04_11) + V_05*(-b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05)) + V_11*(-b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11))
        struct[0].Gy[6,7] = V_03*V_04*(-b_03_04*cos(theta_03 - theta_04) - g_03_04*sin(theta_03 - theta_04)) + V_04*V_05*(-b_04_05*cos(theta_04 - theta_05) + g_04_05*sin(theta_04 - theta_05)) + V_04*V_11*(-b_04_11*cos(theta_04 - theta_11) + g_04_11*sin(theta_04 - theta_11))
        struct[0].Gy[6,8] = V_04*(-b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05))
        struct[0].Gy[6,9] = V_04*V_05*(b_04_05*cos(theta_04 - theta_05) - g_04_05*sin(theta_04 - theta_05))
        struct[0].Gy[6,20] = V_04*(-b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11))
        struct[0].Gy[6,21] = V_04*V_11*(b_04_11*cos(theta_04 - theta_11) - g_04_11*sin(theta_04 - theta_11))
        struct[0].Gy[7,4] = V_04*(b_03_04*cos(theta_03 - theta_04) + g_03_04*sin(theta_03 - theta_04))
        struct[0].Gy[7,5] = V_03*V_04*(-b_03_04*sin(theta_03 - theta_04) + g_03_04*cos(theta_03 - theta_04))
        struct[0].Gy[7,6] = V_03*(b_03_04*cos(theta_03 - theta_04) + g_03_04*sin(theta_03 - theta_04)) + 2*V_04*(-b_03_04 - b_04_05 - b_04_11 - bs_03_04/2 - bs_04_05/2 - bs_04_11/2) + V_05*(b_04_05*cos(theta_04 - theta_05) - g_04_05*sin(theta_04 - theta_05)) + V_11*(b_04_11*cos(theta_04 - theta_11) - g_04_11*sin(theta_04 - theta_11))
        struct[0].Gy[7,7] = V_03*V_04*(b_03_04*sin(theta_03 - theta_04) - g_03_04*cos(theta_03 - theta_04)) + V_04*V_05*(-b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05)) + V_04*V_11*(-b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11))
        struct[0].Gy[7,8] = V_04*(b_04_05*cos(theta_04 - theta_05) - g_04_05*sin(theta_04 - theta_05))
        struct[0].Gy[7,9] = V_04*V_05*(b_04_05*sin(theta_04 - theta_05) + g_04_05*cos(theta_04 - theta_05))
        struct[0].Gy[7,20] = V_04*(b_04_11*cos(theta_04 - theta_11) - g_04_11*sin(theta_04 - theta_11))
        struct[0].Gy[7,21] = V_04*V_11*(b_04_11*sin(theta_04 - theta_11) + g_04_11*cos(theta_04 - theta_11))
        struct[0].Gy[8,6] = V_05*(b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05))
        struct[0].Gy[8,7] = V_04*V_05*(b_04_05*cos(theta_04 - theta_05) + g_04_05*sin(theta_04 - theta_05))
        struct[0].Gy[8,8] = V_04*(b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05)) + 2*V_05*(g_04_05 + g_05_06 + g_05_10) + V_06*(-b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06)) + V_10*(-b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10))
        struct[0].Gy[8,9] = V_04*V_05*(-b_04_05*cos(theta_04 - theta_05) - g_04_05*sin(theta_04 - theta_05)) + V_05*V_06*(-b_05_06*cos(theta_05 - theta_06) + g_05_06*sin(theta_05 - theta_06)) + V_05*V_10*(-b_05_10*cos(theta_05 - theta_10) + g_05_10*sin(theta_05 - theta_10))
        struct[0].Gy[8,10] = V_05*(-b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06))
        struct[0].Gy[8,11] = V_05*V_06*(b_05_06*cos(theta_05 - theta_06) - g_05_06*sin(theta_05 - theta_06))
        struct[0].Gy[8,18] = V_05*(-b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10))
        struct[0].Gy[8,19] = V_05*V_10*(b_05_10*cos(theta_05 - theta_10) - g_05_10*sin(theta_05 - theta_10))
        struct[0].Gy[9,6] = V_05*(b_04_05*cos(theta_04 - theta_05) + g_04_05*sin(theta_04 - theta_05))
        struct[0].Gy[9,7] = V_04*V_05*(-b_04_05*sin(theta_04 - theta_05) + g_04_05*cos(theta_04 - theta_05))
        struct[0].Gy[9,8] = V_04*(b_04_05*cos(theta_04 - theta_05) + g_04_05*sin(theta_04 - theta_05)) + 2*V_05*(-b_04_05 - b_05_06 - b_05_10 - bs_04_05/2 - bs_05_06/2 - bs_05_10/2) + V_06*(b_05_06*cos(theta_05 - theta_06) - g_05_06*sin(theta_05 - theta_06)) + V_10*(b_05_10*cos(theta_05 - theta_10) - g_05_10*sin(theta_05 - theta_10))
        struct[0].Gy[9,9] = V_04*V_05*(b_04_05*sin(theta_04 - theta_05) - g_04_05*cos(theta_04 - theta_05)) + V_05*V_06*(-b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06)) + V_05*V_10*(-b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10))
        struct[0].Gy[9,10] = V_05*(b_05_06*cos(theta_05 - theta_06) - g_05_06*sin(theta_05 - theta_06))
        struct[0].Gy[9,11] = V_05*V_06*(b_05_06*sin(theta_05 - theta_06) + g_05_06*cos(theta_05 - theta_06))
        struct[0].Gy[9,18] = V_05*(b_05_10*cos(theta_05 - theta_10) - g_05_10*sin(theta_05 - theta_10))
        struct[0].Gy[9,19] = V_05*V_10*(b_05_10*sin(theta_05 - theta_10) + g_05_10*cos(theta_05 - theta_10))
        struct[0].Gy[10,8] = V_06*(b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06))
        struct[0].Gy[10,9] = V_05*V_06*(b_05_06*cos(theta_05 - theta_06) + g_05_06*sin(theta_05 - theta_06))
        struct[0].Gy[10,10] = V_05*(b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06)) + 2*V_06*(g_05_06 + g_06_07 + g_06_09) + V_07*(-b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07)) + V_09*(-b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09))
        struct[0].Gy[10,11] = V_05*V_06*(-b_05_06*cos(theta_05 - theta_06) - g_05_06*sin(theta_05 - theta_06)) + V_06*V_07*(-b_06_07*cos(theta_06 - theta_07) + g_06_07*sin(theta_06 - theta_07)) + V_06*V_09*(-b_06_09*cos(theta_06 - theta_09) + g_06_09*sin(theta_06 - theta_09))
        struct[0].Gy[10,12] = V_06*(-b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07))
        struct[0].Gy[10,13] = V_06*V_07*(b_06_07*cos(theta_06 - theta_07) - g_06_07*sin(theta_06 - theta_07))
        struct[0].Gy[10,16] = V_06*(-b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09))
        struct[0].Gy[10,17] = V_06*V_09*(b_06_09*cos(theta_06 - theta_09) - g_06_09*sin(theta_06 - theta_09))
        struct[0].Gy[11,8] = V_06*(b_05_06*cos(theta_05 - theta_06) + g_05_06*sin(theta_05 - theta_06))
        struct[0].Gy[11,9] = V_05*V_06*(-b_05_06*sin(theta_05 - theta_06) + g_05_06*cos(theta_05 - theta_06))
        struct[0].Gy[11,10] = V_05*(b_05_06*cos(theta_05 - theta_06) + g_05_06*sin(theta_05 - theta_06)) + 2*V_06*(-b_05_06 - b_06_07 - b_06_09 - bs_05_06/2 - bs_06_07/2 - bs_06_09/2) + V_07*(b_06_07*cos(theta_06 - theta_07) - g_06_07*sin(theta_06 - theta_07)) + V_09*(b_06_09*cos(theta_06 - theta_09) - g_06_09*sin(theta_06 - theta_09))
        struct[0].Gy[11,11] = V_05*V_06*(b_05_06*sin(theta_05 - theta_06) - g_05_06*cos(theta_05 - theta_06)) + V_06*V_07*(-b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07)) + V_06*V_09*(-b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09))
        struct[0].Gy[11,12] = V_06*(b_06_07*cos(theta_06 - theta_07) - g_06_07*sin(theta_06 - theta_07))
        struct[0].Gy[11,13] = V_06*V_07*(b_06_07*sin(theta_06 - theta_07) + g_06_07*cos(theta_06 - theta_07))
        struct[0].Gy[11,16] = V_06*(b_06_09*cos(theta_06 - theta_09) - g_06_09*sin(theta_06 - theta_09))
        struct[0].Gy[11,17] = V_06*V_09*(b_06_09*sin(theta_06 - theta_09) + g_06_09*cos(theta_06 - theta_09))
        struct[0].Gy[12,10] = V_07*(b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07))
        struct[0].Gy[12,11] = V_06*V_07*(b_06_07*cos(theta_06 - theta_07) + g_06_07*sin(theta_06 - theta_07))
        struct[0].Gy[12,12] = V_06*(b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07)) + 2*V_07*(g_06_07 + g_07_08) + V_08*(-b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08))
        struct[0].Gy[12,13] = V_06*V_07*(-b_06_07*cos(theta_06 - theta_07) - g_06_07*sin(theta_06 - theta_07)) + V_07*V_08*(-b_07_08*cos(theta_07 - theta_08) + g_07_08*sin(theta_07 - theta_08))
        struct[0].Gy[12,14] = V_07*(-b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08))
        struct[0].Gy[12,15] = V_07*V_08*(b_07_08*cos(theta_07 - theta_08) - g_07_08*sin(theta_07 - theta_08))
        struct[0].Gy[13,10] = V_07*(b_06_07*cos(theta_06 - theta_07) + g_06_07*sin(theta_06 - theta_07))
        struct[0].Gy[13,11] = V_06*V_07*(-b_06_07*sin(theta_06 - theta_07) + g_06_07*cos(theta_06 - theta_07))
        struct[0].Gy[13,12] = V_06*(b_06_07*cos(theta_06 - theta_07) + g_06_07*sin(theta_06 - theta_07)) + 2*V_07*(-b_06_07 - b_07_08 - bs_06_07/2 - bs_07_08/2) + V_08*(b_07_08*cos(theta_07 - theta_08) - g_07_08*sin(theta_07 - theta_08))
        struct[0].Gy[13,13] = V_06*V_07*(b_06_07*sin(theta_06 - theta_07) - g_06_07*cos(theta_06 - theta_07)) + V_07*V_08*(-b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08))
        struct[0].Gy[13,14] = V_07*(b_07_08*cos(theta_07 - theta_08) - g_07_08*sin(theta_07 - theta_08))
        struct[0].Gy[13,15] = V_07*V_08*(b_07_08*sin(theta_07 - theta_08) + g_07_08*cos(theta_07 - theta_08))
        struct[0].Gy[14,12] = V_08*(b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08))
        struct[0].Gy[14,13] = V_07*V_08*(b_07_08*cos(theta_07 - theta_08) + g_07_08*sin(theta_07 - theta_08))
        struct[0].Gy[14,14] = V_07*(b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08)) + 2*V_08*g_07_08
        struct[0].Gy[14,15] = V_07*V_08*(-b_07_08*cos(theta_07 - theta_08) - g_07_08*sin(theta_07 - theta_08))
        struct[0].Gy[14,28] = -S_n_08/S_base
        struct[0].Gy[15,12] = V_08*(b_07_08*cos(theta_07 - theta_08) + g_07_08*sin(theta_07 - theta_08))
        struct[0].Gy[15,13] = V_07*V_08*(-b_07_08*sin(theta_07 - theta_08) + g_07_08*cos(theta_07 - theta_08))
        struct[0].Gy[15,14] = V_07*(b_07_08*cos(theta_07 - theta_08) + g_07_08*sin(theta_07 - theta_08)) + 2*V_08*(-b_07_08 - bs_07_08/2)
        struct[0].Gy[15,15] = V_07*V_08*(b_07_08*sin(theta_07 - theta_08) - g_07_08*cos(theta_07 - theta_08))
        struct[0].Gy[15,29] = -S_n_08/S_base
        struct[0].Gy[16,10] = V_09*(b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09))
        struct[0].Gy[16,11] = V_06*V_09*(b_06_09*cos(theta_06 - theta_09) + g_06_09*sin(theta_06 - theta_09))
        struct[0].Gy[16,16] = V_06*(b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09)) + 2*V_09*g_06_09
        struct[0].Gy[16,17] = V_06*V_09*(-b_06_09*cos(theta_06 - theta_09) - g_06_09*sin(theta_06 - theta_09))
        struct[0].Gy[16,43] = -S_n_09/S_base
        struct[0].Gy[17,10] = V_09*(b_06_09*cos(theta_06 - theta_09) + g_06_09*sin(theta_06 - theta_09))
        struct[0].Gy[17,11] = V_06*V_09*(-b_06_09*sin(theta_06 - theta_09) + g_06_09*cos(theta_06 - theta_09))
        struct[0].Gy[17,16] = V_06*(b_06_09*cos(theta_06 - theta_09) + g_06_09*sin(theta_06 - theta_09)) + 2*V_09*(-b_06_09 - bs_06_09/2)
        struct[0].Gy[17,17] = V_06*V_09*(b_06_09*sin(theta_06 - theta_09) - g_06_09*cos(theta_06 - theta_09))
        struct[0].Gy[17,44] = -S_n_09/S_base
        struct[0].Gy[18,8] = V_10*(b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10))
        struct[0].Gy[18,9] = V_05*V_10*(b_05_10*cos(theta_05 - theta_10) + g_05_10*sin(theta_05 - theta_10))
        struct[0].Gy[18,18] = V_05*(b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10)) + 2*V_10*g_05_10
        struct[0].Gy[18,19] = V_05*V_10*(-b_05_10*cos(theta_05 - theta_10) - g_05_10*sin(theta_05 - theta_10))
        struct[0].Gy[18,58] = -S_n_10/S_base
        struct[0].Gy[19,8] = V_10*(b_05_10*cos(theta_05 - theta_10) + g_05_10*sin(theta_05 - theta_10))
        struct[0].Gy[19,9] = V_05*V_10*(-b_05_10*sin(theta_05 - theta_10) + g_05_10*cos(theta_05 - theta_10))
        struct[0].Gy[19,18] = V_05*(b_05_10*cos(theta_05 - theta_10) + g_05_10*sin(theta_05 - theta_10)) + 2*V_10*(-b_05_10 - bs_05_10/2)
        struct[0].Gy[19,19] = V_05*V_10*(b_05_10*sin(theta_05 - theta_10) - g_05_10*cos(theta_05 - theta_10))
        struct[0].Gy[19,59] = -S_n_10/S_base
        struct[0].Gy[20,6] = V_11*(b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11))
        struct[0].Gy[20,7] = V_04*V_11*(b_04_11*cos(theta_04 - theta_11) + g_04_11*sin(theta_04 - theta_11))
        struct[0].Gy[20,20] = V_04*(b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11)) + 2*V_11*g_04_11
        struct[0].Gy[20,21] = V_04*V_11*(-b_04_11*cos(theta_04 - theta_11) - g_04_11*sin(theta_04 - theta_11))
        struct[0].Gy[21,6] = V_11*(b_04_11*cos(theta_04 - theta_11) + g_04_11*sin(theta_04 - theta_11))
        struct[0].Gy[21,7] = V_04*V_11*(-b_04_11*sin(theta_04 - theta_11) + g_04_11*cos(theta_04 - theta_11))
        struct[0].Gy[21,20] = V_04*(b_04_11*cos(theta_04 - theta_11) + g_04_11*sin(theta_04 - theta_11)) + 2*V_11*(-b_04_11 - bs_04_11/2)
        struct[0].Gy[21,21] = V_04*V_11*(b_04_11*sin(theta_04 - theta_11) - g_04_11*cos(theta_04 - theta_11))
        struct[0].Gy[22,4] = V_12*(b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12))
        struct[0].Gy[22,5] = V_03*V_12*(b_03_12*cos(theta_03 - theta_12) + g_03_12*sin(theta_03 - theta_12))
        struct[0].Gy[22,22] = V_03*(b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12)) + 2*V_12*g_03_12
        struct[0].Gy[22,23] = V_03*V_12*(-b_03_12*cos(theta_03 - theta_12) - g_03_12*sin(theta_03 - theta_12))
        struct[0].Gy[23,4] = V_12*(b_03_12*cos(theta_03 - theta_12) + g_03_12*sin(theta_03 - theta_12))
        struct[0].Gy[23,5] = V_03*V_12*(-b_03_12*sin(theta_03 - theta_12) + g_03_12*cos(theta_03 - theta_12))
        struct[0].Gy[23,22] = V_03*(b_03_12*cos(theta_03 - theta_12) + g_03_12*sin(theta_03 - theta_12)) + 2*V_12*(-b_03_12 - bs_03_12/2)
        struct[0].Gy[23,23] = V_03*V_12*(b_03_12*sin(theta_03 - theta_12) - g_03_12*cos(theta_03 - theta_12))
        struct[0].Gy[24,24] = -1
        struct[0].Gy[24,28] = -K_p_08
        struct[0].Gy[24,30] = K_p_08
        struct[0].Gy[25,25] = -1
        struct[0].Gy[25,29] = -K_q_08
        struct[0].Gy[26,14] = -sin(delta_08 - theta_08)
        struct[0].Gy[26,15] = V_08*cos(delta_08 - theta_08)
        struct[0].Gy[26,26] = -R_v_08
        struct[0].Gy[26,27] = X_v_08
        struct[0].Gy[27,14] = -cos(delta_08 - theta_08)
        struct[0].Gy[27,15] = -V_08*sin(delta_08 - theta_08)
        struct[0].Gy[27,25] = 1
        struct[0].Gy[27,26] = -X_v_08
        struct[0].Gy[27,27] = -R_v_08
        struct[0].Gy[28,14] = i_d_08*sin(delta_08 - theta_08) + i_q_08*cos(delta_08 - theta_08)
        struct[0].Gy[28,15] = -V_08*i_d_08*cos(delta_08 - theta_08) + V_08*i_q_08*sin(delta_08 - theta_08)
        struct[0].Gy[28,26] = V_08*sin(delta_08 - theta_08)
        struct[0].Gy[28,27] = V_08*cos(delta_08 - theta_08)
        struct[0].Gy[28,28] = -1
        struct[0].Gy[29,14] = i_d_08*cos(delta_08 - theta_08) - i_q_08*sin(delta_08 - theta_08)
        struct[0].Gy[29,15] = V_08*i_d_08*sin(delta_08 - theta_08) + V_08*i_q_08*cos(delta_08 - theta_08)
        struct[0].Gy[29,26] = V_08*cos(delta_08 - theta_08)
        struct[0].Gy[29,27] = -V_08*sin(delta_08 - theta_08)
        struct[0].Gy[29,29] = -1
        struct[0].Gy[30,28] = 1
        struct[0].Gy[30,30] = -1
        struct[0].Gy[30,31] = -1
        struct[0].Gy[30,32] = 1
        struct[0].Gy[30,37] = K_fpfr_08*Piecewise(np.array([(0, (P_f_min_08 > p_f_08) | (P_f_max_08 < p_f_08)), (1, True)]))
        struct[0].Gy[31,14] = i_d_08*sin(delta_08 - theta_08) + i_q_08*cos(delta_08 - theta_08)
        struct[0].Gy[31,15] = -V_08*i_d_08*cos(delta_08 - theta_08) + V_08*i_q_08*sin(delta_08 - theta_08)
        struct[0].Gy[31,26] = 2*R_s_08*i_d_08 + V_08*sin(delta_08 - theta_08)
        struct[0].Gy[31,27] = 2*R_s_08*i_q_08 + V_08*cos(delta_08 - theta_08)
        struct[0].Gy[31,31] = -1
        struct[0].Gy[32,32] = -1
        struct[0].Gy[32,33] = 2*k_u_08*v_u_08/V_u_max_08**2
        struct[0].Gy[32,34] = -(-v_u_08**2 + v_u_ref_08**2)/V_u_max_08**2
        struct[0].Gy[33,31] = -R_uc_08*S_n_08/(v_u_08 + 0.1)
        struct[0].Gy[33,33] = -R_uc_08*S_n_08*(p_gou_08 - p_t_08)/(v_u_08 + 0.1)**2 - 1
        struct[0].Gy[33,36] = R_uc_08*S_n_08/(v_u_08 + 0.1)
        struct[0].Gy[34,33] = Piecewise(np.array([(0, V_u_min_08 > v_u_08), ((-K_u_0_08 + K_u_max_08)/(-V_u_lt_08 + V_u_min_08), V_u_lt_08 > v_u_08), ((-K_u_0_08 + K_u_max_08)/(-V_u_ht_08 + V_u_max_08), V_u_ht_08 < v_u_08), (0, True)]))
        struct[0].Gy[34,34] = -1
        struct[0].Gy[35,35] = -1
        struct[0].Gy[36,35] = inc_p_gin_08 + p_gin_0_08
        struct[0].Gy[36,36] = -1
        struct[0].Gy[37,24] = -Piecewise(np.array([(1/Droop_08, (omega_08 > 0.5*DB_08 + omega_ref_08) | (omega_08 < -0.5*DB_08 + omega_ref_08)), (0, True)]))
        struct[0].Gy[37,37] = -1
        struct[0].Gy[38,33] = Piecewise(np.array([((-R_lim_08 + R_lim_max_08)/(-V_u_lt_08 + V_u_min_08), V_u_lt_08 > v_u_08), ((-R_lim_08 + R_lim_max_08)/(-V_u_ht_08 + V_u_max_08), V_u_ht_08 < v_u_08), (0, True)]))
        struct[0].Gy[38,38] = -1
        struct[0].Gy[39,39] = -1
        struct[0].Gy[39,43] = -K_p_09
        struct[0].Gy[39,45] = K_p_09
        struct[0].Gy[40,40] = -1
        struct[0].Gy[40,44] = -K_q_09
        struct[0].Gy[41,16] = -sin(delta_09 - theta_09)
        struct[0].Gy[41,17] = V_09*cos(delta_09 - theta_09)
        struct[0].Gy[41,41] = -R_v_09
        struct[0].Gy[41,42] = X_v_09
        struct[0].Gy[42,16] = -cos(delta_09 - theta_09)
        struct[0].Gy[42,17] = -V_09*sin(delta_09 - theta_09)
        struct[0].Gy[42,40] = 1
        struct[0].Gy[42,41] = -X_v_09
        struct[0].Gy[42,42] = -R_v_09
        struct[0].Gy[43,16] = i_d_09*sin(delta_09 - theta_09) + i_q_09*cos(delta_09 - theta_09)
        struct[0].Gy[43,17] = -V_09*i_d_09*cos(delta_09 - theta_09) + V_09*i_q_09*sin(delta_09 - theta_09)
        struct[0].Gy[43,41] = V_09*sin(delta_09 - theta_09)
        struct[0].Gy[43,42] = V_09*cos(delta_09 - theta_09)
        struct[0].Gy[43,43] = -1
        struct[0].Gy[44,16] = i_d_09*cos(delta_09 - theta_09) - i_q_09*sin(delta_09 - theta_09)
        struct[0].Gy[44,17] = V_09*i_d_09*sin(delta_09 - theta_09) + V_09*i_q_09*cos(delta_09 - theta_09)
        struct[0].Gy[44,41] = V_09*cos(delta_09 - theta_09)
        struct[0].Gy[44,42] = -V_09*sin(delta_09 - theta_09)
        struct[0].Gy[44,44] = -1
        struct[0].Gy[45,43] = 1
        struct[0].Gy[45,45] = -1
        struct[0].Gy[45,46] = -1
        struct[0].Gy[45,47] = 1
        struct[0].Gy[45,52] = K_fpfr_09*Piecewise(np.array([(0, (P_f_min_09 > p_f_09) | (P_f_max_09 < p_f_09)), (1, True)]))
        struct[0].Gy[46,16] = i_d_09*sin(delta_09 - theta_09) + i_q_09*cos(delta_09 - theta_09)
        struct[0].Gy[46,17] = -V_09*i_d_09*cos(delta_09 - theta_09) + V_09*i_q_09*sin(delta_09 - theta_09)
        struct[0].Gy[46,41] = 2*R_s_09*i_d_09 + V_09*sin(delta_09 - theta_09)
        struct[0].Gy[46,42] = 2*R_s_09*i_q_09 + V_09*cos(delta_09 - theta_09)
        struct[0].Gy[46,46] = -1
        struct[0].Gy[47,47] = -1
        struct[0].Gy[47,48] = 2*k_u_09*v_u_09/V_u_max_09**2
        struct[0].Gy[47,49] = -(-v_u_09**2 + v_u_ref_09**2)/V_u_max_09**2
        struct[0].Gy[48,46] = -R_uc_09*S_n_09/(v_u_09 + 0.1)
        struct[0].Gy[48,48] = -R_uc_09*S_n_09*(p_gou_09 - p_t_09)/(v_u_09 + 0.1)**2 - 1
        struct[0].Gy[48,51] = R_uc_09*S_n_09/(v_u_09 + 0.1)
        struct[0].Gy[49,48] = Piecewise(np.array([(0, V_u_min_09 > v_u_09), ((-K_u_0_09 + K_u_max_09)/(-V_u_lt_09 + V_u_min_09), V_u_lt_09 > v_u_09), ((-K_u_0_09 + K_u_max_09)/(-V_u_ht_09 + V_u_max_09), V_u_ht_09 < v_u_09), (0, True)]))
        struct[0].Gy[49,49] = -1
        struct[0].Gy[50,50] = -1
        struct[0].Gy[51,50] = inc_p_gin_09 + p_gin_0_09
        struct[0].Gy[51,51] = -1
        struct[0].Gy[52,39] = -Piecewise(np.array([(1/Droop_09, (omega_09 > 0.5*DB_09 + omega_ref_09) | (omega_09 < -0.5*DB_09 + omega_ref_09)), (0, True)]))
        struct[0].Gy[52,52] = -1
        struct[0].Gy[53,48] = Piecewise(np.array([((-R_lim_09 + R_lim_max_09)/(-V_u_lt_09 + V_u_min_09), V_u_lt_09 > v_u_09), ((-R_lim_09 + R_lim_max_09)/(-V_u_ht_09 + V_u_max_09), V_u_ht_09 < v_u_09), (0, True)]))
        struct[0].Gy[53,53] = -1
        struct[0].Gy[54,54] = -1
        struct[0].Gy[54,58] = -K_p_10
        struct[0].Gy[54,60] = K_p_10
        struct[0].Gy[55,55] = -1
        struct[0].Gy[55,59] = -K_q_10
        struct[0].Gy[56,18] = -sin(delta_10 - theta_10)
        struct[0].Gy[56,19] = V_10*cos(delta_10 - theta_10)
        struct[0].Gy[56,56] = -R_v_10
        struct[0].Gy[56,57] = X_v_10
        struct[0].Gy[57,18] = -cos(delta_10 - theta_10)
        struct[0].Gy[57,19] = -V_10*sin(delta_10 - theta_10)
        struct[0].Gy[57,55] = 1
        struct[0].Gy[57,56] = -X_v_10
        struct[0].Gy[57,57] = -R_v_10
        struct[0].Gy[58,18] = i_d_10*sin(delta_10 - theta_10) + i_q_10*cos(delta_10 - theta_10)
        struct[0].Gy[58,19] = -V_10*i_d_10*cos(delta_10 - theta_10) + V_10*i_q_10*sin(delta_10 - theta_10)
        struct[0].Gy[58,56] = V_10*sin(delta_10 - theta_10)
        struct[0].Gy[58,57] = V_10*cos(delta_10 - theta_10)
        struct[0].Gy[58,58] = -1
        struct[0].Gy[59,18] = i_d_10*cos(delta_10 - theta_10) - i_q_10*sin(delta_10 - theta_10)
        struct[0].Gy[59,19] = V_10*i_d_10*sin(delta_10 - theta_10) + V_10*i_q_10*cos(delta_10 - theta_10)
        struct[0].Gy[59,56] = V_10*cos(delta_10 - theta_10)
        struct[0].Gy[59,57] = -V_10*sin(delta_10 - theta_10)
        struct[0].Gy[59,59] = -1
        struct[0].Gy[60,58] = 1
        struct[0].Gy[60,60] = -1
        struct[0].Gy[60,61] = -1
        struct[0].Gy[60,62] = 1
        struct[0].Gy[60,67] = K_fpfr_10*Piecewise(np.array([(0, (P_f_min_10 > p_f_10) | (P_f_max_10 < p_f_10)), (1, True)]))
        struct[0].Gy[61,18] = i_d_10*sin(delta_10 - theta_10) + i_q_10*cos(delta_10 - theta_10)
        struct[0].Gy[61,19] = -V_10*i_d_10*cos(delta_10 - theta_10) + V_10*i_q_10*sin(delta_10 - theta_10)
        struct[0].Gy[61,56] = 2*R_s_10*i_d_10 + V_10*sin(delta_10 - theta_10)
        struct[0].Gy[61,57] = 2*R_s_10*i_q_10 + V_10*cos(delta_10 - theta_10)
        struct[0].Gy[61,61] = -1
        struct[0].Gy[62,62] = -1
        struct[0].Gy[62,63] = 2*k_u_10*v_u_10/V_u_max_10**2
        struct[0].Gy[62,64] = -(-v_u_10**2 + v_u_ref_10**2)/V_u_max_10**2
        struct[0].Gy[63,61] = -R_uc_10*S_n_10/(v_u_10 + 0.1)
        struct[0].Gy[63,63] = -R_uc_10*S_n_10*(p_gou_10 - p_t_10)/(v_u_10 + 0.1)**2 - 1
        struct[0].Gy[63,66] = R_uc_10*S_n_10/(v_u_10 + 0.1)
        struct[0].Gy[64,63] = Piecewise(np.array([(0, V_u_min_10 > v_u_10), ((-K_u_0_10 + K_u_max_10)/(-V_u_lt_10 + V_u_min_10), V_u_lt_10 > v_u_10), ((-K_u_0_10 + K_u_max_10)/(-V_u_ht_10 + V_u_max_10), V_u_ht_10 < v_u_10), (0, True)]))
        struct[0].Gy[64,64] = -1
        struct[0].Gy[65,65] = -1
        struct[0].Gy[66,65] = inc_p_gin_10 + p_gin_0_10
        struct[0].Gy[66,66] = -1
        struct[0].Gy[67,54] = -Piecewise(np.array([(1/Droop_10, (omega_10 > 0.5*DB_10 + omega_ref_10) | (omega_10 < -0.5*DB_10 + omega_ref_10)), (0, True)]))
        struct[0].Gy[67,67] = -1
        struct[0].Gy[68,63] = Piecewise(np.array([((-R_lim_10 + R_lim_max_10)/(-V_u_lt_10 + V_u_min_10), V_u_lt_10 > v_u_10), ((-R_lim_10 + R_lim_max_10)/(-V_u_ht_10 + V_u_max_10), V_u_ht_10 < v_u_10), (0, True)]))
        struct[0].Gy[68,68] = -1
        struct[0].Gy[69,69] = -1
        struct[0].Gy[70,0] = -sin(delta_01 - theta_01)
        struct[0].Gy[70,1] = V_01*cos(delta_01 - theta_01)
        struct[0].Gy[70,70] = -R_v_01
        struct[0].Gy[70,71] = X_v_01
        struct[0].Gy[71,0] = -cos(delta_01 - theta_01)
        struct[0].Gy[71,1] = -V_01*sin(delta_01 - theta_01)
        struct[0].Gy[71,70] = -X_v_01
        struct[0].Gy[71,71] = -R_v_01
        struct[0].Gy[72,0] = i_d_01*sin(delta_01 - theta_01) + i_q_01*cos(delta_01 - theta_01)
        struct[0].Gy[72,1] = -V_01*i_d_01*cos(delta_01 - theta_01) + V_01*i_q_01*sin(delta_01 - theta_01)
        struct[0].Gy[72,70] = V_01*sin(delta_01 - theta_01)
        struct[0].Gy[72,71] = V_01*cos(delta_01 - theta_01)
        struct[0].Gy[72,72] = -1
        struct[0].Gy[73,0] = i_d_01*cos(delta_01 - theta_01) - i_q_01*sin(delta_01 - theta_01)
        struct[0].Gy[73,1] = V_01*i_d_01*sin(delta_01 - theta_01) + V_01*i_q_01*cos(delta_01 - theta_01)
        struct[0].Gy[73,70] = V_01*cos(delta_01 - theta_01)
        struct[0].Gy[73,71] = -V_01*sin(delta_01 - theta_01)
        struct[0].Gy[73,73] = -1
        struct[0].Gy[74,24] = S_n_08*T_p_08/(2*K_p_08*(1000000.0*S_n_01 + S_n_10*T_p_10/(2*K_p_10) + S_n_09*T_p_09/(2*K_p_09) + S_n_08*T_p_08/(2*K_p_08)))
        struct[0].Gy[74,39] = S_n_09*T_p_09/(2*K_p_09*(1000000.0*S_n_01 + S_n_10*T_p_10/(2*K_p_10) + S_n_09*T_p_09/(2*K_p_09) + S_n_08*T_p_08/(2*K_p_08)))
        struct[0].Gy[74,54] = S_n_10*T_p_10/(2*K_p_10*(1000000.0*S_n_01 + S_n_10*T_p_10/(2*K_p_10) + S_n_09*T_p_09/(2*K_p_09) + S_n_08*T_p_08/(2*K_p_08)))
        struct[0].Gy[74,69] = 1000000.0*S_n_01/(1000000.0*S_n_01 + S_n_10*T_p_10/(2*K_p_10) + S_n_09*T_p_09/(2*K_p_09) + S_n_08*T_p_08/(2*K_p_08))
        struct[0].Gy[74,74] = -1
        struct[0].Gy[75,74] = -K_p_agc
        struct[0].Gy[75,75] = -1

        struct[0].Gu[0,0] = -1/S_base
        struct[0].Gu[1,1] = -1/S_base
        struct[0].Gu[2,2] = -1/S_base
        struct[0].Gu[3,3] = -1/S_base
        struct[0].Gu[4,4] = -1/S_base
        struct[0].Gu[5,5] = -1/S_base
        struct[0].Gu[6,6] = -1/S_base
        struct[0].Gu[7,7] = -1/S_base
        struct[0].Gu[8,8] = -1/S_base
        struct[0].Gu[9,9] = -1/S_base
        struct[0].Gu[10,10] = -1/S_base
        struct[0].Gu[11,11] = -1/S_base
        struct[0].Gu[12,12] = -1/S_base
        struct[0].Gu[13,13] = -1/S_base
        struct[0].Gu[14,14] = -1/S_base
        struct[0].Gu[15,15] = -1/S_base
        struct[0].Gu[16,16] = -1/S_base
        struct[0].Gu[17,17] = -1/S_base
        struct[0].Gu[18,18] = -1/S_base
        struct[0].Gu[19,19] = -1/S_base
        struct[0].Gu[20,20] = -1/S_base
        struct[0].Gu[21,21] = -1/S_base
        struct[0].Gu[22,22] = -1/S_base
        struct[0].Gu[23,23] = -1/S_base
        struct[0].Gu[25,24] = K_q_08
        struct[0].Gu[32,25] = -2*k_u_08*v_u_ref_08/V_u_max_08**2
        struct[0].Gu[36,27] = k_cur_sat_08
        struct[0].Gu[37,26] = -Piecewise(np.array([(-1/Droop_08, (omega_08 > 0.5*DB_08 + omega_ref_08) | (omega_08 < -0.5*DB_08 + omega_ref_08)), (0, True)]))
        struct[0].Gu[40,30] = K_q_09
        struct[0].Gu[47,31] = -2*k_u_09*v_u_ref_09/V_u_max_09**2
        struct[0].Gu[51,33] = k_cur_sat_09
        struct[0].Gu[52,32] = -Piecewise(np.array([(-1/Droop_09, (omega_09 > 0.5*DB_09 + omega_ref_09) | (omega_09 < -0.5*DB_09 + omega_ref_09)), (0, True)]))
        struct[0].Gu[55,36] = K_q_10
        struct[0].Gu[62,37] = -2*k_u_10*v_u_ref_10/V_u_max_10**2
        struct[0].Gu[66,39] = k_cur_sat_10
        struct[0].Gu[67,38] = -Piecewise(np.array([(-1/Droop_10, (omega_10 > 0.5*DB_10 + omega_ref_10) | (omega_10 < -0.5*DB_10 + omega_ref_10)), (0, True)]))
        struct[0].Gu[69,44] = 1
        struct[0].Gu[71,43] = 1





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
    Fx_ini_rows = [4, 5, 5, 6, 11, 12, 12, 13, 18, 19, 19, 20, 21, 22]

    Fx_ini_cols = [4, 5, 6, 6, 11, 12, 13, 13, 18, 19, 20, 20, 21, 22]

    Fy_ini_rows = [0, 0, 1, 1, 2, 3, 3, 3, 4, 4, 5, 7, 7, 8, 8, 9, 10, 10, 10, 11, 11, 12, 14, 14, 15, 15, 16, 17, 17, 17, 18, 18, 19, 21, 21, 23]

    Fy_ini_cols = [24, 74, 28, 30, 29, 31, 33, 36, 36, 38, 37, 39, 74, 43, 45, 44, 46, 48, 51, 51, 53, 52, 54, 74, 58, 60, 59, 61, 63, 66, 66, 68, 67, 69, 74, 74]

    Gx_ini_rows = [24, 25, 26, 27, 28, 29, 30, 31, 33, 35, 36, 39, 40, 41, 42, 43, 44, 45, 46, 48, 50, 51, 54, 55, 56, 57, 58, 59, 60, 61, 63, 65, 66, 69, 70, 71, 72, 73, 75]

    Gx_ini_cols = [1, 2, 0, 0, 0, 0, 4, 0, 3, 5, 6, 8, 9, 7, 7, 7, 7, 11, 7, 10, 12, 13, 15, 16, 14, 14, 14, 14, 18, 14, 17, 19, 20, 22, 21, 21, 21, 21, 23]

    Gy_ini_rows = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23, 24, 24, 24, 25, 25, 26, 26, 26, 26, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 32, 32, 32, 33, 33, 33, 34, 34, 35, 36, 36, 37, 37, 38, 38, 39, 39, 39, 40, 40, 41, 41, 41, 41, 42, 42, 42, 42, 42, 43, 43, 43, 43, 43, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 46, 46, 46, 46, 46, 47, 47, 47, 48, 48, 48, 49, 49, 50, 51, 51, 52, 52, 53, 53, 54, 54, 54, 55, 55, 56, 56, 56, 56, 57, 57, 57, 57, 57, 58, 58, 58, 58, 58, 59, 59, 59, 59, 59, 60, 60, 60, 60, 60, 61, 61, 61, 61, 61, 62, 62, 62, 63, 63, 63, 64, 64, 65, 66, 66, 67, 67, 68, 68, 69, 70, 70, 70, 70, 71, 71, 71, 71, 72, 72, 72, 72, 72, 73, 73, 73, 73, 73, 74, 74, 74, 74, 74, 75, 75]

    Gy_ini_cols = [0, 1, 2, 3, 72, 0, 1, 2, 3, 73, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 2, 3, 4, 5, 6, 7, 22, 23, 2, 3, 4, 5, 6, 7, 22, 23, 4, 5, 6, 7, 8, 9, 20, 21, 4, 5, 6, 7, 8, 9, 20, 21, 6, 7, 8, 9, 10, 11, 18, 19, 6, 7, 8, 9, 10, 11, 18, 19, 8, 9, 10, 11, 12, 13, 16, 17, 8, 9, 10, 11, 12, 13, 16, 17, 10, 11, 12, 13, 14, 15, 10, 11, 12, 13, 14, 15, 12, 13, 14, 15, 28, 12, 13, 14, 15, 29, 10, 11, 16, 17, 43, 10, 11, 16, 17, 44, 8, 9, 18, 19, 58, 8, 9, 18, 19, 59, 6, 7, 20, 21, 6, 7, 20, 21, 4, 5, 22, 23, 4, 5, 22, 23, 24, 28, 30, 25, 29, 14, 15, 26, 27, 14, 15, 25, 26, 27, 14, 15, 26, 27, 28, 14, 15, 26, 27, 29, 28, 30, 31, 32, 37, 14, 15, 26, 27, 31, 32, 33, 34, 31, 33, 36, 33, 34, 35, 35, 36, 24, 37, 33, 38, 39, 43, 45, 40, 44, 16, 17, 41, 42, 16, 17, 40, 41, 42, 16, 17, 41, 42, 43, 16, 17, 41, 42, 44, 43, 45, 46, 47, 52, 16, 17, 41, 42, 46, 47, 48, 49, 46, 48, 51, 48, 49, 50, 50, 51, 39, 52, 48, 53, 54, 58, 60, 55, 59, 18, 19, 56, 57, 18, 19, 55, 56, 57, 18, 19, 56, 57, 58, 18, 19, 56, 57, 59, 58, 60, 61, 62, 67, 18, 19, 56, 57, 61, 62, 63, 64, 61, 63, 66, 63, 64, 65, 65, 66, 54, 67, 63, 68, 69, 0, 1, 70, 71, 0, 1, 70, 71, 0, 1, 70, 71, 72, 0, 1, 70, 71, 73, 24, 39, 54, 69, 74, 74, 75]

    return Fx_ini_rows,Fx_ini_cols,Fy_ini_rows,Fy_ini_cols,Gx_ini_rows,Gx_ini_cols,Gy_ini_rows,Gy_ini_cols