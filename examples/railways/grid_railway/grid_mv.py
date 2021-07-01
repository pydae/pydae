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


class grid_mv_class: 

    def __init__(self): 

        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 1
        self.N_y = 170 
        self.N_z = 41 
        self.N_store = 10000 
        self.params_list = [] 
        self.params_values_list  = [] 
        self.inputs_ini_list = ['v_B1_a_r', 'v_B1_a_i', 'v_B1_b_r', 'v_B1_b_i', 'v_B1_c_r', 'v_B1_c_i', 'v_B7_a_r', 'v_B7_a_i', 'v_B7_b_r', 'v_B7_b_i', 'v_B7_c_r', 'v_B7_c_i', 'i_B2lv_n_r', 'i_B2lv_n_i', 'i_B3lv_n_r', 'i_B3lv_n_i', 'i_B4lv_n_r', 'i_B4lv_n_i', 'i_B5lv_n_r', 'i_B5lv_n_i', 'i_B6lv_n_r', 'i_B6lv_n_i', 'i_B2_a_r', 'i_B2_a_i', 'i_B2_b_r', 'i_B2_b_i', 'i_B2_c_r', 'i_B2_c_i', 'i_B3_a_r', 'i_B3_a_i', 'i_B3_b_r', 'i_B3_b_i', 'i_B3_c_r', 'i_B3_c_i', 'i_B4_a_r', 'i_B4_a_i', 'i_B4_b_r', 'i_B4_b_i', 'i_B4_c_r', 'i_B4_c_i', 'i_B5_a_r', 'i_B5_a_i', 'i_B5_b_r', 'i_B5_b_i', 'i_B5_c_r', 'i_B5_c_i', 'i_B6_a_r', 'i_B6_a_i', 'i_B6_b_r', 'i_B6_b_i', 'i_B6_c_r', 'i_B6_c_i', 'p_B2lv_a', 'q_B2lv_a', 'p_B2lv_b', 'q_B2lv_b', 'p_B2lv_c', 'q_B2lv_c', 'p_B3lv_a', 'q_B3lv_a', 'p_B3lv_b', 'q_B3lv_b', 'p_B3lv_c', 'q_B3lv_c', 'p_B4lv_a', 'q_B4lv_a', 'p_B4lv_b', 'q_B4lv_b', 'p_B4lv_c', 'q_B4lv_c', 'p_B5lv_a', 'q_B5lv_a', 'p_B5lv_b', 'q_B5lv_b', 'p_B5lv_c', 'q_B5lv_c', 'p_B6lv_a', 'q_B6lv_a', 'p_B6lv_b', 'q_B6lv_b', 'p_B6lv_c', 'q_B6lv_c', 'u_dummy'] 
        self.inputs_ini_values_list  = [11547.005383792517, 0.0, -5773.502691896256, -10000.000000000002, -5773.502691896264, 10000.0, 11547.005383792517, 0.0, -5773.502691896256, -10000.000000000002, -5773.502691896264, 10000.0, 5.684341886080802e-14, 1.2505552149377763e-12, -3.979039320256561e-13, 1.2505552149377763e-12, -3.979039320256561e-13, 1.1368683772161603e-12, -8.526512829121202e-13, 3.410605131648481e-13, -2.8421709430404007e-13, 4.547473508864641e-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -666000.0014855354, -322558.5103381794, -666000.0014855355, -322558.5103381794, -666000.0014855355, -322558.5103381795, -666000.0023892743, -322558.50336306763, -666000.0023892743, -322558.5033630677, -666000.0023892743, -322558.5033630677, -666000.0029156083, -322558.4994660576, -666000.0029156083, -322558.49946605746, -666000.0029156085, -322558.4994660576, -666000.0012089293, -322558.51338830066, -666000.0012089293, -322558.51338830066, -666000.0012089293, -322558.51338830055, -666000.000925957, -322558.51542340475, -666000.0009259569, -322558.51542340487, -666000.0009259569, -322558.5154234048, 1.0] 
        self.inputs_run_list = ['v_B1_a_r', 'v_B1_a_i', 'v_B1_b_r', 'v_B1_b_i', 'v_B1_c_r', 'v_B1_c_i', 'v_B7_a_r', 'v_B7_a_i', 'v_B7_b_r', 'v_B7_b_i', 'v_B7_c_r', 'v_B7_c_i', 'i_B2lv_n_r', 'i_B2lv_n_i', 'i_B3lv_n_r', 'i_B3lv_n_i', 'i_B4lv_n_r', 'i_B4lv_n_i', 'i_B5lv_n_r', 'i_B5lv_n_i', 'i_B6lv_n_r', 'i_B6lv_n_i', 'i_B2_a_r', 'i_B2_a_i', 'i_B2_b_r', 'i_B2_b_i', 'i_B2_c_r', 'i_B2_c_i', 'i_B3_a_r', 'i_B3_a_i', 'i_B3_b_r', 'i_B3_b_i', 'i_B3_c_r', 'i_B3_c_i', 'i_B4_a_r', 'i_B4_a_i', 'i_B4_b_r', 'i_B4_b_i', 'i_B4_c_r', 'i_B4_c_i', 'i_B5_a_r', 'i_B5_a_i', 'i_B5_b_r', 'i_B5_b_i', 'i_B5_c_r', 'i_B5_c_i', 'i_B6_a_r', 'i_B6_a_i', 'i_B6_b_r', 'i_B6_b_i', 'i_B6_c_r', 'i_B6_c_i', 'p_B2lv_a', 'q_B2lv_a', 'p_B2lv_b', 'q_B2lv_b', 'p_B2lv_c', 'q_B2lv_c', 'p_B3lv_a', 'q_B3lv_a', 'p_B3lv_b', 'q_B3lv_b', 'p_B3lv_c', 'q_B3lv_c', 'p_B4lv_a', 'q_B4lv_a', 'p_B4lv_b', 'q_B4lv_b', 'p_B4lv_c', 'q_B4lv_c', 'p_B5lv_a', 'q_B5lv_a', 'p_B5lv_b', 'q_B5lv_b', 'p_B5lv_c', 'q_B5lv_c', 'p_B6lv_a', 'q_B6lv_a', 'p_B6lv_b', 'q_B6lv_b', 'p_B6lv_c', 'q_B6lv_c', 'u_dummy'] 
        self.inputs_run_values_list = [11547.005383792517, 0.0, -5773.502691896256, -10000.000000000002, -5773.502691896264, 10000.0, 11547.005383792517, 0.0, -5773.502691896256, -10000.000000000002, -5773.502691896264, 10000.0, 5.684341886080802e-14, 1.2505552149377763e-12, -3.979039320256561e-13, 1.2505552149377763e-12, -3.979039320256561e-13, 1.1368683772161603e-12, -8.526512829121202e-13, 3.410605131648481e-13, -2.8421709430404007e-13, 4.547473508864641e-13, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -666000.0014855354, -322558.5103381794, -666000.0014855355, -322558.5103381794, -666000.0014855355, -322558.5103381795, -666000.0023892743, -322558.50336306763, -666000.0023892743, -322558.5033630677, -666000.0023892743, -322558.5033630677, -666000.0029156083, -322558.4994660576, -666000.0029156083, -322558.49946605746, -666000.0029156085, -322558.4994660576, -666000.0012089293, -322558.51338830066, -666000.0012089293, -322558.51338830066, -666000.0012089293, -322558.51338830055, -666000.000925957, -322558.51542340475, -666000.0009259569, -322558.51542340487, -666000.0009259569, -322558.5154234048, 1.0] 
        self.outputs_list = ['v_B1_a_m', 'v_B1_b_m', 'v_B1_c_m', 'v_B7_a_m', 'v_B7_b_m', 'v_B7_c_m', 'v_B2lv_a_m', 'v_B2lv_b_m', 'v_B2lv_c_m', 'v_B2lv_n_m', 'v_B3lv_a_m', 'v_B3lv_b_m', 'v_B3lv_c_m', 'v_B3lv_n_m', 'v_B4lv_a_m', 'v_B4lv_b_m', 'v_B4lv_c_m', 'v_B4lv_n_m', 'v_B5lv_a_m', 'v_B5lv_b_m', 'v_B5lv_c_m', 'v_B5lv_n_m', 'v_B6lv_a_m', 'v_B6lv_b_m', 'v_B6lv_c_m', 'v_B6lv_n_m', 'v_B2_a_m', 'v_B2_b_m', 'v_B2_c_m', 'v_B3_a_m', 'v_B3_b_m', 'v_B3_c_m', 'v_B4_a_m', 'v_B4_b_m', 'v_B4_c_m', 'v_B5_a_m', 'v_B5_b_m', 'v_B5_c_m', 'v_B6_a_m', 'v_B6_b_m', 'v_B6_c_m'] 
        self.x_list = ['x_dummy'] 
        self.y_run_list = ['v_B2lv_a_r', 'v_B2lv_a_i', 'v_B2lv_b_r', 'v_B2lv_b_i', 'v_B2lv_c_r', 'v_B2lv_c_i', 'v_B2lv_n_r', 'v_B2lv_n_i', 'v_B3lv_a_r', 'v_B3lv_a_i', 'v_B3lv_b_r', 'v_B3lv_b_i', 'v_B3lv_c_r', 'v_B3lv_c_i', 'v_B3lv_n_r', 'v_B3lv_n_i', 'v_B4lv_a_r', 'v_B4lv_a_i', 'v_B4lv_b_r', 'v_B4lv_b_i', 'v_B4lv_c_r', 'v_B4lv_c_i', 'v_B4lv_n_r', 'v_B4lv_n_i', 'v_B5lv_a_r', 'v_B5lv_a_i', 'v_B5lv_b_r', 'v_B5lv_b_i', 'v_B5lv_c_r', 'v_B5lv_c_i', 'v_B5lv_n_r', 'v_B5lv_n_i', 'v_B6lv_a_r', 'v_B6lv_a_i', 'v_B6lv_b_r', 'v_B6lv_b_i', 'v_B6lv_c_r', 'v_B6lv_c_i', 'v_B6lv_n_r', 'v_B6lv_n_i', 'v_B2_a_r', 'v_B2_a_i', 'v_B2_b_r', 'v_B2_b_i', 'v_B2_c_r', 'v_B2_c_i', 'v_B3_a_r', 'v_B3_a_i', 'v_B3_b_r', 'v_B3_b_i', 'v_B3_c_r', 'v_B3_c_i', 'v_B4_a_r', 'v_B4_a_i', 'v_B4_b_r', 'v_B4_b_i', 'v_B4_c_r', 'v_B4_c_i', 'v_B5_a_r', 'v_B5_a_i', 'v_B5_b_r', 'v_B5_b_i', 'v_B5_c_r', 'v_B5_c_i', 'v_B6_a_r', 'v_B6_a_i', 'v_B6_b_r', 'v_B6_b_i', 'v_B6_c_r', 'v_B6_c_i', 'i_t_B2_B2lv_a_r', 'i_t_B2_B2lv_a_i', 'i_t_B2_B2lv_b_r', 'i_t_B2_B2lv_b_i', 'i_t_B2_B2lv_c_r', 'i_t_B2_B2lv_c_i', 'i_t_B3_B3lv_a_r', 'i_t_B3_B3lv_a_i', 'i_t_B3_B3lv_b_r', 'i_t_B3_B3lv_b_i', 'i_t_B3_B3lv_c_r', 'i_t_B3_B3lv_c_i', 'i_t_B4_B4lv_a_r', 'i_t_B4_B4lv_a_i', 'i_t_B4_B4lv_b_r', 'i_t_B4_B4lv_b_i', 'i_t_B4_B4lv_c_r', 'i_t_B4_B4lv_c_i', 'i_t_B5_B5lv_a_r', 'i_t_B5_B5lv_a_i', 'i_t_B5_B5lv_b_r', 'i_t_B5_B5lv_b_i', 'i_t_B5_B5lv_c_r', 'i_t_B5_B5lv_c_i', 'i_t_B6_B6lv_a_r', 'i_t_B6_B6lv_a_i', 'i_t_B6_B6lv_b_r', 'i_t_B6_B6lv_b_i', 'i_t_B6_B6lv_c_r', 'i_t_B6_B6lv_c_i', 'i_l_B1_B2_a_r', 'i_l_B1_B2_a_i', 'i_l_B1_B2_b_r', 'i_l_B1_B2_b_i', 'i_l_B1_B2_c_r', 'i_l_B1_B2_c_i', 'i_l_B2_B3_a_r', 'i_l_B2_B3_a_i', 'i_l_B2_B3_b_r', 'i_l_B2_B3_b_i', 'i_l_B2_B3_c_r', 'i_l_B2_B3_c_i', 'i_l_B3_B4_a_r', 'i_l_B3_B4_a_i', 'i_l_B3_B4_b_r', 'i_l_B3_B4_b_i', 'i_l_B3_B4_c_r', 'i_l_B3_B4_c_i', 'i_l_B5_B6_a_r', 'i_l_B5_B6_a_i', 'i_l_B5_B6_b_r', 'i_l_B5_B6_b_i', 'i_l_B5_B6_c_r', 'i_l_B5_B6_c_i', 'i_l_B6_B7_a_r', 'i_l_B6_B7_a_i', 'i_l_B6_B7_b_r', 'i_l_B6_B7_b_i', 'i_l_B6_B7_c_r', 'i_l_B6_B7_c_i', 'i_load_B2lv_a_r', 'i_load_B2lv_a_i', 'i_load_B2lv_b_r', 'i_load_B2lv_b_i', 'i_load_B2lv_c_r', 'i_load_B2lv_c_i', 'i_load_B2lv_n_r', 'i_load_B2lv_n_i', 'i_load_B3lv_a_r', 'i_load_B3lv_a_i', 'i_load_B3lv_b_r', 'i_load_B3lv_b_i', 'i_load_B3lv_c_r', 'i_load_B3lv_c_i', 'i_load_B3lv_n_r', 'i_load_B3lv_n_i', 'i_load_B4lv_a_r', 'i_load_B4lv_a_i', 'i_load_B4lv_b_r', 'i_load_B4lv_b_i', 'i_load_B4lv_c_r', 'i_load_B4lv_c_i', 'i_load_B4lv_n_r', 'i_load_B4lv_n_i', 'i_load_B5lv_a_r', 'i_load_B5lv_a_i', 'i_load_B5lv_b_r', 'i_load_B5lv_b_i', 'i_load_B5lv_c_r', 'i_load_B5lv_c_i', 'i_load_B5lv_n_r', 'i_load_B5lv_n_i', 'i_load_B6lv_a_r', 'i_load_B6lv_a_i', 'i_load_B6lv_b_r', 'i_load_B6lv_b_i', 'i_load_B6lv_c_r', 'i_load_B6lv_c_i', 'i_load_B6lv_n_r', 'i_load_B6lv_n_i'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['v_B2lv_a_r', 'v_B2lv_a_i', 'v_B2lv_b_r', 'v_B2lv_b_i', 'v_B2lv_c_r', 'v_B2lv_c_i', 'v_B2lv_n_r', 'v_B2lv_n_i', 'v_B3lv_a_r', 'v_B3lv_a_i', 'v_B3lv_b_r', 'v_B3lv_b_i', 'v_B3lv_c_r', 'v_B3lv_c_i', 'v_B3lv_n_r', 'v_B3lv_n_i', 'v_B4lv_a_r', 'v_B4lv_a_i', 'v_B4lv_b_r', 'v_B4lv_b_i', 'v_B4lv_c_r', 'v_B4lv_c_i', 'v_B4lv_n_r', 'v_B4lv_n_i', 'v_B5lv_a_r', 'v_B5lv_a_i', 'v_B5lv_b_r', 'v_B5lv_b_i', 'v_B5lv_c_r', 'v_B5lv_c_i', 'v_B5lv_n_r', 'v_B5lv_n_i', 'v_B6lv_a_r', 'v_B6lv_a_i', 'v_B6lv_b_r', 'v_B6lv_b_i', 'v_B6lv_c_r', 'v_B6lv_c_i', 'v_B6lv_n_r', 'v_B6lv_n_i', 'v_B2_a_r', 'v_B2_a_i', 'v_B2_b_r', 'v_B2_b_i', 'v_B2_c_r', 'v_B2_c_i', 'v_B3_a_r', 'v_B3_a_i', 'v_B3_b_r', 'v_B3_b_i', 'v_B3_c_r', 'v_B3_c_i', 'v_B4_a_r', 'v_B4_a_i', 'v_B4_b_r', 'v_B4_b_i', 'v_B4_c_r', 'v_B4_c_i', 'v_B5_a_r', 'v_B5_a_i', 'v_B5_b_r', 'v_B5_b_i', 'v_B5_c_r', 'v_B5_c_i', 'v_B6_a_r', 'v_B6_a_i', 'v_B6_b_r', 'v_B6_b_i', 'v_B6_c_r', 'v_B6_c_i', 'i_t_B2_B2lv_a_r', 'i_t_B2_B2lv_a_i', 'i_t_B2_B2lv_b_r', 'i_t_B2_B2lv_b_i', 'i_t_B2_B2lv_c_r', 'i_t_B2_B2lv_c_i', 'i_t_B3_B3lv_a_r', 'i_t_B3_B3lv_a_i', 'i_t_B3_B3lv_b_r', 'i_t_B3_B3lv_b_i', 'i_t_B3_B3lv_c_r', 'i_t_B3_B3lv_c_i', 'i_t_B4_B4lv_a_r', 'i_t_B4_B4lv_a_i', 'i_t_B4_B4lv_b_r', 'i_t_B4_B4lv_b_i', 'i_t_B4_B4lv_c_r', 'i_t_B4_B4lv_c_i', 'i_t_B5_B5lv_a_r', 'i_t_B5_B5lv_a_i', 'i_t_B5_B5lv_b_r', 'i_t_B5_B5lv_b_i', 'i_t_B5_B5lv_c_r', 'i_t_B5_B5lv_c_i', 'i_t_B6_B6lv_a_r', 'i_t_B6_B6lv_a_i', 'i_t_B6_B6lv_b_r', 'i_t_B6_B6lv_b_i', 'i_t_B6_B6lv_c_r', 'i_t_B6_B6lv_c_i', 'i_l_B1_B2_a_r', 'i_l_B1_B2_a_i', 'i_l_B1_B2_b_r', 'i_l_B1_B2_b_i', 'i_l_B1_B2_c_r', 'i_l_B1_B2_c_i', 'i_l_B2_B3_a_r', 'i_l_B2_B3_a_i', 'i_l_B2_B3_b_r', 'i_l_B2_B3_b_i', 'i_l_B2_B3_c_r', 'i_l_B2_B3_c_i', 'i_l_B3_B4_a_r', 'i_l_B3_B4_a_i', 'i_l_B3_B4_b_r', 'i_l_B3_B4_b_i', 'i_l_B3_B4_c_r', 'i_l_B3_B4_c_i', 'i_l_B5_B6_a_r', 'i_l_B5_B6_a_i', 'i_l_B5_B6_b_r', 'i_l_B5_B6_b_i', 'i_l_B5_B6_c_r', 'i_l_B5_B6_c_i', 'i_l_B6_B7_a_r', 'i_l_B6_B7_a_i', 'i_l_B6_B7_b_r', 'i_l_B6_B7_b_i', 'i_l_B6_B7_c_r', 'i_l_B6_B7_c_i', 'i_load_B2lv_a_r', 'i_load_B2lv_a_i', 'i_load_B2lv_b_r', 'i_load_B2lv_b_i', 'i_load_B2lv_c_r', 'i_load_B2lv_c_i', 'i_load_B2lv_n_r', 'i_load_B2lv_n_i', 'i_load_B3lv_a_r', 'i_load_B3lv_a_i', 'i_load_B3lv_b_r', 'i_load_B3lv_b_i', 'i_load_B3lv_c_r', 'i_load_B3lv_c_i', 'i_load_B3lv_n_r', 'i_load_B3lv_n_i', 'i_load_B4lv_a_r', 'i_load_B4lv_a_i', 'i_load_B4lv_b_r', 'i_load_B4lv_b_i', 'i_load_B4lv_c_r', 'i_load_B4lv_c_i', 'i_load_B4lv_n_r', 'i_load_B4lv_n_i', 'i_load_B5lv_a_r', 'i_load_B5lv_a_i', 'i_load_B5lv_b_r', 'i_load_B5lv_b_i', 'i_load_B5lv_c_r', 'i_load_B5lv_c_i', 'i_load_B5lv_n_r', 'i_load_B5lv_n_i', 'i_load_B6lv_a_r', 'i_load_B6lv_a_i', 'i_load_B6lv_b_r', 'i_load_B6lv_b_i', 'i_load_B6lv_c_r', 'i_load_B6lv_c_i', 'i_load_B6lv_n_r', 'i_load_B6lv_n_i'] 
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
    
    # Inputs:
    v_B1_a_r = struct[0].v_B1_a_r
    v_B1_a_i = struct[0].v_B1_a_i
    v_B1_b_r = struct[0].v_B1_b_r
    v_B1_b_i = struct[0].v_B1_b_i
    v_B1_c_r = struct[0].v_B1_c_r
    v_B1_c_i = struct[0].v_B1_c_i
    v_B7_a_r = struct[0].v_B7_a_r
    v_B7_a_i = struct[0].v_B7_a_i
    v_B7_b_r = struct[0].v_B7_b_r
    v_B7_b_i = struct[0].v_B7_b_i
    v_B7_c_r = struct[0].v_B7_c_r
    v_B7_c_i = struct[0].v_B7_c_i
    i_B2lv_n_r = struct[0].i_B2lv_n_r
    i_B2lv_n_i = struct[0].i_B2lv_n_i
    i_B3lv_n_r = struct[0].i_B3lv_n_r
    i_B3lv_n_i = struct[0].i_B3lv_n_i
    i_B4lv_n_r = struct[0].i_B4lv_n_r
    i_B4lv_n_i = struct[0].i_B4lv_n_i
    i_B5lv_n_r = struct[0].i_B5lv_n_r
    i_B5lv_n_i = struct[0].i_B5lv_n_i
    i_B6lv_n_r = struct[0].i_B6lv_n_r
    i_B6lv_n_i = struct[0].i_B6lv_n_i
    i_B2_a_r = struct[0].i_B2_a_r
    i_B2_a_i = struct[0].i_B2_a_i
    i_B2_b_r = struct[0].i_B2_b_r
    i_B2_b_i = struct[0].i_B2_b_i
    i_B2_c_r = struct[0].i_B2_c_r
    i_B2_c_i = struct[0].i_B2_c_i
    i_B3_a_r = struct[0].i_B3_a_r
    i_B3_a_i = struct[0].i_B3_a_i
    i_B3_b_r = struct[0].i_B3_b_r
    i_B3_b_i = struct[0].i_B3_b_i
    i_B3_c_r = struct[0].i_B3_c_r
    i_B3_c_i = struct[0].i_B3_c_i
    i_B4_a_r = struct[0].i_B4_a_r
    i_B4_a_i = struct[0].i_B4_a_i
    i_B4_b_r = struct[0].i_B4_b_r
    i_B4_b_i = struct[0].i_B4_b_i
    i_B4_c_r = struct[0].i_B4_c_r
    i_B4_c_i = struct[0].i_B4_c_i
    i_B5_a_r = struct[0].i_B5_a_r
    i_B5_a_i = struct[0].i_B5_a_i
    i_B5_b_r = struct[0].i_B5_b_r
    i_B5_b_i = struct[0].i_B5_b_i
    i_B5_c_r = struct[0].i_B5_c_r
    i_B5_c_i = struct[0].i_B5_c_i
    i_B6_a_r = struct[0].i_B6_a_r
    i_B6_a_i = struct[0].i_B6_a_i
    i_B6_b_r = struct[0].i_B6_b_r
    i_B6_b_i = struct[0].i_B6_b_i
    i_B6_c_r = struct[0].i_B6_c_r
    i_B6_c_i = struct[0].i_B6_c_i
    p_B2lv_a = struct[0].p_B2lv_a
    q_B2lv_a = struct[0].q_B2lv_a
    p_B2lv_b = struct[0].p_B2lv_b
    q_B2lv_b = struct[0].q_B2lv_b
    p_B2lv_c = struct[0].p_B2lv_c
    q_B2lv_c = struct[0].q_B2lv_c
    p_B3lv_a = struct[0].p_B3lv_a
    q_B3lv_a = struct[0].q_B3lv_a
    p_B3lv_b = struct[0].p_B3lv_b
    q_B3lv_b = struct[0].q_B3lv_b
    p_B3lv_c = struct[0].p_B3lv_c
    q_B3lv_c = struct[0].q_B3lv_c
    p_B4lv_a = struct[0].p_B4lv_a
    q_B4lv_a = struct[0].q_B4lv_a
    p_B4lv_b = struct[0].p_B4lv_b
    q_B4lv_b = struct[0].q_B4lv_b
    p_B4lv_c = struct[0].p_B4lv_c
    q_B4lv_c = struct[0].q_B4lv_c
    p_B5lv_a = struct[0].p_B5lv_a
    q_B5lv_a = struct[0].q_B5lv_a
    p_B5lv_b = struct[0].p_B5lv_b
    q_B5lv_b = struct[0].q_B5lv_b
    p_B5lv_c = struct[0].p_B5lv_c
    q_B5lv_c = struct[0].q_B5lv_c
    p_B6lv_a = struct[0].p_B6lv_a
    q_B6lv_a = struct[0].q_B6lv_a
    p_B6lv_b = struct[0].p_B6lv_b
    q_B6lv_b = struct[0].q_B6lv_b
    p_B6lv_c = struct[0].p_B6lv_c
    q_B6lv_c = struct[0].q_B6lv_c
    u_dummy = struct[0].u_dummy
    
    # Dynamical states:
    x_dummy = struct[0].x[0,0]
    
    # Algebraic states:
    v_B2lv_a_r = struct[0].y_ini[0,0]
    v_B2lv_a_i = struct[0].y_ini[1,0]
    v_B2lv_b_r = struct[0].y_ini[2,0]
    v_B2lv_b_i = struct[0].y_ini[3,0]
    v_B2lv_c_r = struct[0].y_ini[4,0]
    v_B2lv_c_i = struct[0].y_ini[5,0]
    v_B2lv_n_r = struct[0].y_ini[6,0]
    v_B2lv_n_i = struct[0].y_ini[7,0]
    v_B3lv_a_r = struct[0].y_ini[8,0]
    v_B3lv_a_i = struct[0].y_ini[9,0]
    v_B3lv_b_r = struct[0].y_ini[10,0]
    v_B3lv_b_i = struct[0].y_ini[11,0]
    v_B3lv_c_r = struct[0].y_ini[12,0]
    v_B3lv_c_i = struct[0].y_ini[13,0]
    v_B3lv_n_r = struct[0].y_ini[14,0]
    v_B3lv_n_i = struct[0].y_ini[15,0]
    v_B4lv_a_r = struct[0].y_ini[16,0]
    v_B4lv_a_i = struct[0].y_ini[17,0]
    v_B4lv_b_r = struct[0].y_ini[18,0]
    v_B4lv_b_i = struct[0].y_ini[19,0]
    v_B4lv_c_r = struct[0].y_ini[20,0]
    v_B4lv_c_i = struct[0].y_ini[21,0]
    v_B4lv_n_r = struct[0].y_ini[22,0]
    v_B4lv_n_i = struct[0].y_ini[23,0]
    v_B5lv_a_r = struct[0].y_ini[24,0]
    v_B5lv_a_i = struct[0].y_ini[25,0]
    v_B5lv_b_r = struct[0].y_ini[26,0]
    v_B5lv_b_i = struct[0].y_ini[27,0]
    v_B5lv_c_r = struct[0].y_ini[28,0]
    v_B5lv_c_i = struct[0].y_ini[29,0]
    v_B5lv_n_r = struct[0].y_ini[30,0]
    v_B5lv_n_i = struct[0].y_ini[31,0]
    v_B6lv_a_r = struct[0].y_ini[32,0]
    v_B6lv_a_i = struct[0].y_ini[33,0]
    v_B6lv_b_r = struct[0].y_ini[34,0]
    v_B6lv_b_i = struct[0].y_ini[35,0]
    v_B6lv_c_r = struct[0].y_ini[36,0]
    v_B6lv_c_i = struct[0].y_ini[37,0]
    v_B6lv_n_r = struct[0].y_ini[38,0]
    v_B6lv_n_i = struct[0].y_ini[39,0]
    v_B2_a_r = struct[0].y_ini[40,0]
    v_B2_a_i = struct[0].y_ini[41,0]
    v_B2_b_r = struct[0].y_ini[42,0]
    v_B2_b_i = struct[0].y_ini[43,0]
    v_B2_c_r = struct[0].y_ini[44,0]
    v_B2_c_i = struct[0].y_ini[45,0]
    v_B3_a_r = struct[0].y_ini[46,0]
    v_B3_a_i = struct[0].y_ini[47,0]
    v_B3_b_r = struct[0].y_ini[48,0]
    v_B3_b_i = struct[0].y_ini[49,0]
    v_B3_c_r = struct[0].y_ini[50,0]
    v_B3_c_i = struct[0].y_ini[51,0]
    v_B4_a_r = struct[0].y_ini[52,0]
    v_B4_a_i = struct[0].y_ini[53,0]
    v_B4_b_r = struct[0].y_ini[54,0]
    v_B4_b_i = struct[0].y_ini[55,0]
    v_B4_c_r = struct[0].y_ini[56,0]
    v_B4_c_i = struct[0].y_ini[57,0]
    v_B5_a_r = struct[0].y_ini[58,0]
    v_B5_a_i = struct[0].y_ini[59,0]
    v_B5_b_r = struct[0].y_ini[60,0]
    v_B5_b_i = struct[0].y_ini[61,0]
    v_B5_c_r = struct[0].y_ini[62,0]
    v_B5_c_i = struct[0].y_ini[63,0]
    v_B6_a_r = struct[0].y_ini[64,0]
    v_B6_a_i = struct[0].y_ini[65,0]
    v_B6_b_r = struct[0].y_ini[66,0]
    v_B6_b_i = struct[0].y_ini[67,0]
    v_B6_c_r = struct[0].y_ini[68,0]
    v_B6_c_i = struct[0].y_ini[69,0]
    i_t_B2_B2lv_a_r = struct[0].y_ini[70,0]
    i_t_B2_B2lv_a_i = struct[0].y_ini[71,0]
    i_t_B2_B2lv_b_r = struct[0].y_ini[72,0]
    i_t_B2_B2lv_b_i = struct[0].y_ini[73,0]
    i_t_B2_B2lv_c_r = struct[0].y_ini[74,0]
    i_t_B2_B2lv_c_i = struct[0].y_ini[75,0]
    i_t_B3_B3lv_a_r = struct[0].y_ini[76,0]
    i_t_B3_B3lv_a_i = struct[0].y_ini[77,0]
    i_t_B3_B3lv_b_r = struct[0].y_ini[78,0]
    i_t_B3_B3lv_b_i = struct[0].y_ini[79,0]
    i_t_B3_B3lv_c_r = struct[0].y_ini[80,0]
    i_t_B3_B3lv_c_i = struct[0].y_ini[81,0]
    i_t_B4_B4lv_a_r = struct[0].y_ini[82,0]
    i_t_B4_B4lv_a_i = struct[0].y_ini[83,0]
    i_t_B4_B4lv_b_r = struct[0].y_ini[84,0]
    i_t_B4_B4lv_b_i = struct[0].y_ini[85,0]
    i_t_B4_B4lv_c_r = struct[0].y_ini[86,0]
    i_t_B4_B4lv_c_i = struct[0].y_ini[87,0]
    i_t_B5_B5lv_a_r = struct[0].y_ini[88,0]
    i_t_B5_B5lv_a_i = struct[0].y_ini[89,0]
    i_t_B5_B5lv_b_r = struct[0].y_ini[90,0]
    i_t_B5_B5lv_b_i = struct[0].y_ini[91,0]
    i_t_B5_B5lv_c_r = struct[0].y_ini[92,0]
    i_t_B5_B5lv_c_i = struct[0].y_ini[93,0]
    i_t_B6_B6lv_a_r = struct[0].y_ini[94,0]
    i_t_B6_B6lv_a_i = struct[0].y_ini[95,0]
    i_t_B6_B6lv_b_r = struct[0].y_ini[96,0]
    i_t_B6_B6lv_b_i = struct[0].y_ini[97,0]
    i_t_B6_B6lv_c_r = struct[0].y_ini[98,0]
    i_t_B6_B6lv_c_i = struct[0].y_ini[99,0]
    i_l_B1_B2_a_r = struct[0].y_ini[100,0]
    i_l_B1_B2_a_i = struct[0].y_ini[101,0]
    i_l_B1_B2_b_r = struct[0].y_ini[102,0]
    i_l_B1_B2_b_i = struct[0].y_ini[103,0]
    i_l_B1_B2_c_r = struct[0].y_ini[104,0]
    i_l_B1_B2_c_i = struct[0].y_ini[105,0]
    i_l_B2_B3_a_r = struct[0].y_ini[106,0]
    i_l_B2_B3_a_i = struct[0].y_ini[107,0]
    i_l_B2_B3_b_r = struct[0].y_ini[108,0]
    i_l_B2_B3_b_i = struct[0].y_ini[109,0]
    i_l_B2_B3_c_r = struct[0].y_ini[110,0]
    i_l_B2_B3_c_i = struct[0].y_ini[111,0]
    i_l_B3_B4_a_r = struct[0].y_ini[112,0]
    i_l_B3_B4_a_i = struct[0].y_ini[113,0]
    i_l_B3_B4_b_r = struct[0].y_ini[114,0]
    i_l_B3_B4_b_i = struct[0].y_ini[115,0]
    i_l_B3_B4_c_r = struct[0].y_ini[116,0]
    i_l_B3_B4_c_i = struct[0].y_ini[117,0]
    i_l_B5_B6_a_r = struct[0].y_ini[118,0]
    i_l_B5_B6_a_i = struct[0].y_ini[119,0]
    i_l_B5_B6_b_r = struct[0].y_ini[120,0]
    i_l_B5_B6_b_i = struct[0].y_ini[121,0]
    i_l_B5_B6_c_r = struct[0].y_ini[122,0]
    i_l_B5_B6_c_i = struct[0].y_ini[123,0]
    i_l_B6_B7_a_r = struct[0].y_ini[124,0]
    i_l_B6_B7_a_i = struct[0].y_ini[125,0]
    i_l_B6_B7_b_r = struct[0].y_ini[126,0]
    i_l_B6_B7_b_i = struct[0].y_ini[127,0]
    i_l_B6_B7_c_r = struct[0].y_ini[128,0]
    i_l_B6_B7_c_i = struct[0].y_ini[129,0]
    i_load_B2lv_a_r = struct[0].y_ini[130,0]
    i_load_B2lv_a_i = struct[0].y_ini[131,0]
    i_load_B2lv_b_r = struct[0].y_ini[132,0]
    i_load_B2lv_b_i = struct[0].y_ini[133,0]
    i_load_B2lv_c_r = struct[0].y_ini[134,0]
    i_load_B2lv_c_i = struct[0].y_ini[135,0]
    i_load_B2lv_n_r = struct[0].y_ini[136,0]
    i_load_B2lv_n_i = struct[0].y_ini[137,0]
    i_load_B3lv_a_r = struct[0].y_ini[138,0]
    i_load_B3lv_a_i = struct[0].y_ini[139,0]
    i_load_B3lv_b_r = struct[0].y_ini[140,0]
    i_load_B3lv_b_i = struct[0].y_ini[141,0]
    i_load_B3lv_c_r = struct[0].y_ini[142,0]
    i_load_B3lv_c_i = struct[0].y_ini[143,0]
    i_load_B3lv_n_r = struct[0].y_ini[144,0]
    i_load_B3lv_n_i = struct[0].y_ini[145,0]
    i_load_B4lv_a_r = struct[0].y_ini[146,0]
    i_load_B4lv_a_i = struct[0].y_ini[147,0]
    i_load_B4lv_b_r = struct[0].y_ini[148,0]
    i_load_B4lv_b_i = struct[0].y_ini[149,0]
    i_load_B4lv_c_r = struct[0].y_ini[150,0]
    i_load_B4lv_c_i = struct[0].y_ini[151,0]
    i_load_B4lv_n_r = struct[0].y_ini[152,0]
    i_load_B4lv_n_i = struct[0].y_ini[153,0]
    i_load_B5lv_a_r = struct[0].y_ini[154,0]
    i_load_B5lv_a_i = struct[0].y_ini[155,0]
    i_load_B5lv_b_r = struct[0].y_ini[156,0]
    i_load_B5lv_b_i = struct[0].y_ini[157,0]
    i_load_B5lv_c_r = struct[0].y_ini[158,0]
    i_load_B5lv_c_i = struct[0].y_ini[159,0]
    i_load_B5lv_n_r = struct[0].y_ini[160,0]
    i_load_B5lv_n_i = struct[0].y_ini[161,0]
    i_load_B6lv_a_r = struct[0].y_ini[162,0]
    i_load_B6lv_a_i = struct[0].y_ini[163,0]
    i_load_B6lv_b_r = struct[0].y_ini[164,0]
    i_load_B6lv_b_i = struct[0].y_ini[165,0]
    i_load_B6lv_c_r = struct[0].y_ini[166,0]
    i_load_B6lv_c_i = struct[0].y_ini[167,0]
    i_load_B6lv_n_r = struct[0].y_ini[168,0]
    i_load_B6lv_n_i = struct[0].y_ini[169,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = u_dummy - x_dummy
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy_ini) @ np.ascontiguousarray(struct[0].y_ini)

        struct[0].g[40,0] = 0.598820527961361*v_B1_a_i + 1.10755301189314*v_B1_a_r - 0.171091579417532*v_B1_b_i - 0.316443717683753*v_B1_b_r - 0.171091579417532*v_B1_c_i - 0.316443717683753*v_B1_c_r - 1.28353302446119*v_B2_a_i - 2.23667465123725*v_B2_a_r + 0.385473430243205*v_B2_b_i + 0.643671749092996*v_B2_b_r + 0.385473430243205*v_B2_c_i + 0.643671749092997*v_B2_c_r + 0.996212229189942*v_B2lv_a_i + 0.249053057297486*v_B2lv_a_r - 0.996212229189942*v_B2lv_b_i - 0.249053057297486*v_B2lv_b_r + 0.598820527961361*v_B3_a_i + 1.10755301189314*v_B3_a_r - 0.171091579417532*v_B3_b_i - 0.316443717683753*v_B3_b_r - 0.171091579417532*v_B3_c_i - 0.316443717683753*v_B3_c_r
        struct[0].g[41,0] = 1.10755301189314*v_B1_a_i - 0.598820527961361*v_B1_a_r - 0.316443717683753*v_B1_b_i + 0.171091579417532*v_B1_b_r - 0.316443717683753*v_B1_c_i + 0.171091579417532*v_B1_c_r - 2.23667465123725*v_B2_a_i + 1.28353302446119*v_B2_a_r + 0.643671749092996*v_B2_b_i - 0.385473430243205*v_B2_b_r + 0.643671749092997*v_B2_c_i - 0.385473430243205*v_B2_c_r + 0.249053057297486*v_B2lv_a_i - 0.996212229189942*v_B2lv_a_r - 0.249053057297486*v_B2lv_b_i + 0.996212229189942*v_B2lv_b_r + 1.10755301189314*v_B3_a_i - 0.598820527961361*v_B3_a_r - 0.316443717683753*v_B3_b_i + 0.171091579417532*v_B3_b_r - 0.316443717683753*v_B3_c_i + 0.171091579417532*v_B3_c_r
        struct[0].g[42,0] = -0.171091579417532*v_B1_a_i - 0.316443717683753*v_B1_a_r + 0.59882052796136*v_B1_b_i + 1.10755301189314*v_B1_b_r - 0.171091579417531*v_B1_c_i - 0.316443717683753*v_B1_c_r + 0.385473430243205*v_B2_a_i + 0.643671749092996*v_B2_a_r - 1.28353302446119*v_B2_b_i - 2.23667465123725*v_B2_b_r + 0.385473430243204*v_B2_c_i + 0.643671749092997*v_B2_c_r + 0.996212229189942*v_B2lv_b_i + 0.249053057297486*v_B2lv_b_r - 0.996212229189942*v_B2lv_c_i - 0.249053057297486*v_B2lv_c_r - 0.171091579417532*v_B3_a_i - 0.316443717683753*v_B3_a_r + 0.59882052796136*v_B3_b_i + 1.10755301189314*v_B3_b_r - 0.171091579417531*v_B3_c_i - 0.316443717683753*v_B3_c_r
        struct[0].g[43,0] = -0.316443717683753*v_B1_a_i + 0.171091579417532*v_B1_a_r + 1.10755301189314*v_B1_b_i - 0.59882052796136*v_B1_b_r - 0.316443717683753*v_B1_c_i + 0.171091579417531*v_B1_c_r + 0.643671749092996*v_B2_a_i - 0.385473430243205*v_B2_a_r - 2.23667465123725*v_B2_b_i + 1.28353302446119*v_B2_b_r + 0.643671749092997*v_B2_c_i - 0.385473430243204*v_B2_c_r + 0.249053057297486*v_B2lv_b_i - 0.996212229189942*v_B2lv_b_r - 0.249053057297486*v_B2lv_c_i + 0.996212229189942*v_B2lv_c_r - 0.316443717683753*v_B3_a_i + 0.171091579417532*v_B3_a_r + 1.10755301189314*v_B3_b_i - 0.59882052796136*v_B3_b_r - 0.316443717683753*v_B3_c_i + 0.171091579417531*v_B3_c_r
        struct[0].g[44,0] = -0.171091579417532*v_B1_a_i - 0.316443717683753*v_B1_a_r - 0.171091579417531*v_B1_b_i - 0.316443717683753*v_B1_b_r + 0.59882052796136*v_B1_c_i + 1.10755301189314*v_B1_c_r + 0.385473430243205*v_B2_a_i + 0.643671749092997*v_B2_a_r + 0.385473430243204*v_B2_b_i + 0.643671749092997*v_B2_b_r - 1.28353302446119*v_B2_c_i - 2.23667465123725*v_B2_c_r - 0.996212229189942*v_B2lv_a_i - 0.249053057297486*v_B2lv_a_r + 0.996212229189942*v_B2lv_c_i + 0.249053057297486*v_B2lv_c_r - 0.171091579417532*v_B3_a_i - 0.316443717683753*v_B3_a_r - 0.171091579417531*v_B3_b_i - 0.316443717683753*v_B3_b_r + 0.59882052796136*v_B3_c_i + 1.10755301189314*v_B3_c_r
        struct[0].g[45,0] = -0.316443717683753*v_B1_a_i + 0.171091579417532*v_B1_a_r - 0.316443717683753*v_B1_b_i + 0.171091579417531*v_B1_b_r + 1.10755301189314*v_B1_c_i - 0.59882052796136*v_B1_c_r + 0.643671749092997*v_B2_a_i - 0.385473430243205*v_B2_a_r + 0.643671749092997*v_B2_b_i - 0.385473430243204*v_B2_b_r - 2.23667465123725*v_B2_c_i + 1.28353302446119*v_B2_c_r - 0.249053057297486*v_B2lv_a_i + 0.996212229189942*v_B2lv_a_r + 0.249053057297486*v_B2lv_c_i - 0.996212229189942*v_B2lv_c_r - 0.316443717683753*v_B3_a_i + 0.171091579417532*v_B3_a_r - 0.316443717683753*v_B3_b_i + 0.171091579417531*v_B3_b_r + 1.10755301189314*v_B3_c_i - 0.59882052796136*v_B3_c_r
        struct[0].g[64,0] = 0.598820527961361*v_B5_a_i + 1.10755301189314*v_B5_a_r - 0.171091579417532*v_B5_b_i - 0.316443717683753*v_B5_b_r - 0.171091579417532*v_B5_c_i - 0.316443717683753*v_B5_c_r - 1.28353302446119*v_B6_a_i - 2.23667465123725*v_B6_a_r + 0.385473430243205*v_B6_b_i + 0.643671749092996*v_B6_b_r + 0.385473430243205*v_B6_c_i + 0.643671749092997*v_B6_c_r + 0.996212229189942*v_B6lv_a_i + 0.249053057297486*v_B6lv_a_r - 0.996212229189942*v_B6lv_b_i - 0.249053057297486*v_B6lv_b_r + 0.598820527961361*v_B7_a_i + 1.10755301189314*v_B7_a_r - 0.171091579417532*v_B7_b_i - 0.316443717683753*v_B7_b_r - 0.171091579417532*v_B7_c_i - 0.316443717683753*v_B7_c_r
        struct[0].g[65,0] = 1.10755301189314*v_B5_a_i - 0.598820527961361*v_B5_a_r - 0.316443717683753*v_B5_b_i + 0.171091579417532*v_B5_b_r - 0.316443717683753*v_B5_c_i + 0.171091579417532*v_B5_c_r - 2.23667465123725*v_B6_a_i + 1.28353302446119*v_B6_a_r + 0.643671749092996*v_B6_b_i - 0.385473430243205*v_B6_b_r + 0.643671749092997*v_B6_c_i - 0.385473430243205*v_B6_c_r + 0.249053057297486*v_B6lv_a_i - 0.996212229189942*v_B6lv_a_r - 0.249053057297486*v_B6lv_b_i + 0.996212229189942*v_B6lv_b_r + 1.10755301189314*v_B7_a_i - 0.598820527961361*v_B7_a_r - 0.316443717683753*v_B7_b_i + 0.171091579417532*v_B7_b_r - 0.316443717683753*v_B7_c_i + 0.171091579417532*v_B7_c_r
        struct[0].g[66,0] = -0.171091579417532*v_B5_a_i - 0.316443717683753*v_B5_a_r + 0.59882052796136*v_B5_b_i + 1.10755301189314*v_B5_b_r - 0.171091579417531*v_B5_c_i - 0.316443717683753*v_B5_c_r + 0.385473430243205*v_B6_a_i + 0.643671749092996*v_B6_a_r - 1.28353302446119*v_B6_b_i - 2.23667465123725*v_B6_b_r + 0.385473430243204*v_B6_c_i + 0.643671749092997*v_B6_c_r + 0.996212229189942*v_B6lv_b_i + 0.249053057297486*v_B6lv_b_r - 0.996212229189942*v_B6lv_c_i - 0.249053057297486*v_B6lv_c_r - 0.171091579417532*v_B7_a_i - 0.316443717683753*v_B7_a_r + 0.59882052796136*v_B7_b_i + 1.10755301189314*v_B7_b_r - 0.171091579417531*v_B7_c_i - 0.316443717683753*v_B7_c_r
        struct[0].g[67,0] = -0.316443717683753*v_B5_a_i + 0.171091579417532*v_B5_a_r + 1.10755301189314*v_B5_b_i - 0.59882052796136*v_B5_b_r - 0.316443717683753*v_B5_c_i + 0.171091579417531*v_B5_c_r + 0.643671749092996*v_B6_a_i - 0.385473430243205*v_B6_a_r - 2.23667465123725*v_B6_b_i + 1.28353302446119*v_B6_b_r + 0.643671749092997*v_B6_c_i - 0.385473430243204*v_B6_c_r + 0.249053057297486*v_B6lv_b_i - 0.996212229189942*v_B6lv_b_r - 0.249053057297486*v_B6lv_c_i + 0.996212229189942*v_B6lv_c_r - 0.316443717683753*v_B7_a_i + 0.171091579417532*v_B7_a_r + 1.10755301189314*v_B7_b_i - 0.59882052796136*v_B7_b_r - 0.316443717683753*v_B7_c_i + 0.171091579417531*v_B7_c_r
        struct[0].g[68,0] = -0.171091579417532*v_B5_a_i - 0.316443717683753*v_B5_a_r - 0.171091579417531*v_B5_b_i - 0.316443717683753*v_B5_b_r + 0.59882052796136*v_B5_c_i + 1.10755301189314*v_B5_c_r + 0.385473430243205*v_B6_a_i + 0.643671749092997*v_B6_a_r + 0.385473430243204*v_B6_b_i + 0.643671749092997*v_B6_b_r - 1.28353302446119*v_B6_c_i - 2.23667465123725*v_B6_c_r - 0.996212229189942*v_B6lv_a_i - 0.249053057297486*v_B6lv_a_r + 0.996212229189942*v_B6lv_c_i + 0.249053057297486*v_B6lv_c_r - 0.171091579417532*v_B7_a_i - 0.316443717683753*v_B7_a_r - 0.171091579417531*v_B7_b_i - 0.316443717683753*v_B7_b_r + 0.59882052796136*v_B7_c_i + 1.10755301189314*v_B7_c_r
        struct[0].g[69,0] = -0.316443717683753*v_B5_a_i + 0.171091579417532*v_B5_a_r - 0.316443717683753*v_B5_b_i + 0.171091579417531*v_B5_b_r + 1.10755301189314*v_B5_c_i - 0.59882052796136*v_B5_c_r + 0.643671749092997*v_B6_a_i - 0.385473430243205*v_B6_a_r + 0.643671749092997*v_B6_b_i - 0.385473430243204*v_B6_b_r - 2.23667465123725*v_B6_c_i + 1.28353302446119*v_B6_c_r - 0.249053057297486*v_B6lv_a_i + 0.996212229189942*v_B6lv_a_r + 0.249053057297486*v_B6lv_c_i - 0.996212229189942*v_B6lv_c_r - 0.316443717683753*v_B7_a_i + 0.171091579417532*v_B7_a_r - 0.316443717683753*v_B7_b_i + 0.171091579417531*v_B7_b_r + 1.10755301189314*v_B7_c_i - 0.59882052796136*v_B7_c_r
        struct[0].g[100,0] = -i_l_B1_B2_a_r + 0.598820527961361*v_B1_a_i + 1.10755301189314*v_B1_a_r - 0.171091579417532*v_B1_b_i - 0.316443717683753*v_B1_b_r - 0.171091579417532*v_B1_c_i - 0.316443717683753*v_B1_c_r - 0.598820527961361*v_B2_a_i - 1.10755301189314*v_B2_a_r + 0.171091579417532*v_B2_b_i + 0.316443717683753*v_B2_b_r + 0.171091579417532*v_B2_c_i + 0.316443717683753*v_B2_c_r
        struct[0].g[101,0] = -i_l_B1_B2_a_i + 1.10755301189314*v_B1_a_i - 0.598820527961361*v_B1_a_r - 0.316443717683753*v_B1_b_i + 0.171091579417532*v_B1_b_r - 0.316443717683753*v_B1_c_i + 0.171091579417532*v_B1_c_r - 1.10755301189314*v_B2_a_i + 0.598820527961361*v_B2_a_r + 0.316443717683753*v_B2_b_i - 0.171091579417532*v_B2_b_r + 0.316443717683753*v_B2_c_i - 0.171091579417532*v_B2_c_r
        struct[0].g[102,0] = -i_l_B1_B2_b_r - 0.171091579417532*v_B1_a_i - 0.316443717683753*v_B1_a_r + 0.59882052796136*v_B1_b_i + 1.10755301189314*v_B1_b_r - 0.171091579417531*v_B1_c_i - 0.316443717683753*v_B1_c_r + 0.171091579417532*v_B2_a_i + 0.316443717683753*v_B2_a_r - 0.59882052796136*v_B2_b_i - 1.10755301189314*v_B2_b_r + 0.171091579417531*v_B2_c_i + 0.316443717683753*v_B2_c_r
        struct[0].g[103,0] = -i_l_B1_B2_b_i - 0.316443717683753*v_B1_a_i + 0.171091579417532*v_B1_a_r + 1.10755301189314*v_B1_b_i - 0.59882052796136*v_B1_b_r - 0.316443717683753*v_B1_c_i + 0.171091579417531*v_B1_c_r + 0.316443717683753*v_B2_a_i - 0.171091579417532*v_B2_a_r - 1.10755301189314*v_B2_b_i + 0.59882052796136*v_B2_b_r + 0.316443717683753*v_B2_c_i - 0.171091579417531*v_B2_c_r
        struct[0].g[104,0] = -i_l_B1_B2_c_r - 0.171091579417532*v_B1_a_i - 0.316443717683753*v_B1_a_r - 0.171091579417531*v_B1_b_i - 0.316443717683753*v_B1_b_r + 0.59882052796136*v_B1_c_i + 1.10755301189314*v_B1_c_r + 0.171091579417532*v_B2_a_i + 0.316443717683753*v_B2_a_r + 0.171091579417531*v_B2_b_i + 0.316443717683753*v_B2_b_r - 0.59882052796136*v_B2_c_i - 1.10755301189314*v_B2_c_r
        struct[0].g[105,0] = -i_l_B1_B2_c_i - 0.316443717683753*v_B1_a_i + 0.171091579417532*v_B1_a_r - 0.316443717683753*v_B1_b_i + 0.171091579417531*v_B1_b_r + 1.10755301189314*v_B1_c_i - 0.59882052796136*v_B1_c_r + 0.316443717683753*v_B2_a_i - 0.171091579417532*v_B2_a_r + 0.316443717683753*v_B2_b_i - 0.171091579417531*v_B2_b_r - 1.10755301189314*v_B2_c_i + 0.59882052796136*v_B2_c_r
        struct[0].g[124,0] = -i_l_B6_B7_a_r + 0.598820527961361*v_B6_a_i + 1.10755301189314*v_B6_a_r - 0.171091579417532*v_B6_b_i - 0.316443717683753*v_B6_b_r - 0.171091579417532*v_B6_c_i - 0.316443717683753*v_B6_c_r - 0.598820527961361*v_B7_a_i - 1.10755301189314*v_B7_a_r + 0.171091579417532*v_B7_b_i + 0.316443717683753*v_B7_b_r + 0.171091579417532*v_B7_c_i + 0.316443717683753*v_B7_c_r
        struct[0].g[125,0] = -i_l_B6_B7_a_i + 1.10755301189314*v_B6_a_i - 0.598820527961361*v_B6_a_r - 0.316443717683753*v_B6_b_i + 0.171091579417532*v_B6_b_r - 0.316443717683753*v_B6_c_i + 0.171091579417532*v_B6_c_r - 1.10755301189314*v_B7_a_i + 0.598820527961361*v_B7_a_r + 0.316443717683753*v_B7_b_i - 0.171091579417532*v_B7_b_r + 0.316443717683753*v_B7_c_i - 0.171091579417532*v_B7_c_r
        struct[0].g[126,0] = -i_l_B6_B7_b_r - 0.171091579417532*v_B6_a_i - 0.316443717683753*v_B6_a_r + 0.59882052796136*v_B6_b_i + 1.10755301189314*v_B6_b_r - 0.171091579417531*v_B6_c_i - 0.316443717683753*v_B6_c_r + 0.171091579417532*v_B7_a_i + 0.316443717683753*v_B7_a_r - 0.59882052796136*v_B7_b_i - 1.10755301189314*v_B7_b_r + 0.171091579417531*v_B7_c_i + 0.316443717683753*v_B7_c_r
        struct[0].g[127,0] = -i_l_B6_B7_b_i - 0.316443717683753*v_B6_a_i + 0.171091579417532*v_B6_a_r + 1.10755301189314*v_B6_b_i - 0.59882052796136*v_B6_b_r - 0.316443717683753*v_B6_c_i + 0.171091579417531*v_B6_c_r + 0.316443717683753*v_B7_a_i - 0.171091579417532*v_B7_a_r - 1.10755301189314*v_B7_b_i + 0.59882052796136*v_B7_b_r + 0.316443717683753*v_B7_c_i - 0.171091579417531*v_B7_c_r
        struct[0].g[128,0] = -i_l_B6_B7_c_r - 0.171091579417532*v_B6_a_i - 0.316443717683753*v_B6_a_r - 0.171091579417531*v_B6_b_i - 0.316443717683753*v_B6_b_r + 0.59882052796136*v_B6_c_i + 1.10755301189314*v_B6_c_r + 0.171091579417532*v_B7_a_i + 0.316443717683753*v_B7_a_r + 0.171091579417531*v_B7_b_i + 0.316443717683753*v_B7_b_r - 0.59882052796136*v_B7_c_i - 1.10755301189314*v_B7_c_r
        struct[0].g[129,0] = -i_l_B6_B7_c_i - 0.316443717683753*v_B6_a_i + 0.171091579417532*v_B6_a_r - 0.316443717683753*v_B6_b_i + 0.171091579417531*v_B6_b_r + 1.10755301189314*v_B6_c_i - 0.59882052796136*v_B6_c_r + 0.316443717683753*v_B7_a_i - 0.171091579417532*v_B7_a_r + 0.316443717683753*v_B7_b_i - 0.171091579417531*v_B7_b_r - 1.10755301189314*v_B7_c_i + 0.59882052796136*v_B7_c_r
        struct[0].g[130,0] = i_load_B2lv_a_i*v_B2lv_a_i - i_load_B2lv_a_i*v_B2lv_n_i + i_load_B2lv_a_r*v_B2lv_a_r - i_load_B2lv_a_r*v_B2lv_n_r - p_B2lv_a
        struct[0].g[131,0] = i_load_B2lv_b_i*v_B2lv_b_i - i_load_B2lv_b_i*v_B2lv_n_i + i_load_B2lv_b_r*v_B2lv_b_r - i_load_B2lv_b_r*v_B2lv_n_r - p_B2lv_b
        struct[0].g[132,0] = i_load_B2lv_c_i*v_B2lv_c_i - i_load_B2lv_c_i*v_B2lv_n_i + i_load_B2lv_c_r*v_B2lv_c_r - i_load_B2lv_c_r*v_B2lv_n_r - p_B2lv_c
        struct[0].g[133,0] = -i_load_B2lv_a_i*v_B2lv_a_r + i_load_B2lv_a_i*v_B2lv_n_r + i_load_B2lv_a_r*v_B2lv_a_i - i_load_B2lv_a_r*v_B2lv_n_i - q_B2lv_a
        struct[0].g[134,0] = -i_load_B2lv_b_i*v_B2lv_b_r + i_load_B2lv_b_i*v_B2lv_n_r + i_load_B2lv_b_r*v_B2lv_b_i - i_load_B2lv_b_r*v_B2lv_n_i - q_B2lv_b
        struct[0].g[135,0] = -i_load_B2lv_c_i*v_B2lv_c_r + i_load_B2lv_c_i*v_B2lv_n_r + i_load_B2lv_c_r*v_B2lv_c_i - i_load_B2lv_c_r*v_B2lv_n_i - q_B2lv_c
        struct[0].g[138,0] = i_load_B3lv_a_i*v_B3lv_a_i - i_load_B3lv_a_i*v_B3lv_n_i + i_load_B3lv_a_r*v_B3lv_a_r - i_load_B3lv_a_r*v_B3lv_n_r - p_B3lv_a
        struct[0].g[139,0] = i_load_B3lv_b_i*v_B3lv_b_i - i_load_B3lv_b_i*v_B3lv_n_i + i_load_B3lv_b_r*v_B3lv_b_r - i_load_B3lv_b_r*v_B3lv_n_r - p_B3lv_b
        struct[0].g[140,0] = i_load_B3lv_c_i*v_B3lv_c_i - i_load_B3lv_c_i*v_B3lv_n_i + i_load_B3lv_c_r*v_B3lv_c_r - i_load_B3lv_c_r*v_B3lv_n_r - p_B3lv_c
        struct[0].g[141,0] = -i_load_B3lv_a_i*v_B3lv_a_r + i_load_B3lv_a_i*v_B3lv_n_r + i_load_B3lv_a_r*v_B3lv_a_i - i_load_B3lv_a_r*v_B3lv_n_i - q_B3lv_a
        struct[0].g[142,0] = -i_load_B3lv_b_i*v_B3lv_b_r + i_load_B3lv_b_i*v_B3lv_n_r + i_load_B3lv_b_r*v_B3lv_b_i - i_load_B3lv_b_r*v_B3lv_n_i - q_B3lv_b
        struct[0].g[143,0] = -i_load_B3lv_c_i*v_B3lv_c_r + i_load_B3lv_c_i*v_B3lv_n_r + i_load_B3lv_c_r*v_B3lv_c_i - i_load_B3lv_c_r*v_B3lv_n_i - q_B3lv_c
        struct[0].g[146,0] = i_load_B4lv_a_i*v_B4lv_a_i - i_load_B4lv_a_i*v_B4lv_n_i + i_load_B4lv_a_r*v_B4lv_a_r - i_load_B4lv_a_r*v_B4lv_n_r - p_B4lv_a
        struct[0].g[147,0] = i_load_B4lv_b_i*v_B4lv_b_i - i_load_B4lv_b_i*v_B4lv_n_i + i_load_B4lv_b_r*v_B4lv_b_r - i_load_B4lv_b_r*v_B4lv_n_r - p_B4lv_b
        struct[0].g[148,0] = i_load_B4lv_c_i*v_B4lv_c_i - i_load_B4lv_c_i*v_B4lv_n_i + i_load_B4lv_c_r*v_B4lv_c_r - i_load_B4lv_c_r*v_B4lv_n_r - p_B4lv_c
        struct[0].g[149,0] = -i_load_B4lv_a_i*v_B4lv_a_r + i_load_B4lv_a_i*v_B4lv_n_r + i_load_B4lv_a_r*v_B4lv_a_i - i_load_B4lv_a_r*v_B4lv_n_i - q_B4lv_a
        struct[0].g[150,0] = -i_load_B4lv_b_i*v_B4lv_b_r + i_load_B4lv_b_i*v_B4lv_n_r + i_load_B4lv_b_r*v_B4lv_b_i - i_load_B4lv_b_r*v_B4lv_n_i - q_B4lv_b
        struct[0].g[151,0] = -i_load_B4lv_c_i*v_B4lv_c_r + i_load_B4lv_c_i*v_B4lv_n_r + i_load_B4lv_c_r*v_B4lv_c_i - i_load_B4lv_c_r*v_B4lv_n_i - q_B4lv_c
        struct[0].g[154,0] = i_load_B5lv_a_i*v_B5lv_a_i - i_load_B5lv_a_i*v_B5lv_n_i + i_load_B5lv_a_r*v_B5lv_a_r - i_load_B5lv_a_r*v_B5lv_n_r - p_B5lv_a
        struct[0].g[155,0] = i_load_B5lv_b_i*v_B5lv_b_i - i_load_B5lv_b_i*v_B5lv_n_i + i_load_B5lv_b_r*v_B5lv_b_r - i_load_B5lv_b_r*v_B5lv_n_r - p_B5lv_b
        struct[0].g[156,0] = i_load_B5lv_c_i*v_B5lv_c_i - i_load_B5lv_c_i*v_B5lv_n_i + i_load_B5lv_c_r*v_B5lv_c_r - i_load_B5lv_c_r*v_B5lv_n_r - p_B5lv_c
        struct[0].g[157,0] = -i_load_B5lv_a_i*v_B5lv_a_r + i_load_B5lv_a_i*v_B5lv_n_r + i_load_B5lv_a_r*v_B5lv_a_i - i_load_B5lv_a_r*v_B5lv_n_i - q_B5lv_a
        struct[0].g[158,0] = -i_load_B5lv_b_i*v_B5lv_b_r + i_load_B5lv_b_i*v_B5lv_n_r + i_load_B5lv_b_r*v_B5lv_b_i - i_load_B5lv_b_r*v_B5lv_n_i - q_B5lv_b
        struct[0].g[159,0] = -i_load_B5lv_c_i*v_B5lv_c_r + i_load_B5lv_c_i*v_B5lv_n_r + i_load_B5lv_c_r*v_B5lv_c_i - i_load_B5lv_c_r*v_B5lv_n_i - q_B5lv_c
        struct[0].g[162,0] = i_load_B6lv_a_i*v_B6lv_a_i - i_load_B6lv_a_i*v_B6lv_n_i + i_load_B6lv_a_r*v_B6lv_a_r - i_load_B6lv_a_r*v_B6lv_n_r - p_B6lv_a
        struct[0].g[163,0] = i_load_B6lv_b_i*v_B6lv_b_i - i_load_B6lv_b_i*v_B6lv_n_i + i_load_B6lv_b_r*v_B6lv_b_r - i_load_B6lv_b_r*v_B6lv_n_r - p_B6lv_b
        struct[0].g[164,0] = i_load_B6lv_c_i*v_B6lv_c_i - i_load_B6lv_c_i*v_B6lv_n_i + i_load_B6lv_c_r*v_B6lv_c_r - i_load_B6lv_c_r*v_B6lv_n_r - p_B6lv_c
        struct[0].g[165,0] = -i_load_B6lv_a_i*v_B6lv_a_r + i_load_B6lv_a_i*v_B6lv_n_r + i_load_B6lv_a_r*v_B6lv_a_i - i_load_B6lv_a_r*v_B6lv_n_i - q_B6lv_a
        struct[0].g[166,0] = -i_load_B6lv_b_i*v_B6lv_b_r + i_load_B6lv_b_i*v_B6lv_n_r + i_load_B6lv_b_r*v_B6lv_b_i - i_load_B6lv_b_r*v_B6lv_n_i - q_B6lv_b
        struct[0].g[167,0] = -i_load_B6lv_c_i*v_B6lv_c_r + i_load_B6lv_c_i*v_B6lv_n_r + i_load_B6lv_c_r*v_B6lv_c_i - i_load_B6lv_c_r*v_B6lv_n_i - q_B6lv_c
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = (v_B1_a_i**2 + v_B1_a_r**2)**0.5
        struct[0].h[1,0] = (v_B1_b_i**2 + v_B1_b_r**2)**0.5
        struct[0].h[2,0] = (v_B1_c_i**2 + v_B1_c_r**2)**0.5
        struct[0].h[3,0] = (v_B7_a_i**2 + v_B7_a_r**2)**0.5
        struct[0].h[4,0] = (v_B7_b_i**2 + v_B7_b_r**2)**0.5
        struct[0].h[5,0] = (v_B7_c_i**2 + v_B7_c_r**2)**0.5
        struct[0].h[6,0] = (v_B2lv_a_i**2 + v_B2lv_a_r**2)**0.5
        struct[0].h[7,0] = (v_B2lv_b_i**2 + v_B2lv_b_r**2)**0.5
        struct[0].h[8,0] = (v_B2lv_c_i**2 + v_B2lv_c_r**2)**0.5
        struct[0].h[9,0] = (v_B2lv_n_i**2 + v_B2lv_n_r**2)**0.5
        struct[0].h[10,0] = (v_B3lv_a_i**2 + v_B3lv_a_r**2)**0.5
        struct[0].h[11,0] = (v_B3lv_b_i**2 + v_B3lv_b_r**2)**0.5
        struct[0].h[12,0] = (v_B3lv_c_i**2 + v_B3lv_c_r**2)**0.5
        struct[0].h[13,0] = (v_B3lv_n_i**2 + v_B3lv_n_r**2)**0.5
        struct[0].h[14,0] = (v_B4lv_a_i**2 + v_B4lv_a_r**2)**0.5
        struct[0].h[15,0] = (v_B4lv_b_i**2 + v_B4lv_b_r**2)**0.5
        struct[0].h[16,0] = (v_B4lv_c_i**2 + v_B4lv_c_r**2)**0.5
        struct[0].h[17,0] = (v_B4lv_n_i**2 + v_B4lv_n_r**2)**0.5
        struct[0].h[18,0] = (v_B5lv_a_i**2 + v_B5lv_a_r**2)**0.5
        struct[0].h[19,0] = (v_B5lv_b_i**2 + v_B5lv_b_r**2)**0.5
        struct[0].h[20,0] = (v_B5lv_c_i**2 + v_B5lv_c_r**2)**0.5
        struct[0].h[21,0] = (v_B5lv_n_i**2 + v_B5lv_n_r**2)**0.5
        struct[0].h[22,0] = (v_B6lv_a_i**2 + v_B6lv_a_r**2)**0.5
        struct[0].h[23,0] = (v_B6lv_b_i**2 + v_B6lv_b_r**2)**0.5
        struct[0].h[24,0] = (v_B6lv_c_i**2 + v_B6lv_c_r**2)**0.5
        struct[0].h[25,0] = (v_B6lv_n_i**2 + v_B6lv_n_r**2)**0.5
        struct[0].h[26,0] = (v_B2_a_i**2 + v_B2_a_r**2)**0.5
        struct[0].h[27,0] = (v_B2_b_i**2 + v_B2_b_r**2)**0.5
        struct[0].h[28,0] = (v_B2_c_i**2 + v_B2_c_r**2)**0.5
        struct[0].h[29,0] = (v_B3_a_i**2 + v_B3_a_r**2)**0.5
        struct[0].h[30,0] = (v_B3_b_i**2 + v_B3_b_r**2)**0.5
        struct[0].h[31,0] = (v_B3_c_i**2 + v_B3_c_r**2)**0.5
        struct[0].h[32,0] = (v_B4_a_i**2 + v_B4_a_r**2)**0.5
        struct[0].h[33,0] = (v_B4_b_i**2 + v_B4_b_r**2)**0.5
        struct[0].h[34,0] = (v_B4_c_i**2 + v_B4_c_r**2)**0.5
        struct[0].h[35,0] = (v_B5_a_i**2 + v_B5_a_r**2)**0.5
        struct[0].h[36,0] = (v_B5_b_i**2 + v_B5_b_r**2)**0.5
        struct[0].h[37,0] = (v_B5_c_i**2 + v_B5_c_r**2)**0.5
        struct[0].h[38,0] = (v_B6_a_i**2 + v_B6_a_r**2)**0.5
        struct[0].h[39,0] = (v_B6_b_i**2 + v_B6_b_r**2)**0.5
        struct[0].h[40,0] = (v_B6_c_i**2 + v_B6_c_r**2)**0.5
    

    if mode == 10:

        pass

    if mode == 11:



        struct[0].Gy_ini[130,0] = i_load_B2lv_a_r
        struct[0].Gy_ini[130,1] = i_load_B2lv_a_i
        struct[0].Gy_ini[130,6] = -i_load_B2lv_a_r
        struct[0].Gy_ini[130,7] = -i_load_B2lv_a_i
        struct[0].Gy_ini[130,130] = v_B2lv_a_r - v_B2lv_n_r
        struct[0].Gy_ini[130,131] = v_B2lv_a_i - v_B2lv_n_i
        struct[0].Gy_ini[131,2] = i_load_B2lv_b_r
        struct[0].Gy_ini[131,3] = i_load_B2lv_b_i
        struct[0].Gy_ini[131,6] = -i_load_B2lv_b_r
        struct[0].Gy_ini[131,7] = -i_load_B2lv_b_i
        struct[0].Gy_ini[131,132] = v_B2lv_b_r - v_B2lv_n_r
        struct[0].Gy_ini[131,133] = v_B2lv_b_i - v_B2lv_n_i
        struct[0].Gy_ini[132,4] = i_load_B2lv_c_r
        struct[0].Gy_ini[132,5] = i_load_B2lv_c_i
        struct[0].Gy_ini[132,6] = -i_load_B2lv_c_r
        struct[0].Gy_ini[132,7] = -i_load_B2lv_c_i
        struct[0].Gy_ini[132,134] = v_B2lv_c_r - v_B2lv_n_r
        struct[0].Gy_ini[132,135] = v_B2lv_c_i - v_B2lv_n_i
        struct[0].Gy_ini[133,0] = -i_load_B2lv_a_i
        struct[0].Gy_ini[133,1] = i_load_B2lv_a_r
        struct[0].Gy_ini[133,6] = i_load_B2lv_a_i
        struct[0].Gy_ini[133,7] = -i_load_B2lv_a_r
        struct[0].Gy_ini[133,130] = v_B2lv_a_i - v_B2lv_n_i
        struct[0].Gy_ini[133,131] = -v_B2lv_a_r + v_B2lv_n_r
        struct[0].Gy_ini[134,2] = -i_load_B2lv_b_i
        struct[0].Gy_ini[134,3] = i_load_B2lv_b_r
        struct[0].Gy_ini[134,6] = i_load_B2lv_b_i
        struct[0].Gy_ini[134,7] = -i_load_B2lv_b_r
        struct[0].Gy_ini[134,132] = v_B2lv_b_i - v_B2lv_n_i
        struct[0].Gy_ini[134,133] = -v_B2lv_b_r + v_B2lv_n_r
        struct[0].Gy_ini[135,4] = -i_load_B2lv_c_i
        struct[0].Gy_ini[135,5] = i_load_B2lv_c_r
        struct[0].Gy_ini[135,6] = i_load_B2lv_c_i
        struct[0].Gy_ini[135,7] = -i_load_B2lv_c_r
        struct[0].Gy_ini[135,134] = v_B2lv_c_i - v_B2lv_n_i
        struct[0].Gy_ini[135,135] = -v_B2lv_c_r + v_B2lv_n_r
        struct[0].Gy_ini[138,8] = i_load_B3lv_a_r
        struct[0].Gy_ini[138,9] = i_load_B3lv_a_i
        struct[0].Gy_ini[138,14] = -i_load_B3lv_a_r
        struct[0].Gy_ini[138,15] = -i_load_B3lv_a_i
        struct[0].Gy_ini[138,138] = v_B3lv_a_r - v_B3lv_n_r
        struct[0].Gy_ini[138,139] = v_B3lv_a_i - v_B3lv_n_i
        struct[0].Gy_ini[139,10] = i_load_B3lv_b_r
        struct[0].Gy_ini[139,11] = i_load_B3lv_b_i
        struct[0].Gy_ini[139,14] = -i_load_B3lv_b_r
        struct[0].Gy_ini[139,15] = -i_load_B3lv_b_i
        struct[0].Gy_ini[139,140] = v_B3lv_b_r - v_B3lv_n_r
        struct[0].Gy_ini[139,141] = v_B3lv_b_i - v_B3lv_n_i
        struct[0].Gy_ini[140,12] = i_load_B3lv_c_r
        struct[0].Gy_ini[140,13] = i_load_B3lv_c_i
        struct[0].Gy_ini[140,14] = -i_load_B3lv_c_r
        struct[0].Gy_ini[140,15] = -i_load_B3lv_c_i
        struct[0].Gy_ini[140,142] = v_B3lv_c_r - v_B3lv_n_r
        struct[0].Gy_ini[140,143] = v_B3lv_c_i - v_B3lv_n_i
        struct[0].Gy_ini[141,8] = -i_load_B3lv_a_i
        struct[0].Gy_ini[141,9] = i_load_B3lv_a_r
        struct[0].Gy_ini[141,14] = i_load_B3lv_a_i
        struct[0].Gy_ini[141,15] = -i_load_B3lv_a_r
        struct[0].Gy_ini[141,138] = v_B3lv_a_i - v_B3lv_n_i
        struct[0].Gy_ini[141,139] = -v_B3lv_a_r + v_B3lv_n_r
        struct[0].Gy_ini[142,10] = -i_load_B3lv_b_i
        struct[0].Gy_ini[142,11] = i_load_B3lv_b_r
        struct[0].Gy_ini[142,14] = i_load_B3lv_b_i
        struct[0].Gy_ini[142,15] = -i_load_B3lv_b_r
        struct[0].Gy_ini[142,140] = v_B3lv_b_i - v_B3lv_n_i
        struct[0].Gy_ini[142,141] = -v_B3lv_b_r + v_B3lv_n_r
        struct[0].Gy_ini[143,12] = -i_load_B3lv_c_i
        struct[0].Gy_ini[143,13] = i_load_B3lv_c_r
        struct[0].Gy_ini[143,14] = i_load_B3lv_c_i
        struct[0].Gy_ini[143,15] = -i_load_B3lv_c_r
        struct[0].Gy_ini[143,142] = v_B3lv_c_i - v_B3lv_n_i
        struct[0].Gy_ini[143,143] = -v_B3lv_c_r + v_B3lv_n_r
        struct[0].Gy_ini[146,16] = i_load_B4lv_a_r
        struct[0].Gy_ini[146,17] = i_load_B4lv_a_i
        struct[0].Gy_ini[146,22] = -i_load_B4lv_a_r
        struct[0].Gy_ini[146,23] = -i_load_B4lv_a_i
        struct[0].Gy_ini[146,146] = v_B4lv_a_r - v_B4lv_n_r
        struct[0].Gy_ini[146,147] = v_B4lv_a_i - v_B4lv_n_i
        struct[0].Gy_ini[147,18] = i_load_B4lv_b_r
        struct[0].Gy_ini[147,19] = i_load_B4lv_b_i
        struct[0].Gy_ini[147,22] = -i_load_B4lv_b_r
        struct[0].Gy_ini[147,23] = -i_load_B4lv_b_i
        struct[0].Gy_ini[147,148] = v_B4lv_b_r - v_B4lv_n_r
        struct[0].Gy_ini[147,149] = v_B4lv_b_i - v_B4lv_n_i
        struct[0].Gy_ini[148,20] = i_load_B4lv_c_r
        struct[0].Gy_ini[148,21] = i_load_B4lv_c_i
        struct[0].Gy_ini[148,22] = -i_load_B4lv_c_r
        struct[0].Gy_ini[148,23] = -i_load_B4lv_c_i
        struct[0].Gy_ini[148,150] = v_B4lv_c_r - v_B4lv_n_r
        struct[0].Gy_ini[148,151] = v_B4lv_c_i - v_B4lv_n_i
        struct[0].Gy_ini[149,16] = -i_load_B4lv_a_i
        struct[0].Gy_ini[149,17] = i_load_B4lv_a_r
        struct[0].Gy_ini[149,22] = i_load_B4lv_a_i
        struct[0].Gy_ini[149,23] = -i_load_B4lv_a_r
        struct[0].Gy_ini[149,146] = v_B4lv_a_i - v_B4lv_n_i
        struct[0].Gy_ini[149,147] = -v_B4lv_a_r + v_B4lv_n_r
        struct[0].Gy_ini[150,18] = -i_load_B4lv_b_i
        struct[0].Gy_ini[150,19] = i_load_B4lv_b_r
        struct[0].Gy_ini[150,22] = i_load_B4lv_b_i
        struct[0].Gy_ini[150,23] = -i_load_B4lv_b_r
        struct[0].Gy_ini[150,148] = v_B4lv_b_i - v_B4lv_n_i
        struct[0].Gy_ini[150,149] = -v_B4lv_b_r + v_B4lv_n_r
        struct[0].Gy_ini[151,20] = -i_load_B4lv_c_i
        struct[0].Gy_ini[151,21] = i_load_B4lv_c_r
        struct[0].Gy_ini[151,22] = i_load_B4lv_c_i
        struct[0].Gy_ini[151,23] = -i_load_B4lv_c_r
        struct[0].Gy_ini[151,150] = v_B4lv_c_i - v_B4lv_n_i
        struct[0].Gy_ini[151,151] = -v_B4lv_c_r + v_B4lv_n_r
        struct[0].Gy_ini[154,24] = i_load_B5lv_a_r
        struct[0].Gy_ini[154,25] = i_load_B5lv_a_i
        struct[0].Gy_ini[154,30] = -i_load_B5lv_a_r
        struct[0].Gy_ini[154,31] = -i_load_B5lv_a_i
        struct[0].Gy_ini[154,154] = v_B5lv_a_r - v_B5lv_n_r
        struct[0].Gy_ini[154,155] = v_B5lv_a_i - v_B5lv_n_i
        struct[0].Gy_ini[155,26] = i_load_B5lv_b_r
        struct[0].Gy_ini[155,27] = i_load_B5lv_b_i
        struct[0].Gy_ini[155,30] = -i_load_B5lv_b_r
        struct[0].Gy_ini[155,31] = -i_load_B5lv_b_i
        struct[0].Gy_ini[155,156] = v_B5lv_b_r - v_B5lv_n_r
        struct[0].Gy_ini[155,157] = v_B5lv_b_i - v_B5lv_n_i
        struct[0].Gy_ini[156,28] = i_load_B5lv_c_r
        struct[0].Gy_ini[156,29] = i_load_B5lv_c_i
        struct[0].Gy_ini[156,30] = -i_load_B5lv_c_r
        struct[0].Gy_ini[156,31] = -i_load_B5lv_c_i
        struct[0].Gy_ini[156,158] = v_B5lv_c_r - v_B5lv_n_r
        struct[0].Gy_ini[156,159] = v_B5lv_c_i - v_B5lv_n_i
        struct[0].Gy_ini[157,24] = -i_load_B5lv_a_i
        struct[0].Gy_ini[157,25] = i_load_B5lv_a_r
        struct[0].Gy_ini[157,30] = i_load_B5lv_a_i
        struct[0].Gy_ini[157,31] = -i_load_B5lv_a_r
        struct[0].Gy_ini[157,154] = v_B5lv_a_i - v_B5lv_n_i
        struct[0].Gy_ini[157,155] = -v_B5lv_a_r + v_B5lv_n_r
        struct[0].Gy_ini[158,26] = -i_load_B5lv_b_i
        struct[0].Gy_ini[158,27] = i_load_B5lv_b_r
        struct[0].Gy_ini[158,30] = i_load_B5lv_b_i
        struct[0].Gy_ini[158,31] = -i_load_B5lv_b_r
        struct[0].Gy_ini[158,156] = v_B5lv_b_i - v_B5lv_n_i
        struct[0].Gy_ini[158,157] = -v_B5lv_b_r + v_B5lv_n_r
        struct[0].Gy_ini[159,28] = -i_load_B5lv_c_i
        struct[0].Gy_ini[159,29] = i_load_B5lv_c_r
        struct[0].Gy_ini[159,30] = i_load_B5lv_c_i
        struct[0].Gy_ini[159,31] = -i_load_B5lv_c_r
        struct[0].Gy_ini[159,158] = v_B5lv_c_i - v_B5lv_n_i
        struct[0].Gy_ini[159,159] = -v_B5lv_c_r + v_B5lv_n_r
        struct[0].Gy_ini[162,32] = i_load_B6lv_a_r
        struct[0].Gy_ini[162,33] = i_load_B6lv_a_i
        struct[0].Gy_ini[162,38] = -i_load_B6lv_a_r
        struct[0].Gy_ini[162,39] = -i_load_B6lv_a_i
        struct[0].Gy_ini[162,162] = v_B6lv_a_r - v_B6lv_n_r
        struct[0].Gy_ini[162,163] = v_B6lv_a_i - v_B6lv_n_i
        struct[0].Gy_ini[163,34] = i_load_B6lv_b_r
        struct[0].Gy_ini[163,35] = i_load_B6lv_b_i
        struct[0].Gy_ini[163,38] = -i_load_B6lv_b_r
        struct[0].Gy_ini[163,39] = -i_load_B6lv_b_i
        struct[0].Gy_ini[163,164] = v_B6lv_b_r - v_B6lv_n_r
        struct[0].Gy_ini[163,165] = v_B6lv_b_i - v_B6lv_n_i
        struct[0].Gy_ini[164,36] = i_load_B6lv_c_r
        struct[0].Gy_ini[164,37] = i_load_B6lv_c_i
        struct[0].Gy_ini[164,38] = -i_load_B6lv_c_r
        struct[0].Gy_ini[164,39] = -i_load_B6lv_c_i
        struct[0].Gy_ini[164,166] = v_B6lv_c_r - v_B6lv_n_r
        struct[0].Gy_ini[164,167] = v_B6lv_c_i - v_B6lv_n_i
        struct[0].Gy_ini[165,32] = -i_load_B6lv_a_i
        struct[0].Gy_ini[165,33] = i_load_B6lv_a_r
        struct[0].Gy_ini[165,38] = i_load_B6lv_a_i
        struct[0].Gy_ini[165,39] = -i_load_B6lv_a_r
        struct[0].Gy_ini[165,162] = v_B6lv_a_i - v_B6lv_n_i
        struct[0].Gy_ini[165,163] = -v_B6lv_a_r + v_B6lv_n_r
        struct[0].Gy_ini[166,34] = -i_load_B6lv_b_i
        struct[0].Gy_ini[166,35] = i_load_B6lv_b_r
        struct[0].Gy_ini[166,38] = i_load_B6lv_b_i
        struct[0].Gy_ini[166,39] = -i_load_B6lv_b_r
        struct[0].Gy_ini[166,164] = v_B6lv_b_i - v_B6lv_n_i
        struct[0].Gy_ini[166,165] = -v_B6lv_b_r + v_B6lv_n_r
        struct[0].Gy_ini[167,36] = -i_load_B6lv_c_i
        struct[0].Gy_ini[167,37] = i_load_B6lv_c_r
        struct[0].Gy_ini[167,38] = i_load_B6lv_c_i
        struct[0].Gy_ini[167,39] = -i_load_B6lv_c_r
        struct[0].Gy_ini[167,166] = v_B6lv_c_i - v_B6lv_n_i
        struct[0].Gy_ini[167,167] = -v_B6lv_c_r + v_B6lv_n_r



@numba.njit(cache=True)
def run(t,struct,mode):

    # Parameters:
    
    # Inputs:
    v_B1_a_r = struct[0].v_B1_a_r
    v_B1_a_i = struct[0].v_B1_a_i
    v_B1_b_r = struct[0].v_B1_b_r
    v_B1_b_i = struct[0].v_B1_b_i
    v_B1_c_r = struct[0].v_B1_c_r
    v_B1_c_i = struct[0].v_B1_c_i
    v_B7_a_r = struct[0].v_B7_a_r
    v_B7_a_i = struct[0].v_B7_a_i
    v_B7_b_r = struct[0].v_B7_b_r
    v_B7_b_i = struct[0].v_B7_b_i
    v_B7_c_r = struct[0].v_B7_c_r
    v_B7_c_i = struct[0].v_B7_c_i
    i_B2lv_n_r = struct[0].i_B2lv_n_r
    i_B2lv_n_i = struct[0].i_B2lv_n_i
    i_B3lv_n_r = struct[0].i_B3lv_n_r
    i_B3lv_n_i = struct[0].i_B3lv_n_i
    i_B4lv_n_r = struct[0].i_B4lv_n_r
    i_B4lv_n_i = struct[0].i_B4lv_n_i
    i_B5lv_n_r = struct[0].i_B5lv_n_r
    i_B5lv_n_i = struct[0].i_B5lv_n_i
    i_B6lv_n_r = struct[0].i_B6lv_n_r
    i_B6lv_n_i = struct[0].i_B6lv_n_i
    i_B2_a_r = struct[0].i_B2_a_r
    i_B2_a_i = struct[0].i_B2_a_i
    i_B2_b_r = struct[0].i_B2_b_r
    i_B2_b_i = struct[0].i_B2_b_i
    i_B2_c_r = struct[0].i_B2_c_r
    i_B2_c_i = struct[0].i_B2_c_i
    i_B3_a_r = struct[0].i_B3_a_r
    i_B3_a_i = struct[0].i_B3_a_i
    i_B3_b_r = struct[0].i_B3_b_r
    i_B3_b_i = struct[0].i_B3_b_i
    i_B3_c_r = struct[0].i_B3_c_r
    i_B3_c_i = struct[0].i_B3_c_i
    i_B4_a_r = struct[0].i_B4_a_r
    i_B4_a_i = struct[0].i_B4_a_i
    i_B4_b_r = struct[0].i_B4_b_r
    i_B4_b_i = struct[0].i_B4_b_i
    i_B4_c_r = struct[0].i_B4_c_r
    i_B4_c_i = struct[0].i_B4_c_i
    i_B5_a_r = struct[0].i_B5_a_r
    i_B5_a_i = struct[0].i_B5_a_i
    i_B5_b_r = struct[0].i_B5_b_r
    i_B5_b_i = struct[0].i_B5_b_i
    i_B5_c_r = struct[0].i_B5_c_r
    i_B5_c_i = struct[0].i_B5_c_i
    i_B6_a_r = struct[0].i_B6_a_r
    i_B6_a_i = struct[0].i_B6_a_i
    i_B6_b_r = struct[0].i_B6_b_r
    i_B6_b_i = struct[0].i_B6_b_i
    i_B6_c_r = struct[0].i_B6_c_r
    i_B6_c_i = struct[0].i_B6_c_i
    p_B2lv_a = struct[0].p_B2lv_a
    q_B2lv_a = struct[0].q_B2lv_a
    p_B2lv_b = struct[0].p_B2lv_b
    q_B2lv_b = struct[0].q_B2lv_b
    p_B2lv_c = struct[0].p_B2lv_c
    q_B2lv_c = struct[0].q_B2lv_c
    p_B3lv_a = struct[0].p_B3lv_a
    q_B3lv_a = struct[0].q_B3lv_a
    p_B3lv_b = struct[0].p_B3lv_b
    q_B3lv_b = struct[0].q_B3lv_b
    p_B3lv_c = struct[0].p_B3lv_c
    q_B3lv_c = struct[0].q_B3lv_c
    p_B4lv_a = struct[0].p_B4lv_a
    q_B4lv_a = struct[0].q_B4lv_a
    p_B4lv_b = struct[0].p_B4lv_b
    q_B4lv_b = struct[0].q_B4lv_b
    p_B4lv_c = struct[0].p_B4lv_c
    q_B4lv_c = struct[0].q_B4lv_c
    p_B5lv_a = struct[0].p_B5lv_a
    q_B5lv_a = struct[0].q_B5lv_a
    p_B5lv_b = struct[0].p_B5lv_b
    q_B5lv_b = struct[0].q_B5lv_b
    p_B5lv_c = struct[0].p_B5lv_c
    q_B5lv_c = struct[0].q_B5lv_c
    p_B6lv_a = struct[0].p_B6lv_a
    q_B6lv_a = struct[0].q_B6lv_a
    p_B6lv_b = struct[0].p_B6lv_b
    q_B6lv_b = struct[0].q_B6lv_b
    p_B6lv_c = struct[0].p_B6lv_c
    q_B6lv_c = struct[0].q_B6lv_c
    u_dummy = struct[0].u_dummy
    
    # Dynamical states:
    x_dummy = struct[0].x[0,0]
    
    # Algebraic states:
    v_B2lv_a_r = struct[0].y_run[0,0]
    v_B2lv_a_i = struct[0].y_run[1,0]
    v_B2lv_b_r = struct[0].y_run[2,0]
    v_B2lv_b_i = struct[0].y_run[3,0]
    v_B2lv_c_r = struct[0].y_run[4,0]
    v_B2lv_c_i = struct[0].y_run[5,0]
    v_B2lv_n_r = struct[0].y_run[6,0]
    v_B2lv_n_i = struct[0].y_run[7,0]
    v_B3lv_a_r = struct[0].y_run[8,0]
    v_B3lv_a_i = struct[0].y_run[9,0]
    v_B3lv_b_r = struct[0].y_run[10,0]
    v_B3lv_b_i = struct[0].y_run[11,0]
    v_B3lv_c_r = struct[0].y_run[12,0]
    v_B3lv_c_i = struct[0].y_run[13,0]
    v_B3lv_n_r = struct[0].y_run[14,0]
    v_B3lv_n_i = struct[0].y_run[15,0]
    v_B4lv_a_r = struct[0].y_run[16,0]
    v_B4lv_a_i = struct[0].y_run[17,0]
    v_B4lv_b_r = struct[0].y_run[18,0]
    v_B4lv_b_i = struct[0].y_run[19,0]
    v_B4lv_c_r = struct[0].y_run[20,0]
    v_B4lv_c_i = struct[0].y_run[21,0]
    v_B4lv_n_r = struct[0].y_run[22,0]
    v_B4lv_n_i = struct[0].y_run[23,0]
    v_B5lv_a_r = struct[0].y_run[24,0]
    v_B5lv_a_i = struct[0].y_run[25,0]
    v_B5lv_b_r = struct[0].y_run[26,0]
    v_B5lv_b_i = struct[0].y_run[27,0]
    v_B5lv_c_r = struct[0].y_run[28,0]
    v_B5lv_c_i = struct[0].y_run[29,0]
    v_B5lv_n_r = struct[0].y_run[30,0]
    v_B5lv_n_i = struct[0].y_run[31,0]
    v_B6lv_a_r = struct[0].y_run[32,0]
    v_B6lv_a_i = struct[0].y_run[33,0]
    v_B6lv_b_r = struct[0].y_run[34,0]
    v_B6lv_b_i = struct[0].y_run[35,0]
    v_B6lv_c_r = struct[0].y_run[36,0]
    v_B6lv_c_i = struct[0].y_run[37,0]
    v_B6lv_n_r = struct[0].y_run[38,0]
    v_B6lv_n_i = struct[0].y_run[39,0]
    v_B2_a_r = struct[0].y_run[40,0]
    v_B2_a_i = struct[0].y_run[41,0]
    v_B2_b_r = struct[0].y_run[42,0]
    v_B2_b_i = struct[0].y_run[43,0]
    v_B2_c_r = struct[0].y_run[44,0]
    v_B2_c_i = struct[0].y_run[45,0]
    v_B3_a_r = struct[0].y_run[46,0]
    v_B3_a_i = struct[0].y_run[47,0]
    v_B3_b_r = struct[0].y_run[48,0]
    v_B3_b_i = struct[0].y_run[49,0]
    v_B3_c_r = struct[0].y_run[50,0]
    v_B3_c_i = struct[0].y_run[51,0]
    v_B4_a_r = struct[0].y_run[52,0]
    v_B4_a_i = struct[0].y_run[53,0]
    v_B4_b_r = struct[0].y_run[54,0]
    v_B4_b_i = struct[0].y_run[55,0]
    v_B4_c_r = struct[0].y_run[56,0]
    v_B4_c_i = struct[0].y_run[57,0]
    v_B5_a_r = struct[0].y_run[58,0]
    v_B5_a_i = struct[0].y_run[59,0]
    v_B5_b_r = struct[0].y_run[60,0]
    v_B5_b_i = struct[0].y_run[61,0]
    v_B5_c_r = struct[0].y_run[62,0]
    v_B5_c_i = struct[0].y_run[63,0]
    v_B6_a_r = struct[0].y_run[64,0]
    v_B6_a_i = struct[0].y_run[65,0]
    v_B6_b_r = struct[0].y_run[66,0]
    v_B6_b_i = struct[0].y_run[67,0]
    v_B6_c_r = struct[0].y_run[68,0]
    v_B6_c_i = struct[0].y_run[69,0]
    i_t_B2_B2lv_a_r = struct[0].y_run[70,0]
    i_t_B2_B2lv_a_i = struct[0].y_run[71,0]
    i_t_B2_B2lv_b_r = struct[0].y_run[72,0]
    i_t_B2_B2lv_b_i = struct[0].y_run[73,0]
    i_t_B2_B2lv_c_r = struct[0].y_run[74,0]
    i_t_B2_B2lv_c_i = struct[0].y_run[75,0]
    i_t_B3_B3lv_a_r = struct[0].y_run[76,0]
    i_t_B3_B3lv_a_i = struct[0].y_run[77,0]
    i_t_B3_B3lv_b_r = struct[0].y_run[78,0]
    i_t_B3_B3lv_b_i = struct[0].y_run[79,0]
    i_t_B3_B3lv_c_r = struct[0].y_run[80,0]
    i_t_B3_B3lv_c_i = struct[0].y_run[81,0]
    i_t_B4_B4lv_a_r = struct[0].y_run[82,0]
    i_t_B4_B4lv_a_i = struct[0].y_run[83,0]
    i_t_B4_B4lv_b_r = struct[0].y_run[84,0]
    i_t_B4_B4lv_b_i = struct[0].y_run[85,0]
    i_t_B4_B4lv_c_r = struct[0].y_run[86,0]
    i_t_B4_B4lv_c_i = struct[0].y_run[87,0]
    i_t_B5_B5lv_a_r = struct[0].y_run[88,0]
    i_t_B5_B5lv_a_i = struct[0].y_run[89,0]
    i_t_B5_B5lv_b_r = struct[0].y_run[90,0]
    i_t_B5_B5lv_b_i = struct[0].y_run[91,0]
    i_t_B5_B5lv_c_r = struct[0].y_run[92,0]
    i_t_B5_B5lv_c_i = struct[0].y_run[93,0]
    i_t_B6_B6lv_a_r = struct[0].y_run[94,0]
    i_t_B6_B6lv_a_i = struct[0].y_run[95,0]
    i_t_B6_B6lv_b_r = struct[0].y_run[96,0]
    i_t_B6_B6lv_b_i = struct[0].y_run[97,0]
    i_t_B6_B6lv_c_r = struct[0].y_run[98,0]
    i_t_B6_B6lv_c_i = struct[0].y_run[99,0]
    i_l_B1_B2_a_r = struct[0].y_run[100,0]
    i_l_B1_B2_a_i = struct[0].y_run[101,0]
    i_l_B1_B2_b_r = struct[0].y_run[102,0]
    i_l_B1_B2_b_i = struct[0].y_run[103,0]
    i_l_B1_B2_c_r = struct[0].y_run[104,0]
    i_l_B1_B2_c_i = struct[0].y_run[105,0]
    i_l_B2_B3_a_r = struct[0].y_run[106,0]
    i_l_B2_B3_a_i = struct[0].y_run[107,0]
    i_l_B2_B3_b_r = struct[0].y_run[108,0]
    i_l_B2_B3_b_i = struct[0].y_run[109,0]
    i_l_B2_B3_c_r = struct[0].y_run[110,0]
    i_l_B2_B3_c_i = struct[0].y_run[111,0]
    i_l_B3_B4_a_r = struct[0].y_run[112,0]
    i_l_B3_B4_a_i = struct[0].y_run[113,0]
    i_l_B3_B4_b_r = struct[0].y_run[114,0]
    i_l_B3_B4_b_i = struct[0].y_run[115,0]
    i_l_B3_B4_c_r = struct[0].y_run[116,0]
    i_l_B3_B4_c_i = struct[0].y_run[117,0]
    i_l_B5_B6_a_r = struct[0].y_run[118,0]
    i_l_B5_B6_a_i = struct[0].y_run[119,0]
    i_l_B5_B6_b_r = struct[0].y_run[120,0]
    i_l_B5_B6_b_i = struct[0].y_run[121,0]
    i_l_B5_B6_c_r = struct[0].y_run[122,0]
    i_l_B5_B6_c_i = struct[0].y_run[123,0]
    i_l_B6_B7_a_r = struct[0].y_run[124,0]
    i_l_B6_B7_a_i = struct[0].y_run[125,0]
    i_l_B6_B7_b_r = struct[0].y_run[126,0]
    i_l_B6_B7_b_i = struct[0].y_run[127,0]
    i_l_B6_B7_c_r = struct[0].y_run[128,0]
    i_l_B6_B7_c_i = struct[0].y_run[129,0]
    i_load_B2lv_a_r = struct[0].y_run[130,0]
    i_load_B2lv_a_i = struct[0].y_run[131,0]
    i_load_B2lv_b_r = struct[0].y_run[132,0]
    i_load_B2lv_b_i = struct[0].y_run[133,0]
    i_load_B2lv_c_r = struct[0].y_run[134,0]
    i_load_B2lv_c_i = struct[0].y_run[135,0]
    i_load_B2lv_n_r = struct[0].y_run[136,0]
    i_load_B2lv_n_i = struct[0].y_run[137,0]
    i_load_B3lv_a_r = struct[0].y_run[138,0]
    i_load_B3lv_a_i = struct[0].y_run[139,0]
    i_load_B3lv_b_r = struct[0].y_run[140,0]
    i_load_B3lv_b_i = struct[0].y_run[141,0]
    i_load_B3lv_c_r = struct[0].y_run[142,0]
    i_load_B3lv_c_i = struct[0].y_run[143,0]
    i_load_B3lv_n_r = struct[0].y_run[144,0]
    i_load_B3lv_n_i = struct[0].y_run[145,0]
    i_load_B4lv_a_r = struct[0].y_run[146,0]
    i_load_B4lv_a_i = struct[0].y_run[147,0]
    i_load_B4lv_b_r = struct[0].y_run[148,0]
    i_load_B4lv_b_i = struct[0].y_run[149,0]
    i_load_B4lv_c_r = struct[0].y_run[150,0]
    i_load_B4lv_c_i = struct[0].y_run[151,0]
    i_load_B4lv_n_r = struct[0].y_run[152,0]
    i_load_B4lv_n_i = struct[0].y_run[153,0]
    i_load_B5lv_a_r = struct[0].y_run[154,0]
    i_load_B5lv_a_i = struct[0].y_run[155,0]
    i_load_B5lv_b_r = struct[0].y_run[156,0]
    i_load_B5lv_b_i = struct[0].y_run[157,0]
    i_load_B5lv_c_r = struct[0].y_run[158,0]
    i_load_B5lv_c_i = struct[0].y_run[159,0]
    i_load_B5lv_n_r = struct[0].y_run[160,0]
    i_load_B5lv_n_i = struct[0].y_run[161,0]
    i_load_B6lv_a_r = struct[0].y_run[162,0]
    i_load_B6lv_a_i = struct[0].y_run[163,0]
    i_load_B6lv_b_r = struct[0].y_run[164,0]
    i_load_B6lv_b_i = struct[0].y_run[165,0]
    i_load_B6lv_c_r = struct[0].y_run[166,0]
    i_load_B6lv_c_i = struct[0].y_run[167,0]
    i_load_B6lv_n_r = struct[0].y_run[168,0]
    i_load_B6lv_n_i = struct[0].y_run[169,0]
    
    struct[0].u_run[0,0] = v_B1_a_r
    struct[0].u_run[1,0] = v_B1_a_i
    struct[0].u_run[2,0] = v_B1_b_r
    struct[0].u_run[3,0] = v_B1_b_i
    struct[0].u_run[4,0] = v_B1_c_r
    struct[0].u_run[5,0] = v_B1_c_i
    struct[0].u_run[6,0] = v_B7_a_r
    struct[0].u_run[7,0] = v_B7_a_i
    struct[0].u_run[8,0] = v_B7_b_r
    struct[0].u_run[9,0] = v_B7_b_i
    struct[0].u_run[10,0] = v_B7_c_r
    struct[0].u_run[11,0] = v_B7_c_i
    struct[0].u_run[12,0] = i_B2lv_n_r
    struct[0].u_run[13,0] = i_B2lv_n_i
    struct[0].u_run[14,0] = i_B3lv_n_r
    struct[0].u_run[15,0] = i_B3lv_n_i
    struct[0].u_run[16,0] = i_B4lv_n_r
    struct[0].u_run[17,0] = i_B4lv_n_i
    struct[0].u_run[18,0] = i_B5lv_n_r
    struct[0].u_run[19,0] = i_B5lv_n_i
    struct[0].u_run[20,0] = i_B6lv_n_r
    struct[0].u_run[21,0] = i_B6lv_n_i
    struct[0].u_run[22,0] = i_B2_a_r
    struct[0].u_run[23,0] = i_B2_a_i
    struct[0].u_run[24,0] = i_B2_b_r
    struct[0].u_run[25,0] = i_B2_b_i
    struct[0].u_run[26,0] = i_B2_c_r
    struct[0].u_run[27,0] = i_B2_c_i
    struct[0].u_run[28,0] = i_B3_a_r
    struct[0].u_run[29,0] = i_B3_a_i
    struct[0].u_run[30,0] = i_B3_b_r
    struct[0].u_run[31,0] = i_B3_b_i
    struct[0].u_run[32,0] = i_B3_c_r
    struct[0].u_run[33,0] = i_B3_c_i
    struct[0].u_run[34,0] = i_B4_a_r
    struct[0].u_run[35,0] = i_B4_a_i
    struct[0].u_run[36,0] = i_B4_b_r
    struct[0].u_run[37,0] = i_B4_b_i
    struct[0].u_run[38,0] = i_B4_c_r
    struct[0].u_run[39,0] = i_B4_c_i
    struct[0].u_run[40,0] = i_B5_a_r
    struct[0].u_run[41,0] = i_B5_a_i
    struct[0].u_run[42,0] = i_B5_b_r
    struct[0].u_run[43,0] = i_B5_b_i
    struct[0].u_run[44,0] = i_B5_c_r
    struct[0].u_run[45,0] = i_B5_c_i
    struct[0].u_run[46,0] = i_B6_a_r
    struct[0].u_run[47,0] = i_B6_a_i
    struct[0].u_run[48,0] = i_B6_b_r
    struct[0].u_run[49,0] = i_B6_b_i
    struct[0].u_run[50,0] = i_B6_c_r
    struct[0].u_run[51,0] = i_B6_c_i
    struct[0].u_run[52,0] = p_B2lv_a
    struct[0].u_run[53,0] = q_B2lv_a
    struct[0].u_run[54,0] = p_B2lv_b
    struct[0].u_run[55,0] = q_B2lv_b
    struct[0].u_run[56,0] = p_B2lv_c
    struct[0].u_run[57,0] = q_B2lv_c
    struct[0].u_run[58,0] = p_B3lv_a
    struct[0].u_run[59,0] = q_B3lv_a
    struct[0].u_run[60,0] = p_B3lv_b
    struct[0].u_run[61,0] = q_B3lv_b
    struct[0].u_run[62,0] = p_B3lv_c
    struct[0].u_run[63,0] = q_B3lv_c
    struct[0].u_run[64,0] = p_B4lv_a
    struct[0].u_run[65,0] = q_B4lv_a
    struct[0].u_run[66,0] = p_B4lv_b
    struct[0].u_run[67,0] = q_B4lv_b
    struct[0].u_run[68,0] = p_B4lv_c
    struct[0].u_run[69,0] = q_B4lv_c
    struct[0].u_run[70,0] = p_B5lv_a
    struct[0].u_run[71,0] = q_B5lv_a
    struct[0].u_run[72,0] = p_B5lv_b
    struct[0].u_run[73,0] = q_B5lv_b
    struct[0].u_run[74,0] = p_B5lv_c
    struct[0].u_run[75,0] = q_B5lv_c
    struct[0].u_run[76,0] = p_B6lv_a
    struct[0].u_run[77,0] = q_B6lv_a
    struct[0].u_run[78,0] = p_B6lv_b
    struct[0].u_run[79,0] = q_B6lv_b
    struct[0].u_run[80,0] = p_B6lv_c
    struct[0].u_run[81,0] = q_B6lv_c
    struct[0].u_run[82,0] = u_dummy
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = u_dummy - x_dummy
    
    # Algebraic equations:
    if mode == 3:

        struct[0].g[:,:] = np.ascontiguousarray(struct[0].Gy) @ np.ascontiguousarray(struct[0].y_run) + np.ascontiguousarray(struct[0].Gu) @ np.ascontiguousarray(struct[0].u_run)

        struct[0].g[130,0] = i_load_B2lv_a_i*v_B2lv_a_i - i_load_B2lv_a_i*v_B2lv_n_i + i_load_B2lv_a_r*v_B2lv_a_r - i_load_B2lv_a_r*v_B2lv_n_r - p_B2lv_a
        struct[0].g[131,0] = i_load_B2lv_b_i*v_B2lv_b_i - i_load_B2lv_b_i*v_B2lv_n_i + i_load_B2lv_b_r*v_B2lv_b_r - i_load_B2lv_b_r*v_B2lv_n_r - p_B2lv_b
        struct[0].g[132,0] = i_load_B2lv_c_i*v_B2lv_c_i - i_load_B2lv_c_i*v_B2lv_n_i + i_load_B2lv_c_r*v_B2lv_c_r - i_load_B2lv_c_r*v_B2lv_n_r - p_B2lv_c
        struct[0].g[133,0] = -i_load_B2lv_a_i*v_B2lv_a_r + i_load_B2lv_a_i*v_B2lv_n_r + i_load_B2lv_a_r*v_B2lv_a_i - i_load_B2lv_a_r*v_B2lv_n_i - q_B2lv_a
        struct[0].g[134,0] = -i_load_B2lv_b_i*v_B2lv_b_r + i_load_B2lv_b_i*v_B2lv_n_r + i_load_B2lv_b_r*v_B2lv_b_i - i_load_B2lv_b_r*v_B2lv_n_i - q_B2lv_b
        struct[0].g[135,0] = -i_load_B2lv_c_i*v_B2lv_c_r + i_load_B2lv_c_i*v_B2lv_n_r + i_load_B2lv_c_r*v_B2lv_c_i - i_load_B2lv_c_r*v_B2lv_n_i - q_B2lv_c
        struct[0].g[138,0] = i_load_B3lv_a_i*v_B3lv_a_i - i_load_B3lv_a_i*v_B3lv_n_i + i_load_B3lv_a_r*v_B3lv_a_r - i_load_B3lv_a_r*v_B3lv_n_r - p_B3lv_a
        struct[0].g[139,0] = i_load_B3lv_b_i*v_B3lv_b_i - i_load_B3lv_b_i*v_B3lv_n_i + i_load_B3lv_b_r*v_B3lv_b_r - i_load_B3lv_b_r*v_B3lv_n_r - p_B3lv_b
        struct[0].g[140,0] = i_load_B3lv_c_i*v_B3lv_c_i - i_load_B3lv_c_i*v_B3lv_n_i + i_load_B3lv_c_r*v_B3lv_c_r - i_load_B3lv_c_r*v_B3lv_n_r - p_B3lv_c
        struct[0].g[141,0] = -i_load_B3lv_a_i*v_B3lv_a_r + i_load_B3lv_a_i*v_B3lv_n_r + i_load_B3lv_a_r*v_B3lv_a_i - i_load_B3lv_a_r*v_B3lv_n_i - q_B3lv_a
        struct[0].g[142,0] = -i_load_B3lv_b_i*v_B3lv_b_r + i_load_B3lv_b_i*v_B3lv_n_r + i_load_B3lv_b_r*v_B3lv_b_i - i_load_B3lv_b_r*v_B3lv_n_i - q_B3lv_b
        struct[0].g[143,0] = -i_load_B3lv_c_i*v_B3lv_c_r + i_load_B3lv_c_i*v_B3lv_n_r + i_load_B3lv_c_r*v_B3lv_c_i - i_load_B3lv_c_r*v_B3lv_n_i - q_B3lv_c
        struct[0].g[146,0] = i_load_B4lv_a_i*v_B4lv_a_i - i_load_B4lv_a_i*v_B4lv_n_i + i_load_B4lv_a_r*v_B4lv_a_r - i_load_B4lv_a_r*v_B4lv_n_r - p_B4lv_a
        struct[0].g[147,0] = i_load_B4lv_b_i*v_B4lv_b_i - i_load_B4lv_b_i*v_B4lv_n_i + i_load_B4lv_b_r*v_B4lv_b_r - i_load_B4lv_b_r*v_B4lv_n_r - p_B4lv_b
        struct[0].g[148,0] = i_load_B4lv_c_i*v_B4lv_c_i - i_load_B4lv_c_i*v_B4lv_n_i + i_load_B4lv_c_r*v_B4lv_c_r - i_load_B4lv_c_r*v_B4lv_n_r - p_B4lv_c
        struct[0].g[149,0] = -i_load_B4lv_a_i*v_B4lv_a_r + i_load_B4lv_a_i*v_B4lv_n_r + i_load_B4lv_a_r*v_B4lv_a_i - i_load_B4lv_a_r*v_B4lv_n_i - q_B4lv_a
        struct[0].g[150,0] = -i_load_B4lv_b_i*v_B4lv_b_r + i_load_B4lv_b_i*v_B4lv_n_r + i_load_B4lv_b_r*v_B4lv_b_i - i_load_B4lv_b_r*v_B4lv_n_i - q_B4lv_b
        struct[0].g[151,0] = -i_load_B4lv_c_i*v_B4lv_c_r + i_load_B4lv_c_i*v_B4lv_n_r + i_load_B4lv_c_r*v_B4lv_c_i - i_load_B4lv_c_r*v_B4lv_n_i - q_B4lv_c
        struct[0].g[154,0] = i_load_B5lv_a_i*v_B5lv_a_i - i_load_B5lv_a_i*v_B5lv_n_i + i_load_B5lv_a_r*v_B5lv_a_r - i_load_B5lv_a_r*v_B5lv_n_r - p_B5lv_a
        struct[0].g[155,0] = i_load_B5lv_b_i*v_B5lv_b_i - i_load_B5lv_b_i*v_B5lv_n_i + i_load_B5lv_b_r*v_B5lv_b_r - i_load_B5lv_b_r*v_B5lv_n_r - p_B5lv_b
        struct[0].g[156,0] = i_load_B5lv_c_i*v_B5lv_c_i - i_load_B5lv_c_i*v_B5lv_n_i + i_load_B5lv_c_r*v_B5lv_c_r - i_load_B5lv_c_r*v_B5lv_n_r - p_B5lv_c
        struct[0].g[157,0] = -i_load_B5lv_a_i*v_B5lv_a_r + i_load_B5lv_a_i*v_B5lv_n_r + i_load_B5lv_a_r*v_B5lv_a_i - i_load_B5lv_a_r*v_B5lv_n_i - q_B5lv_a
        struct[0].g[158,0] = -i_load_B5lv_b_i*v_B5lv_b_r + i_load_B5lv_b_i*v_B5lv_n_r + i_load_B5lv_b_r*v_B5lv_b_i - i_load_B5lv_b_r*v_B5lv_n_i - q_B5lv_b
        struct[0].g[159,0] = -i_load_B5lv_c_i*v_B5lv_c_r + i_load_B5lv_c_i*v_B5lv_n_r + i_load_B5lv_c_r*v_B5lv_c_i - i_load_B5lv_c_r*v_B5lv_n_i - q_B5lv_c
        struct[0].g[162,0] = i_load_B6lv_a_i*v_B6lv_a_i - i_load_B6lv_a_i*v_B6lv_n_i + i_load_B6lv_a_r*v_B6lv_a_r - i_load_B6lv_a_r*v_B6lv_n_r - p_B6lv_a
        struct[0].g[163,0] = i_load_B6lv_b_i*v_B6lv_b_i - i_load_B6lv_b_i*v_B6lv_n_i + i_load_B6lv_b_r*v_B6lv_b_r - i_load_B6lv_b_r*v_B6lv_n_r - p_B6lv_b
        struct[0].g[164,0] = i_load_B6lv_c_i*v_B6lv_c_i - i_load_B6lv_c_i*v_B6lv_n_i + i_load_B6lv_c_r*v_B6lv_c_r - i_load_B6lv_c_r*v_B6lv_n_r - p_B6lv_c
        struct[0].g[165,0] = -i_load_B6lv_a_i*v_B6lv_a_r + i_load_B6lv_a_i*v_B6lv_n_r + i_load_B6lv_a_r*v_B6lv_a_i - i_load_B6lv_a_r*v_B6lv_n_i - q_B6lv_a
        struct[0].g[166,0] = -i_load_B6lv_b_i*v_B6lv_b_r + i_load_B6lv_b_i*v_B6lv_n_r + i_load_B6lv_b_r*v_B6lv_b_i - i_load_B6lv_b_r*v_B6lv_n_i - q_B6lv_b
        struct[0].g[167,0] = -i_load_B6lv_c_i*v_B6lv_c_r + i_load_B6lv_c_i*v_B6lv_n_r + i_load_B6lv_c_r*v_B6lv_c_i - i_load_B6lv_c_r*v_B6lv_n_i - q_B6lv_c
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = (v_B1_a_i**2 + v_B1_a_r**2)**0.5
        struct[0].h[1,0] = (v_B1_b_i**2 + v_B1_b_r**2)**0.5
        struct[0].h[2,0] = (v_B1_c_i**2 + v_B1_c_r**2)**0.5
        struct[0].h[3,0] = (v_B7_a_i**2 + v_B7_a_r**2)**0.5
        struct[0].h[4,0] = (v_B7_b_i**2 + v_B7_b_r**2)**0.5
        struct[0].h[5,0] = (v_B7_c_i**2 + v_B7_c_r**2)**0.5
        struct[0].h[6,0] = (v_B2lv_a_i**2 + v_B2lv_a_r**2)**0.5
        struct[0].h[7,0] = (v_B2lv_b_i**2 + v_B2lv_b_r**2)**0.5
        struct[0].h[8,0] = (v_B2lv_c_i**2 + v_B2lv_c_r**2)**0.5
        struct[0].h[9,0] = (v_B2lv_n_i**2 + v_B2lv_n_r**2)**0.5
        struct[0].h[10,0] = (v_B3lv_a_i**2 + v_B3lv_a_r**2)**0.5
        struct[0].h[11,0] = (v_B3lv_b_i**2 + v_B3lv_b_r**2)**0.5
        struct[0].h[12,0] = (v_B3lv_c_i**2 + v_B3lv_c_r**2)**0.5
        struct[0].h[13,0] = (v_B3lv_n_i**2 + v_B3lv_n_r**2)**0.5
        struct[0].h[14,0] = (v_B4lv_a_i**2 + v_B4lv_a_r**2)**0.5
        struct[0].h[15,0] = (v_B4lv_b_i**2 + v_B4lv_b_r**2)**0.5
        struct[0].h[16,0] = (v_B4lv_c_i**2 + v_B4lv_c_r**2)**0.5
        struct[0].h[17,0] = (v_B4lv_n_i**2 + v_B4lv_n_r**2)**0.5
        struct[0].h[18,0] = (v_B5lv_a_i**2 + v_B5lv_a_r**2)**0.5
        struct[0].h[19,0] = (v_B5lv_b_i**2 + v_B5lv_b_r**2)**0.5
        struct[0].h[20,0] = (v_B5lv_c_i**2 + v_B5lv_c_r**2)**0.5
        struct[0].h[21,0] = (v_B5lv_n_i**2 + v_B5lv_n_r**2)**0.5
        struct[0].h[22,0] = (v_B6lv_a_i**2 + v_B6lv_a_r**2)**0.5
        struct[0].h[23,0] = (v_B6lv_b_i**2 + v_B6lv_b_r**2)**0.5
        struct[0].h[24,0] = (v_B6lv_c_i**2 + v_B6lv_c_r**2)**0.5
        struct[0].h[25,0] = (v_B6lv_n_i**2 + v_B6lv_n_r**2)**0.5
        struct[0].h[26,0] = (v_B2_a_i**2 + v_B2_a_r**2)**0.5
        struct[0].h[27,0] = (v_B2_b_i**2 + v_B2_b_r**2)**0.5
        struct[0].h[28,0] = (v_B2_c_i**2 + v_B2_c_r**2)**0.5
        struct[0].h[29,0] = (v_B3_a_i**2 + v_B3_a_r**2)**0.5
        struct[0].h[30,0] = (v_B3_b_i**2 + v_B3_b_r**2)**0.5
        struct[0].h[31,0] = (v_B3_c_i**2 + v_B3_c_r**2)**0.5
        struct[0].h[32,0] = (v_B4_a_i**2 + v_B4_a_r**2)**0.5
        struct[0].h[33,0] = (v_B4_b_i**2 + v_B4_b_r**2)**0.5
        struct[0].h[34,0] = (v_B4_c_i**2 + v_B4_c_r**2)**0.5
        struct[0].h[35,0] = (v_B5_a_i**2 + v_B5_a_r**2)**0.5
        struct[0].h[36,0] = (v_B5_b_i**2 + v_B5_b_r**2)**0.5
        struct[0].h[37,0] = (v_B5_c_i**2 + v_B5_c_r**2)**0.5
        struct[0].h[38,0] = (v_B6_a_i**2 + v_B6_a_r**2)**0.5
        struct[0].h[39,0] = (v_B6_b_i**2 + v_B6_b_r**2)**0.5
        struct[0].h[40,0] = (v_B6_c_i**2 + v_B6_c_r**2)**0.5
    

    if mode == 10:

        pass

    if mode == 11:



        struct[0].Gy[130,0] = i_load_B2lv_a_r
        struct[0].Gy[130,1] = i_load_B2lv_a_i
        struct[0].Gy[130,6] = -i_load_B2lv_a_r
        struct[0].Gy[130,7] = -i_load_B2lv_a_i
        struct[0].Gy[130,130] = v_B2lv_a_r - v_B2lv_n_r
        struct[0].Gy[130,131] = v_B2lv_a_i - v_B2lv_n_i
        struct[0].Gy[131,2] = i_load_B2lv_b_r
        struct[0].Gy[131,3] = i_load_B2lv_b_i
        struct[0].Gy[131,6] = -i_load_B2lv_b_r
        struct[0].Gy[131,7] = -i_load_B2lv_b_i
        struct[0].Gy[131,132] = v_B2lv_b_r - v_B2lv_n_r
        struct[0].Gy[131,133] = v_B2lv_b_i - v_B2lv_n_i
        struct[0].Gy[132,4] = i_load_B2lv_c_r
        struct[0].Gy[132,5] = i_load_B2lv_c_i
        struct[0].Gy[132,6] = -i_load_B2lv_c_r
        struct[0].Gy[132,7] = -i_load_B2lv_c_i
        struct[0].Gy[132,134] = v_B2lv_c_r - v_B2lv_n_r
        struct[0].Gy[132,135] = v_B2lv_c_i - v_B2lv_n_i
        struct[0].Gy[133,0] = -i_load_B2lv_a_i
        struct[0].Gy[133,1] = i_load_B2lv_a_r
        struct[0].Gy[133,6] = i_load_B2lv_a_i
        struct[0].Gy[133,7] = -i_load_B2lv_a_r
        struct[0].Gy[133,130] = v_B2lv_a_i - v_B2lv_n_i
        struct[0].Gy[133,131] = -v_B2lv_a_r + v_B2lv_n_r
        struct[0].Gy[134,2] = -i_load_B2lv_b_i
        struct[0].Gy[134,3] = i_load_B2lv_b_r
        struct[0].Gy[134,6] = i_load_B2lv_b_i
        struct[0].Gy[134,7] = -i_load_B2lv_b_r
        struct[0].Gy[134,132] = v_B2lv_b_i - v_B2lv_n_i
        struct[0].Gy[134,133] = -v_B2lv_b_r + v_B2lv_n_r
        struct[0].Gy[135,4] = -i_load_B2lv_c_i
        struct[0].Gy[135,5] = i_load_B2lv_c_r
        struct[0].Gy[135,6] = i_load_B2lv_c_i
        struct[0].Gy[135,7] = -i_load_B2lv_c_r
        struct[0].Gy[135,134] = v_B2lv_c_i - v_B2lv_n_i
        struct[0].Gy[135,135] = -v_B2lv_c_r + v_B2lv_n_r
        struct[0].Gy[138,8] = i_load_B3lv_a_r
        struct[0].Gy[138,9] = i_load_B3lv_a_i
        struct[0].Gy[138,14] = -i_load_B3lv_a_r
        struct[0].Gy[138,15] = -i_load_B3lv_a_i
        struct[0].Gy[138,138] = v_B3lv_a_r - v_B3lv_n_r
        struct[0].Gy[138,139] = v_B3lv_a_i - v_B3lv_n_i
        struct[0].Gy[139,10] = i_load_B3lv_b_r
        struct[0].Gy[139,11] = i_load_B3lv_b_i
        struct[0].Gy[139,14] = -i_load_B3lv_b_r
        struct[0].Gy[139,15] = -i_load_B3lv_b_i
        struct[0].Gy[139,140] = v_B3lv_b_r - v_B3lv_n_r
        struct[0].Gy[139,141] = v_B3lv_b_i - v_B3lv_n_i
        struct[0].Gy[140,12] = i_load_B3lv_c_r
        struct[0].Gy[140,13] = i_load_B3lv_c_i
        struct[0].Gy[140,14] = -i_load_B3lv_c_r
        struct[0].Gy[140,15] = -i_load_B3lv_c_i
        struct[0].Gy[140,142] = v_B3lv_c_r - v_B3lv_n_r
        struct[0].Gy[140,143] = v_B3lv_c_i - v_B3lv_n_i
        struct[0].Gy[141,8] = -i_load_B3lv_a_i
        struct[0].Gy[141,9] = i_load_B3lv_a_r
        struct[0].Gy[141,14] = i_load_B3lv_a_i
        struct[0].Gy[141,15] = -i_load_B3lv_a_r
        struct[0].Gy[141,138] = v_B3lv_a_i - v_B3lv_n_i
        struct[0].Gy[141,139] = -v_B3lv_a_r + v_B3lv_n_r
        struct[0].Gy[142,10] = -i_load_B3lv_b_i
        struct[0].Gy[142,11] = i_load_B3lv_b_r
        struct[0].Gy[142,14] = i_load_B3lv_b_i
        struct[0].Gy[142,15] = -i_load_B3lv_b_r
        struct[0].Gy[142,140] = v_B3lv_b_i - v_B3lv_n_i
        struct[0].Gy[142,141] = -v_B3lv_b_r + v_B3lv_n_r
        struct[0].Gy[143,12] = -i_load_B3lv_c_i
        struct[0].Gy[143,13] = i_load_B3lv_c_r
        struct[0].Gy[143,14] = i_load_B3lv_c_i
        struct[0].Gy[143,15] = -i_load_B3lv_c_r
        struct[0].Gy[143,142] = v_B3lv_c_i - v_B3lv_n_i
        struct[0].Gy[143,143] = -v_B3lv_c_r + v_B3lv_n_r
        struct[0].Gy[146,16] = i_load_B4lv_a_r
        struct[0].Gy[146,17] = i_load_B4lv_a_i
        struct[0].Gy[146,22] = -i_load_B4lv_a_r
        struct[0].Gy[146,23] = -i_load_B4lv_a_i
        struct[0].Gy[146,146] = v_B4lv_a_r - v_B4lv_n_r
        struct[0].Gy[146,147] = v_B4lv_a_i - v_B4lv_n_i
        struct[0].Gy[147,18] = i_load_B4lv_b_r
        struct[0].Gy[147,19] = i_load_B4lv_b_i
        struct[0].Gy[147,22] = -i_load_B4lv_b_r
        struct[0].Gy[147,23] = -i_load_B4lv_b_i
        struct[0].Gy[147,148] = v_B4lv_b_r - v_B4lv_n_r
        struct[0].Gy[147,149] = v_B4lv_b_i - v_B4lv_n_i
        struct[0].Gy[148,20] = i_load_B4lv_c_r
        struct[0].Gy[148,21] = i_load_B4lv_c_i
        struct[0].Gy[148,22] = -i_load_B4lv_c_r
        struct[0].Gy[148,23] = -i_load_B4lv_c_i
        struct[0].Gy[148,150] = v_B4lv_c_r - v_B4lv_n_r
        struct[0].Gy[148,151] = v_B4lv_c_i - v_B4lv_n_i
        struct[0].Gy[149,16] = -i_load_B4lv_a_i
        struct[0].Gy[149,17] = i_load_B4lv_a_r
        struct[0].Gy[149,22] = i_load_B4lv_a_i
        struct[0].Gy[149,23] = -i_load_B4lv_a_r
        struct[0].Gy[149,146] = v_B4lv_a_i - v_B4lv_n_i
        struct[0].Gy[149,147] = -v_B4lv_a_r + v_B4lv_n_r
        struct[0].Gy[150,18] = -i_load_B4lv_b_i
        struct[0].Gy[150,19] = i_load_B4lv_b_r
        struct[0].Gy[150,22] = i_load_B4lv_b_i
        struct[0].Gy[150,23] = -i_load_B4lv_b_r
        struct[0].Gy[150,148] = v_B4lv_b_i - v_B4lv_n_i
        struct[0].Gy[150,149] = -v_B4lv_b_r + v_B4lv_n_r
        struct[0].Gy[151,20] = -i_load_B4lv_c_i
        struct[0].Gy[151,21] = i_load_B4lv_c_r
        struct[0].Gy[151,22] = i_load_B4lv_c_i
        struct[0].Gy[151,23] = -i_load_B4lv_c_r
        struct[0].Gy[151,150] = v_B4lv_c_i - v_B4lv_n_i
        struct[0].Gy[151,151] = -v_B4lv_c_r + v_B4lv_n_r
        struct[0].Gy[154,24] = i_load_B5lv_a_r
        struct[0].Gy[154,25] = i_load_B5lv_a_i
        struct[0].Gy[154,30] = -i_load_B5lv_a_r
        struct[0].Gy[154,31] = -i_load_B5lv_a_i
        struct[0].Gy[154,154] = v_B5lv_a_r - v_B5lv_n_r
        struct[0].Gy[154,155] = v_B5lv_a_i - v_B5lv_n_i
        struct[0].Gy[155,26] = i_load_B5lv_b_r
        struct[0].Gy[155,27] = i_load_B5lv_b_i
        struct[0].Gy[155,30] = -i_load_B5lv_b_r
        struct[0].Gy[155,31] = -i_load_B5lv_b_i
        struct[0].Gy[155,156] = v_B5lv_b_r - v_B5lv_n_r
        struct[0].Gy[155,157] = v_B5lv_b_i - v_B5lv_n_i
        struct[0].Gy[156,28] = i_load_B5lv_c_r
        struct[0].Gy[156,29] = i_load_B5lv_c_i
        struct[0].Gy[156,30] = -i_load_B5lv_c_r
        struct[0].Gy[156,31] = -i_load_B5lv_c_i
        struct[0].Gy[156,158] = v_B5lv_c_r - v_B5lv_n_r
        struct[0].Gy[156,159] = v_B5lv_c_i - v_B5lv_n_i
        struct[0].Gy[157,24] = -i_load_B5lv_a_i
        struct[0].Gy[157,25] = i_load_B5lv_a_r
        struct[0].Gy[157,30] = i_load_B5lv_a_i
        struct[0].Gy[157,31] = -i_load_B5lv_a_r
        struct[0].Gy[157,154] = v_B5lv_a_i - v_B5lv_n_i
        struct[0].Gy[157,155] = -v_B5lv_a_r + v_B5lv_n_r
        struct[0].Gy[158,26] = -i_load_B5lv_b_i
        struct[0].Gy[158,27] = i_load_B5lv_b_r
        struct[0].Gy[158,30] = i_load_B5lv_b_i
        struct[0].Gy[158,31] = -i_load_B5lv_b_r
        struct[0].Gy[158,156] = v_B5lv_b_i - v_B5lv_n_i
        struct[0].Gy[158,157] = -v_B5lv_b_r + v_B5lv_n_r
        struct[0].Gy[159,28] = -i_load_B5lv_c_i
        struct[0].Gy[159,29] = i_load_B5lv_c_r
        struct[0].Gy[159,30] = i_load_B5lv_c_i
        struct[0].Gy[159,31] = -i_load_B5lv_c_r
        struct[0].Gy[159,158] = v_B5lv_c_i - v_B5lv_n_i
        struct[0].Gy[159,159] = -v_B5lv_c_r + v_B5lv_n_r
        struct[0].Gy[162,32] = i_load_B6lv_a_r
        struct[0].Gy[162,33] = i_load_B6lv_a_i
        struct[0].Gy[162,38] = -i_load_B6lv_a_r
        struct[0].Gy[162,39] = -i_load_B6lv_a_i
        struct[0].Gy[162,162] = v_B6lv_a_r - v_B6lv_n_r
        struct[0].Gy[162,163] = v_B6lv_a_i - v_B6lv_n_i
        struct[0].Gy[163,34] = i_load_B6lv_b_r
        struct[0].Gy[163,35] = i_load_B6lv_b_i
        struct[0].Gy[163,38] = -i_load_B6lv_b_r
        struct[0].Gy[163,39] = -i_load_B6lv_b_i
        struct[0].Gy[163,164] = v_B6lv_b_r - v_B6lv_n_r
        struct[0].Gy[163,165] = v_B6lv_b_i - v_B6lv_n_i
        struct[0].Gy[164,36] = i_load_B6lv_c_r
        struct[0].Gy[164,37] = i_load_B6lv_c_i
        struct[0].Gy[164,38] = -i_load_B6lv_c_r
        struct[0].Gy[164,39] = -i_load_B6lv_c_i
        struct[0].Gy[164,166] = v_B6lv_c_r - v_B6lv_n_r
        struct[0].Gy[164,167] = v_B6lv_c_i - v_B6lv_n_i
        struct[0].Gy[165,32] = -i_load_B6lv_a_i
        struct[0].Gy[165,33] = i_load_B6lv_a_r
        struct[0].Gy[165,38] = i_load_B6lv_a_i
        struct[0].Gy[165,39] = -i_load_B6lv_a_r
        struct[0].Gy[165,162] = v_B6lv_a_i - v_B6lv_n_i
        struct[0].Gy[165,163] = -v_B6lv_a_r + v_B6lv_n_r
        struct[0].Gy[166,34] = -i_load_B6lv_b_i
        struct[0].Gy[166,35] = i_load_B6lv_b_r
        struct[0].Gy[166,38] = i_load_B6lv_b_i
        struct[0].Gy[166,39] = -i_load_B6lv_b_r
        struct[0].Gy[166,164] = v_B6lv_b_i - v_B6lv_n_i
        struct[0].Gy[166,165] = -v_B6lv_b_r + v_B6lv_n_r
        struct[0].Gy[167,36] = -i_load_B6lv_c_i
        struct[0].Gy[167,37] = i_load_B6lv_c_r
        struct[0].Gy[167,38] = i_load_B6lv_c_i
        struct[0].Gy[167,39] = -i_load_B6lv_c_r
        struct[0].Gy[167,166] = v_B6lv_c_i - v_B6lv_n_i
        struct[0].Gy[167,167] = -v_B6lv_c_r + v_B6lv_n_r

    if mode > 12:




        struct[0].Hy[6,0] = 1.0*v_B2lv_a_r*(v_B2lv_a_i**2 + v_B2lv_a_r**2)**(-0.5)
        struct[0].Hy[6,1] = 1.0*v_B2lv_a_i*(v_B2lv_a_i**2 + v_B2lv_a_r**2)**(-0.5)
        struct[0].Hy[7,2] = 1.0*v_B2lv_b_r*(v_B2lv_b_i**2 + v_B2lv_b_r**2)**(-0.5)
        struct[0].Hy[7,3] = 1.0*v_B2lv_b_i*(v_B2lv_b_i**2 + v_B2lv_b_r**2)**(-0.5)
        struct[0].Hy[8,4] = 1.0*v_B2lv_c_r*(v_B2lv_c_i**2 + v_B2lv_c_r**2)**(-0.5)
        struct[0].Hy[8,5] = 1.0*v_B2lv_c_i*(v_B2lv_c_i**2 + v_B2lv_c_r**2)**(-0.5)
        struct[0].Hy[9,6] = 1.0*v_B2lv_n_r*(v_B2lv_n_i**2 + v_B2lv_n_r**2)**(-0.5)
        struct[0].Hy[9,7] = 1.0*v_B2lv_n_i*(v_B2lv_n_i**2 + v_B2lv_n_r**2)**(-0.5)
        struct[0].Hy[10,8] = 1.0*v_B3lv_a_r*(v_B3lv_a_i**2 + v_B3lv_a_r**2)**(-0.5)
        struct[0].Hy[10,9] = 1.0*v_B3lv_a_i*(v_B3lv_a_i**2 + v_B3lv_a_r**2)**(-0.5)
        struct[0].Hy[11,10] = 1.0*v_B3lv_b_r*(v_B3lv_b_i**2 + v_B3lv_b_r**2)**(-0.5)
        struct[0].Hy[11,11] = 1.0*v_B3lv_b_i*(v_B3lv_b_i**2 + v_B3lv_b_r**2)**(-0.5)
        struct[0].Hy[12,12] = 1.0*v_B3lv_c_r*(v_B3lv_c_i**2 + v_B3lv_c_r**2)**(-0.5)
        struct[0].Hy[12,13] = 1.0*v_B3lv_c_i*(v_B3lv_c_i**2 + v_B3lv_c_r**2)**(-0.5)
        struct[0].Hy[13,14] = 1.0*v_B3lv_n_r*(v_B3lv_n_i**2 + v_B3lv_n_r**2)**(-0.5)
        struct[0].Hy[13,15] = 1.0*v_B3lv_n_i*(v_B3lv_n_i**2 + v_B3lv_n_r**2)**(-0.5)
        struct[0].Hy[14,16] = 1.0*v_B4lv_a_r*(v_B4lv_a_i**2 + v_B4lv_a_r**2)**(-0.5)
        struct[0].Hy[14,17] = 1.0*v_B4lv_a_i*(v_B4lv_a_i**2 + v_B4lv_a_r**2)**(-0.5)
        struct[0].Hy[15,18] = 1.0*v_B4lv_b_r*(v_B4lv_b_i**2 + v_B4lv_b_r**2)**(-0.5)
        struct[0].Hy[15,19] = 1.0*v_B4lv_b_i*(v_B4lv_b_i**2 + v_B4lv_b_r**2)**(-0.5)
        struct[0].Hy[16,20] = 1.0*v_B4lv_c_r*(v_B4lv_c_i**2 + v_B4lv_c_r**2)**(-0.5)
        struct[0].Hy[16,21] = 1.0*v_B4lv_c_i*(v_B4lv_c_i**2 + v_B4lv_c_r**2)**(-0.5)
        struct[0].Hy[17,22] = 1.0*v_B4lv_n_r*(v_B4lv_n_i**2 + v_B4lv_n_r**2)**(-0.5)
        struct[0].Hy[17,23] = 1.0*v_B4lv_n_i*(v_B4lv_n_i**2 + v_B4lv_n_r**2)**(-0.5)
        struct[0].Hy[18,24] = 1.0*v_B5lv_a_r*(v_B5lv_a_i**2 + v_B5lv_a_r**2)**(-0.5)
        struct[0].Hy[18,25] = 1.0*v_B5lv_a_i*(v_B5lv_a_i**2 + v_B5lv_a_r**2)**(-0.5)
        struct[0].Hy[19,26] = 1.0*v_B5lv_b_r*(v_B5lv_b_i**2 + v_B5lv_b_r**2)**(-0.5)
        struct[0].Hy[19,27] = 1.0*v_B5lv_b_i*(v_B5lv_b_i**2 + v_B5lv_b_r**2)**(-0.5)
        struct[0].Hy[20,28] = 1.0*v_B5lv_c_r*(v_B5lv_c_i**2 + v_B5lv_c_r**2)**(-0.5)
        struct[0].Hy[20,29] = 1.0*v_B5lv_c_i*(v_B5lv_c_i**2 + v_B5lv_c_r**2)**(-0.5)
        struct[0].Hy[21,30] = 1.0*v_B5lv_n_r*(v_B5lv_n_i**2 + v_B5lv_n_r**2)**(-0.5)
        struct[0].Hy[21,31] = 1.0*v_B5lv_n_i*(v_B5lv_n_i**2 + v_B5lv_n_r**2)**(-0.5)
        struct[0].Hy[22,32] = 1.0*v_B6lv_a_r*(v_B6lv_a_i**2 + v_B6lv_a_r**2)**(-0.5)
        struct[0].Hy[22,33] = 1.0*v_B6lv_a_i*(v_B6lv_a_i**2 + v_B6lv_a_r**2)**(-0.5)
        struct[0].Hy[23,34] = 1.0*v_B6lv_b_r*(v_B6lv_b_i**2 + v_B6lv_b_r**2)**(-0.5)
        struct[0].Hy[23,35] = 1.0*v_B6lv_b_i*(v_B6lv_b_i**2 + v_B6lv_b_r**2)**(-0.5)
        struct[0].Hy[24,36] = 1.0*v_B6lv_c_r*(v_B6lv_c_i**2 + v_B6lv_c_r**2)**(-0.5)
        struct[0].Hy[24,37] = 1.0*v_B6lv_c_i*(v_B6lv_c_i**2 + v_B6lv_c_r**2)**(-0.5)
        struct[0].Hy[25,38] = 1.0*v_B6lv_n_r*(v_B6lv_n_i**2 + v_B6lv_n_r**2)**(-0.5)
        struct[0].Hy[25,39] = 1.0*v_B6lv_n_i*(v_B6lv_n_i**2 + v_B6lv_n_r**2)**(-0.5)
        struct[0].Hy[26,40] = 1.0*v_B2_a_r*(v_B2_a_i**2 + v_B2_a_r**2)**(-0.5)
        struct[0].Hy[26,41] = 1.0*v_B2_a_i*(v_B2_a_i**2 + v_B2_a_r**2)**(-0.5)
        struct[0].Hy[27,42] = 1.0*v_B2_b_r*(v_B2_b_i**2 + v_B2_b_r**2)**(-0.5)
        struct[0].Hy[27,43] = 1.0*v_B2_b_i*(v_B2_b_i**2 + v_B2_b_r**2)**(-0.5)
        struct[0].Hy[28,44] = 1.0*v_B2_c_r*(v_B2_c_i**2 + v_B2_c_r**2)**(-0.5)
        struct[0].Hy[28,45] = 1.0*v_B2_c_i*(v_B2_c_i**2 + v_B2_c_r**2)**(-0.5)
        struct[0].Hy[29,46] = 1.0*v_B3_a_r*(v_B3_a_i**2 + v_B3_a_r**2)**(-0.5)
        struct[0].Hy[29,47] = 1.0*v_B3_a_i*(v_B3_a_i**2 + v_B3_a_r**2)**(-0.5)
        struct[0].Hy[30,48] = 1.0*v_B3_b_r*(v_B3_b_i**2 + v_B3_b_r**2)**(-0.5)
        struct[0].Hy[30,49] = 1.0*v_B3_b_i*(v_B3_b_i**2 + v_B3_b_r**2)**(-0.5)
        struct[0].Hy[31,50] = 1.0*v_B3_c_r*(v_B3_c_i**2 + v_B3_c_r**2)**(-0.5)
        struct[0].Hy[31,51] = 1.0*v_B3_c_i*(v_B3_c_i**2 + v_B3_c_r**2)**(-0.5)
        struct[0].Hy[32,52] = 1.0*v_B4_a_r*(v_B4_a_i**2 + v_B4_a_r**2)**(-0.5)
        struct[0].Hy[32,53] = 1.0*v_B4_a_i*(v_B4_a_i**2 + v_B4_a_r**2)**(-0.5)
        struct[0].Hy[33,54] = 1.0*v_B4_b_r*(v_B4_b_i**2 + v_B4_b_r**2)**(-0.5)
        struct[0].Hy[33,55] = 1.0*v_B4_b_i*(v_B4_b_i**2 + v_B4_b_r**2)**(-0.5)
        struct[0].Hy[34,56] = 1.0*v_B4_c_r*(v_B4_c_i**2 + v_B4_c_r**2)**(-0.5)
        struct[0].Hy[34,57] = 1.0*v_B4_c_i*(v_B4_c_i**2 + v_B4_c_r**2)**(-0.5)
        struct[0].Hy[35,58] = 1.0*v_B5_a_r*(v_B5_a_i**2 + v_B5_a_r**2)**(-0.5)
        struct[0].Hy[35,59] = 1.0*v_B5_a_i*(v_B5_a_i**2 + v_B5_a_r**2)**(-0.5)
        struct[0].Hy[36,60] = 1.0*v_B5_b_r*(v_B5_b_i**2 + v_B5_b_r**2)**(-0.5)
        struct[0].Hy[36,61] = 1.0*v_B5_b_i*(v_B5_b_i**2 + v_B5_b_r**2)**(-0.5)
        struct[0].Hy[37,62] = 1.0*v_B5_c_r*(v_B5_c_i**2 + v_B5_c_r**2)**(-0.5)
        struct[0].Hy[37,63] = 1.0*v_B5_c_i*(v_B5_c_i**2 + v_B5_c_r**2)**(-0.5)
        struct[0].Hy[38,64] = 1.0*v_B6_a_r*(v_B6_a_i**2 + v_B6_a_r**2)**(-0.5)
        struct[0].Hy[38,65] = 1.0*v_B6_a_i*(v_B6_a_i**2 + v_B6_a_r**2)**(-0.5)
        struct[0].Hy[39,66] = 1.0*v_B6_b_r*(v_B6_b_i**2 + v_B6_b_r**2)**(-0.5)
        struct[0].Hy[39,67] = 1.0*v_B6_b_i*(v_B6_b_i**2 + v_B6_b_r**2)**(-0.5)
        struct[0].Hy[40,68] = 1.0*v_B6_c_r*(v_B6_c_i**2 + v_B6_c_r**2)**(-0.5)
        struct[0].Hy[40,69] = 1.0*v_B6_c_i*(v_B6_c_i**2 + v_B6_c_r**2)**(-0.5)

        struct[0].Hu[0,0] = 1.0*v_B1_a_r*(v_B1_a_i**2 + v_B1_a_r**2)**(-0.5)
        struct[0].Hu[0,1] = 1.0*v_B1_a_i*(v_B1_a_i**2 + v_B1_a_r**2)**(-0.5)
        struct[0].Hu[1,2] = 1.0*v_B1_b_r*(v_B1_b_i**2 + v_B1_b_r**2)**(-0.5)
        struct[0].Hu[1,3] = 1.0*v_B1_b_i*(v_B1_b_i**2 + v_B1_b_r**2)**(-0.5)
        struct[0].Hu[2,4] = 1.0*v_B1_c_r*(v_B1_c_i**2 + v_B1_c_r**2)**(-0.5)
        struct[0].Hu[2,5] = 1.0*v_B1_c_i*(v_B1_c_i**2 + v_B1_c_r**2)**(-0.5)
        struct[0].Hu[3,6] = 1.0*v_B7_a_r*(v_B7_a_i**2 + v_B7_a_r**2)**(-0.5)
        struct[0].Hu[3,7] = 1.0*v_B7_a_i*(v_B7_a_i**2 + v_B7_a_r**2)**(-0.5)
        struct[0].Hu[4,8] = 1.0*v_B7_b_r*(v_B7_b_i**2 + v_B7_b_r**2)**(-0.5)
        struct[0].Hu[4,9] = 1.0*v_B7_b_i*(v_B7_b_i**2 + v_B7_b_r**2)**(-0.5)
        struct[0].Hu[5,10] = 1.0*v_B7_c_r*(v_B7_c_i**2 + v_B7_c_r**2)**(-0.5)
        struct[0].Hu[5,11] = 1.0*v_B7_c_i*(v_B7_c_i**2 + v_B7_c_r**2)**(-0.5)



def ini_nn(struct,mode):

    # Parameters:
    
    # Inputs:
    v_B1_a_r = struct[0].v_B1_a_r
    v_B1_a_i = struct[0].v_B1_a_i
    v_B1_b_r = struct[0].v_B1_b_r
    v_B1_b_i = struct[0].v_B1_b_i
    v_B1_c_r = struct[0].v_B1_c_r
    v_B1_c_i = struct[0].v_B1_c_i
    v_B7_a_r = struct[0].v_B7_a_r
    v_B7_a_i = struct[0].v_B7_a_i
    v_B7_b_r = struct[0].v_B7_b_r
    v_B7_b_i = struct[0].v_B7_b_i
    v_B7_c_r = struct[0].v_B7_c_r
    v_B7_c_i = struct[0].v_B7_c_i
    i_B2lv_n_r = struct[0].i_B2lv_n_r
    i_B2lv_n_i = struct[0].i_B2lv_n_i
    i_B3lv_n_r = struct[0].i_B3lv_n_r
    i_B3lv_n_i = struct[0].i_B3lv_n_i
    i_B4lv_n_r = struct[0].i_B4lv_n_r
    i_B4lv_n_i = struct[0].i_B4lv_n_i
    i_B5lv_n_r = struct[0].i_B5lv_n_r
    i_B5lv_n_i = struct[0].i_B5lv_n_i
    i_B6lv_n_r = struct[0].i_B6lv_n_r
    i_B6lv_n_i = struct[0].i_B6lv_n_i
    i_B2_a_r = struct[0].i_B2_a_r
    i_B2_a_i = struct[0].i_B2_a_i
    i_B2_b_r = struct[0].i_B2_b_r
    i_B2_b_i = struct[0].i_B2_b_i
    i_B2_c_r = struct[0].i_B2_c_r
    i_B2_c_i = struct[0].i_B2_c_i
    i_B3_a_r = struct[0].i_B3_a_r
    i_B3_a_i = struct[0].i_B3_a_i
    i_B3_b_r = struct[0].i_B3_b_r
    i_B3_b_i = struct[0].i_B3_b_i
    i_B3_c_r = struct[0].i_B3_c_r
    i_B3_c_i = struct[0].i_B3_c_i
    i_B4_a_r = struct[0].i_B4_a_r
    i_B4_a_i = struct[0].i_B4_a_i
    i_B4_b_r = struct[0].i_B4_b_r
    i_B4_b_i = struct[0].i_B4_b_i
    i_B4_c_r = struct[0].i_B4_c_r
    i_B4_c_i = struct[0].i_B4_c_i
    i_B5_a_r = struct[0].i_B5_a_r
    i_B5_a_i = struct[0].i_B5_a_i
    i_B5_b_r = struct[0].i_B5_b_r
    i_B5_b_i = struct[0].i_B5_b_i
    i_B5_c_r = struct[0].i_B5_c_r
    i_B5_c_i = struct[0].i_B5_c_i
    i_B6_a_r = struct[0].i_B6_a_r
    i_B6_a_i = struct[0].i_B6_a_i
    i_B6_b_r = struct[0].i_B6_b_r
    i_B6_b_i = struct[0].i_B6_b_i
    i_B6_c_r = struct[0].i_B6_c_r
    i_B6_c_i = struct[0].i_B6_c_i
    p_B2lv_a = struct[0].p_B2lv_a
    q_B2lv_a = struct[0].q_B2lv_a
    p_B2lv_b = struct[0].p_B2lv_b
    q_B2lv_b = struct[0].q_B2lv_b
    p_B2lv_c = struct[0].p_B2lv_c
    q_B2lv_c = struct[0].q_B2lv_c
    p_B3lv_a = struct[0].p_B3lv_a
    q_B3lv_a = struct[0].q_B3lv_a
    p_B3lv_b = struct[0].p_B3lv_b
    q_B3lv_b = struct[0].q_B3lv_b
    p_B3lv_c = struct[0].p_B3lv_c
    q_B3lv_c = struct[0].q_B3lv_c
    p_B4lv_a = struct[0].p_B4lv_a
    q_B4lv_a = struct[0].q_B4lv_a
    p_B4lv_b = struct[0].p_B4lv_b
    q_B4lv_b = struct[0].q_B4lv_b
    p_B4lv_c = struct[0].p_B4lv_c
    q_B4lv_c = struct[0].q_B4lv_c
    p_B5lv_a = struct[0].p_B5lv_a
    q_B5lv_a = struct[0].q_B5lv_a
    p_B5lv_b = struct[0].p_B5lv_b
    q_B5lv_b = struct[0].q_B5lv_b
    p_B5lv_c = struct[0].p_B5lv_c
    q_B5lv_c = struct[0].q_B5lv_c
    p_B6lv_a = struct[0].p_B6lv_a
    q_B6lv_a = struct[0].q_B6lv_a
    p_B6lv_b = struct[0].p_B6lv_b
    q_B6lv_b = struct[0].q_B6lv_b
    p_B6lv_c = struct[0].p_B6lv_c
    q_B6lv_c = struct[0].q_B6lv_c
    u_dummy = struct[0].u_dummy
    
    # Dynamical states:
    x_dummy = struct[0].x[0,0]
    
    # Algebraic states:
    v_B2lv_a_r = struct[0].y_ini[0,0]
    v_B2lv_a_i = struct[0].y_ini[1,0]
    v_B2lv_b_r = struct[0].y_ini[2,0]
    v_B2lv_b_i = struct[0].y_ini[3,0]
    v_B2lv_c_r = struct[0].y_ini[4,0]
    v_B2lv_c_i = struct[0].y_ini[5,0]
    v_B2lv_n_r = struct[0].y_ini[6,0]
    v_B2lv_n_i = struct[0].y_ini[7,0]
    v_B3lv_a_r = struct[0].y_ini[8,0]
    v_B3lv_a_i = struct[0].y_ini[9,0]
    v_B3lv_b_r = struct[0].y_ini[10,0]
    v_B3lv_b_i = struct[0].y_ini[11,0]
    v_B3lv_c_r = struct[0].y_ini[12,0]
    v_B3lv_c_i = struct[0].y_ini[13,0]
    v_B3lv_n_r = struct[0].y_ini[14,0]
    v_B3lv_n_i = struct[0].y_ini[15,0]
    v_B4lv_a_r = struct[0].y_ini[16,0]
    v_B4lv_a_i = struct[0].y_ini[17,0]
    v_B4lv_b_r = struct[0].y_ini[18,0]
    v_B4lv_b_i = struct[0].y_ini[19,0]
    v_B4lv_c_r = struct[0].y_ini[20,0]
    v_B4lv_c_i = struct[0].y_ini[21,0]
    v_B4lv_n_r = struct[0].y_ini[22,0]
    v_B4lv_n_i = struct[0].y_ini[23,0]
    v_B5lv_a_r = struct[0].y_ini[24,0]
    v_B5lv_a_i = struct[0].y_ini[25,0]
    v_B5lv_b_r = struct[0].y_ini[26,0]
    v_B5lv_b_i = struct[0].y_ini[27,0]
    v_B5lv_c_r = struct[0].y_ini[28,0]
    v_B5lv_c_i = struct[0].y_ini[29,0]
    v_B5lv_n_r = struct[0].y_ini[30,0]
    v_B5lv_n_i = struct[0].y_ini[31,0]
    v_B6lv_a_r = struct[0].y_ini[32,0]
    v_B6lv_a_i = struct[0].y_ini[33,0]
    v_B6lv_b_r = struct[0].y_ini[34,0]
    v_B6lv_b_i = struct[0].y_ini[35,0]
    v_B6lv_c_r = struct[0].y_ini[36,0]
    v_B6lv_c_i = struct[0].y_ini[37,0]
    v_B6lv_n_r = struct[0].y_ini[38,0]
    v_B6lv_n_i = struct[0].y_ini[39,0]
    v_B2_a_r = struct[0].y_ini[40,0]
    v_B2_a_i = struct[0].y_ini[41,0]
    v_B2_b_r = struct[0].y_ini[42,0]
    v_B2_b_i = struct[0].y_ini[43,0]
    v_B2_c_r = struct[0].y_ini[44,0]
    v_B2_c_i = struct[0].y_ini[45,0]
    v_B3_a_r = struct[0].y_ini[46,0]
    v_B3_a_i = struct[0].y_ini[47,0]
    v_B3_b_r = struct[0].y_ini[48,0]
    v_B3_b_i = struct[0].y_ini[49,0]
    v_B3_c_r = struct[0].y_ini[50,0]
    v_B3_c_i = struct[0].y_ini[51,0]
    v_B4_a_r = struct[0].y_ini[52,0]
    v_B4_a_i = struct[0].y_ini[53,0]
    v_B4_b_r = struct[0].y_ini[54,0]
    v_B4_b_i = struct[0].y_ini[55,0]
    v_B4_c_r = struct[0].y_ini[56,0]
    v_B4_c_i = struct[0].y_ini[57,0]
    v_B5_a_r = struct[0].y_ini[58,0]
    v_B5_a_i = struct[0].y_ini[59,0]
    v_B5_b_r = struct[0].y_ini[60,0]
    v_B5_b_i = struct[0].y_ini[61,0]
    v_B5_c_r = struct[0].y_ini[62,0]
    v_B5_c_i = struct[0].y_ini[63,0]
    v_B6_a_r = struct[0].y_ini[64,0]
    v_B6_a_i = struct[0].y_ini[65,0]
    v_B6_b_r = struct[0].y_ini[66,0]
    v_B6_b_i = struct[0].y_ini[67,0]
    v_B6_c_r = struct[0].y_ini[68,0]
    v_B6_c_i = struct[0].y_ini[69,0]
    i_t_B2_B2lv_a_r = struct[0].y_ini[70,0]
    i_t_B2_B2lv_a_i = struct[0].y_ini[71,0]
    i_t_B2_B2lv_b_r = struct[0].y_ini[72,0]
    i_t_B2_B2lv_b_i = struct[0].y_ini[73,0]
    i_t_B2_B2lv_c_r = struct[0].y_ini[74,0]
    i_t_B2_B2lv_c_i = struct[0].y_ini[75,0]
    i_t_B3_B3lv_a_r = struct[0].y_ini[76,0]
    i_t_B3_B3lv_a_i = struct[0].y_ini[77,0]
    i_t_B3_B3lv_b_r = struct[0].y_ini[78,0]
    i_t_B3_B3lv_b_i = struct[0].y_ini[79,0]
    i_t_B3_B3lv_c_r = struct[0].y_ini[80,0]
    i_t_B3_B3lv_c_i = struct[0].y_ini[81,0]
    i_t_B4_B4lv_a_r = struct[0].y_ini[82,0]
    i_t_B4_B4lv_a_i = struct[0].y_ini[83,0]
    i_t_B4_B4lv_b_r = struct[0].y_ini[84,0]
    i_t_B4_B4lv_b_i = struct[0].y_ini[85,0]
    i_t_B4_B4lv_c_r = struct[0].y_ini[86,0]
    i_t_B4_B4lv_c_i = struct[0].y_ini[87,0]
    i_t_B5_B5lv_a_r = struct[0].y_ini[88,0]
    i_t_B5_B5lv_a_i = struct[0].y_ini[89,0]
    i_t_B5_B5lv_b_r = struct[0].y_ini[90,0]
    i_t_B5_B5lv_b_i = struct[0].y_ini[91,0]
    i_t_B5_B5lv_c_r = struct[0].y_ini[92,0]
    i_t_B5_B5lv_c_i = struct[0].y_ini[93,0]
    i_t_B6_B6lv_a_r = struct[0].y_ini[94,0]
    i_t_B6_B6lv_a_i = struct[0].y_ini[95,0]
    i_t_B6_B6lv_b_r = struct[0].y_ini[96,0]
    i_t_B6_B6lv_b_i = struct[0].y_ini[97,0]
    i_t_B6_B6lv_c_r = struct[0].y_ini[98,0]
    i_t_B6_B6lv_c_i = struct[0].y_ini[99,0]
    i_l_B1_B2_a_r = struct[0].y_ini[100,0]
    i_l_B1_B2_a_i = struct[0].y_ini[101,0]
    i_l_B1_B2_b_r = struct[0].y_ini[102,0]
    i_l_B1_B2_b_i = struct[0].y_ini[103,0]
    i_l_B1_B2_c_r = struct[0].y_ini[104,0]
    i_l_B1_B2_c_i = struct[0].y_ini[105,0]
    i_l_B2_B3_a_r = struct[0].y_ini[106,0]
    i_l_B2_B3_a_i = struct[0].y_ini[107,0]
    i_l_B2_B3_b_r = struct[0].y_ini[108,0]
    i_l_B2_B3_b_i = struct[0].y_ini[109,0]
    i_l_B2_B3_c_r = struct[0].y_ini[110,0]
    i_l_B2_B3_c_i = struct[0].y_ini[111,0]
    i_l_B3_B4_a_r = struct[0].y_ini[112,0]
    i_l_B3_B4_a_i = struct[0].y_ini[113,0]
    i_l_B3_B4_b_r = struct[0].y_ini[114,0]
    i_l_B3_B4_b_i = struct[0].y_ini[115,0]
    i_l_B3_B4_c_r = struct[0].y_ini[116,0]
    i_l_B3_B4_c_i = struct[0].y_ini[117,0]
    i_l_B5_B6_a_r = struct[0].y_ini[118,0]
    i_l_B5_B6_a_i = struct[0].y_ini[119,0]
    i_l_B5_B6_b_r = struct[0].y_ini[120,0]
    i_l_B5_B6_b_i = struct[0].y_ini[121,0]
    i_l_B5_B6_c_r = struct[0].y_ini[122,0]
    i_l_B5_B6_c_i = struct[0].y_ini[123,0]
    i_l_B6_B7_a_r = struct[0].y_ini[124,0]
    i_l_B6_B7_a_i = struct[0].y_ini[125,0]
    i_l_B6_B7_b_r = struct[0].y_ini[126,0]
    i_l_B6_B7_b_i = struct[0].y_ini[127,0]
    i_l_B6_B7_c_r = struct[0].y_ini[128,0]
    i_l_B6_B7_c_i = struct[0].y_ini[129,0]
    i_load_B2lv_a_r = struct[0].y_ini[130,0]
    i_load_B2lv_a_i = struct[0].y_ini[131,0]
    i_load_B2lv_b_r = struct[0].y_ini[132,0]
    i_load_B2lv_b_i = struct[0].y_ini[133,0]
    i_load_B2lv_c_r = struct[0].y_ini[134,0]
    i_load_B2lv_c_i = struct[0].y_ini[135,0]
    i_load_B2lv_n_r = struct[0].y_ini[136,0]
    i_load_B2lv_n_i = struct[0].y_ini[137,0]
    i_load_B3lv_a_r = struct[0].y_ini[138,0]
    i_load_B3lv_a_i = struct[0].y_ini[139,0]
    i_load_B3lv_b_r = struct[0].y_ini[140,0]
    i_load_B3lv_b_i = struct[0].y_ini[141,0]
    i_load_B3lv_c_r = struct[0].y_ini[142,0]
    i_load_B3lv_c_i = struct[0].y_ini[143,0]
    i_load_B3lv_n_r = struct[0].y_ini[144,0]
    i_load_B3lv_n_i = struct[0].y_ini[145,0]
    i_load_B4lv_a_r = struct[0].y_ini[146,0]
    i_load_B4lv_a_i = struct[0].y_ini[147,0]
    i_load_B4lv_b_r = struct[0].y_ini[148,0]
    i_load_B4lv_b_i = struct[0].y_ini[149,0]
    i_load_B4lv_c_r = struct[0].y_ini[150,0]
    i_load_B4lv_c_i = struct[0].y_ini[151,0]
    i_load_B4lv_n_r = struct[0].y_ini[152,0]
    i_load_B4lv_n_i = struct[0].y_ini[153,0]
    i_load_B5lv_a_r = struct[0].y_ini[154,0]
    i_load_B5lv_a_i = struct[0].y_ini[155,0]
    i_load_B5lv_b_r = struct[0].y_ini[156,0]
    i_load_B5lv_b_i = struct[0].y_ini[157,0]
    i_load_B5lv_c_r = struct[0].y_ini[158,0]
    i_load_B5lv_c_i = struct[0].y_ini[159,0]
    i_load_B5lv_n_r = struct[0].y_ini[160,0]
    i_load_B5lv_n_i = struct[0].y_ini[161,0]
    i_load_B6lv_a_r = struct[0].y_ini[162,0]
    i_load_B6lv_a_i = struct[0].y_ini[163,0]
    i_load_B6lv_b_r = struct[0].y_ini[164,0]
    i_load_B6lv_b_i = struct[0].y_ini[165,0]
    i_load_B6lv_c_r = struct[0].y_ini[166,0]
    i_load_B6lv_c_i = struct[0].y_ini[167,0]
    i_load_B6lv_n_r = struct[0].y_ini[168,0]
    i_load_B6lv_n_i = struct[0].y_ini[169,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = u_dummy - x_dummy
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = i_load_B2lv_a_r + 0.996212229189942*v_B2_a_i + 0.249053057297486*v_B2_a_r - 0.996212229189942*v_B2_c_i - 0.249053057297486*v_B2_c_r - 23.0065359477124*v_B2lv_a_i - 5.7516339869281*v_B2lv_a_r + 23.0065359477124*v_B2lv_n_i + 5.7516339869281*v_B2lv_n_r
        struct[0].g[1,0] = i_load_B2lv_a_i + 0.249053057297486*v_B2_a_i - 0.996212229189942*v_B2_a_r - 0.249053057297486*v_B2_c_i + 0.996212229189942*v_B2_c_r - 5.7516339869281*v_B2lv_a_i + 23.0065359477124*v_B2lv_a_r + 5.7516339869281*v_B2lv_n_i - 23.0065359477124*v_B2lv_n_r
        struct[0].g[2,0] = i_load_B2lv_b_r - 0.996212229189942*v_B2_a_i - 0.249053057297486*v_B2_a_r + 0.996212229189942*v_B2_b_i + 0.249053057297486*v_B2_b_r - 23.0065359477124*v_B2lv_b_i - 5.7516339869281*v_B2lv_b_r + 23.0065359477124*v_B2lv_n_i + 5.7516339869281*v_B2lv_n_r
        struct[0].g[3,0] = i_load_B2lv_b_i - 0.249053057297486*v_B2_a_i + 0.996212229189942*v_B2_a_r + 0.249053057297486*v_B2_b_i - 0.996212229189942*v_B2_b_r - 5.7516339869281*v_B2lv_b_i + 23.0065359477124*v_B2lv_b_r + 5.7516339869281*v_B2lv_n_i - 23.0065359477124*v_B2lv_n_r
        struct[0].g[4,0] = i_load_B2lv_c_r - 0.996212229189942*v_B2_b_i - 0.249053057297486*v_B2_b_r + 0.996212229189942*v_B2_c_i + 0.249053057297486*v_B2_c_r - 23.0065359477124*v_B2lv_c_i - 5.7516339869281*v_B2lv_c_r + 23.0065359477124*v_B2lv_n_i + 5.7516339869281*v_B2lv_n_r
        struct[0].g[5,0] = i_load_B2lv_c_i - 0.249053057297486*v_B2_b_i + 0.996212229189942*v_B2_b_r + 0.249053057297486*v_B2_c_i - 0.996212229189942*v_B2_c_r - 5.7516339869281*v_B2lv_c_i + 23.0065359477124*v_B2lv_c_r + 5.7516339869281*v_B2lv_n_i - 23.0065359477124*v_B2lv_n_r
        struct[0].g[6,0] = 23.0065359477124*v_B2lv_a_i + 5.7516339869281*v_B2lv_a_r + 23.0065359477124*v_B2lv_b_i + 5.7516339869281*v_B2lv_b_r + 23.0065359477124*v_B2lv_c_i + 5.7516339869281*v_B2lv_c_r - 69.0196078431372*v_B2lv_n_i - 1017.25490196078*v_B2lv_n_r
        struct[0].g[7,0] = 5.7516339869281*v_B2lv_a_i - 23.0065359477124*v_B2lv_a_r + 5.7516339869281*v_B2lv_b_i - 23.0065359477124*v_B2lv_b_r + 5.7516339869281*v_B2lv_c_i - 23.0065359477124*v_B2lv_c_r - 1017.25490196078*v_B2lv_n_i + 69.0196078431372*v_B2lv_n_r
        struct[0].g[8,0] = i_load_B3lv_a_r + 0.996212229189942*v_B3_a_i + 0.249053057297486*v_B3_a_r - 0.996212229189942*v_B3_c_i - 0.249053057297486*v_B3_c_r - 23.0065359477124*v_B3lv_a_i - 5.7516339869281*v_B3lv_a_r + 23.0065359477124*v_B3lv_n_i + 5.7516339869281*v_B3lv_n_r
        struct[0].g[9,0] = i_load_B3lv_a_i + 0.249053057297486*v_B3_a_i - 0.996212229189942*v_B3_a_r - 0.249053057297486*v_B3_c_i + 0.996212229189942*v_B3_c_r - 5.7516339869281*v_B3lv_a_i + 23.0065359477124*v_B3lv_a_r + 5.7516339869281*v_B3lv_n_i - 23.0065359477124*v_B3lv_n_r
        struct[0].g[10,0] = i_load_B3lv_b_r - 0.996212229189942*v_B3_a_i - 0.249053057297486*v_B3_a_r + 0.996212229189942*v_B3_b_i + 0.249053057297486*v_B3_b_r - 23.0065359477124*v_B3lv_b_i - 5.7516339869281*v_B3lv_b_r + 23.0065359477124*v_B3lv_n_i + 5.7516339869281*v_B3lv_n_r
        struct[0].g[11,0] = i_load_B3lv_b_i - 0.249053057297486*v_B3_a_i + 0.996212229189942*v_B3_a_r + 0.249053057297486*v_B3_b_i - 0.996212229189942*v_B3_b_r - 5.7516339869281*v_B3lv_b_i + 23.0065359477124*v_B3lv_b_r + 5.7516339869281*v_B3lv_n_i - 23.0065359477124*v_B3lv_n_r
        struct[0].g[12,0] = i_load_B3lv_c_r - 0.996212229189942*v_B3_b_i - 0.249053057297486*v_B3_b_r + 0.996212229189942*v_B3_c_i + 0.249053057297486*v_B3_c_r - 23.0065359477124*v_B3lv_c_i - 5.7516339869281*v_B3lv_c_r + 23.0065359477124*v_B3lv_n_i + 5.7516339869281*v_B3lv_n_r
        struct[0].g[13,0] = i_load_B3lv_c_i - 0.249053057297486*v_B3_b_i + 0.996212229189942*v_B3_b_r + 0.249053057297486*v_B3_c_i - 0.996212229189942*v_B3_c_r - 5.7516339869281*v_B3lv_c_i + 23.0065359477124*v_B3lv_c_r + 5.7516339869281*v_B3lv_n_i - 23.0065359477124*v_B3lv_n_r
        struct[0].g[14,0] = 23.0065359477124*v_B3lv_a_i + 5.7516339869281*v_B3lv_a_r + 23.0065359477124*v_B3lv_b_i + 5.7516339869281*v_B3lv_b_r + 23.0065359477124*v_B3lv_c_i + 5.7516339869281*v_B3lv_c_r - 69.0196078431372*v_B3lv_n_i - 1017.25490196078*v_B3lv_n_r
        struct[0].g[15,0] = 5.7516339869281*v_B3lv_a_i - 23.0065359477124*v_B3lv_a_r + 5.7516339869281*v_B3lv_b_i - 23.0065359477124*v_B3lv_b_r + 5.7516339869281*v_B3lv_c_i - 23.0065359477124*v_B3lv_c_r - 1017.25490196078*v_B3lv_n_i + 69.0196078431372*v_B3lv_n_r
        struct[0].g[16,0] = i_load_B4lv_a_r + 0.996212229189942*v_B4_a_i + 0.249053057297486*v_B4_a_r - 0.996212229189942*v_B4_c_i - 0.249053057297486*v_B4_c_r - 23.0065359477124*v_B4lv_a_i - 5.7516339869281*v_B4lv_a_r + 23.0065359477124*v_B4lv_n_i + 5.7516339869281*v_B4lv_n_r
        struct[0].g[17,0] = i_load_B4lv_a_i + 0.249053057297486*v_B4_a_i - 0.996212229189942*v_B4_a_r - 0.249053057297486*v_B4_c_i + 0.996212229189942*v_B4_c_r - 5.7516339869281*v_B4lv_a_i + 23.0065359477124*v_B4lv_a_r + 5.7516339869281*v_B4lv_n_i - 23.0065359477124*v_B4lv_n_r
        struct[0].g[18,0] = i_load_B4lv_b_r - 0.996212229189942*v_B4_a_i - 0.249053057297486*v_B4_a_r + 0.996212229189942*v_B4_b_i + 0.249053057297486*v_B4_b_r - 23.0065359477124*v_B4lv_b_i - 5.7516339869281*v_B4lv_b_r + 23.0065359477124*v_B4lv_n_i + 5.7516339869281*v_B4lv_n_r
        struct[0].g[19,0] = i_load_B4lv_b_i - 0.249053057297486*v_B4_a_i + 0.996212229189942*v_B4_a_r + 0.249053057297486*v_B4_b_i - 0.996212229189942*v_B4_b_r - 5.7516339869281*v_B4lv_b_i + 23.0065359477124*v_B4lv_b_r + 5.7516339869281*v_B4lv_n_i - 23.0065359477124*v_B4lv_n_r
        struct[0].g[20,0] = i_load_B4lv_c_r - 0.996212229189942*v_B4_b_i - 0.249053057297486*v_B4_b_r + 0.996212229189942*v_B4_c_i + 0.249053057297486*v_B4_c_r - 23.0065359477124*v_B4lv_c_i - 5.7516339869281*v_B4lv_c_r + 23.0065359477124*v_B4lv_n_i + 5.7516339869281*v_B4lv_n_r
        struct[0].g[21,0] = i_load_B4lv_c_i - 0.249053057297486*v_B4_b_i + 0.996212229189942*v_B4_b_r + 0.249053057297486*v_B4_c_i - 0.996212229189942*v_B4_c_r - 5.7516339869281*v_B4lv_c_i + 23.0065359477124*v_B4lv_c_r + 5.7516339869281*v_B4lv_n_i - 23.0065359477124*v_B4lv_n_r
        struct[0].g[22,0] = 23.0065359477124*v_B4lv_a_i + 5.7516339869281*v_B4lv_a_r + 23.0065359477124*v_B4lv_b_i + 5.7516339869281*v_B4lv_b_r + 23.0065359477124*v_B4lv_c_i + 5.7516339869281*v_B4lv_c_r - 69.0196078431372*v_B4lv_n_i - 1017.25490196078*v_B4lv_n_r
        struct[0].g[23,0] = 5.7516339869281*v_B4lv_a_i - 23.0065359477124*v_B4lv_a_r + 5.7516339869281*v_B4lv_b_i - 23.0065359477124*v_B4lv_b_r + 5.7516339869281*v_B4lv_c_i - 23.0065359477124*v_B4lv_c_r - 1017.25490196078*v_B4lv_n_i + 69.0196078431372*v_B4lv_n_r
        struct[0].g[24,0] = i_load_B5lv_a_r + 0.996212229189942*v_B5_a_i + 0.249053057297486*v_B5_a_r - 0.996212229189942*v_B5_c_i - 0.249053057297486*v_B5_c_r - 23.0065359477124*v_B5lv_a_i - 5.7516339869281*v_B5lv_a_r + 23.0065359477124*v_B5lv_n_i + 5.7516339869281*v_B5lv_n_r
        struct[0].g[25,0] = i_load_B5lv_a_i + 0.249053057297486*v_B5_a_i - 0.996212229189942*v_B5_a_r - 0.249053057297486*v_B5_c_i + 0.996212229189942*v_B5_c_r - 5.7516339869281*v_B5lv_a_i + 23.0065359477124*v_B5lv_a_r + 5.7516339869281*v_B5lv_n_i - 23.0065359477124*v_B5lv_n_r
        struct[0].g[26,0] = i_load_B5lv_b_r - 0.996212229189942*v_B5_a_i - 0.249053057297486*v_B5_a_r + 0.996212229189942*v_B5_b_i + 0.249053057297486*v_B5_b_r - 23.0065359477124*v_B5lv_b_i - 5.7516339869281*v_B5lv_b_r + 23.0065359477124*v_B5lv_n_i + 5.7516339869281*v_B5lv_n_r
        struct[0].g[27,0] = i_load_B5lv_b_i - 0.249053057297486*v_B5_a_i + 0.996212229189942*v_B5_a_r + 0.249053057297486*v_B5_b_i - 0.996212229189942*v_B5_b_r - 5.7516339869281*v_B5lv_b_i + 23.0065359477124*v_B5lv_b_r + 5.7516339869281*v_B5lv_n_i - 23.0065359477124*v_B5lv_n_r
        struct[0].g[28,0] = i_load_B5lv_c_r - 0.996212229189942*v_B5_b_i - 0.249053057297486*v_B5_b_r + 0.996212229189942*v_B5_c_i + 0.249053057297486*v_B5_c_r - 23.0065359477124*v_B5lv_c_i - 5.7516339869281*v_B5lv_c_r + 23.0065359477124*v_B5lv_n_i + 5.7516339869281*v_B5lv_n_r
        struct[0].g[29,0] = i_load_B5lv_c_i - 0.249053057297486*v_B5_b_i + 0.996212229189942*v_B5_b_r + 0.249053057297486*v_B5_c_i - 0.996212229189942*v_B5_c_r - 5.7516339869281*v_B5lv_c_i + 23.0065359477124*v_B5lv_c_r + 5.7516339869281*v_B5lv_n_i - 23.0065359477124*v_B5lv_n_r
        struct[0].g[30,0] = 23.0065359477124*v_B5lv_a_i + 5.7516339869281*v_B5lv_a_r + 23.0065359477124*v_B5lv_b_i + 5.7516339869281*v_B5lv_b_r + 23.0065359477124*v_B5lv_c_i + 5.7516339869281*v_B5lv_c_r - 69.0196078431372*v_B5lv_n_i - 1017.25490196078*v_B5lv_n_r
        struct[0].g[31,0] = 5.7516339869281*v_B5lv_a_i - 23.0065359477124*v_B5lv_a_r + 5.7516339869281*v_B5lv_b_i - 23.0065359477124*v_B5lv_b_r + 5.7516339869281*v_B5lv_c_i - 23.0065359477124*v_B5lv_c_r - 1017.25490196078*v_B5lv_n_i + 69.0196078431372*v_B5lv_n_r
        struct[0].g[32,0] = i_load_B6lv_a_r + 0.996212229189942*v_B6_a_i + 0.249053057297486*v_B6_a_r - 0.996212229189942*v_B6_c_i - 0.249053057297486*v_B6_c_r - 23.0065359477124*v_B6lv_a_i - 5.7516339869281*v_B6lv_a_r + 23.0065359477124*v_B6lv_n_i + 5.7516339869281*v_B6lv_n_r
        struct[0].g[33,0] = i_load_B6lv_a_i + 0.249053057297486*v_B6_a_i - 0.996212229189942*v_B6_a_r - 0.249053057297486*v_B6_c_i + 0.996212229189942*v_B6_c_r - 5.7516339869281*v_B6lv_a_i + 23.0065359477124*v_B6lv_a_r + 5.7516339869281*v_B6lv_n_i - 23.0065359477124*v_B6lv_n_r
        struct[0].g[34,0] = i_load_B6lv_b_r - 0.996212229189942*v_B6_a_i - 0.249053057297486*v_B6_a_r + 0.996212229189942*v_B6_b_i + 0.249053057297486*v_B6_b_r - 23.0065359477124*v_B6lv_b_i - 5.7516339869281*v_B6lv_b_r + 23.0065359477124*v_B6lv_n_i + 5.7516339869281*v_B6lv_n_r
        struct[0].g[35,0] = i_load_B6lv_b_i - 0.249053057297486*v_B6_a_i + 0.996212229189942*v_B6_a_r + 0.249053057297486*v_B6_b_i - 0.996212229189942*v_B6_b_r - 5.7516339869281*v_B6lv_b_i + 23.0065359477124*v_B6lv_b_r + 5.7516339869281*v_B6lv_n_i - 23.0065359477124*v_B6lv_n_r
        struct[0].g[36,0] = i_load_B6lv_c_r - 0.996212229189942*v_B6_b_i - 0.249053057297486*v_B6_b_r + 0.996212229189942*v_B6_c_i + 0.249053057297486*v_B6_c_r - 23.0065359477124*v_B6lv_c_i - 5.7516339869281*v_B6lv_c_r + 23.0065359477124*v_B6lv_n_i + 5.7516339869281*v_B6lv_n_r
        struct[0].g[37,0] = i_load_B6lv_c_i - 0.249053057297486*v_B6_b_i + 0.996212229189942*v_B6_b_r + 0.249053057297486*v_B6_c_i - 0.996212229189942*v_B6_c_r - 5.7516339869281*v_B6lv_c_i + 23.0065359477124*v_B6lv_c_r + 5.7516339869281*v_B6lv_n_i - 23.0065359477124*v_B6lv_n_r
        struct[0].g[38,0] = 23.0065359477124*v_B6lv_a_i + 5.7516339869281*v_B6lv_a_r + 23.0065359477124*v_B6lv_b_i + 5.7516339869281*v_B6lv_b_r + 23.0065359477124*v_B6lv_c_i + 5.7516339869281*v_B6lv_c_r - 69.0196078431372*v_B6lv_n_i - 1017.25490196078*v_B6lv_n_r
        struct[0].g[39,0] = 5.7516339869281*v_B6lv_a_i - 23.0065359477124*v_B6lv_a_r + 5.7516339869281*v_B6lv_b_i - 23.0065359477124*v_B6lv_b_r + 5.7516339869281*v_B6lv_c_i - 23.0065359477124*v_B6lv_c_r - 1017.25490196078*v_B6lv_n_i + 69.0196078431372*v_B6lv_n_r
        struct[0].g[40,0] = 0.598820527961361*v_B1_a_i + 1.10755301189314*v_B1_a_r - 0.171091579417532*v_B1_b_i - 0.316443717683753*v_B1_b_r - 0.171091579417532*v_B1_c_i - 0.316443717683753*v_B1_c_r - 1.28353302446119*v_B2_a_i - 2.23667465123725*v_B2_a_r + 0.385473430243205*v_B2_b_i + 0.643671749092996*v_B2_b_r + 0.385473430243205*v_B2_c_i + 0.643671749092997*v_B2_c_r + 0.996212229189942*v_B2lv_a_i + 0.249053057297486*v_B2lv_a_r - 0.996212229189942*v_B2lv_b_i - 0.249053057297486*v_B2lv_b_r + 0.598820527961361*v_B3_a_i + 1.10755301189314*v_B3_a_r - 0.171091579417532*v_B3_b_i - 0.316443717683753*v_B3_b_r - 0.171091579417532*v_B3_c_i - 0.316443717683753*v_B3_c_r
        struct[0].g[41,0] = 1.10755301189314*v_B1_a_i - 0.598820527961361*v_B1_a_r - 0.316443717683753*v_B1_b_i + 0.171091579417532*v_B1_b_r - 0.316443717683753*v_B1_c_i + 0.171091579417532*v_B1_c_r - 2.23667465123725*v_B2_a_i + 1.28353302446119*v_B2_a_r + 0.643671749092996*v_B2_b_i - 0.385473430243205*v_B2_b_r + 0.643671749092997*v_B2_c_i - 0.385473430243205*v_B2_c_r + 0.249053057297486*v_B2lv_a_i - 0.996212229189942*v_B2lv_a_r - 0.249053057297486*v_B2lv_b_i + 0.996212229189942*v_B2lv_b_r + 1.10755301189314*v_B3_a_i - 0.598820527961361*v_B3_a_r - 0.316443717683753*v_B3_b_i + 0.171091579417532*v_B3_b_r - 0.316443717683753*v_B3_c_i + 0.171091579417532*v_B3_c_r
        struct[0].g[42,0] = -0.171091579417532*v_B1_a_i - 0.316443717683753*v_B1_a_r + 0.59882052796136*v_B1_b_i + 1.10755301189314*v_B1_b_r - 0.171091579417531*v_B1_c_i - 0.316443717683753*v_B1_c_r + 0.385473430243205*v_B2_a_i + 0.643671749092996*v_B2_a_r - 1.28353302446119*v_B2_b_i - 2.23667465123725*v_B2_b_r + 0.385473430243204*v_B2_c_i + 0.643671749092997*v_B2_c_r + 0.996212229189942*v_B2lv_b_i + 0.249053057297486*v_B2lv_b_r - 0.996212229189942*v_B2lv_c_i - 0.249053057297486*v_B2lv_c_r - 0.171091579417532*v_B3_a_i - 0.316443717683753*v_B3_a_r + 0.59882052796136*v_B3_b_i + 1.10755301189314*v_B3_b_r - 0.171091579417531*v_B3_c_i - 0.316443717683753*v_B3_c_r
        struct[0].g[43,0] = -0.316443717683753*v_B1_a_i + 0.171091579417532*v_B1_a_r + 1.10755301189314*v_B1_b_i - 0.59882052796136*v_B1_b_r - 0.316443717683753*v_B1_c_i + 0.171091579417531*v_B1_c_r + 0.643671749092996*v_B2_a_i - 0.385473430243205*v_B2_a_r - 2.23667465123725*v_B2_b_i + 1.28353302446119*v_B2_b_r + 0.643671749092997*v_B2_c_i - 0.385473430243204*v_B2_c_r + 0.249053057297486*v_B2lv_b_i - 0.996212229189942*v_B2lv_b_r - 0.249053057297486*v_B2lv_c_i + 0.996212229189942*v_B2lv_c_r - 0.316443717683753*v_B3_a_i + 0.171091579417532*v_B3_a_r + 1.10755301189314*v_B3_b_i - 0.59882052796136*v_B3_b_r - 0.316443717683753*v_B3_c_i + 0.171091579417531*v_B3_c_r
        struct[0].g[44,0] = -0.171091579417532*v_B1_a_i - 0.316443717683753*v_B1_a_r - 0.171091579417531*v_B1_b_i - 0.316443717683753*v_B1_b_r + 0.59882052796136*v_B1_c_i + 1.10755301189314*v_B1_c_r + 0.385473430243205*v_B2_a_i + 0.643671749092997*v_B2_a_r + 0.385473430243204*v_B2_b_i + 0.643671749092997*v_B2_b_r - 1.28353302446119*v_B2_c_i - 2.23667465123725*v_B2_c_r - 0.996212229189942*v_B2lv_a_i - 0.249053057297486*v_B2lv_a_r + 0.996212229189942*v_B2lv_c_i + 0.249053057297486*v_B2lv_c_r - 0.171091579417532*v_B3_a_i - 0.316443717683753*v_B3_a_r - 0.171091579417531*v_B3_b_i - 0.316443717683753*v_B3_b_r + 0.59882052796136*v_B3_c_i + 1.10755301189314*v_B3_c_r
        struct[0].g[45,0] = -0.316443717683753*v_B1_a_i + 0.171091579417532*v_B1_a_r - 0.316443717683753*v_B1_b_i + 0.171091579417531*v_B1_b_r + 1.10755301189314*v_B1_c_i - 0.59882052796136*v_B1_c_r + 0.643671749092997*v_B2_a_i - 0.385473430243205*v_B2_a_r + 0.643671749092997*v_B2_b_i - 0.385473430243204*v_B2_b_r - 2.23667465123725*v_B2_c_i + 1.28353302446119*v_B2_c_r - 0.249053057297486*v_B2lv_a_i + 0.996212229189942*v_B2lv_a_r + 0.249053057297486*v_B2lv_c_i - 0.996212229189942*v_B2lv_c_r - 0.316443717683753*v_B3_a_i + 0.171091579417532*v_B3_a_r - 0.316443717683753*v_B3_b_i + 0.171091579417531*v_B3_b_r + 1.10755301189314*v_B3_c_i - 0.59882052796136*v_B3_c_r
        struct[0].g[46,0] = 0.598820527961361*v_B2_a_i + 1.10755301189314*v_B2_a_r - 0.171091579417532*v_B2_b_i - 0.316443717683753*v_B2_b_r - 0.171091579417532*v_B2_c_i - 0.316443717683753*v_B2_c_r - 1.28353302446119*v_B3_a_i - 2.23667465123725*v_B3_a_r + 0.385473430243205*v_B3_b_i + 0.643671749092996*v_B3_b_r + 0.385473430243205*v_B3_c_i + 0.643671749092997*v_B3_c_r + 0.996212229189942*v_B3lv_a_i + 0.249053057297486*v_B3lv_a_r - 0.996212229189942*v_B3lv_b_i - 0.249053057297486*v_B3lv_b_r + 0.598820527961361*v_B4_a_i + 1.10755301189314*v_B4_a_r - 0.171091579417532*v_B4_b_i - 0.316443717683753*v_B4_b_r - 0.171091579417532*v_B4_c_i - 0.316443717683753*v_B4_c_r
        struct[0].g[47,0] = 1.10755301189314*v_B2_a_i - 0.598820527961361*v_B2_a_r - 0.316443717683753*v_B2_b_i + 0.171091579417532*v_B2_b_r - 0.316443717683753*v_B2_c_i + 0.171091579417532*v_B2_c_r - 2.23667465123725*v_B3_a_i + 1.28353302446119*v_B3_a_r + 0.643671749092996*v_B3_b_i - 0.385473430243205*v_B3_b_r + 0.643671749092997*v_B3_c_i - 0.385473430243205*v_B3_c_r + 0.249053057297486*v_B3lv_a_i - 0.996212229189942*v_B3lv_a_r - 0.249053057297486*v_B3lv_b_i + 0.996212229189942*v_B3lv_b_r + 1.10755301189314*v_B4_a_i - 0.598820527961361*v_B4_a_r - 0.316443717683753*v_B4_b_i + 0.171091579417532*v_B4_b_r - 0.316443717683753*v_B4_c_i + 0.171091579417532*v_B4_c_r
        struct[0].g[48,0] = -0.171091579417532*v_B2_a_i - 0.316443717683753*v_B2_a_r + 0.59882052796136*v_B2_b_i + 1.10755301189314*v_B2_b_r - 0.171091579417531*v_B2_c_i - 0.316443717683753*v_B2_c_r + 0.385473430243205*v_B3_a_i + 0.643671749092996*v_B3_a_r - 1.28353302446119*v_B3_b_i - 2.23667465123725*v_B3_b_r + 0.385473430243204*v_B3_c_i + 0.643671749092997*v_B3_c_r + 0.996212229189942*v_B3lv_b_i + 0.249053057297486*v_B3lv_b_r - 0.996212229189942*v_B3lv_c_i - 0.249053057297486*v_B3lv_c_r - 0.171091579417532*v_B4_a_i - 0.316443717683753*v_B4_a_r + 0.59882052796136*v_B4_b_i + 1.10755301189314*v_B4_b_r - 0.171091579417531*v_B4_c_i - 0.316443717683753*v_B4_c_r
        struct[0].g[49,0] = -0.316443717683753*v_B2_a_i + 0.171091579417532*v_B2_a_r + 1.10755301189314*v_B2_b_i - 0.59882052796136*v_B2_b_r - 0.316443717683753*v_B2_c_i + 0.171091579417531*v_B2_c_r + 0.643671749092996*v_B3_a_i - 0.385473430243205*v_B3_a_r - 2.23667465123725*v_B3_b_i + 1.28353302446119*v_B3_b_r + 0.643671749092997*v_B3_c_i - 0.385473430243204*v_B3_c_r + 0.249053057297486*v_B3lv_b_i - 0.996212229189942*v_B3lv_b_r - 0.249053057297486*v_B3lv_c_i + 0.996212229189942*v_B3lv_c_r - 0.316443717683753*v_B4_a_i + 0.171091579417532*v_B4_a_r + 1.10755301189314*v_B4_b_i - 0.59882052796136*v_B4_b_r - 0.316443717683753*v_B4_c_i + 0.171091579417531*v_B4_c_r
        struct[0].g[50,0] = -0.171091579417532*v_B2_a_i - 0.316443717683753*v_B2_a_r - 0.171091579417531*v_B2_b_i - 0.316443717683753*v_B2_b_r + 0.59882052796136*v_B2_c_i + 1.10755301189314*v_B2_c_r + 0.385473430243205*v_B3_a_i + 0.643671749092997*v_B3_a_r + 0.385473430243204*v_B3_b_i + 0.643671749092997*v_B3_b_r - 1.28353302446119*v_B3_c_i - 2.23667465123725*v_B3_c_r - 0.996212229189942*v_B3lv_a_i - 0.249053057297486*v_B3lv_a_r + 0.996212229189942*v_B3lv_c_i + 0.249053057297486*v_B3lv_c_r - 0.171091579417532*v_B4_a_i - 0.316443717683753*v_B4_a_r - 0.171091579417531*v_B4_b_i - 0.316443717683753*v_B4_b_r + 0.59882052796136*v_B4_c_i + 1.10755301189314*v_B4_c_r
        struct[0].g[51,0] = -0.316443717683753*v_B2_a_i + 0.171091579417532*v_B2_a_r - 0.316443717683753*v_B2_b_i + 0.171091579417531*v_B2_b_r + 1.10755301189314*v_B2_c_i - 0.59882052796136*v_B2_c_r + 0.643671749092997*v_B3_a_i - 0.385473430243205*v_B3_a_r + 0.643671749092997*v_B3_b_i - 0.385473430243204*v_B3_b_r - 2.23667465123725*v_B3_c_i + 1.28353302446119*v_B3_c_r - 0.249053057297486*v_B3lv_a_i + 0.996212229189942*v_B3lv_a_r + 0.249053057297486*v_B3lv_c_i - 0.996212229189942*v_B3lv_c_r - 0.316443717683753*v_B4_a_i + 0.171091579417532*v_B4_a_r - 0.316443717683753*v_B4_b_i + 0.171091579417531*v_B4_b_r + 1.10755301189314*v_B4_c_i - 0.59882052796136*v_B4_c_r
        struct[0].g[52,0] = 0.598820527961361*v_B3_a_i + 1.10755301189314*v_B3_a_r - 0.171091579417532*v_B3_b_i - 0.316443717683753*v_B3_b_r - 0.171091579417532*v_B3_c_i - 0.316443717683753*v_B3_c_r - 0.684903767132556*v_B4_a_i - 1.12912163934412*v_B4_a_r + 0.214305342572583*v_B4_b_i + 0.327228031409243*v_B4_b_r + 0.214305342572583*v_B4_c_i + 0.327228031409244*v_B4_c_r + 0.996212229189942*v_B4lv_a_i + 0.249053057297486*v_B4lv_a_r - 0.996212229189942*v_B4lv_b_i - 0.249053057297486*v_B4lv_b_r
        struct[0].g[53,0] = 1.10755301189314*v_B3_a_i - 0.598820527961361*v_B3_a_r - 0.316443717683753*v_B3_b_i + 0.171091579417532*v_B3_b_r - 0.316443717683753*v_B3_c_i + 0.171091579417532*v_B3_c_r - 1.12912163934412*v_B4_a_i + 0.684903767132556*v_B4_a_r + 0.327228031409243*v_B4_b_i - 0.214305342572583*v_B4_b_r + 0.327228031409244*v_B4_c_i - 0.214305342572583*v_B4_c_r + 0.249053057297486*v_B4lv_a_i - 0.996212229189942*v_B4lv_a_r - 0.249053057297486*v_B4lv_b_i + 0.996212229189942*v_B4lv_b_r
        struct[0].g[54,0] = -0.171091579417532*v_B3_a_i - 0.316443717683753*v_B3_a_r + 0.59882052796136*v_B3_b_i + 1.10755301189314*v_B3_b_r - 0.171091579417531*v_B3_c_i - 0.316443717683753*v_B3_c_r + 0.214305342572583*v_B4_a_i + 0.327228031409243*v_B4_a_r - 0.684903767132556*v_B4_b_i - 1.12912163934412*v_B4_b_r + 0.214305342572582*v_B4_c_i + 0.327228031409244*v_B4_c_r + 0.996212229189942*v_B4lv_b_i + 0.249053057297486*v_B4lv_b_r - 0.996212229189942*v_B4lv_c_i - 0.249053057297486*v_B4lv_c_r
        struct[0].g[55,0] = -0.316443717683753*v_B3_a_i + 0.171091579417532*v_B3_a_r + 1.10755301189314*v_B3_b_i - 0.59882052796136*v_B3_b_r - 0.316443717683753*v_B3_c_i + 0.171091579417531*v_B3_c_r + 0.327228031409243*v_B4_a_i - 0.214305342572583*v_B4_a_r - 1.12912163934412*v_B4_b_i + 0.684903767132556*v_B4_b_r + 0.327228031409244*v_B4_c_i - 0.214305342572582*v_B4_c_r + 0.249053057297486*v_B4lv_b_i - 0.996212229189942*v_B4lv_b_r - 0.249053057297486*v_B4lv_c_i + 0.996212229189942*v_B4lv_c_r
        struct[0].g[56,0] = -0.171091579417532*v_B3_a_i - 0.316443717683753*v_B3_a_r - 0.171091579417531*v_B3_b_i - 0.316443717683753*v_B3_b_r + 0.59882052796136*v_B3_c_i + 1.10755301189314*v_B3_c_r + 0.214305342572583*v_B4_a_i + 0.327228031409243*v_B4_a_r + 0.214305342572582*v_B4_b_i + 0.327228031409244*v_B4_b_r - 0.684903767132556*v_B4_c_i - 1.12912163934412*v_B4_c_r - 0.996212229189942*v_B4lv_a_i - 0.249053057297486*v_B4lv_a_r + 0.996212229189942*v_B4lv_c_i + 0.249053057297486*v_B4lv_c_r
        struct[0].g[57,0] = -0.316443717683753*v_B3_a_i + 0.171091579417532*v_B3_a_r - 0.316443717683753*v_B3_b_i + 0.171091579417531*v_B3_b_r + 1.10755301189314*v_B3_c_i - 0.59882052796136*v_B3_c_r + 0.327228031409243*v_B4_a_i - 0.214305342572583*v_B4_a_r + 0.327228031409244*v_B4_b_i - 0.214305342572582*v_B4_b_r - 1.12912163934412*v_B4_c_i + 0.684903767132556*v_B4_c_r - 0.249053057297486*v_B4lv_a_i + 0.996212229189942*v_B4lv_a_r + 0.249053057297486*v_B4lv_c_i - 0.996212229189942*v_B4lv_c_r
        struct[0].g[58,0] = -0.684903767132556*v_B5_a_i - 1.12912163934412*v_B5_a_r + 0.214305342572583*v_B5_b_i + 0.327228031409243*v_B5_b_r + 0.214305342572583*v_B5_c_i + 0.327228031409244*v_B5_c_r + 0.996212229189942*v_B5lv_a_i + 0.249053057297486*v_B5lv_a_r - 0.996212229189942*v_B5lv_b_i - 0.249053057297486*v_B5lv_b_r + 0.598820527961361*v_B6_a_i + 1.10755301189314*v_B6_a_r - 0.171091579417532*v_B6_b_i - 0.316443717683753*v_B6_b_r - 0.171091579417532*v_B6_c_i - 0.316443717683753*v_B6_c_r
        struct[0].g[59,0] = -1.12912163934412*v_B5_a_i + 0.684903767132556*v_B5_a_r + 0.327228031409243*v_B5_b_i - 0.214305342572583*v_B5_b_r + 0.327228031409244*v_B5_c_i - 0.214305342572583*v_B5_c_r + 0.249053057297486*v_B5lv_a_i - 0.996212229189942*v_B5lv_a_r - 0.249053057297486*v_B5lv_b_i + 0.996212229189942*v_B5lv_b_r + 1.10755301189314*v_B6_a_i - 0.598820527961361*v_B6_a_r - 0.316443717683753*v_B6_b_i + 0.171091579417532*v_B6_b_r - 0.316443717683753*v_B6_c_i + 0.171091579417532*v_B6_c_r
        struct[0].g[60,0] = 0.214305342572583*v_B5_a_i + 0.327228031409243*v_B5_a_r - 0.684903767132556*v_B5_b_i - 1.12912163934412*v_B5_b_r + 0.214305342572582*v_B5_c_i + 0.327228031409244*v_B5_c_r + 0.996212229189942*v_B5lv_b_i + 0.249053057297486*v_B5lv_b_r - 0.996212229189942*v_B5lv_c_i - 0.249053057297486*v_B5lv_c_r - 0.171091579417532*v_B6_a_i - 0.316443717683753*v_B6_a_r + 0.59882052796136*v_B6_b_i + 1.10755301189314*v_B6_b_r - 0.171091579417531*v_B6_c_i - 0.316443717683753*v_B6_c_r
        struct[0].g[61,0] = 0.327228031409243*v_B5_a_i - 0.214305342572583*v_B5_a_r - 1.12912163934412*v_B5_b_i + 0.684903767132556*v_B5_b_r + 0.327228031409244*v_B5_c_i - 0.214305342572582*v_B5_c_r + 0.249053057297486*v_B5lv_b_i - 0.996212229189942*v_B5lv_b_r - 0.249053057297486*v_B5lv_c_i + 0.996212229189942*v_B5lv_c_r - 0.316443717683753*v_B6_a_i + 0.171091579417532*v_B6_a_r + 1.10755301189314*v_B6_b_i - 0.59882052796136*v_B6_b_r - 0.316443717683753*v_B6_c_i + 0.171091579417531*v_B6_c_r
        struct[0].g[62,0] = 0.214305342572583*v_B5_a_i + 0.327228031409243*v_B5_a_r + 0.214305342572582*v_B5_b_i + 0.327228031409244*v_B5_b_r - 0.684903767132556*v_B5_c_i - 1.12912163934412*v_B5_c_r - 0.996212229189942*v_B5lv_a_i - 0.249053057297486*v_B5lv_a_r + 0.996212229189942*v_B5lv_c_i + 0.249053057297486*v_B5lv_c_r - 0.171091579417532*v_B6_a_i - 0.316443717683753*v_B6_a_r - 0.171091579417531*v_B6_b_i - 0.316443717683753*v_B6_b_r + 0.59882052796136*v_B6_c_i + 1.10755301189314*v_B6_c_r
        struct[0].g[63,0] = 0.327228031409243*v_B5_a_i - 0.214305342572583*v_B5_a_r + 0.327228031409244*v_B5_b_i - 0.214305342572582*v_B5_b_r - 1.12912163934412*v_B5_c_i + 0.684903767132556*v_B5_c_r - 0.249053057297486*v_B5lv_a_i + 0.996212229189942*v_B5lv_a_r + 0.249053057297486*v_B5lv_c_i - 0.996212229189942*v_B5lv_c_r - 0.316443717683753*v_B6_a_i + 0.171091579417532*v_B6_a_r - 0.316443717683753*v_B6_b_i + 0.171091579417531*v_B6_b_r + 1.10755301189314*v_B6_c_i - 0.59882052796136*v_B6_c_r
        struct[0].g[64,0] = 0.598820527961361*v_B5_a_i + 1.10755301189314*v_B5_a_r - 0.171091579417532*v_B5_b_i - 0.316443717683753*v_B5_b_r - 0.171091579417532*v_B5_c_i - 0.316443717683753*v_B5_c_r - 1.28353302446119*v_B6_a_i - 2.23667465123725*v_B6_a_r + 0.385473430243205*v_B6_b_i + 0.643671749092996*v_B6_b_r + 0.385473430243205*v_B6_c_i + 0.643671749092997*v_B6_c_r + 0.996212229189942*v_B6lv_a_i + 0.249053057297486*v_B6lv_a_r - 0.996212229189942*v_B6lv_b_i - 0.249053057297486*v_B6lv_b_r + 0.598820527961361*v_B7_a_i + 1.10755301189314*v_B7_a_r - 0.171091579417532*v_B7_b_i - 0.316443717683753*v_B7_b_r - 0.171091579417532*v_B7_c_i - 0.316443717683753*v_B7_c_r
        struct[0].g[65,0] = 1.10755301189314*v_B5_a_i - 0.598820527961361*v_B5_a_r - 0.316443717683753*v_B5_b_i + 0.171091579417532*v_B5_b_r - 0.316443717683753*v_B5_c_i + 0.171091579417532*v_B5_c_r - 2.23667465123725*v_B6_a_i + 1.28353302446119*v_B6_a_r + 0.643671749092996*v_B6_b_i - 0.385473430243205*v_B6_b_r + 0.643671749092997*v_B6_c_i - 0.385473430243205*v_B6_c_r + 0.249053057297486*v_B6lv_a_i - 0.996212229189942*v_B6lv_a_r - 0.249053057297486*v_B6lv_b_i + 0.996212229189942*v_B6lv_b_r + 1.10755301189314*v_B7_a_i - 0.598820527961361*v_B7_a_r - 0.316443717683753*v_B7_b_i + 0.171091579417532*v_B7_b_r - 0.316443717683753*v_B7_c_i + 0.171091579417532*v_B7_c_r
        struct[0].g[66,0] = -0.171091579417532*v_B5_a_i - 0.316443717683753*v_B5_a_r + 0.59882052796136*v_B5_b_i + 1.10755301189314*v_B5_b_r - 0.171091579417531*v_B5_c_i - 0.316443717683753*v_B5_c_r + 0.385473430243205*v_B6_a_i + 0.643671749092996*v_B6_a_r - 1.28353302446119*v_B6_b_i - 2.23667465123725*v_B6_b_r + 0.385473430243204*v_B6_c_i + 0.643671749092997*v_B6_c_r + 0.996212229189942*v_B6lv_b_i + 0.249053057297486*v_B6lv_b_r - 0.996212229189942*v_B6lv_c_i - 0.249053057297486*v_B6lv_c_r - 0.171091579417532*v_B7_a_i - 0.316443717683753*v_B7_a_r + 0.59882052796136*v_B7_b_i + 1.10755301189314*v_B7_b_r - 0.171091579417531*v_B7_c_i - 0.316443717683753*v_B7_c_r
        struct[0].g[67,0] = -0.316443717683753*v_B5_a_i + 0.171091579417532*v_B5_a_r + 1.10755301189314*v_B5_b_i - 0.59882052796136*v_B5_b_r - 0.316443717683753*v_B5_c_i + 0.171091579417531*v_B5_c_r + 0.643671749092996*v_B6_a_i - 0.385473430243205*v_B6_a_r - 2.23667465123725*v_B6_b_i + 1.28353302446119*v_B6_b_r + 0.643671749092997*v_B6_c_i - 0.385473430243204*v_B6_c_r + 0.249053057297486*v_B6lv_b_i - 0.996212229189942*v_B6lv_b_r - 0.249053057297486*v_B6lv_c_i + 0.996212229189942*v_B6lv_c_r - 0.316443717683753*v_B7_a_i + 0.171091579417532*v_B7_a_r + 1.10755301189314*v_B7_b_i - 0.59882052796136*v_B7_b_r - 0.316443717683753*v_B7_c_i + 0.171091579417531*v_B7_c_r
        struct[0].g[68,0] = -0.171091579417532*v_B5_a_i - 0.316443717683753*v_B5_a_r - 0.171091579417531*v_B5_b_i - 0.316443717683753*v_B5_b_r + 0.59882052796136*v_B5_c_i + 1.10755301189314*v_B5_c_r + 0.385473430243205*v_B6_a_i + 0.643671749092997*v_B6_a_r + 0.385473430243204*v_B6_b_i + 0.643671749092997*v_B6_b_r - 1.28353302446119*v_B6_c_i - 2.23667465123725*v_B6_c_r - 0.996212229189942*v_B6lv_a_i - 0.249053057297486*v_B6lv_a_r + 0.996212229189942*v_B6lv_c_i + 0.249053057297486*v_B6lv_c_r - 0.171091579417532*v_B7_a_i - 0.316443717683753*v_B7_a_r - 0.171091579417531*v_B7_b_i - 0.316443717683753*v_B7_b_r + 0.59882052796136*v_B7_c_i + 1.10755301189314*v_B7_c_r
        struct[0].g[69,0] = -0.316443717683753*v_B5_a_i + 0.171091579417532*v_B5_a_r - 0.316443717683753*v_B5_b_i + 0.171091579417531*v_B5_b_r + 1.10755301189314*v_B5_c_i - 0.59882052796136*v_B5_c_r + 0.643671749092997*v_B6_a_i - 0.385473430243205*v_B6_a_r + 0.643671749092997*v_B6_b_i - 0.385473430243204*v_B6_b_r - 2.23667465123725*v_B6_c_i + 1.28353302446119*v_B6_c_r - 0.249053057297486*v_B6lv_a_i + 0.996212229189942*v_B6lv_a_r + 0.249053057297486*v_B6lv_c_i - 0.996212229189942*v_B6lv_c_r - 0.316443717683753*v_B7_a_i + 0.171091579417532*v_B7_a_r - 0.316443717683753*v_B7_b_i + 0.171091579417531*v_B7_b_r + 1.10755301189314*v_B7_c_i - 0.59882052796136*v_B7_c_r
        struct[0].g[70,0] = -i_t_B2_B2lv_a_r + 0.0862745098039216*v_B2_a_i + 0.0215686274509804*v_B2_a_r - 0.0431372549019608*v_B2_b_i - 0.0107843137254902*v_B2_b_r - 0.0431372549019608*v_B2_c_i - 0.0107843137254902*v_B2_c_r - 0.996212229189942*v_B2lv_a_i - 0.249053057297486*v_B2lv_a_r + 0.996212229189942*v_B2lv_b_i + 0.249053057297486*v_B2lv_b_r
        struct[0].g[71,0] = -i_t_B2_B2lv_a_i + 0.0215686274509804*v_B2_a_i - 0.0862745098039216*v_B2_a_r - 0.0107843137254902*v_B2_b_i + 0.0431372549019608*v_B2_b_r - 0.0107843137254902*v_B2_c_i + 0.0431372549019608*v_B2_c_r - 0.249053057297486*v_B2lv_a_i + 0.996212229189942*v_B2lv_a_r + 0.249053057297486*v_B2lv_b_i - 0.996212229189942*v_B2lv_b_r
        struct[0].g[72,0] = -i_t_B2_B2lv_b_r - 0.0431372549019608*v_B2_a_i - 0.0107843137254902*v_B2_a_r + 0.0862745098039216*v_B2_b_i + 0.0215686274509804*v_B2_b_r - 0.0431372549019608*v_B2_c_i - 0.0107843137254902*v_B2_c_r - 0.996212229189942*v_B2lv_b_i - 0.249053057297486*v_B2lv_b_r + 0.996212229189942*v_B2lv_c_i + 0.249053057297486*v_B2lv_c_r
        struct[0].g[73,0] = -i_t_B2_B2lv_b_i - 0.0107843137254902*v_B2_a_i + 0.0431372549019608*v_B2_a_r + 0.0215686274509804*v_B2_b_i - 0.0862745098039216*v_B2_b_r - 0.0107843137254902*v_B2_c_i + 0.0431372549019608*v_B2_c_r - 0.249053057297486*v_B2lv_b_i + 0.996212229189942*v_B2lv_b_r + 0.249053057297486*v_B2lv_c_i - 0.996212229189942*v_B2lv_c_r
        struct[0].g[74,0] = -i_t_B2_B2lv_c_r - 0.0431372549019608*v_B2_a_i - 0.0107843137254902*v_B2_a_r - 0.0431372549019608*v_B2_b_i - 0.0107843137254902*v_B2_b_r + 0.0862745098039216*v_B2_c_i + 0.0215686274509804*v_B2_c_r + 0.996212229189942*v_B2lv_a_i + 0.249053057297486*v_B2lv_a_r - 0.996212229189942*v_B2lv_c_i - 0.249053057297486*v_B2lv_c_r
        struct[0].g[75,0] = -i_t_B2_B2lv_c_i - 0.0107843137254902*v_B2_a_i + 0.0431372549019608*v_B2_a_r - 0.0107843137254902*v_B2_b_i + 0.0431372549019608*v_B2_b_r + 0.0215686274509804*v_B2_c_i - 0.0862745098039216*v_B2_c_r + 0.249053057297486*v_B2lv_a_i - 0.996212229189942*v_B2lv_a_r - 0.249053057297486*v_B2lv_c_i + 0.996212229189942*v_B2lv_c_r
        struct[0].g[76,0] = -i_t_B3_B3lv_a_r + 0.0862745098039216*v_B3_a_i + 0.0215686274509804*v_B3_a_r - 0.0431372549019608*v_B3_b_i - 0.0107843137254902*v_B3_b_r - 0.0431372549019608*v_B3_c_i - 0.0107843137254902*v_B3_c_r - 0.996212229189942*v_B3lv_a_i - 0.249053057297486*v_B3lv_a_r + 0.996212229189942*v_B3lv_b_i + 0.249053057297486*v_B3lv_b_r
        struct[0].g[77,0] = -i_t_B3_B3lv_a_i + 0.0215686274509804*v_B3_a_i - 0.0862745098039216*v_B3_a_r - 0.0107843137254902*v_B3_b_i + 0.0431372549019608*v_B3_b_r - 0.0107843137254902*v_B3_c_i + 0.0431372549019608*v_B3_c_r - 0.249053057297486*v_B3lv_a_i + 0.996212229189942*v_B3lv_a_r + 0.249053057297486*v_B3lv_b_i - 0.996212229189942*v_B3lv_b_r
        struct[0].g[78,0] = -i_t_B3_B3lv_b_r - 0.0431372549019608*v_B3_a_i - 0.0107843137254902*v_B3_a_r + 0.0862745098039216*v_B3_b_i + 0.0215686274509804*v_B3_b_r - 0.0431372549019608*v_B3_c_i - 0.0107843137254902*v_B3_c_r - 0.996212229189942*v_B3lv_b_i - 0.249053057297486*v_B3lv_b_r + 0.996212229189942*v_B3lv_c_i + 0.249053057297486*v_B3lv_c_r
        struct[0].g[79,0] = -i_t_B3_B3lv_b_i - 0.0107843137254902*v_B3_a_i + 0.0431372549019608*v_B3_a_r + 0.0215686274509804*v_B3_b_i - 0.0862745098039216*v_B3_b_r - 0.0107843137254902*v_B3_c_i + 0.0431372549019608*v_B3_c_r - 0.249053057297486*v_B3lv_b_i + 0.996212229189942*v_B3lv_b_r + 0.249053057297486*v_B3lv_c_i - 0.996212229189942*v_B3lv_c_r
        struct[0].g[80,0] = -i_t_B3_B3lv_c_r - 0.0431372549019608*v_B3_a_i - 0.0107843137254902*v_B3_a_r - 0.0431372549019608*v_B3_b_i - 0.0107843137254902*v_B3_b_r + 0.0862745098039216*v_B3_c_i + 0.0215686274509804*v_B3_c_r + 0.996212229189942*v_B3lv_a_i + 0.249053057297486*v_B3lv_a_r - 0.996212229189942*v_B3lv_c_i - 0.249053057297486*v_B3lv_c_r
        struct[0].g[81,0] = -i_t_B3_B3lv_c_i - 0.0107843137254902*v_B3_a_i + 0.0431372549019608*v_B3_a_r - 0.0107843137254902*v_B3_b_i + 0.0431372549019608*v_B3_b_r + 0.0215686274509804*v_B3_c_i - 0.0862745098039216*v_B3_c_r + 0.249053057297486*v_B3lv_a_i - 0.996212229189942*v_B3lv_a_r - 0.249053057297486*v_B3lv_c_i + 0.996212229189942*v_B3lv_c_r
        struct[0].g[82,0] = -i_t_B4_B4lv_a_r + 0.0862745098039216*v_B4_a_i + 0.0215686274509804*v_B4_a_r - 0.0431372549019608*v_B4_b_i - 0.0107843137254902*v_B4_b_r - 0.0431372549019608*v_B4_c_i - 0.0107843137254902*v_B4_c_r - 0.996212229189942*v_B4lv_a_i - 0.249053057297486*v_B4lv_a_r + 0.996212229189942*v_B4lv_b_i + 0.249053057297486*v_B4lv_b_r
        struct[0].g[83,0] = -i_t_B4_B4lv_a_i + 0.0215686274509804*v_B4_a_i - 0.0862745098039216*v_B4_a_r - 0.0107843137254902*v_B4_b_i + 0.0431372549019608*v_B4_b_r - 0.0107843137254902*v_B4_c_i + 0.0431372549019608*v_B4_c_r - 0.249053057297486*v_B4lv_a_i + 0.996212229189942*v_B4lv_a_r + 0.249053057297486*v_B4lv_b_i - 0.996212229189942*v_B4lv_b_r
        struct[0].g[84,0] = -i_t_B4_B4lv_b_r - 0.0431372549019608*v_B4_a_i - 0.0107843137254902*v_B4_a_r + 0.0862745098039216*v_B4_b_i + 0.0215686274509804*v_B4_b_r - 0.0431372549019608*v_B4_c_i - 0.0107843137254902*v_B4_c_r - 0.996212229189942*v_B4lv_b_i - 0.249053057297486*v_B4lv_b_r + 0.996212229189942*v_B4lv_c_i + 0.249053057297486*v_B4lv_c_r
        struct[0].g[85,0] = -i_t_B4_B4lv_b_i - 0.0107843137254902*v_B4_a_i + 0.0431372549019608*v_B4_a_r + 0.0215686274509804*v_B4_b_i - 0.0862745098039216*v_B4_b_r - 0.0107843137254902*v_B4_c_i + 0.0431372549019608*v_B4_c_r - 0.249053057297486*v_B4lv_b_i + 0.996212229189942*v_B4lv_b_r + 0.249053057297486*v_B4lv_c_i - 0.996212229189942*v_B4lv_c_r
        struct[0].g[86,0] = -i_t_B4_B4lv_c_r - 0.0431372549019608*v_B4_a_i - 0.0107843137254902*v_B4_a_r - 0.0431372549019608*v_B4_b_i - 0.0107843137254902*v_B4_b_r + 0.0862745098039216*v_B4_c_i + 0.0215686274509804*v_B4_c_r + 0.996212229189942*v_B4lv_a_i + 0.249053057297486*v_B4lv_a_r - 0.996212229189942*v_B4lv_c_i - 0.249053057297486*v_B4lv_c_r
        struct[0].g[87,0] = -i_t_B4_B4lv_c_i - 0.0107843137254902*v_B4_a_i + 0.0431372549019608*v_B4_a_r - 0.0107843137254902*v_B4_b_i + 0.0431372549019608*v_B4_b_r + 0.0215686274509804*v_B4_c_i - 0.0862745098039216*v_B4_c_r + 0.249053057297486*v_B4lv_a_i - 0.996212229189942*v_B4lv_a_r - 0.249053057297486*v_B4lv_c_i + 0.996212229189942*v_B4lv_c_r
        struct[0].g[88,0] = -i_t_B5_B5lv_a_r + 0.0862745098039216*v_B5_a_i + 0.0215686274509804*v_B5_a_r - 0.0431372549019608*v_B5_b_i - 0.0107843137254902*v_B5_b_r - 0.0431372549019608*v_B5_c_i - 0.0107843137254902*v_B5_c_r - 0.996212229189942*v_B5lv_a_i - 0.249053057297486*v_B5lv_a_r + 0.996212229189942*v_B5lv_b_i + 0.249053057297486*v_B5lv_b_r
        struct[0].g[89,0] = -i_t_B5_B5lv_a_i + 0.0215686274509804*v_B5_a_i - 0.0862745098039216*v_B5_a_r - 0.0107843137254902*v_B5_b_i + 0.0431372549019608*v_B5_b_r - 0.0107843137254902*v_B5_c_i + 0.0431372549019608*v_B5_c_r - 0.249053057297486*v_B5lv_a_i + 0.996212229189942*v_B5lv_a_r + 0.249053057297486*v_B5lv_b_i - 0.996212229189942*v_B5lv_b_r
        struct[0].g[90,0] = -i_t_B5_B5lv_b_r - 0.0431372549019608*v_B5_a_i - 0.0107843137254902*v_B5_a_r + 0.0862745098039216*v_B5_b_i + 0.0215686274509804*v_B5_b_r - 0.0431372549019608*v_B5_c_i - 0.0107843137254902*v_B5_c_r - 0.996212229189942*v_B5lv_b_i - 0.249053057297486*v_B5lv_b_r + 0.996212229189942*v_B5lv_c_i + 0.249053057297486*v_B5lv_c_r
        struct[0].g[91,0] = -i_t_B5_B5lv_b_i - 0.0107843137254902*v_B5_a_i + 0.0431372549019608*v_B5_a_r + 0.0215686274509804*v_B5_b_i - 0.0862745098039216*v_B5_b_r - 0.0107843137254902*v_B5_c_i + 0.0431372549019608*v_B5_c_r - 0.249053057297486*v_B5lv_b_i + 0.996212229189942*v_B5lv_b_r + 0.249053057297486*v_B5lv_c_i - 0.996212229189942*v_B5lv_c_r
        struct[0].g[92,0] = -i_t_B5_B5lv_c_r - 0.0431372549019608*v_B5_a_i - 0.0107843137254902*v_B5_a_r - 0.0431372549019608*v_B5_b_i - 0.0107843137254902*v_B5_b_r + 0.0862745098039216*v_B5_c_i + 0.0215686274509804*v_B5_c_r + 0.996212229189942*v_B5lv_a_i + 0.249053057297486*v_B5lv_a_r - 0.996212229189942*v_B5lv_c_i - 0.249053057297486*v_B5lv_c_r
        struct[0].g[93,0] = -i_t_B5_B5lv_c_i - 0.0107843137254902*v_B5_a_i + 0.0431372549019608*v_B5_a_r - 0.0107843137254902*v_B5_b_i + 0.0431372549019608*v_B5_b_r + 0.0215686274509804*v_B5_c_i - 0.0862745098039216*v_B5_c_r + 0.249053057297486*v_B5lv_a_i - 0.996212229189942*v_B5lv_a_r - 0.249053057297486*v_B5lv_c_i + 0.996212229189942*v_B5lv_c_r
        struct[0].g[94,0] = -i_t_B6_B6lv_a_r + 0.0862745098039216*v_B6_a_i + 0.0215686274509804*v_B6_a_r - 0.0431372549019608*v_B6_b_i - 0.0107843137254902*v_B6_b_r - 0.0431372549019608*v_B6_c_i - 0.0107843137254902*v_B6_c_r - 0.996212229189942*v_B6lv_a_i - 0.249053057297486*v_B6lv_a_r + 0.996212229189942*v_B6lv_b_i + 0.249053057297486*v_B6lv_b_r
        struct[0].g[95,0] = -i_t_B6_B6lv_a_i + 0.0215686274509804*v_B6_a_i - 0.0862745098039216*v_B6_a_r - 0.0107843137254902*v_B6_b_i + 0.0431372549019608*v_B6_b_r - 0.0107843137254902*v_B6_c_i + 0.0431372549019608*v_B6_c_r - 0.249053057297486*v_B6lv_a_i + 0.996212229189942*v_B6lv_a_r + 0.249053057297486*v_B6lv_b_i - 0.996212229189942*v_B6lv_b_r
        struct[0].g[96,0] = -i_t_B6_B6lv_b_r - 0.0431372549019608*v_B6_a_i - 0.0107843137254902*v_B6_a_r + 0.0862745098039216*v_B6_b_i + 0.0215686274509804*v_B6_b_r - 0.0431372549019608*v_B6_c_i - 0.0107843137254902*v_B6_c_r - 0.996212229189942*v_B6lv_b_i - 0.249053057297486*v_B6lv_b_r + 0.996212229189942*v_B6lv_c_i + 0.249053057297486*v_B6lv_c_r
        struct[0].g[97,0] = -i_t_B6_B6lv_b_i - 0.0107843137254902*v_B6_a_i + 0.0431372549019608*v_B6_a_r + 0.0215686274509804*v_B6_b_i - 0.0862745098039216*v_B6_b_r - 0.0107843137254902*v_B6_c_i + 0.0431372549019608*v_B6_c_r - 0.249053057297486*v_B6lv_b_i + 0.996212229189942*v_B6lv_b_r + 0.249053057297486*v_B6lv_c_i - 0.996212229189942*v_B6lv_c_r
        struct[0].g[98,0] = -i_t_B6_B6lv_c_r - 0.0431372549019608*v_B6_a_i - 0.0107843137254902*v_B6_a_r - 0.0431372549019608*v_B6_b_i - 0.0107843137254902*v_B6_b_r + 0.0862745098039216*v_B6_c_i + 0.0215686274509804*v_B6_c_r + 0.996212229189942*v_B6lv_a_i + 0.249053057297486*v_B6lv_a_r - 0.996212229189942*v_B6lv_c_i - 0.249053057297486*v_B6lv_c_r
        struct[0].g[99,0] = -i_t_B6_B6lv_c_i - 0.0107843137254902*v_B6_a_i + 0.0431372549019608*v_B6_a_r - 0.0107843137254902*v_B6_b_i + 0.0431372549019608*v_B6_b_r + 0.0215686274509804*v_B6_c_i - 0.0862745098039216*v_B6_c_r + 0.249053057297486*v_B6lv_a_i - 0.996212229189942*v_B6lv_a_r - 0.249053057297486*v_B6lv_c_i + 0.996212229189942*v_B6lv_c_r
        struct[0].g[100,0] = -i_l_B1_B2_a_r + 0.598820527961361*v_B1_a_i + 1.10755301189314*v_B1_a_r - 0.171091579417532*v_B1_b_i - 0.316443717683753*v_B1_b_r - 0.171091579417532*v_B1_c_i - 0.316443717683753*v_B1_c_r - 0.598820527961361*v_B2_a_i - 1.10755301189314*v_B2_a_r + 0.171091579417532*v_B2_b_i + 0.316443717683753*v_B2_b_r + 0.171091579417532*v_B2_c_i + 0.316443717683753*v_B2_c_r
        struct[0].g[101,0] = -i_l_B1_B2_a_i + 1.10755301189314*v_B1_a_i - 0.598820527961361*v_B1_a_r - 0.316443717683753*v_B1_b_i + 0.171091579417532*v_B1_b_r - 0.316443717683753*v_B1_c_i + 0.171091579417532*v_B1_c_r - 1.10755301189314*v_B2_a_i + 0.598820527961361*v_B2_a_r + 0.316443717683753*v_B2_b_i - 0.171091579417532*v_B2_b_r + 0.316443717683753*v_B2_c_i - 0.171091579417532*v_B2_c_r
        struct[0].g[102,0] = -i_l_B1_B2_b_r - 0.171091579417532*v_B1_a_i - 0.316443717683753*v_B1_a_r + 0.59882052796136*v_B1_b_i + 1.10755301189314*v_B1_b_r - 0.171091579417531*v_B1_c_i - 0.316443717683753*v_B1_c_r + 0.171091579417532*v_B2_a_i + 0.316443717683753*v_B2_a_r - 0.59882052796136*v_B2_b_i - 1.10755301189314*v_B2_b_r + 0.171091579417531*v_B2_c_i + 0.316443717683753*v_B2_c_r
        struct[0].g[103,0] = -i_l_B1_B2_b_i - 0.316443717683753*v_B1_a_i + 0.171091579417532*v_B1_a_r + 1.10755301189314*v_B1_b_i - 0.59882052796136*v_B1_b_r - 0.316443717683753*v_B1_c_i + 0.171091579417531*v_B1_c_r + 0.316443717683753*v_B2_a_i - 0.171091579417532*v_B2_a_r - 1.10755301189314*v_B2_b_i + 0.59882052796136*v_B2_b_r + 0.316443717683753*v_B2_c_i - 0.171091579417531*v_B2_c_r
        struct[0].g[104,0] = -i_l_B1_B2_c_r - 0.171091579417532*v_B1_a_i - 0.316443717683753*v_B1_a_r - 0.171091579417531*v_B1_b_i - 0.316443717683753*v_B1_b_r + 0.59882052796136*v_B1_c_i + 1.10755301189314*v_B1_c_r + 0.171091579417532*v_B2_a_i + 0.316443717683753*v_B2_a_r + 0.171091579417531*v_B2_b_i + 0.316443717683753*v_B2_b_r - 0.59882052796136*v_B2_c_i - 1.10755301189314*v_B2_c_r
        struct[0].g[105,0] = -i_l_B1_B2_c_i - 0.316443717683753*v_B1_a_i + 0.171091579417532*v_B1_a_r - 0.316443717683753*v_B1_b_i + 0.171091579417531*v_B1_b_r + 1.10755301189314*v_B1_c_i - 0.59882052796136*v_B1_c_r + 0.316443717683753*v_B2_a_i - 0.171091579417532*v_B2_a_r + 0.316443717683753*v_B2_b_i - 0.171091579417531*v_B2_b_r - 1.10755301189314*v_B2_c_i + 0.59882052796136*v_B2_c_r
        struct[0].g[106,0] = -i_l_B2_B3_a_r + 0.598820527961361*v_B2_a_i + 1.10755301189314*v_B2_a_r - 0.171091579417532*v_B2_b_i - 0.316443717683753*v_B2_b_r - 0.171091579417532*v_B2_c_i - 0.316443717683753*v_B2_c_r - 0.598820527961361*v_B3_a_i - 1.10755301189314*v_B3_a_r + 0.171091579417532*v_B3_b_i + 0.316443717683753*v_B3_b_r + 0.171091579417532*v_B3_c_i + 0.316443717683753*v_B3_c_r
        struct[0].g[107,0] = -i_l_B2_B3_a_i + 1.10755301189314*v_B2_a_i - 0.598820527961361*v_B2_a_r - 0.316443717683753*v_B2_b_i + 0.171091579417532*v_B2_b_r - 0.316443717683753*v_B2_c_i + 0.171091579417532*v_B2_c_r - 1.10755301189314*v_B3_a_i + 0.598820527961361*v_B3_a_r + 0.316443717683753*v_B3_b_i - 0.171091579417532*v_B3_b_r + 0.316443717683753*v_B3_c_i - 0.171091579417532*v_B3_c_r
        struct[0].g[108,0] = -i_l_B2_B3_b_r - 0.171091579417532*v_B2_a_i - 0.316443717683753*v_B2_a_r + 0.59882052796136*v_B2_b_i + 1.10755301189314*v_B2_b_r - 0.171091579417531*v_B2_c_i - 0.316443717683753*v_B2_c_r + 0.171091579417532*v_B3_a_i + 0.316443717683753*v_B3_a_r - 0.59882052796136*v_B3_b_i - 1.10755301189314*v_B3_b_r + 0.171091579417531*v_B3_c_i + 0.316443717683753*v_B3_c_r
        struct[0].g[109,0] = -i_l_B2_B3_b_i - 0.316443717683753*v_B2_a_i + 0.171091579417532*v_B2_a_r + 1.10755301189314*v_B2_b_i - 0.59882052796136*v_B2_b_r - 0.316443717683753*v_B2_c_i + 0.171091579417531*v_B2_c_r + 0.316443717683753*v_B3_a_i - 0.171091579417532*v_B3_a_r - 1.10755301189314*v_B3_b_i + 0.59882052796136*v_B3_b_r + 0.316443717683753*v_B3_c_i - 0.171091579417531*v_B3_c_r
        struct[0].g[110,0] = -i_l_B2_B3_c_r - 0.171091579417532*v_B2_a_i - 0.316443717683753*v_B2_a_r - 0.171091579417531*v_B2_b_i - 0.316443717683753*v_B2_b_r + 0.59882052796136*v_B2_c_i + 1.10755301189314*v_B2_c_r + 0.171091579417532*v_B3_a_i + 0.316443717683753*v_B3_a_r + 0.171091579417531*v_B3_b_i + 0.316443717683753*v_B3_b_r - 0.59882052796136*v_B3_c_i - 1.10755301189314*v_B3_c_r
        struct[0].g[111,0] = -i_l_B2_B3_c_i - 0.316443717683753*v_B2_a_i + 0.171091579417532*v_B2_a_r - 0.316443717683753*v_B2_b_i + 0.171091579417531*v_B2_b_r + 1.10755301189314*v_B2_c_i - 0.59882052796136*v_B2_c_r + 0.316443717683753*v_B3_a_i - 0.171091579417532*v_B3_a_r + 0.316443717683753*v_B3_b_i - 0.171091579417531*v_B3_b_r - 1.10755301189314*v_B3_c_i + 0.59882052796136*v_B3_c_r
        struct[0].g[112,0] = -i_l_B3_B4_a_r + 0.598820527961361*v_B3_a_i + 1.10755301189314*v_B3_a_r - 0.171091579417532*v_B3_b_i - 0.316443717683753*v_B3_b_r - 0.171091579417532*v_B3_c_i - 0.316443717683753*v_B3_c_r - 0.598820527961361*v_B4_a_i - 1.10755301189314*v_B4_a_r + 0.171091579417532*v_B4_b_i + 0.316443717683753*v_B4_b_r + 0.171091579417532*v_B4_c_i + 0.316443717683753*v_B4_c_r
        struct[0].g[113,0] = -i_l_B3_B4_a_i + 1.10755301189314*v_B3_a_i - 0.598820527961361*v_B3_a_r - 0.316443717683753*v_B3_b_i + 0.171091579417532*v_B3_b_r - 0.316443717683753*v_B3_c_i + 0.171091579417532*v_B3_c_r - 1.10755301189314*v_B4_a_i + 0.598820527961361*v_B4_a_r + 0.316443717683753*v_B4_b_i - 0.171091579417532*v_B4_b_r + 0.316443717683753*v_B4_c_i - 0.171091579417532*v_B4_c_r
        struct[0].g[114,0] = -i_l_B3_B4_b_r - 0.171091579417532*v_B3_a_i - 0.316443717683753*v_B3_a_r + 0.59882052796136*v_B3_b_i + 1.10755301189314*v_B3_b_r - 0.171091579417531*v_B3_c_i - 0.316443717683753*v_B3_c_r + 0.171091579417532*v_B4_a_i + 0.316443717683753*v_B4_a_r - 0.59882052796136*v_B4_b_i - 1.10755301189314*v_B4_b_r + 0.171091579417531*v_B4_c_i + 0.316443717683753*v_B4_c_r
        struct[0].g[115,0] = -i_l_B3_B4_b_i - 0.316443717683753*v_B3_a_i + 0.171091579417532*v_B3_a_r + 1.10755301189314*v_B3_b_i - 0.59882052796136*v_B3_b_r - 0.316443717683753*v_B3_c_i + 0.171091579417531*v_B3_c_r + 0.316443717683753*v_B4_a_i - 0.171091579417532*v_B4_a_r - 1.10755301189314*v_B4_b_i + 0.59882052796136*v_B4_b_r + 0.316443717683753*v_B4_c_i - 0.171091579417531*v_B4_c_r
        struct[0].g[116,0] = -i_l_B3_B4_c_r - 0.171091579417532*v_B3_a_i - 0.316443717683753*v_B3_a_r - 0.171091579417531*v_B3_b_i - 0.316443717683753*v_B3_b_r + 0.59882052796136*v_B3_c_i + 1.10755301189314*v_B3_c_r + 0.171091579417532*v_B4_a_i + 0.316443717683753*v_B4_a_r + 0.171091579417531*v_B4_b_i + 0.316443717683753*v_B4_b_r - 0.59882052796136*v_B4_c_i - 1.10755301189314*v_B4_c_r
        struct[0].g[117,0] = -i_l_B3_B4_c_i - 0.316443717683753*v_B3_a_i + 0.171091579417532*v_B3_a_r - 0.316443717683753*v_B3_b_i + 0.171091579417531*v_B3_b_r + 1.10755301189314*v_B3_c_i - 0.59882052796136*v_B3_c_r + 0.316443717683753*v_B4_a_i - 0.171091579417532*v_B4_a_r + 0.316443717683753*v_B4_b_i - 0.171091579417531*v_B4_b_r - 1.10755301189314*v_B4_c_i + 0.59882052796136*v_B4_c_r
        struct[0].g[118,0] = -i_l_B5_B6_a_r + 0.598820527961361*v_B5_a_i + 1.10755301189314*v_B5_a_r - 0.171091579417532*v_B5_b_i - 0.316443717683753*v_B5_b_r - 0.171091579417532*v_B5_c_i - 0.316443717683753*v_B5_c_r - 0.598820527961361*v_B6_a_i - 1.10755301189314*v_B6_a_r + 0.171091579417532*v_B6_b_i + 0.316443717683753*v_B6_b_r + 0.171091579417532*v_B6_c_i + 0.316443717683753*v_B6_c_r
        struct[0].g[119,0] = -i_l_B5_B6_a_i + 1.10755301189314*v_B5_a_i - 0.598820527961361*v_B5_a_r - 0.316443717683753*v_B5_b_i + 0.171091579417532*v_B5_b_r - 0.316443717683753*v_B5_c_i + 0.171091579417532*v_B5_c_r - 1.10755301189314*v_B6_a_i + 0.598820527961361*v_B6_a_r + 0.316443717683753*v_B6_b_i - 0.171091579417532*v_B6_b_r + 0.316443717683753*v_B6_c_i - 0.171091579417532*v_B6_c_r
        struct[0].g[120,0] = -i_l_B5_B6_b_r - 0.171091579417532*v_B5_a_i - 0.316443717683753*v_B5_a_r + 0.59882052796136*v_B5_b_i + 1.10755301189314*v_B5_b_r - 0.171091579417531*v_B5_c_i - 0.316443717683753*v_B5_c_r + 0.171091579417532*v_B6_a_i + 0.316443717683753*v_B6_a_r - 0.59882052796136*v_B6_b_i - 1.10755301189314*v_B6_b_r + 0.171091579417531*v_B6_c_i + 0.316443717683753*v_B6_c_r
        struct[0].g[121,0] = -i_l_B5_B6_b_i - 0.316443717683753*v_B5_a_i + 0.171091579417532*v_B5_a_r + 1.10755301189314*v_B5_b_i - 0.59882052796136*v_B5_b_r - 0.316443717683753*v_B5_c_i + 0.171091579417531*v_B5_c_r + 0.316443717683753*v_B6_a_i - 0.171091579417532*v_B6_a_r - 1.10755301189314*v_B6_b_i + 0.59882052796136*v_B6_b_r + 0.316443717683753*v_B6_c_i - 0.171091579417531*v_B6_c_r
        struct[0].g[122,0] = -i_l_B5_B6_c_r - 0.171091579417532*v_B5_a_i - 0.316443717683753*v_B5_a_r - 0.171091579417531*v_B5_b_i - 0.316443717683753*v_B5_b_r + 0.59882052796136*v_B5_c_i + 1.10755301189314*v_B5_c_r + 0.171091579417532*v_B6_a_i + 0.316443717683753*v_B6_a_r + 0.171091579417531*v_B6_b_i + 0.316443717683753*v_B6_b_r - 0.59882052796136*v_B6_c_i - 1.10755301189314*v_B6_c_r
        struct[0].g[123,0] = -i_l_B5_B6_c_i - 0.316443717683753*v_B5_a_i + 0.171091579417532*v_B5_a_r - 0.316443717683753*v_B5_b_i + 0.171091579417531*v_B5_b_r + 1.10755301189314*v_B5_c_i - 0.59882052796136*v_B5_c_r + 0.316443717683753*v_B6_a_i - 0.171091579417532*v_B6_a_r + 0.316443717683753*v_B6_b_i - 0.171091579417531*v_B6_b_r - 1.10755301189314*v_B6_c_i + 0.59882052796136*v_B6_c_r
        struct[0].g[124,0] = -i_l_B6_B7_a_r + 0.598820527961361*v_B6_a_i + 1.10755301189314*v_B6_a_r - 0.171091579417532*v_B6_b_i - 0.316443717683753*v_B6_b_r - 0.171091579417532*v_B6_c_i - 0.316443717683753*v_B6_c_r - 0.598820527961361*v_B7_a_i - 1.10755301189314*v_B7_a_r + 0.171091579417532*v_B7_b_i + 0.316443717683753*v_B7_b_r + 0.171091579417532*v_B7_c_i + 0.316443717683753*v_B7_c_r
        struct[0].g[125,0] = -i_l_B6_B7_a_i + 1.10755301189314*v_B6_a_i - 0.598820527961361*v_B6_a_r - 0.316443717683753*v_B6_b_i + 0.171091579417532*v_B6_b_r - 0.316443717683753*v_B6_c_i + 0.171091579417532*v_B6_c_r - 1.10755301189314*v_B7_a_i + 0.598820527961361*v_B7_a_r + 0.316443717683753*v_B7_b_i - 0.171091579417532*v_B7_b_r + 0.316443717683753*v_B7_c_i - 0.171091579417532*v_B7_c_r
        struct[0].g[126,0] = -i_l_B6_B7_b_r - 0.171091579417532*v_B6_a_i - 0.316443717683753*v_B6_a_r + 0.59882052796136*v_B6_b_i + 1.10755301189314*v_B6_b_r - 0.171091579417531*v_B6_c_i - 0.316443717683753*v_B6_c_r + 0.171091579417532*v_B7_a_i + 0.316443717683753*v_B7_a_r - 0.59882052796136*v_B7_b_i - 1.10755301189314*v_B7_b_r + 0.171091579417531*v_B7_c_i + 0.316443717683753*v_B7_c_r
        struct[0].g[127,0] = -i_l_B6_B7_b_i - 0.316443717683753*v_B6_a_i + 0.171091579417532*v_B6_a_r + 1.10755301189314*v_B6_b_i - 0.59882052796136*v_B6_b_r - 0.316443717683753*v_B6_c_i + 0.171091579417531*v_B6_c_r + 0.316443717683753*v_B7_a_i - 0.171091579417532*v_B7_a_r - 1.10755301189314*v_B7_b_i + 0.59882052796136*v_B7_b_r + 0.316443717683753*v_B7_c_i - 0.171091579417531*v_B7_c_r
        struct[0].g[128,0] = -i_l_B6_B7_c_r - 0.171091579417532*v_B6_a_i - 0.316443717683753*v_B6_a_r - 0.171091579417531*v_B6_b_i - 0.316443717683753*v_B6_b_r + 0.59882052796136*v_B6_c_i + 1.10755301189314*v_B6_c_r + 0.171091579417532*v_B7_a_i + 0.316443717683753*v_B7_a_r + 0.171091579417531*v_B7_b_i + 0.316443717683753*v_B7_b_r - 0.59882052796136*v_B7_c_i - 1.10755301189314*v_B7_c_r
        struct[0].g[129,0] = -i_l_B6_B7_c_i - 0.316443717683753*v_B6_a_i + 0.171091579417532*v_B6_a_r - 0.316443717683753*v_B6_b_i + 0.171091579417531*v_B6_b_r + 1.10755301189314*v_B6_c_i - 0.59882052796136*v_B6_c_r + 0.316443717683753*v_B7_a_i - 0.171091579417532*v_B7_a_r + 0.316443717683753*v_B7_b_i - 0.171091579417531*v_B7_b_r - 1.10755301189314*v_B7_c_i + 0.59882052796136*v_B7_c_r
        struct[0].g[130,0] = i_load_B2lv_a_i*v_B2lv_a_i - i_load_B2lv_a_i*v_B2lv_n_i + i_load_B2lv_a_r*v_B2lv_a_r - i_load_B2lv_a_r*v_B2lv_n_r - p_B2lv_a
        struct[0].g[131,0] = i_load_B2lv_b_i*v_B2lv_b_i - i_load_B2lv_b_i*v_B2lv_n_i + i_load_B2lv_b_r*v_B2lv_b_r - i_load_B2lv_b_r*v_B2lv_n_r - p_B2lv_b
        struct[0].g[132,0] = i_load_B2lv_c_i*v_B2lv_c_i - i_load_B2lv_c_i*v_B2lv_n_i + i_load_B2lv_c_r*v_B2lv_c_r - i_load_B2lv_c_r*v_B2lv_n_r - p_B2lv_c
        struct[0].g[133,0] = -i_load_B2lv_a_i*v_B2lv_a_r + i_load_B2lv_a_i*v_B2lv_n_r + i_load_B2lv_a_r*v_B2lv_a_i - i_load_B2lv_a_r*v_B2lv_n_i - q_B2lv_a
        struct[0].g[134,0] = -i_load_B2lv_b_i*v_B2lv_b_r + i_load_B2lv_b_i*v_B2lv_n_r + i_load_B2lv_b_r*v_B2lv_b_i - i_load_B2lv_b_r*v_B2lv_n_i - q_B2lv_b
        struct[0].g[135,0] = -i_load_B2lv_c_i*v_B2lv_c_r + i_load_B2lv_c_i*v_B2lv_n_r + i_load_B2lv_c_r*v_B2lv_c_i - i_load_B2lv_c_r*v_B2lv_n_i - q_B2lv_c
        struct[0].g[136,0] = i_load_B2lv_a_r + i_load_B2lv_b_r + i_load_B2lv_c_r + i_load_B2lv_n_r
        struct[0].g[137,0] = i_load_B2lv_a_i + i_load_B2lv_b_i + i_load_B2lv_c_i + i_load_B2lv_n_i
        struct[0].g[138,0] = i_load_B3lv_a_i*v_B3lv_a_i - i_load_B3lv_a_i*v_B3lv_n_i + i_load_B3lv_a_r*v_B3lv_a_r - i_load_B3lv_a_r*v_B3lv_n_r - p_B3lv_a
        struct[0].g[139,0] = i_load_B3lv_b_i*v_B3lv_b_i - i_load_B3lv_b_i*v_B3lv_n_i + i_load_B3lv_b_r*v_B3lv_b_r - i_load_B3lv_b_r*v_B3lv_n_r - p_B3lv_b
        struct[0].g[140,0] = i_load_B3lv_c_i*v_B3lv_c_i - i_load_B3lv_c_i*v_B3lv_n_i + i_load_B3lv_c_r*v_B3lv_c_r - i_load_B3lv_c_r*v_B3lv_n_r - p_B3lv_c
        struct[0].g[141,0] = -i_load_B3lv_a_i*v_B3lv_a_r + i_load_B3lv_a_i*v_B3lv_n_r + i_load_B3lv_a_r*v_B3lv_a_i - i_load_B3lv_a_r*v_B3lv_n_i - q_B3lv_a
        struct[0].g[142,0] = -i_load_B3lv_b_i*v_B3lv_b_r + i_load_B3lv_b_i*v_B3lv_n_r + i_load_B3lv_b_r*v_B3lv_b_i - i_load_B3lv_b_r*v_B3lv_n_i - q_B3lv_b
        struct[0].g[143,0] = -i_load_B3lv_c_i*v_B3lv_c_r + i_load_B3lv_c_i*v_B3lv_n_r + i_load_B3lv_c_r*v_B3lv_c_i - i_load_B3lv_c_r*v_B3lv_n_i - q_B3lv_c
        struct[0].g[144,0] = i_load_B3lv_a_r + i_load_B3lv_b_r + i_load_B3lv_c_r + i_load_B3lv_n_r
        struct[0].g[145,0] = i_load_B3lv_a_i + i_load_B3lv_b_i + i_load_B3lv_c_i + i_load_B3lv_n_i
        struct[0].g[146,0] = i_load_B4lv_a_i*v_B4lv_a_i - i_load_B4lv_a_i*v_B4lv_n_i + i_load_B4lv_a_r*v_B4lv_a_r - i_load_B4lv_a_r*v_B4lv_n_r - p_B4lv_a
        struct[0].g[147,0] = i_load_B4lv_b_i*v_B4lv_b_i - i_load_B4lv_b_i*v_B4lv_n_i + i_load_B4lv_b_r*v_B4lv_b_r - i_load_B4lv_b_r*v_B4lv_n_r - p_B4lv_b
        struct[0].g[148,0] = i_load_B4lv_c_i*v_B4lv_c_i - i_load_B4lv_c_i*v_B4lv_n_i + i_load_B4lv_c_r*v_B4lv_c_r - i_load_B4lv_c_r*v_B4lv_n_r - p_B4lv_c
        struct[0].g[149,0] = -i_load_B4lv_a_i*v_B4lv_a_r + i_load_B4lv_a_i*v_B4lv_n_r + i_load_B4lv_a_r*v_B4lv_a_i - i_load_B4lv_a_r*v_B4lv_n_i - q_B4lv_a
        struct[0].g[150,0] = -i_load_B4lv_b_i*v_B4lv_b_r + i_load_B4lv_b_i*v_B4lv_n_r + i_load_B4lv_b_r*v_B4lv_b_i - i_load_B4lv_b_r*v_B4lv_n_i - q_B4lv_b
        struct[0].g[151,0] = -i_load_B4lv_c_i*v_B4lv_c_r + i_load_B4lv_c_i*v_B4lv_n_r + i_load_B4lv_c_r*v_B4lv_c_i - i_load_B4lv_c_r*v_B4lv_n_i - q_B4lv_c
        struct[0].g[152,0] = i_load_B4lv_a_r + i_load_B4lv_b_r + i_load_B4lv_c_r + i_load_B4lv_n_r
        struct[0].g[153,0] = i_load_B4lv_a_i + i_load_B4lv_b_i + i_load_B4lv_c_i + i_load_B4lv_n_i
        struct[0].g[154,0] = i_load_B5lv_a_i*v_B5lv_a_i - i_load_B5lv_a_i*v_B5lv_n_i + i_load_B5lv_a_r*v_B5lv_a_r - i_load_B5lv_a_r*v_B5lv_n_r - p_B5lv_a
        struct[0].g[155,0] = i_load_B5lv_b_i*v_B5lv_b_i - i_load_B5lv_b_i*v_B5lv_n_i + i_load_B5lv_b_r*v_B5lv_b_r - i_load_B5lv_b_r*v_B5lv_n_r - p_B5lv_b
        struct[0].g[156,0] = i_load_B5lv_c_i*v_B5lv_c_i - i_load_B5lv_c_i*v_B5lv_n_i + i_load_B5lv_c_r*v_B5lv_c_r - i_load_B5lv_c_r*v_B5lv_n_r - p_B5lv_c
        struct[0].g[157,0] = -i_load_B5lv_a_i*v_B5lv_a_r + i_load_B5lv_a_i*v_B5lv_n_r + i_load_B5lv_a_r*v_B5lv_a_i - i_load_B5lv_a_r*v_B5lv_n_i - q_B5lv_a
        struct[0].g[158,0] = -i_load_B5lv_b_i*v_B5lv_b_r + i_load_B5lv_b_i*v_B5lv_n_r + i_load_B5lv_b_r*v_B5lv_b_i - i_load_B5lv_b_r*v_B5lv_n_i - q_B5lv_b
        struct[0].g[159,0] = -i_load_B5lv_c_i*v_B5lv_c_r + i_load_B5lv_c_i*v_B5lv_n_r + i_load_B5lv_c_r*v_B5lv_c_i - i_load_B5lv_c_r*v_B5lv_n_i - q_B5lv_c
        struct[0].g[160,0] = i_load_B5lv_a_r + i_load_B5lv_b_r + i_load_B5lv_c_r + i_load_B5lv_n_r
        struct[0].g[161,0] = i_load_B5lv_a_i + i_load_B5lv_b_i + i_load_B5lv_c_i + i_load_B5lv_n_i
        struct[0].g[162,0] = i_load_B6lv_a_i*v_B6lv_a_i - i_load_B6lv_a_i*v_B6lv_n_i + i_load_B6lv_a_r*v_B6lv_a_r - i_load_B6lv_a_r*v_B6lv_n_r - p_B6lv_a
        struct[0].g[163,0] = i_load_B6lv_b_i*v_B6lv_b_i - i_load_B6lv_b_i*v_B6lv_n_i + i_load_B6lv_b_r*v_B6lv_b_r - i_load_B6lv_b_r*v_B6lv_n_r - p_B6lv_b
        struct[0].g[164,0] = i_load_B6lv_c_i*v_B6lv_c_i - i_load_B6lv_c_i*v_B6lv_n_i + i_load_B6lv_c_r*v_B6lv_c_r - i_load_B6lv_c_r*v_B6lv_n_r - p_B6lv_c
        struct[0].g[165,0] = -i_load_B6lv_a_i*v_B6lv_a_r + i_load_B6lv_a_i*v_B6lv_n_r + i_load_B6lv_a_r*v_B6lv_a_i - i_load_B6lv_a_r*v_B6lv_n_i - q_B6lv_a
        struct[0].g[166,0] = -i_load_B6lv_b_i*v_B6lv_b_r + i_load_B6lv_b_i*v_B6lv_n_r + i_load_B6lv_b_r*v_B6lv_b_i - i_load_B6lv_b_r*v_B6lv_n_i - q_B6lv_b
        struct[0].g[167,0] = -i_load_B6lv_c_i*v_B6lv_c_r + i_load_B6lv_c_i*v_B6lv_n_r + i_load_B6lv_c_r*v_B6lv_c_i - i_load_B6lv_c_r*v_B6lv_n_i - q_B6lv_c
        struct[0].g[168,0] = i_load_B6lv_a_r + i_load_B6lv_b_r + i_load_B6lv_c_r + i_load_B6lv_n_r
        struct[0].g[169,0] = i_load_B6lv_a_i + i_load_B6lv_b_i + i_load_B6lv_c_i + i_load_B6lv_n_i
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = (v_B1_a_i**2 + v_B1_a_r**2)**0.5
        struct[0].h[1,0] = (v_B1_b_i**2 + v_B1_b_r**2)**0.5
        struct[0].h[2,0] = (v_B1_c_i**2 + v_B1_c_r**2)**0.5
        struct[0].h[3,0] = (v_B7_a_i**2 + v_B7_a_r**2)**0.5
        struct[0].h[4,0] = (v_B7_b_i**2 + v_B7_b_r**2)**0.5
        struct[0].h[5,0] = (v_B7_c_i**2 + v_B7_c_r**2)**0.5
        struct[0].h[6,0] = (v_B2lv_a_i**2 + v_B2lv_a_r**2)**0.5
        struct[0].h[7,0] = (v_B2lv_b_i**2 + v_B2lv_b_r**2)**0.5
        struct[0].h[8,0] = (v_B2lv_c_i**2 + v_B2lv_c_r**2)**0.5
        struct[0].h[9,0] = (v_B2lv_n_i**2 + v_B2lv_n_r**2)**0.5
        struct[0].h[10,0] = (v_B3lv_a_i**2 + v_B3lv_a_r**2)**0.5
        struct[0].h[11,0] = (v_B3lv_b_i**2 + v_B3lv_b_r**2)**0.5
        struct[0].h[12,0] = (v_B3lv_c_i**2 + v_B3lv_c_r**2)**0.5
        struct[0].h[13,0] = (v_B3lv_n_i**2 + v_B3lv_n_r**2)**0.5
        struct[0].h[14,0] = (v_B4lv_a_i**2 + v_B4lv_a_r**2)**0.5
        struct[0].h[15,0] = (v_B4lv_b_i**2 + v_B4lv_b_r**2)**0.5
        struct[0].h[16,0] = (v_B4lv_c_i**2 + v_B4lv_c_r**2)**0.5
        struct[0].h[17,0] = (v_B4lv_n_i**2 + v_B4lv_n_r**2)**0.5
        struct[0].h[18,0] = (v_B5lv_a_i**2 + v_B5lv_a_r**2)**0.5
        struct[0].h[19,0] = (v_B5lv_b_i**2 + v_B5lv_b_r**2)**0.5
        struct[0].h[20,0] = (v_B5lv_c_i**2 + v_B5lv_c_r**2)**0.5
        struct[0].h[21,0] = (v_B5lv_n_i**2 + v_B5lv_n_r**2)**0.5
        struct[0].h[22,0] = (v_B6lv_a_i**2 + v_B6lv_a_r**2)**0.5
        struct[0].h[23,0] = (v_B6lv_b_i**2 + v_B6lv_b_r**2)**0.5
        struct[0].h[24,0] = (v_B6lv_c_i**2 + v_B6lv_c_r**2)**0.5
        struct[0].h[25,0] = (v_B6lv_n_i**2 + v_B6lv_n_r**2)**0.5
        struct[0].h[26,0] = (v_B2_a_i**2 + v_B2_a_r**2)**0.5
        struct[0].h[27,0] = (v_B2_b_i**2 + v_B2_b_r**2)**0.5
        struct[0].h[28,0] = (v_B2_c_i**2 + v_B2_c_r**2)**0.5
        struct[0].h[29,0] = (v_B3_a_i**2 + v_B3_a_r**2)**0.5
        struct[0].h[30,0] = (v_B3_b_i**2 + v_B3_b_r**2)**0.5
        struct[0].h[31,0] = (v_B3_c_i**2 + v_B3_c_r**2)**0.5
        struct[0].h[32,0] = (v_B4_a_i**2 + v_B4_a_r**2)**0.5
        struct[0].h[33,0] = (v_B4_b_i**2 + v_B4_b_r**2)**0.5
        struct[0].h[34,0] = (v_B4_c_i**2 + v_B4_c_r**2)**0.5
        struct[0].h[35,0] = (v_B5_a_i**2 + v_B5_a_r**2)**0.5
        struct[0].h[36,0] = (v_B5_b_i**2 + v_B5_b_r**2)**0.5
        struct[0].h[37,0] = (v_B5_c_i**2 + v_B5_c_r**2)**0.5
        struct[0].h[38,0] = (v_B6_a_i**2 + v_B6_a_r**2)**0.5
        struct[0].h[39,0] = (v_B6_b_i**2 + v_B6_b_r**2)**0.5
        struct[0].h[40,0] = (v_B6_c_i**2 + v_B6_c_r**2)**0.5
    

    if mode == 10:

        struct[0].Fx_ini[0,0] = -1

    if mode == 11:


        struct[0].Gy_ini[0,0] = -5.75163398692810
        struct[0].Gy_ini[0,1] = -23.0065359477124
        struct[0].Gy_ini[0,6] = 5.75163398692810
        struct[0].Gy_ini[0,7] = 23.0065359477124
        struct[0].Gy_ini[0,40] = 0.249053057297486
        struct[0].Gy_ini[0,41] = 0.996212229189942
        struct[0].Gy_ini[0,44] = -0.249053057297486
        struct[0].Gy_ini[0,45] = -0.996212229189942
        struct[0].Gy_ini[0,130] = 1
        struct[0].Gy_ini[1,0] = 23.0065359477124
        struct[0].Gy_ini[1,1] = -5.75163398692810
        struct[0].Gy_ini[1,6] = -23.0065359477124
        struct[0].Gy_ini[1,7] = 5.75163398692810
        struct[0].Gy_ini[1,40] = -0.996212229189942
        struct[0].Gy_ini[1,41] = 0.249053057297486
        struct[0].Gy_ini[1,44] = 0.996212229189942
        struct[0].Gy_ini[1,45] = -0.249053057297486
        struct[0].Gy_ini[1,131] = 1
        struct[0].Gy_ini[2,2] = -5.75163398692810
        struct[0].Gy_ini[2,3] = -23.0065359477124
        struct[0].Gy_ini[2,6] = 5.75163398692810
        struct[0].Gy_ini[2,7] = 23.0065359477124
        struct[0].Gy_ini[2,40] = -0.249053057297486
        struct[0].Gy_ini[2,41] = -0.996212229189942
        struct[0].Gy_ini[2,42] = 0.249053057297486
        struct[0].Gy_ini[2,43] = 0.996212229189942
        struct[0].Gy_ini[2,132] = 1
        struct[0].Gy_ini[3,2] = 23.0065359477124
        struct[0].Gy_ini[3,3] = -5.75163398692810
        struct[0].Gy_ini[3,6] = -23.0065359477124
        struct[0].Gy_ini[3,7] = 5.75163398692810
        struct[0].Gy_ini[3,40] = 0.996212229189942
        struct[0].Gy_ini[3,41] = -0.249053057297486
        struct[0].Gy_ini[3,42] = -0.996212229189942
        struct[0].Gy_ini[3,43] = 0.249053057297486
        struct[0].Gy_ini[3,133] = 1
        struct[0].Gy_ini[4,4] = -5.75163398692810
        struct[0].Gy_ini[4,5] = -23.0065359477124
        struct[0].Gy_ini[4,6] = 5.75163398692810
        struct[0].Gy_ini[4,7] = 23.0065359477124
        struct[0].Gy_ini[4,42] = -0.249053057297486
        struct[0].Gy_ini[4,43] = -0.996212229189942
        struct[0].Gy_ini[4,44] = 0.249053057297486
        struct[0].Gy_ini[4,45] = 0.996212229189942
        struct[0].Gy_ini[4,134] = 1
        struct[0].Gy_ini[5,4] = 23.0065359477124
        struct[0].Gy_ini[5,5] = -5.75163398692810
        struct[0].Gy_ini[5,6] = -23.0065359477124
        struct[0].Gy_ini[5,7] = 5.75163398692810
        struct[0].Gy_ini[5,42] = 0.996212229189942
        struct[0].Gy_ini[5,43] = -0.249053057297486
        struct[0].Gy_ini[5,44] = -0.996212229189942
        struct[0].Gy_ini[5,45] = 0.249053057297486
        struct[0].Gy_ini[5,135] = 1
        struct[0].Gy_ini[6,0] = 5.75163398692810
        struct[0].Gy_ini[6,1] = 23.0065359477124
        struct[0].Gy_ini[6,2] = 5.75163398692810
        struct[0].Gy_ini[6,3] = 23.0065359477124
        struct[0].Gy_ini[6,4] = 5.75163398692810
        struct[0].Gy_ini[6,5] = 23.0065359477124
        struct[0].Gy_ini[6,6] = -1017.25490196078
        struct[0].Gy_ini[6,7] = -69.0196078431372
        struct[0].Gy_ini[7,0] = -23.0065359477124
        struct[0].Gy_ini[7,1] = 5.75163398692810
        struct[0].Gy_ini[7,2] = -23.0065359477124
        struct[0].Gy_ini[7,3] = 5.75163398692810
        struct[0].Gy_ini[7,4] = -23.0065359477124
        struct[0].Gy_ini[7,5] = 5.75163398692810
        struct[0].Gy_ini[7,6] = 69.0196078431372
        struct[0].Gy_ini[7,7] = -1017.25490196078
        struct[0].Gy_ini[8,8] = -5.75163398692810
        struct[0].Gy_ini[8,9] = -23.0065359477124
        struct[0].Gy_ini[8,14] = 5.75163398692810
        struct[0].Gy_ini[8,15] = 23.0065359477124
        struct[0].Gy_ini[8,46] = 0.249053057297486
        struct[0].Gy_ini[8,47] = 0.996212229189942
        struct[0].Gy_ini[8,50] = -0.249053057297486
        struct[0].Gy_ini[8,51] = -0.996212229189942
        struct[0].Gy_ini[8,138] = 1
        struct[0].Gy_ini[9,8] = 23.0065359477124
        struct[0].Gy_ini[9,9] = -5.75163398692810
        struct[0].Gy_ini[9,14] = -23.0065359477124
        struct[0].Gy_ini[9,15] = 5.75163398692810
        struct[0].Gy_ini[9,46] = -0.996212229189942
        struct[0].Gy_ini[9,47] = 0.249053057297486
        struct[0].Gy_ini[9,50] = 0.996212229189942
        struct[0].Gy_ini[9,51] = -0.249053057297486
        struct[0].Gy_ini[9,139] = 1
        struct[0].Gy_ini[10,10] = -5.75163398692810
        struct[0].Gy_ini[10,11] = -23.0065359477124
        struct[0].Gy_ini[10,14] = 5.75163398692810
        struct[0].Gy_ini[10,15] = 23.0065359477124
        struct[0].Gy_ini[10,46] = -0.249053057297486
        struct[0].Gy_ini[10,47] = -0.996212229189942
        struct[0].Gy_ini[10,48] = 0.249053057297486
        struct[0].Gy_ini[10,49] = 0.996212229189942
        struct[0].Gy_ini[10,140] = 1
        struct[0].Gy_ini[11,10] = 23.0065359477124
        struct[0].Gy_ini[11,11] = -5.75163398692810
        struct[0].Gy_ini[11,14] = -23.0065359477124
        struct[0].Gy_ini[11,15] = 5.75163398692810
        struct[0].Gy_ini[11,46] = 0.996212229189942
        struct[0].Gy_ini[11,47] = -0.249053057297486
        struct[0].Gy_ini[11,48] = -0.996212229189942
        struct[0].Gy_ini[11,49] = 0.249053057297486
        struct[0].Gy_ini[11,141] = 1
        struct[0].Gy_ini[12,12] = -5.75163398692810
        struct[0].Gy_ini[12,13] = -23.0065359477124
        struct[0].Gy_ini[12,14] = 5.75163398692810
        struct[0].Gy_ini[12,15] = 23.0065359477124
        struct[0].Gy_ini[12,48] = -0.249053057297486
        struct[0].Gy_ini[12,49] = -0.996212229189942
        struct[0].Gy_ini[12,50] = 0.249053057297486
        struct[0].Gy_ini[12,51] = 0.996212229189942
        struct[0].Gy_ini[12,142] = 1
        struct[0].Gy_ini[13,12] = 23.0065359477124
        struct[0].Gy_ini[13,13] = -5.75163398692810
        struct[0].Gy_ini[13,14] = -23.0065359477124
        struct[0].Gy_ini[13,15] = 5.75163398692810
        struct[0].Gy_ini[13,48] = 0.996212229189942
        struct[0].Gy_ini[13,49] = -0.249053057297486
        struct[0].Gy_ini[13,50] = -0.996212229189942
        struct[0].Gy_ini[13,51] = 0.249053057297486
        struct[0].Gy_ini[13,143] = 1
        struct[0].Gy_ini[14,8] = 5.75163398692810
        struct[0].Gy_ini[14,9] = 23.0065359477124
        struct[0].Gy_ini[14,10] = 5.75163398692810
        struct[0].Gy_ini[14,11] = 23.0065359477124
        struct[0].Gy_ini[14,12] = 5.75163398692810
        struct[0].Gy_ini[14,13] = 23.0065359477124
        struct[0].Gy_ini[14,14] = -1017.25490196078
        struct[0].Gy_ini[14,15] = -69.0196078431372
        struct[0].Gy_ini[15,8] = -23.0065359477124
        struct[0].Gy_ini[15,9] = 5.75163398692810
        struct[0].Gy_ini[15,10] = -23.0065359477124
        struct[0].Gy_ini[15,11] = 5.75163398692810
        struct[0].Gy_ini[15,12] = -23.0065359477124
        struct[0].Gy_ini[15,13] = 5.75163398692810
        struct[0].Gy_ini[15,14] = 69.0196078431372
        struct[0].Gy_ini[15,15] = -1017.25490196078
        struct[0].Gy_ini[16,16] = -5.75163398692810
        struct[0].Gy_ini[16,17] = -23.0065359477124
        struct[0].Gy_ini[16,22] = 5.75163398692810
        struct[0].Gy_ini[16,23] = 23.0065359477124
        struct[0].Gy_ini[16,52] = 0.249053057297486
        struct[0].Gy_ini[16,53] = 0.996212229189942
        struct[0].Gy_ini[16,56] = -0.249053057297486
        struct[0].Gy_ini[16,57] = -0.996212229189942
        struct[0].Gy_ini[16,146] = 1
        struct[0].Gy_ini[17,16] = 23.0065359477124
        struct[0].Gy_ini[17,17] = -5.75163398692810
        struct[0].Gy_ini[17,22] = -23.0065359477124
        struct[0].Gy_ini[17,23] = 5.75163398692810
        struct[0].Gy_ini[17,52] = -0.996212229189942
        struct[0].Gy_ini[17,53] = 0.249053057297486
        struct[0].Gy_ini[17,56] = 0.996212229189942
        struct[0].Gy_ini[17,57] = -0.249053057297486
        struct[0].Gy_ini[17,147] = 1
        struct[0].Gy_ini[18,18] = -5.75163398692810
        struct[0].Gy_ini[18,19] = -23.0065359477124
        struct[0].Gy_ini[18,22] = 5.75163398692810
        struct[0].Gy_ini[18,23] = 23.0065359477124
        struct[0].Gy_ini[18,52] = -0.249053057297486
        struct[0].Gy_ini[18,53] = -0.996212229189942
        struct[0].Gy_ini[18,54] = 0.249053057297486
        struct[0].Gy_ini[18,55] = 0.996212229189942
        struct[0].Gy_ini[18,148] = 1
        struct[0].Gy_ini[19,18] = 23.0065359477124
        struct[0].Gy_ini[19,19] = -5.75163398692810
        struct[0].Gy_ini[19,22] = -23.0065359477124
        struct[0].Gy_ini[19,23] = 5.75163398692810
        struct[0].Gy_ini[19,52] = 0.996212229189942
        struct[0].Gy_ini[19,53] = -0.249053057297486
        struct[0].Gy_ini[19,54] = -0.996212229189942
        struct[0].Gy_ini[19,55] = 0.249053057297486
        struct[0].Gy_ini[19,149] = 1
        struct[0].Gy_ini[20,20] = -5.75163398692810
        struct[0].Gy_ini[20,21] = -23.0065359477124
        struct[0].Gy_ini[20,22] = 5.75163398692810
        struct[0].Gy_ini[20,23] = 23.0065359477124
        struct[0].Gy_ini[20,54] = -0.249053057297486
        struct[0].Gy_ini[20,55] = -0.996212229189942
        struct[0].Gy_ini[20,56] = 0.249053057297486
        struct[0].Gy_ini[20,57] = 0.996212229189942
        struct[0].Gy_ini[20,150] = 1
        struct[0].Gy_ini[21,20] = 23.0065359477124
        struct[0].Gy_ini[21,21] = -5.75163398692810
        struct[0].Gy_ini[21,22] = -23.0065359477124
        struct[0].Gy_ini[21,23] = 5.75163398692810
        struct[0].Gy_ini[21,54] = 0.996212229189942
        struct[0].Gy_ini[21,55] = -0.249053057297486
        struct[0].Gy_ini[21,56] = -0.996212229189942
        struct[0].Gy_ini[21,57] = 0.249053057297486
        struct[0].Gy_ini[21,151] = 1
        struct[0].Gy_ini[22,16] = 5.75163398692810
        struct[0].Gy_ini[22,17] = 23.0065359477124
        struct[0].Gy_ini[22,18] = 5.75163398692810
        struct[0].Gy_ini[22,19] = 23.0065359477124
        struct[0].Gy_ini[22,20] = 5.75163398692810
        struct[0].Gy_ini[22,21] = 23.0065359477124
        struct[0].Gy_ini[22,22] = -1017.25490196078
        struct[0].Gy_ini[22,23] = -69.0196078431372
        struct[0].Gy_ini[23,16] = -23.0065359477124
        struct[0].Gy_ini[23,17] = 5.75163398692810
        struct[0].Gy_ini[23,18] = -23.0065359477124
        struct[0].Gy_ini[23,19] = 5.75163398692810
        struct[0].Gy_ini[23,20] = -23.0065359477124
        struct[0].Gy_ini[23,21] = 5.75163398692810
        struct[0].Gy_ini[23,22] = 69.0196078431372
        struct[0].Gy_ini[23,23] = -1017.25490196078
        struct[0].Gy_ini[24,24] = -5.75163398692810
        struct[0].Gy_ini[24,25] = -23.0065359477124
        struct[0].Gy_ini[24,30] = 5.75163398692810
        struct[0].Gy_ini[24,31] = 23.0065359477124
        struct[0].Gy_ini[24,58] = 0.249053057297486
        struct[0].Gy_ini[24,59] = 0.996212229189942
        struct[0].Gy_ini[24,62] = -0.249053057297486
        struct[0].Gy_ini[24,63] = -0.996212229189942
        struct[0].Gy_ini[24,154] = 1
        struct[0].Gy_ini[25,24] = 23.0065359477124
        struct[0].Gy_ini[25,25] = -5.75163398692810
        struct[0].Gy_ini[25,30] = -23.0065359477124
        struct[0].Gy_ini[25,31] = 5.75163398692810
        struct[0].Gy_ini[25,58] = -0.996212229189942
        struct[0].Gy_ini[25,59] = 0.249053057297486
        struct[0].Gy_ini[25,62] = 0.996212229189942
        struct[0].Gy_ini[25,63] = -0.249053057297486
        struct[0].Gy_ini[25,155] = 1
        struct[0].Gy_ini[26,26] = -5.75163398692810
        struct[0].Gy_ini[26,27] = -23.0065359477124
        struct[0].Gy_ini[26,30] = 5.75163398692810
        struct[0].Gy_ini[26,31] = 23.0065359477124
        struct[0].Gy_ini[26,58] = -0.249053057297486
        struct[0].Gy_ini[26,59] = -0.996212229189942
        struct[0].Gy_ini[26,60] = 0.249053057297486
        struct[0].Gy_ini[26,61] = 0.996212229189942
        struct[0].Gy_ini[26,156] = 1
        struct[0].Gy_ini[27,26] = 23.0065359477124
        struct[0].Gy_ini[27,27] = -5.75163398692810
        struct[0].Gy_ini[27,30] = -23.0065359477124
        struct[0].Gy_ini[27,31] = 5.75163398692810
        struct[0].Gy_ini[27,58] = 0.996212229189942
        struct[0].Gy_ini[27,59] = -0.249053057297486
        struct[0].Gy_ini[27,60] = -0.996212229189942
        struct[0].Gy_ini[27,61] = 0.249053057297486
        struct[0].Gy_ini[27,157] = 1
        struct[0].Gy_ini[28,28] = -5.75163398692810
        struct[0].Gy_ini[28,29] = -23.0065359477124
        struct[0].Gy_ini[28,30] = 5.75163398692810
        struct[0].Gy_ini[28,31] = 23.0065359477124
        struct[0].Gy_ini[28,60] = -0.249053057297486
        struct[0].Gy_ini[28,61] = -0.996212229189942
        struct[0].Gy_ini[28,62] = 0.249053057297486
        struct[0].Gy_ini[28,63] = 0.996212229189942
        struct[0].Gy_ini[28,158] = 1
        struct[0].Gy_ini[29,28] = 23.0065359477124
        struct[0].Gy_ini[29,29] = -5.75163398692810
        struct[0].Gy_ini[29,30] = -23.0065359477124
        struct[0].Gy_ini[29,31] = 5.75163398692810
        struct[0].Gy_ini[29,60] = 0.996212229189942
        struct[0].Gy_ini[29,61] = -0.249053057297486
        struct[0].Gy_ini[29,62] = -0.996212229189942
        struct[0].Gy_ini[29,63] = 0.249053057297486
        struct[0].Gy_ini[29,159] = 1
        struct[0].Gy_ini[30,24] = 5.75163398692810
        struct[0].Gy_ini[30,25] = 23.0065359477124
        struct[0].Gy_ini[30,26] = 5.75163398692810
        struct[0].Gy_ini[30,27] = 23.0065359477124
        struct[0].Gy_ini[30,28] = 5.75163398692810
        struct[0].Gy_ini[30,29] = 23.0065359477124
        struct[0].Gy_ini[30,30] = -1017.25490196078
        struct[0].Gy_ini[30,31] = -69.0196078431372
        struct[0].Gy_ini[31,24] = -23.0065359477124
        struct[0].Gy_ini[31,25] = 5.75163398692810
        struct[0].Gy_ini[31,26] = -23.0065359477124
        struct[0].Gy_ini[31,27] = 5.75163398692810
        struct[0].Gy_ini[31,28] = -23.0065359477124
        struct[0].Gy_ini[31,29] = 5.75163398692810
        struct[0].Gy_ini[31,30] = 69.0196078431372
        struct[0].Gy_ini[31,31] = -1017.25490196078
        struct[0].Gy_ini[32,32] = -5.75163398692810
        struct[0].Gy_ini[32,33] = -23.0065359477124
        struct[0].Gy_ini[32,38] = 5.75163398692810
        struct[0].Gy_ini[32,39] = 23.0065359477124
        struct[0].Gy_ini[32,64] = 0.249053057297486
        struct[0].Gy_ini[32,65] = 0.996212229189942
        struct[0].Gy_ini[32,68] = -0.249053057297486
        struct[0].Gy_ini[32,69] = -0.996212229189942
        struct[0].Gy_ini[32,162] = 1
        struct[0].Gy_ini[33,32] = 23.0065359477124
        struct[0].Gy_ini[33,33] = -5.75163398692810
        struct[0].Gy_ini[33,38] = -23.0065359477124
        struct[0].Gy_ini[33,39] = 5.75163398692810
        struct[0].Gy_ini[33,64] = -0.996212229189942
        struct[0].Gy_ini[33,65] = 0.249053057297486
        struct[0].Gy_ini[33,68] = 0.996212229189942
        struct[0].Gy_ini[33,69] = -0.249053057297486
        struct[0].Gy_ini[33,163] = 1
        struct[0].Gy_ini[34,34] = -5.75163398692810
        struct[0].Gy_ini[34,35] = -23.0065359477124
        struct[0].Gy_ini[34,38] = 5.75163398692810
        struct[0].Gy_ini[34,39] = 23.0065359477124
        struct[0].Gy_ini[34,64] = -0.249053057297486
        struct[0].Gy_ini[34,65] = -0.996212229189942
        struct[0].Gy_ini[34,66] = 0.249053057297486
        struct[0].Gy_ini[34,67] = 0.996212229189942
        struct[0].Gy_ini[34,164] = 1
        struct[0].Gy_ini[35,34] = 23.0065359477124
        struct[0].Gy_ini[35,35] = -5.75163398692810
        struct[0].Gy_ini[35,38] = -23.0065359477124
        struct[0].Gy_ini[35,39] = 5.75163398692810
        struct[0].Gy_ini[35,64] = 0.996212229189942
        struct[0].Gy_ini[35,65] = -0.249053057297486
        struct[0].Gy_ini[35,66] = -0.996212229189942
        struct[0].Gy_ini[35,67] = 0.249053057297486
        struct[0].Gy_ini[35,165] = 1
        struct[0].Gy_ini[36,36] = -5.75163398692810
        struct[0].Gy_ini[36,37] = -23.0065359477124
        struct[0].Gy_ini[36,38] = 5.75163398692810
        struct[0].Gy_ini[36,39] = 23.0065359477124
        struct[0].Gy_ini[36,66] = -0.249053057297486
        struct[0].Gy_ini[36,67] = -0.996212229189942
        struct[0].Gy_ini[36,68] = 0.249053057297486
        struct[0].Gy_ini[36,69] = 0.996212229189942
        struct[0].Gy_ini[36,166] = 1
        struct[0].Gy_ini[37,36] = 23.0065359477124
        struct[0].Gy_ini[37,37] = -5.75163398692810
        struct[0].Gy_ini[37,38] = -23.0065359477124
        struct[0].Gy_ini[37,39] = 5.75163398692810
        struct[0].Gy_ini[37,66] = 0.996212229189942
        struct[0].Gy_ini[37,67] = -0.249053057297486
        struct[0].Gy_ini[37,68] = -0.996212229189942
        struct[0].Gy_ini[37,69] = 0.249053057297486
        struct[0].Gy_ini[37,167] = 1
        struct[0].Gy_ini[38,32] = 5.75163398692810
        struct[0].Gy_ini[38,33] = 23.0065359477124
        struct[0].Gy_ini[38,34] = 5.75163398692810
        struct[0].Gy_ini[38,35] = 23.0065359477124
        struct[0].Gy_ini[38,36] = 5.75163398692810
        struct[0].Gy_ini[38,37] = 23.0065359477124
        struct[0].Gy_ini[38,38] = -1017.25490196078
        struct[0].Gy_ini[38,39] = -69.0196078431372
        struct[0].Gy_ini[39,32] = -23.0065359477124
        struct[0].Gy_ini[39,33] = 5.75163398692810
        struct[0].Gy_ini[39,34] = -23.0065359477124
        struct[0].Gy_ini[39,35] = 5.75163398692810
        struct[0].Gy_ini[39,36] = -23.0065359477124
        struct[0].Gy_ini[39,37] = 5.75163398692810
        struct[0].Gy_ini[39,38] = 69.0196078431372
        struct[0].Gy_ini[39,39] = -1017.25490196078
        struct[0].Gy_ini[40,0] = 0.249053057297486
        struct[0].Gy_ini[40,1] = 0.996212229189942
        struct[0].Gy_ini[40,2] = -0.249053057297486
        struct[0].Gy_ini[40,3] = -0.996212229189942
        struct[0].Gy_ini[40,40] = -2.23667465123725
        struct[0].Gy_ini[40,41] = -1.28353302446119
        struct[0].Gy_ini[40,42] = 0.643671749092996
        struct[0].Gy_ini[40,43] = 0.385473430243205
        struct[0].Gy_ini[40,44] = 0.643671749092997
        struct[0].Gy_ini[40,45] = 0.385473430243205
        struct[0].Gy_ini[40,46] = 1.10755301189314
        struct[0].Gy_ini[40,47] = 0.598820527961361
        struct[0].Gy_ini[40,48] = -0.316443717683753
        struct[0].Gy_ini[40,49] = -0.171091579417532
        struct[0].Gy_ini[40,50] = -0.316443717683753
        struct[0].Gy_ini[40,51] = -0.171091579417532
        struct[0].Gy_ini[41,0] = -0.996212229189942
        struct[0].Gy_ini[41,1] = 0.249053057297486
        struct[0].Gy_ini[41,2] = 0.996212229189942
        struct[0].Gy_ini[41,3] = -0.249053057297486
        struct[0].Gy_ini[41,40] = 1.28353302446119
        struct[0].Gy_ini[41,41] = -2.23667465123725
        struct[0].Gy_ini[41,42] = -0.385473430243205
        struct[0].Gy_ini[41,43] = 0.643671749092996
        struct[0].Gy_ini[41,44] = -0.385473430243205
        struct[0].Gy_ini[41,45] = 0.643671749092997
        struct[0].Gy_ini[41,46] = -0.598820527961361
        struct[0].Gy_ini[41,47] = 1.10755301189314
        struct[0].Gy_ini[41,48] = 0.171091579417532
        struct[0].Gy_ini[41,49] = -0.316443717683753
        struct[0].Gy_ini[41,50] = 0.171091579417532
        struct[0].Gy_ini[41,51] = -0.316443717683753
        struct[0].Gy_ini[42,2] = 0.249053057297486
        struct[0].Gy_ini[42,3] = 0.996212229189942
        struct[0].Gy_ini[42,4] = -0.249053057297486
        struct[0].Gy_ini[42,5] = -0.996212229189942
        struct[0].Gy_ini[42,40] = 0.643671749092996
        struct[0].Gy_ini[42,41] = 0.385473430243205
        struct[0].Gy_ini[42,42] = -2.23667465123725
        struct[0].Gy_ini[42,43] = -1.28353302446119
        struct[0].Gy_ini[42,44] = 0.643671749092997
        struct[0].Gy_ini[42,45] = 0.385473430243204
        struct[0].Gy_ini[42,46] = -0.316443717683753
        struct[0].Gy_ini[42,47] = -0.171091579417532
        struct[0].Gy_ini[42,48] = 1.10755301189314
        struct[0].Gy_ini[42,49] = 0.598820527961360
        struct[0].Gy_ini[42,50] = -0.316443717683753
        struct[0].Gy_ini[42,51] = -0.171091579417531
        struct[0].Gy_ini[43,2] = -0.996212229189942
        struct[0].Gy_ini[43,3] = 0.249053057297486
        struct[0].Gy_ini[43,4] = 0.996212229189942
        struct[0].Gy_ini[43,5] = -0.249053057297486
        struct[0].Gy_ini[43,40] = -0.385473430243205
        struct[0].Gy_ini[43,41] = 0.643671749092996
        struct[0].Gy_ini[43,42] = 1.28353302446119
        struct[0].Gy_ini[43,43] = -2.23667465123725
        struct[0].Gy_ini[43,44] = -0.385473430243204
        struct[0].Gy_ini[43,45] = 0.643671749092997
        struct[0].Gy_ini[43,46] = 0.171091579417532
        struct[0].Gy_ini[43,47] = -0.316443717683753
        struct[0].Gy_ini[43,48] = -0.598820527961360
        struct[0].Gy_ini[43,49] = 1.10755301189314
        struct[0].Gy_ini[43,50] = 0.171091579417531
        struct[0].Gy_ini[43,51] = -0.316443717683753
        struct[0].Gy_ini[44,0] = -0.249053057297486
        struct[0].Gy_ini[44,1] = -0.996212229189942
        struct[0].Gy_ini[44,4] = 0.249053057297486
        struct[0].Gy_ini[44,5] = 0.996212229189942
        struct[0].Gy_ini[44,40] = 0.643671749092997
        struct[0].Gy_ini[44,41] = 0.385473430243205
        struct[0].Gy_ini[44,42] = 0.643671749092997
        struct[0].Gy_ini[44,43] = 0.385473430243204
        struct[0].Gy_ini[44,44] = -2.23667465123725
        struct[0].Gy_ini[44,45] = -1.28353302446119
        struct[0].Gy_ini[44,46] = -0.316443717683753
        struct[0].Gy_ini[44,47] = -0.171091579417532
        struct[0].Gy_ini[44,48] = -0.316443717683753
        struct[0].Gy_ini[44,49] = -0.171091579417531
        struct[0].Gy_ini[44,50] = 1.10755301189314
        struct[0].Gy_ini[44,51] = 0.598820527961360
        struct[0].Gy_ini[45,0] = 0.996212229189942
        struct[0].Gy_ini[45,1] = -0.249053057297486
        struct[0].Gy_ini[45,4] = -0.996212229189942
        struct[0].Gy_ini[45,5] = 0.249053057297486
        struct[0].Gy_ini[45,40] = -0.385473430243205
        struct[0].Gy_ini[45,41] = 0.643671749092997
        struct[0].Gy_ini[45,42] = -0.385473430243204
        struct[0].Gy_ini[45,43] = 0.643671749092997
        struct[0].Gy_ini[45,44] = 1.28353302446119
        struct[0].Gy_ini[45,45] = -2.23667465123725
        struct[0].Gy_ini[45,46] = 0.171091579417532
        struct[0].Gy_ini[45,47] = -0.316443717683753
        struct[0].Gy_ini[45,48] = 0.171091579417531
        struct[0].Gy_ini[45,49] = -0.316443717683753
        struct[0].Gy_ini[45,50] = -0.598820527961360
        struct[0].Gy_ini[45,51] = 1.10755301189314
        struct[0].Gy_ini[46,8] = 0.249053057297486
        struct[0].Gy_ini[46,9] = 0.996212229189942
        struct[0].Gy_ini[46,10] = -0.249053057297486
        struct[0].Gy_ini[46,11] = -0.996212229189942
        struct[0].Gy_ini[46,40] = 1.10755301189314
        struct[0].Gy_ini[46,41] = 0.598820527961361
        struct[0].Gy_ini[46,42] = -0.316443717683753
        struct[0].Gy_ini[46,43] = -0.171091579417532
        struct[0].Gy_ini[46,44] = -0.316443717683753
        struct[0].Gy_ini[46,45] = -0.171091579417532
        struct[0].Gy_ini[46,46] = -2.23667465123725
        struct[0].Gy_ini[46,47] = -1.28353302446119
        struct[0].Gy_ini[46,48] = 0.643671749092996
        struct[0].Gy_ini[46,49] = 0.385473430243205
        struct[0].Gy_ini[46,50] = 0.643671749092997
        struct[0].Gy_ini[46,51] = 0.385473430243205
        struct[0].Gy_ini[46,52] = 1.10755301189314
        struct[0].Gy_ini[46,53] = 0.598820527961361
        struct[0].Gy_ini[46,54] = -0.316443717683753
        struct[0].Gy_ini[46,55] = -0.171091579417532
        struct[0].Gy_ini[46,56] = -0.316443717683753
        struct[0].Gy_ini[46,57] = -0.171091579417532
        struct[0].Gy_ini[47,8] = -0.996212229189942
        struct[0].Gy_ini[47,9] = 0.249053057297486
        struct[0].Gy_ini[47,10] = 0.996212229189942
        struct[0].Gy_ini[47,11] = -0.249053057297486
        struct[0].Gy_ini[47,40] = -0.598820527961361
        struct[0].Gy_ini[47,41] = 1.10755301189314
        struct[0].Gy_ini[47,42] = 0.171091579417532
        struct[0].Gy_ini[47,43] = -0.316443717683753
        struct[0].Gy_ini[47,44] = 0.171091579417532
        struct[0].Gy_ini[47,45] = -0.316443717683753
        struct[0].Gy_ini[47,46] = 1.28353302446119
        struct[0].Gy_ini[47,47] = -2.23667465123725
        struct[0].Gy_ini[47,48] = -0.385473430243205
        struct[0].Gy_ini[47,49] = 0.643671749092996
        struct[0].Gy_ini[47,50] = -0.385473430243205
        struct[0].Gy_ini[47,51] = 0.643671749092997
        struct[0].Gy_ini[47,52] = -0.598820527961361
        struct[0].Gy_ini[47,53] = 1.10755301189314
        struct[0].Gy_ini[47,54] = 0.171091579417532
        struct[0].Gy_ini[47,55] = -0.316443717683753
        struct[0].Gy_ini[47,56] = 0.171091579417532
        struct[0].Gy_ini[47,57] = -0.316443717683753
        struct[0].Gy_ini[48,10] = 0.249053057297486
        struct[0].Gy_ini[48,11] = 0.996212229189942
        struct[0].Gy_ini[48,12] = -0.249053057297486
        struct[0].Gy_ini[48,13] = -0.996212229189942
        struct[0].Gy_ini[48,40] = -0.316443717683753
        struct[0].Gy_ini[48,41] = -0.171091579417532
        struct[0].Gy_ini[48,42] = 1.10755301189314
        struct[0].Gy_ini[48,43] = 0.598820527961360
        struct[0].Gy_ini[48,44] = -0.316443717683753
        struct[0].Gy_ini[48,45] = -0.171091579417531
        struct[0].Gy_ini[48,46] = 0.643671749092996
        struct[0].Gy_ini[48,47] = 0.385473430243205
        struct[0].Gy_ini[48,48] = -2.23667465123725
        struct[0].Gy_ini[48,49] = -1.28353302446119
        struct[0].Gy_ini[48,50] = 0.643671749092997
        struct[0].Gy_ini[48,51] = 0.385473430243204
        struct[0].Gy_ini[48,52] = -0.316443717683753
        struct[0].Gy_ini[48,53] = -0.171091579417532
        struct[0].Gy_ini[48,54] = 1.10755301189314
        struct[0].Gy_ini[48,55] = 0.598820527961360
        struct[0].Gy_ini[48,56] = -0.316443717683753
        struct[0].Gy_ini[48,57] = -0.171091579417531
        struct[0].Gy_ini[49,10] = -0.996212229189942
        struct[0].Gy_ini[49,11] = 0.249053057297486
        struct[0].Gy_ini[49,12] = 0.996212229189942
        struct[0].Gy_ini[49,13] = -0.249053057297486
        struct[0].Gy_ini[49,40] = 0.171091579417532
        struct[0].Gy_ini[49,41] = -0.316443717683753
        struct[0].Gy_ini[49,42] = -0.598820527961360
        struct[0].Gy_ini[49,43] = 1.10755301189314
        struct[0].Gy_ini[49,44] = 0.171091579417531
        struct[0].Gy_ini[49,45] = -0.316443717683753
        struct[0].Gy_ini[49,46] = -0.385473430243205
        struct[0].Gy_ini[49,47] = 0.643671749092996
        struct[0].Gy_ini[49,48] = 1.28353302446119
        struct[0].Gy_ini[49,49] = -2.23667465123725
        struct[0].Gy_ini[49,50] = -0.385473430243204
        struct[0].Gy_ini[49,51] = 0.643671749092997
        struct[0].Gy_ini[49,52] = 0.171091579417532
        struct[0].Gy_ini[49,53] = -0.316443717683753
        struct[0].Gy_ini[49,54] = -0.598820527961360
        struct[0].Gy_ini[49,55] = 1.10755301189314
        struct[0].Gy_ini[49,56] = 0.171091579417531
        struct[0].Gy_ini[49,57] = -0.316443717683753
        struct[0].Gy_ini[50,8] = -0.249053057297486
        struct[0].Gy_ini[50,9] = -0.996212229189942
        struct[0].Gy_ini[50,12] = 0.249053057297486
        struct[0].Gy_ini[50,13] = 0.996212229189942
        struct[0].Gy_ini[50,40] = -0.316443717683753
        struct[0].Gy_ini[50,41] = -0.171091579417532
        struct[0].Gy_ini[50,42] = -0.316443717683753
        struct[0].Gy_ini[50,43] = -0.171091579417531
        struct[0].Gy_ini[50,44] = 1.10755301189314
        struct[0].Gy_ini[50,45] = 0.598820527961360
        struct[0].Gy_ini[50,46] = 0.643671749092997
        struct[0].Gy_ini[50,47] = 0.385473430243205
        struct[0].Gy_ini[50,48] = 0.643671749092997
        struct[0].Gy_ini[50,49] = 0.385473430243204
        struct[0].Gy_ini[50,50] = -2.23667465123725
        struct[0].Gy_ini[50,51] = -1.28353302446119
        struct[0].Gy_ini[50,52] = -0.316443717683753
        struct[0].Gy_ini[50,53] = -0.171091579417532
        struct[0].Gy_ini[50,54] = -0.316443717683753
        struct[0].Gy_ini[50,55] = -0.171091579417531
        struct[0].Gy_ini[50,56] = 1.10755301189314
        struct[0].Gy_ini[50,57] = 0.598820527961360
        struct[0].Gy_ini[51,8] = 0.996212229189942
        struct[0].Gy_ini[51,9] = -0.249053057297486
        struct[0].Gy_ini[51,12] = -0.996212229189942
        struct[0].Gy_ini[51,13] = 0.249053057297486
        struct[0].Gy_ini[51,40] = 0.171091579417532
        struct[0].Gy_ini[51,41] = -0.316443717683753
        struct[0].Gy_ini[51,42] = 0.171091579417531
        struct[0].Gy_ini[51,43] = -0.316443717683753
        struct[0].Gy_ini[51,44] = -0.598820527961360
        struct[0].Gy_ini[51,45] = 1.10755301189314
        struct[0].Gy_ini[51,46] = -0.385473430243205
        struct[0].Gy_ini[51,47] = 0.643671749092997
        struct[0].Gy_ini[51,48] = -0.385473430243204
        struct[0].Gy_ini[51,49] = 0.643671749092997
        struct[0].Gy_ini[51,50] = 1.28353302446119
        struct[0].Gy_ini[51,51] = -2.23667465123725
        struct[0].Gy_ini[51,52] = 0.171091579417532
        struct[0].Gy_ini[51,53] = -0.316443717683753
        struct[0].Gy_ini[51,54] = 0.171091579417531
        struct[0].Gy_ini[51,55] = -0.316443717683753
        struct[0].Gy_ini[51,56] = -0.598820527961360
        struct[0].Gy_ini[51,57] = 1.10755301189314
        struct[0].Gy_ini[52,16] = 0.249053057297486
        struct[0].Gy_ini[52,17] = 0.996212229189942
        struct[0].Gy_ini[52,18] = -0.249053057297486
        struct[0].Gy_ini[52,19] = -0.996212229189942
        struct[0].Gy_ini[52,46] = 1.10755301189314
        struct[0].Gy_ini[52,47] = 0.598820527961361
        struct[0].Gy_ini[52,48] = -0.316443717683753
        struct[0].Gy_ini[52,49] = -0.171091579417532
        struct[0].Gy_ini[52,50] = -0.316443717683753
        struct[0].Gy_ini[52,51] = -0.171091579417532
        struct[0].Gy_ini[52,52] = -1.12912163934412
        struct[0].Gy_ini[52,53] = -0.684903767132556
        struct[0].Gy_ini[52,54] = 0.327228031409243
        struct[0].Gy_ini[52,55] = 0.214305342572583
        struct[0].Gy_ini[52,56] = 0.327228031409244
        struct[0].Gy_ini[52,57] = 0.214305342572583
        struct[0].Gy_ini[53,16] = -0.996212229189942
        struct[0].Gy_ini[53,17] = 0.249053057297486
        struct[0].Gy_ini[53,18] = 0.996212229189942
        struct[0].Gy_ini[53,19] = -0.249053057297486
        struct[0].Gy_ini[53,46] = -0.598820527961361
        struct[0].Gy_ini[53,47] = 1.10755301189314
        struct[0].Gy_ini[53,48] = 0.171091579417532
        struct[0].Gy_ini[53,49] = -0.316443717683753
        struct[0].Gy_ini[53,50] = 0.171091579417532
        struct[0].Gy_ini[53,51] = -0.316443717683753
        struct[0].Gy_ini[53,52] = 0.684903767132556
        struct[0].Gy_ini[53,53] = -1.12912163934412
        struct[0].Gy_ini[53,54] = -0.214305342572583
        struct[0].Gy_ini[53,55] = 0.327228031409243
        struct[0].Gy_ini[53,56] = -0.214305342572583
        struct[0].Gy_ini[53,57] = 0.327228031409244
        struct[0].Gy_ini[54,18] = 0.249053057297486
        struct[0].Gy_ini[54,19] = 0.996212229189942
        struct[0].Gy_ini[54,20] = -0.249053057297486
        struct[0].Gy_ini[54,21] = -0.996212229189942
        struct[0].Gy_ini[54,46] = -0.316443717683753
        struct[0].Gy_ini[54,47] = -0.171091579417532
        struct[0].Gy_ini[54,48] = 1.10755301189314
        struct[0].Gy_ini[54,49] = 0.598820527961360
        struct[0].Gy_ini[54,50] = -0.316443717683753
        struct[0].Gy_ini[54,51] = -0.171091579417531
        struct[0].Gy_ini[54,52] = 0.327228031409243
        struct[0].Gy_ini[54,53] = 0.214305342572583
        struct[0].Gy_ini[54,54] = -1.12912163934412
        struct[0].Gy_ini[54,55] = -0.684903767132556
        struct[0].Gy_ini[54,56] = 0.327228031409244
        struct[0].Gy_ini[54,57] = 0.214305342572582
        struct[0].Gy_ini[55,18] = -0.996212229189942
        struct[0].Gy_ini[55,19] = 0.249053057297486
        struct[0].Gy_ini[55,20] = 0.996212229189942
        struct[0].Gy_ini[55,21] = -0.249053057297486
        struct[0].Gy_ini[55,46] = 0.171091579417532
        struct[0].Gy_ini[55,47] = -0.316443717683753
        struct[0].Gy_ini[55,48] = -0.598820527961360
        struct[0].Gy_ini[55,49] = 1.10755301189314
        struct[0].Gy_ini[55,50] = 0.171091579417531
        struct[0].Gy_ini[55,51] = -0.316443717683753
        struct[0].Gy_ini[55,52] = -0.214305342572583
        struct[0].Gy_ini[55,53] = 0.327228031409243
        struct[0].Gy_ini[55,54] = 0.684903767132556
        struct[0].Gy_ini[55,55] = -1.12912163934412
        struct[0].Gy_ini[55,56] = -0.214305342572582
        struct[0].Gy_ini[55,57] = 0.327228031409244
        struct[0].Gy_ini[56,16] = -0.249053057297486
        struct[0].Gy_ini[56,17] = -0.996212229189942
        struct[0].Gy_ini[56,20] = 0.249053057297486
        struct[0].Gy_ini[56,21] = 0.996212229189942
        struct[0].Gy_ini[56,46] = -0.316443717683753
        struct[0].Gy_ini[56,47] = -0.171091579417532
        struct[0].Gy_ini[56,48] = -0.316443717683753
        struct[0].Gy_ini[56,49] = -0.171091579417531
        struct[0].Gy_ini[56,50] = 1.10755301189314
        struct[0].Gy_ini[56,51] = 0.598820527961360
        struct[0].Gy_ini[56,52] = 0.327228031409243
        struct[0].Gy_ini[56,53] = 0.214305342572583
        struct[0].Gy_ini[56,54] = 0.327228031409244
        struct[0].Gy_ini[56,55] = 0.214305342572582
        struct[0].Gy_ini[56,56] = -1.12912163934412
        struct[0].Gy_ini[56,57] = -0.684903767132556
        struct[0].Gy_ini[57,16] = 0.996212229189942
        struct[0].Gy_ini[57,17] = -0.249053057297486
        struct[0].Gy_ini[57,20] = -0.996212229189942
        struct[0].Gy_ini[57,21] = 0.249053057297486
        struct[0].Gy_ini[57,46] = 0.171091579417532
        struct[0].Gy_ini[57,47] = -0.316443717683753
        struct[0].Gy_ini[57,48] = 0.171091579417531
        struct[0].Gy_ini[57,49] = -0.316443717683753
        struct[0].Gy_ini[57,50] = -0.598820527961360
        struct[0].Gy_ini[57,51] = 1.10755301189314
        struct[0].Gy_ini[57,52] = -0.214305342572583
        struct[0].Gy_ini[57,53] = 0.327228031409243
        struct[0].Gy_ini[57,54] = -0.214305342572582
        struct[0].Gy_ini[57,55] = 0.327228031409244
        struct[0].Gy_ini[57,56] = 0.684903767132556
        struct[0].Gy_ini[57,57] = -1.12912163934412
        struct[0].Gy_ini[58,24] = 0.249053057297486
        struct[0].Gy_ini[58,25] = 0.996212229189942
        struct[0].Gy_ini[58,26] = -0.249053057297486
        struct[0].Gy_ini[58,27] = -0.996212229189942
        struct[0].Gy_ini[58,58] = -1.12912163934412
        struct[0].Gy_ini[58,59] = -0.684903767132556
        struct[0].Gy_ini[58,60] = 0.327228031409243
        struct[0].Gy_ini[58,61] = 0.214305342572583
        struct[0].Gy_ini[58,62] = 0.327228031409244
        struct[0].Gy_ini[58,63] = 0.214305342572583
        struct[0].Gy_ini[58,64] = 1.10755301189314
        struct[0].Gy_ini[58,65] = 0.598820527961361
        struct[0].Gy_ini[58,66] = -0.316443717683753
        struct[0].Gy_ini[58,67] = -0.171091579417532
        struct[0].Gy_ini[58,68] = -0.316443717683753
        struct[0].Gy_ini[58,69] = -0.171091579417532
        struct[0].Gy_ini[59,24] = -0.996212229189942
        struct[0].Gy_ini[59,25] = 0.249053057297486
        struct[0].Gy_ini[59,26] = 0.996212229189942
        struct[0].Gy_ini[59,27] = -0.249053057297486
        struct[0].Gy_ini[59,58] = 0.684903767132556
        struct[0].Gy_ini[59,59] = -1.12912163934412
        struct[0].Gy_ini[59,60] = -0.214305342572583
        struct[0].Gy_ini[59,61] = 0.327228031409243
        struct[0].Gy_ini[59,62] = -0.214305342572583
        struct[0].Gy_ini[59,63] = 0.327228031409244
        struct[0].Gy_ini[59,64] = -0.598820527961361
        struct[0].Gy_ini[59,65] = 1.10755301189314
        struct[0].Gy_ini[59,66] = 0.171091579417532
        struct[0].Gy_ini[59,67] = -0.316443717683753
        struct[0].Gy_ini[59,68] = 0.171091579417532
        struct[0].Gy_ini[59,69] = -0.316443717683753
        struct[0].Gy_ini[60,26] = 0.249053057297486
        struct[0].Gy_ini[60,27] = 0.996212229189942
        struct[0].Gy_ini[60,28] = -0.249053057297486
        struct[0].Gy_ini[60,29] = -0.996212229189942
        struct[0].Gy_ini[60,58] = 0.327228031409243
        struct[0].Gy_ini[60,59] = 0.214305342572583
        struct[0].Gy_ini[60,60] = -1.12912163934412
        struct[0].Gy_ini[60,61] = -0.684903767132556
        struct[0].Gy_ini[60,62] = 0.327228031409244
        struct[0].Gy_ini[60,63] = 0.214305342572582
        struct[0].Gy_ini[60,64] = -0.316443717683753
        struct[0].Gy_ini[60,65] = -0.171091579417532
        struct[0].Gy_ini[60,66] = 1.10755301189314
        struct[0].Gy_ini[60,67] = 0.598820527961360
        struct[0].Gy_ini[60,68] = -0.316443717683753
        struct[0].Gy_ini[60,69] = -0.171091579417531
        struct[0].Gy_ini[61,26] = -0.996212229189942
        struct[0].Gy_ini[61,27] = 0.249053057297486
        struct[0].Gy_ini[61,28] = 0.996212229189942
        struct[0].Gy_ini[61,29] = -0.249053057297486
        struct[0].Gy_ini[61,58] = -0.214305342572583
        struct[0].Gy_ini[61,59] = 0.327228031409243
        struct[0].Gy_ini[61,60] = 0.684903767132556
        struct[0].Gy_ini[61,61] = -1.12912163934412
        struct[0].Gy_ini[61,62] = -0.214305342572582
        struct[0].Gy_ini[61,63] = 0.327228031409244
        struct[0].Gy_ini[61,64] = 0.171091579417532
        struct[0].Gy_ini[61,65] = -0.316443717683753
        struct[0].Gy_ini[61,66] = -0.598820527961360
        struct[0].Gy_ini[61,67] = 1.10755301189314
        struct[0].Gy_ini[61,68] = 0.171091579417531
        struct[0].Gy_ini[61,69] = -0.316443717683753
        struct[0].Gy_ini[62,24] = -0.249053057297486
        struct[0].Gy_ini[62,25] = -0.996212229189942
        struct[0].Gy_ini[62,28] = 0.249053057297486
        struct[0].Gy_ini[62,29] = 0.996212229189942
        struct[0].Gy_ini[62,58] = 0.327228031409243
        struct[0].Gy_ini[62,59] = 0.214305342572583
        struct[0].Gy_ini[62,60] = 0.327228031409244
        struct[0].Gy_ini[62,61] = 0.214305342572582
        struct[0].Gy_ini[62,62] = -1.12912163934412
        struct[0].Gy_ini[62,63] = -0.684903767132556
        struct[0].Gy_ini[62,64] = -0.316443717683753
        struct[0].Gy_ini[62,65] = -0.171091579417532
        struct[0].Gy_ini[62,66] = -0.316443717683753
        struct[0].Gy_ini[62,67] = -0.171091579417531
        struct[0].Gy_ini[62,68] = 1.10755301189314
        struct[0].Gy_ini[62,69] = 0.598820527961360
        struct[0].Gy_ini[63,24] = 0.996212229189942
        struct[0].Gy_ini[63,25] = -0.249053057297486
        struct[0].Gy_ini[63,28] = -0.996212229189942
        struct[0].Gy_ini[63,29] = 0.249053057297486
        struct[0].Gy_ini[63,58] = -0.214305342572583
        struct[0].Gy_ini[63,59] = 0.327228031409243
        struct[0].Gy_ini[63,60] = -0.214305342572582
        struct[0].Gy_ini[63,61] = 0.327228031409244
        struct[0].Gy_ini[63,62] = 0.684903767132556
        struct[0].Gy_ini[63,63] = -1.12912163934412
        struct[0].Gy_ini[63,64] = 0.171091579417532
        struct[0].Gy_ini[63,65] = -0.316443717683753
        struct[0].Gy_ini[63,66] = 0.171091579417531
        struct[0].Gy_ini[63,67] = -0.316443717683753
        struct[0].Gy_ini[63,68] = -0.598820527961360
        struct[0].Gy_ini[63,69] = 1.10755301189314
        struct[0].Gy_ini[64,32] = 0.249053057297486
        struct[0].Gy_ini[64,33] = 0.996212229189942
        struct[0].Gy_ini[64,34] = -0.249053057297486
        struct[0].Gy_ini[64,35] = -0.996212229189942
        struct[0].Gy_ini[64,58] = 1.10755301189314
        struct[0].Gy_ini[64,59] = 0.598820527961361
        struct[0].Gy_ini[64,60] = -0.316443717683753
        struct[0].Gy_ini[64,61] = -0.171091579417532
        struct[0].Gy_ini[64,62] = -0.316443717683753
        struct[0].Gy_ini[64,63] = -0.171091579417532
        struct[0].Gy_ini[64,64] = -2.23667465123725
        struct[0].Gy_ini[64,65] = -1.28353302446119
        struct[0].Gy_ini[64,66] = 0.643671749092996
        struct[0].Gy_ini[64,67] = 0.385473430243205
        struct[0].Gy_ini[64,68] = 0.643671749092997
        struct[0].Gy_ini[64,69] = 0.385473430243205
        struct[0].Gy_ini[65,32] = -0.996212229189942
        struct[0].Gy_ini[65,33] = 0.249053057297486
        struct[0].Gy_ini[65,34] = 0.996212229189942
        struct[0].Gy_ini[65,35] = -0.249053057297486
        struct[0].Gy_ini[65,58] = -0.598820527961361
        struct[0].Gy_ini[65,59] = 1.10755301189314
        struct[0].Gy_ini[65,60] = 0.171091579417532
        struct[0].Gy_ini[65,61] = -0.316443717683753
        struct[0].Gy_ini[65,62] = 0.171091579417532
        struct[0].Gy_ini[65,63] = -0.316443717683753
        struct[0].Gy_ini[65,64] = 1.28353302446119
        struct[0].Gy_ini[65,65] = -2.23667465123725
        struct[0].Gy_ini[65,66] = -0.385473430243205
        struct[0].Gy_ini[65,67] = 0.643671749092996
        struct[0].Gy_ini[65,68] = -0.385473430243205
        struct[0].Gy_ini[65,69] = 0.643671749092997
        struct[0].Gy_ini[66,34] = 0.249053057297486
        struct[0].Gy_ini[66,35] = 0.996212229189942
        struct[0].Gy_ini[66,36] = -0.249053057297486
        struct[0].Gy_ini[66,37] = -0.996212229189942
        struct[0].Gy_ini[66,58] = -0.316443717683753
        struct[0].Gy_ini[66,59] = -0.171091579417532
        struct[0].Gy_ini[66,60] = 1.10755301189314
        struct[0].Gy_ini[66,61] = 0.598820527961360
        struct[0].Gy_ini[66,62] = -0.316443717683753
        struct[0].Gy_ini[66,63] = -0.171091579417531
        struct[0].Gy_ini[66,64] = 0.643671749092996
        struct[0].Gy_ini[66,65] = 0.385473430243205
        struct[0].Gy_ini[66,66] = -2.23667465123725
        struct[0].Gy_ini[66,67] = -1.28353302446119
        struct[0].Gy_ini[66,68] = 0.643671749092997
        struct[0].Gy_ini[66,69] = 0.385473430243204
        struct[0].Gy_ini[67,34] = -0.996212229189942
        struct[0].Gy_ini[67,35] = 0.249053057297486
        struct[0].Gy_ini[67,36] = 0.996212229189942
        struct[0].Gy_ini[67,37] = -0.249053057297486
        struct[0].Gy_ini[67,58] = 0.171091579417532
        struct[0].Gy_ini[67,59] = -0.316443717683753
        struct[0].Gy_ini[67,60] = -0.598820527961360
        struct[0].Gy_ini[67,61] = 1.10755301189314
        struct[0].Gy_ini[67,62] = 0.171091579417531
        struct[0].Gy_ini[67,63] = -0.316443717683753
        struct[0].Gy_ini[67,64] = -0.385473430243205
        struct[0].Gy_ini[67,65] = 0.643671749092996
        struct[0].Gy_ini[67,66] = 1.28353302446119
        struct[0].Gy_ini[67,67] = -2.23667465123725
        struct[0].Gy_ini[67,68] = -0.385473430243204
        struct[0].Gy_ini[67,69] = 0.643671749092997
        struct[0].Gy_ini[68,32] = -0.249053057297486
        struct[0].Gy_ini[68,33] = -0.996212229189942
        struct[0].Gy_ini[68,36] = 0.249053057297486
        struct[0].Gy_ini[68,37] = 0.996212229189942
        struct[0].Gy_ini[68,58] = -0.316443717683753
        struct[0].Gy_ini[68,59] = -0.171091579417532
        struct[0].Gy_ini[68,60] = -0.316443717683753
        struct[0].Gy_ini[68,61] = -0.171091579417531
        struct[0].Gy_ini[68,62] = 1.10755301189314
        struct[0].Gy_ini[68,63] = 0.598820527961360
        struct[0].Gy_ini[68,64] = 0.643671749092997
        struct[0].Gy_ini[68,65] = 0.385473430243205
        struct[0].Gy_ini[68,66] = 0.643671749092997
        struct[0].Gy_ini[68,67] = 0.385473430243204
        struct[0].Gy_ini[68,68] = -2.23667465123725
        struct[0].Gy_ini[68,69] = -1.28353302446119
        struct[0].Gy_ini[69,32] = 0.996212229189942
        struct[0].Gy_ini[69,33] = -0.249053057297486
        struct[0].Gy_ini[69,36] = -0.996212229189942
        struct[0].Gy_ini[69,37] = 0.249053057297486
        struct[0].Gy_ini[69,58] = 0.171091579417532
        struct[0].Gy_ini[69,59] = -0.316443717683753
        struct[0].Gy_ini[69,60] = 0.171091579417531
        struct[0].Gy_ini[69,61] = -0.316443717683753
        struct[0].Gy_ini[69,62] = -0.598820527961360
        struct[0].Gy_ini[69,63] = 1.10755301189314
        struct[0].Gy_ini[69,64] = -0.385473430243205
        struct[0].Gy_ini[69,65] = 0.643671749092997
        struct[0].Gy_ini[69,66] = -0.385473430243204
        struct[0].Gy_ini[69,67] = 0.643671749092997
        struct[0].Gy_ini[69,68] = 1.28353302446119
        struct[0].Gy_ini[69,69] = -2.23667465123725
        struct[0].Gy_ini[70,0] = -0.249053057297486
        struct[0].Gy_ini[70,1] = -0.996212229189942
        struct[0].Gy_ini[70,2] = 0.249053057297486
        struct[0].Gy_ini[70,3] = 0.996212229189942
        struct[0].Gy_ini[70,40] = 0.0215686274509804
        struct[0].Gy_ini[70,41] = 0.0862745098039216
        struct[0].Gy_ini[70,42] = -0.0107843137254902
        struct[0].Gy_ini[70,43] = -0.0431372549019608
        struct[0].Gy_ini[70,44] = -0.0107843137254902
        struct[0].Gy_ini[70,45] = -0.0431372549019608
        struct[0].Gy_ini[70,70] = -1
        struct[0].Gy_ini[71,0] = 0.996212229189942
        struct[0].Gy_ini[71,1] = -0.249053057297486
        struct[0].Gy_ini[71,2] = -0.996212229189942
        struct[0].Gy_ini[71,3] = 0.249053057297486
        struct[0].Gy_ini[71,40] = -0.0862745098039216
        struct[0].Gy_ini[71,41] = 0.0215686274509804
        struct[0].Gy_ini[71,42] = 0.0431372549019608
        struct[0].Gy_ini[71,43] = -0.0107843137254902
        struct[0].Gy_ini[71,44] = 0.0431372549019608
        struct[0].Gy_ini[71,45] = -0.0107843137254902
        struct[0].Gy_ini[71,71] = -1
        struct[0].Gy_ini[72,2] = -0.249053057297486
        struct[0].Gy_ini[72,3] = -0.996212229189942
        struct[0].Gy_ini[72,4] = 0.249053057297486
        struct[0].Gy_ini[72,5] = 0.996212229189942
        struct[0].Gy_ini[72,40] = -0.0107843137254902
        struct[0].Gy_ini[72,41] = -0.0431372549019608
        struct[0].Gy_ini[72,42] = 0.0215686274509804
        struct[0].Gy_ini[72,43] = 0.0862745098039216
        struct[0].Gy_ini[72,44] = -0.0107843137254902
        struct[0].Gy_ini[72,45] = -0.0431372549019608
        struct[0].Gy_ini[72,72] = -1
        struct[0].Gy_ini[73,2] = 0.996212229189942
        struct[0].Gy_ini[73,3] = -0.249053057297486
        struct[0].Gy_ini[73,4] = -0.996212229189942
        struct[0].Gy_ini[73,5] = 0.249053057297486
        struct[0].Gy_ini[73,40] = 0.0431372549019608
        struct[0].Gy_ini[73,41] = -0.0107843137254902
        struct[0].Gy_ini[73,42] = -0.0862745098039216
        struct[0].Gy_ini[73,43] = 0.0215686274509804
        struct[0].Gy_ini[73,44] = 0.0431372549019608
        struct[0].Gy_ini[73,45] = -0.0107843137254902
        struct[0].Gy_ini[73,73] = -1
        struct[0].Gy_ini[74,0] = 0.249053057297486
        struct[0].Gy_ini[74,1] = 0.996212229189942
        struct[0].Gy_ini[74,4] = -0.249053057297486
        struct[0].Gy_ini[74,5] = -0.996212229189942
        struct[0].Gy_ini[74,40] = -0.0107843137254902
        struct[0].Gy_ini[74,41] = -0.0431372549019608
        struct[0].Gy_ini[74,42] = -0.0107843137254902
        struct[0].Gy_ini[74,43] = -0.0431372549019608
        struct[0].Gy_ini[74,44] = 0.0215686274509804
        struct[0].Gy_ini[74,45] = 0.0862745098039216
        struct[0].Gy_ini[74,74] = -1
        struct[0].Gy_ini[75,0] = -0.996212229189942
        struct[0].Gy_ini[75,1] = 0.249053057297486
        struct[0].Gy_ini[75,4] = 0.996212229189942
        struct[0].Gy_ini[75,5] = -0.249053057297486
        struct[0].Gy_ini[75,40] = 0.0431372549019608
        struct[0].Gy_ini[75,41] = -0.0107843137254902
        struct[0].Gy_ini[75,42] = 0.0431372549019608
        struct[0].Gy_ini[75,43] = -0.0107843137254902
        struct[0].Gy_ini[75,44] = -0.0862745098039216
        struct[0].Gy_ini[75,45] = 0.0215686274509804
        struct[0].Gy_ini[75,75] = -1
        struct[0].Gy_ini[76,8] = -0.249053057297486
        struct[0].Gy_ini[76,9] = -0.996212229189942
        struct[0].Gy_ini[76,10] = 0.249053057297486
        struct[0].Gy_ini[76,11] = 0.996212229189942
        struct[0].Gy_ini[76,46] = 0.0215686274509804
        struct[0].Gy_ini[76,47] = 0.0862745098039216
        struct[0].Gy_ini[76,48] = -0.0107843137254902
        struct[0].Gy_ini[76,49] = -0.0431372549019608
        struct[0].Gy_ini[76,50] = -0.0107843137254902
        struct[0].Gy_ini[76,51] = -0.0431372549019608
        struct[0].Gy_ini[76,76] = -1
        struct[0].Gy_ini[77,8] = 0.996212229189942
        struct[0].Gy_ini[77,9] = -0.249053057297486
        struct[0].Gy_ini[77,10] = -0.996212229189942
        struct[0].Gy_ini[77,11] = 0.249053057297486
        struct[0].Gy_ini[77,46] = -0.0862745098039216
        struct[0].Gy_ini[77,47] = 0.0215686274509804
        struct[0].Gy_ini[77,48] = 0.0431372549019608
        struct[0].Gy_ini[77,49] = -0.0107843137254902
        struct[0].Gy_ini[77,50] = 0.0431372549019608
        struct[0].Gy_ini[77,51] = -0.0107843137254902
        struct[0].Gy_ini[77,77] = -1
        struct[0].Gy_ini[78,10] = -0.249053057297486
        struct[0].Gy_ini[78,11] = -0.996212229189942
        struct[0].Gy_ini[78,12] = 0.249053057297486
        struct[0].Gy_ini[78,13] = 0.996212229189942
        struct[0].Gy_ini[78,46] = -0.0107843137254902
        struct[0].Gy_ini[78,47] = -0.0431372549019608
        struct[0].Gy_ini[78,48] = 0.0215686274509804
        struct[0].Gy_ini[78,49] = 0.0862745098039216
        struct[0].Gy_ini[78,50] = -0.0107843137254902
        struct[0].Gy_ini[78,51] = -0.0431372549019608
        struct[0].Gy_ini[78,78] = -1
        struct[0].Gy_ini[79,10] = 0.996212229189942
        struct[0].Gy_ini[79,11] = -0.249053057297486
        struct[0].Gy_ini[79,12] = -0.996212229189942
        struct[0].Gy_ini[79,13] = 0.249053057297486
        struct[0].Gy_ini[79,46] = 0.0431372549019608
        struct[0].Gy_ini[79,47] = -0.0107843137254902
        struct[0].Gy_ini[79,48] = -0.0862745098039216
        struct[0].Gy_ini[79,49] = 0.0215686274509804
        struct[0].Gy_ini[79,50] = 0.0431372549019608
        struct[0].Gy_ini[79,51] = -0.0107843137254902
        struct[0].Gy_ini[79,79] = -1
        struct[0].Gy_ini[80,8] = 0.249053057297486
        struct[0].Gy_ini[80,9] = 0.996212229189942
        struct[0].Gy_ini[80,12] = -0.249053057297486
        struct[0].Gy_ini[80,13] = -0.996212229189942
        struct[0].Gy_ini[80,46] = -0.0107843137254902
        struct[0].Gy_ini[80,47] = -0.0431372549019608
        struct[0].Gy_ini[80,48] = -0.0107843137254902
        struct[0].Gy_ini[80,49] = -0.0431372549019608
        struct[0].Gy_ini[80,50] = 0.0215686274509804
        struct[0].Gy_ini[80,51] = 0.0862745098039216
        struct[0].Gy_ini[80,80] = -1
        struct[0].Gy_ini[81,8] = -0.996212229189942
        struct[0].Gy_ini[81,9] = 0.249053057297486
        struct[0].Gy_ini[81,12] = 0.996212229189942
        struct[0].Gy_ini[81,13] = -0.249053057297486
        struct[0].Gy_ini[81,46] = 0.0431372549019608
        struct[0].Gy_ini[81,47] = -0.0107843137254902
        struct[0].Gy_ini[81,48] = 0.0431372549019608
        struct[0].Gy_ini[81,49] = -0.0107843137254902
        struct[0].Gy_ini[81,50] = -0.0862745098039216
        struct[0].Gy_ini[81,51] = 0.0215686274509804
        struct[0].Gy_ini[81,81] = -1
        struct[0].Gy_ini[82,16] = -0.249053057297486
        struct[0].Gy_ini[82,17] = -0.996212229189942
        struct[0].Gy_ini[82,18] = 0.249053057297486
        struct[0].Gy_ini[82,19] = 0.996212229189942
        struct[0].Gy_ini[82,52] = 0.0215686274509804
        struct[0].Gy_ini[82,53] = 0.0862745098039216
        struct[0].Gy_ini[82,54] = -0.0107843137254902
        struct[0].Gy_ini[82,55] = -0.0431372549019608
        struct[0].Gy_ini[82,56] = -0.0107843137254902
        struct[0].Gy_ini[82,57] = -0.0431372549019608
        struct[0].Gy_ini[82,82] = -1
        struct[0].Gy_ini[83,16] = 0.996212229189942
        struct[0].Gy_ini[83,17] = -0.249053057297486
        struct[0].Gy_ini[83,18] = -0.996212229189942
        struct[0].Gy_ini[83,19] = 0.249053057297486
        struct[0].Gy_ini[83,52] = -0.0862745098039216
        struct[0].Gy_ini[83,53] = 0.0215686274509804
        struct[0].Gy_ini[83,54] = 0.0431372549019608
        struct[0].Gy_ini[83,55] = -0.0107843137254902
        struct[0].Gy_ini[83,56] = 0.0431372549019608
        struct[0].Gy_ini[83,57] = -0.0107843137254902
        struct[0].Gy_ini[83,83] = -1
        struct[0].Gy_ini[84,18] = -0.249053057297486
        struct[0].Gy_ini[84,19] = -0.996212229189942
        struct[0].Gy_ini[84,20] = 0.249053057297486
        struct[0].Gy_ini[84,21] = 0.996212229189942
        struct[0].Gy_ini[84,52] = -0.0107843137254902
        struct[0].Gy_ini[84,53] = -0.0431372549019608
        struct[0].Gy_ini[84,54] = 0.0215686274509804
        struct[0].Gy_ini[84,55] = 0.0862745098039216
        struct[0].Gy_ini[84,56] = -0.0107843137254902
        struct[0].Gy_ini[84,57] = -0.0431372549019608
        struct[0].Gy_ini[84,84] = -1
        struct[0].Gy_ini[85,18] = 0.996212229189942
        struct[0].Gy_ini[85,19] = -0.249053057297486
        struct[0].Gy_ini[85,20] = -0.996212229189942
        struct[0].Gy_ini[85,21] = 0.249053057297486
        struct[0].Gy_ini[85,52] = 0.0431372549019608
        struct[0].Gy_ini[85,53] = -0.0107843137254902
        struct[0].Gy_ini[85,54] = -0.0862745098039216
        struct[0].Gy_ini[85,55] = 0.0215686274509804
        struct[0].Gy_ini[85,56] = 0.0431372549019608
        struct[0].Gy_ini[85,57] = -0.0107843137254902
        struct[0].Gy_ini[85,85] = -1
        struct[0].Gy_ini[86,16] = 0.249053057297486
        struct[0].Gy_ini[86,17] = 0.996212229189942
        struct[0].Gy_ini[86,20] = -0.249053057297486
        struct[0].Gy_ini[86,21] = -0.996212229189942
        struct[0].Gy_ini[86,52] = -0.0107843137254902
        struct[0].Gy_ini[86,53] = -0.0431372549019608
        struct[0].Gy_ini[86,54] = -0.0107843137254902
        struct[0].Gy_ini[86,55] = -0.0431372549019608
        struct[0].Gy_ini[86,56] = 0.0215686274509804
        struct[0].Gy_ini[86,57] = 0.0862745098039216
        struct[0].Gy_ini[86,86] = -1
        struct[0].Gy_ini[87,16] = -0.996212229189942
        struct[0].Gy_ini[87,17] = 0.249053057297486
        struct[0].Gy_ini[87,20] = 0.996212229189942
        struct[0].Gy_ini[87,21] = -0.249053057297486
        struct[0].Gy_ini[87,52] = 0.0431372549019608
        struct[0].Gy_ini[87,53] = -0.0107843137254902
        struct[0].Gy_ini[87,54] = 0.0431372549019608
        struct[0].Gy_ini[87,55] = -0.0107843137254902
        struct[0].Gy_ini[87,56] = -0.0862745098039216
        struct[0].Gy_ini[87,57] = 0.0215686274509804
        struct[0].Gy_ini[87,87] = -1
        struct[0].Gy_ini[88,24] = -0.249053057297486
        struct[0].Gy_ini[88,25] = -0.996212229189942
        struct[0].Gy_ini[88,26] = 0.249053057297486
        struct[0].Gy_ini[88,27] = 0.996212229189942
        struct[0].Gy_ini[88,58] = 0.0215686274509804
        struct[0].Gy_ini[88,59] = 0.0862745098039216
        struct[0].Gy_ini[88,60] = -0.0107843137254902
        struct[0].Gy_ini[88,61] = -0.0431372549019608
        struct[0].Gy_ini[88,62] = -0.0107843137254902
        struct[0].Gy_ini[88,63] = -0.0431372549019608
        struct[0].Gy_ini[88,88] = -1
        struct[0].Gy_ini[89,24] = 0.996212229189942
        struct[0].Gy_ini[89,25] = -0.249053057297486
        struct[0].Gy_ini[89,26] = -0.996212229189942
        struct[0].Gy_ini[89,27] = 0.249053057297486
        struct[0].Gy_ini[89,58] = -0.0862745098039216
        struct[0].Gy_ini[89,59] = 0.0215686274509804
        struct[0].Gy_ini[89,60] = 0.0431372549019608
        struct[0].Gy_ini[89,61] = -0.0107843137254902
        struct[0].Gy_ini[89,62] = 0.0431372549019608
        struct[0].Gy_ini[89,63] = -0.0107843137254902
        struct[0].Gy_ini[89,89] = -1
        struct[0].Gy_ini[90,26] = -0.249053057297486
        struct[0].Gy_ini[90,27] = -0.996212229189942
        struct[0].Gy_ini[90,28] = 0.249053057297486
        struct[0].Gy_ini[90,29] = 0.996212229189942
        struct[0].Gy_ini[90,58] = -0.0107843137254902
        struct[0].Gy_ini[90,59] = -0.0431372549019608
        struct[0].Gy_ini[90,60] = 0.0215686274509804
        struct[0].Gy_ini[90,61] = 0.0862745098039216
        struct[0].Gy_ini[90,62] = -0.0107843137254902
        struct[0].Gy_ini[90,63] = -0.0431372549019608
        struct[0].Gy_ini[90,90] = -1
        struct[0].Gy_ini[91,26] = 0.996212229189942
        struct[0].Gy_ini[91,27] = -0.249053057297486
        struct[0].Gy_ini[91,28] = -0.996212229189942
        struct[0].Gy_ini[91,29] = 0.249053057297486
        struct[0].Gy_ini[91,58] = 0.0431372549019608
        struct[0].Gy_ini[91,59] = -0.0107843137254902
        struct[0].Gy_ini[91,60] = -0.0862745098039216
        struct[0].Gy_ini[91,61] = 0.0215686274509804
        struct[0].Gy_ini[91,62] = 0.0431372549019608
        struct[0].Gy_ini[91,63] = -0.0107843137254902
        struct[0].Gy_ini[91,91] = -1
        struct[0].Gy_ini[92,24] = 0.249053057297486
        struct[0].Gy_ini[92,25] = 0.996212229189942
        struct[0].Gy_ini[92,28] = -0.249053057297486
        struct[0].Gy_ini[92,29] = -0.996212229189942
        struct[0].Gy_ini[92,58] = -0.0107843137254902
        struct[0].Gy_ini[92,59] = -0.0431372549019608
        struct[0].Gy_ini[92,60] = -0.0107843137254902
        struct[0].Gy_ini[92,61] = -0.0431372549019608
        struct[0].Gy_ini[92,62] = 0.0215686274509804
        struct[0].Gy_ini[92,63] = 0.0862745098039216
        struct[0].Gy_ini[92,92] = -1
        struct[0].Gy_ini[93,24] = -0.996212229189942
        struct[0].Gy_ini[93,25] = 0.249053057297486
        struct[0].Gy_ini[93,28] = 0.996212229189942
        struct[0].Gy_ini[93,29] = -0.249053057297486
        struct[0].Gy_ini[93,58] = 0.0431372549019608
        struct[0].Gy_ini[93,59] = -0.0107843137254902
        struct[0].Gy_ini[93,60] = 0.0431372549019608
        struct[0].Gy_ini[93,61] = -0.0107843137254902
        struct[0].Gy_ini[93,62] = -0.0862745098039216
        struct[0].Gy_ini[93,63] = 0.0215686274509804
        struct[0].Gy_ini[93,93] = -1
        struct[0].Gy_ini[94,32] = -0.249053057297486
        struct[0].Gy_ini[94,33] = -0.996212229189942
        struct[0].Gy_ini[94,34] = 0.249053057297486
        struct[0].Gy_ini[94,35] = 0.996212229189942
        struct[0].Gy_ini[94,64] = 0.0215686274509804
        struct[0].Gy_ini[94,65] = 0.0862745098039216
        struct[0].Gy_ini[94,66] = -0.0107843137254902
        struct[0].Gy_ini[94,67] = -0.0431372549019608
        struct[0].Gy_ini[94,68] = -0.0107843137254902
        struct[0].Gy_ini[94,69] = -0.0431372549019608
        struct[0].Gy_ini[94,94] = -1
        struct[0].Gy_ini[95,32] = 0.996212229189942
        struct[0].Gy_ini[95,33] = -0.249053057297486
        struct[0].Gy_ini[95,34] = -0.996212229189942
        struct[0].Gy_ini[95,35] = 0.249053057297486
        struct[0].Gy_ini[95,64] = -0.0862745098039216
        struct[0].Gy_ini[95,65] = 0.0215686274509804
        struct[0].Gy_ini[95,66] = 0.0431372549019608
        struct[0].Gy_ini[95,67] = -0.0107843137254902
        struct[0].Gy_ini[95,68] = 0.0431372549019608
        struct[0].Gy_ini[95,69] = -0.0107843137254902
        struct[0].Gy_ini[95,95] = -1
        struct[0].Gy_ini[96,34] = -0.249053057297486
        struct[0].Gy_ini[96,35] = -0.996212229189942
        struct[0].Gy_ini[96,36] = 0.249053057297486
        struct[0].Gy_ini[96,37] = 0.996212229189942
        struct[0].Gy_ini[96,64] = -0.0107843137254902
        struct[0].Gy_ini[96,65] = -0.0431372549019608
        struct[0].Gy_ini[96,66] = 0.0215686274509804
        struct[0].Gy_ini[96,67] = 0.0862745098039216
        struct[0].Gy_ini[96,68] = -0.0107843137254902
        struct[0].Gy_ini[96,69] = -0.0431372549019608
        struct[0].Gy_ini[96,96] = -1
        struct[0].Gy_ini[97,34] = 0.996212229189942
        struct[0].Gy_ini[97,35] = -0.249053057297486
        struct[0].Gy_ini[97,36] = -0.996212229189942
        struct[0].Gy_ini[97,37] = 0.249053057297486
        struct[0].Gy_ini[97,64] = 0.0431372549019608
        struct[0].Gy_ini[97,65] = -0.0107843137254902
        struct[0].Gy_ini[97,66] = -0.0862745098039216
        struct[0].Gy_ini[97,67] = 0.0215686274509804
        struct[0].Gy_ini[97,68] = 0.0431372549019608
        struct[0].Gy_ini[97,69] = -0.0107843137254902
        struct[0].Gy_ini[97,97] = -1
        struct[0].Gy_ini[98,32] = 0.249053057297486
        struct[0].Gy_ini[98,33] = 0.996212229189942
        struct[0].Gy_ini[98,36] = -0.249053057297486
        struct[0].Gy_ini[98,37] = -0.996212229189942
        struct[0].Gy_ini[98,64] = -0.0107843137254902
        struct[0].Gy_ini[98,65] = -0.0431372549019608
        struct[0].Gy_ini[98,66] = -0.0107843137254902
        struct[0].Gy_ini[98,67] = -0.0431372549019608
        struct[0].Gy_ini[98,68] = 0.0215686274509804
        struct[0].Gy_ini[98,69] = 0.0862745098039216
        struct[0].Gy_ini[98,98] = -1
        struct[0].Gy_ini[99,32] = -0.996212229189942
        struct[0].Gy_ini[99,33] = 0.249053057297486
        struct[0].Gy_ini[99,36] = 0.996212229189942
        struct[0].Gy_ini[99,37] = -0.249053057297486
        struct[0].Gy_ini[99,64] = 0.0431372549019608
        struct[0].Gy_ini[99,65] = -0.0107843137254902
        struct[0].Gy_ini[99,66] = 0.0431372549019608
        struct[0].Gy_ini[99,67] = -0.0107843137254902
        struct[0].Gy_ini[99,68] = -0.0862745098039216
        struct[0].Gy_ini[99,69] = 0.0215686274509804
        struct[0].Gy_ini[99,99] = -1
        struct[0].Gy_ini[100,40] = -1.10755301189314
        struct[0].Gy_ini[100,41] = -0.598820527961361
        struct[0].Gy_ini[100,42] = 0.316443717683753
        struct[0].Gy_ini[100,43] = 0.171091579417532
        struct[0].Gy_ini[100,44] = 0.316443717683753
        struct[0].Gy_ini[100,45] = 0.171091579417532
        struct[0].Gy_ini[100,100] = -1
        struct[0].Gy_ini[101,40] = 0.598820527961361
        struct[0].Gy_ini[101,41] = -1.10755301189314
        struct[0].Gy_ini[101,42] = -0.171091579417532
        struct[0].Gy_ini[101,43] = 0.316443717683753
        struct[0].Gy_ini[101,44] = -0.171091579417532
        struct[0].Gy_ini[101,45] = 0.316443717683753
        struct[0].Gy_ini[101,101] = -1
        struct[0].Gy_ini[102,40] = 0.316443717683753
        struct[0].Gy_ini[102,41] = 0.171091579417532
        struct[0].Gy_ini[102,42] = -1.10755301189314
        struct[0].Gy_ini[102,43] = -0.598820527961360
        struct[0].Gy_ini[102,44] = 0.316443717683753
        struct[0].Gy_ini[102,45] = 0.171091579417531
        struct[0].Gy_ini[102,102] = -1
        struct[0].Gy_ini[103,40] = -0.171091579417532
        struct[0].Gy_ini[103,41] = 0.316443717683753
        struct[0].Gy_ini[103,42] = 0.598820527961360
        struct[0].Gy_ini[103,43] = -1.10755301189314
        struct[0].Gy_ini[103,44] = -0.171091579417531
        struct[0].Gy_ini[103,45] = 0.316443717683753
        struct[0].Gy_ini[103,103] = -1
        struct[0].Gy_ini[104,40] = 0.316443717683753
        struct[0].Gy_ini[104,41] = 0.171091579417532
        struct[0].Gy_ini[104,42] = 0.316443717683753
        struct[0].Gy_ini[104,43] = 0.171091579417531
        struct[0].Gy_ini[104,44] = -1.10755301189314
        struct[0].Gy_ini[104,45] = -0.598820527961360
        struct[0].Gy_ini[104,104] = -1
        struct[0].Gy_ini[105,40] = -0.171091579417532
        struct[0].Gy_ini[105,41] = 0.316443717683753
        struct[0].Gy_ini[105,42] = -0.171091579417531
        struct[0].Gy_ini[105,43] = 0.316443717683753
        struct[0].Gy_ini[105,44] = 0.598820527961360
        struct[0].Gy_ini[105,45] = -1.10755301189314
        struct[0].Gy_ini[105,105] = -1
        struct[0].Gy_ini[106,40] = 1.10755301189314
        struct[0].Gy_ini[106,41] = 0.598820527961361
        struct[0].Gy_ini[106,42] = -0.316443717683753
        struct[0].Gy_ini[106,43] = -0.171091579417532
        struct[0].Gy_ini[106,44] = -0.316443717683753
        struct[0].Gy_ini[106,45] = -0.171091579417532
        struct[0].Gy_ini[106,46] = -1.10755301189314
        struct[0].Gy_ini[106,47] = -0.598820527961361
        struct[0].Gy_ini[106,48] = 0.316443717683753
        struct[0].Gy_ini[106,49] = 0.171091579417532
        struct[0].Gy_ini[106,50] = 0.316443717683753
        struct[0].Gy_ini[106,51] = 0.171091579417532
        struct[0].Gy_ini[106,106] = -1
        struct[0].Gy_ini[107,40] = -0.598820527961361
        struct[0].Gy_ini[107,41] = 1.10755301189314
        struct[0].Gy_ini[107,42] = 0.171091579417532
        struct[0].Gy_ini[107,43] = -0.316443717683753
        struct[0].Gy_ini[107,44] = 0.171091579417532
        struct[0].Gy_ini[107,45] = -0.316443717683753
        struct[0].Gy_ini[107,46] = 0.598820527961361
        struct[0].Gy_ini[107,47] = -1.10755301189314
        struct[0].Gy_ini[107,48] = -0.171091579417532
        struct[0].Gy_ini[107,49] = 0.316443717683753
        struct[0].Gy_ini[107,50] = -0.171091579417532
        struct[0].Gy_ini[107,51] = 0.316443717683753
        struct[0].Gy_ini[107,107] = -1
        struct[0].Gy_ini[108,40] = -0.316443717683753
        struct[0].Gy_ini[108,41] = -0.171091579417532
        struct[0].Gy_ini[108,42] = 1.10755301189314
        struct[0].Gy_ini[108,43] = 0.598820527961360
        struct[0].Gy_ini[108,44] = -0.316443717683753
        struct[0].Gy_ini[108,45] = -0.171091579417531
        struct[0].Gy_ini[108,46] = 0.316443717683753
        struct[0].Gy_ini[108,47] = 0.171091579417532
        struct[0].Gy_ini[108,48] = -1.10755301189314
        struct[0].Gy_ini[108,49] = -0.598820527961360
        struct[0].Gy_ini[108,50] = 0.316443717683753
        struct[0].Gy_ini[108,51] = 0.171091579417531
        struct[0].Gy_ini[108,108] = -1
        struct[0].Gy_ini[109,40] = 0.171091579417532
        struct[0].Gy_ini[109,41] = -0.316443717683753
        struct[0].Gy_ini[109,42] = -0.598820527961360
        struct[0].Gy_ini[109,43] = 1.10755301189314
        struct[0].Gy_ini[109,44] = 0.171091579417531
        struct[0].Gy_ini[109,45] = -0.316443717683753
        struct[0].Gy_ini[109,46] = -0.171091579417532
        struct[0].Gy_ini[109,47] = 0.316443717683753
        struct[0].Gy_ini[109,48] = 0.598820527961360
        struct[0].Gy_ini[109,49] = -1.10755301189314
        struct[0].Gy_ini[109,50] = -0.171091579417531
        struct[0].Gy_ini[109,51] = 0.316443717683753
        struct[0].Gy_ini[109,109] = -1
        struct[0].Gy_ini[110,40] = -0.316443717683753
        struct[0].Gy_ini[110,41] = -0.171091579417532
        struct[0].Gy_ini[110,42] = -0.316443717683753
        struct[0].Gy_ini[110,43] = -0.171091579417531
        struct[0].Gy_ini[110,44] = 1.10755301189314
        struct[0].Gy_ini[110,45] = 0.598820527961360
        struct[0].Gy_ini[110,46] = 0.316443717683753
        struct[0].Gy_ini[110,47] = 0.171091579417532
        struct[0].Gy_ini[110,48] = 0.316443717683753
        struct[0].Gy_ini[110,49] = 0.171091579417531
        struct[0].Gy_ini[110,50] = -1.10755301189314
        struct[0].Gy_ini[110,51] = -0.598820527961360
        struct[0].Gy_ini[110,110] = -1
        struct[0].Gy_ini[111,40] = 0.171091579417532
        struct[0].Gy_ini[111,41] = -0.316443717683753
        struct[0].Gy_ini[111,42] = 0.171091579417531
        struct[0].Gy_ini[111,43] = -0.316443717683753
        struct[0].Gy_ini[111,44] = -0.598820527961360
        struct[0].Gy_ini[111,45] = 1.10755301189314
        struct[0].Gy_ini[111,46] = -0.171091579417532
        struct[0].Gy_ini[111,47] = 0.316443717683753
        struct[0].Gy_ini[111,48] = -0.171091579417531
        struct[0].Gy_ini[111,49] = 0.316443717683753
        struct[0].Gy_ini[111,50] = 0.598820527961360
        struct[0].Gy_ini[111,51] = -1.10755301189314
        struct[0].Gy_ini[111,111] = -1
        struct[0].Gy_ini[112,46] = 1.10755301189314
        struct[0].Gy_ini[112,47] = 0.598820527961361
        struct[0].Gy_ini[112,48] = -0.316443717683753
        struct[0].Gy_ini[112,49] = -0.171091579417532
        struct[0].Gy_ini[112,50] = -0.316443717683753
        struct[0].Gy_ini[112,51] = -0.171091579417532
        struct[0].Gy_ini[112,52] = -1.10755301189314
        struct[0].Gy_ini[112,53] = -0.598820527961361
        struct[0].Gy_ini[112,54] = 0.316443717683753
        struct[0].Gy_ini[112,55] = 0.171091579417532
        struct[0].Gy_ini[112,56] = 0.316443717683753
        struct[0].Gy_ini[112,57] = 0.171091579417532
        struct[0].Gy_ini[112,112] = -1
        struct[0].Gy_ini[113,46] = -0.598820527961361
        struct[0].Gy_ini[113,47] = 1.10755301189314
        struct[0].Gy_ini[113,48] = 0.171091579417532
        struct[0].Gy_ini[113,49] = -0.316443717683753
        struct[0].Gy_ini[113,50] = 0.171091579417532
        struct[0].Gy_ini[113,51] = -0.316443717683753
        struct[0].Gy_ini[113,52] = 0.598820527961361
        struct[0].Gy_ini[113,53] = -1.10755301189314
        struct[0].Gy_ini[113,54] = -0.171091579417532
        struct[0].Gy_ini[113,55] = 0.316443717683753
        struct[0].Gy_ini[113,56] = -0.171091579417532
        struct[0].Gy_ini[113,57] = 0.316443717683753
        struct[0].Gy_ini[113,113] = -1
        struct[0].Gy_ini[114,46] = -0.316443717683753
        struct[0].Gy_ini[114,47] = -0.171091579417532
        struct[0].Gy_ini[114,48] = 1.10755301189314
        struct[0].Gy_ini[114,49] = 0.598820527961360
        struct[0].Gy_ini[114,50] = -0.316443717683753
        struct[0].Gy_ini[114,51] = -0.171091579417531
        struct[0].Gy_ini[114,52] = 0.316443717683753
        struct[0].Gy_ini[114,53] = 0.171091579417532
        struct[0].Gy_ini[114,54] = -1.10755301189314
        struct[0].Gy_ini[114,55] = -0.598820527961360
        struct[0].Gy_ini[114,56] = 0.316443717683753
        struct[0].Gy_ini[114,57] = 0.171091579417531
        struct[0].Gy_ini[114,114] = -1
        struct[0].Gy_ini[115,46] = 0.171091579417532
        struct[0].Gy_ini[115,47] = -0.316443717683753
        struct[0].Gy_ini[115,48] = -0.598820527961360
        struct[0].Gy_ini[115,49] = 1.10755301189314
        struct[0].Gy_ini[115,50] = 0.171091579417531
        struct[0].Gy_ini[115,51] = -0.316443717683753
        struct[0].Gy_ini[115,52] = -0.171091579417532
        struct[0].Gy_ini[115,53] = 0.316443717683753
        struct[0].Gy_ini[115,54] = 0.598820527961360
        struct[0].Gy_ini[115,55] = -1.10755301189314
        struct[0].Gy_ini[115,56] = -0.171091579417531
        struct[0].Gy_ini[115,57] = 0.316443717683753
        struct[0].Gy_ini[115,115] = -1
        struct[0].Gy_ini[116,46] = -0.316443717683753
        struct[0].Gy_ini[116,47] = -0.171091579417532
        struct[0].Gy_ini[116,48] = -0.316443717683753
        struct[0].Gy_ini[116,49] = -0.171091579417531
        struct[0].Gy_ini[116,50] = 1.10755301189314
        struct[0].Gy_ini[116,51] = 0.598820527961360
        struct[0].Gy_ini[116,52] = 0.316443717683753
        struct[0].Gy_ini[116,53] = 0.171091579417532
        struct[0].Gy_ini[116,54] = 0.316443717683753
        struct[0].Gy_ini[116,55] = 0.171091579417531
        struct[0].Gy_ini[116,56] = -1.10755301189314
        struct[0].Gy_ini[116,57] = -0.598820527961360
        struct[0].Gy_ini[116,116] = -1
        struct[0].Gy_ini[117,46] = 0.171091579417532
        struct[0].Gy_ini[117,47] = -0.316443717683753
        struct[0].Gy_ini[117,48] = 0.171091579417531
        struct[0].Gy_ini[117,49] = -0.316443717683753
        struct[0].Gy_ini[117,50] = -0.598820527961360
        struct[0].Gy_ini[117,51] = 1.10755301189314
        struct[0].Gy_ini[117,52] = -0.171091579417532
        struct[0].Gy_ini[117,53] = 0.316443717683753
        struct[0].Gy_ini[117,54] = -0.171091579417531
        struct[0].Gy_ini[117,55] = 0.316443717683753
        struct[0].Gy_ini[117,56] = 0.598820527961360
        struct[0].Gy_ini[117,57] = -1.10755301189314
        struct[0].Gy_ini[117,117] = -1
        struct[0].Gy_ini[118,58] = 1.10755301189314
        struct[0].Gy_ini[118,59] = 0.598820527961361
        struct[0].Gy_ini[118,60] = -0.316443717683753
        struct[0].Gy_ini[118,61] = -0.171091579417532
        struct[0].Gy_ini[118,62] = -0.316443717683753
        struct[0].Gy_ini[118,63] = -0.171091579417532
        struct[0].Gy_ini[118,64] = -1.10755301189314
        struct[0].Gy_ini[118,65] = -0.598820527961361
        struct[0].Gy_ini[118,66] = 0.316443717683753
        struct[0].Gy_ini[118,67] = 0.171091579417532
        struct[0].Gy_ini[118,68] = 0.316443717683753
        struct[0].Gy_ini[118,69] = 0.171091579417532
        struct[0].Gy_ini[118,118] = -1
        struct[0].Gy_ini[119,58] = -0.598820527961361
        struct[0].Gy_ini[119,59] = 1.10755301189314
        struct[0].Gy_ini[119,60] = 0.171091579417532
        struct[0].Gy_ini[119,61] = -0.316443717683753
        struct[0].Gy_ini[119,62] = 0.171091579417532
        struct[0].Gy_ini[119,63] = -0.316443717683753
        struct[0].Gy_ini[119,64] = 0.598820527961361
        struct[0].Gy_ini[119,65] = -1.10755301189314
        struct[0].Gy_ini[119,66] = -0.171091579417532
        struct[0].Gy_ini[119,67] = 0.316443717683753
        struct[0].Gy_ini[119,68] = -0.171091579417532
        struct[0].Gy_ini[119,69] = 0.316443717683753
        struct[0].Gy_ini[119,119] = -1
        struct[0].Gy_ini[120,58] = -0.316443717683753
        struct[0].Gy_ini[120,59] = -0.171091579417532
        struct[0].Gy_ini[120,60] = 1.10755301189314
        struct[0].Gy_ini[120,61] = 0.598820527961360
        struct[0].Gy_ini[120,62] = -0.316443717683753
        struct[0].Gy_ini[120,63] = -0.171091579417531
        struct[0].Gy_ini[120,64] = 0.316443717683753
        struct[0].Gy_ini[120,65] = 0.171091579417532
        struct[0].Gy_ini[120,66] = -1.10755301189314
        struct[0].Gy_ini[120,67] = -0.598820527961360
        struct[0].Gy_ini[120,68] = 0.316443717683753
        struct[0].Gy_ini[120,69] = 0.171091579417531
        struct[0].Gy_ini[120,120] = -1
        struct[0].Gy_ini[121,58] = 0.171091579417532
        struct[0].Gy_ini[121,59] = -0.316443717683753
        struct[0].Gy_ini[121,60] = -0.598820527961360
        struct[0].Gy_ini[121,61] = 1.10755301189314
        struct[0].Gy_ini[121,62] = 0.171091579417531
        struct[0].Gy_ini[121,63] = -0.316443717683753
        struct[0].Gy_ini[121,64] = -0.171091579417532
        struct[0].Gy_ini[121,65] = 0.316443717683753
        struct[0].Gy_ini[121,66] = 0.598820527961360
        struct[0].Gy_ini[121,67] = -1.10755301189314
        struct[0].Gy_ini[121,68] = -0.171091579417531
        struct[0].Gy_ini[121,69] = 0.316443717683753
        struct[0].Gy_ini[121,121] = -1
        struct[0].Gy_ini[122,58] = -0.316443717683753
        struct[0].Gy_ini[122,59] = -0.171091579417532
        struct[0].Gy_ini[122,60] = -0.316443717683753
        struct[0].Gy_ini[122,61] = -0.171091579417531
        struct[0].Gy_ini[122,62] = 1.10755301189314
        struct[0].Gy_ini[122,63] = 0.598820527961360
        struct[0].Gy_ini[122,64] = 0.316443717683753
        struct[0].Gy_ini[122,65] = 0.171091579417532
        struct[0].Gy_ini[122,66] = 0.316443717683753
        struct[0].Gy_ini[122,67] = 0.171091579417531
        struct[0].Gy_ini[122,68] = -1.10755301189314
        struct[0].Gy_ini[122,69] = -0.598820527961360
        struct[0].Gy_ini[122,122] = -1
        struct[0].Gy_ini[123,58] = 0.171091579417532
        struct[0].Gy_ini[123,59] = -0.316443717683753
        struct[0].Gy_ini[123,60] = 0.171091579417531
        struct[0].Gy_ini[123,61] = -0.316443717683753
        struct[0].Gy_ini[123,62] = -0.598820527961360
        struct[0].Gy_ini[123,63] = 1.10755301189314
        struct[0].Gy_ini[123,64] = -0.171091579417532
        struct[0].Gy_ini[123,65] = 0.316443717683753
        struct[0].Gy_ini[123,66] = -0.171091579417531
        struct[0].Gy_ini[123,67] = 0.316443717683753
        struct[0].Gy_ini[123,68] = 0.598820527961360
        struct[0].Gy_ini[123,69] = -1.10755301189314
        struct[0].Gy_ini[123,123] = -1
        struct[0].Gy_ini[124,64] = 1.10755301189314
        struct[0].Gy_ini[124,65] = 0.598820527961361
        struct[0].Gy_ini[124,66] = -0.316443717683753
        struct[0].Gy_ini[124,67] = -0.171091579417532
        struct[0].Gy_ini[124,68] = -0.316443717683753
        struct[0].Gy_ini[124,69] = -0.171091579417532
        struct[0].Gy_ini[124,124] = -1
        struct[0].Gy_ini[125,64] = -0.598820527961361
        struct[0].Gy_ini[125,65] = 1.10755301189314
        struct[0].Gy_ini[125,66] = 0.171091579417532
        struct[0].Gy_ini[125,67] = -0.316443717683753
        struct[0].Gy_ini[125,68] = 0.171091579417532
        struct[0].Gy_ini[125,69] = -0.316443717683753
        struct[0].Gy_ini[125,125] = -1
        struct[0].Gy_ini[126,64] = -0.316443717683753
        struct[0].Gy_ini[126,65] = -0.171091579417532
        struct[0].Gy_ini[126,66] = 1.10755301189314
        struct[0].Gy_ini[126,67] = 0.598820527961360
        struct[0].Gy_ini[126,68] = -0.316443717683753
        struct[0].Gy_ini[126,69] = -0.171091579417531
        struct[0].Gy_ini[126,126] = -1
        struct[0].Gy_ini[127,64] = 0.171091579417532
        struct[0].Gy_ini[127,65] = -0.316443717683753
        struct[0].Gy_ini[127,66] = -0.598820527961360
        struct[0].Gy_ini[127,67] = 1.10755301189314
        struct[0].Gy_ini[127,68] = 0.171091579417531
        struct[0].Gy_ini[127,69] = -0.316443717683753
        struct[0].Gy_ini[127,127] = -1
        struct[0].Gy_ini[128,64] = -0.316443717683753
        struct[0].Gy_ini[128,65] = -0.171091579417532
        struct[0].Gy_ini[128,66] = -0.316443717683753
        struct[0].Gy_ini[128,67] = -0.171091579417531
        struct[0].Gy_ini[128,68] = 1.10755301189314
        struct[0].Gy_ini[128,69] = 0.598820527961360
        struct[0].Gy_ini[128,128] = -1
        struct[0].Gy_ini[129,64] = 0.171091579417532
        struct[0].Gy_ini[129,65] = -0.316443717683753
        struct[0].Gy_ini[129,66] = 0.171091579417531
        struct[0].Gy_ini[129,67] = -0.316443717683753
        struct[0].Gy_ini[129,68] = -0.598820527961360
        struct[0].Gy_ini[129,69] = 1.10755301189314
        struct[0].Gy_ini[129,129] = -1
        struct[0].Gy_ini[130,0] = i_load_B2lv_a_r
        struct[0].Gy_ini[130,1] = i_load_B2lv_a_i
        struct[0].Gy_ini[130,6] = -i_load_B2lv_a_r
        struct[0].Gy_ini[130,7] = -i_load_B2lv_a_i
        struct[0].Gy_ini[130,130] = v_B2lv_a_r - v_B2lv_n_r
        struct[0].Gy_ini[130,131] = v_B2lv_a_i - v_B2lv_n_i
        struct[0].Gy_ini[131,2] = i_load_B2lv_b_r
        struct[0].Gy_ini[131,3] = i_load_B2lv_b_i
        struct[0].Gy_ini[131,6] = -i_load_B2lv_b_r
        struct[0].Gy_ini[131,7] = -i_load_B2lv_b_i
        struct[0].Gy_ini[131,132] = v_B2lv_b_r - v_B2lv_n_r
        struct[0].Gy_ini[131,133] = v_B2lv_b_i - v_B2lv_n_i
        struct[0].Gy_ini[132,4] = i_load_B2lv_c_r
        struct[0].Gy_ini[132,5] = i_load_B2lv_c_i
        struct[0].Gy_ini[132,6] = -i_load_B2lv_c_r
        struct[0].Gy_ini[132,7] = -i_load_B2lv_c_i
        struct[0].Gy_ini[132,134] = v_B2lv_c_r - v_B2lv_n_r
        struct[0].Gy_ini[132,135] = v_B2lv_c_i - v_B2lv_n_i
        struct[0].Gy_ini[133,0] = -i_load_B2lv_a_i
        struct[0].Gy_ini[133,1] = i_load_B2lv_a_r
        struct[0].Gy_ini[133,6] = i_load_B2lv_a_i
        struct[0].Gy_ini[133,7] = -i_load_B2lv_a_r
        struct[0].Gy_ini[133,130] = v_B2lv_a_i - v_B2lv_n_i
        struct[0].Gy_ini[133,131] = -v_B2lv_a_r + v_B2lv_n_r
        struct[0].Gy_ini[134,2] = -i_load_B2lv_b_i
        struct[0].Gy_ini[134,3] = i_load_B2lv_b_r
        struct[0].Gy_ini[134,6] = i_load_B2lv_b_i
        struct[0].Gy_ini[134,7] = -i_load_B2lv_b_r
        struct[0].Gy_ini[134,132] = v_B2lv_b_i - v_B2lv_n_i
        struct[0].Gy_ini[134,133] = -v_B2lv_b_r + v_B2lv_n_r
        struct[0].Gy_ini[135,4] = -i_load_B2lv_c_i
        struct[0].Gy_ini[135,5] = i_load_B2lv_c_r
        struct[0].Gy_ini[135,6] = i_load_B2lv_c_i
        struct[0].Gy_ini[135,7] = -i_load_B2lv_c_r
        struct[0].Gy_ini[135,134] = v_B2lv_c_i - v_B2lv_n_i
        struct[0].Gy_ini[135,135] = -v_B2lv_c_r + v_B2lv_n_r
        struct[0].Gy_ini[136,130] = 1
        struct[0].Gy_ini[136,132] = 1
        struct[0].Gy_ini[136,134] = 1
        struct[0].Gy_ini[136,136] = 1
        struct[0].Gy_ini[137,131] = 1
        struct[0].Gy_ini[137,133] = 1
        struct[0].Gy_ini[137,135] = 1
        struct[0].Gy_ini[137,137] = 1
        struct[0].Gy_ini[138,8] = i_load_B3lv_a_r
        struct[0].Gy_ini[138,9] = i_load_B3lv_a_i
        struct[0].Gy_ini[138,14] = -i_load_B3lv_a_r
        struct[0].Gy_ini[138,15] = -i_load_B3lv_a_i
        struct[0].Gy_ini[138,138] = v_B3lv_a_r - v_B3lv_n_r
        struct[0].Gy_ini[138,139] = v_B3lv_a_i - v_B3lv_n_i
        struct[0].Gy_ini[139,10] = i_load_B3lv_b_r
        struct[0].Gy_ini[139,11] = i_load_B3lv_b_i
        struct[0].Gy_ini[139,14] = -i_load_B3lv_b_r
        struct[0].Gy_ini[139,15] = -i_load_B3lv_b_i
        struct[0].Gy_ini[139,140] = v_B3lv_b_r - v_B3lv_n_r
        struct[0].Gy_ini[139,141] = v_B3lv_b_i - v_B3lv_n_i
        struct[0].Gy_ini[140,12] = i_load_B3lv_c_r
        struct[0].Gy_ini[140,13] = i_load_B3lv_c_i
        struct[0].Gy_ini[140,14] = -i_load_B3lv_c_r
        struct[0].Gy_ini[140,15] = -i_load_B3lv_c_i
        struct[0].Gy_ini[140,142] = v_B3lv_c_r - v_B3lv_n_r
        struct[0].Gy_ini[140,143] = v_B3lv_c_i - v_B3lv_n_i
        struct[0].Gy_ini[141,8] = -i_load_B3lv_a_i
        struct[0].Gy_ini[141,9] = i_load_B3lv_a_r
        struct[0].Gy_ini[141,14] = i_load_B3lv_a_i
        struct[0].Gy_ini[141,15] = -i_load_B3lv_a_r
        struct[0].Gy_ini[141,138] = v_B3lv_a_i - v_B3lv_n_i
        struct[0].Gy_ini[141,139] = -v_B3lv_a_r + v_B3lv_n_r
        struct[0].Gy_ini[142,10] = -i_load_B3lv_b_i
        struct[0].Gy_ini[142,11] = i_load_B3lv_b_r
        struct[0].Gy_ini[142,14] = i_load_B3lv_b_i
        struct[0].Gy_ini[142,15] = -i_load_B3lv_b_r
        struct[0].Gy_ini[142,140] = v_B3lv_b_i - v_B3lv_n_i
        struct[0].Gy_ini[142,141] = -v_B3lv_b_r + v_B3lv_n_r
        struct[0].Gy_ini[143,12] = -i_load_B3lv_c_i
        struct[0].Gy_ini[143,13] = i_load_B3lv_c_r
        struct[0].Gy_ini[143,14] = i_load_B3lv_c_i
        struct[0].Gy_ini[143,15] = -i_load_B3lv_c_r
        struct[0].Gy_ini[143,142] = v_B3lv_c_i - v_B3lv_n_i
        struct[0].Gy_ini[143,143] = -v_B3lv_c_r + v_B3lv_n_r
        struct[0].Gy_ini[144,138] = 1
        struct[0].Gy_ini[144,140] = 1
        struct[0].Gy_ini[144,142] = 1
        struct[0].Gy_ini[144,144] = 1
        struct[0].Gy_ini[145,139] = 1
        struct[0].Gy_ini[145,141] = 1
        struct[0].Gy_ini[145,143] = 1
        struct[0].Gy_ini[145,145] = 1
        struct[0].Gy_ini[146,16] = i_load_B4lv_a_r
        struct[0].Gy_ini[146,17] = i_load_B4lv_a_i
        struct[0].Gy_ini[146,22] = -i_load_B4lv_a_r
        struct[0].Gy_ini[146,23] = -i_load_B4lv_a_i
        struct[0].Gy_ini[146,146] = v_B4lv_a_r - v_B4lv_n_r
        struct[0].Gy_ini[146,147] = v_B4lv_a_i - v_B4lv_n_i
        struct[0].Gy_ini[147,18] = i_load_B4lv_b_r
        struct[0].Gy_ini[147,19] = i_load_B4lv_b_i
        struct[0].Gy_ini[147,22] = -i_load_B4lv_b_r
        struct[0].Gy_ini[147,23] = -i_load_B4lv_b_i
        struct[0].Gy_ini[147,148] = v_B4lv_b_r - v_B4lv_n_r
        struct[0].Gy_ini[147,149] = v_B4lv_b_i - v_B4lv_n_i
        struct[0].Gy_ini[148,20] = i_load_B4lv_c_r
        struct[0].Gy_ini[148,21] = i_load_B4lv_c_i
        struct[0].Gy_ini[148,22] = -i_load_B4lv_c_r
        struct[0].Gy_ini[148,23] = -i_load_B4lv_c_i
        struct[0].Gy_ini[148,150] = v_B4lv_c_r - v_B4lv_n_r
        struct[0].Gy_ini[148,151] = v_B4lv_c_i - v_B4lv_n_i
        struct[0].Gy_ini[149,16] = -i_load_B4lv_a_i
        struct[0].Gy_ini[149,17] = i_load_B4lv_a_r
        struct[0].Gy_ini[149,22] = i_load_B4lv_a_i
        struct[0].Gy_ini[149,23] = -i_load_B4lv_a_r
        struct[0].Gy_ini[149,146] = v_B4lv_a_i - v_B4lv_n_i
        struct[0].Gy_ini[149,147] = -v_B4lv_a_r + v_B4lv_n_r
        struct[0].Gy_ini[150,18] = -i_load_B4lv_b_i
        struct[0].Gy_ini[150,19] = i_load_B4lv_b_r
        struct[0].Gy_ini[150,22] = i_load_B4lv_b_i
        struct[0].Gy_ini[150,23] = -i_load_B4lv_b_r
        struct[0].Gy_ini[150,148] = v_B4lv_b_i - v_B4lv_n_i
        struct[0].Gy_ini[150,149] = -v_B4lv_b_r + v_B4lv_n_r
        struct[0].Gy_ini[151,20] = -i_load_B4lv_c_i
        struct[0].Gy_ini[151,21] = i_load_B4lv_c_r
        struct[0].Gy_ini[151,22] = i_load_B4lv_c_i
        struct[0].Gy_ini[151,23] = -i_load_B4lv_c_r
        struct[0].Gy_ini[151,150] = v_B4lv_c_i - v_B4lv_n_i
        struct[0].Gy_ini[151,151] = -v_B4lv_c_r + v_B4lv_n_r
        struct[0].Gy_ini[152,146] = 1
        struct[0].Gy_ini[152,148] = 1
        struct[0].Gy_ini[152,150] = 1
        struct[0].Gy_ini[152,152] = 1
        struct[0].Gy_ini[153,147] = 1
        struct[0].Gy_ini[153,149] = 1
        struct[0].Gy_ini[153,151] = 1
        struct[0].Gy_ini[153,153] = 1
        struct[0].Gy_ini[154,24] = i_load_B5lv_a_r
        struct[0].Gy_ini[154,25] = i_load_B5lv_a_i
        struct[0].Gy_ini[154,30] = -i_load_B5lv_a_r
        struct[0].Gy_ini[154,31] = -i_load_B5lv_a_i
        struct[0].Gy_ini[154,154] = v_B5lv_a_r - v_B5lv_n_r
        struct[0].Gy_ini[154,155] = v_B5lv_a_i - v_B5lv_n_i
        struct[0].Gy_ini[155,26] = i_load_B5lv_b_r
        struct[0].Gy_ini[155,27] = i_load_B5lv_b_i
        struct[0].Gy_ini[155,30] = -i_load_B5lv_b_r
        struct[0].Gy_ini[155,31] = -i_load_B5lv_b_i
        struct[0].Gy_ini[155,156] = v_B5lv_b_r - v_B5lv_n_r
        struct[0].Gy_ini[155,157] = v_B5lv_b_i - v_B5lv_n_i
        struct[0].Gy_ini[156,28] = i_load_B5lv_c_r
        struct[0].Gy_ini[156,29] = i_load_B5lv_c_i
        struct[0].Gy_ini[156,30] = -i_load_B5lv_c_r
        struct[0].Gy_ini[156,31] = -i_load_B5lv_c_i
        struct[0].Gy_ini[156,158] = v_B5lv_c_r - v_B5lv_n_r
        struct[0].Gy_ini[156,159] = v_B5lv_c_i - v_B5lv_n_i
        struct[0].Gy_ini[157,24] = -i_load_B5lv_a_i
        struct[0].Gy_ini[157,25] = i_load_B5lv_a_r
        struct[0].Gy_ini[157,30] = i_load_B5lv_a_i
        struct[0].Gy_ini[157,31] = -i_load_B5lv_a_r
        struct[0].Gy_ini[157,154] = v_B5lv_a_i - v_B5lv_n_i
        struct[0].Gy_ini[157,155] = -v_B5lv_a_r + v_B5lv_n_r
        struct[0].Gy_ini[158,26] = -i_load_B5lv_b_i
        struct[0].Gy_ini[158,27] = i_load_B5lv_b_r
        struct[0].Gy_ini[158,30] = i_load_B5lv_b_i
        struct[0].Gy_ini[158,31] = -i_load_B5lv_b_r
        struct[0].Gy_ini[158,156] = v_B5lv_b_i - v_B5lv_n_i
        struct[0].Gy_ini[158,157] = -v_B5lv_b_r + v_B5lv_n_r
        struct[0].Gy_ini[159,28] = -i_load_B5lv_c_i
        struct[0].Gy_ini[159,29] = i_load_B5lv_c_r
        struct[0].Gy_ini[159,30] = i_load_B5lv_c_i
        struct[0].Gy_ini[159,31] = -i_load_B5lv_c_r
        struct[0].Gy_ini[159,158] = v_B5lv_c_i - v_B5lv_n_i
        struct[0].Gy_ini[159,159] = -v_B5lv_c_r + v_B5lv_n_r
        struct[0].Gy_ini[160,154] = 1
        struct[0].Gy_ini[160,156] = 1
        struct[0].Gy_ini[160,158] = 1
        struct[0].Gy_ini[160,160] = 1
        struct[0].Gy_ini[161,155] = 1
        struct[0].Gy_ini[161,157] = 1
        struct[0].Gy_ini[161,159] = 1
        struct[0].Gy_ini[161,161] = 1
        struct[0].Gy_ini[162,32] = i_load_B6lv_a_r
        struct[0].Gy_ini[162,33] = i_load_B6lv_a_i
        struct[0].Gy_ini[162,38] = -i_load_B6lv_a_r
        struct[0].Gy_ini[162,39] = -i_load_B6lv_a_i
        struct[0].Gy_ini[162,162] = v_B6lv_a_r - v_B6lv_n_r
        struct[0].Gy_ini[162,163] = v_B6lv_a_i - v_B6lv_n_i
        struct[0].Gy_ini[163,34] = i_load_B6lv_b_r
        struct[0].Gy_ini[163,35] = i_load_B6lv_b_i
        struct[0].Gy_ini[163,38] = -i_load_B6lv_b_r
        struct[0].Gy_ini[163,39] = -i_load_B6lv_b_i
        struct[0].Gy_ini[163,164] = v_B6lv_b_r - v_B6lv_n_r
        struct[0].Gy_ini[163,165] = v_B6lv_b_i - v_B6lv_n_i
        struct[0].Gy_ini[164,36] = i_load_B6lv_c_r
        struct[0].Gy_ini[164,37] = i_load_B6lv_c_i
        struct[0].Gy_ini[164,38] = -i_load_B6lv_c_r
        struct[0].Gy_ini[164,39] = -i_load_B6lv_c_i
        struct[0].Gy_ini[164,166] = v_B6lv_c_r - v_B6lv_n_r
        struct[0].Gy_ini[164,167] = v_B6lv_c_i - v_B6lv_n_i
        struct[0].Gy_ini[165,32] = -i_load_B6lv_a_i
        struct[0].Gy_ini[165,33] = i_load_B6lv_a_r
        struct[0].Gy_ini[165,38] = i_load_B6lv_a_i
        struct[0].Gy_ini[165,39] = -i_load_B6lv_a_r
        struct[0].Gy_ini[165,162] = v_B6lv_a_i - v_B6lv_n_i
        struct[0].Gy_ini[165,163] = -v_B6lv_a_r + v_B6lv_n_r
        struct[0].Gy_ini[166,34] = -i_load_B6lv_b_i
        struct[0].Gy_ini[166,35] = i_load_B6lv_b_r
        struct[0].Gy_ini[166,38] = i_load_B6lv_b_i
        struct[0].Gy_ini[166,39] = -i_load_B6lv_b_r
        struct[0].Gy_ini[166,164] = v_B6lv_b_i - v_B6lv_n_i
        struct[0].Gy_ini[166,165] = -v_B6lv_b_r + v_B6lv_n_r
        struct[0].Gy_ini[167,36] = -i_load_B6lv_c_i
        struct[0].Gy_ini[167,37] = i_load_B6lv_c_r
        struct[0].Gy_ini[167,38] = i_load_B6lv_c_i
        struct[0].Gy_ini[167,39] = -i_load_B6lv_c_r
        struct[0].Gy_ini[167,166] = v_B6lv_c_i - v_B6lv_n_i
        struct[0].Gy_ini[167,167] = -v_B6lv_c_r + v_B6lv_n_r
        struct[0].Gy_ini[168,162] = 1
        struct[0].Gy_ini[168,164] = 1
        struct[0].Gy_ini[168,166] = 1
        struct[0].Gy_ini[168,168] = 1
        struct[0].Gy_ini[169,163] = 1
        struct[0].Gy_ini[169,165] = 1
        struct[0].Gy_ini[169,167] = 1
        struct[0].Gy_ini[169,169] = 1



def run_nn(t,struct,mode):

    # Parameters:
    
    # Inputs:
    v_B1_a_r = struct[0].v_B1_a_r
    v_B1_a_i = struct[0].v_B1_a_i
    v_B1_b_r = struct[0].v_B1_b_r
    v_B1_b_i = struct[0].v_B1_b_i
    v_B1_c_r = struct[0].v_B1_c_r
    v_B1_c_i = struct[0].v_B1_c_i
    v_B7_a_r = struct[0].v_B7_a_r
    v_B7_a_i = struct[0].v_B7_a_i
    v_B7_b_r = struct[0].v_B7_b_r
    v_B7_b_i = struct[0].v_B7_b_i
    v_B7_c_r = struct[0].v_B7_c_r
    v_B7_c_i = struct[0].v_B7_c_i
    i_B2lv_n_r = struct[0].i_B2lv_n_r
    i_B2lv_n_i = struct[0].i_B2lv_n_i
    i_B3lv_n_r = struct[0].i_B3lv_n_r
    i_B3lv_n_i = struct[0].i_B3lv_n_i
    i_B4lv_n_r = struct[0].i_B4lv_n_r
    i_B4lv_n_i = struct[0].i_B4lv_n_i
    i_B5lv_n_r = struct[0].i_B5lv_n_r
    i_B5lv_n_i = struct[0].i_B5lv_n_i
    i_B6lv_n_r = struct[0].i_B6lv_n_r
    i_B6lv_n_i = struct[0].i_B6lv_n_i
    i_B2_a_r = struct[0].i_B2_a_r
    i_B2_a_i = struct[0].i_B2_a_i
    i_B2_b_r = struct[0].i_B2_b_r
    i_B2_b_i = struct[0].i_B2_b_i
    i_B2_c_r = struct[0].i_B2_c_r
    i_B2_c_i = struct[0].i_B2_c_i
    i_B3_a_r = struct[0].i_B3_a_r
    i_B3_a_i = struct[0].i_B3_a_i
    i_B3_b_r = struct[0].i_B3_b_r
    i_B3_b_i = struct[0].i_B3_b_i
    i_B3_c_r = struct[0].i_B3_c_r
    i_B3_c_i = struct[0].i_B3_c_i
    i_B4_a_r = struct[0].i_B4_a_r
    i_B4_a_i = struct[0].i_B4_a_i
    i_B4_b_r = struct[0].i_B4_b_r
    i_B4_b_i = struct[0].i_B4_b_i
    i_B4_c_r = struct[0].i_B4_c_r
    i_B4_c_i = struct[0].i_B4_c_i
    i_B5_a_r = struct[0].i_B5_a_r
    i_B5_a_i = struct[0].i_B5_a_i
    i_B5_b_r = struct[0].i_B5_b_r
    i_B5_b_i = struct[0].i_B5_b_i
    i_B5_c_r = struct[0].i_B5_c_r
    i_B5_c_i = struct[0].i_B5_c_i
    i_B6_a_r = struct[0].i_B6_a_r
    i_B6_a_i = struct[0].i_B6_a_i
    i_B6_b_r = struct[0].i_B6_b_r
    i_B6_b_i = struct[0].i_B6_b_i
    i_B6_c_r = struct[0].i_B6_c_r
    i_B6_c_i = struct[0].i_B6_c_i
    p_B2lv_a = struct[0].p_B2lv_a
    q_B2lv_a = struct[0].q_B2lv_a
    p_B2lv_b = struct[0].p_B2lv_b
    q_B2lv_b = struct[0].q_B2lv_b
    p_B2lv_c = struct[0].p_B2lv_c
    q_B2lv_c = struct[0].q_B2lv_c
    p_B3lv_a = struct[0].p_B3lv_a
    q_B3lv_a = struct[0].q_B3lv_a
    p_B3lv_b = struct[0].p_B3lv_b
    q_B3lv_b = struct[0].q_B3lv_b
    p_B3lv_c = struct[0].p_B3lv_c
    q_B3lv_c = struct[0].q_B3lv_c
    p_B4lv_a = struct[0].p_B4lv_a
    q_B4lv_a = struct[0].q_B4lv_a
    p_B4lv_b = struct[0].p_B4lv_b
    q_B4lv_b = struct[0].q_B4lv_b
    p_B4lv_c = struct[0].p_B4lv_c
    q_B4lv_c = struct[0].q_B4lv_c
    p_B5lv_a = struct[0].p_B5lv_a
    q_B5lv_a = struct[0].q_B5lv_a
    p_B5lv_b = struct[0].p_B5lv_b
    q_B5lv_b = struct[0].q_B5lv_b
    p_B5lv_c = struct[0].p_B5lv_c
    q_B5lv_c = struct[0].q_B5lv_c
    p_B6lv_a = struct[0].p_B6lv_a
    q_B6lv_a = struct[0].q_B6lv_a
    p_B6lv_b = struct[0].p_B6lv_b
    q_B6lv_b = struct[0].q_B6lv_b
    p_B6lv_c = struct[0].p_B6lv_c
    q_B6lv_c = struct[0].q_B6lv_c
    u_dummy = struct[0].u_dummy
    
    # Dynamical states:
    x_dummy = struct[0].x[0,0]
    
    # Algebraic states:
    v_B2lv_a_r = struct[0].y_run[0,0]
    v_B2lv_a_i = struct[0].y_run[1,0]
    v_B2lv_b_r = struct[0].y_run[2,0]
    v_B2lv_b_i = struct[0].y_run[3,0]
    v_B2lv_c_r = struct[0].y_run[4,0]
    v_B2lv_c_i = struct[0].y_run[5,0]
    v_B2lv_n_r = struct[0].y_run[6,0]
    v_B2lv_n_i = struct[0].y_run[7,0]
    v_B3lv_a_r = struct[0].y_run[8,0]
    v_B3lv_a_i = struct[0].y_run[9,0]
    v_B3lv_b_r = struct[0].y_run[10,0]
    v_B3lv_b_i = struct[0].y_run[11,0]
    v_B3lv_c_r = struct[0].y_run[12,0]
    v_B3lv_c_i = struct[0].y_run[13,0]
    v_B3lv_n_r = struct[0].y_run[14,0]
    v_B3lv_n_i = struct[0].y_run[15,0]
    v_B4lv_a_r = struct[0].y_run[16,0]
    v_B4lv_a_i = struct[0].y_run[17,0]
    v_B4lv_b_r = struct[0].y_run[18,0]
    v_B4lv_b_i = struct[0].y_run[19,0]
    v_B4lv_c_r = struct[0].y_run[20,0]
    v_B4lv_c_i = struct[0].y_run[21,0]
    v_B4lv_n_r = struct[0].y_run[22,0]
    v_B4lv_n_i = struct[0].y_run[23,0]
    v_B5lv_a_r = struct[0].y_run[24,0]
    v_B5lv_a_i = struct[0].y_run[25,0]
    v_B5lv_b_r = struct[0].y_run[26,0]
    v_B5lv_b_i = struct[0].y_run[27,0]
    v_B5lv_c_r = struct[0].y_run[28,0]
    v_B5lv_c_i = struct[0].y_run[29,0]
    v_B5lv_n_r = struct[0].y_run[30,0]
    v_B5lv_n_i = struct[0].y_run[31,0]
    v_B6lv_a_r = struct[0].y_run[32,0]
    v_B6lv_a_i = struct[0].y_run[33,0]
    v_B6lv_b_r = struct[0].y_run[34,0]
    v_B6lv_b_i = struct[0].y_run[35,0]
    v_B6lv_c_r = struct[0].y_run[36,0]
    v_B6lv_c_i = struct[0].y_run[37,0]
    v_B6lv_n_r = struct[0].y_run[38,0]
    v_B6lv_n_i = struct[0].y_run[39,0]
    v_B2_a_r = struct[0].y_run[40,0]
    v_B2_a_i = struct[0].y_run[41,0]
    v_B2_b_r = struct[0].y_run[42,0]
    v_B2_b_i = struct[0].y_run[43,0]
    v_B2_c_r = struct[0].y_run[44,0]
    v_B2_c_i = struct[0].y_run[45,0]
    v_B3_a_r = struct[0].y_run[46,0]
    v_B3_a_i = struct[0].y_run[47,0]
    v_B3_b_r = struct[0].y_run[48,0]
    v_B3_b_i = struct[0].y_run[49,0]
    v_B3_c_r = struct[0].y_run[50,0]
    v_B3_c_i = struct[0].y_run[51,0]
    v_B4_a_r = struct[0].y_run[52,0]
    v_B4_a_i = struct[0].y_run[53,0]
    v_B4_b_r = struct[0].y_run[54,0]
    v_B4_b_i = struct[0].y_run[55,0]
    v_B4_c_r = struct[0].y_run[56,0]
    v_B4_c_i = struct[0].y_run[57,0]
    v_B5_a_r = struct[0].y_run[58,0]
    v_B5_a_i = struct[0].y_run[59,0]
    v_B5_b_r = struct[0].y_run[60,0]
    v_B5_b_i = struct[0].y_run[61,0]
    v_B5_c_r = struct[0].y_run[62,0]
    v_B5_c_i = struct[0].y_run[63,0]
    v_B6_a_r = struct[0].y_run[64,0]
    v_B6_a_i = struct[0].y_run[65,0]
    v_B6_b_r = struct[0].y_run[66,0]
    v_B6_b_i = struct[0].y_run[67,0]
    v_B6_c_r = struct[0].y_run[68,0]
    v_B6_c_i = struct[0].y_run[69,0]
    i_t_B2_B2lv_a_r = struct[0].y_run[70,0]
    i_t_B2_B2lv_a_i = struct[0].y_run[71,0]
    i_t_B2_B2lv_b_r = struct[0].y_run[72,0]
    i_t_B2_B2lv_b_i = struct[0].y_run[73,0]
    i_t_B2_B2lv_c_r = struct[0].y_run[74,0]
    i_t_B2_B2lv_c_i = struct[0].y_run[75,0]
    i_t_B3_B3lv_a_r = struct[0].y_run[76,0]
    i_t_B3_B3lv_a_i = struct[0].y_run[77,0]
    i_t_B3_B3lv_b_r = struct[0].y_run[78,0]
    i_t_B3_B3lv_b_i = struct[0].y_run[79,0]
    i_t_B3_B3lv_c_r = struct[0].y_run[80,0]
    i_t_B3_B3lv_c_i = struct[0].y_run[81,0]
    i_t_B4_B4lv_a_r = struct[0].y_run[82,0]
    i_t_B4_B4lv_a_i = struct[0].y_run[83,0]
    i_t_B4_B4lv_b_r = struct[0].y_run[84,0]
    i_t_B4_B4lv_b_i = struct[0].y_run[85,0]
    i_t_B4_B4lv_c_r = struct[0].y_run[86,0]
    i_t_B4_B4lv_c_i = struct[0].y_run[87,0]
    i_t_B5_B5lv_a_r = struct[0].y_run[88,0]
    i_t_B5_B5lv_a_i = struct[0].y_run[89,0]
    i_t_B5_B5lv_b_r = struct[0].y_run[90,0]
    i_t_B5_B5lv_b_i = struct[0].y_run[91,0]
    i_t_B5_B5lv_c_r = struct[0].y_run[92,0]
    i_t_B5_B5lv_c_i = struct[0].y_run[93,0]
    i_t_B6_B6lv_a_r = struct[0].y_run[94,0]
    i_t_B6_B6lv_a_i = struct[0].y_run[95,0]
    i_t_B6_B6lv_b_r = struct[0].y_run[96,0]
    i_t_B6_B6lv_b_i = struct[0].y_run[97,0]
    i_t_B6_B6lv_c_r = struct[0].y_run[98,0]
    i_t_B6_B6lv_c_i = struct[0].y_run[99,0]
    i_l_B1_B2_a_r = struct[0].y_run[100,0]
    i_l_B1_B2_a_i = struct[0].y_run[101,0]
    i_l_B1_B2_b_r = struct[0].y_run[102,0]
    i_l_B1_B2_b_i = struct[0].y_run[103,0]
    i_l_B1_B2_c_r = struct[0].y_run[104,0]
    i_l_B1_B2_c_i = struct[0].y_run[105,0]
    i_l_B2_B3_a_r = struct[0].y_run[106,0]
    i_l_B2_B3_a_i = struct[0].y_run[107,0]
    i_l_B2_B3_b_r = struct[0].y_run[108,0]
    i_l_B2_B3_b_i = struct[0].y_run[109,0]
    i_l_B2_B3_c_r = struct[0].y_run[110,0]
    i_l_B2_B3_c_i = struct[0].y_run[111,0]
    i_l_B3_B4_a_r = struct[0].y_run[112,0]
    i_l_B3_B4_a_i = struct[0].y_run[113,0]
    i_l_B3_B4_b_r = struct[0].y_run[114,0]
    i_l_B3_B4_b_i = struct[0].y_run[115,0]
    i_l_B3_B4_c_r = struct[0].y_run[116,0]
    i_l_B3_B4_c_i = struct[0].y_run[117,0]
    i_l_B5_B6_a_r = struct[0].y_run[118,0]
    i_l_B5_B6_a_i = struct[0].y_run[119,0]
    i_l_B5_B6_b_r = struct[0].y_run[120,0]
    i_l_B5_B6_b_i = struct[0].y_run[121,0]
    i_l_B5_B6_c_r = struct[0].y_run[122,0]
    i_l_B5_B6_c_i = struct[0].y_run[123,0]
    i_l_B6_B7_a_r = struct[0].y_run[124,0]
    i_l_B6_B7_a_i = struct[0].y_run[125,0]
    i_l_B6_B7_b_r = struct[0].y_run[126,0]
    i_l_B6_B7_b_i = struct[0].y_run[127,0]
    i_l_B6_B7_c_r = struct[0].y_run[128,0]
    i_l_B6_B7_c_i = struct[0].y_run[129,0]
    i_load_B2lv_a_r = struct[0].y_run[130,0]
    i_load_B2lv_a_i = struct[0].y_run[131,0]
    i_load_B2lv_b_r = struct[0].y_run[132,0]
    i_load_B2lv_b_i = struct[0].y_run[133,0]
    i_load_B2lv_c_r = struct[0].y_run[134,0]
    i_load_B2lv_c_i = struct[0].y_run[135,0]
    i_load_B2lv_n_r = struct[0].y_run[136,0]
    i_load_B2lv_n_i = struct[0].y_run[137,0]
    i_load_B3lv_a_r = struct[0].y_run[138,0]
    i_load_B3lv_a_i = struct[0].y_run[139,0]
    i_load_B3lv_b_r = struct[0].y_run[140,0]
    i_load_B3lv_b_i = struct[0].y_run[141,0]
    i_load_B3lv_c_r = struct[0].y_run[142,0]
    i_load_B3lv_c_i = struct[0].y_run[143,0]
    i_load_B3lv_n_r = struct[0].y_run[144,0]
    i_load_B3lv_n_i = struct[0].y_run[145,0]
    i_load_B4lv_a_r = struct[0].y_run[146,0]
    i_load_B4lv_a_i = struct[0].y_run[147,0]
    i_load_B4lv_b_r = struct[0].y_run[148,0]
    i_load_B4lv_b_i = struct[0].y_run[149,0]
    i_load_B4lv_c_r = struct[0].y_run[150,0]
    i_load_B4lv_c_i = struct[0].y_run[151,0]
    i_load_B4lv_n_r = struct[0].y_run[152,0]
    i_load_B4lv_n_i = struct[0].y_run[153,0]
    i_load_B5lv_a_r = struct[0].y_run[154,0]
    i_load_B5lv_a_i = struct[0].y_run[155,0]
    i_load_B5lv_b_r = struct[0].y_run[156,0]
    i_load_B5lv_b_i = struct[0].y_run[157,0]
    i_load_B5lv_c_r = struct[0].y_run[158,0]
    i_load_B5lv_c_i = struct[0].y_run[159,0]
    i_load_B5lv_n_r = struct[0].y_run[160,0]
    i_load_B5lv_n_i = struct[0].y_run[161,0]
    i_load_B6lv_a_r = struct[0].y_run[162,0]
    i_load_B6lv_a_i = struct[0].y_run[163,0]
    i_load_B6lv_b_r = struct[0].y_run[164,0]
    i_load_B6lv_b_i = struct[0].y_run[165,0]
    i_load_B6lv_c_r = struct[0].y_run[166,0]
    i_load_B6lv_c_i = struct[0].y_run[167,0]
    i_load_B6lv_n_r = struct[0].y_run[168,0]
    i_load_B6lv_n_i = struct[0].y_run[169,0]
    
    # Differential equations:
    if mode == 2:


        struct[0].f[0,0] = u_dummy - x_dummy
    
    # Algebraic equations:
    if mode == 3:


        struct[0].g[0,0] = i_load_B2lv_a_r + 0.996212229189942*v_B2_a_i + 0.249053057297486*v_B2_a_r - 0.996212229189942*v_B2_c_i - 0.249053057297486*v_B2_c_r - 23.0065359477124*v_B2lv_a_i - 5.7516339869281*v_B2lv_a_r + 23.0065359477124*v_B2lv_n_i + 5.7516339869281*v_B2lv_n_r
        struct[0].g[1,0] = i_load_B2lv_a_i + 0.249053057297486*v_B2_a_i - 0.996212229189942*v_B2_a_r - 0.249053057297486*v_B2_c_i + 0.996212229189942*v_B2_c_r - 5.7516339869281*v_B2lv_a_i + 23.0065359477124*v_B2lv_a_r + 5.7516339869281*v_B2lv_n_i - 23.0065359477124*v_B2lv_n_r
        struct[0].g[2,0] = i_load_B2lv_b_r - 0.996212229189942*v_B2_a_i - 0.249053057297486*v_B2_a_r + 0.996212229189942*v_B2_b_i + 0.249053057297486*v_B2_b_r - 23.0065359477124*v_B2lv_b_i - 5.7516339869281*v_B2lv_b_r + 23.0065359477124*v_B2lv_n_i + 5.7516339869281*v_B2lv_n_r
        struct[0].g[3,0] = i_load_B2lv_b_i - 0.249053057297486*v_B2_a_i + 0.996212229189942*v_B2_a_r + 0.249053057297486*v_B2_b_i - 0.996212229189942*v_B2_b_r - 5.7516339869281*v_B2lv_b_i + 23.0065359477124*v_B2lv_b_r + 5.7516339869281*v_B2lv_n_i - 23.0065359477124*v_B2lv_n_r
        struct[0].g[4,0] = i_load_B2lv_c_r - 0.996212229189942*v_B2_b_i - 0.249053057297486*v_B2_b_r + 0.996212229189942*v_B2_c_i + 0.249053057297486*v_B2_c_r - 23.0065359477124*v_B2lv_c_i - 5.7516339869281*v_B2lv_c_r + 23.0065359477124*v_B2lv_n_i + 5.7516339869281*v_B2lv_n_r
        struct[0].g[5,0] = i_load_B2lv_c_i - 0.249053057297486*v_B2_b_i + 0.996212229189942*v_B2_b_r + 0.249053057297486*v_B2_c_i - 0.996212229189942*v_B2_c_r - 5.7516339869281*v_B2lv_c_i + 23.0065359477124*v_B2lv_c_r + 5.7516339869281*v_B2lv_n_i - 23.0065359477124*v_B2lv_n_r
        struct[0].g[6,0] = 23.0065359477124*v_B2lv_a_i + 5.7516339869281*v_B2lv_a_r + 23.0065359477124*v_B2lv_b_i + 5.7516339869281*v_B2lv_b_r + 23.0065359477124*v_B2lv_c_i + 5.7516339869281*v_B2lv_c_r - 69.0196078431372*v_B2lv_n_i - 1017.25490196078*v_B2lv_n_r
        struct[0].g[7,0] = 5.7516339869281*v_B2lv_a_i - 23.0065359477124*v_B2lv_a_r + 5.7516339869281*v_B2lv_b_i - 23.0065359477124*v_B2lv_b_r + 5.7516339869281*v_B2lv_c_i - 23.0065359477124*v_B2lv_c_r - 1017.25490196078*v_B2lv_n_i + 69.0196078431372*v_B2lv_n_r
        struct[0].g[8,0] = i_load_B3lv_a_r + 0.996212229189942*v_B3_a_i + 0.249053057297486*v_B3_a_r - 0.996212229189942*v_B3_c_i - 0.249053057297486*v_B3_c_r - 23.0065359477124*v_B3lv_a_i - 5.7516339869281*v_B3lv_a_r + 23.0065359477124*v_B3lv_n_i + 5.7516339869281*v_B3lv_n_r
        struct[0].g[9,0] = i_load_B3lv_a_i + 0.249053057297486*v_B3_a_i - 0.996212229189942*v_B3_a_r - 0.249053057297486*v_B3_c_i + 0.996212229189942*v_B3_c_r - 5.7516339869281*v_B3lv_a_i + 23.0065359477124*v_B3lv_a_r + 5.7516339869281*v_B3lv_n_i - 23.0065359477124*v_B3lv_n_r
        struct[0].g[10,0] = i_load_B3lv_b_r - 0.996212229189942*v_B3_a_i - 0.249053057297486*v_B3_a_r + 0.996212229189942*v_B3_b_i + 0.249053057297486*v_B3_b_r - 23.0065359477124*v_B3lv_b_i - 5.7516339869281*v_B3lv_b_r + 23.0065359477124*v_B3lv_n_i + 5.7516339869281*v_B3lv_n_r
        struct[0].g[11,0] = i_load_B3lv_b_i - 0.249053057297486*v_B3_a_i + 0.996212229189942*v_B3_a_r + 0.249053057297486*v_B3_b_i - 0.996212229189942*v_B3_b_r - 5.7516339869281*v_B3lv_b_i + 23.0065359477124*v_B3lv_b_r + 5.7516339869281*v_B3lv_n_i - 23.0065359477124*v_B3lv_n_r
        struct[0].g[12,0] = i_load_B3lv_c_r - 0.996212229189942*v_B3_b_i - 0.249053057297486*v_B3_b_r + 0.996212229189942*v_B3_c_i + 0.249053057297486*v_B3_c_r - 23.0065359477124*v_B3lv_c_i - 5.7516339869281*v_B3lv_c_r + 23.0065359477124*v_B3lv_n_i + 5.7516339869281*v_B3lv_n_r
        struct[0].g[13,0] = i_load_B3lv_c_i - 0.249053057297486*v_B3_b_i + 0.996212229189942*v_B3_b_r + 0.249053057297486*v_B3_c_i - 0.996212229189942*v_B3_c_r - 5.7516339869281*v_B3lv_c_i + 23.0065359477124*v_B3lv_c_r + 5.7516339869281*v_B3lv_n_i - 23.0065359477124*v_B3lv_n_r
        struct[0].g[14,0] = 23.0065359477124*v_B3lv_a_i + 5.7516339869281*v_B3lv_a_r + 23.0065359477124*v_B3lv_b_i + 5.7516339869281*v_B3lv_b_r + 23.0065359477124*v_B3lv_c_i + 5.7516339869281*v_B3lv_c_r - 69.0196078431372*v_B3lv_n_i - 1017.25490196078*v_B3lv_n_r
        struct[0].g[15,0] = 5.7516339869281*v_B3lv_a_i - 23.0065359477124*v_B3lv_a_r + 5.7516339869281*v_B3lv_b_i - 23.0065359477124*v_B3lv_b_r + 5.7516339869281*v_B3lv_c_i - 23.0065359477124*v_B3lv_c_r - 1017.25490196078*v_B3lv_n_i + 69.0196078431372*v_B3lv_n_r
        struct[0].g[16,0] = i_load_B4lv_a_r + 0.996212229189942*v_B4_a_i + 0.249053057297486*v_B4_a_r - 0.996212229189942*v_B4_c_i - 0.249053057297486*v_B4_c_r - 23.0065359477124*v_B4lv_a_i - 5.7516339869281*v_B4lv_a_r + 23.0065359477124*v_B4lv_n_i + 5.7516339869281*v_B4lv_n_r
        struct[0].g[17,0] = i_load_B4lv_a_i + 0.249053057297486*v_B4_a_i - 0.996212229189942*v_B4_a_r - 0.249053057297486*v_B4_c_i + 0.996212229189942*v_B4_c_r - 5.7516339869281*v_B4lv_a_i + 23.0065359477124*v_B4lv_a_r + 5.7516339869281*v_B4lv_n_i - 23.0065359477124*v_B4lv_n_r
        struct[0].g[18,0] = i_load_B4lv_b_r - 0.996212229189942*v_B4_a_i - 0.249053057297486*v_B4_a_r + 0.996212229189942*v_B4_b_i + 0.249053057297486*v_B4_b_r - 23.0065359477124*v_B4lv_b_i - 5.7516339869281*v_B4lv_b_r + 23.0065359477124*v_B4lv_n_i + 5.7516339869281*v_B4lv_n_r
        struct[0].g[19,0] = i_load_B4lv_b_i - 0.249053057297486*v_B4_a_i + 0.996212229189942*v_B4_a_r + 0.249053057297486*v_B4_b_i - 0.996212229189942*v_B4_b_r - 5.7516339869281*v_B4lv_b_i + 23.0065359477124*v_B4lv_b_r + 5.7516339869281*v_B4lv_n_i - 23.0065359477124*v_B4lv_n_r
        struct[0].g[20,0] = i_load_B4lv_c_r - 0.996212229189942*v_B4_b_i - 0.249053057297486*v_B4_b_r + 0.996212229189942*v_B4_c_i + 0.249053057297486*v_B4_c_r - 23.0065359477124*v_B4lv_c_i - 5.7516339869281*v_B4lv_c_r + 23.0065359477124*v_B4lv_n_i + 5.7516339869281*v_B4lv_n_r
        struct[0].g[21,0] = i_load_B4lv_c_i - 0.249053057297486*v_B4_b_i + 0.996212229189942*v_B4_b_r + 0.249053057297486*v_B4_c_i - 0.996212229189942*v_B4_c_r - 5.7516339869281*v_B4lv_c_i + 23.0065359477124*v_B4lv_c_r + 5.7516339869281*v_B4lv_n_i - 23.0065359477124*v_B4lv_n_r
        struct[0].g[22,0] = 23.0065359477124*v_B4lv_a_i + 5.7516339869281*v_B4lv_a_r + 23.0065359477124*v_B4lv_b_i + 5.7516339869281*v_B4lv_b_r + 23.0065359477124*v_B4lv_c_i + 5.7516339869281*v_B4lv_c_r - 69.0196078431372*v_B4lv_n_i - 1017.25490196078*v_B4lv_n_r
        struct[0].g[23,0] = 5.7516339869281*v_B4lv_a_i - 23.0065359477124*v_B4lv_a_r + 5.7516339869281*v_B4lv_b_i - 23.0065359477124*v_B4lv_b_r + 5.7516339869281*v_B4lv_c_i - 23.0065359477124*v_B4lv_c_r - 1017.25490196078*v_B4lv_n_i + 69.0196078431372*v_B4lv_n_r
        struct[0].g[24,0] = i_load_B5lv_a_r + 0.996212229189942*v_B5_a_i + 0.249053057297486*v_B5_a_r - 0.996212229189942*v_B5_c_i - 0.249053057297486*v_B5_c_r - 23.0065359477124*v_B5lv_a_i - 5.7516339869281*v_B5lv_a_r + 23.0065359477124*v_B5lv_n_i + 5.7516339869281*v_B5lv_n_r
        struct[0].g[25,0] = i_load_B5lv_a_i + 0.249053057297486*v_B5_a_i - 0.996212229189942*v_B5_a_r - 0.249053057297486*v_B5_c_i + 0.996212229189942*v_B5_c_r - 5.7516339869281*v_B5lv_a_i + 23.0065359477124*v_B5lv_a_r + 5.7516339869281*v_B5lv_n_i - 23.0065359477124*v_B5lv_n_r
        struct[0].g[26,0] = i_load_B5lv_b_r - 0.996212229189942*v_B5_a_i - 0.249053057297486*v_B5_a_r + 0.996212229189942*v_B5_b_i + 0.249053057297486*v_B5_b_r - 23.0065359477124*v_B5lv_b_i - 5.7516339869281*v_B5lv_b_r + 23.0065359477124*v_B5lv_n_i + 5.7516339869281*v_B5lv_n_r
        struct[0].g[27,0] = i_load_B5lv_b_i - 0.249053057297486*v_B5_a_i + 0.996212229189942*v_B5_a_r + 0.249053057297486*v_B5_b_i - 0.996212229189942*v_B5_b_r - 5.7516339869281*v_B5lv_b_i + 23.0065359477124*v_B5lv_b_r + 5.7516339869281*v_B5lv_n_i - 23.0065359477124*v_B5lv_n_r
        struct[0].g[28,0] = i_load_B5lv_c_r - 0.996212229189942*v_B5_b_i - 0.249053057297486*v_B5_b_r + 0.996212229189942*v_B5_c_i + 0.249053057297486*v_B5_c_r - 23.0065359477124*v_B5lv_c_i - 5.7516339869281*v_B5lv_c_r + 23.0065359477124*v_B5lv_n_i + 5.7516339869281*v_B5lv_n_r
        struct[0].g[29,0] = i_load_B5lv_c_i - 0.249053057297486*v_B5_b_i + 0.996212229189942*v_B5_b_r + 0.249053057297486*v_B5_c_i - 0.996212229189942*v_B5_c_r - 5.7516339869281*v_B5lv_c_i + 23.0065359477124*v_B5lv_c_r + 5.7516339869281*v_B5lv_n_i - 23.0065359477124*v_B5lv_n_r
        struct[0].g[30,0] = 23.0065359477124*v_B5lv_a_i + 5.7516339869281*v_B5lv_a_r + 23.0065359477124*v_B5lv_b_i + 5.7516339869281*v_B5lv_b_r + 23.0065359477124*v_B5lv_c_i + 5.7516339869281*v_B5lv_c_r - 69.0196078431372*v_B5lv_n_i - 1017.25490196078*v_B5lv_n_r
        struct[0].g[31,0] = 5.7516339869281*v_B5lv_a_i - 23.0065359477124*v_B5lv_a_r + 5.7516339869281*v_B5lv_b_i - 23.0065359477124*v_B5lv_b_r + 5.7516339869281*v_B5lv_c_i - 23.0065359477124*v_B5lv_c_r - 1017.25490196078*v_B5lv_n_i + 69.0196078431372*v_B5lv_n_r
        struct[0].g[32,0] = i_load_B6lv_a_r + 0.996212229189942*v_B6_a_i + 0.249053057297486*v_B6_a_r - 0.996212229189942*v_B6_c_i - 0.249053057297486*v_B6_c_r - 23.0065359477124*v_B6lv_a_i - 5.7516339869281*v_B6lv_a_r + 23.0065359477124*v_B6lv_n_i + 5.7516339869281*v_B6lv_n_r
        struct[0].g[33,0] = i_load_B6lv_a_i + 0.249053057297486*v_B6_a_i - 0.996212229189942*v_B6_a_r - 0.249053057297486*v_B6_c_i + 0.996212229189942*v_B6_c_r - 5.7516339869281*v_B6lv_a_i + 23.0065359477124*v_B6lv_a_r + 5.7516339869281*v_B6lv_n_i - 23.0065359477124*v_B6lv_n_r
        struct[0].g[34,0] = i_load_B6lv_b_r - 0.996212229189942*v_B6_a_i - 0.249053057297486*v_B6_a_r + 0.996212229189942*v_B6_b_i + 0.249053057297486*v_B6_b_r - 23.0065359477124*v_B6lv_b_i - 5.7516339869281*v_B6lv_b_r + 23.0065359477124*v_B6lv_n_i + 5.7516339869281*v_B6lv_n_r
        struct[0].g[35,0] = i_load_B6lv_b_i - 0.249053057297486*v_B6_a_i + 0.996212229189942*v_B6_a_r + 0.249053057297486*v_B6_b_i - 0.996212229189942*v_B6_b_r - 5.7516339869281*v_B6lv_b_i + 23.0065359477124*v_B6lv_b_r + 5.7516339869281*v_B6lv_n_i - 23.0065359477124*v_B6lv_n_r
        struct[0].g[36,0] = i_load_B6lv_c_r - 0.996212229189942*v_B6_b_i - 0.249053057297486*v_B6_b_r + 0.996212229189942*v_B6_c_i + 0.249053057297486*v_B6_c_r - 23.0065359477124*v_B6lv_c_i - 5.7516339869281*v_B6lv_c_r + 23.0065359477124*v_B6lv_n_i + 5.7516339869281*v_B6lv_n_r
        struct[0].g[37,0] = i_load_B6lv_c_i - 0.249053057297486*v_B6_b_i + 0.996212229189942*v_B6_b_r + 0.249053057297486*v_B6_c_i - 0.996212229189942*v_B6_c_r - 5.7516339869281*v_B6lv_c_i + 23.0065359477124*v_B6lv_c_r + 5.7516339869281*v_B6lv_n_i - 23.0065359477124*v_B6lv_n_r
        struct[0].g[38,0] = 23.0065359477124*v_B6lv_a_i + 5.7516339869281*v_B6lv_a_r + 23.0065359477124*v_B6lv_b_i + 5.7516339869281*v_B6lv_b_r + 23.0065359477124*v_B6lv_c_i + 5.7516339869281*v_B6lv_c_r - 69.0196078431372*v_B6lv_n_i - 1017.25490196078*v_B6lv_n_r
        struct[0].g[39,0] = 5.7516339869281*v_B6lv_a_i - 23.0065359477124*v_B6lv_a_r + 5.7516339869281*v_B6lv_b_i - 23.0065359477124*v_B6lv_b_r + 5.7516339869281*v_B6lv_c_i - 23.0065359477124*v_B6lv_c_r - 1017.25490196078*v_B6lv_n_i + 69.0196078431372*v_B6lv_n_r
        struct[0].g[40,0] = 0.598820527961361*v_B1_a_i + 1.10755301189314*v_B1_a_r - 0.171091579417532*v_B1_b_i - 0.316443717683753*v_B1_b_r - 0.171091579417532*v_B1_c_i - 0.316443717683753*v_B1_c_r - 1.28353302446119*v_B2_a_i - 2.23667465123725*v_B2_a_r + 0.385473430243205*v_B2_b_i + 0.643671749092996*v_B2_b_r + 0.385473430243205*v_B2_c_i + 0.643671749092997*v_B2_c_r + 0.996212229189942*v_B2lv_a_i + 0.249053057297486*v_B2lv_a_r - 0.996212229189942*v_B2lv_b_i - 0.249053057297486*v_B2lv_b_r + 0.598820527961361*v_B3_a_i + 1.10755301189314*v_B3_a_r - 0.171091579417532*v_B3_b_i - 0.316443717683753*v_B3_b_r - 0.171091579417532*v_B3_c_i - 0.316443717683753*v_B3_c_r
        struct[0].g[41,0] = 1.10755301189314*v_B1_a_i - 0.598820527961361*v_B1_a_r - 0.316443717683753*v_B1_b_i + 0.171091579417532*v_B1_b_r - 0.316443717683753*v_B1_c_i + 0.171091579417532*v_B1_c_r - 2.23667465123725*v_B2_a_i + 1.28353302446119*v_B2_a_r + 0.643671749092996*v_B2_b_i - 0.385473430243205*v_B2_b_r + 0.643671749092997*v_B2_c_i - 0.385473430243205*v_B2_c_r + 0.249053057297486*v_B2lv_a_i - 0.996212229189942*v_B2lv_a_r - 0.249053057297486*v_B2lv_b_i + 0.996212229189942*v_B2lv_b_r + 1.10755301189314*v_B3_a_i - 0.598820527961361*v_B3_a_r - 0.316443717683753*v_B3_b_i + 0.171091579417532*v_B3_b_r - 0.316443717683753*v_B3_c_i + 0.171091579417532*v_B3_c_r
        struct[0].g[42,0] = -0.171091579417532*v_B1_a_i - 0.316443717683753*v_B1_a_r + 0.59882052796136*v_B1_b_i + 1.10755301189314*v_B1_b_r - 0.171091579417531*v_B1_c_i - 0.316443717683753*v_B1_c_r + 0.385473430243205*v_B2_a_i + 0.643671749092996*v_B2_a_r - 1.28353302446119*v_B2_b_i - 2.23667465123725*v_B2_b_r + 0.385473430243204*v_B2_c_i + 0.643671749092997*v_B2_c_r + 0.996212229189942*v_B2lv_b_i + 0.249053057297486*v_B2lv_b_r - 0.996212229189942*v_B2lv_c_i - 0.249053057297486*v_B2lv_c_r - 0.171091579417532*v_B3_a_i - 0.316443717683753*v_B3_a_r + 0.59882052796136*v_B3_b_i + 1.10755301189314*v_B3_b_r - 0.171091579417531*v_B3_c_i - 0.316443717683753*v_B3_c_r
        struct[0].g[43,0] = -0.316443717683753*v_B1_a_i + 0.171091579417532*v_B1_a_r + 1.10755301189314*v_B1_b_i - 0.59882052796136*v_B1_b_r - 0.316443717683753*v_B1_c_i + 0.171091579417531*v_B1_c_r + 0.643671749092996*v_B2_a_i - 0.385473430243205*v_B2_a_r - 2.23667465123725*v_B2_b_i + 1.28353302446119*v_B2_b_r + 0.643671749092997*v_B2_c_i - 0.385473430243204*v_B2_c_r + 0.249053057297486*v_B2lv_b_i - 0.996212229189942*v_B2lv_b_r - 0.249053057297486*v_B2lv_c_i + 0.996212229189942*v_B2lv_c_r - 0.316443717683753*v_B3_a_i + 0.171091579417532*v_B3_a_r + 1.10755301189314*v_B3_b_i - 0.59882052796136*v_B3_b_r - 0.316443717683753*v_B3_c_i + 0.171091579417531*v_B3_c_r
        struct[0].g[44,0] = -0.171091579417532*v_B1_a_i - 0.316443717683753*v_B1_a_r - 0.171091579417531*v_B1_b_i - 0.316443717683753*v_B1_b_r + 0.59882052796136*v_B1_c_i + 1.10755301189314*v_B1_c_r + 0.385473430243205*v_B2_a_i + 0.643671749092997*v_B2_a_r + 0.385473430243204*v_B2_b_i + 0.643671749092997*v_B2_b_r - 1.28353302446119*v_B2_c_i - 2.23667465123725*v_B2_c_r - 0.996212229189942*v_B2lv_a_i - 0.249053057297486*v_B2lv_a_r + 0.996212229189942*v_B2lv_c_i + 0.249053057297486*v_B2lv_c_r - 0.171091579417532*v_B3_a_i - 0.316443717683753*v_B3_a_r - 0.171091579417531*v_B3_b_i - 0.316443717683753*v_B3_b_r + 0.59882052796136*v_B3_c_i + 1.10755301189314*v_B3_c_r
        struct[0].g[45,0] = -0.316443717683753*v_B1_a_i + 0.171091579417532*v_B1_a_r - 0.316443717683753*v_B1_b_i + 0.171091579417531*v_B1_b_r + 1.10755301189314*v_B1_c_i - 0.59882052796136*v_B1_c_r + 0.643671749092997*v_B2_a_i - 0.385473430243205*v_B2_a_r + 0.643671749092997*v_B2_b_i - 0.385473430243204*v_B2_b_r - 2.23667465123725*v_B2_c_i + 1.28353302446119*v_B2_c_r - 0.249053057297486*v_B2lv_a_i + 0.996212229189942*v_B2lv_a_r + 0.249053057297486*v_B2lv_c_i - 0.996212229189942*v_B2lv_c_r - 0.316443717683753*v_B3_a_i + 0.171091579417532*v_B3_a_r - 0.316443717683753*v_B3_b_i + 0.171091579417531*v_B3_b_r + 1.10755301189314*v_B3_c_i - 0.59882052796136*v_B3_c_r
        struct[0].g[46,0] = 0.598820527961361*v_B2_a_i + 1.10755301189314*v_B2_a_r - 0.171091579417532*v_B2_b_i - 0.316443717683753*v_B2_b_r - 0.171091579417532*v_B2_c_i - 0.316443717683753*v_B2_c_r - 1.28353302446119*v_B3_a_i - 2.23667465123725*v_B3_a_r + 0.385473430243205*v_B3_b_i + 0.643671749092996*v_B3_b_r + 0.385473430243205*v_B3_c_i + 0.643671749092997*v_B3_c_r + 0.996212229189942*v_B3lv_a_i + 0.249053057297486*v_B3lv_a_r - 0.996212229189942*v_B3lv_b_i - 0.249053057297486*v_B3lv_b_r + 0.598820527961361*v_B4_a_i + 1.10755301189314*v_B4_a_r - 0.171091579417532*v_B4_b_i - 0.316443717683753*v_B4_b_r - 0.171091579417532*v_B4_c_i - 0.316443717683753*v_B4_c_r
        struct[0].g[47,0] = 1.10755301189314*v_B2_a_i - 0.598820527961361*v_B2_a_r - 0.316443717683753*v_B2_b_i + 0.171091579417532*v_B2_b_r - 0.316443717683753*v_B2_c_i + 0.171091579417532*v_B2_c_r - 2.23667465123725*v_B3_a_i + 1.28353302446119*v_B3_a_r + 0.643671749092996*v_B3_b_i - 0.385473430243205*v_B3_b_r + 0.643671749092997*v_B3_c_i - 0.385473430243205*v_B3_c_r + 0.249053057297486*v_B3lv_a_i - 0.996212229189942*v_B3lv_a_r - 0.249053057297486*v_B3lv_b_i + 0.996212229189942*v_B3lv_b_r + 1.10755301189314*v_B4_a_i - 0.598820527961361*v_B4_a_r - 0.316443717683753*v_B4_b_i + 0.171091579417532*v_B4_b_r - 0.316443717683753*v_B4_c_i + 0.171091579417532*v_B4_c_r
        struct[0].g[48,0] = -0.171091579417532*v_B2_a_i - 0.316443717683753*v_B2_a_r + 0.59882052796136*v_B2_b_i + 1.10755301189314*v_B2_b_r - 0.171091579417531*v_B2_c_i - 0.316443717683753*v_B2_c_r + 0.385473430243205*v_B3_a_i + 0.643671749092996*v_B3_a_r - 1.28353302446119*v_B3_b_i - 2.23667465123725*v_B3_b_r + 0.385473430243204*v_B3_c_i + 0.643671749092997*v_B3_c_r + 0.996212229189942*v_B3lv_b_i + 0.249053057297486*v_B3lv_b_r - 0.996212229189942*v_B3lv_c_i - 0.249053057297486*v_B3lv_c_r - 0.171091579417532*v_B4_a_i - 0.316443717683753*v_B4_a_r + 0.59882052796136*v_B4_b_i + 1.10755301189314*v_B4_b_r - 0.171091579417531*v_B4_c_i - 0.316443717683753*v_B4_c_r
        struct[0].g[49,0] = -0.316443717683753*v_B2_a_i + 0.171091579417532*v_B2_a_r + 1.10755301189314*v_B2_b_i - 0.59882052796136*v_B2_b_r - 0.316443717683753*v_B2_c_i + 0.171091579417531*v_B2_c_r + 0.643671749092996*v_B3_a_i - 0.385473430243205*v_B3_a_r - 2.23667465123725*v_B3_b_i + 1.28353302446119*v_B3_b_r + 0.643671749092997*v_B3_c_i - 0.385473430243204*v_B3_c_r + 0.249053057297486*v_B3lv_b_i - 0.996212229189942*v_B3lv_b_r - 0.249053057297486*v_B3lv_c_i + 0.996212229189942*v_B3lv_c_r - 0.316443717683753*v_B4_a_i + 0.171091579417532*v_B4_a_r + 1.10755301189314*v_B4_b_i - 0.59882052796136*v_B4_b_r - 0.316443717683753*v_B4_c_i + 0.171091579417531*v_B4_c_r
        struct[0].g[50,0] = -0.171091579417532*v_B2_a_i - 0.316443717683753*v_B2_a_r - 0.171091579417531*v_B2_b_i - 0.316443717683753*v_B2_b_r + 0.59882052796136*v_B2_c_i + 1.10755301189314*v_B2_c_r + 0.385473430243205*v_B3_a_i + 0.643671749092997*v_B3_a_r + 0.385473430243204*v_B3_b_i + 0.643671749092997*v_B3_b_r - 1.28353302446119*v_B3_c_i - 2.23667465123725*v_B3_c_r - 0.996212229189942*v_B3lv_a_i - 0.249053057297486*v_B3lv_a_r + 0.996212229189942*v_B3lv_c_i + 0.249053057297486*v_B3lv_c_r - 0.171091579417532*v_B4_a_i - 0.316443717683753*v_B4_a_r - 0.171091579417531*v_B4_b_i - 0.316443717683753*v_B4_b_r + 0.59882052796136*v_B4_c_i + 1.10755301189314*v_B4_c_r
        struct[0].g[51,0] = -0.316443717683753*v_B2_a_i + 0.171091579417532*v_B2_a_r - 0.316443717683753*v_B2_b_i + 0.171091579417531*v_B2_b_r + 1.10755301189314*v_B2_c_i - 0.59882052796136*v_B2_c_r + 0.643671749092997*v_B3_a_i - 0.385473430243205*v_B3_a_r + 0.643671749092997*v_B3_b_i - 0.385473430243204*v_B3_b_r - 2.23667465123725*v_B3_c_i + 1.28353302446119*v_B3_c_r - 0.249053057297486*v_B3lv_a_i + 0.996212229189942*v_B3lv_a_r + 0.249053057297486*v_B3lv_c_i - 0.996212229189942*v_B3lv_c_r - 0.316443717683753*v_B4_a_i + 0.171091579417532*v_B4_a_r - 0.316443717683753*v_B4_b_i + 0.171091579417531*v_B4_b_r + 1.10755301189314*v_B4_c_i - 0.59882052796136*v_B4_c_r
        struct[0].g[52,0] = 0.598820527961361*v_B3_a_i + 1.10755301189314*v_B3_a_r - 0.171091579417532*v_B3_b_i - 0.316443717683753*v_B3_b_r - 0.171091579417532*v_B3_c_i - 0.316443717683753*v_B3_c_r - 0.684903767132556*v_B4_a_i - 1.12912163934412*v_B4_a_r + 0.214305342572583*v_B4_b_i + 0.327228031409243*v_B4_b_r + 0.214305342572583*v_B4_c_i + 0.327228031409244*v_B4_c_r + 0.996212229189942*v_B4lv_a_i + 0.249053057297486*v_B4lv_a_r - 0.996212229189942*v_B4lv_b_i - 0.249053057297486*v_B4lv_b_r
        struct[0].g[53,0] = 1.10755301189314*v_B3_a_i - 0.598820527961361*v_B3_a_r - 0.316443717683753*v_B3_b_i + 0.171091579417532*v_B3_b_r - 0.316443717683753*v_B3_c_i + 0.171091579417532*v_B3_c_r - 1.12912163934412*v_B4_a_i + 0.684903767132556*v_B4_a_r + 0.327228031409243*v_B4_b_i - 0.214305342572583*v_B4_b_r + 0.327228031409244*v_B4_c_i - 0.214305342572583*v_B4_c_r + 0.249053057297486*v_B4lv_a_i - 0.996212229189942*v_B4lv_a_r - 0.249053057297486*v_B4lv_b_i + 0.996212229189942*v_B4lv_b_r
        struct[0].g[54,0] = -0.171091579417532*v_B3_a_i - 0.316443717683753*v_B3_a_r + 0.59882052796136*v_B3_b_i + 1.10755301189314*v_B3_b_r - 0.171091579417531*v_B3_c_i - 0.316443717683753*v_B3_c_r + 0.214305342572583*v_B4_a_i + 0.327228031409243*v_B4_a_r - 0.684903767132556*v_B4_b_i - 1.12912163934412*v_B4_b_r + 0.214305342572582*v_B4_c_i + 0.327228031409244*v_B4_c_r + 0.996212229189942*v_B4lv_b_i + 0.249053057297486*v_B4lv_b_r - 0.996212229189942*v_B4lv_c_i - 0.249053057297486*v_B4lv_c_r
        struct[0].g[55,0] = -0.316443717683753*v_B3_a_i + 0.171091579417532*v_B3_a_r + 1.10755301189314*v_B3_b_i - 0.59882052796136*v_B3_b_r - 0.316443717683753*v_B3_c_i + 0.171091579417531*v_B3_c_r + 0.327228031409243*v_B4_a_i - 0.214305342572583*v_B4_a_r - 1.12912163934412*v_B4_b_i + 0.684903767132556*v_B4_b_r + 0.327228031409244*v_B4_c_i - 0.214305342572582*v_B4_c_r + 0.249053057297486*v_B4lv_b_i - 0.996212229189942*v_B4lv_b_r - 0.249053057297486*v_B4lv_c_i + 0.996212229189942*v_B4lv_c_r
        struct[0].g[56,0] = -0.171091579417532*v_B3_a_i - 0.316443717683753*v_B3_a_r - 0.171091579417531*v_B3_b_i - 0.316443717683753*v_B3_b_r + 0.59882052796136*v_B3_c_i + 1.10755301189314*v_B3_c_r + 0.214305342572583*v_B4_a_i + 0.327228031409243*v_B4_a_r + 0.214305342572582*v_B4_b_i + 0.327228031409244*v_B4_b_r - 0.684903767132556*v_B4_c_i - 1.12912163934412*v_B4_c_r - 0.996212229189942*v_B4lv_a_i - 0.249053057297486*v_B4lv_a_r + 0.996212229189942*v_B4lv_c_i + 0.249053057297486*v_B4lv_c_r
        struct[0].g[57,0] = -0.316443717683753*v_B3_a_i + 0.171091579417532*v_B3_a_r - 0.316443717683753*v_B3_b_i + 0.171091579417531*v_B3_b_r + 1.10755301189314*v_B3_c_i - 0.59882052796136*v_B3_c_r + 0.327228031409243*v_B4_a_i - 0.214305342572583*v_B4_a_r + 0.327228031409244*v_B4_b_i - 0.214305342572582*v_B4_b_r - 1.12912163934412*v_B4_c_i + 0.684903767132556*v_B4_c_r - 0.249053057297486*v_B4lv_a_i + 0.996212229189942*v_B4lv_a_r + 0.249053057297486*v_B4lv_c_i - 0.996212229189942*v_B4lv_c_r
        struct[0].g[58,0] = -0.684903767132556*v_B5_a_i - 1.12912163934412*v_B5_a_r + 0.214305342572583*v_B5_b_i + 0.327228031409243*v_B5_b_r + 0.214305342572583*v_B5_c_i + 0.327228031409244*v_B5_c_r + 0.996212229189942*v_B5lv_a_i + 0.249053057297486*v_B5lv_a_r - 0.996212229189942*v_B5lv_b_i - 0.249053057297486*v_B5lv_b_r + 0.598820527961361*v_B6_a_i + 1.10755301189314*v_B6_a_r - 0.171091579417532*v_B6_b_i - 0.316443717683753*v_B6_b_r - 0.171091579417532*v_B6_c_i - 0.316443717683753*v_B6_c_r
        struct[0].g[59,0] = -1.12912163934412*v_B5_a_i + 0.684903767132556*v_B5_a_r + 0.327228031409243*v_B5_b_i - 0.214305342572583*v_B5_b_r + 0.327228031409244*v_B5_c_i - 0.214305342572583*v_B5_c_r + 0.249053057297486*v_B5lv_a_i - 0.996212229189942*v_B5lv_a_r - 0.249053057297486*v_B5lv_b_i + 0.996212229189942*v_B5lv_b_r + 1.10755301189314*v_B6_a_i - 0.598820527961361*v_B6_a_r - 0.316443717683753*v_B6_b_i + 0.171091579417532*v_B6_b_r - 0.316443717683753*v_B6_c_i + 0.171091579417532*v_B6_c_r
        struct[0].g[60,0] = 0.214305342572583*v_B5_a_i + 0.327228031409243*v_B5_a_r - 0.684903767132556*v_B5_b_i - 1.12912163934412*v_B5_b_r + 0.214305342572582*v_B5_c_i + 0.327228031409244*v_B5_c_r + 0.996212229189942*v_B5lv_b_i + 0.249053057297486*v_B5lv_b_r - 0.996212229189942*v_B5lv_c_i - 0.249053057297486*v_B5lv_c_r - 0.171091579417532*v_B6_a_i - 0.316443717683753*v_B6_a_r + 0.59882052796136*v_B6_b_i + 1.10755301189314*v_B6_b_r - 0.171091579417531*v_B6_c_i - 0.316443717683753*v_B6_c_r
        struct[0].g[61,0] = 0.327228031409243*v_B5_a_i - 0.214305342572583*v_B5_a_r - 1.12912163934412*v_B5_b_i + 0.684903767132556*v_B5_b_r + 0.327228031409244*v_B5_c_i - 0.214305342572582*v_B5_c_r + 0.249053057297486*v_B5lv_b_i - 0.996212229189942*v_B5lv_b_r - 0.249053057297486*v_B5lv_c_i + 0.996212229189942*v_B5lv_c_r - 0.316443717683753*v_B6_a_i + 0.171091579417532*v_B6_a_r + 1.10755301189314*v_B6_b_i - 0.59882052796136*v_B6_b_r - 0.316443717683753*v_B6_c_i + 0.171091579417531*v_B6_c_r
        struct[0].g[62,0] = 0.214305342572583*v_B5_a_i + 0.327228031409243*v_B5_a_r + 0.214305342572582*v_B5_b_i + 0.327228031409244*v_B5_b_r - 0.684903767132556*v_B5_c_i - 1.12912163934412*v_B5_c_r - 0.996212229189942*v_B5lv_a_i - 0.249053057297486*v_B5lv_a_r + 0.996212229189942*v_B5lv_c_i + 0.249053057297486*v_B5lv_c_r - 0.171091579417532*v_B6_a_i - 0.316443717683753*v_B6_a_r - 0.171091579417531*v_B6_b_i - 0.316443717683753*v_B6_b_r + 0.59882052796136*v_B6_c_i + 1.10755301189314*v_B6_c_r
        struct[0].g[63,0] = 0.327228031409243*v_B5_a_i - 0.214305342572583*v_B5_a_r + 0.327228031409244*v_B5_b_i - 0.214305342572582*v_B5_b_r - 1.12912163934412*v_B5_c_i + 0.684903767132556*v_B5_c_r - 0.249053057297486*v_B5lv_a_i + 0.996212229189942*v_B5lv_a_r + 0.249053057297486*v_B5lv_c_i - 0.996212229189942*v_B5lv_c_r - 0.316443717683753*v_B6_a_i + 0.171091579417532*v_B6_a_r - 0.316443717683753*v_B6_b_i + 0.171091579417531*v_B6_b_r + 1.10755301189314*v_B6_c_i - 0.59882052796136*v_B6_c_r
        struct[0].g[64,0] = 0.598820527961361*v_B5_a_i + 1.10755301189314*v_B5_a_r - 0.171091579417532*v_B5_b_i - 0.316443717683753*v_B5_b_r - 0.171091579417532*v_B5_c_i - 0.316443717683753*v_B5_c_r - 1.28353302446119*v_B6_a_i - 2.23667465123725*v_B6_a_r + 0.385473430243205*v_B6_b_i + 0.643671749092996*v_B6_b_r + 0.385473430243205*v_B6_c_i + 0.643671749092997*v_B6_c_r + 0.996212229189942*v_B6lv_a_i + 0.249053057297486*v_B6lv_a_r - 0.996212229189942*v_B6lv_b_i - 0.249053057297486*v_B6lv_b_r + 0.598820527961361*v_B7_a_i + 1.10755301189314*v_B7_a_r - 0.171091579417532*v_B7_b_i - 0.316443717683753*v_B7_b_r - 0.171091579417532*v_B7_c_i - 0.316443717683753*v_B7_c_r
        struct[0].g[65,0] = 1.10755301189314*v_B5_a_i - 0.598820527961361*v_B5_a_r - 0.316443717683753*v_B5_b_i + 0.171091579417532*v_B5_b_r - 0.316443717683753*v_B5_c_i + 0.171091579417532*v_B5_c_r - 2.23667465123725*v_B6_a_i + 1.28353302446119*v_B6_a_r + 0.643671749092996*v_B6_b_i - 0.385473430243205*v_B6_b_r + 0.643671749092997*v_B6_c_i - 0.385473430243205*v_B6_c_r + 0.249053057297486*v_B6lv_a_i - 0.996212229189942*v_B6lv_a_r - 0.249053057297486*v_B6lv_b_i + 0.996212229189942*v_B6lv_b_r + 1.10755301189314*v_B7_a_i - 0.598820527961361*v_B7_a_r - 0.316443717683753*v_B7_b_i + 0.171091579417532*v_B7_b_r - 0.316443717683753*v_B7_c_i + 0.171091579417532*v_B7_c_r
        struct[0].g[66,0] = -0.171091579417532*v_B5_a_i - 0.316443717683753*v_B5_a_r + 0.59882052796136*v_B5_b_i + 1.10755301189314*v_B5_b_r - 0.171091579417531*v_B5_c_i - 0.316443717683753*v_B5_c_r + 0.385473430243205*v_B6_a_i + 0.643671749092996*v_B6_a_r - 1.28353302446119*v_B6_b_i - 2.23667465123725*v_B6_b_r + 0.385473430243204*v_B6_c_i + 0.643671749092997*v_B6_c_r + 0.996212229189942*v_B6lv_b_i + 0.249053057297486*v_B6lv_b_r - 0.996212229189942*v_B6lv_c_i - 0.249053057297486*v_B6lv_c_r - 0.171091579417532*v_B7_a_i - 0.316443717683753*v_B7_a_r + 0.59882052796136*v_B7_b_i + 1.10755301189314*v_B7_b_r - 0.171091579417531*v_B7_c_i - 0.316443717683753*v_B7_c_r
        struct[0].g[67,0] = -0.316443717683753*v_B5_a_i + 0.171091579417532*v_B5_a_r + 1.10755301189314*v_B5_b_i - 0.59882052796136*v_B5_b_r - 0.316443717683753*v_B5_c_i + 0.171091579417531*v_B5_c_r + 0.643671749092996*v_B6_a_i - 0.385473430243205*v_B6_a_r - 2.23667465123725*v_B6_b_i + 1.28353302446119*v_B6_b_r + 0.643671749092997*v_B6_c_i - 0.385473430243204*v_B6_c_r + 0.249053057297486*v_B6lv_b_i - 0.996212229189942*v_B6lv_b_r - 0.249053057297486*v_B6lv_c_i + 0.996212229189942*v_B6lv_c_r - 0.316443717683753*v_B7_a_i + 0.171091579417532*v_B7_a_r + 1.10755301189314*v_B7_b_i - 0.59882052796136*v_B7_b_r - 0.316443717683753*v_B7_c_i + 0.171091579417531*v_B7_c_r
        struct[0].g[68,0] = -0.171091579417532*v_B5_a_i - 0.316443717683753*v_B5_a_r - 0.171091579417531*v_B5_b_i - 0.316443717683753*v_B5_b_r + 0.59882052796136*v_B5_c_i + 1.10755301189314*v_B5_c_r + 0.385473430243205*v_B6_a_i + 0.643671749092997*v_B6_a_r + 0.385473430243204*v_B6_b_i + 0.643671749092997*v_B6_b_r - 1.28353302446119*v_B6_c_i - 2.23667465123725*v_B6_c_r - 0.996212229189942*v_B6lv_a_i - 0.249053057297486*v_B6lv_a_r + 0.996212229189942*v_B6lv_c_i + 0.249053057297486*v_B6lv_c_r - 0.171091579417532*v_B7_a_i - 0.316443717683753*v_B7_a_r - 0.171091579417531*v_B7_b_i - 0.316443717683753*v_B7_b_r + 0.59882052796136*v_B7_c_i + 1.10755301189314*v_B7_c_r
        struct[0].g[69,0] = -0.316443717683753*v_B5_a_i + 0.171091579417532*v_B5_a_r - 0.316443717683753*v_B5_b_i + 0.171091579417531*v_B5_b_r + 1.10755301189314*v_B5_c_i - 0.59882052796136*v_B5_c_r + 0.643671749092997*v_B6_a_i - 0.385473430243205*v_B6_a_r + 0.643671749092997*v_B6_b_i - 0.385473430243204*v_B6_b_r - 2.23667465123725*v_B6_c_i + 1.28353302446119*v_B6_c_r - 0.249053057297486*v_B6lv_a_i + 0.996212229189942*v_B6lv_a_r + 0.249053057297486*v_B6lv_c_i - 0.996212229189942*v_B6lv_c_r - 0.316443717683753*v_B7_a_i + 0.171091579417532*v_B7_a_r - 0.316443717683753*v_B7_b_i + 0.171091579417531*v_B7_b_r + 1.10755301189314*v_B7_c_i - 0.59882052796136*v_B7_c_r
        struct[0].g[70,0] = -i_t_B2_B2lv_a_r + 0.0862745098039216*v_B2_a_i + 0.0215686274509804*v_B2_a_r - 0.0431372549019608*v_B2_b_i - 0.0107843137254902*v_B2_b_r - 0.0431372549019608*v_B2_c_i - 0.0107843137254902*v_B2_c_r - 0.996212229189942*v_B2lv_a_i - 0.249053057297486*v_B2lv_a_r + 0.996212229189942*v_B2lv_b_i + 0.249053057297486*v_B2lv_b_r
        struct[0].g[71,0] = -i_t_B2_B2lv_a_i + 0.0215686274509804*v_B2_a_i - 0.0862745098039216*v_B2_a_r - 0.0107843137254902*v_B2_b_i + 0.0431372549019608*v_B2_b_r - 0.0107843137254902*v_B2_c_i + 0.0431372549019608*v_B2_c_r - 0.249053057297486*v_B2lv_a_i + 0.996212229189942*v_B2lv_a_r + 0.249053057297486*v_B2lv_b_i - 0.996212229189942*v_B2lv_b_r
        struct[0].g[72,0] = -i_t_B2_B2lv_b_r - 0.0431372549019608*v_B2_a_i - 0.0107843137254902*v_B2_a_r + 0.0862745098039216*v_B2_b_i + 0.0215686274509804*v_B2_b_r - 0.0431372549019608*v_B2_c_i - 0.0107843137254902*v_B2_c_r - 0.996212229189942*v_B2lv_b_i - 0.249053057297486*v_B2lv_b_r + 0.996212229189942*v_B2lv_c_i + 0.249053057297486*v_B2lv_c_r
        struct[0].g[73,0] = -i_t_B2_B2lv_b_i - 0.0107843137254902*v_B2_a_i + 0.0431372549019608*v_B2_a_r + 0.0215686274509804*v_B2_b_i - 0.0862745098039216*v_B2_b_r - 0.0107843137254902*v_B2_c_i + 0.0431372549019608*v_B2_c_r - 0.249053057297486*v_B2lv_b_i + 0.996212229189942*v_B2lv_b_r + 0.249053057297486*v_B2lv_c_i - 0.996212229189942*v_B2lv_c_r
        struct[0].g[74,0] = -i_t_B2_B2lv_c_r - 0.0431372549019608*v_B2_a_i - 0.0107843137254902*v_B2_a_r - 0.0431372549019608*v_B2_b_i - 0.0107843137254902*v_B2_b_r + 0.0862745098039216*v_B2_c_i + 0.0215686274509804*v_B2_c_r + 0.996212229189942*v_B2lv_a_i + 0.249053057297486*v_B2lv_a_r - 0.996212229189942*v_B2lv_c_i - 0.249053057297486*v_B2lv_c_r
        struct[0].g[75,0] = -i_t_B2_B2lv_c_i - 0.0107843137254902*v_B2_a_i + 0.0431372549019608*v_B2_a_r - 0.0107843137254902*v_B2_b_i + 0.0431372549019608*v_B2_b_r + 0.0215686274509804*v_B2_c_i - 0.0862745098039216*v_B2_c_r + 0.249053057297486*v_B2lv_a_i - 0.996212229189942*v_B2lv_a_r - 0.249053057297486*v_B2lv_c_i + 0.996212229189942*v_B2lv_c_r
        struct[0].g[76,0] = -i_t_B3_B3lv_a_r + 0.0862745098039216*v_B3_a_i + 0.0215686274509804*v_B3_a_r - 0.0431372549019608*v_B3_b_i - 0.0107843137254902*v_B3_b_r - 0.0431372549019608*v_B3_c_i - 0.0107843137254902*v_B3_c_r - 0.996212229189942*v_B3lv_a_i - 0.249053057297486*v_B3lv_a_r + 0.996212229189942*v_B3lv_b_i + 0.249053057297486*v_B3lv_b_r
        struct[0].g[77,0] = -i_t_B3_B3lv_a_i + 0.0215686274509804*v_B3_a_i - 0.0862745098039216*v_B3_a_r - 0.0107843137254902*v_B3_b_i + 0.0431372549019608*v_B3_b_r - 0.0107843137254902*v_B3_c_i + 0.0431372549019608*v_B3_c_r - 0.249053057297486*v_B3lv_a_i + 0.996212229189942*v_B3lv_a_r + 0.249053057297486*v_B3lv_b_i - 0.996212229189942*v_B3lv_b_r
        struct[0].g[78,0] = -i_t_B3_B3lv_b_r - 0.0431372549019608*v_B3_a_i - 0.0107843137254902*v_B3_a_r + 0.0862745098039216*v_B3_b_i + 0.0215686274509804*v_B3_b_r - 0.0431372549019608*v_B3_c_i - 0.0107843137254902*v_B3_c_r - 0.996212229189942*v_B3lv_b_i - 0.249053057297486*v_B3lv_b_r + 0.996212229189942*v_B3lv_c_i + 0.249053057297486*v_B3lv_c_r
        struct[0].g[79,0] = -i_t_B3_B3lv_b_i - 0.0107843137254902*v_B3_a_i + 0.0431372549019608*v_B3_a_r + 0.0215686274509804*v_B3_b_i - 0.0862745098039216*v_B3_b_r - 0.0107843137254902*v_B3_c_i + 0.0431372549019608*v_B3_c_r - 0.249053057297486*v_B3lv_b_i + 0.996212229189942*v_B3lv_b_r + 0.249053057297486*v_B3lv_c_i - 0.996212229189942*v_B3lv_c_r
        struct[0].g[80,0] = -i_t_B3_B3lv_c_r - 0.0431372549019608*v_B3_a_i - 0.0107843137254902*v_B3_a_r - 0.0431372549019608*v_B3_b_i - 0.0107843137254902*v_B3_b_r + 0.0862745098039216*v_B3_c_i + 0.0215686274509804*v_B3_c_r + 0.996212229189942*v_B3lv_a_i + 0.249053057297486*v_B3lv_a_r - 0.996212229189942*v_B3lv_c_i - 0.249053057297486*v_B3lv_c_r
        struct[0].g[81,0] = -i_t_B3_B3lv_c_i - 0.0107843137254902*v_B3_a_i + 0.0431372549019608*v_B3_a_r - 0.0107843137254902*v_B3_b_i + 0.0431372549019608*v_B3_b_r + 0.0215686274509804*v_B3_c_i - 0.0862745098039216*v_B3_c_r + 0.249053057297486*v_B3lv_a_i - 0.996212229189942*v_B3lv_a_r - 0.249053057297486*v_B3lv_c_i + 0.996212229189942*v_B3lv_c_r
        struct[0].g[82,0] = -i_t_B4_B4lv_a_r + 0.0862745098039216*v_B4_a_i + 0.0215686274509804*v_B4_a_r - 0.0431372549019608*v_B4_b_i - 0.0107843137254902*v_B4_b_r - 0.0431372549019608*v_B4_c_i - 0.0107843137254902*v_B4_c_r - 0.996212229189942*v_B4lv_a_i - 0.249053057297486*v_B4lv_a_r + 0.996212229189942*v_B4lv_b_i + 0.249053057297486*v_B4lv_b_r
        struct[0].g[83,0] = -i_t_B4_B4lv_a_i + 0.0215686274509804*v_B4_a_i - 0.0862745098039216*v_B4_a_r - 0.0107843137254902*v_B4_b_i + 0.0431372549019608*v_B4_b_r - 0.0107843137254902*v_B4_c_i + 0.0431372549019608*v_B4_c_r - 0.249053057297486*v_B4lv_a_i + 0.996212229189942*v_B4lv_a_r + 0.249053057297486*v_B4lv_b_i - 0.996212229189942*v_B4lv_b_r
        struct[0].g[84,0] = -i_t_B4_B4lv_b_r - 0.0431372549019608*v_B4_a_i - 0.0107843137254902*v_B4_a_r + 0.0862745098039216*v_B4_b_i + 0.0215686274509804*v_B4_b_r - 0.0431372549019608*v_B4_c_i - 0.0107843137254902*v_B4_c_r - 0.996212229189942*v_B4lv_b_i - 0.249053057297486*v_B4lv_b_r + 0.996212229189942*v_B4lv_c_i + 0.249053057297486*v_B4lv_c_r
        struct[0].g[85,0] = -i_t_B4_B4lv_b_i - 0.0107843137254902*v_B4_a_i + 0.0431372549019608*v_B4_a_r + 0.0215686274509804*v_B4_b_i - 0.0862745098039216*v_B4_b_r - 0.0107843137254902*v_B4_c_i + 0.0431372549019608*v_B4_c_r - 0.249053057297486*v_B4lv_b_i + 0.996212229189942*v_B4lv_b_r + 0.249053057297486*v_B4lv_c_i - 0.996212229189942*v_B4lv_c_r
        struct[0].g[86,0] = -i_t_B4_B4lv_c_r - 0.0431372549019608*v_B4_a_i - 0.0107843137254902*v_B4_a_r - 0.0431372549019608*v_B4_b_i - 0.0107843137254902*v_B4_b_r + 0.0862745098039216*v_B4_c_i + 0.0215686274509804*v_B4_c_r + 0.996212229189942*v_B4lv_a_i + 0.249053057297486*v_B4lv_a_r - 0.996212229189942*v_B4lv_c_i - 0.249053057297486*v_B4lv_c_r
        struct[0].g[87,0] = -i_t_B4_B4lv_c_i - 0.0107843137254902*v_B4_a_i + 0.0431372549019608*v_B4_a_r - 0.0107843137254902*v_B4_b_i + 0.0431372549019608*v_B4_b_r + 0.0215686274509804*v_B4_c_i - 0.0862745098039216*v_B4_c_r + 0.249053057297486*v_B4lv_a_i - 0.996212229189942*v_B4lv_a_r - 0.249053057297486*v_B4lv_c_i + 0.996212229189942*v_B4lv_c_r
        struct[0].g[88,0] = -i_t_B5_B5lv_a_r + 0.0862745098039216*v_B5_a_i + 0.0215686274509804*v_B5_a_r - 0.0431372549019608*v_B5_b_i - 0.0107843137254902*v_B5_b_r - 0.0431372549019608*v_B5_c_i - 0.0107843137254902*v_B5_c_r - 0.996212229189942*v_B5lv_a_i - 0.249053057297486*v_B5lv_a_r + 0.996212229189942*v_B5lv_b_i + 0.249053057297486*v_B5lv_b_r
        struct[0].g[89,0] = -i_t_B5_B5lv_a_i + 0.0215686274509804*v_B5_a_i - 0.0862745098039216*v_B5_a_r - 0.0107843137254902*v_B5_b_i + 0.0431372549019608*v_B5_b_r - 0.0107843137254902*v_B5_c_i + 0.0431372549019608*v_B5_c_r - 0.249053057297486*v_B5lv_a_i + 0.996212229189942*v_B5lv_a_r + 0.249053057297486*v_B5lv_b_i - 0.996212229189942*v_B5lv_b_r
        struct[0].g[90,0] = -i_t_B5_B5lv_b_r - 0.0431372549019608*v_B5_a_i - 0.0107843137254902*v_B5_a_r + 0.0862745098039216*v_B5_b_i + 0.0215686274509804*v_B5_b_r - 0.0431372549019608*v_B5_c_i - 0.0107843137254902*v_B5_c_r - 0.996212229189942*v_B5lv_b_i - 0.249053057297486*v_B5lv_b_r + 0.996212229189942*v_B5lv_c_i + 0.249053057297486*v_B5lv_c_r
        struct[0].g[91,0] = -i_t_B5_B5lv_b_i - 0.0107843137254902*v_B5_a_i + 0.0431372549019608*v_B5_a_r + 0.0215686274509804*v_B5_b_i - 0.0862745098039216*v_B5_b_r - 0.0107843137254902*v_B5_c_i + 0.0431372549019608*v_B5_c_r - 0.249053057297486*v_B5lv_b_i + 0.996212229189942*v_B5lv_b_r + 0.249053057297486*v_B5lv_c_i - 0.996212229189942*v_B5lv_c_r
        struct[0].g[92,0] = -i_t_B5_B5lv_c_r - 0.0431372549019608*v_B5_a_i - 0.0107843137254902*v_B5_a_r - 0.0431372549019608*v_B5_b_i - 0.0107843137254902*v_B5_b_r + 0.0862745098039216*v_B5_c_i + 0.0215686274509804*v_B5_c_r + 0.996212229189942*v_B5lv_a_i + 0.249053057297486*v_B5lv_a_r - 0.996212229189942*v_B5lv_c_i - 0.249053057297486*v_B5lv_c_r
        struct[0].g[93,0] = -i_t_B5_B5lv_c_i - 0.0107843137254902*v_B5_a_i + 0.0431372549019608*v_B5_a_r - 0.0107843137254902*v_B5_b_i + 0.0431372549019608*v_B5_b_r + 0.0215686274509804*v_B5_c_i - 0.0862745098039216*v_B5_c_r + 0.249053057297486*v_B5lv_a_i - 0.996212229189942*v_B5lv_a_r - 0.249053057297486*v_B5lv_c_i + 0.996212229189942*v_B5lv_c_r
        struct[0].g[94,0] = -i_t_B6_B6lv_a_r + 0.0862745098039216*v_B6_a_i + 0.0215686274509804*v_B6_a_r - 0.0431372549019608*v_B6_b_i - 0.0107843137254902*v_B6_b_r - 0.0431372549019608*v_B6_c_i - 0.0107843137254902*v_B6_c_r - 0.996212229189942*v_B6lv_a_i - 0.249053057297486*v_B6lv_a_r + 0.996212229189942*v_B6lv_b_i + 0.249053057297486*v_B6lv_b_r
        struct[0].g[95,0] = -i_t_B6_B6lv_a_i + 0.0215686274509804*v_B6_a_i - 0.0862745098039216*v_B6_a_r - 0.0107843137254902*v_B6_b_i + 0.0431372549019608*v_B6_b_r - 0.0107843137254902*v_B6_c_i + 0.0431372549019608*v_B6_c_r - 0.249053057297486*v_B6lv_a_i + 0.996212229189942*v_B6lv_a_r + 0.249053057297486*v_B6lv_b_i - 0.996212229189942*v_B6lv_b_r
        struct[0].g[96,0] = -i_t_B6_B6lv_b_r - 0.0431372549019608*v_B6_a_i - 0.0107843137254902*v_B6_a_r + 0.0862745098039216*v_B6_b_i + 0.0215686274509804*v_B6_b_r - 0.0431372549019608*v_B6_c_i - 0.0107843137254902*v_B6_c_r - 0.996212229189942*v_B6lv_b_i - 0.249053057297486*v_B6lv_b_r + 0.996212229189942*v_B6lv_c_i + 0.249053057297486*v_B6lv_c_r
        struct[0].g[97,0] = -i_t_B6_B6lv_b_i - 0.0107843137254902*v_B6_a_i + 0.0431372549019608*v_B6_a_r + 0.0215686274509804*v_B6_b_i - 0.0862745098039216*v_B6_b_r - 0.0107843137254902*v_B6_c_i + 0.0431372549019608*v_B6_c_r - 0.249053057297486*v_B6lv_b_i + 0.996212229189942*v_B6lv_b_r + 0.249053057297486*v_B6lv_c_i - 0.996212229189942*v_B6lv_c_r
        struct[0].g[98,0] = -i_t_B6_B6lv_c_r - 0.0431372549019608*v_B6_a_i - 0.0107843137254902*v_B6_a_r - 0.0431372549019608*v_B6_b_i - 0.0107843137254902*v_B6_b_r + 0.0862745098039216*v_B6_c_i + 0.0215686274509804*v_B6_c_r + 0.996212229189942*v_B6lv_a_i + 0.249053057297486*v_B6lv_a_r - 0.996212229189942*v_B6lv_c_i - 0.249053057297486*v_B6lv_c_r
        struct[0].g[99,0] = -i_t_B6_B6lv_c_i - 0.0107843137254902*v_B6_a_i + 0.0431372549019608*v_B6_a_r - 0.0107843137254902*v_B6_b_i + 0.0431372549019608*v_B6_b_r + 0.0215686274509804*v_B6_c_i - 0.0862745098039216*v_B6_c_r + 0.249053057297486*v_B6lv_a_i - 0.996212229189942*v_B6lv_a_r - 0.249053057297486*v_B6lv_c_i + 0.996212229189942*v_B6lv_c_r
        struct[0].g[100,0] = -i_l_B1_B2_a_r + 0.598820527961361*v_B1_a_i + 1.10755301189314*v_B1_a_r - 0.171091579417532*v_B1_b_i - 0.316443717683753*v_B1_b_r - 0.171091579417532*v_B1_c_i - 0.316443717683753*v_B1_c_r - 0.598820527961361*v_B2_a_i - 1.10755301189314*v_B2_a_r + 0.171091579417532*v_B2_b_i + 0.316443717683753*v_B2_b_r + 0.171091579417532*v_B2_c_i + 0.316443717683753*v_B2_c_r
        struct[0].g[101,0] = -i_l_B1_B2_a_i + 1.10755301189314*v_B1_a_i - 0.598820527961361*v_B1_a_r - 0.316443717683753*v_B1_b_i + 0.171091579417532*v_B1_b_r - 0.316443717683753*v_B1_c_i + 0.171091579417532*v_B1_c_r - 1.10755301189314*v_B2_a_i + 0.598820527961361*v_B2_a_r + 0.316443717683753*v_B2_b_i - 0.171091579417532*v_B2_b_r + 0.316443717683753*v_B2_c_i - 0.171091579417532*v_B2_c_r
        struct[0].g[102,0] = -i_l_B1_B2_b_r - 0.171091579417532*v_B1_a_i - 0.316443717683753*v_B1_a_r + 0.59882052796136*v_B1_b_i + 1.10755301189314*v_B1_b_r - 0.171091579417531*v_B1_c_i - 0.316443717683753*v_B1_c_r + 0.171091579417532*v_B2_a_i + 0.316443717683753*v_B2_a_r - 0.59882052796136*v_B2_b_i - 1.10755301189314*v_B2_b_r + 0.171091579417531*v_B2_c_i + 0.316443717683753*v_B2_c_r
        struct[0].g[103,0] = -i_l_B1_B2_b_i - 0.316443717683753*v_B1_a_i + 0.171091579417532*v_B1_a_r + 1.10755301189314*v_B1_b_i - 0.59882052796136*v_B1_b_r - 0.316443717683753*v_B1_c_i + 0.171091579417531*v_B1_c_r + 0.316443717683753*v_B2_a_i - 0.171091579417532*v_B2_a_r - 1.10755301189314*v_B2_b_i + 0.59882052796136*v_B2_b_r + 0.316443717683753*v_B2_c_i - 0.171091579417531*v_B2_c_r
        struct[0].g[104,0] = -i_l_B1_B2_c_r - 0.171091579417532*v_B1_a_i - 0.316443717683753*v_B1_a_r - 0.171091579417531*v_B1_b_i - 0.316443717683753*v_B1_b_r + 0.59882052796136*v_B1_c_i + 1.10755301189314*v_B1_c_r + 0.171091579417532*v_B2_a_i + 0.316443717683753*v_B2_a_r + 0.171091579417531*v_B2_b_i + 0.316443717683753*v_B2_b_r - 0.59882052796136*v_B2_c_i - 1.10755301189314*v_B2_c_r
        struct[0].g[105,0] = -i_l_B1_B2_c_i - 0.316443717683753*v_B1_a_i + 0.171091579417532*v_B1_a_r - 0.316443717683753*v_B1_b_i + 0.171091579417531*v_B1_b_r + 1.10755301189314*v_B1_c_i - 0.59882052796136*v_B1_c_r + 0.316443717683753*v_B2_a_i - 0.171091579417532*v_B2_a_r + 0.316443717683753*v_B2_b_i - 0.171091579417531*v_B2_b_r - 1.10755301189314*v_B2_c_i + 0.59882052796136*v_B2_c_r
        struct[0].g[106,0] = -i_l_B2_B3_a_r + 0.598820527961361*v_B2_a_i + 1.10755301189314*v_B2_a_r - 0.171091579417532*v_B2_b_i - 0.316443717683753*v_B2_b_r - 0.171091579417532*v_B2_c_i - 0.316443717683753*v_B2_c_r - 0.598820527961361*v_B3_a_i - 1.10755301189314*v_B3_a_r + 0.171091579417532*v_B3_b_i + 0.316443717683753*v_B3_b_r + 0.171091579417532*v_B3_c_i + 0.316443717683753*v_B3_c_r
        struct[0].g[107,0] = -i_l_B2_B3_a_i + 1.10755301189314*v_B2_a_i - 0.598820527961361*v_B2_a_r - 0.316443717683753*v_B2_b_i + 0.171091579417532*v_B2_b_r - 0.316443717683753*v_B2_c_i + 0.171091579417532*v_B2_c_r - 1.10755301189314*v_B3_a_i + 0.598820527961361*v_B3_a_r + 0.316443717683753*v_B3_b_i - 0.171091579417532*v_B3_b_r + 0.316443717683753*v_B3_c_i - 0.171091579417532*v_B3_c_r
        struct[0].g[108,0] = -i_l_B2_B3_b_r - 0.171091579417532*v_B2_a_i - 0.316443717683753*v_B2_a_r + 0.59882052796136*v_B2_b_i + 1.10755301189314*v_B2_b_r - 0.171091579417531*v_B2_c_i - 0.316443717683753*v_B2_c_r + 0.171091579417532*v_B3_a_i + 0.316443717683753*v_B3_a_r - 0.59882052796136*v_B3_b_i - 1.10755301189314*v_B3_b_r + 0.171091579417531*v_B3_c_i + 0.316443717683753*v_B3_c_r
        struct[0].g[109,0] = -i_l_B2_B3_b_i - 0.316443717683753*v_B2_a_i + 0.171091579417532*v_B2_a_r + 1.10755301189314*v_B2_b_i - 0.59882052796136*v_B2_b_r - 0.316443717683753*v_B2_c_i + 0.171091579417531*v_B2_c_r + 0.316443717683753*v_B3_a_i - 0.171091579417532*v_B3_a_r - 1.10755301189314*v_B3_b_i + 0.59882052796136*v_B3_b_r + 0.316443717683753*v_B3_c_i - 0.171091579417531*v_B3_c_r
        struct[0].g[110,0] = -i_l_B2_B3_c_r - 0.171091579417532*v_B2_a_i - 0.316443717683753*v_B2_a_r - 0.171091579417531*v_B2_b_i - 0.316443717683753*v_B2_b_r + 0.59882052796136*v_B2_c_i + 1.10755301189314*v_B2_c_r + 0.171091579417532*v_B3_a_i + 0.316443717683753*v_B3_a_r + 0.171091579417531*v_B3_b_i + 0.316443717683753*v_B3_b_r - 0.59882052796136*v_B3_c_i - 1.10755301189314*v_B3_c_r
        struct[0].g[111,0] = -i_l_B2_B3_c_i - 0.316443717683753*v_B2_a_i + 0.171091579417532*v_B2_a_r - 0.316443717683753*v_B2_b_i + 0.171091579417531*v_B2_b_r + 1.10755301189314*v_B2_c_i - 0.59882052796136*v_B2_c_r + 0.316443717683753*v_B3_a_i - 0.171091579417532*v_B3_a_r + 0.316443717683753*v_B3_b_i - 0.171091579417531*v_B3_b_r - 1.10755301189314*v_B3_c_i + 0.59882052796136*v_B3_c_r
        struct[0].g[112,0] = -i_l_B3_B4_a_r + 0.598820527961361*v_B3_a_i + 1.10755301189314*v_B3_a_r - 0.171091579417532*v_B3_b_i - 0.316443717683753*v_B3_b_r - 0.171091579417532*v_B3_c_i - 0.316443717683753*v_B3_c_r - 0.598820527961361*v_B4_a_i - 1.10755301189314*v_B4_a_r + 0.171091579417532*v_B4_b_i + 0.316443717683753*v_B4_b_r + 0.171091579417532*v_B4_c_i + 0.316443717683753*v_B4_c_r
        struct[0].g[113,0] = -i_l_B3_B4_a_i + 1.10755301189314*v_B3_a_i - 0.598820527961361*v_B3_a_r - 0.316443717683753*v_B3_b_i + 0.171091579417532*v_B3_b_r - 0.316443717683753*v_B3_c_i + 0.171091579417532*v_B3_c_r - 1.10755301189314*v_B4_a_i + 0.598820527961361*v_B4_a_r + 0.316443717683753*v_B4_b_i - 0.171091579417532*v_B4_b_r + 0.316443717683753*v_B4_c_i - 0.171091579417532*v_B4_c_r
        struct[0].g[114,0] = -i_l_B3_B4_b_r - 0.171091579417532*v_B3_a_i - 0.316443717683753*v_B3_a_r + 0.59882052796136*v_B3_b_i + 1.10755301189314*v_B3_b_r - 0.171091579417531*v_B3_c_i - 0.316443717683753*v_B3_c_r + 0.171091579417532*v_B4_a_i + 0.316443717683753*v_B4_a_r - 0.59882052796136*v_B4_b_i - 1.10755301189314*v_B4_b_r + 0.171091579417531*v_B4_c_i + 0.316443717683753*v_B4_c_r
        struct[0].g[115,0] = -i_l_B3_B4_b_i - 0.316443717683753*v_B3_a_i + 0.171091579417532*v_B3_a_r + 1.10755301189314*v_B3_b_i - 0.59882052796136*v_B3_b_r - 0.316443717683753*v_B3_c_i + 0.171091579417531*v_B3_c_r + 0.316443717683753*v_B4_a_i - 0.171091579417532*v_B4_a_r - 1.10755301189314*v_B4_b_i + 0.59882052796136*v_B4_b_r + 0.316443717683753*v_B4_c_i - 0.171091579417531*v_B4_c_r
        struct[0].g[116,0] = -i_l_B3_B4_c_r - 0.171091579417532*v_B3_a_i - 0.316443717683753*v_B3_a_r - 0.171091579417531*v_B3_b_i - 0.316443717683753*v_B3_b_r + 0.59882052796136*v_B3_c_i + 1.10755301189314*v_B3_c_r + 0.171091579417532*v_B4_a_i + 0.316443717683753*v_B4_a_r + 0.171091579417531*v_B4_b_i + 0.316443717683753*v_B4_b_r - 0.59882052796136*v_B4_c_i - 1.10755301189314*v_B4_c_r
        struct[0].g[117,0] = -i_l_B3_B4_c_i - 0.316443717683753*v_B3_a_i + 0.171091579417532*v_B3_a_r - 0.316443717683753*v_B3_b_i + 0.171091579417531*v_B3_b_r + 1.10755301189314*v_B3_c_i - 0.59882052796136*v_B3_c_r + 0.316443717683753*v_B4_a_i - 0.171091579417532*v_B4_a_r + 0.316443717683753*v_B4_b_i - 0.171091579417531*v_B4_b_r - 1.10755301189314*v_B4_c_i + 0.59882052796136*v_B4_c_r
        struct[0].g[118,0] = -i_l_B5_B6_a_r + 0.598820527961361*v_B5_a_i + 1.10755301189314*v_B5_a_r - 0.171091579417532*v_B5_b_i - 0.316443717683753*v_B5_b_r - 0.171091579417532*v_B5_c_i - 0.316443717683753*v_B5_c_r - 0.598820527961361*v_B6_a_i - 1.10755301189314*v_B6_a_r + 0.171091579417532*v_B6_b_i + 0.316443717683753*v_B6_b_r + 0.171091579417532*v_B6_c_i + 0.316443717683753*v_B6_c_r
        struct[0].g[119,0] = -i_l_B5_B6_a_i + 1.10755301189314*v_B5_a_i - 0.598820527961361*v_B5_a_r - 0.316443717683753*v_B5_b_i + 0.171091579417532*v_B5_b_r - 0.316443717683753*v_B5_c_i + 0.171091579417532*v_B5_c_r - 1.10755301189314*v_B6_a_i + 0.598820527961361*v_B6_a_r + 0.316443717683753*v_B6_b_i - 0.171091579417532*v_B6_b_r + 0.316443717683753*v_B6_c_i - 0.171091579417532*v_B6_c_r
        struct[0].g[120,0] = -i_l_B5_B6_b_r - 0.171091579417532*v_B5_a_i - 0.316443717683753*v_B5_a_r + 0.59882052796136*v_B5_b_i + 1.10755301189314*v_B5_b_r - 0.171091579417531*v_B5_c_i - 0.316443717683753*v_B5_c_r + 0.171091579417532*v_B6_a_i + 0.316443717683753*v_B6_a_r - 0.59882052796136*v_B6_b_i - 1.10755301189314*v_B6_b_r + 0.171091579417531*v_B6_c_i + 0.316443717683753*v_B6_c_r
        struct[0].g[121,0] = -i_l_B5_B6_b_i - 0.316443717683753*v_B5_a_i + 0.171091579417532*v_B5_a_r + 1.10755301189314*v_B5_b_i - 0.59882052796136*v_B5_b_r - 0.316443717683753*v_B5_c_i + 0.171091579417531*v_B5_c_r + 0.316443717683753*v_B6_a_i - 0.171091579417532*v_B6_a_r - 1.10755301189314*v_B6_b_i + 0.59882052796136*v_B6_b_r + 0.316443717683753*v_B6_c_i - 0.171091579417531*v_B6_c_r
        struct[0].g[122,0] = -i_l_B5_B6_c_r - 0.171091579417532*v_B5_a_i - 0.316443717683753*v_B5_a_r - 0.171091579417531*v_B5_b_i - 0.316443717683753*v_B5_b_r + 0.59882052796136*v_B5_c_i + 1.10755301189314*v_B5_c_r + 0.171091579417532*v_B6_a_i + 0.316443717683753*v_B6_a_r + 0.171091579417531*v_B6_b_i + 0.316443717683753*v_B6_b_r - 0.59882052796136*v_B6_c_i - 1.10755301189314*v_B6_c_r
        struct[0].g[123,0] = -i_l_B5_B6_c_i - 0.316443717683753*v_B5_a_i + 0.171091579417532*v_B5_a_r - 0.316443717683753*v_B5_b_i + 0.171091579417531*v_B5_b_r + 1.10755301189314*v_B5_c_i - 0.59882052796136*v_B5_c_r + 0.316443717683753*v_B6_a_i - 0.171091579417532*v_B6_a_r + 0.316443717683753*v_B6_b_i - 0.171091579417531*v_B6_b_r - 1.10755301189314*v_B6_c_i + 0.59882052796136*v_B6_c_r
        struct[0].g[124,0] = -i_l_B6_B7_a_r + 0.598820527961361*v_B6_a_i + 1.10755301189314*v_B6_a_r - 0.171091579417532*v_B6_b_i - 0.316443717683753*v_B6_b_r - 0.171091579417532*v_B6_c_i - 0.316443717683753*v_B6_c_r - 0.598820527961361*v_B7_a_i - 1.10755301189314*v_B7_a_r + 0.171091579417532*v_B7_b_i + 0.316443717683753*v_B7_b_r + 0.171091579417532*v_B7_c_i + 0.316443717683753*v_B7_c_r
        struct[0].g[125,0] = -i_l_B6_B7_a_i + 1.10755301189314*v_B6_a_i - 0.598820527961361*v_B6_a_r - 0.316443717683753*v_B6_b_i + 0.171091579417532*v_B6_b_r - 0.316443717683753*v_B6_c_i + 0.171091579417532*v_B6_c_r - 1.10755301189314*v_B7_a_i + 0.598820527961361*v_B7_a_r + 0.316443717683753*v_B7_b_i - 0.171091579417532*v_B7_b_r + 0.316443717683753*v_B7_c_i - 0.171091579417532*v_B7_c_r
        struct[0].g[126,0] = -i_l_B6_B7_b_r - 0.171091579417532*v_B6_a_i - 0.316443717683753*v_B6_a_r + 0.59882052796136*v_B6_b_i + 1.10755301189314*v_B6_b_r - 0.171091579417531*v_B6_c_i - 0.316443717683753*v_B6_c_r + 0.171091579417532*v_B7_a_i + 0.316443717683753*v_B7_a_r - 0.59882052796136*v_B7_b_i - 1.10755301189314*v_B7_b_r + 0.171091579417531*v_B7_c_i + 0.316443717683753*v_B7_c_r
        struct[0].g[127,0] = -i_l_B6_B7_b_i - 0.316443717683753*v_B6_a_i + 0.171091579417532*v_B6_a_r + 1.10755301189314*v_B6_b_i - 0.59882052796136*v_B6_b_r - 0.316443717683753*v_B6_c_i + 0.171091579417531*v_B6_c_r + 0.316443717683753*v_B7_a_i - 0.171091579417532*v_B7_a_r - 1.10755301189314*v_B7_b_i + 0.59882052796136*v_B7_b_r + 0.316443717683753*v_B7_c_i - 0.171091579417531*v_B7_c_r
        struct[0].g[128,0] = -i_l_B6_B7_c_r - 0.171091579417532*v_B6_a_i - 0.316443717683753*v_B6_a_r - 0.171091579417531*v_B6_b_i - 0.316443717683753*v_B6_b_r + 0.59882052796136*v_B6_c_i + 1.10755301189314*v_B6_c_r + 0.171091579417532*v_B7_a_i + 0.316443717683753*v_B7_a_r + 0.171091579417531*v_B7_b_i + 0.316443717683753*v_B7_b_r - 0.59882052796136*v_B7_c_i - 1.10755301189314*v_B7_c_r
        struct[0].g[129,0] = -i_l_B6_B7_c_i - 0.316443717683753*v_B6_a_i + 0.171091579417532*v_B6_a_r - 0.316443717683753*v_B6_b_i + 0.171091579417531*v_B6_b_r + 1.10755301189314*v_B6_c_i - 0.59882052796136*v_B6_c_r + 0.316443717683753*v_B7_a_i - 0.171091579417532*v_B7_a_r + 0.316443717683753*v_B7_b_i - 0.171091579417531*v_B7_b_r - 1.10755301189314*v_B7_c_i + 0.59882052796136*v_B7_c_r
        struct[0].g[130,0] = i_load_B2lv_a_i*v_B2lv_a_i - i_load_B2lv_a_i*v_B2lv_n_i + i_load_B2lv_a_r*v_B2lv_a_r - i_load_B2lv_a_r*v_B2lv_n_r - p_B2lv_a
        struct[0].g[131,0] = i_load_B2lv_b_i*v_B2lv_b_i - i_load_B2lv_b_i*v_B2lv_n_i + i_load_B2lv_b_r*v_B2lv_b_r - i_load_B2lv_b_r*v_B2lv_n_r - p_B2lv_b
        struct[0].g[132,0] = i_load_B2lv_c_i*v_B2lv_c_i - i_load_B2lv_c_i*v_B2lv_n_i + i_load_B2lv_c_r*v_B2lv_c_r - i_load_B2lv_c_r*v_B2lv_n_r - p_B2lv_c
        struct[0].g[133,0] = -i_load_B2lv_a_i*v_B2lv_a_r + i_load_B2lv_a_i*v_B2lv_n_r + i_load_B2lv_a_r*v_B2lv_a_i - i_load_B2lv_a_r*v_B2lv_n_i - q_B2lv_a
        struct[0].g[134,0] = -i_load_B2lv_b_i*v_B2lv_b_r + i_load_B2lv_b_i*v_B2lv_n_r + i_load_B2lv_b_r*v_B2lv_b_i - i_load_B2lv_b_r*v_B2lv_n_i - q_B2lv_b
        struct[0].g[135,0] = -i_load_B2lv_c_i*v_B2lv_c_r + i_load_B2lv_c_i*v_B2lv_n_r + i_load_B2lv_c_r*v_B2lv_c_i - i_load_B2lv_c_r*v_B2lv_n_i - q_B2lv_c
        struct[0].g[136,0] = i_load_B2lv_a_r + i_load_B2lv_b_r + i_load_B2lv_c_r + i_load_B2lv_n_r
        struct[0].g[137,0] = i_load_B2lv_a_i + i_load_B2lv_b_i + i_load_B2lv_c_i + i_load_B2lv_n_i
        struct[0].g[138,0] = i_load_B3lv_a_i*v_B3lv_a_i - i_load_B3lv_a_i*v_B3lv_n_i + i_load_B3lv_a_r*v_B3lv_a_r - i_load_B3lv_a_r*v_B3lv_n_r - p_B3lv_a
        struct[0].g[139,0] = i_load_B3lv_b_i*v_B3lv_b_i - i_load_B3lv_b_i*v_B3lv_n_i + i_load_B3lv_b_r*v_B3lv_b_r - i_load_B3lv_b_r*v_B3lv_n_r - p_B3lv_b
        struct[0].g[140,0] = i_load_B3lv_c_i*v_B3lv_c_i - i_load_B3lv_c_i*v_B3lv_n_i + i_load_B3lv_c_r*v_B3lv_c_r - i_load_B3lv_c_r*v_B3lv_n_r - p_B3lv_c
        struct[0].g[141,0] = -i_load_B3lv_a_i*v_B3lv_a_r + i_load_B3lv_a_i*v_B3lv_n_r + i_load_B3lv_a_r*v_B3lv_a_i - i_load_B3lv_a_r*v_B3lv_n_i - q_B3lv_a
        struct[0].g[142,0] = -i_load_B3lv_b_i*v_B3lv_b_r + i_load_B3lv_b_i*v_B3lv_n_r + i_load_B3lv_b_r*v_B3lv_b_i - i_load_B3lv_b_r*v_B3lv_n_i - q_B3lv_b
        struct[0].g[143,0] = -i_load_B3lv_c_i*v_B3lv_c_r + i_load_B3lv_c_i*v_B3lv_n_r + i_load_B3lv_c_r*v_B3lv_c_i - i_load_B3lv_c_r*v_B3lv_n_i - q_B3lv_c
        struct[0].g[144,0] = i_load_B3lv_a_r + i_load_B3lv_b_r + i_load_B3lv_c_r + i_load_B3lv_n_r
        struct[0].g[145,0] = i_load_B3lv_a_i + i_load_B3lv_b_i + i_load_B3lv_c_i + i_load_B3lv_n_i
        struct[0].g[146,0] = i_load_B4lv_a_i*v_B4lv_a_i - i_load_B4lv_a_i*v_B4lv_n_i + i_load_B4lv_a_r*v_B4lv_a_r - i_load_B4lv_a_r*v_B4lv_n_r - p_B4lv_a
        struct[0].g[147,0] = i_load_B4lv_b_i*v_B4lv_b_i - i_load_B4lv_b_i*v_B4lv_n_i + i_load_B4lv_b_r*v_B4lv_b_r - i_load_B4lv_b_r*v_B4lv_n_r - p_B4lv_b
        struct[0].g[148,0] = i_load_B4lv_c_i*v_B4lv_c_i - i_load_B4lv_c_i*v_B4lv_n_i + i_load_B4lv_c_r*v_B4lv_c_r - i_load_B4lv_c_r*v_B4lv_n_r - p_B4lv_c
        struct[0].g[149,0] = -i_load_B4lv_a_i*v_B4lv_a_r + i_load_B4lv_a_i*v_B4lv_n_r + i_load_B4lv_a_r*v_B4lv_a_i - i_load_B4lv_a_r*v_B4lv_n_i - q_B4lv_a
        struct[0].g[150,0] = -i_load_B4lv_b_i*v_B4lv_b_r + i_load_B4lv_b_i*v_B4lv_n_r + i_load_B4lv_b_r*v_B4lv_b_i - i_load_B4lv_b_r*v_B4lv_n_i - q_B4lv_b
        struct[0].g[151,0] = -i_load_B4lv_c_i*v_B4lv_c_r + i_load_B4lv_c_i*v_B4lv_n_r + i_load_B4lv_c_r*v_B4lv_c_i - i_load_B4lv_c_r*v_B4lv_n_i - q_B4lv_c
        struct[0].g[152,0] = i_load_B4lv_a_r + i_load_B4lv_b_r + i_load_B4lv_c_r + i_load_B4lv_n_r
        struct[0].g[153,0] = i_load_B4lv_a_i + i_load_B4lv_b_i + i_load_B4lv_c_i + i_load_B4lv_n_i
        struct[0].g[154,0] = i_load_B5lv_a_i*v_B5lv_a_i - i_load_B5lv_a_i*v_B5lv_n_i + i_load_B5lv_a_r*v_B5lv_a_r - i_load_B5lv_a_r*v_B5lv_n_r - p_B5lv_a
        struct[0].g[155,0] = i_load_B5lv_b_i*v_B5lv_b_i - i_load_B5lv_b_i*v_B5lv_n_i + i_load_B5lv_b_r*v_B5lv_b_r - i_load_B5lv_b_r*v_B5lv_n_r - p_B5lv_b
        struct[0].g[156,0] = i_load_B5lv_c_i*v_B5lv_c_i - i_load_B5lv_c_i*v_B5lv_n_i + i_load_B5lv_c_r*v_B5lv_c_r - i_load_B5lv_c_r*v_B5lv_n_r - p_B5lv_c
        struct[0].g[157,0] = -i_load_B5lv_a_i*v_B5lv_a_r + i_load_B5lv_a_i*v_B5lv_n_r + i_load_B5lv_a_r*v_B5lv_a_i - i_load_B5lv_a_r*v_B5lv_n_i - q_B5lv_a
        struct[0].g[158,0] = -i_load_B5lv_b_i*v_B5lv_b_r + i_load_B5lv_b_i*v_B5lv_n_r + i_load_B5lv_b_r*v_B5lv_b_i - i_load_B5lv_b_r*v_B5lv_n_i - q_B5lv_b
        struct[0].g[159,0] = -i_load_B5lv_c_i*v_B5lv_c_r + i_load_B5lv_c_i*v_B5lv_n_r + i_load_B5lv_c_r*v_B5lv_c_i - i_load_B5lv_c_r*v_B5lv_n_i - q_B5lv_c
        struct[0].g[160,0] = i_load_B5lv_a_r + i_load_B5lv_b_r + i_load_B5lv_c_r + i_load_B5lv_n_r
        struct[0].g[161,0] = i_load_B5lv_a_i + i_load_B5lv_b_i + i_load_B5lv_c_i + i_load_B5lv_n_i
        struct[0].g[162,0] = i_load_B6lv_a_i*v_B6lv_a_i - i_load_B6lv_a_i*v_B6lv_n_i + i_load_B6lv_a_r*v_B6lv_a_r - i_load_B6lv_a_r*v_B6lv_n_r - p_B6lv_a
        struct[0].g[163,0] = i_load_B6lv_b_i*v_B6lv_b_i - i_load_B6lv_b_i*v_B6lv_n_i + i_load_B6lv_b_r*v_B6lv_b_r - i_load_B6lv_b_r*v_B6lv_n_r - p_B6lv_b
        struct[0].g[164,0] = i_load_B6lv_c_i*v_B6lv_c_i - i_load_B6lv_c_i*v_B6lv_n_i + i_load_B6lv_c_r*v_B6lv_c_r - i_load_B6lv_c_r*v_B6lv_n_r - p_B6lv_c
        struct[0].g[165,0] = -i_load_B6lv_a_i*v_B6lv_a_r + i_load_B6lv_a_i*v_B6lv_n_r + i_load_B6lv_a_r*v_B6lv_a_i - i_load_B6lv_a_r*v_B6lv_n_i - q_B6lv_a
        struct[0].g[166,0] = -i_load_B6lv_b_i*v_B6lv_b_r + i_load_B6lv_b_i*v_B6lv_n_r + i_load_B6lv_b_r*v_B6lv_b_i - i_load_B6lv_b_r*v_B6lv_n_i - q_B6lv_b
        struct[0].g[167,0] = -i_load_B6lv_c_i*v_B6lv_c_r + i_load_B6lv_c_i*v_B6lv_n_r + i_load_B6lv_c_r*v_B6lv_c_i - i_load_B6lv_c_r*v_B6lv_n_i - q_B6lv_c
        struct[0].g[168,0] = i_load_B6lv_a_r + i_load_B6lv_b_r + i_load_B6lv_c_r + i_load_B6lv_n_r
        struct[0].g[169,0] = i_load_B6lv_a_i + i_load_B6lv_b_i + i_load_B6lv_c_i + i_load_B6lv_n_i
    
    # Outputs:
    if mode == 3:

        struct[0].h[0,0] = (v_B1_a_i**2 + v_B1_a_r**2)**0.5
        struct[0].h[1,0] = (v_B1_b_i**2 + v_B1_b_r**2)**0.5
        struct[0].h[2,0] = (v_B1_c_i**2 + v_B1_c_r**2)**0.5
        struct[0].h[3,0] = (v_B7_a_i**2 + v_B7_a_r**2)**0.5
        struct[0].h[4,0] = (v_B7_b_i**2 + v_B7_b_r**2)**0.5
        struct[0].h[5,0] = (v_B7_c_i**2 + v_B7_c_r**2)**0.5
        struct[0].h[6,0] = (v_B2lv_a_i**2 + v_B2lv_a_r**2)**0.5
        struct[0].h[7,0] = (v_B2lv_b_i**2 + v_B2lv_b_r**2)**0.5
        struct[0].h[8,0] = (v_B2lv_c_i**2 + v_B2lv_c_r**2)**0.5
        struct[0].h[9,0] = (v_B2lv_n_i**2 + v_B2lv_n_r**2)**0.5
        struct[0].h[10,0] = (v_B3lv_a_i**2 + v_B3lv_a_r**2)**0.5
        struct[0].h[11,0] = (v_B3lv_b_i**2 + v_B3lv_b_r**2)**0.5
        struct[0].h[12,0] = (v_B3lv_c_i**2 + v_B3lv_c_r**2)**0.5
        struct[0].h[13,0] = (v_B3lv_n_i**2 + v_B3lv_n_r**2)**0.5
        struct[0].h[14,0] = (v_B4lv_a_i**2 + v_B4lv_a_r**2)**0.5
        struct[0].h[15,0] = (v_B4lv_b_i**2 + v_B4lv_b_r**2)**0.5
        struct[0].h[16,0] = (v_B4lv_c_i**2 + v_B4lv_c_r**2)**0.5
        struct[0].h[17,0] = (v_B4lv_n_i**2 + v_B4lv_n_r**2)**0.5
        struct[0].h[18,0] = (v_B5lv_a_i**2 + v_B5lv_a_r**2)**0.5
        struct[0].h[19,0] = (v_B5lv_b_i**2 + v_B5lv_b_r**2)**0.5
        struct[0].h[20,0] = (v_B5lv_c_i**2 + v_B5lv_c_r**2)**0.5
        struct[0].h[21,0] = (v_B5lv_n_i**2 + v_B5lv_n_r**2)**0.5
        struct[0].h[22,0] = (v_B6lv_a_i**2 + v_B6lv_a_r**2)**0.5
        struct[0].h[23,0] = (v_B6lv_b_i**2 + v_B6lv_b_r**2)**0.5
        struct[0].h[24,0] = (v_B6lv_c_i**2 + v_B6lv_c_r**2)**0.5
        struct[0].h[25,0] = (v_B6lv_n_i**2 + v_B6lv_n_r**2)**0.5
        struct[0].h[26,0] = (v_B2_a_i**2 + v_B2_a_r**2)**0.5
        struct[0].h[27,0] = (v_B2_b_i**2 + v_B2_b_r**2)**0.5
        struct[0].h[28,0] = (v_B2_c_i**2 + v_B2_c_r**2)**0.5
        struct[0].h[29,0] = (v_B3_a_i**2 + v_B3_a_r**2)**0.5
        struct[0].h[30,0] = (v_B3_b_i**2 + v_B3_b_r**2)**0.5
        struct[0].h[31,0] = (v_B3_c_i**2 + v_B3_c_r**2)**0.5
        struct[0].h[32,0] = (v_B4_a_i**2 + v_B4_a_r**2)**0.5
        struct[0].h[33,0] = (v_B4_b_i**2 + v_B4_b_r**2)**0.5
        struct[0].h[34,0] = (v_B4_c_i**2 + v_B4_c_r**2)**0.5
        struct[0].h[35,0] = (v_B5_a_i**2 + v_B5_a_r**2)**0.5
        struct[0].h[36,0] = (v_B5_b_i**2 + v_B5_b_r**2)**0.5
        struct[0].h[37,0] = (v_B5_c_i**2 + v_B5_c_r**2)**0.5
        struct[0].h[38,0] = (v_B6_a_i**2 + v_B6_a_r**2)**0.5
        struct[0].h[39,0] = (v_B6_b_i**2 + v_B6_b_r**2)**0.5
        struct[0].h[40,0] = (v_B6_c_i**2 + v_B6_c_r**2)**0.5
    

    if mode == 10:

        struct[0].Fx[0,0] = -1

    if mode == 11:


        struct[0].Gy[0,0] = -5.75163398692810
        struct[0].Gy[0,1] = -23.0065359477124
        struct[0].Gy[0,6] = 5.75163398692810
        struct[0].Gy[0,7] = 23.0065359477124
        struct[0].Gy[0,40] = 0.249053057297486
        struct[0].Gy[0,41] = 0.996212229189942
        struct[0].Gy[0,44] = -0.249053057297486
        struct[0].Gy[0,45] = -0.996212229189942
        struct[0].Gy[0,130] = 1
        struct[0].Gy[1,0] = 23.0065359477124
        struct[0].Gy[1,1] = -5.75163398692810
        struct[0].Gy[1,6] = -23.0065359477124
        struct[0].Gy[1,7] = 5.75163398692810
        struct[0].Gy[1,40] = -0.996212229189942
        struct[0].Gy[1,41] = 0.249053057297486
        struct[0].Gy[1,44] = 0.996212229189942
        struct[0].Gy[1,45] = -0.249053057297486
        struct[0].Gy[1,131] = 1
        struct[0].Gy[2,2] = -5.75163398692810
        struct[0].Gy[2,3] = -23.0065359477124
        struct[0].Gy[2,6] = 5.75163398692810
        struct[0].Gy[2,7] = 23.0065359477124
        struct[0].Gy[2,40] = -0.249053057297486
        struct[0].Gy[2,41] = -0.996212229189942
        struct[0].Gy[2,42] = 0.249053057297486
        struct[0].Gy[2,43] = 0.996212229189942
        struct[0].Gy[2,132] = 1
        struct[0].Gy[3,2] = 23.0065359477124
        struct[0].Gy[3,3] = -5.75163398692810
        struct[0].Gy[3,6] = -23.0065359477124
        struct[0].Gy[3,7] = 5.75163398692810
        struct[0].Gy[3,40] = 0.996212229189942
        struct[0].Gy[3,41] = -0.249053057297486
        struct[0].Gy[3,42] = -0.996212229189942
        struct[0].Gy[3,43] = 0.249053057297486
        struct[0].Gy[3,133] = 1
        struct[0].Gy[4,4] = -5.75163398692810
        struct[0].Gy[4,5] = -23.0065359477124
        struct[0].Gy[4,6] = 5.75163398692810
        struct[0].Gy[4,7] = 23.0065359477124
        struct[0].Gy[4,42] = -0.249053057297486
        struct[0].Gy[4,43] = -0.996212229189942
        struct[0].Gy[4,44] = 0.249053057297486
        struct[0].Gy[4,45] = 0.996212229189942
        struct[0].Gy[4,134] = 1
        struct[0].Gy[5,4] = 23.0065359477124
        struct[0].Gy[5,5] = -5.75163398692810
        struct[0].Gy[5,6] = -23.0065359477124
        struct[0].Gy[5,7] = 5.75163398692810
        struct[0].Gy[5,42] = 0.996212229189942
        struct[0].Gy[5,43] = -0.249053057297486
        struct[0].Gy[5,44] = -0.996212229189942
        struct[0].Gy[5,45] = 0.249053057297486
        struct[0].Gy[5,135] = 1
        struct[0].Gy[6,0] = 5.75163398692810
        struct[0].Gy[6,1] = 23.0065359477124
        struct[0].Gy[6,2] = 5.75163398692810
        struct[0].Gy[6,3] = 23.0065359477124
        struct[0].Gy[6,4] = 5.75163398692810
        struct[0].Gy[6,5] = 23.0065359477124
        struct[0].Gy[6,6] = -1017.25490196078
        struct[0].Gy[6,7] = -69.0196078431372
        struct[0].Gy[7,0] = -23.0065359477124
        struct[0].Gy[7,1] = 5.75163398692810
        struct[0].Gy[7,2] = -23.0065359477124
        struct[0].Gy[7,3] = 5.75163398692810
        struct[0].Gy[7,4] = -23.0065359477124
        struct[0].Gy[7,5] = 5.75163398692810
        struct[0].Gy[7,6] = 69.0196078431372
        struct[0].Gy[7,7] = -1017.25490196078
        struct[0].Gy[8,8] = -5.75163398692810
        struct[0].Gy[8,9] = -23.0065359477124
        struct[0].Gy[8,14] = 5.75163398692810
        struct[0].Gy[8,15] = 23.0065359477124
        struct[0].Gy[8,46] = 0.249053057297486
        struct[0].Gy[8,47] = 0.996212229189942
        struct[0].Gy[8,50] = -0.249053057297486
        struct[0].Gy[8,51] = -0.996212229189942
        struct[0].Gy[8,138] = 1
        struct[0].Gy[9,8] = 23.0065359477124
        struct[0].Gy[9,9] = -5.75163398692810
        struct[0].Gy[9,14] = -23.0065359477124
        struct[0].Gy[9,15] = 5.75163398692810
        struct[0].Gy[9,46] = -0.996212229189942
        struct[0].Gy[9,47] = 0.249053057297486
        struct[0].Gy[9,50] = 0.996212229189942
        struct[0].Gy[9,51] = -0.249053057297486
        struct[0].Gy[9,139] = 1
        struct[0].Gy[10,10] = -5.75163398692810
        struct[0].Gy[10,11] = -23.0065359477124
        struct[0].Gy[10,14] = 5.75163398692810
        struct[0].Gy[10,15] = 23.0065359477124
        struct[0].Gy[10,46] = -0.249053057297486
        struct[0].Gy[10,47] = -0.996212229189942
        struct[0].Gy[10,48] = 0.249053057297486
        struct[0].Gy[10,49] = 0.996212229189942
        struct[0].Gy[10,140] = 1
        struct[0].Gy[11,10] = 23.0065359477124
        struct[0].Gy[11,11] = -5.75163398692810
        struct[0].Gy[11,14] = -23.0065359477124
        struct[0].Gy[11,15] = 5.75163398692810
        struct[0].Gy[11,46] = 0.996212229189942
        struct[0].Gy[11,47] = -0.249053057297486
        struct[0].Gy[11,48] = -0.996212229189942
        struct[0].Gy[11,49] = 0.249053057297486
        struct[0].Gy[11,141] = 1
        struct[0].Gy[12,12] = -5.75163398692810
        struct[0].Gy[12,13] = -23.0065359477124
        struct[0].Gy[12,14] = 5.75163398692810
        struct[0].Gy[12,15] = 23.0065359477124
        struct[0].Gy[12,48] = -0.249053057297486
        struct[0].Gy[12,49] = -0.996212229189942
        struct[0].Gy[12,50] = 0.249053057297486
        struct[0].Gy[12,51] = 0.996212229189942
        struct[0].Gy[12,142] = 1
        struct[0].Gy[13,12] = 23.0065359477124
        struct[0].Gy[13,13] = -5.75163398692810
        struct[0].Gy[13,14] = -23.0065359477124
        struct[0].Gy[13,15] = 5.75163398692810
        struct[0].Gy[13,48] = 0.996212229189942
        struct[0].Gy[13,49] = -0.249053057297486
        struct[0].Gy[13,50] = -0.996212229189942
        struct[0].Gy[13,51] = 0.249053057297486
        struct[0].Gy[13,143] = 1
        struct[0].Gy[14,8] = 5.75163398692810
        struct[0].Gy[14,9] = 23.0065359477124
        struct[0].Gy[14,10] = 5.75163398692810
        struct[0].Gy[14,11] = 23.0065359477124
        struct[0].Gy[14,12] = 5.75163398692810
        struct[0].Gy[14,13] = 23.0065359477124
        struct[0].Gy[14,14] = -1017.25490196078
        struct[0].Gy[14,15] = -69.0196078431372
        struct[0].Gy[15,8] = -23.0065359477124
        struct[0].Gy[15,9] = 5.75163398692810
        struct[0].Gy[15,10] = -23.0065359477124
        struct[0].Gy[15,11] = 5.75163398692810
        struct[0].Gy[15,12] = -23.0065359477124
        struct[0].Gy[15,13] = 5.75163398692810
        struct[0].Gy[15,14] = 69.0196078431372
        struct[0].Gy[15,15] = -1017.25490196078
        struct[0].Gy[16,16] = -5.75163398692810
        struct[0].Gy[16,17] = -23.0065359477124
        struct[0].Gy[16,22] = 5.75163398692810
        struct[0].Gy[16,23] = 23.0065359477124
        struct[0].Gy[16,52] = 0.249053057297486
        struct[0].Gy[16,53] = 0.996212229189942
        struct[0].Gy[16,56] = -0.249053057297486
        struct[0].Gy[16,57] = -0.996212229189942
        struct[0].Gy[16,146] = 1
        struct[0].Gy[17,16] = 23.0065359477124
        struct[0].Gy[17,17] = -5.75163398692810
        struct[0].Gy[17,22] = -23.0065359477124
        struct[0].Gy[17,23] = 5.75163398692810
        struct[0].Gy[17,52] = -0.996212229189942
        struct[0].Gy[17,53] = 0.249053057297486
        struct[0].Gy[17,56] = 0.996212229189942
        struct[0].Gy[17,57] = -0.249053057297486
        struct[0].Gy[17,147] = 1
        struct[0].Gy[18,18] = -5.75163398692810
        struct[0].Gy[18,19] = -23.0065359477124
        struct[0].Gy[18,22] = 5.75163398692810
        struct[0].Gy[18,23] = 23.0065359477124
        struct[0].Gy[18,52] = -0.249053057297486
        struct[0].Gy[18,53] = -0.996212229189942
        struct[0].Gy[18,54] = 0.249053057297486
        struct[0].Gy[18,55] = 0.996212229189942
        struct[0].Gy[18,148] = 1
        struct[0].Gy[19,18] = 23.0065359477124
        struct[0].Gy[19,19] = -5.75163398692810
        struct[0].Gy[19,22] = -23.0065359477124
        struct[0].Gy[19,23] = 5.75163398692810
        struct[0].Gy[19,52] = 0.996212229189942
        struct[0].Gy[19,53] = -0.249053057297486
        struct[0].Gy[19,54] = -0.996212229189942
        struct[0].Gy[19,55] = 0.249053057297486
        struct[0].Gy[19,149] = 1
        struct[0].Gy[20,20] = -5.75163398692810
        struct[0].Gy[20,21] = -23.0065359477124
        struct[0].Gy[20,22] = 5.75163398692810
        struct[0].Gy[20,23] = 23.0065359477124
        struct[0].Gy[20,54] = -0.249053057297486
        struct[0].Gy[20,55] = -0.996212229189942
        struct[0].Gy[20,56] = 0.249053057297486
        struct[0].Gy[20,57] = 0.996212229189942
        struct[0].Gy[20,150] = 1
        struct[0].Gy[21,20] = 23.0065359477124
        struct[0].Gy[21,21] = -5.75163398692810
        struct[0].Gy[21,22] = -23.0065359477124
        struct[0].Gy[21,23] = 5.75163398692810
        struct[0].Gy[21,54] = 0.996212229189942
        struct[0].Gy[21,55] = -0.249053057297486
        struct[0].Gy[21,56] = -0.996212229189942
        struct[0].Gy[21,57] = 0.249053057297486
        struct[0].Gy[21,151] = 1
        struct[0].Gy[22,16] = 5.75163398692810
        struct[0].Gy[22,17] = 23.0065359477124
        struct[0].Gy[22,18] = 5.75163398692810
        struct[0].Gy[22,19] = 23.0065359477124
        struct[0].Gy[22,20] = 5.75163398692810
        struct[0].Gy[22,21] = 23.0065359477124
        struct[0].Gy[22,22] = -1017.25490196078
        struct[0].Gy[22,23] = -69.0196078431372
        struct[0].Gy[23,16] = -23.0065359477124
        struct[0].Gy[23,17] = 5.75163398692810
        struct[0].Gy[23,18] = -23.0065359477124
        struct[0].Gy[23,19] = 5.75163398692810
        struct[0].Gy[23,20] = -23.0065359477124
        struct[0].Gy[23,21] = 5.75163398692810
        struct[0].Gy[23,22] = 69.0196078431372
        struct[0].Gy[23,23] = -1017.25490196078
        struct[0].Gy[24,24] = -5.75163398692810
        struct[0].Gy[24,25] = -23.0065359477124
        struct[0].Gy[24,30] = 5.75163398692810
        struct[0].Gy[24,31] = 23.0065359477124
        struct[0].Gy[24,58] = 0.249053057297486
        struct[0].Gy[24,59] = 0.996212229189942
        struct[0].Gy[24,62] = -0.249053057297486
        struct[0].Gy[24,63] = -0.996212229189942
        struct[0].Gy[24,154] = 1
        struct[0].Gy[25,24] = 23.0065359477124
        struct[0].Gy[25,25] = -5.75163398692810
        struct[0].Gy[25,30] = -23.0065359477124
        struct[0].Gy[25,31] = 5.75163398692810
        struct[0].Gy[25,58] = -0.996212229189942
        struct[0].Gy[25,59] = 0.249053057297486
        struct[0].Gy[25,62] = 0.996212229189942
        struct[0].Gy[25,63] = -0.249053057297486
        struct[0].Gy[25,155] = 1
        struct[0].Gy[26,26] = -5.75163398692810
        struct[0].Gy[26,27] = -23.0065359477124
        struct[0].Gy[26,30] = 5.75163398692810
        struct[0].Gy[26,31] = 23.0065359477124
        struct[0].Gy[26,58] = -0.249053057297486
        struct[0].Gy[26,59] = -0.996212229189942
        struct[0].Gy[26,60] = 0.249053057297486
        struct[0].Gy[26,61] = 0.996212229189942
        struct[0].Gy[26,156] = 1
        struct[0].Gy[27,26] = 23.0065359477124
        struct[0].Gy[27,27] = -5.75163398692810
        struct[0].Gy[27,30] = -23.0065359477124
        struct[0].Gy[27,31] = 5.75163398692810
        struct[0].Gy[27,58] = 0.996212229189942
        struct[0].Gy[27,59] = -0.249053057297486
        struct[0].Gy[27,60] = -0.996212229189942
        struct[0].Gy[27,61] = 0.249053057297486
        struct[0].Gy[27,157] = 1
        struct[0].Gy[28,28] = -5.75163398692810
        struct[0].Gy[28,29] = -23.0065359477124
        struct[0].Gy[28,30] = 5.75163398692810
        struct[0].Gy[28,31] = 23.0065359477124
        struct[0].Gy[28,60] = -0.249053057297486
        struct[0].Gy[28,61] = -0.996212229189942
        struct[0].Gy[28,62] = 0.249053057297486
        struct[0].Gy[28,63] = 0.996212229189942
        struct[0].Gy[28,158] = 1
        struct[0].Gy[29,28] = 23.0065359477124
        struct[0].Gy[29,29] = -5.75163398692810
        struct[0].Gy[29,30] = -23.0065359477124
        struct[0].Gy[29,31] = 5.75163398692810
        struct[0].Gy[29,60] = 0.996212229189942
        struct[0].Gy[29,61] = -0.249053057297486
        struct[0].Gy[29,62] = -0.996212229189942
        struct[0].Gy[29,63] = 0.249053057297486
        struct[0].Gy[29,159] = 1
        struct[0].Gy[30,24] = 5.75163398692810
        struct[0].Gy[30,25] = 23.0065359477124
        struct[0].Gy[30,26] = 5.75163398692810
        struct[0].Gy[30,27] = 23.0065359477124
        struct[0].Gy[30,28] = 5.75163398692810
        struct[0].Gy[30,29] = 23.0065359477124
        struct[0].Gy[30,30] = -1017.25490196078
        struct[0].Gy[30,31] = -69.0196078431372
        struct[0].Gy[31,24] = -23.0065359477124
        struct[0].Gy[31,25] = 5.75163398692810
        struct[0].Gy[31,26] = -23.0065359477124
        struct[0].Gy[31,27] = 5.75163398692810
        struct[0].Gy[31,28] = -23.0065359477124
        struct[0].Gy[31,29] = 5.75163398692810
        struct[0].Gy[31,30] = 69.0196078431372
        struct[0].Gy[31,31] = -1017.25490196078
        struct[0].Gy[32,32] = -5.75163398692810
        struct[0].Gy[32,33] = -23.0065359477124
        struct[0].Gy[32,38] = 5.75163398692810
        struct[0].Gy[32,39] = 23.0065359477124
        struct[0].Gy[32,64] = 0.249053057297486
        struct[0].Gy[32,65] = 0.996212229189942
        struct[0].Gy[32,68] = -0.249053057297486
        struct[0].Gy[32,69] = -0.996212229189942
        struct[0].Gy[32,162] = 1
        struct[0].Gy[33,32] = 23.0065359477124
        struct[0].Gy[33,33] = -5.75163398692810
        struct[0].Gy[33,38] = -23.0065359477124
        struct[0].Gy[33,39] = 5.75163398692810
        struct[0].Gy[33,64] = -0.996212229189942
        struct[0].Gy[33,65] = 0.249053057297486
        struct[0].Gy[33,68] = 0.996212229189942
        struct[0].Gy[33,69] = -0.249053057297486
        struct[0].Gy[33,163] = 1
        struct[0].Gy[34,34] = -5.75163398692810
        struct[0].Gy[34,35] = -23.0065359477124
        struct[0].Gy[34,38] = 5.75163398692810
        struct[0].Gy[34,39] = 23.0065359477124
        struct[0].Gy[34,64] = -0.249053057297486
        struct[0].Gy[34,65] = -0.996212229189942
        struct[0].Gy[34,66] = 0.249053057297486
        struct[0].Gy[34,67] = 0.996212229189942
        struct[0].Gy[34,164] = 1
        struct[0].Gy[35,34] = 23.0065359477124
        struct[0].Gy[35,35] = -5.75163398692810
        struct[0].Gy[35,38] = -23.0065359477124
        struct[0].Gy[35,39] = 5.75163398692810
        struct[0].Gy[35,64] = 0.996212229189942
        struct[0].Gy[35,65] = -0.249053057297486
        struct[0].Gy[35,66] = -0.996212229189942
        struct[0].Gy[35,67] = 0.249053057297486
        struct[0].Gy[35,165] = 1
        struct[0].Gy[36,36] = -5.75163398692810
        struct[0].Gy[36,37] = -23.0065359477124
        struct[0].Gy[36,38] = 5.75163398692810
        struct[0].Gy[36,39] = 23.0065359477124
        struct[0].Gy[36,66] = -0.249053057297486
        struct[0].Gy[36,67] = -0.996212229189942
        struct[0].Gy[36,68] = 0.249053057297486
        struct[0].Gy[36,69] = 0.996212229189942
        struct[0].Gy[36,166] = 1
        struct[0].Gy[37,36] = 23.0065359477124
        struct[0].Gy[37,37] = -5.75163398692810
        struct[0].Gy[37,38] = -23.0065359477124
        struct[0].Gy[37,39] = 5.75163398692810
        struct[0].Gy[37,66] = 0.996212229189942
        struct[0].Gy[37,67] = -0.249053057297486
        struct[0].Gy[37,68] = -0.996212229189942
        struct[0].Gy[37,69] = 0.249053057297486
        struct[0].Gy[37,167] = 1
        struct[0].Gy[38,32] = 5.75163398692810
        struct[0].Gy[38,33] = 23.0065359477124
        struct[0].Gy[38,34] = 5.75163398692810
        struct[0].Gy[38,35] = 23.0065359477124
        struct[0].Gy[38,36] = 5.75163398692810
        struct[0].Gy[38,37] = 23.0065359477124
        struct[0].Gy[38,38] = -1017.25490196078
        struct[0].Gy[38,39] = -69.0196078431372
        struct[0].Gy[39,32] = -23.0065359477124
        struct[0].Gy[39,33] = 5.75163398692810
        struct[0].Gy[39,34] = -23.0065359477124
        struct[0].Gy[39,35] = 5.75163398692810
        struct[0].Gy[39,36] = -23.0065359477124
        struct[0].Gy[39,37] = 5.75163398692810
        struct[0].Gy[39,38] = 69.0196078431372
        struct[0].Gy[39,39] = -1017.25490196078
        struct[0].Gy[40,0] = 0.249053057297486
        struct[0].Gy[40,1] = 0.996212229189942
        struct[0].Gy[40,2] = -0.249053057297486
        struct[0].Gy[40,3] = -0.996212229189942
        struct[0].Gy[40,40] = -2.23667465123725
        struct[0].Gy[40,41] = -1.28353302446119
        struct[0].Gy[40,42] = 0.643671749092996
        struct[0].Gy[40,43] = 0.385473430243205
        struct[0].Gy[40,44] = 0.643671749092997
        struct[0].Gy[40,45] = 0.385473430243205
        struct[0].Gy[40,46] = 1.10755301189314
        struct[0].Gy[40,47] = 0.598820527961361
        struct[0].Gy[40,48] = -0.316443717683753
        struct[0].Gy[40,49] = -0.171091579417532
        struct[0].Gy[40,50] = -0.316443717683753
        struct[0].Gy[40,51] = -0.171091579417532
        struct[0].Gy[41,0] = -0.996212229189942
        struct[0].Gy[41,1] = 0.249053057297486
        struct[0].Gy[41,2] = 0.996212229189942
        struct[0].Gy[41,3] = -0.249053057297486
        struct[0].Gy[41,40] = 1.28353302446119
        struct[0].Gy[41,41] = -2.23667465123725
        struct[0].Gy[41,42] = -0.385473430243205
        struct[0].Gy[41,43] = 0.643671749092996
        struct[0].Gy[41,44] = -0.385473430243205
        struct[0].Gy[41,45] = 0.643671749092997
        struct[0].Gy[41,46] = -0.598820527961361
        struct[0].Gy[41,47] = 1.10755301189314
        struct[0].Gy[41,48] = 0.171091579417532
        struct[0].Gy[41,49] = -0.316443717683753
        struct[0].Gy[41,50] = 0.171091579417532
        struct[0].Gy[41,51] = -0.316443717683753
        struct[0].Gy[42,2] = 0.249053057297486
        struct[0].Gy[42,3] = 0.996212229189942
        struct[0].Gy[42,4] = -0.249053057297486
        struct[0].Gy[42,5] = -0.996212229189942
        struct[0].Gy[42,40] = 0.643671749092996
        struct[0].Gy[42,41] = 0.385473430243205
        struct[0].Gy[42,42] = -2.23667465123725
        struct[0].Gy[42,43] = -1.28353302446119
        struct[0].Gy[42,44] = 0.643671749092997
        struct[0].Gy[42,45] = 0.385473430243204
        struct[0].Gy[42,46] = -0.316443717683753
        struct[0].Gy[42,47] = -0.171091579417532
        struct[0].Gy[42,48] = 1.10755301189314
        struct[0].Gy[42,49] = 0.598820527961360
        struct[0].Gy[42,50] = -0.316443717683753
        struct[0].Gy[42,51] = -0.171091579417531
        struct[0].Gy[43,2] = -0.996212229189942
        struct[0].Gy[43,3] = 0.249053057297486
        struct[0].Gy[43,4] = 0.996212229189942
        struct[0].Gy[43,5] = -0.249053057297486
        struct[0].Gy[43,40] = -0.385473430243205
        struct[0].Gy[43,41] = 0.643671749092996
        struct[0].Gy[43,42] = 1.28353302446119
        struct[0].Gy[43,43] = -2.23667465123725
        struct[0].Gy[43,44] = -0.385473430243204
        struct[0].Gy[43,45] = 0.643671749092997
        struct[0].Gy[43,46] = 0.171091579417532
        struct[0].Gy[43,47] = -0.316443717683753
        struct[0].Gy[43,48] = -0.598820527961360
        struct[0].Gy[43,49] = 1.10755301189314
        struct[0].Gy[43,50] = 0.171091579417531
        struct[0].Gy[43,51] = -0.316443717683753
        struct[0].Gy[44,0] = -0.249053057297486
        struct[0].Gy[44,1] = -0.996212229189942
        struct[0].Gy[44,4] = 0.249053057297486
        struct[0].Gy[44,5] = 0.996212229189942
        struct[0].Gy[44,40] = 0.643671749092997
        struct[0].Gy[44,41] = 0.385473430243205
        struct[0].Gy[44,42] = 0.643671749092997
        struct[0].Gy[44,43] = 0.385473430243204
        struct[0].Gy[44,44] = -2.23667465123725
        struct[0].Gy[44,45] = -1.28353302446119
        struct[0].Gy[44,46] = -0.316443717683753
        struct[0].Gy[44,47] = -0.171091579417532
        struct[0].Gy[44,48] = -0.316443717683753
        struct[0].Gy[44,49] = -0.171091579417531
        struct[0].Gy[44,50] = 1.10755301189314
        struct[0].Gy[44,51] = 0.598820527961360
        struct[0].Gy[45,0] = 0.996212229189942
        struct[0].Gy[45,1] = -0.249053057297486
        struct[0].Gy[45,4] = -0.996212229189942
        struct[0].Gy[45,5] = 0.249053057297486
        struct[0].Gy[45,40] = -0.385473430243205
        struct[0].Gy[45,41] = 0.643671749092997
        struct[0].Gy[45,42] = -0.385473430243204
        struct[0].Gy[45,43] = 0.643671749092997
        struct[0].Gy[45,44] = 1.28353302446119
        struct[0].Gy[45,45] = -2.23667465123725
        struct[0].Gy[45,46] = 0.171091579417532
        struct[0].Gy[45,47] = -0.316443717683753
        struct[0].Gy[45,48] = 0.171091579417531
        struct[0].Gy[45,49] = -0.316443717683753
        struct[0].Gy[45,50] = -0.598820527961360
        struct[0].Gy[45,51] = 1.10755301189314
        struct[0].Gy[46,8] = 0.249053057297486
        struct[0].Gy[46,9] = 0.996212229189942
        struct[0].Gy[46,10] = -0.249053057297486
        struct[0].Gy[46,11] = -0.996212229189942
        struct[0].Gy[46,40] = 1.10755301189314
        struct[0].Gy[46,41] = 0.598820527961361
        struct[0].Gy[46,42] = -0.316443717683753
        struct[0].Gy[46,43] = -0.171091579417532
        struct[0].Gy[46,44] = -0.316443717683753
        struct[0].Gy[46,45] = -0.171091579417532
        struct[0].Gy[46,46] = -2.23667465123725
        struct[0].Gy[46,47] = -1.28353302446119
        struct[0].Gy[46,48] = 0.643671749092996
        struct[0].Gy[46,49] = 0.385473430243205
        struct[0].Gy[46,50] = 0.643671749092997
        struct[0].Gy[46,51] = 0.385473430243205
        struct[0].Gy[46,52] = 1.10755301189314
        struct[0].Gy[46,53] = 0.598820527961361
        struct[0].Gy[46,54] = -0.316443717683753
        struct[0].Gy[46,55] = -0.171091579417532
        struct[0].Gy[46,56] = -0.316443717683753
        struct[0].Gy[46,57] = -0.171091579417532
        struct[0].Gy[47,8] = -0.996212229189942
        struct[0].Gy[47,9] = 0.249053057297486
        struct[0].Gy[47,10] = 0.996212229189942
        struct[0].Gy[47,11] = -0.249053057297486
        struct[0].Gy[47,40] = -0.598820527961361
        struct[0].Gy[47,41] = 1.10755301189314
        struct[0].Gy[47,42] = 0.171091579417532
        struct[0].Gy[47,43] = -0.316443717683753
        struct[0].Gy[47,44] = 0.171091579417532
        struct[0].Gy[47,45] = -0.316443717683753
        struct[0].Gy[47,46] = 1.28353302446119
        struct[0].Gy[47,47] = -2.23667465123725
        struct[0].Gy[47,48] = -0.385473430243205
        struct[0].Gy[47,49] = 0.643671749092996
        struct[0].Gy[47,50] = -0.385473430243205
        struct[0].Gy[47,51] = 0.643671749092997
        struct[0].Gy[47,52] = -0.598820527961361
        struct[0].Gy[47,53] = 1.10755301189314
        struct[0].Gy[47,54] = 0.171091579417532
        struct[0].Gy[47,55] = -0.316443717683753
        struct[0].Gy[47,56] = 0.171091579417532
        struct[0].Gy[47,57] = -0.316443717683753
        struct[0].Gy[48,10] = 0.249053057297486
        struct[0].Gy[48,11] = 0.996212229189942
        struct[0].Gy[48,12] = -0.249053057297486
        struct[0].Gy[48,13] = -0.996212229189942
        struct[0].Gy[48,40] = -0.316443717683753
        struct[0].Gy[48,41] = -0.171091579417532
        struct[0].Gy[48,42] = 1.10755301189314
        struct[0].Gy[48,43] = 0.598820527961360
        struct[0].Gy[48,44] = -0.316443717683753
        struct[0].Gy[48,45] = -0.171091579417531
        struct[0].Gy[48,46] = 0.643671749092996
        struct[0].Gy[48,47] = 0.385473430243205
        struct[0].Gy[48,48] = -2.23667465123725
        struct[0].Gy[48,49] = -1.28353302446119
        struct[0].Gy[48,50] = 0.643671749092997
        struct[0].Gy[48,51] = 0.385473430243204
        struct[0].Gy[48,52] = -0.316443717683753
        struct[0].Gy[48,53] = -0.171091579417532
        struct[0].Gy[48,54] = 1.10755301189314
        struct[0].Gy[48,55] = 0.598820527961360
        struct[0].Gy[48,56] = -0.316443717683753
        struct[0].Gy[48,57] = -0.171091579417531
        struct[0].Gy[49,10] = -0.996212229189942
        struct[0].Gy[49,11] = 0.249053057297486
        struct[0].Gy[49,12] = 0.996212229189942
        struct[0].Gy[49,13] = -0.249053057297486
        struct[0].Gy[49,40] = 0.171091579417532
        struct[0].Gy[49,41] = -0.316443717683753
        struct[0].Gy[49,42] = -0.598820527961360
        struct[0].Gy[49,43] = 1.10755301189314
        struct[0].Gy[49,44] = 0.171091579417531
        struct[0].Gy[49,45] = -0.316443717683753
        struct[0].Gy[49,46] = -0.385473430243205
        struct[0].Gy[49,47] = 0.643671749092996
        struct[0].Gy[49,48] = 1.28353302446119
        struct[0].Gy[49,49] = -2.23667465123725
        struct[0].Gy[49,50] = -0.385473430243204
        struct[0].Gy[49,51] = 0.643671749092997
        struct[0].Gy[49,52] = 0.171091579417532
        struct[0].Gy[49,53] = -0.316443717683753
        struct[0].Gy[49,54] = -0.598820527961360
        struct[0].Gy[49,55] = 1.10755301189314
        struct[0].Gy[49,56] = 0.171091579417531
        struct[0].Gy[49,57] = -0.316443717683753
        struct[0].Gy[50,8] = -0.249053057297486
        struct[0].Gy[50,9] = -0.996212229189942
        struct[0].Gy[50,12] = 0.249053057297486
        struct[0].Gy[50,13] = 0.996212229189942
        struct[0].Gy[50,40] = -0.316443717683753
        struct[0].Gy[50,41] = -0.171091579417532
        struct[0].Gy[50,42] = -0.316443717683753
        struct[0].Gy[50,43] = -0.171091579417531
        struct[0].Gy[50,44] = 1.10755301189314
        struct[0].Gy[50,45] = 0.598820527961360
        struct[0].Gy[50,46] = 0.643671749092997
        struct[0].Gy[50,47] = 0.385473430243205
        struct[0].Gy[50,48] = 0.643671749092997
        struct[0].Gy[50,49] = 0.385473430243204
        struct[0].Gy[50,50] = -2.23667465123725
        struct[0].Gy[50,51] = -1.28353302446119
        struct[0].Gy[50,52] = -0.316443717683753
        struct[0].Gy[50,53] = -0.171091579417532
        struct[0].Gy[50,54] = -0.316443717683753
        struct[0].Gy[50,55] = -0.171091579417531
        struct[0].Gy[50,56] = 1.10755301189314
        struct[0].Gy[50,57] = 0.598820527961360
        struct[0].Gy[51,8] = 0.996212229189942
        struct[0].Gy[51,9] = -0.249053057297486
        struct[0].Gy[51,12] = -0.996212229189942
        struct[0].Gy[51,13] = 0.249053057297486
        struct[0].Gy[51,40] = 0.171091579417532
        struct[0].Gy[51,41] = -0.316443717683753
        struct[0].Gy[51,42] = 0.171091579417531
        struct[0].Gy[51,43] = -0.316443717683753
        struct[0].Gy[51,44] = -0.598820527961360
        struct[0].Gy[51,45] = 1.10755301189314
        struct[0].Gy[51,46] = -0.385473430243205
        struct[0].Gy[51,47] = 0.643671749092997
        struct[0].Gy[51,48] = -0.385473430243204
        struct[0].Gy[51,49] = 0.643671749092997
        struct[0].Gy[51,50] = 1.28353302446119
        struct[0].Gy[51,51] = -2.23667465123725
        struct[0].Gy[51,52] = 0.171091579417532
        struct[0].Gy[51,53] = -0.316443717683753
        struct[0].Gy[51,54] = 0.171091579417531
        struct[0].Gy[51,55] = -0.316443717683753
        struct[0].Gy[51,56] = -0.598820527961360
        struct[0].Gy[51,57] = 1.10755301189314
        struct[0].Gy[52,16] = 0.249053057297486
        struct[0].Gy[52,17] = 0.996212229189942
        struct[0].Gy[52,18] = -0.249053057297486
        struct[0].Gy[52,19] = -0.996212229189942
        struct[0].Gy[52,46] = 1.10755301189314
        struct[0].Gy[52,47] = 0.598820527961361
        struct[0].Gy[52,48] = -0.316443717683753
        struct[0].Gy[52,49] = -0.171091579417532
        struct[0].Gy[52,50] = -0.316443717683753
        struct[0].Gy[52,51] = -0.171091579417532
        struct[0].Gy[52,52] = -1.12912163934412
        struct[0].Gy[52,53] = -0.684903767132556
        struct[0].Gy[52,54] = 0.327228031409243
        struct[0].Gy[52,55] = 0.214305342572583
        struct[0].Gy[52,56] = 0.327228031409244
        struct[0].Gy[52,57] = 0.214305342572583
        struct[0].Gy[53,16] = -0.996212229189942
        struct[0].Gy[53,17] = 0.249053057297486
        struct[0].Gy[53,18] = 0.996212229189942
        struct[0].Gy[53,19] = -0.249053057297486
        struct[0].Gy[53,46] = -0.598820527961361
        struct[0].Gy[53,47] = 1.10755301189314
        struct[0].Gy[53,48] = 0.171091579417532
        struct[0].Gy[53,49] = -0.316443717683753
        struct[0].Gy[53,50] = 0.171091579417532
        struct[0].Gy[53,51] = -0.316443717683753
        struct[0].Gy[53,52] = 0.684903767132556
        struct[0].Gy[53,53] = -1.12912163934412
        struct[0].Gy[53,54] = -0.214305342572583
        struct[0].Gy[53,55] = 0.327228031409243
        struct[0].Gy[53,56] = -0.214305342572583
        struct[0].Gy[53,57] = 0.327228031409244
        struct[0].Gy[54,18] = 0.249053057297486
        struct[0].Gy[54,19] = 0.996212229189942
        struct[0].Gy[54,20] = -0.249053057297486
        struct[0].Gy[54,21] = -0.996212229189942
        struct[0].Gy[54,46] = -0.316443717683753
        struct[0].Gy[54,47] = -0.171091579417532
        struct[0].Gy[54,48] = 1.10755301189314
        struct[0].Gy[54,49] = 0.598820527961360
        struct[0].Gy[54,50] = -0.316443717683753
        struct[0].Gy[54,51] = -0.171091579417531
        struct[0].Gy[54,52] = 0.327228031409243
        struct[0].Gy[54,53] = 0.214305342572583
        struct[0].Gy[54,54] = -1.12912163934412
        struct[0].Gy[54,55] = -0.684903767132556
        struct[0].Gy[54,56] = 0.327228031409244
        struct[0].Gy[54,57] = 0.214305342572582
        struct[0].Gy[55,18] = -0.996212229189942
        struct[0].Gy[55,19] = 0.249053057297486
        struct[0].Gy[55,20] = 0.996212229189942
        struct[0].Gy[55,21] = -0.249053057297486
        struct[0].Gy[55,46] = 0.171091579417532
        struct[0].Gy[55,47] = -0.316443717683753
        struct[0].Gy[55,48] = -0.598820527961360
        struct[0].Gy[55,49] = 1.10755301189314
        struct[0].Gy[55,50] = 0.171091579417531
        struct[0].Gy[55,51] = -0.316443717683753
        struct[0].Gy[55,52] = -0.214305342572583
        struct[0].Gy[55,53] = 0.327228031409243
        struct[0].Gy[55,54] = 0.684903767132556
        struct[0].Gy[55,55] = -1.12912163934412
        struct[0].Gy[55,56] = -0.214305342572582
        struct[0].Gy[55,57] = 0.327228031409244
        struct[0].Gy[56,16] = -0.249053057297486
        struct[0].Gy[56,17] = -0.996212229189942
        struct[0].Gy[56,20] = 0.249053057297486
        struct[0].Gy[56,21] = 0.996212229189942
        struct[0].Gy[56,46] = -0.316443717683753
        struct[0].Gy[56,47] = -0.171091579417532
        struct[0].Gy[56,48] = -0.316443717683753
        struct[0].Gy[56,49] = -0.171091579417531
        struct[0].Gy[56,50] = 1.10755301189314
        struct[0].Gy[56,51] = 0.598820527961360
        struct[0].Gy[56,52] = 0.327228031409243
        struct[0].Gy[56,53] = 0.214305342572583
        struct[0].Gy[56,54] = 0.327228031409244
        struct[0].Gy[56,55] = 0.214305342572582
        struct[0].Gy[56,56] = -1.12912163934412
        struct[0].Gy[56,57] = -0.684903767132556
        struct[0].Gy[57,16] = 0.996212229189942
        struct[0].Gy[57,17] = -0.249053057297486
        struct[0].Gy[57,20] = -0.996212229189942
        struct[0].Gy[57,21] = 0.249053057297486
        struct[0].Gy[57,46] = 0.171091579417532
        struct[0].Gy[57,47] = -0.316443717683753
        struct[0].Gy[57,48] = 0.171091579417531
        struct[0].Gy[57,49] = -0.316443717683753
        struct[0].Gy[57,50] = -0.598820527961360
        struct[0].Gy[57,51] = 1.10755301189314
        struct[0].Gy[57,52] = -0.214305342572583
        struct[0].Gy[57,53] = 0.327228031409243
        struct[0].Gy[57,54] = -0.214305342572582
        struct[0].Gy[57,55] = 0.327228031409244
        struct[0].Gy[57,56] = 0.684903767132556
        struct[0].Gy[57,57] = -1.12912163934412
        struct[0].Gy[58,24] = 0.249053057297486
        struct[0].Gy[58,25] = 0.996212229189942
        struct[0].Gy[58,26] = -0.249053057297486
        struct[0].Gy[58,27] = -0.996212229189942
        struct[0].Gy[58,58] = -1.12912163934412
        struct[0].Gy[58,59] = -0.684903767132556
        struct[0].Gy[58,60] = 0.327228031409243
        struct[0].Gy[58,61] = 0.214305342572583
        struct[0].Gy[58,62] = 0.327228031409244
        struct[0].Gy[58,63] = 0.214305342572583
        struct[0].Gy[58,64] = 1.10755301189314
        struct[0].Gy[58,65] = 0.598820527961361
        struct[0].Gy[58,66] = -0.316443717683753
        struct[0].Gy[58,67] = -0.171091579417532
        struct[0].Gy[58,68] = -0.316443717683753
        struct[0].Gy[58,69] = -0.171091579417532
        struct[0].Gy[59,24] = -0.996212229189942
        struct[0].Gy[59,25] = 0.249053057297486
        struct[0].Gy[59,26] = 0.996212229189942
        struct[0].Gy[59,27] = -0.249053057297486
        struct[0].Gy[59,58] = 0.684903767132556
        struct[0].Gy[59,59] = -1.12912163934412
        struct[0].Gy[59,60] = -0.214305342572583
        struct[0].Gy[59,61] = 0.327228031409243
        struct[0].Gy[59,62] = -0.214305342572583
        struct[0].Gy[59,63] = 0.327228031409244
        struct[0].Gy[59,64] = -0.598820527961361
        struct[0].Gy[59,65] = 1.10755301189314
        struct[0].Gy[59,66] = 0.171091579417532
        struct[0].Gy[59,67] = -0.316443717683753
        struct[0].Gy[59,68] = 0.171091579417532
        struct[0].Gy[59,69] = -0.316443717683753
        struct[0].Gy[60,26] = 0.249053057297486
        struct[0].Gy[60,27] = 0.996212229189942
        struct[0].Gy[60,28] = -0.249053057297486
        struct[0].Gy[60,29] = -0.996212229189942
        struct[0].Gy[60,58] = 0.327228031409243
        struct[0].Gy[60,59] = 0.214305342572583
        struct[0].Gy[60,60] = -1.12912163934412
        struct[0].Gy[60,61] = -0.684903767132556
        struct[0].Gy[60,62] = 0.327228031409244
        struct[0].Gy[60,63] = 0.214305342572582
        struct[0].Gy[60,64] = -0.316443717683753
        struct[0].Gy[60,65] = -0.171091579417532
        struct[0].Gy[60,66] = 1.10755301189314
        struct[0].Gy[60,67] = 0.598820527961360
        struct[0].Gy[60,68] = -0.316443717683753
        struct[0].Gy[60,69] = -0.171091579417531
        struct[0].Gy[61,26] = -0.996212229189942
        struct[0].Gy[61,27] = 0.249053057297486
        struct[0].Gy[61,28] = 0.996212229189942
        struct[0].Gy[61,29] = -0.249053057297486
        struct[0].Gy[61,58] = -0.214305342572583
        struct[0].Gy[61,59] = 0.327228031409243
        struct[0].Gy[61,60] = 0.684903767132556
        struct[0].Gy[61,61] = -1.12912163934412
        struct[0].Gy[61,62] = -0.214305342572582
        struct[0].Gy[61,63] = 0.327228031409244
        struct[0].Gy[61,64] = 0.171091579417532
        struct[0].Gy[61,65] = -0.316443717683753
        struct[0].Gy[61,66] = -0.598820527961360
        struct[0].Gy[61,67] = 1.10755301189314
        struct[0].Gy[61,68] = 0.171091579417531
        struct[0].Gy[61,69] = -0.316443717683753
        struct[0].Gy[62,24] = -0.249053057297486
        struct[0].Gy[62,25] = -0.996212229189942
        struct[0].Gy[62,28] = 0.249053057297486
        struct[0].Gy[62,29] = 0.996212229189942
        struct[0].Gy[62,58] = 0.327228031409243
        struct[0].Gy[62,59] = 0.214305342572583
        struct[0].Gy[62,60] = 0.327228031409244
        struct[0].Gy[62,61] = 0.214305342572582
        struct[0].Gy[62,62] = -1.12912163934412
        struct[0].Gy[62,63] = -0.684903767132556
        struct[0].Gy[62,64] = -0.316443717683753
        struct[0].Gy[62,65] = -0.171091579417532
        struct[0].Gy[62,66] = -0.316443717683753
        struct[0].Gy[62,67] = -0.171091579417531
        struct[0].Gy[62,68] = 1.10755301189314
        struct[0].Gy[62,69] = 0.598820527961360
        struct[0].Gy[63,24] = 0.996212229189942
        struct[0].Gy[63,25] = -0.249053057297486
        struct[0].Gy[63,28] = -0.996212229189942
        struct[0].Gy[63,29] = 0.249053057297486
        struct[0].Gy[63,58] = -0.214305342572583
        struct[0].Gy[63,59] = 0.327228031409243
        struct[0].Gy[63,60] = -0.214305342572582
        struct[0].Gy[63,61] = 0.327228031409244
        struct[0].Gy[63,62] = 0.684903767132556
        struct[0].Gy[63,63] = -1.12912163934412
        struct[0].Gy[63,64] = 0.171091579417532
        struct[0].Gy[63,65] = -0.316443717683753
        struct[0].Gy[63,66] = 0.171091579417531
        struct[0].Gy[63,67] = -0.316443717683753
        struct[0].Gy[63,68] = -0.598820527961360
        struct[0].Gy[63,69] = 1.10755301189314
        struct[0].Gy[64,32] = 0.249053057297486
        struct[0].Gy[64,33] = 0.996212229189942
        struct[0].Gy[64,34] = -0.249053057297486
        struct[0].Gy[64,35] = -0.996212229189942
        struct[0].Gy[64,58] = 1.10755301189314
        struct[0].Gy[64,59] = 0.598820527961361
        struct[0].Gy[64,60] = -0.316443717683753
        struct[0].Gy[64,61] = -0.171091579417532
        struct[0].Gy[64,62] = -0.316443717683753
        struct[0].Gy[64,63] = -0.171091579417532
        struct[0].Gy[64,64] = -2.23667465123725
        struct[0].Gy[64,65] = -1.28353302446119
        struct[0].Gy[64,66] = 0.643671749092996
        struct[0].Gy[64,67] = 0.385473430243205
        struct[0].Gy[64,68] = 0.643671749092997
        struct[0].Gy[64,69] = 0.385473430243205
        struct[0].Gy[65,32] = -0.996212229189942
        struct[0].Gy[65,33] = 0.249053057297486
        struct[0].Gy[65,34] = 0.996212229189942
        struct[0].Gy[65,35] = -0.249053057297486
        struct[0].Gy[65,58] = -0.598820527961361
        struct[0].Gy[65,59] = 1.10755301189314
        struct[0].Gy[65,60] = 0.171091579417532
        struct[0].Gy[65,61] = -0.316443717683753
        struct[0].Gy[65,62] = 0.171091579417532
        struct[0].Gy[65,63] = -0.316443717683753
        struct[0].Gy[65,64] = 1.28353302446119
        struct[0].Gy[65,65] = -2.23667465123725
        struct[0].Gy[65,66] = -0.385473430243205
        struct[0].Gy[65,67] = 0.643671749092996
        struct[0].Gy[65,68] = -0.385473430243205
        struct[0].Gy[65,69] = 0.643671749092997
        struct[0].Gy[66,34] = 0.249053057297486
        struct[0].Gy[66,35] = 0.996212229189942
        struct[0].Gy[66,36] = -0.249053057297486
        struct[0].Gy[66,37] = -0.996212229189942
        struct[0].Gy[66,58] = -0.316443717683753
        struct[0].Gy[66,59] = -0.171091579417532
        struct[0].Gy[66,60] = 1.10755301189314
        struct[0].Gy[66,61] = 0.598820527961360
        struct[0].Gy[66,62] = -0.316443717683753
        struct[0].Gy[66,63] = -0.171091579417531
        struct[0].Gy[66,64] = 0.643671749092996
        struct[0].Gy[66,65] = 0.385473430243205
        struct[0].Gy[66,66] = -2.23667465123725
        struct[0].Gy[66,67] = -1.28353302446119
        struct[0].Gy[66,68] = 0.643671749092997
        struct[0].Gy[66,69] = 0.385473430243204
        struct[0].Gy[67,34] = -0.996212229189942
        struct[0].Gy[67,35] = 0.249053057297486
        struct[0].Gy[67,36] = 0.996212229189942
        struct[0].Gy[67,37] = -0.249053057297486
        struct[0].Gy[67,58] = 0.171091579417532
        struct[0].Gy[67,59] = -0.316443717683753
        struct[0].Gy[67,60] = -0.598820527961360
        struct[0].Gy[67,61] = 1.10755301189314
        struct[0].Gy[67,62] = 0.171091579417531
        struct[0].Gy[67,63] = -0.316443717683753
        struct[0].Gy[67,64] = -0.385473430243205
        struct[0].Gy[67,65] = 0.643671749092996
        struct[0].Gy[67,66] = 1.28353302446119
        struct[0].Gy[67,67] = -2.23667465123725
        struct[0].Gy[67,68] = -0.385473430243204
        struct[0].Gy[67,69] = 0.643671749092997
        struct[0].Gy[68,32] = -0.249053057297486
        struct[0].Gy[68,33] = -0.996212229189942
        struct[0].Gy[68,36] = 0.249053057297486
        struct[0].Gy[68,37] = 0.996212229189942
        struct[0].Gy[68,58] = -0.316443717683753
        struct[0].Gy[68,59] = -0.171091579417532
        struct[0].Gy[68,60] = -0.316443717683753
        struct[0].Gy[68,61] = -0.171091579417531
        struct[0].Gy[68,62] = 1.10755301189314
        struct[0].Gy[68,63] = 0.598820527961360
        struct[0].Gy[68,64] = 0.643671749092997
        struct[0].Gy[68,65] = 0.385473430243205
        struct[0].Gy[68,66] = 0.643671749092997
        struct[0].Gy[68,67] = 0.385473430243204
        struct[0].Gy[68,68] = -2.23667465123725
        struct[0].Gy[68,69] = -1.28353302446119
        struct[0].Gy[69,32] = 0.996212229189942
        struct[0].Gy[69,33] = -0.249053057297486
        struct[0].Gy[69,36] = -0.996212229189942
        struct[0].Gy[69,37] = 0.249053057297486
        struct[0].Gy[69,58] = 0.171091579417532
        struct[0].Gy[69,59] = -0.316443717683753
        struct[0].Gy[69,60] = 0.171091579417531
        struct[0].Gy[69,61] = -0.316443717683753
        struct[0].Gy[69,62] = -0.598820527961360
        struct[0].Gy[69,63] = 1.10755301189314
        struct[0].Gy[69,64] = -0.385473430243205
        struct[0].Gy[69,65] = 0.643671749092997
        struct[0].Gy[69,66] = -0.385473430243204
        struct[0].Gy[69,67] = 0.643671749092997
        struct[0].Gy[69,68] = 1.28353302446119
        struct[0].Gy[69,69] = -2.23667465123725
        struct[0].Gy[70,0] = -0.249053057297486
        struct[0].Gy[70,1] = -0.996212229189942
        struct[0].Gy[70,2] = 0.249053057297486
        struct[0].Gy[70,3] = 0.996212229189942
        struct[0].Gy[70,40] = 0.0215686274509804
        struct[0].Gy[70,41] = 0.0862745098039216
        struct[0].Gy[70,42] = -0.0107843137254902
        struct[0].Gy[70,43] = -0.0431372549019608
        struct[0].Gy[70,44] = -0.0107843137254902
        struct[0].Gy[70,45] = -0.0431372549019608
        struct[0].Gy[70,70] = -1
        struct[0].Gy[71,0] = 0.996212229189942
        struct[0].Gy[71,1] = -0.249053057297486
        struct[0].Gy[71,2] = -0.996212229189942
        struct[0].Gy[71,3] = 0.249053057297486
        struct[0].Gy[71,40] = -0.0862745098039216
        struct[0].Gy[71,41] = 0.0215686274509804
        struct[0].Gy[71,42] = 0.0431372549019608
        struct[0].Gy[71,43] = -0.0107843137254902
        struct[0].Gy[71,44] = 0.0431372549019608
        struct[0].Gy[71,45] = -0.0107843137254902
        struct[0].Gy[71,71] = -1
        struct[0].Gy[72,2] = -0.249053057297486
        struct[0].Gy[72,3] = -0.996212229189942
        struct[0].Gy[72,4] = 0.249053057297486
        struct[0].Gy[72,5] = 0.996212229189942
        struct[0].Gy[72,40] = -0.0107843137254902
        struct[0].Gy[72,41] = -0.0431372549019608
        struct[0].Gy[72,42] = 0.0215686274509804
        struct[0].Gy[72,43] = 0.0862745098039216
        struct[0].Gy[72,44] = -0.0107843137254902
        struct[0].Gy[72,45] = -0.0431372549019608
        struct[0].Gy[72,72] = -1
        struct[0].Gy[73,2] = 0.996212229189942
        struct[0].Gy[73,3] = -0.249053057297486
        struct[0].Gy[73,4] = -0.996212229189942
        struct[0].Gy[73,5] = 0.249053057297486
        struct[0].Gy[73,40] = 0.0431372549019608
        struct[0].Gy[73,41] = -0.0107843137254902
        struct[0].Gy[73,42] = -0.0862745098039216
        struct[0].Gy[73,43] = 0.0215686274509804
        struct[0].Gy[73,44] = 0.0431372549019608
        struct[0].Gy[73,45] = -0.0107843137254902
        struct[0].Gy[73,73] = -1
        struct[0].Gy[74,0] = 0.249053057297486
        struct[0].Gy[74,1] = 0.996212229189942
        struct[0].Gy[74,4] = -0.249053057297486
        struct[0].Gy[74,5] = -0.996212229189942
        struct[0].Gy[74,40] = -0.0107843137254902
        struct[0].Gy[74,41] = -0.0431372549019608
        struct[0].Gy[74,42] = -0.0107843137254902
        struct[0].Gy[74,43] = -0.0431372549019608
        struct[0].Gy[74,44] = 0.0215686274509804
        struct[0].Gy[74,45] = 0.0862745098039216
        struct[0].Gy[74,74] = -1
        struct[0].Gy[75,0] = -0.996212229189942
        struct[0].Gy[75,1] = 0.249053057297486
        struct[0].Gy[75,4] = 0.996212229189942
        struct[0].Gy[75,5] = -0.249053057297486
        struct[0].Gy[75,40] = 0.0431372549019608
        struct[0].Gy[75,41] = -0.0107843137254902
        struct[0].Gy[75,42] = 0.0431372549019608
        struct[0].Gy[75,43] = -0.0107843137254902
        struct[0].Gy[75,44] = -0.0862745098039216
        struct[0].Gy[75,45] = 0.0215686274509804
        struct[0].Gy[75,75] = -1
        struct[0].Gy[76,8] = -0.249053057297486
        struct[0].Gy[76,9] = -0.996212229189942
        struct[0].Gy[76,10] = 0.249053057297486
        struct[0].Gy[76,11] = 0.996212229189942
        struct[0].Gy[76,46] = 0.0215686274509804
        struct[0].Gy[76,47] = 0.0862745098039216
        struct[0].Gy[76,48] = -0.0107843137254902
        struct[0].Gy[76,49] = -0.0431372549019608
        struct[0].Gy[76,50] = -0.0107843137254902
        struct[0].Gy[76,51] = -0.0431372549019608
        struct[0].Gy[76,76] = -1
        struct[0].Gy[77,8] = 0.996212229189942
        struct[0].Gy[77,9] = -0.249053057297486
        struct[0].Gy[77,10] = -0.996212229189942
        struct[0].Gy[77,11] = 0.249053057297486
        struct[0].Gy[77,46] = -0.0862745098039216
        struct[0].Gy[77,47] = 0.0215686274509804
        struct[0].Gy[77,48] = 0.0431372549019608
        struct[0].Gy[77,49] = -0.0107843137254902
        struct[0].Gy[77,50] = 0.0431372549019608
        struct[0].Gy[77,51] = -0.0107843137254902
        struct[0].Gy[77,77] = -1
        struct[0].Gy[78,10] = -0.249053057297486
        struct[0].Gy[78,11] = -0.996212229189942
        struct[0].Gy[78,12] = 0.249053057297486
        struct[0].Gy[78,13] = 0.996212229189942
        struct[0].Gy[78,46] = -0.0107843137254902
        struct[0].Gy[78,47] = -0.0431372549019608
        struct[0].Gy[78,48] = 0.0215686274509804
        struct[0].Gy[78,49] = 0.0862745098039216
        struct[0].Gy[78,50] = -0.0107843137254902
        struct[0].Gy[78,51] = -0.0431372549019608
        struct[0].Gy[78,78] = -1
        struct[0].Gy[79,10] = 0.996212229189942
        struct[0].Gy[79,11] = -0.249053057297486
        struct[0].Gy[79,12] = -0.996212229189942
        struct[0].Gy[79,13] = 0.249053057297486
        struct[0].Gy[79,46] = 0.0431372549019608
        struct[0].Gy[79,47] = -0.0107843137254902
        struct[0].Gy[79,48] = -0.0862745098039216
        struct[0].Gy[79,49] = 0.0215686274509804
        struct[0].Gy[79,50] = 0.0431372549019608
        struct[0].Gy[79,51] = -0.0107843137254902
        struct[0].Gy[79,79] = -1
        struct[0].Gy[80,8] = 0.249053057297486
        struct[0].Gy[80,9] = 0.996212229189942
        struct[0].Gy[80,12] = -0.249053057297486
        struct[0].Gy[80,13] = -0.996212229189942
        struct[0].Gy[80,46] = -0.0107843137254902
        struct[0].Gy[80,47] = -0.0431372549019608
        struct[0].Gy[80,48] = -0.0107843137254902
        struct[0].Gy[80,49] = -0.0431372549019608
        struct[0].Gy[80,50] = 0.0215686274509804
        struct[0].Gy[80,51] = 0.0862745098039216
        struct[0].Gy[80,80] = -1
        struct[0].Gy[81,8] = -0.996212229189942
        struct[0].Gy[81,9] = 0.249053057297486
        struct[0].Gy[81,12] = 0.996212229189942
        struct[0].Gy[81,13] = -0.249053057297486
        struct[0].Gy[81,46] = 0.0431372549019608
        struct[0].Gy[81,47] = -0.0107843137254902
        struct[0].Gy[81,48] = 0.0431372549019608
        struct[0].Gy[81,49] = -0.0107843137254902
        struct[0].Gy[81,50] = -0.0862745098039216
        struct[0].Gy[81,51] = 0.0215686274509804
        struct[0].Gy[81,81] = -1
        struct[0].Gy[82,16] = -0.249053057297486
        struct[0].Gy[82,17] = -0.996212229189942
        struct[0].Gy[82,18] = 0.249053057297486
        struct[0].Gy[82,19] = 0.996212229189942
        struct[0].Gy[82,52] = 0.0215686274509804
        struct[0].Gy[82,53] = 0.0862745098039216
        struct[0].Gy[82,54] = -0.0107843137254902
        struct[0].Gy[82,55] = -0.0431372549019608
        struct[0].Gy[82,56] = -0.0107843137254902
        struct[0].Gy[82,57] = -0.0431372549019608
        struct[0].Gy[82,82] = -1
        struct[0].Gy[83,16] = 0.996212229189942
        struct[0].Gy[83,17] = -0.249053057297486
        struct[0].Gy[83,18] = -0.996212229189942
        struct[0].Gy[83,19] = 0.249053057297486
        struct[0].Gy[83,52] = -0.0862745098039216
        struct[0].Gy[83,53] = 0.0215686274509804
        struct[0].Gy[83,54] = 0.0431372549019608
        struct[0].Gy[83,55] = -0.0107843137254902
        struct[0].Gy[83,56] = 0.0431372549019608
        struct[0].Gy[83,57] = -0.0107843137254902
        struct[0].Gy[83,83] = -1
        struct[0].Gy[84,18] = -0.249053057297486
        struct[0].Gy[84,19] = -0.996212229189942
        struct[0].Gy[84,20] = 0.249053057297486
        struct[0].Gy[84,21] = 0.996212229189942
        struct[0].Gy[84,52] = -0.0107843137254902
        struct[0].Gy[84,53] = -0.0431372549019608
        struct[0].Gy[84,54] = 0.0215686274509804
        struct[0].Gy[84,55] = 0.0862745098039216
        struct[0].Gy[84,56] = -0.0107843137254902
        struct[0].Gy[84,57] = -0.0431372549019608
        struct[0].Gy[84,84] = -1
        struct[0].Gy[85,18] = 0.996212229189942
        struct[0].Gy[85,19] = -0.249053057297486
        struct[0].Gy[85,20] = -0.996212229189942
        struct[0].Gy[85,21] = 0.249053057297486
        struct[0].Gy[85,52] = 0.0431372549019608
        struct[0].Gy[85,53] = -0.0107843137254902
        struct[0].Gy[85,54] = -0.0862745098039216
        struct[0].Gy[85,55] = 0.0215686274509804
        struct[0].Gy[85,56] = 0.0431372549019608
        struct[0].Gy[85,57] = -0.0107843137254902
        struct[0].Gy[85,85] = -1
        struct[0].Gy[86,16] = 0.249053057297486
        struct[0].Gy[86,17] = 0.996212229189942
        struct[0].Gy[86,20] = -0.249053057297486
        struct[0].Gy[86,21] = -0.996212229189942
        struct[0].Gy[86,52] = -0.0107843137254902
        struct[0].Gy[86,53] = -0.0431372549019608
        struct[0].Gy[86,54] = -0.0107843137254902
        struct[0].Gy[86,55] = -0.0431372549019608
        struct[0].Gy[86,56] = 0.0215686274509804
        struct[0].Gy[86,57] = 0.0862745098039216
        struct[0].Gy[86,86] = -1
        struct[0].Gy[87,16] = -0.996212229189942
        struct[0].Gy[87,17] = 0.249053057297486
        struct[0].Gy[87,20] = 0.996212229189942
        struct[0].Gy[87,21] = -0.249053057297486
        struct[0].Gy[87,52] = 0.0431372549019608
        struct[0].Gy[87,53] = -0.0107843137254902
        struct[0].Gy[87,54] = 0.0431372549019608
        struct[0].Gy[87,55] = -0.0107843137254902
        struct[0].Gy[87,56] = -0.0862745098039216
        struct[0].Gy[87,57] = 0.0215686274509804
        struct[0].Gy[87,87] = -1
        struct[0].Gy[88,24] = -0.249053057297486
        struct[0].Gy[88,25] = -0.996212229189942
        struct[0].Gy[88,26] = 0.249053057297486
        struct[0].Gy[88,27] = 0.996212229189942
        struct[0].Gy[88,58] = 0.0215686274509804
        struct[0].Gy[88,59] = 0.0862745098039216
        struct[0].Gy[88,60] = -0.0107843137254902
        struct[0].Gy[88,61] = -0.0431372549019608
        struct[0].Gy[88,62] = -0.0107843137254902
        struct[0].Gy[88,63] = -0.0431372549019608
        struct[0].Gy[88,88] = -1
        struct[0].Gy[89,24] = 0.996212229189942
        struct[0].Gy[89,25] = -0.249053057297486
        struct[0].Gy[89,26] = -0.996212229189942
        struct[0].Gy[89,27] = 0.249053057297486
        struct[0].Gy[89,58] = -0.0862745098039216
        struct[0].Gy[89,59] = 0.0215686274509804
        struct[0].Gy[89,60] = 0.0431372549019608
        struct[0].Gy[89,61] = -0.0107843137254902
        struct[0].Gy[89,62] = 0.0431372549019608
        struct[0].Gy[89,63] = -0.0107843137254902
        struct[0].Gy[89,89] = -1
        struct[0].Gy[90,26] = -0.249053057297486
        struct[0].Gy[90,27] = -0.996212229189942
        struct[0].Gy[90,28] = 0.249053057297486
        struct[0].Gy[90,29] = 0.996212229189942
        struct[0].Gy[90,58] = -0.0107843137254902
        struct[0].Gy[90,59] = -0.0431372549019608
        struct[0].Gy[90,60] = 0.0215686274509804
        struct[0].Gy[90,61] = 0.0862745098039216
        struct[0].Gy[90,62] = -0.0107843137254902
        struct[0].Gy[90,63] = -0.0431372549019608
        struct[0].Gy[90,90] = -1
        struct[0].Gy[91,26] = 0.996212229189942
        struct[0].Gy[91,27] = -0.249053057297486
        struct[0].Gy[91,28] = -0.996212229189942
        struct[0].Gy[91,29] = 0.249053057297486
        struct[0].Gy[91,58] = 0.0431372549019608
        struct[0].Gy[91,59] = -0.0107843137254902
        struct[0].Gy[91,60] = -0.0862745098039216
        struct[0].Gy[91,61] = 0.0215686274509804
        struct[0].Gy[91,62] = 0.0431372549019608
        struct[0].Gy[91,63] = -0.0107843137254902
        struct[0].Gy[91,91] = -1
        struct[0].Gy[92,24] = 0.249053057297486
        struct[0].Gy[92,25] = 0.996212229189942
        struct[0].Gy[92,28] = -0.249053057297486
        struct[0].Gy[92,29] = -0.996212229189942
        struct[0].Gy[92,58] = -0.0107843137254902
        struct[0].Gy[92,59] = -0.0431372549019608
        struct[0].Gy[92,60] = -0.0107843137254902
        struct[0].Gy[92,61] = -0.0431372549019608
        struct[0].Gy[92,62] = 0.0215686274509804
        struct[0].Gy[92,63] = 0.0862745098039216
        struct[0].Gy[92,92] = -1
        struct[0].Gy[93,24] = -0.996212229189942
        struct[0].Gy[93,25] = 0.249053057297486
        struct[0].Gy[93,28] = 0.996212229189942
        struct[0].Gy[93,29] = -0.249053057297486
        struct[0].Gy[93,58] = 0.0431372549019608
        struct[0].Gy[93,59] = -0.0107843137254902
        struct[0].Gy[93,60] = 0.0431372549019608
        struct[0].Gy[93,61] = -0.0107843137254902
        struct[0].Gy[93,62] = -0.0862745098039216
        struct[0].Gy[93,63] = 0.0215686274509804
        struct[0].Gy[93,93] = -1
        struct[0].Gy[94,32] = -0.249053057297486
        struct[0].Gy[94,33] = -0.996212229189942
        struct[0].Gy[94,34] = 0.249053057297486
        struct[0].Gy[94,35] = 0.996212229189942
        struct[0].Gy[94,64] = 0.0215686274509804
        struct[0].Gy[94,65] = 0.0862745098039216
        struct[0].Gy[94,66] = -0.0107843137254902
        struct[0].Gy[94,67] = -0.0431372549019608
        struct[0].Gy[94,68] = -0.0107843137254902
        struct[0].Gy[94,69] = -0.0431372549019608
        struct[0].Gy[94,94] = -1
        struct[0].Gy[95,32] = 0.996212229189942
        struct[0].Gy[95,33] = -0.249053057297486
        struct[0].Gy[95,34] = -0.996212229189942
        struct[0].Gy[95,35] = 0.249053057297486
        struct[0].Gy[95,64] = -0.0862745098039216
        struct[0].Gy[95,65] = 0.0215686274509804
        struct[0].Gy[95,66] = 0.0431372549019608
        struct[0].Gy[95,67] = -0.0107843137254902
        struct[0].Gy[95,68] = 0.0431372549019608
        struct[0].Gy[95,69] = -0.0107843137254902
        struct[0].Gy[95,95] = -1
        struct[0].Gy[96,34] = -0.249053057297486
        struct[0].Gy[96,35] = -0.996212229189942
        struct[0].Gy[96,36] = 0.249053057297486
        struct[0].Gy[96,37] = 0.996212229189942
        struct[0].Gy[96,64] = -0.0107843137254902
        struct[0].Gy[96,65] = -0.0431372549019608
        struct[0].Gy[96,66] = 0.0215686274509804
        struct[0].Gy[96,67] = 0.0862745098039216
        struct[0].Gy[96,68] = -0.0107843137254902
        struct[0].Gy[96,69] = -0.0431372549019608
        struct[0].Gy[96,96] = -1
        struct[0].Gy[97,34] = 0.996212229189942
        struct[0].Gy[97,35] = -0.249053057297486
        struct[0].Gy[97,36] = -0.996212229189942
        struct[0].Gy[97,37] = 0.249053057297486
        struct[0].Gy[97,64] = 0.0431372549019608
        struct[0].Gy[97,65] = -0.0107843137254902
        struct[0].Gy[97,66] = -0.0862745098039216
        struct[0].Gy[97,67] = 0.0215686274509804
        struct[0].Gy[97,68] = 0.0431372549019608
        struct[0].Gy[97,69] = -0.0107843137254902
        struct[0].Gy[97,97] = -1
        struct[0].Gy[98,32] = 0.249053057297486
        struct[0].Gy[98,33] = 0.996212229189942
        struct[0].Gy[98,36] = -0.249053057297486
        struct[0].Gy[98,37] = -0.996212229189942
        struct[0].Gy[98,64] = -0.0107843137254902
        struct[0].Gy[98,65] = -0.0431372549019608
        struct[0].Gy[98,66] = -0.0107843137254902
        struct[0].Gy[98,67] = -0.0431372549019608
        struct[0].Gy[98,68] = 0.0215686274509804
        struct[0].Gy[98,69] = 0.0862745098039216
        struct[0].Gy[98,98] = -1
        struct[0].Gy[99,32] = -0.996212229189942
        struct[0].Gy[99,33] = 0.249053057297486
        struct[0].Gy[99,36] = 0.996212229189942
        struct[0].Gy[99,37] = -0.249053057297486
        struct[0].Gy[99,64] = 0.0431372549019608
        struct[0].Gy[99,65] = -0.0107843137254902
        struct[0].Gy[99,66] = 0.0431372549019608
        struct[0].Gy[99,67] = -0.0107843137254902
        struct[0].Gy[99,68] = -0.0862745098039216
        struct[0].Gy[99,69] = 0.0215686274509804
        struct[0].Gy[99,99] = -1
        struct[0].Gy[100,40] = -1.10755301189314
        struct[0].Gy[100,41] = -0.598820527961361
        struct[0].Gy[100,42] = 0.316443717683753
        struct[0].Gy[100,43] = 0.171091579417532
        struct[0].Gy[100,44] = 0.316443717683753
        struct[0].Gy[100,45] = 0.171091579417532
        struct[0].Gy[100,100] = -1
        struct[0].Gy[101,40] = 0.598820527961361
        struct[0].Gy[101,41] = -1.10755301189314
        struct[0].Gy[101,42] = -0.171091579417532
        struct[0].Gy[101,43] = 0.316443717683753
        struct[0].Gy[101,44] = -0.171091579417532
        struct[0].Gy[101,45] = 0.316443717683753
        struct[0].Gy[101,101] = -1
        struct[0].Gy[102,40] = 0.316443717683753
        struct[0].Gy[102,41] = 0.171091579417532
        struct[0].Gy[102,42] = -1.10755301189314
        struct[0].Gy[102,43] = -0.598820527961360
        struct[0].Gy[102,44] = 0.316443717683753
        struct[0].Gy[102,45] = 0.171091579417531
        struct[0].Gy[102,102] = -1
        struct[0].Gy[103,40] = -0.171091579417532
        struct[0].Gy[103,41] = 0.316443717683753
        struct[0].Gy[103,42] = 0.598820527961360
        struct[0].Gy[103,43] = -1.10755301189314
        struct[0].Gy[103,44] = -0.171091579417531
        struct[0].Gy[103,45] = 0.316443717683753
        struct[0].Gy[103,103] = -1
        struct[0].Gy[104,40] = 0.316443717683753
        struct[0].Gy[104,41] = 0.171091579417532
        struct[0].Gy[104,42] = 0.316443717683753
        struct[0].Gy[104,43] = 0.171091579417531
        struct[0].Gy[104,44] = -1.10755301189314
        struct[0].Gy[104,45] = -0.598820527961360
        struct[0].Gy[104,104] = -1
        struct[0].Gy[105,40] = -0.171091579417532
        struct[0].Gy[105,41] = 0.316443717683753
        struct[0].Gy[105,42] = -0.171091579417531
        struct[0].Gy[105,43] = 0.316443717683753
        struct[0].Gy[105,44] = 0.598820527961360
        struct[0].Gy[105,45] = -1.10755301189314
        struct[0].Gy[105,105] = -1
        struct[0].Gy[106,40] = 1.10755301189314
        struct[0].Gy[106,41] = 0.598820527961361
        struct[0].Gy[106,42] = -0.316443717683753
        struct[0].Gy[106,43] = -0.171091579417532
        struct[0].Gy[106,44] = -0.316443717683753
        struct[0].Gy[106,45] = -0.171091579417532
        struct[0].Gy[106,46] = -1.10755301189314
        struct[0].Gy[106,47] = -0.598820527961361
        struct[0].Gy[106,48] = 0.316443717683753
        struct[0].Gy[106,49] = 0.171091579417532
        struct[0].Gy[106,50] = 0.316443717683753
        struct[0].Gy[106,51] = 0.171091579417532
        struct[0].Gy[106,106] = -1
        struct[0].Gy[107,40] = -0.598820527961361
        struct[0].Gy[107,41] = 1.10755301189314
        struct[0].Gy[107,42] = 0.171091579417532
        struct[0].Gy[107,43] = -0.316443717683753
        struct[0].Gy[107,44] = 0.171091579417532
        struct[0].Gy[107,45] = -0.316443717683753
        struct[0].Gy[107,46] = 0.598820527961361
        struct[0].Gy[107,47] = -1.10755301189314
        struct[0].Gy[107,48] = -0.171091579417532
        struct[0].Gy[107,49] = 0.316443717683753
        struct[0].Gy[107,50] = -0.171091579417532
        struct[0].Gy[107,51] = 0.316443717683753
        struct[0].Gy[107,107] = -1
        struct[0].Gy[108,40] = -0.316443717683753
        struct[0].Gy[108,41] = -0.171091579417532
        struct[0].Gy[108,42] = 1.10755301189314
        struct[0].Gy[108,43] = 0.598820527961360
        struct[0].Gy[108,44] = -0.316443717683753
        struct[0].Gy[108,45] = -0.171091579417531
        struct[0].Gy[108,46] = 0.316443717683753
        struct[0].Gy[108,47] = 0.171091579417532
        struct[0].Gy[108,48] = -1.10755301189314
        struct[0].Gy[108,49] = -0.598820527961360
        struct[0].Gy[108,50] = 0.316443717683753
        struct[0].Gy[108,51] = 0.171091579417531
        struct[0].Gy[108,108] = -1
        struct[0].Gy[109,40] = 0.171091579417532
        struct[0].Gy[109,41] = -0.316443717683753
        struct[0].Gy[109,42] = -0.598820527961360
        struct[0].Gy[109,43] = 1.10755301189314
        struct[0].Gy[109,44] = 0.171091579417531
        struct[0].Gy[109,45] = -0.316443717683753
        struct[0].Gy[109,46] = -0.171091579417532
        struct[0].Gy[109,47] = 0.316443717683753
        struct[0].Gy[109,48] = 0.598820527961360
        struct[0].Gy[109,49] = -1.10755301189314
        struct[0].Gy[109,50] = -0.171091579417531
        struct[0].Gy[109,51] = 0.316443717683753
        struct[0].Gy[109,109] = -1
        struct[0].Gy[110,40] = -0.316443717683753
        struct[0].Gy[110,41] = -0.171091579417532
        struct[0].Gy[110,42] = -0.316443717683753
        struct[0].Gy[110,43] = -0.171091579417531
        struct[0].Gy[110,44] = 1.10755301189314
        struct[0].Gy[110,45] = 0.598820527961360
        struct[0].Gy[110,46] = 0.316443717683753
        struct[0].Gy[110,47] = 0.171091579417532
        struct[0].Gy[110,48] = 0.316443717683753
        struct[0].Gy[110,49] = 0.171091579417531
        struct[0].Gy[110,50] = -1.10755301189314
        struct[0].Gy[110,51] = -0.598820527961360
        struct[0].Gy[110,110] = -1
        struct[0].Gy[111,40] = 0.171091579417532
        struct[0].Gy[111,41] = -0.316443717683753
        struct[0].Gy[111,42] = 0.171091579417531
        struct[0].Gy[111,43] = -0.316443717683753
        struct[0].Gy[111,44] = -0.598820527961360
        struct[0].Gy[111,45] = 1.10755301189314
        struct[0].Gy[111,46] = -0.171091579417532
        struct[0].Gy[111,47] = 0.316443717683753
        struct[0].Gy[111,48] = -0.171091579417531
        struct[0].Gy[111,49] = 0.316443717683753
        struct[0].Gy[111,50] = 0.598820527961360
        struct[0].Gy[111,51] = -1.10755301189314
        struct[0].Gy[111,111] = -1
        struct[0].Gy[112,46] = 1.10755301189314
        struct[0].Gy[112,47] = 0.598820527961361
        struct[0].Gy[112,48] = -0.316443717683753
        struct[0].Gy[112,49] = -0.171091579417532
        struct[0].Gy[112,50] = -0.316443717683753
        struct[0].Gy[112,51] = -0.171091579417532
        struct[0].Gy[112,52] = -1.10755301189314
        struct[0].Gy[112,53] = -0.598820527961361
        struct[0].Gy[112,54] = 0.316443717683753
        struct[0].Gy[112,55] = 0.171091579417532
        struct[0].Gy[112,56] = 0.316443717683753
        struct[0].Gy[112,57] = 0.171091579417532
        struct[0].Gy[112,112] = -1
        struct[0].Gy[113,46] = -0.598820527961361
        struct[0].Gy[113,47] = 1.10755301189314
        struct[0].Gy[113,48] = 0.171091579417532
        struct[0].Gy[113,49] = -0.316443717683753
        struct[0].Gy[113,50] = 0.171091579417532
        struct[0].Gy[113,51] = -0.316443717683753
        struct[0].Gy[113,52] = 0.598820527961361
        struct[0].Gy[113,53] = -1.10755301189314
        struct[0].Gy[113,54] = -0.171091579417532
        struct[0].Gy[113,55] = 0.316443717683753
        struct[0].Gy[113,56] = -0.171091579417532
        struct[0].Gy[113,57] = 0.316443717683753
        struct[0].Gy[113,113] = -1
        struct[0].Gy[114,46] = -0.316443717683753
        struct[0].Gy[114,47] = -0.171091579417532
        struct[0].Gy[114,48] = 1.10755301189314
        struct[0].Gy[114,49] = 0.598820527961360
        struct[0].Gy[114,50] = -0.316443717683753
        struct[0].Gy[114,51] = -0.171091579417531
        struct[0].Gy[114,52] = 0.316443717683753
        struct[0].Gy[114,53] = 0.171091579417532
        struct[0].Gy[114,54] = -1.10755301189314
        struct[0].Gy[114,55] = -0.598820527961360
        struct[0].Gy[114,56] = 0.316443717683753
        struct[0].Gy[114,57] = 0.171091579417531
        struct[0].Gy[114,114] = -1
        struct[0].Gy[115,46] = 0.171091579417532
        struct[0].Gy[115,47] = -0.316443717683753
        struct[0].Gy[115,48] = -0.598820527961360
        struct[0].Gy[115,49] = 1.10755301189314
        struct[0].Gy[115,50] = 0.171091579417531
        struct[0].Gy[115,51] = -0.316443717683753
        struct[0].Gy[115,52] = -0.171091579417532
        struct[0].Gy[115,53] = 0.316443717683753
        struct[0].Gy[115,54] = 0.598820527961360
        struct[0].Gy[115,55] = -1.10755301189314
        struct[0].Gy[115,56] = -0.171091579417531
        struct[0].Gy[115,57] = 0.316443717683753
        struct[0].Gy[115,115] = -1
        struct[0].Gy[116,46] = -0.316443717683753
        struct[0].Gy[116,47] = -0.171091579417532
        struct[0].Gy[116,48] = -0.316443717683753
        struct[0].Gy[116,49] = -0.171091579417531
        struct[0].Gy[116,50] = 1.10755301189314
        struct[0].Gy[116,51] = 0.598820527961360
        struct[0].Gy[116,52] = 0.316443717683753
        struct[0].Gy[116,53] = 0.171091579417532
        struct[0].Gy[116,54] = 0.316443717683753
        struct[0].Gy[116,55] = 0.171091579417531
        struct[0].Gy[116,56] = -1.10755301189314
        struct[0].Gy[116,57] = -0.598820527961360
        struct[0].Gy[116,116] = -1
        struct[0].Gy[117,46] = 0.171091579417532
        struct[0].Gy[117,47] = -0.316443717683753
        struct[0].Gy[117,48] = 0.171091579417531
        struct[0].Gy[117,49] = -0.316443717683753
        struct[0].Gy[117,50] = -0.598820527961360
        struct[0].Gy[117,51] = 1.10755301189314
        struct[0].Gy[117,52] = -0.171091579417532
        struct[0].Gy[117,53] = 0.316443717683753
        struct[0].Gy[117,54] = -0.171091579417531
        struct[0].Gy[117,55] = 0.316443717683753
        struct[0].Gy[117,56] = 0.598820527961360
        struct[0].Gy[117,57] = -1.10755301189314
        struct[0].Gy[117,117] = -1
        struct[0].Gy[118,58] = 1.10755301189314
        struct[0].Gy[118,59] = 0.598820527961361
        struct[0].Gy[118,60] = -0.316443717683753
        struct[0].Gy[118,61] = -0.171091579417532
        struct[0].Gy[118,62] = -0.316443717683753
        struct[0].Gy[118,63] = -0.171091579417532
        struct[0].Gy[118,64] = -1.10755301189314
        struct[0].Gy[118,65] = -0.598820527961361
        struct[0].Gy[118,66] = 0.316443717683753
        struct[0].Gy[118,67] = 0.171091579417532
        struct[0].Gy[118,68] = 0.316443717683753
        struct[0].Gy[118,69] = 0.171091579417532
        struct[0].Gy[118,118] = -1
        struct[0].Gy[119,58] = -0.598820527961361
        struct[0].Gy[119,59] = 1.10755301189314
        struct[0].Gy[119,60] = 0.171091579417532
        struct[0].Gy[119,61] = -0.316443717683753
        struct[0].Gy[119,62] = 0.171091579417532
        struct[0].Gy[119,63] = -0.316443717683753
        struct[0].Gy[119,64] = 0.598820527961361
        struct[0].Gy[119,65] = -1.10755301189314
        struct[0].Gy[119,66] = -0.171091579417532
        struct[0].Gy[119,67] = 0.316443717683753
        struct[0].Gy[119,68] = -0.171091579417532
        struct[0].Gy[119,69] = 0.316443717683753
        struct[0].Gy[119,119] = -1
        struct[0].Gy[120,58] = -0.316443717683753
        struct[0].Gy[120,59] = -0.171091579417532
        struct[0].Gy[120,60] = 1.10755301189314
        struct[0].Gy[120,61] = 0.598820527961360
        struct[0].Gy[120,62] = -0.316443717683753
        struct[0].Gy[120,63] = -0.171091579417531
        struct[0].Gy[120,64] = 0.316443717683753
        struct[0].Gy[120,65] = 0.171091579417532
        struct[0].Gy[120,66] = -1.10755301189314
        struct[0].Gy[120,67] = -0.598820527961360
        struct[0].Gy[120,68] = 0.316443717683753
        struct[0].Gy[120,69] = 0.171091579417531
        struct[0].Gy[120,120] = -1
        struct[0].Gy[121,58] = 0.171091579417532
        struct[0].Gy[121,59] = -0.316443717683753
        struct[0].Gy[121,60] = -0.598820527961360
        struct[0].Gy[121,61] = 1.10755301189314
        struct[0].Gy[121,62] = 0.171091579417531
        struct[0].Gy[121,63] = -0.316443717683753
        struct[0].Gy[121,64] = -0.171091579417532
        struct[0].Gy[121,65] = 0.316443717683753
        struct[0].Gy[121,66] = 0.598820527961360
        struct[0].Gy[121,67] = -1.10755301189314
        struct[0].Gy[121,68] = -0.171091579417531
        struct[0].Gy[121,69] = 0.316443717683753
        struct[0].Gy[121,121] = -1
        struct[0].Gy[122,58] = -0.316443717683753
        struct[0].Gy[122,59] = -0.171091579417532
        struct[0].Gy[122,60] = -0.316443717683753
        struct[0].Gy[122,61] = -0.171091579417531
        struct[0].Gy[122,62] = 1.10755301189314
        struct[0].Gy[122,63] = 0.598820527961360
        struct[0].Gy[122,64] = 0.316443717683753
        struct[0].Gy[122,65] = 0.171091579417532
        struct[0].Gy[122,66] = 0.316443717683753
        struct[0].Gy[122,67] = 0.171091579417531
        struct[0].Gy[122,68] = -1.10755301189314
        struct[0].Gy[122,69] = -0.598820527961360
        struct[0].Gy[122,122] = -1
        struct[0].Gy[123,58] = 0.171091579417532
        struct[0].Gy[123,59] = -0.316443717683753
        struct[0].Gy[123,60] = 0.171091579417531
        struct[0].Gy[123,61] = -0.316443717683753
        struct[0].Gy[123,62] = -0.598820527961360
        struct[0].Gy[123,63] = 1.10755301189314
        struct[0].Gy[123,64] = -0.171091579417532
        struct[0].Gy[123,65] = 0.316443717683753
        struct[0].Gy[123,66] = -0.171091579417531
        struct[0].Gy[123,67] = 0.316443717683753
        struct[0].Gy[123,68] = 0.598820527961360
        struct[0].Gy[123,69] = -1.10755301189314
        struct[0].Gy[123,123] = -1
        struct[0].Gy[124,64] = 1.10755301189314
        struct[0].Gy[124,65] = 0.598820527961361
        struct[0].Gy[124,66] = -0.316443717683753
        struct[0].Gy[124,67] = -0.171091579417532
        struct[0].Gy[124,68] = -0.316443717683753
        struct[0].Gy[124,69] = -0.171091579417532
        struct[0].Gy[124,124] = -1
        struct[0].Gy[125,64] = -0.598820527961361
        struct[0].Gy[125,65] = 1.10755301189314
        struct[0].Gy[125,66] = 0.171091579417532
        struct[0].Gy[125,67] = -0.316443717683753
        struct[0].Gy[125,68] = 0.171091579417532
        struct[0].Gy[125,69] = -0.316443717683753
        struct[0].Gy[125,125] = -1
        struct[0].Gy[126,64] = -0.316443717683753
        struct[0].Gy[126,65] = -0.171091579417532
        struct[0].Gy[126,66] = 1.10755301189314
        struct[0].Gy[126,67] = 0.598820527961360
        struct[0].Gy[126,68] = -0.316443717683753
        struct[0].Gy[126,69] = -0.171091579417531
        struct[0].Gy[126,126] = -1
        struct[0].Gy[127,64] = 0.171091579417532
        struct[0].Gy[127,65] = -0.316443717683753
        struct[0].Gy[127,66] = -0.598820527961360
        struct[0].Gy[127,67] = 1.10755301189314
        struct[0].Gy[127,68] = 0.171091579417531
        struct[0].Gy[127,69] = -0.316443717683753
        struct[0].Gy[127,127] = -1
        struct[0].Gy[128,64] = -0.316443717683753
        struct[0].Gy[128,65] = -0.171091579417532
        struct[0].Gy[128,66] = -0.316443717683753
        struct[0].Gy[128,67] = -0.171091579417531
        struct[0].Gy[128,68] = 1.10755301189314
        struct[0].Gy[128,69] = 0.598820527961360
        struct[0].Gy[128,128] = -1
        struct[0].Gy[129,64] = 0.171091579417532
        struct[0].Gy[129,65] = -0.316443717683753
        struct[0].Gy[129,66] = 0.171091579417531
        struct[0].Gy[129,67] = -0.316443717683753
        struct[0].Gy[129,68] = -0.598820527961360
        struct[0].Gy[129,69] = 1.10755301189314
        struct[0].Gy[129,129] = -1
        struct[0].Gy[130,0] = i_load_B2lv_a_r
        struct[0].Gy[130,1] = i_load_B2lv_a_i
        struct[0].Gy[130,6] = -i_load_B2lv_a_r
        struct[0].Gy[130,7] = -i_load_B2lv_a_i
        struct[0].Gy[130,130] = v_B2lv_a_r - v_B2lv_n_r
        struct[0].Gy[130,131] = v_B2lv_a_i - v_B2lv_n_i
        struct[0].Gy[131,2] = i_load_B2lv_b_r
        struct[0].Gy[131,3] = i_load_B2lv_b_i
        struct[0].Gy[131,6] = -i_load_B2lv_b_r
        struct[0].Gy[131,7] = -i_load_B2lv_b_i
        struct[0].Gy[131,132] = v_B2lv_b_r - v_B2lv_n_r
        struct[0].Gy[131,133] = v_B2lv_b_i - v_B2lv_n_i
        struct[0].Gy[132,4] = i_load_B2lv_c_r
        struct[0].Gy[132,5] = i_load_B2lv_c_i
        struct[0].Gy[132,6] = -i_load_B2lv_c_r
        struct[0].Gy[132,7] = -i_load_B2lv_c_i
        struct[0].Gy[132,134] = v_B2lv_c_r - v_B2lv_n_r
        struct[0].Gy[132,135] = v_B2lv_c_i - v_B2lv_n_i
        struct[0].Gy[133,0] = -i_load_B2lv_a_i
        struct[0].Gy[133,1] = i_load_B2lv_a_r
        struct[0].Gy[133,6] = i_load_B2lv_a_i
        struct[0].Gy[133,7] = -i_load_B2lv_a_r
        struct[0].Gy[133,130] = v_B2lv_a_i - v_B2lv_n_i
        struct[0].Gy[133,131] = -v_B2lv_a_r + v_B2lv_n_r
        struct[0].Gy[134,2] = -i_load_B2lv_b_i
        struct[0].Gy[134,3] = i_load_B2lv_b_r
        struct[0].Gy[134,6] = i_load_B2lv_b_i
        struct[0].Gy[134,7] = -i_load_B2lv_b_r
        struct[0].Gy[134,132] = v_B2lv_b_i - v_B2lv_n_i
        struct[0].Gy[134,133] = -v_B2lv_b_r + v_B2lv_n_r
        struct[0].Gy[135,4] = -i_load_B2lv_c_i
        struct[0].Gy[135,5] = i_load_B2lv_c_r
        struct[0].Gy[135,6] = i_load_B2lv_c_i
        struct[0].Gy[135,7] = -i_load_B2lv_c_r
        struct[0].Gy[135,134] = v_B2lv_c_i - v_B2lv_n_i
        struct[0].Gy[135,135] = -v_B2lv_c_r + v_B2lv_n_r
        struct[0].Gy[136,130] = 1
        struct[0].Gy[136,132] = 1
        struct[0].Gy[136,134] = 1
        struct[0].Gy[136,136] = 1
        struct[0].Gy[137,131] = 1
        struct[0].Gy[137,133] = 1
        struct[0].Gy[137,135] = 1
        struct[0].Gy[137,137] = 1
        struct[0].Gy[138,8] = i_load_B3lv_a_r
        struct[0].Gy[138,9] = i_load_B3lv_a_i
        struct[0].Gy[138,14] = -i_load_B3lv_a_r
        struct[0].Gy[138,15] = -i_load_B3lv_a_i
        struct[0].Gy[138,138] = v_B3lv_a_r - v_B3lv_n_r
        struct[0].Gy[138,139] = v_B3lv_a_i - v_B3lv_n_i
        struct[0].Gy[139,10] = i_load_B3lv_b_r
        struct[0].Gy[139,11] = i_load_B3lv_b_i
        struct[0].Gy[139,14] = -i_load_B3lv_b_r
        struct[0].Gy[139,15] = -i_load_B3lv_b_i
        struct[0].Gy[139,140] = v_B3lv_b_r - v_B3lv_n_r
        struct[0].Gy[139,141] = v_B3lv_b_i - v_B3lv_n_i
        struct[0].Gy[140,12] = i_load_B3lv_c_r
        struct[0].Gy[140,13] = i_load_B3lv_c_i
        struct[0].Gy[140,14] = -i_load_B3lv_c_r
        struct[0].Gy[140,15] = -i_load_B3lv_c_i
        struct[0].Gy[140,142] = v_B3lv_c_r - v_B3lv_n_r
        struct[0].Gy[140,143] = v_B3lv_c_i - v_B3lv_n_i
        struct[0].Gy[141,8] = -i_load_B3lv_a_i
        struct[0].Gy[141,9] = i_load_B3lv_a_r
        struct[0].Gy[141,14] = i_load_B3lv_a_i
        struct[0].Gy[141,15] = -i_load_B3lv_a_r
        struct[0].Gy[141,138] = v_B3lv_a_i - v_B3lv_n_i
        struct[0].Gy[141,139] = -v_B3lv_a_r + v_B3lv_n_r
        struct[0].Gy[142,10] = -i_load_B3lv_b_i
        struct[0].Gy[142,11] = i_load_B3lv_b_r
        struct[0].Gy[142,14] = i_load_B3lv_b_i
        struct[0].Gy[142,15] = -i_load_B3lv_b_r
        struct[0].Gy[142,140] = v_B3lv_b_i - v_B3lv_n_i
        struct[0].Gy[142,141] = -v_B3lv_b_r + v_B3lv_n_r
        struct[0].Gy[143,12] = -i_load_B3lv_c_i
        struct[0].Gy[143,13] = i_load_B3lv_c_r
        struct[0].Gy[143,14] = i_load_B3lv_c_i
        struct[0].Gy[143,15] = -i_load_B3lv_c_r
        struct[0].Gy[143,142] = v_B3lv_c_i - v_B3lv_n_i
        struct[0].Gy[143,143] = -v_B3lv_c_r + v_B3lv_n_r
        struct[0].Gy[144,138] = 1
        struct[0].Gy[144,140] = 1
        struct[0].Gy[144,142] = 1
        struct[0].Gy[144,144] = 1
        struct[0].Gy[145,139] = 1
        struct[0].Gy[145,141] = 1
        struct[0].Gy[145,143] = 1
        struct[0].Gy[145,145] = 1
        struct[0].Gy[146,16] = i_load_B4lv_a_r
        struct[0].Gy[146,17] = i_load_B4lv_a_i
        struct[0].Gy[146,22] = -i_load_B4lv_a_r
        struct[0].Gy[146,23] = -i_load_B4lv_a_i
        struct[0].Gy[146,146] = v_B4lv_a_r - v_B4lv_n_r
        struct[0].Gy[146,147] = v_B4lv_a_i - v_B4lv_n_i
        struct[0].Gy[147,18] = i_load_B4lv_b_r
        struct[0].Gy[147,19] = i_load_B4lv_b_i
        struct[0].Gy[147,22] = -i_load_B4lv_b_r
        struct[0].Gy[147,23] = -i_load_B4lv_b_i
        struct[0].Gy[147,148] = v_B4lv_b_r - v_B4lv_n_r
        struct[0].Gy[147,149] = v_B4lv_b_i - v_B4lv_n_i
        struct[0].Gy[148,20] = i_load_B4lv_c_r
        struct[0].Gy[148,21] = i_load_B4lv_c_i
        struct[0].Gy[148,22] = -i_load_B4lv_c_r
        struct[0].Gy[148,23] = -i_load_B4lv_c_i
        struct[0].Gy[148,150] = v_B4lv_c_r - v_B4lv_n_r
        struct[0].Gy[148,151] = v_B4lv_c_i - v_B4lv_n_i
        struct[0].Gy[149,16] = -i_load_B4lv_a_i
        struct[0].Gy[149,17] = i_load_B4lv_a_r
        struct[0].Gy[149,22] = i_load_B4lv_a_i
        struct[0].Gy[149,23] = -i_load_B4lv_a_r
        struct[0].Gy[149,146] = v_B4lv_a_i - v_B4lv_n_i
        struct[0].Gy[149,147] = -v_B4lv_a_r + v_B4lv_n_r
        struct[0].Gy[150,18] = -i_load_B4lv_b_i
        struct[0].Gy[150,19] = i_load_B4lv_b_r
        struct[0].Gy[150,22] = i_load_B4lv_b_i
        struct[0].Gy[150,23] = -i_load_B4lv_b_r
        struct[0].Gy[150,148] = v_B4lv_b_i - v_B4lv_n_i
        struct[0].Gy[150,149] = -v_B4lv_b_r + v_B4lv_n_r
        struct[0].Gy[151,20] = -i_load_B4lv_c_i
        struct[0].Gy[151,21] = i_load_B4lv_c_r
        struct[0].Gy[151,22] = i_load_B4lv_c_i
        struct[0].Gy[151,23] = -i_load_B4lv_c_r
        struct[0].Gy[151,150] = v_B4lv_c_i - v_B4lv_n_i
        struct[0].Gy[151,151] = -v_B4lv_c_r + v_B4lv_n_r
        struct[0].Gy[152,146] = 1
        struct[0].Gy[152,148] = 1
        struct[0].Gy[152,150] = 1
        struct[0].Gy[152,152] = 1
        struct[0].Gy[153,147] = 1
        struct[0].Gy[153,149] = 1
        struct[0].Gy[153,151] = 1
        struct[0].Gy[153,153] = 1
        struct[0].Gy[154,24] = i_load_B5lv_a_r
        struct[0].Gy[154,25] = i_load_B5lv_a_i
        struct[0].Gy[154,30] = -i_load_B5lv_a_r
        struct[0].Gy[154,31] = -i_load_B5lv_a_i
        struct[0].Gy[154,154] = v_B5lv_a_r - v_B5lv_n_r
        struct[0].Gy[154,155] = v_B5lv_a_i - v_B5lv_n_i
        struct[0].Gy[155,26] = i_load_B5lv_b_r
        struct[0].Gy[155,27] = i_load_B5lv_b_i
        struct[0].Gy[155,30] = -i_load_B5lv_b_r
        struct[0].Gy[155,31] = -i_load_B5lv_b_i
        struct[0].Gy[155,156] = v_B5lv_b_r - v_B5lv_n_r
        struct[0].Gy[155,157] = v_B5lv_b_i - v_B5lv_n_i
        struct[0].Gy[156,28] = i_load_B5lv_c_r
        struct[0].Gy[156,29] = i_load_B5lv_c_i
        struct[0].Gy[156,30] = -i_load_B5lv_c_r
        struct[0].Gy[156,31] = -i_load_B5lv_c_i
        struct[0].Gy[156,158] = v_B5lv_c_r - v_B5lv_n_r
        struct[0].Gy[156,159] = v_B5lv_c_i - v_B5lv_n_i
        struct[0].Gy[157,24] = -i_load_B5lv_a_i
        struct[0].Gy[157,25] = i_load_B5lv_a_r
        struct[0].Gy[157,30] = i_load_B5lv_a_i
        struct[0].Gy[157,31] = -i_load_B5lv_a_r
        struct[0].Gy[157,154] = v_B5lv_a_i - v_B5lv_n_i
        struct[0].Gy[157,155] = -v_B5lv_a_r + v_B5lv_n_r
        struct[0].Gy[158,26] = -i_load_B5lv_b_i
        struct[0].Gy[158,27] = i_load_B5lv_b_r
        struct[0].Gy[158,30] = i_load_B5lv_b_i
        struct[0].Gy[158,31] = -i_load_B5lv_b_r
        struct[0].Gy[158,156] = v_B5lv_b_i - v_B5lv_n_i
        struct[0].Gy[158,157] = -v_B5lv_b_r + v_B5lv_n_r
        struct[0].Gy[159,28] = -i_load_B5lv_c_i
        struct[0].Gy[159,29] = i_load_B5lv_c_r
        struct[0].Gy[159,30] = i_load_B5lv_c_i
        struct[0].Gy[159,31] = -i_load_B5lv_c_r
        struct[0].Gy[159,158] = v_B5lv_c_i - v_B5lv_n_i
        struct[0].Gy[159,159] = -v_B5lv_c_r + v_B5lv_n_r
        struct[0].Gy[160,154] = 1
        struct[0].Gy[160,156] = 1
        struct[0].Gy[160,158] = 1
        struct[0].Gy[160,160] = 1
        struct[0].Gy[161,155] = 1
        struct[0].Gy[161,157] = 1
        struct[0].Gy[161,159] = 1
        struct[0].Gy[161,161] = 1
        struct[0].Gy[162,32] = i_load_B6lv_a_r
        struct[0].Gy[162,33] = i_load_B6lv_a_i
        struct[0].Gy[162,38] = -i_load_B6lv_a_r
        struct[0].Gy[162,39] = -i_load_B6lv_a_i
        struct[0].Gy[162,162] = v_B6lv_a_r - v_B6lv_n_r
        struct[0].Gy[162,163] = v_B6lv_a_i - v_B6lv_n_i
        struct[0].Gy[163,34] = i_load_B6lv_b_r
        struct[0].Gy[163,35] = i_load_B6lv_b_i
        struct[0].Gy[163,38] = -i_load_B6lv_b_r
        struct[0].Gy[163,39] = -i_load_B6lv_b_i
        struct[0].Gy[163,164] = v_B6lv_b_r - v_B6lv_n_r
        struct[0].Gy[163,165] = v_B6lv_b_i - v_B6lv_n_i
        struct[0].Gy[164,36] = i_load_B6lv_c_r
        struct[0].Gy[164,37] = i_load_B6lv_c_i
        struct[0].Gy[164,38] = -i_load_B6lv_c_r
        struct[0].Gy[164,39] = -i_load_B6lv_c_i
        struct[0].Gy[164,166] = v_B6lv_c_r - v_B6lv_n_r
        struct[0].Gy[164,167] = v_B6lv_c_i - v_B6lv_n_i
        struct[0].Gy[165,32] = -i_load_B6lv_a_i
        struct[0].Gy[165,33] = i_load_B6lv_a_r
        struct[0].Gy[165,38] = i_load_B6lv_a_i
        struct[0].Gy[165,39] = -i_load_B6lv_a_r
        struct[0].Gy[165,162] = v_B6lv_a_i - v_B6lv_n_i
        struct[0].Gy[165,163] = -v_B6lv_a_r + v_B6lv_n_r
        struct[0].Gy[166,34] = -i_load_B6lv_b_i
        struct[0].Gy[166,35] = i_load_B6lv_b_r
        struct[0].Gy[166,38] = i_load_B6lv_b_i
        struct[0].Gy[166,39] = -i_load_B6lv_b_r
        struct[0].Gy[166,164] = v_B6lv_b_i - v_B6lv_n_i
        struct[0].Gy[166,165] = -v_B6lv_b_r + v_B6lv_n_r
        struct[0].Gy[167,36] = -i_load_B6lv_c_i
        struct[0].Gy[167,37] = i_load_B6lv_c_r
        struct[0].Gy[167,38] = i_load_B6lv_c_i
        struct[0].Gy[167,39] = -i_load_B6lv_c_r
        struct[0].Gy[167,166] = v_B6lv_c_i - v_B6lv_n_i
        struct[0].Gy[167,167] = -v_B6lv_c_r + v_B6lv_n_r
        struct[0].Gy[168,162] = 1
        struct[0].Gy[168,164] = 1
        struct[0].Gy[168,166] = 1
        struct[0].Gy[168,168] = 1
        struct[0].Gy[169,163] = 1
        struct[0].Gy[169,165] = 1
        struct[0].Gy[169,167] = 1
        struct[0].Gy[169,169] = 1

        struct[0].Gu[40,0] = 1.10755301189314
        struct[0].Gu[40,1] = 0.598820527961361
        struct[0].Gu[40,2] = -0.316443717683753
        struct[0].Gu[40,3] = -0.171091579417532
        struct[0].Gu[40,4] = -0.316443717683753
        struct[0].Gu[40,5] = -0.171091579417532
        struct[0].Gu[41,0] = -0.598820527961361
        struct[0].Gu[41,1] = 1.10755301189314
        struct[0].Gu[41,2] = 0.171091579417532
        struct[0].Gu[41,3] = -0.316443717683753
        struct[0].Gu[41,4] = 0.171091579417532
        struct[0].Gu[41,5] = -0.316443717683753
        struct[0].Gu[42,0] = -0.316443717683753
        struct[0].Gu[42,1] = -0.171091579417532
        struct[0].Gu[42,2] = 1.10755301189314
        struct[0].Gu[42,3] = 0.598820527961360
        struct[0].Gu[42,4] = -0.316443717683753
        struct[0].Gu[42,5] = -0.171091579417531
        struct[0].Gu[43,0] = 0.171091579417532
        struct[0].Gu[43,1] = -0.316443717683753
        struct[0].Gu[43,2] = -0.598820527961360
        struct[0].Gu[43,3] = 1.10755301189314
        struct[0].Gu[43,4] = 0.171091579417531
        struct[0].Gu[43,5] = -0.316443717683753
        struct[0].Gu[44,0] = -0.316443717683753
        struct[0].Gu[44,1] = -0.171091579417532
        struct[0].Gu[44,2] = -0.316443717683753
        struct[0].Gu[44,3] = -0.171091579417531
        struct[0].Gu[44,4] = 1.10755301189314
        struct[0].Gu[44,5] = 0.598820527961360
        struct[0].Gu[45,0] = 0.171091579417532
        struct[0].Gu[45,1] = -0.316443717683753
        struct[0].Gu[45,2] = 0.171091579417531
        struct[0].Gu[45,3] = -0.316443717683753
        struct[0].Gu[45,4] = -0.598820527961360
        struct[0].Gu[45,5] = 1.10755301189314
        struct[0].Gu[64,6] = 1.10755301189314
        struct[0].Gu[64,7] = 0.598820527961361
        struct[0].Gu[64,8] = -0.316443717683753
        struct[0].Gu[64,9] = -0.171091579417532
        struct[0].Gu[64,10] = -0.316443717683753
        struct[0].Gu[64,11] = -0.171091579417532
        struct[0].Gu[65,6] = -0.598820527961361
        struct[0].Gu[65,7] = 1.10755301189314
        struct[0].Gu[65,8] = 0.171091579417532
        struct[0].Gu[65,9] = -0.316443717683753
        struct[0].Gu[65,10] = 0.171091579417532
        struct[0].Gu[65,11] = -0.316443717683753
        struct[0].Gu[66,6] = -0.316443717683753
        struct[0].Gu[66,7] = -0.171091579417532
        struct[0].Gu[66,8] = 1.10755301189314
        struct[0].Gu[66,9] = 0.598820527961360
        struct[0].Gu[66,10] = -0.316443717683753
        struct[0].Gu[66,11] = -0.171091579417531
        struct[0].Gu[67,6] = 0.171091579417532
        struct[0].Gu[67,7] = -0.316443717683753
        struct[0].Gu[67,8] = -0.598820527961360
        struct[0].Gu[67,9] = 1.10755301189314
        struct[0].Gu[67,10] = 0.171091579417531
        struct[0].Gu[67,11] = -0.316443717683753
        struct[0].Gu[68,6] = -0.316443717683753
        struct[0].Gu[68,7] = -0.171091579417532
        struct[0].Gu[68,8] = -0.316443717683753
        struct[0].Gu[68,9] = -0.171091579417531
        struct[0].Gu[68,10] = 1.10755301189314
        struct[0].Gu[68,11] = 0.598820527961360
        struct[0].Gu[69,6] = 0.171091579417532
        struct[0].Gu[69,7] = -0.316443717683753
        struct[0].Gu[69,8] = 0.171091579417531
        struct[0].Gu[69,9] = -0.316443717683753
        struct[0].Gu[69,10] = -0.598820527961360
        struct[0].Gu[69,11] = 1.10755301189314
        struct[0].Gu[100,0] = 1.10755301189314
        struct[0].Gu[100,1] = 0.598820527961361
        struct[0].Gu[100,2] = -0.316443717683753
        struct[0].Gu[100,3] = -0.171091579417532
        struct[0].Gu[100,4] = -0.316443717683753
        struct[0].Gu[100,5] = -0.171091579417532
        struct[0].Gu[101,0] = -0.598820527961361
        struct[0].Gu[101,1] = 1.10755301189314
        struct[0].Gu[101,2] = 0.171091579417532
        struct[0].Gu[101,3] = -0.316443717683753
        struct[0].Gu[101,4] = 0.171091579417532
        struct[0].Gu[101,5] = -0.316443717683753
        struct[0].Gu[102,0] = -0.316443717683753
        struct[0].Gu[102,1] = -0.171091579417532
        struct[0].Gu[102,2] = 1.10755301189314
        struct[0].Gu[102,3] = 0.598820527961360
        struct[0].Gu[102,4] = -0.316443717683753
        struct[0].Gu[102,5] = -0.171091579417531
        struct[0].Gu[103,0] = 0.171091579417532
        struct[0].Gu[103,1] = -0.316443717683753
        struct[0].Gu[103,2] = -0.598820527961360
        struct[0].Gu[103,3] = 1.10755301189314
        struct[0].Gu[103,4] = 0.171091579417531
        struct[0].Gu[103,5] = -0.316443717683753
        struct[0].Gu[104,0] = -0.316443717683753
        struct[0].Gu[104,1] = -0.171091579417532
        struct[0].Gu[104,2] = -0.316443717683753
        struct[0].Gu[104,3] = -0.171091579417531
        struct[0].Gu[104,4] = 1.10755301189314
        struct[0].Gu[104,5] = 0.598820527961360
        struct[0].Gu[105,0] = 0.171091579417532
        struct[0].Gu[105,1] = -0.316443717683753
        struct[0].Gu[105,2] = 0.171091579417531
        struct[0].Gu[105,3] = -0.316443717683753
        struct[0].Gu[105,4] = -0.598820527961360
        struct[0].Gu[105,5] = 1.10755301189314
        struct[0].Gu[124,6] = -1.10755301189314
        struct[0].Gu[124,7] = -0.598820527961361
        struct[0].Gu[124,8] = 0.316443717683753
        struct[0].Gu[124,9] = 0.171091579417532
        struct[0].Gu[124,10] = 0.316443717683753
        struct[0].Gu[124,11] = 0.171091579417532
        struct[0].Gu[125,6] = 0.598820527961361
        struct[0].Gu[125,7] = -1.10755301189314
        struct[0].Gu[125,8] = -0.171091579417532
        struct[0].Gu[125,9] = 0.316443717683753
        struct[0].Gu[125,10] = -0.171091579417532
        struct[0].Gu[125,11] = 0.316443717683753
        struct[0].Gu[126,6] = 0.316443717683753
        struct[0].Gu[126,7] = 0.171091579417532
        struct[0].Gu[126,8] = -1.10755301189314
        struct[0].Gu[126,9] = -0.598820527961360
        struct[0].Gu[126,10] = 0.316443717683753
        struct[0].Gu[126,11] = 0.171091579417531
        struct[0].Gu[127,6] = -0.171091579417532
        struct[0].Gu[127,7] = 0.316443717683753
        struct[0].Gu[127,8] = 0.598820527961360
        struct[0].Gu[127,9] = -1.10755301189314
        struct[0].Gu[127,10] = -0.171091579417531
        struct[0].Gu[127,11] = 0.316443717683753
        struct[0].Gu[128,6] = 0.316443717683753
        struct[0].Gu[128,7] = 0.171091579417532
        struct[0].Gu[128,8] = 0.316443717683753
        struct[0].Gu[128,9] = 0.171091579417531
        struct[0].Gu[128,10] = -1.10755301189314
        struct[0].Gu[128,11] = -0.598820527961360
        struct[0].Gu[129,6] = -0.171091579417532
        struct[0].Gu[129,7] = 0.316443717683753
        struct[0].Gu[129,8] = -0.171091579417531
        struct[0].Gu[129,9] = 0.316443717683753
        struct[0].Gu[129,10] = 0.598820527961360
        struct[0].Gu[129,11] = -1.10755301189314
        struct[0].Gu[130,52] = -1
        struct[0].Gu[131,54] = -1
        struct[0].Gu[132,56] = -1
        struct[0].Gu[133,53] = -1
        struct[0].Gu[134,55] = -1
        struct[0].Gu[135,57] = -1
        struct[0].Gu[138,58] = -1
        struct[0].Gu[139,60] = -1
        struct[0].Gu[140,62] = -1
        struct[0].Gu[141,59] = -1
        struct[0].Gu[142,61] = -1
        struct[0].Gu[143,63] = -1
        struct[0].Gu[146,64] = -1
        struct[0].Gu[147,66] = -1
        struct[0].Gu[148,68] = -1
        struct[0].Gu[149,65] = -1
        struct[0].Gu[150,67] = -1
        struct[0].Gu[151,69] = -1
        struct[0].Gu[154,70] = -1
        struct[0].Gu[155,72] = -1
        struct[0].Gu[156,74] = -1
        struct[0].Gu[157,71] = -1
        struct[0].Gu[158,73] = -1
        struct[0].Gu[159,75] = -1
        struct[0].Gu[162,76] = -1
        struct[0].Gu[163,78] = -1
        struct[0].Gu[164,80] = -1
        struct[0].Gu[165,77] = -1
        struct[0].Gu[166,79] = -1
        struct[0].Gu[167,81] = -1





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
    Fx_ini_rows = [0]

    Fx_ini_cols = [0]

    Fy_ini_rows = []

    Fy_ini_cols = []

    Gx_ini_rows = []

    Gx_ini_cols = []

    Gy_ini_rows = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25, 25, 26, 26, 26, 26, 26, 26, 26, 26, 26, 27, 27, 27, 27, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 31, 31, 31, 32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33, 33, 33, 34, 34, 34, 34, 34, 34, 34, 34, 34, 35, 35, 35, 35, 35, 35, 35, 35, 35, 36, 36, 36, 36, 36, 36, 36, 36, 36, 37, 37, 37, 37, 37, 37, 37, 37, 37, 38, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 41, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 43, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 44, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 45, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 47, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 51, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 52, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 53, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 54, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 55, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 56, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 57, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 58, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 59, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 62, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 66, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 67, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 68, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 69, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 71, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 72, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 73, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 74, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 75, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 76, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 77, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 78, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 79, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 81, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 82, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 83, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 84, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 86, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 87, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 88, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 89, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 91, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 92, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 93, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 94, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 95, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 96, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 98, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 100, 100, 100, 100, 100, 100, 100, 101, 101, 101, 101, 101, 101, 101, 102, 102, 102, 102, 102, 102, 102, 103, 103, 103, 103, 103, 103, 103, 104, 104, 104, 104, 104, 104, 104, 105, 105, 105, 105, 105, 105, 105, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 107, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 108, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 109, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 110, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 111, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 112, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 113, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 114, 115, 115, 115, 115, 115, 115, 115, 115, 115, 115, 115, 115, 115, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 116, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 117, 118, 118, 118, 118, 118, 118, 118, 118, 118, 118, 118, 118, 118, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 119, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 121, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 122, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 123, 124, 124, 124, 124, 124, 124, 124, 125, 125, 125, 125, 125, 125, 125, 126, 126, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127, 128, 128, 128, 128, 128, 128, 128, 129, 129, 129, 129, 129, 129, 129, 130, 130, 130, 130, 130, 130, 131, 131, 131, 131, 131, 131, 132, 132, 132, 132, 132, 132, 133, 133, 133, 133, 133, 133, 134, 134, 134, 134, 134, 134, 135, 135, 135, 135, 135, 135, 136, 136, 136, 136, 137, 137, 137, 137, 138, 138, 138, 138, 138, 138, 139, 139, 139, 139, 139, 139, 140, 140, 140, 140, 140, 140, 141, 141, 141, 141, 141, 141, 142, 142, 142, 142, 142, 142, 143, 143, 143, 143, 143, 143, 144, 144, 144, 144, 145, 145, 145, 145, 146, 146, 146, 146, 146, 146, 147, 147, 147, 147, 147, 147, 148, 148, 148, 148, 148, 148, 149, 149, 149, 149, 149, 149, 150, 150, 150, 150, 150, 150, 151, 151, 151, 151, 151, 151, 152, 152, 152, 152, 153, 153, 153, 153, 154, 154, 154, 154, 154, 154, 155, 155, 155, 155, 155, 155, 156, 156, 156, 156, 156, 156, 157, 157, 157, 157, 157, 157, 158, 158, 158, 158, 158, 158, 159, 159, 159, 159, 159, 159, 160, 160, 160, 160, 161, 161, 161, 161, 162, 162, 162, 162, 162, 162, 163, 163, 163, 163, 163, 163, 164, 164, 164, 164, 164, 164, 165, 165, 165, 165, 165, 165, 166, 166, 166, 166, 166, 166, 167, 167, 167, 167, 167, 167, 168, 168, 168, 168, 169, 169, 169, 169]

    Gy_ini_cols = [0, 1, 6, 7, 40, 41, 44, 45, 130, 0, 1, 6, 7, 40, 41, 44, 45, 131, 2, 3, 6, 7, 40, 41, 42, 43, 132, 2, 3, 6, 7, 40, 41, 42, 43, 133, 4, 5, 6, 7, 42, 43, 44, 45, 134, 4, 5, 6, 7, 42, 43, 44, 45, 135, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 46, 47, 50, 51, 138, 8, 9, 14, 15, 46, 47, 50, 51, 139, 10, 11, 14, 15, 46, 47, 48, 49, 140, 10, 11, 14, 15, 46, 47, 48, 49, 141, 12, 13, 14, 15, 48, 49, 50, 51, 142, 12, 13, 14, 15, 48, 49, 50, 51, 143, 8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22, 23, 52, 53, 56, 57, 146, 16, 17, 22, 23, 52, 53, 56, 57, 147, 18, 19, 22, 23, 52, 53, 54, 55, 148, 18, 19, 22, 23, 52, 53, 54, 55, 149, 20, 21, 22, 23, 54, 55, 56, 57, 150, 20, 21, 22, 23, 54, 55, 56, 57, 151, 16, 17, 18, 19, 20, 21, 22, 23, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 31, 58, 59, 62, 63, 154, 24, 25, 30, 31, 58, 59, 62, 63, 155, 26, 27, 30, 31, 58, 59, 60, 61, 156, 26, 27, 30, 31, 58, 59, 60, 61, 157, 28, 29, 30, 31, 60, 61, 62, 63, 158, 28, 29, 30, 31, 60, 61, 62, 63, 159, 24, 25, 26, 27, 28, 29, 30, 31, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 38, 39, 64, 65, 68, 69, 162, 32, 33, 38, 39, 64, 65, 68, 69, 163, 34, 35, 38, 39, 64, 65, 66, 67, 164, 34, 35, 38, 39, 64, 65, 66, 67, 165, 36, 37, 38, 39, 66, 67, 68, 69, 166, 36, 37, 38, 39, 66, 67, 68, 69, 167, 32, 33, 34, 35, 36, 37, 38, 39, 32, 33, 34, 35, 36, 37, 38, 39, 0, 1, 2, 3, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 0, 1, 2, 3, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 2, 3, 4, 5, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 2, 3, 4, 5, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 0, 1, 4, 5, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 0, 1, 4, 5, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 8, 9, 10, 11, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 8, 9, 10, 11, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 10, 11, 12, 13, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 10, 11, 12, 13, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 8, 9, 12, 13, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 8, 9, 12, 13, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 16, 17, 18, 19, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 16, 17, 18, 19, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 18, 19, 20, 21, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 18, 19, 20, 21, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 16, 17, 20, 21, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 16, 17, 20, 21, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 24, 25, 26, 27, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 24, 25, 26, 27, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 26, 27, 28, 29, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 26, 27, 28, 29, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 24, 25, 28, 29, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 24, 25, 28, 29, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 32, 33, 34, 35, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 32, 33, 34, 35, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 34, 35, 36, 37, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 34, 35, 36, 37, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 32, 33, 36, 37, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 32, 33, 36, 37, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 0, 1, 2, 3, 40, 41, 42, 43, 44, 45, 70, 0, 1, 2, 3, 40, 41, 42, 43, 44, 45, 71, 2, 3, 4, 5, 40, 41, 42, 43, 44, 45, 72, 2, 3, 4, 5, 40, 41, 42, 43, 44, 45, 73, 0, 1, 4, 5, 40, 41, 42, 43, 44, 45, 74, 0, 1, 4, 5, 40, 41, 42, 43, 44, 45, 75, 8, 9, 10, 11, 46, 47, 48, 49, 50, 51, 76, 8, 9, 10, 11, 46, 47, 48, 49, 50, 51, 77, 10, 11, 12, 13, 46, 47, 48, 49, 50, 51, 78, 10, 11, 12, 13, 46, 47, 48, 49, 50, 51, 79, 8, 9, 12, 13, 46, 47, 48, 49, 50, 51, 80, 8, 9, 12, 13, 46, 47, 48, 49, 50, 51, 81, 16, 17, 18, 19, 52, 53, 54, 55, 56, 57, 82, 16, 17, 18, 19, 52, 53, 54, 55, 56, 57, 83, 18, 19, 20, 21, 52, 53, 54, 55, 56, 57, 84, 18, 19, 20, 21, 52, 53, 54, 55, 56, 57, 85, 16, 17, 20, 21, 52, 53, 54, 55, 56, 57, 86, 16, 17, 20, 21, 52, 53, 54, 55, 56, 57, 87, 24, 25, 26, 27, 58, 59, 60, 61, 62, 63, 88, 24, 25, 26, 27, 58, 59, 60, 61, 62, 63, 89, 26, 27, 28, 29, 58, 59, 60, 61, 62, 63, 90, 26, 27, 28, 29, 58, 59, 60, 61, 62, 63, 91, 24, 25, 28, 29, 58, 59, 60, 61, 62, 63, 92, 24, 25, 28, 29, 58, 59, 60, 61, 62, 63, 93, 32, 33, 34, 35, 64, 65, 66, 67, 68, 69, 94, 32, 33, 34, 35, 64, 65, 66, 67, 68, 69, 95, 34, 35, 36, 37, 64, 65, 66, 67, 68, 69, 96, 34, 35, 36, 37, 64, 65, 66, 67, 68, 69, 97, 32, 33, 36, 37, 64, 65, 66, 67, 68, 69, 98, 32, 33, 36, 37, 64, 65, 66, 67, 68, 69, 99, 40, 41, 42, 43, 44, 45, 100, 40, 41, 42, 43, 44, 45, 101, 40, 41, 42, 43, 44, 45, 102, 40, 41, 42, 43, 44, 45, 103, 40, 41, 42, 43, 44, 45, 104, 40, 41, 42, 43, 44, 45, 105, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 106, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 107, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 108, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 109, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 110, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 111, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 112, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 113, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 114, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 115, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 116, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 117, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 118, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 119, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 120, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 121, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 122, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 123, 64, 65, 66, 67, 68, 69, 124, 64, 65, 66, 67, 68, 69, 125, 64, 65, 66, 67, 68, 69, 126, 64, 65, 66, 67, 68, 69, 127, 64, 65, 66, 67, 68, 69, 128, 64, 65, 66, 67, 68, 69, 129, 0, 1, 6, 7, 130, 131, 2, 3, 6, 7, 132, 133, 4, 5, 6, 7, 134, 135, 0, 1, 6, 7, 130, 131, 2, 3, 6, 7, 132, 133, 4, 5, 6, 7, 134, 135, 130, 132, 134, 136, 131, 133, 135, 137, 8, 9, 14, 15, 138, 139, 10, 11, 14, 15, 140, 141, 12, 13, 14, 15, 142, 143, 8, 9, 14, 15, 138, 139, 10, 11, 14, 15, 140, 141, 12, 13, 14, 15, 142, 143, 138, 140, 142, 144, 139, 141, 143, 145, 16, 17, 22, 23, 146, 147, 18, 19, 22, 23, 148, 149, 20, 21, 22, 23, 150, 151, 16, 17, 22, 23, 146, 147, 18, 19, 22, 23, 148, 149, 20, 21, 22, 23, 150, 151, 146, 148, 150, 152, 147, 149, 151, 153, 24, 25, 30, 31, 154, 155, 26, 27, 30, 31, 156, 157, 28, 29, 30, 31, 158, 159, 24, 25, 30, 31, 154, 155, 26, 27, 30, 31, 156, 157, 28, 29, 30, 31, 158, 159, 154, 156, 158, 160, 155, 157, 159, 161, 32, 33, 38, 39, 162, 163, 34, 35, 38, 39, 164, 165, 36, 37, 38, 39, 166, 167, 32, 33, 38, 39, 162, 163, 34, 35, 38, 39, 164, 165, 36, 37, 38, 39, 166, 167, 162, 164, 166, 168, 163, 165, 167, 169]

    return Fx_ini_rows,Fx_ini_cols,Fy_ini_rows,Fy_ini_cols,Gx_ini_rows,Gx_ini_cols,Gy_ini_rows,Gy_ini_cols