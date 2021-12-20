import numpy as np
import numba
import scipy.optimize as sopt
import scipy.sparse as sspa
from scipy.sparse.linalg import spsolve,spilu,splu
from numba import cuda
import cffi
import numba.core.typing.cffi_utils as cffi_support

ffi = cffi.FFI()

import newengland_cffi as jacs

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


class newengland_class: 

    def __init__(self): 

        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 111
        self.N_y = 170 
        self.N_z = 49 
        self.N_store = 10000 
        self.params_list = ['S_base', 'g_01_02', 'b_01_02', 'bs_01_02', 'g_01_39', 'b_01_39', 'bs_01_39', 'g_02_03', 'b_02_03', 'bs_02_03', 'g_02_25', 'b_02_25', 'bs_02_25', 'g_03_04', 'b_03_04', 'bs_03_04', 'g_03_18', 'b_03_18', 'bs_03_18', 'g_04_05', 'b_04_05', 'bs_04_05', 'g_04_14', 'b_04_14', 'bs_04_14', 'g_05_06', 'b_05_06', 'bs_05_06', 'g_05_08', 'b_05_08', 'bs_05_08', 'g_06_07', 'b_06_07', 'bs_06_07', 'g_06_11', 'b_06_11', 'bs_06_11', 'g_07_08', 'b_07_08', 'bs_07_08', 'g_08_09', 'b_08_09', 'bs_08_09', 'g_09_39', 'b_09_39', 'bs_09_39', 'g_10_11', 'b_10_11', 'bs_10_11', 'g_10_13', 'b_10_13', 'bs_10_13', 'g_13_14', 'b_13_14', 'bs_13_14', 'g_14_15', 'b_14_15', 'bs_14_15', 'g_15_16', 'b_15_16', 'bs_15_16', 'g_16_17', 'b_16_17', 'bs_16_17', 'g_16_19', 'b_16_19', 'bs_16_19', 'g_16_21', 'b_16_21', 'bs_16_21', 'g_16_24', 'b_16_24', 'bs_16_24', 'g_17_18', 'b_17_18', 'bs_17_18', 'g_17_27', 'b_17_27', 'bs_17_27', 'g_21_22', 'b_21_22', 'bs_21_22', 'g_22_23', 'b_22_23', 'bs_22_23', 'g_23_24', 'b_23_24', 'bs_23_24', 'g_25_26', 'b_25_26', 'bs_25_26', 'g_26_27', 'b_26_27', 'bs_26_27', 'g_26_28', 'b_26_28', 'bs_26_28', 'g_26_29', 'b_26_29', 'bs_26_29', 'g_28_29', 'b_28_29', 'bs_28_29', 'g_12_11', 'b_12_11', 'bs_12_11', 'g_12_13', 'b_12_13', 'bs_12_13', 'g_06_31', 'b_06_31', 'bs_06_31', 'g_10_32', 'b_10_32', 'bs_10_32', 'g_19_33', 'b_19_33', 'bs_19_33', 'g_20_34', 'b_20_34', 'bs_20_34', 'g_22_35', 'b_22_35', 'bs_22_35', 'g_23_36', 'b_23_36', 'bs_23_36', 'g_25_37', 'b_25_37', 'bs_25_37', 'g_02_30', 'b_02_30', 'bs_02_30', 'g_29_38', 'b_29_38', 'bs_29_38', 'g_19_20', 'b_19_20', 'bs_19_20', 'U_01_n', 'U_02_n', 'U_03_n', 'U_04_n', 'U_05_n', 'U_06_n', 'U_07_n', 'U_08_n', 'U_09_n', 'U_10_n', 'U_11_n', 'U_12_n', 'U_13_n', 'U_14_n', 'U_15_n', 'U_16_n', 'U_17_n', 'U_18_n', 'U_19_n', 'U_20_n', 'U_21_n', 'U_22_n', 'U_23_n', 'U_24_n', 'U_25_n', 'U_26_n', 'U_27_n', 'U_28_n', 'U_29_n', 'U_30_n', 'U_31_n', 'U_32_n', 'U_33_n', 'U_34_n', 'U_35_n', 'U_36_n', 'U_37_n', 'U_38_n', 'U_39_n', 'S_n_30', 'Omega_b_30', 'H_30', 'T1d0_30', 'T1q0_30', 'X_d_30', 'X_q_30', 'X1d_30', 'X1q_30', 'D_30', 'R_a_30', 'K_delta_30', 'K_sec_30', 'K_a_30', 'K_ai_30', 'T_r_30', 'V_min_30', 'V_max_30', 'K_aw_30', 'Droop_30', 'T_gov_1_30', 'T_gov_2_30', 'T_gov_3_30', 'K_imw_30', 'omega_ref_30', 'T_wo_30', 'T_1_30', 'T_2_30', 'K_stab_30', 'V_lim_30', 'S_n_31', 'Omega_b_31', 'H_31', 'T1d0_31', 'T1q0_31', 'X_d_31', 'X_q_31', 'X1d_31', 'X1q_31', 'D_31', 'R_a_31', 'K_delta_31', 'K_sec_31', 'K_a_31', 'K_ai_31', 'T_r_31', 'V_min_31', 'V_max_31', 'K_aw_31', 'Droop_31', 'T_gov_1_31', 'T_gov_2_31', 'T_gov_3_31', 'K_imw_31', 'omega_ref_31', 'T_wo_31', 'T_1_31', 'T_2_31', 'K_stab_31', 'V_lim_31', 'S_n_32', 'Omega_b_32', 'H_32', 'T1d0_32', 'T1q0_32', 'X_d_32', 'X_q_32', 'X1d_32', 'X1q_32', 'D_32', 'R_a_32', 'K_delta_32', 'K_sec_32', 'K_a_32', 'K_ai_32', 'T_r_32', 'V_min_32', 'V_max_32', 'K_aw_32', 'Droop_32', 'T_gov_1_32', 'T_gov_2_32', 'T_gov_3_32', 'K_imw_32', 'omega_ref_32', 'T_wo_32', 'T_1_32', 'T_2_32', 'K_stab_32', 'V_lim_32', 'S_n_33', 'Omega_b_33', 'H_33', 'T1d0_33', 'T1q0_33', 'X_d_33', 'X_q_33', 'X1d_33', 'X1q_33', 'D_33', 'R_a_33', 'K_delta_33', 'K_sec_33', 'K_a_33', 'K_ai_33', 'T_r_33', 'V_min_33', 'V_max_33', 'K_aw_33', 'Droop_33', 'T_gov_1_33', 'T_gov_2_33', 'T_gov_3_33', 'K_imw_33', 'omega_ref_33', 'T_wo_33', 'T_1_33', 'T_2_33', 'K_stab_33', 'V_lim_33', 'S_n_34', 'Omega_b_34', 'H_34', 'T1d0_34', 'T1q0_34', 'X_d_34', 'X_q_34', 'X1d_34', 'X1q_34', 'D_34', 'R_a_34', 'K_delta_34', 'K_sec_34', 'K_a_34', 'K_ai_34', 'T_r_34', 'V_min_34', 'V_max_34', 'K_aw_34', 'Droop_34', 'T_gov_1_34', 'T_gov_2_34', 'T_gov_3_34', 'K_imw_34', 'omega_ref_34', 'T_wo_34', 'T_1_34', 'T_2_34', 'K_stab_34', 'V_lim_34', 'S_n_35', 'Omega_b_35', 'H_35', 'T1d0_35', 'T1q0_35', 'X_d_35', 'X_q_35', 'X1d_35', 'X1q_35', 'D_35', 'R_a_35', 'K_delta_35', 'K_sec_35', 'K_a_35', 'K_ai_35', 'T_r_35', 'V_min_35', 'V_max_35', 'K_aw_35', 'Droop_35', 'T_gov_1_35', 'T_gov_2_35', 'T_gov_3_35', 'K_imw_35', 'omega_ref_35', 'T_wo_35', 'T_1_35', 'T_2_35', 'K_stab_35', 'V_lim_35', 'S_n_36', 'Omega_b_36', 'H_36', 'T1d0_36', 'T1q0_36', 'X_d_36', 'X_q_36', 'X1d_36', 'X1q_36', 'D_36', 'R_a_36', 'K_delta_36', 'K_sec_36', 'K_a_36', 'K_ai_36', 'T_r_36', 'V_min_36', 'V_max_36', 'K_aw_36', 'Droop_36', 'T_gov_1_36', 'T_gov_2_36', 'T_gov_3_36', 'K_imw_36', 'omega_ref_36', 'T_wo_36', 'T_1_36', 'T_2_36', 'K_stab_36', 'V_lim_36', 'S_n_37', 'Omega_b_37', 'H_37', 'T1d0_37', 'T1q0_37', 'X_d_37', 'X_q_37', 'X1d_37', 'X1q_37', 'D_37', 'R_a_37', 'K_delta_37', 'K_sec_37', 'K_a_37', 'K_ai_37', 'T_r_37', 'V_min_37', 'V_max_37', 'K_aw_37', 'Droop_37', 'T_gov_1_37', 'T_gov_2_37', 'T_gov_3_37', 'K_imw_37', 'omega_ref_37', 'T_wo_37', 'T_1_37', 'T_2_37', 'K_stab_37', 'V_lim_37', 'S_n_38', 'Omega_b_38', 'H_38', 'T1d0_38', 'T1q0_38', 'X_d_38', 'X_q_38', 'X1d_38', 'X1q_38', 'D_38', 'R_a_38', 'K_delta_38', 'K_sec_38', 'K_a_38', 'K_ai_38', 'T_r_38', 'V_min_38', 'V_max_38', 'K_aw_38', 'Droop_38', 'T_gov_1_38', 'T_gov_2_38', 'T_gov_3_38', 'K_imw_38', 'omega_ref_38', 'T_wo_38', 'T_1_38', 'T_2_38', 'K_stab_38', 'V_lim_38', 'S_n_39', 'Omega_b_39', 'H_39', 'T1d0_39', 'T1q0_39', 'X_d_39', 'X_q_39', 'X1d_39', 'X1q_39', 'D_39', 'R_a_39', 'K_delta_39', 'K_sec_39', 'K_a_39', 'K_ai_39', 'T_r_39', 'V_min_39', 'V_max_39', 'K_aw_39', 'Droop_39', 'T_gov_1_39', 'T_gov_2_39', 'T_gov_3_39', 'K_imw_39', 'omega_ref_39', 'T_wo_39', 'T_1_39', 'T_2_39', 'K_stab_39', 'V_lim_39', 'K_p_agc', 'K_i_agc'] 
        self.params_values_list  = [100000000.0, 2.0570568805614005, -24.155725083163873, 0.0, 1.5974440894568687, -39.93610223642172, 0.0, 5.659555942533739, -65.73791902481499, 0.0, 56.92908262849707, -69.94144437215354, 0.0, 2.8547586630945583, -46.77412271070315, 0.0, 6.17630544637844, -74.6771476698484, 0.0, 4.863813229571985, -77.82101167315176, 0.0, 4.788985333732416, -77.2223885064352, 0.0, 29.41176470588236, -382.3529411764706, 0.0, 6.34517766497462, -88.83248730964468, 0.0, 7.058823529411764, -108.23529411764704, 0.0, 10.335154289089028, -121.06895024361432, 0.0, 18.761726078799253, -215.75984990619136, 0.0, 1.7384994482153926, -27.438056508790762, 0.0, 1.5974440894568687, -39.93610223642172, 0.0, 21.447721179624665, -230.56300268096516, 0.0, 21.447721179624665, -230.56300268096516, 0.0, 8.753160863645206, -98.22991635868509, 0.0, 3.7964271402358, -45.76803830173159, 0.0, 10.093080632499719, -105.41661993944152, 0.0, 8.782936010037641, -111.66875784190715, 0.0, 4.179619132206578, -50.93910817376767, 0.0, 4.3742140084203625, -73.81486139209362, 0.0, 8.595988538681947, -169.05444126074497, 0.0, 10.335154289089028, -121.06895024361432, 0.0, 4.319223868695595, -57.478902252641376, 0.0, 4.068348250610252, -71.19609438567942, 0.0, 6.485084306095979, -103.76134889753567, 0.0, 1.7888505821895528, -28.458986534833794, 0.0, 3.037407572636755, -30.658832686302244, 0.0, 6.420545746388443, -67.41573033707866, 0.0, 1.8982452267961594, -20.92484273259022, 0.0, 1.4471633060318785, -15.868018706489893, 0.0, 6.087750576162108, -65.66073835717702, 0.0, 0.8444118407650373, -22.95744692079945, 0.0, 0.8444118407650373, -22.95744692079945, 0.0, 0.0, -39.99999999999999, 0.0, 0.0, -50.0, 0.0, 3.463117795478157, -70.25181813684263, 0.0, 2.770850651149903, -55.41701302299806, 0.0, 0.0, -69.93006993006992, 0.0, 0.6755935088975665, -36.75228688402762, 0.0, 1.1139992573338284, -43.07463795024137, 0.0, 0.0, -55.248618784530386, 0.0, 3.2786885245901645, -63.934426229508205, 0.0, 3.666265123343634, -72.27779814591736, 0.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 400000.0, 900000000.0, 314.1592653589793, 6.5, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.001, 0.0, 300, 1e-06, 0.02, -10000.0, 5.0, 10, 0.05, 1.0, 2.0, 10.0, 0.01, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 900000000.0, 314.1592653589793, 6.5, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.0, 0.0, 300, 1e-06, 0.02, -10000.0, 5.0, 10, 0.05, 1.0, 2.0, 10.0, 0.01, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 900000000.0, 314.1592653589793, 6.175, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.0, 0.01, 300, 1e-06, 0.02, -10000.0, 5.0, 10, 0.05, 1.0, 2.0, 10.0, 0.0, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 900000000.0, 314.1592653589793, 6.175, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.0, 0.0, 300, 1e-06, 0.02, -10000.0, 5.0, 10, 0.05, 1.0, 2.0, 10.0, 0.01, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 900000000.0, 314.1592653589793, 6.175, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.0, 0.0, 300, 1e-06, 0.02, -10000.0, 5.0, 10, 0.05, 1.0, 2.0, 10.0, 0.01, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 900000000.0, 314.1592653589793, 6.175, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.0, 0.0, 300, 1e-06, 0.02, -10000.0, 5.0, 10, 0.05, 1.0, 2.0, 10.0, 0.01, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 900000000.0, 314.1592653589793, 6.175, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.0, 0.0, 300, 1e-06, 0.02, -10000.0, 5.0, 10, 0.05, 1.0, 2.0, 10.0, 0.01, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 900000000.0, 314.1592653589793, 6.175, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.0, 0.0, 300, 1e-06, 0.02, -10000.0, 5.0, 10, 0.05, 1.0, 2.0, 10.0, 0.01, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 900000000.0, 314.1592653589793, 6.175, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.0, 0.0, 300, 1e-06, 0.02, -10000.0, 5.0, 10, 0.05, 1.0, 2.0, 10.0, 0.01, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 900000000.0, 314.1592653589793, 6.175, 8.0, 0.4, 1.8, 1.7, 0.3, 0.55, 1.0, 0.0025, 0.0, 0.0, 300, 1e-06, 0.02, -10000.0, 5.0, 10, 0.05, 1.0, 2.0, 10.0, 0.01, 1.0, 10.0, 0.1, 0.1, 1.0, 0.1, 0.01, 0.01] 
        self.inputs_ini_list = ['P_01', 'Q_01', 'P_02', 'Q_02', 'P_03', 'Q_03', 'P_04', 'Q_04', 'P_05', 'Q_05', 'P_06', 'Q_06', 'P_07', 'Q_07', 'P_08', 'Q_08', 'P_09', 'Q_09', 'P_10', 'Q_10', 'P_11', 'Q_11', 'P_12', 'Q_12', 'P_13', 'Q_13', 'P_14', 'Q_14', 'P_15', 'Q_15', 'P_16', 'Q_16', 'P_17', 'Q_17', 'P_18', 'Q_18', 'P_19', 'Q_19', 'P_20', 'Q_20', 'P_21', 'Q_21', 'P_22', 'Q_22', 'P_23', 'Q_23', 'P_24', 'Q_24', 'P_25', 'Q_25', 'P_26', 'Q_26', 'P_27', 'Q_27', 'P_28', 'Q_28', 'P_29', 'Q_29', 'P_30', 'Q_30', 'P_31', 'Q_31', 'P_32', 'Q_32', 'P_33', 'Q_33', 'P_34', 'Q_34', 'P_35', 'Q_35', 'P_36', 'Q_36', 'P_37', 'Q_37', 'P_38', 'Q_38', 'P_39', 'Q_39', 'v_ref_30', 'v_pss_30', 'p_c_30', 'p_r_30', 'v_ref_31', 'v_pss_31', 'p_c_31', 'p_r_31', 'v_ref_32', 'v_pss_32', 'p_c_32', 'p_r_32', 'v_ref_33', 'v_pss_33', 'p_c_33', 'p_r_33', 'v_ref_34', 'v_pss_34', 'p_c_34', 'p_r_34', 'v_ref_35', 'v_pss_35', 'p_c_35', 'p_r_35', 'v_ref_36', 'v_pss_36', 'p_c_36', 'p_r_36', 'v_ref_37', 'v_pss_37', 'p_c_37', 'p_r_37', 'v_ref_38', 'v_pss_38', 'p_c_38', 'p_r_38', 'v_ref_39', 'v_pss_39', 'p_c_39', 'p_r_39'] 
        self.inputs_ini_values_list  = [0.0, -0.0, 0.0, -0.0, -322000000.0, -2400000.0, -500000000.0, -184000000.0, 0.0, -0.0, 0.0, -0.0, -233800000.0, -84000000.0, -522000000.0, -176000000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -7500000.0, -88000000.0, 0.0, 0.0, 0.0, 0.0, -320000000.0, -153000000.0, -329000000.0, -32300000.0, 0.0, 0.0, -158000000.0, -30000000.0, 0.0, 0.0, -628000000.0, -103000000.0, -274000000.0, -115000000.0, 0.0, 0.0, -247500000.0, -84600000.0, -308600000.0, 92000000.0, -224000000.0, -47200000.0, -139000000.0, -17000000.0, -281000000.0, -75500000.0, -206000000.0, -27600000.0, -283500000.0, -26900000.0, -47500.0, 0.0, -982000.0, -9200000.0, -983100.0, 0.0, -997200.0, 0.0, -12300.0, 0.0, -49300.0, 0.0, -63500.0, 0.0, -27800.0, 0.0, -26500.0, 0.0, -30000.0, -1104000000.0, 1.03, 0.0, 0.778, 0.0, 1.01, 0.0, 0.778, 0.0, 1.03, 0.0, 0.778, 0.0, 1.01, 0.0, 0.778, 0.0, 1.01, 0.0, 0.778, 0.0, 1.01, 0.0, 0.778, 0.0, 1.01, 0.0, 0.778, 0.0, 1.01, 0.0, 0.778, 0.0, 1.01, 0.0, 0.778, 0.0, 1.01, 0.0, 0.778, 0.0] 
        self.inputs_run_list = ['P_01', 'Q_01', 'P_02', 'Q_02', 'P_03', 'Q_03', 'P_04', 'Q_04', 'P_05', 'Q_05', 'P_06', 'Q_06', 'P_07', 'Q_07', 'P_08', 'Q_08', 'P_09', 'Q_09', 'P_10', 'Q_10', 'P_11', 'Q_11', 'P_12', 'Q_12', 'P_13', 'Q_13', 'P_14', 'Q_14', 'P_15', 'Q_15', 'P_16', 'Q_16', 'P_17', 'Q_17', 'P_18', 'Q_18', 'P_19', 'Q_19', 'P_20', 'Q_20', 'P_21', 'Q_21', 'P_22', 'Q_22', 'P_23', 'Q_23', 'P_24', 'Q_24', 'P_25', 'Q_25', 'P_26', 'Q_26', 'P_27', 'Q_27', 'P_28', 'Q_28', 'P_29', 'Q_29', 'P_30', 'Q_30', 'P_31', 'Q_31', 'P_32', 'Q_32', 'P_33', 'Q_33', 'P_34', 'Q_34', 'P_35', 'Q_35', 'P_36', 'Q_36', 'P_37', 'Q_37', 'P_38', 'Q_38', 'P_39', 'Q_39', 'v_ref_30', 'v_pss_30', 'p_c_30', 'p_r_30', 'v_ref_31', 'v_pss_31', 'p_c_31', 'p_r_31', 'v_ref_32', 'v_pss_32', 'p_c_32', 'p_r_32', 'v_ref_33', 'v_pss_33', 'p_c_33', 'p_r_33', 'v_ref_34', 'v_pss_34', 'p_c_34', 'p_r_34', 'v_ref_35', 'v_pss_35', 'p_c_35', 'p_r_35', 'v_ref_36', 'v_pss_36', 'p_c_36', 'p_r_36', 'v_ref_37', 'v_pss_37', 'p_c_37', 'p_r_37', 'v_ref_38', 'v_pss_38', 'p_c_38', 'p_r_38', 'v_ref_39', 'v_pss_39', 'p_c_39', 'p_r_39'] 
        self.inputs_run_values_list = [0.0, -0.0, 0.0, -0.0, -322000000.0, -2400000.0, -500000000.0, -184000000.0, 0.0, -0.0, 0.0, -0.0, -233800000.0, -84000000.0, -522000000.0, -176000000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -7500000.0, -88000000.0, 0.0, 0.0, 0.0, 0.0, -320000000.0, -153000000.0, -329000000.0, -32300000.0, 0.0, 0.0, -158000000.0, -30000000.0, 0.0, 0.0, -628000000.0, -103000000.0, -274000000.0, -115000000.0, 0.0, 0.0, -247500000.0, -84600000.0, -308600000.0, 92000000.0, -224000000.0, -47200000.0, -139000000.0, -17000000.0, -281000000.0, -75500000.0, -206000000.0, -27600000.0, -283500000.0, -26900000.0, -47500.0, 0.0, -982000.0, -9200000.0, -983100.0, 0.0, -997200.0, 0.0, -12300.0, 0.0, -49300.0, 0.0, -63500.0, 0.0, -27800.0, 0.0, -26500.0, 0.0, -30000.0, -1104000000.0, 1.03, 0.0, 0.778, 0.0, 1.01, 0.0, 0.778, 0.0, 1.03, 0.0, 0.778, 0.0, 1.01, 0.0, 0.778, 0.0, 1.01, 0.0, 0.778, 0.0, 1.01, 0.0, 0.778, 0.0, 1.01, 0.0, 0.778, 0.0, 1.01, 0.0, 0.778, 0.0, 1.01, 0.0, 0.778, 0.0, 1.01, 0.0, 0.778, 0.0] 
        self.outputs_list = ['V_01', 'V_02', 'V_03', 'V_04', 'V_05', 'V_06', 'V_07', 'V_08', 'V_09', 'V_10', 'V_11', 'V_12', 'V_13', 'V_14', 'V_15', 'V_16', 'V_17', 'V_18', 'V_19', 'V_20', 'V_21', 'V_22', 'V_23', 'V_24', 'V_25', 'V_26', 'V_27', 'V_28', 'V_29', 'V_30', 'V_31', 'V_32', 'V_33', 'V_34', 'V_35', 'V_36', 'V_37', 'V_38', 'V_39', 'p_e_30', 'p_e_31', 'p_e_32', 'p_e_33', 'p_e_34', 'p_e_35', 'p_e_36', 'p_e_37', 'p_e_38', 'p_e_39'] 
        self.x_list = ['delta_30', 'omega_30', 'e1q_30', 'e1d_30', 'v_c_30', 'xi_v_30', 'x_gov_1_30', 'x_gov_2_30', 'xi_imw_30', 'x_wo_30', 'x_lead_30', 'delta_31', 'omega_31', 'e1q_31', 'e1d_31', 'v_c_31', 'xi_v_31', 'x_gov_1_31', 'x_gov_2_31', 'xi_imw_31', 'x_wo_31', 'x_lead_31', 'delta_32', 'omega_32', 'e1q_32', 'e1d_32', 'v_c_32', 'xi_v_32', 'x_gov_1_32', 'x_gov_2_32', 'xi_imw_32', 'x_wo_32', 'x_lead_32', 'delta_33', 'omega_33', 'e1q_33', 'e1d_33', 'v_c_33', 'xi_v_33', 'x_gov_1_33', 'x_gov_2_33', 'xi_imw_33', 'x_wo_33', 'x_lead_33', 'delta_34', 'omega_34', 'e1q_34', 'e1d_34', 'v_c_34', 'xi_v_34', 'x_gov_1_34', 'x_gov_2_34', 'xi_imw_34', 'x_wo_34', 'x_lead_34', 'delta_35', 'omega_35', 'e1q_35', 'e1d_35', 'v_c_35', 'xi_v_35', 'x_gov_1_35', 'x_gov_2_35', 'xi_imw_35', 'x_wo_35', 'x_lead_35', 'delta_36', 'omega_36', 'e1q_36', 'e1d_36', 'v_c_36', 'xi_v_36', 'x_gov_1_36', 'x_gov_2_36', 'xi_imw_36', 'x_wo_36', 'x_lead_36', 'delta_37', 'omega_37', 'e1q_37', 'e1d_37', 'v_c_37', 'xi_v_37', 'x_gov_1_37', 'x_gov_2_37', 'xi_imw_37', 'x_wo_37', 'x_lead_37', 'delta_38', 'omega_38', 'e1q_38', 'e1d_38', 'v_c_38', 'xi_v_38', 'x_gov_1_38', 'x_gov_2_38', 'xi_imw_38', 'x_wo_38', 'x_lead_38', 'delta_39', 'omega_39', 'e1q_39', 'e1d_39', 'v_c_39', 'xi_v_39', 'x_gov_1_39', 'x_gov_2_39', 'xi_imw_39', 'x_wo_39', 'x_lead_39', 'xi_freq'] 
        self.y_run_list = ['V_01', 'theta_01', 'V_02', 'theta_02', 'V_03', 'theta_03', 'V_04', 'theta_04', 'V_05', 'theta_05', 'V_06', 'theta_06', 'V_07', 'theta_07', 'V_08', 'theta_08', 'V_09', 'theta_09', 'V_10', 'theta_10', 'V_11', 'theta_11', 'V_12', 'theta_12', 'V_13', 'theta_13', 'V_14', 'theta_14', 'V_15', 'theta_15', 'V_16', 'theta_16', 'V_17', 'theta_17', 'V_18', 'theta_18', 'V_19', 'theta_19', 'V_20', 'theta_20', 'V_21', 'theta_21', 'V_22', 'theta_22', 'V_23', 'theta_23', 'V_24', 'theta_24', 'V_25', 'theta_25', 'V_26', 'theta_26', 'V_27', 'theta_27', 'V_28', 'theta_28', 'V_29', 'theta_29', 'V_30', 'theta_30', 'V_31', 'theta_31', 'V_32', 'theta_32', 'V_33', 'theta_33', 'V_34', 'theta_34', 'V_35', 'theta_35', 'V_36', 'theta_36', 'V_37', 'theta_37', 'V_38', 'theta_38', 'V_39', 'theta_39', 'i_d_30', 'i_q_30', 'p_g_30', 'q_g_30', 'v_f_30', 'p_m_ref_30', 'p_m_30', 'z_wo_30', 'v_pss_30', 'i_d_31', 'i_q_31', 'p_g_31', 'q_g_31', 'v_f_31', 'p_m_ref_31', 'p_m_31', 'z_wo_31', 'v_pss_31', 'i_d_32', 'i_q_32', 'p_g_32', 'q_g_32', 'v_f_32', 'p_m_ref_32', 'p_m_32', 'z_wo_32', 'v_pss_32', 'i_d_33', 'i_q_33', 'p_g_33', 'q_g_33', 'v_f_33', 'p_m_ref_33', 'p_m_33', 'z_wo_33', 'v_pss_33', 'i_d_34', 'i_q_34', 'p_g_34', 'q_g_34', 'v_f_34', 'p_m_ref_34', 'p_m_34', 'z_wo_34', 'v_pss_34', 'i_d_35', 'i_q_35', 'p_g_35', 'q_g_35', 'v_f_35', 'p_m_ref_35', 'p_m_35', 'z_wo_35', 'v_pss_35', 'i_d_36', 'i_q_36', 'p_g_36', 'q_g_36', 'v_f_36', 'p_m_ref_36', 'p_m_36', 'z_wo_36', 'v_pss_36', 'i_d_37', 'i_q_37', 'p_g_37', 'q_g_37', 'v_f_37', 'p_m_ref_37', 'p_m_37', 'z_wo_37', 'v_pss_37', 'i_d_38', 'i_q_38', 'p_g_38', 'q_g_38', 'v_f_38', 'p_m_ref_38', 'p_m_38', 'z_wo_38', 'v_pss_38', 'i_d_39', 'i_q_39', 'p_g_39', 'q_g_39', 'v_f_39', 'p_m_ref_39', 'p_m_39', 'z_wo_39', 'v_pss_39', 'omega_coi', 'p_agc'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['V_01', 'theta_01', 'V_02', 'theta_02', 'V_03', 'theta_03', 'V_04', 'theta_04', 'V_05', 'theta_05', 'V_06', 'theta_06', 'V_07', 'theta_07', 'V_08', 'theta_08', 'V_09', 'theta_09', 'V_10', 'theta_10', 'V_11', 'theta_11', 'V_12', 'theta_12', 'V_13', 'theta_13', 'V_14', 'theta_14', 'V_15', 'theta_15', 'V_16', 'theta_16', 'V_17', 'theta_17', 'V_18', 'theta_18', 'V_19', 'theta_19', 'V_20', 'theta_20', 'V_21', 'theta_21', 'V_22', 'theta_22', 'V_23', 'theta_23', 'V_24', 'theta_24', 'V_25', 'theta_25', 'V_26', 'theta_26', 'V_27', 'theta_27', 'V_28', 'theta_28', 'V_29', 'theta_29', 'V_30', 'theta_30', 'V_31', 'theta_31', 'V_32', 'theta_32', 'V_33', 'theta_33', 'V_34', 'theta_34', 'V_35', 'theta_35', 'V_36', 'theta_36', 'V_37', 'theta_37', 'V_38', 'theta_38', 'V_39', 'theta_39', 'i_d_30', 'i_q_30', 'p_g_30', 'q_g_30', 'v_f_30', 'p_m_ref_30', 'p_m_30', 'z_wo_30', 'v_pss_30', 'i_d_31', 'i_q_31', 'p_g_31', 'q_g_31', 'v_f_31', 'p_m_ref_31', 'p_m_31', 'z_wo_31', 'v_pss_31', 'i_d_32', 'i_q_32', 'p_g_32', 'q_g_32', 'v_f_32', 'p_m_ref_32', 'p_m_32', 'z_wo_32', 'v_pss_32', 'i_d_33', 'i_q_33', 'p_g_33', 'q_g_33', 'v_f_33', 'p_m_ref_33', 'p_m_33', 'z_wo_33', 'v_pss_33', 'i_d_34', 'i_q_34', 'p_g_34', 'q_g_34', 'v_f_34', 'p_m_ref_34', 'p_m_34', 'z_wo_34', 'v_pss_34', 'i_d_35', 'i_q_35', 'p_g_35', 'q_g_35', 'v_f_35', 'p_m_ref_35', 'p_m_35', 'z_wo_35', 'v_pss_35', 'i_d_36', 'i_q_36', 'p_g_36', 'q_g_36', 'v_f_36', 'p_m_ref_36', 'p_m_36', 'z_wo_36', 'v_pss_36', 'i_d_37', 'i_q_37', 'p_g_37', 'q_g_37', 'v_f_37', 'p_m_ref_37', 'p_m_37', 'z_wo_37', 'v_pss_37', 'i_d_38', 'i_q_38', 'p_g_38', 'q_g_38', 'v_f_38', 'p_m_ref_38', 'p_m_38', 'z_wo_38', 'v_pss_38', 'i_d_39', 'i_q_39', 'p_g_39', 'q_g_39', 'v_f_39', 'p_m_ref_39', 'p_m_39', 'z_wo_39', 'v_pss_39', 'omega_coi', 'p_agc'] 
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
   

        

        
        self.max_it,self.itol,self.store = 50,1e-8,1 
        self.lmax_it,self.ltol,self.ldamp=50,1e-8,1.1
        self.mode = 0 

        self.lmax_it_ini,self.ltol_ini,self.ldamp_ini=50,1e-8,1.1

        self.fill_factor_ini,self.drop_tol_ini,self.drop_rule_ini = 10,0.001,'column'       
        self.fill_factor_run,self.drop_tol_run,self.drop_rule_run = 10,0.001,'column' 
        
 
        



        
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
                          fill_factor=self.fill_factor_run,
                          drop_tol=self.drop_tol_run,
                          drop_rule = self.drop_rule_run)
    
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

    sp_jac_ini_ia = [0, 1, 279, 0, 1, 169, 170, 189, 190, 195, 279, 2, 189, 193, 3, 190, 4, 169, 4, 5, 193, 197, 6, 194, 6, 7, 8, 191, 1, 9, 10, 196, 11, 12, 279, 11, 12, 171, 172, 198, 199, 204, 279, 13, 198, 202, 14, 199, 15, 171, 15, 16, 202, 206, 17, 203, 17, 18, 19, 200, 12, 20, 21, 205, 22, 23, 279, 22, 23, 173, 174, 207, 208, 213, 279, 24, 207, 211, 25, 208, 26, 173, 26, 27, 211, 215, 28, 212, 28, 29, 30, 209, 23, 31, 32, 214, 33, 34, 279, 33, 34, 175, 176, 216, 217, 222, 279, 35, 216, 220, 36, 217, 37, 175, 37, 38, 220, 224, 39, 221, 39, 40, 41, 218, 34, 42, 43, 223, 44, 45, 279, 44, 45, 177, 178, 225, 226, 231, 279, 46, 225, 229, 47, 226, 48, 177, 48, 49, 229, 233, 50, 230, 50, 51, 52, 227, 45, 53, 54, 232, 55, 56, 279, 55, 56, 179, 180, 234, 235, 240, 279, 57, 234, 238, 58, 235, 59, 179, 59, 60, 238, 242, 61, 239, 61, 62, 63, 236, 56, 64, 65, 241, 66, 67, 279, 66, 67, 181, 182, 243, 244, 249, 279, 68, 243, 247, 69, 244, 70, 181, 70, 71, 247, 251, 72, 248, 72, 73, 74, 245, 67, 75, 76, 250, 77, 78, 279, 77, 78, 183, 184, 252, 253, 258, 279, 79, 252, 256, 80, 253, 81, 183, 81, 82, 256, 260, 83, 257, 83, 84, 85, 254, 78, 86, 87, 259, 88, 89, 279, 88, 89, 185, 186, 261, 262, 267, 279, 90, 261, 265, 91, 262, 92, 185, 92, 93, 265, 269, 94, 266, 94, 95, 96, 263, 89, 97, 98, 268, 99, 100, 279, 99, 100, 187, 188, 270, 271, 276, 279, 101, 270, 274, 102, 271, 103, 187, 103, 104, 274, 278, 105, 275, 105, 106, 107, 272, 100, 108, 109, 277, 279, 111, 112, 113, 114, 187, 188, 111, 112, 113, 114, 187, 188, 111, 112, 113, 114, 115, 116, 159, 160, 169, 170, 111, 112, 113, 114, 115, 116, 159, 160, 169, 170, 113, 114, 115, 116, 117, 118, 145, 146, 113, 114, 115, 116, 117, 118, 145, 146, 115, 116, 117, 118, 119, 120, 137, 138, 115, 116, 117, 118, 119, 120, 137, 138, 117, 118, 119, 120, 121, 122, 125, 126, 117, 118, 119, 120, 121, 122, 125, 126, 119, 120, 121, 122, 123, 124, 131, 132, 171, 172, 119, 120, 121, 122, 123, 124, 131, 132, 171, 172, 121, 122, 123, 124, 125, 126, 121, 122, 123, 124, 125, 126, 119, 120, 123, 124, 125, 126, 127, 128, 119, 120, 123, 124, 125, 126, 127, 128, 125, 126, 127, 128, 187, 188, 125, 126, 127, 128, 187, 188, 129, 130, 131, 132, 135, 136, 173, 174, 129, 130, 131, 132, 135, 136, 173, 174, 121, 122, 129, 130, 131, 132, 133, 134, 121, 122, 129, 130, 131, 132, 133, 134, 131, 132, 133, 134, 135, 136, 131, 132, 133, 134, 135, 136, 129, 130, 133, 134, 135, 136, 137, 138, 129, 130, 133, 134, 135, 136, 137, 138, 117, 118, 135, 136, 137, 138, 139, 140, 117, 118, 135, 136, 137, 138, 139, 140, 137, 138, 139, 140, 141, 142, 137, 138, 139, 140, 141, 142, 139, 140, 141, 142, 143, 144, 147, 148, 151, 152, 157, 158, 139, 140, 141, 142, 143, 144, 147, 148, 151, 152, 157, 158, 141, 142, 143, 144, 145, 146, 163, 164, 141, 142, 143, 144, 145, 146, 163, 164, 115, 116, 143, 144, 145, 146, 115, 116, 143, 144, 145, 146, 141, 142, 147, 148, 149, 150, 175, 176, 141, 142, 147, 148, 149, 150, 175, 176, 147, 148, 149, 150, 177, 178, 147, 148, 149, 150, 177, 178, 141, 142, 151, 152, 153, 154, 141, 142, 151, 152, 153, 154, 151, 152, 153, 154, 155, 156, 179, 180, 151, 152, 153, 154, 155, 156, 179, 180, 153, 154, 155, 156, 157, 158, 181, 182, 153, 154, 155, 156, 157, 158, 181, 182, 141, 142, 155, 156, 157, 158, 141, 142, 155, 156, 157, 158, 113, 114, 159, 160, 161, 162, 183, 184, 113, 114, 159, 160, 161, 162, 183, 184, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 143, 144, 161, 162, 163, 164, 143, 144, 161, 162, 163, 164, 161, 162, 165, 166, 167, 168, 161, 162, 165, 166, 167, 168, 161, 162, 165, 166, 167, 168, 185, 186, 161, 162, 165, 166, 167, 168, 185, 186, 113, 114, 169, 170, 191, 113, 114, 169, 170, 192, 121, 122, 171, 172, 200, 121, 122, 171, 172, 201, 129, 130, 173, 174, 209, 129, 130, 173, 174, 210, 147, 148, 175, 176, 218, 147, 148, 175, 176, 219, 149, 150, 177, 178, 227, 149, 150, 177, 178, 228, 153, 154, 179, 180, 236, 153, 154, 179, 180, 237, 155, 156, 181, 182, 245, 155, 156, 181, 182, 246, 159, 160, 183, 184, 254, 159, 160, 183, 184, 255, 167, 168, 185, 186, 263, 167, 168, 185, 186, 264, 111, 112, 127, 128, 187, 188, 272, 111, 112, 127, 128, 187, 188, 273, 0, 2, 169, 170, 189, 190, 0, 3, 169, 170, 189, 190, 0, 169, 170, 189, 190, 191, 0, 169, 170, 189, 190, 192, 4, 5, 193, 197, 1, 8, 194, 280, 6, 7, 195, 1, 9, 196, 10, 196, 197, 11, 13, 171, 172, 198, 199, 11, 14, 171, 172, 198, 199, 11, 171, 172, 198, 199, 200, 11, 171, 172, 198, 199, 201, 15, 16, 202, 206, 12, 19, 203, 280, 17, 18, 204, 12, 20, 205, 21, 205, 206, 22, 24, 173, 174, 207, 208, 22, 25, 173, 174, 207, 208, 22, 173, 174, 207, 208, 209, 22, 173, 174, 207, 208, 210, 26, 27, 211, 215, 23, 30, 212, 280, 28, 29, 213, 23, 31, 214, 32, 214, 215, 33, 35, 175, 176, 216, 217, 33, 36, 175, 176, 216, 217, 33, 175, 176, 216, 217, 218, 33, 175, 176, 216, 217, 219, 37, 38, 220, 224, 34, 41, 221, 280, 39, 40, 222, 34, 42, 223, 43, 223, 224, 44, 46, 177, 178, 225, 226, 44, 47, 177, 178, 225, 226, 44, 177, 178, 225, 226, 227, 44, 177, 178, 225, 226, 228, 48, 49, 229, 233, 45, 52, 230, 280, 50, 51, 231, 45, 53, 232, 54, 232, 233, 55, 57, 179, 180, 234, 235, 55, 58, 179, 180, 234, 235, 55, 179, 180, 234, 235, 236, 55, 179, 180, 234, 235, 237, 59, 60, 238, 242, 56, 63, 239, 280, 61, 62, 240, 56, 64, 241, 65, 241, 242, 66, 68, 181, 182, 243, 244, 66, 69, 181, 182, 243, 244, 66, 181, 182, 243, 244, 245, 66, 181, 182, 243, 244, 246, 70, 71, 247, 251, 67, 74, 248, 280, 72, 73, 249, 67, 75, 250, 76, 250, 251, 77, 79, 183, 184, 252, 253, 77, 80, 183, 184, 252, 253, 77, 183, 184, 252, 253, 254, 77, 183, 184, 252, 253, 255, 81, 82, 256, 260, 78, 85, 257, 280, 83, 84, 258, 78, 86, 259, 87, 259, 260, 88, 90, 185, 186, 261, 262, 88, 91, 185, 186, 261, 262, 88, 185, 186, 261, 262, 263, 88, 185, 186, 261, 262, 264, 92, 93, 265, 269, 89, 96, 266, 280, 94, 95, 267, 89, 97, 268, 98, 268, 269, 99, 101, 187, 188, 270, 271, 99, 102, 187, 188, 270, 271, 99, 187, 188, 270, 271, 272, 99, 187, 188, 270, 271, 273, 103, 104, 274, 278, 100, 107, 275, 280, 105, 106, 276, 100, 108, 277, 109, 277, 278, 1, 12, 23, 34, 45, 56, 67, 78, 89, 100, 279, 110, 279, 280]
    sp_jac_ini_ja = [0, 3, 11, 14, 16, 18, 22, 24, 26, 28, 30, 32, 35, 43, 46, 48, 50, 54, 56, 58, 60, 62, 64, 67, 75, 78, 80, 82, 86, 88, 90, 92, 94, 96, 99, 107, 110, 112, 114, 118, 120, 122, 124, 126, 128, 131, 139, 142, 144, 146, 150, 152, 154, 156, 158, 160, 163, 171, 174, 176, 178, 182, 184, 186, 188, 190, 192, 195, 203, 206, 208, 210, 214, 216, 218, 220, 222, 224, 227, 235, 238, 240, 242, 246, 248, 250, 252, 254, 256, 259, 267, 270, 272, 274, 278, 280, 282, 284, 286, 288, 291, 299, 302, 304, 306, 310, 312, 314, 316, 318, 320, 321, 327, 333, 343, 353, 361, 369, 377, 385, 393, 401, 411, 421, 427, 433, 441, 449, 455, 461, 469, 477, 485, 493, 499, 505, 513, 521, 529, 537, 543, 549, 561, 573, 581, 589, 595, 601, 609, 617, 623, 629, 635, 641, 649, 657, 665, 673, 679, 685, 693, 701, 711, 721, 727, 733, 739, 745, 753, 761, 766, 771, 776, 781, 786, 791, 796, 801, 806, 811, 816, 821, 826, 831, 836, 841, 846, 851, 858, 865, 871, 877, 883, 889, 893, 897, 900, 903, 906, 912, 918, 924, 930, 934, 938, 941, 944, 947, 953, 959, 965, 971, 975, 979, 982, 985, 988, 994, 1000, 1006, 1012, 1016, 1020, 1023, 1026, 1029, 1035, 1041, 1047, 1053, 1057, 1061, 1064, 1067, 1070, 1076, 1082, 1088, 1094, 1098, 1102, 1105, 1108, 1111, 1117, 1123, 1129, 1135, 1139, 1143, 1146, 1149, 1152, 1158, 1164, 1170, 1176, 1180, 1184, 1187, 1190, 1193, 1199, 1205, 1211, 1217, 1221, 1225, 1228, 1231, 1234, 1240, 1246, 1252, 1258, 1262, 1266, 1269, 1272, 1275, 1286, 1289]
    sp_jac_ini_nia = 281
    sp_jac_ini_nja = 281
    return sp_jac_ini_ia, sp_jac_ini_ja, sp_jac_ini_nia, sp_jac_ini_nja 

def sp_jac_run_vectors():

    sp_jac_run_ia = [0, 1, 279, 0, 1, 169, 170, 189, 190, 195, 279, 2, 189, 193, 3, 190, 4, 169, 4, 5, 193, 197, 6, 194, 6, 7, 8, 191, 1, 9, 10, 196, 11, 12, 279, 11, 12, 171, 172, 198, 199, 204, 279, 13, 198, 202, 14, 199, 15, 171, 15, 16, 202, 206, 17, 203, 17, 18, 19, 200, 12, 20, 21, 205, 22, 23, 279, 22, 23, 173, 174, 207, 208, 213, 279, 24, 207, 211, 25, 208, 26, 173, 26, 27, 211, 215, 28, 212, 28, 29, 30, 209, 23, 31, 32, 214, 33, 34, 279, 33, 34, 175, 176, 216, 217, 222, 279, 35, 216, 220, 36, 217, 37, 175, 37, 38, 220, 224, 39, 221, 39, 40, 41, 218, 34, 42, 43, 223, 44, 45, 279, 44, 45, 177, 178, 225, 226, 231, 279, 46, 225, 229, 47, 226, 48, 177, 48, 49, 229, 233, 50, 230, 50, 51, 52, 227, 45, 53, 54, 232, 55, 56, 279, 55, 56, 179, 180, 234, 235, 240, 279, 57, 234, 238, 58, 235, 59, 179, 59, 60, 238, 242, 61, 239, 61, 62, 63, 236, 56, 64, 65, 241, 66, 67, 279, 66, 67, 181, 182, 243, 244, 249, 279, 68, 243, 247, 69, 244, 70, 181, 70, 71, 247, 251, 72, 248, 72, 73, 74, 245, 67, 75, 76, 250, 77, 78, 279, 77, 78, 183, 184, 252, 253, 258, 279, 79, 252, 256, 80, 253, 81, 183, 81, 82, 256, 260, 83, 257, 83, 84, 85, 254, 78, 86, 87, 259, 88, 89, 279, 88, 89, 185, 186, 261, 262, 267, 279, 90, 261, 265, 91, 262, 92, 185, 92, 93, 265, 269, 94, 266, 94, 95, 96, 263, 89, 97, 98, 268, 99, 100, 279, 99, 100, 187, 188, 270, 271, 276, 279, 101, 270, 274, 102, 271, 103, 187, 103, 104, 274, 278, 105, 275, 105, 106, 107, 272, 100, 108, 109, 277, 279, 111, 112, 113, 114, 187, 188, 111, 112, 113, 114, 187, 188, 111, 112, 113, 114, 115, 116, 159, 160, 169, 170, 111, 112, 113, 114, 115, 116, 159, 160, 169, 170, 113, 114, 115, 116, 117, 118, 145, 146, 113, 114, 115, 116, 117, 118, 145, 146, 115, 116, 117, 118, 119, 120, 137, 138, 115, 116, 117, 118, 119, 120, 137, 138, 117, 118, 119, 120, 121, 122, 125, 126, 117, 118, 119, 120, 121, 122, 125, 126, 119, 120, 121, 122, 123, 124, 131, 132, 171, 172, 119, 120, 121, 122, 123, 124, 131, 132, 171, 172, 121, 122, 123, 124, 125, 126, 121, 122, 123, 124, 125, 126, 119, 120, 123, 124, 125, 126, 127, 128, 119, 120, 123, 124, 125, 126, 127, 128, 125, 126, 127, 128, 187, 188, 125, 126, 127, 128, 187, 188, 129, 130, 131, 132, 135, 136, 173, 174, 129, 130, 131, 132, 135, 136, 173, 174, 121, 122, 129, 130, 131, 132, 133, 134, 121, 122, 129, 130, 131, 132, 133, 134, 131, 132, 133, 134, 135, 136, 131, 132, 133, 134, 135, 136, 129, 130, 133, 134, 135, 136, 137, 138, 129, 130, 133, 134, 135, 136, 137, 138, 117, 118, 135, 136, 137, 138, 139, 140, 117, 118, 135, 136, 137, 138, 139, 140, 137, 138, 139, 140, 141, 142, 137, 138, 139, 140, 141, 142, 139, 140, 141, 142, 143, 144, 147, 148, 151, 152, 157, 158, 139, 140, 141, 142, 143, 144, 147, 148, 151, 152, 157, 158, 141, 142, 143, 144, 145, 146, 163, 164, 141, 142, 143, 144, 145, 146, 163, 164, 115, 116, 143, 144, 145, 146, 115, 116, 143, 144, 145, 146, 141, 142, 147, 148, 149, 150, 175, 176, 141, 142, 147, 148, 149, 150, 175, 176, 147, 148, 149, 150, 177, 178, 147, 148, 149, 150, 177, 178, 141, 142, 151, 152, 153, 154, 141, 142, 151, 152, 153, 154, 151, 152, 153, 154, 155, 156, 179, 180, 151, 152, 153, 154, 155, 156, 179, 180, 153, 154, 155, 156, 157, 158, 181, 182, 153, 154, 155, 156, 157, 158, 181, 182, 141, 142, 155, 156, 157, 158, 141, 142, 155, 156, 157, 158, 113, 114, 159, 160, 161, 162, 183, 184, 113, 114, 159, 160, 161, 162, 183, 184, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 143, 144, 161, 162, 163, 164, 143, 144, 161, 162, 163, 164, 161, 162, 165, 166, 167, 168, 161, 162, 165, 166, 167, 168, 161, 162, 165, 166, 167, 168, 185, 186, 161, 162, 165, 166, 167, 168, 185, 186, 113, 114, 169, 170, 191, 113, 114, 169, 170, 192, 121, 122, 171, 172, 200, 121, 122, 171, 172, 201, 129, 130, 173, 174, 209, 129, 130, 173, 174, 210, 147, 148, 175, 176, 218, 147, 148, 175, 176, 219, 149, 150, 177, 178, 227, 149, 150, 177, 178, 228, 153, 154, 179, 180, 236, 153, 154, 179, 180, 237, 155, 156, 181, 182, 245, 155, 156, 181, 182, 246, 159, 160, 183, 184, 254, 159, 160, 183, 184, 255, 167, 168, 185, 186, 263, 167, 168, 185, 186, 264, 111, 112, 127, 128, 187, 188, 272, 111, 112, 127, 128, 187, 188, 273, 0, 2, 169, 170, 189, 190, 0, 3, 169, 170, 189, 190, 0, 169, 170, 189, 190, 191, 0, 169, 170, 189, 190, 192, 4, 5, 193, 197, 1, 8, 194, 280, 6, 7, 195, 1, 9, 196, 10, 196, 197, 11, 13, 171, 172, 198, 199, 11, 14, 171, 172, 198, 199, 11, 171, 172, 198, 199, 200, 11, 171, 172, 198, 199, 201, 15, 16, 202, 206, 12, 19, 203, 280, 17, 18, 204, 12, 20, 205, 21, 205, 206, 22, 24, 173, 174, 207, 208, 22, 25, 173, 174, 207, 208, 22, 173, 174, 207, 208, 209, 22, 173, 174, 207, 208, 210, 26, 27, 211, 215, 23, 30, 212, 280, 28, 29, 213, 23, 31, 214, 32, 214, 215, 33, 35, 175, 176, 216, 217, 33, 36, 175, 176, 216, 217, 33, 175, 176, 216, 217, 218, 33, 175, 176, 216, 217, 219, 37, 38, 220, 224, 34, 41, 221, 280, 39, 40, 222, 34, 42, 223, 43, 223, 224, 44, 46, 177, 178, 225, 226, 44, 47, 177, 178, 225, 226, 44, 177, 178, 225, 226, 227, 44, 177, 178, 225, 226, 228, 48, 49, 229, 233, 45, 52, 230, 280, 50, 51, 231, 45, 53, 232, 54, 232, 233, 55, 57, 179, 180, 234, 235, 55, 58, 179, 180, 234, 235, 55, 179, 180, 234, 235, 236, 55, 179, 180, 234, 235, 237, 59, 60, 238, 242, 56, 63, 239, 280, 61, 62, 240, 56, 64, 241, 65, 241, 242, 66, 68, 181, 182, 243, 244, 66, 69, 181, 182, 243, 244, 66, 181, 182, 243, 244, 245, 66, 181, 182, 243, 244, 246, 70, 71, 247, 251, 67, 74, 248, 280, 72, 73, 249, 67, 75, 250, 76, 250, 251, 77, 79, 183, 184, 252, 253, 77, 80, 183, 184, 252, 253, 77, 183, 184, 252, 253, 254, 77, 183, 184, 252, 253, 255, 81, 82, 256, 260, 78, 85, 257, 280, 83, 84, 258, 78, 86, 259, 87, 259, 260, 88, 90, 185, 186, 261, 262, 88, 91, 185, 186, 261, 262, 88, 185, 186, 261, 262, 263, 88, 185, 186, 261, 262, 264, 92, 93, 265, 269, 89, 96, 266, 280, 94, 95, 267, 89, 97, 268, 98, 268, 269, 99, 101, 187, 188, 270, 271, 99, 102, 187, 188, 270, 271, 99, 187, 188, 270, 271, 272, 99, 187, 188, 270, 271, 273, 103, 104, 274, 278, 100, 107, 275, 280, 105, 106, 276, 100, 108, 277, 109, 277, 278, 1, 12, 23, 34, 45, 56, 67, 78, 89, 100, 279, 110, 279, 280]
    sp_jac_run_ja = [0, 3, 11, 14, 16, 18, 22, 24, 26, 28, 30, 32, 35, 43, 46, 48, 50, 54, 56, 58, 60, 62, 64, 67, 75, 78, 80, 82, 86, 88, 90, 92, 94, 96, 99, 107, 110, 112, 114, 118, 120, 122, 124, 126, 128, 131, 139, 142, 144, 146, 150, 152, 154, 156, 158, 160, 163, 171, 174, 176, 178, 182, 184, 186, 188, 190, 192, 195, 203, 206, 208, 210, 214, 216, 218, 220, 222, 224, 227, 235, 238, 240, 242, 246, 248, 250, 252, 254, 256, 259, 267, 270, 272, 274, 278, 280, 282, 284, 286, 288, 291, 299, 302, 304, 306, 310, 312, 314, 316, 318, 320, 321, 327, 333, 343, 353, 361, 369, 377, 385, 393, 401, 411, 421, 427, 433, 441, 449, 455, 461, 469, 477, 485, 493, 499, 505, 513, 521, 529, 537, 543, 549, 561, 573, 581, 589, 595, 601, 609, 617, 623, 629, 635, 641, 649, 657, 665, 673, 679, 685, 693, 701, 711, 721, 727, 733, 739, 745, 753, 761, 766, 771, 776, 781, 786, 791, 796, 801, 806, 811, 816, 821, 826, 831, 836, 841, 846, 851, 858, 865, 871, 877, 883, 889, 893, 897, 900, 903, 906, 912, 918, 924, 930, 934, 938, 941, 944, 947, 953, 959, 965, 971, 975, 979, 982, 985, 988, 994, 1000, 1006, 1012, 1016, 1020, 1023, 1026, 1029, 1035, 1041, 1047, 1053, 1057, 1061, 1064, 1067, 1070, 1076, 1082, 1088, 1094, 1098, 1102, 1105, 1108, 1111, 1117, 1123, 1129, 1135, 1139, 1143, 1146, 1149, 1152, 1158, 1164, 1170, 1176, 1180, 1184, 1187, 1190, 1193, 1199, 1205, 1211, 1217, 1221, 1225, 1228, 1231, 1234, 1240, 1246, 1252, 1258, 1262, 1266, 1269, 1272, 1275, 1286, 1289]
    sp_jac_run_nia = 281
    sp_jac_run_nja = 281
    return sp_jac_run_ia, sp_jac_run_ja, sp_jac_run_nia, sp_jac_run_nja 

def sp_jac_trap_vectors():

    sp_jac_trap_ia = [0, 1, 279, 0, 1, 169, 170, 189, 190, 195, 279, 2, 189, 193, 3, 190, 4, 169, 4, 5, 193, 197, 6, 194, 6, 7, 8, 191, 1, 9, 10, 196, 11, 12, 279, 11, 12, 171, 172, 198, 199, 204, 279, 13, 198, 202, 14, 199, 15, 171, 15, 16, 202, 206, 17, 203, 17, 18, 19, 200, 12, 20, 21, 205, 22, 23, 279, 22, 23, 173, 174, 207, 208, 213, 279, 24, 207, 211, 25, 208, 26, 173, 26, 27, 211, 215, 28, 212, 28, 29, 30, 209, 23, 31, 32, 214, 33, 34, 279, 33, 34, 175, 176, 216, 217, 222, 279, 35, 216, 220, 36, 217, 37, 175, 37, 38, 220, 224, 39, 221, 39, 40, 41, 218, 34, 42, 43, 223, 44, 45, 279, 44, 45, 177, 178, 225, 226, 231, 279, 46, 225, 229, 47, 226, 48, 177, 48, 49, 229, 233, 50, 230, 50, 51, 52, 227, 45, 53, 54, 232, 55, 56, 279, 55, 56, 179, 180, 234, 235, 240, 279, 57, 234, 238, 58, 235, 59, 179, 59, 60, 238, 242, 61, 239, 61, 62, 63, 236, 56, 64, 65, 241, 66, 67, 279, 66, 67, 181, 182, 243, 244, 249, 279, 68, 243, 247, 69, 244, 70, 181, 70, 71, 247, 251, 72, 248, 72, 73, 74, 245, 67, 75, 76, 250, 77, 78, 279, 77, 78, 183, 184, 252, 253, 258, 279, 79, 252, 256, 80, 253, 81, 183, 81, 82, 256, 260, 83, 257, 83, 84, 85, 254, 78, 86, 87, 259, 88, 89, 279, 88, 89, 185, 186, 261, 262, 267, 279, 90, 261, 265, 91, 262, 92, 185, 92, 93, 265, 269, 94, 266, 94, 95, 96, 263, 89, 97, 98, 268, 99, 100, 279, 99, 100, 187, 188, 270, 271, 276, 279, 101, 270, 274, 102, 271, 103, 187, 103, 104, 274, 278, 105, 275, 105, 106, 107, 272, 100, 108, 109, 277, 110, 279, 111, 112, 113, 114, 187, 188, 111, 112, 113, 114, 187, 188, 111, 112, 113, 114, 115, 116, 159, 160, 169, 170, 111, 112, 113, 114, 115, 116, 159, 160, 169, 170, 113, 114, 115, 116, 117, 118, 145, 146, 113, 114, 115, 116, 117, 118, 145, 146, 115, 116, 117, 118, 119, 120, 137, 138, 115, 116, 117, 118, 119, 120, 137, 138, 117, 118, 119, 120, 121, 122, 125, 126, 117, 118, 119, 120, 121, 122, 125, 126, 119, 120, 121, 122, 123, 124, 131, 132, 171, 172, 119, 120, 121, 122, 123, 124, 131, 132, 171, 172, 121, 122, 123, 124, 125, 126, 121, 122, 123, 124, 125, 126, 119, 120, 123, 124, 125, 126, 127, 128, 119, 120, 123, 124, 125, 126, 127, 128, 125, 126, 127, 128, 187, 188, 125, 126, 127, 128, 187, 188, 129, 130, 131, 132, 135, 136, 173, 174, 129, 130, 131, 132, 135, 136, 173, 174, 121, 122, 129, 130, 131, 132, 133, 134, 121, 122, 129, 130, 131, 132, 133, 134, 131, 132, 133, 134, 135, 136, 131, 132, 133, 134, 135, 136, 129, 130, 133, 134, 135, 136, 137, 138, 129, 130, 133, 134, 135, 136, 137, 138, 117, 118, 135, 136, 137, 138, 139, 140, 117, 118, 135, 136, 137, 138, 139, 140, 137, 138, 139, 140, 141, 142, 137, 138, 139, 140, 141, 142, 139, 140, 141, 142, 143, 144, 147, 148, 151, 152, 157, 158, 139, 140, 141, 142, 143, 144, 147, 148, 151, 152, 157, 158, 141, 142, 143, 144, 145, 146, 163, 164, 141, 142, 143, 144, 145, 146, 163, 164, 115, 116, 143, 144, 145, 146, 115, 116, 143, 144, 145, 146, 141, 142, 147, 148, 149, 150, 175, 176, 141, 142, 147, 148, 149, 150, 175, 176, 147, 148, 149, 150, 177, 178, 147, 148, 149, 150, 177, 178, 141, 142, 151, 152, 153, 154, 141, 142, 151, 152, 153, 154, 151, 152, 153, 154, 155, 156, 179, 180, 151, 152, 153, 154, 155, 156, 179, 180, 153, 154, 155, 156, 157, 158, 181, 182, 153, 154, 155, 156, 157, 158, 181, 182, 141, 142, 155, 156, 157, 158, 141, 142, 155, 156, 157, 158, 113, 114, 159, 160, 161, 162, 183, 184, 113, 114, 159, 160, 161, 162, 183, 184, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 143, 144, 161, 162, 163, 164, 143, 144, 161, 162, 163, 164, 161, 162, 165, 166, 167, 168, 161, 162, 165, 166, 167, 168, 161, 162, 165, 166, 167, 168, 185, 186, 161, 162, 165, 166, 167, 168, 185, 186, 113, 114, 169, 170, 191, 113, 114, 169, 170, 192, 121, 122, 171, 172, 200, 121, 122, 171, 172, 201, 129, 130, 173, 174, 209, 129, 130, 173, 174, 210, 147, 148, 175, 176, 218, 147, 148, 175, 176, 219, 149, 150, 177, 178, 227, 149, 150, 177, 178, 228, 153, 154, 179, 180, 236, 153, 154, 179, 180, 237, 155, 156, 181, 182, 245, 155, 156, 181, 182, 246, 159, 160, 183, 184, 254, 159, 160, 183, 184, 255, 167, 168, 185, 186, 263, 167, 168, 185, 186, 264, 111, 112, 127, 128, 187, 188, 272, 111, 112, 127, 128, 187, 188, 273, 0, 2, 169, 170, 189, 190, 0, 3, 169, 170, 189, 190, 0, 169, 170, 189, 190, 191, 0, 169, 170, 189, 190, 192, 4, 5, 193, 197, 1, 8, 194, 280, 6, 7, 195, 1, 9, 196, 10, 196, 197, 11, 13, 171, 172, 198, 199, 11, 14, 171, 172, 198, 199, 11, 171, 172, 198, 199, 200, 11, 171, 172, 198, 199, 201, 15, 16, 202, 206, 12, 19, 203, 280, 17, 18, 204, 12, 20, 205, 21, 205, 206, 22, 24, 173, 174, 207, 208, 22, 25, 173, 174, 207, 208, 22, 173, 174, 207, 208, 209, 22, 173, 174, 207, 208, 210, 26, 27, 211, 215, 23, 30, 212, 280, 28, 29, 213, 23, 31, 214, 32, 214, 215, 33, 35, 175, 176, 216, 217, 33, 36, 175, 176, 216, 217, 33, 175, 176, 216, 217, 218, 33, 175, 176, 216, 217, 219, 37, 38, 220, 224, 34, 41, 221, 280, 39, 40, 222, 34, 42, 223, 43, 223, 224, 44, 46, 177, 178, 225, 226, 44, 47, 177, 178, 225, 226, 44, 177, 178, 225, 226, 227, 44, 177, 178, 225, 226, 228, 48, 49, 229, 233, 45, 52, 230, 280, 50, 51, 231, 45, 53, 232, 54, 232, 233, 55, 57, 179, 180, 234, 235, 55, 58, 179, 180, 234, 235, 55, 179, 180, 234, 235, 236, 55, 179, 180, 234, 235, 237, 59, 60, 238, 242, 56, 63, 239, 280, 61, 62, 240, 56, 64, 241, 65, 241, 242, 66, 68, 181, 182, 243, 244, 66, 69, 181, 182, 243, 244, 66, 181, 182, 243, 244, 245, 66, 181, 182, 243, 244, 246, 70, 71, 247, 251, 67, 74, 248, 280, 72, 73, 249, 67, 75, 250, 76, 250, 251, 77, 79, 183, 184, 252, 253, 77, 80, 183, 184, 252, 253, 77, 183, 184, 252, 253, 254, 77, 183, 184, 252, 253, 255, 81, 82, 256, 260, 78, 85, 257, 280, 83, 84, 258, 78, 86, 259, 87, 259, 260, 88, 90, 185, 186, 261, 262, 88, 91, 185, 186, 261, 262, 88, 185, 186, 261, 262, 263, 88, 185, 186, 261, 262, 264, 92, 93, 265, 269, 89, 96, 266, 280, 94, 95, 267, 89, 97, 268, 98, 268, 269, 99, 101, 187, 188, 270, 271, 99, 102, 187, 188, 270, 271, 99, 187, 188, 270, 271, 272, 99, 187, 188, 270, 271, 273, 103, 104, 274, 278, 100, 107, 275, 280, 105, 106, 276, 100, 108, 277, 109, 277, 278, 1, 12, 23, 34, 45, 56, 67, 78, 89, 100, 279, 110, 279, 280]
    sp_jac_trap_ja = [0, 3, 11, 14, 16, 18, 22, 24, 26, 28, 30, 32, 35, 43, 46, 48, 50, 54, 56, 58, 60, 62, 64, 67, 75, 78, 80, 82, 86, 88, 90, 92, 94, 96, 99, 107, 110, 112, 114, 118, 120, 122, 124, 126, 128, 131, 139, 142, 144, 146, 150, 152, 154, 156, 158, 160, 163, 171, 174, 176, 178, 182, 184, 186, 188, 190, 192, 195, 203, 206, 208, 210, 214, 216, 218, 220, 222, 224, 227, 235, 238, 240, 242, 246, 248, 250, 252, 254, 256, 259, 267, 270, 272, 274, 278, 280, 282, 284, 286, 288, 291, 299, 302, 304, 306, 310, 312, 314, 316, 318, 320, 322, 328, 334, 344, 354, 362, 370, 378, 386, 394, 402, 412, 422, 428, 434, 442, 450, 456, 462, 470, 478, 486, 494, 500, 506, 514, 522, 530, 538, 544, 550, 562, 574, 582, 590, 596, 602, 610, 618, 624, 630, 636, 642, 650, 658, 666, 674, 680, 686, 694, 702, 712, 722, 728, 734, 740, 746, 754, 762, 767, 772, 777, 782, 787, 792, 797, 802, 807, 812, 817, 822, 827, 832, 837, 842, 847, 852, 859, 866, 872, 878, 884, 890, 894, 898, 901, 904, 907, 913, 919, 925, 931, 935, 939, 942, 945, 948, 954, 960, 966, 972, 976, 980, 983, 986, 989, 995, 1001, 1007, 1013, 1017, 1021, 1024, 1027, 1030, 1036, 1042, 1048, 1054, 1058, 1062, 1065, 1068, 1071, 1077, 1083, 1089, 1095, 1099, 1103, 1106, 1109, 1112, 1118, 1124, 1130, 1136, 1140, 1144, 1147, 1150, 1153, 1159, 1165, 1171, 1177, 1181, 1185, 1188, 1191, 1194, 1200, 1206, 1212, 1218, 1222, 1226, 1229, 1232, 1235, 1241, 1247, 1253, 1259, 1263, 1267, 1270, 1273, 1276, 1287, 1290]
    sp_jac_trap_nia = 281
    sp_jac_trap_nja = 281
    return sp_jac_trap_ia, sp_jac_trap_ja, sp_jac_trap_nia, sp_jac_trap_nja 
