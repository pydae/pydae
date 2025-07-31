import numpy as np
import numba
import scipy.optimize as sopt
import scipy.sparse as sspa
from scipy.sparse.linalg import spsolve,spilu,splu
from numba import cuda
import cffi
import numba.core.typing.cffi_utils as cffi_support
from io import BytesIO
import pkgutil
import os

dae_file_mode = 'local'

ffi = cffi.FFI()

if dae_file_mode == 'local':
    import temp_cffi as jacs
if dae_file_mode == 'enviroment':
    import envus.no_enviroment.temp_cffi as jacs
if dae_file_mode == 'colab':
    import temp_cffi as jacs
if dae_file_mode == 'testing':
    from pydae.temp import temp_cffi as jacs
    
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

sp_Fu_run_up_eval = jacs.lib.sp_Fu_run_up_eval
sp_Gu_run_up_eval = jacs.lib.sp_Gu_run_up_eval
sp_Hx_run_up_eval = jacs.lib.sp_Hx_run_up_eval
sp_Hy_run_up_eval = jacs.lib.sp_Hy_run_up_eval
sp_Hu_run_up_eval = jacs.lib.sp_Hu_run_up_eval
sp_Fu_run_xy_eval = jacs.lib.sp_Fu_run_xy_eval
sp_Gu_run_xy_eval = jacs.lib.sp_Gu_run_xy_eval
sp_Hx_run_xy_eval = jacs.lib.sp_Hx_run_xy_eval
sp_Hy_run_xy_eval = jacs.lib.sp_Hy_run_xy_eval
sp_Hu_run_xy_eval = jacs.lib.sp_Hu_run_xy_eval



import json

sin = np.sin
cos = np.cos
atan2 = np.arctan2
sqrt = np.sqrt 
sign = np.sign 
exp = np.exp


class model: 

    def __init__(self,matrices_folder='./build'): 

        array = np.array
        
        self.matrices_folder = matrices_folder
        
        self.dae_file_mode = 'local'
        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 30
        self.N_y = 96 
        self.N_z = 73 
        self.N_store = 100000 
        self.params_list = ['S_base', 'g_POI_GRID', 'b_POI_GRID', 'bs_POI_GRID', 'g_BESS_POIMV', 'b_BESS_POIMV', 'bs_BESS_POIMV', 'g_LV0101_MV0101', 'b_LV0101_MV0101', 'bs_LV0101_MV0101', 'g_MV0101_POIMV', 'b_MV0101_POIMV', 'bs_MV0101_POIMV', 'g_LV0102_MV0102', 'b_LV0102_MV0102', 'bs_LV0102_MV0102', 'g_MV0102_MV0101', 'b_MV0102_MV0101', 'bs_MV0102_MV0101', 'g_LV0103_MV0103', 'b_LV0103_MV0103', 'bs_LV0103_MV0103', 'g_MV0103_MV0102', 'b_MV0103_MV0102', 'bs_MV0103_MV0102', 'g_LV0201_MV0201', 'b_LV0201_MV0201', 'bs_LV0201_MV0201', 'g_MV0201_POIMV', 'b_MV0201_POIMV', 'bs_MV0201_POIMV', 'g_LV0202_MV0202', 'b_LV0202_MV0202', 'bs_LV0202_MV0202', 'g_MV0202_MV0201', 'b_MV0202_MV0201', 'bs_MV0202_MV0201', 'g_LV0203_MV0203', 'b_LV0203_MV0203', 'bs_LV0203_MV0203', 'g_MV0203_MV0202', 'b_MV0203_MV0202', 'bs_MV0203_MV0202', 'g_cc_POIMV_POI', 'b_cc_POIMV_POI', 'tap_POIMV_POI', 'ang_POIMV_POI', 'U_POIMV_n', 'U_POI_n', 'U_GRID_n', 'U_BESS_n', 'U_LV0101_n', 'U_MV0101_n', 'U_LV0102_n', 'U_MV0102_n', 'U_LV0103_n', 'U_MV0103_n', 'U_LV0201_n', 'U_MV0201_n', 'U_LV0202_n', 'U_MV0202_n', 'U_LV0203_n', 'U_MV0203_n', 'K_p_BESS', 'K_i_BESS', 'soc_min_BESS', 'soc_max_BESS', 'S_n_BESS', 'E_kWh_BESS', 'A_loss_BESS', 'B_loss_BESS', 'C_loss_BESS', 'R_bat_BESS', 'S_n_GRID', 'F_n_GRID', 'X_v_GRID', 'R_v_GRID', 'K_delta_GRID', 'K_alpha_GRID', 'K_rocov_GRID', 'I_sc_LV0101', 'I_mp_LV0101', 'V_mp_LV0101', 'V_oc_LV0101', 'N_pv_s_LV0101', 'N_pv_p_LV0101', 'K_vt_LV0101', 'K_it_LV0101', 'v_lvrt_LV0101', 'T_lp1p_LV0101', 'T_lp2p_LV0101', 'T_lp1q_LV0101', 'T_lp2q_LV0101', 'PRampUp_LV0101', 'PRampDown_LV0101', 'QRampUp_LV0101', 'QRampDown_LV0101', 'S_n_LV0101', 'F_n_LV0101', 'U_n_LV0101', 'X_s_LV0101', 'R_s_LV0101', 'I_sc_LV0102', 'I_mp_LV0102', 'V_mp_LV0102', 'V_oc_LV0102', 'N_pv_s_LV0102', 'N_pv_p_LV0102', 'K_vt_LV0102', 'K_it_LV0102', 'v_lvrt_LV0102', 'T_lp1p_LV0102', 'T_lp2p_LV0102', 'T_lp1q_LV0102', 'T_lp2q_LV0102', 'PRampUp_LV0102', 'PRampDown_LV0102', 'QRampUp_LV0102', 'QRampDown_LV0102', 'S_n_LV0102', 'F_n_LV0102', 'U_n_LV0102', 'X_s_LV0102', 'R_s_LV0102', 'I_sc_LV0103', 'I_mp_LV0103', 'V_mp_LV0103', 'V_oc_LV0103', 'N_pv_s_LV0103', 'N_pv_p_LV0103', 'K_vt_LV0103', 'K_it_LV0103', 'v_lvrt_LV0103', 'T_lp1p_LV0103', 'T_lp2p_LV0103', 'T_lp1q_LV0103', 'T_lp2q_LV0103', 'PRampUp_LV0103', 'PRampDown_LV0103', 'QRampUp_LV0103', 'QRampDown_LV0103', 'S_n_LV0103', 'F_n_LV0103', 'U_n_LV0103', 'X_s_LV0103', 'R_s_LV0103', 'I_sc_LV0201', 'I_mp_LV0201', 'V_mp_LV0201', 'V_oc_LV0201', 'N_pv_s_LV0201', 'N_pv_p_LV0201', 'K_vt_LV0201', 'K_it_LV0201', 'v_lvrt_LV0201', 'T_lp1p_LV0201', 'T_lp2p_LV0201', 'T_lp1q_LV0201', 'T_lp2q_LV0201', 'PRampUp_LV0201', 'PRampDown_LV0201', 'QRampUp_LV0201', 'QRampDown_LV0201', 'S_n_LV0201', 'F_n_LV0201', 'U_n_LV0201', 'X_s_LV0201', 'R_s_LV0201', 'I_sc_LV0202', 'I_mp_LV0202', 'V_mp_LV0202', 'V_oc_LV0202', 'N_pv_s_LV0202', 'N_pv_p_LV0202', 'K_vt_LV0202', 'K_it_LV0202', 'v_lvrt_LV0202', 'T_lp1p_LV0202', 'T_lp2p_LV0202', 'T_lp1q_LV0202', 'T_lp2q_LV0202', 'PRampUp_LV0202', 'PRampDown_LV0202', 'QRampUp_LV0202', 'QRampDown_LV0202', 'S_n_LV0202', 'F_n_LV0202', 'U_n_LV0202', 'X_s_LV0202', 'R_s_LV0202', 'I_sc_LV0203', 'I_mp_LV0203', 'V_mp_LV0203', 'V_oc_LV0203', 'N_pv_s_LV0203', 'N_pv_p_LV0203', 'K_vt_LV0203', 'K_it_LV0203', 'v_lvrt_LV0203', 'T_lp1p_LV0203', 'T_lp2p_LV0203', 'T_lp1q_LV0203', 'T_lp2q_LV0203', 'PRampUp_LV0203', 'PRampDown_LV0203', 'QRampUp_LV0203', 'QRampDown_LV0203', 'S_n_LV0203', 'F_n_LV0203', 'U_n_LV0203', 'X_s_LV0203', 'R_s_LV0203', 'K_p_agc', 'K_i_agc', 'K_xif'] 
        self.params_values_list  = [10000000.0, 0.0, -3840.0, 0.0, 0.0, -100.0, 0.0, 1.3846963291914642, -6.9229831951548775, 0.0, 1967.221207182239, -1331.8454135581617, 0.00027712, 1.3846963291914642, -6.9229831951548775, 0.0, 983.6106035911195, -665.9227067790808, 0.00055424, 1.3846963291914642, -6.9229831951548775, 0.0, 983.6106035911195, -665.9227067790808, 0.00055424, 1.3846963291914642, -6.9229831951548775, 0.0, 655.7404023940798, -443.948471186054, 0.00083136, 1.3846963291914642, -6.9229831951548775, 0.0, 983.6106035911195, -665.9227067790808, 0.00055424, 1.3846963291914642, -6.9229831951548775, 0.0, 983.6106035911195, -665.9227067790808, 0.00055424, 0.0, -19.2, 1.0, 0.0, 20000.0, 132000.0, 132000.0, 400.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 400.0, 20000.0, 1e-06, 1e-06, 0.0, 1.0, 1000000.0, 250, 0.0001, 0.0, 0.0001, 0.0, 1000000000.0, 50.0, 0.001, 0.0, 0.001, 1e-06, 1e-06, 8, 3.56, 33.7, 42.1, 23, 1087, -0.16, 0.065, 0.8, 0.02, 0.02, 0.02, 0.02, 5.0, -5.0, 5.0, -5.0, 3000000.0, 50.0, 400, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 23, 1087, -0.16, 0.065, 0.8, 0.02, 0.02, 0.02, 0.02, 5.0, -5.0, 5.0, -5.0, 3000000.0, 50.0, 400, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 23, 1087, -0.16, 0.065, 0.8, 0.02, 0.02, 0.02, 0.02, 5.0, -5.0, 5.0, -5.0, 3000000.0, 50.0, 400, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 23, 1087, -0.16, 0.065, 0.8, 0.02, 0.02, 0.02, 0.02, 5.0, -5.0, 5.0, -5.0, 3000000.0, 50.0, 400, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 23, 1087, -0.16, 0.065, 0.8, 0.02, 0.02, 0.02, 0.02, 5.0, -5.0, 5.0, -5.0, 3000000.0, 50.0, 400, 0.1, 0.0001, 8, 3.56, 33.7, 42.1, 23, 1087, -0.16, 0.065, 0.8, 0.02, 0.02, 0.02, 0.02, 5.0, -5.0, 5.0, -5.0, 3000000.0, 50.0, 400, 0.1, 0.0001, 0.0, 0.0, 0.01] 
        self.inputs_ini_list = ['P_POIMV', 'Q_POIMV', 'P_POI', 'Q_POI', 'P_GRID', 'Q_GRID', 'P_BESS', 'Q_BESS', 'P_LV0101', 'Q_LV0101', 'P_MV0101', 'Q_MV0101', 'P_LV0102', 'Q_LV0102', 'P_MV0102', 'Q_MV0102', 'P_LV0103', 'Q_LV0103', 'P_MV0103', 'Q_MV0103', 'P_LV0201', 'Q_LV0201', 'P_MV0201', 'Q_MV0201', 'P_LV0202', 'Q_LV0202', 'P_MV0202', 'Q_MV0202', 'P_LV0203', 'Q_LV0203', 'P_MV0203', 'Q_MV0203', 'p_s_ref_BESS', 'q_s_ref_BESS', 'soc_ref_BESS', 'alpha_GRID', 'v_ref_GRID', 'omega_ref_GRID', 'delta_ref_GRID', 'phi_GRID', 'rocov_GRID', 'irrad_LV0101', 'temp_deg_LV0101', 'lvrt_ext_LV0101', 'ramp_enable_LV0101', 'p_s_ppc_LV0101', 'q_s_ppc_LV0101', 'i_sa_ref_LV0101', 'i_sr_ref_LV0101', 'irrad_LV0102', 'temp_deg_LV0102', 'lvrt_ext_LV0102', 'ramp_enable_LV0102', 'p_s_ppc_LV0102', 'q_s_ppc_LV0102', 'i_sa_ref_LV0102', 'i_sr_ref_LV0102', 'irrad_LV0103', 'temp_deg_LV0103', 'lvrt_ext_LV0103', 'ramp_enable_LV0103', 'p_s_ppc_LV0103', 'q_s_ppc_LV0103', 'i_sa_ref_LV0103', 'i_sr_ref_LV0103', 'irrad_LV0201', 'temp_deg_LV0201', 'lvrt_ext_LV0201', 'ramp_enable_LV0201', 'p_s_ppc_LV0201', 'q_s_ppc_LV0201', 'i_sa_ref_LV0201', 'i_sr_ref_LV0201', 'irrad_LV0202', 'temp_deg_LV0202', 'lvrt_ext_LV0202', 'ramp_enable_LV0202', 'p_s_ppc_LV0202', 'q_s_ppc_LV0202', 'i_sa_ref_LV0202', 'i_sr_ref_LV0202', 'irrad_LV0203', 'temp_deg_LV0203', 'lvrt_ext_LV0203', 'ramp_enable_LV0203', 'p_s_ppc_LV0203', 'q_s_ppc_LV0203', 'i_sa_ref_LV0203', 'i_sr_ref_LV0203'] 
        self.inputs_ini_values_list  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0, 1.0, 1.0, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1e-05, 1.5, 0.0, 0.0, 0.0] 
        self.inputs_run_list = ['P_POIMV', 'Q_POIMV', 'P_POI', 'Q_POI', 'P_GRID', 'Q_GRID', 'P_BESS', 'Q_BESS', 'P_LV0101', 'Q_LV0101', 'P_MV0101', 'Q_MV0101', 'P_LV0102', 'Q_LV0102', 'P_MV0102', 'Q_MV0102', 'P_LV0103', 'Q_LV0103', 'P_MV0103', 'Q_MV0103', 'P_LV0201', 'Q_LV0201', 'P_MV0201', 'Q_MV0201', 'P_LV0202', 'Q_LV0202', 'P_MV0202', 'Q_MV0202', 'P_LV0203', 'Q_LV0203', 'P_MV0203', 'Q_MV0203', 'p_s_ref_BESS', 'q_s_ref_BESS', 'soc_ref_BESS', 'alpha_GRID', 'v_ref_GRID', 'omega_ref_GRID', 'delta_ref_GRID', 'phi_GRID', 'rocov_GRID', 'irrad_LV0101', 'temp_deg_LV0101', 'lvrt_ext_LV0101', 'ramp_enable_LV0101', 'p_s_ppc_LV0101', 'q_s_ppc_LV0101', 'i_sa_ref_LV0101', 'i_sr_ref_LV0101', 'irrad_LV0102', 'temp_deg_LV0102', 'lvrt_ext_LV0102', 'ramp_enable_LV0102', 'p_s_ppc_LV0102', 'q_s_ppc_LV0102', 'i_sa_ref_LV0102', 'i_sr_ref_LV0102', 'irrad_LV0103', 'temp_deg_LV0103', 'lvrt_ext_LV0103', 'ramp_enable_LV0103', 'p_s_ppc_LV0103', 'q_s_ppc_LV0103', 'i_sa_ref_LV0103', 'i_sr_ref_LV0103', 'irrad_LV0201', 'temp_deg_LV0201', 'lvrt_ext_LV0201', 'ramp_enable_LV0201', 'p_s_ppc_LV0201', 'q_s_ppc_LV0201', 'i_sa_ref_LV0201', 'i_sr_ref_LV0201', 'irrad_LV0202', 'temp_deg_LV0202', 'lvrt_ext_LV0202', 'ramp_enable_LV0202', 'p_s_ppc_LV0202', 'q_s_ppc_LV0202', 'i_sa_ref_LV0202', 'i_sr_ref_LV0202', 'irrad_LV0203', 'temp_deg_LV0203', 'lvrt_ext_LV0203', 'ramp_enable_LV0203', 'p_s_ppc_LV0203', 'q_s_ppc_LV0203', 'i_sa_ref_LV0203', 'i_sr_ref_LV0203'] 
        self.inputs_run_values_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0, 1.0, 1.0, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0, 1000.0, 25.0, 0.0, 1.0, 1.5, 0.0, 0.0, 0.0] 
        self.outputs_list = ['V_POIMV', 'V_POI', 'V_GRID', 'V_BESS', 'V_LV0101', 'V_MV0101', 'V_LV0102', 'V_MV0102', 'V_LV0103', 'V_MV0103', 'V_LV0201', 'V_MV0201', 'V_LV0202', 'V_MV0202', 'V_LV0203', 'V_MV0203', 'p_line_POI_GRID', 'q_line_POI_GRID', 'p_line_GRID_POI', 'q_line_GRID_POI', 'I_line_POI_GRID', 'I_line_GRID_POI', 'p_line_BESS_POIMV', 'q_line_BESS_POIMV', 'p_line_POIMV_BESS', 'q_line_POIMV_BESS', 'I_line_BESS_POIMV', 'I_line_POIMV_BESS', 'p_line_MV0101_POIMV', 'q_line_MV0101_POIMV', 'p_line_POIMV_MV0101', 'q_line_POIMV_MV0101', 'I_line_MV0101_POIMV', 'I_line_POIMV_MV0101', 'p_line_MV0201_POIMV', 'q_line_MV0201_POIMV', 'p_line_POIMV_MV0201', 'q_line_POIMV_MV0201', 'I_line_MV0201_POIMV', 'I_line_POIMV_MV0201', 'p_loss_BESS', 'i_s_BESS', 'e_BESS', 'i_dc_BESS', 'p_s_BESS', 'q_s_BESS', 'alpha_GRID', 'Dv_GRID', 'theta_v_GRID', 'm_ref_LV0101', 'v_sd_LV0101', 'v_sq_LV0101', 'lvrt_LV0101', 'm_ref_LV0102', 'v_sd_LV0102', 'v_sq_LV0102', 'lvrt_LV0102', 'm_ref_LV0103', 'v_sd_LV0103', 'v_sq_LV0103', 'lvrt_LV0103', 'm_ref_LV0201', 'v_sd_LV0201', 'v_sq_LV0201', 'lvrt_LV0201', 'm_ref_LV0202', 'v_sd_LV0202', 'v_sq_LV0202', 'lvrt_LV0202', 'm_ref_LV0203', 'v_sd_LV0203', 'v_sq_LV0203', 'lvrt_LV0203'] 
        self.x_list = ['soc_BESS', 'xi_soc_BESS', 'delta_GRID', 'Domega_GRID', 'Dv_GRID', 'x_p_lp1_LV0101', 'x_p_lp2_LV0101', 'x_q_lp1_LV0101', 'x_q_lp2_LV0101', 'x_p_lp1_LV0102', 'x_p_lp2_LV0102', 'x_q_lp1_LV0102', 'x_q_lp2_LV0102', 'x_p_lp1_LV0103', 'x_p_lp2_LV0103', 'x_q_lp1_LV0103', 'x_q_lp2_LV0103', 'x_p_lp1_LV0201', 'x_p_lp2_LV0201', 'x_q_lp1_LV0201', 'x_q_lp2_LV0201', 'x_p_lp1_LV0202', 'x_p_lp2_LV0202', 'x_q_lp1_LV0202', 'x_q_lp2_LV0202', 'x_p_lp1_LV0203', 'x_p_lp2_LV0203', 'x_q_lp1_LV0203', 'x_q_lp2_LV0203', 'xi_freq'] 
        self.y_run_list = ['V_POIMV', 'theta_POIMV', 'V_POI', 'theta_POI', 'V_GRID', 'theta_GRID', 'V_BESS', 'theta_BESS', 'V_LV0101', 'theta_LV0101', 'V_MV0101', 'theta_MV0101', 'V_LV0102', 'theta_LV0102', 'V_MV0102', 'theta_MV0102', 'V_LV0103', 'theta_LV0103', 'V_MV0103', 'theta_MV0103', 'V_LV0201', 'theta_LV0201', 'V_MV0201', 'theta_MV0201', 'V_LV0202', 'theta_LV0202', 'V_MV0202', 'theta_MV0202', 'V_LV0203', 'theta_LV0203', 'V_MV0203', 'theta_MV0203', 'p_line_pu_POI_GRID', 'q_line_pu_POI_GRID', 'p_line_pu_GRID_POI', 'q_line_pu_GRID_POI', 'p_line_pu_BESS_POIMV', 'q_line_pu_BESS_POIMV', 'p_line_pu_POIMV_BESS', 'q_line_pu_POIMV_BESS', 'p_line_pu_MV0101_POIMV', 'q_line_pu_MV0101_POIMV', 'p_line_pu_POIMV_MV0101', 'q_line_pu_POIMV_MV0101', 'p_line_pu_MV0201_POIMV', 'q_line_pu_MV0201_POIMV', 'p_line_pu_POIMV_MV0201', 'q_line_pu_POIMV_MV0201', 'p_dc_BESS', 'i_dc_BESS', 'v_dc_BESS', 'omega_GRID', 'v_dc_LV0101', 'i_sq_ref_LV0101', 'i_sd_ref_LV0101', 'i_sr_LV0101', 'i_si_LV0101', 'p_s_LV0101', 'q_s_LV0101', 'v_dc_LV0102', 'i_sq_ref_LV0102', 'i_sd_ref_LV0102', 'i_sr_LV0102', 'i_si_LV0102', 'p_s_LV0102', 'q_s_LV0102', 'v_dc_LV0103', 'i_sq_ref_LV0103', 'i_sd_ref_LV0103', 'i_sr_LV0103', 'i_si_LV0103', 'p_s_LV0103', 'q_s_LV0103', 'v_dc_LV0201', 'i_sq_ref_LV0201', 'i_sd_ref_LV0201', 'i_sr_LV0201', 'i_si_LV0201', 'p_s_LV0201', 'q_s_LV0201', 'v_dc_LV0202', 'i_sq_ref_LV0202', 'i_sd_ref_LV0202', 'i_sr_LV0202', 'i_si_LV0202', 'p_s_LV0202', 'q_s_LV0202', 'v_dc_LV0203', 'i_sq_ref_LV0203', 'i_sd_ref_LV0203', 'i_sr_LV0203', 'i_si_LV0203', 'p_s_LV0203', 'q_s_LV0203', 'omega_coi', 'p_agc'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['V_POIMV', 'theta_POIMV', 'V_POI', 'theta_POI', 'V_GRID', 'theta_GRID', 'V_BESS', 'theta_BESS', 'V_LV0101', 'theta_LV0101', 'V_MV0101', 'theta_MV0101', 'V_LV0102', 'theta_LV0102', 'V_MV0102', 'theta_MV0102', 'V_LV0103', 'theta_LV0103', 'V_MV0103', 'theta_MV0103', 'V_LV0201', 'theta_LV0201', 'V_MV0201', 'theta_MV0201', 'V_LV0202', 'theta_LV0202', 'V_MV0202', 'theta_MV0202', 'V_LV0203', 'theta_LV0203', 'V_MV0203', 'theta_MV0203', 'p_line_pu_POI_GRID', 'q_line_pu_POI_GRID', 'p_line_pu_GRID_POI', 'q_line_pu_GRID_POI', 'p_line_pu_BESS_POIMV', 'q_line_pu_BESS_POIMV', 'p_line_pu_POIMV_BESS', 'q_line_pu_POIMV_BESS', 'p_line_pu_MV0101_POIMV', 'q_line_pu_MV0101_POIMV', 'p_line_pu_POIMV_MV0101', 'q_line_pu_POIMV_MV0101', 'p_line_pu_MV0201_POIMV', 'q_line_pu_MV0201_POIMV', 'p_line_pu_POIMV_MV0201', 'q_line_pu_POIMV_MV0201', 'p_dc_BESS', 'i_dc_BESS', 'v_dc_BESS', 'omega_GRID', 'v_dc_LV0101', 'i_sq_ref_LV0101', 'i_sd_ref_LV0101', 'i_sr_LV0101', 'i_si_LV0101', 'p_s_LV0101', 'q_s_LV0101', 'v_dc_LV0102', 'i_sq_ref_LV0102', 'i_sd_ref_LV0102', 'i_sr_LV0102', 'i_si_LV0102', 'p_s_LV0102', 'q_s_LV0102', 'v_dc_LV0103', 'i_sq_ref_LV0103', 'i_sd_ref_LV0103', 'i_sr_LV0103', 'i_si_LV0103', 'p_s_LV0103', 'q_s_LV0103', 'v_dc_LV0201', 'i_sq_ref_LV0201', 'i_sd_ref_LV0201', 'i_sr_LV0201', 'i_si_LV0201', 'p_s_LV0201', 'q_s_LV0201', 'v_dc_LV0202', 'i_sq_ref_LV0202', 'i_sd_ref_LV0202', 'i_sr_LV0202', 'i_si_LV0202', 'p_s_LV0202', 'q_s_LV0202', 'v_dc_LV0203', 'i_sq_ref_LV0203', 'i_sd_ref_LV0203', 'i_sr_LV0203', 'i_si_LV0203', 'p_s_LV0203', 'q_s_LV0203', 'omega_coi', 'p_agc'] 
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
        self.u_ini = np.array(self.inputs_ini_values_list,dtype=np.float64)
        self.p = np.array(self.params_values_list,dtype=np.float64)
        self.xy_0 = np.zeros((self.N_x+self.N_y,),dtype=np.float64)
        self.xy = np.zeros((self.N_x+self.N_y,),dtype=np.float64)
        self.z = np.zeros((self.N_z,),dtype=np.float64)
        
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
        #self.sp_jac_ini = sspa.csr_matrix((data, self.sp_jac_ini_ia, self.sp_jac_ini_ja), shape=(self.sp_jac_ini_nia,self.sp_jac_ini_nja))
           
        if self.dae_file_mode == 'enviroment':
            fobj = BytesIO(pkgutil.get_data(__name__, f'./temp_sp_jac_ini_num.npz'))
            self.sp_jac_ini = sspa.load_npz(fobj)
        else:
            self.sp_jac_ini = sspa.load_npz(f'{self.matrices_folder}/temp_sp_jac_ini_num.npz')
            
            
        self.jac_ini = self.sp_jac_ini.toarray()

        #self.J_ini_d = np.array(self.sp_jac_ini_ia)*0.0
        #self.J_ini_i = np.array(self.sp_jac_ini_ia)
        #self.J_ini_p = np.array(self.sp_jac_ini_ja)
        de_jac_ini_eval(self.jac_ini,x,y,self.u_ini,self.p,self.Dt)
        sp_jac_ini_eval(self.sp_jac_ini.data,x,y,self.u_ini,self.p,self.Dt) 
        self.fill_factor_ini,self.drop_tol_ini,self.drop_rule_ini = 100,1e-10,'basic'       


        ## jac_run
        self.jac_run = np.zeros((self.N_x+self.N_y,self.N_x+self.N_y))
        self.sp_jac_run_ia, self.sp_jac_run_ja, self.sp_jac_run_nia, self.sp_jac_run_nja = sp_jac_run_vectors()
        data = np.array(self.sp_jac_run_ia,dtype=np.float64)

        if self.dae_file_mode == 'enviroment':
            fobj = BytesIO(pkgutil.get_data(__name__, './temp_sp_jac_run_num.npz'))
            self.sp_jac_run = sspa.load_npz(fobj)
        else:
            self.sp_jac_run = sspa.load_npz(f'{self.matrices_folder}/temp_sp_jac_run_num.npz')
        self.jac_run = self.sp_jac_run.toarray()            
           
        self.J_run_d = np.array(self.sp_jac_run_ia)*0.0
        self.J_run_i = np.array(self.sp_jac_run_ia)
        self.J_run_p = np.array(self.sp_jac_run_ja)
        de_jac_run_eval(self.jac_run,x,y,self.u_run,self.p,self.Dt)
        sp_jac_run_eval(self.J_run_d,x,y,self.u_run,self.p,self.Dt)
        
        ## jac_trap
        self.jac_trap = np.zeros((self.N_x+self.N_y,self.N_x+self.N_y))
        self.sp_jac_trap_ia, self.sp_jac_trap_ja, self.sp_jac_trap_nia, self.sp_jac_trap_nja = sp_jac_trap_vectors()
        data = np.array(self.sp_jac_trap_ia,dtype=np.float64)
        #self.sp_jac_trap = sspa.csr_matrix((data, self.sp_jac_trap_ia, self.sp_jac_trap_ja), shape=(self.sp_jac_trap_nia,self.sp_jac_trap_nja))
       
    

        if self.dae_file_mode == 'enviroment':
            fobj = BytesIO(pkgutil.get_data(__name__, './temp_sp_jac_trap_num.npz'))
            self.sp_jac_trap = sspa.load_npz(fobj)
        else:
            self.sp_jac_trap = sspa.load_npz(f'{self.matrices_folder}/temp_sp_jac_trap_num.npz')
            

        self.jac_trap = self.sp_jac_trap.toarray()
        
        #self.J_trap_d = np.array(self.sp_jac_trap_ia)*0.0
        #self.J_trap_i = np.array(self.sp_jac_trap_ia)
        #self.J_trap_p = np.array(self.sp_jac_trap_ja)
        de_jac_trap_eval(self.jac_trap,x,y,self.u_run,self.p,self.Dt)
        sp_jac_trap_eval(self.sp_jac_trap.data,x,y,self.u_run,self.p,self.Dt)
        self.fill_factor_trap,self.drop_tol_trap,self.drop_rule_trap = 100,1e-10,'basic' 
   

        

        
        self.max_it,self.itol,self.store = 50,1e-8,1 
        self.lmax_it,self.ltol,self.ldamp= 50,1e-8,1.0
        self.mode = 0 

        self.lmax_it_ini,self.ltol_ini,self.ldamp_ini=50,1e-8,1.0

        self.sp_Fu_run = sspa.load_npz(f'{self.matrices_folder}/temp_Fu_run_num.npz')
        self.sp_Gu_run = sspa.load_npz(f'{self.matrices_folder}/temp_Gu_run_num.npz')
        self.sp_Hx_run = sspa.load_npz(f'{self.matrices_folder}/temp_Hx_run_num.npz')
        self.sp_Hy_run = sspa.load_npz(f'{self.matrices_folder}/temp_Hy_run_num.npz')
        self.sp_Hu_run = sspa.load_npz(f'{self.matrices_folder}/temp_Hu_run_num.npz')        
        
        self.ss_solver = 2
        self.lsolver = 2
 
        



        
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
        z = self.z
        
        t,it,it_store,xy = daesolver(t,t_end,it,it_store,xy,u,p,z,
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
        self.z = z
 
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
        #self.u_run = np.copy(self.u_ini)
        
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
            print(f'{item:5s} = {self.get_value(item):{value_format}}')

    def report_y(self,value_format='5.2f'):
        for item in self.y_run_list:
            print(f'{item:5s} = {self.get_value(item):{value_format}}')
            
    def report_u(self,value_format='5.2f'):
        for item in self.inputs_run_list:
            print(f'{item:5s} ={self.get_value(item):{value_format}}')

    def report_z(self,value_format='5.2f'):
        for item in self.outputs_list:
            print(f'{item:5s} = {self.get_value(item):{value_format}}')

    def report_params(self,value_format='5.2f'):
        for item in self.params_list:
            print(f'{item:5s} ={self.get_value(item):{value_format}}')
            
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
        
        if it < self.max_it-1:
            
            self.xy_ini = xy_ini
            self.N_iters = it

            self.ini2run()
            
            self.ini_convergence = True
            
        if it >= self.max_it-1:
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
            self.set_value(item, self.data[item])

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
    
        sp_jac_trap_eval(self.sp_jac_trap.data,self.x,self.y_run,self.u_run,self.p,self.Dt)
    
        #self.sp_jac_trap.data = self.J_trap_d 
        
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
        z = self.z
        self.iparams_run = np.zeros(10,dtype=np.float64)
    
        t,it,it_store,xy = spdaesolver(t,t_end,it,it_store,xy,u,p,z,
                                  self.sp_jac_trap.data,self.sp_jac_trap.indices,self.sp_jac_trap.indptr,
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
                                  lmax_it=self.lmax_it,ltol=self.ltol,ldamp=self.ldamp,mode=self.mode,
                                  lsolver = self.lsolver)
    
        self.t = t
        self.it = it
        self.it_store = it_store
        self.xy = xy
        self.z = z

            
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


        if self.N_iters < self.max_it:
            
            self.ini2run()           
            self.ini_convergence = True
            
        if self.N_iters >= self.max_it:
            print(f'Maximum number of iterations (max_it = {self.max_it}) reached without convergence.')
            self.ini_convergence = False
            
        #jac_run_eval_xy(self.jac_run,self.x,self.y_run,self.u_run,self.p)
        #jac_run_eval_up(self.jac_run,self.x,self.y_run,self.u_run,self.p)
        
        return self.ini_convergence

        
    def spss_ini(self):
        J_d,J_i,J_p = csr2pydae(self.sp_jac_ini)
        
        xy_ini,it,iparams = spsstate(self.xy,self.u_ini,self.p,
                 self.sp_jac_ini.data,self.sp_jac_ini.indices,self.sp_jac_ini.indptr,
                 self.P_d,self.P_i,self.P_p,self.perm_r,self.perm_c,
                 self.N_x,self.N_y,
                 max_it=self.max_it,tol=self.itol,
                 lmax_it=self.lmax_it_ini,
                 ltol=self.ltol_ini,
                 ldamp=self.ldamp,solver=self.ss_solver)

 
        self.xy_ini = xy_ini
        self.N_iters = it
        self.iparams = iparams
    
        return xy_ini

    #def import_cffi(self):
        

    def eval_jac_u2z(self):

        '''

        0 =   J_run * xy + FG_u * u
        z = Hxy_run * xy + H_u * u

        xy = -1/J_run * FG_u * u
        z = -Hxy_run/J_run * FG_u * u + H_u * u
        z = (-Hxy_run/J_run * FG_u + H_u ) * u 
        '''
        
        sp_Fu_run_eval(self.sp_Fu_run.data,self.x,self.y_run,self.u_run,self.p,self.Dt)
        sp_Gu_run_eval(self.sp_Gu_run.data,self.x,self.y_run,self.u_run,self.p,self.Dt)
        sp_H_jacs_run_eval(self.sp_Hx_run.data,
                        self.sp_Hy_run.data,
                        self.sp_Hu_run.data,
                        self.x,self.y_run,self.u_run,self.p,self.Dt)
        sp_jac_run = self.sp_jac_run
        sp_jac_run_eval(sp_jac_run.data,
                        self.x,self.y_run,
                        self.u_run,self.p,
                        self.Dt)



        Hxy_run = sspa.bmat([[self.sp_Hx_run,self.sp_Hy_run]])
        FGu_run = sspa.bmat([[self.sp_Fu_run],[self.sp_Gu_run]])
        

        #((sspa.linalg.spsolve(s.sp_jac_ini,-Hxy_run)) @ FGu_run + sp_Hu_run )@s.u_ini

        self.jac_u2z = Hxy_run @ sspa.linalg.spsolve(self.sp_jac_run,-FGu_run) + self.sp_Hu_run  
        
        
    def step(self,t_end,up_dict):
        for item in up_dict:
            self.set_value(item,up_dict[item])

        t = self.t
        p = self.p
        it = self.it
        it_store = self.it_store
        xy = self.xy
        u = self.u_run
        z = self.z

        t,it,xy = daestep(t,t_end,it,
                          xy,u,p,z,
                          self.jac_trap,
                          self.iters,
                          self.Dt,
                          self.N_x,
                          self.N_y,
                          self.N_z,
                          max_it=self.max_it,itol=self.itol,store=self.store)

        self.t = t
        self.it = it
        self.it_store = it_store
        self.xy = xy
        self.z = z
           
            
    def save_run(self,file_name):
        np.savez(file_name,Time=self.Time,
             X=self.X,Y=self.Y,Z=self.Z,
             x_list = self.x_list,
             y_ini_list = self.y_ini_list,
             y_run_list = self.y_run_list,
             u_ini_list=self.u_ini_list,
             u_run_list=self.u_run_list,  
             z_list=self.outputs_list, 
            )
        
    def load_run(self,file_name):
        data = np.load(f'{file_name}.npz')
        self.Time = data['Time']
        self.X = data['X']
        self.Y = data['Y']
        self.Z = data['Z']
        self.x_list = list(data['x_list'] )
        self.y_run_list = list(data['y_run_list'] )
        self.outputs_list = list(data['z_list'] )
        
    def full_jacs_eval(self):
        N_x = self.N_x
        N_y = self.N_y
        N_xy = N_x + N_y
    
        sp_jac_run = self.sp_jac_run
        sp_Fu = self.sp_Fu_run
        sp_Gu = self.sp_Gu_run
        sp_Hx = self.sp_Hx_run
        sp_Hy = self.sp_Hy_run
        sp_Hu = self.sp_Hu_run
        
        x = self.xy[0:N_x]
        y = self.xy[N_x:]
        u = self.u_run
        p = self.p
        Dt = self.Dt
    
        sp_jac_run_eval(sp_jac_run.data,x,y,u,p,Dt)
        
        self.Fx = sp_jac_run[0:N_x,0:N_x]
        self.Fy = sp_jac_run[ 0:N_x,N_x:]
        self.Gx = sp_jac_run[ N_x:,0:N_x]
        self.Gy = sp_jac_run[ N_x:, N_x:]
        
        sp_Fu_run_eval(sp_Fu.data,x,y,u,p,Dt)
        sp_Gu_run_eval(sp_Gu.data,x,y,u,p,Dt)
        sp_H_jacs_run_eval(sp_Hx.data,sp_Hy.data,sp_Hu.data,x,y,u,p,Dt)
        
        self.Fu = sp_Fu
        self.Gu = sp_Gu
        self.Hx = sp_Hx
        self.Hy = sp_Hy
        self.Hu = sp_Hu


@numba.njit() 
def daestep(t,t_end,it,xy,u,p,z,jac_trap,iters,Dt,N_x,N_y,N_z,max_it=50,itol=1e-8,store=1): 


    fg = np.zeros((N_x+N_y,1),dtype=np.float64)
    fg_i = np.zeros((N_x+N_y),dtype=np.float64)
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    #h = np.zeros((N_z),dtype=np.float64)
    
    f_ptr=ffi.from_buffer(np.ascontiguousarray(f))
    g_ptr=ffi.from_buffer(np.ascontiguousarray(g))
    z_ptr=ffi.from_buffer(np.ascontiguousarray(z))
    x_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    jac_trap_ptr=ffi.from_buffer(np.ascontiguousarray(jac_trap))
    
    #de_jac_trap_num_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)    
    de_jac_trap_up_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    de_jac_trap_xy_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    
    if it == 0:
        f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        h_eval(z_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        it_store = 0  

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
                
        h_eval(z_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        xy[:N_x] = x
        xy[N_x:] = y
        
    return t,it,xy


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
def spconjgradm(A_d,A_i,A_p,b,P_d,P_i,P_p,perm_r,perm_c,x,iparams,max_it=100,tol=1e-3, damp=None):
    """
    A function to solve [A]{x} = {b} linear equation system with the 
    preconditioned conjugate gradient method.
    More at: http://en.wikipedia.org/wiki/Conjugate_gradient_method
    ========== Parameters ==========
    A_d,A_i,A_p : sparse matrix 
        components in CRS form A_d = A_crs.data, A_i = A_crs.indices, A_p = A_crs.indptr.
    b : vector
        The right hand side (RHS) vector of the system.
    x : vector
        The starting guess for the solution.
    P_d,P_i,P_p,perm_r,perm_c: preconditioner LU matrix
        components in scipy.spilu form P_d,P_i,P_p,perm_r,perm_c = slu2pydae(M)
        with M = scipy.sparse.linalg.spilu(A_csc) 

    """  
    N   = len(b)
    Ax  = np.zeros(N)
    Ap  = np.zeros(N)
    App = np.zeros(N)
    pAp = np.zeros(N)
    z   = np.zeros(N)
    
    spMvmul(N,A_d,A_i,A_p,x,Ax)
    r = -(Ax - b)
    z = splu_solve(P_d,P_i,P_p,perm_r,perm_c,r) #z = M.solve(r)
    p = z
    zsold = 0.0
    for it in range(N):  # zsold = np.dot(np.transpose(z), z)
        zsold += z[it]*z[it]
    for i in range(max_it):
        spMvmul(N,A_d,A_i,A_p,p,App)  # #App = np.dot(A, p)
        Ap = splu_solve(P_d,P_i,P_p,perm_r,perm_c,App) #Ap = M.solve(App)
        pAp = 0.0
        for it in range(N):
            pAp += p[it]*Ap[it]

        alpha = zsold / pAp
        x = x + alpha*p
        z = z - alpha*Ap
        zz = 0.0
        for it in range(N):  # z.T@z
            zz += z[it]*z[it]
        zsnew = zz
        if np.sqrt(zsnew) < tol:
            break
            
        p = z + (zsnew/zsold)*p
        zsold = zsnew
    iparams[0] = i

    return x


@numba.njit()
def spsstate(xy,u,p,
             J_d,J_i,J_p,
             P_d,P_i,P_p,perm_r,perm_c,
             N_x,N_y,
             max_it=50,tol=1e-8,
             lmax_it=20,ltol=1e-8,ldamp=1.0, solver=2):
    
   
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

    #sp_jac_ini_num_eval(J_d_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
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
        
        if solver==1:
               
            Dxy = sprichardson(J_d,J_i,J_p,-fg,P_d,P_i,P_p,perm_r,perm_c,Dxy,iparams,damp=ldamp,max_it=lmax_it,tol=ltol)
   
        if solver==2:
            
            Dxy = spconjgradm(J_d,J_i,J_p,-fg,P_d,P_i,P_p,perm_r,perm_c,Dxy,iparams,damp=ldamp,max_it=lmax_it,tol=ltol)
            
        xy += Dxy
        #if np.max(np.abs(fg))<tol: break
        if np.linalg.norm(fg,np.inf)<tol: break

    return xy,it,iparams


    
@numba.njit() 
def daesolver(t,t_end,it,it_store,xy,u,p,z,jac_trap,T,X,Y,Z,iters,Dt,N_x,N_y,N_z,decimation,max_it=50,itol=1e-8,store=1): 


    fg = np.zeros((N_x+N_y,1),dtype=np.float64)
    fg_i = np.zeros((N_x+N_y),dtype=np.float64)
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    #h = np.zeros((N_z),dtype=np.float64)
    
    f_ptr=ffi.from_buffer(np.ascontiguousarray(f))
    g_ptr=ffi.from_buffer(np.ascontiguousarray(g))
    z_ptr=ffi.from_buffer(np.ascontiguousarray(z))
    x_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    jac_trap_ptr=ffi.from_buffer(np.ascontiguousarray(jac_trap))
    
    #de_jac_trap_num_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)    
    de_jac_trap_up_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    de_jac_trap_xy_eval(jac_trap_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    
    if it == 0:
        f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        h_eval(z_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        it_store = 0  
        T[0] = t 
        X[0,:] = x  
        Y[0,:] = y  
        Z[0,:] = z  

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
                
        h_eval(z_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        xy[:N_x] = x
        xy[N_x:] = y
        
        # store in channels 
        if store == 1:
            if it >= it_store*decimation: 
                T[it_store+1] = t 
                X[it_store+1,:] = x 
                Y[it_store+1,:] = y
                Z[it_store+1,:] = z
                iters[it_store+1] = iti
                it_store += 1 

    return t,it,it_store,xy
    
@numba.njit() 
def spdaesolver(t,t_end,it,it_store,xy,u,p,z,
                J_d,J_i,J_p,
                P_d,P_i,P_p,perm_r,perm_c,
                T,X,Y,Z,iters,Dt,N_x,N_y,N_z,decimation,
                iparams,
                max_it=50,itol=1e-8,store=1,
                lmax_it=20,ltol=1e-4,ldamp=1.0,mode=0,lsolver=2):

    fg_i = np.zeros((N_x+N_y),dtype=np.float64)
    x = xy[:N_x]
    y = xy[N_x:]
    fg = np.zeros((N_x+N_y,),dtype=np.float64)
    f = fg[:N_x]
    g = fg[N_x:]
    z = np.zeros((N_z),dtype=np.float64)
    Dxy_i_0 = np.zeros(N_x+N_y,dtype=np.float64) 
    f_ptr=ffi.from_buffer(np.ascontiguousarray(f))
    g_ptr=ffi.from_buffer(np.ascontiguousarray(g))
    z_ptr=ffi.from_buffer(np.ascontiguousarray(z))
    x_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    J_d_ptr=ffi.from_buffer(np.ascontiguousarray(J_d))
    
    #sp_jac_trap_num_eval(J_d_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)    
    sp_jac_trap_up_eval( J_d_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    sp_jac_trap_xy_eval( J_d_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt) 
    
    if it == 0:
        f_run_eval(f_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        g_run_eval(g_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        h_eval(z_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        it_store = 0  
        T[0] = t 
        X[0,:] = x  
        Y[0,:] = y  
        Z[0,:] = z 

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
            if lsolver == 1:
                Dxy_i = sprichardson(J_d,J_i,J_p,-fg_i,P_d,P_i,P_p,perm_r,perm_c,
                                     Dxy_i_0,iparams,damp=ldamp,max_it=lmax_it,tol=ltol)
            if lsolver == 2:
                Dxy_i = spconjgradm(J_d,J_i,J_p,-fg_i,P_d,P_i,P_p,perm_r,perm_c,
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
                
        h_eval(z_ptr,x_ptr,y_ptr,u_ptr,p_ptr,Dt)
        xy[:N_x] = x
        xy[N_x:] = y
        
        # store in channels 
        if store == 1:
            if it >= it_store*decimation: 
                T[it_store+1] = t 
                X[it_store+1,:] = x 
                Y[it_store+1,:] = y
                Z[it_store+1,:] = z
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

    sp_jac_run_num_eval( sp_jac_run_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
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

    #de_jac_ini_num_eval(jac_ini_ss_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,1.0)
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

@numba.njit("(float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
def sp_Fu_run_eval(jac,x,y,u,p,Dt):   
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
    
    jac_ptr=ffi.from_buffer(np.ascontiguousarray(jac))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    sp_Fu_run_up_eval( jac_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_Fu_run_xy_eval( jac_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    #return jac

@numba.njit("(float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
def sp_Gu_run_eval(jac,x,y,u,p,Dt):   
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
    
    jac_ptr=ffi.from_buffer(np.ascontiguousarray(jac))
    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    sp_Gu_run_up_eval( jac_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_Gu_run_xy_eval( jac_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    
    #return jac

@numba.njit("(float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64[:],float64)")
def sp_H_jacs_run_eval(H_x,H_y,H_u,x,y,u,p,Dt):   
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
    
    H_x_ptr=ffi.from_buffer(np.ascontiguousarray(H_x))
    H_y_ptr=ffi.from_buffer(np.ascontiguousarray(H_y))
    H_u_ptr=ffi.from_buffer(np.ascontiguousarray(H_u))

    x_c_ptr=ffi.from_buffer(np.ascontiguousarray(x))
    y_c_ptr=ffi.from_buffer(np.ascontiguousarray(y))
    u_c_ptr=ffi.from_buffer(np.ascontiguousarray(u))
    p_c_ptr=ffi.from_buffer(np.ascontiguousarray(p))

    sp_Hx_run_up_eval( H_x_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_Hx_run_xy_eval( H_x_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_Hy_run_up_eval( H_y_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_Hy_run_xy_eval( H_y_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_Hu_run_up_eval( H_u_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)
    sp_Hu_run_xy_eval( H_u_ptr,x_c_ptr,y_c_ptr,u_c_ptr,p_c_ptr,Dt)





def sp_jac_ini_vectors():

    sp_jac_ini_ia = [0, 79, 0, 2, 81, 124, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 13, 14, 15, 15, 16, 17, 17, 18, 19, 19, 20, 21, 21, 22, 23, 23, 24, 25, 25, 26, 27, 27, 28, 29, 124, 30, 31, 32, 33, 36, 37, 40, 41, 52, 53, 30, 31, 32, 33, 36, 37, 40, 41, 52, 53, 30, 31, 32, 33, 34, 35, 30, 31, 32, 33, 34, 35, 2, 4, 32, 33, 34, 35, 2, 4, 32, 33, 34, 35, 0, 1, 30, 31, 36, 37, 30, 31, 36, 37, 38, 39, 40, 41, 87, 38, 39, 40, 41, 88, 30, 31, 38, 39, 40, 41, 44, 45, 30, 31, 38, 39, 40, 41, 44, 45, 42, 43, 44, 45, 94, 42, 43, 44, 45, 95, 40, 41, 42, 43, 44, 45, 48, 49, 40, 41, 42, 43, 44, 45, 48, 49, 46, 47, 48, 49, 101, 46, 47, 48, 49, 102, 44, 45, 46, 47, 48, 49, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 108, 50, 51, 52, 53, 109, 30, 31, 50, 51, 52, 53, 56, 57, 30, 31, 50, 51, 52, 53, 56, 57, 54, 55, 56, 57, 115, 54, 55, 56, 57, 116, 52, 53, 54, 55, 56, 57, 60, 61, 52, 53, 54, 55, 56, 57, 60, 61, 58, 59, 60, 61, 122, 58, 59, 60, 61, 123, 56, 57, 58, 59, 60, 61, 56, 57, 58, 59, 60, 61, 32, 33, 34, 35, 62, 32, 33, 34, 35, 63, 32, 33, 34, 35, 64, 32, 33, 34, 35, 65, 30, 31, 36, 37, 66, 30, 31, 36, 37, 67, 30, 31, 36, 37, 68, 30, 31, 36, 37, 69, 30, 31, 40, 41, 70, 30, 31, 40, 41, 71, 30, 31, 40, 41, 72, 30, 31, 40, 41, 73, 30, 31, 52, 53, 74, 30, 31, 52, 53, 75, 30, 31, 52, 53, 76, 30, 31, 52, 53, 77, 0, 1, 36, 78, 78, 79, 80, 0, 79, 80, 3, 81, 82, 87, 8, 38, 84, 6, 38, 83, 38, 39, 83, 84, 85, 86, 38, 39, 83, 84, 85, 86, 38, 39, 85, 86, 87, 38, 39, 85, 86, 88, 89, 94, 12, 42, 91, 10, 42, 90, 42, 43, 90, 91, 92, 93, 42, 43, 90, 91, 92, 93, 42, 43, 92, 93, 94, 42, 43, 92, 93, 95, 96, 101, 16, 46, 98, 14, 46, 97, 46, 47, 97, 98, 99, 100, 46, 47, 97, 98, 99, 100, 46, 47, 99, 100, 101, 46, 47, 99, 100, 102, 103, 108, 20, 50, 105, 18, 50, 104, 50, 51, 104, 105, 106, 107, 50, 51, 104, 105, 106, 107, 50, 51, 106, 107, 108, 50, 51, 106, 107, 109, 110, 115, 24, 54, 112, 22, 54, 111, 54, 55, 111, 112, 113, 114, 54, 55, 111, 112, 113, 114, 54, 55, 113, 114, 115, 54, 55, 113, 114, 116, 117, 122, 28, 58, 119, 26, 58, 118, 58, 59, 118, 119, 120, 121, 58, 59, 118, 119, 120, 121, 58, 59, 120, 121, 122, 58, 59, 120, 121, 123, 81, 124, 29, 124, 125]
    sp_jac_ini_ja = [0, 2, 3, 6, 7, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29, 30, 32, 33, 35, 36, 38, 39, 41, 42, 44, 46, 56, 66, 72, 78, 84, 90, 96, 100, 105, 110, 118, 126, 131, 136, 144, 152, 157, 162, 168, 174, 179, 184, 192, 200, 205, 210, 218, 226, 231, 236, 242, 248, 253, 258, 263, 268, 273, 278, 283, 288, 293, 298, 303, 308, 313, 318, 323, 328, 332, 335, 338, 340, 342, 345, 348, 354, 360, 365, 370, 372, 375, 378, 384, 390, 395, 400, 402, 405, 408, 414, 420, 425, 430, 432, 435, 438, 444, 450, 455, 460, 462, 465, 468, 474, 480, 485, 490, 492, 495, 498, 504, 510, 515, 520, 522, 525]
    sp_jac_ini_nia = 126
    sp_jac_ini_nja = 126
    return sp_jac_ini_ia, sp_jac_ini_ja, sp_jac_ini_nia, sp_jac_ini_nja 

def sp_jac_run_vectors():

    sp_jac_run_ia = [0, 79, 0, 2, 81, 124, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 13, 14, 15, 15, 16, 17, 17, 18, 19, 19, 20, 21, 21, 22, 23, 23, 24, 25, 25, 26, 27, 27, 28, 29, 124, 30, 31, 32, 33, 36, 37, 40, 41, 52, 53, 30, 31, 32, 33, 36, 37, 40, 41, 52, 53, 30, 31, 32, 33, 34, 35, 30, 31, 32, 33, 34, 35, 2, 4, 32, 33, 34, 35, 2, 4, 32, 33, 34, 35, 0, 1, 30, 31, 36, 37, 30, 31, 36, 37, 38, 39, 40, 41, 87, 38, 39, 40, 41, 88, 30, 31, 38, 39, 40, 41, 44, 45, 30, 31, 38, 39, 40, 41, 44, 45, 42, 43, 44, 45, 94, 42, 43, 44, 45, 95, 40, 41, 42, 43, 44, 45, 48, 49, 40, 41, 42, 43, 44, 45, 48, 49, 46, 47, 48, 49, 101, 46, 47, 48, 49, 102, 44, 45, 46, 47, 48, 49, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 108, 50, 51, 52, 53, 109, 30, 31, 50, 51, 52, 53, 56, 57, 30, 31, 50, 51, 52, 53, 56, 57, 54, 55, 56, 57, 115, 54, 55, 56, 57, 116, 52, 53, 54, 55, 56, 57, 60, 61, 52, 53, 54, 55, 56, 57, 60, 61, 58, 59, 60, 61, 122, 58, 59, 60, 61, 123, 56, 57, 58, 59, 60, 61, 56, 57, 58, 59, 60, 61, 32, 33, 34, 35, 62, 32, 33, 34, 35, 63, 32, 33, 34, 35, 64, 32, 33, 34, 35, 65, 30, 31, 36, 37, 66, 30, 31, 36, 37, 67, 30, 31, 36, 37, 68, 30, 31, 36, 37, 69, 30, 31, 40, 41, 70, 30, 31, 40, 41, 71, 30, 31, 40, 41, 72, 30, 31, 40, 41, 73, 30, 31, 52, 53, 74, 30, 31, 52, 53, 75, 30, 31, 52, 53, 76, 30, 31, 52, 53, 77, 0, 1, 36, 78, 78, 79, 80, 0, 79, 80, 3, 81, 82, 87, 8, 38, 84, 6, 38, 83, 38, 39, 83, 84, 85, 86, 38, 39, 83, 84, 85, 86, 38, 39, 85, 86, 87, 38, 39, 85, 86, 88, 89, 94, 12, 42, 91, 10, 42, 90, 42, 43, 90, 91, 92, 93, 42, 43, 90, 91, 92, 93, 42, 43, 92, 93, 94, 42, 43, 92, 93, 95, 96, 101, 16, 46, 98, 14, 46, 97, 46, 47, 97, 98, 99, 100, 46, 47, 97, 98, 99, 100, 46, 47, 99, 100, 101, 46, 47, 99, 100, 102, 103, 108, 20, 50, 105, 18, 50, 104, 50, 51, 104, 105, 106, 107, 50, 51, 104, 105, 106, 107, 50, 51, 106, 107, 108, 50, 51, 106, 107, 109, 110, 115, 24, 54, 112, 22, 54, 111, 54, 55, 111, 112, 113, 114, 54, 55, 111, 112, 113, 114, 54, 55, 113, 114, 115, 54, 55, 113, 114, 116, 117, 122, 28, 58, 119, 26, 58, 118, 58, 59, 118, 119, 120, 121, 58, 59, 118, 119, 120, 121, 58, 59, 120, 121, 122, 58, 59, 120, 121, 123, 81, 124, 29, 124, 125]
    sp_jac_run_ja = [0, 2, 3, 6, 7, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29, 30, 32, 33, 35, 36, 38, 39, 41, 42, 44, 46, 56, 66, 72, 78, 84, 90, 96, 100, 105, 110, 118, 126, 131, 136, 144, 152, 157, 162, 168, 174, 179, 184, 192, 200, 205, 210, 218, 226, 231, 236, 242, 248, 253, 258, 263, 268, 273, 278, 283, 288, 293, 298, 303, 308, 313, 318, 323, 328, 332, 335, 338, 340, 342, 345, 348, 354, 360, 365, 370, 372, 375, 378, 384, 390, 395, 400, 402, 405, 408, 414, 420, 425, 430, 432, 435, 438, 444, 450, 455, 460, 462, 465, 468, 474, 480, 485, 490, 492, 495, 498, 504, 510, 515, 520, 522, 525]
    sp_jac_run_nia = 126
    sp_jac_run_nja = 126
    return sp_jac_run_ia, sp_jac_run_ja, sp_jac_run_nia, sp_jac_run_nja 

def sp_jac_trap_vectors():

    sp_jac_trap_ia = [0, 79, 0, 1, 2, 81, 124, 3, 4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12, 13, 13, 14, 15, 15, 16, 17, 17, 18, 19, 19, 20, 21, 21, 22, 23, 23, 24, 25, 25, 26, 27, 27, 28, 29, 124, 30, 31, 32, 33, 36, 37, 40, 41, 52, 53, 30, 31, 32, 33, 36, 37, 40, 41, 52, 53, 30, 31, 32, 33, 34, 35, 30, 31, 32, 33, 34, 35, 2, 4, 32, 33, 34, 35, 2, 4, 32, 33, 34, 35, 0, 1, 30, 31, 36, 37, 30, 31, 36, 37, 38, 39, 40, 41, 87, 38, 39, 40, 41, 88, 30, 31, 38, 39, 40, 41, 44, 45, 30, 31, 38, 39, 40, 41, 44, 45, 42, 43, 44, 45, 94, 42, 43, 44, 45, 95, 40, 41, 42, 43, 44, 45, 48, 49, 40, 41, 42, 43, 44, 45, 48, 49, 46, 47, 48, 49, 101, 46, 47, 48, 49, 102, 44, 45, 46, 47, 48, 49, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 108, 50, 51, 52, 53, 109, 30, 31, 50, 51, 52, 53, 56, 57, 30, 31, 50, 51, 52, 53, 56, 57, 54, 55, 56, 57, 115, 54, 55, 56, 57, 116, 52, 53, 54, 55, 56, 57, 60, 61, 52, 53, 54, 55, 56, 57, 60, 61, 58, 59, 60, 61, 122, 58, 59, 60, 61, 123, 56, 57, 58, 59, 60, 61, 56, 57, 58, 59, 60, 61, 32, 33, 34, 35, 62, 32, 33, 34, 35, 63, 32, 33, 34, 35, 64, 32, 33, 34, 35, 65, 30, 31, 36, 37, 66, 30, 31, 36, 37, 67, 30, 31, 36, 37, 68, 30, 31, 36, 37, 69, 30, 31, 40, 41, 70, 30, 31, 40, 41, 71, 30, 31, 40, 41, 72, 30, 31, 40, 41, 73, 30, 31, 52, 53, 74, 30, 31, 52, 53, 75, 30, 31, 52, 53, 76, 30, 31, 52, 53, 77, 0, 1, 36, 78, 78, 79, 80, 0, 79, 80, 3, 81, 82, 87, 8, 38, 84, 6, 38, 83, 38, 39, 83, 84, 85, 86, 38, 39, 83, 84, 85, 86, 38, 39, 85, 86, 87, 38, 39, 85, 86, 88, 89, 94, 12, 42, 91, 10, 42, 90, 42, 43, 90, 91, 92, 93, 42, 43, 90, 91, 92, 93, 42, 43, 92, 93, 94, 42, 43, 92, 93, 95, 96, 101, 16, 46, 98, 14, 46, 97, 46, 47, 97, 98, 99, 100, 46, 47, 97, 98, 99, 100, 46, 47, 99, 100, 101, 46, 47, 99, 100, 102, 103, 108, 20, 50, 105, 18, 50, 104, 50, 51, 104, 105, 106, 107, 50, 51, 104, 105, 106, 107, 50, 51, 106, 107, 108, 50, 51, 106, 107, 109, 110, 115, 24, 54, 112, 22, 54, 111, 54, 55, 111, 112, 113, 114, 54, 55, 111, 112, 113, 114, 54, 55, 113, 114, 115, 54, 55, 113, 114, 116, 117, 122, 28, 58, 119, 26, 58, 118, 58, 59, 118, 119, 120, 121, 58, 59, 118, 119, 120, 121, 58, 59, 120, 121, 122, 58, 59, 120, 121, 123, 81, 124, 29, 124, 125]
    sp_jac_trap_ja = [0, 2, 4, 7, 8, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25, 27, 28, 30, 31, 33, 34, 36, 37, 39, 40, 42, 43, 45, 47, 57, 67, 73, 79, 85, 91, 97, 101, 106, 111, 119, 127, 132, 137, 145, 153, 158, 163, 169, 175, 180, 185, 193, 201, 206, 211, 219, 227, 232, 237, 243, 249, 254, 259, 264, 269, 274, 279, 284, 289, 294, 299, 304, 309, 314, 319, 324, 329, 333, 336, 339, 341, 343, 346, 349, 355, 361, 366, 371, 373, 376, 379, 385, 391, 396, 401, 403, 406, 409, 415, 421, 426, 431, 433, 436, 439, 445, 451, 456, 461, 463, 466, 469, 475, 481, 486, 491, 493, 496, 499, 505, 511, 516, 521, 523, 526]
    sp_jac_trap_nia = 126
    sp_jac_trap_nja = 126
    return sp_jac_trap_ia, sp_jac_trap_ja, sp_jac_trap_nia, sp_jac_trap_nja 
