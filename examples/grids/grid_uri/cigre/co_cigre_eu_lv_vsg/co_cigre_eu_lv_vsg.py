import numpy as np
import numba
import scipy.optimize as sopt
import scipy.sparse as sspa
from scipy.sparse.linalg import spsolve,spilu,splu
from numba import cuda
import cffi
import numba.core.typing.cffi_utils as cffi_support

ffi = cffi.FFI()

import co_cigre_eu_lv_vsg_cffi as jacs

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


class co_cigre_eu_lv_vsg_class: 

    def __init__(self): 

        self.t_end = 10.000000 
        self.Dt = 0.0010000 
        self.decimation = 10.000000 
        self.itol = 1e-6 
        self.Dt_max = 0.001000 
        self.Dt_min = 0.001000 
        self.solvern = 5 
        self.imax = 100 
        self.N_x = 15
        self.N_y = 549 
        self.N_z = 515 
        self.N_store = 10000 
        self.params_list = ['X_R02_sa', 'R_R02_sa', 'X_R02_sb', 'R_R02_sb', 'X_R02_sc', 'R_R02_sc', 'X_R02_sn', 'R_R02_sn', 'S_n_R02', 'X_R02_ng', 'R_R02_ng', 'K_f_R02', 'T_f_R02', 'K_sec_R02', 'K_delta_R02', 'X_R04_sa', 'R_R04_sa', 'X_R04_sb', 'R_R04_sb', 'X_R04_sc', 'R_R04_sc', 'X_R04_sn', 'R_R04_sn', 'S_n_R04', 'X_R04_ng', 'R_R04_ng', 'K_f_R04', 'T_f_R04', 'K_sec_R04', 'K_delta_R04', 'X_R08_sa', 'R_R08_sa', 'X_R08_sb', 'R_R08_sb', 'X_R08_sc', 'R_R08_sc', 'X_R08_sn', 'R_R08_sn', 'S_n_R08', 'X_R08_ng', 'R_R08_ng', 'K_f_R08', 'T_f_R08', 'K_sec_R08', 'K_delta_R08', 'X_I01_sa', 'R_I01_sa', 'X_I01_sb', 'R_I01_sb', 'X_I01_sc', 'R_I01_sc', 'X_I01_sn', 'R_I01_sn', 'S_n_I01', 'X_I01_ng', 'R_I01_ng', 'K_f_I01', 'T_f_I01', 'K_sec_I01', 'K_delta_I01', 'X_C09_sa', 'R_C09_sa', 'X_C09_sb', 'R_C09_sb', 'X_C09_sc', 'R_C09_sc', 'X_C09_sn', 'R_C09_sn', 'S_n_C09', 'X_C09_ng', 'R_C09_ng', 'K_f_C09', 'T_f_C09', 'K_sec_C09', 'K_delta_C09', 'X_C11_sa', 'R_C11_sa', 'X_C11_sb', 'R_C11_sb', 'X_C11_sc', 'R_C11_sc', 'X_C11_sn', 'R_C11_sn', 'S_n_C11', 'X_C11_ng', 'R_C11_ng', 'K_f_C11', 'T_f_C11', 'K_sec_C11', 'K_delta_C11', 'X_C02_sa', 'R_C02_sa', 'X_C02_sb', 'R_C02_sb', 'X_C02_sc', 'R_C02_sc', 'X_C02_sn', 'R_C02_sn', 'S_n_C02', 'X_C02_ng', 'R_C02_ng', 'K_f_C02', 'T_f_C02', 'K_sec_C02', 'K_delta_C02', 'K_agc'] 
        self.params_values_list  = [0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 2000000.0, 0.1, 0.01, 0.1, 1.0, 0.5, 0.001, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 100000.0, 0.1, 0.01, 0.1, 1.0, 0.5, 0.0, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 100000.0, 0.1, 0.01, 0.1, 1.0, 0.5, 0.0, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 100000.0, 0.1, 0.01, 0.1, 1.0, 0.5, 0.0, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 100000.0, 0.1, 0.01, 0.1, 1.0, 0.5, 0.0, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 100000.0, 0.1, 0.01, 0.1, 1.0, 0.5, 0.0, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 0.1, 0.01, 2000000.0, 0.1, 0.01, 0.1, 1.0, 0.5, 0.0, 0.001] 
        self.inputs_ini_list = ['i_R01_n_r', 'i_R01_n_i', 'i_R11_n_r', 'i_R11_n_i', 'i_R15_n_r', 'i_R15_n_i', 'i_R16_n_r', 'i_R16_n_i', 'i_R17_n_r', 'i_R17_n_i', 'i_R18_n_r', 'i_R18_n_i', 'i_I02_n_r', 'i_I02_n_i', 'i_C01_n_r', 'i_C01_n_i', 'i_C12_n_r', 'i_C12_n_i', 'i_C13_n_r', 'i_C13_n_i', 'i_C14_n_r', 'i_C14_n_i', 'i_C17_n_r', 'i_C17_n_i', 'i_C18_n_r', 'i_C18_n_i', 'i_C19_n_r', 'i_C19_n_i', 'i_C20_n_r', 'i_C20_n_i', 'i_MV0_a_r', 'i_MV0_a_i', 'i_MV0_b_r', 'i_MV0_b_i', 'i_MV0_c_r', 'i_MV0_c_i', 'i_R03_a_r', 'i_R03_a_i', 'i_R03_b_r', 'i_R03_b_i', 'i_R03_c_r', 'i_R03_c_i', 'i_R03_n_r', 'i_R03_n_i', 'i_R05_a_r', 'i_R05_a_i', 'i_R05_b_r', 'i_R05_b_i', 'i_R05_c_r', 'i_R05_c_i', 'i_R05_n_r', 'i_R05_n_i', 'i_R06_a_r', 'i_R06_a_i', 'i_R06_b_r', 'i_R06_b_i', 'i_R06_c_r', 'i_R06_c_i', 'i_R06_n_r', 'i_R06_n_i', 'i_R07_a_r', 'i_R07_a_i', 'i_R07_b_r', 'i_R07_b_i', 'i_R07_c_r', 'i_R07_c_i', 'i_R07_n_r', 'i_R07_n_i', 'i_R09_a_r', 'i_R09_a_i', 'i_R09_b_r', 'i_R09_b_i', 'i_R09_c_r', 'i_R09_c_i', 'i_R09_n_r', 'i_R09_n_i', 'i_R10_a_r', 'i_R10_a_i', 'i_R10_b_r', 'i_R10_b_i', 'i_R10_c_r', 'i_R10_c_i', 'i_R10_n_r', 'i_R10_n_i', 'i_R12_a_r', 'i_R12_a_i', 'i_R12_b_r', 'i_R12_b_i', 'i_R12_c_r', 'i_R12_c_i', 'i_R12_n_r', 'i_R12_n_i', 'i_R13_a_r', 'i_R13_a_i', 'i_R13_b_r', 'i_R13_b_i', 'i_R13_c_r', 'i_R13_c_i', 'i_R13_n_r', 'i_R13_n_i', 'i_R14_a_r', 'i_R14_a_i', 'i_R14_b_r', 'i_R14_b_i', 'i_R14_c_r', 'i_R14_c_i', 'i_R14_n_r', 'i_R14_n_i', 'i_C03_a_r', 'i_C03_a_i', 'i_C03_b_r', 'i_C03_b_i', 'i_C03_c_r', 'i_C03_c_i', 'i_C03_n_r', 'i_C03_n_i', 'i_C04_a_r', 'i_C04_a_i', 'i_C04_b_r', 'i_C04_b_i', 'i_C04_c_r', 'i_C04_c_i', 'i_C04_n_r', 'i_C04_n_i', 'i_C05_a_r', 'i_C05_a_i', 'i_C05_b_r', 'i_C05_b_i', 'i_C05_c_r', 'i_C05_c_i', 'i_C05_n_r', 'i_C05_n_i', 'i_C06_a_r', 'i_C06_a_i', 'i_C06_b_r', 'i_C06_b_i', 'i_C06_c_r', 'i_C06_c_i', 'i_C06_n_r', 'i_C06_n_i', 'i_C07_a_r', 'i_C07_a_i', 'i_C07_b_r', 'i_C07_b_i', 'i_C07_c_r', 'i_C07_c_i', 'i_C07_n_r', 'i_C07_n_i', 'i_C08_a_r', 'i_C08_a_i', 'i_C08_b_r', 'i_C08_b_i', 'i_C08_c_r', 'i_C08_c_i', 'i_C08_n_r', 'i_C08_n_i', 'i_C10_a_r', 'i_C10_a_i', 'i_C10_b_r', 'i_C10_b_i', 'i_C10_c_r', 'i_C10_c_i', 'i_C10_n_r', 'i_C10_n_i', 'i_C15_a_r', 'i_C15_a_i', 'i_C15_b_r', 'i_C15_b_i', 'i_C15_c_r', 'i_C15_c_i', 'i_C15_n_r', 'i_C15_n_i', 'i_C16_a_r', 'i_C16_a_i', 'i_C16_b_r', 'i_C16_b_i', 'i_C16_c_r', 'i_C16_c_i', 'i_C16_n_r', 'i_C16_n_i', 'p_R01_a', 'q_R01_a', 'p_R01_b', 'q_R01_b', 'p_R01_c', 'q_R01_c', 'p_R11_a', 'q_R11_a', 'p_R11_b', 'q_R11_b', 'p_R11_c', 'q_R11_c', 'p_R15_a', 'q_R15_a', 'p_R15_b', 'q_R15_b', 'p_R15_c', 'q_R15_c', 'p_R16_a', 'q_R16_a', 'p_R16_b', 'q_R16_b', 'p_R16_c', 'q_R16_c', 'p_R17_a', 'q_R17_a', 'p_R17_b', 'q_R17_b', 'p_R17_c', 'q_R17_c', 'p_R18_a', 'q_R18_a', 'p_R18_b', 'q_R18_b', 'p_R18_c', 'q_R18_c', 'p_I02_a', 'q_I02_a', 'p_I02_b', 'q_I02_b', 'p_I02_c', 'q_I02_c', 'p_C01_a', 'q_C01_a', 'p_C01_b', 'q_C01_b', 'p_C01_c', 'q_C01_c', 'p_C12_a', 'q_C12_a', 'p_C12_b', 'q_C12_b', 'p_C12_c', 'q_C12_c', 'p_C13_a', 'q_C13_a', 'p_C13_b', 'q_C13_b', 'p_C13_c', 'q_C13_c', 'p_C14_a', 'q_C14_a', 'p_C14_b', 'q_C14_b', 'p_C14_c', 'q_C14_c', 'p_C17_a', 'q_C17_a', 'p_C17_b', 'q_C17_b', 'p_C17_c', 'q_C17_c', 'p_C18_a', 'q_C18_a', 'p_C18_b', 'q_C18_b', 'p_C18_c', 'q_C18_c', 'p_C19_a', 'q_C19_a', 'p_C19_b', 'q_C19_b', 'p_C19_c', 'q_C19_c', 'p_C20_a', 'q_C20_a', 'p_C20_b', 'q_C20_b', 'p_C20_c', 'q_C20_c', 'e_R02_an', 'e_R02_bn', 'e_R02_cn', 'phi_R02_ref', 'p_R02_ref', 'omega_R02_ref', 'e_R04_an', 'e_R04_bn', 'e_R04_cn', 'phi_R04_ref', 'p_R04_ref', 'omega_R04_ref', 'e_R08_an', 'e_R08_bn', 'e_R08_cn', 'phi_R08_ref', 'p_R08_ref', 'omega_R08_ref', 'e_I01_an', 'e_I01_bn', 'e_I01_cn', 'phi_I01_ref', 'p_I01_ref', 'omega_I01_ref', 'e_C09_an', 'e_C09_bn', 'e_C09_cn', 'phi_C09_ref', 'p_C09_ref', 'omega_C09_ref', 'e_C11_an', 'e_C11_bn', 'e_C11_cn', 'phi_C11_ref', 'p_C11_ref', 'omega_C11_ref', 'e_C02_an', 'e_C02_bn', 'e_C02_cn', 'phi_C02_ref', 'p_C02_ref', 'omega_C02_ref'] 
        self.inputs_ini_values_list  = [9.934113187615168e-08, 6.465662972104935e-07, -5.285855404421613e-09, 2.5767266842793846e-08, -5.8188276391701876e-08, 1.8186447851148424e-08, -7.404549642875402e-08, -3.0507181850158993e-09, -6.320061627859697e-08, -3.0626162977193117e-08, -9.027333735395215e-08, -5.076831679010074e-08, 9.596957117885552e-09, 9.78681335798981e-08, 1.1069390437601356e-07, 3.7656393244572817e-07, -6.333417922055329e-09, 1.8469232638054778e-08, -6.333417918585882e-09, 1.8469232638054778e-08, -4.978331986232876e-09, 2.833741997304573e-08, -1.3947656710144862e-08, 1.2282061137222633e-08, -3.3346274526391273e-09, 5.946397496799094e-09, -9.1898335902868e-09, 7.378697302468096e-09, -4.594041412044059e-09, 3.6891198516159074e-09, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 36.14016873188073, -2.7159230698219754, 36.14016873188073, -2.7159230698219705, 36.14016873188073, -2.715923069821976, 2.710512654891052, -0.20369423023664768, 2.7105126548910548, -0.20369423023664734, 2.710512654891054, -0.203694230236648, 9.396443870288987, -0.7061399981537125, 9.396443870288989, -0.7061399981537111, 9.396443870288987, -0.7061399981537093, 9.938546401267203, -0.7468788442010429, 9.938546401267198, -0.7468788442010417, 9.938546401267198, -0.7468788442010421, 6.324529528079128, -0.4752865372188446, 6.324529528079128, -0.4752865372188442, 6.324529528079127, -0.47528653721884684, 8.492939651991973, -0.6382419214081639, 8.492939651991968, -0.6382419214081634, 8.492939651991968, -0.638241921408166, 17.883065543856805, 2.927110397067646, 17.883065543856805, 2.9271103970676435, 17.88306554385681, 2.9271103970676418, 21.708100858071823, 1.270452442313968, 21.708100858071823, 1.270452442313971, 21.708100858071816, 1.270452442313971, 3.6180168096786365, 0.21174207371899478, 3.618016809678635, 0.211742073718995, 3.618016809678637, 0.21174207371899478, 3.6180168096786365, 0.2117420737189951, 3.6180168096786356, 0.21174207371899478, 3.6180168096786365, 0.211742073718995, 4.522521012098296, 0.2646775921487437, 4.522521012098298, 0.2646775921487443, 4.522521012098297, 0.26467759214874365, 4.522521012098296, 0.2646775921487429, 4.522521012098297, 0.26467759214874387, 4.522521012098295, 0.26467759214874276, 1.4472067238714543, 0.08469682948759792, 1.4472067238714548, 0.08469682948759794, 1.4472067238714545, 0.08469682948759771, 2.8944134477429087, 0.1693936589751959, 2.894413447742909, 0.16939365897519587, 2.8944134477429095, 0.16939365897519432, 1.4472067238714539, 0.08469682948759785, 1.4472067238714545, 0.08469682948759805, 1.447206723871454, 0.08469682948759805, 230.94010767585033, 230.94010767585033, 230.94010767585033, 0.0, 0.0, 1.0, 230.94010767585033, 230.94010767585033, 230.94010767585033, 0.0, 0.0, 1.0, 230.94010767585033, 230.94010767585033, 230.94010767585033, 0.0, 0.0, 1.0, 230.94010767585033, 230.94010767585033, 230.94010767585033, 0.0, 0.0, 1.0, 230.94010767585033, 230.94010767585033, 230.94010767585033, 0.0, 0.0, 1.0, 230.94010767585033, 230.94010767585033, 230.94010767585033, 0.0, 0.0, 1.0, 230.94010767585033, 230.94010767585033, 230.94010767585033, 0.0, 0.0, 1.0] 
        self.inputs_run_list = ['i_R01_n_r', 'i_R01_n_i', 'i_R11_n_r', 'i_R11_n_i', 'i_R15_n_r', 'i_R15_n_i', 'i_R16_n_r', 'i_R16_n_i', 'i_R17_n_r', 'i_R17_n_i', 'i_R18_n_r', 'i_R18_n_i', 'i_I02_n_r', 'i_I02_n_i', 'i_C01_n_r', 'i_C01_n_i', 'i_C12_n_r', 'i_C12_n_i', 'i_C13_n_r', 'i_C13_n_i', 'i_C14_n_r', 'i_C14_n_i', 'i_C17_n_r', 'i_C17_n_i', 'i_C18_n_r', 'i_C18_n_i', 'i_C19_n_r', 'i_C19_n_i', 'i_C20_n_r', 'i_C20_n_i', 'i_MV0_a_r', 'i_MV0_a_i', 'i_MV0_b_r', 'i_MV0_b_i', 'i_MV0_c_r', 'i_MV0_c_i', 'i_R03_a_r', 'i_R03_a_i', 'i_R03_b_r', 'i_R03_b_i', 'i_R03_c_r', 'i_R03_c_i', 'i_R03_n_r', 'i_R03_n_i', 'i_R05_a_r', 'i_R05_a_i', 'i_R05_b_r', 'i_R05_b_i', 'i_R05_c_r', 'i_R05_c_i', 'i_R05_n_r', 'i_R05_n_i', 'i_R06_a_r', 'i_R06_a_i', 'i_R06_b_r', 'i_R06_b_i', 'i_R06_c_r', 'i_R06_c_i', 'i_R06_n_r', 'i_R06_n_i', 'i_R07_a_r', 'i_R07_a_i', 'i_R07_b_r', 'i_R07_b_i', 'i_R07_c_r', 'i_R07_c_i', 'i_R07_n_r', 'i_R07_n_i', 'i_R09_a_r', 'i_R09_a_i', 'i_R09_b_r', 'i_R09_b_i', 'i_R09_c_r', 'i_R09_c_i', 'i_R09_n_r', 'i_R09_n_i', 'i_R10_a_r', 'i_R10_a_i', 'i_R10_b_r', 'i_R10_b_i', 'i_R10_c_r', 'i_R10_c_i', 'i_R10_n_r', 'i_R10_n_i', 'i_R12_a_r', 'i_R12_a_i', 'i_R12_b_r', 'i_R12_b_i', 'i_R12_c_r', 'i_R12_c_i', 'i_R12_n_r', 'i_R12_n_i', 'i_R13_a_r', 'i_R13_a_i', 'i_R13_b_r', 'i_R13_b_i', 'i_R13_c_r', 'i_R13_c_i', 'i_R13_n_r', 'i_R13_n_i', 'i_R14_a_r', 'i_R14_a_i', 'i_R14_b_r', 'i_R14_b_i', 'i_R14_c_r', 'i_R14_c_i', 'i_R14_n_r', 'i_R14_n_i', 'i_C03_a_r', 'i_C03_a_i', 'i_C03_b_r', 'i_C03_b_i', 'i_C03_c_r', 'i_C03_c_i', 'i_C03_n_r', 'i_C03_n_i', 'i_C04_a_r', 'i_C04_a_i', 'i_C04_b_r', 'i_C04_b_i', 'i_C04_c_r', 'i_C04_c_i', 'i_C04_n_r', 'i_C04_n_i', 'i_C05_a_r', 'i_C05_a_i', 'i_C05_b_r', 'i_C05_b_i', 'i_C05_c_r', 'i_C05_c_i', 'i_C05_n_r', 'i_C05_n_i', 'i_C06_a_r', 'i_C06_a_i', 'i_C06_b_r', 'i_C06_b_i', 'i_C06_c_r', 'i_C06_c_i', 'i_C06_n_r', 'i_C06_n_i', 'i_C07_a_r', 'i_C07_a_i', 'i_C07_b_r', 'i_C07_b_i', 'i_C07_c_r', 'i_C07_c_i', 'i_C07_n_r', 'i_C07_n_i', 'i_C08_a_r', 'i_C08_a_i', 'i_C08_b_r', 'i_C08_b_i', 'i_C08_c_r', 'i_C08_c_i', 'i_C08_n_r', 'i_C08_n_i', 'i_C10_a_r', 'i_C10_a_i', 'i_C10_b_r', 'i_C10_b_i', 'i_C10_c_r', 'i_C10_c_i', 'i_C10_n_r', 'i_C10_n_i', 'i_C15_a_r', 'i_C15_a_i', 'i_C15_b_r', 'i_C15_b_i', 'i_C15_c_r', 'i_C15_c_i', 'i_C15_n_r', 'i_C15_n_i', 'i_C16_a_r', 'i_C16_a_i', 'i_C16_b_r', 'i_C16_b_i', 'i_C16_c_r', 'i_C16_c_i', 'i_C16_n_r', 'i_C16_n_i', 'p_R01_a', 'q_R01_a', 'p_R01_b', 'q_R01_b', 'p_R01_c', 'q_R01_c', 'p_R11_a', 'q_R11_a', 'p_R11_b', 'q_R11_b', 'p_R11_c', 'q_R11_c', 'p_R15_a', 'q_R15_a', 'p_R15_b', 'q_R15_b', 'p_R15_c', 'q_R15_c', 'p_R16_a', 'q_R16_a', 'p_R16_b', 'q_R16_b', 'p_R16_c', 'q_R16_c', 'p_R17_a', 'q_R17_a', 'p_R17_b', 'q_R17_b', 'p_R17_c', 'q_R17_c', 'p_R18_a', 'q_R18_a', 'p_R18_b', 'q_R18_b', 'p_R18_c', 'q_R18_c', 'p_I02_a', 'q_I02_a', 'p_I02_b', 'q_I02_b', 'p_I02_c', 'q_I02_c', 'p_C01_a', 'q_C01_a', 'p_C01_b', 'q_C01_b', 'p_C01_c', 'q_C01_c', 'p_C12_a', 'q_C12_a', 'p_C12_b', 'q_C12_b', 'p_C12_c', 'q_C12_c', 'p_C13_a', 'q_C13_a', 'p_C13_b', 'q_C13_b', 'p_C13_c', 'q_C13_c', 'p_C14_a', 'q_C14_a', 'p_C14_b', 'q_C14_b', 'p_C14_c', 'q_C14_c', 'p_C17_a', 'q_C17_a', 'p_C17_b', 'q_C17_b', 'p_C17_c', 'q_C17_c', 'p_C18_a', 'q_C18_a', 'p_C18_b', 'q_C18_b', 'p_C18_c', 'q_C18_c', 'p_C19_a', 'q_C19_a', 'p_C19_b', 'q_C19_b', 'p_C19_c', 'q_C19_c', 'p_C20_a', 'q_C20_a', 'p_C20_b', 'q_C20_b', 'p_C20_c', 'q_C20_c', 'e_R02_an', 'e_R02_bn', 'e_R02_cn', 'phi_R02_ref', 'p_R02_ref', 'omega_R02_ref', 'e_R04_an', 'e_R04_bn', 'e_R04_cn', 'phi_R04_ref', 'p_R04_ref', 'omega_R04_ref', 'e_R08_an', 'e_R08_bn', 'e_R08_cn', 'phi_R08_ref', 'p_R08_ref', 'omega_R08_ref', 'e_I01_an', 'e_I01_bn', 'e_I01_cn', 'phi_I01_ref', 'p_I01_ref', 'omega_I01_ref', 'e_C09_an', 'e_C09_bn', 'e_C09_cn', 'phi_C09_ref', 'p_C09_ref', 'omega_C09_ref', 'e_C11_an', 'e_C11_bn', 'e_C11_cn', 'phi_C11_ref', 'p_C11_ref', 'omega_C11_ref', 'e_C02_an', 'e_C02_bn', 'e_C02_cn', 'phi_C02_ref', 'p_C02_ref', 'omega_C02_ref'] 
        self.inputs_run_values_list = [9.934113187615168e-08, 6.465662972104935e-07, -5.285855404421613e-09, 2.5767266842793846e-08, -5.8188276391701876e-08, 1.8186447851148424e-08, -7.404549642875402e-08, -3.0507181850158993e-09, -6.320061627859697e-08, -3.0626162977193117e-08, -9.027333735395215e-08, -5.076831679010074e-08, 9.596957117885552e-09, 9.78681335798981e-08, 1.1069390437601356e-07, 3.7656393244572817e-07, -6.333417922055329e-09, 1.8469232638054778e-08, -6.333417918585882e-09, 1.8469232638054778e-08, -4.978331986232876e-09, 2.833741997304573e-08, -1.3947656710144862e-08, 1.2282061137222633e-08, -3.3346274526391273e-09, 5.946397496799094e-09, -9.1898335902868e-09, 7.378697302468096e-09, -4.594041412044059e-09, 3.6891198516159074e-09, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 36.14016873188073, -2.7159230698219754, 36.14016873188073, -2.7159230698219705, 36.14016873188073, -2.715923069821976, 2.710512654891052, -0.20369423023664768, 2.7105126548910548, -0.20369423023664734, 2.710512654891054, -0.203694230236648, 9.396443870288987, -0.7061399981537125, 9.396443870288989, -0.7061399981537111, 9.396443870288987, -0.7061399981537093, 9.938546401267203, -0.7468788442010429, 9.938546401267198, -0.7468788442010417, 9.938546401267198, -0.7468788442010421, 6.324529528079128, -0.4752865372188446, 6.324529528079128, -0.4752865372188442, 6.324529528079127, -0.47528653721884684, 8.492939651991973, -0.6382419214081639, 8.492939651991968, -0.6382419214081634, 8.492939651991968, -0.638241921408166, 17.883065543856805, 2.927110397067646, 17.883065543856805, 2.9271103970676435, 17.88306554385681, 2.9271103970676418, 21.708100858071823, 1.270452442313968, 21.708100858071823, 1.270452442313971, 21.708100858071816, 1.270452442313971, 3.6180168096786365, 0.21174207371899478, 3.618016809678635, 0.211742073718995, 3.618016809678637, 0.21174207371899478, 3.6180168096786365, 0.2117420737189951, 3.6180168096786356, 0.21174207371899478, 3.6180168096786365, 0.211742073718995, 4.522521012098296, 0.2646775921487437, 4.522521012098298, 0.2646775921487443, 4.522521012098297, 0.26467759214874365, 4.522521012098296, 0.2646775921487429, 4.522521012098297, 0.26467759214874387, 4.522521012098295, 0.26467759214874276, 1.4472067238714543, 0.08469682948759792, 1.4472067238714548, 0.08469682948759794, 1.4472067238714545, 0.08469682948759771, 2.8944134477429087, 0.1693936589751959, 2.894413447742909, 0.16939365897519587, 2.8944134477429095, 0.16939365897519432, 1.4472067238714539, 0.08469682948759785, 1.4472067238714545, 0.08469682948759805, 1.447206723871454, 0.08469682948759805, 230.94010767585033, 230.94010767585033, 230.94010767585033, 0.0, 0.0, 1.0, 230.94010767585033, 230.94010767585033, 230.94010767585033, 0.0, 0.0, 1.0, 230.94010767585033, 230.94010767585033, 230.94010767585033, 0.0, 0.0, 1.0, 230.94010767585033, 230.94010767585033, 230.94010767585033, 0.0, 0.0, 1.0, 230.94010767585033, 230.94010767585033, 230.94010767585033, 0.0, 0.0, 1.0, 230.94010767585033, 230.94010767585033, 230.94010767585033, 0.0, 0.0, 1.0, 230.94010767585033, 230.94010767585033, 230.94010767585033, 0.0, 0.0, 1.0] 
        self.outputs_list = ['i_l_R01_R02_a_r', 'i_l_R01_R02_a_i', 'i_l_R01_R02_b_r', 'i_l_R01_R02_b_i', 'i_l_R01_R02_c_r', 'i_l_R01_R02_c_i', 'i_l_R01_R02_n_r', 'i_l_R01_R02_n_i', 'i_l_R02_R03_a_r', 'i_l_R02_R03_a_i', 'i_l_R02_R03_b_r', 'i_l_R02_R03_b_i', 'i_l_R02_R03_c_r', 'i_l_R02_R03_c_i', 'i_l_R02_R03_n_r', 'i_l_R02_R03_n_i', 'i_l_R03_R04_a_r', 'i_l_R03_R04_a_i', 'i_l_R03_R04_b_r', 'i_l_R03_R04_b_i', 'i_l_R03_R04_c_r', 'i_l_R03_R04_c_i', 'i_l_R03_R04_n_r', 'i_l_R03_R04_n_i', 'i_l_R04_R05_a_r', 'i_l_R04_R05_a_i', 'i_l_R04_R05_b_r', 'i_l_R04_R05_b_i', 'i_l_R04_R05_c_r', 'i_l_R04_R05_c_i', 'i_l_R04_R05_n_r', 'i_l_R04_R05_n_i', 'i_l_R05_R06_a_r', 'i_l_R05_R06_a_i', 'i_l_R05_R06_b_r', 'i_l_R05_R06_b_i', 'i_l_R05_R06_c_r', 'i_l_R05_R06_c_i', 'i_l_R05_R06_n_r', 'i_l_R05_R06_n_i', 'i_l_R06_R07_a_r', 'i_l_R06_R07_a_i', 'i_l_R06_R07_b_r', 'i_l_R06_R07_b_i', 'i_l_R06_R07_c_r', 'i_l_R06_R07_c_i', 'i_l_R06_R07_n_r', 'i_l_R06_R07_n_i', 'i_l_R07_R08_a_r', 'i_l_R07_R08_a_i', 'i_l_R07_R08_b_r', 'i_l_R07_R08_b_i', 'i_l_R07_R08_c_r', 'i_l_R07_R08_c_i', 'i_l_R07_R08_n_r', 'i_l_R07_R08_n_i', 'i_l_R08_R09_a_r', 'i_l_R08_R09_a_i', 'i_l_R08_R09_b_r', 'i_l_R08_R09_b_i', 'i_l_R08_R09_c_r', 'i_l_R08_R09_c_i', 'i_l_R08_R09_n_r', 'i_l_R08_R09_n_i', 'i_l_R09_R10_a_r', 'i_l_R09_R10_a_i', 'i_l_R09_R10_b_r', 'i_l_R09_R10_b_i', 'i_l_R09_R10_c_r', 'i_l_R09_R10_c_i', 'i_l_R09_R10_n_r', 'i_l_R09_R10_n_i', 'i_l_R03_R11_a_r', 'i_l_R03_R11_a_i', 'i_l_R03_R11_b_r', 'i_l_R03_R11_b_i', 'i_l_R03_R11_c_r', 'i_l_R03_R11_c_i', 'i_l_R03_R11_n_r', 'i_l_R03_R11_n_i', 'i_l_R04_R12_a_r', 'i_l_R04_R12_a_i', 'i_l_R04_R12_b_r', 'i_l_R04_R12_b_i', 'i_l_R04_R12_c_r', 'i_l_R04_R12_c_i', 'i_l_R04_R12_n_r', 'i_l_R04_R12_n_i', 'i_l_R12_R13_a_r', 'i_l_R12_R13_a_i', 'i_l_R12_R13_b_r', 'i_l_R12_R13_b_i', 'i_l_R12_R13_c_r', 'i_l_R12_R13_c_i', 'i_l_R12_R13_n_r', 'i_l_R12_R13_n_i', 'i_l_R13_R14_a_r', 'i_l_R13_R14_a_i', 'i_l_R13_R14_b_r', 'i_l_R13_R14_b_i', 'i_l_R13_R14_c_r', 'i_l_R13_R14_c_i', 'i_l_R13_R14_n_r', 'i_l_R13_R14_n_i', 'i_l_R14_R15_a_r', 'i_l_R14_R15_a_i', 'i_l_R14_R15_b_r', 'i_l_R14_R15_b_i', 'i_l_R14_R15_c_r', 'i_l_R14_R15_c_i', 'i_l_R14_R15_n_r', 'i_l_R14_R15_n_i', 'i_l_R06_R16_a_r', 'i_l_R06_R16_a_i', 'i_l_R06_R16_b_r', 'i_l_R06_R16_b_i', 'i_l_R06_R16_c_r', 'i_l_R06_R16_c_i', 'i_l_R06_R16_n_r', 'i_l_R06_R16_n_i', 'i_l_R09_R17_a_r', 'i_l_R09_R17_a_i', 'i_l_R09_R17_b_r', 'i_l_R09_R17_b_i', 'i_l_R09_R17_c_r', 'i_l_R09_R17_c_i', 'i_l_R09_R17_n_r', 'i_l_R09_R17_n_i', 'i_l_R10_R18_a_r', 'i_l_R10_R18_a_i', 'i_l_R10_R18_b_r', 'i_l_R10_R18_b_i', 'i_l_R10_R18_c_r', 'i_l_R10_R18_c_i', 'i_l_R10_R18_n_r', 'i_l_R10_R18_n_i', 'i_l_I01_I02_a_r', 'i_l_I01_I02_a_i', 'i_l_I01_I02_b_r', 'i_l_I01_I02_b_i', 'i_l_I01_I02_c_r', 'i_l_I01_I02_c_i', 'i_l_I01_I02_n_r', 'i_l_I01_I02_n_i', 'i_l_C01_C02_a_r', 'i_l_C01_C02_a_i', 'i_l_C01_C02_b_r', 'i_l_C01_C02_b_i', 'i_l_C01_C02_c_r', 'i_l_C01_C02_c_i', 'i_l_C01_C02_n_r', 'i_l_C01_C02_n_i', 'i_l_C02_C03_a_r', 'i_l_C02_C03_a_i', 'i_l_C02_C03_b_r', 'i_l_C02_C03_b_i', 'i_l_C02_C03_c_r', 'i_l_C02_C03_c_i', 'i_l_C02_C03_n_r', 'i_l_C02_C03_n_i', 'i_l_C03_C04_a_r', 'i_l_C03_C04_a_i', 'i_l_C03_C04_b_r', 'i_l_C03_C04_b_i', 'i_l_C03_C04_c_r', 'i_l_C03_C04_c_i', 'i_l_C03_C04_n_r', 'i_l_C03_C04_n_i', 'i_l_C04_C05_a_r', 'i_l_C04_C05_a_i', 'i_l_C04_C05_b_r', 'i_l_C04_C05_b_i', 'i_l_C04_C05_c_r', 'i_l_C04_C05_c_i', 'i_l_C04_C05_n_r', 'i_l_C04_C05_n_i', 'i_l_C05_C06_a_r', 'i_l_C05_C06_a_i', 'i_l_C05_C06_b_r', 'i_l_C05_C06_b_i', 'i_l_C05_C06_c_r', 'i_l_C05_C06_c_i', 'i_l_C05_C06_n_r', 'i_l_C05_C06_n_i', 'i_l_C06_C07_a_r', 'i_l_C06_C07_a_i', 'i_l_C06_C07_b_r', 'i_l_C06_C07_b_i', 'i_l_C06_C07_c_r', 'i_l_C06_C07_c_i', 'i_l_C06_C07_n_r', 'i_l_C06_C07_n_i', 'i_l_C07_C08_a_r', 'i_l_C07_C08_a_i', 'i_l_C07_C08_b_r', 'i_l_C07_C08_b_i', 'i_l_C07_C08_c_r', 'i_l_C07_C08_c_i', 'i_l_C07_C08_n_r', 'i_l_C07_C08_n_i', 'i_l_C08_C09_a_r', 'i_l_C08_C09_a_i', 'i_l_C08_C09_b_r', 'i_l_C08_C09_b_i', 'i_l_C08_C09_c_r', 'i_l_C08_C09_c_i', 'i_l_C08_C09_n_r', 'i_l_C08_C09_n_i', 'i_l_C03_C10_a_r', 'i_l_C03_C10_a_i', 'i_l_C03_C10_b_r', 'i_l_C03_C10_b_i', 'i_l_C03_C10_c_r', 'i_l_C03_C10_c_i', 'i_l_C03_C10_n_r', 'i_l_C03_C10_n_i', 'i_l_C10_C11_a_r', 'i_l_C10_C11_a_i', 'i_l_C10_C11_b_r', 'i_l_C10_C11_b_i', 'i_l_C10_C11_c_r', 'i_l_C10_C11_c_i', 'i_l_C10_C11_n_r', 'i_l_C10_C11_n_i', 'i_l_C11_C12_a_r', 'i_l_C11_C12_a_i', 'i_l_C11_C12_b_r', 'i_l_C11_C12_b_i', 'i_l_C11_C12_c_r', 'i_l_C11_C12_c_i', 'i_l_C11_C12_n_r', 'i_l_C11_C12_n_i', 'i_l_C11_C13_a_r', 'i_l_C11_C13_a_i', 'i_l_C11_C13_b_r', 'i_l_C11_C13_b_i', 'i_l_C11_C13_c_r', 'i_l_C11_C13_c_i', 'i_l_C11_C13_n_r', 'i_l_C11_C13_n_i', 'i_l_C10_C14_a_r', 'i_l_C10_C14_a_i', 'i_l_C10_C14_b_r', 'i_l_C10_C14_b_i', 'i_l_C10_C14_c_r', 'i_l_C10_C14_c_i', 'i_l_C10_C14_n_r', 'i_l_C10_C14_n_i', 'i_l_C05_C15_a_r', 'i_l_C05_C15_a_i', 'i_l_C05_C15_b_r', 'i_l_C05_C15_b_i', 'i_l_C05_C15_c_r', 'i_l_C05_C15_c_i', 'i_l_C05_C15_n_r', 'i_l_C05_C15_n_i', 'i_l_C15_C16_a_r', 'i_l_C15_C16_a_i', 'i_l_C15_C16_b_r', 'i_l_C15_C16_b_i', 'i_l_C15_C16_c_r', 'i_l_C15_C16_c_i', 'i_l_C15_C16_n_r', 'i_l_C15_C16_n_i', 'i_l_C15_C18_a_r', 'i_l_C15_C18_a_i', 'i_l_C15_C18_b_r', 'i_l_C15_C18_b_i', 'i_l_C15_C18_c_r', 'i_l_C15_C18_c_i', 'i_l_C15_C18_n_r', 'i_l_C15_C18_n_i', 'i_l_C16_C17_a_r', 'i_l_C16_C17_a_i', 'i_l_C16_C17_b_r', 'i_l_C16_C17_b_i', 'i_l_C16_C17_c_r', 'i_l_C16_C17_c_i', 'i_l_C16_C17_n_r', 'i_l_C16_C17_n_i', 'i_l_C08_C19_a_r', 'i_l_C08_C19_a_i', 'i_l_C08_C19_b_r', 'i_l_C08_C19_b_i', 'i_l_C08_C19_c_r', 'i_l_C08_C19_c_i', 'i_l_C08_C19_n_r', 'i_l_C08_C19_n_i', 'i_l_C09_C20_a_r', 'i_l_C09_C20_a_i', 'i_l_C09_C20_b_r', 'i_l_C09_C20_b_i', 'i_l_C09_C20_c_r', 'i_l_C09_C20_c_i', 'i_l_C09_C20_n_r', 'i_l_C09_C20_n_i', 'v_R01_a_m', 'v_R01_b_m', 'v_R01_c_m', 'v_R01_n_m', 'v_R11_a_m', 'v_R11_b_m', 'v_R11_c_m', 'v_R11_n_m', 'v_R15_a_m', 'v_R15_b_m', 'v_R15_c_m', 'v_R15_n_m', 'v_R16_a_m', 'v_R16_b_m', 'v_R16_c_m', 'v_R16_n_m', 'v_R17_a_m', 'v_R17_b_m', 'v_R17_c_m', 'v_R17_n_m', 'v_R18_a_m', 'v_R18_b_m', 'v_R18_c_m', 'v_R18_n_m', 'v_I02_a_m', 'v_I02_b_m', 'v_I02_c_m', 'v_I02_n_m', 'v_C01_a_m', 'v_C01_b_m', 'v_C01_c_m', 'v_C01_n_m', 'v_C12_a_m', 'v_C12_b_m', 'v_C12_c_m', 'v_C12_n_m', 'v_C13_a_m', 'v_C13_b_m', 'v_C13_c_m', 'v_C13_n_m', 'v_C14_a_m', 'v_C14_b_m', 'v_C14_c_m', 'v_C14_n_m', 'v_C17_a_m', 'v_C17_b_m', 'v_C17_c_m', 'v_C17_n_m', 'v_C18_a_m', 'v_C18_b_m', 'v_C18_c_m', 'v_C18_n_m', 'v_C19_a_m', 'v_C19_b_m', 'v_C19_c_m', 'v_C19_n_m', 'v_C20_a_m', 'v_C20_b_m', 'v_C20_c_m', 'v_C20_n_m', 'v_MV0_a_m', 'v_MV0_b_m', 'v_MV0_c_m', 'v_I01_a_m', 'v_I01_b_m', 'v_I01_c_m', 'v_I01_n_m', 'v_R02_a_m', 'v_R02_b_m', 'v_R02_c_m', 'v_R02_n_m', 'v_R03_a_m', 'v_R03_b_m', 'v_R03_c_m', 'v_R03_n_m', 'v_R04_a_m', 'v_R04_b_m', 'v_R04_c_m', 'v_R04_n_m', 'v_R05_a_m', 'v_R05_b_m', 'v_R05_c_m', 'v_R05_n_m', 'v_R06_a_m', 'v_R06_b_m', 'v_R06_c_m', 'v_R06_n_m', 'v_R07_a_m', 'v_R07_b_m', 'v_R07_c_m', 'v_R07_n_m', 'v_R08_a_m', 'v_R08_b_m', 'v_R08_c_m', 'v_R08_n_m', 'v_R09_a_m', 'v_R09_b_m', 'v_R09_c_m', 'v_R09_n_m', 'v_R10_a_m', 'v_R10_b_m', 'v_R10_c_m', 'v_R10_n_m', 'v_R12_a_m', 'v_R12_b_m', 'v_R12_c_m', 'v_R12_n_m', 'v_R13_a_m', 'v_R13_b_m', 'v_R13_c_m', 'v_R13_n_m', 'v_R14_a_m', 'v_R14_b_m', 'v_R14_c_m', 'v_R14_n_m', 'v_C02_a_m', 'v_C02_b_m', 'v_C02_c_m', 'v_C02_n_m', 'v_C03_a_m', 'v_C03_b_m', 'v_C03_c_m', 'v_C03_n_m', 'v_C04_a_m', 'v_C04_b_m', 'v_C04_c_m', 'v_C04_n_m', 'v_C05_a_m', 'v_C05_b_m', 'v_C05_c_m', 'v_C05_n_m', 'v_C06_a_m', 'v_C06_b_m', 'v_C06_c_m', 'v_C06_n_m', 'v_C07_a_m', 'v_C07_b_m', 'v_C07_c_m', 'v_C07_n_m', 'v_C08_a_m', 'v_C08_b_m', 'v_C08_c_m', 'v_C08_n_m', 'v_C09_a_m', 'v_C09_b_m', 'v_C09_c_m', 'v_C09_n_m', 'v_C10_a_m', 'v_C10_b_m', 'v_C10_c_m', 'v_C10_n_m', 'v_C11_a_m', 'v_C11_b_m', 'v_C11_c_m', 'v_C11_n_m', 'v_C15_a_m', 'v_C15_b_m', 'v_C15_c_m', 'v_C15_n_m', 'v_C16_a_m', 'v_C16_b_m', 'v_C16_c_m', 'v_C16_n_m', 'p_R02_pos', 'p_R02_neg', 'p_R02_zer', 'e_R02_an', 'e_R02_bn', 'e_R02_cn', 'p_R02_ref', 'omega_R02_ref', 'p_R04_pos', 'p_R04_neg', 'p_R04_zer', 'e_R04_an', 'e_R04_bn', 'e_R04_cn', 'p_R04_ref', 'omega_R04_ref', 'p_R08_pos', 'p_R08_neg', 'p_R08_zer', 'e_R08_an', 'e_R08_bn', 'e_R08_cn', 'p_R08_ref', 'omega_R08_ref', 'p_I01_pos', 'p_I01_neg', 'p_I01_zer', 'e_I01_an', 'e_I01_bn', 'e_I01_cn', 'p_I01_ref', 'omega_I01_ref', 'p_C09_pos', 'p_C09_neg', 'p_C09_zer', 'e_C09_an', 'e_C09_bn', 'e_C09_cn', 'p_C09_ref', 'omega_C09_ref', 'p_C11_pos', 'p_C11_neg', 'p_C11_zer', 'e_C11_an', 'e_C11_bn', 'e_C11_cn', 'p_C11_ref', 'omega_C11_ref', 'p_C02_pos', 'p_C02_neg', 'p_C02_zer', 'e_C02_an', 'e_C02_bn', 'e_C02_cn', 'p_C02_ref', 'omega_C02_ref'] 
        self.x_list = ['phi_R02', 'omega_R02', 'phi_R04', 'omega_R04', 'phi_R08', 'omega_R08', 'phi_I01', 'omega_I01', 'phi_C09', 'omega_C09', 'phi_C11', 'omega_C11', 'phi_C02', 'omega_C02', 'xi_freq'] 
        self.y_run_list = ['v_R01_a_r', 'v_R01_a_i', 'v_R01_b_r', 'v_R01_b_i', 'v_R01_c_r', 'v_R01_c_i', 'v_R01_n_r', 'v_R01_n_i', 'v_R11_a_r', 'v_R11_a_i', 'v_R11_b_r', 'v_R11_b_i', 'v_R11_c_r', 'v_R11_c_i', 'v_R11_n_r', 'v_R11_n_i', 'v_R15_a_r', 'v_R15_a_i', 'v_R15_b_r', 'v_R15_b_i', 'v_R15_c_r', 'v_R15_c_i', 'v_R15_n_r', 'v_R15_n_i', 'v_R16_a_r', 'v_R16_a_i', 'v_R16_b_r', 'v_R16_b_i', 'v_R16_c_r', 'v_R16_c_i', 'v_R16_n_r', 'v_R16_n_i', 'v_R17_a_r', 'v_R17_a_i', 'v_R17_b_r', 'v_R17_b_i', 'v_R17_c_r', 'v_R17_c_i', 'v_R17_n_r', 'v_R17_n_i', 'v_R18_a_r', 'v_R18_a_i', 'v_R18_b_r', 'v_R18_b_i', 'v_R18_c_r', 'v_R18_c_i', 'v_R18_n_r', 'v_R18_n_i', 'v_I02_a_r', 'v_I02_a_i', 'v_I02_b_r', 'v_I02_b_i', 'v_I02_c_r', 'v_I02_c_i', 'v_I02_n_r', 'v_I02_n_i', 'v_C01_a_r', 'v_C01_a_i', 'v_C01_b_r', 'v_C01_b_i', 'v_C01_c_r', 'v_C01_c_i', 'v_C01_n_r', 'v_C01_n_i', 'v_C12_a_r', 'v_C12_a_i', 'v_C12_b_r', 'v_C12_b_i', 'v_C12_c_r', 'v_C12_c_i', 'v_C12_n_r', 'v_C12_n_i', 'v_C13_a_r', 'v_C13_a_i', 'v_C13_b_r', 'v_C13_b_i', 'v_C13_c_r', 'v_C13_c_i', 'v_C13_n_r', 'v_C13_n_i', 'v_C14_a_r', 'v_C14_a_i', 'v_C14_b_r', 'v_C14_b_i', 'v_C14_c_r', 'v_C14_c_i', 'v_C14_n_r', 'v_C14_n_i', 'v_C17_a_r', 'v_C17_a_i', 'v_C17_b_r', 'v_C17_b_i', 'v_C17_c_r', 'v_C17_c_i', 'v_C17_n_r', 'v_C17_n_i', 'v_C18_a_r', 'v_C18_a_i', 'v_C18_b_r', 'v_C18_b_i', 'v_C18_c_r', 'v_C18_c_i', 'v_C18_n_r', 'v_C18_n_i', 'v_C19_a_r', 'v_C19_a_i', 'v_C19_b_r', 'v_C19_b_i', 'v_C19_c_r', 'v_C19_c_i', 'v_C19_n_r', 'v_C19_n_i', 'v_C20_a_r', 'v_C20_a_i', 'v_C20_b_r', 'v_C20_b_i', 'v_C20_c_r', 'v_C20_c_i', 'v_C20_n_r', 'v_C20_n_i', 'v_MV0_a_r', 'v_MV0_a_i', 'v_MV0_b_r', 'v_MV0_b_i', 'v_MV0_c_r', 'v_MV0_c_i', 'v_I01_a_r', 'v_I01_a_i', 'v_I01_b_r', 'v_I01_b_i', 'v_I01_c_r', 'v_I01_c_i', 'v_I01_n_r', 'v_I01_n_i', 'v_R02_a_r', 'v_R02_a_i', 'v_R02_b_r', 'v_R02_b_i', 'v_R02_c_r', 'v_R02_c_i', 'v_R02_n_r', 'v_R02_n_i', 'v_R03_a_r', 'v_R03_a_i', 'v_R03_b_r', 'v_R03_b_i', 'v_R03_c_r', 'v_R03_c_i', 'v_R03_n_r', 'v_R03_n_i', 'v_R04_a_r', 'v_R04_a_i', 'v_R04_b_r', 'v_R04_b_i', 'v_R04_c_r', 'v_R04_c_i', 'v_R04_n_r', 'v_R04_n_i', 'v_R05_a_r', 'v_R05_a_i', 'v_R05_b_r', 'v_R05_b_i', 'v_R05_c_r', 'v_R05_c_i', 'v_R05_n_r', 'v_R05_n_i', 'v_R06_a_r', 'v_R06_a_i', 'v_R06_b_r', 'v_R06_b_i', 'v_R06_c_r', 'v_R06_c_i', 'v_R06_n_r', 'v_R06_n_i', 'v_R07_a_r', 'v_R07_a_i', 'v_R07_b_r', 'v_R07_b_i', 'v_R07_c_r', 'v_R07_c_i', 'v_R07_n_r', 'v_R07_n_i', 'v_R08_a_r', 'v_R08_a_i', 'v_R08_b_r', 'v_R08_b_i', 'v_R08_c_r', 'v_R08_c_i', 'v_R08_n_r', 'v_R08_n_i', 'v_R09_a_r', 'v_R09_a_i', 'v_R09_b_r', 'v_R09_b_i', 'v_R09_c_r', 'v_R09_c_i', 'v_R09_n_r', 'v_R09_n_i', 'v_R10_a_r', 'v_R10_a_i', 'v_R10_b_r', 'v_R10_b_i', 'v_R10_c_r', 'v_R10_c_i', 'v_R10_n_r', 'v_R10_n_i', 'v_R12_a_r', 'v_R12_a_i', 'v_R12_b_r', 'v_R12_b_i', 'v_R12_c_r', 'v_R12_c_i', 'v_R12_n_r', 'v_R12_n_i', 'v_R13_a_r', 'v_R13_a_i', 'v_R13_b_r', 'v_R13_b_i', 'v_R13_c_r', 'v_R13_c_i', 'v_R13_n_r', 'v_R13_n_i', 'v_R14_a_r', 'v_R14_a_i', 'v_R14_b_r', 'v_R14_b_i', 'v_R14_c_r', 'v_R14_c_i', 'v_R14_n_r', 'v_R14_n_i', 'v_C02_a_r', 'v_C02_a_i', 'v_C02_b_r', 'v_C02_b_i', 'v_C02_c_r', 'v_C02_c_i', 'v_C02_n_r', 'v_C02_n_i', 'v_C03_a_r', 'v_C03_a_i', 'v_C03_b_r', 'v_C03_b_i', 'v_C03_c_r', 'v_C03_c_i', 'v_C03_n_r', 'v_C03_n_i', 'v_C04_a_r', 'v_C04_a_i', 'v_C04_b_r', 'v_C04_b_i', 'v_C04_c_r', 'v_C04_c_i', 'v_C04_n_r', 'v_C04_n_i', 'v_C05_a_r', 'v_C05_a_i', 'v_C05_b_r', 'v_C05_b_i', 'v_C05_c_r', 'v_C05_c_i', 'v_C05_n_r', 'v_C05_n_i', 'v_C06_a_r', 'v_C06_a_i', 'v_C06_b_r', 'v_C06_b_i', 'v_C06_c_r', 'v_C06_c_i', 'v_C06_n_r', 'v_C06_n_i', 'v_C07_a_r', 'v_C07_a_i', 'v_C07_b_r', 'v_C07_b_i', 'v_C07_c_r', 'v_C07_c_i', 'v_C07_n_r', 'v_C07_n_i', 'v_C08_a_r', 'v_C08_a_i', 'v_C08_b_r', 'v_C08_b_i', 'v_C08_c_r', 'v_C08_c_i', 'v_C08_n_r', 'v_C08_n_i', 'v_C09_a_r', 'v_C09_a_i', 'v_C09_b_r', 'v_C09_b_i', 'v_C09_c_r', 'v_C09_c_i', 'v_C09_n_r', 'v_C09_n_i', 'v_C10_a_r', 'v_C10_a_i', 'v_C10_b_r', 'v_C10_b_i', 'v_C10_c_r', 'v_C10_c_i', 'v_C10_n_r', 'v_C10_n_i', 'v_C11_a_r', 'v_C11_a_i', 'v_C11_b_r', 'v_C11_b_i', 'v_C11_c_r', 'v_C11_c_i', 'v_C11_n_r', 'v_C11_n_i', 'v_C15_a_r', 'v_C15_a_i', 'v_C15_b_r', 'v_C15_b_i', 'v_C15_c_r', 'v_C15_c_i', 'v_C15_n_r', 'v_C15_n_i', 'v_C16_a_r', 'v_C16_a_i', 'v_C16_b_r', 'v_C16_b_i', 'v_C16_c_r', 'v_C16_c_i', 'v_C16_n_r', 'v_C16_n_i', 'i_t_MV0_R01_a_r', 'i_t_MV0_R01_a_i', 'i_t_MV0_R01_b_r', 'i_t_MV0_R01_b_i', 'i_t_MV0_R01_c_r', 'i_t_MV0_R01_c_i', 'i_t_MV0_I01_a_r', 'i_t_MV0_I01_a_i', 'i_t_MV0_I01_b_r', 'i_t_MV0_I01_b_i', 'i_t_MV0_I01_c_r', 'i_t_MV0_I01_c_i', 'i_t_MV0_C01_a_r', 'i_t_MV0_C01_a_i', 'i_t_MV0_C01_b_r', 'i_t_MV0_C01_b_i', 'i_t_MV0_C01_c_r', 'i_t_MV0_C01_c_i', 'i_load_R01_a_r', 'i_load_R01_a_i', 'i_load_R01_b_r', 'i_load_R01_b_i', 'i_load_R01_c_r', 'i_load_R01_c_i', 'i_load_R01_n_r', 'i_load_R01_n_i', 'i_load_R11_a_r', 'i_load_R11_a_i', 'i_load_R11_b_r', 'i_load_R11_b_i', 'i_load_R11_c_r', 'i_load_R11_c_i', 'i_load_R11_n_r', 'i_load_R11_n_i', 'i_load_R15_a_r', 'i_load_R15_a_i', 'i_load_R15_b_r', 'i_load_R15_b_i', 'i_load_R15_c_r', 'i_load_R15_c_i', 'i_load_R15_n_r', 'i_load_R15_n_i', 'i_load_R16_a_r', 'i_load_R16_a_i', 'i_load_R16_b_r', 'i_load_R16_b_i', 'i_load_R16_c_r', 'i_load_R16_c_i', 'i_load_R16_n_r', 'i_load_R16_n_i', 'i_load_R17_a_r', 'i_load_R17_a_i', 'i_load_R17_b_r', 'i_load_R17_b_i', 'i_load_R17_c_r', 'i_load_R17_c_i', 'i_load_R17_n_r', 'i_load_R17_n_i', 'i_load_R18_a_r', 'i_load_R18_a_i', 'i_load_R18_b_r', 'i_load_R18_b_i', 'i_load_R18_c_r', 'i_load_R18_c_i', 'i_load_R18_n_r', 'i_load_R18_n_i', 'i_load_I02_a_r', 'i_load_I02_a_i', 'i_load_I02_b_r', 'i_load_I02_b_i', 'i_load_I02_c_r', 'i_load_I02_c_i', 'i_load_I02_n_r', 'i_load_I02_n_i', 'i_load_C01_a_r', 'i_load_C01_a_i', 'i_load_C01_b_r', 'i_load_C01_b_i', 'i_load_C01_c_r', 'i_load_C01_c_i', 'i_load_C01_n_r', 'i_load_C01_n_i', 'i_load_C12_a_r', 'i_load_C12_a_i', 'i_load_C12_b_r', 'i_load_C12_b_i', 'i_load_C12_c_r', 'i_load_C12_c_i', 'i_load_C12_n_r', 'i_load_C12_n_i', 'i_load_C13_a_r', 'i_load_C13_a_i', 'i_load_C13_b_r', 'i_load_C13_b_i', 'i_load_C13_c_r', 'i_load_C13_c_i', 'i_load_C13_n_r', 'i_load_C13_n_i', 'i_load_C14_a_r', 'i_load_C14_a_i', 'i_load_C14_b_r', 'i_load_C14_b_i', 'i_load_C14_c_r', 'i_load_C14_c_i', 'i_load_C14_n_r', 'i_load_C14_n_i', 'i_load_C17_a_r', 'i_load_C17_a_i', 'i_load_C17_b_r', 'i_load_C17_b_i', 'i_load_C17_c_r', 'i_load_C17_c_i', 'i_load_C17_n_r', 'i_load_C17_n_i', 'i_load_C18_a_r', 'i_load_C18_a_i', 'i_load_C18_b_r', 'i_load_C18_b_i', 'i_load_C18_c_r', 'i_load_C18_c_i', 'i_load_C18_n_r', 'i_load_C18_n_i', 'i_load_C19_a_r', 'i_load_C19_a_i', 'i_load_C19_b_r', 'i_load_C19_b_i', 'i_load_C19_c_r', 'i_load_C19_c_i', 'i_load_C19_n_r', 'i_load_C19_n_i', 'i_load_C20_a_r', 'i_load_C20_a_i', 'i_load_C20_b_r', 'i_load_C20_b_i', 'i_load_C20_c_r', 'i_load_C20_c_i', 'i_load_C20_n_r', 'i_load_C20_n_i', 'i_R02_a_r', 'i_R02_b_r', 'i_R02_c_r', 'i_R02_n_r', 'i_R02_ng_r', 'e_R02_ng_r', 'i_R02_a_i', 'i_R02_b_i', 'i_R02_c_i', 'i_R02_n_i', 'i_R02_ng_i', 'e_R02_ng_i', 'i_R04_a_r', 'i_R04_b_r', 'i_R04_c_r', 'i_R04_n_r', 'i_R04_ng_r', 'e_R04_ng_r', 'i_R04_a_i', 'i_R04_b_i', 'i_R04_c_i', 'i_R04_n_i', 'i_R04_ng_i', 'e_R04_ng_i', 'i_R08_a_r', 'i_R08_b_r', 'i_R08_c_r', 'i_R08_n_r', 'i_R08_ng_r', 'e_R08_ng_r', 'i_R08_a_i', 'i_R08_b_i', 'i_R08_c_i', 'i_R08_n_i', 'i_R08_ng_i', 'e_R08_ng_i', 'i_I01_a_r', 'i_I01_b_r', 'i_I01_c_r', 'i_I01_n_r', 'i_I01_ng_r', 'e_I01_ng_r', 'i_I01_a_i', 'i_I01_b_i', 'i_I01_c_i', 'i_I01_n_i', 'i_I01_ng_i', 'e_I01_ng_i', 'i_C09_a_r', 'i_C09_b_r', 'i_C09_c_r', 'i_C09_n_r', 'i_C09_ng_r', 'e_C09_ng_r', 'i_C09_a_i', 'i_C09_b_i', 'i_C09_c_i', 'i_C09_n_i', 'i_C09_ng_i', 'e_C09_ng_i', 'i_C11_a_r', 'i_C11_b_r', 'i_C11_c_r', 'i_C11_n_r', 'i_C11_ng_r', 'e_C11_ng_r', 'i_C11_a_i', 'i_C11_b_i', 'i_C11_c_i', 'i_C11_n_i', 'i_C11_ng_i', 'e_C11_ng_i', 'i_C02_a_r', 'i_C02_b_r', 'i_C02_c_r', 'i_C02_n_r', 'i_C02_ng_r', 'e_C02_ng_r', 'i_C02_a_i', 'i_C02_b_i', 'i_C02_c_i', 'i_C02_n_i', 'i_C02_ng_i', 'e_C02_ng_i', 'omega_coi'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['v_R01_a_r', 'v_R01_a_i', 'v_R01_b_r', 'v_R01_b_i', 'v_R01_c_r', 'v_R01_c_i', 'v_R01_n_r', 'v_R01_n_i', 'v_R11_a_r', 'v_R11_a_i', 'v_R11_b_r', 'v_R11_b_i', 'v_R11_c_r', 'v_R11_c_i', 'v_R11_n_r', 'v_R11_n_i', 'v_R15_a_r', 'v_R15_a_i', 'v_R15_b_r', 'v_R15_b_i', 'v_R15_c_r', 'v_R15_c_i', 'v_R15_n_r', 'v_R15_n_i', 'v_R16_a_r', 'v_R16_a_i', 'v_R16_b_r', 'v_R16_b_i', 'v_R16_c_r', 'v_R16_c_i', 'v_R16_n_r', 'v_R16_n_i', 'v_R17_a_r', 'v_R17_a_i', 'v_R17_b_r', 'v_R17_b_i', 'v_R17_c_r', 'v_R17_c_i', 'v_R17_n_r', 'v_R17_n_i', 'v_R18_a_r', 'v_R18_a_i', 'v_R18_b_r', 'v_R18_b_i', 'v_R18_c_r', 'v_R18_c_i', 'v_R18_n_r', 'v_R18_n_i', 'v_I02_a_r', 'v_I02_a_i', 'v_I02_b_r', 'v_I02_b_i', 'v_I02_c_r', 'v_I02_c_i', 'v_I02_n_r', 'v_I02_n_i', 'v_C01_a_r', 'v_C01_a_i', 'v_C01_b_r', 'v_C01_b_i', 'v_C01_c_r', 'v_C01_c_i', 'v_C01_n_r', 'v_C01_n_i', 'v_C12_a_r', 'v_C12_a_i', 'v_C12_b_r', 'v_C12_b_i', 'v_C12_c_r', 'v_C12_c_i', 'v_C12_n_r', 'v_C12_n_i', 'v_C13_a_r', 'v_C13_a_i', 'v_C13_b_r', 'v_C13_b_i', 'v_C13_c_r', 'v_C13_c_i', 'v_C13_n_r', 'v_C13_n_i', 'v_C14_a_r', 'v_C14_a_i', 'v_C14_b_r', 'v_C14_b_i', 'v_C14_c_r', 'v_C14_c_i', 'v_C14_n_r', 'v_C14_n_i', 'v_C17_a_r', 'v_C17_a_i', 'v_C17_b_r', 'v_C17_b_i', 'v_C17_c_r', 'v_C17_c_i', 'v_C17_n_r', 'v_C17_n_i', 'v_C18_a_r', 'v_C18_a_i', 'v_C18_b_r', 'v_C18_b_i', 'v_C18_c_r', 'v_C18_c_i', 'v_C18_n_r', 'v_C18_n_i', 'v_C19_a_r', 'v_C19_a_i', 'v_C19_b_r', 'v_C19_b_i', 'v_C19_c_r', 'v_C19_c_i', 'v_C19_n_r', 'v_C19_n_i', 'v_C20_a_r', 'v_C20_a_i', 'v_C20_b_r', 'v_C20_b_i', 'v_C20_c_r', 'v_C20_c_i', 'v_C20_n_r', 'v_C20_n_i', 'v_MV0_a_r', 'v_MV0_a_i', 'v_MV0_b_r', 'v_MV0_b_i', 'v_MV0_c_r', 'v_MV0_c_i', 'v_I01_a_r', 'v_I01_a_i', 'v_I01_b_r', 'v_I01_b_i', 'v_I01_c_r', 'v_I01_c_i', 'v_I01_n_r', 'v_I01_n_i', 'v_R02_a_r', 'v_R02_a_i', 'v_R02_b_r', 'v_R02_b_i', 'v_R02_c_r', 'v_R02_c_i', 'v_R02_n_r', 'v_R02_n_i', 'v_R03_a_r', 'v_R03_a_i', 'v_R03_b_r', 'v_R03_b_i', 'v_R03_c_r', 'v_R03_c_i', 'v_R03_n_r', 'v_R03_n_i', 'v_R04_a_r', 'v_R04_a_i', 'v_R04_b_r', 'v_R04_b_i', 'v_R04_c_r', 'v_R04_c_i', 'v_R04_n_r', 'v_R04_n_i', 'v_R05_a_r', 'v_R05_a_i', 'v_R05_b_r', 'v_R05_b_i', 'v_R05_c_r', 'v_R05_c_i', 'v_R05_n_r', 'v_R05_n_i', 'v_R06_a_r', 'v_R06_a_i', 'v_R06_b_r', 'v_R06_b_i', 'v_R06_c_r', 'v_R06_c_i', 'v_R06_n_r', 'v_R06_n_i', 'v_R07_a_r', 'v_R07_a_i', 'v_R07_b_r', 'v_R07_b_i', 'v_R07_c_r', 'v_R07_c_i', 'v_R07_n_r', 'v_R07_n_i', 'v_R08_a_r', 'v_R08_a_i', 'v_R08_b_r', 'v_R08_b_i', 'v_R08_c_r', 'v_R08_c_i', 'v_R08_n_r', 'v_R08_n_i', 'v_R09_a_r', 'v_R09_a_i', 'v_R09_b_r', 'v_R09_b_i', 'v_R09_c_r', 'v_R09_c_i', 'v_R09_n_r', 'v_R09_n_i', 'v_R10_a_r', 'v_R10_a_i', 'v_R10_b_r', 'v_R10_b_i', 'v_R10_c_r', 'v_R10_c_i', 'v_R10_n_r', 'v_R10_n_i', 'v_R12_a_r', 'v_R12_a_i', 'v_R12_b_r', 'v_R12_b_i', 'v_R12_c_r', 'v_R12_c_i', 'v_R12_n_r', 'v_R12_n_i', 'v_R13_a_r', 'v_R13_a_i', 'v_R13_b_r', 'v_R13_b_i', 'v_R13_c_r', 'v_R13_c_i', 'v_R13_n_r', 'v_R13_n_i', 'v_R14_a_r', 'v_R14_a_i', 'v_R14_b_r', 'v_R14_b_i', 'v_R14_c_r', 'v_R14_c_i', 'v_R14_n_r', 'v_R14_n_i', 'v_C02_a_r', 'v_C02_a_i', 'v_C02_b_r', 'v_C02_b_i', 'v_C02_c_r', 'v_C02_c_i', 'v_C02_n_r', 'v_C02_n_i', 'v_C03_a_r', 'v_C03_a_i', 'v_C03_b_r', 'v_C03_b_i', 'v_C03_c_r', 'v_C03_c_i', 'v_C03_n_r', 'v_C03_n_i', 'v_C04_a_r', 'v_C04_a_i', 'v_C04_b_r', 'v_C04_b_i', 'v_C04_c_r', 'v_C04_c_i', 'v_C04_n_r', 'v_C04_n_i', 'v_C05_a_r', 'v_C05_a_i', 'v_C05_b_r', 'v_C05_b_i', 'v_C05_c_r', 'v_C05_c_i', 'v_C05_n_r', 'v_C05_n_i', 'v_C06_a_r', 'v_C06_a_i', 'v_C06_b_r', 'v_C06_b_i', 'v_C06_c_r', 'v_C06_c_i', 'v_C06_n_r', 'v_C06_n_i', 'v_C07_a_r', 'v_C07_a_i', 'v_C07_b_r', 'v_C07_b_i', 'v_C07_c_r', 'v_C07_c_i', 'v_C07_n_r', 'v_C07_n_i', 'v_C08_a_r', 'v_C08_a_i', 'v_C08_b_r', 'v_C08_b_i', 'v_C08_c_r', 'v_C08_c_i', 'v_C08_n_r', 'v_C08_n_i', 'v_C09_a_r', 'v_C09_a_i', 'v_C09_b_r', 'v_C09_b_i', 'v_C09_c_r', 'v_C09_c_i', 'v_C09_n_r', 'v_C09_n_i', 'v_C10_a_r', 'v_C10_a_i', 'v_C10_b_r', 'v_C10_b_i', 'v_C10_c_r', 'v_C10_c_i', 'v_C10_n_r', 'v_C10_n_i', 'v_C11_a_r', 'v_C11_a_i', 'v_C11_b_r', 'v_C11_b_i', 'v_C11_c_r', 'v_C11_c_i', 'v_C11_n_r', 'v_C11_n_i', 'v_C15_a_r', 'v_C15_a_i', 'v_C15_b_r', 'v_C15_b_i', 'v_C15_c_r', 'v_C15_c_i', 'v_C15_n_r', 'v_C15_n_i', 'v_C16_a_r', 'v_C16_a_i', 'v_C16_b_r', 'v_C16_b_i', 'v_C16_c_r', 'v_C16_c_i', 'v_C16_n_r', 'v_C16_n_i', 'i_t_MV0_R01_a_r', 'i_t_MV0_R01_a_i', 'i_t_MV0_R01_b_r', 'i_t_MV0_R01_b_i', 'i_t_MV0_R01_c_r', 'i_t_MV0_R01_c_i', 'i_t_MV0_I01_a_r', 'i_t_MV0_I01_a_i', 'i_t_MV0_I01_b_r', 'i_t_MV0_I01_b_i', 'i_t_MV0_I01_c_r', 'i_t_MV0_I01_c_i', 'i_t_MV0_C01_a_r', 'i_t_MV0_C01_a_i', 'i_t_MV0_C01_b_r', 'i_t_MV0_C01_b_i', 'i_t_MV0_C01_c_r', 'i_t_MV0_C01_c_i', 'i_load_R01_a_r', 'i_load_R01_a_i', 'i_load_R01_b_r', 'i_load_R01_b_i', 'i_load_R01_c_r', 'i_load_R01_c_i', 'i_load_R01_n_r', 'i_load_R01_n_i', 'i_load_R11_a_r', 'i_load_R11_a_i', 'i_load_R11_b_r', 'i_load_R11_b_i', 'i_load_R11_c_r', 'i_load_R11_c_i', 'i_load_R11_n_r', 'i_load_R11_n_i', 'i_load_R15_a_r', 'i_load_R15_a_i', 'i_load_R15_b_r', 'i_load_R15_b_i', 'i_load_R15_c_r', 'i_load_R15_c_i', 'i_load_R15_n_r', 'i_load_R15_n_i', 'i_load_R16_a_r', 'i_load_R16_a_i', 'i_load_R16_b_r', 'i_load_R16_b_i', 'i_load_R16_c_r', 'i_load_R16_c_i', 'i_load_R16_n_r', 'i_load_R16_n_i', 'i_load_R17_a_r', 'i_load_R17_a_i', 'i_load_R17_b_r', 'i_load_R17_b_i', 'i_load_R17_c_r', 'i_load_R17_c_i', 'i_load_R17_n_r', 'i_load_R17_n_i', 'i_load_R18_a_r', 'i_load_R18_a_i', 'i_load_R18_b_r', 'i_load_R18_b_i', 'i_load_R18_c_r', 'i_load_R18_c_i', 'i_load_R18_n_r', 'i_load_R18_n_i', 'i_load_I02_a_r', 'i_load_I02_a_i', 'i_load_I02_b_r', 'i_load_I02_b_i', 'i_load_I02_c_r', 'i_load_I02_c_i', 'i_load_I02_n_r', 'i_load_I02_n_i', 'i_load_C01_a_r', 'i_load_C01_a_i', 'i_load_C01_b_r', 'i_load_C01_b_i', 'i_load_C01_c_r', 'i_load_C01_c_i', 'i_load_C01_n_r', 'i_load_C01_n_i', 'i_load_C12_a_r', 'i_load_C12_a_i', 'i_load_C12_b_r', 'i_load_C12_b_i', 'i_load_C12_c_r', 'i_load_C12_c_i', 'i_load_C12_n_r', 'i_load_C12_n_i', 'i_load_C13_a_r', 'i_load_C13_a_i', 'i_load_C13_b_r', 'i_load_C13_b_i', 'i_load_C13_c_r', 'i_load_C13_c_i', 'i_load_C13_n_r', 'i_load_C13_n_i', 'i_load_C14_a_r', 'i_load_C14_a_i', 'i_load_C14_b_r', 'i_load_C14_b_i', 'i_load_C14_c_r', 'i_load_C14_c_i', 'i_load_C14_n_r', 'i_load_C14_n_i', 'i_load_C17_a_r', 'i_load_C17_a_i', 'i_load_C17_b_r', 'i_load_C17_b_i', 'i_load_C17_c_r', 'i_load_C17_c_i', 'i_load_C17_n_r', 'i_load_C17_n_i', 'i_load_C18_a_r', 'i_load_C18_a_i', 'i_load_C18_b_r', 'i_load_C18_b_i', 'i_load_C18_c_r', 'i_load_C18_c_i', 'i_load_C18_n_r', 'i_load_C18_n_i', 'i_load_C19_a_r', 'i_load_C19_a_i', 'i_load_C19_b_r', 'i_load_C19_b_i', 'i_load_C19_c_r', 'i_load_C19_c_i', 'i_load_C19_n_r', 'i_load_C19_n_i', 'i_load_C20_a_r', 'i_load_C20_a_i', 'i_load_C20_b_r', 'i_load_C20_b_i', 'i_load_C20_c_r', 'i_load_C20_c_i', 'i_load_C20_n_r', 'i_load_C20_n_i', 'i_R02_a_r', 'i_R02_b_r', 'i_R02_c_r', 'i_R02_n_r', 'i_R02_ng_r', 'e_R02_ng_r', 'i_R02_a_i', 'i_R02_b_i', 'i_R02_c_i', 'i_R02_n_i', 'i_R02_ng_i', 'e_R02_ng_i', 'i_R04_a_r', 'i_R04_b_r', 'i_R04_c_r', 'i_R04_n_r', 'i_R04_ng_r', 'e_R04_ng_r', 'i_R04_a_i', 'i_R04_b_i', 'i_R04_c_i', 'i_R04_n_i', 'i_R04_ng_i', 'e_R04_ng_i', 'i_R08_a_r', 'i_R08_b_r', 'i_R08_c_r', 'i_R08_n_r', 'i_R08_ng_r', 'e_R08_ng_r', 'i_R08_a_i', 'i_R08_b_i', 'i_R08_c_i', 'i_R08_n_i', 'i_R08_ng_i', 'e_R08_ng_i', 'i_I01_a_r', 'i_I01_b_r', 'i_I01_c_r', 'i_I01_n_r', 'i_I01_ng_r', 'e_I01_ng_r', 'i_I01_a_i', 'i_I01_b_i', 'i_I01_c_i', 'i_I01_n_i', 'i_I01_ng_i', 'e_I01_ng_i', 'i_C09_a_r', 'i_C09_b_r', 'i_C09_c_r', 'i_C09_n_r', 'i_C09_ng_r', 'e_C09_ng_r', 'i_C09_a_i', 'i_C09_b_i', 'i_C09_c_i', 'i_C09_n_i', 'i_C09_ng_i', 'e_C09_ng_i', 'i_C11_a_r', 'i_C11_b_r', 'i_C11_c_r', 'i_C11_n_r', 'i_C11_ng_r', 'e_C11_ng_r', 'i_C11_a_i', 'i_C11_b_i', 'i_C11_c_i', 'i_C11_n_i', 'i_C11_ng_i', 'e_C11_ng_i', 'i_C02_a_r', 'i_C02_b_r', 'i_C02_c_r', 'i_C02_n_r', 'i_C02_ng_r', 'e_C02_ng_r', 'i_C02_a_i', 'i_C02_b_i', 'i_C02_c_i', 'i_C02_n_i', 'i_C02_ng_i', 'e_C02_ng_i', 'omega_coi'] 
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
        self.u_run = np.array(self.u_run_values_list,dtype=np.float64)
 
        self.sp_jac_ini_ia, self.sp_jac_ini_ja, self.sp_jac_ini_nia, self.sp_jac_ini_nja = sp_jac_ini_vectors()
        data = np.array(self.sp_jac_ini_ia,dtype=np.float64)
        self.sp_jac_ini = sspa.csr_matrix((data, self.sp_jac_ini_ia, self.sp_jac_ini_ja), shape=(self.sp_jac_ini_nia,self.sp_jac_ini_nja))

        self.sp_jac_run_ia, self.sp_jac_run_ja, self.sp_jac_run_nia, self.sp_jac_run_nja = sp_jac_run_vectors()
        data = np.array(self.sp_jac_run_ia,dtype=np.float64)
        self.sp_jac_run = sspa.csr_matrix((data, self.sp_jac_run_ia, self.sp_jac_run_ja), shape=(self.sp_jac_run_nia,self.sp_jac_run_nja))

        self.sp_jac_trap_ia, self.sp_jac_trap_ja, self.sp_jac_trap_nia, self.sp_jac_trap_nja = sp_jac_trap_vectors()
        data = np.array(self.sp_jac_trap_ia,dtype=np.float64)
        self.sp_jac_trap = sspa.csr_matrix((data, self.sp_jac_trap_ia, self.sp_jac_trap_ja), shape=(self.sp_jac_trap_nia,self.sp_jac_trap_nja))


        self.J_run_d = np.array(self.sp_jac_run_ia)*0.0
        self.J_run_i = np.array(self.sp_jac_run_ia)
        self.J_run_p = np.array(self.sp_jac_run_ja)
        
        self.J_trap_d = np.array(self.sp_jac_trap_ia)*0.0
        self.J_trap_i = np.array(self.sp_jac_trap_ia)
        self.J_trap_p = np.array(self.sp_jac_trap_ja)
        
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
        
        # dense jacobians
        de_jac_ini_eval(self.jac_ini,x,y,self.u_ini,self.p,self.Dt)
        de_jac_run_eval(self.jac_run,x,y,self.u_ini,self.p,self.Dt)
        de_jac_trap_eval(self.jac_trap,x,y,self.u_run,self.p,self.Dt)
        
        # sparse jacobians   
        sp_jac_ini_eval(self.J_ini_d,x,y,self.u_ini,self.p,self.Dt)
        sp_jac_ini_eval(self.J_run_d,x,y,self.u_ini,self.p,self.Dt)
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
        if name_ in self.inputs_ini_list:
            self.u_ini[self.inputs_ini_list.index(name_)] = value
            return
        elif name_ in self.inputs_run_list:
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
                
        if type(xy_0) == float or type(xy_0) == int:
            self.xy_0 = np.ones(self.N_x+self.N_y,dtype=np.float64)*xy_0


        self.xy_ini = self.ss_ini()
        self.ini2run()

    
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

    sp_jac_ini_ia = [0, 1, 563, 1, 14, 149, 150, 151, 152, 153, 154, 479, 480, 481, 485, 486, 487, 2, 3, 563, 3, 14, 165, 166, 167, 168, 169, 170, 491, 492, 493, 497, 498, 499, 4, 5, 563, 5, 14, 197, 198, 199, 200, 201, 202, 503, 504, 505, 509, 510, 511, 6, 7, 563, 7, 14, 141, 142, 143, 144, 145, 146, 515, 516, 517, 521, 522, 523, 8, 9, 563, 9, 14, 301, 302, 303, 304, 305, 306, 527, 528, 529, 533, 534, 535, 10, 11, 563, 11, 14, 317, 318, 319, 320, 321, 322, 539, 540, 541, 545, 546, 547, 12, 13, 563, 13, 14, 245, 246, 247, 248, 249, 250, 551, 552, 553, 557, 558, 559, 563, 15, 16, 17, 18, 19, 20, 21, 22, 135, 136, 139, 140, 149, 150, 151, 152, 153, 154, 155, 156, 359, 15, 16, 17, 18, 19, 20, 21, 22, 135, 136, 139, 140, 149, 150, 151, 152, 153, 154, 155, 156, 360, 15, 16, 17, 18, 19, 20, 21, 22, 135, 136, 137, 138, 149, 150, 151, 152, 153, 154, 155, 156, 361, 15, 16, 17, 18, 19, 20, 21, 22, 135, 136, 137, 138, 149, 150, 151, 152, 153, 154, 155, 156, 362, 15, 16, 17, 18, 19, 20, 21, 22, 137, 138, 139, 140, 149, 150, 151, 152, 153, 154, 155, 156, 363, 15, 16, 17, 18, 19, 20, 21, 22, 137, 138, 139, 140, 149, 150, 151, 152, 153, 154, 155, 156, 364, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 23, 24, 25, 26, 27, 28, 29, 30, 157, 158, 159, 160, 161, 162, 163, 164, 367, 23, 24, 25, 26, 27, 28, 29, 30, 157, 158, 159, 160, 161, 162, 163, 164, 368, 23, 24, 25, 26, 27, 28, 29, 30, 157, 158, 159, 160, 161, 162, 163, 164, 369, 23, 24, 25, 26, 27, 28, 29, 30, 157, 158, 159, 160, 161, 162, 163, 164, 370, 23, 24, 25, 26, 27, 28, 29, 30, 157, 158, 159, 160, 161, 162, 163, 164, 371, 23, 24, 25, 26, 27, 28, 29, 30, 157, 158, 159, 160, 161, 162, 163, 164, 372, 23, 24, 25, 26, 27, 28, 29, 30, 157, 158, 159, 160, 161, 162, 163, 164, 23, 24, 25, 26, 27, 28, 29, 30, 157, 158, 159, 160, 161, 162, 163, 164, 31, 32, 33, 34, 35, 36, 37, 38, 237, 238, 239, 240, 241, 242, 243, 244, 375, 31, 32, 33, 34, 35, 36, 37, 38, 237, 238, 239, 240, 241, 242, 243, 244, 376, 31, 32, 33, 34, 35, 36, 37, 38, 237, 238, 239, 240, 241, 242, 243, 244, 377, 31, 32, 33, 34, 35, 36, 37, 38, 237, 238, 239, 240, 241, 242, 243, 244, 378, 31, 32, 33, 34, 35, 36, 37, 38, 237, 238, 239, 240, 241, 242, 243, 244, 379, 31, 32, 33, 34, 35, 36, 37, 38, 237, 238, 239, 240, 241, 242, 243, 244, 380, 31, 32, 33, 34, 35, 36, 37, 38, 237, 238, 239, 240, 241, 242, 243, 244, 31, 32, 33, 34, 35, 36, 37, 38, 237, 238, 239, 240, 241, 242, 243, 244, 39, 40, 41, 42, 43, 44, 45, 46, 181, 182, 183, 184, 185, 186, 187, 188, 383, 39, 40, 41, 42, 43, 44, 45, 46, 181, 182, 183, 184, 185, 186, 187, 188, 384, 39, 40, 41, 42, 43, 44, 45, 46, 181, 182, 183, 184, 185, 186, 187, 188, 385, 39, 40, 41, 42, 43, 44, 45, 46, 181, 182, 183, 184, 185, 186, 187, 188, 386, 39, 40, 41, 42, 43, 44, 45, 46, 181, 182, 183, 184, 185, 186, 187, 188, 387, 39, 40, 41, 42, 43, 44, 45, 46, 181, 182, 183, 184, 185, 186, 187, 188, 388, 39, 40, 41, 42, 43, 44, 45, 46, 181, 182, 183, 184, 185, 186, 187, 188, 39, 40, 41, 42, 43, 44, 45, 46, 181, 182, 183, 184, 185, 186, 187, 188, 47, 48, 49, 50, 51, 52, 53, 54, 205, 206, 207, 208, 209, 210, 211, 212, 391, 47, 48, 49, 50, 51, 52, 53, 54, 205, 206, 207, 208, 209, 210, 211, 212, 392, 47, 48, 49, 50, 51, 52, 53, 54, 205, 206, 207, 208, 209, 210, 211, 212, 393, 47, 48, 49, 50, 51, 52, 53, 54, 205, 206, 207, 208, 209, 210, 211, 212, 394, 47, 48, 49, 50, 51, 52, 53, 54, 205, 206, 207, 208, 209, 210, 211, 212, 395, 47, 48, 49, 50, 51, 52, 53, 54, 205, 206, 207, 208, 209, 210, 211, 212, 396, 47, 48, 49, 50, 51, 52, 53, 54, 205, 206, 207, 208, 209, 210, 211, 212, 47, 48, 49, 50, 51, 52, 53, 54, 205, 206, 207, 208, 209, 210, 211, 212, 55, 56, 57, 58, 59, 60, 61, 62, 213, 214, 215, 216, 217, 218, 219, 220, 399, 55, 56, 57, 58, 59, 60, 61, 62, 213, 214, 215, 216, 217, 218, 219, 220, 400, 55, 56, 57, 58, 59, 60, 61, 62, 213, 214, 215, 216, 217, 218, 219, 220, 401, 55, 56, 57, 58, 59, 60, 61, 62, 213, 214, 215, 216, 217, 218, 219, 220, 402, 55, 56, 57, 58, 59, 60, 61, 62, 213, 214, 215, 216, 217, 218, 219, 220, 403, 55, 56, 57, 58, 59, 60, 61, 62, 213, 214, 215, 216, 217, 218, 219, 220, 404, 55, 56, 57, 58, 59, 60, 61, 62, 213, 214, 215, 216, 217, 218, 219, 220, 55, 56, 57, 58, 59, 60, 61, 62, 213, 214, 215, 216, 217, 218, 219, 220, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 407, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 408, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 409, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 410, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 411, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 412, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 71, 72, 73, 74, 75, 76, 77, 78, 135, 136, 139, 140, 245, 246, 247, 248, 249, 250, 251, 252, 415, 71, 72, 73, 74, 75, 76, 77, 78, 135, 136, 139, 140, 245, 246, 247, 248, 249, 250, 251, 252, 416, 71, 72, 73, 74, 75, 76, 77, 78, 135, 136, 137, 138, 245, 246, 247, 248, 249, 250, 251, 252, 417, 71, 72, 73, 74, 75, 76, 77, 78, 135, 136, 137, 138, 245, 246, 247, 248, 249, 250, 251, 252, 418, 71, 72, 73, 74, 75, 76, 77, 78, 137, 138, 139, 140, 245, 246, 247, 248, 249, 250, 251, 252, 419, 71, 72, 73, 74, 75, 76, 77, 78, 137, 138, 139, 140, 245, 246, 247, 248, 249, 250, 251, 252, 420, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 79, 80, 81, 82, 83, 84, 85, 86, 317, 318, 319, 320, 321, 322, 323, 324, 423, 79, 80, 81, 82, 83, 84, 85, 86, 317, 318, 319, 320, 321, 322, 323, 324, 424, 79, 80, 81, 82, 83, 84, 85, 86, 317, 318, 319, 320, 321, 322, 323, 324, 425, 79, 80, 81, 82, 83, 84, 85, 86, 317, 318, 319, 320, 321, 322, 323, 324, 426, 79, 80, 81, 82, 83, 84, 85, 86, 317, 318, 319, 320, 321, 322, 323, 324, 427, 79, 80, 81, 82, 83, 84, 85, 86, 317, 318, 319, 320, 321, 322, 323, 324, 428, 79, 80, 81, 82, 83, 84, 85, 86, 317, 318, 319, 320, 321, 322, 323, 324, 79, 80, 81, 82, 83, 84, 85, 86, 317, 318, 319, 320, 321, 322, 323, 324, 87, 88, 89, 90, 91, 92, 93, 94, 317, 318, 319, 320, 321, 322, 323, 324, 431, 87, 88, 89, 90, 91, 92, 93, 94, 317, 318, 319, 320, 321, 322, 323, 324, 432, 87, 88, 89, 90, 91, 92, 93, 94, 317, 318, 319, 320, 321, 322, 323, 324, 433, 87, 88, 89, 90, 91, 92, 93, 94, 317, 318, 319, 320, 321, 322, 323, 324, 434, 87, 88, 89, 90, 91, 92, 93, 94, 317, 318, 319, 320, 321, 322, 323, 324, 435, 87, 88, 89, 90, 91, 92, 93, 94, 317, 318, 319, 320, 321, 322, 323, 324, 436, 87, 88, 89, 90, 91, 92, 93, 94, 317, 318, 319, 320, 321, 322, 323, 324, 87, 88, 89, 90, 91, 92, 93, 94, 317, 318, 319, 320, 321, 322, 323, 324, 95, 96, 97, 98, 99, 100, 101, 102, 309, 310, 311, 312, 313, 314, 315, 316, 439, 95, 96, 97, 98, 99, 100, 101, 102, 309, 310, 311, 312, 313, 314, 315, 316, 440, 95, 96, 97, 98, 99, 100, 101, 102, 309, 310, 311, 312, 313, 314, 315, 316, 441, 95, 96, 97, 98, 99, 100, 101, 102, 309, 310, 311, 312, 313, 314, 315, 316, 442, 95, 96, 97, 98, 99, 100, 101, 102, 309, 310, 311, 312, 313, 314, 315, 316, 443, 95, 96, 97, 98, 99, 100, 101, 102, 309, 310, 311, 312, 313, 314, 315, 316, 444, 95, 96, 97, 98, 99, 100, 101, 102, 309, 310, 311, 312, 313, 314, 315, 316, 95, 96, 97, 98, 99, 100, 101, 102, 309, 310, 311, 312, 313, 314, 315, 316, 103, 104, 105, 106, 107, 108, 109, 110, 333, 334, 335, 336, 337, 338, 339, 340, 447, 103, 104, 105, 106, 107, 108, 109, 110, 333, 334, 335, 336, 337, 338, 339, 340, 448, 103, 104, 105, 106, 107, 108, 109, 110, 333, 334, 335, 336, 337, 338, 339, 340, 449, 103, 104, 105, 106, 107, 108, 109, 110, 333, 334, 335, 336, 337, 338, 339, 340, 450, 103, 104, 105, 106, 107, 108, 109, 110, 333, 334, 335, 336, 337, 338, 339, 340, 451, 103, 104, 105, 106, 107, 108, 109, 110, 333, 334, 335, 336, 337, 338, 339, 340, 452, 103, 104, 105, 106, 107, 108, 109, 110, 333, 334, 335, 336, 337, 338, 339, 340, 103, 104, 105, 106, 107, 108, 109, 110, 333, 334, 335, 336, 337, 338, 339, 340, 111, 112, 113, 114, 115, 116, 117, 118, 325, 326, 327, 328, 329, 330, 331, 332, 455, 111, 112, 113, 114, 115, 116, 117, 118, 325, 326, 327, 328, 329, 330, 331, 332, 456, 111, 112, 113, 114, 115, 116, 117, 118, 325, 326, 327, 328, 329, 330, 331, 332, 457, 111, 112, 113, 114, 115, 116, 117, 118, 325, 326, 327, 328, 329, 330, 331, 332, 458, 111, 112, 113, 114, 115, 116, 117, 118, 325, 326, 327, 328, 329, 330, 331, 332, 459, 111, 112, 113, 114, 115, 116, 117, 118, 325, 326, 327, 328, 329, 330, 331, 332, 460, 111, 112, 113, 114, 115, 116, 117, 118, 325, 326, 327, 328, 329, 330, 331, 332, 111, 112, 113, 114, 115, 116, 117, 118, 325, 326, 327, 328, 329, 330, 331, 332, 119, 120, 121, 122, 123, 124, 125, 126, 293, 294, 295, 296, 297, 298, 299, 300, 463, 119, 120, 121, 122, 123, 124, 125, 126, 293, 294, 295, 296, 297, 298, 299, 300, 464, 119, 120, 121, 122, 123, 124, 125, 126, 293, 294, 295, 296, 297, 298, 299, 300, 465, 119, 120, 121, 122, 123, 124, 125, 126, 293, 294, 295, 296, 297, 298, 299, 300, 466, 119, 120, 121, 122, 123, 124, 125, 126, 293, 294, 295, 296, 297, 298, 299, 300, 467, 119, 120, 121, 122, 123, 124, 125, 126, 293, 294, 295, 296, 297, 298, 299, 300, 468, 119, 120, 121, 122, 123, 124, 125, 126, 293, 294, 295, 296, 297, 298, 299, 300, 119, 120, 121, 122, 123, 124, 125, 126, 293, 294, 295, 296, 297, 298, 299, 300, 127, 128, 129, 130, 131, 132, 133, 134, 301, 302, 303, 304, 305, 306, 307, 308, 471, 127, 128, 129, 130, 131, 132, 133, 134, 301, 302, 303, 304, 305, 306, 307, 308, 472, 127, 128, 129, 130, 131, 132, 133, 134, 301, 302, 303, 304, 305, 306, 307, 308, 473, 127, 128, 129, 130, 131, 132, 133, 134, 301, 302, 303, 304, 305, 306, 307, 308, 474, 127, 128, 129, 130, 131, 132, 133, 134, 301, 302, 303, 304, 305, 306, 307, 308, 475, 127, 128, 129, 130, 131, 132, 133, 134, 301, 302, 303, 304, 305, 306, 307, 308, 476, 127, 128, 129, 130, 131, 132, 133, 134, 301, 302, 303, 304, 305, 306, 307, 308, 127, 128, 129, 130, 131, 132, 133, 134, 301, 302, 303, 304, 305, 306, 307, 308, 15, 16, 17, 18, 71, 72, 73, 74, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 15, 16, 17, 18, 71, 72, 73, 74, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 17, 18, 19, 20, 73, 74, 75, 76, 135, 136, 137, 138, 139, 140, 143, 144, 145, 146, 17, 18, 19, 20, 73, 74, 75, 76, 135, 136, 137, 138, 139, 140, 143, 144, 145, 146, 15, 16, 19, 20, 71, 72, 75, 76, 135, 136, 137, 138, 139, 140, 141, 142, 145, 146, 15, 16, 19, 20, 71, 72, 75, 76, 135, 136, 137, 138, 139, 140, 141, 142, 145, 146, 63, 64, 65, 66, 67, 68, 69, 70, 135, 136, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 515, 63, 64, 65, 66, 67, 68, 69, 70, 135, 136, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 521, 63, 64, 65, 66, 67, 68, 69, 70, 135, 136, 137, 138, 141, 142, 143, 144, 145, 146, 147, 148, 516, 63, 64, 65, 66, 67, 68, 69, 70, 135, 136, 137, 138, 141, 142, 143, 144, 145, 146, 147, 148, 522, 63, 64, 65, 66, 67, 68, 69, 70, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 517, 63, 64, 65, 66, 67, 68, 69, 70, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 523, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 518, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 524, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 479, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 485, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 480, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 486, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 481, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 487, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 482, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 488, 23, 24, 25, 26, 27, 28, 29, 30, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 23, 24, 25, 26, 27, 28, 29, 30, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 23, 24, 25, 26, 27, 28, 29, 30, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 23, 24, 25, 26, 27, 28, 29, 30, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 23, 24, 25, 26, 27, 28, 29, 30, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 23, 24, 25, 26, 27, 28, 29, 30, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 23, 24, 25, 26, 27, 28, 29, 30, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 23, 24, 25, 26, 27, 28, 29, 30, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 221, 222, 223, 224, 225, 226, 227, 228, 491, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 221, 222, 223, 224, 225, 226, 227, 228, 497, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 221, 222, 223, 224, 225, 226, 227, 228, 492, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 221, 222, 223, 224, 225, 226, 227, 228, 498, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 221, 222, 223, 224, 225, 226, 227, 228, 493, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 221, 222, 223, 224, 225, 226, 227, 228, 499, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 221, 222, 223, 224, 225, 226, 227, 228, 494, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 221, 222, 223, 224, 225, 226, 227, 228, 500, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 39, 40, 41, 42, 43, 44, 45, 46, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 39, 40, 41, 42, 43, 44, 45, 46, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 39, 40, 41, 42, 43, 44, 45, 46, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 39, 40, 41, 42, 43, 44, 45, 46, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 39, 40, 41, 42, 43, 44, 45, 46, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 39, 40, 41, 42, 43, 44, 45, 46, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 39, 40, 41, 42, 43, 44, 45, 46, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 39, 40, 41, 42, 43, 44, 45, 46, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 503, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 509, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 504, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 510, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 505, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 511, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 506, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 512, 47, 48, 49, 50, 51, 52, 53, 54, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 47, 48, 49, 50, 51, 52, 53, 54, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 47, 48, 49, 50, 51, 52, 53, 54, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 47, 48, 49, 50, 51, 52, 53, 54, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 47, 48, 49, 50, 51, 52, 53, 54, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 47, 48, 49, 50, 51, 52, 53, 54, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 47, 48, 49, 50, 51, 52, 53, 54, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 47, 48, 49, 50, 51, 52, 53, 54, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 55, 56, 57, 58, 59, 60, 61, 62, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 55, 56, 57, 58, 59, 60, 61, 62, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 55, 56, 57, 58, 59, 60, 61, 62, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 55, 56, 57, 58, 59, 60, 61, 62, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 55, 56, 57, 58, 59, 60, 61, 62, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 55, 56, 57, 58, 59, 60, 61, 62, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 55, 56, 57, 58, 59, 60, 61, 62, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 55, 56, 57, 58, 59, 60, 61, 62, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 165, 166, 167, 168, 169, 170, 171, 172, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 165, 166, 167, 168, 169, 170, 171, 172, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 165, 166, 167, 168, 169, 170, 171, 172, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 165, 166, 167, 168, 169, 170, 171, 172, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 165, 166, 167, 168, 169, 170, 171, 172, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 165, 166, 167, 168, 169, 170, 171, 172, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 165, 166, 167, 168, 169, 170, 171, 172, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 165, 166, 167, 168, 169, 170, 171, 172, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 31, 32, 33, 34, 35, 36, 37, 38, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 31, 32, 33, 34, 35, 36, 37, 38, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 31, 32, 33, 34, 35, 36, 37, 38, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 31, 32, 33, 34, 35, 36, 37, 38, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 31, 32, 33, 34, 35, 36, 37, 38, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 31, 32, 33, 34, 35, 36, 37, 38, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 31, 32, 33, 34, 35, 36, 37, 38, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 31, 32, 33, 34, 35, 36, 37, 38, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 551, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 557, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 552, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 558, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 553, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 559, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 554, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 560, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 309, 310, 311, 312, 313, 314, 315, 316, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 309, 310, 311, 312, 313, 314, 315, 316, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 309, 310, 311, 312, 313, 314, 315, 316, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 309, 310, 311, 312, 313, 314, 315, 316, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 309, 310, 311, 312, 313, 314, 315, 316, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 309, 310, 311, 312, 313, 314, 315, 316, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 309, 310, 311, 312, 313, 314, 315, 316, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 309, 310, 311, 312, 313, 314, 315, 316, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 325, 326, 327, 328, 329, 330, 331, 332, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 325, 326, 327, 328, 329, 330, 331, 332, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 325, 326, 327, 328, 329, 330, 331, 332, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 325, 326, 327, 328, 329, 330, 331, 332, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 325, 326, 327, 328, 329, 330, 331, 332, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 325, 326, 327, 328, 329, 330, 331, 332, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 325, 326, 327, 328, 329, 330, 331, 332, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 325, 326, 327, 328, 329, 330, 331, 332, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 119, 120, 121, 122, 123, 124, 125, 126, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 119, 120, 121, 122, 123, 124, 125, 126, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 119, 120, 121, 122, 123, 124, 125, 126, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 119, 120, 121, 122, 123, 124, 125, 126, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 119, 120, 121, 122, 123, 124, 125, 126, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 119, 120, 121, 122, 123, 124, 125, 126, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 119, 120, 121, 122, 123, 124, 125, 126, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 119, 120, 121, 122, 123, 124, 125, 126, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 127, 128, 129, 130, 131, 132, 133, 134, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 527, 127, 128, 129, 130, 131, 132, 133, 134, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 533, 127, 128, 129, 130, 131, 132, 133, 134, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 528, 127, 128, 129, 130, 131, 132, 133, 134, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 534, 127, 128, 129, 130, 131, 132, 133, 134, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 529, 127, 128, 129, 130, 131, 132, 133, 134, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 535, 127, 128, 129, 130, 131, 132, 133, 134, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 530, 127, 128, 129, 130, 131, 132, 133, 134, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 536, 95, 96, 97, 98, 99, 100, 101, 102, 253, 254, 255, 256, 257, 258, 259, 260, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 95, 96, 97, 98, 99, 100, 101, 102, 253, 254, 255, 256, 257, 258, 259, 260, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 95, 96, 97, 98, 99, 100, 101, 102, 253, 254, 255, 256, 257, 258, 259, 260, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 95, 96, 97, 98, 99, 100, 101, 102, 253, 254, 255, 256, 257, 258, 259, 260, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 95, 96, 97, 98, 99, 100, 101, 102, 253, 254, 255, 256, 257, 258, 259, 260, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 95, 96, 97, 98, 99, 100, 101, 102, 253, 254, 255, 256, 257, 258, 259, 260, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 95, 96, 97, 98, 99, 100, 101, 102, 253, 254, 255, 256, 257, 258, 259, 260, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 95, 96, 97, 98, 99, 100, 101, 102, 253, 254, 255, 256, 257, 258, 259, 260, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 539, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 545, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 540, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 546, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 541, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 547, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 542, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 548, 111, 112, 113, 114, 115, 116, 117, 118, 269, 270, 271, 272, 273, 274, 275, 276, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 111, 112, 113, 114, 115, 116, 117, 118, 269, 270, 271, 272, 273, 274, 275, 276, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 111, 112, 113, 114, 115, 116, 117, 118, 269, 270, 271, 272, 273, 274, 275, 276, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 111, 112, 113, 114, 115, 116, 117, 118, 269, 270, 271, 272, 273, 274, 275, 276, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 111, 112, 113, 114, 115, 116, 117, 118, 269, 270, 271, 272, 273, 274, 275, 276, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 111, 112, 113, 114, 115, 116, 117, 118, 269, 270, 271, 272, 273, 274, 275, 276, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 111, 112, 113, 114, 115, 116, 117, 118, 269, 270, 271, 272, 273, 274, 275, 276, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 111, 112, 113, 114, 115, 116, 117, 118, 269, 270, 271, 272, 273, 274, 275, 276, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 103, 104, 105, 106, 107, 108, 109, 110, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 103, 104, 105, 106, 107, 108, 109, 110, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 103, 104, 105, 106, 107, 108, 109, 110, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 103, 104, 105, 106, 107, 108, 109, 110, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 103, 104, 105, 106, 107, 108, 109, 110, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 103, 104, 105, 106, 107, 108, 109, 110, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 103, 104, 105, 106, 107, 108, 109, 110, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 103, 104, 105, 106, 107, 108, 109, 110, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 15, 16, 17, 18, 135, 136, 137, 138, 139, 140, 341, 15, 16, 17, 18, 135, 136, 137, 138, 139, 140, 342, 17, 18, 19, 20, 135, 136, 137, 138, 139, 140, 343, 17, 18, 19, 20, 135, 136, 137, 138, 139, 140, 344, 15, 16, 19, 20, 135, 136, 137, 138, 139, 140, 345, 15, 16, 19, 20, 135, 136, 137, 138, 139, 140, 346, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 347, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 348, 135, 136, 137, 138, 139, 140, 143, 144, 145, 146, 349, 135, 136, 137, 138, 139, 140, 143, 144, 145, 146, 350, 135, 136, 137, 138, 139, 140, 141, 142, 145, 146, 351, 135, 136, 137, 138, 139, 140, 141, 142, 145, 146, 352, 71, 72, 73, 74, 135, 136, 137, 138, 139, 140, 353, 71, 72, 73, 74, 135, 136, 137, 138, 139, 140, 354, 73, 74, 75, 76, 135, 136, 137, 138, 139, 140, 355, 73, 74, 75, 76, 135, 136, 137, 138, 139, 140, 356, 71, 72, 75, 76, 135, 136, 137, 138, 139, 140, 357, 71, 72, 75, 76, 135, 136, 137, 138, 139, 140, 358, 15, 16, 21, 22, 359, 360, 17, 18, 21, 22, 361, 362, 19, 20, 21, 22, 363, 364, 15, 16, 21, 22, 359, 360, 17, 18, 21, 22, 361, 362, 19, 20, 21, 22, 363, 364, 359, 361, 363, 365, 360, 362, 364, 366, 23, 24, 29, 30, 367, 368, 25, 26, 29, 30, 369, 370, 27, 28, 29, 30, 371, 372, 23, 24, 29, 30, 367, 368, 25, 26, 29, 30, 369, 370, 27, 28, 29, 30, 371, 372, 367, 369, 371, 373, 368, 370, 372, 374, 31, 32, 37, 38, 375, 376, 33, 34, 37, 38, 377, 378, 35, 36, 37, 38, 379, 380, 31, 32, 37, 38, 375, 376, 33, 34, 37, 38, 377, 378, 35, 36, 37, 38, 379, 380, 375, 377, 379, 381, 376, 378, 380, 382, 39, 40, 45, 46, 383, 384, 41, 42, 45, 46, 385, 386, 43, 44, 45, 46, 387, 388, 39, 40, 45, 46, 383, 384, 41, 42, 45, 46, 385, 386, 43, 44, 45, 46, 387, 388, 383, 385, 387, 389, 384, 386, 388, 390, 47, 48, 53, 54, 391, 392, 49, 50, 53, 54, 393, 394, 51, 52, 53, 54, 395, 396, 47, 48, 53, 54, 391, 392, 49, 50, 53, 54, 393, 394, 51, 52, 53, 54, 395, 396, 391, 393, 395, 397, 392, 394, 396, 398, 55, 56, 61, 62, 399, 400, 57, 58, 61, 62, 401, 402, 59, 60, 61, 62, 403, 404, 55, 56, 61, 62, 399, 400, 57, 58, 61, 62, 401, 402, 59, 60, 61, 62, 403, 404, 399, 401, 403, 405, 400, 402, 404, 406, 63, 64, 69, 70, 407, 408, 65, 66, 69, 70, 409, 410, 67, 68, 69, 70, 411, 412, 63, 64, 69, 70, 407, 408, 65, 66, 69, 70, 409, 410, 67, 68, 69, 70, 411, 412, 407, 409, 411, 413, 408, 410, 412, 414, 71, 72, 77, 78, 415, 416, 73, 74, 77, 78, 417, 418, 75, 76, 77, 78, 419, 420, 71, 72, 77, 78, 415, 416, 73, 74, 77, 78, 417, 418, 75, 76, 77, 78, 419, 420, 415, 417, 419, 421, 416, 418, 420, 422, 79, 80, 85, 86, 423, 424, 81, 82, 85, 86, 425, 426, 83, 84, 85, 86, 427, 428, 79, 80, 85, 86, 423, 424, 81, 82, 85, 86, 425, 426, 83, 84, 85, 86, 427, 428, 423, 425, 427, 429, 424, 426, 428, 430, 87, 88, 93, 94, 431, 432, 89, 90, 93, 94, 433, 434, 91, 92, 93, 94, 435, 436, 87, 88, 93, 94, 431, 432, 89, 90, 93, 94, 433, 434, 91, 92, 93, 94, 435, 436, 431, 433, 435, 437, 432, 434, 436, 438, 95, 96, 101, 102, 439, 440, 97, 98, 101, 102, 441, 442, 99, 100, 101, 102, 443, 444, 95, 96, 101, 102, 439, 440, 97, 98, 101, 102, 441, 442, 99, 100, 101, 102, 443, 444, 439, 441, 443, 445, 440, 442, 444, 446, 103, 104, 109, 110, 447, 448, 105, 106, 109, 110, 449, 450, 107, 108, 109, 110, 451, 452, 103, 104, 109, 110, 447, 448, 105, 106, 109, 110, 449, 450, 107, 108, 109, 110, 451, 452, 447, 449, 451, 453, 448, 450, 452, 454, 111, 112, 117, 118, 455, 456, 113, 114, 117, 118, 457, 458, 115, 116, 117, 118, 459, 460, 111, 112, 117, 118, 455, 456, 113, 114, 117, 118, 457, 458, 115, 116, 117, 118, 459, 460, 455, 457, 459, 461, 456, 458, 460, 462, 119, 120, 125, 126, 463, 464, 121, 122, 125, 126, 465, 466, 123, 124, 125, 126, 467, 468, 119, 120, 125, 126, 463, 464, 121, 122, 125, 126, 465, 466, 123, 124, 125, 126, 467, 468, 463, 465, 467, 469, 464, 466, 468, 470, 127, 128, 133, 134, 471, 472, 129, 130, 133, 134, 473, 474, 131, 132, 133, 134, 475, 476, 127, 128, 133, 134, 471, 472, 129, 130, 133, 134, 473, 474, 131, 132, 133, 134, 475, 476, 471, 473, 475, 477, 472, 474, 476, 478, 0, 149, 155, 479, 485, 0, 151, 155, 480, 486, 0, 153, 155, 481, 487, 155, 482, 484, 488, 479, 480, 481, 482, 483, 483, 484, 489, 0, 150, 156, 479, 485, 0, 152, 156, 480, 486, 0, 154, 156, 481, 487, 156, 482, 488, 490, 485, 486, 487, 488, 489, 483, 489, 490, 2, 165, 171, 491, 497, 2, 167, 171, 492, 498, 2, 169, 171, 493, 499, 171, 494, 496, 500, 491, 492, 493, 494, 495, 495, 496, 501, 2, 166, 172, 491, 497, 2, 168, 172, 492, 498, 2, 170, 172, 493, 499, 172, 494, 500, 502, 497, 498, 499, 500, 501, 495, 501, 502, 4, 197, 203, 503, 509, 4, 199, 203, 504, 510, 4, 201, 203, 505, 511, 203, 506, 508, 512, 503, 504, 505, 506, 507, 507, 508, 513, 4, 198, 204, 503, 509, 4, 200, 204, 504, 510, 4, 202, 204, 505, 511, 204, 506, 512, 514, 509, 510, 511, 512, 513, 507, 513, 514, 6, 141, 147, 515, 521, 6, 143, 147, 516, 522, 6, 145, 147, 517, 523, 147, 518, 520, 524, 515, 516, 517, 518, 519, 519, 520, 525, 6, 142, 148, 515, 521, 6, 144, 148, 516, 522, 6, 146, 148, 517, 523, 148, 518, 524, 526, 521, 522, 523, 524, 525, 519, 525, 526, 8, 301, 307, 527, 533, 8, 303, 307, 528, 534, 8, 305, 307, 529, 535, 307, 530, 532, 536, 527, 528, 529, 530, 531, 531, 532, 537, 8, 302, 308, 527, 533, 8, 304, 308, 528, 534, 8, 306, 308, 529, 535, 308, 530, 536, 538, 533, 534, 535, 536, 537, 531, 537, 538, 10, 317, 323, 539, 545, 10, 319, 323, 540, 546, 10, 321, 323, 541, 547, 323, 542, 544, 548, 539, 540, 541, 542, 543, 543, 544, 549, 10, 318, 324, 539, 545, 10, 320, 324, 540, 546, 10, 322, 324, 541, 547, 324, 542, 548, 550, 545, 546, 547, 548, 549, 543, 549, 550, 12, 245, 251, 551, 557, 12, 247, 251, 552, 558, 12, 249, 251, 553, 559, 251, 554, 556, 560, 551, 552, 553, 554, 555, 555, 556, 561, 12, 246, 252, 551, 557, 12, 248, 252, 552, 558, 12, 250, 252, 553, 559, 252, 554, 560, 562, 557, 558, 559, 560, 561, 555, 561, 562, 1, 3, 5, 7, 9, 11, 13, 563]
    sp_jac_ini_ja = [0, 3, 17, 20, 34, 37, 51, 54, 68, 71, 85, 88, 102, 105, 119, 120, 141, 162, 183, 204, 225, 246, 262, 278, 295, 312, 329, 346, 363, 380, 396, 412, 429, 446, 463, 480, 497, 514, 530, 546, 563, 580, 597, 614, 631, 648, 664, 680, 697, 714, 731, 748, 765, 782, 798, 814, 831, 848, 865, 882, 899, 916, 932, 948, 965, 982, 999, 1016, 1033, 1050, 1066, 1082, 1103, 1124, 1145, 1166, 1187, 1208, 1224, 1240, 1257, 1274, 1291, 1308, 1325, 1342, 1358, 1374, 1391, 1408, 1425, 1442, 1459, 1476, 1492, 1508, 1525, 1542, 1559, 1576, 1593, 1610, 1626, 1642, 1659, 1676, 1693, 1710, 1727, 1744, 1760, 1776, 1793, 1810, 1827, 1844, 1861, 1878, 1894, 1910, 1927, 1944, 1961, 1978, 1995, 2012, 2028, 2044, 2061, 2078, 2095, 2112, 2129, 2146, 2162, 2178, 2196, 2214, 2232, 2250, 2268, 2286, 2307, 2328, 2349, 2370, 2391, 2412, 2429, 2446, 2471, 2496, 2521, 2546, 2571, 2596, 2621, 2646, 2678, 2710, 2742, 2774, 2806, 2838, 2870, 2902, 2935, 2968, 3001, 3034, 3067, 3100, 3133, 3166, 3190, 3214, 3238, 3262, 3286, 3310, 3334, 3358, 3390, 3422, 3454, 3486, 3518, 3550, 3582, 3614, 3638, 3662, 3686, 3710, 3734, 3758, 3782, 3806, 3831, 3856, 3881, 3906, 3931, 3956, 3981, 4006, 4038, 4070, 4102, 4134, 4166, 4198, 4230, 4262, 4286, 4310, 4334, 4358, 4382, 4406, 4430, 4454, 4478, 4502, 4526, 4550, 4574, 4598, 4622, 4646, 4670, 4694, 4718, 4742, 4766, 4790, 4814, 4838, 4862, 4886, 4910, 4934, 4958, 4982, 5006, 5030, 5055, 5080, 5105, 5130, 5155, 5180, 5205, 5230, 5262, 5294, 5326, 5358, 5390, 5422, 5454, 5486, 5510, 5534, 5558, 5582, 5606, 5630, 5654, 5678, 5710, 5742, 5774, 5806, 5838, 5870, 5902, 5934, 5958, 5982, 6006, 6030, 6054, 6078, 6102, 6126, 6150, 6174, 6198, 6222, 6246, 6270, 6294, 6318, 6350, 6382, 6414, 6446, 6478, 6510, 6542, 6574, 6599, 6624, 6649, 6674, 6699, 6724, 6749, 6774, 6806, 6838, 6870, 6902, 6934, 6966, 6998, 7030, 7063, 7096, 7129, 7162, 7195, 7228, 7261, 7294, 7326, 7358, 7390, 7422, 7454, 7486, 7518, 7550, 7574, 7598, 7622, 7646, 7670, 7694, 7718, 7742, 7753, 7764, 7775, 7786, 7797, 7808, 7819, 7830, 7841, 7852, 7863, 7874, 7885, 7896, 7907, 7918, 7929, 7940, 7946, 7952, 7958, 7964, 7970, 7976, 7980, 7984, 7990, 7996, 8002, 8008, 8014, 8020, 8024, 8028, 8034, 8040, 8046, 8052, 8058, 8064, 8068, 8072, 8078, 8084, 8090, 8096, 8102, 8108, 8112, 8116, 8122, 8128, 8134, 8140, 8146, 8152, 8156, 8160, 8166, 8172, 8178, 8184, 8190, 8196, 8200, 8204, 8210, 8216, 8222, 8228, 8234, 8240, 8244, 8248, 8254, 8260, 8266, 8272, 8278, 8284, 8288, 8292, 8298, 8304, 8310, 8316, 8322, 8328, 8332, 8336, 8342, 8348, 8354, 8360, 8366, 8372, 8376, 8380, 8386, 8392, 8398, 8404, 8410, 8416, 8420, 8424, 8430, 8436, 8442, 8448, 8454, 8460, 8464, 8468, 8474, 8480, 8486, 8492, 8498, 8504, 8508, 8512, 8518, 8524, 8530, 8536, 8542, 8548, 8552, 8556, 8562, 8568, 8574, 8580, 8586, 8592, 8596, 8600, 8605, 8610, 8615, 8619, 8624, 8627, 8632, 8637, 8642, 8646, 8651, 8654, 8659, 8664, 8669, 8673, 8678, 8681, 8686, 8691, 8696, 8700, 8705, 8708, 8713, 8718, 8723, 8727, 8732, 8735, 8740, 8745, 8750, 8754, 8759, 8762, 8767, 8772, 8777, 8781, 8786, 8789, 8794, 8799, 8804, 8808, 8813, 8816, 8821, 8826, 8831, 8835, 8840, 8843, 8848, 8853, 8858, 8862, 8867, 8870, 8875, 8880, 8885, 8889, 8894, 8897, 8902, 8907, 8912, 8916, 8921, 8924, 8929, 8934, 8939, 8943, 8948, 8951, 8956, 8961, 8966, 8970, 8975, 8978, 8986]
    sp_jac_ini_nia = 564
    sp_jac_ini_nja = 564
    return sp_jac_ini_ia, sp_jac_ini_ja, sp_jac_ini_nia, sp_jac_ini_nja 

def sp_jac_run_vectors():

    sp_jac_run_ia = [0, 1, 563, 1, 14, 149, 150, 151, 152, 153, 154, 479, 480, 481, 485, 486, 487, 2, 3, 563, 3, 14, 165, 166, 167, 168, 169, 170, 491, 492, 493, 497, 498, 499, 4, 5, 563, 5, 14, 197, 198, 199, 200, 201, 202, 503, 504, 505, 509, 510, 511, 6, 7, 563, 7, 14, 141, 142, 143, 144, 145, 146, 515, 516, 517, 521, 522, 523, 8, 9, 563, 9, 14, 301, 302, 303, 304, 305, 306, 527, 528, 529, 533, 534, 535, 10, 11, 563, 11, 14, 317, 318, 319, 320, 321, 322, 539, 540, 541, 545, 546, 547, 12, 13, 563, 13, 14, 245, 246, 247, 248, 249, 250, 551, 552, 553, 557, 558, 559, 563, 15, 16, 17, 18, 19, 20, 21, 22, 135, 136, 139, 140, 149, 150, 151, 152, 153, 154, 155, 156, 359, 15, 16, 17, 18, 19, 20, 21, 22, 135, 136, 139, 140, 149, 150, 151, 152, 153, 154, 155, 156, 360, 15, 16, 17, 18, 19, 20, 21, 22, 135, 136, 137, 138, 149, 150, 151, 152, 153, 154, 155, 156, 361, 15, 16, 17, 18, 19, 20, 21, 22, 135, 136, 137, 138, 149, 150, 151, 152, 153, 154, 155, 156, 362, 15, 16, 17, 18, 19, 20, 21, 22, 137, 138, 139, 140, 149, 150, 151, 152, 153, 154, 155, 156, 363, 15, 16, 17, 18, 19, 20, 21, 22, 137, 138, 139, 140, 149, 150, 151, 152, 153, 154, 155, 156, 364, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 23, 24, 25, 26, 27, 28, 29, 30, 157, 158, 159, 160, 161, 162, 163, 164, 367, 23, 24, 25, 26, 27, 28, 29, 30, 157, 158, 159, 160, 161, 162, 163, 164, 368, 23, 24, 25, 26, 27, 28, 29, 30, 157, 158, 159, 160, 161, 162, 163, 164, 369, 23, 24, 25, 26, 27, 28, 29, 30, 157, 158, 159, 160, 161, 162, 163, 164, 370, 23, 24, 25, 26, 27, 28, 29, 30, 157, 158, 159, 160, 161, 162, 163, 164, 371, 23, 24, 25, 26, 27, 28, 29, 30, 157, 158, 159, 160, 161, 162, 163, 164, 372, 23, 24, 25, 26, 27, 28, 29, 30, 157, 158, 159, 160, 161, 162, 163, 164, 23, 24, 25, 26, 27, 28, 29, 30, 157, 158, 159, 160, 161, 162, 163, 164, 31, 32, 33, 34, 35, 36, 37, 38, 237, 238, 239, 240, 241, 242, 243, 244, 375, 31, 32, 33, 34, 35, 36, 37, 38, 237, 238, 239, 240, 241, 242, 243, 244, 376, 31, 32, 33, 34, 35, 36, 37, 38, 237, 238, 239, 240, 241, 242, 243, 244, 377, 31, 32, 33, 34, 35, 36, 37, 38, 237, 238, 239, 240, 241, 242, 243, 244, 378, 31, 32, 33, 34, 35, 36, 37, 38, 237, 238, 239, 240, 241, 242, 243, 244, 379, 31, 32, 33, 34, 35, 36, 37, 38, 237, 238, 239, 240, 241, 242, 243, 244, 380, 31, 32, 33, 34, 35, 36, 37, 38, 237, 238, 239, 240, 241, 242, 243, 244, 31, 32, 33, 34, 35, 36, 37, 38, 237, 238, 239, 240, 241, 242, 243, 244, 39, 40, 41, 42, 43, 44, 45, 46, 181, 182, 183, 184, 185, 186, 187, 188, 383, 39, 40, 41, 42, 43, 44, 45, 46, 181, 182, 183, 184, 185, 186, 187, 188, 384, 39, 40, 41, 42, 43, 44, 45, 46, 181, 182, 183, 184, 185, 186, 187, 188, 385, 39, 40, 41, 42, 43, 44, 45, 46, 181, 182, 183, 184, 185, 186, 187, 188, 386, 39, 40, 41, 42, 43, 44, 45, 46, 181, 182, 183, 184, 185, 186, 187, 188, 387, 39, 40, 41, 42, 43, 44, 45, 46, 181, 182, 183, 184, 185, 186, 187, 188, 388, 39, 40, 41, 42, 43, 44, 45, 46, 181, 182, 183, 184, 185, 186, 187, 188, 39, 40, 41, 42, 43, 44, 45, 46, 181, 182, 183, 184, 185, 186, 187, 188, 47, 48, 49, 50, 51, 52, 53, 54, 205, 206, 207, 208, 209, 210, 211, 212, 391, 47, 48, 49, 50, 51, 52, 53, 54, 205, 206, 207, 208, 209, 210, 211, 212, 392, 47, 48, 49, 50, 51, 52, 53, 54, 205, 206, 207, 208, 209, 210, 211, 212, 393, 47, 48, 49, 50, 51, 52, 53, 54, 205, 206, 207, 208, 209, 210, 211, 212, 394, 47, 48, 49, 50, 51, 52, 53, 54, 205, 206, 207, 208, 209, 210, 211, 212, 395, 47, 48, 49, 50, 51, 52, 53, 54, 205, 206, 207, 208, 209, 210, 211, 212, 396, 47, 48, 49, 50, 51, 52, 53, 54, 205, 206, 207, 208, 209, 210, 211, 212, 47, 48, 49, 50, 51, 52, 53, 54, 205, 206, 207, 208, 209, 210, 211, 212, 55, 56, 57, 58, 59, 60, 61, 62, 213, 214, 215, 216, 217, 218, 219, 220, 399, 55, 56, 57, 58, 59, 60, 61, 62, 213, 214, 215, 216, 217, 218, 219, 220, 400, 55, 56, 57, 58, 59, 60, 61, 62, 213, 214, 215, 216, 217, 218, 219, 220, 401, 55, 56, 57, 58, 59, 60, 61, 62, 213, 214, 215, 216, 217, 218, 219, 220, 402, 55, 56, 57, 58, 59, 60, 61, 62, 213, 214, 215, 216, 217, 218, 219, 220, 403, 55, 56, 57, 58, 59, 60, 61, 62, 213, 214, 215, 216, 217, 218, 219, 220, 404, 55, 56, 57, 58, 59, 60, 61, 62, 213, 214, 215, 216, 217, 218, 219, 220, 55, 56, 57, 58, 59, 60, 61, 62, 213, 214, 215, 216, 217, 218, 219, 220, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 407, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 408, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 409, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 410, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 411, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 412, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 71, 72, 73, 74, 75, 76, 77, 78, 135, 136, 139, 140, 245, 246, 247, 248, 249, 250, 251, 252, 415, 71, 72, 73, 74, 75, 76, 77, 78, 135, 136, 139, 140, 245, 246, 247, 248, 249, 250, 251, 252, 416, 71, 72, 73, 74, 75, 76, 77, 78, 135, 136, 137, 138, 245, 246, 247, 248, 249, 250, 251, 252, 417, 71, 72, 73, 74, 75, 76, 77, 78, 135, 136, 137, 138, 245, 246, 247, 248, 249, 250, 251, 252, 418, 71, 72, 73, 74, 75, 76, 77, 78, 137, 138, 139, 140, 245, 246, 247, 248, 249, 250, 251, 252, 419, 71, 72, 73, 74, 75, 76, 77, 78, 137, 138, 139, 140, 245, 246, 247, 248, 249, 250, 251, 252, 420, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 79, 80, 81, 82, 83, 84, 85, 86, 317, 318, 319, 320, 321, 322, 323, 324, 423, 79, 80, 81, 82, 83, 84, 85, 86, 317, 318, 319, 320, 321, 322, 323, 324, 424, 79, 80, 81, 82, 83, 84, 85, 86, 317, 318, 319, 320, 321, 322, 323, 324, 425, 79, 80, 81, 82, 83, 84, 85, 86, 317, 318, 319, 320, 321, 322, 323, 324, 426, 79, 80, 81, 82, 83, 84, 85, 86, 317, 318, 319, 320, 321, 322, 323, 324, 427, 79, 80, 81, 82, 83, 84, 85, 86, 317, 318, 319, 320, 321, 322, 323, 324, 428, 79, 80, 81, 82, 83, 84, 85, 86, 317, 318, 319, 320, 321, 322, 323, 324, 79, 80, 81, 82, 83, 84, 85, 86, 317, 318, 319, 320, 321, 322, 323, 324, 87, 88, 89, 90, 91, 92, 93, 94, 317, 318, 319, 320, 321, 322, 323, 324, 431, 87, 88, 89, 90, 91, 92, 93, 94, 317, 318, 319, 320, 321, 322, 323, 324, 432, 87, 88, 89, 90, 91, 92, 93, 94, 317, 318, 319, 320, 321, 322, 323, 324, 433, 87, 88, 89, 90, 91, 92, 93, 94, 317, 318, 319, 320, 321, 322, 323, 324, 434, 87, 88, 89, 90, 91, 92, 93, 94, 317, 318, 319, 320, 321, 322, 323, 324, 435, 87, 88, 89, 90, 91, 92, 93, 94, 317, 318, 319, 320, 321, 322, 323, 324, 436, 87, 88, 89, 90, 91, 92, 93, 94, 317, 318, 319, 320, 321, 322, 323, 324, 87, 88, 89, 90, 91, 92, 93, 94, 317, 318, 319, 320, 321, 322, 323, 324, 95, 96, 97, 98, 99, 100, 101, 102, 309, 310, 311, 312, 313, 314, 315, 316, 439, 95, 96, 97, 98, 99, 100, 101, 102, 309, 310, 311, 312, 313, 314, 315, 316, 440, 95, 96, 97, 98, 99, 100, 101, 102, 309, 310, 311, 312, 313, 314, 315, 316, 441, 95, 96, 97, 98, 99, 100, 101, 102, 309, 310, 311, 312, 313, 314, 315, 316, 442, 95, 96, 97, 98, 99, 100, 101, 102, 309, 310, 311, 312, 313, 314, 315, 316, 443, 95, 96, 97, 98, 99, 100, 101, 102, 309, 310, 311, 312, 313, 314, 315, 316, 444, 95, 96, 97, 98, 99, 100, 101, 102, 309, 310, 311, 312, 313, 314, 315, 316, 95, 96, 97, 98, 99, 100, 101, 102, 309, 310, 311, 312, 313, 314, 315, 316, 103, 104, 105, 106, 107, 108, 109, 110, 333, 334, 335, 336, 337, 338, 339, 340, 447, 103, 104, 105, 106, 107, 108, 109, 110, 333, 334, 335, 336, 337, 338, 339, 340, 448, 103, 104, 105, 106, 107, 108, 109, 110, 333, 334, 335, 336, 337, 338, 339, 340, 449, 103, 104, 105, 106, 107, 108, 109, 110, 333, 334, 335, 336, 337, 338, 339, 340, 450, 103, 104, 105, 106, 107, 108, 109, 110, 333, 334, 335, 336, 337, 338, 339, 340, 451, 103, 104, 105, 106, 107, 108, 109, 110, 333, 334, 335, 336, 337, 338, 339, 340, 452, 103, 104, 105, 106, 107, 108, 109, 110, 333, 334, 335, 336, 337, 338, 339, 340, 103, 104, 105, 106, 107, 108, 109, 110, 333, 334, 335, 336, 337, 338, 339, 340, 111, 112, 113, 114, 115, 116, 117, 118, 325, 326, 327, 328, 329, 330, 331, 332, 455, 111, 112, 113, 114, 115, 116, 117, 118, 325, 326, 327, 328, 329, 330, 331, 332, 456, 111, 112, 113, 114, 115, 116, 117, 118, 325, 326, 327, 328, 329, 330, 331, 332, 457, 111, 112, 113, 114, 115, 116, 117, 118, 325, 326, 327, 328, 329, 330, 331, 332, 458, 111, 112, 113, 114, 115, 116, 117, 118, 325, 326, 327, 328, 329, 330, 331, 332, 459, 111, 112, 113, 114, 115, 116, 117, 118, 325, 326, 327, 328, 329, 330, 331, 332, 460, 111, 112, 113, 114, 115, 116, 117, 118, 325, 326, 327, 328, 329, 330, 331, 332, 111, 112, 113, 114, 115, 116, 117, 118, 325, 326, 327, 328, 329, 330, 331, 332, 119, 120, 121, 122, 123, 124, 125, 126, 293, 294, 295, 296, 297, 298, 299, 300, 463, 119, 120, 121, 122, 123, 124, 125, 126, 293, 294, 295, 296, 297, 298, 299, 300, 464, 119, 120, 121, 122, 123, 124, 125, 126, 293, 294, 295, 296, 297, 298, 299, 300, 465, 119, 120, 121, 122, 123, 124, 125, 126, 293, 294, 295, 296, 297, 298, 299, 300, 466, 119, 120, 121, 122, 123, 124, 125, 126, 293, 294, 295, 296, 297, 298, 299, 300, 467, 119, 120, 121, 122, 123, 124, 125, 126, 293, 294, 295, 296, 297, 298, 299, 300, 468, 119, 120, 121, 122, 123, 124, 125, 126, 293, 294, 295, 296, 297, 298, 299, 300, 119, 120, 121, 122, 123, 124, 125, 126, 293, 294, 295, 296, 297, 298, 299, 300, 127, 128, 129, 130, 131, 132, 133, 134, 301, 302, 303, 304, 305, 306, 307, 308, 471, 127, 128, 129, 130, 131, 132, 133, 134, 301, 302, 303, 304, 305, 306, 307, 308, 472, 127, 128, 129, 130, 131, 132, 133, 134, 301, 302, 303, 304, 305, 306, 307, 308, 473, 127, 128, 129, 130, 131, 132, 133, 134, 301, 302, 303, 304, 305, 306, 307, 308, 474, 127, 128, 129, 130, 131, 132, 133, 134, 301, 302, 303, 304, 305, 306, 307, 308, 475, 127, 128, 129, 130, 131, 132, 133, 134, 301, 302, 303, 304, 305, 306, 307, 308, 476, 127, 128, 129, 130, 131, 132, 133, 134, 301, 302, 303, 304, 305, 306, 307, 308, 127, 128, 129, 130, 131, 132, 133, 134, 301, 302, 303, 304, 305, 306, 307, 308, 15, 16, 17, 18, 71, 72, 73, 74, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 15, 16, 17, 18, 71, 72, 73, 74, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 17, 18, 19, 20, 73, 74, 75, 76, 135, 136, 137, 138, 139, 140, 143, 144, 145, 146, 17, 18, 19, 20, 73, 74, 75, 76, 135, 136, 137, 138, 139, 140, 143, 144, 145, 146, 15, 16, 19, 20, 71, 72, 75, 76, 135, 136, 137, 138, 139, 140, 141, 142, 145, 146, 15, 16, 19, 20, 71, 72, 75, 76, 135, 136, 137, 138, 139, 140, 141, 142, 145, 146, 63, 64, 65, 66, 67, 68, 69, 70, 135, 136, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 515, 63, 64, 65, 66, 67, 68, 69, 70, 135, 136, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 521, 63, 64, 65, 66, 67, 68, 69, 70, 135, 136, 137, 138, 141, 142, 143, 144, 145, 146, 147, 148, 516, 63, 64, 65, 66, 67, 68, 69, 70, 135, 136, 137, 138, 141, 142, 143, 144, 145, 146, 147, 148, 522, 63, 64, 65, 66, 67, 68, 69, 70, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 517, 63, 64, 65, 66, 67, 68, 69, 70, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 523, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 518, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 524, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 479, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 485, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 480, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 486, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 481, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 487, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 482, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 488, 23, 24, 25, 26, 27, 28, 29, 30, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 23, 24, 25, 26, 27, 28, 29, 30, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 23, 24, 25, 26, 27, 28, 29, 30, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 23, 24, 25, 26, 27, 28, 29, 30, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 23, 24, 25, 26, 27, 28, 29, 30, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 23, 24, 25, 26, 27, 28, 29, 30, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 23, 24, 25, 26, 27, 28, 29, 30, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 23, 24, 25, 26, 27, 28, 29, 30, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 221, 222, 223, 224, 225, 226, 227, 228, 491, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 221, 222, 223, 224, 225, 226, 227, 228, 497, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 221, 222, 223, 224, 225, 226, 227, 228, 492, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 221, 222, 223, 224, 225, 226, 227, 228, 498, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 221, 222, 223, 224, 225, 226, 227, 228, 493, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 221, 222, 223, 224, 225, 226, 227, 228, 499, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 221, 222, 223, 224, 225, 226, 227, 228, 494, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 221, 222, 223, 224, 225, 226, 227, 228, 500, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 39, 40, 41, 42, 43, 44, 45, 46, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 39, 40, 41, 42, 43, 44, 45, 46, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 39, 40, 41, 42, 43, 44, 45, 46, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 39, 40, 41, 42, 43, 44, 45, 46, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 39, 40, 41, 42, 43, 44, 45, 46, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 39, 40, 41, 42, 43, 44, 45, 46, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 39, 40, 41, 42, 43, 44, 45, 46, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 39, 40, 41, 42, 43, 44, 45, 46, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 503, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 509, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 504, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 510, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 505, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 511, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 506, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 512, 47, 48, 49, 50, 51, 52, 53, 54, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 47, 48, 49, 50, 51, 52, 53, 54, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 47, 48, 49, 50, 51, 52, 53, 54, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 47, 48, 49, 50, 51, 52, 53, 54, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 47, 48, 49, 50, 51, 52, 53, 54, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 47, 48, 49, 50, 51, 52, 53, 54, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 47, 48, 49, 50, 51, 52, 53, 54, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 47, 48, 49, 50, 51, 52, 53, 54, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 55, 56, 57, 58, 59, 60, 61, 62, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 55, 56, 57, 58, 59, 60, 61, 62, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 55, 56, 57, 58, 59, 60, 61, 62, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 55, 56, 57, 58, 59, 60, 61, 62, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 55, 56, 57, 58, 59, 60, 61, 62, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 55, 56, 57, 58, 59, 60, 61, 62, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 55, 56, 57, 58, 59, 60, 61, 62, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 55, 56, 57, 58, 59, 60, 61, 62, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 165, 166, 167, 168, 169, 170, 171, 172, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 165, 166, 167, 168, 169, 170, 171, 172, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 165, 166, 167, 168, 169, 170, 171, 172, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 165, 166, 167, 168, 169, 170, 171, 172, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 165, 166, 167, 168, 169, 170, 171, 172, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 165, 166, 167, 168, 169, 170, 171, 172, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 165, 166, 167, 168, 169, 170, 171, 172, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 165, 166, 167, 168, 169, 170, 171, 172, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 31, 32, 33, 34, 35, 36, 37, 38, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 31, 32, 33, 34, 35, 36, 37, 38, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 31, 32, 33, 34, 35, 36, 37, 38, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 31, 32, 33, 34, 35, 36, 37, 38, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 31, 32, 33, 34, 35, 36, 37, 38, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 31, 32, 33, 34, 35, 36, 37, 38, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 31, 32, 33, 34, 35, 36, 37, 38, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 31, 32, 33, 34, 35, 36, 37, 38, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 551, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 557, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 552, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 558, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 553, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 559, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 554, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 560, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 309, 310, 311, 312, 313, 314, 315, 316, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 309, 310, 311, 312, 313, 314, 315, 316, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 309, 310, 311, 312, 313, 314, 315, 316, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 309, 310, 311, 312, 313, 314, 315, 316, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 309, 310, 311, 312, 313, 314, 315, 316, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 309, 310, 311, 312, 313, 314, 315, 316, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 309, 310, 311, 312, 313, 314, 315, 316, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 309, 310, 311, 312, 313, 314, 315, 316, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 325, 326, 327, 328, 329, 330, 331, 332, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 325, 326, 327, 328, 329, 330, 331, 332, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 325, 326, 327, 328, 329, 330, 331, 332, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 325, 326, 327, 328, 329, 330, 331, 332, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 325, 326, 327, 328, 329, 330, 331, 332, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 325, 326, 327, 328, 329, 330, 331, 332, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 325, 326, 327, 328, 329, 330, 331, 332, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 325, 326, 327, 328, 329, 330, 331, 332, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 119, 120, 121, 122, 123, 124, 125, 126, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 119, 120, 121, 122, 123, 124, 125, 126, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 119, 120, 121, 122, 123, 124, 125, 126, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 119, 120, 121, 122, 123, 124, 125, 126, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 119, 120, 121, 122, 123, 124, 125, 126, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 119, 120, 121, 122, 123, 124, 125, 126, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 119, 120, 121, 122, 123, 124, 125, 126, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 119, 120, 121, 122, 123, 124, 125, 126, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 127, 128, 129, 130, 131, 132, 133, 134, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 527, 127, 128, 129, 130, 131, 132, 133, 134, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 533, 127, 128, 129, 130, 131, 132, 133, 134, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 528, 127, 128, 129, 130, 131, 132, 133, 134, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 534, 127, 128, 129, 130, 131, 132, 133, 134, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 529, 127, 128, 129, 130, 131, 132, 133, 134, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 535, 127, 128, 129, 130, 131, 132, 133, 134, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 530, 127, 128, 129, 130, 131, 132, 133, 134, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 536, 95, 96, 97, 98, 99, 100, 101, 102, 253, 254, 255, 256, 257, 258, 259, 260, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 95, 96, 97, 98, 99, 100, 101, 102, 253, 254, 255, 256, 257, 258, 259, 260, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 95, 96, 97, 98, 99, 100, 101, 102, 253, 254, 255, 256, 257, 258, 259, 260, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 95, 96, 97, 98, 99, 100, 101, 102, 253, 254, 255, 256, 257, 258, 259, 260, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 95, 96, 97, 98, 99, 100, 101, 102, 253, 254, 255, 256, 257, 258, 259, 260, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 95, 96, 97, 98, 99, 100, 101, 102, 253, 254, 255, 256, 257, 258, 259, 260, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 95, 96, 97, 98, 99, 100, 101, 102, 253, 254, 255, 256, 257, 258, 259, 260, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 95, 96, 97, 98, 99, 100, 101, 102, 253, 254, 255, 256, 257, 258, 259, 260, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 539, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 545, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 540, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 546, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 541, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 547, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 542, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 548, 111, 112, 113, 114, 115, 116, 117, 118, 269, 270, 271, 272, 273, 274, 275, 276, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 111, 112, 113, 114, 115, 116, 117, 118, 269, 270, 271, 272, 273, 274, 275, 276, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 111, 112, 113, 114, 115, 116, 117, 118, 269, 270, 271, 272, 273, 274, 275, 276, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 111, 112, 113, 114, 115, 116, 117, 118, 269, 270, 271, 272, 273, 274, 275, 276, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 111, 112, 113, 114, 115, 116, 117, 118, 269, 270, 271, 272, 273, 274, 275, 276, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 111, 112, 113, 114, 115, 116, 117, 118, 269, 270, 271, 272, 273, 274, 275, 276, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 111, 112, 113, 114, 115, 116, 117, 118, 269, 270, 271, 272, 273, 274, 275, 276, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 111, 112, 113, 114, 115, 116, 117, 118, 269, 270, 271, 272, 273, 274, 275, 276, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 103, 104, 105, 106, 107, 108, 109, 110, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 103, 104, 105, 106, 107, 108, 109, 110, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 103, 104, 105, 106, 107, 108, 109, 110, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 103, 104, 105, 106, 107, 108, 109, 110, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 103, 104, 105, 106, 107, 108, 109, 110, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 103, 104, 105, 106, 107, 108, 109, 110, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 103, 104, 105, 106, 107, 108, 109, 110, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 103, 104, 105, 106, 107, 108, 109, 110, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 15, 16, 17, 18, 135, 136, 137, 138, 139, 140, 341, 15, 16, 17, 18, 135, 136, 137, 138, 139, 140, 342, 17, 18, 19, 20, 135, 136, 137, 138, 139, 140, 343, 17, 18, 19, 20, 135, 136, 137, 138, 139, 140, 344, 15, 16, 19, 20, 135, 136, 137, 138, 139, 140, 345, 15, 16, 19, 20, 135, 136, 137, 138, 139, 140, 346, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 347, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 348, 135, 136, 137, 138, 139, 140, 143, 144, 145, 146, 349, 135, 136, 137, 138, 139, 140, 143, 144, 145, 146, 350, 135, 136, 137, 138, 139, 140, 141, 142, 145, 146, 351, 135, 136, 137, 138, 139, 140, 141, 142, 145, 146, 352, 71, 72, 73, 74, 135, 136, 137, 138, 139, 140, 353, 71, 72, 73, 74, 135, 136, 137, 138, 139, 140, 354, 73, 74, 75, 76, 135, 136, 137, 138, 139, 140, 355, 73, 74, 75, 76, 135, 136, 137, 138, 139, 140, 356, 71, 72, 75, 76, 135, 136, 137, 138, 139, 140, 357, 71, 72, 75, 76, 135, 136, 137, 138, 139, 140, 358, 15, 16, 21, 22, 359, 360, 17, 18, 21, 22, 361, 362, 19, 20, 21, 22, 363, 364, 15, 16, 21, 22, 359, 360, 17, 18, 21, 22, 361, 362, 19, 20, 21, 22, 363, 364, 359, 361, 363, 365, 360, 362, 364, 366, 23, 24, 29, 30, 367, 368, 25, 26, 29, 30, 369, 370, 27, 28, 29, 30, 371, 372, 23, 24, 29, 30, 367, 368, 25, 26, 29, 30, 369, 370, 27, 28, 29, 30, 371, 372, 367, 369, 371, 373, 368, 370, 372, 374, 31, 32, 37, 38, 375, 376, 33, 34, 37, 38, 377, 378, 35, 36, 37, 38, 379, 380, 31, 32, 37, 38, 375, 376, 33, 34, 37, 38, 377, 378, 35, 36, 37, 38, 379, 380, 375, 377, 379, 381, 376, 378, 380, 382, 39, 40, 45, 46, 383, 384, 41, 42, 45, 46, 385, 386, 43, 44, 45, 46, 387, 388, 39, 40, 45, 46, 383, 384, 41, 42, 45, 46, 385, 386, 43, 44, 45, 46, 387, 388, 383, 385, 387, 389, 384, 386, 388, 390, 47, 48, 53, 54, 391, 392, 49, 50, 53, 54, 393, 394, 51, 52, 53, 54, 395, 396, 47, 48, 53, 54, 391, 392, 49, 50, 53, 54, 393, 394, 51, 52, 53, 54, 395, 396, 391, 393, 395, 397, 392, 394, 396, 398, 55, 56, 61, 62, 399, 400, 57, 58, 61, 62, 401, 402, 59, 60, 61, 62, 403, 404, 55, 56, 61, 62, 399, 400, 57, 58, 61, 62, 401, 402, 59, 60, 61, 62, 403, 404, 399, 401, 403, 405, 400, 402, 404, 406, 63, 64, 69, 70, 407, 408, 65, 66, 69, 70, 409, 410, 67, 68, 69, 70, 411, 412, 63, 64, 69, 70, 407, 408, 65, 66, 69, 70, 409, 410, 67, 68, 69, 70, 411, 412, 407, 409, 411, 413, 408, 410, 412, 414, 71, 72, 77, 78, 415, 416, 73, 74, 77, 78, 417, 418, 75, 76, 77, 78, 419, 420, 71, 72, 77, 78, 415, 416, 73, 74, 77, 78, 417, 418, 75, 76, 77, 78, 419, 420, 415, 417, 419, 421, 416, 418, 420, 422, 79, 80, 85, 86, 423, 424, 81, 82, 85, 86, 425, 426, 83, 84, 85, 86, 427, 428, 79, 80, 85, 86, 423, 424, 81, 82, 85, 86, 425, 426, 83, 84, 85, 86, 427, 428, 423, 425, 427, 429, 424, 426, 428, 430, 87, 88, 93, 94, 431, 432, 89, 90, 93, 94, 433, 434, 91, 92, 93, 94, 435, 436, 87, 88, 93, 94, 431, 432, 89, 90, 93, 94, 433, 434, 91, 92, 93, 94, 435, 436, 431, 433, 435, 437, 432, 434, 436, 438, 95, 96, 101, 102, 439, 440, 97, 98, 101, 102, 441, 442, 99, 100, 101, 102, 443, 444, 95, 96, 101, 102, 439, 440, 97, 98, 101, 102, 441, 442, 99, 100, 101, 102, 443, 444, 439, 441, 443, 445, 440, 442, 444, 446, 103, 104, 109, 110, 447, 448, 105, 106, 109, 110, 449, 450, 107, 108, 109, 110, 451, 452, 103, 104, 109, 110, 447, 448, 105, 106, 109, 110, 449, 450, 107, 108, 109, 110, 451, 452, 447, 449, 451, 453, 448, 450, 452, 454, 111, 112, 117, 118, 455, 456, 113, 114, 117, 118, 457, 458, 115, 116, 117, 118, 459, 460, 111, 112, 117, 118, 455, 456, 113, 114, 117, 118, 457, 458, 115, 116, 117, 118, 459, 460, 455, 457, 459, 461, 456, 458, 460, 462, 119, 120, 125, 126, 463, 464, 121, 122, 125, 126, 465, 466, 123, 124, 125, 126, 467, 468, 119, 120, 125, 126, 463, 464, 121, 122, 125, 126, 465, 466, 123, 124, 125, 126, 467, 468, 463, 465, 467, 469, 464, 466, 468, 470, 127, 128, 133, 134, 471, 472, 129, 130, 133, 134, 473, 474, 131, 132, 133, 134, 475, 476, 127, 128, 133, 134, 471, 472, 129, 130, 133, 134, 473, 474, 131, 132, 133, 134, 475, 476, 471, 473, 475, 477, 472, 474, 476, 478, 0, 149, 155, 479, 485, 0, 151, 155, 480, 486, 0, 153, 155, 481, 487, 155, 482, 484, 488, 479, 480, 481, 482, 483, 483, 484, 489, 0, 150, 156, 479, 485, 0, 152, 156, 480, 486, 0, 154, 156, 481, 487, 156, 482, 488, 490, 485, 486, 487, 488, 489, 483, 489, 490, 2, 165, 171, 491, 497, 2, 167, 171, 492, 498, 2, 169, 171, 493, 499, 171, 494, 496, 500, 491, 492, 493, 494, 495, 495, 496, 501, 2, 166, 172, 491, 497, 2, 168, 172, 492, 498, 2, 170, 172, 493, 499, 172, 494, 500, 502, 497, 498, 499, 500, 501, 495, 501, 502, 4, 197, 203, 503, 509, 4, 199, 203, 504, 510, 4, 201, 203, 505, 511, 203, 506, 508, 512, 503, 504, 505, 506, 507, 507, 508, 513, 4, 198, 204, 503, 509, 4, 200, 204, 504, 510, 4, 202, 204, 505, 511, 204, 506, 512, 514, 509, 510, 511, 512, 513, 507, 513, 514, 6, 141, 147, 515, 521, 6, 143, 147, 516, 522, 6, 145, 147, 517, 523, 147, 518, 520, 524, 515, 516, 517, 518, 519, 519, 520, 525, 6, 142, 148, 515, 521, 6, 144, 148, 516, 522, 6, 146, 148, 517, 523, 148, 518, 524, 526, 521, 522, 523, 524, 525, 519, 525, 526, 8, 301, 307, 527, 533, 8, 303, 307, 528, 534, 8, 305, 307, 529, 535, 307, 530, 532, 536, 527, 528, 529, 530, 531, 531, 532, 537, 8, 302, 308, 527, 533, 8, 304, 308, 528, 534, 8, 306, 308, 529, 535, 308, 530, 536, 538, 533, 534, 535, 536, 537, 531, 537, 538, 10, 317, 323, 539, 545, 10, 319, 323, 540, 546, 10, 321, 323, 541, 547, 323, 542, 544, 548, 539, 540, 541, 542, 543, 543, 544, 549, 10, 318, 324, 539, 545, 10, 320, 324, 540, 546, 10, 322, 324, 541, 547, 324, 542, 548, 550, 545, 546, 547, 548, 549, 543, 549, 550, 12, 245, 251, 551, 557, 12, 247, 251, 552, 558, 12, 249, 251, 553, 559, 251, 554, 556, 560, 551, 552, 553, 554, 555, 555, 556, 561, 12, 246, 252, 551, 557, 12, 248, 252, 552, 558, 12, 250, 252, 553, 559, 252, 554, 560, 562, 557, 558, 559, 560, 561, 555, 561, 562, 1, 3, 5, 7, 9, 11, 13, 563]
    sp_jac_run_ja = [0, 3, 17, 20, 34, 37, 51, 54, 68, 71, 85, 88, 102, 105, 119, 120, 141, 162, 183, 204, 225, 246, 262, 278, 295, 312, 329, 346, 363, 380, 396, 412, 429, 446, 463, 480, 497, 514, 530, 546, 563, 580, 597, 614, 631, 648, 664, 680, 697, 714, 731, 748, 765, 782, 798, 814, 831, 848, 865, 882, 899, 916, 932, 948, 965, 982, 999, 1016, 1033, 1050, 1066, 1082, 1103, 1124, 1145, 1166, 1187, 1208, 1224, 1240, 1257, 1274, 1291, 1308, 1325, 1342, 1358, 1374, 1391, 1408, 1425, 1442, 1459, 1476, 1492, 1508, 1525, 1542, 1559, 1576, 1593, 1610, 1626, 1642, 1659, 1676, 1693, 1710, 1727, 1744, 1760, 1776, 1793, 1810, 1827, 1844, 1861, 1878, 1894, 1910, 1927, 1944, 1961, 1978, 1995, 2012, 2028, 2044, 2061, 2078, 2095, 2112, 2129, 2146, 2162, 2178, 2196, 2214, 2232, 2250, 2268, 2286, 2307, 2328, 2349, 2370, 2391, 2412, 2429, 2446, 2471, 2496, 2521, 2546, 2571, 2596, 2621, 2646, 2678, 2710, 2742, 2774, 2806, 2838, 2870, 2902, 2935, 2968, 3001, 3034, 3067, 3100, 3133, 3166, 3190, 3214, 3238, 3262, 3286, 3310, 3334, 3358, 3390, 3422, 3454, 3486, 3518, 3550, 3582, 3614, 3638, 3662, 3686, 3710, 3734, 3758, 3782, 3806, 3831, 3856, 3881, 3906, 3931, 3956, 3981, 4006, 4038, 4070, 4102, 4134, 4166, 4198, 4230, 4262, 4286, 4310, 4334, 4358, 4382, 4406, 4430, 4454, 4478, 4502, 4526, 4550, 4574, 4598, 4622, 4646, 4670, 4694, 4718, 4742, 4766, 4790, 4814, 4838, 4862, 4886, 4910, 4934, 4958, 4982, 5006, 5030, 5055, 5080, 5105, 5130, 5155, 5180, 5205, 5230, 5262, 5294, 5326, 5358, 5390, 5422, 5454, 5486, 5510, 5534, 5558, 5582, 5606, 5630, 5654, 5678, 5710, 5742, 5774, 5806, 5838, 5870, 5902, 5934, 5958, 5982, 6006, 6030, 6054, 6078, 6102, 6126, 6150, 6174, 6198, 6222, 6246, 6270, 6294, 6318, 6350, 6382, 6414, 6446, 6478, 6510, 6542, 6574, 6599, 6624, 6649, 6674, 6699, 6724, 6749, 6774, 6806, 6838, 6870, 6902, 6934, 6966, 6998, 7030, 7063, 7096, 7129, 7162, 7195, 7228, 7261, 7294, 7326, 7358, 7390, 7422, 7454, 7486, 7518, 7550, 7574, 7598, 7622, 7646, 7670, 7694, 7718, 7742, 7753, 7764, 7775, 7786, 7797, 7808, 7819, 7830, 7841, 7852, 7863, 7874, 7885, 7896, 7907, 7918, 7929, 7940, 7946, 7952, 7958, 7964, 7970, 7976, 7980, 7984, 7990, 7996, 8002, 8008, 8014, 8020, 8024, 8028, 8034, 8040, 8046, 8052, 8058, 8064, 8068, 8072, 8078, 8084, 8090, 8096, 8102, 8108, 8112, 8116, 8122, 8128, 8134, 8140, 8146, 8152, 8156, 8160, 8166, 8172, 8178, 8184, 8190, 8196, 8200, 8204, 8210, 8216, 8222, 8228, 8234, 8240, 8244, 8248, 8254, 8260, 8266, 8272, 8278, 8284, 8288, 8292, 8298, 8304, 8310, 8316, 8322, 8328, 8332, 8336, 8342, 8348, 8354, 8360, 8366, 8372, 8376, 8380, 8386, 8392, 8398, 8404, 8410, 8416, 8420, 8424, 8430, 8436, 8442, 8448, 8454, 8460, 8464, 8468, 8474, 8480, 8486, 8492, 8498, 8504, 8508, 8512, 8518, 8524, 8530, 8536, 8542, 8548, 8552, 8556, 8562, 8568, 8574, 8580, 8586, 8592, 8596, 8600, 8605, 8610, 8615, 8619, 8624, 8627, 8632, 8637, 8642, 8646, 8651, 8654, 8659, 8664, 8669, 8673, 8678, 8681, 8686, 8691, 8696, 8700, 8705, 8708, 8713, 8718, 8723, 8727, 8732, 8735, 8740, 8745, 8750, 8754, 8759, 8762, 8767, 8772, 8777, 8781, 8786, 8789, 8794, 8799, 8804, 8808, 8813, 8816, 8821, 8826, 8831, 8835, 8840, 8843, 8848, 8853, 8858, 8862, 8867, 8870, 8875, 8880, 8885, 8889, 8894, 8897, 8902, 8907, 8912, 8916, 8921, 8924, 8929, 8934, 8939, 8943, 8948, 8951, 8956, 8961, 8966, 8970, 8975, 8978, 8986]
    sp_jac_run_nia = 564
    sp_jac_run_nja = 564
    return sp_jac_run_ia, sp_jac_run_ja, sp_jac_run_nia, sp_jac_run_nja 

def sp_jac_trap_vectors():

    sp_jac_trap_ia = [0, 1, 563, 1, 14, 149, 150, 151, 152, 153, 154, 479, 480, 481, 485, 486, 487, 2, 3, 563, 3, 14, 165, 166, 167, 168, 169, 170, 491, 492, 493, 497, 498, 499, 4, 5, 563, 5, 14, 197, 198, 199, 200, 201, 202, 503, 504, 505, 509, 510, 511, 6, 7, 563, 7, 14, 141, 142, 143, 144, 145, 146, 515, 516, 517, 521, 522, 523, 8, 9, 563, 9, 14, 301, 302, 303, 304, 305, 306, 527, 528, 529, 533, 534, 535, 10, 11, 563, 11, 14, 317, 318, 319, 320, 321, 322, 539, 540, 541, 545, 546, 547, 12, 13, 563, 13, 14, 245, 246, 247, 248, 249, 250, 551, 552, 553, 557, 558, 559, 14, 563, 15, 16, 17, 18, 19, 20, 21, 22, 135, 136, 139, 140, 149, 150, 151, 152, 153, 154, 155, 156, 359, 15, 16, 17, 18, 19, 20, 21, 22, 135, 136, 139, 140, 149, 150, 151, 152, 153, 154, 155, 156, 360, 15, 16, 17, 18, 19, 20, 21, 22, 135, 136, 137, 138, 149, 150, 151, 152, 153, 154, 155, 156, 361, 15, 16, 17, 18, 19, 20, 21, 22, 135, 136, 137, 138, 149, 150, 151, 152, 153, 154, 155, 156, 362, 15, 16, 17, 18, 19, 20, 21, 22, 137, 138, 139, 140, 149, 150, 151, 152, 153, 154, 155, 156, 363, 15, 16, 17, 18, 19, 20, 21, 22, 137, 138, 139, 140, 149, 150, 151, 152, 153, 154, 155, 156, 364, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 23, 24, 25, 26, 27, 28, 29, 30, 157, 158, 159, 160, 161, 162, 163, 164, 367, 23, 24, 25, 26, 27, 28, 29, 30, 157, 158, 159, 160, 161, 162, 163, 164, 368, 23, 24, 25, 26, 27, 28, 29, 30, 157, 158, 159, 160, 161, 162, 163, 164, 369, 23, 24, 25, 26, 27, 28, 29, 30, 157, 158, 159, 160, 161, 162, 163, 164, 370, 23, 24, 25, 26, 27, 28, 29, 30, 157, 158, 159, 160, 161, 162, 163, 164, 371, 23, 24, 25, 26, 27, 28, 29, 30, 157, 158, 159, 160, 161, 162, 163, 164, 372, 23, 24, 25, 26, 27, 28, 29, 30, 157, 158, 159, 160, 161, 162, 163, 164, 23, 24, 25, 26, 27, 28, 29, 30, 157, 158, 159, 160, 161, 162, 163, 164, 31, 32, 33, 34, 35, 36, 37, 38, 237, 238, 239, 240, 241, 242, 243, 244, 375, 31, 32, 33, 34, 35, 36, 37, 38, 237, 238, 239, 240, 241, 242, 243, 244, 376, 31, 32, 33, 34, 35, 36, 37, 38, 237, 238, 239, 240, 241, 242, 243, 244, 377, 31, 32, 33, 34, 35, 36, 37, 38, 237, 238, 239, 240, 241, 242, 243, 244, 378, 31, 32, 33, 34, 35, 36, 37, 38, 237, 238, 239, 240, 241, 242, 243, 244, 379, 31, 32, 33, 34, 35, 36, 37, 38, 237, 238, 239, 240, 241, 242, 243, 244, 380, 31, 32, 33, 34, 35, 36, 37, 38, 237, 238, 239, 240, 241, 242, 243, 244, 31, 32, 33, 34, 35, 36, 37, 38, 237, 238, 239, 240, 241, 242, 243, 244, 39, 40, 41, 42, 43, 44, 45, 46, 181, 182, 183, 184, 185, 186, 187, 188, 383, 39, 40, 41, 42, 43, 44, 45, 46, 181, 182, 183, 184, 185, 186, 187, 188, 384, 39, 40, 41, 42, 43, 44, 45, 46, 181, 182, 183, 184, 185, 186, 187, 188, 385, 39, 40, 41, 42, 43, 44, 45, 46, 181, 182, 183, 184, 185, 186, 187, 188, 386, 39, 40, 41, 42, 43, 44, 45, 46, 181, 182, 183, 184, 185, 186, 187, 188, 387, 39, 40, 41, 42, 43, 44, 45, 46, 181, 182, 183, 184, 185, 186, 187, 188, 388, 39, 40, 41, 42, 43, 44, 45, 46, 181, 182, 183, 184, 185, 186, 187, 188, 39, 40, 41, 42, 43, 44, 45, 46, 181, 182, 183, 184, 185, 186, 187, 188, 47, 48, 49, 50, 51, 52, 53, 54, 205, 206, 207, 208, 209, 210, 211, 212, 391, 47, 48, 49, 50, 51, 52, 53, 54, 205, 206, 207, 208, 209, 210, 211, 212, 392, 47, 48, 49, 50, 51, 52, 53, 54, 205, 206, 207, 208, 209, 210, 211, 212, 393, 47, 48, 49, 50, 51, 52, 53, 54, 205, 206, 207, 208, 209, 210, 211, 212, 394, 47, 48, 49, 50, 51, 52, 53, 54, 205, 206, 207, 208, 209, 210, 211, 212, 395, 47, 48, 49, 50, 51, 52, 53, 54, 205, 206, 207, 208, 209, 210, 211, 212, 396, 47, 48, 49, 50, 51, 52, 53, 54, 205, 206, 207, 208, 209, 210, 211, 212, 47, 48, 49, 50, 51, 52, 53, 54, 205, 206, 207, 208, 209, 210, 211, 212, 55, 56, 57, 58, 59, 60, 61, 62, 213, 214, 215, 216, 217, 218, 219, 220, 399, 55, 56, 57, 58, 59, 60, 61, 62, 213, 214, 215, 216, 217, 218, 219, 220, 400, 55, 56, 57, 58, 59, 60, 61, 62, 213, 214, 215, 216, 217, 218, 219, 220, 401, 55, 56, 57, 58, 59, 60, 61, 62, 213, 214, 215, 216, 217, 218, 219, 220, 402, 55, 56, 57, 58, 59, 60, 61, 62, 213, 214, 215, 216, 217, 218, 219, 220, 403, 55, 56, 57, 58, 59, 60, 61, 62, 213, 214, 215, 216, 217, 218, 219, 220, 404, 55, 56, 57, 58, 59, 60, 61, 62, 213, 214, 215, 216, 217, 218, 219, 220, 55, 56, 57, 58, 59, 60, 61, 62, 213, 214, 215, 216, 217, 218, 219, 220, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 407, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 408, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 409, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 410, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 411, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 412, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 71, 72, 73, 74, 75, 76, 77, 78, 135, 136, 139, 140, 245, 246, 247, 248, 249, 250, 251, 252, 415, 71, 72, 73, 74, 75, 76, 77, 78, 135, 136, 139, 140, 245, 246, 247, 248, 249, 250, 251, 252, 416, 71, 72, 73, 74, 75, 76, 77, 78, 135, 136, 137, 138, 245, 246, 247, 248, 249, 250, 251, 252, 417, 71, 72, 73, 74, 75, 76, 77, 78, 135, 136, 137, 138, 245, 246, 247, 248, 249, 250, 251, 252, 418, 71, 72, 73, 74, 75, 76, 77, 78, 137, 138, 139, 140, 245, 246, 247, 248, 249, 250, 251, 252, 419, 71, 72, 73, 74, 75, 76, 77, 78, 137, 138, 139, 140, 245, 246, 247, 248, 249, 250, 251, 252, 420, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 79, 80, 81, 82, 83, 84, 85, 86, 317, 318, 319, 320, 321, 322, 323, 324, 423, 79, 80, 81, 82, 83, 84, 85, 86, 317, 318, 319, 320, 321, 322, 323, 324, 424, 79, 80, 81, 82, 83, 84, 85, 86, 317, 318, 319, 320, 321, 322, 323, 324, 425, 79, 80, 81, 82, 83, 84, 85, 86, 317, 318, 319, 320, 321, 322, 323, 324, 426, 79, 80, 81, 82, 83, 84, 85, 86, 317, 318, 319, 320, 321, 322, 323, 324, 427, 79, 80, 81, 82, 83, 84, 85, 86, 317, 318, 319, 320, 321, 322, 323, 324, 428, 79, 80, 81, 82, 83, 84, 85, 86, 317, 318, 319, 320, 321, 322, 323, 324, 79, 80, 81, 82, 83, 84, 85, 86, 317, 318, 319, 320, 321, 322, 323, 324, 87, 88, 89, 90, 91, 92, 93, 94, 317, 318, 319, 320, 321, 322, 323, 324, 431, 87, 88, 89, 90, 91, 92, 93, 94, 317, 318, 319, 320, 321, 322, 323, 324, 432, 87, 88, 89, 90, 91, 92, 93, 94, 317, 318, 319, 320, 321, 322, 323, 324, 433, 87, 88, 89, 90, 91, 92, 93, 94, 317, 318, 319, 320, 321, 322, 323, 324, 434, 87, 88, 89, 90, 91, 92, 93, 94, 317, 318, 319, 320, 321, 322, 323, 324, 435, 87, 88, 89, 90, 91, 92, 93, 94, 317, 318, 319, 320, 321, 322, 323, 324, 436, 87, 88, 89, 90, 91, 92, 93, 94, 317, 318, 319, 320, 321, 322, 323, 324, 87, 88, 89, 90, 91, 92, 93, 94, 317, 318, 319, 320, 321, 322, 323, 324, 95, 96, 97, 98, 99, 100, 101, 102, 309, 310, 311, 312, 313, 314, 315, 316, 439, 95, 96, 97, 98, 99, 100, 101, 102, 309, 310, 311, 312, 313, 314, 315, 316, 440, 95, 96, 97, 98, 99, 100, 101, 102, 309, 310, 311, 312, 313, 314, 315, 316, 441, 95, 96, 97, 98, 99, 100, 101, 102, 309, 310, 311, 312, 313, 314, 315, 316, 442, 95, 96, 97, 98, 99, 100, 101, 102, 309, 310, 311, 312, 313, 314, 315, 316, 443, 95, 96, 97, 98, 99, 100, 101, 102, 309, 310, 311, 312, 313, 314, 315, 316, 444, 95, 96, 97, 98, 99, 100, 101, 102, 309, 310, 311, 312, 313, 314, 315, 316, 95, 96, 97, 98, 99, 100, 101, 102, 309, 310, 311, 312, 313, 314, 315, 316, 103, 104, 105, 106, 107, 108, 109, 110, 333, 334, 335, 336, 337, 338, 339, 340, 447, 103, 104, 105, 106, 107, 108, 109, 110, 333, 334, 335, 336, 337, 338, 339, 340, 448, 103, 104, 105, 106, 107, 108, 109, 110, 333, 334, 335, 336, 337, 338, 339, 340, 449, 103, 104, 105, 106, 107, 108, 109, 110, 333, 334, 335, 336, 337, 338, 339, 340, 450, 103, 104, 105, 106, 107, 108, 109, 110, 333, 334, 335, 336, 337, 338, 339, 340, 451, 103, 104, 105, 106, 107, 108, 109, 110, 333, 334, 335, 336, 337, 338, 339, 340, 452, 103, 104, 105, 106, 107, 108, 109, 110, 333, 334, 335, 336, 337, 338, 339, 340, 103, 104, 105, 106, 107, 108, 109, 110, 333, 334, 335, 336, 337, 338, 339, 340, 111, 112, 113, 114, 115, 116, 117, 118, 325, 326, 327, 328, 329, 330, 331, 332, 455, 111, 112, 113, 114, 115, 116, 117, 118, 325, 326, 327, 328, 329, 330, 331, 332, 456, 111, 112, 113, 114, 115, 116, 117, 118, 325, 326, 327, 328, 329, 330, 331, 332, 457, 111, 112, 113, 114, 115, 116, 117, 118, 325, 326, 327, 328, 329, 330, 331, 332, 458, 111, 112, 113, 114, 115, 116, 117, 118, 325, 326, 327, 328, 329, 330, 331, 332, 459, 111, 112, 113, 114, 115, 116, 117, 118, 325, 326, 327, 328, 329, 330, 331, 332, 460, 111, 112, 113, 114, 115, 116, 117, 118, 325, 326, 327, 328, 329, 330, 331, 332, 111, 112, 113, 114, 115, 116, 117, 118, 325, 326, 327, 328, 329, 330, 331, 332, 119, 120, 121, 122, 123, 124, 125, 126, 293, 294, 295, 296, 297, 298, 299, 300, 463, 119, 120, 121, 122, 123, 124, 125, 126, 293, 294, 295, 296, 297, 298, 299, 300, 464, 119, 120, 121, 122, 123, 124, 125, 126, 293, 294, 295, 296, 297, 298, 299, 300, 465, 119, 120, 121, 122, 123, 124, 125, 126, 293, 294, 295, 296, 297, 298, 299, 300, 466, 119, 120, 121, 122, 123, 124, 125, 126, 293, 294, 295, 296, 297, 298, 299, 300, 467, 119, 120, 121, 122, 123, 124, 125, 126, 293, 294, 295, 296, 297, 298, 299, 300, 468, 119, 120, 121, 122, 123, 124, 125, 126, 293, 294, 295, 296, 297, 298, 299, 300, 119, 120, 121, 122, 123, 124, 125, 126, 293, 294, 295, 296, 297, 298, 299, 300, 127, 128, 129, 130, 131, 132, 133, 134, 301, 302, 303, 304, 305, 306, 307, 308, 471, 127, 128, 129, 130, 131, 132, 133, 134, 301, 302, 303, 304, 305, 306, 307, 308, 472, 127, 128, 129, 130, 131, 132, 133, 134, 301, 302, 303, 304, 305, 306, 307, 308, 473, 127, 128, 129, 130, 131, 132, 133, 134, 301, 302, 303, 304, 305, 306, 307, 308, 474, 127, 128, 129, 130, 131, 132, 133, 134, 301, 302, 303, 304, 305, 306, 307, 308, 475, 127, 128, 129, 130, 131, 132, 133, 134, 301, 302, 303, 304, 305, 306, 307, 308, 476, 127, 128, 129, 130, 131, 132, 133, 134, 301, 302, 303, 304, 305, 306, 307, 308, 127, 128, 129, 130, 131, 132, 133, 134, 301, 302, 303, 304, 305, 306, 307, 308, 15, 16, 17, 18, 71, 72, 73, 74, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 15, 16, 17, 18, 71, 72, 73, 74, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 17, 18, 19, 20, 73, 74, 75, 76, 135, 136, 137, 138, 139, 140, 143, 144, 145, 146, 17, 18, 19, 20, 73, 74, 75, 76, 135, 136, 137, 138, 139, 140, 143, 144, 145, 146, 15, 16, 19, 20, 71, 72, 75, 76, 135, 136, 137, 138, 139, 140, 141, 142, 145, 146, 15, 16, 19, 20, 71, 72, 75, 76, 135, 136, 137, 138, 139, 140, 141, 142, 145, 146, 63, 64, 65, 66, 67, 68, 69, 70, 135, 136, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 515, 63, 64, 65, 66, 67, 68, 69, 70, 135, 136, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 521, 63, 64, 65, 66, 67, 68, 69, 70, 135, 136, 137, 138, 141, 142, 143, 144, 145, 146, 147, 148, 516, 63, 64, 65, 66, 67, 68, 69, 70, 135, 136, 137, 138, 141, 142, 143, 144, 145, 146, 147, 148, 522, 63, 64, 65, 66, 67, 68, 69, 70, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 517, 63, 64, 65, 66, 67, 68, 69, 70, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 523, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 518, 63, 64, 65, 66, 67, 68, 69, 70, 141, 142, 143, 144, 145, 146, 147, 148, 524, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 479, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 485, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 480, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 486, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 481, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 487, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 482, 15, 16, 17, 18, 19, 20, 21, 22, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 488, 23, 24, 25, 26, 27, 28, 29, 30, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 23, 24, 25, 26, 27, 28, 29, 30, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 23, 24, 25, 26, 27, 28, 29, 30, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 23, 24, 25, 26, 27, 28, 29, 30, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 23, 24, 25, 26, 27, 28, 29, 30, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 23, 24, 25, 26, 27, 28, 29, 30, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 23, 24, 25, 26, 27, 28, 29, 30, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 23, 24, 25, 26, 27, 28, 29, 30, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 221, 222, 223, 224, 225, 226, 227, 228, 491, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 221, 222, 223, 224, 225, 226, 227, 228, 497, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 221, 222, 223, 224, 225, 226, 227, 228, 492, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 221, 222, 223, 224, 225, 226, 227, 228, 498, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 221, 222, 223, 224, 225, 226, 227, 228, 493, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 221, 222, 223, 224, 225, 226, 227, 228, 499, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 221, 222, 223, 224, 225, 226, 227, 228, 494, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 221, 222, 223, 224, 225, 226, 227, 228, 500, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 39, 40, 41, 42, 43, 44, 45, 46, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 39, 40, 41, 42, 43, 44, 45, 46, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 39, 40, 41, 42, 43, 44, 45, 46, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 39, 40, 41, 42, 43, 44, 45, 46, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 39, 40, 41, 42, 43, 44, 45, 46, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 39, 40, 41, 42, 43, 44, 45, 46, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 39, 40, 41, 42, 43, 44, 45, 46, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 39, 40, 41, 42, 43, 44, 45, 46, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 503, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 509, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 504, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 510, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 505, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 511, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 506, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 512, 47, 48, 49, 50, 51, 52, 53, 54, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 47, 48, 49, 50, 51, 52, 53, 54, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 47, 48, 49, 50, 51, 52, 53, 54, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 47, 48, 49, 50, 51, 52, 53, 54, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 47, 48, 49, 50, 51, 52, 53, 54, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 47, 48, 49, 50, 51, 52, 53, 54, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 47, 48, 49, 50, 51, 52, 53, 54, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 47, 48, 49, 50, 51, 52, 53, 54, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 55, 56, 57, 58, 59, 60, 61, 62, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 55, 56, 57, 58, 59, 60, 61, 62, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 55, 56, 57, 58, 59, 60, 61, 62, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 55, 56, 57, 58, 59, 60, 61, 62, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 55, 56, 57, 58, 59, 60, 61, 62, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 55, 56, 57, 58, 59, 60, 61, 62, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 55, 56, 57, 58, 59, 60, 61, 62, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 55, 56, 57, 58, 59, 60, 61, 62, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 165, 166, 167, 168, 169, 170, 171, 172, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 165, 166, 167, 168, 169, 170, 171, 172, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 165, 166, 167, 168, 169, 170, 171, 172, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 165, 166, 167, 168, 169, 170, 171, 172, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 165, 166, 167, 168, 169, 170, 171, 172, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 165, 166, 167, 168, 169, 170, 171, 172, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 165, 166, 167, 168, 169, 170, 171, 172, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 165, 166, 167, 168, 169, 170, 171, 172, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 31, 32, 33, 34, 35, 36, 37, 38, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 31, 32, 33, 34, 35, 36, 37, 38, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 31, 32, 33, 34, 35, 36, 37, 38, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 31, 32, 33, 34, 35, 36, 37, 38, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 31, 32, 33, 34, 35, 36, 37, 38, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 31, 32, 33, 34, 35, 36, 37, 38, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 31, 32, 33, 34, 35, 36, 37, 38, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 31, 32, 33, 34, 35, 36, 37, 38, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 551, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 557, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 552, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 558, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 553, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 559, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 554, 71, 72, 73, 74, 75, 76, 77, 78, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 560, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 309, 310, 311, 312, 313, 314, 315, 316, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 309, 310, 311, 312, 313, 314, 315, 316, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 309, 310, 311, 312, 313, 314, 315, 316, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 309, 310, 311, 312, 313, 314, 315, 316, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 309, 310, 311, 312, 313, 314, 315, 316, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 309, 310, 311, 312, 313, 314, 315, 316, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 309, 310, 311, 312, 313, 314, 315, 316, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 309, 310, 311, 312, 313, 314, 315, 316, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 325, 326, 327, 328, 329, 330, 331, 332, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 325, 326, 327, 328, 329, 330, 331, 332, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 325, 326, 327, 328, 329, 330, 331, 332, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 325, 326, 327, 328, 329, 330, 331, 332, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 325, 326, 327, 328, 329, 330, 331, 332, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 325, 326, 327, 328, 329, 330, 331, 332, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 325, 326, 327, 328, 329, 330, 331, 332, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 325, 326, 327, 328, 329, 330, 331, 332, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 119, 120, 121, 122, 123, 124, 125, 126, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 119, 120, 121, 122, 123, 124, 125, 126, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 119, 120, 121, 122, 123, 124, 125, 126, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 119, 120, 121, 122, 123, 124, 125, 126, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 119, 120, 121, 122, 123, 124, 125, 126, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 119, 120, 121, 122, 123, 124, 125, 126, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 119, 120, 121, 122, 123, 124, 125, 126, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 119, 120, 121, 122, 123, 124, 125, 126, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 127, 128, 129, 130, 131, 132, 133, 134, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 527, 127, 128, 129, 130, 131, 132, 133, 134, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 533, 127, 128, 129, 130, 131, 132, 133, 134, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 528, 127, 128, 129, 130, 131, 132, 133, 134, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 534, 127, 128, 129, 130, 131, 132, 133, 134, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 529, 127, 128, 129, 130, 131, 132, 133, 134, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 535, 127, 128, 129, 130, 131, 132, 133, 134, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 530, 127, 128, 129, 130, 131, 132, 133, 134, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 536, 95, 96, 97, 98, 99, 100, 101, 102, 253, 254, 255, 256, 257, 258, 259, 260, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 95, 96, 97, 98, 99, 100, 101, 102, 253, 254, 255, 256, 257, 258, 259, 260, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 95, 96, 97, 98, 99, 100, 101, 102, 253, 254, 255, 256, 257, 258, 259, 260, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 95, 96, 97, 98, 99, 100, 101, 102, 253, 254, 255, 256, 257, 258, 259, 260, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 95, 96, 97, 98, 99, 100, 101, 102, 253, 254, 255, 256, 257, 258, 259, 260, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 95, 96, 97, 98, 99, 100, 101, 102, 253, 254, 255, 256, 257, 258, 259, 260, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 95, 96, 97, 98, 99, 100, 101, 102, 253, 254, 255, 256, 257, 258, 259, 260, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 95, 96, 97, 98, 99, 100, 101, 102, 253, 254, 255, 256, 257, 258, 259, 260, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 539, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 545, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 540, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 546, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 541, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 547, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 542, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 548, 111, 112, 113, 114, 115, 116, 117, 118, 269, 270, 271, 272, 273, 274, 275, 276, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 111, 112, 113, 114, 115, 116, 117, 118, 269, 270, 271, 272, 273, 274, 275, 276, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 111, 112, 113, 114, 115, 116, 117, 118, 269, 270, 271, 272, 273, 274, 275, 276, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 111, 112, 113, 114, 115, 116, 117, 118, 269, 270, 271, 272, 273, 274, 275, 276, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 111, 112, 113, 114, 115, 116, 117, 118, 269, 270, 271, 272, 273, 274, 275, 276, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 111, 112, 113, 114, 115, 116, 117, 118, 269, 270, 271, 272, 273, 274, 275, 276, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 111, 112, 113, 114, 115, 116, 117, 118, 269, 270, 271, 272, 273, 274, 275, 276, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 111, 112, 113, 114, 115, 116, 117, 118, 269, 270, 271, 272, 273, 274, 275, 276, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 103, 104, 105, 106, 107, 108, 109, 110, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 103, 104, 105, 106, 107, 108, 109, 110, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 103, 104, 105, 106, 107, 108, 109, 110, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 103, 104, 105, 106, 107, 108, 109, 110, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 103, 104, 105, 106, 107, 108, 109, 110, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 103, 104, 105, 106, 107, 108, 109, 110, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 103, 104, 105, 106, 107, 108, 109, 110, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 103, 104, 105, 106, 107, 108, 109, 110, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 15, 16, 17, 18, 135, 136, 137, 138, 139, 140, 341, 15, 16, 17, 18, 135, 136, 137, 138, 139, 140, 342, 17, 18, 19, 20, 135, 136, 137, 138, 139, 140, 343, 17, 18, 19, 20, 135, 136, 137, 138, 139, 140, 344, 15, 16, 19, 20, 135, 136, 137, 138, 139, 140, 345, 15, 16, 19, 20, 135, 136, 137, 138, 139, 140, 346, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 347, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 348, 135, 136, 137, 138, 139, 140, 143, 144, 145, 146, 349, 135, 136, 137, 138, 139, 140, 143, 144, 145, 146, 350, 135, 136, 137, 138, 139, 140, 141, 142, 145, 146, 351, 135, 136, 137, 138, 139, 140, 141, 142, 145, 146, 352, 71, 72, 73, 74, 135, 136, 137, 138, 139, 140, 353, 71, 72, 73, 74, 135, 136, 137, 138, 139, 140, 354, 73, 74, 75, 76, 135, 136, 137, 138, 139, 140, 355, 73, 74, 75, 76, 135, 136, 137, 138, 139, 140, 356, 71, 72, 75, 76, 135, 136, 137, 138, 139, 140, 357, 71, 72, 75, 76, 135, 136, 137, 138, 139, 140, 358, 15, 16, 21, 22, 359, 360, 17, 18, 21, 22, 361, 362, 19, 20, 21, 22, 363, 364, 15, 16, 21, 22, 359, 360, 17, 18, 21, 22, 361, 362, 19, 20, 21, 22, 363, 364, 359, 361, 363, 365, 360, 362, 364, 366, 23, 24, 29, 30, 367, 368, 25, 26, 29, 30, 369, 370, 27, 28, 29, 30, 371, 372, 23, 24, 29, 30, 367, 368, 25, 26, 29, 30, 369, 370, 27, 28, 29, 30, 371, 372, 367, 369, 371, 373, 368, 370, 372, 374, 31, 32, 37, 38, 375, 376, 33, 34, 37, 38, 377, 378, 35, 36, 37, 38, 379, 380, 31, 32, 37, 38, 375, 376, 33, 34, 37, 38, 377, 378, 35, 36, 37, 38, 379, 380, 375, 377, 379, 381, 376, 378, 380, 382, 39, 40, 45, 46, 383, 384, 41, 42, 45, 46, 385, 386, 43, 44, 45, 46, 387, 388, 39, 40, 45, 46, 383, 384, 41, 42, 45, 46, 385, 386, 43, 44, 45, 46, 387, 388, 383, 385, 387, 389, 384, 386, 388, 390, 47, 48, 53, 54, 391, 392, 49, 50, 53, 54, 393, 394, 51, 52, 53, 54, 395, 396, 47, 48, 53, 54, 391, 392, 49, 50, 53, 54, 393, 394, 51, 52, 53, 54, 395, 396, 391, 393, 395, 397, 392, 394, 396, 398, 55, 56, 61, 62, 399, 400, 57, 58, 61, 62, 401, 402, 59, 60, 61, 62, 403, 404, 55, 56, 61, 62, 399, 400, 57, 58, 61, 62, 401, 402, 59, 60, 61, 62, 403, 404, 399, 401, 403, 405, 400, 402, 404, 406, 63, 64, 69, 70, 407, 408, 65, 66, 69, 70, 409, 410, 67, 68, 69, 70, 411, 412, 63, 64, 69, 70, 407, 408, 65, 66, 69, 70, 409, 410, 67, 68, 69, 70, 411, 412, 407, 409, 411, 413, 408, 410, 412, 414, 71, 72, 77, 78, 415, 416, 73, 74, 77, 78, 417, 418, 75, 76, 77, 78, 419, 420, 71, 72, 77, 78, 415, 416, 73, 74, 77, 78, 417, 418, 75, 76, 77, 78, 419, 420, 415, 417, 419, 421, 416, 418, 420, 422, 79, 80, 85, 86, 423, 424, 81, 82, 85, 86, 425, 426, 83, 84, 85, 86, 427, 428, 79, 80, 85, 86, 423, 424, 81, 82, 85, 86, 425, 426, 83, 84, 85, 86, 427, 428, 423, 425, 427, 429, 424, 426, 428, 430, 87, 88, 93, 94, 431, 432, 89, 90, 93, 94, 433, 434, 91, 92, 93, 94, 435, 436, 87, 88, 93, 94, 431, 432, 89, 90, 93, 94, 433, 434, 91, 92, 93, 94, 435, 436, 431, 433, 435, 437, 432, 434, 436, 438, 95, 96, 101, 102, 439, 440, 97, 98, 101, 102, 441, 442, 99, 100, 101, 102, 443, 444, 95, 96, 101, 102, 439, 440, 97, 98, 101, 102, 441, 442, 99, 100, 101, 102, 443, 444, 439, 441, 443, 445, 440, 442, 444, 446, 103, 104, 109, 110, 447, 448, 105, 106, 109, 110, 449, 450, 107, 108, 109, 110, 451, 452, 103, 104, 109, 110, 447, 448, 105, 106, 109, 110, 449, 450, 107, 108, 109, 110, 451, 452, 447, 449, 451, 453, 448, 450, 452, 454, 111, 112, 117, 118, 455, 456, 113, 114, 117, 118, 457, 458, 115, 116, 117, 118, 459, 460, 111, 112, 117, 118, 455, 456, 113, 114, 117, 118, 457, 458, 115, 116, 117, 118, 459, 460, 455, 457, 459, 461, 456, 458, 460, 462, 119, 120, 125, 126, 463, 464, 121, 122, 125, 126, 465, 466, 123, 124, 125, 126, 467, 468, 119, 120, 125, 126, 463, 464, 121, 122, 125, 126, 465, 466, 123, 124, 125, 126, 467, 468, 463, 465, 467, 469, 464, 466, 468, 470, 127, 128, 133, 134, 471, 472, 129, 130, 133, 134, 473, 474, 131, 132, 133, 134, 475, 476, 127, 128, 133, 134, 471, 472, 129, 130, 133, 134, 473, 474, 131, 132, 133, 134, 475, 476, 471, 473, 475, 477, 472, 474, 476, 478, 0, 149, 155, 479, 485, 0, 151, 155, 480, 486, 0, 153, 155, 481, 487, 155, 482, 484, 488, 479, 480, 481, 482, 483, 483, 484, 489, 0, 150, 156, 479, 485, 0, 152, 156, 480, 486, 0, 154, 156, 481, 487, 156, 482, 488, 490, 485, 486, 487, 488, 489, 483, 489, 490, 2, 165, 171, 491, 497, 2, 167, 171, 492, 498, 2, 169, 171, 493, 499, 171, 494, 496, 500, 491, 492, 493, 494, 495, 495, 496, 501, 2, 166, 172, 491, 497, 2, 168, 172, 492, 498, 2, 170, 172, 493, 499, 172, 494, 500, 502, 497, 498, 499, 500, 501, 495, 501, 502, 4, 197, 203, 503, 509, 4, 199, 203, 504, 510, 4, 201, 203, 505, 511, 203, 506, 508, 512, 503, 504, 505, 506, 507, 507, 508, 513, 4, 198, 204, 503, 509, 4, 200, 204, 504, 510, 4, 202, 204, 505, 511, 204, 506, 512, 514, 509, 510, 511, 512, 513, 507, 513, 514, 6, 141, 147, 515, 521, 6, 143, 147, 516, 522, 6, 145, 147, 517, 523, 147, 518, 520, 524, 515, 516, 517, 518, 519, 519, 520, 525, 6, 142, 148, 515, 521, 6, 144, 148, 516, 522, 6, 146, 148, 517, 523, 148, 518, 524, 526, 521, 522, 523, 524, 525, 519, 525, 526, 8, 301, 307, 527, 533, 8, 303, 307, 528, 534, 8, 305, 307, 529, 535, 307, 530, 532, 536, 527, 528, 529, 530, 531, 531, 532, 537, 8, 302, 308, 527, 533, 8, 304, 308, 528, 534, 8, 306, 308, 529, 535, 308, 530, 536, 538, 533, 534, 535, 536, 537, 531, 537, 538, 10, 317, 323, 539, 545, 10, 319, 323, 540, 546, 10, 321, 323, 541, 547, 323, 542, 544, 548, 539, 540, 541, 542, 543, 543, 544, 549, 10, 318, 324, 539, 545, 10, 320, 324, 540, 546, 10, 322, 324, 541, 547, 324, 542, 548, 550, 545, 546, 547, 548, 549, 543, 549, 550, 12, 245, 251, 551, 557, 12, 247, 251, 552, 558, 12, 249, 251, 553, 559, 251, 554, 556, 560, 551, 552, 553, 554, 555, 555, 556, 561, 12, 246, 252, 551, 557, 12, 248, 252, 552, 558, 12, 250, 252, 553, 559, 252, 554, 560, 562, 557, 558, 559, 560, 561, 555, 561, 562, 1, 3, 5, 7, 9, 11, 13, 563]
    sp_jac_trap_ja = [0, 3, 17, 20, 34, 37, 51, 54, 68, 71, 85, 88, 102, 105, 119, 121, 142, 163, 184, 205, 226, 247, 263, 279, 296, 313, 330, 347, 364, 381, 397, 413, 430, 447, 464, 481, 498, 515, 531, 547, 564, 581, 598, 615, 632, 649, 665, 681, 698, 715, 732, 749, 766, 783, 799, 815, 832, 849, 866, 883, 900, 917, 933, 949, 966, 983, 1000, 1017, 1034, 1051, 1067, 1083, 1104, 1125, 1146, 1167, 1188, 1209, 1225, 1241, 1258, 1275, 1292, 1309, 1326, 1343, 1359, 1375, 1392, 1409, 1426, 1443, 1460, 1477, 1493, 1509, 1526, 1543, 1560, 1577, 1594, 1611, 1627, 1643, 1660, 1677, 1694, 1711, 1728, 1745, 1761, 1777, 1794, 1811, 1828, 1845, 1862, 1879, 1895, 1911, 1928, 1945, 1962, 1979, 1996, 2013, 2029, 2045, 2062, 2079, 2096, 2113, 2130, 2147, 2163, 2179, 2197, 2215, 2233, 2251, 2269, 2287, 2308, 2329, 2350, 2371, 2392, 2413, 2430, 2447, 2472, 2497, 2522, 2547, 2572, 2597, 2622, 2647, 2679, 2711, 2743, 2775, 2807, 2839, 2871, 2903, 2936, 2969, 3002, 3035, 3068, 3101, 3134, 3167, 3191, 3215, 3239, 3263, 3287, 3311, 3335, 3359, 3391, 3423, 3455, 3487, 3519, 3551, 3583, 3615, 3639, 3663, 3687, 3711, 3735, 3759, 3783, 3807, 3832, 3857, 3882, 3907, 3932, 3957, 3982, 4007, 4039, 4071, 4103, 4135, 4167, 4199, 4231, 4263, 4287, 4311, 4335, 4359, 4383, 4407, 4431, 4455, 4479, 4503, 4527, 4551, 4575, 4599, 4623, 4647, 4671, 4695, 4719, 4743, 4767, 4791, 4815, 4839, 4863, 4887, 4911, 4935, 4959, 4983, 5007, 5031, 5056, 5081, 5106, 5131, 5156, 5181, 5206, 5231, 5263, 5295, 5327, 5359, 5391, 5423, 5455, 5487, 5511, 5535, 5559, 5583, 5607, 5631, 5655, 5679, 5711, 5743, 5775, 5807, 5839, 5871, 5903, 5935, 5959, 5983, 6007, 6031, 6055, 6079, 6103, 6127, 6151, 6175, 6199, 6223, 6247, 6271, 6295, 6319, 6351, 6383, 6415, 6447, 6479, 6511, 6543, 6575, 6600, 6625, 6650, 6675, 6700, 6725, 6750, 6775, 6807, 6839, 6871, 6903, 6935, 6967, 6999, 7031, 7064, 7097, 7130, 7163, 7196, 7229, 7262, 7295, 7327, 7359, 7391, 7423, 7455, 7487, 7519, 7551, 7575, 7599, 7623, 7647, 7671, 7695, 7719, 7743, 7754, 7765, 7776, 7787, 7798, 7809, 7820, 7831, 7842, 7853, 7864, 7875, 7886, 7897, 7908, 7919, 7930, 7941, 7947, 7953, 7959, 7965, 7971, 7977, 7981, 7985, 7991, 7997, 8003, 8009, 8015, 8021, 8025, 8029, 8035, 8041, 8047, 8053, 8059, 8065, 8069, 8073, 8079, 8085, 8091, 8097, 8103, 8109, 8113, 8117, 8123, 8129, 8135, 8141, 8147, 8153, 8157, 8161, 8167, 8173, 8179, 8185, 8191, 8197, 8201, 8205, 8211, 8217, 8223, 8229, 8235, 8241, 8245, 8249, 8255, 8261, 8267, 8273, 8279, 8285, 8289, 8293, 8299, 8305, 8311, 8317, 8323, 8329, 8333, 8337, 8343, 8349, 8355, 8361, 8367, 8373, 8377, 8381, 8387, 8393, 8399, 8405, 8411, 8417, 8421, 8425, 8431, 8437, 8443, 8449, 8455, 8461, 8465, 8469, 8475, 8481, 8487, 8493, 8499, 8505, 8509, 8513, 8519, 8525, 8531, 8537, 8543, 8549, 8553, 8557, 8563, 8569, 8575, 8581, 8587, 8593, 8597, 8601, 8606, 8611, 8616, 8620, 8625, 8628, 8633, 8638, 8643, 8647, 8652, 8655, 8660, 8665, 8670, 8674, 8679, 8682, 8687, 8692, 8697, 8701, 8706, 8709, 8714, 8719, 8724, 8728, 8733, 8736, 8741, 8746, 8751, 8755, 8760, 8763, 8768, 8773, 8778, 8782, 8787, 8790, 8795, 8800, 8805, 8809, 8814, 8817, 8822, 8827, 8832, 8836, 8841, 8844, 8849, 8854, 8859, 8863, 8868, 8871, 8876, 8881, 8886, 8890, 8895, 8898, 8903, 8908, 8913, 8917, 8922, 8925, 8930, 8935, 8940, 8944, 8949, 8952, 8957, 8962, 8967, 8971, 8976, 8979, 8987]
    sp_jac_trap_nia = 564
    sp_jac_trap_nja = 564
    return sp_jac_trap_ia, sp_jac_trap_ja, sp_jac_trap_nia, sp_jac_trap_nja 
