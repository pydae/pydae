import numpy as np
import numba
import scipy.optimize as sopt
import scipy.sparse as sspa
from scipy.sparse.linalg import spsolve,spilu,splu
from numba import cuda
import cffi
import numba.core.typing.cffi_utils as cffi_support

ffi = cffi.FFI()

import cigre_europe_residential_4w2w_cffi as jacs

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


class cigre_europe_residential_4w2w_class: 

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
        self.N_y = 345 
        self.N_z = 339 
        self.N_store = 10000 
        self.params_list = ['a_R1', 'b_R1', 'c_R1', 'a_R10', 'b_R10', 'c_R10', 'coef_a_R10', 'coef_b_R10', 'coef_c_R10', 'a_R14', 'b_R14', 'c_R14', 'coef_a_R14', 'coef_b_R14', 'coef_c_R14'] 
        self.params_values_list  = [2.92, 0.45, 0.027, 2.92, 0.45, 0.027, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333, 2.92, 0.45, 0.027, 0.3333333333333333, 0.3333333333333333, 0.3333333333333333] 
        self.inputs_ini_list = ['v_R0_a_r', 'v_R0_a_i', 'v_R0_b_r', 'v_R0_b_i', 'v_R0_c_r', 'v_R0_c_i', 'v_D1_a_r', 'v_D1_a_i', 'v_D1_b_r', 'v_D1_b_i', 'v_D1_c_r', 'v_D1_c_i', 'i_R1_n_r', 'i_R1_n_i', 'i_R11_n_r', 'i_R11_n_i', 'i_R15_n_r', 'i_R15_n_i', 'i_R16_n_r', 'i_R16_n_i', 'i_R17_n_r', 'i_R17_n_i', 'i_R18_n_r', 'i_R18_n_i', 'i_R2_a_r', 'i_R2_a_i', 'i_R2_b_r', 'i_R2_b_i', 'i_R2_c_r', 'i_R2_c_i', 'i_R2_n_r', 'i_R2_n_i', 'i_R3_a_r', 'i_R3_a_i', 'i_R3_b_r', 'i_R3_b_i', 'i_R3_c_r', 'i_R3_c_i', 'i_R3_n_r', 'i_R3_n_i', 'i_R4_a_r', 'i_R4_a_i', 'i_R4_b_r', 'i_R4_b_i', 'i_R4_c_r', 'i_R4_c_i', 'i_R4_n_r', 'i_R4_n_i', 'i_R5_a_r', 'i_R5_a_i', 'i_R5_b_r', 'i_R5_b_i', 'i_R5_c_r', 'i_R5_c_i', 'i_R5_n_r', 'i_R5_n_i', 'i_R6_a_r', 'i_R6_a_i', 'i_R6_b_r', 'i_R6_b_i', 'i_R6_c_r', 'i_R6_c_i', 'i_R6_n_r', 'i_R6_n_i', 'i_R7_a_r', 'i_R7_a_i', 'i_R7_b_r', 'i_R7_b_i', 'i_R7_c_r', 'i_R7_c_i', 'i_R7_n_r', 'i_R7_n_i', 'i_R8_a_r', 'i_R8_a_i', 'i_R8_b_r', 'i_R8_b_i', 'i_R8_c_r', 'i_R8_c_i', 'i_R8_n_r', 'i_R8_n_i', 'i_R9_a_r', 'i_R9_a_i', 'i_R9_b_r', 'i_R9_b_i', 'i_R9_c_r', 'i_R9_c_i', 'i_R9_n_r', 'i_R9_n_i', 'i_R10_a_r', 'i_R10_a_i', 'i_R10_b_r', 'i_R10_b_i', 'i_R10_c_r', 'i_R10_c_i', 'i_R10_n_r', 'i_R10_n_i', 'i_R12_a_r', 'i_R12_a_i', 'i_R12_b_r', 'i_R12_b_i', 'i_R12_c_r', 'i_R12_c_i', 'i_R12_n_r', 'i_R12_n_i', 'i_R13_a_r', 'i_R13_a_i', 'i_R13_b_r', 'i_R13_b_i', 'i_R13_c_r', 'i_R13_c_i', 'i_R13_n_r', 'i_R13_n_i', 'i_R14_a_r', 'i_R14_a_i', 'i_R14_b_r', 'i_R14_b_i', 'i_R14_c_r', 'i_R14_c_i', 'i_R14_n_r', 'i_R14_n_i', 'i_D1_n_r', 'i_D1_n_i', 'i_D3_a_r', 'i_D3_a_i', 'i_D3_b_r', 'i_D3_b_i', 'i_D3_c_r', 'i_D3_c_i', 'i_D3_n_r', 'i_D3_n_i', 'i_D4_a_r', 'i_D4_a_i', 'i_D4_b_r', 'i_D4_b_i', 'i_D4_c_r', 'i_D4_c_i', 'i_D4_n_r', 'i_D4_n_i', 'i_D6_a_r', 'i_D6_a_i', 'i_D6_b_r', 'i_D6_b_i', 'i_D6_c_r', 'i_D6_c_i', 'i_D6_n_r', 'i_D6_n_i', 'i_D9_a_r', 'i_D9_a_i', 'i_D9_b_r', 'i_D9_b_i', 'i_D9_c_r', 'i_D9_c_i', 'i_D9_n_r', 'i_D9_n_i', 'i_D10_a_i', 'i_D10_b_r', 'i_D10_b_i', 'i_D10_c_r', 'i_D10_c_i', 'i_D10_n_i', 'i_D11_b_r', 'i_D11_b_i', 'i_D11_c_r', 'i_D11_c_i', 'i_D16_b_r', 'i_D16_b_i', 'i_D16_c_r', 'i_D16_c_i', 'i_D17_b_r', 'i_D17_b_i', 'i_D17_c_r', 'i_D17_c_i', 'i_D18_b_r', 'i_D18_b_i', 'i_D18_c_r', 'i_D18_c_i', 'i_D14_a_i', 'i_D14_b_r', 'i_D14_b_i', 'i_D14_c_r', 'i_D14_c_i', 'i_D14_n_i', 'i_D15_b_r', 'i_D15_b_i', 'i_D15_c_r', 'i_D15_c_i', 'p_R1_a', 'q_R1_a', 'p_R1_b', 'q_R1_b', 'p_R1_c', 'q_R1_c', 'p_R11_a', 'q_R11_a', 'p_R11_b', 'q_R11_b', 'p_R11_c', 'q_R11_c', 'p_R15_a', 'q_R15_a', 'p_R15_b', 'q_R15_b', 'p_R15_c', 'q_R15_c', 'p_R16_a', 'q_R16_a', 'p_R16_b', 'q_R16_b', 'p_R16_c', 'q_R16_c', 'p_R17_a', 'q_R17_a', 'p_R17_b', 'q_R17_b', 'p_R17_c', 'q_R17_c', 'p_R18_a', 'q_R18_a', 'p_R18_b', 'q_R18_b', 'p_R18_c', 'q_R18_c', 'p_D15_1', 'q_D15_1', 'p_D11_1', 'q_D11_1', 'p_D16_1', 'q_D16_1', 'p_D17_1', 'q_D17_1', 'p_D18_1', 'q_D18_1', 'v_dc_D1', 'q_R1', 'p_R10', 'q_R10', 'p_R14', 'q_R14', 'u_dummy'] 
        self.inputs_ini_values_list  = [11547.0, 0.0, -5773.499999999997, -9999.995337498915, -5773.5000000000055, 9999.99533749891, 800.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.11377181081280696, -0.062435919138408735, 0.038921773012907224, -0.1048963759937358, 0.34454059643019264, -0.7980789681157034, 0.37357657109664544, -0.8707524308167933, 0.3054824960816518, -0.6990681853205416, 0.4382865509753664, -0.9973969692563287, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -63333.333333333765, -20816.659994665173, -63333.33333333563, -20816.659994662015, -63333.33333333578, -20816.65999465767, -4750.000000000053, -1561.2494995996628, -4749.999999999867, -1561.24949959979, -4750.000000000133, -1561.2494995994743, -16466.666666665096, -5412.33159861012, -16466.666666668374, -5412.331598611336, -16466.66666666625, -5412.331598613994, -17416.666666666664, -5724.581498531865, -17416.666666666515, -5724.5814985328325, -17416.66666666606, -5724.581498532622, -11083.333333333874, -3642.9154990660636, -11083.333333333529, -3642.9154990657007, -11083.333333333101, -3642.9154990652314, -14883.33333333232, -4891.91509874379, -14883.333333334123, -4891.91509874528, -14883.333333332876, -4891.915098746649, 174505.2828211094, 0.0, 174505.2828211094, 0.0, 174505.28282110934, 0.0, 174505.28282110937, 0.0, 174505.28282110937, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] 
        self.inputs_run_list = ['v_R0_a_r', 'v_R0_a_i', 'v_R0_b_r', 'v_R0_b_i', 'v_R0_c_r', 'v_R0_c_i', 'v_D1_a_r', 'v_D1_a_i', 'v_D1_b_r', 'v_D1_b_i', 'v_D1_c_r', 'v_D1_c_i', 'i_R1_n_r', 'i_R1_n_i', 'i_R11_n_r', 'i_R11_n_i', 'i_R15_n_r', 'i_R15_n_i', 'i_R16_n_r', 'i_R16_n_i', 'i_R17_n_r', 'i_R17_n_i', 'i_R18_n_r', 'i_R18_n_i', 'i_R2_a_r', 'i_R2_a_i', 'i_R2_b_r', 'i_R2_b_i', 'i_R2_c_r', 'i_R2_c_i', 'i_R2_n_r', 'i_R2_n_i', 'i_R3_a_r', 'i_R3_a_i', 'i_R3_b_r', 'i_R3_b_i', 'i_R3_c_r', 'i_R3_c_i', 'i_R3_n_r', 'i_R3_n_i', 'i_R4_a_r', 'i_R4_a_i', 'i_R4_b_r', 'i_R4_b_i', 'i_R4_c_r', 'i_R4_c_i', 'i_R4_n_r', 'i_R4_n_i', 'i_R5_a_r', 'i_R5_a_i', 'i_R5_b_r', 'i_R5_b_i', 'i_R5_c_r', 'i_R5_c_i', 'i_R5_n_r', 'i_R5_n_i', 'i_R6_a_r', 'i_R6_a_i', 'i_R6_b_r', 'i_R6_b_i', 'i_R6_c_r', 'i_R6_c_i', 'i_R6_n_r', 'i_R6_n_i', 'i_R7_a_r', 'i_R7_a_i', 'i_R7_b_r', 'i_R7_b_i', 'i_R7_c_r', 'i_R7_c_i', 'i_R7_n_r', 'i_R7_n_i', 'i_R8_a_r', 'i_R8_a_i', 'i_R8_b_r', 'i_R8_b_i', 'i_R8_c_r', 'i_R8_c_i', 'i_R8_n_r', 'i_R8_n_i', 'i_R9_a_r', 'i_R9_a_i', 'i_R9_b_r', 'i_R9_b_i', 'i_R9_c_r', 'i_R9_c_i', 'i_R9_n_r', 'i_R9_n_i', 'i_R10_a_r', 'i_R10_a_i', 'i_R10_b_r', 'i_R10_b_i', 'i_R10_c_r', 'i_R10_c_i', 'i_R10_n_r', 'i_R10_n_i', 'i_R12_a_r', 'i_R12_a_i', 'i_R12_b_r', 'i_R12_b_i', 'i_R12_c_r', 'i_R12_c_i', 'i_R12_n_r', 'i_R12_n_i', 'i_R13_a_r', 'i_R13_a_i', 'i_R13_b_r', 'i_R13_b_i', 'i_R13_c_r', 'i_R13_c_i', 'i_R13_n_r', 'i_R13_n_i', 'i_R14_a_r', 'i_R14_a_i', 'i_R14_b_r', 'i_R14_b_i', 'i_R14_c_r', 'i_R14_c_i', 'i_R14_n_r', 'i_R14_n_i', 'i_D1_n_r', 'i_D1_n_i', 'i_D3_a_r', 'i_D3_a_i', 'i_D3_b_r', 'i_D3_b_i', 'i_D3_c_r', 'i_D3_c_i', 'i_D3_n_r', 'i_D3_n_i', 'i_D4_a_r', 'i_D4_a_i', 'i_D4_b_r', 'i_D4_b_i', 'i_D4_c_r', 'i_D4_c_i', 'i_D4_n_r', 'i_D4_n_i', 'i_D6_a_r', 'i_D6_a_i', 'i_D6_b_r', 'i_D6_b_i', 'i_D6_c_r', 'i_D6_c_i', 'i_D6_n_r', 'i_D6_n_i', 'i_D9_a_r', 'i_D9_a_i', 'i_D9_b_r', 'i_D9_b_i', 'i_D9_c_r', 'i_D9_c_i', 'i_D9_n_r', 'i_D9_n_i', 'i_D10_a_i', 'i_D10_b_r', 'i_D10_b_i', 'i_D10_c_r', 'i_D10_c_i', 'i_D10_n_i', 'i_D11_b_r', 'i_D11_b_i', 'i_D11_c_r', 'i_D11_c_i', 'i_D16_b_r', 'i_D16_b_i', 'i_D16_c_r', 'i_D16_c_i', 'i_D17_b_r', 'i_D17_b_i', 'i_D17_c_r', 'i_D17_c_i', 'i_D18_b_r', 'i_D18_b_i', 'i_D18_c_r', 'i_D18_c_i', 'i_D14_a_i', 'i_D14_b_r', 'i_D14_b_i', 'i_D14_c_r', 'i_D14_c_i', 'i_D14_n_i', 'i_D15_b_r', 'i_D15_b_i', 'i_D15_c_r', 'i_D15_c_i', 'p_R1_a', 'q_R1_a', 'p_R1_b', 'q_R1_b', 'p_R1_c', 'q_R1_c', 'p_R11_a', 'q_R11_a', 'p_R11_b', 'q_R11_b', 'p_R11_c', 'q_R11_c', 'p_R15_a', 'q_R15_a', 'p_R15_b', 'q_R15_b', 'p_R15_c', 'q_R15_c', 'p_R16_a', 'q_R16_a', 'p_R16_b', 'q_R16_b', 'p_R16_c', 'q_R16_c', 'p_R17_a', 'q_R17_a', 'p_R17_b', 'q_R17_b', 'p_R17_c', 'q_R17_c', 'p_R18_a', 'q_R18_a', 'p_R18_b', 'q_R18_b', 'p_R18_c', 'q_R18_c', 'p_D15_1', 'q_D15_1', 'p_D11_1', 'q_D11_1', 'p_D16_1', 'q_D16_1', 'p_D17_1', 'q_D17_1', 'p_D18_1', 'q_D18_1', 'v_dc_D1', 'q_R1', 'p_R10', 'q_R10', 'p_R14', 'q_R14', 'u_dummy'] 
        self.inputs_run_values_list = [11547.0, 0.0, -5773.499999999997, -9999.995337498915, -5773.5000000000055, 9999.99533749891, 800.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.11377181081280696, -0.062435919138408735, 0.038921773012907224, -0.1048963759937358, 0.34454059643019264, -0.7980789681157034, 0.37357657109664544, -0.8707524308167933, 0.3054824960816518, -0.6990681853205416, 0.4382865509753664, -0.9973969692563287, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -63333.333333333765, -20816.659994665173, -63333.33333333563, -20816.659994662015, -63333.33333333578, -20816.65999465767, -4750.000000000053, -1561.2494995996628, -4749.999999999867, -1561.24949959979, -4750.000000000133, -1561.2494995994743, -16466.666666665096, -5412.33159861012, -16466.666666668374, -5412.331598611336, -16466.66666666625, -5412.331598613994, -17416.666666666664, -5724.581498531865, -17416.666666666515, -5724.5814985328325, -17416.66666666606, -5724.581498532622, -11083.333333333874, -3642.9154990660636, -11083.333333333529, -3642.9154990657007, -11083.333333333101, -3642.9154990652314, -14883.33333333232, -4891.91509874379, -14883.333333334123, -4891.91509874528, -14883.333333332876, -4891.915098746649, 174505.2828211094, 0.0, 174505.2828211094, 0.0, 174505.28282110934, 0.0, 174505.28282110937, 0.0, 174505.28282110937, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0] 
        self.outputs_list = ['i_l_R1_R2_a_r', 'i_l_R1_R2_a_i', 'i_l_R1_R2_b_r', 'i_l_R1_R2_b_i', 'i_l_R1_R2_c_r', 'i_l_R1_R2_c_i', 'i_l_R1_R2_n_r', 'i_l_R1_R2_n_i', 'i_l_R2_R3_a_r', 'i_l_R2_R3_a_i', 'i_l_R2_R3_b_r', 'i_l_R2_R3_b_i', 'i_l_R2_R3_c_r', 'i_l_R2_R3_c_i', 'i_l_R2_R3_n_r', 'i_l_R2_R3_n_i', 'i_l_R3_R4_a_r', 'i_l_R3_R4_a_i', 'i_l_R3_R4_b_r', 'i_l_R3_R4_b_i', 'i_l_R3_R4_c_r', 'i_l_R3_R4_c_i', 'i_l_R3_R4_n_r', 'i_l_R3_R4_n_i', 'i_l_R4_R5_a_r', 'i_l_R4_R5_a_i', 'i_l_R4_R5_b_r', 'i_l_R4_R5_b_i', 'i_l_R4_R5_c_r', 'i_l_R4_R5_c_i', 'i_l_R4_R5_n_r', 'i_l_R4_R5_n_i', 'i_l_R5_R6_a_r', 'i_l_R5_R6_a_i', 'i_l_R5_R6_b_r', 'i_l_R5_R6_b_i', 'i_l_R5_R6_c_r', 'i_l_R5_R6_c_i', 'i_l_R5_R6_n_r', 'i_l_R5_R6_n_i', 'i_l_R6_R7_a_r', 'i_l_R6_R7_a_i', 'i_l_R6_R7_b_r', 'i_l_R6_R7_b_i', 'i_l_R6_R7_c_r', 'i_l_R6_R7_c_i', 'i_l_R6_R7_n_r', 'i_l_R6_R7_n_i', 'i_l_R7_R8_a_r', 'i_l_R7_R8_a_i', 'i_l_R7_R8_b_r', 'i_l_R7_R8_b_i', 'i_l_R7_R8_c_r', 'i_l_R7_R8_c_i', 'i_l_R7_R8_n_r', 'i_l_R7_R8_n_i', 'i_l_R8_R9_a_r', 'i_l_R8_R9_a_i', 'i_l_R8_R9_b_r', 'i_l_R8_R9_b_i', 'i_l_R8_R9_c_r', 'i_l_R8_R9_c_i', 'i_l_R8_R9_n_r', 'i_l_R8_R9_n_i', 'i_l_R9_R10_a_r', 'i_l_R9_R10_a_i', 'i_l_R9_R10_b_r', 'i_l_R9_R10_b_i', 'i_l_R9_R10_c_r', 'i_l_R9_R10_c_i', 'i_l_R9_R10_n_r', 'i_l_R9_R10_n_i', 'i_l_R3_R11_a_r', 'i_l_R3_R11_a_i', 'i_l_R3_R11_b_r', 'i_l_R3_R11_b_i', 'i_l_R3_R11_c_r', 'i_l_R3_R11_c_i', 'i_l_R3_R11_n_r', 'i_l_R3_R11_n_i', 'i_l_R4_R12_a_r', 'i_l_R4_R12_a_i', 'i_l_R4_R12_b_r', 'i_l_R4_R12_b_i', 'i_l_R4_R12_c_r', 'i_l_R4_R12_c_i', 'i_l_R4_R12_n_r', 'i_l_R4_R12_n_i', 'i_l_R12_R13_a_r', 'i_l_R12_R13_a_i', 'i_l_R12_R13_b_r', 'i_l_R12_R13_b_i', 'i_l_R12_R13_c_r', 'i_l_R12_R13_c_i', 'i_l_R12_R13_n_r', 'i_l_R12_R13_n_i', 'i_l_R13_R14_a_r', 'i_l_R13_R14_a_i', 'i_l_R13_R14_b_r', 'i_l_R13_R14_b_i', 'i_l_R13_R14_c_r', 'i_l_R13_R14_c_i', 'i_l_R13_R14_n_r', 'i_l_R13_R14_n_i', 'i_l_R14_R15_a_r', 'i_l_R14_R15_a_i', 'i_l_R14_R15_b_r', 'i_l_R14_R15_b_i', 'i_l_R14_R15_c_r', 'i_l_R14_R15_c_i', 'i_l_R14_R15_n_r', 'i_l_R14_R15_n_i', 'i_l_R6_R16_a_r', 'i_l_R6_R16_a_i', 'i_l_R6_R16_b_r', 'i_l_R6_R16_b_i', 'i_l_R6_R16_c_r', 'i_l_R6_R16_c_i', 'i_l_R6_R16_n_r', 'i_l_R6_R16_n_i', 'i_l_R9_R17_a_r', 'i_l_R9_R17_a_i', 'i_l_R9_R17_b_r', 'i_l_R9_R17_b_i', 'i_l_R9_R17_c_r', 'i_l_R9_R17_c_i', 'i_l_R9_R17_n_r', 'i_l_R9_R17_n_i', 'i_l_R10_R18_a_r', 'i_l_R10_R18_a_i', 'i_l_R10_R18_b_r', 'i_l_R10_R18_b_i', 'i_l_R10_R18_c_r', 'i_l_R10_R18_c_i', 'i_l_R10_R18_n_r', 'i_l_R10_R18_n_i', 'i_l_D3_D4_a_r', 'i_l_D3_D4_a_i', 'i_l_D3_D4_b_r', 'i_l_D3_D4_b_i', 'i_l_D3_D4_c_r', 'i_l_D3_D4_c_i', 'i_l_D3_D4_n_r', 'i_l_D3_D4_n_i', 'i_l_D4_D6_a_r', 'i_l_D4_D6_a_i', 'i_l_D4_D6_b_r', 'i_l_D4_D6_b_i', 'i_l_D4_D6_c_r', 'i_l_D4_D6_c_i', 'i_l_D4_D6_n_r', 'i_l_D4_D6_n_i', 'i_l_D6_D9_a_r', 'i_l_D6_D9_a_i', 'i_l_D6_D9_b_r', 'i_l_D6_D9_b_i', 'i_l_D6_D9_c_r', 'i_l_D6_D9_c_i', 'i_l_D6_D9_n_r', 'i_l_D6_D9_n_i', 'i_l_D9_D10_a_r', 'i_l_D9_D10_a_i', 'i_l_D9_D10_b_r', 'i_l_D9_D10_b_i', 'i_l_D9_D10_c_r', 'i_l_D9_D10_c_i', 'i_l_D9_D10_n_r', 'i_l_D9_D10_n_i', 'i_l_D3_D11_a_r', 'i_l_D3_D11_a_i', 'i_l_D3_D11_b_r', 'i_l_D3_D11_b_i', 'i_l_D3_D11_c_r', 'i_l_D3_D11_c_i', 'i_l_D3_D11_n_r', 'i_l_D3_D11_n_i', 'i_l_D6_D16_a_r', 'i_l_D6_D16_a_i', 'i_l_D6_D16_b_r', 'i_l_D6_D16_b_i', 'i_l_D6_D16_c_r', 'i_l_D6_D16_c_i', 'i_l_D6_D16_n_r', 'i_l_D6_D16_n_i', 'i_l_D9_D17_a_r', 'i_l_D9_D17_a_i', 'i_l_D9_D17_b_r', 'i_l_D9_D17_b_i', 'i_l_D9_D17_c_r', 'i_l_D9_D17_c_i', 'i_l_D9_D17_n_r', 'i_l_D9_D17_n_i', 'i_l_D10_D18_a_r', 'i_l_D10_D18_a_i', 'i_l_D10_D18_b_r', 'i_l_D10_D18_b_i', 'i_l_D10_D18_c_r', 'i_l_D10_D18_c_i', 'i_l_D10_D18_n_r', 'i_l_D10_D18_n_i', 'i_l_D4_D14_a_r', 'i_l_D4_D14_a_i', 'i_l_D4_D14_b_r', 'i_l_D4_D14_b_i', 'i_l_D4_D14_c_r', 'i_l_D4_D14_c_i', 'i_l_D4_D14_n_r', 'i_l_D4_D14_n_i', 'i_l_D14_D15_a_r', 'i_l_D14_D15_a_i', 'i_l_D14_D15_b_r', 'i_l_D14_D15_b_i', 'i_l_D14_D15_c_r', 'i_l_D14_D15_c_i', 'i_l_D14_D15_n_r', 'i_l_D14_D15_n_i', 'v_R0_a_m', 'v_R0_b_m', 'v_R0_c_m', 'v_D1_a_m', 'v_D1_b_m', 'v_D1_c_m', 'v_R1_a_m', 'v_R1_b_m', 'v_R1_c_m', 'v_R1_n_m', 'v_R11_a_m', 'v_R11_b_m', 'v_R11_c_m', 'v_R11_n_m', 'v_R15_a_m', 'v_R15_b_m', 'v_R15_c_m', 'v_R15_n_m', 'v_R16_a_m', 'v_R16_b_m', 'v_R16_c_m', 'v_R16_n_m', 'v_R17_a_m', 'v_R17_b_m', 'v_R17_c_m', 'v_R17_n_m', 'v_R18_a_m', 'v_R18_b_m', 'v_R18_c_m', 'v_R18_n_m', 'v_D15_a_m', 'v_D15_n_m', 'v_D11_a_m', 'v_D11_n_m', 'v_D16_a_m', 'v_D16_n_m', 'v_D17_a_m', 'v_D17_n_m', 'v_D18_a_m', 'v_D18_n_m', 'v_R2_a_m', 'v_R2_b_m', 'v_R2_c_m', 'v_R2_n_m', 'v_R3_a_m', 'v_R3_b_m', 'v_R3_c_m', 'v_R3_n_m', 'v_R4_a_m', 'v_R4_b_m', 'v_R4_c_m', 'v_R4_n_m', 'v_R5_a_m', 'v_R5_b_m', 'v_R5_c_m', 'v_R5_n_m', 'v_R6_a_m', 'v_R6_b_m', 'v_R6_c_m', 'v_R6_n_m', 'v_R7_a_m', 'v_R7_b_m', 'v_R7_c_m', 'v_R7_n_m', 'v_R8_a_m', 'v_R8_b_m', 'v_R8_c_m', 'v_R8_n_m', 'v_R9_a_m', 'v_R9_b_m', 'v_R9_c_m', 'v_R9_n_m', 'v_R10_a_m', 'v_R10_b_m', 'v_R10_c_m', 'v_R10_n_m', 'v_R12_a_m', 'v_R12_b_m', 'v_R12_c_m', 'v_R12_n_m', 'v_R13_a_m', 'v_R13_b_m', 'v_R13_c_m', 'v_R13_n_m', 'v_R14_a_m', 'v_R14_b_m', 'v_R14_c_m', 'v_R14_n_m', 'v_D1_n_m', 'v_D3_a_m', 'v_D3_b_m', 'v_D3_c_m', 'v_D3_n_m', 'v_D4_a_m', 'v_D4_b_m', 'v_D4_c_m', 'v_D4_n_m', 'v_D6_a_m', 'v_D6_b_m', 'v_D6_c_m', 'v_D6_n_m', 'v_D9_a_m', 'v_D9_b_m', 'v_D9_c_m', 'v_D9_n_m', 'v_D10_a_m', 'v_D10_b_m', 'v_D10_c_m', 'v_D10_n_m', 'v_D11_b_m', 'v_D11_c_m', 'v_D16_b_m', 'v_D16_c_m', 'v_D17_b_m', 'v_D17_c_m', 'v_D18_b_m', 'v_D18_c_m', 'v_D14_a_m', 'v_D14_b_m', 'v_D14_c_m', 'v_D14_n_m', 'v_D15_b_m', 'v_D15_c_m'] 
        self.x_list = ['x_dummy'] 
        self.y_run_list = ['v_R1_a_r', 'v_R1_a_i', 'v_R1_b_r', 'v_R1_b_i', 'v_R1_c_r', 'v_R1_c_i', 'v_R1_n_r', 'v_R1_n_i', 'v_R11_a_r', 'v_R11_a_i', 'v_R11_b_r', 'v_R11_b_i', 'v_R11_c_r', 'v_R11_c_i', 'v_R11_n_r', 'v_R11_n_i', 'v_R15_a_r', 'v_R15_a_i', 'v_R15_b_r', 'v_R15_b_i', 'v_R15_c_r', 'v_R15_c_i', 'v_R15_n_r', 'v_R15_n_i', 'v_R16_a_r', 'v_R16_a_i', 'v_R16_b_r', 'v_R16_b_i', 'v_R16_c_r', 'v_R16_c_i', 'v_R16_n_r', 'v_R16_n_i', 'v_R17_a_r', 'v_R17_a_i', 'v_R17_b_r', 'v_R17_b_i', 'v_R17_c_r', 'v_R17_c_i', 'v_R17_n_r', 'v_R17_n_i', 'v_R18_a_r', 'v_R18_a_i', 'v_R18_b_r', 'v_R18_b_i', 'v_R18_c_r', 'v_R18_c_i', 'v_R18_n_r', 'v_R18_n_i', 'v_D15_a_r', 'v_D15_a_i', 'v_D15_n_r', 'v_D15_n_i', 'v_D11_a_r', 'v_D11_a_i', 'v_D11_n_r', 'v_D11_n_i', 'v_D16_a_r', 'v_D16_a_i', 'v_D16_n_r', 'v_D16_n_i', 'v_D17_a_r', 'v_D17_a_i', 'v_D17_n_r', 'v_D17_n_i', 'v_D18_a_r', 'v_D18_a_i', 'v_D18_n_r', 'v_D18_n_i', 'v_R2_a_r', 'v_R2_a_i', 'v_R2_b_r', 'v_R2_b_i', 'v_R2_c_r', 'v_R2_c_i', 'v_R2_n_r', 'v_R2_n_i', 'v_R3_a_r', 'v_R3_a_i', 'v_R3_b_r', 'v_R3_b_i', 'v_R3_c_r', 'v_R3_c_i', 'v_R3_n_r', 'v_R3_n_i', 'v_R4_a_r', 'v_R4_a_i', 'v_R4_b_r', 'v_R4_b_i', 'v_R4_c_r', 'v_R4_c_i', 'v_R4_n_r', 'v_R4_n_i', 'v_R5_a_r', 'v_R5_a_i', 'v_R5_b_r', 'v_R5_b_i', 'v_R5_c_r', 'v_R5_c_i', 'v_R5_n_r', 'v_R5_n_i', 'v_R6_a_r', 'v_R6_a_i', 'v_R6_b_r', 'v_R6_b_i', 'v_R6_c_r', 'v_R6_c_i', 'v_R6_n_r', 'v_R6_n_i', 'v_R7_a_r', 'v_R7_a_i', 'v_R7_b_r', 'v_R7_b_i', 'v_R7_c_r', 'v_R7_c_i', 'v_R7_n_r', 'v_R7_n_i', 'v_R8_a_r', 'v_R8_a_i', 'v_R8_b_r', 'v_R8_b_i', 'v_R8_c_r', 'v_R8_c_i', 'v_R8_n_r', 'v_R8_n_i', 'v_R9_a_r', 'v_R9_a_i', 'v_R9_b_r', 'v_R9_b_i', 'v_R9_c_r', 'v_R9_c_i', 'v_R9_n_r', 'v_R9_n_i', 'v_R10_a_r', 'v_R10_a_i', 'v_R10_b_r', 'v_R10_b_i', 'v_R10_c_r', 'v_R10_c_i', 'v_R10_n_r', 'v_R10_n_i', 'v_R12_a_r', 'v_R12_a_i', 'v_R12_b_r', 'v_R12_b_i', 'v_R12_c_r', 'v_R12_c_i', 'v_R12_n_r', 'v_R12_n_i', 'v_R13_a_r', 'v_R13_a_i', 'v_R13_b_r', 'v_R13_b_i', 'v_R13_c_r', 'v_R13_c_i', 'v_R13_n_r', 'v_R13_n_i', 'v_R14_a_r', 'v_R14_a_i', 'v_R14_b_r', 'v_R14_b_i', 'v_R14_c_r', 'v_R14_c_i', 'v_R14_n_r', 'v_R14_n_i', 'v_D1_n_r', 'v_D1_n_i', 'v_D3_a_r', 'v_D3_a_i', 'v_D3_b_r', 'v_D3_b_i', 'v_D3_c_r', 'v_D3_c_i', 'v_D3_n_r', 'v_D3_n_i', 'v_D4_a_r', 'v_D4_a_i', 'v_D4_b_r', 'v_D4_b_i', 'v_D4_c_r', 'v_D4_c_i', 'v_D4_n_r', 'v_D4_n_i', 'v_D6_a_r', 'v_D6_a_i', 'v_D6_b_r', 'v_D6_b_i', 'v_D6_c_r', 'v_D6_c_i', 'v_D6_n_r', 'v_D6_n_i', 'v_D9_a_r', 'v_D9_a_i', 'v_D9_b_r', 'v_D9_b_i', 'v_D9_c_r', 'v_D9_c_i', 'v_D9_n_r', 'v_D9_n_i', 'v_D10_a_r', 'v_D10_a_i', 'v_D10_b_r', 'v_D10_b_i', 'v_D10_c_r', 'v_D10_c_i', 'v_D10_n_r', 'v_D10_n_i', 'v_D11_b_r', 'v_D11_b_i', 'v_D11_c_r', 'v_D11_c_i', 'v_D16_b_r', 'v_D16_b_i', 'v_D16_c_r', 'v_D16_c_i', 'v_D17_b_r', 'v_D17_b_i', 'v_D17_c_r', 'v_D17_c_i', 'v_D18_b_r', 'v_D18_b_i', 'v_D18_c_r', 'v_D18_c_i', 'v_D14_a_r', 'v_D14_a_i', 'v_D14_b_r', 'v_D14_b_i', 'v_D14_c_r', 'v_D14_c_i', 'v_D14_n_r', 'v_D14_n_i', 'v_D15_b_r', 'v_D15_b_i', 'v_D15_c_r', 'v_D15_c_i', 'i_t_R0_R1_a_r', 'i_t_R0_R1_a_i', 'i_t_R0_R1_b_r', 'i_t_R0_R1_b_i', 'i_t_R0_R1_c_r', 'i_t_R0_R1_c_i', 'i_l_D1_D3_a_r', 'i_l_D1_D3_a_i', 'i_l_D1_D3_b_r', 'i_l_D1_D3_b_i', 'i_l_D1_D3_c_r', 'i_l_D1_D3_c_i', 'i_l_D1_D3_n_r', 'i_l_D1_D3_n_i', 'i_load_R1_a_r', 'i_load_R1_a_i', 'i_load_R1_b_r', 'i_load_R1_b_i', 'i_load_R1_c_r', 'i_load_R1_c_i', 'i_load_R1_n_r', 'i_load_R1_n_i', 'i_load_R11_a_r', 'i_load_R11_a_i', 'i_load_R11_b_r', 'i_load_R11_b_i', 'i_load_R11_c_r', 'i_load_R11_c_i', 'i_load_R11_n_r', 'i_load_R11_n_i', 'i_load_R15_a_r', 'i_load_R15_a_i', 'i_load_R15_b_r', 'i_load_R15_b_i', 'i_load_R15_c_r', 'i_load_R15_c_i', 'i_load_R15_n_r', 'i_load_R15_n_i', 'i_load_R16_a_r', 'i_load_R16_a_i', 'i_load_R16_b_r', 'i_load_R16_b_i', 'i_load_R16_c_r', 'i_load_R16_c_i', 'i_load_R16_n_r', 'i_load_R16_n_i', 'i_load_R17_a_r', 'i_load_R17_a_i', 'i_load_R17_b_r', 'i_load_R17_b_i', 'i_load_R17_c_r', 'i_load_R17_c_i', 'i_load_R17_n_r', 'i_load_R17_n_i', 'i_load_R18_a_r', 'i_load_R18_a_i', 'i_load_R18_b_r', 'i_load_R18_b_i', 'i_load_R18_c_r', 'i_load_R18_c_i', 'i_load_R18_n_r', 'i_load_R18_n_i', 'i_load_D15_a_r', 'i_load_D15_a_i', 'i_load_D15_n_r', 'i_load_D15_n_i', 'i_load_D11_a_r', 'i_load_D11_a_i', 'i_load_D11_n_r', 'i_load_D11_n_i', 'i_load_D16_a_r', 'i_load_D16_a_i', 'i_load_D16_n_r', 'i_load_D16_n_i', 'i_load_D17_a_r', 'i_load_D17_a_i', 'i_load_D17_n_r', 'i_load_D17_n_i', 'i_load_D18_a_r', 'i_load_D18_a_i', 'i_load_D18_n_r', 'i_load_D18_n_i', 'i_vsc_R1_a_r', 'i_vsc_R1_a_i', 'i_vsc_R1_b_r', 'i_vsc_R1_b_i', 'i_vsc_R1_c_r', 'i_vsc_R1_c_i', 'p_R1', 'p_D1', 'p_loss_R1', 'i_vsc_R10_a_r', 'i_vsc_R10_a_i', 'i_vsc_R10_b_r', 'i_vsc_R10_b_i', 'i_vsc_R10_c_r', 'i_vsc_R10_c_i', 'i_vsc_D10_a_r', 'i_vsc_D10_n_r', 'p_D10', 'p_loss_R10', 'i_vsc_R14_a_r', 'i_vsc_R14_a_i', 'i_vsc_R14_b_r', 'i_vsc_R14_b_i', 'i_vsc_R14_c_r', 'i_vsc_R14_c_i', 'i_vsc_D14_a_r', 'i_vsc_D14_n_r', 'p_D14', 'p_loss_R14'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['v_R1_a_r', 'v_R1_a_i', 'v_R1_b_r', 'v_R1_b_i', 'v_R1_c_r', 'v_R1_c_i', 'v_R1_n_r', 'v_R1_n_i', 'v_R11_a_r', 'v_R11_a_i', 'v_R11_b_r', 'v_R11_b_i', 'v_R11_c_r', 'v_R11_c_i', 'v_R11_n_r', 'v_R11_n_i', 'v_R15_a_r', 'v_R15_a_i', 'v_R15_b_r', 'v_R15_b_i', 'v_R15_c_r', 'v_R15_c_i', 'v_R15_n_r', 'v_R15_n_i', 'v_R16_a_r', 'v_R16_a_i', 'v_R16_b_r', 'v_R16_b_i', 'v_R16_c_r', 'v_R16_c_i', 'v_R16_n_r', 'v_R16_n_i', 'v_R17_a_r', 'v_R17_a_i', 'v_R17_b_r', 'v_R17_b_i', 'v_R17_c_r', 'v_R17_c_i', 'v_R17_n_r', 'v_R17_n_i', 'v_R18_a_r', 'v_R18_a_i', 'v_R18_b_r', 'v_R18_b_i', 'v_R18_c_r', 'v_R18_c_i', 'v_R18_n_r', 'v_R18_n_i', 'v_D15_a_r', 'v_D15_a_i', 'v_D15_n_r', 'v_D15_n_i', 'v_D11_a_r', 'v_D11_a_i', 'v_D11_n_r', 'v_D11_n_i', 'v_D16_a_r', 'v_D16_a_i', 'v_D16_n_r', 'v_D16_n_i', 'v_D17_a_r', 'v_D17_a_i', 'v_D17_n_r', 'v_D17_n_i', 'v_D18_a_r', 'v_D18_a_i', 'v_D18_n_r', 'v_D18_n_i', 'v_R2_a_r', 'v_R2_a_i', 'v_R2_b_r', 'v_R2_b_i', 'v_R2_c_r', 'v_R2_c_i', 'v_R2_n_r', 'v_R2_n_i', 'v_R3_a_r', 'v_R3_a_i', 'v_R3_b_r', 'v_R3_b_i', 'v_R3_c_r', 'v_R3_c_i', 'v_R3_n_r', 'v_R3_n_i', 'v_R4_a_r', 'v_R4_a_i', 'v_R4_b_r', 'v_R4_b_i', 'v_R4_c_r', 'v_R4_c_i', 'v_R4_n_r', 'v_R4_n_i', 'v_R5_a_r', 'v_R5_a_i', 'v_R5_b_r', 'v_R5_b_i', 'v_R5_c_r', 'v_R5_c_i', 'v_R5_n_r', 'v_R5_n_i', 'v_R6_a_r', 'v_R6_a_i', 'v_R6_b_r', 'v_R6_b_i', 'v_R6_c_r', 'v_R6_c_i', 'v_R6_n_r', 'v_R6_n_i', 'v_R7_a_r', 'v_R7_a_i', 'v_R7_b_r', 'v_R7_b_i', 'v_R7_c_r', 'v_R7_c_i', 'v_R7_n_r', 'v_R7_n_i', 'v_R8_a_r', 'v_R8_a_i', 'v_R8_b_r', 'v_R8_b_i', 'v_R8_c_r', 'v_R8_c_i', 'v_R8_n_r', 'v_R8_n_i', 'v_R9_a_r', 'v_R9_a_i', 'v_R9_b_r', 'v_R9_b_i', 'v_R9_c_r', 'v_R9_c_i', 'v_R9_n_r', 'v_R9_n_i', 'v_R10_a_r', 'v_R10_a_i', 'v_R10_b_r', 'v_R10_b_i', 'v_R10_c_r', 'v_R10_c_i', 'v_R10_n_r', 'v_R10_n_i', 'v_R12_a_r', 'v_R12_a_i', 'v_R12_b_r', 'v_R12_b_i', 'v_R12_c_r', 'v_R12_c_i', 'v_R12_n_r', 'v_R12_n_i', 'v_R13_a_r', 'v_R13_a_i', 'v_R13_b_r', 'v_R13_b_i', 'v_R13_c_r', 'v_R13_c_i', 'v_R13_n_r', 'v_R13_n_i', 'v_R14_a_r', 'v_R14_a_i', 'v_R14_b_r', 'v_R14_b_i', 'v_R14_c_r', 'v_R14_c_i', 'v_R14_n_r', 'v_R14_n_i', 'v_D1_n_r', 'v_D1_n_i', 'v_D3_a_r', 'v_D3_a_i', 'v_D3_b_r', 'v_D3_b_i', 'v_D3_c_r', 'v_D3_c_i', 'v_D3_n_r', 'v_D3_n_i', 'v_D4_a_r', 'v_D4_a_i', 'v_D4_b_r', 'v_D4_b_i', 'v_D4_c_r', 'v_D4_c_i', 'v_D4_n_r', 'v_D4_n_i', 'v_D6_a_r', 'v_D6_a_i', 'v_D6_b_r', 'v_D6_b_i', 'v_D6_c_r', 'v_D6_c_i', 'v_D6_n_r', 'v_D6_n_i', 'v_D9_a_r', 'v_D9_a_i', 'v_D9_b_r', 'v_D9_b_i', 'v_D9_c_r', 'v_D9_c_i', 'v_D9_n_r', 'v_D9_n_i', 'v_D10_a_r', 'v_D10_a_i', 'v_D10_b_r', 'v_D10_b_i', 'v_D10_c_r', 'v_D10_c_i', 'v_D10_n_r', 'v_D10_n_i', 'v_D11_b_r', 'v_D11_b_i', 'v_D11_c_r', 'v_D11_c_i', 'v_D16_b_r', 'v_D16_b_i', 'v_D16_c_r', 'v_D16_c_i', 'v_D17_b_r', 'v_D17_b_i', 'v_D17_c_r', 'v_D17_c_i', 'v_D18_b_r', 'v_D18_b_i', 'v_D18_c_r', 'v_D18_c_i', 'v_D14_a_r', 'v_D14_a_i', 'v_D14_b_r', 'v_D14_b_i', 'v_D14_c_r', 'v_D14_c_i', 'v_D14_n_r', 'v_D14_n_i', 'v_D15_b_r', 'v_D15_b_i', 'v_D15_c_r', 'v_D15_c_i', 'i_t_R0_R1_a_r', 'i_t_R0_R1_a_i', 'i_t_R0_R1_b_r', 'i_t_R0_R1_b_i', 'i_t_R0_R1_c_r', 'i_t_R0_R1_c_i', 'i_l_D1_D3_a_r', 'i_l_D1_D3_a_i', 'i_l_D1_D3_b_r', 'i_l_D1_D3_b_i', 'i_l_D1_D3_c_r', 'i_l_D1_D3_c_i', 'i_l_D1_D3_n_r', 'i_l_D1_D3_n_i', 'i_load_R1_a_r', 'i_load_R1_a_i', 'i_load_R1_b_r', 'i_load_R1_b_i', 'i_load_R1_c_r', 'i_load_R1_c_i', 'i_load_R1_n_r', 'i_load_R1_n_i', 'i_load_R11_a_r', 'i_load_R11_a_i', 'i_load_R11_b_r', 'i_load_R11_b_i', 'i_load_R11_c_r', 'i_load_R11_c_i', 'i_load_R11_n_r', 'i_load_R11_n_i', 'i_load_R15_a_r', 'i_load_R15_a_i', 'i_load_R15_b_r', 'i_load_R15_b_i', 'i_load_R15_c_r', 'i_load_R15_c_i', 'i_load_R15_n_r', 'i_load_R15_n_i', 'i_load_R16_a_r', 'i_load_R16_a_i', 'i_load_R16_b_r', 'i_load_R16_b_i', 'i_load_R16_c_r', 'i_load_R16_c_i', 'i_load_R16_n_r', 'i_load_R16_n_i', 'i_load_R17_a_r', 'i_load_R17_a_i', 'i_load_R17_b_r', 'i_load_R17_b_i', 'i_load_R17_c_r', 'i_load_R17_c_i', 'i_load_R17_n_r', 'i_load_R17_n_i', 'i_load_R18_a_r', 'i_load_R18_a_i', 'i_load_R18_b_r', 'i_load_R18_b_i', 'i_load_R18_c_r', 'i_load_R18_c_i', 'i_load_R18_n_r', 'i_load_R18_n_i', 'i_load_D15_a_r', 'i_load_D15_a_i', 'i_load_D15_n_r', 'i_load_D15_n_i', 'i_load_D11_a_r', 'i_load_D11_a_i', 'i_load_D11_n_r', 'i_load_D11_n_i', 'i_load_D16_a_r', 'i_load_D16_a_i', 'i_load_D16_n_r', 'i_load_D16_n_i', 'i_load_D17_a_r', 'i_load_D17_a_i', 'i_load_D17_n_r', 'i_load_D17_n_i', 'i_load_D18_a_r', 'i_load_D18_a_i', 'i_load_D18_n_r', 'i_load_D18_n_i', 'i_vsc_R1_a_r', 'i_vsc_R1_a_i', 'i_vsc_R1_b_r', 'i_vsc_R1_b_i', 'i_vsc_R1_c_r', 'i_vsc_R1_c_i', 'p_R1', 'p_D1', 'p_loss_R1', 'i_vsc_R10_a_r', 'i_vsc_R10_a_i', 'i_vsc_R10_b_r', 'i_vsc_R10_b_i', 'i_vsc_R10_c_r', 'i_vsc_R10_c_i', 'i_vsc_D10_a_r', 'i_vsc_D10_n_r', 'p_D10', 'p_loss_R10', 'i_vsc_R14_a_r', 'i_vsc_R14_a_i', 'i_vsc_R14_b_r', 'i_vsc_R14_b_i', 'i_vsc_R14_c_r', 'i_vsc_R14_c_i', 'i_vsc_D14_a_r', 'i_vsc_D14_n_r', 'p_D14', 'p_loss_R14'] 
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
            #return
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
                
        if type(xy_0) == float or type(xy_0) == int:
            self.xy_0 = np.ones(self.N_x+self.N_y,dtype=np.float64)*xy_0


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

    sp_jac_ini_ia = [0, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 249, 317, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 250, 318, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 251, 319, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 252, 320, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 253, 321, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 254, 322, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 9, 10, 11, 12, 13, 14, 15, 16, 77, 78, 79, 80, 81, 82, 83, 84, 257, 9, 10, 11, 12, 13, 14, 15, 16, 77, 78, 79, 80, 81, 82, 83, 84, 258, 9, 10, 11, 12, 13, 14, 15, 16, 77, 78, 79, 80, 81, 82, 83, 84, 259, 9, 10, 11, 12, 13, 14, 15, 16, 77, 78, 79, 80, 81, 82, 83, 84, 260, 9, 10, 11, 12, 13, 14, 15, 16, 77, 78, 79, 80, 81, 82, 83, 84, 261, 9, 10, 11, 12, 13, 14, 15, 16, 77, 78, 79, 80, 81, 82, 83, 84, 262, 9, 10, 11, 12, 13, 14, 15, 16, 77, 78, 79, 80, 81, 82, 83, 84, 9, 10, 11, 12, 13, 14, 15, 16, 77, 78, 79, 80, 81, 82, 83, 84, 17, 18, 19, 20, 21, 22, 23, 24, 157, 158, 159, 160, 161, 162, 163, 164, 265, 17, 18, 19, 20, 21, 22, 23, 24, 157, 158, 159, 160, 161, 162, 163, 164, 266, 17, 18, 19, 20, 21, 22, 23, 24, 157, 158, 159, 160, 161, 162, 163, 164, 267, 17, 18, 19, 20, 21, 22, 23, 24, 157, 158, 159, 160, 161, 162, 163, 164, 268, 17, 18, 19, 20, 21, 22, 23, 24, 157, 158, 159, 160, 161, 162, 163, 164, 269, 17, 18, 19, 20, 21, 22, 23, 24, 157, 158, 159, 160, 161, 162, 163, 164, 270, 17, 18, 19, 20, 21, 22, 23, 24, 157, 158, 159, 160, 161, 162, 163, 164, 17, 18, 19, 20, 21, 22, 23, 24, 157, 158, 159, 160, 161, 162, 163, 164, 25, 26, 27, 28, 29, 30, 31, 32, 101, 102, 103, 104, 105, 106, 107, 108, 273, 25, 26, 27, 28, 29, 30, 31, 32, 101, 102, 103, 104, 105, 106, 107, 108, 274, 25, 26, 27, 28, 29, 30, 31, 32, 101, 102, 103, 104, 105, 106, 107, 108, 275, 25, 26, 27, 28, 29, 30, 31, 32, 101, 102, 103, 104, 105, 106, 107, 108, 276, 25, 26, 27, 28, 29, 30, 31, 32, 101, 102, 103, 104, 105, 106, 107, 108, 277, 25, 26, 27, 28, 29, 30, 31, 32, 101, 102, 103, 104, 105, 106, 107, 108, 278, 25, 26, 27, 28, 29, 30, 31, 32, 101, 102, 103, 104, 105, 106, 107, 108, 25, 26, 27, 28, 29, 30, 31, 32, 101, 102, 103, 104, 105, 106, 107, 108, 33, 34, 35, 36, 37, 38, 39, 40, 125, 126, 127, 128, 129, 130, 131, 132, 281, 33, 34, 35, 36, 37, 38, 39, 40, 125, 126, 127, 128, 129, 130, 131, 132, 282, 33, 34, 35, 36, 37, 38, 39, 40, 125, 126, 127, 128, 129, 130, 131, 132, 283, 33, 34, 35, 36, 37, 38, 39, 40, 125, 126, 127, 128, 129, 130, 131, 132, 284, 33, 34, 35, 36, 37, 38, 39, 40, 125, 126, 127, 128, 129, 130, 131, 132, 285, 33, 34, 35, 36, 37, 38, 39, 40, 125, 126, 127, 128, 129, 130, 131, 132, 286, 33, 34, 35, 36, 37, 38, 39, 40, 125, 126, 127, 128, 129, 130, 131, 132, 33, 34, 35, 36, 37, 38, 39, 40, 125, 126, 127, 128, 129, 130, 131, 132, 41, 42, 43, 44, 45, 46, 47, 48, 133, 134, 135, 136, 137, 138, 139, 140, 289, 41, 42, 43, 44, 45, 46, 47, 48, 133, 134, 135, 136, 137, 138, 139, 140, 290, 41, 42, 43, 44, 45, 46, 47, 48, 133, 134, 135, 136, 137, 138, 139, 140, 291, 41, 42, 43, 44, 45, 46, 47, 48, 133, 134, 135, 136, 137, 138, 139, 140, 292, 41, 42, 43, 44, 45, 46, 47, 48, 133, 134, 135, 136, 137, 138, 139, 140, 293, 41, 42, 43, 44, 45, 46, 47, 48, 133, 134, 135, 136, 137, 138, 139, 140, 294, 41, 42, 43, 44, 45, 46, 47, 48, 133, 134, 135, 136, 137, 138, 139, 140, 41, 42, 43, 44, 45, 46, 47, 48, 133, 134, 135, 136, 137, 138, 139, 140, 49, 223, 297, 50, 224, 298, 51, 229, 299, 52, 230, 300, 53, 167, 301, 54, 168, 302, 55, 173, 303, 56, 174, 304, 57, 183, 305, 58, 184, 306, 59, 189, 307, 60, 190, 308, 61, 191, 309, 62, 192, 310, 63, 197, 311, 64, 198, 312, 65, 199, 313, 66, 200, 314, 67, 205, 315, 68, 206, 316, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 9, 10, 11, 12, 13, 14, 15, 16, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 9, 10, 11, 12, 13, 14, 15, 16, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 9, 10, 11, 12, 13, 14, 15, 16, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 9, 10, 11, 12, 13, 14, 15, 16, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 9, 10, 11, 12, 13, 14, 15, 16, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 9, 10, 11, 12, 13, 14, 15, 16, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 9, 10, 11, 12, 13, 14, 15, 16, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 9, 10, 11, 12, 13, 14, 15, 16, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 141, 142, 143, 144, 145, 146, 147, 148, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 141, 142, 143, 144, 145, 146, 147, 148, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 141, 142, 143, 144, 145, 146, 147, 148, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 141, 142, 143, 144, 145, 146, 147, 148, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 141, 142, 143, 144, 145, 146, 147, 148, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 141, 142, 143, 144, 145, 146, 147, 148, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 141, 142, 143, 144, 145, 146, 147, 148, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 141, 142, 143, 144, 145, 146, 147, 148, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 25, 26, 27, 28, 29, 30, 31, 32, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 25, 26, 27, 28, 29, 30, 31, 32, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 25, 26, 27, 28, 29, 30, 31, 32, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 25, 26, 27, 28, 29, 30, 31, 32, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 25, 26, 27, 28, 29, 30, 31, 32, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 25, 26, 27, 28, 29, 30, 31, 32, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 25, 26, 27, 28, 29, 30, 31, 32, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 25, 26, 27, 28, 29, 30, 31, 32, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 33, 34, 35, 36, 37, 38, 39, 40, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 33, 34, 35, 36, 37, 38, 39, 40, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 33, 34, 35, 36, 37, 38, 39, 40, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 33, 34, 35, 36, 37, 38, 39, 40, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 33, 34, 35, 36, 37, 38, 39, 40, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 33, 34, 35, 36, 37, 38, 39, 40, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 33, 34, 35, 36, 37, 38, 39, 40, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 33, 34, 35, 36, 37, 38, 39, 40, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 41, 42, 43, 44, 45, 46, 47, 48, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 326, 41, 42, 43, 44, 45, 46, 47, 48, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 327, 41, 42, 43, 44, 45, 46, 47, 48, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 328, 41, 42, 43, 44, 45, 46, 47, 48, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 329, 41, 42, 43, 44, 45, 46, 47, 48, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 330, 41, 42, 43, 44, 45, 46, 47, 48, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 331, 41, 42, 43, 44, 45, 46, 47, 48, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 41, 42, 43, 44, 45, 46, 47, 48, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 85, 86, 87, 88, 89, 90, 91, 92, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 85, 86, 87, 88, 89, 90, 91, 92, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 85, 86, 87, 88, 89, 90, 91, 92, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 85, 86, 87, 88, 89, 90, 91, 92, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 85, 86, 87, 88, 89, 90, 91, 92, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 85, 86, 87, 88, 89, 90, 91, 92, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 85, 86, 87, 88, 89, 90, 91, 92, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 85, 86, 87, 88, 89, 90, 91, 92, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 17, 18, 19, 20, 21, 22, 23, 24, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 336, 17, 18, 19, 20, 21, 22, 23, 24, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 337, 17, 18, 19, 20, 21, 22, 23, 24, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 338, 17, 18, 19, 20, 21, 22, 23, 24, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 339, 17, 18, 19, 20, 21, 22, 23, 24, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 340, 17, 18, 19, 20, 21, 22, 23, 24, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 341, 17, 18, 19, 20, 21, 22, 23, 24, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 17, 18, 19, 20, 21, 22, 23, 24, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 173, 166, 174, 53, 167, 175, 54, 168, 176, 169, 177, 207, 170, 178, 208, 171, 179, 209, 172, 180, 210, 55, 165, 173, 181, 56, 166, 174, 182, 167, 175, 183, 223, 168, 176, 184, 224, 169, 177, 185, 225, 170, 178, 186, 226, 171, 179, 187, 227, 172, 180, 188, 228, 173, 181, 189, 229, 174, 182, 190, 230, 57, 175, 183, 191, 58, 176, 184, 192, 177, 185, 193, 211, 178, 186, 194, 212, 179, 187, 195, 213, 180, 188, 196, 214, 59, 181, 189, 197, 60, 182, 190, 198, 61, 183, 191, 199, 62, 184, 192, 200, 185, 193, 201, 215, 186, 194, 202, 216, 187, 195, 203, 217, 188, 196, 204, 218, 63, 189, 197, 205, 64, 190, 198, 206, 65, 191, 199, 332, 66, 192, 200, 193, 201, 219, 194, 202, 220, 195, 203, 221, 196, 204, 222, 67, 197, 205, 333, 68, 198, 206, 169, 207, 170, 208, 171, 209, 172, 210, 185, 211, 186, 212, 187, 213, 188, 214, 193, 215, 194, 216, 195, 217, 196, 218, 201, 219, 202, 220, 203, 221, 204, 222, 49, 175, 223, 342, 50, 176, 224, 177, 225, 231, 178, 226, 232, 179, 227, 233, 180, 228, 234, 51, 181, 229, 343, 52, 182, 230, 225, 231, 226, 232, 227, 233, 228, 234, 1, 2, 3, 4, 235, 1, 2, 3, 4, 236, 3, 4, 5, 6, 237, 3, 4, 5, 6, 238, 1, 2, 5, 6, 239, 1, 2, 5, 6, 240, 167, 241, 168, 242, 169, 243, 170, 244, 171, 245, 172, 246, 241, 243, 245, 247, 242, 244, 246, 248, 1, 2, 7, 8, 249, 250, 3, 4, 7, 8, 251, 252, 5, 6, 7, 8, 253, 254, 1, 2, 7, 8, 249, 250, 3, 4, 7, 8, 251, 252, 5, 6, 7, 8, 253, 254, 249, 251, 253, 255, 250, 252, 254, 256, 9, 10, 15, 16, 257, 258, 11, 12, 15, 16, 259, 260, 13, 14, 15, 16, 261, 262, 9, 10, 15, 16, 257, 258, 11, 12, 15, 16, 259, 260, 13, 14, 15, 16, 261, 262, 257, 259, 261, 263, 258, 260, 262, 264, 17, 18, 23, 24, 265, 266, 19, 20, 23, 24, 267, 268, 21, 22, 23, 24, 269, 270, 17, 18, 23, 24, 265, 266, 19, 20, 23, 24, 267, 268, 21, 22, 23, 24, 269, 270, 265, 267, 269, 271, 266, 268, 270, 272, 25, 26, 31, 32, 273, 274, 27, 28, 31, 32, 275, 276, 29, 30, 31, 32, 277, 278, 25, 26, 31, 32, 273, 274, 27, 28, 31, 32, 275, 276, 29, 30, 31, 32, 277, 278, 273, 275, 277, 279, 274, 276, 278, 280, 33, 34, 39, 40, 281, 282, 35, 36, 39, 40, 283, 284, 37, 38, 39, 40, 285, 286, 33, 34, 39, 40, 281, 282, 35, 36, 39, 40, 283, 284, 37, 38, 39, 40, 285, 286, 281, 283, 285, 287, 282, 284, 286, 288, 41, 42, 47, 48, 289, 290, 43, 44, 47, 48, 291, 292, 45, 46, 47, 48, 293, 294, 41, 42, 47, 48, 289, 290, 43, 44, 47, 48, 291, 292, 45, 46, 47, 48, 293, 294, 289, 291, 293, 295, 290, 292, 294, 296, 49, 50, 51, 52, 297, 298, 49, 50, 51, 52, 297, 298, 297, 299, 298, 300, 53, 54, 55, 56, 301, 302, 53, 54, 55, 56, 301, 302, 301, 303, 302, 304, 57, 58, 59, 60, 305, 306, 57, 58, 59, 60, 305, 306, 305, 307, 306, 308, 61, 62, 63, 64, 309, 310, 61, 62, 63, 64, 309, 310, 309, 311, 310, 312, 65, 66, 67, 68, 313, 314, 65, 66, 67, 68, 313, 314, 313, 315, 314, 316, 1, 2, 7, 8, 317, 318, 323, 1, 2, 7, 8, 317, 318, 3, 4, 7, 8, 319, 320, 323, 3, 4, 7, 8, 319, 320, 5, 6, 7, 8, 321, 322, 323, 5, 6, 7, 8, 321, 322, 323, 324, 325, 241, 247, 324, 317, 318, 325, 133, 134, 139, 140, 326, 327, 133, 134, 139, 140, 326, 327, 135, 136, 139, 140, 328, 329, 135, 136, 139, 140, 328, 329, 137, 138, 139, 140, 330, 331, 137, 138, 139, 140, 330, 331, 199, 205, 332, 334, 199, 205, 333, 334, 334, 335, 326, 327, 335, 157, 158, 163, 164, 336, 337, 157, 158, 163, 164, 336, 337, 159, 160, 163, 164, 338, 339, 159, 160, 163, 164, 338, 339, 161, 162, 163, 164, 340, 341, 161, 162, 163, 164, 340, 341, 223, 229, 342, 344, 223, 229, 343, 344, 344, 345, 336, 337, 345]
    sp_jac_ini_ja = [0, 1, 19, 37, 55, 73, 91, 109, 125, 141, 158, 175, 192, 209, 226, 243, 259, 275, 292, 309, 326, 343, 360, 377, 393, 409, 426, 443, 460, 477, 494, 511, 527, 543, 560, 577, 594, 611, 628, 645, 661, 677, 694, 711, 728, 745, 762, 779, 795, 811, 814, 817, 820, 823, 826, 829, 832, 835, 838, 841, 844, 847, 850, 853, 856, 859, 862, 865, 868, 871, 895, 919, 943, 967, 991, 1015, 1039, 1063, 1095, 1127, 1159, 1191, 1223, 1255, 1287, 1319, 1351, 1383, 1415, 1447, 1479, 1511, 1543, 1575, 1599, 1623, 1647, 1671, 1695, 1719, 1743, 1767, 1799, 1831, 1863, 1895, 1927, 1959, 1991, 2023, 2047, 2071, 2095, 2119, 2143, 2167, 2191, 2215, 2239, 2263, 2287, 2311, 2335, 2359, 2383, 2407, 2439, 2471, 2503, 2535, 2567, 2599, 2631, 2663, 2688, 2713, 2738, 2763, 2788, 2813, 2837, 2861, 2885, 2909, 2933, 2957, 2981, 3005, 3029, 3053, 3077, 3101, 3125, 3149, 3173, 3197, 3221, 3245, 3270, 3295, 3320, 3345, 3370, 3395, 3419, 3443, 3445, 3447, 3450, 3453, 3456, 3459, 3462, 3465, 3469, 3473, 3477, 3481, 3485, 3489, 3493, 3497, 3501, 3505, 3509, 3513, 3517, 3521, 3525, 3529, 3533, 3537, 3541, 3545, 3549, 3553, 3557, 3561, 3565, 3569, 3573, 3576, 3579, 3582, 3585, 3588, 3592, 3595, 3597, 3599, 3601, 3603, 3605, 3607, 3609, 3611, 3613, 3615, 3617, 3619, 3621, 3623, 3625, 3627, 3631, 3634, 3637, 3640, 3643, 3646, 3650, 3653, 3655, 3657, 3659, 3661, 3666, 3671, 3676, 3681, 3686, 3691, 3693, 3695, 3697, 3699, 3701, 3703, 3707, 3711, 3717, 3723, 3729, 3735, 3741, 3747, 3751, 3755, 3761, 3767, 3773, 3779, 3785, 3791, 3795, 3799, 3805, 3811, 3817, 3823, 3829, 3835, 3839, 3843, 3849, 3855, 3861, 3867, 3873, 3879, 3883, 3887, 3893, 3899, 3905, 3911, 3917, 3923, 3927, 3931, 3937, 3943, 3949, 3955, 3961, 3967, 3971, 3975, 3981, 3987, 3989, 3991, 3997, 4003, 4005, 4007, 4013, 4019, 4021, 4023, 4029, 4035, 4037, 4039, 4045, 4051, 4053, 4055, 4062, 4068, 4075, 4081, 4088, 4094, 4097, 4100, 4103, 4109, 4115, 4121, 4127, 4133, 4139, 4143, 4147, 4149, 4152, 4158, 4164, 4170, 4176, 4182, 4188, 4192, 4196, 4198, 4201]
    sp_jac_ini_nia = 346
    sp_jac_ini_nja = 346
    return sp_jac_ini_ia, sp_jac_ini_ja, sp_jac_ini_nia, sp_jac_ini_nja 

def sp_jac_trap_vectors():

    sp_jac_trap_ia = [0, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 249, 317, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 250, 318, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 251, 319, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 252, 320, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 253, 321, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 254, 322, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 9, 10, 11, 12, 13, 14, 15, 16, 77, 78, 79, 80, 81, 82, 83, 84, 257, 9, 10, 11, 12, 13, 14, 15, 16, 77, 78, 79, 80, 81, 82, 83, 84, 258, 9, 10, 11, 12, 13, 14, 15, 16, 77, 78, 79, 80, 81, 82, 83, 84, 259, 9, 10, 11, 12, 13, 14, 15, 16, 77, 78, 79, 80, 81, 82, 83, 84, 260, 9, 10, 11, 12, 13, 14, 15, 16, 77, 78, 79, 80, 81, 82, 83, 84, 261, 9, 10, 11, 12, 13, 14, 15, 16, 77, 78, 79, 80, 81, 82, 83, 84, 262, 9, 10, 11, 12, 13, 14, 15, 16, 77, 78, 79, 80, 81, 82, 83, 84, 9, 10, 11, 12, 13, 14, 15, 16, 77, 78, 79, 80, 81, 82, 83, 84, 17, 18, 19, 20, 21, 22, 23, 24, 157, 158, 159, 160, 161, 162, 163, 164, 265, 17, 18, 19, 20, 21, 22, 23, 24, 157, 158, 159, 160, 161, 162, 163, 164, 266, 17, 18, 19, 20, 21, 22, 23, 24, 157, 158, 159, 160, 161, 162, 163, 164, 267, 17, 18, 19, 20, 21, 22, 23, 24, 157, 158, 159, 160, 161, 162, 163, 164, 268, 17, 18, 19, 20, 21, 22, 23, 24, 157, 158, 159, 160, 161, 162, 163, 164, 269, 17, 18, 19, 20, 21, 22, 23, 24, 157, 158, 159, 160, 161, 162, 163, 164, 270, 17, 18, 19, 20, 21, 22, 23, 24, 157, 158, 159, 160, 161, 162, 163, 164, 17, 18, 19, 20, 21, 22, 23, 24, 157, 158, 159, 160, 161, 162, 163, 164, 25, 26, 27, 28, 29, 30, 31, 32, 101, 102, 103, 104, 105, 106, 107, 108, 273, 25, 26, 27, 28, 29, 30, 31, 32, 101, 102, 103, 104, 105, 106, 107, 108, 274, 25, 26, 27, 28, 29, 30, 31, 32, 101, 102, 103, 104, 105, 106, 107, 108, 275, 25, 26, 27, 28, 29, 30, 31, 32, 101, 102, 103, 104, 105, 106, 107, 108, 276, 25, 26, 27, 28, 29, 30, 31, 32, 101, 102, 103, 104, 105, 106, 107, 108, 277, 25, 26, 27, 28, 29, 30, 31, 32, 101, 102, 103, 104, 105, 106, 107, 108, 278, 25, 26, 27, 28, 29, 30, 31, 32, 101, 102, 103, 104, 105, 106, 107, 108, 25, 26, 27, 28, 29, 30, 31, 32, 101, 102, 103, 104, 105, 106, 107, 108, 33, 34, 35, 36, 37, 38, 39, 40, 125, 126, 127, 128, 129, 130, 131, 132, 281, 33, 34, 35, 36, 37, 38, 39, 40, 125, 126, 127, 128, 129, 130, 131, 132, 282, 33, 34, 35, 36, 37, 38, 39, 40, 125, 126, 127, 128, 129, 130, 131, 132, 283, 33, 34, 35, 36, 37, 38, 39, 40, 125, 126, 127, 128, 129, 130, 131, 132, 284, 33, 34, 35, 36, 37, 38, 39, 40, 125, 126, 127, 128, 129, 130, 131, 132, 285, 33, 34, 35, 36, 37, 38, 39, 40, 125, 126, 127, 128, 129, 130, 131, 132, 286, 33, 34, 35, 36, 37, 38, 39, 40, 125, 126, 127, 128, 129, 130, 131, 132, 33, 34, 35, 36, 37, 38, 39, 40, 125, 126, 127, 128, 129, 130, 131, 132, 41, 42, 43, 44, 45, 46, 47, 48, 133, 134, 135, 136, 137, 138, 139, 140, 289, 41, 42, 43, 44, 45, 46, 47, 48, 133, 134, 135, 136, 137, 138, 139, 140, 290, 41, 42, 43, 44, 45, 46, 47, 48, 133, 134, 135, 136, 137, 138, 139, 140, 291, 41, 42, 43, 44, 45, 46, 47, 48, 133, 134, 135, 136, 137, 138, 139, 140, 292, 41, 42, 43, 44, 45, 46, 47, 48, 133, 134, 135, 136, 137, 138, 139, 140, 293, 41, 42, 43, 44, 45, 46, 47, 48, 133, 134, 135, 136, 137, 138, 139, 140, 294, 41, 42, 43, 44, 45, 46, 47, 48, 133, 134, 135, 136, 137, 138, 139, 140, 41, 42, 43, 44, 45, 46, 47, 48, 133, 134, 135, 136, 137, 138, 139, 140, 49, 223, 297, 50, 224, 298, 51, 229, 299, 52, 230, 300, 53, 167, 301, 54, 168, 302, 55, 173, 303, 56, 174, 304, 57, 183, 305, 58, 184, 306, 59, 189, 307, 60, 190, 308, 61, 191, 309, 62, 192, 310, 63, 197, 311, 64, 198, 312, 65, 199, 313, 66, 200, 314, 67, 205, 315, 68, 206, 316, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 1, 2, 3, 4, 5, 6, 7, 8, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 9, 10, 11, 12, 13, 14, 15, 16, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 9, 10, 11, 12, 13, 14, 15, 16, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 9, 10, 11, 12, 13, 14, 15, 16, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 9, 10, 11, 12, 13, 14, 15, 16, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 9, 10, 11, 12, 13, 14, 15, 16, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 9, 10, 11, 12, 13, 14, 15, 16, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 9, 10, 11, 12, 13, 14, 15, 16, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 9, 10, 11, 12, 13, 14, 15, 16, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 141, 142, 143, 144, 145, 146, 147, 148, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 141, 142, 143, 144, 145, 146, 147, 148, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 141, 142, 143, 144, 145, 146, 147, 148, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 141, 142, 143, 144, 145, 146, 147, 148, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 141, 142, 143, 144, 145, 146, 147, 148, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 141, 142, 143, 144, 145, 146, 147, 148, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 141, 142, 143, 144, 145, 146, 147, 148, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 141, 142, 143, 144, 145, 146, 147, 148, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 25, 26, 27, 28, 29, 30, 31, 32, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 25, 26, 27, 28, 29, 30, 31, 32, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 25, 26, 27, 28, 29, 30, 31, 32, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 25, 26, 27, 28, 29, 30, 31, 32, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 25, 26, 27, 28, 29, 30, 31, 32, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 25, 26, 27, 28, 29, 30, 31, 32, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 25, 26, 27, 28, 29, 30, 31, 32, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 25, 26, 27, 28, 29, 30, 31, 32, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 33, 34, 35, 36, 37, 38, 39, 40, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 33, 34, 35, 36, 37, 38, 39, 40, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 33, 34, 35, 36, 37, 38, 39, 40, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 33, 34, 35, 36, 37, 38, 39, 40, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 33, 34, 35, 36, 37, 38, 39, 40, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 33, 34, 35, 36, 37, 38, 39, 40, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 33, 34, 35, 36, 37, 38, 39, 40, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 33, 34, 35, 36, 37, 38, 39, 40, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 41, 42, 43, 44, 45, 46, 47, 48, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 326, 41, 42, 43, 44, 45, 46, 47, 48, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 327, 41, 42, 43, 44, 45, 46, 47, 48, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 328, 41, 42, 43, 44, 45, 46, 47, 48, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 329, 41, 42, 43, 44, 45, 46, 47, 48, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 330, 41, 42, 43, 44, 45, 46, 47, 48, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 331, 41, 42, 43, 44, 45, 46, 47, 48, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 41, 42, 43, 44, 45, 46, 47, 48, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 85, 86, 87, 88, 89, 90, 91, 92, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 85, 86, 87, 88, 89, 90, 91, 92, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 85, 86, 87, 88, 89, 90, 91, 92, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 85, 86, 87, 88, 89, 90, 91, 92, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 85, 86, 87, 88, 89, 90, 91, 92, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 85, 86, 87, 88, 89, 90, 91, 92, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 85, 86, 87, 88, 89, 90, 91, 92, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 85, 86, 87, 88, 89, 90, 91, 92, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 17, 18, 19, 20, 21, 22, 23, 24, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 336, 17, 18, 19, 20, 21, 22, 23, 24, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 337, 17, 18, 19, 20, 21, 22, 23, 24, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 338, 17, 18, 19, 20, 21, 22, 23, 24, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 339, 17, 18, 19, 20, 21, 22, 23, 24, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 340, 17, 18, 19, 20, 21, 22, 23, 24, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 341, 17, 18, 19, 20, 21, 22, 23, 24, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 17, 18, 19, 20, 21, 22, 23, 24, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 173, 166, 174, 53, 167, 175, 54, 168, 176, 169, 177, 207, 170, 178, 208, 171, 179, 209, 172, 180, 210, 55, 165, 173, 181, 56, 166, 174, 182, 167, 175, 183, 223, 168, 176, 184, 224, 169, 177, 185, 225, 170, 178, 186, 226, 171, 179, 187, 227, 172, 180, 188, 228, 173, 181, 189, 229, 174, 182, 190, 230, 57, 175, 183, 191, 58, 176, 184, 192, 177, 185, 193, 211, 178, 186, 194, 212, 179, 187, 195, 213, 180, 188, 196, 214, 59, 181, 189, 197, 60, 182, 190, 198, 61, 183, 191, 199, 62, 184, 192, 200, 185, 193, 201, 215, 186, 194, 202, 216, 187, 195, 203, 217, 188, 196, 204, 218, 63, 189, 197, 205, 64, 190, 198, 206, 65, 191, 199, 332, 66, 192, 200, 193, 201, 219, 194, 202, 220, 195, 203, 221, 196, 204, 222, 67, 197, 205, 333, 68, 198, 206, 169, 207, 170, 208, 171, 209, 172, 210, 185, 211, 186, 212, 187, 213, 188, 214, 193, 215, 194, 216, 195, 217, 196, 218, 201, 219, 202, 220, 203, 221, 204, 222, 49, 175, 223, 342, 50, 176, 224, 177, 225, 231, 178, 226, 232, 179, 227, 233, 180, 228, 234, 51, 181, 229, 343, 52, 182, 230, 225, 231, 226, 232, 227, 233, 228, 234, 1, 2, 3, 4, 235, 1, 2, 3, 4, 236, 3, 4, 5, 6, 237, 3, 4, 5, 6, 238, 1, 2, 5, 6, 239, 1, 2, 5, 6, 240, 167, 241, 168, 242, 169, 243, 170, 244, 171, 245, 172, 246, 241, 243, 245, 247, 242, 244, 246, 248, 1, 2, 7, 8, 249, 250, 3, 4, 7, 8, 251, 252, 5, 6, 7, 8, 253, 254, 1, 2, 7, 8, 249, 250, 3, 4, 7, 8, 251, 252, 5, 6, 7, 8, 253, 254, 249, 251, 253, 255, 250, 252, 254, 256, 9, 10, 15, 16, 257, 258, 11, 12, 15, 16, 259, 260, 13, 14, 15, 16, 261, 262, 9, 10, 15, 16, 257, 258, 11, 12, 15, 16, 259, 260, 13, 14, 15, 16, 261, 262, 257, 259, 261, 263, 258, 260, 262, 264, 17, 18, 23, 24, 265, 266, 19, 20, 23, 24, 267, 268, 21, 22, 23, 24, 269, 270, 17, 18, 23, 24, 265, 266, 19, 20, 23, 24, 267, 268, 21, 22, 23, 24, 269, 270, 265, 267, 269, 271, 266, 268, 270, 272, 25, 26, 31, 32, 273, 274, 27, 28, 31, 32, 275, 276, 29, 30, 31, 32, 277, 278, 25, 26, 31, 32, 273, 274, 27, 28, 31, 32, 275, 276, 29, 30, 31, 32, 277, 278, 273, 275, 277, 279, 274, 276, 278, 280, 33, 34, 39, 40, 281, 282, 35, 36, 39, 40, 283, 284, 37, 38, 39, 40, 285, 286, 33, 34, 39, 40, 281, 282, 35, 36, 39, 40, 283, 284, 37, 38, 39, 40, 285, 286, 281, 283, 285, 287, 282, 284, 286, 288, 41, 42, 47, 48, 289, 290, 43, 44, 47, 48, 291, 292, 45, 46, 47, 48, 293, 294, 41, 42, 47, 48, 289, 290, 43, 44, 47, 48, 291, 292, 45, 46, 47, 48, 293, 294, 289, 291, 293, 295, 290, 292, 294, 296, 49, 50, 51, 52, 297, 298, 49, 50, 51, 52, 297, 298, 297, 299, 298, 300, 53, 54, 55, 56, 301, 302, 53, 54, 55, 56, 301, 302, 301, 303, 302, 304, 57, 58, 59, 60, 305, 306, 57, 58, 59, 60, 305, 306, 305, 307, 306, 308, 61, 62, 63, 64, 309, 310, 61, 62, 63, 64, 309, 310, 309, 311, 310, 312, 65, 66, 67, 68, 313, 314, 65, 66, 67, 68, 313, 314, 313, 315, 314, 316, 1, 2, 7, 8, 317, 318, 323, 1, 2, 7, 8, 317, 318, 3, 4, 7, 8, 319, 320, 323, 3, 4, 7, 8, 319, 320, 5, 6, 7, 8, 321, 322, 323, 5, 6, 7, 8, 321, 322, 323, 324, 325, 241, 247, 324, 317, 318, 325, 133, 134, 139, 140, 326, 327, 133, 134, 139, 140, 326, 327, 135, 136, 139, 140, 328, 329, 135, 136, 139, 140, 328, 329, 137, 138, 139, 140, 330, 331, 137, 138, 139, 140, 330, 331, 199, 205, 332, 334, 199, 205, 333, 334, 334, 335, 326, 327, 335, 157, 158, 163, 164, 336, 337, 157, 158, 163, 164, 336, 337, 159, 160, 163, 164, 338, 339, 159, 160, 163, 164, 338, 339, 161, 162, 163, 164, 340, 341, 161, 162, 163, 164, 340, 341, 223, 229, 342, 344, 223, 229, 343, 344, 344, 345, 336, 337, 345]
    sp_jac_trap_ja = [0, 1, 19, 37, 55, 73, 91, 109, 125, 141, 158, 175, 192, 209, 226, 243, 259, 275, 292, 309, 326, 343, 360, 377, 393, 409, 426, 443, 460, 477, 494, 511, 527, 543, 560, 577, 594, 611, 628, 645, 661, 677, 694, 711, 728, 745, 762, 779, 795, 811, 814, 817, 820, 823, 826, 829, 832, 835, 838, 841, 844, 847, 850, 853, 856, 859, 862, 865, 868, 871, 895, 919, 943, 967, 991, 1015, 1039, 1063, 1095, 1127, 1159, 1191, 1223, 1255, 1287, 1319, 1351, 1383, 1415, 1447, 1479, 1511, 1543, 1575, 1599, 1623, 1647, 1671, 1695, 1719, 1743, 1767, 1799, 1831, 1863, 1895, 1927, 1959, 1991, 2023, 2047, 2071, 2095, 2119, 2143, 2167, 2191, 2215, 2239, 2263, 2287, 2311, 2335, 2359, 2383, 2407, 2439, 2471, 2503, 2535, 2567, 2599, 2631, 2663, 2688, 2713, 2738, 2763, 2788, 2813, 2837, 2861, 2885, 2909, 2933, 2957, 2981, 3005, 3029, 3053, 3077, 3101, 3125, 3149, 3173, 3197, 3221, 3245, 3270, 3295, 3320, 3345, 3370, 3395, 3419, 3443, 3445, 3447, 3450, 3453, 3456, 3459, 3462, 3465, 3469, 3473, 3477, 3481, 3485, 3489, 3493, 3497, 3501, 3505, 3509, 3513, 3517, 3521, 3525, 3529, 3533, 3537, 3541, 3545, 3549, 3553, 3557, 3561, 3565, 3569, 3573, 3576, 3579, 3582, 3585, 3588, 3592, 3595, 3597, 3599, 3601, 3603, 3605, 3607, 3609, 3611, 3613, 3615, 3617, 3619, 3621, 3623, 3625, 3627, 3631, 3634, 3637, 3640, 3643, 3646, 3650, 3653, 3655, 3657, 3659, 3661, 3666, 3671, 3676, 3681, 3686, 3691, 3693, 3695, 3697, 3699, 3701, 3703, 3707, 3711, 3717, 3723, 3729, 3735, 3741, 3747, 3751, 3755, 3761, 3767, 3773, 3779, 3785, 3791, 3795, 3799, 3805, 3811, 3817, 3823, 3829, 3835, 3839, 3843, 3849, 3855, 3861, 3867, 3873, 3879, 3883, 3887, 3893, 3899, 3905, 3911, 3917, 3923, 3927, 3931, 3937, 3943, 3949, 3955, 3961, 3967, 3971, 3975, 3981, 3987, 3989, 3991, 3997, 4003, 4005, 4007, 4013, 4019, 4021, 4023, 4029, 4035, 4037, 4039, 4045, 4051, 4053, 4055, 4062, 4068, 4075, 4081, 4088, 4094, 4097, 4100, 4103, 4109, 4115, 4121, 4127, 4133, 4139, 4143, 4147, 4149, 4152, 4158, 4164, 4170, 4176, 4182, 4188, 4192, 4196, 4198, 4201]
    sp_jac_trap_nia = 346
    sp_jac_trap_nja = 346
    return sp_jac_trap_ia, sp_jac_trap_ja, sp_jac_trap_nia, sp_jac_trap_nja 
