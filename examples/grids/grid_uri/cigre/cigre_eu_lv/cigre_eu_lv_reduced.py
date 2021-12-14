import numpy as np
import numba
import scipy.optimize as sopt
import scipy.sparse as sspa
from scipy.sparse.linalg import spsolve,spilu,splu
from numba import cuda
import cffi
import numba.core.typing.cffi_utils as cffi_support

ffi = cffi.FFI()

import cigre_eu_lv_reduced_cffi as jacs

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


class cigre_eu_lv_reduced_class: 

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
        self.N_y = 228 
        self.N_z = 227 
        self.N_store = 10000 
        self.params_list = [] 
        self.params_values_list  = [] 
        self.inputs_ini_list = ['v_MV0_a_r', 'v_MV0_a_i', 'v_MV0_b_r', 'v_MV0_b_i', 'v_MV0_c_r', 'v_MV0_c_i', 'i_R01_n_r', 'i_R01_n_i', 'i_R11_n_r', 'i_R11_n_i', 'i_R15_n_r', 'i_R15_n_i', 'i_R16_n_r', 'i_R16_n_i', 'i_R17_n_r', 'i_R17_n_i', 'i_R18_n_r', 'i_R18_n_i', 'i_I02_n_r', 'i_I02_n_i', 'i_I01_a_r', 'i_I01_a_i', 'i_I01_b_r', 'i_I01_b_i', 'i_I01_c_r', 'i_I01_c_i', 'i_I01_n_r', 'i_I01_n_i', 'i_R02_a_r', 'i_R02_a_i', 'i_R02_b_r', 'i_R02_b_i', 'i_R02_c_r', 'i_R02_c_i', 'i_R02_n_r', 'i_R02_n_i', 'i_R03_a_r', 'i_R03_a_i', 'i_R03_b_r', 'i_R03_b_i', 'i_R03_c_r', 'i_R03_c_i', 'i_R03_n_r', 'i_R03_n_i', 'i_R04_a_r', 'i_R04_a_i', 'i_R04_b_r', 'i_R04_b_i', 'i_R04_c_r', 'i_R04_c_i', 'i_R04_n_r', 'i_R04_n_i', 'i_R05_a_r', 'i_R05_a_i', 'i_R05_b_r', 'i_R05_b_i', 'i_R05_c_r', 'i_R05_c_i', 'i_R05_n_r', 'i_R05_n_i', 'i_R06_a_r', 'i_R06_a_i', 'i_R06_b_r', 'i_R06_b_i', 'i_R06_c_r', 'i_R06_c_i', 'i_R06_n_r', 'i_R06_n_i', 'i_R07_a_r', 'i_R07_a_i', 'i_R07_b_r', 'i_R07_b_i', 'i_R07_c_r', 'i_R07_c_i', 'i_R07_n_r', 'i_R07_n_i', 'i_R08_a_r', 'i_R08_a_i', 'i_R08_b_r', 'i_R08_b_i', 'i_R08_c_r', 'i_R08_c_i', 'i_R08_n_r', 'i_R08_n_i', 'i_R09_a_r', 'i_R09_a_i', 'i_R09_b_r', 'i_R09_b_i', 'i_R09_c_r', 'i_R09_c_i', 'i_R09_n_r', 'i_R09_n_i', 'i_R10_a_r', 'i_R10_a_i', 'i_R10_b_r', 'i_R10_b_i', 'i_R10_c_r', 'i_R10_c_i', 'i_R10_n_r', 'i_R10_n_i', 'i_R12_a_r', 'i_R12_a_i', 'i_R12_b_r', 'i_R12_b_i', 'i_R12_c_r', 'i_R12_c_i', 'i_R12_n_r', 'i_R12_n_i', 'i_R13_a_r', 'i_R13_a_i', 'i_R13_b_r', 'i_R13_b_i', 'i_R13_c_r', 'i_R13_c_i', 'i_R13_n_r', 'i_R13_n_i', 'i_R14_a_r', 'i_R14_a_i', 'i_R14_b_r', 'i_R14_b_i', 'i_R14_c_r', 'i_R14_c_i', 'i_R14_n_r', 'i_R14_n_i', 'p_R01_a', 'q_R01_a', 'p_R01_b', 'q_R01_b', 'p_R01_c', 'q_R01_c', 'p_R11_a', 'q_R11_a', 'p_R11_b', 'q_R11_b', 'p_R11_c', 'q_R11_c', 'p_R15_a', 'q_R15_a', 'p_R15_b', 'q_R15_b', 'p_R15_c', 'q_R15_c', 'p_R16_a', 'q_R16_a', 'p_R16_b', 'q_R16_b', 'p_R16_c', 'q_R16_c', 'p_R17_a', 'q_R17_a', 'p_R17_b', 'q_R17_b', 'p_R17_c', 'q_R17_c', 'p_R18_a', 'q_R18_a', 'p_R18_b', 'q_R18_b', 'p_R18_c', 'q_R18_c', 'p_I02_a', 'q_I02_a', 'p_I02_b', 'q_I02_b', 'p_I02_c', 'q_I02_c', 'u_dummy'] 
        self.inputs_ini_values_list  = [11547.0, 0.0, -5773.499999999997, -9999.995337498915, -5773.5000000000055, 9999.99533749891, -0.11377181080248988, -0.06243591914750368, 0.03892177301212296, -0.104896375995299, 0.34454059642924406, -0.798078968112435, 0.37357657109761533, -0.870752430813539, 0.30548249608112243, -0.699068185326631, 0.4382865509846212, -0.9973969692559024, 0.560284559849066, -1.4893326751834195, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -63333.33333333251, -20816.659994662023, -63333.333333336115, -20816.659994658658, -63333.33333333268, -20816.659994659964, -4750.00000000001, -1561.2494995994844, -4750.000000000485, -1561.2494995995155, -4750.000000000335, -1561.2494995996349, -16466.666666665915, -5412.331598613382, -16466.666666665653, -5412.331598609702, -16466.66666666732, -5412.331598611868, -17416.666666666664, -5724.581498531869, -17416.66666666555, -5724.581498532684, -17416.66666666601, -5724.581498530976, -11083.33333333395, -3642.915499066782, -11083.333333332354, -3642.9154990658967, -11083.333333332734, -3642.915499065095, -14883.333333334538, -4891.915098744748, -14883.333333333598, -4891.91509874537, -14883.333333332406, -4891.915098745633, -28333.333335939395, -17559.422923491937, -28333.333325891002, -17559.422926128485, -28333.33333826515, -17559.42291452118, 1.0] 
        self.inputs_run_list = ['v_MV0_a_r', 'v_MV0_a_i', 'v_MV0_b_r', 'v_MV0_b_i', 'v_MV0_c_r', 'v_MV0_c_i', 'i_R01_n_r', 'i_R01_n_i', 'i_R11_n_r', 'i_R11_n_i', 'i_R15_n_r', 'i_R15_n_i', 'i_R16_n_r', 'i_R16_n_i', 'i_R17_n_r', 'i_R17_n_i', 'i_R18_n_r', 'i_R18_n_i', 'i_I02_n_r', 'i_I02_n_i', 'i_I01_a_r', 'i_I01_a_i', 'i_I01_b_r', 'i_I01_b_i', 'i_I01_c_r', 'i_I01_c_i', 'i_I01_n_r', 'i_I01_n_i', 'i_R02_a_r', 'i_R02_a_i', 'i_R02_b_r', 'i_R02_b_i', 'i_R02_c_r', 'i_R02_c_i', 'i_R02_n_r', 'i_R02_n_i', 'i_R03_a_r', 'i_R03_a_i', 'i_R03_b_r', 'i_R03_b_i', 'i_R03_c_r', 'i_R03_c_i', 'i_R03_n_r', 'i_R03_n_i', 'i_R04_a_r', 'i_R04_a_i', 'i_R04_b_r', 'i_R04_b_i', 'i_R04_c_r', 'i_R04_c_i', 'i_R04_n_r', 'i_R04_n_i', 'i_R05_a_r', 'i_R05_a_i', 'i_R05_b_r', 'i_R05_b_i', 'i_R05_c_r', 'i_R05_c_i', 'i_R05_n_r', 'i_R05_n_i', 'i_R06_a_r', 'i_R06_a_i', 'i_R06_b_r', 'i_R06_b_i', 'i_R06_c_r', 'i_R06_c_i', 'i_R06_n_r', 'i_R06_n_i', 'i_R07_a_r', 'i_R07_a_i', 'i_R07_b_r', 'i_R07_b_i', 'i_R07_c_r', 'i_R07_c_i', 'i_R07_n_r', 'i_R07_n_i', 'i_R08_a_r', 'i_R08_a_i', 'i_R08_b_r', 'i_R08_b_i', 'i_R08_c_r', 'i_R08_c_i', 'i_R08_n_r', 'i_R08_n_i', 'i_R09_a_r', 'i_R09_a_i', 'i_R09_b_r', 'i_R09_b_i', 'i_R09_c_r', 'i_R09_c_i', 'i_R09_n_r', 'i_R09_n_i', 'i_R10_a_r', 'i_R10_a_i', 'i_R10_b_r', 'i_R10_b_i', 'i_R10_c_r', 'i_R10_c_i', 'i_R10_n_r', 'i_R10_n_i', 'i_R12_a_r', 'i_R12_a_i', 'i_R12_b_r', 'i_R12_b_i', 'i_R12_c_r', 'i_R12_c_i', 'i_R12_n_r', 'i_R12_n_i', 'i_R13_a_r', 'i_R13_a_i', 'i_R13_b_r', 'i_R13_b_i', 'i_R13_c_r', 'i_R13_c_i', 'i_R13_n_r', 'i_R13_n_i', 'i_R14_a_r', 'i_R14_a_i', 'i_R14_b_r', 'i_R14_b_i', 'i_R14_c_r', 'i_R14_c_i', 'i_R14_n_r', 'i_R14_n_i', 'p_R01_a', 'q_R01_a', 'p_R01_b', 'q_R01_b', 'p_R01_c', 'q_R01_c', 'p_R11_a', 'q_R11_a', 'p_R11_b', 'q_R11_b', 'p_R11_c', 'q_R11_c', 'p_R15_a', 'q_R15_a', 'p_R15_b', 'q_R15_b', 'p_R15_c', 'q_R15_c', 'p_R16_a', 'q_R16_a', 'p_R16_b', 'q_R16_b', 'p_R16_c', 'q_R16_c', 'p_R17_a', 'q_R17_a', 'p_R17_b', 'q_R17_b', 'p_R17_c', 'q_R17_c', 'p_R18_a', 'q_R18_a', 'p_R18_b', 'q_R18_b', 'p_R18_c', 'q_R18_c', 'p_I02_a', 'q_I02_a', 'p_I02_b', 'q_I02_b', 'p_I02_c', 'q_I02_c', 'u_dummy'] 
        self.inputs_run_values_list = [11547.0, 0.0, -5773.499999999997, -9999.995337498915, -5773.5000000000055, 9999.99533749891, -0.11377181080248988, -0.06243591914750368, 0.03892177301212296, -0.104896375995299, 0.34454059642924406, -0.798078968112435, 0.37357657109761533, -0.870752430813539, 0.30548249608112243, -0.699068185326631, 0.4382865509846212, -0.9973969692559024, 0.560284559849066, -1.4893326751834195, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -63333.33333333251, -20816.659994662023, -63333.333333336115, -20816.659994658658, -63333.33333333268, -20816.659994659964, -4750.00000000001, -1561.2494995994844, -4750.000000000485, -1561.2494995995155, -4750.000000000335, -1561.2494995996349, -16466.666666665915, -5412.331598613382, -16466.666666665653, -5412.331598609702, -16466.66666666732, -5412.331598611868, -17416.666666666664, -5724.581498531869, -17416.66666666555, -5724.581498532684, -17416.66666666601, -5724.581498530976, -11083.33333333395, -3642.915499066782, -11083.333333332354, -3642.9154990658967, -11083.333333332734, -3642.915499065095, -14883.333333334538, -4891.915098744748, -14883.333333333598, -4891.91509874537, -14883.333333332406, -4891.915098745633, -28333.333335939395, -17559.422923491937, -28333.333325891002, -17559.422926128485, -28333.33333826515, -17559.42291452118, 1.0] 
        self.outputs_list = ['i_l_R01_R02_a_r', 'i_l_R01_R02_a_i', 'i_l_R01_R02_b_r', 'i_l_R01_R02_b_i', 'i_l_R01_R02_c_r', 'i_l_R01_R02_c_i', 'i_l_R01_R02_n_r', 'i_l_R01_R02_n_i', 'i_l_R02_R03_a_r', 'i_l_R02_R03_a_i', 'i_l_R02_R03_b_r', 'i_l_R02_R03_b_i', 'i_l_R02_R03_c_r', 'i_l_R02_R03_c_i', 'i_l_R02_R03_n_r', 'i_l_R02_R03_n_i', 'i_l_R03_R04_a_r', 'i_l_R03_R04_a_i', 'i_l_R03_R04_b_r', 'i_l_R03_R04_b_i', 'i_l_R03_R04_c_r', 'i_l_R03_R04_c_i', 'i_l_R03_R04_n_r', 'i_l_R03_R04_n_i', 'i_l_R04_R05_a_r', 'i_l_R04_R05_a_i', 'i_l_R04_R05_b_r', 'i_l_R04_R05_b_i', 'i_l_R04_R05_c_r', 'i_l_R04_R05_c_i', 'i_l_R04_R05_n_r', 'i_l_R04_R05_n_i', 'i_l_R05_R06_a_r', 'i_l_R05_R06_a_i', 'i_l_R05_R06_b_r', 'i_l_R05_R06_b_i', 'i_l_R05_R06_c_r', 'i_l_R05_R06_c_i', 'i_l_R05_R06_n_r', 'i_l_R05_R06_n_i', 'i_l_R06_R07_a_r', 'i_l_R06_R07_a_i', 'i_l_R06_R07_b_r', 'i_l_R06_R07_b_i', 'i_l_R06_R07_c_r', 'i_l_R06_R07_c_i', 'i_l_R06_R07_n_r', 'i_l_R06_R07_n_i', 'i_l_R07_R08_a_r', 'i_l_R07_R08_a_i', 'i_l_R07_R08_b_r', 'i_l_R07_R08_b_i', 'i_l_R07_R08_c_r', 'i_l_R07_R08_c_i', 'i_l_R07_R08_n_r', 'i_l_R07_R08_n_i', 'i_l_R08_R09_a_r', 'i_l_R08_R09_a_i', 'i_l_R08_R09_b_r', 'i_l_R08_R09_b_i', 'i_l_R08_R09_c_r', 'i_l_R08_R09_c_i', 'i_l_R08_R09_n_r', 'i_l_R08_R09_n_i', 'i_l_R09_R10_a_r', 'i_l_R09_R10_a_i', 'i_l_R09_R10_b_r', 'i_l_R09_R10_b_i', 'i_l_R09_R10_c_r', 'i_l_R09_R10_c_i', 'i_l_R09_R10_n_r', 'i_l_R09_R10_n_i', 'i_l_R03_R11_a_r', 'i_l_R03_R11_a_i', 'i_l_R03_R11_b_r', 'i_l_R03_R11_b_i', 'i_l_R03_R11_c_r', 'i_l_R03_R11_c_i', 'i_l_R03_R11_n_r', 'i_l_R03_R11_n_i', 'i_l_R04_R12_a_r', 'i_l_R04_R12_a_i', 'i_l_R04_R12_b_r', 'i_l_R04_R12_b_i', 'i_l_R04_R12_c_r', 'i_l_R04_R12_c_i', 'i_l_R04_R12_n_r', 'i_l_R04_R12_n_i', 'i_l_R12_R13_a_r', 'i_l_R12_R13_a_i', 'i_l_R12_R13_b_r', 'i_l_R12_R13_b_i', 'i_l_R12_R13_c_r', 'i_l_R12_R13_c_i', 'i_l_R12_R13_n_r', 'i_l_R12_R13_n_i', 'i_l_R13_R14_a_r', 'i_l_R13_R14_a_i', 'i_l_R13_R14_b_r', 'i_l_R13_R14_b_i', 'i_l_R13_R14_c_r', 'i_l_R13_R14_c_i', 'i_l_R13_R14_n_r', 'i_l_R13_R14_n_i', 'i_l_R14_R15_a_r', 'i_l_R14_R15_a_i', 'i_l_R14_R15_b_r', 'i_l_R14_R15_b_i', 'i_l_R14_R15_c_r', 'i_l_R14_R15_c_i', 'i_l_R14_R15_n_r', 'i_l_R14_R15_n_i', 'i_l_R06_R16_a_r', 'i_l_R06_R16_a_i', 'i_l_R06_R16_b_r', 'i_l_R06_R16_b_i', 'i_l_R06_R16_c_r', 'i_l_R06_R16_c_i', 'i_l_R06_R16_n_r', 'i_l_R06_R16_n_i', 'i_l_R09_R17_a_r', 'i_l_R09_R17_a_i', 'i_l_R09_R17_b_r', 'i_l_R09_R17_b_i', 'i_l_R09_R17_c_r', 'i_l_R09_R17_c_i', 'i_l_R09_R17_n_r', 'i_l_R09_R17_n_i', 'i_l_R10_R18_a_r', 'i_l_R10_R18_a_i', 'i_l_R10_R18_b_r', 'i_l_R10_R18_b_i', 'i_l_R10_R18_c_r', 'i_l_R10_R18_c_i', 'i_l_R10_R18_n_r', 'i_l_R10_R18_n_i', 'i_l_I01_I02_a_r', 'i_l_I01_I02_a_i', 'i_l_I01_I02_b_r', 'i_l_I01_I02_b_i', 'i_l_I01_I02_c_r', 'i_l_I01_I02_c_i', 'i_l_I01_I02_n_r', 'i_l_I01_I02_n_i', 'v_MV0_a_m', 'v_MV0_b_m', 'v_MV0_c_m', 'v_R01_a_m', 'v_R01_b_m', 'v_R01_c_m', 'v_R01_n_m', 'v_R11_a_m', 'v_R11_b_m', 'v_R11_c_m', 'v_R11_n_m', 'v_R15_a_m', 'v_R15_b_m', 'v_R15_c_m', 'v_R15_n_m', 'v_R16_a_m', 'v_R16_b_m', 'v_R16_c_m', 'v_R16_n_m', 'v_R17_a_m', 'v_R17_b_m', 'v_R17_c_m', 'v_R17_n_m', 'v_R18_a_m', 'v_R18_b_m', 'v_R18_c_m', 'v_R18_n_m', 'v_I02_a_m', 'v_I02_b_m', 'v_I02_c_m', 'v_I02_n_m', 'v_I01_a_m', 'v_I01_b_m', 'v_I01_c_m', 'v_I01_n_m', 'v_R02_a_m', 'v_R02_b_m', 'v_R02_c_m', 'v_R02_n_m', 'v_R03_a_m', 'v_R03_b_m', 'v_R03_c_m', 'v_R03_n_m', 'v_R04_a_m', 'v_R04_b_m', 'v_R04_c_m', 'v_R04_n_m', 'v_R05_a_m', 'v_R05_b_m', 'v_R05_c_m', 'v_R05_n_m', 'v_R06_a_m', 'v_R06_b_m', 'v_R06_c_m', 'v_R06_n_m', 'v_R07_a_m', 'v_R07_b_m', 'v_R07_c_m', 'v_R07_n_m', 'v_R08_a_m', 'v_R08_b_m', 'v_R08_c_m', 'v_R08_n_m', 'v_R09_a_m', 'v_R09_b_m', 'v_R09_c_m', 'v_R09_n_m', 'v_R10_a_m', 'v_R10_b_m', 'v_R10_c_m', 'v_R10_n_m', 'v_R12_a_m', 'v_R12_b_m', 'v_R12_c_m', 'v_R12_n_m', 'v_R13_a_m', 'v_R13_b_m', 'v_R13_c_m', 'v_R13_n_m', 'v_R14_a_m', 'v_R14_b_m', 'v_R14_c_m', 'v_R14_n_m'] 
        self.x_list = ['x_dummy'] 
        self.y_run_list = ['v_R01_a_r', 'v_R01_a_i', 'v_R01_b_r', 'v_R01_b_i', 'v_R01_c_r', 'v_R01_c_i', 'v_R01_n_r', 'v_R01_n_i', 'v_R11_a_r', 'v_R11_a_i', 'v_R11_b_r', 'v_R11_b_i', 'v_R11_c_r', 'v_R11_c_i', 'v_R11_n_r', 'v_R11_n_i', 'v_R15_a_r', 'v_R15_a_i', 'v_R15_b_r', 'v_R15_b_i', 'v_R15_c_r', 'v_R15_c_i', 'v_R15_n_r', 'v_R15_n_i', 'v_R16_a_r', 'v_R16_a_i', 'v_R16_b_r', 'v_R16_b_i', 'v_R16_c_r', 'v_R16_c_i', 'v_R16_n_r', 'v_R16_n_i', 'v_R17_a_r', 'v_R17_a_i', 'v_R17_b_r', 'v_R17_b_i', 'v_R17_c_r', 'v_R17_c_i', 'v_R17_n_r', 'v_R17_n_i', 'v_R18_a_r', 'v_R18_a_i', 'v_R18_b_r', 'v_R18_b_i', 'v_R18_c_r', 'v_R18_c_i', 'v_R18_n_r', 'v_R18_n_i', 'v_I02_a_r', 'v_I02_a_i', 'v_I02_b_r', 'v_I02_b_i', 'v_I02_c_r', 'v_I02_c_i', 'v_I02_n_r', 'v_I02_n_i', 'v_I01_a_r', 'v_I01_a_i', 'v_I01_b_r', 'v_I01_b_i', 'v_I01_c_r', 'v_I01_c_i', 'v_I01_n_r', 'v_I01_n_i', 'v_R02_a_r', 'v_R02_a_i', 'v_R02_b_r', 'v_R02_b_i', 'v_R02_c_r', 'v_R02_c_i', 'v_R02_n_r', 'v_R02_n_i', 'v_R03_a_r', 'v_R03_a_i', 'v_R03_b_r', 'v_R03_b_i', 'v_R03_c_r', 'v_R03_c_i', 'v_R03_n_r', 'v_R03_n_i', 'v_R04_a_r', 'v_R04_a_i', 'v_R04_b_r', 'v_R04_b_i', 'v_R04_c_r', 'v_R04_c_i', 'v_R04_n_r', 'v_R04_n_i', 'v_R05_a_r', 'v_R05_a_i', 'v_R05_b_r', 'v_R05_b_i', 'v_R05_c_r', 'v_R05_c_i', 'v_R05_n_r', 'v_R05_n_i', 'v_R06_a_r', 'v_R06_a_i', 'v_R06_b_r', 'v_R06_b_i', 'v_R06_c_r', 'v_R06_c_i', 'v_R06_n_r', 'v_R06_n_i', 'v_R07_a_r', 'v_R07_a_i', 'v_R07_b_r', 'v_R07_b_i', 'v_R07_c_r', 'v_R07_c_i', 'v_R07_n_r', 'v_R07_n_i', 'v_R08_a_r', 'v_R08_a_i', 'v_R08_b_r', 'v_R08_b_i', 'v_R08_c_r', 'v_R08_c_i', 'v_R08_n_r', 'v_R08_n_i', 'v_R09_a_r', 'v_R09_a_i', 'v_R09_b_r', 'v_R09_b_i', 'v_R09_c_r', 'v_R09_c_i', 'v_R09_n_r', 'v_R09_n_i', 'v_R10_a_r', 'v_R10_a_i', 'v_R10_b_r', 'v_R10_b_i', 'v_R10_c_r', 'v_R10_c_i', 'v_R10_n_r', 'v_R10_n_i', 'v_R12_a_r', 'v_R12_a_i', 'v_R12_b_r', 'v_R12_b_i', 'v_R12_c_r', 'v_R12_c_i', 'v_R12_n_r', 'v_R12_n_i', 'v_R13_a_r', 'v_R13_a_i', 'v_R13_b_r', 'v_R13_b_i', 'v_R13_c_r', 'v_R13_c_i', 'v_R13_n_r', 'v_R13_n_i', 'v_R14_a_r', 'v_R14_a_i', 'v_R14_b_r', 'v_R14_b_i', 'v_R14_c_r', 'v_R14_c_i', 'v_R14_n_r', 'v_R14_n_i', 'i_t_MV0_R01_a_r', 'i_t_MV0_R01_a_i', 'i_t_MV0_R01_b_r', 'i_t_MV0_R01_b_i', 'i_t_MV0_R01_c_r', 'i_t_MV0_R01_c_i', 'i_t_MV0_I01_a_r', 'i_t_MV0_I01_a_i', 'i_t_MV0_I01_b_r', 'i_t_MV0_I01_b_i', 'i_t_MV0_I01_c_r', 'i_t_MV0_I01_c_i', 'i_load_R01_a_r', 'i_load_R01_a_i', 'i_load_R01_b_r', 'i_load_R01_b_i', 'i_load_R01_c_r', 'i_load_R01_c_i', 'i_load_R01_n_r', 'i_load_R01_n_i', 'i_load_R11_a_r', 'i_load_R11_a_i', 'i_load_R11_b_r', 'i_load_R11_b_i', 'i_load_R11_c_r', 'i_load_R11_c_i', 'i_load_R11_n_r', 'i_load_R11_n_i', 'i_load_R15_a_r', 'i_load_R15_a_i', 'i_load_R15_b_r', 'i_load_R15_b_i', 'i_load_R15_c_r', 'i_load_R15_c_i', 'i_load_R15_n_r', 'i_load_R15_n_i', 'i_load_R16_a_r', 'i_load_R16_a_i', 'i_load_R16_b_r', 'i_load_R16_b_i', 'i_load_R16_c_r', 'i_load_R16_c_i', 'i_load_R16_n_r', 'i_load_R16_n_i', 'i_load_R17_a_r', 'i_load_R17_a_i', 'i_load_R17_b_r', 'i_load_R17_b_i', 'i_load_R17_c_r', 'i_load_R17_c_i', 'i_load_R17_n_r', 'i_load_R17_n_i', 'i_load_R18_a_r', 'i_load_R18_a_i', 'i_load_R18_b_r', 'i_load_R18_b_i', 'i_load_R18_c_r', 'i_load_R18_c_i', 'i_load_R18_n_r', 'i_load_R18_n_i', 'i_load_I02_a_r', 'i_load_I02_a_i', 'i_load_I02_b_r', 'i_load_I02_b_i', 'i_load_I02_c_r', 'i_load_I02_c_i', 'i_load_I02_n_r', 'i_load_I02_n_i'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['v_R01_a_r', 'v_R01_a_i', 'v_R01_b_r', 'v_R01_b_i', 'v_R01_c_r', 'v_R01_c_i', 'v_R01_n_r', 'v_R01_n_i', 'v_R11_a_r', 'v_R11_a_i', 'v_R11_b_r', 'v_R11_b_i', 'v_R11_c_r', 'v_R11_c_i', 'v_R11_n_r', 'v_R11_n_i', 'v_R15_a_r', 'v_R15_a_i', 'v_R15_b_r', 'v_R15_b_i', 'v_R15_c_r', 'v_R15_c_i', 'v_R15_n_r', 'v_R15_n_i', 'v_R16_a_r', 'v_R16_a_i', 'v_R16_b_r', 'v_R16_b_i', 'v_R16_c_r', 'v_R16_c_i', 'v_R16_n_r', 'v_R16_n_i', 'v_R17_a_r', 'v_R17_a_i', 'v_R17_b_r', 'v_R17_b_i', 'v_R17_c_r', 'v_R17_c_i', 'v_R17_n_r', 'v_R17_n_i', 'v_R18_a_r', 'v_R18_a_i', 'v_R18_b_r', 'v_R18_b_i', 'v_R18_c_r', 'v_R18_c_i', 'v_R18_n_r', 'v_R18_n_i', 'v_I02_a_r', 'v_I02_a_i', 'v_I02_b_r', 'v_I02_b_i', 'v_I02_c_r', 'v_I02_c_i', 'v_I02_n_r', 'v_I02_n_i', 'v_I01_a_r', 'v_I01_a_i', 'v_I01_b_r', 'v_I01_b_i', 'v_I01_c_r', 'v_I01_c_i', 'v_I01_n_r', 'v_I01_n_i', 'v_R02_a_r', 'v_R02_a_i', 'v_R02_b_r', 'v_R02_b_i', 'v_R02_c_r', 'v_R02_c_i', 'v_R02_n_r', 'v_R02_n_i', 'v_R03_a_r', 'v_R03_a_i', 'v_R03_b_r', 'v_R03_b_i', 'v_R03_c_r', 'v_R03_c_i', 'v_R03_n_r', 'v_R03_n_i', 'v_R04_a_r', 'v_R04_a_i', 'v_R04_b_r', 'v_R04_b_i', 'v_R04_c_r', 'v_R04_c_i', 'v_R04_n_r', 'v_R04_n_i', 'v_R05_a_r', 'v_R05_a_i', 'v_R05_b_r', 'v_R05_b_i', 'v_R05_c_r', 'v_R05_c_i', 'v_R05_n_r', 'v_R05_n_i', 'v_R06_a_r', 'v_R06_a_i', 'v_R06_b_r', 'v_R06_b_i', 'v_R06_c_r', 'v_R06_c_i', 'v_R06_n_r', 'v_R06_n_i', 'v_R07_a_r', 'v_R07_a_i', 'v_R07_b_r', 'v_R07_b_i', 'v_R07_c_r', 'v_R07_c_i', 'v_R07_n_r', 'v_R07_n_i', 'v_R08_a_r', 'v_R08_a_i', 'v_R08_b_r', 'v_R08_b_i', 'v_R08_c_r', 'v_R08_c_i', 'v_R08_n_r', 'v_R08_n_i', 'v_R09_a_r', 'v_R09_a_i', 'v_R09_b_r', 'v_R09_b_i', 'v_R09_c_r', 'v_R09_c_i', 'v_R09_n_r', 'v_R09_n_i', 'v_R10_a_r', 'v_R10_a_i', 'v_R10_b_r', 'v_R10_b_i', 'v_R10_c_r', 'v_R10_c_i', 'v_R10_n_r', 'v_R10_n_i', 'v_R12_a_r', 'v_R12_a_i', 'v_R12_b_r', 'v_R12_b_i', 'v_R12_c_r', 'v_R12_c_i', 'v_R12_n_r', 'v_R12_n_i', 'v_R13_a_r', 'v_R13_a_i', 'v_R13_b_r', 'v_R13_b_i', 'v_R13_c_r', 'v_R13_c_i', 'v_R13_n_r', 'v_R13_n_i', 'v_R14_a_r', 'v_R14_a_i', 'v_R14_b_r', 'v_R14_b_i', 'v_R14_c_r', 'v_R14_c_i', 'v_R14_n_r', 'v_R14_n_i', 'i_t_MV0_R01_a_r', 'i_t_MV0_R01_a_i', 'i_t_MV0_R01_b_r', 'i_t_MV0_R01_b_i', 'i_t_MV0_R01_c_r', 'i_t_MV0_R01_c_i', 'i_t_MV0_I01_a_r', 'i_t_MV0_I01_a_i', 'i_t_MV0_I01_b_r', 'i_t_MV0_I01_b_i', 'i_t_MV0_I01_c_r', 'i_t_MV0_I01_c_i', 'i_load_R01_a_r', 'i_load_R01_a_i', 'i_load_R01_b_r', 'i_load_R01_b_i', 'i_load_R01_c_r', 'i_load_R01_c_i', 'i_load_R01_n_r', 'i_load_R01_n_i', 'i_load_R11_a_r', 'i_load_R11_a_i', 'i_load_R11_b_r', 'i_load_R11_b_i', 'i_load_R11_c_r', 'i_load_R11_c_i', 'i_load_R11_n_r', 'i_load_R11_n_i', 'i_load_R15_a_r', 'i_load_R15_a_i', 'i_load_R15_b_r', 'i_load_R15_b_i', 'i_load_R15_c_r', 'i_load_R15_c_i', 'i_load_R15_n_r', 'i_load_R15_n_i', 'i_load_R16_a_r', 'i_load_R16_a_i', 'i_load_R16_b_r', 'i_load_R16_b_i', 'i_load_R16_c_r', 'i_load_R16_c_i', 'i_load_R16_n_r', 'i_load_R16_n_i', 'i_load_R17_a_r', 'i_load_R17_a_i', 'i_load_R17_b_r', 'i_load_R17_b_i', 'i_load_R17_c_r', 'i_load_R17_c_i', 'i_load_R17_n_r', 'i_load_R17_n_i', 'i_load_R18_a_r', 'i_load_R18_a_i', 'i_load_R18_b_r', 'i_load_R18_b_i', 'i_load_R18_c_r', 'i_load_R18_c_i', 'i_load_R18_n_r', 'i_load_R18_n_i', 'i_load_I02_a_r', 'i_load_I02_a_i', 'i_load_I02_b_r', 'i_load_I02_b_i', 'i_load_I02_c_r', 'i_load_I02_c_i', 'i_load_I02_n_r', 'i_load_I02_n_i'] 
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

    sp_jac_ini_ia = [0, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 173, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 174, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 175, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 176, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 177, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 178, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 9, 10, 11, 12, 13, 14, 15, 16, 73, 74, 75, 76, 77, 78, 79, 80, 181, 9, 10, 11, 12, 13, 14, 15, 16, 73, 74, 75, 76, 77, 78, 79, 80, 182, 9, 10, 11, 12, 13, 14, 15, 16, 73, 74, 75, 76, 77, 78, 79, 80, 183, 9, 10, 11, 12, 13, 14, 15, 16, 73, 74, 75, 76, 77, 78, 79, 80, 184, 9, 10, 11, 12, 13, 14, 15, 16, 73, 74, 75, 76, 77, 78, 79, 80, 185, 9, 10, 11, 12, 13, 14, 15, 16, 73, 74, 75, 76, 77, 78, 79, 80, 186, 9, 10, 11, 12, 13, 14, 15, 16, 73, 74, 75, 76, 77, 78, 79, 80, 9, 10, 11, 12, 13, 14, 15, 16, 73, 74, 75, 76, 77, 78, 79, 80, 17, 18, 19, 20, 21, 22, 23, 24, 153, 154, 155, 156, 157, 158, 159, 160, 189, 17, 18, 19, 20, 21, 22, 23, 24, 153, 154, 155, 156, 157, 158, 159, 160, 190, 17, 18, 19, 20, 21, 22, 23, 24, 153, 154, 155, 156, 157, 158, 159, 160, 191, 17, 18, 19, 20, 21, 22, 23, 24, 153, 154, 155, 156, 157, 158, 159, 160, 192, 17, 18, 19, 20, 21, 22, 23, 24, 153, 154, 155, 156, 157, 158, 159, 160, 193, 17, 18, 19, 20, 21, 22, 23, 24, 153, 154, 155, 156, 157, 158, 159, 160, 194, 17, 18, 19, 20, 21, 22, 23, 24, 153, 154, 155, 156, 157, 158, 159, 160, 17, 18, 19, 20, 21, 22, 23, 24, 153, 154, 155, 156, 157, 158, 159, 160, 25, 26, 27, 28, 29, 30, 31, 32, 97, 98, 99, 100, 101, 102, 103, 104, 197, 25, 26, 27, 28, 29, 30, 31, 32, 97, 98, 99, 100, 101, 102, 103, 104, 198, 25, 26, 27, 28, 29, 30, 31, 32, 97, 98, 99, 100, 101, 102, 103, 104, 199, 25, 26, 27, 28, 29, 30, 31, 32, 97, 98, 99, 100, 101, 102, 103, 104, 200, 25, 26, 27, 28, 29, 30, 31, 32, 97, 98, 99, 100, 101, 102, 103, 104, 201, 25, 26, 27, 28, 29, 30, 31, 32, 97, 98, 99, 100, 101, 102, 103, 104, 202, 25, 26, 27, 28, 29, 30, 31, 32, 97, 98, 99, 100, 101, 102, 103, 104, 25, 26, 27, 28, 29, 30, 31, 32, 97, 98, 99, 100, 101, 102, 103, 104, 33, 34, 35, 36, 37, 38, 39, 40, 121, 122, 123, 124, 125, 126, 127, 128, 205, 33, 34, 35, 36, 37, 38, 39, 40, 121, 122, 123, 124, 125, 126, 127, 128, 206, 33, 34, 35, 36, 37, 38, 39, 40, 121, 122, 123, 124, 125, 126, 127, 128, 207, 33, 34, 35, 36, 37, 38, 39, 40, 121, 122, 123, 124, 125, 126, 127, 128, 208, 33, 34, 35, 36, 37, 38, 39, 40, 121, 122, 123, 124, 125, 126, 127, 128, 209, 33, 34, 35, 36, 37, 38, 39, 40, 121, 122, 123, 124, 125, 126, 127, 128, 210, 33, 34, 35, 36, 37, 38, 39, 40, 121, 122, 123, 124, 125, 126, 127, 128, 33, 34, 35, 36, 37, 38, 39, 40, 121, 122, 123, 124, 125, 126, 127, 128, 41, 42, 43, 44, 45, 46, 47, 48, 129, 130, 131, 132, 133, 134, 135, 136, 213, 41, 42, 43, 44, 45, 46, 47, 48, 129, 130, 131, 132, 133, 134, 135, 136, 214, 41, 42, 43, 44, 45, 46, 47, 48, 129, 130, 131, 132, 133, 134, 135, 136, 215, 41, 42, 43, 44, 45, 46, 47, 48, 129, 130, 131, 132, 133, 134, 135, 136, 216, 41, 42, 43, 44, 45, 46, 47, 48, 129, 130, 131, 132, 133, 134, 135, 136, 217, 41, 42, 43, 44, 45, 46, 47, 48, 129, 130, 131, 132, 133, 134, 135, 136, 218, 41, 42, 43, 44, 45, 46, 47, 48, 129, 130, 131, 132, 133, 134, 135, 136, 41, 42, 43, 44, 45, 46, 47, 48, 129, 130, 131, 132, 133, 134, 135, 136, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 221, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 222, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 223, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 224, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 225, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 226, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 9, 10, 11, 12, 13, 14, 15, 16, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 9, 10, 11, 12, 13, 14, 15, 16, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 9, 10, 11, 12, 13, 14, 15, 16, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 9, 10, 11, 12, 13, 14, 15, 16, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 9, 10, 11, 12, 13, 14, 15, 16, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 9, 10, 11, 12, 13, 14, 15, 16, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 9, 10, 11, 12, 13, 14, 15, 16, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 9, 10, 11, 12, 13, 14, 15, 16, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 137, 138, 139, 140, 141, 142, 143, 144, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 137, 138, 139, 140, 141, 142, 143, 144, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 137, 138, 139, 140, 141, 142, 143, 144, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 137, 138, 139, 140, 141, 142, 143, 144, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 137, 138, 139, 140, 141, 142, 143, 144, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 137, 138, 139, 140, 141, 142, 143, 144, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 137, 138, 139, 140, 141, 142, 143, 144, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 137, 138, 139, 140, 141, 142, 143, 144, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 25, 26, 27, 28, 29, 30, 31, 32, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 25, 26, 27, 28, 29, 30, 31, 32, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 25, 26, 27, 28, 29, 30, 31, 32, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 25, 26, 27, 28, 29, 30, 31, 32, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 25, 26, 27, 28, 29, 30, 31, 32, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 25, 26, 27, 28, 29, 30, 31, 32, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 25, 26, 27, 28, 29, 30, 31, 32, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 25, 26, 27, 28, 29, 30, 31, 32, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 33, 34, 35, 36, 37, 38, 39, 40, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 33, 34, 35, 36, 37, 38, 39, 40, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 33, 34, 35, 36, 37, 38, 39, 40, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 33, 34, 35, 36, 37, 38, 39, 40, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 33, 34, 35, 36, 37, 38, 39, 40, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 33, 34, 35, 36, 37, 38, 39, 40, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 33, 34, 35, 36, 37, 38, 39, 40, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 33, 34, 35, 36, 37, 38, 39, 40, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 41, 42, 43, 44, 45, 46, 47, 48, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 41, 42, 43, 44, 45, 46, 47, 48, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 41, 42, 43, 44, 45, 46, 47, 48, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 41, 42, 43, 44, 45, 46, 47, 48, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 41, 42, 43, 44, 45, 46, 47, 48, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 41, 42, 43, 44, 45, 46, 47, 48, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 41, 42, 43, 44, 45, 46, 47, 48, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 41, 42, 43, 44, 45, 46, 47, 48, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 81, 82, 83, 84, 85, 86, 87, 88, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 17, 18, 19, 20, 21, 22, 23, 24, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 17, 18, 19, 20, 21, 22, 23, 24, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 17, 18, 19, 20, 21, 22, 23, 24, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 17, 18, 19, 20, 21, 22, 23, 24, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 17, 18, 19, 20, 21, 22, 23, 24, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 17, 18, 19, 20, 21, 22, 23, 24, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 17, 18, 19, 20, 21, 22, 23, 24, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 17, 18, 19, 20, 21, 22, 23, 24, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 1, 2, 3, 4, 161, 1, 2, 3, 4, 162, 3, 4, 5, 6, 163, 3, 4, 5, 6, 164, 1, 2, 5, 6, 165, 1, 2, 5, 6, 166, 57, 58, 59, 60, 167, 57, 58, 59, 60, 168, 59, 60, 61, 62, 169, 59, 60, 61, 62, 170, 57, 58, 61, 62, 171, 57, 58, 61, 62, 172, 1, 2, 7, 8, 173, 174, 3, 4, 7, 8, 175, 176, 5, 6, 7, 8, 177, 178, 1, 2, 7, 8, 173, 174, 3, 4, 7, 8, 175, 176, 5, 6, 7, 8, 177, 178, 173, 175, 177, 179, 174, 176, 178, 180, 9, 10, 15, 16, 181, 182, 11, 12, 15, 16, 183, 184, 13, 14, 15, 16, 185, 186, 9, 10, 15, 16, 181, 182, 11, 12, 15, 16, 183, 184, 13, 14, 15, 16, 185, 186, 181, 183, 185, 187, 182, 184, 186, 188, 17, 18, 23, 24, 189, 190, 19, 20, 23, 24, 191, 192, 21, 22, 23, 24, 193, 194, 17, 18, 23, 24, 189, 190, 19, 20, 23, 24, 191, 192, 21, 22, 23, 24, 193, 194, 189, 191, 193, 195, 190, 192, 194, 196, 25, 26, 31, 32, 197, 198, 27, 28, 31, 32, 199, 200, 29, 30, 31, 32, 201, 202, 25, 26, 31, 32, 197, 198, 27, 28, 31, 32, 199, 200, 29, 30, 31, 32, 201, 202, 197, 199, 201, 203, 198, 200, 202, 204, 33, 34, 39, 40, 205, 206, 35, 36, 39, 40, 207, 208, 37, 38, 39, 40, 209, 210, 33, 34, 39, 40, 205, 206, 35, 36, 39, 40, 207, 208, 37, 38, 39, 40, 209, 210, 205, 207, 209, 211, 206, 208, 210, 212, 41, 42, 47, 48, 213, 214, 43, 44, 47, 48, 215, 216, 45, 46, 47, 48, 217, 218, 41, 42, 47, 48, 213, 214, 43, 44, 47, 48, 215, 216, 45, 46, 47, 48, 217, 218, 213, 215, 217, 219, 214, 216, 218, 220, 49, 50, 55, 56, 221, 222, 51, 52, 55, 56, 223, 224, 53, 54, 55, 56, 225, 226, 49, 50, 55, 56, 221, 222, 51, 52, 55, 56, 223, 224, 53, 54, 55, 56, 225, 226, 221, 223, 225, 227, 222, 224, 226, 228]
    sp_jac_ini_ja = [0, 1, 18, 35, 52, 69, 86, 103, 119, 135, 152, 169, 186, 203, 220, 237, 253, 269, 286, 303, 320, 337, 354, 371, 387, 403, 420, 437, 454, 471, 488, 505, 521, 537, 554, 571, 588, 605, 622, 639, 655, 671, 688, 705, 722, 739, 756, 773, 789, 805, 822, 839, 856, 873, 890, 907, 923, 939, 955, 971, 987, 1003, 1019, 1035, 1051, 1067, 1091, 1115, 1139, 1163, 1187, 1211, 1235, 1259, 1291, 1323, 1355, 1387, 1419, 1451, 1483, 1515, 1547, 1579, 1611, 1643, 1675, 1707, 1739, 1771, 1795, 1819, 1843, 1867, 1891, 1915, 1939, 1963, 1995, 2027, 2059, 2091, 2123, 2155, 2187, 2219, 2243, 2267, 2291, 2315, 2339, 2363, 2387, 2411, 2435, 2459, 2483, 2507, 2531, 2555, 2579, 2603, 2635, 2667, 2699, 2731, 2763, 2795, 2827, 2859, 2883, 2907, 2931, 2955, 2979, 3003, 3027, 3051, 3075, 3099, 3123, 3147, 3171, 3195, 3219, 3243, 3267, 3291, 3315, 3339, 3363, 3387, 3411, 3435, 3459, 3483, 3507, 3531, 3555, 3579, 3603, 3627, 3632, 3637, 3642, 3647, 3652, 3657, 3662, 3667, 3672, 3677, 3682, 3687, 3693, 3699, 3705, 3711, 3717, 3723, 3727, 3731, 3737, 3743, 3749, 3755, 3761, 3767, 3771, 3775, 3781, 3787, 3793, 3799, 3805, 3811, 3815, 3819, 3825, 3831, 3837, 3843, 3849, 3855, 3859, 3863, 3869, 3875, 3881, 3887, 3893, 3899, 3903, 3907, 3913, 3919, 3925, 3931, 3937, 3943, 3947, 3951, 3957, 3963, 3969, 3975, 3981, 3987, 3991, 3995]
    sp_jac_ini_nia = 229
    sp_jac_ini_nja = 229
    return sp_jac_ini_ia, sp_jac_ini_ja, sp_jac_ini_nia, sp_jac_ini_nja 

def sp_jac_trap_vectors():

    sp_jac_trap_ia = [0, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 173, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 174, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 175, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 176, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 177, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 178, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 9, 10, 11, 12, 13, 14, 15, 16, 73, 74, 75, 76, 77, 78, 79, 80, 181, 9, 10, 11, 12, 13, 14, 15, 16, 73, 74, 75, 76, 77, 78, 79, 80, 182, 9, 10, 11, 12, 13, 14, 15, 16, 73, 74, 75, 76, 77, 78, 79, 80, 183, 9, 10, 11, 12, 13, 14, 15, 16, 73, 74, 75, 76, 77, 78, 79, 80, 184, 9, 10, 11, 12, 13, 14, 15, 16, 73, 74, 75, 76, 77, 78, 79, 80, 185, 9, 10, 11, 12, 13, 14, 15, 16, 73, 74, 75, 76, 77, 78, 79, 80, 186, 9, 10, 11, 12, 13, 14, 15, 16, 73, 74, 75, 76, 77, 78, 79, 80, 9, 10, 11, 12, 13, 14, 15, 16, 73, 74, 75, 76, 77, 78, 79, 80, 17, 18, 19, 20, 21, 22, 23, 24, 153, 154, 155, 156, 157, 158, 159, 160, 189, 17, 18, 19, 20, 21, 22, 23, 24, 153, 154, 155, 156, 157, 158, 159, 160, 190, 17, 18, 19, 20, 21, 22, 23, 24, 153, 154, 155, 156, 157, 158, 159, 160, 191, 17, 18, 19, 20, 21, 22, 23, 24, 153, 154, 155, 156, 157, 158, 159, 160, 192, 17, 18, 19, 20, 21, 22, 23, 24, 153, 154, 155, 156, 157, 158, 159, 160, 193, 17, 18, 19, 20, 21, 22, 23, 24, 153, 154, 155, 156, 157, 158, 159, 160, 194, 17, 18, 19, 20, 21, 22, 23, 24, 153, 154, 155, 156, 157, 158, 159, 160, 17, 18, 19, 20, 21, 22, 23, 24, 153, 154, 155, 156, 157, 158, 159, 160, 25, 26, 27, 28, 29, 30, 31, 32, 97, 98, 99, 100, 101, 102, 103, 104, 197, 25, 26, 27, 28, 29, 30, 31, 32, 97, 98, 99, 100, 101, 102, 103, 104, 198, 25, 26, 27, 28, 29, 30, 31, 32, 97, 98, 99, 100, 101, 102, 103, 104, 199, 25, 26, 27, 28, 29, 30, 31, 32, 97, 98, 99, 100, 101, 102, 103, 104, 200, 25, 26, 27, 28, 29, 30, 31, 32, 97, 98, 99, 100, 101, 102, 103, 104, 201, 25, 26, 27, 28, 29, 30, 31, 32, 97, 98, 99, 100, 101, 102, 103, 104, 202, 25, 26, 27, 28, 29, 30, 31, 32, 97, 98, 99, 100, 101, 102, 103, 104, 25, 26, 27, 28, 29, 30, 31, 32, 97, 98, 99, 100, 101, 102, 103, 104, 33, 34, 35, 36, 37, 38, 39, 40, 121, 122, 123, 124, 125, 126, 127, 128, 205, 33, 34, 35, 36, 37, 38, 39, 40, 121, 122, 123, 124, 125, 126, 127, 128, 206, 33, 34, 35, 36, 37, 38, 39, 40, 121, 122, 123, 124, 125, 126, 127, 128, 207, 33, 34, 35, 36, 37, 38, 39, 40, 121, 122, 123, 124, 125, 126, 127, 128, 208, 33, 34, 35, 36, 37, 38, 39, 40, 121, 122, 123, 124, 125, 126, 127, 128, 209, 33, 34, 35, 36, 37, 38, 39, 40, 121, 122, 123, 124, 125, 126, 127, 128, 210, 33, 34, 35, 36, 37, 38, 39, 40, 121, 122, 123, 124, 125, 126, 127, 128, 33, 34, 35, 36, 37, 38, 39, 40, 121, 122, 123, 124, 125, 126, 127, 128, 41, 42, 43, 44, 45, 46, 47, 48, 129, 130, 131, 132, 133, 134, 135, 136, 213, 41, 42, 43, 44, 45, 46, 47, 48, 129, 130, 131, 132, 133, 134, 135, 136, 214, 41, 42, 43, 44, 45, 46, 47, 48, 129, 130, 131, 132, 133, 134, 135, 136, 215, 41, 42, 43, 44, 45, 46, 47, 48, 129, 130, 131, 132, 133, 134, 135, 136, 216, 41, 42, 43, 44, 45, 46, 47, 48, 129, 130, 131, 132, 133, 134, 135, 136, 217, 41, 42, 43, 44, 45, 46, 47, 48, 129, 130, 131, 132, 133, 134, 135, 136, 218, 41, 42, 43, 44, 45, 46, 47, 48, 129, 130, 131, 132, 133, 134, 135, 136, 41, 42, 43, 44, 45, 46, 47, 48, 129, 130, 131, 132, 133, 134, 135, 136, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 221, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 222, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 223, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 224, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 225, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 226, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 1, 2, 3, 4, 5, 6, 7, 8, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 9, 10, 11, 12, 13, 14, 15, 16, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 9, 10, 11, 12, 13, 14, 15, 16, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 9, 10, 11, 12, 13, 14, 15, 16, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 9, 10, 11, 12, 13, 14, 15, 16, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 9, 10, 11, 12, 13, 14, 15, 16, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 9, 10, 11, 12, 13, 14, 15, 16, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 9, 10, 11, 12, 13, 14, 15, 16, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 9, 10, 11, 12, 13, 14, 15, 16, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 137, 138, 139, 140, 141, 142, 143, 144, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 137, 138, 139, 140, 141, 142, 143, 144, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 137, 138, 139, 140, 141, 142, 143, 144, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 137, 138, 139, 140, 141, 142, 143, 144, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 137, 138, 139, 140, 141, 142, 143, 144, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 137, 138, 139, 140, 141, 142, 143, 144, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 137, 138, 139, 140, 141, 142, 143, 144, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 137, 138, 139, 140, 141, 142, 143, 144, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 25, 26, 27, 28, 29, 30, 31, 32, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 25, 26, 27, 28, 29, 30, 31, 32, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 25, 26, 27, 28, 29, 30, 31, 32, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 25, 26, 27, 28, 29, 30, 31, 32, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 25, 26, 27, 28, 29, 30, 31, 32, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 25, 26, 27, 28, 29, 30, 31, 32, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 25, 26, 27, 28, 29, 30, 31, 32, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 25, 26, 27, 28, 29, 30, 31, 32, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 33, 34, 35, 36, 37, 38, 39, 40, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 33, 34, 35, 36, 37, 38, 39, 40, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 33, 34, 35, 36, 37, 38, 39, 40, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 33, 34, 35, 36, 37, 38, 39, 40, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 33, 34, 35, 36, 37, 38, 39, 40, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 33, 34, 35, 36, 37, 38, 39, 40, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 33, 34, 35, 36, 37, 38, 39, 40, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 33, 34, 35, 36, 37, 38, 39, 40, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 41, 42, 43, 44, 45, 46, 47, 48, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 41, 42, 43, 44, 45, 46, 47, 48, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 41, 42, 43, 44, 45, 46, 47, 48, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 41, 42, 43, 44, 45, 46, 47, 48, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 41, 42, 43, 44, 45, 46, 47, 48, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 41, 42, 43, 44, 45, 46, 47, 48, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 41, 42, 43, 44, 45, 46, 47, 48, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 41, 42, 43, 44, 45, 46, 47, 48, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 81, 82, 83, 84, 85, 86, 87, 88, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 81, 82, 83, 84, 85, 86, 87, 88, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 17, 18, 19, 20, 21, 22, 23, 24, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 17, 18, 19, 20, 21, 22, 23, 24, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 17, 18, 19, 20, 21, 22, 23, 24, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 17, 18, 19, 20, 21, 22, 23, 24, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 17, 18, 19, 20, 21, 22, 23, 24, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 17, 18, 19, 20, 21, 22, 23, 24, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 17, 18, 19, 20, 21, 22, 23, 24, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 17, 18, 19, 20, 21, 22, 23, 24, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 1, 2, 3, 4, 161, 1, 2, 3, 4, 162, 3, 4, 5, 6, 163, 3, 4, 5, 6, 164, 1, 2, 5, 6, 165, 1, 2, 5, 6, 166, 57, 58, 59, 60, 167, 57, 58, 59, 60, 168, 59, 60, 61, 62, 169, 59, 60, 61, 62, 170, 57, 58, 61, 62, 171, 57, 58, 61, 62, 172, 1, 2, 7, 8, 173, 174, 3, 4, 7, 8, 175, 176, 5, 6, 7, 8, 177, 178, 1, 2, 7, 8, 173, 174, 3, 4, 7, 8, 175, 176, 5, 6, 7, 8, 177, 178, 173, 175, 177, 179, 174, 176, 178, 180, 9, 10, 15, 16, 181, 182, 11, 12, 15, 16, 183, 184, 13, 14, 15, 16, 185, 186, 9, 10, 15, 16, 181, 182, 11, 12, 15, 16, 183, 184, 13, 14, 15, 16, 185, 186, 181, 183, 185, 187, 182, 184, 186, 188, 17, 18, 23, 24, 189, 190, 19, 20, 23, 24, 191, 192, 21, 22, 23, 24, 193, 194, 17, 18, 23, 24, 189, 190, 19, 20, 23, 24, 191, 192, 21, 22, 23, 24, 193, 194, 189, 191, 193, 195, 190, 192, 194, 196, 25, 26, 31, 32, 197, 198, 27, 28, 31, 32, 199, 200, 29, 30, 31, 32, 201, 202, 25, 26, 31, 32, 197, 198, 27, 28, 31, 32, 199, 200, 29, 30, 31, 32, 201, 202, 197, 199, 201, 203, 198, 200, 202, 204, 33, 34, 39, 40, 205, 206, 35, 36, 39, 40, 207, 208, 37, 38, 39, 40, 209, 210, 33, 34, 39, 40, 205, 206, 35, 36, 39, 40, 207, 208, 37, 38, 39, 40, 209, 210, 205, 207, 209, 211, 206, 208, 210, 212, 41, 42, 47, 48, 213, 214, 43, 44, 47, 48, 215, 216, 45, 46, 47, 48, 217, 218, 41, 42, 47, 48, 213, 214, 43, 44, 47, 48, 215, 216, 45, 46, 47, 48, 217, 218, 213, 215, 217, 219, 214, 216, 218, 220, 49, 50, 55, 56, 221, 222, 51, 52, 55, 56, 223, 224, 53, 54, 55, 56, 225, 226, 49, 50, 55, 56, 221, 222, 51, 52, 55, 56, 223, 224, 53, 54, 55, 56, 225, 226, 221, 223, 225, 227, 222, 224, 226, 228]
    sp_jac_trap_ja = [0, 1, 18, 35, 52, 69, 86, 103, 119, 135, 152, 169, 186, 203, 220, 237, 253, 269, 286, 303, 320, 337, 354, 371, 387, 403, 420, 437, 454, 471, 488, 505, 521, 537, 554, 571, 588, 605, 622, 639, 655, 671, 688, 705, 722, 739, 756, 773, 789, 805, 822, 839, 856, 873, 890, 907, 923, 939, 955, 971, 987, 1003, 1019, 1035, 1051, 1067, 1091, 1115, 1139, 1163, 1187, 1211, 1235, 1259, 1291, 1323, 1355, 1387, 1419, 1451, 1483, 1515, 1547, 1579, 1611, 1643, 1675, 1707, 1739, 1771, 1795, 1819, 1843, 1867, 1891, 1915, 1939, 1963, 1995, 2027, 2059, 2091, 2123, 2155, 2187, 2219, 2243, 2267, 2291, 2315, 2339, 2363, 2387, 2411, 2435, 2459, 2483, 2507, 2531, 2555, 2579, 2603, 2635, 2667, 2699, 2731, 2763, 2795, 2827, 2859, 2883, 2907, 2931, 2955, 2979, 3003, 3027, 3051, 3075, 3099, 3123, 3147, 3171, 3195, 3219, 3243, 3267, 3291, 3315, 3339, 3363, 3387, 3411, 3435, 3459, 3483, 3507, 3531, 3555, 3579, 3603, 3627, 3632, 3637, 3642, 3647, 3652, 3657, 3662, 3667, 3672, 3677, 3682, 3687, 3693, 3699, 3705, 3711, 3717, 3723, 3727, 3731, 3737, 3743, 3749, 3755, 3761, 3767, 3771, 3775, 3781, 3787, 3793, 3799, 3805, 3811, 3815, 3819, 3825, 3831, 3837, 3843, 3849, 3855, 3859, 3863, 3869, 3875, 3881, 3887, 3893, 3899, 3903, 3907, 3913, 3919, 3925, 3931, 3937, 3943, 3947, 3951, 3957, 3963, 3969, 3975, 3981, 3987, 3991, 3995]
    sp_jac_trap_nia = 229
    sp_jac_trap_nja = 229
    return sp_jac_trap_ia, sp_jac_trap_ja, sp_jac_trap_nia, sp_jac_trap_nja 
