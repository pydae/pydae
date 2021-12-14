import numpy as np
import numba
import scipy.optimize as sopt
import scipy.sparse as sspa
from scipy.sparse.linalg import spsolve,spilu,splu
from numba import cuda
import cffi
import numba.core.typing.cffi_utils as cffi_support

ffi = cffi.FFI()

import cigre_eu_lv_cffi as jacs

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


class cigre_eu_lv_class: 

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
        self.N_y = 458 
        self.N_z = 459 
        self.N_store = 10000 
        self.params_list = [] 
        self.params_values_list  = [] 
        self.inputs_ini_list = ['v_MV0_a_r', 'v_MV0_a_i', 'v_MV0_b_r', 'v_MV0_b_i', 'v_MV0_c_r', 'v_MV0_c_i', 'i_R01_n_r', 'i_R01_n_i', 'i_R11_n_r', 'i_R11_n_i', 'i_R15_n_r', 'i_R15_n_i', 'i_R16_n_r', 'i_R16_n_i', 'i_R17_n_r', 'i_R17_n_i', 'i_R18_n_r', 'i_R18_n_i', 'i_I02_n_r', 'i_I02_n_i', 'i_C01_n_r', 'i_C01_n_i', 'i_C12_n_r', 'i_C12_n_i', 'i_C13_n_r', 'i_C13_n_i', 'i_C14_n_r', 'i_C14_n_i', 'i_C17_n_r', 'i_C17_n_i', 'i_C18_n_r', 'i_C18_n_i', 'i_C19_n_r', 'i_C19_n_i', 'i_C20_n_r', 'i_C20_n_i', 'i_I01_a_r', 'i_I01_a_i', 'i_I01_b_r', 'i_I01_b_i', 'i_I01_c_r', 'i_I01_c_i', 'i_I01_n_r', 'i_I01_n_i', 'i_R02_a_r', 'i_R02_a_i', 'i_R02_b_r', 'i_R02_b_i', 'i_R02_c_r', 'i_R02_c_i', 'i_R02_n_r', 'i_R02_n_i', 'i_R03_a_r', 'i_R03_a_i', 'i_R03_b_r', 'i_R03_b_i', 'i_R03_c_r', 'i_R03_c_i', 'i_R03_n_r', 'i_R03_n_i', 'i_R04_a_r', 'i_R04_a_i', 'i_R04_b_r', 'i_R04_b_i', 'i_R04_c_r', 'i_R04_c_i', 'i_R04_n_r', 'i_R04_n_i', 'i_R05_a_r', 'i_R05_a_i', 'i_R05_b_r', 'i_R05_b_i', 'i_R05_c_r', 'i_R05_c_i', 'i_R05_n_r', 'i_R05_n_i', 'i_R06_a_r', 'i_R06_a_i', 'i_R06_b_r', 'i_R06_b_i', 'i_R06_c_r', 'i_R06_c_i', 'i_R06_n_r', 'i_R06_n_i', 'i_R07_a_r', 'i_R07_a_i', 'i_R07_b_r', 'i_R07_b_i', 'i_R07_c_r', 'i_R07_c_i', 'i_R07_n_r', 'i_R07_n_i', 'i_R08_a_r', 'i_R08_a_i', 'i_R08_b_r', 'i_R08_b_i', 'i_R08_c_r', 'i_R08_c_i', 'i_R08_n_r', 'i_R08_n_i', 'i_R09_a_r', 'i_R09_a_i', 'i_R09_b_r', 'i_R09_b_i', 'i_R09_c_r', 'i_R09_c_i', 'i_R09_n_r', 'i_R09_n_i', 'i_R10_a_r', 'i_R10_a_i', 'i_R10_b_r', 'i_R10_b_i', 'i_R10_c_r', 'i_R10_c_i', 'i_R10_n_r', 'i_R10_n_i', 'i_R12_a_r', 'i_R12_a_i', 'i_R12_b_r', 'i_R12_b_i', 'i_R12_c_r', 'i_R12_c_i', 'i_R12_n_r', 'i_R12_n_i', 'i_R13_a_r', 'i_R13_a_i', 'i_R13_b_r', 'i_R13_b_i', 'i_R13_c_r', 'i_R13_c_i', 'i_R13_n_r', 'i_R13_n_i', 'i_R14_a_r', 'i_R14_a_i', 'i_R14_b_r', 'i_R14_b_i', 'i_R14_c_r', 'i_R14_c_i', 'i_R14_n_r', 'i_R14_n_i', 'i_C02_a_r', 'i_C02_a_i', 'i_C02_b_r', 'i_C02_b_i', 'i_C02_c_r', 'i_C02_c_i', 'i_C02_n_r', 'i_C02_n_i', 'i_C03_a_r', 'i_C03_a_i', 'i_C03_b_r', 'i_C03_b_i', 'i_C03_c_r', 'i_C03_c_i', 'i_C03_n_r', 'i_C03_n_i', 'i_C04_a_r', 'i_C04_a_i', 'i_C04_b_r', 'i_C04_b_i', 'i_C04_c_r', 'i_C04_c_i', 'i_C04_n_r', 'i_C04_n_i', 'i_C05_a_r', 'i_C05_a_i', 'i_C05_b_r', 'i_C05_b_i', 'i_C05_c_r', 'i_C05_c_i', 'i_C05_n_r', 'i_C05_n_i', 'i_C06_a_r', 'i_C06_a_i', 'i_C06_b_r', 'i_C06_b_i', 'i_C06_c_r', 'i_C06_c_i', 'i_C06_n_r', 'i_C06_n_i', 'i_C07_a_r', 'i_C07_a_i', 'i_C07_b_r', 'i_C07_b_i', 'i_C07_c_r', 'i_C07_c_i', 'i_C07_n_r', 'i_C07_n_i', 'i_C08_a_r', 'i_C08_a_i', 'i_C08_b_r', 'i_C08_b_i', 'i_C08_c_r', 'i_C08_c_i', 'i_C08_n_r', 'i_C08_n_i', 'i_C09_a_r', 'i_C09_a_i', 'i_C09_b_r', 'i_C09_b_i', 'i_C09_c_r', 'i_C09_c_i', 'i_C09_n_r', 'i_C09_n_i', 'i_C10_a_r', 'i_C10_a_i', 'i_C10_b_r', 'i_C10_b_i', 'i_C10_c_r', 'i_C10_c_i', 'i_C10_n_r', 'i_C10_n_i', 'i_C11_a_r', 'i_C11_a_i', 'i_C11_b_r', 'i_C11_b_i', 'i_C11_c_r', 'i_C11_c_i', 'i_C11_n_r', 'i_C11_n_i', 'i_C15_a_r', 'i_C15_a_i', 'i_C15_b_r', 'i_C15_b_i', 'i_C15_c_r', 'i_C15_c_i', 'i_C15_n_r', 'i_C15_n_i', 'i_C16_a_r', 'i_C16_a_i', 'i_C16_b_r', 'i_C16_b_i', 'i_C16_c_r', 'i_C16_c_i', 'i_C16_n_r', 'i_C16_n_i', 'p_R01_a', 'q_R01_a', 'p_R01_b', 'q_R01_b', 'p_R01_c', 'q_R01_c', 'p_R11_a', 'q_R11_a', 'p_R11_b', 'q_R11_b', 'p_R11_c', 'q_R11_c', 'p_R15_a', 'q_R15_a', 'p_R15_b', 'q_R15_b', 'p_R15_c', 'q_R15_c', 'p_R16_a', 'q_R16_a', 'p_R16_b', 'q_R16_b', 'p_R16_c', 'q_R16_c', 'p_R17_a', 'q_R17_a', 'p_R17_b', 'q_R17_b', 'p_R17_c', 'q_R17_c', 'p_R18_a', 'q_R18_a', 'p_R18_b', 'q_R18_b', 'p_R18_c', 'q_R18_c', 'p_I02_a', 'q_I02_a', 'p_I02_b', 'q_I02_b', 'p_I02_c', 'q_I02_c', 'p_C01_a', 'q_C01_a', 'p_C01_b', 'q_C01_b', 'p_C01_c', 'q_C01_c', 'p_C12_a', 'q_C12_a', 'p_C12_b', 'q_C12_b', 'p_C12_c', 'q_C12_c', 'p_C13_a', 'q_C13_a', 'p_C13_b', 'q_C13_b', 'p_C13_c', 'q_C13_c', 'p_C14_a', 'q_C14_a', 'p_C14_b', 'q_C14_b', 'p_C14_c', 'q_C14_c', 'p_C17_a', 'q_C17_a', 'p_C17_b', 'q_C17_b', 'p_C17_c', 'q_C17_c', 'p_C18_a', 'q_C18_a', 'p_C18_b', 'q_C18_b', 'p_C18_c', 'q_C18_c', 'p_C19_a', 'q_C19_a', 'p_C19_b', 'q_C19_b', 'p_C19_c', 'q_C19_c', 'p_C20_a', 'q_C20_a', 'p_C20_b', 'q_C20_b', 'p_C20_c', 'q_C20_c', 'u_dummy'] 
        self.inputs_ini_values_list  = [11547.0, 0.0, -5773.499999999997, -9999.995337498915, -5773.5000000000055, 9999.99533749891, -0.11377181077948251, -0.062435919130905404, 0.03892177301699995, -0.1048963759941941, 0.3445405964303525, -0.7980789681018621, 0.3735765711084902, -0.8707524308128995, 0.30548249608932565, -0.6990681853106011, 0.4382865509675824, -0.9973969692525912, 0.5602845598552193, -1.4893326751963798, -0.007044033458385002, -0.06484125482364789, 0.14160886193487165, -0.24241047488061085, 0.14160886193487165, -0.24241047488061085, 0.1528369816077202, -0.2688059521539401, 0.21660624003369477, -0.3687378481287098, 0.05704213601223884, -0.10272328888077098, 0.12179605046912911, -0.22682846938669243, 0.05965654126371245, -0.11267943612286757, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -63333.33333333113, -20816.65999466039, -63333.333333332266, -20816.65999466152, -63333.33333333332, -20816.659994661335, -4750.000000000083, -1561.2494995993147, -4749.999999999877, -1561.2494995993716, -4750.0000000002865, -1561.2494995995303, -16466.66666666464, -5412.331598610512, -16466.66666666773, -5412.331598611147, -16466.666666666926, -5412.33159861298, -17416.66666666622, -5724.581498531829, -17416.666666668083, -5724.581498530997, -17416.666666667945, -5724.5814985310335, -11083.333333332897, -3642.9154990658376, -11083.333333333037, -3642.9154990651805, -11083.333333334114, -3642.9154990664397, -14883.333333333507, -4891.9150987456205, -14883.333333333638, -4891.915098745632, -14883.33333333336, -4891.915098744581, -28333.333335939387, -17559.42292349193, -28333.333325891297, -17559.42292612909, -28333.333338265, -17559.422914521434, -36000.000000000815, -17435.595774162168, -35999.99999999903, -17435.59577416274, -36000.0000000003, -17435.59577416212, -5999.999999999479, -2905.932629027591, -5999.9999999997335, -2905.932629026383, -6000.000000000693, -2905.932629027016, -5999.999999999479, -2905.932629027591, -5999.9999999997335, -2905.932629026383, -6000.000000000693, -2905.932629027016, -7499.999999999482, -3632.4157862838842, -7500.000000000354, -3632.4157862838765, -7500.000000000065, -3632.4157862839256, -7499.99999999908, -3632.415786284226, -7500.000000000483, -3632.4157862834372, -7500.000000000937, -3632.415786283963, -2399.999999999841, -1162.3730516108112, -2400.0000000000214, -1162.3730516108224, -2400.0000000001646, -1162.3730516106748, -4799.999999999371, -2324.7461032218184, -4799.999999999937, -2324.746103221286, -4800.000000000485, -2324.7461032219417, -2399.9999999996926, -1162.3730516108767, -2400.000000000042, -1162.373051610754, -2400.0000000002724, -1162.3730516109106, 1.0] 
        self.inputs_run_list = ['v_MV0_a_r', 'v_MV0_a_i', 'v_MV0_b_r', 'v_MV0_b_i', 'v_MV0_c_r', 'v_MV0_c_i', 'i_R01_n_r', 'i_R01_n_i', 'i_R11_n_r', 'i_R11_n_i', 'i_R15_n_r', 'i_R15_n_i', 'i_R16_n_r', 'i_R16_n_i', 'i_R17_n_r', 'i_R17_n_i', 'i_R18_n_r', 'i_R18_n_i', 'i_I02_n_r', 'i_I02_n_i', 'i_C01_n_r', 'i_C01_n_i', 'i_C12_n_r', 'i_C12_n_i', 'i_C13_n_r', 'i_C13_n_i', 'i_C14_n_r', 'i_C14_n_i', 'i_C17_n_r', 'i_C17_n_i', 'i_C18_n_r', 'i_C18_n_i', 'i_C19_n_r', 'i_C19_n_i', 'i_C20_n_r', 'i_C20_n_i', 'i_I01_a_r', 'i_I01_a_i', 'i_I01_b_r', 'i_I01_b_i', 'i_I01_c_r', 'i_I01_c_i', 'i_I01_n_r', 'i_I01_n_i', 'i_R02_a_r', 'i_R02_a_i', 'i_R02_b_r', 'i_R02_b_i', 'i_R02_c_r', 'i_R02_c_i', 'i_R02_n_r', 'i_R02_n_i', 'i_R03_a_r', 'i_R03_a_i', 'i_R03_b_r', 'i_R03_b_i', 'i_R03_c_r', 'i_R03_c_i', 'i_R03_n_r', 'i_R03_n_i', 'i_R04_a_r', 'i_R04_a_i', 'i_R04_b_r', 'i_R04_b_i', 'i_R04_c_r', 'i_R04_c_i', 'i_R04_n_r', 'i_R04_n_i', 'i_R05_a_r', 'i_R05_a_i', 'i_R05_b_r', 'i_R05_b_i', 'i_R05_c_r', 'i_R05_c_i', 'i_R05_n_r', 'i_R05_n_i', 'i_R06_a_r', 'i_R06_a_i', 'i_R06_b_r', 'i_R06_b_i', 'i_R06_c_r', 'i_R06_c_i', 'i_R06_n_r', 'i_R06_n_i', 'i_R07_a_r', 'i_R07_a_i', 'i_R07_b_r', 'i_R07_b_i', 'i_R07_c_r', 'i_R07_c_i', 'i_R07_n_r', 'i_R07_n_i', 'i_R08_a_r', 'i_R08_a_i', 'i_R08_b_r', 'i_R08_b_i', 'i_R08_c_r', 'i_R08_c_i', 'i_R08_n_r', 'i_R08_n_i', 'i_R09_a_r', 'i_R09_a_i', 'i_R09_b_r', 'i_R09_b_i', 'i_R09_c_r', 'i_R09_c_i', 'i_R09_n_r', 'i_R09_n_i', 'i_R10_a_r', 'i_R10_a_i', 'i_R10_b_r', 'i_R10_b_i', 'i_R10_c_r', 'i_R10_c_i', 'i_R10_n_r', 'i_R10_n_i', 'i_R12_a_r', 'i_R12_a_i', 'i_R12_b_r', 'i_R12_b_i', 'i_R12_c_r', 'i_R12_c_i', 'i_R12_n_r', 'i_R12_n_i', 'i_R13_a_r', 'i_R13_a_i', 'i_R13_b_r', 'i_R13_b_i', 'i_R13_c_r', 'i_R13_c_i', 'i_R13_n_r', 'i_R13_n_i', 'i_R14_a_r', 'i_R14_a_i', 'i_R14_b_r', 'i_R14_b_i', 'i_R14_c_r', 'i_R14_c_i', 'i_R14_n_r', 'i_R14_n_i', 'i_C02_a_r', 'i_C02_a_i', 'i_C02_b_r', 'i_C02_b_i', 'i_C02_c_r', 'i_C02_c_i', 'i_C02_n_r', 'i_C02_n_i', 'i_C03_a_r', 'i_C03_a_i', 'i_C03_b_r', 'i_C03_b_i', 'i_C03_c_r', 'i_C03_c_i', 'i_C03_n_r', 'i_C03_n_i', 'i_C04_a_r', 'i_C04_a_i', 'i_C04_b_r', 'i_C04_b_i', 'i_C04_c_r', 'i_C04_c_i', 'i_C04_n_r', 'i_C04_n_i', 'i_C05_a_r', 'i_C05_a_i', 'i_C05_b_r', 'i_C05_b_i', 'i_C05_c_r', 'i_C05_c_i', 'i_C05_n_r', 'i_C05_n_i', 'i_C06_a_r', 'i_C06_a_i', 'i_C06_b_r', 'i_C06_b_i', 'i_C06_c_r', 'i_C06_c_i', 'i_C06_n_r', 'i_C06_n_i', 'i_C07_a_r', 'i_C07_a_i', 'i_C07_b_r', 'i_C07_b_i', 'i_C07_c_r', 'i_C07_c_i', 'i_C07_n_r', 'i_C07_n_i', 'i_C08_a_r', 'i_C08_a_i', 'i_C08_b_r', 'i_C08_b_i', 'i_C08_c_r', 'i_C08_c_i', 'i_C08_n_r', 'i_C08_n_i', 'i_C09_a_r', 'i_C09_a_i', 'i_C09_b_r', 'i_C09_b_i', 'i_C09_c_r', 'i_C09_c_i', 'i_C09_n_r', 'i_C09_n_i', 'i_C10_a_r', 'i_C10_a_i', 'i_C10_b_r', 'i_C10_b_i', 'i_C10_c_r', 'i_C10_c_i', 'i_C10_n_r', 'i_C10_n_i', 'i_C11_a_r', 'i_C11_a_i', 'i_C11_b_r', 'i_C11_b_i', 'i_C11_c_r', 'i_C11_c_i', 'i_C11_n_r', 'i_C11_n_i', 'i_C15_a_r', 'i_C15_a_i', 'i_C15_b_r', 'i_C15_b_i', 'i_C15_c_r', 'i_C15_c_i', 'i_C15_n_r', 'i_C15_n_i', 'i_C16_a_r', 'i_C16_a_i', 'i_C16_b_r', 'i_C16_b_i', 'i_C16_c_r', 'i_C16_c_i', 'i_C16_n_r', 'i_C16_n_i', 'p_R01_a', 'q_R01_a', 'p_R01_b', 'q_R01_b', 'p_R01_c', 'q_R01_c', 'p_R11_a', 'q_R11_a', 'p_R11_b', 'q_R11_b', 'p_R11_c', 'q_R11_c', 'p_R15_a', 'q_R15_a', 'p_R15_b', 'q_R15_b', 'p_R15_c', 'q_R15_c', 'p_R16_a', 'q_R16_a', 'p_R16_b', 'q_R16_b', 'p_R16_c', 'q_R16_c', 'p_R17_a', 'q_R17_a', 'p_R17_b', 'q_R17_b', 'p_R17_c', 'q_R17_c', 'p_R18_a', 'q_R18_a', 'p_R18_b', 'q_R18_b', 'p_R18_c', 'q_R18_c', 'p_I02_a', 'q_I02_a', 'p_I02_b', 'q_I02_b', 'p_I02_c', 'q_I02_c', 'p_C01_a', 'q_C01_a', 'p_C01_b', 'q_C01_b', 'p_C01_c', 'q_C01_c', 'p_C12_a', 'q_C12_a', 'p_C12_b', 'q_C12_b', 'p_C12_c', 'q_C12_c', 'p_C13_a', 'q_C13_a', 'p_C13_b', 'q_C13_b', 'p_C13_c', 'q_C13_c', 'p_C14_a', 'q_C14_a', 'p_C14_b', 'q_C14_b', 'p_C14_c', 'q_C14_c', 'p_C17_a', 'q_C17_a', 'p_C17_b', 'q_C17_b', 'p_C17_c', 'q_C17_c', 'p_C18_a', 'q_C18_a', 'p_C18_b', 'q_C18_b', 'p_C18_c', 'q_C18_c', 'p_C19_a', 'q_C19_a', 'p_C19_b', 'q_C19_b', 'p_C19_c', 'q_C19_c', 'p_C20_a', 'q_C20_a', 'p_C20_b', 'q_C20_b', 'p_C20_c', 'q_C20_c', 'u_dummy'] 
        self.inputs_run_values_list = [11547.0, 0.0, -5773.499999999997, -9999.995337498915, -5773.5000000000055, 9999.99533749891, -0.11377181077948251, -0.062435919130905404, 0.03892177301699995, -0.1048963759941941, 0.3445405964303525, -0.7980789681018621, 0.3735765711084902, -0.8707524308128995, 0.30548249608932565, -0.6990681853106011, 0.4382865509675824, -0.9973969692525912, 0.5602845598552193, -1.4893326751963798, -0.007044033458385002, -0.06484125482364789, 0.14160886193487165, -0.24241047488061085, 0.14160886193487165, -0.24241047488061085, 0.1528369816077202, -0.2688059521539401, 0.21660624003369477, -0.3687378481287098, 0.05704213601223884, -0.10272328888077098, 0.12179605046912911, -0.22682846938669243, 0.05965654126371245, -0.11267943612286757, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -63333.33333333113, -20816.65999466039, -63333.333333332266, -20816.65999466152, -63333.33333333332, -20816.659994661335, -4750.000000000083, -1561.2494995993147, -4749.999999999877, -1561.2494995993716, -4750.0000000002865, -1561.2494995995303, -16466.66666666464, -5412.331598610512, -16466.66666666773, -5412.331598611147, -16466.666666666926, -5412.33159861298, -17416.66666666622, -5724.581498531829, -17416.666666668083, -5724.581498530997, -17416.666666667945, -5724.5814985310335, -11083.333333332897, -3642.9154990658376, -11083.333333333037, -3642.9154990651805, -11083.333333334114, -3642.9154990664397, -14883.333333333507, -4891.9150987456205, -14883.333333333638, -4891.915098745632, -14883.33333333336, -4891.915098744581, -28333.333335939387, -17559.42292349193, -28333.333325891297, -17559.42292612909, -28333.333338265, -17559.422914521434, -36000.000000000815, -17435.595774162168, -35999.99999999903, -17435.59577416274, -36000.0000000003, -17435.59577416212, -5999.999999999479, -2905.932629027591, -5999.9999999997335, -2905.932629026383, -6000.000000000693, -2905.932629027016, -5999.999999999479, -2905.932629027591, -5999.9999999997335, -2905.932629026383, -6000.000000000693, -2905.932629027016, -7499.999999999482, -3632.4157862838842, -7500.000000000354, -3632.4157862838765, -7500.000000000065, -3632.4157862839256, -7499.99999999908, -3632.415786284226, -7500.000000000483, -3632.4157862834372, -7500.000000000937, -3632.415786283963, -2399.999999999841, -1162.3730516108112, -2400.0000000000214, -1162.3730516108224, -2400.0000000001646, -1162.3730516106748, -4799.999999999371, -2324.7461032218184, -4799.999999999937, -2324.746103221286, -4800.000000000485, -2324.7461032219417, -2399.9999999996926, -1162.3730516108767, -2400.000000000042, -1162.373051610754, -2400.0000000002724, -1162.3730516109106, 1.0] 
        self.outputs_list = ['i_l_R01_R02_a_r', 'i_l_R01_R02_a_i', 'i_l_R01_R02_b_r', 'i_l_R01_R02_b_i', 'i_l_R01_R02_c_r', 'i_l_R01_R02_c_i', 'i_l_R01_R02_n_r', 'i_l_R01_R02_n_i', 'i_l_R02_R03_a_r', 'i_l_R02_R03_a_i', 'i_l_R02_R03_b_r', 'i_l_R02_R03_b_i', 'i_l_R02_R03_c_r', 'i_l_R02_R03_c_i', 'i_l_R02_R03_n_r', 'i_l_R02_R03_n_i', 'i_l_R03_R04_a_r', 'i_l_R03_R04_a_i', 'i_l_R03_R04_b_r', 'i_l_R03_R04_b_i', 'i_l_R03_R04_c_r', 'i_l_R03_R04_c_i', 'i_l_R03_R04_n_r', 'i_l_R03_R04_n_i', 'i_l_R04_R05_a_r', 'i_l_R04_R05_a_i', 'i_l_R04_R05_b_r', 'i_l_R04_R05_b_i', 'i_l_R04_R05_c_r', 'i_l_R04_R05_c_i', 'i_l_R04_R05_n_r', 'i_l_R04_R05_n_i', 'i_l_R05_R06_a_r', 'i_l_R05_R06_a_i', 'i_l_R05_R06_b_r', 'i_l_R05_R06_b_i', 'i_l_R05_R06_c_r', 'i_l_R05_R06_c_i', 'i_l_R05_R06_n_r', 'i_l_R05_R06_n_i', 'i_l_R06_R07_a_r', 'i_l_R06_R07_a_i', 'i_l_R06_R07_b_r', 'i_l_R06_R07_b_i', 'i_l_R06_R07_c_r', 'i_l_R06_R07_c_i', 'i_l_R06_R07_n_r', 'i_l_R06_R07_n_i', 'i_l_R07_R08_a_r', 'i_l_R07_R08_a_i', 'i_l_R07_R08_b_r', 'i_l_R07_R08_b_i', 'i_l_R07_R08_c_r', 'i_l_R07_R08_c_i', 'i_l_R07_R08_n_r', 'i_l_R07_R08_n_i', 'i_l_R08_R09_a_r', 'i_l_R08_R09_a_i', 'i_l_R08_R09_b_r', 'i_l_R08_R09_b_i', 'i_l_R08_R09_c_r', 'i_l_R08_R09_c_i', 'i_l_R08_R09_n_r', 'i_l_R08_R09_n_i', 'i_l_R09_R10_a_r', 'i_l_R09_R10_a_i', 'i_l_R09_R10_b_r', 'i_l_R09_R10_b_i', 'i_l_R09_R10_c_r', 'i_l_R09_R10_c_i', 'i_l_R09_R10_n_r', 'i_l_R09_R10_n_i', 'i_l_R03_R11_a_r', 'i_l_R03_R11_a_i', 'i_l_R03_R11_b_r', 'i_l_R03_R11_b_i', 'i_l_R03_R11_c_r', 'i_l_R03_R11_c_i', 'i_l_R03_R11_n_r', 'i_l_R03_R11_n_i', 'i_l_R04_R12_a_r', 'i_l_R04_R12_a_i', 'i_l_R04_R12_b_r', 'i_l_R04_R12_b_i', 'i_l_R04_R12_c_r', 'i_l_R04_R12_c_i', 'i_l_R04_R12_n_r', 'i_l_R04_R12_n_i', 'i_l_R12_R13_a_r', 'i_l_R12_R13_a_i', 'i_l_R12_R13_b_r', 'i_l_R12_R13_b_i', 'i_l_R12_R13_c_r', 'i_l_R12_R13_c_i', 'i_l_R12_R13_n_r', 'i_l_R12_R13_n_i', 'i_l_R13_R14_a_r', 'i_l_R13_R14_a_i', 'i_l_R13_R14_b_r', 'i_l_R13_R14_b_i', 'i_l_R13_R14_c_r', 'i_l_R13_R14_c_i', 'i_l_R13_R14_n_r', 'i_l_R13_R14_n_i', 'i_l_R14_R15_a_r', 'i_l_R14_R15_a_i', 'i_l_R14_R15_b_r', 'i_l_R14_R15_b_i', 'i_l_R14_R15_c_r', 'i_l_R14_R15_c_i', 'i_l_R14_R15_n_r', 'i_l_R14_R15_n_i', 'i_l_R06_R16_a_r', 'i_l_R06_R16_a_i', 'i_l_R06_R16_b_r', 'i_l_R06_R16_b_i', 'i_l_R06_R16_c_r', 'i_l_R06_R16_c_i', 'i_l_R06_R16_n_r', 'i_l_R06_R16_n_i', 'i_l_R09_R17_a_r', 'i_l_R09_R17_a_i', 'i_l_R09_R17_b_r', 'i_l_R09_R17_b_i', 'i_l_R09_R17_c_r', 'i_l_R09_R17_c_i', 'i_l_R09_R17_n_r', 'i_l_R09_R17_n_i', 'i_l_R10_R18_a_r', 'i_l_R10_R18_a_i', 'i_l_R10_R18_b_r', 'i_l_R10_R18_b_i', 'i_l_R10_R18_c_r', 'i_l_R10_R18_c_i', 'i_l_R10_R18_n_r', 'i_l_R10_R18_n_i', 'i_l_I01_I02_a_r', 'i_l_I01_I02_a_i', 'i_l_I01_I02_b_r', 'i_l_I01_I02_b_i', 'i_l_I01_I02_c_r', 'i_l_I01_I02_c_i', 'i_l_I01_I02_n_r', 'i_l_I01_I02_n_i', 'i_l_C01_C02_a_r', 'i_l_C01_C02_a_i', 'i_l_C01_C02_b_r', 'i_l_C01_C02_b_i', 'i_l_C01_C02_c_r', 'i_l_C01_C02_c_i', 'i_l_C01_C02_n_r', 'i_l_C01_C02_n_i', 'i_l_C02_C03_a_r', 'i_l_C02_C03_a_i', 'i_l_C02_C03_b_r', 'i_l_C02_C03_b_i', 'i_l_C02_C03_c_r', 'i_l_C02_C03_c_i', 'i_l_C02_C03_n_r', 'i_l_C02_C03_n_i', 'i_l_C03_C04_a_r', 'i_l_C03_C04_a_i', 'i_l_C03_C04_b_r', 'i_l_C03_C04_b_i', 'i_l_C03_C04_c_r', 'i_l_C03_C04_c_i', 'i_l_C03_C04_n_r', 'i_l_C03_C04_n_i', 'i_l_C04_C05_a_r', 'i_l_C04_C05_a_i', 'i_l_C04_C05_b_r', 'i_l_C04_C05_b_i', 'i_l_C04_C05_c_r', 'i_l_C04_C05_c_i', 'i_l_C04_C05_n_r', 'i_l_C04_C05_n_i', 'i_l_C05_C06_a_r', 'i_l_C05_C06_a_i', 'i_l_C05_C06_b_r', 'i_l_C05_C06_b_i', 'i_l_C05_C06_c_r', 'i_l_C05_C06_c_i', 'i_l_C05_C06_n_r', 'i_l_C05_C06_n_i', 'i_l_C06_C07_a_r', 'i_l_C06_C07_a_i', 'i_l_C06_C07_b_r', 'i_l_C06_C07_b_i', 'i_l_C06_C07_c_r', 'i_l_C06_C07_c_i', 'i_l_C06_C07_n_r', 'i_l_C06_C07_n_i', 'i_l_C07_C08_a_r', 'i_l_C07_C08_a_i', 'i_l_C07_C08_b_r', 'i_l_C07_C08_b_i', 'i_l_C07_C08_c_r', 'i_l_C07_C08_c_i', 'i_l_C07_C08_n_r', 'i_l_C07_C08_n_i', 'i_l_C08_C09_a_r', 'i_l_C08_C09_a_i', 'i_l_C08_C09_b_r', 'i_l_C08_C09_b_i', 'i_l_C08_C09_c_r', 'i_l_C08_C09_c_i', 'i_l_C08_C09_n_r', 'i_l_C08_C09_n_i', 'i_l_C03_C10_a_r', 'i_l_C03_C10_a_i', 'i_l_C03_C10_b_r', 'i_l_C03_C10_b_i', 'i_l_C03_C10_c_r', 'i_l_C03_C10_c_i', 'i_l_C03_C10_n_r', 'i_l_C03_C10_n_i', 'i_l_C10_C11_a_r', 'i_l_C10_C11_a_i', 'i_l_C10_C11_b_r', 'i_l_C10_C11_b_i', 'i_l_C10_C11_c_r', 'i_l_C10_C11_c_i', 'i_l_C10_C11_n_r', 'i_l_C10_C11_n_i', 'i_l_C11_C12_a_r', 'i_l_C11_C12_a_i', 'i_l_C11_C12_b_r', 'i_l_C11_C12_b_i', 'i_l_C11_C12_c_r', 'i_l_C11_C12_c_i', 'i_l_C11_C12_n_r', 'i_l_C11_C12_n_i', 'i_l_C11_C13_a_r', 'i_l_C11_C13_a_i', 'i_l_C11_C13_b_r', 'i_l_C11_C13_b_i', 'i_l_C11_C13_c_r', 'i_l_C11_C13_c_i', 'i_l_C11_C13_n_r', 'i_l_C11_C13_n_i', 'i_l_C10_C14_a_r', 'i_l_C10_C14_a_i', 'i_l_C10_C14_b_r', 'i_l_C10_C14_b_i', 'i_l_C10_C14_c_r', 'i_l_C10_C14_c_i', 'i_l_C10_C14_n_r', 'i_l_C10_C14_n_i', 'i_l_C05_C15_a_r', 'i_l_C05_C15_a_i', 'i_l_C05_C15_b_r', 'i_l_C05_C15_b_i', 'i_l_C05_C15_c_r', 'i_l_C05_C15_c_i', 'i_l_C05_C15_n_r', 'i_l_C05_C15_n_i', 'i_l_C15_C16_a_r', 'i_l_C15_C16_a_i', 'i_l_C15_C16_b_r', 'i_l_C15_C16_b_i', 'i_l_C15_C16_c_r', 'i_l_C15_C16_c_i', 'i_l_C15_C16_n_r', 'i_l_C15_C16_n_i', 'i_l_C15_C18_a_r', 'i_l_C15_C18_a_i', 'i_l_C15_C18_b_r', 'i_l_C15_C18_b_i', 'i_l_C15_C18_c_r', 'i_l_C15_C18_c_i', 'i_l_C15_C18_n_r', 'i_l_C15_C18_n_i', 'i_l_C16_C17_a_r', 'i_l_C16_C17_a_i', 'i_l_C16_C17_b_r', 'i_l_C16_C17_b_i', 'i_l_C16_C17_c_r', 'i_l_C16_C17_c_i', 'i_l_C16_C17_n_r', 'i_l_C16_C17_n_i', 'i_l_C08_C19_a_r', 'i_l_C08_C19_a_i', 'i_l_C08_C19_b_r', 'i_l_C08_C19_b_i', 'i_l_C08_C19_c_r', 'i_l_C08_C19_c_i', 'i_l_C08_C19_n_r', 'i_l_C08_C19_n_i', 'i_l_C09_C20_a_r', 'i_l_C09_C20_a_i', 'i_l_C09_C20_b_r', 'i_l_C09_C20_b_i', 'i_l_C09_C20_c_r', 'i_l_C09_C20_c_i', 'i_l_C09_C20_n_r', 'i_l_C09_C20_n_i', 'v_MV0_a_m', 'v_MV0_b_m', 'v_MV0_c_m', 'v_R01_a_m', 'v_R01_b_m', 'v_R01_c_m', 'v_R01_n_m', 'v_R11_a_m', 'v_R11_b_m', 'v_R11_c_m', 'v_R11_n_m', 'v_R15_a_m', 'v_R15_b_m', 'v_R15_c_m', 'v_R15_n_m', 'v_R16_a_m', 'v_R16_b_m', 'v_R16_c_m', 'v_R16_n_m', 'v_R17_a_m', 'v_R17_b_m', 'v_R17_c_m', 'v_R17_n_m', 'v_R18_a_m', 'v_R18_b_m', 'v_R18_c_m', 'v_R18_n_m', 'v_I02_a_m', 'v_I02_b_m', 'v_I02_c_m', 'v_I02_n_m', 'v_C01_a_m', 'v_C01_b_m', 'v_C01_c_m', 'v_C01_n_m', 'v_C12_a_m', 'v_C12_b_m', 'v_C12_c_m', 'v_C12_n_m', 'v_C13_a_m', 'v_C13_b_m', 'v_C13_c_m', 'v_C13_n_m', 'v_C14_a_m', 'v_C14_b_m', 'v_C14_c_m', 'v_C14_n_m', 'v_C17_a_m', 'v_C17_b_m', 'v_C17_c_m', 'v_C17_n_m', 'v_C18_a_m', 'v_C18_b_m', 'v_C18_c_m', 'v_C18_n_m', 'v_C19_a_m', 'v_C19_b_m', 'v_C19_c_m', 'v_C19_n_m', 'v_C20_a_m', 'v_C20_b_m', 'v_C20_c_m', 'v_C20_n_m', 'v_I01_a_m', 'v_I01_b_m', 'v_I01_c_m', 'v_I01_n_m', 'v_R02_a_m', 'v_R02_b_m', 'v_R02_c_m', 'v_R02_n_m', 'v_R03_a_m', 'v_R03_b_m', 'v_R03_c_m', 'v_R03_n_m', 'v_R04_a_m', 'v_R04_b_m', 'v_R04_c_m', 'v_R04_n_m', 'v_R05_a_m', 'v_R05_b_m', 'v_R05_c_m', 'v_R05_n_m', 'v_R06_a_m', 'v_R06_b_m', 'v_R06_c_m', 'v_R06_n_m', 'v_R07_a_m', 'v_R07_b_m', 'v_R07_c_m', 'v_R07_n_m', 'v_R08_a_m', 'v_R08_b_m', 'v_R08_c_m', 'v_R08_n_m', 'v_R09_a_m', 'v_R09_b_m', 'v_R09_c_m', 'v_R09_n_m', 'v_R10_a_m', 'v_R10_b_m', 'v_R10_c_m', 'v_R10_n_m', 'v_R12_a_m', 'v_R12_b_m', 'v_R12_c_m', 'v_R12_n_m', 'v_R13_a_m', 'v_R13_b_m', 'v_R13_c_m', 'v_R13_n_m', 'v_R14_a_m', 'v_R14_b_m', 'v_R14_c_m', 'v_R14_n_m', 'v_C02_a_m', 'v_C02_b_m', 'v_C02_c_m', 'v_C02_n_m', 'v_C03_a_m', 'v_C03_b_m', 'v_C03_c_m', 'v_C03_n_m', 'v_C04_a_m', 'v_C04_b_m', 'v_C04_c_m', 'v_C04_n_m', 'v_C05_a_m', 'v_C05_b_m', 'v_C05_c_m', 'v_C05_n_m', 'v_C06_a_m', 'v_C06_b_m', 'v_C06_c_m', 'v_C06_n_m', 'v_C07_a_m', 'v_C07_b_m', 'v_C07_c_m', 'v_C07_n_m', 'v_C08_a_m', 'v_C08_b_m', 'v_C08_c_m', 'v_C08_n_m', 'v_C09_a_m', 'v_C09_b_m', 'v_C09_c_m', 'v_C09_n_m', 'v_C10_a_m', 'v_C10_b_m', 'v_C10_c_m', 'v_C10_n_m', 'v_C11_a_m', 'v_C11_b_m', 'v_C11_c_m', 'v_C11_n_m', 'v_C15_a_m', 'v_C15_b_m', 'v_C15_c_m', 'v_C15_n_m', 'v_C16_a_m', 'v_C16_b_m', 'v_C16_c_m', 'v_C16_n_m'] 
        self.x_list = ['x_dummy'] 
        self.y_run_list = ['v_R01_a_r', 'v_R01_a_i', 'v_R01_b_r', 'v_R01_b_i', 'v_R01_c_r', 'v_R01_c_i', 'v_R01_n_r', 'v_R01_n_i', 'v_R11_a_r', 'v_R11_a_i', 'v_R11_b_r', 'v_R11_b_i', 'v_R11_c_r', 'v_R11_c_i', 'v_R11_n_r', 'v_R11_n_i', 'v_R15_a_r', 'v_R15_a_i', 'v_R15_b_r', 'v_R15_b_i', 'v_R15_c_r', 'v_R15_c_i', 'v_R15_n_r', 'v_R15_n_i', 'v_R16_a_r', 'v_R16_a_i', 'v_R16_b_r', 'v_R16_b_i', 'v_R16_c_r', 'v_R16_c_i', 'v_R16_n_r', 'v_R16_n_i', 'v_R17_a_r', 'v_R17_a_i', 'v_R17_b_r', 'v_R17_b_i', 'v_R17_c_r', 'v_R17_c_i', 'v_R17_n_r', 'v_R17_n_i', 'v_R18_a_r', 'v_R18_a_i', 'v_R18_b_r', 'v_R18_b_i', 'v_R18_c_r', 'v_R18_c_i', 'v_R18_n_r', 'v_R18_n_i', 'v_I02_a_r', 'v_I02_a_i', 'v_I02_b_r', 'v_I02_b_i', 'v_I02_c_r', 'v_I02_c_i', 'v_I02_n_r', 'v_I02_n_i', 'v_C01_a_r', 'v_C01_a_i', 'v_C01_b_r', 'v_C01_b_i', 'v_C01_c_r', 'v_C01_c_i', 'v_C01_n_r', 'v_C01_n_i', 'v_C12_a_r', 'v_C12_a_i', 'v_C12_b_r', 'v_C12_b_i', 'v_C12_c_r', 'v_C12_c_i', 'v_C12_n_r', 'v_C12_n_i', 'v_C13_a_r', 'v_C13_a_i', 'v_C13_b_r', 'v_C13_b_i', 'v_C13_c_r', 'v_C13_c_i', 'v_C13_n_r', 'v_C13_n_i', 'v_C14_a_r', 'v_C14_a_i', 'v_C14_b_r', 'v_C14_b_i', 'v_C14_c_r', 'v_C14_c_i', 'v_C14_n_r', 'v_C14_n_i', 'v_C17_a_r', 'v_C17_a_i', 'v_C17_b_r', 'v_C17_b_i', 'v_C17_c_r', 'v_C17_c_i', 'v_C17_n_r', 'v_C17_n_i', 'v_C18_a_r', 'v_C18_a_i', 'v_C18_b_r', 'v_C18_b_i', 'v_C18_c_r', 'v_C18_c_i', 'v_C18_n_r', 'v_C18_n_i', 'v_C19_a_r', 'v_C19_a_i', 'v_C19_b_r', 'v_C19_b_i', 'v_C19_c_r', 'v_C19_c_i', 'v_C19_n_r', 'v_C19_n_i', 'v_C20_a_r', 'v_C20_a_i', 'v_C20_b_r', 'v_C20_b_i', 'v_C20_c_r', 'v_C20_c_i', 'v_C20_n_r', 'v_C20_n_i', 'v_I01_a_r', 'v_I01_a_i', 'v_I01_b_r', 'v_I01_b_i', 'v_I01_c_r', 'v_I01_c_i', 'v_I01_n_r', 'v_I01_n_i', 'v_R02_a_r', 'v_R02_a_i', 'v_R02_b_r', 'v_R02_b_i', 'v_R02_c_r', 'v_R02_c_i', 'v_R02_n_r', 'v_R02_n_i', 'v_R03_a_r', 'v_R03_a_i', 'v_R03_b_r', 'v_R03_b_i', 'v_R03_c_r', 'v_R03_c_i', 'v_R03_n_r', 'v_R03_n_i', 'v_R04_a_r', 'v_R04_a_i', 'v_R04_b_r', 'v_R04_b_i', 'v_R04_c_r', 'v_R04_c_i', 'v_R04_n_r', 'v_R04_n_i', 'v_R05_a_r', 'v_R05_a_i', 'v_R05_b_r', 'v_R05_b_i', 'v_R05_c_r', 'v_R05_c_i', 'v_R05_n_r', 'v_R05_n_i', 'v_R06_a_r', 'v_R06_a_i', 'v_R06_b_r', 'v_R06_b_i', 'v_R06_c_r', 'v_R06_c_i', 'v_R06_n_r', 'v_R06_n_i', 'v_R07_a_r', 'v_R07_a_i', 'v_R07_b_r', 'v_R07_b_i', 'v_R07_c_r', 'v_R07_c_i', 'v_R07_n_r', 'v_R07_n_i', 'v_R08_a_r', 'v_R08_a_i', 'v_R08_b_r', 'v_R08_b_i', 'v_R08_c_r', 'v_R08_c_i', 'v_R08_n_r', 'v_R08_n_i', 'v_R09_a_r', 'v_R09_a_i', 'v_R09_b_r', 'v_R09_b_i', 'v_R09_c_r', 'v_R09_c_i', 'v_R09_n_r', 'v_R09_n_i', 'v_R10_a_r', 'v_R10_a_i', 'v_R10_b_r', 'v_R10_b_i', 'v_R10_c_r', 'v_R10_c_i', 'v_R10_n_r', 'v_R10_n_i', 'v_R12_a_r', 'v_R12_a_i', 'v_R12_b_r', 'v_R12_b_i', 'v_R12_c_r', 'v_R12_c_i', 'v_R12_n_r', 'v_R12_n_i', 'v_R13_a_r', 'v_R13_a_i', 'v_R13_b_r', 'v_R13_b_i', 'v_R13_c_r', 'v_R13_c_i', 'v_R13_n_r', 'v_R13_n_i', 'v_R14_a_r', 'v_R14_a_i', 'v_R14_b_r', 'v_R14_b_i', 'v_R14_c_r', 'v_R14_c_i', 'v_R14_n_r', 'v_R14_n_i', 'v_C02_a_r', 'v_C02_a_i', 'v_C02_b_r', 'v_C02_b_i', 'v_C02_c_r', 'v_C02_c_i', 'v_C02_n_r', 'v_C02_n_i', 'v_C03_a_r', 'v_C03_a_i', 'v_C03_b_r', 'v_C03_b_i', 'v_C03_c_r', 'v_C03_c_i', 'v_C03_n_r', 'v_C03_n_i', 'v_C04_a_r', 'v_C04_a_i', 'v_C04_b_r', 'v_C04_b_i', 'v_C04_c_r', 'v_C04_c_i', 'v_C04_n_r', 'v_C04_n_i', 'v_C05_a_r', 'v_C05_a_i', 'v_C05_b_r', 'v_C05_b_i', 'v_C05_c_r', 'v_C05_c_i', 'v_C05_n_r', 'v_C05_n_i', 'v_C06_a_r', 'v_C06_a_i', 'v_C06_b_r', 'v_C06_b_i', 'v_C06_c_r', 'v_C06_c_i', 'v_C06_n_r', 'v_C06_n_i', 'v_C07_a_r', 'v_C07_a_i', 'v_C07_b_r', 'v_C07_b_i', 'v_C07_c_r', 'v_C07_c_i', 'v_C07_n_r', 'v_C07_n_i', 'v_C08_a_r', 'v_C08_a_i', 'v_C08_b_r', 'v_C08_b_i', 'v_C08_c_r', 'v_C08_c_i', 'v_C08_n_r', 'v_C08_n_i', 'v_C09_a_r', 'v_C09_a_i', 'v_C09_b_r', 'v_C09_b_i', 'v_C09_c_r', 'v_C09_c_i', 'v_C09_n_r', 'v_C09_n_i', 'v_C10_a_r', 'v_C10_a_i', 'v_C10_b_r', 'v_C10_b_i', 'v_C10_c_r', 'v_C10_c_i', 'v_C10_n_r', 'v_C10_n_i', 'v_C11_a_r', 'v_C11_a_i', 'v_C11_b_r', 'v_C11_b_i', 'v_C11_c_r', 'v_C11_c_i', 'v_C11_n_r', 'v_C11_n_i', 'v_C15_a_r', 'v_C15_a_i', 'v_C15_b_r', 'v_C15_b_i', 'v_C15_c_r', 'v_C15_c_i', 'v_C15_n_r', 'v_C15_n_i', 'v_C16_a_r', 'v_C16_a_i', 'v_C16_b_r', 'v_C16_b_i', 'v_C16_c_r', 'v_C16_c_i', 'v_C16_n_r', 'v_C16_n_i', 'i_t_MV0_R01_a_r', 'i_t_MV0_R01_a_i', 'i_t_MV0_R01_b_r', 'i_t_MV0_R01_b_i', 'i_t_MV0_R01_c_r', 'i_t_MV0_R01_c_i', 'i_t_MV0_I01_a_r', 'i_t_MV0_I01_a_i', 'i_t_MV0_I01_b_r', 'i_t_MV0_I01_b_i', 'i_t_MV0_I01_c_r', 'i_t_MV0_I01_c_i', 'i_t_MV0_C01_a_r', 'i_t_MV0_C01_a_i', 'i_t_MV0_C01_b_r', 'i_t_MV0_C01_b_i', 'i_t_MV0_C01_c_r', 'i_t_MV0_C01_c_i', 'i_load_R01_a_r', 'i_load_R01_a_i', 'i_load_R01_b_r', 'i_load_R01_b_i', 'i_load_R01_c_r', 'i_load_R01_c_i', 'i_load_R01_n_r', 'i_load_R01_n_i', 'i_load_R11_a_r', 'i_load_R11_a_i', 'i_load_R11_b_r', 'i_load_R11_b_i', 'i_load_R11_c_r', 'i_load_R11_c_i', 'i_load_R11_n_r', 'i_load_R11_n_i', 'i_load_R15_a_r', 'i_load_R15_a_i', 'i_load_R15_b_r', 'i_load_R15_b_i', 'i_load_R15_c_r', 'i_load_R15_c_i', 'i_load_R15_n_r', 'i_load_R15_n_i', 'i_load_R16_a_r', 'i_load_R16_a_i', 'i_load_R16_b_r', 'i_load_R16_b_i', 'i_load_R16_c_r', 'i_load_R16_c_i', 'i_load_R16_n_r', 'i_load_R16_n_i', 'i_load_R17_a_r', 'i_load_R17_a_i', 'i_load_R17_b_r', 'i_load_R17_b_i', 'i_load_R17_c_r', 'i_load_R17_c_i', 'i_load_R17_n_r', 'i_load_R17_n_i', 'i_load_R18_a_r', 'i_load_R18_a_i', 'i_load_R18_b_r', 'i_load_R18_b_i', 'i_load_R18_c_r', 'i_load_R18_c_i', 'i_load_R18_n_r', 'i_load_R18_n_i', 'i_load_I02_a_r', 'i_load_I02_a_i', 'i_load_I02_b_r', 'i_load_I02_b_i', 'i_load_I02_c_r', 'i_load_I02_c_i', 'i_load_I02_n_r', 'i_load_I02_n_i', 'i_load_C01_a_r', 'i_load_C01_a_i', 'i_load_C01_b_r', 'i_load_C01_b_i', 'i_load_C01_c_r', 'i_load_C01_c_i', 'i_load_C01_n_r', 'i_load_C01_n_i', 'i_load_C12_a_r', 'i_load_C12_a_i', 'i_load_C12_b_r', 'i_load_C12_b_i', 'i_load_C12_c_r', 'i_load_C12_c_i', 'i_load_C12_n_r', 'i_load_C12_n_i', 'i_load_C13_a_r', 'i_load_C13_a_i', 'i_load_C13_b_r', 'i_load_C13_b_i', 'i_load_C13_c_r', 'i_load_C13_c_i', 'i_load_C13_n_r', 'i_load_C13_n_i', 'i_load_C14_a_r', 'i_load_C14_a_i', 'i_load_C14_b_r', 'i_load_C14_b_i', 'i_load_C14_c_r', 'i_load_C14_c_i', 'i_load_C14_n_r', 'i_load_C14_n_i', 'i_load_C17_a_r', 'i_load_C17_a_i', 'i_load_C17_b_r', 'i_load_C17_b_i', 'i_load_C17_c_r', 'i_load_C17_c_i', 'i_load_C17_n_r', 'i_load_C17_n_i', 'i_load_C18_a_r', 'i_load_C18_a_i', 'i_load_C18_b_r', 'i_load_C18_b_i', 'i_load_C18_c_r', 'i_load_C18_c_i', 'i_load_C18_n_r', 'i_load_C18_n_i', 'i_load_C19_a_r', 'i_load_C19_a_i', 'i_load_C19_b_r', 'i_load_C19_b_i', 'i_load_C19_c_r', 'i_load_C19_c_i', 'i_load_C19_n_r', 'i_load_C19_n_i', 'i_load_C20_a_r', 'i_load_C20_a_i', 'i_load_C20_b_r', 'i_load_C20_b_i', 'i_load_C20_c_r', 'i_load_C20_c_i', 'i_load_C20_n_r', 'i_load_C20_n_i'] 
        self.xy_list = self.x_list + self.y_run_list 
        self.y_ini_list = ['v_R01_a_r', 'v_R01_a_i', 'v_R01_b_r', 'v_R01_b_i', 'v_R01_c_r', 'v_R01_c_i', 'v_R01_n_r', 'v_R01_n_i', 'v_R11_a_r', 'v_R11_a_i', 'v_R11_b_r', 'v_R11_b_i', 'v_R11_c_r', 'v_R11_c_i', 'v_R11_n_r', 'v_R11_n_i', 'v_R15_a_r', 'v_R15_a_i', 'v_R15_b_r', 'v_R15_b_i', 'v_R15_c_r', 'v_R15_c_i', 'v_R15_n_r', 'v_R15_n_i', 'v_R16_a_r', 'v_R16_a_i', 'v_R16_b_r', 'v_R16_b_i', 'v_R16_c_r', 'v_R16_c_i', 'v_R16_n_r', 'v_R16_n_i', 'v_R17_a_r', 'v_R17_a_i', 'v_R17_b_r', 'v_R17_b_i', 'v_R17_c_r', 'v_R17_c_i', 'v_R17_n_r', 'v_R17_n_i', 'v_R18_a_r', 'v_R18_a_i', 'v_R18_b_r', 'v_R18_b_i', 'v_R18_c_r', 'v_R18_c_i', 'v_R18_n_r', 'v_R18_n_i', 'v_I02_a_r', 'v_I02_a_i', 'v_I02_b_r', 'v_I02_b_i', 'v_I02_c_r', 'v_I02_c_i', 'v_I02_n_r', 'v_I02_n_i', 'v_C01_a_r', 'v_C01_a_i', 'v_C01_b_r', 'v_C01_b_i', 'v_C01_c_r', 'v_C01_c_i', 'v_C01_n_r', 'v_C01_n_i', 'v_C12_a_r', 'v_C12_a_i', 'v_C12_b_r', 'v_C12_b_i', 'v_C12_c_r', 'v_C12_c_i', 'v_C12_n_r', 'v_C12_n_i', 'v_C13_a_r', 'v_C13_a_i', 'v_C13_b_r', 'v_C13_b_i', 'v_C13_c_r', 'v_C13_c_i', 'v_C13_n_r', 'v_C13_n_i', 'v_C14_a_r', 'v_C14_a_i', 'v_C14_b_r', 'v_C14_b_i', 'v_C14_c_r', 'v_C14_c_i', 'v_C14_n_r', 'v_C14_n_i', 'v_C17_a_r', 'v_C17_a_i', 'v_C17_b_r', 'v_C17_b_i', 'v_C17_c_r', 'v_C17_c_i', 'v_C17_n_r', 'v_C17_n_i', 'v_C18_a_r', 'v_C18_a_i', 'v_C18_b_r', 'v_C18_b_i', 'v_C18_c_r', 'v_C18_c_i', 'v_C18_n_r', 'v_C18_n_i', 'v_C19_a_r', 'v_C19_a_i', 'v_C19_b_r', 'v_C19_b_i', 'v_C19_c_r', 'v_C19_c_i', 'v_C19_n_r', 'v_C19_n_i', 'v_C20_a_r', 'v_C20_a_i', 'v_C20_b_r', 'v_C20_b_i', 'v_C20_c_r', 'v_C20_c_i', 'v_C20_n_r', 'v_C20_n_i', 'v_I01_a_r', 'v_I01_a_i', 'v_I01_b_r', 'v_I01_b_i', 'v_I01_c_r', 'v_I01_c_i', 'v_I01_n_r', 'v_I01_n_i', 'v_R02_a_r', 'v_R02_a_i', 'v_R02_b_r', 'v_R02_b_i', 'v_R02_c_r', 'v_R02_c_i', 'v_R02_n_r', 'v_R02_n_i', 'v_R03_a_r', 'v_R03_a_i', 'v_R03_b_r', 'v_R03_b_i', 'v_R03_c_r', 'v_R03_c_i', 'v_R03_n_r', 'v_R03_n_i', 'v_R04_a_r', 'v_R04_a_i', 'v_R04_b_r', 'v_R04_b_i', 'v_R04_c_r', 'v_R04_c_i', 'v_R04_n_r', 'v_R04_n_i', 'v_R05_a_r', 'v_R05_a_i', 'v_R05_b_r', 'v_R05_b_i', 'v_R05_c_r', 'v_R05_c_i', 'v_R05_n_r', 'v_R05_n_i', 'v_R06_a_r', 'v_R06_a_i', 'v_R06_b_r', 'v_R06_b_i', 'v_R06_c_r', 'v_R06_c_i', 'v_R06_n_r', 'v_R06_n_i', 'v_R07_a_r', 'v_R07_a_i', 'v_R07_b_r', 'v_R07_b_i', 'v_R07_c_r', 'v_R07_c_i', 'v_R07_n_r', 'v_R07_n_i', 'v_R08_a_r', 'v_R08_a_i', 'v_R08_b_r', 'v_R08_b_i', 'v_R08_c_r', 'v_R08_c_i', 'v_R08_n_r', 'v_R08_n_i', 'v_R09_a_r', 'v_R09_a_i', 'v_R09_b_r', 'v_R09_b_i', 'v_R09_c_r', 'v_R09_c_i', 'v_R09_n_r', 'v_R09_n_i', 'v_R10_a_r', 'v_R10_a_i', 'v_R10_b_r', 'v_R10_b_i', 'v_R10_c_r', 'v_R10_c_i', 'v_R10_n_r', 'v_R10_n_i', 'v_R12_a_r', 'v_R12_a_i', 'v_R12_b_r', 'v_R12_b_i', 'v_R12_c_r', 'v_R12_c_i', 'v_R12_n_r', 'v_R12_n_i', 'v_R13_a_r', 'v_R13_a_i', 'v_R13_b_r', 'v_R13_b_i', 'v_R13_c_r', 'v_R13_c_i', 'v_R13_n_r', 'v_R13_n_i', 'v_R14_a_r', 'v_R14_a_i', 'v_R14_b_r', 'v_R14_b_i', 'v_R14_c_r', 'v_R14_c_i', 'v_R14_n_r', 'v_R14_n_i', 'v_C02_a_r', 'v_C02_a_i', 'v_C02_b_r', 'v_C02_b_i', 'v_C02_c_r', 'v_C02_c_i', 'v_C02_n_r', 'v_C02_n_i', 'v_C03_a_r', 'v_C03_a_i', 'v_C03_b_r', 'v_C03_b_i', 'v_C03_c_r', 'v_C03_c_i', 'v_C03_n_r', 'v_C03_n_i', 'v_C04_a_r', 'v_C04_a_i', 'v_C04_b_r', 'v_C04_b_i', 'v_C04_c_r', 'v_C04_c_i', 'v_C04_n_r', 'v_C04_n_i', 'v_C05_a_r', 'v_C05_a_i', 'v_C05_b_r', 'v_C05_b_i', 'v_C05_c_r', 'v_C05_c_i', 'v_C05_n_r', 'v_C05_n_i', 'v_C06_a_r', 'v_C06_a_i', 'v_C06_b_r', 'v_C06_b_i', 'v_C06_c_r', 'v_C06_c_i', 'v_C06_n_r', 'v_C06_n_i', 'v_C07_a_r', 'v_C07_a_i', 'v_C07_b_r', 'v_C07_b_i', 'v_C07_c_r', 'v_C07_c_i', 'v_C07_n_r', 'v_C07_n_i', 'v_C08_a_r', 'v_C08_a_i', 'v_C08_b_r', 'v_C08_b_i', 'v_C08_c_r', 'v_C08_c_i', 'v_C08_n_r', 'v_C08_n_i', 'v_C09_a_r', 'v_C09_a_i', 'v_C09_b_r', 'v_C09_b_i', 'v_C09_c_r', 'v_C09_c_i', 'v_C09_n_r', 'v_C09_n_i', 'v_C10_a_r', 'v_C10_a_i', 'v_C10_b_r', 'v_C10_b_i', 'v_C10_c_r', 'v_C10_c_i', 'v_C10_n_r', 'v_C10_n_i', 'v_C11_a_r', 'v_C11_a_i', 'v_C11_b_r', 'v_C11_b_i', 'v_C11_c_r', 'v_C11_c_i', 'v_C11_n_r', 'v_C11_n_i', 'v_C15_a_r', 'v_C15_a_i', 'v_C15_b_r', 'v_C15_b_i', 'v_C15_c_r', 'v_C15_c_i', 'v_C15_n_r', 'v_C15_n_i', 'v_C16_a_r', 'v_C16_a_i', 'v_C16_b_r', 'v_C16_b_i', 'v_C16_c_r', 'v_C16_c_i', 'v_C16_n_r', 'v_C16_n_i', 'i_t_MV0_R01_a_r', 'i_t_MV0_R01_a_i', 'i_t_MV0_R01_b_r', 'i_t_MV0_R01_b_i', 'i_t_MV0_R01_c_r', 'i_t_MV0_R01_c_i', 'i_t_MV0_I01_a_r', 'i_t_MV0_I01_a_i', 'i_t_MV0_I01_b_r', 'i_t_MV0_I01_b_i', 'i_t_MV0_I01_c_r', 'i_t_MV0_I01_c_i', 'i_t_MV0_C01_a_r', 'i_t_MV0_C01_a_i', 'i_t_MV0_C01_b_r', 'i_t_MV0_C01_b_i', 'i_t_MV0_C01_c_r', 'i_t_MV0_C01_c_i', 'i_load_R01_a_r', 'i_load_R01_a_i', 'i_load_R01_b_r', 'i_load_R01_b_i', 'i_load_R01_c_r', 'i_load_R01_c_i', 'i_load_R01_n_r', 'i_load_R01_n_i', 'i_load_R11_a_r', 'i_load_R11_a_i', 'i_load_R11_b_r', 'i_load_R11_b_i', 'i_load_R11_c_r', 'i_load_R11_c_i', 'i_load_R11_n_r', 'i_load_R11_n_i', 'i_load_R15_a_r', 'i_load_R15_a_i', 'i_load_R15_b_r', 'i_load_R15_b_i', 'i_load_R15_c_r', 'i_load_R15_c_i', 'i_load_R15_n_r', 'i_load_R15_n_i', 'i_load_R16_a_r', 'i_load_R16_a_i', 'i_load_R16_b_r', 'i_load_R16_b_i', 'i_load_R16_c_r', 'i_load_R16_c_i', 'i_load_R16_n_r', 'i_load_R16_n_i', 'i_load_R17_a_r', 'i_load_R17_a_i', 'i_load_R17_b_r', 'i_load_R17_b_i', 'i_load_R17_c_r', 'i_load_R17_c_i', 'i_load_R17_n_r', 'i_load_R17_n_i', 'i_load_R18_a_r', 'i_load_R18_a_i', 'i_load_R18_b_r', 'i_load_R18_b_i', 'i_load_R18_c_r', 'i_load_R18_c_i', 'i_load_R18_n_r', 'i_load_R18_n_i', 'i_load_I02_a_r', 'i_load_I02_a_i', 'i_load_I02_b_r', 'i_load_I02_b_i', 'i_load_I02_c_r', 'i_load_I02_c_i', 'i_load_I02_n_r', 'i_load_I02_n_i', 'i_load_C01_a_r', 'i_load_C01_a_i', 'i_load_C01_b_r', 'i_load_C01_b_i', 'i_load_C01_c_r', 'i_load_C01_c_i', 'i_load_C01_n_r', 'i_load_C01_n_i', 'i_load_C12_a_r', 'i_load_C12_a_i', 'i_load_C12_b_r', 'i_load_C12_b_i', 'i_load_C12_c_r', 'i_load_C12_c_i', 'i_load_C12_n_r', 'i_load_C12_n_i', 'i_load_C13_a_r', 'i_load_C13_a_i', 'i_load_C13_b_r', 'i_load_C13_b_i', 'i_load_C13_c_r', 'i_load_C13_c_i', 'i_load_C13_n_r', 'i_load_C13_n_i', 'i_load_C14_a_r', 'i_load_C14_a_i', 'i_load_C14_b_r', 'i_load_C14_b_i', 'i_load_C14_c_r', 'i_load_C14_c_i', 'i_load_C14_n_r', 'i_load_C14_n_i', 'i_load_C17_a_r', 'i_load_C17_a_i', 'i_load_C17_b_r', 'i_load_C17_b_i', 'i_load_C17_c_r', 'i_load_C17_c_i', 'i_load_C17_n_r', 'i_load_C17_n_i', 'i_load_C18_a_r', 'i_load_C18_a_i', 'i_load_C18_b_r', 'i_load_C18_b_i', 'i_load_C18_c_r', 'i_load_C18_c_i', 'i_load_C18_n_r', 'i_load_C18_n_i', 'i_load_C19_a_r', 'i_load_C19_a_i', 'i_load_C19_b_r', 'i_load_C19_b_i', 'i_load_C19_c_r', 'i_load_C19_c_i', 'i_load_C19_n_r', 'i_load_C19_n_i', 'i_load_C20_a_r', 'i_load_C20_a_i', 'i_load_C20_b_r', 'i_load_C20_b_i', 'i_load_C20_c_r', 'i_load_C20_c_i', 'i_load_C20_n_r', 'i_load_C20_n_i'] 
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

    sp_jac_ini_ia = [0, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 339, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 340, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 341, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 342, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 343, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 344, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 347, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 348, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 349, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 350, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 351, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 352, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 17, 18, 19, 20, 21, 22, 23, 24, 217, 218, 219, 220, 221, 222, 223, 224, 355, 17, 18, 19, 20, 21, 22, 23, 24, 217, 218, 219, 220, 221, 222, 223, 224, 356, 17, 18, 19, 20, 21, 22, 23, 24, 217, 218, 219, 220, 221, 222, 223, 224, 357, 17, 18, 19, 20, 21, 22, 23, 24, 217, 218, 219, 220, 221, 222, 223, 224, 358, 17, 18, 19, 20, 21, 22, 23, 24, 217, 218, 219, 220, 221, 222, 223, 224, 359, 17, 18, 19, 20, 21, 22, 23, 24, 217, 218, 219, 220, 221, 222, 223, 224, 360, 17, 18, 19, 20, 21, 22, 23, 24, 217, 218, 219, 220, 221, 222, 223, 224, 17, 18, 19, 20, 21, 22, 23, 24, 217, 218, 219, 220, 221, 222, 223, 224, 25, 26, 27, 28, 29, 30, 31, 32, 161, 162, 163, 164, 165, 166, 167, 168, 363, 25, 26, 27, 28, 29, 30, 31, 32, 161, 162, 163, 164, 165, 166, 167, 168, 364, 25, 26, 27, 28, 29, 30, 31, 32, 161, 162, 163, 164, 165, 166, 167, 168, 365, 25, 26, 27, 28, 29, 30, 31, 32, 161, 162, 163, 164, 165, 166, 167, 168, 366, 25, 26, 27, 28, 29, 30, 31, 32, 161, 162, 163, 164, 165, 166, 167, 168, 367, 25, 26, 27, 28, 29, 30, 31, 32, 161, 162, 163, 164, 165, 166, 167, 168, 368, 25, 26, 27, 28, 29, 30, 31, 32, 161, 162, 163, 164, 165, 166, 167, 168, 25, 26, 27, 28, 29, 30, 31, 32, 161, 162, 163, 164, 165, 166, 167, 168, 33, 34, 35, 36, 37, 38, 39, 40, 185, 186, 187, 188, 189, 190, 191, 192, 371, 33, 34, 35, 36, 37, 38, 39, 40, 185, 186, 187, 188, 189, 190, 191, 192, 372, 33, 34, 35, 36, 37, 38, 39, 40, 185, 186, 187, 188, 189, 190, 191, 192, 373, 33, 34, 35, 36, 37, 38, 39, 40, 185, 186, 187, 188, 189, 190, 191, 192, 374, 33, 34, 35, 36, 37, 38, 39, 40, 185, 186, 187, 188, 189, 190, 191, 192, 375, 33, 34, 35, 36, 37, 38, 39, 40, 185, 186, 187, 188, 189, 190, 191, 192, 376, 33, 34, 35, 36, 37, 38, 39, 40, 185, 186, 187, 188, 189, 190, 191, 192, 33, 34, 35, 36, 37, 38, 39, 40, 185, 186, 187, 188, 189, 190, 191, 192, 41, 42, 43, 44, 45, 46, 47, 48, 193, 194, 195, 196, 197, 198, 199, 200, 379, 41, 42, 43, 44, 45, 46, 47, 48, 193, 194, 195, 196, 197, 198, 199, 200, 380, 41, 42, 43, 44, 45, 46, 47, 48, 193, 194, 195, 196, 197, 198, 199, 200, 381, 41, 42, 43, 44, 45, 46, 47, 48, 193, 194, 195, 196, 197, 198, 199, 200, 382, 41, 42, 43, 44, 45, 46, 47, 48, 193, 194, 195, 196, 197, 198, 199, 200, 383, 41, 42, 43, 44, 45, 46, 47, 48, 193, 194, 195, 196, 197, 198, 199, 200, 384, 41, 42, 43, 44, 45, 46, 47, 48, 193, 194, 195, 196, 197, 198, 199, 200, 41, 42, 43, 44, 45, 46, 47, 48, 193, 194, 195, 196, 197, 198, 199, 200, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 387, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 388, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 389, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 390, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 391, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 392, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 395, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 396, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 397, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 398, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 399, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 400, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 65, 66, 67, 68, 69, 70, 71, 72, 297, 298, 299, 300, 301, 302, 303, 304, 403, 65, 66, 67, 68, 69, 70, 71, 72, 297, 298, 299, 300, 301, 302, 303, 304, 404, 65, 66, 67, 68, 69, 70, 71, 72, 297, 298, 299, 300, 301, 302, 303, 304, 405, 65, 66, 67, 68, 69, 70, 71, 72, 297, 298, 299, 300, 301, 302, 303, 304, 406, 65, 66, 67, 68, 69, 70, 71, 72, 297, 298, 299, 300, 301, 302, 303, 304, 407, 65, 66, 67, 68, 69, 70, 71, 72, 297, 298, 299, 300, 301, 302, 303, 304, 408, 65, 66, 67, 68, 69, 70, 71, 72, 297, 298, 299, 300, 301, 302, 303, 304, 65, 66, 67, 68, 69, 70, 71, 72, 297, 298, 299, 300, 301, 302, 303, 304, 73, 74, 75, 76, 77, 78, 79, 80, 297, 298, 299, 300, 301, 302, 303, 304, 411, 73, 74, 75, 76, 77, 78, 79, 80, 297, 298, 299, 300, 301, 302, 303, 304, 412, 73, 74, 75, 76, 77, 78, 79, 80, 297, 298, 299, 300, 301, 302, 303, 304, 413, 73, 74, 75, 76, 77, 78, 79, 80, 297, 298, 299, 300, 301, 302, 303, 304, 414, 73, 74, 75, 76, 77, 78, 79, 80, 297, 298, 299, 300, 301, 302, 303, 304, 415, 73, 74, 75, 76, 77, 78, 79, 80, 297, 298, 299, 300, 301, 302, 303, 304, 416, 73, 74, 75, 76, 77, 78, 79, 80, 297, 298, 299, 300, 301, 302, 303, 304, 73, 74, 75, 76, 77, 78, 79, 80, 297, 298, 299, 300, 301, 302, 303, 304, 81, 82, 83, 84, 85, 86, 87, 88, 289, 290, 291, 292, 293, 294, 295, 296, 419, 81, 82, 83, 84, 85, 86, 87, 88, 289, 290, 291, 292, 293, 294, 295, 296, 420, 81, 82, 83, 84, 85, 86, 87, 88, 289, 290, 291, 292, 293, 294, 295, 296, 421, 81, 82, 83, 84, 85, 86, 87, 88, 289, 290, 291, 292, 293, 294, 295, 296, 422, 81, 82, 83, 84, 85, 86, 87, 88, 289, 290, 291, 292, 293, 294, 295, 296, 423, 81, 82, 83, 84, 85, 86, 87, 88, 289, 290, 291, 292, 293, 294, 295, 296, 424, 81, 82, 83, 84, 85, 86, 87, 88, 289, 290, 291, 292, 293, 294, 295, 296, 81, 82, 83, 84, 85, 86, 87, 88, 289, 290, 291, 292, 293, 294, 295, 296, 89, 90, 91, 92, 93, 94, 95, 96, 313, 314, 315, 316, 317, 318, 319, 320, 427, 89, 90, 91, 92, 93, 94, 95, 96, 313, 314, 315, 316, 317, 318, 319, 320, 428, 89, 90, 91, 92, 93, 94, 95, 96, 313, 314, 315, 316, 317, 318, 319, 320, 429, 89, 90, 91, 92, 93, 94, 95, 96, 313, 314, 315, 316, 317, 318, 319, 320, 430, 89, 90, 91, 92, 93, 94, 95, 96, 313, 314, 315, 316, 317, 318, 319, 320, 431, 89, 90, 91, 92, 93, 94, 95, 96, 313, 314, 315, 316, 317, 318, 319, 320, 432, 89, 90, 91, 92, 93, 94, 95, 96, 313, 314, 315, 316, 317, 318, 319, 320, 89, 90, 91, 92, 93, 94, 95, 96, 313, 314, 315, 316, 317, 318, 319, 320, 97, 98, 99, 100, 101, 102, 103, 104, 305, 306, 307, 308, 309, 310, 311, 312, 435, 97, 98, 99, 100, 101, 102, 103, 104, 305, 306, 307, 308, 309, 310, 311, 312, 436, 97, 98, 99, 100, 101, 102, 103, 104, 305, 306, 307, 308, 309, 310, 311, 312, 437, 97, 98, 99, 100, 101, 102, 103, 104, 305, 306, 307, 308, 309, 310, 311, 312, 438, 97, 98, 99, 100, 101, 102, 103, 104, 305, 306, 307, 308, 309, 310, 311, 312, 439, 97, 98, 99, 100, 101, 102, 103, 104, 305, 306, 307, 308, 309, 310, 311, 312, 440, 97, 98, 99, 100, 101, 102, 103, 104, 305, 306, 307, 308, 309, 310, 311, 312, 97, 98, 99, 100, 101, 102, 103, 104, 305, 306, 307, 308, 309, 310, 311, 312, 105, 106, 107, 108, 109, 110, 111, 112, 273, 274, 275, 276, 277, 278, 279, 280, 443, 105, 106, 107, 108, 109, 110, 111, 112, 273, 274, 275, 276, 277, 278, 279, 280, 444, 105, 106, 107, 108, 109, 110, 111, 112, 273, 274, 275, 276, 277, 278, 279, 280, 445, 105, 106, 107, 108, 109, 110, 111, 112, 273, 274, 275, 276, 277, 278, 279, 280, 446, 105, 106, 107, 108, 109, 110, 111, 112, 273, 274, 275, 276, 277, 278, 279, 280, 447, 105, 106, 107, 108, 109, 110, 111, 112, 273, 274, 275, 276, 277, 278, 279, 280, 448, 105, 106, 107, 108, 109, 110, 111, 112, 273, 274, 275, 276, 277, 278, 279, 280, 105, 106, 107, 108, 109, 110, 111, 112, 273, 274, 275, 276, 277, 278, 279, 280, 113, 114, 115, 116, 117, 118, 119, 120, 281, 282, 283, 284, 285, 286, 287, 288, 451, 113, 114, 115, 116, 117, 118, 119, 120, 281, 282, 283, 284, 285, 286, 287, 288, 452, 113, 114, 115, 116, 117, 118, 119, 120, 281, 282, 283, 284, 285, 286, 287, 288, 453, 113, 114, 115, 116, 117, 118, 119, 120, 281, 282, 283, 284, 285, 286, 287, 288, 454, 113, 114, 115, 116, 117, 118, 119, 120, 281, 282, 283, 284, 285, 286, 287, 288, 455, 113, 114, 115, 116, 117, 118, 119, 120, 281, 282, 283, 284, 285, 286, 287, 288, 456, 113, 114, 115, 116, 117, 118, 119, 120, 281, 282, 283, 284, 285, 286, 287, 288, 113, 114, 115, 116, 117, 118, 119, 120, 281, 282, 283, 284, 285, 286, 287, 288, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 9, 10, 11, 12, 13, 14, 15, 16, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 9, 10, 11, 12, 13, 14, 15, 16, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 9, 10, 11, 12, 13, 14, 15, 16, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 9, 10, 11, 12, 13, 14, 15, 16, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 9, 10, 11, 12, 13, 14, 15, 16, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 9, 10, 11, 12, 13, 14, 15, 16, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 9, 10, 11, 12, 13, 14, 15, 16, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 9, 10, 11, 12, 13, 14, 15, 16, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 201, 202, 203, 204, 205, 206, 207, 208, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 201, 202, 203, 204, 205, 206, 207, 208, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 201, 202, 203, 204, 205, 206, 207, 208, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 201, 202, 203, 204, 205, 206, 207, 208, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 201, 202, 203, 204, 205, 206, 207, 208, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 201, 202, 203, 204, 205, 206, 207, 208, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 201, 202, 203, 204, 205, 206, 207, 208, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 201, 202, 203, 204, 205, 206, 207, 208, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 25, 26, 27, 28, 29, 30, 31, 32, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 25, 26, 27, 28, 29, 30, 31, 32, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 25, 26, 27, 28, 29, 30, 31, 32, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 25, 26, 27, 28, 29, 30, 31, 32, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 25, 26, 27, 28, 29, 30, 31, 32, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 25, 26, 27, 28, 29, 30, 31, 32, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 25, 26, 27, 28, 29, 30, 31, 32, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 25, 26, 27, 28, 29, 30, 31, 32, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 33, 34, 35, 36, 37, 38, 39, 40, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 33, 34, 35, 36, 37, 38, 39, 40, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 33, 34, 35, 36, 37, 38, 39, 40, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 33, 34, 35, 36, 37, 38, 39, 40, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 33, 34, 35, 36, 37, 38, 39, 40, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 33, 34, 35, 36, 37, 38, 39, 40, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 33, 34, 35, 36, 37, 38, 39, 40, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 33, 34, 35, 36, 37, 38, 39, 40, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 41, 42, 43, 44, 45, 46, 47, 48, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 41, 42, 43, 44, 45, 46, 47, 48, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 41, 42, 43, 44, 45, 46, 47, 48, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 41, 42, 43, 44, 45, 46, 47, 48, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 41, 42, 43, 44, 45, 46, 47, 48, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 41, 42, 43, 44, 45, 46, 47, 48, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 41, 42, 43, 44, 45, 46, 47, 48, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 41, 42, 43, 44, 45, 46, 47, 48, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 145, 146, 147, 148, 149, 150, 151, 152, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 145, 146, 147, 148, 149, 150, 151, 152, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 145, 146, 147, 148, 149, 150, 151, 152, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 145, 146, 147, 148, 149, 150, 151, 152, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 145, 146, 147, 148, 149, 150, 151, 152, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 145, 146, 147, 148, 149, 150, 151, 152, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 145, 146, 147, 148, 149, 150, 151, 152, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 145, 146, 147, 148, 149, 150, 151, 152, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 17, 18, 19, 20, 21, 22, 23, 24, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 17, 18, 19, 20, 21, 22, 23, 24, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 17, 18, 19, 20, 21, 22, 23, 24, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 17, 18, 19, 20, 21, 22, 23, 24, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 17, 18, 19, 20, 21, 22, 23, 24, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 17, 18, 19, 20, 21, 22, 23, 24, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 17, 18, 19, 20, 21, 22, 23, 24, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 17, 18, 19, 20, 21, 22, 23, 24, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 289, 290, 291, 292, 293, 294, 295, 296, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 305, 306, 307, 308, 309, 310, 311, 312, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 305, 306, 307, 308, 309, 310, 311, 312, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 305, 306, 307, 308, 309, 310, 311, 312, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 305, 306, 307, 308, 309, 310, 311, 312, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 305, 306, 307, 308, 309, 310, 311, 312, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 305, 306, 307, 308, 309, 310, 311, 312, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 305, 306, 307, 308, 309, 310, 311, 312, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 305, 306, 307, 308, 309, 310, 311, 312, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 105, 106, 107, 108, 109, 110, 111, 112, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 105, 106, 107, 108, 109, 110, 111, 112, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 105, 106, 107, 108, 109, 110, 111, 112, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 105, 106, 107, 108, 109, 110, 111, 112, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 105, 106, 107, 108, 109, 110, 111, 112, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 105, 106, 107, 108, 109, 110, 111, 112, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 105, 106, 107, 108, 109, 110, 111, 112, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 105, 106, 107, 108, 109, 110, 111, 112, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 113, 114, 115, 116, 117, 118, 119, 120, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 113, 114, 115, 116, 117, 118, 119, 120, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 113, 114, 115, 116, 117, 118, 119, 120, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 113, 114, 115, 116, 117, 118, 119, 120, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 113, 114, 115, 116, 117, 118, 119, 120, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 113, 114, 115, 116, 117, 118, 119, 120, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 113, 114, 115, 116, 117, 118, 119, 120, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 113, 114, 115, 116, 117, 118, 119, 120, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 81, 82, 83, 84, 85, 86, 87, 88, 233, 234, 235, 236, 237, 238, 239, 240, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 81, 82, 83, 84, 85, 86, 87, 88, 233, 234, 235, 236, 237, 238, 239, 240, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 81, 82, 83, 84, 85, 86, 87, 88, 233, 234, 235, 236, 237, 238, 239, 240, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 81, 82, 83, 84, 85, 86, 87, 88, 233, 234, 235, 236, 237, 238, 239, 240, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 81, 82, 83, 84, 85, 86, 87, 88, 233, 234, 235, 236, 237, 238, 239, 240, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 81, 82, 83, 84, 85, 86, 87, 88, 233, 234, 235, 236, 237, 238, 239, 240, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 81, 82, 83, 84, 85, 86, 87, 88, 233, 234, 235, 236, 237, 238, 239, 240, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 81, 82, 83, 84, 85, 86, 87, 88, 233, 234, 235, 236, 237, 238, 239, 240, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 97, 98, 99, 100, 101, 102, 103, 104, 249, 250, 251, 252, 253, 254, 255, 256, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 97, 98, 99, 100, 101, 102, 103, 104, 249, 250, 251, 252, 253, 254, 255, 256, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 97, 98, 99, 100, 101, 102, 103, 104, 249, 250, 251, 252, 253, 254, 255, 256, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 97, 98, 99, 100, 101, 102, 103, 104, 249, 250, 251, 252, 253, 254, 255, 256, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 97, 98, 99, 100, 101, 102, 103, 104, 249, 250, 251, 252, 253, 254, 255, 256, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 97, 98, 99, 100, 101, 102, 103, 104, 249, 250, 251, 252, 253, 254, 255, 256, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 97, 98, 99, 100, 101, 102, 103, 104, 249, 250, 251, 252, 253, 254, 255, 256, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 97, 98, 99, 100, 101, 102, 103, 104, 249, 250, 251, 252, 253, 254, 255, 256, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 89, 90, 91, 92, 93, 94, 95, 96, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 89, 90, 91, 92, 93, 94, 95, 96, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 89, 90, 91, 92, 93, 94, 95, 96, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 89, 90, 91, 92, 93, 94, 95, 96, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 89, 90, 91, 92, 93, 94, 95, 96, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 89, 90, 91, 92, 93, 94, 95, 96, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 89, 90, 91, 92, 93, 94, 95, 96, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 89, 90, 91, 92, 93, 94, 95, 96, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 1, 2, 3, 4, 321, 1, 2, 3, 4, 322, 3, 4, 5, 6, 323, 3, 4, 5, 6, 324, 1, 2, 5, 6, 325, 1, 2, 5, 6, 326, 121, 122, 123, 124, 327, 121, 122, 123, 124, 328, 123, 124, 125, 126, 329, 123, 124, 125, 126, 330, 121, 122, 125, 126, 331, 121, 122, 125, 126, 332, 57, 58, 59, 60, 333, 57, 58, 59, 60, 334, 59, 60, 61, 62, 335, 59, 60, 61, 62, 336, 57, 58, 61, 62, 337, 57, 58, 61, 62, 338, 1, 2, 7, 8, 339, 340, 3, 4, 7, 8, 341, 342, 5, 6, 7, 8, 343, 344, 1, 2, 7, 8, 339, 340, 3, 4, 7, 8, 341, 342, 5, 6, 7, 8, 343, 344, 339, 341, 343, 345, 340, 342, 344, 346, 9, 10, 15, 16, 347, 348, 11, 12, 15, 16, 349, 350, 13, 14, 15, 16, 351, 352, 9, 10, 15, 16, 347, 348, 11, 12, 15, 16, 349, 350, 13, 14, 15, 16, 351, 352, 347, 349, 351, 353, 348, 350, 352, 354, 17, 18, 23, 24, 355, 356, 19, 20, 23, 24, 357, 358, 21, 22, 23, 24, 359, 360, 17, 18, 23, 24, 355, 356, 19, 20, 23, 24, 357, 358, 21, 22, 23, 24, 359, 360, 355, 357, 359, 361, 356, 358, 360, 362, 25, 26, 31, 32, 363, 364, 27, 28, 31, 32, 365, 366, 29, 30, 31, 32, 367, 368, 25, 26, 31, 32, 363, 364, 27, 28, 31, 32, 365, 366, 29, 30, 31, 32, 367, 368, 363, 365, 367, 369, 364, 366, 368, 370, 33, 34, 39, 40, 371, 372, 35, 36, 39, 40, 373, 374, 37, 38, 39, 40, 375, 376, 33, 34, 39, 40, 371, 372, 35, 36, 39, 40, 373, 374, 37, 38, 39, 40, 375, 376, 371, 373, 375, 377, 372, 374, 376, 378, 41, 42, 47, 48, 379, 380, 43, 44, 47, 48, 381, 382, 45, 46, 47, 48, 383, 384, 41, 42, 47, 48, 379, 380, 43, 44, 47, 48, 381, 382, 45, 46, 47, 48, 383, 384, 379, 381, 383, 385, 380, 382, 384, 386, 49, 50, 55, 56, 387, 388, 51, 52, 55, 56, 389, 390, 53, 54, 55, 56, 391, 392, 49, 50, 55, 56, 387, 388, 51, 52, 55, 56, 389, 390, 53, 54, 55, 56, 391, 392, 387, 389, 391, 393, 388, 390, 392, 394, 57, 58, 63, 64, 395, 396, 59, 60, 63, 64, 397, 398, 61, 62, 63, 64, 399, 400, 57, 58, 63, 64, 395, 396, 59, 60, 63, 64, 397, 398, 61, 62, 63, 64, 399, 400, 395, 397, 399, 401, 396, 398, 400, 402, 65, 66, 71, 72, 403, 404, 67, 68, 71, 72, 405, 406, 69, 70, 71, 72, 407, 408, 65, 66, 71, 72, 403, 404, 67, 68, 71, 72, 405, 406, 69, 70, 71, 72, 407, 408, 403, 405, 407, 409, 404, 406, 408, 410, 73, 74, 79, 80, 411, 412, 75, 76, 79, 80, 413, 414, 77, 78, 79, 80, 415, 416, 73, 74, 79, 80, 411, 412, 75, 76, 79, 80, 413, 414, 77, 78, 79, 80, 415, 416, 411, 413, 415, 417, 412, 414, 416, 418, 81, 82, 87, 88, 419, 420, 83, 84, 87, 88, 421, 422, 85, 86, 87, 88, 423, 424, 81, 82, 87, 88, 419, 420, 83, 84, 87, 88, 421, 422, 85, 86, 87, 88, 423, 424, 419, 421, 423, 425, 420, 422, 424, 426, 89, 90, 95, 96, 427, 428, 91, 92, 95, 96, 429, 430, 93, 94, 95, 96, 431, 432, 89, 90, 95, 96, 427, 428, 91, 92, 95, 96, 429, 430, 93, 94, 95, 96, 431, 432, 427, 429, 431, 433, 428, 430, 432, 434, 97, 98, 103, 104, 435, 436, 99, 100, 103, 104, 437, 438, 101, 102, 103, 104, 439, 440, 97, 98, 103, 104, 435, 436, 99, 100, 103, 104, 437, 438, 101, 102, 103, 104, 439, 440, 435, 437, 439, 441, 436, 438, 440, 442, 105, 106, 111, 112, 443, 444, 107, 108, 111, 112, 445, 446, 109, 110, 111, 112, 447, 448, 105, 106, 111, 112, 443, 444, 107, 108, 111, 112, 445, 446, 109, 110, 111, 112, 447, 448, 443, 445, 447, 449, 444, 446, 448, 450, 113, 114, 119, 120, 451, 452, 115, 116, 119, 120, 453, 454, 117, 118, 119, 120, 455, 456, 113, 114, 119, 120, 451, 452, 115, 116, 119, 120, 453, 454, 117, 118, 119, 120, 455, 456, 451, 453, 455, 457, 452, 454, 456, 458]
    sp_jac_ini_ja = [0, 1, 18, 35, 52, 69, 86, 103, 119, 135, 152, 169, 186, 203, 220, 237, 253, 269, 286, 303, 320, 337, 354, 371, 387, 403, 420, 437, 454, 471, 488, 505, 521, 537, 554, 571, 588, 605, 622, 639, 655, 671, 688, 705, 722, 739, 756, 773, 789, 805, 822, 839, 856, 873, 890, 907, 923, 939, 956, 973, 990, 1007, 1024, 1041, 1057, 1073, 1090, 1107, 1124, 1141, 1158, 1175, 1191, 1207, 1224, 1241, 1258, 1275, 1292, 1309, 1325, 1341, 1358, 1375, 1392, 1409, 1426, 1443, 1459, 1475, 1492, 1509, 1526, 1543, 1560, 1577, 1593, 1609, 1626, 1643, 1660, 1677, 1694, 1711, 1727, 1743, 1760, 1777, 1794, 1811, 1828, 1845, 1861, 1877, 1894, 1911, 1928, 1945, 1962, 1979, 1995, 2011, 2027, 2043, 2059, 2075, 2091, 2107, 2123, 2139, 2163, 2187, 2211, 2235, 2259, 2283, 2307, 2331, 2363, 2395, 2427, 2459, 2491, 2523, 2555, 2587, 2619, 2651, 2683, 2715, 2747, 2779, 2811, 2843, 2867, 2891, 2915, 2939, 2963, 2987, 3011, 3035, 3067, 3099, 3131, 3163, 3195, 3227, 3259, 3291, 3315, 3339, 3363, 3387, 3411, 3435, 3459, 3483, 3507, 3531, 3555, 3579, 3603, 3627, 3651, 3675, 3707, 3739, 3771, 3803, 3835, 3867, 3899, 3931, 3955, 3979, 4003, 4027, 4051, 4075, 4099, 4123, 4147, 4171, 4195, 4219, 4243, 4267, 4291, 4315, 4339, 4363, 4387, 4411, 4435, 4459, 4483, 4507, 4531, 4555, 4579, 4603, 4627, 4651, 4675, 4699, 4723, 4747, 4771, 4795, 4819, 4843, 4867, 4891, 4923, 4955, 4987, 5019, 5051, 5083, 5115, 5147, 5171, 5195, 5219, 5243, 5267, 5291, 5315, 5339, 5371, 5403, 5435, 5467, 5499, 5531, 5563, 5595, 5619, 5643, 5667, 5691, 5715, 5739, 5763, 5787, 5811, 5835, 5859, 5883, 5907, 5931, 5955, 5979, 6011, 6043, 6075, 6107, 6139, 6171, 6203, 6235, 6259, 6283, 6307, 6331, 6355, 6379, 6403, 6427, 6459, 6491, 6523, 6555, 6587, 6619, 6651, 6683, 6715, 6747, 6779, 6811, 6843, 6875, 6907, 6939, 6971, 7003, 7035, 7067, 7099, 7131, 7163, 7195, 7219, 7243, 7267, 7291, 7315, 7339, 7363, 7387, 7392, 7397, 7402, 7407, 7412, 7417, 7422, 7427, 7432, 7437, 7442, 7447, 7452, 7457, 7462, 7467, 7472, 7477, 7483, 7489, 7495, 7501, 7507, 7513, 7517, 7521, 7527, 7533, 7539, 7545, 7551, 7557, 7561, 7565, 7571, 7577, 7583, 7589, 7595, 7601, 7605, 7609, 7615, 7621, 7627, 7633, 7639, 7645, 7649, 7653, 7659, 7665, 7671, 7677, 7683, 7689, 7693, 7697, 7703, 7709, 7715, 7721, 7727, 7733, 7737, 7741, 7747, 7753, 7759, 7765, 7771, 7777, 7781, 7785, 7791, 7797, 7803, 7809, 7815, 7821, 7825, 7829, 7835, 7841, 7847, 7853, 7859, 7865, 7869, 7873, 7879, 7885, 7891, 7897, 7903, 7909, 7913, 7917, 7923, 7929, 7935, 7941, 7947, 7953, 7957, 7961, 7967, 7973, 7979, 7985, 7991, 7997, 8001, 8005, 8011, 8017, 8023, 8029, 8035, 8041, 8045, 8049, 8055, 8061, 8067, 8073, 8079, 8085, 8089, 8093, 8099, 8105, 8111, 8117, 8123, 8129, 8133, 8137]
    sp_jac_ini_nia = 459
    sp_jac_ini_nja = 459
    return sp_jac_ini_ia, sp_jac_ini_ja, sp_jac_ini_nia, sp_jac_ini_nja 

def sp_jac_trap_vectors():

    sp_jac_trap_ia = [0, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 339, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 340, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 341, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 342, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 343, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 344, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 347, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 348, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 349, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 350, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 351, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 352, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 9, 10, 11, 12, 13, 14, 15, 16, 137, 138, 139, 140, 141, 142, 143, 144, 17, 18, 19, 20, 21, 22, 23, 24, 217, 218, 219, 220, 221, 222, 223, 224, 355, 17, 18, 19, 20, 21, 22, 23, 24, 217, 218, 219, 220, 221, 222, 223, 224, 356, 17, 18, 19, 20, 21, 22, 23, 24, 217, 218, 219, 220, 221, 222, 223, 224, 357, 17, 18, 19, 20, 21, 22, 23, 24, 217, 218, 219, 220, 221, 222, 223, 224, 358, 17, 18, 19, 20, 21, 22, 23, 24, 217, 218, 219, 220, 221, 222, 223, 224, 359, 17, 18, 19, 20, 21, 22, 23, 24, 217, 218, 219, 220, 221, 222, 223, 224, 360, 17, 18, 19, 20, 21, 22, 23, 24, 217, 218, 219, 220, 221, 222, 223, 224, 17, 18, 19, 20, 21, 22, 23, 24, 217, 218, 219, 220, 221, 222, 223, 224, 25, 26, 27, 28, 29, 30, 31, 32, 161, 162, 163, 164, 165, 166, 167, 168, 363, 25, 26, 27, 28, 29, 30, 31, 32, 161, 162, 163, 164, 165, 166, 167, 168, 364, 25, 26, 27, 28, 29, 30, 31, 32, 161, 162, 163, 164, 165, 166, 167, 168, 365, 25, 26, 27, 28, 29, 30, 31, 32, 161, 162, 163, 164, 165, 166, 167, 168, 366, 25, 26, 27, 28, 29, 30, 31, 32, 161, 162, 163, 164, 165, 166, 167, 168, 367, 25, 26, 27, 28, 29, 30, 31, 32, 161, 162, 163, 164, 165, 166, 167, 168, 368, 25, 26, 27, 28, 29, 30, 31, 32, 161, 162, 163, 164, 165, 166, 167, 168, 25, 26, 27, 28, 29, 30, 31, 32, 161, 162, 163, 164, 165, 166, 167, 168, 33, 34, 35, 36, 37, 38, 39, 40, 185, 186, 187, 188, 189, 190, 191, 192, 371, 33, 34, 35, 36, 37, 38, 39, 40, 185, 186, 187, 188, 189, 190, 191, 192, 372, 33, 34, 35, 36, 37, 38, 39, 40, 185, 186, 187, 188, 189, 190, 191, 192, 373, 33, 34, 35, 36, 37, 38, 39, 40, 185, 186, 187, 188, 189, 190, 191, 192, 374, 33, 34, 35, 36, 37, 38, 39, 40, 185, 186, 187, 188, 189, 190, 191, 192, 375, 33, 34, 35, 36, 37, 38, 39, 40, 185, 186, 187, 188, 189, 190, 191, 192, 376, 33, 34, 35, 36, 37, 38, 39, 40, 185, 186, 187, 188, 189, 190, 191, 192, 33, 34, 35, 36, 37, 38, 39, 40, 185, 186, 187, 188, 189, 190, 191, 192, 41, 42, 43, 44, 45, 46, 47, 48, 193, 194, 195, 196, 197, 198, 199, 200, 379, 41, 42, 43, 44, 45, 46, 47, 48, 193, 194, 195, 196, 197, 198, 199, 200, 380, 41, 42, 43, 44, 45, 46, 47, 48, 193, 194, 195, 196, 197, 198, 199, 200, 381, 41, 42, 43, 44, 45, 46, 47, 48, 193, 194, 195, 196, 197, 198, 199, 200, 382, 41, 42, 43, 44, 45, 46, 47, 48, 193, 194, 195, 196, 197, 198, 199, 200, 383, 41, 42, 43, 44, 45, 46, 47, 48, 193, 194, 195, 196, 197, 198, 199, 200, 384, 41, 42, 43, 44, 45, 46, 47, 48, 193, 194, 195, 196, 197, 198, 199, 200, 41, 42, 43, 44, 45, 46, 47, 48, 193, 194, 195, 196, 197, 198, 199, 200, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 387, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 388, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 389, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 390, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 391, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 392, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 395, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 396, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 397, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 398, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 399, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 400, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 65, 66, 67, 68, 69, 70, 71, 72, 297, 298, 299, 300, 301, 302, 303, 304, 403, 65, 66, 67, 68, 69, 70, 71, 72, 297, 298, 299, 300, 301, 302, 303, 304, 404, 65, 66, 67, 68, 69, 70, 71, 72, 297, 298, 299, 300, 301, 302, 303, 304, 405, 65, 66, 67, 68, 69, 70, 71, 72, 297, 298, 299, 300, 301, 302, 303, 304, 406, 65, 66, 67, 68, 69, 70, 71, 72, 297, 298, 299, 300, 301, 302, 303, 304, 407, 65, 66, 67, 68, 69, 70, 71, 72, 297, 298, 299, 300, 301, 302, 303, 304, 408, 65, 66, 67, 68, 69, 70, 71, 72, 297, 298, 299, 300, 301, 302, 303, 304, 65, 66, 67, 68, 69, 70, 71, 72, 297, 298, 299, 300, 301, 302, 303, 304, 73, 74, 75, 76, 77, 78, 79, 80, 297, 298, 299, 300, 301, 302, 303, 304, 411, 73, 74, 75, 76, 77, 78, 79, 80, 297, 298, 299, 300, 301, 302, 303, 304, 412, 73, 74, 75, 76, 77, 78, 79, 80, 297, 298, 299, 300, 301, 302, 303, 304, 413, 73, 74, 75, 76, 77, 78, 79, 80, 297, 298, 299, 300, 301, 302, 303, 304, 414, 73, 74, 75, 76, 77, 78, 79, 80, 297, 298, 299, 300, 301, 302, 303, 304, 415, 73, 74, 75, 76, 77, 78, 79, 80, 297, 298, 299, 300, 301, 302, 303, 304, 416, 73, 74, 75, 76, 77, 78, 79, 80, 297, 298, 299, 300, 301, 302, 303, 304, 73, 74, 75, 76, 77, 78, 79, 80, 297, 298, 299, 300, 301, 302, 303, 304, 81, 82, 83, 84, 85, 86, 87, 88, 289, 290, 291, 292, 293, 294, 295, 296, 419, 81, 82, 83, 84, 85, 86, 87, 88, 289, 290, 291, 292, 293, 294, 295, 296, 420, 81, 82, 83, 84, 85, 86, 87, 88, 289, 290, 291, 292, 293, 294, 295, 296, 421, 81, 82, 83, 84, 85, 86, 87, 88, 289, 290, 291, 292, 293, 294, 295, 296, 422, 81, 82, 83, 84, 85, 86, 87, 88, 289, 290, 291, 292, 293, 294, 295, 296, 423, 81, 82, 83, 84, 85, 86, 87, 88, 289, 290, 291, 292, 293, 294, 295, 296, 424, 81, 82, 83, 84, 85, 86, 87, 88, 289, 290, 291, 292, 293, 294, 295, 296, 81, 82, 83, 84, 85, 86, 87, 88, 289, 290, 291, 292, 293, 294, 295, 296, 89, 90, 91, 92, 93, 94, 95, 96, 313, 314, 315, 316, 317, 318, 319, 320, 427, 89, 90, 91, 92, 93, 94, 95, 96, 313, 314, 315, 316, 317, 318, 319, 320, 428, 89, 90, 91, 92, 93, 94, 95, 96, 313, 314, 315, 316, 317, 318, 319, 320, 429, 89, 90, 91, 92, 93, 94, 95, 96, 313, 314, 315, 316, 317, 318, 319, 320, 430, 89, 90, 91, 92, 93, 94, 95, 96, 313, 314, 315, 316, 317, 318, 319, 320, 431, 89, 90, 91, 92, 93, 94, 95, 96, 313, 314, 315, 316, 317, 318, 319, 320, 432, 89, 90, 91, 92, 93, 94, 95, 96, 313, 314, 315, 316, 317, 318, 319, 320, 89, 90, 91, 92, 93, 94, 95, 96, 313, 314, 315, 316, 317, 318, 319, 320, 97, 98, 99, 100, 101, 102, 103, 104, 305, 306, 307, 308, 309, 310, 311, 312, 435, 97, 98, 99, 100, 101, 102, 103, 104, 305, 306, 307, 308, 309, 310, 311, 312, 436, 97, 98, 99, 100, 101, 102, 103, 104, 305, 306, 307, 308, 309, 310, 311, 312, 437, 97, 98, 99, 100, 101, 102, 103, 104, 305, 306, 307, 308, 309, 310, 311, 312, 438, 97, 98, 99, 100, 101, 102, 103, 104, 305, 306, 307, 308, 309, 310, 311, 312, 439, 97, 98, 99, 100, 101, 102, 103, 104, 305, 306, 307, 308, 309, 310, 311, 312, 440, 97, 98, 99, 100, 101, 102, 103, 104, 305, 306, 307, 308, 309, 310, 311, 312, 97, 98, 99, 100, 101, 102, 103, 104, 305, 306, 307, 308, 309, 310, 311, 312, 105, 106, 107, 108, 109, 110, 111, 112, 273, 274, 275, 276, 277, 278, 279, 280, 443, 105, 106, 107, 108, 109, 110, 111, 112, 273, 274, 275, 276, 277, 278, 279, 280, 444, 105, 106, 107, 108, 109, 110, 111, 112, 273, 274, 275, 276, 277, 278, 279, 280, 445, 105, 106, 107, 108, 109, 110, 111, 112, 273, 274, 275, 276, 277, 278, 279, 280, 446, 105, 106, 107, 108, 109, 110, 111, 112, 273, 274, 275, 276, 277, 278, 279, 280, 447, 105, 106, 107, 108, 109, 110, 111, 112, 273, 274, 275, 276, 277, 278, 279, 280, 448, 105, 106, 107, 108, 109, 110, 111, 112, 273, 274, 275, 276, 277, 278, 279, 280, 105, 106, 107, 108, 109, 110, 111, 112, 273, 274, 275, 276, 277, 278, 279, 280, 113, 114, 115, 116, 117, 118, 119, 120, 281, 282, 283, 284, 285, 286, 287, 288, 451, 113, 114, 115, 116, 117, 118, 119, 120, 281, 282, 283, 284, 285, 286, 287, 288, 452, 113, 114, 115, 116, 117, 118, 119, 120, 281, 282, 283, 284, 285, 286, 287, 288, 453, 113, 114, 115, 116, 117, 118, 119, 120, 281, 282, 283, 284, 285, 286, 287, 288, 454, 113, 114, 115, 116, 117, 118, 119, 120, 281, 282, 283, 284, 285, 286, 287, 288, 455, 113, 114, 115, 116, 117, 118, 119, 120, 281, 282, 283, 284, 285, 286, 287, 288, 456, 113, 114, 115, 116, 117, 118, 119, 120, 281, 282, 283, 284, 285, 286, 287, 288, 113, 114, 115, 116, 117, 118, 119, 120, 281, 282, 283, 284, 285, 286, 287, 288, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 49, 50, 51, 52, 53, 54, 55, 56, 121, 122, 123, 124, 125, 126, 127, 128, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 1, 2, 3, 4, 5, 6, 7, 8, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 9, 10, 11, 12, 13, 14, 15, 16, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 9, 10, 11, 12, 13, 14, 15, 16, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 9, 10, 11, 12, 13, 14, 15, 16, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 9, 10, 11, 12, 13, 14, 15, 16, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 9, 10, 11, 12, 13, 14, 15, 16, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 9, 10, 11, 12, 13, 14, 15, 16, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 9, 10, 11, 12, 13, 14, 15, 16, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 9, 10, 11, 12, 13, 14, 15, 16, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 201, 202, 203, 204, 205, 206, 207, 208, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 201, 202, 203, 204, 205, 206, 207, 208, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 201, 202, 203, 204, 205, 206, 207, 208, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 201, 202, 203, 204, 205, 206, 207, 208, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 201, 202, 203, 204, 205, 206, 207, 208, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 201, 202, 203, 204, 205, 206, 207, 208, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 201, 202, 203, 204, 205, 206, 207, 208, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 201, 202, 203, 204, 205, 206, 207, 208, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 25, 26, 27, 28, 29, 30, 31, 32, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 25, 26, 27, 28, 29, 30, 31, 32, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 25, 26, 27, 28, 29, 30, 31, 32, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 25, 26, 27, 28, 29, 30, 31, 32, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 25, 26, 27, 28, 29, 30, 31, 32, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 25, 26, 27, 28, 29, 30, 31, 32, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 25, 26, 27, 28, 29, 30, 31, 32, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 25, 26, 27, 28, 29, 30, 31, 32, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 33, 34, 35, 36, 37, 38, 39, 40, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 33, 34, 35, 36, 37, 38, 39, 40, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 33, 34, 35, 36, 37, 38, 39, 40, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 33, 34, 35, 36, 37, 38, 39, 40, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 33, 34, 35, 36, 37, 38, 39, 40, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 33, 34, 35, 36, 37, 38, 39, 40, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 33, 34, 35, 36, 37, 38, 39, 40, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 33, 34, 35, 36, 37, 38, 39, 40, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 41, 42, 43, 44, 45, 46, 47, 48, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 41, 42, 43, 44, 45, 46, 47, 48, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 41, 42, 43, 44, 45, 46, 47, 48, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 41, 42, 43, 44, 45, 46, 47, 48, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 41, 42, 43, 44, 45, 46, 47, 48, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 41, 42, 43, 44, 45, 46, 47, 48, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 41, 42, 43, 44, 45, 46, 47, 48, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 41, 42, 43, 44, 45, 46, 47, 48, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 145, 146, 147, 148, 149, 150, 151, 152, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 145, 146, 147, 148, 149, 150, 151, 152, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 145, 146, 147, 148, 149, 150, 151, 152, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 145, 146, 147, 148, 149, 150, 151, 152, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 145, 146, 147, 148, 149, 150, 151, 152, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 145, 146, 147, 148, 149, 150, 151, 152, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 145, 146, 147, 148, 149, 150, 151, 152, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 145, 146, 147, 148, 149, 150, 151, 152, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 17, 18, 19, 20, 21, 22, 23, 24, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 17, 18, 19, 20, 21, 22, 23, 24, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 17, 18, 19, 20, 21, 22, 23, 24, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 17, 18, 19, 20, 21, 22, 23, 24, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 17, 18, 19, 20, 21, 22, 23, 24, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 17, 18, 19, 20, 21, 22, 23, 24, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 17, 18, 19, 20, 21, 22, 23, 24, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 17, 18, 19, 20, 21, 22, 23, 24, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 57, 58, 59, 60, 61, 62, 63, 64, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 289, 290, 291, 292, 293, 294, 295, 296, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 289, 290, 291, 292, 293, 294, 295, 296, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 305, 306, 307, 308, 309, 310, 311, 312, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 305, 306, 307, 308, 309, 310, 311, 312, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 305, 306, 307, 308, 309, 310, 311, 312, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 305, 306, 307, 308, 309, 310, 311, 312, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 305, 306, 307, 308, 309, 310, 311, 312, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 305, 306, 307, 308, 309, 310, 311, 312, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 305, 306, 307, 308, 309, 310, 311, 312, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 305, 306, 307, 308, 309, 310, 311, 312, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 105, 106, 107, 108, 109, 110, 111, 112, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 105, 106, 107, 108, 109, 110, 111, 112, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 105, 106, 107, 108, 109, 110, 111, 112, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 105, 106, 107, 108, 109, 110, 111, 112, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 105, 106, 107, 108, 109, 110, 111, 112, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 105, 106, 107, 108, 109, 110, 111, 112, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 105, 106, 107, 108, 109, 110, 111, 112, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 105, 106, 107, 108, 109, 110, 111, 112, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 113, 114, 115, 116, 117, 118, 119, 120, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 113, 114, 115, 116, 117, 118, 119, 120, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 113, 114, 115, 116, 117, 118, 119, 120, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 113, 114, 115, 116, 117, 118, 119, 120, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 113, 114, 115, 116, 117, 118, 119, 120, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 113, 114, 115, 116, 117, 118, 119, 120, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 113, 114, 115, 116, 117, 118, 119, 120, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 113, 114, 115, 116, 117, 118, 119, 120, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 81, 82, 83, 84, 85, 86, 87, 88, 233, 234, 235, 236, 237, 238, 239, 240, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 81, 82, 83, 84, 85, 86, 87, 88, 233, 234, 235, 236, 237, 238, 239, 240, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 81, 82, 83, 84, 85, 86, 87, 88, 233, 234, 235, 236, 237, 238, 239, 240, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 81, 82, 83, 84, 85, 86, 87, 88, 233, 234, 235, 236, 237, 238, 239, 240, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 81, 82, 83, 84, 85, 86, 87, 88, 233, 234, 235, 236, 237, 238, 239, 240, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 81, 82, 83, 84, 85, 86, 87, 88, 233, 234, 235, 236, 237, 238, 239, 240, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 81, 82, 83, 84, 85, 86, 87, 88, 233, 234, 235, 236, 237, 238, 239, 240, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 81, 82, 83, 84, 85, 86, 87, 88, 233, 234, 235, 236, 237, 238, 239, 240, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 97, 98, 99, 100, 101, 102, 103, 104, 249, 250, 251, 252, 253, 254, 255, 256, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 97, 98, 99, 100, 101, 102, 103, 104, 249, 250, 251, 252, 253, 254, 255, 256, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 97, 98, 99, 100, 101, 102, 103, 104, 249, 250, 251, 252, 253, 254, 255, 256, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 97, 98, 99, 100, 101, 102, 103, 104, 249, 250, 251, 252, 253, 254, 255, 256, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 97, 98, 99, 100, 101, 102, 103, 104, 249, 250, 251, 252, 253, 254, 255, 256, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 97, 98, 99, 100, 101, 102, 103, 104, 249, 250, 251, 252, 253, 254, 255, 256, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 97, 98, 99, 100, 101, 102, 103, 104, 249, 250, 251, 252, 253, 254, 255, 256, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 97, 98, 99, 100, 101, 102, 103, 104, 249, 250, 251, 252, 253, 254, 255, 256, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 89, 90, 91, 92, 93, 94, 95, 96, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 89, 90, 91, 92, 93, 94, 95, 96, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 89, 90, 91, 92, 93, 94, 95, 96, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 89, 90, 91, 92, 93, 94, 95, 96, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 89, 90, 91, 92, 93, 94, 95, 96, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 89, 90, 91, 92, 93, 94, 95, 96, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 89, 90, 91, 92, 93, 94, 95, 96, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 89, 90, 91, 92, 93, 94, 95, 96, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 1, 2, 3, 4, 321, 1, 2, 3, 4, 322, 3, 4, 5, 6, 323, 3, 4, 5, 6, 324, 1, 2, 5, 6, 325, 1, 2, 5, 6, 326, 121, 122, 123, 124, 327, 121, 122, 123, 124, 328, 123, 124, 125, 126, 329, 123, 124, 125, 126, 330, 121, 122, 125, 126, 331, 121, 122, 125, 126, 332, 57, 58, 59, 60, 333, 57, 58, 59, 60, 334, 59, 60, 61, 62, 335, 59, 60, 61, 62, 336, 57, 58, 61, 62, 337, 57, 58, 61, 62, 338, 1, 2, 7, 8, 339, 340, 3, 4, 7, 8, 341, 342, 5, 6, 7, 8, 343, 344, 1, 2, 7, 8, 339, 340, 3, 4, 7, 8, 341, 342, 5, 6, 7, 8, 343, 344, 339, 341, 343, 345, 340, 342, 344, 346, 9, 10, 15, 16, 347, 348, 11, 12, 15, 16, 349, 350, 13, 14, 15, 16, 351, 352, 9, 10, 15, 16, 347, 348, 11, 12, 15, 16, 349, 350, 13, 14, 15, 16, 351, 352, 347, 349, 351, 353, 348, 350, 352, 354, 17, 18, 23, 24, 355, 356, 19, 20, 23, 24, 357, 358, 21, 22, 23, 24, 359, 360, 17, 18, 23, 24, 355, 356, 19, 20, 23, 24, 357, 358, 21, 22, 23, 24, 359, 360, 355, 357, 359, 361, 356, 358, 360, 362, 25, 26, 31, 32, 363, 364, 27, 28, 31, 32, 365, 366, 29, 30, 31, 32, 367, 368, 25, 26, 31, 32, 363, 364, 27, 28, 31, 32, 365, 366, 29, 30, 31, 32, 367, 368, 363, 365, 367, 369, 364, 366, 368, 370, 33, 34, 39, 40, 371, 372, 35, 36, 39, 40, 373, 374, 37, 38, 39, 40, 375, 376, 33, 34, 39, 40, 371, 372, 35, 36, 39, 40, 373, 374, 37, 38, 39, 40, 375, 376, 371, 373, 375, 377, 372, 374, 376, 378, 41, 42, 47, 48, 379, 380, 43, 44, 47, 48, 381, 382, 45, 46, 47, 48, 383, 384, 41, 42, 47, 48, 379, 380, 43, 44, 47, 48, 381, 382, 45, 46, 47, 48, 383, 384, 379, 381, 383, 385, 380, 382, 384, 386, 49, 50, 55, 56, 387, 388, 51, 52, 55, 56, 389, 390, 53, 54, 55, 56, 391, 392, 49, 50, 55, 56, 387, 388, 51, 52, 55, 56, 389, 390, 53, 54, 55, 56, 391, 392, 387, 389, 391, 393, 388, 390, 392, 394, 57, 58, 63, 64, 395, 396, 59, 60, 63, 64, 397, 398, 61, 62, 63, 64, 399, 400, 57, 58, 63, 64, 395, 396, 59, 60, 63, 64, 397, 398, 61, 62, 63, 64, 399, 400, 395, 397, 399, 401, 396, 398, 400, 402, 65, 66, 71, 72, 403, 404, 67, 68, 71, 72, 405, 406, 69, 70, 71, 72, 407, 408, 65, 66, 71, 72, 403, 404, 67, 68, 71, 72, 405, 406, 69, 70, 71, 72, 407, 408, 403, 405, 407, 409, 404, 406, 408, 410, 73, 74, 79, 80, 411, 412, 75, 76, 79, 80, 413, 414, 77, 78, 79, 80, 415, 416, 73, 74, 79, 80, 411, 412, 75, 76, 79, 80, 413, 414, 77, 78, 79, 80, 415, 416, 411, 413, 415, 417, 412, 414, 416, 418, 81, 82, 87, 88, 419, 420, 83, 84, 87, 88, 421, 422, 85, 86, 87, 88, 423, 424, 81, 82, 87, 88, 419, 420, 83, 84, 87, 88, 421, 422, 85, 86, 87, 88, 423, 424, 419, 421, 423, 425, 420, 422, 424, 426, 89, 90, 95, 96, 427, 428, 91, 92, 95, 96, 429, 430, 93, 94, 95, 96, 431, 432, 89, 90, 95, 96, 427, 428, 91, 92, 95, 96, 429, 430, 93, 94, 95, 96, 431, 432, 427, 429, 431, 433, 428, 430, 432, 434, 97, 98, 103, 104, 435, 436, 99, 100, 103, 104, 437, 438, 101, 102, 103, 104, 439, 440, 97, 98, 103, 104, 435, 436, 99, 100, 103, 104, 437, 438, 101, 102, 103, 104, 439, 440, 435, 437, 439, 441, 436, 438, 440, 442, 105, 106, 111, 112, 443, 444, 107, 108, 111, 112, 445, 446, 109, 110, 111, 112, 447, 448, 105, 106, 111, 112, 443, 444, 107, 108, 111, 112, 445, 446, 109, 110, 111, 112, 447, 448, 443, 445, 447, 449, 444, 446, 448, 450, 113, 114, 119, 120, 451, 452, 115, 116, 119, 120, 453, 454, 117, 118, 119, 120, 455, 456, 113, 114, 119, 120, 451, 452, 115, 116, 119, 120, 453, 454, 117, 118, 119, 120, 455, 456, 451, 453, 455, 457, 452, 454, 456, 458]
    sp_jac_trap_ja = [0, 1, 18, 35, 52, 69, 86, 103, 119, 135, 152, 169, 186, 203, 220, 237, 253, 269, 286, 303, 320, 337, 354, 371, 387, 403, 420, 437, 454, 471, 488, 505, 521, 537, 554, 571, 588, 605, 622, 639, 655, 671, 688, 705, 722, 739, 756, 773, 789, 805, 822, 839, 856, 873, 890, 907, 923, 939, 956, 973, 990, 1007, 1024, 1041, 1057, 1073, 1090, 1107, 1124, 1141, 1158, 1175, 1191, 1207, 1224, 1241, 1258, 1275, 1292, 1309, 1325, 1341, 1358, 1375, 1392, 1409, 1426, 1443, 1459, 1475, 1492, 1509, 1526, 1543, 1560, 1577, 1593, 1609, 1626, 1643, 1660, 1677, 1694, 1711, 1727, 1743, 1760, 1777, 1794, 1811, 1828, 1845, 1861, 1877, 1894, 1911, 1928, 1945, 1962, 1979, 1995, 2011, 2027, 2043, 2059, 2075, 2091, 2107, 2123, 2139, 2163, 2187, 2211, 2235, 2259, 2283, 2307, 2331, 2363, 2395, 2427, 2459, 2491, 2523, 2555, 2587, 2619, 2651, 2683, 2715, 2747, 2779, 2811, 2843, 2867, 2891, 2915, 2939, 2963, 2987, 3011, 3035, 3067, 3099, 3131, 3163, 3195, 3227, 3259, 3291, 3315, 3339, 3363, 3387, 3411, 3435, 3459, 3483, 3507, 3531, 3555, 3579, 3603, 3627, 3651, 3675, 3707, 3739, 3771, 3803, 3835, 3867, 3899, 3931, 3955, 3979, 4003, 4027, 4051, 4075, 4099, 4123, 4147, 4171, 4195, 4219, 4243, 4267, 4291, 4315, 4339, 4363, 4387, 4411, 4435, 4459, 4483, 4507, 4531, 4555, 4579, 4603, 4627, 4651, 4675, 4699, 4723, 4747, 4771, 4795, 4819, 4843, 4867, 4891, 4923, 4955, 4987, 5019, 5051, 5083, 5115, 5147, 5171, 5195, 5219, 5243, 5267, 5291, 5315, 5339, 5371, 5403, 5435, 5467, 5499, 5531, 5563, 5595, 5619, 5643, 5667, 5691, 5715, 5739, 5763, 5787, 5811, 5835, 5859, 5883, 5907, 5931, 5955, 5979, 6011, 6043, 6075, 6107, 6139, 6171, 6203, 6235, 6259, 6283, 6307, 6331, 6355, 6379, 6403, 6427, 6459, 6491, 6523, 6555, 6587, 6619, 6651, 6683, 6715, 6747, 6779, 6811, 6843, 6875, 6907, 6939, 6971, 7003, 7035, 7067, 7099, 7131, 7163, 7195, 7219, 7243, 7267, 7291, 7315, 7339, 7363, 7387, 7392, 7397, 7402, 7407, 7412, 7417, 7422, 7427, 7432, 7437, 7442, 7447, 7452, 7457, 7462, 7467, 7472, 7477, 7483, 7489, 7495, 7501, 7507, 7513, 7517, 7521, 7527, 7533, 7539, 7545, 7551, 7557, 7561, 7565, 7571, 7577, 7583, 7589, 7595, 7601, 7605, 7609, 7615, 7621, 7627, 7633, 7639, 7645, 7649, 7653, 7659, 7665, 7671, 7677, 7683, 7689, 7693, 7697, 7703, 7709, 7715, 7721, 7727, 7733, 7737, 7741, 7747, 7753, 7759, 7765, 7771, 7777, 7781, 7785, 7791, 7797, 7803, 7809, 7815, 7821, 7825, 7829, 7835, 7841, 7847, 7853, 7859, 7865, 7869, 7873, 7879, 7885, 7891, 7897, 7903, 7909, 7913, 7917, 7923, 7929, 7935, 7941, 7947, 7953, 7957, 7961, 7967, 7973, 7979, 7985, 7991, 7997, 8001, 8005, 8011, 8017, 8023, 8029, 8035, 8041, 8045, 8049, 8055, 8061, 8067, 8073, 8079, 8085, 8089, 8093, 8099, 8105, 8111, 8117, 8123, 8129, 8133, 8137]
    sp_jac_trap_nia = 459
    sp_jac_trap_nja = 459
    return sp_jac_trap_ia, sp_jac_trap_ja, sp_jac_trap_nia, sp_jac_trap_nja 
