from pydae.builder.parser import process_system_dict, check_system
from pydae.builder.symbolic import compute_base_jacobians,build_large_jacobians
from pydae.builder.codegen.cffi_builder import sym2c, sym2xyup,generate_and_compile_cffi
from pydae.builder.codegen.ctypes_builder import sym2c, sym2xyup, generate_and_compile_ctypes

#from pydae.builder.codegen.go_builder import generate_and_compile_go
import sympy as sym
import numpy as np
# pydae/builder/core.py
import os
import logging
import json
from sympy import Symbol, Expr

class SympyEncoder(json.JSONEncoder):
    def default(self, obj):
        # If it's a Sympy Symbol or Expression, convert to string
        if isinstance(obj, (Symbol, Expr)):
            return str(obj)
        # For everything else, use the standard behavior
        return super().default(obj)



class Builder:   
    def __init__(self, system_dict, verbose=False, API=False, target='cffi'):
        logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO if not verbose else logging.DEBUG)
        
        self.verbose = verbose
        self.raw_sys = system_dict
        self.name = self.raw_sys.get('name', 'unknown_system')
        self.target = target.lower()
        self.API = API
        self.save_sources = True
        self.uz_jacs = True 

        if not 'alpha_solver' in  self.raw_sys['params_dict']:
            self.raw_sys['params_dict'].update({'alpha_solver':0.5})

        
        # 1. Initialize ALL lists (Dynamic, Algebraic, Outputs, Jacobians)
        self.f_ini_list, self.f_run_list = [], []
        self.g_ini_list, self.g_run_list = [], []
        self.h_list = []
        
        self.jac_ini_list, self.jac_run_list, self.jac_trap_list = [], [], []
        self.Fu_list, self.Gu_list, self.Hx_list = [], [], []
        
        # Folders
        self.matrices_folder = 'build'

        if not os.path.exists(self.matrices_folder):
            os.makedirs(self.matrices_folder)


        
    def build(self):
        """
        The main orchestration pipeline.
        """
        logging.info(f"Starting build pipeline for {self.name} (Target: {self.target})...")
        
        # --- Parsing Phase ---
        self.sys, self.inirun = check_system(self.raw_sys)
        self.sys = process_system_dict(self.sys)

        # Save to file using the custom encoder
        self.system_dict_to_json = {}
        for item in ['x_list', 'y_ini_list', 'y_run_list', 'h_dict']:
            item_name = item
            if item == 'h_dict': 
                item_name = 'h_list'
            self.system_dict_to_json.update({item_name:[]})
            for item2 in self.raw_sys[item]:
                self.system_dict_to_json[item_name] += [str(item2)]

        for item in ['u_ini_dict', 'u_run_dict', 'params_dict']:
            self.system_dict_to_json.update({item:{}})
            for item2 in self.raw_sys[item]:
                self.system_dict_to_json[item].update({str(item2):float(self.raw_sys[item][item2])})
            
        print(self.system_dict_to_json)
        with open("system_data.json", "w") as fobj:
            json.dump(self.system_dict_to_json, fobj, cls=SympyEncoder, indent=4)


        
        # Create dictionaries for the code generator with the symbolic equations
        self.f_ini_list = [{'sym': eq} for eq in self.sys['f']]
        self.f_run_list = [{'sym': eq} for eq in self.sys['f']]
        self.g_ini_list = [{'sym': eq} for eq in self.sys['g']]
        self.g_run_list = [{'sym': eq} for eq in self.sys['g']]
        self.h_list     = [{'sym': eq} for eq in self.sys['h']]
        
        # --- Symbolic Math Phase ---
        # This will populate self.jac_ini_list, etc.
        self.sys = compute_base_jacobians(self.sys, self.inirun)
        build_large_jacobians(self) 
        
        # --- Translation & Code Generation Phase ---
        if self.target == 'cffi':
            logging.info("Translating symbolic equations to C strings...")
            
            # Translate Standard Functions
            sym2c(self.f_ini_list); sym2xyup(self.sys, self.f_ini_list, 'ini')
            sym2c(self.f_run_list); sym2xyup(self.sys, self.f_run_list, 'run')
            sym2c(self.g_ini_list); sym2xyup(self.sys, self.g_ini_list, 'ini')
            sym2c(self.g_run_list); sym2xyup(self.sys, self.g_run_list, 'run')
            sym2c(self.h_list);     sym2xyup(self.sys, self.h_list, 'run')
            
            # Translate Jacobians
            sym2c(self.jac_ini_list);  sym2xyup(self.sys, self.jac_ini_list, 'ini')
            sym2c(self.jac_run_list);  sym2xyup(self.sys, self.jac_run_list, 'run')
            sym2c(self.jac_trap_list); sym2xyup(self.sys, self.jac_trap_list, 'run')
            
            # Generate and Compile
            generate_and_compile_cffi(self)
            
        elif self.target == 'ctypes':
            logging.info("Translating symbolic equations to C strings...")
            
            # Translate Standard Functions
            sym2c(self.f_ini_list); sym2xyup(self.sys, self.f_ini_list, 'ini')
            sym2c(self.f_run_list); sym2xyup(self.sys, self.f_run_list, 'run')
            sym2c(self.g_ini_list); sym2xyup(self.sys, self.g_ini_list, 'ini')
            sym2c(self.g_run_list); sym2xyup(self.sys, self.g_run_list, 'run')
            sym2c(self.h_list);     sym2xyup(self.sys, self.h_list, 'run')
            
            # Translate Jacobians
            sym2c(self.jac_ini_list);  sym2xyup(self.sys, self.jac_ini_list, 'ini')
            sym2c(self.jac_run_list);  sym2xyup(self.sys, self.jac_run_list, 'run')
            sym2c(self.jac_trap_list); sym2xyup(self.sys, self.jac_trap_list, 'run')
            
            # Generate and Compile
            generate_and_compile_ctypes(self)            
        else:
            raise ValueError(f"Target '{self.target}' is not supported yet.")
            
        logging.info("Build pipeline completed successfully!")

def pendulum(model_name):
    L,G,M,K_d,K_lam = sym.symbols('L,G,M,K_d,K_lam', real=True)
    p_x,p_y,v_x,v_y = sym.symbols('p_x,p_y,v_x,v_y', real=True) 
    lam,f_x,theta,u_dummy = sym.symbols('lam,f_x,theta,u_dummy', real=True) 

    dp_x = v_x 
    dp_y = v_y
    dv_x = (-2*p_x*lam + f_x - K_d*v_x)/M
    dv_y = (-M*G - 2*p_y*lam - K_d*v_y)/M   

    g_1 = p_x**2 + p_y**2 - L**2 -lam*K_lam
    g_2 = -theta + sym.atan2(p_x,-p_y) + u_dummy

    params_dict = {'L':5.21,'G':9.81,'M':10.0,'K_d':1e-3,'K_lam':1e-6}  # parameters with default values

    u_ini_dict = {'theta':np.deg2rad(5.0),'u_dummy':0.0}  # input for the initialization problem
    u_run_dict = {'f_x':0,'u_dummy':0.0}                  # input for the running problem, its value is updated 

    sys_dict = {'name':model_name, 'target':'ctypes',
                'params_dict':params_dict,
                'f_list':[dp_x,dp_y,dv_x,dv_y],
                'g_list':[g_1,g_2],
                'x_list':[ p_x, p_y, v_x, v_y],
                'y_ini_list':[lam,f_x],
                'y_run_list':[lam,theta],
                'u_ini_dict':u_ini_dict,
                'u_run_dict':u_run_dict,
                'h_dict':{'E_p':M*G*(p_y+L),'E_k':0.5*M*(v_x**2+v_y**2),'f_x':f_x,'lam':lam}} 

    return sys_dict


def test_pendulum():
    sys_dict = pendulum('temp')
    bld = Builder(sys_dict, target='ctypes')
    bld.build()
    
def test_ieee39():
    from pydae.bmapu import bmapu_builder

    grid = bmapu_builder.bmapu(r'newengland.json')
    grid.construct('newengland')
    bld = Builder(grid.sys_dict, target='ctypes')
    bld.build()
    
def test_nts():
    from pydae.bmapu import bmapu_builder

    grid = bmapu_builder.bmapu(r'nts_mpe_pod.hjson')
    grid.construct('nts')
    bld = Builder(grid.sys_dict, target='ctypes')
    bld.build()
    

    # import pydae.build_cffi as db
    # sys_dict = dae()
    # bldr = db.builder(sys_dict)
    # bldr.build()

    # from pydae import ssa
    # import pendulum 

    # model = pendulum.model()

    # M = 30.0  # mass of the bob (kg)
    # L = 5.21  # length of the pendulum (m)
    # model.ini({'M':M,'L':L,           # parameters setting
    #         'theta':np.deg2rad(0)  # initial desired angle = 0º
    #         },-1)                  # here -1 means that -1 is considered as initial gess for
    #                                 # dynamic and algebraic states

    # model.report_x()  # obtained dynamic states
    # model.report_y()  # obtained algebraic states
    # model.report_z()  # obtained outputs
    # model.report_u()  # obtained algebraic states (theta is both state and output; f_x is both input and output)
    # model.report_params()  # considered parameters

if __name__ == '__main__':

    from pydae.bmapu import bmapu_builder

    grid = bmapu_builder.bmapu(r"C:\Users\jmmau\workspace\ingelectus\lvrt\config_planta_scib.json")
    grid.construct('planta')
    bld = Builder(grid.sys_dict, target='ctypes')
    bld.build()
    