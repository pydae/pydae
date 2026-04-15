import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import sys
import os
import matplotlib.pyplot as plt
import json
import time 

# Import your diagnostic script
from pydae.builder.dae_check import diagnose_dae_model

class Model:
    def __init__(self, model_name, matrices_folder='./build'):
        self.model_name = model_name
        
        # --- 1. RESOLVE SHARED LIBRARY PATH ---
        self.lib_ext = '.dll' if sys.platform == 'win32' else '.so'
        
        # Use the same naming convention as the builder: {name}_ctypes.dll/so
        lib_filename = f"{self.model_name}_ctypes{self.lib_ext}"
        
        # Ensure we are looking for the absolute path of the matrices folder
        self.lib_path = os.path.abspath(os.path.join(matrices_folder, lib_filename))

        # Windows/Conda specific: Ensure the compiler's runtime libs are findable
        if sys.platform == 'win32' and 'CONDA_PREFIX' in os.environ:
            bin_path = os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'bin')
            if os.path.exists(bin_path):
                os.add_dll_directory(bin_path)

        # --- 2. LOAD SHARED LIBRARY ---
        if not os.path.exists(self.lib_path):
            raise FileNotFoundError(f"Shared library not found at {self.lib_path}. "
                                    f"Did you run the builder first?")
        
        try:
            self.solver_lib = ctypes.CDLL(self.lib_path)
        except Exception as e:
            print(f"Error loading library: {e}")
            raise
        
        # --- 2. DEFINE C-TYPES INTERFACE ---
        c_double_array = ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
        c_int_array = ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')

        # 'ini' function signature (18 arguments)
        self.solver_lib.ini.argtypes = [
            c_double_array, c_int_array, c_double_array, c_double_array, 
            c_double_array, c_double_array, c_double_array, c_double_array, 
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, 
            c_double_array, c_double_array, c_int_array, 
            c_double_array, c_double_array, c_double_array # f, g, fg
        ]
        self.solver_lib.ini.restype = ctypes.c_int

        # 'run' function signature (27 arguments)
        self.solver_lib.run.argtypes = [
            ctypes.c_double, ctypes.c_double, c_double_array, c_int_array,
            c_double_array, c_double_array, c_double_array, c_double_array,
            c_double_array, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_double, c_int_array, ctypes.c_double, c_double_array,
            c_double_array, c_int_array, c_double_array, c_double_array,
            c_double_array, c_double_array, ctypes.c_int, ctypes.c_int,
            c_double_array, c_double_array, c_double_array # f, g, fg
        ]
        self.solver_lib.run.restype = ctypes.c_int

        # --- 3. LOAD SYSTEM DATA ---
        with open('system_data.json', 'r') as file:
            self.data_dict = json.load(file)

        self.x_list = self.data_dict['x_list']
        self.y_ini_list = self.data_dict['y_ini_list']
        self.y_run_list = self.data_dict['y_run_list']
        self.u_run_list = list(self.data_dict['u_run_dict'].keys())
        self.u_ini_list = list(self.data_dict['u_ini_dict'].keys())
        self.u_ini_values_list = list(self.data_dict['u_ini_dict'].values())
        self.u_run_values_list = list(self.data_dict['u_run_dict'].values())
        self.params_list = list(self.data_dict['params_dict'].keys())
        self.h_list = self.data_dict['h_list']

        # --- 4. SYSTEM DIMENSIONS & MAPPING ---
        self.N_x = len(self.x_list)
        self.N_y = len(self.y_ini_list)
        self.N_z = len(self.h_list)
        self.N_u = len(self.u_ini_list) 
        self.N_p = len(self.params_list) 
        self.N_xy = self.N_x + self.N_y
        self.xy_0 = np.zeros(self.N_xy, dtype=np.float64)

        self.yini2urun = list(set(self.u_run_list).intersection(set(self.y_ini_list)))
        self.uini2yrun = list(set(self.y_run_list).intersection(set(self.u_ini_list)))    

        # --- 5. SOLVER DEFAULTS ---
        self.N_store = 100_000
        self.Dt = 0.01
        self.alpha = 0.5  # 0.5 = Trapezoidal, 1.0 = Backward Euler
        self.decimation = 1  # Default to saving every step

    def dict2xy0(self, xy_0_dict):
        """Maps a dictionary of initial guesses to the flat xy_0 array."""
        for item, value in xy_0_dict.items():
            if item in self.x_list:
                self.xy_0[self.x_list.index(item)] = value
            elif item in self.y_ini_list:
                self.xy_0[self.y_ini_list.index(item) + self.N_x] = value

    def load_xy_0(self, file_name='xy_0.json'):
        """Loads initial state guesses from a JSON file."""
        with open(file_name) as fobj:
            xy_0_dict = json.load(fobj)
        self.dict2xy0(xy_0_dict)

    def set_value(self, name_, value):
        """Sets a parameter or input safely."""
        if name_ in self.u_ini_list:
            self.u_ini[self.u_ini_list.index(name_)] = value
        if name_ in self.u_run_list:
            self.u_run[self.u_run_list.index(name_)] = value
        elif name_ in self.params_list:
            self.p[self.params_list.index(name_)] = value
        elif name_ not in self.u_ini_list and name_ not in self.u_run_list:
            print(f"Warning: Input or parameter '{name_}' not found.")

    def get_value(self, name):
        """Gets a single scalar value from the current system state."""
        if name in self.u_run_list: return self.u_run[self.u_run_list.index(name)]
        if name in self.x_list:     return self.xy[self.x_list.index(name)]
        if name in self.y_run_list: return self.xy[self.N_x + self.y_run_list.index(name)]
        if name in self.params_list:return self.p[self.params_list.index(name)]
        if name in self.h_list:     return self.z[self.h_list.index(name)]
        return None

    def get_values(self, name):
        """Gets a time-series array of a variable from the stored simulation history."""
        if name in self.x_list:     return self.X[:, self.x_list.index(name)]
        if name in self.y_run_list: return self.Y[:, self.y_run_list.index(name)]
        if name in self.h_list:     return self.Z[:, self.h_list.index(name)]
        return None

    def get_mvalue(self, names):
        """Returns a list of scalar values for multiple variables."""
        return [self.get_value(name) for name in names]

    # --- REPORTING FUNCTIONS ---
    def report_x(self, fmt='5.2f'): [print(f"{i:5s} = {self.get_value(i):{fmt}}") for i in self.x_list]
    def report_y(self, fmt='5.2f'): [print(f"{i:5s} = {self.get_value(i):{fmt}}") for i in self.y_run_list]
    def report_u(self, fmt='5.2f'): [print(f"{i:5s} = {self.get_value(i):{fmt}}") for i in self.u_run_list]
    def report_z(self, fmt='5.2f'): [print(f"{i:5s} = {self.get_value(i):{fmt}}") for i in self.h_list]
    def report_params(self, fmt='5.2f'): [print(f"{i:5s} = {self.get_value(i):{fmt}}") for i in self.params_list]

    # --- CORE SOLVER METHODS ---
    def setup_ini(self, up_dict, xy_0={}):
        """Allocates memory and applies user settings before initialization."""
        self.it = 0
        self.it_store = np.array([0], dtype=np.int32)
        self.t_start = 0.0
        self.t = 0.0
        self.step_counter = 0 # Reset integration step counter

        # Output storage
        self.Time = np.zeros(self.N_store, dtype=np.float64)
        self.X = np.zeros(self.N_store * self.N_x, dtype=np.float64)
        self.Y = np.zeros(self.N_store * self.N_y, dtype=np.float64)
        self.Z = np.zeros(self.N_store * self.N_z, dtype=np.float64)

        # State vectors
        self.u_ini = np.array(self.u_ini_values_list, dtype=np.float64)
        self.u_run = np.array(self.u_run_values_list, dtype=np.float64)
        self.p = np.array(list(self.data_dict['params_dict'].values()), dtype=np.float64)
        self.z = np.zeros(self.N_z, dtype=np.float64)

        for item, val in up_dict.items():
            self.set_value(item, val)

        if isinstance(xy_0, dict):
            self.dict2xy0(xy_0)
        elif isinstance(xy_0, str) and xy_0 != 'eval':
            self.load_xy_0(file_name=xy_0)
        elif isinstance(xy_0, (float, int)):
            self.xy_0 = np.ones(self.N_xy, dtype=np.float64) * xy_0

        self.max_it, self.itol = 100, 1e-8

        # Matrix allocations
        self.jac_ini = np.zeros(self.N_xy * self.N_xy, dtype=np.float64)
        self.pivots_ini = np.zeros(self.N_xy, dtype=np.int32)
        
        self.x_ini, self.y_ini = np.zeros(self.N_x), np.zeros(self.N_y)
        self.xy_ini, self.xy = np.zeros(self.N_xy), np.zeros(self.N_xy)
        
        # Work arrays
        self.Dxy = np.zeros(self.N_xy, dtype=np.float64)
        self.f_w = np.zeros(self.N_x, dtype=np.float64)
        self.g_w = np.zeros(self.N_y, dtype=np.float64)
        self.fg_w = np.zeros(self.N_xy, dtype=np.float64)

        self.ini_int = np.array([1, 0, 0, 0, 0, 0], dtype=np.int32) 
        self.ini_dbl = np.zeros(5, dtype=np.float64)

        # Apply initial guesses
        self.x_ini[:] = self.xy_0[:self.N_x]
        self.y_ini[:] = self.xy_0[self.N_x:]
        self.xy_ini[:self.N_x] = self.x_ini
        self.xy_ini[self.N_x:] = self.y_ini

    def ini2run(self):
        """Transforms initialization states into runtime states."""
        self.y_ini = self.xy_ini[self.N_x:]
        self.y_run = np.copy(self.y_ini)
        
        for item in self.yini2urun:
            self.u_run[self.u_run_list.index(item)] = self.y_ini[self.y_ini_list.index(item)]
                
        for item in self.uini2yrun:
            self.y_run[self.y_run_list.index(item)] = self.u_ini[self.u_ini_list.index(item)]
            
        self.x = self.xy_ini[:self.N_x]
        self.xy[:self.N_x] = self.x
        self.xy[self.N_x:] = self.y_run

    def ini(self, up_dict, xy_0={}):
        """Solves the steady-state initialization system."""
        self.setup_ini(up_dict, xy_0)
        self.ini_int = np.array([1, 0, 0, 0, 0, 0], dtype=np.int32) 

        res_ini = self.solver_lib.ini(
            self.jac_ini, self.pivots_ini, self.x_ini, self.y_ini, self.xy_ini, self.Dxy, self.u_ini, 
            self.p, self.N_x, self.N_y, self.max_it, self.itol, 
            self.z, self.ini_dbl, self.ini_int, self.f_w, self.g_w, self.fg_w
        )

        self.ini_iterations = self.ini_int[2]

        self.ini2run()

        # Automatic Failure Diagnosis
        if res_ini != 0 or np.isnan(self.xy_ini).any():
            print("Initialization failed! Triggering automatic numerical diagnostics...\n")
            self.ini_int[4] = 1 # Enable diagnostic flag
            self.solver_lib.ini(
                self.jac_ini, self.pivots_ini, self.x_ini, self.y_ini, self.xy_ini, self.Dxy, self.u_ini, 
                self.p, self.N_x, self.N_y, self.max_it, self.itol, 
                self.z, self.ini_dbl, self.ini_int, self.f_w, self.g_w, self.fg_w
            )
            diagnose_dae_model(self.jac_ini, self.fg_w, self.N_x, self.N_y, x_names=self.x_list, y_names=self.y_ini_list)
            self.ini_int[4] = 0 
            return False 
        return True

    def run(self, t_end, up_dict, diagnose=False):
            for item, val in up_dict.items():
                self.set_value(item, val)

            # run_int = [NewtonMode, storeMode, it, fact_count, JacRunExtr, DiagnoseMode, Decimation, StepCounter]
            # NEW: Array is now length 8
            run_int = np.array([1, 0, 0, 0, 0, 0, self.decimation, self.step_counter], dtype=np.int32)
            
            run_dbl = np.zeros(5, dtype=np.float64)
            run_dbl[0] = self.alpha 
            
            jac = np.zeros(self.N_xy * self.N_xy, dtype=np.float64)
            pivots = np.zeros(self.N_xy, dtype=np.int32)

            self.x, self.y = self.xy[:self.N_x], self.xy[self.N_x:]
            
            res_run = self.solver_lib.run(
                self.t_start, t_end, jac, pivots, self.x, self.y, self.xy, self.u_run, self.p, 
                self.N_x, self.N_y, self.max_it, self.itol, self.it_store, self.Dt, self.z, run_dbl, run_int,
                self.Time, self.X, self.Y, self.Z, self.N_z, self.N_store,
                self.f_w, self.g_w, self.fg_w
            )
            
            # --- Automatic Runtime Failure Diagnosis ---
            if diagnose:
                if res_run != 0 or np.isnan(self.xy).any():
                    print(f"\nSimulation failed during interval [{self.t_start:.4f} -> {t_end:.4f}]!")
                    print("Triggering runtime numerical diagnostics...\n")
                    run_int[5] = 1 
                    self.solver_lib.run(
                        self.t_start, t_end, jac, pivots, self.x, self.y, self.xy, self.u_run, self.p, 
                        self.N_x, self.N_y, self.max_it, self.itol, self.it_store, self.Dt, self.z, run_dbl, run_int,
                        self.Time, self.X, self.Y, self.Z, self.N_z, self.N_store,
                        self.f_w, self.g_w, self.fg_w
                    )
                    diagnose_dae_model(jac, self.fg_w, self.N_x, self.N_y, x_names=self.x_list, y_names=self.y_run_list)
                    return False
                
            # NEW: Save the counter so it persists across multiple run() calls
            self.step_counter = run_int[7] 
            self.t_start = t_end
            return True

    def jac_run_eval(self):
        """Extracts the pure unscaled steady-state Runtime Jacobian (J_run)."""
        run_int = np.array([1, 0, 0, 0, 1, 0], dtype=np.int32) # Index 4 = 1 for extraction
        run_dbl = np.zeros(5, dtype=np.float64)
        
        self.jac_run_flat = np.zeros(self.N_xy * self.N_xy, dtype=np.float64)
        pivots = np.zeros(self.N_xy, dtype=np.int32)
        self.x, self.y = self.xy[:self.N_x], self.xy[self.N_x:]
        
        self.solver_lib.run(
            self.t_start, 0.0, self.jac_run_flat, pivots, self.x, self.y, self.xy, self.u_run, self.p, 
            self.N_x, self.N_y, self.max_it, self.itol, self.it_store, self.Dt, self.z, run_dbl, run_int,
            self.Time, self.X, self.Y, self.Z, self.N_z, self.N_store,
            self.f_w, self.g_w, self.fg_w
        )
        
        self.jac_run = self.jac_run_flat.reshape((self.N_xy, self.N_xy))

    def post(self):
        """Trims the stored buffers dynamically to exactly the number of iterations stored."""
        idx = self.it_store[0]
        self.Time = self.Time[:idx]
        self.X = self.X[:self.N_x * idx].reshape(idx, self.N_x)
        self.Y = self.Y[:self.N_y * idx].reshape(idx, self.N_y)
        self.Z = self.Z[:self.N_z * idx].reshape(idx, self.N_z)

    def step(self, t_end, up_dict):
            """
            Advances the simulation to t_end WITHOUT storing history.
            Optimized for real-time execution and control loops.
            """
            for item, val in up_dict.items():
                self.set_value(item, val)

            # run_int = [NewtonMode, storeMode, it, fact_count, JacRunExtr, DiagnoseMode, Decimation, StepCounter]
            # Notice index 1 is now set to 1 (Disable Storage)
            run_int = np.array([1, 1, 0, 0, 0, 0, self.decimation, self.step_counter], dtype=np.int32)
            
            run_dbl = np.zeros(5, dtype=np.float64)
            run_dbl[0] = self.alpha 
            
            jac = np.zeros(self.N_xy * self.N_xy, dtype=np.float64)
            pivots = np.zeros(self.N_xy, dtype=np.int32)

            self.x, self.y = self.xy[:self.N_x], self.xy[self.N_x:]
            
            res_run = self.solver_lib.run(
                self.t_start, t_end, jac, pivots, self.x, self.y, self.xy, self.u_run, self.p, 
                self.N_x, self.N_y, self.max_it, self.itol, self.it_store, self.Dt, self.z, run_dbl, run_int,
                self.Time, self.X, self.Y, self.Z, self.N_z, self.N_store,
                self.f_w, self.g_w, self.fg_w
            )
            
            if res_run != 0 or np.isnan(self.xy).any():
                # In real-time, you usually want to raise a hard exception rather than 
                # pausing to print diagnostics, to trigger safety fallbacks.
                raise RuntimeError(f"Real-time simulation diverged at t = {self.t_start:.4f}s")
                
            self.step_counter = run_int[7] 
            self.t_start = t_end
            
            # Return the raw xy array so your real-time controller can read the updated state instantly
            return self.xy
    

# ==========================================
# TESTING FUNCTIONS
# ==========================================

def test_pendulum():
    model = Model()
    M = 30.0  
    L = 5.21  
    deg = 10
    
    p_x_0 = L * np.sin(np.deg2rad(deg))
    p_y_0 = -L * np.cos(np.deg2rad(deg))
    xy_0 = {'p_x': p_x_0, 'p_y': p_y_0, 'lam': 0, 'f_x': 1}
    K_lam = 1e-6
    
    for K_d in [0,10,20]:
        t_1 = time.perf_counter_ns()
        success = model.ini({'M': M, 'L': L, 'K_lam': K_lam, 'theta': np.deg2rad(deg), 'K_d': 0.0}, xy_0=xy_0) 
        t_2 = time.perf_counter_ns()
        print(f'iterations = {model.ini_iterations}')
        if success:
            model.report_x()
            model.report_y()
            model.report_z()
            model.report_params()
            model.alpha = 0.5 # Optional: Set to 1.0 for Backward Euler
            model.Dt = 0.01
            model.decimation = 1
            
            t_3 = time.perf_counter_ns()
            model.run(1.0, {})
            model.run(20.0, {'f_x': 0.0})
            model.run(40.0, {'K_d': K_d})
            t_4 = time.perf_counter_ns()
            model.post()

            print(f'ini time: {(t_2 - t_1)*1e3/1e9} ms, run time: {(t_4 - t_3)*1e3/1e9} ms, ')
                
            plt.figure(figsize=(10, 5))
            plt.plot(model.Time, np.rad2deg(model.get_values('theta')))
            plt.title('Dynamic Response of Pendulum')
            plt.xlabel('Time (s)')
            plt.ylabel('Angle (degrees)')
            plt.grid(True)
            plt.show()


def test_ieee39():
    model = Model()

    model.ini({},xy_0=1)

    model.report_x()
    model.report_y()

    model.run(1.0,{})
    model.run(10.0,{'p_c_30':0.8})
    model.post()

    plt.figure(figsize=(10, 5))

    # Loop through the indices from 30 to 39
    for i in range(30, 40):
        var_name = f'omega_{i}'
        plt.plot(model.Time, model.get_values(var_name), label=var_name)

    plt.title('Dynamic Response starting from Calculated Steady State')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.grid(True)

    # Place the legend; 'best' or 'upper right' usually works well for multiple lines
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
    plt.tight_layout() # Adjust layout to make room for the legend if it's outside
    plt.show()

def test_nts_mpe_pod():

    from pydae.bmapu.lines import change_line

    model = Model()
    model.ini({},xy_0='nts_xy_0.json')
    change_line(model,'2','3',R_pu=0.0,X_pu=0.1,S_mva=100)
    # g_2_3 = 0.00
    # b_2_3 =-100.00
    # bs_2_3 = 0.00

    model.ini({'b_2_3':-1/0.01},xy_0='nts_xy_0.json')
    
    model.ini({'b_2_3':-1/0.1})
    model.ini({'b_2_3':-1/0.2})
    model.ini({'b_2_3':-1/0.3})
    model.ini({'b_2_3':-1/0.4})
    #model.ini({'b_2_3':-1/0.6})    
    model.report_y()
    model.jac_run_eval()

    from pydae.ssa import A_eval, damp_report


    A_eval(model)
    df = damp_report(model)
    print(df.sort_values('Damp'))


def test_planta():
    model = Model()

    gens = ["LV0101"] # ,"LV0102","LV0103","LV0201","LV0202","LV0203","LV0301","LV0302","LV0303"]

    params = {}

    plt.figure(figsize=(10, 5))

    for v_ref_GRID in np.arange(0.3, 1.1, 0.1):
        for item in gens:
            params.update({f'I_max_{item}':1.2, f'i_sr_ref_{item}':1.0, 'T_v_GRID':0.02, f'Epsilon_{item}':1e-6, f'T_lvrt_{item}':0.01})
            params.update({f'p_s_ppc_{item}':0.9})
        params.update({'S_n_GRID':1e9})



        model.ini(params,xy_0='planta_xy_0.json')
        model.report_u()
        model.decimation  = 1

        model.report_y()
        model.report_params()
        model.alpha = 0.5
        model.run(1.0,{})
        model.run(1.1,{'v_ref_GRID':v_ref_GRID})
        #model.run(1.101,{'rocov_GRID':0.0})
        model.run(1.2,{'lvrt_ext_LV0101':1})
        # for i in np.arange(0.0,0.011,0.001):
        #     model.run(1.2+i/100,{'lvrt_ext_LV0101':i*100})
        model.run(5.0,{})
        model.post()
        model.report_z()

        # Loop through the indices from 30 to 39
        for item in gens:
            var_name = f'p_s_{item}'
            plt.plot(model.Time, model.get_values(var_name), label=var_name)
            var_name = f'q_s_{item}'
            plt.plot(model.Time, model.get_values(var_name), label=var_name)

            # var_name = f'lvrt_{item}'
            # plt.plot(model.Time, model.get_values(var_name), label=var_name)
            # var_name = f'V_{item}'
            # plt.plot(model.Time, model.get_values(var_name), label=var_name)
            # var_name = f'i_mod_{item}'
            # plt.plot(model.Time, model.get_values(var_name), label=var_name)
            # var_name = f'i_mod_sat_{item}'
            # plt.plot(model.Time, model.get_values(var_name), label=var_name)

        # var_name = f'V_GRID'
        # plt.plot(model.Time, model.get_values(var_name), label=var_name)

        plt.title('Dynamic Response starting from Calculated Steady State')
        plt.xlabel('Time (s)')
        plt.ylabel('Value')
        plt.grid(True)

    # Place the legend; 'best' or 'upper right' usually works well for multiple lines
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
    plt.tight_layout() # Adjust layout to make room for the legend if it's outside
    plt.show()



if __name__ == "__main__":
    test_planta()





    # model.report_u()
    

    # model.run(1.0,{})
    # model.run(50.0,{'v_ref_4':0.98})
    # model.post()

    # plt.figure(figsize=(10, 5))

    # # Loop through the indices from 30 to 39
    # for i in [1,4]:
    #     var_name = f'omega_{i}'
    #     plt.plot(model.Time, model.get_values(var_name), label=var_name)

    # plt.title('Dynamic Response starting from Calculated Steady State')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Value')
    # plt.grid(True)

    # # Place the legend; 'best' or 'upper right' usually works well for multiple lines
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
    # plt.tight_layout() # Adjust layout to make room for the legend if it's outside
    # plt.show()