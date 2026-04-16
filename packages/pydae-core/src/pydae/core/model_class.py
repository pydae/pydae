# packages/pydae-core/src/pydae/core/model_class.py
"""
The Model class orchestrates the interaction between Python and the compiled 
C solver. It handles memory allocation, data loading, and provides the API 
for steady-state initialization and time-domain simulation.

Sparse backend support
======================
The compiled shared library may use one of four linear-algebra backends:

  * ``dense``       – built-in LU with partial pivoting (default)
  * ``klu``         – SuiteSparse KLU (0-based CSC)
  * ``pardiso``     – Intel MKL PARDISO (1-based CSR)
  * ``accelerate``  – Apple Accelerate Sparse Solvers (0-based CSC, macOS)

The backend that was selected at *build time* is recorded in the JSON
metadata file under the key ``"sparse_backend"`` (``null`` for dense).
At runtime the Model class reads this value to allocate the correct
Jacobian buffer size: ``NNZ`` entries for any sparse backend, or
``N_xy * N_xy`` for the dense fallback.
"""

import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import sys
import os
import json
import importlib

# Optional diagnostic tool
from pydae.core.diagnostics.dae_check import diagnose_dae_model


class Model:
    def __init__(self, model_name, matrices_folder='./build', data_folder='.'):
        self.model_name = model_name
        self.is_cffi = False
        
        # Add the build folder to sys.path to ensure CFFI modules are importable
        sys.path.insert(0, os.path.abspath(matrices_folder))
        
        # 1. Attempt to load the CFFI backend (Preferred for Windows/Sparse compatibility)
        try:
            module = importlib.import_module(f"{self.model_name}_cffi")
            self.ffi = module.ffi
            self.solver_lib = module.lib
            self.is_cffi = True
        except ImportError:
            # 2. Fallback to classic ctypes backend
            self.lib_ext = '.dll' if sys.platform == 'win32' else ('.dylib' if sys.platform == 'darwin' else '.so')
            lib_filename = f"{self.model_name}_ctypes{self.lib_ext}"
            self.lib_path = os.path.abspath(os.path.join(matrices_folder, lib_filename))

            # Fix for Windows Conda environments to find compiler-related DLLs
            if sys.platform == 'win32' and 'CONDA_PREFIX' in os.environ:
                bin_path = os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'bin')
                if os.path.exists(bin_path):
                    os.add_dll_directory(bin_path)

            if not os.path.exists(self.lib_path):
                raise FileNotFoundError(f"Shared library not found for {self.model_name}. Did you run the builder?")
            
            self.solver_lib = ctypes.CDLL(self.lib_path)
            
            # Define pointer types for ctypes
            c_double_p = ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
            c_int_p = ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')

            # Signatures for ini (18 arguments) and run (27 arguments)
            self.solver_lib.ini.argtypes = [
                c_double_p, c_int_p, c_double_p, c_double_p, c_double_p, c_double_p, 
                c_double_p, c_double_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, 
                c_double_p, c_double_p, c_int_p, c_double_p, c_double_p, c_double_p
            ]
            self.solver_lib.run.argtypes = [
                ctypes.c_double, ctypes.c_double, c_double_p, c_int_p, c_double_p, c_double_p, 
                c_double_p, c_double_p, c_double_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, 
                ctypes.c_double, c_int_p, ctypes.c_double, c_double_p, c_double_p, c_int_p, 
                c_double_p, c_double_p, c_double_p, c_double_p, ctypes.c_int, ctypes.c_int, 
                c_double_p, c_double_p, c_double_p
            ]

        # Load system metadata (JSON)
        self._load_system_data(data_folder)
        
        # Default solver settings
        self.max_it = 50
        self.itol = 1e-8
        self.Dt = 0.01
        self.decimation = 1
        self.alpha = 0.5  # 0.5: Trapezoidal, 1.0: Backward Euler
        self.ini_iterations = 0
        self.step_counter = 0
        self.N_store = 10_000  # default pre-allocated rows for Time/X/Y/Z
        

    # --- Memory Pointer Wrappers ---
    def _d(self, arr):
        """Cast numpy double array to C pointer."""
        if self.is_cffi:
            return self.ffi.cast("double *", self.ffi.from_buffer(arr))
        return arr

    def _i(self, arr):
        """Cast numpy int array to C pointer."""
        if self.is_cffi:
            return self.ffi.cast("int *", self.ffi.from_buffer(arr))
        return arr

    def _load_system_data(self, folder):
        """Loads variable names, indices, and dimensions from the JSON metadata."""
        data_file = os.path.join(folder, f"{self.model_name}_data.json")
        with open(data_file, 'r') as fobj:
            self.data_dict = json.load(fobj)

        # ------------------------------------------------------------------
        # Sparse backend metadata (written by the builder at build time)
        # ------------------------------------------------------------------
        self.sparse_backend = self.data_dict.get('sparse_backend', None)
        self.is_sparse = self.sparse_backend is not None
        self.NNZ_ini = self.data_dict.get('NNZ_ini', 0)
        self.NNZ_trap = self.data_dict.get('NNZ_trap', 0)

        # Name Lists
        self.x_list = self.data_dict.get('x_list', [])
        self.y_ini_list = self.data_dict.get('y_ini_list', [])
        self.y_run_list = self.data_dict.get('y_run_list', [])
        self.params_list = list(self.data_dict['params_dict'].keys())
        self.z_list = self.data_dict.get('z_list', [])
        self.u_run_list = list(self.data_dict['u_run_dict'].keys())
        self.u_ini_list = list(self.data_dict['u_ini_dict'].keys())
        self.u_ini_values_list = list(self.data_dict['u_ini_dict'].values())
        self.u_run_values_list = list(self.data_dict['u_run_dict'].values())

        # Dimensions
        self.N_x, self.N_y, self.N_z = len(self.x_list), len(self.y_ini_list), len(self.z_list)
        self.N_xy = self.N_x + self.N_y

        # ------------------------------------------------------------------
        # Jacobian buffer sizes: NNZ for sparse backends, N_xy² for dense
        # ------------------------------------------------------------------
        if self.is_sparse:
            self.jac_size_ini = self.NNZ_ini
            self.jac_size_trap = self.NNZ_trap
        else:
            self.jac_size_ini = self.N_xy * self.N_xy
            self.jac_size_trap = self.N_xy * self.N_xy

        # Working memory buffers
        self.x = np.zeros(self.N_x, dtype=np.float64)
        self.y = np.zeros(self.N_y, dtype=np.float64)
        self.y_ini = np.zeros(self.N_y, dtype=np.float64)
        self.xy = np.zeros(self.N_xy, dtype=np.float64)
        self.xy_ini = np.zeros(self.N_xy, dtype=np.float64)
        self.z = np.zeros(self.N_z, dtype=np.float64)
        self.u_ini = np.array(self.u_ini_values_list, dtype=np.float64)
        self.u_run = np.array(self.u_run_values_list, dtype=np.float64)
        self.p = np.zeros(max(1, len(self.params_list)), dtype=np.float64)

        self.yini2urun = list(set(self.u_run_list).intersection(set(self.y_ini_list)))
        self.uini2yrun = list(set(self.y_run_list).intersection(set(self.u_ini_list)))    

        self.xy_0 = np.zeros(self.N_xy, dtype=np.float64)

        # Load parameter defaults
        self.params_dict = self.data_dict.get('params_dict', {})

        for name, val in self.params_dict.items():
            if name in self.params_dict:
                self.p[self.params_list.index(name)] = val
                if name == 'alpha': self.alpha = val

        # Evaluation workspace
        self.f_w = np.zeros(self.N_x, dtype=np.float64)
        self.g_w = np.zeros(self.N_y, dtype=np.float64)
        self.fg_w = np.zeros(self.N_xy, dtype=np.float64)

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

    def dict2xy0(self, xy_0_dict):
        """Maps a dictionary of initial guesses to the flat xy_0 array."""
        for item, value in xy_0_dict.items():
            if item in self.x_list:
                self.xy_0[self.x_list.index(item)] = value
            elif item in self.y_ini_list:
                self.xy_0[self.y_ini_list.index(item) + self.N_x] = value

    def load_xy_0(self, file_name='xy_0.json'):
        """Load initial guesses for the xy_0 vector from a JSON file.

        The file must contain a flat mapping of variable name -> value.
        Keys that do not match any entry in ``x_list`` or ``y_ini_list``
        are silently ignored.
        """
        with open(file_name) as fobj:
            xy_0_dict = json.loads(fobj.read())
        self.dict2xy0(xy_0_dict)

    def ini(self, params_dict, xy_0='None'):
        """Solves the steady-state (Newton-Raphson)."""
        for k, v in params_dict.items(): self.set_value(k, v)
        if isinstance(xy_0, dict):
            self.dict2xy0(xy_0)
        elif isinstance(xy_0, str) and xy_0 != 'eval':
            self.load_xy_0(file_name=xy_0)
        elif isinstance(xy_0, (float, int)):
            self.xy_0 = np.ones(self.N_xy, dtype=np.float64) * xy_0

        # Apply initial guesses
        self.x[:] = self.xy_0[:self.N_x]
        self.y_ini[:] = self.xy_0[self.N_x:]
        self.xy_ini[:self.N_x] = self.x
        self.xy_ini[self.N_x:] = self.y_ini

        self.it_store = np.array([0], dtype=np.int32)
        self.step_counter = 0
        self.t_start = 0.0

        # Pre-allocate fixed storage (old-pydae pattern): N_store rows,
        # truncated later in post(). Reused across multiple run() calls.
        n = int(self.N_store)
        self.Time = np.zeros(n, dtype=np.float64)
        self.X = np.zeros(n * self.N_x, dtype=np.float64)
        self.Y = np.zeros(n * self.N_y, dtype=np.float64)
        self.Z = np.zeros(n * self.N_z, dtype=np.float64)

        # Jacobian buffer: sparse backends use NNZ, dense uses N_xy²
        self.jac_ini_flat = np.zeros(self.jac_size_ini, dtype=np.float64)
        self.pivots_ini = np.zeros(self.N_xy, dtype=np.int32)
        Dxy = np.zeros(self.N_xy, dtype=np.float64)
        self.ini_int = np.array([0, 0, 0, 0, 0], dtype=np.int32)
        ini_dbl = np.zeros(5, dtype=np.float64)

        res_ini = self.solver_lib.ini(
            self._d(self.jac_ini_flat), self._i(self.pivots_ini), self._d(self.x), self._d(self.y_ini), self._d(self.xy_ini), self._d(Dxy), 
            self._d(self.u_ini), self._d(self.p), self.N_x, self.N_y, self.max_it, self.itol, 
            self._d(self.z), self._d(ini_dbl), self._i(self.ini_int), self._d(self.f_w), self._d(self.g_w), self._d(self.fg_w)
        )

        self.ini_iterations = self.ini_int[2]

        self.ini2run()

        # Automatic Failure Diagnosis
        if res_ini != 0 or np.isnan(self.xy_ini).any():
            print("Initialization failed! Triggering automatic numerical diagnostics...\n")
            self.ini_int[4] = 1 # Enable diagnostic flag
            self.solver_lib.ini(
            self._d(self.jac_ini_flat), self._i(self.pivots_ini), self._d(self.x), self._d(self.y_ini), self._d(self.xy_ini), self._d(Dxy), 
            self._d(self.u_ini), self._d(self.p), self.N_x, self.N_y, self.max_it, self.itol, 
            self._d(self.z), self._d(ini_dbl), self._i(self.ini_int), self._d(self.f_w), self._d(self.g_w), self._d(self.fg_w)
            )
            diagnose_dae_model(
                self.jac_ini_flat, self.fg_w, self.N_x, self.N_y,
                x_names=self.x_list, y_names=self.y_ini_list,
                sparse_backend=self.sparse_backend,
                Ap=self.data_dict.get('Ap_ini'),
                Ai=self.data_dict.get('Ai_ini'),
            )
            self.ini_int[4] = 0 
            return False 
        

        if res_ini != 0 or self.ini_iterations >= self.max_it: return False
        return True

    def run(self, t_end, inputs_dict):
        """Simulates the time-domain evolution."""
        for k, v in inputs_dict.items(): self.set_value(k, v)

        # Storage: pre-allocated in ini() with N_store rows. Only grow
        # (doubling) if the decimation-aware required row count exceeds
        # current capacity. This mirrors the old-pydae pattern and avoids
        # per-run over-allocation.
        self.t_start = getattr(self, 't_start', 0.0)
        dec = max(1, int(self.decimation))
        n_solver_steps = int(np.ceil((t_end - self.t_start) / self.Dt)) + 2
        req_new_rows = int(np.ceil(n_solver_steps / dec)) + 1
        req_store = int(self.it_store[0]) + req_new_rows

        if not hasattr(self, 'Time') or len(self.Time) < req_store:
            # Doubling growth, starting from N_store
            new_size = max(int(self.N_store), len(getattr(self, 'Time', [])))
            while new_size < req_store:
                new_size *= 2
            new_Time = np.zeros(new_size, dtype=np.float64)
            new_X = np.zeros(new_size * self.N_x, dtype=np.float64)
            new_Y = np.zeros(new_size * self.N_y, dtype=np.float64)
            new_Z = np.zeros(new_size * self.N_z, dtype=np.float64)

            if hasattr(self, 'Time') and self.it_store[0] > 0:
                idx = int(self.it_store[0])
                new_Time[:idx] = self.Time[:idx]
                new_X[:idx * self.N_x] = self.X[:idx * self.N_x]
                new_Y[:idx * self.N_y] = self.Y[:idx * self.N_y]
                new_Z[:idx * self.N_z] = self.Z[:idx * self.N_z]
            self.Time, self.X, self.Y, self.Z = new_Time, new_X, new_Y, new_Z

        run_int = np.array([1, 0, 0, 0, self.decimation, 0, 0, self.step_counter], dtype=np.int32)
        run_dbl = np.zeros(5, dtype=np.float64); run_dbl[0] = self.alpha 

        # Jacobian buffer: sparse backends use NNZ, dense uses N_xy²
        jac_run_flat = np.zeros(self.jac_size_trap, dtype=np.float64)
        pivots = np.zeros(self.N_xy, dtype=np.int32)

        self.solver_lib.run(
            self.t_start, t_end, self._d(jac_run_flat), self._i(pivots), self._d(self.x), self._d(self.y_run), self._d(self.xy), 
            self._d(self.u_run), self._d(self.p), self.N_x, self.N_y, self.max_it, self.itol, 
            self._i(self.it_store), self.Dt, self._d(self.z), self._d(run_dbl), self._i(run_int),
            self._d(self.Time), self._d(self.X), self._d(self.Y), self._d(self.Z), self.N_z, len(self.Time),
            self._d(self.f_w), self._d(self.g_w), self._d(self.fg_w)
        )

        self.t_start = t_end
        self.step_counter = run_int[7]

    def post(self):
        """Truncates pre-allocated arrays and reshapes results."""
        idx = self.it_store[0]
        self.Time = self.Time[:idx]
        self.X = self.X[:self.N_x * idx].reshape(idx, self.N_x)
        if self.N_y > 0: self.Y = self.Y[:self.N_y * idx].reshape(idx, self.N_y)
        if self.N_z > 0: self.Z = self.Z[:self.N_z * idx].reshape(idx, self.N_z)

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
        if name in self.z_list:     return self.Z[:, self.z_list.index(name)]
        return None

    # --- REPORTING FUNCTIONS ---
    def report_x(self, fmt='5.2f'): [print(f"{i:5s} = {self.get_value(i):{fmt}}") for i in self.x_list]
    def report_y(self, fmt='5.2f'): [print(f"{i:5s} = {self.get_value(i):{fmt}}") for i in self.y_run_list]
    def report_u(self, fmt='5.2f'): [print(f"{i:5s} = {self.get_value(i):{fmt}}") for i in self.u_run_list]
    def report_z(self, fmt='5.2f'): [print(f"{i:5s} = {self.get_value(i):{fmt}}") for i in self.h_list]
    def report_params(self, fmt='5.2f'): [print(f"{i:5s} = {self.get_value(i):{fmt}}") for i in self.params_list]



# ==========================================
# TESTING FUNCTIONS
# ==========================================

def test_build_pendulum(model_name,  target='ctypes', sparse=False):
    import sympy as sym
    import numpy as numpy
    from pydae.core import Builder

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

    bld = Builder(sys_dict, target=target, sparse=sparse)
    bld.build()

def test_pendulum():

    import time

    model = Model('temp')
    model.report_params()
    M = 30.0  
    L = 5.21  
    deg = 30
    
    p_x_0 = L * np.sin(np.deg2rad(deg))
    p_y_0 = -L * np.cos(np.deg2rad(deg))
    xy_0 = {'p_x': p_x_0, 'p_y': p_y_0, 'lam': 55, 'f_x': 1}
    K_lam = 1e-6
 
    t_1 = time.perf_counter_ns()
    success = model.ini({'M': M, 'L': L, 'K_lam': K_lam, 'theta': np.deg2rad(deg), 'K_d': 0.0}, xy_0=xy_0) 
    t_2 = time.perf_counter_ns()
    model.report_y()

    t_3 = time.perf_counter_ns()
    model.run(1.0, {})
    model.run(20.0, {'f_x': 0.0})
    #model.run(40.0, {'K_d': 10})
    t_4 = time.perf_counter_ns()
    model.post()

    print("model.Time", model.Time)
    print("model.get_values('theta')", model.get_values('theta'))

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(model.Time, np.rad2deg(model.get_values('theta')))