# pydae/core/model_class.py
"""
The Model class loads a compiled DAE shared library and provides
a high-level Python API for initialization, simulation, and analysis.
"""

import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import sys
import os
import json

from pydae.core.diagnostics.dae_check import diagnose_dae_model


class Model:
    def __init__(self, model_name, matrices_folder='./build'):
        self.model_name = model_name
        self.lib_ext = '.dll' if sys.platform == 'win32' else '.so'
        lib_filename = f"{self.model_name}_ctypes{self.lib_ext}"
        self.lib_path = os.path.abspath(os.path.join(matrices_folder, lib_filename))

        if sys.platform == 'win32' and 'CONDA_PREFIX' in os.environ:
            bin_path = os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'bin')
            if os.path.exists(bin_path):
                os.add_dll_directory(bin_path)

        if not os.path.exists(self.lib_path):
            raise FileNotFoundError(
                f"Shared library not found at {self.lib_path}. Did you run the builder first?")
        
        self.solver_lib = ctypes.CDLL(self.lib_path)
        
        c_double_array = ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
        c_int_array = ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')

        self.solver_lib.ini.argtypes = [
            c_double_array, c_int_array, c_double_array, c_double_array, 
            c_double_array, c_double_array, c_double_array, c_double_array, 
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, 
            c_double_array, c_double_array, c_int_array, 
            c_double_array, c_double_array, c_double_array
        ]
        self.solver_lib.ini.restype = ctypes.c_int

        self.solver_lib.run.argtypes = [
            ctypes.c_double, ctypes.c_double, c_double_array, c_int_array,
            c_double_array, c_double_array, c_double_array, c_double_array,
            c_double_array, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_double, c_int_array, ctypes.c_double, c_double_array,
            c_double_array, c_int_array, c_double_array, c_double_array,
            c_double_array, c_double_array, ctypes.c_int, ctypes.c_int,
            c_double_array, c_double_array, c_double_array
        ]
        self.solver_lib.run.restype = ctypes.c_int

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

        self.N_x = len(self.x_list)
        self.N_y = len(self.y_ini_list)
        self.N_z = len(self.h_list)
        self.N_u = len(self.u_ini_list) 
        self.N_p = len(self.params_list) 
        self.N_xy = self.N_x + self.N_y
        self.xy_0 = np.zeros(self.N_xy, dtype=np.float64)

        self.yini2urun = list(set(self.u_run_list).intersection(set(self.y_ini_list)))
        self.uini2yrun = list(set(self.y_run_list).intersection(set(self.u_ini_list)))    

        self.N_store = 100_000
        self.Dt = 0.01
        self.alpha = 0.5
        self.decimation = 1

    # --- Helpers ---
    def dict2xy0(self, xy_0_dict):
        for item, value in xy_0_dict.items():
            if item in self.x_list:
                self.xy_0[self.x_list.index(item)] = value
            elif item in self.y_ini_list:
                self.xy_0[self.y_ini_list.index(item) + self.N_x] = value

    def load_xy_0(self, file_name='xy_0.json'):
        with open(file_name) as fobj:
            self.dict2xy0(json.load(fobj))

    def set_value(self, name_, value):
        if name_ in self.u_ini_list:
            self.u_ini[self.u_ini_list.index(name_)] = value
        if name_ in self.u_run_list:
            self.u_run[self.u_run_list.index(name_)] = value
        elif name_ in self.params_list:
            self.p[self.params_list.index(name_)] = value
        elif name_ not in self.u_ini_list and name_ not in self.u_run_list:
            print(f"Warning: Input or parameter '{name_}' not found.")

    def get_value(self, name):
        if name in self.u_run_list: return self.u_run[self.u_run_list.index(name)]
        if name in self.x_list:     return self.xy[self.x_list.index(name)]
        if name in self.y_run_list: return self.xy[self.N_x + self.y_run_list.index(name)]
        if name in self.params_list:return self.p[self.params_list.index(name)]
        if name in self.h_list:     return self.z[self.h_list.index(name)]
        return None

    def get_values(self, name):
        if name in self.x_list:     return self.X[:, self.x_list.index(name)]
        if name in self.y_run_list: return self.Y[:, self.y_run_list.index(name)]
        if name in self.h_list:     return self.Z[:, self.h_list.index(name)]
        return None

    def get_mvalue(self, names):
        return [self.get_value(name) for name in names]

    def report_x(self, fmt='5.2f'): [print(f"{i:5s} = {self.get_value(i):{fmt}}") for i in self.x_list]
    def report_y(self, fmt='5.2f'): [print(f"{i:5s} = {self.get_value(i):{fmt}}") for i in self.y_run_list]
    def report_u(self, fmt='5.2f'): [print(f"{i:5s} = {self.get_value(i):{fmt}}") for i in self.u_run_list]
    def report_z(self, fmt='5.2f'): [print(f"{i:5s} = {self.get_value(i):{fmt}}") for i in self.h_list]
    def report_params(self, fmt='5.2f'): [print(f"{i:5s} = {self.get_value(i):{fmt}}") for i in self.params_list]

    # --- Core solver ---
    def setup_ini(self, up_dict, xy_0={}):
        self.it = 0
        self.it_store = np.array([0], dtype=np.int32)
        self.t_start = 0.0
        self.t = 0.0
        self.step_counter = 0

        self.Time = np.zeros(self.N_store, dtype=np.float64)
        self.X = np.zeros(self.N_store * self.N_x, dtype=np.float64)
        self.Y = np.zeros(self.N_store * self.N_y, dtype=np.float64)
        self.Z = np.zeros(self.N_store * self.N_z, dtype=np.float64)

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
        self.jac_ini = np.zeros(self.N_xy * self.N_xy, dtype=np.float64)
        self.pivots_ini = np.zeros(self.N_xy, dtype=np.int32)
        self.x_ini, self.y_ini = np.zeros(self.N_x), np.zeros(self.N_y)
        self.xy_ini, self.xy = np.zeros(self.N_xy), np.zeros(self.N_xy)
        self.Dxy = np.zeros(self.N_xy, dtype=np.float64)
        self.f_w = np.zeros(self.N_x, dtype=np.float64)
        self.g_w = np.zeros(self.N_y, dtype=np.float64)
        self.fg_w = np.zeros(self.N_xy, dtype=np.float64)
        self.ini_int = np.array([1, 0, 0, 0, 0, 0], dtype=np.int32) 
        self.ini_dbl = np.zeros(5, dtype=np.float64)
        self.x_ini[:] = self.xy_0[:self.N_x]
        self.y_ini[:] = self.xy_0[self.N_x:]
        self.xy_ini[:self.N_x] = self.x_ini
        self.xy_ini[self.N_x:] = self.y_ini

    def ini2run(self):
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
        self.setup_ini(up_dict, xy_0)
        self.ini_int = np.array([1, 0, 0, 0, 0, 0], dtype=np.int32) 
        res_ini = self.solver_lib.ini(
            self.jac_ini, self.pivots_ini, self.x_ini, self.y_ini, self.xy_ini, self.Dxy, self.u_ini, 
            self.p, self.N_x, self.N_y, self.max_it, self.itol, 
            self.z, self.ini_dbl, self.ini_int, self.f_w, self.g_w, self.fg_w)
        self.ini_iterations = self.ini_int[2]
        self.ini2run()
        if res_ini != 0 or np.isnan(self.xy_ini).any():
            print("Initialization failed! Triggering automatic numerical diagnostics...\n")
            self.ini_int[4] = 1
            self.solver_lib.ini(
                self.jac_ini, self.pivots_ini, self.x_ini, self.y_ini, self.xy_ini, self.Dxy, self.u_ini, 
                self.p, self.N_x, self.N_y, self.max_it, self.itol, 
                self.z, self.ini_dbl, self.ini_int, self.f_w, self.g_w, self.fg_w)
            diagnose_dae_model(self.jac_ini, self.fg_w, self.N_x, self.N_y, 
                             x_names=self.x_list, y_names=self.y_ini_list)
            self.ini_int[4] = 0 
            return False 
        return True

    def run(self, t_end, up_dict, diagnose=False):
        for item, val in up_dict.items():
            self.set_value(item, val)
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
            self.f_w, self.g_w, self.fg_w)
        if diagnose and (res_run != 0 or np.isnan(self.xy).any()):
            print(f"\nSimulation failed during interval [{self.t_start:.4f} -> {t_end:.4f}]!")
            run_int[5] = 1 
            self.solver_lib.run(
                self.t_start, t_end, jac, pivots, self.x, self.y, self.xy, self.u_run, self.p, 
                self.N_x, self.N_y, self.max_it, self.itol, self.it_store, self.Dt, self.z, run_dbl, run_int,
                self.Time, self.X, self.Y, self.Z, self.N_z, self.N_store,
                self.f_w, self.g_w, self.fg_w)
            diagnose_dae_model(jac, self.fg_w, self.N_x, self.N_y, 
                             x_names=self.x_list, y_names=self.y_run_list)
            return False
        self.step_counter = run_int[7] 
        self.t_start = t_end
        return True

    def jac_run_eval(self):
        run_int = np.array([1, 0, 0, 0, 1, 0], dtype=np.int32)
        run_dbl = np.zeros(5, dtype=np.float64)
        self.jac_run_flat = np.zeros(self.N_xy * self.N_xy, dtype=np.float64)
        pivots = np.zeros(self.N_xy, dtype=np.int32)
        self.x, self.y = self.xy[:self.N_x], self.xy[self.N_x:]
        self.solver_lib.run(
            self.t_start, 0.0, self.jac_run_flat, pivots, self.x, self.y, self.xy, self.u_run, self.p, 
            self.N_x, self.N_y, self.max_it, self.itol, self.it_store, self.Dt, self.z, run_dbl, run_int,
            self.Time, self.X, self.Y, self.Z, self.N_z, self.N_store,
            self.f_w, self.g_w, self.fg_w)
        self.jac_run = self.jac_run_flat.reshape((self.N_xy, self.N_xy))

    def post(self):
        idx = self.it_store[0]
        self.Time = self.Time[:idx]
        self.X = self.X[:self.N_x * idx].reshape(idx, self.N_x)
        self.Y = self.Y[:self.N_y * idx].reshape(idx, self.N_y)
        self.Z = self.Z[:self.N_z * idx].reshape(idx, self.N_z)

    def step(self, t_end, up_dict):
        for item, val in up_dict.items():
            self.set_value(item, val)
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
            self.f_w, self.g_w, self.fg_w)
        if res_run != 0 or np.isnan(self.xy).any():
            raise RuntimeError(f"Real-time simulation diverged at t = {self.t_start:.4f}s")
        self.step_counter = run_int[7] 
        self.t_start = t_end
        return self.xy
