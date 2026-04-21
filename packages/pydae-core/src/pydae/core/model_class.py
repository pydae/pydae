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
from scipy.sparse import csc_matrix, csr_matrix
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

        # Load system metadata (JSON) first — needed to derive library name
        self._load_system_data(data_folder)

        # Derive the backend tag for the library filename
        # e.g. "pendulum_cffi_klu", "pendulum_ctypes_dense"
        backend_tag = self.sparse_backend or 'dense'
        target = self.data_dict.get('target', 'cffi')

        # 1. Attempt to load the CFFI backend (Preferred for Windows/Sparse compatibility)
        try:
            module = importlib.import_module(f"{self.model_name}_cffi_{backend_tag}")
            self.ffi = module.ffi
            self.solver_lib = module.lib
            self.is_cffi = True
        except ImportError:
            # 2. Fallback to classic ctypes backend
            self.lib_ext = '.dll' if sys.platform == 'win32' else ('.dylib' if sys.platform == 'darwin' else '.so')
            lib_filename = f"{self.model_name}_ctypes_{backend_tag}{self.lib_ext}"
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
        
        # Default solver settings
        self.max_it = 50
        self.itol = 1e-8
        self.Dt = 0.01
        self.decimation = 1
        self.alpha = 0.5  # 0.5: Trapezoidal, 1.0: Backward Euler
        self.ini_iterations = 0
        self.step_counter = 0
        self.N_store = 10_000  # default pre-allocated rows for Time/X/Y/Z

        # Anchor list for CFFI from_buffer cdata objects to keep them
        # alive through each solver invocation (see _d/_i below).
        self._ffi_pins = []


    # --- Memory Pointer Wrappers ---
    # CFFI's ffi.cast("T*", ffi.from_buffer(arr)) can release the
    # from_buffer cdata before the C call completes on some platforms
    # (observed as intermittent Windows heap corruption at 0xc0000374
    # after solver_lib.ini returns). Pin the from_buffer cdata in
    # ``self._ffi_pins`` for the duration of each solver call; the
    # solver methods clear the list afterward.
    def _d(self, arr):
        """Cast numpy double array to C pointer (cffi) or pass-through (ctypes)."""
        if self.is_cffi:
            buf = self.ffi.from_buffer(arr)
            self._ffi_pins.append(buf)
            return self.ffi.cast("double *", buf)
        return arr

    def _i(self, arr):
        """Cast numpy int array to C pointer (cffi) or pass-through (ctypes)."""
        if self.is_cffi:
            buf = self.ffi.from_buffer(arr)
            self._ffi_pins.append(buf)
            return self.ffi.cast("int *", buf)
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
            xy_0_str = fobj.read()
        xy_0_dict = json.loads(xy_0_str)
    
        for item in xy_0_dict:
            if item in self.x_list:
                self.xy_0[self.x_list.index(item)] = xy_0_dict[item]
            if item in self.y_ini_list:
                self.xy_0[self.y_ini_list.index(item)+self.N_x] = xy_0_dict[item]            


    def save_xy_0(self,file_name = 'xy_0.json'):
        xy_0_dict = {}
        for item in self.x_list:
            xy_0_dict.update({item:self.get_value(item)})
        for item in self.y_ini_list:
            xy_0_dict.update({item:self.get_value(item)})
    
        xy_0_str = json.dumps(xy_0_dict, indent=4)
        with open(file_name,'w') as fobj:
            fobj.write(xy_0_str)
    
     
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

        # Ensure independent buffers (not views) before C solve.
        # Required because ini2run() converts self.x/y_ini to views into xy_ini.
        # On Windows CFFI, passing views to C causes heap corruption.
        self.x = np.copy(self.xy_ini[:self.N_x])
        self.y_ini = np.copy(self.xy_ini[self.N_x:])

        self._ffi_pins.clear()
        res_ini = self.solver_lib.ini(
            self._d(self.jac_ini_flat), self._i(self.pivots_ini), self._d(self.x), self._d(self.y_ini), self._d(self.xy_ini), self._d(Dxy),
            self._d(self.u_ini), self._d(self.p), self.N_x, self.N_y, self.max_it, self.itol,
            self._d(self.z), self._d(ini_dbl), self._i(self.ini_int), self._d(self.f_w), self._d(self.g_w), self._d(self.fg_w)
        )
        self._ffi_pins.clear()

        self.ini_iterations = self.ini_int[2]

        self.ini2run()

        # Automatic Failure Diagnosis
        # Guard np.isnan() in try/except - crashes may occur on some platforms
        # if CFFI heap was corrupted or matplotlib has issues during diagnostics
        ini_failed = False
        try:
            ini_failed = res_ini != 0 or np.isnan(self.xy_ini).any()
        except (RuntimeError, ValueError, FloatingPointError):
            ini_failed = True

        if ini_failed:
            print("Initialization failed! Triggering automatic numerical diagnostics...\n")
            self.ini_int[4] = 1 # Enable diagnostic flag
            self._ffi_pins.clear()
            self.solver_lib.ini(
            self._d(self.jac_ini_flat), self._i(self.pivots_ini), self._d(self.x), self._d(self.y_ini), self._d(self.xy_ini), self._d(Dxy),
            self._d(self.u_ini), self._d(self.p), self.N_x, self.N_y, self.max_it, self.itol,
            self._d(self.z), self._d(ini_dbl), self._i(self.ini_int), self._d(self.f_w), self._d(self.g_w), self._d(self.fg_w)
            )
            self._ffi_pins.clear()
            try:
                diagnose_dae_model(
                    self.jac_ini_flat, self.fg_w, self.N_x, self.N_y,
                    x_names=self.x_list, y_names=self.y_ini_list,
                    sparse_backend=self.sparse_backend,
                    Ap=self.data_dict.get('Ap_ini'),
                    Ai=self.data_dict.get('Ai_ini'),
                )
            except Exception:
                pass  # Diagnostics may fail on some matplotlib environments
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

        # Ensure independent buffers (not views) before C solve
        self.x = np.copy(self.xy[:self.N_x])
        self.y_run = np.copy(self.xy[self.N_x:])

        self.solver_lib.run(
            self.t_start, t_end, self._d(jac_run_flat), self._i(pivots), self._d(self.x), self._d(self.y_run), self._d(self.xy), 
            self._d(self.u_run), self._d(self.p), self.N_x, self.N_y, self.max_it, self.itol, 
            self._i(self.it_store), self.Dt, self._d(self.z), self._d(run_dbl), self._i(run_int),
            self._d(self.Time), self._d(self.X), self._d(self.Y), self._d(self.Z), self.N_z, len(self.Time),
            self._d(self.f_w), self._d(self.g_w), self._d(self.fg_w)
        )

        self.t_start = t_end
        self.step_counter = run_int[7]

    def _sparse_to_dense_jac(self, flat, which='trap'):
        """Expand a NNZ-long Jacobian buffer to a dense ``(N_xy, N_xy)``
        array using the sparsity pattern stored in ``data_dict``.

        The JSON always stores 0-based CSC arrays (column pointers + row
        indices) regardless of backend.  The 1-based offset for PARDISO
        is only applied inside the generated C code, not in the metadata.
        """
        Ap = np.asarray(self.data_dict[f'Ap_{which}'], dtype=np.intp)
        Ai = np.asarray(self.data_dict[f'Ai_{which}'], dtype=np.intp)
        n = self.N_xy

        # All backends: JSON stores 0-based CSC
        M = csc_matrix((flat, Ai, Ap), shape=(n, n))
        return M.toarray()

    def _ensure_jac_trap_argtypes(self):
        """Lazy ctypes argtypes setup for ``jac_trap_eval``."""
        if self.is_cffi:
            return
        fn = self.solver_lib.jac_trap_eval
        if fn.argtypes:
            return
        c_double_p = ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
        fn.argtypes = [c_double_p, c_double_p, c_double_p,
                       c_double_p, c_double_p, ctypes.c_double]

    def jac_run_eval(self):
        """Evaluate the run Jacobian ``jac_run = [[F_x, F_y], [G_x, G_y]]``.

        Calls the compiled ``jac_trap_eval`` at the current operating point
        to obtain ``jac_trap``::

            jac_trap = [[I - alpha*Dt*F_x,  -alpha*Dt*F_y],
                        [       G_x,               G_y    ]]

        then recovers the run-Jacobian blocks::

            F_x = (I - jac_trap[:N_x, :N_x]) / (alpha*Dt)
            F_y = -jac_trap[:N_x, N_x:]      / (alpha*Dt)
            G_x =  jac_trap[N_x:, :N_x]
            G_y =  jac_trap[N_x:, N_x:]
        """
        self._ensure_jac_trap_argtypes()

        # Current operating point - ensure independent buffers (not views)
        self.x = np.copy(self.xy[:self.N_x])
        self.y_run = np.copy(self.xy[self.N_x:])

        jac_trap_flat = np.zeros(self.jac_size_trap, dtype=np.float64)
        self.solver_lib.jac_trap_eval(
            self._d(jac_trap_flat),
            self._d(self.x), self._d(self.y_run),
            self._d(self.u_run), self._d(self.p),
            self.Dt,
        )

        # Dense form of jac_trap
        if self.is_sparse:
            self.jac_trap = self._sparse_to_dense_jac(jac_trap_flat, which='trap')
        else:
            self.jac_trap = jac_trap_flat.reshape((self.N_xy, self.N_xy))

        # Reconstruct the run Jacobian blocks
        N_x = self.N_x
        alpha_dt = self.alpha * self.Dt

        self.F_x = (np.eye(N_x) - self.jac_trap[:N_x, :N_x]) / alpha_dt
        self.F_y = -self.jac_trap[:N_x, N_x:] / alpha_dt
        self.G_x = self.jac_trap[N_x:, :N_x]
        self.G_y = self.jac_trap[N_x:, N_x:]

        self.jac_run = np.block([[self.F_x, self.F_y], [self.G_x, self.G_y]])
        return self.jac_run

    def A_eval(self):
        """Compute the reduced linearized state matrix ``A``.

        Eliminates the algebraic variables from the DAE linearization via the
        Schur complement::

            A = F_x - F_y @ inv(G_y) @ G_x

        Requires that ``jac_run_eval`` has been called (or calls it lazily)
        so that ``F_x``, ``F_y``, ``G_x``, ``G_y`` are available.

        Returns
        -------
        numpy.ndarray
            The ``(N_x, N_x)`` state matrix of the linearized system
            ``dx/dt = A @ dx`` around the current operating point.
        """
        if not hasattr(self, "F_x"):
            self.jac_run_eval()

        if self.N_y > 0:
            # Solve G_y * Z = G_x  -> Z = inv(G_y) @ G_x
            Z = np.linalg.solve(self.G_y, self.G_x)
            self.A = self.F_x - self.F_y @ Z
        else:
            self.A = self.F_x.copy()

        return self.A

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
        if name in self.h_list:     return self.z[self.z_list.index(name)]
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
    def report_z(self, fmt='5.2f'): [print(f"{i:5s} = {self.get_value(i):{fmt}}") for i in self.z_list]
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