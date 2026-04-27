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
import importlib
import json
import logging
import os
import sys

import numpy as np
from numpy.ctypeslib import ndpointer

# Optional diagnostic tool
from pydae.core.diagnostics.dae_check import diagnose_dae_model
from scipy.sparse import csc_matrix

# Padding shield: allocate extra elements to absorb potential C buffer overflows.
# On Windows, heap corruption triggers immediate crashes; this padding provides
# a safety margin so overflows hit padding instead of corrupting heap canaries.
PAD = 50


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

            # Fix for Windows Conda environments to find DLLs (suitesparse, mkl, compiler libs)
            if sys.platform == 'win32' and 'CONDA_PREFIX' in os.environ:
                conda_prefix = os.environ['CONDA_PREFIX']
                # Add multiple potential DLL directories
                dll_dirs = [
                    os.path.join(conda_prefix, 'Library', 'bin'),   # Conda compiler DLLs
                    os.path.join(conda_prefix, 'Library', 'lib'), # SuiteSparse/MKL libs
                    os.path.join(conda_prefix, 'envs', 'test', 'Library', 'bin'),
                    os.path.join(conda_prefix, 'envs', 'test', 'Library', 'lib'),
                    os.path.join(conda_prefix, 'envs', 'test', 'bin'),
                    os.path.join(conda_prefix, 'bin'),
                    os.path.join(sys.exec_prefix, 'Library', 'bin'),
                ]
                for d in dll_dirs:
                    if os.path.exists(d):
                        try:
                            os.add_dll_directory(d)
                            logging.debug(f"[Model] Added DLL directory: {d}")
                        except (OSError, AttributeError):
                            pass  # add_dll_directory not available on older Python

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

        # Flag to track if post() was called (needs reset before next run)
        self._post_called = False

        # Anchor list for CFFI from_buffer cdata objects to keep them
        # alive through each solver invocation (see _d/_i below).
        self._ffi_pins = []

        # ------------------------------------------------------------------
        # Dual-Buffer Architecture: Internal buffers (with PAD) for HPC performance.
        # C solver writes directly to these; public arrays are created in post().
        # ------------------------------------------------------------------
        self._Time = np.zeros(self.N_store + PAD, dtype=np.float64)
        self._X = np.zeros((self.N_store + PAD) * self.N_x, dtype=np.float64)
        self._Y = np.zeros((self.N_store + PAD) * self.N_y, dtype=np.float64)
        self._Z = np.zeros((self.N_store + PAD) * self.N_z, dtype=np.float64)
        self.it_store = np.array([0], dtype=np.int32)

        # Pre-allocate Jacobian/solver buffers once so they are never freed
        # and reallocated between ini() calls — repeated reallocation causes
        # non-deterministic Windows heap corruption with KLU/CFFI.
        self._jac_ini_flat = np.zeros(self.jac_size_ini + PAD, dtype=np.float64)
        self._pivots_ini   = np.zeros(self.N_xy + PAD, dtype=np.int32)
        self._Dxy_ini      = np.zeros(self.N_xy + PAD, dtype=np.float64)
        self._ini_int      = np.array([0, 0, 0, 0, 0], dtype=np.int32)
        self._ini_dbl      = np.zeros(5, dtype=np.float64)

        # Public arrays (created in post() - until then they are None)
        self.Time = None
        self.X = None
        self.Y = None
        self.Z = None


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
        with open(data_file, 'r', encoding='utf-8') as fobj:
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
        # h_dict -> h_list in builder, but read both for backward compatibility
        self.z_list = self.data_dict.get('h_list', self.data_dict.get('z_list', []))
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

        self.xy_0 = np.zeros(self.N_xy + PAD, dtype=np.float64)

        # Load parameter defaults
        self.params_dict = self.data_dict.get('params_dict', {})

        for name, val in self.params_dict.items():
            if name in self.params_dict:
                self.p[self.params_list.index(name)] = val
                if name == 'alpha': self.alpha = val

        # Evaluation workspace
        self.f_w = np.zeros(self.N_x + PAD, dtype=np.float64)
        self.g_w = np.zeros(self.N_y + PAD, dtype=np.float64)
        self.fg_w = np.zeros(self.N_xy + PAD, dtype=np.float64)

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
            self.xy_0 = np.ones(self.N_xy + PAD, dtype=np.float64) * xy_0

        # Apply initial guesses (use exact N_x/N_y slices, ignoring PAD region)
        self.x[:] = self.xy_0[:self.N_x]
        self.y_ini[:] = self.xy_0[self.N_x:self.N_x + self.N_y]
        self.xy_ini[:self.N_x] = self.x
        self.xy_ini[self.N_x:] = self.y_ini

        self.it_store[0] = 0  # Reset storage tracker (use scalar to match C expectations)
        self.step_counter = 0
        self.t_start = 0.0

        # Pre-allocated internal buffers are already in __init__.
        # Reset them by zeroing only the used portion (pad region remains untouched).
        n = int(self.N_store)
        self._Time[:n] = 0.0
        self._X[:n * self.N_x] = 0.0
        self._Y[:n * self.N_y] = 0.0
        self._Z[:n * self.N_z] = 0.0

        # Reuse pre-allocated Jacobian/solver buffers (zeroed in-place).
        # Never reallocate these between calls — doing so causes non-deterministic
        # Windows heap corruption because the old pointer is handed to KLU/CFFI
        # just before the allocation replaces it.
        self._jac_ini_flat[:] = 0.0
        self._pivots_ini[:]   = 0
        self._Dxy_ini[:]      = 0.0
        self._ini_int[:]      = 0
        self._ini_dbl[:]      = 0.0
        # Keep public aliases for diagnostic code that reads self.jac_ini_flat / self.ini_int
        self.jac_ini_flat = self._jac_ini_flat
        self.ini_int      = self._ini_int

        # Ensure independent buffers (not views) before C solve.
        # Required because ini2run() converts self.x/y_ini to views into xy_ini.
        # On Windows CFFI, passing views to C causes heap corruption.
        self.x = np.copy(self.xy_ini[:self.N_x])
        self.y_ini = np.copy(self.xy_ini[self.N_x:])

        self._ffi_pins.clear()
        res_ini = self.solver_lib.ini(
            self._d(self._jac_ini_flat), self._i(self._pivots_ini), self._d(self.x), self._d(self.y_ini), self._d(self.xy_ini), self._d(self._Dxy_ini),
            self._d(self.u_ini), self._d(self.p), self.N_x, self.N_y, self.max_it, self.itol,
            self._d(self.z), self._d(self._ini_dbl), self._i(self._ini_int), self._d(self.f_w), self._d(self.g_w), self._d(self.fg_w)
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
            print(f"  Debug: jac_ini_flat = {self.jac_ini_flat[:self.jac_size_ini]}")
            print(f"  Debug: NNZ_ini = {self.NNZ_ini}, jac_size = {self.jac_size_ini}")
            Ap = self.data_dict.get('Ap_ini', [])
            Ai = self.data_dict.get('Ai_ini', [])
            if Ap and Ai:
                print(f"  Debug: Ap = {Ap}")
                print(f"  Debug: Ai = {Ai}")
            self.ini_int[4] = 1 # Enable diagnostic flag
            self._ffi_pins.clear()
            self.solver_lib.ini(
            self._d(self._jac_ini_flat), self._i(self._pivots_ini), self._d(self.x), self._d(self.y_ini), self._d(self.xy_ini), self._d(self._Dxy_ini),
            self._d(self.u_ini), self._d(self.p), self.N_x, self.N_y, self.max_it, self.itol,
            self._d(self.z), self._d(self._ini_dbl), self._i(self._ini_int), self._d(self.f_w), self._d(self.g_w), self._d(self.fg_w)
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
                import traceback
                print("\n" + "="*50)
                print("CRITICAL: The diagnostic tool itself crashed!")
                print("="*50)
                traceback.print_exc()
                print("="*50 + "\n")
            self.ini_int[4] = 0
            return False


        if res_ini != 0 or self.ini_iterations >= self.max_it: return False
        return True

    def run(self, t_end, inputs_dict):
        """Simulates the time-domain evolution."""
        for k, v in inputs_dict.items(): self.set_value(k, v)

        # Clear FFI pins at start to prevent memory accumulation
        self._ffi_pins.clear()

        # After post(), subsequent runs must zero internal buffers
        # to ensure clean state
        if self._post_called:
            self._Time[:] = 0.0
            self._X[:] = 0.0
            if self.N_y > 0:
                self._Y[:] = 0.0
            if self.N_z > 0:
                self._Z[:] = 0.0
            self._post_called = False
            self.it_store[0] = 0
            self.t_start = 0.0

        dec = max(1, int(self.decimation))
        n_solver_steps = int(np.ceil((t_end - self.t_start) / self.Dt)) + 2
        req_new_rows = int(np.ceil(n_solver_steps / dec)) + 1
        req_store = int(self.it_store[0]) + req_new_rows

        # Check if we would exceed buffer (with PAD). If so, warn but continue -
        # the PAD shield absorbs small overflows.
        buffer_size = self.N_store + PAD
        if req_store > buffer_size:
            import warnings
            warnings.warn(
                f"Storage buffer size ({buffer_size}) may be exceeded. "
                f"Requested: {req_store}. PAD={PAD} shield may absorb overflow.",
                RuntimeWarning
            )

        run_int = np.array([1, 0, 0, 0, self.decimation, 0, 0, self.step_counter], dtype=np.int32)
        run_dbl = np.zeros(5, dtype=np.float64); run_dbl[0] = self.alpha

        # Jacobian buffer: sparse backends use NNZ, dense uses N_xy²
        jac_run_flat = np.zeros(self.jac_size_trap + PAD, dtype=np.float64)
        pivots = np.zeros(self.N_xy + PAD, dtype=np.int32)

        # Ensure independent buffers (not views) before C solve
        self.x = np.copy(self.xy[:self.N_x])
        self.y_run = np.copy(self.xy[self.N_x:])

        # Pass internal buffers to C solver (zero-copy, in-place)
        res_run = self.solver_lib.run(
            self.t_start, t_end, self._d(jac_run_flat), self._i(pivots), self._d(self.x), self._d(self.y_run), self._d(self.xy),
            self._d(self.u_run), self._d(self.p), self.N_x, self.N_y, self.max_it, self.itol,
            self._i(self.it_store), self.Dt, self._d(self.z), self._d(run_dbl), self._i(run_int),
            self._d(self._Time), self._d(self._X), self._d(self._Y), self._d(self._Z), self.N_z, buffer_size,
            self._d(self.f_w), self._d(self.g_w), self._d(self.fg_w)
        )

        # Clear FFI pins after C call to release stale pointers
        self._ffi_pins.clear()

        if res_run != 0:
            raise RuntimeError("C solver run() failed")

        self.t_start = t_end
        self.step_counter = run_int[7]

    def _sparse_to_dense_jac(self, flat, which='trap'):
        """Expand a NNZ-long Jacobian buffer to a dense ``(N_xy, N_xy)``
        array using the sparsity pattern stored in ``data_dict``.

        The JSON always stores 0-based CSC arrays (column pointers + row
        indices) regardless of backend.  The 1-based offset for PARDISO
        is only applied inside the generated C code, not in the metadata.
        """
        Ap = np.asarray(self.data_dict[f'Ap_{which}'], dtype=np.int32)
        Ai = np.asarray(self.data_dict[f'Ai_{which}'], dtype=np.int32)
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

        # PAD guard matches run() for safety; clear stale CFFI pins first
        self._ffi_pins.clear()
        jac_trap_flat = np.zeros(self.jac_size_trap + PAD, dtype=np.float64)
        self.solver_lib.jac_trap_eval(
            self._d(jac_trap_flat),
            self._d(self.x), self._d(self.y_run),
            self._d(self.u_run), self._d(self.p),
            self.Dt,
        )
        self._ffi_pins.clear()

        # Dense form of jac_trap (strip PAD before passing to sparse reconstructor)
        if self.is_sparse:
            self.jac_trap = self._sparse_to_dense_jac(jac_trap_flat[:self.jac_size_trap], which='trap')
        else:
            self.jac_trap = jac_trap_flat[:self.jac_size_trap].reshape((self.N_xy, self.N_xy))

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
        self.jac_run_eval()

        if self.N_y > 0:
            # Solve G_y * Z = G_x  -> Z = inv(G_y) @ G_x
            Z = np.linalg.solve(self.G_y, self.G_x)
            self.A = self.F_x - self.F_y @ Z
        else:
            self.A = self.F_x.copy()

        return self.A

    def post(self):
        """Extract safe public arrays from internal buffers.

        Performs EXACTLY ONE copy from internal buffers to public arrays.
        This breaks the memory sharing with C DLL, allowing safe
        plotting in Jupyter without heap corruption.
        """
        idx = self.it_store[0]

        # Exactly ONE copy per array - breaks C memory sharing
        self.Time = np.copy(self._Time[:idx])

        # Ensure X is always reshaped correctly (idx rows, N_x cols)
        if idx > 0 and self.N_x > 0:
            self.X = np.copy(self._X[:self.N_x * idx].reshape(idx, self.N_x))
        else:
            self.X = np.empty((0, self.N_x))

        if self.N_y > 0:
            self.Y = np.copy(self._Y[:self.N_y * idx].reshape(idx, self.N_y))
        else:
            self.Y = np.empty((0, self.N_y))

        if self.N_z > 0:
            self.Z = np.copy(self._Z[:self.N_z * idx].reshape(idx, self.N_z))
        else:
            self.Z = np.empty((0, self.N_z))

        # Flag that post() was called - next run() must reset buffers
        self._post_called = True

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
        if name in self.z_list:     return self.z[self.z_list.index(name)]
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
    import numpy as numpy
    import sympy as sym

    L,G,M,K_d,K_lam = sym.symbols('L,G,M,K_d,K_lam', real=True)
    p_x,p_y,v_x,v_y = sym.symbols('p_x,p_y,v_x,v_y', real=True)
    lam,f_x,theta,u_dummy = sym.symbols('lam,f_x,theta,u_dummy', real=True)
