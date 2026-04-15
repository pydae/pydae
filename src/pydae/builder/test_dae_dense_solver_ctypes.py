import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import sys
import os
import matplotlib.pyplot as plt
import time 

# 1. Load the shared library
lib_ext = '.dll' if sys.platform == 'win32' else '.so'
lib_path = os.path.abspath(f'./daesolver{lib_ext}')

# Ensure the DLL can find dependencies if using the Conda/MKL approach
if sys.platform == 'win32' and 'CONDA_PREFIX' in os.environ:
    os.add_dll_directory(os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'bin'))

solver_lib = ctypes.CDLL(lib_path)

# 2. Define C types for NumPy arrays
c_double_array = ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
c_int_array = ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')

# 3. Define argument types for 'ini' function
# Signature: int ini(jac, pivots, x, y, xy, Dxy, u, p, Nx, Ny, max_it, itol, z, dblparams, intparams, f, g, fg)
solver_lib.ini.argtypes = [
    c_double_array, c_int_array, c_double_array, c_double_array, 
    c_double_array, c_double_array, c_double_array, c_double_array, 
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, 
    c_double_array, c_double_array, c_int_array, 
    c_double_array, c_double_array, c_double_array
]
solver_lib.ini.restype = ctypes.c_int

# 4. Define argument types for 'run' function
solver_lib.run.argtypes = [
    ctypes.c_double, ctypes.c_double, c_double_array, c_int_array,
    c_double_array, c_double_array, c_double_array, c_double_array,
    c_double_array, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_double, c_int_array, ctypes.c_double, c_double_array,
    c_double_array, c_int_array, c_double_array, c_double_array,
    c_double_array, c_double_array, ctypes.c_int, ctypes.c_int
]
solver_lib.run.restype = ctypes.c_int

def solve_system():
    # --- System Dimensions ---
    N_x, N_y, N_z = 4, 2, 4  # N_z=4 based on h_eval in temp_ctypes.c
    N = N_x + N_y
    
    # --- Solver Parameters ---
    max_it, itol = 100, 1e-8
    t_start, t_end, Dt = 0.0, 10.0, 0.01
    N_store = int((t_end - t_start) / Dt) + 1

    # --- Memory Allocation ---
    jac = np.zeros(N * N, dtype=np.float64)
    pivots = np.zeros(N, dtype=np.int32)
    x, y, xy = np.zeros(N_x), np.zeros(N_y), np.zeros(N)
    u, p, z = np.zeros(2), np.zeros(4), np.zeros(N_z)
    
    # Initialization work arrays
    Dxy, f_w, g_w, fg_w = np.zeros(N), np.zeros(N_x), np.zeros(N_y), np.zeros(N)
    ini_int = np.array([1, 0, 0, 0, 0], dtype=np.int32) # [NewtonMode, storeMode, it, fact_count, ...]
    ini_dbl = np.zeros(5)

    # --- Data Setup ---
    params_dict = {'L': 5.21, 'G': 9.81, 'M': 10.0, 'K_d': 0.1}
    u_ini_dict = {'theta': np.deg2rad(179.999), 'u_dummy': 0.0}
    u_run_dict = {'f_x': 0.0, 'u_dummy': 0.0}

    p[:] = list(params_dict.values())
    
    # Initial Guess for Steady State
    x[:] = [0, 5.21, 0.0, 0.0]
    y[:] = [0.0, 0.0]
    xy[0:N_x], xy[N_x:N] = x, y

    # --- STEP 1: INITIALIZATION ---
    #print("Computing Steady State (ini)...")
    u[:] = list(u_ini_dict.values())
    
    res_ini = solver_lib.ini(
        jac, pivots, x, y, xy, Dxy, u, p, N_x, N_y, max_it, itol, 
        z, ini_dbl, ini_int, f_w, g_w, fg_w
    )

    if res_ini != 0:
        print("Initialization failed.")
        return None
    
    #print(f"Steady State found in {ini_int[2]} iterations.")

    # --- STEP 2: RUN SIMULATION ---
    y[:] = [0.0, u[0]] # Set guess to exactly 5 degrees
    u_run_dict = {'f_x':y[0], 'u_dummy': 0.0}
    u[:] = list(u_run_dict.values()) # Apply running inputs
    
    its = np.array([0], dtype=np.int32)
    run_int = np.array([1, 0, 0], dtype=np.int32)
    run_dbl = np.zeros(5)
    
    Time = np.zeros(N_store)
    X_s, Y_s, Z_s = np.zeros(N_store * N_x), np.zeros(N_store * N_y), np.zeros(N_store * N_z)
    xy[0:N_x], xy[N_x:N] = x, y

    #print("Running Simulation...")

    res_run = solver_lib.run(
        t_start, t_end, jac, pivots, x, y, xy, u, p, 
        N_x, N_y, max_it, itol, its, Dt, z, run_dbl, run_int,
        Time, X_s, Y_s, Z_s, N_z, N_store
    )

    if res_run == 0:
        steps = its[0]
        X_final = X_s.reshape((N_store, N_x))[:steps, :]
        Y_final = Y_s.reshape((N_store, N_y))[:steps, :]
        return Time[:steps], X_final, Y_final
    
    return None

if __name__ == "__main__":
    start_time = time.perf_counter()
    result = solve_system()
    elapsed = (time.perf_counter() - start_time) * 1000
    
    if result:
        Time, X, Y = result
        print(f"Success! Elapsed time: {elapsed:.2f} ms")
        print(Y)
        
        plt.figure(figsize=(10, 5))
        angle_deg = np.rad2deg(Y[:, 1])
        plt.plot(Time, (angle_deg + 180) % 360 - 180)
        plt.title('Dynamic Response starting from Calculated Steady State')
        plt.xlabel('Time (s)'); plt.ylabel('Value'); plt.grid(True); plt.legend()
        plt.show()