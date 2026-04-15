# test_ctypes_run.py
import ctypes
import numpy as np
import numpy.ctypeslib as npct
import os
from scipy.optimize import root

# 1. Definir el tipo exacto de array de NumPy que espera C (1D, float64, contiguo)
array_1d_double = npct.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')

# 2. Cargar la librería generada
dll_path = os.path.join('build', 'temp_ctypes.dll')
lib = ctypes.CDLL(dll_path)

# 1. Cargar la librería C y definir tipos
array_1d_double = npct.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
lib = ctypes.CDLL('build/temp_ctypes.dll')

# Configurar firmas (f_ini, g_ini y jac_ini)
for func in [lib.f_ini_eval, lib.g_ini_eval, lib.jac_ini_eval]:
    func.argtypes = [array_1d_double, array_1d_double, array_1d_double, 
                     array_1d_double, array_1d_double, ctypes.c_double]
    func.restype = None

params_dict = {'L':5.21,'G':9.81,'M':10.0,'K_d':1e-3}  # parameters with default values

u_ini_dict = {'theta':np.deg2rad(5.0),'u_dummy':0.0}  # input for the initialization problem
u_run_dict = {'f_x':0,'u_dummy':0.0}                  # input for the running problem, its value is updated 


# 4. Crear los arrays de prueba en Python
# (Ajusta los tamaños según el número de variables de tu sistema 'temp')
N_data = 4  # Tamaño del vector f
N_x = 4
N_y = 2
N_u = len(u_ini_dict)
N_p = len(params_dict)

N_total = N_x + N_y

# El jacobiano en C rellena un array 1D plano, así que necesitamos tamaño N*N
# Usamos ceros porque la función C solo actualiza los elementos no nulos (sparse)
jac_data_1d = np.zeros(N_total * N_total, dtype=np.float64)

f_out = np.zeros(N_x, dtype=np.float64)
g_out = np.zeros(N_y, dtype=np.float64)

x = np.ones(N_x, dtype=np.float64) * 0.0
y = np.ones(N_y, dtype=np.float64) * 0.5
u = np.array(list(u_ini_dict.values()), dtype=np.float64)
p = np.array(list(params_dict.values()), dtype=np.float64)
Dt = 0.01

# 5. ¡Llamar a la función C a la velocidad del rayo!
lib.f_ini_eval(f_out, x, y, u, p, Dt)
lib.g_ini_eval(g_out, x, y, u, p, Dt)

print("¡Cálculo exitoso desde C!")
print("Resultados en 'data':", f_out)


def system_residuals(z):
    """
    Esta es la función que SciPy va a llamar repetidamente.
    'z' es un vector que contiene [x, y] pegados.
    """
    # 1. Separar el vector z en x e y
    x = z[:N_x]
    y = z[N_x:]
    
    # 2. Llamar a las funciones ultrarrápidas de C
    lib.f_ini_eval(f_out, x, y, u, p, Dt)
    lib.g_ini_eval(g_out, x, y, u, p, Dt)
    
    # 3. Concatenar y devolver los residuos [f, g]
    return np.concatenate((f_out, g_out))

def system_jacobian(z):
    """
    Calcula la matriz Jacobiana exacta usando C.
    SciPy espera que esta función devuelva una matriz 2D de tamaño (N, N).
    """
    x = z[:N_x]
    y = z[N_x:]
    
    # IMPORTANTE: Rellenar con ceros antes de cada llamada por si algún 
    # elemento de la matriz cambió a cero en esta iteración.
    jac_data_1d.fill(0.0)
    
    # Llamar a C (rellenará solo los elementos no nulos usando sus índices de_idx)
    lib.jac_ini_eval(jac_data_1d, x, y, u, p, Dt)
    
    # Redimensionar el array 1D a una vista 2D para SciPy
    return jac_data_1d.reshape((N_total, N_total))

if __name__ == "__main__":
    # Estimación inicial (Initial guess)
    xy_0 = np.ones(N_total, dtype=np.float64) * (-5.21)
    
    print("Buscando el estado estacionario con Jacobiano analítico en C...")
    
    # 4. Magia Pura: Le pasamos 'jac=system_jacobian' al solver
    sol = root(
        system_residuals, 
        xy_0, 
        jac=system_jacobian,  # <--- ¡Aquí inyectamos la velocidad!
        method='hybr'
    )
    
    if sol.success:
        print(f"¡Convergencia alcanzada en {sol.nfev} iteraciones!")
        x_steady = sol.x[:N_x]
        y_steady = sol.x[N_x:]
        print(f"Estados x: {x_steady}")
        print(f"Estados y: {y_steady}")
    else:
        print("El solver no convergió. Motivo:", sol.message)