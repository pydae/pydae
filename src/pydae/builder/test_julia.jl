#import Pkg; Pkg.add("NLsolve")

using NLsolve
using LinearAlgebra

# 1. Configuración de la librería y dimensiones
const lib_name = Sys.iswindows() ? "temp_ctypes.dll" : "temp_ctypes.so"
const lib_path = joinpath(@__DIR__, "build", lib_name)

const N_x = 4
const N_y = 2
const N_total = N_x + N_y

# 2. Pre-asignar memoria
# En Julia, los arrays (Vector/Matrix) se pasan automáticamente como punteros (Ptr) a C
const data_f = zeros(Float64, N_x)
const data_g = zeros(Float64, N_y)
const jac_data_1d = zeros(Float64, N_total * N_total)

const u = [5/180*2*pi,0.0]
# Use double quotes for keys and Dict() constructor
const p = collect(values(Dict("L" => 5.21, "G" => 9.81, "M" => 10.0, "K_d" => 1e-3)))
const Dt = 0.01

# 3. Función de Residuos (mutante, terminada en '!' por convención en Julia)
function system_residuals!(F, z)
    # Usar @view evita copiar memoria al extraer x e y
    x = @view z[1:N_x]
    y = @view z[N_x+1:end]
    
    # Llamada directa a C usando ccall (¡Velocidad pura!)
    ccall((:f_ini_eval, lib_path), Cvoid, 
          (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Float64), 
          data_f, x, y, u, p, Dt)
          
    ccall((:g_ini_eval, lib_path), Cvoid, 
          (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Float64), 
          data_g, x, y, u, p, Dt)
          
    # Rellenar el vector F con los resultados
    F[1:N_x] .= data_f
    F[N_x+1:end] .= data_g
end

# 4. Función del Jacobiano Analítico
function system_jacobian!(J, z)
    x = @view z[1:N_x]
    y = @view z[N_x+1:end]
    
    fill!(jac_data_1d, 0.0)
    
    ccall((:jac_ini_eval, lib_path), Cvoid, 
          (Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Float64), 
          jac_data_1d, x, y, u, p, Dt)
          
    # Cuidado con la memoria: C usa "Row-Major" (filas primero)
    # Julia usa "Column-Major" (columnas primero). 
    # Para mapearlo correctamente, reestructuramos y transponemos.
    J_temp = transpose(reshape(jac_data_1d, N_total, N_total))
    J .= J_temp
end

# 5. Ejecución Principal
function main()
    z_guess = u = [0.0,-5.21,0.0,0.0,0.0,0.0]
    
    println("Buscando el estado estacionario en Julia con Jacobiano C...")
    
    # Envolvemos nuestras funciones para NLsolve
    df = OnceDifferentiable(system_residuals!, system_jacobian!, z_guess, z_guess)
    
    # Resolver usando el método trust_region (similar a hybr de MINPACK)
    sol = nlsolve(df, z_guess, method=:trust_region)
    
    if converged(sol)
        println("¡Convergencia alcanzada!")
        println("Iteraciones: ", sol.iterations)
        
        x_steady = sol.zero[1:N_x]
        y_steady = sol.zero[N_x+1:end]
        
        println("Estados x: ", x_steady)
        println("Estados y: ", y_steady)
    else
        println("El solver no convergió.")
    end
end

main()