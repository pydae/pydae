# -*- coding: utf-8 -*-
"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym


def DTR_Symbolic(self, line_name, bus_j, bus_k):
    """
    Symbolic implementation of the IEEE 738 Dynamic Thermal Rating model.
    Designed for integration with PyDAE or other symbolic solvers.
    """
    name = line_name

    # --- 1. Define State Variables and Inputs (Symbols) ---
    # Algebraic State (Unknown to solve for) - Lowercase
    self.t_conductor = sym.Symbol(f't_conductor_{name}', real=True)

    # Inputs (Control or Disturbance) - Lowercase
    I_j_k,I_k_j = sym.symbols(f"I_{bus_j}_{bus_k},I_{bus_k}_{bus_j}", real=True)
    #I_j_k,I_k_j = sym.symbols(f"I_{bus_j}_{bus_k}_aux,I_{bus_k}_{bus_j}_aux", real=True)

    self.K_dtr = sym.Symbol(f'K_dtr_{name}', real=True)

    self.i_line = I_j_k*self.K_dtr
    self.t_ambient = sym.Symbol(f't_ambient_{name}', real=True)
    self.v_wind = sym.Symbol(f'v_wind_{name}', real=True)
    self.angle_wind = sym.Symbol(f'angle_wind_{name}', real=True)
    self.irradiance = sym.Symbol(f'irradiance_{name}', real=True)

    # Physical Parameters (Fixed constants) - Initial Capital
    self.D_conductor = sym.Symbol(f'D_conductor_{name}', real=True, positive=True)
    self.D_core = sym.Symbol(f'D_core_{name}', real=True, positive=True)
    self.Emissivity = sym.Symbol(f'Emissivity_{name}', real=True)
    self.Absorptivity = sym.Symbol(f'Absorptivity_{name}', real=True)
    self.Elevation = sym.Symbol(f'Elevation_{name}', real=True)
    self.Mass_linear = sym.Symbol(f'Mass_linear_{name}', real=True, positive=True) # [kg/m]
    self.Cp = sym.Symbol(f'Cp_{name}', real=True, positive=True)                   # [J/(kg*C)]

    # Constants - Initial Capital
    self.Sigma = 5.670373e-8  # Stefan-Boltzmann constant
    self.Gravity = 9.81             # Gravity

    # pydae
    # self.f_list = []
    # self.x_list = []
    # self.dae['g'] = []
    # self.dae['y_ini'] = []
    # self.u_dict = {}


    # Initialize Default Values
    # Inputs
    # self.dae['u_ini_dict'].update({str(self.i_line): 0.0})
    # self.dae['u_run_dict'].update({str(self.i_line): 0.0})


    self.dae['u_ini_dict'].update({str(self.t_ambient): 25.0})
    self.dae['u_ini_dict'].update({str(self.v_wind): 1.0})
    self.dae['u_ini_dict'].update({str(self.angle_wind): 90.0})
    self.dae['u_ini_dict'].update({str(self.irradiance): 1000.0})

    
    self.dae['u_run_dict'].update({str(self.t_ambient): 25.0})
    self.dae['u_run_dict'].update({str(self.v_wind): 1.0})
    self.dae['u_run_dict'].update({str(self.angle_wind): 90.0})
    self.dae['u_run_dict'].update({str(self.irradiance): 1000.0})

    # Parameters
    self.dae['params_dict'].update({str(self.D_conductor): 0.0277})
    self.dae['params_dict'].update({str(self.D_core): 0.00308})
    self.dae['params_dict'].update({str(self.Emissivity): 0.8})
    self.dae['params_dict'].update({str(self.Absorptivity): 0.8})
    self.dae['params_dict'].update({str(self.Elevation): 0.0})
    # Default Mass (Approx for ACSR 20mm) and Cp (Aluminum/Steel mix)
    self.dae['params_dict'].update({str(self.Mass_linear): 1.62}) # kg/m
    self.dae['params_dict'].update({str(self.Cp): 900.0})         # J/kg-C

    self.dae['params_dict'].update({str(self.K_dtr): 0.0})         # J/kg-C

    """
    Constructs the symbolic heat balance equations.
    Returns:
        g_list: List of algebraic equations (g(y,u) = 0)
        y_list: List of algebraic variables
    """

    y_list_considered = ['dens','reynolds','rayleigh','nu','p_s','p_c'] #  t_conductor is always considered in y or in x, it doesn't work
    #y_list_considered = [] #  t_conductor is always considered in y, it works
    #y_list_considered = ['p_s','p_c', 'p_j'] #  t_conductor is always considered in y, it works
    # y_list_considered = ['rayleigh','p_s','p_c', 'p_j'] #  t_conductor is always considered in y, it works
    # y_list_considered = ['reynolds','rayleigh','p_s','p_c', 'p_j'] #  t_conductor is always considered in y, it works
    # y_list_considered = ['reynolds','rayleigh','nu','p_s','p_c', 'p_j','R_ac'] #  t_conductor is always considered in y, it works
    # y_list_considered = ['dens','reynolds','rayleigh','nu','p_s','p_c', 'p_j','R_ac'] #  t_conductor is always considered in y, it doesn't work
    #y_list_considered = ['reynolds','rayleigh'] #  t_conductor is always considered in y or in x, it doesn't work

    # --- A. Film Temperature ---
    t_film = 0.5 * (self.t_ambient + self.t_conductor)

    # --- B. Air Properties ---
    # Thermal conductivity (lmb)
    lmb = 2.368e-2 + 7.23e-5*t_film - 2.763e-8*t_film**2

    # Air Density (dens)
    dens_num = 1.293 - 1.525e-4*self.Elevation + 6.379e-9*self.Elevation**2
    self.dens = dens_num / (1 + 0.00367*t_film)

    # Intermediate variable for PyDAE (lowercase)

    if 'dens' in y_list_considered:
        dens = sym.Symbol(f'dens_{line_name}', real=True)
        self.dae['g'] += [self.dens - dens]
        self.dae['y_ini'] += [dens]
        self.dae['y_run'] += [dens]
    else:
        dens = self.dens


    # Dynamic Viscosity (mu_f)
    mu_f = (17.239 + 4.635e-2*t_film - 2.03e-5*t_film**2) * 1e-6

    # Kinematic Viscosity (uf)
    uf = mu_f / dens

    # --- C. Convection (p_c) Calculation ---
    # Reynolds Number
    self.reynolds = self.v_wind * self.D_conductor / uf
    if 'reynolds' in y_list_considered:
        reynolds = sym.Symbol(f'reynolds_{line_name}', real=True)
        self.dae['g'] += [self.reynolds - reynolds]
        self.dae['y_ini'] += [reynolds]
        self.dae['y_run'] += [reynolds]
    else:
        reynolds = self.reynolds

    # Roughness factor
    Rs = self.D_core / (2 * (self.D_conductor - self.D_core))

    # Forced Convection Logic
    cond_high_smooth = sym.And(reynolds > 2650, Rs <= 0.05)
    cond_high_rough  = reynolds > 2650

    B = sym.Piecewise(
        (0.178, cond_high_smooth),
        (0.048, cond_high_rough),
        (0.641, True)
    )

    n = sym.Piecewise(
        (0.633, cond_high_smooth),
        (0.800, cond_high_rough),
        (0.471, True)
    )

    nu_90 = B * reynolds**n

    # Wind Angle Correction
    phi_rad = self.angle_wind/180*np.pi
    seno = sym.Abs(sym.sin(phi_rad))
    eff_angle = sym.Mod(self.angle_wind, 180)

    delta = sym.Piecewise(
        (0.42 + 0.68 * seno**1.08, eff_angle <= 24),
        (0.42 + 0.58 * seno**0.9, True)
    )

    nu_forz = nu_90 * delta

    # Natural Convection
    Gr = (self.D_conductor**3 * (self.t_conductor - self.t_ambient) * self.Gravity) / ((t_film + 273.15) * uf**2)
    Pr_num = (1.9327e-10*t_film**4 - 7.9999e-7*t_film**3 + 1.1407e-3*t_film**2 - 0.4489*t_film + 1057.5) * mu_f / lmb

    self.rayleigh = Gr * Pr_num
    if 'rayleigh' in y_list_considered:
        rayleigh = sym.Symbol(f'rayleigh_{line_name}', real=True)
        self.dae['g'] += [self.rayleigh - rayleigh]
        self.dae['y_ini'] += [rayleigh]
        self.dae['y_run'] += [rayleigh]
    else:
        rayleigh = self.rayleigh


    A_nat = sym.Piecewise(
        (1.02,  sym.And(rayleigh > 0.1, rayleigh < 1e2)),
        (0.85,  sym.And(rayleigh >= 1e2, rayleigh < 1e4)),
        (0.48,  sym.And(rayleigh >= 1e4, rayleigh < 1e7)),
        (0.125, sym.And(rayleigh >= 1e7, rayleigh < 1e12)),
        (0.0, True)
    )

    m_nat = sym.Piecewise(
        (0.148, sym.And(rayleigh > 0.1, rayleigh < 1e2)),
        (0.188, sym.And(rayleigh >= 1e2, rayleigh < 1e4)),
        (0.25,  sym.And(rayleigh >= 1e4, rayleigh < 1e7)),
        (0.333, sym.And(rayleigh >= 1e7, rayleigh < 1e12)),
        (0.0, True)
    )

    nu_nat = A_nat * rayleigh**m_nat

    # Total Cooling Nusselt
    self.nu = sym.Max(nu_nat, nu_forz)
    if 'nu' in y_list_considered:
        nu = sym.Symbol(f'nu_{line_name}', real=True)
        self.dae['g'] += [self.nu - nu]
        self.dae['y_ini'] += [nu]
        self.dae['y_run'] += [nu]
    else:
        nu = self.nu

    # Heat Loss Terms
    self.p_c = np.pi * lmb * (self.t_conductor - self.t_ambient) * nu
    if 'p_c' in y_list_considered:
        p_c = sym.Symbol(f'p_c_{line_name}', real=True)
        self.dae['g'] += [self.p_c - p_c]
        self.dae['y_ini'] += [p_c]
        self.dae['y_run'] += [p_c]
    else:
        p_c = self.p_c

    self.p_r = np.pi * self.D_conductor * self.Sigma * self.Emissivity * ((self.t_conductor + 273.15)**4 - (self.t_ambient + 273.15)**4)
    if 'p_r' in y_list_considered:
        p_r = sym.Symbol(f'p_r_{line_name}', real=True)
        self.dae['g'] += [self.p_r - p_r]
        self.dae['y_ini'] += [p_r]
        self.dae['y_run'] += [p_r]
    else:
        p_r = self.p_r

    self.p_s = self.Absorptivity * self.irradiance * self.D_conductor
    if 'p_s' in y_list_considered:
        p_s = sym.Symbol(f'p_s_{line_name}', real=True)
        self.dae['g'] += [self.p_s - p_s]
        self.dae['y_ini'] += [p_s]
        self.dae['y_run'] += [p_s]
    else:
        p_s = self.p_s

    # Joule Heating
    R_ref_const = 1.04 * 0.000071873143
    self.R_ac = R_ref_const * (1 + 0.000937474 * (self.t_conductor - 20))
    if 'R_ac' in y_list_considered:
        R_ac = sym.Symbol(f'R_ac_{line_name}', real=True)
        self.dae['g'] += [self.R_ac - R_ac]
        self.dae['y_ini'] += [R_ac]
        self.dae['y_run'] += [R_ac]
    else:
        R_ac = self.R_ac

    self.p_j = self.i_line**2 * self.R_ac
    if 'p_j' in y_list_considered:
        p_j = sym.Symbol(f'p_j_{line_name}', real=True)
        self.dae['g'] += [self.p_j - p_j]
        self.dae['y_ini'] += [p_j]
        self.dae['y_run'] += [p_j]
    else:
        p_j = self.p_j

    # --- G. Final Balance Equation ---
    # 0 = Heat_In - Heat_Out
    # Solve for t_conductor
    self.balance = self.p_j + self.p_s - self.p_c - self.p_r
    dt_conductor = 1/(self.Cp*self.Mass_linear)*self.balance

    # Add main algebraic equation for t_conductor
    self.dae['f'] += [     dt_conductor]
    self.dae['x'] += [ self.t_conductor]


    self.dae['xy_0_dict'].update({f"p_line_pu_{line_name}":1})
    self.dae['xy_0_dict'].update({f'dens_{line_name}':1.16})
    self.dae['xy_0_dict'].update({f'reynolds_{line_name}' :  1732.85})
    self.dae['xy_0_dict'].update({f'rayleigh_{line_name}' :  20558.69})
    self.dae['xy_0_dict'].update({f'nu_{line_name}' :  21.49})
    self.dae['xy_0_dict'].update({f'p_c_{line_name}' :  17.69})
    self.dae['xy_0_dict'].update({f'p_s_{line_name}' :  22.16})
    self.dae['xy_0_dict'].update({f'R_ac_{line_name}' :  0.08})
    self.dae['xy_0_dict'].update({f'p_j_{line_name}' :  0.1})
    self.dae['xy_0_dict'].update({f't_conductor_{line_name}' : 35.14})


    self.dae['h_dict'].update({f'dens_{line_name}':dens})
    self.dae['h_dict'].update({f'reynolds_{line_name}' :  reynolds})
    self.dae['h_dict'].update({f'rayleigh_{line_name}' :  rayleigh})
    self.dae['h_dict'].update({f'nu_{line_name}' :  nu})
    self.dae['h_dict'].update({f'p_c_{line_name}' : p_c})
    self.dae['h_dict'].update({f'p_s_{line_name}' : p_s})
    self.dae['h_dict'].update({f'R_ac_{line_name}' : R_ac})
    self.dae['h_dict'].update({f'p_j_{line_name}' :  p_j})
    self.dae['h_dict'].update({f't_conductor_{line_name}' : self.t_conductor})


    return self.dae['g'], self.dae['y_ini'], self.dae['y_run']
    

def add_line_dtr(self, line):
        
    sys = self.system
   
    bus_j = line['bus_j']
    bus_k = line['bus_k']

    idx_j = self.buses_list.index(bus_j)
    idx_k = self.buses_list.index(bus_k)    

    self.A[self.it,idx_j] = 1
    self.A[self.it,idx_k] =-1   
    self.A[self.it+1,idx_j] = 1
    self.A[self.it+2,idx_k] = 1   
    
    line_name = f"{bus_j}_{bus_k}"

    DTR_Symbolic(self, line_name, bus_j, bus_k)
    
    
    g_jk = sym.Symbol(f"g_{line_name}", real=True) 
    b_jk = sym.Symbol(f"b_{line_name}", real=True) 
    bs_jk = sym.Symbol(f"bs_{line_name}", real=True) 
    self.G_primitive[self.it,self.it] = g_jk
    self.B_primitive[self.it,self.it] = b_jk
    self.B_primitive[self.it+1,self.it+1] = bs_jk/2
    self.B_primitive[self.it+2,self.it+2] = bs_jk/2

    if not 'dtr' in line:
        line.update({'dtr':False})      

    if 'X_pu' in line:
        if 'S_mva' in line: S_line = 1e6*line['S_mva']
        R = line['R_pu']*sys['S_base']/S_line  # in pu of the system base
        X = line['X_pu']*sys['S_base']/S_line  # in pu of the system base
        G =  R/(R**2+X**2)
        B = -X/(R**2+X**2)
        self.dae['params_dict'].update({f"g_{line_name}":G})
        self.dae['params_dict'].update({f'b_{line_name}':B})

    if 'X' in line:
        bus_idx = self.buses_list.index(line['bus_j'])
        U_base = self.buses[bus_idx]['U_kV']*1000
        Z_base = U_base**2/sys['S_base']
        R = line['R']/Z_base  # in pu of the system base
        X = line['X']/Z_base  # in pu of the system base
        G =  R/(R**2+X**2)
        B = -X/(R**2+X**2)
        self.dae['params_dict'].update({f"g_{line_name}":G})
        self.dae['params_dict'].update({f'b_{line_name}':B})

    if 'X_km' in line:
        bus_idx = self.buses_list.index(line['bus_j'])
        U_base = self.buses[bus_idx]['U_kV']*1000
        Z_base = U_base**2/sys['S_base']
        R = line['R_km']*line['km']/Z_base  # in pu of the system base

        X = line['X_km']*line['km']/Z_base  # in pu of the system base
        G =  R/(R**2+X**2)
        B = -X/(R**2+X**2)
        self.dae['params_dict'].update({f"g_{line_name}":G})
        self.dae['params_dict'].update({f'b_{line_name}':B})        

    self.dae['params_dict'].update({f'bs_{line_name}':0.0})
    if 'Bs_pu' in line:
        if 'S_mva' in line: S_line = 1e6*line['S_mva']
        Bs = line['Bs_pu']*S_line/sys['S_base']  # in pu of the system base
        bs = Bs
        self.dae['params_dict'][f'bs_{line_name}'] = bs

    if 'Bs_km' in line:
        bus_idx = self.buses_list.index(line['bus_j'])
        U_base = self.buses[bus_idx]['U_kV']*1000
        Z_base = U_base**2/sys['S_base']
        Y_base = 1.0/Z_base
        Bs = line['Bs_km']*line['km']/Y_base # in pu of the system base
        bs = Bs 
        self.dae['params_dict'][f'bs_{line_name}'] = bs
        
    self.it += 3


def change_line(model,bus_j,bus_k, *args,**kwagrs):
    line = kwagrs
    S_base = model.get_value('S_base')
    
    line_name = f"{bus_j}_{bus_k}"
    if 'X_pu' in line:
        if 'S_mva' in line: S_line = 1e6*line['S_mva']
        R = line['R_pu']*S_base/S_line  # in pu of the model base
        X = line['X_pu']*S_base/S_line  # in pu of the model base
    if 'X' in line:
        U_base = model.get_value(f'U_{bus_j}_n') 
        Z_base = U_base**2/S_base
        R = line['R']/Z_base  # in pu of the model base
        X = line['X']/Z_base  # in pu of the model base
    if 'X_km' in line:
        U_base = model.get_value(f'U_{bus_j}_n')
        Z_base = U_base**2/S_base
        R = line['R_km']*line['km']/Z_base  # in pu of the model base
        X = line['X_km']*line['km']/Z_base  # in pu of the model base
    if 'Bs_km' in line:
        U_base = model.get_value(f'U_{bus_j}_n')
        Z_base = U_base**2/S_base
        Y_base = 1.0/Z_base
        Bs = line['Bs_km']*line['km']/Y_base  # in pu of the model base
        bs = Bs
        model.set_value(f'bs_{line_name}',bs)

    G =  R/(R**2+X**2)
    B = -X/(R**2+X**2)
    model.set_value(f"g_{line_name}",G)
    model.set_value(f"b_{line_name}",B)

def get_line_current(model,bus_j,bus_k, units='A'):
    V_j_m,theta_j = model.get_mvalue([f'V_{bus_j}',f'theta_{bus_j}'])
    V_k_m,theta_k = model.get_mvalue([f'V_{bus_k}',f'theta_{bus_k}'])
    name = f'b_{bus_j}_{bus_k}'
    if name in model.params_list:
        B_j_k,G_j_k = model.get_mvalue([f'b_{bus_j}_{bus_k}',f'g_{bus_j}_{bus_k}'])
        Bs_j_k = model.get_value(f'bs_{bus_j}_{bus_k}')
    elif f'b_{bus_k}_{bus_j}' in model.params_list:
        B_j_k,G_j_k = model.get_mvalue([f'b_{bus_k}_{bus_j}',f'g_{bus_k}_{bus_j}'])
        Bs_j_k = model.get_value(f'bs_{bus_k}_{bus_j}')

    
    Y_j_k = G_j_k + 1j*B_j_k 
    V_j = V_j_m*np.exp(1j*theta_j)
    V_k = V_k_m*np.exp(1j*theta_k)
    I_j_k_pu = (V_j - V_k)*Y_j_k + 1j*V_j*Bs_j_k/2

    if units == 'A':
        S_b = model.get_value('S_base')
        U_b = model.get_value(f'U_{bus_j}_n')
        I_b = S_b/(np.sqrt(3)*U_b)
        I_j_k = I_j_k_pu*I_b

    return I_j_k


def test_line_pu_build():
    
    from pydae.bmapu import bmapu_builder
    from pydae.build_v2 import build_mkl,build_numba

    data = {
        "system":{"name":"temp","S_base":100e6, "K_p_agc":0.01,"K_i_agc":0.01, "K_xif":0.01},       
        "buses":[{"name":"1", "P_W":0.0,"Q_var":0.0,"U_kV":20.0},
                 {"name":"2", "P_W":0.0,"Q_var":0.0,"U_kV":20.0}],
        "lines":[{"bus_j":"1", "bus_k":"2", "X_pu":0.15,"R_pu":0.0,"Bs_pu":0.5, "S_mva":900.0, "monitor":True}],
        "sources":[{"bus":"1","type":"vsource","V_mag_pu":1.0,"V_ang_rad":0.0,"K_delta":0.1}]
        }

    grid = bmapu_builder.bmapu(data)
    grid.uz_jacs = False
    grid.verbose = True
    grid.construct(f'temp')
    build_numba(grid.sys_dict)

def test_line_pu_ini():
    
    import temp

    model = temp.model()
    model.ini({'P_2':0e6, "K_xif":0.01},'xy_0.json')
    print(f"V_1 = {model.get_value('V_1'):2.2f}, V_2 = {model.get_value('V_2'):2.2f}")

    model.report_y()

def test_line_km_build():
    
    from pydae.bmapu import bmapu_builder
    from pydae.build_v2 import build_mkl,build_numba

    R_km = 0.0268  # (Ω/km)
    X_km = 0.2766  # (Ω/km)
    B_km = 4.59e-6 # (℧-1/Km)

    N_parallel = 44.0

    Lenght_km = 22000.0/N_parallel

    data = {
        "system":{"name":"temp","S_base":100e6, "K_p_agc":0.01,"K_i_agc":0.01, "K_xif":0.01},       
        "buses":[{"name":"1", "P_W":0.0,"Q_var":0.0,"U_kV":400.0},
                 {"name":"2", "P_W":5000e6,"Q_var":0.0,"U_kV":400.0}],
        "lines":[{"bus_j":"1", "bus_k":"2", 'X_km':X_km/44, 'R_km':R_km/44, 'Bs_km':B_km*44, "km":Lenght_km, "monitor":True}],
        "sources":[{"bus":"1","type":"vsource","V_mag_pu":1.0,"V_ang_rad":0.0,"K_delta":0.1}]
        }

    grid = bmapu_builder.bmapu(data)
    grid.uz_jacs = False
    grid.verbose = True
    grid.construct(f'temp')
    build_numba(grid.sys_dict)

def test_line_km_ini():
    
    import temp

    model = temp.model()
    model.ini({'P_2':5000e6,'Q_2':0, "K_xif":0.01},'xy_0.json')
    print(f"V_1 = {model.get_value('V_1'):2.2f}, V_2 = {model.get_value('V_2'):2.2f}")

    print("report_y:")
    model.report_y()
    print("-----------------------------------------------------------------------")
    print("report_z:")
    model.report_z()

    U_1 = 400e3
    U_2 = 400e3 
    Bs = 4.59e-6*22000
    Q = U_1**2*Bs/2 + U_2**2*Bs/2
    print(f"Q = {Q/1e6:2.2f} Mvar")


if __name__ == "__main__":

    #test_line_pu_build()
    #test_line_pu_ini()

    test_line_km_build()
    test_line_km_ini()


