# -*- coding: utf-8 -*-
"""
Synchronous machine model of order 6 (Subtransient Model) with PSAT Saturation.

**Auxiliar equations**

$$v_d = V \sin(\delta - \theta)$$
$$v_q = V \cos(\delta - \theta)$$
$$p_e = i_d \left(v_d + R_a i_d\right) + i_q \left(v_q + R_a i_q\right)$$
$$v_{sat} = \sqrt{e'^2_q + e'^2_d}$$
$$S_{at} = \frac{B_{sat} (v_{sat} - A_{sat})^2}{v_{sat}} \quad \text{if } v_{sat} > A_{sat} \text{ else } 0$$
$$S_d = S_{at}$$
$$S_q = \frac{X_q}{X_d} S_{at}$$
$$\omega_s = \omega_{coi}$$

**Dynamic equations**

$$\frac{ d\delta}{dt} = \Omega_b \left(\omega - \omega_s \right) - K_{\delta} \delta$$
$$\frac{ d\omega}{dt} = \frac{1}{2H} \left(p_m - p_e - D \left(\omega - \omega_s \right) \right)$$
$$\frac{ de'_q}{dt} = \frac{1}{T'_{d0}} \left(-e'_q(1 + S_d) - \left(X_d - X'_d - \frac{T''_{d0}}{T'_{d0}} \frac{X''_d}{X'_d} (X_d - X'_d)\right)i_d + \left(1 - \frac{T_{AA}}{T'_{d0}}\right)v_f \right)$$
$$\frac{ de'_d}{dt} = \frac{1}{T'_{q0}} \left(-e'_d(1 + S_q) + \left(X_q - X'_q - \frac{T''_{q0}}{T'_{q0}} \frac{X''_q}{X'_q} (X_q - X'_q)\right)i_q \right)$$
$$\frac{ de''_q}{dt} = \frac{1}{T''_{d0}} \left(-e''_q + e'_q - \left(X'_d - X''_d + \frac{T''_{d0}}{T'_{d0}} \frac{X''_d}{X'_d} (X_d - X'_d)\right)i_d + \frac{T_{AA}}{T'_{d0}} v_f \right)$$
$$\frac{ de''_d}{dt} = \frac{1}{T''_{q0}} \left(-e''_d + e'_d + \left(X'_q - X''_q + \frac{T''_{q0}}{T'_{q0}} \frac{X''_q}{X'_q} (X_q - X'_q)\right)i_q \right)$$**Algebraic equations**$$0 = v_q + R_a i_q - e''_q + (X''_d - X_l)i_d$$$$0 = v_d + R_a i_d - e''_d - (X''_q - X_l)i_q$$
$$0 = i_d v_d + i_q v_q - p_g$$
$$0 = i_d v_q - i_q v_d - q_g$$
"""

import numpy as np
import io

def descriptions():
    """
    Single source of truth for model parameters, inputs, states, and outputs.
    """
    descriptions_list = []
    
    # Parameters
    descriptions_list += [{"type": "Parameter", "tex":"S_n" ,       "data":"S_n",     "model":"S_n" ,     "default":100e6, "description":"Nominal power", "units":"VA"}]
    descriptions_list += [{"type": "Parameter", "tex":"F_n" ,       "data":"F_n",     "model":"F_n" ,     "default":50.0,  "description":"Nominal frequency", "units":"Hz"}]
    descriptions_list += [{"type": "Parameter", "tex":"H" ,         "data":"H",       "model":"H" ,       "default":5.0,   "description":"Inertia constant", "units":"s"}]
    descriptions_list += [{"type": "Parameter", "tex":"D" ,         "data":"D",       "model":"D" ,       "default":1.0,   "description":"Damping coefficient", "units":"s"}]
    descriptions_list += [{"type": "Parameter", "tex":"X_d",        "data":"X_d",     "model":"X_d" ,     "default":1.80,  "description":"d-axis synchronous reactance", "units":"pu-m"}]
    descriptions_list += [{"type": "Parameter", "tex":"X_q",        "data":"X_q",     "model":"X_q" ,     "default":1.70,  "description":"q-axis synchronous reactance", "units":"pu-m"}]
    descriptions_list += [{"type": "Parameter", "tex":"X'_d",       "data":"X1d",     "model":"X1d" ,     "default":0.30,  "description":"d-axis transient reactance", "units":"pu-m"}]
    descriptions_list += [{"type": "Parameter", "tex":"X'_q",       "data":"X1q",     "model":"X1q" ,     "default":0.55,  "description":"q-axis transient reactance", "units":"pu-m"}]
    descriptions_list += [{"type": "Parameter", "tex":"X''_d",      "data":"X2d",     "model":"X2d" ,     "default":0.20,  "description":"d-axis subtransient reactance", "units":"pu-m"}]
    descriptions_list += [{"type": "Parameter", "tex":"X''_q",      "data":"X2q",     "model":"X2q" ,     "default":0.25,  "description":"q-axis subtransient reactance", "units":"pu-m"}]
    descriptions_list += [{"type": "Parameter", "tex":"X_l",        "data":"X_l",     "model":"X_l" ,     "default":0.10,  "description":"Leakage reactance", "units":"pu-m"}]
    descriptions_list += [{"type": "Parameter", "tex":"T'_{d0}",    "data":"T1d0",    "model":"T1d0" ,    "default":8.0,   "description":"d-axis open circuit transient time constant", "units":"s"}]
    descriptions_list += [{"type": "Parameter", "tex":"T'_{q0}",    "data":"T1q0",    "model":"T1q0" ,    "default":0.4,   "description":"q-axis open circuit transient time constant", "units":"s"}]
    descriptions_list += [{"type": "Parameter", "tex":"T''_{d0}",   "data":"T2d0",    "model":"T2d0" ,    "default":0.03,  "description":"d-axis open circuit subtransient time constant", "units":"s"}]
    descriptions_list += [{"type": "Parameter", "tex":"T''_{q0}",   "data":"T2q0",    "model":"T2q0" ,    "default":0.05,  "description":"q-axis open circuit subtransient time constant", "units":"s"}]
    descriptions_list += [{"type": "Parameter", "tex":"T_{AA}",     "data":"T_AA",    "model":"T_AA" ,    "default":0.0,   "description":"d-axis additional time constant", "units":"s"}]
    descriptions_list += [{"type": "Parameter", "tex":"R_a" ,       "data":"R_a",     "model":"R_a" ,     "default":0.01,  "description":"Armature resistance", "units":"pu-m"}]
    
    # PSAT Saturation Parameters
    descriptions_list += [{"type": "Parameter", "tex":"S_{1.0}",    "data":"S_10",    "model":"S_10" ,    "default":0.0,   "description":"Saturation factor at 1.0 pu", "units":"-"}] 
    descriptions_list += [{"type": "Parameter", "tex":"S_{1.2}",    "data":"S_12",    "model":"S_12" ,    "default":0.0,   "description":"Saturation factor at 1.2 pu", "units":"-"}] 
    
    descriptions_list += [{"type": "Parameter", "tex":"K_{\\delta}","data":"K_delta", "model":"K_delta" , "default":0.0,   "description":"Reference machine constant", "units":"-"}] 
    descriptions_list += [{"type": "Parameter", "tex":"K_{sec}" ,   "data":"K_sec",   "model":"K_sec" ,   "default":0.0,   "description":"Secondary frequency control participation", "units":"-"}] 
    
    # Inputs
    descriptions_list += [{"type": "Input", "tex":"p_m",  "data":"p_m", "model":"p_m", "default":0.5, "description":"Mechanical power", "units":"pu-m"}]
    descriptions_list += [{"type": "Input", "tex":"v_f",  "data":"v_f", "model":"v_f", "default":1.0, "description":"Field voltage", "units":"pu-m"}]
    
    # Dynamic States
    descriptions_list += [{"type": "Dynamic State", "tex":"\\delta", "data":"", "model":"delta", "default":"", "description":"Rotor angle", "units":"rad"}]
    descriptions_list += [{"type": "Dynamic State", "tex":"\\omega", "data":"", "model":"omega", "default":"", "description":"Rotor speed", "units":"pu"}]
    descriptions_list += [{"type": "Dynamic State", "tex":"e'_q",    "data":"", "model":"e1q",   "default":"", "description":"q-axis transient voltage", "units":"pu-m"}]
    descriptions_list += [{"type": "Dynamic State", "tex":"e'_d",    "data":"", "model":"e1d",   "default":"", "description":"d-axis transient voltage", "units":"pu-m"}]
    descriptions_list += [{"type": "Dynamic State", "tex":"e''_q",   "data":"", "model":"e2q",   "default":"", "description":"q-axis subtransient voltage", "units":"pu-m"}]
    descriptions_list += [{"type": "Dynamic State", "tex":"e''_d",   "data":"", "model":"e2d",   "default":"", "description":"d-axis subtransient voltage", "units":"pu-m"}]
    
    # Algebraic States
    descriptions_list += [{"type": "Algebraic State", "tex":"i_d", "data":"", "model":"i_d", "default":"", "description":"d-axis current", "units":"pu-m"}]
    descriptions_list += [{"type": "Algebraic State", "tex":"i_q", "data":"", "model":"i_q", "default":"", "description":"q-axis current", "units":"pu-m"}]
    descriptions_list += [{"type": "Algebraic State", "tex":"p_g", "data":"", "model":"p_g", "default":"", "description":"Active power", "units":"pu-m"}]
    descriptions_list += [{"type": "Algebraic State", "tex":"q_g", "data":"", "model":"q_g", "default":"", "description":"Reactive power", "units":"pu-m"}]
    
    # Outputs
    descriptions_list += [{"type": "Output", "tex":"p_e", "data":"", "model":"p_e", "default":"", "description":"Electrical power", "units":"pu-m"}]
    descriptions_list += [{"type": "Output", "tex":"v_f", "data":"", "model":"v_f", "default":"", "description":"Field voltage", "units":"pu-m"}]
    descriptions_list += [{"type": "Output", "tex":"p_m", "data":"", "model":"p_m", "default":"", "description":"Mechanical power", "units":"pu-m"}]
    descriptions_list += [{"type": "Output", "tex":"S_{at}", "data":"", "model":"S_at", "default":"", "description":"Evaluated Saturation Factor", "units":"-"}]

    return descriptions_list 

def milano6ord(grid, name, bus_name, data_dict):
    backend = grid.backend
    sin = backend.sin
    cos = backend.cos

    # 1. Fetch metadata and defaults
    meta = descriptions()
    default_map = {item['data']: item['default'] for item in meta if 'data' in item and item['data']}

    # 2. Inputs
    V = backend.symbols(f"V_{bus_name}")
    theta = backend.symbols(f"theta_{bus_name}")
    p_m = backend.symbols(f"p_m_{name}")
    v_f = backend.symbols(f"v_f_{name}")
    omega_coi = backend.symbols("omega_coi")

    # 3. Dynamic states
    delta = backend.symbols(f"delta_{name}")
    omega = backend.symbols(f"omega_{name}")
    e1q = backend.symbols(f"e1q_{name}")
    e1d = backend.symbols(f"e1d_{name}")
    e2q = backend.symbols(f"e2q_{name}")
    e2d = backend.symbols(f"e2d_{name}")

    # 4. Algebraic states
    i_d = backend.symbols(f"i_d_{name}")
    i_q = backend.symbols(f"i_q_{name}")
    p_g = backend.symbols(f"p_g_{name}")
    q_g = backend.symbols(f"q_g_{name}")

    # 5. Parameters
    S_n = backend.symbols(f"S_n_{name}")
    Omega_b = backend.symbols(f"Omega_b_{name}")
    H = backend.symbols(f"H_{name}")
    T1d0 = backend.symbols(f"T1d0_{name}")
    T1q0 = backend.symbols(f"T1q0_{name}")
    T2d0 = backend.symbols(f"T2d0_{name}")
    T2q0 = backend.symbols(f"T2q0_{name}")
    T_AA = backend.symbols(f"T_AA_{name}")
    X_l = backend.symbols(f"X_l_{name}")
    X_d = backend.symbols(f"X_d_{name}")
    X_q = backend.symbols(f"X_q_{name}")
    X1d = backend.symbols(f"X1d_{name}")
    X1q = backend.symbols(f"X1q_{name}")
    X2d = backend.symbols(f"X2d_{name}")
    X2q = backend.symbols(f"X2q_{name}")
    D = backend.symbols(f"D_{name}")
    R_a = backend.symbols(f"R_a_{name}")
    K_delta = backend.symbols(f"K_delta_{name}")

    # 5.1 PSAT Saturation Processing
    S_10 = data_dict.get('S_10', default_map.get('S_10', 0.0))
    S_12 = data_dict.get('S_12', default_map.get('S_12', 0.0))

    if S_10 > 0.0 and S_12 > 0.0:
        R_val = np.sqrt(1.2 * S_12 / S_10)
        A_sat_val = (1.2 - R_val) / (1.0 - R_val)
        B_sat_val = S_10 / (1.0 - A_sat_val)**2
    else:
        A_sat_val = 0.8  # Default threshold
        B_sat_val = 0.0  # Zero ensures linear behavior

    A_sat = backend.symbols(f"A_sat_{name}")
    B_sat = backend.symbols(f"B_sat_{name}")
    
    params_list = ['S_n', 'H', 'T1d0', 'T1q0', 'T2d0', 'T2q0', 'T_AA', 'X_l', 
                   'X_d', 'X_q', 'X1d', 'X1q', 'X2d', 'X2q', 'D', 'R_a', 
                   'K_delta', 'K_sec']
    
    # 6. Auxiliar equations
    v_d = V*sin(delta - theta) 
    v_q = V*cos(delta - theta) 
    p_e = i_d*(v_d + R_a*i_d) + i_q*(v_q + R_a*i_q)     
    omega_s = omega_coi
    
    # PSAT Saturation Logic (Nonlinear)
    v_sat = backend.sqrt(e1q**2 + e1d**2)
    S_at = backend.Piecewise((B_sat * (v_sat - A_sat)**2 / v_sat, v_sat > A_sat), (0.0, True))
    S_d = S_at
    S_q = (X_q / X_d) * S_at
                
    # 7. Dynamic equations            
    ddelta = Omega_b*(omega - omega_s) - K_delta*delta
    domega = 1/(2*H)*(p_m - p_e - D*(omega - omega_s))
    de1q = (      -e1q*(1 + S_d) - (X_d - X1d - T2d0/T1d0*X2d/X1d * (X_d - X1d))*i_d + (1 - T_AA/T1d0)*v_f)/T1d0
    de1d = (      -e1d*(1 + S_q) + (X_q - X1q - T2q0/T1q0*X2q/X1q * (X_q - X1q))*i_q)/T1q0
    de2q = (-e2q + e1q - (X1d - X2d + T2d0/T1d0*X2d/X1d * (X_d - X1d))*i_d + T_AA/T1d0 * v_f)/T2d0
    de2d = (-e2d + e1d + (X1q - X2q + T2q0/T1q0*X2q/X1q * (X_q - X1q))*i_q)/T2q0

    # 8. Algebraic equations   
    g_i_d  = v_q + R_a*i_q - e2q + (X2d - X_l)*i_d
    g_i_q  = v_d + R_a*i_d - e2d - (X2q - X_l)*i_q
    g_p_g  = i_d*v_d + i_q*v_q - p_g  
    g_q_g  = i_d*v_q - i_q*v_d - q_g 
    
    # 9. Assembly 
    f_syn = [ddelta, domega, de1q, de1d, de2q, de2d]
    x_syn = [delta, omega, e1q, e1d, e2q, e2d]
    g_syn = [g_i_d, g_i_q, g_p_g, g_q_g]
    y_syn = [i_d, i_q, p_g, q_g]
    
    grid.H_total += H
    grid.omega_coi_numerator += omega*H*S_n
    grid.omega_coi_denominator += H*S_n

    grid.dae['f'] += f_syn
    grid.dae['x'] += x_syn
    grid.dae['g'] += g_syn
    grid.dae['y_ini'] += y_syn  
    grid.dae['y_run'] += y_syn  
    
    # 10. Dynamic Input Handling 
    val_v_f = data_dict.get('v_f', default_map.get('v_f', 1.0))
    grid.dae['u_ini_dict'].update({f'{v_f}': val_v_f})
    grid.dae['u_run_dict'].update({f'{v_f}': val_v_f})

    val_p_m = data_dict.get('p_m', default_map.get('p_m', 0.5))
    grid.dae['u_ini_dict'].update({f'{p_m}': val_p_m})
    grid.dae['u_run_dict'].update({f'{p_m}': val_p_m})

    # Initialization hints
    grid.dae['xy_0_dict'].update({str(omega): 1.0})
    grid.dae['xy_0_dict'].update({str(e1q): 1.0})
    grid.dae['xy_0_dict'].update({str(e1d): 0.0}) 
    grid.dae['xy_0_dict'].update({str(e2q): 1.0})
    grid.dae['xy_0_dict'].update({str(e2d): 0.0})
    grid.dae['xy_0_dict'].update({str(i_d): 0.5}) 
    grid.dae['xy_0_dict'].update({str(i_q): 0.5}) 
    
    # Outputs (Appending S_at for visibility)
    grid.dae['h_dict'].update({f"p_e_{name}": p_e})
    grid.dae['h_dict'].update({f"v_f_{name}": v_f})
    grid.dae['h_dict'].update({f"v_d_{name}": v_d})
    grid.dae['h_dict'].update({f"v_q_{name}": v_q})
    grid.dae['h_dict'].update({f"S_at_{name}": S_at})
    
    # 11. Dynamic Parameter Handling 
    F_n_val = data_dict.get('F_n', default_map.get('F_n', 50.0))
    grid.dae['params_dict'].update({f"Omega_b_{name}": 2 * np.pi * F_n_val})

    # Inject calculated Saturation Constants
    grid.dae['params_dict'].update({f"A_sat_{name}": A_sat_val})
    grid.dae['params_dict'].update({f"B_sat_{name}": B_sat_val})

    for item in params_list:
        val = data_dict.get(item, default_map.get(item, 0.0))       
        grid.dae['params_dict'].update({f"{item}_{name}": val})
    
    p_W   = p_g * S_n
    q_var = q_g * S_n

    return p_W, q_var

# =============================================================================
# Sphinx Documentation Auto-Generator
# =============================================================================
def dict_list_to_aligned_markdown_table(data: list[dict]) -> str:
    if not data or not isinstance(data, list): return ""
    dict_data = [item for item in data if isinstance(item, dict)]
    if not dict_data: return ""

    all_headers = []
    header_set = set()
    for row_dict in dict_data:
        for key in row_dict.keys():
            if key not in header_set:
                header_set.add(key)
                all_headers.append(key)

    max_widths = {header: len(header) for header in all_headers}
    for row_dict in dict_data:
        for header in all_headers:
            max_widths[header] = max(max_widths[header], len(str(row_dict.get(header, ""))))

    output = io.StringIO()
    padded_headers = [header.ljust(max_widths[header]) for header in all_headers]
    output.write("| " + " | ".join(padded_headers) + " |\n")
    separators = ['-' * max_widths[header] for header in all_headers]
    output.write("| " + " | ".join(separators) + " |\n")

    for row_dict in dict_data:
        padded_values = [str(row_dict.get(header, "")).ljust(max_widths[header]) for header in all_headers]
        output.write("| " + " | ".join(padded_values) + " |\n")

    return output.getvalue().strip()

def generate_sphinx_tables():
    docs = ""
    categories = ["Parameter", "Input", "Dynamic State", "Algebraic State", "Output"]
    full_list = descriptions()
    
    for cat in categories:
        cat_list = [{k: v for k, v in item.items() if k != 'type'} 
                    for item in full_list if item.get('type') == cat]
        
        if cat_list:
            docs += f"\n### {cat}s\n\n"
            docs += dict_list_to_aligned_markdown_table(cat_list)
            docs += "\n"
            
    return docs

__doc__ += generate_sphinx_tables()

# =============================================================================
# Testing Block
# =============================================================================
def test_build():
    from pydae.bps import BpsBuilder
    from pydae.builder.core import Builder
    import pytest

    grid = BpsBuilder('milano6ord.hjson')
    grid.checker()
    grid.uz_jacs = False
    grid.construct('temp_m6')
    bld = Builder(grid.sys_dict, target='ctypes')
    bld.build()
 
def test_run():
    import matplotlib.pyplot as plt
    from pydae.builder.model_class import Model
    
    model = Model('temp_m6')

    # Initialization
    model.ini({'p_m_1': 0.5, 'v_f_1': 1.0}, 'xy_0.json')

    # Run to steady state
    model.run(1.0, {})
    
    # Simulate a step change
    model.run(10.0, {'p_m_1': 1.0, 'v_f_1': 1.5})    

    model.post()

    fig, axes = plt.subplots(4, 1, figsize=(8, 10))
    
    axes[0].plot(model.Time, model.get_values('omega_1'), label='$\\omega$ (pu)', color='b')
    axes[0].set_ylabel('Speed')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(model.Time, model.get_values('e1q_1'), label="$e'_q$ (pu)", color='r')
    axes[1].plot(model.Time, model.get_values('e1d_1'), label="$e'_d$ (pu)", color='g')
    axes[1].set_ylabel('Transient V')
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(model.Time, model.get_values('e2q_1'), label="$e''_q$ (pu)", color='darkred', linestyle='--')
    axes[2].plot(model.Time, model.get_values('e2d_1'), label="$e''_d$ (pu)", color='darkgreen', linestyle='--')
    axes[2].set_ylabel('Subtransient V')
    axes[2].legend()
    axes[2].grid(True)

    # Plot the calculated saturation factor dynamically
    axes[3].plot(model.Time, model.get_values('S_at_1'), label="$S_{at}$ (pu)", color='purple')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Saturation Factor')
    axes[3].legend()
    axes[3].grid(True)
    
    fig.tight_layout()
    fig.savefig('milano6ord_saturation.svg')
    print("Test completed. Plot saved as 'milano6ord_saturation.svg'.")

if __name__ == '__main__':
    test_build()
    test_run()