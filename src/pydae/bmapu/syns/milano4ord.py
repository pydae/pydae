"""
Created on Thu August 10 23:52:55 2022

@author: jmmauricio
"""

import numpy as np
import sympy as sym

def milano4ord(grid,name,bus_name,data_dict):
    r"""
    Synchronous machine model of order 4 from Federico Milano book.

    """

    sin = sym.sin
    cos = sym.cos  

    # inputs
    V = sym.Symbol(f"V_{bus_name}", real=True)
    theta = sym.Symbol(f"theta_{bus_name}", real=True)
    p_m = sym.Symbol(f"p_m_{name}", real=True)
    v_f = sym.Symbol(f"v_f_{name}", real=True)  
    omega_coi = sym.Symbol("omega_coi", real=True)   
        
    # dynamic states
    delta = sym.Symbol(f"delta_{name}", real=True)
    omega = sym.Symbol(f"omega_{name}", real=True)
    e1q = sym.Symbol(f"e1q_{name}", real=True)
    e1d = sym.Symbol(f"e1d_{name}", real=True)

    # algebraic states
    i_d = sym.Symbol(f"i_d_{name}", real=True)
    i_q = sym.Symbol(f"i_q_{name}", real=True)            
    p_g = sym.Symbol(f"p_g_{name}", real=True)
    q_g = sym.Symbol(f"q_g_{name}", real=True)

    # parameters
    S_n = sym.Symbol(f"S_n_{name}", real=True)
    Omega_b = sym.Symbol(f"Omega_b_{name}", real=True)            
    H = sym.Symbol(f"H_{name}", real=True)
    T1d0 = sym.Symbol(f"T1d0_{name}", real=True)
    T1q0 = sym.Symbol(f"T1q0_{name}", real=True)
    X_d = sym.Symbol(f"X_d_{name}", real=True)
    X_q = sym.Symbol(f"X_q_{name}", real=True)
    X1d = sym.Symbol(f"X1d_{name}", real=True)
    X1q = sym.Symbol(f"X1q_{name}", real=True)
    D = sym.Symbol(f"D_{name}", real=True)
    R_a = sym.Symbol(f"R_a_{name}", real=True)
    K_delta = sym.Symbol(f"K_delta_{name}", real=True)
    K_sat = sym.Symbol(f"K_sat_{name}", real=True)
    params_list = ['S_n','H','T1d0','T1q0','X_d','X_q','X1d','X1q','D','R_a','K_delta','K_sec']
    
    # auxiliar
    v_d = V*sin(delta - theta) 
    v_q = V*cos(delta - theta) 
    p_e = i_d*(v_d + R_a*i_d) + i_q*(v_q + R_a*i_q)     
    omega_s = omega_coi
                
    # dynamic equations            
    ddelta = Omega_b*(omega - omega_s) - K_delta*delta
    domega = 1/(2*H)*(p_m - p_e - D*(omega - omega_s))
    de1q = 1/T1d0*(-e1q*K_sat - (X_d - X1d)*i_d + v_f)
    de1d = 1/T1q0*(-e1d       + (X_q - X1q)*i_q)

    # algebraic equations   
    g_i_d  = e1q - R_a*i_q - X1d*i_d - v_q
    g_i_q  = e1d - R_a*i_d + X1q*i_q - v_d 
    g_p_g  = i_d*v_d + i_q*v_q - p_g  
    g_q_g  = i_d*v_q - i_q*v_d - q_g 
    
    # dae 
    f_syn = [ddelta,domega,de1q,de1d]
    x_syn = [ delta, omega, e1q, e1d]
    g_syn = [g_i_d,g_i_q,g_p_g,g_q_g]
    y_syn = [  i_d,  i_q,  p_g,  q_g]
    
    grid.H_total += H
    grid.omega_coi_numerator += omega*H*S_n
    grid.omega_coi_denominator += H*S_n

    grid.dae['f'] += f_syn
    grid.dae['x'] += x_syn
    grid.dae['g'] += g_syn
    grid.dae['y_ini'] += y_syn  
    grid.dae['y_run'] += y_syn  
    
    if 'v_f' in data_dict:
        grid.dae['u_ini_dict'].update({f'{v_f}':{data_dict['v_f']}})
        grid.dae['u_run_dict'].update({f'{v_f}':{data_dict['v_f']}})
    else:
        grid.dae['u_ini_dict'].update({f'{v_f}':1.0})
        grid.dae['u_run_dict'].update({f'{v_f}':1.0})

    if 'p_m' in data_dict:
        grid.dae['u_ini_dict'].update({f'{p_m}':{data_dict['p_m']}})
        grid.dae['u_run_dict'].update({f'{p_m}':{data_dict['p_m']}})
    else:
        grid.dae['u_ini_dict'].update({f'{p_m}':1.0})
        grid.dae['u_run_dict'].update({f'{p_m}':1.0})

    grid.dae['xy_0_dict'].update({str(omega):1.0})
    grid.dae['xy_0_dict'].update({str(e1q):1.0})
    grid.dae['xy_0_dict'].update({str(i_q):1.0})

    
    # outputs
    grid.dae['h_dict'].update({f"p_e_{name}":p_e})
    grid.dae['h_dict'].update({f"v_f_{name}":v_f})
    grid.dae['h_dict'].update({f"p_m_{name}":p_m})
    
    if 'F_n' in data_dict:
        grid.dae['params_dict'].update({f"Omega_b_{name}":2*np.pi*data_dict['F_n']})
    else:
        grid.dae['params_dict'].update({f"Omega_b_{name}":data_dict['Omega_b']})

    for item in params_list:       
        grid.dae['params_dict'].update({f"{item}_{name}":data_dict[item]})
    
    grid.dae['params_dict'].update({f"K_sat_{name}":1.0})

    # if 'avr' in syn_data:
    #     add_avr(grid.dae,syn_data)
    #     grid.dae['u_ini_dict'].pop(str(v_f))
    #     grid.dae['u_run_dict'].pop(str(v_f))
    #     grid.dae['xy_0_dict'].update({str(v_f):1.5})

    # if 'gov' in syn_data:
    #     add_gov(grid.dae,syn_data)  
    #     grid.dae['u_ini_dict'].pop(str(p_m))
    #     grid.dae['u_run_dict'].pop(str(p_m))
    #     grid.dae['xy_0_dict'].update({str(p_m):0.5})

    # if 'pss' in syn_data:
    #     add_pss(grid.dae,syn_data)  

    p_W   = p_g * S_n
    q_var = q_g * S_n

    return p_W,q_var

def descriptions():
    """
    | Constant    | Data       | pydae            | Default   | Description                                  |  Units  |
    | :---------- | :----------| :----------      |       ---:| :------------------------------------------- |:-------:|  
    | $S_n$       | ``S_n``    | ``S_n``     | 20e6      | Nominal power                                | VA      |
    | $H$         | ``H``      | ``H``       | 5.0       | Inertia constaant                            | s       |
    | $D$         | ``D``      | ``D``       | 1.0       | Damping coefficient                          | s       |
    | $X_q$       | ``X_q``    | ``X_q``     | 1.70      | q-axis synchronous reactance                 | pu-m    |
    | $X'_q$      | ``X1q``    | ``X1q``     | 0.55      | q-axis transient reactance                   | pu-m    |
    | $T'_{q0}$   | ``T1q0``   | ``T1q0``    | 0.40      | q-axis open circuit transient time constant  | s       |
    | $X_d$       | ``X_d``    | ``X_d``     | 1.8       | d-axis synchronous reactance                 | pu-m    | 
    | $X'_d$      | ``X1d``    | ``X1d``     | 0.3       | d-axis transient reactance                   | pu-m    |
    | $T'_{d0}$   | ``T1d0``   | ``T1d0``    | 8.0       | d-axis open circuit transient time constant  | s       |   
    | $R_a$       | ``R_a``    | ``R_a``     | 0.01      | Armature resistance                          | pu-m    |    
    | $K_{\delta}$| ``K_delta``| ``K_delta`` | 0.0       | Reference machine constant                   | -       | 
    | $K_{sec}$   | ``K_sec``  | ``K_sec``   | 1.0       | Secondary frequency control participation    | -       | 
    """

    descriptions_list = []
 
    descriptions_list += [{"tex":"S_n" , "data":"S_n", "model":"S_n" , "default":20e6, "description":"Nominal power","units":"VA"}]
    descriptions_list += [{"tex":"H" , "data":"H", "model":"H" , "default":5.0 , "description":"Inertia constant ","units":"s"}]
    descriptions_list += [{"tex":"D" , "data":"D", "model":"D" , "default":1.0 , "description":"Damping coefficient","units":" s "}]
    descriptions_list += [{"tex":"X_q" , "data":"X_q", "model":"X_q" , "default":1.70, "description":"q-axis synchronous reactance ","units":" pu-m"}]
    descriptions_list += [{"tex":"X'_q", "data":"X1q", "model":"X1q" , "default":0.55, "description":"q-axis transient reactance ","units":" pu-m"}]
    descriptions_list += [{"tex":"T'_{q0}" , "data":"T1q0" , "model":"T1q0", "default":0.40, "description":"q-axis open circuit transient time constant","units":" s "}]
    descriptions_list += [{"tex":"X_d" , "data":"X_d", "model":"X_d" , "default":1.8 , "description":"d-axis synchronous reactance ","units":" pu-m"}] 
    descriptions_list += [{"tex":"X'_d", "data":"X1d", "model":"X1d" , "default":0.3 , "description":"d-axis transient reactance ","units":" pu-m"}]
    descriptions_list += [{"tex":"T'_{d0}" , "data":"T1d0" , "model":"T1d0", "default":8.0 , "description":"d-axis open circuit transient time constant","units":" s "}] 
    descriptions_list += [{"tex":"R_a" , "data":"R_a", "model":"R_a" , "default":0.01, "description":"Armature resistance","units":" pu-m"}]
    descriptions_list += [{"tex":"K_{\delta}", "data":"K_delta", "model":"K_delta" , "default":0.0 , "description":"Reference machine constant ","units":" - "}] 
    descriptions_list += [{"tex":"K_{sec}" , "data":"K_sec", "model":"K_sec" , "default":1.0 , "description":"Secondary frequency control participation","units":" - "}] 
        

    return descriptions_list 

def sym_devs():

    name = '_'

    v_d = sym.Symbol(f"v_d_{name}", real=True)
    v_q = sym.Symbol(f"v_q_{name}", real=True)  

    # dynamic states
    e1q = sym.Symbol(f"e1q_{name}", real=True)
    e1d = sym.Symbol(f"e1d_{name}", real=True)

    # algebraic states
    i_d = sym.Symbol(f"i_d_{name}", real=True)
    i_q = sym.Symbol(f"i_q_{name}", real=True)            


    # parameters
    X1d = sym.Symbol(f"X1d_{name}", real=True)
    X1q = sym.Symbol(f"X1q_{name}", real=True)
    R_a = sym.Symbol(f"R_a_{name}", real=True)

    g_i_d  = v_q + R_a*i_q + X1d*i_d - e1q
    g_i_q  = v_d + R_a*i_d - X1q*i_q - e1d

    res = sym.solve([g_i_d,g_i_q],[i_d,i_q])

    print(res)


def test():
    import numpy as np
    import sympy as sym
    import hjson
    from pydae.bmapu.bmapu_builder import bmapu
    import pydae.build_cffi as db
    import pytest

    grid = bmapu('milano4ord.hjson')
    grid.checker()
    grid.uz_jacs = True
    grid.build('temp')

    import temp

    model = temp.model()

    v_ref_1 = 1.05
    model.ini({'p_m_1':0.5,'v_ref_1':v_ref_1},'xy_0.json')

    # assert model.get_value('V_1') == pytest.approx(v_ref_1, rel=0.001)

    model.run(1.0,{})
    model.run(1.1,{'b_shunt_1':-500})
    #model.run(2.0,{'b_shunt_1':0})

    model.post()
    # # assert model.get_value('q_A2') == pytest.approx(-q_ref, rel=0.05)

    # model.ini({'p_m_1':0.5,'v_ref_1':1.0},'xy_0.json')
    # model.run(1.0,{})
    # model.run(15.0,{'v_ref_1':1.05})
    # model.post()

    import matplotlib.pyplot as plt

    fig,axes = plt.subplots()
    axes.plot(model.Time,model.get_values('V_1'))
    fig.savefig('milano4ord_fault.svg')

import io # Used for robust string building

def dict_list_to_aligned_markdown_table(data: list[dict], preserve_key_order: bool = True) -> str:
    """
    Converts a list of dictionaries into a Markdown formatted table
    with columns visually aligned using padding in the raw Markdown source.

    Args:
        data: A list of dictionaries. Keys can vary between dictionaries.
              All unique keys across all dictionaries will be used as columns.
        preserve_key_order: If True (default), attempts to preserve the key
                            order based on first appearance across all dicts
                            (requires Python 3.7+ for dict key order).
                            If False, sorts keys alphabetically.

    Returns:
        A string containing the Markdown formatted table with aligned columns.
        Returns an empty string if the input list is empty, contains no
        dictionaries, or if all dictionaries are empty.

    Raises:
        ValueError: If the input data is not a list or contains items that are not dictionaries.

    Notes:
        - All values are converted to strings using str().
        - Special characters like '|' within cell values are not escaped, which
          might break table rendering in some Markdown processors.
        - Determines column widths based on the longest header name or value
          string length in each respective column across all rows.
        - Collects all unique keys from all dictionaries to form the header.
        - Uses left-alignment padding ('ljust') for cell content.
        - Separator line is padded with '-' characters to match column width.
    """
    if not data:
        return ""

    if not isinstance(data, list):
        raise ValueError("Input must be a list.")

    # Filter out any non-dictionary items and check if any dicts remain
    dict_data = [item for item in data if isinstance(item, dict)]
    if not dict_data:
         # Could raise error, or return empty if list had only non-dicts
         # Let's check if *any* item wasn't a dict first
        if any(not isinstance(item, dict) for item in data):
            raise ValueError("All items in the list must be dictionaries.")
        else: # Original list was non-empty but contained only empty dicts or similar
            return "" # Or handle based on specific needs for lists of only empty dicts

    # --- Collect all unique headers ---
    # Use a list to maintain order if preserving, plus a set for quick lookup.
    all_headers = []
    header_set = set()
    for row_dict in dict_data:
        for key in row_dict.keys():
            if key not in header_set:
                header_set.add(key)
                if key == 'tex':
                    all_headers.append(key)

    if not preserve_key_order:
        all_headers.sort()

    if not all_headers: # Handle case where all dictionaries were empty
        return ""

    # --- Calculate maximum width for each column ---
    # Initialize widths with header lengths
    max_widths = {header: len(header) for header in all_headers}

    # Update widths based on data values
    for row_dict in dict_data:
        for header in all_headers:
            # Get value, default to empty string if key is missing in this dict
            value_str = str(row_dict.get(header, ""))
            # Update the max width for this column if current value is longer
            max_widths[header] = max(max_widths[header], len(value_str))

    # --- Build the table string ---
    output = io.StringIO()

    # Format Header Row
    # Pad each header to its max width and join
    padded_headers = [header.ljust(max_widths[header]) for header in all_headers]
    output.write("| " + " | ".join(padded_headers) + " |\n")

    # Format Separator Row
    # Pad with dashes '-' to match the column width for visual alignment
    # Note: Markdown only requires '---', but padding enhances raw readability.
    separators = ['-' * max_widths[header] for header in all_headers]
    output.write("| " + " | ".join(separators) + " |\n")

    # Format Data Rows
    for row_dict in dict_data:
        padded_values = []
        for header in all_headers:
            value_str = str(row_dict.get(header, ""))
            # Pad each value to its column's max width
            padded_value = value_str.ljust(max_widths[header])
            padded_values.append(padded_value)
        output.write("| " + " | ".join(padded_values) + " |\n")

    return output.getvalue().strip() # Get the full string and remove trailing newline



if __name__ == '__main__':

    #sym_devs()
    #test()

    print(dict_list_to_aligned_markdown_table(descriptions()))