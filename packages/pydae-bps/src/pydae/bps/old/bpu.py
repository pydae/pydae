# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 23:52:55 2021

@author: jmmau
"""

def bal_pu(data_input):
    '''
    

    Parameters
    ----------
    data_input : string or dict
        File path to the system data information or dictionar y with the information.

    Returns
    -------
    dict
        Dictionary with the equations for pydae. 
        
    {
     'sys':{'name':'pf_1','S_base':100e6},       
     'buses':[{'name':'GRI','P_W':0.0,'Q_var':0.0,'U_kV':66.0, 'type':'slack'},
              {'name':'POI','P_W':0.0,'Q_var':0.0,'U_kV':66.0},
              {'name':'PMV','P_W':0.0,'Q_var':0.0,'U_kV':20.0}],
     'lines':[{'bus_j':'GRI','bus_k':'POI','X_km':0.4,'R_km':0.12,'km':20},
              {'bus_j':'POI','bus_k':'PMV','X_pu':0.04,'R_pu':0.01, 'S_mva':50.0}]
    }
        

    '''

    if type(data_input) == str:
        with open(file_path,'r') as fobj:
            data = json.loads(fobj.read().replace("'",'"'))
    elif type(data_input) == dict:
        data = data_input

    sys = data['sys']
    buses = data['buses']
    lines = data['lines']

    params_grid = {'S_base':sys['S_base']}
    S_base = sym.Symbol("S_base", real=True) 
    N_bus = len(buses)
    N_branch = len(lines)
    A = sym.zeros(3*N_branch,N_bus)
    G_primitive = sym.zeros(3*N_branch,3*N_branch)
    B_primitive = sym.zeros(3*N_branch,3*N_branch)
    buses_list = [bus['name'] for bus in buses]
    it = 0
    for line in lines:

        bus_j = line['bus_j']
        bus_k = line['bus_k']

        idx_j = buses_list.index(bus_j)
        idx_k = buses_list.index(bus_k)    

        A[3*it,idx_j] = 1
        A[3*it,idx_k] =-1   
        A[3*it+1,idx_j] = 1
        A[3*it+2,idx_k] = 1   
        
        line_name = f"{bus_j}_{bus_k}"
        g_jk = sym.Symbol(f"g_{line_name}", real=True) 
        b_jk = sym.Symbol(f"b_{line_name}", real=True) 
        bs_jk = sym.Symbol(f"bs_{line_name}", real=True) 
        G_primitive[3*it,3*it] = g_jk
        B_primitive[3*it,3*it] = b_jk
        B_primitive[3*it+1,3*it+1] = bs_jk/2
        B_primitive[3*it+2,3*it+2] = bs_jk/2
        
        if 'X_pu' in line:
            if 'S_mva' in line: S_line = 1e6*line['S_mva']
            R = line['R_pu']*sys['S_base']/S_line  # in pu of the system base
            X = line['X_pu']*sys['S_base']/S_line  # in pu of the system base
            G =  R/(R**2+X**2)
            B = -X/(R**2+X**2)
            params_grid.update({f"g_{line_name}":G})
            params_grid.update({f'b_{line_name}':B})

        if 'X' in line:
            bus_idx = buses_list.index(line['bus_j'])
            U_base = buses[bus_idx]['U_kV']
            Z_base = U_base**2/sys['S_base']
            R = line['R']/Z_base  # in pu of the system base
            X = line['X']/Z_base  # in pu of the system base
            G =  R/(R**2+X**2)
            B = -X/(R**2+X**2)
            params_grid.update({f"g_{line_name}":G})
            params_grid.update({f'b_{line_name}':B})

        if 'X_km' in line:
            bus_idx = buses_list.index(line['bus_j'])
            U_base = buses[bus_idx]['U_kV']*1000
            Z_base = U_base**2/sys['S_base']
            R = line['R_km']*line['km']/Z_base  # in pu of the system base
            X = line['X_km']*line['km']/Z_base  # in pu of the system base
            G =  R/(R**2+X**2)
            B = -X/(R**2+X**2)
            params_grid.update({f"g_{line_name}":G})
            params_grid.update({f'b_{line_name}':B})        

        if 'Bs_pu' in line:
            if 'S_mva' in line: S_line = 1e6*line['S_mva']
            Bs = line['Bs_pu']*S_line/sys['S_base']  # in pu of the system base
            bs = -Bs/2.0
            params_grid.update({f'bs_{line_name}':bs})
 
        if 'Bs_km' in line:
            bus_idx = buses_list.index(line['bus_j'])
            U_base = buses[bus_idx]['U_kV']*1000
            Z_base = U_base**2/sys['S_base']
            Y_base = 1.0/Z_base
            Bs = line['Bs_km']*line['km']/Y_base # in pu of the system base
            bs = Bs 
            params_grid.update({f'bs_{line_name}':bs})
            
        it += 1


    G = A.T * G_primitive * A
    B = A.T * B_primitive * A    
    
    sin = sym.sin
    cos = sym.cos
    y_grid = []
    g = sym.zeros(2*N_bus,1)
    u_grid = {}
    h_grid = {}
    for j in range(N_bus):
        bus_j_name = buses_list[j]
        P_j = sym.Symbol(f"P_{bus_j_name}", real=True)
        Q_j = sym.Symbol(f"Q_{bus_j_name}", real=True)
        g[2*j]   = -P_j/S_base
        g[2*j+1] = -Q_j/S_base
        for k in range(N_bus): 

            bus_k_name = buses_list[k]
            V_j = sym.Symbol(f"V_{bus_j_name}", real=True) 
            V_k = sym.Symbol(f"V_{bus_k_name}", real=True) 
            theta_j = sym.Symbol(f"theta_{bus_j_name}", real=True) 
            theta_k = sym.Symbol(f"theta_{bus_k_name}", real=True) 
            g[2*j]   += V_j*V_k*(G[j,k]*cos(theta_j - theta_k) + B[j,k]*sin(theta_j - theta_k)) 
            g[2*j+1] += V_j*V_k*(G[j,k]*sin(theta_j - theta_k) - B[j,k]*cos(theta_j - theta_k))        
            h_grid.update({f"V_{bus_j_name}":V_j})
        bus = buses[j]
        bus_name = bus['name']
        if 'type' in bus:
            if bus['type'] == 'slack':
                y_grid += [P_j]
                y_grid += [Q_j]
                u_grid.update({f"V_{bus_name}":1.0})
                u_grid.update({f"theta_{bus_name}":0.0})  
        else:
            y_grid += [V_j]
            y_grid += [theta_j]        
            u_grid.update({f"P_{bus_name}":bus['P_W']})
            u_grid.update({f"Q_{bus_name}":bus['Q_var']})    
            
        params_grid.update({f'U_{bus_name}_n':bus['U_kV']*1000})
    g_grid = list(g)     

    if False:
        v_sym_list = []
        for bus in buses_list:
            V_m = sym.Symbol(f'V_{bus}',real=True)
            V_a = sym.Symbol(f'theta_{bus}',real=True)
            v_sym_list += [V_m*sym.exp(sym.I*V_a)]

        sym.Matrix(v_sym_list)

        I_lines = (G_primitive+1j*B_primitive) * A * sym.Matrix(v_sym_list)

        it = 0
        for line in lines:
            I_jk_r = sym.Symbol(f"I_{line['bus_j']}_{line['bus_k']}_r", real=True)
            I_jk_i = sym.Symbol(f"I_{line['bus_j']}_{line['bus_k']}_i", real=True)
            g_grid += [-I_jk_r + sym.re(I_lines[it])]
            g_grid += [-I_jk_i + sym.im(I_lines[it])]
            y_grid += [I_jk_r]
            y_grid += [I_jk_i]
            it += 1
            
    return {'g':g_grid,'y':y_grid,'u':u_grid,'h':h_grid, 
            'params':params_grid, 'data':data, 'A':A, 'B_primitive':B_primitive}