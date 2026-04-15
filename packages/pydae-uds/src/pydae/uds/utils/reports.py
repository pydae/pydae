import numpy as np
import json
import os
import hjson

class style():    
    RED = '\033[31m'
    GREEN = '\033[32m'
    BLUE = '\033[34m'
    RESET = '\033[0m'

# test1 = "ONE"
# test2 = "TWO"


# print(f"{style.RED}{test1} {style.BLUE}{test2}{style.RESET}")

def report_v(grid,data_input,show=True,model='urisi'):
    '''
    

    Parameters
    ----------
    grid : pydae object
        Pydae object modelling a grid.
    data_input : if dict, a dictionary with the grid parameters
                 if string, the path to the .json file containing grid parameters

    show : string, optional
        If report is print or not.
        
    model : string, optional
        Type of implemented model. The default is 'urisi'.

    Returns
    -------
    dict with the results.

    Example
    -------

    '''
    
    
    if type(data_input) == dict:
        data = data_input
        
    if type(data_input) == str:
        if os.path.splitext(data_input)[1] == '.json':
            with open(data_input,'r') as fobj:
                data = json.loads(fobj.read().replace("'",'"'))
        if os.path.splitext(data_input)[1] == '.hjson':
            with open(data_input,'r') as fobj:
                data = hjson.loads(fobj.read().replace("'",'"'))

    buses_dict = {}

    
    for bus in data['buses']:
        if not "acdc" in bus:
            bus.update({'acdc':'AC'})
        
        if bus['acdc'] == 'AC':
            if f"V_{bus['name']}_3_r" in grid.y_ini_list:
                v_n_r,v_n_i = grid.get_mvalue([f"V_{bus['name']}_3_r",f"V_{bus['name']}_3_i"])
                v_n_g = v_n_r + 1j*v_n_i
            else:
                v_n_g = 0.0
            #for ph in ['a','b','c']:
            
            ph = 0
            v_a_r,v_a_i =  grid.get_mvalue([f"V_{bus['name']}_{ph}_r",f"V_{bus['name']}_{ph}_i"])
            v_a_g = v_a_r + 1j*v_a_i
            v_a_n = v_a_g - v_n_g
            v_a_m = np.abs(v_a_n)
            v_a_a = np.rad2deg(np.angle(v_a_n))

            ph = 1
            v_b_r,v_b_i =  grid.get_mvalue([f"V_{bus['name']}_{ph}_r",f"V_{bus['name']}_{ph}_i"])
            v_b_g = v_b_r + 1j*v_b_i
            v_b_n = v_b_g - v_n_g
            v_b_m = np.abs(v_b_n)
            v_b_a = np.rad2deg(np.angle(v_b_n))

            ph = 2
            v_c_r,v_c_i =  grid.get_mvalue([f"V_{bus['name']}_{ph}_r",f"V_{bus['name']}_{ph}_i"])
            v_c_g = v_c_r + 1j*v_c_i
            v_c_n = v_c_g - v_n_g
            v_c_m = np.abs(v_c_n)
            v_c_a = np.rad2deg(np.angle(v_c_n))
    
            alpha = alpha = np.exp(2.0/3*np.pi*1j)
            v_0 =  1/3*(v_a_g+v_b_g+v_c_g)
            v_1 = 1.0/3.0*(v_a_g + v_b_g*alpha + v_c_g*alpha**2)
            v_2 = 1.0/3.0*(v_a_g + v_b_g*alpha**2 + v_c_g*alpha)
            
            # compute unbalanced as in Kersting 1ers edition pg. 266
            v_m_array = [v_a_m,v_b_m,v_c_m]
            v_m_min = np.min(v_m_array)
            v_m_max = np.max(v_m_array)
            max_dev = v_m_max - v_m_min
            v_avg = np.sum(v_m_array)/3
            unbalance = max_dev/v_avg
            
            bus[f"v_{bus['name']}_{'a'}n"] = v_a_m
            
            if show:
                print(f"V_{bus['name']}_{'a'}n: {v_a_m:7.1f}| {v_a_a:6.1f}º V,    V_{bus['name']}_{'a'}g: {np.abs(v_a_g):7.1f}| {np.angle(v_a_g,deg=True):6.1f}º V,    V_1 = {np.abs(v_1):7.1f} V, unb = {unbalance*100:3.2f}%")
                print(f"V_{bus['name']}_{'b'}n: {v_b_m:7.1f}| {v_b_a:6.1f}º V,    V_{bus['name']}_{'b'}g: {np.abs(v_b_g):7.1f}| {np.angle(v_b_g,deg=True):6.1f}º V,    V_2 = {np.abs(v_2):7.1f} V")
                print(f"V_{bus['name']}_{'c'}n: {v_c_m:7.1f}| {v_c_a:6.1f}º V,    V_{bus['name']}_{'c'}g: {np.abs(v_c_g):7.1f}| {np.angle(v_c_g,deg=True):6.1f}º V,    v_0 = {np.abs(v_0):7.1f} V")                   
                
                print(f"  V_{bus['name']}_ng: {np.abs(v_n_g):8.1f}| {np.angle(v_n_g,deg=True):8.1f}º V")
            
            buses_dict[bus['name']] = {f"v_{bus['name']}_{'a'}n":v_b_m,f"v_{bus['name']}_{'b'}n":v_b_m,f"v_{bus['name']}_{'c'}n":v_c_m}
            buses_dict[bus['name']].update({f'v_unb':unbalance,'v_ng':np.abs(v_n_g)})


        if bus['acdc'] == 'DC':

            
            ph = 0
            v_pos_r,v_neg_r =  grid.get_mvalue([f"V_{bus['name']}_0_r",f"V_{bus['name']}_1_r"])
            v_dc = v_pos_r - v_neg_r
    
            
            if show:
                print(f"V_{bus['name']}_dc: {v_dc:7.1f} V, V_{bus['name']}_pos_g: {v_pos_r:7.1f} V, V_{bus['name']}_neg_g: {v_neg_r:7.1f} V")

            # buses_dict[bus['name']] = {f"v_{bus['name']}_{'a'}n":v_b_m,f"v_{bus['name']}_{'b'}n":v_b_m,f"v_{bus['name']}_{'c'}n":v_c_m}
            # buses_dict[bus['name']].update({f'v_unb':unbalance,'v_ng':np.abs(v_n_g)})

    return buses_dict  