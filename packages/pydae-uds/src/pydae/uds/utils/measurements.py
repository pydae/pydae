import numpy as np
import json

def get_v(grid,bus_name,type='V_an_m'):
    '''
    Return desired voltages values as defined by argument type.    

    Parameters
    ----------
    grid : pydae object
        Pydae object modelling a grid.

    bus_name : string
        Name of the bus.

    type : string, optional
        Definition of the required voltage.
        V_an_m -> Phase to neutral voltage module for phase a
        V_an -> Phase to neutral voltage complex value for phase a
        U_ab_m -> Phase a to b voltage module

    Returns
    -------
    Real or complex value, also can be a numpy array, depending on the type argument

    Example
    -------

    '''
    
    a2n = {'a':0,'b':1,'c':2,'n':3}

    if type in ['V_an_m','V_bn_m','V_cn_m','V_an','V_bn','V_cn','V_ng','V_ng_m']:
            
        v_n_r,v_n_i =  grid.get_mvalue([f"V_{bus_name}_3_r",f"V_{bus_name}_3_i"])
        v_n_g = v_n_r + 1j*v_n_i

        if type == 'V_ng':  # neutral to ground voltage
            return v_n_g

        if type == 'V_ng_m':
            return np.abs(v_n_g)
        
        ph = a2n[type[2]]
        v_ph_r,v_ph_i =  grid.get_mvalue([f"V_{bus_name}_{ph}_r",f"V_{bus_name}_{ph}_i"])
        v_ph_g = v_ph_r + 1j*v_ph_i

        if type[-1] == 'm':
            return np.abs(v_ph_g - v_n_g)
        else:
            return v_ph_g - v_n_g

    if type in ['V_abcn_m','V_abcn']:
        v_list = []
        v_n_r,v_n_i =  grid.get_mvalue([f"V_{bus_name}_3_r",f"V_{bus_name}_3_i"])
        v_n_g = v_n_r + 1j*v_n_i

        for it in range(3):

            v_ph_r,v_ph_i =  grid.get_mvalue([f"V_{bus_name}_{it}_r",f"V_{bus_name}_{it}_i"])
            v_ph_g = v_ph_r + 1j*v_ph_i

            if type[-1] == 'm':
                v_list += [np.abs(v_ph_g - v_n_g)]
            else:
                v_list += [v_ph_g - v_n_g]
            
        return np.array(v_list)

    if type in ['U_ab_m','U_bc_m','U_ca_m','U_ba_m','U_cb_m','U_ac_m']:
        ph_1,ph_2 = a2n[type[2]],a2n[type[3]]
        v_1_r,v_1_i =  grid.get_mvalue([f"V_{bus_name}_{ph_1}_r",f"V_{bus_name}_{ph_1}_i"])
        v_2_r,v_2_i =  grid.get_mvalue([f"V_{bus_name}_{ph_2}_r",f"V_{bus_name}_{ph_2}_i"])
        return np.abs((v_1_r - v_2_r) + 1j*(v_1_i - v_2_i))
            
    
def get_i(grid,bus_j,bus_k,type='I_a_m'):
    '''
    Return desired currents values as defined by argument type.    

    Parameters
    ----------
    grid : pydae object
        Pydae object modelling a grid.

    bus_name : string
        Name of the bus.

    type : string, optional
        Definition of the required voltage.
        I_a_m -> Phase current module

    Returns
    -------
    Real or complex value, also can be a numpy array, depending on the type argument

    Example
    -------

    '''
    
    a2n = {'a':0,'b':1,'c':2,'n':3}

    if type in ['I_a_m','I_b_m','I_c_m','I_n_m','I_a','I_b','I_c','I_n']:
        
        ph = a2n[type[2]]
        i_r,i_i =  grid.get_mvalue([f"i_l_{bus_j}_{ph}_{bus_k}_{ph}_r",f"i_l_{bus_j}_{ph}_{bus_k}_{ph}_i"])
        i = i_r + 1j*i_i

        if type[-1] == 'm':
            return np.abs(i)
        else:
            return i
        

def get_power(grid,bus_j,bus_k,type='S_a'): 
    '''
    Return desired complex powers.    

    Parameters
    ----------
    grid : pydae object
        Pydae object modelling a grid.

    bus_name : string
        Name of the bus.

    type : string, optional
        Definition of the required voltage.
        S_a_m -> Phase aparent power

    Returns
    -------
    Real or complex value, also can be a numpy array, depending on the type argument

    Example
    -------

    get_power(model,'B2','B3',type='S_c').real 

    '''
    if type in ['S_a_m','S_b_m','S_c_m','S_a','S_b','S_c']:

        ph = type[2]
        V_phn = get_v(grid,bus_j,type=f'V_{ph}n')
        I_ph = get_i(grid,bus_j,bus_k,type=f'I_{ph}')
        S_ph = V_phn*np.conjugate(I_ph)

        if type[-1] == 'm':
            return np.abs(S_ph)
        else:
            return S_ph
        
    if type in ['S_m','S']:
        S_ph = 0.0

        for ph in ['a','b','c']:
            V_phn = get_v(grid,bus_j,type=f'V_{ph}n')
            I_ph = get_i(grid,bus_j,bus_k,type=f'I_{ph}')
            S_ph += V_phn*np.conjugate(I_ph)

        if type[-1] == 'm':
            return np.abs(S_ph)
        else:
            return S_ph