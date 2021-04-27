#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2020

@author: jmmauricio
"""

import numpy as np
import json


def save(json_file,dictionary, sort_keys=False, indent=4):
    '''
    Convert dictionary to json and save it in json_file path.

    Parameters
    ----------
    json_file : path
         Path of the json data file..
    dictionary : dict
        Dictionary with the loaded data as a dictionary.

    Returns
    -------
    None.

    '''
    
    data_json = json.dumps(dictionary, sort_keys=False, indent=4)    
    with open(json_file, 'w') as fobj:
        fobj.write(data_json)

    

    
def load(json_file):
    '''
    Read json file and convert it to dictionary.

    Parameters
    ----------
    json_file : path
        Path of the json data file.

    Returns
    -------
    data_dictionary: dict
        Dictionary with the loaded data as a dictionary.

    '''
    
    with open(json_file, 'r') as fobj:
        json_data = fobj.read()
        
    json_data = json_data.replace("'",'"')
    data_dictionary = json.loads(json_data)        
    
    return data_dictionary



def get_v(syst,bus_name,phase_name='',v_type='rms_phph',dq_name='DQ'):
    
    if dq_name == 'DQ':
        D_ = 'D'
        Q_ = 'Q'    

        v_d_name = f'v_{bus_name}_{D_}'
        v_q_name = f'v_{bus_name}_{Q_}'

        v_d = syst.get_value(v_d_name)
        v_q = syst.get_value(v_q_name)

        v_m = 0.0
        if v_type == 'rms_phph':
            v_m = np.abs(v_d+1j*v_q)*np.sqrt(3/2)
            return v_m

        if v_type == 'dq_cplx':
            return v_q+1j*v_d

    if dq_name == 'ri':

        v_r_name = f'v_{bus_name}_{phase_name}_r'
        v_i_name = f'v_{bus_name}_{phase_name}_i'

        v_r = syst.get_value(v_r_name)
        v_i = syst.get_value(v_i_name)

        v_m = 0.0
        if v_type == 'rms_phph':
            v_m = np.abs(v_r+1j*v_i)*np.sqrt(3/2)
            return v_m

        if v_type == 'phasor':
            return v_r+1j*v_i


    
def get_i(syst,bus_name,phase_name='',i_type='rms',dq_name='DQ'):

    if dq_name == 'DQ':
        D_ = 'D'
        Q_ = 'Q'    

        i_d_name = f'i_{bus_name}_{D_}'
        i_q_name = f'i_{bus_name}_{Q_}'

        i_d = syst.get_value(i_d_name)
        i_q = syst.get_value(i_q_name)

        i_m = 0.0
        if i_type == 'rms_phph':
            i_m = np.abs(i_d+1j*i_q)*np.sqrt(1/2)
            return i_m

        if i_type == 'dq_cplx':
            return i_q+1j*i_d

    if dq_name == 'ri':

        i_r_name = f'i_{bus_name}_{phase_name}_r'
        i_i_name = f'i_{bus_name}_{phase_name}_i'

        i_r = syst.get_value(i_r_name)
        i_i = syst.get_value(i_i_name)

        i_m = 0.0
        if i_type == 'rms_phph':
            v_m = np.abs(i_r+1j*i_i)*np.sqrt(3/2)
            return v_m

        if i_type == 'phasor':
            return i_r+1j*i_i

    if dq_name == 'r':

        i_r_name = f'i_{bus_name}_{phase_name}_r'
        i_r = syst.get_value(i_r_name)

        if i_type == 'phasor':
            return i_r
        
        
def get_s(syst,bus_name,s_type='cplx'):
    v_dq = get_v(syst,bus_name,v_type='dq_cplx')
    i_dq = get_i(syst,bus_name,i_type='dq_cplx')

    if s_type == 'cplx':
        s = 3/2*v_dq*(i_dq)
        return s


if __name__ == "__main__":
    
    class sys_class():
        
        def __init__(self): 
     
            self.A = np.array([[ 0.00000000e+00,  3.14159265e+02,  0.00000000e+00,
                 0.00000000e+00,  0.00000000e+00],
               [-4.43027144e-01, -1.42857143e-01, -1.44867645e-01,
                 1.69999669e-01,  0.00000000e+00],
               [-1.80455968e-01,  0.00000000e+00, -6.64265905e-01,
                -2.31113959e-03, -2.50000000e+01],
               [ 1.32845727e+00,  0.00000000e+00,  1.35913375e-02,
                -2.58565604e+00,  0.00000000e+00],
               [-3.90435895e-01,  0.00000000e+00,  2.69427382e+00,
                 4.81098835e-01, -2.00000000e+01]])
            
    sys = sys_class()
    print(damp_report(sys))
    
    

    