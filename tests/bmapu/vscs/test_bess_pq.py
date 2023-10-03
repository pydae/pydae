# -*- coding: utf-8 -*-
"""
Created on october 2023

@author: jmmauricio
"""
import os
import importlib.util
import pytest
from pydae.bmapu import bmapu_builder
import importlib
import numpy as np

#@pytest.fixture
def test_build():
        

        P_bess = 10e6
        E_bess = 20e6

        data = {
        "system":{"name":"smib","S_base":100e6, "K_p_agc":0.0,"K_i_agc":0.0,"K_xif":1e-6},       
        "buses":[{"name":"1", "P_W":0.0,"Q_var":0.0,"U_kV":20.0}],
        "vscs":[{"type":"bess_pq","bus":"1","E_kWh":E_bess/1e3,"S_n":P_bess,"K_delta":0.01}],
        "sources":[{"type":"vsource","bus":"1"}]
        }
        
        os.chdir('./src/pydae/temp')
        grid = bmapu_builder.bmapu(data)
        grid.testing = True
        grid.build('temporal')

 
def test_discharge_charge():

    from pydae.temp import temporal    
    model = temporal.model()
    model.Dt = 1.0

    sigma_ref_ini = 0.5

    model.ini({'sigma_ref_1':sigma_ref_ini},'xy_0.json')
    model.run(0.5*3600,{'p_s_ref_1': 1.0}) # half an hour discharging
    model.run(1.0*3600,{'p_s_ref_1':-1.0}) # half an hour charging
    model.post()

    SoC = model.get_values('sigma_1')

    sigma_ini = SoC[0]
    sigma_lower = np.min(SoC)   
    sigma_end = SoC[-1]

    assert sigma_ini == sigma_ref_ini
    assert sigma_lower == pytest.approx(0.25, abs=0.01)
    assert sigma_end == pytest.approx(0.5, abs=0.01)
