# -*- coding: utf-8 -*-
"""
Automated Pytest Suite for pydae Milano Synchronous Machine Models.
Tests milano2ord, milano3ord, milano4ord, and milano6ord for steady-state coherence.
"""

import os
import json
import pytest
from pydae.bps import BpsBuilder
from pydae.builder.core import Builder
from pydae.builder.model_class import Model

# Parameterize the test to run identically for all 4 model types
@pytest.mark.parametrize("model_type", [
    "milano2ord", 
    "milano3ord", 
    "milano4ord", 
    "milano6ord"
])
def test_milano_steady_state(model_type):
    """
    Builds a 2-bus system, initializes the specific Milano model, 
    runs a steady-state simulation, and asserts physical validity.
    """
    sys_name = f"test_{model_type}"
    file_name = f"{sys_name}.json"
    
    # 1. Define the Universal Parameter Superset
    # The models will safely pick only what they need from this dictionary.
    sys_dict = {
        "system": {"name": sys_name, "S_base": 100e6, "K_p_agc": 0.0, "K_i_agc": 0.0, "K_xif": 0.01},
        "buses": [
            {"name": "1", "P_W": 0.0, "Q_var": 0.0, "U_kV": 20.0},
            {"name": "2", "P_W": 0.0, "Q_var": 0.0, "U_kV": 20.0}
        ],
        "lines": [
            {"bus_j": "1", "bus_k": "2", "X_pu": 0.05, "R_pu": 0.01, "Bs_pu": 1e-6, "S_mva": 200}
        ],
        "shunts": [],
        "syns": [
            {
                "bus": "1",
                "type": model_type,
                "S_n": 200e6,
                "F_n": 50.0,
                "X_d": 1.8, "X_q": 1.7, "X_l": 0.2,
                "X1d": 0.3, "X1q": 0.55, 
                "X2d": 0.2, "X2q": 0.25,
                "T1d0": 8.0, "T1q0": 0.4,
                "T2d0": 0.03, "T2q0": 0.05,
                "T_AA": 0.0,
                "R_a": 0.01,
                "H": 5.0, "D": 0.0,
                "S_10": 0.0, "S_12": 0.0, # Linear model for basic testing
                "K_delta": 0.0, "K_sec": 0.0
            }
        ],
        "sources": [{"type": "vsource", "bus": "2"}]
    }

    # Write dictionary to a temporary JSON file for BpsBuilder ingestion
    with open(file_name, 'w') as f:
        json.dump(sys_dict, f, indent=4)

    try:
        # 2. Build and Compile the ctypes Model
        grid = BpsBuilder(file_name)
        grid.checker()
        grid.uz_jacs = False
        grid.construct(sys_name)
        
        bld = Builder(grid.sys_dict, target='ctypes')
        bld.build()

        # 3. Initialize and Run the Simulation
        model = Model(sys_name)
        
        # Handle the specific input requirements for the 2nd order vs higher orders
        if model_type == "milano2ord":
            # 2nd order uses constant transient voltage as input
            inputs = {'p_m_1': 0.5, 'e1q_1': 1.2} 
        else:
            # 3rd, 4th, 6th order use field voltage as input
            inputs = {'p_m_1': 0.5, 'v_f_1': 1.2}

        # Initialize and run to t = 1.0 seconds
        model.ini(inputs, 'xy_0.json')
        model.run(1.0, {})
        model.post()

        # 4. Extract Variables for Assertion
        omega = model.get_value('omega_1')
        v_gen = model.get_value('V_1')
        p_e   = model.get_value('p_e_1')
        p_g   = model.get_value('p_g_1')
        q_g   = model.get_value('q_g_1')

        # 5. Assertions
        
        # Rotor speed must be exactly 1.0 pu in steady state
        assert omega == pytest.approx(1.0, abs=1e-4), f"{model_type} failed: omega = {omega} != 1.0"
        
        # Generator terminal voltage should be close to 1.0 pu (within +/- 10%)
        assert 0.9 <= v_gen <= 1.1, f"{model_type} failed: V_1 = {v_gen} is out of safe bounds"
        
        # Electrical power output should match mechanical power input (0.5 pu) minus stator losses
        # Using a tolerance of 5% (0.025 pu) to account for R_a losses
        assert p_e == pytest.approx(0.5, abs=0.025), f"{model_type} failed: p_e_1 = {p_e} != 0.5"
        
        # Active power injection to the grid should closely match electrical power
        assert p_g == pytest.approx(p_e, abs=1e-1), f"{model_type} failed: p_g_1 doesn't match p_e_1"
        
        # Reactive power should be physically reasonable (e.g., between -1.0 and +1.0 pu)
        assert -1.0 <= q_g <= 1.0, f"{model_type} failed: q_g_1 = {q_g} is out of safe bounds"

    finally:
        # Cleanup: Remove the temporary JSON file so the working directory stays clean
        if os.path.exists(file_name):
            os.remove(file_name)