# pydae/core/builder/core.py
"""
The Builder orchestrates the full pipeline:
  1. Parse & validate the symbolic system dictionary
  2. Compute symbolic Jacobians
  3. Translate to C code
  4. Compile into a shared library (.so / .dll)
"""

from pydae.core.builder.parser import process_system_dict, check_system
from pydae.core.builder.symbolic import compute_base_jacobians, build_large_jacobians

import sympy as sym
import numpy as np
import os
import logging
import json
from sympy import Symbol, Expr


class SympyEncoder(json.JSONEncoder):
    """JSON encoder that handles SymPy Symbol/Expr objects."""
    def default(self, obj):
        if isinstance(obj, (Symbol, Expr)):
            return str(obj)
        return super().default(obj)


class Builder:   
    def __init__(self, system_dict, verbose=False, API=False, target='cffi'):
        logging.basicConfig(
            format='%(asctime)s %(message)s', 
            level=logging.INFO if not verbose else logging.DEBUG
        )
        
        self.verbose = verbose
        self.raw_sys = system_dict
        self.name = self.raw_sys.get('name', 'unknown_system')
        self.target = target.lower()
        self.API = API
        self.save_sources = True
        self.uz_jacs = True 
        
        # Initialize ALL lists (Dynamic, Algebraic, Outputs, Jacobians)
        self.f_ini_list, self.f_run_list = [], []
        self.g_ini_list, self.g_run_list = [], []
        self.h_list = []
        
        self.jac_ini_list, self.jac_run_list, self.jac_trap_list = [], [], []
        self.Fu_list, self.Gu_list, self.Hx_list = [], [], []
        
        # Build output folder
        self.matrices_folder = 'build'
        if not os.path.exists(self.matrices_folder):
            os.makedirs(self.matrices_folder)

    def build(self):
        """The main orchestration pipeline."""
        logging.info(f"Starting build pipeline for {self.name} (Target: {self.target})...")
        
        # --- Parsing Phase ---
        self.sys, self.inirun = check_system(self.raw_sys)
        self.sys = process_system_dict(self.sys)

        # Save system metadata to JSON
        self.system_dict_to_json = {}
        for item in ['x_list', 'y_ini_list', 'y_run_list', 'h_dict']:
            item_name = item if item != 'h_dict' else 'h_list'
            self.system_dict_to_json[item_name] = [str(item2) for item2 in self.raw_sys[item]]

        for item in ['u_ini_dict', 'u_run_dict', 'params_dict']:
            self.system_dict_to_json[item] = {
                str(k): float(v) for k, v in self.raw_sys[item].items()
            }
            
        with open("system_data.json", "w") as fobj:
            json.dump(self.system_dict_to_json, fobj, cls=SympyEncoder, indent=4)

        # Create dictionaries for the code generator with the symbolic equations
        self.f_ini_list = [{'sym': eq} for eq in self.sys['f']]
        self.f_run_list = [{'sym': eq} for eq in self.sys['f']]
        self.g_ini_list = [{'sym': eq} for eq in self.sys['g']]
        self.g_run_list = [{'sym': eq} for eq in self.sys['g']]
        self.h_list     = [{'sym': eq} for eq in self.sys['h']]
        
        # --- Symbolic Math Phase ---
        self.sys = compute_base_jacobians(self.sys, self.inirun)
        build_large_jacobians(self) 
        
        # --- Translation & Code Generation Phase ---
        if self.target == 'cffi':
            self._build_cffi()
        elif self.target == 'ctypes':
            self._build_ctypes()
        else:
            raise ValueError(f"Target '{self.target}' is not supported yet.")
            
        logging.info("Build pipeline completed successfully!")

    def _translate_all(self, sym2c_fn, sym2xyup_fn):
        """Common translation step for both backends."""
        logging.info("Translating symbolic equations to C strings...")
        
        for eq_list in [self.f_ini_list, self.f_run_list, self.g_ini_list, 
                        self.g_run_list, self.h_list]:
            sym2c_fn(eq_list)
        
        sym2xyup_fn(self.sys, self.f_ini_list, 'ini')
        sym2xyup_fn(self.sys, self.f_run_list, 'run')
        sym2xyup_fn(self.sys, self.g_ini_list, 'ini')
        sym2xyup_fn(self.sys, self.g_run_list, 'run')
        sym2xyup_fn(self.sys, self.h_list, 'run')
        
        for jac_list in [self.jac_ini_list, self.jac_run_list, self.jac_trap_list]:
            sym2c_fn(jac_list)
        
        sym2xyup_fn(self.sys, self.jac_ini_list, 'ini')
        sym2xyup_fn(self.sys, self.jac_run_list, 'run')
        sym2xyup_fn(self.sys, self.jac_trap_list, 'run')

    def _build_cffi(self):
        from pydae.core.builder.codegen.cffi_builder import sym2c, sym2xyup, generate_and_compile_cffi
        self._translate_all(sym2c, sym2xyup)
        generate_and_compile_cffi(self)

    def _build_ctypes(self):
        from pydae.core.builder.codegen.ctypes_builder import sym2c, sym2xyup, generate_and_compile_ctypes
        self._translate_all(sym2c, sym2xyup)
        generate_and_compile_ctypes(self)
