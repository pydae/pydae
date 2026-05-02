# pydae/core/builder/sympy_builder.py
"""
The Builder orchestrates the full pipeline:
  1. Parse & validate the symbolic system dictionary
  2. Compute symbolic Jacobians
  3. Translate to C code
  4. Compile into a shared library (.so / .dll) or CFFI extension

Usage
=====
  # Dense, CFFI (default)
  Builder(sys_dict)

  # Sparse with KLU, ctypes
  Builder(sys_dict, target='ctypes', sparse='klu')

  # Sparse with Apple Accelerate, CFFI
  Builder(sys_dict, target='cffi', sparse='accelerate')

  # Sparse with PARDISO, ctypes
  Builder(sys_dict, target='ctypes', sparse='pardiso')

  # Legacy: sparse=True is equivalent to sparse='klu'
  Builder(sys_dict, sparse=True)
"""

import json
import logging
import os

from pydae.core.common.parser import check_system, process_system_dict
from pydae.core.common.symbolic import build_large_jacobians, compute_base_jacobians
from sympy import Expr, Symbol

# Canonical names for the sparse backends
VALID_SPARSE_BACKENDS = {'klu', 'pardiso', 'accelerate'}


class SympyEncoder(json.JSONEncoder):
    """JSON encoder that handles SymPy Symbol/Expr objects."""
    def default(self, obj):
        if isinstance(obj, (Symbol, Expr)):
            return str(obj)
        return super().default(obj)


class Builder:
    def __init__(self, system_dict, verbose=False, API=False, target='cffi', sparse=True):
        """
        Parameters
        ----------
        system_dict : dict
            Symbolic system definition.
        target : str
            Compilation backend: ``'cffi'`` (default) or ``'ctypes'``.
        sparse : bool or str
            * ``False`` / ``None`` → dense solver
            * ``True`` / ``'klu'`` → SuiteSparse KLU (0-based CSC)
            * ``'pardiso'``        → Intel MKL PARDISO (1-based CSR)
            * ``'accelerate'``     → Apple Accelerate (0-based CSC, macOS)
        """
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

        # ------------------------------------------------------------------
        # Normalise the sparse setting
        # ------------------------------------------------------------------
        if sparse is True:
            self.sparse = 'klu'
        elif isinstance(sparse, str) and sparse.lower() in VALID_SPARSE_BACKENDS:
            self.sparse = sparse.lower()
        elif sparse in (False, None):
            self.sparse = False
        else:
            raise ValueError(
                f"Invalid sparse backend '{sparse}'. "
                f"Choose from: False, True, {VALID_SPARSE_BACKENDS}"
            )

        if 'alpha_solver' not in self.raw_sys['params_dict']:
            self.raw_sys['params_dict'].update({'alpha_solver': 0.5})

        # Initialize ALL lists (Dynamic, Algebraic, Outputs, Jacobians)
        self.f_ini_list, self.f_run_list = [], []
        self.g_ini_list, self.g_run_list = [], []
        self.h_list = []

        self.jac_ini_list, self.jac_run_list, self.jac_trap_list = [], [], []
        # Initialize UZ Jacobian lists (for Small Signal Analysis)
        self.Fu_ini_list = []
        self.Fu_run_list = []
        self.Gu_ini_list = []
        self.Gu_run_list = []
        self.Hx_list = []
        self.Hy_ini_list = []
        self.Hy_run_list = []
        self.Hu_ini_list = []
        self.Hu_run_list = []

        # Build output folder
        self.matrices_folder = 'build'
        if not os.path.exists(self.matrices_folder):
            os.makedirs(self.matrices_folder)

    def build(self):
        """The main orchestration pipeline."""
        logging.info(f"Starting build pipeline for {self.name} "
                     f"(target={self.target}, sparse={self.sparse})...")

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

        # ------------------------------------------------------------------
        # Write sparse backend metadata so Model class can size buffers
        # ------------------------------------------------------------------
        self.system_dict_to_json['sparse_backend'] = self.sparse if self.sparse else None
        self.system_dict_to_json['target'] = self.target

        # Create dictionaries for the code generator with the symbolic equations
        self.f_ini_list = [{'sym': eq} for eq in self.sys['f']]
        self.f_run_list = [{'sym': eq} for eq in self.sys['f']]
        self.g_ini_list = [{'sym': eq} for eq in self.sys['g']]
        self.g_run_list = [{'sym': eq} for eq in self.sys['g']]
        self.h_list     = [{'sym': eq} for eq in self.sys['h']]

        # --- Symbolic Math Phase ---
        self.sys = compute_base_jacobians(self.sys, uz_jacs=self.uz_jacs)
        build_large_jacobians(self)

        # ------------------------------------------------------------------
        # NNZ is known only after Jacobians are built — write it now
        # ------------------------------------------------------------------
        if self.sparse:
            _, Ai_ini, Ap_ini = self.jac_ini_sp[:3]
            _, Ai_trap, Ap_trap = self.jac_trap_sp[:3]
            self.system_dict_to_json['NNZ_ini'] = len(self.jac_ini_list)
            self.system_dict_to_json['NNZ_trap'] = len(self.jac_trap_list)
            # Store both sparsity patterns for diagnostics
            self.system_dict_to_json['Ap_ini'] = [int(v) for v in Ap_ini]
            self.system_dict_to_json['Ai_ini'] = [int(v) for v in Ai_ini]
            self.system_dict_to_json['Ap_trap'] = [int(v) for v in Ap_trap]
            self.system_dict_to_json['Ai_trap'] = [int(v) for v in Ai_trap]
        else:
            self.system_dict_to_json['NNZ_ini'] = 0
            self.system_dict_to_json['NNZ_trap'] = 0

        # Write the JSON metadata file
        with open(f"{self.name}_data.json", "w") as fobj:
            json.dump(self.system_dict_to_json, fobj, cls=SympyEncoder, indent=4)

        # --- Translation & Code Generation Phase ---
        if self.target == 'cffi':
            self._build_cffi()
        elif self.target == 'ctypes':
            self._build_ctypes()
        else:
            raise ValueError(f"Target '{self.target}' is not supported. Use 'cffi' or 'ctypes'.")

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

        # UZ Jacobians (need both ini and run patterns to cover all y/u vars)
        if getattr(self, 'uz_jacs', False):
            uz_lists = [self.Fu_ini_list, self.Fu_run_list, self.Gu_ini_list,
                        self.Gu_run_list, self.Hx_list, self.Hy_ini_list,
                        self.Hy_run_list, self.Hu_ini_list, self.Hu_run_list]
            for uz_list in uz_lists:
                sym2c_fn(uz_list)

            # Apply both ini and run patterns so y_ini, y_run, u_ini, u_run all get replaced
            for uz_list in uz_lists:
                sym2xyup_fn(self.sys, uz_list, 'ini')
                sym2xyup_fn(self.sys, uz_list, 'run')

        logging.info("End translating symbolic equations to C strings.")

    def _build_cffi(self):
        from pydae.core.builder.codegen.cffi_builder import generate_and_compile_cffi, sym2c, sym2xyup
        self._translate_all(sym2c, sym2xyup)
        generate_and_compile_cffi(self)

    def _build_ctypes(self):
        from pydae.core.builder.codegen.cffi_builder import sym2c, sym2xyup
        from pydae.core.builder.codegen.ctypes_builder import generate_and_compile_ctypes
        self._translate_all(sym2c, sym2xyup)
        generate_and_compile_ctypes(self)
