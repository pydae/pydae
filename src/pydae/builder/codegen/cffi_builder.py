# codegen/cffi_builder.py
import os
import re
import logging
import sysconfig
import sympy as sym
from cffi import FFI

def sym2c(full_list):
    """Converts a list of SymPy expressions into C code strings."""
    for item in full_list:
        item.update({'ccode': sym.ccode(item['sym'])})

def sym2xyup(sys, full_list, inirun):
    """Replaces variable names in C code strings with C array index syntax."""
    for item in full_list:
        string = item['ccode']
        for it, x in enumerate(sys['x_list']): string = re.sub(rf'\b{x}\b', f'x[{it}]', string)
        y_list = sys['y_ini_list'] if inirun == 'ini' else sys['y_run_list']
        for it, y in enumerate(y_list): string = re.sub(rf'\b{y}\b', f'y[{it}]', string)
        u_dict = sys['u_ini_dict'] if inirun == 'ini' else sys['u_run_dict']
        for it, u in enumerate(u_dict.keys()): string = re.sub(rf'\b{u}\b', f'u[{it}]', string)
        for it, p in enumerate(sys.get('params_dict', {}).keys()): string = re.sub(rf'\b{p}\b', f'p[{it}]', string)
        item.update({'xyup': string})

def _get_c_funcs(func_name, eq_list, is_sparse=False):
    """Helper to generate C function definitions and bodies."""
    def_str = f'void {func_name}(double *data, double *x, double *y, double *u, double *p, double Dt);\n'
    source_str = f'void {func_name}(double *data, double *x, double *y, double *u, double *p, double Dt){{\n\n'
    for it, item in enumerate(eq_list):
        idx = item['de_idx'] if is_sparse else it
        source_str += f"    data[{idx}] = {item['xyup']};\n"
    source_str += '\n}\n\n'
    return def_str, source_str

def generate_and_compile_cffi(builder_obj):
    """Generates C code and compiles it using Python's CFFI."""
    logging.info(f"[CFFI] Generating C code for system: {builder_obj.name}")
    all_defs, all_sources = "", ""
    
    # Generate Dense functions
    for name, eq_list in [('f_ini_eval', builder_obj.f_ini_list), ('f_run_eval', builder_obj.f_run_list)]:
        d, s = _get_c_funcs(name, eq_list)
        all_defs += d; all_sources += s

    # Generate Sparse Jacobians
    for name, eq_list in [('jac_ini_eval', builder_obj.jac_ini_list), ('jac_run_eval', builder_obj.jac_run_list), ('jac_trap_eval', builder_obj.jac_trap_list)]:
        d, s = _get_c_funcs(name, eq_list, is_sparse=True)
        all_defs += d; all_sources += s

    # Compile with CFFI
    ffibuilder = FFI()
    ffibuilder.cdef(all_defs)
    libraries = [] if os.name == 'nt' else ["m"]

    ffibuilder.set_source(f"{builder_obj.name}_cffi", all_sources, libraries=libraries)
    
    ffibuilder.compile(tmpdir=builder_obj.matrices_folder, verbose=builder_obj.verbose)
    logging.info("[CFFI] Compilation finished successfully!")