# codegen/ctypes_builder.py
import os
import platform
import subprocess
import logging

# Reusing string generation functions from cffi_builder
from .cffi_builder import sym2c, sym2xyup, _get_c_funcs

def generate_and_compile_ctypes(builder_obj):
    """
    Generates C code and compiles it into a raw shared library (.so or .dll) 
    using a system compiler (GCC) for use with Python's standard ctypes module.
    """
    logging.info(f"[ctypes] Generating C code for system: {builder_obj.name}")
    all_defs, all_sources = "", ""
    
    # 1. Start C source string
    all_sources += "#include <math.h>\n\n"
    all_sources += "#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n"
    
    # Generate Dense functions
    for name, eq_list in [('f_ini_eval', builder_obj.f_ini_list), ('f_run_eval', builder_obj.f_run_list)]:
        d, s = _get_c_funcs(name, eq_list)
        all_defs += d; all_sources += s

    for name, eq_list in [('g_ini_eval', builder_obj.g_ini_list), ('g_run_eval', builder_obj.g_run_list)]:
        d, s = _get_c_funcs(name, eq_list)
        all_defs += d; all_sources += s

    for name, eq_list in [('h_eval', builder_obj.h_list)]:
        d, s = _get_c_funcs(name, eq_list)
        all_defs += d; all_sources += s

    # Generate Sparse Jacobians
    for name, eq_list in [('jac_ini_eval', builder_obj.jac_ini_list), 
                          ('jac_run_eval', builder_obj.jac_run_list), 
                          ('jac_trap_eval', builder_obj.jac_trap_list)]:
        d, s = _get_c_funcs(name, eq_list, is_sparse=True)
        all_defs += d; all_sources += s

    all_sources += "#ifdef __cplusplus\n}\n#endif\n\n"

    # --- PATH RESOLUTION ---
    
    # Get the absolute path of the directory where this file (ctypes_builder.py) is
    codegen_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go one level up to find 'daesolver_dense.c' in the builder directory
    builder_dir = os.path.dirname(codegen_dir)
    dae_dense_path = os.path.join(builder_dir, 'daesolver_dense.c')
    
    # Ensure the output directory exists
    output_dir = os.path.abspath(builder_obj.matrices_folder)
    os.makedirs(output_dir, exist_ok=True)

    # Define paths for the generated .c file and the final library
    c_file_path = os.path.join(output_dir, f"{builder_obj.name}_ctypes.c")
    
    is_windows = platform.system() == 'Windows'
    lib_ext = '.dll' if is_windows else '.so'
    lib_path = os.path.join(output_dir, f"{builder_obj.name}_ctypes{lib_ext}")

    # 2. Save the generated .c file
    with open(c_file_path, 'w') as f:
        f.write(all_sources)

    # 3. Verify static source file existence
    if not os.path.exists(dae_dense_path):
        logging.error(f"[ctypes] Source file NOT FOUND at: {dae_dense_path}")
        raise FileNotFoundError(f"Could not find daesolver_dense.c at {dae_dense_path}")

    # 4. Construct the GCC command
    # Using f'"{path}"' ensures spaces in folder names don't break the command
    compile_cmd = [
        'gcc', '-shared', '-O3',
        f'"{dae_dense_path}"',
        f'"{c_file_path}"',
        '-o', f'"{lib_path}"'
    ]
    
    if not is_windows:
        compile_cmd.insert(3, '-fPIC') # Position Independent Code for Linux/macOS
        compile_cmd.append('-lm')      # Link math library
        
    cmd_str = ' '.join(compile_cmd)
    logging.info(f"[ctypes] Invoking compiler from: {os.getcwd()}")
    logging.info(f"[ctypes] Command: {cmd_str}")
    
    # 5. Execute Compilation
    try:
        # Use shell=True to help Windows find GCC in the PATH
        subprocess.run(cmd_str, check=True, shell=True)
        logging.info(f"[ctypes] Success! Library saved: {lib_path}")
    except FileNotFoundError:
        logging.error("[ctypes] GCC not found. Please add 'gcc' to your system PATH.")
    except subprocess.CalledProcessError as e:
        logging.error(f"[ctypes] Compilation failed: {e}")