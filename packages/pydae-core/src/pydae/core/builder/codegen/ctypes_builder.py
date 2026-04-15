# codegen/ctypes_builder.py
import os
import platform
import subprocess
import logging

# Reusing string generation functions from cffi_builder
from pydae.core.builder.codegen.cffi_builder import sym2c, sym2xyup, _get_c_funcs

# ---------------------------------------------------------------------------
# Supported sparse solver backends (value used as the C preprocessor define)
# ---------------------------------------------------------------------------
SPARSE_BACKENDS = {
    'klu':        'USE_SPARSE',       # SuiteSparse KLU  (0-based CSC)
    'pardiso':    'USE_PARDISO',      # Intel MKL PARDISO (1-based CSR)
    'accelerate': 'USE_ACCELERATE',   # Apple Accelerate  (0-based CSC, long* Ap)
}


def generate_and_compile_ctypes(builder_obj):
    """
    Generates C code and compiles it into a raw shared library 
    (.dll for Windows, .so for Linux, .dylib for macOS) 
    using a system compiler (GCC/Clang) for use with ctypes.
    
    Sparse backend selection
    ========================
    ``builder_obj.sparse`` can be:
      * False / None  – dense mode (no sparse solver)
      * True / 'klu'  – SuiteSparse KLU  (original behaviour)
      * 'pardiso'     – Intel MKL PARDISO
      * 'accelerate'  – Apple Accelerate Sparse Solvers (macOS only)
    """
    logging.info(f"[ctypes] Generating C code for system: {builder_obj.name}")
    all_defs, all_sources = "", ""
    
    # ------------------------------------------------------------------
    # Normalise the sparse setting into (is_sparse: bool, backend: str)
    # ------------------------------------------------------------------
    raw_sparse = getattr(builder_obj, 'sparse', False)
    if raw_sparse is True:
        # Legacy behaviour: True means KLU
        is_sparse = True
        sparse_backend = 'klu'
    elif isinstance(raw_sparse, str) and raw_sparse.lower() in SPARSE_BACKENDS:
        is_sparse = True
        sparse_backend = raw_sparse.lower()
    else:
        is_sparse = False
        sparse_backend = None

    # ------------------------------------------------------------------
    # OS Detection
    # ------------------------------------------------------------------
    sys_os = platform.system()
    is_windows = sys_os == 'Windows'
    is_mac = sys_os == 'Darwin'
    is_linux = sys_os == 'Linux'

    # Validate backend / platform combinations
    if sparse_backend == 'accelerate' and not is_mac:
        raise RuntimeError(
            "The 'accelerate' sparse backend is only available on macOS. "
            f"Current platform: {sys_os}"
        )
    if sparse_backend == 'pardiso' and is_mac:
        logging.warning(
            "[ctypes] MKL PARDISO on macOS requires Intel MKL to be installed. "
            "Consider using 'accelerate' or 'klu' instead."
        )

    # 1. Start C source string
    all_sources += "#include <math.h>\n\n"
    all_sources += "#ifdef __cplusplus\nextern \"C\" {\n#endif\n\n"
    
    # 2. Inject Sparse Definitions and Constants
    if is_sparse:
        c_define = SPARSE_BACKENDS[sparse_backend]
        all_sources += f"#define {c_define} 1\n\n"
        
        # Separate sparsity patterns for ini and trap
        _, Ai_ini, Ap_ini = builder_obj.jac_ini_sp[:3]
        _, Ai_trap, Ap_trap = builder_obj.jac_trap_sp[:3]
        
        nnz_ini = len(builder_obj.jac_ini_list)
        nnz_trap = len(builder_obj.jac_trap_list)

        def _emit_arrays(ap, ai, nnz, suffix, backend):
            src = ""
            if backend == 'accelerate':
                ap_str = ", ".join(map(str, ap))
                src += f"long Ap_{suffix}[] = {{{ap_str}}};\n"
            elif backend == 'pardiso':
                ap_str = ", ".join(str(v + 1) for v in ap)
                src += f"int Ap_{suffix}[] = {{{ap_str}}};\n"
            else:
                ap_str = ", ".join(map(str, ap))
                src += f"int Ap_{suffix}[] = {{{ap_str}}};\n"
            
            if backend == 'pardiso':
                ai_str = ", ".join(str(v + 1) for v in ai)
            else:
                ai_str = ", ".join(map(str, ai))
            src += f"int Ai_{suffix}[] = {{{ai_str}}};\n"
            src += f"const int NNZ_{suffix} = {nnz};\n\n"
            return src

        all_sources += _emit_arrays(Ap_ini, Ai_ini, nnz_ini, 'ini', sparse_backend)
        all_sources += _emit_arrays(Ap_trap, Ai_trap, nnz_trap, 'trap', sparse_backend)
    
    # 3. Generate Standard Dense functions (f, g, h)
    for name, eq_list in [('f_ini_eval', builder_obj.f_ini_list), ('f_run_eval', builder_obj.f_run_list)]:
        d, s = _get_c_funcs(name, eq_list, is_sparse=False)
        all_defs += d; all_sources += s

    for name, eq_list in [('g_ini_eval', builder_obj.g_ini_list), ('g_run_eval', builder_obj.g_run_list)]:
        d, s = _get_c_funcs(name, eq_list, is_sparse=False)
        all_defs += d; all_sources += s

    for name, eq_list in [('h_eval', builder_obj.h_list)]:
        d, s = _get_c_funcs(name, eq_list, is_sparse=False)
        all_defs += d; all_sources += s

    # 4. Generate Jacobian functions (Aware of Sparse vs Dense mode)
    for name, eq_list in [('jac_ini_eval', builder_obj.jac_ini_list), ('jac_trap_eval', builder_obj.jac_trap_list)]:
        d, s = _get_c_funcs(name, eq_list, is_sparse=is_sparse)
        all_defs += d; all_sources += s

    all_sources += "\n#ifdef __cplusplus\n}\n#endif\n"

    # 5. File Paths
    if is_windows:
        lib_ext = '.dll'
    elif is_mac:
        lib_ext = '.dylib'
    else:
        lib_ext = '.so'

    output_dir = os.path.abspath('build')
    os.makedirs(output_dir, exist_ok=True)
    c_file_path = os.path.join(output_dir, f"temp_ctypes_{builder_obj.name}.c")
    lib_path = os.path.join(output_dir, f"{builder_obj.name}_ctypes{lib_ext}")
    
    dae_dense_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'daesolver', 'daesolver.c'))
    dae_include_dir = os.path.dirname(dae_dense_path)

    # Save the generated .c file
    with open(c_file_path, 'w') as f:
        f.write(all_sources)

    if not os.path.exists(dae_dense_path):
        raise FileNotFoundError(f"Could not find the C solver at {dae_dense_path}")

    # 6. Construct the Compiler Command
    # ------------------------------------------------------------------
    # Choose compiler: clang on macOS (especially for Accelerate), gcc elsewhere
    # ------------------------------------------------------------------
    if is_mac:
        compiler = 'clang'
    else:
        compiler = 'gcc'
    
    compile_cmd = [compiler, '-O3']

    # Platform-specific shared library flags
    if is_windows:
        compile_cmd.append('-shared')
    elif is_mac:
        compile_cmd.extend(['-dynamiclib', '-fPIC'])
    else: # Linux
        compile_cmd.extend(['-shared', '-fPIC'])

    # Include the internal daesolver headers
    compile_cmd.append(f'-I"{dae_include_dir}"')

    # ------------------------------------------------------------------
    # Backend-specific include/library resolution
    # ------------------------------------------------------------------
    if is_sparse and sparse_backend == 'klu':
        _append_klu_paths(compile_cmd, is_windows, is_mac, is_linux)

    elif is_sparse and sparse_backend == 'pardiso':
        _append_pardiso_paths(compile_cmd, is_windows, is_mac, is_linux)

    elif is_sparse and sparse_backend == 'accelerate':
        # No extra include paths needed — Accelerate is a system framework
        pass

    # Add source files and output
    compile_cmd.extend([f'"{dae_dense_path}"', f'"{c_file_path}"', '-o', f'"{lib_path}"'])

    # ------------------------------------------------------------------
    # Linker flags (backend-specific)
    # ------------------------------------------------------------------
    # Math library on POSIX
    if not is_windows:
        compile_cmd.append('-lm')      
        
    if is_sparse and sparse_backend == 'klu':
        compile_cmd.extend(['-lklu', '-lamd', '-lcolamd', '-lbtf', '-lsuitesparseconfig'])
    
    elif is_sparse and sparse_backend == 'pardiso':
        if is_windows:
            compile_cmd.extend(['-lmkl_intel_lp64_dll', '-lmkl_sequential_dll', '-lmkl_core_dll'])
        else:
            compile_cmd.extend(['-lmkl_intel_lp64', '-lmkl_sequential', '-lmkl_core', '-lpthread'])
    
    elif is_sparse and sparse_backend == 'accelerate':
        compile_cmd.extend(['-framework', 'Accelerate'])

    # 7. Execute Compilation
    cmd_str = ' '.join(compile_cmd)
    logging.info(f"[ctypes] Invoking compiler for {sys_os} (backend: {sparse_backend or 'dense'}) from: {os.getcwd()}")
    logging.info(f"[ctypes] Command: {cmd_str}")
    
    try:
        subprocess.run(cmd_str, shell=True, check=True)
        logging.info(f"[ctypes] Compilation successful. Library saved at {lib_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"[ctypes] Compilation FAILED. Error code: {e.returncode}")
        raise e


# ==============================================================================
# Private helpers for include/library path resolution per backend
# ==============================================================================

def _append_klu_paths(compile_cmd, is_windows, is_mac, is_linux):
    """Resolve SuiteSparse/KLU include and library paths."""
    klu_includes, klu_libs = [], []
    
    # Priority 1: Conda Environment
    if 'CONDA_PREFIX' in os.environ:
        prefix = os.environ['CONDA_PREFIX']
        if is_windows:
            klu_includes.append(os.path.join(prefix, 'Library', 'include', 'suitesparse'))
            klu_libs.append(os.path.join(prefix, 'Library', 'lib'))
        else:
            klu_includes.append(os.path.join(prefix, 'include', 'suitesparse'))
            klu_libs.append(os.path.join(prefix, 'lib'))
    
    # Priority 2: System-wide installations
    if is_mac:
        klu_includes.extend(['/opt/homebrew/include/suitesparse', '/usr/local/include/suitesparse'])
        klu_libs.extend(['/opt/homebrew/lib', '/usr/local/lib'])
    elif is_linux:
        klu_includes.append('/usr/include/suitesparse')

    for inc_dir in klu_includes:
        if os.path.exists(inc_dir):
            compile_cmd.append(f'-I"{inc_dir}"')
            break
            
    for lib_dir in klu_libs:
        if os.path.exists(lib_dir):
            compile_cmd.append(f'-L"{lib_dir}"')
            break


def _find_file(filename, search_root):
    """Recursively search for a file. Returns the containing directory or None."""
    for dirpath, dirnames, filenames in os.walk(search_root):
        if filename in filenames:
            return dirpath
    return None


def _append_pardiso_paths(compile_cmd, is_windows, is_mac, is_linux):
    """Resolve Intel MKL include and library paths for PARDISO."""
    import sysconfig

    mkl_inc_dir = None
    mkl_lib_dir = None

    # Priority 1: MKLROOT
    if 'MKLROOT' in os.environ:
        mkl_root = os.environ['MKLROOT']
        c = os.path.join(mkl_root, 'include')
        if os.path.isdir(c): mkl_inc_dir = c
        if is_windows:
            c = os.path.join(mkl_root, 'lib', 'intel64')
        elif is_mac:
            c = os.path.join(mkl_root, 'lib')
        else:
            c = os.path.join(mkl_root, 'lib', 'intel64')
        if os.path.isdir(c): mkl_lib_dir = c

    # Priority 2: Recursive file search (old pydae approach)
    if mkl_inc_dir is None or mkl_lib_dir is None:
        search_root = sysconfig.get_path('data')

        if mkl_inc_dir is None:
            found = _find_file('mkl.h', search_root)
            if found:
                mkl_inc_dir = found
                logging.info(f"[PARDISO] Found mkl.h in: {found}")

        if mkl_lib_dir is None:
            if is_windows:
                lib_file = 'mkl_intel_lp64_dll.lib'
            elif is_mac:
                lib_file = 'libmkl_intel_lp64.dylib'
            else:
                lib_file = 'libmkl_intel_lp64.so'
            found = _find_file(lib_file, search_root)
            if found:
                mkl_lib_dir = found
                logging.info(f"[PARDISO] Found {lib_file} in: {found}")

    # Priority 3: CONDA_PREFIX standard locations
    if 'CONDA_PREFIX' in os.environ:
        prefix = os.environ['CONDA_PREFIX']
        if is_windows:
            if mkl_inc_dir is None:
                c = os.path.join(prefix, 'Library', 'include')
                if os.path.isdir(c): mkl_inc_dir = c
            if mkl_lib_dir is None:
                c = os.path.join(prefix, 'Library', 'lib')
                if os.path.isdir(c): mkl_lib_dir = c
        else:
            if mkl_inc_dir is None:
                c = os.path.join(prefix, 'include')
                if os.path.isdir(c): mkl_inc_dir = c
            if mkl_lib_dir is None:
                c = os.path.join(prefix, 'lib')
                if os.path.isdir(c): mkl_lib_dir = c

    # Priority 4: System paths (Linux)
    if is_linux:
        if mkl_inc_dir is None:
            for d in ['/usr/include/mkl']:
                if os.path.isdir(d): mkl_inc_dir = d; break
        if mkl_lib_dir is None:
            for d in ['/usr/lib/x86_64-linux-gnu',
                      '/opt/intel/oneapi/mkl/latest/lib/intel64']:
                if os.path.isdir(d): mkl_lib_dir = d; break

    # Apply
    if mkl_inc_dir:
        compile_cmd.append(f'-I"{mkl_inc_dir}"')
    if mkl_lib_dir:
        compile_cmd.append(f'-L"{mkl_lib_dir}"')
