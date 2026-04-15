# codegen/cffi_builder.py
import os
import re
import platform
import logging
import sympy as sym
from cffi import FFI

# ---------------------------------------------------------------------------
# Supported sparse solver backends (C preprocessor define, compiler flag)
# ---------------------------------------------------------------------------
SPARSE_BACKENDS = {
    'klu':        'USE_SPARSE',       # SuiteSparse KLU  (0-based CSC)
    'pardiso':    'USE_PARDISO',      # Intel MKL PARDISO (1-based CSR)
    'accelerate': 'USE_ACCELERATE',   # Apple Accelerate  (0-based CSC, long* Ap)
}


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
        item['xyup'] = string

def _get_c_funcs(func_name, eq_list, is_sparse=False):
    def_str = f'void {func_name}(double *data, double *x, double *y, double *u, double *p, double Dt);\n'
    source_str = f'void {func_name}(double *data, double *x, double *y, double *u, double *p, double Dt){{\n\n'
    for it, item in enumerate(eq_list):
        # Dual indexing: 'sp_idx' for 1D sparse arrays, 'de_idx' for 2D flattened Dense arrays
        idx = item['sp_idx'] if is_sparse else item.get('de_idx', it)
        source_str += f"    data[{idx}] = {item['xyup']};\n"
    source_str += '\n}\n\n'
    return def_str, source_str


def generate_and_compile_cffi(builder_obj):
    """
    Generates C code and compiles it using Python's CFFI.
    
    Sparse backend selection
    ========================
    ``builder_obj.sparse`` can be:
      * False / None   → dense mode (no sparse solver)
      * True / 'klu'   → SuiteSparse KLU  (original behaviour)
      * 'pardiso'      → Intel MKL PARDISO
      * 'accelerate'   → Apple Accelerate Sparse Solvers (macOS only)
    """
    logging.info(f"[CFFI] Generating C code for system: {builder_obj.name}")
    ffi = FFI()

    # ------------------------------------------------------------------
    # Normalise the sparse setting into (is_sparse: bool, backend: str)
    # ------------------------------------------------------------------
    raw_sparse = getattr(builder_obj, 'sparse', False)
    if raw_sparse is True:
        is_sparse = True
        sparse_backend = 'klu'
    elif isinstance(raw_sparse, str) and raw_sparse.lower() in SPARSE_BACKENDS:
        is_sparse = True
        sparse_backend = raw_sparse.lower()
    else:
        is_sparse = False
        sparse_backend = None
    
    all_defs, all_sources = "", ""
    
    # 1. Start C source string
    all_sources += "#include <math.h>\n"
    all_sources += '#include "daesolver.h"\n\n'
    
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
            """Emit Ap_<suffix>[], Ai_<suffix>[], NNZ_<suffix> C arrays."""
            src = ""
            if backend == 'accelerate':
                ap_str = ", ".join(map(str, ap))
                src += f"long Ap_{suffix}[] = {{{ap_str}}};\n"
            elif backend == 'pardiso':
                ap_str = ", ".join(str(v + 1) for v in ap)
                src += f"int Ap_{suffix}[] = {{{ap_str}}};\n"
            else:  # klu
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

    # 4. Generate Jacobian functions
    for name, eq_list in [('jac_ini_eval', builder_obj.jac_ini_list), ('jac_trap_eval', builder_obj.jac_trap_list)]:
        d, s = _get_c_funcs(name, eq_list, is_sparse=is_sparse)
        all_defs += d; all_sources += s

    # 5. CDEF: Define the signatures of the functions Python is allowed to call.
    # These are backend-agnostic — the C signatures are identical for all backends.
    ffi.cdef("""
        int ini(double *jac_ini, int *pivots, double *x, double *y, double *xy, double *Dxy, 
                double *u, double *p, int N_x, int N_y, int max_it, double itol, 
                double *z, double *inidblparams, int *iniintparams, double *f, double *g, double *fg);

        int run(double t, double t_end, double *jac_trap, int *pivots, double *x, double *y, double *xy, 
                double *u, double *p, int N_x, int N_y, int max_it, double itol, 
                int *its, double Dt, double *z, double *dblparams, int *intparams, double *Time, 
                double *X, double *Y, double *Z, int N_z, int N_store, double *f, double *g, double *fg);
    """)

    # 6. Paths and OS detection
    output_dir = os.path.abspath('build')
    os.makedirs(output_dir, exist_ok=True)
    
    dae_dense_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'daesolver', 'daesolver.c'))
    dae_include_dir = os.path.dirname(dae_dense_path)
    
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

    # ------------------------------------------------------------------
    # 7. Configure compilation arguments
    # ------------------------------------------------------------------
    include_dirs = [dae_include_dir]
    library_dirs = []
    libraries = []
    extra_compile_args = ['-O3']
    extra_link_args = []

    # CRITICAL: Pass the backend #define as a compiler flag so it is visible
    # to BOTH compilation units (the generated model source AND daesolver.c
    # which is compiled separately via the `sources=` parameter).
    if is_sparse:
        c_define = SPARSE_BACKENDS[sparse_backend]
        extra_compile_args.append(f'-D{c_define}')

    # ------------------------------------------------------------------
    # 8. Backend-specific include / library / linker configuration
    # ------------------------------------------------------------------
    if is_sparse and sparse_backend == 'klu':
        _append_klu_paths(include_dirs, library_dirs, libraries,
                          is_windows, is_mac, is_linux)

    elif is_sparse and sparse_backend == 'pardiso':
        _append_pardiso_paths(include_dirs, library_dirs, libraries,
                              is_windows, is_mac, is_linux)

    elif is_sparse and sparse_backend == 'accelerate':
        # Accelerate is a system framework — no extra include/lib dirs
        extra_link_args.extend(['-framework', 'Accelerate'])

    # ------------------------------------------------------------------
    # 9. Configure CFFI Extension Source
    # ------------------------------------------------------------------
    module_name = f"{builder_obj.name}_cffi"
    
    ffi.set_source(
        module_name,
        all_sources,                      # Generated string with model equations + sparse arrays
        sources=[dae_dense_path],         # daesolver.c compiled as a separate translation unit
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )

    # 10. Execute Compilation
    logging.info(f"[CFFI] Compiling module '{module_name}' "
                 f"(backend: {sparse_backend or 'dense'})")
    try:
        lib_path = ffi.compile(tmpdir=output_dir)
        logging.info(f"[CFFI] Compilation successful. Python Module saved at: {lib_path}")
    except Exception as e:
        logging.error(f"[CFFI] Compilation FAILED. Exception: {e}")
        raise e


# ==============================================================================
# Private helpers — include / library path resolution per backend
# ==============================================================================

def _append_klu_paths(include_dirs, library_dirs, libraries,
                      is_windows, is_mac, is_linux):
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
            include_dirs.append(inc_dir)
            break

    for lib_dir in klu_libs:
        if os.path.exists(lib_dir):
            library_dirs.append(lib_dir)
            break

    libraries.extend(['klu', 'amd', 'colamd', 'btf', 'suitesparseconfig'])


def _find_file(filename, search_root):
    """
    Recursively search for a file starting from search_root.
    Returns the directory containing the file, or None.
    This replicates the old pydae find_file() behavior for locating
    MKL headers and libraries inside Conda's pkgs folder.
    """
    for dirpath, dirnames, filenames in os.walk(search_root):
        if filename in filenames:
            return dirpath
    return None


def _append_pardiso_paths(include_dirs, library_dirs, libraries,
                          is_windows, is_mac, is_linux):
    """
    Resolve Intel MKL include and library paths for PARDISO.
    
    On Windows the MKL libraries have a ``_dll`` suffix
    (e.g. ``mkl_intel_lp64_dll.lib``), which is different from Linux/macOS.
    
    Discovery order:
      1. MKLROOT environment variable (Intel oneAPI installs set this)
      2. Recursive file search under the Python/Conda data path
         (handles Conda ``pkgs/mkl-devel-*`` layouts)
      3. CONDA_PREFIX standard locations
      4. Common system paths (Linux)
    """
    import sysconfig

    mkl_inc_dir = None
    mkl_lib_dir = None

    # ---- Priority 1: MKLROOT ----
    if 'MKLROOT' in os.environ:
        mkl_root = os.environ['MKLROOT']
        candidate_inc = os.path.join(mkl_root, 'include')
        if os.path.isdir(candidate_inc):
            mkl_inc_dir = candidate_inc
        if is_windows:
            candidate_lib = os.path.join(mkl_root, 'lib', 'intel64')
        elif is_mac:
            candidate_lib = os.path.join(mkl_root, 'lib')
        else:
            candidate_lib = os.path.join(mkl_root, 'lib', 'intel64')
        if os.path.isdir(candidate_lib):
            mkl_lib_dir = candidate_lib

    # ---- Priority 2: Recursive file search (old pydae approach) ----
    if mkl_inc_dir is None or mkl_lib_dir is None:
        search_root = sysconfig.get_path('data')  # Anaconda/Python root

        if mkl_inc_dir is None:
            found = _find_file('mkl.h', search_root)
            if found:
                mkl_inc_dir = found
                logging.info(f"[PARDISO] Found mkl.h in: {found}")
            else:
                logging.warning("[PARDISO] mkl.h not found. Is mkl-devel installed?")

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
            else:
                logging.warning(f"[PARDISO] {lib_file} not found. Is mkl-devel installed?")

    # ---- Priority 3: CONDA_PREFIX standard locations ----
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

    # ---- Priority 4: System paths (Linux) ----
    if is_linux:
        if mkl_inc_dir is None:
            for d in ['/usr/include/mkl']:
                if os.path.isdir(d): mkl_inc_dir = d; break
        if mkl_lib_dir is None:
            for d in ['/usr/lib/x86_64-linux-gnu',
                      '/opt/intel/oneapi/mkl/latest/lib/intel64']:
                if os.path.isdir(d): mkl_lib_dir = d; break

    # ---- Apply discovered paths ----
    if mkl_inc_dir:
        include_dirs.append(mkl_inc_dir)
    if mkl_lib_dir:
        library_dirs.append(mkl_lib_dir)

    # ---- Library names (platform-specific) ----
    if is_windows:
        # Windows MKL ships with _dll suffix on the .lib import libraries
        libraries.extend(['mkl_intel_lp64_dll', 'mkl_sequential_dll', 'mkl_core_dll'])
    else:
        libraries.extend(['mkl_intel_lp64', 'mkl_sequential', 'mkl_core'])
        libraries.append('pthread')
