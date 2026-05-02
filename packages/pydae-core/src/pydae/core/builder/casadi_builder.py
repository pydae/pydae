import casadi as ca


class MathBackend:
    def __init__(self, use_casadi):
        self.use_casadi = use_casadi
        self._sym_cache = {}
        if use_casadi:
            import casadi as ca
            self.ca = ca
        else:
            import sympy as sp
            self.sp = sp

    def symbols(self, names, real=True, **kwargs):
        name_list = names.replace(',', ' ').split()
        if self.use_casadi:
            syms = []
            for name in name_list:
                if name not in self._sym_cache:
                    self._sym_cache[name] = self.ca.SX.sym(name)
                syms.append(self._sym_cache[name])
            return syms[0] if len(syms) == 1 else syms
        else:
            return self.sp.symbols(names, real=real, **kwargs)

    def sin(self, x):
        if self.use_casadi:
            return self.ca.sin(x)
        else:
            return self.sp.sin(x)

    def cos(self, x):
        if self.use_casadi:
            return self.ca.cos(x)
        else:
            return self.sp.cos(x)

    def tan(self, x):
        if self.use_casadi:
            return self.ca.tan(x)
        else:
            return self.sp.tan(x)

    def atan2(self, y, x):
        if self.use_casadi:
            return self.ca.atan2(y, x)
        else:
            return self.sp.atan2(y, x)

    def sqrt(self, x):
        if self.use_casadi:
            return self.ca.sqrt(x)
        else:
            return self.sp.sqrt(x)

    def exp(self, x):
        if self.use_casadi:
            return self.ca.exp(x)
        else:
            return self.sp.exp(x)

    def zeros(self, rows, cols=None):
        if cols is None:
            cols = rows
        if self.use_casadi:
            return self.ca.SX.zeros(rows, cols)
        else:
            return self.sp.zeros(rows, cols)

    def Matrix(self, data):
        if self.use_casadi:
            # Check if data contains SX symbols
            def _has_sx(obj):
                if isinstance(obj, self.ca.SX):
                    return True
                if isinstance(obj, list):
                    return any(_has_sx(item) for item in obj)
                return False
            if _has_sx(data):
                # Build SX matrix from nested list
                rows = len(data)
                cols = len(data[0]) if rows > 0 else 0
                M = self.ca.SX(rows, cols)
                for i in range(rows):
                    for j in range(cols):
                        M[i, j] = data[i][j]
                return M
            return self.ca.MX(data)
        else:
            return self.sp.Matrix(data)

    @property
    def I(self):  # noqa: E743
        if self.use_casadi:
            return 1j
        else:
            return self.sp.I

    def re(self, x):
        if self.use_casadi:
            return self.ca.re(x)
        else:
            return self.sp.re(x)

    def im(self, x):
        if self.use_casadi:
            return self.ca.im(x)
        else:
            return self.sp.im(x)

    def Piecewise(self, *args):
        if self.use_casadi:
            return self.ca.if_else(args[0][1], args[0][0], args[1][0])
        else:
            return self.sp.Piecewise(*args)

    def min(self, *args):
        if self.use_casadi:
            if len(args) == 2:
                return self.ca.fmin(args[0], args[1])
            result = args[0]
            for arg in args[1:]:
                result = self.ca.fmin(result, arg)
            return result
        else:
            return self.sp.Min(*args)

    def max(self, *args):
        if self.use_casadi:
            if len(args) == 2:
                return self.ca.fmax(args[0], args[1])
            result = args[0]
            for arg in args[1:]:
                result = self.ca.fmax(result, arg)
            return result
        else:
            return self.sp.Max(*args)

    def hard_limit(self, value, lower_bound, upper_bound):
        """Applies a mathematical hard limit (saturation) to a value.

        Safe for both SymPy algebra and CasADi Newton solvers.
        """
        if self.use_casadi:
            return self.ca.fmax(lower_bound, self.ca.fmin(upper_bound, value))
        else:
            return self.sp.Max(lower_bound, self.sp.Min(upper_bound, value))


class CasadiBuilder:
    def __init__(self, sys_dict):
        self.sys_dict = sys_dict

    def build(self):
        """Compiles the symbolic equations into CasADi solvers and matrix evaluators."""
        print(f"Building model: {self.sys_dict['name']}...")

        # ---------------------------------------------------------
        # 1. Unpack Explicit State and Equation Vectors
        # ---------------------------------------------------------
        f = ca.vertcat(*self.sys_dict['f_list'])
        g = ca.vertcat(*self.sys_dict['g_list'])

        # ---------------------------------------------------------
        # 2. Auto-Extract Symbolic Variables (Backward Compatibility)
        # ---------------------------------------------------------
        if 'h_dict' in self.sys_dict and self.sys_dict['h_dict']:
            h_exprs = ca.vertcat(*list(self.sys_dict['h_dict'].values()))
            all_exprs = ca.vertcat(f, g, h_exprs)
        else:
            all_exprs = ca.vertcat(f, g)

        all_syms = ca.symvar(all_exprs)
        sym_map = {sym.name(): sym for sym in all_syms}

        def extract_syms_from_names(name_list):
            """Match symbol names to the exact CasADi objects from expressions."""
            result = []
            for sym_obj in name_list:
                name = str(sym_obj)
                if name in sym_map:
                    result.append(sym_map[name])
                else:
                    result.append(ca.SX.sym(name))
            return result

        x = ca.vertcat(*extract_syms_from_names(self.sys_dict['x_list']))
        y_ini = ca.vertcat(*extract_syms_from_names(self.sys_dict['y_ini_list']))
        y_run = ca.vertcat(*extract_syms_from_names(self.sys_dict['y_run_list']))

        def extract_syms(target_dict):
            """Matches dictionary keys to the exact CasADi symbolic objects."""
            sym_list = []
            for name in target_dict.keys():
                if name in sym_map:
                    sym_list.append(sym_map[name])
                else:
                    sym_list.append(ca.SX.sym(name))
            return sym_list

        self.sys_dict['p_list'] = extract_syms(self.sys_dict.get('params_dict', {}))
        self.sys_dict['u_ini_list'] = extract_syms(self.sys_dict.get('u_ini_dict', {}))
        self.sys_dict['u_run_list'] = extract_syms(self.sys_dict.get('u_run_dict', {}))

        # ---------------------------------------------------------
        # 2.5. Substitute y_ini→u_run symbols in f/g for runtime
        # ---------------------------------------------------------
        y_ini_syms = self.sys_dict['y_ini_list']
        y_run_syms = self.sys_dict['y_run_list']
        u_run_syms = self.sys_dict['u_run_list']

        yini_names = [str(s) for s in y_ini_syms]
        yrun_names = [str(s) for s in y_run_syms]
        urun_names = [str(s) for s in u_run_syms]

        subs = {}
        for i, name in enumerate(yini_names):
            if name not in yrun_names and name in urun_names:
                subs[y_ini_syms[i]] = u_run_syms[urun_names.index(name)]

        if subs:
            sub_keys = list(subs.keys())
            sub_vals = list(subs.values())
            f_list = [f[i] for i in range(f.size1())]
            g_list = [g[i] for i in range(g.size1())]
            f_list = ca.substitute(f_list, sub_keys, sub_vals)
            g_list = ca.substitute(g_list, sub_keys, sub_vals)
            f = ca.vertcat(*f_list)
            g = ca.vertcat(*g_list)

        p = ca.vertcat(*self.sys_dict['p_list'])
        u_ini = ca.vertcat(*self.sys_dict['u_ini_list'])
        u_run = ca.vertcat(*self.sys_dict['u_run_list'])

        # ---------------------------------------------------------
        # 3. Build Initialization Solver (Rootfinder)
        # ---------------------------------------------------------
        v_ini = ca.vertcat(x, y_ini)
        p_ini = ca.vertcat(p, u_ini)
        eq_ini = ca.vertcat(f, g)

        rf_dict = {'x': v_ini, 'p': p_ini, 'g': eq_ini}
        self.rf = ca.rootfinder('rf', 'newton', rf_dict)

        # ---------------------------------------------------------
        # 4. Store DAE Dictionary for the Runtime Integrator
        # ---------------------------------------------------------
        p_run = ca.vertcat(p, u_run)
        self.dae_dict = {
            'x': x,
            'z': y_run,
            'p': p_run,
            'ode': f,
            'alg': g
        }

        # ---------------------------------------------------------
        # 5. Build Sparse Jacobian Evaluators (save at build time)
        # ---------------------------------------------------------
        # These Jacobians are extremely sparse for power systems.
        # The dense A matrix is computed numerically at runtime via:
        #   A = F_x - F_y @ solve(G_y, G_x)
        self._Fx_fn = ca.Function('Fx', [x, y_run, p_run], [ca.jacobian(f, x)])
        self._Fy_fn = ca.Function('Fy', [x, y_run, p_run], [ca.jacobian(f, y_run)])
        self._Gx_fn = ca.Function('Gx', [x, y_run, p_run], [ca.jacobian(g, x)])
        self._Gy_fn = ca.Function('Gy', [x, y_run, p_run], [ca.jacobian(g, y_run)])

        # Jacobians for input matrices (B, C, D)
        self._Fu_fn = ca.Function('Fu', [x, y_run, p_run], [ca.jacobian(f, u_run)])
        self._Gu_fn = ca.Function('Gu', [x, y_run, p_run], [ca.jacobian(g, u_run)])

        # Output Jacobians (for C, D matrices)
        if 'h_dict' in self.sys_dict and self.sys_dict['h_dict']:
            h = ca.vertcat(*list(self.sys_dict['h_dict'].values()))
            self._Hx_fn = ca.Function('Hx', [x, y_run, p_run], [ca.jacobian(h, x)])
            self._Hy_fn = ca.Function('Hy', [x, y_run, p_run], [ca.jacobian(h, y_run)])
            self._Hu_fn = ca.Function('Hu', [x, y_run, p_run], [ca.jacobian(h, u_run)])
        else:
            self._Hx_fn = None
            self._Hy_fn = None
            self._Hu_fn = None

        print("Build complete!")
        return self

    def export_code(self, filename="model_compiled.c"):
        """Export all CasADi functions to a single C file using CodeGenerator.

        Exports sparse Jacobian Functions (Fx, Fy, Gx, Gy, Fu, Gu) and
        residual/jacobian for initialization. The dense A matrix is computed
        numerically at runtime via A = Fx - Fy @ solve(Gy, Gx).

        Note: CasADi rootfinder objects cannot be directly code-generated.
        Instead, the residual and its Jacobian are exported so the caller can
        use any Newton solver (e.g., ca.rootfinder on the loaded binary).

        Parameters
        ----------
        filename : str
            Output C file path (default: "model_compiled.c").

        Returns
        -------
        str
            Path to the generated C file.
        """
        import os

        c_path = os.path.abspath(filename)
        output_dir = os.path.dirname(c_path)
        stem = os.path.splitext(os.path.basename(c_path))[0]
        os.makedirs(output_dir, exist_ok=True)

        saved_cwd = os.getcwd()
        os.chdir(output_dir)

        try:
            gen = ca.CodeGenerator(stem, {'main': False, 'with_header': True})

            x = ca.vertcat(*self.sys_dict['x_list'])
            y_ini = ca.vertcat(*self.sys_dict['y_ini_list'])
            y_run = ca.vertcat(*self.sys_dict['y_run_list'])
            p = ca.vertcat(*self.sys_dict['p_list'])
            u_ini = ca.vertcat(*self.sys_dict['u_ini_list'])
            u_run = ca.vertcat(*self.sys_dict['u_run_list'])

            f = ca.vertcat(*self.sys_dict['f_list'])
            g = ca.vertcat(*self.sys_dict['g_list'])

            # Substitute y_ini→u_run symbols in f/g for runtime
            yini_names = [str(s) for s in self.sys_dict['y_ini_list']]
            yrun_names = [str(s) for s in self.sys_dict['y_run_list']]
            urun_names = [str(s) for s in self.sys_dict['u_run_list']]
            subs = {}
            for i, name in enumerate(yini_names):
                if name not in yrun_names and name in urun_names:
                    subs[self.sys_dict['y_ini_list'][i]] = self.sys_dict['u_run_list'][urun_names.index(name)]
            if subs:
                sub_keys = list(subs.keys())
                sub_vals = list(subs.values())
                f_list = [f[i] for i in range(f.size1())]
                g_list = [g[i] for i in range(g.size1())]
                f_list = ca.substitute(f_list, sub_keys, sub_vals)
                g_list = ca.substitute(g_list, sub_keys, sub_vals)
                f = ca.vertcat(*f_list)
                g = ca.vertcat(*g_list)

            v_ini = ca.vertcat(x, y_ini)
            p_ini = ca.vertcat(p, u_ini)
            p_run = ca.vertcat(p, u_run)

            # Initialization functions
            res_ini = ca.vertcat(f, g)
            res_fn = ca.Function('residual', [v_ini, p_ini], [res_ini], {'cse': True})
            gen.add(res_fn)

            J_ini = ca.jacobian(res_ini, v_ini)
            jac_fn = ca.Function('jacobian', [v_ini, p_ini], [J_ini], {'cse': True})
            gen.add(jac_fn)

            # Sparse Jacobian Functions (small, avoid dense A at build time)
            gen.add(ca.Function('Fx', [x, y_run, p_run], [ca.jacobian(f, x)], {'cse': True}))
            gen.add(ca.Function('Fy', [x, y_run, p_run], [ca.jacobian(f, y_run)], {'cse': True}))
            gen.add(ca.Function('Gx', [x, y_run, p_run], [ca.jacobian(g, x)], {'cse': True}))
            gen.add(ca.Function('Gy', [x, y_run, p_run], [ca.jacobian(g, y_run)], {'cse': True}))
            gen.add(ca.Function('Fu', [x, y_run, p_run], [ca.jacobian(f, u_run)], {'cse': True}))
            gen.add(ca.Function('Gu', [x, y_run, p_run], [ca.jacobian(g, u_run)], {'cse': True}))

            if 'h_dict' in self.sys_dict and self.sys_dict['h_dict']:
                h = ca.vertcat(*list(self.sys_dict['h_dict'].values()))
                gen.add(ca.Function('Hx', [x, y_run, p_run], [ca.jacobian(h, x)], {'cse': True}))
                gen.add(ca.Function('Hy', [x, y_run, p_run], [ca.jacobian(h, y_run)], {'cse': True}))
                gen.add(ca.Function('Hu', [x, y_run, p_run], [ca.jacobian(h, u_run)], {'cse': True}))
            else:
                n_x = x.size1()
                n_y = y_run.size1()
                n_u = u_run.size1()
                gen.add(ca.Function('Hx', [x, y_run, p_run], [ca.SX.zeros(0, n_x)]))
                gen.add(ca.Function('Hy', [x, y_run, p_run], [ca.SX.zeros(0, n_y)]))
                gen.add(ca.Function('Hu', [x, y_run, p_run], [ca.SX.zeros(0, n_u)]))

            gen.add(ca.Function('ode', [x, y_run, p_run], [f]))
            gen.add(ca.Function('alg', [x, y_run, p_run], [g]))

            gen.generate()
        finally:
            os.chdir(saved_cwd)

        print(f"C code exported to: {c_path}")
        return c_path

    @staticmethod
    def compile_shared_library(c_path, output_name=None, compiler=None):
        """Compile a generated C file into a shared library (.so/.dll).

        Parameters
        ----------
        c_path : str
            Path to the .c file generated by export_code().
        output_name : str, optional
            Output library name (without extension). Defaults to stem of c_path.
        compiler : str, optional
            Compiler to use ('gcc', 'cl', or 'auto'). Defaults to 'auto'.

        Returns
        -------
        str
            Absolute path to the compiled shared library.
        """
        import os
        import platform
        import shutil
        import subprocess

        c_path = os.path.abspath(c_path)
        if output_name is None:
            output_name = os.path.splitext(os.path.basename(c_path))[0]

        output_dir = os.path.dirname(c_path)
        system = platform.system()

        if compiler is None or compiler == 'auto':
            if system == 'Windows':
                compiler = 'cl' if shutil.which('cl') else 'gcc'
            else:
                compiler = 'gcc'

        ext = '.dll' if system == 'Windows' else '.so'
        lib_path = os.path.join(output_dir, f'{output_name}{ext}')

        if compiler == 'cl':
            cl_path = shutil.which('cl')
            if cl_path is None:
                vs_base = r"C:\Program Files\Microsoft Visual Studio"
                if os.path.exists(vs_base):
                    for root, dirs, files in os.walk(vs_base):
                        if 'cl.exe' in files:
                            cl_path = os.path.join(root, 'cl.exe')
                            if 'Hostx64' in root and 'x64' in root:
                                break
                if cl_path is None:
                    raise RuntimeError("MSVC cl.exe not found. Install Visual Studio or use compiler='gcc'.")

            env = os.environ.copy()
            cl_dir = os.path.dirname(cl_path)
            env['PATH'] = cl_dir + os.pathsep + env.get('PATH', '')

            cmd = [cl_path, '/LD', '/O2', '/EHsc', c_path, f'/Fe:{lib_path}']
            print(f"Compiling {c_path} with MSVC...")
            result = subprocess.run(cmd, cwd=output_dir, capture_output=True, text=True, env=env)
        elif compiler == 'gcc':
            gcc = shutil.which('gcc') or shutil.which('x86_64-w64-mingw32-gcc')
            if gcc is None:
                raise RuntimeError("gcc not found. Install MinGW or use compiler='cl' on Windows.")
            if system == 'Windows':
                cmd = [gcc, '-shared', '-O3', c_path, '-o', lib_path]
            else:
                cmd = [gcc, '-shared', '-O3', '-fPIC', c_path, '-o', lib_path]
            print(f"Compiling {c_path} with gcc...")
            result = subprocess.run(cmd, cwd=output_dir, capture_output=True, text=True)
        else:
            raise ValueError(f"Unsupported compiler: {compiler}")

        if result.returncode != 0:
            raise RuntimeError(f"Compilation failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")

        print(f"Shared library compiled: {lib_path}")
        return lib_path

    def compile_jit(self, output_dir=None):
        """Compile all evaluator functions to a single shared library via CasADi JIT.

        Uses CasADi's internal JIT compiler which applies CSE, dead code
        elimination, and other optimizations automatically. Generates much
        smaller binaries than raw CodeGenerator.

        Parameters
        ----------
        output_dir : str, optional
            Directory for the output library. Defaults to current working dir.

        Returns
        -------
        str
            Absolute path to the compiled shared library.
        """
        import os
        import platform
        import shutil

        if output_dir is None:
            output_dir = os.getcwd()
        os.makedirs(output_dir, exist_ok=True)

        stem = self.sys_dict['name']
        ext = '.dll' if platform.system() == 'Windows' else '.so'
        lib_path = os.path.join(output_dir, f'{stem}_compiled{ext}')

        x = ca.vertcat(*self.sys_dict['x_list'])
        y_ini = ca.vertcat(*self.sys_dict['y_ini_list'])
        y_run = ca.vertcat(*self.sys_dict['y_run_list'])
        p = ca.vertcat(*self.sys_dict['p_list'])
        u_ini = ca.vertcat(*self.sys_dict['u_ini_list'])
        u_run = ca.vertcat(*self.sys_dict['u_run_list'])

        f = ca.vertcat(*self.sys_dict['f_list'])
        g = ca.vertcat(*self.sys_dict['g_list'])

        # Substitute y_ini→u_run symbols in f/g for runtime
        yini_names = [str(s) for s in self.sys_dict['y_ini_list']]
        yrun_names = [str(s) for s in self.sys_dict['y_run_list']]
        urun_names = [str(s) for s in self.sys_dict['u_run_list']]
        subs = {}
        for i, name in enumerate(yini_names):
            if name not in yrun_names and name in urun_names:
                subs[self.sys_dict['y_ini_list'][i]] = self.sys_dict['u_run_list'][urun_names.index(name)]
        if subs:
            sub_keys = list(subs.keys())
            sub_vals = list(subs.values())
            f_list = [f[i] for i in range(f.size1())]
            g_list = [g[i] for i in range(g.size1())]
            f_list = ca.substitute(f_list, sub_keys, sub_vals)
            g_list = ca.substitute(g_list, sub_keys, sub_vals)
            f = ca.vertcat(*f_list)
            g = ca.vertcat(*g_list)

        v_ini = ca.vertcat(x, y_ini)
        p_ini = ca.vertcat(p, u_ini)
        p_run = ca.vertcat(p, u_run)

        # Build all functions
        res_ini = ca.vertcat(f, g)
        res_fn = ca.Function('residual', [v_ini, p_ini], [res_ini])
        J_ini = ca.jacobian(res_ini, v_ini)
        jac_fn = ca.Function('jacobian', [v_ini, p_ini], [J_ini])

        F_x = ca.jacobian(f, x)
        F_y_run = ca.jacobian(f, y_run)
        G_x = ca.jacobian(g, x)
        G_y_run = ca.jacobian(g, y_run)
        G_u_run = ca.jacobian(g, u_run)
        F_u_run = ca.jacobian(f, u_run)

        Gy_inv_Gx = ca.solve(G_y_run, G_x)
        A_sym = F_x - ca.mtimes(F_y_run, Gy_inv_Gx)
        A_eval_fn = ca.Function('A_eval', [x, y_run, p_run], [A_sym])

        Gy_inv_Gu = ca.solve(G_y_run, G_u_run)
        B_sym = F_u_run - ca.mtimes(F_y_run, Gy_inv_Gu)
        B_eval_fn = ca.Function('B_eval', [x, y_run, p_run], [B_sym])

        if 'h_dict' in self.sys_dict and self.sys_dict['h_dict']:
            h = ca.vertcat(*list(self.sys_dict['h_dict'].values()))
            H_x = ca.jacobian(h, x)
            H_y_run = ca.jacobian(h, y_run)
            H_u_run = ca.jacobian(h, u_run)

            C_sym = H_x - ca.mtimes(H_y_run, Gy_inv_Gx)
            D_sym = H_u_run - ca.mtimes(H_y_run, Gy_inv_Gu)

            C_eval_fn = ca.Function('C_eval', [x, y_run, p_run], [C_sym])
            D_eval_fn = ca.Function('D_eval', [x, y_run, p_run], [D_sym])
        else:
            n_x = x.size1()
            n_u = u_run.size1()
            C_eval_fn = ca.Function('C_eval', [x, y_run, p_run], [ca.SX.zeros(0, n_x)])
            D_eval_fn = ca.Function('D_eval', [x, y_run, p_run], [ca.SX.zeros(0, n_u)])

        ode_fn = ca.Function('ode', [x, y_run, p_run], [f])
        alg_fn = ca.Function('alg', [x, y_run, p_run], [g])

        # Compile each function separately using JIT
        # Each gets its own .so/.dll since CasADi compiles one function per library
        jit_opts = {'cse': True, 'compiler': 'clang' if shutil.which('clang') else 'gcc'}

        functions = [res_fn, jac_fn, A_eval_fn, B_eval_fn, C_eval_fn, D_eval_fn, ode_fn, alg_fn]
        compiled_paths = []
        for fn in functions:
            path = fn.compile(os.path.join(output_dir, f'{fn.name()}{ext}'), jit_opts)
            compiled_paths.append(path)
            print(f"  JIT compiled: {fn.name()}")

        # Store compiled paths in sys_dict for later loading
        self.sys_dict['_jit_paths'] = {fn.name(): p for fn, p in zip(functions, compiled_paths)}
        self.sys_dict['_jit_main_lib'] = compiled_paths[0]

        print(f"JIT compilation complete: {lib_path}")
        return compiled_paths
