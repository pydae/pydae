import json

import casadi as ca
import numpy as np
from scipy.optimize import root


class CasadiModel:
    def __init__(self, builder=None, binary_path=None, dae_dict=None, sys_dict=None,
                 newton_tol=1e-10, newton_max_iter=50,
                 integrator_reltol=1e-6, integrator_abstol=1e-5,
                 rootfinder_opts=None):
        self._binary_path = binary_path

        # Store tolerance parameters
        self._newton_tol = newton_tol
        self._newton_max_iter = newton_max_iter
        self._integrator_reltol = integrator_reltol
        self._integrator_abstol = integrator_abstol
        self._rootfinder_opts = rootfinder_opts or {"abstol": 1e-9, "reltol": 1e-9, "max_iter": 100}

        if binary_path is not None and dae_dict is not None and sys_dict is not None:
            self._init_from_binary(binary_path, dae_dict, sys_dict)
        elif builder is not None:
            self._init_from_builder(builder)
        else:
            raise ValueError("Either 'builder' or ('binary_path', 'dae_dict', 'sys_dict') must be provided")

        self._alg_fn = self._build_alg_fn()

    def _build_alg_fn(self):
        """Build the algebraic equation evaluation function from dae_dict."""
        if 'alg' not in self.dae_dict or self.dae_dict['alg'] is None:
            return None
        try:
            x_sym = self.dae_dict['x']
            z_sym = self.dae_dict['z']
            p_sym = self.dae_dict['p']
            g_sym = self.dae_dict['alg']

            if isinstance(x_sym, list):
                x_sym = ca.vertcat(*x_sym)
            if isinstance(z_sym, list):
                z_sym = ca.vertcat(*z_sym)
            if isinstance(p_sym, list):
                p_sym = ca.vertcat(*p_sym)
            if isinstance(g_sym, list):
                g_sym = ca.vertcat(*g_sym)

            return ca.Function('alg', [x_sym, z_sym, p_sym], [g_sym])
        except Exception:
            return None

    def _init_from_binary(self, binary_path, dae_dict, sys_dict):
        """Initialize from a precompiled shared library."""
        import os

        if not os.path.exists(binary_path):
            raise FileNotFoundError(f"Precompiled binary not found: {binary_path}")

        self.sys_dict = sys_dict
        self.dae_dict = dae_dict
        self.integrator = None
        self._current_dt = None

        self.x_names = [sym.name() for sym in self.sys_dict['x_list']]
        self.y_ini_names = [sym.name() for sym in self.sys_dict['y_ini_list']]
        self.y_run_names = [sym.name() for sym in self.sys_dict['y_run_list']]
        self.p_names = [sym.name() for sym in self.sys_dict['p_list']]
        self.u_ini_names = [sym.name() for sym in self.sys_dict['u_ini_list']]
        self.u_run_names = [sym.name() for sym in self.sys_dict['u_run_list']]

        # Compatibility aliases (match the ctypes Model surface so reporters
        # like pydae.uds.utils.reports.report_v work on both backends).
        self.params_list = self.p_names
        self.x_list = self.x_names
        self.y_ini_list = self.y_ini_names
        self.y_run_list = self.y_run_names
        self.y_list = self.y_run_names
        self.z_list = list(self.sys_dict.get('h_dict', {}).keys())

        self.N_x = len(self.x_names)
        self.N_y_ini = len(self.y_ini_names)
        self.N_y_run = len(self.y_run_names)

        self.x = np.zeros(self.N_x)
        self.y_ini = np.zeros(self.N_y_ini)
        self.y_run = np.zeros(self.N_y_run)
        self.t = 0.0
        self.Dt = 0.01

        self.p_vals = np.array([self.sys_dict['params_dict'].get(n, 0.0) for n in self.p_names])
        self.u_ini_vals = np.array([self.sys_dict['u_ini_dict'].get(n, 0.0) for n in self.u_ini_names])
        self.u_run_vals = np.array([self.sys_dict['u_run_dict'].get(n, 0.0) for n in self.u_run_names])

        self.Time = []
        self.X = []
        self.Y = []
        self.Z = []

        xy_0_dict = self.sys_dict.get('xy_0_dict', {})
        for i, name in enumerate(self.x_names):
            if name in xy_0_dict:
                self.x[i] = xy_0_dict[name]
        for i, name in enumerate(self.y_ini_names):
            if name in xy_0_dict:
                self.y_ini[i] = xy_0_dict[name]
        for i, name in enumerate(self.y_run_names):
            if name in xy_0_dict:
                self.y_run[i] = xy_0_dict[name]

        # Load compiled functions
        self._Fx_fn = ca.external('Fx', binary_path)
        self._Fy_fn = ca.external('Fy', binary_path)
        self._Gx_fn = ca.external('Gx', binary_path)
        self._Gy_fn = ca.external('Gy', binary_path)
        self._Fu_fn = ca.external('Fu', binary_path)
        self._Gu_fn = ca.external('Gu', binary_path)

        has_hx = self._has_external_symbol(binary_path, 'Hx')
        self._Hx_fn = ca.external('Hx', binary_path) if has_hx else None
        self._Hy_fn = ca.external('Hy', binary_path) if has_hx else None
        self._Hu_fn = ca.external('Hu', binary_path) if has_hx else None

        # Load compiled functions for rootfinding (residual + jacobian)
        self._residual_fn = ca.external('residual', binary_path)
        self._jacobian_fn = ca.external('jacobian', binary_path)
        self._use_external_newton = True
        self.rf = None  # Will use _newton_solve instead

        # Output function not available in binary mode (needs h_dict reconstruction)
        self._h_fn = None

    @staticmethod
    def _has_external_symbol(binary_path, name):
        """Check if a named function exists in the external library."""
        try:
            ca.external(name, binary_path)
            return True
        except Exception:
            return False

    def _init_from_builder(self, builder):
        """Initialize from a CasadiBuilder instance."""
        self.sys_dict = builder.sys_dict
        self.dae_dict = builder.dae_dict
        self.integrator = None
        self._current_dt = None

        # Transfer residual and jacobian functions for external Newton
        if hasattr(builder, '_residual_fn') and builder._residual_fn is not None:
            self._residual_fn = builder._residual_fn
        else:
            self._residual_fn = None
        if hasattr(builder, '_jacobian_fn') and builder._jacobian_fn is not None:
            self._jacobian_fn = builder._jacobian_fn
        else:
            self._jacobian_fn = None

        # Transfer output evaluation function
        if hasattr(builder, '_h_fn') and builder._h_fn is not None:
            self._h_fn = builder._h_fn
        else:
            self._h_fn = None

        # Use builder's rootfinder by default
        if hasattr(builder, 'rf') and builder.rf is not None:
            self.rf = builder.rf
        else:
            self.rf = None

        # If user provided rootfinder_opts, recreate with custom options
        if self._rootfinder_opts and hasattr(builder, 'alg_eq') and builder.alg_eq is not None:
            try:
                self.rf = ca.rootfinder('newton', 'newton', builder.alg_eq, self._rootfinder_opts)
            except Exception:
                pass  # Keep builder's rf if recreation fails

        self.x_names = [sym.name() for sym in self.sys_dict['x_list']]
        self.y_ini_names = [sym.name() for sym in self.sys_dict['y_ini_list']]
        self.y_run_names = [sym.name() for sym in self.sys_dict['y_run_list']]
        self.p_names = [sym.name() for sym in self.sys_dict['p_list']]
        self.u_ini_names = [sym.name() for sym in self.sys_dict['u_ini_list']]
        self.u_run_names = [sym.name() for sym in self.sys_dict['u_run_list']]

        # Compatibility aliases (match the ctypes Model surface so reporters
        # like pydae.uds.utils.reports.report_v work on both backends).
        self.params_list = self.p_names
        self.x_list = self.x_names
        self.y_ini_list = self.y_ini_names
        self.y_run_list = self.y_run_names
        self.y_list = self.y_run_names
        self.z_list = list(self.sys_dict.get('h_dict', {}).keys())

        self.N_x = len(self.x_names)
        self.N_y_ini = len(self.y_ini_names)
        self.N_y_run = len(self.y_run_names)

        self.x = np.zeros(self.N_x)
        self.y_ini = np.zeros(self.N_y_ini)
        self.y_run = np.zeros(self.N_y_run)
        self.t = 0.0
        self.Dt = 0.01

        self.p_vals = np.array([self.sys_dict['params_dict'].get(n, 0.0) for n in self.p_names])
        self.u_ini_vals = np.array([self.sys_dict['u_ini_dict'].get(n, 0.0) for n in self.u_ini_names])
        self.u_run_vals = np.array([self.sys_dict['u_run_dict'].get(n, 0.0) for n in self.u_run_names])

        self.Time = []
        self.X = []
        self.Y = []
        self.Z = []

        xy_0_dict = self.sys_dict.get('xy_0_dict', {})
        for i, name in enumerate(self.x_names):
            if name in xy_0_dict:
                self.x[i] = xy_0_dict[name]
        for i, name in enumerate(self.y_ini_names):
            if name in xy_0_dict:
                self.y_ini[i] = xy_0_dict[name]
        for i, name in enumerate(self.y_run_names):
            if name in xy_0_dict:
                self.y_run[i] = xy_0_dict[name]

        self._Fx_fn = builder._Fx_fn
        self._Fy_fn = builder._Fy_fn
        self._Gx_fn = builder._Gx_fn
        self._Gy_fn = builder._Gy_fn
        self._Fu_fn = builder._Fu_fn
        self._Gu_fn = builder._Gu_fn
        self._Hx_fn = builder._Hx_fn
        self._Hy_fn = builder._Hy_fn
        self._Hu_fn = builder._Hu_fn
        self._binary_path = None

    def _route_dict(self, update_dict, phase):
        """Routes a mixed dictionary to the correct p, u_ini, u_run, x, or y arrays."""
        if not update_dict:
            return
        for key, val in update_dict.items():
            if key in self.p_names:
                self.p_vals[self.p_names.index(key)] = val
            elif phase == 'ini' and key in self.u_ini_names:
                self.u_ini_vals[self.u_ini_names.index(key)] = val
            elif phase == 'run' and key in self.u_run_names:
                self.u_run_vals[self.u_run_names.index(key)] = val
            elif key in self.y_ini_names:
                self.y_ini[self.y_ini_names.index(key)] = val
            elif key in self.y_run_names:
                self.y_run[self.y_run_names.index(key)] = val
            elif key in self.x_names:
                self.x[self.x_names.index(key)] = val

    def _newton_solve(self, x0, p_vec, max_iter=None, tol=None):
        """Solve the DAE residual using exported residual and jacobian functions.

        Performs Newton iterations: x_{k+1} = x_k - J^{-1} * r(x_k)

        Parameters
        ----------
        x0 : array
            Initial guess.
        p_vec : array
            Parameter vector.
        max_iter : int, optional
            Maximum Newton iterations (uses self._newton_max_iter if None).
        tol : float, optional
            Convergence tolerance on residual norm (uses self._newton_tol if None).

        Returns
        -------
        array
            Solved variable vector.
        """
        if max_iter is None:
            max_iter = self._newton_max_iter
        if tol is None:
            tol = self._newton_tol

        x = np.array(x0, dtype=float).flatten()
        for _ in range(max_iter):
            r = np.array(self._residual_fn(x, p_vec)).flatten()
            norm_r = np.linalg.norm(r)
            if norm_r < tol:
                break
            J = np.array(self._jacobian_fn(x, p_vec))
            dx = np.linalg.solve(J, -r)
            x = x + dx
        return x

    def _calc_ic_init(self, v_ini_guess, p_ini_vec):
        """Use IDAS calc_ic to find consistent initial conditions.

        Fallback when the Newton rootfinder fails (e.g., due to NaN
        from y_run-only variables that only appear in runtime equations).
        """
        # We need to construct an initialization DAE that includes ALL
        # variables (x, y_ini, and y_run-only). Since the rootfinder was
        # built with y_run-only vars substituted, we use the full dae_dict
        # but with calc_ic=2 (solve for consistent IC with fixed x).

        # First, populate y_run with xy_0 guesses for y_run-only vars
        xy_0_dict = self.sys_dict.get('xy_0_dict', {})
        for i, name in enumerate(self.y_run_names):
            if name not in self.y_ini_names and name in xy_0_dict:
                self.y_run[i] = xy_0_dict[name]
            elif name not in self.y_ini_names:
                self.y_run[i] = 1.0 if name.startswith(('V_', 'theta_')) else 0.0

        # Also need to set x from the guess
        self.x = v_ini_guess[:self.N_x].copy()

        p_run_vec = np.concatenate((self.p_vals, self.u_run_vals))

        opts = {
            'calc_ic': 1,
            'print_stats': False,
            'max_num_steps': 5000,
            'reltol': self._integrator_reltol,
            'abstol': self._integrator_abstol,
        }
        ic_int = ca.integrator('ic_init', 'idas', self.dae_dict, 0.0, 1e-12, opts)
        res = ic_int(x0=self.x, z0=self.y_run, p=p_run_vec)
        self.x = np.array(res['xf']).flatten()
        self.y_run = np.array(res['zf']).flatten()

        # Return concatenated solution (x, y_ini)
        # Since y_ini vars may differ from y_run, map back by name
        v_solved = np.zeros(self.N_x + self.N_y_ini)
        v_solved[:self.N_x] = self.x
        for i, name in enumerate(self.y_ini_names):
            if name in self.y_run_names:
                v_solved[self.N_x + i] = self.y_run[self.y_run_names.index(name)]
            else:
                v_solved[self.N_x + i] = v_ini_guess[self.N_x + i]

        return v_solved

    def ini2run(self):
        """Transforms initialization states into runtime states."""

        # 1. Build the master value map from the newly solved initialization
        val_map = {name: val for name, val in zip(self.x_names, self.x)}
        val_map.update({name: val for name, val in zip(self.y_ini_names, self.y_ini)})
        val_map.update({name: val for name, val in zip(self.u_ini_names, self.u_ini_vals)})

        # 2. Map solved states to y_run by NAME matching
        for i, name in enumerate(self.y_run_names):
            if name in val_map:
                self.y_run[i] = val_map[name]

        # 3. Map solved references to u_run (Inverse Initialization)
        yini2urun_names = getattr(self, 'yini2urun', [])
        for i, name in enumerate(self.u_run_names):
            if name in yini2urun_names and name in val_map:
                self.u_run_vals[i] = val_map[name]

        # 4. Map u_ini to y_run for variables that are inputs during init
        # but states/outputs during run
        uini2yrun_names = getattr(self, 'uini2yrun', [])
        for i, name in enumerate(self.y_run_names):
            if name in uini2yrun_names and name in val_map:
                self.y_run[i] = val_map[name]

        # 5. Map u_ini to u_run for variables that are inputs in both phases
        uini2urun_names = list(set(self.u_ini_names).intersection(set(self.u_run_names)))
        for i, name in enumerate(self.u_run_names):
            if name in uini2urun_names and name in val_map:
                self.u_run_vals[i] = val_map[name]

        # 6. Auto-swap: y_ini variables that also appear in u_run.
        # These are solved during initialization (e.g. v_ref_G*) but become
        # fixed inputs during runtime. Transfer their solved values.
        yini_and_urun = set(self.y_ini_names).intersection(set(self.u_run_names))
        for name in yini_and_urun:
            if name in val_map:
                idx = self.u_run_names.index(name)
                self.u_run_vals[idx] = val_map[name]

        # 7. Lock in the unified state vector for the CasADi runtime integrator
        self.xy = np.concatenate((self.x, self.y_run))
        self.xy[:self.N_x] = self.x
        self.xy[self.N_x:] = self.y_run

    def recalculate_algebraics(self, tol=1e-8):
        """Re-solve algebraic variables after a parameter discontinuity.

        Use this method when parameters change mid-simulation (e.g., fault
        impedance switching). State variables (x) remain fixed while algebraic
        variables (y_run) jump instantaneously to satisfy g(x, y, p) = 0.

        Parameters
        ----------
        tol : float, optional
            Convergence tolerance for the rootfinder (default: 1e-8).

        Raises
        ------
        RuntimeError
            If the rootfinder fails to converge.
        """
        if self._alg_fn is None:
            raise RuntimeError("Algebraic function (_alg_fn) not available")

        p_run_vec = np.concatenate((self.p_vals, self.u_run_vals))

        def alg_residual(y_guess):
            g_val = self._alg_fn(self.x, y_guess, p_run_vec)
            return np.array(g_val).flatten()

        sol = root(alg_residual, self.y_run, method='hybr', tol=tol)

        if sol.success:
            self.y_run = sol.x.reshape(self.y_run.shape)
        else:
            raise RuntimeError(sol.message)

    def use_external_newton(self, enabled=True):
        """Switch to/from external Newton solver for initialization."""
        self._use_external_newton = enabled

    def ini(self, params_dict, xy_0=None, newton_tol=None, newton_max_iter=None):
        """Initializes the model matching the current pydae API.

        Parameters
        ----------
        params_dict : dict
            Parameters to update.
        xy_0 : str or dict, optional
            Initial guess file path or dictionary.
        newton_tol : float, optional
            Override Newton tolerance for this solve.
        newton_max_iter : int, optional
            Override Newton max iterations for this solve.
        """
        self._route_dict(params_dict, phase='ini')

        p_ini_vec = np.concatenate((self.p_vals, self.u_ini_vals))
        v_ini_guess = np.concatenate((self.x, self.y_ini))

        # Apply explicit user guesses
        if isinstance(xy_0, str):
            with open(xy_0) as fobj:
                xy_0_dict = json.load(fobj)
            for key, val in xy_0_dict.items():
                if key in self.x_names:
                    v_ini_guess[self.x_names.index(key)] = val
                elif key in self.y_ini_names:
                    v_ini_guess[self.N_x + self.y_ini_names.index(key)] = val
        elif isinstance(xy_0, dict):
            for key, val in xy_0.items():
                if key in self.x_names:
                    v_ini_guess[self.x_names.index(key)] = val
                elif key in self.y_ini_names:
                    v_ini_guess[self.N_x + self.y_ini_names.index(key)] = val

        # Solve Initialization
        tol = newton_tol if newton_tol is not None else self._newton_tol
        max_iter = newton_max_iter if newton_max_iter is not None else self._newton_max_iter
        use_external = getattr(self, '_use_external_newton', False)
        if use_external or self.rf is None:
            v_solved = self._newton_solve(v_ini_guess, p_ini_vec, max_iter=max_iter, tol=tol)
        else:
            # Fallback chain (all stages use CasADi AD for the Jacobian — no
            # hand-derived analytic Jacobian anywhere):
            #   1. CasADi 'newton' rootfinder (fast path)
            #   2. _newton_solve: a thin Python loop over the same CasADi
            #      AD-derived _jacobian_fn / _residual_fn; tolerates poorer
            #      seeds than the 'newton' plugin and unlike KINSOL needs no
            #      hand-tuned variable scaling on mixed-magnitude DAEs.
            #   3. IDAS calc_ic (last resort).
            try:
                res = self.rf(x0=v_ini_guess, p=p_ini_vec)
                v_solved = np.array(res['x']).flatten()
            except RuntimeError:
                try:
                    v_solved = self._newton_solve(v_ini_guess, p_ini_vec,
                                                  max_iter=max_iter, tol=tol)
                except (RuntimeError, np.linalg.LinAlgError):
                    v_solved = self._calc_ic_init(v_ini_guess, p_ini_vec)

        # Update initialization arrays
        self.x = v_solved[:self.N_x]
        self.y_ini = v_solved[self.N_x:]
        self.t = 0.0

        # --- THIS IS WHERE INI2RUN GOES ---
        # It takes the solved x and y_ini, and builds y_run and u_run
        self.ini2run()

        # Reset storage (Now Y will correctly store the t=0 y_run values)
        self.Time = [self.t]
        self.X = [self.x.copy()]
        self.Y = [self.y_run.copy()]
        if self._h_fn is not None and 'h_dict' in self.sys_dict:
            p_run_vec = np.concatenate((self.p_vals, self.u_run_vals))
            z0 = np.array(self._h_fn(self.x, self.y_run, p_run_vec)).flatten()
            self.Z = [z0.copy()]
        else:
            self.Z = []


    def run(self, t_end, update_dict=None):
        """Advances simulation, appending to internal storage."""
        self._route_dict(update_dict, phase='run')

        if self._current_dt != self.Dt:
            opts = {
                'calc_ic': True,
                'print_stats': False,
                'max_num_steps': 5000,
                'reltol': self._integrator_reltol,
                'abstol': self._integrator_abstol,
                'linear_solver': 'csparse'  
            }
            self.integrator = ca.integrator('idas_int', 'idas', self.dae_dict, 0.0, self.Dt, opts)
            self._current_dt = self.Dt

        p_run_vec = np.concatenate((self.p_vals, self.u_run_vals))

        while self.t < t_end - 1e-9:
            res = self.integrator(x0=self.x, z0=self.y_run, p=p_run_vec)

            self.x = np.array(res['xf']).flatten()
            self.y_run = np.array(res['zf']).flatten()
            self.t += self.Dt

            self.Time.append(self.t)
            self.X.append(self.x.copy())
            self.Y.append(self.y_run.copy())
            if self._h_fn is not None and 'h_dict' in self.sys_dict:
                p_run_vec = np.concatenate((self.p_vals, self.u_run_vals))
                z_val = np.array(self._h_fn(self.x, self.y_run, p_run_vec)).flatten()
                self.Z.append(z_val.copy())

    def post(self):
        """Converts lists to NumPy arrays for analysis/plotting."""
        self.Time = np.array(self.Time)
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        if self.Z:
            self.Z = np.array(self.Z)

    def A_eval(self):
        """Computes the dense State Matrix (A) at the current operating point.

        Evaluates sparse Jacobians F_x, F_y, G_x, G_y numerically, then
        computes A = F_x - F_y @ solve(G_y, G_x) in NumPy.
        """
        p_run_vec = np.concatenate((self.p_vals, self.u_run_vals))
        Fx = np.array(self._Fx_fn(self.x, self.y_run, p_run_vec))
        Fy = np.array(self._Fy_fn(self.x, self.y_run, p_run_vec))
        Gx = np.array(self._Gx_fn(self.x, self.y_run, p_run_vec))
        Gy = np.array(self._Gy_fn(self.x, self.y_run, p_run_vec))
        self.A = Fx - Fy @ np.linalg.solve(Gy, Gx)
        return self.A

    def BCD_eval(self):
        """Compute the input-output matrices B, C, and D.

        Evaluates sparse Jacobians numerically, then computes:
          B = F_u - F_y @ solve(G_y, G_u)
          C = H_x - H_y @ solve(G_y, G_x)
          D = H_u - H_y @ solve(G_y, G_u)
        """
        p_run_vec = np.concatenate((self.p_vals, self.u_run_vals))
        Fy = np.array(self._Fy_fn(self.x, self.y_run, p_run_vec))
        Fu = np.array(self._Fu_fn(self.x, self.y_run, p_run_vec))
        Gy = np.array(self._Gy_fn(self.x, self.y_run, p_run_vec))
        Gu = np.array(self._Gu_fn(self.x, self.y_run, p_run_vec))

        Gy_inv = np.linalg.inv(Gy)
        self.B = Fu - Fy @ Gy_inv @ Gu

        if self._Hx_fn is not None:
            Hx = np.array(self._Hx_fn(self.x, self.y_run, p_run_vec))
            Hy = np.array(self._Hy_fn(self.x, self.y_run, p_run_vec))
            Hu = np.array(self._Hu_fn(self.x, self.y_run, p_run_vec))
            Gx = np.array(self._Gx_fn(self.x, self.y_run, p_run_vec))
            self.C = Hx - Hy @ Gy_inv @ Gx
            self.D = Hu - Hy @ Gy_inv @ Gu
        else:
            n_u = len(self.u_run_names)
            self.C = np.empty((0, self.N_x))
            self.D = np.empty((0, n_u))

        return self.B, self.C, self.D

    def eval_eigenvalues(self):
        """Computes the eigenvalues of the system at the current operating point."""
        A_matrix = self.A_eval()
        return np.linalg.eigvals(A_matrix)

    def get_value(self, name):
        """Get the current value of a variable by name."""
        if name in self.x_names:
            return self.x[self.x_names.index(name)]
        elif name in self.y_run_names:
            return self.y_run[self.y_run_names.index(name)]
        elif name in self.p_names:
            return self.p_vals[self.p_names.index(name)]
        elif name in self.u_run_names:
            return self.u_run_vals[self.u_run_names.index(name)]
        elif name in self.y_ini_names:
            return self.y_ini[self.y_ini_names.index(name)]
        elif name in self.u_ini_names:
            return self.u_ini_vals[self.u_ini_names.index(name)]
        elif hasattr(self, '_h_fn') and self._h_fn is not None and hasattr(self, 'sys_dict') and 'h_dict' in self.sys_dict and name in self.sys_dict['h_dict']:
            return self._eval_output(name)
        else:
            raise ValueError(f"Variable '{name}' not found in model")

    def get_mvalue(self, names):
        """Get multiple scalar values at once. Returns a list in input order."""
        return [self.get_value(n) for n in names]

    def set_value(self, name, value):
        """Set the value of a parameter, input, or variable by name."""
        if name in self.p_names:
            self.p_vals[self.p_names.index(name)] = value
        elif name in self.u_run_names:
            self.u_run_vals[self.u_run_names.index(name)] = value
        elif name in self.u_ini_names:
            self.u_ini_vals[self.u_ini_names.index(name)] = value
        elif name in self.x_names:
            self.x[self.x_names.index(name)] = value
        elif name in self.y_run_names:
            self.y_run[self.y_run_names.index(name)] = value
        elif name in self.y_ini_names:
            self.y_ini[self.y_ini_names.index(name)] = value
        else:
            raise ValueError(f"Variable '{name}' not found in model")

    def _eval_output(self, name):
        """Evaluate an output expression (from h_dict) at current operating point."""
        import numpy as np

        if hasattr(self, '_h_fn') and self._h_fn is not None:
            # Use compiled CasADi function
            h_all = np.array(self._h_fn(self.x, self.y_run, np.concatenate((self.p_vals, self.u_run_vals)))).flatten()
            h_names = list(self.sys_dict['h_dict'].keys())
            if name in h_names:
                return h_all[h_names.index(name)]
        return 0.0

    def get_values(self, name):
        """Get the time series of a variable by name."""
        if name in self.x_names:
            idx = self.x_names.index(name)
            return np.array([x[idx] for x in self.X])
        elif name in self.y_run_names:
            idx = self.y_run_names.index(name)
            return np.array([y[idx] for y in self.Y])
        elif (
            hasattr(self, '_h_fn')
            and self._h_fn is not None
            and 'h_dict' in self.sys_dict
            and name in self.sys_dict['h_dict']
        ):
            h_names = list(self.sys_dict['h_dict'].keys())
            idx = h_names.index(name)
            return self.Z[:, idx]
        else:
            raise ValueError(f"Variable '{name}' not found in model storage")

    def report_x(self):
        """Print current state values."""
        print("\nStates (x):")
        for name, val in zip(self.x_names, self.x):
            print(f"  {name}: {val:.6e}")

    def report_y(self):
        """Print current algebraic values."""
        print("\nAlgebraic (y_run):")
        for name, val in zip(self.y_run_names, self.y_run):
            print(f"  {name}: {val:.6e}")

    def report_z(self):
        """Print current output values z = h(x, y, u, params)."""
        print("\nOutputs (z):")
        if hasattr(self, '_h_fn') and self._h_fn is not None and 'h_dict' in self.sys_dict:
            h_names = list(self.sys_dict['h_dict'].keys())
            h_vals = np.array(self._h_fn(self.x, self.y_run, np.concatenate((self.p_vals, self.u_run_vals)))).flatten()
            for name, val in zip(h_names, h_vals):
                print(f"  {name}: {val:.6e}")
        else:
            print("  (no outputs defined)")

    def report_u(self):
        """Print current input values."""
        print("\nInputs (u_run):")
        for name, val in zip(self.u_run_names, self.u_run_vals):
            print(f"  {name}: {val:.6e}")

    def report_params(self):
        """Print current parameter values."""
        print("\nParameters (p):")
        for name, val in zip(self.p_names, self.p_vals):
            print(f"  {name}: {val:.6e}")


    @classmethod
    def from_jit(cls, jit_paths, metadata_path):
        """Load a precompiled model from JIT-compiled binaries and metadata JSON.

        Parameters
        ----------
        jit_paths : dict
            Mapping of function name to compiled binary path, e.g.
            {'residual': 'residual.dll', 'Fx': 'Fx.dll', ...}.
        metadata_path : str
            Path to the metadata JSON saved by save_metadata().

        Returns
        -------
        CasadiModel
            Model instance with functions loaded from JIT binaries.
        """
        import os

        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path) as f:
            meta = json.load(f)

        # Convert string param values back to numeric defaults
        def _clean_dict(d, target_keys):
            result = {}
            for k in target_keys:
                v = d.get(k, 0.0)
                result[k] = 0.0 if isinstance(v, str) else v
            return result

        params_dict = _clean_dict(meta.get('params_dict', {}), meta['p_list'])
        u_ini_dict = _clean_dict(meta.get('u_ini_dict', {}), meta['u_ini_list'])
        u_run_dict = _clean_dict(meta.get('u_run_dict', {}), meta['u_run_list'])

        dae_dict = {
            'x': [ca.SX.sym(name) for name in meta['x_list']],
            'z': [ca.SX.sym(name) for name in meta['y_run_list']],
            'p': [ca.SX.sym(name) for name in meta['p_list'] + meta['u_run_list']],
            'ode': meta.get('f_list', []),
            'alg': meta.get('g_list', []),
        }

        sys_dict = {
            'name': meta.get('name', 'unknown'),
            'x_list': [ca.SX.sym(name) for name in meta['x_list']],
            'y_ini_list': [ca.SX.sym(name) for name in meta['y_ini_list']],
            'y_run_list': [ca.SX.sym(name) for name in meta['y_run_list']],
            'p_list': [ca.SX.sym(name) for name in meta['p_list']],
            'u_ini_list': [ca.SX.sym(name) for name in meta['u_ini_list']],
            'u_run_list': [ca.SX.sym(name) for name in meta['u_run_list']],
            'params_dict': params_dict,
            'u_ini_dict': u_ini_dict,
            'u_run_dict': u_run_dict,
            'xy_0_dict': meta.get('xy_0_dict', {}),
            'h_dict': meta.get('h_dict', {}),
            '_jit_paths': jit_paths,
        }

        obj = cls(binary_path=jit_paths['residual'], dae_dict=dae_dict, sys_dict=sys_dict)

        # Override individual functions with JIT-compiled binaries
        obj._Fx_fn = ca.external('Fx', jit_paths['Fx'])
        obj._Fy_fn = ca.external('Fy', jit_paths['Fy'])
        obj._Gx_fn = ca.external('Gx', jit_paths['Gx'])
        obj._Gy_fn = ca.external('Gy', jit_paths['Gy'])
        obj._Fu_fn = ca.external('Fu', jit_paths['Fu'])
        obj._Gu_fn = ca.external('Gu', jit_paths['Gu'])
        obj._residual_fn = ca.external('residual', jit_paths['residual'])
        obj._jacobian_fn = ca.external('jacobian', jit_paths['jacobian'])

        if 'Hx' in jit_paths and os.path.exists(jit_paths['Hx']):
            obj._Hx_fn = ca.external('Hx', jit_paths['Hx'])
            obj._Hy_fn = ca.external('Hy', jit_paths['Hy'])
            obj._Hu_fn = ca.external('Hu', jit_paths['Hu'])
        else:
            obj._Hx_fn = None
            obj._Hy_fn = None
            obj._Hu_fn = None

        return obj

    @classmethod
    def from_binary(cls, binary_path, metadata_path):
        """Load a precompiled model from a binary and its metadata JSON."""
        import os

        if not os.path.exists(binary_path):
            raise FileNotFoundError(f"Binary not found: {binary_path}")

        with open(metadata_path) as f:
            meta = json.load(f)

        def _clean_dict(d, target_keys):
            """Convert string values to 0.0 numeric defaults."""
            result = {}
            for k in target_keys:
                v = d.get(k, 0.0)
                result[k] = 0.0 if isinstance(v, str) else v
            return result

        params_dict = _clean_dict(meta.get('params_dict', {}), meta['p_list'])
        u_ini_dict = _clean_dict(meta.get('u_ini_dict', {}), meta['u_ini_list'])
        u_run_dict = _clean_dict(meta.get('u_run_dict', {}), meta['u_run_list'])

        dae_dict = {
            'x': [ca.SX.sym(name) for name in meta['x_list']],
            'z': [ca.SX.sym(name) for name in meta['y_run_list']],
            'p': [ca.SX.sym(name) for name in meta['p_list'] + meta['u_run_list']],
            'ode': meta.get('f_list', []),
            'alg': meta.get('g_list', []),
        }

        sys_dict = {
            'name': meta.get('name', 'unknown'),
            'x_list': [ca.SX.sym(name) for name in meta['x_list']],
            'y_ini_list': [ca.SX.sym(name) for name in meta['y_ini_list']],
            'y_run_list': [ca.SX.sym(name) for name in meta['y_run_list']],
            'p_list': [ca.SX.sym(name) for name in meta['p_list']],
            'u_ini_list': [ca.SX.sym(name) for name in meta['u_ini_list']],
            'u_run_list': [ca.SX.sym(name) for name in meta['u_run_list']],
            'params_dict': params_dict,
            'u_ini_dict': u_ini_dict,
            'u_run_dict': u_run_dict,
            'xy_0_dict': meta.get('xy_0_dict', {}),
            'h_dict': meta.get('h_dict', {}),
        }

        return cls(binary_path=binary_path, dae_dict=dae_dict, sys_dict=sys_dict)

    @classmethod
    def load_or_build(cls, builder, binary_path=None, c_path=None):
        """Load a precompiled binary if available, otherwise build from the builder.

        Parameters
        ----------
        builder : CasadiBuilder
            Builder instance with compiled sys_dict and evaluator functions.
        binary_path : str, optional
            Path to the precompiled shared library (.so/.dll). If None,
            derives from c_path or defaults to '{name}_compiled.so'.
        c_path : str, optional
            Path to the C source file. Used to derive binary_path if not given.

        Returns
        -------
        CasadiModel
            Model instance loaded from binary or built from scratch.
        """
        import os

        if binary_path is None:
            if c_path is not None:
                stem = os.path.splitext(os.path.basename(c_path))[0]
                ext = '.dll' if os.name == 'nt' else '.so'
                binary_path = os.path.join(os.path.dirname(c_path), f'{stem}{ext}')
            else:
                ext = '.dll' if os.name == 'nt' else '.so'
                binary_path = f"{builder.sys_dict['name']}_compiled{ext}"

        if os.path.exists(binary_path):
            print(f"Loading precompiled model from: {binary_path}")
            return cls(
                binary_path=binary_path,
                dae_dict=builder.dae_dict,
                sys_dict=builder.sys_dict,
            )

        print(f"Binary not found at {binary_path}, falling back to symbolic build.")
        return cls(builder=builder)

    def save_metadata(self, path):
        """Save sys_dict metadata (without symbolic objects) for later binary loading.

        Serializes all dictionary-based metadata so that CasadiModel can be
        reconstructed from a precompiled binary without needing the original builder.

        Parameters
        ----------
        path : str
            Output JSON file path.
        """
        import copy

        meta = copy.deepcopy(self.sys_dict)

        def _sym_to_str(obj):
            if hasattr(obj, 'name'):
                return obj.name()
            return str(obj)

        meta['x_list'] = [_sym_to_str(s) for s in self.sys_dict['x_list']]
        meta['y_ini_list'] = [_sym_to_str(s) for s in self.sys_dict['y_ini_list']]
        meta['y_run_list'] = [_sym_to_str(s) for s in self.sys_dict['y_run_list']]
        meta['p_list'] = [_sym_to_str(s) for s in self.sys_dict['p_list']]
        meta['u_ini_list'] = [_sym_to_str(s) for s in self.sys_dict['u_ini_list']]
        meta['u_run_list'] = [_sym_to_str(s) for s in self.sys_dict['u_run_list']]

        with open(path, 'w') as f:
            json.dump(meta, f, indent=2, default=str)
