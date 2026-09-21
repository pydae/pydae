import json
import warnings

import casadi as ca
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.optimize import root


class _TrapLinBackend:
    """Pluggable linear-solver backend for the fixed-step Newton iteration.

    The iteration matrix ``J`` has a *fixed sparsity pattern* for the whole
    simulation, so the solve follows a factorize-once / solve-many contract:

    - ``refresh(J_dm)`` — (re)factorize from a CasADi ``DM`` Jacobian.
    - ``solve(neg_R)`` — solve ``J @ dw = neg_R`` reusing the stored
      factorization; ``neg_R`` is a 1-D ndarray, returns a 1-D ndarray.
    - ``stale`` / ``mark_stale()`` — whether the factorization must be refreshed
      before the next solve (set after a discontinuity or an adaptive drop).
    """

    def __init__(self):
        self._stale = True

    @property
    def stale(self):
        return self._stale

    def mark_stale(self):
        self._stale = True

    def refresh(self, J_dm):
        raise NotImplementedError

    def solve(self, neg_R):
        raise NotImplementedError


class _ScipyLU(_TrapLinBackend):
    """SciPy sparse LU backend (``scipy.sparse.linalg.splu``). Default; converts
    the CasADi ``DM`` Jacobian to ``scipy.sparse.csc`` before factorizing."""

    def __init__(self):
        super().__init__()
        self._lu = None

    def refresh(self, J_dm):
        r, c = J_dm.sparsity().get_triplet()
        A = sp.csc_matrix((np.array(J_dm.nonzeros()), (r, c)),
                          shape=(J_dm.size1(), J_dm.size2()))
        self._lu = spla.splu(A)
        self._stale = False

    def solve(self, neg_R):
        return self._lu.solve(neg_R)


class _CasadiLinsol(_TrapLinBackend):
    """CasADi-native ``ca.Linsol`` backend using the sfact/nfact/solve reuse
    trio: symbolic factorization once (the sparsity is fixed), numeric
    factorization on ``refresh``, cheap back-substitution on each ``solve``.
    Stays in CasADi ``DM`` — no scipy dependency, no format conversion.

    Recommended plugin: ``'qr'`` (in this factorize-once/solve-many pattern it
    measured 10–60x faster per step than ``'csparse'`` or ``'lapacklu'``)."""

    def __init__(self, sparsity, plugin='qr'):
        super().__init__()
        self._lin = ca.Linsol('trap_lin', plugin, sparsity)
        self._sfact_done = False
        self._Jdm = None

    def refresh(self, J_dm):
        if not self._sfact_done:
            self._lin.sfact(J_dm)          # symbolic factorization ONCE
            self._sfact_done = True
        self._lin.nfact(J_dm)              # numeric factorization
        self._Jdm = J_dm
        self._stale = False

    def solve(self, neg_R):
        dw = self._lin.solve(self._Jdm, ca.DM(neg_R))   # reuses nfact
        return np.array(dw).flatten()


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

        # Time-domain solver selection & fixed-step solver state.
        # Default 'idas_grid': one IDAS call over a grid of output times
        # (~30x faster than the legacy per-Dt loop, identical trajectory).
        self.run_method = 'idas_grid'
        self._grid_integrator = None
        self._grid_key = None
        self._fixed_step_fns = None
        self._fixed_step_alpha = None
        self._lin_backend = None       # pluggable linear-solver backend (scipy / ca.Linsol)
        self._lin_backend_key = None   # (alpha, linsol) the backend was built for
        self._sc = 0                   # fixed-step step counter (drives scheduled refactors)

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
        self.u_ini_list = self.u_ini_names
        self.u_run_list = self.u_run_names
        self.z_list = list(self.sys_dict.get('h_dict', {}).keys())
        self.outputs_list = self.z_list

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
        self.u_ini_list = self.u_ini_names
        self.u_run_list = self.u_run_names
        self.z_list = list(self.sys_dict.get('h_dict', {}).keys())
        self.outputs_list = self.z_list

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
            'max_num_steps': 50,
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

        **The trust region is load-bearing — do not swap ``hybr`` for a plain
        Newton.** It looks like the wrong tool (every caller arrives with an
        excellent initial guess, and densifying the analytic Jacobian at the
        ``np.array`` below throws away a ~97 % sparse ``Gy``), but it was tried
        and measured: on a faulted distribution network the algebraic system is
        badly scaled — ``cond(Gy) ~ 1e10``, from branch currents in amperes
        sitting alongside voltages in pu — while remaining non-singular. The
        undamped Newton direction is then genuinely poor: with the residual
        concentrated on two bus equations it asks for a step of ~34 in a branch
        current, and an Armijo backtracking line search on ``||g||^2`` creeps
        instead of converging. MINPACK's dogleg caps the step and degrades
        toward steepest descent exactly where that happens.

        A sparse replacement therefore needs a *trust region* (Levenberg-
        Marquardt or dogleg) plus variable scaling, not a Newton iteration.

        Parameters
        ----------
        tol : float, optional
            Rootfinder tolerance (default: 1e-8). Note ``root`` forwards this to
            ``hybr`` as ``xtol`` — a *step* tolerance, not a residual bound.

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

        # Supply the analytic algebraic Jacobian (dg/dy) when available. The
        # default finite-difference Jacobian is poorly scaled on large/stiff
        # algebraic systems and makes 'hybr' stall ("not making good progress")
        # even for small parameter jumps.
        alg_jac = None
        if getattr(self, '_Gy_fn', None) is not None:
            def alg_jac(y_guess):
                return np.array(self._Gy_fn(self.x, y_guess, p_run_vec))

        sol = root(alg_residual, self.y_run, jac=alg_jac, method='hybr', tol=tol)

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


    def run(self, t_end, update_dict=None, method=None, **opts):
        """Advance the time-domain simulation to ``t_end``, appending each output
        step to the internal ``Time/X/Y/Z`` storage and advancing ``x/y_run/t``.

        Parameters
        ----------
        t_end : float
            Absolute end time of this segment.
        update_dict : dict, optional
            Name -> value overrides applied via :meth:`_route_dict` before
            stepping (same semantics as the previous ``run(t_end, inputs)``).
        method : {'idas_grid', 'idas', 'trapezoidal', 'backward_euler'}, optional
            Time-domain integration strategy. ``None`` falls back to
            ``self.run_method`` (default ``'idas_grid'``).

            - ``'idas_grid'`` — one IDAS call over a grid of output times
              ``t + Dt*arange(1, n+1)``. SUNDIALS initialises once and reuses its
              factorization across internal steps: ~30x faster than ``'idas'``
              with an identical trajectory. Use per *continuous* segment; call
              :meth:`recalculate_algebraics` after a ``set_value`` discontinuity.
            - ``'idas'`` — legacy stateless IDAS call once per ``Dt``. Each call
              cold-restarts the BDF method (order 1, tiny step, re-factorize), so
              it is slow, but bit-for-bit reproduces the historical behaviour.
            - ``'trapezoidal'`` — fixed-step implicit trapezoid (``alpha=0.5``).
            - ``'backward_euler'`` — fixed-step implicit Euler (``alpha=1.0``).

            The fixed-step solvers build the iteration matrix from CasADi AD,
            factorize it once with SciPy ``splu`` and reuse it (modified Newton):
            constant per-step cost, no adaptive restart. They require a consistent
            starting point — run :meth:`ini` first, and
            :meth:`recalculate_algebraics` after any discontinuity.

        Other Parameters
        ----------------
        **opts
            Method-specific knobs. ``'idas_grid'``/``'idas'``: ``step0`` (initial
            step-size hint), ``calc_ic`` (bool), ``reuse_integrator`` (bool).
            Fixed-step: ``alpha``, ``tol``, ``max_iter``, ``refactor_every``,
            ``refactor_iters``, and ``linsol`` — the linear solver for the Newton
            step. ``linsol='scipy'`` (default) uses ``scipy.sparse.linalg.splu``;
            ``linsol='qr'`` (or any ``ca.Linsol`` plugin, or ``'casadi:<plugin>'``)
            uses a CasADi-native solver with the sfact/nfact/solve reuse trio
            (symbolic factorization once, numeric factorization on refresh, cheap
            back-substitution per iteration). ``'qr'`` is the recommended CasADi
            plugin; an unavailable plugin falls back to SciPy with a warning. The
            linear solve is a small fraction of a step (per-iteration CasADi
            ``Function`` calls dominate), so the CasADi backend is mainly a
            cleanliness / no-scipy-conversion win, not a large end-to-end speedup.
        """
        if method is None:
            method = getattr(self, 'run_method', 'idas_grid')
        self._route_dict(update_dict, phase='run')

        if method == 'idas_grid':
            self._run_idas_grid(t_end, **opts)
        elif method == 'idas':
            self._run_idas_perdt(t_end, **opts)
        elif method in ('trapezoidal', 'backward_euler'):
            alpha = opts.pop('alpha', 0.5 if method == 'trapezoidal' else 1.0)
            self._run_fixed_step(t_end, alpha=alpha, **opts)
        else:
            raise ValueError(
                f"unknown run method {method!r}; expected one of "
                "'idas_grid', 'idas', 'trapezoidal', 'backward_euler'")

    def _append_output(self, xk, zk, tk, p_run_vec):
        """Append one solved output point to the Time/X/Y/Z storage."""
        self.Time.append(float(tk))
        self.X.append(np.asarray(xk).flatten().copy())
        self.Y.append(np.asarray(zk).flatten().copy())
        if self._h_fn is not None and 'h_dict' in self.sys_dict:
            self.Z.append(np.array(self._h_fn(xk, zk, p_run_vec)).flatten())

    def _run_idas_perdt(self, t_end):
        """Legacy solver: a stateless IDAS call once per ``Dt``. Kept for exact
        reproducibility of the historical ``run`` behaviour (BDF cold-start each
        step)."""
        if self._current_dt != self.Dt:
            opts = {
                'calc_ic': True,
                'print_stats': False,
                'max_num_steps': 5000,
                'reltol': self._integrator_reltol,
                'abstol': self._integrator_abstol,
                'linear_solver': 'csparse',
            }
            self.integrator = ca.integrator('idas_int', 'idas', self.dae_dict, 0.0, self.Dt, opts)
            self._current_dt = self.Dt

        p_run_vec = np.concatenate((self.p_vals, self.u_run_vals))
        while self.t < t_end - 1e-9:
            res = self.integrator(x0=self.x, z0=self.y_run, p=p_run_vec)
            self.x = np.array(res['xf']).flatten()
            self.y_run = np.array(res['zf']).flatten()
            self.t += self.Dt
            self._append_output(self.x, self.y_run, self.t, p_run_vec)

    def _run_idas_grid(self, t_end, step0=None, calc_ic=None, reuse_integrator=False):
        """Fast solver: integrate to ``t_end`` in a SINGLE IDAS call over a grid
        of output times. SUNDIALS initialises once, ramps BDF up, and reuses its
        factorization across internal steps instead of cold-restarting per ``Dt``.

        The DAE is autonomous, so integration is done over a *relative* grid
        ``[Dt, 2Dt, ..., n*Dt]`` from t0=0 and shifted by the current time; this
        lets the integrator object be reused across equal-length segments
        (``reuse_integrator=True``).
        """
        Dt = self.Dt
        n = int(round((t_end - self.t) / Dt))
        if n <= 0:
            return
        tout_rel = Dt * np.arange(1, n + 1)

        opts = {
            'print_stats': False,
            'max_num_steps': 100000,
            'reltol': self._integrator_reltol,
            'abstol': self._integrator_abstol,
            'linear_solver': 'csparse',
        }
        if calc_ic is not None:
            opts['calc_ic'] = calc_ic
        if step0 is not None:
            opts['step0'] = step0

        if reuse_integrator:
            key = (n, Dt, step0, calc_ic)
            if self._grid_key != key:
                self._grid_integrator = ca.integrator(
                    'grid', 'idas', self.dae_dict, 0.0, tout_rel, opts)
                self._grid_key = key
            integ = self._grid_integrator
        else:
            integ = ca.integrator('grid', 'idas', self.dae_dict, 0.0, tout_rel, opts)

        p_run_vec = np.concatenate((self.p_vals, self.u_run_vals))
        res = integ(x0=self.x, z0=self.y_run, p=p_run_vec)
        Xf, Zf = np.array(res['xf']), np.array(res['zf'])   # (n_x, n) and (n_z, n)

        t0 = self.t
        for k in range(n):
            self._append_output(Xf[:, k], Zf[:, k], t0 + tout_rel[k], p_run_vec)
        self.x = Xf[:, -1].flatten()
        self.y_run = Zf[:, -1].flatten()
        self.t = float(t0 + tout_rel[-1])

    def _build_fixed_step_fns(self, alpha):
        """Build (and cache) the CasADi functions for the fixed-step implicit
        theta-method: residual ``R``, iteration matrix ``J = dR/d[x,z]`` and the
        frozen-slope evaluator ``f``. ``alpha`` is baked into ``R`` (0.5 =
        trapezoidal, 1.0 = backward Euler), so this is rebuilt only when ``alpha``
        changes; ``Dt`` is a runtime argument, not baked in."""
        d = self.dae_dict
        x, z, p, f, g = d['x'], d['z'], d['p'], d['ode'], d['alg']
        nx = x.numel()
        w = ca.vertcat(x, z)
        Dt = ca.SX.sym('Dt')
        xn = ca.SX.sym('xn', nx)        # x_n (frozen within a step)
        fn = ca.SX.sym('fn', nx)        # f(x_n, z_n) (frozen within a step)
        # theta-method residual: x_{n+1} - x_n - Dt*((1-a)*f_n + a*f_{n+1}) ; g = 0
        R = ca.vertcat(x - xn - Dt * ((1.0 - alpha) * fn + alpha * f), g)
        return {
            'nx': nx,
            'f': ca.Function('f', [x, z, p], [f]),
            'R': ca.Function('R', [w, xn, fn, p, Dt], [R]),
            'J': ca.Function('J', [w, xn, fn, p, Dt], [ca.jacobian(R, w)]),
        }

    @staticmethod
    def _make_lin_backend(linsol, sparsity):
        """Construct a linear-solver backend for the fixed-step Newton iteration.

        ``linsol`` selects the backend:
          - ``'scipy'`` (default) -> :class:`_ScipyLU` (scipy sparse LU).
          - ``'qr'`` / ``'lapacklu'`` / ``'csparse'`` / ``'ma27'`` ... -> a
            :class:`_CasadiLinsol` using that ``ca.Linsol`` plugin.
          - ``'casadi:<plugin>'`` -> explicit CasADi plugin form.
        A requested ``ca.Linsol`` plugin that is unavailable (e.g. an HSL solver
        with no HSL installed) falls back to SciPy with a warning, so a run never
        hard-fails on a missing plugin.
        """
        if linsol in (None, 'scipy'):
            return _ScipyLU()
        plugin = linsol.split(':', 1)[1] if str(linsol).startswith('casadi:') else linsol
        if plugin == 'scipy':
            return _ScipyLU()
        try:
            return _CasadiLinsol(sparsity, plugin)
        except Exception as exc:
            warnings.warn(
                f"ca.Linsol plugin {plugin!r} unavailable ({exc}); "
                "falling back to SciPy sparse LU", RuntimeWarning)
            return _ScipyLU()

    def _run_fixed_step(self, t_end, alpha=0.5, tol=1e-9, max_iter=10,
                        refactor_every=25, refactor_iters=4, linsol='scipy'):
        """Fixed-step implicit theta-method (trapezoidal / backward Euler).

        Modified Newton per step: the iteration matrix ``J`` is factorized by the
        selected ``linsol`` backend and reused for ``refactor_every`` steps. It is
        refreshed on schedule, after a discontinuity (``set_value`` marks the
        backend stale), or adaptively when a step exceeds ``refactor_iters``
        Newton iterations. A step that fails is retried once with a fresh
        factorization before raising.

        ``linsol`` picks the linear solver: ``'scipy'`` (default, scipy sparse LU)
        or a ``ca.Linsol`` plugin such as ``'qr'`` (recommended CasADi-native
        backend). See :meth:`_make_lin_backend`.
        """
        if self._fixed_step_fns is None or self._fixed_step_alpha != alpha:
            self._fixed_step_fns = self._build_fixed_step_fns(alpha)
            self._fixed_step_alpha = alpha
            self._lin_backend = None
        fns = self._fixed_step_fns
        nx, _f, _R, _J = fns['nx'], fns['f'], fns['R'], fns['J']

        # (Re)build the linear-solver backend when alpha or linsol changes. The
        # symbolic factorization is bound to J's fixed sparsity and done once.
        key = (alpha, linsol)
        if self._lin_backend is None or self._lin_backend_key != key:
            self._lin_backend = self._make_lin_backend(linsol, fns['J'].sparsity_out(0))
            self._lin_backend_key = key
        backend = self._lin_backend

        Dt = self.Dt
        n = int(round((t_end - self.t) / Dt))
        if n <= 0:
            return
        p_run_vec = np.concatenate((self.p_vals, self.u_run_vals))
        x = np.asarray(self.x, float).flatten()
        z = np.asarray(self.y_run, float).flatten()

        def newton(w, xf, fnv):
            """In-place modified-Newton solve for one step. Returns the iteration
            count, or None if it did not converge within ``max_iter``."""
            for it in range(max_iter):
                R = np.array(_R(w, xf, fnv, p_run_vec, Dt)).flatten()
                if np.linalg.norm(R) < tol:
                    return it
                if backend.stale or (self._sc % refactor_every == 0 and it == 0):
                    backend.refresh(_J(w, xf, fnv, p_run_vec, Dt))
                w += backend.solve(-R)
            return None

        for _ in range(n):
            fnv = np.array(_f(x, z, p_run_vec)).flatten()
            w = np.concatenate((x, z))
            iters = newton(w, x, fnv)
            if iters is None:                       # stale/ill-conditioned -> fresh J, retry once
                backend.mark_stale()
                w = np.concatenate((x, z))
                iters = newton(w, x, fnv)
                if iters is None:
                    kind = 'trapezoidal' if alpha == 0.5 else f'theta (alpha={alpha:g})'
                    raise RuntimeError(
                        f"fixed-step {kind} did not converge at t={self.t + Dt:.4f} s "
                        "(try a smaller Dt or a looser tol)")
            elif iters > refactor_iters:            # hard step -> refresh J next step
                backend.mark_stale()
            x, z = w[:nx], w[nx:]
            self.t += Dt
            self._sc += 1
            self._append_output(x, z, self.t, p_run_vec)
        self.x, self.y_run = x, z

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
        # A parameter/input/state change invalidates the frozen fixed-step
        # factorization; force a refactor on the next fixed-step run() step.
        if self._lin_backend is not None:
            self._lin_backend.mark_stale()

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
