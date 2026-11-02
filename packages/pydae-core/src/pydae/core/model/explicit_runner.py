# -*- coding: utf-8 -*-
r"""Compiled explicit fixed-step runtime for a ``CasadiModel``: ``run(method='rk4')``.

The DAE is first reduced (:mod:`pydae.core.builder.dae_reduce`): every
algebraic variable an equation gives explicitly is substituted away, which
for many models -- machines with look-up tables, control-block chains --
leaves a plain ODE.  The reduced system is integrated with the classical
fourth-order Runge-Kutta method at the model's ``Dt`` (optionally in
``substeps`` equal pieces).  One step is one compiled function returning
the next state, all algebraic variables and the ``h_dict`` outputs.

Two ways it is called, both from ``CasadiModel.run``:

* **a span of many steps** -- ``mapaccum`` chains them in ONE CasADi call,
  so no Python runs per step;
* **one step at a time** -- a user's control loop calling ``run(t + Dt, ...)``
  every period.  There the cost is the call itself, so the step is evaluated
  through a CasADi ``FunctionBuffer`` reading and writing preallocated NumPy
  arrays in place (0.04 us against ~6.5 us for an ordinary call).

Algebraic variables that are not explicit (the "implicit remainder") are
solved by Newton inside every RK4 stage, warm-started from the previous
step.  The algebraic variables are recomputed from the state at every step,
so they are always consistent: no ``recalculate_algebraics()`` is needed
after a ``set_value`` discontinuity.

**When to use it.**  For models that are not stiff at the chosen ``Dt``:
explicit RK4 is stable only for $|\lambda| \Delta t \lesssim 2.8$ along the
real axis.  A divergent run raises (the model's state is left as it was);
use smaller ``Dt``, more ``substeps``, or an implicit method
(``'trapezoidal'``, ``'idas_grid'``) for stiff systems.

Measured on a field-oriented PMSM drive from FEA tables (4 states, 6
algebraic variables, all eliminated; 0.6 s at 50 us): 3.0 us/step (1.4 of
them integrating, 1.0 evaluating ``h_dict``), against 14 us with Newton on
the algebraics inside a compiled RK4, 15-18 us for ``idas_grid`` and 53 us
for ``trapezoidal``, all on the same trajectory to 1e-6 A.
"""
import casadi as ca
import numpy as np
from pydae.core.builder.dae_reduce import reduce_explicit_algebraics


class ExplicitRunner:
    """Reduced-DAE RK4 runtime bound to one ``CasadiModel``'s ``dae_dict``.

    Parameters
    ----------
    dae_dict : dict
        ``{'x', 'z', 'p', 'ode', 'alg'}`` as ``CasadiBuilder`` produces.
    h_fn : casadi.Function, optional
        ``(x, z, p) -> h``, the model's outputs; evaluated inside the step.
    substeps : int
        RK4 steps per output step ``Dt`` (outputs are stored once per ``Dt``).
    newton_opts : dict, optional
        Options for the ``'newton'`` rootfinder on the implicit remainder.
    """

    def __init__(self, dae_dict, h_fn=None, substeps=1, newton_opts=None):
        d = dae_dict
        self.reduced = r = reduce_explicit_algebraics(d["x"], d["z"], d["p"], d["ode"], d["alg"])
        self.substeps = int(substeps)
        self.nx, self.nzi = r.x.numel(), r.z_imp.numel()
        self.nz, self.np = r.z.numel(), r.p.numel()
        self.z_imp_idx = np.array(r.z_imp_idx, dtype=int)
        self.has_h = h_fn is not None
        self._coeff_fn = r.coeff_fn
        self._coeff_ok_for = None                        # the p vector last checked
        self._runs = {}

        F = ca.Function("f_red", [r.x, r.z_imp, r.p], [r.f])
        Zfull = ca.Function("z_full", [r.x, r.z_imp, r.p], [r.z_full])
        # SX all the way when there is nothing to solve (fastest); MX around
        # the rootfinder otherwise
        Sym = ca.SX if r.is_ode else ca.MX
        s = Sym.sym("s", self.nx + self.nzi)
        p = Sym.sym("p", self.np)
        Dt = Sym.sym("Dt")
        # explicit empty slice: s[nx:] on a length-nx vector gives a 1x1 zero, not 0x1
        x = s[0:self.nx]
        zg = s[self.nx:self.nx + self.nzi] if self.nzi else Sym(0, 1)
        if r.is_ode:
            def alg(xx, zz):
                return zz
        else:
            G = ca.Function("g_imp", [r.z_imp, r.x, r.p], [r.g_imp])
            rf = ca.rootfinder("rf_imp", "newton", G, dict({"abstol": 1e-10}, **(newton_opts or {})))

            def alg(xx, zz):
                return rf(zz, xx, p)

        h = Dt / self.substeps
        for _ in range(self.substeps):
            z1 = alg(x, zg)
            k1 = F(x, z1, p)
            k2 = F(x + h / 2 * k1, alg(x + h / 2 * k1, z1), p)
            k3 = F(x + h / 2 * k2, alg(x + h / 2 * k2, z1), p)
            k4 = F(x + h * k3, alg(x + h * k3, z1), p)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            zg = z1
        zn = alg(x, zg)                                  # consistent at the stored point
        z_all = Zfull(x, zn, p)
        out_h = h_fn(x, z_all, p) if self.has_h else Sym(0, 1)
        self.nh = out_h.numel()
        self._step = ca.Function("rk4_step", [s, p, Dt], [ca.vertcat(x, zn), z_all, out_h])

        # one step at a time: evaluate in place through a FunctionBuffer
        self._b_s = np.zeros(self.nx + self.nzi)
        self._b_p = np.zeros(self.np)
        self._b_dt = np.zeros(1)
        self._o_s = np.zeros(self.nx + self.nzi)
        self._o_z = np.zeros(self.nz)
        self._o_h = np.zeros(self.nh)
        self._buf, self._eval = self._step.buffer()
        for k, a in enumerate((self._b_s, self._b_p, self._b_dt)):
            self._buf.set_arg(k, memoryview(a))
        for k, a in enumerate((self._o_s, self._o_z, self._o_h)):
            self._buf.set_res(k, memoryview(a))

    def _run_fn(self, n):
        if n not in self._runs:
            if len(self._runs) > 8:                      # keep a few segment lengths
                self._runs.pop(next(iter(self._runs)))
            self._runs[n] = self._step.mapaccum("rk4_run", n)
        return self._runs[n]

    def _diverged(self, model, t_end):
        return RuntimeError(f"explicit RK4 diverged within t = {model.t:.6g} .. {t_end:.6g} s: the model "
                            "is stiff at this Dt. Use a smaller Dt, more substeps, or method="
                            "'trapezoidal' / 'idas_grid'.")

    def run(self, model, t_end):
        """Advance ``model`` to ``t_end`` in steps of ``model.Dt``, appending
        every step to its ``Time/X/Y/Z`` storage."""
        Dt = model.Dt
        n = int(round((t_end - model.t) / Dt))
        if n <= 0:
            return
        p_vec = np.concatenate((model.p_vals, model.u_run_vals))
        # check the coefficients the eliminations divide by only when p changed
        if self._coeff_ok_for is None or not np.array_equal(p_vec, self._coeff_ok_for):
            c = np.array(self._coeff_fn(p_vec)).ravel()
            if c.size and np.any(c == 0):
                raise ZeroDivisionError("an eliminated algebraic equation has a zero coefficient "
                                        "at these parameter values (e.g. an inductance set to 0)")
            self._coeff_ok_for = p_vec.copy()
        for name in ("Time", "X", "Y", "Z"):             # post() may have turned them into arrays
            if isinstance(getattr(model, name), np.ndarray):
                setattr(model, name, list(getattr(model, name)))
        store_h = self.has_h and model.sys_dict.get("h_dict")

        if n == 1:
            self._b_s[:self.nx] = model.x
            if self.nzi:
                self._b_s[self.nx:] = np.asarray(model.y_run, float)[self.z_imp_idx]
            self._b_p[:] = p_vec
            self._b_dt[0] = Dt
            self._eval()
            if not (np.isfinite(self._o_s).all() and np.isfinite(self._o_z).all()):
                raise self._diverged(model, t_end)
            x_new, z_new = self._o_s[:self.nx].copy(), self._o_z.copy()
            model.t = float(model.t + Dt)
            model.Time.append(model.t)
            model.X.append(x_new)
            model.Y.append(z_new)
            if store_h:
                model.Z.append(self._o_h.copy())
            model.x, model.y_run = x_new, z_new
            return

        s0 = np.concatenate((np.asarray(model.x, float), np.asarray(model.y_run, float)[self.z_imp_idx]))
        S, Zf, H = self._run_fn(n)(s0, ca.repmat(p_vec, 1, n), ca.repmat(Dt, 1, n))
        S, Zf = np.array(S), np.array(Zf)
        if not (np.all(np.isfinite(S)) and np.all(np.isfinite(Zf))):
            raise self._diverged(model, t_end)
        Xf = S[:self.nx]
        t = model.t + Dt * np.arange(1, n + 1)
        model.Time.extend(t.tolist())
        model.X.extend(Xf.T.copy())
        model.Y.extend(Zf.T.copy())
        if store_h:
            model.Z.extend(np.array(H).T.copy())
        model.x = Xf[:, -1].copy()
        model.y_run = Zf[:, -1].copy()
        model.t = float(t[-1])
