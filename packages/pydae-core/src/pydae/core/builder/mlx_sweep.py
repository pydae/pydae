r"""Batched parameter sweeps for the MLX backend.

:class:`MlxSweep` solves **B copies of the same DAE at once** -- same equations,
different parameters or inputs -- on the Apple Silicon GPU.  It is the reason the
MLX backend exists: as a scalar solver it trails CasADi badly, because a dense AD
Jacobian and per-op GPU dispatch cost almost the same whether you solve one system
or a thousand.  Batching amortises all of that over B.

Canonical use -- reactive power at fixed excitation across 10,000 dispatch points::

    m = MlxBuilder(grid.sys_dict).build()
    m.ini({'p_m_1': 0.5})                                  # nominal anchor
    sw = m.sweep({'p_m_1': np.linspace(0.05, 0.95, 10_000)})
    sw.ini()
    q = sw.get_values('q_g_1')                             # (10000,)

Measured on ``genrou`` (16 unknowns, float32, M-series GPU), against the same
model's own scalar ``ini`` at 11 ms a case:

===========  ==========  =============  ===========
cases (B)    wall time   per case       vs scalar
===========  ==========  =============  ===========
100          0.04 s      0.39 ms        29x
1 000        0.06 s      0.055 ms       200x
10 000       0.29 s      0.029 ms       381x
50 000       0.84 s      0.017 ms       704x
===========  ==========  =============  ===========

Batched transients land in the same place: 0.013 ms per step per case at
B=10,000, against roughly 10 ms per step for a single case.  The speedup grows
with B because the fixed per-op dispatch cost is what is being amortised, so
small sweeps are barely worth batching and large ones are worth a great deal.

How the batching works
----------------------
The equations lowered by :class:`~pydae.core.builder.mlx_builder.MlxBuilder` are
*elementwise* in their arguments, so handing them arrays of shape ``(n_vars, B)``
instead of ``(n_vars,)`` makes ``v[i]`` a ``(B,)`` slice and evaluates B
independent systems in one graph -- no change to the lowering, and B=1 reproduces
the scalar path exactly.  Jacobians come out as ``(n_out, n_var, B)`` and
transpose into the ``(B, n, n)`` that ``mx.linalg.solve`` wants.

They are always the *compressed* ones (``_jac_ini_batched`` /
``_jac_trap_batched``, see :mod:`pydae.core.builder.mlx_sparsity`): a batched AD
pass carries real work, so cutting 87 passes to 8 is worth 18x at B=1,000 and
106x at B=10,000 on the Jacobian alone.  The dense linear solve then dominates
what is left.

Not every case has an answer
----------------------------
A wide sweep routinely contains operating points with **no steady state at all**,
and that boundary is usually the result the user is after.  Sweeping ``p_m`` on a
machine at fixed ``v_f`` stops converging at ``E*V/X_d`` -- the pull-out limit --
and no amount of solver effort will change that.  So non-convergence is reported
as *data*, never raised: :attr:`status` classifies every case, :attr:`converged`
masks them, non-converged entries read back as NaN, and
:meth:`feasible_range` names the span that actually solved.

Every case is solved with a masked Newton iteration: once a case converges (or
fails) its system is replaced *inside the compiled kernel* by the identity, which
both freezes it and keeps the batched LU well conditioned -- MLX aborts the whole
batch if any single element is singular.
"""

from __future__ import annotations

import numpy as np
from pydae.core.builder.mlx_builder import _mx, _on_stream

__all__ = ["MlxSweep", "SweepError", "STATUS_LABELS"]


# Per-case outcome codes.
CONVERGED = 0
MAX_ITER = 1
NONFINITE = 2
SINGULAR = 3

STATUS_LABELS = {
    CONVERGED: "converged",
    MAX_ITER: "max_iter",
    NONFINITE: "non_finite",
    SINGULAR: "singular",
}


class SweepError(RuntimeError):
    """Raised only when ``require_all=True`` and some case did not converge."""

    def __init__(self, message, failures):
        super().__init__(message)
        self.failures = failures


class MlxSweep:
    """B copies of one DAE, solved together.

    Created through :meth:`~pydae.core.builder.mlx_builder.MlxBuilder.sweep`
    rather than directly.

    Parameters
    ----------
    model : MlxBuilder
        A built model.  Run its scalar :meth:`ini` first: that operating point
        seeds every case.
    values : dict
        ``name -> scalar | sequence``.  Sequences define the batch width B;
        scalars and length-1 sequences broadcast.  A swept name is written to
        *every* block that carries it (``params_dict``, ``u_ini_dict`` and
        ``u_run_dict`` alike), because it defines the case rather than a
        phase-specific override.
    chunk : int, optional
        Cases solved per device call.  Defaults to a memory-budget estimate; it
        caps peak memory without changing any result.
    memory_budget_gb : float
        Ceiling used to pick ``chunk`` automatically (default 2.0).  Throughput
        saturates by chunk 4096 on every model measured, so this rarely binds --
        lower it when memory is tight rather than to go faster.  At n=87,
        ``0.5`` gives chunk 2048 and roughly 500 MB, still at full speed.
    ridge_rel : float
        Relative diagonal ridge added to the iteration matrix (default 1e-12,
        below float32 eps).  Turns an exactly singular case into a large finite
        step that is caught per element, instead of an exception that would
        abort the whole batch.
    """

    def __init__(self, model, values=None, *, chunk=None, memory_budget_gb=2.0,
                 itol=None, xtol=None, stall_tol=None, max_iter=None,
                 ridge_rel=1e-12, relax=1.0, verbose=None):
        model._require_built()
        self.model = model
        self.stream = model.stream
        self.dtype = model.dtype
        self.np_dtype = model.np_dtype
        self.verbose = model.verbose if verbose is None else verbose

        self.itol = model.itol if itol is None else itol
        self.xtol = model.xtol if xtol is None else xtol
        self.stall_tol = model.stall_tol if stall_tol is None else stall_tol
        self.max_it = model.max_it if max_iter is None else max_iter
        self.ridge_rel = ridge_rel
        self.relax = relax
        self.memory_budget_gb = memory_budget_gb

        self.cases = self._broadcast_cases(values or {})
        self.B = self._batch_width(self.cases)
        self.chunk = chunk or self._auto_chunk()

        self._kernels = {}
        self._reset_state()
        self._apply_cases()

        self.status = np.full(self.B, MAX_ITER, dtype=np.int8)
        self.residual = np.full(self.B, np.inf)
        self.iterations = np.zeros(self.B, dtype=np.int32)
        self.t = 0.0
        self.Time = np.zeros(0)
        self._series = {}
        self._recorded = []
        self._solved = False

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def grid(cls, model, chunk=None, **axes):
        """Cartesian product of named axes.

        ``MlxSweep.grid(m, p_m_1=np.linspace(0.1, 0.9, 100), v_f_1=[1.4, 1.6])``
        gives B = 200 cases covering every combination.
        """
        names = list(axes)
        mesh = np.meshgrid(*[np.atleast_1d(axes[n]) for n in names], indexing="ij")
        return cls(model, {n: g.ravel() for n, g in zip(names, mesh)}, chunk=chunk)

    def _broadcast_cases(self, values):
        widths = {}
        for name, val in values.items():
            arr = np.atleast_1d(np.asarray(val, dtype=np.float64))
            if arr.ndim != 1:
                raise ValueError(f"sweep value for '{name}' must be 1-D, got shape {arr.shape}")
            widths[name] = arr
        return widths

    @staticmethod
    def _batch_width(cases):
        lengths = {len(v) for v in cases.values() if len(v) > 1}
        if len(lengths) > 1:
            raise ValueError(
                f"sweep sequences have inconsistent lengths {sorted(lengths)}; "
                "every swept name must have the same number of cases (or be scalar)")
        return lengths.pop() if lengths else (1 if cases else 1)

    def _auto_chunk(self):
        r"""Cases per device call, from a memory budget.

        Peak memory tracks ``chunk``, never ``B`` -- that is the whole point of
        chunking, and a sweep of 50,000 cases peaks exactly where one of 10,000
        does.  What a case costs is roughly **eight** ``n``-by-``n`` temporaries:
        the Jacobian the compressed AD expands, its ``(B, n, n)`` transpose, the
        masked-and-ridged copy the kernel builds, and the factorisation inside
        ``mx.linalg.solve``.  The ``40 * n`` term covers the lowered tape's live
        set, which dominates only for very small ``n``.

        Measured with ``mx.get_peak_memory()`` at chunk 1024, float32:

        =========  =====  ==============  =============
        model      n      measured/case   this estimate
        =========  =====  ==============  =============
        ``genrou``  16     10.6 kB         10.5 kB
        ``k12p6``   87    245.6 kB        250.0 kB
        =========  =====  ==============  =============

        Do not trust ``mx.get_peak_memory()`` to check this in **float64**: that
        runs on the CPU stream, which it under-reports (k12p6 float64 measures
        253 kB/case against float32's 246 kB, where it should be about double).
        Measure process RSS there instead.
        """
        n = max(self.model.N_x + self.model.N_y_ini,
                self.model.N_x + self.model.N_y_run)
        itemsize = np.dtype(self.np_dtype).itemsize
        per_case = itemsize * (8 * n * n + 40 * n)
        chunk = int((self.memory_budget_gb * 2 ** 30) // max(per_case, 1))
        # Throughput saturates well before the budget does, so a bigger chunk is
        # bought memory that buys nothing back. Measured on ini(), wall time and
        # peak against chunk:
        #   genrou (n=16), B=32768: 2048 0.38 s/22 MB | 4096 0.33 s/45 MB
        #                           8192 0.34 s/89 MB
        #   k12p6  (n=87), B=16384: 2048 0.98 s/498 MB | 4096 0.96 s/996 MB
        #                           8192 0.97 s/1988 MB | 16384 1.05 s/3943 MB
        # Flat past ~4096 in time, linear in memory, and slower beyond -- so cap
        # there rather than at the budget.
        chunk = max(64, min(chunk, 4096, self.B))
        if chunk > 256:
            chunk = 256 * (chunk // 256)
        return min(chunk, self.B)

    def _reset_state(self):
        """Batched twins of the scalar model's state, each ``(n, B)``."""
        m, B = self.model, self.B

        def tile(v):
            return np.repeat(np.asarray(v, dtype=self.np_dtype)[:, None], B, axis=1)

        self.x = tile(m.x)
        self.y_ini = tile(m.y_ini)
        self.y_run = tile(m.y_run)
        self.p_vals = tile(m.p_vals)
        self.u_ini_vals = tile(m.u_ini_vals)
        self.u_run_vals = tile(m.u_run_vals)
        # c_* are never swept: frozen phase constants, broadcast at call time.
        self.c_ini_vals = tile(m.c_ini_vals)
        self.c_run_vals = tile(m.c_run_vals)

    def _apply_cases(self, only=None):
        """Write each swept value into every block that carries its name.

        ``only`` restricts the write to certain blocks -- used after a solve, so
        re-asserting the case inputs cannot overwrite converged state values.
        """
        blocks = {"p": self.p_vals, "u_ini": self.u_ini_vals, "u_run": self.u_run_vals,
                  "x": self.x, "y_ini": self.y_ini, "y_run": self.y_run}
        for name, vals in self.cases.items():
            slots = self.model._locate(name)
            if not slots:
                raise KeyError(f"unknown variable '{name}' -- not a parameter, input or state")
            row = np.broadcast_to(vals.astype(self.np_dtype), (self.B,))
            for block, idx in slots:
                if only is None or block in only:
                    blocks[block][idx, :] = row

    # ------------------------------------------------------------------
    # device helpers
    # ------------------------------------------------------------------
    @_on_stream
    def _dev(self, values):
        mx = _mx()
        return mx.array(np.ascontiguousarray(values, dtype=self.np_dtype), dtype=self.dtype)

    @staticmethod
    def _pad(block, width):
        """Widen an ``(n, k)`` slice to ``(n, width)`` by repeating the first case."""
        k = block.shape[1]
        if k == width:
            return block
        return np.concatenate([block, np.repeat(block[:, :1], width - k, axis=1)], axis=1)

    def _chunks(self):
        for start in range(0, self.B, self.chunk):
            yield slice(start, min(start + self.chunk, self.B))

    # ------------------------------------------------------------------
    # compiled masked Newton kernels
    # ------------------------------------------------------------------
    def _ini_kernel(self, width):
        key = ("ini", width)
        if key in self._kernels:
            return self._kernels[key]
        mx, m = _mx(), self.model
        n = m.N_x + m.N_y_ini
        fg, jac = m._fg_ini, m._jac_ini_batched
        eye, cpu = mx.eye(n, dtype=self.dtype), mx.cpu

        @mx.compile
        def step(v, p, u, c, mask, relax, ridge):
            """One masked Newton iteration of the batched initialization."""
            R = fg(v, p, u, c)                              # (n, B)
            J = jac(v, p, u, c)                             # (n, n, B)
            Jb = mx.transpose(J, (2, 0, 1))                 # (B, n, n)
            a = mask[:, None, None]
            scale = mx.max(mx.abs(Jb), axis=(1, 2), keepdims=True)
            # frozen cases -> identity (never singular, zero step); active cases
            # get a ridge far below float32 eps purely as a singularity guard.
            Jb = a * Jb + (1.0 - a) * eye + ridge * scale * eye
            Rb = a * mx.transpose(R, (1, 0))[..., None]
            d = mx.transpose(mx.linalg.solve(Jb, -Rb, stream=cpu)[..., 0], (1, 0))
            ok = mx.all(mx.isfinite(d), axis=0)             # (B,)
            d = mx.where(mx.isfinite(d), d, 0.0)
            v_next = v + relax * mask * d
            res = mx.max(mx.abs(R), axis=0)
            rel = mx.max(mx.abs(d), axis=0) / (1.0 + mx.max(mx.abs(v_next), axis=0))
            return v_next, res, rel, ok

        self._kernels[key] = step
        return step

    @_on_stream
    def _f_kernel(self, width):
        """Compiled ``f(x, y, p, u, c)`` -- the frozen slope of each step.

        Worth compiling separately: it runs once per step outside the Newton
        loop, and replaying the lowered tape in Python each time is not free.
        """
        key = ("f", width)
        if key not in self._kernels:
            self._kernels[key] = _mx().compile(self.model._f_run)
        return self._kernels[key]

    def _run_kernel(self, width):
        key = ("run", width)
        if key in self._kernels:
            return self._kernels[key]
        mx, m = _mx(), self.model
        n = m.N_x + m.N_y_run
        residual = m._trap_residual
        jac = m._jac_trap_batched
        eye, cpu = mx.eye(n, dtype=self.dtype), mx.cpu

        @mx.compile
        def step(w, x_k, f_k, p, u, c, dt, alpha, mask, relax, ridge):
            """One masked Newton iteration of the batched trapezoidal step."""
            R = residual(w, x_k, f_k, p, u, c, dt, alpha)
            J = jac(w, x_k, f_k, p, u, c, dt, alpha)
            Jb = mx.transpose(J, (2, 0, 1))
            a = mask[:, None, None]
            scale = mx.max(mx.abs(Jb), axis=(1, 2), keepdims=True)
            Jb = a * Jb + (1.0 - a) * eye + ridge * scale * eye
            Rb = a * mx.transpose(R, (1, 0))[..., None]
            d = mx.transpose(mx.linalg.solve(Jb, -Rb, stream=cpu)[..., 0], (1, 0))
            ok = mx.all(mx.isfinite(d), axis=0)
            d = mx.where(mx.isfinite(d), d, 0.0)
            w_next = w + relax * mask * d
            res = mx.max(mx.abs(R), axis=0)
            rel = mx.max(mx.abs(d), axis=0) / (1.0 + mx.max(mx.abs(w_next), axis=0))
            return w_next, res, rel, ok

        self._kernels[key] = step
        return step

    # ------------------------------------------------------------------
    # batched Newton driver
    # ------------------------------------------------------------------
    def _newton(self, step, w, args, active, max_it=None):
        """Iterate ``step`` until every case has converged or been classified.

        Mirrors the scalar driver's three stopping rules
        (:meth:`MlxBuilder._newton`) applied per case, and returns a status
        vector instead of raising -- one hopeless operating point must not take
        a 10,000-case sweep down with it.
        """
        mx = _mx()
        max_it = self.max_it if max_it is None else max_it
        width = w.shape[1]
        status = np.full(width, MAX_ITER, dtype=np.int8)
        best = np.full(width, np.inf)
        stall = np.zeros(width, dtype=np.int32)
        iters = np.zeros(width, dtype=np.int32)
        res = np.full(width, np.inf)
        mask = active.astype(self.np_dtype)
        relax = self._dev([self.relax])[0]
        ridge = self._dev([self.ridge_rel])[0]

        for _ in range(max_it):
            if not mask.any():
                break
            mask_d = self._dev(mask)
            try:
                w, res_d, rel_d, ok_d = step(w, *args, mask_d, relax, ridge)
                mx.eval(w, res_d, rel_d, ok_d)
                res, rel, ok = np.array(res_d), np.array(rel_d), np.array(ok_d)
            except RuntimeError:
                # A still-active case went exactly singular and MLX aborted the
                # whole batched factorization. Redo this iteration per element in
                # numpy so the offenders can be isolated and the batch continue.
                w, res, rel, ok = self._repair(step, w, args, mask, status)

            act = mask > 0
            iters[act] += 1
            bad = act & (~ok | ~np.isfinite(res))
            status[bad] = np.where(status[bad] == SINGULAR, SINGULAR, NONFINITE)
            done = act & ~bad & ((res < self.itol) | (rel < self.xtol))
            status[done] = CONVERGED
            # Residual floors near 1e-8 whatever the dtype (MLX's sin/cos/exp are
            # float32-accurate), so "stopped improving" must also count as done.
            stall = np.where(act & (res > 0.5 * best), stall + 1, np.where(act, 0, stall))
            best = np.minimum(best, np.where(np.isfinite(res), res, np.inf))
            floored = act & ~bad & ~done & (stall >= 2) & (res < self.stall_tol)
            status[floored] = CONVERGED
            mask = (act & ~bad & ~done & ~floored).astype(self.np_dtype)

        return w, res, status, iters

    def _repair(self, step, w, args, mask, status):
        """Per-element fallback for an aborted batched solve (rare path)."""
        mx, m = _mx(), self.model
        # Rebuild residual and Jacobian without the batched factorization.
        if step is self._kernels.get(("ini", w.shape[1])):
            R_d = m._fg_ini(*((w,) + args))
            J_d = m._jac_ini_batched(*((w,) + args))
        else:
            R_d = m._trap_residual(w, *args)
            J_d = m._jac_trap_batched(w, *args)
        mx.eval(R_d, J_d)
        R = np.array(R_d).astype(np.float64)
        J = np.array(mx.transpose(J_d, (2, 0, 1))).astype(np.float64)
        w_np = np.array(w).astype(np.float64)
        width = w_np.shape[1]
        d = np.zeros_like(w_np)
        ok = np.ones(width, dtype=bool)
        for b in range(width):
            if mask[b] <= 0:
                continue
            try:
                d[:, b] = np.linalg.solve(J[b], -R[:, b])
            except np.linalg.LinAlgError:
                ok[b] = False
                status[b] = SINGULAR
        d[:, ~np.isfinite(d).all(axis=0)] = 0.0
        w_np = w_np + self.relax * mask * d
        res = np.max(np.abs(R), axis=0)
        rel = np.max(np.abs(d), axis=0) / (1.0 + np.max(np.abs(w_np), axis=0))
        return self._dev(w_np), res, rel, ok

    # ------------------------------------------------------------------
    # backward initialization
    # ------------------------------------------------------------------
    @_on_stream
    def ini(self, stages=1, seed="current", require_all=False):
        """Solve every case's steady state ``0 = f = g``.

        Parameters
        ----------
        stages : int
            Homotopy stages.  ``1`` (default) goes straight to the target values.
            Higher values ramp each swept quantity from the model's nominal value
            in equal steps, seeding each stage with the previous solution -- a
            fallback for hard seeds.  It does not help where a case is genuinely
            infeasible, and costs proportionally more time.
        seed : {'current', 'previous'} or dict or str
            ``'current'`` broadcasts the scalar model's converged point;
            ``'previous'`` continues from this sweep's last solution; a dict of
            ``name -> scalar | (B,)`` overlays specific rows; a string is read as
            a JSON ``xy_0`` file.
        require_all : bool
            Raise :class:`SweepError` if any case fails, instead of reporting it.
        """
        m = self.model
        N_x = m.N_x
        v = self._seed_vector(seed)

        nominal = {name: self._nominal_of(name) for name in self.cases}
        targets = {name: np.broadcast_to(vals.astype(self.np_dtype), (self.B,))
                   for name, vals in self.cases.items()}

        n_stages = max(1, int(stages))
        for stage in range(1, n_stages + 1):
            theta = stage / n_stages
            if self.cases:
                staged = {n: nominal[n] + (targets[n] - nominal[n]) * theta
                          for n in self.cases}
                self._write_blocks(staged)
            v, status, residual, iters = self._ini_pass(v)

        self.x = v[:N_x]
        self.y_ini = v[N_x:]
        self.status = status
        self.residual = residual
        self.iterations = iters
        self.t = 0.0
        self._solved = True
        self.ini2run()
        self._report("ini")

        if require_all and not self.converged.all():
            raise SweepError(
                f"{int((~self.converged).sum())} of {self.B} cases did not converge",
                self.failures())
        return self

    def _ini_pass(self, v):
        """One full pass of the batched ini solve over all chunks."""
        status = np.empty(self.B, dtype=np.int8)
        residual = np.empty(self.B)
        iters = np.empty(self.B, dtype=np.int32)
        out = np.empty_like(v)
        width = self.chunk

        for sl in self._chunks():
            k = sl.stop - sl.start
            v_d = self._dev(self._pad(v[:, sl], width))
            args = (self._dev(self._pad(self.p_vals[:, sl], width)),
                    self._dev(self._pad(self.u_ini_vals[:, sl], width)),
                    self._dev(self._pad(self.c_ini_vals[:, sl], width)))
            active = np.zeros(width, dtype=bool)
            active[:k] = True                     # padding columns stay frozen
            step = self._ini_kernel(width)
            v_new, res, st, it = self._newton(step, v_d, args, active)
            out[:, sl] = np.array(v_new).astype(self.np_dtype)[:, :k]
            status[sl], residual[sl], iters[sl] = st[:k], res[:k], it[:k]
        return out, status, residual, iters

    def _seed_vector(self, seed):
        m = self.model
        base = np.concatenate((m.x, m.y_ini)).astype(self.np_dtype)
        if seed == "previous" and self._solved:
            v = np.concatenate((self.x, self.y_ini)).astype(self.np_dtype)
        else:
            v = np.repeat(base[:, None], self.B, axis=1)
        names = m.x_names + m.y_ini_names
        if isinstance(seed, str) and seed not in ("current", "previous"):
            import json
            with open(seed, encoding="utf-8") as fobj:
                seed = json.load(fobj)
        if isinstance(seed, dict):
            for key, val in seed.items():
                if key in names:
                    v[names.index(key), :] = np.broadcast_to(
                        np.asarray(val, dtype=self.np_dtype), (self.B,))
        return v

    def _nominal_of(self, name):
        """The model's pre-sweep value for a swept name (homotopy anchor)."""
        m = self.model
        for block, source in (("p", m.p_vals), ("u_ini", m.u_ini_vals),
                              ("u_run", m.u_run_vals)):
            for blk, idx in m._locate(name):
                if blk == block:
                    return np.full(self.B, source[idx], dtype=self.np_dtype)
        return np.zeros(self.B, dtype=self.np_dtype)

    def _write_blocks(self, staged):
        blocks = {"p": self.p_vals, "u_ini": self.u_ini_vals, "u_run": self.u_run_vals,
                  "x": self.x, "y_ini": self.y_ini, "y_run": self.y_run}
        for name, row in staged.items():
            for block, idx in self.model._locate(name):
                blocks[block][idx, :] = row

    def ini2run(self):
        """Batched twin of :meth:`MlxBuilder.ini2run`.

        Transfers each case's converged ini point onto its run variables and
        inputs -- including the ``y_ini``/``u_ini`` -> ``u_run`` handover, where
        a quantity solved during initialization (a ``v_ref_*``, a dispatch
        setpoint) becomes a fixed per-case input for the transient.
        """
        m = self.model
        val_map = {}
        for i, name in enumerate(m.x_names):
            val_map[name] = self.x[i]
        for i, name in enumerate(m.y_ini_names):
            val_map[name] = self.y_ini[i]
        for i, name in enumerate(m.u_ini_names):
            val_map[name] = self.u_ini_vals[i]

        for i, name in enumerate(m.y_run_names):
            if name in val_map:
                self.y_run[i] = val_map[name]
        ini_side = set(m.y_ini_names) | set(m.u_ini_names)
        for i, name in enumerate(m.u_run_names):
            if name in ini_side and name in val_map:
                self.u_run_vals[i] = val_map[name]
        # A swept name always wins over that handover: it defines the case.
        # Restricted to the input/parameter blocks so converged states survive.
        self._apply_cases(only={"p", "u_ini", "u_run"})

    # ------------------------------------------------------------------
    # batched transient
    # ------------------------------------------------------------------
    def record(self, names, decimation=1):
        """Declare which variables to store during :meth:`run`.

        Only these are kept, as ``(n_samples, B)`` arrays -- storing the full
        state of a large sweep would run to gigabytes.
        """
        names = [names] if isinstance(names, str) else list(names)
        for name in names:
            self._resolve_record(name)
        self._recorded = names
        self.decimation = max(1, int(decimation))
        return self

    def _resolve_record(self, name):
        m = self.model
        if name in m.x_names:
            return ("x", m.x_names.index(name))
        if name in m.y_run_names:
            return ("y", m.y_run_names.index(name))
        if name in m.z_names:
            return ("z", m.z_names.index(name))
        raise KeyError(f"'{name}' is not a state, algebraic variable or output")

    @_on_stream
    def run(self, t_end, update_dict=None, Dt=None, alpha=None, record=None,
            decimation=None, newton_max_iter=8, newton_iters=3, check_every=None):
        """Integrate every case to ``t_end`` with the trapezoidal rule.

        All cases advance on a shared time grid.  ``update_dict`` values may be
        scalars or ``(B,)`` arrays, so each case can take its own input step.
        A case that fails mid-run is frozen and its remaining samples are NaN,
        so a plot breaks visibly instead of flat-lining.

        Parameters
        ----------
        newton_iters : int or None
            Fixed Newton iterations per step (default 3), the fast path: the
            kernel is queued without ever reading a residual back, so the GPU
            pipeline never stalls.  Convergence is audited every ``check_every``
            steps instead of every iteration, which is what makes a long batched
            transient affordable -- host synchronisation, not arithmetic,
            dominates otherwise.  Pass ``None`` for the adaptive driver
            (residual-tested every iteration, up to ``newton_max_iter``).
        check_every : int, optional
            Step interval for the residual audit on the fixed-iteration path.
            Defaults to once per output sample.
        """
        mx, m = _mx(), self.model
        if record is not None:
            self.record(record, decimation or getattr(self, "decimation", 1))
        elif decimation is not None:
            self.decimation = max(1, int(decimation))
        if update_dict:
            self._route_run(update_dict)

        Dt = m.Dt if Dt is None else Dt
        alpha = m.alpha if alpha is None else alpha
        decim = getattr(self, "decimation", 1)
        n_steps = int(round((t_end - self.t) / Dt))
        if n_steps <= 0:
            return self

        slots = {name: self._resolve_record(name) for name in self._recorded}
        n_samples = n_steps // decim + (1 if n_steps % decim else 0)
        series = {name: np.full((n_samples, self.B), np.nan, dtype=self.np_dtype)
                  for name in self._recorded}
        times = np.empty(n_samples)
        width = self.chunk
        N_x = m.N_x
        t_start = self.t

        for sl in self._chunks():
            k = sl.stop - sl.start
            x = self._dev(self._pad(self.x[:, sl], width))
            y = self._dev(self._pad(self.y_run[:, sl], width))
            p = self._dev(self._pad(self.p_vals[:, sl], width))
            u = self._dev(self._pad(self.u_run_vals[:, sl], width))
            c = self._dev(self._pad(self.c_run_vals[:, sl], width))
            dt_d, al_d = self._dev([Dt])[0], self._dev([alpha])[0]
            step = self._run_kernel(width)
            f_eval = self._f_kernel(width)
            alive = np.zeros(width, dtype=bool)
            alive[:k] = self.converged[sl] if self._solved else True
            t = t_start
            sample = 0

            audit = check_every or decim
            mask_d = self._dev(alive.astype(self.np_dtype))
            relax_d, ridge_d = self._dev([self.relax])[0], self._dev([self.ridge_rel])[0]

            for istep in range(n_steps):
                f_k = f_eval(x, y, p, u, c)
                w = mx.concatenate([x, y])
                if newton_iters:
                    # Fast path: queue a fixed number of iterations without ever
                    # pulling a residual back, so nothing blocks on the device.
                    res_d = None
                    for _ in range(int(newton_iters)):
                        w, res_d, _, _ = step(w, x, f_k, p, u, c, dt_d, al_d,
                                              mask_d, relax_d, ridge_d)
                    if (istep + 1) % audit == 0 or istep == n_steps - 1:
                        mx.eval(res_d)
                        res = np.array(res_d)
                        failed = alive & ~(np.isfinite(res) & (res < self.stall_tol))
                        if failed.any():
                            alive = alive & ~failed
                            mask_d = self._dev(alive.astype(self.np_dtype))
                            self._mark_failed(sl, k, failed, NONFINITE)
                else:
                    w, res, st, _ = self._newton(
                        step, w, (x, f_k, p, u, c, dt_d, al_d), alive,
                        max_it=newton_max_iter)
                    failed = alive & (st != CONVERGED)
                    if failed.any():
                        alive = alive & ~failed
                        self._mark_failed(sl, k, failed, None, st)
                x, y = w[:N_x], w[N_x:]
                t += Dt

                if (istep + 1) % decim == 0 or istep == n_steps - 1:
                    if self._recorded:
                        self._store(series, slots, sample, sl, k, x, y, p, u, c, alive)
                    times[sample] = t
                    sample += 1

            mx.eval(x, y)
            self.x[:, sl] = np.array(x).astype(self.np_dtype)[:, :k]
            self.y_run[:, sl] = np.array(y).astype(self.np_dtype)[:, :k]

        self.t = t_start + n_steps * Dt
        self._append_series(series, times)
        return self

    def _mark_failed(self, sl, k, failed, code, statuses=None):
        """Record a mid-transient failure without disturbing earlier verdicts."""
        current = self.status[sl]
        new = np.full(k, code) if code is not None else statuses[:k]
        self.status[sl] = np.where(failed[:k] & (current == CONVERGED), new, current)

    def _store(self, series, slots, sample, sl, k, x, y, p, u, c, alive):
        m = self.model
        z = None
        for name, (kind, idx) in slots.items():
            if kind == "x":
                row = x[idx]
            elif kind == "y":
                row = y[idx]
            else:
                if z is None:
                    z = m._h_run(x, y, p, u, c)
                row = z[idx]
            vals = np.array(row).astype(self.np_dtype)[:k]
            series[name][sample, sl] = np.where(alive[:k], vals, np.nan)

    def _append_series(self, series, times):
        if not self._recorded:
            self.Time = np.concatenate([self.Time, times])
            return
        for name, block in series.items():
            self._series[name] = (block if name not in self._series
                                  else np.concatenate([self._series[name], block]))
        self.Time = np.concatenate([self.Time, times])

    def _route_run(self, update_dict):
        blocks = {"p": self.p_vals, "u_run": self.u_run_vals,
                  "x": self.x, "y_run": self.y_run}
        for name, val in update_dict.items():
            slots = [s for s in self.model._locate(name, phase="run") if s[0] in blocks]
            if not slots:
                raise KeyError(f"unknown runtime variable '{name}'")
            row = np.broadcast_to(np.asarray(val, dtype=self.np_dtype), (self.B,))
            block, idx = slots[0]
            blocks[block][idx, :] = row

    def post(self):
        """Present for symmetry with the scalar model; series are already dense."""
        return self

    # ------------------------------------------------------------------
    # results
    # ------------------------------------------------------------------
    @property
    def converged(self):
        """(B,) bool mask of cases that reached a solution."""
        return self.status == CONVERGED

    def get_values(self, name, masked=True):
        """Current value of ``name`` across all cases -> ``(B,)``.

        Non-converged cases read back as NaN unless ``masked=False``.
        """
        m = self.model
        if name in m.x_names:
            out = self.x[m.x_names.index(name)]
        elif name in m.y_run_names:
            out = self.y_run[m.y_run_names.index(name)]
        elif name in m.y_ini_names:
            out = self.y_ini[m.y_ini_names.index(name)]
        elif name in m.u_run_names:
            out = self.u_run_vals[m.u_run_names.index(name)]
        elif name in m.u_ini_names:
            out = self.u_ini_vals[m.u_ini_names.index(name)]
        elif name in m.p_names:
            out = self.p_vals[m.p_names.index(name)]
        elif name in m.z_names:
            out = self._eval_h()[m.z_names.index(name)]
        else:
            raise KeyError(f"unknown variable '{name}'")
        out = np.asarray(out, dtype=np.float64).copy()
        if masked:
            out[~self.converged] = np.nan
        return out

    def get_mvalue(self, names, masked=True):
        return np.stack([self.get_values(n, masked) for n in names])

    def get_series(self, name):
        """Recorded trajectory of ``name`` -> ``(n_samples, B)``.

        Distinct from :meth:`get_values`, which is always the *current* value
        across cases; a recorded name has both.
        """
        if name not in self._series:
            raise KeyError(
                f"'{name}' was not recorded; pass record=['{name}'] to run()")
        return self._series[name]

    @_on_stream
    def _eval_h(self):
        mx, m = _mx(), self.model
        if m._h_run is None:
            raise KeyError("model has no h_dict outputs")
        out = np.empty((m.N_z, self.B), dtype=self.np_dtype)
        width = self.chunk
        for sl in self._chunks():
            k = sl.stop - sl.start
            z = m._h_run(self._dev(self._pad(self.x[:, sl], width)),
                         self._dev(self._pad(self.y_run[:, sl], width)),
                         self._dev(self._pad(self.p_vals[:, sl], width)),
                         self._dev(self._pad(self.u_run_vals[:, sl], width)),
                         self._dev(self._pad(self.c_run_vals[:, sl], width)))
            mx.eval(z)
            out[:, sl] = np.array(z).astype(self.np_dtype)[:, :k]
        return out

    def feasible_range(self, name):
        """``(min, max)`` of a swept quantity over the cases that solved.

        On a `p_m` sweep at fixed excitation this is the machine's steady-state
        limit -- the answer, not an error.
        """
        if name not in self.cases:
            raise KeyError(f"'{name}' is not one of the swept names {list(self.cases)}")
        vals = np.broadcast_to(self.cases[name], (self.B,))[self.converged]
        if not vals.size:
            return (np.nan, np.nan)
        return (float(vals.min()), float(vals.max()))

    def failures(self):
        """DataFrame of the cases that did not solve, with their swept values."""
        import pandas as pd
        bad = ~self.converged
        data = {"case": np.arange(self.B)[bad]}
        for name, vals in self.cases.items():
            data[name] = np.broadcast_to(vals, (self.B,))[bad]
        data["status"] = [STATUS_LABELS[int(s)] for s in self.status[bad]]
        data["residual"] = self.residual[bad]
        data["iterations"] = self.iterations[bad]
        return pd.DataFrame(data)

    def to_frame(self, names=None, dropna=False):
        """One row per case: the swept values, requested outputs and status."""
        import pandas as pd
        data = {}
        for name, vals in self.cases.items():
            data[name] = np.broadcast_to(vals, (self.B,))
        for name in (names or []):
            data[name] = self.get_values(name)
        data["converged"] = self.converged
        data["status"] = [STATUS_LABELS[int(s)] for s in self.status]
        frame = pd.DataFrame(data)
        return frame[frame["converged"]] if dropna else frame

    def _report(self, phase):
        if not self.verbose:
            return
        ok = int(self.converged.sum())
        line = f"{phase}: {ok}/{self.B} cases converged"
        if ok:
            line += f" (max residual {np.nanmax(self.residual[self.converged]):.1e})"
        if ok < self.B:
            counts = {}
            for s in self.status[~self.converged]:
                counts[STATUS_LABELS[int(s)]] = counts.get(STATUS_LABELS[int(s)], 0) + 1
            detail = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
            line += f"; {self.B - ok} did not: {detail}"
            for name in self.cases:
                lo, hi = self.feasible_range(name)
                line += f"\n  feasible {name}: {lo:.4g} .. {hi:.4g}"
        print(line)

    # ------------------------------------------------------------------
    # batched small-signal analysis
    # ------------------------------------------------------------------
    @_on_stream
    def _blocks(self, sl, width, k):
        mx, m = _mx(), self.model
        args = (self._dev(self._pad(self.x[:, sl], width)),
                self._dev(self._pad(self.y_run[:, sl], width)),
                self._dev(self._pad(self.p_vals[:, sl], width)),
                self._dev(self._pad(self.u_run_vals[:, sl], width)),
                self._dev(self._pad(self.c_run_vals[:, sl], width)))
        out = []
        for block in (m._Fx, m._Fy, m._Gx, m._Gy):
            blk = block(*args)                       # (n_out, n_var, B)
            mx.eval(blk)
            out.append(np.array(mx.transpose(blk, (2, 0, 1))).astype(np.float64)[:k])
        return out

    @_on_stream
    def A_eval(self):
        """State matrix per case: ``A = Fx - Fy Gy^-1 Gx`` -> ``(B, N_x, N_x)``.

        The AD blocks are produced batched on the GPU; the reduction moves to
        numpy in float64, which factorises ``Gy`` once for all ``N_x`` right-hand
        sides (MLX's CPU-only solve would refactorise per column) and keeps the
        error-amplifying step out of float32.
        """
        m = self.model
        A = np.full((self.B, m.N_x, m.N_x), np.nan)
        width = self.chunk
        for sl in self._chunks():
            k = sl.stop - sl.start
            Fx, Fy, Gx, Gy = self._blocks(sl, width, k)
            A[sl] = Fx - Fy @ np.linalg.solve(Gy, Gx)
        A[~self.converged] = np.nan
        return A

    def eigenvalues(self, sort=True):
        """Eigenvalues per case -> ``(B, N_x)`` complex.

        With ``sort=True`` each spectrum is ordered by (real, imag) so a locus
        plotted against the swept parameter is continuous enough to read.
        """
        A = self.A_eval()
        out = np.full((self.B, self.model.N_x), np.nan + 0j)
        good = self.converged & np.isfinite(A).all(axis=(1, 2))
        if good.any():
            w = np.linalg.eigvals(A[good])
            if sort:
                order = np.lexsort((w.imag, w.real), axis=1)
                w = np.take_along_axis(w, order, axis=1)
            out[good] = w
        return out

    def damp(self, case, **kwargs):
        """Damping table for one case, via :func:`pydae.ssa.ssa.damp`."""
        from pydae.ssa.ssa import damp
        return damp(self.A_eval()[case], **kwargs)

    def __repr__(self):
        swept = ", ".join(self.cases) or "none"
        return (f"<MlxSweep B={self.B} swept=[{swept}] chunk={self.chunk} "
                f"converged={int(self.converged.sum())}>")
