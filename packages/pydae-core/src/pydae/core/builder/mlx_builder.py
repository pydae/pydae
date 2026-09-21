r"""MLX (Apple Silicon / Metal) backend for pydae DAE systems.

``MlxBuilder`` is a drop-in alternative to
:class:`pydae.core.builder.casadi_builder.CasadiBuilder`: it takes the same
``sys_dict`` and offers the same *define symbolically -> build -> simulate*
workflow, but neither generates C nor calls a CasADi rootfinder/IDAS.  The
equations are translated into ``mlx.core`` graphs, the Jacobians come from MLX
automatic differentiation, and the Newton-Raphson iteration is wrapped in
``mx.compile`` so each iteration executes as a single fused Metal kernel graph
on the Apple Silicon GPU.

Semi-explicit index-1 form handled here (identical to the rest of pydae)::

    xdot = f(x, y, u, p)
       0 = g(x, y, u, p)
       z = h(x, y, u, p)

Two evaluations of that system are compiled, following the *inicializacion
hacia atras* / *solucion hacia adelante* split:

**Backward initialization** (``ini``) -- the steady state is found by solving,
for the unknown vector ``v = [x; y_ini]``,

.. math::

    0 = f(x, y_{ini}, u_{ini}, p) \\
    0 = g(x, y_{ini}, u_{ini}, p)

i.e. the derivatives are set to zero and the *initialization* algebraic
vector/input split is used.  Variables that are algebraic during the run but
inputs during the initialization (and vice versa) are routed by name, which is
the numeric equivalent of the ``y_ini -> u_run`` symbolic substitution done by
``CasadiBuilder``.

**Forward simulation** (``run``) -- the trajectory is advanced with the
implicit trapezoidal rule using the *run* split ``(y_run, u_run)``.  With
:math:`\alpha = 0.5` (``alpha = 1`` gives backward Euler) each step solves

.. math::

    0 = x_{k+1} - x_k - \Delta t\,[(1-\alpha) f_k + \alpha f_{k+1}] \\
    0 = g(x_{k+1}, y_{k+1}, u_{run}, p)

for :math:`w = [x_{k+1}; y_{k+1}]` by Newton-Raphson, the Jacobian

.. math::

    J = \begin{bmatrix} I - \Delta t\,\alpha F_x & -\Delta t\,\alpha F_y \\
                        G_x & G_y \end{bmatrix}

being obtained by differentiating the whole step residual with MLX AD rather
than assembling the blocks by hand.

Front ends
----------
Both flavours of ``sys_dict`` are accepted, so the class is a genuine drop-in:

* **SymPy** expressions (the default pydae path, ``BpsBuilder(..., use_casadi=False)``)
  are lowered through ``sympy.lambdify`` onto a small numpy-shaped namespace
  backed by ``mlx.core``.
* **CasADi** ``SX`` expressions (``use_casadi=True``) are lowered by walking the
  ``SX`` instruction tape (``Function.instruction_id`` / ``instruction_input``)
  and replaying it with MLX ops -- no CasADi call happens at run time.

Numerical precision -- read this before trusting a tolerance
------------------------------------------------------------
Two MLX properties bound the accuracy of any DAE solved here (measured on
mlx 0.32):

* ``float64`` exists on the **CPU only** -- the Metal backend is ``float32``.
  ``dtype="float32"`` (the default) keeps the whole solve on the GPU;
  ``dtype="float64"`` runs the identical code on the CPU stream.
* ``mx.sin``, ``mx.cos`` and ``mx.exp`` are evaluated to **float32 accuracy
  even on float64 arrays** (relative error ~2e-8; ``sqrt``, ``log``, ``atan``
  and plain arithmetic are exact to double precision).  Every power-system
  model is trigonometric, so its residual bottoms out near 1e-8 in *either*
  dtype -- ``1e-10`` style tolerances are unreachable, and Newton is therefore
  allowed to stop when it demonstrably stops improving (see ``stall_tol``).

``mx.linalg.solve`` is CPU-only in MLX whatever the dtype, so the Newton
*linear solve* always runs on the CPU stream; MLX's unified memory means this
costs no host copy.

Sparsity
--------
Every Jacobian is built by *compressed* AD (see
:mod:`pydae.core.builder.mlx_sparsity`): structurally orthogonal columns are
grouped by Curtis-Powell-Reid colouring and probed together, so one forward
``jvp`` pass per colour replaces one reverse ``vjp`` pass per equation -- 8
passes rather than 87 on the Kundur two-area system.  The few dense rows a
power system always has (the centre-of-inertia speed couples every machine)
would collapse that colouring, so they are recovered by reverse mode instead,
one pass each.  ``sparse=False`` forces plain reverse-mode AD.

The *linear solve* stays dense: MLX arrays are immutable, so a sparse
factorization copies its whole value array on every pivot and measures slower
than the dense solve it would replace.

Performance
-----------
Still not a faster scalar drop-in: one small system cannot keep a GPU busy, and
per-op dispatch dominates whatever the Jacobian costs.  Use this backend where a
C toolchain or CasADi is unavailable, or -- its real purpose -- for GPU-batched
work over many operating points at once (see
:mod:`pydae.core.builder.mlx_sweep`), where compression is worth up to 106x on
the Jacobian alone at B=10,000.

Example
-------
::

    from pydae.core.builder.mlx_builder import MlxBuilder

    m = MlxBuilder(sys_dict).build()
    m.ini({'theta': 0.5}, xy_0={'p_x': 0.9, 'p_y': -5.1})
    m.run(20.0, {'f_x': 0.0})
    m.post()
    m.get_values('theta')
"""

from __future__ import annotations

import functools
import json
import math
import time

import numpy as np

__all__ = ["MlxBuilder"]


# ---------------------------------------------------------------------------
# MLX import (kept lazy so importing pydae on non-Apple platforms still works)
# ---------------------------------------------------------------------------
_MX = None


def _mx():
    """Return the ``mlx.core`` module, with an actionable error if missing."""
    global _MX
    if _MX is None:
        try:
            import mlx.core as mx
        except ImportError as exc:  # pragma: no cover - platform dependent
            raise ImportError(
                "MlxBuilder requires the 'mlx' package (Apple Silicon only). "
                "Install it with `pip install mlx`."
            ) from exc
        _MX = mx
    return _MX


# ---------------------------------------------------------------------------
# numpy-shaped namespace backed by mlx.core
# ---------------------------------------------------------------------------
class _MlxNumpy:
    """Minimal ``numpy`` look-alike whose functions are ``mlx.core`` ops.

    ``sympy.lambdify`` with :class:`~sympy.printing.numpy.NumPyPrinter` emits
    module-qualified calls (``numpy.sin(...)``, ``numpy.select(...)``).  Binding
    the name ``numpy`` to an instance of this class in the generated function's
    globals is therefore all that is needed to retarget lambdified code at MLX.
    """

    _DIRECT = (
        "sin cos tan arcsin arccos arctan arctan2 sinh cosh tanh arcsinh "
        "arccosh arctanh exp expm1 log log2 log10 log1p sqrt rsqrt sign floor "
        "ceil round abs maximum minimum where less less_equal greater "
        "greater_equal equal not_equal logical_and logical_or logical_not "
        "power remainder stack array concatenate zeros ones eye isnan isinf "
        "real imag conj erf square"
    ).split()

    def __init__(self):
        mx = _mx()
        self._mx = mx
        for name in self._DIRECT:
            fun = getattr(mx, name, None)
            if fun is not None:
                setattr(self, name, fun)
        self.pi = math.pi
        self.e = math.e
        self.nan = float("nan")
        self.inf = float("inf")
        self.float32 = mx.float32
        self.float64 = getattr(mx, "float64", mx.float32)

    # -- functions numpy has but mlx does not -------------------------------
    def select(self, condlist, choicelist, default=0.0):
        """``numpy.select`` semantics (first true condition wins) via ``where``.

        SymPy ``Piecewise`` lowers to this.  Folding from the last condition
        backwards keeps the *first* match winning while staying branch-free,
        which is what both ``mx.compile`` tracing and AD need.
        """
        mx = self._mx
        out = default
        for cond, choice in zip(reversed(list(condlist)), reversed(list(choicelist))):
            if cond is True or cond is None:
                out = choice
            else:
                out = mx.where(cond, choice, out)
        return out

    def _reduce(self, fun, a, axis=None):
        """``numpy.amax``/``amin`` over a tuple of scalar expressions.

        ``NumPyPrinter`` renders ``Max(a, b)`` as ``amax((a, b), axis=0)``, which
        is exactly right in the batched ``(n, B)`` layout: the stack is ``(2, B)``
        and reducing axis 0 leaves ``(B,)``.  An ``axis=None`` reduction over a
        batched array would instead collapse the batch and silently return one
        number for every case, so it is refused rather than guessed.
        """
        mx = self._mx
        if isinstance(a, (tuple, list)):
            a = _stack_broadcast([_as_array(v) for v in a])
            if axis is None:
                axis = 0
        elif axis is None and isinstance(a, mx.array) and a.ndim > 1:
            raise ValueError(
                "refusing an axis=None reduction over a batched array: it would "
                "collapse the sweep dimension. Pass an explicit axis.")
        return fun(a, axis=axis)

    def amax(self, a, axis=None):
        return self._reduce(self._mx.max, a, axis)

    def amin(self, a, axis=None):
        return self._reduce(self._mx.min, a, axis)

    max = amax
    min = amin

    def heaviside(self, x, h0=0.5):
        mx = self._mx
        return mx.where(x > 0, 1.0, mx.where(x < 0, 0.0, h0))

    def __getattr__(self, name):  # only reached for genuinely missing names
        raise AttributeError(
            f"the MLX backend has no translation for 'numpy.{name}'. "
            "Rewrite the offending expression (or add the op to _MlxNumpy)."
        )


def _as_array(value, dtype=None):
    """Promote python scalars to ``mx.array``; pass ``mx.array`` through."""
    mx = _mx()
    if isinstance(value, mx.array):
        return value.astype(dtype) if (dtype is not None and value.dtype != dtype) else value
    return mx.array(float(value), dtype=dtype)


def _stack_broadcast(arrs):
    """``mx.stack`` that first broadcasts 0-d entries up to their siblings.

    An equation with no free variables lowers to a python float, which becomes a
    0-d array.  In the scalar case every entry is 0-d and this is a no-op; in the
    batched ``(n, B)`` case the constant rows must be widened to ``(B,)`` or
    ``mx.stack`` refuses the mixed shapes.  Shapes are static at ``mx.compile``
    trace time, so the broadcast costs nothing at run time.
    """
    mx = _mx()
    shape = next((a.shape for a in arrs if a.ndim), ())
    if shape:
        arrs = [a if a.ndim else mx.broadcast_to(a, shape) for a in arrs]
    return mx.stack(arrs)


def _stack(values, dtype):
    """Stack a list of per-equation results into an ``(n,)`` or ``(n, B)`` array."""
    mx = _mx()
    if not values:
        return mx.zeros((0,), dtype=dtype)
    return _stack_broadcast([_as_array(v, dtype) for v in values])


# ---------------------------------------------------------------------------
# Front end 1: SymPy expressions -> MLX callable
# ---------------------------------------------------------------------------
def _sympy_to_mlx(exprs, groups, shim):
    """Lower SymPy expressions to a callable ``f(vec0, vec1, ...) -> list``.

    Parameters
    ----------
    exprs : list
        SymPy expressions (or strings, which are sympified).
    groups : list of (str, list of str)
        Ordered argument groups: ``(vector_name, [symbol names])``.  Every free
        symbol of ``exprs`` must belong to exactly one group; it is replaced by
        ``vector_name[index]`` so the generated code indexes flat MLX arrays.
    """
    import sympy as sp
    from sympy.printing.numpy import NumPyPrinter

    exprs = [sp.sympify(e) for e in exprs]

    slot = {}
    for gi, (_, names) in enumerate(groups):
        for i, name in enumerate(names):
            slot.setdefault(name, (gi, i))

    vecs = [sp.DeferredVector(gname) for gname, _ in groups]

    free = set()
    for expr in exprs:
        free |= expr.free_symbols

    subs, missing = {}, []
    for sym in free:
        name = sym.name
        if name in slot:
            gi, i = slot[name]
            subs[sym] = vecs[gi][i]
        else:
            missing.append(name)
    if missing:
        raise KeyError(
            "symbols present in the equations but not in x/y/u/p: "
            f"{sorted(set(missing))}"
        )

    exprs = [expr.xreplace(subs) for expr in exprs]

    namespace = {"numpy": shim, "functools": functools, "math": math}
    return sp.lambdify(vecs, exprs, modules=[namespace],
                       printer=NumPyPrinter, cse=True)


# ---------------------------------------------------------------------------
# Front end 2: CasADi SX expressions -> MLX callable
# ---------------------------------------------------------------------------
def _casadi_op_table(ca, mx):
    """Map CasADi instruction ids to MLX implementations.

    Comparisons are kept as 0.0/1.0 floats (CasADi's own convention) so that
    products/sums of conditions and ``if_else_zero`` keep working and stay
    differentiable through ``mx.where``.
    """
    def _cmp(fun):
        return lambda a, b: fun(a, b).astype(_as_array(a).dtype)

    unary = {
        ca.OP_NEG: lambda a: -a,
        ca.OP_EXP: mx.exp,
        ca.OP_LOG: mx.log,
        ca.OP_SQRT: mx.sqrt,
        ca.OP_SQ: lambda a: a * a,
        ca.OP_TWICE: lambda a: 2.0 * a,
        ca.OP_SIN: mx.sin,
        ca.OP_COS: mx.cos,
        ca.OP_TAN: mx.tan,
        ca.OP_ASIN: mx.arcsin,
        ca.OP_ACOS: mx.arccos,
        ca.OP_ATAN: mx.arctan,
        ca.OP_SINH: mx.sinh,
        ca.OP_COSH: mx.cosh,
        ca.OP_TANH: mx.tanh,
        ca.OP_ASINH: mx.arcsinh,
        ca.OP_ACOSH: mx.arccosh,
        ca.OP_ATANH: mx.arctanh,
        ca.OP_FABS: mx.abs,
        ca.OP_SIGN: mx.sign,
        ca.OP_FLOOR: mx.floor,
        ca.OP_CEIL: mx.ceil,
        ca.OP_INV: lambda a: 1.0 / a,
        ca.OP_NOT: lambda a: 1.0 - a,
        ca.OP_LIFT: lambda a: a,
        ca.OP_ERF: getattr(mx, "erf", None),
    }
    binary = {
        ca.OP_ADD: lambda a, b: a + b,
        ca.OP_SUB: lambda a, b: a - b,
        ca.OP_MUL: lambda a, b: a * b,
        ca.OP_DIV: lambda a, b: a / b,
        ca.OP_POW: lambda a, b: a ** b,
        ca.OP_CONSTPOW: lambda a, b: a ** b,
        ca.OP_FMIN: mx.minimum,
        ca.OP_FMAX: mx.maximum,
        ca.OP_ATAN2: mx.arctan2,
        ca.OP_FMOD: lambda a, b: a - b * mx.floor(a / b),
        ca.OP_COPYSIGN: lambda a, b: mx.abs(a) * mx.sign(b),
        ca.OP_LT: _cmp(mx.less),
        ca.OP_LE: _cmp(mx.less_equal),
        ca.OP_EQ: _cmp(mx.equal),
        ca.OP_NE: _cmp(mx.not_equal),
        ca.OP_AND: lambda a, b: a * b,
        ca.OP_OR: lambda a, b: mx.maximum(a, b),
        ca.OP_IF_ELSE_ZERO: lambda a, b: mx.where(a != 0, b, 0.0),
    }
    unary = {k: v for k, v in unary.items() if v is not None}
    return unary, binary


def _casadi_to_mlx(exprs, groups):
    """Lower CasADi ``SX`` expressions to a callable ``f(vec0, ...) -> list``.

    The ``SX`` graph is exported as a flat, topologically sorted instruction
    tape and replayed with MLX ops.  Under ``mx.compile`` the replay happens
    once (at trace time), so the per-call cost is the fused Metal graph only.
    """
    import casadi as ca

    mx = _mx()
    expr = ca.vertcat(*exprs) if exprs else ca.SX.zeros(0)
    sym_map = {s.name(): s for s in ca.symvar(expr)}

    ins = []
    for _, names in groups:
        cols = [sym_map.get(n, ca.SX.sym(n)) for n in names]
        ins.append(ca.vertcat(*cols) if cols else ca.SX.zeros(0))

    fun = ca.Function("mlx_tape", ins, [expr])
    unary, binary = _casadi_op_table(ca, mx)

    tape = []
    for k in range(fun.n_instructions()):
        op = fun.instruction_id(k)
        out = list(fun.instruction_output(k))
        inp = list(fun.instruction_input(k))
        const = fun.instruction_constant(k) if op == ca.OP_CONST else None
        tape.append((op, out, inp, const))

    n_out = expr.numel()
    OP_INPUT, OP_OUTPUT, OP_CONST = ca.OP_INPUT, ca.OP_OUTPUT, ca.OP_CONST

    unsupported = {op for op, _, _, _ in tape
                   if op not in binary and op not in unary
                   and op not in (OP_INPUT, OP_OUTPUT, OP_CONST)}
    if unsupported:
        names = {getattr(ca, a): a for a in dir(ca) if a.startswith("OP_")}
        raise NotImplementedError(
            "CasADi operations without an MLX translation: "
            f"{sorted(names.get(op, op) for op in unsupported)}"
        )

    def evaluate(*args):
        work = {}
        out = [0.0] * n_out
        for op, o, i, const in tape:
            if op == OP_INPUT:
                work[o[0]] = args[i[0]][i[1]]
            elif op == OP_OUTPUT:
                out[o[1]] = work[i[0]]
            elif op == OP_CONST:
                work[o[0]] = const
            elif op in binary:
                work[o[0]] = binary[op](work[i[0]], work[i[1]])
            else:
                work[o[0]] = unary[op](work[i[0]])
        return out

    return evaluate


# ---------------------------------------------------------------------------
# AD helpers
# ---------------------------------------------------------------------------
def _jacobian(fun, n_out):
    """Dense Jacobian of ``fun`` with respect to its first argument.

    ``mlx.core`` exposes ``vjp``/``jvp``/``vmap`` but (as of mlx 0.32) no
    ``mx.jacobian``; this builds the reverse-mode equivalent -- one vectorised
    ``vjp`` over the rows of the identity -- and transparently defers to
    ``mx.jacobian`` should a future MLX release provide it.

    Both layouts are handled, chosen by the rank of the primal:

    * ``(n_var,)``   -> Jacobian ``(n_out, n_var)``      -- one system;
    * ``(n_var, B)`` -> Jacobian ``(n_out, n_var, B)``   -- B independent systems
      evaluated at once (the sweep layout).  ``mx.vjp`` requires a cotangent of
      exactly the output's shape, so each basis row is broadcast across the batch.
    """
    mx = _mx()
    native = getattr(mx, "jacobian", None)

    def jac(primal, *rest):
        batched = primal.ndim > 1
        if native is not None and not batched:  # pragma: no cover - future MLX
            # Only trusted for the scalar layout: a native mx.jacobian would not
            # know about the trailing batch axis convention used here.
            return native(lambda v: fun(v, *rest))(primal)

        eye = mx.eye(n_out, dtype=primal.dtype)
        basis = (mx.broadcast_to(eye[:, :, None], (n_out, n_out, primal.shape[1]))
                 if batched else eye)

        def row(cotangent):
            return mx.vjp(lambda v: fun(v, *rest), (primal,), (cotangent,))[1][0]

        return mx.vmap(row)(basis)

    return jac


def _time_call(fun, repeats=5):
    """Milliseconds per call, with one warm-up and the graph forced to run.

    MLX is lazy, so a timing loop that never calls ``mx.eval`` measures graph
    construction and nothing else.
    """
    mx = _mx()
    mx.eval(fun())
    start = time.perf_counter()
    for _ in range(repeats):
        mx.eval(fun())
    return (time.perf_counter() - start) / repeats * 1e3


def _on_stream(method):
    """Run a method's MLX work on the builder's stream.

    Needed because ``float64`` is CPU-only in MLX: with ``dtype='float64'``
    every op (array creation and indexing included) must be issued on the CPU
    stream, while ``float32`` stays on the default Metal device.
    """
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with _mx().stream(self.stream):
            return method(self, *args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# The builder
# ---------------------------------------------------------------------------
class MlxBuilder:
    """Build and solve a pydae ``sys_dict`` on the Apple Silicon GPU with MLX.

    Parameters
    ----------
    sys_dict : dict
        Standard pydae system dictionary (``f_list``, ``g_list``, ``x_list``,
        ``y_ini_list``, ``y_run_list``, ``u_ini_dict``, ``u_run_dict``,
        ``params_dict``, ``h_dict``, ``xy_0_dict``).  Expressions may be SymPy
        or CasADi ``SX``.
    dtype : {'float32', 'float64'}
        ``float32`` keeps everything on the Metal GPU (default).  ``float64``
        runs on the CPU stream (MLX has no double precision on Metal) and is
        the right choice for stiff or badly scaled networks.  See the module
        docstring: MLX's sin/cos/exp are float32-accurate in both cases.
    Dt : float
        Integration step of the trapezoidal rule.
    alpha : float
        ``0.5`` = trapezoidal, ``1.0`` = backward Euler.
    decimation : int
        Store one output sample every ``decimation`` steps.
    newton_tol : float, optional
        Absolute residual tolerance ``itol`` (default 1e-10 for ``float64``,
        1e-5 for ``float32``).
    newton_max_iter : int
        Maximum Newton iterations per solve.
    sparse : bool
        Exploit the structural sparsity of every Jacobian (default).  Each
        block is compressed by Curtis-Powell-Reid colouring, so it costs a
        handful of AD passes instead of one per equation -- 9 instead of 87 on
        the Kundur two-area system.  Per block the compression is used only
        when it needs strictly fewer passes than dense reverse mode, so
        ``sparse=True`` is never slower; ``False`` forces the dense path and
        exists to isolate the compression when something looks wrong.  See
        :mod:`pydae.core.builder.mlx_sparsity`.  Note that the *linear solve*
        stays dense either way: MLX's immutable arrays make an on-device
        sparse factorization slower than the dense one it would replace.
    verbose : bool
        Print build/initialization progress.

    Notes
    -----
    After :meth:`build` the instance carries the name lists ``x_names``,
    ``y_ini_names``, ``y_run_names``, ``u_ini_names``, ``u_run_names``,
    ``p_names`` and ``z_names`` (plus the ``*_list`` aliases of the ``Model`` /
    ``CasadiModel`` accessor surface), and the ``CasadiBuilder``-shaped
    evaluators ``_residual_fn``, ``_jacobian_fn``, ``_h_fn`` and ``_Fx_fn`` …
    ``_Hu_fn``, which take ``(x, y, p_run)`` numpy vectors and return numpy
    arrays.
    """

    def __init__(self, sys_dict, dtype="float32", Dt=0.01, alpha=0.5,
                 decimation=1, newton_tol=None, newton_max_iter=25,
                 sparse=True, verbose=True):
        self.sys_dict = sys_dict
        self.name = sys_dict.get("name", "mlx_model")
        self.verbose = verbose
        self.sparse = sparse
        self.patterns = {}
        self._sparse_used = {}

        mx = _mx()
        if dtype in ("float64", "double", np.float64):
            if not hasattr(mx, "float64"):  # pragma: no cover
                raise ValueError("this MLX build has no float64 support")
            self.dtype = mx.float64
            self.stream = mx.cpu
            default_tol, default_xtol, default_stall = 1e-10, 1e-13, 1e-6
        else:
            self.dtype = mx.float32
            self.stream = mx.default_device()
            # float32 has ~1e-7 relative resolution: an absolute residual
            # below ~1e-5 is the practical floor on a well-scaled system.
            default_tol, default_xtol, default_stall = 1e-5, 1e-6, 1e-4
        self.np_dtype = np.float64 if self.dtype == getattr(mx, "float64", None) else np.float32

        # runtime tunables (same names as the ctypes/CasADi Model surface)
        self.Dt = Dt
        self.alpha = alpha
        self.decimation = decimation
        self.itol = newton_tol if newton_tol is not None else default_tol
        # Newton also stops when the update stops making progress: with the
        # residual criterion alone a solve can stall just above itol. MLX
        # evaluates sin/cos/exp to float32 accuracy even on float64 arrays, so
        # on a trigonometric model (any power system) the residual bottoms out
        # around 1e-8 no matter the dtype -- 'stall_tol' is the residual still
        # accepted once Newton demonstrably stops improving.
        self.xtol = default_xtol
        self.stall_tol = default_stall
        self.max_it = newton_max_iter
        self._floor_warned = False
        self.t = 0.0

        self.Time, self.X, self.Y, self.Z = [], [], [], []
        self._built = False

    # -- small helpers ------------------------------------------------------
    @staticmethod
    def _name(sym):
        if isinstance(sym, str):
            return sym
        namer = getattr(sym, "name", None)
        return namer() if callable(namer) else str(sym)

    def _log(self, msg):
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------------
    # BUILD
    # ------------------------------------------------------------------
    @_on_stream
    def build(self):
        """Translate the system into compiled MLX graphs. Returns ``self``."""
        t0 = time.time()
        mx = _mx()
        sd = self.sys_dict
        self._log(f"Building model '{self.name}' with the MLX backend "
                  f"({'float64/cpu' if self.np_dtype is np.float64 else 'float32/gpu'})...")

        # -- 1. variable names ------------------------------------------
        self.y_ini_names = [self._name(s) for s in sd["y_ini_list"]]
        self.y_run_names = [self._name(s) for s in sd["y_run_list"]]
        # A symbol declared both as a state and as an ini algebraic variable is
        # algebraic during initialization (same rule as CasadiBuilder).
        yini_set = set(self.y_ini_names)
        self.x_names = [n for n in (self._name(s) for s in sd["x_list"])
                        if n not in yini_set]
        self.p_names = list(sd.get("params_dict", {}).keys())
        self.u_ini_names = list(sd.get("u_ini_dict", {}).keys())
        self.u_run_names = list(sd.get("u_run_dict", {}).keys())
        self.z_names = list(sd.get("h_dict", {}).keys())

        # aliases matching the Model / CasadiModel accessor surface
        self.x_list = self.x_names
        self.y_ini_list = self.y_ini_names
        self.y_run_list = self.y_run_names
        self.y_list = self.y_run_names
        self.u_ini_list = self.u_ini_names
        self.u_run_list = self.u_run_names
        self.params_list = self.p_names
        self.z_list = self.z_names
        self.outputs_list = self.z_names

        self.N_x = len(self.x_names)
        self.N_y_ini = len(self.y_ini_names)
        self.N_y_run = len(self.y_run_names)
        self.N_z = len(self.z_names)

        # -- 2. equations and front end ---------------------------------
        f_list = list(sd.get("f_list", []))
        g_list = list(sd.get("g_list", []))
        h_list = list(sd.get("h_dict", {}).values())
        self.N_f, self.N_g = len(f_list), len(g_list)

        self.frontend = self._detect_frontend(f_list + g_list + h_list)
        self._log(f"  front end        : {self.frontend}")
        self._log(f"  states / alg     : {self.N_x} / {self.N_y_run} "
                  f"(ini alg: {self.N_y_ini})")

        # -- 3. name routing (numeric equivalent of the symbolic substitutions)
        eq_names = self._equation_symbol_names(f_list + g_list + h_list)

        # ini phase: unknowns [x; y_ini], knowns [p; u_ini], rest -> constants
        ini_routed = set(self.x_names) | yini_set | set(self.p_names) | set(self.u_ini_names)
        self.c_ini_names = sorted(n for n in eq_names if n not in ini_routed)
        # run phase: unknowns [x; y_run], knowns [p; u_run], rest -> constants
        run_routed = (set(self.x_names) | set(self.y_run_names)
                      | set(self.p_names) | set(self.u_run_names))
        self.c_run_names = sorted(n for n in eq_names if n not in run_routed)

        self.c_ini_vals = np.array([self._guess(n) for n in self.c_ini_names],
                                   dtype=self.np_dtype)
        self.c_run_vals = np.array([self._guess(n) for n in self.c_run_names],
                                   dtype=self.np_dtype)
        if self.c_ini_names:
            self._log(f"  frozen at ini    : {len(self.c_ini_names)} symbol(s) "
                      f"(y_run-only), e.g. {self.c_ini_names[:4]}")
        if self.c_run_names:
            self._log(f"  frozen at run    : {len(self.c_run_names)} symbol(s) "
                      f"(y_ini-only), e.g. {self.c_run_names[:4]}")

        ini_groups = [("v", self.x_names + self.y_ini_names),
                      ("p", self.p_names),
                      ("u", self.u_ini_names),
                      ("c", self.c_ini_names)]
        run_groups = [("x", self.x_names),
                      ("y", self.y_run_names),
                      ("p", self.p_names),
                      ("u", self.u_run_names),
                      ("c", self.c_run_names)]

        # -- 4. lower the equations -------------------------------------
        lower = self._lower_sympy if self.frontend == "sympy" else self._lower_casadi
        fg_ini_raw = lower(f_list + g_list, ini_groups)
        f_run_raw = lower(f_list, run_groups)
        g_run_raw = lower(g_list, run_groups)
        h_run_raw = lower(h_list, run_groups) if h_list else None

        dtype = self.dtype
        N_x, N_f = self.N_x, self.N_f

        def fg_ini(v, p, u, c):
            return _stack(fg_ini_raw(v, p, u, c), dtype)

        def f_run(x, y, p, u, c):
            return _stack(f_run_raw(x, y, p, u, c), dtype)

        def g_run(x, y, p, u, c):
            return _stack(g_run_raw(x, y, p, u, c), dtype)

        self._fg_ini = fg_ini
        self._f_run = f_run
        self._g_run = g_run
        self._h_run = ((lambda x, y, p, u, c: _stack(h_run_raw(x, y, p, u, c), dtype))
                       if h_run_raw is not None else None)

        # -- 5. structural sparsity of every Jacobian --------------------
        self._build_patterns(f_list, g_list, h_list)

        # -- 6. numeric state -------------------------------------------
        # Ahead of the kernels: the seed point is what the sparsity check and
        # the dense-vs-compressed timing run on, and both have to settle before
        # a compiled kernel closes over the Jacobian it will use.
        self._reset_state()

        # 0 = x - x_k - Dt[(1-a) f_k + a f(x,y)] ; 0 = g(x, y)
        def trap_residual(w, x_k, f_k, p, u, c, dt, alpha):
            x_n, y_n = w[:N_x], w[N_x:]
            f_n = f_run(x_n, y_n, p, u, c)
            res_f = x_n - x_k - dt * ((1.0 - alpha) * f_k + alpha * f_n)
            return mx.concatenate([res_f, g_run(x_n, y_n, p, u, c)])

        self._trap_residual = trap_residual

        # -- 7. pick the Jacobian for each phase ------------------------
        jac_ini = self._select_jacobian(fg_ini, N_f + self.N_g, "ini")
        jac_trap = self._select_jacobian(trap_residual, N_x + self.N_y_run,
                                         "trap")
        self._jac_ini, self._jac_trap = jac_ini, jac_trap
        # The sweep always compresses: at B >= 100 the AD passes carry real
        # work, so fewer of them wins by a margin no fixed overhead touches.
        self._jac_ini_batched = self._make_jacobian(
            fg_ini, N_f + self.N_g, "ini", force=True)
        self._jac_trap_batched = self._make_jacobian(
            trap_residual, N_x + self.N_y_run, "trap", force=True)

        # -- 8. compiled Newton step: BACKWARD INITIALIZATION ------------
        cpu = mx.cpu

        @mx.compile
        def newton_step_ini(v, p, u, c, relax):
            """One Newton-Raphson iteration of the initialization system.

            Returns ``(v_next, ||residual||_inf, relative step size)``.  The
            whole iteration -- equation evaluation, AD Jacobian and linear
            solve -- is traced by ``mx.compile`` into a single graph.
            """
            residual = fg_ini(v, p, u, c)
            jacobian = jac_ini(v, p, u, c)
            step = mx.linalg.solve(jacobian, -residual, stream=cpu)
            v_next = v + relax * step
            rel = mx.max(mx.abs(step)) / (1.0 + mx.max(mx.abs(v_next)))
            return v_next, mx.max(mx.abs(residual)), rel

        self._newton_step_ini = newton_step_ini

        # -- 9. compiled Newton step: FORWARD (trapezoidal) --------------
        @mx.compile
        def newton_step_run(w, x_k, f_k, p, u, c, dt, alpha, relax):
            """One Newton-Raphson iteration of the trapezoidal step.

            Returns ``(w_next, ||residual||_inf, relative step size)``.
            """
            residual = trap_residual(w, x_k, f_k, p, u, c, dt, alpha)
            jacobian = jac_trap(w, x_k, f_k, p, u, c, dt, alpha)
            step = mx.linalg.solve(jacobian, -residual, stream=cpu)
            w_next = w + relax * step
            rel = mx.max(mx.abs(step)) / (1.0 + mx.max(mx.abs(w_next)))
            return w_next, mx.max(mx.abs(residual)), rel

        self._newton_step_run = newton_step_run

        # -- 10. state-space blocks (AD, evaluated on demand) ------------
        self._build_block_jacobians()

        # CasadiBuilder-compatible extras
        sd["p_list"] = self.p_names
        sd["u_ini_list"] = self.u_ini_names
        sd["u_run_list"] = self.u_run_names
        self.dae_dict = {"x": self.x_names, "z": self.y_run_names,
                         "p": self.p_names + self.u_run_names,
                         "ode": self._f_run, "alg": self._g_run}

        self._built = True
        self._log(f"Build complete! ({time.time() - t0:.2f} s)")
        return self

    # -- build helpers ------------------------------------------------------
    def _detect_frontend(self, exprs):
        for expr in exprs:
            mod = type(expr).__module__
            if mod.startswith("casadi"):
                return "casadi"
            if mod.startswith("sympy"):
                return "sympy"
        return "sympy"

    def _equation_symbol_names(self, exprs):
        """All free-symbol names appearing in the equations."""
        if self.frontend == "casadi":
            import casadi as ca
            if not exprs:
                return set()
            return {s.name() for s in ca.symvar(ca.vertcat(*exprs))}
        import sympy as sp
        names = set()
        for expr in exprs:
            names |= {s.name for s in sp.sympify(expr).free_symbols}
        return names

    def _guess(self, name):
        """Value frozen into an equation for a symbol absent from this phase.

        Same heuristic as ``CasadiBuilder``: take ``xy_0_dict`` when available,
        otherwise 1.0 for bus voltage magnitudes and 0.0 for everything else.
        """
        xy_0 = self.sys_dict.get("xy_0_dict", {})
        if name in xy_0:
            return float(xy_0[name])
        return 1.0 if name.startswith("V_") else 0.0

    def _lower_sympy(self, exprs, groups):
        return _sympy_to_mlx(exprs, groups, _MlxNumpy())

    def _lower_casadi(self, exprs, groups):
        return _casadi_to_mlx(exprs, groups)

    def _build_patterns(self, f_list, g_list, h_list):
        """Structural sparsity of every Jacobian this backend evaluates.

        The free-symbol scan is the expensive half, so it runs once per
        equation list and each block then just keeps a different set of
        columns.  Patterns are stored on ``self.patterns`` for inspection --
        ``model.patterns['ini']`` reports nnz, colours and pass count.
        """
        from pydae.core.builder.mlx_sparsity import JacobianPattern, symbol_sets

        f_symbols = symbol_sets(f_list, self.frontend)
        g_symbols = symbol_sets(g_list, self.frontend)
        h_symbols = symbol_sets(h_list, self.frontend)

        def pattern(equation_symbols, groups, n_rows, extra_diagonal=0):
            """Nonzeros of d(equations)/d(concatenated variable groups)."""
            rows, cols, offset = [], [], 0
            for names in groups:
                column_of = {name: j + offset for j, name in enumerate(names)}
                for i, symbols in enumerate(equation_symbols):
                    for symbol in symbols:
                        j = column_of.get(symbol)
                        if j is not None:
                            rows.append(i)
                            cols.append(j)
                offset += len(names)
            # The trapezoidal residual x_n - x_k - dt(...) carries an explicit
            # x_n, so its Jacobian has a diagonal the equations themselves
            # never show.  Missing it would silently drop the identity block.
            for i in range(extra_diagonal):
                rows.append(i)
                cols.append(i)
            return JacobianPattern(rows, cols, n_rows, offset)

        x_names, y_run, u_run = self.x_names, self.y_run_names, self.u_run_names
        fg_symbols = f_symbols + g_symbols
        n_fg = self.N_f + self.N_g

        self.patterns["ini"] = pattern(
            fg_symbols, [x_names, self.y_ini_names], n_fg)
        self.patterns["trap"] = pattern(
            fg_symbols, [x_names, y_run], self.N_x + self.N_y_run,
            extra_diagonal=min(self.N_x, self.N_f))
        # The augmented Jacobian [Fx Fy; Gx Gy] of the small-signal pencil --
        # the trapezoidal pattern without the step's identity block.
        self.patterns["aug"] = pattern(fg_symbols, [x_names, y_run], n_fg)
        for key, symbols, n_rows in (("F", f_symbols, self.N_f),
                                     ("G", g_symbols, self.N_g),
                                     ("H", h_symbols, self.N_z)):
            for suffix, names in (("x", x_names), ("y", y_run), ("u", u_run)):
                self.patterns[key + suffix] = pattern(symbols, [names], n_rows)

        if self.sparse:
            for label in ("ini", "trap"):
                block = self.patterns[label]
                self._log(f"  {('sparsity (' + label + ')'):<17}: {block.nnz} nnz "
                          f"({100 * block.density:.1f}%), {block.n_passes} AD "
                          f"passes vs {block.n_rows} dense"
                          + ("" if block.worthwhile else " -- dense path kept"))

    def _seed_point(self, phase):
        """``(primal, rest)`` at the seed state, for checking and timing."""
        if phase == "ini":
            return (self._arr(np.concatenate([self.x, self.y_ini])),
                    (self._arr(self.p_vals), self._arr(self.u_ini_vals),
                     self._arr(self.c_ini_vals)))
        x_dev = self._arr(self.x)
        rest = (self._arr(self.p_vals), self._arr(self.u_run_vals),
                self._arr(self.c_run_vals))
        f_k = self._f_run(x_dev, self._arr(self.y_run), *rest)
        return (self._arr(np.concatenate([self.x, self.y_run])),
                (x_dev, f_k) + rest + (self.Dt, self.alpha))

    def _select_jacobian(self, fun, n_rows, key):
        """Choose dense or compressed AD for the *scalar* solves, by measuring.

        Compression always cuts the number of AD passes, but at batch width 1
        that is not what a pass costs: ``mx.vmap`` batches the identity rows
        into one dispatch, so evaluating 121 of them can cost barely more than
        4, and the fixed price of expanding the compressed result back to a
        dense ``n x n`` then dominates.  Which side wins depends on how much
        arithmetic each equation carries -- compression is 1.3-1.6x ahead on
        the trigonometric power-system models here, and about 15 % behind on a
        model built from many trivially cheap equations.

        Too model-dependent to settle with a rule, and cheap to settle with a
        stopwatch, so the builder times both on the seed point and keeps the
        winner.  Compression has to be clearly ahead to be chosen, which keeps
        timing noise from deciding it.  The batched Jacobians are not chosen
        this way: there the AD passes carry real work and compression wins
        outright.
        """
        pattern = self.patterns.get(key)
        dense = _jacobian(fun, n_rows)
        if not self.sparse or pattern is None or not pattern.worthwhile:
            return dense

        compressed = self._make_jacobian(fun, n_rows, key, force=True)
        primal, rest = self._seed_point(key)
        self._check_jacobian(key, fun, compressed, primal, rest)

        dense_time = _time_call(lambda: dense(primal, *rest))
        sparse_time = _time_call(lambda: compressed(primal, *rest))
        chosen = sparse_time * 1.05 < dense_time
        self._log(f"  {('sparsity (' + key + ')'):<17}: scalar {dense_time:.2f} ms "
                  f"dense vs {sparse_time:.2f} ms compressed -> "
                  f"{'compressed' if chosen else 'dense'}")
        self._sparse_used[key] = chosen
        return compressed if chosen else dense

    def _check_jacobian(self, key, fun, jacobian, primal, rest):
        """Prove a compressed Jacobian against a plain directional derivative.

        One extra AD pass.  The structural pattern can only over-count entries,
        never miss one, so what is actually at risk is the recovery machinery
        -- colouring, the mask, the dense-row split -- and ``J @ tangent``
        against ``jvp(tangent)`` exercises all of it.  Cheap insurance against
        a mis-recovered Jacobian surfacing much later as a Newton that merely
        converges slowly.
        """
        from pydae.core.builder.mlx_sparsity import verify_pattern

        tolerance = 1e-9 if self.np_dtype is np.float64 else 1e-4
        discrepancy, ok = verify_pattern(fun, jacobian, primal, rest, tolerance)
        if not ok:
            raise RuntimeError(
                f"the compressed '{key}' Jacobian disagrees with a plain "
                f"directional derivative by {discrepancy:.2e} (tolerance "
                f"{tolerance:.0e}). This is a bug in the sparsity machinery, "
                f"not in your model -- rebuild with sparse=False and please "
                f"report it.")

    def _make_jacobian(self, fun, n_rows, key, force=False):
        """Jacobian evaluator for ``fun``: compressed when that wins, else dense.

        Compression is skipped whenever it would not need strictly fewer AD
        passes than plain reverse mode, so enabling ``sparse`` cannot make a
        dense system slower -- it just does nothing.  ``force`` takes the
        compressed path without the scalar timing test, for the batched
        callers where it is unconditionally the right choice.
        """
        pattern = self.patterns.get(key)
        if not self.sparse or pattern is None or not pattern.worthwhile:
            return _jacobian(fun, n_rows)
        from pydae.core.builder.mlx_sparsity import compressed_jacobian
        if force:
            self._sparse_used[key] = True
        return compressed_jacobian(fun, pattern, self.dtype, self.stream)

    def _build_block_jacobians(self):
        """AD evaluators for Fx, Fy, Gx, Gy, Fu, Gu, Hx, Hy, Hu."""
        f_run, g_run, h_run = self._f_run, self._g_run, self._h_run
        N_f, N_g, N_z = self.N_f, self.N_g, self.N_z

        def swap(fun, pos):
            """Move argument ``pos`` of ``fun(x, y, p, u, c)`` to the front."""
            def wrapped(first, *rest):
                args = list(rest)
                args.insert(pos, first)
                return fun(*args)
            return wrapped

        def reorder(args, pos):
            args = list(args)
            return [args.pop(pos)] + args

        def make(fun, n_out, pos, key):
            jac = self._make_jacobian(swap(fun, pos), n_out, key)
            return lambda *args: jac(*reorder(args, pos))

        self._Fx = make(f_run, N_f, 0, "Fx")
        self._Fy = make(f_run, N_f, 1, "Fy")
        self._Fu = make(f_run, N_f, 3, "Fu")
        self._Gx = make(g_run, N_g, 0, "Gx")
        self._Gy = make(g_run, N_g, 1, "Gy")
        self._Gu = make(g_run, N_g, 3, "Gu")
        if h_run is not None:
            self._Hx = make(h_run, N_z, 0, "Hx")
            self._Hy = make(h_run, N_z, 1, "Hy")
            self._Hu = make(h_run, N_z, 3, "Hu")
        else:
            self._Hx = self._Hy = self._Hu = None

        # CasadiBuilder-compatible names: fn(x, y, p_run) -> numpy array,
        # with p_run = concatenate((p_vals, u_run_vals)).
        def compat(block):
            if block is None:
                return None

            def call(x, y, p_run):
                p_run = np.asarray(p_run, dtype=self.np_dtype).flatten()
                n_p = len(self.p_names)
                with _mx().stream(self.stream):
                    return np.array(block(
                        self._arr(x), self._arr(y), self._arr(p_run[:n_p]),
                        self._arr(p_run[n_p:]), self._arr(self.c_run_vals)))
            return call

        self._Fx_fn, self._Fy_fn = compat(self._Fx), compat(self._Fy)
        self._Gx_fn, self._Gy_fn = compat(self._Gx), compat(self._Gy)
        self._Fu_fn, self._Gu_fn = compat(self._Fu), compat(self._Gu)
        self._Hx_fn, self._Hy_fn = compat(self._Hx), compat(self._Hy)
        self._Hu_fn = compat(self._Hu)

        def residual_fn(v, p_ini):
            n_p = len(self.p_names)
            p_ini = np.asarray(p_ini, dtype=self.np_dtype).flatten()
            with _mx().stream(self.stream):
                return np.array(self._fg_ini(
                    self._arr(v), self._arr(p_ini[:n_p]),
                    self._arr(p_ini[n_p:]), self._arr(self.c_ini_vals)))

        jac_ini = self._jac_ini

        def jacobian_fn(v, p_ini):
            n_p = len(self.p_names)
            p_ini = np.asarray(p_ini, dtype=self.np_dtype).flatten()
            with _mx().stream(self.stream):
                return np.array(jac_ini(
                    self._arr(v), self._arr(p_ini[:n_p]),
                    self._arr(p_ini[n_p:]), self._arr(self.c_ini_vals)))

        self._residual_fn = residual_fn
        self._jacobian_fn = jacobian_fn

        def h_fn(x, y, p_run):
            if self._h_run is None:
                return np.zeros(0)
            n_p = len(self.p_names)
            p_run = np.asarray(p_run, dtype=self.np_dtype).flatten()
            with _mx().stream(self.stream):
                return np.array(self._h_run(
                    self._arr(x), self._arr(y), self._arr(p_run[:n_p]),
                    self._arr(p_run[n_p:]), self._arr(self.c_run_vals)))

        self._h_fn = h_fn if self._h_run is not None else None

    def _reset_state(self):
        sd = self.sys_dict
        self.x = np.zeros(self.N_x, dtype=self.np_dtype)
        self.y_ini = np.zeros(self.N_y_ini, dtype=self.np_dtype)
        self.y_run = np.zeros(self.N_y_run, dtype=self.np_dtype)
        self.z = np.zeros(self.N_z, dtype=self.np_dtype)
        self.t = 0.0

        self.p_vals = np.array([sd.get("params_dict", {}).get(n, 0.0)
                                for n in self.p_names], dtype=self.np_dtype)
        self.u_ini_vals = np.array([sd.get("u_ini_dict", {}).get(n, 0.0)
                                    for n in self.u_ini_names], dtype=self.np_dtype)
        self.u_run_vals = np.array([sd.get("u_run_dict", {}).get(n, 0.0)
                                    for n in self.u_run_names], dtype=self.np_dtype)

        xy_0 = sd.get("xy_0_dict", {})
        for i, name in enumerate(self.x_names):
            if name in xy_0:
                self.x[i] = xy_0[name]
        for i, name in enumerate(self.y_ini_names):
            if name in xy_0:
                self.y_ini[i] = xy_0[name]
        for i, name in enumerate(self.y_run_names):
            if name in xy_0:
                self.y_run[i] = xy_0[name]

        self.Time, self.X, self.Y, self.Z = [], [], [], []

    @_on_stream
    def _arr(self, values):
        mx = _mx()
        return mx.array(np.asarray(values, dtype=self.np_dtype).flatten(),
                        dtype=self.dtype)

    @_on_stream
    def _arr2(self, values):
        """Build an ``(n, B)`` device array, preserving the batch axis."""
        mx = _mx()
        return mx.array(np.ascontiguousarray(values, dtype=self.np_dtype),
                        dtype=self.dtype)

    def _require_built(self):
        if not self._built:
            raise RuntimeError("call build() before ini()/run()")

    def _locate(self, name, phase=None):
        """Every ``(block, index)`` slot a name occupies.

        ``_route_dict`` deliberately stops at the first match, because a scalar
        update targets one phase.  A *sweep* value defines the case, so it has to
        reach every block that carries the name -- ``p_m_1``, for instance, is
        registered in both ``u_ini_dict`` and ``u_run_dict``, and writing only
        the ini copy would leave the transient running at nominal dispatch.
        """
        blocks = (("p", self.p_names), ("u_ini", self.u_ini_names),
                  ("u_run", self.u_run_names), ("x", self.x_names),
                  ("y_ini", self.y_ini_names), ("y_run", self.y_run_names))
        if phase == "ini":
            blocks = tuple(b for b in blocks if b[0] != "u_run")
        elif phase == "run":
            blocks = tuple(b for b in blocks if b[0] != "u_ini")
        return [(block, names.index(name)) for block, names in blocks
                if name in names]

    def small_signal(self, Shift=-0.5, **kwargs):
        """Sparse small-signal analysis of the current operating point.

        Solves the generalized problem ``Lambda_xi @ phi = lambda * E @ phi`` by
        matrix-free shift-invert Arnoldi, so the reduced state matrix
        ``A = Fx - Fy Gy^-1 Gx`` is never formed and ``Gy`` is never inverted.
        See :func:`pydae.core.builder.mlx_ssa.small_signal` for the full
        signature.

        Examples
        --------
        ::

            model.ini({}, xy_0=...)
            result = model.small_signal(Shift=0.0, N_modes=4)
            result.eigenvalues, result.damping, result.participation
        """
        from pydae.core.builder.mlx_ssa import small_signal
        return small_signal(self, Shift=Shift, **kwargs)

    def sweep(self, values=None, **kwargs):
        """Create a batched :class:`~pydae.core.builder.mlx_sweep.MlxSweep`.

        Parameters
        ----------
        values : dict
            ``name -> scalar | sequence``.  Sequences define the batch width B;
            scalars broadcast to every case.

        Examples
        --------
        ::

            m.ini({'p_m_1': 0.5})                       # nominal anchor
            sw = m.sweep({'p_m_1': np.linspace(0.05, 0.95, 10_000)})
            sw.ini()
            q = sw.get_values('q_g_1')                  # (10000,)
        """
        from pydae.core.builder.mlx_sweep import MlxSweep
        return MlxSweep(self, values, **kwargs)

    # ------------------------------------------------------------------
    # Newton-Raphson driver (shared by ini and by each trapezoidal step)
    # ------------------------------------------------------------------
    def _newton(self, step, w, where):
        """Iterate ``step`` (a compiled Newton iteration) until convergence.

        Three stopping rules, in order of strength:

        1. ``||residual||_inf < itol`` -- the textbook test;
        2. the relative Newton update drops below ``xtol`` -- the iterate has
           stopped moving, so no further digits are available;
        3. the residual stops improving for two iterations while already below
           ``stall_tol`` -- the arithmetic floor has been reached (MLX's
           sin/cos/exp are float32-accurate even in float64, which pins the
           residual of a trigonometric model near 1e-8).

        Returns ``(w, residual)``; raises ``RuntimeError`` if none is met.
        """
        mx = _mx()
        best, stall, res, rel = float("inf"), 0, float("inf"), float("inf")
        for _ in range(self.max_it):
            w, res, rel = step(w)
            mx.eval(w, res, rel)
            res, rel = float(res), float(rel)
            if not np.isfinite(res) or not np.isfinite(rel):
                raise RuntimeError(
                    f"MLX Newton diverged {where} (non-finite residual/step): "
                    "the Jacobian is likely singular -- check for zero rows or "
                    "columns and near-zero pivots.")
            if res < self.itol or rel < self.xtol:
                return w, res
            stall = stall + 1 if res > 0.5 * best else 0
            best = min(best, res)
            if stall >= 2 and res < self.stall_tol:
                if not self._floor_warned:
                    self._floor_warned = True
                    self._log(
                        f"note: Newton stopped improving at ||residual||_inf = "
                        f"{res:.2e} (> itol = {self.itol:.1e}); this is MLX's "
                        "float32-accurate sin/cos/exp floor, not divergence.")
                return w, res
        raise RuntimeError(
            f"MLX Newton-Raphson did not converge {where} in {self.max_it} "
            f"iterations (||residual||_inf = {res:.3e}, itol = {self.itol:.1e}, "
            f"relative step = {rel:.3e}). Improve the initial guess, raise "
            "max_it, or reduce Dt.")

    # ------------------------------------------------------------------
    # BACKWARD INITIALIZATION
    # ------------------------------------------------------------------
    @_on_stream
    def ini(self, params_dict=None, xy_0=None, tol=None, max_iter=None):
        """Solve the steady state ``0 = f(x, y_ini, u_ini, p) = g(...)``.

        Parameters
        ----------
        params_dict : dict, optional
            Parameter / input overrides (routed to ``p`` or ``u_ini``).
        xy_0 : dict or str, optional
            Initial guess (dict, or path to a JSON file as saved by pydae).
        """
        self._require_built()
        self._route_dict(params_dict, phase="ini")

        v0 = np.concatenate((self.x, self.y_ini)).astype(self.np_dtype)
        if isinstance(xy_0, str):
            with open(xy_0, encoding="utf-8") as fobj:
                xy_0 = json.load(fobj)
        if isinstance(xy_0, dict):
            for key, val in xy_0.items():
                if key in self.x_names:
                    v0[self.x_names.index(key)] = val
                elif key in self.y_ini_names:
                    v0[self.N_x + self.y_ini_names.index(key)] = val

        tol = self.itol if tol is None else tol
        max_iter = self.max_it if max_iter is None else max_iter

        v = self._arr(v0)
        p, u = self._arr(self.p_vals), self._arr(self.u_ini_vals)
        c = self._arr(self.c_ini_vals)

        relax = self._arr([1.0])[0]
        itol_saved, max_it_saved = self.itol, self.max_it
        self.itol, self.max_it = tol, max_iter
        try:
            v, res = self._newton(
                lambda w: self._newton_step_ini(w, p, u, c, relax),
                v, "during initialization")
        finally:
            self.itol, self.max_it = itol_saved, max_it_saved

        v_np = np.array(v).astype(self.np_dtype)
        self.x = v_np[:self.N_x]
        self.y_ini = v_np[self.N_x:]
        self.t = 0.0

        self.ini2run()

        self.Time = [self.t]
        self.X = [self.x.copy()]
        self.Y = [self.y_run.copy()]
        self.Z = [self._eval_h()] if self._h_run is not None else []
        self._log(f"ini converged: ||residual||_inf = {res:.3e}")
        return self

    def ini2run(self):
        """Transfer the converged ini point onto the run variables/inputs."""
        val_map = dict(zip(self.x_names, self.x))
        val_map.update(dict(zip(self.y_ini_names, self.y_ini)))
        val_map.update(dict(zip(self.u_ini_names, self.u_ini_vals)))

        for i, name in enumerate(self.y_run_names):
            if name in val_map:
                self.y_run[i] = val_map[name]
        # y_ini/u_ini variables that become fixed inputs during the run
        # (e.g. v_ref_* solved at ini): transfer their converged values.
        ini_side = set(self.y_ini_names) | set(self.u_ini_names)
        for i, name in enumerate(self.u_run_names):
            if name in ini_side and name in val_map:
                self.u_run_vals[i] = val_map[name]

    # ------------------------------------------------------------------
    # FORWARD SIMULATION (implicit trapezoidal rule)
    # ------------------------------------------------------------------
    @_on_stream
    def run(self, t_end, update_dict=None, Dt=None, alpha=None):
        """Integrate to ``t_end`` (absolute time) with the trapezoidal rule.

        Consecutive calls chain seamlessly: each continues from the previous
        end point, so piecewise-constant input changes need no re-``ini()``.
        """
        self._require_built()
        self._route_dict(update_dict, phase="run")

        mx = _mx()
        Dt = self.Dt if Dt is None else Dt
        alpha = self.alpha if alpha is None else alpha

        p, u = self._arr(self.p_vals), self._arr(self.u_run_vals)
        c = self._arr(self.c_run_vals)
        dt_a, alpha_a = self._arr([Dt])[0], self._arr([alpha])[0]
        relax = self._arr([1.0])[0]

        x = self._arr(self.x)
        y = self._arr(self.y_run)

        n_steps = int(round((t_end - self.t) / Dt))
        if n_steps <= 0:
            return self

        for step in range(n_steps):
            f_k = self._f_run(x, y, p, u, c)          # explicit half of the rule
            w = mx.concatenate([x, y])
            w, res = self._newton(
                lambda v: self._newton_step_run(v, x, f_k, p, u, c,
                                                dt_a, alpha_a, relax),
                w, f"at t = {self.t:.4f} s")

            x, y = w[:self.N_x], w[self.N_x:]
            self.t += Dt

            if (step + 1) % self.decimation == 0 or step == n_steps - 1:
                self.x = np.array(x).astype(self.np_dtype)
                self.y_run = np.array(y).astype(self.np_dtype)
                self.Time.append(self.t)
                self.X.append(self.x.copy())
                self.Y.append(self.y_run.copy())
                if self._h_run is not None:
                    self.Z.append(self._eval_h())

        self.x = np.array(x).astype(self.np_dtype)
        self.y_run = np.array(y).astype(self.np_dtype)
        return self

    def post(self):
        """Freeze the stored trajectory into numpy arrays."""
        self.Time = np.array(self.Time)
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        self.Z = np.array(self.Z) if len(self.Z) else np.zeros((len(self.Time), 0))
        return self

    # ------------------------------------------------------------------
    # accessors (same surface as Model / CasadiModel)
    # ------------------------------------------------------------------
    def _route_dict(self, update_dict, phase):
        if not update_dict:
            return
        for key, val in update_dict.items():
            if key in self.p_names:
                self.p_vals[self.p_names.index(key)] = val
            elif phase == "ini" and key in self.u_ini_names:
                self.u_ini_vals[self.u_ini_names.index(key)] = val
            elif phase == "run" and key in self.u_run_names:
                self.u_run_vals[self.u_run_names.index(key)] = val
            elif key in self.y_ini_names:
                self.y_ini[self.y_ini_names.index(key)] = val
            elif key in self.y_run_names:
                self.y_run[self.y_run_names.index(key)] = val
            elif key in self.x_names:
                self.x[self.x_names.index(key)] = val
            else:
                raise KeyError(f"unknown variable '{key}'")

    @_on_stream
    def _eval_h(self):
        z = np.array(self._h_run(self._arr(self.x), self._arr(self.y_run),
                                 self._arr(self.p_vals), self._arr(self.u_run_vals),
                                 self._arr(self.c_run_vals))).astype(self.np_dtype)
        self.z = z
        return z

    def set_value(self, name, value):
        self._route_dict({name: value}, phase="run")

    @_on_stream
    def get_value(self, name):
        """Value of a state, algebraic variable, input, parameter or output."""
        if name in self.x_names:
            return float(self.x[self.x_names.index(name)])
        if name in self.y_run_names:
            return float(self.y_run[self.y_run_names.index(name)])
        if name in self.u_run_names:
            return float(self.u_run_vals[self.u_run_names.index(name)])
        if name in self.p_names:
            return float(self.p_vals[self.p_names.index(name)])
        if name in self.z_names:
            if self._h_run is None:
                raise KeyError(name)
            return float(self._eval_h()[self.z_names.index(name)])
        if name in self.y_ini_names:
            return float(self.y_ini[self.y_ini_names.index(name)])
        raise KeyError(f"unknown variable '{name}'")

    def get_mvalue(self, names):
        return [self.get_value(n) for n in names]

    def get_values(self, name):
        """Time series of ``name`` (call :meth:`post` first)."""
        Time = np.asarray(self.Time)
        if name in self.x_names:
            return np.asarray(self.X)[:, self.x_names.index(name)]
        if name in self.y_run_names:
            return np.asarray(self.Y)[:, self.y_run_names.index(name)]
        if name in self.z_names:
            return np.asarray(self.Z)[:, self.z_names.index(name)]
        if name in self.u_run_names:
            return np.full(len(Time), self.get_value(name))
        if name in self.p_names:
            return np.full(len(Time), self.get_value(name))
        raise KeyError(f"unknown variable '{name}'")

    # ------------------------------------------------------------------
    # small-signal analysis
    # ------------------------------------------------------------------
    @_on_stream
    def _blocks(self):
        x, y = self._arr(self.x), self._arr(self.y_run)
        p, u = self._arr(self.p_vals), self._arr(self.u_run_vals)
        c = self._arr(self.c_run_vals)
        args = (x, y, p, u, c)
        return {k: np.array(fun(*args)).astype(np.float64)
                for k, fun in (("Fx", self._Fx), ("Fy", self._Fy),
                               ("Gx", self._Gx), ("Gy", self._Gy),
                               ("Fu", self._Fu), ("Gu", self._Gu))}

    @_on_stream
    def A_eval(self):
        """Reduced state matrix ``A = Fx - Fy Gy^-1 Gx``."""
        b = self._blocks()
        return b["Fx"] - b["Fy"] @ np.linalg.solve(b["Gy"], b["Gx"])

    @_on_stream
    def BCD_eval(self):
        """``(B, C, D)`` of the linearised system (``A`` from :meth:`A_eval`)."""
        b = self._blocks()
        Gy_inv_Gx = np.linalg.solve(b["Gy"], b["Gx"])
        Gy_inv_Gu = np.linalg.solve(b["Gy"], b["Gu"])
        B = b["Fu"] - b["Fy"] @ Gy_inv_Gu
        if self._Hx is None:
            return B, None, None
        x, y = self._arr(self.x), self._arr(self.y_run)
        p, u = self._arr(self.p_vals), self._arr(self.u_run_vals)
        c = self._arr(self.c_run_vals)
        Hx = np.array(self._Hx(x, y, p, u, c)).astype(np.float64)
        Hy = np.array(self._Hy(x, y, p, u, c)).astype(np.float64)
        Hu = np.array(self._Hu(x, y, p, u, c)).astype(np.float64)
        return B, Hx - Hy @ Gy_inv_Gx, Hu - Hy @ Gy_inv_Gu

    def eval_eigenvalues(self):
        return np.linalg.eigvals(self.A_eval())

    def save_xy_0(self, path="xy_0.json"):
        data = {**dict(zip(self.x_names, map(float, self.x))),
                **dict(zip(self.y_run_names, map(float, self.y_run)))}
        with open(path, "w", encoding="utf-8") as fobj:
            json.dump(data, fobj, indent=2)
        return path
