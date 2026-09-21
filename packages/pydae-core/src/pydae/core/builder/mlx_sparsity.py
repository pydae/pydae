r"""
Structural sparsity for the MLX backend
=======================================

Every Jacobian the MLX backend needs -- the initialization Jacobian, the
trapezoidal Jacobian of each time step, and the state-space blocks
:math:`F_x, F_y, G_x, G_y, \dots` -- used to be built the obvious way: one
reverse-mode ``vjp`` pass per equation, ``mx.vmap``-ed over the rows of the
identity.  That costs :math:`N` passes for an :math:`N`-equation system whether
the Jacobian has :math:`N^2` nonzeros or :math:`5N`, and a power-system
Jacobian is emphatically the second kind (4.6 % dense on the Kundur two-area
system, 0.8 % on a 24-machine one).

This module removes that waste with the standard Curtis-Powell-Reid (CPR)
compression, plus one refinement that a power system makes mandatory.

Column compression
------------------

Two columns are *structurally orthogonal* when no equation contains both of
their variables.  Their derivatives can then be recovered from a single
directional derivative that perturbs both at once: nothing in the result mixes
them, because no row is fed by both.  So partition the columns into groups of
mutually orthogonal columns -- a graph colouring on "share an equation" -- and
take one forward-mode ``jvp`` pass per group with a tangent that is 1.0 in
every column of the group.  The number of passes drops from ``n_cols`` to the
number of colours, which is bounded below by the densest row and in practice
sits very close to it.

Measured on the models in this repository (columns -> passes):

===================  ========  ==========  =========================
model                 n         dense AD    compressed
===================  ========  ==========  =========================
``genrou``            16        16          7
``k12p6``             87        87          9
6x ``k12p6``          507       507         11  (10 colours + 1 row)
===================  ========  ==========  =========================

Dense rows, and why plain colouring is not enough
-------------------------------------------------

A single dense row destroys the colouring: if one equation contains every
variable, then every pair of columns shares a row, every pair conflicts, and
the colouring degenerates to ``n_cols`` colours -- back to where we started.
Power systems have exactly such a row.  The centre-of-inertia speed

.. math::   \omega_{coi} = \sum_i H_i \omega_i \big/ \sum_i H_i

touches every machine, so it grows denser as the network grows -- precisely
when compression matters most.  On the 24-machine system above, that one
equation takes the colouring from 11 passes to 29.

The fix is a cut-down Coleman-Verma bicolouring: pull the few dense rows out of
the conflict graph, colour what is left, and recover those rows the other way
round -- one reverse-mode ``vjp`` pass each, which yields a whole row at a time
regardless of how dense it is.  Forward mode compresses columns, reverse mode
compresses rows, and the total is ``n_colours + n_dense_rows``.  Which rows to
treat as dense is chosen by trying the densest few and keeping whichever split
needs the fewest passes overall.

What is *not* compressed, and why
---------------------------------

Only the Jacobian *evaluation*.  The linear solve stays dense, on purpose and
against expectation: MLX arrays are immutable, so the in-place update at the
heart of any sparse factorization (``values.at[dst].add(...)``) allocates and
copies the entire value array on every pivot.  A batched sparse LU of the
``k12p6`` Jacobian needs 170 such stages; measured over a ``(2246, 10000)``
value array that is 331 ms, against 603 ms for the dense
``mx.linalg.solve`` it would have replaced -- a 1.8x gain in exchange for a
static pivot order that can fail on a badly scaled operating point.  Not worth
it.  Sparsity pays in the AD passes, which is where the time actually goes: at
B=10,000 on ``k12p6`` the Jacobian was 2049 ms of a 2670 ms Newton iteration.

The pattern itself is a *structural over-approximation*: an entry is assumed
nonzero when the variable's symbol occurs anywhere in the equation, without
asking whether it cancels.  That can only ever add entries, never miss one, so
a compressed Jacobian is exact wherever the dense one is -- an over-counted
entry just gets computed and comes out zero.  :func:`verify_pattern` checks
the recovery machinery against a plain ``jvp`` on every build.
"""

from __future__ import annotations

import numpy as np

__all__ = ["JacobianPattern", "symbol_sets", "compressed_jacobian",
           "verify_pattern"]


# ---------------------------------------------------------------------------
# structural pattern extraction
# ---------------------------------------------------------------------------
def symbol_sets(exprs, frontend):
    """Free-symbol names of each expression, as a list of sets.

    Computed once per equation list and reused for every variable grouping
    (``Fx`` and ``Fy`` differ only in which columns they keep), because this is
    the expensive half of pattern extraction on a large model.
    """
    if not exprs:
        return []
    if frontend == "casadi":
        import casadi as ca
        return [{s.name() for s in ca.symvar(expr)} for expr in exprs]
    import sympy as sp
    return [{s.name for s in sp.sympify(expr).free_symbols} for expr in exprs]


def _pattern_indices(equation_symbols, var_names, offset=0):
    """Structural nonzeros of d(equations)/d(var_names) as (rows, cols)."""
    column_of = {name: j + offset for j, name in enumerate(var_names)}
    rows, cols = [], []
    for i, names in enumerate(equation_symbols):
        for name in names:
            j = column_of.get(name)
            if j is not None:
                rows.append(i)
                cols.append(j)
    return rows, cols


# ---------------------------------------------------------------------------
# colouring
# ---------------------------------------------------------------------------
def _greedy_colour(conflict, degree):
    """Largest-first greedy colouring of a scipy CSR conflict graph."""
    n_cols = conflict.shape[0]
    colour = np.full(n_cols, -1, dtype=np.int64)
    for column in np.argsort(-degree, kind="stable"):
        neighbours = conflict.indices[conflict.indptr[column]:
                                      conflict.indptr[column + 1]]
        taken = {colour[k] for k in neighbours if colour[k] >= 0}
        candidate = 0
        while candidate in taken:
            candidate += 1
        colour[column] = candidate
    return colour


def _conflict_graph(rows, cols, n_rows, n_cols, excluded):
    """Columns conflict when some *kept* row contains both of them."""
    from scipy.sparse import csr_matrix
    if len(rows) == 0:
        return csr_matrix((n_cols, n_cols), dtype=np.int8), np.zeros(n_cols)
    keep = np.ones(len(rows), dtype=bool)
    if len(excluded):
        keep = ~np.isin(rows, excluded)
    occupancy = csr_matrix(
        (np.ones(int(keep.sum()), dtype=np.int8), (rows[keep], cols[keep])),
        shape=(n_rows, n_cols))
    conflict = (occupancy.T @ occupancy).tocsr()
    conflict.setdiag(0)
    conflict.eliminate_zeros()
    degree = np.diff(conflict.indptr).astype(np.float64)
    return conflict, degree


class JacobianPattern:
    """Structural sparsity of one Jacobian block, with its CPR colouring.

    Parameters
    ----------
    rows, cols : sequence of int
        Structural nonzeros of ``d(equations)/d(variables)``.
    n_rows, n_cols : int
        Shape of the Jacobian.
    max_dense_rows : int
        Upper bound on how many rows may be handed to reverse mode.  Each one
        costs a pass, so there is no point exploring far.

    Attributes
    ----------
    column_colour : ndarray of int, shape (n_cols,)
        Colour of each column; columns sharing a colour are perturbed together.
    n_colours : int
        Number of forward passes.
    dense_rows : ndarray of int
        Rows recovered by reverse mode, one pass each.
    n_passes : int
        ``n_colours + len(dense_rows)`` -- what the compression costs, against
        ``n_rows`` for the dense reverse-mode Jacobian.
    """

    def __init__(self, rows, cols, n_rows, n_cols, max_dense_rows=24):
        self.n_rows = int(n_rows)
        self.n_cols = int(n_cols)
        # Deduplicate: recovery scatters with an accumulating `.at[].add`, so a
        # repeated (row, col) would double that entry. The trapezoidal pattern
        # hits this whenever a state appears in its own derivative -- the
        # explicit x_n of the step and the equation's own reference to it land
        # on the same diagonal slot.
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        cols = np.asarray(cols, dtype=np.int64).reshape(-1)
        if rows.size and self.n_cols:
            unique = np.unique(rows * self.n_cols + cols)
            rows, cols = np.divmod(unique, self.n_cols)
        self.rows, self.cols = rows, cols
        self.nnz = int(self.rows.size)

        row_nnz = np.bincount(self.rows, minlength=self.n_rows) if self.nnz \
            else np.zeros(self.n_rows, dtype=np.int64)
        self.max_row_nnz = int(row_nnz.max()) if self.n_rows else 0

        self.column_colour, self.dense_rows = self._choose_split(
            row_nnz, max_dense_rows)
        self.n_colours = (int(self.column_colour.max()) + 1
                          if self.n_cols and self.column_colour.size else 0)
        self.n_passes = self.n_colours + len(self.dense_rows)

        self._build_recovery()

    # -- construction ---------------------------------------------------
    def _choose_split(self, row_nnz, max_dense_rows):
        """Pick how many dense rows to hand to reverse mode.

        Excluding the densest row can only lower the colour count and always
        costs exactly one extra pass, so walk the rows in decreasing density
        and keep the split with the fewest total passes.  A short patience
        window stops the search once exclusion stops paying.
        """
        if self.n_cols == 0 or self.nnz == 0:
            return np.zeros(self.n_cols, dtype=np.int64), np.zeros(0, dtype=np.int64)

        by_density = np.argsort(-row_nnz, kind="stable")
        cap = int(min(max_dense_rows, self.n_rows))
        best_colour, best_dense, best_passes, stale = None, None, None, 0

        for n_excluded in range(cap + 1):
            excluded = by_density[:n_excluded]
            # Nothing dense left to gain from.
            if n_excluded and row_nnz[by_density[n_excluded - 1]] <= 1:
                break
            conflict, degree = self._conflict(excluded)
            colour = _greedy_colour(conflict, degree)
            passes = int(colour.max()) + 1 + n_excluded
            if best_passes is None or passes < best_passes:
                best_colour, best_dense, best_passes, stale = (
                    colour, excluded.copy(), passes, 0)
            else:
                stale += 1
                if stale >= 4:
                    break
        return best_colour, np.sort(best_dense)

    def _conflict(self, excluded):
        return _conflict_graph(self.rows, self.cols, self.n_rows, self.n_cols,
                               excluded)

    def _build_recovery(self):
        """Arrays that expand the compressed passes back to a dense Jacobian.

        Recovery is deliberately built from *structured* operations only -- a
        ``take`` along one axis, a transpose and a multiply -- rather than an
        ``n_rows * n_cols`` list of scalar slot indices.  Both compute the same
        thing, but per-element gather/scatter over a flat index list measured
        15 % *slower* than plain dense reverse mode on a 121-equation model,
        which threw away the entire saving; the structured form keeps it.

        Entries in dense rows are excluded from the forward-mode mask: reverse
        mode writes those rows whole, and letting both contribute would double
        every entry where they overlap.
        """
        self.forward_mask = np.zeros((self.n_rows, self.n_cols), dtype=bool)
        if self.nnz:
            self.forward_mask[self.rows, self.cols] = True
        if len(self.dense_rows):
            self.forward_mask[self.dense_rows, :] = False

    # -- seed matrices --------------------------------------------------
    def seed_matrix(self, np_dtype):
        """``(n_colours, n_cols)`` tangents -- 1.0 in every column of a colour."""
        seeds = np.zeros((max(self.n_colours, 0), self.n_cols), dtype=np_dtype)
        if self.n_colours:
            seeds[self.column_colour, np.arange(self.n_cols)] = 1.0
        return seeds

    def dense_basis(self, np_dtype):
        """``(n_dense_rows, n_rows)`` cotangents -- one identity row each."""
        basis = np.zeros((len(self.dense_rows), self.n_rows), dtype=np_dtype)
        if len(self.dense_rows):
            basis[np.arange(len(self.dense_rows)), self.dense_rows] = 1.0
        return basis

    # -- reporting ------------------------------------------------------
    @property
    def worthwhile(self):
        """True when compression needs strictly fewer passes than dense AD."""
        return 0 < self.n_passes < self.n_rows

    @property
    def density(self):
        cells = self.n_rows * self.n_cols
        return self.nnz / cells if cells else 0.0

    def __repr__(self):
        return (f"<JacobianPattern {self.n_rows}x{self.n_cols} "
                f"nnz={self.nnz} ({100 * self.density:.1f}%) "
                f"passes={self.n_passes} "
                f"({self.n_colours} colours + {len(self.dense_rows)} rows) "
                f"vs {self.n_rows} dense>")


# ---------------------------------------------------------------------------
# the compressed evaluator
# ---------------------------------------------------------------------------
def compressed_jacobian(fun, pattern, dtype, stream):
    """Dense Jacobian of ``fun`` built from ``pattern.n_passes`` AD passes.

    A drop-in replacement for
    :func:`~pydae.core.builder.mlx_builder._jacobian`: same call signature,
    same two output layouts, same values.  It differentiates with respect to
    the first argument and returns

    * ``(n_rows, n_cols)``        for a ``(n_cols,)`` primal -- one system;
    * ``(n_rows, n_cols, B)``     for a ``(n_cols, B)`` primal -- the sweep
      layout, B independent systems at once.

    The result is dense because that is what ``mx.linalg.solve`` consumes; the
    saving is in how it was *computed*, not in how it is stored.
    """
    from pydae.core.builder.mlx_builder import _mx

    mx = _mx()
    n_rows, n_cols = pattern.n_rows, pattern.n_cols
    n_colours = pattern.n_colours
    n_dense = len(pattern.dense_rows)
    np_dtype = np.float64 if dtype == getattr(mx, "float64", None) else np.float32

    with mx.stream(stream):
        seed_matrix = mx.array(pattern.seed_matrix(np_dtype), dtype=dtype)
        column_colour = mx.array(pattern.column_colour.astype(np.int32))
        forward_mask = mx.array(pattern.forward_mask.astype(np_dtype), dtype=dtype)
        dense_basis = mx.array(pattern.dense_basis(np_dtype), dtype=dtype)
        dense_rows = mx.array(pattern.dense_rows.astype(np.int32))
        mx.eval(seed_matrix, column_colour, forward_mask, dense_basis, dense_rows)

    def jacobian(primal, *rest):
        batched = primal.ndim > 1
        width = primal.shape[1] if batched else None

        def evaluate(vector):
            return fun(vector, *rest)

        if n_colours:
            seeds = (mx.broadcast_to(seed_matrix[:, :, None],
                                     (n_colours, n_cols, width))
                     if batched else seed_matrix)

            def forward_pass(tangent):
                return mx.jvp(evaluate, (primal,), (tangent,))[1][0]

            # (n_colours, n_rows[, B]): pass k holds the derivative of every
            # row with respect to all of colour k's columns at once.  Entry
            # (i, j) is therefore column j's colour probed at row i -- unique
            # because no kept row carries two columns of the same colour --
            # so one take along the colour axis expands the whole Jacobian.
            compressed = mx.vmap(forward_pass)(seeds)
            out = mx.take(compressed, column_colour, axis=0)   # (n_cols, n_rows, ..)
            out = mx.transpose(out, (1, 0, 2) if batched else (1, 0))
            # Zero the structural holes, and the dense rows with them.
            out = out * (forward_mask[:, :, None] if batched else forward_mask)
        else:
            out = mx.zeros((n_rows, n_cols, width) if batched
                           else (n_rows, n_cols), dtype=primal.dtype)

        if n_dense:
            basis = (mx.broadcast_to(dense_basis[:, :, None],
                                     (n_dense, n_rows, width))
                     if batched else dense_basis)

            def reverse_pass(cotangent):
                return mx.vjp(evaluate, (primal,), (cotangent,))[1][0]

            # One whole row per pass, however dense it is; the mask above left
            # these rows at zero, so a scatter along axis 0 drops them in.
            rows = mx.vmap(reverse_pass)(basis)      # (n_dense, n_cols[, B])
            out = out.at[dense_rows].add(rows)

        return out

    return jacobian


def verify_pattern(fun, jacobian, primal, rest, tolerance):
    """Check a compressed Jacobian against a plain directional derivative.

    One extra ``jvp`` pass, run once per block at build time.  ``J @ tangent``
    against ``jvp(fun, tangent)`` for a random tangent exercises the whole
    recovery path -- colouring, scatter slots, dense-row split -- and catches
    an indexing mistake immediately rather than as a Newton that mysteriously
    fails to converge.

    Returns the relative discrepancy; the caller decides what to do with it.
    """
    from pydae.core.builder.mlx_builder import _mx

    mx = _mx()
    generator = np.random.default_rng(0)
    np_dtype = np.float64 if primal.dtype == getattr(mx, "float64", None) \
        else np.float32
    tangent = mx.array(generator.standard_normal(primal.shape[0]).astype(np_dtype),
                       dtype=primal.dtype)

    direct = mx.jvp(lambda vector: fun(vector, *rest), (primal,), (tangent,))[1][0]
    through = mx.matmul(jacobian(primal, *rest), tangent)
    mx.eval(direct, through)

    direct_np = np.array(direct, dtype=np.float64)
    through_np = np.array(through, dtype=np.float64)
    scale = max(float(np.max(np.abs(direct_np))), 1.0)
    discrepancy = float(np.max(np.abs(direct_np - through_np))) / scale
    return discrepancy, discrepancy <= tolerance
