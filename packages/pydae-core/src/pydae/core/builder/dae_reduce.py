# -*- coding: utf-8 -*-
r"""Structural reduction of a semi-explicit DAE: eliminate the algebraic
variables that are explicit, keep the rest implicit.

pydae models carry many algebraic variables whose equations are explicit
in disguise -- a current read from a table, ``i - I(lam) = 0``; a controller
output, ``z - clip(K_p e + K_i xi) = 0``; a flux relation,
``lam - L i - Psi = 0``.  Solving them every step with Newton is pure cost.
This module finds them and substitutes them away symbolically:

$$\dot x = f(x, z, p), \quad 0 = g(x, z, p) \quad\longrightarrow\quad
  \dot x = \tilde f(x, z_I, p), \quad 0 = \tilde g(x, z_I, p), \quad
  z_E = e(x, z_I, p)$$

with $z_E$ the eliminated variables and $z_I$ the implicit remainder (often
empty: the DAE was an ODE).

**The rule.**  An equation $g_i$ is solved for a variable $z_j$ when it is
*affine* in $z_j$, $g_i = c\,z_j + r$, with a coefficient $c$ that depends
on neither states nor algebraic variables -- a number, or a parameter such
as an inductance.  Then $z_j = -r/c$, exactly.  Affinity is read from the
expression tree (sums, negations, products and quotients by $z_j$-free
factors), not from the derivative: CasADi's derivative of an interpolant
call keeps zero-seeded forward nodes that make a constant coefficient look
state-dependent.  The solution is substituted into every remaining
equation and every earlier solution, and the search repeats until nothing
more qualifies, so chains and linear loops between algebraic variables
resolve too.  Anything nonlinear in its own unknowns (a saturation curve,
a power-flow equation) stays implicit: the reduction never changes the
DAE's solution, only how much of it needs Newton.

For each equation the variable at the same index is tried first (pydae's
builders pair ``g_list[i]`` with ``y_run_list[i]``), then the others.

CasADi ``SX`` only.  A parameter-valued coefficient is divided by at run
time; :attr:`ReducedDAE.coeff_fn` lets the runtime check it is not zero.
"""
from dataclasses import dataclass, field

import casadi as ca


@dataclass
class ReducedDAE:
    """The DAE after elimination.  Every expression is in (x, z_imp, p)."""
    x: ca.SX
    p: ca.SX
    z: ca.SX                    # all algebraic variables, original order
    f: ca.SX                    # reduced right-hand side
    z_exp_idx: list             # indices (into z) of the eliminated variables
    z_exp: ca.SX                # their expressions, same order
    z_imp_idx: list             # indices of the implicit remainder
    z_imp: ca.SX                # its symbols
    g_imp: ca.SX                # its residuals
    coeffs: list = field(default_factory=list)   # (z index, coefficient) that were divided by

    @property
    def is_ode(self):
        return len(self.z_imp_idx) == 0

    @property
    def z_full(self):
        """All algebraic variables, original order, as expressions in (x, z_imp, p)."""
        nz = self.z.numel()
        out = [None] * nz
        for k, j in enumerate(self.z_exp_idx):
            out[j] = self.z_exp[k]
        for k, j in enumerate(self.z_imp_idx):
            out[j] = self.z_imp[k]
        return ca.vertcat(*out) if nz else ca.SX(0, 1)

    @property
    def coeff_fn(self):
        """p -> the coefficients the eliminations divide by (check none is 0)."""
        c = ca.vertcat(*[c for _, c in self.coeffs]) if self.coeffs else ca.SX(0, 1)
        return ca.Function("reduce_coeffs", [self.p], [c])

    def report(self, names=None):
        """One line per variable: eliminated or implicit."""
        names = names or [self.z[j].name() for j in range(self.z.numel())]
        lines = [f"{len(self.z_exp_idx)} of {self.z.numel()} algebraic variables eliminated, "
                 f"{len(self.z_imp_idx)} implicit"]
        lines += [f"  implicit: {names[j]}" for j in self.z_imp_idx]
        return "\n".join(lines)


def _affine_split(e, zj):
    """Write the scalar ``e`` as ``c * zj + r`` by walking its expression tree
    through + - neg * / nodes, descending only into subtrees that contain zj.

    Returns (c, r), or None when zj sits under any other operation (or is
    multiplied by / divides something that also contains it).  Structural on
    purpose: CasADi's derivative of an external call (an interpolant, a
    Function) keeps forward-mode nodes with zero seeds, so a Jacobian test
    sees ``1 - fwd_lut(..., 0)`` and wrongly calls the coefficient
    state-dependent.  The tree never looks inside a subtree free of zj.
    """
    if not ca.depends_on(e, zj):
        return ca.SX(0), e
    if ca.is_equal(e, zj):
        return ca.SX(1), ca.SX(0)
    if e.is_symbolic() or e.is_constant():
        return None
    op = e.op()
    if op in (ca.OP_ADD, ca.OP_SUB):
        a, b = _affine_split(e.dep(0), zj), _affine_split(e.dep(1), zj)
        if a is None or b is None:
            return None
        return (a[0] + b[0], a[1] + b[1]) if op == ca.OP_ADD else (a[0] - b[0], a[1] - b[1])
    if op == ca.OP_NEG:
        a = _affine_split(e.dep(0), zj)
        return None if a is None else (-a[0], -a[1])
    if op == ca.OP_TWICE:
        a = _affine_split(e.dep(0), zj)
        return None if a is None else (2 * a[0], 2 * a[1])
    if op == ca.OP_MUL:
        d0, d1 = e.dep(0), e.dep(1)
        if not ca.depends_on(d0, zj):
            a = _affine_split(d1, zj)
            return None if a is None else (d0 * a[0], d0 * a[1])
        if not ca.depends_on(d1, zj):
            a = _affine_split(d0, zj)
            return None if a is None else (d1 * a[0], d1 * a[1])
        return None
    if op == ca.OP_DIV and not ca.depends_on(e.dep(1), zj):
        a = _affine_split(e.dep(0), zj)
        return None if a is None else (a[0] / e.dep(1), a[1] / e.dep(1))
    return None


def _solve_for(gi, zj, x, z):
    """(solution of gi = 0 for zj, coefficient) when gi is affine in zj with a
    coefficient free of x and z (a number or a parameter expression), else None."""
    split = _affine_split(gi, zj)
    if split is not None:
        c, r = split
    else:
        # fallback: the derivative (catches affinity the tree walk cannot see)
        c = ca.jacobian(gi, zj)
        if ca.depends_on(c, z):
            return None
        r = ca.substitute(gi, zj, ca.SX(0))
    if c.is_zero() or ca.depends_on(c, z) or (x.numel() and ca.depends_on(c, x)):
        return None
    return -r / c, c


def reduce_explicit_algebraics(x, z, p, f, g):
    """Eliminate every algebraic variable an equation gives explicitly.

    Parameters are the pieces of a semi-explicit DAE as ``SX`` column
    vectors: states ``x``, algebraic variables ``z`` (each a symbol), the
    parameter vector ``p`` (parameters and inputs), right-hand side ``f``,
    residuals ``g`` (as many as ``z``).  Returns a :class:`ReducedDAE`.
    """
    # CasadiBuilder hands over an empty DM when a block (no algebraics, no
    # parameters) is empty: make every piece SX
    x, z, p, f, g = (v if isinstance(v, ca.SX) else ca.SX(v) for v in (x, z, p, f, g))
    nz = z.numel()
    if g.numel() != nz:
        raise ValueError(f"{g.numel()} algebraic equations for {nz} algebraic variables")
    zs = [z[j] for j in range(nz)]
    names = {zs[j].name(): j for j in range(nz)}
    eqs = {i: g[i] for i in range(nz)}          # equation index -> residual (substituted as we go)
    remaining = set(range(nz))                  # unsolved variables
    solved = {}                                 # variable index -> expression
    coeffs = []

    progress = True
    while progress and eqs:
        progress = False
        for i in sorted(eqs):
            gi = eqs[i]
            present = {names[s.name()] for s in ca.symvar(gi) if s.name() in names} & remaining
            for j in sorted(present, key=lambda j: (j != i, j)):      # the paired variable first
                sol = _solve_for(gi, zs[j], x, z)
                if sol is None:
                    continue
                e, c = sol
                solved[j] = e
                if not (c.is_constant() and float(c) in (1.0, -1.0)):
                    coeffs.append((j, c))
                del eqs[i]
                remaining.discard(j)
                # put the solution into every remaining equation and every earlier
                # solution (equation and variable indices overlap: keep them apart)
                keys_e = list(eqs)
                keys_s = [k for k in solved if k != j]
                new = ca.substitute([eqs[k] for k in keys_e] + [solved[k] for k in keys_s], [zs[j]], [e])
                for k, v in zip(keys_e, new[:len(keys_e)]):
                    eqs[k] = v
                for k, v in zip(keys_s, new[len(keys_e):]):
                    solved[k] = v
                progress = True
                break

    exp_idx = sorted(solved)
    imp_idx = sorted(remaining)
    exp_syms = [zs[j] for j in exp_idx]
    exp_exprs = [solved[j] for j in exp_idx]
    f_red = ca.substitute([f], exp_syms, exp_exprs)[0] if exp_idx else f
    g_imp = ca.vertcat(*[eqs[i] for i in sorted(eqs)]) if eqs else ca.SX(0, 1)
    return ReducedDAE(x=x, p=p, z=z, f=f_red,
                      z_exp_idx=exp_idx, z_exp=ca.vertcat(*exp_exprs) if exp_idx else ca.SX(0, 1),
                      z_imp_idx=imp_idx, z_imp=ca.vertcat(*[zs[j] for j in imp_idx]) if imp_idx else ca.SX(0, 1),
                      g_imp=g_imp, coeffs=coeffs)
