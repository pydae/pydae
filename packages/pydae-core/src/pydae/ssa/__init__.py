"""
pydae.ssa — Small-signal (eigenvalue) analysis utilities.

Re-exports the public API of ``pydae.ssa.ssa`` so users can write::

    import pydae.ssa as ssa
    ssa.A_eval(model)
    damp = ssa.damp_report(model)
"""

from pydae.ssa.ssa import (
    A_eval,
    acker,
    add_arrow,
    damp,
    damp_report,
    discretise_time,
    dlqr,
    eig,
    eval_A,
    eval_A_ini,
    eval_ss,
    lead_design,
    left2df,
    lqr,
    participation,
    pi_design,
    plot_eig,
    plot_shapes,
    plot_vectors,
    right2df,
    shape2df,
    ss_eval,
    get_mode,
)
from pydae.ssa.residue import residue

__all__ = [
    "A_eval",
    "acker",
    "add_arrow",
    "damp",
    "damp_report",
    "discretise_time",
    "dlqr",
    "eig",
    "eval_A",
    "eval_A_ini",
    "eval_ss",
    "lead_design",
    "left2df",
    "lqr",
    "participation",
    "pi_design",
    "plot_eig",
    "plot_shapes",
    "plot_vectors",
    "residue",
    "right2df",
    "shape2df",
    "ss_eval",
    "get_mode"
]
