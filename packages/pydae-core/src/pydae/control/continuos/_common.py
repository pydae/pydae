# -*- coding: utf-8 -*-
"""Shared helpers for the ``pydae.control.continuos`` block library.

Every continuous control block (``pi``, ``pid``, ``lpf1``, ``lpf2``,
``leadlag``, ``washout``, ``stab_1lpf_2wo_3ll``) takes the same
``(dae, name, backend, data_dict=None, input=None, output=None)``
signature and uses :func:`resolve_input` / :func:`combine_inputs` here
so the wiring semantics (h_dict expression, existing-symbol alias, or
new global input) are identical across the library.
"""


def resolve_input(input_name, dae, backend, default_val=0.0):
    """Resolve an ``input=`` name to an expression in the surrounding ``dae``.

    Resolution order:

    1. ``dae['h_dict'][input_name]`` -- use the derived expression
       directly. No new free symbol is introduced; the constraints
       that produced the h_dict entry stay in force.
    2. Name already declared as a state, algebraic variable, input,
       or parameter -- alias the cached backend symbol.
    3. Otherwise -- create the name as a new global input, added to
       both ``u_ini_dict`` and ``u_run_dict`` with ``default_val``.
    """
    if input_name in dae['h_dict']:
        return dae['h_dict'][input_name]

    known = set()
    known.update(str(s) for s in dae.get('x_list', []))
    known.update(str(s) for s in dae.get('y_run_list', []))
    known.update(dae.get('u_run_dict', {}).keys())
    known.update(dae.get('params_dict', {}).keys())

    sym = backend.symbols(input_name)
    if input_name in known:
        return sym

    dae['u_ini_dict'][input_name] = default_val
    dae['u_run_dict'][input_name] = default_val
    return sym


def combine_inputs(input, dae, backend, default_val=0.0):
    """Combine an ``input=`` argument (``str`` or ``list[str]``) into a
    single expression.

    For a string, returns the resolved expression directly.

    For a list, returns ``input[0] - sum(input[1:])`` -- the
    conventional PI error signal (measurement minus one or more
    reference signals). For non-error use, a one-element list is
    equivalent to passing the string.

    A ``None`` ``input`` means "no aliasing" and is the caller's
    responsibility (each block creates its own fresh ``u_<block>_{name}``
    input in that case).
    """
    if isinstance(input, str):
        return resolve_input(input, dae, backend, default_val)
    names = list(input)
    u_expr = resolve_input(names[0], dae, backend, default_val)
    for extra in names[1:]:
        u_expr = u_expr - resolve_input(extra, dae, backend, 0.0)
    return u_expr
