# -*- coding: utf-8 -*-
r"""Rotor mechanics, shared by every machine model.

Two modes, chosen by the machine's ``mech`` key:

* ``"imposed"`` (default) -- the speed $\omega_m$ is an input, as on a test
  bench or behind a stiff load.  No mechanical state.
* ``"free"`` -- the speed is a state:

  $$J\frac{d\omega_m}{dt} = T_e - T_l - B\,\omega_m$$

  and the initialisation uses pydae's ini/run swap (the same pattern as
  ``V \leftrightarrow v_{ref}`` in pydae-bps): during ``ini()`` the speed is
  given (``omega_ini``, an input) and the load torque $T_l$ is solved for;
  during ``run()`` $T_l$ is the input, holding the value ``ini()`` found, and
  ``omega_ini`` is an algebraic variable that just follows $\omega_m$.  A
  machine therefore starts in mechanical equilibrium at the speed asked for.
"""


def descriptions():
    """Parameters, inputs and states the mechanics add to a machine."""
    d = []
    d += [{"type": "Parameter", "tex": "J", "data": "J", "model": "J", "default": 0.01, "description": "Rotor inertia (mech = free)", "units": "kg.m^2"}]
    d += [{"type": "Parameter", "tex": "B", "data": "B", "model": "B", "default": 0.0, "description": "Viscous friction (mech = free)", "units": "N.m.s/rad"}]
    d += [{"type": "Input", "tex": r"\omega_m", "data": "omega_m", "model": "omega_m", "default": 0.0, "description": "Mechanical speed (mech = imposed; the ini speed when free)", "units": "rad/s"}]
    d += [{"type": "Input", "tex": "T_l", "data": "T_l", "model": "T_l", "default": 0.0, "description": "Load torque, run phase (mech = free; solved during ini)", "units": "N.m"}]
    d += [{"type": "Dynamic State", "tex": r"\omega_m", "data": "", "model": "omega_m", "default": "", "description": "Mechanical speed (mech = free)", "units": "rad/s"}]
    d += [{"type": "Algebraic State", "tex": r"\omega_{ini}", "data": "", "model": "omega_ini", "default": "", "description": "Follows omega_m during run (mech = free)", "units": "rad/s"}]
    return d


def add_mechanics(dae, name, backend, data, T_e):
    """Append the rotor to ``dae`` and return the speed symbol omega_m_{name}.

    ``T_e`` is the machine's electromagnetic torque expression [N.m].
    """
    mech = data.get("mech", "imposed")
    omega_m = backend.symbols(f"omega_m_{name}")
    val_omega = float(data.get("omega_m", 0.0))

    if mech == "imposed":
        dae["u_ini_dict"][f"{omega_m}"] = val_omega
        dae["u_run_dict"][f"{omega_m}"] = val_omega
    elif mech == "free":
        J = backend.symbols(f"J_{name}")
        B = backend.symbols(f"B_{name}")
        T_l = backend.symbols(f"T_l_{name}")
        omega_ini = backend.symbols(f"omega_ini_{name}")
        dae["f_list"] += [(T_e - T_l - B * omega_m) / J]
        dae["x_list"] += [omega_m]
        # ini: omega_m pinned to omega_ini (input), T_l solved.
        # run: T_l is the input (ini's value), omega_ini follows omega_m.
        dae["g_list"] += [omega_m - omega_ini]
        dae["y_ini_list"] += [T_l]
        dae["y_run_list"] += [omega_ini]
        dae["u_ini_dict"][f"{omega_ini}"] = val_omega
        dae["u_run_dict"][f"{T_l}"] = float(data.get("T_l", 0.0))
        dae["params_dict"].update({f"{J}": float(data.get("J", 0.01)),
                                   f"{B}": float(data.get("B", 0.0))})
        dae["xy_0_dict"].update({f"{omega_m}": val_omega, f"{omega_ini}": val_omega,
                                 f"{T_l}": float(data.get("T_l", 0.0))})
    else:
        raise ValueError(f"machine {name}: mech must be 'imposed' or 'free', not {mech!r}")

    dae["h_dict"][f"P_m_{name}"] = T_e * omega_m
    return omega_m
