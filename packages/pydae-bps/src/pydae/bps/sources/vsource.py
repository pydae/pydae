# -*- coding: utf-8 -*-
r"""
Ideal voltage source — infinite-bus equivalent for pydae-bps.

A ``vsource`` connected to a bus pins that bus's voltage magnitude and
angle to their reference values, acting as a **stiff Thevenin source**
with zero internal impedance.  It is used to model:

- An **infinite bus** (slack node) in single-machine or multi-machine
  studies where one terminal is an ideal grid equivalent.
- The **New York system equivalent** in the IEEE 39-bus benchmark, where
  the New England area is connected to an external, very stiff grid.

**Auxiliar equations**

None.

**Dynamic equations**

$$\dot{V}_{dummy} = v_{ref} - V_{dummy}$$

A dummy dynamic state is added so that the state vector has a consistent
dimension.  This state has no physical meaning; it simply settles to
:math:`v_{ref}` at steady state.

**Algebraic equations**

$$0 = V_k - v_{ref}$$
$$0 = \theta_k - \theta_{ref}$$

These replace the standard network bus equations for bus :math:`k`,
treating :math:`V_k` and :math:`\theta_k` as fixed inputs rather than
unknown algebraic states.

**COI contribution**

The vsource contributes :math:`H = 10^6` s to the centre-of-inertia
(COI) computation, pinning :math:`\omega_{COI} \approx 1` pu and
establishing the absolute angle reference.

**Inputs (runtime settable)**

| Symbol | Default | Description | Units |
|---|---|---|---|
| ``v_ref_{name}`` | 1.0 pu | Bus voltage magnitude setpoint | pu |
| ``theta_ref_{name}`` | 0.0 rad | Bus voltage angle setpoint | rad |
"""
import io

import sympy as sym  # noqa: F401 - reserved for symbolic_dev()


def descriptions():
    """
    Single source of truth for model parameters, inputs, states, and outputs.
    """
    descriptions_list = []

    # Inputs
    descriptions_list += [
        {"type": "Input", "tex": "v_{ref}", "data": "v_ref",
         "model": "v_ref", "default": 1.0,
         "description": "Bus voltage magnitude setpoint", "units": "pu"}]
    descriptions_list += [
        {"type": "Input", "tex": "\\theta_{ref}", "data": "theta_ref",
         "model": "theta_ref", "default": 0.0,
         "description": "Bus voltage angle setpoint", "units": "rad"}]

    # Dynamic States
    descriptions_list += [
        {"type": "Dynamic State", "tex": "V_{dummy}", "data": "",
         "model": "V_dummy", "default": "",
         "description": "Dummy state for consistent vector sizing",
         "units": "pu"}]

    # Outputs
    descriptions_list += [
        {"type": "Output", "tex": "V_{dummy}", "data": "",
         "model": "V_dummy_out", "default": "",
         "description": "Dummy state output", "units": "pu"}]

    return descriptions_list


def vsource(grid, name, bus_name, data_dict):
    """
    Attach an ideal voltage source to *bus_name* inside *grid*.

    Replaces the bus V and theta equations with algebraic pin constraints and
    adds a dummy state ``V_dummy_{name}`` for consistent vector sizing.

    Parameters
    ----------
    grid : BpsBuilder
        Builder instance that accumulates DAE components.
    name : str
        Unique instance identifier (e.g. ``"1"``).
    bus_name : str
        Name of the bus the source is connected to.
    data_dict : dict
        User-supplied parameter values (not used; all parameters are ideal).

    Returns
    -------
    None
        The vsource modifies grid.dae in-place and does not inject power
        (it pins the bus voltage directly).
    """
    backend = grid.backend

    # 1. Fetch metadata and defaults
    meta = descriptions()
    default_map = {
        item["data"]: item["default"]
        for item in meta if "data" in item and item["data"]
    }

    # 2. Bus-side variables
    V = backend.symbols(f"V_{bus_name}")
    theta = backend.symbols(f"theta_{bus_name}")

    # 3. Inputs
    v_ref = backend.symbols(f"v_ref_{name}")
    theta_ref = backend.symbols(f"theta_ref_{name}")

    # 4. Dummy dynamic state
    V_dummy = backend.symbols(f"V_dummy_{name}")

    # 5. Dynamic equation
    dV_dummy = v_ref - V_dummy

    # 6. Algebraic equations (pin constraints)
    g_V = V - v_ref
    g_theta = theta - theta_ref

    # 7. Assembly
    grid.dae["f"] += [dV_dummy]
    grid.dae["x"] += [V_dummy]

    # Replace the bus V and theta algebraic equations with pin constraints
    idx_V = next(
        i for i, y in enumerate(grid.dae["y_ini"]) if str(y) == str(V)
    )
    idx_theta = next(
        i for i, y in enumerate(grid.dae["y_ini"]) if str(y) == str(theta)
    )

    grid.dae["g"][idx_V] = g_V
    grid.dae["g"][idx_theta] = g_theta

    # 8. COI contribution (dominant inertia)
    H = 1e6
    grid.H_total += H
    grid.omega_coi_numerator += H
    grid.omega_coi_denominator += H

    # 9. Dynamic input handling
    v_ref_val = data_dict.get("v_ref", default_map.get("v_ref", 1.0))
    theta_ref_val = data_dict.get(
        "theta_ref", default_map.get("theta_ref", 0.0)
    )

    grid.dae["u_ini_dict"].update({str(v_ref): v_ref_val})
    grid.dae["u_run_dict"].update({str(v_ref): v_ref_val})

    grid.dae["u_ini_dict"].update({str(theta_ref): theta_ref_val})
    grid.dae["u_run_dict"].update({str(theta_ref): theta_ref_val})

    # 10. Initialization hints
    grid.dae["xy_0_dict"].update({str(V): v_ref_val})
    grid.dae["xy_0_dict"].update({str(theta): theta_ref_val})
    grid.dae["xy_0_dict"].update({str(V_dummy): v_ref_val})

    # 11. Outputs
    grid.dae["h_dict"].update({f"V_dummy_{name}": V_dummy})


# =============================================================================
# Sphinx Documentation Auto-Generator
# =============================================================================
def dict_list_to_aligned_markdown_table(data: list[dict]) -> str:
    if not data or not isinstance(data, list):
        return ""
    dict_data = [item for item in data if isinstance(item, dict)]
    if not dict_data:
        return ""

    all_headers = []
    header_set = set()
    for row_dict in dict_data:
        for key in row_dict.keys():
            if key not in header_set:
                header_set.add(key)
                all_headers.append(key)

    max_widths = {header: len(header) for header in all_headers}
    for row_dict in dict_data:
        for header in all_headers:
            max_widths[header] = max(
                max_widths[header], len(str(row_dict.get(header, "")))
            )

    output = io.StringIO()
    padded_headers = [
        header.ljust(max_widths[header]) for header in all_headers
    ]
    output.write("| " + " | ".join(padded_headers) + " |\n")
    separators = ["-" * max_widths[header] for header in all_headers]
    output.write("| " + " | ".join(separators) + " |\n")

    for row_dict in dict_data:
        padded_values = [
            str(row_dict.get(header, "")).ljust(max_widths[header])
            for header in all_headers
        ]
        output.write("| " + " | ".join(padded_values) + " |\n")

    return output.getvalue().strip()


def generate_sphinx_tables():
    docs = ""
    categories = [
        "Parameter", "Input", "Dynamic State", "Algebraic State", "Output"
    ]
    full_list = descriptions()

    for cat in categories:
        cat_list = [
            {k: v for k, v in item.items() if k != "type"}
            for item in full_list if item.get("type") == cat
        ]

        if cat_list:
            docs += f"\n### {cat}s\n\n"
            docs += dict_list_to_aligned_markdown_table(cat_list)
            docs += "\n"

    return docs


__doc__ += generate_sphinx_tables()


def symbolic_dev():
    """
    Symbolic steady-state analysis for vsource model.
    Trivial: V = v_ref, theta = theta_ref at equilibrium.
    """
    v_ref = sym.Symbol("v_ref", real=True, positive=True)
    theta_ref = sym.Symbol("theta_ref", real=True)
    V = sym.Symbol("V", real=True, positive=True)
    theta = sym.Symbol("theta", real=True)
    V_dummy = sym.Symbol("V_dummy", real=True, positive=True)

    g_V = V - v_ref
    g_theta = theta - theta_ref
    f_dummy = v_ref - V_dummy

    sol = sym.solve([g_V, g_theta, f_dummy], [V, theta, V_dummy])
    print(f"Steady state: {sol}")


# =============================================================================
# Testing Block
# =============================================================================
def test_build():
    from pydae.bps import BpsBuilder
    from pydae.core import Builder

    grid = BpsBuilder("vsource.hjson")
    grid.checker()
    grid.uz_jacs = False
    grid.construct("temp_vsource")
    bld = Builder(grid.sys_dict, target="ctypes", sparse=False)
    bld.build()


def test_ini():
    from pydae.core import Model

    model = Model("temp_vsource")

    model.ini({"v_ref_1": 1.0, "theta_ref_1": 0.0}, "xy_0.json")
    model.report_x()
    model.report_y()


def test_run():
    import matplotlib.pyplot as plt
    from pydae.core import Model

    model = Model("temp_vsource")

    model.ini({"v_ref_1": 1.0, "theta_ref_1": 0.0}, "xy_0.json")
    model.report_x()
    model.report_y()

    model.run(1.0, {})

    # Step voltage at t=1s
    model.run(5.0, {"v_ref_1": 1.05})

    model.post()

    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    axes[0].plot(model.Time, model.get_values("V_1"),
                 label="$V$ (pu)", color="b")
    axes[0].set_ylabel("Voltage")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(model.Time, model.get_values("V_dummy_1"),
                 label="$V_{dummy}$ (pu)", color="g")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Dummy State")
    axes[1].legend()
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig("vsource_response.svg")
    print("Test completed. Plot saved as 'vsource_response.svg'.")


if __name__ == "__main__":
    # symbolic_dev()
    test_build()
    test_ini()
    # test_run()
