---
description: Scaffold a new pydae-bps power device (syn machine, VSC, wind turbine, load) that injects P/Q into a bus
argument-hint: "<family>/<name>  e.g. syns/my_machine | vscs/my_vsc | wecs/my_wind | loads/my_load"
---

Scaffold a new **power-injecting device** at `packages/pydae-bps/src/pydae/bps/$ARGUMENTS.py` plus a sibling `.hjson` fixture. Power devices return `(p_W, q_var)` in absolute units — the builder injects those into the bus power balance. Controllers do not return power; use `/new-controller` for those.

**Canonical template: `packages/pydae-bps/src/pydae/bps/syns/milano4ord.py`.** Read it before writing. The scaffold must mirror its structure — but note the docstring convention is upgraded (see below).

Before writing:
1. Parse `$ARGUMENTS` as `<family>/<name>`. Family must be one of `syns`, `vscs`, `wecs`, `loads`. If not, stop and ask.
2. Refuse if the target file exists.
3. Read a neighbour in the same family. For `syns/`, `milano4ord.py` is the reference. For `vscs/`, pick an existing vsc module; same for `wecs/` and `loads/`. Confirm with the user which variables are dynamic states vs. algebraic states before writing equations.

The generated `.py` must include:
- **Raw (`r"""..."""`) module docstring** — this is the new norm (`milano4ord.py` is non-raw for legacy reasons; do not copy that aspect). Cover: auxiliary equations, dynamic equations, algebraic equations, and the `(p_W, q_var)` injection convention. LaTeX dollar-math is fine.
- `def descriptions():` returning the full list of Parameter / Input / Dynamic State / Algebraic State / Output dicts — consumed by docs autosummary and by the builder to resolve defaults.
- Builder function `def <name>(grid, name, bus_name, data_dict):` — signature matches `milano4ord.py` exactly. Key differences from controllers:
  - First arg is `grid` (has `.dae`, `.H_total`, `.omega_coi_numerator`, `.omega_coi_denominator`), not `dae`.
  - Reads bus state via `V_{bus_name}`, `theta_{bus_name}`, and (for inertial devices) `omega_coi`.
  - Appends to `grid.dae['f' | 'g' | 'x' | 'y_ini' | 'y_run' | 'params_dict' | 'u_ini_dict' | 'u_run_dict' | 'h_dict' | 'xy_0_dict']`.
  - For rotating machines, **must** contribute to COI: `grid.H_total += H`, `grid.omega_coi_numerator += omega*H*S_n`, `grid.omega_coi_denominator += H*S_n`. Skip for non-inertial devices (most VSCs, loads).
  - Parameters per-instance via `_{name}` suffix. Use the `default_map` pattern: `default_map = {item['data']: item['default'] for item in descriptions() if item.get('data')}`, then `val = data_dict.get(key, default_map.get(key, fallback))`.
  - Algebraic variables **must** include `p_g` and `q_g` (per-unit on the device base `S_n`) with residuals `0 = i_d*v_d + i_q*v_q - p_g` and `0 = i_d*v_q - i_q*v_d - q_g` (or equivalent for non-synchronous devices).
  - Ends with:
    ```python
    p_W   = p_g * S_n
    q_var = q_g * S_n
    return p_W, q_var
    ```
  - Frequency handling: compute `Omega_b = 2*pi*F_n` from a data-driven `F_n` and register via `grid.dae['params_dict']`.

Keep the test block modelled on `milano4ord.py`:
- `def test_build():` uses `BpsBuilder('<name>.hjson')`, then `Builder(grid.sys_dict, target='ctypes').build()`.
- `def test_run():` loads `Model('temp_<name>')`, does `model.ini(...)` + chained `model.run(...)`, then `model.post()` and plots one or two observables to `<name>.svg` for quick visual inspection.
- Both are invoked under `if __name__ == '__main__':`.

The `.hjson` fixture: copy from a sibling, update the device-config block. Include a `vsource` on a second bus so the initialisation has a swing reference.

After generating:
1. Register the new `type` in the family dispatcher (`syns/syns.py`, `vscs/vscs.py`, `wecs/wecs.py`, `loads/loads.py`). The dispatcher is what routes `{"type": "<name>"}` entries from user HJSON into the new builder.
2. Add a Sphinx stub at `docs/pydae-bps/models/<family>/<name>.rst`.
3. Suggest running the module as a script (`uv run python packages/pydae-bps/src/pydae/bps/<family>/<name>.py`) to exercise `test_build` + `test_run`.

Do not commit. Report the files created.
