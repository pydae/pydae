---
description: Scaffold a new pydae-bps controller / non-power device (AVR, PSS, governor, POD)
argument-hint: "<family>/<name>  e.g. avrs/my_avr | psss/my_pss | govs/my_gov | pods/my_pod"
---

Scaffold a new **controller** module at `packages/pydae-bps/src/pydae/bps/$ARGUMENTS.py` plus a sibling `.hjson` fixture. Controllers do not inject grid power — they read existing `x`, `y`, `u`, `p` (typically a bus voltage, machine speed, or electrical power) and produce a signal (`v_f`, `p_m`, `v_pss`, …) that another model consumes.

**Canonical template: `packages/pydae-bps/src/pydae/bps/avrs/sexs.py`.** Read it before writing. The scaffold must mirror its structure.

Before writing:
1. Parse `$ARGUMENTS` as `<family>/<name>`. Family must be one of `avrs`, `psss`, `govs`, `pods`. If not, stop and ask the user to pick a valid family.
2. Refuse if `packages/pydae-bps/src/pydae/bps/$ARGUMENTS.py` already exists — offer to edit it instead.
3. Read a neighbour in the same family for family-specific idioms (signal names, reference variables, output symbol). Ask the user which reference input the controller uses (e.g. `V_bus` for AVRs, `omega` for governors, `p_g` / `omega` for PSSs) before committing to the equation skeleton.

The generated `.py` must include:
- A raw (`r"""..."""`) module docstring with: signal path (LaTeX), any ini/run variable swap semantics, configuration example, and value-transfer notes. LaTeX dollar-math is fine — **never** use non-raw strings (they break on `\xi`, `\nu`, etc.).
- `def descriptions():` returning the list of parameter / input / dynamic-state / algebraic-state / output dicts — single source of truth for the docs autosummary.
- Builder function `def <name>(dae, data, name, bus_name):` — signature exactly matches `sexs.py`. **Returns nothing.** The function appends to: `dae['f']`, `dae['g']`, `dae['x']`, `dae['y_ini']`, `dae['y_run']`, `dae['params_dict']`, `dae['u_ini_dict']`, `dae['u_run_dict']`, `dae['h_dict']`, `dae['xy_0_dict']`.
- All parameters and state symbols suffixed with `_{name}` (e.g. `K_a_{name}`). Bus quantities use `_{bus_name}` (or a remote bus if `data[…]` has `'bus'`).
- Steady-state initialisation hints in `xy_0_dict` — calculate them from the algebraic residuals, not from thin air. Copy the commentary pattern from `sexs.py`.

**ini/run swap rule (voltage-setpoint controllers only, like AVRs):**
Replace in place — do not append-then-delete:
```python
if v_t in dae['y_ini']:
    idx_V = dae['y_ini'].index(v_t)
    dae['y_ini'][idx_V] = v_ref
else:
    dae['y_ini'] += [v_ref]
```
This preserves downstream code that references `y_ini` by integer index (notably `vsource`'s `g[idx_V] = ...`). Controllers that do not regulate a voltage setpoint (most governors, PSSs, PODs) do **not** need this swap — just wire normal inputs.

Also include an in-module `def test():` that:
- Builds a minimal two-bus network from the sibling `.hjson` via `BpsBuilder('<name>.hjson')`.
- Calls `bld = Builder(grid.sys_dict, target="ctypes", sparse=False); bld.build()`.
- Runs `model.ini(...)`, asserts on an observable (e.g. `V_1 ≈ v_set` for an AVR), then does a second `model.ini` + chained `model.run` to exercise the run phase.
- Prints first/last timestamps + values. End with `if __name__ == '__main__': test()`.

The `.hjson` fixture: copy the closest sibling's fixture, update the component-config block to reference the new type and its new parameters. Keep the two-bus layout and the `vsource` on bus 2 so the PV-bus initialisation has somewhere to regulate against.

After generating:
1. Register the new `type` in the family dispatcher (`avrs/avrs.py`, `govs/govs.py`, `psss/psss.py`, `pods/pods.py`). Add an `elif` branch mirroring an existing entry.
2. Add a Sphinx stub at `docs/pydae-bps/models/<family>/<name>.rst` — autosummary pulls from `descriptions()`.
3. Suggest `uv run python -m pydae.bps.<family>.<name>` (or call `test()` directly) to validate.

Do not commit. Report the files created.
