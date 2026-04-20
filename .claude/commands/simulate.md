---
description: Build and simulate a pydae model end-to-end
argument-hint: "<model-name>  (e.g. pendulum | milano2ord_pm)"
---

Drive a compile + initialise + run cycle for `$ARGUMENTS`.

Workflow:
1. Locate the example or test that constructs `$ARGUMENTS`. Check in order:
   - `examples/$ARGUMENTS.py`
   - `examples/$ARGUMENTS_build.py` + `examples/$ARGUMENTS_example.py`
   - component self-tests under `packages/pydae-bps/src/pydae/bps/**/` (call the module's `test()`)
2. If no builder script is found, ask the user for the `sys_dict` source. Do not fabricate parameters.
3. Build:
   ```python
   from pydae.core import Builder, Model
   bld = Builder(sys_dict, target="ctypes", sparse=False)
   bld.build()
   ```
   Prefer `target="cffi"` on Windows if the user reports ctypes DLL-load issues.
4. Initialise, run, and report:
   ```python
   model = Model("$ARGUMENTS")
   model.ini(params_dict, 'xy_0.json')
   model.run(t_end, inputs_dict)
   model.post()
   ```
5. Print the first and last sample of one or two observables. If `ini()` raises, read `jacobian_diagnostic.png` / the Jacobian diagnostic output and summarise the likely cause (zero rows, near-zero pivot, conditioning) before retrying.

Do not commit. Do not write new files under the repo — keep build artefacts in the current working dir.
