# Models

The `pydae-uds` library ships a library of unbalanced-distribution component
models. Pages are auto-generated from each model file's module docstring and
its `descriptions()` function.

```{toctree}
:maxdepth: 2

sources/index
transformers/index
lines/index
loads/index
shunts/index
vscs/index
vsgs/index
vsc_ctrls/index
ess/index
fcs/index
pvs/index
genapes/index
miscellaneous/index
```

To regenerate these pages after adding a new model or updating an existing
one, run:

```bash
python docs/pydae-uds/_scripts/generate_model_pages.py
```
