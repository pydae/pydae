# Models

The `pydae-bps` library ships a library of power-system component models.
Pages are auto-generated from each model file's module docstring and its
`descriptions()` function.

```{toctree}
:maxdepth: 2

syns/index
avrs/index
govs/index
psss/index
vsgs/index
vscs/index
vsc_ctrls/index
wecs/index
pvs/index
loads/index
lines/index
pods/index
sources/index
```

To regenerate these pages after adding a new model or updating an existing
one, run:

```bash
python docs/pydae-bps/_scripts/generate_model_pages.py
```
