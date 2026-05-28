# smib_vsc_pq_inf

*Automatic voltage regulators — pydae-bps model.*

## Model description

*(No module docstring provided — add one to the source file and re-run the generator.)*

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `smib_vsc_pq_inf` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

```{note}
This model does not yet define a `descriptions()` function. Add one to
`packages/pydae-bps/src/pydae/bps/avrs/smib_vsc_pq_inf.py` and re-run `docs/pydae-bps/_scripts/generate_model_pages.py`
to populate this section automatically. See
`packages/pydae-bps/src/pydae/bps/syns/milano2ord.py` for a reference
implementation.
```


## Source

- Module: `pydae.bps.avrs.smib_vsc_pq_inf`
- File: [`packages/pydae-bps/src/pydae/bps/avrs/smib_vsc_pq_inf.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/avrs/smib_vsc_pq_inf.py)
