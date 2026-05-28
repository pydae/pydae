# uvsg_high

*Virtual synchronous generators — pydae-bps model.*

## Model description

Created on Thu August 10 23:52:55 2022

@author: jmmauricio

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `uvsg_high` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

```{note}
This model does not yet define a `descriptions()` function. Add one to
`packages/pydae-bps/src/pydae/bps/vsgs/uvsg_high.py` and re-run `docs/pydae-bps/_scripts/generate_model_pages.py`
to populate this section automatically. See
`packages/pydae-bps/src/pydae/bps/syns/milano2ord.py` for a reference
implementation.
```


## Source

- Module: `pydae.bps.vsgs.uvsg_high`
- File: [`packages/pydae-bps/src/pydae/bps/vsgs/uvsg_high.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/vsgs/uvsg_high.py)
