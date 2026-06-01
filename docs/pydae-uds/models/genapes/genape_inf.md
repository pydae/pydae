# genape_inf

*GENAPE sources — pydae-uds model.*

## Model description

Created on Thu August 10 23:52:55 2022

@author: jmmauricio

## Usage

```python
from pydae.uds import UdsBuilder

grid = UdsBuilder("my_network.hjson")
grid.construct("my_system")
```

The `genape_inf` model is instantiated by including an entry in the relevant
section of the network HJSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

```{note}
This model does not yet define a `descriptions()` function. Add one to
`packages/pydae-uds/src/pydae/uds/genapes/genape_inf.py` and re-run `docs/pydae-uds/_scripts/generate_model_pages.py`
to populate this section automatically. See
`packages/pydae-bps/src/pydae/bps/syns/milano4ord.py` for a reference
implementation.
```


## Source

- Module: `pydae.uds.genapes.genape_inf`
- File: [`packages/pydae-uds/src/pydae/uds/genapes/genape_inf.py`](https://github.com/pydae/pydae/tree/master/packages/pydae-uds/src/pydae/uds/genapes/genape_inf.py)
