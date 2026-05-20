# Plant Power Controllers (ppcs)

Plant-level supervisory controllers that monitor the point of
interconnection (POI) and command one or several WECC converter units
(``weccs``).

Unlike the ``reec`` local controls (nested inside a ``weccs`` entry),
plant controllers are declared as a separate top-level section in the
HJSON file, identified by ``ppcs``.  Each entry lists the converter
names it governs via the ``weccs`` key.

```hjson
ppcs: [{
    type:    "repc_a",
    reg_bus: "POI",
    weccs:   ["gen1", "gen2"],
    // ...
}]
```

```{toctree}
:maxdepth: 1

repc_a
repc_d
```
