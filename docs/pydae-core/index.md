# pydae-core

**The DAE solver engine at the heart of the pydae ecosystem.**

`pydae-core` combines symbolic computation ([SymPy](https://www.sympy.org) or
[CasADi](https://web.casadi.org)) with compiled C code (via `ctypes` / `CFFI`)
or native CasADi integrators (IDAS) to provide a fast, user-friendly solver
for Differential-Algebraic Equation (DAE) systems of the form:

$$
\begin{aligned}
\dot{x} &= f(x, y, u, p) \\
0 &= g(x, y, u, p)
\end{aligned}
$$

It is developed with power-systems analysis in mind but is general enough for
any engineering or scientific problem that can be posed as a DAE.

```{admonition} Install
:class: tip

    pip install pydae
```

## Documentation sections

```{toctree}
:maxdepth: 2
:caption: Getting started

overview
getting_started
architecture
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
ssa
realtime_api
```

## Related packages

- [`pydae-bps`](https://pydae-bps.readthedocs.io) — balanced power systems
  builder (transmission-level, PSS/E / PSAT-style networks).
- [`pydae-uds`](https://pydae-uds.readthedocs.io) — unbalanced distribution
  systems builder (three-phase, four-wire networks).

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
