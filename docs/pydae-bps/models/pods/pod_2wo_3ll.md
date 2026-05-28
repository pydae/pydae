# pod_2wo_3ll

*Power oscillation dampers — pydae-bps model.*

## Model description

Created on Thu August 10 23:52:55 2022

@author: jmmauricio

## Usage

```python
from pydae.bps import BpsBuilder

grid = BpsBuilder("my_network.json")
grid.construct("my_system")
```

The `pod_2wo_3ll` model is instantiated by including an entry in the relevant
section of the network JSON (see [Overview](../../overview.md)).

## Parameters, inputs, states, outputs

### Parameters

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $$K_{p,pll}$$ | `K_p_pll` | 1.0 |  | PLL proportional gain |
| $$K_{i,pll}$$ | `K_i_pll` | 1.0 |  | PLL integral gain |
| $$T_{pll}$$ | `T_pll` | 0.01 |  | PLL filter time constant |
| $$K_{stab}$$ | `K_stab` | 1.0 |  | POD stabilizer gain |
| $$T_{lpf}$$ | `T_lpf` | 0.02 |  | Low-pass filter time constant |
| $$T_{wo1}$$ | `T_wo1` | 2.0 |  | Washout 1 time constant |
| $$T_{wo2}$$ | `T_wo2` | 2.0 |  | Washout 2 time constant |
| $$T_1$$ | `T_1` | 0.1 |  | Lead-lag 1 numerator time constant |
| $$T_2$$ | `T_2` | 0.02 |  | Lead-lag 1 denominator time constant |
| $$T_3$$ | `T_3` | 0.1 |  | Lead-lag 2 numerator time constant |
| $$T_4$$ | `T_4` | 0.02 |  | Lead-lag 2 denominator time constant |
| $$T_5$$ | `T_5` | 0.1 |  | Lead-lag 3 numerator time constant |
| $$T_6$$ | `T_6` | 0.02 |  | Lead-lag 3 denominator time constant |

### Inputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $$V_s$$ | `V` |  |  | Bus voltage magnitude |
| $$\theta_s$$ | `theta` |  |  | Bus voltage angle |
| $$\omega_\mathrm{coi}$$ | `omega_coi` |  |  | Center of inertia speed |

### Dynamic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $$\theta_{pll}$$ | `theta_pll` |  |  | PLL angle |
| $$\xi_{pll}$$ | `xi_pll` |  |  | PLL integrator state |
| $$\omega_{pll,f}$$ | `omega_pll_f` |  |  | Filtered PLL speed |
| $$x_{lpf}$$ | `x_lpf_pod` |  |  | Low-pass filter state |
| $$x_{wo1}$$ | `x_wo1_pod` |  |  | Washout 1 state |
| $$x_{wo2}$$ | `x_wo2_pod` |  |  | Washout 2 state |
| $$x_{12}$$ | `x_12_pod` |  |  | Lead-lag 1-2 state |
| $$x_{34}$$ | `x_34_pod` |  |  | Lead-lag 3-4 state |
| $$x_{56}$$ | `x_56_pod` |  |  | Lead-lag 5-6 state |

### Algebraic States

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $$\mathrm{rocof}$$ | `rocof` |  |  | Rate of change of frequency |
| $$\mathrm{pod\_out}$$ | `pod_out` |  |  | POD output (saturated) |

### Outputs

| Symbol | Variable | Default | Units | Description |
|---|---|---|---|---|
| $$\omega_{pll}$$ | `omega_pll` |  |  | PLL speed |
| $$\omega_{pll,f}$$ | `omega_pll_f` |  |  | Filtered PLL speed |
| $$\mathrm{rocof}$$ | `rocof` |  |  | Rate of change of frequency |


## Source

- Module: `pydae.bps.pods.pod_2wo_3ll`
- File: [`packages/pydae-bps/src/pydae/bps/pods/pod_2wo_3ll.py`](https://github.com/pydae/pydae/tree/main/packages/pydae-bps/src/pydae/bps/pods/pod_2wo_3ll.py)
