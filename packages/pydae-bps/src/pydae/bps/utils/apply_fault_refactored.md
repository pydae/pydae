# `pydae-bps` Fault Manager Module Documentation

## 1. Introduction

The `FaultManager` module (historically accessed via `apply_fault`) is designed to handle the robust application, holding, and clearing of shunt faults in `pydae-bps` dynamic power system simulations.

Because severe grid faults inherently cause profound mathematical singularities in power flow equations, this module employs advanced continuation solvers and Low-Voltage Ride-Through (LVRT) injection modeling to ensure the Newton-Raphson algebraic solver survives the transient event without crashing.

---

## 2. Theoretical Background: The Singularity Problem

### 2.1 The Constant Power (PQ) Singularity

Under normal grid conditions, grid-following converters (such as PV inverters or Battery Energy Storage Systems) operate in constant-power (PQ) mode. The relationship between apparent power $S$, terminal voltage $V$, and injected current $I$ is given by:

$$I = \frac{S^*}{V^*}$$

During a severe shunt fault, the terminal voltage at the faulted bus drops precipitously toward zero ($\vert{}V\vert{} \to 0$). To maintain constant power, the converter's control loops theoretically demand an exponentially increasing current ($\vert{}I\vert{} \to \infty$). In the simulation environment, this causes the algebraic Jacobian matrix to go singular, resulting in a `RuntimeError` during the Newton-Raphson solve.

### 2.2 The LVRT Solution (Bounded Current Injection)

To bypass this singularity, the module detects grid-following converters (devices exposing the `lvrt_ext` interface) and forces them into **Low-Voltage Ride-Through (LVRT)** mode. In this mode, the converter abandons constant-power targets and instead injects a bounded, safe current proportional to the remaining voltage:

$$P_s = \vert{}V\vert{} i_{sa\_ref}$$

$$Q_s = \vert{}V\vert{} i_{sr\_ref}$$

Where:

* $i_{sa\_ref}$: Active current reference (typically **0.0** during a fault).
* $i_{sr\_ref}$: Reactive current reference (typically **0.5** pu to support grid voltage).

By binding the current, the algebraic equations remain well-conditioned even as $\vert{}V\vert{} \to 0$.

---

## 3. The Adaptive Continuation Engine

Because transitioning the grid from a healthy state to a faulted state creates a massive discontinuity, the `FaultManager` does not apply the fault in a single Newton solve. Instead, it uses a **Homotopy Continuation** method.

The `_adaptive_continue` method bridges the gap between the initial state and the target state (e.g., fault admittance ramped from 0 to its final value).

**How it works:**

1. **Step Forward:** The solver attempts to advance the fault admittance by a fractional step `step_size`.
2. **Solve Algebra:** The underlying `model.recalculate_algebraics()` is called.
3. **Adaptive Rollback:** If the algebraic solver diverges (throws a `RuntimeError`), the state is rolled back to the last successful checkpoint, and the `step_size` is halved.
4. **Acceleration:** If the solver succeeds, the state is saved, and the `step_size` is multiplied by 1.5 for the next push.

This guarantees that computational effort is only spent where the operating point is stiff.

---

## 4. LVRT Relay Strategies

The module provides two distinct strategies for transitioning converters between PQ mode and LVRT mode. The transition variable is $L_{ext}$, where $L_{ext} = 1$ denotes full LVRT and $L_{ext} = 0$ denotes normal PQ.

### 4.1 The Hard Relay (Default)

The hard relay mimics physical, latched undervoltage relays. It acts as a binary switch.

* **Logic:** If the terminal voltage $\vert{}V\vert{} \le V_{lvrt}$, the unit switches instantly to $L_{ext} = 1$. If the voltage recovers above the threshold, it latches back to $L_{ext} = 0$ and never returns to LVRT during the fault hold.
* **Application Algorithm (`_apply_fault_hard_relay`):**
1. The solver assumes the worst-case scenario and attempts to push *all* converters into LVRT while simultaneously ramping in the fault admittance.
2. If this fails (usually because the fault is mild and massive reactive injection causes overvoltage), it falls back to ramping the fault with units in PQ.
3. Once the fault is applied, a monotone demotion loop begins. It checks the voltage of every unit currently in LVRT. If $\vert{}V\vert{} > V_{lvrt}$, the unit is commanded back to PQ mode. The system is re-solved, and the loop repeats until no further units demote.


* **Clearing Algorithm:** A hysteresis of 0.1 pu ($V_{relax} = V_{lvrt} - 0.1$) is used to clear the units back to PQ to prevent chattering.

### 4.2 The Soft Continuous Relay (`lvrt_soft=True`)

The soft relay replaces the binary mode switch with a continuous mathematical function. It crosses the intermediate fault-severity band smoothly, preventing mode-discontinuity crashes in symmetric-plant arrays.

* **Logic:** Each unit's LVRT activation is a continuous ramp of its own undervoltage defined by the threshold $V_{lvrt}$ and a bandwidth $V_{band}$:
$$L_{ext}^{target} = \min\left(1.0, \max\left(0.0, \frac{V_{lvrt} - \vert{}V\vert{}}{V_{band}}\right)\right)$$


* **Damped Fixed-Point Iteration (`_relax_lvrt`):**
Because $L_{ext}$ depends on $\vert{}V\vert{}$, and $\vert{}V\vert{}$ depends on the reactive support triggered by $L_{ext}$, this forms an implicit loop. The solver resolves this at *every continuation step* using a damped fixed-point sweep:
$$L_{ext}^{(k+1)} = L_{ext}^{(k)} + \alpha \left( L_{ext}^{target} - L_{ext}^{(k)} \right)$$


Where the damping factor $\alpha = 0.5$. The sweep repeats up to 20 times until the maximum change in $L_{ext}$ across all devices is strictly less than **1e-5**.

### 4.3 Comparison of Relay Options

| Feature | Hard Relay (Default) | Soft Relay |
| --- | --- | --- |
| **Transition Type** | Binary / Latched (0.0 or 1.0) | Continuous (0.0 to 1.0) |
| **Physical Realism** | High (mimics standard undervoltage hardware) | Low (abstracted solver convenience) |
| **Mathematical Stability** | Prone to folding in symmetric-plants | Highly robust across stiff operating points |
| **Control Logic** | Demote-only monotonic loops | Damped fixed-point iterative sweeps |

---

## 5. Event Chronology (`execute_fault`)

Regardless of the chosen relay strategy, the fault event follows a strict three-phase chronology handled by the `execute_fault()` method:

1. **Apply (Frozen Time):** The simulation clock $t$ is paused. The target bus admittance ($G_{fault}$ and $B_{fault}$) is introduced via continuation. The converters are routed into LVRT based on the chosen relay strategy.
2. **Hold (Dynamic Integration):** The target modes and references are collected into a dictionary (`hold_cmd`). The underlying CasADi integrator is invoked via `model.run(t_incept + t_clear, hold_cmd)`. This simulates the grid's dynamic physical response over the duration of the fault.
3. **Clear (Frozen Time):** Time is paused again. A reverse continuation ramps the fault admittance back to zero ($G=0$, $B=-1e-6$). Converters are returned to their standard PQ operating states, allowing the post-fault simulation to resume cleanly.

---

## 6. API Reference

### `FaultManager(model, bus_name, ...)`

**Initialization Parameters:**

* **`model`** (`CasadiModel`): The pre-initialized simulation model. Must contain a `"faults"` dictionary in `model.sys_dict`.
* **`bus_name`** (`str`): The exact string identifier of the bus where the fault will be applied.
* **`lvrt_iq`** (`float`): Reactive current reference $i_{sr\_ref}$ injected during LVRT. Default is **0.5**.
* **`lvrt_id`** (`float`): Active current reference $i_{sa\_ref}$ injected during LVRT. Default is **0.0**.
* **`v_lvrt`** (`float`): The terminal voltage threshold ($V_{lvrt}$) below which LVRT engages. Default is **0.5**.
* **`lvrt_soft`** (`bool`): Toggle between the default Hard Relay (`False`) and the Continuous Soft Relay (`True`).
* **`lvrt_band`** (`float`): The voltage bandwidth ($V_{band}$) used exclusively by the Soft Relay. Full LVRT occurs at $V_{lvrt} - V_{band}$. Default is **0.2**.
* **`fault_steps`** (`int`): The nominal number of baseline continuation steps. The solver will adaptively increase or decrease this based on stiffness. Default is **100**.
* **`verbose`** (`bool`): If True, prints a summary of the resolved per-device modes to standard output.

### `execute_fault(t_clear)`

* **`t_clear`** (`float`): The duration in seconds that the fault is held on the system before being cleared.
* **Returns** (`Dict[str, float]`): A dictionary mapping each device name to its resolved $L_{ext}$ activation state during the fault hold period.