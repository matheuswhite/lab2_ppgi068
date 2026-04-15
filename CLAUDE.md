# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Master's degree lab assignment implementing control systems identification in Rust. The program analyzes two predefined transfer function systems (first and second in `src/system/`) and processes six experimental CSV datasets from `samples/`. It produces console output and PNG plots in `images/`.

## Commands

```bash
cargo build          # compile
cargo run            # build and run (outputs to stdout + saves plots to images/)
cargo check          # fast type/borrow check without linking
cargo clippy         # lint
```

There are no tests. The program itself is the experiment — all output goes to stdout and `images/`.

## Architecture

The program is structured around six question modules (`src/questions/question1.rs` through `question6.rs`), each implementing a step of the lab assignment. `main.rs` wires everything together by calling each question function with system instances.

### Signal flow (aule framework)

The `aule` crate provides the block-based DSP framework. Components implement the `Block` trait, producing typed `Signal` values. Signals are composed via `.pack()` and consumed via `.output(dt)`. The `Time` iterator drives the simulation loop, yielding a timestamped `dt` value each step.

```
Time iterator → Step/RandomSignal → DifferenceEquation.output(input) → Plotter
```

### System trait (`src/system/mod.rs`)

`System` trait (extends `StepResponse`) provides:
- `poles_analysis()` — roots of characteristic polynomial, stability classification, damping ratios, natural frequencies
- `to_dtf(dt)` — discretises the continuous transfer function using Tustin (bilinear) method
- `to_diff_eq(dt, noise_coeffs)` — returns a `DifferenceEquation` ready for simulation

Discretisation uses the Tustin transform: substituting `s = (2/dt) * (z-1)/(z+1)` via polynomial arithmetic with the `aule::continuous::Polynomial` type.

### DifferenceEquation (`src/diff_eq.rs`)

Implements `Block<Input=(Signal, Signal), Output=Signal>`. Holds circular buffers of past inputs and outputs. The `simulate()` method returns a `SimulationResult { inputs, outputs, noises }`.

### Identification algorithms

| File | Algorithm | Model type |
|------|-----------|------------|
| `src/ordinary_least_squares.rs` | Batch OLS | ARX / ARMAX |
| `src/recursive_least_squares.rs` | RLS | ARX |
| `src/recursive_extended_least_squares.rs` | RELS | ARMAX |

All use `ndarray::Array2` for matrix operations. The regressor matrix `phi` is built from lagged input/output/noise vectors; `ndarray-inverse` provides `inv()`.

### Shared question utilities (`src/questions/mod.rs`)

- `identify_systems()` — sweeps orders 1–5, fits OLS, prints residue mean
- `SSE::eval()` — sum of squared errors
- `AkaikeInformationCriterion::eval()` / `BayesianInformationCriterion::eval()` — model selection criteria

### Key known issues (tracked in `analise_roteiro_laboratorio_2.md`)

- RLS/RELS regressor vector is misaligned for model orders > 1 (off-by-one in circular buffer indexing)
- AIC/BIC parameter counts are incomplete (don't include noise model parameters)
- `identify_systems` uses `enable_noise` to switch between ARX (`noise_order=0`) and ARMAX (`noise_order=1`); the ARMAX path in questions 4–6 has correctness issues
- Plots are saved to `images/` and also displayed inline (terminal-dependent)
