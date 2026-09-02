# SchwingerBosonXXZ

[![Build Status](https://github.com/Hao-Phys/SchwingerBosonXXZ.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Hao-Phys/SchwingerBosonXXZ.jl/actions/workflows/CI.yml?query=branch%3Amain)

Schwinger-boson mean-field and Gaussian-fluctuation calculations for the XXZ
Heisenberg model on a triangular lattice.

## Magnetic fields

`SchwingerBosonSystem` supports two distinct magnetic fields:

- `h_SB` is the sublattice-dependent symmetry-breaking field used to select
  and stabilize a local ordered state. Its directions are set by `θs` in the
  x-z plane.
- `h_ext` is a uniform physical external-field magnitude in units of `J`. Its
  Cartesian direction is set by `h_ext_direction`.

The external-field direction accepts either a length-3 tuple or vector and is
normalized when the system is constructed or when `set_external_field!` is
called.

```julia
using SchwingerBosonXXZ

sbs = SchwingerBosonSystem(
    1.0,  # J
    1.5,  # Δ
    0.5,  # S
    0.01, # T/J
    12;   # L
    h_ext = 0.2,
    h_ext_direction = (1.0, 0.0, 1.0),
)

# Stored as the normalized direction (1/√2, 0, 1/√2).
set_external_field!(sbs, 0.1, [0.0, 1.0, 0.0])
```

Both fields may be nonzero; their contributions to the mean-field Hamiltonian
are additive.
