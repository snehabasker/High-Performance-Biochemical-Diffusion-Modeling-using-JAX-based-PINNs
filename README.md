# JAX Physics-Informed Neural Network for Diffusion

A PINN in JAX that solves the 1-D diffusion equation `u_t = D u_xx` on `x, t ∈ [0, 1]`, a toy model of molecule (e.g. drug) diffusion through tissue.
Uses `jax.grad` (nested for `u_xx`), `vmap`, `jit` and Optax Adam; 4×(64 tanh) MLP.

* `High_Performance_Biochemical_Diffusion_Modeling_using_JAX_based_PINNs.ipynb` – original notebook (PDE-residual loss only).
* `pinn_diffusion.py` – validated version with initial/boundary conditions and an exact-solution comparison (`python pinn_diffusion.py`; ~20 min on CPU).

## Validation (`pinn_diffusion.py`)

Setup: `D = 0.1`, `u(0, x) = sin(πx)`, `u(t, 0) = u(t, 1) = 0`. Exact solution `u = exp(−Dπ²t) sin(πx)`.
Error measured on a 101×101 (t, x) grid, 3 seeds, 10 000 Adam steps, 2 000 collocation points.

| Loss | Relative L2 error vs exact (mean ± std) | Max abs error |
|---|---|---|
| PDE residual only (original notebook's loss) | **0.983 ± 0.034** | ≈ 1.0 |
| **Residual + initial + boundary conditions** | **0.0006 ± 0.0001** | 0.0014 – 0.0025 |

The residual-only loss reaches a PDE residual of ~1e-8 while being completely wrong (error ≈ 100 %): without initial/boundary
conditions the network collapses to a near-trivial solution that satisfies the equation but not the problem. Adding IC/BC terms
(weight 10) reduces the error by ~1600×.

## Limitations

* A 1-D linear PDE with a known analytic solution – a verification benchmark, not a model of real tissue or measured drug data.
* The constant coefficient `D = 0.1` is illustrative (the original notebook used 0.01); no real pharmacokinetic parameters.
* Single configuration; no comparison with a finite-difference/FEM solver or runtime benchmark yet.
