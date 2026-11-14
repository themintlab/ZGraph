# ZGraph — Open Design Items & Deferred Benchmarks

This document tracks open architectural design questions and deferred benchmark implementations for `ZGraph`.

---

## Open Design Items

### 1. Continuous Sub-Grid Decision Boundary Refinement
- **Files:** `src/zgraph/solvers/boundaries.py`, `src/zgraph/solvers/functional.py`
- **Context:** `extract_decision_boundary` currently locates the discrete sweep grid index `boundary_idx = jnp.argmin(jnp.abs(delta_omega), axis=-1)` where the top two uncollapsed branches selected via `jax.lax.top_k` cross. Because `argmin` is piecewise constant, the boundary is quantized to the grid spacing and its gradient with respect to model parameters (`jax.grad`) is zero.
- **Formulation:**
  - Given the competing uncollapsed branch pair $(i_1, i_2)$ along sweep direction $\mathbf{d}$, project onto the equal-potential boundary $(z^+_{i_1} + \tau g_{i_1}) - (z^+_{i_2} + \tau g_{i_2}) = 0$ (or $\alpha^*_{i_1}(\mathbf{z}) - \alpha^*_{i_2}(\mathbf{z}) = 0$ on the null-potential isosurface $z^+(\mathbf{z}) = 0$) via closed-form affine projection or 1D Newton-Raphson refinement inside `jax.lax.while_loop`:
    $$\alpha_{t+1} = \alpha_t - \frac{(z^+_{i_1}(\mathbf{z}_t) + \tau g_{i_1}) - (z^+_{i_2}(\mathbf{z}_t) + \tau g_{i_2})}{(\mathbf{n}_{i_1}(\mathbf{z}_t) - \mathbf{n}_{i_2}(\mathbf{z}_t))^T \mathbf{d}}$$
  - Attach a `@jax.custom_jvp` rule applying the Implicit Function Theorem on the equal-potential manifold:
    $$\frac{\partial \alpha^*}{\partial \theta} = -\frac{\frac{\partial (z^+_{i_1} + \tau g_{i_1})}{\partial \theta} - \frac{\partial (z^+_{i_2} + \tau g_{i_2})}{\partial \theta}}{(\mathbf{n}_{i_1} - \mathbf{n}_{i_2})^T \mathbf{d}}$$
- **Decision Needed:** Whether to add `refine: bool = True` to `extract_decision_boundary` and/or expose a standalone `project_to_boundary` solver.

### 2. Vector `SignalNode` vs. Direct Vector Evaluation (`subgraph_list=None`)
- **Files:** `src/zgraph/core/leaf_nodes.py`, `src/zgraph/core/znodes.py`
- **Context:** Constructing a leaf `ZNode` over an $N$-dimensional input vector $\mathbf{z}$ currently requires creating $N$ scalar `SignalNode(0), ..., SignalNode(N-1)` objects, and `ZNode.branch_potentials` evaluates `[subgraph(signals) for subgraph in self.subgraphs]` followed by `jnp.stack`.
- **Options to Discuss:**
  - **Option A:** Allow `SignalNode` to accept a slice or index list so a single leaf node can pass a vector slice.
  - **Option B:** Allow `subgraph_list=None` on `ZNode(M_matrix)`, directly evaluating `self.M @ signals` without `SignalNode` wrappers.
  - **Option C:** Support both or retain explicit scalar `SignalNodes`.

### 3. Constrained Parameterization Wrappers for Training
- **Context:** During gradient-based parameter estimation (e.g., with Optax), general Difference-of-Convex (DC) networks treat $(M, \boldsymbol{g})$ as unconstrained real arrays, whereas scale convexity requires $\tau \ge 0$ (e.g., via `Softplus` or `exp`), monotone Input-Convex Neural Network (ICNN) priors require $M_{ij} \ge 0$, and shift-equivariance priors enforce row normalization $M \mathbf{u} = c_u \mathbf{1}_K$.
- **Decision Needed:** Should constrained parameterization helpers live inside `zgraph` (e.g., a `zgraph.nn` module or `ZNode` option), or belong in downstream training/application packages?

### 4. Multi-Subgraph `Ensemble` Resampling
- **Files:** `src/zgraph/core/base.py`
- **Context:** `BaseNode._execute_vectorized` applies a single `eqx.filter_vmap` over axis `0` of every `Ensemble` leaf in a DAG, requiring all `Ensemble` leaves to share the same sample count $S$. Combining independently calibrated subgraphs with mismatched sample counts ($S_A \neq S_B$) requires resampling each subgraph's `Ensemble` leaves to a common $S$—reusing the same random index draw within a subgraph to preserve intra-subgraph parameter covariance while drawing independent indices across distinct subgraphs.
- **Decision Needed:** Should this be an explicit utility `zg.align_ensembles(graph, num_samples=None, key=...)` (defaulting `num_samples` to `max(S_i)`), automatic behavior in `BaseNode`, or handled in the application layer?

---

## Deferred Benchmarks

1. **Neutronics / Deep Branching Transport:** Multi-stage attenuation, scattering, and multiplying DAG across $100+$ stages demonstrating overflow-free deep soft-tropical propagation, reverse-mode AD as exact adjoint importance $\mathbf{n}$, and criticality as the null-potential isosurface $z^+(\mathbf{z}) = 0$ via single-pass `gauge_fix` projection $\alpha^*$.
2. **Hierarchical Simplicial Cluster & Super-Cell Statistical Mechanics:** Two-tier binary ($A, B$) lattice cluster hierarchy at thermal scale $\tau = k_B T \in [0, \infty)$ combining a bottom-up $K=1$ simplicial recurrence and binomial cluster reduction node wired into a super-cell node, demonstrating exact site fractions $\mathbf{n}$ and pair-bond marginals $\mathbf{p}$ with $\mathcal{O}(1)$ Euler entropy $\partial z^+ / \partial \tau$, canonical free-energy inversion $z^*(\mathbf{n})$ via `legendre_solve` with $z^+(\mathbf{z}^*) = 0$ gauge fixing, $\tau \to 0^+$ ground-state hull and `top_k` implicit-differentiation boundary calibration, and vectorized `Ensemble` pushforward across an $A_2B_2$ dominance-complex bifurcation.
