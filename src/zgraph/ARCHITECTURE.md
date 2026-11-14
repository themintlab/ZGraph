# ZGraph Architecture & Design Philosophy

`zgraph` is a **Soft-Tropical Sum-Product Network (Factor Graph)** compiler for differentiable belief propagation and convex optimization, built on JAX and Equinox:

- **ICNN & DC Closure:** Each node maps child log-potentials $\mathbf{z} \in \mathbb{R}^N$ via incidence matrix $M \in \mathbb{R}^{K \times N}$, log-degeneracy $\boldsymbol{g} \in \mathbb{R}^K$, and scale $\tau \in [0, \infty)$ to $z^+(\mathbf{z}) = \tau \, \text{LSE}(\tau^{-1} M \mathbf{z} + \boldsymbol{g})$, with simplex branch probabilities $\mathbf{p} = \text{softmax}(\tau^{-1} M \mathbf{z} + \boldsymbol{g}) \in \Delta^{K-1}$, marginals $\mathbf{n} = \nabla_{\mathbf{z}} z^+ = M^T \mathbf{p}$, and PSD Hessian $\mathbf{H}_{\mathbf{z}} = \nabla^2_{\mathbf{z}} z^+ \succeq 0$. Non-negative weights ($M \ge 0$) form monotone **Input-Convex Neural Networks (ICNNs)** where reverse-mode AD (`jax.grad` / `jax.vjp`) executes exact sum-product belief propagation; general signed weights ($M \in \mathbb{R}^{K \times N}$) form **Difference-of-Convex (DC)** networks.
- **Perspective Homogeneity:** Nodes and composite DAGs are degree-1 positively homogeneous on $(\mathbf{z}, \tau) \in \mathbb{R}^N \times [0, \infty)$, satisfying Euler's identity $z^+(\mathbf{z}) = \mathbf{z}^T \mathbf{n} + \tau (\partial z^+ / \partial \tau)$ and $z^*(\mathbf{n}) = \mathbf{z}^{*T} \mathbf{n} - z^+(\mathbf{z}^*) = -\tau (\partial z^+ / \partial \tau)$.

To ensure speed and portability across platforms and hardware architectures, all code within this package MUST adhere to the following strict architectural directives. This ensures robust compilation via `jax.jit` and seamless compatibility with `jax`.

`zgraph` is purely mathematical and has no knowledge of the application domain. All labelling of variables, modules, and physical significance (e.g., mapping gradients to CALPHAD composition tie-lines) are relegated to independent domain packages (like `thermograph`).

---

## Architectural Division

The `zgraph` codebase is strictly divided into three distinct categories based on their mathematical role:

1. **`core/` (The Soft-Tropical Forward Pass)**
   Contains the structural math primitives (`ZNode`, `ZMatMulNode`, `ProductNode`, `marginalize`) defining the graphical model:
   $$z^+(\mathbf{z}) = \tau \, \text{LSE}\!\left(\tau^{-1} M \mathbf{z} + \boldsymbol{g}\right)$$
   Executes a zero-scale-safe ($\tau \to 0^+$) shifted-`LSE` `evaluate()` pass (wrapped by `__call__()`)—with compile-time linear bypass for $K = 1$ and horizontal operator fusion (`+`, `-`, `*`, `|`)—exposing the scalar potential $z^+$, uncollapsed branch potentials $z_i^+ + \tau g_i$ (`.branch_potentials()`, aliased as `.logits()`), and simplex probabilities $\mathbf{p}$ (`.probabilities()`).

2. **`transforms/` (Graph-to-Graph Operators)**
   Contains functional operators (`legendre_transform`, `legendre_dual`) that reshape coordinates and return a *new* `BaseNode` graph:
   - `legendre_transform`: Forward parametric Legendre-Fenchel transform evaluating $z^*(\mathbf{n}) = \mathbf{z}^T \mathbf{n} - z^+(\mathbf{z})$ and dual marginals $\mathbf{n} = \nabla_{\mathbf{z}} z^+$.
   - `legendre_dual`: Inverse Legendre-Fenchel node wrapping `legendre_solve` to evaluate $z^*(\mathbf{n}) = \sup_{\mathbf{z}} \{\mathbf{z}^T \mathbf{n} - z^+(\mathbf{z})\}$ at target marginals $\mathbf{n}$.

3. **`solvers/` (Execution Routines)**
   Contains numerical and analytic routines (`legendre_solve`, `gauge_fix`, `extract_decision_boundary`) operating on a **callable function** and an **input domain tensor**:
   - `legendre_solve`: Damped pseudoinverse Newton-Raphson solver on $\text{range}(\mathbf{H}_{\mathbf{z}})$ with optional gauge anchoring onto $z^+(\mathbf{z}^*) = 0$ and $\mathcal{O}(1)$-memory implicit differentiation.
   - `gauge_fix`: Single-pass affine projection onto a target potential isosurface (default null-potential isosurface $z^+(\mathbf{z}) = 0$) along shift-equivariant directions.
   - `extract_decision_boundary`: Hardware `jax.lax.top_k` solver locating pairwise branch crossovers $(z_i^+ + \tau g_i) = (z_j^+ + \tau g_j)$ from uncollapsed potentials with implicit differentiation sensitivities.

---

## Core Design Principles

To prevent breaking JAX/Equinox native `vmap` and `jax.jit` compatibility, all contributors must strictly adhere to the following rules:

1. **Pure Math, No Python Objects:**
   **No standard Python objects (strings, lists of strings, dictionaries) or Python control flow (`if` statements based on string matching or traced values) may exist inside `zgraph` `eqx.Module` classes or their `evaluate()` passes.**
   - *Reason:* `jax.jit` requires strict static typing. Dictionaries or string parsing cause graph breaks and kernel compilation failures. All domain knowledge (names, metadata) must remain in the application layer.

2. **Structural Immutability:**
   `zgraph` routing nodes (e.g., `ZNode`) are structurally immutable after creation. You may **NOT** include methods that mutate the underlying `list` or `tuple` (within `eqx.Module`) or swap subgraphs in-place (e.g., `self.subgraphs[2] = new_model`).
   - *Reason:* Swapping nodes in-place corrupts JAX's tracer graph and invalidates fused XLA kernels.

3. **GPU & Device Management (The Buffer Rule):**
   - **Static indices and configuration flags** MUST be registered using `eqx.field(static=True)` or as integer/float JAX arrays.
   - **Learnable constants** must use standard JAX arrays (Equinox treats all floating-point JAX arrays as trainable parameters unless marked static).
   - **Never** store lists of integers or floats as raw attributes (e.g., `self.indices = [0, 1]`) if they are indexed inside the `evaluate()` pass.

4. **Tensor-Only Communication:**
   All inputs and outputs between `zgraph` modules must be `jax.Array` types (or tuples of `jax.Array` for multi-output transforms like `LegendreTransform`).

5. **Computational Efficiency (No Python Loops over Batches/States):**
   **Never** use Python `for` or `while` loops over spatial dimensions, batches, or branches inside an `evaluate()` pass. All operations must be vectorized using native JAX tensor operations, broadcasting, or `vmap`.

6. **Clean Code & Strict Typing:**
   All functions and methods must use explicit type hints from the built-in `typing` module.

---

## 4. The Functor Pattern (`BaseNode`)

To maximize both code readability and hardware performance, `zgraph` implements a strict Template Method design pattern via the `BaseNode` base class.

* **Pristine Subclasses:** Subclasses (`ZNode`, `ZMatMulNode`, `ProductNode`, `LegendreTransform`, `LegendreDualNode`) must NEVER implement `__call__`. Instead, they implement their pure, unvectorized scalar mathematical logic inside `evaluate(self, signals)`.
* **The Interceptor:** The `BaseNode` base class takes absolute ownership of `__call__`. When a user evaluates `system(signals)`, the base class automatically intercepts the execution, detects batch dimensions or uncertainty parameters (`Ensemble`), constructs the necessary JAX masks, wraps `evaluate` in `jax.vmap` and `eqx.filter_vmap`, and executes the hardware-optimized kernel.

## 5. Universal Uncertainty Quantification (UQ)

ZGraph treats aleatoric fluctuations (scale $\tau$) and epistemic parameter uncertainty (`Ensemble`) distinctly:
* **Aleatoric / Entropic:** Handled natively inside the core scalar logic via exact `LogSumExp` partition reductions at scale $\tau$.
* **Epistemic:** Executed externally via vectorized Monte Carlo parameter ensembles mapped over the graph via JAX.

### The `Ensemble` PyTree Wrapper
To distinguish parameter uncertainty distributions from vector constants (such as incidence weights $M$ or degeneracy biases $\boldsymbol{g}$), uncertainty parameters are wrapped in `Ensemble(eqx.Module)` with leading sample dimension $S$.
* The `__call__` interceptor recursively scans the PyTree for `Ensemble` objects and dynamically unwraps them inside `eqx.filter_vmap` before passing raw arrays into the pure `evaluate` method.

### The `.ensemble` BatchProxy
For secondary node methods (such as `.branch_potentials(signals)`, `.logits(signals)`, or `.probabilities(signals)`), the base class provides a `BatchProxy` via the `.ensemble` property:
* **Usage:** `system.ensemble.branch_potentials(signals)`
