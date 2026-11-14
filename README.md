# ZGraph

[![Unit Tests](https://github.com/themintlab/ZGraph/actions/workflows/unit-tests.yml/badge.svg?branch=main)](https://github.com/themintlab/ZGraph/actions/workflows/unit-tests.yml)

**Differentiable Belief Propagation and Convex Optimization via Soft-Tropical Sum-Product Networks.**

`zgraph` is a hardware-accelerated, domain-agnostic graph compiler built on JAX and Equinox (`jax.jit`, `jax.vmap`, `jax.grad`). It compiles hierarchical factor-graph contractions, sum-product belief propagation, and Legendre-Fenchel optimization into directed acyclic graphs (DAGs) of fused soft-tropical $\text{LogSumExp}$ ($\text{LSE}$) operators—unifying monotone **Input-Convex Neural Networks (ICNNs)** ($M \ge 0$) and **Difference-of-Convex (DC) networks** ($M \in \mathbb{R}^{K \times N}$).

---

## Core Philosophy: Physics-Free Pure Math

`zgraph` is **strictly domain-agnostic and physics-free**.

The engine operates exclusively on unlabeled numerical tensors (`jax.Array`). It contains zero domain-specific logic, zero unit conversions, and zero Python object overhead (such as string matching or dictionary lookups inside loops).

All physical nomenclature, thermodynamic ensembles, chemical potential mappings ($\mu$), state variables ($T, P$), phase definitions, and human-readable metadata are relegated to higher-level application layers such as [**ThermoGraph**](https://github.com/themintlab/ThermoGraph).

### Separation of Concerns

| Feature / Responsibility | **`zgraph`** (Math Engine) | **`thermograph`** (Application Layer) |
| :--- | :--- | :--- |
| **Domain Scope** | Abstract factor graphs, soft-tropical semirings | Thermodynamics, statistical mechanics, CALPHAD |
| **Data Types** | Unlabeled JAX Arrays (`jax.Array`) | Named variables, species, phases, physical units |
| **Core Operations** | Incidence projection, parameterized $\text{LogSumExp}$ | Ensemble definitions $(T, P, \mu)$, driving forces |
| **Optimization Target** | `jax.jit`, XLA compilation, `jax.vmap` vectorization | Phase equilibrium, convex hulls, phase-field coupling |
| **Graph Mutations** | Structurally immutable execution graphs | Dynamic model assembly & configuration |

---

## Key Pillars

### 1. Physics-Free Pure Tensor Math
By enforcing pure tensor-only signatures (`__call__(signals: Array) -> Array`), `zgraph` guarantees zero graph breaks during JIT compilation. Nodes do not store string names, metadata dicts, or conditional Python logic.

### 2. Soft-Tropical Sum-Product Algebra
Every node maps incoming child log-potentials $\mathbf{z} \in \mathbb{R}^N$ via incidence matrix $M \in \mathbb{R}^{K \times N}$, log-degeneracy bias $\boldsymbol{g} \in \mathbb{R}^K$, and scale $\tau \in [0, \infty)$ into a scalar log-potential:
$$z^+(\mathbf{z}) = \tau \, \text{LogSumExp}\!\left( \tau^{-1} M \mathbf{z} + \boldsymbol{g} \right)$$
- **Zero-Scale Limit ($\tau \to 0^+$):** Shifted-$\text{LSE}$ evaluation smoothly deforms sum-product marginalization ($\tau > 0$) into hard-tropical selection $\max_i (M\mathbf{z})_i$ ($\tau = 0$) without overflow or `NaN`s.
- **Perspective Homogeneity:** $z^+(\mathbf{z}, \tau)$ is degree-1 positively homogeneous on $\mathbb{R}^N \times [0, \infty)$ and satisfies Euler's decomposition $z^+(\mathbf{z}) = \mathbf{z}^T \mathbf{n} + \tau (\partial z^+ / \partial \tau)$ across composite DAGs.
- **ICNN & DC Closure:** Non-negative incidence weights ($M \ge 0$) yield globally monotone **Input-Convex Neural Networks (ICNNs)**; arbitrary real weights ($M = M^+ - M^-$) form **Difference-of-Convex (DC)** networks.

### 3. Hardware-Native Compilation & Ensemble UQ
Engine nodes (`ZNode`, `ZMatMulNode`, `ProductNode`, `ConstantNode`, `SignalNode`) compile via `jax.jit` into fused XLA kernels with compile-time linear bypass for single-branch ($K = 1$) nodes, horizontal operator fusion (`+`, `-`, `*`, `|`), and automatic `vmap` vectorization across state batches and `Ensemble` parameter distributions.

### 4. Belief Propagation, Legendre-Fenchel Duality, & Boundary Solvers
- **Reverse-Mode AD = Exact Belief Propagation:** With simplex branch probabilities $\mathbf{p} = \text{softmax}(\tau^{-1} M \mathbf{z} + \boldsymbol{g}) \in \Delta^{K-1}$, local marginals $\mathbf{n} = \nabla_{\mathbf{z}} z^+ = M^T \mathbf{p}$, and PSD susceptibility $\mathbf{H}_{\mathbf{z}} = \nabla^2_{\mathbf{z}} z^+ \succeq 0$, reverse-mode AD (`jax.grad` / `jax.vjp`) executes exact sum-product belief propagation in one backward sweep.
- **Convex Legendre-Fenchel Duality:** Provides both forward parametric transforms (`legendre_transform`) and damped pseudoinverse Newton-Raphson inversion (`legendre_solve`, `legendre_dual`) with $\mathcal{O}(1)$-memory implicit differentiation:
$$z^*(\mathbf{n}) = \sup_{\mathbf{z}} \left\{ \mathbf{z}^T \mathbf{n} - z^+(\mathbf{z}) \right\}$$
- **Isosurface & Crossover Solvers:** `gauge_fix` projects potentials onto the null-potential isosurface $z^+(\mathbf{z}) = 0$ in a single pass along shift-equivariant directions, and `extract_decision_boundary` locates pairwise branch crossovers from uncollapsed potentials $\{z_i^+ + \tau g_i\}$ via `jax.lax.top_k` with implicit sensitivities.

---

## Engine Usage Example

At the `zgraph` level, models are built using purely mathematical node abstractions:

```python
import jax.numpy as jnp
import zgraph as zg

# 1. Define incidence matrix M (K branches x N child factors)
M_matrix = jnp.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.5, 0.5]
])

# 2. Define child subgraphs (Constant / Signal nodes)
subgraphs = [
    zg.ConstantNode(init_val=1.5),
    zg.SignalNode(0)
]

# 3. Instantiate the ZNode engine with scale tau and optional log-degeneracy g
znode = zg.ZNode(M_matrix=M_matrix, subgraph_list=subgraphs, tau=1.0)

# 4. Pure tensor input evaluation
signals = jnp.array([0.2, 1.0])  # Unlabeled numerical input tensor
potential = znode(signals)       # Fully differentiable scalar tensor output
```

## Interactive Examples

Explore interactive Jupyter notebooks running natively on Google Colab:

- **2D Binary Equilibrium** (`binary.ipynb`): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/themintlab/ZGraph/blob/main/examples/binary.ipynb)
- **3D Extruded Dimension** (`Extruded_dimension.ipynb`): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/themintlab/ZGraph/blob/main/examples/Extruded_dimension.ipynb)
- **Stoichiometric Equilibrium** (`stoichiometric.ipynb`): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/themintlab/ZGraph/blob/main/examples/stoichiometric.ipynb)
- **Simple Phase Diagram** (`phase_diagram.ipynb`): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/themintlab/ZGraph/blob/main/examples/phase_diagram.ipynb)

---

## Application Layer

`zgraph` supports hierarchical factor-graph modeling across diverse domains, including deep branching transport (adjoint importance and criticality on $z^+(\mathbf{z}) = 0$) and multiscale statistical mechanics (cluster and super-cell partition networks).

For thermodynamic potential evaluation, CALPHAD materials modeling, phase diagram construction, and user-facing APIs built on `zgraph`, see the [**ThermoGraph**](https://github.com/themintlab/ThermoGraph) repository.
