# ZGraph AI Directives

**ATTENTION AI CODING ASSISTANTS:**
Modify `zgraph` strictly following these rules (see `src/zgraph/ARCHITECTURE.md`):

1. **Soft-Tropical Sum-Product Graph First**: `zgraph` is a domain-agnostic soft-tropical factor-graph engine computing $z^+(\mathbf{z}) = \tau \text{LSE}(\tau^{-1} M \mathbf{z} + \boldsymbol{g})$.
2. **Architectural Division (Core vs Transforms vs Solvers)**:
   - **`core/`**: Math primitives (`ZNode`, `ZMatMulNode`, `marginalize`). Defines the forward pass and topology.
   - **`transforms/`**: Graph-to-Graph operators (`legendre_transform`, `legendre_dual`). They reshape the topology or coordinates and return evaluatable `BaseNode` modules.
   - **`solvers/`**: Execution routines (`extract_decision_boundary`, `gauge_fix`, `legendre_solve`). They take a **Callable Function** + **Input Domain Tensors**, execute numerical or analytic projections/inversions, and return solution tensors.
3. **Domain Agnosticism**: Keep `zgraph` strictly mathematical. Domain-specific physical logic (like mapping gradients to CALPHAD composition tie-lines) must live in outer domain packages like `thermograph`.
4. **Pure Math**: No Python objects (strings, lists, dicts) or value-dependent Python control flow in `BaseNode.evaluate()`.
5. **Immutable Nodes**: Routing nodes are structurally immutable. Instantiate new nodes if topology changes.
6. **Static vs Dynamic**: Equinox handles static fields natively (`eqx.field(static=True)`).
7. **Arrays Only**: All `evaluate()` / `__call__()` I/O must be `jax.Array` (or tuples of `jax.Array`).
8. **Vectorize**: No Python loops over batch or branch dimensions inside `evaluate()`. Use native JAX ops.
9. **Strict Typing**: Use explicit `typing` hints.
10. **Functor Pattern**: Subclasses implement unbatched `evaluate(self, signals)`; `BaseNode.__call__` automatically handles batching (`jax.vmap`) and `Ensemble` vectorization (`eqx.filter_vmap`). Apply `jax.jit` at the top level after the graph is built.
11. **PyTrees**: Use `jax.tree_util.tree_map`.
12. **Style**: Be concise and token-efficient. Write efficient, clear code. Prefer Plotly for visualizations. Adhere to PEP 8 naming conventions.
