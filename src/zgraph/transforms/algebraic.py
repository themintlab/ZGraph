import jax
import jax.numpy as jnp
import equinox as eqx
from jax import value_and_grad
from jax.tree_util import tree_map
from typing import List, Optional, Sequence, Tuple, Any, TypeVar
from zgraph.core.base import BaseNode
from zgraph.solvers.dual import legendre_solve
import zgraph.transforms.functional as F

T = TypeVar("T")


class LegendreTransform(BaseNode):
    """
    Parametric Legendre-Fenchel Transform Node.
    Evaluates a base module at primal coordinates x, returning the convex conjugate
    z^* = x_I^T grad_{x_I} z^+(x) - z^+(x) (equal to -tau * dz^+/dtau for a full
    transform by degree-1 perspective homogeneity) and the dual coordinate vector
    (with grad_{x_I} z^+(x) placed at transform_indices).

    Inherits from BaseNode for automatic multi-dimensional batching and Ensemble UQ.
    """
    base_model: eqx.Module
    idx: jax.Array

    def __init__(self, base_model: eqx.Module, transform_indices: Sequence[int]):
        self.base_model = base_model
        self.idx = jnp.atleast_1d(jnp.asarray(transform_indices, dtype=jnp.int32))

    def evaluate(self, x: jax.Array) -> Tuple[jax.Array, jax.Array]:
        phi, grad = value_and_grad(self.base_model)(x)
        psi, dual_coords = F.apply_legendre(x, phi, grad, self.idx)
        return psi, dual_coords


class LegendreDualNode(BaseNode):
    """
    Inverse Legendre-Fenchel Dual Node.
    Takes prescribed target marginals n at transform_indices (and conditioning
    signals elsewhere), solves grad_z z^+(z^*) = n via damped pseudoinverse
    Newton-Raphson iteration, and returns the scalar conjugate potential z^*(n).
    """
    base_model: eqx.Module
    idx: jax.Array
    anchor_gauge: bool = eqx.field(static=True, default=False)
    rtol_pinv: float = eqx.field(static=True, default=1e-6)
    tol: float = eqx.field(static=True, default=1e-7)
    max_iter: int = eqx.field(static=True, default=64)

    def __init__(
        self,
        base_model: eqx.Module,
        transform_indices: Sequence[int],
        *,
        anchor_gauge: bool = False,
        rtol_pinv: float = 1e-6,
        tol: float = 1e-7,
        max_iter: int = 64,
    ):
        self.base_model = base_model
        self.idx = jnp.atleast_1d(jnp.asarray(transform_indices, dtype=jnp.int32))
        self.anchor_gauge = anchor_gauge
        self.rtol_pinv = rtol_pinv
        self.tol = tol
        self.max_iter = max_iter

    def dual_state(self, n_state: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Returns (z^*(n), dual_x) where dual_x[idx] = z^*(n)."""
        return legendre_solve(
            self.base_model,
            n_state,
            self.idx,
            anchor_gauge=self.anchor_gauge,
            rtol_pinv=self.rtol_pinv,
            tol=self.tol,
            max_iter=self.max_iter,
        )

    def evaluate(self, n_state: jax.Array) -> jax.Array:
        z_conj, _ = self.dual_state(n_state)
        return z_conj


def legendre_transform(modules: Any, transform_indices: List[int]) -> Any:
    """Maps parametric LegendreTransform over an arbitrary PyTree of modules."""
    return tree_map(
        lambda m: LegendreTransform(m, transform_indices),
        modules,
        is_leaf=lambda x: isinstance(x, eqx.Module),
    )


def legendre_dual(modules: Any, transform_indices: Sequence[int], **solver_kwargs) -> Any:
    """Maps inverse LegendreDualNode over an arbitrary PyTree of modules."""
    return tree_map(
        lambda m: LegendreDualNode(m, transform_indices, **solver_kwargs),
        modules,
        is_leaf=lambda x: isinstance(x, eqx.Module),
    )
