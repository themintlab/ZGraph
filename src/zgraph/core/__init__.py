from .base import BaseNode, Ensemble
from .znodes import (
    ZNode,
    ZMatMulNode,
)
from .utility_nodes import PiecewiseNode, StandardProductNode
from .leaf_nodes import ConstantNode, SignalNode, SignalNodes
from .functional import marginalize

import jax.numpy as jnp

# Convenience Alias for standard non-tropical multiplication
ProductNode = StandardProductNode


def AdditionNode(subgraph_list, weights=None):
    if weights is None:
        weights = jnp.ones(len(subgraph_list), dtype=float)
    elif not (isinstance(weights, jnp.ndarray) and jnp.issubdtype(weights.dtype, jnp.floating)):
        weights = jnp.asarray(weights, dtype=float)
    M = jnp.expand_dims(weights, 0)
    return ZNode(M_matrix=M, subgraph_list=subgraph_list)


def DivisionNode(numerator, denominator):
    return ZNode(
        M_matrix=jnp.asarray([[1.0, -1.0]], dtype=float),
        subgraph_list=[numerator, denominator],
    )


from .constants import TROPICAL_ZERO_BARRIER


def TropicalZeroNode(barrier_value=TROPICAL_ZERO_BARRIER):
    """
    Tropical Zero Element (additive identity for ⊕_tau).
    Represented numerically as ConstantNode(TROPICAL_ZERO_BARRIER) (-1e30 by default)
    rather than IEEE -inf to prevent 0.0 * (-inf) = NaN during incidence matrix
    multiplication (M @ z) and automatic differentiation.
    """
    return ConstantNode(barrier_value)


def TropicalIdentityNode():
    """Tropical Identity Element (for ⊗). Represented as constant 0.0."""
    return ConstantNode(0.0)


__all__ = [
    "BaseNode",
    "Ensemble",
    "ZNode",
    "ZMatMulNode",
    "ProductNode",
    "AdditionNode",
    "DivisionNode",
    "StandardProductNode",
    "TropicalZeroNode",
    "TropicalIdentityNode",
    "PiecewiseNode",
    "ConstantNode",
    "SignalNode",
    "SignalNodes",
    "marginalize",
]
