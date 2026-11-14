import jax
import jax.numpy as jnp
import equinox as eqx
from typing import List, Union, Optional
from . import functional as F
from .base import BaseNode
from .leaf_nodes import ConstantNode
from .constants import MIN_SAFE_TAU


class ZNode(BaseNode):
    """
    Fused Soft-Tropical Sum-Product Node.
    Computes the fused soft-tropical operator:
        z^+(z) = tau * LSE(tau^{-1} M z + g)
    combining linear incidence projection (M z) with a degeneracy-shifted,
    scale-parameterized LogSumExp reduction. For M >= 0, composition forms a
    monotone Input-Convex Neural Network (ICNN) whose reverse-mode AD executes
    exact sum-product belief propagation; for general real M = M^+ - M^-, it
    forms a Difference-of-Convex (DC) network.
    """
    M: jax.Array
    tau: eqx.Module
    g: Optional[jax.Array]
    subgraphs: List[eqx.Module]

    def __init__(
        self,
        M_matrix: Optional[Union[jax.Array, List[List[float]], List[float]]] = None,
        subgraph_list: Optional[List[eqx.Module]] = None,
        tau: Optional[Union[eqx.Module, float, int, jax.Array]] = None,
        g: Optional[Union[jax.Array, List[float]]] = None,
    ):
        if subgraph_list is None:
            subgraph_list = []

        if M_matrix is None:
            M_matrix_tensor = jnp.eye(len(subgraph_list), dtype=float)
        elif isinstance(M_matrix, jax.Array) and jnp.issubdtype(M_matrix.dtype, jnp.floating):
            M_matrix_tensor = M_matrix
        else:
            M_matrix_tensor = jnp.asarray(M_matrix, dtype=float)

        if M_matrix_tensor.ndim == 1:
            M_matrix_tensor = jnp.expand_dims(M_matrix_tensor, 0)

        if M_matrix_tensor.ndim != 2:
            raise ValueError(f"M_matrix must be a 2D tensor, got {M_matrix_tensor.ndim}D.")

        num_factors = M_matrix_tensor.shape[1]
        if num_factors != len(subgraph_list):
            raise ValueError(
                f"Dimension mismatch: M_matrix expects {num_factors} factors (columns), "
                f"but received {len(subgraph_list)} subgraphs."
            )

        self.M = M_matrix_tensor

        if tau is None:
            self.tau = ConstantNode(1.0)
        elif isinstance(tau, (int, float, jax.Array)):
            self.tau = ConstantNode(tau)
        elif isinstance(tau, eqx.Module):
            self.tau = tau
        else:
            raise TypeError("tau must be an eqx.Module, a numeric value (int/float), or a scalar jax.Array.")

        if g is None:
            self.g = None
        else:
            if isinstance(g, jax.Array) and jnp.issubdtype(g.dtype, jnp.floating):
                g_tensor = g
            else:
                g_tensor = jnp.asarray(g, dtype=float)
            if g_tensor.ndim != 1 or g_tensor.shape[0] != M_matrix_tensor.shape[0]:
                raise ValueError(
                    f"g must be a 1D vector of length {M_matrix_tensor.shape[0]} matching M_matrix rows, "
                    f"got shape {g_tensor.shape}."
                )
            self.g = g_tensor

        self.subgraphs = list(subgraph_list)

    def branch_potentials(self, signals: jax.Array) -> jax.Array:
        """
        Evaluates the uncollapsed degeneracy-shifted branch potentials:
            zeta_i = (M z)_i + tau * g_i
        """
        z = jnp.stack([subgraph(signals) for subgraph in self.subgraphs], axis=-1)
        branch_z = jnp.matmul(self.M, z)
        if self.g is not None:
            tau_val = self.tau(signals)
            branch_z = branch_z + tau_val * self.g
        return branch_z

    def logits(self, signals: jax.Array) -> jax.Array:
        """
        Alias for branch_potentials(signals), returning uncollapsed branch potentials:
            zeta_i = (M z)_i + tau * g_i
        """
        return self.branch_potentials(signals)

    def probabilities(self, signals: jax.Array) -> jax.Array:
        """
        Evaluates the branch softmax probability vector p in Delta^{K-1}:
            p = softmax(tau^{-1} M z + g)
        """
        branch_z = self.branch_potentials(signals)
        tau_val = jnp.maximum(self.tau(signals), MIN_SAFE_TAU)
        z_max = jnp.max(branch_z, axis=-1, keepdims=True)
        safe_z_max = jnp.where(jnp.isneginf(z_max), 0.0, z_max)
        return jax.nn.softmax((branch_z - safe_z_max) / tau_val, axis=-1)

    def evaluate(self, local_signals: jax.Array) -> jax.Array:
        branch_z = self.branch_potentials(local_signals)
        tau_val = self.tau(local_signals)
        return F.marginalize(branch_z, tau_val)


class ZMatMulNode(BaseNode):
    """
    Soft-Tropical Matrix Multiplication:
        C_{ij} = ⊕_{tau, k} (A_{ik} ⊗ B_{kj}) = tau * LSE_k((A_{ik} + B_{kj}) / tau)
    Acts as a routing or message-passing layer between sets of states.
    """
    node_A: eqx.Module
    node_B: eqx.Module
    tau: eqx.Module

    def __init__(
        self,
        node_A: eqx.Module,
        node_B: eqx.Module,
        tau: Optional[Union[eqx.Module, float, int, jax.Array]] = None,
    ):
        if not isinstance(node_A, eqx.Module) or not isinstance(node_B, eqx.Module):
            raise TypeError("Both node_A and node_B must be eqx.Module instances.")
        self.node_A = node_A
        self.node_B = node_B

        if tau is None:
            self.tau = ConstantNode(1.0)
        elif isinstance(tau, (int, float, jax.Array)):
            self.tau = ConstantNode(tau)
        elif isinstance(tau, eqx.Module):
            self.tau = tau
        else:
            raise TypeError("tau must be an eqx.Module, a numeric value (int/float), or a scalar jax.Array.")

    def evaluate(self, local_signals: jax.Array) -> jax.Array:
        A = self.node_A(local_signals)
        B = self.node_B(local_signals)

        A_expanded = jnp.expand_dims(A, axis=-1)
        B_expanded = jnp.expand_dims(B, axis=-3)

        summed = A_expanded + B_expanded

        # Move contraction index k to last axis for marginalize
        summed_k_last = jnp.moveaxis(summed, -2, -1)

        tau_val = self.tau(local_signals)
        return F.marginalize(summed_k_last, tau_val)
