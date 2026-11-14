import jax
import jax.numpy as jnp
from functools import partial
from typing import Any, Callable, Optional, Sequence, Tuple


def _eval_subspace(
    model: Callable[[jax.Array], jax.Array],
    base_x: jax.Array,
    idx: jax.Array,
    z_sub: jax.Array,
) -> jax.Array:
    x = base_x.at[idx].set(z_sub, unique_indices=True)
    return jnp.squeeze(model(x))


@partial(jax.custom_jvp, nondiff_argnums=(3, 4, 5, 6))
def _solve_legendre_single(
    model: Any,
    target_x: jax.Array,
    idx: jax.Array,
    anchor_gauge: bool,
    rtol_pinv: float,
    tol: float,
    max_iter: int,
    z_init_sub: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """
    Single-point damped pseudoinverse Newton-Raphson solver for the convex
    Legendre-Fenchel conjugate z^*(n) = sup_z { z^T n - z^+(z) }.

    Decorated with @jax.custom_jvp so Newton iterations run off the AD tape at
    O(1) memory cost; JAX automatically transposes the linear JVP rule into a
    custom VJP for reverse-mode AD while supporting forward-mode and higher-order AD.
    """
    n_target = target_x[idx]
    base_x = jax.lax.stop_gradient(target_x)
    n_stop = jax.lax.stop_gradient(n_target)
    model_stop = jax.lax.stop_gradient(model)

    def f_primal(z):
        return _eval_subspace(model_stop, base_x, idx, z)

    val_and_grad_fn = jax.value_and_grad(f_primal)
    hess_fn = jax.hessian(f_primal)

    def dual_obj(z):
        return jnp.dot(z, n_stop) - f_primal(z)

    z0 = jax.lax.stop_gradient(z_init_sub)
    phi0, g0 = val_and_grad_fn(z0)
    H0 = hess_fn(z0)
    dz0 = -jnp.matmul(jnp.linalg.pinv(H0, rtol=rtol_pinv), g0 - n_stop)
    decr0 = jnp.dot(n_stop - g0, dz0)

    init_state = (jnp.asarray(0, dtype=jnp.int32), z0, phi0, g0, dz0, decr0)

    def cond_fn(state):
        step, _, _, _, dz, decr = state
        step_norm = jnp.max(jnp.abs(dz))
        not_converged = jnp.logical_or(decr > tol * tol, step_norm > tol)
        return jnp.logical_and(step < max_iter, not_converged)

    def body_fn(state):
        step, z, phi, g, dz, decr = state
        obj_curr = jnp.dot(z, n_stop) - phi
        armijo_slope = 1e-4 * jnp.maximum(decr, 0.0)

        def ls_cond(alpha):
            z_try = z + alpha * dz
            obj_try = dual_obj(z_try)
            return jnp.logical_and(alpha > 1e-6, obj_try < obj_curr + alpha * armijo_slope)

        def ls_body(alpha):
            return 0.5 * alpha

        alpha = jax.lax.while_loop(ls_cond, ls_body, jnp.asarray(1.0, dtype=z.dtype))
        z_next = z + alpha * dz
        phi_next, g_next = val_and_grad_fn(z_next)
        H_next = hess_fn(z_next)
        dz_next = -jnp.matmul(jnp.linalg.pinv(H_next, rtol=rtol_pinv), g_next - n_stop)
        decr_next = jnp.dot(n_stop - g_next, dz_next)
        return (step + 1, z_next, phi_next, g_next, dz_next, decr_next)

    _, z_star, _, _, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)

    if anchor_gauge:
        u = jnp.ones_like(z_star)
        phi_star, c_u = jax.jvp(f_primal, (z_star,), (u,))
        safe_c_u = jnp.where(jnp.abs(c_u) > 1e-12, c_u, 1.0)
        shift = jnp.where(jnp.abs(c_u) > 1e-12, -phi_star / safe_c_u, 0.0)
        z_star = z_star + shift * u

    phi_final = _eval_subspace(model, target_x, idx, z_star)
    z_conj = jnp.dot(z_star, n_target) - phi_final
    dual_x = target_x.at[idx].set(z_star, unique_indices=True)
    return z_conj, dual_x


@_solve_legendre_single.defjvp
def _solve_legendre_single_jvp(
    anchor_gauge: bool,
    rtol_pinv: float,
    tol: float,
    max_iter: int,
    primals,
    tangents,
):
    """
    Implicit Function Theorem JVP rule, automatically transposed by JAX into a
    VJP during reverse-mode AD (jax.grad / jax.vjp).
    """
    model, target_x, idx, z_init_sub = primals
    d_model, d_target_x, _, _ = tangents

    z_conj, dual_x = _solve_legendre_single(
        model, target_x, idx, anchor_gauge, rtol_pinv, tol, max_iter, z_init_sub
    )
    z_star = dual_x[idx]
    n_target = target_x[idx]
    d_n = d_target_x[idx]

    # Mask out d_target_x on transform_indices so d_x_cond only carries conditioning tangents
    d_x_cond = d_target_x.at[idx].set(jnp.zeros_like(d_n), unique_indices=True)

    def val_and_grad_wrt_z(m, tx):
        return jax.value_and_grad(lambda z: _eval_subspace(m, tx, idx, z))(z_star)

    (phi_val, g_z), (d_phi_theta, d_g_z_theta) = jax.jvp(
        val_and_grad_wrt_z,
        (model, target_x),
        (d_model, d_x_cond),
    )

    H_z = jax.hessian(lambda z: _eval_subspace(model, target_x, idx, z))(z_star)
    H_pinv = jnp.linalg.pinv(H_z, rtol=rtol_pinv)

    # Implicit Function Theorem: H_z dz^* + d_g_z_theta = dn => dz^* = H_z^\dagger (dn - d_g_z_theta)
    d_z_star = jnp.matmul(H_pinv, d_n - d_g_z_theta)

    if anchor_gauge:
        u = jnp.ones_like(z_star)
        c_u = jnp.dot(u, g_z)
        safe_c_u = jnp.where(jnp.abs(c_u) > 1e-12, c_u, 1.0)
        # Tangent orthogonality on z^+(z^*, theta) = 0: g_z^T dz^* + d_phi_theta = 0
        gauge_corr = jnp.where(
            jnp.abs(c_u) > 1e-12,
            -(jnp.dot(g_z, d_z_star) + d_phi_theta) / safe_c_u,
            0.0,
        )
        d_z_star = d_z_star + gauge_corr * u

    # Envelope theorem: dz^* = z^{*T} dn - d_phi_theta + (n - g_z)^T dz^*
    d_z_conj = jnp.dot(z_star, d_n) - d_phi_theta + jnp.dot(n_target - g_z, d_z_star)
    d_dual_x = d_x_cond.at[idx].set(d_z_star, unique_indices=True)

    return (z_conj, dual_x), (d_z_conj, d_dual_x)


def legendre_solve(
    model_fn: Callable[[jax.Array], jax.Array],
    target_state: jax.Array,
    transform_indices: Optional[Sequence[int]] = None,
    *,
    z_init: Optional[jax.Array] = None,
    anchor_gauge: bool = False,
    rtol_pinv: float = 1e-6,
    tol: float = 1e-7,
    max_iter: int = 64,
) -> Tuple[jax.Array, jax.Array]:
    r"""
    Differentiable Legendre-Fenchel solver.
    Inverts a soft-tropical node or DAG from target marginals n (at transform_indices
    of target_state) to conjugate potentials z^*(n) and evaluates the convex conjugate
    z^*(n) = z^{*T} n - z^+(z^*) (equal to -tau * dz^+/dtau for a full transform by
    degree-1 perspective homogeneity).

    Implicit differentiation via @jax.custom_jvp runs at O(1) memory cost and is
    automatically transposed by JAX into a VJP for reverse-mode AD (jax.grad / jax.vjp),
    while also supporting forward-mode AD (jax.jvp) and dual Hessians.

    Supports arbitrary leading batch dimensions on target_state.

    Args:
        model_fn: A BaseNode or callable mapping 1D coordinates x -> scalar z^+(x).
        target_state: Input tensor of shape (..., D) containing target marginals n
            at transform_indices and fixed conditioning coordinates elsewhere.
        transform_indices: Sequence of coordinate indices to invert. Defaults to all indices.
        z_init: Optional initial guess for z^* on transform_indices. Shape: (..., |I|) or (|I|,).
        anchor_gauge: If True, projects z^* onto the null-potential isosurface z^+(z^*) = 0
            in a single pass so z^*(n) reduces to z^{*T} n.
        rtol_pinv: Relative singular-value threshold for Moore-Penrose pseudoinverse H_z^\dagger.
        tol: Convergence tolerance on the Newton decrement.
        max_iter: Maximum damped Newton-Raphson iterations.

    Returns:
        Tuple[jax.Array, jax.Array]:
            - z_conj: Conjugate potential z^*(n) of shape (...,)
            - dual_state: State tensor of shape (..., D) with z^*(n) placed at transform_indices.
    """
    dim = target_state.shape[-1]
    if transform_indices is None:
        idx = jnp.arange(dim, dtype=jnp.int32)
    else:
        idx = jnp.atleast_1d(jnp.asarray(transform_indices, dtype=jnp.int32))

    if idx.size == 0:
        phi = model_fn(target_state)
        return -phi, target_state

    if z_init is None:
        z_init_sub = jnp.zeros(target_state.shape[:-1] + (idx.size,), dtype=target_state.dtype)
    else:
        z_init_arr = jnp.asarray(z_init, dtype=target_state.dtype)
        z_init_sub = jnp.broadcast_to(z_init_arr, target_state.shape[:-1] + (idx.size,))

    def _single(tx, z0):
        return _solve_legendre_single(
            model_fn, tx, idx, anchor_gauge, rtol_pinv, tol, max_iter, z0
        )

    solve_fn = _single
    num_batch_dims = target_state.ndim - 1
    for _ in range(num_batch_dims):
        solve_fn = jax.vmap(solve_fn, in_axes=(0, 0))

    return solve_fn(target_state, z_init_sub)
