import jax
import jax.numpy as jnp
import pytest

import zgraph as zg
from zgraph import (
    BaseNode,
    ConstantNode,
    LegendreDualNode,
    LegendreTransform,
    SignalNode,
    SignalNodes,
    TropicalZeroNode,
    ZNode,
    gauge_fix,
    legendre_dual,
    legendre_solve,
    legendre_transform,
)


def test_log_degeneracy_g_parameter():
    z0, z1 = SignalNodes(0, 1)
    M = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    g = jnp.array([jnp.log(2.0), jnp.log(3.0)])
    tau = 2.5

    node = ZNode(M, [z0, z1], tau=tau, g=g)
    signals = jnp.array([1.2, -0.4])

    expected_branch_potentials = M @ signals + tau * g
    assert jnp.allclose(node.branch_potentials(signals), expected_branch_potentials, atol=1e-6)
    assert jnp.allclose(node.logits(signals), expected_branch_potentials, atol=1e-6)

    expected_probs = jax.nn.softmax((M @ signals) / tau + g)
    assert jnp.allclose(node.probabilities(signals), expected_probs, atol=1e-6)

    expected_val = tau * jax.scipy.special.logsumexp((M @ signals) / tau + g)
    assert jnp.allclose(node(signals), expected_val, atol=1e-6)

    # Check gradient wrt g: d z^+ / d g = tau * p
    def eval_with_g(g_vec):
        n = ZNode(M, [z0, z1], tau=tau, g=g_vec)
        return n(signals)

    grad_g = jax.grad(eval_with_g)(g)
    assert jnp.allclose(grad_g, tau * expected_probs, atol=1e-6)


def test_zero_scale_limit_value_and_derivatives():
    z0, z1 = SignalNodes(0, 1)
    tau_sig = SignalNode(2)
    M = jnp.array([[1.0, 2.0], [3.0, -1.0]])
    g = jnp.array([0.7, -1.3])

    node = ZNode(M, [z0, z1], tau=tau_sig, g=g)

    # For z = [1.0, 2.0]: branch 0 is 1 + 4 = 5.0; branch 1 is 3 - 2 = 1.0.
    signals_at_zero = jnp.array([1.0, 2.0, 0.0])

    # 1. Exact value at tau = 0 is max_i (M z)_i = 5.0
    val = node(signals_at_zero)
    assert jnp.allclose(val, 5.0, atol=1e-6)

    # 2. Reverse-mode gradient wrt (z0, z1, tau) at tau = 0:
    # grad_z = M_{0, :} = [1.0, 2.0], d z^+ / d tau = g_0 = 0.7
    grad_all = jax.grad(node)(signals_at_zero)
    assert jnp.allclose(grad_all[:2], jnp.array([1.0, 2.0]), atol=1e-6)
    assert jnp.allclose(grad_all[2], g[0], atol=1e-6)

    # 3. Forward-mode JVP at tau = 0
    _, jvp_tau = jax.jvp(node, (signals_at_zero,), (jnp.array([0.0, 0.0, 1.0]),))
    assert jnp.allclose(jvp_tau, g[0], atol=1e-6)

    # 4. Second derivative wrt tau at tau = 0 stays finite (zero)
    hess_all = jax.hessian(node)(signals_at_zero)
    assert jnp.isfinite(hess_all).all()
    assert jnp.allclose(hess_all[2, 2], 0.0, atol=1e-5)


def test_euler_identity_and_hessian_properties():
    z0, z1 = SignalNodes(0, 1)
    tau_sig = SignalNode(2)
    M = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    g = jnp.array([0.2, -0.5, 0.1])

    node = ZNode(M, [z0, z1], tau=tau_sig, g=g)
    signals = jnp.array([0.8, -0.3, 1.5])

    val = node(signals)
    grads = jax.grad(node)(signals)
    n_vec = grads[:2]
    d_dtau = grads[2]

    # Euler's homogeneous function theorem: z^+ = z^T n + tau * (dz^+ / dtau)
    euler_rhs = jnp.dot(signals[:2], n_vec) + signals[2] * d_dtau
    assert jnp.allclose(val, euler_rhs, atol=1e-5)

    # Hessian H_z is positive semi-definite
    hess = jax.hessian(node)(signals)
    H_z = hess[:2, :2]
    eigvals = jnp.linalg.eigvalsh(H_z)
    assert jnp.all(eigvals >= -1e-6)


def test_tropical_zero_node_is_true_identity():
    z0 = SignalNode(0)
    zero = TropicalZeroNode()
    node = ZNode(jnp.eye(2), [z0, zero], tau=1.5)

    x = jnp.array([3.2])
    # LSE([3.2, TROPICAL_ZERO_BARRIER]) == 3.2 regardless of tau
    assert jnp.allclose(node(x), 3.2, atol=1e-6)
    assert jnp.allclose(node.probabilities(x), jnp.array([1.0, 0.0]), atol=1e-6)
    assert jnp.allclose(jax.grad(node)(x), jnp.array([1.0]), atol=1e-6)


def test_or_operator_preserves_non_unit_tau():
    z0, z1 = SignalNodes(0, 1)
    phase_a = ZNode(jnp.eye(2), [z0, z1], tau=2.0)
    phase_b = ZNode(jnp.eye(2), [z0, z1], tau=3.0)

    combined = phase_a | phase_b
    signals = jnp.array([1.0, -0.5])

    expected = jax.scipy.special.logsumexp(
        jnp.array([phase_a(signals), phase_b(signals)])
    )
    assert jnp.allclose(combined(signals), expected, atol=1e-6)


def test_gauge_fix_handles_arbitrary_directional_gain():
    z0, z1 = SignalNodes(0, 1)
    # Negated chemical potential convention: z^+ = -mu_A => c_u = -1
    node_neg = ZNode([[-1.0, 0.0]], [z0, z1], tau=1.0)
    x_init = jnp.array([2.5, -1.0])
    x_proj = gauge_fix(node_neg, x_init, shift_indices=0)
    assert jnp.allclose(node_neg(x_proj), 0.0, atol=1e-6)

    # Explicit direction vector u with custom target_val
    node_weighted = ZNode([[2.0, -3.0]], [z0, z1], tau=1.0)
    u = jnp.array([0.5, 1.0])
    x_proj_u = gauge_fix(node_weighted, x_init, direction=u, target_val=1.25)
    assert jnp.allclose(node_weighted(x_proj_u), 1.25, atol=1e-6)


def test_legendre_transform_convex_fenchel_sign_and_direct_batching():
    t, mu1, mu2 = SignalNodes(0, 1, 2)
    rt = ZNode([[1.0]], [t])
    phase = ZNode(
        jnp.array([[1.0, -1.0, 0.0], [0.0, 0.0, -1.0]]),
        [rt, mu1, mu2],
        tau=rt,
    )

    lt = legendre_transform(phase, [1, 2])
    assert isinstance(lt, BaseNode)

    pt = jnp.array([2.0, -1.0, 0.5])
    z_star, x_dual = lt(pt)

    # Verify x_dual[1:] == grad_{mu1, mu2} z^+ and x_dual[0] == T
    full_grad = jax.grad(phase)(pt)
    assert jnp.allclose(x_dual[0], pt[0], atol=1e-6)
    assert jnp.allclose(x_dual[1:], full_grad[1:], atol=1e-6)

    # Verify convex Fenchel sign: z^* = z^T n - z^+(z)
    expected_z_star = jnp.dot(pt[1:], x_dual[1:]) - phase(pt)
    assert jnp.allclose(z_star, expected_z_star, atol=1e-6)

    # Verify direct batching on LegendreTransform without graph_to_function
    batch_pts = jnp.stack([pt, pt * 1.5], axis=0)
    batch_z_star, batch_x_dual = lt(batch_pts)
    assert batch_z_star.shape == (2,)
    assert batch_x_dual.shape == (2, 3)


def test_legendre_solve_and_dual_node_with_implicit_gradients():
    # Binary ideal solution with chemical potentials z = [T, mu1, mu2]
    # where z^+ = T * LSE([ (mu1 - g1)/T, (mu2 - g2)/T ])
    t, mu1, mu2 = SignalNodes(0, 1, 2)
    b1 = ZNode([-0.5, 1.0], [t, mu1])
    b2 = ZNode([0.3, 1.0], [t, mu2])
    phase = ZNode(jnp.eye(2), [b1, b2], tau=t)

    dual_node = legendre_dual(
        phase,
        transform_indices=[1, 2],
        anchor_gauge=True,
        tol=1e-6,
    )

    T_val = 2.0
    x1, x2 = 0.35, 0.65
    dual_input = jnp.array([T_val, x1, x2])

    z_star, z_opt = dual_node.dual_state(dual_input)
    assert jnp.allclose(dual_node(dual_input), z_star, atol=1e-6)

    # 1. At z_opt, gradient wrt [mu1, mu2] matches target [x1, x2]
    grad_at_opt = jax.grad(phase)(z_opt)[1:]
    assert jnp.allclose(grad_at_opt, jnp.array([x1, x2]), atol=1e-5)

    # 2. Gauge is anchored: z^+(z_opt) == 0
    assert jnp.allclose(phase(z_opt), 0.0, atol=1e-5)

    # 3. Fenchel value matches z^T n - z^+(z_opt)
    expected_z_star = jnp.dot(z_opt[1:], jnp.array([x1, x2])) - phase(z_opt)
    assert jnp.allclose(z_star, expected_z_star, atol=1e-5)

    # 4. Implicit gradient wrt dual_input = [T, n1, n2] matches finite differences
    analytic_grad = jax.grad(dual_node)(dual_input)
    eps = 1e-3
    fd_grad = jnp.zeros_like(dual_input)
    for i in range(3):
        e = jnp.zeros_like(dual_input).at[i].set(eps)
        fd_i = (dual_node(dual_input + e) - dual_node(dual_input - e)) / (2 * eps)
        fd_grad = fd_grad.at[i].set(fd_i)

    assert jnp.allclose(analytic_grad, fd_grad, atol=1e-3)
    # And d z^* / d n == mu_opt (since z^+(z_opt) == 0)
    assert jnp.allclose(analytic_grad[1:], z_opt[1:], atol=1e-4)


def test_save_and_load_roundtrip(tmp_path):
    z0, z1 = SignalNodes(0, 1)
    node = ZNode(
        jnp.array([[1.0, 0.5], [-0.5, 2.0]]),
        [z0, z1],
        tau=1.25,
        g=jnp.array([0.1, -0.2]),
    )
    archive = tmp_path / "model.zg"
    zg.save(node, str(archive))
    loaded = zg.load(str(archive))

    pt = jnp.array([0.7, -1.1])
    assert jnp.allclose(node(pt), loaded(pt), atol=1e-6)
