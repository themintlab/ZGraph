import equinox as eqx
import jax
import jax.numpy as jnp

from zgraph import SignalNodes, ZNode
from zgraph.transforms import legendre_transform


def _build_binary_graph():
    r = 8.314
    t, mu1, mu2 = SignalNodes(0, 1, 2)

    rt = ZNode([[r]], [t])
    mu1a = ZNode([2, -1], [rt, mu1])
    mu2a = ZNode([-1], [mu2])
    mu1b = ZNode([-1], [mu1])
    mu2b = ZNode([1, -1], [rt, mu2])

    phase_a = ZNode(jnp.eye(2), [mu1a, mu2a], tau=rt)
    phase_b = ZNode(jnp.eye(2), [mu1b, mu2b], tau=rt)
    system = ZNode(jnp.eye(2), [phase_a, phase_b], tau=0)
    return phase_a, phase_b, system


def test_graph_and_legendre_paths_compile_and_execute():
    phase_a, phase_b, system = _build_binary_graph()

    t_val = jnp.array(298.15)
    mu1_vals = jnp.linspace(-10.0, 10.0, num=32)
    mu2_vals = -mu1_vals
    t_flat = jnp.broadcast_to(t_val, mu1_vals.shape)
    input_tensor = jnp.stack([t_flat, mu1_vals, mu2_vals], axis=-1)

    phase_a_fn, phase_b_fn, system_fn = [
        eqx.filter_jit(f) for f in (phase_a, phase_b, system)
    ]
    base_out = system_fn(input_tensor)

    legendre_modules = legendre_transform([phase_a, phase_b, system], [1, 2])
    leg_a_fn, leg_b_fn, leg_s_fn = [eqx.filter_jit(f) for f in legendre_modules]
    psi, x_dual = leg_s_fn(input_tensor)

    assert base_out.shape == (32,)
    assert psi.shape == (32,)
    assert x_dual.shape == (32, 3)
    assert jnp.isfinite(base_out).all()
    assert jnp.isfinite(psi).all()
    assert jnp.isfinite(x_dual).all()
