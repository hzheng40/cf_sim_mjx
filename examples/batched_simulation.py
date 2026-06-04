from __future__ import annotations

import jax
import jax.numpy as jnp

from crazyflie_mjx_sim import CrazyflieConfig, make_env


env = make_env(
    "multi_reach_avoid",
    CrazyflieConfig(scenario="multi_reach_avoid", num_agents=4, num_goals=4, num_obstacles=2),
)
batch_size = 32
horizon = 100
keys = jax.random.split(jax.random.PRNGKey(0), batch_size)

reset_batch = jax.jit(jax.vmap(env.reset))
step_batch = jax.jit(jax.vmap(env.step_rollout, in_axes=(0, 0, 0)))
obs, state = reset_batch(keys)
actions = jnp.zeros((batch_size, env.num_agents, env.action_size))


def step(carry, t):
    state = carry
    step_keys = jax.random.split(jax.random.PRNGKey(t), batch_size)
    obs, state = step_batch(step_keys, state, actions)
    return state, obs


state, obs_trace = jax.lax.scan(step, state, jnp.arange(horizon))
jax.block_until_ready(obs_trace)
print("batched obs trace:", obs_trace.shape)
