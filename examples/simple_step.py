from __future__ import annotations

import jax
import jax.numpy as jnp

from crazyflie_mjx_sim import CrazyflieConfig, make_env


env = make_env("single", CrazyflieConfig(scenario="single"))
obs, state = env.reset(jax.random.PRNGKey(0))
action = jnp.zeros((env.num_agents, env.action_size))
next_obs, next_state, reward, done, info = env.step_env(jax.random.PRNGKey(1), state, action)

print("obs shape:", obs.shape)
print("next obs shape:", next_obs.shape)
print("reward:", reward)
print("done:", done)
print("contact:", info["in_contact"])
