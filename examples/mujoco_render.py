from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
from PIL import Image

from crazyflie_mjx_sim import CrazyflieConfig, make_env


env = make_env("single", CrazyflieConfig(scenario="single"))
_, state = env.reset(jax.random.PRNGKey(0))
action = jnp.zeros((env.num_agents, env.action_size))
for _ in range(20):
    _, state = env.step_rollout(jax.random.PRNGKey(1), state, action)

frame = env.render(state.data, camera="overview")
env.close()
out = Path("artifacts/example_mujoco_frame.png")
out.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(frame).save(out)
print(out)
