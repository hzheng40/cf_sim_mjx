"""Brax PPO hover example. Run with uv run --extra cuda13 --extra train."""
from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.base import Env, State
from brax.io import model
from brax.training.agents.ppo import networks, train as ppo

from crazyflie_mjx_sim import CrazyflieConfig, make_env


class HoverEnv(Env):
    """Brax interface with all motor/PID state inside pipeline_state.

    Brax owns episode limits, batching and automatic resets. The base simulator
    remains unchanged. Position observations are relative to the hover target.
    """

    def __init__(self, render=False):
        self.sim = make_env(config=CrazyflieConfig(
            control_mode="ctbr", use_motor_dynamics=True,
            disable_collisions=True, disable_visualization=not render,
            single_start_box=(-0.25, 0.25, -0.25, 0.25, 0.35, 0.85),
        ))
        self.target = jnp.array([0.0, 0.0, 0.6])

    @property
    def observation_size(self):
        return 12

    @property
    def action_size(self):
        return 4

    @property
    def backend(self):
        return "mjx"

    def _obs(self, sim_state):
        return self.sim.obs_fn(sim_state)[0].at[:3].add(-self.target)

    def reset(self, rng):
        _, sim_state = self.sim.reset(rng)
        zero = jnp.array(0.0)
        return State(pipeline_state=sim_state, obs=self._obs(sim_state),
                     reward=zero, done=zero,
                     metrics={"position_error": zero, "upright": zero})

    def step(self, state, action):
        _, sim_state = self.sim.step_rollout(
            jax.random.PRNGKey(0), state.pipeline_state, action[None, :])
        obs = self._obs(sim_state)
        distance = jnp.linalg.norm(obs[:3])
        upright = -obs[11]
        failed = (sim_state.data.qpos[2] < 0.08) | (distance > 2.0) | (upright < 0.2)
        reward = (jnp.exp(-4.0 * jnp.sum(obs[:3] ** 2))
                  + 0.2 * upright - 0.05 * jnp.sum(obs[3:6] ** 2)
                  - 0.005 * jnp.sum(obs[6:9] ** 2)
                  - 0.01 * jnp.sum(action ** 2) - failed.astype(jnp.float32))
        # Preserve Brax wrapper bookkeeping and EvalWrapper's reward metric.
        metrics = dict(state.metrics)
        metrics.update(position_error=distance, upright=upright)
        return state.replace(pipeline_state=sim_state, obs=obs, reward=reward,
                             done=failed.astype(jnp.float32), metrics=metrics)


NETWORK_FACTORY = functools.partial(
    networks.make_ppo_networks, policy_hidden_layer_sizes=(128, 128),
    value_hidden_layer_sizes=(256, 128), activation=jax.nn.elu)


def train(output="artifacts/hover_ppo", *, num_timesteps=2_000_000,
          num_envs=128, batch_size=128, episode_length=1000,
          num_evals=5, seed=32):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    history = []

    def progress(steps, metrics):
        row = {"steps": int(steps), **{k: float(v) for k, v in metrics.items()}}
        history.append(row)
        (output / "metrics.json").write_text(json.dumps(history, indent=2))
        print(f"steps={steps} reward={row.get('eval/episode_reward', float('nan')):.3f}", flush=True)

    make_inference_fn, params, _ = ppo.train(
        environment=HoverEnv(), num_timesteps=num_timesteps, num_envs=num_envs,
        batch_size=batch_size, num_minibatches=4, num_updates_per_batch=4,
        unroll_length=16, episode_length=episode_length, action_repeat=5,
        num_evals=num_evals, num_eval_envs=4, learning_rate=3e-4,
        max_devices_per_host=1,
        discounting=0.99, entropy_cost=0.001, normalize_observations=True,
        network_factory=NETWORK_FACTORY, seed=seed, progress_fn=progress,
    )
    model.save_params(str(output / "params.pkl"), params)
    return make_inference_fn, params, history


def load_policy(checkpoint):
    """Load only checkpoints you trust (Brax parameter files use pickle)."""
    from brax.training.acme import running_statistics
    network = NETWORK_FACTORY(12, 4, preprocess_observations_fn=running_statistics.normalize)
    return networks.make_inference_fn(network), model.load_params(str(checkpoint))


def render_rollout(make_inference_fn, params, output="artifacts/hover_ppo/rollout.mp4",
                   *, steps=1000, seed=123, render_every=20):
    """Deterministic evaluation, 5 physics steps/action, streamed video frames."""
    import mediapy as media
    import imageio_ffmpeg
    media.set_ffmpeg(imageio_ffmpeg.get_ffmpeg_exe())
    env = HoverEnv(render=True)
    policy = jax.jit(make_inference_fn(params, deterministic=True))
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    rng = jax.random.PRNGKey(seed)
    state = reset(rng)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    positions, rewards = [], []
    try:
        with media.VideoWriter(str(output), shape=(720, 1080),
                               fps=1 / (env.sim.cfg.timestep * render_every)) as video:
            for i in range(steps):
                if i % 5 == 0:
                    rng, key = jax.random.split(rng)
                    action, _ = policy(state.obs, key)
                state = step(state, action)
                positions.append(np.asarray(state.pipeline_state.data.qpos[:3]))
                rewards.append(float(state.reward))
                if i % render_every == 0:
                    video.add_image(env.sim.render(state.pipeline_state.data))
                if bool(state.done):
                    break
    finally:
        env.sim.close()
    np.savez(output.with_suffix(".npz"), positions=positions, rewards=rewards)
    print(f"Rendered {len(rewards)} steps to {output}; return={sum(rewards):.3f}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/hover_ppo")
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()
    print("JAX:", jax.default_backend(), jax.devices(), flush=True)
    if args.checkpoint:
        inference, params = load_policy(args.checkpoint)
    else:
        inference, params, _ = train(args.output,
            num_timesteps=1280 if args.smoke else args.steps,
            num_envs=4 if args.smoke else args.num_envs,
            batch_size=4 if args.smoke else 128,
            episode_length=40 if args.smoke else 1000,
            num_evals=2 if args.smoke else 5)
    if not args.no_render:
        render_rollout(inference, params, Path(args.output) / "rollout.mp4",
                       steps=40 if args.smoke else 1000)


if __name__ == "__main__":
    main()
