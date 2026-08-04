from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from crazyflie_mjx_sim import CrazyflieConfig, generate_web_report, make_env


def test_single_step_cpu_smoke():
    env = make_env("single", CrazyflieConfig(scenario="single", horizon=20))
    obs, state = env.reset(jax.random.PRNGKey(0))
    action = jnp.zeros((env.num_agents, env.action_size))
    for i in range(10):
        obs, state, reward, done, info = env.step_env(jax.random.PRNGKey(i + 1), state, action)
    assert obs.shape == (1, 12)
    assert reward.shape == (1,)
    assert done.shape == ()
    assert "in_contact" in info
    assert np.isfinite(np.asarray(obs)).all()


def test_make_env_honors_config_scenario():
    env = make_env(config=CrazyflieConfig(scenario="multi_reach_avoid", num_agents=2, num_goals=2, num_obstacles=1))
    assert env.scenario == "multi_reach_avoid"
    assert env.num_agents == 2


def test_multi_reach_avoid_step_cpu_smoke():
    env = make_env(
        "multi_reach_avoid",
        CrazyflieConfig(scenario="multi_reach_avoid", num_agents=4, num_goals=4, num_obstacles=2, horizon=20),
    )
    obs, state = env.reset(jax.random.PRNGKey(2))
    action = jnp.zeros((env.num_agents, env.action_size))
    for i in range(10):
        obs, state, reward, done, info = env.step_env(jax.random.PRNGKey(i + 3), state, action)
    assert obs.shape == (4, env.obs_size)
    assert reward.shape == (4,)
    assert "goal_distance" in info
    assert "nearest_obstacle_clearance" in info
    assert np.isfinite(np.asarray(obs)).all()


def test_batched_reset_and_rollout():
    for scenario, cfg in [
        ("single", CrazyflieConfig(scenario="single", horizon=10)),
        ("multi_reach_avoid", CrazyflieConfig(scenario="multi_reach_avoid", num_agents=3, num_goals=3, num_obstacles=1, horizon=10)),
    ]:
        env = make_env(scenario, cfg)
        keys = jax.random.split(jax.random.PRNGKey(4), 4)
        reset_batch = jax.jit(jax.vmap(env.reset))
        step_batch = jax.jit(jax.vmap(env.step_rollout, in_axes=(0, 0, 0)))
        obs, state = reset_batch(keys)
        action = jnp.zeros((4, env.num_agents, env.action_size))
        obs, state = step_batch(keys, state, action)
        assert obs.shape == (4, env.num_agents, env.obs_size)
        assert np.isfinite(np.asarray(obs)).all()


def test_velocity_yaw_rate_control_requires_motor_dynamics():
    with pytest.raises(ValueError, match="requires use_motor_dynamics=True"):
        make_env(
            "single",
            CrazyflieConfig(
                scenario="single",
                control_mode="velocity_yaw_rate",
                use_motor_dynamics=False,
            ),
        )


def test_velocity_yaw_rate_control_maps_to_ctbr_and_batches():
    env = make_env(
        "single",
        CrazyflieConfig(
            scenario="single",
            control_mode="velocity_yaw_rate",
            disable_collisions=True,
            disable_visualization=True,
        ),
    )
    keys = jax.random.split(jax.random.PRNGKey(10), 2)
    obs, state = jax.jit(jax.vmap(env.reset))(keys)
    actions = jnp.asarray(
        [
            [[1.0, 0.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0, 1.0]],
        ]
    )
    obs, state, _, _, info = jax.jit(jax.vmap(env.step_env, in_axes=(0, 0, 0)))(
        keys,
        state,
        actions,
    )
    ctbr = np.asarray(info["ctbr_action"])
    assert obs.shape == (2, 1, env.obs_size)
    assert ctbr.shape == (2, 1, 4)
    assert ctbr[0, 0, 2] > 0.0
    assert ctbr[1, 0, 3] > 0.0
    assert env.metadata()["control"]["mode"] == "velocity_yaw_rate"


def test_velocity_yaw_rate_control_is_end_to_end_differentiable():
    env = make_env(
        "single",
        CrazyflieConfig(
            scenario="single",
            control_mode="velocity_yaw_rate",
            disable_collisions=True,
            disable_visualization=True,
        ),
    )
    key = jax.random.PRNGKey(11)
    _, state = env.reset(key)

    def next_qvel(action):
        _, next_state = env.step_rollout(key, state, action)
        return next_state.data.qvel

    jacobian = jax.jacfwd(next_qvel)(jnp.zeros((1, env.action_size)))
    jacobian_np = np.asarray(jacobian)
    assert jacobian_np.shape == (6, 1, 4)
    assert np.isfinite(jacobian_np).all()
    assert (np.linalg.norm(jacobian_np[:, 0, :], axis=0) > 1e-8).all()


def test_mujoco_render_rgb():
    env = make_env("single", CrazyflieConfig(scenario="single", horizon=5))
    try:
        _, state = env.reset(jax.random.PRNGKey(5))
        frame = env.render(state.data, camera="overview")
    finally:
        env.close()
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.shape[0] > 0 and frame.shape[1] > 0


def test_web_report_artifacts(tmp_path: Path):
    payload = {
        "name": "smoke",
        "dt": 0.01,
        "positions": np.zeros((3, 1, 3)),
        "quaternions": np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (3, 1, 1)),
        "scene": {"goals": [], "obstacles": []},
        "metrics": {"reward_mean": 0.0},
    }
    index = generate_web_report(tmp_path, None, [payload], {"scenario": "single"})
    assert index.exists()
    assert (index.parent / "report_data.js").exists()
    assert (index.parent / "assets" / "cf2" / "cf2_0.obj").exists()


def test_minimal_source_surface():
    root = Path(__file__).resolve().parents[1] / "src" / "crazyflie_mjx_sim"
    forbidden = ("nfsp", "svgd", "wandb", "stljax")
    for path in root.rglob("*.py"):
        text = path.read_text().lower()
        for term in forbidden:
            assert term not in text, f"{term} found in {path}"
