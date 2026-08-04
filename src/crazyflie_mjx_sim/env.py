from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from flax import struct
from mujoco import mjx

from .config import CrazyflieConfig, ScenarioName
from .mjcf import make_cf2_scene_xml
from .motor_dynamics import MotorDynamics, RandomizedDynamicsParams
from .sampling import sample_non_overlapping_squares
from .velocity_controller import VelocityYawRateController


@struct.dataclass
class EnvState:
    data: mjx.Data
    step: chex.Array
    current_motor_speeds: chex.Array
    prev_omega_err_integral: chex.Array
    prev_omega_meas: chex.Array
    prev_action: chex.Array
    dynamics_params: RandomizedDynamicsParams
    goal_positions: chex.Array
    obstacle_positions: chex.Array
    current_goal_bounds: chex.Array
    in_contact: chex.Array


class CrazyflieMJXEnv:
    """Array-first Crazyflie MJX simulator.

    Observations and actions are arrays with leading shape ``[num_agents]``.
    ``step_rollout`` is the fast simulation path. ``step_env`` adds lightweight
    reward/info/done outputs but still leaves reset ownership to the caller.
    """

    def __init__(self, scenario: ScenarioName | None = None, config: CrazyflieConfig | None = None):
        if config is None:
            cfg = CrazyflieConfig(scenario=scenario or "single")
        elif scenario is None:
            cfg = config
        else:
            cfg = replace(config, scenario=scenario)
        self.cfg = cfg.normalized()
        self.scenario = self.cfg.scenario
        self.num_agents = int(self.cfg.num_agents)
        self.num_goals = int(self.cfg.num_goals)
        self.num_obstacles = int(self.cfg.num_obstacles)
        if self.cfg.control_mode not in {"ctbr", "velocity_yaw_rate"}:
            raise ValueError(f"Unsupported control mode: {self.cfg.control_mode!r}")
        if self.cfg.control_mode == "velocity_yaw_rate" and not self.cfg.use_motor_dynamics:
            raise ValueError("velocity_yaw_rate control requires use_motor_dynamics=True")
        self.action_size = 4
        self.base_obs_size = 12
        self.obs_size = self.base_obs_size if self.scenario == "single" else (
            self.base_obs_size
            + self.num_goals * 3
            + self.num_obstacles * 3
            + (self.num_agents - 1) * 3
            + self.num_agents
        )

        remove_actuators = bool(self.cfg.use_motor_dynamics)
        xml = make_cf2_scene_xml(
            num_agents=self.num_agents,
            num_goals=self.num_goals,
            num_obstacles=self.num_obstacles,
            start_z=self.cfg.start_z,
            goal_z=self.cfg.goal_z,
            goal_width=self.cfg.goal_width,
            obs_width=self.cfg.obs_width,
            obs_height=self.cfg.obs_height,
            cf_col_radius=self.cfg.cf_col_radius,
            world_rect=self.cfg.world_rect,
            disable_visualization=self.cfg.disable_visualization,
            disable_collisions=self.cfg.disable_collisions,
            enable_world_bound_collisions=self.cfg.enable_world_bound_collisions,
            world_bound_collision_thickness=self.cfg.world_bound_collision_thickness,
            keep_only_contact_sensors=self.cfg.keep_only_contact_sensors,
            remove_actuators=remove_actuators,
            scene_name=f"crazyflie_{self.scenario}",
        )
        self.mj_model = mujoco.MjModel.from_xml_string(xml)
        self.mj_model.opt.timestep = self.cfg.timestep
        self.mj_model.opt.integrator = (
            mujoco.mjtIntegrator.mjINT_EULER
            if self.cfg.use_euler_integrator
            else mujoco.mjtIntegrator.mjINT_RK4
        )
        if self.cfg.solver == "cg":
            self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        elif self.cfg.solver == "newton":
            self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        else:
            raise ValueError(f"Unsupported solver: {self.cfg.solver!r}")
        self.mj_model.opt.iterations = self.cfg.iterations
        self.mj_model.opt.ls_iterations = self.cfg.ls_iterations
        self.mj_model.opt.tolerance = self.cfg.tolerance
        self.mj_model.opt.ls_tolerance = self.cfg.ls_tolerance

        self._mjx_impl = self.cfg.mjx_backend
        self.m = mjx.put_model(self.mj_model, impl=self._mjx_impl) if self._mjx_impl else mjx.put_model(self.mj_model)
        self._cf_body_ids = [
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, f"cf_{i}/body")
            for i in range(self.num_agents)
        ]
        if any(body_id < 0 for body_id in self._cf_body_ids):
            raise RuntimeError("Failed to locate every Crazyflie body in generated MJCF.")
        self.motor_dynamics = MotorDynamics(self.cfg, self.mj_model, self._cf_body_ids)
        self.velocity_controller = VelocityYawRateController(
            self.cfg,
            mass=self.motor_dynamics.mass,
            weight=self.motor_dynamics.weight,
        )
        self._initial_motor_speeds = self.cfg.hover_motor_speed * jnp.ones((self.num_agents, 4))
        self._zero_omega = jnp.zeros((self.num_agents, 3))
        self._zero_action = jnp.zeros((self.num_agents, 4))
        self._world_rect = jnp.asarray(self.cfg.world_rect, dtype=jnp.float32)
        self._single_start_box = jnp.asarray(self.cfg.single_start_box, dtype=jnp.float32)
        self._goal_half = jnp.asarray([self.cfg.goal_width / 2.0] * 3, dtype=jnp.float32)
        self._obs_half = jnp.asarray(
            [self.cfg.obs_width / 2.0, self.cfg.obs_width / 2.0, self.cfg.obs_height / 2.0],
            dtype=jnp.float32,
        )
        self._other_agent_indices = self._build_other_agent_indices()
        self._renderer = None

    def _build_other_agent_indices(self) -> jnp.ndarray:
        if self.num_agents <= 1:
            return jnp.zeros((self.num_agents, 0), dtype=jnp.int32)
        return jnp.asarray(
            [[j for j in range(self.num_agents) if j != i] for i in range(self.num_agents)],
            dtype=jnp.int32,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "num_agents": self.num_agents,
            "num_goals": self.num_goals,
            "num_obstacles": self.num_obstacles,
            "obs_size": self.obs_size,
            "action_size": self.action_size,
            "control": {
                "mode": self.cfg.control_mode,
                "command": (
                    "normalized_collective_thrust_and_body_rates"
                    if self.cfg.control_mode == "ctbr"
                    else "normalized_heading_velocity_and_yaw_rate"
                ),
                "velocity_scale": [
                    self.cfg.velocity_scale_xy,
                    self.cfg.velocity_scale_xy,
                    self.cfg.velocity_scale_z,
                ],
                "yaw_rate_scale": self.cfg.velocity_yaw_rate_scale,
            },
            "timestep": self.cfg.timestep,
            "horizon": self.cfg.horizon,
            "runtime": {
                "disable_visualization": self.cfg.disable_visualization,
                "disable_collisions": self.cfg.disable_collisions,
                "enable_world_bound_collisions": self.cfg.enable_world_bound_collisions,
                "mj_solver": self.cfg.solver,
                "mj_iterations": self.cfg.iterations,
                "mj_ls_iterations": self.cfg.ls_iterations,
                "mj_tolerance": self.cfg.tolerance,
                "mj_ls_tolerance": self.cfg.ls_tolerance,
                "mjx_backend": self.cfg.mjx_backend,
                "use_euler_integrator": self.cfg.use_euler_integrator,
            },
        }

    @partial(jax.jit, static_argnums=0)
    def _sample_layout(self, key: chex.PRNGKey) -> tuple[chex.Array, chex.Array, chex.Array]:
        if self.scenario == "single":
            pos_key = key
            low = self._single_start_box[jnp.asarray([0, 2, 4])]
            high = self._single_start_box[jnp.asarray([1, 3, 5])]
            start_positions = jax.random.uniform(pos_key, (1, 3), minval=low, maxval=high)
            return (
                start_positions,
                jnp.zeros((0, 3), dtype=jnp.float32),
                jnp.zeros((0, 3), dtype=jnp.float32),
            )

        total = self.num_agents + self.num_goals + self.num_obstacles
        halfsizes = jnp.concatenate(
            [
                jnp.full((self.num_agents,), self.cfg.cf_col_radius),
                jnp.full((self.num_goals,), self.cfg.goal_width / 2.0),
                jnp.full((self.num_obstacles,), self.cfg.obs_width / 2.0),
            ],
            axis=0,
        )
        centers_xy = sample_non_overlapping_squares(
            key,
            halfsizes,
            self._world_rect[:4],
            pair_gap=self.cfg.reset_pair_gap,
            attempts=self.cfg.reset_attempts,
            relax_iters=self.cfg.reset_relax_iters,
        )
        starts_xy = centers_xy[: self.num_agents]
        goals_xy = centers_xy[self.num_agents : self.num_agents + self.num_goals]
        obs_xy = centers_xy[self.num_agents + self.num_goals :]
        start_positions = jnp.concatenate(
            [starts_xy, jnp.full((self.num_agents, 1), self.cfg.start_z, dtype=centers_xy.dtype)],
            axis=1,
        )
        goal_positions = jnp.concatenate(
            [goals_xy, jnp.full((self.num_goals, 1), self.cfg.goal_z, dtype=centers_xy.dtype)],
            axis=1,
        )
        obstacle_positions = jnp.concatenate(
            [obs_xy, jnp.full((self.num_obstacles, 1), self.cfg.obs_height / 2.0, dtype=centers_xy.dtype)],
            axis=1,
        )
        return start_positions, goal_positions, obstacle_positions

    @partial(jax.jit, static_argnums=0)
    def _goal_bounds(self, goal_positions: chex.Array) -> chex.Array:
        if self.num_goals == 0:
            return jnp.zeros((0, 6), dtype=jnp.float32)
        half = self._goal_half
        return jnp.concatenate([goal_positions - half[None, :], goal_positions + half[None, :]], axis=1)

    @partial(jax.jit, static_argnums=0)
    def reset(self, key: chex.PRNGKey) -> tuple[chex.Array, EnvState]:
        data = mjx.make_data(self.m, impl=self._mjx_impl) if self._mjx_impl else mjx.make_data(self.m)
        key, layout_key, yaw_key, dynamics_key = jax.random.split(key, 4)
        start_positions, goal_positions, obstacle_positions = self._sample_layout(layout_key)

        qpos = data.qpos.reshape((self.num_agents, 7))
        yaw = jax.random.uniform(yaw_key, (self.num_agents,), minval=-0.05, maxval=0.05)
        quat = jnp.stack(
            [jnp.cos(0.5 * yaw), jnp.zeros_like(yaw), jnp.zeros_like(yaw), jnp.sin(0.5 * yaw)],
            axis=1,
        )
        qpos = qpos.at[:, 0:3].set(start_positions)
        qpos = qpos.at[:, 3:7].set(quat)
        qvel = jnp.zeros_like(data.qvel)

        mocap_pos = data.mocap_pos
        if self.num_obstacles > 0:
            mocap_pos = mocap_pos.at[: self.num_obstacles].set(obstacle_positions.astype(mocap_pos.dtype))
        if self.num_goals > 0:
            goal_start = self.num_obstacles
            mocap_pos = mocap_pos.at[goal_start : goal_start + self.num_goals].set(goal_positions.astype(mocap_pos.dtype))

        data = data.replace(qpos=qpos.reshape((-1,)), qvel=qvel, mocap_pos=mocap_pos)
        data = mjx.forward(self.m, data)
        dynamics_params = jax.lax.cond(
            self.cfg.randomize_dynamics,
            lambda k: MotorDynamics.sample_randomized_params(k, self.cfg),
            lambda k: self.motor_dynamics.get_nominal_params(),
            dynamics_key,
        )
        state = EnvState(
            data=data,
            step=jnp.array(0, dtype=jnp.int32),
            current_motor_speeds=self._initial_motor_speeds,
            prev_omega_err_integral=self._zero_omega,
            prev_omega_meas=self._zero_omega,
            prev_action=self._zero_action,
            dynamics_params=dynamics_params,
            goal_positions=goal_positions,
            obstacle_positions=obstacle_positions,
            current_goal_bounds=self._goal_bounds(goal_positions),
            in_contact=jnp.zeros((self.num_agents,), dtype=jnp.bool_),
        )
        return self.obs_fn(state), state

    @partial(jax.jit, static_argnums=0)
    def obs_fn(self, state: EnvState) -> chex.Array:
        data = state.data
        rotmats = data.xmat[jnp.asarray(self._cf_body_ids), :, :]
        rotmats_t = rotmats.transpose((0, 2, 1))
        qpos = data.qpos.reshape((self.num_agents, 7))
        qvel = data.qvel.reshape((self.num_agents, 6))
        pos = qpos[:, 0:3]
        lin_vel_body = jnp.einsum("bij,bj->bi", rotmats_t, qvel[:, 0:3])
        ang_vel_body = jnp.einsum("bij,bj->bi", rotmats_t, qvel[:, 3:6])
        gravity_local = jnp.einsum("bij,j->bi", rotmats_t, jnp.asarray([0.0, 0.0, -1.0]))
        base = jnp.concatenate([pos, lin_vel_body, ang_vel_body, gravity_local], axis=1)
        if self.scenario == "single":
            return base

        goal_diffs = state.goal_positions[None, :, :] - pos[:, None, :]
        goal_local = jnp.einsum("bij,bkj->bki", rotmats_t, goal_diffs).reshape(
            (self.num_agents, self.num_goals * 3)
        )
        obs_diffs = state.obstacle_positions[None, :, :] - pos[:, None, :]
        obs_local = jnp.einsum("bij,bkj->bki", rotmats_t, obs_diffs).reshape(
            (self.num_agents, self.num_obstacles * 3)
        )
        agent_diffs = pos[None, :, :] - pos[:, None, :]
        other_all = jnp.einsum("bij,bkj->bki", rotmats_t, agent_diffs)
        other_local = jnp.take_along_axis(other_all, self._other_agent_indices[:, :, None], axis=1).reshape(
            (self.num_agents, (self.num_agents - 1) * 3)
        )
        agent_id = jax.nn.one_hot(jnp.arange(self.num_agents), self.num_agents, dtype=pos.dtype)
        return jnp.concatenate([base, goal_local, obs_local, other_local, agent_id], axis=1)

    @partial(jax.jit, static_argnums=0)
    def _apply_action_dynamics(
        self,
        state: EnvState,
        actions: chex.Array,
    ) -> tuple[EnvState, chex.Array, chex.Array]:
        data0 = state.data
        normalized_actions = jnp.clip(actions, -1.0, 1.0)
        ctbr_actions = normalized_actions

        if self.cfg.use_motor_dynamics:
            xmats = data0.xmat[jnp.asarray(self._cf_body_ids), :, :]
            xmats_t = xmats.transpose((0, 2, 1))
            qvel = data0.qvel.reshape((self.num_agents, 6))
            body_ang_vels = jnp.einsum("bij,bj->bi", xmats_t, qvel[:, 3:6])
            body_lin_vels = jnp.einsum("bij,bj->bi", xmats_t, qvel[:, 0:3])
            if self.cfg.control_mode == "velocity_yaw_rate":
                ctbr_actions = jax.vmap(self.velocity_controller.command_to_ctbr)(
                    normalized_actions,
                    xmats,
                    qvel[:, 0:3],
                )
            (
                body_thrust,
                body_moment,
                motor_speeds,
                omega_err_integral,
                omega_meas,
            ) = jax.vmap(
                self.motor_dynamics.forces_and_torques_from_ctbr,
                in_axes=(0, 0, 0, 0, 0, 0, None),
            )(
                ctbr_actions,
                body_ang_vels,
                body_lin_vels,
                state.current_motor_speeds,
                state.prev_omega_err_integral,
                state.prev_omega_meas,
                state.dynamics_params,
            )
            world_thrust = jnp.einsum("bij,bj->bi", xmats, body_thrust)
            world_moments = jnp.einsum("bij,bj->bi", xmats, body_moment)
            xfrc = jnp.zeros((self.m.nbody, 6))
            xfrc = xfrc.at[jnp.asarray(self._cf_body_ids), :3].set(world_thrust)
            xfrc = xfrc.at[jnp.asarray(self._cf_body_ids), 3:].set(world_moments)
            new_data = mjx.step(self.m, data0.replace(xfrc_applied=xfrc))
            new_motor_speeds = motor_speeds
            new_integral = omega_err_integral
            new_meas = omega_meas
        else:
            ctrl = normalized_actions.at[:, 0].set((normalized_actions[:, 0] + 1.0) * 0.35 / 2.0)
            new_data = mjx.step(self.m, data0.replace(ctrl=ctrl.reshape((-1,))))
            new_motor_speeds = state.current_motor_speeds
            new_integral = state.prev_omega_err_integral
            new_meas = state.prev_omega_meas

        new_state = state.replace(
            data=new_data,
            step=state.step + 1,
            current_motor_speeds=new_motor_speeds,
            prev_omega_err_integral=new_integral,
            prev_omega_meas=new_meas,
            prev_action=normalized_actions,
        )
        return new_state, normalized_actions, ctbr_actions

    @partial(jax.jit, static_argnums=0)
    def step_rollout(self, key: chex.PRNGKey, state: EnvState, actions: chex.Array) -> tuple[chex.Array, EnvState]:
        del key
        new_state, _, _ = self._apply_action_dynamics(state, actions)
        return self.obs_fn(new_state), new_state

    @partial(jax.jit, static_argnums=0)
    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        actions: chex.Array,
    ) -> tuple[chex.Array, EnvState, chex.Array, chex.Array, dict[str, chex.Array]]:
        del key
        new_state, normalized_actions, ctbr_actions = self._apply_action_dynamics(state, actions)
        in_contact = self._contact_flags(new_state.data)
        new_state = new_state.replace(in_contact=in_contact)
        reward, info = self._reward_info(new_state, normalized_actions, in_contact)
        info["ctbr_action"] = ctbr_actions
        done = new_state.step >= self.cfg.horizon
        return self.obs_fn(new_state), new_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def _contact_flags(self, data: mjx.Data) -> chex.Array:
        if self.cfg.disable_collisions or self.mj_model.nsensordata == 0:
            return jnp.zeros((self.num_agents,), dtype=jnp.bool_)
        return data.sensordata.reshape((-1,))[: self.num_agents] > 0.0

    @partial(jax.jit, static_argnums=0)
    def _reward_info(
        self,
        state: EnvState,
        actions: chex.Array,
        in_contact: chex.Array,
    ) -> tuple[chex.Array, dict[str, chex.Array]]:
        qpos = state.data.qpos.reshape((self.num_agents, 7))
        pos = qpos[:, 0:3]
        action_energy = jnp.mean(jnp.square(actions), axis=1)
        world_margin = self._inside_world_margin(pos)

        if self.num_goals > 0:
            assigned = jnp.arange(self.num_agents) % self.num_goals
            assigned_goals = state.goal_positions[assigned]
            goal_distance = jnp.linalg.norm(pos - assigned_goals, axis=1)
            in_goal = self._inside_boxes(pos, state.current_goal_bounds)
            assigned_in_goal = in_goal[jnp.arange(self.num_agents), assigned]
        else:
            goal_distance = jnp.zeros((self.num_agents,), dtype=pos.dtype)
            assigned_in_goal = jnp.zeros((self.num_agents,), dtype=jnp.bool_)

        if self.num_obstacles > 0:
            obstacle_clearance = self._nearest_obstacle_clearance(pos, state.obstacle_positions)
        else:
            obstacle_clearance = jnp.full((self.num_agents,), 1.0, dtype=pos.dtype)

        reward = (
            -goal_distance
            + 2.0 * assigned_in_goal.astype(pos.dtype)
            + 0.1 * world_margin
            + 0.05 * obstacle_clearance
            - 2.0 * in_contact.astype(pos.dtype)
            - 0.01 * action_energy
        )
        if self.scenario == "single":
            reward = 0.1 * world_margin - 0.01 * action_energy - 2.0 * in_contact.astype(pos.dtype)

        return reward, {
            "in_contact": in_contact,
            "goal_reached": assigned_in_goal,
            "goal_distance": goal_distance,
            "nearest_obstacle_clearance": obstacle_clearance,
            "world_margin": world_margin,
            "action_energy": action_energy,
        }

    @partial(jax.jit, static_argnums=0)
    def _inside_world_margin(self, pos: chex.Array) -> chex.Array:
        x_min, x_max, y_min, y_max, z_min, z_max = self._world_rect
        return jnp.minimum(
            jnp.minimum(pos[:, 0] - x_min, x_max - pos[:, 0]),
            jnp.minimum(
                jnp.minimum(pos[:, 1] - y_min, y_max - pos[:, 1]),
                jnp.minimum(pos[:, 2] - z_min, z_max - pos[:, 2]),
            ),
        )

    @partial(jax.jit, static_argnums=0)
    def _inside_boxes(self, pos: chex.Array, boxes: chex.Array) -> chex.Array:
        in_x = (pos[:, None, 0] >= boxes[None, :, 0]) & (pos[:, None, 0] <= boxes[None, :, 3])
        in_y = (pos[:, None, 1] >= boxes[None, :, 1]) & (pos[:, None, 1] <= boxes[None, :, 4])
        in_z = (pos[:, None, 2] >= boxes[None, :, 2]) & (pos[:, None, 2] <= boxes[None, :, 5])
        return in_x & in_y & in_z

    @partial(jax.jit, static_argnums=0)
    def _nearest_obstacle_clearance(self, pos: chex.Array, obstacle_positions: chex.Array) -> chex.Array:
        half = self._obs_half
        delta = jnp.abs(pos[:, None, :] - obstacle_positions[None, :, :]) - half[None, None, :]
        outside = jnp.linalg.norm(jnp.maximum(delta, 0.0), axis=2)
        inside = jnp.minimum(jnp.max(delta, axis=2), 0.0)
        signed_distance = outside + inside
        return jnp.min(signed_distance, axis=1)

    def render(self, data: mjx.Data, camera: str | None = "overview") -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.mj_model, height=720, width=1080)
        mj_data = mjx.get_data(self.mj_model, data)
        camera_name = camera
        if camera_name is not None:
            camera_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if camera_id < 0:
                camera_name = None
        self._renderer.update_scene(mj_data, camera=camera_name)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def make_env(scenario: ScenarioName | None = None, config: CrazyflieConfig | None = None) -> CrazyflieMJXEnv:
    return CrazyflieMJXEnv(scenario=scenario, config=config)
