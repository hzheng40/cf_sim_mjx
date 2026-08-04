from __future__ import annotations

from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp


class VelocityYawRateController:
    """Differentiable velocity/yaw-rate outer loop producing CTBR commands.

    Horizontal velocity commands are interpreted in a yaw-aligned heading
    frame, while vertical velocity is expressed along world Z. The returned
    collective-thrust/body-rate (CTBR) command is normalized to ``[-1, 1]`` so
    it can be passed directly to :class:`MotorDynamics`.
    """

    def __init__(self, cfg: Any, *, mass: float, weight: float):
        self._mass = float(mass)
        self._weight = float(weight)
        self._gravity = self._weight / self._mass
        self._velocity_scale = jnp.asarray(
            [cfg.velocity_scale_xy, cfg.velocity_scale_xy, cfg.velocity_scale_z],
            dtype=jnp.float32,
        )
        self._velocity_kp = jnp.asarray(
            [cfg.velocity_kp_xy, cfg.velocity_kp_xy, cfg.velocity_kp_z],
            dtype=jnp.float32,
        )
        self._max_accel_xy = float(cfg.velocity_max_accel_xy)
        self._max_accel_z = float(cfg.velocity_max_accel_z)
        self._max_tilt = float(cfg.velocity_max_tilt)
        self._attitude_kp = float(cfg.velocity_attitude_kp)
        self._yaw_rate_scale = float(cfg.velocity_yaw_rate_scale)
        self._body_rate_scale = jnp.asarray(
            [cfg.body_rate_scale_xy, cfg.body_rate_scale_xy, cfg.body_rate_scale_z],
            dtype=jnp.float32,
        )
        self._max_collective_thrust = self._weight * float(cfg.thrust_to_weight)
        self._normalized_thrust_scale = 0.5 * self._max_collective_thrust
        self._eps = float(cfg.velocity_controller_epsilon)

    def _limit_norm(self, vector: chex.Array, limit: float) -> chex.Array:
        norm = jnp.sqrt(jnp.sum(jnp.square(vector)) + self._eps**2)
        return vector * jnp.minimum(1.0, limit / norm)

    @staticmethod
    def _vee(skew: chex.Array) -> chex.Array:
        return jnp.asarray([skew[2, 1], skew[0, 2], skew[1, 0]])

    @partial(jax.jit, static_argnums=0)
    def command_to_ctbr(
        self,
        normalized_command: chex.Array,
        body_to_world: chex.Array,
        linear_velocity_world: chex.Array,
    ) -> chex.Array:
        """Maps ``[vx, vy, vz, yaw_rate]`` to normalized CTBR."""
        velocity_heading = normalized_command[:3] * self._velocity_scale
        yaw_rate = normalized_command[3] * self._yaw_rate_scale

        yaw = jnp.arctan2(body_to_world[1, 0], body_to_world[0, 0])
        cos_yaw = jnp.cos(yaw)
        sin_yaw = jnp.sin(yaw)
        velocity_des_world = jnp.asarray(
            [
                cos_yaw * velocity_heading[0] - sin_yaw * velocity_heading[1],
                sin_yaw * velocity_heading[0] + cos_yaw * velocity_heading[1],
                velocity_heading[2],
            ]
        )

        velocity_error = velocity_des_world - linear_velocity_world
        accel_des = self._velocity_kp * velocity_error
        accel_xy = self._limit_norm(accel_des[:2], self._max_accel_xy)
        accel_z = jnp.clip(accel_des[2], -self._max_accel_z, self._max_accel_z)

        specific_force_z = jnp.maximum(self._gravity + accel_z, self._eps)
        max_force_xy = specific_force_z * jnp.tan(self._max_tilt)
        specific_force_xy = self._limit_norm(accel_xy, max_force_xy)
        specific_force_world = jnp.concatenate([specific_force_xy, jnp.asarray([specific_force_z])])
        force_des_world = self._mass * specific_force_world

        body_z_des = force_des_world / jnp.sqrt(
            jnp.sum(jnp.square(force_des_world)) + self._eps**2
        )
        heading_des = jnp.asarray([cos_yaw, sin_yaw, 0.0])
        body_y_des_raw = jnp.cross(body_z_des, heading_des)
        body_y_des = body_y_des_raw / jnp.sqrt(
            jnp.sum(jnp.square(body_y_des_raw)) + self._eps**2
        )
        body_x_des = jnp.cross(body_y_des, body_z_des)
        rotation_des = jnp.stack([body_x_des, body_y_des, body_z_des], axis=1)

        attitude_skew = 0.5 * (
            rotation_des.T @ body_to_world - body_to_world.T @ rotation_des
        )
        attitude_error = self._vee(attitude_skew)
        yaw_rate_des = jnp.asarray([0.0, 0.0, yaw_rate])
        body_rate_feedforward = body_to_world.T @ rotation_des @ yaw_rate_des
        body_rate_des = -self._attitude_kp * attitude_error + body_rate_feedforward
        normalized_body_rate = jnp.clip(body_rate_des / self._body_rate_scale, -1.0, 1.0)

        collective_thrust = jnp.dot(force_des_world, body_to_world[:, 2])
        collective_thrust = jnp.clip(collective_thrust, 0.0, self._max_collective_thrust)
        normalized_thrust = collective_thrust / self._normalized_thrust_scale - 1.0
        return jnp.clip(
            jnp.concatenate([jnp.asarray([normalized_thrust]), normalized_body_rate]),
            -1.0,
            1.0,
        )
