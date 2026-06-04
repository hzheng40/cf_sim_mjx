from __future__ import annotations

from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import mujoco
import numpy as np


class RandomizedDynamicsParams(NamedTuple):
    k_aero: chex.Array
    kp_omega: chex.Array
    ki_omega: chex.Array
    kd_omega: chex.Array


class MotorDynamics:
    """Crazyflie motor and body-rate dynamics for normalized CTBR actions."""

    def __init__(self, cfg: NamedTuple, model: mujoco.MjModel, cf_body_ids: list[int]):
        self.cfg = cfg
        self.cf_body_ids = cf_body_ids

        r2o2 = np.sqrt(2.0) / 2.0
        rotor_base = jnp.array(
            [
                [r2o2, r2o2, 0.0],
                [r2o2, -r2o2, 0.0],
                [-r2o2, -r2o2, 0.0],
                [-r2o2, r2o2, 0.0],
            ]
        )
        self._rotor_positions = cfg.arm_length * rotor_base
        self._rotor_directions = jnp.array([1.0, -1.0, 1.0, -1.0])
        self.k = cfg.k_m / cfg.k_eta

        cross_terms = jnp.cross(self._rotor_positions, jnp.array([0.0, 0.0, 1.0]))
        self.f_to_TM = jnp.concatenate(
            [
                jnp.ones((1, 4)),
                cross_terms[:, :2].T,
                (self.k * self._rotor_directions)[None, :],
            ],
            axis=0,
        )
        self.TM_to_f = jnp.linalg.inv(self.f_to_TM)
        self._TM_to_f_T = self.TM_to_f.T
        self._f_to_TM_T = self.f_to_TM.T

        body_id = cf_body_ids[0]
        inertia = jnp.asarray(model.body_inertia[body_id])
        self.mass = float(model.body_mass[body_id])
        gravity_mag = abs(float(model.opt.gravity[2]))
        self.weight = self.mass * gravity_mag
        q = jnp.asarray(model.body_iquat[body_id])
        w, x, y, z = q
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        rot = jnp.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
            ]
        )
        self.inertia_tensor = jnp.asarray(rot.T @ jnp.diag(inertia) @ rot)

        self._kp_omega = jnp.asarray(cfg.kp_omega)
        self._ki_omega = jnp.asarray(cfg.ki_omega)
        self._kd_omega = jnp.asarray(cfg.kd_omega)
        self._body_rate_scale = jnp.array(
            [cfg.body_rate_scale_xy, cfg.body_rate_scale_xy, cfg.body_rate_scale_z]
        )
        self._pid_loop_dt = cfg.timestep
        self._pid_loop_rate = 1.0 / self._pid_loop_dt
        self._i_limits = jnp.array([cfg.i_limit_rp, cfg.i_limit_rp, cfg.i_limit_y])
        self._use_i_limits = bool((cfg.i_limit_rp > 0) or (cfg.i_limit_y > 0))
        self._zero_tol = 1e-4

        self._k_eta = cfg.k_eta
        self._inv_k_eta = 1.0 / cfg.k_eta
        self._k_aero = jnp.asarray(cfg.k_aero)
        self._motor_speed_min = cfg.motor_speed_min
        self._motor_speed_max = cfg.motor_speed_max
        self._motor_speed_step = cfg.timestep / cfg.tau_m
        self._thrust_scale = 0.5 * self.weight * cfg.thrust_to_weight
        self._z_axis = jnp.array([0.0, 0.0, 1.0])

    def get_nominal_params(self) -> RandomizedDynamicsParams:
        return RandomizedDynamicsParams(
            k_aero=self._k_aero,
            kp_omega=self._kp_omega,
            ki_omega=self._ki_omega,
            kd_omega=self._kd_omega,
        )

    @staticmethod
    def sample_randomized_params(key: chex.PRNGKey, cfg: NamedTuple) -> RandomizedDynamicsParams:
        keys = jax.random.split(key, 7)
        k_aero_scale = jax.random.uniform(
            keys[0], (3,), minval=cfg.k_aero_scale_min, maxval=cfg.k_aero_scale_max
        )
        kp_rp_scale = jax.random.uniform(
            keys[1], (), minval=cfg.kp_omega_rp_scale_min, maxval=cfg.kp_omega_rp_scale_max
        )
        ki_rp_scale = jax.random.uniform(
            keys[2], (), minval=cfg.ki_omega_rp_scale_min, maxval=cfg.ki_omega_rp_scale_max
        )
        kd_rp_scale = jax.random.uniform(
            keys[3], (), minval=cfg.kd_omega_rp_scale_min, maxval=cfg.kd_omega_rp_scale_max
        )
        kp_y_scale = jax.random.uniform(
            keys[4], (), minval=cfg.kp_omega_y_scale_min, maxval=cfg.kp_omega_y_scale_max
        )
        ki_y_scale = jax.random.uniform(
            keys[5], (), minval=cfg.ki_omega_y_scale_min, maxval=cfg.ki_omega_y_scale_max
        )
        kd_y_scale = jax.random.uniform(
            keys[6], (), minval=cfg.kd_omega_y_scale_min, maxval=cfg.kd_omega_y_scale_max
        )
        return RandomizedDynamicsParams(
            k_aero=jnp.asarray(cfg.k_aero) * k_aero_scale,
            kp_omega=jnp.array(
                [cfg.kp_omega_rp * kp_rp_scale, cfg.kp_omega_rp * kp_rp_scale, cfg.kp_omega_y * kp_y_scale]
            ),
            ki_omega=jnp.array(
                [cfg.ki_omega_rp * ki_rp_scale, cfg.ki_omega_rp * ki_rp_scale, cfg.ki_omega_y * ki_y_scale]
            ),
            kd_omega=jnp.array(
                [cfg.kd_omega_rp * kd_rp_scale, cfg.kd_omega_rp * kd_rp_scale, cfg.kd_omega_y * kd_y_scale]
            ),
        )

    @partial(jax.jit, static_argnums=0)
    def _compute_motor_speeds(self, wrench_des: chex.Array) -> chex.Array:
        f_des = jnp.einsum("i,ij->j", wrench_des, self._TM_to_f_T)
        motor_speed_squared = f_des * self._inv_k_eta
        motor_speeds_des = jnp.sign(motor_speed_squared) * jnp.sqrt(jnp.abs(motor_speed_squared))
        return jnp.clip(motor_speeds_des, self._motor_speed_min, self._motor_speed_max)

    @partial(jax.jit, static_argnums=0)
    def _get_moment_from_ctbr(
        self,
        action: chex.Array,
        root_com_ang_vel_b: chex.Array,
        previous_omega_err_integral: chex.Array,
        previous_omega_meas: chex.Array,
        kp_omega: chex.Array,
        ki_omega: chex.Array,
        kd_omega: chex.Array,
    ) -> tuple[chex.Array, chex.Array, chex.Array]:
        omega_des = action[1:4] * self._body_rate_scale
        omega_err = omega_des - root_com_ang_vel_b
        omega_err_integral = previous_omega_err_integral + omega_err * self._pid_loop_dt
        if self._use_i_limits:
            omega_err_integral = jnp.clip(omega_err_integral, -self._i_limits, self._i_limits)
        is_first_iter = jnp.abs(previous_omega_meas) < self._zero_tol
        omega_meas_prev = jnp.where(is_first_iter, root_com_ang_vel_b, previous_omega_meas)
        omega_meas_dot = (root_com_ang_vel_b - omega_meas_prev) * self._pid_loop_rate
        omega_dot = kp_omega * omega_err + ki_omega * omega_err_integral - kd_omega * omega_meas_dot
        cmd_moment = jnp.einsum("ij,j->i", self.inertia_tensor, omega_dot)
        return cmd_moment, omega_err_integral, root_com_ang_vel_b

    @partial(jax.jit, static_argnums=0)
    def forces_and_torques_from_ctbr(
        self,
        action: chex.Array,
        root_com_ang_vel_b: chex.Array,
        root_com_lin_vel_b: chex.Array,
        current_motor_speeds: chex.Array,
        previous_omega_err_integral: chex.Array,
        previous_omega_meas: chex.Array,
        dynamics_params: RandomizedDynamicsParams,
    ) -> tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        thrust_hover = self._thrust_scale * (action[0] + 1.0)
        moment_des, omega_err_integral, omega_meas = self._get_moment_from_ctbr(
            action,
            root_com_ang_vel_b,
            previous_omega_err_integral,
            previous_omega_meas,
            dynamics_params.kp_omega,
            dynamics_params.ki_omega,
            dynamics_params.kd_omega,
        )
        wrench_des = jnp.concatenate([jnp.array([thrust_hover]), moment_des])
        motor_speeds_des = self._compute_motor_speeds(wrench_des)
        motor_speeds = current_motor_speeds + (motor_speeds_des - current_motor_speeds) * self._motor_speed_step
        motor_speeds = jnp.clip(motor_speeds, self._motor_speed_min, self._motor_speed_max)
        motor_forces = self._k_eta * motor_speeds**2
        wrench = jnp.einsum("i,ij->j", motor_forces, self._f_to_TM_T)
        theta_dot = jnp.sum(motor_speeds)
        drag = -theta_dot * dynamics_params.k_aero * root_com_lin_vel_b
        thrust = drag + self._z_axis * wrench[0]
        return thrust, wrench[1:], motor_speeds, omega_err_integral, omega_meas
