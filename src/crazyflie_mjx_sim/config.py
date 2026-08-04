from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

ScenarioName = Literal["single", "multi_reach_avoid"]
ControlMode = Literal["ctbr", "velocity_yaw_rate"]


@dataclass(frozen=True)
class CrazyflieConfig:
    """Static configuration for Crazyflie MJX environments."""

    scenario: ScenarioName = "single"
    num_agents: int = 1
    num_goals: int = 0
    num_obstacles: int = 0
    horizon: int = 1000
    timestep: float = 0.002

    use_motor_dynamics: bool = True
    randomize_dynamics: bool = False
    control_mode: ControlMode = "ctbr"
    arm_length: float = 0.043
    k_eta: float = 2.3e-08
    k_m: float = 7.8e-10
    tau_m: float = 0.005
    motor_speed_min: float = 0.0
    motor_speed_max: float = 2500.0
    hover_motor_speed: float = 1700.0
    thrust_to_weight: float = 2.0
    k_aero_xy: float = 9.1785e-7
    k_aero_z: float = 10.311e-7

    kp_omega_rp: float = 250.0
    ki_omega_rp: float = 500.0
    kd_omega_rp: float = 2.5
    i_limit_rp: float = 33.3
    kp_omega_y: float = 120.0
    ki_omega_y: float = 16.70
    kd_omega_y: float = 0.0
    i_limit_y: float = 166.7
    body_rate_scale_xy: float = 100.0 * np.pi / 180.0
    body_rate_scale_z: float = 200.0 * np.pi / 180.0

    velocity_scale_xy: float = 1.0
    velocity_scale_z: float = 0.5
    velocity_yaw_rate_scale: float = 200.0 * np.pi / 180.0
    velocity_kp_xy: float = 2.0
    velocity_kp_z: float = 4.0
    velocity_attitude_kp: float = 4.0
    velocity_max_accel_xy: float = 4.0
    velocity_max_accel_z: float = 6.0
    velocity_max_tilt: float = 30.0 * np.pi / 180.0
    velocity_controller_epsilon: float = 1e-6

    k_aero_scale_min: float = 0.5
    k_aero_scale_max: float = 2.0
    kp_omega_rp_scale_min: float = 0.85
    kp_omega_rp_scale_max: float = 1.15
    ki_omega_rp_scale_min: float = 0.85
    ki_omega_rp_scale_max: float = 1.15
    kd_omega_rp_scale_min: float = 0.7
    kd_omega_rp_scale_max: float = 1.2
    kp_omega_y_scale_min: float = 0.85
    kp_omega_y_scale_max: float = 1.15
    ki_omega_y_scale_min: float = 0.85
    ki_omega_y_scale_max: float = 1.15
    kd_omega_y_scale_min: float = 0.7
    kd_omega_y_scale_max: float = 1.2

    world_rect: tuple[float, float, float, float, float, float] = (-3.0, 3.0, -3.0, 3.0, 0.1, 2.0)
    start_z: float = 0.5
    single_start_box: tuple[float, float, float, float, float, float] = (-0.25, 0.25, -0.25, 0.25, 0.45, 0.65)
    goal_z: float = 1.0
    goal_width: float = 0.4
    obs_width: float = 0.6
    obs_height: float = 1.0
    cf_col_radius: float = 0.06
    low_z_threshold: float = 0.061

    solver: Literal["cg", "newton"] = "cg"
    iterations: int = 1
    ls_iterations: int = 4
    tolerance: float = 0.0
    ls_tolerance: float = 0.0
    mjx_backend: Literal["cpu", "warp"] | None = None
    use_euler_integrator: bool = True

    disable_visualization: bool = False
    disable_collisions: bool = False
    enable_world_bound_collisions: bool = False
    world_bound_collision_thickness: float = 0.05
    keep_only_contact_sensors: bool = True

    reset_pair_gap: float = 0.15
    reset_attempts: int = 80
    reset_relax_iters: int = 80

    @property
    def k_aero(self):
        return np.asarray([self.k_aero_xy, self.k_aero_xy, self.k_aero_z], dtype=np.float32)

    @property
    def kp_omega(self):
        return np.asarray([self.kp_omega_rp, self.kp_omega_rp, self.kp_omega_y], dtype=np.float32)

    @property
    def ki_omega(self):
        return np.asarray([self.ki_omega_rp, self.ki_omega_rp, self.ki_omega_y], dtype=np.float32)

    @property
    def kd_omega(self):
        return np.asarray([self.kd_omega_rp, self.kd_omega_rp, self.kd_omega_y], dtype=np.float32)

    def normalized(self) -> "CrazyflieConfig":
        """Returns a scenario-consistent copy with static counts filled in."""
        if self.scenario == "single":
            return replace(self, num_agents=1, num_goals=0, num_obstacles=0)
        if self.scenario == "multi_reach_avoid":
            num_agents = max(1, int(self.num_agents))
            num_goals = int(self.num_goals) if int(self.num_goals) > 0 else num_agents
            num_obstacles = int(self.num_obstacles) if int(self.num_obstacles) > 0 else 2
            return replace(
                self,
                num_agents=num_agents,
                num_goals=num_goals,
                num_obstacles=num_obstacles,
            )
        raise ValueError(f"Unsupported scenario: {self.scenario!r}")
