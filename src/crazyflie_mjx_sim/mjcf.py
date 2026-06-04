from __future__ import annotations

from pathlib import Path

import numpy as np


def assets_dir() -> Path:
    return Path(__file__).resolve().parent / "assets"


def _fmt(*values) -> str:
    return " ".join(f"{float(v):g}" for v in values)


def _grid_xy(count: int, zone: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    if count <= 0:
        return []
    xmin, xmax, ymin, ymax = zone
    rows = int(np.ceil(np.sqrt(count)))
    cols = int(np.ceil(count / rows))
    xs = np.linspace(xmin, xmax, cols)
    ys = np.linspace(ymin, ymax, rows)
    return [(float(x), float(y)) for y in ys for x in xs][:count]


def _drone_body_block(
    *,
    index: int,
    pos: tuple[float, float, float],
    cf_col_radius: float,
    disable_visualization: bool,
    disable_collisions: bool,
) -> str:
    p = f"cf_{index}/"
    x, y, z = pos
    color = "blue" if index == 0 else "red"
    propeller = "blue_propeller_plastic" if index == 0 else "red_propeller_plastic"
    camera = "" if disable_visualization else f'<camera name="{p}track" pos="-1 0 .5" xyaxes="0 -1 0 1 0 2" mode="trackcom"/>'
    visual_geoms = (
        ""
        if disable_visualization
        else f"""
        <geom mesh="cf2_0" material="{propeller}" class="visual"/>
        <geom mesh="cf2_1" material="medium_gloss_plastic" class="visual"/>
        <geom mesh="cf2_2" material="polished_gold" class="visual"/>
        <geom mesh="cf2_3" material="polished_plastic" class="visual"/>
        <geom mesh="cf2_4" material="burnished_chrome" class="visual"/>
        <geom mesh="cf2_5" material="body_frame_plastic" class="visual"/>
        <geom mesh="cf2_6" material="{color}" class="visual"/>"""
    )
    collides = 0 if disable_collisions else 1
    return f"""
      <body name="{p}body" pos="{_fmt(x, y, z)}" quat="1 0 0 0" childclass="cf2">
        <freejoint/>
        <inertial pos="0 0 0" mass="0.027" diaginertia="2.3951e-5 2.3951e-5 3.2347e-5"/>
        {camera}
{visual_geoms}
        <geom name="{p}col" type="sphere" size="{cf_col_radius:g}" rgba="0 0 1 0" class="collision" contype="{collides}" conaffinity="{collides}"/>
        <site name="{p}imu" group="5"/>
        <site name="{p}actuation" group="5"/>
      </body>""".rstrip()


def _sensor_block(index: int, keep_only_contact_sensors: bool) -> str:
    p = f"cf_{index}/"
    if keep_only_contact_sensors:
        return f'    <contact name="{p}body_contact" geom1="{p}col"/>'
    return "\n".join(
        [
            f'    <contact name="{p}body_contact" geom1="{p}col"/>',
            f'    <gyro name="{p}body_gyro" site="{p}imu"/>',
            f'    <accelerometer name="{p}body_linacc" site="{p}imu"/>',
            f'    <framequat name="{p}body_quat" objtype="site" objname="{p}imu"/>',
        ]
    )


def _actuator_block(index: int) -> str:
    p = f"cf_{index}/"
    return f"""
    <motor class="cf2" ctrlrange="0 0.35" gear="0 0 1 0 0 0" site="{p}actuation" name="{p}body_thrust"/>
    <motor class="cf2" ctrlrange="-1 1" gear="0 0 0 -0.00001 0 0" site="{p}actuation" name="{p}x_moment"/>
    <motor class="cf2" ctrlrange="-1 1" gear="0 0 0 0 -0.00001 0" site="{p}actuation" name="{p}y_moment"/>
    <motor class="cf2" ctrlrange="-1 1" gear="0 0 0 0 0 -0.00001" site="{p}actuation" name="{p}z_moment"/>""".rstrip()


def _box_body(name: str, pos: tuple[float, float, float], halfsize: tuple[float, float, float], rgba: str, collides: bool) -> str:
    contype = 1 if collides else 0
    return f"""
      <body name="{name}_body" mocap="true">
        <geom name="{name}" type="box" pos="0 0 0" size="{_fmt(*halfsize)}" rgba="{rgba}" contype="{contype}" conaffinity="{contype}"/>
      </body>""".rstrip()


def _world_boundaries(
    world_rect: tuple[float, float, float, float, float, float],
    thickness: float,
    *,
    disable_visualization: bool,
    disable_collisions: bool,
) -> str:
    wx_min, wx_max, wy_min, wy_max, wz_min, wz_max = [float(v) for v in world_rect]
    x_mid, y_mid, z_mid = 0.5 * (wx_min + wx_max), 0.5 * (wy_min + wy_max), 0.5 * (wz_min + wz_max)
    x_half, y_half, z_half = 0.5 * (wx_max - wx_min), 0.5 * (wy_max - wy_min), 0.5 * (wz_max - wz_min)
    contype = 0 if disable_collisions else 1
    alpha = 0.0 if disable_visualization else 0.16
    specs = [
        ("world_x_min", (wx_min - thickness, y_mid, z_mid), (thickness, y_half + thickness, z_half + thickness)),
        ("world_x_max", (wx_max + thickness, y_mid, z_mid), (thickness, y_half + thickness, z_half + thickness)),
        ("world_y_min", (x_mid, wy_min - thickness, z_mid), (x_half + thickness, thickness, z_half + thickness)),
        ("world_y_max", (x_mid, wy_max + thickness, z_mid), (x_half + thickness, thickness, z_half + thickness)),
        ("world_z_min", (x_mid, y_mid, wz_min - thickness), (x_half + thickness, y_half + thickness, thickness)),
        ("world_z_max", (x_mid, y_mid, wz_max + thickness), (x_half + thickness, y_half + thickness, thickness)),
    ]
    return "\n".join(
        f'      <geom name="{name}" type="box" pos="{_fmt(*pos)}" size="{_fmt(*size)}" rgba="0.15 0.15 0.15 {alpha:g}" contype="{contype}" conaffinity="{contype}"/>'
        for name, pos, size in specs
    )


def make_cf2_scene_xml(
    *,
    num_agents: int,
    num_goals: int,
    num_obstacles: int,
    start_z: float,
    goal_z: float,
    goal_width: float,
    obs_width: float,
    obs_height: float,
    cf_col_radius: float,
    world_rect: tuple[float, float, float, float, float, float],
    disable_visualization: bool,
    disable_collisions: bool,
    enable_world_bound_collisions: bool,
    world_bound_collision_thickness: float,
    keep_only_contact_sensors: bool,
    remove_actuators: bool,
    scene_name: str,
) -> str:
    include_path = assets_dir() / "cf2_asset.xml"
    spawn_xy = _grid_xy(num_agents, (-0.4, 0.4, -0.4, 0.4))
    drones = [
        _drone_body_block(
            index=i,
            pos=(x, y, start_z),
            cf_col_radius=cf_col_radius,
            disable_visualization=disable_visualization,
            disable_collisions=disable_collisions,
        )
        for i, (x, y) in enumerate(spawn_xy)
    ]
    obstacle_half = (obs_width / 2.0, obs_width / 2.0, obs_height / 2.0)
    goal_half = (goal_width / 2.0, goal_width / 2.0, goal_width / 2.0)
    obstacles = [
        _box_body(
            f"obstacle_{i}",
            (0.0, 0.0, obs_height / 2.0),
            obstacle_half,
            "1.0 0.5 0.5 1.0",
            not disable_collisions,
        )
        for i in range(num_obstacles)
    ]
    goals = [
        _box_body(
            f"goal_{i}",
            (0.0, 0.0, goal_z),
            goal_half,
            "0.2 0.8 0.4 0.3",
            False,
        )
        for i in range(num_goals)
    ]
    sensors = [_sensor_block(i, keep_only_contact_sensors) for i in range(num_agents)]
    actuators = [_actuator_block(i) for i in range(num_agents)]
    world_bounds = (
        _world_boundaries(
            world_rect,
            world_bound_collision_thickness,
            disable_visualization=disable_visualization,
            disable_collisions=disable_collisions,
        )
        if enable_world_bound_collisions
        else ""
    )
    visual_assets = (
        ""
        if disable_visualization
        else """  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-20" elevation="-20" ellipsoidinertia="true" offwidth="1920" offheight="1080"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.57 0.57 0.57" rgb2="0.6 0.6 0.6" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>"""
    )
    visual_world = (
        ""
        if disable_visualization
        else """      <light pos="0 0 5" dir="0 0 -1" directional="true"/>
      <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
      <camera name="overview" mode="fixed" pos="7.0 -4.5 5.5" xyaxes="0.848 0.530 0.000 -0.249 0.398 0.883" fovy="55"/>
      <camera name="topdown" mode="fixed" pos="0 0 8.0" xyaxes="1 0 0 0 1 0" fovy="60"/>"""
    )
    actuator_xml = "" if remove_actuators else "  <actuator>\n" + "\n".join(actuators) + "\n  </actuator>"
    return f"""
<mujoco model="{scene_name}">
  <compiler inertiafromgeom="false" autolimits="true"/>
  <include file="{include_path}"/>
  <statistic center="0 0 0.2" extent="2.0" meansize=".1"/>
{visual_assets}
  <worldbody>
{visual_world}
{world_bounds}
{chr(10).join(obstacles)}
{chr(10).join(goals)}
{chr(10).join(drones)}
  </worldbody>
  <sensor>
{chr(10).join(sensors)}
  </sensor>
{actuator_xml}
</mujoco>
""".strip()
