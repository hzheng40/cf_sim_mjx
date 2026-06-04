from __future__ import annotations

import argparse
import functools
import json
import os
import socketserver
import time
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

from .config import CrazyflieConfig
from .runtime import add_runtime_args, configure_jax_cache, configure_runtime


def _preconfigure_runtime() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    add_runtime_args(parser)
    args, _ = parser.parse_known_args()
    configure_runtime(args)
    return args


_RUNTIME_ARGS = _preconfigure_runtime()

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from . import make_env
from .web import generate_web_report


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run minimal Crazyflie MJX simulation demos.")
    add_runtime_args(parser)
    parser.add_argument("--scenario", choices=["single", "multi_reach_avoid"], default="single")
    parser.add_argument("--horizon", type=int, default=300)
    parser.add_argument("--timestep", type=float, default=0.002)
    parser.add_argument("--num-agents", type=int, default=4)
    parser.add_argument("--num-goals", type=int, default=0)
    parser.add_argument("--num-obstacles", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--use-motor-dynamics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--randomize-dynamics", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--disable-visualization", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--disable-collisions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-world-bound-collisions", action="store_true")
    parser.add_argument("--world-bound-collision-thickness", type=float, default=0.05)
    parser.add_argument("--mj-solver", choices=["cg", "newton"], default="cg")
    parser.add_argument("--mj-iterations", type=int, default=1)
    parser.add_argument("--mj-ls-iterations", type=int, default=4)
    parser.add_argument("--mj-tolerance", type=float, default=0.0)
    parser.add_argument("--mj-ls-tolerance", type=float, default=0.0)
    parser.add_argument("--mjx-backend", choices=["cpu", "warp"], default=None)
    parser.add_argument("--use-euler-integrator", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--render-mode", choices=["none", "mujoco", "web", "both"], default="web")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/demo"))
    parser.add_argument("--camera", type=str, default="overview")
    parser.add_argument("--render-stride", type=int, default=5)
    parser.add_argument("--max-rollouts", type=int, default=4)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--serve-port", type=int, default=8000)
    return parser


def _make_config(args: argparse.Namespace) -> CrazyflieConfig:
    return CrazyflieConfig(
        scenario=args.scenario,
        num_agents=args.num_agents,
        num_goals=args.num_goals,
        num_obstacles=args.num_obstacles,
        horizon=args.horizon,
        timestep=args.timestep,
        use_motor_dynamics=args.use_motor_dynamics,
        randomize_dynamics=args.randomize_dynamics,
        disable_visualization=args.disable_visualization,
        disable_collisions=args.disable_collisions,
        enable_world_bound_collisions=args.enable_world_bound_collisions,
        world_bound_collision_thickness=args.world_bound_collision_thickness,
        solver=args.mj_solver,
        iterations=args.mj_iterations,
        ls_iterations=args.mj_ls_iterations,
        tolerance=args.mj_tolerance,
        ls_tolerance=args.mj_ls_tolerance,
        mjx_backend=args.mjx_backend,
        use_euler_integrator=args.use_euler_integrator,
    )


def _rollout(env, args: argparse.Namespace):
    batch_size = int(args.batch_size)
    horizon = int(args.horizon)
    keys = jax.random.split(jax.random.PRNGKey(args.seed), batch_size)
    reset_batch = jax.jit(jax.vmap(env.reset))
    step_batch = jax.jit(jax.vmap(env.step_env, in_axes=(0, 0, 0)))
    obs0, state0 = reset_batch(keys)
    actions = jnp.zeros((batch_size, env.num_agents, env.action_size), dtype=obs0.dtype)

    def scan_step(carry, _t):
        state, key = carry
        key, step_key = jax.random.split(key)
        step_keys = jax.random.split(step_key, batch_size)
        obs, state, reward, done, info = step_batch(step_keys, state, actions)
        qpos = state.data.qpos.reshape((batch_size, env.num_agents, 7))
        out = {
            "obs": obs,
            "qpos": qpos,
            "reward": reward,
            "done": done,
            "in_contact": info["in_contact"],
            "goal_reached": info["goal_reached"],
            "goal_distance": info["goal_distance"],
        }
        return (state, key), out

    t0 = time.perf_counter()
    (final_state, _), trace = jax.lax.scan(scan_step, (state0, jax.random.PRNGKey(args.seed + 999)), jnp.arange(horizon))
    jax.block_until_ready(trace["qpos"])
    elapsed = time.perf_counter() - t0
    return state0, final_state, trace, elapsed


def _scene_boxes(env, state0, idx: int) -> dict[str, list[dict[str, Any]]]:
    goal_positions = np.asarray(state0.goal_positions[idx])
    obstacle_positions = np.asarray(state0.obstacle_positions[idx])
    return {
        "goals": [
            {"center": p.tolist(), "halfsize": [env.cfg.goal_width / 2.0] * 3}
            for p in goal_positions
        ],
        "obstacles": [
            {
                "center": p.tolist(),
                "halfsize": [env.cfg.obs_width / 2.0, env.cfg.obs_width / 2.0, env.cfg.obs_height / 2.0],
            }
            for p in obstacle_positions
        ],
    }


def _rollout_payloads(env, state0, trace, args: argparse.Namespace) -> list[dict[str, Any]]:
    qpos = np.asarray(trace["qpos"])
    rewards = np.asarray(trace["reward"])
    contacts = np.asarray(trace["in_contact"])
    goals = np.asarray(trace["goal_reached"])
    stride = max(1, int(args.render_stride))
    payloads = []
    for idx in range(min(int(args.max_rollouts), qpos.shape[1])):
        q = qpos[::stride, idx]
        payloads.append(
            {
                "name": f"{args.scenario}_rollout_{idx:03d}",
                "dt": float(args.timestep * stride),
                "positions": q[:, :, 0:3],
                "quaternions": q[:, :, 3:7],
                "scene": _scene_boxes(env, state0, idx),
                "metrics": {
                    "reward_mean": float(rewards[:, idx].mean()),
                    "reward_final_sum": float(rewards[-1, idx].sum()),
                    "contact_steps": int(contacts[:, idx].sum()),
                    "goal_reached_final": int(goals[-1, idx].sum()),
                },
            }
        )
    return payloads


def _serve(web_dir: Path, port: int) -> None:
    handler = functools.partial(SimpleHTTPRequestHandler, directory=str(web_dir))
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"serving: http://127.0.0.1:{port}/")
        httpd.serve_forever()


def main() -> None:
    args = _build_parser().parse_args()
    configure_runtime(args)
    configure_jax_cache(args)
    if args.horizon <= 0 or args.batch_size <= 0:
        raise ValueError("--horizon and --batch-size must be positive")
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.scenario, _make_config(args))
    state0, final_state, trace, elapsed = _rollout(env, args)
    summary = {
        **env.metadata(),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "rollout_seconds": elapsed,
        "sim_steps": int(args.batch_size * args.horizon),
    }
    (out_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    np.savez(
        out_dir / "rollouts.npz",
        qpos=np.asarray(trace["qpos"]),
        reward=np.asarray(trace["reward"]),
        in_contact=np.asarray(trace["in_contact"]),
        goal_reached=np.asarray(trace["goal_reached"]),
    )

    if args.render_mode in {"mujoco", "both"}:
        frame = env.render(final_state.data, camera=args.camera)
        Image.fromarray(frame).save(out_dir / "mujoco_frame.png")
        env.close()
        print(f"mujoco_frame: {out_dir / 'mujoco_frame.png'}")

    web_index = None
    if args.render_mode in {"web", "both"}:
        web_index = generate_web_report(
            out_dir,
            env,
            summary=summary,
            rollout_payloads=_rollout_payloads(env, state0, trace, args),
        )
        print(f"web_report: {web_index}")

    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    if args.serve:
        if web_index is None:
            raise ValueError("--serve requires --render-mode web or both")
        _serve(web_index.parent, args.serve_port)


if __name__ == "__main__":
    main()
