# Crazyflie MJX Sim

Minimal Crazyflie simulation environments built on MuJoCo MJX. This repo keeps
the Crazyflie dynamics, batched MJX stepping, collision toggles, and browser
rendering pieces, but intentionally excludes STL specs, policy optimization,
training scripts, W&B, notebooks, and non-Crazyflie environments.

## Included Scenarios

- `single`: one Crazyflie, no goals or obstacles.
- `multi_reach_avoid`: multiple homogeneous Crazyflies with goals, obstacles,
  optional world bounds, and collision-aware helper rewards.

The public API is array-first. Observations and actions are shaped
`[num_agents, dim]` for one environment, and examples show `jax.vmap` for
`[batch, num_agents, dim]` simulation.

## Setup

```bash
cd /root/crazyflie_mjx_sim
uv sync
```

The minimal dependency set is JAX, MuJoCo/MJX, NumPy, Flax structs, Chex, and
Pillow for image examples. No training libraries are required.

For GPU machines, install a CUDA-enabled JAX extra that matches the local driver.
For example:

```bash
uv sync --extra cuda13
```

CPU execution works with the base dependency set:

```bash
JAX_PLATFORMS=cpu uv run python -m crazyflie_mjx_sim.demo --jax-platform cpu
```

## Environment API

```python
import jax
import jax.numpy as jnp

from crazyflie_mjx_sim import CrazyflieConfig, make_env

env = make_env("single", CrazyflieConfig(scenario="single"))
obs, state = env.reset(jax.random.PRNGKey(0))

actions = jnp.zeros((env.num_agents, env.action_size))
obs, state, reward, done, info = env.step_env(jax.random.PRNGKey(1), state, actions)
```

`step_env` returns lightweight helper rewards, `done`, and info dictionaries for
interactive or evaluation-style code. It does not autoreset; the caller owns
reset decisions.

For the fast rollout path:

```python
obs, state = env.step_rollout(jax.random.PRNGKey(2), state, actions)
```

`step_rollout` computes no reward, no done flag, no contact info, and no hidden
reset.

## Control Modes

The default `ctbr` mode accepts normalized collective-thrust and body-rate
commands in the order `[thrust, roll_rate, pitch_rate, yaw_rate]`.

The differentiable `velocity_yaw_rate` mode accepts normalized commands in the
order `[forward_velocity, left_velocity, vertical_velocity, yaw_rate]`:

```python
cfg = CrazyflieConfig(
    scenario="single",
    control_mode="velocity_yaw_rate",
    use_motor_dynamics=True,
)
env = make_env(config=cfg)
```

The same mode is available from the demo CLI:

```bash
uv run --extra cuda13 python -m crazyflie_mjx_sim.demo \
  --scenario single \
  --control-mode velocity_yaw_rate
```

Horizontal velocity is expressed in the yaw-aligned heading frame and vertical
velocity is expressed along world Z. The default normalized scales are 1 m/s
horizontal, 0.5 m/s vertical, and 200 degrees/s yaw rate. The outer velocity
and attitude controller produces CTBR commands that continue through the
existing body-rate PID, motor allocation, motor lag, and MJX dynamics. This
mode requires `use_motor_dynamics=True` and remains compatible with `jax.jit`,
`jax.vmap`, and automatic differentiation through `step_rollout`.

Controller limits and gains can be tuned with `velocity_scale_xy`,
`velocity_scale_z`, `velocity_yaw_rate_scale`, `velocity_kp_xy`,
`velocity_kp_z`, `velocity_attitude_kp`, `velocity_max_accel_xy`,
`velocity_max_accel_z`, and `velocity_max_tilt` on `CrazyflieConfig`.
`step_env` exposes the generated normalized low-level command as
`info["ctbr_action"]`. Gradients flow through the controller, motor dynamics,
and MJX step; clipping, saturation, and contact transitions retain their usual
piecewise-smooth boundary behavior.

## Observations

`single` observation, shape `[1, 12]`:

- world position
- body-frame linear velocity
- body-frame angular velocity
- gravity direction in the body frame

`multi_reach_avoid` observation, shape
`[num_agents, 12 + 3*num_goals + 3*num_obstacles + 3*(num_agents-1) + num_agents]`:

- the same 12 base features
- local vectors to all goals
- local vectors to all obstacles
- local vectors to all other agents
- agent id one-hot

## Batched Simulation

```python
import jax
import jax.numpy as jnp

from crazyflie_mjx_sim import CrazyflieConfig, make_env

env = make_env(
    "multi_reach_avoid",
    CrazyflieConfig(scenario="multi_reach_avoid", num_agents=4, num_goals=4, num_obstacles=2),
)

keys = jax.random.split(jax.random.PRNGKey(0), 128)
reset_batch = jax.jit(jax.vmap(env.reset))
step_batch = jax.jit(jax.vmap(env.step_rollout, in_axes=(0, 0, 0)))

obs, state = reset_batch(keys)
actions = jnp.zeros((128, env.num_agents, env.action_size))
obs, state = step_batch(keys, state, actions)
```

Scenario entity counts are static after `make_env(...)` so JAX can compile
efficient batched programs.

## Demo CLI

```bash
uv run python -m crazyflie_mjx_sim.demo --scenario single
```

```bash
uv run python -m crazyflie_mjx_sim.demo \
  --scenario multi_reach_avoid \
  --num-agents 4 \
  --num-goals 4 \
  --num-obstacles 3 \
  --horizon 500 \
  --batch-size 8 \
  --render-mode both \
  --output-dir artifacts/multi_demo
```

Main scenario/runtime knobs:

```text
--horizon
--timestep
--num-agents
--num-goals
--num-obstacles
--seed
--batch-size
--use-motor-dynamics / --no-use-motor-dynamics
--randomize-dynamics / --no-randomize-dynamics
--control-mode ctbr|velocity_yaw_rate
```

Visualization and collision defaults are on. For high-throughput rollouts:

```bash
uv run python -m crazyflie_mjx_sim.demo \
  --scenario multi_reach_avoid \
  --disable-visualization \
  --disable-collisions \
  --render-mode none \
  --batch-size 1024
```

World-bound collision boxes are opt-in:

```bash
uv run python -m crazyflie_mjx_sim.demo \
  --scenario multi_reach_avoid \
  --enable-world-bound-collisions \
  --world-bound-collision-thickness 0.05
```

## Rendering

### MuJoCo RGB

```python
frame = env.render(state.data, camera="overview")
```

The browser helper accepts the public form
`generate_web_report(out_dir, env, rollout_payloads, summary)`.

Or via CLI:

```bash
uv run python -m crazyflie_mjx_sim.demo \
  --scenario single \
  --render-mode mujoco \
  --output-dir artifacts/single_rgb
```

The output frame is written to `artifacts/single_rgb/mujoco_frame.png`.
`mujoco.Renderer` is created lazily only when `render(...)` is called.

### Web Report

```bash
uv run python -m crazyflie_mjx_sim.demo \
  --scenario multi_reach_avoid \
  --batch-size 4 \
  --horizon 300 \
  --render-mode web \
  --output-dir artifacts/multi_web
```

The report is self-contained under:

```text
artifacts/multi_web/web_report/
```

It contains `index.html`, `report_data.js`, CF2 OBJ assets, and Three.js vendor
modules when the machine can download them. Serve it with:

```bash
uv run python -m http.server 8000 -d artifacts/multi_web/web_report
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

The CLI can also serve after generating:

```bash
uv run python -m crazyflie_mjx_sim.demo \
  --scenario single \
  --render-mode web \
  --serve \
  --serve-port 8000
```

## XLA Runtime Guidance

The demo CLI sets runtime environment variables before importing JAX.

GPU defaults:

```text
MUJOCO_GL=egl
JAX_PLATFORMS=cuda
XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async
XLA_PYTHON_CLIENT_PREALLOCATE=false
XLA_PYTHON_CLIENT_MEM_FRACTION=.99
```

Relevant CLI knobs:

```text
--jax-platform gpu|cpu
--cuda-visible-devices 0
--xla-python-client-allocator cuda_async
--xla-mem-fraction .99
--xla-triton-gemm-any / --no-xla-triton-gemm-any
--enable-persistent-cache / --no-enable-persistent-cache
--jax-cache-dir ~/.cache/jax/crazyflie_mjx_sim
--xla-flags-extra "..."
```

Persistent JAX compilation cache is enabled by default for GPU mode. The optional
Triton GEMM flag maps to:

```text
--xla_gpu_triton_gemm_any=True
```

No XLA command-buffer workaround is set by default. Pass any special workaround
explicitly through `--xla-flags-extra`.

## MJX Optimizer Guidance

Fast defaults:

```text
--mj-solver cg
--mj-iterations 1
--mj-ls-iterations 4
--mj-tolerance 0
--mj-ls-tolerance 0
--use-euler-integrator
```

More accurate or debug-oriented settings:

```bash
uv run python -m crazyflie_mjx_sim.demo \
  --mj-solver newton \
  --mj-iterations 8 \
  --mj-ls-iterations 8 \
  --no-use-euler-integrator
```

For large batched throughput runs, `--disable-visualization` strips visual geoms
before MJX compilation, and `--disable-collisions` strips collision contacts.

## Examples

```bash
uv run python examples/simple_step.py
uv run python examples/batched_simulation.py
uv run python examples/mujoco_render.py
uv run python examples/web_report.py
```

## Tests

```bash
uv run python -m py_compile $(find src examples -name '*.py')
JAX_PLATFORMS=cpu MUJOCO_GL=egl uv run pytest -q
```

The tests cover single and multi-agent CPU stepping, batched reset/rollout with
`jax.jit(jax.vmap(...))`, MuJoCo RGB rendering, web report artifact generation,
and a source-surface check that rejects copied training/STL terms.
