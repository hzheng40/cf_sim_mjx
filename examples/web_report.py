from __future__ import annotations

import subprocess
import sys


cmd = [
    sys.executable,
    "-m",
    "crazyflie_mjx_sim.demo",
    "--jax-platform",
    "cpu",
    "--scenario",
    "multi_reach_avoid",
    "--horizon",
    "100",
    "--batch-size",
    "2",
    "--render-mode",
    "web",
    "--output-dir",
    "artifacts/example_web",
]
subprocess.run(cmd, check=True)
