from __future__ import annotations

from functools import partial

import chex
import jax
import jax.numpy as jnp
import numpy as np


def build_grid_offsets(count: int) -> jnp.ndarray:
    if count <= 0:
        return jnp.zeros((0, 2))
    rows = int(np.ceil(np.sqrt(count)))
    cols = int(np.ceil(count / rows))
    x_vals = np.zeros((1,)) if cols <= 1 else np.linspace(-1.0, 1.0, cols)
    y_vals = np.zeros((1,)) if rows <= 1 else np.linspace(-1.0, 1.0, rows)
    grid_x, grid_y = np.meshgrid(x_vals, y_vals, indexing="xy")
    return jnp.asarray(np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)[:count])


def _sample_centers_uniform(key, n, halfsizes_eff, world_rect):
    dim = len(world_rect) // 2
    keys = jax.random.split(key, dim)
    coords = []
    for d in range(dim):
        min_val = world_rect[2 * d]
        max_val = world_rect[2 * d + 1]
        coords.append(jax.random.uniform(keys[d], (n,), minval=min_val + halfsizes_eff, maxval=max_val - halfsizes_eff))
    return jnp.stack(coords, axis=-1)


def _valid_squares_with_gaps(centers, halfsizes, pair_gap=0.0, eps=1e-6):
    diff = centers[:, None, :] - centers[None, :, :]
    d_inf = jnp.max(jnp.abs(diff), axis=-1) + jnp.eye(centers.shape[0]) * 1e9
    req = halfsizes[:, None] + halfsizes[None, :] + pair_gap
    return jnp.all(d_inf >= (req + eps))


def _relax(centers, halfsizes, pair_gap, iters, step_scale=0.3, eps=1e-6):
    n = centers.shape[0]
    req = halfsizes[:, None] + halfsizes[None, :] + pair_gap

    def body(_, c):
        diff = c[:, None, :] - c[None, :, :]
        adiff = jnp.abs(diff)
        d_inf = jnp.max(adiff, axis=-1) + jnp.eye(n) * 1e9
        overlap = jnp.maximum(0.0, req + eps - d_inf)
        axis = jnp.argmax(adiff, axis=-1)
        axis_one_hot = jax.nn.one_hot(axis, c.shape[-1])
        direction = axis_one_hot * jnp.sign(diff)
        disp = jnp.sum(0.5 * overlap[..., None] * direction, axis=1)
        return c + step_scale * disp

    return jax.lax.fori_loop(0, iters, body, centers)


def _clamp_inside(centers, halfsizes, world_rect, wall_gap):
    dim = centers.shape[-1]
    eff = halfsizes + wall_gap
    coords = []
    for d in range(dim):
        min_val = world_rect[2 * d]
        max_val = world_rect[2 * d + 1]
        coords.append(jnp.clip(centers[:, d], min_val + eff, max_val - eff))
    return jnp.stack(coords, axis=-1)


@partial(jax.jit, static_argnums=(4, 5, 6))
def sample_non_overlapping_squares(
    key: chex.PRNGKey,
    halfsizes: chex.Array,
    world_rect: chex.Array,
    pair_gap: float = 0.15,
    attempts: int = 80,
    relax_iters: int = 80,
    wall_gap: float = 0.0,
) -> chex.Array:
    n = halfsizes.shape[0]
    keys = jax.random.split(key, attempts)

    def one_attempt(k):
        c0 = _sample_centers_uniform(k, n, halfsizes + wall_gap, world_rect)
        c1 = _relax(c0, halfsizes, pair_gap, relax_iters // 2)
        c1 = _clamp_inside(c1, halfsizes, world_rect, wall_gap)
        c2 = _relax(c1, halfsizes, pair_gap, relax_iters // 2)
        return _clamp_inside(c2, halfsizes, world_rect, wall_gap)

    def scan_body(carry, i):
        chosen, ok = carry
        candidate = one_attempt(keys[i])
        candidate_ok = _valid_squares_with_gaps(candidate, halfsizes, pair_gap)
        chosen = jax.lax.select(ok, chosen, candidate)
        return (chosen, jnp.logical_or(ok, candidate_ok)), None

    init = jnp.zeros((n, len(world_rect) // 2), dtype=halfsizes.dtype)
    (chosen, ok), _ = jax.lax.scan(scan_body, (init, jnp.array(False)), jnp.arange(attempts))
    return jax.lax.select(ok, chosen, one_attempt(keys[-1]))
