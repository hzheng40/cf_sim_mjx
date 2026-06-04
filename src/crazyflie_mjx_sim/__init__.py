"""Minimal MJX Crazyflie simulation package."""

from .config import CrazyflieConfig


def make_env(*args, **kwargs):
    """Creates a Crazyflie MJX environment.

    The import is intentionally lazy so CLI runtime flags can set JAX/XLA
    environment variables before importing JAX.
    """
    from .env import make_env as _make_env

    return _make_env(*args, **kwargs)


def generate_web_report(*args, **kwargs):
    """Generates a browser playback report for rollout payloads."""
    from .web import generate_web_report as _generate_web_report

    return _generate_web_report(*args, **kwargs)


__all__ = ["CrazyflieConfig", "generate_web_report", "make_env"]
