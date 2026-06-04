from __future__ import annotations

import argparse
import os


def append_xla_flag(flag: str) -> None:
    flags = os.environ.get("XLA_FLAGS", "")
    parts = flags.split()
    if flag not in parts:
        os.environ["XLA_FLAGS"] = f"{flags} {flag}".strip()


def add_runtime_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--jax-platform", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--cuda-visible-devices", type=str, default=None)
    parser.add_argument("--xla-python-client-allocator", type=str, default="cuda_async")
    parser.add_argument("--xla-mem-fraction", type=str, default=".99")
    parser.add_argument("--xla-triton-gemm-any", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-persistent-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jax-cache-dir", type=str, default="/tmp/jax_cache")
    parser.add_argument(
        "--xla-flags-extra",
        type=str,
        default="",
        help="Explicit extra XLA_FLAGS. No command-buffer workaround is added unless passed here.",
    )


def configure_runtime(args: argparse.Namespace) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.jax_platform == "gpu":
        os.environ.setdefault("JAX_PLATFORMS", "cuda")
        os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", args.xla_python_client_allocator)
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", args.xla_mem_fraction)
    else:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if args.xla_triton_gemm_any:
        append_xla_flag("--xla_gpu_triton_gemm_any=True")
    if args.xla_flags_extra:
        for flag in args.xla_flags_extra.split():
            append_xla_flag(flag)


def configure_jax_cache(args: argparse.Namespace) -> None:
    if not args.enable_persistent_cache:
        return
    import jax

    jax.config.update("jax_compilation_cache_dir", args.jax_cache_dir)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
