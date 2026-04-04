"""Accelera package root.

This package ships with optional, generated public API modules under `accelera/api`.
Those can pull heavy optional deps (e.g., torch). So we keep API import optional
to avoid breaking basic imports like `from accelera.src.core.parallelizer import Parallelizer`.
"""

from __future__ import annotations

import os

from accelera.src.config import config

# Make pybind11-built modules importable if they're present.
config.ensure_bindings_on_syspath()

# Add everything in /api/ to the module search path.
__path__.append(str(config.api_dir))  # noqa: F405


try:
    from accelera.api import *  # type: ignore  # noqa: F403, E402
except ModuleNotFoundError:
    # Optional dependencies missing (e.g., torch). Keep base package usable.
    if config.STRICT_API_IMPORT:
        raise


del os
