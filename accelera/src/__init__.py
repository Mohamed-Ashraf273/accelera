# Make the core classes available at the package level
import os

from accelera.src.config import config

# Add the API modules to ensure the public API is accessible at this level
try:
    from accelera.api import *  # noqa: F403
except ImportError:
    pass
