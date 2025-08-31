import sys
import os

# Add everything in /api/ to the module search path.
__path__.append(os.path.join(os.path.dirname(__file__), "api"))  # noqa: F405

from aistudio.api import *  # noqa: F403, E402

current_dir = os.path.dirname(__file__)
repo_root = current_dir

while not os.path.exists(os.path.join(repo_root, "aistudio")):
    parent = os.path.dirname(repo_root)
    if parent == repo_root:
        raise RuntimeError("Cannot find repo root containing 'aistudio/'")
    repo_root = parent

build_path = os.path.join(repo_root, "build", "bindings")

if build_path not in sys.path:
    sys.path.insert(0, build_path)

del os, sys, current_dir, repo_root, parent, build_path
