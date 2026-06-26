import keyword
import os
import shutil
import sys
from pathlib import Path

from accelera.src.config import config

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


skip_names = config.INIT_GENERATOR_SKIP_NAMES | frozenset(
    {
        ".accelera_cache",
        ".cache",
        ".coverage",
        ".eggs",
        ".ipynb_checkpoints",
        ".mypy_cache",
        ".nox",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "AutogluonModels",
        "bin",
        "build",
        "cache",
        "catboost_info",
        "dist",
        "env",
        "experiments",
        "htmlcov",
        "lib",
        "lib64",
        "logs",
        "models",
        "node_modules",
        "open_mp_generator",
        "report",
        "venv",
    }
)
template_header = config.INIT_GENERATOR_TEMPLATE_HEADER


def should_skip_name(name):
    return (
        name in skip_names
        or name.startswith(".")
        or name.endswith((".egg-info", ".dist-info"))
    )


def should_skip_module(name):
    return name.endswith("_test") or name.startswith("test_")


def get_structure(root):
    """Returns a set of relative directory paths under root"""
    dirs_set = set()
    for dirpath, dirnames, _ in os.walk(root):
        dirnames[:] = [
            dirname for dirname in dirnames if not should_skip_name(dirname)
        ]
        rel = os.path.relpath(dirpath, root)
        if rel != ".":
            dirs_set.add(rel)
    return dirs_set


def remove_ignored_dirs(root):
    for dirpath, dirnames, _ in os.walk(root, topdown=True):
        ignored = [dirname for dirname in dirnames if should_skip_name(dirname)]
        for dirname in ignored:
            full_path = os.path.join(dirpath, dirname)
            print(f"[remove ignored] {full_path}")
            shutil.rmtree(full_path)
        dirnames[:] = [
            dirname for dirname in dirnames if not should_skip_name(dirname)
        ]


def is_valid_identifier(name):
    return name.isidentifier() and not keyword.iskeyword(name)


def is_importable_path(path):
    return all(is_valid_identifier(part) for part in path.split(os.sep))


def generate_init_for_dir(src_dir, api_dir, src_package_prefix, is_root=False):
    lines = [template_header]
    src_rel = os.path.relpath(src_dir, "accelera/src")
    can_import_from_dir = src_rel == "." or is_importable_path(src_rel)

    for entry in sorted(os.listdir(src_dir)):
        if should_skip_name(entry):
            continue

        src_path = os.path.join(src_dir, entry)
        name, ext = os.path.splitext(entry)

        # Skip non-Python files, hidden files, and __init__.py
        if (
            ext != ".py"
            or name.startswith("_")
            or name == "__init__"
            or should_skip_module(name)
            or not can_import_from_dir
            or not is_valid_identifier(name)
        ):
            continue

        # Only import version.py at root
        if is_root and name == "version":
            lines.append(
                "from accelera.src.version import __version__ as __version__\n"
            )
        else:
            lines.append(f"from {src_package_prefix} import {name} as {name}\n")

    # Import subpackages
    for entry in sorted(os.listdir(src_dir)):
        if should_skip_name(entry):
            continue
        src_path = os.path.join(src_dir, entry)
        if (
            os.path.isdir(src_path)
            and not entry.startswith("_")
            and can_import_from_dir
            and is_valid_identifier(entry)
        ):
            lines.append(f"from {src_package_prefix} import {entry} as {entry}\n")

    os.makedirs(api_dir, exist_ok=True)
    init_file = os.path.join(api_dir, "__init__.py")
    with open(init_file, "w") as f:
        f.writelines(lines)
    print(f"[generated] {init_file}")


def sync_api_with_src(src_root, api_root):
    remove_ignored_dirs(api_root)

    # Remove extra dirs in API
    src_dirs = get_structure(src_root)
    api_dirs = get_structure(api_root)

    for extra in api_dirs - src_dirs:
        full_path = os.path.join(api_root, extra)
        print(f"[remove] {full_path}")
        shutil.rmtree(full_path)

    # Generate __init__.py for all src dirs including root
    for dir_rel in src_dirs | {""}:
        src_dir = os.path.join(src_root, dir_rel)
        api_dir = os.path.join(api_root, dir_rel)
        src_package_prefix = "accelera.src" + (
            f".{dir_rel.replace(os.sep, '.')}" if dir_rel else ""
        )
        generate_init_for_dir(
            src_dir, api_dir, src_package_prefix, is_root=(dir_rel == "")
        )


if __name__ == "__main__":
    sync_api_with_src("accelera/src", "accelera/api")
