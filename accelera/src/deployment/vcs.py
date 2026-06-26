import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from accelera.src.config import config

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def _path(name):
    val = globals().get(name)
    if val is not None:
        return val
    if name == "project_root":
        return str(config.deployment_root)
    if name == "experiments_dir":
        return str(config.deployment_experiments_dir)
    if name == "models_dir":
        return str(config.deployment_models_dir)
    if name == "config_file":
        return str(config.deployment_config_file)
    if name == "index_file":
        return str(config.deployment_index_file)
    raise AttributeError(name)


def __getattr__(name):
    try:
        return _path(name)
    except AttributeError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


###helpers
def calc_hash(t, message):
    encoded = f"{t}{message}".encode()
    return hashlib.sha1(encoded).hexdigest()[:7]


def resolve_hash(index, short_hash):
    return [c for c in index["commits"] if c["hash"].startswith(short_hash)][0]


def save_index(index):
    index_file = _path("index_file")
    with open(index_file, "w") as f:
        json.dump(index, f)


def load_index():
    index_file = _path("index_file")
    if not os.path.exists(index_file):
        print(f"Index file not found at {index_file}")
        raise SystemExit(1)
    with open(index_file, "r") as f:
        return json.load(f)


################### Commands
def init(args):
    if globals().get("index_file") is not None:
        index_file = globals()["index_file"]
        experiments_dir = globals()["experiments_dir"]
        if os.path.exists(index_file):
            print("Deployment module initialized already")
            return
        os.makedirs(experiments_dir, exist_ok=True)
        save_index({"head": None, "deployed": None, "commits": []})
        print(f"Deployment initialized at {experiments_dir}")
        return

    experiments_dir = _path("experiments_dir")
    index_file = _path("index_file")
    if os.path.exists(index_file):
        print("Deployment module already initialized")
        return
    os.makedirs(experiments_dir, exist_ok=True)
    with open(index_file, "w") as f:
        json.dump({"head": None, "deployed": None, "commits": []}, f)
    print(f"Deployment initialized at {experiments_dir}")


def commit(args):
    message = args.message
    if not message:
        print("Commit mesage is required")
        raise SystemExit(1)

    config_file = _path("config_file")
    experiments_dir = _path("experiments_dir")

    if not os.path.exists(config_file):
        print(f"Config file not found at {config_file}")
        raise SystemExit(1)

    try:
        with open(config_file, "r") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"Failed to parse config file: {e}")
        raise SystemExit(1)

    models_cfg = cfg.get("models", {})
    if not isinstance(models_cfg, dict):
        print("Invalid config 'models' should be a dict")
        raise SystemExit(1)

    config_dir = os.path.dirname(os.path.abspath(config_file))
    resolved_models = {}
    missing_paths = []
    for name, path in models_cfg.items():
        abs_path = (
            os.path.join(config_dir, path) if not os.path.isabs(path) else path
        )
        abs_path = os.path.abspath(abs_path)
        if not os.path.isfile(abs_path):
            missing_paths.append(f"{name}: {path} (resolved: {abs_path})")
        else:
            resolved_models[name] = abs_path

    if missing_paths:
        print(f"no exist model at this path {', '.join(missing_paths)}")
        raise SystemExit(1)

    index = load_index()
    timestamp = datetime.now().isoformat()

    commit_hash = calc_hash(timestamp, message)
    existing = {c["hash"] for c in index["commits"]}
    while commit_hash in existing:
        commit_hash = calc_hash(timestamp, message + commit_hash)

    commit_dir = os.path.join(experiments_dir, commit_hash)
    os.makedirs(commit_dir, exist_ok=True)

    dest_models = os.path.join(commit_dir, "models")
    os.makedirs(dest_models, exist_ok=True)

    new_cfg = dict(cfg)
    new_models_cfg = {}
    for name, abs_path in resolved_models.items():
        filename = os.path.basename(abs_path)
        dest_path = os.path.join(dest_models, filename)
        shutil.copy2(abs_path, dest_path)
        new_models_cfg[name] = f"models/{filename}"

    new_cfg["models"] = new_models_cfg
    with open(os.path.join(commit_dir, "config.json"), "w") as f:
        json.dump(new_cfg, f, indent=2)

    parent = index["head"]
    metadata = {
        "hash": commit_hash,
        "message": message,
        "timestamp": timestamp,
        "parent": parent,
    }
    with open(os.path.join(commit_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)

    index["commits"].append(
        {
            "hash": commit_hash,
            "message": message,
            "timestamp": timestamp,
            "parent": parent,
        }
    )

    index["head"] = commit_hash
    save_index(index)

    print(f"{commit_hash} {message}")

    model_files = os.listdir(dest_models)
    print(
        {f"config.json + {len(model_files)} model files: {', '.join(model_files)} "}
    )


def log(args):
    index = load_index()

    if not index["commits"]:
        print("no commits yet")
        return

    deployed = index.get("deployed")

    for commit in reversed(index["commits"]):
        tag = ""
        if commit["hash"] == index["head"]:
            tag += "(HEAD)"
        if commit["hash"] == deployed:
            tag += "(deployed)"

        print(f"commit {commit['hash']} {tag}")
        print(f"Date: {commit['timestamp']}")
        print(commit["message"])


def show(args):
    index = load_index()
    commit = resolve_hash(index, args.hash)
    experiments_dir = _path("experiments_dir")
    commit_dir = os.path.join(experiments_dir, commit["hash"])

    deployed = index.get("deployed")
    tag = ""
    if commit["hash"] == index["head"]:
        tag += "(HEAD)"
    if commit["hash"] == deployed:
        tag += "(deployed)"

    print(f"commit {commit['hash']} {tag}")
    print(f"Date: {commit['timestamp']}")
    print(commit["message"])

    config_path = os.path.join(commit_dir, "config.json")
    if os.path.exists(config_path):
        print("Config:")
        with open(config_path, "r") as f:
            config = json.load(f)
        print(json.dumps(config))

    models_path = os.path.join(commit_dir, "models")
    if os.path.isdir(models_path):
        print("model files:")
        for fname in sorted(os.listdir(models_path)):
            print(os.path.join(models_path, fname))


def deploy(args):
    index = load_index()
    commit = resolve_hash(index, args.hash)
    experiments_dir = _path("experiments_dir")
    commit_dir = os.path.join(experiments_dir, commit["hash"])

    config = os.path.join(commit_dir, "config.json")
    models = os.path.join(commit_dir, "models")

    config_file = _path("config_file")
    models_dir = _path("models_dir")

    shutil.copy2(config, config_file)

    if os.path.exists(models_dir):
        shutil.rmtree(models_dir)
    shutil.copytree(models, models_dir)

    index["deployed"] = commit["hash"]
    save_index(index)

    print(f"deployed {commit['hash']}")
    print("run deployment.py to build and start the container")


def status(args):
    index = load_index()

    head = index["head"]
    deployed = index["deployed"]

    print(f"commits number: {len(index['commits'])}")

    if head:
        head_commit = None

        for c in index["commits"]:
            if c["hash"] == head:
                head_commit = c
                break

        if head_commit:
            print(f"Head: {head_commit['hash']} {head_commit['message']}")
        else:
            print("HEAD: None")

    if deployed:
        dep_comit = next(
            (c for c in index["commits"] if c["hash"] == deployed), None
        )
        if dep_comit:
            print(f"Deployed: {dep_comit['hash']} {dep_comit['message']}")
        else:
            print("Deployed: None")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python experiment.py init
  python experiment.py commit -m "commit message"
  python experiment.py log
  python experiment.py deploy <commit hash>
  python experiment.py show <commit hash>
  python experiment.py status
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="available commands")
    subparsers.add_parser("init", help="init directory")

    commit_parser = subparsers.add_parser(
        "commit", help="save curr models and config file"
    )
    commit_parser.add_argument(
        "-m", "--message", required=True, help="commit message"
    )

    subparsers.add_parser("log", help="show all commits")

    show_parser = subparsers.add_parser(
        "show", help="show details of a specific commit hash"
    )
    show_parser.add_argument("hash", help="commit hash")

    deploy_parser = subparsers.add_parser(
        "deploy", help="Restore this commit to deploy"
    )
    deploy_parser.add_argument("hash", help="commit hash")

    subparsers.add_parser("status", help="Show current HEAD and deployed commit")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "init": init,
        "commit": commit,
        "log": log,
        "show": show,
        "deploy": deploy,
        "status": status,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
