import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime

from accelera.src.config import config

project_root = str(config.deployment_root)
experiments_dir = str(config.deployment_experiments_dir)
models_dir = str(config.deployment_models_dir)
config_file = str(config.deployment_config_file)
index_file = str(config.deployment_index_file)


def calculate_hash(t, message):
    encoded = f"{t}{message}".encode()
    return hashlib.sha1(encoded).hexdigest()[:7]


def resolve_hash(index, short_hash):
    return [c for c in index["commits"] if c["hash"].startswith(short_hash)][0]


def save_index(index):
    with open(index_file, "w") as f:
        json.dump(index, f)


def load_index():
    if not os.path.exists(index_file):
        print(f"Index file not found at {index_file}")
        raise SystemExit(1)
    with open(index_file, "r") as f:
        return json.load(f)


def load_config():
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


def get_configured_models(config):
    models = config.get("models")
    if not isinstance(models, dict) or not models:
        print("config.json must contain a models object")
        raise SystemExit(1)
    return models


def resolve_model_path(path):
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def model_snapshot_path(model_name, source_path, used_names):
    file_name = os.path.basename(source_path)
    if file_name not in used_names:
        used_names.add(file_name)
        return file_name

    safe_name = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in model_name
    )
    file_name = f"{safe_name}_{file_name}"
    used_names.add(file_name)
    return file_name


def copy_configured_models(config, commit_dir):
    models = get_configured_models(config)
    dest_models = os.path.join(commit_dir, "models")
    os.makedirs(dest_models, exist_ok=True)

    saved_models = {}
    used_names = set()

    for name, path in models.items():
        source = resolve_model_path(path)
        if not os.path.isfile(source):
            print(f"Model file not found for {name}: {source}")
            raise SystemExit(1)

        file_name = model_snapshot_path(name, source, used_names)
        destination = os.path.join(dest_models, file_name)
        shutil.copy2(source, destination)
        saved_models[name] = os.path.join("models", file_name)

    config["models"] = saved_models
    return dest_models, saved_models


################### Commands
def init(args):
    if os.path.exists(index_file):
        print("Deployment module already initialized")
        return
    os.makedirs(experiments_dir, exist_ok=True)
    save_index({"head": None, "deployed": None, "commits": []})
    print(f"Deployment initialized at {experiments_dir}")


def commit(args):
    message = args.message
    if not message:
        print("Commit Message is required")
        raise SystemExit(1)

    if not os.path.exists(config_file):
        print(f"Config file not found at {config_file}")
        raise SystemExit(1)

    if not os.path.isdir(models_dir):
        print(f"Models directroy not found at {models_dir}")
        raise SystemExit(1)

    index = load_index()
    timestamp = datetime.now().isoformat()

    commit_hash = calculate_hash(timestamp, message)
    existing = {c["hash"] for c in index["commits"]}
    while commit_hash in existing:
        commit_hash = calculate_hash(timestamp, message + commit_hash)

    commit_dir = os.path.join(experiments_dir, commit_hash)
    os.makedirs(commit_dir, exist_ok=True)

    shutil.copy2(config_file, os.path.join(commit_dir, "config.json"))

    dest_models = os.path.join(commit_dir, "models")
    shutil.copytree(models_dir, dest_models)

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

    model_files = sorted(os.listdir(dest_models))
    print(
        {f"config.json + {len(saved_models)} model files: {', '.join(model_files)} "}
    )


def log(args):
    index = load_index()

    if not index["commits"]:
        print("No commits yet")
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
    commit_dir = os.path.join(experiments_dir, commit["hash"])

    config = os.path.join(commit_dir, "config.json")
    models = os.path.join(commit_dir, "models")

    shutil.copy2(config, config_file)

    if os.path.exists(models_dir):
        shutil.rmtree(models_dir)
    shutil.copytree(models, models_dir)

    index["deployed"] = commit["hash"]
    save_index(index)

    print(f"deployed {commit['hash']}")
    print(
        "'python accelera_deployment/deployment.py' to build and start the container"
    )


def status(args):
    index = load_index()

    head = index["head"]
    deployed = index["deployed"]

    print(f"commits number: {len(index['commits'])}")

    if head:
        head_commit = next((c for c in index["commits"] if c["hash"] == head), None)
        if head_commit:
            print(f"Head: {head_commit['hash']} {head_commit['message']}")
        else:
            print("HEAD: None")

    if deployed:
        dep_commit = next(
            (c for c in index["commits"] if c["hash"] == deployed), None
        )
        if dep_commit:
            print(f"Deployed: {dep_commit['hash']} {dep_commit['message']}")
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
        "commit", help="save current models and config file"
    )
    commit_parser.add_argument(
        "-m", "--message", required=True, help="commit message"
    )

    subparsers.add_parser("log", help="show all commits")

    show_parser = subparsers.add_parser(
        "show", help="show details of a specific commit"
    )
    show_parser.add_argument("hash", help="commit hash")

    deploy_parser = subparsers.add_parser(
        "deploy", help="Restore this commit to deploy it"
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
