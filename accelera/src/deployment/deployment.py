import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

current = Path(__file__).resolve()
for parent in current.parents:
    if (parent / "accelera" / "src" / "config.py").exists():
        break

repo_root = parent
sys.path.insert(0, str(repo_root))

from accelera.src.config import config

service_source_dir = Path(__file__).resolve().parent
project_root = str(config.deployment_root if config else service_source_dir)
os.makedirs(project_root, exist_ok=True)
os.chdir(project_root)


def sync_files():
    service_runtime_dir = Path("accelera_deployment")
    service_runtime_dir.mkdir(exist_ok=True)
    for name in (
        "deployment.py",
        "modelservice.py",
        "schema_validation.py",
        "server.py",
        "tracking.py",
    ):
        shutil.copy2(service_source_dir / name, service_runtime_dir / name)

    accelera_pkg_src = Path(config.REPO_ROOT) / "accelera"
    accelera_pkg_dest = service_runtime_dir / "accelera"
    os.makedirs(accelera_pkg_dest, exist_ok=True)
    subprocess.run(
        [
            "rsync",
            "-a",
            "--delete",
            "--exclude",
            "__pycache__",
            "--exclude",
            "*.pyc",
            str(accelera_pkg_src) + "/",
            str(accelera_pkg_dest) + "/",
        ],
        check=True,
    )


def load_configurations():
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def validate_model_paths(configurations):
    models = configurations.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("config.json must contain a 'models' dict ")

    config_file = config.deployment_config_file
    config_dir = os.path.dirname(os.path.abspath(config_file))
    resolved_models = {}
    missing_paths = []
    for name, path in models.items():
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

    local_models_dir = os.path.join(config.deployment_root, "models")
    os.makedirs(local_models_dir, exist_ok=True)

    dest_models = {}
    for name, abs_path in resolved_models.items():
        filename = os.path.basename(abs_path)
        dest_path = os.path.join(local_models_dir, filename)
        if abs_path != dest_path:
            shutil.copy2(abs_path, dest_path)
        dest_models[name] = f"models/{filename}"

    configurations["models"] = dest_models
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(configurations, f, indent=2)

    return dest_models


def validate_port(port):
    port = int(str(port))
    if port < 1 or port > 65535:
        raise ValueError(f"port should be between 1 and 65535 {port}")
    return port


def write_requirements():
    with open("accelera_deployment/requirements.txt", "w", encoding="utf-8") as req:
        req.write("fastapi==0.136.1\n")
        req.write("great-expectations==1.18.1\n")
        req.write("uvicorn[standard]==0.46.0\n")
        req.write("scikit-learn==1.8.0\n")
        req.write("category-encoders==2.9.0\n")
        req.write("numpy==2.4.4\n")
        req.write("pandas==3.0.2\n")
        req.write("pydantic==2.13.3\n")
        req.write("python-multipart==0.0.27\n")
        req.write("joblib\n")
        req.write("lightgbm\n")
        req.write("catboost\n")
        req.write("xgboost\n")
        req.write("ConfigSpace\n")


def write_dockerfile(configurations):
    models = validate_model_paths(configurations)

    with open("Dockerfile", "w", encoding="utf-8") as f:
        f.write("FROM python:3.11-slim\n")
        f.write(
            "RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*\n"
        )
        f.write("WORKDIR /app\n")
        f.write("COPY accelera_deployment/requirements.txt requirements.txt \n")
        f.write(
            "RUN python -m pip install --no-cache-dir --prefer-binary "
            "--timeout 120 --retries 10 -r requirements.txt\n"
        )
        f.write("COPY accelera_deployment/server.py server.py\n")
        f.write("COPY accelera_deployment/modelservice.py modelservice.py\n")
        f.write(
            "COPY accelera_deployment/schema_validation.py schema_validation.py\n"
        )
        f.write("COPY accelera_deployment/tracking.py tracking.py\n")
        f.write("COPY accelera_deployment/accelera/ /app/accelera/\n")
        f.write("COPY config.json config.json\n")
        for pkl in models.values():
            f.write(f"COPY {pkl} /app/{pkl}\n")
        f.write("EXPOSE 8000\n")
        f.write(
            'CMD ["sh", "-c", '
            '"uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]\n'
        )
    print("Dockerfile written sucessfully")


def write_files(_args):
    sync_files()
    configurations = load_configurations()
    write_requirements()
    write_dockerfile(configurations)


def build(args):
    cmd = ["docker", "build"]
    if getattr(args, "no_cache", False):
        cmd.append("--no-cache")
    cmd.extend(["-t", "ml-model", "."])
    subprocess.run(cmd, check=True)


def run_local(_args):
    port = validate_port(os.environ.get("PORT", "8000"))
    running = subprocess.run(
        ["docker", "ps", "-q", "--filter", f"publish={port}"],
        check=True,
        capture_output=True,
        text=True,
    )
    container_ids = running.stdout.split()
    if container_ids:
        subprocess.run(["docker", "stop", *container_ids], check=True)

    print("\n--- Starting container  ---\n")
    print(f" API: http://localhost:{port}")
    print(f" gui: http://localhost:{port}/gui")
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "-p",
            f"{port}:{port}",
            "-e",
            f"PORT={port}",
            "ml-model",
        ],
        check=True,
    )


def local(_args):
    write_files(_args)
    build(_args)
    run_local(_args)


def heroku_login(_args):
    subprocess.run(["heroku", "login"], check=True)


def heroku_create(args):
    subprocess.run(
        ["heroku", "create", args.app, "--stack", "container"], check=True
    )


def heroku_container_login(_args):
    subprocess.run(["heroku", "container:login"], check=True)


def heroku_push(args):
    write_files(args)
    subprocess.run(
        ["heroku", "container:push", "web", "--app", args.app], check=True
    )


def heroku_release(args):
    subprocess.run(
        ["heroku", "container:release", "web", "--app", args.app], check=True
    )
    print(f"GUI: https://{args.app}.herokuapp.com/gui")


def heroku_open(args):
    subprocess.run(["heroku", "open", "--app", args.app], check=True)


def heroku_deploy(args):
    heroku_login(args)
    if args.create:
        heroku_create(args)
    heroku_container_login(args)
    heroku_push(args)
    heroku_release(args)
    heroku_open(args)


def ec2_deploy(args):
    write_files(args)

    target = f"{args.user}@{args.host}"
    remote_root = f"{args.remote_dir.rstrip('/')}/deployment_module"
    quoted_remote_root = shlex.quote(remote_root)
    if remote_root.startswith("~/"):
        quoted_remote_root = "~/" + shlex.quote(remote_root[2:])
    script = remote_script(args)

    run_remote(args, f"mkdir -p {quoted_remote_root}")

    rsync_target = f"{target}:{remote_root}"
    rsync_sources = ["Dockerfile", "config.json", "accelera_deployment", "models"]
    print("Syncing build inputs to EC2...")

    ssh_trans = " ".join(shlex.quote(part) for part in configure_ssh(args))
    subprocess.run(
        [
            "rsync",
            "-av",
            "--info=progress2",
            "--delete",
            "--exclude",
            "__pycache__",
            "--exclude",
            "*.pyc",
            "-e",
            ssh_trans,
            *rsync_sources,
            rsync_target,
        ],
        check=True,
    )

    print("starting container on EC2")
    run_remote(args, script)
    check_ec2_public_url(args)


def ec2_stop(args):
    container_name = args.container
    cmd = f"sudo docker stop {shlex.quote(container_name)} || true"
    run_remote(args, cmd)
    print(f"stopped container '{container_name}'")


def ec2_get_logs(args):
    container_name = args.container
    cmd = f"sudo docker logs -f {shlex.quote(container_name)}"
    run_remote(args, cmd)


def configure_ssh(args):
    command = ["ssh", "-o", "StrictHostKeyChecking=accept-new"]
    if args.key:
        command.extend(["-i", os.path.expanduser(args.key)])
    return command


def run_remote(args, command):
    remote_command = f"bash -lc {shlex.quote(command)}"
    subprocess.run(
        [*configure_ssh(args), f"{args.user}@{args.host}", remote_command],
        check=True,
    )


def check_ec2_public_url(args):
    port = validate_port(args.port)
    url = f"http://{args.host}:{port}/health"
    gui_url = f"http://{args.host}:{port}/gui"

    print("Checking API")
    try:
        with urllib.request.urlopen(url, timeout=8) as response:
            if response.status < 400:
                print("Public health check: OK")
                print(f"GUI URL: {gui_url}")
                return
    except OSError as err:
        print("API health check failed")
        print("You need to open the TCP port")
        print(f"error {err}")


def remote_script(args):
    remote_root = f"{args.remote_dir.rstrip('/')}/deployment_module"
    quoted_remote_root = shlex.quote(remote_root)
    if remote_root.startswith("~/"):
        quoted_remote_root = "~/" + shlex.quote(remote_root[2:])

    port = validate_port(args.port)
    image_name = args.image
    container_name = args.container
    docker_ps_format = (
        "container={.Names} image={.Image} status={.Status} ports={.Ports}"
    )
    build_cmd = ["docker", "build"]
    if getattr(args, "no_cache", False):
        build_cmd.append("--no-cache")
    build_cmd.extend(["-t", image_name, "."])
    docker_build = "sudo " + " ".join(shlex.quote(part) for part in build_cmd)
    health_command = (
        f"sudo docker exec {shlex.quote(container_name)} "
        'python -c "import urllib.request; '
        f"urllib.request.urlopen('http://127.0.0.1:{port}/health', "
        'timeout=3).read()"'
    )
    if args.install_docker:
        docker_setup = """
if ! command -v docker >/dev/null 2>&1; then
  sudo apt-get update && sudo apt-get install -y docker.io || sudo yum install -y docker
  sudo systemctl enable docker && sudo systemctl start docker
fi
"""
    else:
        docker_setup = """
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker not installed on EC2 h Use --install-docker." >&2
  exit 1
fi
"""

    return f"""
set -e
cd {quoted_remote_root}
{docker_setup.strip()}
{docker_build}
sudo docker rm -f {shlex.quote(container_name)} >/dev/null 2>&1 || true
sudo docker run -d --restart unless-stopped --name {shlex.quote(container_name)} -p {port}:{port} -e PORT={port} {shlex.quote(image_name)}
echo "Waiting for health check..."
for attempt in {{1..15}}; do
  if {health_command} >/dev/null 2>&1; then
    echo "Local container health check: OK"
    break
  fi
  sleep 2
done
{health_command} >/dev/null 2>&1 || (echo "health check failed" >&2 && exit 1)
sudo docker ps --filter name={shlex.quote(container_name)} --format {shlex.quote(docker_ps_format)}
echo "Application URL: http://{args.host}:{port}"
echo "GUI URL: http://{args.host}:{port}/gui"
""".strip()


def configure_deployment(model_path: str, df=None, target_col=None) -> None:
    abs_models_path = os.path.abspath(model_path)
    models_dir = str(config.deployment_models_dir)
    os.makedirs(models_dir, exist_ok=True)

    filename = os.path.basename(abs_models_path)
    shutil.copy2(abs_models_path, os.path.join(models_dir, filename))

    config_file = str(config.deployment_config_file)
    config_data = {}
    if os.path.exists(config_file):
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except Exception:
            pass

    config_data["models"] = {"model_path": f"models/{filename}"}

    if df is not None:
        features = []
        for col in df.columns:
            if col != target_col:
                dtype = str(df[col].dtype)
                feat_type = (
                    "integer"
                    if "int" in dtype
                    else ("number" if "float" in dtype else "string")
                )
                features.append({"name": col, "type": feat_type})
        config_data["schema"] = {"features": features}
    else:
        config_data.setdefault("schema", {"features": []})

    config_data.setdefault(
        "tracking", {"enabled": True, "path": "prediction_logs/predictions.jsonl"}
    )

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    orig_cwd = os.getcwd()
    try:
        os.chdir(str(config.deployment_root))
        sync_files()
        write_requirements()
        write_dockerfile(config_data)
    finally:
        os.chdir(orig_cwd)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python accelera_deployment/deployment.py local
  python accelera_deployment/deployment.py heroku-deploy --app accelera1 --create
  python accelera_deployment/deployment.py heroku-push --app accelera1
  python accelera_deployment/deployment.py ec2-deploy --host 1.2.3.4 \\
    --user ec2-user --key ~/.ssh/key.pem
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="avalable commands")
    subparsers.add_parser("prepare", help="write requirements and files")
    build_parser = subparsers.add_parser("build", help="build the Docker image")
    build_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="force Docker to rebuild all layers instead of using the cache",
    )

    subparsers.add_parser("run-local", help="run the local Docker container")

    local_parser = subparsers.add_parser(
        "local", help="prepare, build, and run locally"
    )
    local_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="force Docker to rebuild all layers instead of using the cache",
    )

    subparsers.add_parser("heroku-login", help="run Heroku login")

    create_parser = subparsers.add_parser("heroku-create", help="create Heroku app")
    create_parser.add_argument("--app", default="accelera1", help="Heroku app name")

    subparsers.add_parser(
        "heroku-container-login", help="login to Heroku container registry"
    )

    push_parser = subparsers.add_parser(
        "heroku-push", help="push Docker image to Heroku"
    )
    push_parser.add_argument("--app", default="accelera1", help="Heroku app name")

    release_parser = subparsers.add_parser(
        "heroku-release", help="release web container on Heroku"
    )
    release_parser.add_argument("--app", default="accelera1", help="Heroku app name")

    open_parser = subparsers.add_parser("heroku-open", help="open Heroku app")
    open_parser.add_argument("--app", default="accelera1", help="Heroku app name")

    deploy_parser = subparsers.add_parser(
        "heroku-deploy", help="run full Heroku deployment sequence"
    )
    deploy_parser.add_argument("--app", default="accelera1", help="Heroku app name")
    deploy_parser.add_argument(
        "--create",
        action="store_true",
        help="create the app before pushing if needed",
    )

    ec2_parser = subparsers.add_parser(
        "ec2-deploy",
        help=(
            "sync the deployment module to EC2, build the image, and run the "
            "container"
        ),
    )
    ec2_parser.add_argument("--host", required=True, help="EC2 public IP or DNS")
    ec2_parser.add_argument("--user", default="ec2-user", help="SSH user name")
    ec2_parser.add_argument("--key", help="SSH private key file path")
    ec2_parser.add_argument(
        "--remote-dir",
        default="~/deployment-app",
        help="directory to sync the deployment module into on EC2",
    )
    ec2_parser.add_argument(
        "--port", default="8000", help="public port to expose on EC2"
    )
    ec2_parser.add_argument(
        "--image", default="ml-model", help="Docker image name to build"
    )
    ec2_parser.add_argument(
        "--container",
        default="ml-model",
        help="Docker container name to run on EC2",
    )
    ec2_parser.add_argument(
        "--install-docker",
        action="store_true",
        help="attempt to install and start Docker on the EC2 host if it is missing",
    )
    ec2_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="force Docker to rebuild all layers on EC2 instead of using the cache",
    )

    ec2_stop_parser = subparsers.add_parser(
        "ec2-stop", help="stop the EC2 container"
    )
    ec2_stop_parser.add_argument(
        "--host", required=True, help="EC2 public IP or DNS"
    )
    ec2_stop_parser.add_argument("--user", default="ec2-user", help="SSH user name")
    ec2_stop_parser.add_argument("--key", help="SSH private key file path")
    ec2_stop_parser.add_argument(
        "--container", default="ml-model", help="Docker container name"
    )

    ec2_logs_parser = subparsers.add_parser(
        "ec2-logs", help="tail logs from the EC2 container"
    )
    ec2_logs_parser.add_argument(
        "--host", required=True, help="EC2 public IP or DNS"
    )
    ec2_logs_parser.add_argument("--user", default="ec2-user", help="SSH user name")
    ec2_logs_parser.add_argument("--key", help="SSH private key file path")
    ec2_logs_parser.add_argument(
        "--container", default="ml-model", help="Docker container name"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    commands = {
        "prepare": write_files,
        "build": build,
        "run-local": run_local,
        "local": local,
        "heroku-login": heroku_login,
        "heroku-create": heroku_create,
        "heroku-container-login": heroku_container_login,
        "heroku-push": heroku_push,
        "heroku-release": heroku_release,
        "heroku-open": heroku_open,
        "heroku-deploy": heroku_deploy,
        "ec2-deploy": ec2_deploy,
        "ec2-stop": ec2_stop,
        "ec2-logs": ec2_get_logs,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
