import importlib
import os
from types import SimpleNamespace

import pytest


@pytest.fixture
def deployment_module(monkeypatch):
    cwd = os.getcwd()
    module = importlib.import_module("accelera.src.deployment.deployment")
    monkeypatch.chdir(cwd)
    return module


def test_valdate_port_accepts_numeric_ports(deployment_module):
    assert deployment_module.validate_port(8000) == 8000
    assert deployment_module.validate_port("65535") == 65535


def test_validate_port_rejects_invalid_ports(deployment_module):
    for port in ["abc", "0", "65536", "-1"]:
        with pytest.raises(ValueError):
            deployment_module.validate_port(port)


def test_validate_mode_paths_accepts_existing_relative_paths(
    deployment_module,
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".accelera_deployment").mkdir()
    (tmp_path / ".accelera_deployment" / "models").mkdir()
    (tmp_path / ".accelera_deployment" / "models" / "model.pkl").write_bytes(
        b"model"
    )

    models = deployment_module.validate_model_paths(
        {"models": {"main": "models/model.pkl"}}
    )

    assert models == {"main": "models/model.pkl"}
    # assert models == {"main": "models/model.pkl"}


def test_validate_model_pats_rejects_bad_model_mapping(
    deployment_module,
):
    with pytest.raises(ValueError, match="models"):
        deployment_module.validate_model_paths({})


def test_validate_model_paths_rejects_mising_relative_paths(
    deployment_module,
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".accelera_deployment").mkdir()

    with pytest.raises(SystemExit):
        deployment_module.validate_model_paths(
            {"models": {"main": "models/missing.pkl"}}
        )


def test_write_files_syncs_service_soures_and_writes_build_files(
    deployment_module,
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".accelera_deployment").mkdir()
    (tmp_path / ".accelera_deployment" / "models").mkdir()
    (tmp_path / ".accelera_deployment" / "models" / "model.pkl").write_bytes(
        b"model"
    )
    (tmp_path / "config.json").write_text(
        '{"models": {"main": "models/model.pkl"}}',
        encoding="utf-8",
    )

    deployment_module.write_files(SimpleNamespace())

    assert (tmp_path / "accelera_deployment" / "server.py").exists()
    assert (tmp_path / "accelera_deployment" / "modelservice.py").exists()
    requirements = (tmp_path / "accelera_deployment" / "requirements.txt").read_text(
        encoding="utf-8"
    )
    dockerfile = (tmp_path / "Dockerfile").read_text(encoding="utf-8")
    assert "fastapi" in requirements
    assert "COPY accelera_deployment/server.py server.py" in dockerfile
    assert "COPY models/model.pkl /app/models/model.pkl" in dockerfile


def test_build_runs_docker_build_command(deployment_module, monkeypatch):
    calls = []

    def fake_run(command, check, **_kwargs):
        calls.append((command, check))

    monkeypatch.setattr(deployment_module.subprocess, "run", fake_run)

    deployment_module.build(SimpleNamespace(no_cache=True))

    assert calls == [
        (
            ["docker", "build", "--no-cache", "-t", "ml-model", "."],
            True,
        )
    ]


def test_run_local_stops_existing_container_and_runs_ne_one(
    deployment_module,
    monkeypatch,
    capsys,
):
    calls = []
    monkeypatch.setenv("PORT", "9000")

    def run(command, check, **kwargs):
        calls.append((command, check, kwargs))
        if command[:2] == ["docker", "ps"]:
            return SimpleNamespace(stdout="abc123\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(deployment_module.subprocess, "run", run)
    deployment_module.run_local(SimpleNamespace())
    assert calls[0][0] == ["docker", "ps", "-q", "--filter", "publish=9000"]
    assert calls[1][0] == ["docker", "stop", "abc123"]
    assert calls[2][0] == [
        "docker",
        "run",
        "--rm",
        "-p",
        "9000:9000",
        "-e",
        "PORT=9000",
        "ml-model",
    ]
    assert "http://localhost:9000/gui" in capsys.readouterr().out


def test_heroku_deploy_runs_steps_in_order(deployment_module, monkeypatch):
    calls = []
    for name in (
        "heroku_login",
        "heroku_create",
        "heroku_container_login",
        "heroku_push",
        "heroku_release",
        "heroku_open",
    ):
        monkeypatch.setattr(
            deployment_module,
            name,
            lambda _args, name=name: calls.append(name),
        )

    deployment_module.heroku_deploy(SimpleNamespace(create=True))
    assert calls == [
        "heroku_login",
        "heroku_create",
        "heroku_container_login",
        "heroku_push",
        "heroku_release",
        "heroku_open",
    ]


def test_remote_helpers_quote_and_build_expected_commands(deployment_module):
    args = SimpleNamespace(
        user="ubuntu",
        host="example.com",
        key="~/.ssh/key.pem",
        remote_dir="~/apps/",
        port="8080",
        image="image-name",
        container="container-name",
        install_docker=False,
        no_cache=True,
    )

    assert "-i" in deployment_module.configure_ssh(args)
    script = deployment_module.remote_script(args)
    assert "cd ~/apps/deployment_module" in script
    assert "sudo docker build --no-cache -t image-name ." in script
    assert "-p 8080:8080" in script
    assert "container-name" in script


def test_run_remote_use_ssh_transport(deployment_module, monkeypatch):
    calls = []

    def fake_run(command, check):
        calls.append((command, check))

    monkeypatch.setattr(deployment_module.subprocess, "run", fake_run)
    args = SimpleNamespace(user="ubuntu", host="example.com", key=None)

    deployment_module.run_remote(args, "echo hi")

    assert calls[0][0][:4] == [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "ubuntu@example.com",
    ]
    assert "echo hi" in calls[0][0][4]
    assert calls[0][1] is True


def test_ec2_deploy_prepares_sync(deployment_module, monkeypatch):
    calls = []
    args = SimpleNamespace(
        user="ubuntu",
        host="example.com",
        key=None,
        remote_dir="~/apps",
        port="8080",
        image="ml-model",
        container="ml-model",
        install_docker=False,
        no_cache=False,
    )
    monkeypatch.setattr(
        deployment_module, "write_files", lambda _args: calls.append("write_files")
    )
    monkeypatch.setattr(
        deployment_module,
        "run_remote",
        lambda _args, command: calls.append(("remote", command)),
    )
    monkeypatch.setattr(
        deployment_module,
        "check_ec2_public_url",
        lambda _args: calls.append("check"),
    )
    monkeypatch.setattr(
        deployment_module.subprocess,
        "run",
        lambda command, check: calls.append(("rsync", command, check)),
    )
    deployment_module.ec2_deploy(args)
    assert calls[0] == "write_files"
    assert calls[1][0] == "remote"
    assert calls[2][0] == "rsync"
    assert calls[3][0] == "remote"
    assert calls[4] == "check"


def test_ec2_stop_and_logs_remote_commands(deployment_module, monkeypatch):
    calls = []
    monkeypatch.setattr(
        deployment_module,
        "run_remote",
        lambda _args, command: calls.append(command),
    )
    args = SimpleNamespace(container="ml-model", host="example.com")
    deployment_module.ec2_stop(args)
    deployment_module.ec2_get_logs(args)
    assert calls == [
        "sudo docker stop ml-model || true",
        "sudo docker logs -f ml-model",
    ]
