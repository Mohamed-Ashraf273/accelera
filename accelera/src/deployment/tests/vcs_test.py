import json
import sys
from types import SimpleNamespace

import pytest

from accelera.src.deployment import vcs


@pytest.fixture
def isolated_vcs(tmp_path, monkeypatch):
    deployment_root = tmp_path / "deployment"
    experiments_dir = deployment_root / "experiments"
    models_dir = deployment_root / "models"
    config_file = deployment_root / "config.json"
    index_file = experiments_dir / "experiments.json"

    monkeypatch.setattr(vcs, "project_root", str(deployment_root))
    monkeypatch.setattr(vcs, "experiments_dir", str(experiments_dir))
    monkeypatch.setattr(vcs, "models_dir", str(models_dir))
    monkeypatch.setattr(vcs, "config_file", str(config_file))
    monkeypatch.setattr(vcs, "index_file", str(index_file))
    return SimpleNamespace(
        root=deployment_root,
        experiments_dir=experiments_dir,
        models_dir=models_dir,
        config_file=config_file,
        index_file=index_file,
    )


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def test_calculate_hash_is_stable_short_sha():
    hah_value = vcs.calc_hash("2026-06-21T00:00:00", "deploy")
    assert hah_value == vcs.calc_hash("2026-06-21T00:00:00", "deploy")
    assert len(hah_value) == 7
    assert vcs.calc_hash("t1", "deploy") != vcs.calc_hash("t2", "deploy")


def test_resolv_hash_matches_short_prefix():
    index = {
        "commits": [
            {"hash": "abcdef1", "message": "first"},
            {"hash": "1234567", "message": "second"},
        ]
    }

    assert vcs.resolve_hash(index, "abc")["message"] == "first"


def test_resolve_hash_raisesfor_unknown_prefix():
    with pytest.raises(IndexError):
        vcs.resolve_hash({"commits": []}, "missing")


def test_save_and_load_index_round_trip(isolated_vcs):
    index = {
        "head": "abc",
        "deployed": None,
        "commits": [{"hash": "abc", "message": "saved"}],
    }
    isolated_vcs.experiments_dir.mkdir(parents=True)

    vcs.save_index(index)

    assert vcs.load_index() == index


def test_init_creates_empty_inex_and_is_idempotent(isolated_vcs, capsys):
    vcs.init(SimpleNamespace())

    assert read_json(isolated_vcs.index_file) == {
        "head": None,
        "deployed": None,
        "commits": [],
    }
    assert "Deployment initialized" in capsys.readouterr().out

    vcs.init(SimpleNamespace())

    assert "initialized already" in capsys.readouterr().out


def test_commit_snapshots_conig_models_and_updates_head(isolated_vcs):
    vcs.init(SimpleNamespace())
    write_json(
        isolated_vcs.config_file,
        {"models": {"main": "models/model.pkl", "scaler": "models/scaler.pkl"}},
    )
    isolated_vcs.models_dir.mkdir()
    (isolated_vcs.models_dir / "model.pkl").write_bytes(b"model")
    (isolated_vcs.models_dir / "scaler.pkl").write_bytes(b"scaler")

    vcs.commit(SimpleNamespace(message="first model"))

    index = read_json(isolated_vcs.index_file)
    assert len(index["commits"]) == 1
    commit = index["commits"][0]
    assert index["head"] == commit["hash"]
    assert commit["message"] == "first model"
    assert commit["parent"] is None

    commit_dir = isolated_vcs.experiments_dir / commit["hash"]
    assert read_json(commit_dir / "config.json") == {
        "models": {"main": "models/model.pkl", "scaler": "models/scaler.pkl"}
    }
    assert (commit_dir / "models" / "model.pkl").read_bytes() == b"model"
    assert (commit_dir / "models" / "scaler.pkl").read_bytes() == b"scaler"
    assert read_json(commit_dir / "metadata.json")["hash"] == commit["hash"]


def test_second_commit_records_previos_head_as_parent(isolated_vcs):
    vcs.init(SimpleNamespace())
    write_json(isolated_vcs.config_file, {"version": 1})
    isolated_vcs.models_dir.mkdir()
    (isolated_vcs.models_dir / "model.pkl").write_bytes(b"model-v1")

    vcs.commit(SimpleNamespace(message="first"))
    first_hash = read_json(isolated_vcs.index_file)["head"]

    write_json(isolated_vcs.config_file, {"version": 2})
    (isolated_vcs.models_dir / "model.pkl").write_bytes(b"model-v2")
    vcs.commit(SimpleNamespace(message="second"))

    index = read_json(isolated_vcs.index_file)
    assert len(index["commits"]) == 2
    assert index["commits"][1]["parent"] == first_hash
    assert index["head"] == index["commits"][1]["hash"]


def test_commit_requires_message_confg_and_models(isolated_vcs, capsys):
    with pytest.raises(SystemExit):
        vcs.commit(SimpleNamespace(message=""))
    assert "Commit mesage is required" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        vcs.commit(SimpleNamespace(message="missing config"))
    assert "Config file not found" in capsys.readouterr().out

    write_json(isolated_vcs.config_file, {"models": {"main": "models/missing.pkl"}})
    with pytest.raises(SystemExit):
        vcs.commit(SimpleNamespace(message="missing models"))
    assert "no exist model" in capsys.readouterr().out


def test_deploy_restres_config_models_and_marks_deployed(isolated_vcs):
    commit_hash = "abcdef1"
    commit_dir = isolated_vcs.experiments_dir / commit_hash
    write_json(
        isolated_vcs.index_file,
        {
            "head": commit_hash,
            "deployed": None,
            "commits": [
                {
                    "hash": commit_hash,
                    "message": "ready",
                    "timestamp": "2026-06-20T00:00:00",
                    "parent": None,
                }
            ],
        },
    )
    write_json(commit_dir / "config.json", {"active": True})
    (commit_dir / "models").mkdir(parents=True)
    (commit_dir / "models" / "model.pkl").write_bytes(b"deployed-model")
    write_json(isolated_vcs.config_file, {"active": False})
    isolated_vcs.models_dir.mkdir()
    (isolated_vcs.models_dir / "stale.pkl").write_bytes(b"stale")

    vcs.deploy(SimpleNamespace(hash="abc"))
    assert read_json(isolated_vcs.config_file) == {"active": True}
    assert not (isolated_vcs.models_dir / "stale.pkl").exists()
    assert (isolated_vcs.models_dir / "model.pkl").read_bytes() == b"deployed-model"
    assert read_json(isolated_vcs.index_file)["deployed"] == commit_hash


def test_log_prints_commits_in_reverse_order_with_tags(isolated_vcs, capsys):
    write_json(
        isolated_vcs.index_file,
        {
            "head": "2222222",
            "deployed": "1111111",
            "commits": [
                {
                    "hash": "1111111",
                    "message": "first",
                    "timestamp": "2026-06-20T00:00:00",
                    "parent": None,
                },
                {
                    "hash": "2222222",
                    "message": "second",
                    "timestamp": "2026-06-21T00:00:00",
                    "parent": "1111111",
                },
            ],
        },
    )
    vcs.log(SimpleNamespace())

    output = capsys.readouterr().out
    assert output.index("commit 2222222 (HEAD)") < output.index(
        "commit 1111111 (deployed)"
    )
    assert "second" in output
    assert "first" in output


def test_log_prints_empty_message_when_no_commits(isolated_vcs, capsys):
    write_json(
        isolated_vcs.index_file, {"head": None, "deployed": None, "commits": []}
    )

    vcs.log(SimpleNamespace())

    assert "no commits yet" in capsys.readouterr().out


def test_show_prints_commit_config_and_model_files(isolated_vcs, capsys):
    commit_hash = "abcdef1"
    commit_dir = isolated_vcs.experiments_dir / commit_hash
    write_json(
        isolated_vcs.index_file,
        {
            "head": commit_hash,
            "deployed": commit_hash,
            "commits": [
                {
                    "hash": commit_hash,
                    "message": "ready",
                    "timestamp": "2026-06-21T00:00:00",
                    "parent": None,
                }
            ],
        },
    )
    write_json(commit_dir / "config.json", {"active": True})
    (commit_dir / "models").mkdir(parents=True)
    (commit_dir / "models" / "b.pkl").write_bytes(b"b")
    (commit_dir / "models" / "a.pkl").write_bytes(b"a")
    vcs.show(SimpleNamespace(hash="abc"))
    output = capsys.readouterr().out
    assert "commit abcdef1 (HEAD)(deployed)" in output
    assert '"active": true' in output
    assert output.index("a.pkl") < output.index("b.pkl")


def test_status_prints_head_and_deployed_commits(isolated_vcs, capsys):
    write_json(
        isolated_vcs.index_file,
        {
            "head": "2222222",
            "deployed": "1111111",
            "commits": [
                {"hash": "1111111", "message": "prod"},
                {"hash": "2222222", "message": "latest"},
            ],
        },
    )
    vcs.status(SimpleNamespace())
    output = capsys.readouterr().out
    assert "commits number: 2" in output
    assert "Head: 2222222 latest" in output
    assert "Deployed: 1111111 prod" in output


def test_load_index_exits_when_index_is_mising(isolated_vcs, capsys):
    with pytest.raises(SystemExit):
        vcs.load_index()
    assert "Index file not found" in capsys.readouterr().out


def test_main_dispatches_selected_command(monkeypatch):
    called = []
    monkeypatch.setattr(vcs, "status", lambda args: called.append(args.command))
    monkeypatch.setattr(sys, "argv", ["vcs.py", "status"])
    vcs.main()
    assert called == ["status"]


def test_main_without_command_prints_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["vcs.py"])
    vcs.main()
    assert "available commands" in capsys.readouterr().out
