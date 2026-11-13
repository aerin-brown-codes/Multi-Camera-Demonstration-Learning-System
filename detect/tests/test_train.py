import pytest
import yaml

from detect import train
from conftest import write_image_and_labels


def test_run_folder_that_already_exists_is_never_trained_over(cfg, tmp_path):
    config_file = tmp_path / "train.yaml"
    config_file.write_text(yaml.safe_dump(cfg))
    run_dir = tmp_path / "runs" / "first"
    run_dir.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="already holds a run"):
        train.start(config_file, run_dir)
    assert list(run_dir.iterdir()) == []


class ToolThatOnlyNotesItsSettings:
    def __init__(self, weights):
        pass

    def train(self, **settings):
        ToolThatOnlyNotesItsSettings.settings = settings


@pytest.fixture
def started_run(cfg, tmp_path, monkeypatch):
    for source in ("sim", "real"):
        for n in range(4):
            write_image_and_labels(tmp_path / source, "train", f"{n:03d}", [(0, (10, 10, 50, 50))])
        write_image_and_labels(tmp_path / source, "check", "000", [])
    config_file = tmp_path / "train.yaml"
    config_file.write_text(yaml.safe_dump({**cfg, "model": str(tmp_path / "weights" / "start.pt"), "image_size": 864,
                                           "epochs": 50, "patience": 15, "batch": 16, "workers": 8, "device": "auto"}))
    monkeypatch.setattr(train, "YOLO", ToolThatOnlyNotesItsSettings)
    monkeypatch.setattr(train, "device_to_use", lambda asked: f"picked for {asked}")
    monkeypatch.setenv("COMMIT", "abc123")
    run_dir = tmp_path / "runs" / "first"
    train.start(config_file, run_dir)
    return run_dir


def test_tool_is_told_when_to_stop_and_gets_a_device_picked_on_this_machine(started_run):
    settings = ToolThatOnlyNotesItsSettings.settings
    assert settings["patience"] == 15 and settings["epochs"] == 50
    assert settings["device"] == "picked for auto"


def test_run_keeps_the_commit_it_was_trained_from(started_run):
    assert (started_run / "commit.txt").read_text() == "abc123\n"


def test_commit_comes_from_git_where_none_is_given(monkeypatch):
    monkeypatch.delenv("COMMIT", raising=False)
    commit = train.commit_of_the_code()
    assert len(commit) == 40 and set(commit) <= set("0123456789abcdef")
