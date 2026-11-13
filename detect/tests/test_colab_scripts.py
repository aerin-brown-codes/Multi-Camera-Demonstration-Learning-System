import importlib.util
from pathlib import Path

import pytest

from detect.config import REPO

SCRIPTS = Path(__file__).parents[1] / "colab"


def script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run, session = script("run"), script("session")


def test_queue_is_names_with_their_config_files_in_the_order_given():
    assert run.runs_asked_for("run1=train.yaml, run2=run2.yaml") == [("run1", "train.yaml"), ("run2", "run2.yaml")]


def test_what_one_queue_was_asked_does_not_reach_the_next_queue_of_the_session(monkeypatch):
    monkeypatch.setenv("RUNS", "trial=trial.yaml")
    monkeypatch.setenv("KEEP_THE_MACHINE", "1")
    asked = run.asked_and_forgotten_so_the_next_queue_of_this_session_starts_clean()
    assert asked["RUNS"] == "trial=trial.yaml" and asked["KEEP_THE_MACHINE"] == "1" and asked["RESUME"] == ""
    again = run.asked_and_forgotten_so_the_next_queue_of_this_session_starts_clean()
    assert again["KEEP_THE_MACHINE"] == "", "a trial that kept the machine would make the overnight queue keep it too"


def test_queue_with_a_mistyped_config_or_no_commit_is_refused_before_anything_trains(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "CONFIGS", tmp_path)
    (tmp_path / "train.yaml").touch()
    asked = {"RUNS": "run1=train.yaml,run2=run2.yml", "COMMIT": "abc123"}
    with pytest.raises(FileNotFoundError, match="run2.yml"):
        run.queue_that_can_start(asked)
    with pytest.raises(KeyError, match="COMMIT"):
        run.queue_that_can_start({"RUNS": "run1=train.yaml", "COMMIT": ""})
    assert run.queue_that_can_start({"RUNS": "run1=train.yaml", "COMMIT": "abc123"}) == [("run1", "train.yaml")]


@pytest.fixture
def run_dir(tmp_path):
    (tmp_path / "weights").mkdir()
    return tmp_path


def test_first_time_every_run_is_started_and_a_folder_in_the_way_is_left_for_training_to_refuse(run_dir):
    (run_dir / "weights" / "last.pt").touch()
    assert run.what_to_do(run_dir, carrying_on=False) == run.START


def test_carrying_on_skips_a_scored_run_resumes_one_with_weights_and_starts_the_rest(run_dir, tmp_path):
    assert run.what_to_do(tmp_path / "never_started", carrying_on=True) == run.START
    assert run.what_to_do(run_dir, carrying_on=True) == run.START, "cut before any weights were saved"
    (run_dir / "weights" / "last.pt").touch()
    assert run.what_to_do(run_dir, carrying_on=True) == run.CARRY_ON
    (run_dir / "scores.md").touch()
    assert run.what_to_do(run_dir, carrying_on=True) == run.SKIP


def test_progress_bar_redrawn_in_place_is_passed_on_as_its_last_state():
    assert run.last_state_of_each_line("epoch 1\n 10%\r 50%\r100%\n\nepoch 2\n") == ["epoch 1", "100%", "epoch 2"]


def test_names_the_scripts_rely_on_are_the_ones_training_writes():
    from detect import score, train
    assert run.WEIGHTS_TO_CARRY_ON_FROM == train.WEIGHTS_TO_CARRY_ON_FROM
    assert (run.CONFIGS.relative_to(run.REPO) / "train.yaml").as_posix() == train.DEFAULT_CONFIG.relative_to(REPO).as_posix()
    assert run.SCORES_OF_A_FINISHED_RUN == "scores.md" and "scores.md" in Path(score.__file__).read_text()


def test_colab_installs_the_training_tool_at_the_version_the_mac_has():
    import ultralytics
    assert session.version_in_the_lock((REPO / "uv.lock").read_text(), "ultralytics") == ultralytics.__version__
