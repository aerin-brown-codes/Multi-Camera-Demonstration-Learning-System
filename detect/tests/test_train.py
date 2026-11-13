import pytest
import yaml

from detect import train


def test_run_folder_that_already_exists_is_never_trained_over(cfg, tmp_path):
    config_file = tmp_path / "train.yaml"
    config_file.write_text(yaml.safe_dump(cfg))
    run_dir = tmp_path / "runs" / "first"
    run_dir.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="already holds a run"):
        train.start(config_file, run_dir)
    assert list(run_dir.iterdir()) == []
