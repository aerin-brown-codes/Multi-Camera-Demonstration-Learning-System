from pathlib import Path

import yaml

REPO = Path(__file__).parents[3]
CONFIGS = Path(__file__).parent / "configs"
SET_TRAINED_ON, SET_WATCHED_WHILE_TRAINING, SET_SCORED = "train", "check", "test"


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def from_repo(path_text: str) -> Path:
    return REPO / path_text
