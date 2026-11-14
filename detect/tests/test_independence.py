import ast
import subprocess
import sys
from pathlib import Path

import detect

SOURCE = Path(detect.__file__).parent
NOT_TO_BE_IMPORTED = {"sim", "mujoco"}


def test_no_file_of_detect_imports_sim_or_mujoco():
    for file in SOURCE.rglob("*.py"):
        for node in ast.walk(ast.parse(file.read_text())):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            for name in names:
                assert name.split(".")[0] not in NOT_TO_BE_IMPORTED, f"{file.name} imports {name}"


def test_importing_every_module_of_detect_loads_neither_sim_nor_mujoco():
    modules = [f"detect.{file.stem}" for file in SOURCE.glob("*.py") if file.stem != "__init__"]
    code = (f"import importlib, sys\nfor m in {modules}: importlib.import_module(m)\n"
            f"print(sorted({NOT_TO_BE_IMPORTED} & set(sys.modules)))")
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "[]"


def test_training_scoring_and_the_detector_load_neither_feed_nor_zmq_because_colab_has_neither():
    code = ("import sys, detect.train, detect.score, detect.detector\n"
            "print(sorted({'feed', 'zmq'} & set(sys.modules)))")
    loaded = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert loaded == "[]"
