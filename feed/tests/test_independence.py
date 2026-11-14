import ast
import subprocess
import sys
from pathlib import Path

import feed

SOURCE = Path(feed.__file__).parent
NOT_TO_BE_IMPORTED = {"sim", "mujoco", "detect", "torch", "ultralytics"}


def test_no_file_of_feed_imports_the_sim_the_detector_or_their_tools():
    for file in SOURCE.rglob("*.py"):
        for node in ast.walk(ast.parse(file.read_text())):
            names = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            for name in names:
                assert name.split(".")[0] not in NOT_TO_BE_IMPORTED, f"{file.name} imports {name}"


def test_importing_every_module_of_feed_loads_none_of_them():
    modules = [f"feed.{file.stem}" for file in SOURCE.glob("*.py") if file.stem != "__init__"]
    code = (f"import importlib, sys\nfor m in {modules}: importlib.import_module(m)\n"
            f"print(sorted({NOT_TO_BE_IMPORTED} & set(sys.modules)))")
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip() == "[]"
