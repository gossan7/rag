from typing import List, Dict, Any
from pathlib import Path


def find_files(config: Dict[str, Any]) -> Dict[str, List[str]]:
    files_path: Dict[str, List[str]] = {}
    extensions = config["input"]["extensions"]
    excludes = set(config["input"]["excludes"])
    root = Path(config["input"]["path"])
    for type in extensions:
        files_path[type] = []
        for path in root.rglob(f"*{type}"):
            if not path.is_file():
                continue
            if excludes & set(path.parts):
                continue
            files_path[type].append(str(path))
    return files_path


def load_files(config: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    files: Dict[str, Dict[str, str]] = {}
    files_path = find_files(config)

    for extension in files_path:
        files[extension] = {}
        for path in files_path[extension]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    files[extension][path] = f.read()

            except Exception as e:
                raise Exception(
                    "An error has occured during file processing " +
                    f"(reading)\n{e}"
                )

    return files
