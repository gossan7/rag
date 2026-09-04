from collections.abc import Iterator
from pathlib import Path

from ..errors import IndexingError
from ..config import Config
from ..aliases import FileData


def read_file(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    except Exception as e:
        raise IndexingError(
            "An error has occured during file processing " +
            f"(reading)\n{e}"
        )


def iter_files(config: Config) -> Iterator[FileData]:
    root = config.paths.raw
    for ext, ext_config in config.indexing.file_types.items():
        if ext_config.enabled is False:
            continue

        str_ext = ext.value
        for path in root.rglob(f"*{str_ext}"):
            if not path.is_file() or path.stat().st_size == 0:
                continue
            if config.indexing.excludes & set(path.parts):
                continue
            yield (ext, str(path), read_file(path))
