from collections.abc import Iterator
from pathlib import Path
import os

from ..errors import IndexingError
from ..enums import FileType
from ..aliases import FileData
from ..logger import Logger


def read_file(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    except Exception as e:
        raise IndexingError(
            "An error has occured during file processing " +
            f"(reading)\n{e}"
        )


def file_paths(
    root: Path,
    excludes: set[str],
) -> list[Path]:
    paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname not in excludes
        ]

        directory = Path(dirpath)

        for filename in filenames:
            paths.append(directory / filename)
    return paths


def file_generator(
    root: Path,
    file_types: set[FileType],
    excludes: set[str],
    logger: Logger
) -> Iterator[FileData]:
    paths = file_paths(root, excludes)
    str_file_types = set([ftype.value for ftype in file_types])
    for path in logger.loading_bar("Indexing", paths):
        assert isinstance(path, Path)
        if not path.is_file() or path.stat().st_size == 0:
            continue
        if path.suffix not in str_file_types:
            continue
        yield (FileType(path.suffix), str(path), read_file(path))
