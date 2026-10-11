from collections.abc import Iterator
from pathlib import Path
import os

from ..errors import IndexingError
from ..enums import FileType
from ..aliases import FileData


def read_file(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    except UnicodeDecodeError as e:
        raise IndexingError(
            f"Cannot read corpus file '{path}': it is not valid UTF-8 text "
            f"({e.reason} at byte {e.start}). Fix or remove the file, or add "
            "its directory to 'indexing.excludes'."
        ) from e
    except OSError as e:
        raise IndexingError(
            f"Cannot read corpus file '{path}': {e}"
        ) from e


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


def count_files(
    root: Path,
    file_types: set[FileType],
    excludes: set[str]
) -> int:
    c = 0
    paths = file_paths(root, excludes)
    str_file_types = set([ftype.value for ftype in file_types])
    for path in paths:
        assert isinstance(path, Path)
        if not path.is_file() or path.stat().st_size == 0:
            continue
        if path.suffix not in str_file_types:
            continue
        c += 1
    return c


def file_generator(
    root: Path,
    file_types: set[FileType],
    excludes: set[str]
) -> Iterator[FileData]:
    paths = file_paths(root, excludes)
    str_file_types = set([ftype.value for ftype in file_types])
    for path in paths:
        assert isinstance(path, Path)
        if not path.is_file() or path.stat().st_size == 0:
            continue
        if path.suffix not in str_file_types:
            continue
        yield (FileType(path.suffix), str(path), read_file(path))
