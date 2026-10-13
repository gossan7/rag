from collections.abc import Iterator
from pathlib import Path
import os

from ..errors import IndexingError
from ..enums import FileType
from ..aliases import FileData
from ..logger import Logger


def read_file(path: Path) -> str:
    """Read a corpus file as UTF-8 text.

    Args:
        path: Path of the file.

    Returns:
        The decoded content, with universal newlines translation.

    Raises:
        IndexingError: If the file is not valid UTF-8 or cannot be
            read.
    """
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
    """List the files of a directory tree, skipping excluded folders.

    Args:
        root: Directory to walk.
        excludes: Directory names that are not entered, at any depth.

    Returns:
        The path of every file found, each prefixed by ``root`` as it
        was given.
    """
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
    """Count the non-empty corpus files of the given types.

    Args:
        root: Corpus directory.
        file_types: File types to keep.
        excludes: Directory names that are not entered.

    Returns:
        The number of candidate files. ``file_generator`` may yield
        fewer, since it skips the files it cannot read.
    """
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
    excludes: set[str],
    logger: Logger
) -> Iterator[FileData]:
    """Yield the non-empty corpus files of the given types.

    A file that cannot be read as UTF-8 text is skipped with a warning.

    Args:
        root: Corpus directory.
        file_types: File types to keep.
        excludes: Directory names that are not entered.
        logger: Logger receiving the warnings about skipped files.

    Yields:
        The type, path and decoded content of each file.
    """
    paths = file_paths(root, excludes)
    str_file_types = set([ftype.value for ftype in file_types])
    for path in paths:
        assert isinstance(path, Path)
        if not path.is_file() or path.stat().st_size == 0:
            continue
        if path.suffix not in str_file_types:
            continue
        try:
            content = read_file(path)
        except IndexingError as e:
            logger.warning(f"Skipping corpus file: {e}")
            continue
        yield (FileType(path.suffix), str(path), content)
