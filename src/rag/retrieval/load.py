from collections.abc import Iterator
from pathlib import Path

from ..errors import IndexingError
from ..enums import FileType
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


def file_generator(
    root: Path,
    file_types: set[FileType],
    excludes: set[str]
) -> Iterator[FileData]:
    for ftype in file_types:
        ext = ftype.value
        for path in root.rglob(f"*{ext}"):
            if not path.is_file() or path.stat().st_size == 0:
                continue
            if excludes & set(path.parts):
                continue
            yield (ftype, str(path), read_file(path))
