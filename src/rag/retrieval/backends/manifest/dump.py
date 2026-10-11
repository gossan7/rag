import json

from pathlib import Path

from ....errors import RagError
from .manifest import Manifest


def dump_manifest(
    manifest: Manifest,
    file_name: str,
    directory: Path
) -> None:
    """Write an index manifest to a JSON file.

    The content is first written to a sibling ``.tmp`` file which
    then replaces the destination. The directory is created when
    missing.

    Args:
        manifest: Manifest to serialize.
        file_name: Name of the destination file.
        directory: Directory receiving the file.

    Raises:
        RagError: If the directory or the file cannot be written.
    """
    raw_manifest = manifest.model_dump(mode="json")
    path = directory / Path(file_name)
    text_manifest = json.dumps(raw_manifest)

    try:
        directory.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text_manifest, encoding="utf-8")
        tmp.replace(path)
    except OSError as e:
        raise RagError(
            f"Cannot write index manifest '{path}': {e}"
        ) from e
