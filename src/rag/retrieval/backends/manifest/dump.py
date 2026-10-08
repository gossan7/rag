import json

from pathlib import Path

from ....errors import RagError
from .manifest import Manifest


def dump_manifest(
    manifest: Manifest,
    file_name: str,
    directory: Path
) -> None:
    raw_manifest = manifest.model_dump(mode="json")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / Path(file_name)
    text_manifest = json.dumps(raw_manifest)

    try:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(text_manifest, encoding="utf-8")
        tmp.replace(path)
    except Exception as e:
        raise RagError(
            f"Could not write the manifest at {path}:\n"
            f"{e}"
        )
