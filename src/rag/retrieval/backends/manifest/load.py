from pathlib import Path

import json

from ....errors import RagError
from .manifest import Manifest


def load_manifest(
    file_name: str,
    manifest_dir: Path
) -> Manifest:
    manifest_path = manifest_dir / Path(file_name)
    try:
        with open(manifest_path, encoding="utf-8", mode="r") as f:
            raw = json.load(f)
            manifest = Manifest.model_validate(raw)
    except Exception as e:
        raise RagError(
            f"Could not open the manifest at {manifest_path}:\n{e}"
        )
    return manifest
