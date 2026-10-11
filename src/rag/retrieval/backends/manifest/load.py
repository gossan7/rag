from pydantic import ValidationError

from pathlib import Path
import json

from ....errors import RagError, format_validation_error
from .manifest import Manifest


def load_manifest(
    file_name: str,
    manifest_dir: Path
) -> Manifest:
    manifest_path = manifest_dir / Path(file_name)
    try:
        with open(manifest_path, encoding="utf-8", mode="r") as f:
            raw = json.load(f)
    except FileNotFoundError as e:
        raise RagError(
            f"Index manifest '{manifest_path}' not found."
        ) from e
    except (OSError, UnicodeDecodeError) as e:
        raise RagError(
            f"Cannot read index manifest '{manifest_path}': {e}"
        ) from e
    except json.JSONDecodeError as e:
        raise RagError(
            f"Index manifest '{manifest_path}' is not valid JSON: {e}"
        ) from e

    try:
        manifest = Manifest.model_validate(raw)
    except ValidationError as e:
        raise RagError(
            f"Index manifest '{manifest_path}' is malformed:\n"
            f"{format_validation_error(e)}"
        ) from e
    return manifest
