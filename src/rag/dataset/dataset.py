from pydantic import ValidationError

from pathlib import Path
import json

from ..data_models import RagDataset
from ..errors import RagError, format_validation_error


def load_dataset(path: Path) -> RagDataset:
    if not path.is_file():
        raise RagError(
            f"Cannot load dataset '{path}': file not found."
        )

    try:
        with open(path, mode="r", encoding="utf-8") as f:
            raw_dataset = json.load(f)
    except (OSError, UnicodeDecodeError) as e:
        raise RagError(f"Cannot read dataset '{path}': {e}") from e
    except json.JSONDecodeError as e:
        raise RagError(
            f"Cannot load dataset '{path}': not valid JSON ({e})."
        ) from e

    try:
        dataset = RagDataset.model_validate(raw_dataset)
    except ValidationError as e:
        raise RagError(
            f"Invalid dataset '{path}':\n{format_validation_error(e)}"
        ) from e

    return dataset
