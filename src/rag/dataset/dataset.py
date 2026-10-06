from pydantic import ValidationError

from pathlib import Path
import json

from ..data_models import RagDataset
from ..errors import RagError


def load_dataset(path: Path) -> RagDataset:
    if not path.is_file():
        raise RagError(f"Could not open {path}")

    try:
        with open(path, mode="r", encoding="utf-8") as f:
            raw_dataset = json.load(f)
    except Exception as e:
        raise RagError(f"Could not open {path}:\n{e}")

    try:
        dataset = RagDataset.model_validate(raw_dataset)
    except ValidationError as e:
        raise RagError(f"Invalid dataset {path}:\n{e}")

    return dataset
