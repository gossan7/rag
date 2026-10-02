from pydantic import ValidationError

from pathlib import Path
import json

from ..data_models import RagDataset, StudentSearchResults
from ..errors import SearchError


def load_dataset(path: Path, save_path: Path) -> RagDataset:
    if not path.is_file():
        raise SearchError(f"Could not open {path}")

    try:
        with open(path, mode="r", encoding="utf-8") as f:
            raw_dataset = json.load(f)
    except Exception as e:
        raise SearchError(f"Could not open {path}:\n{e}")

    try:
        dataset = RagDataset.model_validate(raw_dataset)
    except ValidationError as e:
        raise SearchError(f"Invalid dataset {path}:\n{e}")

    return dataset


def dump_results(
    search_results: StudentSearchResults,
    save_path: Path
) -> None:
    try:
        with open(save_path, mode="w", encoding="utf-8") as f:
            f.write(search_results.model_dump_json(indent=2))
    except Exception as e:
        raise SearchError(f"Could not dump search results {save_path}:\n{e}")
