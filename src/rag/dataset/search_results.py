from pydantic import ValidationError

from pathlib import Path
import json

from ..data_models import RagSearchResults
from ..errors import SearchError


def load_results(path: Path) -> RagSearchResults:
    if not path.is_file():
        raise SearchError(f"Could not open {path}")

    try:
        with open(path, mode="r", encoding="utf-8") as f:
            raw_search_results = json.load(f)
    except Exception as e:
        raise SearchError(f"Could not open {path}:\n{e}")

    try:
        search_results = RagSearchResults.model_validate(
            raw_search_results
        )
    except ValidationError as e:
        raise SearchError(f"Invalid search results {path}:\n{e}")

    return search_results


def dump_results(
    search_results: RagSearchResults,
    save_path: Path
) -> None:
    try:
        with open(save_path, mode="w", encoding="utf-8") as f:
            f.write(search_results.model_dump_json(indent=2))
    except Exception as e:
        raise SearchError(f"Could not dump search results {save_path}:\n{e}")
