from pydantic import ValidationError

from pathlib import Path
import json

from ..data_models import RagSearchResults
from ..errors import SearchError, format_validation_error


def load_results(path: Path) -> RagSearchResults:
    if not path.is_file():
        raise SearchError(
            f"Cannot load search results '{path}': file not found."
        )

    try:
        with open(path, mode="r", encoding="utf-8") as f:
            raw_search_results = json.load(f)
    except (OSError, UnicodeDecodeError) as e:
        raise SearchError(
            f"Cannot read search results '{path}': {e}"
        ) from e
    except json.JSONDecodeError as e:
        raise SearchError(
            f"Cannot load search results '{path}': not valid JSON ({e})."
        ) from e

    try:
        search_results = RagSearchResults.model_validate(
            raw_search_results
        )
    except ValidationError as e:
        raise SearchError(
            f"Invalid search results '{path}':\n"
            f"{format_validation_error(e)}"
        ) from e

    return search_results


def dump_results(
    search_results: RagSearchResults,
    save_path: Path
) -> None:
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, mode="w", encoding="utf-8") as f:
            f.write(search_results.model_dump_json(indent=2))
    except OSError as e:
        raise SearchError(
            f"Cannot write search results to '{save_path}': {e}"
        ) from e
