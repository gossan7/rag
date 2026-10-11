from pathlib import Path
from ..data_models import RagSearchResultsAndAnswer
from ..errors import AnswerError


def dump_answers(
    answers: RagSearchResultsAndAnswer,
    save_path: Path
) -> None:
    """Write the answers of a dataset to an indented JSON file.

    Missing parent directories are created.

    Args:
        answers: Answers to serialize.
        save_path: Destination file, overwritten when it exists.

    Raises:
        AnswerError: If the file cannot be written.
    """
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, mode="w", encoding="utf-8") as f:
            f.write(answers.model_dump_json(indent=2))
    except OSError as e:
        raise AnswerError(
            f"Cannot write answers to '{save_path}': {e}"
        ) from e
