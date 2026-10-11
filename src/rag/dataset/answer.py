from pathlib import Path
from ..data_models import RagSearchResultsAndAnswer
from ..errors import AnswerError


def dump_answers(
    answers: RagSearchResultsAndAnswer,
    save_path: Path
) -> None:
    try:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, mode="w", encoding="utf-8") as f:
            f.write(answers.model_dump_json(indent=2))
    except OSError as e:
        raise AnswerError(
            f"Cannot write answers to '{save_path}': {e}"
        ) from e
