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
    except Exception as e:
        raise AnswerError(f"Could not dump answers at {save_path}:\n{e}")
