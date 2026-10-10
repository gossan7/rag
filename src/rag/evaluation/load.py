from pathlib import Path

from ..data_models import AnsweredQuestion, MinimalSource, Source
from ..dataset import load_results, load_dataset
from ..errors import EvaluationError


def load_and_validate(
    rag_search_results_path: str,
    dataset_path: str
) -> tuple[dict[str, list[MinimalSource]],
           dict[str, list[Source]], set[str]]:
    results = load_results(Path(rag_search_results_path))
    dataset = load_dataset(Path(dataset_path))

    if len(results.search_results) != len(dataset.rag_questions):
        raise EvaluationError("Search results don't match dataset.")

    truth_sources: dict[str, list[MinimalSource]] = {}
    for question in dataset.rag_questions:
        if not isinstance(question, AnsweredQuestion):
            raise EvaluationError(
                "Dataset should contains AnsweredQuestion only."
            )
        truth_sources[question.question_id] = question.sources

    result_sources: dict[str, list[Source]] = {}
    for result in results.search_results:
        result_sources[result.question_id] = result.retrieved_sources

    if truth_sources.keys() - result_sources.keys():
        raise EvaluationError("Search results don't match dataset.")
    if result_sources.keys() - truth_sources.keys():
        raise EvaluationError("Search results don't match dataset.")

    question_ids = set(truth_sources.keys())

    return (truth_sources, result_sources, question_ids)
