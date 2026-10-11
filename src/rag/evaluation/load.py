from pathlib import Path

from ..data_models import AnsweredQuestion, MinimalSource, Source
from ..dataset import load_results, load_dataset
from ..errors import EvaluationError


def load_and_validate(
    rag_search_results_path: str,
    dataset_path: str
) -> tuple[dict[str, list[MinimalSource]],
           dict[str, list[Source]], set[str]]:
    """Load search results and their ground truth for evaluation.

    Args:
        rag_search_results_path: Path of the search results file.
        dataset_path: Path of the answered dataset.

    Returns:
        A tuple made of the expected sources by question identifier,
        the retrieved sources by question identifier, and the set of
        question identifiers.

    Raises:
        EvaluationError: If either file holds no question, if the
            numbers of questions differ, if a dataset question has no
            expected sources, or if the question identifiers of the
            two files do not match.
        RagError: If one of the files cannot be loaded.
    """
    results = load_results(Path(rag_search_results_path))
    dataset = load_dataset(Path(dataset_path))

    if len(dataset.rag_questions) == 0:
        raise EvaluationError(
            f"Cannot evaluate: dataset '{dataset_path}' contains no question."
        )

    if len(results.search_results) == 0:
        raise EvaluationError(
            f"Cannot evaluate: search results '{rag_search_results_path}' "
            "contain no question."
        )

    mismatch = (
        f"Cannot evaluate: search results '{rag_search_results_path}' do "
        f"not match dataset '{dataset_path}'"
    )
    hint = "Make sure the search results were produced from this dataset."

    if len(results.search_results) != len(dataset.rag_questions):
        raise EvaluationError(
            f"{mismatch}: {len(results.search_results)} search results for "
            f"{len(dataset.rag_questions)} dataset questions. {hint}"
        )

    truth_sources: dict[str, list[MinimalSource]] = {}
    for question in dataset.rag_questions:
        if not isinstance(question, AnsweredQuestion):
            raise EvaluationError(
                f"Cannot evaluate: question '{question.question_id}' in "
                f"dataset '{dataset_path}' has no expected 'sources' and "
                "'answer'. Evaluation requires an answered dataset."
            )
        truth_sources[question.question_id] = question.sources

    result_sources: dict[str, list[Source]] = {}
    for result in results.search_results:
        result_sources[result.question_id] = result.retrieved_sources

    unsearched = sorted(truth_sources.keys() - result_sources.keys())
    if unsearched:
        raise EvaluationError(
            f"{mismatch}: {len(unsearched)} dataset question(s) have no "
            f"search result (e.g. question_id '{unsearched[0]}'). {hint}"
        )
    unknown = sorted(result_sources.keys() - truth_sources.keys())
    if unknown:
        raise EvaluationError(
            f"{mismatch}: {len(unknown)} search result(s) refer to "
            f"questions missing from the dataset (e.g. question_id "
            f"'{unknown[0]}'). {hint}"
        )

    question_ids = set(truth_sources.keys())

    return (truth_sources, result_sources, question_ids)
