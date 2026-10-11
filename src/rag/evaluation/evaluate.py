from ..config import Config
from ..logger import Logger
from ..errors import EvaluationError
from .load import load_and_validate
from .iou import iou


def evaluate(
    config: Config,
    student_search_results_path: str,
    dataset_path: str,
    logger: Logger
) -> None:
    """Compute and print recall@k of search results against a dataset.

    For each question, the retrieved sources are scanned in rank
    order: an expected source is found once a retrieved source of
    the same file reaches the configured minimum IoU with it, and
    each expected source is counted at most once. Recall@k is the
    share of expected sources found within the first k results,
    averaged over the questions. When a question has fewer results
    than an evaluated k, its last recall value is reused and that k
    is flagged as ``estimated`` (some questions) or ``lower bound``
    (every question).

    Args:
        config: Configuration providing the k values and the IoU.
        student_search_results_path: Path of the search results file.
        dataset_path: Path of the answered dataset.
        logger: Logger used for the progress bar and the report.

    Raises:
        EvaluationError: If the two files do not describe the same
            questions or if a question has no expected source.
        RagError: If one of the files cannot be loaded.
    """
    truth_sources, result_sources, question_ids = load_and_validate(
        student_search_results_path, dataset_path
    )

    full_recall: dict[int, float] = {k: 0.0 for k in config.evaluation.recall}
    all_recall: list[int] = sorted(config.evaluation.recall)
    correct: set[int] = set()
    incorrect: set[int] = set()

    for id in logger.loading_bar("Evaluating", question_ids):
        found = set()
        recall = 0
        n_sources = len(truth_sources[id])
        n_results = min(len(result_sources[id]), all_recall[-1])
        if n_sources == 0:
            raise EvaluationError(
                f"Cannot evaluate: question '{id}' has an empty 'sources' "
                f"list in dataset '{dataset_path}'."
            )

        for i, r_source in enumerate(result_sources[id]):
            if i >= all_recall[-1]:
                break

            for j, t_source in enumerate(truth_sources[id]):
                if j in found:
                    continue

                score = iou(t_source, r_source)
                if score >= config.evaluation.min_iou:
                    recall += 1
                    found.add(j)

            if i + 1 in all_recall:
                full_recall[i + 1] += recall / n_sources
                correct.add(i + 1)

        if n_results < all_recall[-1]:
            for i in range(n_results, all_recall[-1]):
                if i + 1 in all_recall:
                    full_recall[i + 1] += recall / n_sources
                    incorrect.add(i + 1)

    estimated = correct & incorrect
    correct = correct - estimated
    incorrect = incorrect - estimated

    n_questions = len(question_ids)
    for k, v in full_recall.items():
        full_recall[k] = (v / n_questions)

    if incorrect or estimated:
        logger.warning(
            "Some questions have fewer retrieved sources than the evaluated "
            "k: the affected values are flagged 'estimated' (some "
            "questions) or 'lower bound' (all questions)."
        )

    logger.result(f"Questions evaluated: {n_questions}")

    width = len(f"Recall@{all_recall[-1]}:")
    for k in all_recall:
        if k in incorrect:
            confidence = " (lower bound)"
        elif k in estimated:
            confidence = " (estimated)"
        else:
            confidence = ""

        score = full_recall[k]
        percent = score * 100
        label = f"Recall@{k}:".ljust(width)
        logger.result(f"{label} {score:.2f} ({percent:.2f}%){confidence}")
