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
    truth_sources, result_sources, question_ids = load_and_validate(
        student_search_results_path, dataset_path
    )

    full_recall = {k: 0 for k in config.evaluation.recall}
    all_recall = sorted(config.evaluation.recall)

    for id in logger.loading_bar("Evaluating", question_ids):
        found = set()
        recall = 0
        n_sources = len(truth_sources[id])
        n_results = min(len(result_sources[id]), all_recall[-1])
        if n_sources == 0:
            raise EvaluationError(
                f"Dataset is invalid (missing sources for question_id '{id}')"
            )

        for i, r_source in enumerate(result_sources[id]):
            if i >= all_recall[-1]:
                continue

            for j, t_source in enumerate(truth_sources[id]):
                if j in found:
                    continue

                score = iou(t_source, r_source)
                if score >= config.evaluation.min_iou:
                    recall += 1
                    found.add(j)

            if i + 1 in all_recall:
                full_recall[i + 1] += recall / n_sources

        if n_results < all_recall[-1]:
            for j in range(i+1, all_recall[-1]):
                if j + 1 in all_recall:
                    full_recall[j + 1] += recall / n_sources

    n_questions = len(question_ids)
    for k, v in full_recall.items():
        full_recall[k] = (v / n_questions)

    print(f"Questions evaluated: {n_questions}")
    for k in all_recall:
        score = full_recall[k]
        percent = score * 100
        print(f"Recall@{k}: {score:.2f} ({(percent):.2f})")
