from ..data_models import (
    Source,
    UnansweredQuestion,
    SearchResults,
    merge_scored_sources
)
from .backends import RetrievalBackend


def search(
    question: UnansweredQuestion,
    k: int,
    backends: list[RetrievalBackend]
) -> SearchResults:
    """Retrieve the best sources of a question by weighted rank fusion.

    Each backend is asked for ``2 * k`` hits and returns rank-based
    scores scaled by its weight. The scores of a chunk returned by
    several backends are added, and the ``k`` chunks with the highest
    totals are kept in decreasing order. A blank question or a
    non-positive ``k`` gives an empty result without querying the
    backends.

    Args:
        question: Question to search.
        k: Maximum number of sources to return.
        backends: Backends with an open search session.

    Returns:
        The question with its retrieved sources, best first.

    Raises:
        SearchError: If a backend fails.
    """
    query = question.question.strip()
    if not query or k <= 0:
        return SearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=[]
        )

    merged: dict[Source, float] = {}
    for bkd in backends:
        merged = merge_scored_sources(
            merged, bkd.search(query, k*2)
        )
    out = list(merged.keys())
    out.sort(key=merged.__getitem__, reverse=True)
    out = out[:min(k, len(out))]

    search_results = SearchResults(
        question_id=question.question_id,
        question=question.question,
        retrieved_sources=out
    )

    return search_results
