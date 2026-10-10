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
