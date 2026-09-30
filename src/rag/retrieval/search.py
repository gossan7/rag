from ..data_models import (
    MinimalSource,
    UnansweredQuestion,
    MinimalSearchResults,
    merge_scored_sources
)
from .backends import RetrievalBackend


def search(
    question: UnansweredQuestion,
    k: int,
    backends: list[RetrievalBackend]
) -> MinimalSearchResults:
    query = question.question.strip()
    if not query or k <= 0:
        return MinimalSearchResults(
            question_id=question.question_id,
            question=query,
            retrieved_sources=[]
        )

    merged: dict[MinimalSource, float] = {}
    for bkd in backends:
        merged = merge_scored_sources(
            merged, bkd.search(query, k*2)
        )
    out = list(merged.keys())
    out.sort(key=merged.__getitem__, reverse=True)
    out = out[:min(k, len(out))]

    search_results = MinimalSearchResults(
        question_id=question.question_id,
        question=query,
        retrieved_sources=out
    )

    return search_results
