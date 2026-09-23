from ..data_models import MinimalSource, merge_scored_sources
from .backends import RetrievalBackend


def search(
    query: str,
    k: int,
    backends: list[RetrievalBackend]
) -> dict[MinimalSource, int]:
    search_results: dict[MinimalSource, int] = {}
    for bkd in backends:
        search_results = merge_scored_sources(
            search_results, bkd.search(query, k*2)
        )
    return search_results
