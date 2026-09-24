from ..data_models import MinimalSource, merge_scored_sources
from .backends import RetrievalBackend


def search(
    query: str,
    k: int,
    backends: list[RetrievalBackend]
) -> list[MinimalSource]:
    query = query.strip()
    if not query or k <= 0:
        return []

    search_results: dict[MinimalSource, float] = {}
    for bkd in backends:
        search_results = merge_scored_sources(
            search_results, bkd.search(query, k*2)
        )
    out = list(search_results.keys())
    out.sort(key=search_results.__getitem__, reverse=True)
    return out[:min(k, len(out))]
