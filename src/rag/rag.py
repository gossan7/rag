from typing import Type

from .config import Config
from .enums import Backend
from .retrieval.backends import RetrievalBackend, BM25Backend

from .retrieval.index import index
from .retrieval.search import search

RETRIEVAL_BACKENDS: dict[Backend, Type[RetrievalBackend]] = {
    Backend.BM25: BM25Backend,
    Backend.TFIDF: "bar",
    Backend.MINILM: "baz"
}


class RAG:
    def __init__(self, config: Config):
        self.config = config
        self._backends: None | list[RetrievalBackend] = None

    def index(self, max_chunk_size: int):
        index(self.config, max_chunk_size, self.backends)

    def search(self, query: str, k: int = 5):
        search_results = search(query, k, self.backends)
        for source, score in search_results.items():
            print(source.file_path)
            print(source.first_character_index)
            print(source.last_character_index)
            print(score)
            print()

    def search_dataset(self):
        pass

    def answer(self):
        pass

    def asnwer_dataset(self):
        pass

    def evaluate(self):
        pass

    @property
    def backends(self) -> list[RetrievalBackend]:
        if self._backends is not None:
            return self._backends

        self._backends = [
            RETRIEVAL_BACKENDS[bkd](self.config)
            for bkd, bkd_config in self.config.retrieval.backends.items()
            if bkd_config.enabled is True
        ]

        return self._backends
