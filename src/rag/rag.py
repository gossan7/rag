from typing import Type
from functools import cached_property

from .config import Config, load_config
from .enums import Backend
from .retrieval.backends import (
    RetrievalBackend,
    BM25Backend,
    MiniLMBackend,
    TFIDFBackend
)
from .logger import Logger
from .retrieval.index import index
from .retrieval.search import search


RETRIEVAL_BACKENDS: dict[Backend, Type[RetrievalBackend]] = {
    Backend.TFIDF: TFIDFBackend,
    Backend.BM25: BM25Backend,
    Backend.MINILM: MiniLMBackend
}


class RAG:
    def __init__(self, config_file: str):
        self.config: Config = load_config(config_file)
        self.logger = Logger()

    def index(self, max_chunk_size: int) -> None:
        index(self.config, max_chunk_size, self.backends, self.logger)

    def search(self, query: str, k: int = 5):
        search_results = search(query, k, self.backends)
        for source in search_results:
            print(source.file_path)
            print(source.first_character_index)
            print(source.last_character_index)
            print()

    def search_dataset(self):
        pass

    def answer(self):
        pass

    def asnwer_dataset(self):
        pass

    def evaluate(self):
        pass

    @cached_property
    def backends(self) -> list[RetrievalBackend]:
        return [
            RETRIEVAL_BACKENDS[bkd](self.config, self.logger)
            for bkd, bkd_config in self.config.retrieval.backends.items()
            if bkd_config.enabled is True
        ]
