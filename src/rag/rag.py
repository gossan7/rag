from typing import Type
from functools import cached_property
from hashlib import sha256
from pathlib import Path

from .config import Config, load_config
from .data_models import UnansweredQuestion, StudentSearchResults
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
from .retrieval.dataset import load_dataset, dump_results


RETRIEVAL_BACKENDS: dict[Backend, Type[RetrievalBackend]] = {
    Backend.TFIDF: TFIDFBackend,
    Backend.BM25: BM25Backend,
    Backend.MINILM: MiniLMBackend
}


class RAG:
    def __init__(self, config_file: str):
        self.config: Config = load_config(config_file)
        self.logger = Logger()

    @cached_property
    def backends(self) -> list[RetrievalBackend]:
        return [
            RETRIEVAL_BACKENDS[bkd](self.config, self.logger)
            for bkd, bkd_config in self.config.retrieval.backends.items()
            if bkd_config.enabled is True
        ]

    def index(self, max_chunk_size: int) -> None:
        index(self.config, max_chunk_size, self.backends, self.logger)

    def search(self, query: str, k: int = 5) -> None:
        question = UnansweredQuestion(
            question_id=sha256(query.strip().encode("utf-8")).hexdigest(),
            question=query
        )

        search_results = search(question, k, self.backends)
        for source in search_results.retrieved_sources:
            path = source.file_path
            start = source.first_character_index
            end = source.last_character_index
            print(path, f"[{start}:{end}]")

    def search_dataset(
        self,
        dataset_path: str,
        k: int,
        save_directory: str
    ) -> None:
        input_path = Path(dataset_path)
        output_path = Path(save_directory) / input_path.name
        dataset = load_dataset(input_path, output_path)

        search_results = StudentSearchResults(
            search_results=[],
            k=k
        )

        for question in self.logger.loading_bar(
            "Search",
            dataset.rag_questions
        ):
            search_results.search_results.append(
                search(question, k, self.backends)
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        dump_results(search_results, output_path)

    def answer(self) -> None:
        pass

    def asnwer_dataset(self) -> None:
        pass

    def evaluate(self) -> None:
        pass
