from typing import Type
from collections.abc import Generator
from functools import cached_property
from hashlib import sha256
from pathlib import Path
from contextlib import contextmanager, ExitStack

from .config import Config, load_config
from .errors import RagError, IndexingError
from .data_models import (
    UnansweredQuestion, RagSearchResults, RagSearchResultsAndAnswer
)
from .enums import Backend
from .retrieval.load import count_files, file_generator
from .retrieval.backends import (
    RetrievalBackend,
    TFIDFBackend,
    BM25Backend,
    MiniLMBackend
)
from .logger import Logger
from .retrieval.index import index
from .retrieval.search import search
from .retrieval.check import check
from .dataset import load_dataset, load_results, dump_results, dump_answers
from .evaluation import evaluate
from .generation import Answerer


RETRIEVAL_BACKENDS: dict[Backend, Type[RetrievalBackend]] = {
    Backend.TFIDF: TFIDFBackend,
    Backend.BM25: BM25Backend,
    Backend.MINILM: MiniLMBackend
}


class RAG:
    def __init__(self, config_file: str):
        self.config: Config = load_config(config_file)
        self.logger = Logger()
        self.answerer = Answerer(self.config, self.logger)

    @cached_property
    def backends(self) -> list[RetrievalBackend]:
        return [
            RETRIEVAL_BACKENDS[bkd](self.config, self.logger)
            for bkd, bkd_config in self.config.retrieval.backends.items()
            if bkd_config.enabled is True
        ]

    def index(self, max_chunk_size: int) -> None:
        if max_chunk_size < 1 or max_chunk_size > 2000:
            raise IndexingError(f"max_chunk_size is invalid: {max_chunk_size}")

        if not self.config.paths.raw.is_dir():
            raise IndexingError(
                f"Corpus directory not found: {self.config.paths.raw}"
            )

        file_types = set((
            ftype for ftype, ftype_config
            in self.config.indexing.file_types.items()
            if ftype_config.enabled is True
        ))

        n_files = count_files(
            root=self.config.paths.raw,
            file_types=file_types,
            excludes=self.config.indexing.excludes,
        )

        if n_files == 0:
            raise IndexingError(
                "Empty corpus."
            )

        file_iterator = self.logger.loading_bar(
            "Indexing",
            file_generator(
                root=self.config.paths.raw,
                file_types=file_types,
                excludes=self.config.indexing.excludes,
            ),
            n_files
        )

        with self.index_session(max_chunk_size):
            index(self.config, max_chunk_size, self.backends, file_iterator)

    def search(self, query: str, k: int = 5) -> None:
        question = UnansweredQuestion(
            question_id=sha256(query.strip().encode("utf-8")).hexdigest(),
            question=query
        )

        with self.search_session():
            check(self.config, self.backends, self.logger)
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
        dataset = load_dataset(input_path)

        search_results = RagSearchResults(
            search_results=[],
            k=k
        )

        questions = dataset.rag_questions

        with self.search_session():
            check(self.config, self.backends, self.logger)
            for question in self.logger.loading_bar("Search", questions):
                search_results.search_results.append(
                    search(question, k, self.backends)
                )

        dump_results(search_results, output_path)

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str
    ) -> None:
        evaluate(
            self.config,
            student_search_results_path,
            dataset_path,
            self.logger
        )

    def answer(self, query: str, k: int = 5) -> None:
        question = UnansweredQuestion(
            question=query
        )

        with self.search_session():
            check(self.config, self.backends, self.logger)
            results = search(question, k, self.backends)

        answer = self.answerer.answer(results)
        print(answer.answer)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str
    ) -> None:
        input_path = Path(student_search_results_path)
        output_path = Path(save_directory) / input_path.name

        search_results = load_results(input_path)
        answers = RagSearchResultsAndAnswer(
            search_results=[],
            k=search_results.k
        )

        for results in self.logger.loading_bar(
            "answering", search_results.search_results
        ):
            try:
                answers.search_results.append(
                    self.answerer.answer(results)
                )
            except RagError as e:
                self.logger.warning(
                    "An error has occured while processing dataset. "
                    f"question will be skipped:\n{e}"
                )

        dump_answers(answers, output_path)

    @contextmanager
    def index_session(
        self,
        max_chunk_size: int,
    ) -> Generator[None, None, None]:
        with ExitStack() as stack:
            for backend in self.backends:
                stack.enter_context(backend.index_session(max_chunk_size))
            yield

    @contextmanager
    def search_session(self) -> Generator[None, None, None]:
        with ExitStack() as stack:
            for backend in self.backends:
                stack.enter_context(backend.search_session())
            yield
