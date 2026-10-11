from rich.markdown import Markdown

from typing import Type
from collections.abc import Generator
from functools import cached_property
from hashlib import sha256
from pathlib import Path
from contextlib import contextmanager, ExitStack
from time import perf_counter


from .config import Config, load_config
from .errors import (
    RagError, IndexingError, SearchError, AnswerError, ModelLoadError
)
from .data_models import (
    UnansweredQuestion,
    RagSearchResults,
    RagSearchResultsAndAnswer,
    SearchResults,
    Answer
)
from .enums import Backend
from .retrieval import count_files, file_generator
from .retrieval import (
    RetrievalBackend,
    TFIDFBackend,
    BM25Backend,
    MiniLMBackend
)
from .logger import Logger
from .retrieval import index
from .retrieval import search
from .retrieval import check
from .dataset import load_dataset, load_results, dump_results, dump_answers
from .evaluation import evaluate
from .generation import Answerer


RETRIEVAL_BACKENDS: dict[Backend, Type[RetrievalBackend]] = {
    Backend.TFIDF: TFIDFBackend,
    Backend.BM25: BM25Backend,
    Backend.MINILM: MiniLMBackend
}


class RAG:
    def __init__(self, config_file: str) -> None:
        self.config: Config = load_config(config_file)
        self.logger: Logger = Logger()
        self.answerer: Answerer = Answerer(self.config, self.logger)

    @cached_property
    def backends(self) -> list[RetrievalBackend]:
        return [
            RETRIEVAL_BACKENDS[bkd](self.config, self.logger)
            for bkd, bkd_config in self.config.retrieval.backends.items()
            if bkd_config.enabled is True
        ]

    def index(self, max_chunk_size: int) -> None:
        if max_chunk_size < 1 or max_chunk_size > 2000:
            raise IndexingError(
                f"Invalid max_chunk_size {max_chunk_size}: expected an "
                "integer between 1 and 2000."
            )

        raw = self.config.paths.raw
        if not raw.is_dir():
            raise IndexingError(
                f"Cannot index: corpus directory '{raw}' not found. "
                "Check 'paths.raw' in the configuration."
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
            enabled = ", ".join(sorted(ftype.value for ftype in file_types))
            raise IndexingError(
                f"Cannot index: no non-empty {enabled} file found in "
                f"corpus directory '{raw}'. Check 'paths.raw' and "
                "'indexing.file_types' in the configuration."
            )

        start = perf_counter()
        names = ", ".join(bkd.backend_type.value for bkd in self.backends)
        self.logger.info(
            f"Indexing corpus '{raw}': "
            f"{self.logger.count(n_files, 'file')}, backends: {names}"
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

        self.ensure_indexable()
        try:
            with self.index_session(max_chunk_size):
                for note in self.logger.merge_labels([
                    (bkd.backend_type.value, note)
                    for bkd in self.backends for note in bkd.index_notes
                ]):
                    self.logger.info(note)
                index(
                    self.config, max_chunk_size, self.backends, file_iterator
                )
        except RagError as e:
            raise IndexingError(f"{e}{self.aborted_index_note()}") from e
        except KeyboardInterrupt:
            self.logger.close_bars()
            note = self.aborted_index_note().strip()
            if note:
                self.logger.warning(note)
            raise

        for summary in self.logger.merge_labels([
            (
                bkd.backend_type.value,
                f"{self.logger.count(bkd.indexed_files, 'file')} indexed "
                f"({self.logger.count(bkd.indexed_chunks, 'chunk')}), "
                f"{n_files - bkd.indexed_files} unchanged, "
                f"{bkd.removed_files} removed"
            )
            for bkd in self.backends
        ]):
            self.logger.info(summary)
        self.logger.success(
            "Indexing completed in "
            f"{self.logger.duration(perf_counter() - start)}."
        )

    def search(self, query: str, k: int = 5) -> SearchResults:
        question = UnansweredQuestion(
            question_id=sha256(query.strip().encode("utf-8")).hexdigest(),
            question=query
        )

        with self.search_session():
            check(self.config, self.backends, self.logger)
            search_results = search(question, k, self.backends)

        n_results = len(search_results.retrieved_sources)
        for source in search_results.retrieved_sources:
            path = source.file_path
            start = source.first_character_index
            end = source.last_character_index
            self.logger.result(f"{path} [{start}:{end}]")

        if n_results:
            self.logger.success(
                "Search completed: "
                f"{self.logger.count(n_results, 'result')} found."
            )
        else:
            self.logger.warning("No result found for this query.")
        return search_results

    def search_dataset(
        self,
        dataset_path: str,
        k: int,
        save_directory: str
    ) -> None:
        input_path = Path(dataset_path)
        output_path = Path(save_directory) / input_path.name
        if output_path.resolve() == input_path.resolve():
            raise SearchError(
                f"Refusing to overwrite dataset '{input_path}' with its "
                f"own search results: save_directory '{save_directory}' is "
                "the directory of the dataset. Choose another directory."
            )
        dataset = load_dataset(input_path)

        search_results = RagSearchResults(
            search_results=[],
            k=k
        )

        questions = dataset.rag_questions

        with self.search_session():
            check(self.config, self.backends, self.logger)
            self.logger.info(
                f"Searching {self.logger.count(len(questions), 'question')} "
                f"from '{input_path}' (k={k})..."
            )
            for question in self.logger.loading_bar("Searching", questions):
                search_results.search_results.append(
                    search(question, k, self.backends)
                )

        dump_results(search_results, output_path)
        self.logger.success(f"Search results saved to '{output_path}'.")

    def answer(self, query: str, k: int = 5) -> Answer:
        question = UnansweredQuestion(
            question=query
        )

        with self.search_session():
            check(self.config, self.backends, self.logger)
            results = search(question, k, self.backends)

        answer = self.answerer.answer(results)
        self.logger.result(Markdown(answer.answer))
        if self.answerer.last_from_cache:
            self.logger.success("Answer loaded from the answer cache.")
        else:
            self.logger.success("Answer generated.")
        return answer

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str
    ) -> None:
        input_path = Path(student_search_results_path)
        output_path = Path(save_directory) / input_path.name
        if output_path.resolve() == input_path.resolve():
            raise AnswerError(
                f"Refusing to overwrite search results '{input_path}' with "
                f"their answers: save_directory '{save_directory}' is the "
                "directory of the search results. Choose another directory."
            )

        search_results = load_results(input_path)
        answers = RagSearchResultsAndAnswer(
            search_results=[],
            k=search_results.k
        )

        n_questions = len(search_results.search_results)
        failed = 0
        cached = 0
        self.logger.info(
            f"Answering {self.logger.count(n_questions, 'question')} "
            f"from '{input_path}'..."
        )

        for results in self.logger.loading_bar(
            "Answering", search_results.search_results
        ):
            try:
                answers.search_results.append(
                    self.answerer.answer(results)
                )
                cached += self.answerer.last_from_cache
            except ModelLoadError:
                raise
            except RagError as e:
                failed += 1
                self.logger.warning(
                    f"Question '{results.question_id}' could not be "
                    f"answered and gets a placeholder answer:\n{e}"
                )
                answers.search_results.append(
                    Answer(
                        question_id=results.question_id,
                        question=results.question,
                        retrieved_sources=results.retrieved_sources,
                        answer="An error has occurred while "
                        "processing this question."
                    )
                )

        dump_answers(answers, output_path)
        summary = (
            f"Answers saved to '{output_path}' "
            f"({n_questions - failed - cached} generated, "
            f"{cached} from cache, {failed} failed)."
        )
        if failed:
            self.logger.warning(summary)
        else:
            self.logger.success(summary)

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
        self.logger.success("Evaluation completed.")

    def ensure_indexable(self) -> None:
        problems: list[str] = []
        for backend in self.backends:
            try:
                backend.ensure_indexable()
            except IndexingError as e:
                problems.append(str(e))
        if problems:
            raise IndexingError("\n".join(problems))

    def aborted_index_note(self) -> str:
        incomplete = [
            f"- '{backend.index_path}'" for backend in self.backends
            if backend.marker.is_file()
        ]
        if not incomplete:
            return ""
        return (
            "\nIndexing was aborted. These indexes are left marked as "
            "incomplete and are never repaired or deleted automatically; "
            "delete them manually, then run the index command again:\n"
            + "\n".join(incomplete)
        )

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
