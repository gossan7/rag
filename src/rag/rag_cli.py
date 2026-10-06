from functools import cached_property

from .rag import RAG


class RagCLI:
    def __init__(self) -> None:
        self._config_file: str = "default.yaml"

    @cached_property
    def rag(self) -> RAG:
        return RAG(self._config_file)

    def index(
        self,
        max_chunk_size: int = 2000,
        config_file: str = "default.yaml"
    ) -> None:
        self._config_file = config_file
        self.rag.index(max_chunk_size)

    def search(
        self,
        query: str,
        k: int = 5,
        config_file: str = "default.yaml"
    ) -> None:
        self._config_file = config_file
        self.rag.search(query, k)

    def search_dataset(
        self,
        dataset_path: str,
        k: int,
        save_directory: str,
        config_file: str = "default.yaml"
    ) -> None:
        self._config_file = config_file
        self.rag.search_dataset(dataset_path, k, save_directory)

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        config_file: str = "default.yaml"
    ) -> None:
        self._config_file = config_file
        self.rag.evaluate(student_search_results_path, dataset_path)