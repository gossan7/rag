from functools import cached_property

from .rag import RAG
from .errors import RagError


def text_argument(name: str, value: object, allow_empty: bool = False) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise RagError(
            f"Invalid value for --{name}: {value!r}. Expected text; "
            "quote the value if it contains special characters."
        )
    text = str(value)
    if not allow_empty and not text.strip():
        raise RagError(f"Invalid value for --{name}: it must not be empty.")
    return text


def integer_argument(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RagError(
            f"Invalid value for --{name}: {value!r}. Expected an integer."
        )
    return value


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
        max_chunk_size = integer_argument("max_chunk_size", max_chunk_size)
        self._config_file = text_argument("config_file", config_file)
        self.rag.index(max_chunk_size)

    def search(
        self,
        query: str,
        k: int = 5,
        config_file: str = "default.yaml"
    ) -> None:
        query = text_argument("query", query, allow_empty=True)
        k = integer_argument("k", k)
        self._config_file = text_argument("config_file", config_file)
        self.rag.search(query, k)

    def search_dataset(
        self,
        dataset_path: str,
        k: int,
        save_directory: str,
        config_file: str = "default.yaml"
    ) -> None:
        dataset_path = text_argument("dataset_path", dataset_path)
        k = integer_argument("k", k)
        save_directory = text_argument("save_directory", save_directory)
        self._config_file = text_argument("config_file", config_file)
        self.rag.search_dataset(dataset_path, k, save_directory)

    def answer(
        self,
        query: str,
        k: int = 5,
        config_file: str = "default.yaml"
    ) -> None:
        query = text_argument("query", query, allow_empty=True)
        k = integer_argument("k", k)
        self._config_file = text_argument("config_file", config_file)
        self.rag.answer(query, k)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
        config_file: str = "default.yaml"
    ) -> None:
        student_search_results_path = text_argument(
            "student_search_results_path", student_search_results_path
        )
        save_directory = text_argument("save_directory", save_directory)
        self._config_file = text_argument("config_file", config_file)
        self.rag.answer_dataset(student_search_results_path, save_directory)

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        config_file: str = "default.yaml"
    ) -> None:
        student_search_results_path = text_argument(
            "student_search_results_path", student_search_results_path
        )
        dataset_path = text_argument("dataset_path", dataset_path)
        self._config_file = text_argument("config_file", config_file)
        self.rag.evaluate(student_search_results_path, dataset_path)
