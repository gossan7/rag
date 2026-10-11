from functools import cached_property

from .rag import RAG
from .errors import RagError


def text_argument(name: str, value: object, allow_empty: bool = False) -> str:
    """Validate a command-line value expected to be text.

    Python Fire converts values that look like Python literals, so
    numbers are accepted and converted back to text, while booleans
    and containers are rejected.

    Args:
        name: Name of the option, used in error messages.
        value: Value received from Python Fire.
        allow_empty: Whether a blank value is accepted.

    Returns:
        The value as text.

    Raises:
        RagError: If the value is not text or a number, or if it is
            blank while ``allow_empty`` is false.
    """
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
    """Validate a command-line value expected to be an integer.

    Args:
        name: Name of the option, used in error messages.
        value: Value received from Python Fire.

    Returns:
        The value, unchanged.

    Raises:
        RagError: If the value is a boolean or not an integer.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise RagError(
            f"Invalid value for --{name}: {value!r}. Expected an integer."
        )
    return value


class RagCLI:
    """Command-line interface exposed through Python Fire.

    Each public method is a command. Commands validate their raw
    arguments, select the configuration file and delegate to the
    ``RAG`` service, which is only created once arguments are valid.
    """

    def __init__(self) -> None:
        """Select the default configuration file."""
        self._config_file: str = "default.yaml"

    @cached_property
    def _rag(self) -> RAG:
        """Return the RAG service, created on first access.

        The service is built from the configuration file selected by the
        running command.

        Raises:
            ConfigError: If the configuration is missing or invalid.
            AnswerError: If the answer cache cannot be opened.
        """
        return RAG(self._config_file)

    def index(
        self,
        max_chunk_size: int = 2000,
        config_file: str = "default.yaml"
    ) -> None:
        """Build or update the indexes of the configured corpus.

        Args:
            max_chunk_size: Maximum chunk size in characters (1-2000).
            config_file: Configuration file name in ``config``.

        Raises:
            RagError: If an argument is invalid or indexing fails.
        """
        max_chunk_size = integer_argument("max_chunk_size", max_chunk_size)
        self._config_file = text_argument("config_file", config_file)
        self._rag.index(max_chunk_size)

    def search(
        self,
        query: str,
        k: int = 5,
        config_file: str = "default.yaml"
    ) -> None:
        """Print the top-k sources of a single query.

        Args:
            query: Text of the query; a blank query yields no result.
            k: Maximum number of sources to return.
            config_file: Configuration file name in ``config``.

        Raises:
            RagError: If an argument is invalid or the search fails.
        """
        query = text_argument("query", query, allow_empty=True)
        k = integer_argument("k", k)
        self._config_file = text_argument("config_file", config_file)
        self._rag.search(query, k)

    def search_dataset(
        self,
        dataset_path: str,
        k: int,
        save_directory: str,
        config_file: str = "default.yaml"
    ) -> None:
        """Search a whole dataset and write a search results JSON file.

        Args:
            dataset_path: Path of the dataset JSON file.
            k: Maximum number of sources per question.
            save_directory: Directory receiving the results, saved under
                the file name of the dataset.
            config_file: Configuration file name in ``config``.

        Raises:
            RagError: If an argument is invalid or the search fails.
        """
        dataset_path = text_argument("dataset_path", dataset_path)
        k = integer_argument("k", k)
        save_directory = text_argument("save_directory", save_directory)
        self._config_file = text_argument("config_file", config_file)
        self._rag.search_dataset(dataset_path, k, save_directory)

    def answer(
        self,
        query: str,
        k: int = 5,
        config_file: str = "default.yaml"
    ) -> None:
        """Answer a single query from its retrieved sources.

        Args:
            query: Text of the query.
            k: Maximum number of sources given to the model.
            config_file: Configuration file name in ``config``.

        Raises:
            RagError: If an argument is invalid, the search fails or no
                answer can be generated.
        """
        query = text_argument("query", query, allow_empty=True)
        k = integer_argument("k", k)
        self._config_file = text_argument("config_file", config_file)
        self._rag.answer(query, k)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
        config_file: str = "default.yaml"
    ) -> None:
        """Answer a search results file and write an answers JSON file.

        Args:
            student_search_results_path: Path of the search results.
            save_directory: Directory receiving the answers, saved under
                the file name of the search results.
            config_file: Configuration file name in ``config``.

        Raises:
            RagError: If an argument is invalid, the input cannot be
                loaded, the model cannot be loaded or the output cannot
                be written.
        """
        student_search_results_path = text_argument(
            "student_search_results_path", student_search_results_path
        )
        save_directory = text_argument("save_directory", save_directory)
        self._config_file = text_argument("config_file", config_file)
        self._rag.answer_dataset(student_search_results_path, save_directory)

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        config_file: str = "default.yaml"
    ) -> None:
        """Report recall@k of search results against a dataset.

        Args:
            student_search_results_path: Path of the search results.
            dataset_path: Path of the ground-truth answered dataset.
            config_file: Configuration file name in ``config``.

        Raises:
            RagError: If an argument is invalid or the files cannot be
                loaded or compared.
        """
        student_search_results_path = text_argument(
            "student_search_results_path", student_search_results_path
        )
        dataset_path = text_argument("dataset_path", dataset_path)
        self._config_file = text_argument("config_file", config_file)
        self._rag.evaluate(student_search_results_path, dataset_path)
