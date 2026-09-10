from typing import ClassVar
from abc import ABC, abstractmethod
from hashlib import sha256, _Hash
from pathlib import Path

from ...config import Config, load_config
from ...enums import FileType, Backend
from ...data_models import MinimalSource
from ...aliases import FileData
from ...errors import IndexingError


class RetrievalBackend(ABC):
    backend_type: ClassVar[Backend]

    def __init__(self, config: Config):
        self.curr_config = config

        self.path = Path(
            str(self.curr_config.paths.processed)
            + self.curr_config.retrieval.backends[self.backend_type].path
        )
        self.path.mkdir(parents=True, exist_ok=True)

        self.uncached_extensions: set[FileType] = set()
        self.curr_file_cache: dict[str, _Hash] = dict()
        self.prev_file_cache: dict[str, _Hash] = dict()
        self.computed_files: set[str] = set()

        if self._previous_run_exists():
            self._load_prev_config()
            self._load_index()
        else:
            if not self._dir_is_empty():
                raise IndexingError(
                    f"Can't create an index at {self.path}"
                    ": directory not empty."
                )
            self.prev_config = None
            self._create_index()

        # if anything is already in path, check if it is a valid prewvious run

        # detect if any previous run was done, if true:
        # load previous config (if previous chunking is different for an extension we dont keep anything from that extension)
        # load index
        # get all hashs in a dict
        # as we index files, we check if the extension is in the ones we need to index what ever the previous run was. we check if its already in the hashed files, if the hash has changed we reindex it.
        # when indexing a file, we remove it from the dict

    def is_cached(self, file_data: FileData):
        ext, file_path, file_content = file_data

        if ext in self.uncached_extensions:
            return False

        if file_path not in self.prev_file_cache:
            return False

        return self.prev_file_cache[file_content] \
            == sha256(file_content.encode("utf-8"))

    def cache_file(self, file_data: FileData):
        _, file_path, file_content = file_data
        self.computed_files.add(file_path)
        self.curr_file_cache[file_path] = sha256(
            file_content.encode("utf-8")
        )

    def register_file(self, file_data: FileData) -> None:
        _, file_path, _ = file_data
        self.computed_files.add(file_path)

    def _dir_is_empty(self):
        try:
            _ = next(self.path.rglob(""))
        except StopIteration:
            return True
        return False

    def _load_prev_config(self):
        self.prev_config = load_config(
            file_name="config.yaml", config_dir=self.path
        )

    def _save_curr_config(self):
        pass

    @abstractmethod
    def _previous_run_exists(self) -> bool:
        pass

    @abstractmethod
    def _create_index(self):
        pass

    @abstractmethod
    def _load_index(self):
        pass

    @abstractmethod
    def add_document(self, chunk: MinimalSource, file_data: FileData):
        pass

    @abstractmethod
    def clean_index(self):
        pass

    @abstractmethod
    def save_index(self):
        pass

    @abstractmethod
    def search_index(self, query: str, k: int):
        pass
