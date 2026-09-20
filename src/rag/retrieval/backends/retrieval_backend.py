from typing import ClassVar
from abc import ABC, abstractmethod
from pathlib import Path
import json

from ...config import Config, load_config, dump_config
from ...enums import FileType, Backend
from ...data_models import MinimalSource
from ...aliases import FileData
from ...errors import IndexingError


class RetrievalBackend(ABC):
    backend_type: ClassVar[Backend]

    def __init__(self, config: Config):
        self.curr_config = config

        self.path = Path(
            str(self.curr_config.paths.processed) + "/"
            + self.curr_config.retrieval.backends[self.backend_type].path
        )
        self.path.mkdir(parents=True, exist_ok=True)

        self.uncached_extensions: set[FileType] = set()
        self.curr_file_cache: dict[str, str] = dict()
        self.prev_file_cache: dict[str, str] = dict()
        self.computed_files: set[str] = set()

        if self._previous_run_exists():
            self._load_prev_config()
            self._load_index()
            self._load_prev_manifest()
        else:
            if not self._dir_is_empty():
                raise IndexingError(
                    f"Can't create an index at {self.path}"
                    ": directory not empty."
                )
            self.prev_config = None
            self._create_index()

    def is_cached(self, file_data: FileData, file_hash: str):
        ext, file_path, _ = file_data

        if ext in self.uncached_extensions:
            return False

        if file_path not in self.prev_file_cache:
            return False

        return self.prev_file_cache[file_path] == file_hash

    def cache_file(self, file_data: FileData, file_hash: str):
        _, file_path, _ = file_data
        self.computed_files.add(file_path)
        self.curr_file_cache[file_path] = file_hash

    def register_file(self, file_data: FileData) -> None:
        _, file_path, _ = file_data
        self.computed_files.add(file_path)

    def _dir_is_empty(self):
        return not any(self.path.iterdir())

    def _load_prev_config(self):
        self.prev_config = load_config(
            file_name="config.yaml", config_dir=self.path
        )

    def _save_curr_config(self):
        dump_config(
            self.curr_config,
            file_name="config.yaml",
            config_dir=self.path
        )

    def _load_prev_manifest(self) -> None:
        manifest_path = Path(str(self.path) + "/manifest.json")
        try:
            with open(manifest_path, encoding="utf-8") as f:
                self.prev_file_cache = json.load(f)
        except Exception as e:
            raise IndexingError(
                f"Could not open the manifest at {manifest_path}:\n"
                + f"{e}"
            )

    def _save_curr_manifest(self) -> None:
        manifest_path = Path(str(self.path) + "/manifest.json")
        try:
            with open(manifest_path, mode="w", encoding="utf-8") as f:
                json.dump(self.curr_file_cache, f)
        except Exception as e:
            raise IndexingError(
                f"Could not write the manifest at {manifest_path}:\n"
                + f"{e}"
            )

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
    def add_document(
        self,
        chunk: MinimalSource,
        file_data: FileData,
        file_hash: str
    ) -> None:
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
