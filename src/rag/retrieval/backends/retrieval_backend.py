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

        self.path = (
            self.curr_config.paths.processed
            / self.curr_config.retrieval.backends[self.backend_type].path
        )

        self.path.mkdir(parents=True, exist_ok=True)

        self.curr_file_cache: dict[str, str] = dict()
        self.prev_file_cache: dict[str, str] = dict()
        self.curr_max_chunk_size = 0
        self.prev_max_chunk_size = 0
        self.prev_config: Config | None = None

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
            self._create_index()

        self.uncached_extensions: set[FileType] = set([
            ext for ext, cfg in self.curr_config.indexing.file_types.items()
            if not self.prev_config or cfg.chunking_method !=
            self.prev_config.indexing.file_types[ext].chunking_method
        ])

    def is_cached(self, file_data: FileData, file_hash: str):
        if self.prev_max_chunk_size != self.curr_max_chunk_size:
            return False

        ext, file_path, _ = file_data

        if ext in self.uncached_extensions:
            return False

        if file_path not in self.prev_file_cache:
            return False

        return self.prev_file_cache[file_path] == file_hash

    def register_file(self, file_data: FileData, file_hash: str):
        _, file_path, _ = file_data
        self.curr_file_cache[file_path] = file_hash

    def begin_index(self, max_chunk_size: int) -> None:
        self.curr_max_chunk_size = max_chunk_size
        self.curr_file_cache = {}

    def end_index(self) -> None:
        self.prev_max_chunk_size = self.curr_max_chunk_size
        self.prev_file_cache = self.curr_file_cache.copy()
        self.prev_config = self.curr_config.model_copy(deep=True)
        self.uncached_extensions.clear()

    def search(self, query: str, k: int) -> dict[MinimalSource, int]:
        search_results: dict[MinimalSource, int] = {}
        out = self.search_index(query, k)
        out.sort(key=lambda x: x[1], reverse=True)
        max_score = len(out)
        rank = 0
        last_score = float('inf')
        for source, score in out:
            if score != last_score:
                rank += 1
            search_results[source] = max_score - rank + 1
        return search_results

    def _dir_is_empty(self) -> bool:
        return not any(self.path.iterdir())

    def _load_prev_config(self) -> None:
        self.prev_config = load_config(
            file_name="config.yaml", config_dir=self.path
        )

    def _save_curr_config(self) -> None:
        dump_config(
            self.curr_config,
            file_name="config.yaml",
            config_dir=self.path
        )

    def _load_prev_manifest(self) -> None:
        manifest_path = Path(str(self.path) + "/manifest.json")
        try:
            with open(manifest_path, encoding="utf-8") as f:
                self.prev_max_chunk_size, self.prev_file_cache = json.load(f)
        except Exception as e:
            raise IndexingError(
                f"Could not open the manifest at {manifest_path}:\n"
                + f"{e}"
            )

    def _save_curr_manifest(self) -> None:
        manifest_path = Path(str(self.path) + "/manifest.json")
        manifest = [self.curr_max_chunk_size, self.curr_file_cache]
        try:
            tmp = manifest_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(manifest), encoding="utf-8")
            tmp.replace(manifest_path)
        except Exception as e:
            raise IndexingError(
                f"Could not write the manifest at {manifest_path}:\n"
                + f"{e}"
            )

    @abstractmethod
    def _previous_run_exists(self) -> bool:
        pass

    @abstractmethod
    def _create_index(self) -> None:
        pass

    @abstractmethod
    def _load_index(self) -> None:
        pass

    @abstractmethod
    def delete_indexed_file(self, file_path: str) -> None:
        pass

    @abstractmethod
    def add_document(
        self,
        chunk: MinimalSource,
        file_data: FileData,
    ) -> None:
        pass

    @abstractmethod
    def clean_index(self) -> None:
        pass

    @abstractmethod
    def save_index(self) -> None:
        pass

    @abstractmethod
    def search_index(
        self,
        query: str,
        k: int
    ) -> list[tuple[MinimalSource, int]]:
        pass
