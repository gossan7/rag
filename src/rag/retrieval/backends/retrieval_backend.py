from typing import ClassVar
from abc import ABC, abstractmethod
from pathlib import Path
import json

from ...config import Config, load_config, dump_config
from ...enums import Backend
from ...data_models import MinimalSource
from ...aliases import FileData
from ...errors import IndexingError
from ...logger import Logger
from .manifest import Manifest


MANIFEST_FILENAME = "manifest.json"
CONFIG_FILENAME = "config.yaml"


class RetrievalBackend(ABC):
    backend_type: ClassVar[Backend]
    batch_size: ClassVar[int] = 2048

    def __init__(self, config: Config, logger: Logger) -> None:
        self.config: Config = config
        self.logger: Logger = logger
        self.index_path: Path = (
            config.paths.processed
            / config.retrieval.backends[self.backend_type].path
        )

        self.previous_config: Config | None = None
        self.previous_manifest: Manifest | None = None

    @abstractmethod
    def _create_index(self) -> None:
        pass

    @abstractmethod
    def _load_index(self) -> None:
        pass

    def _load_previous_config(self) -> None:
        self.previous_config = load_config(
            file_name="config.yaml", config_dir=self.index_path
        )

    def _load_previous_manifest(self) -> None:
        manifest_path = self.index_path / Path(MANIFEST_FILENAME)
        try:
            with open(manifest_path, encoding="utf-8") as f:
                self.previous_manifest = Manifest.model_validate(json.load(f))
        except Exception as e:
            raise IndexingError(
                f"Could not open the manifest at {manifest_path}:\n{e}"
            )

    def _dump_config(self) -> None:
        dump_config(
            config=self.config,
            file_name=CONFIG_FILENAME,
            config_dir=self.index_path
        )

    def _dump_manifest(self) -> None:
        manifest_path = self.index_path / Path(MANIFEST_FILENAME)
        raw_manifest = self.manifest.model_dump(mode="json")
        try:
            tmp = manifest_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(raw_manifest), encoding="utf-8")
            tmp.replace(manifest_path)
        except Exception as e:
            raise IndexingError(
                f"Could not write the manifest at {manifest_path}:\n"
                + f"{e}"
            )

    def _file_in_manifest(self, file_path: str) -> bool:
        if self.previous_manifest is None:
            return False
        return file_path in self.previous_manifest.files

    def _file_matches_manifest(self, file_path: str, file_hash: str) -> bool:
        if self._file_in_manifest(file_path) is False:
            return False
        return file_hash == self.previous_manifest.files[file_path]

    def begin_index(self, max_chunk_size: int) -> None:
        self.index_path.mkdir(parents=True, exist_ok=True)
        if not (self.index_path / Path(MANIFEST_FILENAME)).is_file():
            if any(self.index_path.iterdir()):
                raise IndexingError(
                    "Index directory is invalid. "
                    f"Remove {str(self.index_path)} or change paths in config."
                )
            self._create_index()
            self.cached_ext = set()
        else:
            self._load_previous_config()
            self._load_previous_manifest()
            self._load_index()
            self.cached_ext = set([
                ext for ext, ext_cfg in self.config.indexing.file_types.items()
                if ext_cfg.chunking_method ==
                self.previous_config.indexing.file_types[ext].chunking_method
            ])

            if self.previous_manifest.max_chunk_size != max_chunk_size:
                self.previous_config = None
                self.previous_manifest = None
                self._delete_all_documents_from_index()
                self.cached_ext = set()

        self.manifest = Manifest(
            max_chunk_size=max_chunk_size,
            files={}
        )
        self.to_commit = 0

    def register_file(self, file_data: FileData, file_hash: str) -> None:
        _, file_path, _ = file_data
        self.manifest.files[file_path] = file_hash

    def need_file(self, file_data: FileData, file_hash: str) -> bool:
        if self.previous_manifest:
            if self.previous_manifest.max_chunk_size != \
                    self.manifest.max_chunk_size:
                return True

        ext, file_path, _ = file_data
        if ext not in self.cached_ext:
            return True
        if self._file_matches_manifest(file_path, file_hash):
            return False
        return True

    def prepare_for_file(self, file_data: FileData, file_hash: str) -> None:
        _, file_path, _ = file_data
        self._delete_file_from_index(file_path)

    def add_chunk_to_index(
        self,
        chunk: MinimalSource,
        chunk_content: str
    ) -> None:
        self._add_document_to_index(
            chunk=chunk,
            chunk_content=chunk_content
        )

        self.to_commit += 1
        if self.to_commit >= self.batch_size:
            self._commit_index()
            self.to_commit = 0

    @abstractmethod
    def _add_document_to_index(
        self,
        chunk: MinimalSource,
        chunk_content: str
    ) -> None:
        pass

    @abstractmethod
    def _delete_file_from_index(self, file_path: str) -> None:
        pass

    @abstractmethod
    def _delete_all_documents_from_index(self) -> None:
        pass

    @abstractmethod
    def _commit_index(self) -> None:
        pass

    def _clean_index(self) -> None:
        if self.previous_manifest is None:
            return

        remaining_files = self.previous_manifest.files.keys() \
            - self.manifest.files.keys()
        for file_path in remaining_files:
            self._delete_file_from_index(file_path)

    @abstractmethod
    def _save_index(self) -> None:
        pass

    def end_index(self) -> None:
        self._clean_index()
        self._save_index()
        self._dump_config()
        self._dump_manifest()
        self.previous_config = self.config
        self.previous_manifest = self.manifest

    def begin_search(self) -> None:
        raise NotImplementedError

    def search(self, query: str, k: int) -> dict[MinimalSource, float]:
        search_results: dict[MinimalSource, float] = {}
        out = self.get_index_hits(query, k)
        out.sort(key=lambda x: x[1], reverse=True)
        weight = self.config.retrieval.backends[self.backend_type].weight
        rank = 0
        last_score: float | None = None
        for source, score in out:
            if score != last_score:
                rank += 1
                last_score = score
            search_results[source] = float(((k - rank + 1) / k) * weight)
        return search_results

    @abstractmethod
    def get_index_hits(
        self,
        query: str,
        k: int
    ) -> list[tuple[MinimalSource, float]]:
        pass
