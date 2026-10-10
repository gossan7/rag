from diskcache import Cache

from collections.abc import Callable, Generator
from functools import wraps
from contextlib import contextmanager
from typing import ClassVar, Concatenate, ParamSpec, TypeVar, TypeAlias
from abc import ABC, abstractmethod
from pathlib import Path
from hashlib import sha256
import json

from ...config import Config, load_config, dump_config
from ...enums import Backend, FileType
from ...data_models import Source
from ...errors import RagError, IndexingError, SearchError
from ...logger import Logger
from .manifest import Manifest, load_manifest, dump_manifest
from .backend_mode import BackendMode

DATA_PATH = Path("data")
MANIFEST_FILENAME = "manifest.json"
CONFIG_FILENAME = "config.yaml"
DIRTY_MARK_FILENAME = ".indexing"
MAX_CACHE_SIZE = 512

P = ParamSpec("P")
R = TypeVar("R")

RetrievalBackendMethod: TypeAlias = \
    Callable[Concatenate["RetrievalBackend", P], R]


def requires_mode(
    mode: BackendMode | None,
) -> Callable[[RetrievalBackendMethod[P, R]], RetrievalBackendMethod[P, R]]:
    def decorator(
        func: RetrievalBackendMethod[P, R]
    ) -> RetrievalBackendMethod[P, R]:
        @wraps(func)
        def wrapper(
            self: "RetrievalBackend",
            /,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> R:
            if self.mode is mode:
                return func(self, *args, **kwargs)
            match mode:
                case BackendMode.INDEXING:
                    raise IndexingError(
                        "Operation requires an active indexing session."
                    )

                case BackendMode.SEARCHING:
                    raise SearchError(
                        "Operation requires an active search session."
                    )

                case None:
                    raise RagError(
                        "Operation requires no active session."
                    )

                case _:
                    raise RagError(f"Unknown backend mode: {mode}")

        return wrapper
    return decorator


class RetrievalBackend(ABC):
    backend_type: ClassVar[Backend]
    batch_size: ClassVar[int] = 2048

    def __init__(self, config: Config, logger: Logger) -> None:
        self.config: Config = config
        self.logger: Logger = logger
        self.index_path: Path = config.paths.processed / \
            config.retrieval.backends[self.backend_type].path
        self.data_path: Path = self.index_path / DATA_PATH

        self.marker: Path = self.index_path / Path(DIRTY_MARK_FILENAME)
        self.mode: BackendMode | None = None
        self.prev_config: Config | None = None
        self.prev_manifest: Manifest | None = None
        self.search_cache: Cache | None = None

    def _index_is_empty(self) -> bool:
        if not self.index_path.exists():
            return True
        return not any(self.index_path.iterdir())

    def _index_is_valid(self) -> bool:
        if self._index_is_empty():
            return False
        if not (self.index_path / Path(MANIFEST_FILENAME)).is_file():
            return False
        if not (self.index_path / Path(CONFIG_FILENAME)).is_file():
            return False
        if (self.index_path / Path(DIRTY_MARK_FILENAME)).is_file():
            return False
        if not self.data_path.is_dir():
            return False
        return True

    def _load_metadatas(self) -> None:
        if self._index_is_empty():
            self.prev_config = None
            self.prev_manifest = None
            return

        self.prev_config = load_config(CONFIG_FILENAME, self.index_path)
        self.prev_manifest = load_manifest(MANIFEST_FILENAME, self.index_path)

    def _dump_metadatas(self) -> None:
        dump_config(self.config, CONFIG_FILENAME, self.index_path)
        dump_manifest(self.manifest, MANIFEST_FILENAME, self.index_path)

    def _file_in_manifest(self, file_path: str) -> bool:
        if self.prev_manifest is None:
            return False
        return file_path in self.prev_manifest.files

    def _file_matches_manifest(self, file_path: str, file_hash: str) -> bool:
        if self.prev_manifest is None:
            return False
        if file_path not in self.prev_manifest.files:
            return False
        return file_hash == self.prev_manifest.files[file_path]

    @abstractmethod
    def _ensure_shared_resources(self) -> None:
        pass

    @abstractmethod
    def _create_index(self) -> None:
        pass

    @abstractmethod
    def _load_index(self) -> None:
        pass

    @abstractmethod
    def _open_writer(self) -> None:
        pass

    @abstractmethod
    def _close_writer(self) -> None:
        pass

    @abstractmethod
    def _open_reader(self) -> None:
        pass

    @abstractmethod
    def _close_reader(self) -> None:
        pass

    @abstractmethod
    def _add_chunk_to_index(
        self,
        file_type: FileType,
        chunk: Source,
        chunk_content: str,
    ) -> None:
        pass

    @abstractmethod
    def _remove_file_from_index(self, file_path: str) -> None:
        pass

    @abstractmethod
    def _remove_file_type_from_index(self, file_type: FileType) -> None:
        pass

    @abstractmethod
    def _remove_all_documents_from_index(self) -> None:
        pass

    @abstractmethod
    def _commit_index(self) -> None:
        pass

    @abstractmethod
    def get_index_hits(
        self,
        query: str,
        k: int
    ) -> list[tuple[Source, float]]:
        pass

    @requires_mode(None)
    def _begin_index(self, max_chunk_size: int) -> None:
        is_new = self._index_is_empty()
        if not is_new and not self._index_is_valid():
            raise IndexingError(
                "Invalid index."
            )

        self._load_metadatas()
        self._ensure_shared_resources()

        self.index_path.mkdir(parents=True, exist_ok=True)
        self.data_path.mkdir(parents=True, exist_ok=True)
        self.marker.touch()

        if is_new:
            self._create_index()
        else:
            self._load_index()
        self._open_writer()

        self.cached_ftypes = set()
        self.to_commit: int = 0
        self.manifest: Manifest = Manifest(
            max_chunk_size=max_chunk_size,
            files={}
        )

        if self.prev_config is not None and self.prev_manifest is not None:
            if self.prev_manifest.max_chunk_size == max_chunk_size:
                for ftype, cfg in self.config.indexing.file_types.items():
                    prev_cfg = self.prev_config.indexing.file_types[ftype]

                    if prev_cfg.enabled and cfg.enabled:
                        if prev_cfg.chunking_method == cfg.chunking_method:
                            self.cached_ftypes.add(ftype)

                        else:
                            self._remove_file_type_from_index(ftype)

                    elif prev_cfg.enabled and not cfg.enabled:
                        self._remove_file_type_from_index(ftype)

            else:
                self._remove_all_documents_from_index()

        self.mode = BackendMode.INDEXING

    @requires_mode(BackendMode.INDEXING)
    def register_file(self, file_path: str, file_hash: str) -> None:
        self.manifest.files[file_path] = file_hash

    @requires_mode(BackendMode.INDEXING)
    def need_file(
        self,
        file_type: FileType,
        file_path: str,
        file_hash: str
    ) -> bool:
        if file_type not in self.cached_ftypes:
            return True
        if not self._file_in_manifest(file_path):
            return True
        if not self._file_matches_manifest(file_path, file_hash):
            return True
        return False

    @requires_mode(BackendMode.INDEXING)
    def prepare_file(self, file_type: FileType, file_path: str) -> None:
        if file_type not in self.cached_ftypes:
            return
        if not self._file_in_manifest(file_path):
            return
        self._remove_file_from_index(file_path)

    @requires_mode(BackendMode.INDEXING)
    def add_chunk(
        self,
        file_type: FileType,
        chunk: Source,
        chunk_content: str,
    ) -> None:
        self._add_chunk_to_index(
            file_type=file_type,
            chunk=chunk,
            chunk_content=chunk_content,
        )

        self.to_commit += 1
        if self.to_commit >= self.batch_size:
            self._commit_index()
            self.to_commit = 0

    def _clean_index(self) -> None:
        if self.prev_manifest is None:
            return

        remaining_files = self.prev_manifest.files.keys() - \
            self.manifest.files.keys()

        for file_path in remaining_files:
            ftype = FileType(Path(file_path).suffix)
            if ftype in self.cached_ftypes:
                self._remove_file_from_index(file_path)

    def _abort_index(self) -> None:
        try:
            self._close_writer()
        finally:
            self.mode = None

    def _end_index(self) -> None:
        try:
            self._clean_index()
            self._commit_index()
            self._close_writer()
            self._dump_metadatas()
            self.marker.unlink(missing_ok=True)
        finally:
            self.mode = None

    @requires_mode(None)
    def _begin_search(self) -> None:
        if self._index_is_empty() or not self._index_is_valid():
            raise SearchError("Invalid index.")

        self._load_metadatas()
        if self.prev_config is None or self.prev_manifest is None:
            raise SearchError("Invalid index.")

        self._ensure_shared_resources()
        self._load_index()
        self._open_reader()

        self.search_cache = Cache(
            directory=self.index_path / Path("search_cache"),
            size_limit=MAX_CACHE_SIZE * 1024 * 1024,
            eviction_policy="least-recently-used",
        )

        raw_manifest = self.prev_manifest.model_dump(mode="json")
        raw_file_types = self.prev_config.indexing.model_dump(mode="json")[
            "file_types"
        ]

        canonical_manifest = json.dumps(
            raw_manifest,
            sort_keys=True,
            separators=(',', ':')
        )

        canonical_file_types = json.dumps(
            raw_file_types,
            sort_keys=True,
            separators=(',', ':')
        )

        canonical = canonical_manifest + canonical_file_types
        self.search_fingerprint = sha256(
            canonical.encode("utf-8")
        ).hexdigest()

        self.mode = BackendMode.SEARCHING

    @requires_mode(BackendMode.SEARCHING)
    def search(self, query: str, k: int) -> dict[Source, float]:
        assert self.search_cache is not None
        value = self.search_cache.get((query, k, self.search_fingerprint))
        if value is None:
            out = self.get_index_hits(query, k)
            self.search_cache.set((query, k, self.search_fingerprint), out)

        else:
            out = [
                (Source.model_validate(source), score)
                for source, score in value
            ]

        search_results: dict[Source, float] = {}
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

    def _end_search(self) -> None:
        try:
            if self.search_cache is not None:
                self.search_cache.close()
                self.search_cache = None
        finally:
            try:
                self._close_reader()
            finally:
                self.mode = None

    @contextmanager
    def index_session(
        self,
        max_chunk_size: int
    ) -> Generator["RetrievalBackend", None, None]:
        try:
            self._begin_index(max_chunk_size)
            yield self
            self._end_index()
        except BaseException:
            self._abort_index()
            raise

    @contextmanager
    def search_session(self) -> Generator["RetrievalBackend", None, None]:
        try:
            self._begin_search()
            yield self
        finally:
            self._end_search()
