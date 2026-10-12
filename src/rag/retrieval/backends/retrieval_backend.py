from diskcache import Cache

from collections.abc import Callable, Generator
from functools import wraps
from contextlib import contextmanager
from typing import ClassVar, Concatenate, ParamSpec, TypeVar, TypeAlias
from abc import ABC, abstractmethod
from pathlib import Path
from hashlib import sha256
import json
import sqlite3

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
    """Build a decorator restricting a backend method to one mode.

    Args:
        mode: Session mode the backend must be in, or None when no
            session may be open.

    Returns:
        A decorator for ``RetrievalBackend`` methods.
    """
    def decorator(
        func: RetrievalBackendMethod[P, R]
    ) -> RetrievalBackendMethod[P, R]:
        """Wrap a backend method with the session mode check.

        Args:
            func: Method to protect.

        Returns:
            The wrapped method, keeping its signature.
        """
        @wraps(func)
        def wrapper(
            self: "RetrievalBackend",
            /,
            *args: P.args,
            **kwargs: P.kwargs,
        ) -> R:
            """Call the method when the backend is in the required mode.

            Raises:
                IndexingError: If an indexing session is required.
                SearchError: If a search session is required.
                RagError: If no session must be open, or if the required
                    mode is unknown.
            """
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
    """Persistent, incrementally updated retrieval index.

    This base class implements what every backend shares: the index
    directory layout (engine data, a copy of the configuration and a
    manifest of file hashes), indexing and search sessions, the
    detection of files to re-index, the interruption marker, and the
    per-backend search cache with rank-based scoring. Subclasses
    only implement the storage engine operations.

    An index directory holding the interruption marker, or missing
    its manifest, configuration or data, is never repaired or
    deleted: indexing and searching it are refused until it is
    removed manually.

    Attributes:
        backend_type: Backend implemented by the subclass.
        batch_size: Number of chunks added between two commits.
        storage_errors: Engine exceptions converted into application
            errors.
    """

    backend_type: ClassVar[Backend]
    batch_size: ClassVar[int] = 2048
    storage_errors: ClassVar[tuple[type[Exception], ...]] = (
        OSError, ValueError, RuntimeError, sqlite3.Error
    )

    def __init__(self, config: Config, logger: Logger) -> None:
        """Compute the index paths and reset the session state.

        Nothing is created on disk and no index is opened here.

        Args:
            config: Validated application configuration.
            logger: Logger used by the backend.
        """
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
        self.index_notes: list[str] = []
        self.indexed_files: int = 0
        self.indexed_chunks: int = 0
        self.removed_files: int = 0

    def _describe(self) -> str:
        """Return a short description of the index for messages."""
        return f"{self.backend_type.value} index at '{self.index_path}'"

    def _recovery_hint(self) -> str:
        """Return the manual recovery procedure of a broken index."""
        return (
            "The index is never repaired or deleted automatically: delete "
            f"'{self.index_path}' manually, then run the index command again."
        )

    def _index_problem(self) -> str | None:
        """Look for a reason why a non-empty index cannot be used.

        Returns:
            A description of the first problem found among a leftover
            interruption marker, a missing manifest, a missing stored
            configuration and a missing data directory, or None.
        """
        if self.marker.is_file():
            return (
                f"a previous indexing run did not complete "
                f"('{DIRTY_MARK_FILENAME}' marker found), so the index may "
                "be inconsistent"
            )
        if not (self.index_path / Path(MANIFEST_FILENAME)).is_file():
            return f"'{MANIFEST_FILENAME}' is missing"
        if not (self.index_path / Path(CONFIG_FILENAME)).is_file():
            return f"'{CONFIG_FILENAME}' is missing"
        if not self.data_path.is_dir():
            return f"the '{DATA_PATH}' directory is missing"
        return None

    def _write_error(self, error: Exception) -> IndexingError:
        """Build the error reported when writing to the index fails.

        Args:
            error: Exception raised by the storage engine.

        Returns:
            The indexing error to raise.
        """
        return IndexingError(
            f"Cannot write to the {self._describe()}: {error}"
        )

    def _index_is_empty(self) -> bool:
        """Tell whether the index directory is missing or empty."""
        if not self.index_path.exists():
            return True
        return not any(self.index_path.iterdir())

    def _load_metadatas(self) -> None:
        """Load the configuration and manifest stored with the index.

        Both are reset to None when the index is empty.

        Raises:
            RagError: If a stored file is missing or malformed.
        """
        if self._index_is_empty():
            self.prev_config = None
            self.prev_manifest = None
            return

        self.prev_config = load_config(CONFIG_FILENAME, self.index_path)
        self.prev_manifest = load_manifest(MANIFEST_FILENAME, self.index_path)

    def _dump_metadatas(self) -> None:
        """Store the current configuration and manifest with the index.

        Raises:
            RagError: If a file cannot be written.
        """
        dump_config(self.config, CONFIG_FILENAME, self.index_path)
        dump_manifest(self.manifest, MANIFEST_FILENAME, self.index_path)

    def _file_in_manifest(self, file_path: str) -> bool:
        """Tell whether a file belongs to the previously stored manifest.

        Args:
            file_path: Path of the corpus file.
        """
        if self.prev_manifest is None:
            return False
        return file_path in self.prev_manifest.files

    def _file_matches_manifest(self, file_path: str, file_hash: str) -> bool:
        """Tell whether a file is indexed with the given content hash.

        Args:
            file_path: Path of the corpus file.
            file_hash: SHA-256 hex digest of its current content.

        Returns:
            True when the previously stored manifest holds the file with
            the same hash.
        """
        if self.prev_manifest is None:
            return False
        if file_path not in self.prev_manifest.files:
            return False
        return file_hash == self.prev_manifest.files[file_path]

    @abstractmethod
    def _ensure_shared_resources(self) -> None:
        """Create the resources needed by both indexing and searching."""
        pass

    @abstractmethod
    def _create_index(self) -> None:
        """Create a new, empty engine index in the data directory."""
        pass

    @abstractmethod
    def _load_index(self) -> None:
        """Open the existing engine index of the data directory."""
        pass

    @abstractmethod
    def _open_writer(self) -> None:
        """Acquire what the engine needs to modify the index."""
        pass

    @abstractmethod
    def _close_writer(self) -> None:
        """Release the writing resources, whether or not they are open."""
        pass

    @abstractmethod
    def _open_reader(self) -> None:
        """Acquire what the engine needs to query the index."""
        pass

    @abstractmethod
    def _close_reader(self) -> None:
        """Release the reading resources, whether or not they are open."""
        pass

    @abstractmethod
    def _add_chunk_to_index(
        self,
        file_type: FileType,
        chunk: Source,
        chunk_content: str,
    ) -> None:
        """Add one chunk to the engine index.

        Args:
            file_type: Type of the file the chunk comes from.
            chunk: Location and content hash of the chunk.
            chunk_content: Text of the chunk.
        """
        pass

    @abstractmethod
    def _remove_file_from_index(self, file_path: str) -> None:
        """Remove every chunk of a file from the engine index.

        Args:
            file_path: Path of the corpus file.
        """
        pass

    @abstractmethod
    def _remove_file_type_from_index(self, file_type: FileType) -> None:
        """Remove every chunk of a file type from the engine index.

        Args:
            file_type: File type to remove.
        """
        pass

    @abstractmethod
    def _remove_all_documents_from_index(self) -> None:
        """Remove every chunk from the engine index."""
        pass

    @abstractmethod
    def _commit_index(self) -> None:
        """Persist the pending additions and removals."""
        pass

    @abstractmethod
    def get_index_hits(
        self,
        query: str,
        k: int
    ) -> list[tuple[Source, float]]:
        """Query the engine index, without cache or rank conversion.

        Args:
            query: Non-blank text of the query.
            k: Maximum number of hits.

        Returns:
            At most ``k`` pairs made of a source and its raw engine
            score, where a higher score means a better match. Raw scores
            are not comparable across backends.
        """
        pass

    def ensure_indexable(self) -> None:
        """Check that the index is absent, empty or complete.

        Raises:
            IndexingError: If the index exists and is incomplete, with
                the manual recovery procedure.
        """
        if self._index_is_empty():
            return
        problem = self._index_problem()
        if problem is not None:
            raise IndexingError(
                f"Cannot update the {self._describe()}: {problem}.\n"
                f"{self._recovery_hint()}"
            )

    @requires_mode(None)
    def _begin_index(self, max_chunk_size: int) -> None:
        """Open an indexing session.

        The stored configuration and manifest are loaded, the
        interruption marker is created, the engine index is created or
        opened for writing, and the documents made obsolete by a
        configuration change are removed. The marker stays in place
        until the session ends successfully.

        Args:
            max_chunk_size: Maximum chunk size in characters.

        Raises:
            IndexingError: If the existing index is incomplete, if its
                metadata is unreadable or if it cannot be opened for
                writing.
        """
        is_new = self._index_is_empty()
        self.ensure_indexable()

        try:
            self._load_metadatas()
        except RagError as e:
            raise IndexingError(
                f"Cannot update the {self._describe()}: its metadata is "
                f"unreadable.\n{e}\n{self._recovery_hint()}"
            ) from e
        self._ensure_shared_resources()

        self.index_notes = []
        self.indexed_files = 0
        self.indexed_chunks = 0
        self.removed_files = 0
        self.cached_ftypes: set[FileType] = set()
        self.to_commit: int = 0
        self.manifest: Manifest = Manifest(
            max_chunk_size=max_chunk_size,
            files={}
        )

        try:
            self.index_path.mkdir(parents=True, exist_ok=True)
            self.data_path.mkdir(parents=True, exist_ok=True)
            self.marker.touch()

            if is_new:
                self._create_index()
                self.index_notes.append("creating a new index")
            else:
                self._load_index()
            self._open_writer()

            if self.prev_config is not None and self.prev_manifest is not None:
                self._prepare_incremental_index(max_chunk_size)
        except self.storage_errors as e:
            raise IndexingError(
                f"Cannot open the {self._describe()} for writing: {e}"
            ) from e

        self.mode = BackendMode.INDEXING

    def _prepare_incremental_index(self, max_chunk_size: int) -> None:
        """Remove what a configuration change makes obsolete.

        A different maximum chunk size empties the whole index. A file
        type whose chunking method changed or which was disabled has its
        chunks removed. Only file types enabled before and now, with the
        same chunking method, remain eligible for file-level reuse.

        Args:
            max_chunk_size: Maximum chunk size of the current run.
        """
        assert self.prev_config is not None
        assert self.prev_manifest is not None
        prev_size = self.prev_manifest.max_chunk_size
        if prev_size != max_chunk_size:
            self._remove_all_documents_from_index()
            self.index_notes.append(
                f"max_chunk_size changed ({prev_size} -> {max_chunk_size}), "
                "rebuilding the whole index"
            )
            return

        for ftype, cfg in self.config.indexing.file_types.items():
            prev_cfg = self.prev_config.indexing.file_types[ftype]

            if prev_cfg.enabled and cfg.enabled:
                if prev_cfg.chunking_method == cfg.chunking_method:
                    self.cached_ftypes.add(ftype)

                else:
                    self._remove_file_type_from_index(ftype)
                    self.index_notes.append(
                        f"chunking method changed for {ftype.value} files "
                        f"({prev_cfg.chunking_method.value} -> "
                        f"{cfg.chunking_method.value}), re-indexing them"
                    )

            elif prev_cfg.enabled and not cfg.enabled:
                self._remove_file_type_from_index(ftype)
                self.index_notes.append(
                    f"{ftype.value} files are now disabled, removing them "
                    "from the index"
                )

    @requires_mode(BackendMode.INDEXING)
    def register_file(self, file_path: str, file_hash: str) -> None:
        """Record a corpus file in the manifest of the current run.

        Args:
            file_path: Path of the corpus file.
            file_hash: SHA-256 hex digest of its content.
        """
        self.manifest.files[file_path] = file_hash

    @requires_mode(BackendMode.INDEXING)
    def need_file(
        self,
        file_type: FileType,
        file_path: str,
        file_hash: str
    ) -> bool:
        """Tell whether a file must be chunked and indexed again.

        Args:
            file_type: Type of the file.
            file_path: Path of the corpus file.
            file_hash: SHA-256 hex digest of its current content.

        Returns:
            False only when the file type is eligible for reuse and the
            file is already indexed with the same content hash.
        """
        if file_type not in self.cached_ftypes:
            return True
        if not self._file_in_manifest(file_path):
            return True
        if not self._file_matches_manifest(file_path, file_hash):
            return True
        return False

    @requires_mode(BackendMode.INDEXING)
    def prepare_file(self, file_type: FileType, file_path: str) -> None:
        """Remove the previous chunks of a file about to be re-indexed.

        Nothing is removed for a file the index does not hold yet, or
        whose file type was already removed as a whole.

        Args:
            file_type: Type of the file.
            file_path: Path of the corpus file.

        Raises:
            IndexingError: If the engine cannot remove the chunks.
        """
        self.indexed_files += 1
        if file_type not in self.cached_ftypes:
            return
        if not self._file_in_manifest(file_path):
            return
        try:
            self._remove_file_from_index(file_path)
        except self.storage_errors as e:
            raise self._write_error(e) from e

    @requires_mode(BackendMode.INDEXING)
    def add_chunk(
        self,
        file_type: FileType,
        chunk: Source,
        chunk_content: str,
    ) -> None:
        """Add a chunk to the index, committing once per batch.

        Args:
            file_type: Type of the file the chunk comes from.
            chunk: Location and content hash of the chunk.
            chunk_content: Text of the chunk.

        Raises:
            IndexingError: If the engine cannot store the chunk.
        """
        try:
            self._add_chunk_to_index(
                file_type=file_type,
                chunk=chunk,
                chunk_content=chunk_content,
            )

            self.indexed_chunks += 1
            self.to_commit += 1
            if self.to_commit >= self.batch_size:
                self._commit_index()
                self.to_commit = 0
        except self.storage_errors as e:
            raise self._write_error(e) from e

    def _clean_index(self) -> None:
        """Remove the chunks of files that left the corpus.

        A file leaves the corpus when the previous manifest holds it and
        the current run did not register it.
        """
        if self.prev_manifest is None:
            return

        remaining_files = self.prev_manifest.files.keys() - \
            self.manifest.files.keys()
        self.removed_files = len(remaining_files)

        for file_path in remaining_files:
            ftype = FileType(Path(file_path).suffix)
            if ftype in self.cached_ftypes:
                self._remove_file_from_index(file_path)

    def _abort_index(self) -> None:
        """Close the writer of an aborted session, keeping the marker.

        Pending changes are not committed and the stored manifest and
        configuration are left untouched.
        """
        try:
            self._close_writer()
        finally:
            self.mode = None

    def _end_index(self) -> None:
        """Complete an indexing session.

        Removes the files that left the corpus, commits, closes the
        writer, stores the new configuration and manifest, and finally
        deletes the interruption marker.

        Raises:
            IndexingError: If the engine cannot write the index.
            RagError: If the configuration or manifest cannot be stored.
        """
        try:
            self._clean_index()
            self._commit_index()
            self._close_writer()
            self._dump_metadatas()
            self.marker.unlink(missing_ok=True)
        except self.storage_errors as e:
            raise self._write_error(e) from e
        finally:
            self.mode = None

    @requires_mode(None)
    def _begin_search(self) -> None:
        """Open a search session on the persisted index.

        Loads the stored configuration and manifest, opens the engine
        index and the search cache, and computes the fingerprint of the
        indexed snapshot (SHA-256 of the manifest and of the indexed
        file type settings) used in the search cache keys.

        Raises:
            SearchError: If there is no index, if it is incomplete, if
                its metadata is unreadable or if it cannot be opened.
        """
        if self._index_is_empty():
            raise SearchError(
                f"Cannot search the {self._describe()}: no index found. "
                "Run the index command first."
            )
        problem = self._index_problem()
        if problem is not None:
            raise SearchError(
                f"Cannot search the {self._describe()}: {problem}.\n"
                f"{self._recovery_hint()}"
            )

        try:
            self._load_metadatas()
        except RagError as e:
            raise SearchError(
                f"Cannot search the {self._describe()}: its metadata is "
                f"unreadable.\n{e}\n{self._recovery_hint()}"
            ) from e
        assert self.prev_config is not None
        assert self.prev_manifest is not None

        self._ensure_shared_resources()
        try:
            self._load_index()
            self._open_reader()

            self.search_cache = Cache(
                directory=self.index_path / Path("search_cache"),
                size_limit=MAX_CACHE_SIZE * 1024 * 1024,
                eviction_policy="least-recently-used",
            )
        except self.storage_errors as e:
            raise SearchError(
                f"Cannot open the {self._describe()}: {e}\n"
                f"If the index is corrupted: {self._recovery_hint()}"
            ) from e

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
        """Return the rank-based weighted scores of the best chunks.

        Raw engine hits are read from the persistent search cache when
        the same query and ``k`` were already run on the same indexed
        snapshot, and stored there otherwise. Hits are then ranked by
        decreasing raw score, equal raw scores sharing a rank, and each
        source receives ``(k - rank + 1) / k * weight`` where ``weight``
        is the configured weight of the backend.

        Args:
            query: Non-blank text of the query.
            k: Maximum number of hits requested from the engine.

        Returns:
            The fusion score of each retrieved source.

        Raises:
            SearchError: If the engine or the cache fails.
        """
        assert self.search_cache is not None
        try:
            value = self.search_cache.get((query, k, self.search_fingerprint))
            if value is None:
                out = self.get_index_hits(query, k)
                self.search_cache.set((query, k, self.search_fingerprint), out)

            else:
                out = [
                    (Source.model_validate(source), score)
                    for source, score in value
                ]
        except self.storage_errors as e:
            raise SearchError(
                f"Search failed in the {self._describe()}: {e}"
            ) from e

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
        """Close the search cache and the engine reader."""
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
        """Run an indexing session as a context manager.

        The session is completed when the block exits normally. On any
        exception, including a keyboard interruption, the session is
        aborted, the interruption marker is left in the index directory
        and the exception is re-raised.

        Args:
            max_chunk_size: Maximum chunk size in characters.

        Yields:
            The backend, ready to register files and receive chunks.
        """
        try:
            self._begin_index(max_chunk_size)
            yield self
            self._end_index()
        except BaseException:
            self._abort_index()
            raise

    @contextmanager
    def search_session(self) -> Generator["RetrievalBackend", None, None]:
        """Run a search session as a context manager.

        Yields:
            The backend, ready to be searched.
        """
        try:
            self._begin_search()
            yield self
        finally:
            self._end_search()
