from collections.abc import Iterator
from typing import Any, Type

from ..config import Config
from ..enums import Backend, ChunkingMethod
from ..aliases import FileData
from ..data_models import MinimalSource
from .backends.retrieval_backend import RetrievalBackend
from .chunk import chunk_file
from .load import file_generator


RETRIEVAL_BACKENDS: dict[Backend, Type[RetrievalBackend]] = {
    Backend.BM25: "foo",
    Backend.TFIDF: "bar",
    Backend.MINILM: "baz"
}


def register_file(
    file_data: FileData, backends: list[RetrievalBackend]
) -> None:
    for backend in backends:
        backend.register_file(file_data)


def index_chunk(
    file_data: FileData,
    backends: list[RetrievalBackend],
    chunk: MinimalSource,
):
    for backend in backends:
        backend.add_document(chunk, file_data)


def index(
    config: Config,
    prev_config: Config,
    manifest: dict[str, Any],
    file_iterator: Iterator[FileData],
    max_chunk_size: int
) -> None:
    backends: list[RetrievalBackend] = [
        RETRIEVAL_BACKENDS[backend](config)
        for backend, backend_config in config.retrieval.backends.items()
        if backend_config.enabled is True
    ]

    file_types = set((
        ftype for ftype, ftype_config in config.indexing.file_types.items()
        if ftype_config.enabled is True
    ))

    file_iterator = file_generator(
        root=config.paths.raw,
        file_types=file_types,
        excludes=config.indexing.excludes
    )

    for file_data in file_iterator:
        if all(backend.is_cached(file_data) for backend in backends):
            register_file(file_data, backends)
            continue

        for chunk in chunk_file(config, file_data, max_chunk_size):
            index_chunk(file_data, backends, chunk)

    for backend in backends:
        backend.clean_index()
        backend.save_index()
