from collections.abc import Iterator
from typing import Any, Type

from ..config import Config
from ..enums import Backend, ChunkingMethod
from ..aliases import FileData
from ..data_models import MinimalSource
from .backends.backend import RetrievalBackend
from .chunk import chunk_file
from .load import file_generator


RETRIEVAL_BACKENDS: dict[Backend, Type[RetrievalBackend]] = {
    Backend.BM25: "foo",
    Backend.TFIDF: "bar",
    Backend.MINILM: "baz"
}


def index_chunk(
    chunking_method: ChunkingMethod,
    file_data: FileData,
    backends: list[RetrievalBackend],
    max_chunk_size: int
):
    for backend in backends:
        pass


def index(
    config: Config,
    prev_config: Config,
    manifest: dict[str, Any],
    file_iterator: Iterator[FileData],
    max_chunk_size: int
) -> None:
    backends: list[RetrievalBackend] = [
        RETRIEVAL_BACKENDS[backend](config)
        for backend, config in config.retrieval.backends.items()
        if config.enabled is True
    ]

    file_iterator = file_generator(
        root=config.paths.raw,
        extensions=extensions,
        excludes=config.indexing.excludes
    )

    for file_data in file_iterator:
        if any((backend.is_cached(file_data) for backend in backends)):
            chunks = chunk_file(config, file_data, max_chunk_size)
            for chunk in chunks:
                index_chunk(file_data, backends)
