from typing import Type
from hashlib import sha256

from ..config import Config
from ..enums import Backend
from ..aliases import FileData
from ..data_models import MinimalSource
from .backends.retrieval_backend import RetrievalBackend
from .backends.bm25_backend import BM25Backend
from .chunk import chunk_file
from .load import file_generator


RETRIEVAL_BACKENDS: dict[Backend, Type[RetrievalBackend]] = {
    Backend.BM25: BM25Backend,
    Backend.TFIDF: "bar",
    Backend.MINILM: "baz"
}


def cache_file(
    file_data: FileData, file_hash: str, backends: list[RetrievalBackend]
) -> None:
    for backend in backends:
        backend.cache_file(file_data, file_hash)


def index_chunk(
    file_data: FileData,
    file_hash: str,
    backends: list[RetrievalBackend],
    chunk: MinimalSource,
):
    print("feur")
    for backend in backends:
        backend.add_document(chunk, file_data, file_hash)


def index(
    config: Config,
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
        _, _, file_content = file_data
        file_hash = sha256(file_content.encode("utf-8")).hexdigest()
        if all(backend.is_cached(file_data, file_hash)
               for backend in backends):
            cache_file(file_data, file_hash, backends)
            continue

        for chunk in chunk_file(config, file_data, max_chunk_size):
            index_chunk(file_data, file_hash, backends, chunk)

    for backend in backends:
        backend.clean_index()
        backend.save_index()
