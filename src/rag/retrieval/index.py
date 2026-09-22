from hashlib import sha256

from ..config import Config
from ..errors import IndexingError
from .backends.retrieval_backend import RetrievalBackend
from .chunk import chunk_file
from .load import file_generator


def index(
    config: Config,
    max_chunk_size: int,
    backends: list[RetrievalBackend]
) -> None:
    if max_chunk_size < 1 or max_chunk_size > 2000:
        raise IndexingError(f"max_chunk_size is invalide: {max_chunk_size}")

    for bkd in backends:
        bkd.register_chunk_size(max_chunk_size)

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
        _, file_path, file_content = file_data
        file_hash = sha256(file_content.encode("utf-8")).hexdigest()

        for bkd in backends:
            bkd.register_file(file_data, file_hash)

        dirty = [
            bkd for bkd in backends
            if not bkd.is_cached(file_data, file_hash)
        ]

        if not dirty:
            continue

        for bkd in dirty:
            bkd.delete_indexed_file(file_path)

        for chunk in chunk_file(config, file_data, max_chunk_size):
            for bkd in dirty:
                bkd.add_document(chunk, file_data)

    for backend in backends:
        backend.clean_index()
        backend.save_index()
