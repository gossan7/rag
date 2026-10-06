from hashlib import sha256

from ..config import Config
from ..errors import IndexingError
from ..logger import Logger
from .backends.retrieval_backend import RetrievalBackend
from .chunk import chunk_file
from .load import file_generator, count_files


def index(
    config: Config,
    max_chunk_size: int,
    backends: list[RetrievalBackend],
    logger: Logger
) -> None:
    if max_chunk_size < 1 or max_chunk_size > 2000:
        raise IndexingError(f"max_chunk_size is invalid: {max_chunk_size}")
    if not config.paths.raw.is_dir():
        raise IndexingError(
            f"Corpus directory not found: {config.paths.raw}"
        )

    for bkd in backends:
        bkd.begin_index(max_chunk_size)

    file_types = set((
        ftype for ftype, ftype_config in config.indexing.file_types.items()
        if ftype_config.enabled is True
    ))

    n_files = count_files(
        root=config.paths.raw,
        file_types=file_types,
        excludes=config.indexing.excludes,
    )

    file_iterator = file_generator(
        root=config.paths.raw,
        file_types=file_types,
        excludes=config.indexing.excludes,
    )

    for file_data in logger.loading_bar("Indexing", file_iterator, n_files):
        _, _, file_content = file_data
        file_hash = sha256(file_content.encode("utf-8")).hexdigest()

        for bkd in backends:
            bkd.register_file(file_data, file_hash)

        dirty = [
            bkd for bkd in backends
            if bkd.need_file(file_data, file_hash)
        ]

        if not dirty:
            continue

        for bkd in dirty:
            bkd.prepare_for_file(file_data)

        for chunk in chunk_file(config, file_data, max_chunk_size):
            chunk_content = file_content[
                chunk.first_character_index:chunk.last_character_index
            ]

            for bkd in dirty:
                bkd.add_chunk_to_index(chunk, chunk_content)

    for backend in backends:
        backend.end_index()
