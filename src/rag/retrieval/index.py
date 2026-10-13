from hashlib import sha256

from collections.abc import Iterator
from pathlib import Path

from ..config import Config
from ..aliases import FileData
from ..data_models import Source
from .backends.retrieval_backend import RetrievalBackend
from .chunk import chunk_file


def index(
    config: Config,
    max_chunk_size: int,
    backends: list[RetrievalBackend],
    file_iterator: Iterator[FileData]
) -> None:
    """Feed the corpus files to the backends that need them.

    Every file is registered in the manifest of each backend. A file
    is chunked only when at least one backend needs it, and each
    chunk is given the SHA-256 of its UTF-8 encoded text before
    being added to those backends. The text handed to the backends
    is the chunk prefixed with the path of its file relative to the
    corpus directory, so that file names are searchable; the hash
    and the character range still describe the chunk alone. Must run
    inside an indexing session of every backend.

    Args:
        config: Configuration giving the chunking method per type.
        max_chunk_size: Maximum chunk size in characters.
        backends: Backends with an open indexing session.
        file_iterator: Type, path and content of each corpus file.

    Raises:
        IndexingError: If a backend cannot be written to.
    """
    for file_data in file_iterator:
        file_type, file_path, file_content = file_data
        file_hash = sha256(file_content.encode("utf-8")).hexdigest()

        for bkd in backends:
            bkd.register_file(file_path, file_hash)

        dirty = [
            bkd for bkd in backends
            if bkd.need_file(file_type, file_path, file_hash)
        ]

        if not dirty:
            continue

        for bkd in dirty:
            bkd.prepare_file(file_type, file_path)

        for chunk in chunk_file(config, file_data, max_chunk_size):
            file_name = str(Path(file_path).relative_to(config.paths.raw))
            chunk_content = file_content[
                chunk.first_character_index:chunk.last_character_index
            ]
            indexed_content = file_name + '\n' + chunk_content

            chunk_hash = sha256(chunk_content.encode("utf-8")).hexdigest()

            chunk_with_hash = Source(
                file_path=chunk.file_path,
                first_character_index=chunk.first_character_index,
                last_character_index=chunk.last_character_index,
                chunk_hash=chunk_hash
            )

            for bkd in dirty:
                bkd.add_chunk(file_type, chunk_with_hash, indexed_content)
