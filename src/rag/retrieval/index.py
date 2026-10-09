from hashlib import sha256

from collections.abc import Iterator

from ..config import Config
from ..aliases import FileData
from .backends.retrieval_backend import RetrievalBackend
from .chunk import chunk_file


def index(
    config: Config,
    max_chunk_size: int,
    backends: list[RetrievalBackend],
    file_iterator: Iterator[FileData]
) -> None:
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
            chunk_content = file_content[
                chunk.first_character_index:chunk.last_character_index
            ]

            for bkd in dirty:
                bkd.add_chunk(file_type, chunk, chunk_content)
