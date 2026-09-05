from typing import TypeAlias, Callable
from collections.abc import Iterator

from ..config import Config
from ..aliases import FileData
from ..enums import ChunkingMethod
from ..data_models import MinimalSource
from .chunking import window_chunk, langchain_chunk, ast_chunk

Chunker: TypeAlias = Callable[[FileData, int], list[MinimalSource]]


DEFAULT_MIN_CHUNK_SIZE = 400

CHUNKING: dict[ChunkingMethod, Chunker] = {
    ChunkingMethod.WINDOW: window_chunk,
    ChunkingMethod.LANGCHAIN: langchain_chunk,
    ChunkingMethod.AST: ast_chunk
}


def chunk_file(
    config: Config, file_data: FileData, max_chunk_size: int
) -> list[MinimalSource]:
    ext, _, _ = file_data
    chunking_method = config.indexing.file_types[ext].chunking_method
    return CHUNKING[chunking_method](file_data, max_chunk_size)


def chunked_file_iterator(
    config: Config,
    max_chunk_size: int,
    file_iterator: Iterator[FileData]
) -> Iterator[tuple[list[MinimalSource], FileData]]:
    for file_data in file_iterator:
        for source in chunk_file(config, file_data, max_chunk_size):
