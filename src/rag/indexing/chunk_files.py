from typing import TypeAlias, Callable

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
