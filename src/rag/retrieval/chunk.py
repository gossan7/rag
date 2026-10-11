from typing import TypeAlias, Callable

from ..config import Config
from ..aliases import FileData
from ..enums import ChunkingMethod
from ..data_models import MinimalSource
from .chunking import window_chunk, langchain_chunk, ast_chunk

Chunker: TypeAlias = Callable[[FileData, int], list[MinimalSource]]


CHUNKING: dict[ChunkingMethod, Chunker] = {
    ChunkingMethod.WINDOW: window_chunk,
    ChunkingMethod.LANGCHAIN: langchain_chunk,
    ChunkingMethod.AST: ast_chunk
}


def chunk_file(
    config: Config, file_data: FileData, max_chunk_size: int
) -> list[MinimalSource]:
    """Split a file with the chunking method of its file type.

    Args:
        config: Configuration mapping file types to chunking methods.
        file_data: Type, path and decoded content of the file.
        max_chunk_size: Maximum chunk size in characters.

    Returns:
        The chunk locations, as character ranges of the file content.
    """
    ext, _, _ = file_data
    chunking_method = config.indexing.file_types[ext].chunking_method
    return CHUNKING[chunking_method](file_data, max_chunk_size)
