from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
import ast

from typing import TypeAlias, Callable

from ..config import Config
from ..aliases import FileData
from ..enums import FileType, ChunkingMethod
from ..data_models import MinimalSource, can_merge, merge_sources


Chunker: TypeAlias = Callable[[Config, FileData], list[MinimalSource]]

DEFAULT_MIN_CHUNK_SIZE = 400


def window_chunk(
    config: Config,
    file_data: FileData,
    max_chunk_size: int = 2000
) -> list[MinimalSource]:
    chunks: list[MinimalSource] = []
    _, file_path, file_content = file_data

    file_length = len(file_content)
    for start in range(0, file_length, max_chunk_size):
        end = min(start + max_chunk_size, file_length)
        chunks.append(MinimalSource(
            file_path=file_path,
            first_character_index=start,
            last_character_index=end
        ))
    return chunks


def line_offsets(content: str) -> list[int]:
    offsets = [0]
    for i, char in enumerate(content):
        if char == "\n":
            offsets.append(i + 1)
    return offsets


def recursive_ast_chunk(
    node: ast.AST,
    offsets: list[int],
    chunks: list[MinimalSource],
    file_length: int,
    max_chunk_size: int,
    min_chunk_size: int,
    file_path: str
) -> bool:
    for child in ast.iter_child_nodes(node):
        start = node.lineno
        end = node.end_lineno if node.end_lineno is not None else file_length
        node_size = offsets[end] - offsets[start]
        candidate = MinimalSource(
            file_path=file_path,
            first_character_index=start,
            last_character_index=end
        )

        # if node > max_size: recurse
        if node_size > max_chunk_size:
            pass
            # recursive_ast_chunk(  # TEMP: so i know if i didnt use a parameter
            #     node=child,
            #     offsets=offsets,
            #     chunks=chunks,
            #     file_length=file_length,
            #     max_chunk_size=max_chunk_size,
            #     min_chunk_size=min_chunk_size
            # )
        elif chunks and can_merge(chunks[-1], candidate):
            chunks[-1] = merge_sources(chunks[-1], candidate)

    return True


def ast_chunk(
    config: Config,
    file_data: FileData,
    max_chunk_size: int = 2000
) -> list[MinimalSource]:
    chunks: list[MinimalSource] = []
    ext, file_path, file_content = file_data

    min_chunk_size = min(DEFAULT_MIN_CHUNK_SIZE, max_chunk_size // 5)
    file_length = len(file_content)
    offsets = line_offsets(file_content)
    tree = ast.parse(file_content)

    return chunks


LANGUAGE: dict[FileType, Language | None] = {
    FileType.PYTHON: Language.PYTHON,
    FileType.MARKDOWN: Language.MARKDOWN,
    FileType.TEXT: None
}


def lc_chunk(
    config: Config,
    file_data: FileData,
    max_chunk_size: int = 2000
) -> list[MinimalSource]:
    chunks: list[MinimalSource] = []
    ext, file_path, file_content = file_data
    language = LANGUAGE[ext]
    kwargs = {
        "chunk_size": max_chunk_size,
        "chunk_overlap": min(100, max_chunk_size // 10),
        "add_start_index": True
    }

    if language is None:
        splitter = RecursiveCharacterTextSplitter(kwargs=kwargs)
    else:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=language,
            kwargs=kwargs
        )

    documents = splitter.create_documents([file_content])
    for doc in documents:
        start = doc.metadata["start_index"]
        end = start + len(doc.page_content)
        # TEMP
        assert file_content[start:end] == doc.page_content
        chunks.append(MinimalSource(
            file_path=file_path,
            first_character_index=start,
            last_character_index=end
        ))
    return chunks


CHUNKING: dict[ChunkingMethod, Chunker] = {
    ChunkingMethod.WINDOW: window_chunk,
    ChunkingMethod.AST: ast_chunk,
    ChunkingMethod.LANGCHAIN: lc_chunk
}


def chunk_file(config: Config, file_data: FileData) -> list[MinimalSource]:
    ext, _, _ = file_data
    chunking_method = config.indexing.file_types[ext].chunking_method
    return CHUNKING[chunking_method](config, file_data)
