from typing import List, Dict, Any
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
import json
import ast

from .load_files import load_files
from .create_indexes import bm25_index, tfidf_index, chroma_index
from ..structures import MinimalSource


def line_offsets(content: str) -> list[int]:
    """offsets[k] = index du 1er caractère de la ligne k+1 (1-indexée)."""
    offsets = [0]
    for i, char in enumerate(content):
        if char == "\n":
            offsets.append(i + 1)
    return offsets


def window_chunk(
    files: Dict[str, str],
    max_chunk_size: int,
) -> List[MinimalSource]:
    chunks: List[MinimalSource] = []
    for path, content in files.items():
        file_length = len(content)
        for i in range(0, file_length, max_chunk_size):
            chunks.append(MinimalSource(
                file_path=path,
                first_character_index=i,
                last_character_index=min(
                    i + max_chunk_size,
                    file_length
                )
            ))
    return chunks


def create_ast_chunk(
        chunks: List[MinimalSource],
        node: ast.stmt,
        path: str,
        offsets: List[int],
        max_chunk_size: int,
        content: str
) -> None:
    start = offsets[node.lineno - 1]
    if node.end_lineno and node.end_lineno < len(offsets):
        end = offsets[node.end_lineno]
    else:
        end = len(content)

    pos = start
    while pos < end:
        current_end = min(
            pos + max_chunk_size,
            end,
        )

        chunks.append(
            MinimalSource(
                file_path=path,
                first_character_index=pos,
                last_character_index=current_end,
            )
        )

        pos = current_end


def ast_chunk(
    files: Dict[str, str],
    max_chunk_length: int,
) -> List[MinimalSource]:
    chunks: List[MinimalSource] = []

    for path, content in files.items():
        tree = ast.parse(content)
        offsets = line_offsets(content)

        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                class_start = offsets[node.lineno - 1]

                for item in node.body:
                    if not isinstance(
                        item,
                        (ast.FunctionDef, ast.AsyncFunctionDef),
                    ):
                        continue

                    if (
                        item.end_lineno is not None
                        and item.end_lineno < len(offsets)
                    ):
                        method_end = offsets[item.end_lineno]
                    else:
                        method_end = len(content)

                    chunks.append(
                        MinimalSource(
                            file_path=path,
                            first_character_index=class_start,
                            last_character_index=method_end,
                        )
                    )

            else:
                create_ast_chunk(
                    chunks,
                    node,
                    path,
                    offsets,
                    max_chunk_length,
                    content,
                )

    return chunks


def langchain_language_chunk(
    files: Dict[str, str],
    max_chunk_size: int,
    language: Language
) -> List[MinimalSource]:
    chunks: List[MinimalSource] = []
    chunk_overlap = min(100, max_chunk_size // 10)

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=language,
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )

    for path in files:
        documents = splitter.create_documents([files[path]])
        for doc in documents:
            start = doc.metadata["start_index"]
            end = start + len(doc.page_content)
            assert files[path][start:end] == doc.page_content
            chunks.append(MinimalSource(
                file_path=path,
                first_character_index=start,
                last_character_index=end
            ))

    return chunks


def langchain_python_chunk(
    files: Dict[str, str],
    max_chunk_size: int,
) -> List[MinimalSource]:
    return langchain_language_chunk(
        files=files,
        max_chunk_size=max_chunk_size,
        language=Language.PYTHON
    )


def langchain_markdown_chunk(
    files: Dict[str, str],
    max_chunk_size: int,
) -> List[MinimalSource]:
    return langchain_language_chunk(
        files=files,
        max_chunk_size=max_chunk_size,
        language=Language.MARKDOWN
    )


def langchain_text_chunk(
    files: Dict[str, str],
    max_chunk_size: int,
) -> List[MinimalSource]:
    chunks: List[MinimalSource] = []

    chunk_overlap = min(100, max_chunk_size // 10)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True
    )

    for path in files:
        documents = splitter.create_documents([files[path]])
        for doc in documents:
            start = doc.metadata["start_index"]
            end = start + len(doc.page_content)
            assert files[path][start:end] == doc.page_content
            chunks.append(MinimalSource(
                file_path=path,
                first_character_index=start,
                last_character_index=end
            ))

    return chunks


CHUNKING = {
    "window": window_chunk,
    "ast": ast_chunk,
    "langchain_python": langchain_python_chunk,
    "langchain_markdown": langchain_markdown_chunk,
    "langchain_text": langchain_text_chunk
}


INDEXING = [
    (bm25_index, "bm25"),
    (tfidf_index, "tf-idf"),
    (chroma_index, "chroma")
]


def create_chunks(config: Dict[str, Any], max_chunk_size: int) -> None:
    files = load_files(config)
    chunks: List[MinimalSource] = []
    for extension in files:
        curr_chunks = CHUNKING[config["chunking"][extension]["method"]](
            files[extension],
            max_chunk_size,
        )
        chunks.extend(curr_chunks)

        for func, name in INDEXING:
            func(
                chunks=curr_chunks,
                files=files[extension],
                indexing_config=config["indexing"][name],
            )

    json_file = config["chunking"]["save_path"]
    Path(json_file).parent.mkdir(parents=True, exist_ok=True)

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "file_path": chunk.file_path,
                    "first_character_index": chunk.first_character_index,
                    "last_character_index": chunk.last_character_index
                }
                for chunk in chunks
            ],
            f,
            indent=2
        )
