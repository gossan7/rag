from typing import List, Dict, Any
from pathlib import Path
import json
import ast

from .load_files import load_files
from .bm25 import bm25_index
from ..structures import MinimalSource


def window_chunk(
    files: Dict[str, str],
    max_chunk_length: int,
    retrieval: Dict[str, bool]
) -> List[MinimalSource]:
    chunks: List[MinimalSource] = []
    for path, content in files.items():
        file_length = len(content)
        for i in range(0, file_length, max_chunk_length):
            chunks.append(MinimalSource(
                file_path=path,
                first_character_index=i,
                last_character_index=min(
                    i + max_chunk_length,
                    file_length
                )
            ))
    return chunks


def line_offsets(content: str) -> list[int]:
    """offsets[k] = index du 1er caractère de la ligne k+1 (1-indexée)."""
    offsets = [0]
    for i, char in enumerate(content):
        if char == "\n":
            offsets.append(i + 1)
    return offsets


def create_ast_chunk(
        chunks: List[MinimalSource],
        node: ast.stmt,
        path: str,
        offsets: List[int],
        max_chunk_length: int,
        content: str
) -> None:
    start = offsets[node.lineno - 1]
    if node.end_lineno and node.end_lineno < len(offsets):
        end = offsets[node.end_lineno]
    else:
        end = len(content)

    pos = start
    while pos < end > max_chunk_length:
        current_end = min(pos + max_chunk_length, end)
        chunks.append(MinimalSource(
            file_path=path,
            first_character_index=pos,
            last_character_index=current_end
        ))
        pos = current_end


def ast_chunk(
    files: Dict[str, str],
    max_chunk_length: int,
    retrieval: Dict[str, bool]
) -> List[MinimalSource]:
    chunks: List[MinimalSource] = []
    for path, content in files.items():
        tree = ast.parse(content)
        offsets = line_offsets(content)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    create_ast_chunk(
                        chunks,
                        item,
                        path,
                        offsets,
                        max_chunk_length,
                        content
                    )
            else:
                create_ast_chunk(
                    chunks,
                    node,
                    path,
                    offsets,
                    max_chunk_length,
                    content
                )
    return chunks


CHUNKING = {
    "window": window_chunk,
    "ast": ast_chunk
}


def create_chunks(
    config: Dict[str, Any],
    max_chunk_length: int
) -> Dict[str, List[MinimalSource]]:

    files = load_files(config, max_chunk_length)
    chunks: Dict[str, List[MinimalSource]] = {}
    for extension in files:
        chunks[extension] = CHUNKING[config["chunking"][extension]["method"]](
            files[extension],
            max_chunk_length,
            config["retrieval"]
        )
        bm25_index(
            chunks[extension],
            files[extension],
            config["retrieval"]["bm25"],
            extension
        )

        json_file = config["chunking"][extension]["save_path"] + \
            extension[1:] + ".json"

        Path(json_file).parent.mkdir(parents=True, exist_ok=True)

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "file_path": chunk.file_path,
                        "first_character_index": chunk.first_character_index,
                        "last_character_index": chunk.last_character_index
                    }
                    for chunk in chunks[extension]
                ],
                f,
                indent=2
            )

    return chunks
