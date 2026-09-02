import ast

from typing import Protocol, cast

from ...aliases import FileData
from ...data_models import (
    MinimalSource,
    can_merge_sources,
    merge_sources,
    get_merged_source_size
)


class LocatedNode(Protocol):
    lineno: int
    col_offset: int
    end_lineno: int
    end_col_offset: int


def has_statement_body(node: ast.AST) -> bool:
    for _, value in ast.iter_fields(node):
        if isinstance(value, list) \
                and any(isinstance(item, ast.stmt) for item in value):
            return True

    return False


def is_structural(node: ast.AST) -> bool:
    return isinstance(node, ast.stmt) or has_statement_body(node)


def is_located(node: ast.AST) -> bool:
    return (
        isinstance(getattr(node, "lineno", None), int)
        and isinstance(getattr(node, "col_offset", None), int)
        and isinstance(getattr(node, "end_lineno", None), int)
        and isinstance(getattr(node, "end_col_offset", None), int)
    )


def get_boundaries(node: ast.AST, offsets: list[int]):
    located = cast(LocatedNode, node)

    return (
        offsets[located.lineno] + located.col_offset,
        offsets[located.end_lineno] + located.end_col_offset
    )


def recursive_ast_chunk(
    node: ast.AST,
    offsets: list[int],
    d_chunks: list[tuple[int, MinimalSource]],
    file_content: str,
    file_length: int,
    max_chunk_size: int,
    file_path: str,
    prev_start: int,
    prev_end: int,
    depth: int
) -> None:
    children: list[ast.AST] = [
        child for child in ast.iter_child_nodes(node)
        if is_located(child) and is_structural(child)
    ]
    children.sort(key=lambda child: get_boundaries(child, offsets))

    if not children:
        if isinstance(node, ast.Module):
            start = 0
            end = file_length
        else:
            located = cast(LocatedNode, node)
            start = offsets[located.lineno] + located.col_offset
            end = offsets[located.end_lineno] + located.end_col_offset \
                if located.end_lineno is not None \
                and located.end_col_offset is not None \
                else file_length

        for first_char_idx in range(start, end, max_chunk_size):
            last_char_idx = min(first_char_idx + max_chunk_size, end)
            print(first_char_idx, last_char_idx)
            d_chunks.append((depth, MinimalSource(
                file_path=file_path,
                first_character_index=first_char_idx,
                last_character_index=last_char_idx
            )))
        return

    prev = prev_start
    for child in children:
        start, end = get_boundaries(child, offsets)

        if start - prev > 0 and file_content[prev:start].strip():
            d_chunks.append((depth, MinimalSource(
                file_path=file_path,
                first_character_index=prev,
                last_character_index=start
            )))

        recursive_ast_chunk(
            node=child,
            offsets=offsets,
            d_chunks=d_chunks,
            file_content=file_content,
            file_length=file_length,
            max_chunk_size=max_chunk_size,
            file_path=file_path,
            prev_start=start,
            prev_end=end,
            depth=depth+1
        )

        prev = end

    if prev_end - end > 0 and file_content[end:prev_end].strip():
        d_chunks.append((depth, MinimalSource(
            file_path=file_path,
            first_character_index=end,
            last_character_index=prev_end
        )))


def line_offsets(content: str) -> list[int]:
    offsets = [-1, 0]
    for i, char in enumerate(content):
        if char == "\n":
            offsets.append(i + 1)
    return offsets


def ast_chunk(
    file_data: FileData,
    max_chunk_size: int = 2000
) -> list[MinimalSource]:
    d_chunks: list[tuple[int, MinimalSource]] = []
    _, file_path, file_content = file_data

    file_length = len(file_content)
    offsets = line_offsets(file_content)
    tree = ast.parse(file_content)

    recursive_ast_chunk(
        node=tree,
        offsets=offsets,
        d_chunks=d_chunks,
        file_content=file_content,
        file_length=file_length,
        max_chunk_size=max_chunk_size,
        file_path=file_path,
        prev_start=0,
        prev_end=file_length,
        depth=0
    )

    merged: list[tuple[int, MinimalSource]] = []
    for depth, source in d_chunks:
        if not merged:
            merged.append((depth, source))
            continue

        prev_depth, prev_source = merged[-1]
        if prev_depth < depth \
                and get_merged_source_size(prev_source, source) <= max_chunk_size:
            merged[-1] = (prev_depth, merge_sources(prev_source, source))

        else:
            merged.append((depth, source))

    return merged  # [source for _, source in d_chunks]


if __name__ == "__name__":
    with open("ast.py") as f:
        ast_chunk(
            (None, "foo.py", f.read()),
            2000
        )
