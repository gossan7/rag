import ast

from typing import Protocol, cast

from ...enums import FileType
from ...aliases import FileData
from ...data_models import (
    MinimalSource,
    merge_sources,
    get_merged_source_size
)


class LocatedNode(Protocol):
    lineno: int
    col_offset: int
    end_lineno: int
    end_col_offset: int


def get_line_offsets(file_content: str):
    offsets = [-1, 0]
    for line in file_content.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def has_boundaries(node: ast.AST) -> bool:
    return (
        isinstance(getattr(node, "lineno", None), int)
        and isinstance(getattr(node, "col_offset", None), int)
        and isinstance(getattr(node, "end_lineno", None), int)
        and isinstance(getattr(node, "end_col_offset", None), int)
    )


def get_boundaries(node: ast.AST, offsets: list[int]):
    located_node = cast(LocatedNode, node)
    return (
        offsets[located_node.lineno] + located_node.col_offset,
        offsets[located_node.end_lineno] + located_node.end_col_offset
    )


def recursive_chunk(
    node: ast.AST,
    offsets: list[int],
    max_chunk_size: int,
    file_path: int,
    start: int,
    end: int
) -> list[MinimalSource]:
    print("a")
    if end - start <= max_chunk_size:
        return [MinimalSource(
            file_path=file_path,
            first_character_index=start,
            last_character_index=end
        )]

    print("b")

    chunks: list[MinimalSource] = []
    children: list[LocatedNode] = [
        child for child in ast.iter_child_nodes(node)
        if has_boundaries(child)
    ]
    children.sort(key=lambda x: get_boundaries(x, offsets))

    if not children:
        print("c")

    child_start = start
    for child in children:
        child_end = offsets[child.end_lineno] + child.end_col_offset
        for source in recursive_chunk(
            node=child,
            offsets=offsets,
            max_chunk_size=max_chunk_size,
            file_path=file_path,
            start=child_start,
            end=child_end
        ):
            if not chunks:
                chunks.append(source)
                continue

            prev_source = chunks[-1]
            if get_merged_source_size(prev_source, source) <= max_chunk_size:
                chunks[-1] = merge_sources(prev_source, source)

            else:
                chunks.append(source)

        child_start = child_end

    return chunks


def ast_chunk(
    file_data: FileData,
    max_chunk_size: int = 2000
) -> list[MinimalSource]:
    ext, file_path, file_content = file_data
    assert ext == FileType.PYTHON  # TEMP

    # for child in ast.iter_child_nodes(ast.parse(file_content)):
    #     print(child)
    #     print(has_boundaries(node=child))

    return recursive_chunk(
        node=ast.parse(file_content),
        offsets=get_line_offsets(file_content),
        max_chunk_size=max_chunk_size,
        file_path=file_path,
        start=0,
        end=len(file_content),
    )
