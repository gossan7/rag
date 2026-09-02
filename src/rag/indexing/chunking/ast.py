import ast

from ...config import Config
from ...aliases import FileData
from ...data_models import MinimalSource


def get_boundaries(node: ast.stmt, offsets: list[int]):
    assert node.end_lineno is not None
    assert node.end_col_offset is not None

    return (
        offsets[node.lineno] + node.col_offset,
        offsets[node.end_lineno] + node.end_col_offset
    )


def recursive_ast_chunk(
    node: ast.stmt | ast.Module,
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
    children = [
        child for child in ast.iter_child_nodes(node)
        if isinstance(child, ast.stmt)
    ]
    children.sort(key=lambda x: get_boundaries(x, offsets))

    if not children:
        if isinstance(node, ast.Module):
            start = 0
            end = file_length
        else:
            start = offsets[node.lineno] + node.col_offset
            end = offsets[node.end_lineno] + node.end_col_offset \
                if node.end_lineno is not None \
                and node.end_col_offset is not None \
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
        assert child.end_lineno is not None
        assert child.end_col_offset is not None

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
    config: Config,
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
    for depth, source in reversed(enumerate(d_chunks)):
        if not merged:
            merged.append((depth, source))

    return d_chunks  # [source for _, source in d_chunks]
