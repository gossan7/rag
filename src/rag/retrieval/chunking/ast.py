import ast

from typing import Protocol, cast
from bisect import bisect_right


from ...aliases import FileData
from ...data_models import (
    MinimalSource,
    merge_sources,
    get_merged_source_size
)
from .langchain import langchain_chunk


class LocatedNode(Protocol):
    """Structural type of AST nodes carrying source positions.

    Line numbers are 1-based and column offsets are UTF-8 byte
    offsets within the line, as reported by the ``ast`` module.
    """

    lineno: int
    col_offset: int
    end_lineno: int
    end_col_offset: int


def get_line_offsets(file_content: str) -> list[int]:
    """Map 1-based line numbers to character offsets in a file.

    Args:
        file_content: Decoded content of the file.

    Returns:
        A list whose item ``n`` is the character index at which line
        ``n`` starts. Item 0 is a placeholder and the list ends with
        the length of the content, so that item ``n + 1`` is always
        the end of line ``n``.
    """
    offsets = [-1, 0]
    for line in file_content.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    offsets.append(len(file_content))
    return offsets


def has_statement_body(node: ast.AST) -> bool:
    """Tell whether a node directly holds a list of statements.

    Args:
        node: AST node to inspect.

    Returns:
        True when one of the node fields is a list containing at
        least one statement, as for an exception handler.
    """
    return any(
        isinstance(value, list)
        and any(isinstance(x, ast.stmt) for x in value)
        for _, value in ast.iter_fields(node)
    )


def is_structural(node: ast.AST) -> bool:
    """Tell whether a node is a unit the chunker may split around.

    Args:
        node: AST node to inspect.

    Returns:
        True for statements and for nodes holding statements.
    """
    return isinstance(node, ast.stmt) or has_statement_body(node)


def has_boundaries(node: ast.AST) -> bool:
    """Tell whether a node carries complete source positions.

    Args:
        node: AST node to inspect.

    Returns:
        True when the start and end lines and columns are integers.
    """
    return (
        isinstance(getattr(node, "lineno", None), int)
        and isinstance(getattr(node, "col_offset", None), int)
        and isinstance(getattr(node, "end_lineno", None), int)
        and isinstance(getattr(node, "end_col_offset", None), int)
    )


def get_boundaries(
    node: ast.AST,
    offsets: list[int],
    file_content: str
) -> tuple[int, int]:
    """Convert the positions of a node into character indices.

    The ``ast`` module reports columns as UTF-8 byte offsets. On a
    line holding non-ASCII characters, the byte prefix is decoded to
    count the characters it represents; on an ASCII line the byte
    offset already is the character offset.

    Args:
        node: AST node with complete source positions.
        offsets: Line offsets computed by ``get_line_offsets``.
        file_content: Decoded content of the file.

    Returns:
        The start (inclusive) and end (exclusive) character indices
        of the node in the file content.
    """
    located = cast(LocatedNode, node)

    lineno, col_offset = located.lineno, located.col_offset
    first_line = file_content[offsets[lineno]:offsets[lineno + 1]]
    if first_line.isascii():
        start_index = offsets[lineno] + col_offset
    else:
        start_index = offsets[lineno] + len(
            first_line.encode("utf-8")[:col_offset].decode("utf-8")
        )

    end_lineno, end_col_offset = located.end_lineno, located.end_col_offset
    last_line = file_content[offsets[end_lineno]:offsets[end_lineno + 1]]
    if last_line.isascii():
        end_index = offsets[end_lineno] + end_col_offset
    else:
        end_index = offsets[end_lineno] + len(
            last_line.encode("utf-8")[:end_col_offset].decode("utf-8")
        )

    return (start_index, end_index)


def window_split_node(
    file_path: str,
    start: int,
    end: int,
    offsets: list[int],
    max_chunk_size: int
) -> list[MinimalSource]:
    """Split a range without usable AST structure along its lines.

    Lines are grouped greedily into chunks of at most
    ``max_chunk_size`` characters. A single line longer than that is
    cut into fixed-size pieces.

    Args:
        file_path: Path of the file.
        start: First character index of the range.
        end: Character index following the range.
        offsets: Line offsets computed by ``get_line_offsets``.
        max_chunk_size: Maximum chunk size in characters.

    Returns:
        Contiguous chunks covering the range exactly, in order.
    """
    chunks: list[MinimalSource] = []

    i = bisect_right(offsets, start)
    l_start = start
    while offsets[i - 1] < end:
        l_end = min(offsets[i], end)
        for char_idx in range(l_start, l_end, max_chunk_size):
            source = MinimalSource(
                file_path=file_path,
                first_character_index=char_idx,
                last_character_index=min(l_end, char_idx + max_chunk_size)
            )

            if chunks and get_merged_source_size(
                chunks[-1], source
            ) <= max_chunk_size:
                chunks[-1] = merge_sources(chunks[-1], source)
            else:
                chunks.append(source)
        l_start = l_end
        i += 1
    return chunks


def recursive_chunk(
    node: ast.AST,
    offsets: list[int],
    max_chunk_size: int,
    file_path: str,
    start: int,
    end: int,
    file_content: str
) -> list[MinimalSource]:
    """Chunk a character range by following the AST structure.

    A range that fits in ``max_chunk_size`` is returned whole.
    Otherwise it is divided at the ends of the structural children
    of the node: each child receives the text from the end of its
    previous sibling (thus including its decorators, leading
    comments and blank lines) and the last child extends to the end
    of the range. Children are chunked recursively and neighbouring
    chunks are merged greedily while they fit. A range that is too
    large and has no structural child is split along its lines.

    Args:
        node: AST node the range belongs to.
        offsets: Line offsets computed by ``get_line_offsets``.
        max_chunk_size: Maximum chunk size in characters.
        file_path: Path of the file.
        start: First character index of the range.
        end: Character index following the range.
        file_content: Decoded content of the file.

    Returns:
        Contiguous, non-overlapping chunks covering the range, none
        longer than ``max_chunk_size`` characters.
    """
    if end - start <= max_chunk_size:
        return [MinimalSource(
            file_path=file_path,
            first_character_index=start,
            last_character_index=end
        )]

    chunks: list[MinimalSource] = []
    children = [
        child for child in ast.iter_child_nodes(node)
        if has_boundaries(child) and is_structural(child)
    ]

    if not children:
        return window_split_node(
            file_path=file_path,
            start=start,
            end=end,
            offsets=offsets,
            max_chunk_size=max_chunk_size
        )

    children.sort(key=lambda x: get_boundaries(x, offsets, file_content))

    child_start = start
    for i, child in enumerate(children):
        _, child_end = get_boundaries(child, offsets, file_content)
        for source in recursive_chunk(
            node=child,
            offsets=offsets,
            max_chunk_size=max_chunk_size,
            file_path=file_path,
            start=child_start,
            end=child_end if i != len(children) - 1 else end,
            file_content=file_content
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
    """Split a Python file along its syntactic structure.

    Chunks follow statement boundaries (functions, classes, blocks),
    cover the whole file without overlap and are expressed as
    character indices. A file that cannot be parsed falls back to
    the recursive text splitter.

    Args:
        file_data: Type, path and decoded content of the file.
        max_chunk_size: Maximum chunk size in characters.

    Returns:
        The chunk locations, in file order.
    """
    _, file_path, file_content = file_data

    try:
        node = ast.parse(file_content)
    except (SyntaxError, ValueError):
        return langchain_chunk(file_data, max_chunk_size)

    return recursive_chunk(
        node=node,
        offsets=get_line_offsets(file_content),
        max_chunk_size=max_chunk_size,
        file_path=file_path,
        start=0,
        end=len(file_content),
        file_content=file_content
    )
