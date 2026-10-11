from hashlib import sha256

from .structures import MinimalSource, Source
from ..errors import RagError


def get_source_size(source: MinimalSource) -> int:
    return source.last_character_index - source.first_character_index


def can_merge_sources(source1: MinimalSource, source2: MinimalSource) -> bool:
    return source1.file_path == source2.file_path


def merge_sources(
    source1: MinimalSource,
    source2: MinimalSource
) -> MinimalSource:
    assert can_merge_sources(source1, source2)

    file_path = source1.file_path
    first_character_index = min(
        source1.first_character_index, source2.first_character_index
    )
    last_character_index = max(
        source1.last_character_index, source2.last_character_index
    )

    return MinimalSource(
        file_path=file_path,
        first_character_index=first_character_index,
        last_character_index=last_character_index
    )


def get_merged_source_size(
    source1: MinimalSource, source2: MinimalSource
) -> int:
    return get_source_size(merge_sources(source1, source2))


def merge_scored_sources(
    sources1: dict[Source, float],
    sources2: dict[Source, float]
) -> dict[Source, float]:
    new_sources: dict[Source, float] = {}
    for source, score in sources1.items():
        new_sources.setdefault(source, 0)
        new_sources[source] += score

    for source, score in sources2.items():
        new_sources.setdefault(source, 0)
        new_sources[source] += score
    return new_sources


def source_is_valid(source: Source, content: str | None = None) -> bool:
    file_path = source.file_path
    start = source.first_character_index
    end = source.last_character_index
    chunk_hash = source.chunk_hash

    if content is None:
        try:
            with open(file_path, mode="r", encoding="utf-8") as f:
                file_content = f.read()
        except (OSError, UnicodeDecodeError):
            return False
        if len(file_content) < end:
            return False

        content = file_content[start:end]

    if chunk_hash is None:
        return True

    content_hash = sha256(content.encode("utf-8")).hexdigest()
    return chunk_hash == content_hash


def load_source(source: Source) -> str:
    file_path = source.file_path
    start = source.first_character_index
    end = source.last_character_index

    try:
        with open(file_path, mode="r", encoding="utf-8") as f:
            file_content = f.read()
    except FileNotFoundError as e:
        raise RagError(
            f"Source file '{file_path}' no longer exists."
        ) from e
    except (OSError, UnicodeDecodeError) as e:
        raise RagError(
            f"Cannot read source file '{file_path}': {e}"
        ) from e
    if len(file_content) < end:
        raise RagError(
            f"Source file '{file_path}' is shorter than the indexed range "
            f"[{start}:{end}]: it was modified after indexing."
        )

    chunk_content = file_content[start:end]
    if not source_is_valid(source, content=chunk_content):
        raise RagError(
            f"Source '{file_path}' [{start}:{end}] no longer matches its "
            "indexed content (chunk hash mismatch): the file was modified "
            "after indexing."
        )

    return chunk_content
