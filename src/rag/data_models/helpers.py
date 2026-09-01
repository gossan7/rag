from .structures import MinimalSource


def get_source_size(source: MinimalSource) -> int:
    return source.last_character_index - source.first_character_index


def can_merge(source1: MinimalSource, source2: MinimalSource) -> bool:
    if source1.file_path != source2.file_path:
        return False

    down_chunk_start = max(
        source1.first_character_index, source2.first_character_index
    )
    up_chunk_end = min(
        source1.last_character_index, source2.last_character_index
    )
    if up_chunk_end < down_chunk_start:
        return False

    return True


def get_merged_source_size(
    source1: MinimalSource, source2: MinimalSource
) -> int:
    assert can_merge(source1, source2)

    first_character_index = min(
        source1.first_character_index, source2.first_character_index
    )
    last_character_index = max(
        source1.last_character_index, source2.last_character_index
    )

    return last_character_index - first_character_index


def merge_sources(source1: MinimalSource, source2: MinimalSource):
    assert can_merge(source1, source2)

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
