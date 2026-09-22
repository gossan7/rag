from .structures import MinimalSource


def get_source_size(source: MinimalSource) -> int:
    return source.last_character_index - source.first_character_index


def can_merge_sources(source1: MinimalSource, source2: MinimalSource) -> bool:
    return source1.file_path == source2.file_path


def merge_sources(source1: MinimalSource, source2: MinimalSource):
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
