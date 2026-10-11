from ..data_models import MinimalSource


def iou(source1: MinimalSource, source2: MinimalSource) -> float:
    """Compute the intersection over union of two source ranges.

    Args:
        source1: First source.
        source2: Second source.

    Returns:
        0 when the sources are in different files, otherwise the
        number of shared characters divided by the number of
        characters covered by either range.
    """
    if source1.file_path != source2.file_path:
        return 0

    start1, end1 = source1.first_character_index, source1.last_character_index
    start2, end2 = source2.first_character_index, source2.last_character_index
    length1 = end1 - start1
    length2 = end2 - start2

    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = length1 + length2 - intersection

    return intersection / union
