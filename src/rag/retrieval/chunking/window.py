from ...aliases import FileData
from ...data_models import MinimalSource


def window_chunk(
    file_data: FileData,
    max_chunk_size: int = 2000
) -> list[MinimalSource]:
    """Split a file into consecutive fixed-size windows.

    Args:
        file_data: Type, path and decoded content of the file.
        max_chunk_size: Size of each window in characters.

    Returns:
        Contiguous, non-overlapping chunk locations; only the last
        one may be shorter than ``max_chunk_size``.
    """
    chunks: list[MinimalSource] = []
    _, file_path, file_content = file_data

    file_length = len(file_content)
    for start in range(0, file_length, max_chunk_size):
        end = min(start + max_chunk_size, file_length)
        chunks.append(MinimalSource(
            file_path=file_path,
            first_character_index=start,
            last_character_index=end
        ))
    return chunks
