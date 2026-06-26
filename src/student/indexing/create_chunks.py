from typing import List, Dict, Tuple, Any

from .load_files import load_files
from ..structures import MinimalSource


def window_chunk(
    files: List[Tuple[str, str]],
    max_chunk_length: int
) -> List[MinimalSource]:
    chunks = []
    for path, content in files:
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


CHUNKING = {
    "window": window_chunk,
}


def create_chunks(
    config: Dict[str, Any],
    max_chunk_length: int
) -> Dict[str, List[MinimalSource]]:

    files = load_files(config, max_chunk_length)
    chunks: Dict[str, List[MinimalSource]] = {}
    for extension in files:
        chunks[extension] = CHUNKING[config["chunking_strategy"][extension]](
            files[extension],
            max_chunk_length
        )
    return chunks
