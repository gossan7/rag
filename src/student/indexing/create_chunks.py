from typing import List, Dict, Any
from pathlib import Path
import json

from .load_files import load_files
from .bm25 import bm25_index
from ..structures import MinimalSource


def window_chunk(
    files: Dict[str, str],
    max_chunk_length: int,
    retrieval: Dict[str, bool]
) -> List[MinimalSource]:
    chunks = []
    id = 0
    for path, content in files.items():
        file_length = len(content)
        for i in range(0, file_length, max_chunk_length):
            chunks.append(MinimalSource(
                chunk_id=id,
                file_path=path,
                first_character_index=i,
                last_character_index=min(
                    i + max_chunk_length,
                    file_length
                )
            ))
            id += 1
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
        chunks[extension] = CHUNKING[config["chunking"][extension]["method"]](
            files[extension],
            max_chunk_length,
            config["retrieval"]
        )
        bm25_index(
            chunks[extension],
            files[extension],
            config["retrieval"]["bm25"],
            extension
        )

        json_file = config["chunking"][extension]["save_path"] + \
            extension[1:] + ".json"

        Path(json_file).parent.mkdir(parents=True, exist_ok=True)

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    chunk.chunk_id: {
                        "file_path": chunk.file_path,
                        "first_character_index": chunk.first_character_index,
                        "last_character_index": chunk.last_character_index
                    }
                    for chunk in chunks[extension]
                },
                f,
                indent=2
            )

    return chunks
