from typing import List, Dict
import bm25s

from ..structures import MinimalSource


def bm25_index(
    chunks: List[MinimalSource],
    files: Dict[str, str],
    retrieval_config: Dict[str, bool | str],
    extension: str
):
    if retrieval_config["active"] is False:
        return

    text_chunks = [
        files[source.file_path]
        [source.first_character_index:source.last_character_index]
        for source in chunks
    ]

    retriever = bm25s.BM25()
    tokenized = bm25s.tokenize(text_chunks)
    retriever.index(tokenized)
    retriever.save(f"{retrieval_config["save_path"]}{extension[1:]}")
