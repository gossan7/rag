from collections.abc import Iterator
from typing import Any

from ..config import Config
from ..enums import Backend
from ..aliases import FileData
from ..data_models import MinimalSource
from .chunk import chunk_file


def bm25_index():
    pass


def tfidf_index():
    pass


def minilm_index():
    pass


def index_chunk(chunk: MinimalSource, file_content: str):
    pass


BACKENDS = {
    Backend.BM25: "foo",
    Backend.TFIDF: "bar",
    Backend.MINILM: "baz"
}


def index(
    config: Config,
    prev_config: Config,
    manifest: dict[str, Any],
    file_iterator: Iterator[FileData],
    max_chunk_size: int
) -> None:
    backends = [
        BACKENDS[backend]
        for back, config in
        ]
    for backend, backend_config in config.retrieval.backends.items():
        if backend_config.enabled:


    for file_data in file_iterator:
        chunks = chunk_file(config, file_data, max_chunk_size)
        for chunk in chunks:
            