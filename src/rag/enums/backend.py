from enum import Enum


class Backend(str, Enum):
    """Retrieval backends, named as in the configuration."""

    TFIDF = "tfidf"
    BM25 = "bm25"
    MINILM = "minilm"
