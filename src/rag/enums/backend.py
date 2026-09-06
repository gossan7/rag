from enum import Enum


class Backend(str, Enum):
    TFIDF = "tfidf"
    BM25 = "bm25"
    MINILM = "minilm"
