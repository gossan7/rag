from enum import Enum


class Backend(str, Enum):
    TF_IDF = "tfidf"
    BM25 = "bm25"
    CHROMA = "chroma"
