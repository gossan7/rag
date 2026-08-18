from enum import Enum


class Backend(Enum):
    TF_IDF = "tfidf"
    BM25 = "bm25"
    CHROMA = "chroma"
