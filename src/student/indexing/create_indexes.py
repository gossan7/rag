from pathlib import Path
from typing import List, Dict
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
import bm25s
import joblib
import chromadb

from ..structures import MinimalSource


def bm25_index(
    chunks: List[MinimalSource],
    files: Dict[str, str],
    retrieval_config: Dict[str, bool | str],
    extension: str
) -> None:
    if retrieval_config["active"] is False:
        return

    text_chunks = [
        files[source.file_path]
        [source.first_character_index:source.last_character_index]
        for source in chunks
    ]

    if not text_chunks:
        return

    save_path = (
        Path(str(retrieval_config["save_path"]))
        / extension.lstrip(".")
    )

    save_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    retriever = bm25s.BM25()
    tokenized = bm25s.tokenize(text_chunks)
    retriever.index(tokenized)
    retriever.save(
        f"{retrieval_config['save_path']}{extension[1:]}"
    )


def tfidf_index(
    chunks: List[MinimalSource],
    files: Dict[str, str],
    retrieval_config: Dict[str, bool | str],
    extension: str,
) -> None:
    if retrieval_config["active"] is False:
        return

    text_chunks = [
        files[source.file_path][
            source.first_character_index:
            source.last_character_index
        ]
        for source in chunks
    ]

    if not text_chunks:
        return

    vectorizer = TfidfVectorizer(
        token_pattern="(?u)\\b\\w+\\b",
    )

    matrix = vectorizer.fit_transform(text_chunks)

    save_path = (
        Path(str(retrieval_config["save_path"]))
        / extension[1:]
    )

    save_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        vectorizer,
        save_path / "vectorizer.joblib",
    )

    save_npz(
        save_path / "matrix.npz",
        matrix,
    )


def chroma_index(
    chunks: List[MinimalSource],
    files: Dict[str, str],
    retrieval_config: Dict[str, bool | str],
    extension: str,
) -> None:
    if retrieval_config["active"] is False:
        return

    save_path = (
        Path(str(retrieval_config["save_path"]))
        / extension.lstrip(".")
    )

    save_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(save_path),
    )

    collection = client.get_or_create_collection(
        name="chunks",
    )

    documents = [
        files[source.file_path][
            source.first_character_index:
            source.last_character_index
        ]
        for source in chunks
    ]

    ids = [
        str(index)
        for index in range(len(chunks))
    ]

    metadatas = [
        {
            "file_path": source.file_path,
            "first_character_index":
                source.first_character_index,
            "last_character_index":
                source.last_character_index,
        }
        for source in chunks
    ]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )
