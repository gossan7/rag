from chromadb.api.types import Embeddable, EmbeddingFunction
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)
import chromadb

from typing import cast

from ...enums import Backend
from ...data_models import MinimalSource
from ...aliases import FileData
from ...config import Config
from .retrieval_backend import RetrievalBackend

BATCH_SIZE = 64


class MiniLMBackend(RetrievalBackend):
    backend_type = Backend.MINILM

    def __init__(self, config: Config) -> None:
        self._client: chromadb.ClientAPI | None = None
        self.ids: list[str] = []
        self.documents: list[str] = []
        self.metadatas: list[chromadb.Metadata] = []
        embedding_function = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device="cuda",
            normalize_embeddings=True,
        )
        self.embedding_function = cast(
            EmbeddingFunction[Embeddable],
            embedding_function,
        )
        super().__init__(config)

    @property
    def client(self) -> chromadb.ClientAPI:
        if self._client is None:
            self._client = chromadb.PersistentClient(self.path)
        return self._client

    def _flush(self) -> None:
        if len(self.ids) == 0:
            return

        self.collection.add(
            ids=self.ids,
            documents=self.documents,
            metadatas=self.metadatas
        )

        self.ids.clear()
        self.documents.clear()
        self.metadatas.clear()

    def _previous_run_exists(self) -> bool:
        return not self._dir_is_empty()

    def _create_index(self) -> None:
        self.collection = self.client.create_collection(
            "minilm", embedding_function=self.embedding_function
        )

    def _load_index(self) -> None:
        self.collection = self.client.get_collection(
            "minilm", embedding_function=self.embedding_function
        )

    def delete_indexed_file(self, file_path: str) -> None:
        self.collection.delete(where={"file_path": file_path})

    def add_document(
        self,
        chunk: MinimalSource,
        file_data: FileData,
    ) -> None:
        path = chunk.file_path
        start = chunk.first_character_index
        end = chunk.last_character_index
        _, _, file_content = file_data

        self.ids.append(f"{path}:{start}:{end}")
        self.documents.append(file_content[start:end])
        self.metadatas.append({
            "file_path": path,
            "first_character_index": start,
            "last_character_index": end
        })

        if len(self.ids) >= BATCH_SIZE:
            self._flush()

    def save_index(self) -> None:
        self._flush()
        self._save_curr_config()
        self._save_curr_manifest()

    def search_index(
        self,
        query: str,
        k: int
    ) -> list[tuple[MinimalSource, float]]:
        raise NotImplementedError
