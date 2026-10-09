from dotenv import load_dotenv
from chromadb.config import Settings
import chromadb

from typing import Literal, TYPE_CHECKING, cast, ClassVar
from collections.abc import Callable
from functools import cached_property
import warnings
import os

from ...enums import Backend, FileType
from ...data_models import MinimalSource
from ...config import Config
from ...logger import Logger
from .retrieval_backend import RetrievalBackend

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from chromadb.api import ClientAPI
    from chromadb.api.types import Embeddings, Metadata
    from chromadb.api.models.Collection import Collection


os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "minilm"
CPU_BACKEND: Literal['openvino'] = "openvino"
ENCODING_BATCH_SIZE = 128

warnings.filterwarnings(
    "ignore",
    message=r"invalid escape sequence '\\\.'",
    category=SyntaxWarning,
)


class MiniLMBackend(RetrievalBackend):
    backend_type: ClassVar[Backend] = Backend.MINILM

    def __init__(self, config: Config, logger: Logger) -> None:
        super().__init__(config, logger)

        self.client: ClientAPI | None = None
        self.collection: Collection | None = None
        self.ids: list[str] = []
        self.documents: list[str] = []
        self.metadatas: list[Metadata] = []

    @cached_property
    def model(self) -> "SentenceTransformer":
        self.logger.info("Loading MiniLM...")
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as transformers_logging
        disable_progress_bar = cast(
            Callable[[], None],
            transformers_logging.disable_progress_bar,
        )
        set_verbosity_error = cast(
            Callable[[], None],
            transformers_logging.set_verbosity_error,
        )
        disable_progress_bar()
        set_verbosity_error()
        load_dotenv()

        if torch.cuda.is_available():
            self.logger.info(
                "CUDA available."
            )
            model = SentenceTransformer(
                model_name_or_path=MODEL_NAME,
                device="cuda",
                model_kwargs={"torch_dtype": torch.float16},
            )
        else:
            self.logger.info(
                "CUDA not available, will default to cpu + OpenVINO."
                " (quantized)"
            )
            model = SentenceTransformer(
                model_name_or_path=MODEL_NAME,
                backend=CPU_BACKEND,
                device="cpu",
                model_kwargs={
                    "file_name": "openvino/openvino_model_qint8_quantized.xml"
                }
            )

        self.logger.info("MiniLM loaded.")
        return model

    def _ensure_shared_resources(self) -> None:
        pass

    def _create_index(self) -> None:
        self.client = chromadb.PersistentClient(
            self.data_path,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            configuration={
                "hnsw": {
                    "space": "cosine",
                }
            },
        )

    def _load_index(self) -> None:
        self.client = chromadb.PersistentClient(
            self.data_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_collection(COLLECTION_NAME)

    def _open_writer(self) -> None:
        pass

    def _close_writer(self) -> None:
        pass

    def _open_reader(self) -> None:
        pass

    def _close_reader(self) -> None:
        pass

    def _add_chunk_to_index(
        self,
        file_type: FileType,
        chunk: MinimalSource,
        chunk_content: str
    ) -> None:
        path = chunk.file_path
        start = chunk.first_character_index
        end = chunk.last_character_index

        self.ids.append(f"{path}:{start}:{end}")
        self.documents.append(chunk_content)
        self.metadatas.append({
            "file_type": file_type.value,
            "file_path": path,
            "first_character_index": start,
            "last_character_index": end
        })

    def _remove_file_from_index(self, file_path: str) -> None:
        assert self.collection is not None
        self.collection.delete(where={"file_path": file_path})

    def _remove_file_type_from_index(self, file_type: FileType) -> None:
        assert self.collection is not None
        self.collection.delete(where={"file_type": file_type.value})

    def _remove_all_documents_from_index(self) -> None:
        assert self.client is not None
        self.client.delete_collection(name=COLLECTION_NAME)
        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            configuration={
                "hnsw": {
                    "space": "cosine",
                }
            },
        )

    def _commit_index(self) -> None:
        assert self.client is not None
        assert self.collection is not None

        if not self.documents:
            return

        embeddings: Embeddings = self.model.encode(
            self.documents,
            batch_size=ENCODING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        batch_size = self.client.get_max_batch_size()
        for start in range(0, len(self.ids), batch_size):
            end = min(start + batch_size, len(self.ids))
            self.collection.add(
                ids=self.ids[start:end],
                embeddings=embeddings[start:end],
                metadatas=self.metadatas[start:end],
            )

        self.ids.clear()
        self.documents.clear()
        self.metadatas.clear()

    def get_index_hits(
        self,
        query: str,
        k: int
    ) -> list[tuple[MinimalSource, float]]:
        assert self.collection is not None
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k,
            include=["metadatas", "distances"],
        )

        metadatas = cast(
            list[list[dict[str, str | int]]],
            results["metadatas"]
        )
        distances = cast(list[list[float]], results["distances"])

        out: list[tuple[MinimalSource, float]] = []
        for metadata, distance in zip(metadatas[0], distances[0]):
            file_path = cast(str, metadata["file_path"])
            start = cast(int, metadata["first_character_index"])
            end = cast(int, metadata["last_character_index"])

            out.append((
                MinimalSource(
                    file_path=file_path,
                    first_character_index=start,
                    last_character_index=end
                ),
                -distance
            ))
        return out
