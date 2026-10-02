from dotenv import load_dotenv

from typing import Literal, cast, TYPE_CHECKING
from functools import cached_property
import warnings

from ...enums import Backend
from ...data_models import MinimalSource
from ...config import Config
from ...logger import Logger
from .retrieval_backend import RetrievalBackend

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from chromadb.api.types import Embeddings
    from chromadb import Metadata, ClientAPI


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CPU_BACKEND: Literal['openvino'] = "openvino"
FLUSH_BATCH_SIZE = 1024


class MiniLMBackend(RetrievalBackend):
    backend_type = Backend.MINILM
    batch_size = 512

    def __init__(self, config: Config, logger: Logger) -> None:
        self.ids: list[str] = []
        self.documents: list[str] = []
        self.embeddings: Embeddings = []
        self.metadatas: list[Metadata] = []
        self.to_flush = 0
        super().__init__(config, logger)

    @cached_property
    def model(self) -> "SentenceTransformer":
        self.logger.info("Loading MiniLM...")
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as transformers_logging
        from huggingface_hub.utils import disable_progress_bars
        transformers_logging.disable_progress_bar()
        disable_progress_bars()
        transformers_logging.set_verbosity_error()
        warnings.filterwarnings(
            "ignore",
            category=SyntaxWarning,
            module=r"optimum\.intel\.openvino\.modeling_base",
        )
        load_dotenv()
        if torch.cuda.is_available():
            model = SentenceTransformer(
                MODEL_NAME,
                device="cuda",
                model_kwargs={"torch_dtype": torch.float16},
            )
        else:
            self.logger.warning(
                "CUDA not available, will default to cpu + OpenVINO."
                " (quantized)"
            )
            model = SentenceTransformer(
                MODEL_NAME,
                backend=CPU_BACKEND,
                device="cpu",
                model_kwargs={
                    "file_name":
                        "openvino/openvino_model_qint8_quantized.xml",
                    "ov_config": {
                        "INFERENCE_PRECISION_HINT": "f16",
                    }
                }
            )
        self.logger.info("MiniLM loaded.")
        return model

    @cached_property
    def client(self) -> "ClientAPI":
        import chromadb
        return chromadb.PersistentClient(
            self.index_path,
            settings=chromadb.Settings(anonymized_telemetry=False)
        )

    def _flush(self) -> None:
        self.collection.add(
            ids=self.ids,
            embeddings=self.embeddings,
            metadatas=self.metadatas
        )

        self.ids.clear()
        self.embeddings.clear()
        self.metadatas.clear()

    def _create_index(self) -> None:
        self.collection = self.client.get_or_create_collection(
            name="minilm",
            configuration={
                "hnsw": {
                    "space": "cosine",
                }
            },
        )

    def _load_index(self) -> None:
        self.collection = self.client.get_collection("minilm")

    def _add_document_to_index(
        self,
        chunk: MinimalSource,
        chunk_content: str
    ) -> None:
        path = chunk.file_path
        start = chunk.first_character_index
        end = chunk.last_character_index

        self.ids.append(f"{path}:{start}:{end}")
        self.documents.append(chunk_content)
        self.metadatas.append({
            "file_path": path,
            "first_character_index": start,
            "last_character_index": end
        })

    def _delete_file_from_index(self, file_path: str) -> None:
        self.collection.delete(where={"file_path": file_path})

    def _commit_index(self) -> None:
        self.embeddings.extend(
            self.model.encode(
                self.documents,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False
            ).tolist()
        )
        self.documents.clear()

        if self.to_flush >= FLUSH_BATCH_SIZE:
            self._flush()
            self.to_flush = 0

    def _save_index(self) -> None:
        if self.to_flush > 0:
            self._flush()
            self.to_flush = 0

    def get_index_hits(
        self,
        query: str,
        k: int
    ) -> list[tuple[MinimalSource, float]]:
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

        metadatas = cast(list, results["metadatas"])
        distances = cast(list, results["distances"])

        out: list[tuple[MinimalSource, float]] = []
        for metadata, distance in zip(metadatas[0], distances[0]):
            out.append((
                MinimalSource(
                    file_path=metadata["file_path"],
                    first_character_index=metadata["first_character_index"],
                    last_character_index=metadata["last_character_index"]
                ),
                -distance
            ))
        return out
