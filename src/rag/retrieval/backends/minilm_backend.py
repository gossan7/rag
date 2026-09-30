from chromadb.api.types import Embeddings
from transformers.utils import logging as transformers_logging
from huggingface_hub.utils import disable_progress_bars
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import chromadb
import torch

from typing import Literal, cast
from functools import cached_property
import warnings

from ...enums import Backend
from ...data_models import MinimalSource
from ...aliases import FileData
from ...config import Config
from ...logger import Logger
from .retrieval_backend import RetrievalBackend


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CPU_BACKEND: Literal['openvino'] = "openvino"

EMBEDDING_BATCH_SIZE = 512
FLUSH_BATCH_SIZE = 1024


class MiniLMBackend(RetrievalBackend):
    backend_type = Backend.MINILM

    def __init__(self, config: Config, logger: Logger) -> None:
        self.ids: list[str] = []
        self.documents: list[str] = []
        self.embeddings: Embeddings = []
        self.metadatas: list[chromadb.Metadata] = []

        super().__init__(config, logger)

    @cached_property
    def model(self) -> SentenceTransformer:
        transformers_logging.disable_progress_bar()
        disable_progress_bars()
        transformers_logging.set_verbosity_error()
        warnings.filterwarnings(
            "ignore",
            category=SyntaxWarning,
            module=r"optimum\.intel\.openvino\.modeling_base",
        )
        load_dotenv()
        self.logger.info("Loading MiniLM...")
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
    def client(self) -> chromadb.ClientAPI:
        return chromadb.PersistentClient(
            self.path,
            settings=chromadb.Settings(anonymized_telemetry=False)
        )

    def _embed(self) -> None:
        if len(self.documents) == 0:
            return

        self.embeddings.extend(
            self.model.encode(
                self.documents,
                batch_size=EMBEDDING_BATCH_SIZE
            ).tolist()
        )
        self.documents.clear()

        if len(self.embeddings) >= FLUSH_BATCH_SIZE:
            self._flush()

    def _flush(self) -> None:
        if len(self.ids) == 0:
            return

        self.collection.add(
            ids=self.ids,
            embeddings=self.embeddings,
            metadatas=self.metadatas
        )

        self.ids.clear()
        self.embeddings.clear()
        self.metadatas.clear()

    def _previous_run_exists(self) -> bool:
        return not self._dir_is_empty()

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

        if len(self.documents) >= EMBEDDING_BATCH_SIZE:
            self._embed()

    def save_index(self) -> None:
        self._embed()
        self._flush()
        self._save_curr_config()
        self._save_curr_manifest()

    def search_index(
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
