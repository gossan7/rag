from typing import Dict, Set
from pydantic import BaseModel
from pathlib import Path

from ..enums import FileType, ChunkingMethod, Backend


class BaseConfig(BaseModel):
    pass


class FileTypeConfig(BaseConfig):
    chunking_method: ChunkingMethod
    enabled: bool


class BackendConfig(BaseConfig):
    enabled: bool
    weight: float


class PathsConfig(BaseConfig):
    raw: Path
    processed: Path
    datasets: Path
    output: Path


class IndexingConfig(BaseConfig):
    excludes: Set[str]
    file_types: Dict[FileType, FileTypeConfig]


class RetrievalConfig(BaseConfig):
    backends: Dict[Backend, BackendConfig]


class SearchConfig(BaseConfig):
    auto_translate: bool


class AnswerConfig(BaseConfig):
    model_name: str
    max_context_length: int


class Config(BaseConfig):
    paths: PathsConfig
    indexing: IndexingConfig
    retrieval: RetrievalConfig
    search: SearchConfig
    answer: AnswerConfig
