from typing import Dict, Set
from pydantic import BaseModel
from pathlib import Path

from ..enums import FileType, ChunkingMethod, Backend


class PathsConfig(BaseModel):
    raw: Path
    processed: Path
    datasets: Path
    output: Path


class FileTypeConfig(BaseModel):
    chunking_method: ChunkingMethod
    enabled: bool


class BackendConfig(BaseModel):
    enabled: bool


class IndexingConfig(BaseModel):
    excludes: Set[str]
    file_types: Dict[FileType, FileTypeConfig]


class RetrievalConfig(BaseModel):
    backends: Dict[Backend, BackendConfig]


class SearchConfig(BaseModel):
    auto_translate: bool


class AnswerConfig(BaseModel):
    model_name: str
    max_context_length: int


class Config(BaseModel):
    paths: PathsConfig
    indexing: IndexingConfig
    retrieval: RetrievalConfig
    search: SearchConfig
    answer: AnswerConfig
