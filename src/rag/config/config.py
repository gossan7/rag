from typing import Dict, Set
from pydantic import BaseModel, model_validator
from pathlib import Path
from huggingface_hub import repo_exists

from ..errors import ConfigError
from ..enums import FileType, ChunkingMethod, Backend


class BaseConfig(BaseModel):
    pass


class FileTypeConfig(BaseConfig):
    chunking_method: ChunkingMethod
    enabled: bool


class BackendConfig(BaseConfig):
    enabled: bool
    weight: float

    @model_validator(mode="after")
    def validate_config(self) -> "BackendConfig":
        if self.enabled and self.weight <= 0.0:
            raise ConfigError(
                "Backend weight should be more that 0.0 if enabled"
            )
        return self


class PathsConfig(BaseConfig):
    raw: Path
    processed: Path
    datasets: Path
    output: Path

    @model_validator(mode="after")
    def validate_config(self) -> "PathsConfig":
        for path in (self.raw, self.processed, self.datasets, self.output):
            if path.is_file():
                raise ConfigError(
                    f"Path is not a directory: {path}"
                )
        return self


class IndexingConfig(BaseConfig):
    excludes: Set[str]
    file_types: Dict[FileType, FileTypeConfig]

    @model_validator(mode="after")
    def validate_config(self) -> "IndexingConfig":
        if all((not ftype.enabled for ftype in self.file_types.values())):
            raise ConfigError(
                "At least one file type should be enabled"
            )
        return self


class RetrievalConfig(BaseConfig):
    backends: Dict[Backend, BackendConfig]

    @model_validator(mode="after")
    def validate_config(self) -> "RetrievalConfig":
        if all((not backend.enabled for backend in self.backends.values())):
            raise ConfigError(
                "At least one retrieval backend should be enabled"
            )
        return self


class SearchConfig(BaseConfig):
    auto_translate: bool


class AnswerConfig(BaseConfig):
    model_name: str
    max_context_length: int

    @model_validator(mode="after")
    def validate_config(self) -> "AnswerConfig":
        if not repo_exists(self.model_name):
            raise ConfigError(
                f"Model name \'{self.model_name}\' does not exist."
            )
        if not 2000 <= self.max_context_length <= 20000:
            raise ConfigError(
                "max_context_length should be >= 2000, <= 20000"
            )
        return self


class Config(BaseConfig):
    paths: PathsConfig
    indexing: IndexingConfig
    retrieval: RetrievalConfig
    search: SearchConfig
    answer: AnswerConfig
