from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..enums import FileType, ChunkingMethod, Backend


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileTypeConfig(BaseConfig):
    chunking_method: ChunkingMethod
    enabled: bool


class BackendConfig(BaseConfig):
    enabled: bool
    weight: float = Field(gt=0.0, le=1.0)
    path: str


class PathsConfig(BaseConfig):
    raw: Path
    processed: Path
    datasets: Path
    output: Path

    @model_validator(mode="after")
    def validate_config(self) -> "PathsConfig":
        for path in (self.raw, self.processed, self.datasets, self.output):
            if path.exists() and not path.is_dir():
                raise ValueError(
                    f"Path is not a directory: {path}"
                )
        return self


class IndexingConfig(BaseConfig):
    excludes: set[str]
    file_types: dict[FileType, FileTypeConfig]

    @model_validator(mode="after")
    def validate_config(self) -> "IndexingConfig":
        if not any((ftype.enabled for ftype in self.file_types.values())):
            raise ValueError(
                "At least one file type should be enabled"
            )
        if any((ftype.chunking_method == ChunkingMethod.AST
                for key, ftype in self.file_types.items()
                if key != FileType.PYTHON)):
            raise ValueError(
                "Chunking method 'ast' can only be used with .py files"
            )
        return self


class RetrievalConfig(BaseConfig):
    backends: dict[Backend, BackendConfig]

    @model_validator(mode="after")
    def validate_config(self) -> "RetrievalConfig":
        if not any((backend.enabled for backend in self.backends.values())):
            raise ValueError(
                "At least one retrieval backend should be enabled"
            )
        return self


class EvaluationConfig(BaseConfig):
    recall: list[int]
    min_iou: float = Field(le=1.0, ge=0.05)

    @model_validator(mode="after")
    def validate_config(self) -> "EvaluationConfig":
        if not self.recall:
            raise ValueError(
                "Evaluation should have at least one value for recall."
            )
        if any((k <= 0 for k in self.recall)):
            raise ValueError(
                "Evaliation recall should all be >= 1."
            )
        return self


class SearchConfig(BaseConfig):
    auto_translate: bool


class AnswerConfig(BaseConfig):
    model_name: str = Field(min_length=3, max_length=50)
    max_context_length: int = Field(ge=2000, le=20000)
    preprompt: str


class Config(BaseConfig):
    paths: PathsConfig
    indexing: IndexingConfig
    retrieval: RetrievalConfig
    evaluation: EvaluationConfig
    search: SearchConfig
    answer: AnswerConfig
