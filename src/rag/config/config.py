from pydantic import (
    BaseModel, ConfigDict, Field, field_validator, model_validator
)

from pathlib import Path
from typing import Annotated

from ..enums import FileType, ChunkingMethod, Backend


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def is_inside(child: Path, parent: Path) -> bool:
    return parent in child.parents


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileTypeConfig(BaseConfig):
    chunking_method: ChunkingMethod
    enabled: bool


class BackendConfig(BaseConfig):
    enabled: bool
    weight: float = Field(gt=0.0, le=1.0)
    path: str = Field(min_length=1)

    @field_validator("path")
    @classmethod
    def validate_path(cls, path: str) -> str:
        if not path.strip():
            raise ValueError("path must not be blank")
        return path


class PathsConfig(BaseConfig):
    raw: Path
    processed: Path
    datasets: Path
    output: Path

    @field_validator("raw", "processed", "datasets", "output")
    @classmethod
    def validate_directory(cls, path: Path) -> Path:
        if path.exists() and not path.is_dir():
            raise ValueError(
                f"'{path}' exists but is not a directory"
            )
        return path

    @model_validator(mode="after")
    def validate_config(self) -> "PathsConfig":
        seen: dict[Path, str] = {}
        for name in ("raw", "processed", "datasets", "output"):
            resolved = getattr(self, name).resolve()
            if resolved in seen:
                raise ValueError(
                    f"'{seen[resolved]}' and '{name}' resolve to the same "
                    f"directory: '{display_path(resolved)}'. "
                    "Each one needs its own directory."
                )
            seen[resolved] = name
        return self


class IndexingConfig(BaseConfig):
    excludes: set[str]
    file_types: dict[FileType, FileTypeConfig]

    @field_validator("excludes")
    @classmethod
    def validate_excludes(cls, excludes: set[str]) -> set[str]:
        for name in sorted(excludes):
            if not name.strip() or "/" in name or "\\" in name:
                raise ValueError(
                    f"'{name}' is not a directory name. Entries are "
                    "matched against single directory names, not paths."
                )
        return excludes

    @model_validator(mode="after")
    def validate_config(self) -> "IndexingConfig":
        missing = [
            ftype.value for ftype in FileType if ftype not in self.file_types
        ]
        if missing:
            raise ValueError(
                "'file_types' has no entry for: "
                f"{', '.join(missing)}. Declare every file type and set "
                "'enabled: false' on the ones to skip."
            )
        if not any((ftype.enabled for ftype in self.file_types.values())):
            raise ValueError(
                "At least one file type must be enabled."
            )
        if any((ftype.chunking_method == ChunkingMethod.AST
                for key, ftype in self.file_types.items()
                if key != FileType.PYTHON)):
            raise ValueError(
                "Chunking method 'ast' can only be used with .py files."
            )
        return self


class RetrievalConfig(BaseConfig):
    backends: dict[Backend, BackendConfig]

    @model_validator(mode="after")
    def validate_config(self) -> "RetrievalConfig":
        if not any((backend.enabled for backend in self.backends.values())):
            raise ValueError(
                "At least one retrieval backend must be enabled."
            )
        return self


class EvaluationConfig(BaseConfig):
    recall: list[Annotated[int, Field(ge=1)]] = Field(min_length=1)
    min_iou: float = Field(le=1.0, ge=0.05)


class AnswerConfig(BaseConfig):
    model_name: str = Field(min_length=3, max_length=50)
    path: Path
    thinking_mode: bool
    temperature: float = Field(gt=0.0, le=1.0)
    max_input_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(ge=0)
    preprompt: str


class Config(BaseConfig):
    paths: PathsConfig
    indexing: IndexingConfig
    retrieval: RetrievalConfig
    evaluation: EvaluationConfig
    answer: AnswerConfig

    @model_validator(mode="after")
    def validate_config(self) -> "Config":
        processed = self.paths.processed
        storage: dict[str, Path] = {
            f"retrieval.backends.{backend.value}.path": processed / cfg.path
            for backend, cfg in self.retrieval.backends.items()
        }
        storage["answer.path"] = processed / self.answer.path

        resolved = {
            field: path.resolve() for field, path in storage.items()
        }
        shared = {
            "paths.raw": self.paths.raw.resolve(),
            "paths.datasets": self.paths.datasets.resolve(),
            "paths.output": self.paths.output.resolve(),
        }
        root = processed.resolve()
        problems: list[str] = []

        for field, path in resolved.items():
            shown = display_path(path)
            if storage[field].exists() and not storage[field].is_dir():
                problems.append(
                    f"'{field}' resolves to '{shown}', which exists but is "
                    "not a directory."
                )
            if path == root:
                problems.append(
                    f"'{field}' resolves to 'paths.processed' itself "
                    f"('{shown}'). It must be a dedicated subdirectory."
                )

            for other, other_path in shared.items():
                if path == other_path:
                    problems.append(
                        f"'{field}' and '{other}' resolve to the same "
                        f"directory: '{shown}'."
                    )
                elif is_inside(path, other_path):
                    problems.append(
                        f"'{field}' ('{shown}') is inside '{other}' "
                        f"('{display_path(other_path)}'). Generated indexes "
                        "and caches must be stored outside of it."
                    )
                elif is_inside(other_path, path):
                    problems.append(
                        f"'{other}' ('{display_path(other_path)}') is "
                        f"inside '{field}' ('{shown}'). Deleting that "
                        "index or cache to rebuild it would delete it too."
                    )

        fields = list(resolved)
        for i, field in enumerate(fields):
            for other in fields[i + 1:]:
                if resolved[field] == resolved[other]:
                    problems.append(
                        f"'{field}' and '{other}' resolve to the same "
                        f"directory: '{display_path(resolved[field])}'."
                    )
                    continue
                inner, outer = field, other
                if is_inside(resolved[other], resolved[field]):
                    inner, outer = other, field
                if is_inside(resolved[inner], resolved[outer]):
                    problems.append(
                        f"'{inner}' ('{display_path(resolved[inner])}') is "
                        f"inside '{outer}' "
                        f"('{display_path(resolved[outer])}'). Each "
                        "index and cache needs its own directory."
                    )

        if problems:
            raise ValueError("\n".join(problems))
        return self
