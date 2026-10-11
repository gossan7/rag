from pydantic import (
    BaseModel, ConfigDict, Field, field_validator, model_validator
)

from pathlib import Path
from typing import Annotated

from ..enums import FileType, ChunkingMethod, Backend


def display_path(path: Path) -> str:
    """Return a path relative to the working directory when possible.

    Args:
        path: Path to display, expected to be already resolved.

    Returns:
        The path relative to the resolved current working directory,
        or the path unchanged when it lies outside of it.
    """
    try:
        return str(path.relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def is_inside(child: Path, parent: Path) -> bool:
    """Tell whether a path is strictly located inside another one.

    The comparison is purely lexical (it relies on ``Path.parents``),
    so both paths should be resolved beforehand. A path is not
    considered to be inside itself.

    Args:
        child: Path that may be nested.
        parent: Candidate ancestor directory.

    Returns:
        True when ``parent`` is one of the ancestors of ``child``.
    """
    return parent in child.parents


class BaseConfig(BaseModel):
    """Base class of every configuration section.

    Unknown fields are rejected so that a typo in a YAML key is
    reported instead of being silently ignored.
    """

    model_config = ConfigDict(extra="forbid")


class FileTypeConfig(BaseConfig):
    """Indexing settings of one file extension.

    Attributes:
        chunking_method: Strategy used to split files of this type.
        enabled: Whether files of this type are indexed.
    """

    chunking_method: ChunkingMethod
    enabled: bool


class BackendConfig(BaseConfig):
    """Settings of one retrieval backend.

    Attributes:
        enabled: Whether the backend is used for indexing and search.
        weight: Weight of the backend in the rank fusion, in ]0, 1].
        path: Directory of the backend index, relative to
            ``paths.processed``.
    """

    enabled: bool
    weight: float = Field(gt=0.0, le=1.0)
    path: str = Field(min_length=1)

    @field_validator("path")
    @classmethod
    def validate_path(cls, path: str) -> str:
        """Reject an index path made only of whitespace.

        Args:
            path: Configured index directory name.

        Returns:
            The path, unchanged.

        Raises:
            ValueError: If the path is blank.
        """
        if not path.strip():
            raise ValueError("path must not be blank")
        return path


class PathsConfig(BaseConfig):
    """Directories used by the application.

    The four directories must resolve to distinct locations.

    Attributes:
        raw: Corpus directory that is walked and indexed.
        processed: Root directory of the indexes and caches.
        datasets: Directory of the question datasets.
        output: Directory of the generated result files.
    """

    raw: Path
    processed: Path
    datasets: Path
    output: Path

    @field_validator("raw", "processed", "datasets", "output")
    @classmethod
    def validate_directory(cls, path: Path) -> Path:
        """Reject a configured path that exists but is not a directory.

        A path that does not exist yet is accepted.

        Args:
            path: Configured directory.

        Returns:
            The path, unchanged.

        Raises:
            ValueError: If the path exists and is not a directory.
        """
        if path.exists() and not path.is_dir():
            raise ValueError(
                f"'{path}' exists but is not a directory"
            )
        return path

    @model_validator(mode="after")
    def validate_config(self) -> "PathsConfig":
        """Ensure that no two configured directories are the same one.

        Paths are compared after resolution, so two different spellings
        of the same directory are detected.

        Returns:
            The validated model.

        Raises:
            ValueError: If two entries resolve to the same directory.
        """
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
    """Corpus discovery and chunking settings.

    Every supported file type must be declared, at least one must be
    enabled, and the ``ast`` chunking method is reserved to ``.py``
    files.

    Attributes:
        excludes: Directory names pruned while walking the corpus.
        file_types: Settings of each supported file extension.
    """

    excludes: set[str]
    file_types: dict[FileType, FileTypeConfig]

    @field_validator("excludes")
    @classmethod
    def validate_excludes(cls, excludes: set[str]) -> set[str]:
        """Ensure that every exclusion is a single directory name.

        Args:
            excludes: Configured directory names.

        Returns:
            The exclusions, unchanged.

        Raises:
            ValueError: If an entry is blank or contains a path separator.
        """
        for name in sorted(excludes):
            if not name.strip() or "/" in name or "\\" in name:
                raise ValueError(
                    f"'{name}' is not a directory name. Entries are "
                    "matched against single directory names, not paths."
                )
        return excludes

    @model_validator(mode="after")
    def validate_config(self) -> "IndexingConfig":
        """Check the consistency of the declared file types.

        Returns:
            The validated model.

        Raises:
            ValueError: If a supported file type is not declared, if no
                file type is enabled, or if the ``ast`` chunking method
                is assigned to a file type other than ``.py``.
        """
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
    """Retrieval backends and their fusion weights.

    Attributes:
        backends: Settings of each declared backend. At least one of
            them must be enabled.
    """

    backends: dict[Backend, BackendConfig]

    @model_validator(mode="after")
    def validate_config(self) -> "RetrievalConfig":
        """Ensure that at least one retrieval backend is enabled.

        Returns:
            The validated model.

        Raises:
            ValueError: If every declared backend is disabled.
        """
        if not any((backend.enabled for backend in self.backends.values())):
            raise ValueError(
                "At least one retrieval backend must be enabled."
            )
        return self


class EvaluationConfig(BaseConfig):
    """Settings of the recall@k evaluation.

    Attributes:
        recall: Non-empty list of the k values to report, each >= 1.
        min_iou: Minimum intersection over union, in [0.05, 1], for a
            retrieved source to match an expected one.
    """

    recall: list[Annotated[int, Field(ge=1)]] = Field(min_length=1)
    min_iou: float = Field(le=1.0, ge=0.05)


class AnswerConfig(BaseConfig):
    """Settings of the answer generation.

    The whole section is part of the answer cache key, so changing
    any of its fields invalidates the previously cached answers.

    Attributes:
        model_name: Hugging Face identifier of the generation model.
        path: Directory of the answer cache, relative to
            ``paths.processed``.
        thinking_mode: Whether the chat template enables thinking.
        temperature: Sampling temperature, in ]0, 1].
        max_input_tokens: Maximum size of the prompt, in tokens.
        max_output_tokens: Maximum number of generated tokens.
        preprompt: Optional extra user message sent before the
            sources; ignored when blank.
    """

    model_name: str = Field(min_length=3, max_length=50)
    path: Path
    thinking_mode: bool
    temperature: float = Field(gt=0.0, le=1.0)
    max_input_tokens: int = Field(ge=256)
    max_output_tokens: int = Field(ge=256)
    preprompt: str


class Config(BaseConfig):
    """Complete validated application configuration.

    Besides the validation of each section, the storage directories
    (one per backend index plus the answer cache, all located under
    ``paths.processed``) are checked against each other and against
    the corpus, dataset and output directories.

    Attributes:
        paths: Directories used by the application.
        indexing: Corpus discovery and chunking settings.
        retrieval: Retrieval backends and fusion weights.
        evaluation: Recall@k evaluation settings.
        answer: Answer generation settings.
    """

    paths: PathsConfig
    indexing: IndexingConfig
    retrieval: RetrievalConfig
    evaluation: EvaluationConfig
    answer: AnswerConfig

    @model_validator(mode="after")
    def validate_config(self) -> "Config":
        """Check that the storage directories cannot collide.

        Each backend index directory and the answer cache directory
        must be a dedicated subdirectory of ``paths.processed``: it must
        not be an existing non-directory, ``paths.processed`` itself, or
        equal to, inside of, or a parent of the corpus, dataset or
        output directory. Two storage directories must neither be equal
        nor nested. Every problem found is reported at once.

        Returns:
            The validated model.

        Raises:
            ValueError: If at least one storage directory is unsafe. The
                message holds one line per problem.
        """
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
