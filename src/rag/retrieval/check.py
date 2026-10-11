from hashlib import sha256

from ..config import Config
from ..errors import SearchError
from ..logger import Logger
from .backends.retrieval_backend import RetrievalBackend
from .load import file_generator, count_files


def check(
    config: Config,
    backends: list[RetrievalBackend],
    logger: Logger
) -> None:
    """Warn about differences between the corpus and the indexes.

    The stored configuration of each backend is compared with the
    current one (file types enabled or disabled, chunking method),
    then the corpus is read and hashed to count, per backend, the
    files that are not indexed, modified since indexing, or indexed
    but gone. Differences are reported as warnings only: the indexes
    are neither modified nor invalidated, and searching continues on
    the indexed snapshot.

    Args:
        config: Current application configuration.
        backends: Backends with an open search session.
        logger: Logger receiving the progress bar and the warnings.

    Raises:
        SearchError: If the corpus directory is missing or if a
            backend has no loaded configuration or manifest.
        IndexingError: If a corpus file cannot be read.
    """
    if not config.paths.raw.is_dir():
        raise SearchError(
            f"Cannot check the indexes: corpus directory "
            f"'{config.paths.raw}' not found. Check 'paths.raw' in the "
            "configuration."
        )
    logger.info(f"Checking indexes against corpus '{config.paths.raw}'...")

    warnings: list[tuple[str, str]] = []
    for bkd in backends:
        bkd_name = bkd.backend_type.value
        prev = bkd.prev_config
        if prev is None:
            raise SearchError(
                f"Cannot check the {bkd_name} index at '{bkd.index_path}': "
                "its stored configuration is not loaded."
            )

        for ftype, prev_ftype_config in prev.indexing.file_types.items():
            ftype_config = config.indexing.file_types[ftype]
            if not ftype_config.enabled and not prev_ftype_config.enabled:
                continue
            if ftype_config.enabled and not prev_ftype_config.enabled:
                warnings.append((
                    bkd_name,
                    f"{ftype.value} files are enabled in the configuration "
                    "but were not indexed; they will not appear in results."
                ))
            elif not ftype_config.enabled and prev_ftype_config.enabled:
                warnings.append((
                    bkd_name,
                    f"{ftype.value} files are disabled in the configuration "
                    "but are still indexed; they may appear in results."
                ))
            elif ftype_config.chunking_method \
                    != prev_ftype_config.chunking_method:
                warnings.append((
                    bkd_name,
                    f"{ftype.value} files were indexed with chunking method "
                    f"'{prev_ftype_config.chunking_method.value}', but the "
                    "configuration now uses "
                    f"'{ftype_config.chunking_method.value}'."
                ))

    file_types = set((
        ftype for ftype, ftype_config in config.indexing.file_types.items()
        if ftype_config.enabled is True
    ))

    n_files = count_files(
        root=config.paths.raw,
        file_types=file_types,
        excludes=config.indexing.excludes,
    )

    file_iterator = file_generator(
        root=config.paths.raw,
        file_types=file_types,
        excludes=config.indexing.excludes,
    )

    all_files = set()
    incorrect_files = {bkd.backend_type: 0 for bkd in backends}

    for file_data in logger.loading_bar(
        "Checking corpus", file_iterator, n_files
    ):
        _, file_path, file_content = file_data
        all_files.add(file_path)
        file_hash = sha256(file_content.encode("utf-8")).hexdigest()
        for bkd in backends:
            if bkd._file_in_manifest(file_path) and \
                    not bkd._file_matches_manifest(file_path, file_hash):
                incorrect_files[bkd.backend_type] += 1

    for bkd in backends:
        bkd_name = bkd.backend_type.value
        if bkd.prev_manifest is None:
            raise SearchError(
                f"Cannot check the {bkd_name} index at '{bkd.index_path}': "
                "its manifest is not loaded."
            )

        missing = len(all_files - bkd.prev_manifest.files.keys())
        incorrect = incorrect_files[bkd.backend_type]
        extra = len(bkd.prev_manifest.files.keys() - all_files)

        differences: list[str] = []
        if missing:
            differences.append(
                f"{logger.count(missing, 'corpus file')} not indexed"
            )
        if incorrect:
            differences.append(
                f"{logger.count(incorrect, 'file')} modified since indexing"
            )
        if extra:
            differences.append(
                f"{logger.count(extra, 'indexed file')} no longer in the "
                "corpus"
            )
        if differences:
            warnings.append((
                bkd_name,
                "index is out of date with the corpus "
                f"({n_files} files): {', '.join(differences)}."
            ))

    for warning in logger.merge_labels(warnings):
        logger.warning(warning)
    if warnings:
        logger.info(
            "Search continues on the indexed snapshot. "
            "Run the index command to update it."
        )
