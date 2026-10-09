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
    logger.info("Checking corpus and indexes coherence...")
    if not config.paths.raw.is_dir():
        raise SearchError(
            f"Corpus directory not found: {config.paths.raw}"
        )

    for bkd in backends:
        bkd_name = bkd.backend_type.value
        prev = bkd.prev_config
        if prev is None:
            raise SearchError(f"{bkd_name} index is invalid (fatal).")

        for ftype, prev_ftype_config in prev.indexing.file_types.items():
            ftype_config = config.indexing.file_types[ftype]
            if not ftype_config.enabled and not prev_ftype_config.enabled:
                continue
            if ftype_config.enabled and not prev_ftype_config.enabled:
                logger.warning(
                    f"{bkd_name} is missing {ftype.value} files."
                )
            if not ftype_config.enabled and prev_ftype_config.enabled:
                logger.warning(
                    f"{bkd_name} contains {ftype.value} files."
                )
            if ftype_config.chunking_method \
                    != prev_ftype_config.chunking_method:
                logger.warning(
                    f"{bkd_name} has a different chunking_method "
                    f"for {ftype.value} files than current config."
                )

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
        "Sanity check", file_iterator, n_files
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
            raise SearchError(f"{bkd_name} index is invalid (fatal).")

        missing = len(all_files - bkd.prev_manifest.files.keys())
        incorrect = incorrect_files[bkd.backend_type]
        extra = len(bkd.prev_manifest.files.keys() - all_files)

        if missing or incorrect or extra:
            logger.warning(
                f"{bkd_name} is invalid: "
                f"{missing}/{n_files} missing files, "
                f"{incorrect}/{n_files} changed files, "
                f"{extra} extra files"
            )
