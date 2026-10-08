from diskcache import Cache

from typing import ClassVar
from abc import ABC, abstractmethod
from pathlib import Path
from hashlib import sha256
import json

from ...config import Config, load_config, dump_config
from ...enums import Backend, FileType
from ...data_models import MinimalSource
from ...aliases import FileData
from ...errors import IndexingError, SearchError
from ...logger import Logger
from .manifest import Manifest


MANIFEST_FILENAME = "manifest.json"
CONFIG_FILENAME = "config.yaml"
MAX_CACHE_SIZE = 512


class RetrievalBackend(ABC):
    backend_type: ClassVar[Backend]
    batch_size: ClassVar[int] = 2048


# decorators for mode (requires_mode=...)


# _index_is_empty
#   any(os_walk)

# _index_is_valid
#   metadatas here and .indexing not here return true
#   return false

# _load_metadatas
#   if config and manifest in index_path load else none

# _dump_metadatas
#   atomic metadatas dump

# _ensure_shared_ressources
#   if loaded return else load

# _file_in_manifest
# _file_matches_manifest


# abs
# _open_writer
#   if _index_is_empty: create
#   elif _index_is_valid: load
#   open writer if needed by backend

# _close_writer
#   close writer if needed

# abs
# _open_reader
# _close_reader

# abs
# _add_chunk_to_index

# abs
# _remove_file_from_index
# _remove_file_type_from_index
# _remove_all_documents_from_index

# abs
# _commit_index

# _clean_index


# _begin_index
#   if mode is not None: raise
#   _load_metadatas
#   _load_shared_ressources
#   _open_writer
#   create .indexing
#   create new manifest
#   to_commit to 0
#   sets mode to indexing

# @requires_mode(indexing)
# resgister_file
#   add to curr_manifest

# @requires_mode(indexing)
# need_file
#   file not in cached_ftypes
#   file not in manifest
#   file doesnt match manifest
#   else false

# @requires_mode(indexing)
# prepare_file
#   if ftype in cached_ftype and file_in_manifest delete_file

# @requires_mode(indexing)
# add_chunk

# _abort_index
#   sets mode to None
#   _close_writer

# _end_index
#   sets mode to None
#   _clean_index
#   _commit_index
#   _close_writer
#   _dump_metadatas
#   delete .indexing

# index_session
#   context manager (begin, abort, end)


# _begin_search
#   if mode is not None raise
#   if _index_is_empty or not _index_is_valid: raise
#   _open_shared_ressources
#   _open_reader
#   sets mode to searching

# @requires_mode(searching)
# search

# _end_search
#   sets mode to None
#   _close_reader

# search_session
#   context manager (begin, end)
