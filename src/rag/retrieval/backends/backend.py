from typing import ClassVar
from abc import ABC, abstractmethod
from hashlib import sha256

from ...config import Config, load_config
from ...enums import FileType, Backend
from ...aliases import FileData


class RetrievalBackend(ABC):
    backend_type: ClassVar[Backend]

    def __init__(self, config: Config):
        self.backend_config = config.retrieval.backends[self.backend_type]
        self.uncached_extensions: set[FileType] = set()
        self.file_cache: dict[str, str] = dict()

        # if anything is already in path, check if it is a valid prewvious run

        # detect if any previous run was done, if true:
        # load previous config (if previous chunking is different for an extension we dont keep anything from that extension)
        # load index
        # get all hashs in a dict
        # as we index files, we check if the extension is in the ones we need to index what ever the previous run was. we check if its already in the hashed files, if the hash has changed we reindex it.
        # when indexing a file, we remove it from the dict

    @abstractmethod
    def create_index(self):
        pass

    @abstractmethod
    def load_index(self):
        pass

    @abstractmethod
    def add_document(self):
        pass

    @abstractmethod
    def save_index(self):
        pass

    def is_cached(self, file_data: FileData):
        ext, file_path, file_content = file_data

        if ext in self.uncached_extensions:
            return False

        if file_path not in self.file_cache:
            return False

        return self.file_cache[file_content] \
            == sha256(file_content.encode("utf-8"))
