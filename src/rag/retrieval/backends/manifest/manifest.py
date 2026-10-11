from pydantic import BaseModel


class Manifest(BaseModel):
    """Record of the corpus files held by a backend index.

    Attributes:
        max_chunk_size: Chunk size the index was built with.
        files: SHA-256 hex digest of each indexed file content, by
            file path.
    """

    max_chunk_size: int
    files: dict[str, str]
