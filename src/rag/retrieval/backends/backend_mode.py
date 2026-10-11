from enum import Enum, auto


class BackendMode(Enum):
    """Session currently open on a retrieval backend."""

    INDEXING = auto()
    SEARCHING = auto()
