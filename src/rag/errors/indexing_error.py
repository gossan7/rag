from .rag_error import RagError


class IndexingError(RagError):
    """Error raised when the corpus cannot be read or indexed."""
    pass
