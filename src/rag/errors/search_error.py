from .rag_error import RagError


class SearchError(RagError):
    """Error raised when an index cannot be opened or searched."""
    pass
