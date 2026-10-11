from .rag_error import RagError


class ConfigError(RagError):
    """Error raised when a configuration cannot be loaded or written."""
    pass
