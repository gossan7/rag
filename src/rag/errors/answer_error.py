from .rag_error import RagError


class AnswerError(RagError):
    """Error raised when an answer cannot be generated or saved."""
    pass
