from .rag_error import RagError


class EvaluationError(RagError):
    """Error raised when the evaluation inputs are inconsistent."""
    pass
