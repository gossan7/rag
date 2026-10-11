from .answer_error import AnswerError


class ModelLoadError(AnswerError):
    """Error raised when the generation model cannot be loaded.

    Unlike other answer errors, it aborts a whole dataset run since
    no further question could be answered.
    """
    pass
