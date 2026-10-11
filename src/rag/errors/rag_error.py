class RagError(Exception):
    """Base class of the expected, user-facing application errors.

    The command-line entry point reports these errors as a message
    and a non-zero exit status, and the HTTP API as a 400 response.
    """
    pass
