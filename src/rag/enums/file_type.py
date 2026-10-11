from enum import Enum


class FileType(str, Enum):
    """Supported corpus file types, identified by their extension."""

    PYTHON = ".py"
    MARKDOWN = ".md"
    TEXT = ".txt"
