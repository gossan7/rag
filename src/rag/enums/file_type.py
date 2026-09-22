from enum import Enum


class FileType(str, Enum):
    PYTHON = ".py"
    MARKDOWN = ".md"
    TEXT = ".txt"
