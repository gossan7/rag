from enum import Enum


class ChunkingMethod(str, Enum):
    """Chunking strategies, named as in the configuration."""

    AST = "ast"
    LANGCHAIN = "langchain"
    WINDOW = "window"
