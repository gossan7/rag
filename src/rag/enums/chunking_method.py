from enum import Enum


class ChunkingMethod(str, Enum):
    AST = "ast"
    LANGCHAIN = "langchain"
    WINDOW = "window"
