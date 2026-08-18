from enum import Enum


class ChunkingMethod(Enum):
    AST = "ast"
    LANGCHAIN = "langchain"
    WINDOW = "window"
