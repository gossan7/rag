from .rag import RAG
from .config import load_config


class RagCLI:
    def __init__(self, config_file: str = "default.yaml") -> None:
        config = load_config(config_file)
        self.rag = RAG(config)

    def index(self, max_chunk_size: int = 2000) -> None:
        self.rag.index(max_chunk_size)
