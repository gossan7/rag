from .rag import RAG


class RagCLI:
    def __init__(self, config_file: str = "default.yaml") -> None:
        self.rag = RAG(config_file)

    def index(self, max_chunk_size: int = 2000) -> None:
        self.rag.index(max_chunk_size)

    def search(self, query: str, k: int = 5) -> None:
        self.rag.search(query, k)
