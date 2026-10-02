import tantivy

from ...enums import Backend
from ...data_models import MinimalSource
from ...config import Config
from ...logger import Logger
from .retrieval_backend import RetrievalBackend


class BM25Backend(RetrievalBackend):
    backend_type = Backend.BM25

    def __init__(self, config: Config, logger: Logger) -> None:
        self._build_schema()
        super().__init__(config, logger)

    def _build_schema(self) -> None:
        builder = tantivy.SchemaBuilder()
        builder.add_text_field(
            name="file_path",
            stored=True,
            tokenizer_name="raw",
            index_option="basic"
        )
        builder.add_unsigned_field(
            name="first_character_index",
            stored=True,
        )
        builder.add_unsigned_field(
            name="last_character_index",
            stored=True
        )
        builder.add_text_field(
            name="body",
            stored=False,
            tokenizer_name="default",
            index_option="freq"
        )

        self.schema = builder.build()

    def _create_index(self) -> None:
        self.index = tantivy.Index(
            schema=self.schema,
            path=str(self.index_path),
            reuse=False
        )
        self.writer = self.index.writer()

    def _load_index(self) -> None:
        self.index = tantivy.Index(
            schema=self.schema,
            path=str(self.index_path),
            reuse=True
        )
        self.writer = self.index.writer()

    def _add_document_to_index(
        self,
        chunk: MinimalSource,
        chunk_content: str
    ) -> None:
        doc = tantivy.Document(
            file_path=chunk.file_path,
            body=chunk_content,
            first_character_index=chunk.first_character_index,
            last_character_index=chunk.last_character_index
        )
        self.writer.add_document(doc)

    def _delete_file_from_index(self, file_path: str) -> None:
        self.writer.delete_all_documents(
            field_name="file_path",
            field_value=file_path
        )

    def _commit_index(self) -> None:
        self.writer.commit()

    def _save_index(self) -> None:
        self.index.reload()

    def get_index_hits(
        self,
        query: str,
        k: int
    ) -> list[tuple[MinimalSource, float]]:
        parsed_query, _ = self.index.parse_query_lenient(
            query=query,
            default_field_names=["body"],
        )

        searcher = self.index.searcher()
        results = searcher.search(
            query=parsed_query,
            limit=k
        )

        out: list[tuple[MinimalSource, float]] = []
        for score, address in results.hits:
            doc = searcher.doc(address)
            source = MinimalSource(
                file_path=doc["file_path"][0],
                first_character_index=doc["first_character_index"][0],
                last_character_index=doc["last_character_index"][0]
            )
            out.append((source, float(score)))
        return out
