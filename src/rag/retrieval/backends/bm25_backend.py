import tantivy

from typing import ClassVar

from ...config import Config
from ...logger import Logger
from ...enums import Backend, FileType
from ...data_models import Source
from .retrieval_backend import RetrievalBackend


class BM25Backend(RetrievalBackend):
    backend_type: ClassVar[Backend] = Backend.BM25

    def __init__(self, config: Config, logger: Logger) -> None:
        super().__init__(config, logger)

        self.schema: tantivy.Schema | None = None
        self.index: tantivy.Index | None = None
        self.writer: tantivy.IndexWriter | None = None
        self.searcher: tantivy.Searcher | None = None

    def _ensure_shared_resources(self) -> None:
        if self.schema is not None:
            return

        builder = tantivy.SchemaBuilder()
        builder.add_text_field(
            name="file_type",
            stored=False,
            tokenizer_name="raw",
            index_option="basic"
        )
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
            name="chunk_hash",
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
        assert self.schema is not None
        self.index = tantivy.Index(
            schema=self.schema,
            path=str(self.data_path),
            reuse=False
        )

    def _load_index(self) -> None:
        assert self.schema is not None
        self.index = tantivy.Index(
            schema=self.schema,
            path=str(self.data_path),
            reuse=True
        )

    def _open_writer(self) -> None:
        assert self.index is not None
        self.writer = self.index.writer()

    def _close_writer(self) -> None:
        if self.writer is None:
            return
        self.writer.wait_merging_threads()
        self.writer = None

    def _open_reader(self) -> None:
        assert self.index is not None
        self.searcher = self.index.searcher()

    def _close_reader(self) -> None:
        if self.searcher is None:
            return
        self.searcher = None

    def _add_chunk_to_index(
        self,
        file_type: FileType,
        chunk: Source,
        chunk_content: str
    ) -> None:
        assert self.writer is not None
        doc = tantivy.Document(
            file_type=file_type.value,
            file_path=chunk.file_path,
            body=chunk_content,
            first_character_index=chunk.first_character_index,
            last_character_index=chunk.last_character_index,
            chunk_hash=chunk.chunk_hash
        )
        self.writer.add_document(doc)

    def _remove_file_from_index(self, file_path: str) -> None:
        assert self.writer is not None
        self.writer.delete_documents(
            field_name="file_path",
            field_value=file_path
        )

    def _remove_file_type_from_index(self, file_type: FileType) -> None:
        assert self.writer is not None
        self.writer.delete_documents(
            field_name="file_type",
            field_value=file_type.value
        )

    def _remove_all_documents_from_index(self) -> None:
        assert self.writer is not None
        self.writer.delete_all_documents()

    def _commit_index(self) -> None:
        assert self.writer is not None
        self.writer.commit()

    def get_index_hits(
        self,
        query: str,
        k: int
    ) -> list[tuple[Source, float]]:
        assert self.index is not None
        assert self.searcher is not None
        parsed_query, _ = self.index.parse_query_lenient(
            query=query,
            default_field_names=["body"],
        )

        results = self.searcher.search(
            query=parsed_query,
            limit=k
        )

        out: list[tuple[Source, float]] = []
        for score, address in results.hits:
            doc = self.searcher.doc(address)
            source = Source(
                file_path=doc["file_path"][0],
                first_character_index=doc["first_character_index"][0],
                last_character_index=doc["last_character_index"][0],
                chunk_hash=doc["chunk_hash"][0]
            )
            out.append((source, float(score)))
        return out
