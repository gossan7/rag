import tantivy

from typing import ClassVar

from ...config import Config
from ...logger import Logger
from ...enums import Backend, FileType
from ...data_models import Source
from .retrieval_backend import RetrievalBackend


class BM25Backend(RetrievalBackend):
    """Lexical retrieval backend scoring chunks with Tantivy BM25.

    Each chunk is a Tantivy document whose body is tokenized and
    indexed with positions but not stored, next to the stored
    source location and chunk hash. The file path and file type are
    indexed as raw terms to allow removing the chunks of a file or
    of a file type.
    """

    backend_type: ClassVar[Backend] = Backend.BM25

    def __init__(self, config: Config, logger: Logger) -> None:
        """Reset the Tantivy handles; nothing is opened yet.

        Args:
            config: Validated application configuration.
            logger: Logger used by the backend.
        """
        super().__init__(config, logger)

        self.schema: tantivy.Schema | None = None
        self.index: tantivy.Index | None = None
        self.writer: tantivy.IndexWriter | None = None
        self.searcher: tantivy.Searcher | None = None

    def _ensure_shared_resources(self) -> None:
        """Build the Tantivy schema once."""
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
            index_option="position"
        )

        self.schema = builder.build()
        self.query_analyzer = (
            tantivy.TextAnalyzerBuilder(tantivy.Tokenizer.simple())
            .filter(tantivy.Filter.remove_long(40))
            .filter(tantivy.Filter.lowercase())
            .build()
        )

    def _create_index(self) -> None:
        """Create a new Tantivy index in the data directory."""
        assert self.schema is not None
        self.index = tantivy.Index(
            schema=self.schema,
            path=str(self.data_path),
            reuse=False
        )

    def _load_index(self) -> None:
        """Open the Tantivy index stored in the data directory."""
        assert self.schema is not None
        self.index = tantivy.Index(
            schema=self.schema,
            path=str(self.data_path),
            reuse=True
        )

    def _open_writer(self) -> None:
        """Create the index writer."""
        assert self.index is not None
        self.writer = self.index.writer()

    def _close_writer(self) -> None:
        """Wait for the merging threads and drop the writer, if any."""
        if self.writer is None:
            return
        self.writer.wait_merging_threads()
        self.writer = None

    def _open_reader(self) -> None:
        """Create a searcher on the current state of the index."""
        assert self.index is not None
        self.searcher = self.index.searcher()

    def _close_reader(self) -> None:
        """Drop the searcher, if any."""
        if self.searcher is None:
            return
        self.searcher = None

    def _add_chunk_to_index(
        self,
        file_type: FileType,
        chunk: Source,
        chunk_content: str
    ) -> None:
        """Add a chunk as a Tantivy document.

        Args:
            file_type: Type of the file the chunk comes from.
            chunk: Location and content hash of the chunk.
            chunk_content: Text of the chunk, indexed as the body.
        """
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
        """Delete the documents whose file path matches exactly.

        Args:
            file_path: Path of the corpus file.
        """
        assert self.writer is not None
        self.writer.delete_documents(
            field_name="file_path",
            field_value=file_path
        )

    def _remove_file_type_from_index(self, file_type: FileType) -> None:
        """Delete the documents of a file type.

        Args:
            file_type: File type to remove.
        """
        assert self.writer is not None
        self.writer.delete_documents(
            field_name="file_type",
            field_value=file_type.value
        )

    def _remove_all_documents_from_index(self) -> None:
        """Delete every document of the index."""
        assert self.writer is not None
        self.writer.delete_all_documents()

    def _commit_index(self) -> None:
        """Commit the pending additions and deletions."""
        assert self.writer is not None
        self.writer.commit()

    def get_index_hits(
        self,
        query: str,
        k: int
    ) -> list[tuple[Source, float]]:
        """Query the body field and return BM25-scored hits.

        The query is not interpreted as Tantivy query syntax: it is
        tokenized and lowercased, over-long tokens are dropped, and the
        remaining tokens are combined as optional terms. A query that
        yields no token gives no hit.

        Args:
            query: Non-blank text of the query.
            k: Maximum number of hits.

        Returns:
            At most ``k`` pairs made of a source and its BM25 score.
        """
        assert self.index is not None
        assert self.searcher is not None
        assert self.schema is not None

        tokens = self.query_analyzer.analyze(query)

        if not tokens:
            return []

        parsed_query = tantivy.Query.boolean_query([
            (
                tantivy.Occur.Should,
                tantivy.Query.term_query(
                    self.schema,
                    "body",
                    token,
                    index_option="freq",
                ),
            )
            for token in tokens
        ])

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
