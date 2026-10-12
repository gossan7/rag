from xapian import Database, WritableDatabase
import xapian

from typing import ClassVar
from hashlib import sha256

from ...config import Config
from ...logger import Logger
from ...enums import Backend, FileType
from ...data_models import Source
from .retrieval_backend import RetrievalBackend


class TFIDFBackend(RetrievalBackend):
    """Lexical retrieval backend scoring chunks with Xapian TF-IDF.

    Each chunk is a Xapian document whose terms come from the chunk
    text and whose data is the JSON serialized source. Two boolean
    terms, derived from the file path and the file type, allow
    removing the chunks of a file or of a file type.
    """

    backend_type: ClassVar[Backend] = Backend.TFIDF
    storage_errors: ClassVar[tuple[type[Exception], ...]] = (
        *RetrievalBackend.storage_errors, xapian.Error
    )

    def __init__(self, config: Config, logger: Logger) -> None:
        """Create the term generator; no database is opened yet.

        Args:
            config: Validated application configuration.
            logger: Logger used by the backend.
        """
        super().__init__(config, logger)
        self.term_generator = xapian.TermGenerator()
        self.database: Database | WritableDatabase | None = None

    def _ensure_shared_resources(self) -> None:
        """Do nothing: this backend has no shared resource."""
        pass

    def _create_index(self) -> None:
        """Do nothing: the database is created when the writer opens."""
        pass

    def _load_index(self) -> None:
        """Do nothing: the database is opened by the reader or writer."""
        pass

    def _open_writer(self) -> None:
        """Open the Xapian database for writing, creating it if needed."""
        self.database = xapian.WritableDatabase(
            str(self.data_path),
            xapian.DB_CREATE_OR_OPEN,
        )

    def _close_writer(self) -> None:
        """Close the writable database when one is open."""
        if self.database is None:
            return
        self.database.close()
        self.database = None

    def _open_reader(self) -> None:
        """Open the existing Xapian database read-only."""
        self.database = xapian.Database(
            str(self.data_path),
            xapian.DB_OPEN
        )

    def _close_reader(self) -> None:
        """Close the read-only database when one is open."""
        if self.database is None:
            return
        self.database.close()
        self.database = None

    def _add_chunk_to_index(
        self,
        file_type: FileType,
        chunk: Source,
        chunk_content: str
    ) -> None:
        """Add a chunk as a Xapian document.

        The text is indexed without term positions. The file path and
        file type are stored as hashed boolean terms prefixed with ``F``
        and ``E``.

        Args:
            file_type: Type of the file the chunk comes from.
            chunk: Location and content hash of the chunk.
            chunk_content: Text of the chunk.
        """
        assert self.database is not None
        file_path = chunk.file_path
        doc = xapian.Document()
        self.term_generator.set_document(doc)
        self.term_generator.index_text_without_positions(chunk_content)
        doc.set_data(chunk.model_dump_json())
        doc.add_boolean_term(
            'F' + sha256(file_path.encode("utf-8")).hexdigest()
        )
        doc.add_boolean_term(
            'E' + sha256(file_type.value.encode("utf-8")).hexdigest()
        )
        self.database.add_document(doc)

    def _remove_file_from_index(self, file_path: str) -> None:
        """Delete the documents carrying the boolean term of a file.

        Args:
            file_path: Path of the corpus file.
        """
        assert self.database is not None
        self.database.delete_document(
            'F' + sha256(file_path.encode("utf-8")).hexdigest()
        )

    def _remove_file_type_from_index(self, file_type: FileType) -> None:
        """Delete the documents carrying the boolean term of a file type.

        Args:
            file_type: File type to remove.
        """
        assert self.database is not None
        self.database.delete_document(
            'E' + sha256(file_type.value.encode("utf-8")).hexdigest()
        )

    def _remove_all_documents_from_index(self) -> None:
        """Replace the database by a new empty writable one."""
        assert self.database is not None
        database = self.database
        self.database = None
        database.close()

        self.database = xapian.WritableDatabase(
            str(self.data_path),
            xapian.DB_CREATE_OR_OVERWRITE,
        )

    def _commit_index(self) -> None:
        """Commit the pending changes of the writable database."""
        assert self.database is not None
        self.database.commit()

    def get_index_hits(
        self,
        query: str,
        k: int
    ) -> list[tuple[Source, float]]:
        """Query the database with the Xapian TF-IDF weighting scheme.

        The query goes through the Xapian query parser with boolean
        operators disabled, so words such as AND or NOT are plain terms.

        Args:
            query: Non-blank text of the query.
            k: Maximum number of hits.

        Returns:
            At most ``k`` pairs made of a source and its TF-IDF weight.
        """
        assert self.database is not None
        parser = xapian.QueryParser()
        parser.set_database(self.database)

        parsed_query = parser.parse_query(
            query,
            xapian.QueryParser.FLAG_PHRASE
            | xapian.QueryParser.FLAG_NO_POSITIONS
        )

        enquire = xapian.Enquire(self.database)
        enquire.set_weighting_scheme(xapian.TfIdfWeight())
        enquire.set_query(parsed_query)

        matches = enquire.get_mset(0, k)

        out: list[tuple[Source, float]] = []

        for match in matches:
            source = Source.model_validate_json(
                match.document.get_data()
            )

            out.append(
                (source, float(match.weight))
            )

        return out
