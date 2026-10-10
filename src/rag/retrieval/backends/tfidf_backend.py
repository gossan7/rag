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
    backend_type: ClassVar[Backend] = Backend.TFIDF

    def __init__(self, config: Config, logger: Logger) -> None:
        super().__init__(config, logger)
        self.term_generator = xapian.TermGenerator()
        self.database: Database | WritableDatabase | None = None

    def _ensure_shared_resources(self) -> None:
        pass

    def _create_index(self) -> None:
        pass

    def _load_index(self) -> None:
        pass

    def _open_writer(self) -> None:
        self.database = xapian.WritableDatabase(
            str(self.data_path),
            xapian.DB_CREATE_OR_OPEN,
        )

    def _close_writer(self) -> None:
        if self.database is None:
            return
        self.database.close()
        self.database = None

    def _open_reader(self) -> None:
        self.database = xapian.Database(
            str(self.data_path),
            xapian.DB_OPEN
        )

    def _close_reader(self) -> None:
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
        assert self.database is not None
        self.database.delete_document(
            'F' + sha256(file_path.encode("utf-8")).hexdigest()
        )

    def _remove_file_type_from_index(self, file_type: FileType) -> None:
        assert self.database is not None
        self.database.delete_document(
            'E' + sha256(file_type.value.encode("utf-8")).hexdigest()
        )

    def _remove_all_documents_from_index(self) -> None:
        assert self.database is not None
        database = self.database
        self.database = None
        database.close()

        self.database = xapian.WritableDatabase(
            str(self.data_path),
            xapian.DB_CREATE_OR_OVERWRITE,
        )

    def _commit_index(self) -> None:
        assert self.database is not None
        self.database.commit()

    def get_index_hits(
        self,
        query: str,
        k: int
    ) -> list[tuple[Source, float]]:
        assert self.database is not None
        parser = xapian.QueryParser()
        parser.set_database(self.database)

        parsed_query = parser.parse_query(
            query,
            xapian.QueryParser.FLAG_DEFAULT
            | xapian.QueryParser.FLAG_NO_POSITIONS,
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
