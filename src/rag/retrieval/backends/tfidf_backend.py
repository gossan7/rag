import xapian

from hashlib import sha256

from ...enums import Backend
from ...data_models import MinimalSource
from ...config import Config
from ...logger import Logger
from .retrieval_backend import RetrievalBackend


class TFIDFBackend(RetrievalBackend):
    backend_type = Backend.TFIDF

    def __init__(self, config: Config, logger: Logger) -> None:
        self.term_generator = xapian.TermGenerator()
        super().__init__(config, logger)

    def _create_index(self) -> None:
        self.database = xapian.WritableDatabase(
            str(self.index_path),
            xapian.DB_CREATE,
        )

    def _load_index(self) -> None:
        self.database = xapian.WritableDatabase(
            str(self.index_path),
            xapian.DB_OPEN,
        )

    def _add_document_to_index(
        self,
        chunk: MinimalSource,
        chunk_content: str
    ) -> None:
        file_path = chunk.file_path
        doc = xapian.Document()
        self.term_generator.set_document(doc)
        self.term_generator.index_text(chunk_content)
        doc.set_data(chunk.model_dump_json())
        doc.add_boolean_term(sha256(file_path.encode("utf-8")).hexdigest())
        self.database.add_document(doc)

    def _delete_file_from_index(self, file_path: str) -> None:
        self.database.delete_document(
            sha256(file_path.encode("utf-8")).hexdigest()
        )

    def _delete_all_documents_from_index(self) -> None:
        self.database.close()
        self.database = xapian.WritableDatabase(
            str(self.index_path),
            xapian.DB_CREATE_OR_OVERWRITE,
        )

    def _commit_index(self) -> None:
        self.database.commit()

    def _save_index(self) -> None:
        self.database.commit()
        self.to_commit = 0

    def get_index_hits(
        self,
        query: str,
        k: int
    ) -> list[tuple[MinimalSource, float]]:
        parser = xapian.QueryParser()
        parser.set_database(self.database)

        parsed_query = parser.parse_query(query)

        enquire = xapian.Enquire(self.database)
        enquire.set_weighting_scheme(xapian.TfIdfWeight())
        enquire.set_query(parsed_query)

        matches = enquire.get_mset(0, k)

        out: list[tuple[MinimalSource, float]] = []

        for match in matches:
            source = MinimalSource.model_validate_json(
                match.document.get_data()
            )

            out.append(
                (source, float(match.weight))
            )

        return out
