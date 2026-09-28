import xapian

from hashlib import sha256

from ...enums import Backend
from ...data_models import MinimalSource
from ...aliases import FileData
from ...config import Config
from ...logger import Logger
from .retrieval_backend import RetrievalBackend


class TFIDFBackend(RetrievalBackend):
    backend_type = Backend.TFIDF

    def __init__(self, config: Config, logger: Logger):
        self.term_generator = xapian.TermGenerator()
        super().__init__(config, logger)

    def _previous_run_exists(self) -> bool:
        try:
            return xapian.Database.check(str(self.path)) == 0
        except Exception:
            return False

    def _create_index(self) -> None:
        self.database = xapian.WritableDatabase(
            str(self.path),
            xapian.DB_CREATE,
        )

    def _load_index(self) -> None:
        self.database = xapian.WritableDatabase(
            str(self.path),
            xapian.DB_OPEN,
        )

    def delete_indexed_file(self, file_path: str) -> None:
        self.database.delete_document(
            sha256(file_path.encode("utf-8")).hexdigest()
            )

    def add_document(
        self,
        chunk: MinimalSource,
        file_data: FileData,
    ) -> None:
        _, file_path, file_content = file_data
        start = chunk.first_character_index
        end = chunk.last_character_index
        doc = xapian.Document()
        self.term_generator.set_document(doc)
        self.term_generator.index_text(file_content[start:end])
        doc.set_data(chunk.model_dump_json())
        doc.add_boolean_term(sha256(file_path.encode("utf-8")).hexdigest())
        self.database.add_document(doc)

    def save_index(self) -> None:
        self.database.commit()
        self._save_curr_config()
        self._save_curr_manifest()

    def search_index(
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
