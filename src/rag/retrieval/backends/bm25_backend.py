import tantivy

from ...enums import Backend
from ...data_models import MinimalSource
from ...aliases import FileData
from ...config import Config
from ...logger import Logger
from .retrieval_backend import RetrievalBackend


BATCH_SIZE = 2048


class BM25Backend(RetrievalBackend):
    backend_type = Backend.BM25

    def __init__(self, config: Config, logger: Logger) -> None:
        self._build_schema()
        self.to_commit = 0
        super().__init__(config, logger)

    def _build_schema(self):
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

    def _previous_run_exists(self) -> bool:
        return tantivy.Index.exists(str(self.path))

    def _create_index(self):
        if self.to_commit == 0:
            return

        self.index = tantivy.Index(
            schema=self.schema,
            path=str(self.path),
            reuse=False
        )
        self.writer = self.index.writer()

    def _load_index(self):
        self.index = tantivy.Index(
            schema=self.schema,
            path=str(self.path),
            reuse=True
        )
        self.writer = self.index.writer()

    def delete_indexed_file(self, file_path: str) -> None:
        self.writer.delete_documents(
            field_name="file_path",
            field_value=file_path
        )

    def add_document(
        self,
        chunk: MinimalSource,
        file_data: FileData,
    ) -> None:
        _, file_path, file_content = file_data
        start = chunk.first_character_index
        end = chunk.last_character_index

        doc = tantivy.Document(
            file_path=file_path,
            body=file_content[start:end],
            first_character_index=start,
            last_character_index=end
        )
        self.writer.add_document(doc)

        self.to_commit += 1
        if self.to_commit >= BATCH_SIZE:
            self.writer.commit()
            self.to_commit = 0

    def save_index(self):
        self.writer.commit()
        self.index.reload()
        self._save_curr_config()
        self._save_curr_manifest()

    def search_index(
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
            score = score
            doc = searcher.doc(address)
            source = MinimalSource(
                file_path=doc["file_path"][0],
                first_character_index=doc["first_character_index"][0],
                last_character_index=doc["last_character_index"][0]
            )
            out.append((source, float(score)))
        return out
