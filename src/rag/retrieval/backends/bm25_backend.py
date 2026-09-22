import tantivy

from ...enums import Backend
from ...data_models import MinimalSource
from ...aliases import FileData
from .retrieval_backend import RetrievalBackend


class BM25Backend(RetrievalBackend):
    backend_type = Backend.BM25

    def __init__(self, config):
        self._build_schema()
        super().__init__(config)

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
        if file_path in self.prev_file_cache:
            self.writer.delete_documents(
                field_name="file_path",
                field_value=file_path
            )
            del self.prev_file_cache[file_path]

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

    def clean_index(self):
        remaining_files = set(self.prev_file_cache.keys()) \
            - self.computed_files

        for file_path in remaining_files:
            self.writer.delete_documents(
                field_name="file_path",
                field_value=file_path
            )

    def save_index(self):
        self.writer.commit()
        self._save_curr_config()
        self._save_curr_manifest()

    def search_index(self, query: str, k: int):
        pass
