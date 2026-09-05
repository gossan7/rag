from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

from ...aliases import FileData
from ...enums import FileType
from ...data_models import MinimalSource


LANGUAGE: dict[FileType, Language | None] = {
    FileType.PYTHON: Language.PYTHON,
    FileType.MARKDOWN: Language.MARKDOWN,
    FileType.TEXT: None
}


def langchain_chunk(
    file_data: FileData,
    max_chunk_size: int = 2000
) -> list[MinimalSource]:
    chunks: list[MinimalSource] = []
    ext, file_path, file_content = file_data
    language = LANGUAGE[ext]
    kwargs = {
        "chunk_size": max_chunk_size,
        "chunk_overlap": min(100, max_chunk_size // 10),
        "add_start_index": True
    }

    if language is None:
        splitter = RecursiveCharacterTextSplitter(**kwargs)
    else:
        splitter = RecursiveCharacterTextSplitter.from_language(
            **kwargs,
            language=language,
        )

    documents = splitter.create_documents([file_content])
    for doc in documents:
        start = doc.metadata["start_index"]
        end = start + len(doc.page_content)
        # TEMP
        assert file_content[start:end] == doc.page_content
        chunks.append(MinimalSource(
            file_path=file_path,
            first_character_index=start,
            last_character_index=end
        ))
    return chunks
