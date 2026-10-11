from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

from typing import Any

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
    """Split a file with the LangChain recursive character splitter.

    Markdown and Python files use the separators of their language
    (headings, code fences, definitions); plain text uses the default
    separators. Consecutive chunks may overlap by up to 100
    characters, or a tenth of ``max_chunk_size`` when that is
    smaller. Sizes are measured in characters and surrounding
    whitespace is not part of the chunks.

    Args:
        file_data: Type, path and decoded content of the file.
        max_chunk_size: Maximum chunk size in characters.

    Returns:
        The chunk locations, located from the start index reported by
        the splitter and the length of each chunk.
    """
    chunks: list[MinimalSource] = []
    ext, file_path, file_content = file_data
    language = LANGUAGE[ext]
    kwargs: dict[str, Any] = {
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
        chunks.append(MinimalSource(
            file_path=file_path,
            first_character_index=start,
            last_character_index=end
        ))
    return chunks
