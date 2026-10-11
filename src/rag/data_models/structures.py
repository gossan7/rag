from pydantic import BaseModel, Field, model_validator
import uuid


class MinimalSource(BaseModel, frozen=True):
    """Location of a chunk inside a corpus file.

    The model is frozen, hence hashable. Indices are Unicode
    character indices in the decoded file, with an exclusive end,
    and the range is never empty.

    Attributes:
        file_path: Path of the file, written as in the corpus walk.
        first_character_index: Index of the first character.
        last_character_index: Index following the last character.
    """

    file_path: str
    first_character_index: int = Field(ge=0)
    last_character_index: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_source(self) -> "MinimalSource":
        """Ensure that the character range is not empty or reversed.

        Returns:
            The validated model.

        Raises:
            ValueError: If the start index is not lower than the end one.
        """
        if self.first_character_index >= self.last_character_index:
            raise ValueError(
                "MinimalSource invalid: first_character_index" +
                " should be < last_character_index"
            )
        return self


class Source(MinimalSource, frozen=True):
    """Source location carrying the hash of its indexed content.

    Attributes:
        chunk_hash: SHA-256 hex digest of the UTF-8 encoded chunk as
            it was at indexing time, or None when unknown (for
            instance for results produced by another tool).
    """

    chunk_hash: str | None = None


class UnansweredQuestion(BaseModel):
    """Question waiting for retrieval or generation.

    Attributes:
        question_id: Identifier, a random UUID4 when not provided.
        question: Text of the question.
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Question with its reference sources and answer.

    Attributes:
        sources: Expected source locations, used as ground truth.
        answer: Reference answer.
    """

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Dataset of questions, answered or not.

    Attributes:
        rag_questions: Questions of the dataset.
    """

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Retrieved sources of one question, in the subject format.

    Attributes:
        question_id: Identifier of the question.
        question: Text of the question.
        retrieved_sources: Sources ordered from most to least relevant.
    """

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class SearchResults(BaseModel):
    """Retrieved sources of one question, with chunk hashes.

    Same fields as ``MinimalSearchResults``, but each source may
    carry the hash used to verify it against the current file.

    Attributes:
        question_id: Identifier of the question.
        question: Text of the question.
        retrieved_sources: Sources ordered from most to least relevant.
    """

    question_id: str
    question: str
    retrieved_sources: list[Source]


class MinimalAnswer(MinimalSearchResults):
    """Search results of one question with its answer.

    Attributes:
        answer: Generated answer.
    """

    answer: str


class Answer(SearchResults):
    """Hash-carrying search results of one question with its answer.

    Attributes:
        answer: Generated answer.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """Search results of a whole dataset, in the subject format.

    Attributes:
        search_results: Results of each question.
        k: Number of results requested per question.
    """

    search_results: list[MinimalSearchResults]
    k: int


class RagSearchResults(BaseModel):
    """Search results of a whole dataset, with chunk hashes.

    This is the model read and written by the application. It is a
    superset of ``StudentSearchResults``: sources only gain an
    optional ``chunk_hash`` field.

    Attributes:
        search_results: Results of each question.
        k: Number of results requested per question.
    """

    search_results: list[SearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Answers of a whole dataset, in the subject format.

    Attributes:
        search_results: Results and answer of each question.
        k: Number of results requested per question.
    """

    search_results: list[MinimalAnswer]
    k: int


class RagSearchResultsAndAnswer(BaseModel):
    """Answers of a whole dataset, with chunk hashes.

    This is the model written by the application. It is a superset
    of ``StudentSearchResultsAndAnswer``.

    Attributes:
        search_results: Results and answer of each question.
        k: Number of results requested per question.
    """

    search_results: list[Answer]
    k: int
