from pydantic import BaseModel, Field, model_validator
import uuid


class MinimalSource(BaseModel, frozen=True):
    file_path: str
    first_character_index: int = Field(ge=0)
    last_character_index: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_source(self) -> "MinimalSource":
        if self.first_character_index >= self.last_character_index:
            raise ValueError(
                "MinimalSource invalid: first_character_index" +
                " should be < last_character_index"
            )
        return self


class Source(MinimalSource, frozen=True):
    chunk_hash: str | None = None


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class SearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: list[Source]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class Answer(SearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int


class RagSearchResults(BaseModel):
    search_results: list[SearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: list[MinimalAnswer]
    k: int


class RagSearchResultsAndAnswer(BaseModel):
    search_results: list[Answer]
    k: int
