from typing import List
from pydantic import BaseModel, Field
import uuid

from .minimal_source import MinimalSource


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str
