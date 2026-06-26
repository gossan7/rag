from typing import List
from pydantic import BaseModel

from .minimal_source import MinimalSource


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str
