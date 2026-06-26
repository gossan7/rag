from typing import List
from pydantic import BaseModel

from .question import AnsweredQuestion, UnansweredQuestion


class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]
