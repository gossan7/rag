from typing import List
from pydantic import BaseModel

from .minimal_search import MinimalSearchResults, MinimalAnswer


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int
