from pydantic import BaseModel


class MinimalSource(BaseModel):
    chunk_id: int
    file_path: str
    first_character_index: int
    last_character_index: int
