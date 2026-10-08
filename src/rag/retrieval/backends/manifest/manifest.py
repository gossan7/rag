from pydantic import BaseModel


class Manifest(BaseModel):
    max_chunk_size: int
    files: dict[str, str]
