from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from threading import Lock

from .rag import RAG
from .errors import RagError
from .data_models import SearchResults, Answer


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    app.state.rag = RAG("default.yaml")
    app.state.lock = Lock()
    try:
        yield
    finally:
        app.state.rag.answerer.answer_cache.close()


app = FastAPI(lifespan=lifespan)


@app.exception_handler(RagError)
async def rag_error_handler(
    request: Request,
    exc: RagError,
) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


@app.post("/search", response_model=SearchResults)
def search_endpoint(
    payload: QueryRequest,
    request: Request,
) -> SearchResults:
    with request.app.state.lock:
        return request.app.state.rag.search(
            payload.query, payload.k
        )


@app.post("/answer", response_model=Answer)
def answer_endpoint(
    payload: QueryRequest,
    request: Request,
) -> Answer:
    with request.app.state.lock:
        return request.app.state.rag.answer(
            payload.query, payload.k
        )
