from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from threading import Lock
from argparse import ArgumentParser
from typing import cast

from .rag import RAG
from .config import load_config
from .errors import RagError, ConfigError
from .data_models import SearchResults, Answer
from .logger import Logger


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
        return cast(SearchResults, request.app.state.rag.search(
            payload.query, payload.k
        ))


@app.post("/answer", response_model=Answer)
def answer_endpoint(
    payload: QueryRequest,
    request: Request,
) -> Answer:
    with request.app.state.lock:
        return cast(Answer, request.app.state.rag.answer(
            payload.query, payload.k
        ))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    try:
        load_config("default.yaml")
    except ConfigError as e:
        Logger().error(f"HTTP server cannot start:\n{e}")
        raise SystemExit(1) from None

    uvicorn.run(
        "src.rag.rag_http_api:app",
        host=args.host,
        port=args.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
