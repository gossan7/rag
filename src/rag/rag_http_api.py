from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from threading import Lock
from argparse import ArgumentParser
from typing import cast
import sys

from .rag import RAG
from .config import load_config
from .errors import RagError, ConfigError
from .data_models import SearchResults, Answer
from .logger import Logger


class QueryRequest(BaseModel):
    """Body of the search and answer HTTP requests.

    Attributes:
        query: Text of the query, at least one character long.
        k: Maximum number of sources, at least 1 (5 by default).
    """

    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Create the shared RAG service for the lifetime of the server.

    A single ``RAG`` instance built from ``default.yaml`` and a lock
    are stored in the application state at startup. The answer cache
    is closed at shutdown.

    Args:
        app: Application whose state receives the shared resources.

    Yields:
        None, while the server handles requests.
    """
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
    """Convert an application error into a 400 JSON response.

    Args:
        request: Request being handled.
        exc: Application error raised by the endpoint.

    Returns:
        A response whose ``detail`` field holds the error message.
    """
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


@app.post("/search", response_model=SearchResults)
def search_endpoint(
    payload: QueryRequest,
    request: Request,
) -> SearchResults:
    """Return the top-k sources of a query.

    The shared RAG service is used under a lock, so requests are
    processed one at a time.

    Args:
        payload: Query and number of sources requested.
        request: Request giving access to the application state.

    Returns:
        The question and its retrieved sources.
    """
    with request.app.state.lock:
        return cast(SearchResults, request.app.state.rag.search(
            payload.query, payload.k
        ))


@app.post("/answer", response_model=Answer)
def answer_endpoint(
    payload: QueryRequest,
    request: Request,
) -> Answer:
    """Return a generated answer to a query with its sources.

    The shared RAG service is used under a lock, so requests are
    processed one at a time.

    Args:
        payload: Query and number of sources requested.
        request: Request giving access to the application state.

    Returns:
        The question, its retrieved sources and the answer.
    """
    with request.app.state.lock:
        return cast(Answer, request.app.state.rag.answer(
            payload.query, payload.k
        ))


def main() -> None:
    """Parse the command line and run the HTTP server.

    The default configuration is validated first so that an invalid
    file is reported as a plain error message. The server runs a
    single worker and listens on 127.0.0.1:8000 unless ``--host`` or
    ``--port`` say otherwise.

    Raises:
        SystemExit: With status 1 if the configuration is invalid.
    """
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
    try:
        main()
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(2)
