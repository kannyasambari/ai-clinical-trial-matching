"""
backend/logger.py
──────────────────
Structured logging setup.

• JSON-formatted lines → easy ingestion by Datadog, CloudWatch, etc.
• Console handler for local dev.
• Request/response middleware for FastAPI — logs every call with duration.
"""

import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from pythonjsonlogger import jsonlogger

from backend.config import LOG_LEVEL


def setup_logging() -> None:
    """Call once at app startup."""
    handler   = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    root.handlers = [handler]   # replace default handler

    # Quieten noisy libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.WARNING)


async def logging_middleware(request: Request, call_next: Callable) -> Response:
    """FastAPI middleware — attach to app with app.middleware('http')."""
    request_id = str(uuid.uuid4())[:8]
    logger     = logging.getLogger("api.request")

    start = time.perf_counter()
    logger.info(
        "Request started",
        extra={
            "request_id": request_id,
            "method":     request.method,
            "path":       request.url.path,
            "client_ip":  request.client.host if request.client else "unknown",
        },
    )

    try:
        response = await call_next(request)
    except Exception as exc:
        logger.error(
            "Unhandled exception",
            extra={"request_id": request_id, "error": str(exc)},
        )
        raise

    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    logger.info(
        "Request completed",
        extra={
            "request_id":  request_id,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    return response