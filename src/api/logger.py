"""
RAG pipeline logger for structured step-by-step logging.
"""
import logging
import sys
from typing import Any


class RAGLogger:
    """Logger for RAG pipeline steps with consistent formatting."""

    def __init__(self, name: str = "rag"):
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
            )
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def log_step(self, step: int, message: str, **extra: Any) -> None:
        """Log a pipeline step with optional extra context."""
        extra_str = " - ".join(f"{k}={v}" for k, v in extra.items()) if extra else ""
        full_message = f"{step}. {message}"
        if extra_str:
            full_message = f"{full_message} ({extra_str})"
        self._logger.info(full_message)
        sys.stdout.flush()
        sys.stderr.flush()
