"""Shared utilities for Chimera Agent Baseline."""

"""Shared utilities for Chimera Agent Baseline."""

import logging
import os
import warnings

LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"

# Third-party loggers that are excessively noisy at INFO level.
_QUIET_LOGGERS = [
    "vllm",
    "httpx",
    "httpcore",
    "chromadb",
    "mcp",
]


def setup_logging(
    level: str = "INFO",
    worker_id: int = 0,
) -> None:
    """Configure logging for the application.

    Call this once from each entry point (``run.py``, ``inference.py``).
    The MCP server subprocess has its own ``setup_logging`` call since
    it runs in a separate process.

    Args:
        level: Root log level (DEBUG, INFO, WARNING, ERROR). Noisy
            third-party loggers (vLLM, httpx, chromadb) are clamped to
            WARNING unless *level* is DEBUG.
    """

    # Suppress FutureWarnings in this process.
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )

    # Ensure spawned subprocesses (e.g. vLLM EngineCore) also suppress
    # FutureWarnings unless the user explicitly configured otherwise.
    os.environ.setdefault(
        "PYTHONWARNINGS",
        "ignore::FutureWarning",
    )

    # Reduce vLLM subprocess log spam.
    os.environ.setdefault(
        "VLLM_LOGGING_LEVEL",
        "WARNING",
    )

    numeric_level = getattr(
        logging,
        level.upper(),
        logging.INFO,
    )

    log_format = (
        f"%(asctime)s [worker={worker_id}] "
        "%(name)s %(levelname)s %(message)s"
    )

    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        force=True,
    )

    # Quiet down noisy libraries unless we're in DEBUG mode.
    if numeric_level > logging.DEBUG:
        for name in _QUIET_LOGGERS:
            logging.getLogger(name).setLevel(logging.WARNING)
