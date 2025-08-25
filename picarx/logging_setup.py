import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import os
from typing import Optional


def _default_log_dir() -> Path:
    # Cross-platform user-scoped log directory: ~/.picarx/logs
    home = Path(os.path.expanduser("~"))
    return home / ".picarx" / "logs"


def init_logger(name: str = "picarx", level: Optional[int] = None, log_dir: Optional[str] = None) -> logging.Logger:
    """Initialize and return a configured logger.

    - Rotates at ~1MB with 3 backups
    - Logs to console and a user-writable file
    - Respects PICARX_LOG_LEVEL and PICARX_LOG_DIR env vars
    """
    logger = logging.getLogger(name)
    # Avoid messages bubbling up to parent loggers (which would duplicate output
    # when both parent and child have handlers).
    logger.propagate = False
    if logger.handlers:
        # already configured
        return logger

    env_level = os.getenv("PICARX_LOG_LEVEL", "INFO").upper()
    level = level if level is not None else getattr(logging, env_level, logging.INFO)
    logger.setLevel(level)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    dir_path = Path(os.getenv("PICARX_LOG_DIR", log_dir) or _default_log_dir())
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(dir_path / f"{name}.log", maxBytes=1_000_000, backupCount=3)
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        # If file handler fails (e.g., permissions), still keep console logging
        pass

    # Final safeguard against duplicate propagation
    logger.propagate = False

    logger.debug("Logger initialized")
    return logger
