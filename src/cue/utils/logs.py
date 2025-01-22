import os
import logging
from logging.handlers import TimedRotatingFileHandler

from .console import rich_handler

_logger: logging.Logger = logging.getLogger("mini-agent")
httpx_logger: logging.Logger = logging.getLogger("httpx")


class SimpleFormatter(logging.Formatter):
    def format(self, record):
        # Use only the first character of the levelname
        record.shortlevel = record.levelname[0]
        return super().format(record)


def _basic_config() -> None:
    logging.basicConfig(
        format="[%(asctime)s - %(name)s:%(lineno)d - %(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _setup_development_config()


def _setup_development_config() -> None:
    env = os.environ.get("CUE_LOG", "debug")
    environment = os.getenv("ENVIRONMENT", "development")

    # use dedicated logger for development
    logger = logging.getLogger()

    # Remove and close all existing handlers to prevent duplication and resource leaks
    for handler in logger.handlers[:]:
        handler.close()  # Close the handler to release the file resource
        logger.removeHandler(handler)  # Remove the handler from the logger

    handlers = []
    formatter = SimpleFormatter(
        "[%(asctime)s.%(msecs)03d][%(shortlevel)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    rich_handler.setLevel(logging.DEBUG)
    rich_handler.setFormatter(formatter)
    handlers.append(rich_handler)

    if environment.lower() in ("development", "testing"):
        # Determine the base directory (three levels up)
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        log_dir = os.path.join(base_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        error_log_path = os.path.join(log_dir, "error.log")
        debug_log_path = os.path.join(log_dir, "debug.log")

        # File handler for errors
        error_log_handler = TimedRotatingFileHandler(
            error_log_path, when="midnight", interval=1, backupCount=5, encoding="utf-8"
        )
        error_log_handler.setLevel(logging.ERROR)
        error_log_handler.setFormatter(formatter)
        handlers.append(error_log_handler)

        # TimedRotatingFileHandler for debug logs
        debug_log_handler = TimedRotatingFileHandler(
            debug_log_path, when="midnight", interval=1, backupCount=5, encoding="utf-8"
        )
        debug_log_handler.setLevel(logging.DEBUG)
        debug_log_handler.setFormatter(formatter)
        handlers.append(debug_log_handler)

    # Add all handlers to the root logger
    for handler in handlers:
        logger.addHandler(handler)
    # **Set the root logger level here**
    logger.setLevel(logging.DEBUG if env == "debug" else logging.INFO)
    logging.getLogger("asyncio").setLevel(logging.WARN)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("passlib").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("anthropic").setLevel(logging.INFO)
    logging.getLogger("openai").setLevel(logging.INFO)


def setup_logging() -> None:
    env = os.environ.get("CUE_LOG", "info")
    if env == "debug":
        _basic_config()
        _logger.setLevel(logging.DEBUG)
        httpx_logger.setLevel(logging.INFO)
    elif env == "info":
        _basic_config()
        _logger.setLevel(logging.INFO)
        httpx_logger.setLevel(logging.WARNING)
