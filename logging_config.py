from os import getenv
from sys import stdout
from colorlog_formatter import ColorFormatter


def is_debugging():
    return getenv("DEBUG_MODE") == "1"


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": ColorFormatter,
            "format": "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            "datefmt": "%Y-%m-%dT%H:%M:%S%z",
        },
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "stream": stdout,
            "formatter": "default",
        },
    },
    "root": {
        "handlers": ["default"],
        "level": "DEBUG" if is_debugging() else "INFO",
    },
    "loggers": {
        "chat": {"propagate": True},
        "main": {"propagate": True},
        "startup": {"propagate": True},
        "uvicorn": {"propagate": True},
        "uvicorn.error": {"propagate": True},
        "uvicorn.access": {"propagate": True},
        "ultralytics": {"propagate": True},
    },
}
