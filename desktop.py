"""Entry point of the desktop application: the API in a pywebview window."""

import logging
import multiprocessing
import sys
from logging.handlers import RotatingFileHandler

import webview

from TrailPredictor.api import create_app
from TrailPredictor.api.server import BackgroundServer
from TrailPredictor.utils import DATA_DIR

logger = logging.getLogger(__name__)

WINDOW_TITLE = "Trail Predictor"
WINDOW_SIZE = (1400, 900)  # px
WINDOW_MIN_SIZE = (900, 600)  # px
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
LOG_FILE = DATA_DIR / "logs" / "desktop.log"
LOG_FILE_BYTES = 1_000_000
LOG_FILE_BACKUPS = 3


def configure_logging() -> None:
    """Log to the console, when there is one, and to ``LOG_FILE``.

    The packaged app has no console (``sys.stderr`` is None), so the file is
    the only trace of its errors.

    Raises
    ------
    OSError
        If the log file cannot be created and there is no console either.
    """
    handlers: list[logging.Handler] = []
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler())
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            RotatingFileHandler(
                LOG_FILE,
                maxBytes=LOG_FILE_BYTES,
                backupCount=LOG_FILE_BACKUPS,
                encoding="utf-8",
            )
        )
    except OSError:
        if not handlers:
            raise
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=handlers)
    logger.info("Data directory: %s", DATA_DIR)


def main() -> None:
    """Start the API server, open the window and stop the server on close."""
    multiprocessing.freeze_support()  # no-op unless a frozen child process
    configure_logging()
    server = BackgroundServer(create_app())
    server.start()
    try:
        width, height = WINDOW_SIZE
        webview.create_window(
            WINDOW_TITLE,
            server.url,
            width=width,
            height=height,
            min_size=WINDOW_MIN_SIZE,
        )
        webview.start()  # blocks until the window is closed
    finally:
        server.stop()


if __name__ == "__main__":
    main()
