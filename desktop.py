"""Entry point of the desktop application: the API in a pywebview window."""

import logging

import webview

from TrailPredictor.api import create_app
from TrailPredictor.api.server import BackgroundServer

logger = logging.getLogger(__name__)

WINDOW_TITLE = "Trail Predictor"
WINDOW_SIZE = (1400, 900)  # px
WINDOW_MIN_SIZE = (900, 600)  # px
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def main() -> None:
    """Start the API server, open the window and stop the server on close."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
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
