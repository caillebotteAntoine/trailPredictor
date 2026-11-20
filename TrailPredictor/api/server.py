"""Run the API with uvicorn in a background thread, on a free local port."""

import logging
import socket
import threading
import time

import uvicorn
from fastapi import FastAPI

logger = logging.getLogger(__name__)

LOCALHOST = "127.0.0.1"
STARTUP_TIMEOUT_S = 10.0
SHUTDOWN_TIMEOUT_S = 5.0
_POLL_INTERVAL_S = 0.05


class ServerStartError(RuntimeError):
    """Raised when the server does not start in time."""


def find_free_port(host: str = LOCALHOST) -> int:
    """Return a TCP port that is free on `host`.

    Parameters
    ----------
    host : str, optional
        Interface to bind. Default is ``LOCALHOST``.

    Returns
    -------
    int
        Port chosen by the operating system.

    Notes
    -----
    The port is released before returning, so another process could take it
    in between; this is unlikely on a desktop.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((host, 0))
        port: int = probe.getsockname()[1]
        return port


class BackgroundServer:
    """Uvicorn server running in a daemon thread.

    Parameters
    ----------
    app : fastapi.FastAPI
        Application to serve.
    host : str, optional
        Interface to listen on. Default is ``LOCALHOST``, so the API is not
        reachable from the network.
    port : int or None, optional
        Port to listen on. Default is None: a free port is chosen.
    """

    def __init__(
        self, app: FastAPI, host: str = LOCALHOST, port: int | None = None
    ) -> None:
        self.host = host
        self.port = find_free_port(host) if port is None else port
        # log_config=None: keep the logging configured by the entry point
        config = uvicorn.Config(app, host=self.host, port=self.port, log_config=None)
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run, name="uvicorn", daemon=True
        )

    @property
    def url(self) -> str:
        """Return the base URL of the server.

        Returns
        -------
        str
            ``http://<host>:<port>/``.
        """
        return f"http://{self.host}:{self.port}/"

    def start(self, timeout: float = STARTUP_TIMEOUT_S) -> None:
        """Start the server and wait until it accepts connections.

        Parameters
        ----------
        timeout : float, optional
            Maximum waiting time, in seconds. Default is ``STARTUP_TIMEOUT_S``.

        Raises
        ------
        ServerStartError
            If the server stopped or did not start within `timeout`.
        """
        self._thread.start()
        deadline = time.monotonic() + timeout
        while not self._server.started:
            if not self._thread.is_alive() or time.monotonic() > deadline:
                self.stop()
                raise ServerStartError(f"The server did not start on {self.url}")
            time.sleep(_POLL_INTERVAL_S)
        logger.info("Server started on %s", self.url)

    def stop(self, timeout: float = SHUTDOWN_TIMEOUT_S) -> None:
        """Ask the server to exit and wait for its thread.

        Parameters
        ----------
        timeout : float, optional
            Maximum waiting time, in seconds. Default is ``SHUTDOWN_TIMEOUT_S``.
        """
        self._server.should_exit = True
        if self._thread.is_alive():
            self._thread.join(timeout)
        logger.info("Server on %s stopped", self.url)
