"""Tests for :mod:`TrailPredictor.api.server`."""

import socket

import httpx

from TrailPredictor.api import create_app
from TrailPredictor.api.server import LOCALHOST, BackgroundServer, find_free_port


def test_find_free_port_can_be_bound() -> None:
    port = find_free_port()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((LOCALHOST, port))

    assert 0 < port < 2**16


def test_background_server_serves_the_api() -> None:
    server = BackgroundServer(create_app())
    server.start()
    try:
        response = httpx.get(f"{server.url}api/health", timeout=5.0)
    finally:
        server.stop()

    assert server.url == f"http://{LOCALHOST}:{server.port}/"
    assert response.json() == {"status": "ok"}


def test_background_server_uses_the_given_port() -> None:
    port = find_free_port()

    assert BackgroundServer(create_app(), port=port).port == port


def test_stopped_server_refuses_connections() -> None:
    server = BackgroundServer(create_app())
    server.start()
    server.stop()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        assert probe.connect_ex((LOCALHOST, server.port)) != 0
