from __future__ import annotations

import sys

import pytest

from app.engines.base import EngineError, EngineNotLoadedError
from app.engines.http_engine import HttpEngine


def build_fake_http_engine() -> HttpEngine:
    return HttpEngine(
        engine_id="fake_http",
        display_name="Fake HTTP",
        module_path="tests.fake_isolated_backend",
        class_name="FakeIsolatedEngine",
        python_env_var="TEST_HTTP_PYTHON",
        default_python=sys.executable,
        fallback_languages=["pl"],
        default_port=19091,
    )


def test_http_engine_lifecycle(monkeypatch):
    monkeypatch.setenv("TEST_HTTP_PYTHON", sys.executable)
    engine = build_fake_http_engine()

    try:
        with pytest.raises(EngineNotLoadedError):
            engine.synthesize("hej", voice="default", language="pl")
    except EngineError as err:
        if "Operation not permitted" in str(err):
            pytest.skip("sandbox blocks local test server sockets")
        raise

    status = engine.load()
    assert status.loaded is True
    assert status.state == "ready"

    result = engine.synthesize("hej", voice="default", language="pl")
    assert result.engine_id == "fake_isolated"
    assert result.backend == "fake-isolated"
    assert result.voice == "default"

    engine.unload()
    assert engine.status().loaded is False


class _FakeProcess:
    def poll(self):
        return None


def test_http_engine_status_reports_loading_during_runner_startup():
    engine = build_fake_http_engine()
    engine._process = _FakeProcess()
    engine._port = 19091

    def failing_request(method, path, payload=None, *, timeout=None):
        raise EngineError("Fake HTTP HTTP runner request failed: [Errno 111] Connection refused")

    engine._request = failing_request

    status = engine.status()

    assert status.state == "loading"
    assert status.loading is True
    assert status.loaded is False
    assert "Connection refused" in (status.last_error or "")
