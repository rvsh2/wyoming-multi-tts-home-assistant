"""FastAPI application for control panel and debug synthesis."""

from __future__ import annotations

import base64
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.engines.base import EngineError, EngineNotLoadedError
from app.engines.manager import EngineManager

from .models import ActivateEngineRequest, EngineOptionsRequest, SelectEngineRequest, SynthesizeRequest


def _json_response(payload) -> JSONResponse:
    return JSONResponse(payload)


def _bad_request(err: EngineError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(err))


def _server_error(err: Exception) -> HTTPException:
    return HTTPException(status_code=500, detail=str(err))


def _synthesis_payload(result) -> dict:
    payload = result.asdict()
    payload["wav_base64"] = base64.b64encode(result.wav_audio).decode("ascii")
    return payload


def create_http_app(manager: EngineManager) -> FastAPI:
    app = FastAPI(title="wyoming-multi-tts")
    root = Path(__file__).resolve().parent.parent
    static_dir = root / "static"
    template_path = root / "templates" / "index.html"

    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return template_path.read_text(encoding="utf-8")

    @app.get("/health")
    async def health():
        return _json_response(manager.health_payload())

    @app.get("/api/status")
    async def api_status():
        return _json_response(manager.active_status())

    @app.get("/api/engines")
    async def api_engines():
        return _json_response({"engines": manager.list_engines()})

    @app.post("/api/engines/select")
    async def api_select_engine(request: SelectEngineRequest):
        try:
            return _json_response(await manager.select_engine(request.engine_id))
        except EngineError as err:
            raise _bad_request(err) from err

    @app.post("/api/engines/activate")
    async def api_activate_engine(request: ActivateEngineRequest):
        try:
            return _json_response(await manager.activate_engine(request.engine_id))
        except EngineError as err:
            raise _bad_request(err) from err

    @app.post("/api/engines/options")
    async def api_engine_options(request: EngineOptionsRequest):
        try:
            return _json_response(await manager.set_active_engine_options(request.options))
        except EngineError as err:
            raise _bad_request(err) from err

    @app.post("/api/engines/load")
    async def api_load_engine():
        try:
            return _json_response(await manager.load_active_engine())
        except Exception as err:  # pragma: no cover - runtime dependent
            raise _server_error(err) from err

    @app.post("/api/engines/unload")
    async def api_unload_engine():
        return _json_response(await manager.unload_active_engine())

    @app.get("/api/voices")
    async def api_voices():
        return _json_response({"voices": [voice.__dict__ for voice in manager.active_voices()]})

    @app.post("/api/synthesize")
    async def api_synthesize(request: SynthesizeRequest):
        try:
            result = await manager.synthesize(
                text=request.text,
                voice=request.voice,
                language=request.language,
                options=request.options,
            )
        except EngineNotLoadedError as err:
            raise HTTPException(status_code=409, detail=str(err)) from err
        except EngineError as err:
            raise _bad_request(err) from err
        except Exception as err:  # pragma: no cover - runtime dependent
            raise _server_error(err) from err

        return _json_response(_synthesis_payload(result))

    return app
