from __future__ import annotations

import asyncio

import httpx

from app.engines.base import EngineSynthesisResult, EngineStatus, EngineVoice, SynthesisMetrics, TtsEngine
from app.engines.manager import EngineManager
from app.http.server import create_http_app
from app.state.session_state import SessionStateStore


class LoadedFakeEngine(TtsEngine):
    def __init__(self) -> None:
        self.last_options: dict | None = None

    def engine_id(self) -> str:
        return "fake"

    def display_name(self) -> str:
        return "Fake"

    def supported_languages(self) -> list[str]:
        return ["pl"]

    def list_voices(self) -> list[EngineVoice]:
        return [EngineVoice(id="default", label="Default", languages=["pl"], default_language="pl")]

    def load(self, device_preference: str | None = None) -> EngineStatus:
        return self.status()

    def unload(self) -> None:
        return None

    def is_loaded(self) -> bool:
        return True

    def status(self) -> EngineStatus:
        return EngineStatus(
            engine_id="fake",
            display_name="Fake",
            state="ready",
            loaded=True,
            device="cpu",
            available_voices=self.list_voices(),
            extra={
                "runtime_options": {
                    "seed": {"type": "integer", "default": 777, "min": 0},
                }
            },
        )

    def synthesize(self, text: str, *, voice: str | None, language: str | None, options: dict | None = None) -> EngineSynthesisResult:
        self.last_options = dict(options or {})
        return EngineSynthesisResult(
            engine_id="fake",
            voice=voice or "default",
            language=language or "pl",
            device="cpu",
            sample_rate=24000,
            channels=1,
            sample_width=2,
            wav_audio=b"RIFF1234",
            pcm_audio=b"\x00\x00\x01\x01",
            backend="fake",
            metrics=SynthesisMetrics(
                load_time_ms=5.0,
                synthesis_time_ms=8.0,
                end_to_end_time_ms=8.0,
                audio_duration_ms=10.0,
                real_time_factor=0.8,
                cold_start=False,
            ),
        )

    def health_payload(self) -> dict:
        return self.status().asdict()


def test_http_endpoints(tmp_path):
    engine = LoadedFakeEngine()
    manager = EngineManager({"fake": engine}, SessionStateStore(tmp_path / "session.json"))
    app = create_http_app(manager)

    async def run_test():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            assert (await client.get("/health")).status_code == 200
            assert (await client.get("/api/engines")).status_code == 200
            assert (await client.post("/api/engines/activate", json={"engine_id": "fake"})).status_code == 200
            options_response = await client.post(
                "/api/engines/options",
                json={"options": {"seed": 4321}},
            )
            assert options_response.status_code == 200
            assert options_response.json()["engine_options"]["seed"] == 4321
            response = await client.post(
                "/api/synthesize",
                json={"text": "hej", "voice": "default", "language": "pl"},
            )
            assert response.status_code == 200
            payload = response.json()
            assert payload["engine_id"] == "fake"
            assert payload["metrics"]["synthesis_time_ms"] == 8.0
            assert engine.last_options == {"seed": 4321}

    asyncio.run(run_test())
