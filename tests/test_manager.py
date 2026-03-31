from __future__ import annotations

import asyncio

from app.engines.base import EngineSynthesisResult, EngineStatus, EngineVoice, SynthesisMetrics, TtsEngine
from app.engines.manager import EngineManager
from app.state.session_state import SessionState, SessionStateStore


class FakeEngine(TtsEngine):
    def __init__(self, engine_id: str, runtime_options: dict | None = None) -> None:
        self._engine_id = engine_id
        self._loaded = False
        self._runtime_options = runtime_options or {}
        self.last_options: dict | None = None

    def engine_id(self) -> str:
        return self._engine_id

    def display_name(self) -> str:
        return self._engine_id.upper()

    def supported_languages(self) -> list[str]:
        return ["pl"]

    def list_voices(self) -> list[EngineVoice]:
        return [EngineVoice(id="default", label="default", languages=["pl"], default_language="pl")]

    def load(self, device_preference: str | None = None) -> EngineStatus:
        self._loaded = True
        return self.status()

    def unload(self) -> None:
        self._loaded = False

    def is_loaded(self) -> bool:
        return self._loaded

    def status(self) -> EngineStatus:
        return EngineStatus(
            engine_id=self._engine_id,
            display_name=self.display_name(),
            state="ready" if self._loaded else "not_loaded",
            loaded=self._loaded,
            device="cpu",
            available_voices=self.list_voices(),
            extra={"runtime_options": self._runtime_options},
        )

    def synthesize(self, text: str, *, voice: str | None, language: str | None, options: dict | None = None) -> EngineSynthesisResult:
        self.last_options = dict(options or {})
        return EngineSynthesisResult(
            engine_id=self._engine_id,
            voice=voice or "default",
            language=language or "pl",
            device="cpu",
            sample_rate=24000,
            channels=1,
            sample_width=2,
            wav_audio=b"RIFF",
            pcm_audio=b"\x00\x00",
            backend="fake",
            metrics=SynthesisMetrics(
                load_time_ms=1.0,
                synthesis_time_ms=2.0,
                end_to_end_time_ms=2.0,
                audio_duration_ms=1.0,
                real_time_factor=2.0,
                cold_start=False,
            ),
        )

    def health_payload(self) -> dict:
        return self.status().asdict()


def test_manager_switches_engines(tmp_path):
    store = SessionStateStore(tmp_path / "session.json")
    manager = EngineManager(
        {
            "a": FakeEngine("a"),
            "b": FakeEngine("b"),
        },
        store,
    )

    assert manager.active_engine.engine_id() == "a"

    asyncio.run(manager.select_engine("b"))
    assert manager.active_engine.engine_id() == "b"
    restored = store.load(default_engine_id="a")
    assert restored.active_engine_loaded is False
    assert restored.autoload_active_engine is False


def test_manager_persists_session(tmp_path):
    store = SessionStateStore(tmp_path / "session.json")
    manager = EngineManager({"a": FakeEngine("a")}, store)

    asyncio.run(manager.synthesize(text="x", voice="default", language="pl"))

    restored = store.load(default_engine_id="a")
    assert restored.last_voice == "default"
    assert restored.last_language == "pl"
    assert restored.selection_for("a") == ("default", "pl")


def test_activate_engine_marks_loaded_and_autoload(tmp_path):
    store = SessionStateStore(tmp_path / "session.json")
    manager = EngineManager(
        {
            "a": FakeEngine("a"),
            "b": FakeEngine("b"),
        },
        store,
    )

    asyncio.run(manager.activate_engine("b"))

    restored = store.load(default_engine_id="a")
    assert restored.active_engine_id == "b"
    assert restored.active_engine_loaded is True
    assert restored.autoload_active_engine is True


def test_autoload_active_engine_sync_loads_persisted_engine(tmp_path):
    store = SessionStateStore(tmp_path / "session.json")
    store.save(SessionState(active_engine_id="a", autoload_active_engine=True, active_engine_loaded=True))
    manager = EngineManager({"a": FakeEngine("a")}, store)

    payload = manager.autoload_active_engine_sync()

    assert payload is not None
    assert payload["loaded"] is True
    assert payload["ready"] is True


def test_health_payload_contains_engine_summary(tmp_path):
    store = SessionStateStore(tmp_path / "session.json")
    manager = EngineManager({"a": FakeEngine("a")}, store)

    payload = manager.health_payload()

    assert payload["active_engine_id"] == "a"
    assert payload["available_voice_count"] == 1
    assert payload["engines"][0]["engine_id"] == "a"


def test_manager_keeps_last_voice_and_language_per_engine(tmp_path):
    store = SessionStateStore(tmp_path / "session.json")
    manager = EngineManager(
        {
            "a": FakeEngine("a"),
            "b": FakeEngine("b"),
        },
        store,
    )

    asyncio.run(manager.synthesize(text="x", voice="voice-a", language="pl"))
    asyncio.run(manager.activate_engine("b"))
    asyncio.run(manager.synthesize(text="x", voice="voice-b", language="en"))

    restored = store.load(default_engine_id="a")
    assert restored.selection_for("a") == ("voice-a", "pl")
    assert restored.selection_for("b") == ("voice-b", "en")
    assert restored.last_voice == "voice-b"
    assert restored.last_language == "en"


def test_manager_persists_engine_options_and_merges_them_into_synthesis(tmp_path):
    runtime_options = {
        "seed": {"type": "integer", "default": 777, "min": 0},
        "latency": {"type": "select", "default": "balanced", "choices": ["normal", "balanced"]},
        "normalize": {"type": "boolean", "default": True},
        "chunk_length": {"type": "integer", "default": 200, "min": 100, "max": 400},
    }
    engine = FakeEngine("fish_s2_pro", runtime_options=runtime_options)
    store = SessionStateStore(tmp_path / "session.json")
    manager = EngineManager({"fish_s2_pro": engine}, store)

    asyncio.run(
        manager.set_active_engine_options(
            {
                "seed": 1234,
                "latency": "normal",
                "chunk_length": 240,
                "normalize": False,
            }
        )
    )
    asyncio.run(manager.synthesize(text="x", voice="default", language="pl"))

    restored = store.load(default_engine_id="fish_s2_pro")
    assert restored.engine_options["fish_s2_pro"]["seed"] == 1234
    assert restored.engine_options["fish_s2_pro"]["latency"] == "normal"
    assert engine.last_options == {
        "seed": 1234,
        "latency": "normal",
        "chunk_length": 240,
        "normalize": False,
    }
