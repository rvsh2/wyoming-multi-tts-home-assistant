"""Engine manager for active backend selection and lifecycle."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
import threading
from typing import Any

from app.state.session_state import SessionStateStore
from app.system_resources import engine_memory_hint, resource_usage_payload

from .base import EngineError, EngineSynthesisResult, EngineVoice, TtsEngine


class EngineManager:
    def __init__(self, registry: dict[str, TtsEngine], state_store: SessionStateStore) -> None:
        if not registry:
            raise ValueError("Engine registry cannot be empty")
        self.registry = registry
        self.state_store = state_store
        self._async_lock: asyncio.Lock | None = None
        self._async_lock_loop: asyncio.AbstractEventLoop | None = None
        self._state_lock = threading.RLock()
        default_engine_id = next(iter(registry))
        self.session = state_store.load(default_engine_id=default_engine_id)
        session_updated = False
        if self.session.active_engine_id == "qwen_tts_polish":
            if self.session.last_language == "pl":
                self.session.last_language = "en"
                session_updated = True
            if self.session.last_voice == "polish_speaker":
                self.session.last_voice = None
                session_updated = True
        if self.session.active_engine_id not in registry:
            self.session.active_engine_id = default_engine_id
            self.session.autoload_active_engine = False
            self.session.active_engine_loaded = False
            session_updated = True
        if session_updated:
            self.state_store.save(self.session)

    @asynccontextmanager
    async def locked(self):
        lock = self._get_async_lock()
        async with lock:
            yield

    def _get_async_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._async_lock is None or self._async_lock_loop is not loop:
            self._async_lock = asyncio.Lock()
            self._async_lock_loop = loop
        return self._async_lock

    async def _run_blocking(self, func):
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        def runner() -> None:
            try:
                result = func()
            except Exception as err:
                loop.call_soon_threadsafe(future.set_exception, err)
                return
            loop.call_soon_threadsafe(future.set_result, result)

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        return await future

    @property
    def active_engine(self) -> TtsEngine:
        return self.registry[self.session.active_engine_id]

    @staticmethod
    def _normalize_voice(voice: EngineVoice | dict[str, Any]) -> EngineVoice:
        if isinstance(voice, EngineVoice):
            return voice
        return EngineVoice.fromdict(voice)

    def _engine_memory_hint(self, engine: TtsEngine, device: str | None, *, prefer_gpu: bool = False) -> str | None:
        return engine_memory_hint(
            engine.engine_id(),
            device,
            prefer_gpu=prefer_gpu,
        )

    def _engine_summary(self, engine: TtsEngine, *, prefer_gpu: bool = False) -> dict[str, Any]:
        status = engine.status().asdict()
        status_extra = dict(status.get("extra", {}))
        status_extra["memory_hint"] = self._engine_memory_hint(
            engine,
            status.get("device"),
            prefer_gpu=prefer_gpu,
        )
        status["extra"] = status_extra
        selected = engine.engine_id() == self.session.active_engine_id
        active = selected and bool(status["loaded"]) and status["state"] == "ready"
        return {
            "engine_id": engine.engine_id(),
            "display_name": engine.display_name(),
            "status": status,
            "selected": selected,
            "active": active,
        }

    def _health_engine_entry(self, engine: TtsEngine) -> dict[str, Any]:
        status = engine.status()
        return {
            "engine_id": engine.engine_id(),
            "display_name": engine.display_name(),
            "selected": engine.engine_id() == self.session.active_engine_id,
            "state": status.state,
            "loaded": status.loaded,
            "device": status.device,
            "memory_hint": self._engine_memory_hint(engine, status.device),
        }

    def _engine_runtime_options(self, engine: TtsEngine) -> dict[str, dict[str, Any]]:
        runtime_options = engine.status().extra.get("runtime_options", {})
        if not isinstance(runtime_options, dict):
            return {}
        return {
            str(option_name): dict(option_spec)
            for option_name, option_spec in runtime_options.items()
            if isinstance(option_spec, dict)
        }

    def _engine_option_defaults(self, engine: TtsEngine) -> dict[str, Any]:
        defaults: dict[str, Any] = {}
        for option_name, option_spec in self._engine_runtime_options(engine).items():
            if "default" in option_spec:
                defaults[option_name] = option_spec["default"]
        return defaults

    def _persisted_engine_options(self, engine_id: str) -> dict[str, Any]:
        persisted = self.session.engine_options.get(engine_id, {})
        if not isinstance(persisted, dict):
            return {}
        return dict(persisted)

    def _resolved_engine_options(self, engine: TtsEngine) -> dict[str, Any]:
        resolved = self._engine_option_defaults(engine)
        resolved.update(self._persisted_engine_options(engine.engine_id()))
        return resolved

    @staticmethod
    def _coerce_engine_option_value(option_name: str, option_spec: dict[str, Any], value: Any) -> Any:
        option_type = option_spec.get("type")
        if option_type == "boolean":
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"false", "0", "off", "no", ""}:
                    return False
                if normalized in {"true", "1", "on", "yes"}:
                    return True
            return bool(value)
        if option_type == "integer":
            if value in (None, ""):
                return option_spec.get("default")
            try:
                coerced = int(value)
            except (TypeError, ValueError) as err:
                raise EngineError(f"Invalid integer value for '{option_name}'") from err
            minimum = option_spec.get("min")
            maximum = option_spec.get("max")
            if minimum is not None and coerced < int(minimum):
                raise EngineError(f"Value for '{option_name}' must be >= {minimum}")
            if maximum is not None and coerced > int(maximum):
                raise EngineError(f"Value for '{option_name}' must be <= {maximum}")
            return coerced
        if option_type == "select":
            choices = [str(choice) for choice in option_spec.get("choices", [])]
            coerced = str(value) if value is not None else str(option_spec.get("default", ""))
            if choices and coerced not in choices:
                raise EngineError(f"Invalid value for '{option_name}'")
            return coerced
        if option_type == "string":
            if value is None:
                return str(option_spec.get("default", ""))
            return str(value)
        return value

    def _sanitize_engine_options(self, engine: TtsEngine, options: dict[str, Any]) -> dict[str, Any]:
        runtime_options = self._engine_runtime_options(engine)
        if not runtime_options:
            if options:
                raise EngineError(f"{engine.display_name()} does not expose configurable runtime options")
            return {}
        sanitized: dict[str, Any] = {}
        for option_name, option_spec in runtime_options.items():
            if option_name not in options:
                continue
            sanitized[option_name] = self._coerce_engine_option_value(
                option_name,
                option_spec,
                options[option_name],
            )
        return sanitized

    def list_engines(self) -> list[dict[str, Any]]:
        prefer_gpu = resource_usage_payload(self.active_engine.status().device).get("kind") == "vram"
        return [
            self._engine_summary(engine, prefer_gpu=prefer_gpu)
            for engine in self.registry.values()
        ]

    def active_status(self) -> dict[str, Any]:
        payload = self.active_engine.status().asdict()
        payload_extra = dict(payload.get("extra", {}))
        payload_extra["memory_hint"] = engine_memory_hint(
            self.session.active_engine_id,
            payload.get("device"),
            prefer_gpu=resource_usage_payload(payload.get("device")).get("kind") == "vram",
        )
        payload["extra"] = payload_extra
        payload["resource_usage"] = resource_usage_payload(payload.get("device"))
        payload["active_engine_id"] = self.session.active_engine_id
        payload["last_voice"] = self.session.last_voice
        payload["last_language"] = self.session.last_language
        payload["autoload_active_engine"] = self.session.autoload_active_engine
        payload["active_engine_loaded"] = self.session.active_engine_loaded
        payload["engine_options"] = self._resolved_engine_options(self.active_engine)
        payload["runtime_options"] = self._engine_runtime_options(self.active_engine)
        payload["selected"] = True
        payload["ready"] = bool(payload["loaded"]) and payload["state"] == "ready"
        return payload

    def health_payload(self) -> dict[str, Any]:
        active = self.active_engine.status()
        active_voices = [self._normalize_voice(voice) for voice in active.available_voices]
        resource_usage = resource_usage_payload(active.device)
        return {
            "status": "ok" if active.loaded and active.state == "ready" else "degraded",
            "ready": active.loaded and active.state == "ready",
            "active_engine_id": self.session.active_engine_id,
            "active_engine_name": active.display_name,
            "active_engine_state": active.state,
            "device": active.device,
            "resource_usage": resource_usage,
            "load_time_ms": active.load_time_ms,
            "last_error": active.last_error,
            "available_voice_count": len(active_voices),
            "available_voices": [voice.id for voice in active_voices],
            "last_voice": self.session.last_voice,
            "last_language": self.session.last_language,
            "engine_options": self._resolved_engine_options(self.active_engine),
            "engines": [self._health_engine_entry(engine) for engine in self.registry.values()],
        }

    def should_autoload(self) -> bool:
        return self.session.autoload_active_engine and self.session.active_engine_loaded

    def _save_session(self) -> None:
        with self._state_lock:
            self.session.last_voice, self.session.last_language = self.session.selection_for(self.session.active_engine_id)
            self.state_store.save(self.session)

    async def select_engine(self, engine_id: str) -> dict[str, Any]:
        async with self.locked():
            if engine_id not in self.registry:
                raise EngineError(f"Unknown engine '{engine_id}'")
            if engine_id == self.session.active_engine_id:
                return self.active_status()
            current_engine = self.active_engine
            await self._run_blocking(current_engine.unload)
            self.session.active_engine_id = engine_id
            self.session.active_engine_loaded = False
            self.session.autoload_active_engine = False
            self._save_session()
            return self.active_status()

    async def activate_engine(self, engine_id: str) -> dict[str, Any]:
        async with self.locked():
            if engine_id not in self.registry:
                raise EngineError(f"Unknown engine '{engine_id}'")

            current_engine_id = self.session.active_engine_id
            target_engine = self.registry[engine_id]

            if engine_id != current_engine_id:
                current_engine = self.active_engine
                await self._run_blocking(current_engine.unload)

            status = await self._run_blocking(target_engine.load)
            self.session.active_engine_id = engine_id
            self.session.active_engine_loaded = status.loaded
            self.session.autoload_active_engine = status.loaded
            self._save_session()
            return self.active_status()

    async def load_active_engine(self) -> dict[str, Any]:
        async with self.locked():
            engine = self.active_engine
            status = await self._run_blocking(engine.load)
            self.session.active_engine_loaded = status.loaded
            self.session.autoload_active_engine = status.loaded
            self._save_session()
            return self.active_status()

    async def unload_active_engine(self) -> dict[str, Any]:
        async with self.locked():
            engine = self.active_engine
            await self._run_blocking(engine.unload)
            self.session.active_engine_loaded = False
            self.session.autoload_active_engine = False
            self._save_session()
            return self.active_status()

    async def set_active_engine_options(self, options: dict[str, Any]) -> dict[str, Any]:
        async with self.locked():
            engine = self.active_engine
            sanitized = self._sanitize_engine_options(engine, options)
            self.session.engine_options[self.session.active_engine_id] = sanitized
            self._save_session()
            return self.active_status()

    def autoload_active_engine_sync(self) -> dict[str, Any] | None:
        with self._state_lock:
            if not self.should_autoload():
                return None
            if self.active_engine.is_loaded():
                self.session.active_engine_loaded = True
                self.session.autoload_active_engine = True
                self._save_session()
                return self.active_status()
            status = self.active_engine.load()
            self.session.active_engine_loaded = status.loaded
            self.session.autoload_active_engine = status.loaded
            self._save_session()
            return self.active_status()

    async def autoload_active_engine(self) -> dict[str, Any] | None:
        async with self.locked():
            if not self.should_autoload():
                return None
            if self.active_engine.is_loaded():
                self.session.active_engine_loaded = True
                self.session.autoload_active_engine = True
                self._save_session()
                return self.active_status()
            engine = self.active_engine
            status = await self._run_blocking(engine.load)
            self.session.active_engine_loaded = status.loaded
            self.session.autoload_active_engine = status.loaded
            self._save_session()
            return self.active_status()

    def active_voices(self) -> list[EngineVoice]:
        return [self._normalize_voice(voice) for voice in self.active_engine.list_voices()]

    async def synthesize(
        self,
        *,
        text: str,
        voice: str | None,
        language: str | None,
        options: dict[str, Any] | None = None,
    ) -> EngineSynthesisResult:
        async with self.locked():
            started = time.perf_counter()
            engine = self.active_engine
            merged_options = self._resolved_engine_options(engine)
            if options:
                merged_options.update(self._sanitize_engine_options(engine, options))
            result = await self._run_blocking(
                lambda: engine.synthesize(
                    text,
                    voice=voice,
                    language=language,
                    options=merged_options or None,
                )
            )
            result.metrics.end_to_end_time_ms = round((time.perf_counter() - started) * 1000.0, 2)
            self.session.set_selection_for(
                self.session.active_engine_id,
                voice=result.voice,
                language=result.language,
            )
            self._save_session()
            return result
