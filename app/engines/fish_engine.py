"""Fish Audio S2 Pro adapter backed by the official fish-speech API server."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from typing import Any

from huggingface_hub import snapshot_download

from app.audio.audio_utils import pcm16_bytes_from_audio_file

from .base import EngineError, EngineNotLoadedError, EngineSynthesisResult, EngineVoice, TtsEngine
from .helpers import preferred_device, snapshot_download_local_first, status_from_values, synthesis_metrics, with_cpu_fallback


SUPPORTED_LANGUAGES = [
    "pl", "en", "zh", "ja", "ko", "es", "pt", "ar", "ru", "fr", "de", "sv", "it", "tr", "no", "nl", "fi", "cs",
]

REQUIRED_MODEL_FILES = [
    "codec.pth",
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
]

MODEL_ALLOW_PATTERNS = [
    "codec.pth",
    "config.json",
    "model.safetensors.index.json",
    "model-*.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "README.md",
]

FISH_DEFAULT_SEED = int(os.getenv("FISH_DEFAULT_SEED", "777"))
FISH_DEFAULT_LATENCY = os.getenv("FISH_DEFAULT_LATENCY", "balanced")
FISH_DEFAULT_NORMALIZE = os.getenv("FISH_DEFAULT_NORMALIZE", "true").lower() != "false"
FISH_DEFAULT_CHUNK_LENGTH = int(os.getenv("FISH_DEFAULT_CHUNK_LENGTH", "200"))


class FishS2ProEngine(TtsEngine):
    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or os.getenv("FISH_S2_PRO_MODEL_ID", "fishaudio/s2-pro")
        self._state = "not_loaded"
        self._device: str | None = None
        self._load_time_ms: float | None = None
        self._last_loaded_at: float | None = None
        self._last_error: str | None = None
        self._sample_rate = 44100
        self._server_process: subprocess.Popen[bytes] | None = None
        self._server_url: str | None = None
        self._server_port: int | None = None
        self._server_log_handle: Any | None = None
        self._server_log_path: Path | None = None
        self._model_source: str | None = None
        self._reference_ids: list[str] = []
        self._compile_enabled = False
        self._ready_timeout_seconds = float(os.getenv("FISH_READY_TIMEOUT_SECONDS", "240"))

    def engine_id(self) -> str:
        return "fish_s2_pro"

    def display_name(self) -> str:
        return "Fish Audio S2 Pro"

    def supported_languages(self) -> list[str]:
        return list(SUPPORTED_LANGUAGES)

    def list_voices(self) -> list[EngineVoice]:
        voices = [
            EngineVoice(
                id="default",
                label="Default",
                languages=self.supported_languages(),
                default_language="pl",
                description="Fish Audio default timbre selection",
            )
        ]
        for reference_id in self._reference_ids:
            voices.append(
                EngineVoice(
                    id=reference_id,
                    label=reference_id,
                    languages=self.supported_languages(),
                    default_language="pl",
                    description="Fish Audio persisted reference voice",
                )
            )
        return voices

    def load(self, device_preference: str | None = None):
        if self.is_loaded():
            return self.status()
        self._state = "loading"
        self._last_error = None
        try:
            _, device, load_time_ms = with_cpu_fallback(
                lambda selected: self._load_server(device_preference or selected)
            )
            self._device = device
            self._load_time_ms = load_time_ms
            self._last_loaded_at = time.time()
            self._state = "ready"
        except Exception as err:
            self._state = "error"
            self._last_error = str(err)
            self._stop_server()
            raise
        return self.status()

    def unload(self) -> None:
        self._state = "unloading"
        self._stop_server()
        self._state = "not_loaded"

    def is_loaded(self) -> bool:
        return self._server_process is not None and self._server_process.poll() is None and self._server_url is not None

    def status(self):
        return status_from_values(
            engine_id=self.engine_id(),
            display_name=self.display_name(),
            state=self._state,
            loaded=self.is_loaded(),
            device=self._device or preferred_device(),
            load_time_ms=self._load_time_ms,
            last_loaded_at=self._last_loaded_at,
            last_error=self._last_error,
            supports_streaming=True,
            voices=self.list_voices(),
            extra={
                "model_id": self.model_id,
                "model_source": self._model_source,
                "sample_rate": self._sample_rate,
                "server_url": self._server_url,
                "compile_enabled": self._compile_enabled,
                "supports_language_control": False,
                "runtime_options": self._runtime_options(),
            },
        )

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None,
        language: str | None,
        options: dict[str, Any] | None = None,
    ):
        if not self.is_loaded() or self._server_url is None:
            raise EngineNotLoadedError("Fish S2 Pro engine is not loaded")

        resolved_voice = voice or "default"
        resolved_options = self._resolved_options(options)
        payload: dict[str, Any] = {
            "text": text.strip(),
            "format": "wav",
            "latency": str(resolved_options["latency"]),
            "streaming": False,
            "normalize": bool(resolved_options["normalize"]),
            "chunk_length": int(resolved_options["chunk_length"]),
        }
        if resolved_voice != "default":
            payload["reference_id"] = resolved_voice
        seed = resolved_options.get("seed")
        if seed is not None:
            payload["seed"] = int(seed)

        started = time.perf_counter()
        wav_audio = self._request_bytes("/v1/tts", payload, timeout=240)
        end_to_end_time_ms = round((time.perf_counter() - started) * 1000.0, 2)
        pcm_audio = pcm16_bytes_from_audio_file(wav_audio)

        return EngineSynthesisResult(
            engine_id=self.engine_id(),
            voice=resolved_voice,
            language=(language or "pl").lower(),
            device=self._device or "cpu",
            sample_rate=self._sample_rate,
            channels=1,
            sample_width=2,
            wav_audio=wav_audio,
            pcm_audio=pcm_audio,
            metrics=synthesis_metrics(
                load_time_ms=self._load_time_ms,
                synthesis_time_ms=end_to_end_time_ms,
                end_to_end_time_ms=end_to_end_time_ms,
                pcm_bytes=len(pcm_audio),
                sample_rate=self._sample_rate,
                cold_start=False,
            ),
            backend="fish-speech-api",
        )

    def health_payload(self) -> dict[str, Any]:
        return self.status().asdict()

    def _runtime_options(self) -> dict[str, dict[str, Any]]:
        return {
            "seed": {
                "type": "integer",
                "label": "Seed",
                "default": FISH_DEFAULT_SEED,
                "min": 0,
                "description": "Keeps Fish voice generation more stable between requests.",
            },
            "latency": {
                "type": "select",
                "label": "Latency",
                "default": FISH_DEFAULT_LATENCY,
                "choices": ["normal", "balanced"],
                "description": "Balanced is safer, normal can favor lower latency.",
            },
            "normalize": {
                "type": "boolean",
                "label": "Normalize",
                "default": FISH_DEFAULT_NORMALIZE,
                "description": "Normalize loudness before returning WAV output.",
            },
            "chunk_length": {
                "type": "integer",
                "label": "Chunk length",
                "default": FISH_DEFAULT_CHUNK_LENGTH,
                "min": 100,
                "max": 400,
                "description": "Longer chunks can sound smoother but may increase latency.",
            },
        }

    def _resolved_options(self, options: dict[str, Any] | None) -> dict[str, Any]:
        resolved = {
            option_name: option_spec.get("default")
            for option_name, option_spec in self._runtime_options().items()
        }
        for option_name, option_value in (options or {}).items():
            if option_value in (None, "") or option_name not in resolved:
                continue
            resolved[option_name] = option_value
        return resolved

    def _load_server(self, device: str) -> bool:
        model_source = self._resolve_model_source()
        codec_path = Path(model_source) / "codec.pth"
        if not codec_path.exists():
            raise RuntimeError(f"Fish codec checkpoint is missing at {codec_path}")

        compile_candidates = [self._compile_requested_for(device)]
        if compile_candidates[0]:
            compile_candidates.append(False)

        last_error: Exception | None = None
        for compile_enabled in dict.fromkeys(compile_candidates):
            try:
                self._start_server(
                    model_source=model_source,
                    codec_path=codec_path,
                    device=device,
                    compile_enabled=compile_enabled,
                )
                self._refresh_reference_ids()
                self._model_source = model_source
                self._compile_enabled = compile_enabled
                return True
            except Exception as err:
                last_error = err
                self._stop_server()
        assert last_error is not None
        raise last_error

    def _resolve_model_source(self) -> str:
        token = os.getenv("HF_TOKEN")
        model_source = snapshot_download_local_first(
            self.model_id,
            allow_patterns=MODEL_ALLOW_PATTERNS,
            token=token,
        )
        if self._has_required_model_files(Path(model_source)):
            return model_source
        fetched_source = snapshot_download(
            repo_id=self.model_id,
            repo_type="model",
            allow_patterns=MODEL_ALLOW_PATTERNS,
            token=token,
        )
        if self._has_required_model_files(Path(fetched_source)):
            return fetched_source
        raise RuntimeError(f"Fish S2 Pro snapshot is incomplete after fetch: {fetched_source}")

    def _start_server(
        self,
        *,
        model_source: str,
        codec_path: Path,
        device: str,
        compile_enabled: bool,
    ) -> None:
        repo_dir = self._repo_dir()
        if not repo_dir.exists():
            raise RuntimeError(f"Fish Speech repository is missing at {repo_dir}")

        self._ensure_reference_symlink(repo_dir)
        port = self._reserve_free_port()
        log_path = Path(tempfile.mkstemp(prefix="fish-api-", suffix=".log")[1])
        log_handle = log_path.open("ab")
        args = [
            sys.executable,
            "tools/api_server.py",
            "--listen",
            f"127.0.0.1:{port}",
            "--device",
            device,
            "--workers",
            "1",
            "--llama-checkpoint-path",
            model_source,
            "--decoder-checkpoint-path",
            str(codec_path),
        ]
        if compile_enabled:
            args.append("--compile")

        process = subprocess.Popen(
            args,
            cwd=str(repo_dir),
            env=os.environ.copy(),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        self._server_process = process
        self._server_url = f"http://127.0.0.1:{port}"
        self._server_port = port
        self._server_log_handle = log_handle
        self._server_log_path = log_path
        self._wait_until_ready()

    def _wait_until_ready(self) -> None:
        assert self._server_url is not None
        deadline = time.monotonic() + self._ready_timeout_seconds
        while time.monotonic() < deadline:
            if self._server_process is not None and self._server_process.poll() is not None:
                raise RuntimeError(self._server_failure_message())
            try:
                payload = self._request_json("/v1/health", None, timeout=5)
            except Exception:
                time.sleep(1)
                continue
            if payload.get("status") == "ok":
                return
            time.sleep(1)
        raise RuntimeError(self._server_failure_message("Fish Speech API server timed out during startup"))

    def _refresh_reference_ids(self) -> None:
        try:
            payload = self._request_json("/v1/references/list", None, timeout=10)
        except Exception:
            self._reference_ids = []
            return
        reference_ids = payload.get("reference_ids")
        if isinstance(reference_ids, list):
            self._reference_ids = [str(item) for item in reference_ids if str(item).strip()]

    def _request_json(
        self,
        path: str,
        payload: dict[str, Any] | None,
        *,
        timeout: int,
    ) -> dict[str, Any]:
        raw = self._request_bytes(path, payload, timeout=timeout, accept_json=True)
        return json.loads(raw.decode("utf-8"))

    def _request_bytes(
        self,
        path: str,
        payload: dict[str, Any] | None,
        *,
        timeout: int,
        accept_json: bool = False,
    ) -> bytes:
        if self._server_url is None:
            raise EngineError("Fish Speech API server is not running")
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers: dict[str, str] = {}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json" if accept_json else "*/*"
        request = urllib.request.Request(
            url=f"{self._server_url}{path}",
            data=body,
            headers=headers,
            method="GET" if payload is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as err:
            detail = err.read().decode("utf-8", errors="replace")
            raise EngineError(f"Fish Speech API returned HTTP {err.code}: {detail}") from err
        except urllib.error.URLError as err:
            raise EngineError(f"Fish Speech API request failed: {err.reason}") from err

    def _compile_requested_for(self, device: str) -> bool:
        if not device.startswith("cuda"):
            return False
        flag = os.getenv("FISH_ENABLE_COMPILE", "1").strip().lower()
        return flag in {"1", "true", "yes", "on"}

    def _has_required_model_files(self, model_source: Path) -> bool:
        if not all((model_source / relative_path).exists() for relative_path in REQUIRED_MODEL_FILES):
            return False
        safetensor_files = list(model_source.glob("model-*.safetensors"))
        return bool(safetensor_files)

    def _repo_dir(self) -> Path:
        return Path(os.getenv("FISH_SPEECH_REPO_DIR", "/opt/fish-speech"))

    def _reference_dir(self) -> Path:
        return Path(os.getenv("FISH_REFERENCE_DIR", "/data/speakers/fish"))

    def _ensure_reference_symlink(self, repo_dir: Path) -> None:
        reference_dir = self._reference_dir()
        reference_dir.mkdir(parents=True, exist_ok=True)
        repo_reference_dir = repo_dir / "references"
        if repo_reference_dir.is_symlink():
            if repo_reference_dir.resolve() == reference_dir.resolve():
                return
            repo_reference_dir.unlink()
        elif repo_reference_dir.exists():
            return
        repo_reference_dir.symlink_to(reference_dir, target_is_directory=True)

    def _reserve_free_port(self) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return int(sock.getsockname()[1])

    def _server_failure_message(self, prefix: str | None = None) -> str:
        base = prefix or "Fish Speech API server exited unexpectedly"
        if self._server_log_path is None or not self._server_log_path.exists():
            return base
        try:
            tail = self._server_log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
        except OSError:
            return base
        if not tail:
            return base
        return f"{base}: {' | '.join(tail)}"

    def _stop_server(self) -> None:
        if self._server_process is not None and self._server_process.poll() is None:
            self._server_process.terminate()
            try:
                self._server_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
                self._server_process.wait(timeout=10)
        self._server_process = None
        self._server_url = None
        self._server_port = None
        self._reference_ids = []
        if self._server_log_handle is not None:
            self._server_log_handle.close()
        self._server_log_handle = None
        self._server_log_path = None
