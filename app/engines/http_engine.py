"""HTTP-backed engine wrapper for isolated runtimes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from .base import EngineError, EngineNotLoadedError, EngineSynthesisResult, EngineStatus, TtsEngine


class HttpEngine(TtsEngine):
    def __init__(
        self,
        *,
        engine_id: str,
        display_name: str,
        module_path: str,
        class_name: str,
        python_env_var: str,
        default_python: str,
        fallback_languages: list[str],
        host: str = "127.0.0.1",
        port_env_var: str | None = None,
        default_port: int | None = None,
    ) -> None:
        self._engine_id = engine_id
        self._display_name = display_name
        self._module_path = module_path
        self._class_name = class_name
        self._python_env_var = python_env_var
        self._default_python = default_python
        self._fallback_languages = list(fallback_languages)
        self._host = host
        self._port_env_var = port_env_var
        self._default_port = default_port
        self._process: subprocess.Popen[str] | None = None
        self._port: int | None = None
        self._http_timeout_seconds = float(
            os.getenv(f"{engine_id.upper()}_HTTP_TIMEOUT_SECONDS", "300")
        )
        self._startup_timeout_seconds = float(
            os.getenv(f"{engine_id.upper()}_HTTP_STARTUP_TIMEOUT_SECONDS", "60")
        )
        self._log_path = Path(f"/tmp/{engine_id}-http-runner.log")
        self._cached_status = EngineStatus(
            engine_id=engine_id,
            display_name=display_name,
            available_voices=[],
            extra={
                "python_env_var": python_env_var,
                "http_timeout_seconds": self._http_timeout_seconds,
            },
        )

    def engine_id(self) -> str:
        return self._engine_id

    def display_name(self) -> str:
        return self._display_name

    def supported_languages(self) -> list[str]:
        return self._fallback_languages

    def list_voices(self) -> list:
        return list(self._cached_status.available_voices)

    def load(self, device_preference: str | None = None) -> EngineStatus:
        self._ensure_process()
        payload = self._request("POST", "/load", {"device_preference": device_preference})
        self._cached_status = EngineStatus.fromdict(payload["status"])
        return self._cached_status

    def unload(self) -> None:
        if self._process is None:
            payload = self._cached_status.asdict()
            payload.update({"state": "not_loaded", "loaded": False, "loading": False})
            self._cached_status = EngineStatus.fromdict(payload)
            return
        try:
            payload = self._request("POST", "/unload", {})
            self._cached_status = EngineStatus.fromdict(payload["status"])
        finally:
            self._stop_process()

    def is_loaded(self) -> bool:
        return self.status().loaded

    def status(self) -> EngineStatus:
        if self._process is None:
            return self._cached_status
        try:
            payload = self._request("GET", "/status")
            self._cached_status = EngineStatus.fromdict(payload["status"])
        except EngineError as err:
            if self._process.poll() is None:
                self._cached_status = self._status_during_runner_transition(err)
                return self._cached_status
            self._stop_process()
            raise
        return self._cached_status

    def synthesize(
        self,
        text: str,
        *,
        voice: str | None,
        language: str | None,
        options: dict[str, Any] | None = None,
    ) -> EngineSynthesisResult:
        self._ensure_process()
        payload = self._request(
            "POST",
            "/synthesize",
            {
                "text": text,
                "voice": voice,
                "language": language,
                "options": options or {},
            },
        )
        return EngineSynthesisResult.from_transport_dict(payload["result"])

    def health_payload(self) -> dict[str, Any]:
        return self.status().asdict()

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        self._port = self._resolve_port()
        python_path = self._resolve_python()
        self._cached_status = self._status_during_runner_transition()
        env = os.environ.copy()
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = self._log_path.open("a+", encoding="utf-8")
        self._process = subprocess.Popen(
            [
                python_path,
                "-m",
                "app.engines.http_runner",
                "--module",
                self._module_path,
                "--class-name",
                self._class_name,
                "--host",
                self._host,
                "--port",
                str(self._port),
            ],
            cwd=str(Path(__file__).resolve().parents[2]),
            env=env,
            stdout=log_handle,
            stderr=log_handle,
            text=True,
            bufsize=1,
        )
        self._wait_until_ready()
        return self._process

    def _resolve_python(self) -> str:
        configured = os.getenv(self._python_env_var)
        if configured:
            return configured
        candidate = Path(self._default_python)
        if candidate.exists():
            return str(candidate)
        return sys.executable

    def _resolve_port(self) -> int:
        configured = os.getenv(self._port_env_var or "")
        if configured:
            return int(configured)
        if self._default_port is not None:
            return self._default_port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((self._host, 0))
            return int(sock.getsockname()[1])

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self._startup_timeout_seconds
        while time.monotonic() < deadline:
            if self._process is None:
                break
            if self._process.poll() is not None:
                raise EngineError(
                    f"{self._display_name} HTTP runner exited during startup: {self._read_log_tail()}"
                )
            try:
                self._request("GET", "/health", timeout=2.0)
                return
            except EngineError:
                time.sleep(0.1)
        self._stop_process()
        raise EngineError(
            f"{self._display_name} HTTP runner did not become ready: {self._read_log_tail()}"
        )

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if self._port is None:
            raise EngineError(f"{self._display_name} HTTP runner has no assigned port")
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib_request.Request(
            f"http://{self._host}:{self._port}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        try:
            with urllib_request.urlopen(req, timeout=timeout or self._http_timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib_error.HTTPError as err:
            body = err.read().decode("utf-8", errors="ignore")
            try:
                error_payload = json.loads(body)
            except json.JSONDecodeError as decode_err:
                raise EngineError(
                    f"{self._display_name} HTTP runner returned invalid error response: {body}"
                ) from decode_err
            self._raise_http_error(error_payload)
        except urllib_error.URLError as err:
            raise EngineError(
                f"{self._display_name} HTTP runner request failed: {err.reason}"
            ) from err
        if response_payload.get("ok"):
            return response_payload
        self._raise_http_error(response_payload)

    def _status_during_runner_transition(self, err: EngineError | None = None) -> EngineStatus:
        payload = self._cached_status.asdict()
        if self._cached_status.loaded:
            payload.update(
                {
                    "state": "error",
                    "loaded": False,
                    "loading": False,
                }
            )
        else:
            payload.update(
                {
                    "state": "loading",
                    "loaded": False,
                    "loading": True,
                }
            )
        if err is not None:
            payload["last_error"] = str(err)
        return EngineStatus.fromdict(payload)

    def _raise_http_error(self, payload: dict[str, Any]) -> None:
        error_type = str(payload.get("error_type", "EngineError"))
        error_text = str(payload.get("error", "Unknown HTTP runner error"))
        if error_type == "EngineNotLoadedError":
            raise EngineNotLoadedError(error_text)
        raise EngineError(error_text)

    def _read_log_tail(self) -> str:
        if not self._log_path.exists():
            return ""
        try:
            lines = self._log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return ""
        return " | ".join(lines[-20:])

    def _stop_process(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)
        self._process = None
        self._port = None
