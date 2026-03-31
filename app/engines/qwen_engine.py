"""Qwen3 TTS adapter."""

from __future__ import annotations

import importlib.util
import logging
import os
import time
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    class _TorchShim:
        bfloat16 = "bfloat16"
        float32 = "float32"

    torch = _TorchShim()  # type: ignore[assignment]

from app.audio.audio_utils import float32_to_pcm16, pcm16_to_wav_bytes

from .base import EngineError, EngineNotLoadedError, EngineSynthesisResult, EngineVoice, TtsEngine
from .helpers import cleanup_torch, preferred_device, snapshot_download_local_first, status_from_values, synthesis_metrics, with_cpu_fallback


QWEN_LANGUAGE_CODE_TO_NAME = {
    "zh": "chinese",
    "en": "english",
    "ja": "japanese",
    "ko": "korean",
    "de": "german",
    "fr": "french",
    "ru": "russian",
    "pt": "portuguese",
    "es": "spanish",
    "it": "italian",
}

QWEN_NAME_TO_LANGUAGE_CODE = {
    name: code for code, name in QWEN_LANGUAGE_CODE_TO_NAME.items()
}

QWEN_DEFAULT_SPEAKERS = [
    ("Vivian", "zh", "Bright young female voice."),
    ("Serena", "zh", "Warm, gentle young female voice."),
    ("Uncle_Fu", "zh", "Seasoned male voice with a mellow timbre."),
    ("Dylan", "zh", "Youthful Beijing male voice."),
    ("Eric", "zh", "Lively Chengdu male voice."),
    ("Ryan", "en", "Dynamic male voice with rhythm."),
    ("Aiden", "en", "Sunny American male voice."),
    ("Ono_Anna", "ja", "Playful Japanese female voice."),
    ("Sohee", "ko", "Warm Korean female voice."),
]

LOGGER = logging.getLogger(__name__)


class QwenTtsEngine(TtsEngine):
    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or os.getenv("QWEN_TTS_MODEL_ID", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
        self._model: Any | None = None
        self._state = "not_loaded"
        self._device: str | None = None
        self._load_time_ms: float | None = None
        self._last_loaded_at: float | None = None
        self._last_error: str | None = None
        self._speakers = {speaker: {"native_language": language, "description": description} for speaker, language, description in QWEN_DEFAULT_SPEAKERS}
        self._sample_rate = 24000
        self._backend_supported_languages: list[str] = list(QWEN_NAME_TO_LANGUAGE_CODE)

    def engine_id(self) -> str:
        return "qwen_tts_polish"

    def display_name(self) -> str:
        return "Qwen3-TTS"

    def supported_languages(self) -> list[str]:
        return list(QWEN_LANGUAGE_CODE_TO_NAME)

    def list_voices(self) -> list[EngineVoice]:
        return [
            EngineVoice(
                id=speaker,
                label=speaker,
                languages=self.supported_languages(),
                default_language=metadata["native_language"],
                description=metadata["description"],
            )
            for speaker, metadata in self._speakers.items()
        ]

    def _load_model(self, device: str):
        from qwen_tts import Qwen3TTSModel

        model_source = snapshot_download_local_first(
            self.model_id,
            token=os.getenv("HF_TOKEN"),
        )
        kwargs: dict[str, Any] = {
            "device_map": device if device.startswith("cuda") else "cpu",
            "dtype": torch.bfloat16 if device.startswith("cuda") else torch.float32,
        }
        if device.startswith("cuda") and importlib.util.find_spec("flash_attn") is not None:
            kwargs["attn_implementation"] = "flash_attention_2"
        model = Qwen3TTSModel.from_pretrained(model_source, **kwargs)
        get_supported = getattr(model, "get_supported_speakers", None)
        if callable(get_supported):
            speakers = get_supported()
            if speakers:
                current_metadata = self._speakers
                self._speakers = {
                    str(item): current_metadata.get(
                        str(item),
                        {"native_language": "en", "description": "Official Qwen3-TTS speaker."},
                    )
                    for item in speakers
                }
        get_supported_languages = getattr(model, "get_supported_languages", None)
        if callable(get_supported_languages):
            supported_languages = get_supported_languages()
            if supported_languages:
                self._backend_supported_languages = [str(item).lower() for item in supported_languages]
        return model

    def load(self, device_preference: str | None = None):
        if self._model is not None:
            return self.status()
        self._state = "loading"
        self._last_error = None
        try:
            model, device, load_time_ms = with_cpu_fallback(lambda selected: self._load_model(device_preference or selected))
            self._model = model
            self._device = device
            self._load_time_ms = load_time_ms
            self._last_loaded_at = time.time()
            self._state = "ready"
        except Exception as err:
            self._state = "error"
            self._last_error = str(err)
            raise
        return self.status()

    def unload(self) -> None:
        self._state = "unloading"
        self._model = None
        cleanup_torch()
        self._state = "not_loaded"

    def is_loaded(self) -> bool:
        return self._model is not None

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
            supports_streaming=False,
            voices=self.list_voices(),
            extra={
                "model_id": self.model_id,
                "backend_supported_languages": self._backend_supported_languages,
                "wyoming_program_name": "qwen_tts",
                "runtime_options": {
                    "instruct": {
                        "type": "string",
                        "label": "Instruction",
                        "default": "",
                        "placeholder": "Optional speaking style or delivery hint",
                        "description": "Optional style prompt passed to Qwen3-TTS.",
                    }
                },
                "language_note": (
                    "Official Qwen3-TTS supports Chinese, English, Japanese, Korean, German, "
                    "French, Russian, Portuguese, Spanish, and Italian. Polish is not supported."
                ),
            },
        )

    def _resolve_generation_controls(
        self,
        language: str | None,
        options: dict[str, Any] | None,
    ) -> tuple[str, str, str]:
        requested_language = (language or "en").strip().lower()
        resolved_language = requested_language
        if resolved_language not in QWEN_LANGUAGE_CODE_TO_NAME:
            raise EngineError(
                f"Qwen3-TTS does not support language '{requested_language}'. "
                f"Supported languages: {', '.join(sorted(QWEN_LANGUAGE_CODE_TO_NAME))}"
            )
        backend_language = QWEN_LANGUAGE_CODE_TO_NAME[resolved_language]
        instruct = str((options or {}).get("instruct", "") or "").strip()
        return resolved_language, backend_language, instruct

    def _resolve_voice_and_language(
        self,
        voice: str | None,
        language: str | None,
        options: dict[str, Any] | None,
    ) -> tuple[str, str, str, str]:
        available_speakers = list(self._speakers)
        resolved_voice = voice or available_speakers[0]
        if resolved_voice not in self._speakers:
            resolved_voice = available_speakers[0]
        requested_language = language or self._speakers[resolved_voice]["native_language"]
        resolved_language, backend_language, instruct = self._resolve_generation_controls(requested_language, options)
        return resolved_voice, resolved_language, backend_language, instruct

    def _generate_audio(self, *, text: str, backend_language: str, voice: str, instruct: str) -> tuple[Any, int]:
        inference_mode = getattr(torch, "inference_mode", None)
        if callable(inference_mode):
            with inference_mode():
                return self._model.generate_custom_voice(
                    text=text.strip(),
                    language=backend_language,
                    speaker=voice,
                    instruct=instruct,
                )
        return self._model.generate_custom_voice(
            text=text.strip(),
            language=backend_language,
            speaker=voice,
            instruct=instruct,
        )

    @staticmethod
    def _audio_to_pcm(audio: Any, sample_rate: int) -> tuple[bytes, bytes]:
        if hasattr(audio, "tolist"):
            audio = audio.tolist()
        pcm_audio = float32_to_pcm16(audio)
        wav_audio = pcm16_to_wav_bytes(pcm_audio, sample_rate=sample_rate)
        return pcm_audio, wav_audio

    def synthesize(self, text: str, *, voice: str | None, language: str | None, options: dict[str, Any] | None = None):
        if self._model is None:
            raise EngineNotLoadedError("Qwen TTS engine is not loaded")
        resolved_voice, resolved_language, backend_language, instruct = self._resolve_voice_and_language(
            voice,
            language,
            options,
        )
        started = time.perf_counter()
        LOGGER.info(
            "Qwen synth start: speaker=%s backend_language=%s resolved_language=%s device=%s",
            resolved_voice,
            backend_language,
            resolved_language,
            self._device or "cpu",
        )
        wavs, sample_rate = self._generate_audio(
            text=text,
            backend_language=backend_language,
            voice=resolved_voice,
            instruct=instruct,
        )
        synthesis_time_ms = round((time.perf_counter() - started) * 1000.0, 2)
        LOGGER.info(
            "Qwen synth done: speaker=%s backend_language=%s resolved_language=%s duration_ms=%.2f",
            resolved_voice,
            backend_language,
            resolved_language,
            synthesis_time_ms,
        )
        pcm_audio, wav_audio = self._audio_to_pcm(wavs[0], int(sample_rate))
        return EngineSynthesisResult(
            engine_id=self.engine_id(),
            voice=resolved_voice,
            language=resolved_language,
            device=self._device or "cpu",
            sample_rate=int(sample_rate),
            channels=1,
            sample_width=2,
            wav_audio=wav_audio,
            pcm_audio=pcm_audio,
            metrics=synthesis_metrics(
                load_time_ms=self._load_time_ms,
                synthesis_time_ms=synthesis_time_ms,
                end_to_end_time_ms=synthesis_time_ms,
                pcm_bytes=len(pcm_audio),
                sample_rate=int(sample_rate),
                cold_start=False,
            ),
            backend="qwen-tts",
        )

    def health_payload(self) -> dict[str, Any]:
        return self.status().asdict()
