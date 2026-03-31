"""XTTS-v2 engine adapter."""

from __future__ import annotations

import os
import time
from typing import Any

from app.audio.audio_utils import float32_to_pcm16, pcm16_to_wav_bytes

from .base import EngineNotLoadedError, EngineSynthesisResult, EngineVoice, TtsEngine
from .helpers import cleanup_torch, env_path, preferred_device, status_from_values, synthesis_metrics, with_cpu_fallback
from .speaker_store import SpeakerStore


SUPPORTED_LANGUAGES = [
    "ar", "cs", "de", "en", "es", "fr", "hi", "hu", "it", "ja", "ko", "nl", "pl", "pt", "ru", "tr", "zh-cn",
]


class XttsEngine(TtsEngine):
    def __init__(self, speaker_dir: str | None = None, model_dir: str | None = None) -> None:
        self.model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        self.speaker_store = SpeakerStore(speaker_dir or env_path("SPEAKER_DIR", "/data/speakers"))
        self.model_dir = model_dir or os.getenv("XTTS_MODEL_DIR")
        self._tts = None
        self._state = "not_loaded"
        self._device: str | None = None
        self._load_time_ms: float | None = None
        self._last_loaded_at: float | None = None
        self._last_error: str | None = None
        self._builtin_voices: list[str] = []

    def engine_id(self) -> str:
        return "xtts_v2"

    def display_name(self) -> str:
        return "XTTS-v2"

    def supported_languages(self) -> list[str]:
        return list(SUPPORTED_LANGUAGES)

    def list_voices(self) -> list[EngineVoice]:
        names = [*self._builtin_voices, *self.speaker_store.profile_names()]
        if not names:
            names = ["default"]
        voices = []
        for name in names:
            voices.append(
                EngineVoice(
                    id=name,
                    label=name,
                    languages=self.supported_languages(),
                    default_language="pl",
                    description="XTTS voice",
                )
            )
        return voices

    @staticmethod
    def _patch_transformers() -> None:
        import torch
        import transformers
        from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
        from transformers.generation.beam_search import BeamSearchScorer, ConstrainedBeamSearchScorer
        from transformers.generation.utils import GenerationMixin

        transformers.BeamSearchScorer = getattr(transformers, "BeamSearchScorer", BeamSearchScorer)
        transformers.ConstrainedBeamSearchScorer = getattr(
            transformers,
            "ConstrainedBeamSearchScorer",
            ConstrainedBeamSearchScorer,
        )
        transformers.DisjunctiveConstraint = getattr(transformers, "DisjunctiveConstraint", DisjunctiveConstraint)
        transformers.PhrasalConstraint = getattr(transformers, "PhrasalConstraint", PhrasalConstraint)

        from TTS.tts.layers.xtts.gpt import GPT2InferenceModel

        if not hasattr(GPT2InferenceModel, "generate"):
            GPT2InferenceModel.generate = GenerationMixin.generate

        current_load = torch.load
        if getattr(current_load, "_xtts_compat", False):
            return

        def compat_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return current_load(*args, **kwargs)

        compat_load._xtts_compat = True  # type: ignore[attr-defined]
        torch.load = compat_load

    def _load_model(self, device: str):
        os.environ["COQUI_TOS_AGREED"] = "1"
        self._patch_transformers()
        from TTS.api import TTS

        if self.model_dir:
            os.environ.setdefault("TTS_HOME", self.model_dir)
        return TTS(self.model_name, gpu=device.startswith("cuda"))

    def load(self, device_preference: str | None = None):
        if self._tts is not None:
            return self.status()
        self._state = "loading"
        self._last_error = None
        try:
            tts, device, load_time_ms = with_cpu_fallback(lambda selected: self._load_model(device_preference or selected))
            self._tts = tts
            self._device = device
            self._load_time_ms = load_time_ms
            self._last_loaded_at = time.time()
            speaker_manager = getattr(self._tts.synthesizer.tts_model, "speaker_manager", None)
            speakers = getattr(speaker_manager, "speakers", None) or {}
            self._builtin_voices = sorted(str(name) for name in speakers.keys())
            self._state = "ready"
        except Exception as err:
            self._state = "error"
            self._last_error = str(err)
            raise
        return self.status()

    def unload(self) -> None:
        self._state = "unloading"
        self._tts = None
        cleanup_torch()
        self._state = "not_loaded"

    def is_loaded(self) -> bool:
        return self._tts is not None

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
                "model_name": self.model_name,
                "runtime_options": {
                    "split_sentences": {
                        "type": "boolean",
                        "label": "Split Sentences",
                        "default": True,
                        "description": "Split long text into sentences before synthesis for safer generation.",
                    }
                },
            },
        )

    def synthesize(self, text: str, *, voice: str | None, language: str | None, options: dict[str, Any] | None = None):
        if self._tts is None:
            raise EngineNotLoadedError("XTTS engine is not loaded")
        resolved_language = (language or "pl").lower()
        resolved_voice = voice or "default"
        builtin_voice = resolved_voice if resolved_voice in self._builtin_voices else None
        profile = None if builtin_voice else self.speaker_store.get_profile(resolved_voice)
        speaker_wav = SpeakerStore.wav_paths(profile)
        split_sentences = True if options is None else bool(options.get("split_sentences", True))
        started = time.perf_counter()
        audio = self._tts.synthesizer.tts(
            text=text.strip(),
            speaker_name=builtin_voice,
            language_name=resolved_language,
            speaker_wav=speaker_wav,
            split_sentences=split_sentences,
        )
        if hasattr(audio, "tolist"):
            audio = audio.tolist()
        synthesis_time_ms = round((time.perf_counter() - started) * 1000.0, 2)
        pcm_audio = float32_to_pcm16(audio)
        wav_audio = pcm16_to_wav_bytes(pcm_audio, sample_rate=24000)
        return EngineSynthesisResult(
            engine_id=self.engine_id(),
            voice=resolved_voice,
            language=resolved_language,
            device=self._device or "cpu",
            sample_rate=24000,
            channels=1,
            sample_width=2,
            wav_audio=wav_audio,
            pcm_audio=pcm_audio,
            metrics=synthesis_metrics(
                load_time_ms=self._load_time_ms,
                synthesis_time_ms=synthesis_time_ms,
                end_to_end_time_ms=synthesis_time_ms,
                pcm_bytes=len(pcm_audio),
                sample_rate=24000,
                cold_start=False,
            ),
            backend="coqui-tts",
        )

    def health_payload(self) -> dict[str, Any]:
        return self.status().asdict()
