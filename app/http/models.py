"""HTTP request and response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SelectEngineRequest(BaseModel):
    engine_id: str


class ActivateEngineRequest(BaseModel):
    engine_id: str


class EngineOptionsRequest(BaseModel):
    options: dict


class SynthesizeRequest(BaseModel):
    text: str = Field(min_length=1)
    voice: str | None = None
    language: str | None = None
    options: dict | None = None


class OpenAiSpeechRequest(BaseModel):
    model: str = Field(min_length=1)
    input: str = Field(min_length=1)
    voice: str | None = None
    response_format: str = "wav"
    speed: float | None = None
