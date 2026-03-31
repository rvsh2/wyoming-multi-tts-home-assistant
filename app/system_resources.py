"""System resource probes for VRAM and RAM usage."""

from __future__ import annotations

import subprocess
from typing import Any


ENGINE_MEMORY_HINTS: dict[str, dict[str, str]] = {
    "chatterbox": {
        "gpu": "Observed ~3-4 GiB VRAM",
        "cpu": "CPU fallback, expect high RAM usage",
    },
    "whisperspeech": {
        "gpu": "Observed ~3-4 GiB VRAM",
        "cpu": "CPU fallback, expect high RAM usage",
    },
    "xtts_v2": {
        "gpu": "Observed ~2-3 GiB VRAM",
        "cpu": "CPU fallback, expect high RAM usage",
    },
    "qwen_tts_polish": {
        "gpu": "Observed ~2-3 GiB VRAM",
        "cpu": "CPU fallback, expect high RAM usage",
    },
    "mms_tts_pol": {
        "gpu": "Observed <1 GiB VRAM",
        "cpu": "Low RAM footprint on CPU",
    },
    "fish_s2_pro": {
        "gpu": "Observed ~18-20 GiB VRAM",
        "cpu": "CPU fallback, expect very high RAM usage",
    },
}


def _bytes_to_gib(value: int) -> float:
    return round(value / (1024 ** 3), 2)


def _read_meminfo() -> dict[str, int]:
    payload: dict[str, int] = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as handle:
        for line in handle:
            key, _, raw_value = line.partition(":")
            parts = raw_value.strip().split()
            if not parts:
                continue
            payload[key] = int(parts[0]) * 1024
    return payload


def ram_usage_payload() -> dict[str, Any]:
    meminfo = _read_meminfo()
    total = meminfo.get("MemTotal", 0)
    available = meminfo.get("MemAvailable", 0)
    used = max(total - available, 0)
    return {
        "kind": "ram",
        "label": "RAM",
        "unit": "GiB",
        "total_bytes": total,
        "used_bytes": used,
        "free_bytes": available,
        "total_gib": _bytes_to_gib(total),
        "used_gib": _bytes_to_gib(used),
        "free_gib": _bytes_to_gib(available),
        "display": f"{_bytes_to_gib(used)} / {_bytes_to_gib(total)} GiB",
    }


def _normalize_gpu_index(device: str | None) -> int | None:
    if not device or not device.startswith("cuda"):
        return None
    if ":" not in device:
        return 0
    try:
        return int(device.split(":", 1)[1])
    except ValueError:
        return 0


def vram_usage_payload(device: str | None) -> dict[str, Any] | None:
    gpu_index = _normalize_gpu_index(device)
    if gpu_index is None:
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    line = result.stdout.strip().splitlines()
    if not line:
        return None
    try:
        total_mib, used_mib, free_mib = [int(part.strip()) for part in line[0].split(",")]
    except ValueError:
        return None
    total_bytes = total_mib * 1024 * 1024
    used_bytes = used_mib * 1024 * 1024
    free_bytes = free_mib * 1024 * 1024
    return {
        "kind": "vram",
        "label": f"VRAM cuda:{gpu_index}",
        "unit": "GiB",
        "gpu_index": gpu_index,
        "total_bytes": total_bytes,
        "used_bytes": used_bytes,
        "free_bytes": free_bytes,
        "total_gib": _bytes_to_gib(total_bytes),
        "used_gib": _bytes_to_gib(used_bytes),
        "free_gib": _bytes_to_gib(free_bytes),
        "display": f"{_bytes_to_gib(used_bytes)} / {_bytes_to_gib(total_bytes)} GiB",
    }


def resource_usage_payload(device: str | None) -> dict[str, Any]:
    vram = vram_usage_payload(device)
    if vram is not None:
        return vram
    return ram_usage_payload()


def gpu_available() -> bool:
    return vram_usage_payload("cuda:0") is not None


def engine_memory_hint(engine_id: str, device: str | None, *, prefer_gpu: bool = False) -> str | None:
    hints = ENGINE_MEMORY_HINTS.get(engine_id)
    if not hints:
        return None
    if device and device.startswith("cuda"):
        return hints.get("gpu")
    if device is None and (prefer_gpu or gpu_available()):
        return hints.get("gpu")
    return hints.get("cpu") or hints.get("gpu")
