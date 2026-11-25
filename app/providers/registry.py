"""
Provider registry / factory.

Selects the correct TranscriptionProvider and AnalysisProvider based on
environment variables. New providers are registered here only.
"""
from __future__ import annotations

import functools

from app.core.config import get_settings
from app.providers.base import AnalysisProvider, TranscriptionProvider

@functools.lru_cache(maxsize=1)
def get_transcription_provider() -> TranscriptionProvider:
    settings = get_settings()
    name = settings.TRANSCRIPTION_PROVIDER.lower()

    if name == "whisper":
        from app.providers.whisper_provider import WhisperProvider
        return WhisperProvider()

    raise ValueError(
        f"Unknown TRANSCRIPTION_PROVIDER='{name}'. "
        "Valid options: whisper"
    )

@functools.lru_cache(maxsize=1)
def get_analysis_provider() -> AnalysisProvider:
    settings = get_settings()
    name = settings.ANALYSIS_PROVIDER.lower()

    if name == "gemini":
        has_key = bool(settings.GEMINI_API_KEY and settings.GEMINI_API_KEY.strip() != "your_gemini_api_key_here")
        if has_key:
            from app.providers.gemini_provider import GeminiProvider
            return GeminiProvider()
        else:

            from app.providers.local_analysis_provider import LocalAnalysisProvider
            return LocalAnalysisProvider()

    if name == "local":
        from app.providers.local_analysis_provider import LocalAnalysisProvider
        return LocalAnalysisProvider()

    raise ValueError(
        f"Unknown ANALYSIS_PROVIDER='{name}'. "
        "Valid options: gemini, local"
    )
