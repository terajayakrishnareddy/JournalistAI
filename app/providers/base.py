"""
Abstract interfaces for pluggable transcription and analysis providers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class WordResult:
    word: str
    start: float
    end: float
    probability: Optional[float] = None

@dataclass
class SegmentResult:
    segment_index: int
    start: float
    end: float
    text: str
    speaker: Optional[str] = None
    confidence: Optional[float] = None
    words: list[WordResult] = field(default_factory=list)

@dataclass
class TranscriptionResult:
    segments: list[SegmentResult]
    language_detected: Optional[str] = None
    provider_name: str = "unknown"
    model_name: str = "unknown"

class TranscriptionProvider(ABC):
    """All transcription backends must implement this interface."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str = "auto",
        vocabulary_hint: Optional[str] = None,
        fast_mode: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe the audio file at `audio_path`.

        Args:
            audio_path:      Path to a 16 kHz mono WAV file.
            language:        BCP-47 language code or "auto" for auto-detection.
            vocabulary_hint: Optional initial prompt / vocabulary list.

        Returns:
            TranscriptionResult with one SegmentResult per segment.
        """
        ...

@dataclass
class QuoteCandidate:
    quote_text: str
    segment_index: Optional[int]
    speaker: Optional[str]
    reason: str

@dataclass
class ClaimCandidate:
    claim_text: str
    segment_index: Optional[int]
    category: str
    reason: str
    suggested_verification_steps: str

@dataclass
class AnalysisResult:
    summary: str
    key_topics: list[str]
    quotes: list[QuoteCandidate]
    claims: list[ClaimCandidate]
    raw_response: Optional[str] = None

class AnalysisProvider(ABC):
    """All analysis backends must implement this interface."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def analyze(
        self,
        segments: list[SegmentResult],
        *,
        language: str = "en",
        title: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> AnalysisResult:
        """
        Analyze a list of transcript segments.

        The provider MUST NOT label any claim true/false.
        The provider MUST return verbatim text for quotes (validated externally).
        """
        ...
