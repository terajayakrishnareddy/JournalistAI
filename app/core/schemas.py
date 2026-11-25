"""
Pydantic schemas for request / response validation.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

class InterviewCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    interview_date: Optional[str] = None
    language: str = Field("auto", pattern=r"^(auto|[a-zA-Z]{2,3}(-[a-zA-Z0-9]+)?)$")
    category: Optional[str] = Field(None, max_length=200)
    notes: Optional[str] = None
    vocabulary_hint: Optional[str] = Field(
        None,
        description="Names, jargon, or keywords to improve Whisper accuracy (used as initial_prompt)",
    )

class InterviewUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    interview_date: Optional[str] = None
    language: Optional[str] = Field(None, pattern=r"^(auto|[a-zA-Z]{2,3}(-[a-zA-Z0-9]+)?)$")
    category: Optional[str] = Field(None, max_length=200)
    notes: Optional[str] = None
    vocabulary_hint: Optional[str] = None

class InterviewResponse(BaseModel):
    id: int
    uuid: str
    title: str
    interview_date: Optional[str]
    language: str
    category: Optional[str]
    notes: Optional[str]
    vocabulary_hint: Optional[str]
    status: str
    error_message: Optional[str]
    original_filename: Optional[str]
    file_size_bytes: Optional[int]
    duration_seconds: Optional[float]
    media_type: Optional[str]
    transcription_provider: Optional[str]
    analysis_provider: Optional[str]
    whisper_model: Optional[str]
    diarization_run: bool
    created_at: datetime
    updated_at: datetime
    processing_started_at: Optional[datetime]
    processing_completed_at: Optional[datetime]

    model_config = {"from_attributes": True}

class InterviewStatusResponse(BaseModel):
    id: int
    uuid: str
    status: str
    error_message: Optional[str]
    duration_seconds: Optional[float]
    processing_started_at: Optional[datetime]
    processing_completed_at: Optional[datetime]

    model_config = {"from_attributes": True}

class WordTimestamp(BaseModel):
    word: str
    start: float
    end: float
    probability: Optional[float] = None

class SegmentResponse(BaseModel):
    id: int
    segment_index: int
    start_time: float
    end_time: float
    original_text: str
    corrected_text: Optional[str]
    speaker: Optional[str]
    confidence: Optional[float]
    words: Optional[list[WordTimestamp]] = None

    model_config = {"from_attributes": True}

class TranscriptResponse(BaseModel):
    interview_id: int
    total_segments: int
    diarization_run: bool
    segments: list[SegmentResponse]

class SegmentCorrectionRequest(BaseModel):
    corrected_text: Optional[str] = None
    speaker: Optional[str] = None

class SpeakerRenameRequest(BaseModel):
    old_speaker: str
    new_speaker: str

class QuoteResponse(BaseModel):
    id: int
    segment_id: Optional[int]
    segment_index: Optional[int]
    speaker: Optional[str]
    quote_text: str
    reason: Optional[str]

    model_config = {"from_attributes": True}

class ClaimUpdateRequest(BaseModel):
    status: str = Field(..., pattern=r"^(needs_verification|supported|refuted|unverified)$")
    verification_notes: Optional[str] = None
    evidence_url: Optional[str] = None
    evidence_title: Optional[str] = None
    evidence_snippet: Optional[str] = None

class ClaimResponse(BaseModel):
    id: int
    interview_id: int
    segment_id: Optional[int]
    segment_index: Optional[int]
    claim_text: str
    category: Optional[str]
    reason: Optional[str]
    suggested_verification_steps: Optional[str]
    status: str
    verification_notes: Optional[str] = None
    evidence_url: Optional[str] = None
    evidence_title: Optional[str] = None
    evidence_snippet: Optional[str] = None
    verified_at: Optional[datetime] = None

    model_config = {"from_attributes": True}

class EvidenceSearchResponse(BaseModel):
    claim_id: int
    claim_text: str
    query_used: str
    results_found: int
    evidence_title: Optional[str]
    evidence_url: Optional[str]
    evidence_snippet: Optional[str]
    suggested_verdict: str

class AnalysisResponse(BaseModel):
    summary: Optional[str]
    key_topics: list[str]
    quotes: list[QuoteResponse]
    claims: list[ClaimResponse]

class ProcessRequest(BaseModel):
    run_diarization: bool = False
    fast_mode: bool = False

class SettingsUpdateRequest(BaseModel):
    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None
