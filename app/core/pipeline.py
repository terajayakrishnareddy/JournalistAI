"""
Processing pipeline orchestrator.

Each stage is independently testable. Stages:
  1. validate_upload        – ffprobe + size/duration checks
  2. extract_audio          – ffmpeg: video→wav or audio→16kHz mono wav
  3. transcribe             – TranscriptionProvider
  4. diarize (optional)     – pyannote.audio (skipped gracefully)
  5. analyze                – AnalysisProvider (Gemini)
  6. persist                – save everything to DB

Status values used:  uploaded → queued → extracting_audio → transcribing →
                     identifying_speakers → extracting_insights → completed
                     (or failed with error_message at any stage)
"""
from __future__ import annotations

import importlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.core.audio_processor import extract_and_convert
from app.core.config import get_settings
from app.core.database import (
    Analysis,
    Claim,
    Interview,
    Quote,
    TranscriptSegment,
)
from app.core.media_validator import validate_media
from app.providers.base import SegmentResult
from app.providers.registry import get_analysis_provider, get_transcription_provider

logger = logging.getLogger(__name__)
settings = get_settings()

def _set_status(db: Session, interview: Interview, status: str, error: Optional[str] = None) -> None:
    interview.status = status
    if error:
        interview.error_message = error
        logger.error("Interview %s → %s: %s", interview.uuid, status, error)
    else:
        logger.info("Interview %s → %s", interview.uuid, status)
    interview.updated_at = datetime.now(timezone.utc)
    db.commit()

def _validate_quotes(quotes, segments: list[SegmentResult]) -> list:
    """
    Drop any quote whose text does not appear verbatim in the transcript.
    This is an explicit code-level safeguard — no fabricated quotes reach the DB.
    """
    full_transcript = " ".join(s.text for s in segments)
    valid = []
    for q in quotes:
        if q.quote_text in full_transcript:
            valid.append(q)
        else:
            logger.warning(
                "Quote dropped — text not found verbatim in transcript: %r",
                q.quote_text[:80],
            )
    return valid

def validate_upload(
    db: Session,
    interview: Interview,
    upload_path: Path,
    file_size_bytes: int,
) -> tuple[str, float]:
    """
    Validates the uploaded file with ffprobe.

    Returns:
        (media_type, duration_seconds)
    Raises:
        ValueError with a human-readable message.
    """
    media_type, duration = validate_media(upload_path, file_size_bytes)
    interview.media_type = media_type
    interview.duration_seconds = duration
    interview.file_size_bytes = file_size_bytes
    db.commit()
    return media_type, duration

def extract_audio(
    db: Session,
    interview: Interview,
    upload_path: Path,
    work_dir: Path,
) -> Path:
    _set_status(db, interview, "extracting_audio")
    try:
        wav_path = extract_and_convert(
            upload_path,
            work_dir,
            media_type=interview.media_type or "audio",
        )
    except RuntimeError as exc:
        raise RuntimeError(str(exc))

    interview.audio_path = str(wav_path.relative_to(settings.STORAGE_ROOT))
    db.commit()
    return wav_path

def transcribe(
    db: Session,
    interview: Interview,
    wav_path: Path,
    fast_mode: bool = False,
) -> list[SegmentResult]:
    _set_status(db, interview, "transcribing")

    provider = get_transcription_provider()
    result = provider.transcribe(
        wav_path,
        language=interview.language or "auto",
        vocabulary_hint=interview.vocabulary_hint,
        fast_mode=fast_mode,
    )

    interview.transcription_provider = result.provider_name
    interview.whisper_model = result.model_name
    if (not interview.language or interview.language == "auto") and result.language_detected:
        interview.language = result.language_detected

    for seg in result.segments:
        words_json = json.dumps(
            [
                {
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "probability": w.probability,
                }
                for w in seg.words
            ]
        ) if seg.words else None

        db_seg = TranscriptSegment(
            interview_id=interview.id,
            segment_index=seg.segment_index,
            start_time=seg.start,
            end_time=seg.end,
            original_text=seg.text,
            confidence=seg.confidence,
            words_json=words_json,
        )
        db.add(db_seg)

    db.commit()
    logger.info("Saved %d transcript segments for interview %s.", len(result.segments), interview.uuid)
    return result.segments

def diarize(
    db: Session,
    interview: Interview,
    wav_path: Path,
    segments: list[SegmentResult],
) -> list[SegmentResult]:
    """
    Attempt speaker diarization via pyannote.audio.
    Skips gracefully if pyannote is not installed or HF_TOKEN is not set.
    """
    _set_status(db, interview, "identifying_speakers")

    hf_token = get_settings().HF_TOKEN
    if not hf_token:
        logger.info("Diarization skipped: HF_TOKEN not set.")
        interview.diarization_run = False
        db.commit()
        return segments

    try:
        pyannote_mod = importlib.import_module("pyannote.audio")
        pipeline_cls = getattr(pyannote_mod, "Pipeline")
    except (ImportError, AttributeError):
        logger.info("Diarization skipped: pyannote.audio not installed.")
        interview.diarization_run = False
        db.commit()
        return segments

    try:
        logger.info("Running pyannote diarization…")
        try:
            pipeline = pipeline_cls.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token,
            )
        except TypeError:
            pipeline = pipeline_cls.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
        diarization = pipeline(str(wav_path))

        def _find_speaker(start: float, end: float) -> Optional[str]:
            best_speaker = None
            best_overlap = 0.0
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                overlap = min(turn.end, end) - max(turn.start, start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = speaker
            return best_speaker

        db_segs = (
            db.query(TranscriptSegment)
            .filter_by(interview_id=interview.id)
            .all()
        )
        seg_map = {s.segment_index: s for s in db_segs}

        for seg in segments:
            speaker = _find_speaker(seg.start, seg.end)
            seg.speaker = speaker
            if seg.segment_index in seg_map:
                seg_map[seg.segment_index].speaker = speaker

        interview.diarization_run = True
        db.commit()
        logger.info("Diarization complete.")

    except Exception as exc:

        logger.warning("Diarization failed (%s): %s. Continuing without speakers.", type(exc).__name__, exc)
        interview.diarization_run = False
        db.commit()

    return segments

def extract_insights(
    db: Session,
    interview: Interview,
    segments: list[SegmentResult],
) -> None:
    _set_status(db, interview, "extracting_insights")

    provider = get_analysis_provider()
    try:
        result = provider.analyze(
            segments,
            language=interview.language or "en",
            title=interview.title,
            notes=interview.notes,
        )
    except Exception as exc:
        if provider.name != "local":
            logger.warning("Primary analysis provider '%s' failed (%s); falling back to LocalAnalysisProvider", provider.name, exc)
            from app.providers.local_analysis_provider import LocalAnalysisProvider
            provider = LocalAnalysisProvider()
            result = provider.analyze(
                segments,
                language=interview.language or "en",
                title=interview.title,
                notes=interview.notes,
            )
        else:
            raise

    interview.analysis_provider = provider.name

    valid_quotes = _validate_quotes(result.quotes, segments)

    db_segs = db.query(TranscriptSegment).filter_by(interview_id=interview.id).all()
    idx_to_db_id = {s.segment_index: s.id for s in db_segs}
    idx_to_speaker = {s.segment_index: s.speaker for s in db_segs}

    for q in valid_quotes:
        seg_db_id = idx_to_db_id.get(q.segment_index) if q.segment_index is not None else None
        speaker = q.speaker or (idx_to_speaker.get(q.segment_index) if q.segment_index is not None else None)
        db.add(Quote(
            interview_id=interview.id,
            segment_id=seg_db_id,
            segment_index=q.segment_index,
            speaker=speaker,
            quote_text=q.quote_text,
            reason=q.reason,
        ))

    for c in result.claims:
        seg_db_id = idx_to_db_id.get(c.segment_index) if c.segment_index is not None else None
        db.add(Claim(
            interview_id=interview.id,
            segment_id=seg_db_id,
            segment_index=c.segment_index,
            claim_text=c.claim_text,
            category=c.category,
            reason=c.reason,
            suggested_verification_steps=c.suggested_verification_steps,
            status="needs_verification",
        ))

    db.add(Analysis(
        interview_id=interview.id,
        summary=result.summary,
        key_topics_json=json.dumps(result.key_topics),
        raw_response_json=result.raw_response,
    ))

    db.commit()
    logger.info(
        "Insights saved: %d quotes, %d claims for interview %s.",
        len(valid_quotes), len(result.claims), interview.uuid,
    )

def run_pipeline(
    interview_id: int,
    run_diarization: bool = False,
    fast_mode: bool = False,
) -> None:
    """
    Execute all processing stages for a given interview.
    Uses a fresh DB session (not the request session).
    """
    from app.core.database import SessionLocal

    db = SessionLocal()
    try:
        interview = db.query(Interview).filter_by(id=interview_id).first()
        if not interview:
            logger.error("run_pipeline: interview id=%s not found.", interview_id)
            return

        interview.processing_started_at = datetime.now(timezone.utc)
        _set_status(db, interview, "queued")

        upload_path = settings.STORAGE_ROOT / interview.upload_path
        work_dir = upload_path.parent

        try:
            validate_upload(db, interview, upload_path, interview.file_size_bytes or 0)
        except (ValueError, RuntimeError) as exc:
            _set_status(db, interview, "failed", str(exc))
            return

        try:
            wav_path = extract_audio(db, interview, upload_path, work_dir)
        except RuntimeError as exc:
            _set_status(db, interview, "failed", str(exc))
            return

        try:
            segments = transcribe(db, interview, wav_path, fast_mode=fast_mode)
        except Exception as exc:
            _set_status(db, interview, "failed", f"Transcription error: {exc}")
            return

        if run_diarization and settings.ENABLE_DIARIZATION:
            segments = diarize(db, interview, wav_path, segments)
        else:
            interview.diarization_run = False
            db.commit()

        try:
            extract_insights(db, interview, segments)
        except Exception as exc:

            logger.error("Insight extraction failed: %s", exc)
            interview.error_message = f"Insights skipped: {exc}"
            db.commit()

        interview.processing_completed_at = datetime.now(timezone.utc)
        _set_status(db, interview, "completed")

    except Exception as exc:
        logger.exception("Unexpected pipeline error for interview %s:", interview_id)
        try:
            interview = db.query(Interview).filter_by(id=interview_id).first()
            if interview:
                _set_status(db, interview, "failed", f"Unexpected error: {exc}")
        except Exception:
            pass
    finally:
        db.close()
