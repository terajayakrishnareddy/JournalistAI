"""
FastAPI router for all /api/interviews endpoints.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import re
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import (
    Analysis,
    Claim,
    Interview,
    Quote,
    TranscriptSegment,
    get_db,
)
from app.core.media_validator import validate_media
from app.core.pipeline import extract_insights, run_pipeline
from app.providers.base import SegmentResult
from app.core.schemas import (
    AnalysisResponse,
    ClaimResponse,
    ClaimUpdateRequest,
    EvidenceSearchResponse,
    InterviewCreate,
    InterviewResponse,
    InterviewStatusResponse,
    InterviewUpdate,
    ProcessRequest,
    QuoteResponse,
    SegmentCorrectionRequest,
    SegmentResponse,
    SettingsUpdateRequest,
    SpeakerRenameRequest,
    TranscriptResponse,
    WordTimestamp,
)

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/interviews", tags=["interviews"])

def _get_interview_or_404(interview_id: int, db: Session) -> Interview:
    interview = db.query(Interview).filter_by(id=interview_id).first()
    if not interview:
        raise HTTPException(status_code=404, detail=f"Interview {interview_id} not found.")
    return interview

def _safe_filename(original_name: str) -> str:
    """Return a sanitised filename with no path traversal risk."""
    name = unicodedata.normalize("NFKD", original_name)
    name = re.sub(r"[^\w.\-]", "_", name)
    name = re.sub(r"\.{2,}", "_", name)
    return name[:200] or "upload"

def _segment_to_schema(seg: TranscriptSegment) -> SegmentResponse:
    words = None
    if seg.words_json:
        raw = json.loads(seg.words_json)
        words = [WordTimestamp(**w) for w in raw]
    return SegmentResponse(
        id=seg.id,
        segment_index=seg.segment_index,
        start_time=seg.start_time,
        end_time=seg.end_time,
        original_text=seg.original_text,
        corrected_text=seg.corrected_text,
        speaker=seg.speaker,
        confidence=seg.confidence,
        words=words,
    )

@router.post("", status_code=status.HTTP_201_CREATED, response_model=InterviewResponse)
async def create_interview(
    payload: InterviewCreate,
    db: Session = Depends(get_db),
):
    interview_uuid = str(uuid.uuid4())
    interview = Interview(
        uuid=interview_uuid,
        title=payload.title,
        interview_date=payload.interview_date,
        language=payload.language,
        category=payload.category,
        notes=payload.notes,
        vocabulary_hint=payload.vocabulary_hint,
        status="waiting_for_upload",
    )
    db.add(interview)
    db.commit()
    db.refresh(interview)
    return interview

@router.patch("/{interview_id}", response_model=InterviewResponse)
def update_interview(
    interview_id: int,
    payload: InterviewUpdate,
    db: Session = Depends(get_db),
):
    interview = _get_interview_or_404(interview_id, db)
    if payload.title is not None and payload.title.strip():
        interview.title = payload.title.strip()
    if payload.interview_date is not None:
        interview.interview_date = payload.interview_date
    if payload.language is not None:
        interview.language = payload.language
    if payload.category is not None:
        interview.category = payload.category.strip() or None
    if payload.notes is not None:
        interview.notes = payload.notes.strip() or None
    if payload.vocabulary_hint is not None:
        interview.vocabulary_hint = payload.vocabulary_hint.strip() or None
    db.commit()
    db.refresh(interview)
    return interview

@router.post("/{interview_id}/audio", status_code=status.HTTP_200_OK, response_model=InterviewResponse)
async def upload_audio(
    interview_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    interview = _get_interview_or_404(interview_id, db)

    if interview.status not in ("waiting_for_upload", "uploaded", "failed"):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot replace audio when status is '{interview.status}'.",
        )

    upload_dir = settings.STORAGE_ROOT / "uploads" / interview.uuid
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = _safe_filename(file.filename or "upload")

    dest_path = upload_dir / safe_name

    size = 0
    try:
        async with aiofiles.open(dest_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                if size + len(chunk) > settings.MAX_UPLOAD_BYTES:
                    await out.close()
                    dest_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum size of {settings.MAX_UPLOAD_BYTES // (1024**3)} GB.",
                    )
                await out.write(chunk)
                size += len(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"File write error: {exc}")

    try:
        media_type, duration = validate_media(dest_path, size)
    except (ValueError, RuntimeError) as exc:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc))

    interview.original_filename = file.filename
    interview.upload_path = str(dest_path.relative_to(settings.STORAGE_ROOT))
    interview.file_size_bytes = size
    interview.media_type = media_type
    interview.duration_seconds = duration
    interview.status = "uploaded"
    interview.error_message = None
    interview.updated_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(interview)
    return interview

@router.post("/{interview_id}/process", status_code=status.HTTP_202_ACCEPTED)
async def start_processing(
    interview_id: int,
    background_tasks: BackgroundTasks,
    payload: ProcessRequest = None,
    db: Session = Depends(get_db),
):
    if payload is None:
        payload = ProcessRequest()
    interview = _get_interview_or_404(interview_id, db)

    if not interview.upload_path:
        raise HTTPException(status_code=422, detail="No audio file uploaded yet.")

    if interview.status in ("queued", "extracting_audio", "transcribing", "identifying_speakers", "extracting_insights"):
        return {"message": "Processing is already in progress.", "interview_id": interview_id, "status": interview.status}

    db.query(TranscriptSegment).filter_by(interview_id=interview_id).delete()
    db.query(Quote).filter_by(interview_id=interview_id).delete()
    db.query(Claim).filter_by(interview_id=interview_id).delete()
    db.query(Analysis).filter_by(interview_id=interview_id).delete()
    interview.audio_path = None
    interview.error_message = None
    interview.status = "queued"
    db.commit()

    background_tasks.add_task(run_pipeline, interview_id, payload.run_diarization, payload.fast_mode)
    return {"message": "Processing started.", "interview_id": interview_id}

@router.get("", response_model=list[InterviewResponse])
def list_interviews(
    skip: int = 0,
    limit: int = Query(default=50, le=200),
    db: Session = Depends(get_db),
):
    return (
        db.query(Interview)
        .order_by(Interview.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

@router.get("/{interview_id}", response_model=InterviewResponse)
def get_interview(interview_id: int, db: Session = Depends(get_db)):
    return _get_interview_or_404(interview_id, db)

@router.api_route("/{interview_id}/audio", methods=["GET", "HEAD"])
def get_interview_audio(
    interview_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    interview = _get_interview_or_404(interview_id, db)
    target_path = None
    media_type = "audio/wav"

    if interview.audio_path:
        candidate = settings.STORAGE_ROOT / interview.audio_path
        if candidate.exists():
            target_path = candidate
            media_type = "audio/wav"

    if not target_path and interview.upload_path:
        candidate = settings.STORAGE_ROOT / interview.upload_path
        if candidate.exists():
            target_path = candidate
            suffix = candidate.suffix.lower()
            if suffix in (".mp3", ".mpeg", ".mpg", ".mpga"):
                media_type = "audio/mpeg"
            elif suffix in (".m4a", ".aac"):
                media_type = "audio/mp4"
            elif suffix in (".ogg", ".opus"):
                media_type = "audio/ogg"
            elif suffix in (".flac",):
                media_type = "audio/flac"
            elif suffix in (".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".3gp", ".wmv", ".ts", ".asf"):
                media_type = f"video/{suffix.lstrip('.')}"
            else:
                media_type = "audio/wav"

    if not target_path or not target_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found on disk.")

    headers = {
        "Accept-Ranges": "bytes",
        "Cache-Control": "public, max-age=3600",
    }
    return FileResponse(
        path=str(target_path),
        media_type=media_type,
        headers=headers,
        content_disposition_type="inline",
        method=request.method,
    )

@router.get("/{interview_id}/status", response_model=InterviewStatusResponse)
def get_status(interview_id: int, db: Session = Depends(get_db)):
    return _get_interview_or_404(interview_id, db)

@router.get("/{interview_id}/transcript", response_model=TranscriptResponse)
def get_transcript(interview_id: int, db: Session = Depends(get_db)):
    interview = _get_interview_or_404(interview_id, db)
    segs = (
        db.query(TranscriptSegment)
        .filter_by(interview_id=interview_id)
        .order_by(TranscriptSegment.segment_index)
        .all()
    )
    return TranscriptResponse(
        interview_id=interview_id,
        total_segments=len(segs),
        diarization_run=interview.diarization_run,
        segments=[_segment_to_schema(s) for s in segs],
    )

@router.patch("/{interview_id}/transcript/{segment_id}", response_model=SegmentResponse)
def correct_segment(
    interview_id: int,
    segment_id: int,
    payload: SegmentCorrectionRequest,
    db: Session = Depends(get_db),
):
    _get_interview_or_404(interview_id, db)
    seg = db.query(TranscriptSegment).filter_by(id=segment_id, interview_id=interview_id).first()
    if not seg:
        raise HTTPException(status_code=404, detail="Segment not found.")
    if payload.corrected_text is not None:
        seg.corrected_text = payload.corrected_text
    if payload.speaker is not None:
        seg.speaker = payload.speaker.strip() or None
    db.commit()
    db.refresh(seg)
    return _segment_to_schema(seg)

@router.get("/{interview_id}/quotes", response_model=list[QuoteResponse])
def get_quotes(interview_id: int, db: Session = Depends(get_db)):
    _get_interview_or_404(interview_id, db)
    return db.query(Quote).filter_by(interview_id=interview_id).all()

@router.get("/{interview_id}/claims", response_model=list[ClaimResponse])
def get_claims(interview_id: int, db: Session = Depends(get_db)):
    _get_interview_or_404(interview_id, db)
    return db.query(Claim).filter_by(interview_id=interview_id).all()

@router.patch("/{interview_id}/claims/{claim_id}", response_model=ClaimResponse)
def update_claim(
    interview_id: int,
    claim_id: int,
    payload: ClaimUpdateRequest,
    db: Session = Depends(get_db),
):
    _get_interview_or_404(interview_id, db)
    claim = db.query(Claim).filter_by(id=claim_id, interview_id=interview_id).first()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found.")

    claim.status = payload.status
    if payload.verification_notes is not None:
        claim.verification_notes = payload.verification_notes
    if payload.evidence_url is not None:
        claim.evidence_url = payload.evidence_url
    if payload.evidence_title is not None:
        claim.evidence_title = payload.evidence_title
    if payload.evidence_snippet is not None:
        claim.evidence_snippet = payload.evidence_snippet

    claim.verified_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(claim)
    return claim

@router.post("/{interview_id}/claims/{claim_id}/verify-evidence", response_model=EvidenceSearchResponse)
def search_claim_evidence(
    interview_id: int,
    claim_id: int,
    db: Session = Depends(get_db),
):
    import html
    import urllib.parse
    import urllib.request

    _get_interview_or_404(interview_id, db)
    claim = db.query(Claim).filter_by(id=claim_id, interview_id=interview_id).first()
    if not claim:
        raise HTTPException(status_code=404, detail="Claim not found.")

    clean_text = re.sub(r"[^\w\s]", " ", claim.claim_text)
    words = [w for w in clean_text.split() if len(w) > 2 and w.lower() not in ("that", "this", "with", "from", "they", "have", "been", "were")]
    query_str = " ".join(words[:7]) if words else claim.claim_text[:50]

    encoded_q = urllib.parse.quote(query_str)
    wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_q}&utf8=&format=json"

    evidence_title = None
    evidence_url = None
    evidence_snippet = None
    suggested_verdict = "unverified"
    results_found = 0

    try:
        req = urllib.request.Request(
            wiki_url,
            headers={"User-Agent": "JournalistAI-FactCheck/1.0 (Educational Academic Project)"},
        )
        with urllib.request.urlopen(req, timeout=6) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            search_items = res_data.get("query", {}).get("search", [])
            results_found = len(search_items)
            if search_items:
                top_hit = search_items[0]
                evidence_title = top_hit.get("title")
                raw_snippet = top_hit.get("snippet", "")

                clean_snippet = html.unescape(re.sub(r"<[^>]+>", "", raw_snippet))
                evidence_snippet = clean_snippet
                safe_title = urllib.parse.quote(evidence_title.replace(" ", "_"))
                evidence_url = f"https://en.wikipedia.org/wiki/{safe_title}"

                claim_lower = claim.claim_text.lower()
                snippet_lower = clean_snippet.lower()
                overlap_count = sum(1 for w in words if w.lower() in snippet_lower)
                if overlap_count >= 2:
                    suggested_verdict = "supported"
                else:
                    suggested_verdict = "needs_verification"

                if not claim.evidence_url:
                    claim.evidence_url = evidence_url
                    claim.evidence_title = evidence_title
                    claim.evidence_snippet = evidence_snippet
                    db.commit()
    except Exception as exc:
        logger.warning("Evidence search error: %s", exc)

    return EvidenceSearchResponse(
        claim_id=claim.id,
        claim_text=claim.claim_text,
        query_used=query_str,
        results_found=results_found,
        evidence_title=evidence_title,
        evidence_url=evidence_url,
        evidence_snippet=evidence_snippet,
        suggested_verdict=suggested_verdict,
    )

@router.get("/{interview_id}/analysis", response_model=AnalysisResponse)
def get_analysis(interview_id: int, db: Session = Depends(get_db)):
    _get_interview_or_404(interview_id, db)
    analysis = db.query(Analysis).filter_by(interview_id=interview_id).first()
    quotes = db.query(Quote).filter_by(interview_id=interview_id).all()
    claims = db.query(Claim).filter_by(interview_id=interview_id).all()

    return AnalysisResponse(
        summary=analysis.summary if analysis else None,
        key_topics=analysis.key_topics if analysis else [],
        quotes=[QuoteResponse.model_validate(q) for q in quotes],
        claims=[ClaimResponse.model_validate(c) for c in claims],
    )

@router.post("/{interview_id}/speakers/rename")
def rename_speaker(
    interview_id: int,
    payload: SpeakerRenameRequest,
    db: Session = Depends(get_db),
):
    _get_interview_or_404(interview_id, db)
    old_spk = payload.old_speaker.strip()
    new_spk = payload.new_speaker.strip()
    if not new_spk:
        raise HTTPException(status_code=400, detail="New speaker name cannot be empty.")

    segs = (
        db.query(TranscriptSegment)
        .filter_by(interview_id=interview_id)
        .all()
    )
    count = 0
    for s in segs:
        current = (s.speaker or "").strip()
        if current.lower() == old_spk.lower() or (not current and old_spk.lower() in ("speaker", "unassigned", "speaker 1")):
            s.speaker = new_spk
            count += 1

    quotes = db.query(Quote).filter_by(interview_id=interview_id).all()
    for q in quotes:
        if (q.speaker or "").strip().lower() == old_spk.lower():
            q.speaker = new_spk

    db.commit()
    return {"updated_segments": count, "old_speaker": old_spk, "new_speaker": new_spk}

def _format_srt_time(seconds: float) -> str:
    millis = int(round((seconds - int(seconds)) * 1000))
    secs = int(seconds) % 60
    mins = (int(seconds) // 60) % 60
    hours = int(seconds) // 3600
    return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"

def _format_vtt_time(seconds: float) -> str:
    millis = int(round((seconds - int(seconds)) * 1000))
    secs = int(seconds) % 60
    mins = (int(seconds) // 60) % 60
    hours = int(seconds) // 3600
    return f"{hours:02d}:{mins:02d}:{secs:02d}.{millis:03d}"

@router.get("/{interview_id}/export")
def export_transcript(
    interview_id: int,
    format: str = Query("txt", pattern=r"^(txt|json|csv|srt|vtt)$"),
    db: Session = Depends(get_db),
):
    interview = _get_interview_or_404(interview_id, db)
    segs = (
        db.query(TranscriptSegment)
        .filter_by(interview_id=interview_id)
        .order_by(TranscriptSegment.segment_index)
        .all()
    )

    safe_title = re.sub(r"[^\w\-]", "_", interview.title[:50])

    if format == "txt":
        lines = [f"# {interview.title}\n"]
        for s in segs:
            text = s.corrected_text or s.original_text
            speaker_part = f"[{s.speaker}] " if s.speaker else ""
            lines.append(f"[{s.start_time:.2f}s - {s.end_time:.2f}s] {speaker_part}{text}")
        content = "\n".join(lines)
        return StreamingResponse(
            io.BytesIO(content.encode("utf-8")),
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="{safe_title}.txt"'},
        )

    elif format == "json":
        data = {
            "interview_id": interview_id,
            "title": interview.title,
            "segments": [
                {
                    "segment_index": s.segment_index,
                    "start": s.start_time,
                    "end": s.end_time,
                    "speaker": s.speaker,
                    "original_text": s.original_text,
                    "corrected_text": s.corrected_text,
                    "confidence": s.confidence,
                }
                for s in segs
            ],
        }
        return StreamingResponse(
            io.BytesIO(json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{safe_title}.json"'},
        )

    elif format == "srt":
        srt_lines = []
        for i, s in enumerate(segs, 1):
            text = s.corrected_text or s.original_text
            speaker_tag = f"[{s.speaker}] " if s.speaker else ""
            srt_lines.append(f"{i}\n{_format_srt_time(s.start_time)} --> {_format_srt_time(s.end_time)}\n{speaker_tag}{text}\n")
        content = "\n".join(srt_lines)
        return StreamingResponse(
            io.BytesIO(content.encode("utf-8")),
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="{safe_title}.srt"'},
        )

    elif format == "vtt":
        vtt_lines = ["WEBVTT\n"]
        for i, s in enumerate(segs, 1):
            text = s.corrected_text or s.original_text
            speaker_tag = f"<v {s.speaker}>" if s.speaker else ""
            speaker_end = "</v>" if s.speaker else ""
            vtt_lines.append(f"{i}\n{_format_vtt_time(s.start_time)} --> {_format_vtt_time(s.end_time)}\n{speaker_tag}{text}{speaker_end}\n")
        content = "\n".join(vtt_lines)
        return StreamingResponse(
            io.BytesIO(content.encode("utf-8")),
            media_type="text/vtt",
            headers={"Content-Disposition": f'attachment; filename="{safe_title}.vtt"'},
        )

    else:
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=["segment_index", "start", "end", "speaker", "original_text", "corrected_text", "confidence"],
        )
        writer.writeheader()
        for s in segs:
            writer.writerow({
                "segment_index": s.segment_index,
                "start": s.start_time,
                "end": s.end_time,
                "speaker": s.speaker or "",
                "original_text": s.original_text,
                "corrected_text": s.corrected_text or "",
                "confidence": s.confidence or "",
            })
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode("utf-8")),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{safe_title}.csv"'},
        )

@router.delete("/{interview_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_interview(interview_id: int, db: Session = Depends(get_db)):
    interview = _get_interview_or_404(interview_id, db)

    upload_dir = settings.STORAGE_ROOT / "uploads" / interview.uuid
    if upload_dir.exists():
        import shutil
        shutil.rmtree(upload_dir, ignore_errors=True)

    db.delete(interview)
    db.commit()

@router.post("/{interview_id}/extract-insights")
def trigger_extract_insights(interview_id: int, db: Session = Depends(get_db)):
    interview = _get_interview_or_404(interview_id, db)
    db_segs = (
        db.query(TranscriptSegment)
        .filter_by(interview_id=interview_id)
        .order_by(TranscriptSegment.segment_index)
        .all()
    )
    if not db_segs:
        if interview.status in ("queued", "extracting_audio", "transcribing", "identifying_speakers"):
            raise HTTPException(
                status_code=409,
                detail=f"Transcription is currently in progress ({interview.status.replace('_', ' ')}). Please wait for transcription to finish before extracting insights.",
            )
        raise HTTPException(
            status_code=400,
            detail="No transcript segments found to analyze. Please transcribe the audio first.",
        )

    db.query(Quote).filter_by(interview_id=interview_id).delete()
    db.query(Claim).filter_by(interview_id=interview_id).delete()
    db.query(Analysis).filter_by(interview_id=interview_id).delete()
    db.commit()

    segments = [
        SegmentResult(
            segment_index=s.segment_index,
            start=s.start_time,
            end=s.end_time,
            text=s.corrected_text or s.original_text,
            speaker=s.speaker,
            confidence=s.confidence,
        )
        for s in db_segs
    ]

    extract_insights(db, interview, segments)
    interview.status = "completed"
    interview.updated_at = datetime.now(timezone.utc)
    db.commit()

    quotes_count = db.query(Quote).filter_by(interview_id=interview_id).count()
    claims_count = db.query(Claim).filter_by(interview_id=interview_id).count()
    analysis = db.query(Analysis).filter_by(interview_id=interview_id).first()

    return {
        "message": "Insights extracted successfully.",
        "interview_id": interview_id,
        "quotes_count": quotes_count,
        "claims_count": claims_count,
        "summary": analysis.summary if analysis else None,
        "provider": interview.analysis_provider,
    }

@router.get("/config/settings")
def get_app_settings():
    s = get_settings()
    has_key = bool(s.GEMINI_API_KEY and s.GEMINI_API_KEY.strip() != "your_gemini_api_key_here")
    return {
        "gemini_api_key_configured": has_key,
        "gemini_model": s.GEMINI_MODEL,
        "transcription_provider": s.TRANSCRIPTION_PROVIDER,
        "whisper_model": s.WHISPER_MODEL_SIZE,
    }

@router.get("/{interview_id}/dossier")
def get_dossier_report(interview_id: int, db: Session = Depends(get_db)):
    from fastapi.responses import HTMLResponse
    import html

    interview = _get_interview_or_404(interview_id, db)
    segs = db.query(TranscriptSegment).filter_by(interview_id=interview_id).order_by(TranscriptSegment.segment_index).all()
    quotes = db.query(Quote).filter_by(interview_id=interview_id).all()
    claims = db.query(Claim).filter_by(interview_id=interview_id).all()
    analysis = db.query(Analysis).filter_by(interview_id=interview_id).first()

    summary_text = analysis.summary if analysis and analysis.summary else "No editorial synthesis generated."
    key_topics = analysis.key_topics if analysis and analysis.key_topics else []

    quotes_html = "".join([
        f"""<div class='item-card'>
            <div class='quote-text'>“{html.escape(q.quote_text)}”</div>
            <div class='meta'>Speaker: <strong>{html.escape(q.speaker or 'Unassigned')}</strong> | Segment #{q.segment_index} | <em>{html.escape(q.reason or 'Notable quote')}</em></div>
        </div>""" for q in quotes
    ]) or "<p><em>No candidate quotations extracted.</em></p>"

    claims_html = "".join([
        f"""<div class='item-card claim-{c.status}'>
            <div style='display:flex; justify-content:space-between;'>
                <strong>{html.escape(c.claim_text)}</strong>
                <span class='badge badge-{c.status}'>{c.status.upper().replace('_', ' ')}</span>
            </div>
            <div class='meta' style='margin-top:6px;'>Category: <strong>{html.escape(c.category or 'General')}</strong> | Suggested: <em>{html.escape(c.suggested_verification_steps or 'N/A')}</em></div>
            {f"<div class='evidence-box'><strong>Evidence Source:</strong> <a href='{html.escape(c.evidence_url)}' target='_blank'>{html.escape(c.evidence_title or c.evidence_url)}</a><p style='margin-top:4px;'>{html.escape(c.evidence_snippet or '')}</p></div>" if c.evidence_url else ""}
            {f"<div class='notes-box'><strong>Verification Notes:</strong> {html.escape(c.verification_notes)}</div>" if c.verification_notes else ""}
        </div>""" for c in claims
    ]) or "<p><em>No claims flagged for fact-checking.</em></p>"

    transcript_html = "".join([
        f"""<tr>
            <td style='white-space:nowrap; color:#64748b;'>{s.start_time:.1f}s - {s.end_time:.1f}s</td>
            <td><strong>{html.escape(s.speaker or 'Speaker')}</strong></td>
            <td>{html.escape(s.corrected_text or s.original_text)}</td>
        </tr>""" for s in segs
    ]) or "<tr><td colspan='3'>No transcript segments found.</td></tr>"

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Journalistic Dossier — {html.escape(interview.title)}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #1e293b; max-width: 900px; margin: 40px auto; padding: 0 20px; }}
        h1 {{ font-size: 28px; margin-bottom: 4px; color: #0f172a; border-bottom: 3px solid #e11d48; padding-bottom: 12px; }}
        .header-meta {{ display: flex; justify-content: space-between; font-size: 13px; color: #64748b; margin-bottom: 30px; }}
        h2 {{ font-size: 18px; text-transform: uppercase; letter-spacing: 0.05em; color: #0f172a; border-bottom: 1px solid #e2e8f0; padding-bottom: 6px; margin-top: 36px; }}
        .item-card {{ background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 14px 16px; margin-bottom: 12px; }}
        .quote-text {{ font-size: 16px; font-weight: 500; color: #0f172a; font-style: italic; }}
        .meta {{ font-size: 12px; color: #64748b; margin-top: 4px; }}
        .badge {{ font-size: 11px; font-weight: 700; padding: 3px 8px; border-radius: 6px; text-transform: uppercase; }}
        .badge-needs_verification {{ background: #fef3c7; color: #92400e; }}
        .badge-supported {{ background: #dcfce7; color: #166534; }}
        .badge-refuted {{ background: #fee2e2; color: #991b1b; }}
        .badge-unverified {{ background: #f1f5f9; color: #475569; }}
        .evidence-box {{ background: #fff1f2; border-left: 3px solid #e11d48; border-radius: 6px; padding: 8px 12px; margin-top: 8px; font-size: 13px; }}
        .notes-box {{ background: #fffbeb; border-left: 3px solid #f59e0b; border-radius: 6px; padding: 8px 12px; margin-top: 8px; font-size: 13px; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 14px; margin-top: 10px; }}
        th, td {{ padding: 8px 10px; border-bottom: 1px solid #e2e8f0; text-align: left; }}
        th {{ background: #f1f5f9; color: #475569; font-weight: 600; }}
        .print-btn {{ background: #0f172a; color: #fff; padding: 8px 16px; border: none; border-radius: 6px; font-weight: 600; cursor: pointer; float: right; display: inline-flex; align-items: center; gap: 6px; }}
        @media print {{ .print-btn {{ display: none; }} body {{ margin: 0; padding: 0; }} }}
    </style>
</head>
<body>
    <button class='print-btn' onclick='window.print()'><svg width='15' height='15' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><polyline points='6 9 6 2 18 2 18 9'/><path d='M6 18H4a2 2 0 0 1-2-2v-5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v5a2 2 0 0 1-2 2h-2'/><rect x='6' y='14' width='12' height='8'/></svg> Print / Save as PDF</button>
    <h1>Investigative Dossier & Fact-Check Report</h1>
    <div class='header-meta'>
        <div><strong>Story:</strong> {html.escape(interview.title)} ({html.escape(interview.media_type or 'Audio').upper()})</div>
        <div><strong>Date:</strong> {interview.created_at.strftime('%B %d, %Y') if interview.created_at else 'N/A'} | <strong>Status:</strong> {interview.status.upper()}</div>
    </div>

    <h2>1. Executive Summary & Core Insights</h2>
    <div class='item-card'>
        <p>{html.escape(summary_text)}</p>
        <p style='margin-top:10px;'><strong>Key Topics:</strong> {' '.join([f"<span class='badge' style='background:#e2e8f0; margin-right:4px;'>{html.escape(t)}</span>" for t in key_topics]) if key_topics else 'None recorded'}</p>
    </div>

    <h2>2. Candidate Quotations (Verbatim Validated)</h2>
    {quotes_html}

    <h2>3. Factual Claims & Verification Dossier</h2>
    {claims_html}

    <h2>4. Complete Time-Synchronized Transcript</h2>
    <table>
        <thead>
            <tr><th>Timestamp</th><th>Speaker</th><th>Statement</th></tr>
        </thead>
        <tbody>
            {transcript_html}
        </tbody>
    </table>

    <footer style='margin-top:50px; text-align:center; font-size:12px; color:#94a3b8; border-top:1px solid #e2e8f0; padding-top:16px;'>
        JournalistAI Newsroom Intelligence Platform — Investigative Integrity Audit Report
    </footer>
</body>
</html>"""
    return HTMLResponse(content=html_content)

@router.get("/{interview_id}/evaluation")
def get_evaluation_metrics(interview_id: int, db: Session = Depends(get_db)):
    interview = _get_interview_or_404(interview_id, db)
    segs = db.query(TranscriptSegment).filter_by(interview_id=interview_id).all()
    claims = db.query(Claim).filter_by(interview_id=interview_id).all()
    quotes = db.query(Quote).filter_by(interview_id=interview_id).all()

    total_words = sum(len((s.corrected_text or s.original_text).split()) for s in segs)
    edited_segs = sum(1 for s in segs if s.corrected_text is not None)
    correction_rate = round((edited_segs / len(segs) * 100), 1) if segs else 0.0

    rtf = None
    processing_time_s = None
    if interview.processing_started_at and interview.processing_completed_at:
        diff = (interview.processing_completed_at - interview.processing_started_at).total_seconds()
        processing_time_s = round(diff, 2)
        if interview.duration_seconds and interview.duration_seconds > 0:
            rtf = round(diff / interview.duration_seconds, 3)

    verified_claims = sum(1 for c in claims if c.status in ("supported", "refuted"))
    claim_verification_rate = round((verified_claims / len(claims) * 100), 1) if claims else 0.0

    return {
        "interview_id": interview.id,
        "title": interview.title,
        "duration_seconds": interview.duration_seconds,
        "processing_time_seconds": processing_time_s,
        "real_time_factor_rtf": rtf,
        "total_segments": len(segs),
        "total_words": total_words,
        "edited_segments_count": edited_segs,
        "human_correction_rate_percent": correction_rate,
        "quotes_extracted_count": len(quotes),
        "claims_extracted_count": len(claims),
        "claims_verified_count": verified_claims,
        "claim_verification_rate_percent": claim_verification_rate,
        "transcription_provider": interview.transcription_provider or "whisper",
        "whisper_model": interview.whisper_model or "large-v3",
        "analysis_provider": interview.analysis_provider or "local",
    }

@router.post("/config/settings")
def update_app_settings(payload: SettingsUpdateRequest):
    import os
    env_path = Path(".env")
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []

    updates = {}
    if payload.gemini_api_key is not None:
        key_val = payload.gemini_api_key.strip()
        os.environ["GEMINI_API_KEY"] = key_val
        updates["GEMINI_API_KEY"] = key_val
    if payload.gemini_model is not None:
        model_val = payload.gemini_model.strip()
        if model_val:
            os.environ["GEMINI_MODEL"] = model_val
            updates["GEMINI_MODEL"] = model_val

    if updates:
        seen = set()
        new_lines = []
        for line in lines:
            matched_key = next((k for k in updates if line.startswith(f"{k}=")), None)
            if matched_key:
                new_lines.append(f"{matched_key}={updates[matched_key]}")
                seen.add(matched_key)
            else:
                new_lines.append(line)
        for k, v in updates.items():
            if k not in seen:
                new_lines.append(f"{k}={v}")
        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    from app.core.config import get_settings as _get_settings
    _get_settings.cache_clear()
    from app.providers.registry import get_analysis_provider
    get_analysis_provider.cache_clear()

    return {"message": "Settings updated successfully."}
