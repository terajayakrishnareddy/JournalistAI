"""
GeminiProvider – analysis using Google Gemini API (google-genai SDK).

PRIVACY NOTE: This provider sends transcript text to Google's servers.
              Do NOT use it if the content is confidential unless you have
              reviewed Google's data-handling policies for the Gemini API.

The prompt instructs the model to:
  1. Extract candidate quotations (MUST be verbatim from the transcript).
  2. Identify claims to verify — status is always "needs_verification".
  3. Write a short summary and key topics.

All quote text is validated against the actual transcript in pipeline.py
before saving to the database.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

from app.core.config import get_settings
from app.providers.base import (
    AnalysisProvider,
    AnalysisResult,
    ClaimCandidate,
    QuoteCandidate,
    SegmentResult,
)

logger = logging.getLogger(__name__)
settings = get_settings()

try:
    from google import genai
    from google.genai import types as genai_types
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

_SYSTEM_PROMPT = """You are an AI assistant for investigative journalism.
You will be given a verbatim transcript of an interview or meeting.
Your job is to help journalists by:
1. Identifying quotable passages (candidate quotations).
2. Flagging factual claims that may need independent verification.
3. Writing a concise summary and listing key topics.

RULES YOU MUST FOLLOW:
- Candidate quotations MUST be copied EXACTLY, character-for-character, from the transcript.
  Do not paraphrase, clean up, or alter the quote text in any way.
- For every claim, the status field MUST be "needs_verification".
  You MUST NOT assess whether any claim is true, false, likely, or unlikely.
- Respond ONLY with a JSON object in the schema described below. No markdown fences.
"""

_USER_PROMPT_TEMPLATE = """Transcript title: {title}
Notes: {notes}

TRANSCRIPT (segments):
{transcript_text}

Respond with a JSON object exactly matching this schema:
{{
  "summary": "<2-4 sentence factual summary>",
  "key_topics": ["<topic1>", "<topic2>", ...],
  "quotes": [
    {{
      "quote_text": "<EXACT verbatim text from a segment>",
      "segment_index": <integer, 0-based>,
      "speaker": "<speaker label or null>",
      "reason": "<why this is notable or quotable>"
    }}
  ],
  "claims": [
    {{
      "claim_text": "<the specific factual claim>",
      "segment_index": <integer, 0-based>,
      "category": "<statistics|financial|policy|scientific|historical|date>",
      "reason": "<why this warrants verification>",
      "suggested_verification_steps": "<concrete steps a journalist should take>",
      "status": "needs_verification"
    }}
  ]
}}
"""

class GeminiProvider(AnalysisProvider):
    """Analysis backend using Google Gemini."""

    def __init__(self) -> None:
        self._client = None

    @property
    def name(self) -> str:
        return "gemini"

    def _ensure_client(self):
        if self._client is not None:
            return
        if not _GENAI_AVAILABLE:
            raise RuntimeError(
                "google-genai is not installed. Run: pip install google-genai"
            )
        if not settings.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )
        self._client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def analyze(
        self,
        segments: list[SegmentResult],
        *,
        language: str = "en",
        title: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> AnalysisResult:
        self._ensure_client()

        transcript_lines: list[str] = []
        for seg in segments:
            speaker_label = f"[{seg.speaker}] " if seg.speaker else ""
            transcript_lines.append(
                f"[{seg.segment_index}] {seg.start:.1f}s-{seg.end:.1f}s "
                f"{speaker_label}{seg.text}"
            )
        transcript_text = "\n".join(transcript_lines)

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            title=title or "Untitled",
            notes=notes or "None",
            transcript_text=transcript_text,
        )

        logger.info("Sending transcript to Gemini for analysis (%d segments).", len(segments))

        response = self._client.models.generate_content(
            model=settings.GEMINI_MODEL,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )

        raw_text = response.text
        logger.info("Received Gemini response (%d chars).", len(raw_text))

        data = self._parse_response(raw_text)

        quotes = [
            QuoteCandidate(
                quote_text=q["quote_text"],
                segment_index=q.get("segment_index"),
                speaker=q.get("speaker"),
                reason=q.get("reason", ""),
            )
            for q in data.get("quotes", [])
        ]

        claims = [
            ClaimCandidate(
                claim_text=c["claim_text"],
                segment_index=c.get("segment_index"),
                category=c.get("category", "general"),
                reason=c.get("reason", ""),
                suggested_verification_steps=c.get("suggested_verification_steps", ""),

            )
            for c in data.get("claims", [])
        ]

        return AnalysisResult(
            summary=data.get("summary", ""),
            key_topics=data.get("key_topics", []),
            quotes=quotes,
            claims=claims,
            raw_response=raw_text,
        )

    @staticmethod
    def _parse_response(raw: str) -> dict:
        """Parse JSON from model response, stripping markdown fences if present."""

        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
        cleaned = re.sub(r"```\s*$", "", cleaned.strip(), flags=re.MULTILINE)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Gemini returned invalid JSON: {exc}\nRaw response (first 500 chars):\n{raw[:500]}"
            )
