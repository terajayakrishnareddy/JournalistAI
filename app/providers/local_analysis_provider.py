"""
LocalAnalysisProvider – local heuristic insight extractor.

Provides candidate quotations, fact-check claims, summary, and key topics
directly on device without external API calls. Serves as a standalone local
engine and reliable fallback when GEMINI_API_KEY is not configured.
"""
from __future__ import annotations

import collections
import re
from typing import Optional

from app.providers.base import (
    AnalysisProvider,
    AnalysisResult,
    ClaimCandidate,
    QuoteCandidate,
    SegmentResult,
)

_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "cannot", "could", "did",
    "do", "does", "doing", "don't", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn't", "it",
    "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor",
    "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours",
    "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves",
    "yeah", "yes", "um", "uh", "like", "okay", "right", "know", "think", "mean",
}

_CLAIM_NUM_PATTERN = re.compile(
    r"\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+)\s*(%|percent|dollars?|\$|euros?|million|billion|thousand|years?|months?|days?|hours?|teams?|projects?|clients?|customers?|people|employees?)\b",
    re.IGNORECASE,
)
_CLAIM_YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})\b")
_CLAIM_PERCENT_PATTERN = re.compile(r"\b\d+(\.\d+)?\s*(%|percent)\b", re.IGNORECASE)
_CLAIM_FINANCIAL_PATTERN = re.compile(r"(\$|€|£|₹|\bRs\.?|\bdollars?|\bbudget|\brevenue|\bcost|\bprofit|\bsales)\b", re.IGNORECASE)

class LocalAnalysisProvider(AnalysisProvider):
    """Local rule-based extractor for quotes, claims, summary, and topics."""

    @property
    def name(self) -> str:
        return "local"

    def analyze(
        self,
        segments: list[SegmentResult],
        *,
        language: str = "en",
        title: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> AnalysisResult:
        if not segments:
            return AnalysisResult(
                summary="No transcript segments found to analyze.",
                key_topics=[],
                quotes=[],
                claims=[],
                raw_response=None,
            )

        quotes = self._extract_quotes(segments)
        claims = self._extract_claims(segments)
        summary, key_topics = self._extract_summary_and_topics(segments, title)

        return AnalysisResult(
            summary=summary,
            key_topics=key_topics,
            quotes=quotes,
            claims=claims,
            raw_response="Generated via LocalAnalysisProvider",
        )

    def _extract_quotes(self, segments: list[SegmentResult]) -> list[QuoteCandidate]:
        candidates: list[QuoteCandidate] = []

        opinion_markers = [
            "i believe", "my experience", "the key is", "in my opinion", "what i learned",
            "the biggest challenge", "we decided", "our goal was", "the most important",
            "i discovered", "what surprised me", "i always say", "my approach", "i noticed",
            "the result was", "we managed to", "i led", "i managed", "my responsibility",
        ]

        for s in segments:
            text = s.text.strip()
            lower = text.lower()
            if len(text) < 30 or len(text) > 300:
                continue

            reason = None
            for marker in opinion_markers:
                if marker in lower:
                    reason = f"Reflects key statement or perspective regarding: '{marker}'"
                    break

            if not reason and len(text) >= 50 and text.endswith((".", "!")):

                if any(lower.startswith(prefix) for prefix in ("i ", "we ", "my ", "our ", "the ")):
                    reason = "Substantive declarative statement from speaker."

            if reason:
                candidates.append(
                    QuoteCandidate(
                        quote_text=text,
                        segment_index=s.segment_index,
                        speaker=s.speaker,
                        reason=reason,
                    )
                )

            if len(candidates) >= 12:
                break

        if not candidates:
            for s in segments[:10]:
                t = s.text.strip()
                if len(t) >= 40:
                    candidates.append(
                        QuoteCandidate(
                            quote_text=t,
                            segment_index=s.segment_index,
                            speaker=s.speaker,
                            reason="Key passage from discussion.",
                        )
                    )
                if len(candidates) >= 5:
                    break

        return candidates

    def _extract_claims(self, segments: list[SegmentResult]) -> list[ClaimCandidate]:
        claims: list[ClaimCandidate] = []
        seen_texts = set()

        for s in segments:
            text = s.text.strip()
            if text in seen_texts or len(text) < 15:
                continue

            category = None
            reason = None
            steps = None

            if _CLAIM_PERCENT_PATTERN.search(text):
                category = "statistics"
                reason = "Contains percentage or statistical metric assertion."
                steps = "Verify baseline dataset, sample size, and original reporting context for metric accuracy."
            elif _CLAIM_FINANCIAL_PATTERN.search(text) and _CLAIM_NUM_PATTERN.search(text):
                category = "financial"
                reason = "Involves monetary values, fiscal data, or commercial figures."
                steps = "Cross-reference with audited financial statements, annual reports, or budget disclosures."
            elif _CLAIM_YEAR_PATTERN.search(text):
                category = "historical"
                reason = "Mentions specific calendar years or chronological milestones."
                steps = "Check public records, press archives, or official timelines to confirm milestone dates."
            elif _CLAIM_NUM_PATTERN.search(text):
                category = "statistics"
                reason = "Contains verifiable quantitative numerical assertions."
                steps = "Confirm count against relevant organizational records or documented figures."

            if category and reason:
                seen_texts.add(text)
                claims.append(
                    ClaimCandidate(
                        claim_text=text,
                        segment_index=s.segment_index,
                        category=category,
                        reason=reason,
                        suggested_verification_steps=steps or "Review primary source documentation to independently corroborate.",
                    )
                )

            if len(claims) >= 15:
                break

        return claims

    def _extract_summary_and_topics(
        self,
        segments: list[SegmentResult],
        title: Optional[str],
    ) -> tuple[str, list[str]]:
        all_text = " ".join(s.text for s in segments)
        words = re.findall(r"\b[a-zA-Z]{3,}\b", all_text.lower())
        filtered_words = [w for w in words if w not in _STOPWORDS]

        counts = collections.Counter(filtered_words)
        top_words = [word.capitalize() for word, cnt in counts.most_common(12) if cnt >= 2]

        first_few = [s.text.strip() for s in segments[:4] if len(s.text.strip()) > 20]
        sample_intro = " ".join(first_few)[:280]
        if sample_intro and not sample_intro.endswith("."):
            sample_intro += "..."

        subject = title or "This interview"
        summary = (
            f"Transcript review for '{subject}'. Discussion covers key topics including "
            f"{', '.join(top_words[:5]) if top_words else 'the subject matter'}. "
            f"Primary segment overview: \"{sample_intro}\""
        )

        return summary, top_words[:8]
