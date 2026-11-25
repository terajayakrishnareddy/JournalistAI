"""
WhisperProvider – high-performance parallel transcription using faster-whisper.

Uses:
  - Configurable model size with int8 compute type on CPU
  - Parallel multi-worker chunk processing across CPU cores
  - Fast mode that processes only the first 5-minute window for a quick preview
  - Optional word-level timestamps (disabled by default for 2-3x speedup)
  - Configurable beam size (1 for ultra-fast greedy decoding)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from app.core.config import get_settings
from app.providers.base import (
    SegmentResult,
    TranscriptionProvider,
    TranscriptionResult,
    WordResult,
)

logger = logging.getLogger(__name__)
settings = get_settings()

# Compatibility patch for PyAV 14+ with faster-whisper (metadata_errors argument removed in PyAV)
try:
    import av
    _orig_av_open = av.open
    def _patched_av_open(*args, **kwargs):
        kwargs.pop("metadata_errors", None)
        return _orig_av_open(*args, **kwargs)
    av.open = _patched_av_open
except Exception:
    pass

try:
    from faster_whisper import WhisperModel as _WhisperModel
    _FASTER_WHISPER_AVAILABLE = True
except ImportError:
    _FASTER_WHISPER_AVAILABLE = False
    _WhisperModel = None

class WhisperProvider(TranscriptionProvider):
    """
    faster-whisper backend with parallel multi-worker chunk processing.
    """

    def __init__(self) -> None:
        self._model = None
        self._loaded_model_size: Optional[str] = None
        self._num_workers: int = 1

    @property
    def name(self) -> str:
        return "whisper"

    def _ensure_model(self) -> None:
        model_size = settings.WHISPER_MODEL_SIZE
        if self._model is not None and self._loaded_model_size == model_size:
            return

        if not _FASTER_WHISPER_AVAILABLE:
            raise RuntimeError(
                "faster-whisper is not installed. "
                "Run: pip install faster-whisper"
            )

        total_cores = settings.WHISPER_CPU_THREADS if settings.WHISPER_CPU_THREADS > 0 else (os.cpu_count() or 4)
        num_workers = 1

        logger.info(
            "Loading Whisper model '%s' (device=%s, compute=%s, cpu_threads=%d)",
            model_size,
            settings.WHISPER_DEVICE,
            settings.WHISPER_COMPUTE_TYPE,
            total_cores,
        )
        try:
            self._model = _WhisperModel(
                model_size,
                device=settings.WHISPER_DEVICE,
                compute_type=settings.WHISPER_COMPUTE_TYPE,
                cpu_threads=total_cores,
                num_workers=1,
                local_files_only=True,
            )
        except Exception:
            self._model = _WhisperModel(
                model_size,
                device=settings.WHISPER_DEVICE,
                compute_type=settings.WHISPER_COMPUTE_TYPE,
                cpu_threads=total_cores,
                num_workers=1,
            )
        self._loaded_model_size = model_size
        self._num_workers = 1
        logger.info("Whisper model initialized with %d CPU threads.", total_cores)

    def _get_audio_duration(self, audio_path: Path) -> Optional[float]:
        import wave
        try:
            with wave.open(str(audio_path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    return round(frames / float(rate), 2)
        except Exception:
            pass
        return None

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str = "auto",
        vocabulary_hint: Optional[str] = None,
        fast_mode: bool = False,
    ) -> TranscriptionResult:
        self._ensure_model()

        lang_arg = None if language == "auto" else language
        duration = self._get_audio_duration(audio_path)
        logger.info("Transcribe called on '%s' (duration=%.1fs, fast_mode=%s)", audio_path.name, duration or 0, fast_mode)

        actual_audio_path = audio_path
        temp_slice = None
        if fast_mode:
            target_duration = min(duration or 60.0, 120.0)
            logger.info("Fast mode active: processing first %.1fs window", target_duration)
            if duration and duration > target_duration:
                temp_slice = audio_path.with_name(f"{audio_path.stem}_fast_slice.wav")
                import subprocess
                try:
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", str(audio_path), "-t", f"{target_duration:.1f}", "-acodec", "copy", str(temp_slice)],
                        check=True,
                        capture_output=True,
                    )
                    actual_audio_path = temp_slice
                except Exception as e:
                    logger.warning("Fast mode slice fallback: %s", e)

        raw_segments, detected_lang = self._transcribe_single(
            actual_audio_path,
            language=lang_arg,
            vocabulary_hint=vocabulary_hint,
        )
        if temp_slice and temp_slice.exists():
            try:
                temp_slice.unlink(missing_ok=True)
            except Exception:
                pass

        results: list[SegmentResult] = []
        for idx, seg in enumerate(raw_segments):
            words: list[WordResult] = []
            if getattr(seg, "words", None):
                for w in seg.words:
                    words.append(
                        WordResult(
                            word=w.word,
                            start=w.start,
                            end=w.end,
                            probability=w.probability,
                        )
                    )

            confidence: Optional[float] = None
            if hasattr(seg, "avg_logprob") and seg.avg_logprob is not None:
                import math
                confidence = round(math.exp(seg.avg_logprob), 4)

            results.append(
                SegmentResult(
                    segment_index=idx,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                    confidence=confidence,
                    words=words,
                )
            )

        logger.info(
            "Transcription complete: %d segments, detected_lang=%s",
            len(results),
            detected_lang,
        )

        return TranscriptionResult(
            segments=results,
            language_detected=detected_lang,
            provider_name=self.name,
            model_name=settings.WHISPER_MODEL_SIZE,
        )

    def _transcribe_parallel(
        self,
        audio_path: Path,
        *,
        target_duration: float,
        language: Optional[str],
        vocabulary_hint: Optional[str],
        num_workers: int,
    ) -> tuple[list, Optional[str]]:
        from concurrent.futures import ThreadPoolExecutor

        chunk_sec = target_duration / num_workers
        clips = [f"{i * chunk_sec:.1f},{(i + 1) * chunk_sec:.1f}" for i in range(num_workers)]

        def _worker(clip_str: str):
            segs, info = self._model.transcribe(
                str(audio_path),
                language=language,
                beam_size=settings.WHISPER_BEAM_SIZE,
                word_timestamps=settings.WHISPER_WORD_TIMESTAMPS,
                condition_on_previous_text=False,
                initial_prompt=vocabulary_hint or None,
                clip_timestamps=clip_str,
            )
            return list(segs), getattr(info, "language", None)

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            results = list(pool.map(_worker, clips))

        all_segs = []
        detected_lang = None
        for seg_list, lang in results:
            all_segs.extend(seg_list)
            if not detected_lang and lang:
                detected_lang = lang

        all_segs.sort(key=lambda s: s.start)
        return all_segs, detected_lang

    def _transcribe_single(
        self,
        audio_path: Path,
        *,
        language: Optional[str],
        vocabulary_hint: Optional[str],
        clip_str: Optional[str] = None,
    ) -> tuple[list, Optional[str]]:
        kwargs = {
            "language": language,
            "beam_size": settings.WHISPER_BEAM_SIZE,
            "vad_filter": True,
            "word_timestamps": settings.WHISPER_WORD_TIMESTAMPS,
            "condition_on_previous_text": False,
            "initial_prompt": vocabulary_hint or None,
        }
        if clip_str:
            kwargs["clip_timestamps"] = clip_str
            kwargs["vad_filter"] = False

        segs_gen, info = self._model.transcribe(str(audio_path), **kwargs)
        return list(segs_gen), getattr(info, "language", None)
