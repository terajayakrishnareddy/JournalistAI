"""
Media validation helpers using ffprobe.

- Validates the real file content (not just extension).
- Returns duration, codec, streams info.
- Raises ValueError with a human-readable message on failure.

NOTE: Browser MediaRecorder (WebM/Opus) files often omit duration in the
container header. We fall back to ffmpeg -f null decode to measure real
duration in that case.
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".flac", ".mp4", ".mkv", ".mov", ".webm", ".ogg", ".aac",
    ".mpeg", ".mpg", ".mpga", ".m4v", ".avi", ".wma", ".opus", ".3gp", ".wmv", ".ts", ".asf",
}

def probe_file(path: Path) -> dict:
    """Run ffprobe and return parsed JSON."""
    cmd = [
        settings.FFPROBE_BIN,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError:
        raise RuntimeError(
            f"ffprobe not found at '{settings.FFPROBE_BIN}'. "
            "Please install ffmpeg (which includes ffprobe) and add it to PATH."
        )
    if result.returncode != 0:
        detail = result.stderr.strip()[:120] if result.stderr.strip() else ""
        raise ValueError(
            "Could not read the uploaded file. "
            "If this was a mic recording, please ensure you recorded some audio (not an empty recording). "
            + (f"Technical detail: {detail}" if detail else
               "Please check the file is a valid audio or video recording.")
        )
    return json.loads(result.stdout)

def _measure_duration_via_ffmpeg(path: Path) -> float | None:
    """
    Fallback: decode the full file with ffmpeg -f null to get real duration.
    Browser WebM/Opus files recorded via MediaRecorder API do not store
    duration in the container header, so ffprobe reports N/A. This decodes
    the actual stream and reads the time from ffmpeg stderr output.
    """
    cmd = [
        settings.FFMPEG_BIN,
        "-v", "error",
        "-stats",
        "-i", str(path),
        "-f", "null",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        matches = re.findall(r"time=(\d+):(\d+):([\d.]+)", result.stderr)
        if matches:
            h, m, s = matches[-1]
            return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception as exc:
        logger.warning("ffmpeg duration fallback failed: %s", exc)
    return None

def validate_media(path: Path, file_size_bytes: int) -> tuple[str, float]:
    """
    Validate file content with ffprobe.

    Returns:
        (media_type, duration_seconds) where media_type is "audio" or "video".

    Raises:
        ValueError: human-readable rejection reason.
    """

    if file_size_bytes > settings.MAX_UPLOAD_BYTES:
        raise ValueError(
            f"File too large: {file_size_bytes / 1e9:.2f} GB "
            f"(max {settings.MAX_UPLOAD_BYTES / 1e9:.2f} GB)"
        )

    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Extension '{path.suffix}' is not in the allowed list.")

    probe = probe_file(path)

    streams = probe.get("streams", [])
    fmt = probe.get("format", {})

    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    if not has_audio:
        raise ValueError("File has no audio stream.")

    media_type = "video" if has_video else "audio"

    duration: float | None = None

    raw = fmt.get("duration")
    if raw and raw != "N/A":
        try:
            duration = float(raw)
        except (ValueError, TypeError):
            pass

    if not duration or duration <= 0:
        for s in streams:
            raw = s.get("duration")
            if raw and raw != "N/A":
                try:
                    candidate = float(raw)
                    if candidate > 0:
                        duration = candidate
                        break
                except (ValueError, TypeError):
                    pass

    if not duration or duration <= 0:
        logger.info(
            "Duration not in container headers for '%s' — using ffmpeg decode fallback.", path.name
        )
        duration = _measure_duration_via_ffmpeg(path)

    if not duration or duration <= 0:
        raise ValueError(
            "Could not determine the recording duration. "
            "If you used Live Mic Recording, please speak for at least 1 second before stopping."
        )

    if duration > settings.MAX_DURATION_SECONDS:
        raise ValueError(
            f"Duration {duration / 3600:.2f} h exceeds the maximum of "
            f"{settings.MAX_DURATION_SECONDS / 3600:.2f} h."
        )

    return media_type, duration
