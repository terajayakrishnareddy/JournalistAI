"""
Audio preprocessing module.

- Extracts audio from video using ffmpeg.
- Converts any audio to 16 kHz mono WAV (required by Whisper).
- All output is written to the interview's working directory.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from app.core.config import get_settings

settings = get_settings()

def extract_and_convert(
    input_path: Path,
    output_dir: Path,
    *,
    media_type: str = "audio",
) -> Path:
    """
    Extract audio (if video) and convert to 16 kHz mono WAV.

    Args:
        input_path:  Path to the uploaded file.
        output_dir:  Directory where the processed WAV will be written.
        media_type:  "audio" or "video".

    Returns:
        Path to the output 16 kHz mono WAV file.

    Raises:
        RuntimeError: if ffmpeg exits with a non-zero code.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_wav = output_dir / "audio_16k_mono.wav"

    cmd = [
        settings.FFMPEG_BIN,
        "-y",
        "-i", str(input_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        str(out_wav),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"ffmpeg not found at '{settings.FFMPEG_BIN}'. "
            "Please install ffmpeg and add it to PATH."
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
        )

    return out_wav
