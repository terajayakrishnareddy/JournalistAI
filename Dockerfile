# ---------------------------------------------------------------------------
# JournalistAI — Hugging Face Spaces / Railway / Docker deployment
# Runs the FastAPI backend (transcription + insight extraction) and serves
# the static frontend from the same origin.
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# System dependencies: ffmpeg/ffprobe for audio extraction + build tooling.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application.
COPY . .

# Runtime defaults tuned for free-tier hosting (CPU, int8 quantization).
# Override WHISPER_MODEL_SIZE to "medium" or "large-v3" on a bigger machine.
ENV WHISPER_MODEL_SIZE=small \
    WHISPER_DEVICE=cpu \
    WHISPER_COMPUTE_TYPE=int8 \
    STORAGE_ROOT=/data/storage \
    DATABASE_URL=sqlite:////data/storage/db.sqlite3 \
    FFMPEG_BIN=ffmpeg \
    FFPROBE_BIN=ffprobe \
    PYTHONUNBUFFERED=1

# Persistent storage directory (mount a volume here on Railway/HF Spaces).
RUN mkdir -p /data/storage

# Hugging Face Spaces sets PORT=7860; Railway sets its own PORT.
# We bind to 0.0.0.0 so the container is reachable from outside.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
