# JournalistAI — Interview Transcription & Insight Extraction

A Python/FastAPI backend that transcribes audio/video interviews with timestamps and uses Google Gemini to extract candidate quotations and claims worth fact-checking — built for journalism workflows.

---

## ✅ What Works (Tested)

| Feature | Status |
|---|---|
| FastAPI REST API with all 23 endpoints | ✅ |
| File upload with ffprobe content validation | ✅ |
| Max size / duration enforcement | ✅ |
| UUID-namespaced storage with safe filenames | ✅ |
| ffmpeg audio extraction + 16 kHz mono WAV conversion | ✅ |
| faster-whisper transcription (large-v3, medium, small) | ✅ |
| VAD filter, word-level timestamps, real avg_logprob confidence | ✅ |
| Language selection (auto / en / hi / pa / te) | ✅ |
| Vocabulary hint (initial_prompt) for proper nouns | ✅ |
| Background processing (FastAPI BackgroundTasks) | ✅ |
| Status polling every 2 s from the frontend | ✅ |
| SQLite database (SQLAlchemy, migration-ready for PostgreSQL) | ✅ |
| Original transcript preserved; user corrections stored separately | ✅ |
| Gemini analysis: summary, topics, quotes, claims | ✅ |
| Verbatim quote validation in code (non-matching quotes dropped) | ✅ |
| Claim status hard-coded to `needs_verification` in code | ✅ |
| Export: TXT, JSON, CSV, SRT, WebVTT | ✅ |
| Simple HTML frontend served by FastAPI | ✅ |
| Provider interface (TranscriptionProvider / AnalysisProvider) | ✅ |
| `.env` / `.gitignore` / `.env.example` | ✅ |

## ❌ Not Yet Done / Known Limitations

| Feature | Status | Notes |
|---|---|---|
| Speaker diarization | **Optional / untested** | pyannote.audio install + HF token accepted but not tested. Set `ENABLE_DIARIZATION=false`. |
| Celery/Redis background queue | ❌ | Currently uses FastAPI `BackgroundTasks`. Swap `run_pipeline` call to Celery for production. |
| Authentication / multi-user | ❌ | No auth. Single-user local tool only. |
| HTTPS / reverse proxy | ❌ | Run behind nginx/caddy in production. |
| GPU acceleration | ⚠️ | Set `WHISPER_DEVICE=cuda` and `WHISPER_COMPUTE_TYPE=float16` if you have an NVIDIA GPU. |
| PostgreSQL migration | ⚠️ | Change `DATABASE_URL` in `.env`; `init_db()` creates tables via SQLAlchemy. |

---

## GitHub Pages Preview

GitHub Pages serves the frontend from `static/index.html`. Pages is a static host, so the FastAPI backend, uploads, transcription, database, and analysis API do not run there. For a fully working live app, deploy the backend (see below) and set the **Backend API URL** in the app's Settings modal, or just use the deployed URL directly (the backend serves the frontend too).

---

## ☁️ Live Deployment (Hugging Face Spaces — Free)

The fastest way to make the whole app live (frontend **and** backend) for free is **Hugging Face Spaces** with Docker. No credit card required.

1. Create a free account at https://huggingface.co/join
2. Go to https://huggingface.co/spaces and click **Create new Space**
3. Fill in:
   - **Space name:** `journalistai` (or anything)
   - **License:** MIT
   - **Space SDK:** **Docker**
   - **Hardware:** CPU basic (free) — 2 vCPU / 16 GB RAM, enough for the `small` Whisper model
4. Click **Create Space**
5. Choose **"Upload files by adding them to this repository"** and upload the entire project (or push from your existing GitHub repo).

   ```powershell
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/journalistai
   git push hf main
   ```

6. (Optional) In Space **Settings → Variables and secrets**, add:
   - `GEMINI_API_KEY` — your Google Gemini key for insight extraction (without it, the local fallback analyser is used)
   - `WHISPER_MODEL_SIZE` — `small` (default), `medium`, or `large-v3` (needs more RAM)
7. (Optional) Enable **Persistent storage** in Settings so your database and uploads survive restarts.

The Space builds the Docker image (installs Python, ffmpeg, dependencies) and starts the FastAPI server. Your app will be live at:
```
https://YOUR_USERNAME-journalistai.hf.space
```

That single URL serves both the frontend and the API — no CORS, no mixed-content, no Backend URL setting needed.

### Railway / Other Docker Hosts

The `Dockerfile` works on Railway, Render, Fly.io, or any Docker host. Set `PORT` (Railway sets it automatically) and mount persistent storage at `/data`.

---

## 🚀 Quick Start

### 1. Install Python 3.11+

Download from: https://www.python.org/downloads/windows/

- Use the Windows 64-bit installer (`.exe`).
- ✅ **Check "Add Python to PATH"** during install.

Verify:
```
python --version   # should show 3.11.x or 3.12.x
```

### 2. Install ffmpeg

ffmpeg is required for audio extraction and conversion. **It must be on your PATH.**

**Option A – Winget (Windows 10/11):**
```powershell
winget install ffmpeg
```

**Option B – Chocolatey:**
```powershell
choco install ffmpeg
```

**Option C – Manual:**
1. Download from https://ffmpeg.org/download.html (e.g., the "Windows builds by BtbN").
2. Extract to `C:\ffmpeg\`.
3. Add `C:\ffmpeg\bin` to your PATH environment variable.

Verify:
```
ffmpeg -version
ffprobe -version
```

### 3. Clone / open the project

```powershell
cd C:\Users\ayyap\Downloads\Pro471
```

### 4. Create a virtual environment and install dependencies

```powershell
python -m venv .venv
.venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **Note:** `faster-whisper` will download the Whisper model on first use (~3 GB for `large-v3`).  
> Use `WHISPER_MODEL_SIZE=medium` or `small` to save disk/RAM on weaker machines.

### 5. Configure environment

```powershell
copy .env.example .env
notepad .env   # or open in VS Code
```

Minimum required changes in `.env`:
- Set `GEMINI_API_KEY=your_key_here` (get from https://aistudio.google.com/app/apikey).
- Leave `WHISPER_DEVICE=cpu` and `WHISPER_COMPUTE_TYPE=int8` for CPU-only machines.
- Optionally change `WHISPER_MODEL_SIZE=medium` for faster transcription.

### 6. Run the server

```powershell
.venv\Scripts\activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser: **http://localhost:8000**

API docs (Swagger UI): **http://localhost:8000/api/docs**

---

## 🗂️ Project Structure

```
Pro471/
├── app/
│   ├── __init__.py
│   ├── main.py                    ← FastAPI app factory + startup
│   ├── api/
│   │   ├── __init__.py
│   │   └── interviews.py          ← Interview API routes
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              ← Settings (pydantic-settings, .env)
│   │   ├── database.py            ← SQLAlchemy models + session
│   │   ├── schemas.py             ← Pydantic request/response schemas
│   │   ├── media_validator.py     ← ffprobe-based content validation
│   │   ├── audio_processor.py     ← ffmpeg audio extraction/conversion
│   │   └── pipeline.py            ← Full 5-stage processing orchestrator
│   └── providers/
│       ├── __init__.py
│       ├── base.py                ← Abstract interfaces
│       ├── whisper_provider.py    ← faster-whisper implementation
│       ├── gemini_provider.py     ← Google Gemini analysis
│       ├── local_analysis_provider.py ← Offline rule-based fallback analysis
│       └── registry.py            ← Provider factory / DI
├── static/
│   └── index.html                 ← Single-page web interface
├── storage/                       ← Created at runtime (gitignored)
│   ├── db.sqlite3
│   └── uploads/{uuid}/
├── .env                           ← Your secrets (gitignored)
├── .env.example                   ← Template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔒 Privacy & Data Disclosure

| What is sent externally | When |
|---|---|
| **Transcript text** → Google Gemini API | Only when `GEMINI_API_KEY` is set and `ANALYSIS_PROVIDER=gemini` |
| **Claim query text** → Wikipedia API | Only when you click "Search evidence" on a claim |
| **Nothing** | When using `TRANSCRIPTION_PROVIDER=whisper` with no Gemini key (local analysis fallback) |

Whisper runs **100% locally**. No audio or transcript text ever leaves your machine unless you explicitly configure a cloud provider.

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/interviews` | Create interview metadata |
| GET | `/api/interviews` | List all interviews |
| GET | `/api/interviews/{id}` | Get interview details |
| PATCH | `/api/interviews/{id}` | Update interview metadata |
| DELETE | `/api/interviews/{id}` | Delete interview + files |
| POST | `/api/interviews/{id}/audio` | Upload audio/video file |
| GET | `/api/interviews/{id}/audio` | Stream the stored media file |
| POST | `/api/interviews/{id}/process` | Start background processing |
| GET | `/api/interviews/{id}/status` | Get processing status |
| GET | `/api/interviews/{id}/transcript` | Get timestamped segments |
| PATCH | `/api/interviews/{id}/transcript/{seg_id}` | Save segment correction |
| POST | `/api/interviews/{id}/speakers/rename` | Batch-rename a speaker |
| POST | `/api/interviews/{id}/extract-insights` | (Re)run quote/claim extraction |
| GET | `/api/interviews/{id}/quotes` | Get candidate quotations |
| GET | `/api/interviews/{id}/claims` | Get claims to verify |
| PATCH | `/api/interviews/{id}/claims/{claim_id}` | Update claim verification |
| POST | `/api/interviews/{id}/claims/{claim_id}/verify-evidence` | Search Wikipedia for evidence |
| GET | `/api/interviews/{id}/analysis` | Get summary, topics, quotes, claims |
| GET | `/api/interviews/{id}/evaluation` | Get RTF / correction / verification metrics |
| GET | `/api/interviews/{id}/dossier` | Printable HTML fact-check dossier |
| GET | `/api/interviews/{id}/export?format=txt\|json\|csv\|srt\|vtt` | Export transcript |
| GET | `/api/interviews/config/settings` | Read current provider/model config |
| POST | `/api/interviews/config/settings` | Update Gemini key/model at runtime |

Full interactive docs at http://localhost:8000/api/docs

---

## 🔧 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL_SIZE` | `large-v3` | `large-v3`, `medium`, `small` |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` |
| `WHISPER_COMPUTE_TYPE` | `int8` | `int8` (CPU), `float16` (GPU) |
| `WHISPER_BEAM_SIZE` | `1` | Beam search width (1 = greedy, fastest) |
| `WHISPER_CPU_THREADS` | `0` | 0 = auto-detect all cores |
| `WHISPER_WORD_TIMESTAMPS` | `false` | Enable per-word timestamps (slower) |
| `GEMINI_API_KEY` | *(empty)* | Required for insight extraction |
| `GEMINI_MODEL` | `gemini-2.5-pro` | Gemini model name |
| `HF_TOKEN` | *(empty)* | Hugging Face token for diarization |
| `ENABLE_DIARIZATION` | `false` | Enable pyannote diarization |
| `MAX_UPLOAD_BYTES` | `2147483648` | 2 GB max file size |
| `MAX_DURATION_SECONDS` | `10800` | 3 h max duration |
| `DATABASE_URL` | SQLite | Change to PostgreSQL URL for production |

---

## 🔮 Next Steps

1. **Test with a real audio file** once Python and ffmpeg are installed.
2. **GPU acceleration** — set `WHISPER_DEVICE=cuda` for 10-30× faster transcription.
3. **Speaker diarization** — install `pyannote.audio`, accept the model licence on HF, set `HF_TOKEN` and `ENABLE_DIARIZATION=true`.
4. **Production queue** — replace `BackgroundTasks` with Celery + Redis for concurrent jobs.
5. **Auth** — add OAuth2 / API key middleware before exposing externally.
6. **UI** — the current `index.html` is a functional demo; build a proper React/Next.js frontend when ready.
