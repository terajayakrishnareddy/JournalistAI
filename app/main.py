"""
FastAPI application factory and startup.
"""
from __future__ import annotations

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

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.interviews import router as interviews_router
from app.core.config import get_settings
from app.core.database import init_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Startup and (future) shutdown logic."""

    for subdir in ("uploads",):
        (settings.STORAGE_ROOT / subdir).mkdir(parents=True, exist_ok=True)

    init_db()

    try:
        from app.core.database import SessionLocal, Interview
        with SessionLocal() as db:
            interrupted = db.query(Interview).filter(
                Interview.status.in_(["extracting_audio", "transcribing", "identifying_speakers", "extracting_insights", "queued"])
            ).all()
            for item in interrupted:
                item.status = "failed"
                item.error_message = "Processing was interrupted by server restart. Click 'Fast 20s Mode' or 'Full Re-run' to resume."
            if interrupted:
                db.commit()
                logger.info("Auto-recovered %d interrupted interview job(s).", len(interrupted))
    except Exception as exc:
        logger.warning("Job recovery check deferred: %s", exc)

    try:
        from app.providers.registry import get_transcription_provider
        provider = get_transcription_provider()
        if hasattr(provider, "_ensure_model"):
            # Run the (potentially slow) model download in a background thread
            # so the app starts accepting requests immediately. The model will
            # be ready by the time the first transcription is requested; if not,
            # transcribe() calls _ensure_model() itself.
            import threading
            threading.Thread(
                target=provider._ensure_model,
                name="whisper-prewarm",
                daemon=True,
            ).start()
    except Exception as exc:
        logger.warning("Whisper pre-warm deferred: %s", exc)

    logger.info(
        "JournalistAI started. Provider: %s | Model: %s",
        settings.TRANSCRIPTION_PROVIDER,
        settings.WHISPER_MODEL_SIZE,
    )
    yield

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description=(
            "AI-powered interview/meeting transcription and insight extraction for journalism. "
            "Transcription (Whisper) runs locally. "
            "Insight analysis (Gemini) sends transcript text to Google — see README.md."
        ),
        version="0.1.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        # Allow any origin so the frontend can be hosted on GitHub Pages (or
        # anywhere else) and still reach this FastAPI backend. Credentials are
        # disabled because the API uses no cookies/session auth.
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def _no_cache_ui(request: Request, call_next):
        response = await call_next(request)
        if request.url.path in ("/", "/index.html"):
            response.headers["Cache-Control"] = "no-store, max-age=0"
        return response

    app.include_router(interviews_router)

    from fastapi.responses import RedirectResponse
    @app.get("/docs", include_in_schema=False)
    def redirect_to_docs():
        return RedirectResponse(url="/api/docs")

    static_dir = Path(__file__).parent.parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app

app = create_app()
