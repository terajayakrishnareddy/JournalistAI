"""
SQLAlchemy models and database session management.
Uses SQLite by default; DATABASE_URL can point to PostgreSQL later.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker

from app.core.config import get_settings

def _now() -> datetime:
    return datetime.now(timezone.utc)

settings = get_settings()

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
    echo=settings.DEBUG,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

class Interview(Base):
    __tablename__ = "interviews"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String(36), unique=True, nullable=False, index=True)

    title = Column(String(500), nullable=False)
    interview_date = Column(String(50), nullable=True)
    language = Column(String(20), default="auto")
    category = Column(String(200), nullable=True)
    notes = Column(Text, nullable=True)
    vocabulary_hint = Column(Text, nullable=True)

    status = Column(String(50), default="waiting_for_upload", nullable=False)
    error_message = Column(Text, nullable=True)

    upload_path = Column(String(1000), nullable=True)
    audio_path = Column(String(1000), nullable=True)

    original_filename = Column(String(500), nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    media_type = Column(String(20), nullable=True)

    transcription_provider = Column(String(50), nullable=True)
    analysis_provider = Column(String(50), nullable=True)
    whisper_model = Column(String(50), nullable=True)

    diarization_run = Column(Boolean, default=False)

    created_at = Column(DateTime, default=_now)
    updated_at = Column(DateTime, default=_now, onupdate=_now)
    processing_started_at = Column(DateTime, nullable=True)
    processing_completed_at = Column(DateTime, nullable=True)

    segments = relationship("TranscriptSegment", back_populates="interview", cascade="all, delete-orphan")
    quotes = relationship("Quote", back_populates="interview", cascade="all, delete-orphan")
    claims = relationship("Claim", back_populates="interview", cascade="all, delete-orphan")
    analysis = relationship("Analysis", back_populates="interview", uselist=False, cascade="all, delete-orphan")

class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"

    id = Column(Integer, primary_key=True, index=True)
    interview_id = Column(Integer, ForeignKey("interviews.id", ondelete="CASCADE"), nullable=False, index=True)

    segment_index = Column(Integer, nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)

    original_text = Column(Text, nullable=False)

    corrected_text = Column(Text, nullable=True)

    speaker = Column(String(100), nullable=True)
    confidence = Column(Float, nullable=True)
    words_json = Column(Text, nullable=True)

    interview = relationship("Interview", back_populates="segments")

    __table_args__ = (UniqueConstraint("interview_id", "segment_index", name="uq_seg_idx"),)

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    interview_id = Column(Integer, ForeignKey("interviews.id", ondelete="CASCADE"), unique=True, nullable=False)

    summary = Column(Text, nullable=True)
    key_topics_json = Column(Text, nullable=True)
    raw_response_json = Column(Text, nullable=True)

    created_at = Column(DateTime, default=_now)

    interview = relationship("Interview", back_populates="analysis")

    @property
    def key_topics(self) -> list[str]:
        return json.loads(self.key_topics_json) if self.key_topics_json else []

class Quote(Base):
    __tablename__ = "quotes"

    id = Column(Integer, primary_key=True, index=True)
    interview_id = Column(Integer, ForeignKey("interviews.id", ondelete="CASCADE"), nullable=False, index=True)

    segment_id = Column(Integer, nullable=True)
    segment_index = Column(Integer, nullable=True)
    speaker = Column(String(100), nullable=True)
    quote_text = Column(Text, nullable=False)
    reason = Column(Text, nullable=True)

    interview = relationship("Interview", back_populates="quotes")

class Claim(Base):
    __tablename__ = "claims"

    id = Column(Integer, primary_key=True, index=True)
    interview_id = Column(Integer, ForeignKey("interviews.id", ondelete="CASCADE"), nullable=False, index=True)

    segment_id = Column(Integer, nullable=True)
    segment_index = Column(Integer, nullable=True)
    claim_text = Column(Text, nullable=False)
    category = Column(String(100), nullable=True)
    reason = Column(Text, nullable=True)
    suggested_verification_steps = Column(Text, nullable=True)
    status = Column(String(50), default="needs_verification")

    verification_notes = Column(Text, nullable=True)
    evidence_url = Column(String(1000), nullable=True)
    evidence_title = Column(String(500), nullable=True)
    evidence_snippet = Column(Text, nullable=True)
    verified_at = Column(DateTime, nullable=True)

    interview = relationship("Interview", back_populates="claims")

def get_db():
    """FastAPI dependency – yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db() -> None:
    """Create all tables (idempotent) and execute migrations for extended columns."""
    Base.metadata.create_all(bind=engine)

    with engine.connect() as conn:
        try:
            from sqlalchemy import text
            cur_cols = [row[1] for row in conn.execute(text("PRAGMA table_info(claims);")).fetchall()]
            new_cols = [
                ("verification_notes", "TEXT"),
                ("evidence_url", "VARCHAR(1000)"),
                ("evidence_title", "VARCHAR(500)"),
                ("evidence_snippet", "TEXT"),
                ("verified_at", "DATETIME"),
            ]
            for col_name, col_type in new_cols:
                if col_name not in cur_cols:
                    conn.execute(text(f"ALTER TABLE claims ADD COLUMN {col_name} {col_type};"))
                    conn.commit()
        except Exception:
            pass
