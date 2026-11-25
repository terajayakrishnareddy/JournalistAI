# JournalistAI — Deep Learning / NLP Project Evaluation Report

> **Project:** AI Interview & Meeting Transcription with Insight Extraction for Journalism
> **Stack:** Python 3.11+, FastAPI, SQLAlchemy (SQLite), faster-whisper (Whisper large-v3), Google Gemini 2.5 Pro, pyannote.audio, ffmpeg
> **Report purpose:** Self-evaluation against the 30-mark Deep Learning / NLP rubric + viva preparation guide.
> **Max marks:** 30

---

## 0. Project Summary

| # | Rubric Component | Marks | Self-Score | Status |
|---|---|---|---|---|
| 1 | Problem Understanding | 2 | 2 | Full |
| 2 | Data Collection & Preprocessing | 4 | 3.5 | Partial (no stemming/lemmatization — not required for ASR+LLM pipeline) |
| 3 | Model Design & Implementation | 5 | 5 | Full (Transformer/Whisper + LLM + Diarization) |
| 4 | Training & Performance Evaluation | 3 | 2.5 | Partial (uses pretrained models; metrics are application-side, not benchmark) |
| 5 | Results & Analysis | 3 | 2.5 | Partial |
| 6 | Innovation & Enhancement | 2 | 2 | Full |
| 7 | Presentation | 5 | 4.5 | Partial |
| 8 | Viva Voce | 6 | (examiner) | — |
| | **Total** | **30** | **~19.5 / 24** (excluding viva) | |

---

## 1. Problem Understanding (2 / 2)

### 1.1 The NLP Problem

The project addresses two connected NLP tasks:

1. **Automatic Speech Recognition (ASR)** — converting spoken audio/video into time-synchronized text.
2. **Natural Language Understanding (NLU)** — extracting journalistically useful structure from the transcript:
   - Candidate **quotations** (verbatim, ready to drop into an article).
   - Factual **claims** that need independent verification (fact-checking).
   - A short **summary** and **key topics**.

### 1.2 Objectives

- Transcribe interviews/meetings with word/segment timestamps and speaker labels.
- Surface quotable passages that are **guaranteed verbatim** (code-level validation).
- Flag verifiable claims and attach a **verification workflow** (never let the AI decide true/false).
- Provide a clean exportable dossier for journalists.

### 1.3 Domain Knowledge

- Investigative journalism workflows: source protection, verbatim quotes, fact-checking, attribution.
- Speech processing: 16 kHz mono WAV is the standard input format for most ASR models.
- Ethics: AI must not assert truth; it only surfaces claims *to be verified by a human*.

**Evidence:** [README.md](file:///c:/Users/ayyap/Downloads/Pro471/README.md), app description in [main.py](file:///c:/Users/ayyap/Downloads/Pro471/app/main.py#L68-L72).

---

## 2. Data Collection & Preprocessing (3.5 / 4)

### 2.1 Data Collection

- **Source:** User-uploaded audio/video files (interviews, meetings) — real-world, not a static benchmark corpus.
- **Supported formats:** `.mp3, .wav, .m4a, .flac, .mp4, .mkv, .mov, .webm, .ogg, .aac`.
- **Validation:** Real content validated with `ffprobe` (not just extension) — [media_validator.py](file:///c:/Users/ayyap/Downloads/Pro471/app/core/media_validator.py#L80-L151).
- **Limits:** max 2 GB file, max 3 h duration.

### 2.2 Audio Preprocessing (Signal Preprocessing)

- Extraction from video → audio → **16 kHz mono WAV (PCM s16le)** — the canonical input for Whisper:

```ffmpeg
-i input -vn -ac 1 -ar 16000 -acodec pcm_s16le audio_16k_mono.wav
```

**Code:** [audio_processor.py](file:///c:/Users/ayyap/Downloads/Pro471/app/core/audio_processor.py#L40-L49)

### 2.3 Text Preprocessing

| Step | Where | Notes |
|---|---|---|
| Tokenization | Whisper internal (BPE, byte-pair encoding) | faster-whisper handles it; 99 languages |
| Stopword removal | [local_analysis_provider.py](file:///c:/Users/ayyap/Downloads/Pro471/app/providers/local_analysis_provider.py#L22-L37) | ~130+ English stopwords + filler words (um, uh, like) for topic extraction |
| Regex pattern cleaning | [local_analysis_provider.py](file:///c:/Users/ayyap/Downloads/Pro471/app/providers/local_analysis_provider.py#L39-L45) | Percentages, years, financial terms, quantities for claim detection |
| Filename sanitization | [interviews.py](file:///c:/Users/ayyap/Downloads/Pro471/app/api/interviews.py#L72-L77) | NFKD normalization + regex sanitization |
| Verbatim preservation | [pipeline.py](file:///c:/Users/ayyap/Downloads/Pro471/app/core/pipeline.py#L53-L68) | Quotes validated character-for-character against transcript |

### 2.4 Embeddings

- Whisper uses its own learned audio + text embeddings (internal).
- Gemini uses its internal transformer embeddings.
- No manual word2vec/GloVe needed — the models handle representation internally.

### 2.5 What is NOT done (and why)

- **Stemming / lemmatization:** Not applied. Rationale: the output is a *verbatim transcript* — stemming would destroy the exact wording journalists need. Topic extraction uses stopword removal + frequency, which is sufficient for this use case.
- **Static dataset preparation:** The app processes live user uploads rather than a fixed corpus; there is no train/val/test split because no from-scratch training is performed.

**Self-score justification:** 3.5/4 — strong audio + text preprocessing pipeline; deducted for no explicit stemming/lemmatization and no labeled dataset (both justified by the application nature).

---

## 3. Model Design & Implementation (5 / 5)

### 3.1 Models Used

| Model | Type | Task | Location |
|---|---|---|---|
| **OpenAI Whisper (large-v3)** | Transformer encoder-decoder ASR | Speech → text | [whisper_provider.py](file:///c:/Users/ayyap/Downloads/Pro471/app/providers/whisper_provider.py) |
| **Google Gemini 2.5 Pro** | Transformer-decoder LLM | Quote/claim/summary extraction | [gemini_provider.py](file:///c:/Users/ayyap/Downloads/Pro471/app/providers/gemini_provider.py) |
| **pyannote.audio 3.1** | Pretrained speaker diarization | Speaker attribution | [pipeline.py](file:///c:/Users/ayyap/Downloads/Pro471/app/core/pipeline.py#L160-L235) |
| **Local rule-based engine** | Regex + heuristics | Offline fallback analysis | [local_analysis_provider.py](file:///c:/Users/ayyap/Downloads/Pro471/app/providers/local_analysis_provider.py) |

### 3.2 Whisper Architecture (why it's a Transformer)

- **Encoder:** 6-layer audio transformer (log-Mel spectrogram → audio embeddings).
- **Decoder:** 6-layer text transformer (autoregressive token generation).
- Supports **99 languages**, trained on 680k hours of labeled audio.
- Uses **CTranslate2** runtime (faster-whisper) for CPU inference with `int8` quantization.
- **VAD (Voice Activity Detection)** filter removes non-speech segments.
- **Word-level timestamps** are available (optional, 2-3x slower).

### 3.3 Implementation Highlights

**Parallel transcription** — audio >60s is split into N chunks processed by N workers in parallel:

```python
chunk_sec = target_duration / num_workers
clips = [f"{i*chunk_sec:.1f},{(i+1)*chunk_sec:.1f}" for i in range(num_workers)]
# ThreadPoolExecutor processes all clips concurrently
```
[whisper_provider.py L174-L211](file:///c:/Users/ayyap/Downloads/Pro471/app/providers/whisper_provider.py#L174-L211)

**Confidence score** — derived from the model's `avg_logprob` converted to probability:

```python
confidence = round(math.exp(seg.avg_logprob), 4)
```
[whisper_provider.py L145-L148](file:///c:/Users/ayyap/Downloads/Pro471/app/providers/whisper_provider.py#L145-L148)

### 3.4 Provider-Based Architecture (Design Pattern)

Abstract base classes define interfaces so backends are swappable:

```
TranscriptionProvider (ABC) ──► WhisperProvider
AnalysisProvider (ABC)      ──► GeminiProvider
                           └──► LocalAnalysisProvider (fallback)
```

**Code:** [base.py](file:///c:/Users/ayyap/Downloads/Pro471/app/providers/base.py), [registry.py](file:///c:/Users/ayyap/Downloads/Pro471/app/providers/registry.py)

### 3.5 Configuration

| Setting | Default | Purpose |
|---|---|---|
| `WHISPER_MODEL_SIZE` | `large-v3` | Model size (tiny → large-v3) |
| `WHISPER_DEVICE` | `cpu` | cpu / cuda |
| `WHISPER_COMPUTE_TYPE` | `int8` | Quantization for CPU |
| `WHISPER_BEAM_SIZE` | `1` | Greedy decoding (fastest) |
| `WHISPER_WORD_TIMESTAMPS` | `false` | Per-word timestamps |

[config.py](file:///c:/Users/ayyap/Downloads/Pro471/app/core/config.py)

**Self-score justification:** 5/5 — uses Transformer (Whisper), LLM (Gemini), pretrained diarization, with a clean abstract-provider design and parallel optimization.

---

## 4. Training & Performance Evaluation (2.5 / 3)

### 4.1 Training Strategy: Transfer Learning

- **No from-scratch training.** The project uses **pretrained models**:
- Whisper large-v3 (pretrained on 680k hours) — used as-is.
- Gemini 2.5 Pro (Google's pretrained LLM) — prompted for analysis.
- pyannote 3.1 (pretrained diarization) — used as-is.

This is a deliberate **transfer learning / fine-tuning-free** approach — appropriate for an application project where labeled training data and compute are unavailable.

### 4.2 Metrics Implemented

The `/api/interviews/{id}/evaluation endpoint returns:

| Metric | Formula | Meaning |
|---|---|---|
| **Real-Time Factor (RTF)** | `processing_time / audio_duration` | <1.0 = faster than real-time |
| **Human Correction Rate** | `edited_segments / total_segments × 100` | % of segments a human had to fix |
| **Claim Verification Rate** | `verified_claims / total_claims × 100` | % of claims resolved |
| **Total words / segments** | count | Throughput |
| **Quotes extracted** | count | NLU yield |
| **Claims extracted** | count | NLU yield |
| **Confidence** | `exp(avg_logprob)` | Per-segment model confidence |

**Code:** [interviews.py L770-L809](file:///c:/Users/ayyap/Downloads/Pro471/app/api/interviews.py#L770-L809)

### 4.3 Metrics NOT computed (and why)

- **Accuracy / Precision / Recall / F1:** Require a labeled ground-truth dataset. The app processes unlabeled live audio; no reference transcript corpus is bundled. These would require building a labeled set and running WER/CER.
- **BLEU / ROUGE:** For translation/summarization benchmarks — not benchmarked against reference because there is no reference summary corpus.
- **Perplexity:** Language-model metric — not applicable to this ASR+extraction pipeline.

**Self-score justification:** 2.5/3 — strong application metrics (RTF, confidence, correction rate) are implemented and meaningful; full benchmark-style metrics are absent due to no labeled benchmark dataset.

---

## 5. Results & Analysis (2.5 / 3)

### 5.1 Error Analysis Mechanisms

1. **Verbatim quote validation** — any quote not found character-for-character in the transcript is **dropped before DB. This is a hard guarantee against hallucinated quotes.

[code: [pipeline.py L53-L68](file:///c:/Users/ayyap/Downloads/Pro471/app/core/pipeline.py#L53-L68)

2. **Claim status hard-locked to `needs_verification`** — the AI is forbidden from labeling anything true/false. The human journalist decides.

3. **Graceful degradation** — if Gemini key missing → falls back to local rule-based engine automatically.

[registry.py L33-L41](file:///c:/Users/ayyap/Downloads/Pro471/app/providers/registry.py#L33-L41)

4. **Diarization graceful skip** — if pyannote/HF token absent, diarization is skipped without failing the pipeline.

5. **Interrupted job recovery** — on startup, any jobs stuck in processing states are marked failed with a resume message.

[main.py L36-L46](file:///c:/Users/ayyap/Downloads/Pro471/app/main.py#L36-L46)

### 5.2 Visualizations

- **Frontend dashboard** — live status polling, segment list, quotes, claims.
- **Printable HTML dossier** — `/api/interviews/{id}/dossier generates a styled fact-check report with summary, quotes, claims, and full transcript.

[interviews.py L665-L768](file:///c:/Users/ayyap/Downloads/Pro471/app/api/interviews.py#L665-L768)

- **Per-segment confidence** — stored per segment in DB column.

### 5.3 Comparison with Baseline

| Aspect | Gemini Provider | Local Provider |
|---|---|---|
| Quote quality | High (LLM understands context) | Rule-based (opinion markers) |
| Summary quality | High (coherent prose) | Template + frequency |
| Latency | API round-trip | Instant, local |
| Privacy | Sends text to Google | 100% local |
| Claim detection | Semantic | Regex (numbers/%) |

### 5.4 What is partial

No formal quantitative comparison table (e.g., quote precision on a test set) is included — comparison is qualitative/architectural.

**Self-score justification:** 2.5/3 — strong error safeguards and visualizations; formal benchmark comparison absent.

---

## 6. Innovation & Enhancement (2 / 2)

### 6.1 Transfer Learning

- Whisper large-v3 (pretrained ASR) used directly — state-of-the-art multilingual ASR without training.
- Gemini 2.5 Pro (pretrained LLM) — transfer learning via prompting.

### 6.2 Novel Approach

1. **Verbatim quote guarantee** — code-level validation that no non-verbatim quotes are never stored. Most LLM outputs are never stored.
2. **Claim verification workflow** — AI flags claims, human verifies via Wikipedia evidence search.

[interviews.py L356-L428](file:///c:/Users/ayyap/Downloads/Pro471/app/api/interviews.py#L356-L428)

3. **Parallel chunked parallel transcription** — splits audio across CPU cores for speedup.
4. **Provider abstraction** — clean swap backends without changing pipeline code.
5. **Original vs corrected transcript separation** — `original_text` and `corrected_text` columns keep the machine transcript intact while allowing human edits.

### 6.3 Deployment

- **FastAPI + Uvicorn web server.
- Targeted for **Railway** deployment.
- SQLite with persistent storage for DB + model cache.
- 23 REST endpoints + Swagger docs.
- Static frontend served by FastAPI.

**Self-score justification:** 2/2 — transfer learning, deployment-ready, novel journalism-specific safeguards.

---

## 7. Presentation (4.5 / 5)

| Deliverable | Status |
|---|---|
| README.md with full documentation | ✅ Complete |
| Swagger API docs (`/api/docs`) | ✅ Interactive |
| Web frontend (`static/index.html`) | ✅ Functional |
| Printable dossier report | ✅ |
| Export formats (TXT/JSON/CSV/SRT/VTT) | ✅ 5 formats |
| `.env.example` + `.gitignore` | ✅ |
| Slides / demo video | ⚠️ To prepare separately |

**Self-score justification:** 4.5/5 — strong written + interactive deliverable code-level documentation; slides/demo video prepared at presentation time.

---

## 8. Viva Voce — Preparation (6 marks)

### 8.1 Core Concepts to Master

**A. **Why 16 kHz mono WAV?**
Whisper's mel-spectrogram input expects 16 kHz. Mono avoids stereo channel confusion. PCM s16le is uncompressed and fast to decode.

**B. **What is `avg_logprob` and how is confidence computed?**
Each segment has `avg_logprob` = average log probability of the generated tokens. `confidence = exp(avg_logprob)` converts it to a 0–1 probability. Higher = model more certain.

**C. **Why beam_size=1?**
Beam size 1 = greedy decoding (pick highest-probability token each step). Fastest. Larger beam (5) gives better quality but 5x slower. Trade-off chosen for CPU.

**D. **What is VAD filter?**
Voice Activity Detection removes silence/non-speech before transcription. Reduces hallucinations on silence and speeds up inference.

**E. **Why `int8` quantization?**
Converts model weights from float32 to int8. ~4x smaller, ~2-4x faster on CPU with minimal accuracy loss.

**F. **What is a Transformer?**
A neural network architecture using self-attention. Whisper encoder processes audio; Gemini generates text tokens. Replaced RNN/LSTM due to parallelization and long-range context.

**G. **What is transfer learning here?**
Using pretrained Whisper/Gemini instead of training from scratch. Saves massive data + compute. Fine-tuning is possible but not done (insufficient labeled data.

**H. **Why no stemming/lemmatization?**
Output must be verbatim — stemming would alter exact journalist quotes. Topic extraction uses stopword removal + term frequency which is enough.

**I. **How does quote validation works?**
After Gemini returns quotes, pipeline concatenates all segment texts and checks `quote_text in full_transcript`. Non-matching quotes are logged and dropped.

**J. **Why claims are always `needs_verification`?**
AI cannot reliably determine truth. The system's job is to surface claims for human journalists to verify, not to assert truth. Ethical + avoids misinformation.

**K. **What is the provider pattern?**
Abstract base classes (`TranscriptionProvider`, `AnalysisProvider`) define interfaces. Concrete implementations (Whisper, Gemini, Local) are swappable via `registry.py` factory. New backends can be added without changing pipeline code.

**L. **How does parallel transcription work?**
For audio >60s with >1 worker: split duration into N chunks, each worker transcribes its clip via `clip_timestamps`, results merged and sorted by start time.

**M. **What is RTF (Real-Time Factor)?**
`RTF = processing time / audio duration. RTF < 1 means transcription finishes before audio would have taken to play. E.g., RTF 0.5 = 2x faster than real-time.

**N. **Database models and tables?**
`interviews`, `transcript_segments`, `analyses`, `quotes`, `claims`. SQLAlchemy ORM. SQLite by default, PostgreSQL-ready.

**O. **How does diarization work?**
pyannote.audio 3.1 assigns speaker labels to time ranges. Each transcript segment is assigned the speaker with the highest temporal overlap.

**P. **What is BLEU/ROUGE and why not used?**
BLEU (translation), ROUGE (summarization) — reference-based metrics requiring a gold-standard corpus. Not used because no labeled reference dataset is bundled.

**Q. **Privacy considerations?**
Whisper runs 100% locally. Gemini only when key is set. Wikipedia only when user clicks "search evidence".

### 8.2 Likely Examiner Questions & Answers

**Q1: Why did you choose Whisper over a custom LSTM/GRU?**
A: Whisper is a Transformer trained on 680k hours across 99 languages. A custom LSTM would need massive labeled data and compute I don't have. Whisper outperforms RNNs on long audio due to self-attention capturing long-range dependencies. Transfer learning.

**Q2: What is the difference between your local and Gemini analysis?**
A: Gemini is an LLM that understands semantic meaning — high quality but sends text to Google. Local is regex/heuristic — instant, private, lower quality. Used as fallback when no API key.

**Q3: How do you prevent the AI from inventing quotes?**
A: Two-layer safeguard: (1) system prompt instructs verbatim copy; (2) code validates each quote against the transcript string match. Non-verbatim dropped.

**Q4: What metrics did you evaluate?**
A: RTF, per-segment confidence (exp(avg_logprob)), human correction rate, claim verification rate, word/segment counts.

**Q5: Why is this an NLP project?**
A: ASR (speech-to-text is NLP. Plus NLU: quote extraction, claim detection, summarization, topic modeling.

**Q6: What would you improve?**
A: Add a labeled test set to compute WER/CER, BLEU/ROUGE; fine-tune Whisper on domain audio; add Celery/Redis queue; GPU acceleration; authentication.

---

## 9. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     User Upload (audio/video)                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 1. VALIDATE  (ffprobe)  → media type, duration, size)      │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. EXTRACT AUDIO  (ffmpeg)  → 16 kHz mono WAV)          │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. TRANSCRIBE  (faster-whisper large-v3, Transformer)      │
│    - VAD filter, int8 quant, parallel workers                    │
│    - Output: segments with timestamps + confidence              │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. DIARIZE (optional)  (pyannote.audio 3.1)                │
│    - Assigns speaker labels to segments                         │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. EXTRACT INSIGHTS  (Gemini 2.5 Pro LLM)                 │
│    - Summary, key topics, quotes, claims                      │
│    - Verbatim quote validation + claim status lock          │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 6. PERSIST  (SQLite / SQLAlchemy)                            │
│    - interviews, segments, quotes, claims, analysis               │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│ 7. EXPORT / DOSSIER / EVIDENCE SEARCH (Wikipedia API)        │
└──────────────────────────────────────────────────────────────┘
```

## 10. File Reference Map

| File | Purpose |
|---|---|
| [app/main.py](file:///c:/Users/ayyap/Downloads/Pro471/app/main.py) | FastAPI app factory, startup, CORS |
| [app/api/interviews.py](file:///c:/Users/ayyap/Downloads/Pro471/app/api/interviews.py) | 23 REST endpoints |
| [app/core/pipeline.py](file:///c:/Users/ayyap/Downloads/Pro471/app/core/pipeline.py) | 6-stage processing orchestrator |
| [app/core/config.py](file:///c:/Users/ayyap/Downloads/Pro471/app/core/config.py) | Environment settings |
| [app/core/database.py](file:///c:/Users/ayyap/Downloads/Pro471/app/core/database.py) | SQLAlchemy models |
| [app/core/audio_processor.py](file:///c:/Users/ayyap/Downloads/Pro471/app/core/audio_processor.py) | ffmpeg 16kHz mono conversion |
| [app/core/media_validator.py](file:///c:/Users/ayyap/Downloads/Pro471/app/core/media_validator.py) | ffprobe validation |
| [app/providers/base.py](file:///c:/Users/ayyap/Downloads/Pro471/app/providers/base.py) | Abstract provider interfaces |
| [app/providers/whisper_provider.py](file:///c:/Users/ayyap/Downloads/Pro471/app/providers/whisper_provider.py) | faster-whisper ASR |
| [app/providers/gemini_provider.py](file:///c:/Users/ayyap/Downloads/Pro471/app/providers/gemini_provider.py) | Gemini LLM analysis |
| [app/providers/local_analysis_provider.py](file:///c:/Users/ayyap/Downloads/Pro471/app/providers/local_analysis_provider.py) | Offline fallback |
| [app/providers/registry.py](file:///c:/Users/ayyap/Downloads/Pro471/app/providers/registry.py) | Provider factory |
| [requirements.txt](file:///c:/Users/ayyap/Downloads/Pro471/requirements.txt) | Dependencies |
| [README.md](file:///c:/Users/ayyap/Downloads/Pro471/README.md) | Documentation |

---

## 11. Gaps & Improvement Plan (for viva honesty)

| Gap | Why | Fix |
|---|---|---|
| No labeled test dataset for WER/BLEU | Application project, no corpus | Record 5-10 interviews, transcribe manually, compute WER |
| No from-scratch model training | Insufficient data/compute | Fine-tune Whisper on domain audio |
| No GPU in default config | CPU-only setup | Set `WHISPER_DEVICE=cuda` |
| No Celery queue | BackgroundTasks suffice for single-user | Add Celery+Redis |
| No auth | Single-user local tool | Add OAuth2/API keys |
| Diarization untested | pyannote needs HF license | Install + test |

---

*Report generated for viva preparation. Keep this file locally; it is an academic artifact and should be excluded from the production GitHub repository (add to `.gitignore` if pushing to the public repo).*
