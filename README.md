# Movie Awards Video Analysis

A forensic linguistic analysis of 124 episodes of [Justin & Tyler's Movie Awards](https://www.youtube.com/@BridgewatersFinest) — a YouTube show where two childhood friends review movies. ~122 hours of transcripts turned into an interactive D3.js infographic: who talks more, who interrupts more, who's right more often, speech patterns, sentiment, rants, and a model that predicts which speaker will bring up Tyler's rabbit (Castle) next.

**Live site:** [the-movie-awards.ca](https://the-movie-awards.ca)

## Why I built this

I wanted a project that touched the full stack of a real data pipeline — ingestion, a resumable/incremental ETL step, an ML-adjacent analysis layer, and a presentation layer — against messy, real-world data (YouTube auto-captions, two people talking over each other, years of inconsistent video formats) instead of a clean Kaggle CSV.

## Pipeline

```
extract_data.py  →  diarize.py (optional)  →  analyze_data.py  →  public/data.json  →  Vercel
   (scrape +           (speaker ID              (NLP/ML on            (static D3.js
   transcripts)        via audio)               transcripts)           infographic)
```

- **`extract_data.py`** — Scrapes the channel via `yt-dlp`, pulls transcripts via `youtube-transcript-api`, stores everything in Turso (cloud SQLite). Incremental by default (only fetches new videos); `--full` forces a re-scrape.
- **`diarize.py`** — Downloads audio and runs `pyannote.audio` speaker diarization to label which lines belong to Justin vs. Tyler. GPU-accelerated when available (CPU: ~30 min/video; RTX 3060 Ti via WSL2: ~2-3 min/video).
- **`analyze_data.py`** — ~40 analysis functions over the transcripts: word frequency, filler words, superlatives, sentiment, rant detection, interruptions, monologues, agreement/disagreement, TF-IDF "fingerprints" per video, a Markov-chain quote generator (with an LLM quote generator as a swappable upgrade), a trained classifier that predicts Castle mentions, and 14 head-to-head Justin-vs-Tyler comparison stats. Outputs one `data.json` consumed by the frontend.
- **`public/index.html`** — Single-file D3.js infographic, no frontend build step, deployed as a static site on Vercel.

## What I learned

**Diarization is the hard part, not the NLP.** Separating "who said what" from a two-person audio track sounds trivial and isn't — overlapping speech, laughter, and one speaker trailing off mid-sentence all confuse pyannote's segmentation. I ended up treating diarization as a separate, optional pipeline stage so the rest of the analysis could run (in a degraded form) without it, rather than making it a hard dependency.

**Pin your ML library versions, or read the changelog before upgrading.** Upgrading to pyannote 3.x silently broke the pipeline in two ways at once: the auth kwarg was renamed (`use_auth_token` → `token`), and the pipeline's return type changed from an `Annotation` to a wrapper object (`DiarizeOutput`) that only exposes the annotation as an attribute. Neither failure was a crash — it was a `TypeError` several lines of CPU-bound processing later, which made it look like a diarization quality problem until I diffed against the library source.

**GPU passthrough through WSL2 has sharp edges that aren't in the official docs.** The biggest trap: installing an NVIDIA Linux driver *inside* WSL2 looks correct but silently breaks `nvidia-smi` and GPU passthrough, because WSL2 is supposed to use the Windows-side driver only. Wrote this up in [`GPU_DIARIZATION_SETUP.md`](GPU_DIARIZATION_SETUP.md) so I wouldn't have to re-debug it next time.

**Incremental pipelines need idempotency designed in from the start, not bolted on.** Because new episodes get added indefinitely and re-running extraction/diarization on all 124 videos is expensive (and, for diarization, slow even on GPU), both stages check Turso for existing rows before doing work. Retrofitting that after the fact would have meant a second migration pass over data I'd already processed.

**A single `data.json` file is a legitimate "backend."** With ~122 hours of source material boiled down to aggregate stats, the entire analysis output comfortably fits in one static JSON file, which means the frontend needs zero API calls, zero server, and no database at runtime — just a CDN. Simpler stack, nothing to keep alive.

## Stack

| Layer | Tool | Why |
|---|---|---|
| Extraction | `yt-dlp`, `youtube-transcript-api` | Channel scraping + caption pull |
| Storage | Turso (libSQL/SQLite) | Serverless, cheap, SQL I already know |
| Diarization | `pyannote.audio` | Pretrained speaker-diarization models, GPU-capable |
| NLP/ML | `nltk`, `scikit-learn` | Tokenization, TF-IDF, the Castle classifier |
| LLM | `google-genai` (Gemini) | Optional quote generation, falls back to a hand-rolled Markov chain on rate limits |
| Frontend | D3.js (vanilla, no framework) | Full control over a bespoke infographic layout |
| Hosting | Vercel | Static deploy, custom domain via Cloudflare |
| Testing | `pytest`, `ruff` | 71 tests across extraction and analysis logic |

## Running it locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# create a .env with:
#   TURSO_DATABASE_URL=...
#   TURSO_AUTH_TOKEN=...
#   HF_TOKEN=...        # for diarize.py (pyannote model access)
#   GOOGLE_API_KEY=...  # optional, for Gemini quote generation

make extract    # pull new videos into Turso (incremental)
make diarize    # optional: speaker-label the audio (needs ffmpeg + HF_TOKEN)
make analyze    # run analysis, write public/data.json
make serve      # preview at localhost:8080
```

See [`GPU_DIARIZATION_SETUP.md`](GPU_DIARIZATION_SETUP.md) for GPU-accelerated diarization under WSL2.

## Testing

```bash
pytest          # 71 tests, extraction + analysis logic
ruff check      # linting
```

No tests run against production data or the live Turso instance — extraction and analysis logic are tested against fixtures.
