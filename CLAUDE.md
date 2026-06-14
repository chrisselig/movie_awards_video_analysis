## Project Description

Forensic linguistic analysis of the Justin & Tyler Movie Awards YouTube channel (@BridgewatersFinest). Extracts transcripts, runs NLP analysis (word frequency, filler words, superlatives, vocabulary richness, hot takes, agreement/disagreement), and presents results as an interactive D3.js infographic deployed on Vercel.

**Architecture:**
- `extract_data.py` — Scrapes YouTube channel, downloads transcripts, stores in Turso (incremental)
- `diarize.py` — Speaker diarization via pyannote.audio (requires ffmpeg + HF_TOKEN + GPU recommended)
- `analyze_data.py` — Runs all NLP analysis, outputs `public/data.json`
- `public/index.html` — D3.js interactive infographic (deployed to Vercel)
- Data stored in Turso (cloud SQLite)

**Pipeline:** `extract_data.py` -> `diarize.py` (optional) -> `analyze_data.py` -> deploy

## General Rules

- When the user asks you to create specific files (CLAUDE.md, skills, configs), create them in the same session — do not defer or forget. Treat explicit file creation requests as hard requirements.
- When the user asks "how do I access X" or "where is X deployed", give the direct URL or command first, then explain details only if asked.
- Prefer editing existing files over creating new ones.
- Keep functions small and single-purpose. If a function does more than one thing, split it.
- Use type hints on public function signatures. Skip them on trivial helpers.
- All database schema changes must be backwards-compatible — use ALTER TABLE with try/except, never DROP TABLE in production.

## Testing & Linting

This project uses Python (primary), HTML/JS (frontend), and YAML (configs). Always run `ruff check` and `pytest` after Python changes.

Testing is mandatory. Every new feature, bug fix, or refactor must include tests. Do not skip tests or defer them. CI must pass before merging.

### General Testing Rules

1. **Tests must run in CI** — if it's not in the CI pipeline, it doesn't count
2. **Use descriptive test names** — name should explain the scenario and expected outcome
3. **One assertion per concept** — test one behavior per test function
4. **Test boundaries** — empty inputs, nulls, max values, negative numbers
5. **Never test against production data** — always use fixtures or in-memory DBs
6. **Tests must be independent** — no shared mutable state, no ordering dependencies
7. **Run tests before pushing** — `pytest` locally before `git push`

## Code Style

- Python: Follow PEP 8, enforced by `ruff`. Max line length 120.
- Use f-strings for string formatting (not .format() or %).
- Imports: stdlib first, then third-party, then local. One import per line for third-party.
- Constants at module level in UPPER_SNAKE_CASE.
- Use `log()` helper for print statements (ensures flush=True for pipeline scripts).

## Data & Database

- **Turso** (cloud SQLite) is the single source of truth for video metadata and transcripts.
- Schema changes: Always add migration logic (ALTER TABLE with try/except) in create_tables functions. Never assume a clean DB.
- The `extract_data.py` pipeline is **incremental by default** — it only fetches new videos. Use `--full` flag to re-process everything.
- Never commit `.env` files or auth tokens.

## Deployment

- Frontend is a static HTML file with D3.js — deployed to Vercel via `public/` directory.
- `vercel.json` configures the deployment.
- `data.json` must be regenerated (`python analyze_data.py`) before each deploy.
- The analysis pipeline runs on-demand (not scheduled) since videos aren't uploaded on a schedule.

## CI/CD

When modifying CI/CD configs, always verify all dependencies (ruff, pytest, etc.) are included in requirements files and that build-time env vars are handled gracefully with fallbacks.

## Security

- Never commit credentials, API tokens, or `.env` files.
- Turso auth tokens and HF tokens go in `.env` only.
- Sanitize any user-facing data in the frontend (transcript text displayed in tooltips).

## Git Workflow

- Commit messages: imperative mood, explain the "why" not the "what".
- Run `pytest` and `ruff check` before every commit.
- Don't commit generated files (`data.json`, `audio/`, `venv/`).
