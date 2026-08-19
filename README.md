# LawAgent

An AI-powered legal research pipeline that automatically extracts keywords from a case summary, scrapes relevant case law, and generates parallel prosecution and defence arguments backed by semantic vector search.

## Architecture

```
case summary (text)
      │
      ▼
KeywordExtractorAgent   ← gemini-2.0-flash — extracts one legal keyword
      │
      ▼
DocumentFetcherAgent    ← gemini-2.0-flash — triggers scraper via getDocs tool
      │  (KanoonSpider scrapes Indian Kanoon, chunks & embeds into Qdrant)
      ▼
┌─────────────────────────────────┐
│       ParallelVectorResearch    │
│  Prosecution_Research │ Defence_Research  ← gemini-2.5-pro each
│  (fetches from Qdrant, builds arguments)  │
└─────────────────────────────────┘
      │
      ▼
Results saved to Google Cloud Storage (optional)
```

**Key files:**
- [agent.py](agent.py) — agent definitions and pipeline entrypoint
- [utils0.py](utils0.py) — Scrapy-based web scraper (`KanoonSpider`) + Qdrant upsert
- [utils1.py](utils1.py) — HTML stripping, sentence-aware chunking, embeddings, Qdrant search
- [instructions/](instructions/) — prompt instructions for prosecution and defence agents

## Setup

### Prerequisites

- Python 3.10+
- A Google API key with Gemini access
- (Optional) Google Cloud Storage bucket for saving results

### Installation

```bash
pip install -r requirements_fixed.txt
```

> The `scrapper` module (containing `KanoonSpider`) must be present in the project root. It is a custom Scrapy spider not included in this repo.

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Gemini API key |
| `GCS_BUCKET_NAME` | No | GCS bucket for saving research output; skipped if unset |
| `MODEL_FAST` | No | Model for keyword/doc-fetch agents (default: `gemini-2.0-flash`) |
| `MODEL_RESEARCH` | No | Model for prosecution/defence agents (default: `gemini-2.5-pro`) |

```bash
export GOOGLE_API_KEY=your-key-here
export GCS_BUCKET_NAME=your-bucket   # optional
```

### Local case files

Place `.txt` or `.html` case files in `case_files/` before running. These are embedded into the `Usercase_pdf` Qdrant collection on first query and searched alongside scraped case law.

## Usage

```python
import asyncio
from agent import run_pipeline

asyncio.run(run_pipeline("The accused is charged with criminal breach of trust under Section 406 IPC."))
```

Or run directly:

```bash
python agent.py
```

## Data storage

- **`qdrant_db/`** — local Qdrant vector database (auto-created)
  - `case_pdf` collection — scraped case law from Indian Kanoon
  - `Usercase_pdf` collection — user-supplied case files from `case_files/`
- **GCS** (if configured) — timestamped research output at `raw_texts/YYYYMMDDTHHMMSS/`

## Notes

- Re-running the scraper for the same keyword is idempotent: deterministic UUIDs (`uuid5`) prevent duplicate chunks in Qdrant.
- The scraper runs in a subprocess with a 300-second timeout to keep the ADK event loop clean.
- Planning documents for past fix and optimisation passes are in [docs/](docs/).
