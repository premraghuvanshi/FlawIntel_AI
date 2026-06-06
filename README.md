# ⬡ FlawIntel AI — Enterprise Defect Intelligence Platform

**FlawIntel** is a production-ready, 3-layer monolithic Streamlit application that ingests negative e-commerce reviews, clusters them with dynamic K-Means, and extracts structured product defect intelligence using the Groq LLM API (with a deterministic heuristic fallback).

---

## Architecture

```
flawintel/
├── app.py                    # Root orchestrator — zero business logic
├── requirements.txt
├── README.md
├── frontend/
│   ├── auth_page.py          # Login / registration UI
│   ├── workspace_page.py     # Data ingestion + pipeline trigger
│   ├── analytics_page.py     # Full 4-zone dashboard
│   └── style.css             # Industrial dark theme
├── backend/
│   ├── auth.py               # SQLite auth, sessions, history
│   ├── data_preprocessing.py # AI column mapper + filter + cap
│   ├── ml_engine.py          # MiniLM embedding + dynamic K-Means
│   ├── llm_fetch.py          # Async Groq + heuristic fallback
│   └── apify_fetch.py        # Live Apify scraper integration
└── data/
    └── storage.db            # Auto-generated at runtime
```

---

## Quick Start

### 1. Python Version

Requires **Python 3.13**. Verify:

```bash
python --version
# Python 3.13.x
```

### 2. Install Dependencies

```bash
cd flawintel
pip install -r requirements.txt
```

> **Note:** The `torch` line in `requirements.txt` installs the CPU-only wheel from PyTorch's custom index to avoid C++ compilation errors on Python 3.13.

### 3. Run the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## Configuration

All secrets are entered at runtime via the Streamlit UI (never hardcoded):

| Secret | Where to Enter | Required For |
|---|---|---|
| **Groq API Key** | Sidebar → API Credentials | LLM extraction (primary engine) |
| **Apify API Token** | Sidebar → API Credentials | Live URL scraping |

If no Groq key is supplied, the pipeline automatically falls back to the local **Heuristic Engine** (no API calls, always succeeds).

---

## Data Flow

```
CSV Upload ──┐
             ├──► AI Column Mapper (MiniLM cosine sim)
URL Fetch  ──┘         │
                       ▼
              Rating Filter (≤ 3★)
                       │
                       ▼
              Hard Cap (80 rows)
                       │
                       ▼
              MiniLM Embedding (384-D)
                       │
                       ▼
              Dynamic K-Means (k=2..8, silhouette)
                       │
                       ▼
              Groq LLM (async) ──[fail]──► Heuristic Engine
                       │
                       ▼
              SQLite Persistence
                       │
                       ▼
              Analytics Dashboard (4 Zones)
```

---

## Analytics Dashboard — 4 Zones

| Zone | Content |
|------|---------|
| **Zone 1** | Ingestion Profiling — raw rows, filtered rows, capped volume, active engine, column mapping |
| **Zone 2** | Diagnostic KPIs — clusters (k), silhouette score, pipeline latency, mapping confidence % |
| **Zone 3** | Graphical Matrices — Flaw Leaderboard, Severity Profile, Density Apportionment, Processing Funnel |
| **Zone 4** | Tabular Grid — Feature Mentioned, Sentiment Score, Specific Complaint (downloadable CSV) |

---

## CSV Format

Any CSV with at least one review-text column is accepted. The AI Column Mapper uses `all-MiniLM-L6-v2` to identify the correct columns — no hardcoded names required.

**Example schemas that work:**

```
review_body, stars          ← Amazon style
comment, rating             ← Generic
text, score                 ← API export
reviewText, overall         ← Amazon raw API
```

Reviews with ratings **> 3** are dropped. The pipeline processes a maximum of **80 rows**.

---

## Database Schema

SQLite at `data/storage.db`:

- `users` — credentials (PBKDF2-HMAC-SHA256, 260k iterations)
- `sessions` — UUID tokens, 24h TTL, auto-purge on validation
- `analysis_history` — full run metadata per user
- `extracted_complaints` — per-cluster extraction results

---

## Heuristic Fallback Engine

When Groq is unavailable:

| Field | Logic |
|---|---|
| `Feature_Mentioned` | Most frequent non-stopword token across cluster reviews |
| `Specific_Complaint` | Longest individual review string (≤ 300 chars) |
| `Sentiment_Score` | Linear interpolation across `[-1.0, -0.3]` based on cluster position |

---

## Dependency Notes

- `torch==2.6.0+cpu` — CPU-only build, no CUDA required, no C++ compilation
- `sentence-transformers==3.0.0` — requires `torch` pre-installed
- `nest-asyncio==1.6.0` — patches the Streamlit event loop for `asyncio.run()`
- `groq==0.5.0` — official Groq Python SDK with async support

---

## License

Internal enterprise tooling. All rights reserved.
