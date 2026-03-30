# NUST Admission Helper Chatbot

An AI-powered chatbot that helps prospective students navigate NUST (National University of Sciences and Technology) admissions — fully offline, privacy-first, running locally on your machine.

Built for the **NUST Local Chatbot Competition 2026**.

---

## Features

- **AI Chat** — Conversational Q&A about NUST admissions powered by a local LLM (phi3:mini via Ollama)
- **Merit Calculator** — Calculates your aggregate using the official NUST formula and lists eligible programs with margin/status
- **Program Browser** — All 26 NUST undergraduate programs with school, duration, seats, and cutoff aggregates
- **RAG Pipeline** — Answers backed by a curated knowledge base stored in ChromaDB; scraped web content supplements it
- **Hallucination-Free Cutoffs** — Merit cutoffs are stored in a permanent `data/cutoffs.json` and injected directly into every LLM prompt — scraped data can never override them
- **Live Scraper** — Fetches fresh content from nust.edu.pk and ugadmissions.nust.edu.pk on demand
- **Auto-Refresh Scheduler** — Background thread checks for website changes every 30 days using content hashing
- **100% Local** — No internet required for AI inference; all embeddings run on-device

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Frontend | HTML5, CSS3, JavaScript (vanilla) |
| Backend | FastAPI + Uvicorn |
| LLM | Ollama (phi3:mini or any compatible model) |
| Vector DB | ChromaDB (persistent local storage) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Scraping | requests + BeautifulSoup4 |
| Async HTTP | httpx |
| Scheduling | threading (daemon background thread) |
| Notifications | plyer (Windows desktop notifications) |

---

## Prerequisites

1. **Python 3.11+** — Download from [python.org](https://python.org)
2. **Ollama** — Download from [ollama.com](https://ollama.com)
3. **phi3:mini model** — Pulled via Ollama (~2.4 GB)

---

## Setup

### Step 1: Install Ollama

Download and install from [ollama.com/download](https://ollama.com/download), then start it:

```bash
ollama serve
```

### Step 2: Pull the AI Model

```bash
ollama pull phi3:mini
```

If you have limited RAM, a lighter alternative:

```bash
ollama pull llama3.2:1b   # ~1.3 GB
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

First-time install downloads PyTorch and sentence-transformers — allow 5–10 minutes.

### Step 4: Run

```bash
python main.py
```

On startup, the app will:
1. Verify Ollama is running and auto-detect the best available model
2. Load `data/nust_knowledge.json` into ChromaDB (first run: ~2–3 min for embedding download)
3. Start the FastAPI server on port 8000
4. Open your browser at `http://127.0.0.1:8000`

---

## Usage

### Chat

Type any admissions question and press Enter. Example questions:

- "What is the cutoff for BS Computer Science?"
- "Am I eligible for SEECS? My aggregate is 78%"
- "Calculate my aggregate: NET 165, FSc 1000/1100, Matric 950/1100"
- "What NET score do I need for BS AI?"
- "When is NET-3 2026?"
- "How should I prepare for NET?"

### Merit Calculator

Click **Merit Calculator**, enter your NET score, FSc marks/total, and Matric marks/total. The calculator returns:

- Your aggregate percentage (with per-component breakdown)
- Programs you are **Likely** eligible for (aggregate ≥ cutoff + 2%)
- Programs where you are **Borderline** (within 2% of cutoff)
- Programs just out of reach (within 5% below cutoff)
- Contextual advice based on your scores

### Program Browser

Click **Programs** to see all 26 undergraduate programs with school, duration, seats, and closing merit cutoff.

### Refresh Data

Click **Refresh** to scrape 8 NUST pages for the latest information (NET schedule, fees, eligibility, etc.) and re-embed into ChromaDB.

> **Note:** Refresh only updates general knowledge. Merit cutoffs are never affected by refresh — they are permanently fixed in `data/cutoffs.json`.

---

## Merit Formula

```
Aggregate = (NET / 200 × 75) + (FSc / FSc_Total × 15) + (Matric / Matric_Total × 10)
```

| Component | Weight | Max Contribution |
|-----------|--------|-----------------|
| NET (out of 200) | 75% | 75 points |
| FSc / Intermediate | 15% | 15 points |
| Matric / SSC | 10% | 10 points |

**Important:**
- There is **no Hafiz bonus** and no extracurricular bonus at NUST. Admission is purely aggregate-based.
- SAT score ≥ 1100 is accepted instead of NET for eligible applicants.
- O/A-Level students must obtain an IBCC equivalence certificate first.

---

## How Cutoffs Work (No Hallucination)

All 26 program cutoffs are stored in `data/cutoffs.json` — the single source of truth. This file is **never modified** by the scraper or the refresh button.

Every time the LLM answers a cutoff question, the correct numbers are injected directly into its system prompt with a hard override instruction. Additionally, the RAG retriever automatically filters out any scraped chunks that contain merit/cutoff figures, so stale web data can never contradict the authoritative values.

To update cutoffs for a new admissions year, edit only `data/cutoffs.json`.

---

## Project Structure

```
NUST admission helper/
├── main.py                      # Entry point — startup checks + Uvicorn server
├── index.html                   # Frontend UI
├── requirements.txt
├── README.md
│
├── ui/
│   └── app.py                   # FastAPI app — all API routes, merit calculation logic
│
├── chatbot/
│   ├── llm.py                   # Async Ollama connector (streaming + health check)
│   └── chain.py                 # RAG + LLM orchestrator, system prompt, canned replies
│
├── rag/
│   ├── embedder.py              # ChromaDB manager (upsert, search, clear)
│   └── retriever.py             # RAG retriever — confidence scoring + cutoff chunk filter
│
├── scraper/
│   ├── scrape.py                # Scrapes 8 NUST pages → data/raw/
│   └── cleaner.py               # Cleans + chunks raw text → data/cleaned/
│
├── scheduler/
│   └── update_job.py            # 30-day background scheduler with content-hash change detection
│
└── data/
    ├── cutoffs.json             # PERMANENT merit cutoffs — never overwritten by scraper
    ├── nust_knowledge.json      # Curated static knowledge base (loaded into ChromaDB on startup)
    ├── raw/                     # Raw scraped JSON files
    ├── cleaned/                 # Chunked text ready for embedding
    ├── content_hash.json        # Hash of last scraped content (used by scheduler)
    ├── app.log
    └── scraper.log
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serve index.html |
| GET | `/api/health` | Ollama status, RAG readiness, scraper status |
| POST | `/api/chat` | SSE streaming chat |
| GET | `/api/programs` | List all 26 programs with cutoffs |
| POST | `/api/merit` | Calculate aggregate and eligible programs |
| POST | `/api/refresh` | Trigger live scrape + re-embed |

**Merit Calculation request:**
```bash
curl -X POST http://localhost:8000/api/merit \
  -H "Content-Type: application/json" \
  -d '{
    "net_score": 165,
    "fsc_marks": 1000,
    "fsc_total": 1100,
    "matric_marks": 950,
    "matric_total": 1100
  }'
```

**Chat (SSE stream):**
```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the cutoff for BS CS?", "history": [], "session": {}}'
```

---

## Chatbot Behaviour

The chain in `chatbot/chain.py` handles messages in this order:

1. **Casual / greetings** — canned reply, no LLM call
2. **NET prep questions** — structured canned answer with study tips and resource links
3. **Merit calculation in-message** — computed directly in Python (no LLM), returns a formatted table
4. **Everything else** — RAG retrieval → LLM with system prompt containing injected cutoffs

The system prompt always includes the full, authoritative cutoff list loaded from `data/cutoffs.json`, with an explicit override instruction so the LLM ignores any conflicting training data.

---

## Troubleshooting

**Ollama not running:**
```bash
ollama serve
# In another terminal:
ollama pull phi3:mini
```

**ChromaDB slow on first run:**
It downloads the `all-MiniLM-L6-v2` embedding model (~90 MB). Subsequent startups are fast.

**Port already in use:**
```bash
PORT=8080 python main.py
```

**Low RAM:**
```bash
ollama pull llama3.2:1b
MODEL=llama3.2:1b python main.py
```

**Windows Defender blocking model downloads:**
Whitelist the Python and Ollama directories in your antivirus settings.

---

## Competition Details

**Competition:** NUST Local Chatbot Competition 2026
**Category:** AI-Powered Student Services

Key differentiators:
- Fully offline — no API costs, no data leaves your machine
- Hallucination-free cutoff data via permanent `cutoffs.json` + RAG filter
- Merit calculator grounded in the official NUST formula
- Live scraping capability with content-hash change detection
- Structured short-circuit replies for common questions (no unnecessary LLM calls)

---

*Not affiliated with NUST official admissions. Always verify at [nust.edu.pk](https://nust.edu.pk) and [ugadmissions.nust.edu.pk](https://ugadmissions.nust.edu.pk).*
