# Dhruv Tangri — Digital Twin

Live demo: https://digital-twin-alpha-dun.vercel.app/?_vercel_share=5qw896qVaS5tAVT3o2UtyhtbBI7Ew7oo

(NOTE: Due to free version of render the first reply will take time since the render takes time to start after that the latency should be really low.)

A RAG-powered digital twin that answers questions as Dhruv — grounded in my actual career history, engineering opinions, projects, and live GitHub activity.

---

## What it does

Ask the twin anything about Dhruv — his background, technical opinions, projects, work experience, or what he has been building lately. It responds in first person using retrieved context from his personal knowledge base and live GitHub data.

Example questions:
- "Tell me about yourself"
- "What do you think about RAG vs fine-tuning?"
- "Walk me through your work at KPMG"
- "What have you been building lately on GitHub?"
- "What does your ideal next role look like?"

---

## Architecture

```
INGESTION PIPELINE 

  data/profile.json    ─┐
  data/docs/*.md        ├──► Loaders → Chunkers → PII Redactor → OpenAI Embedder → FAISS Index
  GitHub API (live)    ─┘

RETRIEVAL PIPELINE (on every user query)

  User question
      → Embed with OpenAI text-embedding-3-small (1536 dims)
      → Cosine similarity search in FAISS (IndexFlatIP)
      → Top-K chunks retrieved + always-inject profile summary
      → Prompt assembled with context + conversation history
      → gpt-4o-mini streams response token by token via SSE
      → Frontend renders tokens in real time
```

### Backend — 14 modules

```
backend/
  config.py          all constants and paths — single source of truth
  chunkers.py        ParagraphChunker (JSON, READMEs) + SectionChunker (Markdown)
  loaders.py         JSONProfileLoader + MarkdownLoader — both extend BaseLoader
  github_loader.py   GitHub API — repos, commits, READMEs — extends BaseLoader
  pii_redactor.py    Presidio + regex PII detection and redaction before embedding
  embedder.py        OpenAI text-embedding-3-small — extends BaseEmbedder
  store.py           FAISS IndexFlatIP — build, save, load, search
  ingest.py          orchestrator: loaders → PII redact → embed → store
  retrieval.py       query embed → FAISS search → always-inject summary → deduplicate
  prompt.py          chunk formatting + system prompt + persona definition
  schemas.py         Pydantic models: ChatRequest, Message, HealthResponse
  dependencies.py    FastAPI singleton management via Depends()
  routes.py          POST /chat (SSE streaming), GET /health, GET /config
  main.py            FastAPI app, CORS middleware, lifespan context manager
```

### Frontend — React 18 + TypeScript + Vite

```
frontend/
  src/App.tsx        full chat UI — sidebar + messages + SSE streaming input
  src/index.css      dark terminal aesthetic, amber accents
  src/main.tsx       React entry point
  index.html         Google Fonts 
  vite.config.ts     proxies /chat and /health to backend in dev
```

---

## GitHub Integration

`github_loader.py` fetches live data from the GitHub REST API on every ingestion run. Three endpoints per repo:

```
GET /users/{username}/repos               → repo metadata (name, description, language, topics, stars)
GET /repos/{username}/{repo}/commits      → last 20 commit messages with dates
GET /repos/{username}/{repo}/readme       → README content, base64 decoded
```

Each non-fork repo produces up to 3 chunks in the FAISS index. READMEs are cleaned (strip HTML, badges, image tags) then chunked by paragraph using the existing `ParagraphChunker`.

Adding more integrations follows the same pattern — `BaseLoader` abstraction means any new source plugs in with one new file and one new line in `ingest.py`:

```python
class NotionLoader(BaseLoader):
    def load(self, path: Path) -> list[dict]:
        # 1. Authenticate via Notion OAuth
        # 2. Fetch pages via Notion API
        # 3. Chunk with existing ParagraphChunker
        # 4. Return list of chunk dicts

# ingest.py — one new line:
(NotionLoader(), None),
```

### PII Redaction

All chunks pass through `PIIRedactor` before embedding. Nothing sensitive enters the FAISS index.

Two-layer detection:

**Presidio (Microsoft)** — NLP-based detection of:
- Email addresses
- Phone numbers
- Credit card numbers
- US SSNs, passport numbers, driver licenses
- Bank account numbers, IBANs
- IP addresses

**Regex fallback** — catches secrets Presidio misses:
- GitHub personal access tokens (`ghp_...`)
- OpenAI API keys (`sk-...`)
- AWS access keys (`AKIA...`)
- Bearer tokens
- Private keys

Redacted values are replaced with readable tokens: `dhruv@gmail.com` → `[EMAIL REDACTED]`

The ingestion summary shows exactly which chunks were redacted and why — full audit trail in the `pii_redacted` field on each chunk dict.

Intentionally NOT redacted: person names, organizations, locations, dates — these are core identity context for the twin.

---

## Data sources

| Source | Type | Content |
|---|---|---|
| `data/profile.json` | Structured | Resume, skills, education, work history, projects |
| `data/docs/career_story.md` | Narrative | Career arc, F1 origin story, motivations |
| `data/docs/engineering_philosophy.md` | Narrative | Problem-solving approach and tradeoffs |
| `data/docs/opinions_on_tech.md` | Narrative | RAG vs fine-tuning, stack opinions, AI trends |
| `data/docs/projects.md` | Narrative | Deep dives on RegHealth Navigator, Legal Doc Assistant |
| `data/docs/personality_and_working_style.md` | Narrative | Communication style, working preferences |
| `data/docs/faqs.md` | Narrative | Interview-ready answers |
| GitHub API (live) | Live | Repos, commits, READMEs — fetched fresh on every ingest |

---

## Tech stack

| Layer | Technology | Reason |
|---|---|---|
| Embeddings | OpenAI text-embedding-3-small | No local model — zero memory footprint on server |
| Vector store | FAISS IndexFlatIP | Zero infra, exact cosine search, fast at this scale |
| LLM | gpt-4o-mini | Cost efficient, quality sufficient for personal Q&A |
| PII detection | Microsoft Presidio + regex | NLP-based entity detection + pattern matching for secrets |
| Backend | FastAPI + uvicorn | Async, typed, Depends() injection, SSE streaming |
| Frontend | React 18 + TypeScript + Vite | Type-safe, fast dev experience |
| GitHub data | GitHub REST API v3 | Live repo/commit/README ingestion |
| Deployment | Render + Vercel | Zero config, free tier, fast for demos |

---

## Running locally

### Prerequisites

- Python 3.11
- Node.js 18+
- conda or any Python virtual environment

### 1. Clone and set up environment

```bash
git clone https://github.com/TangriDhruv/digital-twin.git
cd digital-twin

conda create -n digital-twin python=3.11 -y
conda activate digital-twin

pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:
```
OPENAI_API_KEY=your-openai-api-key
GITHUB_TOKEN=your-github-token        # optional, raises rate limit to 5000/hr
```

### 3. Update your GitHub username

In `backend/config.py`:
```python
GITHUB_USERNAME = "your-github-username"
```

### 4. Run ingestion

Builds the FAISS index from all data sources. PII is redacted before embedding.

```bash
python backend/ingest.py
```

Expected output:
```
INFO | PII redaction: 2 items redacted across 2/90 chunks
── Ingestion Summary ─────────────────────────────
  career_story                           8 chunks
  engineering_philosophy                 6 chunks
  faqs                                   9 chunks
  github_commits                         8 chunks
  github_readme                         12 chunks
  github_repos                           8 chunks
  opinions_on_tech                       7 chunks
  personality_and_working_style          7 chunks
  projects                               8 chunks
  summary                                1 chunks
  TOTAL                                 ~90 chunks

  ── PII Redacted in 2 chunks ──────────
  [github_readme / digital-twin] → ['EMAIL_ADDRESS']
──────────────────────────────────────────────────
```

### 5. Start the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

### 6. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**

---

## Deployment

### Backend — Render

- **Runtime:** Python 3.11
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
- **Environment variables:** `OPENAI_API_KEY`, `GITHUB_TOKEN`

The FAISS index is committed to the repo — no ingestion needed on Render. The server loads the pre-built index on startup.

Note: Render free tier spins down after 15 minutes of inactivity. First request after sleep takes ~10 seconds. Open the `/health` endpoint before your demo to wake it.

### Frontend — Vercel

- **Root directory:** `frontend`
- **Framework preset:** Vite
- **Build command:** `npm run build`
- **Output directory:** `dist`
- **Environment variable:** `VITE_API_URL=https://your-render-url.onrender.com`

---

## Tradeoffs and next steps

**Tradeoffs made:**

- **OpenAI embeddings over local model** — Offloads embedding computation entirely to the API, eliminating local memory overhead. A model like all-MiniLM-L6-v2 requires ~200MB at runtime; at the scale of this dataset, API-based embeddings are cost-negligible and the simpler operational trade-off.
- **FAISS over managed vector DB** — zero infrastructure for a single-user twin. Pinecone/Milvus add operational overhead not justified at 90 chunks.
- **Dense search only over hybrid** — 90 conversational chunks retrieve accurately with vector search alone. BM25 adds value at scale or for domain-specific terminology (e.g. regulation numbers in RegHealth Navigator).
- **Index committed to repo** —  The FAISS index is pre-built and version-controlled so the application is immediately queryable on startup. This avoids requiring users to run ingestion.py as a prerequisite before the frontend can serve responses, keeping the setup path as frictionless as possible.


**Next steps:**

- OAuth login (GitHub or Google) for access control
- Notion + Google Drive integrations via same BaseLoader pattern
- RAGAS eval harness — 20 question test set with automated scoring
- Re-ranking with cross-encoder for better retrieval precision
- Contextual retrieval (Anthropic) — prepend LLM-generated context to each chunk before embedding
- Incremental indexing — re-embed only changed documents instead of full rebuild
- Hybrid search — BM25 + dense for better recall on specific technical terms

### Authentication Design 

The `FastAPI Depends()` architecture makes auth a clean addition without touching any route logic.

**Simple approach (API Key):**
```python
def verify_api_key(x_api_key: str = Header(default="")):
    expected = os.getenv("TWIN_API_KEY", "")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

# In route:
async def chat(request: ChatRequest, _=Depends(verify_api_key), ...):
```

**Production approach (OAuth 2.0 + JWT):**
1. User clicks "Login with GitHub" on frontend
2. Redirect to GitHub OAuth with `client_id` and `public_repo` scope
3. GitHub returns authorization code to `/callback`
4. Server exchanges code for GitHub access token (server-to-server)
5. Server issues signed JWT with user identity + 24hr expiry
6. Frontend stores JWT in memory (not localStorage — XSS risk)
7. Every request: `Authorization: Bearer <jwt>`
8. FastAPI `Depends()` validates JWT signature + expiry

