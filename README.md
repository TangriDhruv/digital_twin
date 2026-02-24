# Dhruv Tangri — Digital Twin

---

## What it does

Ask the twin anything about Dhruv — his background, technical opinions, projects, work experience, or what he's been building lately. It responds in first person using retrieved context from his personal knowledge base and live GitHub data.

**Example questions:**
- "Tell me about yourself"
- "What do you think about RAG vs fine-tuning?"
- "Walk me through your work at KPMG"
- "What have you been building lately on GitHub?"
- "What does your ideal next role look like?"

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                   │
│                   (run once on startup)                 │
└─────────────────────────────────────────────────────────┘

  data/profile.json          ─┐
  data/docs/*.md              ├──► Loaders → Chunkers → MPNet Embedder
  GitHub API (live)          ─┘              ↓
                                        FAISS Index
                                     (faiss_index/)

┌─────────────────────────────────────────────────────────┐
│                    RETRIEVAL PIPELINE                   │
│                   (on every user query)                 │
└─────────────────────────────────────────────────────────┘

  User question
      → Embed with MPNet (all-mpnet-base-v2)
      → Cosine similarity search in FAISS
      → Top-K chunks retrieved
      → Always inject: profile summary chunk
      → Prompt assembled with context + history
      → gpt-4o-mini streams response token by token
      → Frontend renders via SSE
```

### Backend structure

```
backend/
  config.py          # all constants and paths — single source of truth
  chunkers.py        # text splitting strategies (ParagraphChunker, SectionChunker)
  loaders.py         # file loaders (JSONProfileLoader, MarkdownLoader)
  github_loader.py   # GitHub API loader (repos, commits, READMEs)
  embedder.py        # MPNet embedding wrapper
  store.py           # FAISS index build/save/load/search
  ingest.py          # ingestion orchestrator — wires loaders → embedder → store
  retrieval.py       # query embedding + FAISS search + always-inject logic
  prompt.py          # prompt assembly — context formatting + system prompt
  schemas.py         # Pydantic request/response models
  dependencies.py    # FastAPI singleton management
  routes.py          # HTTP endpoint handlers
  main.py            # app creation, CORS, lifespan
```

**Design principles:** SOLID throughout. Every component depends on abstractions (`BaseLoader`, `BaseEmbedder`, `BaseVectorStore`, `BaseRetriever`) not concrete implementations. Adding a new data source (e.g. Notion) means writing one new loader class — nothing else changes.

### Frontend structure

```
frontend/
  src/
    App.tsx        # full chat UI — sidebar + messages + input
    index.css      # all styles
    main.tsx       # React entry point
  index.html       # Google Fonts
  vite.config.ts   # proxies /chat and /health to backend:8000
```

---

## Phase 2: Data Integrations (Option 1)

The GitHub integration (`github_loader.py`) fetches live data from the GitHub REST API on every ingestion run:

- **Repos** — name, description, language, topics, stars, last updated
- **Commits** — last 20 commit messages per repo with dates
- **READMEs** — full README content, cleaned and chunked by paragraph

Each non-fork repo produces up to 3 chunks in the FAISS index. This means the twin always answers questions about recent activity from actual GitHub data, not static files.

**Adding more integrations** is straightforward — the `BaseLoader` abstraction means any new source (Notion, Slack, Google Drive) follows the same pattern:

```python
class NotionLoader(BaseLoader):
    def load(self, path: Path) -> list[dict]:
        # 1. Authenticate via Notion OAuth
        # 2. Fetch pages via Notion API
        # 3. Chunk content using existing ParagraphChunker
        # 4. Return list of chunk dicts
```

Then add one line to `ingest.py` — nothing else changes.

**Authentication (not implemented — design below):**

The cleanest production approach would be OAuth + JWT:
1. User clicks "Login with GitHub"
2. Redirected to GitHub auth page, approves access
3. GitHub returns authorization code to callback URL
4. Server exchanges code for access token
5. Server issues a signed JWT with expiry
6. Frontend includes JWT in `Authorization: Bearer` header
7. FastAPI `Depends()` validates JWT on every protected route

For a simpler single-owner twin, header-based API key auth (10 lines via FastAPI `Depends()`) is sufficient and can be added without changing any route logic.

---

## Data sources

| Source | Type | Content |
|---|---|---|
| `data/profile.json` | Structured | Resume, skills, education, work history, projects |
| `data/docs/career_story.md` | Narrative | Career arc, origin story, motivations |
| `data/docs/engineering_philosophy.md` | Narrative | How Dhruv approaches problems and tradeoffs |
| `data/docs/opinions_on_tech.md` | Narrative | Opinions on RAG, stacks, AI trends |
| `data/docs/projects.md` | Narrative | Deep dives on key projects |
| `data/docs/personality_and_working_style.md` | Narrative | Communication style, working preferences |
| `data/docs/faqs.md` | Narrative | Interview-ready answers |
| GitHub API (live) | Live | Repos, commits, READMEs |

---

## Tech stack

| Layer | Technology |
|---|---|
| Embeddings | `sentence-transformers/all-mpnet-base-v2` |
| Vector store | FAISS (IndexFlatIP, exact cosine similarity) |
| LLM | OpenAI `gpt-4o-mini` |
| Backend | FastAPI + uvicorn |
| Frontend | React 18 + TypeScript + Vite |
| GitHub data | GitHub REST API v3 |

---

## Running locally

### Prerequisites

- Python 3.11
- Node.js 18+
- conda (recommended) or any Python virtual environment

### 1. Clone and set up environment

```bash
git clone https://github.com/yourusername/digital-twin.git
cd digital-twin

conda create -n digital-twin python=3.11 -y
conda activate digital-twin

pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your values:
```
OPENAI_API_KEY=your-openai-api-key
GITHUB_TOKEN=your-github-token        # optional but recommended
```

### 3. Update your GitHub username

In `backend/config.py`:
```python
GITHUB_USERNAME = "your-github-username"
```

### 4. Run ingestion

Builds the FAISS index from all data sources including live GitHub data:

```bash
python backend/ingest.py
```

Expected output:
```
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
  ...
  TOTAL                                 ~75 chunks
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

- **Backend:** Render (Web Service, Python 3.11)
  - Build: `pip install -r requirements.txt`
  - Start: `python backend/ingest.py && uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
  - Environment variables: `OPENAI_API_KEY`, `GITHUB_TOKEN`

- **Frontend:** Vercel
  - Root directory: `frontend`
  - Framework: Vite
  - Environment variables: `VITE_API_URL=https://your-render-url.onrender.com`

---

## Tradeoffs and next steps

**Tradeoffs made:**
- Used `all-mpnet-base-v2` (local) over OpenAI embeddings — avoids per-token cost, fully reproducible, no API dependency for ingestion
- FAISS over a managed vector DB (Pinecone, Milvus) — zero infrastructure for a single-user twin, exact search at this scale is fast enough
- Static re-ingestion over webhooks — simple and reliable; GitHub webhooks would be the next step for real-time freshness
- No auth implemented — scope decision; the FastAPI `Depends()` architecture makes it a clean addition without touching any route logic

**Next steps:**
- Add OAuth login (GitHub or Google) so access can be controlled per-user
- Add more integrations — Notion for notes, Google Drive for documents
- Add a retrieval eval harness — embed a test set of 20 questions with expected answers, score recall@k and response quality automatically
- Hybrid search — combine FAISS dense retrieval with BM25 keyword search for better precision on specific technical terms
- Scheduled re-ingestion — APScheduler or GitHub webhooks to keep the index fresh without restarting the server
