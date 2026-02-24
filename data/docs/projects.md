# Projects

## RegHealth Navigator (May 2025 - July 2025)

**What it is:**
A full-stack application for healthcare professionals to analyze Medicare regulations using semantic search and AI-powered document comparison. It automatically fetches regulations from the Federal Register, processes them, and lets users query across them conversationally.

**Why I built it:**
Medicare regulations are dense, frequently updated, and hard to navigate. Healthcare professionals were spending hours manually comparing documents to understand what changed and what it meant for their workflows. I wanted to make that instant.

**Tech stack:** React, Flask, FAISS, OpenAI embeddings

**What was genuinely hard:**
The chunking strategy was the hardest problem. In RAG systems people often underestimate how much retrieval quality depends on how you split your documents before embedding them. Regulatory documents are particularly tricky — they have hierarchical structure (sections, subsections, cross-references) where a chunk that looks complete on its own is actually meaningless without its parent context.

Too large a chunk and you embed too much noise, the semantic signal gets diluted and retrieval becomes imprecise. Too small and you lose the context needed to answer the question correctly. Fixed-size chunking completely ignored document structure. I ended up needing to build structure-aware chunking that respected section boundaries and injected parent context into child chunks — so each chunk carried enough context to be retrievable AND interpretable on its own.

**What I'm proud of:**
The incremental processing system. Rather than re-embedding the entire Federal Register on every update, it detects only new or modified regulations and processes those. That's what drove the 80% reduction in processing costs — a small engineering decision with a large economic impact.

**What I'd do differently:**
I'd invest more time upfront designing the chunking strategy before writing any embedding code. I treated it as a detail early on and it became the central problem. I'd also evaluate chunking approaches more rigorously — using retrieval metrics like recall@k to compare strategies rather than relying on qualitative spot-checks. Additionally I'd add a hybrid search layer combining dense vector search with sparse BM25 retrieval, since regulatory text has a lot of specific terminology and identifiers that keyword search handles better than embeddings alone.

---

## Legal Document Assistant (September 2025 - December 2025)

**What it is:**
A full-stack app that automates filling out .docx templates through conversational AI. You upload a document with placeholders, chat with the assistant, and it intelligently detects and fills fields using context from the conversation.

**Why I built it:**
Legal and compliance teams spend significant time on repetitive document preparation. The interesting technical challenge was making the field detection smart enough to handle ambiguous or inconsistently named placeholders — not just a find-and-replace.

**Tech stack:** React 19, TypeScript, FastAPI, OpenAI GPT-4 Mini

**What I'm proud of:**
The smart field matching using LLM context analysis — the system understands that "Party A" and "Client Name" might refer to the same entity based on conversation context, rather than requiring exact matches. Real-time progress tracking made the experience feel responsive even on longer documents.

---

## COVID-19 Outcomes & Vaccine Uptake Prediction (January - March 2025)

**What it is:**
Predictive models built on 4,000+ survey observations to estimate COVID-19 test positivity rates and vaccine uptake across populations.

**Tech stack:** Python, scikit-learn, XGBoost, Random Forest, MLP, Linear Regression

**What I'm proud of:**
Achieving R2 = 0.879 while keeping the methodology interpretable enough to translate into actual public health recommendations — not just a benchmark number. Feature selection and multivariate analysis were done carefully to ensure the model wasn't just fitting noise.

---

## Smart Journal Entry Testing Tool — KPMG (2021-2022)

**What it is:**
An internal ML tool for audit that detected outliers in financial journal entries, helping auditors identify anomalies that warranted investigation.

**Why it mattered:**
Manual audit of journal entries is time-consuming and prone to missing subtle patterns. Automating the outlier detection let auditors focus on the cases that actually mattered.

**Tech stack:** Python, Isolation Forest, internal KPMG data infrastructure

**My role:** Led a team of 3 end-to-end — technical design, model development, and client presentation. Won the "Above and Beyond" award.

**What was hard:**
Financial data is highly sensitive and heavily governed. Every modeling decision had to be explainable and auditable. We couldn't use a black box — the auditors needed to understand and trust why the model flagged something.
